//! EXIF orientation through the document layer (#285): an iPhone-style JPEG
//! (stored landscape, Orientation = 6) opens as an upright portrait document,
//! and every export writes Orientation = 1 so it is never rotated twice.

use photocraft_codecs::{ChannelLayout, EncodeOptions, Format, Image, SampleType, decode, exif_orientation};
use photocraft_io::{ExportOptions, export, import};
use photocraft_raw::testgen::{TiffBuilder, Val};

/// EXIF block with only an Orientation tag.
fn exif(o: u16) -> Vec<u8> {
    let mut v = b"II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0".to_vec();
    v.extend_from_slice(&o.to_le_bytes());
    v.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
    v
}

fn jpeg_with_exif(jpeg: &[u8], exif: &[u8]) -> Vec<u8> {
    let mut seg = b"Exif\0\0".to_vec();
    seg.extend_from_slice(exif);
    let mut out = jpeg[..2].to_vec();
    out.extend_from_slice(&[0xFF, 0xE1]);
    out.extend_from_slice(&((seg.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(&seg);
    out.extend_from_slice(&jpeg[2..]);
    out
}

/// Upright 16×32 portrait: red top-left quarter, green top-right, blue bottom half.
fn upright_color(x: usize, y: usize) -> [u8; 3] {
    match (x < 8, y < 16) {
        (true, true) => [230, 20, 20],
        (false, true) => [20, 230, 20],
        _ => [20, 20, 230],
    }
}

/// The same scene as a camera stores it for Orientation = 6 (32×16 landscape).
fn stored_orientation_6() -> Image {
    let (w, h) = (16, 32);
    let mut px = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            // Upright (x, y) sits at stored (y, w - 1 - x) in a 32-wide image.
            let (sx, sy) = (y, w - 1 - x);
            px[(sy * h + sx) * 3..][..3].copy_from_slice(&upright_color(x, y));
        }
    }
    Image::from_u8(h as u32, w as u32, ChannelLayout::Rgb, px).unwrap()
}

fn iphone_jpeg() -> Vec<u8> {
    let jpeg = photocraft_codecs::encode(
        &stored_orientation_6(),
        Format::Jpeg,
        &EncodeOptions { jpeg_quality: 100, jpeg_chroma_subsampling: false, ..Default::default() },
    )
    .unwrap();
    jpeg_with_exif(&jpeg, &exif(6))
}

fn assert_upright(img: &Image, what: &str) {
    assert_eq!(img.dimensions(), (16, 32), "{what}");
    let img = img.convert(ChannelLayout::Rgb, SampleType::U8);
    for (x, y) in [(3, 4), (12, 4), (3, 28), (12, 28)] {
        let i = (y * 16 + x) * 3;
        let got = &img.data()[i..i + 3];
        let want = upright_color(x, y);
        assert!(got.iter().zip(want).all(|(a, b)| (i32::from(*a) - i32::from(b)).abs() <= 12), "{what} ({x},{y}): {got:?} vs {want:?}");
    }
}

#[test]
fn rotated_jpeg_opens_upright_and_exports_with_orientation_1() {
    let r = import("IMG_0001.JPG", &iphone_jpeg()).unwrap();
    let d = &r.document;
    assert_eq!((d.size.width, d.size.height), (16, 32), "portrait, like Photoshop");
    assert_eq!(exif_orientation(d.metadata.exif.as_deref().unwrap()), 1, "the document's EXIF says upright");

    for ext in ["jpg", "png", "tif", "webp"] {
        let out = export(d, ext, &ExportOptions::default()).unwrap().bytes;
        assert_eq!(exif_orientation(&out), 1, "{ext}: a TIFF file's own tag");
        let back = decode(&out).unwrap();
        if let Some(e) = &back.meta.exif {
            assert_eq!(exif_orientation(e), 1, "{ext}");
        }
        assert_upright(&back, ext);
        // Open → save → reopen: the same upright document.
        let again = import(&format!("again.{ext}"), &out).unwrap().document;
        assert_eq!((again.size.width, again.size.height), (16, 32), "{ext}");
    }
}

#[test]
fn stale_orientation_in_document_metadata_is_never_exported() {
    // A document whose EXIF still says 6 (a PSD from elsewhere, or rotated by hand as in #285).
    let mut d = import("IMG_0001.JPG", &iphone_jpeg()).unwrap().document;
    d.metadata.exif = Some(std::sync::Arc::new(exif(6)));
    d.metadata.xmp = Some(r#"<rdf:Description tiff:Orientation="6"/>"#.into());
    let none = ExportOptions { xmp: photocraft_io::XmpEmbed::None, ..ExportOptions::default() };
    let jpg = export(&d, "jpg", &none).unwrap().bytes;
    let back = decode(&jpg).unwrap();
    assert_eq!(exif_orientation(back.meta.exif.as_deref().unwrap()), 1);
    // Export As's Metadata: None carries no XMP at all (#647), so nothing stale can hide in it;
    // when the packet is embedded (the default), the orientation inside is rewritten upright.
    assert!(back.meta.xmp.is_none());
    let jpg = export(&d, "jpg", &ExportOptions::default()).unwrap().bytes;
    let back = decode(&jpg).unwrap();
    assert!(back.meta.xmp.as_deref().unwrap().contains(r#"tiff:Orientation="1""#));
    assert_upright(&back, "jpg");

    let psd = export(&d, "psd", &ExportOptions::default()).unwrap().bytes;
    let file = photocraft_psd::PsdFile::from_bytes(&psd).unwrap();
    let e = file.resources.iter().find(|r| r.id == 1058).expect("EXIF resource");
    assert_eq!(exif_orientation(&e.data), 1);
    let x = file.resources.iter().find(|r| r.id == 1060).expect("XMP resource");
    assert!(String::from_utf8_lossy(&x.data).contains(r#"tiff:Orientation="1""#));
}

/// A NEF-like raw that `photocraft-raw` can't develop yet, with IFD0 Orientation and a
/// baseline JPEG preview stored as the sensor reads out.
fn nef(ifd0_orientation: u16, preview: Vec<u8>) -> Vec<u8> {
    let mut t = TiffBuilder::default();
    let strip = t.blob(vec![0; 64]);
    let len = preview.len() as u32;
    let preview = t.blob(preview);
    let raw = t.ifd(vec![
        (256, Val::Long(vec![8])),
        (257, Val::Long(vec![8])),
        (258, Val::Short(vec![12])),
        (259, Val::Short(vec![34713])),
        (262, Val::Short(vec![32803])),
        (273, Val::Blobs(vec![strip])),
        (279, Val::Long(vec![64])),
        (33421, Val::Short(vec![2, 2])),
        (33422, Val::Byte(vec![0, 1, 1, 2])),
    ]);
    let ifd0 = t.ifd(vec![
        (271, Val::Ascii("NIKON CORPORATION".into())),
        (274, Val::Short(vec![ifd0_orientation])),
        (330, Val::Ifds(vec![raw])),
        (513, Val::Blobs(vec![preview])),
        (514, Val::Long(vec![len])),
    ]);
    t.chain = vec![ifd0];
    t.build()
}

#[test]
fn raw_preview_is_turned_upright_exactly_once() {
    let plain = photocraft_codecs::encode(
        &stored_orientation_6(),
        Format::Jpeg,
        &EncodeOptions { jpeg_quality: 100, jpeg_chroma_subsampling: false, ..Default::default() },
    )
    .unwrap();
    // The raw's IFD0 orientation applies to a preview without its own.
    let r = import("DSC_0001.NEF", &nef(6, plain.clone())).unwrap();
    let png = export(&r.document, "png", &ExportOptions::default()).unwrap().bytes;
    assert_upright(&decode(&png).unwrap(), "IFD0 orientation");
    // A preview with its own EXIF orientation is not turned a second time by IFD0's.
    let r = import("DSC_0002.NEF", &nef(6, jpeg_with_exif(&plain, &exif(6)))).unwrap();
    let png = export(&r.document, "png", &ExportOptions::default()).unwrap().bytes;
    assert_upright(&decode(&png).unwrap(), "preview orientation");
    // Upright raws stay as they are.
    let r = import("DSC_0003.NEF", &nef(1, plain)).unwrap();
    assert_eq!((r.document.size.width, r.document.size.height), (32, 16));
}

#[test]
fn malformed_exif_opens_without_panicking() {
    let jpeg = photocraft_codecs::encode(&stored_orientation_6(), Format::Jpeg, &EncodeOptions::default()).unwrap();
    let mut huge = exif(6);
    huge[8..10].copy_from_slice(&u16::MAX.to_le_bytes());
    huge.truncate(14);
    for e in [Vec::new(), b"MM\0*".to_vec(), exif(0), exif(9), exif(u16::MAX), huge] {
        let r = import("x.jpg", &jpeg_with_exif(&jpeg, &e)).unwrap();
        assert_eq!((r.document.size.width, r.document.size.height), (32, 16), "{e:?} reads as Orientation 1");
    }
}
