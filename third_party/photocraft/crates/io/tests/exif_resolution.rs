//! #1691: the EXIF resolution of an imported photo becomes the document's, and a PSD save keeps
//! its EXIF copy in step with the ResolutionInfo resource.

use photocraft_codecs::{ChannelLayout, EncodeOptions, Format, Image, Metadata, SampleType, exif_resolution};
use photocraft_io::*;

/// Big-endian EXIF: XResolution 300/1, YResolution 300/1, ResolutionUnit 2.
fn camera_exif() -> Vec<u8> {
    let mut v = b"MM\0*".to_vec();
    v.extend_from_slice(&8u32.to_be_bytes());
    v.extend_from_slice(&3u16.to_be_bytes());
    let data = 8 + 2 + 3 * 12 + 4;
    for (tag, ty, value) in [(282u16, 5u16, data as u32), (283, 5, data as u32 + 8)] {
        v.extend_from_slice(&tag.to_be_bytes());
        v.extend_from_slice(&ty.to_be_bytes());
        v.extend_from_slice(&1u32.to_be_bytes());
        v.extend_from_slice(&value.to_be_bytes());
    }
    v.extend_from_slice(&296u16.to_be_bytes());
    v.extend_from_slice(&3u16.to_be_bytes());
    v.extend_from_slice(&1u32.to_be_bytes());
    v.extend_from_slice(&[0, 2, 0, 0]);
    v.extend_from_slice(&0u32.to_be_bytes());
    for _ in 0..2 {
        v.extend_from_slice(&300u32.to_be_bytes());
        v.extend_from_slice(&1u32.to_be_bytes());
    }
    v
}

#[test]
fn webp_with_exif_resolution_imports_at_it_and_psd_save_follows_the_document() {
    let meta = Metadata { exif: Some(camera_exif()), ..Default::default() };
    let img = Image::from_raw(6, 4, ChannelLayout::Rgb, SampleType::U8, vec![50; 72]).unwrap().with_meta(meta);
    let webp = photocraft_codecs::encode(&img, Format::WebP, &EncodeOptions::default()).unwrap();
    let mut doc = import("photo.webp", &webp).unwrap().document;
    assert_eq!(doc.resolution_dpi, 300.0);

    doc.resolution_dpi = 240.0;
    let psd = export(&doc, "photo.psd", &ExportOptions::default()).unwrap().bytes;
    let back = import("photo.psd", &psd).unwrap().document;
    assert_eq!(back.resolution_dpi, 240.0);
    assert_eq!(back.metadata.exif.as_deref().and_then(|e| exif_resolution(e)), Some((240.0, 240.0)));
}
