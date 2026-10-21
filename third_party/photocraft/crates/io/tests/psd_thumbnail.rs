//! Preferences › File Handling › Image Previews: PSD/PSB saves embed Photoshop's thumbnail
//! resource (1036: 28-byte header + JFIF, ≤ 160 px) unless the previews are turned off.

use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::{Document, Layer};
use photocraft_geom::Size;
use photocraft_io::{ExportOptions, export, import};
use photocraft_psd::PsdFile;
use photocraft_psd::resources::ids;

/// A `w`×`h` document of `mode`/`depth` holding one layer of `rgba` (given as straight sRGB-ish
/// RGBA and converted to the document's model), covering the left `cover` fraction.
fn doc(mode: ColorMode, depth: SampleType, w: u32, h: u32, rgba: [f32; 4], cover: f32) -> Document {
    let mut d = Document::new("thumb", Size::new(w, h), mode, depth);
    let fmt = d.pixel_format();
    let mut px = vec![0.0; fmt.channels()];
    photocraft_raster::from_rgba_into(&fmt, rgba, &mut px);
    let mut layer = Layer::raster("pixels", fmt);
    let right = ((w as f32 * cover).round() as i32).max(1);
    layer.surface_mut().unwrap().fill_rect(photocraft_geom::Rect::new(0, 0, right, h as i32), &px);
    d.layers.push(layer);
    d
}

/// The 1036 resource of a saved file, its header checked; returns (width, height, RGB pixels).
fn thumbnail(bytes: &[u8]) -> Option<(u32, u32, Vec<[f32; 3]>)> {
    let file = PsdFile::from_bytes(bytes).expect("parse");
    assert!(!file.resources.iter().any(|r| r.id == ids::THUMBNAIL_PS4), "only the RGB (1036) thumbnail is written");
    let mut found = file.resources.iter().filter(|r| r.id == ids::THUMBNAIL);
    let r = found.next()?;
    assert!(found.next().is_none(), "one thumbnail resource");
    let d = &r.data;
    let u = |i: usize| u32::from_be_bytes(d[i..i + 4].try_into().unwrap());
    let (format, w, h, width_bytes, total, compressed) = (u(0), u(4), u(8), u(12), u(16), u(20));
    assert_eq!(format, 1, "kJpegRGB");
    assert!(w >= 1 && h >= 1 && w.max(h) <= 160, "{w}x{h}");
    assert_eq!(width_bytes, (w * 24).div_ceil(32) * 4);
    assert_eq!(total, width_bytes * h);
    assert_eq!(compressed as usize, d.len() - 28);
    assert_eq!(&d[24..28], &[0, 24, 0, 1], "24 bits per pixel, 1 plane");
    let jfif = &d[28..];
    assert!(jfif.starts_with(&[0xFF, 0xD8, 0xFF, 0xE0]) && &jfif[6..11] == b"JFIF\0", "a JFIF stream");
    let img = photocraft_codecs::decode(jfif).expect("decodes");
    assert_eq!(img.dimensions(), (w, h));
    let n = img.layout().channels();
    assert_eq!(n, 3, "RGB JPEG");
    let px = img.to_normalized().as_chunks::<3>().0.to_vec();
    Some((w, h, px))
}

fn near(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
    a.iter().zip(&b).all(|(x, y)| (x - y).abs() <= tol)
}

#[test]
fn always_embeds_a_160px_jfif_thumbnail() {
    let d = doc(ColorMode::Rgb, SampleType::U8, 640, 320, [1.0, 0.0, 0.0, 1.0], 1.0);
    let out = export(&d, "x.psd", &ExportOptions::default()).unwrap();
    let (w, h, px) = thumbnail(&out.bytes).expect("Always Save writes the thumbnail");
    assert_eq!((w, h), (160, 80));
    let mid = px[(40 * 160 + 80) as usize];
    assert!(near(mid, [1.0, 0.0, 0.0], 0.08), "{mid:?}");
    assert!(out.warnings.iter().all(|w| !w.contains("preview")), "{:?}", out.warnings);
}

#[test]
fn small_documents_keep_their_size() {
    let d = doc(ColorMode::Rgb, SampleType::U8, 37, 5, [0.0, 0.0, 1.0, 1.0], 1.0);
    let (w, h, _) = thumbnail(&export(&d, "x.psd", &ExportOptions::default()).unwrap().bytes).unwrap();
    assert_eq!((w, h), (37, 5));
}

#[test]
fn never_writes_no_thumbnail() {
    let d = doc(ColorMode::Rgb, SampleType::U8, 64, 64, [0.0, 1.0, 0.0, 1.0], 1.0);
    for name in ["x.psd", "x.psb"] {
        let out = export(&d, name, &ExportOptions { image_previews: false, ..Default::default() }).unwrap();
        assert!(thumbnail(&out.bytes).is_none(), "{name}");
    }
}

#[test]
fn psb_gets_the_thumbnail_too() {
    let d = doc(ColorMode::Rgb, SampleType::U8, 300, 600, [0.0, 1.0, 0.0, 1.0], 1.0);
    let out = export(&d, "x.psb", &ExportOptions::default()).unwrap();
    assert_eq!(out.bytes[5], 2);
    let (w, h, _) = thumbnail(&out.bytes).unwrap();
    assert_eq!((w, h), (80, 160));
}

#[test]
fn transparency_is_shown_over_white() {
    let d = doc(ColorMode::Rgb, SampleType::U8, 160, 160, [0.0, 0.0, 0.0, 1.0], 0.5);
    let (w, _, px) = thumbnail(&export(&d, "x.psd", &ExportOptions::default()).unwrap().bytes).unwrap();
    let row = 80 * w as usize;
    assert!(near(px[row + 20], [0.0; 3], 0.08), "{:?}", px[row + 20]);
    assert!(near(px[row + 140], [1.0; 3], 0.08), "{:?}", px[row + 140]);
}

#[test]
fn every_depth_and_mode_gets_an_rgb_thumbnail() {
    for mode in [ColorMode::Rgb, ColorMode::Grayscale, ColorMode::Cmyk, ColorMode::Lab] {
        for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
            if depth == SampleType::F32 && matches!(mode, ColorMode::Cmyk | ColorMode::Lab) {
                continue; // Photoshop has no 32-bit CMYK or Lab.
            }
            let d = doc(mode, depth, 200, 100, [1.0, 1.0, 1.0, 1.0], 1.0);
            let out = export(&d, "x.psd", &ExportOptions::default()).unwrap();
            let (w, h, px) = thumbnail(&out.bytes).unwrap_or_else(|| panic!("{mode:?} {depth:?}: no thumbnail"));
            assert_eq!((w, h), (160, 80), "{mode:?} {depth:?}");
            let mid = px[(40 * 160 + 80) as usize];
            // White paper is white in every model (CMYK through its profile, with BPC).
            assert!(near(mid, [1.0; 3], 0.1), "{mode:?} {depth:?}: {mid:?}");
        }
    }
}

#[test]
fn cmyk_and_16_bit_colours_survive_the_conversion() {
    for (mode, depth) in
        [(ColorMode::Rgb, SampleType::U16), (ColorMode::Rgb, SampleType::F32), (ColorMode::Cmyk, SampleType::U8), (ColorMode::Lab, SampleType::U16)]
    {
        let d = doc(mode, depth, 100, 100, [1.0, 0.0, 0.0, 1.0], 1.0);
        let (_, _, px) = thumbnail(&export(&d, "x.psd", &ExportOptions::default()).unwrap().bytes).unwrap();
        let [r, g, b] = px[50 * 100 + 50];
        assert!(r > 0.7 && g < 0.35 && b < 0.35, "{mode:?} {depth:?}: red reads as {:?}", [r, g, b]);
    }
}

#[test]
fn saved_files_round_trip_and_resave_a_fresh_thumbnail() {
    let d = doc(ColorMode::Rgb, SampleType::U8, 120, 60, [1.0, 0.0, 0.0, 1.0], 1.0);
    let first = export(&d, "x.psd", &ExportOptions::default()).unwrap();
    let back = import("x.psd", &first.bytes).unwrap().document;
    assert_eq!((back.size.width, back.size.height), (120, 60));
    // The thumbnail is regenerated, never kept as an opaque resource that would go stale.
    assert!(back.metadata.psd_resources.iter().all(|(id, _, _)| *id != ids::THUMBNAIL));
    let again = export(&back, "x.psd", &ExportOptions::default()).unwrap();
    thumbnail(&again.bytes).expect("one fresh thumbnail");
    let off = export(&back, "x.psd", &ExportOptions { image_previews: false, ..Default::default() }).unwrap();
    assert!(thumbnail(&off.bytes).is_none());
}

#[test]
fn the_document_profile_is_honoured() {
    // Mid gray in a linear-sRGB document is sRGB ≈ 0.735, not 0.5: the thumbnail goes through
    // the embedded profile rather than assuming sRGB.
    let mut d = doc(ColorMode::Rgb, SampleType::U16, 50, 50, [0.5, 0.5, 0.5, 1.0], 1.0);
    d.icc_profile = Some(std::sync::Arc::new(photocraft_cms::Builtin::LinearSrgb.profile().to_bytes().to_vec()));
    let (_, _, px) = thumbnail(&export(&d, "x.psd", &ExportOptions::default()).unwrap().bytes).unwrap();
    let mid = px[25 * 50 + 25];
    assert!(near(mid, [0.735; 3], 0.03), "{mid:?}");
}
