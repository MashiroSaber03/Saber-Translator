//! Indexed Color and Duotone documents in flat exports.

use photocraft_color::{Color, ColorMode, SampleType};
use photocraft_doc::{ColorTable, Document, Duotone, DuotoneInk, Size};

fn doc(mode: ColorMode) -> Document {
    let mut d = Document::with_background("m", Size::new(4, 2), ColorMode::Rgb, SampleType::U8, Color::rgba(1.0, 0.0, 0.0, 1.0));
    d.mode = mode;
    d
}

#[test]
fn indexed_png_is_palette_png() {
    let mut d = doc(ColorMode::Indexed);
    d.color_table = Some(ColorTable { colors: vec![[0, 0, 0], [255, 0, 0]], transparent: None });
    let r = photocraft_io::export(&d, "x.png", &Default::default()).unwrap();
    // IHDR colour type 3 = indexed; a PLTE chunk follows.
    assert_eq!(r.bytes[25], 3);
    assert!(r.bytes.windows(4).any(|w| w == b"PLTE"));
    let back = photocraft_io::import("x.png", &r.bytes).unwrap().document;
    let px = back.layers[0].surface().unwrap().rgba(1, 1);
    assert_eq!(&px[..3], &[1.0, 0.0, 0.0]);
    // Other formats keep the expanded pixels.
    let j = photocraft_io::export(&d, "x.tif", &Default::default()).unwrap();
    assert!(!j.bytes.is_empty());
}

#[test]
fn duotone_exports_the_inks_as_rgb() {
    // Mid gray background.
    let mut d = Document::with_background("m", Size::new(4, 2), ColorMode::Grayscale, SampleType::U8, Color::rgba(0.5, 0.5, 0.5, 1.0));
    d.mode = ColorMode::Duotone;
    d.duotone = Some(Duotone { inks: vec![DuotoneInk::new("Black", [0.0; 3]), DuotoneInk::new("Orange", [1.0, 0.5, 0.0])], psd_raw: None });
    let r = photocraft_io::export(&d, "x.png", &Default::default()).unwrap();
    assert!(r.warnings.iter().any(|w| w.contains("Duotone")), "{:?}", r.warnings);
    let back = photocraft_io::import("x.png", &r.bytes).unwrap().document;
    let px = back.layers[0].surface().unwrap().rgba(0, 0);
    assert!(px[0] > px[2] + 0.05, "warm: {px:?}");
}

#[test]
fn indexed_png_keeps_the_document_resolution() {
    // #3047: the palette PNG lost its pHYs chunk and reopened at 72 ppi.
    let mut d = doc(ColorMode::Indexed);
    d.resolution_dpi = 300.0;
    d.color_table = Some(ColorTable { colors: vec![[0, 0, 0], [255, 0, 0]], transparent: None });
    let r = photocraft_io::export(&d, "x.png", &Default::default()).unwrap();
    assert_eq!(r.bytes[25], 3, "still a palette PNG");
    let back = photocraft_io::import("x.png", &r.bytes).unwrap().document;
    assert!((back.resolution_dpi - 300.0).abs() < 0.01, "{}", back.resolution_dpi);
}
