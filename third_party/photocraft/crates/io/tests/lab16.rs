//! 16-bit Lab PSDs: Photoshop stores a*/b* as `32768 + 256·a` (0..65280 spans −128..127), not
//! on the full 0..65535 range. Layer pixels, the merged image and patterns use that scale, and
//! the documents' a*/b* stay on the 8-bit scale `(a + 128) / 255` at every depth.

use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Document, Layer, LayerContent};
use photocraft_geom::{Rect, Size};
use photocraft_io::*;
use photocraft_psd::PsdFile;
use photocraft_raster::Surface;

/// a* = 52.29, b* = −85.08 (a gradient stop of psd-tools 4x4_16bit_lab) on the document scale.
const A: f32 = (52.29 + 128.0) / 255.0;
const B: f32 = (-85.08 + 128.0) / 255.0;

fn lab_doc(depth: SampleType) -> Document {
    let mut d = Document::new("lab", Size::new(3, 2), ColorMode::Lab, depth);
    let fmt = PixelFormat::new(ColorMode::Lab, depth, true);
    let mut s = Surface::new(fmt);
    let px: Vec<f32> = (0..6).flat_map(|_| [0.1907, A, B, 1.0]).collect();
    s.write_region(Rect::new(0, 0, 3, 2), &px);
    d.layers.push(Layer::new("px", LayerContent::Raster(s)));
    d
}

fn be16(b: &[u8], i: usize) -> u16 {
    u16::from_be_bytes([b[2 * i], b[2 * i + 1]])
}

#[test]
fn lab16_chroma_uses_photoshops_scale() {
    let doc = lab_doc(SampleType::U16);
    let out = export(&doc, "x.psd", &ExportOptions::default()).unwrap();
    let file = PsdFile::from_bytes(&out.bytes).unwrap();
    assert_eq!(file.header.depth, 16);
    // Merged image: planes L, a, b.
    let m = file.decode_merged().unwrap();
    let n = 6;
    let a = be16(&m, n);
    let b = be16(&m, 2 * n);
    assert!((i32::from(a) - (32768 + (256.0f32 * 52.29).round() as i32)).abs() <= 1, "a = {a}");
    assert!((i32::from(b) - (32768 - (256.0f32 * 85.08).round() as i32)).abs() <= 1, "b = {b}");
    // Layer channel 1 (a*) the same.
    let rec = &file.layers()[0];
    let plane = rec.decode_channel(1, 16, file.header.version).unwrap();
    assert!((i32::from(be16(&plane, 0)) - i32::from(a)).abs() <= 1);
    // And back: the document values survive.
    let back = import("x.psd", &out.bytes).unwrap().document;
    let s = back.layers[0].surface().unwrap();
    let p = s.pixel(1, 1);
    assert!((p[1] - A).abs() < 2e-5 && (p[2] - B).abs() < 2e-5, "{p:?}");
}

#[test]
fn lab_round_trips_at_every_depth() {
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let doc = lab_doc(depth);
        let out = export(&doc, "x.psd", &ExportOptions::default()).unwrap();
        let back = import("x.psd", &out.bytes).unwrap().document;
        let p = back.layers[0].surface().unwrap().pixel(2, 0);
        let tol = if depth == SampleType::U8 { 0.5 / 255.0 + 1e-6 } else { 2e-5 };
        assert!((p[1] - A).abs() <= tol && (p[2] - B).abs() <= tol, "{depth:?}: {p:?}");
        let a = photocraft_compose::flatten(&doc).px;
        let b = photocraft_compose::flatten(&back).px;
        for (x, y) in a.iter().zip(&b) {
            for c in 0..4 {
                assert!((x[c] - y[c]).abs() <= 1.0 / 255.0, "{depth:?}: {x:?} vs {y:?}");
            }
        }
    }
}
