//! Exports that render the composite in bands (issue #49): results must equal the one-shot
//! composite, across band boundaries, depths and alpha.

use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Document, Layer, LayerContent};
use photocraft_geom::{Rect, Size};
use photocraft_io::*;
use photocraft_raster::Surface;

/// Two layers over more than one export band (a band is ~8 MP), with a gradient, so every row
/// differs; `alpha` < 1 leaves the top-right translucent.
fn layered(w: u32, h: u32, depth: SampleType, alpha: f32) -> Document {
    let mut d = Document::new("b", Size::new(w, h), ColorMode::Rgb, depth);
    let fmt = d.pixel_format();
    let mut bg = Surface::new(fmt);
    let mut row = Vec::with_capacity(w as usize * 4);
    for y in 0..h as i32 {
        row.clear();
        for x in 0..w as i32 {
            let a = if x > w as i32 / 2 && y < h as i32 / 2 { alpha } else { 1.0 };
            row.extend([x as f32 / w as f32, y as f32 / h as f32, 0.5, a]);
        }
        bg.write_region(Rect::new(0, y, w as i32, y + 1), &row);
    }
    d.layers.push(Layer::new("bg", LayerContent::Raster(bg)));
    let mut top = Layer::raster("top", fmt);
    top.surface_mut().unwrap().fill_rect(Rect::new(10, 10, w as i32 - 10, 40), &[1.0, 0.0, 0.0, 1.0]);
    top.opacity = 0.5;
    d.layers.push(top);
    d
}

fn composite_q(doc: &Document, scale: f32) -> Vec<f32> {
    photocraft_compose::flatten(doc).px.iter().flat_map(|p| [p[0], p[1], p[2], p[3]]).map(|v| (v.clamp(0.0, 1.0) * scale).round()).collect()
}

#[test]
fn layered_png_export_spans_bands() {
    // 3000 × 3000 = 9 MP: two bands.
    for (depth, scale) in [(SampleType::U8, 255.0), (SampleType::U16, 65535.0)] {
        for alpha in [1.0, 0.5] {
            let d = layered(3000, 3000, depth, alpha);
            let r = export(&d, "x.png", &ExportOptions::default()).unwrap();
            let img = photocraft_codecs::decode(&r.bytes).unwrap();
            assert_eq!(img.layout().has_alpha(), alpha < 1.0, "{depth:?} alpha {alpha}");
            let got: Vec<f32> = img.to_rgba_f32().iter().map(|v| (v * scale).round()).collect();
            assert!(got == composite_q(&d, scale), "{depth:?} alpha {alpha}: pixels differ");
        }
    }
}

#[test]
fn single_layer_native_export_strips_opaque_alpha() {
    for alpha in [1.0, 0.25] {
        let mut d = layered(700, 300, SampleType::U16, alpha);
        d.layers.truncate(1);
        let r = export(&d, "x.tiff", &ExportOptions::default()).unwrap();
        let back = import("x.tiff", &r.bytes).unwrap().document;
        let (a, b) = (d.layers[0].surface().unwrap(), back.layers[0].surface().unwrap());
        assert_eq!(a.read_region(d.bounds()), b.read_region(d.bounds()), "alpha {alpha}");
        let img = photocraft_codecs::decode(&r.bytes).unwrap();
        assert_eq!(img.layout().has_alpha(), alpha < 1.0);
    }

    let mut d = layered(8, 4, SampleType::U16, 1.0);
    d.layers.truncate(1);
    // A fractional edge exercises coverage rather than merely cropping raw samples.
    d.layers[0].vector_mask = Some(photocraft_doc::VectorMask::new(photocraft_vector::shapes::rect(0.0, 0.0, 4.5, 4.0)));
    let r = export(&d, "x.png", &ExportOptions::default()).unwrap();
    let img = photocraft_codecs::decode(&r.bytes).unwrap();
    assert_eq!(img.sample_type(), photocraft_codecs::SampleType::U16);
    assert!(img.layout().has_alpha());
    let got: Vec<f32> = img.to_rgba_f32().iter().map(|v| (v * 65535.0).round()).collect();
    assert_eq!(got, composite_q(&d, 65535.0), "active vector mask: pixels differ");
}

#[test]
fn psd_merged_image_matches_composite_and_drops_near_opaque_alpha() {
    // Alpha 0.999 rounds to 255: no alpha channel, so the merged image must not be matted.
    for alpha in [1.0, 0.999, 0.5] {
        let d = layered(400, 300, SampleType::U8, alpha);
        let file = document_to_psd(&d);
        assert_eq!(file.merged_has_alpha(), alpha < 0.99, "alpha {alpha}");
        let merged = merged_composite(&file).unwrap();
        let want = photocraft_compose::flatten(&d).px;
        let m = merged.iter().zip(&want).flat_map(|(a, b)| (0..4).map(move |i| (a[i] - b[i]).abs())).fold(0.0f32, f32::max);
        // Un-matting divides the 8-bit rounding error by alpha.
        assert!(m <= 1.0 / (255.0 * alpha) + 1e-4, "alpha {alpha}: max diff {m}");
    }
}

#[test]
fn flat_import_converts_rgb_in_bands() {
    // An opaque RGB file becomes an RGBA document (converted in ~32 MB bands: 36 MB here).
    let (w, h) = (4000u32, 3000u32);
    let px: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();
    let img = photocraft_codecs::Image::from_u8(w, h, photocraft_codecs::ChannelLayout::Rgb, px.clone()).unwrap();
    let bytes = photocraft_codecs::encode(&img, photocraft_codecs::Format::Png, &Default::default()).unwrap();
    let doc = import("x.png", &bytes).unwrap().document;
    let s = doc.layers[0].surface().unwrap();
    assert_eq!(s.format(), PixelFormat::RGBA8);
    let back = s.to_interleaved(doc.bounds());
    let rgb: Vec<u8> = back.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
    assert_eq!(rgb, px);
    assert!(back.as_chunks::<4>().0.iter().all(|p| p[3] == 255));
}
