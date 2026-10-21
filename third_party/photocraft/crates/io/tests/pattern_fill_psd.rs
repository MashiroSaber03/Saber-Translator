//! A Pattern Fill layer looks the same after a PSD or layered-TIFF save and reopen in every
//! colour mode and depth (#1907): the pixels written for it are the pattern, not transparency.

use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Document, Fill, Layer, LayerContent, Pattern};
use photocraft_geom::{Rect, Size};
use photocraft_io::*;
use photocraft_raster::Surface;

/// An RGB checkerboard (as the default patterns are), whatever the document's mode.
fn checker() -> Pattern {
    let mut s = Surface::new(PixelFormat::RGBA8);
    s.fill_rect(Rect::new(0, 0, 8, 8), &[0.8, 0.8, 0.8, 1.0]);
    s.fill_rect(Rect::new(0, 0, 4, 4), &[0.2, 0.4, 0.9, 1.0]);
    s.fill_rect(Rect::new(4, 4, 8, 8), &[0.2, 0.4, 0.9, 1.0]);
    Pattern { id: "059289fb-0000-4000-8000-000000000001".into(), name: "Checker".into(), width: 8, height: 8, surface: s }
}

fn doc(mode: ColorMode, depth: SampleType) -> Document {
    let mut d = Document::new("p", Size::new(40, 24), mode, depth);
    d.patterns.push(checker());
    let fill = Fill::Pattern { name: "Checker".into(), scale: 1.0, id: checker().id, angle: 0.0, link: true, phase: (0.0, 0.0) };
    d.layers.push(Layer::new("Pattern Fill 1", LayerContent::Fill(fill)));
    d
}

/// How far `back`'s composite is from `d`'s pattern stored in `d`'s pixel format (the pattern
/// as Edit › Fill would paint it: an RGB pattern is grey in a Grayscale document).
fn max_diff(d: &Document, back: &Document) -> f32 {
    let fmt = d.pixel_format();
    let want = photocraft_compose::flatten(d).px.iter().map(|p| photocraft_raster::to_rgba(&fmt, &photocraft_raster::from_rgba(&fmt, *p))).collect::<Vec<_>>();
    let got = photocraft_compose::flatten(back);
    want.iter().zip(&got.px).flat_map(|(p, q)| p.iter().zip(q).map(|(u, v)| (u - v).abs())).fold(0.0, f32::max)
}

#[test]
fn pattern_fill_survives_psd_and_layered_tiff_in_every_mode() {
    for mode in [ColorMode::Rgb, ColorMode::Grayscale, ColorMode::Cmyk, ColorMode::Lab] {
        for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let d = doc(mode, depth);
            let before = photocraft_compose::flatten(&d);
            assert!(before.px.iter().all(|p| p[3] > 0.99), "{mode:?}/{depth:?}: the pattern renders before saving");
            for (path, opts) in [("x.psd", ExportOptions::default()), ("x.tif", ExportOptions { tiff_layers: true, ..Default::default() })] {
                if mode == ColorMode::Lab && path == "x.tif" {
                    continue; // Lab documents are saved as a flat TIFF (with a warning).
                }
                let r = export(&d, path, &opts).expect("export");
                let back = import(path, &r.bytes).expect("import").document;
                let names: Vec<&str> = back.layers.iter().map(|l| l.name.as_str()).collect();
                let l = back
                    .layers
                    .iter()
                    .find(|l| l.name == "Pattern Fill 1")
                    .unwrap_or_else(|| panic!("{mode:?}/{depth:?} {path}: layers {names:?}, warnings {:?}", r.warnings));
                assert!(matches!(l.content, LayerContent::Fill(Fill::Pattern { .. })), "{mode:?}/{depth:?} {path}");
                let diff = max_diff(&d, &back);
                assert!(diff < 0.02, "{mode:?}/{depth:?} {path}: the reopened pattern fill differs by {diff}");
            }
        }
    }
}

/// A pattern fill seen through a vector mask (Photoshop's pattern-filled shape) stays a pattern
/// fill layer after a save and reopen: its stored pixels are the unclipped fill, which must not
/// turn it into a shape layer drawing the pattern over the whole canvas (psd-tools
/// adjustment-fillers.psd).
#[test]
fn vector_masked_pattern_fill_survives_psd() {
    for mode in [ColorMode::Rgb, ColorMode::Cmyk] {
        let mut d = doc(mode, SampleType::U8);
        let fmt = d.pixel_format();
        let mut white = Layer::raster("Background", fmt);
        white.surface_mut().expect("raster").fill_rect(Rect::new(0, 0, 40, 24), &photocraft_raster::from_rgba(&fmt, [1.0, 1.0, 1.0, 1.0]));
        d.layers.insert(0, white);
        d.layers[1].vector_mask = Some(photocraft_doc::VectorMask::new(photocraft_vector::shapes::rect(8.0, 4.0, 16.0, 12.0)));
        let r = export(&d, "x.psd", &ExportOptions::default()).expect("export");
        let back = import("x.psd", &r.bytes).expect("import").document;
        let l = back.layers.iter().find(|l| l.name == "Pattern Fill 1").expect("layer");
        assert!(matches!(l.content, LayerContent::Fill(Fill::Pattern { .. })), "{mode:?}: reopened as a pattern fill");
        assert!(l.vector_mask.is_some(), "{mode:?}: keeps its vector mask");
        let diff = max_diff(&d, &back);
        assert!(diff < 0.02, "{mode:?}: the reopened layer differs by {diff}");
    }
}
