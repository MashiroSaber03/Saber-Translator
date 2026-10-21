//! Every adjustment kind and every blend mode through document → PSD → document.
//!
//! Asserts doc-model equality where the PSD encoding is lossless, and that the re-imported
//! document flattens to the same composite (within 1/255) in every case, so nothing a user made
//! in PhotoCraft silently disappears or changes when Photoshop opens the file.

mod common;

use std::sync::Arc;

use common::*;
use photocraft_color::{BlendMode, ColorMode, SampleType};
use photocraft_doc::adjust::{CurvePoint, LevelsChannel};
use photocraft_doc::adjust::{HueRange, ToneSpace};
use photocraft_doc::*;
use photocraft_geom::Rect;
use photocraft_io::*;
use photocraft_psd::PsdFile;

const TOL: f32 = 1.0 / 255.0 + 1e-5;

fn export_import(doc: &Document) -> (Document, Vec<String>, PsdFile) {
    let r = export(doc, "x.psd", &ExportOptions::default()).expect("export");
    let file = PsdFile::from_bytes(&r.bytes).expect("parse");
    let back = import("x.psd", &r.bytes).expect("import").document;
    (back, r.warnings, file)
}

fn assert_composite_eq(a: &Document, b: &Document, ctx: &str) {
    let (x, y) = (photocraft_compose::flatten(a), photocraft_compose::flatten(b));
    assert_eq!(x.rect, y.rect, "{ctx}: composite rect");
    let m = max_diff(&x.px, &y.px);
    assert!(m <= TOL, "{ctx}: composite differs by {m} (> 1/255)");
}

fn q16(v: u16) -> f32 {
    f32::from(v) / 65535.0
}

/// Every [`Adjustment`] kind with representative settings, and whether its PSD encoding is
/// lossless (the document model comes back equal) or only renders the same.
fn adjustments() -> Vec<(Adjustment, bool)> {
    let pts = |v: &[(u8, u8)]| v.iter().map(|&(i, o)| CurvePoint { input: g(i), output: g(o) }).collect::<Vec<_>>();
    let lc = |a: u8, b: u8, gamma: f32| LevelsChannel { in_black: g(a), in_white: g(b), gamma, out_black: g(5), out_white: g(250) };
    let lut = photocraft_cms::lutfile::LutFile::from_fn("t", 5, |c| [c[1], c[2] * 0.5, 1.0 - c[0]]);
    let all = vec![
        (Adjustment::BrightnessContrast { brightness: 30.0, contrast: -20.0, legacy: false }, true),
        (Adjustment::BrightnessContrast { brightness: -40.0, contrast: 50.0, legacy: true }, true),
        (
            Adjustment::Levels {
                master: lc(10, 240, 1.2),
                per_channel: [lc(0, 255, 1.0), lc(5, 250, 0.8), lc(20, 200, 1.5)],
                space: ToneSpace::Rgb,
                black: Default::default(),
            },
            true,
        ),
        (
            Adjustment::Curves {
                master: pts(&[(0, 0), (128, 150), (255, 255)]),
                per_channel: [pts(&[(0, 10), (255, 255)]), pts(&[(0, 0), (255, 245)]), pts(&[(0, 0), (64, 32), (255, 255)])],
                space: ToneSpace::Rgb,
                black: Vec::new(),
                pencil: false,
            },
            true,
        ),
        (Adjustment::Exposure { exposure: 0.75, offset: -0.0125, gamma: 0.9 }, true),
        (Adjustment::Vibrance { vibrance: 60.0, saturation: -25.0 }, true),
        (Adjustment::HueSaturation { hue: 40.0, saturation: -30.0, lightness: 10.0, colorize: false, ranges: HueRange::defaults() }, true),
        (Adjustment::HueSaturation { hue: 200.0, saturation: 50.0, lightness: -10.0, colorize: true, ranges: HueRange::defaults() }, true),
        (
            Adjustment::ColorBalance { shadows: [20.0, -10.0, 5.0], midtones: [-15.0, 10.0, 30.0], highlights: [0.0, 5.0, -20.0], preserve_luminosity: true },
            true,
        ),
        (Adjustment::ColorBalance { shadows: [60.0, 0.0, 0.0], midtones: [0.0; 3], highlights: [0.0, 0.0, 40.0], preserve_luminosity: false }, true),
        (Adjustment::BlackWhite { weights: [40.0, 60.0, 40.0, 60.0, 20.0, 80.0], tint: None }, true),
        (Adjustment::BlackWhite { weights: [70.0, 20.0, 50.0, -10.0, 90.0, 130.0], tint: Some([0.9, 0.7, 0.5]) }, true),
        (Adjustment::PhotoFilter { color: [q16(60620), q16(35455), q16(0)], density: 0.4, preserve_luminosity: true }, true),
        (Adjustment::PhotoFilter { color: [q16(0), q16(39321), q16(65535)], density: 0.3, preserve_luminosity: false }, true),
        (Adjustment::ChannelMixer { matrix: [[0.5, 0.3, 0.2, 0.0], [0.1, 0.8, 0.1, 0.05], [0.0, 0.2, 0.9, -0.05]], monochrome: false }, true),
        (Adjustment::ChannelMixer { matrix: [[0.4, 0.4, 0.2, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], monochrome: true }, true),
        (Adjustment::ColorLookup { name: "Look.cube".into(), lut: Some(Arc::new(lut.data)), size: 5, tetrahedral: false, dither: false }, true),
        // No table chosen: written as an identity cube (renders the same, comes back with a table).
        (Adjustment::ColorLookup { name: String::new(), lut: None, size: 0, tetrahedral: false, dither: false }, false),
        (Adjustment::Invert, true),
        (Adjustment::Posterize { levels: 4 }, true),
        (Adjustment::Threshold { level: g(140) }, true),
        (
            Adjustment::GradientMap {
                stops: vec![(0.0, [q16(6554), 0.0, q16(19661)]), (0.5, [1.0, 0.2, 0.0]), (1.0, [1.0, 1.0, 0.8])],
                reverse: false,
                dither: false,
            },
            true,
        ),
        (Adjustment::GradientMap { stops: vec![(0.0, [0.0, 0.0, 1.0]), (1.0, [1.0, 1.0, 0.0])], reverse: true, dither: false }, true),
        // Fewer than two stops: written as the equivalent two-stop gradient.
        (Adjustment::GradientMap { stops: vec![], reverse: false, dither: false }, false),
        (Adjustment::GradientMap { stops: vec![(0.4, [0.2, 0.6, 0.4])], reverse: false, dither: false }, false),
        (Adjustment::SelectiveColor { relative: true, adjustments: std::array::from_fn(|r| [r as f32 * 10.0 - 40.0, 5.0, -30.0, 20.0]) }, true),
        (Adjustment::SelectiveColor { relative: false, adjustments: std::array::from_fn(|r| [0.0, r as f32 * 5.0, 0.0, -10.0]) }, true),
        // A block this version cannot read (a truncated Selective Color) is kept verbatim.
        (Adjustment::Unsupported { psd_key: "selc".into(), raw: vec![0, 1, 0, 0] }, true),
    ];
    // Compile-time guard: a new Adjustment kind must be added to the list above.
    for (a, _) in &all {
        match a {
            Adjustment::BrightnessContrast { .. }
            | Adjustment::Levels { .. }
            | Adjustment::Curves { .. }
            | Adjustment::Exposure { .. }
            | Adjustment::Vibrance { .. }
            | Adjustment::HueSaturation { .. }
            | Adjustment::ColorBalance { .. }
            | Adjustment::BlackWhite { .. }
            | Adjustment::PhotoFilter { .. }
            | Adjustment::ChannelMixer { .. }
            | Adjustment::ColorLookup { .. }
            | Adjustment::Invert
            | Adjustment::Posterize { .. }
            | Adjustment::Threshold { .. }
            | Adjustment::GradientMap { .. }
            | Adjustment::SelectiveColor { .. }
            | Adjustment::Unsupported { .. } => {}
        }
    }
    all
}

/// A colourful base, a clipping base, and `adj` three times: plain; at half opacity with a mask
/// and a blend mode; clipped to a pixel layer.
fn adjustment_doc(adj: &Adjustment, depth: SampleType) -> Document {
    let mut d = Document::new("adj", photocraft_geom::Size::new(32, 20), ColorMode::Rgb, depth);
    let fmt = d.pixel_format();
    d.layers.push(raster("Background", fmt, d.bounds(), 1, false));
    let mut plain = Layer::new("plain", LayerContent::Adjustment(adj.clone()));
    plain.opacity = 1.0;
    d.layers.push(plain);
    let mut masked = Layer::new("masked", LayerContent::Adjustment(adj.clone()));
    masked.opacity = g(128);
    masked.blend = BlendMode::Multiply;
    masked.mask = Some(LayerMask { surface: mask_surface(depth, Rect::new(4, 3, 20, 15), 1.0, 3), enabled: true, linked: true, density: 1.0, feather: 0.0 });
    d.layers.push(masked);
    d.layers.push(raster("clip base", fmt, Rect::new(10, 2, 28, 18), 9, true));
    let mut clipped = Layer::new("clipped", LayerContent::Adjustment(adj.clone()));
    clipped.clipped = true;
    clipped.fill_opacity = g(200);
    d.layers.push(clipped);
    d
}

fn check_adjustment(adj: &Adjustment, lossless: bool, depth: SampleType) {
    let ctx = format!("{adj:?} @ {depth:?}");
    let d = adjustment_doc(adj, depth);
    let (back, warnings, file) = export_import(&d);
    assert!(!warnings.iter().any(|w| w.contains("adjustment")), "{ctx}: export warned {warnings:?}");
    // The adjustment layers carry an adjustment block, never an empty pixel layer.
    let with_adj = file.layers().iter().filter(|l| l.blocks.iter().any(|b| photocraft_io::ADJUSTMENT_KEYS.contains(&&b.key))).count();
    assert_eq!(with_adj, 3, "{ctx}: adjustment blocks written");
    assert_eq!(back.layers.len(), d.layers.len(), "{ctx}: layer count");
    for (x, y) in d.layers.iter().zip(&back.layers) {
        match (&x.content, &y.content) {
            (LayerContent::Adjustment(a), LayerContent::Adjustment(b)) => {
                if lossless {
                    assert_eq!(a, b, "{ctx}: adjustment");
                } else {
                    assert_eq!(std::mem::discriminant(a), std::mem::discriminant(b), "{ctx}: kind");
                }
                assert_eq!((x.blend, x.opacity, x.fill_opacity, x.clipped), (y.blend, y.opacity, y.fill_opacity, y.clipped), "{ctx}: {}", x.name);
                assert_eq!(x.mask.is_some(), y.mask.is_some(), "{ctx}: {} mask", x.name);
            }
            (LayerContent::Raster(_), LayerContent::Raster(_)) => {}
            (p, q) => panic!("{ctx}: {} became {}", p.kind_name(), q.kind_name()),
        }
    }
    if lossless {
        assert_docs_eq(&d, &back);
    }
    assert_composite_eq(&d, &back, &ctx);
    // A second generation is a fixed point for the document model.
    let (again, _, _) = export_import(&back);
    for (x, y) in back.layers.iter().zip(&again.layers) {
        if let (LayerContent::Adjustment(a), LayerContent::Adjustment(b)) = (&x.content, &y.content) {
            assert_eq!(a, b, "{ctx}: second generation");
        }
    }
}

#[test]
fn every_adjustment_kind_round_trips_8bit() {
    for (adj, lossless) in adjustments() {
        check_adjustment(&adj, lossless, SampleType::U8);
    }
}

#[test]
fn every_adjustment_kind_round_trips_16bit() {
    for (adj, lossless) in adjustments() {
        check_adjustment(&adj, lossless, SampleType::U16);
    }
}

/// Photoshop's `clrL` has no interpolation setting, so tetrahedral comes back trilinear: the
/// save must say so (#1347). Trilinear saves without a warning.
#[test]
fn color_lookup_tetrahedral_warns_on_psd_save() {
    let lut = photocraft_cms::lutfile::LutFile::from_fn("t", 2, |c| if c == [1.0; 3] { [1.0; 3] } else { [0.0; 3] });
    for tetrahedral in [true, false] {
        let adj = Adjustment::ColorLookup { name: "test.cube".into(), lut: Some(Arc::new(lut.data.clone())), size: 2, tetrahedral, dither: false };
        let mut d = Document::new("lut", photocraft_geom::Size::new(8, 8), ColorMode::Rgb, SampleType::U8);
        let fmt = d.pixel_format();
        d.layers.push(raster("Background", fmt, d.bounds(), 1, false));
        d.layers.push(Layer::new("Look", LayerContent::Adjustment(adj)));
        let (back, warnings, _) = export_import(&d);
        let warned = warnings.iter().any(|w| w.contains("\"Look\"") && w.contains("tetrahedral") && w.contains("trilinear"));
        assert_eq!(warned, tetrahedral, "tetrahedral={tetrahedral}: {warnings:?}");
        let content = back.layers.get(1).map(|l| &l.content);
        let Some(LayerContent::Adjustment(Adjustment::ColorLookup { tetrahedral: t, lut: Some(_), size: 2, .. })) = content else {
            panic!("Color Lookup lost: {:?}", content.map(LayerContent::kind_name))
        };
        assert!(!t, "PSD reopens as trilinear");
    }
}

/// One document per blend mode: the mode on a pixel layer (with fill opacity and a mask), on a
/// clipped layer, on a group, inside a pass-through group, and on an adjustment layer.
fn blend_doc(mode: BlendMode) -> Document {
    let mut d = Document::new("blend", photocraft_geom::Size::new(24, 16), ColorMode::Rgb, SampleType::U8);
    let fmt = d.pixel_format();
    d.layers.push(raster("Background", fmt, d.bounds(), 1, false));
    let mut l = raster("layer", fmt, Rect::new(2, 1, 20, 12), 2, true);
    l.blend = mode;
    l.fill_opacity = g(150);
    l.opacity = g(220);
    l.mask =
        Some(LayerMask { surface: mask_surface(SampleType::U8, Rect::new(3, 2, 15, 10), 1.0, 4), enabled: true, linked: true, density: 1.0, feather: 0.0 });
    d.layers.push(l);
    let mut clipped = raster("clipped", fmt, Rect::new(0, 0, 12, 16), 3, true);
    clipped.clipped = true;
    clipped.blend = mode;
    d.layers.push(clipped);
    let mut inner = raster("in pass-through", fmt, Rect::new(8, 4, 24, 16), 4, true);
    inner.blend = mode;
    d.layers.push(Layer::group("pass-through", vec![inner]));
    let mut grp = Layer::group("group", vec![raster("group child", fmt, Rect::new(4, 6, 18, 14), 5, true)]);
    grp.blend = mode;
    grp.opacity = g(200);
    d.layers.push(grp);
    let mut adj = Layer::new("adjustment", LayerContent::Adjustment(Adjustment::Invert));
    adj.blend = mode;
    adj.opacity = g(100);
    d.layers.push(adj);
    d
}

#[test]
fn every_blend_mode_round_trips() {
    let mut modes = BlendMode::LAYER_MODES.to_vec();
    modes.push(BlendMode::PassThrough);
    for mode in modes {
        if mode == BlendMode::PassThrough {
            // Pass-through is a group-only mode; blend_doc's "pass-through" group covers it,
            // and a pass-through group with an adjustment inside checks it affects the backdrop.
            let mut d = blend_doc(BlendMode::Normal);
            let fmt = d.pixel_format();
            let adj = Layer::new("inside", LayerContent::Adjustment(Adjustment::Invert));
            d.layers.push(Layer::group("pass-through adj", vec![raster("px", fmt, Rect::new(0, 0, 6, 6), 6, true), adj]));
            let (back, _, file) = export_import(&d);
            assert!(file.layers().iter().any(|l| l.blend_mode == photocraft_psd::BlendMode::PassThrough), "pass key written");
            assert_docs_eq(&d, &back);
            assert_composite_eq(&d, &back, "pass-through");
            continue;
        }
        let d = blend_doc(mode);
        let (back, _, _) = export_import(&d);
        assert_docs_eq(&d, &back);
        assert_composite_eq(&d, &back, mode.label());
    }
}
