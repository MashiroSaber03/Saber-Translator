//! Shared test helpers: document generators and structural comparison.
#![allow(dead_code)]

use std::sync::Arc;

use photocraft_color::{BlendMode, Color, ColorMode, PixelFormat, SampleType};
use photocraft_doc::adjust::{CurvePoint, LevelsChannel};
use photocraft_doc::*;
use photocraft_geom::{Rect, Size};
use photocraft_raster::Surface;

/// Quantize to the 1/255 grid so values survive 8-bit PSD fields.
pub fn g(v: u8) -> f32 {
    f32::from(v) / 255.0
}

/// Pattern surface of `fmt` over `r` (deterministic; values on the 8-bit
/// grid so every depth round-trips exactly).
pub fn pattern(fmt: PixelFormat, r: Rect, seed: u32, alpha: bool) -> Surface {
    let mut s = Surface::new(fmt);
    let n = fmt.channels();
    let mut vals = Vec::with_capacity(r.width() as usize * r.height() as usize * n);
    for y in r.y0..r.y1 {
        for x in r.x0..r.x1 {
            for c in 0..n {
                let v = if c == n - 1 && fmt.alpha {
                    if alpha { (i64::from(x + y) * 37 + i64::from(seed)).rem_euclid(256) } else { 255 }
                } else {
                    (i64::from(x * 13 + y * 7) + i64::from(seed) * 31 + c as i64 * 50).rem_euclid(256)
                };
                vals.push(v as f32 / 255.0);
            }
        }
    }
    s.write_region(r, &vals);
    s.prune();
    s
}

pub fn mask_surface(sample: SampleType, r: Rect, default: f32, seed: u32) -> Surface {
    let fmt = PixelFormat::new(ColorMode::Grayscale, sample, false);
    let mut s = Surface::with_default(fmt, &[default]);
    let vals: Vec<f32> = (0..r.width() * r.height()).map(|i| ((i * 29 + seed) % 256) as f32 / 255.0).collect();
    s.write_region(r, &vals);
    s.prune();
    s
}

pub fn fill_color(mode: ColorMode) -> Color {
    match mode {
        ColorMode::Cmyk => Color { mode: ColorMode::Cmyk, c: [0.2, 0.4, 0.6, 0.1], alpha: 1.0 },
        ColorMode::Grayscale => Color::gray(0.25),
        _ => Color::rgb(1.0, 0.5, 0.25),
    }
}

/// Feature flags for generated docs.
#[derive(Clone, Copy)]
pub struct Features {
    pub groups: bool,
    pub masks: bool,
    pub adjustments: bool,
    pub fills: bool,
    pub all_blends: bool,
    pub extras: bool,
}

impl Features {
    pub const ALL: Features = Features { groups: true, masks: true, adjustments: true, fills: true, all_blends: true, extras: true };
    pub const PIXELS: Features = Features { groups: true, masks: true, adjustments: false, fills: false, all_blends: true, extras: false };
}

pub fn raster(name: &str, fmt: PixelFormat, r: Rect, seed: u32, alpha: bool) -> Layer {
    Layer::new(name, LayerContent::Raster(pattern(fmt, r, seed, alpha)))
}

/// A generated document exercising the level 1-3 mapping.
pub fn gen_doc(mode: ColorMode, depth: SampleType, f: Features) -> Document {
    let size = Size::new(24, 16);
    let mut d = Document::new("gen", size, mode, depth);
    let fmt = d.pixel_format();
    let mut bg = raster("Background", fmt, d.bounds(), 1, false);
    bg.locks.transparency = true;
    bg.locks.position = true;
    d.layers.push(bg);

    let mut neg = raster("Negative \u{1F600}", fmt, Rect::new(-5, -3, 7, 6), 2, true);
    neg.opacity = g(200);
    neg.fill_opacity = g(128);
    neg.blend = BlendMode::Multiply;
    neg.label = LabelColor::Blue;
    // Advanced Blending switches round-trip as `infx` / `tsly` (and the rest below).
    neg.advanced.blend_interior = true;
    neg.advanced.transparency_shapes = false;
    d.layers.push(neg);

    let mut hidden = raster("Hidden", fmt, Rect::new(3, 3, 9, 9), 3, true);
    hidden.visible = false;
    d.layers.push(hidden);

    let mut clipped = raster("Clipped", fmt, Rect::new(0, 0, 10, 10), 4, true);
    clipped.clipped = true;
    clipped.blend = BlendMode::Screen;
    d.layers.push(clipped);

    d.layers.push(Layer::new("Empty", LayerContent::Raster(Surface::new(fmt))));

    if f.masks {
        let mut m = raster("Masked", fmt, Rect::new(2, 2, 14, 12), 5, true);
        m.mask = Some(LayerMask { surface: mask_surface(depth, Rect::new(4, 4, 10, 9), 1.0, 1), enabled: true, linked: true, density: 1.0, feather: 0.0 });
        m.advanced.layer_mask_hides_effects = true;
        m.advanced.vector_mask_hides_effects = true;
        d.layers.push(m);
        let mut m2 = raster("Masked2", fmt, Rect::new(-2, 1, 6, 5), 6, true);
        m2.mask =
            Some(LayerMask { surface: mask_surface(depth, Rect::new(-1, 0, 3, 4), 0.0, 2), enabled: false, linked: false, density: g(128), feather: 2.5 });
        d.layers.push(m2);
    }

    if f.groups {
        let inner_child = raster("Inner child", fmt, Rect::new(1, 1, 5, 5), 7, true);
        let mut inner = Layer::group("Inner", vec![inner_child]);
        inner.blend = BlendMode::Multiply;
        inner.opacity = g(180);
        if let LayerContent::Group(gr) = &mut inner.content {
            gr.expanded = false;
        }
        inner.advanced.knockout = photocraft_doc::Knockout::Shallow;
        let a = raster("Outer child", fmt, Rect::new(6, 2, 12, 8), 8, true);
        let mut outer = Layer::group("Outer", vec![a, inner]);
        outer.locks.position = true;
        outer.label = LabelColor::Red;
        outer.advanced.knockout = photocraft_doc::Knockout::Deep;
        outer.advanced.blend_clipped = false;
        d.layers.push(outer);
        d.layers.push(Layer::group("Empty group", vec![]));
    }

    if f.all_blends {
        for (i, m) in BlendMode::LAYER_MODES.iter().enumerate() {
            if *m == BlendMode::Dissolve {
                continue; // random, excluded from composite comparisons
            }
            let mut l = raster(&format!("blend {}", m.label()), fmt, Rect::from_xywh(i as i32 % 20, i as i32 % 12, 3, 2), 10 + i as u32, true);
            l.blend = *m;
            d.layers.push(l);
        }
    }

    if f.adjustments {
        let adjs = vec![
            Adjustment::Invert,
            Adjustment::Threshold { level: g(100) },
            Adjustment::Posterize { levels: 5 },
            Adjustment::BrightnessContrast { brightness: 20.0, contrast: -10.0, legacy: false },
            Adjustment::HueSaturation { hue: 30.0, saturation: -20.0, lightness: 5.0, colorize: false, ranges: photocraft_doc::adjust::HueRange::defaults() },
            Adjustment::Exposure { exposure: 0.5, offset: 0.0, gamma: 1.0 },
            Adjustment::Levels {
                master: LevelsChannel { in_black: g(10), in_white: g(240), gamma: 1.2, out_black: 0.0, out_white: 1.0 },
                per_channel: Default::default(),
                space: tone_space(mode),
                black: Default::default(),
            },
            Adjustment::Curves {
                master: vec![CurvePoint { input: 0.0, output: 0.0 }, CurvePoint { input: g(128), output: g(150) }, CurvePoint { input: 1.0, output: 1.0 }],
                per_channel: [
                    vec![CurvePoint { input: 0.0, output: 0.0 }, CurvePoint { input: 1.0, output: 1.0 }],
                    vec![CurvePoint { input: 0.0, output: 0.0 }, CurvePoint { input: 1.0, output: 1.0 }],
                    vec![CurvePoint { input: 0.0, output: 0.0 }, CurvePoint { input: 1.0, output: 1.0 }],
                ],
                space: tone_space(mode),
                black: Vec::new(),
                pencil: false,
            },
            Adjustment::Unsupported { psd_key: "selc".into(), raw: vec![0, 1, 0, 0] },
        ];
        // Colour adjustments (Photoshop offers them for colour documents only: they would tint
        // a grayscale composite).
        let colour = vec![
            Adjustment::Vibrance { vibrance: 40.0, saturation: -10.0 },
            Adjustment::ColorBalance { shadows: [10.0, 0.0, -5.0], midtones: [0.0, 20.0, 0.0], highlights: [-30.0, 0.0, 15.0], preserve_luminosity: true },
            Adjustment::BlackWhite { weights: [50.0, 60.0, 40.0, 60.0, 20.0, 70.0], tint: Some([1.0, 0.8, 0.6]) },
            Adjustment::PhotoFilter { color: [1.0, f32::from(32768u16) / 65535.0, 0.0], density: 0.25, preserve_luminosity: false },
            Adjustment::ChannelMixer { matrix: [[0.8, 0.2, 0.0, 0.0], [0.0, 1.0, 0.0, 0.1], [0.1, 0.0, 0.9, 0.0]], monochrome: false },
        ];
        let adjs = adjs.into_iter().chain(colour.into_iter().filter(|_| mode != ColorMode::Grayscale));
        for (i, a) in adjs.enumerate() {
            let mut l = Layer::new(format!("adj {i}"), LayerContent::Adjustment(a));
            l.opacity = g(if i % 2 == 0 { 255 } else { 128 });
            d.layers.push(l);
        }
    }

    if f.fills {
        let mut solid = Layer::new("Solid fill", LayerContent::Fill(Fill::Solid(fill_color(mode))));
        solid.opacity = g(64);
        d.layers.push(solid);
    }

    if f.extras {
        d.resolution_dpi = 300.0;
        d.icc_profile = Some(Arc::new(vec![1, 2, 3, 4]));
        d.metadata.xmp = Some("<x:xmpmeta/>".into());
        d.metadata.exif = Some(Arc::new(b"MM\0*\0\0\0\x08".to_vec()));
        d.metadata.psd_resources.push((4000, String::new(), Arc::new(vec![9, 9, 9])));
        d.guides.horizontal.push(5.5);
        d.guides.vertical.push(12.0);
        d.channels.push(AlphaChannel::new("Selection", mask_surface(depth, d.bounds(), 0.0, 3)));
    }
    d
}

fn pruned(s: &Surface) -> Surface {
    let mut s = s.clone();
    s.prune();
    s
}

fn surface_eq(a: &Surface, b: &Surface, ctx: &str) {
    let (a, b) = (pruned(a), pruned(b));
    assert_eq!(a.format(), b.format(), "{ctx}: format");
    assert_eq!(a.default_pixel(), b.default_pixel(), "{ctx}: default pixel");
    let r = a.content_bounds().union(&b.content_bounds());
    assert_eq!(a.content_bounds(), b.content_bounds(), "{ctx}: bounds");
    if !r.is_empty() {
        assert!(a.to_interleaved(r) == b.to_interleaved(r), "{ctx}: pixels differ");
    }
}

fn color_close(a: &Color, b: &Color, ctx: &str) {
    assert_eq!(a.mode, b.mode, "{ctx}: color mode");
    for i in 0..4 {
        assert!((a.c[i] - b.c[i]).abs() < 1e-4, "{ctx}: color {a:?} vs {b:?}");
    }
}

pub fn assert_layers_eq(a: &Layer, b: &Layer, path: &str) {
    let ctx = format!("{path}/{}", a.name);
    assert_eq!(a.name, b.name, "{ctx}: name");
    assert_eq!(a.visible, b.visible, "{ctx}: visible");
    assert_eq!(a.locks, b.locks, "{ctx}: locks");
    assert_eq!(a.blend, b.blend, "{ctx}: blend");
    assert_eq!(a.opacity, b.opacity, "{ctx}: opacity");
    assert_eq!(a.fill_opacity, b.fill_opacity, "{ctx}: fill");
    assert_eq!(a.clipped, b.clipped, "{ctx}: clipped");
    assert_eq!(a.label, b.label, "{ctx}: label");
    assert_eq!(a.advanced, b.advanced, "{ctx}: advanced blending");
    assert_eq!(a.excluded_channels, b.excluded_channels, "{ctx}: channel restrictions");
    assert_eq!(a.blend_if, b.blend_if, "{ctx}: blend if");
    assert_eq!(a.effects.psd_raw, b.effects.psd_raw, "{ctx}: effects raw");
    assert_eq!(a.effects.enabled, b.effects.enabled, "{ctx}: effects enabled");
    if a.psd_id.is_some() {
        assert_eq!(a.psd_id, b.psd_id, "{ctx}: psd id");
    }
    for blk in &a.psd_blocks {
        assert!(b.psd_blocks.contains(blk), "{ctx}: preserved block {:?} missing", String::from_utf8_lossy(&blk.0));
    }
    if let Some(fc) = &a.fill_cache {
        let other = b.fill_cache.as_ref().expect("fill cache kept");
        surface_eq(&fc.surface, &other.surface, &format!("{ctx}: fill cache"));
    }
    match (&a.mask, &b.mask) {
        (None, None) => {}
        (Some(x), Some(y)) => {
            assert_eq!(x.enabled, y.enabled, "{ctx}: mask enabled");
            assert_eq!(x.linked, y.linked, "{ctx}: mask linked");
            assert_eq!(x.density, y.density, "{ctx}: mask density");
            assert_eq!(x.feather, y.feather, "{ctx}: mask feather");
            surface_eq(&x.surface, &y.surface, &format!("{ctx}: mask"));
        }
        _ => panic!("{ctx}: mask presence differs"),
    }
    match (&a.content, &b.content) {
        (LayerContent::Raster(x), LayerContent::Raster(y)) => surface_eq(x, y, &ctx),
        (LayerContent::Group(x), LayerContent::Group(y)) => {
            assert_eq!(x.expanded, y.expanded, "{ctx}: expanded");
            assert_eq!(x.children.len(), y.children.len(), "{ctx}: child count");
            for (p, q) in x.children.iter().zip(&y.children) {
                assert_layers_eq(p, q, &ctx);
            }
        }
        (LayerContent::Adjustment(x), LayerContent::Adjustment(y)) => assert_eq!(x, y, "{ctx}: adjustment"),
        (LayerContent::Fill(Fill::Solid(x)), LayerContent::Fill(Fill::Solid(y))) => color_close(x, y, &ctx),
        (LayerContent::Fill(x), LayerContent::Fill(y)) => assert_eq!(x, y, "{ctx}: fill"),
        (LayerContent::Text(x), LayerContent::Text(y)) => {
            assert_eq!(x.text, y.text, "{ctx}: text");
            surface_eq(x.cache.as_ref().unwrap(), y.cache.as_ref().unwrap(), &ctx);
        }
        (LayerContent::Shape(x), LayerContent::Shape(y)) => {
            surface_eq(x.cache.as_ref().unwrap(), y.cache.as_ref().unwrap(), &ctx);
        }
        (LayerContent::Smart(x), LayerContent::Smart(y)) => {
            surface_eq(x.cache.as_ref().unwrap(), y.cache.as_ref().unwrap(), &ctx);
        }
        (x, y) => panic!("{ctx}: kind {} vs {}", x.kind_name(), y.kind_name()),
    }
}

pub fn assert_docs_eq(a: &Document, b: &Document) {
    assert_eq!(a.size, b.size);
    assert_eq!(a.mode, b.mode);
    assert_eq!(a.depth, b.depth);
    assert!((a.resolution_dpi - b.resolution_dpi).abs() < 1e-3, "dpi");
    assert_eq!(a.icc_profile, b.icc_profile, "icc");
    assert_eq!(a.metadata, b.metadata, "metadata");
    assert_eq!(a.guides, b.guides, "guides");
    assert_eq!(a.channels.len(), b.channels.len(), "alpha channels");
    for (x, y) in a.channels.iter().zip(&b.channels) {
        assert_eq!(x.name, y.name);
        surface_eq(&x.surface, &y.surface, &format!("channel {}", x.name));
    }
    assert_eq!(a.layers.len(), b.layers.len(), "root layer count");
    for (x, y) in a.layers.iter().zip(&b.layers) {
        assert_layers_eq(x, y, "");
    }
}

/// Max per-channel difference between two RGBA float buffers, over pixels
/// where either is not fully transparent (color is irrelevant at alpha 0).
pub fn max_diff(a: &[[f32; 4]], b: &[[f32; 4]]) -> f32 {
    let mut m = 0.0f32;
    for (p, q) in a.iter().zip(b) {
        if p[3] <= 0.0 && q[3] <= 0.0 {
            continue;
        }
        for c in 0..4 {
            let (x, y) = if c < 3 { (p[c] * p[3], q[c] * q[3]) } else { (p[c], q[c]) };
            m = m.max((x - y).abs());
        }
    }
    m
}

/// The Levels/Curves channel space a PSD of `mode` stores (its records are the document's channels).
fn tone_space(mode: ColorMode) -> photocraft_doc::adjust::ToneSpace {
    match mode {
        ColorMode::Cmyk => photocraft_doc::adjust::ToneSpace::Cmyk,
        ColorMode::Lab => photocraft_doc::adjust::ToneSpace::Lab,
        _ => photocraft_doc::adjust::ToneSpace::Rgb,
    }
}

/// Strict structural check of a written PSD/PSB (#200): every tagged block must
/// re-parse with its section sizes matching, laid out as Photoshop writes it so
/// that readers which step from block to block by length (psd-tools) stay
/// aligned. Returns one message per problem; empty means clean.
///
/// - the file parses, with no unparsed bytes left in the layer and mask section;
/// - the layer info body is padded to a multiple of 4;
/// - global (document-level) blocks are padded to a multiple of 4;
/// - layer blocks have an even length with no pad byte outside it;
/// - blocks with inner lengths (`PlLd`, `SoLd`, `SoLE`, `lnk2`/`lnk3`/`lnkD`, `lfx2`)
///   re-parse exactly (`TaggedBlock::check_structure`).
pub fn strict_block_errors(bytes: &[u8]) -> Vec<String> {
    let file = match photocraft_psd::PsdFile::from_bytes(bytes) {
        Ok(f) => f,
        Err(e) => return vec![format!("parse: {e}")],
    };
    let v = file.header.version;
    let mut errs = Vec::new();
    if !file.layer_mask_trailing.is_empty() {
        errs.push(format!("{} unparsed bytes at the end of the layer and mask section", file.layer_mask_trailing.len()));
    }
    if let Some(li) = &file.layer_info {
        let body = li.unpadded_len(v).unwrap_or(1);
        let pad = li.padding.as_ref().map_or(body % 2, |p| p.len() as u64);
        // In a global `Lr16`/`Lr32` block the padding to 4 follows the block, outside its length.
        let after = match &file.layer_info_placement {
            photocraft_psd::LayerInfoPlacement::GlobalBlock { padding, .. } => padding.as_ref().map_or((4 - (body + pad) % 4) % 4, |p| p.len() as u64),
            photocraft_psd::LayerInfoPlacement::Section => 0,
        };
        if (body + pad + after) % 4 != 0 {
            errs.push(format!("layer info: {body} bytes + {pad} padding is not a multiple of 4"));
        }
    }
    for b in &file.global_blocks {
        let pad = b.padding.as_ref().map_or((4 - b.data.len() % 4) % 4, Vec::len);
        if (b.data.len() + pad) % 4 != 0 {
            errs.push(format!("global {}: {} data bytes + {pad} padding is not a multiple of 4", b.key_str(), b.data.len()));
        }
        if let Err(e) = b.check_structure() {
            errs.push(format!("global {}: {e}", b.key_str()));
        }
    }
    for (i, l) in file.layers().iter().enumerate() {
        for b in &l.blocks {
            if b.data.len() % 2 != 0 || b.padding.as_ref().is_some_and(|p| !p.is_empty()) {
                errs.push(format!("layer {i} {}: {} data bytes with padding outside the length", b.key_str(), b.data.len()));
            }
            if let Err(e) = b.check_structure() {
                errs.push(format!("layer {i} {}: {e}", b.key_str()));
            }
        }
    }
    errs
}
