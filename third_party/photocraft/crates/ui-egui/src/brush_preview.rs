//! Brush previews for the Brush Settings and Brushes panels: the S-curve stroke strip and tip
//! thumbnails, rendered with the real brush engine and cached per slot on a cheap signature of the
//! settings, so a preview re-renders only when something that changes it changes.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use egui::Color32;
use photocraft_engine::BrushSettings;
use photocraft_engine::paint::{self, Pattern, StrokePoint, TipShape};

/// Values cached per named slot, each tagged with the signature it was rendered for. A slot holds
/// one value, so dragging a slider replaces the slot's texture instead of piling up new ones.
#[derive(Clone)]
pub struct SigCache<T> {
    slots: HashMap<String, (u64, T)>,
    /// Number of renders so far (tests and the perf overlay).
    pub renders: u64,
}

impl<T> Default for SigCache<T> {
    fn default() -> Self {
        Self { slots: HashMap::new(), renders: 0 }
    }
}

impl<T: Clone> SigCache<T> {
    /// The value for `slot`, rendered by `render` only when the slot is empty or `sig` changed.
    pub fn get(&mut self, slot: &str, sig: u64, render: impl FnOnce() -> T) -> T {
        if let Some((s, v)) = self.slots.get(slot)
            && *s == sig
        {
            return v.clone();
        }
        self.renders += 1;
        let v = render();
        self.slots.insert(slot.to_string(), (sig, v.clone()));
        v
    }
    /// Drop slots not in `keep` (presets deleted from the session).
    pub fn retain(&mut self, keep: impl Fn(&str) -> bool) {
        self.slots.retain(|k, _| keep(k));
    }
    pub fn len(&self) -> usize {
        self.slots.len()
    }
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

fn js<T: serde::Serialize>(h: &mut DefaultHasher, v: &T) {
    serde_json::to_vec(v).unwrap_or_default().hash(h);
}

/// Size plus a strided sample of a bitmap: imported tips can be millions of pixels.
fn tile_sig(h: &mut DefaultHasher, t: &paint::GrayTile) {
    (t.width, t.height, t.data.len()).hash(h);
    let step = (t.data.len() / 4096).max(1);
    t.data.iter().step_by(step).for_each(|v| v.hash(h));
}

/// A stored tile (#1843) by its key, size and whether its full bitmap is loaded (the preview
/// drawn until then is replaced once it is).
fn stored_sig(h: &mut DefaultHasher, r: &paint::StoredTile) {
    (&r.key, r.width, r.height, r.is_loaded()).hash(h);
}

fn tip_shape_sig(h: &mut DefaultHasher, t: &TipShape) {
    match t {
        TipShape::Round => 0u8.hash(h),
        TipShape::Sampled(g) => tile_sig(h, g),
        TipShape::Stored(r) => stored_sig(h, r),
    }
}

/// Cheap signature of everything that changes a brush's stroke preview.
pub fn preview_sig(b: &BrushSettings) -> u64 {
    let mut h = DefaultHasher::new();
    js(&mut h, &(b.size, b.hardness, b.spacing, b.opacity, b.flow, b.pressure_size, b.pressure_opacity, b.erase, b.mode, b.angle, b.roundness));
    js(&mut h, &(b.flip_x, b.flip_y, b.aliased, b.noise, b.wet_edges, b.build_up, b.build_up_rate, b.protect_texture, b.seed));
    js(&mut h, &(&b.shape_dynamics, &b.scattering, &b.color_dynamics, &b.transfer, &b.pose, &b.smoothing));
    tip_shape_sig(&mut h, &b.tip);
    let d = &b.dual_brush;
    js(&mut h, &(d.enabled, d.mode, d.size, d.hardness, d.roundness, d.angle, d.spacing, d.scatter, d.both_axes, d.count, d.flip));
    tip_shape_sig(&mut h, &d.tip);
    let t = &b.texture;
    js(&mut h, &(t.enabled, t.invert, t.scale, t.brightness, t.contrast, t.each_tip, t.mode, t.depth, t.depth_jitter));
    match &t.pattern {
        Pattern::Tile(g) => tile_sig(&mut h, g),
        Pattern::Stored(r) => stored_sig(&mut h, r),
        p => js(&mut h, p),
    }
    h.finish()
}

/// Signature of a tip thumbnail's inputs.
pub fn tip_sig(tip: &TipShape, hardness: f32, angle: f32, roundness: f32, flip: (bool, bool)) -> u64 {
    let mut h = DefaultHasher::new();
    tip_shape_sig(&mut h, tip);
    (hardness.to_bits(), angle.to_bits(), roundness.to_bits(), flip).hash(&mut h);
    h.finish()
}

/// A stroke rendered with the brush (scaled down to fit), as straight RGBA8: Photoshop's preview
/// S-curve with pressure tapering in and out.
pub fn preview_pixels(b: &BrushSettings, w: u32, h: u32, color: [f32; 4]) -> Vec<u8> {
    use photocraft_color::{ColorMode, PixelFormat, SampleType};
    let mut brush = b.clone();
    let k = ((h as f32 * 0.55) / brush.size.max(1.0)).min(1.0);
    brush.size = (brush.size * k).max(1.0);
    brush.dual_brush.size = (brush.dual_brush.size * k).max(1.0);
    brush.color = color;
    brush.erase = false;
    brush.seed = 7;
    // Build-up adds dabs over (simulated) time: the preview has none.
    brush.build_up = false;
    let n = 64;
    let points: Vec<StrokePoint> = (0..=n)
        .map(|i| {
            let s = i as f64 / n as f64;
            let x = 10.0 + s * (w as f64 - 20.0);
            let y = h as f64 / 2.0 - (s * std::f64::consts::TAU).sin() * h as f64 * 0.22;
            StrokePoint::new(x, y, (s * std::f64::consts::PI).sin().max(0.05) as f32)
        })
        .collect();
    let fmt = PixelFormat::new(ColorMode::Rgb, SampleType::U8, true);
    let mut s = photocraft_raster::Surface::new(fmt);
    paint::apply_stroke(&mut s, &paint::Stroke { brush, points }, None, false);
    let r = photocraft_geom::Rect::new(0, 0, w as i32, h as i32);
    let mut px = vec![[0.0f32; 4]; (w * h) as usize];
    s.read_rgba_into(r, &mut px);
    px.iter().flat_map(|p| p.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)).collect()
}

/// Box-downsample a tip bitmap so its larger side is at most `max` (cheap bilinear lookups after).
fn shrink(g: &paint::GrayTile, max: u32) -> (u32, u32, Vec<f32>) {
    let (w, h) = (g.width.max(1), g.height.max(1));
    if !g.is_valid() {
        return (1, 1, vec![0.0]);
    }
    let f = (w.max(h) as f32 / max.max(1) as f32).max(1.0);
    let (nw, nh) = (((w as f32 / f).ceil() as u32).max(1), ((h as f32 / f).ceil() as u32).max(1));
    let mut sum = vec![0.0f32; (nw * nh) as usize];
    let mut cnt = vec![0u32; (nw * nh) as usize];
    for y in 0..h {
        let ny = ((y as f32 / f) as u32).min(nh - 1);
        for x in 0..w {
            let nx = ((x as f32 / f) as u32).min(nw - 1);
            let i = (ny * nw + nx) as usize;
            sum[i] += f32::from(g.data[(y * w + x) as usize]) / 65535.0;
            cnt[i] += 1;
        }
    }
    (nw, nh, sum.iter().zip(&cnt).map(|(s, c)| if *c > 0 { s / *c as f32 } else { 0.0 }).collect())
}

/// Coverage (0..1) of a brush tip drawn into an `n × n` cell, with the tip's angle, roundness,
/// flips and (for computed tips) hardness: the Brush Tip Shape grid and Brushes panel thumbnails.
pub fn tip_alpha(tip: &TipShape, hardness: f32, angle: f32, roundness: f32, flip: (bool, bool), n: u32) -> Vec<f32> {
    let n = n.clamp(2, 512);
    let c = n as f32 / 2.0;
    let r = c - 1.0;
    let (sin, cos) = angle.to_radians().sin_cos();
    let round = roundness.clamp(0.01, 1.0);
    // A stored tip draws from its saved preview unless loaded (#1843).
    let sampled = tip.bitmap().map(|g| shrink(g, n * 2));
    let mut out = vec![0.0f32; (n * n) as usize];
    // 2×2 supersampling keeps small thumbnails smooth.
    for y in 0..n {
        for x in 0..n {
            let mut acc = 0.0;
            for (sx, sy) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                let (dx, dy) = (x as f32 + sx - c, c - (y as f32 + sy));
                // Into tip space: undo the counter-clockwise angle, then the roundness squash.
                let mut u = dx * cos + dy * sin;
                let mut v = (-dx * sin + dy * cos) / round;
                if flip.0 {
                    u = -u;
                }
                if flip.1 {
                    v = -v;
                }
                acc += match &sampled {
                    None => paint::dab_coverage(u.hypot(v), r, hardness),
                    Some((tw, th, data)) => {
                        // The tip's larger side spans the cell diameter.
                        let k = (*tw).max(*th) as f32 / (2.0 * r);
                        let tx = u * k + *tw as f32 / 2.0 - 0.5;
                        let ty = -v * k + *th as f32 / 2.0 - 0.5;
                        bilinear(data, *tw, *th, tx, ty)
                    }
                };
            }
            out[(y * n + x) as usize] = (acc / 4.0).clamp(0.0, 1.0);
        }
    }
    out
}

fn bilinear(data: &[f32], w: u32, h: u32, x: f32, y: f32) -> f32 {
    if !(x > -1.0 && y > -1.0 && x < w as f32 && y < h as f32) {
        return 0.0;
    }
    let at = |xi: i32, yi: i32| -> f32 {
        if xi < 0 || yi < 0 || xi >= w as i32 || yi >= h as i32 { 0.0 } else { data.get((yi as u32 * w + xi as u32) as usize).copied().unwrap_or(0.0) }
    };
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (xi, yi) = (x0 as i32, y0 as i32);
    let top = at(xi, yi) * (1.0 - fx) + at(xi + 1, yi) * fx;
    let bottom = at(xi, yi + 1) * (1.0 - fx) + at(xi + 1, yi + 1) * fx;
    top * (1.0 - fy) + bottom * fy
}

fn upload(ctx: &egui::Context, name: &str, w: u32, h: u32, rgba: &[u8]) -> egui::TextureHandle {
    let img = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], rgba);
    ctx.load_texture(name, img, egui::TextureOptions::LINEAR)
}

fn cache_id() -> egui::Id {
    egui::Id::new("brush-preview-cache")
}

/// Run `f` on the context's preview cache. The cache is taken out of egui's memory while `f` runs,
/// because uploading a texture needs the context (which `data_mut` holds locked).
pub fn with_cache<R>(ctx: &egui::Context, f: impl FnOnce(&mut SigCache<egui::TextureHandle>) -> R) -> R {
    let mut cache = ctx.data_mut(|d| std::mem::take(d.get_temp_mut_or_default::<SigCache<egui::TextureHandle>>(cache_id())));
    let r = f(&mut cache);
    ctx.data_mut(|d| d.insert_temp(cache_id(), cache));
    r
}

/// Renders done by the preview cache so far.
pub fn render_count(ctx: &egui::Context) -> u64 {
    with_cache(ctx, |c| c.renders)
}

/// The stroke preview of `b` for `slot`, re-rendered only when the brush or size changes.
pub fn stroke_texture(ctx: &egui::Context, slot: &str, b: &BrushSettings, w: u32, h: u32, color: Color32) -> egui::TextureHandle {
    let mut hs = DefaultHasher::new();
    (preview_sig(b), w, h, color).hash(&mut hs);
    let sig = hs.finish();
    with_cache(ctx, |cache| {
        cache.get(slot, sig, || {
            let c = color.to_normalized_gamma_f32();
            upload(ctx, slot, w, h, &preview_pixels(b, w, h, [c[0], c[1], c[2], 1.0]))
        })
    })
}

/// A tip thumbnail for `slot` (`n × n`; `shape` is hardness, angle, roundness), re-rendered only
/// when the tip changes.
pub fn tip_texture(ctx: &egui::Context, slot: &str, tip: &TipShape, shape: (f32, f32, f32), n: u32, color: Color32) -> egui::TextureHandle {
    let (hardness, angle, roundness) = shape;
    let mut hs = DefaultHasher::new();
    (tip_sig(tip, hardness, angle, roundness, (false, false)), n, color).hash(&mut hs);
    let sig = hs.finish();
    with_cache(ctx, |cache| {
        cache.get(slot, sig, || {
            let a = tip_alpha(tip, hardness, angle, roundness, (false, false), n);
            let px: Vec<u8> = a.iter().flat_map(|v| [color.r(), color.g(), color.b(), (v * 255.0).round() as u8]).collect();
            upload(ctx, slot, n, n, &px)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sig_cache_renders_only_on_change() {
        let mut c: SigCache<u32> = SigCache::default();
        let mut b = BrushSettings::default();
        let mut calls = 0;
        for _ in 0..5 {
            c.get("strip", preview_sig(&b), || {
                calls += 1;
                calls
            });
        }
        assert_eq!((calls, c.renders), (1, 1));
        b.shape_dynamics.size.jitter = 0.5;
        assert_eq!(c.get("strip", preview_sig(&b), || 99), 99);
        assert_eq!(c.get("strip", preview_sig(&b), || 100), 99);
        assert_eq!((c.renders, c.len()), (2, 1));
        // Colour and the session's foreground don't matter; the brush's own fields do.
        let base = preview_sig(&BrushSettings::default());
        assert_eq!(base, preview_sig(&BrushSettings { color: [1.0, 0.0, 0.0, 1.0], ..Default::default() }));
        assert_ne!(base, preview_sig(&BrushSettings { texture: paint::Texture { enabled: true, ..Default::default() }, ..Default::default() }));
        c.retain(|_| false);
        assert!(c.is_empty());
    }

    #[test]
    fn preview_draws_a_stroke_that_fits() {
        let mut b = BrushSettings { size: 400.0, ..Default::default() };
        let px = preview_pixels(&b, 120, 40, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(px.len(), 120 * 40 * 4);
        let covered = px.as_chunks::<4>().0.iter().filter(|p| p[3] > 128).count();
        assert!(covered > 200 && covered < 120 * 40 / 2, "{covered}");
        // Dynamics change the preview.
        b.scattering.enabled = true;
        b.scattering.scatter.jitter = 3.0;
        assert_ne!(preview_pixels(&b, 120, 40, [1.0, 0.0, 0.0, 1.0]), px);
        // The Brush Settings strip with several sections on stays cheap to re-render.
        b.shape_dynamics.enabled = true;
        b.shape_dynamics.size.jitter = 0.4;
        b.texture.enabled = true;
        b.dual_brush.enabled = true;
        let t0 = std::time::Instant::now();
        preview_pixels(&b, 488, 76, [1.0; 4]);
        eprintln!("strip render: {:.2} ms", t0.elapsed().as_secs_f64() * 1e3);
    }

    #[test]
    fn stored_tips_draw_their_saved_preview_until_loaded() {
        // #1843: library presets keep a small preview, and thumbnails never need the full tip.
        let full = paint::GrayTile::from_fn(900, 300, |x, _| if x < 450 { 1.0 } else { 0.4 });
        let r =
            paint::StoredTile { key: "k".into(), width: 900, height: 300, preview: std::sync::Arc::new(full.preview(paint::tile::PREVIEW_SIDE)), full: None };
        let loaded = paint::StoredTile { full: Some(std::sync::Arc::new(full.clone())), ..r.clone() };
        let n = 36;
        let thumb = |t: &TipShape| tip_alpha(t, 1.0, 0.0, 1.0, (false, false), n);
        let (from_preview, from_full) = (thumb(&TipShape::Stored(r.clone())), thumb(&TipShape::Sampled(full)));
        let worst = from_preview.iter().zip(&from_full).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 0.05, "preview thumbnail differs by {worst}");
        // Loading the tip re-renders the thumbnail; the stroke preview paints with the preview.
        let sig = |t: paint::StoredTile| tip_sig(&TipShape::Stored(t), 1.0, 0.0, 1.0, (false, false));
        assert_ne!(sig(r.clone()), sig(loaded));
        let b = BrushSettings { tip: TipShape::Stored(r), size: 900.0, ..Default::default() };
        let px = preview_pixels(&b, 120, 40, [1.0, 0.0, 0.0, 1.0]);
        assert!(px.as_chunks::<4>().0.iter().any(|p| p[3] > 128));
    }

    #[test]
    fn tip_thumbnails_follow_shape_angle_and_roundness() {
        let n = 32;
        let at = |a: &[f32], x: u32, y: u32| a[(y * n + x) as usize];
        let hard = tip_alpha(&TipShape::Round, 1.0, 0.0, 1.0, (false, false), n);
        assert!(at(&hard, 16, 16) > 0.99 && at(&hard, 0, 0) < 0.01 && at(&hard, 29, 16) > 0.9);
        let soft = tip_alpha(&TipShape::Round, 0.0, 0.0, 1.0, (false, false), n);
        assert!(at(&soft, 26, 16) < at(&hard, 26, 16));
        // 25 % roundness: flat ellipse, wide along x; rotated 90°, tall along y.
        let flat = tip_alpha(&TipShape::Round, 1.0, 0.0, 0.25, (false, false), n);
        assert!(at(&flat, 28, 16) > 0.9 && at(&flat, 16, 4) < 0.01);
        let tall = tip_alpha(&TipShape::Round, 1.0, 90.0, 0.25, (false, false), n);
        assert!(at(&tall, 16, 4) > 0.9 && at(&tall, 28, 16) < 0.01);
        // A sampled tip: a 300×100 bar fills the cell's width, centred.
        let bar = paint::GrayTile::from_fn(300, 100, |_, _| 1.0);
        let s = tip_alpha(&TipShape::Sampled(bar), 1.0, 0.0, 1.0, (false, false), n);
        assert!(at(&s, 3, 16) > 0.9 && at(&s, 16, 2) < 0.01);
        // Degenerate input never panics.
        let empty = paint::GrayTile { width: 0, height: 0, data: vec![].into() };
        assert!(tip_alpha(&TipShape::Sampled(empty), 1.0, 0.0, 0.0, (true, true), 0).iter().all(|v| *v == 0.0));
    }
}
