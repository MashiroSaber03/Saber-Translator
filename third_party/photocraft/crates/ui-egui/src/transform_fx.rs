//! Layer styles in the Free Transform live preview (#3653).
//!
//! The box previews the moving pixels as a texture on a mesh (transform_tex.rs), which leaves
//! the layer's style out: Stroke, Drop Shadow and the rest vanished while the box was open.
//! Baking the style into that texture would go wrong as soon as the box scales or turns, because
//! a style isn't transformed with its layer: it is drawn again from the layer's new shape (a
//! stroke keeps its width, a shadow its angle and distance). So a styled layer is composited
//! here instead. The moving texels are resampled through the box, the style is laid over them by
//! the CPU compositor's `composite_with_effects` (the code that draws the layer once the
//! transform is applied), and the result is drawn in place of the mesh.
//!
//! The work follows the screen, not the document: the composite is made at the resolution it is
//! shown at (never finer than the document), with the style's sizes scaled to match, and within
//! a pixel [`budget`] that keeps a rebuild near [`DRAG_MS`] while the pointer is down and near
//! [`REST_MS`] once the box rests. Past the budget the resolution drops instead of the frame
//! rate, and comes back when the pointer is released. It is rebuilt only when the box, the zoom
//! or the layer changes.
//!
//! The composite is drawn over transparency with Normal blending, like the mesh it replaces: a
//! layer's or an effect's blend mode shows against the layers under it only after the commit.

use std::sync::{Arc, Mutex, PoisonError};

use egui::{Color32, ColorImage, TextureHandle, TextureId, TextureOptions};
use photocraft_algo::transform::Homography;
use photocraft_color::{BlendMode, PixelFormat};
use photocraft_compose::Buffer;
use photocraft_compose::effects::{self, TextureCtx};
use photocraft_doc::{Document, Effect, Layer, LayerContent};
use photocraft_geom::Rect;
use photocraft_raster::Surface;

use crate::transform_tex::PreviewTextures;

/// What a rebuild should take while the pointer is down (a drag rebuilds every frame), in
/// milliseconds on the machine [`pixel_cost_ns`] was measured on.
pub const DRAG_MS: f64 = 30.0;
/// What a rebuild should take once the box rests.
pub const REST_MS: f64 = 120.0;
/// The fewest work pixels a budget asks for, however costly the style.
pub const MIN_PIXELS: f64 = 0.25e6;
/// The most work pixels in a composite, whatever was asked for.
pub const MAX_PIXELS: f64 = 8.0e6;
/// Work coordinates stay within this, so rectangle arithmetic on them cannot overflow.
const COORD_LIMIT: f64 = 5.0e8;
/// At this many screen pixels per document pixel the composite is shown with crisp texels,
/// like the canvas around it.
const CRISP_ZOOM: f32 = 2.0;

/// Where and how finely a styled preview is composited. Rectangles are in work pixels: document
/// pixels times `scale`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    /// Work pixels per document pixel, at most 1.
    pub scale: f64,
    /// The box: the layer's bounds after the transform.
    pub bounds: Rect,
    /// What is composited and drawn: the bounds grown by the style's reach, inside the canvas.
    pub shown: Rect,
    /// The moving texels resampled for it: `shown` grown by twice the reach, so every effect
    /// over `shown` sees all of the shape it depends on, also past the canvas edge.
    pub content: Rect,
}

impl Plan {
    /// `shown` in document pixels: where the composite is drawn.
    pub fn doc_rect(&self) -> [f32; 4] {
        let k = 1.0 / self.scale;
        [self.shown.x0, self.shown.y0, self.shown.x1, self.shown.y1].map(|v| (f64::from(v) * k) as f32)
    }
}

/// Rough cost of compositing one work pixel with `layer`'s style, in nanoseconds: measured per
/// effect at default settings on an Apple M3 Pro, release build. A Stroke costs the most by far
/// (two chamfer distance fields over the whole composite, 46 ns whatever its width); shadows and
/// glows build one blurred map (5 to 7 ns), Satin and Bevel & Emboss a little more (8 to 14 ns)
/// and overlays next to nothing. Resampling, laying the style on and the upload add about 12 ns.
pub fn pixel_cost_ns(layer: &Layer) -> f64 {
    let enabled = || layer.effects.items.iter().filter(|e| e.enabled());
    let maps: f64 = enabled()
        .map(|e| match e {
            Effect::Stroke(_) | Effect::ColorOverlay { .. } | Effect::GradientOverlay { .. } | Effect::PatternOverlay { .. } => 1.0,
            Effect::Satin(_) | Effect::BevelEmboss(_) => 12.0,
            Effect::DropShadow(_) | Effect::InnerShadow(_) | Effect::OuterGlow(_) | Effect::InnerGlow(_) => 6.0,
        })
        .sum();
    // Every Stroke reads the same two distance fields.
    let fields = if enabled().any(|e| matches!(e, Effect::Stroke(_))) { 46.0 } else { 0.0 };
    12.0 + maps + fields
}

/// Work pixels a composite of `layer` may have: what fits in [`DRAG_MS`] while the pointer is
/// down and in [`REST_MS`] once the box rests.
pub fn budget(layer: &Layer, dragging: bool) -> f64 {
    let ms = if dragging { DRAG_MS } else { REST_MS };
    (ms * 1.0e6 / pixel_cost_ns(layer)).clamp(MIN_PIXELS, MAX_PIXELS)
}

/// The moving layer as the compositor styles it at `scale`: its style at that scale with its
/// opacity, fill and Advanced Blending, over plain pixels (the texels are rendered already, with
/// the masks applied).
pub fn work_layer(layer: &Layer, scale: f64) -> Layer {
    let mut l = layer.clone();
    l.content = LayerContent::Raster(Surface::new(PixelFormat::RGBA8));
    l.mask = None;
    l.vector_mask = None;
    l.fill_cache = None;
    l.blend = BlendMode::Normal;
    photocraft_engine::image_cmds::scale_effects(&mut l.effects, scale as f32);
    l.effects.reference = l.effects.reference.map(|(x, y)| (x * scale, y * scale));
    l
}

/// Plan the composite of `layer` moved onto `quad` (document pixels, clockwise from top-left) in
/// a document of `canvas` pixels, shown at `zoom` screen pixels per document pixel, within
/// `budget` work pixels. `None` when nothing of it is on the canvas or the box isn't finite.
pub fn plan(layer: &Layer, canvas: [u32; 2], quad: [[f64; 2]; 4], zoom: f32, budget: f64) -> Option<(Plan, Layer)> {
    if quad.iter().flatten().any(|v| !v.is_finite()) {
        return None;
    }
    let (min, max) = quad.iter().fold(([f64::MAX; 2], [f64::MIN; 2]), |(lo, hi), q| ([lo[0].min(q[0]), lo[1].min(q[1])], [hi[0].max(q[0]), hi[1].max(q[1])]));
    let budget = if budget.is_finite() { budget.clamp(1.0, MAX_PIXELS) } else { MIN_PIXELS };
    let mut scale = if zoom.is_finite() && zoom > 0.0 { f64::from(zoom).min(1.0) } else { 1.0 };
    // The reach of the style is in work pixels too, so the area isn't simply the scale squared:
    // shrink until it fits (a few rounds settle it).
    for round in 0..6 {
        let work = work_layer(layer, scale);
        let reach = effects::margin(&work);
        let edge = |v: f64, up: bool| {
            let v = (v * scale).clamp(-COORD_LIMIT, COORD_LIMIT);
            (if up { v.ceil() } else { v.floor() }) as i32
        };
        let bounds = Rect::new(edge(min[0], false), edge(min[1], false), edge(max[0], true), edge(max[1], true));
        let canvas = Rect::new(0, 0, edge(f64::from(canvas[0]), true), edge(f64::from(canvas[1]), true));
        let shown = bounds.inflate(reach).intersect(&canvas);
        if bounds.is_empty() || shown.is_empty() {
            return None;
        }
        let content = shown.inflate(2 * reach);
        let area = f64::from(content.width()) * f64::from(content.height());
        if area <= budget {
            return Some((Plan { scale, bounds, shown, content }, work));
        }
        if round == 5 {
            break;
        }
        scale *= (budget / area).sqrt() * 0.98;
        if !(scale.is_finite() && scale > 0.0) {
            return None;
        }
    }
    None
}

/// One row of `out` per `y`, in parallel on native targets.
fn rows<T: Send>(out: &mut [T], width: usize, y0: i32, row: impl Fn(i32, &mut [T]) + Sync) {
    bands(out, width, |i, o| row(y0.saturating_add(i as i32), o));
}

/// The moving texels (`image`, premultiplied, of which `uv` covers the box's source rectangle)
/// drawn over `rect` as straight-alpha floats. `to_unit` takes a work pixel to the unit square
/// of the source rectangle, and `inside` is a work point within the box. The edges of the
/// rectangle fade over one texel, as a resampled layer's do; `nearest` keeps texels whole.
pub fn resample(image: &ColorImage, uv: [f32; 2], to_unit: &Homography, inside: [f64; 2], rect: Rect, nearest: bool) -> Buffer {
    let mut out = Buffer::transparent(rect);
    let [iw, ih] = image.size;
    let (tw, th) = (f64::from(uv[0]) * iw as f64, f64::from(uv[1]) * ih as f64);
    let m = to_unit.0;
    // The side of the horizon the box is on: a projective map sends the far side elsewhere.
    let side = (m[6] * inside[0] + m[7] * inside[1] + m[8]).signum();
    if iw == 0 || ih == 0 || !(tw > 0.0 && th > 0.0) || !side.is_finite() {
        return out;
    }
    let texel = |x: i64, y: i64| -> [f32; 4] {
        let (x, y) = (x.clamp(0, iw as i64 - 1) as usize, y.clamp(0, ih as i64 - 1) as usize);
        image.pixels.get(y * iw + x).map_or([0.0; 4], |c| c.to_array().map(f32::from))
    };
    let fade = |t: f64, n: f64| (t + 0.5).min(n - t + 0.5).clamp(0.0, 1.0) as f32;
    let straight = |p: [f32; 4]| -> [f32; 4] {
        let c = Color32::from_rgba_premultiplied(p[0].round() as u8, p[1].round() as u8, p[2].round() as u8, p[3].round() as u8);
        let [r, g, b, _] = c.to_srgba_unmultiplied();
        // The alpha keeps its fraction: a soft edge would band at 8 bits.
        [f32::from(r) / 255.0, f32::from(g) / 255.0, f32::from(b) / 255.0, (p[3] / 255.0).clamp(0.0, 1.0)]
    };
    let width = rect.width() as usize;
    rows(&mut out.px, width, rect.y0, |y, row| {
        let py = f64::from(y) + 0.5;
        for (i, o) in row.iter_mut().enumerate() {
            let px = f64::from(rect.x0) + i as f64 + 0.5;
            let w = m[6] * px + m[7] * py + m[8];
            if w * side <= 1e-12 {
                continue;
            }
            let (tx, ty) = ((m[0] * px + m[1] * py + m[2]) / w * tw, (m[3] * px + m[4] * py + m[5]) / w * th);
            if !(tx.is_finite() && ty.is_finite()) {
                continue;
            }
            if nearest {
                if tx >= 0.0 && ty >= 0.0 && tx < tw && ty < th {
                    *o = straight(texel(tx as i64, ty as i64));
                }
                continue;
            }
            let cover = fade(tx, tw) * fade(ty, th);
            if cover <= 0.0 {
                continue;
            }
            let (fx, fy) = (tx - 0.5, ty - 0.5);
            let (x0, y0) = (fx.floor(), fy.floor());
            let (ax, ay) = ((fx - x0) as f32, (fy - y0) as f32);
            let (x0, y0) = (x0 as i64, y0 as i64);
            let (a, b, c, d) = (texel(x0, y0), texel(x0 + 1, y0), texel(x0, y0 + 1), texel(x0 + 1, y0 + 1));
            let mut p = [0.0f32; 4];
            for (k, v) in p.iter_mut().enumerate() {
                let (top, bottom) = (a[k] + (b[k] - a[k]) * ax, c[k] + (d[k] - c[k]) * ax);
                *v = (top + (bottom - top) * ay) * cover;
            }
            if p[3] > 0.0 {
                *o = straight(p);
            }
        }
    });
    out
}

/// The texels a composite reads and how they map onto the box.
pub struct Source<'a> {
    /// The moving texels at some level of their mip chain, premultiplied.
    pub image: &'a ColorImage,
    /// The extent of `image` covering the box's source rectangle.
    pub uv: [f32; 2],
    /// The box (document pixels, clockwise from top-left).
    pub quad: [[f64; 2]; 4],
    /// Nearest Neighbor interpolation.
    pub nearest: bool,
}

/// Composite the moving texels with `work`'s style over `plan.shown`, on transparency. `work`
/// and `plan` come from [`plan`]; `doc` gives the global light and the patterns.
pub fn compose(doc: &Document, work: &Layer, plan: &Plan, source: &Source) -> Option<Buffer> {
    let s = plan.scale;
    let to_quad = Homography::rect_to_quad([0.0, 0.0, 1.0, 1.0], source.quad.map(|q| [q[0] * s, q[1] * s]))?;
    let to_unit = to_quad.inverse()?;
    let inside = source.quad.iter().fold([0.0; 2], |c, q| [c[0] + q[0] * s / 4.0, c[1] + q[1] * s / 4.0]);
    let content = resample(source.image, source.uv, &to_unit, inside, plan.content, source.nearest);
    let shape: Vec<f32> = content.px.iter().map(|p| p[3]).collect();
    let anchor = work.effects.reference.unwrap_or((f64::from(plan.bounds.x0), f64::from(plan.bounds.y0)));
    let maps = effects::build_maps(work, shape, plan.content, &doc.global_light, &TextureCtx { rect: plan.content, patterns: &doc.patterns, anchor });
    // The maps hold what every pixel needs of its surroundings, so the style is laid over the
    // content band by band, in parallel, as the compositor does tile by tile.
    let shown = plan.shown;
    let width = shown.width() as usize;
    let mut out = Buffer::transparent(shown);
    bands(&mut out.px, width * BAND_ROWS, |i, px| {
        let y0 = shown.y0.saturating_add((i * BAND_ROWS) as i32);
        let band = Rect::new(shown.x0, y0, shown.x1, y0.saturating_add((px.len() / width.max(1)) as i32));
        let mut over = Buffer::transparent(band);
        effects::composite_with_effects(work, &crop(&content, band), &mut over, &maps, plan.bounds, &doc.patterns);
        for (o, p) in px.iter_mut().zip(over.px) {
            *o = p;
        }
    });
    Some(out)
}

/// Rows composited per parallel band.
const BAND_ROWS: usize = 64;

/// `rect` of `buf` (transparent where `buf` doesn't reach).
fn crop(buf: &Buffer, rect: Rect) -> Buffer {
    let mut out = Buffer::transparent(rect);
    let part = rect.intersect(&buf.rect);
    if part.is_empty() {
        return out;
    }
    let (sw, dw, n) = (buf.rect.width() as usize, rect.width() as usize, part.width() as usize);
    for y in part.y0..part.y1 {
        let s0 = (y - buf.rect.y0) as usize * sw + (part.x0 - buf.rect.x0) as usize;
        let d0 = (y - rect.y0) as usize * dw + (part.x0 - rect.x0) as usize;
        if let (Some(src), Some(dst)) = (buf.px.get(s0..s0 + n), out.px.get_mut(d0..d0 + n)) {
            dst.copy_from_slice(src);
        }
    }
    out
}

/// `out` in chunks of `len`, each with its index, in parallel on native targets.
fn bands<T: Send>(out: &mut [T], len: usize, band: impl Fn(usize, &mut [T]) + Sync) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(len.max(1)).enumerate().for_each(|(i, o)| band(i, o));
    }
    #[cfg(target_arch = "wasm32")]
    for (i, o) in out.chunks_mut(len.max(1)).enumerate() {
        band(i, o);
    }
}

/// `buf` as premultiplied texels.
fn image(buf: &Buffer) -> ColorImage {
    let (w, h) = (buf.rect.width() as usize, buf.rect.height() as usize);
    let mut px = vec![Color32::TRANSPARENT; w * h];
    rows(&mut px, w, 0, |y, row| {
        let src = buf.px.get(y as usize * w..).unwrap_or(&[]);
        for (o, p) in row.iter_mut().zip(src) {
            let v = p.map(|c| (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            *o = Color32::from_rgba_unmultiplied(v[0], v[1], v[2], v[3]);
        }
    });
    ColorImage::new([w, h], px)
}

/// What a styled preview is drawn from this frame.
pub struct Frame<'a> {
    /// The document: its size, global light and patterns.
    pub doc: &'a Document,
    /// The moving layer as it is now: its style, opacity, fill and Advanced Blending.
    pub layer: &'a Layer,
    /// Changes whenever `doc` does.
    pub revision: u64,
    pub texels: &'a PreviewTextures,
    pub quad: [[f64; 2]; 4],
    /// How much the box enlarges its source (1 = same size).
    pub quad_scale: f32,
    pub nearest: bool,
    /// Screen pixels per document pixel.
    pub zoom: f32,
    /// The pointer is down: composite within the smaller [`budget`].
    pub dragging: bool,
}

impl Frame<'_> {
    /// The plan for this frame, the layer it composites and the texel level it reads.
    pub fn plan(&self) -> Option<(Plan, Layer, u32)> {
        let (plan, work) = plan(self.layer, [self.doc.size.width, self.doc.size.height], self.quad, self.zoom, budget(self.layer, self.dragging))?;
        let level = PreviewTextures::choose(plan.scale as f32, self.quad_scale, self.nearest).0;
        Some((plan, work, level))
    }

    /// The composite a [`Frame::plan`] describes.
    pub fn compose(&self, plan: &Plan, work: &Layer, level: u32) -> Option<Buffer> {
        let (texels, uv): (Arc<ColorImage>, [f32; 2]) = self.texels.level_image(level)?;
        compose(self.doc, work, plan, &Source { image: &texels, uv, quad: self.quad, nearest: self.nearest })
    }
}

#[derive(Clone, PartialEq)]
struct Key {
    revision: u64,
    quad: [u64; 8],
    plan: Plan,
    level: u32,
    nearest: bool,
    crisp: bool,
}

struct Shown {
    key: Key,
    tex: TextureHandle,
    rect: [f32; 4],
}

#[derive(Default)]
struct State {
    shown: Option<Shown>,
    /// Composites made so far.
    builds: u64,
}

/// The styled composite of a Free Transform session's moving layer, kept until the box, the
/// zoom or the layer changes.
pub struct StyledPreview {
    name: String,
    state: Mutex<State>,
}

impl StyledPreview {
    /// `name` keys the texture in egui.
    pub fn new(name: String) -> Self {
        Self { name, state: Mutex::new(State::default()) }
    }

    /// How many times the composite was made: once per change of the box, the zoom or the layer.
    pub fn builds(&self) -> u64 {
        self.state.lock().unwrap_or_else(PoisonError::into_inner).builds
    }

    /// The composite for `frame` and the document rectangle it covers (x0, y0, x1, y1),
    /// compositing it again if anything it depends on changed. `None` when there is nothing to
    /// draw this way: the caller falls back to the plain texels.
    pub fn texture(&self, ctx: &egui::Context, frame: &Frame) -> Option<(TextureId, [f32; 4])> {
        let (plan, work, level) = frame.plan()?;
        let crisp = plan.scale >= 1.0 && frame.zoom >= CRISP_ZOOM;
        let q = frame.quad.map(|q| q.map(f64::to_bits));
        let key = Key {
            revision: frame.revision,
            quad: [q[0][0], q[0][1], q[1][0], q[1][1], q[2][0], q[2][1], q[3][0], q[3][1]],
            plan,
            level,
            nearest: frame.nearest,
            crisp,
        };
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(s) = state.shown.as_ref().filter(|s| s.key == key) {
            return Some((s.tex.id(), s.rect));
        }
        let out = frame.compose(&plan, &work, level)?;
        let options = if crisp { TextureOptions::NEAREST } else { TextureOptions::LINEAR };
        let tex = ctx.load_texture(self.name.clone(), image(&out), options);
        let (id, rect) = (tex.id(), plan.doc_rect());
        state.shown = Some(Shown { key, tex, rect });
        state.builds += 1;
        Some((id, rect))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::Color;
    use photocraft_doc::{Contour, FxCommon, FxPaint, Shadow, StrokeFx, StrokePosition};

    const BLUE: Color32 = Color32::from_rgb(0, 0, 255);

    fn stroke(size: f32) -> Effect {
        Effect::Stroke(StrokeFx {
            common: FxCommon::new(BlendMode::Normal, 1.0),
            size,
            position: StrokePosition::Outside,
            paint: FxPaint::Color(Color::rgb(1.0, 0.0, 0.0)),
        })
    }

    fn shadow(distance: f32) -> Effect {
        Effect::DropShadow(Shadow {
            common: FxCommon::new(BlendMode::Normal, 1.0),
            color: Color::rgb(0.0, 1.0, 0.0),
            // Lit from the top-left: the shadow falls to the bottom-right.
            angle: 135.0,
            use_global_light: false,
            distance,
            spread: 1.0,
            size: 0.0,
            contour: Contour::Linear,
            anti_alias: false,
            noise: 0.0,
            knocks_out: true,
        })
    }

    fn styled(effects: Vec<Effect>) -> Layer {
        let mut l = Layer::raster("styled", PixelFormat::RGBA8);
        l.effects.items = effects;
        l
    }

    fn doc() -> Document {
        Document::new("t", photocraft_geom::Size::new(200, 200), photocraft_color::ColorMode::Rgb, photocraft_color::SampleType::U8)
    }

    fn rect_quad(r: [f64; 4]) -> [[f64; 2]; 4] {
        [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]]
    }

    /// The composite of a blue `w`×`h` block of texels moved onto `quad`, and its plan.
    fn composite(layer: &Layer, w: usize, h: usize, quad: [[f64; 2]; 4], zoom: f32) -> (Buffer, Plan) {
        let texels = ColorImage::new([w, h], vec![BLUE; w * h]);
        let (plan, work) = plan(layer, [200, 200], quad, zoom, budget(layer, false)).expect("on the canvas");
        let out = compose(&doc(), &work, &plan, &Source { image: &texels, uv: [1.0, 1.0], quad, nearest: false }).expect("composited");
        (out, plan)
    }

    /// The composite at a document point.
    fn at(out: &Buffer, plan: &Plan, x: f64, y: f64) -> [f32; 4] {
        let (x, y) = ((x * plan.scale).floor() as i32, (y * plan.scale).floor() as i32);
        if out.rect.contains(x, y) { out.get(x, y) } else { [0.0; 4] }
    }

    fn is(p: [f32; 4], rgb: [f32; 3]) -> bool {
        p[3] > 0.99 && p.iter().zip(rgb).all(|(a, b)| (a - b).abs() < 0.02)
    }

    const RED: [f32; 3] = [1.0, 0.0, 0.0];
    const GREEN: [f32; 3] = [0.0, 1.0, 0.0];
    const FILL: [f32; 3] = [0.0, 0.0, 1.0];

    #[test]
    fn the_style_is_drawn_around_the_moved_pixels() {
        // Moved, not resized: the block at 60..100 × 80..100 with a 4 px stroke around it.
        let l = styled(vec![stroke(4.0)]);
        let (out, plan) = composite(&l, 40, 20, rect_quad([60.0, 80.0, 100.0, 100.0]), 1.0);
        assert_eq!(plan.scale, 1.0);
        assert!(is(at(&out, &plan, 80.0, 90.0), FILL), "the pixels themselves");
        assert!(is(at(&out, &plan, 60.0, 80.0), FILL) && is(at(&out, &plan, 99.0, 99.0), FILL), "exact at the corners: no resampling blur");
        for (x, y) in [(57.0, 90.0), (102.0, 90.0), (80.0, 77.0), (80.0, 102.0)] {
            assert!(is(at(&out, &plan, x, y), RED), "stroke at {x},{y}: {:?}", at(&out, &plan, x, y));
        }
        for (x, y) in [(54.0, 90.0), (105.0, 90.0), (80.0, 74.0), (80.0, 105.0)] {
            assert_eq!(at(&out, &plan, x, y)[3], 0.0, "nothing past the stroke at {x},{y}");
        }
    }

    #[test]
    fn a_stroke_keeps_its_width_when_the_box_is_enlarged() {
        // The 20×20 block enlarged four times: the stroke stays 4 px wide, not 16.
        let l = styled(vec![stroke(4.0)]);
        let (out, plan) = composite(&l, 20, 20, rect_quad([60.0, 60.0, 140.0, 140.0]), 1.0);
        assert!(is(at(&out, &plan, 100.0, 100.0), FILL));
        assert!(is(at(&out, &plan, 57.0, 100.0), RED) && is(at(&out, &plan, 142.0, 100.0), RED));
        assert_eq!(at(&out, &plan, 53.0, 100.0)[3], 0.0, "{:?}", at(&out, &plan, 53.0, 100.0));
        assert_eq!(at(&out, &plan, 146.0, 100.0)[3], 0.0);
    }

    #[test]
    fn a_shadow_keeps_its_direction_when_the_box_is_turned() {
        // Turned a half turn (the quad's corners swapped round): the shadow still falls to the
        // bottom-right of the block, not to the top-left with the pixels.
        let l = styled(vec![shadow(10.0)]);
        let upright = rect_quad([80.0, 80.0, 120.0, 120.0]);
        let turned = [upright[2], upright[3], upright[0], upright[1]];
        for quad in [upright, turned] {
            let (out, plan) = composite(&l, 40, 40, quad, 1.0);
            assert!(is(at(&out, &plan, 100.0, 100.0), FILL));
            assert!(is(at(&out, &plan, 124.0, 124.0), GREEN), "{:?}", at(&out, &plan, 124.0, 124.0));
            assert_eq!(at(&out, &plan, 76.0, 76.0)[3], 0.0, "no shadow on the lit side");
        }
    }

    #[test]
    fn a_zoomed_out_view_composites_at_screen_resolution() {
        // At 50 % the work is a quarter of the pixels and the 4 px stroke is 2 work pixels,
        // still 4 document pixels where it is drawn.
        let l = styled(vec![stroke(4.0)]);
        let (out, plan) = composite(&l, 40, 40, rect_quad([60.0, 60.0, 100.0, 100.0]), 0.5);
        assert_eq!(plan.scale, 0.5);
        assert_eq!(plan.bounds, Rect::new(30, 30, 50, 50));
        assert_eq!(out.rect, plan.shown);
        assert!(is(at(&out, &plan, 80.0, 80.0), FILL));
        assert!(is(at(&out, &plan, 57.0, 80.0), RED) && is(at(&out, &plan, 103.0, 80.0), RED));
        assert_eq!(at(&out, &plan, 53.0, 80.0)[3], 0.0);
        let r = plan.doc_rect();
        assert!(r[0] <= 56.0 && r[2] >= 104.0, "{r:?}");
        // Zoomed in, the work is never finer than the document.
        assert_eq!(composite(&l, 40, 40, rect_quad([60.0, 60.0, 100.0, 100.0]), 8.0).1.scale, 1.0);
    }

    #[test]
    fn the_part_past_the_canvas_is_left_out_but_still_shapes_the_style() {
        // The block hangs over the left edge: nothing is composited past the canvas, and no
        // stroke is drawn along the canvas edge, where the layer goes on.
        let l = styled(vec![stroke(4.0)]);
        let (out, plan) = composite(&l, 40, 20, rect_quad([-20.0, 80.0, 20.0, 100.0]), 1.0);
        assert_eq!(plan.shown.x0, 0);
        assert!(is(at(&out, &plan, 0.0, 90.0), FILL), "{:?}", at(&out, &plan, 0.0, 90.0));
        assert!(is(at(&out, &plan, 22.0, 90.0), RED));
        // Wholly off the canvas: nothing to composite.
        assert!(plan_of(&l, rect_quad([-90.0, 0.0, -50.0, 20.0])).is_none());
    }

    fn plan_of(l: &Layer, quad: [[f64; 2]; 4]) -> Option<Plan> {
        plan(l, [200, 200], quad, 1.0, budget(l, false)).map(|p| p.0)
    }

    #[test]
    fn opacity_and_fill_apply_as_in_the_compositor() {
        // Fill 0 %: the pixels go, the stroke stays. Opacity 50 %: all of it at half strength.
        let mut l = styled(vec![stroke(4.0)]);
        l.fill_opacity = 0.0;
        let (out, plan) = composite(&l, 40, 40, rect_quad([60.0, 60.0, 100.0, 100.0]), 1.0);
        assert_eq!(at(&out, &plan, 80.0, 80.0)[3], 0.0);
        assert!(is(at(&out, &plan, 57.0, 80.0), RED));
        l.fill_opacity = 1.0;
        l.opacity = 0.5;
        let (out, plan) = composite(&l, 40, 40, rect_quad([60.0, 60.0, 100.0, 100.0]), 1.0);
        assert!((at(&out, &plan, 80.0, 80.0)[3] - 0.5).abs() < 0.01 && (at(&out, &plan, 57.0, 80.0)[3] - 0.5).abs() < 0.01);
    }

    #[test]
    fn the_work_stays_within_the_budget() {
        // A box over a huge canvas at 100 %: the scale drops until the composite fits.
        let l = styled(vec![stroke(8.0), shadow(40.0)]);
        let quad = rect_quad([0.0, 0.0, 12_000.0, 8_000.0]);
        for budget in [budget(&l, true), budget(&l, false)] {
            let (p, work) = plan(&l, [12_000, 8_000], quad, 1.0, budget).expect("planned");
            let area = f64::from(p.content.width()) * f64::from(p.content.height());
            assert!(area <= budget && area > budget * 0.5, "{area} of {budget}");
            assert!(p.scale < 1.0);
            // The style shrank with it.
            let Some(Effect::Stroke(s)) = work.effects.items.first() else { panic!("stroke") };
            assert!((f64::from(s.size) - 8.0 * p.scale).abs() < 1e-3);
        }
    }

    #[test]
    fn hostile_boxes_are_refused_not_composited() {
        let l = styled(vec![stroke(4.0)]);
        let ok = rect_quad([60.0, 60.0, 100.0, 100.0]);
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut q = ok;
            q[2][0] = bad;
            assert!(plan_of(&l, q).is_none());
        }
        // Astronomically large or far away: planned within the budget or refused, never a huge
        // allocation or an overflow.
        for q in [rect_quad([-1e300, -1e300, 1e300, 1e300]), rect_quad([1e18, 1e18, 1e18 + 40.0, 1e18 + 40.0]), rect_quad([0.0, 0.0, 1e12, 1.0])] {
            if let Some(p) = plan_of(&l, q) {
                assert!(f64::from(p.content.width()) * f64::from(p.content.height()) <= budget(&l, false), "{p:?}");
            }
        }
        // A collapsed box, a zero or NaN zoom and an absurd budget.
        assert!(plan_of(&l, [[80.0, 80.0]; 4]).is_none());
        for zoom in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(plan(&l, [200, 200], ok, zoom, budget(&l, false)).map(|p| p.0.scale), Some(1.0));
        }
        for budget in [f64::NAN, f64::INFINITY, -5.0, 0.0] {
            if let Some((p, _)) = plan(&l, [200, 200], ok, 1.0, budget) {
                assert!(f64::from(p.content.width()) * f64::from(p.content.height()) <= MAX_PIXELS);
            }
        }
        // A degenerate quad that passes the plan (a bow tie collapsed onto a line) composites
        // to nothing instead of dividing by zero.
        let line = [[60.0, 60.0], [100.0, 100.0], [60.0, 60.0], [100.0, 100.0]];
        if let Some((p, work)) = plan(&l, [200, 200], line, 1.0, budget(&l, false)) {
            let texels = ColorImage::new([4, 4], vec![BLUE; 16]);
            assert!(compose(&doc(), &work, &p, &Source { image: &texels, uv: [1.0, 1.0], quad: line, nearest: false }).is_none());
        }
    }

    /// The style is laid on in parallel bands: the result is the one a single pass gives.
    #[test]
    fn bands_join_without_a_seam() {
        let l = styled(vec![stroke(5.0), shadow(12.0)]);
        let quad = [[40.0, 20.0], [150.0, 40.0], [130.0, 180.0], [20.0, 150.0]];
        let texels = ColorImage::new([30, 40], (0..1200).map(|i| if i % 7 == 0 { Color32::TRANSPARENT } else { BLUE }).collect());
        let (plan, work) = plan(&l, [200, 200], quad, 1.0, budget(&l, false)).expect("planned");
        assert!(plan.shown.height() as usize > 2 * BAND_ROWS, "{:?}", plan.shown);
        let source = Source { image: &texels, uv: [1.0, 1.0], quad, nearest: false };
        let banded = compose(&doc(), &work, &plan, &source).expect("composited");
        // The same in one pass.
        let to_unit = Homography::rect_to_quad([0.0, 0.0, 1.0, 1.0], quad).and_then(|h| h.inverse()).expect("invertible");
        let content = resample(&texels, [1.0, 1.0], &to_unit, [85.0, 97.5], plan.content, false);
        let shape = content.px.iter().map(|p| p[3]).collect();
        let anchor = (f64::from(plan.bounds.x0), f64::from(plan.bounds.y0));
        let maps = effects::build_maps(&work, shape, plan.content, &doc().global_light, &TextureCtx { rect: plan.content, patterns: &[], anchor });
        let mut whole = Buffer::transparent(plan.shown);
        effects::composite_with_effects(&work, &crop(&content, plan.shown), &mut whole, &maps, plan.bounds, &[]);
        assert_eq!(banded.rect, whole.rect);
        assert!(banded.px == whole.px, "banded and whole composites differ");
        assert!(whole.px.iter().any(|p| is(*p, RED)) && whole.px.iter().any(|p| is(*p, GREEN)) && whole.px.iter().any(|p| is(*p, FILL)));
    }

    #[test]
    fn the_budget_follows_what_the_style_costs() {
        // A drag gets a quarter of the time a resting box does, and a Stroke (two distance
        // fields) far fewer pixels than a shadow.
        let (stroked, shadowed) = (styled(vec![stroke(4.0)]), styled(vec![shadow(10.0)]));
        assert!(budget(&stroked, true) < budget(&stroked, false));
        assert!((budget(&shadowed, false) / budget(&shadowed, true) - REST_MS / DRAG_MS).abs() < 1e-9);
        assert!(budget(&stroked, true) * 3.0 < budget(&shadowed, true), "{} vs {}", budget(&stroked, true), budget(&shadowed, true));
        // Two strokes share their distance fields; an effect switched off costs nothing.
        assert!(pixel_cost_ns(&styled(vec![stroke(4.0), stroke(9.0)])) < pixel_cost_ns(&stroked) + 2.0);
        let mut off = styled(vec![stroke(4.0), shadow(10.0)]);
        if let Some(Effect::Stroke(s)) = off.effects.items.first_mut() {
            s.common.enabled = false;
        }
        assert_eq!(pixel_cost_ns(&off), pixel_cost_ns(&shadowed));
        // However heavy or light the style, the budget stays within its limits.
        let heavy = styled((0..200).map(|_| shadow(10.0)).collect());
        assert_eq!(budget(&heavy, true), MIN_PIXELS);
        assert_eq!(budget(&styled(Vec::new()), false), MAX_PIXELS);
    }

    #[test]
    fn resampling_handles_empty_texels_and_nearest_neighbor() {
        let unit = Homography::rect_to_quad([0.0, 0.0, 1.0, 1.0], rect_quad([0.0, 0.0, 8.0, 8.0])).and_then(|h| h.inverse()).expect("invertible");
        let r = Rect::new(-2, -2, 10, 10);
        let empty = ColorImage::new([0, 0], Vec::new());
        assert!(resample(&empty, [1.0, 1.0], &unit, [4.0, 4.0], r, false).px.iter().all(|p| p[3] == 0.0));
        // A 2×2 checker enlarged four times: Nearest Neighbor keeps hard texels, bilinear blends.
        let checker = ColorImage::new([2, 2], vec![Color32::WHITE, Color32::BLACK, Color32::BLACK, Color32::WHITE]);
        let hard = resample(&checker, [1.0, 1.0], &unit, [4.0, 4.0], r, true);
        assert_eq!(hard.get(3, 0), [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(hard.get(4, 0), [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(hard.get(-1, 0)[3], 0.0);
        let soft = resample(&checker, [1.0, 1.0], &unit, [4.0, 4.0], r, false);
        let mid = soft.get(3, 0);
        assert!(mid[0] > 0.2 && mid[0] < 0.8, "{mid:?}");
        // The uv extent limits what is read: half the texels, so only the left column shows.
        let half = resample(&checker, [0.5, 1.0], &unit, [4.0, 4.0], r, true);
        assert_eq!(half.get(7, 0), [1.0, 1.0, 1.0, 1.0]);
    }
}
