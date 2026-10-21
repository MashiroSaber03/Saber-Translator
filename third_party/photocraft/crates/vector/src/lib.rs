//! # photocraft-vector
//!
//! Rasterization of vector data from `photocraft-doc`: paths (with Photoshop path operations),
//! shape layers (fill + stroke), vector masks, and the inverse direction (tracing a coverage
//! mask back into a path for "Make Work Path").
//!
//! * [`edit`]: point-level path editing (Direct Selection, Convert Point) and hit testing.
//! * [`flatten`]: cubic Bézier flattening with a distance tolerance.
//! * [`map`]: non-affine maps of paths (Distort, Perspective, Warp).
//! * [`raster`]: exact area-coverage scanline rasterizer (non-zero / even-odd per component,
//!   boolean combination of components, inversion).
//! * [`stroke`]: polygon stroking with miter/round/bevel joins, butt/round/square caps, dashes.
//! * [`shapes`]: live shapes (rectangles with corner radii, ellipses, polygons, stars, lines).
//! * [`trace`]: contour tracing + curve fitting (selection → path).
//!
//! Output is either raw coverage (`f32` per pixel) or sparse tiled [`Surface`]s in any
//! [`PixelFormat`] (only tiles that differ from the default are allocated). Pure Rust, no
//! platform dependencies; builds for `wasm32` (single-threaded there, rayon elsewhere).
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod edit;
pub mod flatten;
pub mod map;
pub mod raster;
pub mod shapes;
pub mod stroke;
pub mod trace;

use photocraft_color::{ColorMode, PixelFormat};
use photocraft_doc::{Fill, FillRule, GradientStyle, Path, PathOp, ShapeLayer, ShapeStroke, StrokeAlign, VectorMask};
use photocraft_geom::{Rect, TILE_SIZE};
use photocraft_raster::Surface;

pub use flatten::{Polyline, flatten_path, flatten_subpath};
pub use raster::Rasterizer;
pub use stroke::{StrokeStyle, stroke_polygons};

/// Default flattening tolerance in pixels (curve-to-chord distance).
pub const DEFAULT_TOLERANCE: f64 = 0.01;

/// Compiles the fill area of `path`: one component per subpath (filled with the path's rule)
/// folded with the subpath operations; open subpaths are closed implicitly.
pub fn fill_rasterizer(path: &Path, tol: f64) -> Rasterizer {
    let mut r = Rasterizer::new(path.inverted);
    for c in path.components() {
        let polys: Vec<Vec<(f64, f64)>> = path.subpaths[c.clone()].iter().map(|s| flatten_subpath(s, tol).pts).collect();
        r.add_component(&polys, path.effective_op(c.start), path.fill_rule);
    }
    r
}

/// Compiles the stroke outline of `path` (all subpaths, one non-zero component).
pub fn stroke_rasterizer(path: &Path, style: &StrokeStyle, tol: f64) -> Rasterizer {
    // Stroke outlines tolerate a coarser flattening than fills (the offset hides chord error).
    let lines = flatten_path(path, (tol * 4.0).min(style.width.max(0.01) * 0.1).max(1e-3));
    let polys = stroke_polygons(&lines, style, tol);
    let mut r = Rasterizer::new(false);
    r.add_component(&polys, PathOp::Combine, FillRule::NonZero);
    r
}

/// Stroke geometry for a shape stroke (dash lengths are multiples of the width in the model).
pub fn stroke_style(s: &ShapeStroke) -> StrokeStyle {
    let w = f64::from(s.width.max(0.0));
    StrokeStyle {
        width: w,
        cap: s.cap,
        join: s.join,
        miter_limit: f64::from(s.miter_limit.max(1.0)),
        dashes: s.dashes.iter().map(|d| f64::from(*d) * w).collect(),
        dash_offset: f64::from(s.dash_offset) * w,
    }
}

/// Fill coverage of `path` over `rect` (row-major, `0..=1`).
pub fn path_coverage(path: &Path, rect: Rect) -> Vec<f32> {
    fill_rasterizer(path, DEFAULT_TOLERANCE).render(rect)
}

enum MaskGeometry {
    Constant(f32),
    Path(Rasterizer),
}

/// A vector mask's immutable geometry and density, reusable across render rectangles.
pub struct CompiledVectorMask {
    geometry: MaskGeometry,
    density: Option<f32>,
}

impl CompiledVectorMask {
    /// Compiles only enabled, nonempty paths using the default flattening tolerance.
    pub fn new(m: &VectorMask) -> Self {
        let geometry = if !m.enabled {
            MaskGeometry::Constant(1.0)
        } else if m.path.is_empty() {
            // An empty Photoshop vector mask reveals all, unlike an empty fill path.
            MaskGeometry::Constant(if m.path.inverted { 0.0 } else { 1.0 })
        } else {
            MaskGeometry::Path(fill_rasterizer(&m.path, DEFAULT_TOLERANCE))
        };
        Self { geometry, density: (m.enabled && m.density < 1.0).then(|| m.density.clamp(0.0, 1.0)) }
    }

    /// Effective mask values over `rect`, with invocation-local rasterization state.
    pub fn render(&self, rect: Rect) -> Vec<f32> {
        let mut v = match &self.geometry {
            MaskGeometry::Constant(value) => vec![*value; rect.width() as usize * rect.height() as usize],
            MaskGeometry::Path(rasterizer) => rasterizer.render(rect),
        };
        self.apply_density(&mut v);
        v
    }

    /// Effective mask values without entering a thread pool. Use this while
    /// initializing a shared result that other Rayon workers may wait for.
    pub fn render_sequential(&self, rect: Rect) -> Vec<f32> {
        let mut v = match &self.geometry {
            MaskGeometry::Constant(value) => vec![*value; rect.width() as usize * rect.height() as usize],
            MaskGeometry::Path(rasterizer) => {
                let mut v = vec![0.0; rect.width() as usize * rect.height() as usize];
                if !v.is_empty() {
                    rasterizer.render_into(rect, &mut v);
                }
                v
            }
        };
        self.apply_density(&mut v);
        v
    }

    fn apply_density(&self, v: &mut [f32]) {
        if let Some(d) = self.density {
            for x in v {
                *x = 1.0 - d * (1.0 - *x);
            }
        }
    }
}

/// Effective vector-mask values over `rect` (density applied; all ones when disabled). Like
/// Photoshop, a vector mask without any subpath reveals everything (hides everything when
/// inverted): "Add Vector Mask" starts from an empty, revealing mask.
pub fn vector_mask_values(m: &VectorMask, rect: Rect) -> Vec<f32> {
    CompiledVectorMask::new(m).render(rect)
}

/// Tile-aligned bands (rows of tiles) covering `r`.
fn tile_bands(r: Rect) -> Vec<Rect> {
    let mut out = Vec::new();
    if r.is_empty() {
        return out;
    }
    let mut y = r.y0.div_euclid(TILE_SIZE) * TILE_SIZE;
    while y < r.y1 {
        let band = Rect::new(r.x0, y.max(r.y0), r.x1, (y + TILE_SIZE).min(r.y1));
        out.push(band);
        y += TILE_SIZE;
    }
    out
}

/// Splits a band buffer (`channels` floats per pixel) into tile-aligned pieces, skipping
/// pieces where `keep` is false for every pixel.
fn band_tiles(band: Rect, data: &[f32], channels: usize, keep: impl Fn(&[f32]) -> bool) -> Vec<(Rect, Vec<f32>)> {
    let w = band.width() as usize;
    let mut out = Vec::new();
    let mut x = band.x0.div_euclid(TILE_SIZE) * TILE_SIZE;
    while x < band.x1 {
        let tr = Rect::new(x.max(band.x0), band.y0, (x + TILE_SIZE).min(band.x1), band.y1);
        let tw = tr.width() as usize;
        let mut buf = Vec::with_capacity(tw * tr.height() as usize * channels);
        let mut any = false;
        for row in 0..tr.height() as usize {
            let o = (row * w + (tr.x0 - band.x0) as usize) * channels;
            let slice = &data[o..o + tw * channels];
            if !any && slice.chunks_exact(channels).any(&keep) {
                any = true;
            }
            buf.extend_from_slice(slice);
        }
        if any {
            out.push((tr, buf));
        }
        x += TILE_SIZE;
    }
    out
}

/// Runs `f` over the bands of `r` (in parallel groups on native targets) and writes the
/// resulting pieces into `surface`.
fn render_bands(surface: &mut Surface, r: Rect, f: &(dyn Fn(Rect) -> Vec<(Rect, Vec<f32>)> + Sync)) {
    let bands = tile_bands(r);
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        let group = rayon::current_num_threads().max(1) * 2;
        for chunk in bands.chunks(group) {
            let parts: Vec<Vec<(Rect, Vec<f32>)>> = chunk.par_iter().map(|b| f(*b)).collect();
            for (tr, v) in parts.into_iter().flatten() {
                surface.write_region(tr, &v);
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    for b in bands {
        for (tr, v) in f(b) {
            surface.write_region(tr, &v);
        }
    }
}

/// Renders a rasterizer's coverage into a single-channel surface of `format` (grayscale
/// model, no alpha, e.g. a selection or mask) over `clip`. Outside `clip` the surface reads 0,
/// or 1 for inverted paths whose coverage is written everywhere inside `clip`.
pub fn coverage_surface(r: &Rasterizer, format: PixelFormat, clip: Rect) -> Surface {
    let fmt = PixelFormat::new(ColorMode::Grayscale, format.sample, false);
    let mut s = Surface::new(fmt);
    let area = if r.is_inverted() { clip } else { r.pixel_bounds().map_or(Rect::EMPTY, |b| b.intersect(&clip)) };
    render_bands(&mut s, area, &|band| {
        let cov = r.render(band);
        band_tiles(band, &cov, 1, |p| p[0] > 0.0)
    });
    s
}

/// Colour source for fills and strokes.
#[derive(Clone, Debug)]
pub enum Paint {
    None,
    /// Straight RGBA.
    Solid([f32; 4]),
    Gradient {
        /// Colour stops sorted by location (straight RGBA).
        stops: Vec<(f32, [f32; 4])>,
        /// Colour midpoint per segment between consecutive sorted stops (missing = 0.5).
        midpoints: Vec<f32>,
        /// Opacity stops `(location, opacity)` sorted by location; empty = each stop's alpha.
        opacity_stops: Vec<(f32, f32)>,
        style: GradientStyle,
        /// Effective `(angle, scale, offset)` after whole-pixel snapping ([`gradient_layout`]).
        layout: (f32, f32, (f32, f32)),
        reverse: bool,
        dither: bool,
        /// Layout frame (`x0, y0, x1, y1`): the shape's bounds, or the canvas when the fill
        /// does not align with the layer.
        frame: (f64, f64, f64, f64),
    },
}

impl Paint {
    /// Paint for a model fill, with gradients laid out over `bounds` (`x0, y0, x1, y1`) — the
    /// layer's frame — or over `canvas` when the fill is not aligned with the layer. Every
    /// gradient field is honoured (midpoints, opacity stops, centre offset, dither, unsorted
    /// stops), rendering the same pixels as the compositor's fill layers; the parity tests in
    /// `photocraft-compose` (which this crate cannot depend on) keep the two in step. Pattern
    /// fills are not rendered (transparent): they need the document's pattern table, which
    /// lives above this crate.
    pub fn from_fill(f: &Fill, bounds: (f64, f64, f64, f64), canvas: (f64, f64, f64, f64)) -> Paint {
        let rgba = |c: &photocraft_color::Color| {
            let v = c.to_rgb();
            [v[0], v[1], v[2], c.alpha]
        };
        match f {
            Fill::Solid(c) => Paint::Solid(rgba(c)),
            // No `..` here on purpose: a new gradient field must fail to compile until the
            // shape renderer honours it (the same sentinel the document/compositor pair uses).
            Fill::Gradient { stops, angle, scale, style, reverse, opacity_stops, midpoints, offset, dither, align } => {
                let frame = if *align { bounds } else { canvas };
                let mut stops: Vec<(f32, [f32; 4])> = stops.iter().map(|(p, c)| (*p, rgba(c))).collect();
                stops.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut opacity = opacity_stops.clone();
                opacity.sort_by(|a, b| a.0.total_cmp(&b.0));
                Paint::Gradient {
                    stops,
                    midpoints: midpoints.clone(),
                    opacity_stops: opacity,
                    style: *style,
                    layout: gradient_layout(*style, *angle, *scale, *offset, frame),
                    reverse: *reverse,
                    dither: *dither,
                    frame,
                }
            }
            Fill::Pattern { .. } => Paint::None,
        }
    }

    /// Straight RGBA at pixel centre `(x, y)`.
    pub fn at(&self, x: f64, y: f64) -> [f32; 4] {
        match self {
            Paint::None => [0.0; 4],
            Paint::Solid(c) => *c,
            Paint::Gradient { stops, midpoints, opacity_stops, style, layout, reverse, dither: dith, frame } => {
                let t = gradient_t(*style, layout.0, layout.1, *reverse, layout.2, *frame, x, y);
                let mut c = sample_stops(stops, midpoints, opacity_stops, t);
                if *dith {
                    dither(&mut c, x.floor() as i32, y.floor() as i32);
                }
                c
            }
        }
    }
}

/// `(chord, diagonal)` gradient lengths over a `w × h` frame at `angle` (radians): Linear and
/// Reflected gradients span the chord, the other styles the half-diagonal. Mirrors the
/// compositor's `effects::gradient_units` (fitted against psd-tools shape-fx2, layer_effects).
fn gradient_units(angle: f64, w: f64, h: f64) -> (f64, f64) {
    let (s, c) = angle.sin_cos();
    let len = ((c * w).powi(2) + (s * h).powi(2)).sqrt().max(1.0);
    let chord = (w / c.abs().max(1e-6)).min(h / s.abs().max(1e-6)).max(1.0);
    (chord, len)
}

/// Effective `(angle, scale, offset)` for a gradient laid out in `frame`, snapping Linear and
/// Reflected end points to whole pixels as Photoshop does: on small frames that changes the
/// effective angle. Mirrors the compositor's `fill_layout::gradient_layout`, which the GPU
/// compositor shares.
fn gradient_layout(style: GradientStyle, angle: f32, scale: f32, offset: (f32, f32), frame: (f64, f64, f64, f64)) -> (f32, f32, (f32, f32)) {
    let unchanged = (angle, scale, offset);
    if !matches!(style, GradientStyle::Linear | GradientStyle::Reflected)
        || !angle.is_finite()
        || !scale.is_finite()
        || !offset.0.is_finite()
        || !offset.1.is_finite()
    {
        return unchanged;
    }
    let w = (frame.2 - frame.0).max(1.0);
    let h = (frame.3 - frame.1).max(1.0);
    let (cx, cy) = (frame.0 + w / 2.0 + f64::from(offset.0) * w, frame.1 + h / 2.0 + f64::from(offset.1) * h);
    // Unscaled chord length along `a` (radians), as `gradient_t`.
    let chord_of = |a: f64| gradient_units(a, w, h).0;
    let a = f64::from(angle).to_radians();
    let half = chord_of(a) * f64::from(scale.max(1e-3)) / 2.0;
    let (s, c) = a.sin_cos();
    let (dx, dy) = (c * half, -s * half);
    // Truncate to the pixel grid (with a little slack for rounding error in exact cases).
    let snap = |v: f64| (v + 1e-4).floor();
    let end = (snap(cx + dx), snap(cy + dy));
    let (start, mid) = match style {
        GradientStyle::Reflected => ((cx, cy), (cx, cy)),
        _ => {
            let s0 = (snap(cx - dx), snap(cy - dy));
            (s0, ((s0.0 + end.0) / 2.0, (s0.1 + end.1) / 2.0))
        }
    };
    let v = (end.0 - start.0, end.1 - start.1);
    let len = v.0.hypot(v.1);
    if len < 0.5 {
        return unchanged;
    }
    let a2 = (-v.1).atan2(v.0);
    // Reflected spans half the chord from the centre; Linear the whole chord.
    let span = if style == GradientStyle::Reflected { 2.0 * len } else { len };
    let scale2 = span / chord_of(a2);
    let shift = (((mid.0 - cx) / w) as f32, ((mid.1 - cy) / h) as f32);
    (a2.to_degrees() as f32, scale2 as f32, (offset.0 + shift.0, offset.1 + shift.1))
}

/// Gradient parameter `t` (`0..=1`) at pixel centre `(x, y)` in `frame`: the centre moves by
/// `offset` (a fraction of the frame), Linear and Reflected gradients sample the pixel's
/// top-left corner (Photoshop's whole-pixel end points). Mirrors the compositor's
/// `effects::gradient_t` (which the parity tests keep in step); unlike the compositor, the
/// frame stays exact fractional shape bounds.
#[allow(clippy::too_many_arguments)]
fn gradient_t(style: GradientStyle, angle: f32, scale: f32, reverse: bool, offset: (f32, f32), frame: (f64, f64, f64, f64), x: f64, y: f64) -> f32 {
    let w = (frame.2 - frame.0).max(1.0);
    let h = (frame.3 - frame.1).max(1.0);
    let cx = frame.0 + w / 2.0 + f64::from(offset.0) * w;
    let cy = frame.1 + h / 2.0 + f64::from(offset.1) * h;
    let a = f64::from(angle).to_radians();
    let (s, c) = a.sin_cos();
    let (dx, dy) = (x - cx, y - cy);
    let along = dx * c - dy * s;
    let across = dx * s + dy * c;
    let (chord, len) = gradient_units(a, w, h);
    let (chord, len) = (chord * f64::from(scale.max(1e-3)), len * f64::from(scale.max(1e-3)));
    let corner = 0.5 * (c - s);
    let mut t = match style {
        GradientStyle::Linear => (along - corner) / chord + 0.5,
        GradientStyle::Reflected => ((along - corner) / (chord / 2.0)).abs(),
        GradientStyle::Radial => dx.hypot(dy) / (len / 2.0),
        GradientStyle::Diamond => (along.abs() + across.abs()) / (len / 2.0),
        // Clockwise sweep starting at the gradient angle.
        GradientStyle::Angle => ((a - (-dy).atan2(dx)) / std::f64::consts::TAU).rem_euclid(1.0),
    };
    t = t.clamp(0.0, 1.0);
    (if reverse { 1.0 - t } else { t }) as f32
}

/// Where a segment of relative position `u` lands once its midpoint is `m` (0.5 = unchanged):
/// the colour is half-way at `m`, linear on either side. Mirrors the compositor's
/// `gradient_fill::midpoint_remap`.
fn midpoint_remap(u: f32, m: f32) -> f32 {
    let m = m.clamp(0.05, 0.95);
    if (m - 0.5).abs() < 1e-6 {
        u
    } else if u <= m {
        0.5 * u / m
    } else {
        0.5 + 0.5 * (u - m) / (1.0 - m)
    }
}

/// Colour stops with midpoints at `t` (stops sorted by location).
fn sample_color(stops: &[(f32, [f32; 4])], mids: &[f32], t: f32) -> [f32; 4] {
    let (Some(first), Some(last)) = (stops.first(), stops.last()) else { return [0.0; 4] };
    if t <= first.0 {
        return first.1;
    }
    for (i, w) in stops.windows(2).enumerate() {
        let (a, b) = (&w[0], &w[1]);
        if t <= b.0 {
            let u = if b.0 > a.0 { (t - a.0) / (b.0 - a.0) } else { 0.0 };
            let k = midpoint_remap(u, mids.get(i).copied().unwrap_or(0.5));
            return std::array::from_fn(|ch| a.1[ch] + (b.1[ch] - a.1[ch]) * k);
        }
    }
    last.1
}

/// Opacity stops at `t` (stops sorted by location; 1.0 when empty).
fn sample_opacity(stops: &[(f32, f32)], t: f32) -> f32 {
    let (Some(first), Some(last)) = (stops.first(), stops.last()) else { return 1.0 };
    if t <= first.0 {
        return first.1;
    }
    for w in stops.windows(2) {
        if t <= w[1].0 {
            let u = if w[1].0 > w[0].0 { (t - w[0].0) / (w[1].0 - w[0].0) } else { 0.0 };
            return w[0].1 + (w[1].1 - w[0].1) * u;
        }
    }
    last.1
}

/// Colour and alpha at `t`: colour stops with midpoints, alpha times opacity stops. Mirrors
/// the compositor's `gradient_fill::Ramp::sample`.
fn sample_stops(stops: &[(f32, [f32; 4])], mids: &[f32], opacity: &[(f32, f32)], t: f32) -> [f32; 4] {
    let mut c = sample_color(stops, mids, t);
    if !opacity.is_empty() {
        c[3] *= sample_opacity(opacity, t);
    }
    c
}

/// Adds one level of the shared dither noise to a colour's channels (as the Gradient tool and
/// the compositor's fill layers do).
fn dither(c: &mut [f32; 4], x: i32, y: i32) {
    let n = (photocraft_color::dither_noise(x, y) - 0.5) / 255.0;
    for ch in c.iter_mut().take(3) {
        *ch = (*ch + n).clamp(0.0, 1.0);
    }
}

/// Everything needed to render a shape layer, compiled once.
pub struct CompiledShape {
    fill: Option<(Rasterizer, Paint)>,
    stroke: Option<(Rasterizer, Paint, f32)>,
    /// Fill area used to clip inside/outside strokes.
    align: Option<(Rasterizer, bool)>,
    bounds: Option<Rect>,
}

impl CompiledShape {
    /// Compiles the shape at flattening tolerance `tol`. `canvas` is the frame gradients
    /// without "Align with layer" are laid out in.
    pub fn new(shape: &ShapeLayer, tol: f64, canvas: Rect) -> Self {
        let pb = shape.path.control_bounds().unwrap_or((0.0, 0.0, 0.0, 0.0));
        let cb = (f64::from(canvas.x0), f64::from(canvas.y0), f64::from(canvas.x1), f64::from(canvas.y1));
        let area = fill_rasterizer(&shape.path, tol);
        let mut bounds = None;
        let fill = shape.fill.as_ref().map(|f| {
            bounds = area.pixel_bounds();
            (area.clone(), Paint::from_fill(f, pb, cb))
        });
        let all_closed = shape.path.subpaths.iter().all(|s| s.closed);
        let mut align = None;
        let stroke = shape.stroke.as_ref().filter(|s| s.width > 0.0 && s.opacity > 0.0).map(|s| {
            let mut st = stroke_style(s);
            if all_closed && s.align != StrokeAlign::Center {
                // Inside/outside strokes: a centred stroke twice as wide, clipped by the fill area.
                st.width *= 2.0;
                align = Some((area.clone(), s.align == StrokeAlign::Inside));
            }
            let r = stroke_rasterizer(&shape.path, &st, tol);
            let sb = r.pixel_bounds();
            bounds = match (bounds, sb) {
                (Some(a), Some(b)) => Some(a.union(&b)),
                (a, b) => a.or(b),
            };
            (r, Paint::from_fill(&s.paint, pb, cb), s.opacity.clamp(0.0, 1.0))
        });
        if shape.path.inverted && shape.fill.is_some() {
            bounds = Some(Rect::new(i32::MIN / 4, i32::MIN / 4, i32::MAX / 4, i32::MAX / 4));
        }
        CompiledShape { fill, stroke, align, bounds }
    }

    /// Pixel bounds of the rendering (`None` = nothing visible).
    pub fn bounds(&self) -> Option<Rect> {
        self.bounds
    }

    /// Straight RGBA for every pixel of `rect`.
    pub fn render_rgba(&self, rect: Rect) -> Vec<[f32; 4]> {
        let n = rect.width() as usize * rect.height() as usize;
        let mut out = vec![[0.0f32; 4]; n];
        let w = rect.width() as usize;
        if let Some((r, paint)) = &self.fill {
            let cov = r.render(rect);
            for (i, (o, c)) in out.iter_mut().zip(&cov).enumerate() {
                if *c > 0.0 {
                    let (x, y) = ((rect.x0 + (i % w) as i32) as f64 + 0.5, (rect.y0 + (i / w) as i32) as f64 + 0.5);
                    let mut p = paint.at(x, y);
                    p[3] *= c;
                    *o = p;
                }
            }
        }
        if let Some((r, paint, opacity)) = &self.stroke {
            let mut cov = r.render(rect);
            if let Some((area, inside)) = &self.align {
                let a = area.render(rect);
                for (c, f) in cov.iter_mut().zip(&a) {
                    *c *= if *inside { *f } else { 1.0 - *f };
                }
            }
            for (i, (o, c)) in out.iter_mut().zip(&cov).enumerate() {
                if *c <= 0.0 {
                    continue;
                }
                let (x, y) = ((rect.x0 + (i % w) as i32) as f64 + 0.5, (rect.y0 + (i / w) as i32) as f64 + 0.5);
                let s = paint.at(x, y);
                let sa = s[3] * c * opacity;
                let da = o[3];
                let a = sa + da * (1.0 - sa);
                if a > 0.0 {
                    for k in 0..3 {
                        o[k] = (s[k] * sa + o[k] * da * (1.0 - sa)) / a;
                    }
                }
                o[3] = a;
            }
        }
        out
    }

    /// Renders into a new surface of `format` over `clip` (typically the canvas: it bounds the
    /// render and anchors gradients whose fill is not aligned with the layer).
    pub fn render(&self, format: PixelFormat, clip: Rect) -> Surface {
        let mut s = Surface::new(format);
        let Some(b) = self.bounds else { return s };
        let area = b.intersect(&clip);
        let ch = format.channels();
        render_bands(&mut s, area, &|band| {
            let rgba = self.render_rgba(band);
            let mut vals = vec![0.0f32; rgba.len() * ch];
            for (p, o) in rgba.iter().zip(vals.chunks_exact_mut(ch)) {
                if p[3] > 0.0 {
                    photocraft_raster::from_rgba_into(&format, *p, o);
                }
            }
            let alpha = format.alpha;
            band_tiles(band, &vals, ch, |px| if alpha { px[ch - 1] > 0.0 } else { px.iter().any(|v| *v != 0.0) })
        });
        s
    }
}

/// Renders a shape layer's appearance (fill, then stroke over it) in `format`, clipped to
/// `clip` — the canvas, which also lays out gradients whose fill is not aligned with the layer.
pub fn render_shape(shape: &ShapeLayer, format: PixelFormat, clip: Rect) -> Surface {
    CompiledShape::new(shape, DEFAULT_TOLERANCE, clip).render(format, clip)
}

#[cfg(test)]
mod tests;
