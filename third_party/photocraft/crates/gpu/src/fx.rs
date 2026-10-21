//! Effect maps on the GPU (`photocraft_compose::effects::build_maps`).
//!
//! Every enabled effect of a layer becomes a small *map program*: a chain of single-channel
//! fragment passes over the layer's effect region (shift, dilate, separable Gaussian blur, glow
//! ramp, bevel height and shading, contour, stroke band), each mirroring one step of the CPU
//! builders. Programs are pure functions of the effect settings and the global light.
//!
//! Inputs come from the CPU, computed by compose itself so they match it bit for bit: the
//! layer's **shape** (`compose::layer_shape`) and the **distance fields** of it that the effects
//! use (`compose::effects::distance_field`). Both are
//! computed per layer state and uploaded once, and only where the layer changed:
//!
//! - damage is the union of the layer's tiles whose copy-on-write `Arc` changed, so a brush dab
//!   recomputes one 256² tile of shape;
//! - a distance field is exact up to the distance its effects can use (beyond it every consumer
//!   saturates), so it is recomputed over the damage grown by that reach, from a window grown by
//!   the reach again; large windows split into bands computed in parallel, except the Euclidean
//!   fields (glows, chiselled bevels), which are one window that `distance_field` computes on all
//!   threads itself;
//! - each program re-runs only over the damage grown by its own reach, and only for effects
//!   whose settings changed; an unrelated edit (another layer, an adjustment, the effect's colour
//!   or opacity) reuses everything.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use photocraft_compose::effects::BevelPaint;
use photocraft_compose::effects::FieldKind;
use photocraft_doc::{BevelTechnique, Contour, Effect, GlobalLight, GlowSource, GlowTechnique, Layer, LayerContent, Pattern};
use photocraft_geom::{Rect, TileCoord};
use photocraft_raster::{Surface, Tile};

use crate::plan::{Kernel, stroke_widths};

/// Input of a map stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum In {
    /// The layer's alpha over the region.
    Shape,
    /// A distance field of the shape.
    Field(FieldKind),
    /// Output of stage `n`.
    Val(usize),
}

#[derive(Clone, Debug)]
pub(crate) struct Stage {
    pub kernel: Kernel,
    pub a: Option<In>,
    pub b: Option<In>,
    /// Bound as `layer_tex` (the shape for `× shape` steps).
    pub s: Option<In>,
    /// LUT row 0 (blur weights, contour, sine profile).
    pub lut: Option<Arc<Vec<f32>>>,
    pub p0: [f32; 4],
    pub p1: [f32; 4],
    /// Pattern placement (`p3`: origin, rotation) and `p4` (inverse scale, tile size) for stages
    /// that sample `pattern` (bevel texture).
    pub p3: [f32; 4],
    pub p4: [f32; 4],
    pub pattern: Option<Pattern>,
    /// How far (px) the stage reads from its inputs around the output pixel.
    pub radius: i32,
    /// Final map index this stage writes (else a temporary).
    pub out: Option<usize>,
}

/// The passes building one effect's maps.
#[derive(Clone, Debug)]
pub(crate) struct MapProgram {
    /// Identity of the program (settings that change the maps; not colours or opacities).
    pub key: u64,
    pub maps: usize,
    pub stages: Vec<Stage>,
    /// Distance fields read, with the distance up to which each must be exact.
    pub fields: Vec<(FieldKind, i32)>,
}

/// How far a field value depends on the shape, for a field exact up to `reach`.
pub(crate) fn field_radius(kind: FieldKind, reach: i32) -> i32 {
    // The nearest seed lies within the reach (seeds may start up to 1 px in, at their sub-pixel
    // edge); stroke fields classify seeds by their 3×3 local coverage, one more pixel.
    match kind {
        // A bevel field also looks at the edge pixels around the foot of the perpendicular (3 px
        // further); each of those takes its direction from the edge pixels within 3 px of it, and
        // those read their own 3×3 coverage neighbourhood (2 px). Measured, the smallest exact halo
        // is reach + 1 (`banded_bevel_fields_match_whole_region_on_anti_aliased_edges`).
        FieldKind::BevelInside | FieldKind::BevelOutside => reach + 9,
        _ => reach + 3,
    }
}

impl MapProgram {
    fn empty() -> Self {
        MapProgram { key: 0, maps: 0, stages: Vec::new(), fields: Vec::new() }
    }

    /// Per-stage output window for a shape damage rect `d` (all inside `region`): what each stage
    /// must recompute so every final map is exact over `d` grown by its reach. `field_reach`:
    /// the reach each field was computed with.
    pub fn windows(&self, d: Rect, region: Rect, field_reach: &dyn Fn(FieldKind) -> i32) -> Vec<Rect> {
        let n = self.stages.len();
        let reach = self.stage_reach(field_reach);
        let mut need = vec![Rect::EMPTY; n];
        for (i, s) in self.stages.iter().enumerate() {
            if s.out.is_some() {
                need[i] = d.inflate(reach[i]).intersect(&region);
            }
        }
        for i in (0..n).rev() {
            let w = need[i];
            if w.is_empty() {
                continue;
            }
            let s = &self.stages[i];
            for inp in [s.a, s.b, s.s].into_iter().flatten() {
                if let In::Val(k) = inp {
                    let grown = w.inflate(s.radius).intersect(&region);
                    need[k] = if need[k].is_empty() { grown } else { need[k].union(&grown) };
                }
            }
        }
        need
    }

    /// Per stage, how far (px) its output depends on the shape.
    fn stage_reach(&self, field_reach: &dyn Fn(FieldKind) -> i32) -> Vec<i32> {
        let mut reach: Vec<i32> = Vec::with_capacity(self.stages.len());
        for s in &self.stages {
            let of = |i: Option<In>| match i {
                Some(In::Val(k)) => reach.get(k).copied().unwrap_or(0),
                Some(In::Field(f)) => field_radius(f, field_reach(f)),
                _ => 0,
            };
            let r = of(s.a).max(of(s.b)).max(of(s.s)) + s.radius;
            reach.push(r);
        }
        reach
    }

    /// How far (px) any final map depends on the shape: maps computed over a window are exact
    /// at least this far inside it.
    pub fn reach(&self, field_reach: &dyn Fn(FieldKind) -> i32) -> i32 {
        self.stage_reach(field_reach).into_iter().zip(&self.stages).filter(|(_, s)| s.out.is_some()).map(|(r, _)| r).max().unwrap_or(0)
    }

    /// Physical temporary index per stage (None for final maps), reusing temporaries after
    /// their last read; returns (assignment, temporaries needed).
    pub fn temps(&self) -> (Vec<Option<usize>>, usize) {
        let n = self.stages.len();
        let mut last_use = vec![0usize; n];
        for (i, s) in self.stages.iter().enumerate() {
            for inp in [s.a, s.b, s.s].into_iter().flatten() {
                if let In::Val(k) = inp {
                    last_use[k] = last_use[k].max(i);
                }
            }
        }
        let mut free: Vec<usize> = Vec::new();
        let mut count = 0;
        let mut out = vec![None; n];
        let mut held: Vec<(usize, usize)> = Vec::new();
        for (i, (stage, slot)) in self.stages.iter().zip(out.iter_mut()).enumerate() {
            // Release temporaries whose last reader was before this stage.
            held.retain(|&(st, t)| {
                if last_use[st] < i {
                    free.push(t);
                    false
                } else {
                    true
                }
            });
            if stage.out.is_some() {
                continue;
            }
            let t = free.pop().unwrap_or_else(|| {
                count += 1;
                count - 1
            });
            *slot = Some(t);
            held.push((i, t));
        }
        (out, count)
    }
}

struct B {
    stages: Vec<Stage>,
    fields: Vec<(FieldKind, i32)>,
}

fn stage(kernel: Kernel, a: Option<In>, b: Option<In>, p0: [f32; 4], radius: i32) -> Stage {
    Stage { kernel, a, b, s: None, lut: None, p0, p1: [0.0; 4], p3: [0.0; 4], p4: [0.0; 4], pattern: None, radius, out: None }
}

impl B {
    fn push(&mut self, s: Stage) -> In {
        self.stages.push(s);
        In::Val(self.stages.len() - 1)
    }
    fn field(&mut self, kind: FieldKind, width: f32) -> In {
        self.fields.push((kind, reach_for(width)));
        In::Field(kind)
    }
    /// `Map::shifted` by whole pixels (`invert`: of `1 - a`); identity shifts of the plain map
    /// are skipped.
    fn shift(&mut self, src: In, dx: f32, dy: f32, outside: f32, invert: bool) -> In {
        if dx == 0.0 && dy == 0.0 && !invert {
            return src;
        }
        let r = dx.abs().max(dy.abs()) as i32;
        self.push(stage(Kernel::MShift, Some(src), None, [dx, dy, outside, f32::from(u8::from(invert))], r))
    }
    /// `dilate(src, r)` given `dist_outside(src)`.
    fn dilate(&mut self, src: In, dist: In, r: f32) -> In {
        self.push(stage(Kernel::MDilate, Some(src), Some(dist), [r, 0.0, 0.0, 0.0], 0))
    }
    fn blur(&mut self, src: In, size: f32) -> In {
        let Some(k) = blur_kernel(size) else { return src };
        self.conv(src, k)
    }
    /// Separable convolution with a symmetric kernel `(radius, weights)` (two `MBlur` passes).
    fn conv(&mut self, src: In, (r, w): (i32, Vec<f32>)) -> In {
        if r <= 0 {
            return src;
        }
        let w = Arc::new(w);
        let mut h = stage(Kernel::MBlur, Some(src), None, [0.0, r as f32, 0.0, 0.0], r);
        h.lut = Some(w.clone());
        let h = self.push(h);
        let mut v = stage(Kernel::MBlur, Some(h), None, [1.0, r as f32, 0.0, 0.0], r);
        v.lut = Some(w);
        self.push(v)
    }
    fn finish(&mut self, a: In, b: Option<In>, invert: bool, contour: &Contour, times_shape: bool, out: usize) {
        self.finish_lut(a, b, invert, contour_lut(contour), times_shape, out);
    }
    /// A glow's last step: its contour over the Range, then (centre-source inner glows) the
    /// inverse, then × the shape when inner.
    fn finish_glow(&mut self, a: In, lut: Option<Vec<f32>>, inner: bool, center: bool) {
        let mut s =
            stage(Kernel::MFinish, Some(a), None, [0.0, if center { 2.0 } else { 0.0 }, f32::from(u8::from(lut.is_some())), f32::from(u8::from(inner))], 0);
        s.lut = lut.map(Arc::new);
        if inner {
            s.s = Some(In::Shape);
        }
        s.out = Some(0);
        self.stages.push(s);
    }

    fn finish_lut(&mut self, a: In, b: Option<In>, invert: bool, lut: Option<Vec<f32>>, times_shape: bool, out: usize) {
        let flag = |v: bool| f32::from(u8::from(v));
        let mut s = stage(Kernel::MFinish, Some(a), b, [flag(b.is_some()), flag(invert), flag(lut.is_some()), flag(times_shape)], 0);
        s.lut = lut.map(Arc::new);
        if times_shape {
            s.s = Some(In::Shape);
        }
        s.out = Some(out);
        self.stages.push(s);
    }
}

/// Distance up to which a field must be exact for a band / dilation of width `w`
/// (`clamp(w + 0.5 - d)`, with `d` up to 1.5 px beyond the pixel distance).
fn reach_for(w: f32) -> i32 {
    w.clamp(0.0, photocraft_compose::effects::MAX_REACH).ceil() as i32 + 2
}

/// `compose::effects::blur`'s kernel (a tent of the effect size): (radius, normalised weights),
/// or None when it is the identity.
pub(crate) fn blur_kernel(size: f32) -> Option<(i32, Vec<f32>)> {
    if !size.is_finite() {
        return None;
    }
    let k = photocraft_compose::effects::tent_kernel(size);
    (k.0 > 0).then_some(k)
}

fn contour_lut(c: &Contour) -> Option<Vec<f32>> {
    match c {
        Contour::Linear => None,
        Contour::Custom { points, corners, .. } => Some(photocraft_compose::adjust::contour_curve_lut(points, corners)),
    }
}

/// `compose::effects::offset`: shadow offset away from the light, whole pixels.
fn offset(angle: f32, distance: f32) -> (f32, f32) {
    let a = angle.to_radians();
    ((-a.cos() * distance).round(), (a.sin() * distance).round())
}

/// The map program of one enabled effect (`build_maps`), without pattern sources.
#[cfg(test)]
pub(crate) fn program(e: &Effect, light: &GlobalLight, vector_shape: bool) -> MapProgram {
    program_with(e, light, vector_shape, &[], (0.0, 0.0))
}

/// The map program of one enabled effect (`build_maps`); bevel textures tile `patterns` from
/// `anchor` (`effects::TextureCtx`).
pub(crate) fn program_with(e: &Effect, light: &GlobalLight, vector_shape: bool, patterns: &[Pattern], anchor: (f64, f64)) -> MapProgram {
    let mut b = B { stages: Vec::new(), fields: Vec::new() };
    let maps = match e {
        Effect::DropShadow(s) | Effect::InnerShadow(s) => {
            // shadow_map
            let inner = matches!(e, Effect::InnerShadow(_));
            let angle = if s.use_global_light { light.angle } else { s.angle };
            let (dx, dy) = offset(angle, s.distance);
            let src = b.shift(In::Shape, dx, dy, if inner { 1.0 } else { 0.0 }, inner);
            let (r, bw) = photocraft_compose::effects::spread_split(s.size, s.spread);
            let mut m = src;
            if r > 0.0 {
                // dist_outside of the shifted map is the shifted field (integer offsets; shifted-in
                // pixels are far away for a drop shadow, inside for an inner shadow).
                let f = b.field(if inner { FieldKind::ChokeInside } else { FieldKind::StrokeOutside }, r);
                let d = if dx == 0.0 && dy == 0.0 {
                    f
                } else {
                    b.push(stage(Kernel::MShift, Some(f), None, [dx, dy, if inner { -0.5 } else { 1e10 }, 0.0], dx.abs().max(dy.abs()) as i32))
                };
                m = b.dilate(src, d, r);
            }
            let m = b.blur(m, bw);
            b.finish(m, None, false, &s.contour, inner, 0);
            1
        }
        Effect::OuterGlow(g) | Effect::InnerGlow(g) => {
            // glow_map
            let inner = matches!(e, Effect::InnerGlow(_));
            let center = inner && g.source == GlowSource::Center;
            match g.technique {
                GlowTechnique::Precise => {
                    let d = b.field(if inner { FieldKind::Inside } else { FieldKind::Outside }, g.size);
                    let solid = g.size * g.spread;
                    let soft = (g.size + 1.0 - solid).max(1e-3);
                    let m = b.push(stage(Kernel::MGlow, Some(d), None, [solid, soft, 0.0, 0.0], 0));
                    b.finish_glow(m, photocraft_compose::effects::glow_lut(g), inner, center);
                }
                GlowTechnique::Softer => {
                    // Inner glows (edge, and centre as 1 - the edge result) spread 1 - alpha.
                    let src = if inner { b.shift(In::Shape, 0.0, 0.0, 0.0, true) } else { In::Shape };
                    let (r, bw) = photocraft_compose::effects::spread_split(g.size, g.spread);
                    let mut m = src;
                    if r > 0.0 {
                        let d = b.field(if inner { FieldKind::ChokeInside } else { FieldKind::StrokeOutside }, r);
                        m = b.dilate(src, d, r);
                    }
                    let m = b.blur(m, bw);
                    b.finish_glow(m, photocraft_compose::effects::glow_lut(g), inner, center);
                }
            }
            1
        }
        Effect::Satin(s) => {
            // satin_map
            let (dx, dy) = offset(s.angle, photocraft_compose::effects::satin_distance(s));
            let a = b.shift(In::Shape, dx, dy, 0.0, false);
            let a = b.conv(a, photocraft_compose::effects::tent_kernel(s.size));
            let c = b.shift(In::Shape, -dx, -dy, 0.0, false);
            let c = b.conv(c, photocraft_compose::effects::tent_kernel(s.size));
            b.finish_lut(a, Some(c), s.invert, photocraft_compose::effects::satin_lut(&s.contour), true, 0);
            1
        }
        Effect::BevelEmboss(bv) => {
            // bevel_maps: height map (tent blur of the shape, or chiselled distance ramps), then
            // highlight / shadow shading inside and / or outside the shape.
            let g = photocraft_compose::effects::bevel_geom(bv);
            let size = bv.size.max(1.0);
            let paint = |p: BevelPaint| match p {
                BevelPaint::Outer => 0.0,
                BevelPaint::Both => 1.0,
                BevelPaint::Inner => 2.0,
            };
            let mut h = if bv.technique == BevelTechnique::Smooth {
                b.conv(In::Shape, photocraft_compose::effects::tent_kernel(g.width))
            } else {
                let din = b.field(FieldKind::BevelInside, size);
                let dout = b.field(FieldKind::BevelOutside, size);
                let soft = g.chisel_soft > 0.0;
                let h = b.push(stage(Kernel::MBevelH, Some(din), Some(dout), [paint(g.paint), size, f32::from(u8::from(soft)), 0.0], 0));
                // As `effects::bevel_height`: only emboss styles blur a Chisel Soft ramp.
                if soft && g.paint == BevelPaint::Both { b.conv(h, photocraft_compose::effects::tent_kernel(g.chisel_soft)) } else { h }
            };
            if let Some(c) = &bv.contour {
                // Contour element: the height through the contour (`bevel_contour_lut`).
                let lut = photocraft_compose::effects::bevel_contour_lut(c);
                let mut st = stage(Kernel::MFinish, Some(h), None, [0.0, 0.0, 1.0, 0.0], 0);
                st.lut = Some(Arc::new(lut));
                h = b.push(st);
            }
            if let Some(t) = &bv.texture
                && let Some(pat) = photocraft_doc::pattern::find(patterns, &t.id, &t.name).filter(|p| p.is_paintable())
            {
                // Texture element (`effects::bevel_height`): luminance × depth / unit added.
                let unit = if bv.depth.abs() > 1e-6 { (g.depth / bv.depth).abs().max(1e-3) } else { g.width.max(1.0) };
                let pl = photocraft_compose::pattern::Placement::anchored(anchor, t.link, t.phase, t.scale, 0.0);
                let (origin, cs, inv) = pl.parts();
                let mut st = stage(Kernel::MBevelTex, Some(h), None, [t.depth / unit, f32::from(u8::from(t.invert)), 0.0, 0.0], 0);
                st.p3 = [origin.0 as f32, origin.1 as f32, cs.0 as f32, cs.1 as f32];
                st.p4 = [inv as f32, pat.width as f32, pat.height as f32, 0.0];
                st.pattern = Some(pat.clone());
                h = b.push(st);
            }
            if bv.soften >= 1.0 {
                h = b.conv(h, photocraft_compose::effects::tent_kernel(bv.soften));
            }
            let (angle, altitude) = if bv.use_global_light { (light.angle, light.altitude) } else { (bv.angle, bv.altitude) };
            let (sa, ca) = angle.to_radians().sin_cos();
            let (se, ce) = altitude.to_radians().sin_cos();
            let lut = contour_lut(&bv.gloss_contour).map(Arc::new);
            // Region (`fs_mbevelshade`): 0 inside, 1 under the edge and outside, 2 / 3 emboss /
            // pillow emboss (both halves in one map).
            let passes: Vec<(f32, f32)> = match g.paint {
                BevelPaint::Inner => vec![(g.depth, 0.0)],
                BevelPaint::Outer => vec![(g.depth, 1.0)],
                BevelPaint::Both => vec![(g.depth, if g.pillow { 3.0 } else { 2.0 })],
            };
            let mut out = 0;
            for (depth, region) in &passes {
                for which in [0.0, 1.0] {
                    let mut s = stage(Kernel::MBevelShade, Some(h), None, [ca * ce, -sa * ce, se, se], 1);
                    s.p1 = [*depth, *region, which, f32::from(u8::from(lut.is_some()))];
                    s.p3 = [f32::from(u8::from(g.paint == BevelPaint::Outer && bv.technique != BevelTechnique::Smooth)), 0.0, 0.0, 0.0];
                    s.lut = lut.clone();
                    s.s = Some(In::Shape);
                    s.out = Some(out);
                    out += 1;
                    b.stages.push(s);
                }
            }
            out
        }
        Effect::Stroke(st) => {
            // composite_with_effects' stroke bands (maps 0 outside, 1 inside)
            let (in_w, out_w) = stroke_widths(st);
            if out_w > 0.0 {
                // Narrow strokes keep the fitted chamfer, wide ones the exact transform (#3646);
                // the CPU picks the same field by the same width (`stroke_outside_exact`).
                let kind = match (vector_shape, photocraft_compose::effects::stroke_outside_exact(out_w)) {
                    (true, true) => FieldKind::StrokeOutsideVector,
                    (true, false) => FieldKind::StrokeOutsideVectorChamfer,
                    (false, true) => FieldKind::StrokeOutsideExact,
                    (false, false) => FieldKind::StrokeOutside,
                };
                let d = b.field(kind, out_w);
                let mut s = stage(Kernel::MStroke, Some(d), None, [out_w, 0.0, 0.0, 0.0], 0);
                s.out = Some(0);
                b.stages.push(s);
            }
            if in_w > 0.0 {
                let d = b.field(FieldKind::StrokeInside, in_w);
                let mut s = stage(Kernel::MStroke, Some(d), None, [in_w, 0.0, 0.0, 0.0], 0);
                s.out = Some(1);
                b.stages.push(s);
            }
            2
        }
        Effect::ColorOverlay { .. } | Effect::GradientOverlay { .. } | Effect::PatternOverlay { .. } => return MapProgram::empty(),
    };
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for s in &b.stages {
        (format!("{:?}", s.kernel), s.a, s.b, s.s, s.radius, s.out).hash(&mut h);
        for v in s.p0.iter().chain(&s.p1).chain(&s.p3).chain(&s.p4) {
            v.to_bits().hash(&mut h);
        }
        if let Some(p) = &s.pattern {
            (&p.id, &p.name, p.width, p.height).hash(&mut h);
            for (c, t) in p.surface.tiles() {
                (c.tx, c.ty, Arc::as_ptr(t) as usize).hash(&mut h);
            }
        }
        if let Some(l) = &s.lut {
            for v in l.iter() {
                v.to_bits().hash(&mut h);
            }
        }
    }
    (maps, &b.fields).hash(&mut h);
    MapProgram { key: h.finish() | 1, maps, stages: b.stages, fields: b.fields }
}

// ---------------------------------------------------------------------------------------------
// CPU inputs: shape and distance fields

/// Row-major crop of the region-sized `shape` to `r` (inside `region`).
fn crop(shape: &[f32], region: Rect, r: Rect) -> Vec<f32> {
    let sw = region.width() as usize;
    let mut out = Vec::with_capacity(r.width() as usize * r.height() as usize);
    for y in r.y0..r.y1 {
        let o = (y - region.y0) as usize * sw + (r.x0 - region.x0) as usize;
        out.extend_from_slice(&shape[o..o + r.width() as usize]);
    }
    out
}

/// Rows `[y0, y1)` of `r` split into bands of about `rows`.
fn bands(r: Rect, rows: i32) -> Vec<Rect> {
    let mut out = Vec::new();
    let mut y = r.y0;
    while y < r.y1 {
        out.push(Rect::new(r.x0, y, r.x1, (y + rows).min(r.y1)));
        y += rows;
    }
    out
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn par_map<T: Send, R: Send>(items: Vec<T>, f: impl Fn(T) -> R + Sync + Send) -> Vec<R> {
    use rayon::prelude::*;
    items.into_par_iter().map(f).collect()
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn par_map<T, R>(items: Vec<T>, f: impl Fn(T) -> R) -> Vec<R> {
    items.into_iter().map(f).collect()
}

/// Band height for parallel field / shape computation.
const BAND: i32 = 128;

/// Field `kind` (exact up to `reach`) over `out` (inside `region`), computed from the
/// region-sized `shape`. Values within the reach depend only on the shape within
/// [`field_radius`], so a part is computed from a window grown by it.
pub(crate) fn field(kind: FieldKind, reach: i32, shape: &[f32], region: Rect, out: Rect) -> Vec<f32> {
    field_banded(kind, reach, shape, region, out, field_radius(kind, reach) + 1)
}

/// Whether `distance_field` spreads `kind` over threads itself, line by line: the exact Euclidean
/// fields of precise glows and chiselled bevels. Those are computed from one window; for the others
/// (the chamfer fields, a sequential sweep) the window is cut into bands that go in parallel. Without
/// threads (wasm) there is nothing to spread over, and bands keep the memory small.
fn threaded_inside(kind: FieldKind) -> bool {
    !cfg!(target_arch = "wasm32")
        && matches!(kind, FieldKind::Outside | FieldKind::Inside | FieldKind::BevelOutside | FieldKind::BevelInside | FieldKind::StrokeOutsideExact)
}

/// [`field`] with the halo given (the tests shrink it to see that they notice).
///
/// A band needs a halo of `reach + 9` rows for the bevel fields, so at a large size most of what a band
/// computes is its halo (three times the region at size 250), the bands only keep as many threads busy
/// as there are bands, and threads inside a band would pile onto those of the other bands. The Euclidean
/// fields run on all threads inside `distance_field`, from one window. That is also the window the CPU
/// compositor computes them from, so a rebuilt field is exactly the one it gives.
fn field_banded(kind: FieldKind, reach: i32, shape: &[f32], region: Rect, out: Rect, halo: i32) -> Vec<f32> {
    if threaded_inside(kind) {
        let win = out.inflate(halo).intersect(&region);
        if win.is_empty() {
            return Vec::new();
        }
        let f = photocraft_compose::effects::distance_field(kind, crop(shape, region, win), win.width() as usize, win.height() as usize, reach as f32);
        return if win == out { f } else { crop(&f, win, out) };
    }
    let parts = par_map(bands(out, BAND.max(halo)), |band| {
        let win = band.inflate(halo).intersect(&region);
        let f = photocraft_compose::effects::distance_field(kind, crop(shape, region, win), win.width() as usize, win.height() as usize, reach as f32);
        crop(&f, win, band)
    });
    parts.concat()
}

/// The layer's shape (`compose::layer_shape`) over `r`, in parallel bands.
pub(crate) fn shape(doc: &photocraft_doc::Document, layer: &Layer, r: Rect) -> Vec<f32> {
    if r.is_empty() {
        return Vec::new();
    }
    par_map(bands(r, BAND), |band| photocraft_compose::layer_shape(doc, layer, band)).concat()
}

/// Write `v` (row-major over `r`) into the region-sized `shape`.
pub(crate) fn paste(shape: &mut [f32], region: Rect, r: Rect, v: &[f32]) {
    let sw = region.width() as usize;
    let w = r.width() as usize;
    for (row, src) in v.chunks_exact(w).enumerate() {
        let o = (r.y0 - region.y0) as usize * sw + row * sw + (r.x0 - region.x0) as usize;
        shape[o..o + w].copy_from_slice(src);
    }
}

// ---------------------------------------------------------------------------------------------
// Shape identity and damage

pub(crate) type Tiles = HashMap<TileCoord, Arc<Tile>>;

pub(crate) fn snapshot(s: &Surface) -> Tiles {
    s.tiles().map(|(c, t)| (*c, t.clone())).collect()
}

/// Union of the tiles that differ between `old` and `s`.
pub(crate) fn damage(old: &Tiles, s: &Surface) -> Rect {
    let mut d = Rect::EMPTY;
    let mut add = |r: Rect| d = if d.is_empty() { r } else { d.union(&r) };
    for (c, t) in s.tiles() {
        if old.get(c).is_none_or(|o| !Arc::ptr_eq(o, t)) {
            add(c.rect());
        }
    }
    for c in old.keys() {
        if s.tile(*c).is_none() {
            add(c.rect());
        }
    }
    d
}

/// The surfaces a non-group layer's shape is read from: (content, mask).
pub(crate) fn shape_sources(layer: &Layer) -> (Option<&Surface>, Option<&Surface>) {
    let content = match &layer.content {
        LayerContent::Fill(f) => layer.fill_cache.as_ref().filter(|c| c.fill == *f).map(|c| &c.surface),
        _ => layer.surface(),
    };
    let mask = layer.mask.as_ref().filter(|m| m.enabled).map(|m| &m.surface);
    (content, mask)
}

/// Everything a non-group layer's shape depends on except tile contents and position.
pub(crate) fn shape_key(layer: &Layer, canvas: Rect) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    layer.id.0.hash(&mut h);
    let (content, mask) = shape_sources(layer);
    match &layer.content {
        LayerContent::Fill(f) => {
            format!("{f:?}").hash(&mut h);
            let fr = photocraft_compose::fill_frame(layer, canvas);
            (fr.x0, fr.y0, fr.x1, fr.y1, content.is_some()).hash(&mut h);
        }
        // A filled shape's effect shape follows its outline too.
        LayerContent::Shape(sh) => {
            std::mem::discriminant(&layer.content).hash(&mut h);
            if photocraft_compose::effect_outline(layer).is_some() {
                format!("{:?}", sh.path).hash(&mut h);
            }
        }
        other => std::mem::discriminant(other).hash(&mut h),
    }
    for s in [content, mask].into_iter().flatten() {
        format!("{:?}{:?}", s.format(), s.default_pixel()).hash(&mut h);
    }
    if let Some(m) = &layer.mask {
        (m.enabled, m.density.to_bits(), m.feather.to_bits()).hash(&mut h);
    }
    format!("{:?}", layer.vector_mask).hash(&mut h);
    h.finish()
}

/// Deep identity of a group (children, their pixels by tile pointer, settings).
pub(crate) fn group_key(layer: &Layer, light: &GlobalLight) -> u64 {
    fn surface_fp(s: &Surface, h: &mut std::collections::hash_map::DefaultHasher) {
        format!("{:?}{:?}", s.format(), s.default_pixel()).hash(h);
        for (c, t) in s.tiles() {
            (c.tx, c.ty, Arc::as_ptr(t) as usize).hash(h);
        }
    }
    fn walk(l: &Layer, h: &mut std::collections::hash_map::DefaultHasher) {
        l.id.0.hash(h);
        (l.visible, l.clipped, l.opacity.to_bits(), l.fill_opacity.to_bits(), format!("{:?}", l.blend)).hash(h);
        match &l.content {
            LayerContent::Group(g) => {
                for c in &g.children {
                    walk(c, h);
                }
            }
            LayerContent::Fill(f) => {
                format!("{f:?}").hash(h);
                if let Some(c) = &l.fill_cache {
                    surface_fp(&c.surface, h);
                }
            }
            LayerContent::Adjustment(a) => format!("{a:?}").hash(h),
            _ => {
                if let Some(s) = l.surface() {
                    surface_fp(s, h);
                }
            }
        }
        if let Some(m) = &l.mask {
            (m.enabled, m.density.to_bits(), m.feather.to_bits()).hash(h);
            surface_fp(&m.surface, h);
        }
        format!("{:?}{:?}", l.vector_mask, l.effects).hash(h);
        h.write_u8(0xfe);
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (light.angle.to_bits(), light.altitude.to_bits()).hash(&mut h);
    match &layer.content {
        LayerContent::Group(g) => {
            for c in &g.children {
                walk(c, &mut h);
            }
            if let Some(m) = &layer.mask {
                (m.enabled, m.density.to_bits(), m.feather.to_bits()).hash(&mut h);
                surface_fp(&m.surface, &mut h);
            }
        }
        _ => walk(layer, &mut h),
    }
    h.finish()
}

#[cfg(test)]
#[allow(clippy::unreachable)] // clippy.toml exempts unwrap/expect/panic in tests, not unreachable!
mod tests {
    use super::*;
    use photocraft_color::BlendMode;
    use photocraft_doc::FxCommon;

    #[test]
    fn programs_key_on_geometry_not_colour() {
        let light = GlobalLight::default();
        let Effect::DropShadow(mut s) = Effect::default_drop_shadow() else { unreachable!() };
        let a = program(&Effect::DropShadow(s.clone()), &light, false);
        s.color = photocraft_color::Color::rgb(1.0, 0.0, 0.0);
        s.common = FxCommon::new(BlendMode::Screen, 0.3);
        let b = program(&Effect::DropShadow(s.clone()), &light, false);
        assert_eq!(a.key, b.key);
        s.size += 1.0;
        assert_ne!(a.key, program(&Effect::DropShadow(s), &light, false).key);
    }

    #[test]
    fn windows_grow_by_reach_and_temps_are_reused() {
        let light = GlobalLight::default();
        let Effect::DropShadow(mut s) = Effect::default_drop_shadow() else { unreachable!() };
        s.spread = 0.5;
        let p = program(&Effect::DropShadow(s), &light, false);
        let region = Rect::new(-100, -100, 1000, 1000);
        let d = Rect::new(0, 0, 10, 10);
        let fr = |_| 5;
        let w = p.windows(d, region, &fr);
        let last = *w.last().unwrap();
        assert!(last.contains_rect(&d.inflate(7)), "{last:?}");
        // Earlier stages need at least as much as later ones read.
        assert!(w[0].width() >= last.width());
        let (t, n) = p.temps();
        assert!(n <= 3, "{n} temporaries for {} stages", p.stages.len());
        assert!(t.last().unwrap().is_none());
    }

    /// A distance field computed in bands from grown windows equals the whole-region field
    /// wherever it is within its reach (the property incremental updates rely on).
    #[test]
    fn banded_fields_match_whole_region_within_reach() {
        let region = Rect::new(-20, -10, 300, 290);
        let (w, h) = (region.width() as usize, region.height() as usize);
        let mut shape = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let (fx, fy) = (x as f32 - 150.0, y as f32 - 140.0);
                let r = (fx * fx + fy * fy).sqrt() + 8.0 * (fy * 0.07).sin() * (fx * 0.05).cos();
                shape[y * w + x] = (90.0 - r).clamp(0.0, 1.0) * if (x / 13 + y / 17) % 5 == 0 { 0.6 } else { 1.0 };
            }
        }
        for kind in [
            FieldKind::Outside,
            FieldKind::Inside,
            FieldKind::BevelOutside,
            FieldKind::BevelInside,
            FieldKind::ChokeInside,
            FieldKind::StrokeOutside,
            FieldKind::StrokeInside,
            FieldKind::StrokeOutsideExact,
            FieldKind::StrokeOutsideVector,
            FieldKind::StrokeOutsideVectorChamfer,
        ] {
            let reach = 9;
            let whole = photocraft_compose::effects::distance_field(kind, shape.clone(), w, h, reach as f32);
            let banded = field(kind, reach, &shape, region, region);
            let part = Rect::new(40, 30, 170, 120);
            let partial = field(kind, reach, &shape, region, part);
            for (i, (a, b)) in whole.iter().zip(&banded).enumerate() {
                if a.min(*b) < reach as f32 {
                    assert!((a - b).abs() < 1e-5, "{kind:?} at {i}: {a} vs {b}");
                }
            }
            for (i, b) in partial.iter().enumerate() {
                let (x, y) = (part.x0 + (i % part.width() as usize) as i32, part.y0 + (i / part.width() as usize) as i32);
                let a = whole[(y - region.y0) as usize * w + (x - region.x0) as usize];
                if a.min(*b) < reach as f32 {
                    assert!((a - b).abs() < 1e-5, "{kind:?} at ({x},{y}): {a} vs {b}");
                }
            }
        }
    }
    #[test]
    fn windowed_bevel_fields_match_whole_region_on_anti_aliased_edges() {
        // The bevel fields read the edge pixels around the foot of the perpendicular and take the
        // direction of each from its neighbours, so the halo of a part of the region has to cover all of
        // that. A wide bar with clean anti-aliased edges (almost along the 128 row bands the other
        // fields use, across them and at an angle) puts sharp edge pixels where bands would meet and
        // inside the partial window. The whole region is one window, so it is exactly the CPU's field.
        let region = Rect::new(-20, -10, 300, 290);
        let (w, h) = (region.width() as usize, region.height() as usize);
        let part = Rect::new(40, 30, 170, 120);
        let mut too_small_noticed = false;
        for deg in [1.5f32, 5.0, 30.0, 88.5] {
            let (sn, cs) = deg.to_radians().sin_cos();
            let inside = |x: f32, y: f32| {
                let (u, v) = ((x - 150.0) * cs + (y - 128.3) * sn, -(x - 150.0) * sn + (y - 128.3) * cs);
                u.abs() <= 140.0 && v.abs() <= 60.0
            };
            let mut shape = vec![0.0f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    let hits = (0..64).filter(|k| inside(x as f32 + ((k % 8) as f32 + 0.5) / 8.0, y as f32 + ((k / 8) as f32 + 0.5) / 8.0)).count();
                    shape[y * w + x] = hits as f32 / 64.0;
                }
            }
            for reach in [9, 24] {
                for kind in [FieldKind::BevelInside, FieldKind::BevelOutside] {
                    let whole = photocraft_compose::effects::distance_field(kind, shape.clone(), w, h, reach as f32);
                    let differs = |part_field: &[f32], out: Rect| {
                        let ow = out.width() as usize;
                        part_field
                            .iter()
                            .enumerate()
                            .filter(|&(i, b)| {
                                let (x, y) = (out.x0 + (i % ow) as i32, out.y0 + (i / ow) as i32);
                                let a = whole[(y - region.y0) as usize * w + (x - region.x0) as usize];
                                a.min(*b) < reach as f32 && (a - b).abs() > 1e-5
                            })
                            .count()
                    };
                    let all = field(kind, reach, &shape, region, region);
                    assert!(
                        all.iter().zip(&whole).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{kind:?}, {deg} deg, reach {reach}: the whole region is not the CPU's field"
                    );
                    assert_eq!(
                        differs(&field(kind, reach, &shape, region, part), part),
                        0,
                        "{kind:?}, {deg} deg, reach {reach}: partial window differs from whole"
                    );
                    // The test can see a halo that is too small: two rows past the part is not enough.
                    too_small_noticed |= differs(&field_banded(kind, reach, &shape, region, part, 2), part) > 0;
                }
            }
        }
        assert!(too_small_noticed, "a 2 row halo went unnoticed, so this test guards nothing");
    }

    /// #3646: a stroke's outer edge is the fitted 5 × 5 chamfer while it is narrow and the exact
    /// Euclidean transform once it is wide — the chamfer's octagon facets a wide stroke. The GPU
    /// builds it from the field its width picks (`stroke_outside_exact`), while the spread of drop
    /// shadows and softer outer glows keeps the chamfer field (fitted to Photoshop, #224) — so
    /// routing a wide stroke through the chamfer, or a shadow through the exact field, fails here.
    #[test]
    fn a_stroke_uses_the_chamfer_narrow_and_the_exact_field_wide() {
        use photocraft_color::{BlendMode, Color};
        use photocraft_doc::{FxCommon, FxPaint, StrokeFx, StrokePosition};
        let light = GlobalLight::default();
        let stroke =
            |size, position| Effect::Stroke(StrokeFx { common: FxCommon::new(BlendMode::Normal, 1.0), size, position, paint: FxPaint::Color(Color::BLACK) });
        // Narrow (< STROKE_CHAMFER_MAX): the chamfer, exactly as before #3646.
        let narrow = program(&stroke(6.0, StrokePosition::Outside), &light, false);
        assert!(narrow.fields.iter().any(|(k, _)| *k == FieldKind::StrokeOutside), "a narrow raster outside stroke keeps the chamfer: {:?}", narrow.fields);
        assert!(!narrow.fields.iter().any(|(k, _)| *k == FieldKind::StrokeOutsideExact), "…not the exact field: {:?}", narrow.fields);
        // Wide: the exact transform.
        let wide = program(&stroke(40.0, StrokePosition::Outside), &light, false);
        assert!(
            wide.fields.iter().any(|(k, _)| *k == FieldKind::StrokeOutsideExact),
            "a wide raster outside stroke needs the exact outer field: {:?}",
            wide.fields
        );
        assert!(!wide.fields.iter().any(|(k, _)| *k == FieldKind::StrokeOutside), "…not the chamfer field: {:?}", wide.fields);
        // A shape layer's outline stroke follows the same width rule; the inside edge keeps the chamfer.
        let vector_narrow = program(&stroke(6.0, StrokePosition::Outside), &light, true);
        assert!(vector_narrow.fields.iter().any(|(k, _)| *k == FieldKind::StrokeOutsideVectorChamfer), "{:?}", vector_narrow.fields);
        let vector_wide = program(&stroke(40.0, StrokePosition::Outside), &light, true);
        assert!(vector_wide.fields.iter().any(|(k, _)| *k == FieldKind::StrokeOutsideVector), "{:?}", vector_wide.fields);
        let inner = program(&stroke(40.0, StrokePosition::Inside), &light, false);
        assert!(inner.fields.iter().any(|(k, _)| *k == FieldKind::StrokeInside), "{:?}", inner.fields);
        // The shadow / softer-glow spread is unchanged: still the chamfer.
        let Effect::DropShadow(mut s) = Effect::default_drop_shadow() else { unreachable!() };
        s.spread = 0.5;
        let shadow = program(&Effect::DropShadow(s), &light, false);
        assert!(shadow.fields.iter().any(|(k, _)| *k == FieldKind::StrokeOutside), "the shadow spread keeps the chamfer: {:?}", shadow.fields);
    }
}
