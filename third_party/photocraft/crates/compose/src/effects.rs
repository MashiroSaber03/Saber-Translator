//! CPU reference renderer for layer effects (layer styles).
//!
//! Render order, bottom to top (Photoshop's layer-style stack read upwards):
//! drop shadows, outer glows, outer bevel, then the layer itself (content at
//! fill opacity) with pattern/gradient/colour overlays, satin, inner glows,
//! inner shadows, strokes and the inner bevel on top. Exterior effects blend
//! straight into the backdrop with their own modes; the layer blends with its
//! mode and the interior effects paint over the result, or blend with it as a
//! group (Blend Interior Effects as Group, [`premerges_layer`]). Emboss and pillow emboss then
//! shade the composited result (inside and outside halves). Layer opacity
//! applies to the whole stack; fill opacity only to the layer's own pixels.
//!
//! Interior effects are painted relative to the layer's shape and then take the shape's alpha
//! (an overlay recolours a half-transparent edge without adding coverage). The outside parts
//! of strokes blend onto the backdrop with their own modes, an upper stroke covering the
//! lower ones. Drop shadows are knocked out through see-through fill, and beneath the whole shape
//! when the layer's mode lets what's under it show through (see [`shadow_see_through`]).
//!
//! Shapes come from the layer's alpha (after its mask). A stroke's outer edge, and precise glows /
//! chiselled bevels, use an exact Euclidean transform: a 5 × 5 chamfer's level sets are polygons,
//! which a wide stroke shows as flat facets (#3646). Inside stroke edges and spread / choke keep the
//! cheaper chamfer. Soft falloffs are two box blurs (a tent).

use photocraft_color::blend::BlendMode;
use photocraft_doc::Pattern;
use photocraft_doc::{
    Bevel, BevelStyle, BevelTechnique, Contour, Effect, FxPaint, GlobalLight, Glow, GlowSource, GlowTechnique, Gradient, GradientStyle, Layer, Shadow,
    StrokePosition,
};
use photocraft_geom::Rect;

use crate::pattern::{PREPARED_PATTERN_BYTES, Placement, PreparedPatterns};
use crate::{Buffer, psblend};

/// `true` if the layer has at least one enabled effect and the master switch is on.
pub fn has_effects(layer: &Layer) -> bool {
    layer.effects.enabled && layer.effects.items.iter().any(Effect::enabled)
}

/// Pixels an effect stack can reach beyond the layer's shape.
pub fn margin(layer: &Layer) -> i32 {
    let mut m = 0.0f32;
    for e in layer.effects.items.iter().filter(|e| e.enabled()) {
        let r = match e {
            Effect::DropShadow(s) | Effect::InnerShadow(s) => s.distance + s.size,
            Effect::OuterGlow(g) | Effect::InnerGlow(g) => g.size,
            Effect::Stroke(s) => s.size,
            Effect::Satin(s) => satin_distance(s) + s.size,
            Effect::BevelEmboss(b) => b.size + b.soften,
            _ => 0.0,
        };
        if r.is_finite() {
            m = m.max(r);
        }
    }
    // Bounded so malformed values can't request huge buffers; effects that
    // reach further are clipped at this distance.
    m.clamp(0.0, MAX_REACH).ceil() as i32 + 2
}

/// Maximum distance (px) an effect is rendered from the layer's shape.
pub const MAX_REACH: f32 = 512.0;

/// Whether every enabled effect's maps are local: each map pixel depends only on the shape
/// within [`margin`] of it (shifts, chamfer distances compared with a size, tent blurs), so
/// maps built over a part of the layer's region padded by twice the margin equal the
/// whole-region maps over that part (#1909). Not local: gradient strokes (laid out over the whole
/// shape's extent), Precise glows and chiselled bevels (exact Euclidean distances, whose f32
/// arithmetic depends on where the region starts) and noise (it speckles any coverage above
/// zero, so it would pick up the blur's float noise, which depends on where the region starts).
pub(crate) fn maps_are_local(layer: &Layer) -> bool {
    layer.effects.items.iter().filter(|e| e.enabled()).all(|e| match e {
        Effect::Stroke(s) => !matches!(s.paint, FxPaint::Gradient(_)),
        Effect::DropShadow(s) | Effect::InnerShadow(s) => s.noise <= 0.0,
        Effect::OuterGlow(g) | Effect::InnerGlow(g) => g.technique != GlowTechnique::Precise && g.noise <= 0.0,
        Effect::BevelEmboss(b) => b.technique == BevelTechnique::Smooth,
        _ => true,
    })
}

/// A single-channel map over a rectangle.
#[derive(Clone)]
struct Map {
    w: usize,
    h: usize,
    v: Vec<f32>,
}

impl Map {
    fn new(w: usize, h: usize, fill: f32) -> Self {
        Map { w, h, v: vec![fill; w * h] }
    }
    fn get(&self, x: i64, y: i64) -> f32 {
        if x < 0 || y < 0 || x >= self.w as i64 || y >= self.h as i64 { 0.0 } else { self.v[y as usize * self.w + x as usize] }
    }
    /// Shifted copy (bilinear), outside reads `outside`.
    fn shifted(&self, dx: f32, dy: f32, outside: f32) -> Map {
        let mut out = Map::new(self.w, self.h, 0.0);
        // A hostile distance (±1e30, inf, NaN) would overflow the i64 offsets below; no shift beyond
        // 2^30 px can land inside a map, so clamp (non-finite shifts read as none).
        let reach = |d: f32| if d.is_finite() { d.clamp(-1.0e9, 1.0e9) } else { 0.0 };
        let (dx, dy) = (reach(dx), reach(dy));
        let (fx, fy) = (dx.floor(), dy.floor());
        let (ax, ay) = (dx - fx, dy - fy);
        let s = |x: i64, y: i64| if x < 0 || y < 0 || x >= self.w as i64 || y >= self.h as i64 { outside } else { self.v[y as usize * self.w + x as usize] };
        for y in 0..self.h as i64 {
            for x in 0..self.w as i64 {
                let (sx, sy) = (x - fx as i64, y - fy as i64);
                let v00 = s(sx, sy);
                let v10 = s(sx - 1, sy);
                let v01 = s(sx, sy - 1);
                let v11 = s(sx - 1, sy - 1);
                let top = v00 * (1.0 - ax) + v10 * ax;
                let bot = v01 * (1.0 - ax) + v11 * ax;
                out.v[y as usize * self.w + x as usize] = top * (1.0 - ay) + bot * ay;
            }
        }
        out
    }
    fn map(mut self, f: impl Fn(f32) -> f32) -> Self {
        for v in &mut self.v {
            *v = f(*v);
        }
        self
    }
}

/// 1D squared distance transform (Felzenszwalb & Huttenlocher).
fn dt1(f: &[f32], out: &mut [f32], v: &mut [usize], z: &mut [f32], near: &mut [usize]) {
    let n = f.len();
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;
    for q in 1..n {
        loop {
            let p = v[k];
            let s = ((f[q] + (q * q) as f32) - (f[p] + (p * p) as f32)) / (2.0 * q as f32 - 2.0 * p as f32);
            if s <= z[k] && k > 0 {
                k -= 1;
                continue;
            }
            if s <= z[k] {
                // k == 0 and parabola dominates: replace.
                v[0] = q;
                z[0] = f32::NEG_INFINITY;
                z[1] = f32::INFINITY;
                break;
            }
            k += 1;
            v[k] = q;
            z[k] = s;
            z[k + 1] = f32::INFINITY;
            break;
        }
    }
    k = 0;
    for (q, o) in out.iter_mut().enumerate() {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let p = v[k];
        let d = q as f32 - p as f32;
        *o = d * d + f[p];
        near[q] = p;
    }
}

/// Exact Euclidean distance from every pixel to the nearest pixel where
/// `inside` is true (0 on those pixels).
#[cfg(test)]
fn edt(inside: &[bool], w: usize, h: usize) -> Vec<f32> {
    edt_nearest(inside, w, h).0
}

/// Maps with fewer pixels (or other units of work) than this are not worth spawning threads for.
const PAR_MIN: usize = 1 << 16;

/// Most threads one pass uses (as for the blurs, [`rows_par`]).
const PAR_THREADS: usize = 16;

#[cfg(test)]
thread_local! {
    /// Test hook: the thread count [`par_threads`] gives on this thread, whatever the work.
    static THREADS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// How many threads a pass over `work` pixels (or similar units) uses.
fn par_threads(work: usize) -> usize {
    #[cfg(test)]
    if let Some(n) = THREADS.with(std::cell::Cell::get) {
        return n;
    }
    if cfg!(target_arch = "wasm32") || work < PAR_MIN { 1 } else { std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, PAR_THREADS) }
}

/// Runs `f(scratch, u, a_u, b_u)` for every unit `u` of two slices: unit `u` of `a` is
/// `a[u * unit_a..][..unit_a]` (the last may be shorter), the same of `b` with `unit_b`, which is 0
/// when there is no second slice. `work` says how big the whole pass is ([`par_threads`]). Groups of
/// units go to scoped OS threads, each with a scratch value from `init`; not to rayon, as for
/// [`rows_par`]: maps are built from inside rayon tiles, where a waiting worker steals other tiles.
/// A unit is computed from its own arguments alone, so the result does not depend on the threads.
fn par_units<S, A: Send, B: Send>(
    work: usize,
    a: &mut [A],
    unit_a: usize,
    b: &mut [B],
    unit_b: usize,
    init: impl Fn() -> S + Sync,
    f: impl Fn(&mut S, usize, &mut [A], &mut [B]) + Sync,
) {
    if unit_a == 0 {
        return;
    }
    let units = a.len().div_ceil(unit_a);
    let threads = par_threads(work).min(units);
    if threads <= 1 {
        run_units(&mut init(), 0, a, unit_a, b, unit_b, &f);
        return;
    }
    // Many more groups than threads, so that a heavy one (an edge-rich stretch) does not hold the rest up.
    let group = units.div_ceil(threads * 8).max(1);
    let mut groups = Vec::new();
    let (mut rest_a, mut rest_b, mut first) = (a, b, 0usize);
    while !rest_a.is_empty() {
        let (ga, na) = rest_a.split_at_mut((group * unit_a).min(rest_a.len()));
        let (gb, nb) = rest_b.split_at_mut((group * unit_b).min(rest_b.len()));
        groups.push((first, ga, gb));
        (rest_a, rest_b, first) = (na, nb, first + group);
    }
    let queue = std::sync::Mutex::new(groups.into_iter());
    let worker = || {
        let mut scratch = init();
        loop {
            let next = queue.lock().unwrap_or_else(std::sync::PoisonError::into_inner).next();
            let Some((first, ga, gb)) = next else { break };
            run_units(&mut scratch, first, ga, unit_a, gb, unit_b, &f);
        }
    };
    std::thread::scope(|sc| {
        // A thread that cannot be spawned leaves its share to the others.
        for _ in 1..threads {
            let _ = std::thread::Builder::new().spawn_scoped(sc, worker);
        }
        worker();
    });
}

fn run_units<S, A, B>(scratch: &mut S, first: usize, a: &mut [A], unit_a: usize, b: &mut [B], unit_b: usize, f: &impl Fn(&mut S, usize, &mut [A], &mut [B])) {
    for (k, ua) in a.chunks_mut(unit_a).enumerate() {
        let ub = b.get_mut(k * unit_b..((k + 1) * unit_b).min(b.len())).unwrap_or(&mut []);
        f(scratch, first + k, ua, ub);
    }
}

/// A `w × h` buffer of `fill`, then filled row by row through `f(y, row)`.
fn par_fill<T: Clone + Send>(w: usize, h: usize, fill: T, f: impl Fn(usize, &mut [T]) + Sync) -> Vec<T> {
    let mut out = vec![fill; w.saturating_mul(h)];
    par_units(out.len(), &mut out, w.max(1), &mut [] as &mut [()], 0, || (), |_, y, row, _| f(y, row));
    out
}

/// Working buffers of the 1D distance transform, for lines of up to `n` pixels.
struct Dt1Scratch {
    f: Vec<f32>,
    o: Vec<f32>,
    v: Vec<usize>,
    z: Vec<f32>,
    near: Vec<usize>,
}

impl Dt1Scratch {
    fn new(n: usize) -> Self {
        Dt1Scratch { f: vec![0.0; n], o: vec![0.0; n], v: vec![0; n], z: vec![0.0; n + 1], near: vec![0; n] }
    }
}

/// [`edt`] plus the index of the nearest `inside` pixel (as `u32`: a map past 2³² pixels has no
/// nearest pixel to give, and every distance is then "far").
///
/// Both passes run line by line in parallel: the columns, then the rows. The columns' results are
/// kept column by column (`gc[x * h + y]`) so that each thread writes its own stretch of them.
fn edt_nearest(inside: &[bool], w: usize, h: usize) -> (Vec<f32>, Vec<u32>) {
    const INF: f32 = 1e20;
    let total = w.saturating_mul(h);
    if total != inside.len() || total > u32::MAX as usize {
        return (vec![INF; inside.len()], vec![0; inside.len()]);
    }
    let n = w.max(h);
    let mut gc = vec![0.0f32; total];
    let mut nc = vec![0u32; total];
    par_units(
        total,
        &mut gc,
        h.max(1),
        &mut nc,
        h.max(1),
        || Dt1Scratch::new(n),
        |t, x, g, near| {
            for (y, f) in t.f[..h].iter_mut().enumerate() {
                *f = if inside[y * w + x] { 0.0 } else { INF };
            }
            dt1(&t.f[..h], &mut t.o[..h], &mut t.v, &mut t.z, &mut t.near[..h]);
            g.copy_from_slice(&t.o[..h]);
            for (d, &p) in near.iter_mut().zip(&t.near[..h]) {
                *d = p as u32;
            }
        },
    );
    let mut g = vec![0.0f32; total];
    let mut nearest = vec![0u32; total];
    par_units(
        total,
        &mut g,
        w.max(1),
        &mut nearest,
        w.max(1),
        || Dt1Scratch::new(n),
        |t, y, out, near| {
            for (x, f) in t.f[..w].iter_mut().enumerate() {
                *f = gc[x * h + y];
            }
            dt1(&t.f[..w], &mut t.o[..w], &mut t.v, &mut t.z, &mut t.near[..w]);
            for (x, (d, ne)) in out.iter_mut().zip(near.iter_mut()).enumerate() {
                *d = t.o[x].sqrt();
                let px = t.near[x];
                *ne = (nc[px * h + y] as usize * w + px) as u32;
            }
        },
    );
    (g, nearest)
}

/// Chamfer distance transform with a 5 × 5 mask of exact Euclidean step lengths (1, √2, √5),
/// plus the index of the nearest `inside` pixel (0 on inside pixels). Photoshop's effect
/// distances follow this metric: its stroke corners measure 3.236 = √5 + 1 at offset (1, 3)
/// and 3.650 = √5 + √2 at (2, 3), where the exact distances are 3.162 and 3.606.
#[cfg(test)]
fn chamfer_nearest(inside: &[bool], w: usize, h: usize) -> (Vec<f32>, Vec<usize>) {
    chamfer_from(inside.iter().map(|&b| if b { 0.0 } else { CHAMFER_INF }).collect(), w, h)
}

const CHAMFER_INF: f32 = 1e20;

/// [`chamfer_nearest`] from per-pixel start values (`CHAMFER_INF` = not a seed): the result is
/// the minimum over seeds of start value + path length, so partly covered seeds can start at
/// their sub-pixel edge offset.
fn chamfer_from(mut d: Vec<f32>, w: usize, h: usize) -> (Vec<f32>, Vec<usize>) {
    const S2: f32 = std::f32::consts::SQRT_2;
    const S5: f32 = 2.236_068;
    // Forward-pass neighbours (already visited in raster order); the backward pass mirrors them.
    const FWD: [(i64, i64, f32); 8] = [(-1, 0, 1.0), (-1, -1, S2), (0, -1, 1.0), (1, -1, S2), (-1, -2, S5), (1, -2, S5), (-2, -1, S5), (2, -1, S5)];
    let mut near: Vec<usize> = (0..w * h).collect();
    let (wi, hi) = (w as i64, h as i64);
    let relax = |x: i64, y: i64, sign: i64, d: &mut Vec<f32>, near: &mut Vec<usize>| {
        let i = (y * wi + x) as usize;
        let mut best = d[i];
        let mut bn = near[i];
        for &(ox, oy, wt) in &FWD {
            let (nx, ny) = (x + ox * sign, y + oy * sign);
            if nx < 0 || ny < 0 || nx >= wi || ny >= hi {
                continue;
            }
            let j = (ny * wi + nx) as usize;
            let c = d[j] + wt;
            if c < best {
                best = c;
                bn = near[j];
            }
        }
        d[i] = best;
        near[i] = bn;
    };
    for y in 0..hi {
        for x in 0..wi {
            relax(x, y, 1, &mut d, &mut near);
        }
    }
    for y in (0..hi).rev() {
        for x in (0..wi).rev() {
            relax(x, y, -1, &mut d, &mut near);
        }
    }
    (d, near)
}

/// Distance from each pixel centre outside the shape to the shape's edge.
/// Pixels with non-zero alpha are inside; the edge inside the nearest such
/// pixel is placed by its alpha (coverage), as Photoshop does. Inside pixels
/// get ≤ 0.
fn dist_outside(s: &Map) -> Vec<f32> {
    dist_outside_by(s, Metric::Euclidean)
}

/// Distance metric for effect distance fields.
#[derive(Clone, Copy, PartialEq)]
enum Metric {
    /// Exact Euclidean (effect edges).
    Euclidean,
    /// 5 × 5 chamfer (inside strokes, spread / choke; see [`chamfer_nearest`]).
    Chamfer,
}

/// Outside width from which a stroke's outer edge uses the exact Euclidean transform instead of
/// the fitted 5 × 5 chamfer.
///
/// The chamfer's level sets are octagons: measured against a point source, its distance exceeds
/// the Euclidean radius by up to `0.027 · r` px — 0.48 px at r = 18 (the widest outside stroke any
/// Photoshop oracle matches with the chamfer) and 1.6 px at r = 60, where a wide stroke's boundary
/// visibly facets (#3646). Half a pixel of deviation is reached at r ≈ 19, so strokes narrower
/// than 20 px keep the chamfer, whose sub-pixel edge seeding Photoshop matches at small offsets,
/// and wider ones stay round.
pub const STROKE_CHAMFER_MAX: f32 = 20.0;

/// Whether a stroke whose outside edge is `out_w` px wide uses the exact Euclidean transform (round)
/// rather than the 5 × 5 chamfer's octagon ([`STROKE_CHAMFER_MAX`]). The CPU compositor and the GPU
/// compositor's `program_with` both call this, so a stroke picks the same field in each.
pub fn stroke_outside_exact(out_w: f32) -> bool {
    out_w >= STROKE_CHAMFER_MAX
}

fn dist_outside_by(s: &Map, m: Metric) -> Vec<f32> {
    if m == Metric::Chamfer {
        // Seeds start at their sub-pixel edge offset (1 − coverage).
        let start = s.v.iter().map(|&a| if a > INSIDE_EPS { 1.0 - a.min(1.0) } else { CHAMFER_INF }).collect();
        let (d, _) = chamfer_from(start, s.w, s.h);
        return d.iter().zip(&s.v).map(|(d, &a)| if a > INSIDE_EPS { -0.5 } else { d - 0.5 }).collect();
    }
    let inside = covered(s, |a| a > INSIDE_EPS);
    let (d, near) = edt_nearest(&inside, s.w, s.h);
    par_fill(s.w, s.h, 0.0f32, |y, row| {
        for (x, out) in row.iter_mut().enumerate() {
            let i = y * s.w + x;
            *out = if inside[i] { -0.5 } else { d[i] - 0.5 + (1.0 - s.v[near[i] as usize].min(1.0)) };
        }
    })
}

/// Which pixels of `s` satisfy `test`, row by row in parallel.
fn covered(s: &Map, test: impl Fn(f32) -> bool + Sync) -> Vec<bool> {
    par_fill(s.w, s.h, false, |y, row| {
        for (e, &a) in row.iter_mut().zip(s.v.get(y * s.w..).unwrap_or_default()) {
            *e = test(a);
        }
    })
}

/// Distance from each pixel centre inside the shape to the shape's edge
/// (≤ 0 outside).
fn dist_inside(s: &Map) -> Vec<f32> {
    dist_inside_by(s, Metric::Euclidean)
}

fn dist_inside_by(s: &Map, m: Metric) -> Vec<f32> {
    if m == Metric::Chamfer {
        // Partly covered edge pixels (alpha below their neighbourhood's) are seeds starting at
        // their coverage, mirroring `dist_outside`: a 25 % edge column puts the edge 0.25 px in.
        let cov = local_coverage(s);
        let start =
            s.v.iter()
                .zip(&cov.v)
                .map(|(&a, &c)| {
                    if a <= INSIDE_EPS {
                        0.0
                    } else if c < 1.0 - INSIDE_EPS {
                        c
                    } else {
                        CHAMFER_INF
                    }
                })
                .collect();
        let (d, _) = chamfer_from(start, s.w, s.h);
        return d.iter().zip(&s.v).map(|(d, &a)| if a <= INSIDE_EPS { -0.5 } else { d - 0.5 }).collect();
    }
    let outside = covered(s, |a| a <= INSIDE_EPS);
    let (d, _) = edt_nearest(&outside, s.w, s.h);
    par_fill(s.w, s.h, 0.0f32, |y, row| {
        for (x, out) in row.iter_mut().enumerate() {
            let i = y * s.w + x;
            *out = if outside[i] { -0.5 } else { d[i] - 0.5 };
        }
    })
}

/// Alpha relative to the 3×3 neighbourhood maximum: separates edge coverage
/// from the fill's own opacity on shape layers.
fn local_coverage(s: &Map) -> Map {
    let mut out = s.clone();
    for y in 0..s.h as i64 {
        for x in 0..s.w as i64 {
            let a = s.get(x, y);
            if a <= INSIDE_EPS {
                continue;
            }
            let mut mx = a;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    mx = mx.max(s.get(x + dx, y + dy));
                }
            }
            out.v[y as usize * s.w + x as usize] = (a / mx).min(1.0);
        }
    }
    out
}

/// Signed distance from the centre of a pixel of coverage `a` to the straight edge crossing it,
/// positive when the centre is inside the shape. `n` is the edge's unit normal (its signs don't
/// matter).
///
/// A unit pixel cut by a straight edge at distance `r` from its centre (on the side that holds at
/// most half of the pixel) is covered `1/2 - r/u` while the edge crosses two opposite sides, and
/// `((u + v)/2 - r)² / (2uv)` once it only cuts off a corner, with `u ≥ v` the larger and smaller
/// component of the normal. Inverting that places the edge by its coverage at any angle. For an
/// axis-aligned edge (`v = 0`) it is the `a - 1/2` that [`dist_outside`] assumes.
fn edge_offset(n: [f32; 2], a: f32) -> f32 {
    if !a.is_finite() {
        return 0.0;
    }
    let (u, v) = (n[0].abs().max(n[1].abs()), n[0].abs().min(n[1].abs()));
    let (c, inside) = if a > 0.5 { (1.0 - a, true) } else { (a, false) };
    let c = c.clamp(0.0, 0.5);
    // The coverage below which the edge only cuts off a corner.
    let corner = if u > 0.0 { v / (2.0 * u) } else { 0.0 };
    let r = if c >= corner { (0.5 - c) * u } else { 0.5 * (u + v) - (2.0 * u * v * c).sqrt() };
    if inside { r } else { -r }
}

/// A sharp, anti-aliased edge pixel: where it puts the straight edge crossing it.
#[derive(Clone, Copy)]
struct SharpEdge {
    /// Signed distance from the pixel's centre to the edge, positive inside ([`edge_offset`]).
    offset: f32,
    /// The edge's unit normal pointing into the shape (`[0, 0]` when the coverage gives no direction).
    normal: [f32; 2],
}

/// Half the spacing of the edge pixels along an edge (at most √2 / 2, along a diagonal): a pixel
/// within it of the foot of the perpendicular sees the edge as a line, one further out as its end.
const EDGE_REACH: f32 = 0.75;

impl SharpEdge {
    /// Distance to the edge from the pixel `(dx, dy)` away from this one, positive on the shape's
    /// side (`inside`) or on the empty side. It is the distance to the edge's line where the
    /// perpendicular lands within [`EDGE_REACH`] of this pixel, and to the end of that stretch
    /// beyond, so the distance to a straight edge is exact and does not follow the pixel staircase.
    /// `None` when the edge has no direction or `(dx, dy)` is on the wrong side of it.
    fn distance(&self, dx: f32, dy: f32, inside: bool) -> Option<f32> {
        let [nx, ny] = self.normal;
        if nx == 0.0 && ny == 0.0 {
            return None;
        }
        let signed = self.offset + dx * nx + dy * ny;
        let across = if inside { signed } else { -signed };
        let along = (dx * -ny + dy * nx).abs();
        (across > 0.0).then(|| across.hypot((along - EDGE_REACH).max(0.0)))
    }
}

/// A sharp edge pixel and where it sits.
#[derive(Clone, Copy)]
struct EdgePixel {
    /// Index of the pixel in the map.
    at: u32,
    /// The pixel's coverage ([`local_coverage`] of it).
    cov: f32,
    edge: SharpEdge,
}

/// The sharp edge pixels of a map: a short list, and for every pixel a 4 byte slot (0 = no edge, k =
/// the k-th of the list). Maps are big and edge pixels few, so a per-pixel `Option<SharpEdge>` (16
/// bytes) made every pass over it, and the copies the refinement needs, memory-bound.
struct SharpEdges {
    w: usize,
    h: usize,
    slot: Vec<u32>,
    pixels: Vec<EdgePixel>,
}

impl SharpEdges {
    fn empty(w: usize, h: usize) -> Self {
        SharpEdges { w, h, slot: Vec::new(), pixels: Vec::new() }
    }

    /// The edge of pixel `i`, if it is a sharp edge pixel.
    fn get(&self, i: usize) -> Option<SharpEdge> {
        let k = (*self.slot.get(i)? as usize).checked_sub(1)?;
        self.pixels.get(k).map(|p| p.edge)
    }

    /// The edge of pixel `(x, y)`, if it is a sharp edge pixel (`None` off the map).
    fn at(&self, x: usize, y: usize) -> Option<SharpEdge> {
        if x >= self.w || y >= self.h { None } else { self.get(y * self.w + x) }
    }

    fn has(&self, x: usize, y: usize) -> bool {
        self.at(x, y).is_some()
    }

    fn is_empty(&self) -> bool {
        self.pixels.is_empty()
    }

    /// Every sharp edge pixel, as (index in the map, its edge), in pixel order.
    #[cfg(test)]
    fn iter(&self) -> impl Iterator<Item = (usize, SharpEdge)> + '_ {
        self.pixels.iter().map(|p| (p.at as usize, p.edge))
    }
}

/// Where the sharp, anti-aliased edge pixels of `s` put the edge ([`edge_offset`], along the normal
/// the coverage gradient gives). Sharp pixels are partly covered and touch both an empty and a full
/// pixel; hard edges and soft ramps have none and keep the `alpha - 1/2` of [`dist_outside`].
///
/// An edge pixel is covered and has an empty neighbour, so a pass over which pixels are covered finds
/// the few candidates without computing [`local_coverage`] of the whole map; the coverage is then
/// worked out where it is needed, by the same formula. Rows are independent and go in parallel.
fn sharp_edges(s: &Map) -> SharpEdges {
    let (w, h) = (s.w, s.h);
    if w == 0 || h == 0 || s.v.len() != w * h || s.v.len() >= u32::MAX as usize {
        return SharpEdges::empty(w, h);
    }
    let (wi, hi) = (w as i64, h as i64);
    // `local_coverage` of the pixel at (x, y), on the map.
    let cov_at = |x: i64, y: i64| -> f32 {
        let a = s.v[y as usize * w + x as usize];
        if a <= INSIDE_EPS {
            return a;
        }
        let mut mx = a;
        for dy in -1..=1 {
            for dx in -1..=1 {
                mx = mx.max(s.get(x + dx, y + dy));
            }
        }
        (a / mx).min(1.0)
    };
    // The coverage with the map's border repeated, for the gradient.
    let cov_clamped = |x: i64, y: i64| cov_at(x.max(0).min(wi - 1), y.max(0).min(hi - 1));
    // Whether a pixel is not empty. Off the map counts as covered: only a neighbour on the map can be empty.
    let solid = |x: i64, y: i64| -> bool {
        if x < 0 || y < 0 || x >= wi || y >= hi {
            return true;
        }
        let a = s.v[y as usize * w + x as usize];
        a.is_nan() || a > INSIDE_EPS
    };
    let mut rows: Vec<Vec<EdgePixel>> = vec![Vec::new(); h];
    par_units(
        w * h,
        &mut rows,
        1,
        &mut [] as &mut [()],
        0,
        || (),
        |_, y, row, _| {
            let Some(row) = row.first_mut() else { return };
            let y = y as i64;
            for x in 0..wi {
                if !solid(x, y) || (-1..=1).all(|dy| (-1..=1).all(|dx| solid(x + dx, y + dy))) {
                    continue;
                }
                let i = (y * wi + x) as usize;
                let c = cov_at(x, y);
                if s.v[i] <= INSIDE_EPS || c >= 1.0 - INSIDE_EPS {
                    continue;
                }
                let (mut empty, mut full) = (false, false);
                for (dx, dy) in (-1..=1).flat_map(|dy| (-1..=1).map(move |dx| (dx, dy))) {
                    let (nx, ny) = (x + dx, y + dy);
                    if (dx, dy) == (0, 0) || nx < 0 || ny < 0 || nx >= wi || ny >= hi {
                        continue;
                    }
                    empty |= s.get(nx, ny) <= INSIDE_EPS;
                    full |= cov_at(nx, ny) >= 1.0 - INSIDE_EPS;
                }
                if !(empty && full) {
                    continue;
                }
                let at = &cov_clamped;
                let gx = (at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2.0 * at(x - 1, y) + at(x - 1, y + 1));
                let gy = (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2.0 * at(x, y - 1) + at(x + 1, y - 1));
                let len = gx.hypot(gy);
                let (normal, n) = if len.is_finite() && len > 1e-6 { ([gx / len, gy / len], [gx / len, gy / len]) } else { ([0.0, 0.0], [1.0, 0.0]) };
                row.push(EdgePixel { at: i as u32, cov: c, edge: SharpEdge { offset: edge_offset(n, c), normal } });
            }
        },
    );
    let mut pixels: Vec<EdgePixel> = Vec::with_capacity(rows.iter().map(Vec::len).sum());
    for row in rows {
        pixels.extend(row);
    }
    let mut slot = vec![0u32; w * h];
    for (k, p) in pixels.iter().enumerate() {
        if let Some(s) = slot.get_mut(p.at as usize) {
            *s = k as u32 + 1;
        }
    }
    refine_edge_normals(&mut pixels, &slot, w, h);
    SharpEdges { w, h, slot, pixels }
}

/// How far (in pixels) around an edge pixel [`refine_edge_normals`] looks for its neighbours.
const NORMAL_RADIUS: i64 = 3;

/// The 3×3 gradient of the coverage gives an edge's direction to 1-2 degrees, and its error is
/// systematic, so it tilts the distance of every pixel the same way and shows as a faint ripple.
/// The edge points of a straight edge's pixels (each pixel's centre moved to its edge) lie on one
/// line, and the line through the points of the pixels around one gives the direction to a fraction
/// of a degree. Where those points are not on a line (corners, tight curves) the gradient's normal
/// stays.
fn refine_edge_normals(pixels: &mut [EdgePixel], slot: &[u32], w: usize, h: usize) {
    let (wi, hi) = (w as i64, h as i64);
    let points: Vec<[f32; 2]> = pixels
        .iter()
        .map(|p| [(p.at as i64 % wi) as f32 - p.edge.offset * p.edge.normal[0], (p.at as i64 / wi) as f32 - p.edge.offset * p.edge.normal[1]])
        .collect();
    let before: Vec<SharpEdge> = pixels.iter().map(|p| p.edge).collect();
    // Every pixel reads the unrefined edges around it and writes its own, so they go in parallel.
    let work = pixels.len().saturating_mul(32);
    par_units(work, pixels, 1, &mut [] as &mut [()], 0, Vec::<[f32; 2]>::new, |near, k, pixel, _| {
        let (Some(pixel), Some(&e)) = (pixel.first_mut(), before.get(k)) else { return };
        let (x, y) = (pixel.at as i64 % wi, pixel.at as i64 / wi);
        let (mut n, mut sum) = (0.0f32, [0.0f32; 2]);
        near.clear();
        for (dx, dy) in (-NORMAL_RADIUS..=NORMAL_RADIUS).flat_map(|dy| (-NORMAL_RADIUS..=NORMAL_RADIUS).map(move |dx| (dx, dy))) {
            let (nx, ny) = (x + dx, y + dy);
            if dx * dx + dy * dy > NORMAL_RADIUS * NORMAL_RADIUS || nx < 0 || ny < 0 || nx >= wi || ny >= hi {
                continue;
            }
            let Some(j) = slot.get((ny * wi + nx) as usize).and_then(|&s| (s as usize).checked_sub(1)) else { continue };
            let (Some(o), Some(&p)) = (before.get(j), points.get(j)) else { continue };
            if o.normal[0] * e.normal[0] + o.normal[1] * e.normal[1] > 0.9 {
                near.push(p);
                sum = [sum[0] + p[0], sum[1] + p[1]];
                n += 1.0;
            }
        }
        if near.len() < 3 {
            return;
        }
        let mean = [sum[0] / n, sum[1] / n];
        let (mut sxx, mut sxy, mut syy) = (0.0f32, 0.0f32, 0.0f32);
        for p in near.iter() {
            let (dx, dy) = (p[0] - mean[0], p[1] - mean[1]);
            sxx += dx * dx;
            sxy += dx * dy;
            syy += dy * dy;
        }
        // Spread along and across the best line through the points.
        let half = 0.5 * (sxx + syy);
        let root = (0.25 * (sxx - syy) * (sxx - syy) + sxy * sxy).sqrt();
        let (along, across) = (((half + root) / n).max(0.0).sqrt(), ((half - root) / n).max(0.0).sqrt());
        if !(along > EDGE_LINE_MIN_SPREAD && across < EDGE_LINE_MAX_OFF) {
            return;
        }
        let angle = 0.5 * (2.0 * sxy).atan2(sxx - syy);
        let mut normal = [-angle.sin(), angle.cos()];
        if normal[0] * e.normal[0] + normal[1] * e.normal[1] < 0.0 {
            normal = [-normal[0], -normal[1]];
        }
        pixel.edge.normal = normal;
        pixel.edge.offset = edge_offset(normal, pixel.cov);
    });
}

/// The edge points around a pixel must spread at least this far (px, standard deviation) along
/// their line to give it a direction, and stay within [`EDGE_LINE_MAX_OFF`] of it.
const EDGE_LINE_MIN_SPREAD: f32 = 0.7;
const EDGE_LINE_MAX_OFF: f32 = 0.1;

/// How far from the foot of the perpendicular (in pixels) [`edge_distance`] looks for edge pixels.
const FOOT_RADIUS: i64 = 2;

/// The distance from pixel `p` to the sharp edge whose pixel `q` is nearest to it, or `None` when
/// `q` gives no usable distance. The nearest pixel by its centre is not the best one: it can sit a
/// few pixels along the edge from the foot of the perpendicular and deeper than the edge, and the
/// further `p` is from the edge the further, so its distance ripples along the edge. So this takes
/// the smallest distance of the edge pixels around that foot, which is exact for a straight edge.
fn edge_distance(sharp: &SharpEdges, p: [usize; 2], q: [usize; 2], inside: bool) -> Option<f32> {
    let (px, py) = (p[0] as f32, p[1] as f32);
    let e = sharp.at(q[0], q[1])?;
    let mut best = e.distance(px - q[0] as f32, py - q[1] as f32, inside)?;
    // Where p's perpendicular meets q's line.
    let signed = e.offset + (px - q[0] as f32) * e.normal[0] + (py - q[1] as f32) * e.normal[1];
    let (fx, fy) = ((px - signed * e.normal[0]).round() as i64, (py - signed * e.normal[1]).round() as i64);
    for sy in (fy - FOOT_RADIUS).max(0)..=(fy + FOOT_RADIUS).min(sharp.h as i64 - 1) {
        for sx in (fx - FOOT_RADIUS).max(0)..=(fx + FOOT_RADIUS).min(sharp.w as i64 - 1) {
            if let Some(near) = sharp.at(sx as usize, sy as usize)
                && let Some(d) = near.distance(px - sx as f32, py - sy as f32, inside)
            {
                best = best.min(d);
            }
        }
    }
    Some(best)
}

/// [`dist_inside`] for chiselled bevels. A sharp edge pixel puts the edge where its coverage says,
/// at any angle, and the pixels near it measure to that edge ([`edge_distance`]), so the distance
/// ramps stay planar along an oblique edge instead of following the pixel staircase. The plain
/// distance (to the nearest empty pixel, less half a pixel) never falls short of the true one, so
/// it caps the result where an edge lies exactly on a pixel boundary and has no sharp pixel; a mask
/// without sharp pixels gives exactly [`dist_inside`]. Values past `reach` pixels are not refined.
fn bevel_dist_inside(s: &Map, sharp: &SharpEdges, reach: f32) -> Vec<f32> {
    let empty = covered(s, |a| a <= INSIDE_EPS);
    let (d, near) = edt_nearest(&empty, s.w, s.h);
    let at = |x: usize, y: usize| -> f32 {
        let i = y * s.w + x;
        if empty[i] {
            return -0.5;
        }
        if let Some(e) = sharp.get(i) {
            return e.offset;
        }
        let plain = d[i] - 0.5;
        if d[i] > reach + EDGE_MARGIN || sharp.is_empty() {
            return plain;
        }
        // The sharp pixel next to the nearest empty one that is nearest to this pixel.
        let (px, py) = (x as i64, y as i64);
        let (ox, oy) = ((near[i] as usize % s.w) as i64, (near[i] as usize / s.w) as i64);
        let q = (-1..=1)
            .flat_map(|dy| (-1..=1).map(move |dx| (ox + dx, oy + dy)))
            .filter(|&(x, y)| x >= 0 && y >= 0 && sharp.has(x as usize, y as usize))
            .min_by_key(|&(x, y)| (x - px) * (x - px) + (y - py) * (y - py));
        match q.and_then(|(x, y)| edge_distance(sharp, [px as usize, py as usize], [x as usize, y as usize], true)) {
            Some(refined) => refined.min(plain),
            None => plain,
        }
    };
    par_fill(s.w, s.h, 0.0f32, |y, row| {
        for (x, out) in row.iter_mut().enumerate() {
            *out = at(x, y);
        }
    })
}

/// [`dist_outside`] for chiselled bevels, with sharp edge pixels placed as in [`bevel_dist_inside`].
fn bevel_dist_outside(s: &Map, sharp: &SharpEdges, reach: f32) -> Vec<f32> {
    let inside = covered(s, |a| a > INSIDE_EPS);
    let (d, near) = edt_nearest(&inside, s.w, s.h);
    let at = |x: usize, y: usize| -> f32 {
        let i = y * s.w + x;
        if let Some(e) = sharp.get(i) {
            return -e.offset;
        }
        if inside[i] {
            return -0.5;
        }
        let q = near[i] as usize;
        let Some(e) = sharp.get(q) else { return d[i] - (s.v[q].min(1.0) - 0.5) };
        let (p, qp) = ([x, y], [q % s.w, q / s.w]);
        let measured = if d[i] <= reach + EDGE_MARGIN {
            edge_distance(sharp, p, qp, false)
        } else {
            e.distance(p[0] as f32 - qp[0] as f32, p[1] as f32 - qp[1] as f32, false)
        };
        measured.unwrap_or(d[i] - e.offset)
    };
    par_fill(s.w, s.h, 0.0f32, |y, row| {
        for (x, out) in row.iter_mut().enumerate() {
            *out = at(x, y);
        }
    })
}

/// Pixels past the reach that are still refined, so the value is right where the ramp ends.
const EDGE_MARGIN: f32 = 2.0;

/// Alpha above which a pixel belongs to the layer's shape (half an 8-bit step).
const INSIDE_EPS: f32 = 0.5 / 255.0;

/// The soft falloff of an effect of `size` pixels: two box passes of width `size` (a tent),
/// as Photoshop does. Fitted on psd-tools layer_effects.psd against the earlier Gaussian
/// (sigma = 0.4 × size): drop shadow coverage error 0.0064 → 0.0038, satin 0.014 → 0.002.
fn blur(m: &mut Map, size: f32) {
    if size.is_finite() {
        tent(m, size);
    }
}

/// Grows the shape by `r` pixels (anti-aliased), keeping the original soft edge. Distances use
/// the 5 × 5 chamfer metric, as for strokes: Photoshop's spread of a group drop shadow around an
/// ellipse (ag-psd group-drop-shadows) is round at 0° and 45° and falls short in between, peaking
/// near 15° where the chamfer metric overestimates most (5/255 lighter with an exact Euclidean
/// dilation, within 2/255 with the chamfer).
fn dilate(s: &Map, r: f32) -> Map {
    if r <= 0.0 {
        return s.clone();
    }
    let d = dist_outside_by(s, Metric::Chamfer);
    let mut out = s.clone();
    for (o, d) in out.v.iter_mut().zip(d) {
        // Partly covered pixels (distance sentinel < 0) grow by `r` from their own coverage, so a
        // 2 % spread doesn't make every anti-aliased edge pixel opaque (ag-psd drop shadow).
        *o = if d < 0.0 { (*o + r).min(1.0) } else { o.max((r + 0.5 - d).clamp(0.0, 1.0)) };
    }
    out
}

fn contour_lut(c: &Contour) -> Option<Vec<f32>> {
    match c {
        Contour::Linear => None,
        Contour::Custom { points, corners, .. } => Some(crate::adjust::contour_curve_lut(points, corners)),
    }
}

/// Satin's coverage transfer: the contour read from its far end, `|c(1 − v) − c(1)|`. Measured in
/// Photoshop 27.11: Cone and Cone − Inverted Satins are identical, and a Gaussian Satin runs under
/// the diagonal (0.36 at 0.5) while the same Gaussian on a glow or gloss runs over it (0.65).
/// `None` = identity.
pub fn satin_lut(c: &Contour) -> Option<Vec<f32>> {
    let lut = contour_lut(c)?;
    let end = lut.last().copied().unwrap_or(1.0);
    Some(lut.iter().rev().map(|v| (v - end).abs()).collect())
}

/// Interpolated lookup in a 0..=1 table (as the GPU's `lut()`).
fn lut_at(l: &[f32], v: f32) -> f32 {
    let x = v.clamp(0.0, 1.0) * (l.len() - 1) as f32;
    let i = x.floor() as usize;
    let j = (i + 1).min(l.len() - 1);
    l[i] + (l[j] - l[i]) * (x - i as f32)
}

/// A glow's coverage transfer: its contour applied over the Range (Photoshop maps the glow's
/// coverage `v` through `contour(min(v / range, 1))`, so the default 50 % doubles a blurred
/// edge's 0.5 to full strength). `None` = identity.
pub fn glow_lut(g: &Glow) -> Option<Vec<f32>> {
    ranged_lut(&g.contour, g.range)
}

/// A contour applied over a Range (`contour(min(v / range, 1))`); `None` = identity.
pub fn ranged_lut(contour: &Contour, range: f32) -> Option<Vec<f32>> {
    let base = contour_lut(contour);
    let range = if range.is_finite() { range.clamp(0.01, 1.0) } else { 1.0 };
    if range >= 0.999 {
        return base;
    }
    let n = 4096;
    Some(
        (0..n)
            .map(|k| {
                let v = (k as f32 / (n - 1) as f32 / range).min(1.0);
                base.as_ref().map_or(v, |l| lut_at(l, v))
            })
            .collect(),
    )
}

/// The bevel Contour element's height transfer: the contour alone. Unlike a glow's, its Range is
/// not a `v / range` stretch: that flattened a Linear contour at the default 50 % to a bevel half
/// as wide as Photoshop's (a real 18 px smooth inner bevel at 136 % depth: top-edge highlight gone
/// 15 px in, Photoshop's still 0.19 at 16 px), and dropping it lowers the bad pixels of every
/// contour oracle (bevel-inner-contour-ramp-peak 25.9 → 21.8 %, -double-ring-aa-r70 25.2 → 21.0 %,
/// bevel-pillow-chisel-hard-gloss-and-contour 11.4 → 7.2 %). What Range does instead is still open.
pub fn bevel_contour_lut(c: &photocraft_doc::BevelContour) -> Vec<f32> {
    contour_lut(&c.contour).unwrap_or_else(|| (0..4096).map(|k| k as f32 / 4095.0).collect())
}

fn apply_lut(m: Map, l: Option<Vec<f32>>) -> Map {
    match l {
        None => m,
        Some(l) => m.map(|v| lut_at(&l, v)),
    }
}

/// Deterministic per-pixel value in 0..=1, hashed from document coordinates
/// (the same hash as the GPU's `dissolve_noise`), so the speckle is stable
/// across renders, frames and tiles.
pub fn hash_noise(x: i64, y: i64) -> f32 {
    let mut h = (x as i32 as u32).wrapping_mul(0x8da6_b343) ^ (y as i32 as u32).wrapping_mul(0xd816_3841) ^ 0x9e37_79b9;
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    (h & 0xffff) as f32 / 65536.0
}

/// Effect noise (Photoshop's Noise slider): monochrome speckle in the coverage,
/// applied after the contour. `amount` is 0..=1; 0 leaves the map untouched.
fn noise(m: &mut Map, amount: f32, x0: i32, y0: i32) {
    if !amount.is_finite() || amount <= 0.0 {
        return;
    }
    for y in 0..m.h {
        for x in 0..m.w {
            let v = &mut m.v[y * m.w + x];
            if *v <= 0.0 {
                continue;
            }
            let n = hash_noise(i64::from(x0 + x as i32), i64::from(y0 + y as i32));
            *v = (*v + (n - 0.5) * 2.0 * amount).clamp(0.0, 1.0);
        }
    }
}

fn apply_contour(m: Map, c: &Contour) -> Map {
    match contour_lut(c) {
        None => m,
        Some(l) => m.map(|v| {
            let x = v.clamp(0.0, 1.0) * (l.len() - 1) as f32;
            let i = x.floor() as usize;
            let j = (i + 1).min(l.len() - 1);
            l[i] + (l[j] - l[i]) * (x - i as f32)
        }),
    }
}

fn rgb(c: &photocraft_color::Color) -> [f32; 3] {
    c.to_rgb()
}

/// Offset of an effect lit from `angle` at `distance` (shadow falls away from
/// the light), rounded to whole pixels as Photoshop does.
fn offset(angle: f32, distance: f32) -> (f32, f32) {
    let a = angle.to_radians();
    ((-a.cos() * distance).round(), (a.sin() * distance).round())
}

/// The lengths a 100 % gradient at `angle` spans in a `w` × `h` frame: the chord (Linear and
/// Reflected) and the ellipse-norm length (Radial, Angle, Diamond; its half is the radius).
pub fn gradient_units(angle: f32, w: f32, h: f32) -> (f32, f32) {
    let (s, c) = angle.to_radians().sin_cos();
    // Gradient length: the bounds' extent along the angle as an ellipse
    // norm (a unit gradient scaled to the bounds); fitted on psd-tools
    // gradient-styles.psd (cached Photoshop renderings of every style).
    let len = ((c * w).powi(2) + (s * h).powi(2)).sqrt().max(1.0);
    // Linear / Reflected span the chord of the bounds through their centre along the angle:
    // min(w / |cos|, h / |sin|). A 45° gradient on a 29 px square runs 41 px (psd-tools
    // shape-fx2), an 87° one on a 600 × 60 text line 60 px (layer_effects); axis-aligned angles
    // span the width / height.
    let chord = (w / c.abs().max(1e-6)).min(h / s.abs().max(1e-6)).max(1.0);
    (chord, len)
}

/// Gradient parameter `t` in `0..=1` for pixel centre `(x, y)` inside `bounds`.
#[allow(clippy::too_many_arguments)]
pub fn gradient_t(style: GradientStyle, angle: f32, scale: f32, reverse: bool, offset: (f32, f32), bounds: Rect, x: f32, y: f32) -> f32 {
    let w = bounds.width().max(1) as f32;
    let h = bounds.height().max(1) as f32;
    let cx = bounds.x0 as f32 + w / 2.0 + offset.0 * w;
    let cy = bounds.y0 as f32 + h / 2.0 + offset.1 * h;
    let (s, c) = angle.to_radians().sin_cos();
    let (dx, dy) = (x - cx, y - cy);
    // Distance along the gradient direction (y axis points down).
    let along = dx * c - dy * s;
    let across = dx * s + dy * c;
    let (chord, len) = gradient_units(angle, w, h);
    let (chord, len) = (chord * scale.max(1e-3), len * scale.max(1e-3));
    // Linear / Reflected sample the pixel's top-left corner, half a pixel before its centre
    // (with the whole-pixel end points of `fill_layout`: layer_effects' overlay 5.8 → 1.8/255,
    // shape-fx2 2.8 → 1.1/255, gradient-fill.psd 1.9 → 0.5/255).
    let corner = 0.5 * (c - s);
    let mut t = match style {
        GradientStyle::Linear => (along - corner) / chord + 0.5,
        GradientStyle::Reflected => ((along - corner) / (chord / 2.0)).abs(),
        GradientStyle::Radial => (dx * dx + dy * dy).sqrt() / (len / 2.0),
        GradientStyle::Diamond => (along.abs() + across.abs()) / (len / 2.0),
        // Clockwise sweep starting at the gradient angle.
        GradientStyle::Angle => ((angle.to_radians() - (-dy).atan2(dx)) / std::f32::consts::TAU).rem_euclid(1.0),
    };
    t = t.clamp(0.0, 1.0);
    if reverse { 1.0 - t } else { t }
}

/// Samples colour and opacity stops at `t`.
pub fn sample_gradient(g: &Gradient, t: f32) -> [f32; 4] {
    PreparedGradient::new(g).sample(t)
}

// Prepared only for one paint call: CMYK conversions must use that call's active profile.
struct PreparedGradient {
    color: Vec<(f32, [f32; 3])>,
    opacity: Vec<(f32, [f32; 3])>,
}

impl PreparedGradient {
    fn new(g: &Gradient) -> Self {
        Self { color: g.stops.iter().map(|(p, c)| (*p, rgb(c))).collect(), opacity: g.opacity_stops.iter().map(|(p, a)| (*p, [*a; 3])).collect() }
    }

    fn sample(&self, t: f32) -> [f32; 4] {
        let color = sample_stops(&self.color, t);
        let alpha = if self.opacity.is_empty() { 1.0 } else { sample_stops(&self.opacity, t)[0] };
        [color[0], color[1], color[2], alpha]
    }
}

fn sample_stops(stops: &[(f32, [f32; 3])], t: f32) -> [f32; 3] {
    match stops {
        [] => [t; 3],
        [s] => s.1,
        _ => {
            if t <= stops[0].0 {
                return stops[0].1;
            }
            for w in stops.windows(2) {
                if t <= w[1].0 {
                    let k = if w[1].0 > w[0].0 { (t - w[0].0) / (w[1].0 - w[0].0) } else { 0.0 };
                    return std::array::from_fn(|i| w[0].1[i] + (w[1].1[i] - w[0].1[i]) * k);
                }
            }
            stops[stops.len() - 1].1
        }
    }
}

/// Composite `color × coverage` into `dst` (both over the same rectangle).
fn paint(dst: &mut Buffer, m: &Map, color: impl Fn(usize) -> [f32; 4], blend: BlendMode, opacity: f32) {
    for (i, p) in dst.px.iter_mut().enumerate() {
        let a = m.v[i] * opacity;
        if a <= 0.0 {
            continue;
        }
        let c = color(i);
        *p = psblend::composite(blend, *p, [c[0], c[1], c[2], c[3] * a], 1.0);
    }
}

fn paint_color(dst: &mut Buffer, m: &Map, c: [f32; 3], blend: BlendMode, opacity: f32) {
    paint(dst, m, |_| [c[0], c[1], c[2], 1.0], blend, opacity);
}

/// Spread / choke of a soft effect: (dilation radius, blur width). The dilation is whole pixels
/// (`size × spread` rounded: a 2 % spread of a 5 px shadow doesn't grow it at all), the blur
/// keeps `size × (1 − spread)` (ag-psd effects 10.6 → 3.8 % of pixels off).
pub fn spread_split(size: f32, spread: f32) -> (f32, f32) {
    ((size * spread).round(), size * (1.0 - spread))
}

fn shadow_map(shape: &Map, s: &Shadow, light: &GlobalLight, inner: bool, origin: (i32, i32)) -> Map {
    let angle = if s.use_global_light { light.angle } else { s.angle };
    let (dx, dy) = offset(angle, s.distance);
    let src = if inner { shape.clone().map(|a| 1.0 - a) } else { shape.clone() };
    let mut m = src.shifted(dx, dy, if inner { 1.0 } else { 0.0 });
    let (r, bw) = spread_split(s.size, s.spread);
    m = dilate(&m, r);
    blur(&mut m, bw);
    let mut m = apply_contour(m, &s.contour);
    noise(&mut m, s.noise, origin.0, origin.1);
    if inner {
        for (v, a) in m.v.iter_mut().zip(&shape.v) {
            *v *= a;
        }
    }
    m
}

fn glow_map(shape: &Map, g: &Glow, inner: bool, origin: (i32, i32)) -> Map {
    // An inner glow is built from the edge inwards; a centre-source glow is its inverse, taken
    // after the contour and Range (Photoshop 27.11, measured: the centre glow is exactly
    // 1 − the edge glow, for both techniques).
    let src = if inner { shape.clone().map(|a| 1.0 - a) } else { shape.clone() };
    let mut m = match g.technique {
        GlowTechnique::Precise => {
            // Coverage falls linearly from 0.5 at the edge to 0 at `size + 1` px (centre to
            // centre), solid within the spread / choke: after the default 50 % Range that is
            // Photoshop's 1 − D / (size + 1) beyond the solid part (measured: outer and inner,
            // sizes 10 and 20, spread 0 and 40 %, choke 0 and 30 %: within 1 level; other Range
            // values are unmeasured). The distances here are 0.5 px short of centre to centre.
            let d = if inner { dist_inside(shape) } else { dist_outside(&src) };
            let solid = g.size * g.spread;
            let soft = (g.size + 1.0 - solid).max(1e-3);
            let mut m = Map::new(shape.w, shape.h, 0.0);
            for (o, d) in m.v.iter_mut().zip(d) {
                let d = d + 0.5;
                *o = if d <= solid { 1.0 } else { 0.5 * (1.0 - (d - solid) / soft).clamp(0.0, 1.0) };
            }
            m
        }
        GlowTechnique::Softer => {
            let (r, bw) = spread_split(g.size, g.spread);
            let mut m = dilate(&src, r);
            blur(&mut m, bw);
            m
        }
    };
    m = apply_lut(m, glow_lut(g));
    if inner && g.source == GlowSource::Center {
        m = m.map(|v| 1.0 - v);
    }
    noise(&mut m, g.noise, origin.0, origin.1);
    if inner {
        for (v, a) in m.v.iter_mut().zip(&shape.v) {
            *v *= a;
        }
    }
    m
}

/// Paints a pattern (looked up in `patterns`) through coverage `m`; missing patterns paint nothing.
#[allow(clippy::too_many_arguments)]
fn paint_pattern(
    dst: &mut Buffer,
    m: &Map,
    patterns: &PreparedPatterns<'_>,
    name: &str,
    id: &str,
    place: Placement,
    big: Rect,
    blend: BlendMode,
    opacity: f32,
) {
    let Some(tile) = patterns.get(id, name) else { return };
    let w = big.width() as usize;
    paint(
        dst,
        m,
        |i| {
            let (x, y) = (f64::from(big.x0 + (i % w) as i32) + 0.5, f64::from(big.y0 + (i / w) as i32) + 0.5);
            let (u, v) = place.map(x, y);
            tile.sample(u, v)
        },
        blend,
        opacity,
    );
}

#[allow(clippy::too_many_arguments)]
fn paint_fx(
    dst: &mut Buffer,
    m: &Map,
    p: &FxPaint,
    shape_bounds: Rect,
    anchor: (f64, f64),
    big: Rect,
    blend: BlendMode,
    opacity: f32,
    patterns: &PreparedPatterns<'_>,
) {
    match p {
        FxPaint::Color(c) => paint_color(dst, m, rgb(c), blend, opacity),
        FxPaint::Gradient(g) => {
            let prepared = PreparedGradient::new(g);
            let (angle, scale, offset) = crate::fill_layout::gradient_layout(g.style, g.angle, g.scale, g.offset, shape_bounds);
            let w = big.width() as usize;
            paint(
                dst,
                m,
                |i| {
                    let (x, y) = ((big.x0 + (i % w) as i32) as f32 + 0.5, (big.y0 + (i / w) as i32) as f32 + 0.5);
                    prepared.sample(gradient_t(g.style, angle, scale, g.reverse, offset, shape_bounds, x, y))
                },
                blend,
                opacity,
            )
        }
        FxPaint::Pattern { name, id, scale } => {
            paint_pattern(dst, m, patterns, name, id, Placement::anchored(anchor, true, (0.0, 0.0), *scale, 0.0), big, blend, opacity)
        }
    }
}

/// The opacity gain of a gradient glow: it is opaque from strength `range²` on (see [`paint_glow`]).
pub fn glow_gradient_gain(range: f32) -> f32 {
    let r = if range.is_finite() { range.clamp(0.01, 1.0) } else { 1.0 };
    1.0 / (r * r)
}

/// A glow's paint over its map `m`. A gradient runs along the glow rather than across the
/// canvas: Photoshop colours the glow where its (ranged, contoured) strength is `v` with the
/// gradient at `1 - v`, so the first stop hugs the edge, and the glow is opaque from `v = range²`
/// on (photoshop corpus outer-glow-gradient.psd: 1.4/255 mean, against bands across the canvas;
/// psd-tools layer_params.psd). Glows have no gradient angle, style or Reverse.
#[allow(clippy::too_many_arguments)]
fn paint_glow(dst: &mut Buffer, m: &Map, g: &Glow, shape_bounds: Rect, anchor: (f64, f64), big: Rect, patterns: &PreparedPatterns<'_>) {
    let FxPaint::Gradient(gradient) = &g.paint else {
        return paint_fx(dst, m, &g.paint, shape_bounds, anchor, big, g.common.blend, g.common.opacity, patterns);
    };
    let prepared = PreparedGradient::new(gradient);
    let gain = glow_gradient_gain(g.range);
    let strength = Map { w: m.w, h: m.h, v: m.v.iter().map(|v| (v * gain).min(1.0)).collect() };
    paint(dst, &strength, |i| prepared.sample(1.0 - m.v.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0)), g.common.blend, g.common.opacity);
}

/// A box width as the blurs use it: at least 1, at most [`MAX_REACH`] (an effect never reaches
/// further, see [`margin`]), and 1 for NaN. A size or softness from a command or a file can be
/// anything, and an unbounded width sizes the kernel and its buffers (#1543).
fn box_width(w: f32) -> f32 {
    if w.is_nan() { 1.0 } else { w.clamp(1.0, MAX_REACH) }
}

/// Normalised weights of a centred box of (fractional) width `w`: tap `i` gets the overlap of
/// `[i - 0.5, i + 0.5]` with `[-w/2, w/2]`.
fn box_weights(w: f32) -> Vec<f32> {
    let w = box_width(w);
    let half = w / 2.0;
    let r = (half - 0.5).ceil().max(0.0) as i64;
    let v: Vec<f32> = (-r..=r).map(|i| ((i as f32 + 0.5).min(half) - (i as f32 - 0.5).max(-half)).max(0.0)).collect();
    let sum: f32 = v.iter().sum();
    v.iter().map(|x| x / sum).collect()
}

/// The 1D kernel of two box passes of width `w` (a tent of half-width ≈ `w`): `(radius, weights)`.
/// Photoshop's bevel height maps are this blur of the shape (fitted on the corpus: a smooth
/// inner bevel of size 5 matches within 0.7 % mean, size 41 within 2.6 %).
pub fn tent_kernel(w: f32) -> (i32, Vec<f32>) {
    let b = box_weights(w);
    let n = b.len() * 2 - 1;
    let mut t = vec![0.0f32; n];
    for (i, x) in b.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            t[i + j] += x * y;
        }
    }
    ((n as i32 - 1) / 2, t)
}

/// Box geometry for a (fractional) width: (`r`, end-tap weight `f`, 1 / width) — the
/// [`box_weights`] taps are `r - 1` full ones each side of the centre plus the two end taps at `f`.
fn box_geom(bw: f32) -> (i64, f64, f64) {
    let bw = box_width(bw);
    let half = bw / 2.0;
    let r = (half - 0.5).ceil().max(0.0) as i64;
    let f = f64::from((half - (r as f32 - 0.5)).clamp(0.0, 1.0));
    (r, f, 1.0 / f64::from(bw))
}

/// One box pass over `src` (zero outside it) evaluated at `x0 .. x0 + dst.len()`, as a running
/// sum (O(1) per sample whatever the width).
fn box_line(src: &[f64], x0: i64, dst: &mut [f64], bw: f32) {
    let n = src.len() as i64;
    let (r, f, norm) = box_geom(bw);
    let at = |i: i64| if i >= 0 && i < n { src[i as usize] } else { 0.0 };
    if r == 0 {
        for (k, d) in dst.iter_mut().enumerate() {
            *d = at(x0 + k as i64);
        }
        return;
    }
    let mut inner: f64 = (x0 - (r - 1)..=x0 + (r - 1)).map(at).sum();
    for (k, d) in dst.iter_mut().enumerate() {
        let x = x0 + k as i64;
        *d = (inner + f * (at(x - r) + at(x + r))) * norm;
        inner += at(x + r) - at(x - r + 1);
    }
}

/// Two box passes along a line, exactly the convolution with [`tent_kernel`] (zero outside): the
/// first pass is evaluated `r` samples beyond each end so the second sees what lies there.
fn tent_line(src: &[f32], dst: &mut [f32], bw: f32) {
    let (r, _, _) = box_geom(bw);
    let s: Vec<f64> = src.iter().map(|v| f64::from(*v)).collect();
    let mut mid = vec![0.0f64; src.len() + 2 * r as usize];
    box_line(&s, -r, &mut mid, bw);
    let mut out = vec![0.0f64; src.len()];
    // `mid[k]` holds position k - r: output position p reads index p + r.
    box_line(&mid, r, &mut out, bw);
    for (d, v) in dst.iter_mut().zip(out) {
        *d = v as f32;
    }
}

/// Rows of a `w`-wide row-major buffer, each through `f(src_row, dst_row)`, in parallel on
/// scoped OS threads. Not rayon: maps can be built from inside a rayon tile, where a waiting
/// worker steals other tiles. Map builds no longer block on each other (#276), but the blur
/// keeps out of the tile pool.
fn rows_par(src: &[f32], dst: &mut [f32], w: usize, f: impl Fn(&[f32], &mut [f32]) + Sync) {
    if w == 0 {
        return;
    }
    let threads =
        if cfg!(target_arch = "wasm32") || src.len() < 1 << 16 { 1 } else { std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 16) };
    if threads == 1 {
        dst.chunks_mut(w).zip(src.chunks(w)).for_each(|(d, s)| f(s, d));
        return;
    }
    let rows = (src.len() / w).div_ceil(threads);
    std::thread::scope(|sc| {
        for (d, s) in dst.chunks_mut(rows * w).zip(src.chunks(rows * w)) {
            let f = &f;
            sc.spawn(move || d.chunks_mut(w).zip(s.chunks(w)).for_each(|(d, s)| f(s, d)));
        }
    });
}

fn transpose_map(v: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; v.len()];
    const B: usize = 32;
    for by in (0..h).step_by(B) {
        for bx in (0..w).step_by(B) {
            for y in by..(by + B).min(h) {
                for x in bx..(bx + B).min(w) {
                    out[x * h + y] = v[y * w + x];
                }
            }
        }
    }
    out
}

/// Two box passes of width `w` along each axis (the tent [`tent_kernel`] describes, which the GPU
/// convolves directly).
fn tent(m: &mut Map, w: f32) {
    if tent_kernel(w).0 == 0 || m.v.is_empty() {
        return;
    }
    tent_fast(m, w);
}

/// [`tent`] as a direct convolution with [`tent_kernel`] (the GPU's way; tests compare).
#[cfg(test)]
fn tent_direct(m: &mut Map, w: f32) {
    {
        let (r, k) = tent_kernel(w);
        let (mw, mh) = (m.w as i64, m.h as i64);
        let mut tmp = vec![0.0f32; m.v.len()];
        for y in 0..mh {
            for x in 0..mw {
                let mut acc = 0.0;
                for (i, kv) in k.iter().enumerate() {
                    let xx = x + i as i64 - r as i64;
                    if xx >= 0 && xx < mw {
                        acc += m.v[(y * mw + xx) as usize] * kv;
                    }
                }
                tmp[(y * mw + x) as usize] = acc;
            }
        }
        for y in 0..mh {
            for x in 0..mw {
                let mut acc = 0.0;
                for (i, kv) in k.iter().enumerate() {
                    let yy = y + i as i64 - r as i64;
                    if yy >= 0 && yy < mh {
                        acc += tmp[(yy * mw + x) as usize] * kv;
                    }
                }
                m.v[(y * mw + x) as usize] = acc;
            }
        }
    }
}

fn tent_fast(m: &mut Map, w: f32) {
    let (mw, mh) = (m.w, m.h);
    let twice = |s: &[f32], d: &mut [f32]| tent_line(s, d, w);
    let mut a = vec![0.0f32; m.v.len()];
    rows_par(&m.v, &mut a, mw, twice);
    let t = transpose_map(&a, mw, mh);
    let mut b = vec![0.0f32; t.len()];
    rows_par(&t, &mut b, mh, twice);
    m.v = transpose_map(&b, mh, mw);
}

/// Where a bevel's maps paint: inside the shape (with the layer), outside it (onto the
/// backdrop), or both (emboss styles: maps 0–1 inside, 2–3 outside).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BevelPaint {
    Inner,
    Outer,
    Both,
}

/// Height below which an outer bevel pixel counts as off the bevel (beyond the blur's reach):
/// above float noise of the running-sum blur (~3e-7), far below anything that shades.
pub const BEVEL_H_EPS: f32 = 1e-5;

/// Bevel geometry shared by the CPU maps and the GPU programs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BevelGeom {
    pub paint: BevelPaint,
    /// Box width of the smooth height blur (size; half the size, rounded up, for emboss styles).
    pub width: f32,
    /// Height-map gradient scale: depth × width (smooth) or depth × size (chisel), signed by
    /// direction.
    pub depth: f32,
    /// Pillow emboss: the outside slopes face the other way.
    pub pillow: bool,
    /// Chisel Soft: extra tent width over the chiselled height map.
    pub chisel_soft: f32,
}

pub fn bevel_geom(b: &Bevel) -> BevelGeom {
    let size = b.size.max(1.0);
    let emboss = matches!(b.style, BevelStyle::Emboss | BevelStyle::PillowEmboss);
    let paint = match b.style {
        BevelStyle::OuterBevel => BevelPaint::Outer,
        BevelStyle::Emboss | BevelStyle::PillowEmboss => BevelPaint::Both,
        BevelStyle::InnerBevel | BevelStyle::StrokeEmboss => BevelPaint::Inner,
    };
    // Emboss styles straddle the edge with half the size, rounded up to whole pixels: a 41 px
    // emboss blurs over 21 (psd-tools layer_effects: 20.5 left narrow letter parts up to
    // 5.7/255 dark, 21 is within 2.2).
    let width = if emboss { (size / 2.0).ceil() } else { size };
    let smooth = b.technique == BevelTechnique::Smooth;
    let sign = if b.up { 1.0 } else { -1.0 };
    BevelGeom {
        paint,
        width,
        // A bevel rises `depth × width / 2` px: Smooth with a slope falling from `depth` at the
        // edge (the tent), a chisel with the constant slope of its ramp (see `bevel_chisel_h`).
        depth: b.depth * if smooth { width } else { size / 2.0 } * sign,
        pillow: b.style == BevelStyle::PillowEmboss,
        chisel_soft: if b.technique == BevelTechnique::ChiselSoft { (size / 4.0).max(1.0) } else { 0.0 },
    }
}

/// Bevel height map in `0..=1`. Smooth: the shape blurred by two box passes of the bevel
/// width. Chisel: linear ramps of the distance to the edge (size wide; emboss styles straddle the
/// edge), slightly blurred for Chisel Soft. The distance puts the edge by the anti-aliased
/// coverage of its pixels ([`bevel_dist_inside`]), so a facet is planar at any angle. Soften blurs
/// the result.
fn bevel_height(shape: &Map, b: &Bevel, g: &BevelGeom, tex: &TextureCtx, patterns: &PreparedPatterns<'_>) -> Map {
    let size = b.size.max(1.0);
    let mut h = if b.technique == BevelTechnique::Smooth {
        let mut h = shape.clone();
        tent(&mut h, g.width);
        h
    } else {
        let sharp = sharp_edges(shape);
        let (din, dout) = (bevel_dist_inside(shape, &sharp, size), bevel_dist_outside(shape, &sharp, size));
        let mut h = Map::new(shape.w, shape.h, 0.0);
        let soft = g.chisel_soft > 0.0;
        for i in 0..h.v.len() {
            h.v[i] = bevel_chisel_h(g.paint, size, din[i], dout[i], soft);
        }
        // Inner and outer Chisel Soft are not blurred in Photoshop (their facets equal Chisel
        // Hard's across a straight edge); emboss styles keep the blur until measured.
        if soft && g.paint == BevelPaint::Both {
            tent(&mut h, g.chisel_soft);
        }
        h
    };
    // Contour element: the height profile through the contour.
    if let Some(c) = &b.contour {
        h = apply_lut(h, Some(bevel_contour_lut(c)));
    }
    // Texture element: the pattern's luminance as extra height, scaled so a full-contrast step
    // at 100 % depth slopes like the bevel at its depth.
    if let Some(t) = &b.texture
        && let Some(tile) = patterns.get(&t.id, &t.name)
    {
        let unit = if b.depth.abs() > 1e-6 { (g.depth / b.depth).abs().max(1e-3) } else { g.width.max(1.0) };
        let place = Placement::anchored(tex.anchor, t.link, t.phase, t.scale, 0.0);
        let k = t.depth / unit;
        let (x0, y0) = (tex.rect.x0, tex.rect.y0);
        for y in 0..h.h {
            for x in 0..h.w {
                let (u, v) = place.map(f64::from(x0 + x as i32) + 0.5, f64::from(y0 + y as i32) + 0.5);
                let p = tile.sample(u, v);
                let l = (0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) * p[3];
                h.v[y * h.w + x] += k * if t.invert { 1.0 - l } else { l };
            }
        }
    }
    if b.soften >= 1.0 {
        tent(&mut h, b.soften);
    }
    h
}

/// Where a bevel texture tiles: the maps' rectangle, the document's patterns and the linked
/// origin (the effects reference point, else the layer's top-left).
pub struct TextureCtx<'a> {
    pub rect: Rect,
    pub patterns: &'a [Pattern],
    pub anchor: (f64, f64),
}

/// Chiselled height from the inside / outside distances (keep in sync with `fs_mbevelh`).
///
/// Photoshop 27.11, measured across straight edges (sizes 10 and 20, depth 100 and 300): a chisel
/// facet has the constant slope `depth × size / (2 (size + 1))`, i.e. one height step of
/// `1 / (size + 1)` per pixel at the `depth × size / 2` scale of [`bevel_geom`]. Inner Hard's ramp
/// runs from half a pixel beyond the edge to `size + ½` (the facet is flat right up to the edge
/// pixel); Inner Soft starts a pixel further out, so its edge pixel is steeper; Outer Hard ends a
/// pixel further out. All within 1 level of Photoshop on those profiles.
pub fn bevel_chisel_h(paint: BevelPaint, size: f32, din: f32, dout: f32, soft: bool) -> f32 {
    let ramp = size + 1.0;
    match paint {
        BevelPaint::Outer if !soft => {
            let top = (size + 2.0) / ramp;
            if dout > 0.0 { ((size + 1.5 - dout) / ramp).clamp(0.0, top) } else { top }
        }
        // Outer Soft: Outer Hard's ramp a pixel shorter, with a steeper edge pixel.
        BevelPaint::Outer => {
            if dout > 0.0 {
                1.0 - ((dout + 0.5) / ramp).clamp(0.0, 1.0)
            } else {
                (size + 2.0) / ramp
            }
        }
        BevelPaint::Both => {
            let half = size / 2.0 + 0.5;
            if din > 0.0 { 0.5 + 0.5 * (din / half).clamp(0.0, 1.0) } else { 0.5 - 0.5 * (dout / half).clamp(0.0, 1.0) }
        }
        BevelPaint::Inner if soft => {
            if din > 0.0 {
                ((din + 1.5) / ramp).clamp(0.0, (size + 2.0) / ramp)
            } else {
                0.0
            }
        }
        BevelPaint::Inner => {
            if din > 0.0 {
                ((din + 0.5) / ramp).clamp(0.0, 1.0)
            } else {
                0.0
            }
        }
    }
}

/// Bevel highlight and shadow maps: `[hi, sh]` (inner or outer) or `[hi, sh, hi_out, sh_out]`
/// (emboss styles).
fn bevel_maps(shape: &Map, b: &Bevel, light: &GlobalLight, tex: &TextureCtx, patterns: &PreparedPatterns<'_>) -> (Vec<Map>, BevelPaint) {
    let g = bevel_geom(b);
    let hmap = bevel_height(shape, b, &g, tex, patterns);
    let (angle, altitude) = if b.use_global_light { (light.angle, light.altitude) } else { (b.angle, b.altitude) };
    let (sa, ca) = angle.to_radians().sin_cos();
    let (se, ce) = altitude.to_radians().sin_cos();
    let light_v = [ca * ce, -sa * ce, se];
    // Region: 0 = inside (× the shape's alpha), 1 = under the layer's edge and outside.
    // A pixel is on the bevel where its height is raised. Emboss styles also shade the last pixel
    // past the blur's reach (a raised 4-neighbour: it still slopes; psd-tools layer_effects'
    // emboss shadow runs one row past it), outer bevels don't (Photoshop oracle
    // bevel-outer-smooth).
    // Outer chisel bevels too: the pixel where their ramp reaches zero still slopes (Photoshop
    // 27.11 shades it, half strength, at `size + 1½` px out).
    let reach_past = g.paint == BevelPaint::Both || (g.paint == BevelPaint::Outer && b.technique != BevelTechnique::Smooth);
    let on_bevel = |x: i64, y: i64| {
        hmap.get(x, y) > BEVEL_H_EPS || (reach_past && [(1, 0), (-1, 0), (0, 1), (0, -1)].iter().any(|(dx, dy)| hmap.get(x + dx, y + dy) > BEVEL_H_EPS))
    };
    let shade_into = |depth: f32, region_kind: u8| -> (Map, Map) {
        let (mut hi, mut sh) = (Map::new(shape.w, shape.h, 0.0), Map::new(shape.w, shape.h, 0.0));
        for y in 0..shape.h as i64 {
            for x in 0..shape.w as i64 {
                let i = y as usize * shape.w + x as usize;
                let gx = (hmap.get(x + 1, y) - hmap.get(x - 1, y)) * 0.5 * depth;
                let gy = (hmap.get(x, y + 1) - hmap.get(x, y - 1)) * 0.5 * depth;
                let n = [-gx, -gy, 1.0];
                let len = (n[0] * n[0] + n[1] * n[1] + 1.0).sqrt();
                let shade = (n[0] * light_v[0] + n[1] * light_v[1] + n[2] * light_v[2]) / len;
                let region = if region_kind == 0 { shape.v[i] } else { f32::from(shape.v[i] < 1.0 - INSIDE_EPS && on_bevel(x, y)) };
                let k = shade - se;
                if k > 0.0 {
                    hi.v[i] = (k / (1.0 - se).max(1e-3)).clamp(0.0, 1.0) * region;
                } else {
                    sh.v[i] = (-k / se.max(1e-3)).clamp(0.0, 1.0) * region;
                }
            }
        }
        (apply_contour(hi, &b.gloss_contour), apply_contour(sh, &b.gloss_contour))
    };
    let maps = match g.paint {
        BevelPaint::Inner | BevelPaint::Outer => {
            let (hi, sh) = shade_into(g.depth, u8::from(g.paint == BevelPaint::Outer));
            vec![hi, sh]
        }
        BevelPaint::Both => {
            // One map per colour over the whole bevel: the inside half where the shape is (× its
            // alpha), the outside half (pillow: facing the other way) for the rest of each pixel.
            // A 1 % edge pixel of a pillow emboss takes the outside shading (Photoshop oracle
            // bevel-pillow-smooth), a 75 % one of an emboss mostly the inside (layer_effects).
            let (hi, sh) = shade_into(g.depth, 0);
            let (ho, so) = shade_into(if g.pillow { -g.depth } else { g.depth }, 1);
            let mix = |a: Map, b: Map| -> Map {
                let v = a.v.iter().zip(&b.v).zip(&shape.v).map(|((x, y), s)| if *s > INSIDE_EPS { *x } else { 0.0 } + y * (1.0 - s.min(1.0))).collect();
                Map { w: a.w, h: a.h, v }
            };
            vec![mix(hi, ho), mix(sh, so)]
        }
    };
    (maps, g.paint)
}

/// Effect maps derived from a layer's shape (its alpha), computed once over the layer's whole
/// region and cropped per render tile. They depend only on the shape, the effect settings and the
/// global light, never on the backdrop, so they can be cached across tiles and edits.
pub struct FxMaps {
    rect: Rect,
    shape: Map,
    /// Per enabled effect (in `items` order): the maps it paints through.
    per: Vec<Vec<Map>>,
    bevel_paint: Vec<BevelPaint>,
    din: Option<Vec<f32>>,
    /// Stroke outside distance below [`STROKE_CHAMFER_MAX`]: the fitted 5 × 5 chamfer.
    dout: Option<Vec<f32>>,
    /// Stroke outside distance at or above [`STROKE_CHAMFER_MAX`]: the exact Euclidean transform.
    dout_exact: Option<Vec<f32>>,
    /// Shape layers: chamfer distance outside the vector outline (from local coverage).
    vdout: Option<Vec<f32>>,
    /// Shape layers: exact Euclidean distance outside the vector outline (from local coverage).
    vdout_exact: Option<Vec<f32>>,
    /// `shape` joins the alpha with a filled shape's outline (`crate::effect_outline`).
    outline: bool,
    /// Frames of gradient strokes by outside width (bits): see [`stroke_frame`].
    frames: Vec<(u32, Rect)>,
}

impl FxMaps {
    /// The frame of a gradient stroke `st` of this layer (see [`stroke_frame`]), if built.
    pub fn stroke_frame(&self, st: &photocraft_doc::StrokeFx) -> Option<Rect> {
        self.frames.iter().find(|f| f.0 == stroke_widths(st).1.to_bits()).map(|f| f.1)
    }
}

impl FxMaps {
    /// Approximate heap size (for cache budgeting).
    pub fn bytes(&self) -> usize {
        let n = self.shape.v.len();
        let maps: usize = self.per.iter().map(Vec::len).sum();
        (1 + maps
            + usize::from(self.din.is_some())
            + usize::from(self.dout.is_some())
            + usize::from(self.dout_exact.is_some())
            + usize::from(self.vdout.is_some())
            + usize::from(self.vdout_exact.is_some()))
            * n
            * 4
    }

    fn crop_vec(&self, v: &[f32], to: Rect, fill: f32) -> Vec<f32> {
        let (w, h) = (to.width() as usize, to.height() as usize);
        let mut out = vec![fill; w * h];
        let r = self.rect;
        let inter = r.intersect(&to);
        if inter.is_empty() {
            return out;
        }
        let sw = r.width() as usize;
        for y in inter.y0..inter.y1 {
            let s0 = (y - r.y0) as usize * sw + (inter.x0 - r.x0) as usize;
            let d0 = (y - to.y0) as usize * w + (inter.x0 - to.x0) as usize;
            let n = inter.width() as usize;
            out[d0..d0 + n].copy_from_slice(&v[s0..s0 + n]);
        }
        out
    }

    fn crop(&self, m: &Map, to: Rect, fill: f32) -> Map {
        Map { w: to.width() as usize, h: to.height() as usize, v: self.crop_vec(&m.v, to, fill) }
    }
}

/// The distance the two satin copies are offset by. Photoshop draws Distance 0 as 1 px (an
/// edge band of 2/size, measured in Photoshop 27.11), not as two copies cancelling out.
pub fn satin_distance(s: &photocraft_doc::effects::Satin) -> f32 {
    s.distance.max(1.0)
}

fn satin_map(shape: &Map, s: &photocraft_doc::effects::Satin) -> Map {
    let (dx, dy) = offset(s.angle, satin_distance(s));
    let mut a = shape.shifted(dx, dy, 0.0);
    let mut b = shape.shifted(-dx, -dy, 0.0);
    // Two box passes of the satin size (fitted: 0.2 % mean error on psd-tools layer_effects).
    tent(&mut a, s.size);
    tent(&mut b, s.size);
    let mut m = Map { w: shape.w, h: shape.h, v: a.v.iter().zip(&b.v).map(|(x, y)| (x - y).abs()).collect() };
    if s.invert {
        m = m.map(|v| 1.0 - v);
    }
    let mut m = apply_lut(m, satin_lut(&s.contour));
    for (v, a) in m.v.iter_mut().zip(&shape.v) {
        *v *= a;
    }
    m
}

/// Build every effect map for `layer` from its alpha `shape` over `rect`.
pub fn build_maps(layer: &Layer, shape: Vec<f32>, rect: Rect, light: &GlobalLight, tex: &TextureCtx) -> FxMaps {
    let prepared = PreparedPatterns::new(tex.patterns, PREPARED_PATTERN_BYTES);
    build_maps_prepared(layer, shape, rect, light, tex, &prepared)
}

pub(crate) fn build_maps_prepared(
    layer: &Layer,
    shape: Vec<f32>,
    rect: Rect,
    light: &GlobalLight,
    tex: &TextureCtx,
    patterns: &PreparedPatterns<'_>,
) -> FxMaps {
    let (w, h) = (rect.width() as usize, rect.height() as usize);
    let shape = Map { w, h, v: shape };
    let items: Vec<&Effect> = layer.effects.items.iter().filter(|e| e.enabled()).collect();
    let mut bevel_paint = vec![BevelPaint::Inner; items.len()];
    let per: Vec<Vec<Map>> = items
        .iter()
        .enumerate()
        .map(|(i, e)| match e {
            Effect::DropShadow(s) => vec![shadow_map(&shape, s, light, false, (rect.x0, rect.y0))],
            Effect::InnerShadow(s) => vec![shadow_map(&shape, s, light, true, (rect.x0, rect.y0))],
            Effect::OuterGlow(g) => vec![glow_map(&shape, g, false, (rect.x0, rect.y0))],
            Effect::InnerGlow(g) => vec![glow_map(&shape, g, true, (rect.x0, rect.y0))],
            Effect::Satin(s) => vec![satin_map(&shape, s)],
            Effect::BevelEmboss(b) => {
                let (maps, paint) = bevel_maps(&shape, b, light, tex, patterns);
                bevel_paint[i] = paint;
                maps
            }
            _ => Vec::new(),
        })
        .collect();
    let has_stroke = items.iter().any(|e| matches!(e, Effect::Stroke(_)));
    let vector_shape = matches!(layer.content, photocraft_doc::LayerContent::Shape(_));
    // A stroke's outer edge keeps the 5 × 5 chamfer while its outside width is under
    // `STROKE_CHAMFER_MAX` — Photoshop matches the chamfer's sub-pixel edge seeding at small
    // offsets — and uses the exact Euclidean transform at or above it, where the chamfer's octagon
    // facets a wide stroke (#3646). Inside stroke distances always keep the chamfer. Each outer
    // field is built only when a stroke of that width is present; the compositor picks per stroke,
    // so one layer may hold both a narrow and a wide stroke.
    let out_widths: Vec<f32> =
        items.iter().filter_map(|e| if let Effect::Stroke(st) = e { Some(stroke_widths(st).1) } else { None }).filter(|w| *w > 0.0).collect();
    let chamfer_out = out_widths.iter().any(|&w| !stroke_outside_exact(w));
    let exact_out = out_widths.iter().any(|&w| stroke_outside_exact(w));
    let (din, dout, dout_exact) = if has_stroke {
        (
            Some(dist_inside_by(&shape, Metric::Chamfer)),
            chamfer_out.then(|| dist_outside_by(&shape, Metric::Chamfer)),
            exact_out.then(|| dist_outside_by(&shape, Metric::Euclidean)),
        )
    } else {
        (None, None, None)
    };
    // Without an outline (unfilled shapes), estimate the vector outline from local coverage; it
    // picks the chamfer or the exact transform the same way.
    let outline = crate::effect_outline(layer).is_some();
    let (vdout, vdout_exact) = if has_stroke && vector_shape && !outline {
        let lc = local_coverage(&shape);
        (chamfer_out.then(|| dist_outside_by(&lc, Metric::Chamfer)), exact_out.then(|| dist_outside_by(&lc, Metric::Euclidean)))
    } else {
        (None, None)
    };
    let mut frames: Vec<(u32, Rect)> = Vec::new();
    for e in &items {
        if let Effect::Stroke(st) = e
            && matches!(st.paint, FxPaint::Gradient(_))
        {
            let out_w = stroke_widths(st).1;
            if !frames.iter().any(|f| f.0 == out_w.to_bits()) {
                frames.push((out_w.to_bits(), stroke_frame(&shape, out_w, rect)));
            }
        }
    }
    FxMaps { rect, shape, per, bevel_paint, din, dout, dout_exact, vdout, vdout_exact, outline, frames }
}

/// Far outside any shape (distance fill for cropped distance fields).
const FAR: f32 = 1.0e9;

/// Keep an exterior effect behind the layer's shape when Fill is partly or wholly
/// transparent. The visible part beneath a fully opaque fill is already covered by the
/// layer, so avoid attenuating anti-aliased edges twice in that case.
fn knock_out_exterior(map: &mut Map, shape: &Map, fill_opacity: f32) {
    let see_through = 1.0 - fill_opacity.clamp(0.0, 1.0);
    for (coverage, alpha) in map.v.iter_mut().zip(&shape.v) {
        *coverage *= 1.0 - alpha * see_through;
    }
}

/// How far the layer lets a knocked-out drop shadow show beneath its own shape (Layer Knocks Out
/// Drop Shadow), 0 = covered, 1 = knocked out wholly. In a normal mode an opaque layer covers the
/// shadow anyway, so only see-through fill knocks it out (and anti-aliased edges are not
/// attenuated twice). In any other mode the layer blends with what's beneath it, shadow and all:
/// an Exclusion or Difference layer would invert it, Multiply darken by it, Screen show it as
/// grey. Photoshop keeps it out from under the shape, so here it's knocked out wholly.
pub fn shadow_see_through(layer: &Layer) -> f32 {
    if matches!(layer.blend, BlendMode::Normal | BlendMode::Dissolve | BlendMode::PassThrough) { 1.0 - layer.fill_opacity.clamp(0.0, 1.0) } else { 1.0 }
}

/// Blend Interior Effects as Group off (Photoshop's default): the layer's mode applies to its
/// own pixels only, and the interior effects paint over the blended result. The layer is merged
/// into the backdrop before they paint, so an Overlay colour overlay on a black Color-mode layer
/// tints the grey beneath it (#3996). Normal-like modes give the same result either way.
pub fn premerges_layer(layer: &Layer) -> bool {
    !layer.advanced.blend_interior && !matches!(layer.blend, BlendMode::Normal | BlendMode::PassThrough | BlendMode::Dissolve)
}

/// Composites `content` (the layer's own pixels over `big`, alpha already
/// masked, clipped layers applied) plus its effects into `backdrop`.
pub fn composite_with_effects(layer: &Layer, content: &Buffer, backdrop: &mut Buffer, maps: &FxMaps, layer_bounds: Rect, patterns: &[Pattern]) {
    let prepared = PreparedPatterns::new(patterns, PREPARED_PATTERN_BYTES);
    composite_with_effects_prepared(layer, content, backdrop, maps, layer_bounds, &prepared, None);
}

pub(crate) fn composite_with_effects_prepared(
    layer: &Layer,
    content: &Buffer,
    backdrop: &mut Buffer,
    maps: &FxMaps,
    layer_bounds: Rect,
    patterns: &PreparedPatterns<'_>,
    vstroke: Option<VectorStroke<'_>>,
) {
    let big = content.rect;
    let (w, h) = (big.width() as usize, big.height() as usize);
    // The effect shape: the content's alpha (with a split-off vector stroke, `content` is the
    // unmasked fill and the mask applies to fill ∪ stroke), joined with a filled shape's outline.
    let kmask = |i: usize| vstroke.as_ref().and_then(|v| v.mask).and_then(|m| m.get(i)).copied().unwrap_or(1.0);
    let union = |i: usize, a: f32| vstroke.as_ref().and_then(|v| v.stroke.px.get(i)).map_or(a, |s| kmask(i) * (a + s[3] * (1.0 - a)));
    // Transparency Shapes Layer off: the shape is the whole layer (its masks, as the maps were
    // built), and the content's own transparency acts like fill opacity within it.
    let shapeless = !layer.advanced.transparency_shapes;
    let shape = if shapeless {
        maps.crop(&maps.shape, big, 0.0)
    } else if maps.outline {
        let o = maps.crop(&maps.shape, big, 0.0);
        Map { w, h, v: o.v.iter().zip(&content.px).enumerate().map(|(i, (o, p))| o.max(union(i, p[3]))).collect() }
    } else {
        Map { w, h, v: content.px.iter().enumerate().map(|(i, p)| union(i, p[3])).collect() }
    };
    let relative = shapeless || maps.outline || vstroke.is_some();
    let fx = |i: usize, k: usize| maps.crop(&maps.per[i][k], big, 0.0);
    // Layer bounds (gradients aligned with the layer use the whole layer,
    // independent of the render rect).
    let sb = layer_bounds;
    // Linked patterns tile from the effects reference point (else the layer's top-left).
    let anchor = layer.effects.reference.unwrap_or((f64::from(sb.x0), f64::from(sb.y0)));
    let rect = backdrop.rect;
    let before = backdrop.clone();
    // Exterior effects are painted on a backdrop copy extended to `big`.
    let mut work = Buffer::transparent(big);
    for y in rect.y0.max(big.y0)..rect.y1.min(big.y1) {
        for x in rect.x0.max(big.x0)..rect.x1.min(big.x1) {
            work.px[((y - big.y0) as usize) * w + (x - big.x0) as usize] = before.get(x, y);
        }
    }
    let items: Vec<&Effect> = layer.effects.items.iter().filter(|e| e.enabled()).collect();
    let vector_shape = matches!(layer.content, photocraft_doc::LayerContent::Shape(_));
    // Multiple instances: the first listed is on top, so paint in reverse.
    let rev = || items.iter().copied().enumerate().rev();

    for (i, e) in rev() {
        if let Effect::DropShadow(s) = e {
            let mut m = fx(i, 0);
            if s.knocks_out {
                // The layer hides the shadow beneath it where its fill is see-through (at 100 %
                // fill in a normal mode it covers it anyway; at 0 % the shape shows the bare
                // backdrop), and wholly in a mode that blends with it ([`shadow_see_through`]).
                knock_out_exterior(&mut m, &shape, 1.0 - shadow_see_through(layer));
            }
            paint_color(&mut work, &m, rgb(&s.color), s.common.blend, s.common.opacity);
        }
    }
    for (i, e) in rev() {
        if let Effect::OuterGlow(g) = e {
            let mut m = fx(i, 0);
            // The glow is behind the shape, not painted across its interior when Fill is 0 %.
            // Preserve the existing anti-aliased edge when the fill is fully opaque.
            knock_out_exterior(&mut m, &shape, layer.fill_opacity);
            paint_glow(&mut work, &m, g, sb, anchor, big, patterns);
        }
    }

    // The layer: content at fill opacity, then interior effects. Interior effects are painted
    // relative to the layer's shape (alpha = coverage within the shape), then the shape's alpha
    // applies: a colour overlay at 100 % replaces the colour of a half-transparent edge pixel and
    // keeps its alpha, as in Photoshop.
    let fill = layer.fill_opacity;
    // Blend Interior Effects as Group: the interior effects (overlays, satin, inner glow) are
    // combined with the content first, and fill opacity applies to the combination.
    let interior_group = layer.advanced.blend_interior && fill < 1.0;
    let content_fill = if interior_group { 1.0 } else { fill };
    let inside = |a: f32| a > INSIDE_EPS;
    // Within an outline (or a split-off vector stroke) the content's own transparency (a fading
    // gradient fill) acts like fill opacity: the effects still cover the whole shape.
    let lay_alpha = |i: usize, p: &[f32; 4], a: f32| {
        if !inside(a) {
            0.0
        } else if relative {
            content_fill * (kmask(i) * p[3] / a).min(1.0)
        } else {
            content_fill
        }
    };
    let mut lay = Buffer { rect: big, px: content.px.iter().zip(&shape.v).enumerate().map(|(i, (p, a))| [p[0], p[1], p[2], lay_alpha(i, p, *a)]).collect() };
    let gamma = crate::text_gamma(layer);
    let premerge = premerges_layer(layer);
    if premerge {
        for (lp, wp) in lay.px.iter_mut().zip(&work.px) {
            *lp = psblend::composite_gamma(layer.blend, *wp, *lp, 1.0, gamma);
        }
    }
    let rel = |m: Map| -> Map {
        let v = m.v.iter().zip(&shape.v).map(|(m, a)| if inside(*a) { (m / a).min(1.0) } else { 0.0 }).collect();
        Map { w: m.w, h: m.h, v }
    };
    let full = Map { w, h, v: shape.v.iter().map(|a| if inside(*a) { 1.0 } else { 0.0 }).collect() };
    for (_, e) in rev() {
        if let Effect::PatternOverlay { common, name, id, scale, angle, link, phase } = e {
            paint_pattern(&mut lay, &full, patterns, name, id, Placement::anchored(anchor, *link, *phase, *scale, *angle), big, common.blend, common.opacity);
        }
    }
    for (_, e) in rev() {
        if let Effect::GradientOverlay { common, gradient, .. } = e {
            paint_fx(&mut lay, &full, &FxPaint::Gradient(gradient.clone()), sb, anchor, big, common.blend, common.opacity, patterns)
        }
    }
    for (_, e) in rev() {
        if let Effect::ColorOverlay { common, color } = e {
            paint_color(&mut lay, &full, rgb(color), common.blend, common.opacity);
        }
    }
    for (i, e) in rev() {
        if let Effect::Satin(s) = e {
            let m = rel(fx(i, 0));
            paint_color(&mut lay, &m, rgb(&s.color), s.common.blend, s.common.opacity);
        }
    }
    for (i, e) in rev() {
        if let Effect::InnerGlow(g) = e {
            paint_glow(&mut lay, &rel(fx(i, 0)), g, sb, anchor, big, patterns);
        }
    }
    if interior_group {
        for p in &mut lay.px {
            p[3] *= fill.max(0.0);
        }
    }
    for (i, e) in rev() {
        if let Effect::InnerShadow(s) = e {
            let m = rel(fx(i, 0));
            paint_color(&mut lay, &m, rgb(&s.color), s.common.blend, s.common.opacity);
        }
    }
    // A stroked shape's vector stroke: Photoshop draws it above the fill's clipped layers and
    // interior effects, below the stroke effect (psd-tools stroke-composite).
    if let Some(vs) = &vstroke {
        for (i, ((p, s), a)) in lay.px.iter_mut().zip(&vs.stroke.px).zip(&shape.v).enumerate() {
            if s[3] > 0.0 && inside(*a) {
                *p = psblend::composite(BlendMode::Normal, *p, [s[0], s[1], s[2], fill * (kmask(i) * s[3] / a).min(1.0)], 1.0);
            }
        }
    }
    // Strokes. Inside parts are painted over the layer (bottom instance first); outside parts
    // are slid beneath it, so the first listed (top) instance is processed first.
    let strokes: Vec<&photocraft_doc::StrokeFx> = items.iter().filter_map(|e| if let Effect::Stroke(s) = e { Some(s) } else { None }).collect();
    let ready = !strokes.is_empty();
    let din = if ready { maps.din.as_ref().map(|a| maps.crop_vec(a, big, 0.0)).unwrap_or_default() } else { Vec::new() };
    // A stroke's outer edge picks the chamfer or the exact field by its width
    // ([`stroke_outside_exact`]); both are cropped here, and only the ones a stroke needs exist.
    let crop_out = |v: &Option<Vec<f32>>| v.as_ref().map(|b| maps.crop_vec(b, big, FAR));
    let (dout, dout_exact, vdout, vdout_exact) =
        if ready { (crop_out(&maps.dout), crop_out(&maps.dout_exact), crop_out(&maps.vdout), crop_out(&maps.vdout_exact)) } else { (None, None, None, None) };
    let widths = stroke_widths;
    // A gradient stroke aligned with the layer spans the stroke's own extent.
    let frame = |st: &photocraft_doc::StrokeFx| maps.stroke_frame(st).unwrap_or(sb);
    for st in strokes.iter().rev().copied() {
        let (in_w, _) = widths(st);
        if in_w <= 0.0 {
            continue;
        }
        let mut m = Map::new(w, h, 0.0);
        for ((mv, a), dv) in m.v.iter_mut().zip(&shape.v).zip(&din) {
            *mv = if inside(*a) { (in_w + 0.5 - dv).clamp(0.0, 1.0) } else { 0.0 };
        }
        paint_fx(&mut lay, &m, &st.paint, frame(st), anchor, big, st.common.blend, st.common.opacity, patterns);
    }
    for (i, e) in rev() {
        if let Effect::BevelEmboss(b) = e
            && maps.bevel_paint[i] == BevelPaint::Inner
        {
            let (hi, sh) = (rel(fx(i, 0)), rel(fx(i, 1)));
            paint_color(&mut lay, &hi, rgb(&b.highlight_color), b.highlight.blend, b.highlight.opacity);
            paint_color(&mut lay, &sh, rgb(&b.shadow_color), b.shadow.blend, b.shadow.opacity);
        }
    }
    // The shape's own alpha.
    for (p, a) in lay.px.iter_mut().zip(&shape.v) {
        p[3] *= a.min(1.0);
    }
    // Shape layers without a filled outline stroke their vector outline, estimated from local
    // coverage; every other layer strokes its shape. The width then picks the field.
    let vector_outline = vector_shape && !maps.outline;
    // Outside parts lie beneath the layer and blend onto the backdrop with their own mode. A
    // higher stroke covers the ones below it (Photoshop: a Multiply stroke listed above a wider
    // Normal stroke multiplies the backdrop, not the lower stroke). Each stroke's share is its
    // coverage × opacity not yet taken by strokes above; for Normal strokes this equals stacking
    // them top over bottom.
    if strokes.iter().any(|st| widths(st).1 > 0.0) {
        let base = work.clone();
        let mut acc = vec![[0.0f32; 4]; w * h]; // premultiplied Σ share·blended
        let mut cover = vec![0.0f32; w * h];
        for st in strokes.iter().copied() {
            let (_, out_w) = widths(st);
            if out_w <= 0.0 {
                continue;
            }
            // Shape layers: Photoshop strokes the vector outline (estimated from local coverage
            // for unfilled shapes); the stroke never shows through the shape's pixels. Along a
            // filled shape's outline it covers the part of each edge pixel outside the path.
            let exact = stroke_outside_exact(out_w);
            let d: &[f32] = if vector_outline {
                (if exact { vdout_exact.as_deref() } else { vdout.as_deref() }).unwrap_or_default()
            } else if exact {
                dout_exact.as_deref().unwrap_or_default()
            } else {
                dout.as_deref().unwrap_or_default()
            };
            let mut share = vec![0.0f32; w * h];
            for (i, sh) in share.iter_mut().enumerate() {
                let band = (out_w + 0.5 - d[i]).clamp(0.0, 1.0);
                let k = if maps.outline {
                    band * outline_share(shape.v[i], lay.px[i][3])
                } else if inside(shape.v[i]) {
                    if vector_shape {
                        0.0
                    } else {
                        // The stroke lies outside the layer's pixels. Beneath the layer it may show
                        // only through the part of the pixel the shape doesn't cover (1 - a); the
                        // layer composited on top covers c = a × fill, so the share beneath is
                        // (1 - a) / (1 - c). At 100 % fill that is 1 (the layer hides the rest, as
                        // before); at 0 % fill the interior stays clear (Fill 0 % + Outside stroke).
                        let c = lay.px[i][3].clamp(0.0, 1.0);
                        if c >= 1.0 { 1.0 } else { ((1.0 - shape.v[i].clamp(0.0, 1.0)) / (1.0 - c)).clamp(0.0, 1.0) }
                    }
                } else {
                    band
                };
                *sh = k * st.common.opacity * (1.0 - cover[i]);
                cover[i] += *sh;
            }
            // The stroke blended over the backdrop at full coverage.
            let mut blended = base.clone();
            let ones = Map { w, h, v: share.iter().map(|&s| if s > 0.0 { 1.0 } else { 0.0 }).collect() };
            paint_fx(&mut blended, &ones, &st.paint, frame(st), anchor, big, st.common.blend, 1.0, patterns);
            for ((a, b), s) in acc.iter_mut().zip(&blended.px).zip(&share) {
                if *s > 0.0 {
                    for c in 0..3 {
                        a[c] += s * b[c] * b[3];
                    }
                    a[3] += s * b[3];
                }
            }
        }
        for ((p, a), c) in work.px.iter_mut().zip(&acc).zip(&cover) {
            if *c <= 0.0 {
                continue;
            }
            let k = 1.0 - c.min(1.0);
            let alpha = p[3] * k + a[3];
            if alpha > 0.0 {
                *p = [(p[0] * p[3] * k + a[0]) / alpha, (p[1] * p[3] * k + a[1]) / alpha, (p[2] * p[3] * k + a[2]) / alpha, alpha.min(1.0)];
            }
        }
    }
    for (i, e) in rev() {
        if let Effect::BevelEmboss(b) = e
            && maps.bevel_paint[i] == BevelPaint::Outer
        {
            let (hi, sh) = (fx(i, 0), fx(i, 1));
            paint_color(&mut work, &hi, rgb(&b.highlight_color), b.highlight.blend, b.highlight.opacity);
            paint_color(&mut work, &sh, rgb(&b.shadow_color), b.shadow.blend, b.shadow.opacity);
        }
    }

    // Layer (with interior effects) onto the exterior result, in the layer's mode.
    let mode = if premerge || layer.blend == BlendMode::PassThrough { BlendMode::Normal } else { layer.blend };
    for (wp, lp) in work.px.iter_mut().zip(&lay.px) {
        if lp[3] > 0.0 {
            *wp = psblend::composite_gamma(mode, *wp, *lp, 1.0, gamma);
        }
    }
    // Emboss styles shade the composited layer (their maps cover inside and outside). Painting
    // them into the layer before its (text-gamma) composite left a type layer's lit top edges up
    // to 6/255 dark (psd-tools layer_effects Emboss: 7.8 → 5.7/255).
    for (i, e) in rev() {
        if let Effect::BevelEmboss(b) = e
            && maps.bevel_paint[i] == BevelPaint::Both
        {
            paint_color(&mut work, &fx(i, 0), rgb(&b.highlight_color), b.highlight.blend, b.highlight.opacity);
            paint_color(&mut work, &fx(i, 1), rgb(&b.shadow_color), b.shadow.blend, b.shadow.opacity);
        }
    }
    // Layer opacity applies to the whole stack.
    let op = layer.opacity;
    for y in rect.y0..rect.y1 {
        for x in rect.x0..rect.x1 {
            let i = ((y - rect.y0) as usize) * rect.width() as usize + (x - rect.x0) as usize;
            let a = before.px[i];
            let b = if x >= big.x0 && x < big.x1 && y >= big.y0 && y < big.y1 { work.px[((y - big.y0) as usize) * w + (x - big.x0) as usize] } else { a };
            backdrop.px[i] = mix_premul(a, b, op);
        }
    }
}

/// Coverage of an outside stroke beneath a filled shape's edge pixel, for a pixel the outline
/// covers `cov` of and the layer (with its interior effects) ends at alpha `l`. Photoshop treats
/// the stroke and the shape as disjoint areas: the stroke fills the part outside the path and
/// the layer adds its own alpha (psd-tools stroke-effects: a 58 % covered edge pixel of an
/// opaque fill ends opaque, one of a faded fill at 42 % + its alpha). Composited beneath the
/// layer, `share + l (1 - share) = (1 - cov) + l`.
pub fn outline_share(cov: f32, l: f32) -> f32 {
    let outside = (1.0 - cov.clamp(0.0, 1.0)).max(0.0);
    let rest = 1.0 - l.clamp(0.0, 1.0);
    if rest <= 1e-6 { 1.0 } else { (outside / rest).min(1.0) }
}

/// (inside width, outside width) of a stroke.
pub fn stroke_widths(st: &photocraft_doc::StrokeFx) -> (f32, f32) {
    match st.position {
        StrokePosition::Outside => (0.0, st.size),
        StrokePosition::Inside => (st.size, 0.0),
        StrokePosition::Center => (st.size / 2.0, st.size / 2.0),
    }
}

/// The frame a gradient stroke aligned with the layer is laid out in: the pixel bounds of the
/// effect shape (over `rect`) grown by the stroke's outside width less one pixel. Photoshop
/// spans the gradient over the stroke's extent, not the layer's: a 4 px outside stroke around a
/// 22 px square runs its 90° gradient over 28 px (psd-tools stroke-effects), a 3 px one around
/// 36 px of type over 40 px (psd-tools effect-stroke-gradient).
fn stroke_frame(shape: &Map, out_w: f32, rect: Rect) -> Rect {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for y in 0..shape.h {
        for x in 0..shape.w {
            if shape.v[y * shape.w + x] > INSIDE_EPS {
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
            }
        }
    }
    if x0 == usize::MAX {
        return rect;
    }
    let grow = if out_w.is_finite() { (out_w.min(MAX_REACH).ceil() as i32 - 1).max(0) } else { 0 };
    Rect::new(rect.x0 + x0 as i32, rect.y0 + y0 as i32, rect.x0 + x1 as i32, rect.y0 + y1 as i32).inflate(grow)
}

/// A stroked shape's vector stroke, composited apart from its fill (which the compositor then
/// passes unmasked): the stroke's pixels over the render rect and the layer's mask values.
pub(crate) struct VectorStroke<'a> {
    pub stroke: &'a Buffer,
    pub mask: Option<&'a [f32]>,
}

/// Premultiplied linear interpolation between two straight-alpha pixels.
fn mix_premul(a: [f32; 4], b: [f32; 4], k: f32) -> [f32; 4] {
    let alpha = a[3] + (b[3] - a[3]) * k;
    if alpha <= 0.0 {
        return [0.0; 4];
    }
    let mut out = [0.0; 4];
    for c in 0..3 {
        out[c] = (a[c] * a[3] + (b[c] * b[3] - a[c] * a[3]) * k) / alpha;
    }
    out[3] = alpha;
    out
}

/// A distance field of an effect shape, as the map builders above compute it (for the GPU
/// compositor, which builds the rest of the maps itself; keep in sync with `shadow_map`,
/// `glow_map`, `bevel_maps` and `build_maps`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FieldKind {
    /// `dist_outside` of the shape: precise outer glows.
    Outside,
    /// `dist_inside` of the shape: precise inner glows.
    Inside,
    /// `bevel_dist_outside` of the shape: the outside distance of chiselled bevels.
    BevelOutside,
    /// `bevel_dist_inside` of the shape: the inside distance of chiselled bevels.
    BevelInside,
    /// Chamfer distance outside the shape: a narrow raster stroke's outer edge (see
    /// [`stroke_outside_exact`]), and the spread of drop shadows and softer outer glows
    /// ([`dilate`]); their metric is fitted to Photoshop (#224) and keeps the sub-pixel seeding.
    StrokeOutside,
    /// Chamfer distance outside `1 - alpha`: the choke of inner shadows and softer inner glows.
    ChokeInside,
    /// Stroke distance inside the shape.
    StrokeInside,
    /// Exact Euclidean distance outside the shape: a wide raster stroke's outer edge (see
    /// [`stroke_outside_exact`]). The chamfer's level sets are polygons (an ~8-fold octagon), so a
    /// wide stroke's outer boundary facets; the shadow / glow spread above keeps the chamfer (#3646).
    StrokeOutsideExact,
    /// Stroke distance outside a shape layer's outline (from its local coverage), exact Euclidean.
    StrokeOutsideVector,
    /// Chamfer distance outside a shape layer's outline (from its local coverage): a narrow vector
    /// stroke's outer edge, where the chamfer matches Photoshop (see [`STROKE_CHAMFER_MAX`]).
    StrokeOutsideVectorChamfer,
}

/// The `kind` distance field of the shape `alpha` (row-major `w × h`). Values past `reach` pixels
/// from the shape's edge are only meant to be large (the bevel fields skip refining them).
pub fn distance_field(kind: FieldKind, alpha: Vec<f32>, w: usize, h: usize, reach: f32) -> Vec<f32> {
    let s = Map { w, h, v: alpha };
    match kind {
        FieldKind::Outside => dist_outside(&s),
        FieldKind::Inside => dist_inside(&s),
        FieldKind::BevelOutside => bevel_dist_outside(&s, &sharp_edges(&s), reach),
        FieldKind::BevelInside => bevel_dist_inside(&s, &sharp_edges(&s), reach),
        FieldKind::StrokeOutside => dist_outside_by(&s, Metric::Chamfer),
        FieldKind::ChokeInside => dist_outside_by(&s.map(|a| 1.0 - a), Metric::Chamfer),
        FieldKind::StrokeInside => dist_inside_by(&s, Metric::Chamfer),
        FieldKind::StrokeOutsideExact => dist_outside_by(&s, Metric::Euclidean),
        FieldKind::StrokeOutsideVector => dist_outside_by(&local_coverage(&s), Metric::Euclidean),
        FieldKind::StrokeOutsideVectorChamfer => dist_outside_by(&local_coverage(&s), Metric::Chamfer),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs `f` with every parallel pass on this thread using `n` threads (1 = in order), whatever its size.
    fn with_threads<R>(n: usize, f: impl FnOnce() -> R) -> R {
        struct Restore(Option<usize>);
        impl Drop for Restore {
            fn drop(&mut self) {
                THREADS.with(|t| t.set(self.0));
            }
        }
        let _restore = Restore(THREADS.with(|t| t.replace(Some(n))));
        f()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn edt_is_exact() {
        let (w, h) = (7, 5);
        let mut inside = vec![false; w * h];
        inside[2 * w + 3] = true;
        let d = edt(&inside, w, h);
        assert_eq!(d[2 * w + 3], 0.0);
        assert!((d[2 * w + 6] - 3.0).abs() < 1e-6);
        assert!((d[0] - (9.0f32 + 4.0).sqrt()).abs() < 1e-5);
        assert!((d[4 * w + 5] - (4.0f32 + 4.0).sqrt()).abs() < 1e-5);
    }

    #[test]
    fn edt_two_seeds() {
        let (w, h) = (10, 1);
        let mut inside = vec![false; w];
        inside[0] = true;
        inside[9] = true;
        let d = edt(&inside, w, h);
        assert_eq!(d, vec![0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn blur_preserves_mass() {
        let mut m = Map::new(21, 21, 0.0);
        m.v[10 * 21 + 10] = 1.0;
        blur(&mut m, 6.0);
        let sum: f32 = m.v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "{sum}");
        assert!(m.v[10 * 21 + 10] < 1.0);
    }

    #[test]
    fn gradient_geometry() {
        let b = Rect::new(0, 0, 100, 100);
        let t = |s, x, y| gradient_t(s, 0.0, 1.0, false, (0.0, 0.0), b, x, y);
        assert!(t(GradientStyle::Linear, 0.0, 50.0) < 0.01);
        // Linear / reflected sample pixel corners: the centre is reached half a pixel later.
        assert!((t(GradientStyle::Linear, 50.5, 50.0) - 0.5).abs() < 1e-6);
        assert!(t(GradientStyle::Linear, 100.0, 50.0) > 0.99);
        assert!(t(GradientStyle::Reflected, 50.5, 50.0) < 1e-6);
        assert!((t(GradientStyle::Reflected, 0.0, 50.0) - 1.0).abs() < 1e-6);
        assert!(t(GradientStyle::Radial, 50.0, 50.0) < 1e-6);
        assert!((t(GradientStyle::Radial, 50.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((t(GradientStyle::Diamond, 75.0, 75.0) - 1.0).abs() < 1e-6);
        let ang = t(GradientStyle::Angle, 50.0, 0.0);
        assert!((ang - 0.75).abs() < 1e-3, "{ang}");
        let rev = gradient_t(GradientStyle::Linear, 0.0, 1.0, true, (0.0, 0.0), b, 0.0, 50.0);
        assert!(rev > 0.99);
    }

    #[test]
    fn effect_gradients_snap_end_points_and_sample_corners() {
        // 87° over 600 × 60 at (99, 422): end points (397, 482) and (400, 422), so the overlay runs
        // along (3, -60) from its midpoint (398.5, 452), sampled at pixel corners.
        let frame = Rect::new(99, 422, 699, 482);
        let (a, sc, o) = crate::fill_layout::gradient_layout(GradientStyle::Linear, 87.0, 1.0, (0.0, 0.0), frame);
        assert!((a - (60f32).atan2(3.0).to_degrees()).abs() < 1e-3, "{a}");
        let t = |x: f32, y: f32| gradient_t(GradientStyle::Linear, a, sc, false, o, frame, x, y);
        assert!((t(399.0, 452.5) - 0.5).abs() < 1e-4, "{}", t(399.0, 452.5));
        assert!(t(400.5, 422.5) > 0.99 && t(397.5, 482.5) < 0.01);
    }

    #[test]
    fn chamfer_uses_euclidean_step_lengths() {
        let (w, h) = (9, 9);
        let mut inside = vec![false; w * h];
        inside[0] = true;
        let (d, near) = chamfer_nearest(&inside, w, h);
        let at = |x: usize, y: usize| d[y * w + x];
        assert!((at(3, 0) - 3.0).abs() < 1e-5);
        assert!((at(2, 2) - 2.0 * std::f32::consts::SQRT_2).abs() < 1e-5);
        assert!((at(1, 2) - 5f32.sqrt()).abs() < 1e-5);
        assert!((at(1, 3) - (5f32.sqrt() + 1.0)).abs() < 1e-5, "knight + straight, not √10");
        assert!((at(2, 3) - (5f32.sqrt() + std::f32::consts::SQRT_2)).abs() < 1e-5);
        assert!(near.iter().all(|&n| n == 0));
    }

    #[test]
    fn spread_dilates_with_the_chamfer_metric() {
        // One opaque pixel grown by 3 px: at offset (1, 3) the chamfer distance is √5 + 1 = 3.236
        // (Euclidean 3.162), so the anti-aliased rim keeps 3 + 0.5 − (3.236 − 0.5) = 0.764.
        let mut m = Map::new(9, 9, 0.0);
        m.v[4 * 9 + 4] = 1.0;
        let d = dilate(&m, 3.0);
        assert!((d.v[7 * 9 + 5] - 0.764).abs() < 1e-3, "{}", d.v[7 * 9 + 5]);
        // (2, 3): √5 + √2 = 3.650 (Euclidean 3.606) leaves 0.350; axis steps are exact.
        assert!((d.v[7 * 9 + 6] - (4.0 - 5f32.sqrt() - std::f32::consts::SQRT_2)).abs() < 1e-4, "{}", d.v[7 * 9 + 6]);
        assert_eq!(d.v[4 * 9 + 7], 1.0);
    }

    #[test]
    fn running_sum_tent_matches_the_kernel() {
        let (w, h) = (37usize, 23usize);
        let src: Vec<f32> = (0..w * h).map(|i| ((i * 7919) % 101) as f32 / 100.0).collect();
        for width in [1.0f32, 2.0, 3.5, 5.0, 8.25, 41.0] {
            let mut m = Map { w, h, v: src.clone() };
            tent(&mut m, width);
            let mut want = Map { w, h, v: src.clone() };
            tent_direct(&mut want, width);
            for (a, b) in m.v.iter().zip(&want.v) {
                assert!((a - b).abs() < 1e-5, "width {width}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn tent_kernel_is_two_boxes() {
        let (r, k) = tent_kernel(5.0);
        assert_eq!(r, 4);
        let sum: f32 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!((k[4] - 5.0 / 25.0).abs() < 1e-6 && (k[0] - 1.0 / 25.0).abs() < 1e-6);
        // Fractional widths taper the end taps; width 1 is the identity.
        let (r, k) = tent_kernel(4.0);
        assert_eq!(r, 4);
        assert!(k[0] > 0.0 && k[0] < 1.0 / 25.0);
        assert_eq!(tent_kernel(1.0), (0, vec![1.0]));
        // #1543: a huge, infinite or NaN width is capped, not turned into a kernel of 2^62 taps.
        for w in [1e30, f32::INFINITY, f32::MAX] {
            assert_eq!(tent_kernel(w), tent_kernel(MAX_REACH), "{w}");
        }
        assert_eq!(tent_kernel(f32::NAN), (0, vec![1.0]));
    }

    fn no_tex() -> TextureCtx<'static> {
        TextureCtx { rect: Rect::new(0, 0, 1, 1), patterns: &[], anchor: (0.0, 0.0) }
    }

    fn square(n: usize, x0: usize, x1: usize) -> Map {
        let mut m = Map::new(n, n, 0.0);
        for y in x0..x1 {
            for x in x0..x1 {
                m.v[y * n + x] = 1.0;
            }
        }
        m
    }

    fn bevel_of(style: BevelStyle, technique: BevelTechnique) -> Bevel {
        Bevel {
            enabled: true,
            style,
            technique,
            depth: 1.0,
            up: true,
            size: 5.0,
            soften: 0.0,
            angle: 90.0,
            altitude: 30.0,
            use_global_light: false,
            gloss_contour: Contour::Linear,
            highlight: photocraft_doc::FxCommon::new(BlendMode::Screen, 0.75),
            highlight_color: photocraft_color::Color::WHITE,
            shadow: photocraft_doc::FxCommon::new(BlendMode::Multiply, 0.75),
            shadow_color: photocraft_color::Color::BLACK,
            contour: None,
            texture: None,
        }
    }

    #[test]
    fn smooth_inner_bevel_lights_the_top_edge_inside_only() {
        let shape = square(40, 10, 30);
        let (maps, paint) = bevel_maps(
            &shape,
            &bevel_of(BevelStyle::InnerBevel, BevelTechnique::Smooth),
            &GlobalLight::default(),
            &no_tex(),
            &PreparedPatterns::new(&[], PREPARED_PATTERN_BYTES),
        );
        assert_eq!(paint, BevelPaint::Inner);
        let at = |m: &Map, x: usize, y: usize| m.v[y * 40 + x];
        // Light from the top: highlight along the top edge, shadow along the bottom, flat middle.
        assert!(at(&maps[0], 20, 10) > 0.5 && at(&maps[1], 20, 10) == 0.0);
        assert!(at(&maps[1], 20, 29) > 0.5);
        assert!(at(&maps[0], 20, 20) < 1e-3 && at(&maps[1], 20, 20) < 1e-3);
        // Nothing outside the shape; the ramp reaches about `size` inside.
        assert_eq!(at(&maps[0], 20, 8), 0.0);
        assert!(at(&maps[0], 20, 15) < 0.05);
    }

    /// Photoshop 27.11, measured across a straight edge (size 10, depth 100 %, light from the top
    /// at 30°): a chisel facet has the slope `depth × size / (2 (size + 1))`, flat right up to the
    /// edge pixel for Inner Hard; Inner Soft's edge pixel is half as steep again; Outer Hard
    /// still shades, partly, the pixel where its ramp reaches zero `size + 1½` px out.
    #[test]
    fn chisel_facets_match_photoshop() {
        let amount = |g: f32| {
            let (se, ce) = 30f32.to_radians().sin_cos();
            ((g * ce + se) / (1.0 + g * g).sqrt() - se) / (1.0 - se)
        };
        let n = 80;
        let shape = square(n, 20, 60);
        let maps = |style, technique| {
            let mut b = bevel_of(style, technique);
            b.size = 10.0;
            bevel_maps(&shape, &b, &GlobalLight::default(), &no_tex(), &PreparedPatterns::new(&[], PREPARED_PATTERN_BYTES)).0
        };
        let at = |m: &Map, y: usize| m.v[y * n + 40];
        let facet = amount(10.0 / 22.0);
        let hard = maps(BevelStyle::InnerBevel, BevelTechnique::ChiselHard);
        for y in 20..30 {
            assert!((at(&hard[0], y) - facet).abs() < 0.01, "inner hard facet at row {y}: {} vs {facet}", at(&hard[0], y));
        }
        assert!(at(&hard[0], 30) > 0.05 && at(&hard[0], 30) < facet, "one partial pixel at the end");
        let soft = maps(BevelStyle::InnerBevel, BevelTechnique::ChiselSoft);
        assert!((at(&soft[0], 20) - amount(1.5 * 10.0 / 22.0)).abs() < 0.01, "steeper edge pixel: {}", at(&soft[0], 20));
        assert!((at(&soft[0], 24) - facet).abs() < 0.01, "then Chisel Hard's facet");
        let outer = maps(BevelStyle::OuterBevel, BevelTechnique::ChiselHard);
        // Below the shape (rows 60..): shadow facet through 10½ px, partial at 11½.
        assert!((at(&outer[1], 70) - at(&outer[1], 61)).abs() < 1e-4);
        assert!(at(&outer[1], 71) > 0.05 && at(&outer[1], 71) < at(&outer[1], 70));
        assert_eq!(at(&outer[1], 72), 0.0);
    }

    #[test]
    fn emboss_paints_both_sides_and_pillow_flips_the_outside() {
        let shape = square(40, 10, 30);
        let l = GlobalLight::default();
        let (e, paint) =
            bevel_maps(&shape, &bevel_of(BevelStyle::Emboss, BevelTechnique::Smooth), &l, &no_tex(), &PreparedPatterns::new(&[], PREPARED_PATTERN_BYTES));
        assert_eq!((paint, e.len()), (BevelPaint::Both, 2));
        let at = |m: &Map, x: usize, y: usize| m.v[y * 40 + x];
        // Emboss: the slope across the top edge faces the light on both sides.
        assert!(at(&e[0], 20, 10) > 0.3 && at(&e[0], 20, 9) > 0.3);
        let (p, _) =
            bevel_maps(&shape, &bevel_of(BevelStyle::PillowEmboss, BevelTechnique::Smooth), &l, &no_tex(), &PreparedPatterns::new(&[], PREPARED_PATTERN_BYTES));
        assert!(at(&p[0], 20, 10) > 0.3 && at(&p[1], 20, 9) > 0.3, "pillow: outside top edge in shadow");
    }

    #[test]
    fn emboss_width_rounds_up_and_shades_past_the_blur() {
        let mut b = bevel_of(BevelStyle::Emboss, BevelTechnique::Smooth);
        b.size = 41.0;
        let g = bevel_geom(&b);
        assert_eq!((g.width, g.depth), (21.0, 21.0));
        b.size = 5.0;
        assert_eq!(bevel_geom(&b).width, 3.0);
        // The first row past the height blur's reach still slopes (its upper neighbour is raised),
        // so it shades; the row after it does not.
        let shape = square(60, 20, 40);
        b.size = 9.0;
        b.angle = 90.0;
        let (m, _) = bevel_maps(&shape, &b, &GlobalLight::default(), &no_tex(), &PreparedPatterns::new(&[], PREPARED_PATTERN_BYTES));
        let reach = tent_kernel(bevel_geom(&b).width).0 as usize;
        let row = 39 + reach + 1; // last shape row + reach + 1: height 0, neighbour above raised
        assert!(m[1].v[row * 60 + 30] > 0.0, "{}", m[1].v[row * 60 + 30]);
        assert_eq!(m[1].v[(row + 1) * 60 + 30], 0.0);
    }

    /// A `len` x `wid` bar through the middle of a `w` x `h` map at `deg` degrees, anti-aliased by
    /// 16 x 16 supersampling (the coverage a scanline rasteriser gives).
    fn aa_bar(w: usize, h: usize, deg: f32, len: f32, wid: f32) -> Map {
        let (sn, cs) = deg.to_radians().sin_cos();
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        let mut m = Map::new(w, h, 0.0);
        const N: usize = 16;
        for y in 0..h {
            for x in 0..w {
                let mut hit = 0;
                for sy in 0..N {
                    for sx in 0..N {
                        let (px, py) = (x as f32 + (sx as f32 + 0.5) / N as f32 - cx, y as f32 + (sy as f32 + 0.5) / N as f32 - cy);
                        let (u, v) = (px * cs + py * sn, -px * sn + py * cs);
                        hit += usize::from(u.abs() <= len / 2.0 && v.abs() <= wid / 2.0);
                    }
                }
                m.v[y * w + x] = hit as f32 / (N * N) as f32;
            }
        }
        m
    }

    /// Ripple and bias of a distance field on an `aa_bar` (`inside` or outside it): the worst
    /// standard deviation of the error among pixels at the same true distance from a long edge
    /// (0.5 px bins, 0.6..8 px, away from the ends), and the worst mean error of a bin. A constant
    /// offset is no ripple: only a field that varies along the edge shades in stripes.
    fn bar_ripple(f: &[f32], w: usize, h: usize, deg: f32, wid: f32, inside: bool) -> (f32, f32) {
        let (sn, cs) = deg.to_radians().sin_cos();
        let mut bins: Vec<Vec<f32>> = vec![Vec::new(); 16];
        for y in 0..h {
            for x in 0..w {
                let (px, py) = (x as f32 + 0.5 - w as f32 / 2.0, y as f32 + 0.5 - h as f32 / 2.0);
                let (u, v) = (px * cs + py * sn, -px * sn + py * cs);
                let truth = if inside { wid / 2.0 - v.abs() } else { v.abs() - wid / 2.0 };
                if u.abs() < 40.0 && (0.6..8.0).contains(&truth) {
                    bins[(truth * 2.0) as usize].push(f[y * w + x] - truth);
                }
            }
        }
        let (mut ripple, mut bias) = (0.0f32, 0.0f32);
        for b in bins.iter().filter(|b| b.len() > 8) {
            let mean = b.iter().sum::<f32>() / b.len() as f32;
            let sd = (b.iter().map(|e| (e - mean) * (e - mean)).sum::<f32>() / b.len() as f32).sqrt();
            ripple = ripple.max(sd);
            bias = bias.max(mean.abs());
        }
        (ripple, bias)
    }

    /// The worst standard deviation of the chisel shading (highlight minus shadow) along lines
    /// parallel to an `aa_bar` 2..6 px in from its edges, the way Issue #1926 measured stripes.
    fn bar_shade_ripple(deg: f32, light: f32) -> f32 {
        let (w, h) = (220usize, 220usize);
        let bar = aa_bar(w, h, deg, 150.0, 16.0);
        let mut b = bevel_of(BevelStyle::InnerBevel, BevelTechnique::ChiselHard);
        (b.size, b.depth, b.angle, b.altitude) = (8.0, 2.2, light, 32.0);
        let (m, _) = bevel_maps(&bar, &b, &GlobalLight::default(), &no_tex(), &PreparedPatterns::new(&[], PREPARED_PATTERN_BYTES));
        let (sn, cs) = deg.to_radians().sin_cos();
        let mut worst = 0.0f32;
        for o in [-6.0f32, -5.0, -4.0, -3.0, -2.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
            let line: Vec<f32> = (-60..=60)
                .map(|k| {
                    let t = k as f32 * 0.5;
                    let (x, y) = (w as f32 / 2.0 + t * cs - o * sn, h as f32 / 2.0 + t * sn + o * cs);
                    let i = y.floor() as usize * w + x.floor() as usize;
                    m[0].v[i] - m[1].v[i]
                })
                .collect();
            let mean = line.iter().sum::<f32>() / line.len() as f32;
            worst = worst.max((line.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / line.len() as f32).sqrt());
        }
        worst
    }

    /// Coverage of a unit pixel by the half plane `n . p + s >= 0` (`p` from the pixel's centre),
    /// by 64 x 64 supersampling.
    fn half_plane_coverage(n: [f32; 2], s: f32) -> f32 {
        const N: usize = 64;
        let mut hit = 0;
        for y in 0..N {
            for x in 0..N {
                let (px, py) = ((x as f32 + 0.5) / N as f32 - 0.5, (y as f32 + 0.5) / N as f32 - 0.5);
                hit += usize::from(n[0] * px + n[1] * py + s >= 0.0);
            }
        }
        hit as f32 / (N * N) as f32
    }

    #[test]
    fn edge_offset_places_a_straight_edge_by_its_coverage() {
        for step in 0..=12 {
            let (sn, cs) = (step as f32 * 7.5).to_radians().sin_cos();
            for k in -12..=12 {
                let s = k as f32 * 0.05;
                let a = half_plane_coverage([cs, sn], s);
                // Where the edge only just touches the pixel the coverage hardly moves with it.
                if !(0.03..=0.97).contains(&a) {
                    continue;
                }
                let got = edge_offset([cs, sn], a);
                assert!((got - s).abs() < 0.03, "{}deg s {s}: coverage {a} gave {got}", step as f32 * 7.5);
            }
        }
        // Axis-aligned edges keep `alpha - 1/2`; a pixel no coverage says anything about is 0.
        assert!((edge_offset([1.0, 0.0], 0.25) + 0.25).abs() < 1e-6);
        assert!((edge_offset([0.0, 1.0], 0.9) - 0.4).abs() < 1e-6);
        assert_eq!(edge_offset([0.6, 0.8], f32::NAN), 0.0);
    }

    #[test]
    fn bevel_distances_equal_the_plain_ones_without_anti_aliasing() {
        // Hard edges (no partly covered pixel) and soft ramps (no sharp one) keep the plain fields.
        let (w, h) = (96usize, 80usize);
        let mut maps = Vec::new();
        for kind in 0..4 {
            let mut m = Map::new(w, h, 0.0);
            let mut seed = 12345u32;
            for y in 0..h {
                for x in 0..w {
                    let (dx, dy) = (x as f32 - 48.0, y as f32 - 40.0);
                    m.v[y * w + x] = match kind {
                        0 => f32::from((50.0..90.0).contains(&(x as f32 * 0.6 + y as f32))),
                        1 => f32::from(dx * dx + dy * dy < 30.0 * 30.0),
                        2 => {
                            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                            f32::from(seed >> 28 < 5 && x > 10 && x < 80)
                        }
                        _ => ((20.0 - (dx * dx + dy * dy).sqrt()) / 8.0).clamp(0.0, 1.0),
                    };
                }
            }
            maps.push(m);
        }
        for (i, m) in maps.iter().enumerate() {
            let sharp = sharp_edges(m);
            assert!(sharp.is_empty(), "map {i} has no anti-aliased edge");
            assert_eq!(bevel_dist_inside(m, &sharp, 1e9), dist_inside(m), "inside, map {i}");
            assert_eq!(bevel_dist_outside(m, &sharp, 1e9), dist_outside(m), "outside, map {i}");
        }
    }

    #[test]
    fn chisel_distance_has_no_ripple_along_oblique_edges() {
        // The plain distance to a pixel's centre ripples with the staircase of an oblique edge (the
        // stripes of Issue #1926); at 0 and 45 degrees the staircase is regular and it does not.
        let (w, h) = (160, 160);
        for deg in [0.0f32, 30.0, 45.0, 60.0, 75.0, 120.0, 150.0] {
            let bar = aa_bar(w, h, deg, 120.0, 40.0);
            let sharp = sharp_edges(&bar);
            let inside = bar_ripple(&bevel_dist_inside(&bar, &sharp, 40.0), w, h, deg, 40.0, true);
            let outside = bar_ripple(&bevel_dist_outside(&bar, &sharp, 40.0), w, h, deg, 40.0, false);
            for (what, (ripple, bias)) in [("inside", inside), ("outside", outside)] {
                assert!(ripple < 0.008 && bias < 0.01, "{deg} deg {what}: ripple {ripple}, bias {bias}");
            }
        }
        let plain = aa_bar(w, h, 60.0, 120.0, 40.0);
        let (ripple, _) = bar_ripple(&dist_inside(&plain), w, h, 60.0, 40.0, true);
        assert!(ripple > 0.1, "the plain field still ripples ({ripple}), so the test above means something");
    }

    #[test]
    fn edge_normals_follow_straight_edges_to_a_fraction_of_a_degree() {
        // The 3x3 gradient is 1-3 degrees off and the error is systematic; the line through the edge
        // points of the pixels around one is not.
        let (w, h, wid) = (160usize, 160usize, 40.0f32);
        for deg in [30.0f32, 60.0, 75.0, 120.0, 150.0] {
            let sharp = sharp_edges(&aa_bar(w, h, deg, 120.0, wid));
            let (sn, cs) = deg.to_radians().sin_cos();
            let (mut worst, mut seen) = (0.0f32, 0);
            for (i, e) in sharp.iter() {
                let (px, py) = ((i % w) as f32 + 0.5 - w as f32 / 2.0, (i / w) as f32 + 0.5 - h as f32 / 2.0);
                let (u, v) = (px * cs + py * sn, -px * sn + py * cs);
                if u.abs() > 40.0 || (wid / 2.0 - v.abs()).abs() > 1.5 {
                    continue;
                }
                let into = if v > 0.0 { [sn, -cs] } else { [-sn, cs] };
                worst = worst.max((e.normal[0] * into[0] + e.normal[1] * into[1]).clamp(-1.0, 1.0).acos().to_degrees());
                seen += 1;
            }
            assert!(seen > 100 && worst < 0.6, "{deg} deg: {seen} edge pixels, normal off by up to {worst} deg");
        }
    }

    #[test]
    fn chisel_distance_is_close_on_round_shapes() {
        // Circles (a hard stretch of edge at each pole, where the edge lies on a pixel boundary) and
        // rounded rectangles on whole pixels (straight hard edges joined to anti-aliased arcs): the
        // distance to the sharp pixels alone is up to 0.3 px too long there, so the plain distance
        // caps it. Errors are against the exact distance to the shape, 0.6..8 px from its edge.
        let worst = |f: &[f32], w: usize, h: usize, sdf: &dyn Fn(f32, f32) -> f32, inside: bool| {
            let mut mx = 0.0f32;
            for y in 0..h {
                for x in 0..w {
                    let t = sdf(x as f32 + 0.5, y as f32 + 0.5);
                    let truth = if inside { -t } else { t };
                    if (0.6..8.0).contains(&truth) {
                        mx = mx.max((f[y * w + x] - truth).abs());
                    }
                }
            }
            mx
        };
        for r in [20.0f32, 40.0, 70.0] {
            let (w, h) = (200usize, 200usize);
            let m = aa_disc(w, h, r);
            let sharp = sharp_edges(&m);
            let sdf = |x: f32, y: f32| (x - 100.0).hypot(y - 100.0) - r;
            let (i, o) = (worst(&bevel_dist_inside(&m, &sharp, 12.0), w, h, &sdf, true), worst(&bevel_dist_outside(&m, &sharp, 12.0), w, h, &sdf, false));
            assert!(i < 0.05 && o < 0.07, "disc {r}: inside off by {i}, outside by {o}");
        }
        for rad in [6.0f32, 12.0, 24.0] {
            let (w, h) = (200usize, 160usize);
            let m = aa_round_rect(w, h, 20.0, 20.0, 180.0, 140.0, rad);
            let sharp = sharp_edges(&m);
            let sdf = |x: f32, y: f32| {
                let (qx, qy) = ((x - 100.0).abs() - (80.0 - rad), (y - 80.0).abs() - (60.0 - rad));
                qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0) - rad
            };
            let (i, o) = (worst(&bevel_dist_inside(&m, &sharp, 12.0), w, h, &sdf, true), worst(&bevel_dist_outside(&m, &sharp, 12.0), w, h, &sdf, false));
            assert!(i < 0.08 && o < 0.2, "rounded rect, radius {rad}: inside off by {i}, outside by {o}");
        }
    }

    #[test]
    fn chisel_facets_are_flat_along_oblique_edges() {
        // Issue #1926: the facets of a chiselled bar at 30, 60, 120, 150 degrees were striped.
        for deg in [0.0f32, 30.0, 45.0, 60.0, 75.0, 120.0, 150.0] {
            for light in [-90.0f32, 0.0, 45.0, 120.0] {
                let ripple = bar_shade_ripple(deg, light);
                assert!(ripple < 0.03, "{deg} deg bar, light {light}: shading ripple {ripple}");
            }
        }
    }

    #[test]
    fn bevel_distances_survive_degenerate_maps() {
        let cases: [(usize, usize, f32); 7] = [(0, 0, 0.0), (1, 1, 0.5), (1, 1, 1.0), (2, 2, 0.5), (3, 1, 0.3), (5, 5, f32::NAN), (4, 4, 1.0)];
        for (w, h, a) in cases {
            let m = Map::new(w, h, a);
            let sharp = sharp_edges(&m);
            for reach in [0.0, 8.0, f32::NAN, f32::INFINITY, -3.0] {
                assert_eq!(bevel_dist_inside(&m, &sharp, reach).len(), w * h);
                assert_eq!(bevel_dist_outside(&m, &sharp, reach).len(), w * h);
            }
        }
        // A single partly covered pixel in an empty map, and an edge on the map's border.
        let mut m = Map::new(5, 5, 0.0);
        m.v[12] = 0.4;
        let _ = (bevel_dist_inside(&m, &sharp_edges(&m), 4.0), bevel_dist_outside(&m, &sharp_edges(&m), 4.0));
        let mut m = Map::new(8, 8, 0.0);
        for y in 0..8 {
            m.v[y * 8 + 7] = 1.0;
            m.v[y * 8 + 6] = 0.5;
        }
        let _ = (bevel_dist_inside(&m, &sharp_edges(&m), 4.0), bevel_dist_outside(&m, &sharp_edges(&m), 4.0));
    }

    #[test]
    fn par_units_visits_every_unit_once_with_its_own_slices() {
        // Lengths that leave a short last unit, no second slice, units of different sizes, and nothing at all.
        let cases: [(usize, usize, usize, usize); 8] =
            [(0, 3, 0, 0), (1, 1, 1, 1), (10, 3, 0, 0), (64, 8, 64, 8), (65, 8, 13, 1), (200, 1, 200, 1), (7, 7, 14, 2), (9, 4, 0, 5)];
        for (la, ua, lb, ub) in cases {
            for threads in [1, 2, 3, 8, 64] {
                let (mut a, mut b) = (vec![0u32; la], vec![0u32; lb]);
                let calls = std::sync::atomic::AtomicUsize::new(0);
                with_threads(threads, || {
                    par_units(
                        la,
                        &mut a,
                        ua,
                        &mut b,
                        ub,
                        || (),
                        |_, u, sa, sb| {
                            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            assert!(sa.len() <= ua && sb.len() <= ub);
                            for (k, v) in sa.iter_mut().enumerate() {
                                *v = (u * ua + k) as u32 + 1;
                            }
                            for (k, v) in sb.iter_mut().enumerate() {
                                *v = (u * ub + k) as u32 + 1;
                            }
                        },
                    );
                });
                assert_eq!(calls.into_inner(), la.div_ceil(ua), "{la} {ua} {lb} {ub} on {threads}");
                assert!(a.iter().enumerate().all(|(i, &v)| v == i as u32 + 1), "a: {la} {ua} {lb} {ub} on {threads}");
                // `b` is written for the units that exist in `a` and no further.
                let written = (la.div_ceil(ua) * ub).min(lb);
                assert!(b.iter().enumerate().all(|(i, &v)| v == if i < written { i as u32 + 1 } else { 0 }), "b: {la} {ua} {lb} {ub} on {threads}");
            }
        }
        // A unit size of 0 does nothing.
        let mut a = vec![1u8; 4];
        par_units(4, &mut a, 0, &mut [] as &mut [()], 0, || (), |_, _, _, _| panic!("no units"));
    }

    #[test]
    fn nearest_distances_do_not_depend_on_the_threads() {
        // Sizes that do not divide into the thread groups, lines of 1 pixel, and a noisy map where
        // many pixels are equally far from two seeds (the nearest one must resolve the same way).
        for (w, h) in [(37usize, 23usize), (1, 40), (40, 1), (2, 2), (101, 3), (64, 64)] {
            let mut seed = 7u32;
            let inside: Vec<bool> = (0..w * h)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 24).is_multiple_of(11)
                })
                .collect();
            let (d1, n1) = with_threads(1, || edt_nearest(&inside, w, h));
            for threads in [2, 3, 8] {
                let (d, n) = with_threads(threads, || edt_nearest(&inside, w, h));
                assert_eq!((bits(&d), n), (bits(&d1), n1.clone()), "{w}x{h} on {threads} threads");
            }
            // And the nearest pixel is a seed at exactly that distance.
            for (i, (&d, &q)) in d1.iter().zip(&n1).enumerate() {
                if inside.iter().any(|&b| b) {
                    let (dx, dy) = ((i % w) as f32 - (q as usize % w) as f32, (i / w) as f32 - (q as usize / w) as f32);
                    assert!(inside[q as usize] && (dx.hypot(dy) - d).abs() < 1e-3, "{w}x{h} at {i}: {d} vs seed {q}");
                }
            }
        }
    }

    #[test]
    fn nearest_distances_survive_degenerate_sizes() {
        // The slice does not match the size, or there is nothing: no panic, and a result per input pixel.
        for (len, w, h) in [(0usize, 0usize, 0usize), (0, 5, 0), (0, 0, 5), (6, 2, 2), (6, 4, 2), (1, 0, 1)] {
            let (d, n) = edt_nearest(&vec![true; len], w, h);
            assert_eq!((d.len(), n.len()), (len, len), "{len} px as {w}x{h}");
        }
    }

    #[test]
    fn chisel_fields_do_not_depend_on_the_threads() {
        // Edge pixels at every angle, curves, patchy opacity and an edge along the border, on sizes that
        // leave short last groups; one thread (in order) is the reference.
        let mut maps = vec![
            aa_disc(97, 83, 30.0),
            aa_round_rect(131, 77, 9.0, 11.0, 120.0, 66.0, 14.0),
            aa_bar(120, 120, 33.0, 100.0, 30.0),
            aa_bar(90, 130, 60.0, 150.0, 25.0),
        ];
        let mut patchy = aa_disc(101, 101, 40.0);
        for (i, v) in patchy.v.iter_mut().enumerate() {
            if (i % 101 / 13 + i / 101 / 17) % 5 == 0 {
                *v *= 0.6;
            }
        }
        maps.push(patchy);
        maps.push(aa_bar(64, 64, 40.0, 220.0, 30.0));
        for (k, m) in maps.iter().enumerate() {
            for reach in [3.0f32, 12.0] {
                let one = with_threads(1, || (sharp_edges(m), distance_field_of(m, reach)));
                for threads in [2, 5, 16] {
                    let many = with_threads(threads, || (sharp_edges(m), distance_field_of(m, reach)));
                    assert_eq!(one.0.slot, many.0.slot, "map {k} sharp slots on {threads} threads");
                    let edges = |s: &SharpEdges| {
                        s.pixels.iter().map(|p| (p.at, p.cov.to_bits(), p.edge.offset.to_bits(), p.edge.normal.map(f32::to_bits))).collect::<Vec<_>>()
                    };
                    assert_eq!(edges(&one.0), edges(&many.0), "map {k} sharp edges on {threads} threads");
                    for (a, b) in one.1.iter().zip(&many.1) {
                        assert_eq!(bits(a), bits(b), "map {k}, reach {reach} on {threads} threads");
                    }
                }
            }
        }
    }

    /// The four fields that use the distance transform, of `m`.
    fn distance_field_of(m: &Map, reach: f32) -> Vec<Vec<f32>> {
        [FieldKind::Inside, FieldKind::Outside, FieldKind::BevelInside, FieldKind::BevelOutside]
            .map(|k| distance_field(k, m.v.clone(), m.w, m.h, reach))
            .to_vec()
    }

    fn aa_disc(w: usize, h: usize, r: f32) -> Map {
        let mut m = Map::new(w, h, 0.0);
        const N: usize = 16;
        for y in 0..h {
            for x in 0..w {
                let mut hit = 0;
                for sy in 0..N {
                    for sx in 0..N {
                        let (px, py) = (x as f32 + (sx as f32 + 0.5) / N as f32 - w as f32 / 2.0, y as f32 + (sy as f32 + 0.5) / N as f32 - h as f32 / 2.0);
                        hit += usize::from(px * px + py * py <= r * r);
                    }
                }
                m.v[y * w + x] = hit as f32 / (N * N) as f32;
            }
        }
        m
    }

    /// Integer-aligned rounded rectangle (x0..x1 x y0..y1, corner radius `rad`), anti-aliased by supersampling.
    fn aa_round_rect(w: usize, h: usize, x0: f32, y0: f32, x1: f32, y1: f32, rad: f32) -> Map {
        let sdf = |px: f32, py: f32| {
            let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
            let (qx, qy) = ((px - cx).abs() - ((x1 - x0) / 2.0 - rad), (py - cy).abs() - ((y1 - y0) / 2.0 - rad));
            qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0) - rad
        };
        let mut m = Map::new(w, h, 0.0);
        const N: usize = 16;
        for y in 0..h {
            for x in 0..w {
                let mut hit = 0;
                for sy in 0..N {
                    for sx in 0..N {
                        hit += usize::from(sdf(x as f32 + (sx as f32 + 0.5) / N as f32, y as f32 + (sy as f32 + 0.5) / N as f32) <= 0.0);
                    }
                }
                m.v[y * w + x] = hit as f32 / (N * N) as f32;
            }
        }
        m
    }

    #[test]
    fn chisel_hard_has_flat_facets() {
        let shape = square(48, 10, 38);
        let (m, _) = bevel_maps(
            &shape,
            &bevel_of(BevelStyle::InnerBevel, BevelTechnique::ChiselHard),
            &GlobalLight::default(),
            &no_tex(),
            &PreparedPatterns::new(&[], PREPARED_PATTERN_BYTES),
        );
        let at = |x: usize, y: usize| m[0].v[y * 48 + x];
        // A constant slope: equal highlight along the top facet.
        assert!(at(24, 12) > 0.1 && (at(24, 12) - at(24, 13)).abs() < 1e-3);
    }

    #[test]
    fn linear_bevel_contour_leaves_the_height_alone_at_any_range() {
        for range in [0.5, 0.7, 1.0] {
            let lut = bevel_contour_lut(&photocraft_doc::BevelContour { contour: Contour::Linear, range, anti_alias: false });
            for v in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
                assert!((lut_at(&lut, v) - v).abs() < 1e-3, "range {range}: {v} -> {}", lut_at(&lut, v));
            }
        }
    }

    #[test]
    fn bevel_contour_and_texture_shape_the_height() {
        let shape = square(40, 6, 34);
        let l = GlobalLight::default();
        let plain = bevel_of(BevelStyle::InnerBevel, BevelTechnique::Smooth);
        let (a, _) = bevel_maps(&shape, &plain, &l, &no_tex(), &PreparedPatterns::new(&[], PREPARED_PATTERN_BYTES));
        // A contour reshapes the profile (here a ramp folded at its middle).
        let mut c = plain.clone();
        let fold = vec![
            photocraft_doc::adjust::CurvePoint { input: 0.0, output: 0.0 },
            photocraft_doc::adjust::CurvePoint { input: 0.5, output: 1.0 },
            photocraft_doc::adjust::CurvePoint { input: 1.0, output: 0.0 },
        ];
        c.contour = Some(photocraft_doc::BevelContour {
            contour: Contour::Custom { name: "fold".into(), points: fold, corners: vec![] },
            range: 1.0,
            anti_alias: false,
        });
        let (b, _) = bevel_maps(&shape, &c, &l, &no_tex(), &PreparedPatterns::new(&[], PREPARED_PATTERN_BYTES));
        let diff: f32 = a[1].v.iter().zip(&b[1].v).map(|(x, y)| (x - y).abs()).sum();
        assert!(diff > 5.0, "{diff}");
        // A striped texture lights the flat middle.
        let mut sf = photocraft_raster::Surface::new(photocraft_color::PixelFormat::RGBA8);
        sf.fill_rect(Rect::new(0, 0, 4, 2), &[0.0, 0.0, 0.0, 1.0]);
        sf.fill_rect(Rect::new(0, 2, 4, 4), &[1.0, 1.0, 1.0, 1.0]);
        let pats = [Pattern::new("stripes", sf, 4, 4)];
        let mut t = plain.clone();
        t.texture = Some(photocraft_doc::BevelTexture {
            name: "stripes".into(),
            id: String::new(),
            scale: 1.0,
            depth: 1.0,
            invert: false,
            link: true,
            phase: (0.0, 0.0),
        });
        let tex = TextureCtx { rect: Rect::new(0, 0, 40, 40), patterns: &pats, anchor: (0.0, 0.0) };
        let (m, _) = bevel_maps(&shape, &t, &l, &tex, &PreparedPatterns::new(&pats, PREPARED_PATTERN_BYTES));
        let mid: f32 = (16..24).map(|y| m[0].v[y * 40 + 20] + m[1].v[y * 40 + 20]).sum();
        let flat: f32 = (16..24).map(|y| a[0].v[y * 40 + 20] + a[1].v[y * 40 + 20]).sum();
        assert!(mid > flat + 0.5, "{mid} vs {flat}");
    }

    #[test]
    fn glow_range_stretches_the_contour() {
        let mut g = Glow {
            common: photocraft_doc::FxCommon::new(BlendMode::Screen, 1.0),
            paint: FxPaint::Color(photocraft_color::Color::WHITE),
            technique: GlowTechnique::Softer,
            spread: 0.0,
            size: 10.0,
            contour: Contour::Linear,
            anti_alias: false,
            range: 0.5,
            jitter: 0.0,
            noise: 0.0,
            source: GlowSource::Edge,
        };
        let l = glow_lut(&g).unwrap();
        assert!((lut_at(&l, 0.25) - 0.5).abs() < 1e-3 && (lut_at(&l, 0.75) - 1.0).abs() < 1e-6);
        g.range = 1.0;
        assert!(glow_lut(&g).is_none());
    }

    /// Photoshop 27.11, measured on a 128 px square (Normal, black, 100 %, Range 50 %): a precise
    /// glow's coverage across the edge, and a centre-source inner glow as 1 − the edge glow.
    #[test]
    fn glows_match_photoshop() {
        let glow = |technique, size: f32, spread: f32, source| Glow {
            common: photocraft_doc::FxCommon::new(BlendMode::Normal, 1.0),
            paint: FxPaint::Color(photocraft_color::Color::BLACK),
            technique,
            spread,
            size,
            contour: Contour::Linear,
            anti_alias: false,
            range: 0.5,
            jitter: 0.0,
            noise: 0.0,
            source,
        };
        let n = 60;
        let mut shape = Map::new(n, n, 0.0);
        for y in 20..40 {
            for x in 20..40 {
                shape.v[y * n + x] = 1.0;
            }
        }
        let at = |m: &Map, x: usize| m.v[30 * n + x];
        // Outer, precise, size 10: Photoshop's black glow over white reads 47, 93, 140, 186, 232
        // at 2, 4, 6, 8 and 10 px out from the edge pixel (alpha 1 − D / 11).
        let m = glow_map(&shape, &glow(GlowTechnique::Precise, 10.0, 0.0, GlowSource::Edge), false, (0, 0));
        for (d, ps) in [(2, 47.0f32), (4, 93.0), (6, 140.0), (8, 186.0), (10, 232.0)] {
            let got = 255.0 * (1.0 - at(&m, 20 - d));
            assert!((got - ps).abs() <= 1.5, "outer precise at {d} px: {got} vs Photoshop {ps}");
        }
        // Spread 40 %: solid for the first 4 px.
        let m = glow_map(&shape, &glow(GlowTechnique::Precise, 10.0, 0.4, GlowSource::Edge), false, (0, 0));
        assert!((at(&m, 16) - 1.0).abs() < 1e-6 && at(&m, 14) < 1.0);
        // Centre source = 1 − edge, for both techniques.
        for t in [GlowTechnique::Precise, GlowTechnique::Softer] {
            let e = glow_map(&shape, &glow(t, 6.0, 0.0, GlowSource::Edge), true, (0, 0));
            let c = glow_map(&shape, &glow(t, 6.0, 0.0, GlowSource::Center), true, (0, 0));
            for x in 20..40 {
                assert!((at(&c, x) - (1.0 - at(&e, x))).abs() < 1e-5, "{t:?} at {x}");
            }
        }
    }

    #[test]
    fn offsets_follow_light() {
        let (dx, dy) = offset(120.0, 10.0);
        assert_eq!((dx, dy), (5.0, 9.0));
    }

    fn square_shape(n: usize) -> Map {
        let mut m = Map::new(n, n, 0.0);
        for y in 5..n - 5 {
            for x in 5..n - 5 {
                m.v[y * n + x] = 1.0;
            }
        }
        m
    }

    #[test]
    fn satin_at_distance_zero_matches_photoshop() {
        // Photoshop 27.11, Satin black 100 %, angle 90°, Linear contour, distance 0, size 5 over a
        // #808080 square: the rows from the top edge read 82, 102, 123, 128.
        let s = photocraft_doc::effects::Satin {
            common: photocraft_doc::FxCommon::new(BlendMode::Normal, 1.0),
            color: photocraft_color::Color::BLACK,
            angle: 90.0,
            distance: 0.0,
            size: 5.0,
            contour: Contour::Linear,
            anti_alias: false,
            invert: false,
        };
        let m = satin_map(&square_shape(60), &s);
        for (row, gray) in [(5, 82.0), (7, 102.0), (9, 123.0), (11, 128.0), (30, 128.0)] {
            let want = 1.0 - gray / 128.0;
            let got = m.v[row * 60 + 30];
            assert!((got - want).abs() < 0.02, "row {row}: {got} vs {want}");
        }
    }

    fn shadow(noise: f32, origin: (i32, i32)) -> Map {
        let s = Shadow {
            common: photocraft_doc::FxCommon::new(BlendMode::Multiply, 1.0),
            color: photocraft_color::Color::BLACK,
            angle: 0.0,
            use_global_light: false,
            distance: 0.0,
            spread: 0.0,
            size: 6.0,
            contour: Contour::Linear,
            anti_alias: false,
            noise,
            knocks_out: true,
        };
        shadow_map(&square_shape(40), &s, &GlobalLight::default(), false, origin)
    }

    #[test]
    fn shadow_noise_speckles_soft_coverage_deterministically() {
        let plain = shadow(0.0, (0, 0));
        let noisy = shadow(0.5, (0, 0));
        let again = shadow(0.5, (0, 0));
        let (mut diff, mut identical) = (0.0f32, true);
        for i in 0..plain.v.len() {
            diff += (plain.v[i] - noisy.v[i]).abs();
            if noisy.v[i] != again.v[i] {
                identical = false;
            }
        }
        assert!(diff > 5.0, "noise changes the soft band: {diff}");
        assert!(identical, "same coords, same speckle");
        // Every value stays a coverage.
        assert!(noisy.v.iter().all(|v| (0.0..=1.0).contains(v)));
        // The speckle moves with the document origin, not the tile.
        let moved = shadow(0.5, (100, 100));
        assert_ne!(noisy.v, moved.v, "hashed from document coordinates");
        // Full noise can punch holes even in the solid part.
        let heavy = shadow(1.0, (0, 0));
        assert!(heavy.v.iter().any(|v| *v < 0.5) && heavy.v.iter().any(|v| *v > 0.5), "heavy noise spans the range",);
    }

    #[test]
    fn glow_noise_follows_the_contour_and_range() {
        let g = |noise: f32| Glow {
            common: photocraft_doc::FxCommon::new(BlendMode::Screen, 1.0),
            paint: FxPaint::Color(photocraft_color::Color::WHITE),
            technique: GlowTechnique::Softer,
            spread: 0.0,
            size: 6.0,
            contour: Contour::Linear,
            anti_alias: false,
            range: 0.5,
            jitter: 0.0,
            noise,
            source: GlowSource::Edge,
        };
        let plain = glow_map(&square_shape(40), &g(0.0), false, (0, 0));
        let noisy = glow_map(&square_shape(40), &g(0.6), false, (0, 0));
        let diff: f32 = plain.v.iter().zip(&noisy.v).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 5.0, "glow noise changes the coverage: {diff}");
    }

    #[test]
    fn shifting_by_hostile_distances_does_not_overflow() {
        let m = square_shape(12);
        for d in [1.0e30, -1.0e30, f32::INFINITY, f32::NAN] {
            let s = m.shifted(d, d, 0.0);
            assert_eq!(s.v.len(), 144);
        }
    }
}
