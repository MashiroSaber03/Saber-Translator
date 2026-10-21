//! Non-affine maps of paths, so shape layers and vector masks stay vectors through Edit ›
//! Transform › Distort, Perspective and Warp (#3009).
//!
//! * [`map_points`] sends every anchor and control point through the map and keeps the knots as
//!   they are: what a projective map (Distort, Perspective) needs. Straight segments stay exactly
//!   straight; a curve keeps its exact end tangents and approximates the (rational) image between.
//! * [`map_path`] follows a curved map (Warp): it splits each segment where the mapped control
//!   points stray from the mapped curve by more than a tolerance, so the result adds anchors where
//!   the warp bends it.

use photocraft_doc::{Knot, Path, Subpath};
use photocraft_geom::Point;

/// Every anchor and control point of `path` through `f` (no subdivision).
pub fn map_points(path: &Path, f: impl Fn(Point) -> Point) -> Path {
    let mut out = path.clone();
    for s in &mut out.subpaths {
        for k in &mut s.knots {
            *k = Knot { anchor: f(k.anchor), in_ctrl: f(k.in_ctrl), out_ctrl: f(k.out_ctrl), smooth: k.smooth };
        }
    }
    out
}

/// Pieces a path may be split into, in total: bounds the work and memory on hostile input.
const MAX_PIECES: usize = 1 << 20;
/// Deepest split of one segment (2^6 = 64 pieces).
const MAX_DEPTH: u32 = 6;

/// `path` through a smooth, possibly non-linear map `f`: each segment is split in halves until its
/// mapped control points follow the mapped curve within `tol` px (at most 64 pieces a segment).
/// A straight segment whose image is straight stays a straight segment.
pub fn map_path(path: &Path, f: impl Fn(Point) -> Point, tol: f64) -> Path {
    let tol = if tol.is_finite() && tol > 0.0 { tol } else { 0.1 };
    let segments: usize = path.subpaths.iter().map(|s| s.knots.len()).sum();
    // Fewer splits for huge paths, so the result stays bounded.
    let mut depth = MAX_DEPTH;
    while depth > 0 && segments.saturating_mul(1 << depth) > MAX_PIECES {
        depth -= 1;
    }
    let mut out = path.clone();
    out.subpaths = path.subpaths.iter().map(|s| map_subpath(s, &f, tol, depth)).collect();
    out
}

fn map_subpath(s: &Subpath, f: &impl Fn(Point) -> Point, tol: f64, depth: u32) -> Subpath {
    let n = s.knots.len();
    let segs = s.segments();
    if segs.is_empty() {
        return map_points(&Path::new(vec![s.clone()]), f).subpaths.pop().unwrap_or_default();
    }
    // The mapped pieces of each segment (segment i runs from knot i to knot i + 1).
    let pieces: Vec<Vec<[Point; 4]>> = segs.iter().map(|seg| map_segment(seg, f, tol, depth)).collect();
    let mut knots = Vec::with_capacity(pieces.iter().map(Vec::len).sum::<usize>() + 1);
    for (i, k) in s.knots.iter().enumerate() {
        let incoming = if i > 0 {
            pieces.get(i - 1)
        } else if s.closed {
            pieces.get(n - 1)
        } else {
            None
        };
        let in_ctrl = incoming.and_then(|p| p.last()).map_or_else(|| f(k.in_ctrl), |p| p[2]);
        let outgoing = pieces.get(i);
        let out_ctrl = outgoing.and_then(|p| p.first()).map_or_else(|| f(k.out_ctrl), |p| p[1]);
        knots.push(Knot { anchor: f(k.anchor), in_ctrl, out_ctrl, smooth: k.smooth });
        // The anchors the split added inside segment i: smooth, as the curve runs through them.
        if let Some(p) = outgoing {
            for w in p.windows(2) {
                knots.push(Knot { anchor: w[1][0], in_ctrl: w[0][2], out_ctrl: w[1][1], smooth: true });
            }
        }
    }
    Subpath { closed: s.closed, knots, op: s.op }
}

fn lerp(a: Point, b: Point, t: f64) -> Point {
    Point::new(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
}

fn dist(a: Point, b: Point) -> f64 {
    (a.x - b.x).hypot(a.y - b.y)
}

fn eval(s: &[Point; 4], t: f64) -> Point {
    let (ab, bc, cd) = (lerp(s[0], s[1], t), lerp(s[1], s[2], t), lerp(s[2], s[3], t));
    lerp(lerp(ab, bc, t), lerp(bc, cd, t), t)
}

/// De Casteljau split at `t = ½`.
fn halves(s: &[Point; 4]) -> ([Point; 4], [Point; 4]) {
    let (ab, bc, cd) = (lerp(s[0], s[1], 0.5), lerp(s[1], s[2], 0.5), lerp(s[2], s[3], 0.5));
    let (abc, bcd) = (lerp(ab, bc, 0.5), lerp(bc, cd, 0.5));
    let m = lerp(abc, bcd, 0.5);
    ([s[0], ab, abc, m], [m, bcd, cd, s[3]])
}

/// Distance from `p` to the line through `a` and `b` (to `a` when they coincide).
fn off_line(p: Point, a: Point, b: Point) -> f64 {
    let l = dist(a, b);
    if l < 1e-12 { dist(p, a) } else { ((b.x - a.x) * (a.y - p.y) - (a.x - p.x) * (b.y - a.y)).abs() / l }
}

const PROBES: [f64; 3] = [0.25, 0.5, 0.75];

fn map_segment(seg: &[Point; 4], f: &impl Fn(Point) -> Point, tol: f64, depth: u32) -> Vec<[Point; 4]> {
    // A straight segment (retracted handles) whose image is straight stays one.
    if seg[1] == seg[0] && seg[2] == seg[3] {
        let (a, b) = (f(seg[0]), f(seg[3]));
        if PROBES.iter().all(|&t| off_line(f(lerp(seg[0], seg[3], t)), a, b) <= tol) {
            return vec![[a, a, b, b]];
        }
        // Curved by the map: the same line as an evenly parametrized cubic, then split.
        let line = [seg[0], lerp(seg[0], seg[3], 1.0 / 3.0), lerp(seg[0], seg[3], 2.0 / 3.0), seg[3]];
        let mut out = Vec::new();
        split_mapped(&line, f, tol, depth, &mut out);
        return out;
    }
    let mut out = Vec::new();
    split_mapped(seg, f, tol, depth, &mut out);
    out
}

fn split_mapped(seg: &[Point; 4], f: &impl Fn(Point) -> Point, tol: f64, depth: u32, out: &mut Vec<[Point; 4]>) {
    let mapped = seg.map(f);
    let fits = PROBES.iter().all(|&t| dist(f(eval(seg, t)), eval(&mapped, t)) <= tol);
    if fits || depth == 0 {
        out.push(mapped);
        return;
    }
    let (a, b) = halves(seg);
    split_mapped(&a, f, tol, depth - 1, out);
    split_mapped(&b, f, tol, depth - 1, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> Path {
        Path::new(vec![Subpath::polygon(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])])
    }

    #[test]
    fn map_points_moves_every_point_and_keeps_the_knots() {
        let p = map_points(&square(), |q| Point::new(q.x * 2.0, q.y + q.x));
        let k = &p.subpaths[0].knots;
        assert_eq!(k.len(), 4);
        assert_eq!(k[2].anchor, Point::new(20.0, 20.0));
        assert_eq!(k[2].in_ctrl, k[2].anchor);
    }

    #[test]
    fn map_path_keeps_lines_that_stay_straight_and_splits_bent_ones() {
        // Linear map: no splits, straight segments stay straight.
        let p = map_path(&square(), |q| Point::new(q.x + q.y * 0.5, q.y), 0.1);
        assert_eq!(p.subpaths[0].knots.len(), 4);
        assert!(p.subpaths[0].closed);
        // A bend: the top edge (y = 0) arcs, so it gains anchors that lie on the mapped curve.
        let bend = |q: Point| Point::new(q.x, q.y - (q.x - 5.0).powi(2) * 0.2);
        let p = map_path(&square(), bend, 0.01);
        let k = &p.subpaths[0].knots;
        assert!(k.len() > 4, "{}", k.len());
        assert_eq!(k[0].anchor, bend(Point::new(0.0, 0.0)));
        for s in p.subpaths[0].segments() {
            let mid = eval(&s, 0.5);
            // Every point on the top edge maps to y = -(x - 5)² / 5.
            if s[0].y <= 0.0 && s[3].y <= 0.0 {
                assert!((mid.y + (mid.x - 5.0).powi(2) * 0.2).abs() < 0.05, "{mid:?}");
            }
        }
        // The vertical sides are only shifted, so they stay single straight segments.
        let right = k.iter().position(|k| k.anchor == bend(Point::new(10.0, 0.0))).unwrap();
        assert_eq!(k[right + 1].anchor, bend(Point::new(10.0, 10.0)));
    }
}
