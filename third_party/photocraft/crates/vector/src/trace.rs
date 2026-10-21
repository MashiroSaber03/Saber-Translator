//! Coverage mask → path ("Make Work Path" from a selection).
//!
//! 1. Marching squares at level 0.5 over pixel centres (linear interpolation along cell edges,
//!    saddles resolved with the cell average) gives closed, non-intersecting contours.
//! 2. Corners are found where the direction over a ±1.5 px window turns by more than 60°.
//! 3. Each run between corners is fitted with cubic Béziers by least squares with Newton
//!    reparameterisation, splitting at the worst point until within `tolerance` (P. J.
//!    Schneider, "An Algorithm for Automatically Fitting Digitized Curves", Graphics Gems, 1990).
//!
//! Contours are emitted as subpaths: the first combines, the rest *exclude*, which is the
//! even-odd area for nested, non-crossing contours regardless of their orientation.
//!
//! [`fit_trace`] runs steps 2 and 3 on a freehand pointer trace (the Freeform Pen), open or
//! closed, with a corner window that grows with the tolerance.

use std::collections::HashMap;

use photocraft_doc::{Knot, Path, PathOp, Subpath};
use photocraft_geom::{Point, Rect};

type P = (f64, f64);

/// Traces the 0.5 iso-contours of `values` (row-major over `rect`, values outside read as
/// `outside`) and fits curves within `tolerance` px.
pub fn trace_mask(values: &[f32], rect: Rect, outside: f32, tolerance: f64) -> Path {
    let loops = contours(values, rect, outside);
    let mut subpaths = Vec::new();
    for (i, lp) in loops.iter().enumerate() {
        if lp.len() < 3 {
            continue;
        }
        // Fit to half the tolerance: the worst point may deviate by the full amount only rarely.
        let mut s = fit_closed(lp, (tolerance * 0.5).max(0.05), 1.5);
        s.op = if i == 0 { PathOp::Combine } else { PathOp::Exclude };
        subpaths.push(s);
    }
    Path::new(subpaths)
}

/// Closed iso-contours at 0.5 through pixel centres.
pub fn contours(values: &[f32], rect: Rect, outside: f32) -> Vec<Vec<P>> {
    let w = rect.width() as i64;
    let h = rect.height() as i64;
    let get = |x: i64, y: i64| -> f32 { if x < 0 || y < 0 || x >= w || y >= h { outside } else { values[(y * w + x) as usize] } };
    // Edge ids: horizontal edge between (x,y)-(x+1,y) = (x, y, 0); vertical (x,y)-(x,y+1) = (x, y, 1).
    let lvl = 0.5f32;
    let point = |a: (i64, i64), b: (i64, i64)| -> P {
        let (va, vb) = (get(a.0, a.1), get(b.0, b.1));
        let t = if (vb - va).abs() > 1e-9 { f64::from((lvl - va) / (vb - va)).clamp(0.0, 1.0) } else { 0.5 };
        let (ax, ay) = (a.0 as f64 + 0.5, a.1 as f64 + 0.5);
        let (bx, by) = (b.0 as f64 + 0.5, b.1 as f64 + 0.5);
        (f64::from(rect.x0) + ax + (bx - ax) * t, f64::from(rect.y0) + ay + (by - ay) * t)
    };
    // Segments from edge -> edge, oriented with the inside on the left (y down).
    let mut next: HashMap<(i64, i64, u8), (i64, i64, u8)> = HashMap::new();
    for cy in -1..h {
        for cx in -1..w {
            let tl = get(cx, cy) >= lvl;
            let tr = get(cx + 1, cy) >= lvl;
            let br = get(cx + 1, cy + 1) >= lvl;
            let bl = get(cx, cy + 1) >= lvl;
            let idx = u8::from(tl) | u8::from(tr) << 1 | u8::from(br) << 2 | u8::from(bl) << 3;
            if idx == 0 || idx == 15 {
                continue;
            }
            let top = (cx, cy, 0u8);
            let bottom = (cx, cy + 1, 0u8);
            let left = (cx, cy, 1u8);
            let right = (cx + 1, cy, 1u8);
            // Walking along the contour with the inside on the left-hand side.
            let mut seg = |a: (i64, i64, u8), b: (i64, i64, u8)| {
                next.insert(a, b);
            };
            match idx {
                1 => seg(left, top),
                2 => seg(top, right),
                3 => seg(left, right),
                4 => seg(right, bottom),
                5 => {
                    let centre = (get(cx, cy) + get(cx + 1, cy) + get(cx + 1, cy + 1) + get(cx, cy + 1)) / 4.0;
                    if centre >= lvl {
                        seg(left, bottom);
                        seg(right, top);
                    } else {
                        seg(left, top);
                        seg(right, bottom);
                    }
                }
                6 => seg(top, bottom),
                7 => seg(left, bottom),
                8 => seg(bottom, left),
                9 => seg(bottom, top),
                10 => {
                    let centre = (get(cx, cy) + get(cx + 1, cy) + get(cx + 1, cy + 1) + get(cx, cy + 1)) / 4.0;
                    if centre >= lvl {
                        seg(top, left);
                        seg(bottom, right);
                    } else {
                        seg(top, right);
                        seg(bottom, left);
                    }
                }
                11 => seg(bottom, right),
                12 => seg(right, left),
                13 => seg(right, top),
                14 => seg(top, left),
                _ => {}
            }
        }
    }
    let edge_point = |e: (i64, i64, u8)| if e.2 == 0 { point((e.0, e.1), (e.0 + 1, e.1)) } else { point((e.0, e.1), (e.0, e.1 + 1)) };
    let mut keys: Vec<(i64, i64, u8)> = next.keys().copied().collect();
    keys.sort_unstable_by_key(|k| (k.1, k.0, k.2));
    let mut out = Vec::new();
    for start in keys {
        if !next.contains_key(&start) {
            continue;
        }
        let mut lp = Vec::new();
        let mut cur = start;
        while let Some(n) = next.remove(&cur) {
            lp.push(edge_point(cur));
            cur = n;
            if cur == start {
                break;
            }
        }
        // Drop consecutive duplicates (interpolation can land on a corner).
        lp.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9);
        if lp.len() >= 3 {
            out.push(lp);
        }
    }
    out
}

fn sub(a: P, b: P) -> P {
    (a.0 - b.0, a.1 - b.1)
}
fn add(a: P, b: P) -> P {
    (a.0 + b.0, a.1 + b.1)
}
fn mul(a: P, k: f64) -> P {
    (a.0 * k, a.1 * k)
}
fn dot(a: P, b: P) -> f64 {
    a.0 * b.0 + a.1 * b.1
}
fn len(a: P) -> f64 {
    a.0.hypot(a.1)
}
fn norm(a: P) -> P {
    let l = len(a);
    if l > 1e-12 { (a.0 / l, a.1 / l) } else { (0.0, 0.0) }
}

/// Point `window` px of arc length away from `i` (forwards or backwards) on a closed loop, or on
/// an open polyline, where the walk stops at the ends.
fn walk(lp: &[P], i: usize, window: f64, fwd: bool, closed: bool) -> P {
    let n = lp.len();
    let mut acc = 0.0;
    let mut cur = i;
    for _ in 0..n {
        if !closed && (if fwd { cur + 1 >= n } else { cur == 0 }) {
            break;
        }
        let nx = if fwd { (cur + 1) % n } else { (cur + n - 1) % n };
        let d = len(sub(lp[nx], lp[cur]));
        if acc + d >= window {
            let t = (window - acc) / d.max(1e-12);
            return add(lp[cur], mul(sub(lp[nx], lp[cur]), t));
        }
        acc += d;
        cur = nx;
    }
    lp[cur]
}

/// Indices of corner points: where the direction over ±`window` px turns by more than 60°. The
/// ends of an open polyline are never corners.
fn corners(lp: &[P], window: f64, closed: bool) -> Vec<usize> {
    let n = lp.len();
    let cos_thresh = (60f64).to_radians().cos();
    let turn: Vec<f64> = (0..n)
        .map(|i| {
            if !closed && (i == 0 || i + 1 == n) {
                return 1.0;
            }
            let a = norm(sub(lp[i], walk(lp, i, window, false, closed)));
            let b = norm(sub(walk(lp, i, window, true, closed), lp[i]));
            dot(a, b)
        })
        .collect();
    let mut out = Vec::new();
    for i in 0..n {
        if turn[i] >= cos_thresh {
            continue;
        }
        // Local minimum of the cosine within ±window (ties: first index).
        let mut is_min = true;
        let mut acc = 0.0;
        let mut j = i;
        while acc < window {
            if !closed && j + 1 >= n {
                break;
            }
            let nx = (j + 1) % n;
            acc += len(sub(lp[nx], lp[j]));
            j = nx;
            if j == i {
                break;
            }
            if turn[j] < turn[i] {
                is_min = false;
                break;
            }
        }
        acc = 0.0;
        j = i;
        while is_min && acc < window {
            if !closed && j == 0 {
                break;
            }
            let pv = (j + n - 1) % n;
            acc += len(sub(lp[j], lp[pv]));
            j = pv;
            if j == i {
                break;
            }
            if turn[j] <= turn[i] {
                is_min = false;
            }
        }
        if is_min {
            out.push(i);
        }
    }
    out
}

/// Fits a closed loop with cubic Béziers; `window` is the corner detection window.
fn fit_closed(lp: &[P], tol: f64, window: f64) -> Subpath {
    let n = lp.len();
    let cs = corners(lp, window, true);
    let mut segs: Vec<[P; 4]> = Vec::new();
    let mut corner_at_start: Vec<bool> = Vec::new();
    if cs.is_empty() {
        // Smooth loop: start at 0 with a shared tangent at the seam.
        let mut pts: Vec<P> = lp.to_vec();
        pts.push(lp[0]);
        let t = norm(sub(lp[1 % n], lp[n - 1]));
        let before = segs.len();
        fit_cubic(&pts, t, mul(t, -1.0), tol * tol, &mut segs, 0);
        corner_at_start.extend((before..segs.len()).map(|_| false));
    } else {
        for (k, &c) in cs.iter().enumerate() {
            let end = cs[(k + 1) % cs.len()];
            let mut pts = vec![lp[c]];
            let mut j = c;
            loop {
                j = (j + 1) % n;
                pts.push(lp[j]);
                if j == end {
                    break;
                }
            }
            let m = pts.len();
            let t1 = norm(sub(pts[1.min(m - 1)], pts[0]));
            let t2 = norm(sub(pts[m.saturating_sub(2)], pts[m - 1]));
            let before = segs.len();
            fit_cubic(&pts, t1, t2, tol * tol, &mut segs, 0);
            corner_at_start.extend((before..segs.len()).enumerate().map(|(q, _)| q == 0));
        }
    }
    let mut knots: Vec<Knot> = Vec::with_capacity(segs.len());
    let pt = |p: P| Point::new(p.0, p.1);
    for (i, s) in segs.iter().enumerate() {
        knots.push(Knot { anchor: pt(s[0]), in_ctrl: pt(s[0]), out_ctrl: pt(s[1]), smooth: !corner_at_start[i] });
    }
    let m = segs.len();
    for i in 0..m {
        knots[(i + 1) % m].in_ctrl = pt(segs[i][2]);
    }
    Subpath { closed: true, knots, op: PathOp::Combine }
}

/// Most pointer samples [`fit_trace`] fits: a longer trace is thinned evenly first, so the work
/// stays bounded for any stroke.
pub const MAX_TRACE_POINTS: usize = 4096;

/// Fits a freehand pointer trace (the Freeform Pen's drag) with cubic Béziers that stay within
/// `tolerance` px of every sample: Photoshop's Curve Fit, clamped to 0.1–100 px (2 px when not
/// finite). Turns sharper than 60° over ±2 × `tolerance` become corner anchors, the other inner
/// anchors are smooth. A `closed` trace becomes a closed subpath. Non-finite samples are skipped;
/// `None` when fewer than two distinct points remain (three when `closed`).
pub fn fit_trace(points: &[[f64; 2]], tolerance: f64, closed: bool) -> Option<Subpath> {
    let tol = if tolerance.is_finite() { tolerance.clamp(0.1, 100.0) } else { 2.0 };
    let window = (2.0 * tol).max(1.5);
    let min_step = (tol * 0.25).min(0.5);
    let stride = points.len().div_ceil(MAX_TRACE_POINTS).max(1);
    let last = points.len().saturating_sub(1);
    let mut lp: Vec<P> = Vec::with_capacity(points.len().min(MAX_TRACE_POINTS + 1));
    for (i, p) in points.iter().enumerate() {
        // Far-out coordinates would overflow the least-squares sums.
        if (i % stride != 0 && i != last) || !p.iter().all(|v| v.is_finite() && v.abs() < 1e9) {
            continue;
        }
        let q = (p[0], p[1]);
        match lp.last_mut() {
            // The release point is kept: it replaces a sample just before it.
            Some(l) if len(sub(q, *l)) < min_step => {
                if i == last && lp.len() > 1 {
                    *lp.last_mut()? = q;
                }
            }
            _ => lp.push(q),
        }
    }
    if closed {
        while lp.len() > 3 && lp.last().zip(lp.first()).is_some_and(|(a, b)| len(sub(*a, *b)) < min_step) {
            lp.pop();
        }
        return (lp.len() >= 3).then(|| fit_closed(&lp, tol, window));
    }
    (lp.len() >= 2).then(|| fit_open(&lp, tol, window))
}

/// Fits an open polyline with cubic Béziers, one run between consecutive corners at a time.
fn fit_open(lp: &[P], tol: f64, window: f64) -> Subpath {
    let n = lp.len();
    let mut cuts = vec![0];
    cuts.extend(corners(lp, window, false));
    cuts.push(n - 1);
    cuts.dedup();
    let mut segs: Vec<[P; 4]> = Vec::new();
    let mut corner_at_start: Vec<bool> = Vec::new();
    for w in cuts.windows(2) {
        let Some(pts) = lp.get(w[0]..=w[1]) else { continue };
        let m = pts.len();
        if m < 2 {
            continue;
        }
        let (a, b) = (pts[0], pts[m - 1]);
        let before = segs.len();
        if pts.iter().all(|&q| segment_distance(q, a, b) <= tol) {
            // A straight run (a drag along a ruler, a zig-zag leg) is one straight segment: end
            // tangents estimated from jittery samples would bend it.
            let third = mul(sub(b, a), 1.0 / 3.0);
            segs.push([a, add(a, third), sub(b, third), b]);
        } else {
            // End tangents over a stretch of the trace, not the first jittery sample.
            let reach = (2.0 * window).min(len(sub(b, a)) / 3.0).max(window);
            let t1 = norm(sub(walk(pts, 0, reach, true, false), a));
            let t2 = norm(sub(walk(pts, m - 1, reach, false, false), b));
            fit_cubic(pts, t1, t2, tol * tol, &mut segs, 0);
        }
        corner_at_start.extend((before..segs.len()).map(|k| k == before));
    }
    let pt = |p: P| Point::new(p.0, p.1);
    let mut knots: Vec<Knot> = Vec::with_capacity(segs.len() + 1);
    for (i, s) in segs.iter().enumerate() {
        let in_ctrl = if i == 0 { s[0] } else { segs[i - 1][2] };
        knots.push(Knot { anchor: pt(s[0]), in_ctrl: pt(in_ctrl), out_ctrl: pt(s[1]), smooth: i > 0 && !corner_at_start[i] });
    }
    if let Some(s) = segs.last() {
        knots.push(Knot { anchor: pt(s[3]), in_ctrl: pt(s[2]), out_ctrl: pt(s[3]), smooth: false });
    }
    Subpath { closed: false, knots, op: PathOp::Combine }
}

/// Distance from `q` to the segment `a`–`b`.
fn segment_distance(q: P, a: P, b: P) -> f64 {
    let ab = sub(b, a);
    let l2 = dot(ab, ab);
    let t = if l2 > 1e-18 { (dot(sub(q, a), ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
    len(sub(q, add(a, mul(ab, t))))
}

fn bezier(b: &[P; 4], t: f64) -> P {
    let mt = 1.0 - t;
    let (a, bb, c, d) = (mt * mt * mt, 3.0 * mt * mt * t, 3.0 * mt * t * t, t * t * t);
    (a * b[0].0 + bb * b[1].0 + c * b[2].0 + d * b[3].0, a * b[0].1 + bb * b[1].1 + c * b[2].1 + d * b[3].1)
}

fn bezier_d1(b: &[P; 4], t: f64) -> P {
    let mt = 1.0 - t;
    let q0 = mul(sub(b[1], b[0]), 3.0);
    let q1 = mul(sub(b[2], b[1]), 3.0);
    let q2 = mul(sub(b[3], b[2]), 3.0);
    add(add(mul(q0, mt * mt), mul(q1, 2.0 * mt * t)), mul(q2, t * t))
}

fn bezier_d2(b: &[P; 4], t: f64) -> P {
    let r0 = mul(add(sub(b[2], mul(b[1], 2.0)), b[0]), 6.0);
    let r1 = mul(add(sub(b[3], mul(b[2], 2.0)), b[1]), 6.0);
    add(mul(r0, 1.0 - t), mul(r1, t))
}

fn chord_params(pts: &[P]) -> Vec<f64> {
    let mut u = vec![0.0; pts.len()];
    for i in 1..pts.len() {
        u[i] = u[i - 1] + len(sub(pts[i], pts[i - 1]));
    }
    let total = u[pts.len() - 1];
    if total > 0.0 {
        for v in &mut u {
            *v /= total;
        }
    }
    u
}

/// Least-squares control points with fixed end tangents.
fn generate(pts: &[P], u: &[f64], t1: P, t2: P) -> [P; 4] {
    let (p0, p3) = (pts[0], pts[pts.len() - 1]);
    let mut c = [[0.0f64; 2]; 2];
    let mut x = [0.0f64; 2];
    for (i, &t) in u.iter().enumerate() {
        let mt = 1.0 - t;
        let b1 = 3.0 * mt * mt * t;
        let b2 = 3.0 * mt * t * t;
        let a1 = mul(t1, b1);
        let a2 = mul(t2, b2);
        c[0][0] += dot(a1, a1);
        c[0][1] += dot(a1, a2);
        c[1][1] += dot(a2, a2);
        let b0 = mt * mt * mt;
        let b3 = t * t * t;
        let tmp = sub(pts[i], add(mul(p0, b0 + b1), mul(p3, b2 + b3)));
        x[0] += dot(a1, tmp);
        x[1] += dot(a2, tmp);
    }
    c[1][0] = c[0][1];
    let det = c[0][0] * c[1][1] - c[1][0] * c[0][1];
    let det0 = c[0][0] * x[1] - c[1][0] * x[0];
    let det1 = x[0] * c[1][1] - x[1] * c[0][1];
    let (mut al, mut ar) = if det.abs() > 1e-12 { (det1 / det, det0 / det) } else { (0.0, 0.0) };
    let seg = len(sub(p3, p0));
    let eps = 1e-6 * seg;
    if al < eps || ar < eps {
        al = seg / 3.0;
        ar = seg / 3.0;
    }
    [p0, add(p0, mul(t1, al)), add(p3, mul(t2, ar)), p3]
}

fn max_error(pts: &[P], b: &[P; 4], u: &[f64]) -> (f64, usize) {
    let mut best = (0.0, pts.len() / 2);
    for i in 1..pts.len() - 1 {
        let d = sub(bezier(b, u[i]), pts[i]);
        let e = dot(d, d);
        if e >= best.0 {
            best = (e, i);
        }
    }
    best
}

fn reparam(pts: &[P], u: &[f64], b: &[P; 4]) -> Vec<f64> {
    pts.iter()
        .zip(u)
        .map(|(p, &t)| {
            let q = bezier(b, t);
            let d1 = bezier_d1(b, t);
            let d2 = bezier_d2(b, t);
            let num = dot(sub(q, *p), d1);
            let den = dot(d1, d1) + dot(sub(q, *p), d2);
            if den.abs() < 1e-12 { t } else { (t - num / den).clamp(0.0, 1.0) }
        })
        .collect()
}

fn fit_cubic(pts: &[P], t1: P, t2: P, err2: f64, out: &mut Vec<[P; 4]>, depth: u32) {
    let n = pts.len();
    if n == 2 || depth > 40 {
        let d = len(sub(pts[n - 1], pts[0])) / 3.0;
        out.push([pts[0], add(pts[0], mul(t1, d)), add(pts[n - 1], mul(t2, d)), pts[n - 1]]);
        return;
    }
    let mut u = chord_params(pts);
    let mut b = generate(pts, &u, t1, t2);
    let (mut e, mut split) = max_error(pts, &b, &u);
    if e < err2 {
        out.push(b);
        return;
    }
    if e < err2 * 16.0 {
        for _ in 0..6 {
            let u2 = reparam(pts, &u, &b);
            let b2 = generate(pts, &u2, t1, t2);
            let (e2, s2) = max_error(pts, &b2, &u2);
            u = u2;
            b = b2;
            e = e2;
            split = s2;
            if e < err2 {
                out.push(b);
                return;
            }
        }
    }
    let split = split.clamp(1, n - 2);
    let tc = norm(sub(pts[split - 1], pts[split + 1]));
    let tc = if len(tc) < 0.5 { norm(sub(pts[split - 1], pts[split])) } else { tc };
    fit_cubic(&pts[..=split], t1, tc, err2, out, depth + 1);
    fit_cubic(&pts[split..], mul(tc, -1.0), t2, err2, out, depth + 1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_contour_orientation_and_count() {
        let (w, h) = (6, 5);
        let mut v = vec![0.0f32; w * h];
        for y in 1..4 {
            for x in 1..5 {
                v[y * w + x] = 1.0;
            }
        }
        let c = contours(&v, Rect::new(0, 0, w as i32, h as i32), 0.0);
        assert_eq!(c.len(), 1);
        // Inside on the left in y-down screen space = counter-clockwise = negative shoelace area.
        let a: f64 = (0..c[0].len())
            .map(|i| {
                let (p, q) = (c[0][i], c[0][(i + 1) % c[0].len()]);
                p.0 * q.1 - q.0 * p.1
            })
            .sum::<f64>()
            * 0.5;
        assert!(a < 0.0, "area {a}");
        let a = -a;
        // Pixels 1..5 x 1..4 → contour through their outer half-pixel boundary (4×3 area with
        // 45° chamfered corners: 12 − 4·0.125).
        assert!((a - 11.5).abs() < 1e-9, "area {a}");
    }

    #[test]
    fn traced_rect_has_corners() {
        let (w, h) = (40, 30);
        let mut v = vec![0.0f32; w * h];
        for y in 5..25 {
            for x in 5..35 {
                v[y * w + x] = 1.0;
            }
        }
        let p = trace_mask(&v, Rect::new(0, 0, w as i32, h as i32), 0.0, 1.0);
        assert_eq!(p.subpaths.len(), 1);
        let k = &p.subpaths[0].knots;
        assert!(k.len() >= 4 && k.len() <= 12, "{} knots", k.len());
        assert!(k.iter().filter(|k| !k.smooth).count() >= 4);
    }

    /// Largest distance from a sample to the fitted subpath (each segment densely sampled).
    fn max_deviation(sp: &Subpath, samples: &[[f64; 2]]) -> f64 {
        let curve: Vec<P> = sp
            .segments()
            .iter()
            .flat_map(|s| {
                let b = [(s[0].x, s[0].y), (s[1].x, s[1].y), (s[2].x, s[2].y), (s[3].x, s[3].y)];
                (0..=400).map(move |k| bezier(&b, k as f64 / 400.0))
            })
            .collect();
        samples.iter().map(|q| curve.iter().map(|c| len(sub(*c, (q[0], q[1])))).fold(f64::MAX, f64::min)).fold(0.0, f64::max)
    }

    /// Freeform Pen fixtures at the default 2 px Curve Fit. Every sample stays within the
    /// tolerance (plus 0.05 px for sampling the fitted curve); the anchor bounds follow from the
    /// geometry: a straight drag needs no inner anchor, the zig-zag's five legs need their four
    /// corners and nothing between, and a circle of radius 80 is within 0.03 px of four cubic
    /// quarter arcs, so the fit never needs more than eight anchors; a sine never more than two
    /// segments per half period.
    #[test]
    fn fit_trace_follows_fixtures_with_few_anchors() {
        use std::f64::consts::TAU;
        let tol = 2.0;
        // Straight drag with ±0.5 px of hand jitter, 3 px apart.
        let line: Vec<[f64; 2]> = (0..=100).map(|i| [10.0 + 3.0 * i as f64, 50.0 + if i % 2 == 0 { 0.5 } else { -0.5 }]).collect();
        let sp = fit_trace(&line, tol, false).unwrap();
        assert!(!sp.closed);
        assert_eq!(sp.knots.len(), 2, "a straight drag is one segment");
        assert!(max_deviation(&sp, &line) <= tol + 0.05);
        // Zig-zag: five 60 px legs at right angles, sampled every 2 px.
        let turns = [[0.0, 0.0], [60.0, 0.0], [60.0, 60.0], [120.0, 60.0], [120.0, 120.0], [180.0, 120.0]];
        let zig: Vec<[f64; 2]> = turns
            .windows(2)
            .flat_map(|w| (0..30).map(move |k| [w[0][0] + (w[1][0] - w[0][0]) * k as f64 / 30.0, w[0][1] + (w[1][1] - w[0][1]) * k as f64 / 30.0]))
            .chain([turns[5]])
            .collect();
        let sp = fit_trace(&zig, tol, false).unwrap();
        let anchors: Vec<[f64; 2]> = sp.knots.iter().map(|k| [k.anchor.x, k.anchor.y]).collect();
        assert_eq!(anchors, turns, "the corners are kept as anchors, nothing between");
        assert!(sp.knots[1..5].iter().all(|k| !k.smooth));
        assert!(max_deviation(&sp, &zig) <= tol + 0.05);
        // Circle, radius 80, 180 samples, closed.
        let circle: Vec<[f64; 2]> = (0..180).map(|i| (i as f64).to_radians() * 2.0).map(|a| [100.0 + 80.0 * a.cos(), 100.0 + 80.0 * a.sin()]).collect();
        let sp = fit_trace(&circle, tol, true).unwrap();
        assert!(sp.closed);
        assert!((2..=8).contains(&sp.knots.len()), "{} anchors", sp.knots.len());
        assert!(sp.knots.iter().all(|k| k.smooth), "a circle has no corners");
        assert!(max_deviation(&sp, &circle) <= tol + 0.05);
        // Two periods of a sine wave (amplitude 40, wavelength 200) with the same jitter: at most
        // two segments per half period.
        let sine: Vec<[f64; 2]> =
            (0..=200).map(|i| i as f64 * 2.0).enumerate().map(|(i, x)| [x, 100.0 + 40.0 * (x / 200.0 * TAU).sin() + 0.5 - (i % 2) as f64]).collect();
        let sp = fit_trace(&sine, tol, false).unwrap();
        assert!((3..=9).contains(&sp.knots.len()), "{} anchors", sp.knots.len());
        assert!(sp.knots[1..sp.knots.len() - 1].iter().all(|k| k.smooth));
        assert!(max_deviation(&sp, &sine) <= tol + 0.05);
    }

    #[test]
    fn fit_trace_survives_hostile_input() {
        assert!(fit_trace(&[], 2.0, false).is_none());
        assert!(fit_trace(&[[1.0, 1.0]; 50], 2.0, false).is_none());
        assert!(fit_trace(&[[f64::NAN, 0.0], [f64::INFINITY, 1.0], [1e300, 2.0]], f64::NAN, true).is_none());
        let long: Vec<[f64; 2]> = (0..200_000).map(|i| [i as f64 * 0.01, ((i as f64) * 0.001).sin() * 40.0]).collect();
        let sp = fit_trace(&long, -5.0, false).unwrap();
        assert!(sp.knots.len() >= 2 && sp.knots.iter().all(|k| k.anchor.x.is_finite() && k.out_ctrl.y.is_finite()));
    }
}
