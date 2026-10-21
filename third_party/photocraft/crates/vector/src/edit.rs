//! Point-level path editing, Photoshop's Direct Selection, Convert Point and Add / Delete Anchor
//! Point tools: move anchors, drag direction handles, reshape a segment by dragging it, convert
//! between smooth and corner points, insert and remove anchors, and hit-test a path's parts. Pure geometry on [`Path`]; the engine's `path.*` edit
//! commands and the canvas previews share it, so a preview is exactly what the command commits.
//!
//! Knots are addressed as `[subpath, knot]`. Every edit validates its indices and returns an
//! error message instead of panicking.

use photocraft_doc::{Knot, Path};
use photocraft_geom::Point;

/// One of a knot's two direction handles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Handle {
    /// The control point of the segment arriving at the knot.
    In,
    /// The control point of the segment leaving the knot.
    Out,
}

impl Handle {
    pub fn parse(s: &str) -> Option<Handle> {
        match s {
            "in" => Some(Handle::In),
            "out" => Some(Handle::Out),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Handle::In => "in",
            Handle::Out => "out",
        }
    }
}

/// The part of a path under the pointer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Hit {
    Anchor([usize; 2]),
    Handle([usize; 2], Handle),
    /// The segment leaving knot `[subpath, knot]`, at curve parameter `t`.
    Segment([usize; 2], f64),
}

fn add(p: Point, d: [f64; 2]) -> Point {
    Point::new(p.x + d[0], p.y + d[1])
}

fn dist(a: Point, b: Point) -> f64 {
    (a.x - b.x).hypot(a.y - b.y)
}

fn knot_mut(path: &mut Path, [s, k]: [usize; 2]) -> Result<&mut Knot, String> {
    path.subpaths.get_mut(s).and_then(|sp| sp.knots.get_mut(k)).ok_or_else(|| format!("no knot {k} in subpath {s}"))
}

/// The knot ending the segment that leaves `[s, k]` (wrapping on closed subpaths).
fn segment_end(path: &Path, [s, k]: [usize; 2]) -> Result<[usize; 2], String> {
    let sp = path.subpaths.get(s).ok_or_else(|| format!("no subpath {s}"))?;
    let n = sp.knots.len();
    match k.checked_add(1) {
        Some(next) if k < n && next < n => Ok([s, next]),
        Some(_) if k < n && sp.closed && n > 1 => Ok([s, 0]),
        _ => Err(format!("no segment leaves knot {k} of subpath {s}")),
    }
}

fn handle_mut(k: &mut Knot, h: Handle) -> &mut Point {
    match h {
        Handle::In => &mut k.in_ctrl,
        Handle::Out => &mut k.out_ctrl,
    }
}

fn handle(k: &Knot, h: Handle) -> Point {
    match h {
        Handle::In => k.in_ctrl,
        Handle::Out => k.out_ctrl,
    }
}

fn other(h: Handle) -> Handle {
    match h {
        Handle::In => Handle::Out,
        Handle::Out => Handle::In,
    }
}

/// A smooth knot keeps its handles collinear: point handle `h`'s opposite away from it, keeping
/// the opposite's length (a retracted handle stays retracted, as in Photoshop).
fn keep_smooth(k: &mut Knot, h: Handle) {
    if !k.smooth {
        return;
    }
    let (a, moved) = (k.anchor, handle(k, h));
    let opp = handle_mut(k, other(h));
    let (len, d) = (dist(*opp, a), dist(moved, a));
    if len > 1e-9 && d > 1e-9 {
        *opp = Point::new(a.x - (moved.x - a.x) / d * len, a.y - (moved.y - a.y) / d * len);
    }
}

/// Moves anchors (with their handles) by `d`. Each knot moves once, however often it is listed.
pub fn move_anchors(path: &mut Path, anchors: &[[usize; 2]], d: [f64; 2]) -> Result<(), String> {
    let mut list = anchors.to_vec();
    list.sort_unstable();
    list.dedup();
    for r in &list {
        knot_mut(path, *r)?;
    }
    for r in list {
        let k = knot_mut(path, r)?;
        *k = Knot { anchor: add(k.anchor, d), in_ctrl: add(k.in_ctrl, d), out_ctrl: add(k.out_ctrl, d), smooth: k.smooth };
    }
    Ok(())
}

/// Puts handle `h` of a knot at `to`. A smooth knot turns its other handle to stay collinear,
/// unless `independent` (Convert Point / ⌥), which makes it a corner whose handles move apart.
pub fn move_handle(path: &mut Path, knot: [usize; 2], h: Handle, to: Point, independent: bool) -> Result<(), String> {
    let k = knot_mut(path, knot)?;
    *handle_mut(k, h) = to;
    if independent {
        k.smooth = false;
    }
    keep_smooth(k, h);
    Ok(())
}

/// Convert Point: without `out`, a corner with retracted handles (a click on an anchor); with
/// `out`, a smooth knot whose handles are `out` and its mirror (a drag from an anchor).
pub fn convert_point(path: &mut Path, knot: [usize; 2], out: Option<Point>) -> Result<(), String> {
    let k = knot_mut(path, knot)?;
    let a = k.anchor;
    *k = match out {
        Some(o) => Knot::smooth(a, Point::new(2.0 * a.x - o.x, 2.0 * a.y - o.y), o),
        None => Knot::corner(a.x, a.y),
    };
    Ok(())
}

/// Drags the segment leaving `knot` at curve parameter `t` by `d`. A straight segment (both
/// handles retracted) moves with its two anchors; a curve reshapes so the point at `t` follows
/// the pointer exactly, with the nearer handle moving more. Smooth end knots stay smooth.
pub fn bend_segment(path: &mut Path, knot: [usize; 2], t: f64, d: [f64; 2]) -> Result<(), String> {
    let end = segment_end(path, knot)?;
    let retracted = |k: &Knot, h: Handle| dist(handle(k, h), k.anchor) < 1e-9;
    let a = *knot_mut(path, knot)?;
    let b = *knot_mut(path, end)?;
    if retracted(&a, Handle::Out) && retracted(&b, Handle::In) {
        return move_anchors(path, &[knot, end], d);
    }
    // B(t) moves by 3(1−t)²t·Δc1 + 3(1−t)t²·Δc2. With Δc1 = (1−t)·s·d and Δc2 = t·s·d that is
    // 3t(1−t)((1−t)² + t²)·s·d, so s below makes it exactly d. The ends are clamped: a drag right
    // at an anchor would need an unbounded handle move.
    let t = if t.is_finite() { t.clamp(0.05, 0.95) } else { 0.5 };
    let u = 1.0 - t;
    let s = 1.0 / (3.0 * t * u * (u * u + t * t));
    let (w1, w2) = (u * s, t * s);
    let ka = knot_mut(path, knot)?;
    ka.out_ctrl = add(ka.out_ctrl, [d[0] * w1, d[1] * w1]);
    keep_smooth(ka, Handle::Out);
    let kb = knot_mut(path, end)?;
    kb.in_ctrl = add(kb.in_ctrl, [d[0] * w2, d[1] * w2]);
    keep_smooth(kb, Handle::In);
    Ok(())
}

fn lerp(a: Point, b: Point, t: f64) -> Point {
    Point::new(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
}

/// Add Anchor Point: inserts an anchor into the segment leaving `knot` at curve parameter `t`
/// (strictly between 0 and 1) and returns the new knot's `[subpath, knot]`. A curve is split at
/// `t` (de Casteljau), so its shape does not change and the new anchor is smooth; on a straight
/// segment (both handles retracted) the new anchor is a corner on the line.
pub fn add_anchor(path: &mut Path, knot: [usize; 2], t: f64) -> Result<[usize; 2], String> {
    if !(t.is_finite() && t > 0.0 && t < 1.0) {
        return Err("`t` must be strictly between 0 and 1".into());
    }
    let end = segment_end(path, knot)?;
    let retracted = |k: &Knot, h: Handle| dist(handle(k, h), k.anchor) < 1e-9;
    let a = *knot_mut(path, knot)?;
    let b = *knot_mut(path, end)?;
    let new = if retracted(&a, Handle::Out) && retracted(&b, Handle::In) {
        // Hit testing and hover feedback use the cubic parameter, which is not linear even
        // when both handles are retracted. Keep the inserted corner at that same point.
        let p = eval(&[a.anchor, a.out_ctrl, b.in_ctrl, b.anchor], t);
        Knot::corner(p.x, p.y)
    } else {
        let (p01, p12, p23) = (lerp(a.anchor, a.out_ctrl, t), lerp(a.out_ctrl, b.in_ctrl, t), lerp(b.in_ctrl, b.anchor, t));
        let (p012, p123) = (lerp(p01, p12, t), lerp(p12, p23, t));
        knot_mut(path, knot)?.out_ctrl = p01;
        knot_mut(path, end)?.in_ctrl = p23;
        Knot::smooth(lerp(p012, p123, t), p012, p123)
    };
    let [s, k] = knot;
    let at = k.checked_add(1).ok_or_else(|| format!("no segment leaves knot {k} of subpath {s}"))?;
    let sp = path.subpaths.get_mut(s).ok_or_else(|| format!("no subpath {s}"))?;
    // `at` ≤ len: the segment leaving `k` exists, so `k` < len.
    sp.knots.insert(at.min(sp.knots.len()), new);
    Ok([s, at])
}

/// Delete Anchor Point: removes a knot. Its neighbours are joined by one segment that keeps the
/// previous anchor's out handle and the next anchor's in handle; a closed subpath stays closed.
/// Removing a subpath's last anchor removes the subpath. Refused (unchanged) when it would leave
/// the path with no anchors at all. Returns whether the subpath was removed.
pub fn delete_anchor(path: &mut Path, [s, k]: [usize; 2]) -> Result<bool, String> {
    knot_mut(path, [s, k])?;
    let total: usize = path.subpaths.iter().map(|sp| sp.knots.len()).fold(0, usize::saturating_add);
    if total <= 1 {
        return Err("can't delete the path's only anchor point; delete the path instead".into());
    }
    let sp = path.subpaths.get_mut(s).ok_or_else(|| format!("no subpath {s}"))?;
    if sp.knots.len() <= 1 {
        path.subpaths.remove(s);
        return Ok(true);
    }
    sp.knots.remove(k);
    Ok(false)
}

/// Every anchor of subpath `s` (⌥-click with Direct Selection selects the whole subpath).
pub fn subpath_anchors(path: &Path, s: usize) -> Vec<[usize; 2]> {
    path.subpaths.get(s).map_or_else(Vec::new, |sp| (0..sp.knots.len()).map(|k| [s, k]).collect())
}

/// Anchors inside the box `[x0, y0, x1, y1]` (a Direct Selection marquee), in path order.
pub fn anchors_in(path: &Path, [x0, y0, x1, y1]: [f64; 4]) -> Vec<[usize; 2]> {
    let (x0, x1, y0, y1) = (x0.min(x1), x0.max(x1), y0.min(y1), y0.max(y1));
    let mut out = Vec::new();
    for (s, sp) in path.subpaths.iter().enumerate() {
        for (k, kn) in sp.knots.iter().enumerate() {
            if (x0..=x1).contains(&kn.anchor.x) && (y0..=y1).contains(&kn.anchor.y) {
                out.push([s, k]);
            }
        }
    }
    out
}

/// The Curvature Pen's knots, `[anchor, in, out]`, through `anchors`: a smooth curve that passes
/// through every anchor, Catmull-Rom style. An inner anchor's handles lie on the line parallel
/// to its neighbours' chord (so they are collinear), each a third of the distance to the
/// neighbour on its side long (the chordal scaling that keeps uneven spacing from overshooting).
/// The ends of an open path aim at their neighbour's nearer handle. `corners[i]` retracts anchor
/// `i`'s handles (a corner point); a missing entry is smooth. Coincident points give retracted
/// handles, never NaN.
pub fn curvature_knots(anchors: &[[f64; 2]], corners: &[bool], closed: bool) -> Vec<[[f64; 2]; 3]> {
    let n = anchors.len();
    let mut knots: Vec<[[f64; 2]; 3]> = anchors.iter().map(|&a| [a; 3]).collect();
    let dist = |a: [f64; 2], b: [f64; 2]| (b[0] - a[0]).hypot(b[1] - a[1]);
    let unit = |a: [f64; 2], b: [f64; 2]| {
        let l = dist(a, b);
        (l > 1e-9).then(|| [(b[0] - a[0]) / l, (b[1] - a[1]) / l])
    };
    let corner = |i: usize| corners.get(i).copied().unwrap_or(false);
    let inner = |i: usize| closed || (i > 0 && i + 1 < n);
    for (i, &p) in anchors.iter().enumerate() {
        if n < 3 || corner(i) || !inner(i) {
            continue;
        }
        let prev = anchors[(i + n - 1) % n];
        let next = anchors[(i + 1) % n];
        let Some(d) = unit(prev, next).or_else(|| unit(p, next)) else { continue };
        let (li, lo) = (dist(prev, p) / 3.0, dist(p, next) / 3.0);
        knots[i] = [p, [p[0] - d[0] * li, p[1] - d[1] * li], [p[0] + d[0] * lo, p[1] + d[1] * lo]];
    }
    if !closed && n >= 3 {
        for (end, nb, take_in) in [(0, 1, true), (n - 1, n - 2, false)] {
            if corner(end) {
                continue;
            }
            let p = anchors[end];
            let aim = if take_in { knots[nb][1] } else { knots[nb][2] };
            let Some(d) = unit(p, aim) else { continue };
            let l = dist(p, anchors[nb]) / 3.0;
            let h = [p[0] + d[0] * l, p[1] + d[1] * l];
            if take_in {
                knots[end][2] = h;
            } else {
                knots[end][1] = h;
            }
        }
    }
    knots
}

/// The point at parameter `t` of the cubic segment `[anchor, out, in, anchor]`.
pub fn eval(seg: &[Point; 4], t: f64) -> Point {
    let u = 1.0 - t;
    let (a, b, c, d) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
    Point::new(a * seg[0].x + b * seg[1].x + c * seg[2].x + d * seg[3].x, a * seg[0].y + b * seg[1].y + c * seg[2].y + d * seg[3].y)
}

/// The part of `path` within `tol` of `p`: one of the `handles` shown (they sit on top), else an
/// anchor, else a segment; the nearest wins within each kind.
pub fn hit(path: &Path, p: Point, tol: f64, handles: &[([usize; 2], Handle)]) -> Option<Hit> {
    let nearest = |it: &mut dyn Iterator<Item = (f64, Hit)>| it.filter(|(d, _)| *d <= tol).min_by(|a, b| a.0.total_cmp(&b.0)).map(|(_, h)| h);
    let knot = |r: [usize; 2]| path.subpaths.get(r[0]).and_then(|sp| sp.knots.get(r[1]));
    let on_handle = nearest(&mut handles.iter().filter_map(|&(r, h)| {
        let k = knot(r)?;
        Some((dist(handle(k, h), p), Hit::Handle(r, h)))
    }));
    if on_handle.is_some() {
        return on_handle;
    }
    let anchors =
        path.subpaths.iter().enumerate().flat_map(|(s, sp)| sp.knots.iter().enumerate().map(move |(k, kn)| (dist(kn.anchor, p), Hit::Anchor([s, k]))));
    if let Some(h) = nearest(&mut anchors.into_iter()) {
        return Some(h);
    }
    // Segments: the nearest of 64 samples, refined once around it (plenty at screen tolerances).
    const N: usize = 64;
    let mut segs = path.subpaths.iter().enumerate().flat_map(|(s, sp)| {
        sp.segments().into_iter().enumerate().map(move |(k, seg)| {
            let best = (0..=N).map(|i| i as f64 / N as f64).map(|t| (dist(eval(&seg, t), p), t)).min_by(|a, b| a.0.total_cmp(&b.0)).unwrap_or((f64::MAX, 0.5));
            let step = 1.0 / N as f64;
            let fine = (0..=16)
                .map(|i| (best.1 - step + 2.0 * step * i as f64 / 16.0).clamp(0.0, 1.0))
                .map(|t| (dist(eval(&seg, t), p), t))
                .fold(best, |a, b| if b.0 < a.0 { b } else { a });
            (fine.0, Hit::Segment([s, k], fine.1))
        })
    });
    nearest(&mut segs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_doc::Subpath;

    fn tri() -> Path {
        Path::new(vec![Subpath::polygon(&[(0.0, 0.0), (100.0, 0.0), (50.0, 80.0)])])
    }

    /// A smooth knot at (50, 0) between corners, handles 20 px either side.
    fn curve() -> Path {
        let mut sp = Subpath::polyline(&[(0.0, 50.0), (50.0, 0.0), (100.0, 50.0)]);
        sp.knots[1] = Knot::smooth(Point::new(50.0, 0.0), Point::new(30.0, 0.0), Point::new(70.0, 0.0));
        Path::new(vec![sp])
    }

    fn at(path: &Path, s: usize, k: usize) -> Knot {
        path.subpaths[s].knots[k]
    }

    #[test]
    fn anchors_move_with_their_handles_once_each() {
        let mut p = curve();
        move_anchors(&mut p, &[[0, 1], [0, 1]], [5.0, -2.0]).unwrap();
        let k = at(&p, 0, 1);
        assert_eq!((k.anchor, k.in_ctrl, k.out_ctrl), (Point::new(55.0, -2.0), Point::new(35.0, -2.0), Point::new(75.0, -2.0)));
        assert_eq!(at(&p, 0, 0).anchor, Point::new(0.0, 50.0), "other anchors stay");
    }

    #[test]
    fn bad_indices_are_errors_and_change_nothing() {
        let mut p = tri();
        assert!(move_anchors(&mut p, &[[0, 0], [0, 9]], [1.0, 1.0]).is_err());
        assert_eq!(p, tri(), "validated before any knot moved");
        assert!(move_handle(&mut p, [3, 0], Handle::In, Point::new(0.0, 0.0), false).is_err());
        assert!(bend_segment(&mut p, [0, 7], 0.5, [1.0, 1.0]).is_err());
        assert!(convert_point(&mut p, [usize::MAX, usize::MAX], None).is_err());
        let mut open = curve();
        assert!(bend_segment(&mut open, [0, 2], 0.5, [1.0, 1.0]).is_err(), "an open subpath's last knot starts no segment");
        assert!(bend_segment(&mut open, [0, usize::MAX], 0.5, [1.0, 1.0]).is_err());
    }

    #[test]
    fn smooth_handles_stay_collinear_and_keep_their_length() {
        let mut p = curve();
        move_handle(&mut p, [0, 1], Handle::Out, Point::new(50.0, 30.0), false).unwrap();
        let k = at(&p, 0, 1);
        assert_eq!(k.out_ctrl, Point::new(50.0, 30.0));
        assert!(dist(k.in_ctrl, Point::new(50.0, -20.0)) < 1e-9, "opposite turned, length 20 kept: {:?}", k.in_ctrl);
        assert!(k.smooth);
    }

    #[test]
    fn independent_handles_break_the_knot() {
        let mut p = curve();
        move_handle(&mut p, [0, 1], Handle::In, Point::new(40.0, 20.0), true).unwrap();
        let k = at(&p, 0, 1);
        assert_eq!((k.in_ctrl, k.out_ctrl, k.smooth), (Point::new(40.0, 20.0), Point::new(70.0, 0.0), false));
    }

    #[test]
    fn convert_point_retracts_or_pulls_mirrored_handles() {
        let mut p = curve();
        convert_point(&mut p, [0, 1], None).unwrap();
        assert_eq!(at(&p, 0, 1), Knot::corner(50.0, 0.0));
        convert_point(&mut p, [0, 0], Some(Point::new(10.0, 60.0))).unwrap();
        let k = at(&p, 0, 0);
        assert_eq!((k.in_ctrl, k.out_ctrl, k.smooth), (Point::new(-10.0, 40.0), Point::new(10.0, 60.0), true));
    }

    #[test]
    fn straight_segments_move_with_both_anchors() {
        let mut p = tri();
        // The closing segment (2 → 0) wraps.
        bend_segment(&mut p, [0, 2], 0.5, [0.0, 10.0]).unwrap();
        assert_eq!((at(&p, 0, 2).anchor, at(&p, 0, 0).anchor, at(&p, 0, 1).anchor), (Point::new(50.0, 90.0), Point::new(0.0, 10.0), Point::new(100.0, 0.0)));
    }

    #[test]
    fn curves_bend_through_the_dragged_point() {
        for t in [0.2, 0.5, 0.8] {
            let mut p = curve();
            let seg = |p: &Path| p.subpaths[0].segments()[1];
            let before = eval(&seg(&p), t);
            bend_segment(&mut p, [0, 1], t, [3.0, 12.0]).unwrap();
            let after = eval(&seg(&p), t);
            assert!(dist(after, Point::new(before.x + 3.0, before.y + 12.0)) < 1e-9, "t={t}: {after:?}");
            let k = at(&p, 0, 1);
            let (vi, vo) = ((k.in_ctrl.x - 50.0, k.in_ctrl.y), (k.out_ctrl.x - 50.0, k.out_ctrl.y));
            assert!((vi.0 * vo.1 - vi.1 * vo.0).abs() < 1e-9 && vi.0 * vo.0 + vi.1 * vo.1 < 0.0, "the smooth knot stayed smooth");
        }
    }

    #[test]
    fn hits_prefer_handles_then_anchors_then_segments() {
        let p = curve();
        let shown = [([0, 1], Handle::Out)];
        assert_eq!(hit(&p, Point::new(71.0, 1.0), 3.0, &shown), Some(Hit::Handle([0, 1], Handle::Out)));
        assert_eq!(hit(&p, Point::new(71.0, 1.0), 3.0, &[]), None, "hidden handles aren't hit");
        assert_eq!(hit(&p, Point::new(49.0, 1.0), 3.0, &shown), Some(Hit::Anchor([0, 1])));
        let Some(Hit::Segment(r, t)) = hit(&tri(), Point::new(50.0, 1.0), 3.0, &[]) else { panic!("segment") };
        assert_eq!(r, [0, 0]);
        assert!((t - 0.5).abs() < 0.02, "{t}");
        assert_eq!(hit(&tri(), Point::new(50.0, 40.0), 3.0, &[]), None);
    }

    #[test]
    fn adding_to_a_curve_splits_it_without_changing_its_shape() {
        for t in [0.5, 0.13, 0.9] {
            let mut p = curve();
            let seg = curve().subpaths[0].segments()[0];
            let r = add_anchor(&mut p, [0, 0], t).unwrap();
            assert_eq!((r, p.subpaths[0].knots.len()), ([0, 1], 4));
            let k = at(&p, 0, 1);
            assert!(k.smooth, "a split curve gets a smooth anchor");
            assert!(dist(k.anchor, eval(&seg, t)) < 1e-9, "the new anchor is where it was clicked");
            // 100 evenly spaced parameters of the old segment land on the two halves.
            let halves = p.subpaths[0].segments();
            for i in 0..=100 {
                let u = i as f64 / 100.0;
                let q = if u <= t { eval(&halves[0], u / t) } else { eval(&halves[1], (u - t) / (1.0 - t)) };
                assert!(dist(q, eval(&seg, u)) < 0.01, "t={t} u={u}");
            }
            assert_eq!(halves[2], curve().subpaths[0].segments()[1], "the next segment is untouched");
        }
    }

    #[test]
    fn adding_to_a_straight_segment_uses_the_hit_position() {
        for (points, closed) in [
            (vec![(0.0, 0.0), (100.0, 0.0)], false),
            (vec![(0.0, 0.0), (0.0, 100.0)], false),
            (vec![(0.0, 0.0), (100.0, 100.0)], false),
            (vec![(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)], true),
        ] {
            for t in [0.25, 0.75] {
                let sp = if closed { Subpath::polygon(&points) } else { Subpath::polyline(&points) };
                let mut p = Path::new(vec![sp]);
                let segment = if closed { 2 } else { 0 };
                let clicked = eval(&p.subpaths[0].segments()[segment], t);
                let Some(Hit::Segment(knot, hit_t)) = hit(&p, clicked, 1.0, &[]) else { panic!("expected a segment hit at {clicked:?}") };
                let added = add_anchor(&mut p, knot, hit_t).unwrap();
                let inserted = at(&p, added[0], added[1]);
                assert!(dist(inserted.anchor, clicked) < 1e-9, "t={t}: inserted {:?}, clicked {clicked:?}", inserted.anchor);
                assert_eq!(inserted, Knot::corner(clicked.x, clicked.y), "straight segments retain retracted handles");
            }
        }
    }

    #[test]
    fn adding_to_a_straight_segment_makes_a_corner_on_the_line() {
        let mut p = tri();
        // The closing segment (2 → 0) wraps: the new anchor goes last.
        assert_eq!(add_anchor(&mut p, [0, 2], 0.5).unwrap(), [0, 3]);
        assert_eq!(at(&p, 0, 3), Knot::corner(25.0, 40.0));
        assert_eq!(add_anchor(&mut p, [0, 0], 0.25).unwrap(), [0, 1]);
        assert_eq!(at(&p, 0, 1), Knot::corner(15.625, 0.0));
        assert_eq!(p.subpaths[0].knots.len(), 5);
        assert!(p.subpaths[0].closed);
    }

    #[test]
    fn deleting_joins_the_neighbours_with_their_handles() {
        let mut sq = Path::new(vec![Subpath::polygon(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])]);
        assert!(!delete_anchor(&mut sq, [0, 2]).unwrap());
        assert_eq!((sq.subpaths[0].knots.len(), sq.subpaths[0].closed), (3, true));
        let mut c = Path::new(vec![Subpath::polyline(&[(0.0, 50.0), (50.0, 0.0), (100.0, 50.0)])]);
        c.subpaths[0].knots[0].out_ctrl = Point::new(10.0, 20.0);
        c.subpaths[0].knots[2].in_ctrl = Point::new(90.0, 20.0);
        delete_anchor(&mut c, [0, 1]).unwrap();
        assert_eq!(c.subpaths[0].segments(), vec![[Point::new(0.0, 50.0), Point::new(10.0, 20.0), Point::new(90.0, 20.0), Point::new(100.0, 50.0)]]);
    }

    #[test]
    fn deleting_a_subpaths_last_anchor_removes_it_but_never_the_last_anchor() {
        let mut p = tri();
        p.subpaths.push(Subpath::polyline(&[(5.0, 5.0)]));
        assert!(delete_anchor(&mut p, [1, 0]).unwrap());
        assert_eq!(p, tri());
        let mut one = Path::new(vec![Subpath::polyline(&[(5.0, 5.0)])]);
        let before = one.clone();
        assert!(delete_anchor(&mut one, [0, 0]).is_err());
        assert_eq!(one, before);
    }

    #[test]
    fn bad_anchor_edits_are_errors_and_change_nothing() {
        let mut p = curve();
        for t in [0.0, 1.0, -0.5, 2.0, f64::NAN, f64::INFINITY] {
            assert!(add_anchor(&mut p, [0, 0], t).is_err(), "t={t}");
        }
        assert!(add_anchor(&mut p, [0, 2], 0.5).is_err(), "an open subpath's last knot starts no segment");
        assert!(add_anchor(&mut p, [0, usize::MAX], 0.5).is_err());
        assert!(add_anchor(&mut p, [usize::MAX, 0], 0.5).is_err());
        assert!(delete_anchor(&mut p, [0, 3]).is_err());
        assert!(delete_anchor(&mut p, [1, 0]).is_err());
        assert_eq!(p, curve());
    }

    #[test]
    fn marquee_and_subpath_selections() {
        let p = tri();
        assert_eq!(anchors_in(&p, [110.0, -5.0, 40.0, 5.0]), vec![[0, 1]]);
        assert_eq!(subpath_anchors(&p, 0), vec![[0, 0], [0, 1], [0, 2]]);
        assert!(subpath_anchors(&p, 4).is_empty());
    }
}
