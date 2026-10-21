//! Parametric shapes: Affinity stores the shape's parameters and its box, never the outline.
//! Formulas were fitted to Affinity's own renders (the document thumbnails) of public files.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use crate::model::{Affine, Path, Point, Reader, SubPath};
use crate::stream::{ObjId, Tag, Value};

/// Handle length of a quarter circle of radius 1.
const KAPPA: f64 = 0.552_284_749_830_793_4;
/// Vertices of a polygon or star.
const MAX_SIDES: i64 = 10_000;

/// The last value of a tag: a shape changed from one kind to another keeps a stale first block.
fn last<'a>(r: &'a Reader, id: ObjId, tag: &[u8; 4]) -> Option<&'a Value> {
    let t = Tag::of(tag);
    r.s.object(id)?.fields.iter().rev().find(|(k, _)| *k == t).map(|(_, v)| v)
}

fn num(r: &Reader, id: ObjId, tag: &[u8; 4]) -> Option<f64> {
    match last(r, id, tag)? {
        Value::Float(f) if f.is_finite() => Some(*f),
        Value::UInt(u) => Some(*u as f64),
        Value::Int(i) => Some(*i as f64),
        _ => None,
    }
}

fn sides(r: &Reader, id: ObjId, tag: &[u8; 4], default: f64) -> usize {
    let n = num(r, id, tag).unwrap_or(default).round();
    n.clamp(3.0, MAX_SIDES as f64) as usize
}

fn pt(x: f64, y: f64) -> Point {
    Point { x, y }
}

fn polygon(pts: impl IntoIterator<Item = Point>) -> Option<SubPath> {
    let mut it = pts.into_iter();
    let start = it.next()?;
    let segments = it.map(|p| [p, p, p]).collect();
    Some(SubPath { start, segments, closed: true })
}

/// Outline of `shape` (a `Shpe` object) in the node box `b` = (x0, y0, x1, y1).
pub(crate) fn shape(r: &mut Reader, shape: ObjId, b: [f64; 4], world: Affine) -> Option<(Path, bool)> {
    let [x0, y0, x1, y1] = b;
    let (w, h) = (x1 - x0, y1 - y0);
    let (cx, cy, rx, ry) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0, w / 2.0, h / 2.0);
    let on_ellipse = |a: f64, f: f64| pt(cx + f * rx * a.cos(), cy + f * ry * a.sin());
    let class = r.s.class(shape)?;
    let sub = match &class.0.to_be_bytes() {
        b"ShNR" | b"ShRR" => {
            let types = match last(r, shape, b"CTyp") {
                Some(Value::Array(a)) => a.iter().map(|t| if let Value::Enum { id, .. } = t { Some(*id) } else { None }).collect(),
                _ => Vec::new(),
            };
            let radii = match last(r, shape, b"ShCR") {
                Some(Value::Floats(v)) => v.iter().map(|f| if f.is_finite() && *f > 0.0 { *f } else { 0.0 }).collect(),
                _ => Vec::new(),
            };
            // Corners follow the first one (top left) unless they were unlocked; then each has
            // its own radius and type, clockwise from the top left.
            let locked = !matches!(last(r, shape, b"Lock"), Some(Value::Bool(false)));
            let corners: [(f64, Option<u16>); 4] = std::array::from_fn(|i| {
                let i = if locked { 0 } else { i };
                (radii.get(i).copied().unwrap_or(0.0), types.get(i).copied().flatten())
            });
            if types.is_empty() || corners.iter().all(|(f, _)| *f == 0.0) {
                polygon([pt(x0, y0), pt(x1, y0), pt(x1, y1), pt(x0, y1)])?
            } else {
                if corners.iter().any(|(f, t)| *f > 0.0 && !matches!(t, Some(0 | 1 | 4))) {
                    r.warn("rectangles with concave or other corner types (imported with round corners)");
                }
                let absolute = r.s.bool(shape, b"AbSz") == Some(true);
                let [a, b2, c, d, ..] = world.0;
                let (sx, sy) = ((a * a + b2 * b2).sqrt(), (c * c + d * d).sqrt());
                cornered_rect(
                    b,
                    corners.map(|(f, t)| {
                        let (crx, cry) = if !absolute {
                            (f * w.min(h), f * w.min(h))
                        } else if sx > 0.0 && sy > 0.0 {
                            // An absolute radius is in document pixels, whatever the node's scale.
                            (f / sx, f / sy)
                        } else {
                            (0.0, 0.0)
                        };
                        (crx, cry, t == Some(1))
                    }),
                )
            }
        }
        b"ShpE" => ellipse(cx, cy, rx, ry),
        b"ShPy" => {
            let n = sides(r, shape, b"Side", 5.0);
            let a = |i: usize| -FRAC_PI_2 + TAU * i as f64 / n as f64;
            if r.s.object(shape).is_some_and(|_| matches!(last(r, shape, b"Smth"), Some(Value::Bool(true)))) {
                // Smoothed: tangent handles of length curv·4/3·tan(π/2n) along the inscribed ellipse.
                let k = num(r, shape, b"Curv").unwrap_or(0.0) * 4.0 / 3.0 * (PI / (2.0 * n as f64)).tan();
                let v = |i: usize| on_ellipse(a(i), 1.0);
                let t = |i: usize| pt(-rx * a(i).sin(), ry * a(i).cos());
                let segments = (0..n)
                    .map(|i| {
                        let (p, q, tp, tq) = (v(i), v(i + 1), t(i), t(i + 1));
                        [pt(p.x + k * tp.x, p.y + k * tp.y), pt(q.x - k * tq.x, q.y - k * tq.y), q]
                    })
                    .collect();
                SubPath { start: v(0), segments, closed: true }
            } else {
                polygon((0..n).map(|i| on_ellipse(a(i), 1.0)))?
            }
        }
        b"ShSt" => {
            let n = sides(r, shape, b"Pnts", 5.0);
            let q = num(r, shape, b"IRad").unwrap_or(0.382).clamp(0.0, 1.0);
            if [b"CrcI", b"CrcO", b"CrvL", b"CrvR"].iter().any(|t| num(r, shape, t).is_some_and(|v| v.abs() > 1e-9)) {
                r.warn("stars with rounded or curved points (imported with sharp points)");
            }
            polygon((0..2 * n).map(|i| on_ellipse(-FRAC_PI_2 + PI * i as f64 / n as f64, if i % 2 == 0 { 1.0 } else { q })))?
        }
        b"ShSS" => {
            let n = sides(r, shape, b"Side", 5.0);
            let ri = num(r, shape, b"COut").unwrap_or(0.5).clamp(0.0, 1.0);
            let step = PI / n as f64;
            let (l, hw, inner) = (step.cos(), ri * step.sin(), ri * step.cos());
            polygon((0..n).flat_map(|k| {
                let phi = -FRAC_PI_2 + step + TAU * k as f64 / n as f64;
                let (c, s) = (phi.cos(), phi.sin());
                [(inner, -hw), (l, -hw), (l, hw), (inner, hw)].map(|(u, v)| pt(cx + rx * (u * c - v * s), cy + ry * (u * s + v * c)))
            }))?
        }
        b"ShPi" => {
            let raw_start = num(r, shape, b"AngS").unwrap_or(0.0);
            let raw_end = num(r, shape, b"AngE").unwrap_or(TAU);
            // Normalize in constant time. Repeatedly adding TAU can hang on finite values
            // large enough that floating-point addition no longer changes the endpoint.
            let start = raw_start.rem_euclid(TAU);
            let delta = raw_end - raw_start;
            let sweep = if raw_end >= raw_start {
                delta.min(TAU)
            } else if delta.is_finite() {
                delta.rem_euclid(TAU)
            } else {
                (raw_end.rem_euclid(TAU) - start).rem_euclid(TAU)
            };
            let inner = num(r, shape, b"IRad").unwrap_or(0.0).clamp(0.0, 1.0);
            pie(cx, cy, rx, ry, start, start + sweep, inner)
        }
        b"ShpT" => {
            let p = num(r, shape, b"Pos ").unwrap_or(0.5);
            polygon([pt(x0 + p * w, y0), pt(x1, y1), pt(x0, y1)])?
        }
        b"ShTz" => {
            let (l, rr) = (num(r, shape, b"PosL").unwrap_or(0.25), num(r, shape, b"PosR").unwrap_or(0.75));
            polygon([pt(x0 + l * w, y0), pt(x0 + rr * w, y0), pt(x1, y1), pt(x0, y1)])?
        }
        b"ShCl" => cloud(cx, cy, rx, ry, sides(r, shape, b"Bubl", 12.0)),
        b"ShHt" => heart(b),
        b"ShCg" => cog(r, shape, cx, cy, rx, ry),
        b"ShCR" => callout_rect(r, shape, b, false),
        b"ShCE" => callout_ellipse(r, shape, cx, cy, rx, ry),
        b"ShDA" => arrow(r, shape, b),
        b"ShDS" => double_star(r, shape, cx, cy, rx, ry),
        b"ShTr" => tear(r, shape, b),
        b"ShCr" => crescent(cx, cy, rx, ry),
        b"ShpD" => polygon([pt(cx, y0), pt(x1, cy), pt(cx, y1), pt(x0, cy)])?,
        b"ShSg" => segment(r, shape, cx, cy, rx, ry),
        _ => {
            r.warn("an unrecognized parametric shape (imported as its bounding ellipse)");
            ellipse(cx, cy, rx, ry)
        }
    };
    let mut path = Path { subpaths: vec![sub] };
    if class == Tag::of(b"ShCg") {
        let hole = num(r, shape, b"Hole").unwrap_or(0.3).clamp(0.0, 0.9);
        if hole > 0.0 {
            path.subpaths.push(ellipse(cx, cy, rx * hole, ry * hole));
        }
        return Some((path, true));
    }
    Some((path, false))
}

fn cloud(cx: f64, cy: f64, rx: f64, ry: f64, bubbles: usize) -> SubPath {
    let n = bubbles.clamp(3, 128) * 4;
    let points: Vec<Point> = (0..n)
        .map(|i| {
            let a = TAU * i as f64 / n as f64 - FRAC_PI_2;
            let lobe = (0.5 - 0.5 * (TAU * i as f64 / 4.0).cos()).powi(2);
            let radius = 0.78 + 0.22 * lobe;
            pt(cx + rx * radius * a.cos(), cy + ry * radius * a.sin())
        })
        .collect();
    let segments = (0..n)
        .map(|i| {
            let p0 = points[(i + n - 1) % n];
            let p1 = points[i];
            let p2 = points[(i + 1) % n];
            let p3 = points[(i + 2) % n];
            [pt(p1.x + (p2.x - p0.x) / 6.0, p1.y + (p2.y - p0.y) / 6.0), pt(p2.x - (p3.x - p1.x) / 6.0, p2.y - (p3.y - p1.y) / 6.0), p2]
        })
        .collect();
    SubPath { start: points[0], segments, closed: true }
}

fn heart([x0, y0, x1, y1]: [f64; 4]) -> SubPath {
    let (w, h) = (x1 - x0, y1 - y0);
    let p = |x: f64, y: f64| pt(x0 + x * w, y0 + y * h);
    let start = p(0.5, 0.92);
    let segments = vec![
        [p(0.38, 0.85), p(0.04, 0.67), p(0.04, 0.37)],
        [p(0.04, 0.02), p(0.39, 0.00), p(0.5, 0.27)],
        [p(0.61, 0.00), p(0.96, 0.02), p(0.96, 0.37)],
        [p(0.96, 0.67), p(0.62, 0.85), start],
    ];
    SubPath { start, segments, closed: true }
}

fn cog(r: &Reader, shape: ObjId, cx: f64, cy: f64, rx: f64, ry: f64) -> SubPath {
    let teeth = sides(r, shape, b"Teth", 12.0);
    let tooth = num(r, shape, b"TtSz").unwrap_or(0.3).clamp(0.05, 0.9);
    let notch = num(r, shape, b"NtSz").unwrap_or(0.45).clamp(0.05, 0.95);
    let tip = 1.0;
    let root = (1.0 - 0.2 * tooth - 0.08 * notch).clamp(0.55, 0.95);
    let mut pts = Vec::with_capacity(teeth * 4);
    for i in 0..teeth {
        let base = TAU * i as f64 / teeth as f64 - FRAC_PI_2;
        for (offset, radius) in [(-0.5, root), (-0.32, tip), (0.32, tip), (0.5, root)] {
            let a = base + offset * TAU / teeth as f64;
            pts.push(pt(cx + rx * radius * a.cos(), cy + ry * radius * a.sin()));
        }
    }
    polygon(pts).unwrap_or_else(|| rectangle_sub(cx - rx, cy - ry, cx + rx, cy + ry))
}

fn callout_rect(r: &Reader, shape: ObjId, [x0, y0, x1, y1]: [f64; 4], _ellipse: bool) -> SubPath {
    let (w, h) = (x1 - x0, y1 - y0);
    let tail_h = num(r, shape, b"TlHg").unwrap_or(0.2).clamp(0.0, 0.8) * h;
    let tail_w = num(r, shape, b"TlWd").unwrap_or(0.1).clamp(0.02, 0.8) * w;
    let tip_x = x0 + num(r, shape, b"TlRP").unwrap_or(0.5).clamp(0.0, 1.0) * w;
    let base_x = x0 + num(r, shape, b"TlEP").unwrap_or(0.4).clamp(0.0, 1.0) * w;
    let a = (base_x - tail_w / 2.0).clamp(x0, x1);
    let c = (base_x + tail_w / 2.0).clamp(x0, x1);
    let body_bottom = y1 - tail_h;
    let radius = (w.min(body_bottom - y0) * 0.12).max(0.0);
    // The three contours overlap inside the fill; no internal edge is visible in a filled shape.
    // Join the tail to the rounded body in one closed outline to avoid a seam in stroked shapes.
    let start = pt(x0 + radius, y0);
    let segments = vec![
        [pt(x1 - radius, y0), pt(x1 - radius, y0), pt(x1 - radius, y0)],
        [pt(x1 - (1.0 - KAPPA) * radius, y0), pt(x1, y0 + (1.0 - KAPPA) * radius), pt(x1, y0 + radius)],
        [pt(x1, body_bottom - radius), pt(x1, body_bottom - radius), pt(x1, body_bottom - radius)],
        [pt(x1, body_bottom), pt(x1, body_bottom), pt(x1 - radius, body_bottom)],
        [pt(c + radius * 0.2, body_bottom), pt(tip_x, y1 - tail_h * 0.15), pt(tip_x, y1)],
        [pt(tip_x, y1), pt(tip_x, y1), pt(tip_x, y1)],
        [pt(tip_x, y1 - tail_h * 0.15), pt(a - radius * 0.2, body_bottom), pt(a, body_bottom)],
        [pt(x0 + radius, body_bottom), pt(x0, body_bottom), pt(x0, body_bottom - radius)],
        [pt(x0, y0 + radius), pt(x0, y0 + radius), pt(x0, y0 + radius)],
        [pt(x0, y0 + (1.0 - KAPPA) * radius), pt(x0 + (1.0 - KAPPA) * radius, y0), start],
    ];
    SubPath { start, segments, closed: true }
}

fn callout_ellipse(r: &Reader, shape: ObjId, cx: f64, cy: f64, rx: f64, ry: f64) -> SubPath {
    let tail_h = num(r, shape, b"TlHg").unwrap_or(0.2).clamp(0.0, 0.8);
    let pos = num(r, shape, b"TlEP").unwrap_or(0.25).clamp(0.0, 1.0);
    let angle = num(r, shape, b"TlAn").unwrap_or(0.0);
    let a0 = pos * TAU - angle;
    let a1 = a0 + 0.35;
    let start = pt(cx + rx * a1.cos(), cy + ry * a1.sin());
    let end = pt(cx + rx * a0.cos(), cy + ry * a0.sin());
    let tip = pt(cx + rx * (a0 + 0.18).cos() * 1.08, cy + ry * (a0 + 0.18).sin() * (1.0 + tail_h));
    // A single ellipse-like outline with a short triangular tail at its lower edge.
    let mut segments = Vec::new();
    arc(cx, cy, rx, ry, a1, a0 + TAU, &mut segments);
    segments.push([end, end, tip]);
    segments.push([tip, tip, start]);
    SubPath { start, segments, closed: true }
}

fn arrow(r: &Reader, shape: ObjId, [x0, y0, x1, y1]: [f64; 4]) -> SubPath {
    let h = y1 - y0;
    let thick = num(r, shape, b"Thck").unwrap_or(0.35).clamp(0.05, 0.9) * h;
    let shaft = (y0 + y1 - thick) / 2.0;
    let lower = shaft + thick;
    let head_start = x1 - h * 0.45;
    polygon([pt(x0, shaft), pt(head_start, shaft), pt(head_start, y0), pt(x1, (y0 + y1) / 2.0), pt(head_start, y1), pt(head_start, lower), pt(x0, lower)])
        .unwrap_or_else(|| rectangle_sub(x0, shaft, x1, lower))
}

fn double_star(r: &Reader, shape: ObjId, cx: f64, cy: f64, rx: f64, ry: f64) -> SubPath {
    let n = sides(r, shape, b"Pnts", 6.0).saturating_mul(2).clamp(8, 64);
    let inner = num(r, shape, b"IRad").unwrap_or(0.6).clamp(0.05, 0.95);
    let outer = num(r, shape, b"PRad").unwrap_or(0.9).clamp(0.2, 1.0);
    polygon((0..2 * n).map(|i| {
        let a = -FRAC_PI_2 + PI * i as f64 / n as f64;
        let radius = if i % 2 == 0 { outer } else { outer * inner };
        pt(cx + rx * radius * a.cos(), cy + ry * radius * a.sin())
    }))
    .unwrap_or_else(|| rectangle_sub(cx - rx, cy - ry, cx + rx, cy + ry))
}

fn tear(r: &Reader, shape: ObjId, [x0, y0, x1, y1]: [f64; 4]) -> SubPath {
    let (w, h) = (x1 - x0, y1 - y0);
    let bulge = num(r, shape, b"Ball").unwrap_or(0.25).clamp(0.0, 0.5);
    let bend = num(r, shape, b"Bend").unwrap_or(0.0).clamp(-0.5, 0.5);
    let top = pt((x0 + x1) / 2.0, y0);
    let bottom = pt((x0 + x1) / 2.0 + bend * w * 0.25, y1);
    let segments = vec![
        [pt(x0 + w * 0.25, y0 + h * 0.42), pt(x0 + w * (0.5 - bulge), y1 - h * 0.03), bottom],
        [pt(x1 - w * (0.5 - bulge), y1 - h * 0.03), pt(x1 - w * 0.25, y0 + h * 0.42), top],
    ];
    SubPath { start: top, segments, closed: true }
}

fn crescent(cx: f64, cy: f64, rx: f64, ry: f64) -> SubPath {
    let start = pt(cx, cy - ry);
    let mut segments = Vec::new();
    arc(cx, cy, rx, ry, -FRAC_PI_2, -3.0 * FRAC_PI_2, &mut segments);
    // The inner arc returns from the lower tip to the upper tip, leaving a crescent open to the right.
    let ix = cx + rx * 0.72;
    let irx = rx * 0.86;
    let iry = ry * 0.82;
    let from = pt(ix, cy + iry);
    segments.push([from, from, from]);
    let mut inner = Vec::new();
    arc(ix, cy, irx, iry, FRAC_PI_2, -FRAC_PI_2, &mut inner);
    segments.extend(inner);
    SubPath { start, segments, closed: true }
}

fn segment(r: &Reader, shape: ObjId, cx: f64, cy: f64, rx: f64, ry: f64) -> SubPath {
    // Keep hostile but finite angles in a small range before constructing arc controls.
    let angle = num(r, shape, b"Angl").unwrap_or(0.0).rem_euclid(TAU);
    let p0 = num(r, shape, b"Pos0").unwrap_or(0.25).clamp(0.0, 1.0);
    let p1 = num(r, shape, b"Pos1").unwrap_or(0.75).clamp(0.0, 1.0);
    let a0 = angle + (p0 - 0.5) * TAU;
    let mut a1 = angle + (p1 - 0.5) * TAU;
    while a1 < a0 {
        a1 += TAU;
    }
    let mut segments = Vec::new();
    arc(cx, cy, rx, ry, a0, a1.min(a0 + TAU), &mut segments);
    let start = pt(cx + rx * a0.cos(), cy + ry * a0.sin());
    let end = pt(cx + rx * a1.cos(), cy + ry * a1.sin());
    segments.push([end, end, start]);
    SubPath { start, segments, closed: true }
}

fn rectangle_sub(x0: f64, y0: f64, x1: f64, y1: f64) -> SubPath {
    rectangle(x0, y0, x1, y1)
}

pub(crate) fn rectangle(x0: f64, y0: f64, x1: f64, y1: f64) -> SubPath {
    SubPath { start: pt(x0, y0), segments: [pt(x1, y0), pt(x1, y1), pt(x0, y1)].map(|p| [p, p, p]).to_vec(), closed: true }
}

fn ellipse(cx: f64, cy: f64, rx: f64, ry: f64) -> SubPath {
    let (kx, ky) = (KAPPA * rx, KAPPA * ry);
    SubPath {
        start: pt(cx + rx, cy),
        segments: vec![
            [pt(cx + rx, cy + ky), pt(cx + kx, cy + ry), pt(cx, cy + ry)],
            [pt(cx - kx, cy + ry), pt(cx - rx, cy + ky), pt(cx - rx, cy)],
            [pt(cx - rx, cy - ky), pt(cx - kx, cy - ry), pt(cx, cy - ry)],
            [pt(cx + kx, cy - ry), pt(cx + rx, cy - ky), pt(cx + rx, cy)],
        ],
        closed: true,
    }
}

/// A rectangle whose corners (top left, top right, bottom right, bottom left) are each rounded,
/// or cut straight across, by their (horizontal, vertical) radii. Two radii that don't fit along
/// one side shrink in proportion until they do.
fn cornered_rect([x0, y0, x1, y1]: [f64; 4], corners: [(f64, f64, bool); 4]) -> SubPath {
    let fit = |r: f64, pair: f64, side: f64| if pair > side { side * (r / pair) } else { r };
    let [tl, tr, br, bl] = corners;
    let (w, h) = (x1 - x0, y1 - y0);
    let rx = [fit(tl.0, tl.0 + tr.0, w), fit(tr.0, tl.0 + tr.0, w), fit(br.0, bl.0 + br.0, w), fit(bl.0, bl.0 + br.0, w)];
    let ry = [fit(tl.1, tl.1 + bl.1, h), fit(tr.1, tr.1 + br.1, h), fit(br.1, tr.1 + br.1, h), fit(bl.1, tl.1 + bl.1, h)];
    let (kx, ky) = (rx.map(|r| (1.0 - KAPPA) * r), ry.map(|r| (1.0 - KAPPA) * r));
    // (where the corner leaves the previous side, its two handles, where it joins the next side)
    let arcs = [
        [pt(x1 - rx[1], y0), pt(x1 - kx[1], y0), pt(x1, y0 + ky[1]), pt(x1, y0 + ry[1])],
        [pt(x1, y1 - ry[2]), pt(x1, y1 - ky[2]), pt(x1 - kx[2], y1), pt(x1 - rx[2], y1)],
        [pt(x0 + rx[3], y1), pt(x0 + kx[3], y1), pt(x0, y1 - ky[3]), pt(x0, y1 - ry[3])],
        [pt(x0, y0 + ry[0]), pt(x0, y0 + ky[0]), pt(x0 + kx[0], y0), pt(x0 + rx[0], y0)],
    ];
    let mut segments = Vec::with_capacity(8);
    for ([s, c1, c2, e], straight) in arcs.into_iter().zip([tr.2, br.2, bl.2, tl.2]) {
        segments.push([s, s, s]);
        if s != e {
            segments.push(if straight { [e, e, e] } else { [c1, c2, e] });
        }
    }
    SubPath { start: pt(x0 + rx[0], y0), segments, closed: true }
}

/// Elliptical arc from angle `a0` to `a1` (radians, measured y-up as Affinity's pie angles are),
/// as cubic segments of at most a quarter turn.
fn arc(cx: f64, cy: f64, rx: f64, ry: f64, a0: f64, a1: f64, out: &mut Vec<[Point; 3]>) {
    let n = ((a1 - a0).abs() / FRAC_PI_2).ceil().clamp(1.0, 8.0) as usize;
    let step = (a1 - a0) / n as f64;
    let k = 4.0 / 3.0 * (step / 4.0).tan();
    let p = |a: f64| pt(cx + rx * a.cos(), cy - ry * a.sin());
    let d = |a: f64| pt(-rx * a.sin(), -ry * a.cos());
    for i in 0..n {
        let (s, e) = (a0 + step * i as f64, a0 + step * (i + 1) as f64);
        let (ps, pe, ds, de) = (p(s), p(e), d(s), d(e));
        out.push([pt(ps.x + k * ds.x, ps.y + k * ds.y), pt(pe.x - k * de.x, pe.y - k * de.y), pe]);
    }
}

fn pie(cx: f64, cy: f64, rx: f64, ry: f64, a0: f64, a1: f64, inner: f64) -> SubPath {
    let start = pt(cx + rx * a0.cos(), cy - ry * a0.sin());
    let mut segments = Vec::new();
    arc(cx, cy, rx, ry, a0, a1, &mut segments);
    if inner > 0.0 {
        let p = pt(cx + inner * rx * a1.cos(), cy - inner * ry * a1.sin());
        segments.push([p, p, p]);
        arc(cx, cy, inner * rx, inner * ry, a1, a0, &mut segments);
    } else {
        let c = pt(cx, cy);
        segments.push([c, c, c]);
    }
    SubPath { start, segments, closed: true }
}

/// A polyline with circular corners of the given radii (Affinity's live corners on curves).
pub(crate) fn fillet(pts: &[(Point, f64)], closed: bool) -> SubPath {
    let n = pts.len();
    let get = |i: isize| -> Option<Point> {
        if closed { pts.get(i.rem_euclid(n as isize) as usize).map(|p| p.0) } else { usize::try_from(i).ok().and_then(|i| pts.get(i)).map(|p| p.0) }
    };
    let mut corners: Vec<(Point, Point, Point, Point)> = Vec::with_capacity(n);
    for (i, (p, radius)) in pts.iter().enumerate() {
        let (prev, next) = (get(i as isize - 1), get(i as isize + 1));
        let (Some(a), Some(b)) = (prev, next) else {
            corners.push((*p, *p, *p, *p));
            continue;
        };
        let (u1, l1) = unit(*p, a);
        let (u2, l2) = unit(*p, b);
        let cos = (u1.x * u2.x + u1.y * u2.y).clamp(-1.0, 1.0);
        let phi = cos.acos();
        if *radius <= 0.0 || phi < 1e-6 || (PI - phi) < 1e-6 {
            corners.push((*p, *p, *p, *p));
            continue;
        }
        let t = (radius / (phi / 2.0).tan()).min(l1 / 2.0).min(l2 / 2.0);
        let r = t * (phi / 2.0).tan();
        let k = 4.0 / 3.0 * ((PI - phi) / 4.0).tan() * r;
        let s = pt(p.x + u1.x * t, p.y + u1.y * t);
        let e = pt(p.x + u2.x * t, p.y + u2.y * t);
        corners.push((s, pt(s.x - u1.x * k, s.y - u1.y * k), pt(e.x - u2.x * k, e.y - u2.y * k), e));
    }
    let Some(first) = corners.first().copied() else { return SubPath { start: pt(0.0, 0.0), segments: vec![], closed } };
    let mut segments = Vec::with_capacity(2 * n);
    let mut cur = first.0;
    let start = cur;
    for (i, (s, c1, c2, e)) in corners.iter().enumerate() {
        if i > 0 {
            segments.push([cur, *s, *s]);
        }
        if s != e {
            segments.push([*c1, *c2, *e]);
        }
        cur = *e;
    }
    if closed && cur != start {
        segments.push([cur, start, start]);
    }
    SubPath { start, segments, closed }
}

fn unit(from: Point, to: Point) -> (Point, f64) {
    let (dx, dy) = (to.x - from.x, to.y - from.y);
    let l = (dx * dx + dy * dy).sqrt();
    if l > 0.0 { (pt(dx / l, dy / l), l) } else { (pt(0.0, 0.0), 0.0) }
}
