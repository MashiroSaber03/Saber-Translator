//! Brush tip outlines for the painting cursor: closed rings in units of the brush radius
//! (1 = size / 2), screen-oriented (x right, y down), ready for the UI to scale by the radius.

use std::collections::HashMap;

use crate::brush::{BrushSettings, TipShape};
use crate::tile::GrayTile;

/// Segments in the outline of a computed round tip.
const ELLIPSE_SEGMENTS: usize = 64;
/// Budget for all traced ring vertices; the smallest rings are dropped first.
const MAX_POINTS: usize = 512;
/// A traced ring smaller than this many bitmap pixels squared is noise.
const MIN_AREA: f32 = 0.9;
/// "Full size" outlines everything above 1/255; the normal cursor outlines the 50 % contour.
const FULL_THRESHOLD: f32 = 1.0 / 255.0;
const HALF_THRESHOLD: f32 = 0.5;

/// The tip outline as closed rings, or `None` when the tip has no usable shape. `full` outlines
/// the whole tip; otherwise the 50 % contour (the Photoshop normal brush tip cursor).
pub fn tip_outline(b: &BrushSettings, full: bool) -> Option<Vec<Vec<[f32; 2]>>> {
    if !b.size.is_finite() || b.size <= 0.0 || !b.angle.is_finite() || !b.roundness.is_finite() {
        return None;
    }
    let ro = b.roundness.clamp(0.0, 1.0).max(0.5 / (b.size / 2.0)).min(1.0);
    let (sn, cs) = b.angle.to_radians().sin_cos();
    let (fx, fy) = (if b.flip_x { -1.0 } else { 1.0 }, if b.flip_y { -1.0 } else { 1.0 });
    match &b.tip {
        TipShape::Round => {
            // The normal cursor is the 50 % contour, matching the plain-circle path (#2744):
            // the soft tip falls off as (1 - t²)⁴, not the old smoothstep midpoint.
            let f = if full { 1.0 } else { crate::half_coverage_radius(1.0, b.hardness) };
            let ring = (0..ELLIPSE_SEGMENTS)
                .map(|i| {
                    let t = std::f32::consts::TAU * i as f32 / ELLIPSE_SEGMENTS as f32;
                    // Tip-local (y up), rotated CCW by the angle, then flipped to screen y down.
                    let (lx, ly) = (t.cos() * f, t.sin() * f * ro);
                    [lx * cs - ly * sn, -(lx * sn + ly * cs)]
                })
                .collect();
            Some(vec![ring])
        }
        TipShape::Sampled(_) | TipShape::Stored(_) => {
            let t = b.tip.bitmap()?;
            if !t.is_valid() {
                return None;
            }
            trace(t, if full { FULL_THRESHOLD } else { HALF_THRESHOLD }, (sn, cs, fx, fy, ro))
        }
    }
}

/// Trace the contour(s) above `thr` (resampled ~2x, capped at 160 px a side) and map them with
/// the same tip transform as `render::BrushContext::rasterize`.
fn trace(t: &GrayTile, thr: f32, xf: (f32, f32, f32, f32, f32)) -> Option<Vec<Vec<[f32; 2]>>> {
    let (sn, cs, fx, fy, ro) = xf;
    let (tw, th) = (t.width as f32, t.height as f32);
    let big = tw.max(th);
    let f = if big <= 160.0 { 2.0 } else { 160.0 / big };
    let (gw, gh) = (((tw * f).ceil() as usize).max(1), ((th * f).ceil() as usize).max(1));
    let mut vals = vec![0.0f32; gw * gh];
    for gy in 0..gh {
        for gx in 0..gw {
            let (px, py) = ((gx as f32 + 0.5) / f - 0.5, (gy as f32 + 0.5) / f - 0.5);
            vals[gy * gw + gx] = sample(t, px, py);
        }
    }
    let mut rings: Vec<Vec<[f32; 2]>> = march(&vals, gw, gh, thr).into_iter().filter(|r| area(r) >= MIN_AREA * f * f).collect();
    if rings.is_empty() {
        return None;
    }
    rings.sort_by(|a, b| area(b).total_cmp(&area(a)));
    let mut tol = 0.5;
    let mut thin: Vec<Vec<[f32; 2]>>;
    loop {
        thin = rings.iter().map(|r| simplify(r, tol)).collect();
        if thin.iter().map(Vec::len).sum::<usize>() <= MAX_POINTS || tol >= 8.0 {
            break;
        }
        tol *= 1.5;
    }
    let mut budget = MAX_POINTS;
    thin.retain(|r| {
        let keep = r.len() >= 3 && r.len() <= budget;
        if keep {
            budget -= r.len();
        }
        keep
    });
    if thin.is_empty() {
        return None;
    }
    let s = 2.0 / big;
    let map = |[gx, gy]: [f32; 2]| {
        let (px, py) = ((gx + 0.5) / f - 0.5, (gy + 0.5) / f - 0.5);
        let (u, v) = ((px - tw / 2.0) * s * fx, -(py - th / 2.0) * s * ro * fy);
        let (ux, uy) = (u * cs - v * sn, u * sn + v * cs);
        [ux, -uy]
    };
    let rings: Vec<Vec<[f32; 2]>> = thin.into_iter().map(|r| r.into_iter().map(map).collect()).collect();
    rings.iter().flatten().all(|p| p[0].is_finite() && p[1].is_finite()).then_some(rings)
}

/// Bilinear sample with zero outside (tips fade to nothing at their edges).
fn sample(t: &GrayTile, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (tx, ty) = (x - x0, y - y0);
    let (x0, y0) = (x0 as i64, y0 as i64);
    let get = |x: i64, y: i64| if x < 0 || y < 0 || x >= t.width as i64 || y >= t.height as i64 { 0.0 } else { t.get(x as u32, y as u32) };
    let a = get(x0, y0) + (get(x0 + 1, y0) - get(x0, y0)) * tx;
    let b = get(x0, y0 + 1) + (get(x0 + 1, y0 + 1) - get(x0, y0 + 1)) * tx;
    a + (b - a) * ty
}

/// Shoelace area of a ring.
fn area(r: &[[f32; 2]]) -> f32 {
    if r.len() < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..r.len() {
        let (a, b) = (r[i], r[(i + 1) % r.len()]);
        s += a[0] * b[1] - b[0] * a[1];
    }
    (s * 0.5).abs()
}

/// Link segments end-to-end into closed rings (quantised endpoint keys absorb float noise).
fn stitch(segs: Vec<([f32; 2], [f32; 2])>) -> Vec<Vec<[f32; 2]>> {
    let key = |p: [f32; 2]| ((p[0] * 4096.0).round() as i64, (p[1] * 4096.0).round() as i64);
    let mut ends: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, (a, b)) in segs.iter().enumerate() {
        ends.entry(key(*a)).or_default().push(i);
        ends.entry(key(*b)).or_default().push(i);
    }
    let mut used = vec![false; segs.len()];
    let mut rings = Vec::new();
    for s in 0..segs.len() {
        if used[s] {
            continue;
        }
        used[s] = true;
        let mut ring = vec![segs[s].0, segs[s].1];
        while let Some(tail) = ring.last().copied() {
            let Some(&next) = ends.get(&key(tail)).and_then(|c| c.iter().find(|&&i| !used[i])) else { break };
            used[next] = true;
            let (a, b) = segs[next];
            let p = if key(a) == key(tail) { b } else { a };
            if key(p) == key(ring[0]) {
                break;
            }
            ring.push(p);
        }
        if ring.len() >= 3 {
            rings.push(ring);
        }
    }
    rings
}

/// Marching squares over `v` (w by h samples at lattice nodes), outside treated as 0 so every
/// contour closes; returns rings in node coordinates.
fn march(v: &[f32], w: usize, h: usize, thr: f32) -> Vec<Vec<[f32; 2]>> {
    let val = |x: i64, y: i64| -> f32 { if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 { 0.0 } else { v[y as usize * w + x as usize] } };
    let lerp = |v0: f32, v1: f32, a: (f32, f32), b: (f32, f32)| -> [f32; 2] {
        let t = if (v1 - v0).abs() < 1e-9 { 0.5 } else { ((thr - v0) / (v1 - v0)).clamp(0.0, 1.0) };
        [a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t]
    };
    let mut segs: Vec<([f32; 2], [f32; 2])> = Vec::new();
    for y in -1..h as i64 {
        for x in -1..w as i64 {
            let (va, vb, vc, vd) = (val(x, y), val(x + 1, y), val(x + 1, y + 1), val(x, y + 1));
            let idx = u8::from(va >= thr) | (u8::from(vb >= thr) << 1) | (u8::from(vc >= thr) << 2) | (u8::from(vd >= thr) << 3);
            if idx == 0 || idx == 15 {
                continue;
            }
            let (fx, fy) = (x as f32, y as f32);
            let top = lerp(va, vb, (fx, fy), (fx + 1.0, fy));
            let right = lerp(vb, vc, (fx + 1.0, fy), (fx + 1.0, fy + 1.0));
            let bottom = lerp(vd, vc, (fx, fy + 1.0), (fx + 1.0, fy + 1.0));
            let left = lerp(va, vd, (fx, fy), (fx, fy + 1.0));
            let mid = (va + vb + vc + vd) * 0.25 >= thr;
            match idx {
                1 | 14 => segs.push((top, left)),
                2 | 13 => segs.push((top, right)),
                3 | 12 => segs.push((right, left)),
                4 | 11 => segs.push((right, bottom)),
                6 | 9 => segs.push((top, bottom)),
                7 | 8 => segs.push((left, bottom)),
                5 | 10 => {
                    let opposite = (idx == 5) != mid;
                    if opposite {
                        segs.push((top, right));
                        segs.push((left, bottom));
                    } else {
                        segs.push((top, left));
                        segs.push((right, bottom));
                    }
                }
                _ => {}
            }
        }
    }
    stitch(segs)
}
/// Douglas-Peucker on a closed ring (split at the point farthest from the first).
fn simplify(ring: &[[f32; 2]], tol: f32) -> Vec<[f32; 2]> {
    if ring.len() <= 4 {
        return ring.to_vec();
    }
    let (mut far, mut best) = (0, 0.0);
    for (i, p) in ring.iter().enumerate().skip(1) {
        let d = (p[0] - ring[0][0]).powi(2) + (p[1] - ring[0][1]).powi(2);
        if d > best {
            best = d;
            far = i;
        }
    }
    let mut open: Vec<[f32; 2]> = ring[far..].to_vec();
    open.push(ring[0]);
    let mut out = Vec::new();
    dp(&ring[..=far], tol, &mut out);
    out.pop();
    dp(&open, tol, &mut out);
    out.pop();
    out
}

/// Iterative Douglas-Peucker: keep the points of `pts` farther than `tol` from the chord.
fn dp(pts: &[[f32; 2]], tol: f32, out: &mut Vec<[f32; 2]>) {
    let n = pts.len();
    if n == 0 {
        return;
    }
    if n < 3 {
        out.extend_from_slice(pts);
        return;
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack = vec![(0usize, n - 1)];
    while let Some((i, j)) = stack.pop() {
        if j <= i + 1 {
            continue;
        }
        let (mut dmax, mut k) = (0.0f32, i);
        for m in i + 1..j {
            let d = seg_dist2(pts[m], pts[i], pts[j]);
            if d > dmax {
                dmax = d;
                k = m;
            }
        }
        if dmax > tol * tol {
            keep[k] = true;
            stack.push((i, k));
            stack.push((k, j));
        }
    }
    for (i, p) in pts.iter().enumerate() {
        if keep[i] {
            out.push(*p);
        }
    }
}

/// Squared distance from `p` to the segment `a`-`b`.
fn seg_dist2(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let (abx, aby) = (b[0] - a[0], b[1] - a[1]);
    let t = (((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / (abx * abx + aby * aby).max(1e-12)).clamp(0.0, 1.0);
    (p[0] - a[0] - abx * t).powi(2) + (p[1] - a[1] - aby * t).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_radius(rings: &[Vec<[f32; 2]>]) -> f32 {
        rings.iter().flatten().fold(0.0f32, |m, p| m.max(p[0].hypot(p[1])))
    }

    /// The normal cursor of a rotated or flattened round tip is the same 50 % contour as the
    /// plain-circle path (#2744), and the full cursor reaches the whole radius.
    #[test]
    fn round_outlines_sit_on_the_half_coverage_contour() {
        for hardness in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let b = BrushSettings { size: 40.0, angle: 33.0, roundness: 0.4, hardness, ..Default::default() };
            let normal = tip_outline(&b, false).expect("a rotated round tip traces");
            let full = tip_outline(&b, true).expect("a rotated round tip traces");
            assert!((max_radius(&normal) - crate::half_coverage_radius(1.0, hardness)).abs() < 1e-4, "hardness {hardness}");
            assert!((max_radius(&full) - 1.0).abs() < 1e-4, "hardness {hardness}");
        }
    }

    /// A sampled tip traces its 50 % alpha contour, inside the tip's square; a blank tip has none.
    #[test]
    fn sampled_outlines_stay_inside_the_tip_and_blank_tips_have_none() {
        let disc = GrayTile::from_fn(32, 32, |x, y| {
            let (dx, dy) = (x as f32 - 15.5, y as f32 - 15.5);
            (1.0 - (dx * dx + dy * dy).sqrt() / 16.0).clamp(0.0, 1.0)
        });
        let b = BrushSettings { size: 20.0, tip: TipShape::Sampled(disc), ..Default::default() };
        let rings = tip_outline(&b, false).expect("a sampled disc traces");
        assert!(rings.iter().all(|r| r.len() >= 3));
        assert!(rings.iter().flatten().all(|p| p[0].is_finite() && p[1].is_finite() && p[0].hypot(p[1]) <= 1.001));
        let blank = GrayTile::from_fn(8, 8, |_, _| 0.0);
        let b = BrushSettings { size: 20.0, tip: TipShape::Sampled(blank), ..Default::default() };
        assert!(tip_outline(&b, false).is_none());
    }

    /// Hostile brush data yields no outline instead of NaNs or a panic.
    #[test]
    fn hostile_settings_get_no_outline() {
        for size in [f32::NAN, f32::INFINITY, 0.0, -1.0] {
            let b = BrushSettings { size, angle: 33.0, roundness: 0.5, ..Default::default() };
            assert!(tip_outline(&b, false).is_none(), "size {size}");
        }
        for angle in [f32::NAN, f32::INFINITY] {
            let b = BrushSettings { size: 40.0, angle, roundness: 0.5, ..Default::default() };
            assert!(tip_outline(&b, false).is_none(), "angle {angle}");
        }
    }
}
