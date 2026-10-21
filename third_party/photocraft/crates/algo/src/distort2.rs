//! Displace, Shear, ZigZag (inverse mapping over the reference bounds).

use photocraft_geom::Rect;

use crate::distort::{centre, remap};
use crate::image::Image;
use crate::other::edge_of;
use crate::{Ctx, UndefinedAreas, ZigZagStyle};

/// Max displacement at 100 % scale and full map contrast, px.
const DISPLACE_PX: f32 = 128.0;

/// Bilinear read of channel `c` of a map at map-local coordinates (clamped).
fn map_sample(map: &Image, u: f32, v: f32, c: usize) -> f32 {
    let r = map.rect;
    let (w, h) = (r.width() as i32, r.height() as i32);
    let (fx, fy) = (u - 0.5, v - 0.5);
    let (x0, y0) = (fx.floor(), fy.floor());
    let (ax, ay) = (fx - x0, fy - y0);
    let g = |x: i32, y: i32| map.get(r.x0 + x.clamp(0, w - 1), r.y0 + y.clamp(0, h - 1), c);
    let (x0, y0) = (x0 as i32, y0 as i32);
    let top = g(x0, y0) + (g(x0 + 1, y0) - g(x0, y0)) * ax;
    let bot = g(x0, y0 + 1) + (g(x0 + 1, y0 + 1) - g(x0, y0 + 1)) * ax;
    top + (bot - top) * ay
}

/// Displace: the map's first channel shifts pixels horizontally and its
/// second (or first, for one-channel maps) vertically; mid-gray = no shift.
/// The map is stretched over the bounds or tiled from their top-left.
#[allow(clippy::too_many_arguments)]
pub(crate) fn displace(src: &Image, out: Rect, ctx: &Ctx, scale: (f32, f32), stretch: bool, undefined: UndefinedAreas, map: Option<&Image>) -> Vec<f32> {
    let Some(map) = map.filter(|m| !m.rect.is_empty()) else { return src.crop(out) };
    let b = ctx.bounds;
    let (mw, mh) = (map.rect.width() as f32, map.rect.height() as f32);
    let (bw, bh) = (b.width().max(1) as f32, b.height().max(1) as f32);
    // Colour channels of the map (alpha excluded when present: 2 = gray+alpha, 4 = RGBA…).
    let vc = if map.ch >= 3 { 1 } else { 0 };
    let (sx, sy) = (scale.0 / 100.0 * DISPLACE_PX, scale.1 / 100.0 * DISPLACE_PX);
    remap(src, out, ctx, edge_of(undefined), |x, y| {
        let (lx, ly) = (x - b.x0 as f32, y - b.y0 as f32);
        let (u, v) = if stretch { (lx / bw * mw, ly / bh * mh) } else { (lx.rem_euclid(mw), ly.rem_euclid(mh)) };
        let dx = (map_sample(map, u, v, 0) - 0.5) * 2.0 * sx;
        let dy = (map_sample(map, u, v, vc) - 0.5) * 2.0 * sy;
        (x + dx, y + dy)
    })
}

/// Catmull–Rom interpolation through sorted `(t, value)` points (clamped ends).
fn curve_at(pts: &[[f32; 2]], t: f32) -> f32 {
    match pts.len() {
        0 => 0.0,
        1 => pts[0][1],
        _ => {
            if t <= pts[0][0] {
                return pts[0][1];
            }
            let last = pts.len() - 1;
            if t >= pts[last][0] {
                return pts[last][1];
            }
            let i = pts.windows(2).position(|w| t >= w[0][0] && t <= w[1][0]).unwrap_or(0);
            let p1 = pts[i];
            let p2 = pts[i + 1];
            let p0 = if i > 0 { pts[i - 1] } else { p1 };
            let p3 = if i + 2 <= last { pts[i + 2] } else { p2 };
            let span = (p2[0] - p1[0]).max(1e-6);
            let s = (t - p1[0]) / span;
            // Tangents scaled to the span (non-uniform Catmull–Rom).
            let m1 = if i > 0 { (p2[1] - p0[1]) / (p2[0] - p0[0]).max(1e-6) * span } else { p2[1] - p1[1] };
            let m2 = if i + 2 <= last { (p3[1] - p1[1]) / (p3[0] - p1[0]).max(1e-6) * span } else { p2[1] - p1[1] };
            let (s2, s3) = (s * s, s * s * s);
            (2.0 * s3 - 3.0 * s2 + 1.0) * p1[1] + (s3 - 2.0 * s2 + s) * m1 + (-2.0 * s3 + 3.0 * s2) * p2[1] + (s3 - s2) * m2
        }
    }
}

/// Shear: each row shifts horizontally by the curve through `points`
/// (`[t, offset]`, t top→bottom, offset in half-widths).
pub(crate) fn shear(src: &Image, out: Rect, ctx: &Ctx, points: &[[f32; 2]], undefined: UndefinedAreas) -> Vec<f32> {
    let b = ctx.bounds;
    let mut pts: Vec<[f32; 2]> = points.iter().map(|p| [p[0].clamp(0.0, 1.0), p[1].clamp(-1.0, 1.0)]).collect();
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]));
    pts.dedup_by(|a, b| (a[0] - b[0]).abs() < 1e-6);
    let (bw, bh) = (b.width().max(1) as f32, b.height().max(1) as f32);
    // Per-row offsets are shared by every pixel of the row.
    let rows: Vec<f32> = (out.y0..out.y1).map(|y| curve_at(&pts, (y as f32 + 0.5 - b.y0 as f32) / bh) * bw / 2.0).collect();
    remap(src, out, ctx, edge_of(undefined), |x, y| {
        let i = ((y - 0.5) as i32 - out.y0).clamp(0, rows.len() as i32 - 1) as usize;
        (x - rows[i], y)
    })
}

/// ZigZag: radial ripples inside the bounds' inscribed circle — angular
/// (around centre), radial (out from centre) or both (pond ripples).
pub(crate) fn zigzag(src: &Image, out: Rect, ctx: &Ctx, amount: f32, ridges: f32, style: ZigZagStyle) -> Vec<f32> {
    let (cx, cy, rr) = centre(ctx.bounds);
    let a = (amount / 100.0).clamp(-1.0, 1.0);
    let k = ridges.clamp(0.0, 20.0).max(0.5);
    remap(src, out, ctx, crate::image::Edge::Repeat, |x, y| {
        let (dx, dy) = (x - cx, y - cy);
        let r = (dx * dx + dy * dy).sqrt();
        if r >= rr || r <= 0.0 || a == 0.0 {
            return (x, y);
        }
        let rn = r / rr;
        let wave = (std::f32::consts::TAU * k * rn).sin() * (1.0 - rn);
        let (radial, angular) = match style {
            ZigZagStyle::OutFromCenter => (a * wave * rr / (4.0 * k), 0.0),
            ZigZagStyle::AroundCenter => (0.0, a * wave * std::f32::consts::PI / (2.0 * k)),
            ZigZagStyle::PondRipples => (a * wave * rr / (8.0 * k), a * wave * std::f32::consts::PI / (4.0 * k)),
        };
        let th = dy.atan2(dx) + angular;
        let rs = (r + radial).max(0.0);
        (cx + rs * th.cos(), cy + rs * th.sin())
    })
}

#[cfg(test)]
mod tests {
    use super::curve_at;

    #[test]
    fn curve_passes_through_points_and_clamps() {
        let pts = [[0.0, 0.0], [0.5, 0.4], [1.0, -0.2]];
        assert!((curve_at(&pts, 0.5) - 0.4).abs() < 1e-6);
        assert!((curve_at(&pts, 1.0) + 0.2).abs() < 1e-6);
        assert_eq!(curve_at(&pts, -1.0), 0.0);
        assert_eq!(curve_at(&[[0.0, 0.3], [1.0, 0.3]], 0.42), 0.3);
    }
}
