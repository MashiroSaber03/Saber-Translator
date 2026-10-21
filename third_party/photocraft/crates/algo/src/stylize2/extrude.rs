//! Rasterize each swept block only where it can hit, in the original cell order.

use super::{ExtrudeSpec, extrude_reach, read_px};
use crate::fxutil::{MAXC, luma, ncol, subtractive, unpremul_px};
use crate::image::Image;
use crate::noise::hash01;
use crate::{Ctx, ExtrudeType};
use photocraft_geom::Rect;

const MAX_PIXELS: usize = 4_194_304;
const MAX_CELLS: i64 = 262_144;
// Beyond this, f32 pixel centres can lose enough precision to widen the sweep bounds.
const MAX_COORD: i32 = 1 << 22;

pub(super) fn blocks(src: &Image, out: Rect, ctx: &Ctx, spec: &ExtrudeSpec) -> Option<Vec<f32>> {
    if spec.kind != ExtrudeType::Blocks || src.ch == 0 || src.ch > MAXC || !spec.size.is_finite() || !spec.depth.is_finite() {
        return None;
    }
    let b = ctx.bounds;
    if [b.x0, b.y0, b.x1, b.y1, out.x0, out.y0, out.x1, out.y1].iter().any(|v| v.unsigned_abs() > MAX_COORD as u32) {
        return None;
    }
    let n = src.ch;
    let w = out.width() as usize;
    let count = w.checked_mul(out.height() as usize)?;
    if count > MAX_PIXELS {
        return None;
    }
    let mut heights = Vec::new();
    heights.try_reserve_exact(count).ok()?;
    heights.resize(count, f32::NEG_INFINITY);
    let mut res = src.crop(out);
    let area = out.intersect(&b);
    if area.is_empty() {
        return Some(res);
    }
    let cs = spec.size.clamp(2.0, 255.0).round() as i32;
    let reach = extrude_reach(spec.depth);
    let k = (reach / cs as f32).ceil() as i32 + 1;
    let (gx0, gy0) = ((area.x0 - b.x0).div_euclid(cs), (area.y0 - b.y0).div_euclid(cs));
    let (gx1, gy1) = ((area.x1 - 1 - b.x0).div_euclid(cs), (area.y1 - 1 - b.y0).div_euclid(cs));
    let (cx0, cy0) = ((gx0 - k).max(0), (gy0 - k).max(0));
    let (cx1, cy1) = ((gx1 + k).min((b.width() as i32 - 1) / cs), (gy1 + k).min((b.height() as i32 - 1) / cs));
    if i64::from(cx1 - cx0 + 1).checked_mul(i64::from(cy1 - cy0 + 1))? > MAX_CELLS {
        return None;
    }
    let (bcx, bcy) = ((b.x0 + b.x1) as f32 / 2.0, (b.y0 + b.y1) as f32 / 2.0);
    let half_diag = ((b.width() as f32).hypot(b.height() as f32) / 2.0).max(1.0);
    let half = cs as f32 / 2.0;
    let cc = ncol(ctx, n);
    let sub = subtractive(ctx);
    // Ascending gy, then gx, is the tie order of every pixel's old candidate search.
    for cy in cy0..=cy1 {
        for cx in cx0..=cx1 {
            let r = Rect::new(b.x0 + cx * cs, b.y0 + cy * cs, b.x0 + (cx + 1) * cs, b.y0 + (cy + 1) * cs);
            let clipped = r.intersect(&b);
            if spec.mask_incomplete && clipped != r {
                continue;
            }
            let mut col = [0.0f32; MAXC];
            let mut cnt = 0.0;
            let step = (cs / 8).max(1) as usize;
            for yy in (clipped.y0..clipped.y1).step_by(step) {
                for xx in (clipped.x0..clipped.x1).step_by(step) {
                    let a = if ctx.alpha { src.get(xx, yy, n - 1) } else { 1.0 };
                    for (c, v) in col.iter_mut().enumerate().take(n) {
                        let s = src.get(xx, yy, c);
                        *v += if ctx.alpha && c < n - 1 { s * a } else { s };
                    }
                    cnt += 1.0;
                }
            }
            if cnt > 0.0 {
                for v in col.iter_mut().take(n) {
                    *v /= cnt;
                }
            }
            unpremul_px(&mut col[..n], ctx.alpha);
            let h = if spec.level_based { luma(ctx, &col[..n]).clamp(0.0, 1.0) } else { hash01(cx, cy, 21, spec.seed) };
            if !h.is_finite() {
                return None;
            }
            let (mx, my) = (r.x0 as f32 + half, r.y0 as f32 + half);
            let off = reach * h / half_diag;
            let (ox, oy) = ((mx - bcx) * off, (my - bcy) * off);
            // Two extra pixels conservatively cover rounding in the unchanged interval test.
            let sweep = Rect::new(
                (mx.min(mx + ox) - half).floor() as i32 - 2,
                (my.min(my + oy) - half).floor() as i32 - 2,
                (mx.max(mx + ox) + half).ceil() as i32 + 2,
                (my.max(my + oy) + half).ceil() as i32 + 2,
            )
            .intersect(&area);
            for y in sweep.y0..sweep.y1 {
                let gy = (y - b.y0).div_euclid(cs);
                if (cy - gy).abs() > k {
                    continue;
                }
                let fy = y as f32 + 0.5;
                let Some(iy) = interval(fy, my, oy, half) else { continue };
                for x in sweep.x0..sweep.x1 {
                    if (cx - (x - b.x0).div_euclid(cs)).abs() > k {
                        continue;
                    }
                    let i = (y - out.y0) as usize * w + (x - out.x0) as usize;
                    let height = heights.get_mut(i)?;
                    if *height >= h {
                        continue;
                    }
                    let fx = x as f32 + 0.5;
                    let Some(ix) = interval(fx, mx, ox, half) else { continue };
                    let (lo, hi) = (ix.0.max(iy.0), ix.1.min(iy.1));
                    if lo > hi {
                        continue;
                    }
                    *height = h;
                    let face = if hi >= 1.0 {
                        0
                    } else if ix.1 < iy.1 {
                        if ox > 0.0 { 1 } else { 2 }
                    } else if oy > 0.0 {
                        3
                    } else {
                        4
                    };
                    let px = res.get_mut(i * n..(i + 1) * n)?;
                    if face == 0 && !spec.solid_front {
                        let (sx, sy) = ((fx - (mx - bcx) * off).floor() as i32, (fy - (my - bcy) * off).floor() as i32);
                        read_px(src, sx.clamp(r.x0, r.x1 - 1), sy.clamp(r.y0, r.y1 - 1), px);
                    } else {
                        let shade = match face {
                            0 => 1.0,
                            1 => 0.85,
                            2 => 0.55,
                            3 => 0.95,
                            _ => 0.45,
                        };
                        for (v, c) in px.iter_mut().zip(&col).take(cc) {
                            *v = if sub { 1.0 - (1.0 - *c) * shade } else { *c * shade }.clamp(0.0, 1.0);
                        }
                        if ctx.alpha {
                            *px.last_mut()? = col.get(n - 1).copied()?;
                        }
                    }
                }
            }
        }
    }
    Some(res)
}

fn interval(p: f32, m: f32, o: f32, half: f32) -> Option<(f32, f32)> {
    if o.abs() < 1e-6 {
        return ((p - m).abs() <= half).then_some((0.0, 1.0));
    }
    let (a, b) = ((p - m - half) / o, (p - m + half) / o);
    let (lo, hi) = (a.min(b).max(0.0), a.max(b).min(1.0));
    (lo <= hi).then_some((lo, hi))
}
