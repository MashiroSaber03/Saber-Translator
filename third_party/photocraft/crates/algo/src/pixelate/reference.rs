//! Original scalar kernels, kept as independent test oracles.
use super::*;
pub(crate) fn halftone(src: &Image, out: Rect, ctx: &Ctx, max_radius: f32, angles: [f32; 4]) -> Vec<f32> {
    let n = src.ch;
    let cc = ncol(ctx, n);
    let rmax = max_radius.max(1.0);
    // Cell side such that a full-size dot just covers the cell's corners.
    let cell = rmax * std::f32::consts::SQRT_2;
    let sub = subtractive(ctx);
    let rgb = via_rgb(ctx);
    let mut res = src.crop(out);
    let b = ctx.bounds;
    let (ox, oy) = (b.x0 as f32, b.y0 as f32);
    let rot: Vec<(f32, f32)> = (0..cc.max(3)).map(|c| angles[c.min(3)].to_radians().sin_cos()).collect();
    let clip = b.intersect(&src.rect);
    if clip.is_empty() {
        return res;
    }
    // Channel value at a point averaged over five taps (for Lab, of the RGB conversion).
    let value = |x: f32, y: f32, c: usize| -> f32 {
        let d = cell * 0.25;
        let mut acc = 0.0;
        let mut px = [0.0f32; MAXC];
        for (dx, dy) in [(0.0, 0.0), (-d, -d), (d, -d), (-d, d), (d, d)] {
            // Clamp to the bounds so cells straddling the edge read real pixels.
            let (ix, iy) = (((x + dx).floor() as i32).clamp(clip.x0, clip.x1 - 1), ((y + dy).floor() as i32).clamp(clip.y0, clip.y1 - 1));
            if rgb {
                for (k, p) in px.iter_mut().enumerate().take(n) {
                    *p = src.get(ix, iy, k);
                }
                acc += rgba(ctx, &px[..n])[c];
            } else {
                acc += src.get(ix, iy, c);
            }
        }
        acc / 5.0
    };
    let w = out.width() as usize;
    let chans = if rgb { 3 } else { cc };
    let mut vals = [0.0f32; MAXC];
    for (i, px) in res.chunks_exact_mut(n).enumerate() {
        let (x, y) = (out.x0 + (i % w) as i32, out.y0 + (i / w) as i32);
        let (fx, fy) = (x as f32 + 0.5 - ox, y as f32 + 0.5 - oy);
        for (c, v) in vals.iter_mut().enumerate().take(chans) {
            let (s, co) = rot[c];
            // Rotate into screen space.
            let (u, w2) = (fx * co + fy * s, -fx * s + fy * co);
            let (cu, cv) = ((u / cell).floor(), (w2 / cell).floor());
            let mut cov: f32 = 0.0;
            for dj in -1..=1 {
                for di in -1..=1 {
                    let (mu, mv) = ((cu + di as f32 + 0.5) * cell, (cv + dj as f32 + 0.5) * cell);
                    let d = ((u - mu).powi(2) + (w2 - mv).powi(2)).sqrt();
                    if d >= rmax + 0.5 {
                        continue;
                    }
                    // Back to document space to read the cell's value.
                    let (dx, dy) = (mu * co - mv * s + ox, mu * s + mv * co + oy);
                    let raw = value(dx, dy, c).clamp(0.0, 1.0);
                    let ink = if sub && !rgb { raw } else { 1.0 - raw };
                    // Dot area tracks ink: πr² = ink·cell² until dots touch, then grow to cover the corners.
                    let touch = std::f32::consts::FRAC_PI_4;
                    let r = if ink <= touch {
                        cell * (ink / std::f32::consts::PI).sqrt()
                    } else {
                        cell * (0.5 + (ink - touch) / (1.0 - touch) * (std::f32::consts::FRAC_1_SQRT_2 - 0.5))
                    };
                    // Tiny dots cannot cover more than their own area.
                    cov = cov.max((r - d + 0.5).clamp(0.0, 1.0).min(std::f32::consts::PI * r * r));
                }
            }
            *v = if sub && !rgb { cov } else { 1.0 - cov };
        }
        if rgb {
            let a = if ctx.alpha { px[n - 1] } else { 1.0 };
            set_rgba(ctx, px, [vals[0], vals[1], vals[2], a]);
        } else {
            px[..cc].copy_from_slice(&vals[..cc]);
        }
    }
    res
}

pub(crate) fn facet(src: &Image, out: Rect, ctx: &Ctx) -> Vec<f32> {
    let n = src.ch;
    let r = 2i32;
    let mut res = src.crop(out);
    let mut tmp = [0.0f32; MAXC];
    for (i, px) in res.chunks_exact_mut(n).enumerate() {
        let (x, y) = xy(out, i);
        let mut best = (f32::MAX, [0.0f32; MAXC]);
        for (qx, qy) in [(-r, -r), (0, -r), (-r, 0), (0, 0)] {
            let mut sum = [0.0f32; MAXC];
            let (mut s, mut s2) = (0.0f32, 0.0f32);
            for yy in y + qy..=y + qy + r {
                for xx in x + qx..=x + qx + r {
                    for (c, t) in tmp.iter_mut().enumerate().take(n) {
                        *t = src.get(xx, yy, c);
                        sum[c] += *t;
                    }
                    let l = luma(ctx, &tmp[..n]);
                    s += l;
                    s2 += l * l;
                }
            }
            let cnt = ((r + 1) * (r + 1)) as f32;
            let var = s2 / cnt - (s / cnt).powi(2);
            if var < best.0 {
                for v in sum.iter_mut() {
                    *v /= cnt;
                }
                best = (var, sum);
            }
        }
        px.copy_from_slice(&best.1[..n]);
    }
    res
}
