//! Frozen Tilt-Shift kernel from base a28d786638ffbb84c5fb7d5d30ea1c3d594cd3bc.
use super::*;
/// Gaussian σ for a gallery blur amount in px.
fn sigma_of(blur: f32) -> f32 {
    blur.clamp(0.0, 500.0) * 0.6
}

const LEVELS: usize = 6;

/// Blurs `src` over `out` with a per-pixel σ (`sigma(x, y)`, ≤ `smax`).
fn variable_blur(src: &Image, out: Rect, ctx: &Ctx, smax: f32, trim_levels: bool, sigma: impl Fn(f32, f32) -> f32) -> Vec<f32> {
    let n = src.ch;
    if n == 0 || out.is_empty() {
        return Vec::new();
    }
    if smax < 0.3 {
        return src.crop(out);
    }
    let win = src.rect;
    let (ww, wh) = (win.width() as usize, win.height() as usize);
    let (ow, oh) = (out.width() as usize, out.height() as usize);
    let Some(row_len) = ww.checked_mul(n).filter(|&len| len > 0) else { return Vec::new() };
    let Some(source_len) = row_len.checked_mul(wh) else { return Vec::new() };
    let Some(pixels) = ow.checked_mul(oh) else { return Vec::new() };
    let Some(samples) = pixels.checked_mul(n) else { return Vec::new() };
    if trim_levels && src.data.len() != source_len {
        return Vec::new();
    }
    // Level σs: 0, then smax / 2^(LEVELS-1) … smax.
    let lv: Vec<f32> = std::iter::once(0.0).chain((0..LEVELS).map(|k| smax / 2f32.powi((LEVELS - 1 - k) as i32))).collect();
    let sig: Vec<f32> = (0..pixels)
        .map(|i| {
            let (x, y) = xy(out, i);
            sigma(x as f32 + 0.5, y as f32 + 0.5).clamp(0.0, smax)
        })
        .collect();
    // Hat-function weight of level k for σ s.
    let weight = |k: usize, s: f32| -> f32 {
        let c = lv[k];
        if k > 0 && s >= lv[k - 1] && s <= c {
            (s - lv[k - 1]) / (c - lv[k - 1]).max(1e-6)
        } else if k + 1 < lv.len() && s >= c && s <= lv[k + 1] {
            (lv[k + 1] - s) / (lv[k + 1] - c).max(1e-6)
        } else if k + 1 == lv.len() && s >= c {
            1.0
        } else {
            0.0
        }
    };
    let p = if trim_levels {
        // The window is exactly src.rect: read interleaved channels together,
        // retaining the same multiplication and alpha order as premul_window.
        let mut p = src.data.clone();
        premultiply(&mut p, n, ctx.alpha);
        p
    } else {
        premul_window(src, win, ctx.alpha)
    };
    let mut acc = vec![0.0f32; samples];
    let mut buf = Vec::new();
    for (k, &lvk) in lv.iter().enumerate() {
        if !sig.iter().any(|&s| weight(k, s) > 0.0) {
            continue;
        }
        let (level, lw): (&[f32], usize) = if k == 0 {
            (&p, ww)
        } else {
            // Only Iris opts in. Each level needs its own box-cascade reach,
            // not the maximum blur's trailing halo. Keep the original top/left:
            // moving the running sums' start changes floating-point cancellation.
            let trim = trim_levels && win.intersect(&out) == out;
            let reach = if !trim || lvk < 0.3 { 0 } else { gauss_box_radii(lvk).iter().sum::<usize>() };
            let reach = i32::try_from(reach).unwrap_or(i32::MAX);
            let lw = if trim { (i64::from(out.x1.saturating_add(reach).min(win.x1)) - i64::from(win.x0)) as usize } else { ww };
            let lh = if trim { (i64::from(out.y1.saturating_add(reach).min(win.y1)) - i64::from(win.y0)) as usize } else { wh };
            buf.clear();
            if buf.capacity() == 0 {
                // Match the old single full-window allocation. Row appends must
                // not geometrically grow the capacity beyond that window.
                buf.reserve_exact(p.len());
            }
            if trim {
                for row in p.chunks_exact(row_len).take(lh) {
                    if let Some(row) = row.get(..lw * n) {
                        buf.extend_from_slice(row);
                    }
                }
            } else {
                buf.extend_from_slice(&p);
            }
            gauss_blur_n(&mut buf, lw, lh, n, lvk);
            (&buf, lw)
        };
        for (i, &s) in sig.iter().enumerate() {
            let wk = weight(k, s);
            if wk <= 0.0 {
                continue;
            }
            let (x, y) = xy(out, i);
            let o = (((y - win.y0) as usize) * lw + (x - win.x0) as usize) * n;
            if let (Some(dst), Some(px)) = (acc.get_mut(i * n..(i + 1) * n), level.get(o..o + n)) {
                for (dst, px) in dst.iter_mut().zip(px) {
                    *dst += px * wk;
                }
            }
        }
    }
    for px in acc.chunks_exact_mut(n) {
        unpremul_px(px, ctx.alpha);
    }
    acc
}

fn short_side(b: Rect) -> f32 {
    (b.width().min(b.height()) as f32).max(1.0)
}

/// Tilt-Shift: a sharp band through the centre at `angle`, blurring with
/// distance beyond `focus` over `transition` (both fractions of the shorter side).
#[allow(clippy::too_many_arguments)]
pub(crate) fn tilt_shift(src: &Image, out: Rect, ctx: &Ctx, blur: f32, centre: (f32, f32), angle: f32, focus: f32, transition: f32) -> Vec<f32> {
    let b = ctx.bounds;
    let ss = short_side(b);
    let (cx, cy) = (b.x0 as f32 + centre.0 * b.width() as f32, b.y0 as f32 + centre.1 * b.height() as f32);
    // Band normal (the band runs along `angle`).
    let (s, c) = angle.to_radians().sin_cos();
    let (nx, ny) = (s, c);
    let smax = sigma_of(blur);
    let (f0, f1) = (focus.max(0.0), focus.max(0.0) + transition.max(1e-3));
    variable_blur(src, out, ctx, smax, false, |x, y| {
        let d = ((x - cx) * nx + (y - cy) * ny).abs() / ss;
        smax * smoothstep(f0, f1, d)
    })
}
