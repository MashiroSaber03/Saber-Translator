//! Frozen Iris Blur kernel from base 7722172585a01cbdb93c06f0f5ff2634fcb17999.
use super::*;
fn sigma_of(blur: f32) -> f32 {
    blur.clamp(0.0, 500.0) * 0.6
}

const LEVELS: usize = 6;

/// Blurs `src` over `out` with a per-pixel σ (`sigma(x, y)`, ≤ `smax`).
fn variable_blur(src: &Image, out: Rect, ctx: &Ctx, smax: f32, sigma: impl Fn(f32, f32) -> f32) -> Vec<f32> {
    let n = src.ch;
    if smax < 0.3 {
        return src.crop(out);
    }
    let win = src.rect;
    let (ww, wh) = (win.width() as usize, win.height() as usize);
    let (ow, oh) = (out.width() as usize, out.height() as usize);
    // Level σs: 0, then smax / 2^(LEVELS-1) … smax.
    let lv: Vec<f32> = std::iter::once(0.0).chain((0..LEVELS).map(|k| smax / 2f32.powi((LEVELS - 1 - k) as i32))).collect();
    let sig: Vec<f32> = (0..ow * oh)
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
    let p = premul_window(src, win, ctx.alpha);
    let mut acc = vec![0.0f32; ow * oh * n];
    let mut buf = Vec::new();
    for (k, &lvk) in lv.iter().enumerate() {
        if !sig.iter().any(|&s| weight(k, s) > 0.0) {
            continue;
        }
        let level: &[f32] = if k == 0 {
            &p
        } else {
            buf.clear();
            buf.extend_from_slice(&p);
            gauss_blur_n(&mut buf, ww, wh, n, lvk);
            &buf
        };
        for (i, &s) in sig.iter().enumerate() {
            let wk = weight(k, s);
            if wk <= 0.0 {
                continue;
            }
            let (x, y) = xy(out, i);
            let o = (((y - win.y0) as usize) * ww + (x - win.x0) as usize) * n;
            for c in 0..n {
                acc[i * n + c] += level[o + c] * wk;
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

/// Normalized (super)elliptical distance of a point from an iris/spin pin.
#[allow(clippy::too_many_arguments)]
fn pin_distance(x: f32, y: f32, b: Rect, px: f32, py: f32, rx: f32, ry: f32, angle: f32, m: f32) -> (f32, f32, f32) {
    let ss = short_side(b);
    let (cx, cy) = (b.x0 as f32 + px * b.width() as f32, b.y0 as f32 + py * b.height() as f32);
    let angle_deg = if angle.is_finite() { angle } else { 0.0 };
    let (s, c) = angle_deg.to_radians().sin_cos();
    let (dx, dy) = (x - cx, y - cy);
    let (u, v) = (dx * c + dy * s, -dx * s + dy * c);
    let (u, v) = (u / (rx.max(1e-3) * ss), v / (ry.max(1e-3) * ss));
    let e = if (m - 2.0).abs() < 1e-3 { (u * u + v * v).sqrt() } else { (u.abs().powf(m) + v.abs().powf(m)).powf(1.0 / m) };
    if !e.is_finite() { (f32::INFINITY, 0.0, 0.0) } else { (e, u, v) }
}

/// Iris Blur: everything outside each pin's ellipse is blurred; the sharp
/// core is `feather` of the ellipse. Sharpness from any pin wins.
pub(crate) fn iris(src: &Image, out: Rect, ctx: &Ctx, pins: &[IrisPin]) -> Vec<f32> {
    let b = ctx.bounds;
    let smax = pins.iter().map(|p| sigma_of(p.blur)).fold(0.0, f32::max);
    if pins.is_empty() {
        return src.crop(out);
    }
    variable_blur(src, out, ctx, smax, |x, y| {
        pins.iter()
            .map(|p| {
                let m = 2.0 + p.roundness.clamp(0.0, 100.0) / 100.0 * 6.0;
                let (e, _, _) = pin_distance(x, y, b, p.x, p.y, p.radius_x, p.radius_y, p.angle, m);
                sigma_of(p.blur) * smoothstep(p.feather.clamp(0.0, 0.999), 1.0, e)
            })
            .fold(f32::MAX, f32::min)
    })
}
