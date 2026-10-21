//! Filter › Other: Custom (5×5 convolution) and HSB/HSL.

use photocraft_color::ColorMode;
use photocraft_geom::Rect;

use crate::fxutil::{ncol, rgba, set_rgba, xy};
use crate::image::Image;
use crate::{Ctx, HsbModel};

/// Custom: `Σ kernel·pixel / scale + offset/255` per colour channel (alpha
/// kept). Kernels shorter than 25 entries are zero-padded.
pub(crate) fn custom(src: &Image, out: Rect, ctx: &Ctx, kernel: &[f32], scale: f32, offset: f32) -> Vec<f32> {
    let n = src.ch;
    let cc = ncol(ctx, n);
    let k: Vec<(i32, i32, f32)> =
        (0..25).filter_map(|i| kernel.get(i).copied().filter(|v| *v != 0.0).map(|v| (i as i32 % 5 - 2, i as i32 / 5 - 2, v))).collect();
    let s = if scale == 0.0 { 1.0 } else { scale };
    let off = offset / 255.0;
    let mut res = src.crop(out);
    if k.is_empty() {
        return res;
    }
    for (i, px) in res.chunks_exact_mut(n).enumerate() {
        let (x, y) = xy(out, i);
        for (c, v) in px.iter_mut().enumerate().take(cc) {
            let sum: f32 = k.iter().map(|&(dx, dy, w)| src.get(x + dx, y + dy, c) * w).sum();
            *v = sum / s + off;
        }
    }
    res
}

/// RGB → HSB (hue, saturation, brightness), all 0–1.
pub(crate) fn rgb_to_hsb(c: [f32; 3]) -> [f32; 3] {
    let (mx, mn) = (c[0].max(c[1]).max(c[2]), c[0].min(c[1]).min(c[2]));
    let d = mx - mn;
    let h = hue(c, mx, d);
    [h, if mx > 0.0 { d / mx } else { 0.0 }, mx]
}

/// RGB → HSL, all 0–1.
pub(crate) fn rgb_to_hsl(c: [f32; 3]) -> [f32; 3] {
    let (mx, mn) = (c[0].max(c[1]).max(c[2]), c[0].min(c[1]).min(c[2]));
    let d = mx - mn;
    let l = (mx + mn) / 2.0;
    let s = if d <= 0.0 { 0.0 } else { d / (1.0 - (2.0 * l - 1.0).abs()).max(1e-7) };
    [hue(c, mx, d), s.clamp(0.0, 1.0), l]
}

fn hue(c: [f32; 3], mx: f32, d: f32) -> f32 {
    if d <= 0.0 {
        return 0.0;
    }
    let h = if mx == c[0] {
        ((c[1] - c[2]) / d).rem_euclid(6.0)
    } else if mx == c[1] {
        (c[2] - c[0]) / d + 2.0
    } else {
        (c[0] - c[1]) / d + 4.0
    };
    h / 6.0
}

/// Chroma/min helper: RGB from hue, chroma and the value to add.
fn from_hcm(h: f32, chroma: f32, m: f32) -> [f32; 3] {
    let hp = h.rem_euclid(1.0) * 6.0;
    let x = chroma * (1.0 - (hp.rem_euclid(2.0) - 1.0).abs());
    let (r, g, b) = match hp as i32 {
        0 => (chroma, x, 0.0),
        1 => (x, chroma, 0.0),
        2 => (0.0, chroma, x),
        3 => (0.0, x, chroma),
        4 => (x, 0.0, chroma),
        _ => (chroma, 0.0, x),
    };
    [r + m, g + m, b + m]
}

pub(crate) fn hsb_to_rgb(c: [f32; 3]) -> [f32; 3] {
    let chroma = c[2] * c[1];
    from_hcm(c[0], chroma, c[2] - chroma)
}

pub(crate) fn hsl_to_rgb(c: [f32; 3]) -> [f32; 3] {
    let chroma = (1.0 - (2.0 * c[2] - 1.0).abs()) * c[1];
    from_hcm(c[0], chroma, c[2] - chroma / 2.0)
}

/// HSB/HSL: reinterprets the three colour channels — encodes RGB as
/// H, S, B/L in the R, G, B channels, or decodes such channels back to RGB.
pub(crate) fn hsb_hsl(src: &Image, out: Rect, ctx: &Ctx, input: HsbModel, output: HsbModel) -> Vec<f32> {
    let n = src.ch;
    let mut res = src.crop(out);
    if input == output {
        return res;
    }
    let native = ctx.mode == ColorMode::Rgb;
    for px in res.chunks_exact_mut(n) {
        let c4 = if native { [px[0], px[1], px[2], 1.0] } else { rgba(ctx, px) };
        let c = [c4[0].clamp(0.0, 1.0), c4[1].clamp(0.0, 1.0), c4[2].clamp(0.0, 1.0)];
        let rgb = match input {
            HsbModel::Rgb => c,
            HsbModel::Hsb => hsb_to_rgb(c),
            HsbModel::Hsl => hsl_to_rgb(c),
        };
        let o = match output {
            HsbModel::Rgb => rgb,
            HsbModel::Hsb => rgb_to_hsb(rgb),
            HsbModel::Hsl => rgb_to_hsl(rgb),
        };
        if native {
            px[..3].copy_from_slice(&o);
        } else {
            set_rgba(ctx, px, [o[0], o[1], o[2], c4[3]]);
        }
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hsb_hsl_round_trip() {
        for c in [[0.2, 0.7, 0.4], [1.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.1, 0.2, 0.9], [0.9, 0.8, 0.05]] {
            let a = hsb_to_rgb(rgb_to_hsb(c));
            let b = hsl_to_rgb(rgb_to_hsl(c));
            for k in 0..3 {
                assert!((a[k] - c[k]).abs() < 1e-5 && (b[k] - c[k]).abs() < 1e-5, "{c:?} {a:?} {b:?}");
            }
        }
        assert_eq!(rgb_to_hsb([1.0, 0.0, 0.0]), [0.0, 1.0, 1.0]);
        let l = rgb_to_hsl([0.0, 0.0, 1.0]);
        assert!((l[0] - 2.0 / 3.0).abs() < 1e-6 && (l[2] - 0.5).abs() < 1e-6);
    }
}
