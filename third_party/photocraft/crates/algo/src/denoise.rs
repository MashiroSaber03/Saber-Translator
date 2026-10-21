//! Reduce Noise: edge-preserving luminance smoothing, chroma smoothing guided
//! by the luminance, and optional detail sharpening.
//!
//! Smoothing uses the guided filter (He, Sun & Tang, "Guided Image
//! Filtering", ECCV 2010): O(1) per pixel whatever the radius, and edge-aware
//! because the local linear model follows the guide's variance.

use photocraft_color::ColorMode;
use photocraft_geom::Rect;

use crate::Ctx;
use crate::fxutil::{MAXC, box_blur_n, gauss_blur_n, rgba, set_rgba};
use crate::image::Image;

pub(crate) struct DenoiseSpec {
    pub strength: f32,
    pub preserve: f32,
    pub color: f32,
    pub sharpen: f32,
    pub jpeg: bool,
}

const MAX_LUMA_R: usize = 5;
const MAX_CHROMA_R: usize = 9;

/// Pixels read beyond an output tile (two box passes per guided filter, plus sharpening).
pub(crate) fn reach() -> f32 {
    (2 * MAX_LUMA_R + 2 * MAX_CHROMA_R + 6) as f32
}

/// Guided filter of `p` with guide `g` (both `w × h`), box radius `r`.
fn guided(g: &[f32], p: &[f32], w: usize, h: usize, r: usize, eps: f32) -> Vec<f32> {
    let len = w * h;
    // Interleave the four box-filtered moments to run one pass.
    let mut m = vec![0.0f32; len * 4];
    for i in 0..len {
        m[i * 4] = g[i];
        m[i * 4 + 1] = p[i];
        m[i * 4 + 2] = g[i] * g[i];
        m[i * 4 + 3] = g[i] * p[i];
    }
    box_blur_n(&mut m, w, h, 4, r);
    let mut ab = vec![0.0f32; len * 2];
    for i in 0..len {
        let (mg, mp, mgg, mgp) = (m[i * 4], m[i * 4 + 1], m[i * 4 + 2], m[i * 4 + 3]);
        let var = (mgg - mg * mg).max(0.0);
        let a = (mgp - mg * mp) / (var + eps);
        ab[i * 2] = a;
        ab[i * 2 + 1] = mp - a * mg;
    }
    box_blur_n(&mut ab, w, h, 2, r);
    (0..len).map(|i| ab[i * 2] * g[i] + ab[i * 2 + 1]).collect()
}

pub(crate) fn reduce_noise(src: &Image, out: Rect, ctx: &Ctx, spec: &DenoiseSpec) -> Vec<f32> {
    let n = src.ch;
    let win = src.rect;
    let (ww, wh) = (win.width() as usize, win.height() as usize);
    let len = ww * wh;
    let s = spec.strength.clamp(0.0, 10.0) / 10.0;
    let col = spec.color.clamp(0.0, 100.0) / 100.0;
    let sh = spec.sharpen.clamp(0.0, 100.0) / 100.0;
    if (s <= 0.0 && col <= 0.0 && sh <= 0.0 && !spec.jpeg) || len == 0 {
        return src.crop(out);
    }
    // Decompose into luma + two chroma planes.
    #[derive(PartialEq)]
    enum Model {
        Gray,
        Ycc,
        Lab,
        ViaRgb,
    }
    let model = match ctx.mode {
        ColorMode::Grayscale | ColorMode::Bitmap | ColorMode::Duotone => Model::Gray,
        ColorMode::Rgb => Model::Ycc,
        ColorMode::Lab => Model::Lab,
        _ => Model::ViaRgb,
    };
    let mut yv = vec![0.0f32; len];
    let mut c1 = vec![0.0f32; if model == Model::Gray { 0 } else { len }];
    let mut c2 = vec![0.0f32; if model == Model::Gray { 0 } else { len }];
    let mut px = [0.0f32; MAXC];
    for (i, yy) in yv.iter_mut().enumerate() {
        let (x, y) = (win.x0 + (i % ww) as i32, win.y0 + (i / ww) as i32);
        for (c, v) in px.iter_mut().enumerate().take(n) {
            *v = src.get(x, y, c);
        }
        match model {
            Model::Gray => *yy = px[0],
            Model::Lab => {
                *yy = px[0];
                c1[i] = px[1];
                c2[i] = px[2];
            }
            Model::Ycc | Model::ViaRgb => {
                let c = if model == Model::Ycc { [px[0], px[1], px[2], 1.0] } else { rgba(ctx, &px[..n]) };
                let l = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
                *yy = l;
                c1[i] = c[2] - l;
                c2[i] = c[0] - l;
            }
        }
    }
    // JPEG deblocking: soften luma across the 8×8 grid lines (document-anchored).
    if spec.jpeg {
        let orig = yv.clone();
        for yy in 1..wh.saturating_sub(1) {
            for xx in 1..ww - 1 {
                let (dx, dy) = ((win.x0 + xx as i32).rem_euclid(8), (win.y0 + yy as i32).rem_euclid(8));
                if dx == 0 || dx == 7 || dy == 0 || dy == 7 {
                    let mut a = 0.0;
                    for j in 0..3 {
                        for k in 0..3 {
                            a += orig[(yy + j - 1) * ww + xx + k - 1];
                        }
                    }
                    yv[yy * ww + xx] = 0.5 * orig[yy * ww + xx] + 0.5 * a / 9.0;
                }
            }
        }
    }
    // Luminance: edge-preserving smoothing; Preserve Details shrinks the regularisation.
    if s > 0.0 {
        let r = (1.0 + spec.strength.clamp(0.0, 10.0) * 0.4).round() as usize;
        let keep = spec.preserve.clamp(0.0, 100.0) / 100.0;
        let eps = (0.004 + 0.05 * s).powi(2) * (1.05 - keep);
        let d = guided(&yv, &yv, ww, wh, r.min(MAX_LUMA_R), eps);
        let k = s.sqrt();
        for (v, dv) in yv.iter_mut().zip(d) {
            *v += (dv - *v) * k;
        }
    }
    // Chroma: guided by the cleaned luma so colour edges stay put.
    if col > 0.0 && model != Model::Gray {
        let r = (1.0 + col * 8.0).round() as usize;
        let eps = (0.01 + 0.05 * col).powi(2);
        for plane in [&mut c1, &mut c2] {
            let d = guided(&yv, plane, ww, wh, r.min(MAX_CHROMA_R), eps);
            for (v, dv) in plane.iter_mut().zip(d) {
                *v += (dv - *v) * col;
            }
        }
    }
    // Sharpen Details: unsharp mask on luma.
    if sh > 0.0 {
        let mut bl = yv.clone();
        gauss_blur_n(&mut bl, ww, wh, 1, 1.0);
        for (v, b) in yv.iter_mut().zip(bl) {
            *v += (*v - b) * sh * 1.2;
        }
    }
    let mut res = src.crop(out);
    for (i, p) in res.chunks_exact_mut(n).enumerate() {
        let (x, y) = crate::fxutil::xy(out, i);
        let k = ((y - win.y0) as usize) * ww + (x - win.x0) as usize;
        match model {
            Model::Gray => p[0] = yv[k],
            Model::Lab => {
                p[0] = yv[k];
                p[1] = c1[k];
                p[2] = c2[k];
            }
            Model::Ycc | Model::ViaRgb => {
                let l = yv[k];
                let (b, r) = (c1[k] + l, c2[k] + l);
                let g = (l - 0.299 * r - 0.114 * b) / 0.587;
                if model == Model::Ycc {
                    p[0] = r;
                    p[1] = g;
                    p[2] = b;
                } else {
                    let a = if ctx.alpha { p[n - 1] } else { 1.0 };
                    set_rgba(ctx, p, [r, g, b, a]);
                }
            }
        }
    }
    res
}
