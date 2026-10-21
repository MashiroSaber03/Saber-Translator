//! Frozen exact Unsharp Mask from a28d7866, independent of production helpers.
use crate::Ctx;
use crate::image::{Image, premultiply, unpremultiply};
use photocraft_geom::Rect;
pub(crate) fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    if sigma < 0.05 {
        return vec![1.0];
    }
    let r = (sigma * 3.0).ceil() as i32;
    let k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = k.iter().sum();
    k.into_iter().map(|v| v / s).collect()
}

/// Separable convolution of `src` (premultiplied internally) for `out`.
/// `src` must cover `out` grown by the kernel radii.
pub(crate) fn conv_sep(src: &Image, out: Rect, kx: &[f32], ky: &[f32], alpha: bool) -> Vec<f32> {
    let n = src.ch;
    let (rx, ry) = ((kx.len() / 2) as i32, (ky.len() / 2) as i32);
    let (ow, oh) = (out.width() as usize, out.height() as usize);
    let th = oh + 2 * ry as usize;
    // Premultiplied copy of the needed source window.
    let win = Rect::new(out.x0 - rx, out.y0 - ry, out.x1 + rx, out.y1 + ry);
    let ww = win.width() as usize;
    let mut p = vec![0.0f32; ww * win.height() as usize * n];
    for y in win.y0..win.y1 {
        for x in win.x0..win.x1 {
            let o = ((y - win.y0) as usize * ww + (x - win.x0) as usize) * n;
            for c in 0..n {
                p[o + c] = src.get(x, y, c);
            }
        }
    }
    premultiply(&mut p, n, alpha);
    // Horizontal pass: (ow × th).
    let mut tmp = vec![0.0f32; ow * th * n];
    for ty in 0..th {
        let row = &p[ty * ww * n..(ty + 1) * ww * n];
        let dst = &mut tmp[ty * ow * n..(ty + 1) * ow * n];
        match n {
            1 => conv_row::<1>(row, dst, kx),
            2 => conv_row::<2>(row, dst, kx),
            3 => conv_row::<3>(row, dst, kx),
            4 => conv_row::<4>(row, dst, kx),
            5 => conv_row::<5>(row, dst, kx),
            _ => {
                for ox in 0..ow {
                    let d = &mut dst[ox * n..(ox + 1) * n];
                    for (i, kv) in kx.iter().enumerate() {
                        let s = &row[(ox + i) * n..(ox + i + 1) * n];
                        for c in 0..n {
                            d[c] += s[c] * kv;
                        }
                    }
                }
            }
        }
    }
    // Vertical pass.
    let mut res = vec![0.0f32; ow * oh * n];
    for oy in 0..oh {
        let d = &mut res[oy * ow * n..(oy + 1) * ow * n];
        for (i, kv) in ky.iter().enumerate() {
            let s = &tmp[(oy + i) * ow * n..(oy + i + 1) * ow * n];
            for (dv, sv) in d.iter_mut().zip(s) {
                *dv += sv * kv;
            }
        }
    }
    unpremultiply(&mut res, n, alpha);
    res
}

// Fixed channel counts let LLVM keep one pixel's accumulators in registers and vectorize
// channels. The tap order is unchanged; unusual multichannel images keep the generic path.
fn conv_row<const N: usize>(src: &[f32], dst: &mut [f32], kernel: &[f32]) {
    let pixels = src.as_chunks::<N>().0;
    for (x, out) in dst.as_chunks_mut::<N>().0.iter_mut().enumerate() {
        let mut acc = [0.0f32; N];
        for (pixel, weight) in pixels.get(x..).unwrap_or(&[]).iter().zip(kernel) {
            for c in 0..N {
                acc[c] += pixel[c] * weight;
            }
        }
        *out = acc;
    }
}

fn with_blur(src: &Image, out: Rect, ctx: &Ctx, radius: f32, mut f: impl FnMut(f32, f32) -> f32) -> Vec<f32> {
    let k = gaussian_kernel(radius);
    let blurred = conv_sep(src, out, &k, &k, ctx.alpha);
    let mut res = src.crop(out);
    let n = src.ch;
    let cc = if ctx.alpha { n - 1 } else { n };
    for (px, bl) in res.chunks_exact_mut(n).zip(blurred.chunks_exact(n)) {
        if ctx.alpha && px[n - 1] <= 0.0 {
            continue;
        }
        for c in 0..cc {
            px[c] = f(px[c], bl[c]);
        }
    }
    res
}

pub(crate) fn unsharp(src: &Image, out: Rect, ctx: &Ctx, amount: f32, radius: f32, threshold: f32) -> Vec<f32> {
    let k = amount / 100.0;
    let t = threshold / 255.0;
    with_blur(src, out, ctx, radius, |o, b| {
        let d = o - b;
        if d.abs() + 1e-6 >= t { o + k * d } else { o }
    })
}
