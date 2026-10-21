//! Shared helpers for the second filter batch: colour-model access, window
//! buffers, single-channel box/Gaussian blurs and jittered cell points.

use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_geom::Rect;
use photocraft_raster::{from_rgba_into, to_rgba};

use crate::Ctx;
use crate::image::Image;
use crate::noise::hash01;

/// Max channels per pixel we keep on the stack (CMYK + alpha + spare).
pub(crate) const MAXC: usize = 8;

/// The context's model as an `f32` format (for colour conversions).
#[inline]
pub(crate) fn fmt(ctx: &Ctx) -> PixelFormat {
    PixelFormat::new(ctx.mode, SampleType::F32, ctx.alpha)
}

/// Colour channels (excluding alpha).
#[inline]
pub(crate) fn ncol(ctx: &Ctx, n: usize) -> usize {
    if ctx.alpha { n - 1 } else { n }
}

/// Whether channel values are ink amounts (1 = dark) rather than light.
#[inline]
pub(crate) fn subtractive(ctx: &Ctx) -> bool {
    ctx.mode == ColorMode::Cmyk
}

/// Whether per-channel "brightness" operations should run on an RGB
/// conversion instead of the native channels (Lab's a/b are not intensities).
#[inline]
pub(crate) fn via_rgb(ctx: &Ctx) -> bool {
    ctx.mode == ColorMode::Lab
}

/// Document coordinates of the `i`th pixel of `out`.
#[inline]
pub(crate) fn xy(out: Rect, i: usize) -> (i32, i32) {
    let w = out.width() as usize;
    (out.x0 + (i % w) as i32, out.y0 + (i / w) as i32)
}

/// Straight sRGB RGBA of a native pixel.
#[inline]
pub(crate) fn rgba(ctx: &Ctx, px: &[f32]) -> [f32; 4] {
    to_rgba(&fmt(ctx), px)
}

/// Writes straight sRGB RGBA into a native pixel.
#[inline]
pub(crate) fn set_rgba(ctx: &Ctx, px: &mut [f32], c: [f32; 4]) {
    let mut tmp = [0.0f32; MAXC];
    let n = from_rgba_into(&fmt(ctx), c, &mut tmp);
    let m = n.min(px.len());
    px[..m].copy_from_slice(&tmp[..m]);
}

/// A straight sRGB RGBA colour in the native model.
pub(crate) fn native(ctx: &Ctx, c: [f32; 4]) -> [f32; MAXC] {
    let mut tmp = [0.0f32; MAXC];
    from_rgba_into(&fmt(ctx), c, &mut tmp);
    tmp
}

/// Rec. 601 luma of a native pixel.
#[inline]
pub(crate) fn luma(ctx: &Ctx, px: &[f32]) -> f32 {
    match ctx.mode {
        ColorMode::Grayscale | ColorMode::Bitmap | ColorMode::Duotone => px[0],
        ColorMode::Lab => px[0],
        ColorMode::Rgb => 0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2],
        _ => {
            let c = rgba(ctx, px);
            0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        }
    }
}

/// Interleaved premultiplied copy of `src` over `win` (zero outside).
pub(crate) fn premul_window(src: &Image, win: Rect, alpha: bool) -> Vec<f32> {
    let n = src.ch;
    let mut v = Vec::with_capacity(win.width() as usize * win.height() as usize * n);
    for y in win.y0..win.y1 {
        for x in win.x0..win.x1 {
            let a = if alpha { src.get(x, y, n - 1) } else { 1.0 };
            for c in 0..n {
                let s = src.get(x, y, c);
                v.push(if alpha && c < n - 1 { s * a } else { s });
            }
        }
    }
    v
}

/// Unpremultiplies one interleaved pixel in place.
#[inline]
pub(crate) fn unpremul_px(px: &mut [f32], alpha: bool) {
    if !alpha {
        return;
    }
    let n = px.len();
    let a = px[n - 1];
    for v in px.iter_mut().take(n - 1) {
        *v = if a > 1e-7 { *v / a } else { 0.0 };
    }
}

/// In-place running-sum box blur of every row of a `w × h × n` interleaved
/// buffer with radius `r` (edges clamped). `scratch` must hold `w × n`.
fn box_rows_n(buf: &mut [f32], w: usize, n: usize, r: usize, scratch: &mut Vec<f32>) {
    if r == 0 || w == 0 {
        return;
    }
    let norm = 1.0 / (2 * r + 1) as f64;
    for row in buf.chunks_exact_mut(w * n) {
        scratch.clear();
        scratch.extend_from_slice(row);
        // f64 sums: rounding must not depend on where a window starts (tile independence).
        let at = |x: isize, c: usize| scratch[(x.clamp(0, w as isize - 1) as usize) * n + c] as f64;
        for c in 0..n {
            let mut acc: f64 = (-(r as isize)..=r as isize).map(|x| at(x, c)).sum();
            for x in 0..w {
                row[x * n + c] = (acc * norm) as f32;
                acc += at(x as isize + r as isize + 1, c) - at(x as isize - r as isize, c);
            }
        }
    }
}

/// Same as [`box_rows_n`] over columns (no transpose; walks with a stride).
fn box_cols_n(buf: &mut [f32], w: usize, h: usize, n: usize, r: usize, scratch: &mut Vec<f32>) {
    if r == 0 || h == 0 {
        return;
    }
    let norm = 1.0 / (2 * r + 1) as f64;
    for x in 0..w {
        scratch.clear();
        for y in 0..h {
            scratch.extend_from_slice(&buf[(y * w + x) * n..(y * w + x + 1) * n]);
        }
        let at = |y: isize, c: usize| scratch[(y.clamp(0, h as isize - 1) as usize) * n + c] as f64;
        for c in 0..n {
            let mut acc: f64 = (-(r as isize)..=r as isize).map(|y| at(y, c)).sum();
            for y in 0..h {
                buf[(y * w + x) * n + c] = (acc * norm) as f32;
                acc += at(y as isize + r as isize + 1, c) - at(y as isize - r as isize, c);
            }
        }
    }
}

/// Box blur of an interleaved buffer (radius `r` in both directions).
pub(crate) fn box_blur_n(buf: &mut [f32], w: usize, h: usize, n: usize, r: usize) {
    let mut scratch = Vec::with_capacity(w.max(h) * n);
    box_rows_n(buf, w, n, r, &mut scratch);
    box_cols_n(buf, w, h, n, r, &mut scratch);
}

/// Odd box widths whose 3-fold convolution approximates a Gaussian of `sigma`
/// (Kovesi 2010), returned as radii.
pub(crate) fn gauss_box_radii(sigma: f32) -> [usize; 3] {
    let n = 3.0f32;
    let w_ideal = (12.0 * sigma * sigma / n + 1.0).sqrt();
    let mut wl = w_ideal.floor() as i32;
    if wl % 2 == 0 {
        wl -= 1;
    }
    let wl = wl.max(1);
    let wu = wl + 2;
    let m = ((12.0 * sigma * sigma - n * (wl * wl) as f32 - 4.0 * n * wl as f32 - 3.0 * n) / (-4.0 * wl as f32 - 4.0)).round().clamp(0.0, n) as usize;
    let mut radii = std::array::from_fn(|i| if i < m { (wl as usize - 1) / 2 } else { (wu as usize - 1) / 2 });
    // Kovesi's rounded triple degenerates below sigma ~ 0.577: `wl` floors to 1 and every
    // radius becomes 0, so the passes would be no-ops and callers with documented small
    // sigmas (Camera Raw `sharpenRadius` 0.5, colour noise, HDR Toning `radius` 1) would
    // silently change nothing (issue #706). One radius-1 pass is the smallest real blur;
    // it only engages where the ideal widths are already at their floor, leaving larger
    // sigmas exactly as the paper prescribes.
    if radii.iter().all(|&r| r == 0) {
        radii[0] = 1;
    }
    radii
}

/// Approximate Gaussian blur (three box passes) of an interleaved buffer.
/// Exact small kernels are not needed here: callers use it for smoothing
/// fields and blur levels, where cost independent of `sigma` matters more.
pub(crate) fn gauss_blur_n(buf: &mut [f32], w: usize, h: usize, n: usize, sigma: f32) {
    if sigma < 0.3 {
        return;
    }
    let mut scratch = Vec::with_capacity(w.max(h) * n);
    for r in gauss_box_radii(sigma) {
        box_rows_n(buf, w, n, r, &mut scratch);
    }
    for r in gauss_box_radii(sigma) {
        box_cols_n(buf, w, h, n, r, &mut scratch);
    }
}

/// Jittered feature point of grid cell `(cx, cy)` (cells of `cell` px anchored at `origin`).
#[inline]
pub(crate) fn cell_point(cx: i32, cy: i32, cell: f32, seed: u32, origin: (f32, f32)) -> (f32, f32) {
    (origin.0 + (cx as f32 + 0.1 + 0.8 * hash01(cx, cy, 11, seed)) * cell, origin.1 + (cy as f32 + 0.1 + 0.8 * hash01(cx, cy, 12, seed)) * cell)
}

#[inline]
pub(crate) fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    if e1 <= e0 {
        return if x >= e1 { 1.0 } else { 0.0 };
    }
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Adds the premultiplied bilinear sample of `src` at `(x, y)` (pixel centres at .5, edges
/// clamped to `clip`, which must lie inside `src.rect`) into `acc`. A fast path for blurs that
/// average many samples; unpremultiply the sum once at the end.
#[inline]
pub(crate) fn add_sample_premul(src: &Image, x: f32, y: f32, clip: Rect, alpha: bool, acc: &mut [f32]) {
    if !x.is_finite() || !y.is_finite() || clip.is_empty() {
        return;
    }
    let n = src.ch;
    let fx = (x - 0.5).clamp(clip.x0 as f32, (clip.x1 - 1) as f32);
    let fy = (y - 0.5).clamp(clip.y0 as f32, (clip.y1 - 1) as f32);
    let (x0, y0) = (fx.floor() as i32, fy.floor() as i32);
    let (x1, y1) = ((x0 + 1).min(clip.x1 - 1), (y0 + 1).min(clip.y1 - 1));
    let (ax, ay) = (fx - x0 as f32, fy - y0 as f32);
    let w = src.rect.width() as usize;
    let idx = |xx: i32, yy: i32| ((yy - src.rect.y0) as usize * w + (xx - src.rect.x0) as usize) * n;
    for (i, k) in [(idx(x0, y0), (1.0 - ax) * (1.0 - ay)), (idx(x1, y0), ax * (1.0 - ay)), (idx(x0, y1), (1.0 - ax) * ay), (idx(x1, y1), ax * ay)] {
        if k <= 0.0 {
            continue;
        }
        let px = &src.data[i..i + n];
        let a = if alpha { px[n - 1] * k } else { k };
        if alpha {
            for c in 0..n - 1 {
                acc[c] += px[c] * a;
            }
            acc[n - 1] += a;
        } else {
            for c in 0..n {
                acc[c] += px[c] * a;
            }
        }
    }
}
