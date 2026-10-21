//! Blurs: Gaussian, box, motion, radial, surface.

use photocraft_geom::Rect;
use photocraft_raster::Interrupt;

use crate::image::{Edge, Image, premultiply, unpremultiply};
use crate::photo_util::{par_map, par_rows};
use crate::{Ctx, FilterParams, RadialMethod, RadialQuality};

#[cfg(test)]
use self::motion as motion_rows;

pub(crate) mod motion_apply;
#[cfg(all(test, not(target_arch = "wasm32")))]
mod motion_bench;
mod motion_fft;
#[cfg(test)]
mod motion_fft_tests;
#[cfg(test)]
mod motion_tests;
pub(crate) mod radial;

/// Normalized Gaussian kernel with standard deviation `sigma` (radius 3σ).
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

pub(crate) fn gaussian(src: &Image, out: Rect, ctx: &Ctx, radius: f32) -> Vec<f32> {
    // Exact kernel for small radii; beyond that three box passes approximate the Gaussian with
    // cost independent of the radius (Kovesi, "Fast almost-Gaussian filtering", 2010).
    match box_widths(&FilterParams::GaussianBlur { radius }) {
        Some(boxes) => box_passes(src, out, ctx.alpha, &boxes),
        None => {
            let k = gaussian_kernel(radius);
            conv_sep(src, out, &k, &k, ctx.alpha)
        }
    }
}

/// Box widths (odd) whose `n`-fold convolution has standard deviation `sigma`.
fn boxes_for_gauss(sigma: f32, n: usize) -> Vec<usize> {
    let nf = n as f32;
    let w_ideal = (12.0 * sigma * sigma / nf + 1.0).sqrt();
    let mut wl = w_ideal.floor() as i32;
    if wl % 2 == 0 {
        wl -= 1;
    }
    let wl = wl.max(1);
    let wu = wl + 2;
    let m_ideal = (12.0 * sigma * sigma - nf * (wl * wl) as f32 - 4.0 * nf * wl as f32 - 3.0 * nf) / (-4.0 * wl as f32 - 4.0);
    let m = m_ideal.round().clamp(0.0, nf) as usize;
    (0..n).map(|i| if i < m { wl as usize } else { wu as usize }).collect()
}

/// The box widths (odd) a blur runs as running-sum passes along each axis, or `None` when it uses a
/// direct kernel (small radii, where the kernel is quicker).
pub(crate) fn box_widths(params: &FilterParams) -> Option<Vec<usize>> {
    match *params {
        FilterParams::GaussianBlur { radius } if radius > 4.0 => Some(boxes_for_gauss(radius, 3)),
        FilterParams::BoxBlur { radius } => {
            let r = radius.max(0.0).round() as usize;
            (r > 4).then(|| vec![2 * r + 1])
        }
        _ => None,
    }
}

/// Scratch buffers of one worker for [`box_blur`].
#[derive(Default)]
struct Scratch {
    line: Vec<f32>,
    next: Vec<f32>,
    acc: Vec<f32>,
}

/// One running-sum box pass of radius `r` along a line of elements of `w` floats each (a pixel's
/// channels, or a strip of a row): element `i` of `dst` is the mean of elements `i ..= i + 2r` of
/// `src`, which is `2r` elements longer.
fn box_pass(src: &[f32], dst: &mut [f32], w: usize, r: usize, acc: &mut Vec<f32>) {
    // Pixels are short elements: fixed sizes let the compiler keep the sums in registers.
    match w {
        1 => box_pass_px::<1>(src, dst, r),
        2 => box_pass_px::<2>(src, dst, r),
        3 => box_pass_px::<3>(src, dst, r),
        4 => box_pass_px::<4>(src, dst, r),
        5 => box_pass_px::<5>(src, dst, r),
        0 => {}
        _ => {
            let norm = 1.0 / (2 * r + 1) as f32;
            acc.clear();
            acc.resize(w, 0.0);
            for e in src.chunks_exact(w).take(2 * r + 1) {
                for (a, s) in acc.iter_mut().zip(e) {
                    *a += s;
                }
            }
            for (i, d) in dst.chunks_exact_mut(w).enumerate() {
                for (d, a) in d.iter_mut().zip(acc.iter()) {
                    *d = a * norm;
                }
                let (add, sub) = ((i + 2 * r + 1) * w, i * w);
                if let (Some(add), Some(sub)) = (src.get(add..add + w), src.get(sub..sub + w)) {
                    for ((a, add), sub) in acc.iter_mut().zip(add).zip(sub) {
                        *a += add - sub;
                    }
                }
            }
        }
    }
}

/// [`box_pass`] for elements of `N` floats.
fn box_pass_px<const N: usize>(src: &[f32], dst: &mut [f32], r: usize) {
    let (src, dst) = (src.as_chunks::<N>().0, dst.as_chunks_mut::<N>().0);
    let norm = 1.0 / (2 * r + 1) as f32;
    let mut acc = [0.0f32; N];
    for e in src.iter().take(2 * r + 1) {
        for c in 0..N {
            acc[c] += e[c];
        }
    }
    for (i, d) in dst.iter_mut().enumerate() {
        for c in 0..N {
            d[c] = acc[c] * norm;
        }
        if let (Some(add), Some(sub)) = (src.get(i + 2 * r + 1), src.get(i)) {
            for c in 0..N {
                acc[c] += add[c] - sub[c];
            }
        }
    }
}

/// Box passes of the given radii in turn along the line in `s.line` (`w`-float elements, as many
/// as `dst` holds plus twice the radii's sum; consumed), ending in `dst`.
fn line_passes(dst: &mut [f32], w: usize, radii: &[usize], s: &mut Scratch) {
    let Scratch { line, next, acc } = s;
    let Some(len) = dst.len().checked_div(w) else { return };
    let mut tail: usize = radii.iter().sum();
    for (k, &r) in radii.iter().enumerate() {
        tail -= r;
        if k + 1 == radii.len() {
            box_pass(line, dst, w, r, acc);
        } else {
            next.resize((len + 2 * tail) * w, 0.0);
            box_pass(line, next, w, r, acc);
            std::mem::swap(line, next);
        }
    }
}

/// Rows of the horizontal passes handed to one worker (they share its scratch buffers).
const ROWS_PER_TASK: usize = 8;
/// Width in pixels of the column strips the vertical passes work on (a divisor of the tile size).
pub(crate) const STRIP: i32 = 64;

/// Samples of a column strip of an output area.
pub(crate) type Strip = (Rect, Vec<f32>);

/// Box blurs of the given (odd) widths, one after another along each axis, of the `n`-channel
/// pixels of `src` extended beyond it by repeating its edge pixels (alpha-weighted when `alpha`).
/// `read(row, buf)` fills `buf` with the samples of a one-pixel-high rectangle inside `src`.
///
/// Running sums make the cost independent of the widths, and only `src`'s own rows and the output
/// columns are blurred: the repeated edges are never read or blurred as 2-D margins, so a huge
/// radius costs about what a small one does. Returns the straight results for `out` as column
/// strips (`None` when `ctl` was cancelled).
pub(crate) fn box_blur(
    src: Rect,
    read: &(dyn Fn(Rect, &mut Vec<f32>) + Sync),
    out: Rect,
    n: usize,
    alpha: bool,
    boxes: &[usize],
    ctl: &Interrupt,
) -> Option<Vec<Strip>> {
    let radii: Vec<usize> = boxes.iter().map(|w| w / 2).collect();
    let reach = i32::try_from(radii.iter().sum::<usize>()).unwrap_or(i32::MAX);
    let (ow, oh) = (out.width() as usize, out.height() as usize);
    let used = src.intersect(&out.inflate(reach));
    if used.is_empty() || n == 0 {
        // Nothing to read: transparent, as `Image::get` reads outside an image.
        return Some(strips(out).map(|t| (t, vec![0.0; t.width() as usize * oh * n])).collect());
    }
    // Horizontal passes over `used`'s rows only (the repeated rows above and below it would just
    // repeat their results), for the output columns. Each row is extended to the passes' reach by
    // repeating its end pixels.
    let mut h = vec![0.0f32; used.height() as usize * ow * n];
    let (left, right) = ((used.x0 - out.x0.saturating_sub(reach)) as usize, (out.x1.saturating_add(reach) - used.x1) as usize);
    par_rows(&mut h, ow * ROWS_PER_TASK, n, |task, chunk| {
        let (mut row, mut s) = (Vec::new(), Scratch::default());
        for (k, dst) in chunk.chunks_exact_mut(ow * n).enumerate() {
            if ctl.cancelled() {
                return;
            }
            let y = used.y0 + (task * ROWS_PER_TASK + k) as i32;
            read(Rect::new(used.x0, y, used.x1, y + 1), &mut row);
            premultiply(&mut row, n, alpha);
            let (Some(first), Some(last)) = (row.get(..n), row.get(row.len().saturating_sub(n)..)) else { return };
            s.line.clear();
            (0..left).for_each(|_| s.line.extend_from_slice(first));
            s.line.extend_from_slice(&row);
            (0..right).for_each(|_| s.line.extend_from_slice(last));
            line_passes(dst, n, &radii, &mut s);
        }
    });
    // Vertical passes, strip by strip, the same way down the columns.
    let (top, bottom) = ((used.y0 - out.y0.saturating_sub(reach)) as usize, (out.y1.saturating_add(reach) - used.y1) as usize);
    let (rows, stride) = (used.height() as usize, ow * n);
    let strips: Vec<Rect> = strips(out).collect();
    let done = par_map(strips.len(), |k| {
        let t = strips[k];
        if ctl.cancelled() {
            return (t, Vec::new());
        }
        let (sx, w) = ((t.x0 - out.x0) as usize * n, t.width() as usize * n);
        let row = |y: usize| &h[y * stride + sx..y * stride + sx + w];
        let mut s = Scratch::default();
        (0..top).for_each(|_| s.line.extend_from_slice(row(0)));
        (0..rows).for_each(|y| s.line.extend_from_slice(row(y)));
        (0..bottom).for_each(|_| s.line.extend_from_slice(row(rows - 1)));
        let mut res = vec![0.0f32; oh * w];
        line_passes(&mut res, w, &radii, &mut s);
        unpremultiply(&mut res, n, alpha);
        (t, res)
    });
    (!ctl.cancelled()).then_some(done)
}

/// Column strips of `out` at multiples of [`STRIP`], so each lies inside one tile column.
fn strips(out: Rect) -> impl Iterator<Item = Rect> {
    (out.x0.div_euclid(STRIP) * STRIP..out.x1)
        .step_by(STRIP as usize)
        .map(move |x| Rect::new(x.max(out.x0), out.y0, x.saturating_add(STRIP).min(out.x1), out.y1))
}
const _: () = assert!(photocraft_geom::TILE_SIZE % STRIP == 0);

/// [`box_blur`] of an image (edges repeated) into one interleaved buffer for `out`.
fn box_passes(src: &Image, out: Rect, alpha: bool, boxes: &[usize]) -> Vec<f32> {
    let n = src.ch;
    let read = |r: Rect, buf: &mut Vec<f32>| {
        buf.clear();
        buf.extend_from_slice(src.row(r));
    };
    let ow = out.width() as usize * n;
    let mut res = vec![0.0f32; ow * out.height() as usize];
    if res.is_empty() {
        return res;
    }
    for (t, data) in box_blur(src.rect, &read, out, n, alpha, boxes, &Interrupt::NONE).unwrap_or_default() {
        let (x0, w) = ((t.x0 - out.x0) as usize * n, t.width() as usize * n);
        for (dst, s) in res.chunks_exact_mut(ow).zip(data.chunks_exact(w)) {
            dst[x0..x0 + w].copy_from_slice(s);
        }
    }
    res
}

pub(crate) fn boxed(src: &Image, out: Rect, ctx: &Ctx, radius: f32) -> Vec<f32> {
    match box_widths(&FilterParams::BoxBlur { radius }) {
        Some(boxes) => box_passes(src, out, ctx.alpha, &boxes),
        None => {
            let r = radius.max(0.0).round() as usize;
            let k = vec![1.0 / (2 * r + 1) as f32; 2 * r + 1];
            conv_sep(src, out, &k, &k, ctx.alpha)
        }
    }
}

fn average_samples(src: &Image, out: Rect, ctx: &Ctx, offsets: impl Fn(f32, f32, &mut Vec<(f32, f32)>) + Sync) -> Vec<f32> {
    let n = src.ch;
    let (ow, oh) = (out.width() as usize, out.height() as usize);
    let mut res = vec![0.0; ow * oh * n];
    if ow == 0 || oh == 0 || n == 0 {
        return res;
    }
    let process_row = |y: i32, row: &mut [f32]| {
        let mut pts = Vec::new();
        let mut tmp = vec![0.0f32; n];
        let mut acc = vec![0.0f32; n];
        for x in out.x0..out.x1 {
            let (cx, cy) = (x as f32 + 0.5, y as f32 + 0.5);
            pts.clear();
            offsets(cx, cy, &mut pts);
            for a in acc.iter_mut() {
                *a = 0.0;
            }
            for &(sx, sy) in &pts {
                src.sample(sx, sy, Edge::Transparent, src.rect, ctx.alpha, &mut tmp);
                // Accumulate premultiplied.
                let a = if ctx.alpha { tmp[n - 1] } else { 1.0 };
                for c in 0..n {
                    acc[c] += if ctx.alpha && c < n - 1 { tmp[c] * a } else { tmp[c] };
                }
            }
            let k = 1.0 / pts.len().max(1) as f32;
            for (dst, value) in row[(x - out.x0) as usize * n..][..n].iter_mut().zip(&acc) {
                *dst = *value * k;
            }
            if ctx.alpha {
                let alpha = acc[n - 1] * k;
                let dst = &mut row[(x - out.x0) as usize * n..][..n];
                for value in dst.iter_mut().take(n - 1) {
                    *value = if alpha > 1e-7 { *value / alpha } else { 0.0 };
                }
            }
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        res.par_chunks_mut(ow * n).enumerate().for_each(|(row, data)| process_row(out.y0 + row as i32, data));
    }
    #[cfg(target_arch = "wasm32")]
    {
        res.chunks_mut(ow * n).enumerate().for_each(|(row, data)| process_row(out.y0 + row as i32, data));
    }
    res
}

/// Sample offsets of a motion blur of `distance` at `angle`, relative to the pixel centre.
fn motion_offsets(angle: f32, distance: f32) -> Vec<(f32, f32)> {
    let d = distance.abs();
    let (s, c) = angle.to_radians().sin_cos();
    let steps = d.ceil() as i32;
    (0..=steps)
        .map(|i| {
            let t = i as f32 / steps as f32 - 0.5;
            (c * d * t, -s * d * t)
        })
        .collect()
}

/// Motion blur: the mean of bilinear samples along a line centred on the pixel. The offsets are the
/// same for every pixel, so their bilinear taps are merged once into a fixed kernel, which is
/// applied tap by tap to whole rows of a premultiplied window (the same premultiplied mean the
/// per-sample loop takes, summed in another order).
pub(crate) fn motion(src: &Image, out: Rect, ctx: &Ctx, angle: f32, distance: f32) -> Vec<f32> {
    if distance.abs() < 0.5 {
        return src.crop(out);
    }
    let offs = motion_offsets(angle, distance);
    let k = 1.0 / offs.len() as f32;
    // Merged taps (dx, dy, weight), in the order first seen.
    let mut taps: Vec<(i32, i32, f32)> = Vec::new();
    let mut slot = std::collections::HashMap::new();
    for &(ox, oy) in &offs {
        let (x0, y0) = (ox.floor(), oy.floor());
        let (ax, ay) = (ox - x0, oy - y0);
        let (x0, y0) = (x0 as i32, y0 as i32);
        for (dx, dy, w) in [(0, 0, (1.0 - ax) * (1.0 - ay)), (1, 0, ax * (1.0 - ay)), (0, 1, (1.0 - ax) * ay), (1, 1, ax * ay)] {
            if w <= 0.0 {
                continue;
            }
            let i = *slot.entry((x0 + dx, y0 + dy)).or_insert_with(|| {
                taps.push((x0 + dx, y0 + dy, 0.0));
                taps.len() - 1
            });
            taps[i].2 += w * k;
        }
    }
    let reach = taps.iter().map(|t| t.0.abs().max(t.1.abs())).max().unwrap_or(0);
    let n = src.ch;
    let win = out.inflate(reach);
    let ww = win.width() as usize;
    // Premultiplied window; outside the source reads as transparent, as the sampler does.
    let mut pw = vec![0.0f32; ww * win.height() as usize * n];
    for y in win.y0..win.y1 {
        for x in win.x0..win.x1 {
            let i = ((y - win.y0) as usize * ww + (x - win.x0) as usize) * n;
            let a = if ctx.alpha { src.get(x, y, n - 1) } else { 1.0 };
            for c in 0..n {
                let v = src.get(x, y, c);
                pw[i + c] = if ctx.alpha && c < n - 1 { v * a } else { v };
            }
        }
    }
    let (ow, oh) = (out.width() as usize, out.height() as usize);
    let mut res = vec![0.0f32; ow * oh * n];
    let r = reach as usize;
    for (oy, row) in res.chunks_exact_mut(ow * n).enumerate() {
        for &(dx, dy, w) in &taps {
            let start = ((oy + r).wrapping_add_signed(dy as isize) * ww + r.wrapping_add_signed(dx as isize)) * n;
            for (o, v) in row.iter_mut().zip(&pw[start..start + ow * n]) {
                *o += w * v;
            }
        }
        if ctx.alpha {
            for px in row.chunks_exact_mut(n) {
                let a = px[n - 1];
                for v in px.iter_mut().take(n - 1) {
                    *v = if a > 1e-7 { *v / a } else { 0.0 };
                }
            }
        }
    }
    res
}

fn radial_intervals(path_length: f32, quality: RadialQuality) -> usize {
    let max = match quality {
        RadialQuality::Draft => 64,
        RadialQuality::Good => 256,
        RadialQuality::Best => 4096,
    };

    if path_length.is_nan() {
        return 1;
    }
    path_length.ceil().clamp(1.0, max as f32) as usize
}

pub(crate) fn radial(src: &Image, out: Rect, ctx: &Ctx, amount: f32, method: RadialMethod, quality: RadialQuality, center: (f32, f32)) -> Vec<f32> {
    let b = ctx.bounds;
    let (cx, cy) = (b.x0 as f32 + b.width() as f32 * center.0, b.y0 as f32 + b.height() as f32 * center.1);
    let amount = amount.clamp(0.0, 100.0);
    average_samples(src, out, ctx, |x, y, pts| {
        let (dx, dy) = (x - cx, y - cy);
        let r = (dx * dx + dy * dy).sqrt();
        match method {
            RadialMethod::Spin => {
                // Arc of `amount` degrees centred on the pixel.
                let arc = amount.to_radians();
                let n = radial_intervals(arc * r, quality);
                for i in 0..=n {
                    let t = (i as f32 / n as f32 - 0.5) * arc;
                    let (s, c) = t.sin_cos();
                    pts.push((cx + dx * c - dy * s, cy + dx * s + dy * c));
                }
            }
            RadialMethod::Zoom => {
                // Samples along the ray, up to amount/2 % closer to the centre.
                let span = amount / 200.0;
                let n = radial_intervals(span * r, quality);
                for i in 0..=n {
                    let k = 1.0 - span * i as f32 / n as f32;
                    pts.push((cx + dx * k, cy + dy * k));
                }
            }
        }
    })
}

/// [`surface`] for 8-bit samples (every colour sample in the window is `k / 255`; `None`
/// otherwise): a 256-bin histogram of each channel over the window slides along the row, adding
/// and removing one column per step, so a pixel costs O(radius + levels) instead of O(radius^2).
/// Each level is weighted with the same expression as the direct sum.
fn surface_8bit(src: &Image, out: Rect, ctx: &Ctx, r: i32, t: f32, reach: usize) -> Option<Vec<f32>> {
    let n = src.ch;
    let cc = if ctx.alpha { n - 1 } else { n };
    let win = out.inflate(r);
    let (ww, wh) = (win.width() as usize, win.height() as usize);
    // Level of every colour sample in the window; `SKIP` where the pixel is transparent (the
    // direct sum skips those). Outside the source reads as 0, as `Image::get` does.
    const SKIP: u16 = 256;
    let mut lv = vec![0u16; ww * wh * cc];
    for y in win.y0..win.y1 {
        for x in win.x0..win.x1 {
            let i = ((y - win.y0) as usize * ww + (x - win.x0) as usize) * cc;
            let skip = ctx.alpha && src.get(x, y, n - 1) <= 0.0;
            for c in 0..cc {
                let v = src.get(x, y, c);
                let k = (v * 255.0).round();
                if !(0.0..=255.0).contains(&k) || (k / 255.0).to_bits() != v.to_bits() {
                    return None;
                }
                lv[i + c] = if skip { SKIP } else { k as u16 };
            }
        }
    }
    let (ow, oh) = (out.width() as usize, out.height() as usize);
    let side = (2 * r + 1) as usize;
    let mut res = vec![0.0f32; ow * oh * n];
    let mut hist = [0u32; 257];
    for oy in 0..oh {
        for c in 0..cc {
            hist.fill(0);
            // Window of output column 0: window rows oy..oy + side, columns 0..side.
            let col = |hist: &mut [u32; 257], wx: usize, add: bool| {
                for wy in oy..oy + side {
                    let k = lv[(wy * ww + wx) * cc + c] as usize;
                    if add {
                        hist[k] += 1;
                    } else {
                        hist[k] -= 1;
                    }
                }
            };
            for wx in 0..side {
                col(&mut hist, wx, true);
            }
            for ox in 0..ow {
                let p = src.px(out.x0 + ox as i32, out.y0 + oy as i32);
                let v0 = p[c];
                let k0 = (v0 * 255.0).round() as usize;
                let (mut acc, mut wsum) = (0.0f64, 0.0f64);
                for (k, &cnt) in hist.iter().enumerate().take(256.min(k0 + reach + 1)).skip(k0.saturating_sub(reach)) {
                    if cnt == 0 {
                        continue;
                    }
                    let v = k as f32 / 255.0;
                    let w = (1.0 - (v - v0).abs() / t).max(0.0);
                    acc += f64::from(v * w) * f64::from(cnt);
                    wsum += f64::from(w) * f64::from(cnt);
                }
                res[(oy * ow + ox) * n + c] = if wsum > 0.0 { (acc / wsum) as f32 } else { v0 };
                if ox + 1 < ow {
                    col(&mut hist, ox, false);
                    col(&mut hist, ox + side, true);
                }
            }
        }
        if ctx.alpha {
            for ox in 0..ow {
                res[(oy * ow + ox) * n + n - 1] = src.px(out.x0 + ox as i32, out.y0 + oy as i32)[n - 1];
            }
        }
    }
    Some(res)
}

pub(crate) fn surface(src: &Image, out: Rect, ctx: &Ctx, radius: f32, threshold: f32) -> Vec<f32> {
    let r = radius.max(0.0).round() as i32;
    let t = (threshold.max(1.0) / 255.0) * 2.5;
    // Levels within the threshold of the centre (the only ones with any weight), on each side.
    let reach = ((t * 255.0).ceil() as usize).min(255);
    let side = (2 * r + 1) as usize;
    if side * side > 2 * reach + 1 + 4 * side
        && let Some(res) = surface_8bit(src, out, ctx, r, t, reach)
    {
        return res;
    }
    // 16-bit and float (and 8-bit where the histogram declined): the ranked sums overtake the
    // direct sum near radius 8 on a 256 px tile and are about 10x faster at radius 100.
    if r >= RANKED_MIN_RADIUS
        && let Some(res) = surface_ranked(src, out, ctx, r, t)
    {
        return res;
    }
    surface_direct(src, out, ctx, r, t)
}

/// Radius from which [`surface_ranked`] beats [`surface_direct`] (measured by `surface_timing`).
const RANKED_MIN_RADIUS: i32 = 10;

/// Fenwick tree over value ranks, each node holding (count, Σv, Σv²) of its range.
struct RankSums(Vec<[f64; 3]>);

impl RankSums {
    fn add(&mut self, rank: usize, sign: f64, v: f64) {
        let mut i = rank + 1;
        while let Some(node) = self.0.get_mut(i) {
            node[0] += sign;
            node[1] += sign * v;
            node[2] += sign * v * v;
            i += i & i.wrapping_neg();
        }
    }

    /// Sums over the ranks `0..end`.
    fn prefix(&self, end: usize) -> [f64; 3] {
        let mut acc = [0.0; 3];
        let mut i = end.min(self.0.len().saturating_sub(1));
        while i > 0 {
            if let Some(node) = self.0.get(i) {
                acc[0] += node[0];
                acc[1] += node[1];
                acc[2] += node[2];
            }
            i -= i & i.wrapping_neg();
        }
        acc
    }

    /// Sums over the ranks `start..end`.
    fn range(&self, start: usize, end: usize) -> [f64; 3] {
        let (a, b) = (self.prefix(start), self.prefix(end));
        [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
    }
}

/// [`surface`] at any bit depth. The weight `1 - |v - v0| / t` is linear in `v` on each side of
/// the centre value, so over the samples within the threshold below it `Σw = ((t - v0)·n + Σv) / t`
/// and `Σv·w = ((t - v0)·Σv + Σv²) / t`, and symmetrically above it. A Fenwick tree over the
/// window's distinct values holds the count, Σv and Σv² and slides along the row (one column out
/// and one in per step), so a pixel costs O(radius · log n) instead of O(radius^2). `None` when a
/// sample isn't finite; the direct sum handles those.
fn surface_ranked(src: &Image, out: Rect, ctx: &Ctx, r: i32, t: f32) -> Option<Vec<f32>> {
    let n = src.ch;
    let cc = if ctx.alpha { n - 1 } else { n };
    let win = out.inflate(r);
    let (ww, wh) = (win.width() as usize, win.height() as usize);
    // Every colour sample of the window (outside the source reads as 0, as `Image::get` does),
    // and whether its pixel counts (the direct sum skips transparent ones).
    let mut vals = Vec::with_capacity(ww * wh * cc);
    let mut opaque = Vec::with_capacity(ww * wh);
    for y in win.y0..win.y1 {
        for x in win.x0..win.x1 {
            opaque.push(!(ctx.alpha && src.get(x, y, n - 1) <= 0.0));
            for c in 0..cc {
                let v = src.get(x, y, c);
                if !v.is_finite() {
                    return None;
                }
                // `+ 0.0` turns -0.0 into 0.0, so equal values share a rank.
                vals.push(v + 0.0);
            }
        }
    }
    let mut levels = vals.clone();
    levels.sort_unstable_by(f32::total_cmp);
    levels.dedup();
    let rank: Vec<usize> = vals.iter().map(|v| levels.partition_point(|l| l < v)).collect();
    let mut tree = RankSums(vec![[0.0; 3]; levels.len() + 1]);
    let (ow, oh) = (out.width() as usize, out.height() as usize);
    if ow == 0 || oh == 0 {
        return Some(Vec::new());
    }
    let side = (2 * r + 1) as usize;
    let t64 = f64::from(t);
    let mut res = vec![0.0f32; ow * oh * n];
    for oy in 0..oh {
        for c in 0..cc {
            let column = |tree: &mut RankSums, wx: usize, sign: f64| {
                for wy in oy..oy + side {
                    let p = wy * ww + wx;
                    if opaque.get(p) == Some(&true)
                        && let (Some(&k), Some(&v)) = (rank.get(p * cc + c), vals.get(p * cc + c))
                    {
                        tree.add(k, sign, f64::from(v));
                    }
                }
            };
            for wx in 0..side {
                column(&mut tree, wx, 1.0);
            }
            for ox in 0..ow {
                let v0 = src.px(out.x0 + ox as i32, out.y0 + oy as i32)[c];
                // Ranks strictly within the threshold below v0 (and v0 itself), then above it.
                let lo = levels.partition_point(|&l| 1.0 - (l - v0).abs() / t <= 0.0 && l < v0);
                let mid = levels.partition_point(|&l| l <= v0);
                let hi = levels.partition_point(|&l| l <= v0 || 1.0 - (l - v0).abs() / t > 0.0);
                let (below, above) = (tree.range(lo, mid), tree.range(mid, hi));
                let v0d = f64::from(v0);
                let wsum = ((t64 - v0d) * below[0] + below[1] + (t64 + v0d) * above[0] - above[1]) / t64;
                let acc = ((t64 - v0d) * below[1] + below[2] + (t64 + v0d) * above[1] - above[2]) / t64;
                res[(oy * ow + ox) * n + c] = if below[0] + above[0] > 0.0 && wsum > 0.0 { (acc / wsum) as f32 } else { v0 };
                if ox + 1 < ow {
                    column(&mut tree, ox, -1.0);
                    column(&mut tree, ox + side, 1.0);
                }
            }
            // Empty the tree for the next row and channel.
            for wx in ow - 1..ow - 1 + side {
                column(&mut tree, wx, -1.0);
            }
        }
        if ctx.alpha {
            for ox in 0..ow {
                res[(oy * ow + ox) * n + n - 1] = src.px(out.x0 + ox as i32, out.y0 + oy as i32)[n - 1];
            }
        }
    }
    Some(res)
}

/// [`surface`] by summing the whole (2r + 1)^2 window of every pixel.
fn surface_direct(src: &Image, out: Rect, ctx: &Ctx, r: i32, t: f32) -> Vec<f32> {
    let n = src.ch;
    let cc = if ctx.alpha { n - 1 } else { n };
    let mut res = Vec::with_capacity(out.width() as usize * out.height() as usize * n);
    for y in out.y0..out.y1 {
        for x in out.x0..out.x1 {
            let p = src.px(x, y);
            for (c, &v0) in p.iter().enumerate().take(cc) {
                let (mut acc, mut wsum) = (0.0, 0.0);
                for yy in y - r..=y + r {
                    for xx in x - r..=x + r {
                        if ctx.alpha && src.get(xx, yy, n - 1) <= 0.0 {
                            continue;
                        }
                        let v = src.get(xx, yy, c);
                        let w = (1.0 - (v - v0).abs() / t).max(0.0);
                        acc += v * w;
                        wsum += w;
                    }
                }
                res.push(if wsum > 0.0 { acc / wsum } else { v0 });
            }
            if ctx.alpha {
                res.push(p[n - 1]);
            }
        }
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::{ColorMode, PixelFormat, SampleType};
    use photocraft_raster::Surface;

    #[test]
    fn radial_blur_samples_large_arcs_beyond_64_intervals() {
        let bounds = Rect::new(0, 0, 1000, 1000);
        let out = Rect::new(999, 500, 1000, 501);
        let ctx = Ctx { bounds, mode: photocraft_color::ColorMode::Grayscale, alpha: false };
        let mut img = Image::new(bounds, 1);
        let (cx, cy) = (500.0f32, 500.0f32);
        let (x, y) = (999.5f32, 500.5f32);
        let (dx, dy) = (x - cx, y - cy);
        let arc = 100.0f32.to_radians();
        let count = radial_intervals(arc * dx.hypot(dy), RadialQuality::Good);
        assert_eq!(count, 256);
        assert_eq!(radial_intervals(f32::INFINITY, RadialQuality::Draft), 64);
        assert_eq!(radial_intervals(f32::INFINITY, RadialQuality::Good), 256);
        assert_eq!(radial_intervals(f32::INFINITY, RadialQuality::Best), 4096);
        assert_eq!(radial_intervals(f32::NAN, RadialQuality::Best), 1);

        // Put a small bright patch halfway between two samples from the former 64-interval
        // limit. Dense sampling should pick it up; the coarse path misses it entirely.
        let sparse_midpoint = (31.5 / 64.0 - 0.5) * arc;
        let (s, c) = sparse_midpoint.sin_cos();
        let px = (cx + dx * c - dy * s - 0.5).round() as i32;
        let py = (cy + dx * s + dy * c - 0.5).round() as i32;
        for yy in py - 1..=py + 1 {
            for xx in px - 1..=px + 1 {
                let index = (yy - bounds.y0) as usize * bounds.width() as usize + (xx - bounds.x0) as usize;
                img.data[index] = 1.0;
            }
        }

        let sample_path = |x: f32, y: f32, pts: &mut Vec<(f32, f32)>, intervals: usize| {
            let (dx, dy) = (x - cx, y - cy);
            for i in 0..=intervals {
                let t = (i as f32 / intervals as f32 - 0.5) * arc;
                let (s, c) = t.sin_cos();
                pts.push((cx + dx * c - dy * s, cy + dx * s + dy * c));
            }
        };
        let actual = radial(&img, out, &ctx, 100.0, RadialMethod::Spin, RadialQuality::Good, (0.5, 0.5))[0];
        let expected = average_samples(&img, out, &ctx, |x, y, pts| sample_path(x, y, pts, 8192))[0];
        let coarse = average_samples(&img, out, &ctx, |x, y, pts| sample_path(x, y, pts, 64))[0];

        assert!(actual > 0.001, "dense radial samples should resolve the bright patch, got {actual}");
        assert!((actual - expected).abs() < 0.0015, "radial result {actual} differs from dense reference {expected}");
        assert!(coarse < expected * 0.1, "former 64-interval sampling unexpectedly resolved the patch: {coarse} vs {expected}");
    }

    #[test]
    fn fixed_channel_convolution_matches_scalar_tap_order() {
        fn check<const N: usize>() {
            for width in [1, 7, 256] {
                for radius in [0.0, 0.5, 3.0, 4.0] {
                    let k = gaussian_kernel(radius);
                    let src: Vec<f32> = (0..(width + k.len() - 1) * N).map(|i| ((i * 73) % 251) as f32 / 250.0).collect();
                    let mut expected = vec![0.0f32; width * N];
                    for x in 0..width {
                        for (i, w) in k.iter().enumerate() {
                            for c in 0..N {
                                expected[x * N + c] += src[(x + i) * N + c] * w;
                            }
                        }
                    }
                    let mut actual = vec![0.0f32; width * N];
                    conv_row::<N>(&src, &mut actual, &k);
                    assert_eq!(actual, expected, "channels={N} width={width} radius={radius}");
                }
            }
        }
        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
        check::<5>();
    }

    #[test]
    fn box_gaussian_matches_exact_kernel() {
        // A hard vertical edge (premultiplied RGBA-ish, 2 channels) blurred with σ = 12.
        let src_rect = Rect::new(-60, -60, 180, 68);
        let mut img = Image::new(src_rect, 2);
        for y in src_rect.y0..src_rect.y1 {
            for x in src_rect.x0..src_rect.x1 {
                let i = ((y - src_rect.y0) as usize * src_rect.width() as usize + (x - src_rect.x0) as usize) * 2;
                img.data[i] = if x < 60 { 1.0 } else { 0.0 };
                img.data[i + 1] = 1.0;
            }
        }
        let out = Rect::new(0, 0, 120, 8);
        let k = gaussian_kernel(12.0);
        let exact = conv_sep(&img, out, &k, &k, true);
        let fast = box_passes(&img, out, true, &boxes_for_gauss(12.0, 3));
        let err = exact.iter().zip(&fast).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(err < 0.02, "max error {err}");
        assert_eq!(boxes_for_gauss(10.0, 3).len(), 3);
    }

    #[test]
    fn box_passes_match_the_direct_box_kernel() {
        // Noisy RGBA with varying alpha, so premultiplication and every channel are exercised.
        for r in [5usize, 17, 60] {
            let (w, h, m) = (40, 24, r as i32 + 2);
            let src_rect = Rect::new(-m, -m, w + m, h + m);
            let mut img = Image::new(src_rect, 4);
            let mut s = 0x9e37_79b9u32;
            for v in img.data.iter_mut() {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                *v = (s % 256) as f32 / 255.0;
            }
            let out = Rect::new(0, 0, w, h);
            let k = vec![1.0 / (2 * r + 1) as f32; 2 * r + 1];
            let direct = conv_sep(&img, out, &k, &k, true);
            let ctx = Ctx { bounds: src_rect, mode: crate::ColorMode::Rgb, alpha: true };
            let fast = boxed(&img, out, &ctx, r as f32);
            let err = direct.iter().zip(&fast).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(err < 1e-5, "radius {r}: max error {err}");
        }
    }

    /// The definition the running sums must match: each box applied in turn as a direct
    /// convolution (cost per pixel proportional to the radius) to `img`, which covers `out` grown
    /// by every box's reach (its own edge handling included).
    fn direct_boxes(img: &Image, out: Rect, boxes: &[usize], alpha: bool) -> Vec<f32> {
        let mut cur = img.clone();
        let mut left: i32 = boxes.iter().map(|w| (w / 2) as i32).sum();
        for &w in boxes {
            left -= (w / 2) as i32;
            let k = vec![1.0 / w as f32; w];
            let rect = out.inflate(left);
            cur = Image { rect, ch: img.ch, data: conv_sep(&cur, rect, &k, &k, alpha) };
        }
        cur.data
    }

    /// Noisy pixels over `r` (alpha varying, some fully transparent).
    fn noisy(fmt: PixelFormat, r: Rect, seed: u32) -> Surface {
        let mut s = Surface::new(fmt);
        let mut st = seed;
        let mut v = Vec::new();
        for _ in 0..r.width() * r.height() {
            for c in 0..fmt.channels() {
                st ^= st << 13;
                st ^= st >> 17;
                st ^= st << 5;
                let x = (st % 256) as f32 / 255.0;
                v.push(if fmt.alpha && c + 1 == fmt.channels() && st.is_multiple_of(5) { 0.0 } else { x });
            }
        }
        s.write_region(r, &v);
        s
    }

    #[test]
    fn layer_box_blurs_match_direct_convolution_at_any_radius() {
        // Gaussian and Box Blur at radii up to several times the layer (37 × 23), filtered as a
        // document layer (canvas edge repeated, as Photoshop does) and on their own (transparent
        // beyond), at every depth, with and without alpha, 1 to 5 channels.
        let canvas = Rect::new(0, 0, 37, 23);
        let formats = [
            PixelFormat::RGBA8,
            PixelFormat::RGBA16,
            PixelFormat::RGBA32F,
            PixelFormat::GRAY8,
            PixelFormat::CMYKA8,
            PixelFormat::new(ColorMode::Lab, SampleType::F32, false),
        ];
        let params = [
            FilterParams::GaussianBlur { radius: 4.5 },
            FilterParams::GaussianBlur { radius: 9.3 },
            FilterParams::GaussianBlur { radius: 40.0 },
            FilterParams::BoxBlur { radius: 5.0 },
            FilterParams::BoxBlur { radius: 70.0 },
        ];
        for (i, fmt) in formats.into_iter().enumerate() {
            let s = noisy(fmt, canvas, 0x9e37_79b9 + i as u32);
            // A level of the depth (colour and alpha are each rounded on write) plus float noise.
            let tol = match fmt.sample {
                SampleType::U8 => 1.0 / 255.0,
                SampleType::U16 => 1.0 / 65535.0,
                _ => 0.0,
            } + 2e-5;
            for p in &params {
                let boxes = box_widths(p).expect("a box-pass blur");
                let reach: i32 = boxes.iter().map(|w| (w / 2) as i32).sum();
                let area = crate::output_area(p, s.content_bounds(), canvas, None);
                for (got, img, out) in [
                    (crate::apply_in(&s, p, area, canvas, None, canvas), Image::read_clamped(&s, canvas.inflate(reach), canvas), canvas),
                    (crate::apply(&s, p, area, canvas, None), Image::read(&s, area.inflate(reach)), area),
                ] {
                    // Compared alpha-weighted: colour is meaningless where (nearly) transparent.
                    let (mut got, mut want) = (got.read_region(out), direct_boxes(&img, out, &boxes, fmt.alpha));
                    premultiply(&mut got, img.ch, fmt.alpha);
                    premultiply(&mut want, img.ch, fmt.alpha);
                    let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                    assert!(err <= tol, "{} on {fmt:?} over {out:?}: max error {err}", p.label());
                }
            }
        }
    }

    #[test]
    fn layer_box_blurs_mix_by_the_selection_and_stop_when_cancelled() {
        let canvas = Rect::new(0, 0, 37, 23);
        let s = noisy(PixelFormat::RGBA32F, canvas, 7);
        let p = FilterParams::GaussianBlur { radius: 12.0 };
        let full = crate::apply_in(&s, &p, canvas, canvas, None, canvas);
        let half = Surface::with_default(PixelFormat::new(ColorMode::Grayscale, SampleType::F32, false), &[0.5]);
        let mixed = crate::apply_in(&s, &p, canvas, canvas, Some(&half), canvas);
        for (x, y) in [(0, 0), (18, 11), (36, 22)] {
            let (o, f, m) = (s.pixel(x, y), full.pixel(x, y), mixed.pixel(x, y));
            assert!((m[3] - (o[3] + f[3]) / 2.0).abs() < 1e-5, "({x},{y}) alpha");
            for c in 0..3 {
                assert!((m[c] * m[3] - (o[c] * o[3] + f[c] * f[3]) / 2.0).abs() < 1e-5, "({x},{y}) premultiplied channel {c}");
            }
        }
        let cancel = || true;
        let ctl = Interrupt::cancel_only(&cancel);
        assert!(crate::apply_in_with(&s, &p, canvas, canvas, None, canvas, &ctl).is_none());
    }

    #[test]
    fn surface_blur_8bit_histogram_matches_the_direct_sum() {
        // 8-bit levels with some fully transparent pixels, which the sum skips.
        let src_rect = Rect::new(-14, -14, 40, 34);
        let mut img = Image::new(src_rect, 4);
        let mut s = 0x6c07_8965u32;
        for px in img.data.as_chunks_mut::<4>().0 {
            for v in px.iter_mut() {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                *v = (s % 256) as f32 / 255.0;
            }
            if s.is_multiple_of(7) {
                px[3] = 0.0;
            }
        }
        let ctx = Ctx { bounds: src_rect, mode: crate::ColorMode::Rgb, alpha: true };
        // The output reaches the source edge, where samples read as transparent.
        let out = Rect::new(-2, 0, 26, 20);
        for (r, threshold) in [(1, 2.0), (4, 15.0), (12, 60.0), (12, 255.0)] {
            let t = (threshold / 255.0) * 2.5;
            let reach = ((t * 255.0f32).ceil() as usize).min(255);
            let fast = surface_8bit(&img, out, &ctx, r, t, reach).expect("8-bit input");
            let direct = surface_direct(&img, out, &ctx, r, t);
            let err = direct.iter().zip(&fast).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(err < 1e-5, "r {r} threshold {threshold}: max error {err}");
        }
        // Not 8-bit: the histogram declines and the direct sum is used.
        // (0, 0) is inside every window above.
        let i = (14 * src_rect.width() as usize + 14) * 4;
        img.data[i] = 0.5 / 255.0;
        assert!(surface_8bit(&img, out, &ctx, 4, 0.1, 26).is_none());
    }

    #[test]
    fn motion_kernel_matches_the_per_pixel_samples() {
        // Noisy RGBA with some transparent pixels; the output reaches the source edge.
        let src_rect = Rect::new(-30, -30, 50, 40);
        let mut img = Image::new(src_rect, 4);
        let mut s = 0x1b87_3593u32;
        for px in img.data.as_chunks_mut::<4>().0 {
            for v in px.iter_mut() {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                *v = (s % 1000) as f32 / 999.0;
            }
            if s.is_multiple_of(5) {
                px[3] = 0.0;
            }
        }
        let out = Rect::new(-12, -6, 30, 22);
        for alpha in [true, false] {
            let ctx = Ctx { bounds: src_rect, mode: crate::ColorMode::Rgb, alpha };
            for (angle, distance) in [(0.0, 1.0), (0.0, 9.0), (30.0, 7.0), (90.0, 12.0), (-45.0, 20.0), (137.0, 33.5), (200.0, 40.0)] {
                let fast = motion(&img, out, &ctx, angle, distance);
                let offs = motion_offsets(angle, distance);
                let slow = average_samples(&img, out, &ctx, |x, y, pts| pts.extend(offs.iter().map(|&(dx, dy)| (x + dx, y + dy))));
                let err = slow.iter().zip(&fast).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                assert!(err < 1e-5, "alpha {alpha} angle {angle} distance {distance}: max error {err}");
            }
        }
    }

    /// The direct sum with f64 accumulators: the reference both fast paths are held to.
    fn surface_reference(src: &Image, out: Rect, ctx: &Ctx, r: i32, t: f32) -> Vec<f32> {
        let n = src.ch;
        let cc = if ctx.alpha { n - 1 } else { n };
        let mut res = Vec::new();
        for y in out.y0..out.y1 {
            for x in out.x0..out.x1 {
                let p = src.px(x, y);
                for &v0 in p.iter().take(cc) {
                    let c = res.len() % n;
                    let (mut acc, mut wsum) = (0.0f64, 0.0f64);
                    for yy in y - r..=y + r {
                        for xx in x - r..=x + r {
                            if ctx.alpha && src.get(xx, yy, n - 1) <= 0.0 {
                                continue;
                            }
                            let v = src.get(xx, yy, c);
                            let w = (1.0 - (v - v0).abs() / t).max(0.0);
                            acc += f64::from(v) * f64::from(w);
                            wsum += f64::from(w);
                        }
                    }
                    res.push(if wsum > 0.0 { (acc / wsum) as f32 } else { v0 });
                }
                if ctx.alpha {
                    res.push(p[n - 1]);
                }
            }
        }
        res
    }

    #[test]
    fn surface_blur_ranked_sums_match_the_weighted_window_at_any_depth() {
        // 16-bit levels and arbitrary floats (including negatives and values above 1), with some
        // fully transparent pixels, which the sum skips.
        for (quant, channels, alpha) in [(Some(65535.0f32), 4, true), (None, 4, true), (None, 1, false)] {
            let src_rect = Rect::new(-20, -20, 48, 40);
            let mut img = Image::new(src_rect, channels);
            let mut s = 0x1b87_3593u32;
            for px in img.data.chunks_mut(channels) {
                for v in px.iter_mut() {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    let raw = (s % 10_000) as f32 / 10_000.0;
                    *v = quant.map_or(raw * 1.5 - 0.25, |q| (raw * q).round() / q);
                }
                if alpha && s.is_multiple_of(7) {
                    px[channels - 1] = 0.0;
                }
            }
            let ctx = Ctx { bounds: src_rect, mode: crate::ColorMode::Rgb, alpha };
            // The output reaches the source edge, where samples read as 0.
            let out = Rect::new(-6, 0, 30, 22);
            for (r, threshold) in [(1, 2.0), (4, 15.0), (12, 60.0), (16, 255.0)] {
                let t = (threshold / 255.0) * 2.5;
                let ranked = surface_ranked(&img, out, &ctx, r, t).expect("finite input");
                let reference = surface_reference(&img, out, &ctx, r, t);
                assert_eq!(ranked.len(), reference.len());
                let err = reference.iter().zip(&ranked).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                assert!(err < 1e-6, "{quant:?} r {r} threshold {threshold}: max error {err}");
            }
        }
    }

    #[test]
    fn surface_blur_ranked_declines_non_finite_samples_and_handles_empty_output() {
        let src_rect = Rect::new(0, 0, 8, 8);
        let mut img = Image::new(src_rect, 1);
        img.data.fill(0.5);
        let ctx = Ctx { bounds: src_rect, mode: crate::ColorMode::Grayscale, alpha: false };
        assert_eq!(surface_ranked(&img, Rect::new(2, 2, 2, 6), &ctx, 2, 0.1), Some(Vec::new()));
        img.data[9] = f32::NAN;
        assert!(surface_ranked(&img, Rect::new(0, 0, 8, 8), &ctx, 2, 0.1).is_none());
        // The public entry still answers, through the direct sum.
        assert_eq!(surface(&img, Rect::new(0, 0, 8, 8), &ctx, 12.0, 15.0).len(), 64);
    }

    /// Release-mode timings of the direct sum and the ranked path on one 256 px tile of
    /// 16-bit-like and float noise (`cargo test -p photocraft-algo --release -- --ignored --nocapture surface_timing`).
    #[test]
    #[ignore]
    fn surface_timing() {
        for (label, quant) in [("16-bit", Some(65535.0f32)), ("float", None)] {
            for r in [2, 4, 8, 16, 32, 64, 100] {
                let out = Rect::new(0, 0, 256, 256);
                let src_rect = out.inflate(r);
                let mut img = Image::new(src_rect, 4);
                let mut s = 0x2545_f491u32;
                let len = img.data.len();
                for (i, v) in img.data.iter_mut().enumerate() {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    // Smooth gradient plus noise, so the threshold keeps a realistic share.
                    let x = (i / 4) as f32 / len as f32 * 4.0;
                    let raw = (0.5 + 0.4 * (x * 6.0).sin() + (s % 1000) as f32 / 20000.0).clamp(0.0, 1.0);
                    *v = quant.map_or(raw, |q| (raw * q).round() / q);
                }
                let ctx = Ctx { bounds: src_rect, mode: crate::ColorMode::Rgb, alpha: true };
                let t = (15.0 / 255.0) * 2.5;
                let a = std::time::Instant::now();
                let direct = surface_direct(&img, out, &ctx, r, t);
                let td = a.elapsed();
                let a = std::time::Instant::now();
                let ranked = surface_ranked(&img, out, &ctx, r, t).expect("finite");
                let tr = a.elapsed();
                let err = direct.iter().zip(&ranked).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                println!("{label} r {r:3}: direct {td:>10.2?}  ranked {tr:>10.2?}  max err {err:.2e}");
            }
        }
    }
}
