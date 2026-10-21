//! High-quality separable resampling of surfaces (Image Size).

use photocraft_geom::{Rect, TILE_SIZE};
use photocraft_raster::Surface;
use serde::{Deserialize, Serialize};

use crate::image::{premultiply, unpremultiply};

/// Resampling method (Photoshop's Image Size options, and Lanczos).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum Resample {
    Nearest,
    Bilinear,
    #[default]
    Bicubic,
    /// Photoshop's Bicubic Smoother, for enlargement.
    BicubicSmoother,
    /// Photoshop's Bicubic Sharper, for reduction.
    BicubicSharper,
    /// Photoshop's Bicubic Automatic: Smoother where the image grows, Sharper where it shrinks.
    BicubicAutomatic,
    Lanczos,
    /// Lanczos followed by a light sharpening when enlarging; Bicubic Sharper when reducing, as
    /// Photoshop's Preserve Details reduces.
    PreserveDetails,
}

/// Sub-pixel positions Photoshop's filters resolve: a sample's offset is rounded to 1/128 pixel.
const PHASES: f64 = 128.0;

/// The filter of one axis.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kernel {
    Nearest,
    /// Linear interpolation, never widened. Below half size, each output pixel is the plain
    /// average of the source pixels it covers.
    Bilinear,
    /// The Keys cubic with parameter `a`, stretched over `stretch` source pixels.
    Keys {
        a: f64,
        stretch: f64,
    },
    /// Lanczos-3, widened by 1/scale when reducing.
    Lanczos,
}

/// Photoshop's bicubic filters, measured on Photoshop 25.4 (the Keys cubic, phase table and
/// stretch fitted to within 1/4000 of every weight). `s` is the scale the filter is designed for.
///
/// - Bicubic: a = −0.75.
/// - Bicubic Smoother: stretched by 1.15; a = −0.625, and −0.625 − 0.5·(1 − s) when reducing.
/// - Bicubic Sharper: stretched by 1.05; a = −1, and −1 − 1.6·(1 − s) when reducing.
///
/// Reducing widens the filter by 1/s, down to ¼; below that [`taps`] first averages the image
/// down to four times the target.
fn keys(method: Resample, s: f64) -> Kernel {
    let s = s.clamp(0.25, 1.0);
    let (a, stretch) = match method {
        Resample::BicubicSmoother => (-0.625 - 0.5 * (1.0 - s), 1.15),
        Resample::BicubicSharper => (-1.0 - 1.6 * (1.0 - s), 1.05),
        _ => (-0.75, 1.0),
    };
    Kernel::Keys { a, stretch: stretch / s }
}

fn keys_weight(x: f64, a: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0
    } else if x < 2.0 {
        a * (x * x * x - 5.0 * x * x + 8.0 * x - 4.0)
    } else {
        0.0
    }
}

fn lanczos_weight(x: f64) -> f64 {
    let x = x.abs();
    if x < 1e-9 {
        1.0
    } else if x < 3.0 {
        let px = std::f64::consts::PI * x;
        3.0 * px.sin() * (px / 3.0).sin() / (px * px)
    } else {
        0.0
    }
}

/// One resampling pass: scale factors and the filter of each axis.
#[derive(Clone, Copy, Debug)]
struct Pass {
    sx: f64,
    sy: f64,
    kx: Kernel,
    ky: Kernel,
}

/// Photoshop's passes for `filter` at `(sx, sy)`.
///
/// A bicubic filter is designed for the height's scale, or the width's when the height stays,
/// and serves both axes; the axis with the smaller scale goes first, and the image is clipped and
/// stored between the passes. Bicubic Automatic with one axis growing and the other shrinking is
/// two Image Sizes: the shrinking axis with Sharper, then the growing one with Smoother, each
/// designed for its own scale. Nearest, Bilinear and Lanczos run both axes in one pass.
fn plan(filter: Resample, sx: f64, sy: f64) -> Vec<Pass> {
    let both = |k: Kernel| vec![Pass { sx, sy, kx: k, ky: k }];
    match filter {
        Resample::Nearest => both(Kernel::Nearest),
        Resample::Bilinear => both(Kernel::Bilinear),
        Resample::Lanczos => both(Kernel::Lanczos),
        Resample::PreserveDetails if sx > 1.0 || sy > 1.0 => both(Kernel::Lanczos),
        Resample::PreserveDetails => plan(Resample::BicubicSharper, sx, sy),
        Resample::BicubicAutomatic if sx <= 1.0 && sy <= 1.0 => plan(Resample::BicubicSharper, sx, sy),
        Resample::BicubicAutomatic if sx >= 1.0 && sy >= 1.0 => plan(Resample::BicubicSmoother, sx, sy),
        Resample::BicubicAutomatic => {
            let (sharper, smoother) = (Resample::BicubicSharper, Resample::BicubicSmoother);
            if sx < 1.0 {
                vec![Pass { sx, sy: 1.0, kx: keys(sharper, sx), ky: Kernel::Nearest }, Pass { sx: 1.0, sy, kx: Kernel::Nearest, ky: keys(smoother, sy) }]
            } else {
                vec![Pass { sx: 1.0, sy, kx: Kernel::Nearest, ky: keys(sharper, sy) }, Pass { sx, sy: 1.0, kx: keys(smoother, sx), ky: Kernel::Nearest }]
            }
        }
        Resample::Bicubic | Resample::BicubicSmoother | Resample::BicubicSharper => {
            let k = keys(filter, if sy != 1.0 { sy } else { sx });
            if sx == 1.0 || sy == 1.0 {
                both(k)
            } else if sy < sx {
                vec![Pass { sx: 1.0, sy, kx: k, ky: k }, Pass { sx, sy: 1.0, kx: k, ky: k }]
            } else {
                vec![Pass { sx, sy: 1.0, kx: k, ky: k }, Pass { sx: 1.0, sy, kx: k, ky: k }]
            }
        }
    }
}

/// Per-output-sample source window and normalized weights.
struct Taps {
    start: i32,
    weights: Vec<f32>,
    /// Total weight of taps outside the source's content, which all read the default pixel.
    outside: f32,
}

/// The source position of output sample `o`, `(o + 0.5) / scale − 0.5`, kept on the edge
/// samples (`[first, last]`) when there is an edge, and rounded to Photoshop's 1/128 pixel.
fn position(o: i32, scale: f64, edge: Option<(i32, i32)>) -> f64 {
    let mut u = (o as f64 + 0.5) / scale - 0.5;
    if let Some((a, b)) = edge {
        u = u.clamp(a as f64, b.max(a) as f64);
    }
    let j = u.floor();
    j + ((u - j) * PHASES).round() / PHASES
}

/// The source pixels `[floor(o / scale), floor((o + 1) / scale))` output pixel `o` covers.
fn cover(o: i32, scale: f64) -> (i32, i32) {
    let a = (o as f64 / scale + 1e-9).floor() as i32;
    (a, ((o + 1) as f64 / scale + 1e-9).floor().max(a as f64 + 1.0) as i32)
}

/// Unclamped weights of output sample `o` over source samples `lo..lo + len`.
fn weights(o: i32, scale: f64, kernel: Kernel, edge: Option<(i32, i32)>) -> (i32, Vec<f64>) {
    match kernel {
        Kernel::Nearest => {
            let u = (o as f64 + 0.5) / scale - 0.5;
            ((u + 0.5).floor() as i32, vec![1.0])
        }
        Kernel::Bilinear if scale < 0.5 => {
            let (a, b) = cover(o, scale);
            (a, vec![1.0 / (b - a) as f64; (b - a) as usize])
        }
        Kernel::Bilinear => {
            let u = position(o, scale, edge);
            let j = u.floor();
            (j as i32, vec![1.0 - (u - j), u - j])
        }
        Kernel::Keys { a, stretch } if scale < 0.25 => {
            // Averaged down to four times the target first (pixels of `inner`), then filtered.
            let inner = 4.0 * scale;
            let inner_edge = edge.map(|(a, b)| ((a as f64 * inner - 1e-9).ceil() as i32, ((b + 1) as f64 * inner - 1e-9).ceil() as i32 - 1));
            let (k0, kw) = weights(o, 0.25, Kernel::Keys { a, stretch }, inner_edge);
            let clamp = |k: i32| inner_edge.map_or(k, |(a, b)| k.clamp(a, b.max(a)));
            let boxes: Vec<(i32, i32)> = (k0..).zip(&kw).map(|(k, _)| cover(clamp(k), inner)).collect();
            let lo = boxes.iter().map(|b| b.0).min().unwrap_or(0);
            let hi = boxes.iter().map(|b| b.1).max().unwrap_or(lo);
            let mut w = vec![0.0; (hi - lo).max(0) as usize];
            for ((a, b), v) in boxes.into_iter().zip(kw) {
                for slot in w.get_mut((a - lo) as usize..(b - lo) as usize).into_iter().flatten() {
                    *slot += v / (b - a) as f64;
                }
            }
            (lo, w)
        }
        Kernel::Keys { a, stretch } => {
            let u = position(o, scale, edge);
            let lo = (u - 2.0 * stretch).floor() as i32;
            let hi = (u + 2.0 * stretch).ceil() as i32;
            let w: Vec<f64> = (lo..=hi).map(|i| keys_weight((u - i as f64) / stretch, a)).collect();
            let sum: f64 = w.iter().sum();
            (lo, w.into_iter().map(|v| v / sum).collect())
        }
        Kernel::Lanczos => {
            let stretch = if scale < 1.0 { 1.0 / scale } else { 1.0 };
            let u = (o as f64 + 0.5) / scale - 0.5;
            let u = edge.map_or(u, |(a, b)| u.clamp(a as f64, b.max(a) as f64));
            let lo = (u - 3.0 * stretch).ceil() as i32;
            let hi = (u + 3.0 * stretch).floor() as i32;
            let w: Vec<f64> = (lo..=hi).map(|i| lanczos_weight((i as f64 - u) / stretch)).collect();
            let sum: f64 = w.iter().sum();
            let k = if sum.abs() > 1e-12 { 1.0 / sum } else { 1.0 };
            (lo, w.into_iter().map(|v| v * k).collect())
        }
    }
}

/// Taps for mapping output coordinate `o` (in `[dst0, dst1)`) to the source at `scale`
/// (a scale of 1 copies). With `edge` (`[first, last]` source index), taps outside it read the
/// nearest sample inside it: the image repeats its edge instead of fading out.
/// Taps outside `content` (`[first, last]`) all read the default pixel, so they are folded into
/// `outside` instead of being read: the read window never grows past the content, however small
/// the scale (#713).
fn taps(dst0: i32, dst1: i32, scale: f64, kernel: Kernel, edge: Option<(i32, i32)>, content: (i32, i32)) -> Vec<Taps> {
    let clamp = |i: i32| edge.map_or(i, |(a, b)| i.clamp(a, b));
    let kernel = if scale == 1.0 { Kernel::Nearest } else { kernel };
    (dst0..dst1)
        .map(|o| {
            let (lo, w) = weights(o, scale, kernel, edge);
            let hi = lo + w.len() as i32 - 1;
            // Fold the weights of taps beyond the edge onto the edge sample.
            let start = clamp(lo);
            let mut folded = vec![0.0f32; (clamp(hi) - start + 1).max(1) as usize];
            for (i, v) in (lo..=hi).zip(w) {
                if let Some(slot) = folded.get_mut((clamp(i) - start) as usize) {
                    *slot += v as f32;
                }
            }
            // Keep the part inside the content; the rest reads the default pixel.
            let (a, b) = (start.max(content.0), (start + folded.len() as i32 - 1).min(content.1));
            if a > b {
                return Taps { start: content.0, weights: Vec::new(), outside: folded.iter().sum() };
            }
            let (i0, i1) = ((a - start) as usize, (b - start) as usize + 1);
            let outside = folded.get(..i0).unwrap_or(&[]).iter().chain(folded.get(i1..).unwrap_or(&[])).sum();
            Taps { start: a, weights: folded.get(i0..i1).unwrap_or(&[]).to_vec(), outside }
        })
        .collect()
}

/// Output rectangle for `r` scaled by `(sx, sy)` about the origin.
pub fn scaled_rect(r: Rect, sx: f64, sy: f64) -> Rect {
    if r.is_empty() {
        return Rect::EMPTY;
    }
    Rect::new((r.x0 as f64 * sx).floor() as i32, (r.y0 as f64 * sy).floor() as i32, (r.x1 as f64 * sx).ceil() as i32, (r.y1 as f64 * sy).ceil() as i32)
}

/// Resizes a surface by `(sx, sy)` about the document origin. The surface's
/// default pixel is kept (masks keep their "reveal all" background).
pub fn resize_surface(s: &Surface, sx: f64, sy: f64, filter: Resample) -> Surface {
    resize(s, sx, sy, filter, None)
}

/// [`resize_surface`] for a layer, mask or channel of a document whose canvas is `canvas`
/// (Image Size): at the canvas edges the pixels repeat outwards, so content that reaches an edge
/// keeps it (an opaque layer stays opaque to the border) instead of fading into the default pixel.
/// Pixels beyond the canvas count as the surface's own, and inside the canvas nothing changes.
pub fn resize_surface_in_canvas(s: &Surface, sx: f64, sy: f64, filter: Resample, canvas: Rect) -> Surface {
    resize(s, sx, sy, filter, Some(canvas))
}

/// Samples one source read may hold (64 MB of `f32`); larger windows are read in row chunks.
const READ_BUDGET: usize = 1 << 24;

// Preserve tap order and edge weights while exposing the common channel counts to LLVM.
fn horizontal_fixed<const N: usize>(row: &[f32], out: &mut [f32], taps: &[Taps], x_lo: i32, def: &[f32]) {
    let pixels = row.as_chunks::<N>().0;
    for (t, d) in taps.iter().zip(out.as_chunks_mut::<N>().0) {
        let mut acc = [0.0f32; N];
        for (k, w) in t.weights.iter().enumerate() {
            if let Some(sp) = pixels.get((t.start + k as i32 - x_lo) as usize) {
                for c in 0..N {
                    acc[c] += sp[c] * w;
                }
            }
        }
        for (c, v) in acc.iter_mut().enumerate() {
            *v += def.get(c).copied().unwrap_or(0.0) * t.outside;
        }
        *d = acc;
    }
}

fn horizontal_dynamic(row: &[f32], out: &mut [f32], taps: &[Taps], x_lo: i32, def: &[f32], n: usize) {
    for (t, d) in taps.iter().zip(out.chunks_exact_mut(n)) {
        for (k, w) in t.weights.iter().enumerate() {
            let sx_ = (t.start + k as i32 - x_lo) as usize;
            if let Some(sp) = row.get(sx_ * n..(sx_ + 1) * n) {
                for c in 0..n {
                    d[c] += sp[c] * w;
                }
            }
        }
        for c in 0..n {
            d[c] += def[c] * t.outside;
        }
    }
}

fn resize(s: &Surface, sx: f64, sy: f64, filter: Resample, canvas: Option<Rect>) -> Surface {
    resize_with_budget(s, sx, sy, filter, canvas, READ_BUDGET)
}

/// Runs `filter`'s passes. Between two passes the image is stored in its own format, which
/// clips it (and rounds it at 8 and 16 bits) as Photoshop does; the smaller scale goes first, so
/// the intermediate image is never larger than the result.
fn resize_with_budget(s: &Surface, sx: f64, sy: f64, filter: Resample, canvas: Option<Rect>, budget: usize) -> Surface {
    let mut cur: Option<Surface> = None;
    let mut canvas = canvas;
    for p in plan(filter, sx, sy) {
        let next = resize_pass(cur.as_ref().unwrap_or(s), p, canvas, budget);
        canvas = canvas.map(|c| scaled_rect(c, p.sx, p.sy));
        cur = Some(next);
    }
    let mut out = cur.unwrap_or_else(|| s.clone());
    if filter == Resample::PreserveDetails && (sx > 1.0 || sy > 1.0) {
        let dst = out.content_bounds();
        let p = crate::FilterParams::UnsharpMask { amount: 30.0, radius: 0.6 * sx.max(sy) as f32, threshold: 0.0 };
        let area = crate::output_area(&p, dst, dst, None);
        out = crate::apply(&out, &p, area, dst, None);
    }
    out.prune();
    out
}

/// One pass. With `canvas`, the content and the canvas repeat their edge outwards (see
/// [`resize_surface_in_canvas`]).
fn resize_pass(s: &Surface, pass: Pass, canvas: Option<Rect>, budget: usize) -> Surface {
    let Pass { sx, sy, kx, ky } = pass;
    let edge = canvas.map(|c| s.content_bounds().union(&c)).filter(|a| !a.is_empty());
    let fmt = s.format();
    let mut out = Surface::with_default(fmt, &s.default_pixel());
    let src = s.content_bounds();
    let dst = scaled_rect(src, sx, sy);
    if dst.is_empty() {
        return out;
    }
    let n = fmt.channels();
    let alpha = fmt.alpha;
    // With `edge`, every tap is clamped into the edge area, which holds the content.
    let area = edge.unwrap_or(src);
    let hx = taps(dst.x0, dst.x1, sx, kx, edge.map(|r| (r.x0, r.x1 - 1)), (area.x0, area.x1 - 1));
    let vy = taps(dst.y0, dst.y1, sy, ky, edge.map(|r| (r.y0, r.y1 - 1)), (area.y0, area.y1 - 1));
    let x_lo = hx.iter().filter(|t| !t.weights.is_empty()).map(|t| t.start).min().unwrap_or(area.x0);
    let x_hi = hx.iter().map(|t| t.start + t.weights.len() as i32).max().unwrap_or(area.x0).max(x_lo);
    let mut def = s.default_pixel();
    premultiply(&mut def, n, alpha);
    let dw = dst.width() as usize;
    const BAND: usize = 64;
    let bands: Vec<(usize, usize)> = (0..vy.len()).step_by(BAND).map(|b| (b, (b + BAND).min(vy.len()))).collect();
    // Horizontal pass of one source row (`row` spans `x_lo..x_hi`) into `dw` samples.
    let horizontal = |row: &[f32], out: &mut [f32]| match n {
        1 => horizontal_fixed::<1>(row, out, &hx, x_lo, &def),
        2 => horizontal_fixed::<2>(row, out, &hx, x_lo, &def),
        3 => horizontal_fixed::<3>(row, out, &hx, x_lo, &def),
        4 => horizontal_fixed::<4>(row, out, &hx, x_lo, &def),
        5 => horizontal_fixed::<5>(row, out, &hx, x_lo, &def),
        _ => horizontal_dynamic(row, out, &hx, x_lo, &def, n),
    };
    // A source row of nothing but the default pixel, after the horizontal pass.
    let def_row = {
        let row: Vec<f32> = def.iter().copied().cycle().take((x_hi - x_lo) as usize * n).collect();
        let mut out = vec![0.0f32; dw * n];
        horizontal(&row, &mut out);
        out
    };
    // At most one tile row per read anyway, so this always fits an `i32`.
    let rows_per_read = (budget / ((x_hi - x_lo).max(1) as usize * n).max(dw * n)).clamp(1, TILE_SIZE as usize) as i32;
    let run = |&(b0, b1): &(usize, usize)| -> (Rect, Vec<f32>) {
        let band = vy.get(b0..b1).unwrap_or(&[]);
        let y_lo = band.iter().filter(|t| !t.weights.is_empty()).map(|t| t.start).min().unwrap_or(area.y0);
        let y_hi = band.iter().map(|t| t.start + t.weights.len() as i32).max().unwrap_or(area.y0).max(y_lo);
        let mut res = vec![0.0f32; band.len() * dw * n];
        // Source rows stream through in bounded reads and go straight into the output rows that
        // use them, so memory is the read budget plus this band's output, whatever the scale or
        // the source size (#713). In ascending row order, as one read of the whole window would.
        let add = |res: &mut [f32], y: i32, row: &[f32]| {
            for (t, d) in band.iter().zip(res.chunks_exact_mut(dw * n)) {
                if let Some(w) = t.weights.get((y - t.start) as usize).filter(|_| y >= t.start) {
                    for (dv, sv) in d.iter_mut().zip(row) {
                        *dv += sv * w;
                    }
                }
            }
        };
        let mut rows = Vec::new();
        let mut hrows = Vec::new();
        let mut y = y_lo;
        while y < y_hi {
            let y1 = ((y.div_euclid(TILE_SIZE) + 1) * TILE_SIZE).min(y_hi);
            if s.has_tiles_in(Rect::new(x_lo, y, x_hi, y1)) {
                let mut c0 = y;
                while c0 < y1 {
                    let c1 = (c0 + rows_per_read).min(y1);
                    let read = Rect::new(x_lo, c0, x_hi, c1);
                    s.read_region_into(read, &mut rows);
                    premultiply(&mut rows, n, alpha);
                    hrows.clear();
                    hrows.resize((c1 - c0) as usize * dw * n, 0.0);
                    let rw = read.width() as usize * n;
                    for (row, out) in rows.chunks_exact(rw.max(1)).zip(hrows.chunks_exact_mut(dw * n)) {
                        horizontal(row, out);
                    }
                    for (r, h) in (c0..c1).zip(hrows.chunks_exact(dw * n)) {
                        add(&mut res, r, h);
                    }
                    c0 = c1;
                }
            } else {
                // Nothing stored in this tile row: every row in it is the default row, so each
                // output row takes it once, with the summed weight of those rows.
                for (t, d) in band.iter().zip(res.chunks_exact_mut(dw * n)) {
                    let (a, b) = ((y - t.start).max(0) as usize, (y1 - t.start).max(0) as usize);
                    let w: f32 = t.weights.get(a.min(t.weights.len())..b.min(t.weights.len())).unwrap_or(&[]).iter().sum();
                    if w != 0.0 {
                        for (dv, sv) in d.iter_mut().zip(&def_row) {
                            *dv += sv * w;
                        }
                    }
                }
            }
            y = y1;
        }
        for (t, d) in band.iter().zip(res.chunks_exact_mut(dw * n)) {
            for (dv, sv) in d.iter_mut().zip(&def_row) {
                *dv += sv * t.outside;
            }
        }
        if alpha {
            for px in res.chunks_exact_mut(n) {
                px[n - 1] = px[n - 1].clamp(0.0, 1.0);
            }
        }
        unpremultiply(&mut res, n, alpha);
        (Rect::new(dst.x0, dst.y0 + b0 as i32, dst.x1, dst.y0 + b1 as i32), res)
    };
    // Bands go into the output a group at a time, so the `f32` rows held at once are bounded by
    // the group, not by the whole destination (#1544).
    #[cfg(not(target_arch = "wasm32"))]
    let group = rayon::current_num_threads().max(1) * 2;
    #[cfg(target_arch = "wasm32")]
    let group = 1;
    for chunk in bands.chunks(group) {
        #[cfg(not(target_arch = "wasm32"))]
        let parts: Vec<(Rect, Vec<f32>)> = {
            use rayon::prelude::*;
            chunk.par_iter().map(run).collect()
        };
        #[cfg(target_arch = "wasm32")]
        let parts: Vec<(Rect, Vec<f32>)> = chunk.iter().map(run).collect();
        for (r, d) in parts {
            out.write_region(r, &d);
        }
    }
    out
}

/// Moves a surface by whole pixels (default pixel kept).
pub fn translate_surface(s: &Surface, dx: i32, dy: i32) -> Surface {
    s.translated(dx, dy, s.content_bounds())
}

/// Keeps only the pixels inside `keep` (others become the default pixel).
pub fn crop_surface(s: &Surface, keep: Rect) -> Surface {
    let mut out = Surface::with_default(s.format(), &s.default_pixel());
    let r = s.content_bounds().intersect(&keep);
    if !r.is_empty() {
        out.write_interleaved(r, &s.to_interleaved(r));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::{ColorMode, PixelFormat, SampleType};

    fn ramp(sample: SampleType) -> Surface {
        let mut s = Surface::new(PixelFormat::new(ColorMode::Rgb, sample, true));
        let r = Rect::new(0, 0, 64, 32);
        let v: Vec<f32> = (0..32).flat_map(|_| (0..64).flat_map(|x| [x as f32 / 63.0, 0.5, 1.0 - x as f32 / 63.0, 1.0])).collect();
        s.write_region(r, &v);
        s
    }

    #[test]
    fn fixed_channel_rows_match_dynamic_resampling() {
        fn check<const N: usize>() {
            for scale in [0.12, 0.5, 1.0, 2.3] {
                for filter in [Kernel::Nearest, Kernel::Bilinear, keys(Resample::Bicubic, scale), keys(Resample::BicubicSharper, scale), Kernel::Lanczos] {
                    for clamp in [None, Some((-3, 31))] {
                        let ts = taps(-5, 41, scale, filter, clamp, (-3, 31));
                        let row: Vec<f32> = (0..35 * N).map(|i| ((i * 73) % 251) as f32 / 250.0).collect();
                        let def: Vec<f32> = (0..N).map(|i| i as f32 / 7.0).collect();
                        let mut expected = vec![0.0; ts.len() * N];
                        let mut actual = expected.clone();
                        horizontal_dynamic(&row, &mut expected, &ts, -3, &def, N);
                        horizontal_fixed::<N>(&row, &mut actual, &ts, -3, &def);
                        assert_eq!(actual, expected, "N={N} scale={scale} filter={filter:?} clamp={clamp:?}");
                    }
                }
            }
        }
        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
        check::<5>();
    }

    /// Photoshop 25.4's Image Size of a 16-bit greyscale pattern, in levels of 32768. Covers the
    /// filters and their 2-D rules: Sharper reducing the width while the height grows is designed
    /// for the height (not widened) and runs the width first; Bicubic Automatic splits a mixed
    /// resize into Sharper and Smoother; Bicubic widens for the height's 0.6 while the width
    /// grows; Smoother below ¼ averages down first; Bilinear below ½ averages whole pixels. On black
    /// and white the overshoot is clipped between the passes, so their order shows.
    #[test]
    fn matches_photoshop_image_size() {
        // (black and white pattern, filter, size, new size, Photoshop's levels)
        type Probe = (bool, Resample, (i32, i32), (i32, i32), &'static [u16]);
        let cases: [Probe; 9] = [
            (
                false,
                Resample::BicubicSharper,
                (12, 10),
                (7, 16),
                &[
                    3777, 13165, 22219, 16176, 10460, 19488, 29643, 11249, 12634, 21696, 13330, 10319, 18965, 29120, 21280, 11204, 20279, 16184, 8200, 17548,
                    27703, 21824, 10182, 19259, 26967, 5238, 16528, 26683, 17366, 8895, 17922, 32768, 2402, 15190, 25345, 17493, 7834, 16903, 31316, 1748,
                    14172, 24326, 24718, 6368, 15746, 30390, 0, 13107, 23169, 31215, 4790, 14456, 27639, 5080, 11597, 21880, 29649, 3864, 13507, 24136, 20600,
                    10264, 20931, 28009, 0, 12446, 21385, 28735, 8754, 19642, 26154, 4049, 10895, 20832, 23523, 7620, 18485, 23204, 20126, 8787, 19818, 21447,
                    6173, 17467, 20390, 31862, 6616, 18480, 24094, 6500, 16129, 19749, 27472, 5810, 17460, 28281, 7655, 15108, 18394, 25515, 4427, 16043,
                    28749, 7026, 13691, 17871, 24992, 3904, 15520, 27200, 6075, 13168,
                ],
            ),
            (
                false,
                Resample::BicubicAutomatic,
                (12, 10),
                (7, 16),
                &[
                    4201, 12337, 27677, 15894, 4816, 20021, 29631, 10348, 9882, 27304, 14492, 3766, 19574, 29046, 18743, 5944, 25299, 16116, 4147, 17829,
                    27787, 19399, 4206, 21790, 23748, 7966, 15117, 26569, 15941, 2695, 19445, 28341, 10104, 13050, 25465, 16251, 2613, 18023, 26770, 8432,
                    11951, 24281, 22800, 7141, 15351, 24790, 5953, 11004, 23130, 28322, 10921, 12440, 24940, 9659, 9368, 22012, 27611, 10006, 10488, 26419,
                    18657, 7071, 20807, 25492, 6937, 9239, 26574, 22363, 5435, 19688, 24808, 9136, 8184, 25295, 19708, 3939, 18677, 25053, 18234, 7120, 24061,
                    17941, 2545, 17568, 24869, 23748, 6081, 21401, 21862, 6018, 15056, 23602, 22177, 4859, 17816, 28450, 11911, 11653, 22253, 20263, 3595,
                    15790, 29734, 12969, 9683, 21668, 19678, 3010, 15424, 28429, 11728, 9300,
                ],
            ),
            (
                false,
                Resample::Bicubic,
                (12, 10),
                (20, 6),
                &[
                    11517, 10000, 8617, 9364, 12272, 16362, 21396, 25508, 25506, 19474, 10887, 5289, 5842, 10286, 14986, 18566, 21910, 25511, 28611, 29922,
                    24788, 17949, 8497, 4593, 7309, 12594, 17150, 20945, 24892, 26196, 22979, 16123, 9691, 7483, 10061, 14653, 18841, 22420, 25514, 26824,
                    25702, 23396, 17586, 11378, 8127, 8882, 12310, 17039, 22345, 25871, 24700, 18125, 9951, 5493, 6802, 11215, 15576, 19259, 22384, 23694,
                    26692, 26985, 22732, 14270, 6889, 4950, 8348, 13583, 17529, 21570, 25966, 27275, 22817, 14643, 8068, 6911, 10694, 15930, 19443, 20705,
                    21499, 23413, 25601, 24336, 18665, 11686, 7800, 8947, 12974, 18067, 22707, 25285, 23077, 16645, 9789, 6586, 8147, 12329, 15935, 17465,
                    17962, 20462, 24615, 25377, 20036, 11458, 5421, 5415, 9528, 14134, 17782, 22482, 26926, 27479, 21881, 13308, 7533, 7734, 11549, 13992,
                ],
            ),
            (
                false,
                Resample::BicubicSmoother,
                (40, 3),
                (9, 3),
                &[
                    11315, 18740, 17475, 15974, 16503, 16042, 18215, 14712, 16238, 18482, 14351, 14664, 18012, 20260, 13338, 15511, 18468, 18275, 16790, 19168,
                    14745, 14998, 20341, 18150, 13819, 16493, 16432,
                ],
            ),
            (
                false,
                Resample::Bilinear,
                (12, 10),
                (5, 4),
                &[11316, 14541, 17920, 10855, 24679, 19763, 9933, 23757, 6247, 20070, 25601, 5325, 19149, 17306, 15463, 20992, 16384, 14541, 23143, 10854],
            ),
            (
                false,
                Resample::Nearest,
                (12, 10),
                (7, 16),
                &[
                    1638, 12698, 23757, 3482, 9011, 20070, 31130, 1638, 12698, 23757, 3482, 9011, 20070, 31130, 31130, 10854, 21914, 1638, 7168, 18227, 29286,
                    29286, 9011, 20070, 31130, 5325, 16384, 27443, 29286, 9011, 20070, 31130, 5325, 16384, 27443, 27443, 7168, 18227, 29286, 3482, 14541,
                    25600, 25600, 5325, 16384, 27443, 1638, 12698, 23757, 25600, 5325, 16384, 27443, 1638, 12698, 23757, 23757, 3482, 14541, 25600, 31130,
                    10854, 21914, 23757, 3482, 14541, 25600, 31130, 10854, 21914, 21914, 1638, 12698, 23757, 29286, 9011, 20070, 20070, 31130, 10854, 21914,
                    27443, 7168, 18227, 20070, 31130, 10854, 21914, 27443, 7168, 18227, 18227, 29286, 9011, 20070, 25600, 5325, 16384, 16384, 27443, 7168,
                    18227, 23757, 3482, 14541, 16384, 27443, 7168, 18227, 23757, 3482, 14541,
                ],
            ),
            (
                true,
                Resample::Bicubic,
                (12, 10),
                (20, 6),
                &[
                    9960, 6313, 993, 0, 4525, 14807, 26472, 32768, 32768, 23510, 10514, 0, 0, 0, 10110, 22576, 31764, 32768, 32768, 32768, 26136, 18594, 6164,
                    0, 0, 2565, 14499, 26983, 32768, 32768, 26358, 14709, 3969, 0, 0, 6602, 17734, 27952, 32768, 32768, 31590, 27808, 18307, 7429, 0, 0, 3712,
                    14435, 26473, 32768, 31131, 20665, 7535, 0, 0, 2177, 13238, 25049, 32768, 32768, 32768, 32768, 26136, 14174, 2548, 0, 0, 6190, 18705,
                    30273, 32768, 32768, 25517, 13238, 2177, 0, 0, 7250, 19418, 26624, 32768, 32768, 32768, 28203, 18522, 7200, 0, 0, 3285, 14794, 26862,
                    32768, 29913, 19013, 6667, 0, 0, 4443, 14506, 20751, 17344, 23066, 31075, 32361, 23949, 10738, 0, 0, 0, 10110, 22657, 32768, 32768, 32768,
                    22658, 10192, 1004, 0, 0, 0,
                ],
            ),
            (
                true,
                Resample::BicubicSharper,
                (12, 10),
                (7, 16),
                &[
                    801, 3050, 32489, 16260, 0, 29819, 32767, 9161, 1781, 29578, 14350, 0, 32768, 32767, 21132, 0, 25409, 17862, 0, 26925, 32767, 22769, 0,
                    24839, 27566, 0, 9397, 32768, 19813, 0, 27795, 32768, 0, 0, 32768, 20744, 0, 25797, 32768, 0, 0, 32768, 27555, 0, 11184, 32768, 0, 0,
                    32768, 32768, 0, 0, 32768, 5110, 0, 30386, 32768, 0, 0, 32768, 20385, 0, 23267, 32768, 0, 0, 32768, 29130, 0, 19191, 32768, 2670, 0, 31290,
                    25387, 0, 21263, 32767, 20198, 0, 21586, 24510, 0, 23714, 32768, 32768, 0, 14165, 27804, 1007, 14254, 32768, 30149, 0, 16077, 32112, 2886,
                    1884, 21043, 29607, 0, 16384, 32768, 3566, 0, 12683, 29616, 0, 16384, 32768, 3196, 0,
                ],
            ),
            (
                true,
                Resample::BicubicSharper,
                (12, 10),
                (20, 6),
                &[
                    9273, 5043, 0, 0, 2698, 14484, 28464, 32768, 32768, 24841, 10877, 0, 0, 0, 9684, 22346, 31492, 32768, 32768, 32768, 25960, 18430, 4978, 0,
                    0, 977, 15047, 28475, 32768, 32768, 27939, 14157, 1813, 0, 0, 5296, 17828, 28439, 32768, 32768, 32768, 27849, 17576, 6209, 0, 0, 2169,
                    15076, 28935, 32768, 32768, 22167, 7399, 0, 0, 211, 12977, 24638, 32768, 32768, 32768, 32768, 25960, 14244, 824, 0, 0, 5093, 18802, 31993,
                    32768, 32768, 26467, 12977, 211, 0, 0, 6300, 19418, 26712, 32768, 32768, 32768, 27907, 17874, 5820, 0, 0, 1853, 16055, 29518, 32768, 32768,
                    20889, 5880, 0, 0, 2879, 13798, 20185, 19010, 25565, 32768, 32302, 25431, 11478, 0, 0, 0, 9684, 23083, 32768, 32768, 32768, 23084, 10422,
                    1276, 0, 0, 0,
                ],
            ),
        ];
        for (binary, filter, (w, h), (wo, ho), want) in cases {
            let mut s = Surface::new(PixelFormat::new(ColorMode::Grayscale, SampleType::U16, false));
            let canvas = Rect::new(0, 0, w, h);
            let level = |x: i32, y: i32| match ((x * 37 + y * 101) % 17) as f32 {
                k if binary => {
                    if k > 8.0 {
                        32767.0 / 32768.0
                    } else {
                        0.0
                    }
                }
                k => ((k / 16.0 * 0.9 + 0.05) * 32768.0).round() / 32768.0,
            };
            let px: Vec<f32> = (0..h).flat_map(|y| (0..w).map(move |x| level(x, y))).collect();
            s.write_region(canvas, &px);
            let o = resize_surface_in_canvas(&s, wo as f64 / w as f64, ho as f64 / h as f64, filter, canvas);
            let got = o.read_region(Rect::new(0, 0, wo, ho));
            assert_eq!(got.len(), want.len(), "{filter:?}");
            let worst = got.iter().zip(want).map(|(g, &p)| (g * 32768.0 - p as f32).abs()).fold(0.0f32, f32::max);
            assert!(worst <= 8.0, "{filter:?} {w}x{h} to {wo}x{ho}: {worst} levels of 32768 from Photoshop");
        }
    }

    /// Photoshop's Bicubic doubles with the Keys cubic at a = −0.75 (Catmull-Rom is −0.5): an
    /// impulse spreads as −9, −27, 67, 225 /256 and back, as Photoshop 25.4 gives it.
    #[test]
    fn bicubic_is_photoshops_keys_cubic() {
        let mut s = Surface::new(PixelFormat::new(ColorMode::Grayscale, SampleType::F32, false));
        let mut row = [0.25; 30];
        row[10] = 0.75;
        s.write_region(Rect::new(0, 0, 30, 1), &row);
        let o = resize_surface(&s, 2.0, 1.0, Resample::Bicubic);
        let got: Vec<f32> = o.read_region(Rect::new(17, 0, 25, 1)).iter().map(|v| (v - 0.25) * 512.0).collect();
        for (g, want) in got.iter().zip([-9.0, -27.0, 67.0, 225.0, 225.0, 67.0, -27.0, -9.0]) {
            assert!((g - want).abs() < 1e-3, "{got:?}");
        }
    }

    #[test]
    fn identity_scale_is_lossless() {
        for f in [Resample::Nearest, Resample::Bilinear, Resample::Bicubic, Resample::Lanczos] {
            let s = ramp(SampleType::U16);
            let o = resize_surface(&s, 1.0, 1.0, f);
            let r = Rect::new(0, 0, 64, 32);
            let (a, b) = (s.read_region(r), o.read_region(r));
            let d = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            assert!(d < 1e-4, "{f:?} {d}");
        }
    }

    #[test]
    fn half_and_double_sizes() {
        for f in [Resample::Nearest, Resample::Bilinear, Resample::Bicubic, Resample::Lanczos, Resample::PreserveDetails] {
            let s = ramp(SampleType::F32);
            let half = resize_surface(&s, 0.5, 0.5, f);
            assert_eq!(half.content_bounds(), Rect::new(0, 0, 32, 16), "{f:?}");
            let p = half.pixel(16, 8);
            assert!((p[1] - 0.5).abs() < 0.02 && (p[3] - 1.0).abs() < 1e-3, "{f:?} {p:?}");
            let dbl = resize_surface(&s, 2.0, 2.0, f);
            assert_eq!(dbl.content_bounds(), Rect::new(0, 0, 128, 64), "{f:?}");
            // Mid-gray channel and the ramp's monotonicity are preserved.
            let row: Vec<f32> = (8..120).map(|x| dbl.pixel(x, 30)[0]).collect();
            assert!(row.windows(2).all(|w| w[1] >= w[0] - 0.02), "{f:?}");
        }
    }

    #[test]
    fn downscale_averages_checkerboard() {
        let mut s = Surface::new(PixelFormat::RGBA8);
        for y in 0..64 {
            for x in 0..64 {
                let v = ((x + y) % 2) as f32;
                s.write_pixel(x, y, &[v, v, v, 1.0]);
            }
        }
        let o = resize_surface(&s, 0.25, 0.25, Resample::Bicubic);
        let p = o.pixel(8, 8);
        assert!((p[0] - 0.5).abs() < 0.05, "{p:?}");
    }

    #[test]
    fn transparent_edges_do_not_darken() {
        let mut s = Surface::new(PixelFormat::RGBA32F);
        s.fill_rect(Rect::new(10, 10, 20, 20), &[1.0, 0.0, 0.0, 1.0]);
        let o = resize_surface(&s, 1.7, 1.7, Resample::Lanczos);
        let edge = o.pixel(17, 25);
        assert!(edge[3] > 0.0 && (edge[0] - 1.0).abs() < 0.05 && edge[1].abs() < 0.05, "{edge:?}");
    }

    /// Image Size of a layer that fills the canvas: its edges stay opaque and keep their colour,
    /// up or down, at every depth; a layer away from the edges keeps its soft border.
    #[test]
    fn canvas_edges_repeat_instead_of_fading() {
        let canvas = Rect::new(0, 0, 30, 20);
        let colour = [0.2, 0.6, 1.0, 1.0];
        let filters = [Resample::Nearest, Resample::Bilinear, Resample::Bicubic, Resample::Lanczos, Resample::PreserveDetails];
        for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let fmt = PixelFormat::new(ColorMode::Rgb, sample, true);
            let mut s = Surface::new(fmt);
            s.fill_rect(canvas, &colour);
            for f in filters {
                for k in [2.0, 1.37, 0.5] {
                    let o = resize_surface_in_canvas(&s, k, k, f, canvas);
                    let dst = scaled_rect(canvas, k, k);
                    assert_eq!(o.content_bounds(), dst, "{sample:?} {f:?} {k}");
                    for (x, y) in [(0, 0), (dst.x1 - 1, 0), (0, dst.y1 - 1), (dst.x1 - 1, dst.y1 - 1), (dst.x1 / 2, 0), (0, dst.y1 / 2)] {
                        let p = o.pixel(x, y);
                        let off = p.iter().zip(colour).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                        assert!(off < 0.01, "{sample:?} {f:?} {k} ({x},{y}): {p:?}");
                    }
                }
            }
            // Inside the canvas the default pixel is still what lies around a layer.
            let mut inner = Surface::new(fmt);
            inner.fill_rect(Rect::new(10, 5, 20, 15), &[1.0, 0.0, 0.0, 1.0]);
            let r = Rect::new(0, 0, 60, 40);
            let a = resize_surface_in_canvas(&inner, 1.7, 1.7, Resample::Lanczos, canvas);
            assert_eq!(a.read_region(r), resize_surface(&inner, 1.7, 1.7, Resample::Lanczos).read_region(r), "{sample:?}");
        }
    }

    /// Two tiles in opposite corners of a 100000 px square: the content bounds are huge but
    /// almost nothing is stored. Shrinking it to one pixel must neither allocate a window
    /// proportional to the scale (#713) nor read the empty middle.
    #[test]
    fn tiny_target_of_a_huge_sparse_source() {
        let mut s = Surface::new(PixelFormat::RGBA32F);
        s.fill_rect(Rect::new(0, 0, 256, 256), &[1.0, 1.0, 1.0, 1.0]);
        s.fill_rect(Rect::new(99_840, 99_840, 100_096, 100_096), &[1.0, 1.0, 1.0, 1.0]);
        let k = 1.0 / 100_096.0;
        let o = resize_surface(&s, k, k, Resample::Bicubic);
        assert_eq!(o.content_bounds(), Rect::new(0, 0, 1, 1));
        let p = o.pixel(0, 0);
        // Two 256² tiles out of 100096²: nearly transparent, and white where it isn't.
        assert!(p[3] >= 0.0 && p[3] < 1e-3 && p.iter().all(|v| v.is_finite()), "{p:?}");
        // Squashed to one row but kept wide: the whole height feeds one band of output rows,
        // which must not hold a full-height intermediate.
        let o = resize_surface(&s, 1000.0 / 100_096.0, k, Resample::Bicubic);
        let b = o.content_bounds();
        assert!(b.height() == 1 && (1000..=1001).contains(&b.width()), "{b:?}");
        assert!(o.pixel(0, 0)[3] > 0.0 && o.pixel(500, 0)[3] == 0.0, "{:?} {:?}", o.pixel(0, 0), o.pixel(500, 0));
    }

    /// Tile rows with nothing stored are not read; they count as rows of the default pixel,
    /// exactly as reading them would (and within rounding for a non-zero default).
    #[test]
    fn skipped_empty_rows_match_reading_them() {
        for (fmt, def, tol) in [(PixelFormat::RGBA32F, vec![0.0, 0.0, 0.0, 0.0], 0.0), (PixelFormat::GRAY8, vec![1.0], 1e-6)] {
            let mut sparse = Surface::with_default(fmt, &def);
            sparse.fill_rect(Rect::new(0, 0, 300, 40), &vec![0.5; def.len()]);
            sparse.fill_rect(Rect::new(100, 1500, 400, 1540), &vec![0.25; def.len()]);
            // The same pixels with the empty rows stored as tiles of the default pixel.
            let mut dense = sparse.clone();
            dense.fill_rect(Rect::new(0, 40, 400, 1500), &def);
            assert!(dense.tile_count() > sparse.tile_count());
            for f in [Resample::Bilinear, Resample::Bicubic, Resample::Lanczos] {
                for (kx, ky) in [(0.013, 0.013), (0.37, 0.37), (1.0, 0.01)] {
                    let (a, b) = (resize_surface(&sparse, kx, ky, f), resize_surface(&dense, kx, ky, f));
                    let r = a.content_bounds().union(&b.content_bounds());
                    let d = a.read_region(r).iter().zip(b.read_region(r)).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
                    assert!(d <= tol, "{fmt:?} {f:?} {kx} {ky}: {d}");
                }
            }
        }
    }

    /// Reading the source in row chunks gives exactly what one read of the whole window gives.
    #[test]
    fn chunked_reads_match_a_single_read() {
        let mut s = ramp(SampleType::F32);
        s.fill_rect(Rect::new(600, 700, 640, 720), &[0.3, 0.9, 0.1, 0.5]);
        let mut mask = Surface::with_default(PixelFormat::GRAY8, &[1.0]);
        mask.fill_rect(Rect::new(0, 0, 40, 30), &[0.0]);
        mask.fill_rect(Rect::new(500, 900, 530, 940), &[0.25]);
        for src in [&s, &mask] {
            for f in [Resample::Nearest, Resample::Bilinear, Resample::Bicubic, Resample::Lanczos] {
                for k in [0.013, 0.37, 1.6] {
                    let whole = resize_with_budget(src, k, k, f, None, usize::MAX);
                    let chunked = resize_with_budget(src, k, k, f, None, 1);
                    let r = whole.content_bounds().union(&chunked.content_bounds());
                    assert_eq!(whole.read_region(r), chunked.read_region(r), "{f:?} {k}");
                }
            }
        }
    }

    #[test]
    fn masks_keep_default_pixel() {
        let fmt = PixelFormat::GRAY8;
        let mut m = Surface::with_default(fmt, &[1.0]);
        m.fill_rect(Rect::new(0, 0, 10, 10), &[0.0]);
        let o = resize_surface(&m, 2.0, 2.0, Resample::Bilinear);
        assert_eq!(o.default_pixel(), vec![1.0]);
        assert!(o.pixel(5, 5)[0] < 0.01);
    }

    #[test]
    fn translate_and_crop() {
        let s = ramp(SampleType::U8);
        let t = translate_surface(&s, 5, -3);
        assert_eq!(t.pixel(15, 7), s.pixel(10, 10));
        let c = crop_surface(&s, Rect::new(0, 0, 10, 10));
        assert_eq!(c.content_bounds(), Rect::new(0, 0, 10, 10));
    }

    /// #1114: nearest neighbor reduction must sample single source pixels, never average.
    #[test]
    fn nearest_neighbor_reduction_does_not_average() {
        let mut s = Surface::new(PixelFormat::GRAY8);
        for y in 0..8 {
            for x in 0..8 {
                if (x + y) % 2 == 0 {
                    s.fill_rect(Rect::new(x, y, x + 1, y + 1), &[1.0]);
                }
            }
        }
        let reduced = resize_surface(&s, 0.5, 0.5, Resample::Nearest);
        for y in 0..4 {
            for x in 0..4 {
                let p = reduced.pixel(x, y)[0];
                assert!(p == 0.0 || p == 1.0, "nearest produced an averaged value {p} at ({x}, {y})");
            }
        }
    }
}
