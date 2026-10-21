//! Edge refinement ("Select and Mask" / Refine Edge) and colour decontamination.
//!
//! * Soft edges come from the **guided filter** (K. He, J. Sun and X. Tang, "Guided Image
//!   Filtering", ECCV 2010 / PAMI 2013): filtering the binary selection with the image as guide
//!   fits a local linear model `α ≈ aᵀI + b` in every window, so the mask takes on the image's
//!   soft transitions (hair, blur) inside a band of `radius` pixels around the selection edge.
//!   "Smart radius" adapts the band per pixel to the transition width observed in the image.
//! * Then, in Photoshop's order: Smooth, Feather (Gaussian, σ = feather / 2), Contrast and
//!   Shift Edge (grey dilation / erosion).
//! * **Decontaminate colours** replaces fringe colours by the normalised-convolution average of
//!   nearby fully selected pixels, blended by `amount` and the fringe's transparency.
//!
//! Work is tiled: only tiles near the selection boundary are processed (in parallel on native
//! targets), so cost scales with the boundary length rather than the selection area.

use photocraft_color::PixelFormat;
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde::{Deserialize, Serialize};

use crate::segment::{RgbImage, Sampler};
use crate::selection::{Region, edt};

/// Select and Mask parameters (Photoshop's units).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct RefineParams {
    /// Edge-detection band radius in pixels (0 = no edge detection).
    pub radius: f32,
    /// Adapt the band to the edge's observed softness.
    pub smart_radius: bool,
    /// 0–100.
    pub smooth: f32,
    /// Pixels.
    pub feather: f32,
    /// 0–100.
    pub contrast: f32,
    /// −100–100. 100% moves the edge by `max(radius, 4)` pixels. Values between whole
    /// pixels blend the two neighbouring shifts, so the slider does not jump.
    pub shift_edge: f32,
}

impl Default for RefineParams {
    fn default() -> Self {
        Self { radius: 0.0, smart_radius: false, smooth: 0.0, feather: 0.0, contrast: 0.0, shift_edge: 0.0 }
    }
}

/// Guided-filter regulariser (colour guide, samples in 0–1).
pub const EPS: f32 = 1e-4;
const TILE: i32 = 256;

impl RefineParams {
    fn gf_radius(&self) -> usize {
        self.radius.ceil().clamp(1.0, 64.0) as usize
    }
    fn smooth_sigma(&self) -> f32 {
        self.smooth.clamp(0.0, 100.0) / 100.0 * 4.0
    }
    fn feather_sigma(&self) -> f32 {
        self.feather.max(0.0) / 2.0
    }
    /// Signed shift in pixels, fractional. 100% is `max(radius, 4)` px, clamped to ±64.
    fn shift_amount(&self) -> f32 {
        let px = self.shift_edge.clamp(-100.0, 100.0) / 100.0 * self.radius.max(4.0);
        if px.is_finite() { px.clamp(-64.0, 64.0) } else { 0.0 }
    }
    /// How many pixels the mask can move: the farther of the two integer radii a fractional
    /// shift blends. Whole-pixel amounts stay exact (`ceil(4) == 4`).
    fn shift_px(&self) -> i32 {
        let px = self.shift_amount();
        let reach = px.abs().ceil();
        let signed = if px.is_sign_negative() { -reach } else { reach };
        if signed.is_finite() { signed as i32 } else { 0 }
    }
    fn uses_image(&self) -> bool {
        self.radius > 0.0
    }
    /// Pixels read around an output pixel.
    pub fn halo(&self) -> i32 {
        let edge = if self.uses_image() {
            let g = self.gf_radius() as i32;
            (self.radius.ceil() as i32).max(2 * g + if self.smart_radius { g + 1 } else { 0 })
        } else {
            0
        };
        edge + (3.0 * self.smooth_sigma()).ceil() as i32 + (3.0 * self.feather_sigma()).ceil() as i32 + self.shift_px().abs() + 2
    }
    /// How far the refined mask can extend beyond the original.
    pub fn spread(&self) -> i32 {
        (if self.uses_image() { self.radius.ceil() as i32 } else { 0 })
            + (3.0 * self.smooth_sigma()).ceil() as i32
            + (3.0 * self.feather_sigma()).ceil() as i32
            + self.shift_px().max(0)
            + 1
    }
}

/// Box mean with radius `r` (window clipped to the image, divided by the pixel count).
pub fn box_mean(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut tmp = vec![0.0f32; w * h];
    let mut pre = vec![0.0f64; w.max(h) + 1];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        for x in 0..w {
            pre[x + 1] = pre[x] + row[x] as f64;
        }
        for x in 0..w {
            let (lo, hi) = (x.saturating_sub(r), (x + r + 1).min(w));
            tmp[y * w + x] = ((pre[hi] - pre[lo]) / (hi - lo) as f64) as f32;
        }
    }
    let mut out = vec![0.0f32; w * h];
    let mut col = vec![0.0f64; (h + 1) * w];
    for y in 0..h {
        for x in 0..w {
            col[(y + 1) * w + x] = col[y * w + x] + tmp[y * w + x] as f64;
        }
    }
    for y in 0..h {
        let (lo, hi) = (y.saturating_sub(r), (y + r + 1).min(h));
        let n = (hi - lo) as f64;
        for x in 0..w {
            out[y * w + x] = ((col[hi * w + x] - col[lo * w + x]) / n) as f32;
        }
    }
    out
}

/// Guided filter with a grayscale guide.
pub fn guided_filter_gray(guide: &[f32], p: &[f32], w: usize, h: usize, r: usize, eps: f32) -> Vec<f32> {
    let mi = box_mean(guide, w, h, r);
    let mp = box_mean(p, w, h, r);
    let ip: Vec<f32> = guide.iter().zip(p).map(|(a, b)| a * b).collect();
    let ii: Vec<f32> = guide.iter().map(|a| a * a).collect();
    let mip = box_mean(&ip, w, h, r);
    let mii = box_mean(&ii, w, h, r);
    let mut a = vec![0.0f32; w * h];
    let mut b = vec![0.0f32; w * h];
    for i in 0..w * h {
        let var = mii[i] - mi[i] * mi[i];
        let cov = mip[i] - mi[i] * mp[i];
        a[i] = cov / (var + eps);
        b[i] = mp[i] - a[i] * mi[i];
    }
    let ma = box_mean(&a, w, h, r);
    let mb = box_mean(&b, w, h, r);
    (0..w * h).map(|i| ma[i] * guide[i] + mb[i]).collect()
}

/// Guided filter with a colour guide (He et al., Eq. 19–21: `a_k = (Σ_k + εU)⁻¹ cov_k(I, p)`).
pub fn guided_filter_color(guide: &RgbImage, p: &[f32], r: usize, eps: f32) -> Vec<f32> {
    let (w, h) = (guide.w, guide.h);
    let n = w * h;
    let ch = |c: usize| -> Vec<f32> { guide.px.iter().map(|q| q[c]).collect() };
    let chans = [ch(0), ch(1), ch(2)];
    let m: Vec<Vec<f32>> = chans.iter().map(|c| box_mean(c, w, h, r)).collect();
    let mp = box_mean(p, w, h, r);
    let mip: Vec<Vec<f32>> = chans.iter().map(|c| box_mean(&c.iter().zip(p).map(|(a, b)| a * b).collect::<Vec<_>>(), w, h, r)).collect();
    let pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
    let mii: Vec<Vec<f32>> = pairs.iter().map(|&(a, b)| box_mean(&chans[a].iter().zip(&chans[b]).map(|(x, y)| x * y).collect::<Vec<_>>(), w, h, r)).collect();
    let mut av = [vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]];
    let mut bv = vec![0.0f32; n];
    for i in 0..n {
        let mu = [m[0][i] as f64, m[1][i] as f64, m[2][i] as f64];
        let s = |k: usize, a: usize, b: usize| mii[k][i] as f64 - mu[a] * mu[b];
        let e = eps as f64;
        let (rr, rg, rb, gg, gb, bb) = (s(0, 0, 0) + e, s(1, 0, 1), s(2, 0, 2), s(3, 1, 1) + e, s(4, 1, 2), s(5, 2, 2) + e);
        let cov = [mip[0][i] as f64 - mu[0] * mp[i] as f64, mip[1][i] as f64 - mu[1] * mp[i] as f64, mip[2][i] as f64 - mu[2] * mp[i] as f64];
        // Inverse of the symmetric 3×3 matrix.
        let i00 = gg * bb - gb * gb;
        let i01 = gb * rb - rg * bb;
        let i02 = rg * gb - gg * rb;
        let i11 = rr * bb - rb * rb;
        let i12 = rb * rg - rr * gb;
        let i22 = rr * gg - rg * rg;
        let det = rr * i00 + rg * i01 + rb * i02;
        let (a0, a1, a2) = if det.abs() < 1e-30 {
            (0.0, 0.0, 0.0)
        } else {
            (
                (i00 * cov[0] + i01 * cov[1] + i02 * cov[2]) / det,
                (i01 * cov[0] + i11 * cov[1] + i12 * cov[2]) / det,
                (i02 * cov[0] + i12 * cov[1] + i22 * cov[2]) / det,
            )
        };
        av[0][i] = a0 as f32;
        av[1][i] = a1 as f32;
        av[2][i] = a2 as f32;
        bv[i] = (mp[i] as f64 - a0 * mu[0] - a1 * mu[1] - a2 * mu[2]) as f32;
    }
    let ma: Vec<Vec<f32>> = av.iter().map(|a| box_mean(a, w, h, r)).collect();
    let mb = box_mean(&bv, w, h, r);
    (0..n).map(|i| ma[0][i] * chans[0][i] + ma[1][i] * chans[1][i] + ma[2][i] * chans[2][i] + mb[i]).collect()
}

/// Separable Gaussian blur (edge-clamped).
pub fn gaussian_blur(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma < 0.1 {
        return src.to_vec();
    }
    let r = (sigma * 3.0).ceil() as i64;
    let k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = k.iter().sum();
    let k: Vec<f32> = k.iter().map(|v| v / s).collect();
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut a = 0.0;
            for (i, kv) in k.iter().enumerate() {
                let xx = (x as i64 + i as i64 - r).clamp(0, w as i64 - 1) as usize;
                a += src[y * w + xx] * kv;
            }
            tmp[y * w + x] = a;
        }
    }
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut a = 0.0;
            for (i, kv) in k.iter().enumerate() {
                let yy = (y as i64 + i as i64 - r).clamp(0, h as i64 - 1) as usize;
                a += tmp[yy * w + x] * kv;
            }
            out[y * w + x] = a;
        }
    }
    out
}

/// Grey dilation (`grow`) or erosion with the octagonal element of `r` alternating
/// 3×3 square/cross steps. Masks without NaNs or negative zero use O(w*h*log(r+1))
/// work; those values retain the original comparison order. Invalid sizes return an empty buffer.
pub fn morph(src: &[f32], w: usize, h: usize, r: usize, grow: bool) -> Vec<f32> {
    morphology::morph(src, w, h, r, grow)
}

mod brush;
mod morphology;

pub use brush::{EdgeBrush, EdgeStroke, MAX_SWEPT_AREA, SIZE_RANGE as BRUSH_SIZE_RANGE, swept_area};

/// Running max (or min) over a window of radius `r` (edge-clipped), van Herk / Gil–Werman:
/// three comparisons per sample whatever the radius.
fn running_extreme(src: &[f32], out: &mut [f32], r: usize, max: bool, g: &mut Vec<f32>, hb: &mut Vec<f32>) {
    let n = src.len();
    let k = 2 * r + 1;
    let f = |a: f32, b: f32| if max { a.max(b) } else { a.min(b) };
    let pad = if max { f32::NEG_INFINITY } else { f32::INFINITY };
    let m = (n + 2 * r).div_ceil(k) * k;
    g.clear();
    g.resize(m, pad);
    hb.clear();
    hb.resize(m, pad);
    let p = |j: usize| if j >= r && j < r + n { src[j - r] } else { pad };
    for j in 0..m {
        g[j] = if j % k == 0 { p(j) } else { f(g[j - 1], p(j)) };
    }
    for j in (0..m).rev() {
        hb[j] = if j % k == k - 1 || j == m - 1 { p(j) } else { f(hb[j + 1], p(j)) };
    }
    for (i, o) in out.iter_mut().enumerate() {
        *o = f(hb[i], g[i + 2 * r]);
    }
}

/// Running max (or min) over a square window of radius `r` (edge-clipped), separable.
fn max_square(src: &[f32], w: usize, h: usize, r: usize, max: bool) -> Vec<f32> {
    let (mut g, mut hb) = (Vec::new(), Vec::new());
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        running_extreme(&src[y * w..(y + 1) * w], &mut tmp[y * w..(y + 1) * w], r.min(w.saturating_sub(1)), max, &mut g, &mut hb);
    }
    let mut out = vec![0.0f32; w * h];
    let (mut col, mut res) = (vec![0.0f32; h], vec![0.0f32; h]);
    for x in 0..w {
        for y in 0..h {
            col[y] = tmp[y * w + x];
        }
        running_extreme(&col, &mut res, r.min(h.saturating_sub(1)), max, &mut g, &mut hb);
        for y in 0..h {
            out[y * w + x] = res[y];
        }
    }
    out
}

/// Per-pixel transition width (pixels) of the guide's luminance: local range / local max
/// gradient over a window of radius `r` (a sharp step gives ~2, a ramp of width L gives ~L).
pub fn edge_width(img: &RgbImage, r: usize) -> Vec<f32> {
    let (w, h) = (img.w, img.h);
    let y: Vec<f32> = img.px.iter().map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]).collect();
    let mut g = vec![0.0f32; w * h];
    for yy in 0..h {
        for xx in 0..w {
            let at = |x: usize, y2: usize| y[y2 * w + x];
            let gx = (at((xx + 1).min(w - 1), yy) - at(xx.saturating_sub(1), yy)) / 2.0;
            let gy = (at(xx, (yy + 1).min(h - 1)) - at(xx, yy.saturating_sub(1))) / 2.0;
            g[yy * w + xx] = (gx * gx + gy * gy).sqrt();
        }
    }
    let gm = max_square(&g, w, h, r, true);
    let hi = max_square(&y, w, h, r, true);
    let lo = max_square(&y, w, h, r, false);
    (0..w * h).map(|i| (hi[i] - lo[i]) / gm[i].max(1e-4)).collect()
}

/// Refines a soft mask buffer `m` (`w × h`) with `img` as guide (required when `radius > 0`).
pub fn refine_buffer(img: Option<&RgbImage>, m: &[f32], w: usize, h: usize, p: &RefineParams) -> Vec<f32> {
    refine_buffer_with(img, m, w, h, p, None, morph)
}

/// The Refine Edge Brush over one buffer: signed coverage (`w × h`, see [`EdgeBrush`]) and the
/// box radius of its colour estimate.
#[derive(Clone, Copy)]
struct BrushBuf<'a> {
    v: &'a [f32],
    window: usize,
}

fn refine_buffer_with<F>(img: Option<&RgbImage>, m: &[f32], w: usize, h: usize, p: &RefineParams, brush: Option<BrushBuf>, morphology: F) -> Vec<f32>
where
    F: Fn(&[f32], usize, usize, usize, bool) -> Vec<f32> + Copy,
{
    let mut a = m.to_vec();
    if let (true, Some(img)) = (p.uses_image(), img) {
        let bin: Vec<bool> = m.iter().map(|v| *v >= 0.5).collect();
        let mut boundary = vec![false; w * h];
        let mut any = false;
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let v = bin[i];
                let diff = (x > 0 && bin[i - 1] != v) || (x + 1 < w && bin[i + 1] != v) || (y > 0 && bin[i - w] != v) || (y + 1 < h && bin[i + w] != v);
                boundary[i] = diff;
                any |= diff;
            }
        }
        if any {
            let dist = edt(&boundary, w, h);
            let r = p.gf_radius();
            let q = guided_filter_color(img, m, r, EPS);
            let widths = p.smart_radius.then(|| edge_width(img, r));
            for i in 0..w * h {
                let reff = match &widths {
                    Some(wd) => wd[i].clamp(1.5, p.radius.max(1.5)),
                    None => p.radius,
                };
                let mut wgt = (reff + 1.0 - dist[i]).clamp(0.0, 1.0);
                // Restored by the brush: the original edge.
                if let Some(b) = brush.and_then(|b| b.v.get(i)).filter(|b| **b < 0.0) {
                    wgt *= 1.0 + b;
                }
                if wgt > 0.0 {
                    a[i] = m[i] + (q[i].clamp(0.0, 1.0) - m[i]) * wgt;
                }
            }
        }
    }
    // Added by the brush: coverage estimated from the image.
    if let (Some(b), Some(img)) = (brush.filter(|b| b.v.iter().any(|v| *v > 0.0)), img) {
        let est = brush::estimate(img, m, b.v, w, h, b.window);
        for ((ai, ei), bi) in a.iter_mut().zip(&est).zip(b.v) {
            if *bi > 0.0 {
                *ai += (ei - *ai) * bi;
            }
        }
    }
    let ss = p.smooth_sigma();
    if ss >= 0.3 {
        a = gaussian_blur(&a, w, h, ss);
        let gain = 1.0 + ss;
        for v in &mut a {
            *v = ((*v - 0.5) * gain + 0.5).clamp(0.0, 1.0);
        }
    }
    a = gaussian_blur(&a, w, h, p.feather_sigma());
    if p.contrast > 0.0 {
        let gain = 1.0 / (1.0 - 0.99 * p.contrast.clamp(0.0, 100.0) / 100.0);
        for v in &mut a {
            *v = ((*v - 0.5) * gain + 0.5).clamp(0.0, 1.0);
        }
    }
    let amount = p.shift_amount();
    if amount.abs() >= 1.0e-4 {
        a = shift_mask(&a, w, h, amount, morphology);
    }
    a
}

/// Grow or shrink `src` by a fractional pixel distance. Whole pixels use one morphology;
/// the remainder blends that result with the next pixel, so 0.3 px is not the same step as 1 px.
fn shift_mask<F>(src: &[f32], w: usize, h: usize, amount: f32, morphology: F) -> Vec<f32>
where
    F: Fn(&[f32], usize, usize, usize, bool) -> Vec<f32> + Copy,
{
    let mag = amount.abs();
    if !mag.is_finite() || mag < 1.0e-4 {
        return src.to_vec();
    }
    let grow = amount > 0.0;
    let whole = mag.floor() as usize;
    let frac = mag - whole as f32;
    if frac < 1.0e-4 {
        return morphology(src, w, h, whole, grow);
    }
    let lo = if whole == 0 { src.to_vec() } else { morphology(src, w, h, whole, grow) };
    let hi = morphology(src, w, h, whole + 1, grow);
    let n = w.saturating_mul(h);
    if lo.len() < n || hi.len() < n || lo.len() != hi.len() {
        return hi;
    }
    lo.into_iter().zip(hi).map(|(a, b)| a + (b - a) * frac).collect()
}

/// Refines a selection mask. `mask(r)` returns coverage over `r` (row-major); `content` bounds
/// its non-zero pixels; the result is clipped to `canvas`. `sampler` supplies the guide image
/// (only read when `radius > 0`). Returns `None` when nothing remains selected.
pub fn refine_mask(sampler: &dyn Sampler, mask: &(dyn Fn(Rect) -> Vec<f32> + Sync), content: Rect, canvas: Rect, p: &RefineParams) -> Option<Region> {
    refine_mask_with(sampler, mask, content, canvas, p, None, morph)
}

/// [`refine_mask`] with the Refine Edge Brush: where `brush` adds, coverage comes from the image
/// (the area can reach beyond the edge-detection band); where it restores, the original edge
/// stays. The sampler is read wherever the brush adds, whatever the radius.
pub fn refine_mask_brushed(
    sampler: &dyn Sampler,
    mask: &(dyn Fn(Rect) -> Vec<f32> + Sync),
    content: Rect,
    canvas: Rect,
    p: &RefineParams,
    brush: Option<&EdgeBrush>,
) -> Option<Region> {
    refine_mask_with(sampler, mask, content, canvas, p, brush, morph)
}

fn refine_mask_with<F>(
    sampler: &dyn Sampler,
    mask: &(dyn Fn(Rect) -> Vec<f32> + Sync),
    content: Rect,
    canvas: Rect,
    p: &RefineParams,
    brush: Option<&EdgeBrush>,
    morphology: F,
) -> Option<Region>
where
    F: Fn(&[f32], usize, usize, usize, bool) -> Vec<f32> + Copy + Sync + Send,
{
    let content = content.intersect(&canvas);
    if content.is_empty() {
        return None;
    }
    let bbox = content.inflate(p.spread()).union(&brush.map_or(Rect::EMPTY, EdgeBrush::added_bounds)).intersect(&canvas);
    let halo = p.halo() + brush.map_or(0, |b| i32::try_from(b.window() * 4).unwrap_or(i32::MAX));
    let mut tiles = Vec::new();
    let mut y = bbox.y0;
    while y < bbox.y1 {
        let mut x = bbox.x0;
        while x < bbox.x1 {
            tiles.push(Rect::new(x, y, (x + TILE).min(bbox.x1), (y + TILE).min(bbox.y1)));
            x += TILE;
        }
        y += TILE;
    }
    let run = |t: &Rect| -> (Rect, Vec<u8>) {
        let r = t.inflate(halo).intersect(&canvas);
        let m = mask(r);
        let (rw, rh) = (r.width() as usize, r.height() as usize);
        let first = m.first().copied().unwrap_or(0.0);
        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        let (cx, cy, cw, ch) = ((t.x0 - r.x0) as usize, (t.y0 - r.y0) as usize, t.width() as usize, t.height() as usize);
        let core = |buf: &[f32]| -> Vec<u8> {
            let mut out = Vec::with_capacity(cw * ch);
            for yy in 0..ch {
                out.extend(buf[(cy + yy) * rw + cx..(cy + yy) * rw + cx + cw].iter().map(|v| q(*v)));
            }
            out
        };
        let painted = brush.filter(|b| !b.bounds().intersect(&r).is_empty()).map(|b| b.read(r));
        let adds = painted.as_ref().is_some_and(|v| v.iter().any(|v| *v > 0.0));
        if !adds && (first == 0.0 || first == 1.0) && m.iter().all(|v| *v == first) {
            return (*t, vec![q(first); cw * ch]);
        }
        let img = (p.uses_image() || adds).then(|| sampler.rgb(r));
        let bb = match (&painted, brush) {
            (Some(v), Some(b)) => Some(BrushBuf { v, window: b.window() }),
            _ => None,
        };
        let out = refine_buffer_with(img.as_ref(), &m, rw, rh, p, bb, morphology);
        (*t, core(&out))
    };
    #[cfg(not(target_arch = "wasm32"))]
    let results: Vec<(Rect, Vec<u8>)> = {
        use rayon::prelude::*;
        tiles.par_iter().map(run).collect()
    };
    #[cfg(target_arch = "wasm32")]
    let results: Vec<(Rect, Vec<u8>)> = tiles.iter().map(run).collect();
    let bw = bbox.width() as usize;
    let mut out = vec![0u8; bw * bbox.height() as usize];
    for (t, data) in results {
        let cw = t.width() as usize;
        for (yy, row) in data.chunks_exact(cw).enumerate() {
            let o = ((t.y0 - bbox.y0) as usize + yy) * bw + (t.x0 - bbox.x0) as usize;
            out[o..o + cw].copy_from_slice(row);
        }
    }
    crate::segment::trim_region(Region { bbox, mask: out })
}

/// A mask reader over a [`Region`] (zero outside its box).
pub fn region_reader(reg: &Region) -> impl Fn(Rect) -> Vec<f32> + Sync + '_ {
    move |r: Rect| {
        let mut v = Vec::with_capacity(r.width() as usize * r.height() as usize);
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                v.push(reg.at(x, y));
            }
        }
        v
    }
}

/// A mask reader over a selection surface (channel 0).
pub fn surface_reader(s: &Surface) -> impl Fn(Rect) -> Vec<f32> + Sync + '_ {
    move |r: Rect| crate::selection::mask_from_surface(Some(s), r)
}

/// GRAY8 surface (default 0) holding a region's coverage (a selection or a layer mask).
pub fn region_surface(reg: &Region) -> Surface {
    let mut s = Surface::from_interleaved(PixelFormat::GRAY8, reg.bbox, &reg.mask);
    s.prune();
    s
}

fn tiles_of(area: Rect) -> Vec<Rect> {
    let mut tiles = Vec::new();
    let mut y = area.y0;
    while y < area.y1 {
        let mut x = area.x0;
        while x < area.x1 {
            tiles.push(Rect::new(x, y, (x + TILE).min(area.x1), (y + TILE).min(area.y1)));
            x += TILE;
        }
        y += TILE;
    }
    tiles
}

fn par_tiles<T: Send>(tiles: &[Rect], f: impl Fn(&Rect) -> Option<T> + Sync + Send) -> Vec<T> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        tiles.par_iter().filter_map(f).collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        tiles.iter().filter_map(f).collect()
    }
}

/// Decontaminates colours in the soft fringe of `alpha`: each fringe pixel moves towards the
/// average colour of nearby fully selected pixels (normalised convolution with a box of radius
/// `max(radius, 3)`, widened ×4 where no such pixel is near), by `amount`% scaled by its
/// transparency. Works in the surface's own colour channels (any model / depth).
pub fn decontaminate(src: &Surface, alpha: &Region, radius: f32, amount: f32) -> Surface {
    let fmt = src.format();
    let n = fmt.channels();
    let nc = n - fmt.alpha as usize;
    let rad = radius.max(3.0).ceil() as usize;
    let halo = (rad * 4) as i32;
    let amount = amount.clamp(0.0, 100.0) / 100.0;
    let soft = |a: f32| a > 0.02 && a < 0.98;
    let results = par_tiles(&tiles_of(alpha.bbox), |t| {
        if !(t.y0..t.y1).any(|y| (t.x0..t.x1).any(|x| soft(alpha.at(x, y)))) {
            return None;
        }
        let r = t.inflate(halo);
        let (rw, rh) = (r.width() as usize, r.height() as usize);
        let mut px = Vec::new();
        src.read_region_into(r, &mut px);
        let al: Vec<f32> = (r.y0..r.y1).flat_map(|y| (r.x0..r.x1).map(move |x| alpha.at(x, y))).collect();
        let wts: Vec<f32> = (0..rw * rh).map(|i| if al[i] >= 0.98 { if fmt.alpha { px[i * n + n - 1] } else { 1.0 } } else { 0.0 }).collect();
        let den = box_mean(&wts, rw, rh, rad);
        let den2 = box_mean(&wts, rw, rh, rad * 4);
        let mut nums = Vec::with_capacity(nc);
        for c in 0..nc {
            let v: Vec<f32> = (0..rw * rh).map(|i| wts[i] * px[i * n + c]).collect();
            nums.push((box_mean(&v, rw, rh, rad), box_mean(&v, rw, rh, rad * 4)));
        }
        let (cx, cy, cw, ch) = ((t.x0 - r.x0) as usize, (t.y0 - r.y0) as usize, t.width() as usize, t.height() as usize);
        let mut out = Vec::with_capacity(cw * ch * n);
        for yy in 0..ch {
            for xx in 0..cw {
                let i = (cy + yy) * rw + cx + xx;
                let a = al[i];
                let k = if soft(a) { amount * ((1.0 - a) * 4.0).clamp(0.0, 1.0) } else { 0.0 };
                for c in 0..n {
                    let v = px[i * n + c];
                    if c >= nc || k == 0.0 {
                        out.push(v);
                        continue;
                    }
                    let f = if den[i] > 1e-4 {
                        nums[c].0[i] / den[i]
                    } else if den2[i] > 1e-4 {
                        nums[c].1[i] / den2[i]
                    } else {
                        v
                    };
                    out.push(v + (f - v) * k);
                }
            }
        }
        Some((*t, out))
    });
    let mut out = src.clone();
    for (t, data) in results {
        out.write_region(t, &data);
    }
    out
}

/// The pixels of `src` with their alpha multiplied by `alpha` (zero outside its box); the result
/// has an alpha channel even if `src` does not.
pub fn masked_copy(src: &Surface, alpha: &Region) -> Surface {
    let fmt = src.format();
    let n = fmt.channels();
    let out_fmt = PixelFormat { alpha: true, ..fmt };
    let m = out_fmt.channels();
    let results = par_tiles(&tiles_of(alpha.bbox), |t| {
        let mut px = Vec::new();
        src.read_region_into(*t, &mut px);
        let mut out = Vec::with_capacity(t.width() as usize * t.height() as usize * m);
        let mut any = false;
        let mut i = 0;
        for y in t.y0..t.y1 {
            for x in t.x0..t.x1 {
                let a = alpha.at(x, y);
                let p = &px[i * n..(i + 1) * n];
                out.extend_from_slice(&p[..m - 1]);
                let sa = if fmt.alpha { p[n - 1] } else { 1.0 };
                out.push(sa * a);
                any |= sa * a > 0.0;
                i += 1;
            }
        }
        any.then_some((*t, out))
    });
    let mut out = Surface::new(out_fmt);
    for (t, data) in results {
        out.write_region(t, &data);
    }
    out.prune();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::ImageSampler;
    use photocraft_color::SampleType;

    /// Horizontal blend from `a` (left) to `b` (right) across a Gaussian-blurred edge at x = 40.
    fn blurred_edge(w: usize, h: usize, sigma: f32) -> (RgbImage, Vec<f32>) {
        let (a, b) = ([0.9f32, 0.8, 0.2], [0.1f32, 0.2, 0.6]);
        // True foreground fraction (a = foreground) = 1 − Φ((x − 40) / σ).
        let erf = |x: f32| {
            let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
            let y = 1.0 - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_7) * t - 0.284_496_74) * t + 0.254_829_6) * t * (-x * x).exp();
            if x >= 0.0 { y } else { -y }
        };
        let alpha: Vec<f32> = (0..w).map(|x| 0.5 * (1.0 - erf((x as f32 + 0.5 - 40.0) / (sigma * std::f32::consts::SQRT_2)))).collect();
        let img = RgbImage::from_fn(w, h, |x, _| [0, 1, 2].map(|c| a[c] * alpha[x] + b[c] * (1.0 - alpha[x])));
        (img, alpha)
    }

    #[test]
    fn max_square_matches_naive() {
        let (w, h) = (17, 11);
        let src: Vec<f32> = (0..w * h).map(|i| ((i * 37) % 23) as f32).collect();
        for r in [1usize, 2, 5] {
            for max in [true, false] {
                let fast = max_square(&src, w, h, r, max);
                for y in 0..h {
                    for x in 0..w {
                        let mut v = if max { f32::MIN } else { f32::MAX };
                        for yy in y.saturating_sub(r)..(y + r + 1).min(h) {
                            for xx in x.saturating_sub(r)..(x + r + 1).min(w) {
                                v = if max { v.max(src[yy * w + xx]) } else { v.min(src[yy * w + xx]) };
                            }
                        }
                        assert_eq!(fast[y * w + x], v, "r={r} max={max} ({x},{y})");
                    }
                }
            }
        }
    }

    #[test]
    fn box_mean_matches_naive() {
        let (w, h) = (13, 9);
        let src: Vec<f32> = (0..w * h).map(|i| ((i * 7) % 11) as f32).collect();
        let fast = box_mean(&src, w, h, 2);
        for y in 0..h {
            for x in 0..w {
                let (mut s, mut n) = (0.0, 0.0);
                for yy in y.saturating_sub(2)..(y + 3).min(h) {
                    for xx in x.saturating_sub(2)..(x + 3).min(w) {
                        s += src[yy * w + xx];
                        n += 1.0;
                    }
                }
                assert!((fast[y * w + x] - s / n).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn guided_filter_matting_softens_blurred_edge() {
        let (w, h) = (80usize, 20usize);
        let (img, truth) = blurred_edge(w, h, 3.0);
        let m: Vec<f32> = (0..w * h).map(|i| if (i % w) < 40 { 1.0 } else { 0.0 }).collect();
        let p = RefineParams { radius: 10.0, ..Default::default() };
        let a = refine_buffer(Some(&img), &m, w, h, &p);
        let row = &a[10 * w..11 * w];
        let soft = row.iter().filter(|v| **v > 0.05 && **v < 0.95).count();
        assert!(soft >= 6, "only {soft} soft pixels: {row:?}");
        for x in 0..w {
            assert!((row[x] - truth[x]).abs() < 0.12, "x={x}: {} vs {}", row[x], truth[x]);
            if x > 0 {
                assert!(row[x] <= row[x - 1] + 0.02, "not monotone at {x}");
            }
        }
        // Outside the band the mask is untouched.
        assert_eq!(row[5], 1.0);
        assert_eq!(row[75], 0.0);
        // Smart radius keeps a sharp edge sharp.
        let sharp = RgbImage::from_fn(w, h, |x, _| if x < 40 { [0.9, 0.8, 0.2] } else { [0.1, 0.2, 0.6] });
        let s = refine_buffer(Some(&sharp), &m, w, h, &RefineParams { radius: 10.0, smart_radius: true, ..Default::default() });
        let soft = s[10 * w..11 * w].iter().filter(|v| **v > 0.05 && **v < 0.95).count();
        assert!(soft <= 2, "sharp edge became soft: {soft}");
    }

    #[test]
    fn feather_contrast_and_shift() {
        let (w, h) = (60usize, 10usize);
        let m: Vec<f32> = (0..w * h).map(|i| if (i % w) < 30 { 1.0 } else { 0.0 }).collect();
        let f = refine_buffer(None, &m, w, h, &RefineParams { feather: 6.0, ..Default::default() });
        assert!(f[5 * w + 29] > 0.5 && f[5 * w + 29] < 0.8 && f[5 * w + 31] > 0.1);
        let c = refine_buffer(None, &f, w, h, &RefineParams { contrast: 100.0, ..Default::default() });
        assert!(c[5 * w + 29] == 1.0 && c[5 * w + 31] == 0.0);
        let grown = refine_buffer(None, &m, w, h, &RefineParams { shift_edge: 100.0, ..Default::default() });
        assert_eq!(grown[5 * w + 33], 1.0);
        assert_eq!(grown[5 * w + 34], 0.0);
        let shrunk = refine_buffer(None, &m, w, h, &RefineParams { shift_edge: -50.0, ..Default::default() });
        assert_eq!(shrunk[5 * w + 27], 1.0);
        assert_eq!(shrunk[5 * w + 28], 0.0);
        // Radius 0 scales 100% to 4 px, so 12.5% is half a pixel and 25% is one pixel.
        // The pixel just outside a hard edge fades in instead of snapping on.
        let half = refine_buffer(None, &m, w, h, &RefineParams { shift_edge: 12.5, ..Default::default() });
        let one = refine_buffer(None, &m, w, h, &RefineParams { shift_edge: 25.0, ..Default::default() });
        assert_eq!(m[5 * w + 30], 0.0);
        assert!((half[5 * w + 30] - 0.5).abs() < 0.02, "half pixel {}", half[5 * w + 30]);
        assert_eq!(one[5 * w + 30], 1.0);
        assert_eq!(half[5 * w + 31], 0.0, "the blend must not jump a second pixel");
        let pulled = refine_buffer(None, &m, w, h, &RefineParams { shift_edge: -12.5, ..Default::default() });
        assert!((pulled[5 * w + 29] - 0.5).abs() < 0.02, "half-pixel contract {}", pulled[5 * w + 29]);
        let sm = refine_buffer(None, &m, w, h, &RefineParams { smooth: 50.0, ..Default::default() });
        assert!(sm[5 * w + 10] == 1.0 && sm[5 * w + 50] == 0.0);
    }

    #[test]
    fn refine_mask_tiles_match_single_buffer() {
        // Tiled processing gives the same result as processing the whole image at once.
        let (w, h) = (600usize, 300usize);
        let (img, _) = blurred_edge(w, h, 2.0);
        let s = ImageSampler { img: &img, origin: (0, 0) };
        let mut mask = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f32 - 40.0, y as f32 - 150.0);
                mask[y * w + x] = if dx * dx * 0.05 + dy * dy < 120.0 * 120.0 || x < 40 { 255 } else { 0 };
            }
        }
        let reg = Region { bbox: Rect::new(0, 0, w as i32, h as i32), mask };
        let p = RefineParams { radius: 6.0, feather: 2.0, contrast: 10.0, ..Default::default() };
        let canvas = reg.bbox;
        let out = refine_mask(&s, &region_reader(&reg), canvas, canvas, &p).unwrap();
        let whole: Vec<f32> = reg.mask.iter().map(|v| *v as f32 / 255.0).collect();
        let full = refine_buffer(Some(&img), &whole, w, h, &p);
        let mut maxd = 0.0f32;
        for y in 0..h {
            for x in 0..w {
                maxd = maxd.max((out.at(x as i32, y as i32) - full[y * w + x]).abs());
            }
        }
        assert!(maxd <= 1.5 / 255.0, "tiles differ by {maxd}");
    }

    #[test]
    fn decontaminate_and_masked_copy_any_depth() {
        for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let fmt = PixelFormat::RGBA8.with_sample(sample);
            let mut s = Surface::new(fmt);
            // Foreground red on the left, background blue on the right; fringe pixel at x = 10 is a mix.
            s.fill_rect(Rect::new(0, 0, 10, 8), &[1.0, 0.0, 0.0, 1.0]);
            s.fill_rect(Rect::new(10, 0, 20, 8), &[0.5, 0.0, 0.5, 1.0]);
            s.fill_rect(Rect::new(11, 0, 20, 8), &[0.0, 0.0, 1.0, 1.0]);
            let mask: Vec<u8> = (0..20 * 8)
                .map(|i| match i % 20 {
                    x if x < 10 => 255,
                    10 => 128,
                    _ => 0,
                })
                .collect();
            let a = Region { bbox: Rect::new(0, 0, 20, 8), mask };
            let d = decontaminate(&s, &a, 3.0, 100.0);
            let p = d.pixel(10, 4);
            assert!(p[0] > 0.9 && p[2] < 0.1, "{sample:?}: {p:?}");
            assert_eq!(d.pixel(15, 4), s.pixel(15, 4));
            let c = masked_copy(&d, &a);
            assert_eq!(c.format(), fmt);
            assert!((c.pixel(10, 4)[3] - 128.0 / 255.0).abs() < 0.01);
            assert_eq!(c.pixel(15, 4)[3], 0.0);
            assert_eq!(c.pixel(3, 3)[0], 1.0);
        }
        // Sources without alpha gain an alpha channel.
        let mut g = Surface::new(PixelFormat::GRAY8);
        g.fill_rect(Rect::new(0, 0, 4, 4), &[0.5]);
        let a = Region { bbox: Rect::new(0, 0, 2, 4), mask: vec![255; 8] };
        let c = masked_copy(&g, &a);
        assert_eq!(c.format(), PixelFormat::GRAYA8);
        assert_eq!(c.pixel(3, 1)[1], 0.0);
        assert!((c.pixel(1, 1)[0] - 0.5).abs() < 0.01 && c.pixel(1, 1)[1] == 1.0);
    }
}
