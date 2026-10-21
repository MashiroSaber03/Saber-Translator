//! Quick Selection brush, behaving like Photoshop's.
//!
//! Photoshop 25.4 was measured as a black box: synthetic images and public-domain paintings,
//! strokes driven on its canvas, the exact selection masks read back (no Photoshop code or
//! assets were used). What it does:
//!
//! - The brush footprint is always selected, whatever colours it covers.
//! - Beyond the footprint the selection grows to image edges, but every selected pixel costs a
//!   little: a click selects a whole object only when the object is small enough to be worth it
//!   for that brush. With a 30 px brush a click selects a 220 px red square on blue but not a
//!   240 px one, a 120 x 400 bar but not a 160 x 400 one, and a weak edge (a step of 15 levels)
//!   only around a 100 px square. A click in a flat area selects about the brush disc.
//! - A stroke is one set of seeds: a short drag inside a square too big for a click selects it.
//! - Colours elsewhere in the image make no difference (no global colour model).
//! - It works on the whole document when it has at most 512 x 512 pixels and at exactly half
//!   resolution above (any size, any brush), aligned to the document's corner; the mask is then
//!   scaled up bilinearly (edges show a 25 % / 75 % step).
//!
//! The model reproducing this is one min cut (Boykov & Jolly, ICCV 2001; max-flow by Boykov &
//! Kolmogorov, PAMI 2004) over the working image minimising
//!
//! ```text
//!   sum over selected pixels p beyond the footprint of  AREA * (1 + DIST * d(p)^POW)
//!   + sum over cut neighbour pairs p~q of  w(|I_p - I_q|) / |p - q|
//!   w(D) = FLOOR + 1 / (1 + (D / SIGMA)^K)
//! ```
//!
//! with the (slightly grown) footprint hard foreground and `d` the distance from it. The
//! constants were fitted to Photoshop's masks.

use photocraft_geom::Rect;

use super::{FREE, HARD_FG, Region, RgbImage, Sampler};

/// The model's constants (see the module docs); [`Tuning::default`] is the fit to Photoshop.
#[derive(Clone, Copy, Debug)]
pub struct Tuning {
    /// Cost of each selected pixel beyond the footprint, times `1 + dist * d^pow` with `d` its
    /// distance from the footprint (working pixels).
    pub area: f32,
    pub dist: f32,
    pub pow: f32,
    /// Cut weight between neighbours: `floor + 1 / (1 + (|Ip - Iq| / sigma)^k)` (RGB in 0..1).
    pub floor: f32,
    pub sigma: f32,
    pub k: f32,
    /// The footprint grows by `grow + grow_frac * radius` working pixels (Photoshop's settled click).
    pub grow: f32,
    pub grow_frac: f32,
    /// Larger working windows are cut coarse to fine: at half size first, then again only in a
    /// band around that boundary (Lombaert, Sun, Grady & Xu, "A Multilevel Banded Graph Cuts
    /// Method for Fast Image Segmentation", ICCV 2005).
    pub direct_px: usize,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            area: 0.004079,
            dist: 0.002651,
            pow: 0.98885,
            floor: 0.018847,
            sigma: 0.048450,
            k: 1.98549,
            grow: 1.69371,
            grow_frac: 0.103003,
            direct_px: 300_000,
        }
    }
}

/// The working step for a document of `w` x `h` pixels: Photoshop works on the whole document
/// up to 512 x 512 pixels and at half resolution above.
pub fn working_step(w: usize, h: usize) -> usize {
    if w.saturating_mul(h) <= 512 * 512 { 1 } else { 2 }
}

/// Grows a selection from a brush stroke (`points` in document pixels, brush diameter `size`)
/// on the document `canvas`. Returns the new region (to be added to or subtracted from the
/// current selection), or `None` when the stroke misses the canvas.
pub fn quick_select(sampler: &dyn Sampler, canvas: Rect, points: &[(f32, f32)], size: f32) -> Option<Region> {
    quick_select_with(sampler, canvas, points, size, &Tuning::default())
}

/// [`quick_select`] with explicit model constants.
pub fn quick_select_with(sampler: &dyn Sampler, canvas: Rect, points: &[(f32, f32)], size: f32, t: &Tuning) -> Option<Region> {
    quick_select_using::<true>(sampler, canvas, points, size, t)
}

pub(super) fn quick_select_using<const OPTIMIZED: bool>(sampler: &dyn Sampler, canvas: Rect, points: &[(f32, f32)], size: f32, t: &Tuning) -> Option<Region> {
    quick_select_batched::<OPTIMIZED>(sampler, canvas, points, size, t, SPECULATE).map(|(region, _)| region)
}

/// [`quick_select_using`] cutting `batch` window sizes at a time (1: one after another). Also
/// returns how many window sizes, smallest first, it took to find the kept one.
pub(super) fn quick_select_batched<const OPTIMIZED: bool>(
    sampler: &dyn Sampler,
    canvas: Rect,
    points: &[(f32, f32)],
    size: f32,
    t: &Tuning,
    batch: usize,
) -> Option<(Region, usize)> {
    if points.is_empty() || canvas.is_empty() {
        return None;
    }
    let r = size.max(1.0) / 2.0;
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for &(x, y) in points {
        x0 = x0.min(x - r);
        y0 = y0.min(y - r);
        x1 = x1.max(x + r);
        y1 = y1.max(y + r);
    }
    let clamp = |v: f32| v.clamp(i32::MIN as f32 / 2.0, i32::MAX as f32 / 2.0) as i32;
    let bbox = Rect::new(clamp(x0.floor()), clamp(y0.floor()), clamp(x1.ceil()).saturating_add(1), clamp(y1.ceil()).saturating_add(1)).intersect(&canvas);
    if bbox.is_empty() {
        return None;
    }
    let step = working_step(canvas.width() as usize, canvas.height() as usize) as i32;
    let rw = r / step as f32;
    // Work in a window around the stroke; grow it while the selection runs into its border.
    // The window's sides are free like the canvas' edges, so a selection that stays clear of
    // them is kept; one that reaches a side inside the canvas is recomputed in a larger window.
    // Index of the smallest window kept so far in this batch: larger ones stop cutting.
    let kept = std::sync::atomic::AtomicUsize::new(usize::MAX);
    let attempt = |index: usize, margin: i32| -> Attempt {
        let cancel = || kept.load(std::sync::atomic::Ordering::Relaxed) < index;
        if cancel() {
            return Attempt::Cancelled;
        }
        let snap0 = |v: i32, o: i32| o + (v - o).div_euclid(step) * step;
        let grown = bbox.inflate(margin).intersect(&canvas);
        let window = Rect::new(snap0(grown.x0, canvas.x0), snap0(grown.y0, canvas.y0), grown.x1, grown.y1);
        let img = sampler.rgb_scaled(window, step as usize);
        let (lw, lh) = (img.w, img.h);
        let s = step as f32;
        let pts: Vec<(f32, f32)> = points.iter().map(|&(x, y)| ((x - window.x0 as f32) / s, (y - window.y0 as f32) / s)).collect();
        let seeds = stroke_mask(&pts, rw.max(0.5), lw, lh);
        if !seeds.iter().any(|v| *v) {
            return Attempt::NoSeeds;
        }
        if cancel() {
            return Attempt::Cancelled;
        }
        let inner = [window.x0 > canvas.x0, window.y0 > canvas.y0, window.x1 < canvas.x1, window.y1 < canvas.y1];
        let Some(low) = select_working_using::<OPTIMIZED>(&img, &seeds, rw, t, &cancel) else { return Attempt::Cancelled };
        let touches = |side: usize| match side {
            0 => (0..lh).any(|y| low[y * lw]),
            1 => (0..lw).any(|x| low[x]),
            2 => (0..lh).any(|y| low[y * lw + lw - 1]),
            _ => (0..lw).any(|x| low[(lh - 1) * lw + x]),
        };
        if (0..4).any(|k| inner[k] && touches(k)) {
            return Attempt::Touches;
        }
        kept.fetch_min(index, std::sync::atomic::Ordering::Relaxed);
        Attempt::Done { window, low, lw, lh }
    };
    let mut margin = (48.0f32).max(2.0 * rw) as i32 * step;
    let mut sizes = 0;
    loop {
        // The attempts don't depend on each other, so a few window sizes are cut at once and the
        // first in growing order that stays clear of its sides is kept: the same result as trying
        // them one after another, without waiting for the smaller ones to fail first. Once one is
        // kept, the larger ones still cutting stop, so a stroke settled by a small window costs
        // about what that window alone costs.
        let batch = margins_to_try(margin, bbox, canvas, batch);
        kept.store(usize::MAX, std::sync::atomic::Ordering::Relaxed);
        #[cfg(not(target_arch = "wasm32"))]
        let results: Vec<Attempt> = {
            use rayon::prelude::*;
            batch.par_iter().enumerate().map(|(i, &m)| attempt(i, m)).collect()
        };
        #[cfg(target_arch = "wasm32")]
        let results: Vec<Attempt> = batch.iter().enumerate().map(|(i, &m)| attempt(i, m)).collect();
        for result in results {
            sizes += 1;
            match result {
                Attempt::NoSeeds => return None,
                // Only a window larger than one already kept is cancelled, and that one comes
                // first in this order, so a cancelled attempt is never reached.
                Attempt::Touches | Attempt::Cancelled => continue,
                Attempt::Done { window, low, lw, lh } => {
                    let mask = upsample(&low, lw, lh, step as usize, window.width() as usize, window.height() as usize);
                    return super::trim_region(Region { bbox: window, mask }).map(|region| (region, sizes));
                }
            }
        }
        let &last = batch.last()?;
        margin = last.saturating_mul(2);
    }
}

/// One window size's outcome in [`quick_select_using`].
enum Attempt {
    /// The stroke misses the window.
    NoSeeds,
    /// The selection reaches a side of the window inside the canvas.
    Touches,
    /// A smaller window of the same batch was kept first.
    Cancelled,
    Done {
        window: Rect,
        low: Vec<bool>,
        lw: usize,
        lh: usize,
    },
}

/// Window sizes cut together: most strokes settle within this many doublings.
const SPECULATE: usize = 4;

/// The next `count` margins from `margin`, doubling, stopping at the first one whose window
/// already covers the whole canvas (a larger one would be the same window).
pub(super) fn margins_to_try(margin: i32, bbox: Rect, canvas: Rect, count: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(count.max(1));
    let mut m = margin;
    for _ in 0..count.max(1) {
        out.push(m);
        if bbox.inflate(m).intersect(&canvas) == canvas {
            break;
        }
        m = m.saturating_mul(2);
    }
    out
}

/// The model at working resolution: `seeds` (the brush footprint, `radius` working pixels) are
/// selected; returns the selection.
pub fn select_working(img: &RgbImage, seeds: &[bool], radius: f32, t: &Tuning) -> Vec<bool> {
    select_working_using::<true>(img, seeds, radius, t, &|| false).unwrap_or_default()
}

fn select_working_using<const OPTIMIZED: bool>(img: &RgbImage, seeds: &[bool], radius: f32, t: &Tuning, cancel: &dyn Fn() -> bool) -> Option<Vec<bool>> {
    let (w, h) = (img.w, img.h);
    let d = crate::selection::edt(seeds, w, h);
    let grow = (t.grow + t.grow_frac * radius).max(0.0);
    let cost_fg: Vec<f32> = d.iter().map(|d| t.area * (1.0 + t.dist * (d - grow).max(0.0).powf(t.pow))).collect();
    let fixed: Vec<u8> = d.iter().map(|d| if *d <= grow { HARD_FG } else { FREE }).collect();
    let (floor, sigma, k) = (t.floor, t.sigma.max(1e-4), t.k);
    let weight = move |d2: f32| floor + 1.0 / (1.0 + (d2.sqrt() / sigma).powf(k));
    if cancel() {
        return None;
    }
    cut_levels::<OPTIMIZED>(img, &fixed, &cost_fg, &weight, 1.0, t.direct_px.max(1024), cancel)
}

/// Band half-width (pixels) re-cut at each finer level.
const BAND: f32 = 3.0;

/// Min cut with only foreground costs, coarse to fine above `direct_px` pixels: the half-size
/// problem (colours averaged, pixel costs summed, cut weights doubled since a coarse side spans two
/// fine ones) gives a boundary, and the full-size cut is redone only within `BAND` of it. Seeds
/// stay foreground.
fn cut_levels<const OPTIMIZED: bool>(
    img: &RgbImage,
    fixed: &[u8],
    cost_fg: &[f32],
    weight: &dyn Fn(f32) -> f32,
    scale: f32,
    direct_px: usize,
    cancel: &dyn Fn() -> bool,
) -> Option<Vec<bool>> {
    let (w, h) = (img.w, img.h);
    let zeros = vec![0.0f32; w * h];
    if w * h <= direct_px || w < 8 || h < 8 {
        return super::grid_cut_impl::<OPTIMIZED>(img, cost_fg, &zeros, fixed, |d2| scale * weight(d2), cancel);
    }
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let coarse_img = img.downsample(2);
    let mut cfix = vec![FREE; cw * ch];
    let mut ccost = vec![0.0f32; cw * ch];
    for y in 0..h {
        for x in 0..w {
            let c = (y / 2) * cw + x / 2;
            ccost[c] += cost_fg[y * w + x];
            if fixed[y * w + x] == HARD_FG {
                cfix[c] = HARD_FG;
            }
        }
    }
    let coarse = cut_levels::<OPTIMIZED>(&coarse_img, &cfix, &ccost, weight, scale * 2.0, direct_px, cancel)?;
    if cancel() {
        return None;
    }
    let proj: Vec<bool> = (0..w * h).map(|i| coarse[(i / w / 2) * cw + (i % w) / 2]).collect();
    // Band: within BAND of the projected boundary.
    let edge: Vec<bool> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let v = proj[i];
            (x > 0 && proj[i - 1] != v) || (x + 1 < w && proj[i + 1] != v) || (y > 0 && proj[i - w] != v) || (y + 1 < h && proj[i + w] != v)
        })
        .collect();
    if !edge.iter().any(|e| *e) {
        return Some((0..w * h).map(|i| proj[i] || fixed[i] == HARD_FG).collect());
    }
    let near = crate::selection::edt(&edge, w, h);
    let banded: Vec<u8> = (0..w * h)
        .map(|i| {
            if fixed[i] == HARD_FG {
                HARD_FG
            } else if near[i] <= BAND {
                FREE
            } else if proj[i] {
                HARD_FG
            } else {
                super::HARD_BG
            }
        })
        .collect();
    super::grid_cut_impl::<OPTIMIZED>(img, cost_fg, &zeros, &banded, |d2| scale * weight(d2), cancel)
}

/// Pixels within `radius` (working pixels) of the polyline `pts`, plus each point's own pixel.
pub fn stroke_mask(pts: &[(f32, f32)], radius: f32, w: usize, h: usize) -> Vec<bool> {
    let mut m = vec![false; w * h];
    let segs: Vec<((f32, f32), (f32, f32))> = if pts.len() == 1 { vec![(pts[0], pts[0])] } else { pts.windows(2).map(|p| (p[0], p[1])).collect() };
    for (a, b) in segs {
        let x0 = ((a.0.min(b.0) - radius).floor().max(0.0)) as usize;
        let y0 = ((a.1.min(b.1) - radius).floor().max(0.0)) as usize;
        let x1 = ((a.0.max(b.0) + radius).ceil().max(0.0) as usize).min(w);
        let y1 = ((a.1.max(b.1) + radius).ceil().max(0.0) as usize).min(h);
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let len2 = dx * dx + dy * dy;
        for y in y0..y1 {
            for x in x0..x1 {
                let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                let t = if len2 > 0.0 { (((px - a.0) * dx + (py - a.1) * dy) / len2).clamp(0.0, 1.0) } else { 0.0 };
                let (qx, qy) = (a.0 + t * dx - px, a.1 + t * dy - py);
                if qx * qx + qy * qy <= radius * radius {
                    m[y * w + x] = true;
                }
            }
        }
    }
    for &(x, y) in pts {
        if x >= 0.0 && y >= 0.0 && (x as usize) < w && (y as usize) < h {
            m[y as usize * w + x as usize] = true;
        }
    }
    m
}

/// Bilinear upsampling (pixel centres aligned, edges clamped) of a working-resolution selection
/// to 8-bit coverage, as Photoshop scales its half-resolution result.
pub fn upsample(low: &[bool], lw: usize, lh: usize, step: usize, w: usize, h: usize) -> Vec<u8> {
    if step == 1 {
        return low.iter().map(|b| if *b { 255 } else { 0 }).collect();
    }
    let at = |x: i64, y: i64| -> f32 { if low[(y.clamp(0, lh as i64 - 1) as usize) * lw + x.clamp(0, lw as i64 - 1) as usize] { 1.0 } else { 0.0 } };
    let s = step as f32;
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        let fy = (y as f32 + 0.5) / s - 0.5;
        let (iy, ty) = (fy.floor() as i64, fy - fy.floor());
        for x in 0..w {
            let fx = (x as f32 + 0.5) / s - 0.5;
            let (ix, tx) = (fx.floor() as i64, fx - fx.floor());
            let v = (at(ix, iy) * (1.0 - tx) + at(ix + 1, iy) * tx) * (1.0 - ty) + (at(ix, iy + 1) * (1.0 - tx) + at(ix + 1, iy + 1) * tx) * ty;
            out[y * w + x] = (v * 255.0).round() as u8;
        }
    }
    out
}
