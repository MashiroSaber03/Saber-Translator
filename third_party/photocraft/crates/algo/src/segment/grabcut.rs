//! GrabCut and rectangle Object Selection.
//!
//! Implemented from C. Rother, V. Kolmogorov and A. Blake, "GrabCut: Interactive Foreground
//! Extraction using Iterated Graph Cuts", SIGGRAPH 2004: a trimap (definite / probable
//! foreground and background), 5-component full-covariance GMMs per side, and iterated
//! "assign components → learn GMMs → min cut" steps with the contrast-sensitive smoothness term
//! `γ e^{−β‖z_m − z_n‖²} / dist(m, n)` (γ = 50, β from the mean squared neighbour difference).

use photocraft_geom::Rect;
use photocraft_raster::{Cancelled, Interrupt};

use super::gmm::{DEFAULT_REG, Gmm};
use super::{FREE, HARD_BG, HARD_FG, Region, RgbImage, Sampler, contrast_beta, data_costs, grid_cut_cancellable, subsample};

/// Definite background.
pub const BG: u8 = 0;
/// Definite foreground.
pub const FG: u8 = 1;
/// Probable background.
pub const PR_BG: u8 = 2;
/// Probable foreground.
pub const PR_FG: u8 = 3;

/// GrabCut's smoothness weight.
pub const GAMMA: f32 = 50.0;
/// Components per GMM.
pub const K: usize = 5;
const MAX_SAMPLES: usize = 40_000;

fn is_fg(t: u8) -> bool {
    t == FG || t == PR_FG
}

/// Runs `iters` GrabCut iterations, updating the probable labels of `trimap` in place. Returns
/// the final (foreground, background) models, or `None` if either side has no pixels.
pub fn grabcut(img: &RgbImage, trimap: &mut [u8], iters: usize) -> Option<(Gmm, Gmm)> {
    grabcut_with(img, trimap, iters, &Interrupt::NONE).unwrap_or_default()
}

/// [`grabcut`] that stops with `Err(Cancelled)` once `ctl` is cancelled, asked each iteration
/// and inside its min cut. `trimap` is then partly updated.
pub fn grabcut_with(img: &RgbImage, trimap: &mut [u8], iters: usize, ctl: &Interrupt) -> Result<Option<(Gmm, Gmm)>, Cancelled> {
    let cancel = || ctl.cancelled();
    let beta = contrast_beta(img);
    let fixed: Vec<u8> = trimap
        .iter()
        .map(|t| match *t {
            FG => HARD_FG,
            BG => HARD_BG,
            _ => FREE,
        })
        .collect();
    let mut models: Option<(Gmm, Gmm)> = None;
    for _ in 0..iters.max(1) {
        ctl.check()?;
        let (mut fs, mut bs) = (Vec::new(), Vec::new());
        for (p, t) in img.px.iter().zip(trimap.iter()) {
            if is_fg(*t) { fs.push(*p) } else { bs.push(*p) }
        }
        let (fs, bs) = (subsample(&fs, MAX_SAMPLES), subsample(&bs, MAX_SAMPLES));
        let (fg, bg) = match &models {
            None => match (Gmm::fit(&fs, K, DEFAULT_REG), Gmm::fit(&bs, K, DEFAULT_REG)) {
                (Some(f), Some(b)) => (f, b),
                _ => return Ok(None),
            },
            Some((f, b)) => (f.refit(&fs).unwrap_or_else(|| f.clone()), b.refit(&bs).unwrap_or_else(|| b.clone())),
        };
        let (cf, cb) = data_costs(img, &fg, &bg);
        // `grid_cut`'s weight, with the cut stopping on cancel.
        let cut = grid_cut_cancellable(img, &cf, &cb, &fixed, |d2| GAMMA * (-beta * d2).exp(), &cancel).ok_or(Cancelled)?;
        let mut changed = 0usize;
        for (t, fgnd) in trimap.iter_mut().zip(&cut) {
            if *t == PR_BG || *t == PR_FG {
                let nt = if *fgnd { PR_FG } else { PR_BG };
                changed += (nt != *t) as usize;
                *t = nt;
            }
        }
        models = Some((fg, bg));
        if changed * 2000 < trimap.len() {
            break;
        }
    }
    Ok(models)
}

/// Object Selection (rectangle mode): GrabCut initialised from `rect` (outside = background,
/// inside = probable foreground). Runs on a working copy of about `work_px` pixels, then re-cuts
/// the boundary at full resolution. `canvas` bounds the analysis.
pub fn object_select(sampler: &dyn Sampler, canvas: Rect, rect: Rect, work_px: usize) -> Option<Region> {
    let rect = rect.intersect(&canvas);
    if rect.width() < 2 || rect.height() < 2 {
        return None;
    }
    let margin = ((rect.width().max(rect.height()) as f32 * 0.15).ceil() as i32).max(8);
    let window = rect.inflate(margin).intersect(&canvas);
    let step = super::scale_for(window, work_px);
    let img = sampler.rgb_scaled(window, step);
    let (lw, lh) = (img.w, img.h);
    let mut trimap = vec![BG; lw * lh];
    // Working pixel (x, y) covers doc pixels [x0 + x·step, …); its centre decides inside/outside.
    let inside = |x: usize, y: usize| {
        let (cx, cy) = (window.x0 + (x * step + step / 2) as i32, window.y0 + (y * step + step / 2) as i32);
        rect.contains(cx, cy)
    };
    let mut any_bg = false;
    for y in 0..lh {
        for x in 0..lw {
            if inside(x, y) {
                trimap[y * lw + x] = PR_FG;
            } else {
                any_bg = true;
            }
        }
    }
    if !any_bg {
        // The rectangle covers the whole canvas: use a thin outer ring as background.
        let ring = (lw.min(lh) / 50).max(1);
        for y in 0..lh {
            for x in 0..lw {
                if x < ring || y < ring || x + ring >= lw || y + ring >= lh {
                    trimap[y * lw + x] = BG;
                }
            }
        }
    }
    let (fg, bg) = grabcut(&img, &mut trimap, 8)?;
    let low: Vec<bool> = trimap.iter().map(|t| is_fg(*t)).collect();
    let low = super::clean_mask(&low, lw, lh, 0.1, 0.002);
    super::finish_region(sampler, window, step, &low, lw, lh, Some((&fg, &bg)))
}

/// Working-resolution budget (pixels) for [`object_at`]: small enough to run while the pointer
/// hovers.
pub const OBJECT_AT_WORK_PX: usize = 60_000;

/// Object Selection's Object Finder (hover / click): the object under document point `(x, y)`.
///
/// A heuristic, not a semantic segmenter (that is #2788): on a working-resolution copy of the
/// canvas the area around the point is grown over pixels of similar colour (CIELAB, bounded both
/// by the distance to the seed's running mean and by the local edge step), so it stops at the
/// object's outline. A region that grows over most of the canvas is the background: nothing is
/// found. Otherwise GrabCut refines it inside its padded bounding box (the grown area probable
/// foreground, the cursor definite foreground, outside the box definite background), the part
/// connected to the cursor is kept and the boundary is re-cut at full resolution, as the
/// rectangle mode does.
pub fn object_at(sampler: &dyn Sampler, canvas: Rect, x: f32, y: f32, work_px: usize) -> Option<Region> {
    if canvas.width() < 4 || canvas.height() < 4 || !x.is_finite() || !y.is_finite() {
        return None;
    }
    let (px, py) = (x.floor() as i32, y.floor() as i32);
    if !canvas.contains(px, py) {
        return None;
    }
    let step = super::scale_for(canvas, work_px.max(1024));
    let img = sampler.rgb_scaled(canvas, step);
    let (w, h) = (img.w, img.h);
    if w < 2 || h < 2 {
        return None;
    }
    // Lightly smoothed, so noise and texture don't stop the growth; edges survive.
    let lab: Vec<[f32; 3]> = box3(&box3(&img.px, w, h), w, h).iter().map(|c| super::slic::rgb_to_lab(*c)).collect();
    let sx = (((px - canvas.x0) as usize) / step).min(w - 1);
    let sy = (((py - canvas.y0) as usize) / step).min(h - 1);
    let grown = grow_similar(&lab, w, h, sx, sy);
    let count = grown.iter().filter(|g| **g).count();
    // Background (or a flat image): no single object under the cursor.
    if count * 10 > w * h * 6 || count < 4 {
        return None;
    }
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0usize, 0usize);
    for (i, g) in grown.iter().enumerate() {
        if *g {
            let (gx, gy) = (i % w, i / w);
            (x0, y0, x1, y1) = (x0.min(gx), y0.min(gy), x1.max(gx + 1), y1.max(gy + 1));
        }
    }
    let pad = ((x1 - x0).max(y1 - y0) / 6).max(2);
    let (bx0, by0, bx1, by1) = (x0.saturating_sub(pad), y0.saturating_sub(pad), (x1 + pad).min(w), (y1 + pad).min(h));
    let mut trimap = vec![BG; w * h];
    let mut any_bg = false;
    for yy in 0..h {
        for xx in 0..w {
            let i = yy * w + xx;
            let inside = xx >= bx0 && xx < bx1 && yy >= by0 && yy < by1;
            trimap[i] = if grown[i] {
                PR_FG
            } else if inside {
                PR_BG
            } else {
                any_bg = true;
                BG
            };
        }
    }
    if !any_bg {
        // The box is the whole canvas: the frame is background.
        for yy in 0..h {
            for xx in 0..w {
                if (xx == 0 || yy == 0 || xx + 1 == w || yy + 1 == h) && !(xx == sx && yy == sy) {
                    trimap[yy * w + xx] = BG;
                }
            }
        }
    }
    trimap[sy * w + sx] = FG;
    let (fg, bg) = grabcut(&img, &mut trimap, 5)?;
    let low: Vec<bool> = trimap.iter().map(|t| is_fg(*t)).collect();
    let mut seeds = vec![false; w * h];
    seeds[sy * w + sx] = true;
    let low = super::keep_seeded(&low, &seeds, w, h);
    let low = super::clean_mask(&low, w, h, 0.1, 0.01);
    if !low.iter().any(|v| *v) {
        return None;
    }
    super::finish_region(sampler, canvas, step, &low, w, h, Some((&fg, &bg)))
}

/// Region growing (4-connected) from `(sx, sy)` over pixels whose CIELAB colour is within a
/// distance of the seed area's running mean and whose step from the neighbour it was reached
/// from is small, so it stops at edges.
fn grow_similar(lab: &[[f32; 3]], w: usize, h: usize, sx: usize, sy: usize) -> Vec<bool> {
    const MEAN_DIST: f32 = 32.0;
    const EDGE_STEP: f32 = 10.0;
    let d = |a: [f32; 3], b: [f32; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    let mut out = vec![false; w * h];
    let Some(&seed) = lab.get(sy * w + sx) else { return out };
    let (mut sum, mut n) = ([seed[0] as f64, seed[1] as f64, seed[2] as f64], 1.0f64);
    let mut stack = vec![(sx, sy)];
    out[sy * w + sx] = true;
    while let Some((x, y)) = stack.pop() {
        let c = lab[y * w + x];
        let mean = [(sum[0] / n) as f32, (sum[1] / n) as f32, (sum[2] / n) as f32];
        let nbrs = [(x.wrapping_sub(1), y), (x + 1, y), (x, y.wrapping_sub(1)), (x, y + 1)];
        for (nx, ny) in nbrs {
            if nx >= w || ny >= h {
                continue;
            }
            let i = ny * w + nx;
            if out[i] {
                continue;
            }
            let q = lab[i];
            if d(q, c) <= EDGE_STEP && d(q, mean) <= MEAN_DIST {
                out[i] = true;
                sum = [sum[0] + q[0] as f64, sum[1] + q[1] as f64, sum[2] + q[2] as f64];
                n += 1.0;
                stack.push((nx, ny));
            }
        }
    }
    out
}

/// 3×3 box blur (clamped at the edges).
fn box3(px: &[[f32; 3]], w: usize, h: usize) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0f32; 3]; px.len()];
    for y in 0..h {
        for x in 0..w {
            let (mut acc, mut n) = ([0.0f32; 3], 0.0f32);
            for yy in y.saturating_sub(1)..(y + 2).min(h) {
                for xx in x.saturating_sub(1)..(x + 2).min(w) {
                    if let Some(p) = px.get(yy * w + xx) {
                        acc = [acc[0] + p[0], acc[1] + p[1], acc[2] + p[2]];
                        n += 1.0;
                    }
                }
            }
            if let Some(o) = out.get_mut(y * w + x) {
                *o = acc.map(|v| v / n.max(1.0));
            }
        }
    }
    out
}
