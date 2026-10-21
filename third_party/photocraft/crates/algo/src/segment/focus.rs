//! Focus Area (heuristic): selects the in-focus parts of an image.
//!
//! Sharpness is measured as local Laplacian energy (the classic "sum of modified Laplacian" /
//! Laplacian focus measure, Nayar & Nakagawa, "Shape from Focus", PAMI 1994) on a working copy,
//! normalised by its 99th percentile. `noise` subtracts a noise floor; `range` (0–1, Photoshop's
//! "In-Focus Range") lowers the threshold as it grows. The thresholded map is smoothed with a
//! guided filter (image as guide) so it follows object edges, cleaned, and upsampled with a soft
//! edge.

use photocraft_geom::Rect;

use super::{Region, Sampler, clean_mask};
use crate::matting::{box_mean, guided_filter_color};

/// Working-resolution budget (pixels).
pub const WORK_PX: usize = 1_000_000;

/// Per-pixel normalised sharpness (`0..=1`) of `img`.
pub fn sharpness(img: &super::RgbImage, noise: f32) -> Vec<f32> {
    let (w, h) = (img.w, img.h);
    let y: Vec<f32> = img.px.iter().map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]).collect();
    let at = |x: usize, yy: usize| y[yy * w + x];
    let mut lap = vec![0.0f32; w * h];
    for yy in 0..h {
        for x in 0..w {
            let c = at(x, yy);
            let l = (2.0 * c - at(x.saturating_sub(1), yy) - at((x + 1).min(w - 1), yy)).abs()
                + (2.0 * c - at(x, yy.saturating_sub(1)) - at(x, (yy + 1).min(h - 1))).abs();
            lap[yy * w + x] = l;
        }
    }
    let r = (w.min(h) / 60).max(3);
    let e = box_mean(&lap, w, h, r);
    let mut sorted = e.clone();
    let k = ((sorted.len() as f32 * 0.99) as usize).min(sorted.len().saturating_sub(1));
    let p99 = *sorted.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1;
    let floor = noise.clamp(0.0, 1.0) * 0.5;
    e.iter().map(|v| ((v / p99.max(1e-6)).min(1.0) - floor).max(0.0) / (1.0 - floor).max(1e-3)).collect()
}

/// Focus Area selection over `canvas`.
pub fn focus_area(sampler: &dyn Sampler, canvas: Rect, range: f32, noise: f32) -> Option<Region> {
    if canvas.width() < 4 || canvas.height() < 4 {
        return None;
    }
    let step = super::scale_for(canvas, WORK_PX);
    let img = sampler.rgb_scaled(canvas, step);
    let (w, h) = (img.w, img.h);
    let s = sharpness(&img, noise);
    let t = 0.05 + 0.9 * (1.0 - range.clamp(0.0, 1.0)).powi(2);
    let bin: Vec<f32> = s.iter().map(|v| if *v >= t { 1.0 } else { 0.0 }).collect();
    let r = (w.min(h) / 50).max(2);
    let smooth = guided_filter_color(&img, &bin, r, 1e-2);
    let mask: Vec<bool> = smooth.iter().map(|v| *v >= 0.5).collect();
    let mask = clean_mask(&mask, w, h, 0.05, 0.02);
    if !mask.iter().any(|v| *v) {
        return None;
    }
    // Soft upsampling: bilinear field, steepened to about a working pixel of transition.
    let field: Vec<f32> = mask.iter().zip(&smooth).map(|(m, v)| if *m { v.clamp(0.5, 1.0) } else { v.clamp(0.0, 0.5) }).collect();
    let (fw, fh) = (canvas.width() as usize, canvas.height() as usize);
    let st = step as f32;
    let mut out = vec![0u8; fw * fh];
    let row = |yy: usize, o: &mut [u8]| {
        let fy = ((yy as f32 + 0.5) / st - 0.5).clamp(0.0, (h - 1) as f32);
        let (y0, ty) = (fy.floor() as usize, fy.fract());
        let y1 = (y0 + 1).min(h - 1);
        for (xx, px) in o.iter_mut().enumerate() {
            let fx = ((xx as f32 + 0.5) / st - 0.5).clamp(0.0, (w - 1) as f32);
            let (x0, tx) = (fx.floor() as usize, fx.fract());
            let x1 = (x0 + 1).min(w - 1);
            let top = field[y0 * w + x0] * (1.0 - tx) + field[y0 * w + x1] * tx;
            let bot = field[y1 * w + x0] * (1.0 - tx) + field[y1 * w + x1] * tx;
            let v = top * (1.0 - ty) + bot * ty;
            *px = (((v - 0.5) * 3.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(fw).enumerate().for_each(|(y, o)| row(y, o));
    }
    #[cfg(target_arch = "wasm32")]
    out.chunks_mut(fw).enumerate().for_each(|(y, o)| row(y, o));
    super::trim_region(Region { bbox: canvas, mask: out })
}
