//! Smart selection: interactive segmentation for Photoshop's Quick Selection, Object Selection,
//! Select Subject and Focus Area.
//!
//! Everything here is classical (no learned models) and implemented from the published papers:
//!
//! * [`maxflow`]: hybrid push/relabel (Goldberg & Tarjan, JACM 1988) and Boykov–Kolmogorov
//!   min-cut / max-flow (Boykov & Kolmogorov, "An Experimental
//!   Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision", PAMI 2004).
//! * [`gmm`]: Gaussian mixture colour models (k-means++ initialisation, Arthur & Vassilvitskii
//!   2007; EM refinement).
//! * [`slic`]: SLIC superpixels (Achanta et al., "SLIC Superpixels Compared to State-of-the-Art
//!   Superpixel Methods", PAMI 2012).
//! * [`grabcut`]: GrabCut (Rother, Kolmogorov & Blake, SIGGRAPH 2004): iterated graph cuts
//!   (Boykov & Jolly, ICCV 2001) with GMM colour models.
//! * [`quick`]: Quick Selection brush: one contrast-sensitive min cut with an area cost, its
//!   behaviour and constants measured on Photoshop (see the module).
//! * [`subject`]: Select Subject: superpixel saliency (boundary connectivity + background-weighted
//!   contrast, Zhu et al., "Saliency Optimization from Robust Background Detection", CVPR 2014;
//!   frequency-tuned saliency, Achanta et al., CVPR 2009) seeding a GrabCut. Heuristic.
//! * [`focus`]: Focus Area: local Laplacian energy (sharpness) thresholding. Heuristic.
//!
//! Large images are processed at a reduced working resolution and the boundary is then re-cut
//! at full resolution in a narrow band (tile by tile, in parallel on native targets).
//!
//! Colour analysis runs on straight RGB samples in `0..=1` read through a [`Sampler`], so any
//! bit depth or colour model can be segmented; results are 8-bit coverage [`Region`]s.

pub mod focus;
pub mod gmm;
pub mod grabcut;
pub mod maxflow;
pub mod quick;
pub mod sky;
pub mod slic;
pub mod subject;

use photocraft_geom::Rect;
use photocraft_raster::{Cancelled, Interrupt, Surface};

pub use crate::selection::Region;
use gmm::Gmm;
use maxflow::Graph;

/// Row-major RGB image, samples in `0..=1`.
#[derive(Clone, Debug, PartialEq)]
pub struct RgbImage {
    pub w: usize,
    pub h: usize,
    pub px: Vec<[f32; 3]>,
}

impl RgbImage {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, px: vec![[0.0; 3]; w * h] }
    }
    pub fn from_fn(w: usize, h: usize, mut f: impl FnMut(usize, usize) -> [f32; 3]) -> Self {
        let mut px = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                px.push(f(x, y));
            }
        }
        Self { w, h, px }
    }
    #[inline]
    pub fn at(&self, x: usize, y: usize) -> [f32; 3] {
        self.px[y * self.w + x]
    }
    /// Straight RGBA to RGB, compositing transparent pixels over mid gray (so transparency is a
    /// colour the models can separate).
    pub fn from_rgba(px: &[[f32; 4]], w: usize, h: usize) -> Self {
        debug_assert_eq!(px.len(), w * h);
        let px = px
            .iter()
            .map(|p| {
                let a = p[3].clamp(0.0, 1.0);
                [0, 1, 2].map(|c| p[c].clamp(0.0, 1.0) * a + 0.5 * (1.0 - a))
            })
            .collect();
        Self { w, h, px }
    }
    /// Box-average downsampling by an integer `step` (output `ceil(w/step) × ceil(h/step)`).
    pub fn downsample(&self, step: usize) -> RgbImage {
        if step <= 1 {
            return self.clone();
        }
        let (ow, oh) = (self.w.div_ceil(step), self.h.div_ceil(step));
        let mut out = RgbImage::new(ow, oh);
        for oy in 0..oh {
            for ox in 0..ow {
                let mut acc = [0.0f32; 3];
                let mut n = 0.0;
                for y in oy * step..((oy + 1) * step).min(self.h) {
                    for x in ox * step..((ox + 1) * step).min(self.w) {
                        let p = self.at(x, y);
                        for c in 0..3 {
                            acc[c] += p[c];
                        }
                        n += 1.0;
                    }
                }
                out.px[oy * ow + ox] = acc.map(|v| v / n);
            }
        }
        out
    }
}

/// Where the segmentation algorithms read colours from (a layer, the composite, a test image).
pub trait Sampler: Sync {
    /// Straight RGBA (`0..=1`) over `r` (document pixels), row-major.
    fn rgba(&self, r: Rect) -> Vec<[f32; 4]>;

    /// RGB over `r` at full resolution.
    fn rgb(&self, r: Rect) -> RgbImage {
        RgbImage::from_rgba(&self.rgba(r), r.width() as usize, r.height() as usize)
    }

    /// RGB over `r`, box-downsampled by `step`, read in horizontal strips (in parallel on native)
    /// so the full-resolution area is never held in memory at once.
    fn rgb_scaled(&self, r: Rect, step: usize) -> RgbImage {
        if step <= 1 {
            return self.rgb(r);
        }
        let (w, h) = (r.width() as usize, r.height() as usize);
        let (ow, oh) = (w.div_ceil(step), h.div_ceil(step));
        let rows_per = (64usize.div_ceil(step)).max(1);
        let strips: Vec<usize> = (0..oh.div_ceil(rows_per)).collect();
        let run = |si: &usize| -> Vec<[f32; 3]> {
            let oy0 = si * rows_per;
            let oy1 = (oy0 + rows_per).min(oh);
            let y0 = r.y0 + (oy0 * step) as i32;
            let y1 = (r.y0 + (oy1 * step) as i32).min(r.y1);
            let strip = self.rgb(Rect::new(r.x0, y0, r.x1, y1));
            strip.downsample(step).px
        };
        #[cfg(not(target_arch = "wasm32"))]
        let parts: Vec<Vec<[f32; 3]>> = {
            use rayon::prelude::*;
            strips.par_iter().map(run).collect()
        };
        #[cfg(target_arch = "wasm32")]
        let parts: Vec<Vec<[f32; 3]>> = strips.iter().map(run).collect();
        let mut px = Vec::with_capacity(ow * oh);
        for p in parts {
            px.extend(p);
        }
        RgbImage { w: ow, h: oh, px }
    }
}

/// Samples a raster surface (any format; converted to straight RGBA).
pub struct SurfaceSampler<'a>(pub &'a Surface);

impl Sampler for SurfaceSampler<'_> {
    fn rgba(&self, r: Rect) -> Vec<[f32; 4]> {
        let mut px = vec![[0.0f32; 4]; r.width() as usize * r.height() as usize];
        if !r.is_empty() {
            self.0.read_rgba_into(r, &mut px);
        }
        px
    }
}

/// Samples an in-memory RGB image placed at `origin` (tests, benchmarks).
pub struct ImageSampler<'a> {
    pub img: &'a RgbImage,
    pub origin: (i32, i32),
}

impl Sampler for ImageSampler<'_> {
    fn rgba(&self, r: Rect) -> Vec<[f32; 4]> {
        let mut out = Vec::with_capacity(r.width() as usize * r.height() as usize);
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                let (ix, iy) = (x - self.origin.0, y - self.origin.1);
                if ix < 0 || iy < 0 || ix as usize >= self.img.w || iy as usize >= self.img.h {
                    out.push([0.0; 4]);
                } else {
                    let p = self.img.at(ix as usize, iy as usize);
                    out.push([p[0], p[1], p[2], 1.0]);
                }
            }
        }
        out
    }
}

/// Integer working-resolution step so that `area` holds at most about `max_px` pixels.
pub fn scale_for(area: Rect, max_px: usize) -> usize {
    let n = area.width() as f64 * area.height() as f64;
    ((n / max_px.max(1) as f64).sqrt().ceil() as usize).max(1)
}

#[inline]
fn d2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let (x, y, z) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    x * x + y * y + z * z
}

/// The 4 forward neighbour offsets of an 8-connected grid and their lengths.
const NB: [(i32, i32, f32); 4] = [(1, 0, 1.0), (0, 1, 1.0), (1, 1, std::f32::consts::SQRT_2), (-1, 1, std::f32::consts::SQRT_2)];

/// GrabCut's contrast parameter: `β = 1 / (2 ⟨‖z_m − z_n‖²⟩)` over 8-neighbour pairs.
pub fn contrast_beta(img: &RgbImage) -> f32 {
    let (w, h) = (img.w as i32, img.h as i32);
    let mut sum = 0.0f64;
    let mut n = 0u64;
    for y in 0..h {
        for x in 0..w {
            let a = img.px[(y * w + x) as usize];
            for (dx, dy, _) in NB {
                let (nx, ny) = (x + dx, y + dy);
                if nx >= 0 && nx < w && ny < h {
                    sum += d2(a, img.px[(ny * w + nx) as usize]) as f64;
                    n += 1;
                }
            }
        }
    }
    if n == 0 || sum <= 1e-12 { 0.0 } else { (n as f64 / (2.0 * sum)) as f32 }
}

/// Pixel label constraint for [`grid_cut`].
pub const FREE: u8 = 0;
pub const HARD_FG: u8 = 1;
pub const HARD_BG: u8 = 2;

/// Binary labelling of a grid by one min cut (Boykov & Jolly 2001):
/// minimises `Σ D_p(l_p) + Σ_{p~q} [l_p ≠ l_q] γ e^{-β‖z_p−z_q‖²} / dist(p,q)` over the
/// 8-connected grid. `cost_fg[i]` / `cost_bg[i]` are the data costs of labelling pixel `i`
/// foreground / background; `fixed[i]` is [`FREE`], [`HARD_FG`] or [`HARD_BG`]. Only free pixels
/// become graph nodes (links to fixed neighbours fold into terminal capacities). Returns
/// `true` for foreground.
pub fn grid_cut(img: &RgbImage, cost_fg: &[f32], cost_bg: &[f32], fixed: &[u8], gamma: f32, beta: f32) -> Vec<bool> {
    grid_cut_weighted::<true>(img, cost_fg, cost_bg, fixed, |d2| gamma * (-beta * d2).exp())
}

/// [`grid_cut`] with any edge weight: separating neighbours costs `weight(‖z_p−z_q‖²) / dist(p,q)`.
/// Only links touching a free pixel need a weight; fixed/fixed links cannot change the cut.
pub fn grid_cut_with(img: &RgbImage, cost_fg: &[f32], cost_bg: &[f32], fixed: &[u8], weight: impl Fn(f32) -> f32) -> Vec<bool> {
    grid_cut_weighted::<true>(img, cost_fg, cost_bg, fixed, weight)
}

/// [`grid_cut_with`] that stops when `cancel` returns true (asked every few thousand max-flow
/// steps): `None` then. Not cancelled, the same cut.
pub fn grid_cut_cancellable(
    img: &RgbImage,
    cost_fg: &[f32],
    cost_bg: &[f32],
    fixed: &[u8],
    weight: impl Fn(f32) -> f32,
    cancel: &dyn Fn() -> bool,
) -> Option<Vec<bool>> {
    grid_cut_impl::<true>(img, cost_fg, cost_bg, fixed, weight, cancel)
}

mod edge_starts;

// The const parameter also runs the previous full scan in before/after workflow tests.
fn grid_cut_weighted<const OPTIMIZED: bool>(img: &RgbImage, cost_fg: &[f32], cost_bg: &[f32], fixed: &[u8], weight: impl Fn(f32) -> f32) -> Vec<bool> {
    grid_cut_impl::<OPTIMIZED>(img, cost_fg, cost_bg, fixed, weight, &|| false).unwrap_or_default()
}

fn grid_cut_impl<const OPTIMIZED: bool>(
    img: &RgbImage,
    cost_fg: &[f32],
    cost_bg: &[f32],
    fixed: &[u8],
    weight: impl Fn(f32) -> f32,
    cancel: &dyn Fn() -> bool,
) -> Option<Vec<bool>> {
    let (w, h) = (img.w as i32, img.h as i32);
    let Some(n) = img.w.checked_mul(img.h).filter(|&n| n < u32::MAX as usize - 2) else { return Some(Vec::new()) };
    if img.w > i32::MAX as usize || img.h > i32::MAX as usize || [img.px.len(), cost_fg.len(), cost_bg.len(), fixed.len()].iter().any(|&len| len < n) {
        return Some(Vec::new());
    }
    let mut free = Vec::new();
    let mut idx = vec![u32::MAX; n];
    let mut count = 0u32;
    for i in 0..n {
        if fixed[i] == FREE {
            idx[i] = count;
            count += 1;
        }
    }
    let mut out: Vec<bool> = fixed.iter().map(|f| *f == HARD_FG).collect();
    if count == 0 {
        return Some(out);
    }
    if cancel() {
        return None;
    }
    let mut g = Graph::with_capacity(count as usize, count as usize * 4);
    for i in 0..n {
        if idx[i] != u32::MAX {
            g.add_tweights(idx[i] as usize, cost_bg[i], cost_fg[i]);
        }
    }
    let use_sparse = OPTIMIZED && (count as usize) < n / 8;
    if use_sparse {
        free.extend((0..n).filter(|&i| idx[i] != u32::MAX))
    }
    let mut visit = |i: usize, x: i32, y: i32| {
        let a = img.px[i];
        for (dx, dy, len) in NB {
            let (nx, ny) = (x + dx, y + dy);
            if nx < 0 || nx >= w || ny >= h {
                continue;
            }
            let j = (ny as usize) * img.w + nx as usize;
            if OPTIMIZED && idx[i] == u32::MAX && idx[j] == u32::MAX {
                continue;
            }
            let wt = weight(d2(a, img.px[j])) / len;
            match (idx[i] != u32::MAX, idx[j] != u32::MAX) {
                (true, true) => g.add_edge(idx[i] as usize, idx[j] as usize, wt, wt),
                (true, false) => add_fixed_link(&mut g, idx[i] as usize, fixed[j], wt),
                (false, true) => add_fixed_link(&mut g, idx[j] as usize, fixed[i], wt),
                (false, false) => {}
            }
        }
    };
    if use_sparse {
        for i in edge_starts::Starts::new(&free, img.w) {
            visit(i, (i % img.w) as i32, (i / img.w) as i32);
        }
    } else {
        // Keep the original dense raster traversal, including its arithmetic order.
        for y in 0..h {
            for x in 0..w {
                visit((y as usize) * img.w + x as usize, x, y)
            }
        }
    }
    if cancel() {
        return None;
    }
    if OPTIMIZED {
        g.maxflow_with(cancel)?;
    } else {
        g.maxflow_bk();
    }
    for i in 0..n {
        if idx[i] != u32::MAX {
            out[i] = g.in_source(idx[i] as usize);
        }
    }
    Some(out)
}

fn add_fixed_link(g: &mut Graph, node: usize, fixed: u8, wt: f32) {
    if fixed == HARD_FG {
        // Labelling `node` background cuts the link.
        g.add_tweights(node, wt, 0.0);
    } else {
        g.add_tweights(node, 0.0, wt);
    }
}

/// Data costs `(−ln p(z|fg), −ln p(z|bg))` for every pixel.
pub fn data_costs(img: &RgbImage, fg: &Gmm, bg: &Gmm) -> (Vec<f32>, Vec<f32>) {
    let eval = |p: &[f32; 3]| (fg.neg_log(*p), bg.neg_log(*p));
    #[cfg(not(target_arch = "wasm32"))]
    let v: Vec<(f32, f32)> = {
        use rayon::prelude::*;
        img.px.par_iter().map(eval).collect()
    };
    #[cfg(target_arch = "wasm32")]
    let v: Vec<(f32, f32)> = img.px.iter().map(eval).collect();
    v.into_iter().unzip()
}

/// Labels 8-connected components of `mask`; returns (labels with 0 = background, sizes indexed by label).
pub fn components(mask: &[bool], w: usize, h: usize) -> (Vec<u32>, Vec<usize>) {
    let mut lab = vec![0u32; w * h];
    let mut sizes = vec![0usize];
    let mut stack = Vec::new();
    for s in 0..w * h {
        if !mask[s] || lab[s] != 0 {
            continue;
        }
        let l = sizes.len() as u32;
        sizes.push(0);
        lab[s] = l;
        stack.push(s);
        while let Some(i) = stack.pop() {
            sizes[l as usize] += 1;
            let (x, y) = ((i % w) as i32, (i / w) as i32);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if mask[j] && lab[j] == 0 {
                        lab[j] = l;
                        stack.push(j);
                    }
                }
            }
        }
    }
    (lab, sizes)
}

/// Keeps the connected components of `mask` that contain a `seed` pixel.
pub fn keep_seeded(mask: &[bool], seeds: &[bool], w: usize, h: usize) -> Vec<bool> {
    let (lab, sizes) = components(mask, w, h);
    let mut keep = vec![false; sizes.len()];
    for (l, s) in lab.iter().zip(seeds) {
        if *s && *l != 0 {
            keep[*l as usize] = true;
        }
    }
    lab.iter().map(|l| keep[*l as usize] && *l != 0).collect()
}

/// Keeps components at least `frac` of the largest one and fills holes (4-connected background
/// areas not touching the border) smaller than `hole_frac` of the kept area.
pub fn clean_mask(mask: &[bool], w: usize, h: usize, frac: f32, hole_frac: f32) -> Vec<bool> {
    let (lab, sizes) = components(mask, w, h);
    let largest = sizes.iter().skip(1).copied().max().unwrap_or(0);
    if largest == 0 {
        return vec![false; w * h];
    }
    let min = (largest as f32 * frac).ceil() as usize;
    let mut out: Vec<bool> = lab.iter().map(|l| *l != 0 && sizes[*l as usize] >= min).collect();
    let area = out.iter().filter(|v| **v).count();
    let max_hole = (area as f32 * hole_frac) as usize;
    let inv: Vec<bool> = out.iter().map(|v| !v).collect();
    let (hl, hs) = components4(&inv, w, h);
    let mut touches = vec![false; hs.len()];
    for x in 0..w {
        touches[hl[x] as usize] = true;
        touches[hl[(h - 1) * w + x] as usize] = true;
    }
    for y in 0..h {
        touches[hl[y * w] as usize] = true;
        touches[hl[y * w + w - 1] as usize] = true;
    }
    for (o, l) in out.iter_mut().zip(&hl) {
        if *l != 0 && !touches[*l as usize] && hs[*l as usize] <= max_hole {
            *o = true;
        }
    }
    out
}

fn components4(mask: &[bool], w: usize, h: usize) -> (Vec<u32>, Vec<usize>) {
    let mut lab = vec![0u32; w * h];
    let mut sizes = vec![0usize];
    let mut stack = Vec::new();
    for s in 0..w * h {
        if !mask[s] || lab[s] != 0 {
            continue;
        }
        let l = sizes.len() as u32;
        sizes.push(0);
        lab[s] = l;
        stack.push(s);
        while let Some(i) = stack.pop() {
            sizes[l as usize] += 1;
            let (x, y) = (i % w, i / w);
            let n = [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)];
            for j in n.into_iter().flatten() {
                if mask[j] && lab[j] == 0 {
                    lab[j] = l;
                    stack.push(j);
                }
            }
        }
    }
    (lab, sizes)
}

/// Softens a hard 0/255 mask's edges: edge pixels blend with their 3×3 neighbourhood (the same
/// rule as the magic wand's anti-aliasing).
pub fn antialias_u8(mask: &mut [u8], w: usize, h: usize) {
    let src = mask.to_vec();
    let at = |x: i32, y: i32, v: u8| if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 { v } else { src[y as usize * w + x as usize] };
    let row = |y: usize, out: &mut [u8]| {
        for (x, o) in out.iter_mut().enumerate() {
            let v = src[y * w + x];
            let mut sum = 0u32;
            let mut edge = false;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let n = at(x as i32 + dx, y as i32 + dy, v);
                    edge |= n != v;
                    sum += n as u32;
                }
            }
            if edge {
                let avg = sum as f32 / (9.0 * 255.0);
                let f = if v > 127 { 0.5 + 0.5 * avg } else { 0.5 * avg };
                *o = (f * 255.0).round() as u8;
            }
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        mask.par_chunks_mut(w).enumerate().for_each(|(y, r)| row(y, r));
    }
    #[cfg(target_arch = "wasm32")]
    mask.chunks_mut(w).enumerate().for_each(|(y, r)| row(y, r));
}

/// Shrinks a region's box to its non-zero pixels (`None` if empty).
pub fn trim_region(r: Region) -> Option<Region> {
    let w = r.bbox.width() as usize;
    if w == 0 {
        return None;
    }
    let h = r.mask.len() / w;
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for y in 0..h {
        let row = &r.mask[y * w..(y + 1) * w];
        if let (Some(a), Some(b)) = (row.iter().position(|v| *v != 0), row.iter().rposition(|v| *v != 0)) {
            x0 = x0.min(a);
            x1 = x1.max(b + 1);
            y0 = y0.min(y);
            y1 = y + 1;
        }
    }
    if x0 == usize::MAX {
        return None;
    }
    if (x0, y0, x1, y1) == (0, 0, w, h) {
        return Some(r);
    }
    let nw = x1 - x0;
    let mut mask = Vec::with_capacity(nw * (y1 - y0));
    for y in y0..y1 {
        mask.extend_from_slice(&r.mask[y * w + x0..y * w + x1]);
    }
    let b = r.bbox;
    Some(Region { bbox: Rect::new(b.x0 + x0 as i32, b.y0 + y0 as i32, b.x0 + x1 as i32, b.y0 + y1 as i32), mask })
}

/// Turns a working-resolution labelling over `window` (sampled with `step`) into a full-resolution
/// region. With `step > 1` the boundary is re-cut at full resolution in a band about `step`
/// pixels wide around the upsampled boundary, using the given colour models, tile by tile.
/// The result is anti-aliased by one pixel.
pub fn finish_region(sampler: &dyn Sampler, window: Rect, step: usize, low: &[bool], lw: usize, lh: usize, models: Option<(&Gmm, &Gmm)>) -> Option<Region> {
    finish_region_with(sampler, window, step, low, lw, lh, models, &Interrupt::NONE).unwrap_or_default()
}

/// [`finish_region`] that stops with `Err(Cancelled)` once `ctl` is cancelled, asked before each
/// band tile and inside its min cut.
#[allow(clippy::too_many_arguments)]
pub fn finish_region_with(
    sampler: &dyn Sampler,
    window: Rect,
    step: usize,
    low: &[bool],
    lw: usize,
    lh: usize,
    models: Option<(&Gmm, &Gmm)>,
    ctl: &Interrupt,
) -> Result<Option<Region>, Cancelled> {
    let (w, h) = (window.width() as usize, window.height() as usize);
    let mut mask = vec![0u8; w * h];
    if step <= 1 {
        for (m, l) in mask.iter_mut().zip(low) {
            *m = if *l { 255 } else { 0 };
        }
    } else {
        // Band blocks: working pixels with a differently labelled pixel within one block.
        let mut band = vec![false; lw * lh];
        for y in 0..lh {
            for x in 0..lw {
                let v = low[y * lw + x];
                'n: for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                        if nx >= 0 && ny >= 0 && (nx as usize) < lw && (ny as usize) < lh && low[ny as usize * lw + nx as usize] != v {
                            band[y * lw + x] = true;
                            break 'n;
                        }
                    }
                }
            }
        }
        let fill = |y: usize, row: &mut [u8]| {
            let ly = (y / step).min(lh - 1);
            for (x, o) in row.iter_mut().enumerate() {
                *o = if low[ly * lw + (x / step).min(lw - 1)] { 255 } else { 0 };
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            mask.par_chunks_mut(w).enumerate().for_each(|(y, r)| fill(y, r));
        }
        #[cfg(target_arch = "wasm32")]
        mask.chunks_mut(w).enumerate().for_each(|(y, r)| fill(y, r));
        if let Some((fg, bg)) = models {
            refine_band(sampler, window, step, low, &band, lw, lh, fg, bg, &mut mask, ctl)?;
        }
    }
    antialias_u8(&mut mask, w, h);
    Ok(trim_region(Region { bbox: window, mask }))
}

/// Full-resolution graph cut restricted to the band blocks (see [`finish_region`]).
#[allow(clippy::too_many_arguments)]
fn refine_band(
    sampler: &dyn Sampler,
    window: Rect,
    step: usize,
    low: &[bool],
    band: &[bool],
    lw: usize,
    lh: usize,
    fg: &Gmm,
    bg: &Gmm,
    mask: &mut [u8],
    ctl: &Interrupt,
) -> Result<(), Cancelled> {
    const T: i32 = 192;
    const HALO: i32 = 12;
    let w = window.width() as usize;
    let block = |x: i32, y: i32| -> usize { ((y - window.y0) as usize / step).min(lh - 1) * lw + ((x - window.x0) as usize / step).min(lw - 1) };
    let mut tiles = Vec::new();
    let mut ty = window.y0;
    while ty < window.y1 {
        let mut tx = window.x0;
        while tx < window.x1 {
            let t = Rect::new(tx, ty, (tx + T).min(window.x1), (ty + T).min(window.y1));
            // Any band block overlapping the tile?
            let (bx0, by0) = ((t.x0 - window.x0) as usize / step, (t.y0 - window.y0) as usize / step);
            let (bx1, by1) = (((t.x1 - 1 - window.x0) as usize / step).min(lw - 1), ((t.y1 - 1 - window.y0) as usize / step).min(lh - 1));
            if (by0..=by1).any(|by| (bx0..=bx1).any(|bx| band[by * lw + bx])) {
                tiles.push(t);
            }
            tx += T;
        }
        ty += T;
    }
    let cancel = || ctl.cancelled();
    let run = |t: &Rect| -> Result<(Rect, Vec<bool>), Cancelled> {
        ctl.check()?;
        let r = t.inflate(HALO).intersect(&window);
        let img = sampler.rgb(r);
        let (rw, rh) = (r.width() as usize, r.height() as usize);
        let mut fixed = vec![FREE; rw * rh];
        let mut cfg = vec![0.0f32; rw * rh];
        let mut cbg = vec![0.0f32; rw * rh];
        for yy in 0..rh {
            for xx in 0..rw {
                let i = yy * rw + xx;
                let b = block(r.x0 + xx as i32, r.y0 + yy as i32);
                let lab = low[b];
                if !band[b] {
                    fixed[i] = if lab { HARD_FG } else { HARD_BG };
                    continue;
                }
                let p = img.px[i];
                // Colour likelihoods plus a weak prior towards the working-resolution label.
                cfg[i] = fg.neg_log(p) + if lab { 0.0 } else { 0.4 };
                cbg[i] = bg.neg_log(p) + if lab { 0.4 } else { 0.0 };
            }
        }
        let beta = contrast_beta(&img);
        // `grid_cut`'s weight, with the cut stopping on cancel.
        let cut = grid_cut_cancellable(&img, &cfg, &cbg, &fixed, |d2| 50.0 * (-beta * d2).exp(), &cancel).ok_or(Cancelled)?;
        let (cw, ch) = (t.width() as usize, t.height() as usize);
        let mut core = Vec::with_capacity(cw * ch);
        for yy in 0..ch {
            let sy = (t.y0 - r.y0) as usize + yy;
            let sx = (t.x0 - r.x0) as usize;
            core.extend_from_slice(&cut[sy * rw + sx..sy * rw + sx + cw]);
        }
        Ok((*t, core))
    };
    // A cancelled tile ends the collect; tiles not yet started see `ctl` and return at once.
    #[cfg(not(target_arch = "wasm32"))]
    let results: Vec<(Rect, Vec<bool>)> = {
        use rayon::prelude::*;
        tiles.par_iter().map(run).collect::<Result<_, _>>()?
    };
    #[cfg(target_arch = "wasm32")]
    let results: Vec<(Rect, Vec<bool>)> = tiles.iter().map(run).collect::<Result<_, _>>()?;
    for (t, core) in results {
        let cw = t.width() as usize;
        for (yy, row) in core.chunks_exact(cw).enumerate() {
            let o = (t.y0 - window.y0) as usize * w + yy * w + (t.x0 - window.x0) as usize;
            for (m, v) in mask[o..o + cw].iter_mut().zip(row) {
                *m = if *v { 255 } else { 0 };
            }
        }
    }
    Ok(())
}

/// Deterministic pseudo-random numbers (xorshift) for sampling and tests.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform in `[0, 1)`.
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Standard normal (Box–Muller).
    pub fn normal(&mut self) -> f32 {
        let u = self.f32().max(1e-7);
        let v = self.f32();
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
}

/// Every `k`-th element so that at most `max` remain.
pub fn subsample<T: Copy>(v: &[T], max: usize) -> Vec<T> {
    if v.len() <= max {
        return v.to_vec();
    }
    let k = v.len().div_ceil(max);
    v.iter().step_by(k).copied().collect()
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod workflow_tests;
