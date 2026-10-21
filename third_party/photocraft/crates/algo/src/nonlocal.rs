//! Non-local patch-based completion, the Remove Tool's fill: A. Newson, A. Almansa, Y. Gousseau,
//! P. Pérez, *Non-Local Patch-Based Image Inpainting*, Image Processing On Line 7 (2017), 373–385.
//! Like [`crate::inpaint::complete`] it alternates, coarse to fine, a PatchMatch search for each
//! patch's nearest fully-known patch (C. Barnes et al., SIGGRAPH 2009) with a similarity-weighted
//! vote of the matches (Y. Wexler, E. Shechtman, M. Irani, TPAMI 2007). What it changes:
//!
//! * **Texture features** join the patch distance: the local mean of `|∂x|` and `|∂y|` (after
//!   Liu and Caselles), weighted by [`Params::lambda`]. Colour alone can't tell grass from a blurred
//!   background of the same mean colour, and a smooth guess in the hole then attracts smooth
//!   sources until the fill is a blur. The features are filled along with the colour.
//! * **Onion-peel initialisation** at the coarsest level: the hole is filled one boundary layer at
//!   a time from partially known patches, which carries structures in where a harmonic first guess
//!   would blur them.
//! * **The match field is upsampled** between levels, not the image: each finer level is rebuilt
//!   from its own full-resolution sources instead of from an enlarged blur.
//! * **Each final pixel comes from the best patch covering it** instead of the weighted vote, so
//!   the fill keeps the grain of its sources.
//! * **Seams are hidden in the gradient domain** (not in the paper): a membrane carries the colour
//!   mismatch along the hole's edge into the fill (P. Pérez et al., *Poisson Image Editing*, 2003),
//!   and steps between areas copied from different places, where texture doesn't hide them, are
//!   spread out by a smooth correction (after S. Darabi et al., *Image Melding*, 2012).
//! * **The number of levels follows the hole's thickness** (`log2(2·No/N)`, `No` the erosions that
//!   empty the hole, `N` the patch side), not its extent, so long thin holes stay at fine scales.
//!
//! Buffers are interleaved `w × h × ch` normalised floats (any colour model and depth) with a
//! `w × h` hole mask. Deterministic for a seed, whatever the thread count.

use crate::inpaint::Rng;
use photocraft_raster::{Cancelled, Interrupt};

/// Below this many grid cells a level runs single-threaded (thread hand-off costs more than it saves).
#[cfg(not(target_arch = "wasm32"))]
const PAR_MIN: usize = 128 * 128;

/// Parameters for [`complete_with`]. The defaults are the paper's, except for the PatchMatch
/// passes per iteration (two instead of ten, for speed).
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    /// Patch radius (patches are `2r+1` square).
    pub patch_radius: usize,
    /// Weight of the texture features against the colour in the patch distance.
    pub lambda: f32,
    /// Most EM iterations (search, then vote) per level.
    pub max_iters: usize,
    /// PatchMatch passes per EM iteration.
    pub pm_passes: usize,
    /// A level is done once an iteration changes the hole by less than this on average (0..1 units).
    pub tolerance: f32,
    pub seed: u64,
}

impl Default for Params {
    fn default() -> Self {
        Self { patch_radius: 3, lambda: 50.0, max_iters: 10, pm_passes: 2, tolerance: 0.1 / 255.0, seed: 1 }
    }
}

/// One pyramid level: `cha` channels per cell (the colour, then the two texture features).
struct Level {
    w: usize,
    h: usize,
    img: Vec<f32>,
    hole: Vec<bool>,
}

/// Fully-known patch centres (the whole patch on the grid and outside the hole), as a mask and a list.
fn valid_sources(w: usize, h: usize, hole: &[bool], r: usize) -> (Vec<bool>, Vec<(i32, i32)>) {
    let mut valid = vec![false; w * h];
    let mut list = Vec::new();
    if w < 2 * r + 1 || h < 2 * r + 1 {
        return (valid, list);
    }
    // Summed-area table of the hole.
    let w1 = w + 1;
    let mut sat = vec![0u32; w1 * (h + 1)];
    for y in 0..h {
        let mut row = 0u32;
        for x in 0..w {
            row += u32::from(hole[y * w + x]);
            sat[(y + 1) * w1 + x + 1] = sat[y * w1 + x + 1] + row;
        }
    }
    for y in r..h - r {
        for x in r..w - r {
            let (x0, y0, x1, y1) = (x - r, y - r, x + r + 1, y + r + 1);
            if sat[y1 * w1 + x1] + sat[y0 * w1 + x0] == sat[y0 * w1 + x1] + sat[y1 * w1 + x0] {
                valid[y * w + x] = true;
                list.push((x as i32, y as i32));
            }
        }
    }
    (valid, list)
}

/// The texture features at every known cell: the mean of `|∂x|` and of `|∂y|` of the channel mean
/// over a `side`-wide window, counting only derivatives between two known cells. Hole cells get 0
/// (the completion fills them in).
fn texture_features(w: usize, h: usize, ch: usize, img: &[f32], hole: &[bool], side: usize) -> (Vec<f32>, Vec<f32>) {
    let g: Vec<f32> = img.chunks_exact(ch).map(|p| p.iter().sum::<f32>() / ch as f32).collect();
    let w1 = w + 1;
    // Summed-area tables of each derivative and of where it is defined.
    let table = |dx: usize, dy: usize| {
        let mut sum = vec![0.0f64; w1 * (h + 1)];
        let mut cnt = vec![0u32; w1 * (h + 1)];
        for y in 0..h {
            let (mut rs, mut rc) = (0.0f64, 0u32);
            for x in 0..w {
                let i = y * w + x;
                let (nx, ny) = (x + dx, y + dy);
                if nx < w && ny < h && !hole[i] && !hole[ny * w + nx] {
                    rs += f64::from((g[ny * w + nx] - g[i]).abs());
                    rc += 1;
                }
                sum[(y + 1) * w1 + x + 1] = sum[y * w1 + x + 1] + rs;
                cnt[(y + 1) * w1 + x + 1] = cnt[y * w1 + x + 1] + rc;
            }
        }
        let half = side / 2;
        (0..w * h)
            .map(|i| {
                if hole[i] {
                    return 0.0;
                }
                let (x, y) = (i % w, i / w);
                let (x0, y0, x1, y1) = (x.saturating_sub(half), y.saturating_sub(half), (x + half + 1).min(w), (y + half + 1).min(h));
                let s = sum[y1 * w1 + x1] + sum[y0 * w1 + x0] - sum[y0 * w1 + x1] - sum[y1 * w1 + x0];
                let n = cnt[y1 * w1 + x1] + cnt[y0 * w1 + x0] - cnt[y0 * w1 + x1] - cnt[y1 * w1 + x0];
                if n == 0 { 0.0 } else { (s / f64::from(n)) as f32 }
            })
            .collect::<Vec<f32>>()
    };
    (table(1, 0), table(0, 1))
}

/// The next coarser level: colour from a 3×3 Gaussian (σ = 1.5, as in the paper) over the known
/// cells, texture features by nearest-neighbour subsampling (blurring them would erase them), and a
/// cell is in the hole when any of the four it covers is.
fn downsample(l: &Level, cha: usize, ch: usize) -> Level {
    const K: [f32; 3] = [0.6412, 0.8007, 0.6412];
    let (w, h) = (l.w.div_ceil(2), l.h.div_ceil(2));
    let mut img = vec![0.0f32; w * h * cha];
    let mut hole = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x * 2, y * 2);
            let o = (y * w + x) * cha;
            hole[y * w + x] = (0..2).any(|dy| (0..2).any(|dx| fx + dx < l.w && fy + dy < l.h && l.hole[(fy + dy) * l.w + fx + dx]));
            let mut wsum = 0.0f32;
            for (ky, oy) in (-1i32..=1).enumerate() {
                for (kx, ox) in (-1i32..=1).enumerate() {
                    let (sx, sy) = (fx as i32 + ox, fy as i32 + oy);
                    if sx < 0 || sy < 0 || sx as usize >= l.w || sy as usize >= l.h {
                        continue;
                    }
                    let si = sy as usize * l.w + sx as usize;
                    if l.hole[si] {
                        continue;
                    }
                    let k = K[kx] * K[ky];
                    for c in 0..ch {
                        img[o + c] += l.img[si * cha + c] * k;
                    }
                    wsum += k;
                }
            }
            if wsum > 0.0 {
                img[o..o + ch].iter_mut().for_each(|v| *v /= wsum);
            }
            let si = (fy * l.w + fx) * cha;
            img[o + ch..o + cha].copy_from_slice(&l.img[si + ch..si + cha]);
        }
    }
    Level { w, h, img, hole }
}

/// How many levels: `⌈log2(2·No/N)⌉` (at least 1), `No` the hole's thickness in erosions and `N`
/// the patch side, while the coarsest level stays at least three patches across.
fn level_count(w: usize, h: usize, hole: &[bool], r: usize) -> usize {
    let n = 2 * r + 1;
    let no = crate::remove::hole_radius(w, h, hole).min(1 << 20);
    let want = (2.0 * f64::from(no) / n as f64).log2().ceil().max(1.0) as usize;
    let (mut lw, mut lh, mut levels) = (w, h, 1);
    while levels < want && lw / 2 >= 3 * n && lh / 2 >= 3 * n {
        (lw, lh, levels) = (lw.div_ceil(2), lh.div_ceil(2), levels + 1);
    }
    levels
}

/// Per-level search and reconstruction state.
struct Solver<'a> {
    w: usize,
    h: usize,
    cha: usize,
    /// Colour channels (the first `ch` of `cha`): the convergence test looks at these only.
    ch: usize,
    r: i32,
    img: Vec<f32>,
    hole: &'a [bool],
    valid: Vec<bool>,
    valid_list: Vec<(i32, i32)>,
    /// Patch centres whose patch overlaps the hole: the patches the energy is summed over.
    targets: Vec<bool>,
    ctl: &'a Interrupt<'a>,
}

impl Solver<'_> {
    #[inline]
    fn is_valid(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h && self.valid[y as usize * self.w + x as usize]
    }

    fn random_source(&self, rng: &mut Rng) -> (i32, i32) {
        let n = self.valid_list.len().max(1) as u64;
        self.valid_list.get((rng.next() % n) as usize).copied().unwrap_or((0, 0))
    }

    /// Mean squared difference between the target patch at `t` (clipped to the grid; with `known`,
    /// only its known cells) and the source patch at `s`. Gives up (∞) once it can't beat `cutoff`.
    fn dist(&self, t: (i32, i32), s: (i32, i32), cutoff: f32, known: Option<&[bool]>) -> f32 {
        let (w, h, cha, r) = (self.w as i32, self.h as i32, self.cha, self.r);
        let full = ((2 * r + 1) * (2 * r + 1)) as f32 * cha as f32;
        let (mut sum, mut n) = (0.0f32, 0usize);
        for dy in -r..=r {
            let ty = t.1 + dy;
            if ty < 0 || ty >= h {
                continue;
            }
            for dx in -r..=r {
                let tx = t.0 + dx;
                if tx < 0 || tx >= w {
                    continue;
                }
                let tc = ty as usize * self.w + tx as usize;
                if known.is_some_and(|k| !k[tc]) {
                    continue;
                }
                let (ti, si) = (tc * cha, ((s.1 + dy) as usize * self.w + (s.0 + dx) as usize) * cha);
                for c in 0..cha {
                    let d = self.img[ti + c] - self.img[si + c];
                    sum += d * d;
                }
                n += cha;
            }
            // The mean can only end up at or above sum / (full patch size).
            if known.is_none() && sum > cutoff * full {
                return f32::INFINITY;
            }
        }
        if n == 0 { f32::INFINITY } else { sum / n as f32 }
    }

    /// One PatchMatch pass over the targets, in parallel row bands (propagation stays in a band).
    fn patchmatch(&self, nnf: &mut [(i32, i32)], cost: &mut [f32], pass: usize, seed: u64, radius: i32) {
        const BAND: usize = 16;
        let w = self.w;
        let reverse = pass % 2 == 1;
        let band = |bi: usize, nn: &mut [(i32, i32)], co: &mut [f32]| {
            if self.ctl.cancelled() {
                return;
            }
            let mut rng = Rng::new(seed ^ (bi as u64).wrapping_mul(0x2545_F491_4F6C_DD1D) ^ ((pass as u64) << 40));
            let rows = nn.len() / w;
            let step: i32 = if reverse { -1 } else { 1 };
            for ry in 0..rows {
                let ly = if reverse { rows - 1 - ry } else { ry };
                let y = bi * BAND + ly;
                for rx in 0..w {
                    let x = if reverse { w - 1 - rx } else { rx };
                    if !self.targets[y * w + x] {
                        continue;
                    }
                    let li = ly * w + x;
                    let t = (x as i32, y as i32);
                    let (mut best, mut bc) = (nn[li], co[li]);
                    let try_cand = |cand: (i32, i32), best: &mut (i32, i32), bc: &mut f32| {
                        if cand != *best && self.is_valid(cand.0, cand.1) {
                            let d = self.dist(t, cand, *bc, None);
                            if d < *bc {
                                (*best, *bc) = (cand, d);
                            }
                        }
                    };
                    // Propagation from the neighbours visited before, within this band.
                    let nx = x as i32 - step;
                    if nx >= 0 && (nx as usize) < w {
                        let c = nn[ly * w + nx as usize];
                        try_cand((c.0 + step, c.1), &mut best, &mut bc);
                    }
                    let nly = ly as i32 - step;
                    if nly >= 0 && (nly as usize) < rows {
                        let c = nn[nly as usize * w + x];
                        try_cand((c.0, c.1 + step), &mut best, &mut bc);
                    }
                    // Random search in exponentially shrinking windows.
                    let mut rad = radius;
                    while rad >= 1 {
                        let cand = (best.0 + rng.range(-rad, rad), best.1 + rng.range(-rad, rad));
                        try_cand(cand, &mut best, &mut bc);
                        rad /= 2;
                    }
                    nn[li] = best;
                    co[li] = bc;
                }
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        if w * self.h >= PAR_MIN {
            use rayon::prelude::*;
            nnf.par_chunks_mut(BAND * w).zip(cost.par_chunks_mut(BAND * w)).enumerate().for_each(|(bi, (nn, co))| band(bi, nn, co));
            return;
        }
        for (bi, (nn, co)) in nnf.chunks_mut(BAND * w).zip(cost.chunks_mut(BAND * w)).enumerate() {
            band(bi, nn, co);
        }
    }

    /// Refresh every target's distance to its current match (hole cells change between iterations).
    fn refresh_costs(&self, nnf: &[(i32, i32)], cost: &mut [f32]) {
        let one = |(i, c): (usize, &mut f32)| {
            *c = if self.targets[i] { self.dist(((i % self.w) as i32, (i / self.w) as i32), nnf[i], f32::INFINITY, None) } else { f32::INFINITY };
        };
        #[cfg(not(target_arch = "wasm32"))]
        if self.w * self.h >= PAR_MIN {
            use rayon::prelude::*;
            cost.par_iter_mut().enumerate().for_each(one);
            return;
        }
        cost.iter_mut().enumerate().for_each(one);
    }

    /// `pick(x, y)` for every hole cell, in parallel rows: the cells it returns something for.
    fn per_hole_cell<T: Send>(&self, pick: impl Fn(usize, usize) -> Option<T> + Sync) -> Vec<(usize, T)> {
        let (w, h) = (self.w, self.h);
        let row = |y: usize| -> Vec<(usize, T)> {
            if self.ctl.cancelled() {
                return Vec::new();
            }
            (0..w).filter(|&x| self.hole[y * w + x]).filter_map(|x| pick(x, y).map(|v| (y * w + x, v))).collect()
        };
        #[cfg(not(target_arch = "wasm32"))]
        let rows: Vec<Vec<(usize, T)>> = if w * h >= PAR_MIN {
            use rayon::prelude::*;
            (0..h).into_par_iter().map(row).collect()
        } else {
            (0..h).map(row).collect()
        };
        #[cfg(target_arch = "wasm32")]
        let rows: Vec<Vec<(usize, T)>> = (0..h).map(row).collect();
        rows.into_iter().flatten().collect()
    }

    /// The cells of the patches covering hole cell `(x, y)`: each match's source cell for it, with
    /// that patch's cost.
    fn sources(&self, x: usize, y: usize, nnf: &[(i32, i32)], cost: &[f32], mut f: impl FnMut(usize, f32)) {
        let (w, h, r) = (self.w as i32, self.h as i32, self.r);
        for dy in -r..=r {
            let ty = y as i32 + dy;
            if ty < 0 || ty >= h {
                continue;
            }
            for dx in -r..=r {
                let tx = x as i32 + dx;
                if tx < 0 || tx >= w {
                    continue;
                }
                let ti = ty as usize * self.w + tx as usize;
                if !self.targets[ti] || !cost[ti].is_finite() {
                    continue;
                }
                let s = nnf[ti];
                let (sx, sy) = (s.0 - dx, s.1 - dy);
                if sx >= 0 && sy >= 0 && sx < w && sy < h {
                    f(sy as usize * self.w + sx as usize, cost[ti]);
                }
            }
        }
    }

    /// Re-estimate the hole by the similarity-weighted vote of the overlapping matches (`uniform`:
    /// all weights equal, for a field fresh from the coarser level whose costs mean nothing yet).
    /// Returns the mean absolute change of the colour channels.
    fn vote(&mut self, nnf: &[(i32, i32)], cost: &[f32], uniform: bool) -> f32 {
        let mut cs: Vec<f32> = cost.iter().zip(&self.targets).filter(|(c, t)| **t && c.is_finite()).map(|(c, _)| *c).collect();
        // σ² from the 75th percentile of the match costs, as Wexler et al. and the paper.
        let sigma2 = if cs.is_empty() {
            1.0
        } else {
            let k = (cs.len() * 3 / 4).min(cs.len() - 1);
            let (_, v, _) = cs.select_nth_unstable_by(k, f32::total_cmp);
            v.max(1e-6)
        };
        let cha = self.cha;
        let new = self.per_hole_cell(|x, y| {
            let mut v = vec![0.0f32; cha];
            let mut wsum = 0.0f32;
            self.sources(x, y, nnf, cost, |si, c| {
                let wt = if uniform { 1.0 } else { (-c / (2.0 * sigma2)).exp().max(1e-8) };
                for (o, s) in v.iter_mut().zip(&self.img[si * cha..(si + 1) * cha]) {
                    *o += s * wt;
                }
                wsum += wt;
            });
            (wsum > 0.0).then(|| {
                v.iter_mut().for_each(|o| *o /= wsum);
                v
            })
        });
        let (mut change, mut n) = (0.0f64, 0usize);
        for (i, v) in new {
            for (c, nv) in v.iter().enumerate().take(self.ch) {
                change += f64::from((nv - self.img[i * cha + c]).abs());
            }
            n += self.ch;
            self.img[i * cha..(i + 1) * cha].copy_from_slice(&v);
        }
        if n == 0 { 0.0 } else { (change / n as f64) as f32 }
    }

    /// The final reconstruction (the paper's eq. 5): each hole cell copies the source cell of the
    /// lowest-cost patch covering it. Returns `(hole cell, source cell, that patch's cost)`.
    fn copy_best(&mut self, nnf: &[(i32, i32)], cost: &[f32]) -> Vec<(usize, usize, f32)> {
        let cha = self.cha;
        let picked = self.per_hole_cell(|x, y| {
            let mut best: Option<(usize, f32)> = None;
            self.sources(x, y, nnf, cost, |si, c| {
                if best.is_none_or(|(_, bc)| c < bc) {
                    best = Some((si, c));
                }
            });
            best
        });
        for &(i, (si, _)) in &picked {
            self.img.copy_within(si * cha..(si + 1) * cha, i * cha);
        }
        picked.into_iter().map(|(i, (si, c))| (i, si, c)).collect()
    }

    /// Hide the colour seam along the hole's edge (gradient-domain blending, P. Pérez, M. Gangnet,
    /// A. Blake, *Poisson Image Editing*, SIGGRAPH 2003): at each known cell next to the hole, the
    /// difference between its value and the one the copied source predicts there (the source cell's
    /// own neighbour), as a median along the edge, is spread over the hole as a harmonic membrane
    /// and added to the fill. Only slow changes of brightness and colour move; the copied texture
    /// stays.
    fn blend_seams(&mut self, copied: &[(usize, usize, f32)], texture_scale: f32) {
        /// Radius of the median taken along the edge.
        const EDGE_MEDIAN: usize = 8;
        /// Largest shift the edge passes on, beyond twice its local texture.
        const MAX_SHIFT: f32 = 0.05;
        let (w, h, cha, ch) = (self.w, self.h, self.cha, self.ch);
        if copied.is_empty() {
            return;
        }
        let unscale = if texture_scale > 0.0 { 0.5 / texture_scale } else { 0.0 };
        // Work on the hole's bounding box grown by one cell, which holds the ring of known cells.
        let (x0, y0, x1, y1) =
            copied.iter().fold((w, h, 0, 0), |(x0, y0, x1, y1), (p, _, _)| (x0.min(p % w), y0.min(p / w), x1.max(p % w + 1), y1.max(p / w + 1)));
        let (x0, y0, x1, y1) = (x0.saturating_sub(1), y0.saturating_sub(1), (x1 + 1).min(w), (y1 + 1).min(h));
        let (bw, bh) = (x1.saturating_sub(x0), y1.saturating_sub(y0));
        let local = |i: usize| (i / w - y0) * bw + (i % w - x0);
        let img = &self.img;
        let hole = self.hole;
        let channel = |c: usize| -> Vec<f32> {
            let mut v = vec![0.0f32; bw * bh];
            let mut n = vec![0u8; bw * bh];
            for &(p, s, _) in copied {
                let (px, py, sx, sy) = ((p % w) as i32, (p / w) as i32, (s % w) as i32, (s / w) as i32);
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let (qx, qy, tx, ty) = (px + dx, py + dy, sx + dx, sy + dy);
                    let inside = |x: i32, y: i32| x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h;
                    if !inside(qx, qy) || !inside(tx, ty) {
                        continue;
                    }
                    let (q, t) = (qy as usize * w + qx as usize, ty as usize * w + tx as usize);
                    if hole[q] {
                        continue;
                    }
                    let lq = local(q);
                    v[lq] += img[q * cha + c] - img[t * cha + c];
                    n[lq] = n[lq].saturating_add(1);
                }
            }
            for (v, n) in v.iter_mut().zip(&n) {
                if *n > 0 {
                    *v /= f32::from(*n);
                }
            }
            // Each edge cell takes the median difference of the edge cells around it, capped at
            // `MAX_SHIFT` plus twice the local texture: a steady, moderate shift (a sky gradient, a
            // change of light) survives; a lone or large one (a structure crossing the edge, foam
            // meeting a dark source) would only glow into the fill.
            let raw = v.clone();
            let mut near = Vec::new();
            for lq in (0..bw * bh).filter(|&lq| n[lq] > 0) {
                let (qx, qy) = (lq % bw, lq / bw);
                near.clear();
                for y in qy.saturating_sub(EDGE_MEDIAN)..(qy + EDGE_MEDIAN + 1).min(bh) {
                    for x in qx.saturating_sub(EDGE_MEDIAN)..(qx + EDGE_MEDIAN + 1).min(bw) {
                        if n[y * bw + x] > 0 {
                            near.push(raw[y * bw + x]);
                        }
                    }
                }
                let mid = near.len() / 2;
                if !near.is_empty() {
                    let (_, m, _) = near.select_nth_unstable_by(mid, f32::total_cmp);
                    let q = (y0 + qy) * w + x0 + qx;
                    let texture = (img[q * cha + ch] + img[q * cha + ch + 1]) * unscale;
                    let cap = MAX_SHIFT + 2.0 * texture;
                    v[lq] = m.clamp(-cap, cap);
                }
            }
            // Known cells off the ring take the value of the nearest ring cell (breadth first), so
            // the solver's coarse levels, which average known cells, see the ring's values rather
            // than zeros; then the hole is solved.
            let mut queue: std::collections::VecDeque<usize> = (0..bw * bh).filter(|&lq| n[lq] > 0).collect();
            let mut seen: Vec<bool> = n.iter().map(|n| *n > 0).collect();
            let in_hole = |lq: usize| hole[(y0 + lq / bw) * w + x0 + lq % bw];
            while let Some(lq) = queue.pop_front() {
                let (x, y) = (lq % bw, lq / bw);
                let around = [(x > 0).then(|| lq - 1), (x + 1 < bw).then(|| lq + 1), (y > 0).then(|| lq - bw), (y + 1 < bh).then(|| lq + bw)];
                for nb in around.into_iter().flatten() {
                    if !seen[nb] && !in_hole(nb) {
                        seen[nb] = true;
                        v[nb] = v[lq];
                        queue.push_back(nb);
                    }
                }
            }
            let unknown: Vec<bool> = (0..bw * bh).map(in_hole).collect();
            crate::poisson::solve_membrane(bw, bh, &unknown, &mut v);
            v
        };
        #[cfg(not(target_arch = "wasm32"))]
        let shifts: Vec<Vec<f32>> = {
            use rayon::prelude::*;
            (0..ch).into_par_iter().map(channel).collect()
        };
        #[cfg(target_arch = "wasm32")]
        let shifts: Vec<Vec<f32>> = (0..ch).map(channel).collect();
        for &(p, _, _) in copied {
            let lp = local(p);
            for (c, shift) in shifts.iter().enumerate() {
                self.img[p * cha + c] += shift[lp];
            }
        }
    }

    /// Remove the visible seams between areas copied from different places (gradient-domain
    /// fusion, after S. Darabi et al., *Image Melding*, SIGGRAPH 2012). Across a seam, two
    /// neighbouring cells should differ as their sources do: as the better-matched side's source
    /// says. Where the actual step departs from that by more than `SEAM_CONTRAST` times the local
    /// texture `(Tx + Ty) / 2`, the seam shows (within a texture, where both steps are random and
    /// about that size, the departure is already about √2 times it). A correction `δ` is solved for those steps only (`Δδ = div e`, `δ = 0` at known
    /// cells, red-black SOR) and added to the fill: it is smooth away from the seams, so the copied
    /// texture keeps its grain while each step is spread out. Steps hidden by texture, and the edge
    /// of the hole (matched by `blend_seams`), add nothing.
    fn fuse_seams(&mut self, copied: &[(usize, usize, f32)], texture_scale: f32, iters: usize) {
        const SEAM_CONTRAST: f32 = 3.0;
        const OMEGA: f32 = 1.8;
        let (w, h, cha, ch) = (self.w, self.h, self.cha, self.ch);
        let mut src = vec![(usize::MAX, f32::INFINITY); w * h];
        for &(p, s, c) in copied {
            src[p] = (s, c);
        }
        let cell = |x: i32, y: i32| (x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h).then(|| y as usize * w + x as usize);
        let unscale = if texture_scale > 0.0 { 0.5 / texture_scale } else { 0.0 };
        // `div e` per copied cell and channel: the sum over its visible-seam edges of how far the
        // step to the neighbour is off (`e(p → q)`, and `−e` seen from q).
        let mut div = vec![0.0f32; w * h * ch];
        let mut any = false;
        for &(p, sp, cp) in copied {
            let (px, py) = ((p % w) as i32, (p / w) as i32);
            let texture = (self.img[p * cha + ch] + self.img[p * cha + ch + 1]) * unscale;
            for (dx, dy) in [(1, 0), (0, 1)] {
                let Some(q) = cell(px + dx, py + dy) else { continue };
                let (sq, cq) = src[q];
                if sq == usize::MAX {
                    continue;
                }
                // The step each side's source predicts (p's source one cell on, q's one cell back).
                let from_p = cell((sp % w) as i32 + dx, (sp / w) as i32 + dy);
                if from_p == Some(sq) {
                    continue;
                }
                let from_q = cell((sq % w) as i32 - dx, (sq / w) as i32 - dy);
                let (a, b) = match (from_p, from_q) {
                    (Some(n), Some(m)) => {
                        if cp <= cq {
                            (sp, n)
                        } else {
                            (m, sq)
                        }
                    }
                    (Some(n), None) => (sp, n),
                    (None, Some(m)) => (m, sq),
                    (None, None) => continue,
                };
                let e: Vec<f32> = (0..ch).map(|c| (self.img[b * cha + c] - self.img[a * cha + c]) - (self.img[q * cha + c] - self.img[p * cha + c])).collect();
                if e.iter().map(|v| v.abs()).sum::<f32>() / ch as f32 <= SEAM_CONTRAST * texture {
                    continue;
                }
                for (c, v) in e.iter().enumerate() {
                    div[p * ch + c] += v;
                    div[q * ch + c] -= v;
                }
                any = true;
            }
        }
        if !any {
            return;
        }
        // δ_p ← (Σ δ_q − div_p) / deg over the in-grid neighbours, δ = 0 at known cells. δ is
        // harmonic away from the seams and fades within a few dozen cells, so only cells within
        // `FUSE_BAND` of a seam are solved for.
        const FUSE_BAND: usize = 16;
        let seams: Vec<bool> = (0..w * h).map(|i| div[i * ch..(i + 1) * ch].iter().any(|v| *v != 0.0)).collect();
        let band = crate::remove::dilate(w, h, &seams, FUSE_BAND);
        let halves: [Vec<usize>; 2] = [0, 1].map(|colour| copied.iter().map(|(p, _, _)| *p).filter(|&p| band[p] && (p % w + p / w) % 2 == colour).collect());
        let mut delta = vec![0.0f32; w * h * ch];
        let mut next: Vec<f32> = Vec::new();
        for _ in 0..iters {
            if self.ctl.cancelled() {
                return;
            }
            for cells in &halves {
                next.clear();
                next.resize(cells.len() * ch, 0.0);
                let d = &delta;
                let relax = |(&p, out): (&usize, &mut [f32])| {
                    let (x, y) = ((p % w) as i32, (p / w) as i32);
                    let mut deg = 0.0f32;
                    out.fill(0.0);
                    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                        let Some(q) = cell(x + dx, y + dy) else { continue };
                        deg += 1.0;
                        out.iter_mut().zip(&d[q * ch..(q + 1) * ch]).for_each(|(s, v)| *s += v);
                    }
                    for (c, o) in out.iter_mut().enumerate() {
                        let cur = d[p * ch + c];
                        *o = cur + OMEGA * ((*o - div[p * ch + c]) / deg.max(1.0) - cur);
                    }
                };
                #[cfg(not(target_arch = "wasm32"))]
                {
                    use rayon::prelude::*;
                    cells.par_iter().zip(next.par_chunks_mut(ch)).for_each(relax);
                }
                #[cfg(target_arch = "wasm32")]
                cells.iter().zip(next.chunks_mut(ch)).for_each(relax);
                for (p, v) in cells.iter().zip(next.chunks(ch)) {
                    delta[p * ch..(p + 1) * ch].copy_from_slice(v);
                }
            }
        }
        for &(p, _, _) in copied {
            for c in 0..ch {
                self.img[p * cha + c] += delta[p * ch + c];
            }
        }
    }

    /// Onion-peel initialisation (the paper's §3.4): fill the hole one boundary layer at a time.
    /// Each known patch centre next to the layer gets a match by its known cells; each layer cell
    /// is then the weighted vote of those matches. Leaves those matches in `nnf` (`has` marks them).
    fn onion_peel(&mut self, nnf: &mut [(i32, i32)], has: &mut [bool], seed: u64) -> Result<(), Cancelled> {
        let (w, h, r, cha) = (self.w, self.h, self.r, self.cha);
        let mut known: Vec<bool> = self.hole.iter().map(|m| !*m).collect();
        // The cells still unknown, and per layer the known centres found (lists, so a layer costs
        // its own size rather than a scan of the grid).
        let mut unknown: Vec<usize> = (0..w * h).filter(|&i| self.hole[i]).collect();
        let mut centre = vec![false; w * h];
        let mut layer_cost = vec![f32::INFINITY; w * h];
        let mut layer_no = 0u64;
        loop {
            self.ctl.check()?;
            let neighbour_known = |i: usize, known: &[bool]| {
                let (x, y) = (i % w, i / w);
                (x > 0 && known[i - 1]) || (x + 1 < w && known[i + 1]) || (y > 0 && known[i - w]) || (y + 1 < h && known[i + w])
            };
            let layer: Vec<usize> = unknown.iter().copied().filter(|&i| neighbour_known(i, &known)).collect();
            if layer.is_empty() {
                return Ok(());
            }
            // Known centres whose patch reaches into the layer.
            let mut centres = Vec::new();
            for &i in &layer {
                let (x, y) = ((i % w) as i32, (i / w) as i32);
                for cy in (y - r).max(0)..=(y + r).min(h as i32 - 1) {
                    for cx in (x - r).max(0)..=(x + r).min(w as i32 - 1) {
                        let ci = cy as usize * w + cx as usize;
                        if known[ci] && !centre[ci] {
                            centre[ci] = true;
                            centres.push(ci);
                        }
                    }
                }
            }
            for &ci in &centres {
                centre[ci] = false;
            }
            let this = &*self;
            let known_ref = &known;
            let (nnf_ref, has_ref) = (&*nnf, &*has);
            let search = |&ci: &usize| -> (usize, (i32, i32), f32) {
                let t = ((ci % w) as i32, (ci / w) as i32);
                let mut rng = Rng::new(seed ^ (ci as u64).wrapping_mul(0x9E37_79B9) ^ (layer_no << 48));
                let (mut best, mut bc) = ((0, 0), f32::INFINITY);
                let try_cand = |cand: (i32, i32), best: &mut (i32, i32), bc: &mut f32| {
                    if this.is_valid(cand.0, cand.1) {
                        let d = this.dist(t, cand, *bc, Some(known_ref));
                        if d < *bc {
                            (*best, *bc) = (cand, d);
                        }
                    }
                };
                if has_ref[ci] {
                    try_cand(nnf_ref[ci], &mut best, &mut bc);
                }
                // Neighbours' matches, shifted (propagation), and a few random sources.
                for (ox, oy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let (nx, ny) = (t.0 + ox, t.1 + oy);
                    if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h && has_ref[ny as usize * w + nx as usize] {
                        let s = nnf_ref[ny as usize * w + nx as usize];
                        try_cand((s.0 - ox, s.1 - oy), &mut best, &mut bc);
                    }
                }
                for _ in 0..8 {
                    try_cand(this.random_source(&mut rng), &mut best, &mut bc);
                }
                for _ in 0..2 {
                    let mut rad = w.max(h) as i32;
                    while rad >= 1 {
                        try_cand((best.0 + rng.range(-rad, rad), best.1 + rng.range(-rad, rad)), &mut best, &mut bc);
                        rad /= 2;
                    }
                }
                (ci, best, bc)
            };
            #[cfg(not(target_arch = "wasm32"))]
            let found: Vec<(usize, (i32, i32), f32)> = {
                use rayon::prelude::*;
                centres.par_iter().map(search).collect()
            };
            #[cfg(target_arch = "wasm32")]
            let found: Vec<(usize, (i32, i32), f32)> = centres.iter().map(search).collect();
            let mut cs = Vec::with_capacity(found.len());
            for &(ci, s, c) in &found {
                if c.is_finite() {
                    nnf[ci] = s;
                    has[ci] = true;
                    layer_cost[ci] = c;
                    cs.push(c);
                }
            }
            let sigma2 = if cs.is_empty() {
                1.0
            } else {
                let k = (cs.len() * 3 / 4).min(cs.len() - 1);
                let (_, v, _) = cs.select_nth_unstable_by(k, f32::total_cmp);
                v.max(1e-6)
            };
            let mut values = Vec::with_capacity(layer.len());
            for &i in &layer {
                let (x, y) = ((i % w) as i32, (i / w) as i32);
                let mut v = vec![0.0f32; cha];
                let mut wsum = 0.0f32;
                for cy in (y - r).max(0)..=(y + r).min(h as i32 - 1) {
                    for cx in (x - r).max(0)..=(x + r).min(w as i32 - 1) {
                        let ci = cy as usize * w + cx as usize;
                        if !layer_cost[ci].is_finite() {
                            continue;
                        }
                        let s = nnf[ci];
                        let (sx, sy) = (s.0 + x - cx, s.1 + y - cy);
                        if sx < 0 || sy < 0 || sx as usize >= w || sy as usize >= h {
                            continue;
                        }
                        let si = sy as usize * w + sx as usize;
                        let wt = (-layer_cost[ci] / (2.0 * sigma2)).exp().max(1e-8);
                        for (o, s) in v.iter_mut().zip(&self.img[si * cha..(si + 1) * cha]) {
                            *o += s * wt;
                        }
                        wsum += wt;
                    }
                }
                if wsum > 0.0 {
                    v.iter_mut().for_each(|o| *o /= wsum);
                    values.push((i, v));
                }
            }
            for &(ci, _, _) in &found {
                layer_cost[ci] = f32::INFINITY;
            }
            if values.is_empty() {
                // No known patch reaches the layer: leave the rest to the EM iterations.
                return Ok(());
            }
            for (i, v) in values {
                self.img[i * cha..(i + 1) * cha].copy_from_slice(&v);
                known[i] = true;
            }
            unknown.retain(|&i| !known[i]);
            layer_no += 1;
        }
    }
}

/// Complete `hole` in `img` (see the module docs). `Ok(None)` when no fully-known patch exists to
/// copy from; `Err(Cancelled)` when `ctl` was cancelled. A buffer or mask of the wrong size, or no
/// hole, comes back unchanged.
pub fn complete_with(w: usize, h: usize, ch: usize, img: &[f32], hole: &[bool], p: &Params, ctl: &Interrupt) -> Result<Option<Vec<f32>>, Cancelled> {
    let n = w.saturating_mul(h);
    if ch == 0 || hole.len() != n || img.len() != n.saturating_mul(ch) || !hole.iter().any(|m| *m) {
        return Ok(Some(img.to_vec()));
    }
    let r = p.patch_radius.max(1);
    let levels = level_count(w, h, hole, r);
    // Texture features at full resolution, averaged over about a coarsest-level cell (`2^levels`).
    let (tx, ty) = texture_features(w, h, ch, img, hole, 1 << levels.min(8));
    let cha = ch + 2;
    let k = p.lambda.max(0.0).sqrt();
    let mut data = Vec::with_capacity(n * cha);
    for (i, px) in img.chunks_exact(ch).enumerate() {
        data.extend_from_slice(px);
        data.extend_from_slice(&[tx[i] * k, ty[i] * k]);
    }
    let mut pyramid = vec![Level { w, h, img: data, hole: hole.to_vec() }];
    while pyramid.len() < levels {
        ctl.check()?;
        let Some(next) = pyramid.last().map(|l| downsample(l, cha, ch)) else { break };
        // A level with nothing to copy from can't seed the finer ones: stop above it.
        if valid_sources(next.w, next.h, &next.hole, r).1.is_empty() {
            break;
        }
        pyramid.push(next);
    }
    let total_px: usize = pyramid.iter().map(|l| l.w * l.h).sum();
    let mut done_px = 0usize;
    let mut prev: Option<(usize, Vec<(i32, i32)>)> = None;
    let mut result = None;
    for (li, level) in pyramid.iter().enumerate().rev() {
        ctl.check()?;
        let (lw, lh) = (level.w, level.h);
        let (valid, valid_list) = valid_sources(lw, lh, &level.hole, r);
        if valid_list.is_empty() {
            return Ok(None);
        }
        let ri = r as i32;
        let mut targets = vec![false; lw * lh];
        for (i, _) in level.hole.iter().enumerate().filter(|(_, m)| **m) {
            let (x, y) = ((i % lw) as i32, (i / lw) as i32);
            for ty in (y - ri).max(0)..=(y + ri).min(lh as i32 - 1) {
                for tx in (x - ri).max(0)..=(x + ri).min(lw as i32 - 1) {
                    targets[ty as usize * lw + tx as usize] = true;
                }
            }
        }
        let mut s = Solver { w: lw, h: lh, cha, ch, r: ri, img: level.img.clone(), hole: &level.hole, valid, valid_list, targets, ctl };
        let mut nnf = vec![(0i32, 0i32); lw * lh];
        let mut cost = vec![f32::INFINITY; lw * lh];
        let mut rng = Rng::new(p.seed ^ ((li as u64) << 20));
        match prev.take() {
            None => {
                // Coarsest level: onion peel, then random matches for the targets it didn't reach.
                let mut has = vec![false; lw * lh];
                s.onion_peel(&mut nnf, &mut has, p.seed ^ 0x0c10_4e00)?;
                for i in 0..lw * lh {
                    if s.targets[i] && !has[i] {
                        nnf[i] = s.random_source(&mut rng);
                    }
                }
            }
            Some((pw, pnnf)) => {
                // Upsample the coarser match field, then rebuild the hole from it.
                for y in 0..lh {
                    for x in 0..lw {
                        let i = y * lw + x;
                        if !s.targets[i] {
                            continue;
                        }
                        let c = pnnf.get((y / 2) * pw + x / 2).copied().unwrap_or((-1, -1));
                        let cand = (c.0 * 2 + (x % 2) as i32, c.1 * 2 + (y % 2) as i32);
                        nnf[i] = if s.is_valid(cand.0, cand.1) {
                            cand
                        } else if s.is_valid(c.0 * 2, c.1 * 2) {
                            (c.0 * 2, c.1 * 2)
                        } else {
                            s.random_source(&mut rng)
                        };
                    }
                }
                // Uniform weights: the costs only need to be finite.
                for (c, t) in cost.iter_mut().zip(&s.targets) {
                    if *t {
                        *c = 0.0;
                    }
                }
                s.vote(&nnf, &cost, true);
            }
        }
        // Wide random search at the coarsest level, where the structure is decided; finer levels
        // refine the upsampled field locally.
        let coarsest = li + 1 == pyramid.len();
        let full = lw.max(lh) as i32;
        for it in 0..p.max_iters.max(1) {
            ctl.check()?;
            s.refresh_costs(&nnf, &mut cost);
            let radius = if coarsest || it == 0 { full } else { (8 * (2 * ri + 1)).min(full) };
            for pass in 0..p.pm_passes.max(1) {
                ctl.check()?;
                s.patchmatch(&mut nnf, &mut cost, pass + it * p.pm_passes, p.seed.wrapping_add((li * 1000 + it) as u64), radius);
            }
            ctl.check()?;
            let change = s.vote(&nnf, &cost, false);
            ctl.check()?;
            if change < p.tolerance {
                break;
            }
        }
        if li == 0 {
            s.refresh_costs(&nnf, &mut cost);
            let copied = s.copy_best(&nnf, &cost);
            ctl.check()?;
            s.blend_seams(&copied, k);
            ctl.check()?;
            s.fuse_seams(&copied, k, 100);
            ctl.check()?;
            result = Some(s.img);
        }
        done_px += lw * lh;
        ctl.progress(done_px as f32 / total_px.max(1) as f32);
        prev = Some((lw, nnf));
    }
    let Some(data) = result else { return Ok(None) };
    // The colour channels; known cells exactly as given.
    let mut out = img.to_vec();
    for (i, m) in hole.iter().enumerate() {
        if *m {
            out[i * ch..(i + 1) * ch].copy_from_slice(&data[i * cha..i * cha + ch]);
        }
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(i: usize) -> f32 {
        ((i as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Mean |∂x| + |∂y| over the masked cells (one channel).
    fn grain(w: usize, img: &[f32], m: &[bool]) -> f32 {
        let (mut s, mut n) = (0.0, 0);
        for (i, _) in m.iter().enumerate().filter(|(_, m)| **m) {
            if i % w + 1 < w && i + w < img.len() {
                s += (img[i + 1] - img[i]).abs() + (img[i + w] - img[i]).abs();
                n += 1;
            }
        }
        s / n.max(1) as f32
    }

    /// The failure that motivates the texture features: a hole in noise next to a flat area of the
    /// same mean must be filled with noise, not with the flat area.
    #[test]
    fn a_hole_in_texture_next_to_a_flat_area_of_the_same_mean_gets_texture() {
        let (w, h) = (96, 64);
        let img: Vec<f32> = (0..w * h).map(|i| if i % w < 48 { 0.5 } else { 0.25 + 0.5 * noise(i) }).collect();
        let hole: Vec<bool> = (0..w * h).map(|i| (60..84).contains(&(i % w)) && (20..44).contains(&(i / w))).collect();
        let out = complete_with(w, h, 1, &img, &hole, &Params::default(), &Interrupt::NONE).unwrap().unwrap();
        let (filled, truth) = (grain(w, &out, &hole), grain(w, &img, &hole));
        assert!(filled > truth * 0.7, "the fill is smoother than the texture around it: {filled} vs {truth}");
        assert!(out.iter().zip(&img).zip(&hole).all(|((o, i), m)| *m || o == i), "known cells are kept");
    }

    #[test]
    fn a_hole_in_a_gradient_follows_the_gradient() {
        let (w, h, ch) = (80, 60, 2);
        let img: Vec<f32> = (0..w * h).flat_map(|i| [(i % w) as f32 / w as f32, 1.0]).collect();
        let hole: Vec<bool> = (0..w * h).map(|i| (30..50).contains(&(i % w)) && (20..40).contains(&(i / w))).collect();
        let out = complete_with(w, h, ch, &img, &hole, &Params::default(), &Interrupt::NONE).unwrap().unwrap();
        for (i, _) in hole.iter().enumerate().filter(|(_, m)| **m) {
            assert!((out[i * ch] - img[i * ch]).abs() < 0.1, "cell {i}: {} vs {}", out[i * ch], img[i * ch]);
        }
    }

    #[test]
    fn levels_follow_the_hole_thickness() {
        let (w, h) = (400, 400);
        let line: Vec<bool> = (0..w * h).map(|i| i / w == 200 && (20..380).contains(&(i % w))).collect();
        assert_eq!(level_count(w, h, &line, 3), 1, "a thin line stays at full resolution");
        let disc: Vec<bool> = (0..w * h).map(|i| ((i % w) as f32 - 200.0).hypot((i / w) as f32 - 200.0) < 90.0).collect();
        assert_eq!(level_count(w, h, &disc, 3), 5, "⌈log2(2·90/7)⌉");
    }

    #[test]
    fn nothing_to_copy_from_and_bad_sizes() {
        let img = vec![0.5f32; 12 * 12];
        assert_eq!(complete_with(12, 12, 1, &img, &[true; 144], &Params::default(), &Interrupt::NONE).unwrap(), None);
        assert_eq!(complete_with(12, 12, 1, &img, &[true; 3], &Params::default(), &Interrupt::NONE).unwrap(), Some(img.clone()));
        assert_eq!(complete_with(12, 12, 0, &img, &[true; 144], &Params::default(), &Interrupt::NONE).unwrap(), Some(img.clone()));
    }

    #[test]
    fn cancelled_is_an_error() {
        let (w, h) = (64, 64);
        let img: Vec<f32> = (0..w * h).map(noise).collect();
        let hole: Vec<bool> = (0..w * h).map(|i| (20..40).contains(&(i % w)) && (20..40).contains(&(i / w))).collect();
        let cancel = || true;
        let progress = |_: f32| {};
        assert_eq!(complete_with(w, h, 1, &img, &hole, &Params::default(), &Interrupt::new(&cancel, &progress)), Err(Cancelled));
    }
}
