//! Multiresolution blending for Edit › Auto-Blend Layers.
//!
//! P. J. Burt, E. H. Adelson, *A Multiresolution Spline With Application to Image Mosaics*,
//! ACM TOG 1983: each image is split into a Laplacian pyramid (band-pass levels plus a low-pass
//! residual), the blend weights into a Gaussian pyramid, every level is blended with its own
//! smoothed weights and the result collapsed, so seams are invisible at every scale.
//!
//! Weights come from [`stack_weights`] (focus stacking: per pixel, the sharpest image wins, by
//! the smoothed magnitude of the luminance Laplacian) or [`panorama_weights`] (each pixel goes to
//! the image whose opaque area it is deepest inside).
//!
//! [`fuse_stack`] is the focus stack's merge: E. H. Adelson, C. H. Anderson, J. R. Bergen,
//! P. J. Burt, J. M. Ogden, *Pyramid methods in image processing*, RCA Engineer 1984 (the
//! multifocus composite: per pyramid coefficient, keep the image with the most band-pass energy,
//! then expand-and-add) with the selection rules of W. Wang, F. Chang, *A Multi-focus Image Fusion
//! Method Based on Laplacian Pyramid*, Journal of Computers 6(12), 2011 (maximum region energy
//! for the band-pass levels, local deviation and entropy for the residual), plus halo control.
//!
//! Buffers are interleaved `w × h × ch` normalised floats of any colour model and depth;
//! deterministic.

/// One pyramid level.
#[derive(Clone, Debug, PartialEq)]
pub struct Level {
    pub w: usize,
    pub h: usize,
    pub px: Vec<f32>,
}

use crate::photo_util::{par_map, par_rows, par_rows2};

const K: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

/// 5-tap binomial filter + decimation by two (Burt & Adelson's REDUCE).
pub fn reduce(l: &Level, ch: usize) -> Level {
    let (w, h) = (l.w.div_ceil(2), l.h.div_ceil(2));
    // Horizontal pass on even columns, then vertical on even rows.
    let mut tmp = vec![0.0f32; w * l.h * ch];
    par_rows(&mut tmp, w, ch, |y, trow| {
        for x in 0..w {
            for (k, wt) in K.iter().enumerate() {
                let sx = (2 * x as i64 + k as i64 - 2).clamp(0, l.w as i64 - 1) as usize;
                for c in 0..ch {
                    trow[x * ch + c] += wt * l.px[(y * l.w + sx) * ch + c];
                }
            }
        }
    });
    let mut px = vec![0.0f32; w * h * ch];
    par_rows(&mut px, w, ch, |y, prow| {
        for (k, wt) in K.iter().enumerate() {
            let sy = (2 * y as i64 + k as i64 - 2).clamp(0, l.h as i64 - 1) as usize;
            for x in 0..w {
                for c in 0..ch {
                    prow[x * ch + c] += wt * tmp[(sy * w + x) * ch + c];
                }
            }
        }
    });
    Level { w, h, px }
}

/// Upsample `l` to `w × h` (EXPAND: zero insertion + the same filter, ×4 gain).
pub fn expand(l: &Level, ch: usize, w: usize, h: usize) -> Vec<f32> {
    let mut tmp = vec![0.0f32; w * l.h * ch];
    par_rows(&mut tmp, w, ch, |y, trow| {
        for x in 0..w {
            for (k, wt) in K.iter().enumerate() {
                let t = x as i64 + k as i64 - 2;
                if t.rem_euclid(2) != 0 {
                    continue;
                }
                let sx = (t / 2).clamp(0, l.w as i64 - 1) as usize;
                for c in 0..ch {
                    trow[x * ch + c] += 2.0 * wt * l.px[(y * l.w + sx) * ch + c];
                }
            }
        }
    });
    let mut out = vec![0.0f32; w * h * ch];
    par_rows(&mut out, w, ch, |y, orow| {
        for (k, wt) in K.iter().enumerate() {
            let t = y as i64 + k as i64 - 2;
            if t.rem_euclid(2) != 0 {
                continue;
            }
            let sy = (t / 2).clamp(0, l.h as i64 - 1) as usize;
            for x in 0..w {
                for c in 0..ch {
                    orow[x * ch + c] += 2.0 * wt * tmp[(sy * w + x) * ch + c];
                }
            }
        }
    });
    out
}

/// Gaussian pyramid with `levels` levels (level 0 = the input).
pub fn gaussian(w: usize, h: usize, ch: usize, px: &[f32], levels: usize) -> Vec<Level> {
    let mut out = vec![Level { w, h, px: px.to_vec() }];
    while out.len() < levels.max(1) {
        let Some(last) = out.last() else { break };
        if last.w <= 1 && last.h <= 1 {
            break;
        }
        let next = reduce(last, ch);
        out.push(next);
    }
    out
}

/// Laplacian pyramid: band-pass levels, the last level is the low-pass residual.
pub fn laplacian(w: usize, h: usize, ch: usize, px: &[f32], levels: usize) -> Vec<Level> {
    let g = gaussian(w, h, ch, px, levels);
    let mut out = Vec::with_capacity(g.len());
    for i in 0..g.len() {
        if i + 1 == g.len() {
            out.push(g[i].clone());
        } else {
            let up = expand(&g[i + 1], ch, g[i].w, g[i].h);
            out.push(Level { w: g[i].w, h: g[i].h, px: g[i].px.iter().zip(&up).map(|(a, b)| a - b).collect() });
        }
    }
    out
}

/// Rebuild an image from its Laplacian pyramid.
pub fn collapse(pyr: &[Level], ch: usize) -> Vec<f32> {
    let mut cur = pyr.last().cloned().unwrap_or(Level { w: 0, h: 0, px: Vec::new() });
    for l in pyr.iter().rev().skip(1) {
        let up = expand(&cur, ch, l.w, l.h);
        cur = Level { w: l.w, h: l.h, px: l.px.iter().zip(&up).map(|(a, b)| a + b).collect() };
    }
    cur.px
}

/// Number of levels so the coarsest level is a few pixels across.
pub fn auto_levels(w: usize, h: usize) -> usize {
    let mut n = 1;
    let mut s = w.min(h);
    while s > 8 && n < 10 {
        s /= 2;
        n += 1;
    }
    n
}

/// Blend `images` with per-pixel `weights` (any non-negative values; normalised per level).
pub fn blend(w: usize, h: usize, ch: usize, images: &[&[f32]], weights: &[Vec<f32>], levels: usize) -> Vec<f32> {
    assert_eq!(images.len(), weights.len());
    let levels = levels.max(1);
    let mut acc: Option<Vec<Level>> = None;
    let mut wsum: Option<Vec<Level>> = None;
    for (img, wt) in images.iter().zip(weights) {
        let lp = laplacian(w, h, ch, img, levels);
        let gp = gaussian(w, h, 1, wt, levels);
        let a = acc.get_or_insert_with(|| lp.iter().map(|l| Level { w: l.w, h: l.h, px: vec![0.0; l.px.len()] }).collect());
        let s = wsum.get_or_insert_with(|| gp.iter().map(|l| Level { w: l.w, h: l.h, px: vec![0.0; l.px.len()] }).collect());
        for (li, (l, g)) in lp.iter().zip(&gp).enumerate() {
            for i in 0..l.w * l.h {
                let k = g.px[i];
                s[li].px[i] += k;
                for c in 0..ch {
                    a[li].px[i * ch + c] += k * l.px[i * ch + c];
                }
            }
        }
    }
    let (Some(mut a), Some(s)) = (acc, wsum) else { return vec![0.0; w * h * ch] };
    for (l, sl) in a.iter_mut().zip(&s) {
        for i in 0..l.w * l.h {
            let k = sl.px[i];
            for c in 0..ch {
                l.px[i * ch + c] = if k > 1e-6 { l.px[i * ch + c] / k } else { 0.0 };
            }
        }
    }
    collapse(&a, ch).into_iter().map(|v| v.clamp(0.0, 1.0)).collect()
}

/// Focus-stack fusion settings for [`fuse_stack`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StackFusion {
    /// Region-energy window radius for the band-pass levels: 1 = the 3×3 binomial window of
    /// Wang & Chang; 0 = the per-node |L| of Adelson et al.
    pub energy_radius: usize,
    /// Halo control hardness: 0 = off (every level picks its own winner); otherwise the levels
    /// coarser than `guide` are the mean of the images weighed by the guide's region energy to
    /// this power, capped at [`HALO_MAX`]. Default 2.
    pub halo: f32,
    /// The guide level for halo control (0 = the finest band-pass level); clamped to the last
    /// band-pass level.
    pub guide: usize,
    /// Pyramid levels, the residual included; `None` = [`stack_levels`].
    pub levels: Option<usize>,
}

impl Default for StackFusion {
    fn default() -> Self {
        StackFusion { energy_radius: 1, halo: 2.0, guide: 2, levels: None }
    }
}

/// Largest useful halo hardness: a weight's exponent is clamped to ±60 (a span of 1e52), which
/// hardness 8 uses up over the practical range of energies.
pub const HALO_MAX: f32 = 8.0;
/// Reference energy the halo weights are relative to (a well-textured quarter-resolution band),
/// so that `(RE / ref)^p` stays within f32 up to [`HALO_MAX`].
const HALO_REF: f32 = 1e-4;
/// Floor added to the energy before weighing: below it the images average.
const HALO_FLOOR: f32 = 1e-8;
/// Coverage below which an image has no say at a pixel.
const COVERED: f32 = 0.5;
/// The energy of an image where it does not cover the pixel: below any real energy, so any
/// covering image wins over it.
const NO_ENERGY: f32 = -1.0;

/// Levels for [`fuse_stack`]: as many band-pass levels as keep the residual's short side ≥ 32
/// px, so the residual still has the local structure its deviation / entropy rule measures.
pub fn stack_levels(w: usize, h: usize) -> usize {
    let mut n = 1;
    let mut s = w.min(h);
    while s.div_ceil(2) >= 32 && n < 10 {
        s = s.div_ceil(2);
        n += 1;
    }
    n
}

/// The weight a coarse coefficient gets from the guide's region energy `re` at hardness `p`:
/// `((re + floor) / ref)^p`, its exponent clamped to ±60.
fn halo_weight(re: f32, p: f32) -> f32 {
    (p * ((re.max(0.0) + HALO_FLOOR) / HALO_REF).ln()).clamp(-60.0, 60.0).exp()
}

/// Binomial window weights of radius `r` (row 2r of Pascal's triangle / 4^r).
fn binomial(r: usize) -> Vec<f32> {
    let n = 2 * r + 1;
    let mut row = vec![1u64; n];
    for i in 1..n {
        for j in (1..i).rev() {
            row[j] += row[j - 1];
        }
    }
    let s = row.iter().sum::<u64>() as f32;
    row.into_iter().map(|v| v as f32 / s).collect()
}

/// Separable weighted window sum of a plane (`wt` of odd length, clamped borders).
fn window_sum(w: usize, h: usize, src: &[f32], wt: &[f32]) -> Vec<f32> {
    let r = wt.len() / 2;
    if r == 0 || w == 0 || h == 0 {
        return src.to_vec();
    }
    let mut tmp = vec![0.0f32; w * h];
    par_rows(&mut tmp, w, 1, |y, row| {
        let s = &src[y * w..y * w + w];
        for (x, o) in row.iter_mut().enumerate() {
            *o = wt.iter().enumerate().map(|(t, k)| k * s[(x + t).saturating_sub(r).min(w - 1)]).sum();
        }
    });
    let mut out = vec![0.0f32; w * h];
    par_rows(&mut out, w, 1, |y, row| {
        for (t, k) in wt.iter().enumerate() {
            let sy = (y + t).saturating_sub(r).min(h - 1);
            for (o, v) in row.iter_mut().zip(&tmp[sy * w..sy * w + w]) {
                *o += k * v;
            }
        }
    });
    out
}

/// Region energy of a band-pass luminance level: the window sum of L² (Wang & Chang eq. 13);
/// `NO_ENERGY` where the image's coverage `cover` is below [`COVERED`].
fn region_energy(l: &Level, cover: &Level, win: &[f32]) -> Vec<f32> {
    let e: Vec<f32> = l.px.iter().map(|v| v * v).collect();
    let mut re = window_sum(l.w, l.h, &e, win);
    for (r, a) in re.iter_mut().zip(&cover.px) {
        if *a < COVERED {
            *r = NO_ENERGY;
        }
    }
    re
}

/// Local deviation (Wang & Chang eq. 10) and local entropy (eq. 11, 256 gray levels) of a
/// plane over a 5×5 window, clamped borders.
fn deviation_entropy(w: usize, h: usize, y: &[f32]) -> (Vec<f32>, Vec<f32>) {
    const R: usize = 2;
    const BINS: usize = 256;
    let n = (2 * R + 1) * (2 * R + 1);
    let inv_n = 1.0 / n as f32;
    let (mut dev, mut ent) = (vec![0.0f32; w * h], vec![0.0f32; w * h]);
    if w == 0 || h == 0 {
        return (dev, ent);
    }
    par_rows2(&mut dev, &mut ent, w, |i, drow, erow| {
        let mut hist = [0u32; BINS];
        let mut touched: Vec<usize> = Vec::with_capacity(n);
        for j in 0..w {
            let (mut s, mut s2) = (0.0f32, 0.0f32);
            touched.clear();
            for dy in 0..=2 * R {
                let yy = (i + dy).saturating_sub(R).min(h - 1);
                for dx in 0..=2 * R {
                    let v = y[yy * w + (j + dx).saturating_sub(R).min(w - 1)];
                    s += v;
                    s2 += v * v;
                    let q = ((v.clamp(0.0, 1.0) * (BINS - 1) as f32 + 0.5) as usize).min(BINS - 1);
                    if hist[q] == 0 {
                        touched.push(q);
                    }
                    hist[q] += 1;
                }
            }
            let mean = s * inv_n;
            drow[j] = (s2 * inv_n - mean * mean).max(0.0);
            let mut e = 0.0f32;
            for &q in &touched {
                let p = hist[q] as f32 * inv_n;
                e -= p * p.ln();
                hist[q] = 0;
            }
            erow[j] = e;
        }
    });
    (dev, ent)
}

/// One image's residual (low-pass) level, with its luminance and coverage at that level.
struct Residual {
    px: Level,
    luma: Vec<f32>,
    cover: Vec<f32>,
}

/// Fuse the images' residuals by Wang & Chang's region-information rule (eq. 12), generalised
/// to N images as a Pareto rule: an image is dominated where another covering image is at least
/// as good on both local deviation and local entropy and strictly better on one; the fused value
/// is the mean of the non-dominated images. Images that do not cover the pixel are left out
/// unless none does.
fn fuse_residuals(tops: &[Residual], ch: usize) -> Level {
    let Some(t0) = tops.first() else { return Level { w: 0, h: 0, px: Vec::new() } };
    let (w, h, n) = (t0.px.w, t0.px.h, tops.len());
    let measures: Vec<(Vec<f32>, Vec<f32>)> = par_map(n, |k| deviation_entropy(w, h, &tops[k].luma));
    let mut out = vec![0.0f32; w * h * ch];
    let mut keep = vec![false; n];
    for i in 0..w * h {
        let any_cover = tops.iter().any(|t| t.cover[i] >= COVERED);
        let eligible = |a: usize| !any_cover || tops[a].cover[i] >= COVERED;
        let mut kept = 0usize;
        for a in 0..n {
            let (da, ea) = (measures[a].0[i], measures[a].1[i]);
            let dominated = eligible(a)
                && (0..n).any(|b| {
                    let (db, eb) = (measures[b].0[i], measures[b].1[i]);
                    b != a && eligible(b) && db >= da && eb >= ea && (db > da || eb > ea)
                });
            keep[a] = eligible(a) && !dominated;
            kept += keep[a] as usize;
        }
        let inv = 1.0 / kept.max(1) as f32;
        for (t, _) in tops.iter().zip(&keep).filter(|(_, k)| **k) {
            for c in 0..ch {
                out[i * ch + c] += t.px.px[i * ch + c] * inv;
            }
        }
    }
    Level { w, h, px: out }
}

/// Halo control: fold the levels coarser than the guide `g` as `Σ w·L` and `Σ w`, the guide's
/// weights `wgt` REDUCEd to each level's size. `src` is the image's pyramid; `None` scales the
/// first image's levels, already in `acc`, in place.
fn fold_halo(acc: &mut [Level], wsum: &mut Vec<Vec<f32>>, src: Option<&[Level]>, g: usize, mut wgt: Level, ch: usize) {
    for li in g + 1..acc.len() {
        wgt = reduce(&wgt, 1);
        let a = &mut acc[li];
        let (aw, wp) = (a.w, &wgt.px);
        match src {
            Some(src) => {
                if let Some(s) = wsum.get_mut(li - g - 1) {
                    for (s, k) in s.iter_mut().zip(wp) {
                        *s += k;
                    }
                }
                let sp = &src[li].px;
                par_rows(&mut a.px, aw, ch, |y, row| {
                    for (x, px) in row.chunks_mut(ch).enumerate() {
                        let (i, k) = (y * aw + x, wp[y * aw + x]);
                        for (c, v) in px.iter_mut().enumerate() {
                            *v += k * sp[i * ch + c];
                        }
                    }
                });
            }
            None => {
                par_rows(&mut a.px, aw, ch, |y, row| {
                    for (x, px) in row.chunks_mut(ch).enumerate() {
                        let k = wp[y * aw + x];
                        px.iter_mut().for_each(|v| *v *= k);
                    }
                });
                wsum.push(wp.clone());
            }
        }
    }
}

/// Multifocus composite of `images` (interleaved `w × h × ch`; `luma[k]` is image k's luminance,
/// `alpha[k]` its coverage): per Laplacian-pyramid coefficient, the image whose luminance band
/// has the largest region energy wins (Adelson et al. 1984's multifocus composite with the
/// maximum-region-energy rule of Wang & Chang 2011), the residual follows the deviation +
/// entropy rule, and the blending between images happens in the reconstruction itself, so every
/// scale is taken from the image that is sharp at that scale. Ties keep the earlier image, and an
/// image has no say where its coverage is below one half.
///
/// With halo control (`p.halo` > 0) the levels coarser than `p.guide` do not pick winners:
/// beside a bright object, the images focused behind it carry the object's defocused copy spread
/// over the background, strong coarse energy where the image that has the object sharp has
/// none, so a per-level pick collects that glow. Instead those levels are the mean of the images
/// weighed by the guide's region energy to the power `p.halo` (REDUCEd to each level's size, as
/// a multiresolution spline blends with a weight mask), so the coarse structure follows the images
/// the guide found sharp, and where none is the images average.
///
/// Images are folded in one at a time: only the running fused pyramid, one best-energy (or
/// weight-sum) plane per level and the small residuals are held. Output is clamped to 0..=1.
pub fn fuse_stack(w: usize, h: usize, ch: usize, images: &[&[f32]], luma: &[Vec<f32>], alpha: &[Vec<f32>], p: StackFusion) -> Vec<f32> {
    // Levels the pyramid can actually have (REDUCE stops at 1 × 1).
    let levels = {
        let want = p.levels.unwrap_or_else(|| stack_levels(w, h)).max(1);
        let (mut n, mut a, mut b) = (1, w, h);
        while (a > 1 || b > 1) && n < want {
            (a, b, n) = (a.div_ceil(2), b.div_ceil(2), n + 1);
        }
        n
    };
    let bands = levels - 1;
    let halo = (bands > 0 && p.halo > 0.0).then(|| (p.guide.min(bands - 1), p.halo.min(HALO_MAX)));
    // The band-pass levels that pick their own winner: all of them, or up to the guide.
    let nsel = halo.map_or(bands, |(g, _)| g + 1);
    let win = binomial(p.energy_radius);
    let mut acc: Vec<Level> = Vec::new();
    // Per selecting level, the winning energy so far.
    let mut best: Vec<Vec<f32>> = Vec::new();
    // With halo control, per level coarser than the guide, the weight sum so far.
    let mut wsum: Vec<Vec<f32>> = Vec::new();
    let mut tops: Vec<Residual> = Vec::new();
    for ((img, lm), al) in images.iter().zip(luma).zip(alpha) {
        let lp = laplacian(w, h, ch, img, levels);
        let ll = laplacian(w, h, 1, lm, levels);
        let ga = gaussian(w, h, 1, al, levels);
        let (Some(top), Some(tl), Some(ta)) = (lp.last(), ll.last(), ga.last()) else { continue };
        if lp.len() != levels || ll.len() != levels || ga.len() != levels {
            continue;
        }
        if halo.is_none() {
            tops.push(Residual { px: top.clone(), luma: tl.px.clone(), cover: ta.px.clone() });
        }
        let energies: Vec<Vec<f32>> = (0..nsel).map(|li| region_energy(&ll[li], &ga[li], &win)).collect();
        let wgt =
            halo.map(|(g, hard)| Level { w: ll[g].w, h: ll[g].h, px: energies[g].iter().map(|&e| if e < 0.0 { 0.0 } else { halo_weight(e, hard) }).collect() });
        if acc.is_empty() {
            acc = lp;
            best = energies;
            if let (Some((g, _)), Some(wgt)) = (halo, wgt) {
                fold_halo(&mut acc, &mut wsum, None, g, wgt, ch);
            }
            continue;
        }
        for li in 0..nsel {
            let (lvl, lw) = (&lp[li], lp[li].w);
            let take: Vec<bool> = energies[li]
                .iter()
                .zip(best[li].iter_mut())
                .map(|(e, b)| {
                    if *e > *b {
                        *b = *e;
                        true
                    } else {
                        false
                    }
                })
                .collect();
            par_rows(&mut acc[li].px, lw, ch, |y, row| {
                for (x, px) in row.chunks_mut(ch).enumerate() {
                    if take[y * lw + x] {
                        px.copy_from_slice(&lvl.px[(y * lw + x) * ch..(y * lw + x + 1) * ch]);
                    }
                }
            });
        }
        if let (Some((g, _)), Some(wgt)) = (halo, wgt) {
            fold_halo(&mut acc, &mut wsum, Some(&lp), g, wgt, ch);
        }
    }
    if acc.is_empty() {
        return vec![0.0; w * h * ch];
    }
    match halo {
        Some((g, _)) => {
            // The weighted means: Σ w·L / Σ w.
            for li in g + 1..levels {
                let (a, s) = (&mut acc[li], &wsum[li - g - 1]);
                let aw = a.w;
                par_rows(&mut a.px, aw, ch, |y, row| {
                    for (x, px) in row.chunks_mut(ch).enumerate() {
                        let k = s[y * aw + x];
                        px.iter_mut().for_each(|v| *v = if k > 0.0 { *v / k } else { 0.0 });
                    }
                });
            }
        }
        None => {
            if let Some(last) = acc.last_mut() {
                *last = fuse_residuals(&tops, ch);
            }
        }
    }
    collapse(&acc, ch).into_iter().map(|v| v.clamp(0.0, 1.0)).collect()
}

fn box_blur1(w: usize, h: usize, v: &[f32], r: usize) -> Vec<f32> {
    let pass = |src: &[f32], horizontal: bool| -> Vec<f32> {
        let mut dst = vec![0.0f32; src.len()];
        let (n, m) = if horizontal { (h, w) } else { (w, h) };
        for line in 0..n {
            let at = |k: i64| {
                let k = k.clamp(0, m as i64 - 1) as usize;
                if horizontal { src[line * w + k] } else { src[k * w + line] }
            };
            let mut s: f32 = (-(r as i64)..=r as i64).map(at).sum();
            for k in 0..m {
                let i = if horizontal { line * w + k } else { k * w + line };
                dst[i] = s / (2 * r + 1) as f32;
                s += at(k as i64 + r as i64 + 1) - at(k as i64 - r as i64);
            }
        }
        dst
    };
    pass(&pass(v, true), false)
}

/// One-hot winner maps: weight 1 for the image with the largest score at each pixel.
fn winner_take_all(scores: &[Vec<f32>], n: usize) -> Vec<Vec<f32>> {
    let mut out = vec![vec![0.0f32; n]; scores.len()];
    for i in 0..n {
        let mut best = (0usize, f32::MIN);
        for (k, s) in scores.iter().enumerate() {
            if s[i] > best.1 {
                best = (k, s[i]);
            }
        }
        if best.1 > f32::MIN {
            out[best.0][i] = 1.0;
        }
    }
    out
}

/// Focus-stack weights: per pixel, the image with the most local detail (smoothed |∇²L|) wins.
/// `luma[k]` is image k's luminance, `alpha[k]` its coverage.
pub fn stack_weights(w: usize, h: usize, luma: &[Vec<f32>], alpha: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let radius = (w.min(h) / 64).clamp(2, 12);
    let scores: Vec<Vec<f32>> = luma
        .iter()
        .zip(alpha)
        .map(|(l, a)| {
            let mut lap = vec![0.0f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    let p = |xx: usize, yy: usize| l[yy * w + xx];
                    let (xl, xr, yu, yd) = (x.saturating_sub(1), (x + 1).min(w - 1), y.saturating_sub(1), (y + 1).min(h - 1));
                    lap[y * w + x] = (p(xl, y) + p(xr, y) + p(x, yu) + p(x, yd) - 4.0 * p(x, y)).abs();
                }
            }
            box_blur1(w, h, &lap, radius).iter().zip(a).map(|(s, a)| if *a > 0.5 { *s } else { -1.0 }).collect()
        })
        .collect();
    winner_take_all(&scores, w * h)
}

/// Panorama weights: each pixel goes to the image whose opaque area it lies deepest inside
/// (chamfer distance to that image's transparent pixels), so seams run through overlaps.
pub fn panorama_weights(w: usize, h: usize, alpha: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let scores: Vec<Vec<f32>> = alpha
        .iter()
        .map(|a| {
            // Two-pass 3-4 chamfer distance transform from transparent pixels / the border.
            let inf = 1.0e9f32;
            let mut d: Vec<f32> = a.iter().map(|v| if *v > 0.5 { inf } else { 0.0 }).collect();
            let at = |d: &[f32], x: i64, y: i64| if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 { 0.0 } else { d[y as usize * w + x as usize] };
            for y in 0..h as i64 {
                for x in 0..w as i64 {
                    let i = y as usize * w + x as usize;
                    let m = d[i].min(at(&d, x - 1, y) + 3.0).min(at(&d, x, y - 1) + 3.0).min(at(&d, x - 1, y - 1) + 4.0).min(at(&d, x + 1, y - 1) + 4.0);
                    d[i] = m;
                }
            }
            for y in (0..h as i64).rev() {
                for x in (0..w as i64).rev() {
                    let i = y as usize * w + x as usize;
                    let m = d[i].min(at(&d, x + 1, y) + 3.0).min(at(&d, x, y + 1) + 3.0).min(at(&d, x + 1, y + 1) + 4.0).min(at(&d, x - 1, y + 1) + 4.0);
                    d[i] = m;
                }
            }
            d.iter().map(|v| if *v <= 0.0 { -1.0 } else { *v }).collect()
        })
        .collect();
    winner_take_all(&scores, w * h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn laplacian_round_trip_is_exact() {
        let (w, h) = (37, 23);
        let img: Vec<f32> = (0..w * h * 2).map(|i| ((i * 37 % 101) as f32) / 101.0).collect();
        let pyr = laplacian(w, h, 2, &img, 5);
        assert_eq!(pyr.len(), 5);
        let back = collapse(&pyr, 2);
        assert!(img.iter().zip(&back).all(|(a, b)| (a - b).abs() < 1e-4));
    }

    #[test]
    fn blend_with_one_hot_weights_reproduces_each_side() {
        let (w, h) = (64, 32);
        let a = vec![0.2f32; w * h];
        let b = vec![0.8f32; w * h];
        let wa: Vec<f32> = (0..w * h).map(|i| if i % w < 32 { 1.0 } else { 0.0 }).collect();
        let wb: Vec<f32> = wa.iter().map(|v| 1.0 - v).collect();
        let out = blend(w, h, 1, &[&a, &b], &[wa, wb], auto_levels(w, h));
        assert!((out[5 * w + 2] - 0.2).abs() < 1e-3);
        assert!((out[5 * w + 61] - 0.8).abs() < 1e-3);
        // The seam is a smooth ramp, not a step.
        let mid = out[5 * w + 31];
        assert!(mid > 0.3 && mid < 0.7, "{mid}");
    }

    #[test]
    fn stack_picks_the_sharp_image() {
        let (w, h) = (48, 48);
        // Image 0 is sharp on the left (checker), image 1 on the right.
        let checker = |x: usize, y: usize| if (x / 2 + y / 2).is_multiple_of(2) { 0.1 } else { 0.9 };
        let i0: Vec<f32> = (0..w * h).map(|i| if i % w < 24 { checker(i % w, i / w) } else { 0.5 }).collect();
        let i1: Vec<f32> = (0..w * h).map(|i| if i % w >= 24 { checker(i % w, i / w) } else { 0.5 }).collect();
        let ones = vec![1.0f32; w * h];
        let wts = stack_weights(w, h, &[i0, i1], &[ones.clone(), ones]);
        assert_eq!(wts[0][20 * w + 4], 1.0);
        assert_eq!(wts[1][20 * w + 44], 1.0);
    }

    #[test]
    fn panorama_splits_the_overlap() {
        let (w, h) = (60, 20);
        let a: Vec<f32> = (0..w * h).map(|i| if i % w < 40 { 1.0 } else { 0.0 }).collect();
        let b: Vec<f32> = (0..w * h).map(|i| if i % w >= 20 { 1.0 } else { 0.0 }).collect();
        let wts = panorama_weights(w, h, &[a, b]);
        assert_eq!(wts[0][10 * w + 5], 1.0);
        assert_eq!(wts[1][10 * w + 55], 1.0);
        // The seam falls inside the overlap (20..40), near its middle.
        let seam = (0..w).find(|x| wts[1][10 * w + x] == 1.0).unwrap();
        assert!((25..=35).contains(&seam), "{seam}");
    }

    fn checker_img(w: usize, h: usize, cell: usize, ch: usize) -> Vec<f32> {
        let mut px = vec![0.0f32; w * h * ch];
        for y in 0..h {
            for x in 0..w {
                let v = if ((x / cell) + (y / cell)).is_multiple_of(2) { 0.2 } else { 0.8 };
                let p = &mut px[(y * w + x) * ch..(y * w + x + 1) * ch];
                for (c, o) in p.iter_mut().enumerate() {
                    *o = match c {
                        0 => v,
                        1 => v * 0.9,
                        2 => v * 0.5,
                        _ => 1.0,
                    };
                }
            }
        }
        px
    }

    fn luma_of(w: usize, h: usize, ch: usize, px: &[f32]) -> Vec<f32> {
        (0..w * h).map(|i| if ch >= 3 { 0.2126 * px[i * ch] + 0.7152 * px[i * ch + 1] + 0.0722 * px[i * ch + 2] } else { px[i * ch] }).collect()
    }

    /// Blur every channel with the 5-tap kernel `passes` times (no decimation).
    fn blur_img(w: usize, h: usize, ch: usize, px: &[f32], passes: usize) -> Vec<f32> {
        let mut planes: Vec<Vec<f32>> = (0..ch).map(|c| (0..w * h).map(|i| px[i * ch + c]).collect()).collect();
        for _ in 0..passes {
            for p in planes.iter_mut() {
                *p = window_sum(w, h, p, &K);
            }
        }
        (0..w * h * ch).map(|i| planes[i % ch][i / ch]).collect()
    }

    fn rms(a: &[f32], b: &[f32]) -> f32 {
        (a.iter().zip(b).map(|(x, y)| ((x - y) * (x - y)) as f64).sum::<f64>() / a.len() as f64).sqrt() as f32
    }

    #[test]
    fn stack_levels_keep_the_residual_above_32_px() {
        assert_eq!(stack_levels(64, 64), 2);
        assert_eq!(stack_levels(48, 100), 1);
        assert_eq!(stack_levels(2070, 1378), 6);
        assert_eq!(stack_levels(8280, 5520), 8);
        assert_eq!(stack_levels(100_000, 100_000), 10);
    }

    #[test]
    fn binomial_window_and_window_sum() {
        assert_eq!(binomial(0), vec![1.0]);
        assert_eq!(binomial(1), vec![0.25, 0.5, 0.25]);
        assert!(binomial(2).iter().zip(&K).all(|(a, k)| (a - k).abs() < 1e-6));
        let flat = vec![0.3f32; 7 * 5];
        assert!(window_sum(7, 5, &flat, &binomial(1)).iter().all(|v| (v - 0.3).abs() < 1e-6), "a window sum of unit weight keeps a flat plane");
        assert_eq!(window_sum(7, 5, &flat, &binomial(0)), flat);
    }

    #[test]
    fn deviation_and_entropy_measure_local_structure() {
        let (w, h) = (20, 20);
        let flat = vec![0.5f32; w * h];
        let (d, e) = deviation_entropy(w, h, &flat);
        assert!(d.iter().all(|v| *v == 0.0) && e.iter().all(|v| *v == 0.0));
        let check = luma_of(w, h, 1, &checker_img(w, h, 1, 1));
        let (d, e) = deviation_entropy(w, h, &check);
        assert!(d[10 * w + 10] > 0.08 && e[10 * w + 10] > 0.6, "{} {}", d[10 * w + 10], e[10 * w + 10]);
    }

    #[test]
    fn fuse_stack_identical_images_are_a_fixed_point() {
        let (w, h, ch) = (70, 50, 3);
        let im = checker_img(w, h, 7, ch);
        let lm = luma_of(w, h, ch, &im);
        let ones = vec![1.0f32; w * h];
        let three = StackFusion { levels: Some(3), ..Default::default() };
        for p in [three, StackFusion { halo: 0.0, ..three }, StackFusion { guide: 1, ..three }, StackFusion { energy_radius: 0, ..three }] {
            let out = fuse_stack(w, h, ch, &[&im, &im, &im], &[lm.clone(), lm.clone(), lm.clone()], &[ones.clone(), ones.clone(), ones.clone()], p);
            assert!(rms(&out, &im) < 1e-5, "{p:?}: {}", rms(&out, &im));
        }
        // One image, and none, are fine too.
        assert!(rms(&fuse_stack(w, h, ch, &[&im], std::slice::from_ref(&lm), std::slice::from_ref(&ones), three), &im) < 1e-5);
        assert_eq!(fuse_stack(w, h, ch, &[], &[], &[], three), vec![0.0; w * h * ch]);
        // More levels than the image has are clamped.
        let many = StackFusion { levels: Some(40), ..Default::default() };
        assert!(rms(&fuse_stack(w, h, ch, &[&im], &[lm], &[ones], many), &im) < 1e-5);
    }

    #[test]
    fn fuse_stack_picks_the_sharp_half_from_each_image() {
        let (w, h, ch) = (96, 64, 4);
        let sharp = checker_img(w, h, 6, ch);
        let soft = blur_img(w, h, ch, &sharp, 6);
        // Image A is sharp on the left and blurred on the right, image B the opposite.
        let (mut a, mut b) = (sharp.clone(), sharp.clone());
        for i in 0..w * h {
            for c in 0..ch {
                if i % w >= w / 2 {
                    a[i * ch + c] = soft[i * ch + c];
                } else {
                    b[i * ch + c] = soft[i * ch + c];
                }
            }
        }
        let (la, lb) = (luma_of(w, h, ch, &a), luma_of(w, h, ch, &b));
        let ones = vec![1.0f32; w * h];
        for p in [
            StackFusion { halo: 0.0, ..Default::default() },
            StackFusion { halo: 0.0, energy_radius: 0, ..Default::default() },
            StackFusion { guide: 0, ..Default::default() },
        ] {
            let out = fuse_stack(w, h, ch, &[&a, &b], &[la.clone(), lb.clone()], &[ones.clone(), ones.clone()], p);
            let err = rms(&out, &sharp);
            assert!(err < 0.3 * rms(&a, &sharp), "{p:?}: fused rms {err}");
            assert!(out.iter().skip(3).step_by(ch).all(|a| (a - 1.0).abs() < 1e-4), "alpha stays opaque");
            // Away from the seam each side is the sharp image. Without halo control the residual
            // (one REDUCE here) averages where the blurred side's entropy beats the checker's, so
            // the finer scales are exact and the whole is within a few percent; with halo control
            // the guide's energy weighs the residual too.
            let (e1, e2) = ((out[(20 * w + 10) * ch] - sharp[(20 * w + 10) * ch]).abs(), (out[(20 * w + 85) * ch + 1] - sharp[(20 * w + 85) * ch + 1]).abs());
            let tol = if p.halo > 0.0 { 1e-3 } else { 0.1 };
            assert!(e1 < tol && e2 < tol, "{p:?}: {e1} {e2}");
        }
        // The old full-resolution weights and spline blend do worse on the same pair.
        let wts = stack_weights(w, h, &[la, lb], &[ones.clone(), ones]);
        let old = blend(w, h, ch, &[&a, &b], &wts, auto_levels(w, h));
        let new =
            fuse_stack(w, h, ch, &[&a, &b], &[luma_of(w, h, ch, &a), luma_of(w, h, ch, &b)], &[vec![1.0; w * h], vec![1.0; w * h]], StackFusion::default());
        assert!(rms(&new, &sharp) <= rms(&old, &sharp), "new {} old {}", rms(&new, &sharp), rms(&old, &sharp));
    }

    #[test]
    fn fuse_stack_ignores_uncovered_pixels() {
        let (w, h, ch) = (96, 64, 4);
        let sharp = checker_img(w, h, 6, ch);
        // Image A: flat gray everywhere. Image B: the sharp checker on the left half, garbage
        // (opaque white in the colour channels, alpha 0) on the right.
        let a: Vec<f32> = (0..w * h * ch).map(|i| if i % ch == 3 { 1.0 } else { 0.5 }).collect();
        let b: Vec<f32> = (0..w * h * ch)
            .map(|i| {
                if (i / ch) % w < w / 2 {
                    sharp[i]
                } else if i % ch == 3 {
                    0.0
                } else {
                    1.0
                }
            })
            .collect();
        let alpha_b: Vec<f32> = (0..w * h).map(|i| if i % w < w / 2 { 1.0 } else { 0.0 }).collect();
        let (la, lb) = (luma_of(w, h, ch, &a), luma_of(w, h, ch, &b));
        for p in [StackFusion { halo: 0.0, ..Default::default() }, StackFusion { guide: 0, ..Default::default() }] {
            let out = fuse_stack(w, h, ch, &[&a, &b], &[la.clone(), lb.clone()], &[vec![1.0; w * h], alpha_b.clone()], p);
            // Left: the checker; right: A's gray, not B's white.
            assert!((out[(30 * w + 12) * ch] - sharp[(30 * w + 12) * ch]).abs() < 0.05, "{p:?}");
            for x in [60, 70, 90] {
                let px = &out[(30 * w + x) * ch..(30 * w + x + 1) * ch];
                assert!(px[..3].iter().all(|v| (v - 0.5).abs() < 0.03), "{p:?}: x {x} {px:?}");
            }
        }
    }

    /// The halo mechanism in one piece: image A is sharp fine texture; image B is the texture
    /// defocused with a broad bright bump on it (coarse structure only, as a defocused copy of a
    /// bright object has). Every level picking its own winner takes the bump, since B alone has
    /// energy at the coarse levels; with halo control the guide finds A sharp everywhere, so the
    /// coarse levels follow A and the bump stays out.
    #[test]
    fn halo_control_keeps_the_coarse_levels_with_the_guide() {
        let (w, h, ch) = (192, 144, 3);
        let a = checker_img(w, h, 6, ch); // a 12 px period: the guide (level 2, 4 px) sees it
        let mut b = blur_img(w, h, ch, &a, 64); // σ ≈ 8 px: the texture is gone
        for y in 0..h {
            for x in 0..w {
                let d2 = (x as f32 - 96.0).powi(2) + (y as f32 - 72.0).powi(2);
                let bump = 0.35 * (-d2 / (2.0 * 20.0f32.powi(2))).exp();
                for c in 0..ch {
                    let v = &mut b[(y * w + x) * ch + c];
                    *v = (*v + bump).min(1.0);
                }
            }
        }
        let ones = vec![1.0f32; w * h];
        let fuse = |halo: f32, imgs: [&[f32]; 2]| {
            let l = [luma_of(w, h, ch, imgs[0]), luma_of(w, h, ch, imgs[1])];
            fuse_stack(w, h, ch, &imgs, &l, &[ones.clone(), ones.clone()], StackFusion { halo, guide: 2, levels: Some(5), ..Default::default() })
        };
        let off = rms(&fuse(0.0, [&a, &b]), &a);
        assert!(off > 0.05, "without halo control the bump gets in: rms {off}");
        for halo in [1.0, 2.0, 4.0, 8.0, 100.0] {
            let on = rms(&fuse(halo, [&a, &b]), &a);
            assert!(on < 0.15 * off, "halo {halo}: rms {on} vs {off} without");
        }
        // The order of the images makes no difference.
        assert!(rms(&fuse(2.0, [&b, &a]), &a) < 0.15 * off);
        assert!((halo_weight(HALO_REF - HALO_FLOOR, 3.0) - 1.0).abs() < 1e-5);
        assert!(halo_weight(1e-3, 2.0) > halo_weight(1e-4, 2.0));
        assert!(halo_weight(0.0, 8.0) > 0.0 && halo_weight(1e3, 8.0).is_finite());
    }

    #[test]
    fn residual_rule_takes_the_image_that_wins_deviation_and_entropy() {
        // With a single level the whole image is the residual: the checker (2 px cells, so every
        // 5×5 window, the clamped corner ones included, straddles an edge) wins both measures over
        // the flat image everywhere, so the fused result is the checker.
        let (w, h, ch) = (40, 40, 3);
        let a = checker_img(w, h, 2, ch);
        let b = vec![0.5f32; w * h * ch];
        let ones = vec![1.0f32; w * h];
        let one = StackFusion { levels: Some(1), halo: 0.0, ..Default::default() };
        let out = fuse_stack(w, h, ch, &[&b, &a], &[luma_of(w, h, ch, &b), luma_of(w, h, ch, &a)], &[ones.clone(), ones.clone()], one);
        assert!(rms(&out, &a) < 1e-6);
        // Identical measures average (the Pareto rule keeps both).
        let out = fuse_stack(w, h, ch, &[&b, &a], &[luma_of(w, h, ch, &a), luma_of(w, h, ch, &a)], &[ones.clone(), ones], one);
        let mean: Vec<f32> = a.iter().zip(&b).map(|(x, y)| (x + y) / 2.0).collect();
        assert!(rms(&out, &mean) < 1e-6);
    }
}
