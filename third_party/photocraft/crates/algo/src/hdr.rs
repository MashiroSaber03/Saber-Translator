//! File › Automate › Merge to HDR Pro: exposure estimation, alignment, camera response recovery,
//! radiance merge, ghost removal and the 16/8-bit tone mapping methods.
//!
//! * **Alignment**: G. Ward, *Fast, Robust Image Registration for Compositing High Dynamic Range
//!   Photographs from Hand-Held Exposures*, JGT 2003 — median threshold bitmaps (invariant to
//!   exposure) with an exclusion band around the median, XOR-counted over an image pyramid.
//! * **Response curve**: P. Debevec, J. Malik, *Recovering High Dynamic Range Radiance Maps from
//!   Photographs*, SIGGRAPH 1997 — `g(Z) = ln E + ln Δt` solved by weighted linear least squares
//!   with a smoothness term and `g(128) = 0`, per channel, on stratified sample pixels.
//! * **Merge**: the hat-weighted average of `g(Z) − ln Δt` over the exposures.
//! * **Ghosts**: where the per-exposure radiance estimates disagree (high weighted variance of
//!   their logarithms) the result comes from one base exposure only, as Photoshop's *Remove
//!   ghosts* does; the mask is feathered.
//! * **Tone mapping** for 16/8-bit output: Local Adaptation ([`crate::tone::hdr_toning`]),
//!   Exposure and Gamma, Highlight Compression (E. Reinhard et al., *Photographic Tone
//!   Reproduction for Digital Images*, SIGGRAPH 2002, global operator) and Equalize Histogram
//!   (histogram equalisation of log luminance, after G. Ward Larson et al., *A Visibility
//!   Matching Tone Reproduction Operator*, TVCG 1997, without the ceiling).
//!
//! Pixels are straight RGBA in `0..=1` (display-encoded inputs); radiance is linear.

#![allow(clippy::needless_range_loop)] // index loops mirror the maths (pixels × channels)

use crate::photo_util::{Normal, linear_to_srgb, par_rows, srgb_to_linear};
use crate::tone::HdrToning;

fn luma(c: [f32; 4]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Hat weight on a 0..=255 code value.
fn hat(z: f64) -> f64 {
    if z <= 127.5 { z } else { 255.0 - z }.max(0.0)
}

/// Relative exposures (linear factors, the darkest = 1) estimated from the images themselves:
/// consecutive images (sorted by brightness) are compared on pixels well exposed in both.
pub fn estimate_exposures(imgs: &[&[[f32; 4]]]) -> Vec<f64> {
    let n = imgs.len();
    let mean: Vec<f64> = imgs.iter().map(|im| im.iter().map(|p| luma(*p) as f64).sum::<f64>() / im.len().max(1) as f64).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| mean[a].total_cmp(&mean[b]));
    let mut ex = vec![1.0f64; n];
    for k in 1..n {
        let (a, b) = (order[k - 1], order[k]);
        let mut ratios: Vec<f64> = imgs[a]
            .iter()
            .zip(imgs[b].iter())
            .step_by(7)
            .filter_map(|(pa, pb)| {
                let (la, lb) = (luma(*pa), luma(*pb));
                ((0.08..0.92).contains(&la) && (0.08..0.92).contains(&lb)).then(|| srgb_to_linear(lb) as f64 / srgb_to_linear(la).max(1e-6) as f64)
            })
            .collect();
        ratios.sort_by(f64::total_cmp);
        let r = if ratios.is_empty() { 2.0 } else { ratios[ratios.len() / 2].max(1.0) };
        ex[b] = ex[a] * r;
    }
    ex
}

// ---------- alignment ----------

fn shrink(w: usize, h: usize, g: &[f32]) -> (usize, usize, Vec<f32>) {
    let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
    let mut out = vec![0.0f32; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            let p = |xx: usize, yy: usize| g[yy.min(h - 1) * w + xx.min(w - 1)];
            out[y * nw + x] = 0.25 * (p(2 * x, 2 * y) + p(2 * x + 1, 2 * y) + p(2 * x, 2 * y + 1) + p(2 * x + 1, 2 * y + 1));
        }
    }
    (nw, nh, out)
}

fn bitmaps(g: &[f32]) -> (Vec<bool>, Vec<bool>) {
    let mut s: Vec<f32> = g.to_vec();
    s.sort_by(f32::total_cmp);
    let med = s[s.len() / 2];
    (g.iter().map(|v| *v > med).collect(), g.iter().map(|v| (v - med).abs() > 4.0 / 255.0).collect())
}

/// Translation `(dx, dy)` that aligns grey image `img` to `reference` (Ward's MTB), searching
/// up to `2^levels` px.
pub fn mtb_offset(w: usize, h: usize, reference: &[f32], img: &[f32], levels: usize) -> (i32, i32) {
    if levels == 0 || w < 16 || h < 16 {
        return (0, 0);
    }
    let (cur_x, cur_y) = {
        let (sw, sh, sr) = shrink(w, h, reference);
        let (_, _, si) = shrink(w, h, img);
        let (x, y) = mtb_offset(sw, sh, &sr, &si, levels - 1);
        (x * 2, y * 2)
    };
    let (tr, er) = bitmaps(reference);
    let (ti, ei) = bitmaps(img);
    let mut best = (u64::MAX, 0, 0);
    for dy in -1..=1 {
        for dx in -1..=1 {
            let (ox, oy) = (cur_x + dx, cur_y + dy);
            let mut err = 0u64;
            for y in 0..h as i32 {
                let sy = y - oy;
                if sy < 0 || sy >= h as i32 {
                    continue;
                }
                for x in 0..w as i32 {
                    let sx = x - ox;
                    if sx < 0 || sx >= w as i32 {
                        continue;
                    }
                    let (a, b) = ((y as usize) * w + x as usize, (sy as usize) * w + sx as usize);
                    if (tr[a] != ti[b]) && er[a] && ei[b] {
                        err += 1;
                    }
                }
            }
            if err < best.0 {
                best = (err, ox, oy);
            }
        }
    }
    (best.1, best.2)
}

// ---------- response curve ----------

/// Sample pixel indices for response recovery: a spatial grid filtered to pixels whose
/// neighbourhood is flat in the middle exposure (robust to small misalignment), stratified by
/// its intensity.
pub fn sample_pixels(w: usize, h: usize, middle: &[[f32; 4]], count: usize) -> Vec<usize> {
    let mut by_bin: Vec<Vec<usize>> = vec![Vec::new(); 32];
    let step = ((w * h / (count * 40).max(1)) as f64).sqrt().max(1.0) as usize;
    for y in (2..h.saturating_sub(2)).step_by(step) {
        for x in (2..w.saturating_sub(2)).step_by(step) {
            let i = y * w + x;
            let l = luma(middle[i]);
            let flat = [i - 2, i + 2, i - 2 * w, i + 2 * w].iter().all(|&j| (luma(middle[j]) - l).abs() < 0.04);
            if flat {
                by_bin[((l * 31.99) as usize).min(31)].push(i);
            }
        }
    }
    // Round-robin over the bins so every intensity is represented.
    let mut out = Vec::new();
    let mut k = 0;
    while out.len() < count && by_bin.iter().any(|b| k < b.len()) {
        for b in &by_bin {
            if let Some(&i) = b.get(k * 7 % b.len().max(1)).filter(|_| k < b.len()) {
                out.push(i);
                if out.len() >= count {
                    break;
                }
            }
        }
        k += 1;
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Debevec–Malik response curve: `z[j][i]` is the code value (0..=255) of sample `i` in exposure
/// `j`; returns `g(0..=255)` (log exposure per code value, `g(128) = 0`), smoothed by `lambda`.
pub fn response_curve(z: &[Vec<u8>], log_dt: &[f64], lambda: f64) -> Vec<f64> {
    let p = z.len();
    let n = z.first().map_or(0, Vec::len);
    let mut ne = Normal::new(256 + n);
    for j in 0..p {
        for i in 0..n {
            let zz = z[j][i] as usize;
            let w = hat(zz as f64) + 1.0;
            ne.add(&[(zz, 1.0), (256 + i, -1.0)], log_dt[j], w);
        }
    }
    ne.add(&[(128, 1.0)], 0.0, 100.0);
    for k in 1..255 {
        let w = lambda.sqrt() * (hat(k as f64) + 1.0);
        ne.add(&[(k - 1, 1.0), (k, -2.0), (k + 1, 1.0)], 0.0, w);
    }
    let Some(x) = ne.solve() else {
        // Fall back to the sRGB curve.
        return (0..256).map(|k| (srgb_to_linear(k as f32 / 255.0).max(1e-5) as f64).ln() - (srgb_to_linear(128.0 / 255.0) as f64).ln()).collect();
    };
    let mut g: Vec<f64> = x[..256].to_vec();
    // Enforce monotonicity (noise at the extremes can fold the curve).
    for k in (0..128).rev() {
        g[k] = g[k].min(g[k + 1] - 1e-4);
    }
    for k in 129..256 {
        g[k] = g[k].max(g[k - 1] + 1e-4);
    }
    g
}

fn g_at(g: &[f64], v: f32) -> f64 {
    let x = (v.clamp(0.0, 1.0) * 255.0) as f64;
    let i = (x.floor() as usize).min(254);
    let t = x - i as f64;
    g[i] * (1.0 - t) + g[i + 1] * t
}

/// Merge options.
#[derive(Clone, Debug)]
pub struct MergeOptions {
    /// Linear exposure factors (time × gain), one per image.
    pub exposures: Vec<f64>,
    pub remove_ghosts: bool,
    /// Exposure used where ghosts are removed (`None`: the one with most well-exposed pixels).
    pub ghost_base: Option<usize>,
    /// Response curves per channel (`None`: recovered from the images).
    pub response: Option<[Vec<f64>; 3]>,
}

/// Result of [`merge`].
#[derive(Clone, Debug)]
pub struct Merged {
    /// Linear radiance, straight RGBA.
    pub px: Vec<[f32; 4]>,
    pub response: [Vec<f64>; 3],
    pub ghost_base: usize,
    /// Fraction of pixels handled as ghosts.
    pub ghost_fraction: f64,
    /// Dynamic range of the result in stops (1st..99.9th percentile of luminance).
    pub stops: f64,
}

/// Merges aligned exposures (`w × h` each) into a radiance map, scaled so the base exposure's
/// well-exposed pixels keep their linear values.
pub fn merge(w: usize, h: usize, imgs: &[&[[f32; 4]]], opts: &MergeOptions) -> Merged {
    let p = imgs.len();
    let log_dt: Vec<f64> = opts.exposures.iter().map(|e| e.max(1e-12).ln()).collect();
    let well = |im: &[[f32; 4]]| im.iter().step_by(3).filter(|q| (0.1..0.9).contains(&luma(**q))).count();
    let base = opts.ghost_base.filter(|b| *b < p).unwrap_or_else(|| (0..p).max_by_key(|&j| well(imgs[j])).unwrap_or(0));
    let response = opts.response.clone().unwrap_or_else(|| {
        let mut order: Vec<usize> = (0..p).collect();
        order.sort_by(|&a, &b| opts.exposures[a].total_cmp(&opts.exposures[b]));
        let mid = order[p / 2];
        let samples = sample_pixels(w, h, imgs[mid], 160);
        std::array::from_fn(|c| {
            let z: Vec<Vec<u8>> = imgs.iter().map(|im| samples.iter().map(|&i| (im[i][c].clamp(0.0, 1.0) * 255.0).round() as u8).collect()).collect();
            response_curve(&z, &log_dt, 40.0)
        })
    });
    let mut px = vec![[0.0f32; 4]; w * h];
    let mut ghost = vec![0.0f32; w * h];
    let resp = &response;
    par_rows(&mut px, w, 1, |y, row| {
        for (x, out) in row.iter_mut().enumerate() {
            let i = y * w + x;
            let mut alpha = 0.0f32;
            for c in 0..3 {
                let (mut num, mut den) = (0.0f64, 0.0f64);
                for j in 0..p {
                    let v = imgs[j][i][c];
                    let wt = hat(v.clamp(0.0, 1.0) as f64 * 255.0);
                    num += wt * (g_at(&resp[c], v) - log_dt[j]);
                    den += wt;
                }
                let le = if den > 1e-6 {
                    num / den
                } else {
                    // Saturated everywhere: darkest exposure if bright, brightest if dark.
                    let bright = imgs.iter().map(|im| im[i][c]).sum::<f32>() / p as f32 > 0.5;
                    let j = (0..p).min_by(|&a, &b| if bright { log_dt[a].total_cmp(&log_dt[b]) } else { log_dt[b].total_cmp(&log_dt[a]) }).unwrap_or(0);
                    g_at(&resp[c], imgs[j][i][c]) - log_dt[j]
                };
                out[c] = le.exp() as f32;
            }
            for im in imgs {
                alpha = alpha.max(im[i][3]);
            }
            out[3] = alpha;
        }
    });
    let mut ghost_fraction = 0.0;
    if opts.remove_ghosts && p > 1 {
        // Weighted variance of per-exposure log radiance (luminance).
        let score: Vec<f32> = crate::photo_util::par_map(w * h, |i| {
            let (mut s, mut s2, mut sw) = (0.0f64, 0.0f64, 0.0f64);
            for j in 0..p {
                let q = imgs[j][i];
                let l = luma(q);
                let wt = hat(l as f64 * 255.0);
                if wt < 20.0 {
                    continue;
                }
                let lr = (0..3).map(|c| 0.33 * (g_at(&response[c], q[c]) - log_dt[j])).sum::<f64>();
                s += wt * lr;
                s2 += wt * lr * lr;
                sw += wt;
            }
            if sw <= 0.0 {
                return 0.0;
            }
            let m = s / sw;
            ((s2 / sw - m * m).max(0.0)).sqrt() as f32
        });
        for i in 0..w * h {
            ghost[i] = if score[i] > 0.35 { 1.0 } else { 0.0 };
        }
        // Grow and feather the mask.
        let r = (w.max(h) as f32 / 200.0).max(2.0);
        crate::fxutil::gauss_blur_n(&mut ghost, w, h, 1, r);
        for g in ghost.iter_mut() {
            *g = (*g * 3.0).min(1.0);
        }
        ghost_fraction = ghost.iter().filter(|g| **g > 0.5).count() as f64 / (w * h).max(1) as f64;
        let ld = log_dt[base];
        for i in 0..w * h {
            if ghost[i] <= 0.0 {
                continue;
            }
            let q = imgs[base][i];
            for c in 0..3 {
                let e = (g_at(&response[c], q[c]) - ld).exp() as f32;
                px[i][c] = px[i][c] * (1.0 - ghost[i]) + e * ghost[i];
            }
        }
    }
    // Scale per channel so the base exposure's well-exposed pixels map to their linear values.
    for c in 0..3 {
        let mut r: Vec<f64> = (0..w * h)
            .step_by(37)
            .filter_map(|i| {
                let v = imgs[base][i][c];
                ((0.15..0.85).contains(&v) && px[i][c] > 0.0).then(|| srgb_to_linear(v) as f64 / (px[i][c] as f64 * opts.exposures[base]))
            })
            .collect();
        r.sort_by(f64::total_cmp);
        let k = if r.is_empty() { 1.0 } else { r[r.len() / 2] } * opts.exposures[base];
        for q in px.iter_mut() {
            q[c] = (q[c] as f64 * k) as f32;
        }
    }
    let mut l: Vec<f32> = px.iter().step_by(13).filter(|q| q[3] > 0.0).map(|q| luma(*q).max(1e-9)).collect();
    l.sort_by(f32::total_cmp);
    let stops = if l.is_empty() { 0.0 } else { (l[(l.len() as f64 * 0.999) as usize % l.len()] as f64 / l[l.len() / 100] as f64).log2() };
    Merged { px, response, ghost_base: base, ghost_fraction, stops }
}

// ---------- tone mapping ----------

/// The four 16/8-bit conversion methods of Merge to HDR Pro / HDR Toning.
#[derive(Clone, Debug, PartialEq)]
pub enum ToneMethod {
    LocalAdaptation(HdrToning),
    ExposureGamma { exposure: f32, gamma: f32 },
    HighlightCompression,
    EqualizeHistogram,
}

/// Tone maps linear radiance to display-encoded (sRGB curve) values in `0..=1`.
pub fn tone_map(px: &mut [[f32; 4]], w: usize, h: usize, method: &ToneMethod) {
    let n = px.iter().filter(|q| q[3] > 0.0).count().max(1);
    let log_mean = (px.iter().filter(|q| q[3] > 0.0).map(|q| (luma(*q).max(1e-6) as f64).ln()).sum::<f64>() / n as f64).exp() as f32;
    match method {
        ToneMethod::LocalAdaptation(p) => {
            // Key the scene to middle grey, compress in log luminance, then encode.
            let k = 0.18 / log_mean.max(1e-6);
            for q in px.iter_mut() {
                for c in 0..3 {
                    q[c] *= k;
                }
            }
            crate::tone::hdr_toning(px, w, h, p);
            for q in px.iter_mut() {
                for c in 0..3 {
                    q[c] = linear_to_srgb(q[c].clamp(0.0, 1.0));
                }
            }
        }
        ToneMethod::ExposureGamma { exposure, gamma } => {
            let k = 2f32.powf(*exposure);
            let g = gamma.clamp(0.1, 10.0);
            for q in px.iter_mut() {
                for c in 0..3 {
                    q[c] = linear_to_srgb((q[c] * k).clamp(0.0, 1.0)).powf(1.0 / g);
                }
            }
        }
        ToneMethod::HighlightCompression => {
            let k = 0.18 / log_mean.max(1e-6);
            for q in px.iter_mut() {
                let l = luma(*q) * k;
                let ld = l / (1.0 + l);
                let r = if l > 1e-9 { ld / l * k } else { 0.0 };
                for c in 0..3 {
                    q[c] = linear_to_srgb((q[c] * r).clamp(0.0, 1.0));
                }
            }
        }
        ToneMethod::EqualizeHistogram => {
            const BINS: usize = 1024;
            let lv: Vec<f32> = px.iter().map(|q| luma(*q).max(1e-7).ln()).collect();
            let (lo, hi) = lv.iter().zip(px.iter()).filter(|(_, q)| q[3] > 0.0).fold((f32::MAX, f32::MIN), |(a, b), (v, _)| (a.min(*v), b.max(*v)));
            let span = (hi - lo).max(1e-6);
            let mut hist = vec![0f64; BINS];
            for (v, q) in lv.iter().zip(px.iter()) {
                if q[3] > 0.0 {
                    hist[(((v - lo) / span) * (BINS - 1) as f32) as usize] += 1.0;
                }
            }
            let total: f64 = hist.iter().sum::<f64>().max(1.0);
            let mut cdf = vec![0f32; BINS];
            let mut acc = 0.0;
            for b in 0..BINS {
                acc += hist[b];
                cdf[b] = (acc / total) as f32;
            }
            for (v, q) in lv.iter().zip(px.iter_mut()) {
                let target = cdf[(((v - lo) / span).clamp(0.0, 1.0) * (BINS - 1) as f32) as usize];
                // Display value from the CDF (encoded), keep colour ratios in linear light.
                let tl = srgb_to_linear(target);
                let l = luma(*q).max(1e-7);
                for c in 0..3 {
                    q[c] = linear_to_srgb((q[c] / l * tl).clamp(0.0, 1.0));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A radiance map with a wide range, the camera's response a gamma-ish curve.
    fn scene(w: usize, h: usize) -> Vec<[f32; 3]> {
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 / w as f32, (i / w) as f32 / h as f32);
                let base = 0.002 * 2f32.powf(x * 12.0); // 12 stops left to right
                let tex = 1.0 + 0.3 * ((y * 40.0).sin());
                [base * tex, base * tex * 0.8, base * tex * 0.6]
            })
            .collect()
    }

    fn camera(e: f32) -> f32 {
        // A smooth non-sRGB response.
        (e.max(0.0).powf(0.45)).min(1.0)
    }

    fn shoot(sc: &[[f32; 3]], dt: f32) -> Vec<[f32; 4]> {
        sc.iter().map(|r| [camera(r[0] * dt), camera(r[1] * dt), camera(r[2] * dt), 1.0]).map(|p| p.map(|v| (v * 255.0).round() / 255.0)).collect()
    }

    #[test]
    fn exposure_estimation_and_merge_recover_radiance() {
        let (w, h) = (192usize, 64usize);
        let sc = scene(w, h);
        let dts = [0.25f32, 1.0, 4.0, 16.0];
        let shots: Vec<Vec<[f32; 4]>> = dts.iter().map(|d| shoot(&sc, *d)).collect();
        let refs: Vec<&[[f32; 4]]> = shots.iter().map(Vec::as_slice).collect();
        let est = estimate_exposures(&refs);
        // Ratios between consecutive exposures are 4× (estimated through an sRGB guess, so loose).
        for k in 1..4 {
            let r = est[k] / est[k - 1];
            assert!((2.0..8.0).contains(&r), "ratio {r} ({est:?})");
        }
        let m =
            merge(w, h, &refs, &MergeOptions { exposures: dts.iter().map(|d| *d as f64).collect(), remove_ghosts: false, ghost_base: None, response: None });
        // Radiance proportional to the scene: check log-ratios across the 12-stop ramp.
        let probe = |x: usize| m.px[(h / 2) * w + x][1] as f64 / sc[(h / 2) * w + x][1] as f64;
        let k0 = probe(w / 2);
        // Within the bracket's range (brighter pixels clip in every exposure).
        for x in [10, 50, 100, 150, 170] {
            let r = probe(x) / k0;
            assert!((r.ln()).abs() < 0.25, "x {x}: ratio {r}");
        }
        assert!(m.stops > 9.0, "stops {}", m.stops);
        // The recovered response is monotonic.
        assert!(m.response[0].windows(2).all(|p| p[1] > p[0]));
    }

    #[test]
    fn ghosts_come_from_the_base_exposure() {
        let (w, h) = (64usize, 64usize);
        let sc: Vec<[f32; 3]> = vec![[0.2, 0.2, 0.2]; w * h];
        let mut shots: Vec<Vec<[f32; 4]>> = [0.5f32, 1.0, 2.0].iter().map(|d| shoot(&sc, *d)).collect();
        // A "person" only in the third exposure.
        for y in 20..40 {
            for x in 20..40 {
                shots[2][y * w + x] = [0.2, 0.2, 0.2, 1.0];
            }
        }
        let refs: Vec<&[[f32; 4]]> = shots.iter().map(Vec::as_slice).collect();
        let opts = |g| MergeOptions { exposures: vec![0.5, 1.0, 2.0], remove_ghosts: g, ghost_base: Some(1), response: None };
        let plain = merge(w, h, &refs, &opts(false));
        let clean = merge(w, h, &refs, &opts(true));
        let at = |m: &Merged, x: usize, y: usize| m.px[y * w + x][1];
        let bg = at(&clean, 5, 5);
        assert!((at(&clean, 30, 30) / bg - 1.0).abs() < 0.15, "ghost kept: {} vs {}", at(&clean, 30, 30), bg);
        assert!((at(&plain, 30, 30) / at(&plain, 5, 5) - 1.0).abs() > 0.2);
        assert!(clean.ghost_fraction > 0.05 && clean.ghost_base == 1);
    }

    #[test]
    fn mtb_finds_shifts() {
        let (w, h) = (128usize, 96usize);
        let img: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                0.5 + 0.4 * ((x * 0.21).sin() * (y * 0.17).cos())
            })
            .collect();
        let (dx, dy) = (5i32, -3i32);
        let shifted: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as i32 - dx, (i / w) as i32 - dy);
                // Darker exposure of the same scene, shifted.
                img[(y.clamp(0, h as i32 - 1) as usize) * w + x.clamp(0, w as i32 - 1) as usize] * 0.6
            })
            .collect();
        let (ox, oy) = mtb_offset(w, h, &shifted, &img, 4);
        assert_eq!((ox, oy), (dx, dy));
    }

    #[test]
    fn tone_mapping_methods_produce_display_range() {
        let (w, h) = (96usize, 32usize);
        let sc = scene(w, h);
        for method in [
            ToneMethod::LocalAdaptation(HdrToning { radius: 7.0, strength: 0.52, ..Default::default() }),
            ToneMethod::ExposureGamma { exposure: 0.0, gamma: 1.0 },
            ToneMethod::HighlightCompression,
            ToneMethod::EqualizeHistogram,
        ] {
            let mut px: Vec<[f32; 4]> = sc.iter().map(|r| [r[0], r[1], r[2], 1.0]).collect();
            tone_map(&mut px, w, h, &method);
            assert!(px.iter().all(|q| q[..3].iter().all(|v| (0.0..=1.0).contains(v))));
            let row = |x: usize| px[(h / 2) * w + x][1];
            assert!(row(w - 2) > row(2), "{method:?}: not increasing");
            if !matches!(method, ToneMethod::ExposureGamma { .. }) {
                // Compression keeps both ends of a 12-stop ramp visible.
                assert!(row(4) > 0.01 && row(w - 4) < 0.999, "{method:?}: {} {}", row(4), row(w - 4));
            }
        }
    }
}
