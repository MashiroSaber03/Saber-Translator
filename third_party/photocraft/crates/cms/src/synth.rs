//! Synthetic CMYK output profiles.
//!
//! Adobe's CMYK profiles (U.S. Web Coated SWOP, Coated FOGRA39 …) are proprietary, and the
//! freely downloadable characterisation-based profiles come with redistribution conditions, so
//! Photocraft ships a CMYK profile it generates itself from a documented printer model:
//!
//! * **Forward model** (CMYK → XYZ): Yule–Nielsen modified Neugebauer (n = 2) over the 16
//!   ink overprints, with Demichel coverage weights and a parabolic tone value increase (dot
//!   gain) per ink. The eight K-free primaries use the published ISO 12647-2 coated-paper Lab
//!   aim values (paper, C, M, Y, the three two-colour overprints) and an estimated three-colour
//!   black; overprints with black are derived multiplicatively with a first-surface reflection
//!   floor.
//! * **AToB** (all intents): 11⁴ `lut16` table of the forward model, relative colorimetric
//!   (paper → L* 100).
//! * **BToA1** (colorimetric): for every Lab node, black generation (GCR) from the gray
//!   component of the K-free solution, then a bounded least-squares solve for C, M, Y with the
//!   total ink limit; out-of-gamut colours land on the closest reachable colour (clipping).
//! * **BToA0/BToA2** (perceptual/saturation): the target is first compressed into the gamut —
//!   L* is scaled onto the device black, chroma is softly compressed against a gamut boundary
//!   descriptor sampled from the forward model — then solved as above.
//!
//! The model is plausible for coated offset, not a measured characterisation: use a real
//! profile from your print provider for production print.

use crate::Intent;
use crate::clut::Clut;
use crate::curve::Curve;
use crate::math;
use crate::pipeline::Stage;
use crate::profile::{ColorSpace, Lut, LutKind, Pcs, Profile, ProfileClass};

pub const DESCRIPTION: &str = "Photocraft Coated CMYK (synthetic, 300% TAC, medium GCR)";

/// Parameters of the synthetic profile.
#[derive(Clone, Debug)]
pub struct CmykParams {
    pub description: String,
    /// Total area coverage limit (3.0 = 300 %).
    pub tac: f64,
    /// Gray component (min of C, M, Y of the K-free solution) where black generation starts.
    pub gcr_start: f64,
    /// Maximum black.
    pub k_max: f64,
    /// Exponent of the black generation curve above `gcr_start`.
    pub gcr_gamma: f64,
    /// Tone value increase at 50 % for C, M, Y, K.
    pub tvi: [f64; 4],
    /// Grid points of the AToB (4D) and BToA (3D) tables.
    pub grid_a2b: usize,
    pub grid_b2a: usize,
}

impl Default for CmykParams {
    fn default() -> Self {
        CmykParams {
            description: DESCRIPTION.into(),
            tac: 3.0,
            gcr_start: 0.2,
            k_max: 0.95,
            gcr_gamma: 1.1,
            tvi: [0.14, 0.14, 0.14, 0.17],
            grid_a2b: 11,
            grid_b2a: 21,
        }
    }
}

/// The printer model.
#[derive(Clone, Debug)]
pub struct CmykModel {
    /// Absolute XYZ (D50) of the 16 overprints, indexed by ink bit mask (C=1, M=2, Y=4, K=8),
    /// stored as `X^(1/n)` for the Yule–Nielsen sum.
    prim_n: [[f64; 3]; 16],
    /// Absolute XYZ of the paper.
    pub paper: [f64; 3],
    tvi: [f64; 4],
}

const YN: f64 = 2.0;

impl CmykModel {
    pub fn coated(tvi: [f64; 4]) -> Self {
        let lab = |l: f64, a: f64, b: f64| math::lab_to_xyz([l, a, b], math::D50);
        // ISO 12647-2 (coated, paper type 1/2) Lab aims, D50/2°; CMY overprint estimated.
        let w = lab(95.0, 0.0, -2.0);
        let base: [[f64; 3]; 8] = [
            w,                       // paper
            lab(55.0, -37.0, -50.0), // C
            lab(48.0, 74.0, -3.0),   // M
            lab(24.0, 22.0, -46.0),  // C+M (blue)
            lab(89.0, -5.0, 93.0),   // Y
            lab(50.0, -65.0, 27.0),  // C+Y (green)
            lab(47.0, 68.0, 48.0),   // M+Y (red)
            lab(23.0, 0.0, 0.0),     // C+M+Y
        ];
        let k = lab(16.0, 0.0, 0.0);
        // Overprints with black: multiplicative in reflectance above a first-surface floor.
        let floor = 0.006;
        let mut prim = [[0.0; 3]; 16];
        for m in 0..16 {
            let p = base[m & 7];
            prim[m] = if m & 8 == 0 {
                p
            } else {
                std::array::from_fn(|i| {
                    let s = floor * w[i];
                    s + (p[i] - s) * (k[i] - s) / (w[i] - s)
                })
            };
        }
        let prim_n = prim.map(|p| p.map(|v| v.max(0.0).powf(1.0 / YN)));
        CmykModel { prim_n, paper: w, tvi }
    }

    /// Absolute XYZ of an ink combination (each coverage in `[0, 1]`).
    pub fn xyz(&self, cmyk: [f64; 4]) -> [f64; 3] {
        let a: [f64; 4] = std::array::from_fn(|i| {
            let v = cmyk[i].clamp(0.0, 1.0);
            (v + 4.0 * self.tvi[i] * v * (1.0 - v)).clamp(0.0, 1.0)
        });
        let mut s = [0.0; 3];
        for m in 0..16 {
            let mut w = 1.0;
            for (i, ai) in a.iter().enumerate() {
                w *= if m & (1 << i) != 0 { *ai } else { 1.0 - ai };
            }
            if w == 0.0 {
                continue;
            }
            for (acc, p) in s.iter_mut().zip(&self.prim_n[m]) {
                *acc += w * p;
            }
        }
        s.map(|v| v.powf(YN))
    }

    /// ICC-relative Lab (paper = L* 100).
    pub fn lab(&self, cmyk: [f64; 4]) -> [f64; 3] {
        let x = self.xyz(cmyk);
        let rel = [x[0] * math::D50[0] / self.paper[0], x[1] * math::D50[1] / self.paper[1], x[2] * math::D50[2] / self.paper[2]];
        math::xyz_to_lab(rel, math::D50)
    }
}

/// Bounded least-squares solve for C, M, Y given K, minimising ΔE76 to `target`, subject to
/// `C + M + Y ≤ limit` (projected Levenberg–Marquardt with a numerical Jacobian).
fn solve_cmy(model: &CmykModel, target: [f64; 3], k: f64, limit: f64, seeds: &[[f64; 3]]) -> ([f64; 3], f64) {
    let project = |mut x: [f64; 3]| {
        for _ in 0..4 {
            for v in &mut x {
                *v = v.clamp(0.0, 1.0);
            }
            let s: f64 = x.iter().sum();
            if s <= limit + 1e-12 {
                break;
            }
            let active = x.iter().filter(|v| **v > 0.0).count().max(1) as f64;
            let d = (s - limit) / active;
            for v in &mut x {
                if *v > 0.0 {
                    *v -= d;
                }
            }
        }
        x
    };
    let err = |x: [f64; 3]| {
        let l = model.lab([x[0], x[1], x[2], k]);
        [l[0] - target[0], l[1] - target[1], l[2] - target[2]]
    };
    let norm2 = |e: [f64; 3]| e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    let mut best = ([0.0; 3], f64::INFINITY);
    for seed in seeds {
        let mut x = project(*seed);
        let mut e = err(x);
        let mut f = norm2(e);
        let mut lambda = 1e-3;
        for _ in 0..60 {
            // Jacobian (3×3) by forward differences.
            let h = 1e-5;
            let mut j = [[0.0; 3]; 3];
            for c in 0..3 {
                let mut xp = x;
                xp[c] += h;
                let ep = err(xp);
                for r in 0..3 {
                    j[r][c] = (ep[r] - e[r]) / h;
                }
            }
            // (JᵀJ + λ diag) δ = −Jᵀe
            let mut a = [[0.0; 3]; 3];
            let mut g = [0.0; 3];
            for r in 0..3 {
                for c in 0..3 {
                    a[r][c] = (0..3).map(|k| j[k][r] * j[k][c]).sum();
                }
                g[r] = -(0..3).map(|k| j[k][r] * e[k]).sum::<f64>();
            }
            let mut improved = false;
            for _ in 0..8 {
                let mut al = a;
                for (d, row) in al.iter_mut().enumerate() {
                    row[d] += lambda * (1.0 + a[d][d]);
                }
                let Some(inv) = math::invert(&al) else {
                    lambda *= 10.0;
                    continue;
                };
                let step = math::apply(&inv, g);
                let xn = project([x[0] + step[0], x[1] + step[1], x[2] + step[2]]);
                let en = err(xn);
                let fnew = norm2(en);
                if fnew < f {
                    let moved = (0..3).map(|i| (xn[i] - x[i]).abs()).fold(0.0, f64::max);
                    x = xn;
                    e = en;
                    f = fnew;
                    lambda = (lambda * 0.3).max(1e-9);
                    improved = moved > 1e-9;
                    break;
                }
                lambda *= 10.0;
            }
            if !improved || f < 1e-10 {
                break;
            }
        }
        if f < best.1 {
            best = (x, f);
        }
        if best.1 < 1e-6 {
            break;
        }
    }
    (best.0, best.1.sqrt())
}

const SEEDS: [[f64; 3]; 5] = [[0.3, 0.3, 0.3], [0.05, 0.05, 0.05], [0.8, 0.8, 0.8], [0.9, 0.2, 0.2], [0.2, 0.9, 0.9]];

/// Colorimetric separation of a relative Lab target: GCR black generation then a CMY solve.
pub fn separate(model: &CmykModel, p: &CmykParams, target: [f64; 3], warm: Option<[f64; 4]>) -> ([f64; 4], f64) {
    let mut seeds: Vec<[f64; 3]> = warm.map(|w| vec![[w[0], w[1], w[2]]]).unwrap_or_default();
    seeds.extend_from_slice(&SEEDS);
    let (cmy0, _) = solve_cmy(model, target, 0.0, 3.0, &seeds);
    let g = cmy0[0].min(cmy0[1]).min(cmy0[2]);
    let k = if g <= p.gcr_start { 0.0 } else { p.k_max * ((g - p.gcr_start) / (1.0 - p.gcr_start)).powf(p.gcr_gamma) };
    let mut seeds2 = vec![[(cmy0[0] - k).max(0.0), (cmy0[1] - k).max(0.0), (cmy0[2] - k).max(0.0)], cmy0];
    seeds2.extend_from_slice(&seeds);
    let (cmy, de) = solve_cmy(model, target, k, (p.tac - k).max(0.0), &seeds2);
    ([cmy[0], cmy[1], cmy[2], k], de)
}

/// Gamut boundary descriptor: maximum chroma per (L*, hue) bin, sampled from the model.
struct Gbd {
    l_bins: usize,
    h_bins: usize,
    cmax: Vec<f64>,
    black_l: f64,
}

impl Gbd {
    fn new(model: &CmykModel, p: &CmykParams) -> Self {
        let (l_bins, h_bins) = (41usize, 72usize);
        let mut cmax = vec![0.0f64; l_bins * h_bins];
        let n = 21;
        let mut black_l = 100.0f64;
        for ci in 0..n {
            for mi in 0..n {
                for yi in 0..n {
                    for ki in 0..11 {
                        let v = [ci as f64 / (n - 1) as f64, mi as f64 / (n - 1) as f64, yi as f64 / (n - 1) as f64, ki as f64 / 10.0];
                        if v.iter().sum::<f64>() > p.tac + 1e-9 {
                            continue;
                        }
                        let lab = model.lab(v);
                        black_l = black_l.min(lab[0]);
                        let c = lab[1].hypot(lab[2]);
                        let h = lab[2].atan2(lab[1]).rem_euclid(std::f64::consts::TAU);
                        let li = ((lab[0] / 100.0) * (l_bins - 1) as f64).round().clamp(0.0, (l_bins - 1) as f64) as usize;
                        let hi = ((h / std::f64::consts::TAU) * h_bins as f64).round() as usize % h_bins;
                        let e = &mut cmax[li * h_bins + hi];
                        *e = e.max(c);
                    }
                }
            }
        }
        // Fill holes and smooth: max over a 3×3 neighbourhood, then a box blur.
        let get = |v: &Vec<f64>, l: isize, h: isize| {
            let l = l.clamp(0, l_bins as isize - 1) as usize;
            let h = h.rem_euclid(h_bins as isize) as usize;
            v[l * h_bins + h]
        };
        let mut filled = cmax.clone();
        for l in 0..l_bins as isize {
            for h in 0..h_bins as isize {
                let mut m = 0.0f64;
                for dl in -1..=1 {
                    for dh in -1..=1 {
                        m = m.max(get(&cmax, l + dl, h + dh));
                    }
                }
                filled[l as usize * h_bins + h as usize] = m;
            }
        }
        let mut smooth = filled.clone();
        for l in 0..l_bins as isize {
            for h in 0..h_bins as isize {
                let mut s = 0.0;
                for dl in -1..=1 {
                    for dh in -1..=1 {
                        s += get(&filled, l + dl, h + dh);
                    }
                }
                smooth[l as usize * h_bins + h as usize] = s / 9.0;
            }
        }
        Gbd { l_bins, h_bins, cmax: smooth, black_l }
    }

    fn max_chroma(&self, l: f64, h: f64) -> f64 {
        let lp = (l / 100.0).clamp(0.0, 1.0) * (self.l_bins - 1) as f64;
        let hp = (h.rem_euclid(std::f64::consts::TAU) / std::f64::consts::TAU) * self.h_bins as f64;
        let (l0, h0) = (lp.floor() as usize, hp.floor() as usize);
        let (fl, fh) = (lp - l0 as f64, hp - h0 as f64);
        let l1 = (l0 + 1).min(self.l_bins - 1);
        let (h0m, h1m) = (h0 % self.h_bins, (h0 + 1) % self.h_bins);
        let v = |l: usize, h: usize| self.cmax[l * self.h_bins + h];
        let a = v(l0, h0m) + (v(l0, h1m) - v(l0, h0m)) * fh;
        let b = v(l1, h0m) + (v(l1, h1m) - v(l1, h0m)) * fh;
        a + (b - a) * fl
    }

    /// Perceptual gamut compression of a relative Lab colour.
    fn compress(&self, lab: [f64; 3]) -> [f64; 3] {
        let l = self.black_l + lab[0].clamp(0.0, 100.0) * (100.0 - self.black_l) / 100.0;
        let c = lab[1].hypot(lab[2]);
        if c < 1e-9 {
            return [l, 0.0, 0.0];
        }
        let h = lab[2].atan2(lab[1]);
        let cm = self.max_chroma(l, h).max(1e-3);
        let knee = 0.8 * cm;
        let c2 = if c <= knee { c } else { knee + (cm - knee) * ((c - knee) / (cm - knee)).tanh() };
        [l, c2 * h.cos(), c2 * h.sin()]
    }
}

fn lab_node_legacy(i: usize, n: usize) -> f64 {
    i as f64 / (n - 1) as f64
}

/// Decodes a legacy (lut16) Lab node coordinate triple.
fn decode_v2(v: [f64; 3]) -> [f64; 3] {
    [v[0] * 100.0 * 65535.0 / 65280.0, v[1] * 65535.0 / 256.0 - 128.0, v[2] * 65535.0 / 256.0 - 128.0]
}

fn encode_v2(lab: [f64; 3]) -> [f32; 3] {
    [
        (lab[0] / (100.0 * 65535.0 / 65280.0)).clamp(0.0, 1.0) as f32,
        ((lab[1] + 128.0) * 256.0 / 65535.0).clamp(0.0, 1.0) as f32,
        ((lab[2] + 128.0) * 256.0 / 65535.0).clamp(0.0, 1.0) as f32,
    ]
}

#[cfg(not(target_arch = "wasm32"))]
fn par_map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    use rayon::prelude::*;
    (0..n).into_par_iter().map(f).collect()
}

#[cfg(target_arch = "wasm32")]
fn par_map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    (0..n).map(f).collect()
}

fn b2a_table(model: &CmykModel, p: &CmykParams, gbd: Option<&Gbd>) -> Clut {
    let n = p.grid_b2a;
    let nodes = n * n * n;
    let vals = par_map(n * n, |row| {
        // One (L, a) row along b, warm-starting each solve from its neighbour.
        let (li, ai) = (row / n, row % n);
        let mut warm = None;
        let mut out = Vec::with_capacity(n * 4);
        for bi in 0..n {
            let lab = decode_v2([lab_node_legacy(li, n), lab_node_legacy(ai, n), lab_node_legacy(bi, n)]);
            let target = match gbd {
                Some(g) => g.compress(lab),
                None => lab,
            };
            let (cmyk, _) = separate(model, p, target, warm);
            warm = Some(cmyk);
            out.extend(cmyk.map(|v| v as f32));
        }
        out
    });
    let data: Vec<f32> = vals.into_iter().flatten().collect();
    debug_assert_eq!(data.len(), nodes * 4);
    Clut::new(vec![n; 3], 4, data)
}

fn lut16(clut: Clut) -> Lut {
    let (i, o) = (clut.inputs, clut.outputs);
    Lut {
        kind: LutKind::Lut16,
        inputs: i,
        outputs: o,
        stages: vec![Stage::Curves(vec![Curve::Table(vec![0.0, 1.0]); i]), Stage::Clut(clut), Stage::Curves(vec![Curve::Table(vec![0.0, 1.0]); o])],
    }
}

/// Generates a synthetic CMYK profile.
pub fn cmyk_profile(p: &CmykParams) -> Profile {
    let model = CmykModel::coated(p.tvi);
    let g = p.grid_a2b;
    let a2b = Clut::sample(vec![g; 4], 3, |inp, out| {
        let lab = model.lab([inp[0] as f64, inp[1] as f64, inp[2] as f64, inp[3] as f64]);
        out.copy_from_slice(&encode_v2(lab));
    });
    let gbd = Gbd::new(&model, p);
    let colorimetric = b2a_table(&model, p, None);
    let perceptual = b2a_table(&model, p, Some(&gbd));
    let a2b = lut16(a2b);
    let (b0, b1) = (lut16(perceptual), lut16(colorimetric));
    Profile {
        version: (4, 0x30),
        class: ProfileClass::Output,
        color_space: ColorSpace::Cmyk,
        pcs: Pcs::Lab,
        rendering_intent: Intent::Perceptual,
        description: p.description.clone(),
        copyright: crate::builtin::COPYRIGHT.into(),
        white_point: model.paper.map(math::s15f16_round),
        black_point: None,
        chad: None,
        matrix: None,
        trc: None,
        gray_trc: None,
        a2b: [Some(a2b.clone()), Some(a2b.clone()), Some(a2b)],
        b2a: [Some(b0.clone()), Some(b1), Some(b0)],
        bytes: None,
        hash: Default::default(),
    }
    .with_encoded_bytes()
}

/// The default coated profile (what `profiles/photocraft-coated-cmyk.icc` contains).
pub fn coated_cmyk() -> Profile {
    cmyk_profile(&CmykParams::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_anchors() {
        let m = CmykModel::coated(CmykParams::default().tvi);
        let w = m.lab([0.0; 4]);
        assert!((w[0] - 100.0).abs() < 1e-9 && w[1].abs() < 1e-9 && w[2].abs() < 1e-9, "{w:?}");
        let k = m.lab([0.0, 0.0, 0.0, 1.0]);
        assert!(k[0] > 12.0 && k[0] < 22.0, "{k:?}");
        let rich = m.lab([0.75, 0.65, 0.6, 0.9]);
        assert!(rich[0] < k[0], "{rich:?}");
        // Monotonic in each ink.
        for ink in 0..4 {
            let mut prev = 101.0;
            for i in 0..=20 {
                let mut v = [0.2, 0.2, 0.2, 0.2];
                v[ink] = i as f64 / 20.0;
                let l = m.lab(v)[0];
                assert!(l < prev + 1e-9, "ink {ink} step {i}: {l} >= {prev}");
                prev = l;
            }
        }
    }

    #[test]
    fn separation_hits_in_gamut_targets() {
        let m = CmykModel::coated(CmykParams::default().tvi);
        let p = CmykParams::default();
        let reachable = [[0.3, 0.4, 0.1, 0.1], [0.6, 0.1, 0.5, 0.0], [0.05, 0.1, 0.7, 0.0], [0.5, 0.45, 0.45, 0.4]].map(|v| m.lab(v));
        for target in [[50.0, 0.0, 0.0]].into_iter().chain(reachable) {
            let (cmyk, de) = separate(&m, &p, target, None);
            assert!(de < 1.5, "{target:?} -> {cmyk:?} ΔE {de}");
            assert!(cmyk.iter().sum::<f64>() <= p.tac + 1e-6);
        }
    }
}
