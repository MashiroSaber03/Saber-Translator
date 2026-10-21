//! Direct (intensity-based) registration for Edit › Auto-Align Layers, for layers that nearly
//! coincide: a focus bracket, an exposure bracket, a burst. Feature matching
//! ([`crate::panorama`]) wants distinctive corners seen from two places; such layers are the same
//! view with a sub-pixel shift, a little scale (focus breathing) and rotation, and much of each
//! layer defocused, so there is often nothing to match and nothing to verify. Here the whole
//! overlap is the evidence: the pose that makes the warped target look like the reference.
//!
//! The target is registered to the reference coarse-to-fine on a Gaussian pyramid of the
//! luminance. The cost of a pose is the RMS of the difference between the reference and the
//! target warped by it, the mean difference removed (so exposure and tone changes don't count),
//! over the pixels both cover. Each level runs a bounded Nelder-Mead search whose simplex starts
//! one pixel of that level wide and stops at a tenth of one, so the coarse levels only hand the
//! next one a start within its pixel and the finest level searched settles sub-pixel. The pose
//! is resolution independent (shifts as fractions of the size, a scale, an angle), so a fit that
//! stops a few levels short of full resolution still applies at full resolution, and the cost
//! at full resolution, the dominant expense, is never paid.
//!
//! [`Pose`] is the transform searched: a similarity (shift, scale, rotation) about the layer's
//! centre, with aspect and shear for an affine fit and two perspective terms for a projective
//! one; [`Motion`] says which of them move. Everything is deterministic.

use crate::panorama::Motion;
use crate::photo_util::{par_map, par_rows2};
use crate::transform::Homography;

/// A layer's transform onto the reference, about the layer's centre: a similarity (shift, scale,
/// rotation); with `aspect` and `shear` an affine transform; with `px` and `py` a projective one.
/// The reference sits at [`Pose::IDENTITY`]; [`Pose::matrix`] gives it as a 3×3 homography.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    /// Shift, as a fraction of the width.
    pub dx: f64,
    /// Shift, as a fraction of the height.
    pub dy: f64,
    pub scale: f64,
    /// Rotation in radians.
    pub rot: f64,
    /// Vertical over horizontal scale (1 = isotropic).
    pub aspect: f64,
    /// Horizontal shear per unit of height (0 = none).
    pub shear: f64,
    /// Perspective: the divisor `1 + px·u/w + py·v/h` over the centred pixel `(u, v)`.
    pub px: f64,
    pub py: f64,
}

impl Pose {
    /// The number of parameters (the length of [`Pose::to_vec`]).
    pub const N: usize = 8;
    pub const IDENTITY: Pose = Pose { dx: 0.0, dy: 0.0, scale: 1.0, rot: 0.0, aspect: 1.0, shear: 0.0, px: 0.0, py: 0.0 };
    /// The search box's half-width around the start, per parameter: 10 % of the layer in shift,
    /// 10 % in scale, 5° in rotation, 5 % in aspect and shear, 5 % in each perspective term.
    const SPAN: [f64; Pose::N] = [0.10, 0.10, 0.10, 0.087_266_462_599_716_47, 0.05, 0.05, 0.05, 0.05];

    pub fn to_vec(self) -> [f64; Pose::N] {
        [self.dx, self.dy, self.scale, self.rot, self.aspect, self.shear, self.px, self.py]
    }

    pub fn from_vec(v: &[f64; Pose::N]) -> Pose {
        Pose { dx: v[0], dy: v[1], scale: v[2], rot: v[3], aspect: v[4], shear: v[5], px: v[6], py: v[7] }
    }

    /// Which parameters a search over `motion` may move: shifts, then scale and rotation, then
    /// aspect and shear, then the perspective terms.
    pub fn free(motion: Motion) -> [bool; Pose::N] {
        match motion {
            Motion::Translation => [true, true, false, false, false, false, false, false],
            Motion::Euclidean => [true, true, false, true, false, false, false, false],
            Motion::Similarity => [true, true, true, true, false, false, false, false],
            Motion::Homography => [true; Pose::N],
        }
    }

    /// The homography mapping the layer's pixels onto the reference's: the linear part is
    /// `scale · R(rot) · [[1, shear], [0, aspect]]` about the centre, the perspective terms act on
    /// the centred pixel; without them the last row is exactly `[0, 0, 1]`.
    pub fn matrix(&self, w: usize, h: usize) -> Homography {
        let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
        let (c, s) = (self.rot.cos(), self.rot.sin());
        let sc = self.scale;
        let (a, b, d, e) = (sc * c, sc * (c * self.shear - s * self.aspect), sc * s, sc * (s * self.shear + c * self.aspect));
        let (gx, gy) = (self.px / w as f64, self.py / h as f64);
        // T(c) · [[a, b, t], [d, e, t'], [gx, gy, 1]] · T(−c), written out.
        let (a2, b2, d2, e2) = (a + cx * gx, b + cx * gy, d + cy * gx, e + cy * gy);
        let tx = cx + self.dx * w as f64 - (a2 * cx + b2 * cy);
        let ty = cy + self.dy * h as f64 - (d2 * cx + e2 * cy);
        Homography([a2, b2, tx, d2, e2, ty, gx, gy, 1.0 - gx * cx - gy * cy])
    }
}

/// A luminance plane with, where the layer does not cover its whole area, its coverage.
pub struct Plane<'a> {
    pub w: usize,
    pub h: usize,
    pub luma: &'a [f32],
    pub valid: Option<&'a [bool]>,
}

/// The outcome of [`register`].
#[derive(Clone, Debug, PartialEq)]
pub struct Registration {
    pub pose: Pose,
    /// The pose as a homography of the target's pixels onto the reference's.
    pub h: Homography,
    /// The cost at the pose: the RMS of the mean-removed difference, at the finest level searched.
    pub rms: f64,
    /// Pixels of the finest level searched that both layers cover at the pose.
    pub overlap: usize,
    /// The correlation of the reference with the warped target over those pixels (Pearson, −1
    /// to 1): how much of the reference the pose explains, whatever the exposure. Near 1 for a
    /// bracket of one scene, near 0 for unrelated pictures; 0 without overlap or texture.
    pub correlation: f64,
}

/// One pyramid level: the plane, its coverage and its size.
struct Lvl {
    luma: Vec<f32>,
    valid: Option<Vec<bool>>,
    w: usize,
    h: usize,
}

const K: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

/// Gaussian REDUCE of a plane and its coverage (5-tap binomial, clamped borders, phase-0
/// subsample); a reduced pixel is covered when more than half its support is.
fn reduce(l: &Lvl) -> Lvl {
    let (w, h) = (l.w.div_ceil(2), l.h.div_ceil(2));
    let cover: Vec<f32> = l.valid.as_ref().map(|v| v.iter().map(|b| *b as u8 as f32).collect()).unwrap_or_default();
    let pass = |src: &[f32]| -> Vec<f32> {
        let mut tmp = vec![0.0f32; w * l.h];
        for (y, row) in tmp.chunks_mut(w).enumerate() {
            let s = &src[y * l.w..(y + 1) * l.w];
            for (x, o) in row.iter_mut().enumerate() {
                *o = K.iter().enumerate().map(|(k, wt)| wt * s[(2 * x + k).saturating_sub(2).min(l.w - 1)]).sum();
            }
        }
        let mut out = vec![0.0f32; w * h];
        for (y, row) in out.chunks_mut(w).enumerate() {
            for (k, wt) in K.iter().enumerate() {
                let sy = (2 * y + k).saturating_sub(2).min(l.h - 1);
                for (o, v) in row.iter_mut().zip(&tmp[sy * w..(sy + 1) * w]) {
                    *o += wt * v;
                }
            }
        }
        out
    };
    let luma = pass(&l.luma);
    let valid = (!cover.is_empty()).then(|| pass(&cover).iter().map(|c| *c > 0.5).collect());
    Lvl { luma, valid, w, h }
}

/// The search pyramid of a plane: level 0 is the plane, each next level half its size, while
/// the height exceeds 64 and the width 8.
fn levels(p: &Plane) -> Vec<Lvl> {
    let mut out = vec![Lvl { luma: p.luma.to_vec(), valid: p.valid.map(<[bool]>::to_vec), w: p.w, h: p.h }];
    while let Some(last) = out.last() {
        if last.h <= 64 || last.w <= 8 {
            break;
        }
        let next = reduce(last);
        out.push(next);
    }
    out
}

/// The cubic B-spline (4 taps) at fraction `t`.
#[inline]
fn spline4(t: f64) -> [f64; 4] {
    [
        ((-1.0 / 3.0 * t + 0.8) * t - 0.466_666_67) * t,
        ((t - 1.8) * t - 0.2) * t + 1.0,
        ((1.2 - t) * t + 0.8) * t,
        ((1.0 / 3.0 * t - 0.2) * t - 0.133_333_34) * t,
    ]
}

/// The source points along output row `y` under `inv`: `(x0, y0, dx, dy)` such that pixel `x`
/// reads `(x0 + dx·x, y0 + dy·x)`; `None` for a projective `inv`, handled pixel by pixel.
#[inline]
fn affine_row(inv: &Homography, y: f64) -> Option<(f64, f64, f64, f64)> {
    let m = &inv.0;
    (m[6] == 0.0 && m[7] == 0.0 && m[8] == 1.0).then(|| (m[1] * y + m[2], m[4] * y + m[5], m[0], m[3]))
}

/// `src` sampled at `(sx, sy)` with the cubic B-spline, edge-clamped.
#[inline]
fn sample(src: &[f32], w: usize, h: usize, sx: f64, sy: f64) -> f32 {
    let (x0, y0) = (sx.floor(), sy.floor());
    let wx = spline4(sx - x0);
    let wy = spline4(sy - y0);
    let (x0, y0) = (x0 as i64, y0 as i64);
    let mut acc = 0.0f64;
    for (j, wyj) in wy.iter().enumerate() {
        let yy = (y0 + j as i64 - 1).clamp(0, h as i64 - 1) as usize;
        let mut r = 0.0f64;
        for (i, wxi) in wx.iter().enumerate() {
            let xx = (x0 + i as i64 - 1).clamp(0, w as i64 - 1) as usize;
            r += wxi * f64::from(src[yy * w + xx]);
        }
        acc += wyj * r;
    }
    acc as f32
}

/// Whether the target covers the source point (nearest pixel of its coverage).
#[inline]
fn covered(valid: Option<&[bool]>, w: usize, sx: f64, sy: f64) -> bool {
    valid.is_none_or(|v| v[(sy + 0.5) as usize * w + (sx + 0.5) as usize])
}

/// The cost of `pose`: the RMS of `reference − target ∘ pose`, the mean removed, over the pixels
/// of the reference that both cover; 1e9 when fewer than 16 do. One row-parallel pass folds Σr,
/// Σt, Σr², Σt², Σrt and the count: rms = √((Σd² − (Σd)²/n) / n) with d = r − t, and the
/// correlation of r and t over the same pixels, also returned (0 when either has no spread).
fn cost(reference: &Lvl, target: &Lvl, pose: &Pose) -> (f64, usize, f64) {
    let Some(inv) = pose.matrix(target.w, target.h).inverse() else { return (1e9, 0, 0.0) };
    let (xmax, ymax) = ((target.w - 1) as f64, (target.h - 1) as f64);
    let rows = par_map(reference.h, |y| {
        let (mut sr, mut st, mut sr2, mut st2, mut srt, mut cnt) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
        let yf = y as f64;
        let aff = affine_row(&inv, yf);
        let row = &reference.luma[y * reference.w..(y + 1) * reference.w];
        for (x, &r) in row.iter().enumerate() {
            if reference.valid.as_ref().is_some_and(|v| !v[y * reference.w + x]) {
                continue;
            }
            let (sx, sy) = match aff {
                Some((x0, y0, dx, dy)) => (x0 + dx * x as f64, y0 + dy * x as f64),
                None => inv.apply(x as f64, yf),
            };
            let inside = sx >= 0.0 && sx <= xmax && sy >= 0.0 && sy <= ymax;
            if !inside || !covered(target.valid.as_deref(), target.w, sx, sy) {
                continue;
            }
            let (r, t) = (f64::from(r), f64::from(sample(&target.luma, target.w, target.h, sx, sy)));
            sr += r;
            st += t;
            sr2 += r * r;
            st2 += t * t;
            srt += r * t;
            cnt += 1;
        }
        (sr, st, sr2, st2, srt, cnt)
    });
    let (sr, st, sr2, st2, srt, cnt) =
        rows.iter().fold((0.0, 0.0, 0.0, 0.0, 0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3, a.4 + b.4, a.5 + b.5));
    if cnt < 16 {
        return (1e9, cnt, 0.0);
    }
    let n = cnt as f64;
    // Central moments: Σ(r−r̄)², Σ(t−t̄)², Σ(r−r̄)(t−t̄).
    let (vr, vt, cov) = ((sr2 - sr * sr / n).max(0.0), (st2 - st * st / n).max(0.0), srt - sr * st / n);
    let rms = ((vr + vt - 2.0 * cov) / n).max(0.0).sqrt();
    let correlation = if vr > 0.0 && vt > 0.0 { (cov / (vr * vt).sqrt()).clamp(-1.0, 1.0) } else { 0.0 };
    (rms, cnt, correlation)
}

/// Full-resolution pixels a point at the layer's edge moves per unit of each parameter (the
/// worst case over the layer): the width and height for the shifts, the half-layer for scale
/// and rotation, the half-height for aspect and shear, a quarter-layer for the perspective terms.
fn param_gain(w: usize, h: usize) -> [f64; Pose::N] {
    let (w, h) = (w as f64, h as f64);
    let r = w.max(h) / 2.0;
    [w, h, r, r, h / 2.0, h / 2.0, w / 4.0, h / 4.0]
}

/// The simplex's first step at a level: one pixel of that level.
const STEP_PX: f64 = 1.0;
/// The simplex's stopping size at a level: a tenth of a pixel of that level.
const TOL_PX: f64 = 0.1;

/// The search's schedule at pyramid level `lvl` (a pixel there is `2^lvl` full-resolution
/// pixels) over the parameters `free`: the first step and the stopping size of the simplex, per
/// parameter, in the parameter's units.
fn level_steps(free: &[usize], lvl: usize, w: usize, h: usize) -> (Vec<f64>, Vec<f64>) {
    let gain = param_gain(w, h);
    let px = (1u64 << lvl.min(60)) as f64;
    (free.iter().map(|&k| STEP_PX * px / gain[k]).collect(), free.iter().map(|&k| TOL_PX * px / gain[k]).collect())
}

/// Nelder-Mead's stopping rule on a simplex sorted by `fv`: the values agree to 1e-4 relative
/// and every vertex lies within `tol[k]` of the best along each axis.
fn converged(simplex: &[Vec<f64>], fv: &[f64], tol: &[f64]) -> bool {
    let n = tol.len();
    let (Some(first), Some(last), Some(best)) = (fv.first(), fv.get(n), simplex.first()) else { return true };
    if (last - first).abs() > 1e-4 * (1.0 + first.abs()) {
        return false;
    }
    simplex[1..].iter().all(|v| (0..n).all(|k| (v[k] - best[k]).abs() <= tol[k]))
}

/// Bounded Nelder-Mead: minimises `f` over the box `[lo, hi]` from `x0`, the first simplex `x0`
/// moved by `step[k]` along each axis, until the vertices agree to 1e-4 relative in `f` and lie
/// within `tol[k]` of the best along every axis, or 200 iterations.
fn nelder_mead<F: Fn(&[f64]) -> f64>(f: &F, x0: &[f64], lo: &[f64], hi: &[f64], step: &[f64], tol: &[f64]) -> Vec<f64> {
    let n = x0.len();
    if n == 0 {
        return Vec::new();
    }
    let clamp = |v: &mut [f64]| {
        for k in 0..n {
            v[k] = v[k].clamp(lo[k], hi[k]);
        }
    };
    let mut simplex: Vec<Vec<f64>> = vec![x0.to_vec()];
    for k in 0..n {
        let mut v = x0.to_vec();
        v[k] += step[k];
        clamp(&mut v);
        simplex.push(v);
    }
    let mut fv: Vec<f64> = simplex.iter().map(|v| f(v)).collect();
    let (alpha, gamma, rho, sigma) = (1.0, 2.0, 0.5, 0.5);
    for _ in 0..200 {
        let mut idx: Vec<usize> = (0..=n).collect();
        idx.sort_by(|&a, &b| fv[a].total_cmp(&fv[b]));
        simplex = idx.iter().map(|&i| simplex[i].clone()).collect();
        fv = idx.iter().map(|&i| fv[i]).collect();
        if converged(&simplex, &fv, tol) {
            break;
        }
        let mut c = vec![0.0; n];
        for v in &simplex[..n] {
            for k in 0..n {
                c[k] += v[k] / n as f64;
            }
        }
        let mut xr: Vec<f64> = (0..n).map(|k| c[k] + alpha * (c[k] - simplex[n][k])).collect();
        clamp(&mut xr);
        let fr = f(&xr);
        if fr < fv[0] {
            let mut xe: Vec<f64> = (0..n).map(|k| c[k] + gamma * (xr[k] - c[k])).collect();
            clamp(&mut xe);
            let fe = f(&xe);
            if fe < fr {
                (simplex[n], fv[n]) = (xe, fe);
            } else {
                (simplex[n], fv[n]) = (xr, fr);
            }
        } else if fr < fv[n - 1] {
            (simplex[n], fv[n]) = (xr, fr);
        } else {
            let mut xc: Vec<f64> = (0..n).map(|k| c[k] + rho * (simplex[n][k] - c[k])).collect();
            clamp(&mut xc);
            let fc = f(&xc);
            if fc < fv[n] {
                (simplex[n], fv[n]) = (xc, fc);
            } else {
                let best = simplex[0].clone();
                for (v, fvi) in simplex[1..].iter_mut().zip(&mut fv[1..]) {
                    for k in 0..n {
                        v[k] = best[k] + sigma * (v[k] - best[k]);
                    }
                    clamp(v);
                    *fvi = f(v);
                }
            }
        }
    }
    let best = (0..=n).min_by(|&a, &b| fv[a].total_cmp(&fv[b])).unwrap_or(0);
    simplex[best].clone()
}

/// Registers `target` to `reference` (planes of the same size) with the parameters `motion`
/// moves, from `init` (the previous layer's pose, or the identity), stopping `coarsen` levels
/// short of full resolution (at least one level is searched).
pub fn register(reference: &Plane, target: &Plane, motion: Motion, init: Pose, coarsen: usize) -> Registration {
    let (w, h) = (target.w, target.h);
    let pref = levels(reference);
    let ptgt = levels(target);
    let n = pref.len().min(ptgt.len());
    let free: Vec<usize> = (0..Pose::N).filter(|&k| Pose::free(motion)[k]).collect();
    let iv = init.to_vec();
    let mut cur = iv;
    let lo: Vec<f64> = free.iter().map(|&k| iv[k] - Pose::SPAN[k]).collect();
    let hi: Vec<f64> = free.iter().map(|&k| iv[k] + Pose::SPAN[k]).collect();
    let finest = coarsen.min(n.saturating_sub(1));
    let (mut rms, mut overlap, mut correlation) = (1e9, 0, 0.0);
    for lvl in (finest..n).rev() {
        let (rl, tl) = (&pref[lvl], &ptgt[lvl]);
        let snap = cur;
        let f = |xf: &[f64]| -> f64 {
            let mut v = snap;
            for (k, &idx) in free.iter().enumerate() {
                v[idx] = xf[k];
            }
            cost(rl, tl, &Pose::from_vec(&v)).0
        };
        let x0: Vec<f64> = free.iter().map(|&k| cur[k]).collect();
        let (step, tol) = level_steps(&free, lvl, w, h);
        let best = nelder_mead(&f, &x0, &lo, &hi, &step, &tol);
        for (k, &idx) in free.iter().enumerate() {
            cur[idx] = best[k];
        }
        (rms, overlap, correlation) = cost(rl, tl, &Pose::from_vec(&cur));
    }
    let pose = Pose::from_vec(&cur);
    Registration { pose, h: pose.matrix(w, h), rms, overlap, correlation }
}

/// `src` (with its coverage) warped by `pose` into a plane of the same size with the cubic
/// B-spline, and the coverage of the result: the points that land inside the source, on pixels
/// it covers.
pub fn warp(src: &Plane, pose: &Pose) -> (Vec<f32>, Vec<bool>) {
    let (w, h) = (src.w, src.h);
    let mut out = vec![0.0f32; w * h];
    let mut valid = vec![false; w * h];
    let Some(inv) = pose.matrix(w, h).inverse() else { return (out, valid) };
    let (xmax, ymax) = ((w - 1) as f64, (h - 1) as f64);
    par_rows2(&mut out, &mut valid, w, |y, orow, vrow| {
        let yf = y as f64;
        let aff = affine_row(&inv, yf);
        for (x, (o, v)) in orow.iter_mut().zip(vrow.iter_mut()).enumerate() {
            let (sx, sy) = match aff {
                Some((x0, y0, dx, dy)) => (x0 + dx * x as f64, y0 + dy * x as f64),
                None => inv.apply(x as f64, yf),
            };
            let inside = sx >= 0.0 && sx <= xmax && sy >= 0.0 && sy <= ymax;
            *v = inside && covered(src.valid, w, sx, sy);
            if inside {
                *o = sample(src.luma, w, h, sx, sy);
            }
        }
    });
    (out, valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth random texture: a sum of a few sinusoids, so sub-pixel resampling is faithful.
    fn texture(w: usize, h: usize, seed: u32) -> Vec<f32> {
        let s = seed as f32;
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                let v =
                    (x * 0.21 + s).sin() * (y * 0.17 - s).cos() + (x * 0.071 + y * 0.093 + s * 0.5).sin() * 0.7 + ((x * 0.33 - y * 0.29) * 1.3 + s).cos() * 0.4;
                0.5 + 0.2 * v
            })
            .collect()
    }

    fn plane(w: usize, h: usize, luma: &[f32]) -> Plane<'_> {
        Plane { w, h, luma, valid: None }
    }

    #[test]
    fn pose_matrix_is_the_identity_at_the_identity_and_shifts_in_pixels() {
        assert_eq!(Pose::IDENTITY.matrix(640, 480), Homography::IDENTITY);
        let p = Pose { dx: 0.1, dy: -0.05, ..Pose::IDENTITY };
        let (x, y) = p.matrix(200, 100).apply(50.0, 20.0);
        assert!((x - 70.0).abs() < 1e-9 && (y - 15.0).abs() < 1e-9);
        // Scale and rotation are about the centre.
        let p = Pose { scale: 1.5, rot: 0.3, ..Pose::IDENTITY };
        let (x, y) = p.matrix(200, 100).apply(100.0, 50.0);
        assert!((x - 100.0).abs() < 1e-9 && (y - 50.0).abs() < 1e-9);
        assert_eq!(Pose::from_vec(&p.to_vec()), p);
        assert_eq!(Pose::free(Motion::Translation).iter().filter(|f| **f).count(), 2);
        assert_eq!(Pose::free(Motion::Homography), [true; Pose::N]);
    }

    #[test]
    fn warp_by_the_identity_is_exact_and_marks_coverage() {
        let (w, h) = (40, 30);
        let src = texture(w, h, 1);
        let (o, v) = warp(&plane(w, h, &src), &Pose::IDENTITY);
        assert!(o.iter().zip(&src).all(|(a, b)| (a - b).abs() < 1e-6));
        assert!(v.iter().all(|b| *b));
        // A shift of 10 px to the right leaves the left 10 columns uncovered.
        let (_, v) = warp(&plane(w, h, &src), &Pose { dx: 10.0 / w as f64, ..Pose::IDENTITY });
        assert!((0..h).all(|y| !v[y * w] && !v[y * w + 9] && v[y * w + 10]));
        // The source's own coverage carries over.
        let valid: Vec<bool> = (0..w * h).map(|i| i % w < 20).collect();
        let (_, v) = warp(&Plane { w, h, luma: &src, valid: Some(&valid) }, &Pose::IDENTITY);
        assert!(v[5 * w + 10] && !v[5 * w + 30]);
    }

    #[test]
    fn cost_is_zero_at_the_truth_and_the_sentinel_without_overlap() {
        let (w, h) = (96, 72);
        let src = texture(w, h, 2);
        let l = levels(&plane(w, h, &src));
        assert!(cost(&l[0], &l[0], &Pose::IDENTITY).0 < 1e-9);
        assert_eq!(cost(&l[0], &l[0], &Pose { dx: 2.0, ..Pose::IDENTITY }).0, 1e9);
        // The mean is removed: a brighter copy costs nothing.
        let bright: Vec<f32> = src.iter().map(|v| v + 0.2).collect();
        let lb = levels(&plane(w, h, &bright));
        assert!(cost(&l[0], &lb[0], &Pose::IDENTITY).0 < 1e-6);
        assert!(cost(&l[0], &lb[0], &Pose { dx: 0.02, ..Pose::IDENTITY }).0 > 1e-3);
    }

    /// The correlation tells a fit from a non-fit whatever the exposure: a stop brighter copy of
    /// the texture correlates fully at the truth (where the mean-removed RMS does not vanish),
    /// another texture hardly at all, and a flat plane not at all.
    #[test]
    fn correlation_is_exposure_independent_and_low_for_unrelated_pictures() {
        let (w, h) = (96, 72);
        let src = texture(w, h, 2);
        let l = levels(&plane(w, h, &src));
        let doubled: Vec<f32> = src.iter().map(|v| v * 2.0).collect();
        let ld = levels(&plane(w, h, &doubled));
        let (rms, _, corr) = cost(&l[0], &ld[0], &Pose::IDENTITY);
        assert!(rms > 0.05 && corr > 0.9999, "rms {rms} correlation {corr}");
        let other = levels(&plane(w, h, &texture(w, h, 7)));
        let (_, _, corr) = cost(&l[0], &other[0], &Pose::IDENTITY);
        assert!(corr.abs() < 0.3, "unrelated textures correlate at {corr}");
        let flat = levels(&plane(w, h, &vec![0.5; w * h]));
        assert_eq!(cost(&l[0], &flat[0], &Pose::IDENTITY).2, 0.0);
        let r = register(&plane(w, h, &src), &plane(w, h, &doubled), Motion::Similarity, Pose::IDENTITY, 0);
        assert!(r.correlation > 0.999, "{r:?}");
    }

    #[test]
    fn pyramid_stops_above_64_rows() {
        let l = levels(&plane(40, 300, &vec![0.25; 40 * 300]));
        assert_eq!(l.iter().map(|l| (l.w, l.h)).collect::<Vec<_>>(), vec![(40, 300), (20, 150), (10, 75), (5, 38)]);
        assert!(l.iter().all(|l| l.luma.iter().all(|v| (v - 0.25).abs() < 1e-6)));
        let valid: Vec<bool> = (0..40 * 300).map(|i| i % 40 < 20).collect();
        let l = levels(&Plane { w: 40, h: 300, luma: &vec![0.25; 40 * 300], valid: Some(&valid) });
        let v = l[2].valid.as_ref().unwrap();
        assert!(v[10 * 10 + 2] && !v[10 * 10 + 8]);
    }

    #[test]
    fn nelder_mead_finds_the_minimum_of_a_bowl() {
        let f = |x: &[f64]| (x[0] - 0.3).powi(2) + 2.0 * (x[1] + 0.2).powi(2);
        let r = nelder_mead(&f, &[0.0, 0.0], &[-1.0, -1.0], &[1.0, 1.0], &[0.1, 0.1], &[1e-6, 1e-6]);
        assert!((r[0] - 0.3).abs() < 1e-4 && (r[1] + 0.2).abs() < 1e-4, "{r:?}");
        // The box binds.
        let r = nelder_mead(&f, &[0.0, 0.0], &[-1.0, -1.0], &[0.1, 1.0], &[0.1, 0.1], &[1e-6, 1e-6]);
        assert!((r[0] - 0.1).abs() < 1e-6, "{r:?}");
        assert!(nelder_mead(&f, &[], &[], &[], &[], &[]).is_empty());
        let (step, tol) = level_steps(&[0, 2], 3, 8000, 6000);
        assert!((step[0] * 8000.0 - 8.0).abs() < 1e-9 && (tol[1] * 4000.0 - 0.8).abs() < 1e-9);
    }

    /// A layer shifted, scaled and rotated a little registers back to within a tenth of a pixel
    /// at every motion model that contains the truth, with or without coarsening.
    #[test]
    fn registers_a_similarity_to_a_tenth_of_a_pixel() {
        let (w, h) = (320, 240);
        let reference = texture(w, h, 3);
        let truth = Pose { dx: 3.3 / w as f64, dy: -2.1 / h as f64, scale: 1.012, rot: 0.6f64.to_radians(), ..Pose::IDENTITY };
        // The pose maps the target's pixels onto the reference's: target(u) = reference(M u).
        let m = truth.matrix(w, h);
        let target: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = m.apply((i % w) as f64, (i / w) as f64);
                sample(&reference, w, h, x.clamp(0.0, (w - 1) as f64), y.clamp(0.0, (h - 1) as f64))
            })
            .collect();
        for (motion, coarsen) in [(Motion::Similarity, 0), (Motion::Similarity, 1), (Motion::Homography, 0)] {
            let r = register(&plane(w, h, &reference), &plane(w, h, &target), motion, Pose::IDENTITY, coarsen);
            let (tx, ty) = truth.matrix(w, h).apply(0.0, 0.0);
            let (rx, ry) = r.h.apply(0.0, 0.0);
            let corner = ((tx - rx).hypot(ty - ry)).max({
                let (tx, ty) = truth.matrix(w, h).apply(w as f64, h as f64);
                let (rx, ry) = r.h.apply(w as f64, h as f64);
                (tx - rx).hypot(ty - ry)
            });
            assert!(corner < 0.1 * (1 << coarsen) as f64 + 0.05, "{motion:?} coarsen {coarsen}: corners off by {corner} px, {:?}", r.pose);
            let searched = (w >> coarsen) * (h >> coarsen);
            assert!(r.rms < 0.01 && r.overlap > searched / 2, "{motion:?}: rms {} overlap {} of {searched}", r.rms, r.overlap);
        }
        // Translation only: the shift is found, the rest stays put.
        let t = Pose { dx: 4.0 / w as f64, dy: 3.0 / h as f64, ..Pose::IDENTITY };
        let target: Vec<f32> = (0..w * h)
            .map(|i| sample(&reference, w, h, ((i % w) as f64 + 4.0).clamp(0.0, (w - 1) as f64), ((i / w) as f64 + 3.0).clamp(0.0, (h - 1) as f64)))
            .collect();
        let r = register(&plane(w, h, &reference), &plane(w, h, &target), Motion::Translation, Pose::IDENTITY, 0);
        assert!(((r.pose.dx - t.dx) * w as f64).abs() < 0.1 && ((r.pose.dy - t.dy) * h as f64).abs() < 0.1, "{:?}", r.pose);
        assert_eq!((r.pose.scale, r.pose.rot), (1.0, 0.0));
    }

    /// Only the covered part counts: a target whose right half is garbage still registers on
    /// its left half when its coverage says so.
    #[test]
    fn coverage_keeps_uncovered_pixels_out_of_the_fit() {
        let (w, h) = (320, 240);
        let reference = texture(w, h, 4);
        let mut target: Vec<f32> = (0..w * h).map(|i| sample(&reference, w, h, ((i % w) as f64 + 2.5).clamp(0.0, (w - 1) as f64), (i / w) as f64)).collect();
        let valid: Vec<bool> = (0..w * h).map(|i| i % w < w / 2).collect();
        for (i, v) in target.iter_mut().enumerate() {
            if !valid[i] {
                *v = ((i * 7919) % 1000) as f32 / 1000.0;
            }
        }
        let r = register(&plane(w, h, &reference), &Plane { w, h, luma: &target, valid: Some(&valid) }, Motion::Translation, Pose::IDENTITY, 0);
        assert!((r.pose.dx * w as f64 - 2.5).abs() < 0.1, "{:?}", r.pose);
        assert!(r.overlap < w * h / 2 + w && r.overlap > w * h / 3);
    }
}
