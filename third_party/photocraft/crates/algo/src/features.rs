//! Image registration for Edit › Auto-Align Layers: corners, binary descriptors, matching and
//! robust model fitting.
//!
//! * Corners: C. Harris, M. Stephens, *A Combined Corner and Edge Detector*, Alvey Vision
//!   Conference 1988 (structure tensor `det − k·trace²`, Gaussian window), with non-maximum
//!   suppression and a minimum spacing so corners spread over the image.
//! * Descriptors: steered BRIEF as in E. Rublee, V. Rabaud, K. Konolige, G. Bradski, *ORB: an
//!   efficient alternative to SIFT or SURF*, ICCV 2011 — 256 intensity comparisons (M. Calonder
//!   et al., *BRIEF*, ECCV 2010) on a smoothed 31×31 patch, rotated by the patch's
//!   intensity-centroid orientation.
//! * Matching: brute-force Hamming distance with D. Lowe's ratio test (*Distinctive Image
//!   Features from Scale-Invariant Keypoints*, IJCV 2004) and a mutual-nearest check.
//! * Fitting: RANSAC (M. Fischler, R. Bolles, *Random Sample Consensus*, CACM 1981) over
//!   translation, similarity or homography models; homographies use the normalised DLT of
//!   R. Hartley, *In Defense of the Eight-Point Algorithm*, PAMI 1997. The winning model is refit
//!   to all inliers by least squares.
//!
//! Everything is deterministic (fixed-seed RNG). Images are `w × h` luminance in `0..=1`.

use crate::photo_util::par_map;
use crate::transform::Homography;

/// A detected corner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Corner {
    pub x: f32,
    pub y: f32,
    pub score: f32,
}

/// A 256-bit descriptor.
pub type Descriptor = [u64; 4];

/// Model fitted by [`ransac`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    Translation,
    /// Rotation, uniform scale and translation.
    Similarity,
    Homography,
}

/// Patch radius used by descriptors (corners closer than this to the border are dropped).
pub const PATCH_RADIUS: usize = 15;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn blur(w: usize, h: usize, v: &[f32], r: usize) -> Vec<f32> {
    let pass = |src: &[f32], horizontal: bool| -> Vec<f32> {
        let mut dst = vec![0.0f32; src.len()];
        let (n, m) = if horizontal { (h, w) } else { (w, h) };
        for line in 0..n {
            let at = |k: i64| {
                let k = k.clamp(0, m as i64 - 1) as usize;
                if horizontal { src[line * w + k] } else { src[k * w + line] }
            };
            let mut acc: f32 = (-(r as i64)..=r as i64).map(at).sum();
            for k in 0..m {
                let i = if horizontal { line * w + k } else { k * w + line };
                dst[i] = acc / (2 * r + 1) as f32;
                acc += at(k as i64 + r as i64 + 1) - at(k as i64 - r as i64);
            }
        }
        dst
    };
    pass(&pass(v, true), false)
}

/// Harris corners: at most `max` corners at least `min_dist` pixels apart, strongest first,
/// only where `valid` (if given) is true for the whole descriptor patch.
pub fn harris(w: usize, h: usize, img: &[f32], max: usize, min_dist: f32, valid: Option<&[bool]>) -> Vec<Corner> {
    if w < 2 * PATCH_RADIUS + 3 || h < 2 * PATCH_RADIUS + 3 {
        return Vec::new();
    }
    let (mut xx, mut yy, mut xy) = (vec![0.0f32; w * h], vec![0.0f32; w * h], vec![0.0f32; w * h]);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let p = |dx: i64, dy: i64| img[((y as i64 + dy) as usize) * w + (x as i64 + dx) as usize];
            // Sobel.
            let gx = (p(1, -1) + 2.0 * p(1, 0) + p(1, 1)) - (p(-1, -1) + 2.0 * p(-1, 0) + p(-1, 1));
            let gy = (p(-1, 1) + 2.0 * p(0, 1) + p(1, 1)) - (p(-1, -1) + 2.0 * p(0, -1) + p(1, -1));
            let i = y * w + x;
            xx[i] = gx * gx;
            yy[i] = gy * gy;
            xy[i] = gx * gy;
        }
    }
    // Two box passes approximate the Gaussian window.
    let (xx, yy, xy) = (blur(w, h, &blur(w, h, &xx, 1), 1), blur(w, h, &blur(w, h, &yy, 1), 1), blur(w, h, &blur(w, h, &xy, 1), 1));
    let r: Vec<f32> = (0..w * h).map(|i| xx[i] * yy[i] - xy[i] * xy[i] - 0.04 * (xx[i] + yy[i]).powi(2)).collect();
    let peak = r.iter().copied().fold(0.0f32, f32::max);
    if peak <= 0.0 {
        return Vec::new();
    }
    let m = PATCH_RADIUS + 1;
    let mut cand: Vec<Corner> = Vec::new();
    for y in m..h - m {
        for x in m..w - m {
            let v = r[y * w + x];
            if v < peak * 0.01 {
                continue;
            }
            let is_max = (-1i64..=1).all(|dy| (-1i64..=1).all(|dx| (dx == 0 && dy == 0) || r[((y as i64 + dy) as usize) * w + (x as i64 + dx) as usize] <= v));
            if !is_max {
                continue;
            }
            if let Some(ok) = valid {
                let all = (y - PATCH_RADIUS..=y + PATCH_RADIUS).step_by(5).all(|yy| (x - PATCH_RADIUS..=x + PATCH_RADIUS).step_by(5).all(|xx| ok[yy * w + xx]));
                if !all {
                    continue;
                }
            }
            cand.push(Corner { x: x as f32, y: y as f32, score: v });
        }
    }
    cand.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.y.total_cmp(&b.y)).then(a.x.total_cmp(&b.x)));
    // Minimum spacing via a coarse occupancy grid.
    let cell = min_dist.max(1.0);
    let (gw, gh) = ((w as f32 / cell).ceil() as usize + 1, (h as f32 / cell).ceil() as usize + 1);
    let mut grid: Vec<Vec<(f32, f32)>> = vec![Vec::new(); gw * gh];
    let mut out = Vec::new();
    for c in cand {
        let (gx, gy) = ((c.x / cell) as usize, (c.y / cell) as usize);
        let near = (gy.saturating_sub(1)..=(gy + 1).min(gh - 1))
            .any(|yy| (gx.saturating_sub(1)..=(gx + 1).min(gw - 1)).any(|xx| grid[yy * gw + xx].iter().any(|&(px, py)| (px - c.x).hypot(py - c.y) < min_dist)));
        if near {
            continue;
        }
        grid[gy * gw + gx].push((c.x, c.y));
        out.push(c);
        if out.len() >= max {
            break;
        }
    }
    out
}

/// The 256 BRIEF test pairs (fixed, Gaussian around the centre, σ = 31/5, clipped to the patch).
fn pairs() -> &'static [[f32; 4]; 256] {
    static P: std::sync::OnceLock<[[f32; 4]; 256]> = std::sync::OnceLock::new();
    P.get_or_init(|| {
        let mut rng = Rng(0x0B21_EF00);
        let mut gauss = || {
            // Box–Muller.
            let (u, v) = (rng.unit().max(1e-12), rng.unit());
            ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos() * 31.0 / 5.0).clamp(-13.0, 13.0) as f32
        };
        let mut p = [[0.0f32; 4]; 256];
        for q in &mut p {
            *q = [gauss(), gauss(), gauss(), gauss()];
        }
        p
    })
}

/// Steered-BRIEF descriptors for `corners` (orientation from the intensity centroid unless
/// `upright`). Returns one descriptor per corner, in order.
pub fn describe(w: usize, h: usize, img: &[f32], corners: &[Corner], upright: bool) -> Vec<Descriptor> {
    let smooth = blur(w, h, img, 2);
    let at = |x: f32, y: f32| -> f32 {
        let (xi, yi) = ((x.round() as i64).clamp(0, w as i64 - 1) as usize, (y.round() as i64).clamp(0, h as i64 - 1) as usize);
        smooth[yi * w + xi]
    };
    corners
        .iter()
        .map(|c| {
            let (s, co) = if upright {
                (0.0, 1.0)
            } else {
                let (mut m10, mut m01) = (0.0f32, 0.0f32);
                let r = PATCH_RADIUS as i64;
                for dy in -r..=r {
                    for dx in -r..=r {
                        if dx * dx + dy * dy <= r * r {
                            let v = at(c.x + dx as f32, c.y + dy as f32);
                            m10 += dx as f32 * v;
                            m01 += dy as f32 * v;
                        }
                    }
                }
                m01.atan2(m10).sin_cos()
            };
            let mut d = [0u64; 4];
            for (i, p) in pairs().iter().enumerate() {
                let rot = |x: f32, y: f32| (c.x + x * co - y * s, c.y + x * s + y * co);
                let (ax, ay) = rot(p[0], p[1]);
                let (bx, by) = rot(p[2], p[3]);
                if at(ax, ay) < at(bx, by) {
                    d[i / 64] |= 1 << (i % 64);
                }
            }
            d
        })
        .collect()
}

pub fn hamming(a: &Descriptor, b: &Descriptor) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

/// Matches `(index in a, index in b)` passing the ratio test and the mutual-nearest check.
pub fn match_descriptors(a: &[Descriptor], b: &[Descriptor], ratio: f32) -> Vec<(usize, usize)> {
    let best = |d: &Descriptor, set: &[Descriptor]| -> Option<(usize, u32, u32)> {
        let mut first = (usize::MAX, u32::MAX);
        let mut second = u32::MAX;
        for (j, e) in set.iter().enumerate() {
            let h = hamming(d, e);
            if h < first.1 {
                second = first.1;
                first = (j, h);
            } else if h < second {
                second = h;
            }
        }
        (first.0 != usize::MAX).then_some((first.0, first.1, second))
    };
    // Each query descriptor searches b independently (and cross-checks against a): parallel,
    // order preserved for determinism.
    let matched = par_map(a.len(), |i| {
        let (j, d1, d2) = best(&a[i], b)?;
        if d1 > 80 || (d2 != u32::MAX && d1 as f32 > ratio * d2 as f32) {
            return None;
        }
        best(&b[j], a).is_some_and(|(k, _, _)| k == i).then_some((i, j))
    });
    matched.into_iter().flatten().collect()
}

// ------------------------------------------------------------------ model fitting

/// Solve the `n × n` system `a · x = b` (row-major) by Gaussian elimination with partial pivoting.
fn solve(n: usize, mut a: Vec<f64>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i * n + c].abs().total_cmp(&a[j * n + c].abs()))?;
        if a[p * n + c].abs() < 1e-12 {
            return None;
        }
        if p != c {
            for k in 0..n {
                a.swap(c * n + k, p * n + k);
            }
            b.swap(c, p);
        }
        for r in c + 1..n {
            let f = a[r * n + c] / a[c * n + c];
            for k in c..n {
                a[r * n + k] -= f * a[c * n + k];
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|k| a[r * n + k] * x[k]).sum();
        x[r] = (b[r] - s) / a[r * n + r];
    }
    Some(x)
}

/// Least squares `min |A x − b|` through the normal equations.
fn least_squares(rows: &[(Vec<f64>, f64)], n: usize) -> Option<Vec<f64>> {
    let mut ata = vec![0.0; n * n];
    let mut atb = vec![0.0; n];
    for (r, b) in rows {
        for i in 0..n {
            atb[i] += r[i] * b;
            for j in 0..n {
                ata[i * n + j] += r[i] * r[j];
            }
        }
    }
    solve(n, ata, atb)
}

/// Hartley normalisation: centroid to the origin, mean distance √2.
fn normaliser(pts: &[[f64; 2]]) -> Homography {
    let n = pts.len().max(1) as f64;
    let (cx, cy) = (pts.iter().map(|p| p[0]).sum::<f64>() / n, pts.iter().map(|p| p[1]).sum::<f64>() / n);
    let d = pts.iter().map(|p| (p[0] - cx).hypot(p[1] - cy)).sum::<f64>() / n;
    let s = if d > 1e-12 { std::f64::consts::SQRT_2 / d } else { 1.0 };
    Homography([s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0])
}

/// Fit a model mapping `src[i]` → `dst[i]` by least squares (exact for minimal sets).
pub fn fit(model: Model, src: &[[f64; 2]], dst: &[[f64; 2]]) -> Option<Homography> {
    let n = src.len();
    match model {
        Model::Translation => {
            if n == 0 {
                return None;
            }
            let (tx, ty) = src.iter().zip(dst).fold((0.0, 0.0), |(ax, ay), (s, d)| (ax + d[0] - s[0], ay + d[1] - s[1]));
            Some(Homography([1.0, 0.0, tx / n as f64, 0.0, 1.0, ty / n as f64, 0.0, 0.0, 1.0]))
        }
        Model::Similarity => {
            if n < 2 {
                return None;
            }
            // x' = a·x − b·y + tx, y' = b·x + a·y + ty.
            let rows: Vec<(Vec<f64>, f64)> =
                src.iter().zip(dst).flat_map(|(s, d)| [(vec![s[0], -s[1], 1.0, 0.0], d[0]), (vec![s[1], s[0], 0.0, 1.0], d[1])]).collect();
            let v = least_squares(&rows, 4)?;
            Some(Homography([v[0], -v[1], v[2], v[1], v[0], v[3], 0.0, 0.0, 1.0]))
        }
        Model::Homography => {
            if n < 4 {
                return None;
            }
            let (ts, td) = (normaliser(src), normaliser(dst));
            let rows: Vec<(Vec<f64>, f64)> = src
                .iter()
                .zip(dst)
                .flat_map(|(s, d)| {
                    let (x, y) = ts.apply(s[0], s[1]);
                    let (u, v) = td.apply(d[0], d[1]);
                    [(vec![x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y], u), (vec![0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y], v)]
                })
                .collect();
            let h = least_squares(&rows, 8)?;
            let hn = Homography([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0]);
            let m = td.inverse()?.mul(&hn).mul(&ts);
            let k = m.0[8];
            (k.abs() > 1e-12).then(|| Homography(m.0.map(|v| v / k)))
        }
    }
}

fn min_points(model: Model) -> usize {
    match model {
        Model::Translation => 1,
        Model::Similarity => 2,
        Model::Homography => 4,
    }
}

/// RANSAC: the model with the most correspondences within `threshold` pixels after `iters`
/// random minimal samples, refit to its inliers. Returns the model and the inlier indices.
pub fn ransac(model: Model, src: &[[f64; 2]], dst: &[[f64; 2]], iters: usize, threshold: f64, seed: u64) -> Option<(Homography, Vec<usize>)> {
    let k = min_points(model);
    let n = src.len();
    if n < k {
        return None;
    }
    let mut rng = Rng(seed ^ 0xA11C_0DE5);
    let inliers_of = |h: &Homography| -> Vec<usize> {
        (0..n)
            .filter(|&i| {
                let (x, y) = h.apply(src[i][0], src[i][1]);
                x.is_finite() && (x - dst[i][0]).hypot(y - dst[i][1]) <= threshold
            })
            .collect()
    };
    let mut best: Option<(Homography, Vec<usize>)> = None;
    for _ in 0..iters.max(1) {
        let mut idx: Vec<usize> = Vec::with_capacity(k);
        while idx.len() < k {
            let i = rng.below(n);
            if !idx.contains(&i) {
                idx.push(i);
            }
            if n == k {
                idx = (0..k).collect();
            }
        }
        let (s, d): (Vec<[f64; 2]>, Vec<[f64; 2]>) = idx.iter().map(|&i| (src[i], dst[i])).unzip();
        let Some(h) = fit(model, &s, &d) else { continue };
        let inl = inliers_of(&h);
        if best.as_ref().is_none_or(|(_, b)| inl.len() > b.len()) {
            best = Some((h, inl));
        }
    }
    let (h, inl) = best?;
    if inl.len() < k {
        return None;
    }
    // Refit to all inliers, then re-collect.
    let (s, d): (Vec<[f64; 2]>, Vec<[f64; 2]>) = inl.iter().map(|&i| (src[i], dst[i])).unzip();
    let h2 = fit(model, &s, &d).unwrap_or(h);
    let inl2 = inliers_of(&h2);
    Some(if inl2.len() >= inl.len() { (h2, inl2) } else { (h, inl) })
}

/// Register `moving` onto `reference` (both `w × h` luminance; optional validity masks for
/// transparent areas). Returns the model mapping `moving` coordinates to `reference`
/// coordinates and the inlier count.
pub fn register(
    w: usize,
    h: usize,
    reference: &[f32],
    moving: &[f32],
    valid_ref: Option<&[bool]>,
    valid_mov: Option<&[bool]>,
    model: Model,
) -> Option<(Homography, usize)> {
    let upright = model == Model::Translation;
    let min_dist = (w.max(h) as f32 / 60.0).clamp(4.0, 24.0);
    let ca = harris(w, h, reference, 600, min_dist, valid_ref);
    let cb = harris(w, h, moving, 600, min_dist, valid_mov);
    if ca.len() < 4 || cb.len() < 4 {
        return None;
    }
    let (da, db) = (describe(w, h, reference, &ca, upright), describe(w, h, moving, &cb, upright));
    let matches = match_descriptors(&db, &da, 0.85);
    let src: Vec<[f64; 2]> = matches.iter().map(|&(i, _)| [cb[i].x as f64, cb[i].y as f64]).collect();
    let dst: Vec<[f64; 2]> = matches.iter().map(|&(_, j)| [ca[j].x as f64, ca[j].y as f64]).collect();
    let threshold = (w.max(h) as f64 / 400.0).clamp(1.5, 4.0);
    let (hm, inl) = ransac(model, &src, &dst, 1000, threshold, 7)?;
    (inl.len() >= min_points(model).max(6)).then_some((hm, inl.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic textured test image: blobs of varying brightness.
    fn texture(w: usize, h: usize) -> Vec<f32> {
        let mut rng = Rng(42);
        let mut img = vec![0.3f32; w * h];
        for _ in 0..140 {
            let (cx, cy) = (rng.unit() * w as f64, rng.unit() * h as f64);
            let r = 3.0 + rng.unit() * 9.0;
            let v = rng.unit() as f32;
            let sq = rng.unit() > 0.5;
            for y in 0..h {
                for x in 0..w {
                    let (dx, dy) = (x as f64 - cx, y as f64 - cy);
                    let inside = if sq { dx.abs() < r && dy.abs() < r * 0.6 } else { dx.hypot(dy) < r };
                    if inside {
                        img[y * w + x] = v;
                    }
                }
            }
        }
        img
    }

    /// Resample `img` by `h` (destination → source lookup with `inv`), bilinear, edge-clamped.
    fn warp(w: usize, hgt: usize, img: &[f32], inv: &Homography) -> Vec<f32> {
        let mut out = vec![0.3f32; w * hgt];
        for y in 0..hgt {
            for x in 0..w {
                let (sx, sy) = inv.apply(x as f64, y as f64);
                if sx < 0.0 || sy < 0.0 || sx >= (w - 1) as f64 || sy >= (hgt - 1) as f64 {
                    continue;
                }
                let (x0, y0) = (sx.floor() as usize, sy.floor() as usize);
                let (tx, ty) = ((sx - x0 as f64) as f32, (sy - y0 as f64) as f32);
                let p = |xx: usize, yy: usize| img[yy * w + xx];
                let top = p(x0, y0) + (p(x0 + 1, y0) - p(x0, y0)) * tx;
                let bot = p(x0, y0 + 1) + (p(x0 + 1, y0 + 1) - p(x0, y0 + 1)) * tx;
                out[y * w + x] = top + (bot - top) * ty;
            }
        }
        out
    }

    #[test]
    fn harris_finds_square_corners() {
        let (w, h) = (80, 80);
        let mut img = vec![0.0f32; w * h];
        for y in 30..50 {
            for x in 30..50 {
                img[y * w + x] = 1.0;
            }
        }
        let c = harris(w, h, &img, 10, 5.0, None);
        assert!(c.len() >= 4, "{c:?}");
        for (x, y) in [(30.0, 30.0), (49.0, 49.0)] {
            assert!(c.iter().any(|k| (k.x - x).abs() <= 2.0 && (k.y - y).abs() <= 2.0), "corner near ({x},{y}): {c:?}");
        }
    }

    #[test]
    fn fits_are_exact_on_clean_data() {
        let src = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [5.0, 3.0]];
        let truth = Homography([1.1, 0.05, 3.0, -0.02, 0.95, -2.0, 0.001, 0.0005, 1.0]);
        let dst: Vec<[f64; 2]> = src
            .iter()
            .map(|p| {
                let (x, y) = truth.apply(p[0], p[1]);
                [x, y]
            })
            .collect();
        let h = fit(Model::Homography, &src, &dst).unwrap();
        for p in &src {
            let (a, b) = (h.apply(p[0], p[1]), truth.apply(p[0], p[1]));
            assert!((a.0 - b.0).abs() < 1e-6 && (a.1 - b.1).abs() < 1e-6);
        }
        let sim = Homography([0.9, -0.2, 4.0, 0.2, 0.9, 1.0, 0.0, 0.0, 1.0]);
        let dst: Vec<[f64; 2]> = src
            .iter()
            .map(|p| {
                let (x, y) = sim.apply(p[0], p[1]);
                [x, y]
            })
            .collect();
        let h = fit(Model::Similarity, &src, &dst).unwrap();
        assert!(h.0.iter().zip(sim.0).all(|(a, b)| (a - b).abs() < 1e-9));
    }

    #[test]
    fn ransac_rejects_outliers() {
        let mut src = Vec::new();
        let mut dst = Vec::new();
        for i in 0..40 {
            let p = [(i * 7 % 50) as f64, (i * 13 % 50) as f64];
            src.push(p);
            dst.push(if i % 4 == 0 { [p[0] * 3.0 + 40.0, -p[1]] } else { [p[0] + 12.0, p[1] - 5.0] });
        }
        let (h, inl) = ransac(Model::Translation, &src, &dst, 100, 1.0, 1).unwrap();
        assert_eq!(inl.len(), 30);
        assert!((h.0[2] - 12.0).abs() < 1e-9 && (h.0[5] + 5.0).abs() < 1e-9);
    }

    #[test]
    fn registers_translation_and_similarity() {
        let (w, h) = (192, 160);
        let reference = texture(w, h);
        // The moving image is the reference shifted by (+9, −6): registration maps it back.
        let shift = Homography([1.0, 0.0, 9.0, 0.0, 1.0, -6.0, 0.0, 0.0, 1.0]);
        let moving = warp(w, h, &reference, &shift.inverse().unwrap());
        let (m, n) = register(w, h, &reference, &moving, None, None, Model::Translation).unwrap();
        assert!(n >= 6);
        assert!((m.0[2] + 9.0).abs() < 1.0 && (m.0[5] - 6.0).abs() < 1.0, "{m:?}");
        // Rotated 4° and scaled 1.03 about the centre.
        let (c, s) = (4f64.to_radians().cos() * 1.03, 4f64.to_radians().sin() * 1.03);
        let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
        let sim = Homography([c, -s, cx - c * cx + s * cy, s, c, cy - s * cx - c * cy, 0.0, 0.0, 1.0]);
        let moving = warp(w, h, &reference, &sim.inverse().unwrap());
        let (m, _) = register(w, h, &reference, &moving, None, None, Model::Similarity).unwrap();
        // m should undo sim: m · sim ≈ identity at a few points.
        for p in [[40.0, 40.0], [150.0, 120.0], [96.0, 80.0]] {
            let (x, y) = sim.apply(p[0], p[1]);
            let (bx, by) = m.apply(x, y);
            assert!((bx - p[0]).abs() < 1.5 && (by - p[1]).abs() < 1.5, "{p:?} → {bx},{by}");
        }
    }

    #[test]
    fn descriptors_are_deterministic() {
        let (w, h) = (96, 96);
        let img = texture(w, h);
        let c = harris(w, h, &img, 50, 6.0, None);
        assert_eq!(describe(w, h, &img, &c, false), describe(w, h, &img, &c, false));
        assert_eq!(hamming(&[0, 0, 0, 0], &[1, 3, 0, u64::MAX]), 67);
    }
}
