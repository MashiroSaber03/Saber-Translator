//! Panorama stitching for File › Automate › Photomerge and Edit › Auto-Align / Auto-Blend Layers.
//!
//! The pipeline follows M. Brown, D. G. Lowe, *Automatic Panoramic Image Stitching using
//! Invariant Features*, IJCV 2007, with classical parts throughout:
//!
//! 1. **Features**: Harris corners + steered BRIEF ([`crate::features`]).
//! 2. **Pair matching**: ratio-test matches, RANSAC homography, and Brown–Lowe's probabilistic
//!    verification (`inliers > 8 + 0.3·matches`).
//! 3. **Focal length** (for cylindrical / spherical layouts): from the pairwise homographies of
//!    a rotating camera (H. Shum, R. Szeliski, *Construction of Panoramic Image Mosaics with
//!    Global and Local Alignment*, IJCV 2000: `H = K R K⁻¹` makes the columns and rows of
//!    `K⁻¹HK` orthonormal), or from EXIF.
//! 4. **Projection + motion model** per layout: Perspective (plane, homography), Cylindrical
//!    and Spherical (keypoints mapped onto the cylinder / sphere, rigid motion there), Collage
//!    (similarity), Reposition (translation). Auto picks perspective unless the result is too
//!    stretched, then cylindrical or spherical (Photoshop's rule of thumb).
//! 5. **Chained alignment + bundle adjustment**: a maximum spanning tree over the match graph
//!    from the reference image gives initial transforms; Levenberg–Marquardt then minimises
//!    the distance between all matched points in the panorama (optionally refining one radial
//!    distortion coefficient, Photoshop's *Geometric Distortion Correction*, and the focal
//!    length of the cylinder / sphere).
//! 6. **Photometric**: gain compensation (Brown–Lowe §6) jointly with a radial vignetting model
//!    `log V(r) = a·r² + b·r⁴` shared by all images (D. Goldman, *Vignette and Exposure
//!    Calibration and Compensation*, PAMI 2010), solved by linear least squares in log space.
//! 7. **Seams**: per new image, a minimum-cost path (dynamic programming, as in seam carving:
//!    S. Avidan, A. Shamir, SIGGRAPH 2007) through its overlap with the composite on the colour
//!    difference, so seams avoid places where the images disagree (moving objects, parallax).
//! 8. **Multi-band blending**: Laplacian pyramids (P. Burt, E. Adelson, *A Multiresolution
//!    Spline with Application to Image Mosaics*, ACM TOG 1983) of each image (holes filled by
//!    push-pull so edges don't darken) weighted by the Gaussian pyramid of its seam mask.
//!
//! Everything is deterministic. Registration works on luminance in `0..=1` at a reduced scale;
//! [`Placement::scaled`] maps the result back to full resolution.

#![allow(clippy::needless_range_loop)] // index loops mirror the maths (pixels × channels)

use crate::features::{self, Descriptor, Model};
use crate::photo_util::{Normal, Rng, par_map, solve_dense};
use crate::transform::Homography;

/// Photomerge layouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Auto,
    Perspective,
    Cylindrical,
    Spherical,
    Collage,
    Reposition,
}

impl Layout {
    pub fn parse(s: &str) -> Option<Layout> {
        Some(match s {
            "auto" => Layout::Auto,
            "perspective" => Layout::Perspective,
            "cylindrical" => Layout::Cylindrical,
            "spherical" => Layout::Spherical,
            "collage" => Layout::Collage,
            "reposition" => Layout::Reposition,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Layout::Auto => "auto",
            Layout::Perspective => "perspective",
            Layout::Cylindrical => "cylindrical",
            Layout::Spherical => "spherical",
            Layout::Collage => "collage",
            Layout::Reposition => "reposition",
        }
    }
    fn projection(self) -> Projection {
        match self {
            Layout::Cylindrical => Projection::Cylinder,
            Layout::Spherical => Projection::Sphere,
            _ => Projection::Plane,
        }
    }
    fn motion(self) -> Motion {
        match self {
            Layout::Auto | Layout::Perspective => Motion::Homography,
            Layout::Cylindrical | Layout::Spherical => Motion::Euclidean,
            Layout::Collage => Motion::Similarity,
            Layout::Reposition => Motion::Translation,
        }
    }
}

/// Surface an image is projected onto before its 2-D motion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Projection {
    Plane,
    Cylinder,
    Sphere,
}

/// 2-D motion models in projected space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Motion {
    Translation,
    /// Rotation + translation.
    Euclidean,
    /// Rotation, uniform scale, translation.
    Similarity,
    Homography,
}

impl Motion {
    fn min_points(self) -> usize {
        match self {
            Motion::Translation => 1,
            Motion::Euclidean | Motion::Similarity => 2,
            Motion::Homography => 4,
        }
    }
    fn n_params(self) -> usize {
        match self {
            Motion::Translation => 2,
            Motion::Euclidean => 3,
            Motion::Similarity => 4,
            Motion::Homography => 8,
        }
    }
    fn to_params(self, h: &Homography) -> Vec<f64> {
        let m = h.0.map(|v| v / h.0[8]);
        match self {
            Motion::Translation => vec![m[2], m[5]],
            Motion::Euclidean => vec![m[3].atan2(m[0]), m[2], m[5]],
            Motion::Similarity => vec![m[0], m[3], m[2], m[5]],
            Motion::Homography => m[..8].to_vec(),
        }
    }
    fn to_homography(self, p: &[f64]) -> Homography {
        match self {
            Motion::Translation => Homography([1.0, 0.0, p[0], 0.0, 1.0, p[1], 0.0, 0.0, 1.0]),
            Motion::Euclidean => {
                let (s, c) = p[0].sin_cos();
                Homography([c, -s, p[1], s, c, p[2], 0.0, 0.0, 1.0])
            }
            Motion::Similarity => Homography([p[0], -p[1], p[2], p[1], p[0], p[3], 0.0, 0.0, 1.0]),
            Motion::Homography => Homography([p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 1.0]),
        }
    }
}

/// Least-squares fit of `motion` mapping `src[i]` → `dst[i]`.
pub fn fit_motion(motion: Motion, src: &[[f64; 2]], dst: &[[f64; 2]]) -> Option<Homography> {
    match motion {
        Motion::Translation => features::fit(Model::Translation, src, dst),
        Motion::Similarity => features::fit(Model::Similarity, src, dst),
        Motion::Homography => features::fit(Model::Homography, src, dst),
        Motion::Euclidean => {
            let n = src.len();
            if n < 2 {
                return None;
            }
            let nf = n as f64;
            let cs = src.iter().fold([0.0, 0.0], |a, p| [a[0] + p[0] / nf, a[1] + p[1] / nf]);
            let cd = dst.iter().fold([0.0, 0.0], |a, p| [a[0] + p[0] / nf, a[1] + p[1] / nf]);
            let (mut sxx, mut sxy) = (0.0, 0.0);
            for (s, d) in src.iter().zip(dst) {
                let (a, b) = ([s[0] - cs[0], s[1] - cs[1]], [d[0] - cd[0], d[1] - cd[1]]);
                sxx += a[0] * b[0] + a[1] * b[1];
                sxy += a[0] * b[1] - a[1] * b[0];
            }
            let th = sxy.atan2(sxx);
            let (s, c) = th.sin_cos();
            Some(Homography([c, -s, cd[0] - (c * cs[0] - s * cs[1]), s, c, cd[1] - (s * cs[0] + c * cs[1]), 0.0, 0.0, 1.0]))
        }
    }
}

/// RANSAC over [`Motion`] models; returns the refit model and its inlier indices.
pub fn ransac_motion(motion: Motion, src: &[[f64; 2]], dst: &[[f64; 2]], iters: usize, threshold: f64, seed: u64) -> Option<(Homography, Vec<usize>)> {
    let k = motion.min_points();
    let n = src.len();
    if n < k {
        return None;
    }
    let mut rng = Rng(seed ^ 0x5EED_0FA1);
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
        if n == k {
            idx = (0..k).collect();
        }
        while idx.len() < k {
            let i = rng.below(n);
            if !idx.contains(&i) {
                idx.push(i);
            }
        }
        let (s, d): (Vec<[f64; 2]>, Vec<[f64; 2]>) = idx.iter().map(|&i| (src[i], dst[i])).unzip();
        let Some(h) = fit_motion(motion, &s, &d) else { continue };
        let inl = inliers_of(&h);
        if best.as_ref().is_none_or(|(_, b)| inl.len() > b.len()) {
            best = Some((h, inl));
        }
    }
    let (h, inl) = best?;
    if inl.len() < k {
        return None;
    }
    let (s, d): (Vec<[f64; 2]>, Vec<[f64; 2]>) = inl.iter().map(|&i| (src[i], dst[i])).unzip();
    let h2 = fit_motion(motion, &s, &d).unwrap_or(h);
    let inl2 = inliers_of(&h2);
    Some(if inl2.len() >= inl.len() { (h2, inl2) } else { (h, inl) })
}

/// Where one image lands in the panorama: image px → (undistort) → projection → 2-D motion.
#[derive(Clone, Debug, PartialEq)]
pub struct Placement {
    pub projection: Projection,
    /// Focal length in px (the projection radius).
    pub focal: f64,
    /// Projection centre in image px (the image centre).
    pub center: [f64; 2],
    /// Radial distortion `k1` on radii normalised by the focal length (0 = none).
    pub k1: f64,
    /// Projected coordinates → panorama coordinates.
    pub h: Homography,
}

impl Placement {
    /// Image px → projected coordinates (centred, in px of the focal length).
    pub fn project(&self, x: f64, y: f64) -> [f64; 2] {
        let (mut dx, mut dy) = (x - self.center[0], y - self.center[1]);
        if self.k1 != 0.0 {
            let s = 1.0 + self.k1 * (dx * dx + dy * dy) / (self.focal * self.focal);
            dx *= s;
            dy *= s;
        }
        let f = self.focal;
        match self.projection {
            Projection::Plane => [dx, dy],
            Projection::Cylinder => [f * dx.atan2(f), f * dy / dx.hypot(f)],
            Projection::Sphere => [f * dx.atan2(f), f * dy.atan2(dx.hypot(f))],
        }
    }

    /// Projected → image px (`None` outside the valid hemisphere).
    pub fn unproject(&self, a: f64, b: f64) -> Option<(f64, f64)> {
        let f = self.focal;
        let (mut dx, mut dy) = match self.projection {
            Projection::Plane => (a, b),
            Projection::Cylinder => {
                let th = a / f;
                if th.abs() >= 1.55 {
                    return None;
                }
                (f * th.tan(), b / th.cos())
            }
            Projection::Sphere => {
                let (th, ph) = (a / f, b / f);
                if th.abs() >= 1.55 || ph.abs() >= 1.55 {
                    return None;
                }
                (f * th.tan(), f * ph.tan() / th.cos())
            }
        };
        if self.k1 != 0.0 {
            // Invert r_u = r_d (1 + k1 r_d² / f²) by Newton steps on r_d.
            let ru = dx.hypot(dy);
            if ru > 1e-9 {
                let mut rd = ru;
                for _ in 0..8 {
                    let g = rd * (1.0 + self.k1 * rd * rd / (f * f)) - ru;
                    let dg = 1.0 + 3.0 * self.k1 * rd * rd / (f * f);
                    if dg.abs() < 1e-9 {
                        break;
                    }
                    rd -= g / dg;
                }
                if !(rd.is_finite() && rd > 0.0) {
                    return None;
                }
                dx *= rd / ru;
                dy *= rd / ru;
            }
        }
        Some((dx + self.center[0], dy + self.center[1]))
    }

    /// Image px → panorama px.
    pub fn forward(&self, x: f64, y: f64) -> (f64, f64) {
        let p = self.project(x, y);
        self.h.apply(p[0], p[1])
    }

    /// Panorama px → image px.
    pub fn inverse(&self, u: f64, v: f64) -> Option<(f64, f64)> {
        let inv = self.h.inverse()?;
        let m = &inv.0;
        let w = m[6] * u + m[7] * v + m[8];
        if w <= 1e-12 {
            return None;
        }
        let (a, b) = inv.apply(u, v);
        self.unproject(a, b)
    }

    /// The same placement for images `k×` larger (registration ran at `1/k` scale).
    pub fn scaled(&self, k: f64) -> Placement {
        let s = Homography([k, 0.0, 0.0, 0.0, k, 0.0, 0.0, 0.0, 1.0]);
        let si = Homography([1.0 / k, 0.0, 0.0, 0.0, 1.0 / k, 0.0, 0.0, 0.0, 1.0]);
        Placement {
            projection: self.projection,
            focal: self.focal * k,
            center: [self.center[0] * k, self.center[1] * k],
            k1: self.k1,
            h: s.mul(&self.h).mul(&si),
        }
    }

    /// Shifted in the panorama.
    pub fn translated(&self, dx: f64, dy: f64) -> Placement {
        let t = Homography([1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0]);
        Placement { h: t.mul(&self.h), ..self.clone() }
    }

    /// Bounds `[x0, y0, x1, y1]` of a `w × h` image in the panorama (sampled along its edges).
    pub fn bounds(&self, w: f64, h: f64) -> [f64; 4] {
        let mut b = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
        let n = 32;
        for i in 0..=n {
            let t = i as f64 / n as f64;
            for (x, y) in [(t * w, 0.0), (t * w, h), (0.0, t * h), (w, t * h)] {
                let (u, v) = self.forward(x, y);
                if u.is_finite() && v.is_finite() {
                    b = [b[0].min(u), b[1].min(v), b[2].max(u), b[3].max(v)];
                }
            }
        }
        b
    }
}

/// Keypoints and descriptors of one image (registration scale).
#[derive(Clone, Debug, Default)]
pub struct ImageFeatures {
    pub w: usize,
    pub h: usize,
    pub points: Vec<[f64; 2]>,
    pub desc: Vec<Descriptor>,
}

/// Detects up to `max` corners of a `w × h` luminance image and describes them.
pub fn detect(w: usize, h: usize, luma: &[f32], valid: Option<&[bool]>, max: usize) -> ImageFeatures {
    let min_dist = (w.max(h) as f32 / 70.0).clamp(3.0, 20.0);
    let corners = features::harris(w, h, luma, max, min_dist, valid);
    let desc = features::describe(w, h, luma, &corners, false);
    ImageFeatures { w, h, points: corners.iter().map(|c| [c.x as f64, c.y as f64]).collect(), desc }
}

/// One image for [`detect_many`]: `(w, h, luma, valid)`.
pub type PreparedImage = (usize, usize, Vec<f32>, Option<Vec<bool>>);

/// [`detect`] for several images `(w, h, luma, valid)` in parallel.
pub fn detect_many(imgs: &[PreparedImage], max: usize) -> Vec<ImageFeatures> {
    par_map(imgs.len(), |i| {
        let (w, h, l, v) = &imgs[i];
        detect(*w, *h, l, v.as_deref(), max)
    })
}

/// A verified image pair: point correspondences (image `a` px, image `b` px).
#[derive(Clone, Debug)]
pub struct PairMatch {
    pub a: usize,
    pub b: usize,
    pub pts_a: Vec<[f64; 2]>,
    pub pts_b: Vec<[f64; 2]>,
    /// Homography `b → a` in image px.
    pub homography: Homography,
}

/// Matches every pair and keeps those that pass RANSAC + the Brown–Lowe verification.
pub fn match_pairs(feats: &[ImageFeatures]) -> Vec<PairMatch> {
    let n = feats.len();
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|a| (a + 1..n).map(move |b| (a, b))).collect();
    let out = par_map(pairs.len(), |k| {
        let (a, b) = pairs[k];
        let (fa, fb) = (&feats[a], &feats[b]);
        if fa.points.len() < 6 || fb.points.len() < 6 {
            return None;
        }
        let m = features::match_descriptors(&fb.desc, &fa.desc, 0.8);
        if m.len() < 8 {
            return None;
        }
        let src: Vec<[f64; 2]> = m.iter().map(|&(i, _)| fb.points[i]).collect();
        let dst: Vec<[f64; 2]> = m.iter().map(|&(_, j)| fa.points[j]).collect();
        let thr = (fa.w.max(fa.h) as f64 / 300.0).clamp(1.5, 4.0);
        let (hm, inl) = ransac_motion(Motion::Homography, &src, &dst, 1200, thr, (a * 131 + b) as u64)?;
        if (inl.len() as f64) < 8.0 + 0.3 * m.len() as f64 || inl.len() < 10 {
            return None;
        }
        Some(PairMatch { a, b, pts_a: inl.iter().map(|&i| dst[i]).collect(), pts_b: inl.iter().map(|&i| src[i]).collect(), homography: hm })
    });
    out.into_iter().flatten().collect()
}

/// Focal length (px) of a rotating camera from a homography between two of its images (image
/// px, both centred at `c`), or `None` when the homography doesn't constrain it.
pub fn focal_from_homography(h: &Homography, ca: [f64; 2], cb: [f64; 2]) -> Option<f64> {
    let ta = Homography([1.0, 0.0, -ca[0], 0.0, 1.0, -ca[1], 0.0, 0.0, 1.0]);
    let tb = Homography([1.0, 0.0, cb[0], 0.0, 1.0, cb[1], 0.0, 0.0, 1.0]);
    let m = ta.mul(h).mul(&tb).0;
    let pick = |c1: (f64, f64), c2: (f64, f64)| -> Option<f64> {
        // Each candidate is (numerator, denominator) of f²; prefer the better-conditioned one.
        let ok = |(n, d): (f64, f64)| (d.abs() > 1e-12 && n / d > 0.0).then_some((n / d, d.abs()));
        match (ok(c1), ok(c2)) {
            (Some(a), Some(b)) => Some(if a.1 >= b.1 { a.0 } else { b.0 }),
            (Some(a), None) | (None, Some(a)) => Some(a.0),
            _ => None,
        }
    };
    // Columns of K⁻¹HK orthogonal and of equal norm (focal of b) …
    let f1 = pick((-(m[0] * m[1] + m[3] * m[4]), m[6] * m[7]), (m[1] * m[1] + m[4] * m[4] - m[0] * m[0] - m[3] * m[3], m[6] * m[6] - m[7] * m[7]));
    // … rows likewise (focal of a).
    let f0 = pick((-(m[2] * m[5]), m[0] * m[3] + m[1] * m[4]), (m[5] * m[5] - m[2] * m[2], m[0] * m[0] + m[1] * m[1] - m[3] * m[3] - m[4] * m[4]));
    match (f0, f1) {
        (Some(a), Some(b)) => Some((a * b).sqrt().sqrt()),
        (Some(a), None) | (None, Some(a)) => Some(a.sqrt()),
        _ => None,
    }
}

/// Options for [`align`].
#[derive(Clone, Debug)]
pub struct AlignOptions {
    pub layout: Layout,
    /// Focal length in registration px (EXIF); estimated when `None`.
    pub focal: Option<f64>,
    /// Image that stays put (`None`: the best-connected one).
    pub reference: Option<usize>,
    /// Refine a radial distortion coefficient (Geometric Distortion Correction).
    pub geometric: bool,
}

/// Result of [`align`].
#[derive(Clone, Debug)]
pub struct Alignment {
    /// The layout actually used (Auto resolved).
    pub layout: Layout,
    pub reference: usize,
    /// Per image; `None` for images that couldn't be connected.
    pub placements: Vec<Option<Placement>>,
    pub focal: f64,
    pub k1: f64,
    /// Verified pairs `(a, b, inliers)`.
    pub pairs: Vec<(usize, usize, usize)>,
    /// RMS reprojection error (px, registration scale) after bundle adjustment.
    pub rms: f64,
}

struct PairData {
    a: usize,
    b: usize,
    pa: Vec<[f64; 2]>,
    pb: Vec<[f64; 2]>,
}

fn base_placement(f: &ImageFeatures, projection: Projection, focal: f64, k1: f64) -> Placement {
    Placement { projection, focal, center: [f.w as f64 / 2.0, f.h as f64 / 2.0], k1, h: Homography::IDENTITY }
}

/// Aligns images from their features. `None` when fewer than two images connect.
pub fn align(feats: &[ImageFeatures], matches: &[PairMatch], opts: &AlignOptions) -> Option<Alignment> {
    let n = feats.len();
    if n < 2 || matches.is_empty() {
        return None;
    }
    let focal = opts.focal.filter(|f| *f > 1.0).unwrap_or_else(|| {
        let mut fs: Vec<f64> = matches
            .iter()
            .filter_map(|m| {
                let (fa, fb) = (&feats[m.a], &feats[m.b]);
                focal_from_homography(&m.homography, [fa.w as f64 / 2.0, fa.h as f64 / 2.0], [fb.w as f64 / 2.0, fb.h as f64 / 2.0])
            })
            .filter(|f| f.is_finite())
            .collect();
        let side = feats.iter().map(|f| f.w.max(f.h)).max().unwrap_or(1) as f64;
        fs.retain(|f| *f > side * 0.2 && *f < side * 20.0);
        fs.sort_by(f64::total_cmp);
        if fs.is_empty() { side } else { fs[fs.len() / 2] }
    });
    if opts.layout != Layout::Auto {
        return align_layout(feats, matches, opts, opts.layout, focal);
    }
    // Auto: perspective unless it is too stretched (Photoshop's rule of thumb), then cylindrical,
    // or spherical when the panorama is also tall.
    if let Some(a) = align_layout(feats, matches, opts, Layout::Perspective, focal)
        && !too_stretched(feats, &a)
    {
        return Some(Alignment { layout: Layout::Perspective, ..a });
    }
    let cyl = align_layout(feats, matches, opts, Layout::Cylindrical, focal)?;
    let (mut y0, mut y1) = (f64::MAX, f64::MIN);
    for (p, f) in cyl.placements.iter().zip(feats) {
        if let Some(p) = p {
            let b = p.bounds(f.w as f64, f.h as f64);
            y0 = y0.min(b[1]);
            y1 = y1.max(b[3]);
        }
    }
    if (y1 - y0) / cyl.focal > 1.4 {
        return align_layout(feats, matches, opts, Layout::Spherical, cyl.focal);
    }
    Some(cyl)
}

/// A perspective panorama is too stretched when an image grows or shrinks a lot, flips, or
/// the field of view exceeds about 110°.
fn too_stretched(feats: &[ImageFeatures], a: &Alignment) -> bool {
    let (mut x0, mut x1) = (f64::MAX, f64::MIN);
    for (p, f) in a.placements.iter().zip(feats) {
        let Some(p) = p else { continue };
        let (w, h) = (f.w as f64, f.h as f64);
        let c: Vec<(f64, f64)> = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)].iter().map(|&(x, y)| p.forward(x, y)).collect();
        // Signed area (shoelace); orientation must be preserved.
        let area: f64 = (0..4).map(|i| c[i].0 * c[(i + 1) % 4].1 - c[(i + 1) % 4].0 * c[i].1).sum::<f64>() / 2.0;
        let ratio = area / (w * h);
        if !ratio.is_finite() || !(0.3..=3.0).contains(&ratio) {
            return true;
        }
        let m = &p.h.0;
        for &(x, y) in &[(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)] {
            let q = p.project(x, y);
            if m[6] * q[0] + m[7] * q[1] + m[8] <= 0.0 {
                return true;
            }
        }
        for (u, _) in &c {
            x0 = x0.min(*u);
            x1 = x1.max(*u);
        }
    }
    // Horizontal field of view through the reference's focal length.
    let fov = 2.0 * ((x1 - x0) / 2.0 / a.focal).atan();
    fov > 110f64.to_radians()
}

fn align_layout(feats: &[ImageFeatures], matches: &[PairMatch], opts: &AlignOptions, layout: Layout, focal: f64) -> Option<Alignment> {
    let n = feats.len();
    let projection = layout.projection();
    let motion = layout.motion();
    // Pairwise motions in projected space.
    let mut pairs: Vec<PairData> = Vec::new();
    let mut edges: Vec<(usize, usize, usize, Homography)> = Vec::new();
    for m in matches {
        let (pa, pb) = (base_placement(&feats[m.a], projection, focal, 0.0), base_placement(&feats[m.b], projection, focal, 0.0));
        let qa: Vec<[f64; 2]> = m.pts_a.iter().map(|p| pa.project(p[0], p[1])).collect();
        let qb: Vec<[f64; 2]> = m.pts_b.iter().map(|p| pb.project(p[0], p[1])).collect();
        let thr = (feats[m.a].w.max(feats[m.a].h) as f64 / 250.0).clamp(2.0, 5.0);
        let Some((hm, inl)) = ransac_motion(motion, &qb, &qa, 600, thr, (m.a * 977 + m.b) as u64) else { continue };
        if inl.len() < 8 || (inl.len() as f64) < 0.5 * m.pts_a.len() as f64 {
            continue;
        }
        edges.push((m.a, m.b, inl.len(), hm));
        // Bundle adjustment uses at most 80 evenly spread inliers per pair.
        let step = inl.len().div_ceil(80).max(1);
        let sel: Vec<usize> = inl.iter().step_by(step).copied().collect();
        pairs.push(PairData { a: m.a, b: m.b, pa: sel.iter().map(|&i| m.pts_a[i]).collect(), pb: sel.iter().map(|&i| m.pts_b[i]).collect() });
    }
    if edges.is_empty() {
        return None;
    }
    let mut degree = vec![0usize; n];
    for e in &edges {
        degree[e.0] += e.2;
        degree[e.1] += e.2;
    }
    let reference = opts.reference.filter(|r| *r < n).unwrap_or_else(|| {
        // Best connected; ties go to the middle image.
        (0..n).max_by(|&i, &j| degree[i].cmp(&degree[j]).then_with(|| (j as i64 - n as i64 / 2).abs().cmp(&(i as i64 - n as i64 / 2).abs()))).unwrap_or(0)
    });
    // Maximum spanning tree (Prim) from the reference; transforms map projected → panorama.
    let mut tf: Vec<Option<Homography>> = vec![None; n];
    let rc = [feats[reference].w as f64 / 2.0, feats[reference].h as f64 / 2.0];
    tf[reference] = Some(Homography([1.0, 0.0, rc[0], 0.0, 1.0, rc[1], 0.0, 0.0, 1.0]));
    loop {
        let mut best: Option<(usize, usize, Homography)> = None;
        for e in &edges {
            // e.3 maps b → a.
            let cand = match (tf[e.0], tf[e.1]) {
                (Some(ta), None) => Some((e.1, ta.mul(&e.3))),
                (None, Some(tb)) => e.3.inverse().map(|inv| (e.0, tb.mul(&inv))),
                _ => None,
            };
            if let Some((node, t)) = cand
                && best.as_ref().is_none_or(|b| e.2 > b.1)
            {
                best = Some((node, e.2, t));
            }
        }
        let Some((node, _, t)) = best else { break };
        tf[node] = Some(t);
    }
    let placed: Vec<usize> = (0..n).filter(|&i| tf[i].is_some()).collect();
    if placed.len() < 2 {
        return None;
    }
    // Bundle adjustment (Levenberg–Marquardt, numeric Jacobian).
    let movable: Vec<usize> = placed.iter().copied().filter(|&i| i != reference).collect();
    let mp = motion.n_params();
    let refine_focal = projection != Projection::Plane;
    let n_glob = usize::from(opts.geometric) + usize::from(refine_focal);
    let mut params: Vec<f64> = movable.iter().filter_map(|&i| tf[i].as_ref()).flat_map(|t| motion.to_params(t)).collect();
    if opts.geometric {
        params.push(0.0);
    }
    if refine_focal {
        params.push(focal.ln());
    }
    let n_par = params.len();
    let decode = |p: &[f64]| -> Vec<Option<Placement>> {
        let k1 = if opts.geometric { p[movable.len() * mp] } else { 0.0 };
        let f = if refine_focal { p[n_par - 1].exp() } else { focal };
        (0..n)
            .map(|i| {
                let h = if i == reference {
                    tf[reference]?
                } else {
                    let k = movable.iter().position(|&m| m == i)?;
                    motion.to_homography(&p[k * mp..(k + 1) * mp])
                };
                Some(Placement { h, ..base_placement(&feats[i], projection, f, k1) })
            })
            .collect()
    };
    let residuals = |p: &[f64]| -> Vec<f64> {
        let pl = decode(p);
        let mut r = Vec::new();
        for pd in &pairs {
            let (Some(a), Some(b)) = (&pl[pd.a], &pl[pd.b]) else { continue };
            for (qa, qb) in pd.pa.iter().zip(&pd.pb) {
                let (ua, va) = a.forward(qa[0], qa[1]);
                let (ub, vb) = b.forward(qb[0], qb[1]);
                r.push(ua - ub);
                r.push(va - vb);
            }
        }
        // Soft priors keep the global terms sane when few pairs constrain them.
        if opts.geometric {
            r.push(p[movable.len() * mp] * 20.0);
        }
        if refine_focal {
            r.push((p[n_par - 1] - focal.ln()) * 2.0);
        }
        r
    };
    let cost = |r: &[f64]| r.iter().map(|v| v * v).sum::<f64>();
    let mut cur = residuals(&params);
    let mut lambda = 1e-3;
    if n_par > 0 && !movable.is_empty() {
        for _ in 0..40 {
            let m_res = cur.len();
            let cols: Vec<Vec<f64>> = par_map(n_par, |j| {
                let step = 1e-6 * (1.0 + params[j].abs());
                let mut pp = params.clone();
                pp[j] += step;
                let rp = residuals(&pp);
                pp[j] -= 2.0 * step;
                let rm = residuals(&pp);
                (0..m_res).map(|i| (rp[i] - rm[i]) / (2.0 * step)).collect()
            });
            let mut jtj = vec![0.0; n_par * n_par];
            let mut jtr = vec![0.0; n_par];
            for a in 0..n_par {
                jtr[a] = -cols[a].iter().zip(&cur).map(|(j, r)| j * r).sum::<f64>();
                for b in a..n_par {
                    let v: f64 = cols[a].iter().zip(&cols[b]).map(|(x, y)| x * y).sum();
                    jtj[a * n_par + b] = v;
                    jtj[b * n_par + a] = v;
                }
            }
            let mut improved = false;
            for _ in 0..8 {
                let mut a = jtj.clone();
                for d in 0..n_par {
                    a[d * n_par + d] += lambda * (jtj[d * n_par + d] + 1e-9);
                }
                let Some(delta) = solve_dense(n_par, a, jtr.clone()) else {
                    lambda *= 10.0;
                    continue;
                };
                let trial: Vec<f64> = params.iter().zip(&delta).map(|(p, d)| p + d).collect();
                let rt = residuals(&trial);
                if cost(&rt) < cost(&cur) {
                    let gain = cost(&cur) - cost(&rt);
                    params = trial;
                    cur = rt;
                    lambda = (lambda * 0.3).max(1e-9);
                    improved = gain > 1e-10 * cost(&cur).max(1e-12);
                    break;
                }
                lambda *= 10.0;
            }
            if !improved {
                break;
            }
        }
    }
    let placements = decode(&params);
    let n_res = pairs.iter().map(|p| p.pa.len()).sum::<usize>().max(1);
    let rms = (cost(&cur[..(cur.len() - n_glob).min(cur.len())]) / n_res as f64).sqrt();
    let k1 = if opts.geometric { params[movable.len() * mp] } else { 0.0 };
    let focal = if refine_focal { params[n_par - 1].exp() } else { focal };
    Some(Alignment { layout, reference, placements, focal, k1, pairs: edges.iter().map(|e| (e.0, e.1, e.2)).collect(), rms })
}

// ---------- photometric ----------

/// One image warped into the panorama: an axis-aligned region with colour channels and coverage.
#[derive(Clone, Debug)]
pub struct RoiImage {
    pub x0: i32,
    pub y0: i32,
    pub w: usize,
    pub h: usize,
    /// Colour channels per pixel (no alpha).
    pub ch: usize,
    pub px: Vec<f32>,
    /// Coverage `0..=1`.
    pub alpha: Vec<f32>,
}

impl RoiImage {
    fn at(&self, x: i32, y: i32) -> Option<usize> {
        let (lx, ly) = (x - self.x0, y - self.y0);
        (lx >= 0 && ly >= 0 && (lx as usize) < self.w && (ly as usize) < self.h).then(|| ly as usize * self.w + lx as usize)
    }
}

/// Placements and source sizes (for the vignetting radius).
pub type Geometry<'a> = (&'a [Placement], &'a [(usize, usize)]);

/// Gains (per image, per channel) and the vignetting model found by [`photometric`].
#[derive(Clone, Debug, PartialEq)]
pub struct Photometric {
    pub gains: Vec<Vec<f32>>,
    /// `log V(r) = a·r² + b·r⁴`, `r` = distance from the image centre over the half diagonal.
    pub vignette: [f64; 2],
}

impl Photometric {
    /// Multiplier that corrects a pixel of image `i` at normalised squared radius `r2`.
    pub fn factor(&self, i: usize, c: usize, r2: f64) -> f32 {
        let v = (self.vignette[0] * r2 + self.vignette[1] * r2 * r2).exp();
        (1.0 / (self.gains[i][c] as f64 * v)) as f32
    }
}

/// Estimates exposure gains (`gain`) and a shared vignetting falloff (`vignette`) from the
/// overlaps of `imgs` (`placements`/`sizes` give each panorama pixel's source radius).
/// Without `geometry` (placements and source sizes) the vignetting model is not estimated.
pub fn photometric(imgs: &[RoiImage], geometry: Option<Geometry>, gain: bool, vignette: bool) -> Photometric {
    let vignette = vignette && geometry.is_some();
    let n = imgs.len();
    let ch = imgs.first().map_or(1, |i| i.ch);
    let identity = Photometric { gains: vec![vec![1.0; ch]; n], vignette: [0.0, 0.0] };
    if n == 0 || (!gain && !vignette) {
        return identity;
    }
    // Overlap samples on a grid: (a, b, colours, radii²).
    let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for im in imgs {
        x0 = x0.min(im.x0);
        y0 = y0.min(im.y0);
        x1 = x1.max(im.x0 + im.w as i32);
        y1 = y1.max(im.y0 + im.h as i32);
    }
    let step = (((x1 - x0) as i64 * (y1 - y0) as i64 / 40_000).max(1) as f64).sqrt().ceil() as i32;
    let r2_of = |i: usize, u: i32, v: i32| -> Option<f64> {
        let Some((placements, sizes)) = geometry else { return Some(0.0) };
        let (sx, sy) = placements[i].inverse(u as f64 + 0.5, v as f64 + 0.5)?;
        let (w, h) = (sizes[i].0 as f64, sizes[i].1 as f64);
        let (dx, dy) = (sx - w / 2.0, sy - h / 2.0);
        Some((dx * dx + dy * dy) / (w * w / 4.0 + h * h / 4.0))
    };
    type Sample = (usize, usize, Vec<f32>, Vec<f32>, f64, f64);
    let mut samples: Vec<Sample> = Vec::new();
    let mut y = y0;
    while y < y1 {
        let mut x = x0;
        while x < x1 {
            let cover: Vec<(usize, usize)> = (0..n).filter_map(|i| imgs[i].at(x, y).filter(|&k| imgs[i].alpha[k] > 0.999).map(|k| (i, k))).collect();
            for p in 0..cover.len() {
                for q in p + 1..cover.len() {
                    let ((a, ka), (b, kb)) = (cover[p], cover[q]);
                    let ca = imgs[a].px[ka * ch..(ka + 1) * ch].to_vec();
                    let cb = imgs[b].px[kb * ch..(kb + 1) * ch].to_vec();
                    if ca.iter().chain(&cb).any(|v| !(0.02..=0.98).contains(v)) {
                        continue;
                    }
                    // Disagreeing content (parallax, moving objects) says nothing about exposure.
                    if ca.iter().zip(&cb).any(|(x, y)| (x / y).ln().abs() > 1.0) {
                        continue;
                    }
                    let (Some(ra), Some(rb)) = (r2_of(a, x, y), r2_of(b, x, y)) else { continue };
                    samples.push((a, b, ca, cb, ra, rb));
                }
            }
            x += step;
        }
        y += step;
    }
    if samples.len() < 20 {
        return identity;
    }
    // Unknowns: log gain per image (per channel), then a, b (shared, from mean log intensity).
    let sigma_n = 0.05;
    let sigma_g = if gain { 0.15 } else { 1e-4 };
    let mut out = identity.clone();
    let nv = if vignette { 2 } else { 0 };
    // Vignetting is solved on the channel mean, then gains per channel with it fixed.
    let mut ne = Normal::new(n + nv);
    for (a, b, ca, cb, ra, rb) in &samples {
        let la = ca.iter().map(|v| (*v as f64).ln()).sum::<f64>() / ch as f64;
        let lb = cb.iter().map(|v| (*v as f64).ln()).sum::<f64>() / ch as f64;
        let mut row = vec![(*a, 1.0), (*b, -1.0)];
        if vignette {
            row.push((n, ra - rb));
            row.push((n + 1, ra * ra - rb * rb));
        }
        ne.add(&row, la - lb, 1.0 / sigma_n);
    }
    for i in 0..n {
        ne.add(&[(i, 1.0)], 0.0, 1.0 / sigma_g);
    }
    if vignette {
        ne.add(&[(n, 1.0)], 0.0, 0.5);
        ne.add(&[(n + 1, 1.0)], 0.0, 0.5);
    }
    let Some(sol) = ne.solve() else { return out };
    if vignette {
        // Vignetting only darkens; clamp a runaway brightening fit.
        out.vignette = [sol[n].clamp(-2.0, 0.5), sol[n + 1].clamp(-2.0, 2.0)];
    }
    if gain {
        for c in 0..ch {
            let mut ne = Normal::new(n);
            for (a, b, ca, cb, ra, rb) in &samples {
                let va = out.vignette[0] * ra + out.vignette[1] * ra * ra;
                let vb = out.vignette[0] * rb + out.vignette[1] * rb * rb;
                ne.add(&[(*a, 1.0), (*b, -1.0)], ((ca[c] as f64).ln() - va) - ((cb[c] as f64).ln() - vb), 1.0 / sigma_n);
            }
            for i in 0..n {
                ne.add(&[(i, 1.0)], 0.0, 1.0 / sigma_g);
            }
            if let Some(g) = ne.solve() {
                for i in 0..n {
                    out.gains[i][c] = g[i].exp().clamp(0.2, 5.0) as f32;
                }
            }
        }
    }
    out
}

// ---------- seams ----------

/// Seam labels on a `w × h` grid (`-1` = no image): images are added in `order`; each new image
/// takes the part of its overlap with the composite that lies on its side of the cheapest path
/// through the overlap (cost = colour difference).
pub fn seam_labels(w: usize, h: usize, luma: &[Vec<f32>], cover: &[Vec<bool>], order: &[usize]) -> Vec<i32> {
    let mut label = vec![-1i32; w * h];
    for &img in order {
        let cov = &cover[img];
        let mut overlap = vec![false; w * h];
        let (mut sx, mut sy, mut no) = (0.0f64, 0.0f64, 0usize);
        let (mut nx, mut ny, mut nn) = (0.0f64, 0.0f64, 0usize);
        for i in 0..w * h {
            if !cov[i] {
                continue;
            }
            nx += (i % w) as f64;
            ny += (i / w) as f64;
            nn += 1;
            if label[i] >= 0 {
                overlap[i] = true;
                sx += (i % w) as f64;
                sy += (i / w) as f64;
                no += 1;
            } else {
                label[i] = img as i32;
            }
        }
        if no == 0 || nn == no {
            continue;
        }
        let (ox, oy) = (sx / no as f64, sy / no as f64);
        let (cx, cy) = (nx / nn as f64, ny / nn as f64);
        let vertical = (cx - ox).abs() >= (cy - oy).abs();
        // Image on the increasing side along the seam's cross axis?
        let positive = if vertical { cx > ox } else { cy > oy };
        let cost = |i: usize| -> f32 { 0.002 + (luma[img][i] - luma[label[i] as usize][i]).abs() };
        let (lines, len) = if vertical { (h, w) } else { (w, h) };
        let idx = |line: usize, k: usize| if vertical { line * w + k } else { k * w + line };
        // DP over lines (rows for a vertical seam), within the overlap.
        let mut acc = vec![f32::INFINITY; w * h];
        let mut from = vec![usize::MAX; w * h];
        let mut prev_line: Option<usize> = None;
        for line in 0..lines {
            let any = (0..len).any(|k| overlap[idx(line, k)]);
            if !any {
                continue;
            }
            for k in 0..len {
                let i = idx(line, k);
                if !overlap[i] {
                    continue;
                }
                let mut best = (0.0f32, usize::MAX);
                if let Some(pl) = prev_line.filter(|pl| *pl + 1 == line) {
                    best = (f32::INFINITY, usize::MAX);
                    for dk in [-1i64, 0, 1] {
                        let kk = k as i64 + dk;
                        if kk < 0 || kk >= len as i64 {
                            continue;
                        }
                        let j = idx(pl, kk as usize);
                        if acc[j] < best.0 {
                            best = (acc[j], kk as usize);
                        }
                    }
                    if best.1 == usize::MAX {
                        best = (0.0, usize::MAX);
                    }
                }
                acc[i] = best.0 + cost(i);
                from[i] = best.1;
            }
            prev_line = Some(line);
        }
        // Backtrack from the cheapest end; lines not reached keep their own best cell.
        let mut seam = vec![usize::MAX; lines];
        let mut line = lines;
        while line > 0 {
            line -= 1;
            if seam[line] != usize::MAX {
                continue;
            }
            let Some(k) = (0..len).filter(|&k| overlap[idx(line, k)]).min_by(|&a, &b| acc[idx(line, a)].total_cmp(&acc[idx(line, b)])) else { continue };
            let (mut l, mut k) = (line, k);
            loop {
                seam[l] = k;
                let f = from[idx(l, k)];
                if f == usize::MAX || l == 0 {
                    break;
                }
                l -= 1;
                if seam[l] != usize::MAX {
                    break;
                }
                k = f;
            }
        }
        for line in 0..lines {
            let s = seam[line];
            if s == usize::MAX {
                continue;
            }
            for k in 0..len {
                let i = idx(line, k);
                if overlap[i] && ((positive && k > s) || (!positive && k < s)) {
                    label[i] = img as i32;
                }
            }
        }
    }
    label
}

// ---------- multi-band blending ----------

/// Number of pyramid levels used by [`multiband`] for a seam mask of this size.
pub fn blend_levels(w: usize, h: usize) -> usize {
    let mut s = w.min(h);
    let mut n = 1;
    while s > 16 && n < 7 {
        s /= 2;
        n += 1;
    }
    n
}

fn reduce(w: usize, h: usize, ch: usize, src: &[f32]) -> (usize, usize, Vec<f32>) {
    let (nw, nh) = (w.div_ceil(2), h.div_ceil(2));
    let k = [1.0f32, 4.0, 6.0, 4.0, 1.0];
    // Separable 5-tap binomial, then decimate.
    let mut tmp = vec![0.0f32; nw * h * ch];
    for y in 0..h {
        for x in 0..nw {
            for c in 0..ch {
                let mut s = 0.0;
                for (t, kv) in k.iter().enumerate() {
                    let xx = (2 * x as i64 + t as i64 - 2).clamp(0, w as i64 - 1) as usize;
                    s += kv * src[(y * w + xx) * ch + c];
                }
                tmp[(y * nw + x) * ch + c] = s / 16.0;
            }
        }
    }
    let mut out = vec![0.0f32; nw * nh * ch];
    for y in 0..nh {
        for x in 0..nw {
            for c in 0..ch {
                let mut s = 0.0;
                for (t, kv) in k.iter().enumerate() {
                    let yy = (2 * y as i64 + t as i64 - 2).clamp(0, h as i64 - 1) as usize;
                    s += kv * tmp[(yy * nw + x) * ch + c];
                }
                out[(y * nw + x) * ch + c] = s / 16.0;
            }
        }
    }
    (nw, nh, out)
}

fn expand(sw: usize, sh: usize, ch: usize, src: &[f32], w: usize, h: usize) -> Vec<f32> {
    // Bilinear upsampling at half-pixel-aligned positions.
    let mut out = vec![0.0f32; w * h * ch];
    for y in 0..h {
        let fy = ((y as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, (sh - 1) as f32);
        let (y0, ty) = (fy.floor() as usize, fy - fy.floor());
        let y1 = (y0 + 1).min(sh - 1);
        for x in 0..w {
            let fx = ((x as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, (sw - 1) as f32);
            let (x0, tx) = (fx.floor() as usize, fx - fx.floor());
            let x1 = (x0 + 1).min(sw - 1);
            for c in 0..ch {
                let p = |xx: usize, yy: usize| src[(yy * sw + xx) * ch + c];
                let top = p(x0, y0) + (p(x1, y0) - p(x0, y0)) * tx;
                let bot = p(x0, y1) + (p(x1, y1) - p(x0, y1)) * tx;
                out[(y * w + x) * ch + c] = top + (bot - top) * ty;
            }
        }
    }
    out
}

/// Fills pixels with zero `wt` from their surroundings (push-pull), so pyramid levels don't
/// pull in black from outside an image.
fn push_pull(w: usize, h: usize, ch: usize, px: &mut [f32], wt: &[f32]) {
    if w <= 1 && h <= 1 {
        return;
    }
    let (nw, nh) = (w.div_ceil(2), h.div_ceil(2));
    let mut cpx = vec![0.0f32; nw * nh * ch];
    let mut cw = vec![0.0f32; nw * nh];
    for y in 0..h {
        for x in 0..w {
            let (i, j) = (y * w + x, (y / 2) * nw + x / 2);
            let a = wt[i];
            cw[j] += a;
            for c in 0..ch {
                cpx[j * ch + c] += a * px[i * ch + c];
            }
        }
    }
    for j in 0..nw * nh {
        if cw[j] > 0.0 {
            for c in 0..ch {
                cpx[j * ch + c] /= cw[j];
            }
            cw[j] = 1.0;
        }
    }
    if cw.contains(&0.0) && cw.iter().any(|v| *v > 0.0) {
        push_pull(nw, nh, ch, &mut cpx, &cw);
    }
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if wt[i] < 1.0 {
                let j = (y / 2) * nw + x / 2;
                let a = wt[i];
                for c in 0..ch {
                    px[i * ch + c] = a * px[i * ch + c] + (1.0 - a) * cpx[j * ch + c];
                }
            }
        }
    }
}

/// Multi-band blend of `imgs` into a `w × h` panorama (origin 0,0) with one weight map per image
/// over the same grid as its ROI (`weights[i]` has `w_i × h_i` entries). Returns the colour
/// channels (`ch` per pixel) and the total coverage.
pub fn multiband(w: usize, h: usize, imgs: &[RoiImage], weights: &[Vec<f32>], levels: usize) -> (Vec<f32>, Vec<f32>) {
    let ch = imgs.first().map_or(1, |i| i.ch);
    let levels = levels.max(1);
    let align = 1i32 << (levels - 1);
    // Level sizes of the panorama.
    let mut sizes = vec![(w, h)];
    for _ in 1..levels {
        let Some(&(pw, ph)) = sizes.last() else { break };
        sizes.push((pw.div_ceil(2), ph.div_ceil(2)));
    }
    let mut acc: Vec<Vec<f32>> = sizes.iter().map(|&(lw, lh)| vec![0.0; lw * lh * ch]).collect();
    let mut wsum: Vec<Vec<f32>> = sizes.iter().map(|&(lw, lh)| vec![0.0; lw * lh]).collect();
    let margin = align * 4;
    // Each image on its ROI grown by a margin and aligned to the coarsest level's grid.
    let bands = par_map(imgs.len(), |k| {
        let im = &imgs[k];
        let gx0 = ((im.x0 - margin).max(0) / align) * align;
        let gy0 = ((im.y0 - margin).max(0) / align) * align;
        let gx1 = (im.x0 + im.w as i32 + margin).min(w as i32);
        let gy1 = (im.y0 + im.h as i32 + margin).min(h as i32);
        let (gw, gh) = ((gx1 - gx0).max(1) as usize, (gy1 - gy0).max(1) as usize);
        let mut px = vec![0.0f32; gw * gh * ch];
        let mut cov = vec![0.0f32; gw * gh];
        let mut wt = vec![0.0f32; gw * gh];
        for y in 0..im.h {
            for x in 0..im.w {
                let (gx, gy) = (im.x0 + x as i32 - gx0, im.y0 + y as i32 - gy0);
                if gx < 0 || gy < 0 || gx as usize >= gw || gy as usize >= gh {
                    continue;
                }
                let (i, g) = (y * im.w + x, gy as usize * gw + gx as usize);
                cov[g] = im.alpha[i];
                wt[g] = weights[k][i] * im.alpha[i].min(1.0);
                px[g * ch..(g + 1) * ch].copy_from_slice(&im.px[i * ch..(i + 1) * ch]);
            }
        }
        let bin: Vec<f32> = cov.iter().map(|a| if *a > 0.5 { 1.0 } else { 0.0 }).collect();
        push_pull(gw, gh, ch, &mut px, &bin);
        // Laplacian of the colours, Gaussian of the weights.
        let mut lap: Vec<(usize, usize, Vec<f32>)> = Vec::new();
        let mut gau: Vec<Vec<f32>> = Vec::new();
        let (mut cw, mut chh, mut cur, mut cwt) = (gw, gh, px, wt);
        for l in 0..levels {
            if l + 1 == levels {
                lap.push((cw, chh, cur.clone()));
                gau.push(cwt.clone());
                break;
            }
            let (nw, nh, down) = reduce(cw, chh, ch, &cur);
            let up = expand(nw, nh, ch, &down, cw, chh);
            lap.push((cw, chh, cur.iter().zip(&up).map(|(a, b)| a - b).collect()));
            gau.push(cwt.clone());
            let (_, _, wdown) = reduce(cw, chh, 1, &cwt);
            cw = nw;
            chh = nh;
            cur = down;
            cwt = wdown;
        }
        (gx0, gy0, lap, gau)
    });
    for (gx0, gy0, lap, gau) in &bands {
        for (l, ((lw, lh, band), g)) in lap.iter().zip(gau).enumerate() {
            let (ox, oy) = ((gx0 >> l) as usize, (gy0 >> l) as usize);
            let (pw, ph) = sizes[l];
            for y in 0..*lh {
                let py = oy + y;
                if py >= ph {
                    break;
                }
                for x in 0..*lw {
                    let px = ox + x;
                    if px >= pw {
                        break;
                    }
                    let (i, p) = (y * lw + x, py * pw + px);
                    let k = g[i];
                    if k <= 0.0 {
                        continue;
                    }
                    wsum[l][p] += k;
                    for c in 0..ch {
                        acc[l][p * ch + c] += k * band[i * ch + c];
                    }
                }
            }
        }
    }
    for l in 0..levels {
        let (pw, ph) = sizes[l];
        for p in 0..pw * ph {
            let k = wsum[l][p];
            for c in 0..ch {
                acc[l][p * ch + c] = if k > 1e-8 { acc[l][p * ch + c] / k } else { 0.0 };
            }
        }
    }
    // Collapse.
    // `levels ≥ 1`, so there is always a coarsest level.
    let Some(mut cur) = acc.pop() else { return (vec![0.0; w * h * ch], vec![0.0; w * h]) };
    for l in (0..levels - 1).rev() {
        let (pw, ph) = sizes[l];
        let (sw, sh) = sizes[l + 1];
        let up = expand(sw, sh, ch, &cur, pw, ph);
        cur = acc[l].iter().zip(&up).map(|(a, b)| a + b).collect();
    }
    // Coverage: union of the images.
    let mut cov = vec![0.0f32; w * h];
    for im in imgs {
        for y in 0..im.h {
            for x in 0..im.w {
                let (px, py) = (im.x0 + x as i32, im.y0 + y as i32);
                if px < 0 || py < 0 || px as usize >= w || py as usize >= h {
                    continue;
                }
                let p = py as usize * w + px as usize;
                cov[p] = cov[p].max(im.alpha[y * im.w + x]);
            }
        }
    }
    (cur, cov)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smooth random texture: blobs over a gradient (bounded detail so Harris finds corners).
    pub(crate) fn scene(w: usize, h: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng(seed);
        let mut img: Vec<f32> = (0..w * h).map(|i| 0.25 + 0.2 * ((i % w) as f32 / w as f32)).collect();
        for _ in 0..(w * h / 900).max(40) {
            let cx = (rng.next() % w as u64) as f64;
            let cy = (rng.next() % h as u64) as f64;
            let r = 3.0 + (rng.next() % 9) as f64;
            let v = (rng.next() % 1000) as f32 / 1000.0;
            let sq = rng.next().is_multiple_of(2);
            let (x0, x1) = ((cx - r).max(0.0) as usize, ((cx + r) as usize + 1).min(w));
            let (y0, y1) = ((cy - r).max(0.0) as usize, ((cy + r) as usize + 1).min(h));
            for y in y0..y1 {
                for x in x0..x1 {
                    let (dx, dy) = (x as f64 - cx, y as f64 - cy);
                    if if sq { dx.abs() < r && dy.abs() < r * 0.6 } else { dx.hypot(dy) < r } {
                        img[y * w + x] = v;
                    }
                }
            }
        }
        img
    }

    fn sample(img: &[f32], w: usize, h: usize, x: f64, y: f64) -> Option<f32> {
        if x < 0.0 || y < 0.0 || x > (w - 1) as f64 || y > (h - 1) as f64 {
            return None;
        }
        Some(crate::photo_util::bilinear(img, w, h, 1, 0, x as f32, y as f32))
    }

    /// Crops `tw × th` views of a scene through `map` (view px → scene px).
    fn view(scene: &[f32], sw: usize, sh: usize, tw: usize, th: usize, map: impl Fn(f64, f64) -> (f64, f64)) -> Vec<f32> {
        (0..tw * th)
            .map(|i| {
                let (x, y) = map((i % tw) as f64, (i / tw) as f64);
                sample(scene, sw, sh, x, y).unwrap_or(0.0)
            })
            .collect()
    }

    #[test]
    fn recovers_translations_and_homographies() {
        let (sw, sh) = (640, 300);
        let sc = scene(sw, sh, 3);
        let (tw, th) = (260, 220);
        // Three overlapping crops at known offsets.
        let offs = [(20.0, 30.0), (190.0, 40.0), (360.0, 25.0)];
        let imgs: Vec<Vec<f32>> = offs.iter().map(|&(ox, oy)| view(&sc, sw, sh, tw, th, |x, y| (x + ox, y + oy))).collect();
        let feats: Vec<ImageFeatures> = imgs.iter().map(|im| detect(tw, th, im, None, 700)).collect();
        let matches = match_pairs(&feats);
        assert!(matches.len() >= 2, "{}", matches.len());
        for layout in [Layout::Reposition, Layout::Collage, Layout::Perspective] {
            let a = align(&feats, &matches, &AlignOptions { layout, focal: None, reference: Some(1), geometric: false }).unwrap();
            assert_eq!(a.reference, 1);
            for (i, p) in a.placements.iter().enumerate() {
                let p = p.as_ref().unwrap();
                // A point at image px (50, 60) is scene (50+ox, 60+oy); relative to image 1.
                let (u, v) = p.forward(50.0, 60.0);
                let (eu, ev) = (50.0 + offs[i].0 - offs[1].0, 60.0 + offs[i].1 - offs[1].1);
                assert!((u - eu).abs() < 1.0 && (v - ev).abs() < 1.0, "{layout:?} image {i}: ({u},{v}) vs ({eu},{ev})");
            }
            assert!(a.rms < 1.0, "{layout:?} rms {}", a.rms);
        }
        // A projective view is recovered by the perspective layout.
        let hq = Homography([1.05, 0.03, 180.0, -0.02, 1.0, 35.0, 0.0001, -0.00005, 1.0]);
        let im2 = view(&sc, sw, sh, tw, th, |x, y| hq.apply(x, y));
        let feats2 = vec![feats[0].clone(), detect(tw, th, &im2, None, 700)];
        let m2 = match_pairs(&feats2);
        let a = align(&feats2, &m2, &AlignOptions { layout: Layout::Perspective, focal: None, reference: Some(0), geometric: false }).unwrap();
        let p = a.placements[1].as_ref().unwrap();
        // Points inside the overlap (extrapolation beyond it is loosely constrained).
        for (x, y) in [(20.0, 30.0), (70.0, 180.0), (40.0, 100.0)] {
            let (sx, sy) = hq.apply(x, y);
            let (u, v) = p.forward(x, y);
            assert!((u - (sx - 20.0)).abs() < 1.5 && (v - (sy - 30.0)).abs() < 1.5, "({u},{v}) vs ({},{})", sx - 20.0, sy - 30.0);
        }
    }

    #[test]
    fn rotating_camera_focal_and_cylindrical_layout() {
        // Views of a cylinder-mapped scene from a camera panning by 25° with f = 300 px.
        let f = 300.0f64;
        let (sw, sh) = (1400, 500);
        let sc = scene(sw, sh, 9);
        let (tw, th) = (320, 240);
        let pans = [-25f64, 0.0, 25.0];
        let imgs: Vec<Vec<f32>> = pans
            .iter()
            .map(|&deg| {
                view(&sc, sw, sh, tw, th, |x, y| {
                    let (dx, dy) = (x - tw as f64 / 2.0, y - th as f64 / 2.0);
                    let th_ = dx.atan2(f) + deg.to_radians();
                    let hh = dy / dx.hypot(f);
                    (sw as f64 / 2.0 + f * th_, sh as f64 / 2.0 + f * hh)
                })
            })
            .collect();
        let feats: Vec<ImageFeatures> = imgs.iter().map(|im| detect(tw, th, im, None, 700)).collect();
        let matches = match_pairs(&feats);
        assert_eq!(matches.iter().filter(|m| m.b == m.a + 1).count(), 2);
        let est: Vec<f64> = matches.iter().filter_map(|m| focal_from_homography(&m.homography, [160.0, 120.0], [160.0, 120.0])).collect();
        assert!(est.iter().any(|e| (e - f).abs() / f < 0.15), "focal estimates {est:?}");
        let a = align(&feats, &matches, &AlignOptions { layout: Layout::Cylindrical, focal: None, reference: Some(1), geometric: false }).unwrap();
        assert!((a.focal - f).abs() / f < 0.1, "focal {}", a.focal);
        let p = a.placements[2].as_ref().unwrap();
        // The centre of the right image sits 25° along the cylinder.
        let (u, _) = p.forward(160.0, 120.0);
        let expect = 160.0 + a.focal * 25f64.to_radians();
        assert!((u - expect).abs() < 4.0, "{u} vs {expect}");
        assert!(a.rms < 1.5, "rms {}", a.rms);
        // Round trip through the placement.
        let (u, v) = p.forward(40.0, 200.0);
        let (x, y) = p.inverse(u, v).unwrap();
        assert!((x - 40.0).abs() < 1e-6 && (y - 200.0).abs() < 1e-6);
    }

    #[test]
    fn placement_round_trips_all_projections() {
        for projection in [Projection::Plane, Projection::Cylinder, Projection::Sphere] {
            let p =
                Placement { projection, focal: 400.0, center: [200.0, 150.0], k1: 0.05, h: Homography([0.98, -0.1, 30.0, 0.1, 0.98, -12.0, 0.0, 0.0, 1.0]) };
            for (x, y) in [(0.0, 0.0), (390.0, 20.0), (123.0, 280.0)] {
                let (u, v) = p.forward(x, y);
                let (bx, by) = p.inverse(u, v).unwrap();
                assert!((bx - x).abs() < 1e-6 && (by - y).abs() < 1e-6, "{projection:?}");
            }
            let s = p.scaled(2.0);
            let (u, v) = p.forward(100.0, 50.0);
            let (u2, v2) = s.forward(200.0, 100.0);
            assert!((u2 - 2.0 * u).abs() < 1e-6 && (v2 - 2.0 * v).abs() < 1e-6);
        }
    }

    #[test]
    fn photometric_recovers_gain_and_vignette() {
        // Two half-overlapping views of a flat-ish scene: the second one 30 % brighter.
        let (w, h) = (200usize, 100usize);
        let base = |x: i32, y: i32| 0.3 + 0.2 * ((x as f32 * 0.05).sin() * (y as f32 * 0.07).cos()).abs();
        let mut imgs = Vec::new();
        let mut pls = Vec::new();
        for (k, (x0, g)) in [(0, 1.0f32), (100, 1.3f32)].into_iter().enumerate() {
            let mut px = Vec::new();
            for y in 0..h as i32 {
                for x in 0..w as i32 {
                    px.push(base(x + x0, y) * g);
                }
            }
            imgs.push(RoiImage { x0, y0: 0, w, h, ch: 1, px, alpha: vec![1.0; w * h] });
            pls.push(Placement {
                projection: Projection::Plane,
                focal: 200.0,
                center: [100.0, 50.0],
                k1: 0.0,
                h: Homography([1.0, 0.0, 100.0 + x0 as f64, 0.0, 1.0, 50.0, 0.0, 0.0, 1.0]),
            });
            let _ = k;
        }
        let ph = photometric(&imgs, Some((&pls, &[(w, h), (w, h)])), true, false);
        let ratio = ph.gains[1][0] / ph.gains[0][0];
        assert!((ratio - 1.3).abs() < 0.03, "{ph:?}");
        // With vignetting on, the flat scene (no falloff) gives a near-zero model.
        let pv = photometric(&imgs, Some((&pls, &[(w, h), (w, h)])), true, true);
        assert!(pv.vignette[0].abs() < 0.1, "{pv:?}");
    }

    #[test]
    fn seams_avoid_disagreement_and_multiband_is_seamless() {
        // Two images over a 120×40 strip, overlapping in 40..80; the second has a bright
        // "moving object" at x 50..58 that the first doesn't.
        let (w, h) = (120usize, 40usize);
        let mut luma = vec![vec![0.5f32; w * h], vec![0.5f32; w * h]];
        let cover: Vec<Vec<bool>> = vec![(0..w * h).map(|i| i % w < 80).collect(), (0..w * h).map(|i| i % w >= 40).collect()];
        for y in 0..h {
            for x in 50..58 {
                luma[1][y * w + x] = 1.0;
            }
        }
        let labels = seam_labels(w, h, &luma, &cover, &[0, 1]);
        // The object column belongs to image 0 (the seam passes to its right), or entirely to 1.
        for y in 0..h {
            let owner: Vec<i32> = (50..58).map(|x| labels[y * w + x]).collect();
            assert!(owner.iter().all(|o| *o == owner[0]), "row {y} split the object: {owner:?}");
        }
        assert_eq!(labels[5], 0);
        assert_eq!(labels[w - 1], 1);
        // Multi-band blend of two flat images at different levels joins smoothly.
        let mk = |x0: i32, v: f32| RoiImage { x0, y0: 0, w: 80, h, ch: 1, px: vec![v; 80 * h], alpha: vec![1.0; 80 * h] };
        let imgs = [mk(0, 0.3), mk(40, 0.7)];
        let wts: Vec<Vec<f32>> = (0..2)
            .map(|k| {
                (0..80 * h)
                    .map(|i| {
                        let x = (i % 80) as i32 + imgs[k].x0;
                        if (x < 60) == (k == 0) { 1.0 } else { 0.0 }
                    })
                    .collect()
            })
            .collect();
        let (out, cov) = multiband(w, h, &imgs, &wts, 4);
        let row: Vec<f32> = (0..w).map(|x| out[20 * w + x]).collect();
        assert!((row[2] - 0.3).abs() < 0.02 && (row[w - 3] - 0.7).abs() < 0.02, "{row:?}");
        let max_jump = row.windows(2).map(|p| (p[1] - p[0]).abs()).fold(0.0f32, f32::max);
        assert!(max_jump < 0.1, "max step {max_jump}");
        assert!(cov.iter().all(|c| *c == 1.0));
    }
}
