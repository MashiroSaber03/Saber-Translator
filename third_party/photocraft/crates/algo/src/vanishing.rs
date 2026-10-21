//! Filter › Vanishing Point: perspective planes, pasting onto a plane and cloning in perspective.
//!
//! A plane is a quad the user aligns with a flat surface in the photo. Its two vanishing points
//! (where opposite edges meet) give the camera's focal length for a centred principal point,
//! `f² = −(v₁ − c)·(v₂ − c)` (R. Hartley, A. Zisserman, *Multiple View Geometry*, 2nd ed., §8.8:
//! orthogonal vanishing directions), which lifts the quad to 3-D: each corner is back-projected
//! onto the plane through the vanishing directions' normal. That gives the plane's metric aspect
//! (so pasted images keep their proportions) and lets perpendicular planes be torn off an edge
//! by rotating about it in 3-D and projecting back.
//!
//! Pasting maps the image's rectangle into the plane's metric coordinates and warps it with the
//! composed homography, clipped to the plane. Clone stamping copies, for each destination dab,
//! the pixels at the same metric offset from the source point, so texture shrinks with distance
//! as it does in the photo.

use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde::{Deserialize, Serialize};

use crate::transform::{Homography, Interp, warp_surface};

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn mul(a: V3, k: f64) -> V3 {
    [a[0] * k, a[1] * k, a[2] * k]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: V3) -> V3 {
    mul(a, 1.0 / dot(a, a).sqrt().max(1e-300))
}

/// Intersection of lines (a, b) and (c, d) in homogeneous coordinates.
fn meet(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> V3 {
    let l1 = cross([a[0], a[1], 1.0], [b[0], b[1], 1.0]);
    let l2 = cross([c[0], c[1], 1.0], [d[0], d[1], 1.0]);
    cross(l1, l2)
}

/// A perspective plane: corners in image px, clockwise from the plane's top-left.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VpPlane {
    pub corners: [[f64; 2]; 4],
}

/// Plane edges for tearing off a perpendicular plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Edge {
    Top,
    Right,
    Bottom,
    Left,
}

/// The camera and a plane lifted to 3-D.
#[derive(Clone, Debug)]
pub struct Lifted {
    pub focal: f64,
    pub center: [f64; 2],
    /// 3-D corners (camera coordinates, plane at unit distance along its normal).
    pub pts: [V3; 4],
}

impl VpPlane {
    /// Unit square (u, v) → image.
    pub fn homography(&self) -> Option<Homography> {
        Homography::rect_to_quad([0.0, 0.0, 1.0, 1.0], self.corners)
    }

    /// Focal length (px) implied by the plane's two vanishing points, if both are finite and
    /// consistent with a camera centred at `c`.
    pub fn focal(&self, c: [f64; 2]) -> Option<f64> {
        let q = &self.corners;
        let v1 = meet(q[0], q[1], q[3], q[2]);
        let v2 = meet(q[0], q[3], q[1], q[2]);
        if v1[2].abs() < 1e-9 || v2[2].abs() < 1e-9 {
            return None;
        }
        let (a, b) = ([v1[0] / v1[2] - c[0], v1[1] / v1[2] - c[1]], [v2[0] / v2[2] - c[0], v2[1] / v2[2] - c[1]]);
        let f2 = -(a[0] * b[0] + a[1] * b[1]);
        (f2 > 1.0).then(|| f2.sqrt())
    }

    /// Lifts the plane to 3-D with focal `f` (camera at the origin looking along +z).
    pub fn lift(&self, f: f64, c: [f64; 2]) -> Lifted {
        let ray = |p: [f64; 2]| [(p[0] - c[0]) / f, (p[1] - c[1]) / f, 1.0];
        let q = &self.corners;
        // Vanishing directions (or the edge directions when they're parallel in the image).
        let dir = |a: [f64; 2], b: [f64; 2], cc: [f64; 2], d: [f64; 2]| -> V3 {
            let v = meet(a, b, cc, d);
            if v[2].abs() < 1e-9 * (v[0].abs() + v[1].abs()).max(1.0) { norm([v[0], v[1], 0.0]) } else { norm(ray([v[0] / v[2], v[1] / v[2]])) }
        };
        let d1 = dir(q[0], q[1], q[3], q[2]);
        let d2 = dir(q[0], q[3], q[1], q[2]);
        let mut n = norm(cross(d1, d2));
        if n[2] < 0.0 {
            n = mul(n, -1.0);
        }
        let pts = q.map(|p| {
            let r = ray(p);
            let lam = 1.0 / dot(n, r).abs().max(1e-9);
            mul(r, lam)
        });
        Lifted { focal: f, center: c, pts }
    }

    /// Turn of each corner (`sin` of the angle between consecutive edges, positive when the
    /// corners run clockwise on screen); `None` when an edge is shorter than a pixel or a corner
    /// is not finite.
    fn turns(&self) -> Option<[f64; 4]> {
        let q = &self.corners;
        if q.iter().flatten().any(|v| !v.is_finite() || v.abs() > 1e9) {
            return None;
        }
        let e = |i: usize| [q[(i + 1) % 4][0] - q[i][0], q[(i + 1) % 4][1] - q[i][1]];
        let mut out = [0.0; 4];
        for (i, t) in out.iter_mut().enumerate() {
            let (a, b) = (e(i), e((i + 1) % 4));
            let (la, lb) = (a[0].hypot(a[1]), b[0].hypot(b[1]));
            if la < 1.0 || lb < 1.0 {
                return None;
            }
            *t = (a[0] * b[1] - a[1] * b[0]) / (la * lb);
        }
        Some(out)
    }

    /// How usable the plane is, the way Photoshop colours its grid: [`PlaneCheck::Invalid`] for
    /// a quad that isn't convex (self-intersecting, a corner pushed inside, a near-straight or
    /// near-zero corner), [`PlaneCheck::Questionable`] for a convex quad whose perspective can't
    /// come from a camera centred at `center` (or with a corner sharper than 10°), else
    /// [`PlaneCheck::Valid`].
    pub fn check(&self, center: [f64; 2]) -> PlaneCheck {
        let Some(t) = self.turns() else { return PlaneCheck::Invalid };
        let same_sign = t.iter().all(|v| *v > 0.0) || t.iter().all(|v| *v < 0.0);
        let min = t.iter().fold(f64::MAX, |m, v| m.min(v.abs()));
        if !same_sign || min < 1f64.to_radians().sin() || self.homography().is_none() {
            return PlaneCheck::Invalid;
        }
        if min < 10f64.to_radians().sin() {
            return PlaneCheck::Questionable;
        }
        // Both vanishing points finite but no real focal length fits them.
        let q = &self.corners;
        let (v1, v2) = (meet(q[0], q[1], q[3], q[2]), meet(q[0], q[3], q[1], q[2]));
        let finite = |v: V3| v[2].abs() > 1e-9 * (v[0].abs() + v[1].abs()).max(1.0);
        if finite(v1) && finite(v2) && self.focal(center).is_none() {
            return PlaneCheck::Questionable;
        }
        PlaneCheck::Valid
    }

    /// The same quad with its corners clockwise on screen, starting at the top-left one
    /// (the order [`paste`] and [`Scene::tear_off`] expect), however they were clicked.
    pub fn normalized(&self) -> VpPlane {
        let mut q = self.corners;
        if self.turns().is_some_and(|t| t.iter().sum::<f64>() < 0.0) {
            q = [q[0], q[3], q[2], q[1]];
        }
        let first = (0..4).min_by(|&a, &b| (q[a][0] + q[a][1]).total_cmp(&(q[b][0] + q[b][1]))).unwrap_or(0);
        VpPlane { corners: [q[first], q[(first + 1) % 4], q[(first + 2) % 4], q[(first + 3) % 4]] }
    }
}

/// Plane validity ([`VpPlane::check`]): Photoshop draws a valid plane's grid blue, a doubtful
/// one yellow and an invalid one red.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PlaneCheck {
    Valid,
    Questionable,
    Invalid,
}

impl Lifted {
    pub fn project(&self, x: V3) -> Option<[f64; 2]> {
        (x[2] > 1e-9).then(|| [self.center[0] + self.focal * x[0] / x[2], self.center[1] + self.focal * x[1] / x[2]])
    }
    /// Metric side lengths (u along top edge, v along left edge).
    pub fn lengths(&self) -> (f64, f64) {
        let l = |a: V3, b: V3| dot(sub(a, b), sub(a, b)).sqrt();
        ((l(self.pts[0], self.pts[1]) + l(self.pts[3], self.pts[2])) / 2.0, (l(self.pts[0], self.pts[3]) + l(self.pts[1], self.pts[2])) / 2.0)
    }
}

/// A scene: planes plus the shared focal length.
#[derive(Clone, Debug)]
pub struct Scene {
    pub planes: Vec<VpPlane>,
    pub focal: f64,
    pub center: [f64; 2],
}

impl Scene {
    /// Focal length from the first plane (or `fallback` px when it is fronto-parallel).
    pub fn new(first: VpPlane, center: [f64; 2], focal: Option<f64>, fallback: f64) -> Scene {
        let f = focal.filter(|f| *f > 1.0).or_else(|| first.focal(center)).unwrap_or(fallback);
        Scene { planes: vec![first], focal: f, center }
    }

    pub fn lifted(&self, i: usize) -> Lifted {
        self.planes[i].lift(self.focal, self.center)
    }

    /// Tears a plane off edge `edge` of plane `from`, turned by `angle` degrees about that edge
    /// (90 = perpendicular), extending `depth` times the parent's adjacent side. Returns its index.
    pub fn tear_off(&mut self, from: usize, edge: Edge, angle: f64, depth: f64) -> Option<usize> {
        let lf = self.lifted(from);
        let p = lf.pts;
        // Edge endpoints (a → b, clockwise) and the parent's inward direction.
        let (ia, ib, ic) = match edge {
            Edge::Top => (0, 1, 3),
            Edge::Right => (1, 2, 0),
            Edge::Bottom => (2, 3, 1),
            Edge::Left => (3, 0, 2),
        };
        let (a, b) = (p[ia], p[ib]);
        let axis = norm(sub(b, a));
        let inward = sub(p[ic], p[ia]);
        let len = dot(inward, inward).sqrt() * depth.max(0.05);
        // The new plane continues outward (−inward), turned about the edge by 180° − angle; of
        // the two turning directions pick the one that faces the camera with the parent in front
        // of it (a room corner, as Photoshop tears planes off).
        let out = mul(norm(inward), -1.0);
        let th = (180.0 - angle).to_radians();
        let mut best: Option<(f64, [[f64; 2]; 4])> = None;
        for sgn in [1.0, -1.0] {
            let (s, c) = (sgn * th).sin_cos();
            let rot = add(add(mul(out, c), mul(cross(axis, out), s)), mul(axis, dot(axis, out) * (1.0 - c)));
            let (a2, b2) = (add(a, mul(rot, len)), add(b, mul(rot, len)));
            let (Some(pa), Some(pb), Some(pa2), Some(pb2)) = (lf.project(a), lf.project(b), lf.project(a2), lf.project(b2)) else { continue };
            let mut n2 = norm(cross(sub(b, a), sub(a2, a)));
            if dot(n2, a) > 0.0 {
                n2 = mul(n2, -1.0);
            }
            let score = dot(n2, inward);
            if best.as_ref().is_none_or(|(sc, _)| score > *sc) {
                best = Some((score, [pa, pb, pa2, pb2]));
            }
        }
        let (_, [pa, pb, pa2, pb2]) = best?;
        // Keep corners clockwise from the new plane's top-left (shared edge first or last).
        let corners = match edge {
            Edge::Top => [pa2, pb2, pb, pa],
            Edge::Right => [pa, pa2, pb2, pb],
            Edge::Bottom => [pb, pa, pa2, pb2],
            Edge::Left => [pb2, pb, pa, pa2],
        };
        self.planes.push(VpPlane { corners });
        Some(self.planes.len() - 1)
    }

    /// Plane-metric frame of plane `i`: (homography unit square → image, metric width, height).
    pub fn frame(&self, i: usize) -> Option<(Homography, f64, f64)> {
        let h = self.planes[i].homography()?;
        let (lu, lv) = self.lifted(i).lengths();
        Some((h, lu, lv))
    }

    /// The plane whose quad contains image point `p`.
    pub fn plane_at(&self, p: [f64; 2]) -> Option<usize> {
        (0..self.planes.len()).rev().find(|&i| {
            self.planes[i].homography().and_then(|h| h.inverse()).is_some_and(|inv| {
                let (u, v) = inv.apply(p[0], p[1]);
                (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v)
            })
        })
    }
}

/// Coverage mask (0/1 surface values) of a plane quad over `area`.
fn quad_mask(h: &Homography, area: Rect) -> Vec<f32> {
    let Some(inv) = h.inverse() else { return vec![0.0; (area.width() * area.height()) as usize] };
    let w = area.width() as usize;
    (0..(area.width() * area.height()) as usize)
        .map(|i| {
            let (x, y) = (area.x0 as f64 + (i % w) as f64 + 0.5, area.y0 as f64 + (i / w) as f64 + 0.5);
            let (u, v) = inv.apply(x, y);
            if (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v) { 1.0 } else { 0.0 }
        })
        .collect()
}

/// Pastes `src` (content in `src_rect`) onto plane `i` with its top-left at plane coordinates
/// `at` (0..1) and its width `width` (fraction of the plane's width), aspect preserved in the
/// plane's metric. Returns the warped image clipped to the plane (transparent elsewhere).
pub fn paste(scene: &Scene, i: usize, src: &Surface, src_rect: Rect, at: [f64; 2], width: f64) -> Result<Option<Surface>, photocraft_raster::AllocationError> {
    let Some((h, lu, lv)) = scene.frame(i) else { return Ok(None) };
    let (sw, sh) = (src_rect.width() as f64, src_rect.height() as f64);
    if sw <= 0.0 || sh <= 0.0 {
        return Ok(None);
    }
    // Source px → plane (u, v): u spans `width`, v keeps the metric aspect.
    let su = width / sw;
    let sv = width * lu / lv / sw;
    let to_plane = Homography([su, 0.0, at[0] - src_rect.x0 as f64 * su, 0.0, sv, at[1] - src_rect.y0 as f64 * sv, 0.0, 0.0, 1.0]);
    let full = h.mul(&to_plane);
    let mut out = warp_surface(src, src_rect, &full, Interp::Bicubic)?;
    // Clip to the plane.
    let b = out.content_bounds();
    if b.is_empty() {
        return Ok(Some(out));
    }
    let mask = quad_mask(&h, b);
    let fmt = out.format();
    let n = fmt.channels();
    let mut data = out.read_region(b);
    for (px, m) in data.chunks_exact_mut(n).zip(&mask) {
        px[n - 1] *= m;
    }
    out.write_region(b, &data);
    out.prune();
    Ok(Some(out))
}

/// Perspective clone stamp: one stroke of dabs at image points `points` on the destination
/// plane, copying from `source` (image px) with the offset fixed in plane-metric units at the
/// first dab. `size` is the brush diameter in px at the first dab; `hardness` 0..=1.
pub fn clone_stroke(scene: &Scene, surf: &mut Surface, source: [f64; 2], points: &[[f64; 2]], size: f64, hardness: f64, opacity: f32) -> usize {
    let Some(first) = points.first() else { return 0 };
    let (Some(ps), Some(pd)) = (scene.plane_at(source), scene.plane_at(*first)) else { return 0 };
    let (Some((hs, lus, lvs)), Some((hd, lud, lvd))) = (scene.frame(ps), scene.frame(pd)) else { return 0 };
    let (Some(hs_inv), Some(hd_inv)) = (hs.inverse(), hd.inverse()) else { return 0 };
    let metric = |inv: &Homography, lu: f64, lv: f64, p: [f64; 2]| {
        let (u, v) = inv.apply(p[0], p[1]);
        [u * lu, v * lv]
    };
    let ms = metric(&hs_inv, lus, lvs, source);
    let md0 = metric(&hd_inv, lud, lvd, *first);
    let offset = [ms[0] - md0[0], ms[1] - md0[1]];
    // Brush radius in metric units (from the image size at the first dab).
    let probe = metric(&hd_inv, lud, lvd, [first[0] + size / 2.0, first[1]]);
    let r_m = ((probe[0] - md0[0]).hypot(probe[1] - md0[1])).max(1e-9);
    let src_copy = surf.clone();
    let fmt = surf.format();
    let n = fmt.channels();
    let mut dabs = 0;
    for p in points {
        let mc = metric(&hd_inv, lud, lvd, *p);
        // Image-space footprint: map a metric circle's bounding points.
        let mut bb = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
        for k in 0..16 {
            let a = k as f64 / 16.0 * std::f64::consts::TAU;
            let (u, v) = ((mc[0] + r_m * a.cos()) / lud, (mc[1] + r_m * a.sin()) / lvd);
            let (x, y) = hd.apply(u, v);
            bb = [bb[0].min(x), bb[1].min(y), bb[2].max(x), bb[3].max(y)];
        }
        let area = Rect::new(bb[0].floor() as i32, bb[1].floor() as i32, (bb[2].ceil() as i32).saturating_add(1), (bb[3].ceil() as i32).saturating_add(1));
        if area.is_empty() || area.width() > 4096 || area.height() > 4096 {
            continue;
        }
        let mut data = surf.read_region(area);
        let w = area.width() as usize;
        let mut changed = false;
        for (i, px) in data.chunks_exact_mut(n).enumerate() {
            let (x, y) = (area.x0 as f64 + (i % w) as f64 + 0.5, area.y0 as f64 + (i / w) as f64 + 0.5);
            let m = metric(&hd_inv, lud, lvd, [x, y]);
            let d = (m[0] - mc[0]).hypot(m[1] - mc[1]) / r_m;
            if d >= 1.0 {
                continue;
            }
            let hard = hardness.clamp(0.0, 0.99);
            let fall = if d <= hard { 1.0 } else { 1.0 - (d - hard) / (1.0 - hard) };
            let k = (fall * fall * (3.0 - 2.0 * fall)) as f32 * opacity;
            // Source pixel at the same metric offset.
            let sm = [m[0] + offset[0], m[1] + offset[1]];
            let (sx, sy) = hs.apply(sm[0] / lus, sm[1] / lvs);
            let s = sample(&src_copy, sx - 0.5, sy - 0.5, n);
            for c in 0..n {
                px[c] += (s[c] - px[c]) * k;
            }
            changed = true;
        }
        if changed {
            surf.write_region(area, &data);
            dabs += 1;
        }
    }
    dabs
}

fn sample(s: &Surface, x: f64, y: f64, n: usize) -> Vec<f32> {
    let (x0, y0) = (x.floor() as i32, y.floor() as i32);
    let (tx, ty) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);
    let r = s.read_region(Rect::new(x0, y0, x0 + 2, y0 + 2));
    (0..n).map(|c| (r[c] * (1.0 - tx) + r[n + c] * tx) * (1.0 - ty) + (r[2 * n + c] * (1.0 - tx) + r[3 * n + c] * tx) * ty).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::PixelFormat;

    /// A ground plane seen by a camera with f = 500 px: projects 3-D points.
    fn project(f: f64, c: [f64; 2], p: V3) -> [f64; 2] {
        [c[0] + f * p[0] / p[2], c[1] + f * p[1] / p[2]]
    }

    fn floor_quad(f: f64, c: [f64; 2]) -> VpPlane {
        // A 2 × 1 rectangle on a tilted plane, rotated about the vertical axis too.
        let (rx, ry) = (0.6f64, 0.4f64);
        let rot = |p: V3| {
            let (s, co) = rx.sin_cos();
            let q = [p[0], co * p[1] - s * p[2], s * p[1] + co * p[2]];
            let (s2, c2) = ry.sin_cos();
            [c2 * q[0] + s2 * q[2], q[1], -s2 * q[0] + c2 * q[2]]
        };
        let corners = [[-1.0, -0.5, 0.0], [1.0, -0.5, 0.0], [1.0, 0.5, 0.0], [-1.0, 0.5, 0.0]].map(|p: V3| project(f, c, add(rot(p), [0.0, 0.0, 4.0])));
        VpPlane { corners }
    }

    #[test]
    fn focal_and_metric_aspect_from_one_plane() {
        let c = [400.0, 300.0];
        let pl = floor_quad(500.0, c);
        let f = pl.focal(c).unwrap();
        assert!((f - 500.0).abs() < 1.0, "{f}");
        let (lu, lv) = pl.lift(f, c).lengths();
        assert!((lu / lv - 2.0).abs() < 0.01, "aspect {}", lu / lv);
    }

    #[test]
    fn plane_check_and_corner_order() {
        let c = [400.0, 300.0];
        assert_eq!(floor_quad(500.0, c).check(c), PlaneCheck::Valid);
        let square = VpPlane { corners: [[10.0, 10.0], [110.0, 10.0], [110.0, 110.0], [10.0, 110.0]] };
        assert_eq!(square.check(c), PlaneCheck::Valid, "fronto-parallel");
        let bowtie = VpPlane { corners: [[10.0, 10.0], [110.0, 110.0], [110.0, 10.0], [10.0, 110.0]] };
        assert_eq!(bowtie.check(c), PlaneCheck::Invalid);
        let dent = VpPlane { corners: [[10.0, 10.0], [110.0, 10.0], [40.0, 40.0], [10.0, 110.0]] };
        assert_eq!(dent.check(c), PlaneCheck::Invalid, "concave");
        let sliver = VpPlane { corners: [[10.0, 10.0], [300.0, 10.0], [300.0, 40.0], [200.0, 40.0]] };
        assert_eq!(sliver.check(c), PlaneCheck::Questionable, "a corner sharper than 10°");
        let nan = VpPlane { corners: [[f64::NAN, 10.0], [110.0, 10.0], [110.0, 110.0], [10.0, 110.0]] };
        assert_eq!(nan.check(c), PlaneCheck::Invalid);
        // Clicked counter-clockwise from the bottom-right: clockwise from the top-left.
        let ccw = VpPlane { corners: [[110.0, 110.0], [110.0, 10.0], [10.0, 10.0], [10.0, 110.0]] };
        assert_eq!(ccw.normalized(), square);
    }

    #[test]
    fn tear_off_is_perpendicular_and_shares_the_edge() {
        let c = [400.0, 300.0];
        let mut sc = Scene::new(floor_quad(500.0, c), c, None, 800.0);
        let k = sc.tear_off(0, Edge::Top, 90.0, 1.0).unwrap();
        let p = &sc.planes[k];
        // Shared edge.
        assert!((p.corners[3][0] - sc.planes[0].corners[0][0]).abs() < 1e-6 && (p.corners[2][0] - sc.planes[0].corners[1][0]).abs() < 1e-6);
        // In 3-D the new plane is perpendicular.
        let (a, b) = (sc.lifted(0), sc.lifted(k));
        let n1 = norm(cross(sub(a.pts[1], a.pts[0]), sub(a.pts[3], a.pts[0])));
        let n2 = norm(cross(sub(b.pts[1], b.pts[0]), sub(b.pts[3], b.pts[0])));
        assert!(dot(n1, n2).abs() < 0.02, "{}", dot(n1, n2));
    }

    #[test]
    fn paste_and_clone_in_perspective() {
        let c = [400.0, 300.0];
        let sc = Scene::new(floor_quad(500.0, c), c, None, 800.0);
        let mut img = Surface::new(PixelFormat::RGBA8);
        img.fill_rect(Rect::new(0, 0, 100, 50), &[1.0, 0.0, 0.0, 1.0]);
        let out = paste(&sc, 0, &img, Rect::new(0, 0, 100, 50), [0.25, 0.25], 0.5).unwrap().unwrap();
        // The pasted rectangle lands inside the plane, around plane (0.5, 0.5).
        let h = sc.planes[0].homography().unwrap();
        let (x, y) = h.apply(0.5, 0.5);
        assert!(out.rgba(x as i32, y as i32)[3] > 0.9);
        let (x, y) = h.apply(0.1, 0.1);
        assert!(out.rgba(x as i32, y as i32)[3] < 0.1);
        // Clone: paint a red dot near the far edge from a red source near the near edge.
        let mut s = Surface::new(PixelFormat::RGBA8);
        s.fill_rect(Rect::new(0, 0, 800, 600), &[0.2, 0.2, 0.2, 1.0]);
        let (sx, sy) = h.apply(0.5, 0.8);
        s.fill_rect(Rect::new(sx as i32 - 6, sy as i32 - 6, sx as i32 + 6, sy as i32 + 6), &[1.0, 0.0, 0.0, 1.0]);
        let (dx, dy) = h.apply(0.5, 0.3);
        let n = clone_stroke(&sc, &mut s, [sx, sy], &[[dx, dy]], 10.0, 0.8, 1.0);
        assert_eq!(n, 1);
        assert!(s.rgba(dx as i32, dy as i32)[0] > 0.8, "{:?}", s.rgba(dx as i32, dy as i32));
    }
}
