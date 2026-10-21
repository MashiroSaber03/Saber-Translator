//! Edit › Perspective Warp: planes (quads) drawn over the image in Layout mode, whose corners are
//! then moved in Warp mode; the image follows a piecewise-homography map.
//!
//! Each plane maps its source quad onto its destination quad with a homography
//! `M_i = H_dst ∘ H_src⁻¹` (through the unit square, so `(u, v)` are the plane's projective
//! coordinates). Two homographies that agree on an edge's endpoints map the edge onto the same
//! segment but generally with a different spacing along it, so on its own a piecewise map tears
//! at shared edges. For every edge shared by two planes we therefore take the average of both
//! planes' images of each edge point as the target and add the difference, faded linearly
//! across the plane (weight `1 − u`, `u`, `1 − v` or `v` from that edge to the opposite one).
//! The correction vanishes at the corners (shared corners have one destination), so the map is
//! exactly continuous across shared edges, exactly the homography for a lone plane, and exactly
//! the identity when nothing moved. Points outside every plane use the nearest plane, with
//! `(u, v)` clamped for the corrections.
//!
//! Corners that coincide (within half a pixel) in the source are linked: they share one
//! destination (the mean of the given ones), as Photoshop's snapped planes do.

use serde::{Deserialize, Serialize};

use crate::transform::Homography;

/// One plane: source quad (Layout mode) and destination quad (Warp mode), corners clockwise from
/// top-left, document pixels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Plane {
    pub src: [[f64; 2]; 4],
    pub dst: [[f64; 2]; 4],
}

impl Plane {
    pub fn identity(q: [[f64; 2]; 4]) -> Self {
        Plane { src: q, dst: q }
    }
}

const LINK_EPS: f64 = 0.5;

/// Unit square corners in plane order.
const UNIT: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

fn unit_to_quad(q: &[[f64; 2]; 4]) -> Option<Homography> {
    Homography::rect_to_quad([0.0, 0.0, 1.0, 1.0], *q)
}

/// Groups of linked corners: `(plane, corner)` pairs that coincide in the source.
pub fn linked_corners(planes: &[Plane]) -> Vec<Vec<(usize, usize)>> {
    let mut groups: Vec<Vec<(usize, usize)>> = Vec::new();
    for (pi, p) in planes.iter().enumerate() {
        for (ci, c) in p.src.iter().enumerate() {
            let g = groups.iter_mut().find(|g| {
                let (a, b) = g[0];
                let o = planes[a].src[b];
                (o[0] - c[0]).abs() <= LINK_EPS && (o[1] - c[1]).abs() <= LINK_EPS
            });
            match g {
                Some(g) => g.push((pi, ci)),
                None => groups.push(vec![(pi, ci)]),
            }
        }
    }
    groups
}

/// Snaps linked corners to one source and one destination position (their means).
pub fn unify(planes: &mut [Plane]) {
    for g in linked_corners(planes) {
        if g.len() < 2 {
            continue;
        }
        let n = g.len() as f64;
        let s = g.iter().fold([0.0, 0.0], |a, &(p, c)| [a[0] + planes[p].src[c][0] / n, a[1] + planes[p].src[c][1] / n]);
        let d = g.iter().fold([0.0, 0.0], |a, &(p, c)| [a[0] + planes[p].dst[c][0] / n, a[1] + planes[p].dst[c][1] / n]);
        for &(p, c) in &g {
            planes[p].src[c] = s;
            planes[p].dst[c] = d;
        }
    }
}

/// Edge `k` of a plane runs from corner `k` to corner `k + 1`.
fn edge_uv(k: usize, u: f64, v: f64) -> ([f64; 2], f64) {
    // (point on the edge in plane coordinates, weight of that edge's correction)
    match k {
        0 => ([u, 0.0], 1.0 - v),
        1 => ([1.0, v], u),
        2 => ([u, 1.0], v),
        _ => ([0.0, v], 1.0 - u),
    }
}

struct PlaneMap {
    src: Homography,
    src_inv: Homography,
    dst: Homography,
    quad: [[f64; 2]; 4],
    /// For each edge: the neighbouring plane sharing it, if any.
    shared: [Option<usize>; 4],
}

/// A prepared perspective warp (forward map, source → destination).
pub struct PerspectiveMap {
    planes: Vec<PlaneMap>,
}

/// Straighten modes (Photoshop's Warp-mode buttons).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Straighten {
    Horizontal,
    Vertical,
    Auto,
}

impl Straighten {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "h" | "horizontal" | "level" => Some(Straighten::Horizontal),
            "v" | "vertical" => Some(Straighten::Vertical),
            "auto" | "both" => Some(Straighten::Auto),
            _ => None,
        }
    }
}

/// Makes the destination edges that are near-horizontal exactly horizontal and/or near-vertical
/// exactly vertical (within 45°), moving linked corners together.
pub fn straighten(planes: &mut [Plane], mode: Straighten) {
    unify(planes);
    let groups = linked_corners(planes);
    let group_of = |p: usize, c: usize| groups.iter().position(|g| g.contains(&(p, c))).unwrap_or(0);
    let mut pos: Vec<[f64; 2]> = groups.iter().map(|g| planes[g[0].0].dst[g[0].1]).collect();
    let edges: Vec<(usize, usize)> =
        (0..planes.len()).flat_map(|p| (0..4).map(move |k| (p, k))).map(|(p, k)| (group_of(p, k), group_of(p, (k + 1) % 4))).collect();
    for (a, b) in edges {
        let (pa, pb) = (pos[a], pos[b]);
        let (dx, dy) = ((pb[0] - pa[0]).abs(), (pb[1] - pa[1]).abs());
        if dy <= dx && matches!(mode, Straighten::Horizontal | Straighten::Auto) {
            let y = (pa[1] + pb[1]) / 2.0;
            pos[a][1] = y;
            pos[b][1] = y;
        } else if dx < dy && matches!(mode, Straighten::Vertical | Straighten::Auto) {
            let x = (pa[0] + pb[0]) / 2.0;
            pos[a][0] = x;
            pos[b][0] = x;
        }
    }
    for (g, p) in groups.iter().zip(pos) {
        for &(pi, ci) in g {
            planes[pi].dst[ci] = p;
        }
    }
}

impl PerspectiveMap {
    /// Prepares the map. `None` if a quad is degenerate (or there are no planes).
    pub fn new(planes: &[Plane]) -> Option<Self> {
        if planes.is_empty() {
            return None;
        }
        let mut planes = planes.to_vec();
        unify(&mut planes);
        let near = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).abs() <= LINK_EPS && (a[1] - b[1]).abs() <= LINK_EPS;
        let mut maps = Vec::with_capacity(planes.len());
        for (i, p) in planes.iter().enumerate() {
            let src = unit_to_quad(&p.src)?;
            let src_inv = src.inverse()?;
            let dst = unit_to_quad(&p.dst)?;
            let mut shared = [None; 4];
            for (k, s) in shared.iter_mut().enumerate() {
                let (a, b) = (p.src[k], p.src[(k + 1) % 4]);
                *s = planes
                    .iter()
                    .enumerate()
                    .find(|(j, o)| {
                        *j != i
                            && (0..4).any(|m| {
                                let (c, d) = (o.src[m], o.src[(m + 1) % 4]);
                                (near(a, c) && near(b, d)) || (near(a, d) && near(b, c))
                            })
                    })
                    .map(|(j, _)| j);
            }
            maps.push(PlaneMap { src, src_inv, dst, quad: p.src, shared });
        }
        Some(PerspectiveMap { planes: maps })
    }

    /// The plane containing `p` (or the nearest one) and `p`'s plane coordinates.
    fn locate(&self, x: f64, y: f64) -> (usize, f64, f64) {
        let mut best = (0, f64::MAX, 0.0, 0.0);
        for (i, pm) in self.planes.iter().enumerate() {
            let (u, v) = pm.src_inv.apply(x, y);
            if (-1e-9..=1.0 + 1e-9).contains(&u) && (-1e-9..=1.0 + 1e-9).contains(&v) {
                return (i, u, v);
            }
            let d = dist_to_quad(&pm.quad, [x, y]);
            if d < best.1 {
                best = (i, d, u, v);
            }
        }
        (best.0, best.2, best.3)
    }

    /// Maps a source point to the destination.
    pub fn map(&self, x: f64, y: f64) -> (f64, f64) {
        let (i, u, v) = self.locate(x, y);
        let pm = &self.planes[i];
        let (mut ox, mut oy) = pm.dst.apply(u, v);
        if !ox.is_finite() || !oy.is_finite() {
            return (x, y);
        }
        let (cu, cv) = (u.clamp(0.0, 1.0), v.clamp(0.0, 1.0));
        for (k, sh) in pm.shared.iter().enumerate() {
            let Some(j) = *sh else { continue };
            let (e, w) = edge_uv(k, cu, cv);
            if w <= 0.0 {
                continue;
            }
            let mine = pm.dst.apply(e[0], e[1]);
            let (qx, qy) = pm.src.apply(e[0], e[1]);
            let other = &self.planes[j];
            let (ou, ov) = other.src_inv.apply(qx, qy);
            let theirs = other.dst.apply(ou, ov);
            let target = ((mine.0 + theirs.0) / 2.0, (mine.1 + theirs.1) / 2.0);
            ox += w * (target.0 - mine.0);
            oy += w * (target.1 - mine.1);
        }
        (ox, oy)
    }

    /// Whether every plane maps onto itself (identity).
    pub fn is_identity(planes: &[Plane]) -> bool {
        planes.iter().all(|p| p.src.iter().zip(&p.dst).all(|(a, b)| (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9))
    }
}

fn dist_to_quad(q: &[[f64; 2]; 4], p: [f64; 2]) -> f64 {
    (0..4)
        .map(|k| {
            let (a, b) = (q[k], q[(k + 1) % 4]);
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let l2 = dx * dx + dy * dy;
            let t = if l2 > 0.0 { (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
            ((a[0] + t * dx - p[0]).powi(2) + (a[1] + t * dy - p[1]).powi(2)).sqrt()
        })
        .fold(f64::MAX, f64::min)
}

/// Corners of the unit square in plane order (for callers drawing plane grids).
pub fn unit_corners() -> [[f64; 2]; 4] {
    UNIT
}

/// A point of plane `p` at plane coordinates `(u, v)`, in source space.
pub fn plane_point(p: &Plane, u: f64, v: f64, dst: bool) -> [f64; 2] {
    let q = if dst { &p.dst } else { &p.src };
    unit_to_quad(q)
        .map(|h| {
            let (x, y) = h.apply(u, v);
            [x, y]
        })
        .unwrap_or(q[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: (f64, f64), b: (f64, f64), tol: f64) -> bool {
        (a.0 - b.0).abs() <= tol && (a.1 - b.1).abs() <= tol
    }

    #[test]
    fn identity_and_a_known_homography() {
        let q = [[10.0, 10.0], [90.0, 12.0], [85.0, 70.0], [12.0, 75.0]];
        let m = PerspectiveMap::new(&[Plane::identity(q)]).unwrap();
        for p in [(20.0, 20.0), (50.0, 40.0), (0.0, 0.0), (120.0, 90.0)] {
            assert!(close(m.map(p.0, p.1), p, 1e-9), "{p:?} → {:?}", m.map(p.0, p.1));
        }
        // Square → trapezoid: corners land exactly, the centre lands where the homography says.
        let src = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]];
        let dst = [[20.0, 0.0], [80.0, 0.0], [100.0, 100.0], [0.0, 100.0]];
        let m = PerspectiveMap::new(&[Plane { src, dst }]).unwrap();
        for (s, d) in src.iter().zip(&dst) {
            assert!(close(m.map(s[0], s[1]), (d[0], d[1]), 1e-9));
        }
        let h = Homography::rect_to_quad([0.0, 0.0, 100.0, 100.0], dst).unwrap();
        assert!(close(m.map(50.0, 50.0), h.apply(50.0, 50.0), 1e-9));
        // The diagonals' crossing stays the crossing: (50, 50) maps to the trapezoid's diagonal intersection.
        let c = m.map(50.0, 50.0);
        assert!((c.0 - 50.0).abs() < 1e-9 && c.1 < 50.0, "{c:?}");
    }

    #[test]
    fn shared_edges_stay_continuous() {
        // Two planes sharing the edge x = 50 (with slightly mismatched corners that get linked).
        let a = Plane { src: [[0.0, 0.0], [50.0, 0.0], [50.0, 100.0], [0.0, 100.0]], dst: [[0.0, 10.0], [50.0, 0.0], [55.0, 100.0], [0.0, 90.0]] };
        let b = Plane { src: [[50.2, 0.0], [100.0, 0.0], [100.0, 100.0], [50.0, 100.2]], dst: [[50.0, 0.0], [110.0, 20.0], [100.0, 80.0], [55.0, 100.0]] };
        let m = PerspectiveMap::new(&[a, b]).unwrap();
        // After linking, the shared edge runs from (50.1, 0) to (50, 100.1).
        for t in [0.05, 0.25, 0.5, 0.77, 0.95] {
            let (x, y) = (50.1 - 0.1 * t, 100.1 * t);
            let l = m.map(x - 1e-6, y);
            let r = m.map(x + 1e-6, y);
            assert!(close(l, r, 1e-4), "t={t}: {l:?} vs {r:?}");
        }
        assert_eq!(linked_corners(&[Plane::identity([[0.0; 2]; 4])]).len(), 1);
    }

    #[test]
    fn straighten_levels_and_plumbs_edges() {
        let mut ps =
            vec![Plane { src: [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]], dst: [[0.0, 3.0], [100.0, -2.0], [96.0, 100.0], [4.0, 104.0]] }];
        straighten(&mut ps, Straighten::Vertical);
        assert_eq!(ps[0].dst[1][0], ps[0].dst[2][0]);
        assert_eq!(ps[0].dst[0][0], ps[0].dst[3][0]);
        assert_ne!(ps[0].dst[0][1], ps[0].dst[1][1]);
        straighten(&mut ps, Straighten::Auto);
        assert_eq!(ps[0].dst[0][1], ps[0].dst[1][1]);
        assert_eq!(ps[0].dst[2][1], ps[0].dst[3][1]);
    }
}
