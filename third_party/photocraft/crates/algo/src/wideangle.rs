//! Filter › Adaptive Wide Angle: straighten lines that a wide-angle or fisheye lens bends.
//!
//! The user marks constraints by their two end points; the lens model (equidistant fisheye
//! `r = f·θ`, rectilinear perspective `r = f·tan θ`, or a full 2:1 equirectangular sphere) turns
//! them into rays, and the scene line between them is the great circle through both rays,
//! which is sampled back into the image as the curve the user sees. A content-preserving mesh
//! warp then makes every constrained curve straight (and horizontal or vertical when asked)
//! while keeping each mesh triangle as similar as possible to its original shape:
//!
//! * shape term: T. Igarashi, T. Moscovich, J. Hughes, *As-Rigid-As-Possible Shape
//!   Manipulation*, SIGGRAPH 2005 (the similarity / "as-similar-as-possible" step), as used for
//!   wide-angle correction by R. Carroll, M. Agrawala, A. Agarwala, *Optimizing
//!   Content-Preserving Projections for Wide-Angle Images*, SIGGRAPH 2009;
//! * line term: every sample of a constrained curve lies on one output line (normal fixed for
//!   horizontal / vertical constraints, re-estimated a few times for free ones), as in
//!   Carroll et al.
//!
//! The linear least-squares problems are solved with conjugate gradients. The result is
//! rendered by textured triangles ([`crate::warp::warp_triangles`]).

use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde::{Deserialize, Serialize};

use crate::photo_util::SparseLs;
use crate::transform::Interp;

/// Lens projection of the source image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum WideModel {
    #[default]
    Auto,
    Fisheye,
    Perspective,
    FullSpherical,
}

impl WideModel {
    pub fn parse(s: &str) -> Option<WideModel> {
        Some(match s {
            "auto" => WideModel::Auto,
            "fisheye" => WideModel::Fisheye,
            "perspective" => WideModel::Perspective,
            "fullSpherical" | "spherical" => WideModel::FullSpherical,
            _ => return None,
        })
    }
}

/// Constraint orientation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum Orientation {
    #[default]
    Free,
    Horizontal,
    Vertical,
}

/// A constraint: the end points of a scene line (image px).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Constraint {
    pub a: [f64; 2],
    pub b: [f64; 2],
    #[serde(default)]
    pub orientation: Orientation,
}

/// Adaptive Wide Angle settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct WideAngle {
    pub model: WideModel,
    /// Focal length in mm (35 mm frame after the crop factor).
    pub focal_length: f64,
    pub crop_factor: f64,
    /// Output scale in percent.
    pub scale: f64,
    pub constraints: Vec<Constraint>,
}

impl Default for WideAngle {
    fn default() -> Self {
        WideAngle { model: WideModel::Auto, focal_length: 0.0, crop_factor: 1.0, scale: 100.0, constraints: Vec::new() }
    }
}

/// Camera model over an image frame.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub model: WideModel,
    /// Focal length in px.
    pub f: f64,
    pub c: [f64; 2],
    pub size: [f64; 2],
}

impl Camera {
    /// Resolves `Auto` and the focal length: 35 mm-equivalent `focal_mm × crop` over a 36 mm
    /// long side; `0` picks a default (fisheye 8 mm-ish, perspective 24 mm).
    pub fn new(p: &WideAngle, frame: Rect) -> Camera {
        let (w, h) = (frame.width().max(1) as f64, frame.height().max(1) as f64);
        let f35 = p.focal_length * p.crop_factor.max(0.1);
        let model = match p.model {
            WideModel::Auto => {
                if (w / h - 2.0).abs() < 0.01 {
                    WideModel::FullSpherical
                } else if f35 > 0.0 && f35 < 17.0 {
                    WideModel::Fisheye
                } else {
                    WideModel::Perspective
                }
            }
            m => m,
        };
        let f35 = if f35 > 0.0 {
            f35
        } else if model == WideModel::Fisheye {
            12.0
        } else {
            24.0
        };
        Camera { model, f: f35 / 36.0 * w.max(h), c: [(frame.x0 + frame.x1) as f64 / 2.0, (frame.y0 + frame.y1) as f64 / 2.0], size: [w, h] }
    }

    /// Image px → unit ray (x right, y down, z forward).
    pub fn ray(&self, x: f64, y: f64) -> [f64; 3] {
        let (dx, dy) = (x - self.c[0], y - self.c[1]);
        let v = match self.model {
            WideModel::FullSpherical => {
                let lon = dx / self.size[0] * std::f64::consts::TAU;
                let lat = dy / self.size[1] * std::f64::consts::PI;
                [lon.sin() * lat.cos(), lat.sin(), lon.cos() * lat.cos()]
            }
            WideModel::Fisheye => {
                let r = dx.hypot(dy);
                let th = r / self.f;
                if r < 1e-12 { [0.0, 0.0, 1.0] } else { [th.sin() * dx / r, th.sin() * dy / r, th.cos()] }
            }
            _ => [dx, dy, self.f],
        };
        norm(v)
    }

    /// Unit ray → image px.
    pub fn pixel(&self, d: [f64; 3]) -> Option<[f64; 2]> {
        match self.model {
            WideModel::FullSpherical => {
                let lon = d[0].atan2(d[2]);
                let lat = d[1].clamp(-1.0, 1.0).asin();
                Some([self.c[0] + lon / std::f64::consts::TAU * self.size[0], self.c[1] + lat / std::f64::consts::PI * self.size[1]])
            }
            WideModel::Fisheye => {
                let th = d[2].clamp(-1.0, 1.0).acos();
                let s = d[0].hypot(d[1]);
                if s < 1e-12 {
                    return Some(self.c);
                }
                Some([self.c[0] + self.f * th * d[0] / s, self.c[1] + self.f * th * d[1] / s])
            }
            _ => (d[2] > 1e-6).then(|| [self.c[0] + self.f * d[0] / d[2], self.c[1] + self.f * d[1] / d[2]]),
        }
    }

    /// `n + 1` image points along the scene line (great circle) from `a` to `b`.
    pub fn arc(&self, a: [f64; 2], b: [f64; 2], n: usize) -> Vec<[f64; 2]> {
        let (da, db) = (self.ray(a[0], a[1]), self.ray(b[0], b[1]));
        let omega = dot(da, db).clamp(-1.0, 1.0).acos();
        (0..=n)
            .filter_map(|k| {
                let t = k as f64 / n as f64;
                let d = if omega < 1e-9 {
                    da
                } else {
                    let (sa, sb) = (((1.0 - t) * omega).sin() / omega.sin(), (t * omega).sin() / omega.sin());
                    norm([sa * da[0] + sb * db[0], sa * da[1] + sb * db[1], sa * da[2] + sb * db[2]])
                };
                self.pixel(d)
            })
            .collect()
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(v: [f64; 3]) -> [f64; 3] {
    let l = dot(v, v).sqrt().max(1e-300);
    [v[0] / l, v[1] / l, v[2] / l]
}

/// The solved warp: a grid over the frame with each vertex's output position.
#[derive(Clone, Debug)]
pub struct WideMesh {
    pub nx: usize,
    pub ny: usize,
    /// `(output, source)` per vertex, row-major `(nx + 1) × (ny + 1)`.
    pub verts: Vec<([f64; 2], [f64; 2])>,
    /// Each constraint's curve in the source (for overlays).
    pub curves: Vec<Vec<[f64; 2]>>,
    /// Max deviation from straight of the constrained curves after the warp (px).
    pub residual: f64,
}

impl WideMesh {
    pub fn triangles(&self) -> Vec<[usize; 3]> {
        let w1 = self.nx + 1;
        let mut t = Vec::with_capacity(self.nx * self.ny * 2);
        for j in 0..self.ny {
            for i in 0..self.nx {
                let a = j * w1 + i;
                t.push([a, a + 1, a + w1 + 1]);
                t.push([a, a + w1 + 1, a + w1]);
            }
        }
        t
    }

    /// Output position of source point `p` (bilinear in its grid cell).
    pub fn map(&self, frame: Rect, p: [f64; 2]) -> [f64; 2] {
        let (cw, ch) = (frame.width() as f64 / self.nx as f64, frame.height() as f64 / self.ny as f64);
        let gx = ((p[0] - frame.x0 as f64) / cw).clamp(0.0, self.nx as f64 - 1e-9);
        let gy = ((p[1] - frame.y0 as f64) / ch).clamp(0.0, self.ny as f64 - 1e-9);
        let (i, j) = (gx.floor() as usize, gy.floor() as usize);
        let (tx, ty) = (gx - i as f64, gy - j as f64);
        let w1 = self.nx + 1;
        let v = |ii: usize, jj: usize| self.verts[jj * w1 + ii].0;
        let (a, b, c, d) = (v(i, j), v(i + 1, j), v(i, j + 1), v(i + 1, j + 1));
        [
            (a[0] * (1.0 - tx) + b[0] * tx) * (1.0 - ty) + (c[0] * (1.0 - tx) + d[0] * tx) * ty,
            (a[1] * (1.0 - tx) + b[1] * tx) * (1.0 - ty) + (c[1] * (1.0 - tx) + d[1] * tx) * ty,
        ]
    }
}

/// Solves the content-preserving warp for `p` over `frame`.
pub fn solve(p: &WideAngle, frame: Rect) -> WideMesh {
    let cam = Camera::new(p, frame);
    let (fw, fh) = (frame.width().max(1) as f64, frame.height().max(1) as f64);
    let cells = 48.0;
    let cell = fw.max(fh) / cells;
    let nx = ((fw / cell).round() as usize).clamp(2, 96);
    let ny = ((fh / cell).round() as usize).clamp(2, 96);
    let (cw, ch) = (fw / nx as f64, fh / ny as f64);
    let w1 = nx + 1;
    let nv = w1 * (ny + 1);
    let src: Vec<[f64; 2]> = (0..nv).map(|k| [frame.x0 as f64 + (k % w1) as f64 * cw, frame.y0 as f64 + (k / w1) as f64 * ch]).collect();
    let curves: Vec<Vec<[f64; 2]>> = p.constraints.iter().map(|c| cam.arc(c.a, c.b, 24)).collect();
    let mut x: Vec<f64> = src.iter().flat_map(|s| [s[0], s[1]]).collect();
    // Bilinear weights of a source point in the grid.
    let weights = |q: [f64; 2]| -> [(usize, f64); 4] {
        let gx = ((q[0] - frame.x0 as f64) / cw).clamp(0.0, nx as f64 - 1e-9);
        let gy = ((q[1] - frame.y0 as f64) / ch).clamp(0.0, ny as f64 - 1e-9);
        let (i, j) = (gx.floor() as usize, gy.floor() as usize);
        let (tx, ty) = (gx - i as f64, gy - j as f64);
        let a = j * w1 + i;
        [(a, (1.0 - tx) * (1.0 - ty)), (a + 1, tx * (1.0 - ty)), (a + w1, (1.0 - tx) * ty), (a + w1 + 1, tx * ty)]
    };
    let tris = {
        let mut t = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                let a = j * w1 + i;
                t.push([a, a + 1, a + w1 + 1]);
                t.push([a, a + w1 + 1, a + w1]);
                // Both diagonals so the shape term is symmetric.
                t.push([a, a + 1, a + w1]);
                t.push([a + 1, a + w1 + 1, a + w1]);
            }
        }
        t
    };
    let active: Vec<usize> = (0..curves.len()).filter(|&k| curves[k].len() >= 3).collect();
    if !active.is_empty() {
        let mut normals: Vec<[f64; 2]> = active.iter().map(|_| [0.0, 1.0]).collect();
        for round in 0..4 {
            let mut ls = SparseLs::new(2 * nv);
            // Shape: v0 = v1 + u (v2 − v1) + v R90 (v2 − v1), with (u, v) from the source.
            for t in &tris {
                for rot in 0..3 {
                    let (i0, i1, i2) = (t[rot], t[(rot + 1) % 3], t[(rot + 2) % 3]);
                    let (p0, p1, p2) = (src[i0], src[i1], src[i2]);
                    let e = [p2[0] - p1[0], p2[1] - p1[1]];
                    let l2 = e[0] * e[0] + e[1] * e[1];
                    let d = [p0[0] - p1[0], p0[1] - p1[1]];
                    let u = (d[0] * e[0] + d[1] * e[1]) / l2;
                    let v = (d[0] * -e[1] + d[1] * e[0]) / l2;
                    // x: v0x − v1x − u (v2x − v1x) + v (v2y − v1y) = 0
                    ls.add(vec![(2 * i0, 1.0), (2 * i1, -1.0 + u), (2 * i2, -u), (2 * i2 + 1, v), (2 * i1 + 1, -v)], 0.0, 1.0);
                    // y: v0y − v1y − u (v2y − v1y) − v (v2x − v1x) = 0
                    ls.add(vec![(2 * i0 + 1, 1.0), (2 * i1 + 1, -1.0 + u), (2 * i2 + 1, -u), (2 * i2, -v), (2 * i1, v)], 0.0, 1.0);
                }
            }
            // Weak anchoring to the source (fixes the global similarity).
            for (k, s) in src.iter().enumerate() {
                ls.add(vec![(2 * k, 1.0)], s[0], 0.02);
                ls.add(vec![(2 * k + 1, 1.0)], s[1], 0.02);
            }
            // Lines.
            for (ci, &k) in active.iter().enumerate() {
                let c = &curves[k];
                let n = match p.constraints[k].orientation {
                    Orientation::Horizontal => [0.0, 1.0],
                    Orientation::Vertical => [1.0, 0.0],
                    Orientation::Free => normals[ci],
                };
                let w0 = weights(c[0]);
                for q in &c[1..] {
                    let wq = weights(*q);
                    let mut row: Vec<(usize, f64)> = Vec::with_capacity(16);
                    for &(i, w) in &wq {
                        row.push((2 * i, n[0] * w));
                        row.push((2 * i + 1, n[1] * w));
                    }
                    for &(i, w) in &w0 {
                        row.push((2 * i, -n[0] * w));
                        row.push((2 * i + 1, -n[1] * w));
                    }
                    ls.add(row, 0.0, 10.0);
                }
            }
            x = ls.solve(&x, 400 + 200 * round);
            // Re-estimate free normals from the current end points.
            let pos = |q: [f64; 2]| -> [f64; 2] { weights(q).iter().fold([0.0, 0.0], |acc, &(i, w)| [acc[0] + w * x[2 * i], acc[1] + w * x[2 * i + 1]]) };
            let mut changed = false;
            for (ci, &k) in active.iter().enumerate() {
                let c = &curves[k];
                let (a, b) = (pos(c[0]), pos(c[c.len() - 1]));
                let d = [b[0] - a[0], b[1] - a[1]];
                let l = d[0].hypot(d[1]).max(1e-9);
                let nn = [-d[1] / l, d[0] / l];
                if (nn[0] - normals[ci][0]).abs() + (nn[1] - normals[ci][1]).abs() > 1e-3 {
                    changed = true;
                }
                normals[ci] = nn;
            }
            if round == 0 && active.iter().all(|&k| p.constraints[k].orientation != Orientation::Free) {
                break;
            }
            if round > 0 && !changed {
                break;
            }
        }
    }
    // Scale about the centre.
    let s = (p.scale / 100.0).clamp(0.1, 10.0);
    let c = cam.c;
    let verts: Vec<([f64; 2], [f64; 2])> = (0..nv).map(|k| ([c[0] + (x[2 * k] - c[0]) * s, c[1] + (x[2 * k + 1] - c[1]) * s], src[k])).collect();
    let mut mesh = WideMesh { nx, ny, verts, curves, residual: 0.0 };
    // Straightness residual: distance of curve samples from the chord of their mapped ends.
    let mut worst = 0.0f64;
    for (k, cv) in mesh.curves.iter().enumerate() {
        if cv.len() < 3 {
            continue;
        }
        let pts: Vec<[f64; 2]> = cv.iter().map(|q| mesh.map(frame, *q)).collect();
        let (a, b) = (pts[0], pts[pts.len() - 1]);
        let d = [b[0] - a[0], b[1] - a[1]];
        let l = d[0].hypot(d[1]).max(1e-9);
        for q in &pts {
            worst = worst.max(((q[0] - a[0]) * d[1] - (q[1] - a[1]) * d[0]).abs() / l);
        }
        let _ = k;
    }
    mesh.residual = worst;
    mesh
}

/// Applies Adaptive Wide Angle to `src` over `frame` (pixels outside the warped image become
/// transparent).
pub fn apply(src: &Surface, frame: Rect, p: &WideAngle, interp: Interp) -> Surface {
    let mesh = solve(p, frame);
    let tris = mesh.triangles();
    crate::warp::warp_triangles(src, frame, &mesh.verts, &tris, interp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_models_round_trip() {
        let frame = Rect::new(0, 0, 600, 400);
        for model in [WideModel::Fisheye, WideModel::Perspective] {
            let cam = Camera::new(&WideAngle { model, focal_length: 15.0, ..Default::default() }, frame);
            for (x, y) in [(10.0, 20.0), (300.0, 200.0), (590.0, 350.0)] {
                let p = cam.pixel(cam.ray(x, y)).unwrap();
                assert!((p[0] - x).abs() < 1e-6 && (p[1] - y).abs() < 1e-6, "{model:?}");
            }
        }
        let eq = Camera::new(&WideAngle::default(), Rect::new(0, 0, 800, 400));
        assert_eq!(eq.model, WideModel::FullSpherical);
        let p = eq.pixel(eq.ray(123.0, 77.0)).unwrap();
        assert!((p[0] - 123.0).abs() < 1e-6 && (p[1] - 77.0).abs() < 1e-6);
        // Perspective arcs are straight; fisheye arcs bend.
        let cam = Camera::new(&WideAngle { model: WideModel::Perspective, ..Default::default() }, frame);
        let arc = cam.arc([50.0, 50.0], [550.0, 80.0], 10);
        let mid = arc[5];
        assert!((mid[1] - 65.0).abs() < 2.0, "{mid:?}");
        let fish = Camera::new(&WideAngle { model: WideModel::Fisheye, focal_length: 8.0, ..Default::default() }, frame);
        let arc = fish.arc([50.0, 60.0], [550.0, 60.0], 10);
        assert!(arc[5][1] < 50.0, "a line near the top bows away from the centre: {:?}", arc[5]);
    }

    #[test]
    fn constraints_become_straight_and_level() {
        let frame = Rect::new(0, 0, 480, 320);
        let p = WideAngle {
            model: WideModel::Fisheye,
            focal_length: 8.0,
            constraints: vec![
                Constraint { a: [60.0, 70.0], b: [420.0, 70.0], orientation: Orientation::Horizontal },
                Constraint { a: [80.0, 60.0], b: [80.0, 260.0], orientation: Orientation::Free },
            ],
            ..Default::default()
        };
        let cam = Camera::new(&p, frame);
        let before = {
            let c = cam.arc([60.0, 70.0], [420.0, 70.0], 24);
            c.iter().map(|q| (q[1] - 70.0).abs()).fold(0.0, f64::max)
        };
        assert!(before > 10.0, "the fisheye curve bows: {before}");
        let mesh = solve(&p, frame);
        assert!(mesh.residual < 2.0, "residual {}", mesh.residual);
        // The horizontal constraint is level.
        let pts: Vec<[f64; 2]> = mesh.curves[0].iter().map(|q| mesh.map(frame, *q)).collect();
        let ys: Vec<f64> = pts.iter().map(|q| q[1]).collect();
        let spread = ys.iter().cloned().fold(f64::MIN, f64::max) - ys.iter().cloned().fold(f64::MAX, f64::min);
        assert!(spread < 2.0, "{spread}");
        // Far from the constraints the image barely moves.
        let c = mesh.map(frame, [300.0, 250.0]);
        assert!((c[0] - 300.0).abs() < 80.0 && (c[1] - 250.0).abs() < 40.0, "{c:?}");
    }

    #[test]
    fn no_constraints_is_identity_and_renders() {
        use photocraft_color::PixelFormat;
        let frame = Rect::new(0, 0, 64, 48);
        let mut s = Surface::new(PixelFormat::RGBA8);
        s.fill_rect(frame, &[0.2, 0.4, 0.6, 1.0]);
        let mesh = solve(&WideAngle::default(), frame);
        assert!(mesh.verts.iter().all(|(o, s)| (o[0] - s[0]).abs() < 1e-9 && (o[1] - s[1]).abs() < 1e-9));
        let out = apply(&s, frame, &WideAngle { scale: 80.0, ..Default::default() }, Interp::Bilinear);
        assert!(out.pixel(32, 24)[3] > 0.99 && out.pixel(1, 1)[3] < 0.01);
    }
}

#[cfg(test)]
mod tile_tests {
    use super::*;
    use photocraft_color::PixelFormat;

    #[test]
    fn no_gaps_across_tiles() {
        let frame = Rect::new(0, 0, 772, 517);
        let mut s = Surface::new(PixelFormat::RGBA8);
        s.fill_rect(frame, &[0.2, 0.4, 0.6, 1.0]);
        for p in [
            WideAngle::default(),
            WideAngle {
                model: WideModel::Fisheye,
                focal_length: 12.0,
                constraints: vec![Constraint { a: [20.0, 30.0], b: [700.0, 35.0], orientation: Orientation::Horizontal }],
                ..Default::default()
            },
        ] {
            let out = apply(&s, frame, &p, Interp::Bilinear);
            let holes: Vec<(i32, i32)> = (80..440).flat_map(|y| (80..690).map(move |x| (x, y))).filter(|&(x, y)| out.rgba(x, y)[3] < 0.5).collect();
            let m = solve(&p, frame);
            assert!(
                holes.is_empty(),
                "{} holes, first {:?} last {:?}; mesh {}x{} v0 {:?} vlast {:?} bounds {:?}",
                holes.len(),
                holes.first(),
                holes.last(),
                m.nx,
                m.ny,
                m.verts[0],
                m.verts.last(),
                out.content_bounds()
            );
        }
    }
}
