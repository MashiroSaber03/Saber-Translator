//! Edit › Puppet Warp: pins on a triangle mesh over the layer's opaque region, deformed
//! as-rigidly-as-possible.
//!
//! * **Mesh.** The layer's coverage (alpha above a threshold) becomes a signed distance field,
//!   offset by `expansion` to grow or shrink the region. A square grid at the density's spacing,
//!   split into triangles, is fitted to that outline (2D isosurface stuffing): vertices near the
//!   outline snap onto it and the triangles it crosses are cut along it, so the mesh edge follows
//!   the shape, and separate shapes get separate meshes. Inside, the mesh stays a uniform grid,
//!   which is all ARAP needs.
//! * **Deformation.** The per-triangle ARAP energy of Liu et al. 2008 (the triangle form of
//!   Sorkine & Alexa 2007), `E = Σ_t Σ_(i,j)∈t w_ij |(v_i − v_j) − R_t (x_i − x_j)|²` with
//!   cotangent weights, minimized by alternating a local step (best rotation per triangle from
//!   the 2×2 covariance: `atan2(S10 − S01, S00 + S11)`) and a global step (one sparse SPD solve
//!   per axis, Jacobi-preconditioned conjugate gradients, warm-started). Pins are stiff soft
//!   constraints on the barycentric point under them. The first global step uses `R = I`
//!   (a harmonic deformation), which is the classic initialisation. Modes change the local
//!   step: Rigid fits rotations, Distort fits similarities (rotation × scale, i.e. ASAP), and
//!   Normal fits rotation × √scale, in between. A pin's rotation fixes `R` on the triangles
//!   around it. Mesh components without a pin stay where they are.
//! * **Rendering.** Triangles are rasterized with [`crate::warp::warp_triangles`] (tile-parallel,
//!   premultiplied, bilinear/bicubic), back to front by the depth of their nearest pin.

use std::collections::{HashMap, HashSet};

use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde::{Deserialize, Serialize};

use crate::transform::Interp;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PuppetMode {
    Rigid,
    #[default]
    Normal,
    Distort,
}

impl PuppetMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "rigid" => Some(PuppetMode::Rigid),
            "normal" => Some(PuppetMode::Normal),
            "distort" => Some(PuppetMode::Distort),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PuppetDensity {
    Fewer,
    #[default]
    Normal,
    More,
}

impl PuppetDensity {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "fewer" | "fewerpoints" => Some(PuppetDensity::Fewer),
            "normal" => Some(PuppetDensity::Normal),
            "more" | "morepoints" => Some(PuppetDensity::More),
            _ => None,
        }
    }

    fn cells(self) -> f64 {
        match self {
            PuppetDensity::Fewer => 12.0,
            PuppetDensity::Normal => 20.0,
            PuppetDensity::More => 32.0,
        }
    }
}

/// A pin: where it was placed (`src`) and where it is now (`dst`), document pixels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PuppetPin {
    pub src: [f64; 2],
    pub dst: [f64; 2],
    /// Fixed rotation of the mesh around the pin (degrees); `None` = automatic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rotate: Option<f64>,
    /// Pin depth: where the mesh overlaps itself, triangles nearest a higher pin draw on top.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub depth: i32,
}

fn is_zero(v: &i32) -> bool {
    *v == 0
}

fn d_expansion() -> f64 {
    2.0
}

/// A complete puppet warp (the command's params and a smart filter's stored data).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PuppetWarp {
    #[serde(default)]
    pub pins: Vec<PuppetPin>,
    #[serde(default)]
    pub mode: PuppetMode,
    #[serde(default)]
    pub density: PuppetDensity,
    /// Grows (or with a negative value shrinks) the mesh beyond the opaque region, px.
    #[serde(default = "d_expansion")]
    pub expansion: f64,
}

impl PuppetWarp {
    pub fn is_identity(&self) -> bool {
        self.pins.iter().all(|p| p.src == p.dst && p.rotate.is_none_or(|r| r.rem_euclid(360.0) == 0.0))
    }
}

/// A triangle mesh in document pixels.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PuppetMesh {
    pub verts: Vec<[f64; 2]>,
    pub tris: Vec<[usize; 3]>,
    /// Grid spacing (px).
    pub spacing: f64,
}

/// Coverage threshold: alpha above this counts as part of the puppet.
const ALPHA_MIN: f32 = 0.05;

/// The outline field's longest side, in blocks: how finely the mesh edge follows the outline.
const FIELD_BLOCKS: f64 = 1024.0;

/// The puppet's region as a signed distance field over a block grid: negative inside the
/// opaque area grown (or shrunk) by the expansion, positive outside, in px.
struct Field {
    x0: f64,
    y0: f64,
    /// Block size (px).
    g: f64,
    bw: usize,
    bh: usize,
    /// Signed distance from each block centre to the opaque outline (px, negative inside).
    sd: Vec<f32>,
    expansion: f64,
}

impl Field {
    fn new(src: &Surface, bounds: Rect, expansion: f64) -> Field {
        let long = f64::from(bounds.width().max(bounds.height()));
        let grow = expansion.max(0.0).ceil();
        let g = ((long + 2.0 * grow + 8.0) / FIELD_BLOCKS).ceil().max(1.0) as i32;
        let area = bounds.inflate(grow as i32 + 2 * g);
        let bw = ((area.width() as i32 + g - 1) / g).max(1) as usize;
        let bh = ((area.height() as i32 + g - 1) / g).max(1) as usize;
        let mut cov = vec![false; bw * bh];
        let fmt = src.format();
        let n = fmt.channels();
        let block = |bx: usize, by: usize| {
            let (x, y) = (area.x0 + bx as i32 * g, area.y0 + by as i32 * g);
            Rect::new(x, y, x + g, y + g)
        };
        if !fmt.alpha {
            for (i, c) in cov.iter_mut().enumerate() {
                *c = !block(i % bw, i / bw).intersect(&bounds).is_empty();
            }
        } else {
            // One block row per task (rows are independent).
            let scan = |by: usize, out: &mut [bool]| {
                let r = Rect::new(area.x0, area.y0 + by as i32 * g, area.x0 + bw as i32 * g, area.y0 + (by as i32 + 1) * g).intersect(&bounds);
                if r.is_empty() || !src.has_tiles_in(r) {
                    return;
                }
                let row = src.read_region(r);
                let rw = r.width() as usize;
                for (i, p) in row.chunks_exact(n).enumerate() {
                    if p.last().is_some_and(|&a| a > ALPHA_MIN) {
                        let x = r.x0 + (i % rw) as i32;
                        if let Some(o) = out.get_mut(((x - area.x0) / g) as usize) {
                            *o = true;
                        }
                    }
                }
            };
            #[cfg(not(target_arch = "wasm32"))]
            {
                use rayon::prelude::*;
                cov.par_chunks_mut(bw).enumerate().for_each(|(by, out)| scan(by, out));
            }
            #[cfg(target_arch = "wasm32")]
            cov.chunks_mut(bw).enumerate().for_each(|(by, out)| scan(by, out));
        }
        let outside = edt(&cov, true, bw, bh);
        let inside = edt(&cov, false, bw, bh);
        let gf = f64::from(g);
        let sd = cov
            .iter()
            .zip(outside.iter().zip(&inside))
            .map(|(&c, (&o, &i))| if c { (-(i.sqrt() - 0.5) * gf) as f32 } else { ((o.sqrt() - 0.5) * gf) as f32 })
            .collect();
        Field { x0: f64::from(area.x0), y0: f64::from(area.y0), g: gf, bw, bh, sd, expansion }
    }

    /// The field at a block centre (exact, no interpolation).
    fn at(&self, bx: usize, by: usize) -> f64 {
        self.sd.get(by * self.bw + bx).map_or(f64::MAX, |&d| f64::from(d) - self.expansion)
    }

    /// The field at `p` (bilinear between block centres; clamped to the grid, which ends
    /// outside the region).
    fn value(&self, p: [f64; 2]) -> f64 {
        let fx = ((p[0] - self.x0) / self.g - 0.5).clamp(0.0, (self.bw - 1) as f64);
        let fy = ((p[1] - self.y0) / self.g - 0.5).clamp(0.0, (self.bh - 1) as f64);
        let (ix, iy) = (fx.floor() as usize, fy.floor() as usize);
        let (ix1, iy1) = ((ix + 1).min(self.bw - 1), (iy + 1).min(self.bh - 1));
        let (tx, ty) = (fx - ix as f64, fy - iy as f64);
        let top = self.at(ix, iy) * (1.0 - tx) + self.at(ix1, iy) * tx;
        let bottom = self.at(ix, iy1) * (1.0 - tx) + self.at(ix1, iy1) * tx;
        top * (1.0 - ty) + bottom * ty
    }

    /// The outline (where the field crosses zero) as closed loops, traced by marching squares
    /// over the block centres. The field's border blocks lie outside, so every loop closes.
    fn outline(&self) -> Vec<Vec<[f64; 2]>> {
        let (bw, bh) = (self.bw, self.bh);
        if bw < 2 || bh < 2 {
            return Vec::new();
        }
        let inside = |x: usize, y: usize| self.at(x, y) < 0.0;
        // A crossing is named by its block edge: 2·block for the edge to the right neighbour,
        // 2·block + 1 for the edge to the one below.
        let id = |x: usize, y: usize, down: bool| 2 * (y * bw + x) + usize::from(down);
        let point = |id: usize| {
            let b = id / 2;
            let (x, y) = (b % bw, b / bw);
            let (x2, y2) = if id.is_multiple_of(2) { (x + 1, y) } else { (x, y + 1) };
            let (fa, fb) = (self.at(x, y), self.at(x2, y2));
            let t = if (fa - fb).abs() > 1e-12 { (fa / (fa - fb)).clamp(0.0, 1.0) } else { 0.5 };
            let (dx, dy) = ((x2 - x) as f64, (y2 - y) as f64);
            [self.x0 + (x as f64 + 0.5 + dx * t) * self.g, self.y0 + (y as f64 + 0.5 + dy * t) * self.g]
        };
        let mut links: HashMap<usize, [usize; 2]> = HashMap::new();
        let mut link = |a: usize, b: usize| {
            for (p, q) in [(a, b), (b, a)] {
                let e = links.entry(p).or_insert([usize::MAX; 2]);
                if e[0] == usize::MAX {
                    e[0] = q;
                } else {
                    e[1] = q;
                }
            }
        };
        for y in 0..bh - 1 {
            for x in 0..bw - 1 {
                let c = [inside(x, y), inside(x + 1, y), inside(x + 1, y + 1), inside(x, y + 1)];
                if c.iter().all(|&v| v == c[0]) {
                    continue;
                }
                // Top, right, bottom, left.
                let edges = [id(x, y, false), id(x + 1, y, true), id(x, y + 1, false), id(x, y, true)];
                let cross = [c[0] != c[1], c[1] != c[2], c[3] != c[2], c[0] != c[3]];
                let ks: Vec<usize> = (0..4).filter(|&k| cross[k]).collect();
                if let [a, b] = ks[..] {
                    link(edges[a], edges[b]);
                } else if ks.len() == 4 {
                    // A saddle: the centre decides which diagonal pair is joined.
                    let centre = self.at(x, y) + self.at(x + 1, y) + self.at(x + 1, y + 1) + self.at(x, y + 1) < 0.0;
                    if c[0] == centre {
                        link(edges[0], edges[1]);
                        link(edges[2], edges[3]);
                    } else {
                        link(edges[3], edges[0]);
                        link(edges[1], edges[2]);
                    }
                }
            }
        }
        let mut keys: Vec<usize> = links.keys().copied().collect();
        keys.sort_unstable();
        let mut seen: HashSet<usize> = HashSet::with_capacity(keys.len());
        let mut loops = Vec::new();
        for &start in &keys {
            if seen.contains(&start) {
                continue;
            }
            let mut lp = Vec::new();
            let (mut prev, mut cur) = (usize::MAX, start);
            loop {
                seen.insert(cur);
                lp.push(point(cur));
                let Some(&[a, b]) = links.get(&cur) else { break };
                let next = if a != prev { a } else { b };
                if next == usize::MAX || seen.contains(&next) {
                    break;
                }
                (prev, cur) = (cur, next);
            }
            if lp.len() >= 3 {
                loops.push(lp);
            }
        }
        loops
    }
}

/// Squared Euclidean distance (in blocks) from every block to the nearest block whose `mask`
/// equals `site` (Felzenszwalb & Huttenlocher's separable lower-envelope transform). About
/// 1e20 when there is none.
fn edt(mask: &[bool], site: bool, w: usize, h: usize) -> Vec<f64> {
    const FAR: f64 = 1e20;
    let mut d: Vec<f64> = mask.iter().map(|&m| if m == site { 0.0 } else { FAR }).collect();
    let mut f = vec![0.0; w.max(h)];
    let mut out = vec![0.0; w.max(h)];
    let mut v = vec![0usize; w.max(h)];
    let mut z = vec![0.0; w.max(h) + 1];
    for x in 0..w {
        for (y, fy) in f[..h].iter_mut().enumerate() {
            *fy = d[y * w + x];
        }
        edt_1d(&f[..h], &mut out[..h], &mut v, &mut z);
        for (y, o) in out[..h].iter().enumerate() {
            d[y * w + x] = *o;
        }
    }
    for row in d.chunks_mut(w) {
        f[..w].copy_from_slice(row);
        edt_1d(&f[..w], row, &mut v, &mut z);
    }
    d
}

/// One-dimensional squared distance transform of the sampled function `f`.
fn edt_1d(f: &[f64], out: &mut [f64], v: &mut [usize], z: &mut [f64]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    let meet = |q: usize, p: usize| {
        let (qf, pf) = (q as f64, p as f64);
        ((f[q] + qf * qf) - (f[p] + pf * pf)) / (2.0 * (qf - pf))
    };
    for q in 1..n {
        let mut s = meet(q, v[k]);
        while k > 0 && s <= z[k] {
            k -= 1;
            s = meet(q, v[k]);
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    k = 0;
    for (q, o) in out.iter_mut().enumerate() {
        let qf = q as f64;
        while z[k + 1] < qf {
            k += 1;
        }
        let p = v[k];
        *o = (qf - p as f64).powi(2) + f[p];
    }
}

/// Distance from `p` to the segment `a`–`b`.
fn seg_dist(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 0.0 { (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
    dist2(p, [a[0] + dx * t, a[1] + dy * t]).sqrt()
}

/// Simplifies a closed loop (Douglas–Peucker, within `tol` px), which keeps its corners.
fn simplify(lp: &[[f64; 2]], tol: f64) -> Vec<[f64; 2]> {
    let n = lp.len();
    if n < 4 {
        return lp.to_vec();
    }
    let far = (1..n).max_by(|&a, &b| dist2(lp[0], lp[a]).total_cmp(&dist2(lp[0], lp[b]))).unwrap_or(n / 2);
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[far] = true;
    // Spans (a, b) of the loop; b == n stands for point 0.
    let mut stack = vec![(0, far), (far, n)];
    while let Some((a, b)) = stack.pop() {
        let (pa, pb) = (lp[a], lp[b % n]);
        let mut best = (0.0, 0);
        for (i, &p) in lp.iter().enumerate().take(b).skip(a + 1) {
            let d = seg_dist(p, pa, pb);
            if d > best.0 {
                best = (d, i);
            }
        }
        if best.0 > tol {
            keep[best.1] = true;
            stack.push((a, best.1));
            stack.push((best.1, b));
        }
    }
    let out: Vec<[f64; 2]> = (0..n).filter(|&i| keep[i]).map(|i| lp[i]).collect();
    if out.len() >= 3 {
        return out;
    }
    // A speck thinner than the tolerance: keep a few of its points.
    (0..n).step_by((n / 4).max(1)).map(|i| lp[i]).collect()
}

/// Splits the sides of a closed loop longer than `step` into equal parts.
fn densify(lp: &[[f64; 2]], step: f64) -> Vec<[f64; 2]> {
    let mut out = Vec::with_capacity(lp.len());
    for (k, &a) in lp.iter().enumerate() {
        let b = lp[(k + 1) % lp.len()];
        let parts = (dist2(a, b).sqrt() / step).ceil().clamp(1.0, 1e4) as usize;
        for i in 0..parts {
            out.push(lerp(a, b, i as f64 / parts as f64));
        }
    }
    out
}

fn lerp(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

fn orient(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Positive when `d` lies inside the circumcircle of the positively oriented `a`, `b`, `c`.
fn in_circle(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> f64 {
    let (ax, ay) = (a[0] - d[0], a[1] - d[1]);
    let (bx, by) = (b[0] - d[0], b[1] - d[1]);
    let (cx, cy) = (c[0] - d[0], c[1] - d[1]);
    (ax * ax + ay * ay) * (bx * cy - cx * by) - (bx * bx + by * by) * (ax * cy - cx * ay) + (cx * cx + cy * cy) * (ax * by - bx * ay)
}

const NONE: usize = usize::MAX;

/// A triangle of the Delaunay triangulation: vertices (positively oriented) and the neighbour
/// across the edge opposite each vertex.
#[derive(Clone, Copy)]
struct DTri {
    v: [usize; 3],
    n: [usize; 3],
    alive: bool,
}

/// Incremental Delaunay triangulation (Bowyer–Watson), started from a triangle around the area.
/// Vertices 0..3 are that triangle's.
struct Delaunay {
    pts: Vec<[f64; 2]>,
    tris: Vec<DTri>,
    stamp: Vec<u32>,
    pass: u32,
    last: usize,
}

impl Delaunay {
    fn new(lo: [f64; 2], hi: [f64; 2]) -> Delaunay {
        let (cx, cy) = (0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]));
        let d = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1.0) * 4.0;
        let mut pts = vec![[cx - 2.0 * d, cy - d], [cx + 2.0 * d, cy - d], [cx, cy + 2.0 * d]];
        if orient(pts[0], pts[1], pts[2]) < 0.0 {
            pts.swap(1, 2);
        }
        Delaunay { pts, tris: vec![DTri { v: [0, 1, 2], n: [NONE; 3], alive: true }], stamp: vec![0], pass: 0, last: 0 }
    }

    /// The triangle holding `p`: a visibility walk from the last one made.
    fn locate(&self, p: [f64; 2]) -> Option<usize> {
        let mut t = if self.tris.get(self.last).is_some_and(|t| t.alive) { self.last } else { self.tris.iter().position(|t| t.alive)? };
        for _ in 0..self.tris.len() + 3 {
            let tri = self.tris.get(t)?;
            let step = (0..3).find(|&k| orient(self.pts[tri.v[(k + 1) % 3]], self.pts[tri.v[(k + 2) % 3]], p) < 0.0);
            match step {
                None => return Some(t),
                Some(k) if tri.n[k] != NONE => t = tri.n[k],
                Some(_) => return None,
            }
        }
        self.tris.iter().position(|tri| tri.alive && (0..3).all(|k| orient(self.pts[tri.v[(k + 1) % 3]], self.pts[tri.v[(k + 2) % 3]], p) >= 0.0))
    }

    /// Inserts `p`; returns its vertex index (an existing vertex's when `p` coincides with it).
    fn insert(&mut self, p: [f64; 2]) -> Option<usize> {
        let t0 = self.locate(p)?;
        if let Some(&v) = self.tris[t0].v.iter().find(|&&v| dist2(self.pts[v], p) < 1e-12) {
            return Some(v);
        }
        self.pass = self.pass.wrapping_add(1);
        let pass = self.pass;
        // The cavity: every triangle whose circumcircle holds `p`, grown from `t0`.
        let mut bad = vec![t0];
        self.stamp[t0] = pass;
        let mut i = 0;
        while let Some(&t) = bad.get(i) {
            i += 1;
            for k in 0..3 {
                let nb = self.tris[t].n[k];
                if nb == NONE || self.stamp[nb] == pass {
                    continue;
                }
                let [a, b, c] = self.tris[nb].v.map(|v| self.pts[v]);
                if in_circle(a, b, c, p) > 0.0 {
                    self.stamp[nb] = pass;
                    bad.push(nb);
                }
            }
        }
        let mut rim: Vec<(usize, usize, usize)> = Vec::new();
        for &t in &bad {
            let tri = self.tris[t];
            for k in 0..3 {
                let nb = tri.n[k];
                if nb == NONE || self.stamp[nb] != pass {
                    rim.push((tri.v[(k + 1) % 3], tri.v[(k + 2) % 3], nb));
                }
            }
            self.tris[t].alive = false;
        }
        let pi = self.pts.len();
        self.pts.push(p);
        let base = self.tris.len();
        for (i, &(a, b, o)) in rim.iter().enumerate() {
            self.tris.push(DTri { v: [a, b, pi], n: [NONE, NONE, o], alive: true });
            self.stamp.push(0);
            if let Some(ot) = self.tris.get_mut(o) {
                for k in 0..3 {
                    let (x, y) = (ot.v[(k + 1) % 3], ot.v[(k + 2) % 3]);
                    if (x == b && y == a) || (x == a && y == b) {
                        ot.n[k] = base + i;
                    }
                }
            }
        }
        for (i, &(a, b, _)) in rim.iter().enumerate() {
            let across_b = rim.iter().position(|r| r.0 == b).map_or(NONE, |j| base + j);
            let across_a = rim.iter().position(|r| r.1 == a).map_or(NONE, |j| base + j);
            self.tris[base + i].n[0] = across_b;
            self.tris[base + i].n[1] = across_a;
        }
        self.last = base;
        Some(pi)
    }

    fn edges(&self) -> HashSet<(usize, usize)> {
        let mut e = HashSet::new();
        for t in self.tris.iter().filter(|t| t.alive) {
            for k in 0..3 {
                let (a, b) = (t.v[k], t.v[(k + 1) % 3]);
                e.insert((a.min(b), a.max(b)));
            }
        }
        e
    }
}

/// Even-odd point-in-polygon test against closed outline segments, bucketed by row.
struct Inside<'a> {
    pts: &'a [[f64; 2]],
    segs: &'a [(usize, usize)],
    y0: f64,
    row: f64,
    rows: Vec<Vec<usize>>,
}

impl<'a> Inside<'a> {
    fn new(pts: &'a [[f64; 2]], segs: &'a [(usize, usize)], row: f64) -> Self {
        let y0 = segs.iter().flat_map(|&(a, b)| [pts[a][1], pts[b][1]]).fold(f64::MAX, f64::min);
        let y1 = segs.iter().flat_map(|&(a, b)| [pts[a][1], pts[b][1]]).fold(f64::MIN, f64::max);
        let n = if y1 >= y0 { ((y1 - y0) / row).floor().clamp(0.0, 1e5) as usize + 1 } else { 0 };
        let mut rows = vec![Vec::new(); n];
        for (i, &(a, b)) in segs.iter().enumerate() {
            let (lo, hi) = (pts[a][1].min(pts[b][1]), pts[a][1].max(pts[b][1]));
            let (r0, r1) = (((lo - y0) / row).floor() as usize, ((hi - y0) / row).floor() as usize);
            for r in rows.iter_mut().take(r1 + 1).skip(r0) {
                r.push(i);
            }
        }
        Inside { pts, segs, y0, row, rows }
    }

    fn contains(&self, p: [f64; 2]) -> bool {
        if p[1] < self.y0 {
            return false;
        }
        let Some(row) = self.rows.get(((p[1] - self.y0) / self.row).floor() as usize) else { return false };
        let mut odd = false;
        for &i in row {
            let (a, b) = (self.pts[self.segs[i].0], self.pts[self.segs[i].1]);
            if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]) {
                odd = !odd;
            }
        }
        odd
    }
}

/// Most outline points the mesh takes; a busier outline is simplified more coarsely.
const MAX_OUTLINE_POINTS: usize = 20_000;

/// Builds the mesh over the covered part of `src` inside `bounds`.
///
/// The outline of the opaque region grown by `expansion` is traced, simplified (corners kept)
/// and split into sides no longer than the density's spacing; a triangular lattice at that
/// spacing fills the inside, at least half a step from the outline. The mesh is the
/// Delaunay triangulation of those points with the outline's sides recovered (split until
/// each is an edge), minus the triangles outside the outline.
pub fn build_mesh(src: &Surface, bounds: Rect, density: PuppetDensity, expansion: f64) -> PuppetMesh {
    if bounds.is_empty() {
        return PuppetMesh::default();
    }
    let e = if expansion.is_finite() { expansion.clamp(-200.0, 200.0) } else { 0.0 };
    // The density counts steps across the region the mesh covers (grown by the expansion).
    let long = f64::from(bounds.width().max(bounds.height())) + 2.0 * e.max(0.0);
    let spacing = (long / density.cells()).max(2.0);
    let field = Field::new(src, bounds, e);
    let loops = field.outline();
    let empty = PuppetMesh { spacing, ..Default::default() };
    if loops.is_empty() {
        return empty;
    }
    let mut tol = (0.5 * field.g).max(0.5);
    let rings = loop {
        let rings: Vec<Vec<[f64; 2]>> = loops.iter().map(|l| densify(&simplify(l, tol), spacing)).collect();
        if rings.iter().map(Vec::len).sum::<usize>() <= MAX_OUTLINE_POINTS || tol > spacing {
            break rings;
        }
        tol *= 2.0;
    };
    let (lo, hi) = ([field.x0, field.y0], [field.x0 + field.bw as f64 * field.g, field.y0 + field.bh as f64 * field.g]);
    let mut dl = Delaunay::new(lo, hi);
    let mut segs: Vec<(usize, usize)> = Vec::new();
    for ring in &rings {
        let idx: Vec<usize> = ring.iter().filter_map(|&p| dl.insert(p)).collect();
        for (k, &a) in idx.iter().enumerate() {
            let b = idx[(k + 1) % idx.len()];
            if a != b {
                segs.push((a, b));
            }
        }
    }
    // The inside: a triangular lattice, kept half a step clear of the outline.
    let rise = spacing * 3f64.sqrt() / 2.0;
    let rows = ((hi[1] - lo[1]) / rise).ceil() as usize;
    let cols = ((hi[0] - lo[0]) / spacing).ceil() as usize;
    for j in 0..=rows {
        for i in 0..=cols {
            let p = [lo[0] + (i as f64 + 0.5 * (j % 2) as f64) * spacing, lo[1] + j as f64 * rise];
            if field.value(p) < -0.5 * spacing {
                dl.insert(p);
            }
        }
    }
    // Recover the outline's sides: split a side the triangulation lacks at its midpoint.
    let mut budget = segs.len() * 2;
    for _ in 0..8 {
        let edges = dl.edges();
        let mut next = Vec::with_capacity(segs.len());
        let mut split = false;
        for &(a, b) in &segs {
            if budget == 0 || edges.contains(&(a.min(b), a.max(b))) {
                next.push((a, b));
                continue;
            }
            match dl.insert(lerp(dl.pts[a], dl.pts[b], 0.5)) {
                Some(c) if c != a && c != b => {
                    next.push((a, c));
                    next.push((c, b));
                    budget -= 1;
                    split = true;
                }
                _ => next.push((a, b)),
            }
        }
        segs = next;
        if !split {
            break;
        }
    }
    let inside = Inside::new(&dl.pts, &segs, spacing);
    let mut tris: Vec<[usize; 3]> = Vec::new();
    for t in dl.tris.iter().filter(|t| t.alive && t.v.iter().all(|&v| v >= 3)) {
        let [a, b, c] = t.v.map(|v| dl.pts[v]);
        if orient(a, b, c) > 1e-9 && inside.contains([(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0]) {
            tris.push(t.v);
        }
    }
    if tris.is_empty() {
        return empty;
    }
    // Keep the vertices in use, numbered for a narrow band in the solver's matrix.
    let pos = &dl.pts;
    let order = band_order(pos, &tris);
    let mut remap = vec![usize::MAX; pos.len()];
    for (new, &old) in order.iter().enumerate() {
        remap[old] = new;
    }
    PuppetMesh { verts: order.iter().map(|&i| pos[i]).collect(), tris: tris.iter().map(|t| t.map(|i| remap[i])).collect(), spacing }
}

/// The vertices `tris` use, in reverse Cuthill–McKee order (breadth first from the topmost
/// vertex of each piece, lower degrees first), which keeps the solver's banded factor small.
fn band_order(pos: &[[f64; 2]], tris: &[[usize; 3]]) -> Vec<usize> {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); pos.len()];
    for t in tris {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            adj[a].push(b);
            adj[b].push(a);
        }
    }
    for a in &mut adj {
        a.sort_unstable();
        a.dedup();
    }
    let mut starts: Vec<usize> = (0..pos.len()).filter(|&i| !adj[i].is_empty()).collect();
    starts.sort_by(|&a, &b| pos[a][1].total_cmp(&pos[b][1]).then(pos[a][0].total_cmp(&pos[b][0])));
    let mut seen = vec![false; pos.len()];
    let mut order = Vec::with_capacity(starts.len());
    for s in starts {
        if seen[s] {
            continue;
        }
        seen[s] = true;
        let first = order.len();
        order.push(s);
        let mut i = first;
        while let Some(&v) = order.get(i) {
            i += 1;
            let mut next: Vec<usize> = adj[v].iter().copied().filter(|&w| !seen[w]).collect();
            next.sort_by_key(|&w| adj[w].len());
            for w in next {
                seen[w] = true;
                order.push(w);
            }
        }
    }
    order.reverse();
    order
}

/// A compressed sparse row matrix (symmetric, used by CG).
struct Csr {
    rows: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<f64>,
    diag: Vec<f64>,
}

impl Csr {
    fn from_triplets(n: usize, mut t: Vec<(usize, usize, f64)>) -> Csr {
        t.sort_by_key(|a| (a.0, a.1));
        let mut rows = vec![0usize; n + 1];
        let mut cols = Vec::with_capacity(t.len());
        let mut vals: Vec<f64> = Vec::with_capacity(t.len());
        let mut last = (usize::MAX, usize::MAX);
        for (i, j, v) in t {
            if (i, j) == last {
                *vals.last_mut().unwrap_or(&mut 0.0) += v;
                continue;
            }
            last = (i, j);
            cols.push(j);
            vals.push(v);
            rows[i + 1] = cols.len();
        }
        for i in 0..n {
            rows[i + 1] = rows[i + 1].max(rows[i]);
        }
        let mut diag = vec![1.0; n];
        for (i, d) in diag.iter_mut().enumerate() {
            for k in rows[i]..rows[i + 1] {
                if cols[k] == i && vals[k] > 0.0 {
                    *d = vals[k];
                }
            }
        }
        Csr { rows, cols, vals, diag }
    }

    fn mul(&self, x: &[f64], out: &mut [f64]) {
        for (i, o) in out.iter_mut().enumerate() {
            let mut s = 0.0;
            for k in self.rows[i]..self.rows[i + 1] {
                s += self.vals[k] * x[self.cols[k]];
            }
            *o = s;
        }
    }

    /// Jacobi-preconditioned conjugate gradients, warm-started from `x`.
    fn solve(&self, b: &[f64], x: &mut [f64]) {
        let n = b.len();
        let mut r = vec![0.0; n];
        self.mul(x, &mut r);
        for i in 0..n {
            r[i] = b[i] - r[i];
        }
        let bn: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
        let mut z: Vec<f64> = r.iter().zip(&self.diag).map(|(r, d)| r / d).collect();
        let mut p = z.clone();
        let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let mut ap = vec![0.0; n];
        for _ in 0..(4 * n).clamp(50, 2000) {
            if r.iter().map(|v| v * v).sum::<f64>().sqrt() <= 1e-12 * bn {
                break;
            }
            self.mul(&p, &mut ap);
            let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
            if pap.abs() < 1e-300 {
                break;
            }
            let alpha = rz / pap;
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            for i in 0..n {
                z[i] = r[i] / self.diag[i];
            }
            let rz2: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
            let beta = rz2 / rz;
            rz = rz2;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
        }
    }
}

/// Banded Cholesky factor `A = L Lᵀ` (the mesh numbers its vertices in reverse Cuthill–McKee
/// order, so the system's bandwidth is about one breadth-first front): factored once per pin
/// set, then each global step is two triangular solves.
struct BandChol {
    n: usize,
    p: usize,
    /// Row `i` holds `L[i][i - p ..= i]`.
    l: Vec<f64>,
}

impl BandChol {
    /// `None` when the band would be too large (or the matrix isn't positive definite).
    fn new(a: &Csr) -> Option<BandChol> {
        let n = a.diag.len();
        let mut p = 0;
        for i in 0..n {
            for k in a.rows[i]..a.rows[i + 1] {
                p = p.max(i.abs_diff(a.cols[k]));
            }
        }
        if n.saturating_mul(p + 1) > 40_000_000 {
            return None;
        }
        let w = p + 1;
        let mut l = vec![0.0f64; n * w];
        for i in 0..n {
            for k in a.rows[i]..a.rows[i + 1] {
                let j = a.cols[k];
                if j <= i {
                    l[i * w + (j + p - i)] = a.vals[k];
                }
            }
        }
        for i in 0..n {
            let lo = i.saturating_sub(p);
            for j in lo..=i {
                let jlo = j.saturating_sub(p).max(lo);
                let mut s = l[i * w + (j + p - i)];
                for k in jlo..j {
                    s -= l[i * w + (k + p - i)] * l[j * w + (k + p - j)];
                }
                if i == j {
                    if s <= 0.0 || !s.is_finite() {
                        return None;
                    }
                    l[i * w + p] = s.sqrt();
                } else {
                    l[i * w + (j + p - i)] = s / l[j * w + p];
                }
            }
        }
        Some(BandChol { n, p, l })
    }

    #[allow(clippy::needless_range_loop)] // band index arithmetic reads clearer with indices
    fn solve(&self, b: &[f64], x: &mut [f64]) {
        let (n, p, w) = (self.n, self.p, self.p + 1);
        let mut y = b.to_vec();
        for i in 0..n {
            let mut s = y[i];
            for k in i.saturating_sub(p)..i {
                s -= self.l[i * w + (k + p - i)] * y[k];
            }
            y[i] = s / self.l[i * w + p];
        }
        for i in (0..n).rev() {
            let mut s = y[i];
            for k in i + 1..(i + p + 1).min(n) {
                s -= self.l[k * w + (i + p - k)] * x[k];
            }
            x[i] = s / self.l[i * w + p];
        }
    }
}

/// A pin bound to the mesh: barycentric weights on a triangle's vertices.
#[derive(Clone, Debug)]
struct Bound {
    verts: [usize; 3],
    bary: [f64; 3],
}

/// Prepared ARAP solver for one mesh and a set of pin source positions.
pub struct PuppetSolver {
    pub mesh: PuppetMesh,
    /// Cotangent weight of each triangle's edges: `w[t][k]` for the edge opposite corner `k`.
    w: Vec<[f64; 3]>,
    bound: Vec<Bound>,
    /// Triangles whose rotation a pin fixes: (pin index, triangles).
    pin_tris: Vec<Vec<usize>>,
    /// Vertices of components without pins (held at rest).
    anchored: Vec<bool>,
    matrix: Csr,
    chol: Option<BandChol>,
    pin_weight: f64,
}

fn cot(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    // Cotangent of the angle at `a`.
    let (u, v) = ([b[0] - a[0], b[1] - a[1]], [c[0] - a[0], c[1] - a[1]]);
    let cross = (u[0] * v[1] - u[1] * v[0]).abs();
    if cross < 1e-12 { 0.0 } else { (u[0] * v[0] + u[1] * v[1]) / cross }
}

fn locate(mesh: &PuppetMesh, p: [f64; 2]) -> Bound {
    for t in &mesh.tris {
        let [a, b, c] = t.map(|i| mesh.verts[i]);
        let det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
        if det.abs() < 1e-12 {
            continue;
        }
        let l1 = ((p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])) / det;
        let l2 = ((b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1])) / det;
        let l0 = 1.0 - l1 - l2;
        if l0 >= -1e-9 && l1 >= -1e-9 && l2 >= -1e-9 {
            return Bound { verts: *t, bary: [l0, l1, l2] };
        }
    }
    // Outside the mesh: the nearest vertex.
    let i = (0..mesh.verts.len()).min_by(|&a, &b| dist2(mesh.verts[a], p).total_cmp(&dist2(mesh.verts[b], p))).unwrap_or(0);
    Bound { verts: [i, i, i], bary: [1.0, 0.0, 0.0] }
}

fn dist2(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

impl PuppetSolver {
    /// Prepares the solver for pins placed at `pin_src` (rotation-fixed pins listed in
    /// `fixed_rotation`).
    pub fn new(mesh: PuppetMesh, pin_src: &[[f64; 2]], fixed_rotation: &[bool]) -> Self {
        let nv = mesh.verts.len();
        let w: Vec<[f64; 3]> = mesh
            .tris
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| mesh.verts[i]);
                [0.5 * cot(a, b, c), 0.5 * cot(b, c, a), 0.5 * cot(c, a, b)]
            })
            .collect();
        // Without vertices there is nothing to bind a pin to (`locate` would fall back to a vertex 0
        // that doesn't exist, #951); every pin loop zips `bound`, so the pins are then ignored.
        let bound: Vec<Bound> = if nv == 0 { Vec::new() } else { pin_src.iter().map(|p| locate(&mesh, *p)).collect() };
        // Connected components (union-find over triangle edges).
        let mut parent: Vec<usize> = (0..nv).collect();
        fn find(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        for t in &mesh.tris {
            for k in 1..3 {
                let (a, b) = (find(&mut parent, t[0]), find(&mut parent, t[k]));
                if a != b {
                    parent[a] = b;
                }
            }
        }
        let mut pinned_root = vec![false; nv];
        for b in &bound {
            for &v in &b.verts {
                let r = find(&mut parent, v);
                pinned_root[r] = true;
            }
        }
        let anchored: Vec<bool> = (0..nv).map(|v| !pinned_root[find(&mut parent, v)]).collect();
        // Rotation-fixed pins: the triangles around the pin's nearest vertex.
        let pin_tris: Vec<Vec<usize>> = bound
            .iter()
            .zip(pin_src)
            .enumerate()
            .map(|(k, (b, p))| {
                if !fixed_rotation.get(k).copied().unwrap_or(false) {
                    return Vec::new();
                }
                let v = *b.verts.iter().min_by(|&&a, &&c| dist2(mesh.verts[a], *p).total_cmp(&dist2(mesh.verts[c], *p))).unwrap_or(&0);
                mesh.tris.iter().enumerate().filter(|(_, t)| t.contains(&v)).map(|(i, _)| i).collect()
            })
            .collect();
        let mut trip: Vec<(usize, usize, f64)> = Vec::with_capacity(mesh.tris.len() * 12 + nv);
        let mut diag_sum = 0.0;
        for (t, wt) in mesh.tris.iter().zip(&w) {
            for k in 0..3 {
                let (i, j) = (t[(k + 1) % 3], t[(k + 2) % 3]);
                let wk = wt[k];
                trip.push((i, i, wk));
                trip.push((j, j, wk));
                trip.push((i, j, -wk));
                trip.push((j, i, -wk));
                diag_sum += 2.0 * wk;
            }
        }
        let pin_weight = 1e4 * (diag_sum / nv.max(1) as f64).max(1.0);
        for b in &bound {
            for a in 0..3 {
                for c in 0..3 {
                    trip.push((b.verts[a], b.verts[c], pin_weight * b.bary[a] * b.bary[c]));
                }
            }
        }
        for (v, &an) in anchored.iter().enumerate() {
            if an {
                trip.push((v, v, pin_weight));
            }
        }
        let matrix = Csr::from_triplets(nv, trip);
        let chol = BandChol::new(&matrix);
        PuppetSolver { mesh, w, bound, pin_tris, anchored, matrix, chol, pin_weight }
    }

    /// The ARAP energy of `v` for the given per-triangle linear maps (plus the pin terms).
    fn energy_with(&self, v: &[[f64; 2]], rots: &[[f64; 4]], pin_dst: &[[f64; 2]]) -> f64 {
        let x = &self.mesh.verts;
        let mut e = 0.0;
        for ((t, wt), r) in self.mesh.tris.iter().zip(&self.w).zip(rots) {
            for k in 0..3 {
                let (i, j) = (t[(k + 1) % 3], t[(k + 2) % 3]);
                let d = [v[i][0] - v[j][0], v[i][1] - v[j][1]];
                let ex = [x[i][0] - x[j][0], x[i][1] - x[j][1]];
                let re = [r[0] * ex[0] + r[1] * ex[1], r[2] * ex[0] + r[3] * ex[1]];
                e += wt[k] * ((d[0] - re[0]).powi(2) + (d[1] - re[1]).powi(2));
            }
        }
        for (b, d) in self.bound.iter().zip(pin_dst) {
            let p = (0..3).fold([0.0, 0.0], |a, k| [a[0] + b.bary[k] * v[b.verts[k]][0], a[1] + b.bary[k] * v[b.verts[k]][1]]);
            e += self.pin_weight * dist2(p, *d);
        }
        e
    }

    fn local(&self, v: &[[f64; 2]], mode: PuppetMode, fixed: &[Option<f64>]) -> Vec<[f64; 4]> {
        let x = &self.mesh.verts;
        let mut rots: Vec<[f64; 4]> = self
            .mesh
            .tris
            .iter()
            .zip(&self.w)
            .map(|(t, wt)| {
                let mut s = [0.0f64; 4];
                let mut norm = 0.0;
                for k in 0..3 {
                    let (i, j) = (t[(k + 1) % 3], t[(k + 2) % 3]);
                    let d = [v[i][0] - v[j][0], v[i][1] - v[j][1]];
                    let e = [x[i][0] - x[j][0], x[i][1] - x[j][1]];
                    s[0] += wt[k] * d[0] * e[0];
                    s[1] += wt[k] * d[0] * e[1];
                    s[2] += wt[k] * d[1] * e[0];
                    s[3] += wt[k] * d[1] * e[1];
                    norm += wt[k] * (e[0] * e[0] + e[1] * e[1]);
                }
                let (c, sn) = (s[0] + s[3], s[2] - s[1]);
                let th = sn.atan2(c);
                let scale = match mode {
                    PuppetMode::Rigid => 1.0,
                    PuppetMode::Normal => (c.hypot(sn) / norm.max(1e-12)).sqrt(),
                    PuppetMode::Distort => c.hypot(sn) / norm.max(1e-12),
                };
                let scale = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
                let (sth, cth) = th.sin_cos();
                [scale * cth, -scale * sth, scale * sth, scale * cth]
            })
            .collect();
        for (k, tris) in self.pin_tris.iter().enumerate() {
            if let Some(Some(deg)) = fixed.get(k) {
                let (s, c) = deg.to_radians().sin_cos();
                for &t in tris {
                    rots[t] = [c, -s, s, c];
                }
            }
        }
        rots
    }

    fn global(&self, rots: &[[f64; 4]], pin_dst: &[[f64; 2]], v: &mut [[f64; 2]]) {
        let nv = v.len();
        let x = &self.mesh.verts;
        let mut bx = vec![0.0; nv];
        let mut by = vec![0.0; nv];
        for ((t, wt), r) in self.mesh.tris.iter().zip(&self.w).zip(rots) {
            for k in 0..3 {
                let (i, j) = (t[(k + 1) % 3], t[(k + 2) % 3]);
                let e = [x[i][0] - x[j][0], x[i][1] - x[j][1]];
                let re = [r[0] * e[0] + r[1] * e[1], r[2] * e[0] + r[3] * e[1]];
                bx[i] += wt[k] * re[0];
                by[i] += wt[k] * re[1];
                bx[j] -= wt[k] * re[0];
                by[j] -= wt[k] * re[1];
            }
        }
        for (b, d) in self.bound.iter().zip(pin_dst) {
            for k in 0..3 {
                bx[b.verts[k]] += self.pin_weight * b.bary[k] * d[0];
                by[b.verts[k]] += self.pin_weight * b.bary[k] * d[1];
            }
        }
        for (i, &an) in self.anchored.iter().enumerate() {
            if an {
                bx[i] += self.pin_weight * x[i][0];
                by[i] += self.pin_weight * x[i][1];
            }
        }
        let mut sx: Vec<f64> = v.iter().map(|p| p[0]).collect();
        let mut sy: Vec<f64> = v.iter().map(|p| p[1]).collect();
        match &self.chol {
            Some(c) => {
                c.solve(&bx, &mut sx);
                c.solve(&by, &mut sy);
            }
            None => {
                self.matrix.solve(&bx, &mut sx);
                self.matrix.solve(&by, &mut sy);
            }
        }
        for (i, p) in v.iter_mut().enumerate() {
            *p = [sx[i], sy[i]];
        }
    }

    /// Deforms the mesh: `pin_dst` (one per pin given to [`Self::new`]), optional fixed pin
    /// rotations (degrees), `iterations` local/global rounds, warm-started from `init` when
    /// given. Returns the deformed vertices and the rigid ARAP energy after each round.
    pub fn solve(
        &self,
        pin_dst: &[[f64; 2]],
        rotations: &[Option<f64>],
        mode: PuppetMode,
        iterations: usize,
        init: Option<&[[f64; 2]]>,
    ) -> (Vec<[f64; 2]>, Vec<f64>) {
        let nv = self.mesh.verts.len();
        let mut v: Vec<[f64; 2]> = match init {
            Some(i) if i.len() == nv => i.to_vec(),
            _ => self.mesh.verts.clone(),
        };
        if nv == 0 {
            return (v, Vec::new());
        }
        if init.is_none_or(|i| i.len() != nv) {
            // Start from R = I, or with rotated pins, each triangle turned by the inverse-distance
            // weighted angle of the rotated pins (ARAP alone propagates a turn only slowly).
            let rotated: Vec<([f64; 2], f64)> = self
                .bound
                .iter()
                .zip(rotations)
                .filter_map(|(b, r)| {
                    r.map(|deg| {
                        (
                            (0..3).fold([0.0, 0.0], |a, k| {
                                [a[0] + b.bary[k] * self.mesh.verts[b.verts[k]][0], a[1] + b.bary[k] * self.mesh.verts[b.verts[k]][1]]
                            }),
                            deg.to_radians(),
                        )
                    })
                })
                .collect();
            let r: Vec<[f64; 4]> = self
                .mesh
                .tris
                .iter()
                .map(|t| {
                    if rotated.is_empty() {
                        return [1.0, 0.0, 0.0, 1.0];
                    }
                    let c = t.iter().fold([0.0, 0.0], |a, &i| [a[0] + self.mesh.verts[i][0] / 3.0, a[1] + self.mesh.verts[i][1] / 3.0]);
                    let (mut sw, mut sa) = (0.0, 0.0);
                    for (p, a) in &rotated {
                        let w = 1.0 / (dist2(*p, c) + 1.0);
                        sw += w;
                        sa += w * a;
                    }
                    let (s, co) = (sa / sw).sin_cos();
                    [co, -s, s, co]
                })
                .collect();
            self.global(&r, pin_dst, &mut v);
        }
        let mut energies = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let rots = self.local(&v, mode, rotations);
            self.global(&rots, pin_dst, &mut v);
            let rigid = self.local(&v, PuppetMode::Rigid, rotations);
            energies.push(self.energy_with(&v, &rigid, pin_dst));
        }
        (v, energies)
    }

    /// Energy of the current state with the rigid best fit (for monitoring).
    pub fn energy(&self, v: &[[f64; 2]], pin_dst: &[[f64; 2]], rotations: &[Option<f64>]) -> f64 {
        let rots = self.local(v, PuppetMode::Rigid, rotations);
        self.energy_with(v, &rots, pin_dst)
    }

    /// Triangle draw order (back to front) for pins with depths.
    pub fn draw_order(&self, depths: &[i32], pin_src: &[[f64; 2]]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.mesh.tris.len()).collect();
        if depths.iter().all(|d| *d == 0) || pin_src.is_empty() {
            return order;
        }
        let depth_of = |t: usize| {
            let c = self.mesh.tris[t].iter().fold([0.0, 0.0], |a, &i| [a[0] + self.mesh.verts[i][0] / 3.0, a[1] + self.mesh.verts[i][1] / 3.0]);
            let k = (0..pin_src.len()).min_by(|&a, &b| dist2(pin_src[a], c).total_cmp(&dist2(pin_src[b], c))).unwrap_or(0);
            depths.get(k).copied().unwrap_or(0)
        };
        let d: Vec<i32> = order.iter().map(|&t| depth_of(t)).collect();
        order.sort_by_key(|&t| d[t]);
        order
    }
}

/// Number of local/global rounds used by the command.
pub const ITERATIONS: usize = 40;

/// The deformed mesh of a puppet warp: (solver, deformed vertices, draw order).
pub fn deform(src: &Surface, bounds: Rect, w: &PuppetWarp, iterations: usize) -> (PuppetSolver, Vec<[f64; 2]>, Vec<usize>) {
    let mesh = build_mesh(src, bounds, w.density, w.expansion);
    let pin_src: Vec<[f64; 2]> = w.pins.iter().map(|p| p.src).collect();
    let fixed: Vec<bool> = w.pins.iter().map(|p| p.rotate.is_some()).collect();
    let solver = PuppetSolver::new(mesh, &pin_src, &fixed);
    let dst: Vec<[f64; 2]> = w.pins.iter().map(|p| p.dst).collect();
    let rot: Vec<Option<f64>> = w.pins.iter().map(|p| p.rotate).collect();
    let (v, _) = solver.solve(&dst, &rot, w.mode, iterations, None);
    let depths: Vec<i32> = w.pins.iter().map(|p| p.depth).collect();
    let order = solver.draw_order(&depths, &pin_src);
    (solver, v, order)
}

/// Rasterizes the mesh deformation of `src` (inside `bounds`): the deformed triangles only, so
/// pixels outside the mesh are dropped.
pub fn render(src: &Surface, bounds: Rect, mesh: &PuppetMesh, deformed: &[[f64; 2]], order: &[usize], interp: Interp) -> Surface {
    let verts: Vec<([f64; 2], [f64; 2])> = deformed.iter().zip(&mesh.verts).map(|(d, s)| (*d, *s)).collect();
    let tris: Vec<[usize; 3]> = order.iter().map(|&t| mesh.tris[t]).collect();
    crate::warp::warp_triangles(src, bounds, &verts, &tris, interp)
}

/// The largest area (px) a puppet warp's result may span: 2³⁰, about 16000 tiles. A pin dragged
/// absurdly far stretches the mesh over far more, and the renderer allocates bookkeeping for
/// every tile of it before drawing (#1019).
pub const MAX_WARP_PIXELS: f64 = (1u64 << 30) as f64;

/// Applies a puppet warp to `src` (content inside `bounds`). `None` when the deformed mesh would
/// span more than [`MAX_WARP_PIXELS`].
pub fn puppet_warp(src: &Surface, bounds: Rect, w: &PuppetWarp, interp: Interp) -> Option<Surface> {
    puppet_warp_with_mask(src, bounds, w, interp, None).map(|(out, _)| out)
}

/// [`puppet_warp`], also moving `mask` (a one-channel layer mask linked to `src`) with the same
/// mesh, as Photoshop moves a linked mask with its layer (#4910). The mask is `None` in the
/// result when none was given.
pub fn puppet_warp_with_mask(src: &Surface, bounds: Rect, w: &PuppetWarp, interp: Interp, mask: Option<&Surface>) -> Option<(Surface, Option<Surface>)> {
    let (solver, v, order) = deform(src, bounds, w, ITERATIONS);
    // No triangles (an expansion that shrinks every pixel away): nothing to warp. Rendering the
    // empty mesh would drop every pixel of the layer.
    if solver.mesh.tris.is_empty() {
        return Some((src.clone(), mask.cloned()));
    }
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for p in &v {
        (x0, y0, x1, y1) = (x0.min(p[0]), y0.min(p[1]), x1.max(p[0]), y1.max(p[1]));
    }
    let area = (x1 - x0 + 1.0) * (y1 - y0 + 1.0);
    if area.is_nan() || area > MAX_WARP_PIXELS {
        return None;
    }
    let out = render(src, bounds, &solver.mesh, &v, &order, interp);
    Some((out, mask.map(|m| render_mask(m, &solver.mesh, &v, &order, interp))))
}

/// Moves a one-channel layer mask with a deformed mesh: the mask under the mesh is lifted,
/// deformed like the layer's pixels and laid over the rest. The area the mesh leaves takes the
/// mask's default value (as [`crate::warp::warp_mesh_gray`] leaves it); outside the mesh the
/// mask keeps its values.
pub fn render_mask(mask: &Surface, mesh: &PuppetMesh, deformed: &[[f64; 2]], order: &[usize], interp: Interp) -> Surface {
    let mut out = mask.clone();
    if mesh.tris.is_empty() || deformed.len() != mesh.verts.len() || mask.format().channels() != 1 {
        return out;
    }
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for p in &mesh.verts {
        (x0, y0, x1, y1) = (x0.min(p[0]), y0.min(p[1]), x1.max(p[0]), y1.max(p[1]));
    }
    if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) {
        return out;
    }
    // The mesh's source area, a pixel wider so its edge samples stay inside.
    let lim = f64::from(1 << 20);
    let src = Rect::new(x0.max(-lim).floor() as i32 - 1, y0.max(-lim).floor() as i32 - 1, x1.min(lim).ceil() as i32 + 1, y1.min(lim).ceil() as i32 + 1);
    if src.is_empty() {
        return out;
    }
    let default = mask.default_pixel().first().copied().unwrap_or(0.0);
    let fmt = mask.format();
    let mut tmp = Surface::new(photocraft_color::PixelFormat::new(fmt.mode, fmt.sample, true));
    tmp.write_region(src, &mask.read_region(src).iter().flat_map(|g| [*g, 1.0]).collect::<Vec<f32>>());
    let tris: Vec<[usize; 3]> = order.iter().filter_map(|&t| mesh.tris.get(t).copied()).collect();
    // Lift: where the undeformed mesh covers the mask, the mask takes its default value.
    let still: Vec<([f64; 2], [f64; 2])> = mesh.verts.iter().map(|s| (*s, *s)).collect();
    let lifted = crate::warp::warp_triangles(&tmp, src, &still, &tris, Interp::Nearest);
    let b = lifted.content_bounds();
    if !b.is_empty() {
        let under = out.read_region(b);
        let flat: Vec<f32> = lifted.read_region(b).as_chunks::<2>().0.iter().zip(&under).map(|(p, u)| default * p[1] + u * (1.0 - p[1])).collect();
        out.write_region(b, &flat);
    }
    // Lay the deformed mask over what is left.
    let moved: Vec<([f64; 2], [f64; 2])> = deformed.iter().zip(&mesh.verts).map(|(d, s)| (*d, *s)).collect();
    let warped = crate::warp::warp_triangles(&tmp, src, &moved, &tris, interp);
    let b = warped.content_bounds();
    if !b.is_empty() {
        let under = out.read_region(b);
        let flat: Vec<f32> = warped.read_region(b).as_chunks::<2>().0.iter().zip(&under).map(|(p, u)| p[0] * p[1] + u * (1.0 - p[1])).collect();
        out.write_region(b, &flat);
    }
    out.prune();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::{ColorMode, PixelFormat, SampleType};

    fn blob(st: SampleType) -> Surface {
        let mut s = Surface::new(PixelFormat::new(ColorMode::Rgb, st, true));
        // An L shape: a vertical bar and a foot.
        for y in 10..70 {
            for x in 20..36 {
                s.write_pixel(x, y, &[((x + y) % 7) as f32 / 6.0, 0.5, (y as f32) / 80.0, 1.0]);
            }
        }
        for y in 56..70 {
            for x in 36..70 {
                s.write_pixel(x, y, &[0.2, ((x * 3) % 5) as f32 / 4.0, 0.8, 1.0]);
            }
        }
        s
    }

    #[test]
    fn mesh_covers_the_shape_only() {
        let s = blob(SampleType::U8);
        let b = s.content_bounds();
        let m = build_mesh(&s, b, PuppetDensity::Normal, 2.0);
        assert!(m.tris.len() > 20);
        let inside = |p: [f64; 2]| {
            m.tris.iter().any(|t| {
                let xs = t.map(|i| m.verts[i]);
                let (x0, x1) = (xs.iter().map(|v| v[0]).fold(f64::MAX, f64::min), xs.iter().map(|v| v[0]).fold(f64::MIN, f64::max));
                let (y0, y1) = (xs.iter().map(|v| v[1]).fold(f64::MAX, f64::min), xs.iter().map(|v| v[1]).fold(f64::MIN, f64::max));
                p[0] >= x0 && p[0] <= x1 && p[1] >= y0 && p[1] <= y1
            })
        };
        assert!(inside([28.0, 30.0]) && inside([60.0, 62.0]));
        assert!(!inside([60.0, 20.0]), "the empty corner of the L has no mesh");
        let more = build_mesh(&s, b, PuppetDensity::More, 2.0);
        let fewer = build_mesh(&s, b, PuppetDensity::Fewer, 2.0);
        assert!(more.tris.len() > m.tris.len() && m.tris.len() > fewer.tris.len());
    }

    fn in_mesh(m: &PuppetMesh, p: [f64; 2]) -> bool {
        m.tris.iter().any(|t| {
            let [a, b, c] = t.map(|i| m.verts[i]);
            let (d0, d1, d2) = (orient(a, b, p), orient(b, c, p), orient(c, a, p));
            (d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0) || (d0 <= 0.0 && d1 <= 0.0 && d2 <= 0.0)
        })
    }

    #[test]
    fn mesh_follows_the_outline() {
        // #3105: the mesh was every grid cell touching the shape, a staircase up to a cell
        // outside it. It now follows the outline grown by the expansion.
        let mut s = Surface::new(PixelFormat::new(ColorMode::Rgb, SampleType::U8, true));
        for y in 40..160 {
            for x in 40..160 {
                if (f64::from(x) + 0.5 - 100.0).hypot(f64::from(y) + 0.5 - 100.0) <= 60.0 {
                    s.write_pixel(x, y, &[0.3, 0.6, 0.9, 1.0]);
                }
            }
        }
        let m = build_mesh(&s, s.content_bounds(), PuppetDensity::Normal, 2.0);
        let r = |v: &[f64; 2]| (v[0] - 100.0).hypot(v[1] - 100.0);
        assert!(m.verts.iter().all(|v| r(v) <= 63.0), "a vertex outside the grown disc");
        assert!(m.verts.iter().filter(|v| r(v) > 61.0).count() >= 40, "the outline carries vertices");
        let area: f64 = m.tris.iter().map(|t| orient(m.verts[t[0]], m.verts[t[1]], m.verts[t[2]]).abs() / 2.0).sum();
        let disc = std::f64::consts::PI * 62.0 * 62.0;
        assert!((area / disc - 1.0).abs() < 0.03, "{area} vs {disc}");
        // Sharp corners and a concave corner: every opaque pixel is still inside the mesh, even
        // without an expansion, and the empty corner of the L is not.
        let l = blob(SampleType::U8);
        let m = build_mesh(&l, l.content_bounds(), PuppetDensity::Fewer, 0.0);
        for (x, y) in (20..36).flat_map(|x| (10..70).map(move |y| (x, y))).chain((36..70).flat_map(|x| (56..70).map(move |y| (x, y)))) {
            assert!(in_mesh(&m, [f64::from(x) + 0.5, f64::from(y) + 0.5]), "({x}, {y}) is outside the mesh");
        }
        assert!(!in_mesh(&m, [40.0, 50.0]) && !in_mesh(&m, [60.0, 20.0]));
    }

    #[test]
    fn no_pins_or_unmoved_pins_are_identity() {
        for st in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let s = blob(st);
            let b = s.content_bounds();
            for pins in [
                vec![],
                vec![
                    PuppetPin { src: [28.0, 20.0], dst: [28.0, 20.0], rotate: None, depth: 0 },
                    PuppetPin { src: [60.0, 62.0], dst: [60.0, 62.0], rotate: None, depth: 0 },
                ],
            ] {
                let w = PuppetWarp { pins, mode: PuppetMode::Normal, density: PuppetDensity::Normal, expansion: 2.0 };
                let out = puppet_warp(&s, b, &w, Interp::Bicubic).unwrap();
                let worst = s.read_region(b).iter().zip(out.read_region(b)).map(|(a, c)| (a - c).abs()).fold(0.0f32, f32::max);
                assert!(worst <= 1.0 / 255.0 + 1e-6, "{st:?}: {worst}");
            }
        }
    }

    #[test]
    fn an_expansion_that_empties_the_mesh_warps_nothing() {
        // #951: with pins and no vertices, `locate` fell back to vertex 0 and the solver indexed
        // its empty union-find.
        let solver = PuppetSolver::new(PuppetMesh::default(), &[[10.0, 10.0]], &[true]);
        assert!(solver.mesh.verts.is_empty());
        let s = blob(SampleType::U8);
        let b = s.content_bounds();
        let w = PuppetWarp {
            pins: vec![PuppetPin { src: [28.0, 30.0], dst: [38.0, 25.0], rotate: None, depth: 0 }],
            mode: PuppetMode::Normal,
            density: PuppetDensity::Normal,
            expansion: -200.0,
        };
        assert!(build_mesh(&s, b, w.density, w.expansion).tris.is_empty());
        // Nothing to warp: the pixels stay, rather than being dropped outside an empty mesh.
        let out = puppet_warp(&s, b, &w, Interp::Bilinear).unwrap();
        assert_eq!(out.read_region(b), s.read_region(b));
    }

    #[test]
    fn a_pin_dragged_absurdly_far_is_refused_before_rendering() {
        // #1019: the renderer sized its tile bookkeeping by the deformed mesh, so one far pin
        // allocated hundreds of MB before drawing a pixel.
        let s = blob(SampleType::U8);
        let b = s.content_bounds();
        let pin = |dst: [f64; 2]| PuppetWarp {
            pins: vec![
                PuppetPin { src: [28.0, 20.0], dst: [28.0, 20.0], rotate: None, depth: 0 },
                PuppetPin { src: [60.0, 62.0], dst, rotate: None, depth: 0 },
            ],
            mode: PuppetMode::Normal,
            density: PuppetDensity::Normal,
            expansion: 2.0,
        };
        assert!(puppet_warp(&s, b, &pin([1048576.0, 1048576.0]), Interp::Bilinear).is_none());
        assert!(puppet_warp(&s, b, &pin([-1e7, 62.0]), Interp::Bilinear).is_none());
        let near = puppet_warp(&s, b, &pin([70.0, 66.0]), Interp::Bilinear).expect("a near pin warps");
        assert!(!near.content_bounds().is_empty());
    }

    #[test]
    fn a_linked_mask_moves_with_the_mesh_and_leaves_its_default_behind() {
        // #4910. A hide-all mask (default 0) that reveals the bar's top (y 10..30): moved down by
        // 8, the revealed patch moves along, what it left is hidden again, and the mask away
        // from the puppet keeps its values.
        let s = blob(SampleType::U8);
        let b = s.content_bounds();
        for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let mut mask = Surface::with_default(PixelFormat::new(ColorMode::Grayscale, sample, false), &[0.0]);
            mask.fill_rect(Rect::new(20, 10, 36, 30), &[1.0]);
            mask.fill_rect(Rect::new(90, 90, 94, 94), &[0.5]);
            let w = PuppetWarp {
                pins: vec![PuppetPin { src: [28.0, 30.0], dst: [28.0, 38.0], rotate: None, depth: 0 }],
                mode: PuppetMode::Rigid,
                density: PuppetDensity::Normal,
                expansion: 2.0,
            };
            let (_, moved) = puppet_warp_with_mask(&s, b, &w, Interp::Bilinear, Some(&mask)).expect("warps");
            let m = moved.expect("a mask in, a mask out");
            let at = |x, y| m.sample_channel(x, y, 0);
            assert!(at(28, 20) > 0.99 && at(28, 36) > 0.99, "{sample:?}: the revealed patch moved down");
            assert!(at(28, 15) < 0.01, "{sample:?}: its old top is hidden again");
            assert!((at(91, 91) - 0.5).abs() <= 1.0 / 255.0, "{sample:?}: away from the mesh nothing changes");
            assert!(puppet_warp_with_mask(&s, b, &w, Interp::Bilinear, None).expect("warps").1.is_none());
        }
    }

    #[test]
    fn a_single_pin_translates_rigidly() {
        let s = blob(SampleType::U8);
        let b = s.content_bounds();
        let w = PuppetWarp {
            pins: vec![PuppetPin { src: [28.0, 30.0], dst: [38.0, 25.0], rotate: None, depth: 0 }],
            mode: PuppetMode::Rigid,
            density: PuppetDensity::Normal,
            expansion: 2.0,
        };
        let (solver, v, _) = deform(&s, b, &w, ITERATIONS);
        for (d, x) in v.iter().zip(&solver.mesh.verts) {
            assert!((d[0] - x[0] - 10.0).abs() < 1e-3 && (d[1] - x[1] + 5.0).abs() < 1e-3, "{d:?} {x:?}");
        }
        let out = puppet_warp(&s, b, &w, Interp::Bilinear).unwrap();
        assert_eq!(out.content_bounds(), b.translate(10, -5));
        assert_eq!(out.pixel(30, 20), s.pixel(20, 25));
        // A fixed rotation turns the whole mesh about the pin.
        let w = PuppetWarp { pins: vec![PuppetPin { src: [28.0, 30.0], dst: [28.0, 30.0], rotate: Some(90.0), depth: 0 }], ..w };
        let (solver, v, _) = deform(&s, b, &w, ITERATIONS);
        let i = (0..v.len()).max_by(|&a, &c| solver.mesh.verts[a][0].total_cmp(&solver.mesh.verts[c][0])).unwrap();
        let (x, d) = (solver.mesh.verts[i], v[i]);
        // (dx, dy) → (−dy, dx) for +90° with y down.
        let (dx, dy) = (x[0] - 28.0, x[1] - 30.0);
        assert!((d[0] - (28.0 - dy)).abs() < 1.0 && (d[1] - (30.0 + dx)).abs() < 1.0, "{x:?} → {d:?}");
    }

    #[test]
    fn arap_energy_decreases_and_modes_differ() {
        let s = blob(SampleType::U8);
        let b = s.content_bounds();
        let mesh = build_mesh(&s, b, PuppetDensity::Normal, 2.0);
        let src = [[28.0, 14.0], [64.0, 62.0], [28.0, 62.0]];
        let dst = [[10.0, 30.0], [70.0, 50.0], [28.0, 62.0]];
        let solver = PuppetSolver::new(mesh, &src, &[false; 3]);
        let rot = [None; 3];
        let (v, e) = solver.solve(&dst, &rot, PuppetMode::Rigid, 30, None);
        assert!(e.len() == 30);
        let start = solver.energy(&solver.solve(&dst, &rot, PuppetMode::Rigid, 0, None).0, &dst, &rot);
        assert!(e[0] <= start * (1.0 + 1e-9), "{} vs {start}", e[0]);
        for w in e.windows(2) {
            assert!(w[1] <= w[0] * (1.0 + 1e-6) + 1e-9, "{w:?}");
        }
        assert!(e[29] < start);
        // Pins are honoured closely.
        for (b, d) in solver.bound.iter().zip(&dst) {
            let p = (0..3).fold([0.0, 0.0], |a, k| [a[0] + b.bary[k] * v[b.verts[k]][0], a[1] + b.bary[k] * v[b.verts[k]][1]]);
            assert!(dist2(p, *d) < 0.25, "{p:?} vs {d:?}");
        }
        let (vd, _) = solver.solve(&dst, &rot, PuppetMode::Distort, 30, None);
        assert_ne!(v, vd);
    }

    #[test]
    fn depth_orders_triangles() {
        let s = blob(SampleType::U8);
        let b = s.content_bounds();
        let mesh = build_mesh(&s, b, PuppetDensity::Fewer, 2.0);
        let pins = [[28.0, 14.0], [64.0, 62.0]];
        let solver = PuppetSolver::new(mesh, &pins, &[false; 2]);
        let order = solver.draw_order(&[1, 0], &pins);
        let near0 = |t: usize| {
            let c = solver.mesh.tris[t].iter().fold([0.0, 0.0], |a, &i| [a[0] + solver.mesh.verts[i][0] / 3.0, a[1] + solver.mesh.verts[i][1] / 3.0]);
            dist2(c, pins[0]) < dist2(c, pins[1])
        };
        // Triangles nearest the raised pin come last, all others first.
        let k = order.iter().position(|&t| near0(t)).unwrap();
        assert!(order[..k].iter().all(|&t| !near0(t)) && order[k..].iter().all(|&t| near0(t)));
    }
}
