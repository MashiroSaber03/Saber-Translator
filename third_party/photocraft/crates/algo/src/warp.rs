//! Mesh warps (Edit › Transform › Warp, smart-object warps): resampling a surface through any
//! forward map, such as a [`photocraft_geom::warp::Warp`] (bicubic Bezier patches or a preset).
//!
//! The source rectangle is subdivided into a fine grid whose vertices are mapped forward; each
//! cell's two triangles are rasterized in destination space with the source position
//! interpolated barycentrically, then the source is sampled there (premultiplied, nearest /
//! bilinear / Photoshop's Keys bicubic, like Free Transform). Cells are a few source pixels across, so
//! the piecewise-affine inverse is within a small fraction of a pixel of the true one; affine
//! maps (the identity included) are reproduced exactly. Where the warp shrinks the source, each
//! pixel instead averages a grid of bilinear subsamples over its source footprint (about one per
//! source pixel), so thin lines don't alias. Where the warp folds over itself the
//! later (lower-right) cell wins. Work is split per destination tile (parallel on native).

use photocraft_color::PixelFormat;
use photocraft_geom::Rect;
use photocraft_raster::Surface;

use crate::transform::{Interp, bicubic};

/// Grid cell size (source px) used to approximate the map.
const CELL: f64 = 4.0;
/// Most cells along one axis (bounds memory and time on huge layers).
const MAX_CELLS: usize = 512;
/// Most subsamples per axis when a warp shrinks the source; also the longest source step per
/// destination pixel the filter follows (bounds the footprint each tile reads).
const MAX_SS: f64 = 8.0;

/// Places a smart object's source image (`src`, its pixels in `src_rect`) in the document:
/// through `warp` (source space) and then the affine `t` (source → document) in one resampling
/// pass, an exact shift for whole-pixel translations (so conversions and re-renders are
/// lossless), bicubic otherwise. Shared by the engine's re-render and PSD export's filter cache.
pub fn place_source(
    src: &Surface,
    src_rect: Rect,
    t: &photocraft_geom::Affine,
    warp: Option<&photocraft_geom::warp::Warp>,
) -> Result<Surface, photocraft_raster::AllocationError> {
    let [a, b, c, d, e, f] = t.m;
    if let Some(w) = warp.filter(|w| !w.is_identity()) {
        let map = |x: f64, y: f64| {
            let (u, v) = w.map(x, y);
            (a * u + c * v + e, b * u + d * v + f)
        };
        return Ok(warp_mesh_surface(src, src_rect, &map, Interp::Bicubic));
    }
    let near = |x: f64, y: f64| (x - y).abs() < 1e-9;
    if near(a, 1.0) && near(b, 0.0) && near(c, 0.0) && near(d, 1.0) && near(e, e.round()) && near(f, f.round()) {
        return Ok(crate::resample::translate_surface(src, e.round() as i32, f.round() as i32));
    }
    crate::transform::warp_surface(src, src_rect, &crate::transform::Homography([a, c, e, b, d, f, 0.0, 0.0, 1.0]), Interp::Bicubic)
}

/// [`place_source`] through a projective map `h` (source → document; Distort, Perspective): the
/// warp first (in source space), then `h`.
pub fn place_source_projective(
    src: &Surface,
    src_rect: Rect,
    h: &crate::transform::Homography,
    warp: Option<&photocraft_geom::warp::Warp>,
) -> Result<Surface, photocraft_raster::AllocationError> {
    if let Some(w) = warp.filter(|w| !w.is_identity()) {
        let map = |x: f64, y: f64| {
            let (u, v) = w.map(x, y);
            h.apply(u, v)
        };
        return Ok(warp_mesh_surface(src, src_rect, &map, Interp::Bicubic));
    }
    crate::transform::warp_surface(src, src_rect, h, Interp::Bicubic)
}

/// Warps the content of `src` inside `src_rect` through the forward map `f` (source document
/// coordinates → destination document coordinates). Output has `src`'s format with alpha
/// added; pixels outside the warped area are transparent.
pub fn warp_mesh_surface(src: &Surface, src_rect: Rect, f: &(dyn Fn(f64, f64) -> (f64, f64) + Sync), interp: Interp) -> Surface {
    if src_rect.is_empty() {
        let fmt = src.format();
        return Surface::new(PixelFormat::new(fmt.mode, fmt.sample, true));
    }
    let (sw, sh) = (f64::from(src_rect.width()), f64::from(src_rect.height()));
    let nx = ((sw / CELL).ceil() as usize).clamp(1, MAX_CELLS);
    let ny = ((sh / CELL).ceil() as usize).clamp(1, MAX_CELLS);
    // Forward-mapped vertices (and their source positions).
    let mut verts = Vec::with_capacity((nx + 1) * (ny + 1));
    for j in 0..=ny {
        for i in 0..=nx {
            let sx = f64::from(src_rect.x0) + sw * i as f64 / nx as f64;
            let sy = f64::from(src_rect.y0) + sh * j as f64 / ny as f64;
            let (dx, dy) = f(sx, sy);
            verts.push(([dx, dy], [sx, sy]));
        }
    }
    let w1 = nx + 1;
    let mut tris = Vec::with_capacity(nx * ny * 2);
    for j in 0..ny {
        for i in 0..nx {
            let a = j * w1 + i;
            tris.push([a, a + 1, a + w1 + 1]);
            tris.push([a, a + w1 + 1, a + w1]);
        }
    }
    warp_triangles(src, src_rect, &verts, &tris, interp)
}

/// Rasterizes textured triangles: each vertex is `(destination, source)` in document pixels and
/// each triangle samples `src` (restricted to `src_rect`) through the barycentric
/// interpolation of its source positions. Later triangles in `tris` draw over earlier ones (the
/// order is the depth order). Output has `src`'s format with alpha added.
pub fn warp_triangles(src: &Surface, src_rect: Rect, verts: &[([f64; 2], [f64; 2])], tris: &[[usize; 3]], interp: Interp) -> Surface {
    let mut fmt = src.format();
    let converted;
    let src = if fmt.alpha {
        src
    } else {
        fmt = PixelFormat::new(fmt.mode, fmt.sample, true);
        converted = src.convert(fmt);
        &converted
    };
    let mut out = Surface::new(fmt);
    if src_rect.is_empty() || tris.is_empty() {
        return out;
    }
    if verts.iter().any(|v| !v.0[0].is_finite() || !v.0[1].is_finite()) {
        return out;
    }
    let lim = f64::from(1 << 20);
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for v in tris.iter().flatten().map(|&i| &verts[i]) {
        x0 = x0.min(v.0[0]);
        y0 = y0.min(v.0[1]);
        x1 = x1.max(v.0[0]);
        y1 = y1.max(v.0[1]);
    }
    let dst = Rect::new(x0.max(-lim).floor() as i32 - 1, y0.max(-lim).floor() as i32 - 1, x1.min(lim).ceil() as i32 + 1, y1.min(lim).ceil() as i32 + 1);
    // Triangles: (vertex indices), binned per destination tile.
    let tiles: Vec<Rect> = dst.tiles().map(|tc| tc.rect().intersect(&dst)).filter(|r| !r.is_empty()).collect();
    let mut bins: Vec<Vec<[usize; 3]>> = vec![Vec::new(); tiles.len()];
    let ts = photocraft_geom::TILE_SIZE;
    // Bin origin on the tile grid (the first tile is clipped to `dst`, so its x0/y0 may not be).
    let tx0 = tiles.iter().map(|t| t.x0).min().unwrap_or(0).div_euclid(ts) * ts;
    let ty0 = tiles.iter().map(|t| t.y0).min().unwrap_or(0).div_euclid(ts) * ts;
    let cols = tiles.iter().map(|t| (t.x0 - tx0) / ts).max().unwrap_or(0) as usize + 1;
    let grid_of = |t: &Rect| ((t.y0 - ty0) / ts) as usize * cols + ((t.x0 - tx0) / ts) as usize;
    let mut lookup = vec![usize::MAX; cols * (tiles.iter().map(|t| (t.y0 - ty0) / ts).max().unwrap_or(0) as usize + 1)];
    for (k, t) in tiles.iter().enumerate() {
        lookup[grid_of(t)] = k;
    }
    for &tri in tris {
        let ps = tri.map(|v| verts[v].0);
        let bx0 = ps.iter().map(|p| p[0]).fold(f64::MAX, f64::min).floor() as i32;
        let by0 = ps.iter().map(|p| p[1]).fold(f64::MAX, f64::min).floor() as i32;
        let bx1 = ps.iter().map(|p| p[0]).fold(f64::MIN, f64::max).ceil() as i32;
        let by1 = ps.iter().map(|p| p[1]).fold(f64::MIN, f64::max).ceil() as i32;
        let (cx0, cy0) = ((bx0.max(dst.x0) - tx0).div_euclid(ts), (by0.max(dst.y0) - ty0).div_euclid(ts));
        let (cx1, cy1) = ((bx1.min(dst.x1 - 1) - tx0).div_euclid(ts), (by1.min(dst.y1 - 1) - ty0).div_euclid(ts));
        for cy in cy0.max(0)..=cy1 {
            for cx in cx0.max(0)..=cx1 {
                let g = cy as usize * cols + cx as usize;
                if let Some(&k) = lookup.get(g)
                    && k != usize::MAX
                {
                    bins[k].push(tri);
                }
            }
        }
    }
    let n = fmt.channels();
    let a = n - 1;
    let work = |(t, tris): (&Rect, &Vec<[usize; 3]>)| -> Option<(Rect, Vec<f32>)> {
        if tris.is_empty() {
            return None;
        }
        let w = t.width() as usize;
        let h = t.height() as usize;
        // Subsamples along one destination axis, from the source step `j` it makes: one where
        // the warp keeps or enlarges the source, else about one per source pixel the output pixel
        // spans (an area average, as the transform path gets from its pre-reduction; a single
        // bicubic sample aliases thin lines when shrinking). The subsamples are bilinear: the
        // box does the filtering, and it is ~4x cheaper than bicubic at no loss.
        let count = |j: [f64; 2]| if interp == Interp::Nearest { 1 } else { j[0].hypot(j[1]).round().clamp(1.0, MAX_SS) as usize };
        // Source position per destination pixel (NaN = not covered) and the triangle it came
        // from. Per triangle: its source step per destination pixel along x and y (`[dx_u, dx_v,
        // dy_u, dy_v]`, the inverse map's Jacobian, constant over a triangle; zero when
        // point-sampled) and the subsamples along each.
        let mut uv = vec![[f64::NAN; 2]; w * h];
        let mut owner = vec![0u32; w * h];
        let mut jac: Vec<([f64; 4], usize, usize)> = Vec::with_capacity(tris.len());
        for (k, tri) in tris.iter().enumerate() {
            let [(p0, s0), (p1, s1), (p2, s2)] = tri.map(|v| verts[v]);
            let det = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]);
            if det.abs() < 1e-12 {
                jac.push(([0.0; 4], 1, 1));
                continue;
            }
            let (d1, d2) = ([s1[0] - s0[0], s1[1] - s0[1]], [s2[0] - s0[0], s2[1] - s0[1]]);
            let (l1x, l1y, l2x, l2y) = ((p2[1] - p0[1]) / det, (p0[0] - p2[0]) / det, (p0[1] - p1[1]) / det, (p1[0] - p0[0]) / det);
            // Capped, so a sliver triangle can't ask for an unbounded footprint.
            let cap = |j: [f64; 2]| {
                let l = j[0].hypot(j[1]);
                if l > MAX_SS { [j[0] * MAX_SS / l, j[1] * MAX_SS / l] } else { j }
            };
            let jx = cap([l1x * d1[0] + l2x * d2[0], l1x * d1[1] + l2x * d2[1]]);
            let jy = cap([l1y * d1[0] + l2y * d2[0], l1y * d1[1] + l2y * d2[1]]);
            let (nx, ny) = (count(jx), count(jy));
            jac.push(if nx * ny == 1 { ([0.0; 4], 1, 1) } else { ([jx[0], jx[1], jy[0], jy[1]], nx, ny) });
            let bx0 = (p0[0].min(p1[0]).min(p2[0]) - 0.5).floor().max(f64::from(t.x0)) as i32;
            let by0 = (p0[1].min(p1[1]).min(p2[1]) - 0.5).floor().max(f64::from(t.y0)) as i32;
            let bx1 = (p0[0].max(p1[0]).max(p2[0]) + 0.5).ceil().min(f64::from(t.x1)) as i32;
            let by1 = (p0[1].max(p1[1]).max(p2[1]) + 0.5).ceil().min(f64::from(t.y1)) as i32;
            let eps = -1e-9;
            for y in by0..by1 {
                let py = f64::from(y) + 0.5;
                for x in bx0..bx1 {
                    let px = f64::from(x) + 0.5;
                    let l1 = ((px - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (py - p0[1])) / det;
                    let l2 = ((p1[0] - p0[0]) * (py - p0[1]) - (px - p0[0]) * (p1[1] - p0[1])) / det;
                    let l0 = 1.0 - l1 - l2;
                    if l0 < eps || l1 < eps || l2 < eps {
                        continue;
                    }
                    let o = (y - t.y0) as usize * w + (x - t.x0) as usize;
                    uv[o] = [l0 * s0[0] + l1 * s1[0] + l2 * s2[0], l0 * s0[1] + l1 * s1[1] + l2 * s2[1]];
                    owner[o] = k as u32;
                }
            }
        }
        // Source footprint of the covered pixels, including the spread of their subsamples.
        let (mut fx0, mut fy0, mut fx1, mut fy1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for (q, j) in uv.iter().zip(&owner).filter(|(q, _)| !q[0].is_nan()).map(|(q, &k)| (q, jac[k as usize].0)) {
            let (ex, ey) = (0.5 * (j[0].abs() + j[2].abs()), 0.5 * (j[1].abs() + j[3].abs()));
            fx0 = fx0.min(q[0] - ex);
            fy0 = fy0.min(q[1] - ey);
            fx1 = fx1.max(q[0] + ex);
            fy1 = fy1.max(q[1] + ey);
        }
        if fx0 > fx1 {
            return None;
        }
        let foot = Rect::new(fx0.floor() as i32 - 3, fy0.floor() as i32 - 3, fx1.ceil() as i32 + 3, fy1.ceil() as i32 + 3).intersect(&src_rect);
        if foot.is_empty() || !src.has_tiles_in(foot) {
            return None;
        }
        let mut px = src.read_region(foot);
        for p in px.chunks_exact_mut(n) {
            let al = p[a];
            for v in &mut p[..a] {
                *v *= al;
            }
        }
        let fw = foot.width() as usize;
        let at = |x: i32, y: i32, c: usize| -> f64 {
            if x < foot.x0 || y < foot.y0 || x >= foot.x1 || y >= foot.y1 {
                0.0
            } else {
                f64::from(px[((y - foot.y0) as usize * fw + (x - foot.x0) as usize) * n + c])
            }
        };
        let mut outp = vec![0.0f32; w * h * n];
        let mut any = false;
        let mut acc = [0.0f64; 8];
        let mut one = [0.0f64; 8];
        // One premultiplied sample at source position (u, v) (pixel centres at integers).
        let point = |interp: Interp, u: f64, v: f64, acc: &mut [f64; 8]| {
            acc[..n].fill(0.0);
            match interp {
                Interp::Nearest => {
                    let (ix, iy) = ((u + 0.5).floor() as i32, (v + 0.5).floor() as i32);
                    for (c, s) in acc.iter_mut().enumerate().take(n) {
                        *s = at(ix, iy, c);
                    }
                }
                Interp::Bilinear => {
                    let (ix, iy) = (u.floor() as i32, v.floor() as i32);
                    let (fx, fy) = (u - f64::from(ix), v - f64::from(iy));
                    for (dy, wy) in [(0, 1.0 - fy), (1, fy)] {
                        for (dx, wx) in [(0, 1.0 - fx), (1, fx)] {
                            for (c, s) in acc.iter_mut().enumerate().take(n) {
                                *s += at(ix + dx, iy + dy, c) * wx * wy;
                            }
                        }
                    }
                }
                Interp::Bicubic => {
                    // Snap to the pixel grid when within float noise, so identity is exact.
                    let snap = |t: f64| if (t - t.round()).abs() < 1e-7 { t.round() } else { t };
                    let (u, v) = (snap(u), snap(v));
                    let (ix, iy) = (u.floor() as i32, v.floor() as i32);
                    let (wx, wy) = (bicubic(u - f64::from(ix)), bicubic(v - f64::from(iy)));
                    for (j, wyj) in wy.iter().enumerate() {
                        for (i, wxi) in wx.iter().enumerate() {
                            let k = wxi * wyj;
                            if k == 0.0 {
                                continue;
                            }
                            for (c, s) in acc.iter_mut().enumerate().take(n) {
                                *s += at(ix - 1 + i as i32, iy - 1 + j as i32, c) * k;
                            }
                        }
                    }
                }
            }
        };
        for (o, (q, &k)) in uv.iter().zip(&owner).enumerate() {
            if q[0].is_nan() {
                continue;
            }
            let (j, nx, ny) = jac[k as usize];
            if nx * ny == 1 {
                point(interp, q[0] - 0.5, q[1] - 0.5, &mut acc);
            } else {
                acc[..n].fill(0.0);
                for sy in 0..ny {
                    let t = (sy as f64 + 0.5) / ny as f64 - 0.5;
                    for sx in 0..nx {
                        let s = (sx as f64 + 0.5) / nx as f64 - 0.5;
                        point(Interp::Bilinear, q[0] + s * j[0] + t * j[2] - 0.5, q[1] + s * j[1] + t * j[3] - 0.5, &mut one);
                        for (c, v) in acc.iter_mut().zip(&one).take(n) {
                            *c += v;
                        }
                    }
                }
                let k = 1.0 / (nx * ny) as f64;
                for c in acc.iter_mut().take(n) {
                    *c *= k;
                }
            }
            let al = acc[a].clamp(0.0, 1.0);
            // Float noise (a vertex a hair off the pixel grid) must not leave invisible
            // colour behind in transparent pixels.
            if al <= 1e-6 {
                continue;
            }
            any = true;
            let b = o * n;
            for c in 0..a {
                outp[b + c] = (acc[c] / al).clamp(0.0, 1.0) as f32;
            }
            outp[b + a] = al as f32;
        }
        any.then_some((*t, outp))
    };
    #[cfg(not(target_arch = "wasm32"))]
    let done: Vec<(Rect, Vec<f32>)> = {
        use rayon::prelude::*;
        tiles.par_iter().zip(bins.par_iter()).filter_map(work).collect()
    };
    #[cfg(target_arch = "wasm32")]
    let done: Vec<(Rect, Vec<f32>)> = tiles.iter().zip(bins.iter()).filter_map(work).collect();
    for (r, v) in done {
        out.write_region(r, &v);
    }
    out.prune();
    out
}

/// [`warp_mesh_surface`] for single-channel surfaces whose outside reads as their default value
/// (layer masks, selections): the content is warped with alpha and flattened back onto the
/// default.
pub fn warp_mesh_gray(s: &Surface, f: &(dyn Fn(f64, f64) -> (f64, f64) + Sync), interp: Interp) -> Surface {
    let default = s.default_pixel().first().copied().unwrap_or(0.0);
    let fmt = s.format();
    let src = s.content_bounds();
    let mut out = Surface::with_default(fmt, &[default]);
    if src.is_empty() {
        return out;
    }
    let with_alpha = PixelFormat::new(fmt.mode, fmt.sample, true);
    let mut tmp = Surface::new(with_alpha);
    let v = s.read_region(src);
    tmp.write_region(src, &v.iter().flat_map(|g| [*g, 1.0]).collect::<Vec<f32>>());
    let w = warp_mesh_surface(&tmp, src, f, interp);
    // Clear the old region and write the warped one separately, as `warp_gray` does: one dense
    // `src ∪ warped` buffer grows with the square of how far a mesh moves the content (#995).
    // `Surface` is tile-sparse, so a distant region costs only its own tiles.
    out.write_region(src, &vec![default; src.width() as usize * src.height() as usize]);
    let b = w.content_bounds();
    if !b.is_empty() {
        let px = w.read_region(b);
        let flat: Vec<f32> = px.as_chunks::<2>().0.iter().map(|p| p[0] * p[1] + default * (1.0 - p[1])).collect();
        out.write_region(b, &flat);
    }
    out.prune();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::{ColorMode, SampleType};
    use photocraft_geom::warp::{BezierMesh, Warp, WarpStyle};

    fn sample(s: SampleType) -> Surface {
        let mut surf = Surface::new(PixelFormat::new(ColorMode::Rgb, s, true));
        for y in 0..24 {
            for x in 0..40 {
                let v = ((x * 7 + y * 13) % 17) as f32 / 16.0;
                surf.write_pixel(x + 5, y + 3, &[v, 1.0 - v, (x as f32) / 40.0, 1.0]);
            }
        }
        surf
    }

    #[test]
    fn a_mesh_warp_far_off_canvas_writes_only_the_two_regions() {
        // #995: the old and warped regions were cleared as one dense `src ∪ warped` buffer, about
        // 37 GiB of floats for a 48 px selection moved 100000 px. Same fix as `warp_gray` (b70fbd4).
        for sample in [SampleType::U8, SampleType::F32] {
            let mut sel = Surface::new(PixelFormat::new(ColorMode::Grayscale, sample, false));
            sel.fill_rect(Rect::new(8, 8, 56, 56), &[1.0]);
            for d in [100_000.0, -100_000.0, 300.0] {
                let out = warp_mesh_gray(&sel, &|x, y| (x + d, y + d), Interp::Bilinear);
                let k = d as i32;
                assert_eq!(out.content_bounds(), Rect::new(8 + k, 8 + k, 56 + k, 56 + k), "{sample:?} {d}");
                assert_eq!(out.sample_channel(30 + k, 30 + k, 0), 1.0, "{sample:?} {d}: moved");
                assert_eq!(out.sample_channel(30, 30, 0), 0.0, "{sample:?} {d}: old region cleared");
            }
        }
    }

    #[test]
    fn identity_warp_is_a_no_op_at_every_depth() {
        for s in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let src = sample(s);
            let r = src.content_bounds();
            let b = [f64::from(r.x0), f64::from(r.y0), f64::from(r.x1), f64::from(r.y1)];
            for w in [Warp::none(b), Warp::custom(BezierMesh::identity(b, 1, 1), b), Warp::preset(WarpStyle::Arc, 0.0, b)] {
                for interp in [Interp::Nearest, Interp::Bilinear, Interp::Bicubic] {
                    let out = warp_mesh_surface(&src, r, &|x, y| w.map(x, y), interp);
                    assert_eq!(out.content_bounds(), r, "{s:?} {interp:?}");
                    let (a, o) = (src.read_region(r), out.read_region(r));
                    let worst = a.iter().zip(&o).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
                    assert!(worst <= 1.0 / 255.0, "{s:?} {interp:?}: {worst}");
                }
            }
        }
    }

    #[test]
    fn translation_and_known_arc_points() {
        let src = sample(SampleType::U8);
        let r = src.content_bounds();
        let out = warp_mesh_surface(&src, r, &|x, y| (x + 10.0, y - 2.0), Interp::Bilinear);
        assert_eq!(out.content_bounds(), r.translate(10, -2));
        assert_eq!(out.pixel(20, 10), src.pixel(10, 12));
        // Arc 50 % (Photoshop): the bottom corners keep their pixels, the top rises and spreads.
        let b = [f64::from(r.x0), f64::from(r.y0), f64::from(r.x1), f64::from(r.y1)];
        let w = Warp::preset(WarpStyle::Arc, 50.0, b);
        let arc = warp_mesh_surface(&src, r, &|x, y| w.map(x, y), Interp::Bilinear);
        let corner = w.map(f64::from(r.x0), f64::from(r.y1));
        assert!((corner.0 - f64::from(r.x0)).abs() < 1e-9 && (corner.1 - f64::from(r.y1)).abs() < 1e-9, "{corner:?}");
        let (ex, ey) = w.map(f64::from(r.x0) + 2.5, f64::from(r.y0) + 2.5);
        let e = arc.pixel(ex.floor() as i32, ey.floor() as i32);
        assert!(e[3] > 0.2, "a point near the top-left corner lands at its mapped position: {e:?}");
        let c = arc.content_bounds();
        assert!(c.y0 < r.y0 - 2 && c.x0 < r.x0 && c.x1 > r.x1, "the top rises and spreads past the box: {c:?}");
    }

    #[test]
    fn a_shrinking_mesh_warp_is_prefiltered() {
        // A warped smart object placed at ~1/5 size: a grid of 1 px lines every 4 px must come
        // out its area average (9/16 white) everywhere, not point-sampled moiré (crinkled,
        // doubled or missing lines), like the affine path's `large_downscale_is_prefiltered`.
        for s in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let (w, h) = (400, 320);
            let mut src = Surface::new(PixelFormat::new(ColorMode::Rgb, s, true));
            let px: Vec<f32> =
                (0..h).flat_map(|y| (0..w).map(move |x| if x % 4 == 0 || y % 4 == 0 { 0.0 } else { 1.0 })).flat_map(|v| [v, v, v, 1.0]).collect();
            let r = Rect::new(0, 0, w, h);
            src.write_region(r, &px);
            let warp = Warp::preset(WarpStyle::Arc, 15.0, [0.0, 0.0, f64::from(w), f64::from(h)]);
            let t = photocraft_geom::Affine { m: [0.2, 0.0, 0.0, 0.2, 7.0, 5.0] };
            let out = place_source(&src, r, &t, Some(&warp)).unwrap();
            let b = out.content_bounds();
            let o = out.read_region(b);
            let opaque: Vec<f32> = o.as_chunks::<4>().0.iter().filter(|p| p[3] >= 0.999).map(|p| p[0]).collect();
            assert!(opaque.len() > 2000, "{s:?}: {} opaque pixels", opaque.len());
            let worst = opaque.iter().map(|v| (v - 0.5625).abs()).fold(0.0f32, f32::max);
            // An area average over the ~5 px footprint still covers one or two lines per axis
            // (up to 0.2 off); point sampling lands wholly on or off a line (up to 0.56 off).
            assert!(worst < 0.25, "{s:?}: worst deviation from the area average {worst}");
        }
    }

    #[test]
    fn gray_masks_keep_their_default() {
        let mut m = Surface::with_default(PixelFormat::GRAY8, &[1.0]);
        m.fill_rect(Rect::new(4, 4, 10, 10), &[0.0]);
        let o = warp_mesh_gray(&m, &|x, y| (x + 5.0, y), Interp::Bilinear);
        assert_eq!(o.pixel(0, 0), vec![1.0]);
        assert_eq!(o.pixel(12, 6), vec![0.0]);
        assert_eq!(o.pixel(5, 6), vec![1.0]);
    }
}
