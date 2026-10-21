//! Stylize: Diffuse, Extrude, Tiles, Trace Contour, Wind.

use photocraft_geom::Rect;

mod extrude;
mod wind;

use crate::fxutil::{MAXC, luma, native, ncol, subtractive, xy};
use crate::image::Image;
use crate::noise::hash01;
use crate::{Ctx, DiffuseMode, ExtrudeType, TileFill, WindMethod};

fn read_px(src: &Image, x: i32, y: i32, buf: &mut [f32]) {
    for (c, v) in buf.iter_mut().enumerate() {
        *v = src.get(x, y, c);
    }
}

/// Diffuse: every pixel is swapped for a random neighbour (within 1 px);
/// Darken/Lighten Only accept the neighbour only when darker/lighter;
/// Anisotropic picks the most similar neighbour, softening noise.
pub(crate) fn diffuse(src: &Image, out: Rect, ctx: &Ctx, mode: DiffuseMode, seed: u32) -> Vec<f32> {
    let n = src.ch;
    let mut res = src.crop(out);
    let mut nb = [0.0f32; MAXC];
    for (i, px) in res.chunks_exact_mut(n).enumerate() {
        let (x, y) = xy(out, i);
        match mode {
            DiffuseMode::Anisotropic => {
                let me = luma(ctx, px);
                let mut best = (f32::MAX, (0, 0));
                // Random start so ties do not bias one direction.
                let start = (hash01(x, y, 3, seed) * 8.0) as usize;
                const RING: [(i32, i32); 8] = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];
                for k in 0..8 {
                    let (dx, dy) = RING[(start + k) % 8];
                    read_px(src, x + dx, y + dy, &mut nb[..n]);
                    let d = (luma(ctx, &nb[..n]) - me).abs();
                    if d < best.0 {
                        best = (d, (dx, dy));
                    }
                }
                let (dx, dy) = best.1;
                read_px(src, x + dx, y + dy, &mut nb[..n]);
                // Half-way move: anisotropic diffusion softens rather than shuffles.
                for c in 0..n {
                    px[c] = 0.5 * (px[c] + nb[c]);
                }
            }
            _ => {
                let dx = (hash01(x, y, 1, seed) * 3.0) as i32 - 1;
                let dy = (hash01(x, y, 2, seed) * 3.0) as i32 - 1;
                read_px(src, x + dx, y + dy, &mut nb[..n]);
                if ctx.alpha && nb[n - 1] <= 0.0 && px[n - 1] > 0.0 && mode != DiffuseMode::Normal {
                    continue;
                }
                let take = match mode {
                    DiffuseMode::DarkenOnly => luma(ctx, &nb[..n]) < luma(ctx, px),
                    DiffuseMode::LightenOnly => luma(ctx, &nb[..n]) > luma(ctx, px),
                    _ => true,
                };
                if take {
                    px.copy_from_slice(&nb[..n]);
                }
            }
        }
    }
    res
}

/// Extrude parameters.
pub(crate) struct ExtrudeSpec {
    pub kind: ExtrudeType,
    pub size: f32,
    pub depth: f32,
    pub level_based: bool,
    pub solid_front: bool,
    pub mask_incomplete: bool,
    pub seed: u32,
}

/// Max on-screen displacement of an extruded block's front face, px.
pub(crate) fn extrude_reach(depth: f32) -> f32 {
    depth.clamp(0.0, 255.0) * 0.25
}

/// Extrude: the bounds are cut into square cells. Blocks are pushed toward
/// the viewer (front faces shift away from the bounds' centre by their
/// height, with shaded sides); Pyramids raise four shaded triangular faces.
/// Heights are random or follow the cell's brightness.
pub(crate) fn extrude(src: &Image, out: Rect, ctx: &Ctx, spec: &ExtrudeSpec) -> Vec<f32> {
    extrude::blocks(src, out, ctx, spec).unwrap_or_else(|| extrude_direct(src, out, ctx, spec))
}

fn extrude_direct(src: &Image, out: Rect, ctx: &Ctx, spec: &ExtrudeSpec) -> Vec<f32> {
    let n = src.ch;
    let cs = spec.size.clamp(2.0, 255.0).round() as i32;
    let b = ctx.bounds;
    let (bcx, bcy) = ((b.x0 + b.x1) as f32 / 2.0, (b.y0 + b.y1) as f32 / 2.0);
    let half_diag = ((b.width() as f32).hypot(b.height() as f32) / 2.0).max(1.0);
    let reach = extrude_reach(spec.depth);
    let mut res = src.crop(out);
    let cell_rect = |gx: i32, gy: i32| Rect::new(b.x0 + gx * cs, b.y0 + gy * cs, b.x0 + (gx + 1) * cs, b.y0 + (gy + 1) * cs);
    let incomplete = |r: &Rect| r.intersect(&b) != *r;
    // Average colour and height of a cell (cached for the last cell queried per pixel row).
    let cell_info = |gx: i32, gy: i32| -> ([f32; MAXC], f32) {
        let r = cell_rect(gx, gy).intersect(&b);
        let mut acc = [0.0f32; MAXC];
        let mut cnt = 0.0;
        // Sparse sampling keeps big cells cheap; the average only colours faces.
        let step = (cs / 8).max(1);
        let mut yy = r.y0;
        while yy < r.y1 {
            let mut xx = r.x0;
            while xx < r.x1 {
                let a = if ctx.alpha { src.get(xx, yy, n - 1) } else { 1.0 };
                for (c, v) in acc.iter_mut().enumerate().take(n) {
                    let s = src.get(xx, yy, c);
                    *v += if ctx.alpha && c < n - 1 { s * a } else { s };
                }
                cnt += 1.0;
                xx += step;
            }
            yy += step;
        }
        if cnt > 0.0 {
            for v in acc.iter_mut().take(n) {
                *v /= cnt;
            }
        }
        crate::fxutil::unpremul_px(&mut acc[..n], ctx.alpha);
        let h = if spec.level_based { luma(ctx, &acc[..n]).clamp(0.0, 1.0) } else { hash01(gx, gy, 21, spec.seed) };
        (acc, h)
    };
    let shade = |px: &mut [f32], col: &[f32], k: f32| {
        let cc = ncol(ctx, n);
        let sub = subtractive(ctx);
        for c in 0..cc {
            px[c] = if sub { 1.0 - (1.0 - col[c]) * k } else { col[c] * k }.clamp(0.0, 1.0);
        }
        if ctx.alpha {
            px[n - 1] = col[n - 1];
        }
    };
    type CellInfo = ((i32, i32), ([f32; MAXC], f32));
    let mut cache: Vec<CellInfo> = Vec::with_capacity(32);
    let w = out.width() as usize;
    for (i, px) in res.chunks_exact_mut(n).enumerate() {
        let (x, y) = (out.x0 + (i % w) as i32, out.y0 + (i / w) as i32);
        if !b.contains(x, y) {
            continue;
        }
        if i % w == 0 {
            cache.clear();
        }
        let mut info = |gx: i32, gy: i32| -> ([f32; MAXC], f32) {
            if let Some((_, v)) = cache.iter().find(|(k, _)| *k == (gx, gy)) {
                return *v;
            }
            let v = cell_info(gx, gy);
            if cache.len() >= 64 {
                cache.remove(0);
            }
            cache.push(((gx, gy), v));
            v
        };
        let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
        let (gx, gy) = ((x - b.x0).div_euclid(cs), (y - b.y0).div_euclid(cs));
        match spec.kind {
            ExtrudeType::Pyramids => {
                let r = cell_rect(gx, gy);
                if spec.mask_incomplete && incomplete(&r) {
                    continue;
                }
                let (col, h) = info(gx, gy);
                let (u, v) = (fx - (r.x0 as f32 + cs as f32 / 2.0), fy - (r.y0 as f32 + cs as f32 / 2.0));
                // Face normals of a pyramid of height H over a cell of half-size s; light from the top-left.
                let hgt = h * spec.depth.clamp(1.0, 255.0) / 255.0 * cs as f32;
                let s = cs as f32 / 2.0;
                let (nx, ny) = if u.abs() >= v.abs() { (u.signum() * hgt, 0.0) } else { (0.0, v.signum() * hgt) };
                let norm = (nx * nx + ny * ny + s * s).sqrt();
                let (lx, ly, lz) = (-0.5f32, -0.5f32, std::f32::consts::FRAC_1_SQRT_2);
                let lambert = ((nx * lx + ny * ly + s * lz) / norm).max(0.0);
                shade(px, &col, 0.35 + 0.9 * lambert);
            }
            ExtrudeType::Blocks => {
                // Candidate blocks: own cell and the neighbours whose front faces can reach here.
                let k = (reach / cs as f32).ceil() as i32 + 1;
                let mut best: Option<(f32, i32, i32, u8)> = None; // (height, gx, gy, face: 0 front, 1..4 sides)
                for dy in -k..=k {
                    for dx in -k..=k {
                        let (cx, cy) = (gx + dx, gy + dy);
                        let r = cell_rect(cx, cy);
                        if r.x1 <= b.x0 || r.y1 <= b.y0 || r.x0 >= b.x1 || r.y0 >= b.y1 {
                            continue;
                        }
                        if spec.mask_incomplete && incomplete(&r) {
                            continue;
                        }
                        let h = info(cx, cy).1;
                        if best.is_some_and(|bb| bb.0 >= h) {
                            continue;
                        }
                        // Front face offset grows with height and distance from the centre.
                        let (mx, my) = (r.x0 as f32 + cs as f32 / 2.0, r.y0 as f32 + cs as f32 / 2.0);
                        let off = reach * h / half_diag;
                        let (ox, oy) = ((mx - bcx) * off, (my - bcy) * off);
                        let half = cs as f32 / 2.0;
                        // Sweep t ∈ [0, 1] from base to front: inside if some t puts the pixel in the square.
                        let interval = |p: f32, m: f32, o: f32| -> Option<(f32, f32)> {
                            // |p − (m + o t)| ≤ half
                            if o.abs() < 1e-6 {
                                return ((p - m).abs() <= half).then_some((0.0, 1.0));
                            }
                            let (a, bb) = ((p - m - half) / o, (p - m + half) / o);
                            let (lo, hi) = (a.min(bb).max(0.0), a.max(bb).min(1.0));
                            (lo <= hi).then_some((lo, hi))
                        };
                        let (Some(ix), Some(iy)) = (interval(fx, mx, ox), interval(fy, my, oy)) else { continue };
                        let (lo, hi) = (ix.0.max(iy.0), ix.1.min(iy.1));
                        if lo > hi {
                            continue;
                        }
                        let face = if hi >= 1.0 {
                            0
                        } else if ix.1 < iy.1 {
                            // Left the square through an x side first.
                            if ox > 0.0 { 1 } else { 2 }
                        } else if oy > 0.0 {
                            3
                        } else {
                            4
                        };
                        best = Some((h, cx, cy, face));
                    }
                }
                let Some((h, cx, cy, face)) = best else { continue };
                let r = cell_rect(cx, cy);
                let (mx, my) = (r.x0 as f32 + cs as f32 / 2.0, r.y0 as f32 + cs as f32 / 2.0);
                let off = reach * h / half_diag;
                let col = info(cx, cy).0;
                if face == 0 {
                    if spec.solid_front {
                        shade(px, &col, 1.0);
                    } else {
                        // Image texture shifted with the face.
                        let (sx, sy) = ((fx - (mx - bcx) * off).floor() as i32, (fy - (my - bcy) * off).floor() as i32);
                        let (sx, sy) = (sx.clamp(r.x0, r.x1 - 1), sy.clamp(r.y0, r.y1 - 1));
                        for (c, v) in px.iter_mut().enumerate() {
                            *v = src.get(sx, sy, c);
                        }
                    }
                } else {
                    // Light from the top-left: the left/top sides are lit.
                    let k = match face {
                        1 => 0.85, // left side (block shifted right)
                        2 => 0.55, // right side
                        3 => 0.95, // top side
                        _ => 0.45, // bottom side
                    };
                    shade(px, &col, k);
                }
            }
        }
    }
    res
}

/// Tiles: the bounds are cut into square tiles (`count` along the shorter
/// side), each shifted randomly by up to `max_offset` % of its size; later
/// tiles overlap earlier ones and gaps get the chosen fill.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tiles(src: &Image, out: Rect, ctx: &Ctx, count: u32, max_offset: f32, fill: TileFill, colours: ([f32; 4], [f32; 4]), seed: u32) -> Vec<f32> {
    let n = src.ch;
    let b = ctx.bounds;
    let ts = (b.width().min(b.height()) as f32 / count.clamp(1, 99) as f32).max(1.0);
    let m = max_offset.clamp(0.0, 99.0) / 100.0 * ts;
    let fg = native(ctx, colours.0);
    let bg = native(ctx, colours.1);
    let cc = ncol(ctx, n);
    let mut res = src.crop(out);
    let w = out.width() as usize;
    let offset = |gx: i32, gy: i32| ((hash01(gx, gy, 31, seed) * 2.0 - 1.0) * m, (hash01(gx, gy, 32, seed) * 2.0 - 1.0) * m);
    for (i, px) in res.chunks_exact_mut(n).enumerate() {
        let (x, y) = (out.x0 + (i % w) as i32, out.y0 + (i / w) as i32);
        if !b.contains(x, y) {
            continue;
        }
        let (fx, fy) = (x as f32 + 0.5 - b.x0 as f32, y as f32 + 0.5 - b.y0 as f32);
        let (gx, gy) = ((fx / ts).floor() as i32, (fy / ts).floor() as i32);
        // Topmost tile (drawn last = highest row-major index) covering the pixel.
        let mut hit = None;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (cx, cy) = (gx + dx, gy + dy);
                let (ox, oy) = offset(cx, cy);
                let (u, v) = (fx - ox - cx as f32 * ts, fy - oy - cy as f32 * ts);
                if u >= 0.0 && v >= 0.0 && u < ts && v < ts && (cx as f32 * ts) < b.width() as f32 && (cy as f32 * ts) < b.height() as f32 && cx >= 0 && cy >= 0
                {
                    let order = (cy, cx);
                    if hit.is_none_or(|(o, _, _)| order > o) {
                        hit = Some((order, ox, oy));
                    }
                }
            }
        }
        match hit {
            Some((_, ox, oy)) => {
                let (sx, sy) = ((x as f32 + 0.5 - ox).floor() as i32, (y as f32 + 0.5 - oy).floor() as i32);
                let (sx, sy) = (sx.clamp(b.x0, b.x1 - 1), sy.clamp(b.y0, b.y1 - 1));
                for (c, v) in px.iter_mut().enumerate() {
                    *v = src.get(sx, sy, c);
                }
            }
            None => match fill {
                TileFill::Background => px.copy_from_slice(&bg[..n]),
                TileFill::Foreground => px.copy_from_slice(&fg[..n]),
                TileFill::Inverse => {
                    for v in px.iter_mut().take(cc) {
                        *v = 1.0 - *v;
                    }
                }
                TileFill::Unaltered => {}
            },
        }
    }
    res
}

/// Trace Contour: per channel, outlines where values cross `level` (0–255)
/// with a 1 px line (dark in additive models) on a white canvas.
pub(crate) fn trace_contour(src: &Image, out: Rect, ctx: &Ctx, level: f32, upper: bool) -> Vec<f32> {
    let n = src.ch;
    let cc = ncol(ctx, n);
    let sub = subtractive(ctx);
    let bounds = ctx.bounds.intersect(&src.rect);
    let area = if bounds.is_empty() { src.rect } else { bounds };
    let t = level.clamp(0.0, 255.0) / 255.0;
    let bright = |x: i32, y: i32, c: usize| {
        let v = src.get_edge(x, y, c, crate::image::Edge::Repeat, area);
        if sub { 1.0 - v } else { v }
    };
    let mut res = src.crop(out);
    for (i, px) in res.chunks_exact_mut(n).enumerate() {
        let (x, y) = xy(out, i);
        for (c, v) in px.iter_mut().enumerate().take(cc) {
            let me = bright(x, y, c) > t;
            let line = me == upper && [(1, 0), (-1, 0), (0, 1), (0, -1)].iter().any(|(dx, dy)| (bright(x + dx, y + dy, c) > t) != me);
            let paper = if line { 0.0 } else { 1.0 };
            *v = if sub { 1.0 - paper } else { paper };
        }
    }
    res
}

/// Longest streak (or stagger shift) of a Wind method, px.
pub(crate) fn wind_reach(method: WindMethod) -> f32 {
    match method {
        WindMethod::Wind => 16.0,
        WindMethod::Blast => 40.0,
        WindMethod::Stagger => 12.0,
    }
}

/// Wind: streaks trail downwind from edges (Blast: longer and denser);
/// Stagger shifts random row segments along the wind.
pub(crate) fn wind(src: &Image, out: Rect, ctx: &Ctx, method: WindMethod, from_right: bool, seed: u32) -> Vec<f32> {
    wind::streaks(src, out, ctx, method, from_right, seed).unwrap_or_else(|| wind_direct(src, out, ctx, method, from_right, seed))
}

fn wind_direct(src: &Image, out: Rect, ctx: &Ctx, method: WindMethod, from_right: bool, seed: u32) -> Vec<f32> {
    let n = src.ch;
    let lmax = wind_reach(method) as i32;
    // Streaks move downwind: from the left they extend to the right (+x).
    let d = if from_right { -1 } else { 1 };
    let mut res = src.crop(out);
    let mut tmp = [0.0f32; MAXC];
    let mut tmp2 = [0.0f32; MAXC];
    let lum = |x: i32, y: i32, buf: &mut [f32; MAXC]| {
        read_px(src, x, y, &mut buf[..n]);
        luma(ctx, &buf[..n])
    };
    let w = out.width() as usize;
    for (i, px) in res.chunks_exact_mut(n).enumerate() {
        let (x, y) = (out.x0 + (i % w) as i32, out.y0 + (i / w) as i32);
        if method == WindMethod::Stagger {
            // Rows split into segments of 8–40 px, each shifted downwind by 0–lmax px.
            let seg_len = 8 + (hash01(0, y, 41, seed) * 32.0) as i32;
            let phase = (hash01(1, y, 42, seed) * seg_len as f32) as i32;
            let seg = (x * d + phase).div_euclid(seg_len);
            let shift = (hash01(seg, y, 43, seed).powi(2) * lmax as f32) as i32;
            read_px(src, x - d * shift, y, &mut tmp[..n]);
            px.copy_from_slice(&tmp[..n]);
            continue;
        }
        let strength = if method == WindMethod::Blast { 8.0 } else { 4.0 };
        let mut best = (0.0f32, x);
        for k in 1..=lmax {
            let s = x - d * k;
            // Each source pixel starts a streak of random length; only edges start visible ones.
            let len = (hash01(s, y, 44, seed).powi(2) * lmax as f32).max(1.0);
            if (k as f32) > len {
                continue;
            }
            let edge = ((lum(s, y, &mut tmp) - lum(s - d, y, &mut tmp2)).abs() * strength).min(1.0);
            let wgt = edge * (1.0 - k as f32 / (len + 1.0));
            if wgt > best.0 {
                best = (wgt, s);
            }
        }
        if best.0 > 0.0 {
            read_px(src, best.1, y, &mut tmp[..n]);
            for c in 0..n {
                px[c] += (tmp[c] - px[c]) * best.0;
            }
        }
    }
    res
}

#[cfg(test)]
mod tests;
