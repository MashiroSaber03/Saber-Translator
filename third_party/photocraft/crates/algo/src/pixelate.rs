//! Pixelate: Color Halftone, Crystallize, Facet, Fragment, Mezzotint, Pointillize.
//!
//! Cell layouts are anchored at the reference bounds' origin and random
//! choices hash document coordinates, so output is independent of tiling.

use photocraft_geom::Rect;

mod facet;
#[cfg(test)]
mod reference;
mod screen;
#[cfg(test)]
mod tests;

use crate::fxutil::{MAXC, cell_point, luma, native, ncol, rgba, set_rgba, subtractive, via_rgb, xy};
use crate::image::Image;
use crate::noise::hash01;
use crate::{Ctx, MezzotintType};

/// Colour Halftone: each channel is screened at its own angle into dots
/// whose area follows the channel's ink (1 − value in additive models).
pub(crate) fn color_halftone(src: &Image, out: Rect, ctx: &Ctx, max_radius: f32, angles: [f32; 4]) -> Vec<f32> {
    let n = src.ch;
    let cc = ncol(ctx, n);
    let rmax = max_radius.max(1.0);
    // Cell side such that a full-size dot just covers the cell's corners.
    let cell = rmax * std::f32::consts::SQRT_2;
    let sub = subtractive(ctx);
    let rgb = via_rgb(ctx);
    let mut res = src.crop(out);
    let b = ctx.bounds;
    let (ox, oy) = (b.x0 as f32, b.y0 as f32);
    let rot: Vec<(f32, f32)> = (0..cc.max(3)).map(|c| angles[c.min(3)].to_radians().sin_cos()).collect();
    let clip = b.intersect(&src.rect);
    if clip.is_empty() {
        return res;
    }
    // Channel value at a point averaged over five taps (for Lab, of the RGB conversion).
    let value = |x: f32, y: f32, c: usize| -> f32 {
        let d = cell * 0.25;
        let mut acc = 0.0;
        let mut px = [0.0f32; MAXC];
        for (dx, dy) in [(0.0, 0.0), (-d, -d), (d, -d), (-d, d), (d, d)] {
            // Clamp to the bounds so cells straddling the edge read real pixels.
            let (ix, iy) = (((x + dx).floor() as i32).clamp(clip.x0, clip.x1 - 1), ((y + dy).floor() as i32).clamp(clip.y0, clip.y1 - 1));
            if rgb {
                for (k, p) in px.iter_mut().enumerate().take(n) {
                    *p = src.get(ix, iy, k);
                }
                acc += rgba(ctx, &px[..n])[c];
            } else {
                acc += src.get(ix, iy, c);
            }
        }
        acc / 5.0
    };
    let w = out.width() as usize;
    let chans = if rgb { 3 } else { cc };
    // Keep cache metadata on the stack; only the optional, bounded tables allocate.
    let mut screens: [Option<screen::Screen>; MAXC] = std::array::from_fn(|c| {
        if c >= chans {
            return None;
        }
        let &(s, co) = rot.get(c)?;
        screen::Screen::new(out, (ox, oy), (s, co), cell)
    });
    let mut vals = [0.0f32; MAXC];
    for (i, px) in res.chunks_exact_mut(n).enumerate() {
        let (x, y) = (out.x0 + (i % w) as i32, out.y0 + (i / w) as i32);
        let (fx, fy) = (x as f32 + 0.5 - ox, y as f32 + 0.5 - oy);
        for (c, v) in vals.iter_mut().enumerate().take(chans) {
            let (s, co) = rot[c];
            // Rotate into screen space.
            let (u, w2) = (fx * co + fy * s, -fx * s + fy * co);
            let (cu, cv) = ((u / cell).floor(), (w2 / cell).floor());
            let mut cov: f32 = 0.0;
            for dj in -1..=1 {
                for di in -1..=1 {
                    let (mu, mv) = ((cu + di as f32 + 0.5) * cell, (cv + dj as f32 + 0.5) * cell);
                    let d = ((u - mu).powi(2) + (w2 - mv).powi(2)).sqrt();
                    if d >= rmax + 0.5 {
                        continue;
                    }
                    // Back to document space to read the cell's value.
                    let (dx, dy) = (mu * co - mv * s + ox, mu * s + mv * co + oy);
                    // The five source taps and ink-to-radius mapping depend only on
                    // the screen cell, not on the output pixel inside that cell.
                    let radius = || {
                        let raw = value(dx, dy, c).clamp(0.0, 1.0);
                        let ink = if sub && !rgb { raw } else { 1.0 - raw };
                        let touch = std::f32::consts::FRAC_PI_4;
                        if ink <= touch {
                            cell * (ink / std::f32::consts::PI).sqrt()
                        } else {
                            cell * (0.5 + (ink - touch) / (1.0 - touch) * (std::f32::consts::FRAC_1_SQRT_2 - 0.5))
                        }
                    };
                    let r = match screens.get_mut(c).and_then(Option::as_mut) {
                        Some(screen) => screen.radius(cu + di as f32, cv + dj as f32, radius),
                        None => radius(),
                    };
                    // Tiny dots cannot cover more than their own area.
                    cov = cov.max((r - d + 0.5).clamp(0.0, 1.0).min(std::f32::consts::PI * r * r));
                }
            }
            *v = if sub && !rgb { cov } else { 1.0 - cov };
        }
        if rgb {
            let a = if ctx.alpha { px[n - 1] } else { 1.0 };
            set_rgba(ctx, px, [vals[0], vals[1], vals[2], a]);
        } else {
            px[..cc].copy_from_slice(&vals[..cc]);
        }
    }
    res
}

/// Crystallize: Voronoi cells around jittered points, each filled with the colour at its
/// point, as Photoshop's are: cells stay solid, so a black-and-white image (a layer mask
/// being distressed) keeps hard edges and its weight instead of greying where cells straddle
/// an edge. A cell whose point falls outside the image takes the (premultiplied) average
/// colour of its pixels.
pub(crate) fn crystallize(src: &Image, out: Rect, ctx: &Ctx, cell_size: f32, seed: u32) -> Vec<f32> {
    let n = src.ch;
    let cell = cell_size.max(2.0);
    let b = ctx.bounds;
    let origin = (b.x0 as f32, b.y0 as f32);
    // Cells whose territory can reach `out`, and the window holding all their pixels.
    let c0 = (((out.x0 - b.x0) as f32 / cell).floor() as i32 - 2, ((out.y0 - b.y0) as f32 / cell).floor() as i32 - 2);
    let c1 = (((out.x1 - b.x0) as f32 / cell).floor() as i32 + 2, ((out.y1 - b.y0) as f32 / cell).floor() as i32 + 2);
    let (gw, gh) = ((c1.0 - c0.0 + 1) as usize, (c1.1 - c0.1 + 1) as usize);
    let owner = |x: i32, y: i32| -> (i32, i32) {
        let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
        let (cx, cy) = (((fx - origin.0) / cell).floor() as i32, ((fy - origin.1) / cell).floor() as i32);
        let mut best = (f32::MAX, (cx, cy));
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (px, py) = cell_point(cx + dx, cy + dy, cell, seed, origin);
                let d = (px - fx).powi(2) + (py - fy).powi(2);
                if d < best.0 {
                    best = (d, (cx + dx, cy + dy));
                }
            }
        }
        best.1
    };
    let mut acc = vec![0.0f32; gw * gh * (n + 1)];
    let win = Rect::new(
        b.x0 + (c0.0 as f32 * cell).floor() as i32,
        b.y0 + (c0.1 as f32 * cell).floor() as i32,
        b.x0 + ((c1.0 + 1) as f32 * cell).ceil() as i32,
        b.y0 + ((c1.1 + 1) as f32 * cell).ceil() as i32,
    )
    .intersect(&src.rect)
    .intersect(&b);
    let mut owners = vec![(0i32, 0i32); out.width() as usize * out.height() as usize];
    for y in win.y0..win.y1 {
        for x in win.x0..win.x1 {
            let (cx, cy) = owner(x, y);
            if out.contains(x, y) {
                owners[((y - out.y0) as usize) * out.width() as usize + (x - out.x0) as usize] = (cx, cy);
            }
            if cx < c0.0 || cy < c0.1 || cx > c1.0 || cy > c1.1 {
                continue;
            }
            let k = ((cy - c0.1) as usize * gw + (cx - c0.0) as usize) * (n + 1);
            let a = if ctx.alpha { src.get(x, y, n - 1) } else { 1.0 };
            for c in 0..n {
                let s = src.get(x, y, c);
                acc[k + c] += if ctx.alpha && c < n - 1 { s * a } else { s };
            }
            acc[k + n] += 1.0;
        }
    }
    // Pixels of `out` outside the bounds keep their own value; resolve owners for any missed.
    let mut res = src.crop(out);
    for (i, px) in res.chunks_exact_mut(n).enumerate() {
        let (x, y) = xy(out, i);
        if !b.contains(x, y) {
            continue;
        }
        let (cx, cy) = if win.contains(x, y) { owners[i] } else { owner(x, y) };
        let (sx, sy) = cell_point(cx, cy, cell, seed, origin);
        let (sx, sy) = (sx.floor() as i32, sy.floor() as i32);
        if src.rect.contains(sx, sy) && b.contains(sx, sy) {
            for (c, v) in px.iter_mut().enumerate() {
                *v = src.get(sx, sy, c);
            }
            continue;
        }
        if cx < c0.0 || cy < c0.1 || cx > c1.0 || cy > c1.1 {
            continue;
        }
        let k = ((cy - c0.1) as usize * gw + (cx - c0.0) as usize) * (n + 1);
        let cnt = acc[k + n];
        if cnt <= 0.0 {
            continue;
        }
        for c in 0..n {
            px[c] = acc[k + c] / cnt;
        }
        crate::fxutil::unpremul_px(px, ctx.alpha);
    }
    res
}

/// Facet: a Kuwahara filter (radius 2) — each pixel takes the mean of the
/// least-varied of its four overlapping quadrants, clumping similar colours
/// into flat facets while keeping edges.
pub(crate) fn facet(src: &Image, out: Rect, ctx: &Ctx) -> Vec<f32> {
    facet::run(src, out, ctx).unwrap_or_else(|| facet_direct(src, out, ctx))
}

fn facet_direct(src: &Image, out: Rect, ctx: &Ctx) -> Vec<f32> {
    let n = src.ch;
    let r = 2i32;
    let mut res = src.crop(out);
    let mut tmp = [0.0f32; MAXC];
    for (i, px) in res.chunks_exact_mut(n).enumerate() {
        let (x, y) = xy(out, i);
        let mut best = (f32::MAX, [0.0f32; MAXC]);
        for (qx, qy) in [(-r, -r), (0, -r), (-r, 0), (0, 0)] {
            let mut sum = [0.0f32; MAXC];
            let (mut s, mut s2) = (0.0f32, 0.0f32);
            for yy in y + qy..=y + qy + r {
                for xx in x + qx..=x + qx + r {
                    for (c, t) in tmp.iter_mut().enumerate().take(n) {
                        *t = src.get(xx, yy, c);
                        sum[c] += *t;
                    }
                    let l = luma(ctx, &tmp[..n]);
                    s += l;
                    s2 += l * l;
                }
            }
            let cnt = ((r + 1) * (r + 1)) as f32;
            let var = s2 / cnt - (s / cnt).powi(2);
            if var < best.0 {
                for v in sum.iter_mut() {
                    *v /= cnt;
                }
                best = (var, sum);
            }
        }
        px.copy_from_slice(&best.1[..n]);
    }
    res
}

/// Fragment: the average of four copies offset by 4 px (left, right, up, down).
pub(crate) fn fragment(src: &Image, out: Rect, ctx: &Ctx) -> Vec<f32> {
    let n = src.ch;
    let mut res = Vec::with_capacity(out.width() as usize * out.height() as usize * n);
    let mut acc = [0.0f32; MAXC];
    for y in out.y0..out.y1 {
        for x in out.x0..out.x1 {
            acc[..n].fill(0.0);
            for (dx, dy) in [(-4, 0), (4, 0), (0, -4), (0, 4)] {
                let a = if ctx.alpha { src.get(x + dx, y + dy, n - 1) } else { 1.0 };
                for (c, v) in acc.iter_mut().enumerate().take(n) {
                    let s = src.get(x + dx, y + dy, c);
                    *v += if ctx.alpha && c < n - 1 { s * a } else { s } * 0.25;
                }
            }
            crate::fxutil::unpremul_px(&mut acc[..n], ctx.alpha);
            res.extend_from_slice(&acc[..n]);
        }
    }
    res
}

/// Random threshold for Mezzotint pattern `kind` at a pixel.
fn mezzo_threshold(kind: MezzotintType, x: i32, y: i32, c: u32, seed: u32) -> f32 {
    let lines = |len: i32| {
        let phase = (hash01(0, y, c + 40, seed) * len as f32) as i32;
        hash01((x + phase).div_euclid(len), y, c, seed)
    };
    let strokes = |len: i32| {
        let d = x - y;
        let phase = (hash01(d, 0, c + 50, seed) * len as f32) as i32;
        hash01((x + y + phase).div_euclid(len), d, c + 60, seed)
    };
    match kind {
        MezzotintType::FineDots => hash01(x, y, c, seed),
        MezzotintType::MediumDots => hash01(x.div_euclid(2), y.div_euclid(2), c, seed),
        MezzotintType::GrainyDots => 0.5 * hash01(x, y, c, seed) + 0.5 * hash01(x.div_euclid(3), y.div_euclid(3), c + 7, seed),
        MezzotintType::CoarseDots => hash01(x.div_euclid(3), y.div_euclid(3), c, seed),
        MezzotintType::ShortLines => lines(6),
        MezzotintType::MediumLines => lines(14),
        MezzotintType::LongLines => lines(32),
        MezzotintType::ShortStrokes => strokes(5),
        MezzotintType::MediumStrokes => strokes(10),
        MezzotintType::LongStrokes => strokes(20),
    }
}

/// Mezzotint: every colour channel is thresholded against a random pattern,
/// giving pure channel values (fully saturated dots/lines in colour images).
pub(crate) fn mezzotint(src: &Image, out: Rect, ctx: &Ctx, kind: MezzotintType, seed: u32) -> Vec<f32> {
    let n = src.ch;
    let cc = ncol(ctx, n);
    let rgb = via_rgb(ctx);
    let mut res = src.crop(out);
    for (i, px) in res.chunks_exact_mut(n).enumerate() {
        let (x, y) = xy(out, i);
        if rgb {
            let mut c = rgba(ctx, px);
            for (k, v) in c.iter_mut().enumerate().take(3) {
                *v = if *v > mezzo_threshold(kind, x, y, k as u32, seed) { 1.0 } else { 0.0 };
            }
            set_rgba(ctx, px, c);
        } else {
            for (k, v) in px.iter_mut().enumerate().take(cc) {
                *v = if *v > mezzo_threshold(kind, x, y, k as u32, seed) { 1.0 } else { 0.0 };
            }
        }
    }
    res
}

/// Pointillize: randomly sized dots, coloured from the image at their centre,
/// on a canvas of the background colour.
pub(crate) fn pointillize(src: &Image, out: Rect, ctx: &Ctx, cell_size: f32, seed: u32, background: [f32; 4]) -> Vec<f32> {
    let n = src.ch;
    let cell = cell_size.max(2.0);
    let b = ctx.bounds;
    let origin = (b.x0 as f32, b.y0 as f32);
    let bg = native(ctx, background);
    let mut res = src.crop(out);
    let mut dot = [0.0f32; MAXC];
    for (i, px) in res.chunks_exact_mut(n).enumerate() {
        let (x, y) = xy(out, i);
        let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
        let (cx, cy) = (((fx - origin.0) / cell).floor() as i32, ((fy - origin.1) / cell).floor() as i32);
        // The dot whose edge the pixel is deepest inside wins.
        let mut best: Option<(f32, i32, i32, f32, f32)> = None;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (gx, gy) = (cx + dx, cy + dy);
                let (px0, py0) = cell_point(gx, gy, cell, seed, origin);
                let rad = cell * (0.45 + 0.35 * hash01(gx, gy, 13, seed));
                let d = ((px0 - fx).powi(2) + (py0 - fy).powi(2)).sqrt();
                let depth = rad - d;
                if depth > -0.5 && best.is_none_or(|bb| depth > bb.0) {
                    best = Some((depth, gx, gy, px0, py0));
                }
            }
        }
        let a_orig = if ctx.alpha { px[n - 1] } else { 1.0 };
        // Canvas first, then the dot over it with an anti-aliased edge.
        let mut canvas = [0.0f32; MAXC];
        canvas[..n].copy_from_slice(&bg[..n]);
        if ctx.alpha {
            canvas[n - 1] = a_orig.max(bg[n - 1].min(1.0));
        }
        if let Some((depth, gx, gy, px0, py0)) = best {
            let (sx, sy) = (px0.floor() as i32, py0.floor() as i32);
            for (c, d) in dot.iter_mut().enumerate().take(n) {
                *d = src.get(sx, sy, c);
            }
            // Slight per-dot brightness jitter, as in pointillist strokes.
            let j = (hash01(gx, gy, 17, seed) - 0.5) * 0.08;
            let cc = ncol(ctx, n);
            let sub = subtractive(ctx);
            for d in dot.iter_mut().take(cc) {
                *d = (*d + if sub { -j } else { j }).clamp(0.0, 1.0);
            }
            let k = (depth + 0.5).clamp(0.0, 1.0);
            for c in 0..n {
                canvas[c] += (dot[c] - canvas[c]) * k;
            }
        }
        px.copy_from_slice(&canvas[..n]);
    }
    res
}
