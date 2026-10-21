//! Pixel work of the Filter Gallery ([`crate::artistic`]).
//!
//! Every effect runs on an sRGB working copy of the tile's input window (straight colour; the
//! source alpha is kept) and may read up to [`effect_reach`] pixels around each output pixel,
//! so a stack of effects reads the sum of their reaches. Noise, strokes and textures are
//! functions of document coordinates, so results do not depend on the tiling.

// Effects walk several per-pixel planes in lockstep; index loops read clearest.
#![allow(clippy::needless_range_loop)]

use std::f32::consts::PI;

use crate::photo_util::{par_rows, par_rows2};

use photocraft_geom::Rect;

use crate::Ctx;
use crate::artistic::GalleryEffect;
use crate::artistic::GalleryFilter as F;
use crate::fxutil::{box_blur_n, cell_point, gauss_blur_n, gauss_box_radii, rgba, set_rgba};
use crate::image::Image;
use crate::noise::hash01;

type Px = [f32; 3];

/// The working window: sRGB colour of every pixel of `r`.
pub(crate) struct Win {
    r: Rect,
    w: usize,
    h: usize,
    c: Vec<Px>,
}

impl Win {
    #[inline]
    fn xy(&self, i: usize) -> (i32, i32) {
        (self.r.x0 + (i % self.w) as i32, self.r.y0 + (i / self.w) as i32)
    }
    fn lum(&self) -> Vec<f32> {
        self.c.iter().map(lum).collect()
    }
    /// Bilinear sample at document coordinates (pixel centres at .5), clamped to the window.
    fn sample(&self, x: f32, y: f32) -> Px {
        let fx = (x - 0.5 - self.r.x0 as f32).clamp(0.0, (self.w - 1) as f32);
        let fy = (y - 0.5 - self.r.y0 as f32).clamp(0.0, (self.h - 1) as f32);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (ax, ay) = (fx - x0 as f32, fy - y0 as f32);
        let g = |xx: usize, yy: usize| self.c[yy * self.w + xx];
        let top = mix(g(x0, y0), g(x1, y0), ax);
        let bot = mix(g(x0, y1), g(x1, y1), ax);
        mix(top, bot, ay)
    }
    /// The pixel nearest to document `(x, y)`, clamped to the window.
    fn at(&self, x: i32, y: i32) -> Px {
        let xx = (x - self.r.x0).clamp(0, self.w as i32 - 1) as usize;
        let yy = (y - self.r.y0).clamp(0, self.h as i32 - 1) as usize;
        self.c[yy * self.w + xx]
    }
}

/// Hermite step from `e0` to `e1`; `e0 > e1` gives the falling step.
#[inline]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    if (e1 - e0).abs() < 1e-9 {
        return if x >= e1 { 1.0 } else { 0.0 };
    }
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn lum(p: &Px) -> f32 {
    0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]
}
#[inline]
fn mix(a: Px, b: Px, t: f32) -> Px {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}
#[inline]
fn map(p: Px, f: impl Fn(f32) -> f32) -> Px {
    [f(p[0]), f(p[1]), f(p[2])]
}
#[inline]
fn grey(v: f32) -> Px {
    [v, v, v]
}
#[inline]
fn rgb(c: [f32; 4]) -> Px {
    [c[0], c[1], c[2]]
}
#[inline]
fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}
#[inline]
fn posterize(v: f32, levels: f32) -> f32 {
    let l = (levels - 1.0).max(1.0);
    (clamp01(v) * l).round() / l
}
#[inline]
fn saturate(p: Px, k: f32) -> Px {
    let l = lum(&p);
    map(p, |v| l + (v - l) * k)
}

/// Pixels a Gaussian blur of `sigma` reads (three box passes per axis).
fn gs(sigma: f32) -> i32 {
    if sigma < 0.3 { 0 } else { gauss_box_radii(sigma).iter().sum::<usize>() as i32 + 1 }
}

fn blur1(p: &mut [f32], w: usize, h: usize, sigma: f32) {
    gauss_blur_n(p, w, h, 1, sigma);
}
fn blur3(c: &mut [Px], w: usize, h: usize, sigma: f32) {
    gauss_blur_n(c.as_flattened_mut(), w, h, 3, sigma);
}

/// Sobel gradient (per pixel, edges clamped) of a plane.
fn sobel(p: &[f32], w: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
    let at = |x: isize, y: isize| p[(y.clamp(0, h as isize - 1) as usize) * w + x.clamp(0, w as isize - 1) as usize];
    let mut gx = vec![0.0; w * h];
    let mut gy = vec![0.0; w * h];
    par_rows2(&mut gx, &mut gy, w, |y, gxr, gyr| {
        let y = y as isize;
        for x in 0..w as isize {
            let xi = x as usize;
            gxr[xi] = ((at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2.0 * at(x - 1, y) + at(x - 1, y + 1))) / 8.0;
            gyr[xi] = ((at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2.0 * at(x, y - 1) + at(x + 1, y - 1))) / 8.0;
        }
    });
    (gx, gy)
}

fn grad_mag(p: &[f32], w: usize, h: usize) -> Vec<f32> {
    let (gx, gy) = sobel(p, w, h);
    gx.iter().zip(&gy).map(|(a, b)| (a * a + b * b).sqrt()).collect()
}

/// Running-mean blur of radius `r` steps along direction `d` ((1,0), (0,1), (1,1) or (1,-1))
/// of an interleaved `n`-channel buffer.
fn line_blur(buf: &mut [f32], n: usize, w: usize, h: usize, d: (i32, i32), r: usize) {
    if r == 0 || w == 0 || h == 0 {
        return;
    }
    let (wi, hi) = (w as i32, h as i32);
    let inside = |x: i32, y: i32| x >= 0 && y >= 0 && x < wi && y < hi;
    let norm = 1.0 / (2 * r + 1) as f64;
    let mut idx = Vec::new();
    let mut vals = Vec::new();
    for sy in 0..hi {
        for sx in 0..wi {
            if inside(sx - d.0, sy - d.1) {
                continue; // not the start of a line
            }
            idx.clear();
            let (mut x, mut y) = (sx, sy);
            while inside(x, y) {
                idx.push((y as usize * w + x as usize) * n);
                x += d.0;
                y += d.1;
            }
            let m = idx.len() as isize;
            for c in 0..n {
                vals.clear();
                vals.extend(idx.iter().map(|&o| buf[o + c] as f64));
                let at = |k: isize| vals[k.clamp(0, m - 1) as usize];
                let mut acc: f64 = (-(r as isize)..=r as isize).map(at).sum();
                for (k, &o) in idx.iter().enumerate() {
                    buf[o + c] = (acc * norm) as f32;
                    acc += at(k as isize + r as isize + 1) - at(k as isize - r as isize);
                }
            }
        }
    }
}

fn line_blur3(c: &mut [Px], w: usize, h: usize, d: (i32, i32), r: usize) {
    line_blur(c.as_flattened_mut(), 3, w, h, d, r);
}

fn direction(name: &str) -> (i32, i32) {
    match name {
        "horizontal" => (1, 0),
        "vertical" => (0, 1),
        "leftDiagonal" => (1, 1),
        _ => (1, -1), // rightDiagonal: bottom-left to top-right
    }
}

/// Kuwahara smoothing (the flattest of four quadrant means): the painterly base of several
/// Artistic filters. Box means make the cost independent of `r`.
fn kuwahara(win: &mut Win, r: usize) {
    let h2 = r.div_ceil(2).max(1);
    let (w, h) = (win.w, win.h);
    let mut m: Vec<f32> = Vec::with_capacity(w * h * 5);
    for p in &win.c {
        let l = lum(p);
        m.extend_from_slice(&[p[0], p[1], p[2], l, l * l]);
    }
    box_blur_n(&mut m, w, h, 5, h2);
    let o = h2 as isize;
    let get = |x: isize, y: isize| {
        let i = ((y.clamp(0, h as isize - 1) as usize) * w + x.clamp(0, w as isize - 1) as usize) * 5;
        &m[i..i + 5]
    };
    par_rows(&mut win.c, w, 1, |y, row| {
        let y = y as isize;
        for x in 0..w as isize {
            let mut best = f32::MAX;
            let mut col = [0.0; 3];
            for (dx, dy) in [(-o, -o), (o, -o), (-o, o), (o, o)] {
                let q = get(x + dx, y + dy);
                let var = q[4] - q[3] * q[3];
                if var < best {
                    best = var;
                    col = [q[0], q[1], q[2]];
                }
            }
            row[x as usize] = col;
        }
    });
}

fn kuwahara_reach(r: usize) -> i32 {
    2 * r.div_ceil(2).max(1) as i32 + 1
}

fn unsharp(win: &mut Win, sigma: f32, amount: f32) {
    if amount <= 0.0 {
        return;
    }
    let mut b = win.c.clone();
    blur3(&mut b, win.w, win.h, sigma);
    for (p, q) in win.c.iter_mut().zip(&b) {
        *p = [p[0] + (p[0] - q[0]) * amount, p[1] + (p[1] - q[1]) * amount, p[2] + (p[2] - q[2]) * amount];
    }
}

/// Smooth value noise (0–1) on a unit lattice.
fn vnoise(x: f32, y: f32, seed: u32) -> f32 {
    let (ix, iy) = (x.floor() as i32, y.floor() as i32);
    let (tx, ty) = (x - ix as f32, y - iy as f32);
    let (sx, sy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
    let h = |a: i32, b: i32| hash01(a, b, 0, seed);
    let top = h(ix, iy) + (h(ix + 1, iy) - h(ix, iy)) * sx;
    let bot = h(ix, iy + 1) + (h(ix + 1, iy + 1) - h(ix, iy + 1)) * sx;
    top + (bot - top) * sy
}

/// Three-octave value noise (0–1).
fn fbm(x: f32, y: f32, seed: u32) -> f32 {
    (vnoise(x, y, seed) * 0.57 + vnoise(x * 2.03, y * 2.03, seed ^ 0x51) * 0.29 + vnoise(x * 4.1, y * 4.1, seed ^ 0xa3) * 0.14) / 1.0
}

/// Stroke texture: value noise stretched `len` px along the direction `d` and `wid` px across.
fn strokes(x: i32, y: i32, d: (i32, i32), len: f32, wid: f32, seed: u32) -> f32 {
    let (dx, dy) = (d.0 as f32, d.1 as f32);
    let l = (dx * dx + dy * dy).sqrt().max(1e-6);
    let (ux, uy) = (dx / l, dy / l);
    let (x, y) = (x as f32, y as f32);
    let u = x * ux + y * uy;
    let v = -x * uy + y * ux;
    vnoise(u / len.max(0.5), v / wid.max(0.3), seed)
}

/// Repaints in discrete strokes `len` px long and `wid` px wide along `d`: every stroke takes the
/// colour of `src` at its middle, and neighbouring strokes start at staggered points, so the
/// image reads as separate strokes rather than a smear. Also returns where the pixel lies along
/// its stroke (0 at the start, 1 at the end). Reads up to `len / 2 + 1` px away.
fn paint_stroke(src: &Win, x: i32, y: i32, d: (i32, i32), len: f32, wid: f32, seed: u32) -> (Px, f32) {
    let (dx, dy) = (d.0 as f32, d.1 as f32);
    let l = (dx * dx + dy * dy).sqrt().max(1e-6);
    let (ux, uy) = (dx / l, dy / l);
    let (len, wid) = (len.max(1.0), wid.max(0.5));
    let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
    let u = fx * ux + fy * uy;
    let v = -fx * uy + fy * ux;
    let lane = (v / wid).floor() as i32;
    let off = hash01(lane, 0, 0, seed) * len;
    let t = ((u + off) / len).rem_euclid(1.0);
    // Back to the stroke's middle along the direction.
    let du = (0.5 - t) * len;
    (src.sample(fx + du * ux, fy + du * uy), t)
}

/// Unit vector toward the light in image coordinates (y down).
fn light_dir(name: &str) -> (f32, f32) {
    let s = std::f32::consts::FRAC_1_SQRT_2;
    match name {
        "bottom" => (0.0, 1.0),
        "bottomLeft" => (-s, s),
        "left" => (-1.0, 0.0),
        "topLeft" => (-s, -s),
        "topRight" => (s, -s),
        "right" => (1.0, 0.0),
        "bottomRight" => (s, s),
        _ => (0.0, -1.0),
    }
}

/// Lambert shading factor (1 = flat) of a height field with gradient `(gx, gy)`.
#[inline]
fn shade(gx: f32, gy: f32, light: (f32, f32), k: f32) -> f32 {
    let (nx, ny, nz) = (-k * gx, -k * gy, 1.0);
    let nl = (nx * nx + ny * ny + nz * nz).sqrt();
    let (lx, ly, lz) = (light.0, light.1, 1.0);
    let ll = (lx * lx + ly * ly + lz * lz).sqrt();
    let d = (nx * lx + ny * ly + nz * lz) / (nl * ll);
    d / (lz / ll)
}

/// Procedural textures (0–1 heights) at document coordinates; `s` scales them (1 = 100 %).
fn texture(kind: &str, x: f32, y: f32, s: f32) -> f32 {
    let (x, y) = (x / s, y / s);
    match kind {
        "brick" => {
            let (bw, bh) = (24.0, 12.0);
            let row = (y / bh).floor();
            let xo = x + if row as i32 % 2 == 0 { 0.0 } else { bw / 2.0 };
            let (u, v) = (xo.rem_euclid(bw), y.rem_euclid(bh));
            let d = u.min(bw - u).min(v.min(bh - v));
            smoothstep(0.6, 2.2, d) * (0.85 + 0.15 * vnoise(x / 3.0, y / 3.0, 91))
        }
        "burlap" => weave(x, y, 7.0, 0.35),
        "canvas" => weave(x, y, 3.5, 0.18),
        "sandstone" => fbm(x / 7.0, y / 7.0, 93) * 0.7 + hash01(x as i32, y as i32, 0, 94) * 0.3,
        "blocks" => {
            let b = 18.0;
            let (u, v) = (x.rem_euclid(b) / b, y.rem_euclid(b) / b);
            (u.min(1.0 - u).min(v.min(1.0 - v)) * 4.0).min(1.0)
        }
        "frosted" => fbm(x / 2.5, y / 2.5, 95),
        "tinyLens" => {
            let p = 12.0;
            let (u, v) = ((x.rem_euclid(p) / p - 0.5) * 2.0, (y.rem_euclid(p) / p - 0.5) * 2.0);
            (1.0 - (u * u + v * v)).max(0.0).sqrt()
        }
        _ => 0.5,
    }
}

/// Basket weave of threads `p` px wide with `noise` irregularity.
fn weave(x: f32, y: f32, p: f32, noise: f32) -> f32 {
    let (cx, cy) = ((x / p).floor() as i32, (y / p).floor() as i32);
    let over_h = (cx + cy).rem_euclid(2) == 0;
    let (u, v) = (x.rem_euclid(p) / p, y.rem_euclid(p) / p);
    // A thread is a rounded ridge across its cell.
    let ridge = if over_h { (v * PI).sin() } else { (u * PI).sin() };
    let n = vnoise(x / (p * 0.7), y / (p * 2.0), 97);
    ridge * (1.0 - noise) + n * noise
}

/// Texturizer core: relief-shades the window with a procedural texture.
fn texturize(win: &mut Win, kind: &str, scaling: f32, relief: f32, light: &str, invert: bool) {
    texturize_by(win, kind, scaling, relief, light, invert, |_| 1.0);
}

/// [`texturize`] with the relief scaled per pixel by `weight` of its colour (0–1).
fn texturize_by(win: &mut Win, kind: &str, scaling: f32, relief: f32, light: &str, invert: bool, weight: impl Fn(&Px) -> f32) {
    if relief <= 0.0 {
        return;
    }
    let s = (scaling / 100.0).clamp(0.5, 2.0);
    let l = light_dir(light);
    let k = relief / 50.0 * 3.0;
    let sign = if invert { -1.0 } else { 1.0 };
    for i in 0..win.c.len() {
        let (x, y) = win.xy(i);
        let (fx, fy) = (x as f32, y as f32);
        let gx = (texture(kind, fx + 1.0, fy, s) - texture(kind, fx - 1.0, fy, s)) * 0.5 * sign;
        let gy = (texture(kind, fx, fy + 1.0, s) - texture(kind, fx, fy - 1.0, s)) * 0.5 * sign;
        let f = 1.0 + (shade(gx, gy, l, k).clamp(0.0, 2.0) - 1.0) * weight(&win.c[i]);
        win.c[i] = map(win.c[i], |v| v * f);
    }
}

/// Closest and second-closest jittered cell points (distances and the closest point).
fn voronoi(x: f32, y: f32, cell: f32, seed: u32) -> (f32, f32, (f32, f32)) {
    let (cx, cy) = ((x / cell).floor() as i32, (y / cell).floor() as i32);
    let (mut d1, mut d2, mut p1) = (f32::MAX, f32::MAX, (x, y));
    for j in -1..=1 {
        for i in -1..=1 {
            let p = cell_point(cx + i, cy + j, cell, seed, (0.0, 0.0));
            let d = ((p.0 - x).powi(2) + (p.1 - y).powi(2)).sqrt();
            if d < d1 {
                d2 = d1;
                d1 = d;
                p1 = p;
            } else if d < d2 {
                d2 = d;
            }
        }
    }
    (d1, d2, p1)
}

/// Brightens highlights (Film Grain / Smudge Stick "Highlight Area" and "Intensity").
fn highlights(p: Px, l: f32, area: f32, intensity: f32) -> Px {
    if area <= 0.0 {
        return p;
    }
    let m = smoothstep(1.0 - area / 20.0 * 0.75, 1.02, l) * (1.0 - intensity / 10.0 * 0.6);
    map(p, |v| v + (1.0 - v) * m)
}

/// Pixels an effect reads around an output pixel.
pub(crate) fn effect_reach(e: &GalleryEffect) -> i32 {
    let g = |k: &str| e.get(k);
    match e.filter {
        F::ColoredPencil => 2,
        F::Cutout => gs(cutout_sigma(e)),
        F::DryBrush | F::Fresco => kuwahara_reach(1 + g("brushSize") as usize) + gs(1.0) + 2,
        F::FilmGrain | F::DiffuseGlow | F::ChalkCharcoal | F::Reticulation | F::Grain | F::MosaicTiles | F::Texturizer => 0,
        F::RoughPastels => gs(0.6 + 1.2) + ((2.0 + g("strokeLength")) / 2.0) as i32 + 3,
        F::NeonGlow => gs(g("glowSize").abs() * 0.6 + 0.5),
        F::PaintDaubs => kuwahara_reach(daubs_radius(e)) + gs(1.5) + 1,
        F::PaletteKnife => kuwahara_reach((g("strokeSize") / 3.0).max(1.0) as usize) + gs(g("softness") * 0.5),
        F::PlasticWrap => gs(plastic_sigma(e)) + 2,
        F::PosterEdges => gs(0.6 + g("edgeThickness") * 0.6) + 2,
        F::SmudgeStick => (g("strokeLength") * 1.5) as i32 + 2,
        F::Sponge => kuwahara_reach(1 + g("brushSize") as usize),
        F::Underpainting => gs(0.5 + g("brushSize") * 0.4),
        F::Watercolor => kuwahara_reach(watercolor_radius(e)) + 2,
        F::AccentedEdges => gs(0.4 + g("smoothness") * 0.3) + gs(g("edgeWidth") * 0.5) + 2,
        F::AngledStrokes => gs(0.8) + (g("strokeLength").max(3.0) / 2.0) as i32 + gs(1.0) + 3,
        F::Crosshatch => (g("strokeLength") / 3.0) as i32 + gs(1.0) + 2,
        F::DarkStrokes => 9,
        F::InkOutlines => (g("strokeLength") / 2.0) as i32 + 3,
        F::Spatter => (g("sprayRadius") * 1.5).ceil() as i32 + 2,
        F::SprayedStrokes => g("sprayRadius").ceil() as i32 + (g("strokeLength") / 2.0) as i32 + 3,
        F::SumiE => gs(g("strokeWidth") * 0.2) + (g("strokeWidth") / 2.0) as i32 + 2,
        F::Glass => (g("distortion") * 2.0).ceil() as i32 + 2,
        F::OceanRipple => (g("rippleMagnitude") * 0.9).ceil() as i32 + 2,
        F::BasRelief => gs(basrelief_sigma(e)) + 2,
        F::Charcoal => gs(1.0) + 2,
        F::Chrome => gs(1.0 + g("smoothness") * 0.4) + 2,
        F::ConteCrayon => 0,
        F::GraphicPen => (g("strokeLength") / 2.0) as i32 + 2,
        F::HalftonePattern => gs(0.5 + g("size") * 0.3),
        F::NotePaper => gs(1.0) + gs(1.2) + 2,
        F::Photocopy => gs(1.0 + g("detail") * 0.5),
        F::Plaster => gs(0.5 + g("smoothness") * 0.8) + gs(1.5) + 2,
        F::Stamp => gs(0.3 + g("smoothness") * 0.3),
        F::TornEdges => gs(1.0),
        F::WaterPaper => 2 * (g("fiberLength") / 6.0) as i32 + 2,
        F::GlowingEdges => gs(0.3 + g("smoothness") * 0.25) + gs(g("edgeWidth") * 0.4) + 2,
        F::Craquelure => 0,
        F::Patchwork => patch_size(e).ceil() as i32 + 2,
        F::StainedGlass => (glass_cell(e) * 2.5).ceil() as i32 + 2,
    }
}

fn cutout_sigma(e: &GalleryEffect) -> f32 {
    0.5 + e.get("edgeSimplicity") * (1.1 - 0.3 * e.get("edgeFidelity"))
}
fn daubs_radius(e: &GalleryEffect) -> usize {
    let wide = matches!(e.choice("brushType"), "wideSharp" | "wideBlurry");
    let r = e.get("brushSize") / 2.0 * if wide { 1.5 } else { 1.0 };
    r.max(1.0) as usize
}
fn plastic_sigma(e: &GalleryEffect) -> f32 {
    0.5 + e.get("smoothness") * 0.5 + (15.0 - e.get("detail")) * 0.2
}
fn watercolor_radius(e: &GalleryEffect) -> usize {
    1 + ((15.0 - e.get("brushDetail")) / 2.5) as usize
}
fn basrelief_sigma(e: &GalleryEffect) -> f32 {
    0.3 + e.get("smoothness") * 0.5 + (15.0 - e.get("detail")) * 0.12
}
fn patch_size(e: &GalleryEffect) -> f32 {
    3.0 + e.get("squareSize") * 1.5
}
fn glass_cell(e: &GalleryEffect) -> f32 {
    e.get("cellSize") * 1.6 + 1.0
}

/// Total reach of a stack of effects.
pub(crate) fn reach(effects: &[GalleryEffect]) -> i32 {
    effects.iter().map(effect_reach).sum()
}

/// Runs a stack of effects over `out` (the source covers `out` grown by [`reach`]).
pub(crate) fn run(effects: &[GalleryEffect], src: &Image, out: Rect, ctx: &Ctx) -> Vec<f32> {
    let n = src.ch;
    let r = src.rect;
    let (w, h) = (r.width() as usize, r.height() as usize);
    if effects.is_empty() || w == 0 || h == 0 {
        return src.crop(out);
    }
    let mut win = Win { r, w, h, c: src.data.chunks_exact(n).map(|px| rgb(rgba(ctx, px))).collect() };
    for e in effects {
        apply(e, &mut win, ctx);
    }
    let mut res = src.crop(out);
    let ow = out.width() as usize;
    for (i, px) in res.chunks_exact_mut(n).enumerate() {
        let (x, y) = (out.x0 + (i % ow) as i32, out.y0 + (i / ow) as i32);
        let a = if ctx.alpha { px[n - 1] } else { 1.0 };
        let c = win.at(x, y);
        set_rgba(ctx, px, [clamp01(c[0]), clamp01(c[1]), clamp01(c[2]), a]);
        if ctx.alpha {
            px[n - 1] = a;
        }
    }
    res
}

fn apply(e: &GalleryEffect, win: &mut Win, ctx: &Ctx) {
    let g = |k: &str| e.get(k);
    let (w, h) = (win.w, win.h);
    let (fg, bg) = (rgb(e.foreground), rgb(e.background));
    match e.filter {
        // ---------------- Artistic ----------------
        F::ColoredPencil => {
            let pw = g("pencilWidth");
            let pressure = g("strokePressure") / 15.0;
            let paper = map(bg, |v| v * (0.7 + 0.3 * g("paperBrightness") / 50.0));
            let l = win.lum();
            let e2 = grad_mag(&l, w, h);
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let d = 1.0 - l[i];
                let h1 = strokes(x, y, (1, 1), 5.0 + pw * 3.0, 0.5 + pw * 0.45, 11);
                let h2 = strokes(x, y, (1, -1), 5.0 + pw * 3.0, 0.5 + pw * 0.45, 12);
                let hatch = h1.max(h2 * smoothstep(0.3, 0.8, d));
                let cov = clamp01(d * (0.55 + pressure) + e2[i] * 3.0 - (1.0 - hatch) * 0.8 + 0.25);
                win.c[i] = mix(paper, map(win.c[i], |v| v * 0.88), cov);
            }
        }
        F::Cutout => {
            blur3(&mut win.c, w, h, cutout_sigma(e));
            let levels = g("numberOfLevels").round();
            for p in &mut win.c {
                // Flat tones per luminance level, keeping each region's hue.
                let l = lum(p).max(1e-4);
                let k = posterize(l, levels) / l;
                *p = map(*p, |v| posterize(clamp01(v * k), levels * 2.0));
            }
        }
        F::DryBrush | F::Fresco => {
            let fresco = e.filter == F::Fresco;
            kuwahara(win, 1 + g("brushSize") as usize);
            unsharp(win, 1.0, if fresco { 0.9 } else { 0.5 });
            let levels = 3.0 + g("brushDetail");
            let amp = 0.02 + (g("texture") - 1.0) * 0.05;
            let l = if fresco { grad_mag(&win.lum(), w, h) } else { Vec::new() };
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let n = (hash01(x, y, 0, 21) - 0.5) * amp;
                let mut p = map(win.c[i], |v| posterize(v, levels) * 0.6 + v * 0.4 + n);
                if fresco {
                    p = map(p, |v| clamp01((v - 0.5) * 1.3 + 0.45).powf(1.15) * (1.0 - clamp01(l[i] * 2.5) * 0.6));
                }
                win.c[i] = p;
            }
        }
        F::FilmGrain => {
            let grain = g("grain") / 20.0;
            let (area, inten) = (g("highlightArea"), g("intensity"));
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let l = lum(&win.c[i]);
                let n = (hash01(x, y, 0, 31) - 0.5) * grain * 0.5 * (0.3 + 2.8 * l * (1.0 - l));
                win.c[i] = highlights(map(win.c[i], |v| v + n), l, area, inten);
            }
        }
        F::NeonGlow => {
            let size = g("glowSize");
            let br = g("glowBrightness") / 50.0;
            let glow = rgb(e.color);
            let l = win.lum();
            let mut m: Vec<f32> = if size >= 0.0 { l.clone() } else { l.iter().map(|v| 1.0 - v).collect() };
            blur1(&mut m, w, h, size.abs() * 0.6 + 0.5);
            for i in 0..win.c.len() {
                let base = mix(fg, bg, l[i]);
                let k = clamp01(m[i] * 1.2) * br * 2.0;
                win.c[i] = [0, 1, 2].map(|c| 1.0 - (1.0 - base[c]) * (1.0 - clamp01(glow[c] * k)));
            }
        }
        F::PaintDaubs => {
            let ty = e.choice("brushType");
            kuwahara(win, daubs_radius(e));
            let sharp = g("sharpness") / 40.0;
            match ty {
                "wideBlurry" => blur3(&mut win.c, w, h, 1.5),
                "wideSharp" => unsharp(win, 1.5, sharp * 4.0),
                _ => unsharp(win, 1.2, sharp * 2.5),
            }
            if matches!(ty, "lightRough" | "darkRough" | "sparkle") {
                let edges = if ty == "sparkle" { grad_mag(&win.lum(), w, h) } else { Vec::new() };
                for i in 0..win.c.len() {
                    let (x, y) = win.xy(i);
                    let n = fbm(x as f32 / 2.0, y as f32 / 2.0, 41) - 0.5;
                    win.c[i] = match ty {
                        "lightRough" => map(win.c[i], |v| v + n.max(0.0) * 0.35),
                        "darkRough" => map(win.c[i], |v| v * (1.0 - n.max(0.0) * 0.7)),
                        _ => map(win.c[i], |v| v + clamp01(edges[i] * 4.0) * smoothstep(0.75, 0.95, hash01(x, y, 0, 42)) * 0.8),
                    };
                }
            }
        }
        F::PaletteKnife => {
            kuwahara(win, (g("strokeSize") / 3.0).max(1.0) as usize);
            let levels = 2.0 + g("strokeDetail") * 3.0;
            for p in &mut win.c {
                *p = map(*p, |v| posterize(v, levels) * 0.5 + v * 0.5);
            }
            blur3(&mut win.c, w, h, g("softness") * 0.5);
        }
        F::PlasticWrap => {
            let hs = g("highlightStrength") / 20.0;
            let mut l = win.lum();
            blur1(&mut l, w, h, plastic_sigma(e));
            let (gx, gy) = sobel(&l, w, h);
            let lv = normalize3([-0.45, -0.65, 1.0]);
            let hv = normalize3([lv[0], lv[1], lv[2] + 1.0]);
            for i in 0..win.c.len() {
                let k = 14.0;
                let n = normalize3([-k * gx[i], -k * gy[i], 1.0]);
                let ndh = (n[0] * hv[0] + n[1] * hv[1] + n[2] * hv[2]).max(0.0);
                let tilt = 1.0 - n[2];
                let spec = ndh.powf(30.0) * smoothstep(0.01, 0.12, tilt) + smoothstep(0.25, 0.6, tilt) * 0.25;
                win.c[i] = map(win.c[i], |v| v * 0.88 + spec * hs * 1.3);
            }
        }
        F::PosterEdges => {
            let mut l = win.lum();
            blur1(&mut l, w, h, 0.6 + g("edgeThickness") * 0.6);
            let edge = grad_mag(&l, w, h);
            let inten = g("edgeIntensity") / 10.0;
            let levels = g("posterization") + 2.0;
            for i in 0..win.c.len() {
                let ink = smoothstep(0.015, 0.06, edge[i] * (1.0 + inten * 3.0)) * (0.55 + inten * 0.45);
                win.c[i] = map(win.c[i], |v| posterize(v, levels) * (1.0 - ink));
            }
        }
        F::RoughPastels => {
            // Chalk strokes up to the right; Stroke Detail keeps more of the image inside them.
            let len = g("strokeLength");
            let det = (g("strokeDetail") - 1.0) / 19.0;
            if len >= 1.0 {
                let mut soft = win.c.clone();
                blur3(&mut soft, w, h, 0.6 + (1.0 - det) * 1.2);
                let src = Win { r: win.r, w, h, c: soft };
                for i in 0..win.c.len() {
                    let (x, y) = win.xy(i);
                    let (c, t) = paint_stroke(&src, x, y, (1, -1), 2.0 + len, 1.5, 51);
                    // A chalk stroke is a little lighter in its middle, grainy along its length.
                    let n = (strokes(x, y, (1, -1), 2.0 + len * 0.5, 0.8, 52) - 0.5) * 0.12 + (0.5 - (t - 0.5).abs()) * 0.06;
                    win.c[i] = map(mix(c, win.c[i], det * 0.6), |v| v + n);
                }
            }
            // Bright areas are thick chalk; in dark ones it's scraped off to show the texture.
            texturize_by(win, e.choice("texture"), g("scaling"), g("relief"), e.choice("light"), e.flag("invert"), |p| 0.15 + 0.85 * (1.0 - clamp01(lum(p))));
        }
        F::SmudgeStick => {
            line_blur3(&mut win.c, w, h, (1, 1), (g("strokeLength") * 1.5) as usize);
            let (area, inten) = (g("highlightArea"), g("intensity"));
            for p in &mut win.c {
                let l = lum(p);
                let q = map(*p, |v| v * (0.7 + 0.3 * l));
                *p = highlights(q, l, area, inten);
            }
        }
        F::Sponge => {
            kuwahara(win, 1 + g("brushSize") as usize);
            let def = g("definition") / 25.0;
            let sc = 1.5 + g("smoothness") * 0.5;
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let s = fbm(x as f32 / sc, y as f32 / sc, 61);
                let hole = smoothstep(0.58, 0.75, s);
                win.c[i] = map(win.c[i], |v| clamp01((v - 0.5) * (1.0 + def * 0.6) + 0.5) * (1.0 - hole * def * 0.4));
            }
        }
        F::Underpainting => {
            let orig = win.c.clone();
            blur3(&mut win.c, w, h, 0.5 + g("brushSize") * 0.4);
            for (p, o) in win.c.iter_mut().zip(&orig) {
                *p = saturate(mix(*o, *p, 0.8), 1.15);
            }
            let cov = g("textureCoverage") / 40.0;
            texturize(win, e.choice("texture"), g("scaling"), g("relief") * (0.5 + cov * 1.5), e.choice("light"), e.flag("invert"));
        }
        F::Watercolor => {
            kuwahara(win, watercolor_radius(e));
            let sh = g("shadowIntensity") / 10.0;
            let edge = grad_mag(&win.lum(), w, h);
            let amp = (g("texture") - 1.0) * 0.04;
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let n = (fbm(x as f32 / 3.0, y as f32 / 3.0, 71) - 0.5) * amp;
                let k = 1.0 - clamp01(edge[i] * (3.0 + sh * 8.0)) * 0.6;
                win.c[i] = map(saturate(win.c[i], 1.2), |v| clamp01(v * k + n).powf(1.1 + sh * 0.6));
            }
        }
        // ---------------- Brush Strokes ----------------
        F::AccentedEdges => {
            let mut l = win.lum();
            blur1(&mut l, w, h, 0.4 + g("smoothness") * 0.3);
            let mut edge = grad_mag(&l, w, h);
            blur1(&mut edge, w, h, g("edgeWidth") * 0.5);
            let eb = g("edgeBrightness") / 50.0;
            for i in 0..win.c.len() {
                let k = clamp01(edge[i] * 6.0);
                win.c[i] =
                    if eb >= 0.5 { map(win.c[i], |v| v + (1.0 - v) * k * (eb - 0.5) * 2.0) } else { map(win.c[i], |v| v * (1.0 - k * (0.5 - eb) * 2.0)) };
            }
        }
        F::AngledStrokes => {
            // Light areas are painted in strokes going one way, dark areas the other way;
            // Direction Balance moves the split (0: all down-right, 100: all up-right).
            let len = g("strokeLength").max(3.0);
            let bal = g("directionBalance") / 100.0;
            let mut soft = win.c.clone();
            blur3(&mut soft, w, h, 0.8);
            let l: Vec<f32> = soft.iter().map(lum).collect();
            let src = Win { r: win.r, w, h, c: soft };
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let k = smoothstep(1.0 - bal - 0.05, 1.0 - bal + 0.05, l[i]);
                let (down, _) = paint_stroke(&src, x, y, (1, 1), len, 2.0, 83);
                let (up, _) = paint_stroke(&src, x, y, (1, -1), len, 2.0, 84);
                let n = (strokes(x, y, (1, 1), len * 0.5, 0.7, 85) - 0.5) * (1.0 - k) + (strokes(x, y, (1, -1), len * 0.5, 0.7, 86) - 0.5) * k;
                win.c[i] = map(mix(down, up, k), |v| v + n * 0.1);
            }
            unsharp(win, 1.0, 0.2 + g("sharpness") / 10.0 * 2.3);
        }
        F::Crosshatch => {
            let len = g("strokeLength");
            let r = (len / 3.0) as usize;
            let st = g("strength");
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let n = (hash01(x, y, 0, 81) - 0.5) * 0.12 * st;
                win.c[i] = map(win.c[i], |v| v + n);
            }
            let mut a = win.c.clone();
            line_blur3(&mut a, w, h, (1, 1), r);
            line_blur3(&mut win.c, w, h, (1, -1), r);
            for i in 0..win.c.len() {
                win.c[i] = mix(a[i], win.c[i], 0.5);
            }
            unsharp(win, 1.0, g("sharpness") / 20.0 * 3.0);
        }
        F::DarkStrokes => {
            let (bal, bi, wi) = (g("balance") / 10.0, g("blackIntensity") / 10.0, g("whiteIntensity") / 10.0);
            let l = win.lum();
            let mut a = win.c.clone();
            line_blur3(&mut a, w, h, (1, 1), 3);
            line_blur3(&mut win.c, w, h, (1, -1), 8);
            let t = 0.25 + bal * 0.5;
            for i in 0..win.c.len() {
                let d = map(a[i], |v| v * (1.0 - bi * 0.85));
                let li = map(win.c[i], |v| v + (1.0 - v) * wi * 0.5);
                win.c[i] = mix(d, li, smoothstep(t - 0.1, t + 0.1, l[i]));
            }
        }
        F::InkOutlines => {
            let (di, li) = (g("darkIntensity") / 50.0, g("lightIntensity") / 50.0);
            let l = win.lum();
            let edge = grad_mag(&l, w, h);
            line_blur3(&mut win.c, w, h, (1, 1), (g("strokeLength") / 2.0) as usize);
            for i in 0..win.c.len() {
                let ink = clamp01(edge[i] * 5.0) * di * 1.5;
                let (lt, dk) = (smoothstep(0.5, 1.0, l[i]) * li, smoothstep(0.5, 0.0, l[i]) * di * 0.8);
                win.c[i] = map(win.c[i], |v| (v + (1.0 - v) * lt) * (1.0 - dk) * (1.0 - clamp01(ink)));
            }
        }
        F::Spatter => {
            // Smoothness smooths the random displacement: a larger feature size and, as with any
            // averaged noise, a smaller reach. Low smoothness sprays specks out to the full
            // radius; high smoothness leaves a torn but attached, hard edge (what Photoshop
            // shows at Smoothness 15).
            let rad = g("sprayRadius");
            let feature = 1.0 + g("smoothness") * 0.35;
            let reach = (rad * 1.2 / feature).min(rad);
            let src = Win { r: win.r, w, h, c: win.c.clone() };
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let (fx, fy) = (x as f32 / feature, y as f32 / feature);
                let dx = (fbm(fx, fy, 91) - 0.5) * 3.0 * reach;
                let dy = (fbm(fx, fy, 92) - 0.5) * 3.0 * reach;
                win.c[i] = src.sample(x as f32 + 0.5 + dx, y as f32 + 0.5 + dy);
            }
        }
        F::SprayedStrokes => {
            let rad = g("sprayRadius");
            let sc = 1.2;
            let src = Win { r: win.r, w, h, c: win.c.clone() };
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let (fx, fy) = (x as f32 / sc, y as f32 / sc);
                let dx = (vnoise(fx, fy, 91) - 0.5) * 2.0 * rad;
                let dy = (vnoise(fx, fy, 92) - 0.5) * 2.0 * rad;
                win.c[i] = src.sample(x as f32 + 0.5 + dx, y as f32 + 0.5 + dy);
            }
            line_blur3(&mut win.c, w, h, direction(e.choice("strokeDirection")), (g("strokeLength") / 2.0) as usize);
        }
        F::SumiE => {
            let sw = g("strokeWidth");
            let (pr, con) = (g("strokePressure") / 15.0, g("contrast") / 40.0);
            blur3(&mut win.c, w, h, sw * 0.2);
            line_blur3(&mut win.c, w, h, (1, 1), (sw / 2.0) as usize);
            for p in &mut win.c {
                let l = lum(p);
                let ink = smoothstep(0.65, 0.15, l).powf(1.0 - pr * 0.5);
                *p = map(*p, |v| clamp01((v - 0.5) * (1.0 + con) + 0.5) * (1.0 - ink * 0.9 * (0.5 + con)));
            }
        }
        // ---------------- Distort ----------------
        F::DiffuseGlow => {
            let (gr, ga, ca) = (g("graininess") / 10.0, g("glowAmount") / 20.0, g("clearAmount") / 20.0);
            let thr = 0.2 + ca * 0.7;
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let l = lum(&win.c[i]);
                let mut m = smoothstep(thr - 0.3, thr + 0.1, l) * ga * 1.4;
                if hash01(x, y, 0, 101) < gr * 0.3 * (m + 0.15) {
                    m += 0.5;
                }
                win.c[i] = mix(win.c[i], bg, clamp01(m));
            }
        }
        F::Glass => {
            let dist = g("distortion");
            let step = 0.5 + g("smoothness") * 0.5;
            let kind = e.choice("texture");
            let s = (g("scaling") / 100.0).clamp(0.5, 2.0);
            let sign = if e.flag("invert") { -1.0 } else { 1.0 };
            let max = dist * 2.0;
            let src = Win { r: win.r, w, h, c: win.c.clone() };
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let (fx, fy) = (x as f32, y as f32);
                let gx = texture(kind, fx + step, fy, s) - texture(kind, fx - step, fy, s);
                let gy = texture(kind, fx, fy + step, s) - texture(kind, fx, fy - step, s);
                let dx = (gx * sign * dist * 3.0).clamp(-max, max);
                let dy = (gy * sign * dist * 3.0).clamp(-max, max);
                win.c[i] = src.sample(fx + 0.5 + dx, fy + 0.5 + dy);
            }
        }
        F::OceanRipple => {
            let s = 2.0 + g("rippleSize") * 1.2;
            let mag = g("rippleMagnitude") * 0.9;
            let src = Win { r: win.r, w, h, c: win.c.clone() };
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let (fx, fy) = (x as f32 / s, y as f32 / s);
                let dx = (vnoise(fx, fy, 111) - 0.5) * 2.0 * mag;
                let dy = (vnoise(fx, fy, 112) - 0.5) * 2.0 * mag;
                win.c[i] = src.sample(x as f32 + 0.5 + dx, y as f32 + 0.5 + dy);
            }
        }
        // ---------------- Sketch ----------------
        F::BasRelief => {
            let mut l = win.lum();
            let orig = l.clone();
            blur1(&mut l, w, h, basrelief_sigma(e));
            let (gx, gy) = sobel(&l, w, h);
            let lt = light_dir(e.choice("light"));
            let k = 3.0 + g("detail") * 0.4;
            for i in 0..win.c.len() {
                let t = clamp01(orig[i] * 0.5 + 0.25 + k * -(gx[i] * lt.0 + gy[i] * lt.1));
                win.c[i] = mix(fg, bg, t);
            }
        }
        F::ChalkCharcoal => {
            let (ca, ch, pr) = (g("charcoalArea") / 20.0, g("chalkArea") / 20.0, g("strokePressure") / 5.0);
            let mid = mix(fg, bg, 0.5);
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let l = lum(&win.c[i]);
                let cn = strokes(x, y, (1, 1), 9.0, 0.9, 121) - 0.5;
                let kn = strokes(x, y, (1, -1), 9.0, 0.9, 122) - 0.5;
                let cc = smoothstep(0.25, 0.6, (1.0 - l) * (0.4 + ca * 1.2) + cn * (0.6 + pr * 0.4) - 0.15);
                let kc = smoothstep(0.25, 0.6, l * (0.4 + ch * 1.2) + kn * (0.6 + pr * 0.4) - 0.15);
                win.c[i] = mix(mix(mid, fg, cc), bg, kc * (1.0 - cc));
            }
        }
        F::Charcoal => {
            let (th, det, bal) = (g("charcoalThickness"), g("detail") / 5.0, g("lightDarkBalance") / 100.0);
            let mut l = win.lum();
            blur1(&mut l, w, h, 1.0);
            let edge = grad_mag(&l, w, h);
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let sn = strokes(x, y, (1, 1), 8.0 + th * 3.0, 0.6 + th * 0.5, 131) - 0.5;
                let cov = smoothstep(1.0 - bal - 0.2, 1.0 - bal + 0.2, 1.0 - l[i] + sn * 0.6) + clamp01(edge[i] * det * 6.0);
                win.c[i] = mix(bg, fg, clamp01(cov));
            }
        }
        F::Chrome => {
            let (det, sm) = (g("detail") / 10.0, g("smoothness") / 10.0);
            let mut l = win.lum();
            blur1(&mut l, w, h, 1.0 + sm * 4.0);
            let edge = grad_mag(&l, w, h);
            for i in 0..win.c.len() {
                let v = 0.5 + 0.5 * ((l[i] * (2.5 + det * 3.0) + edge[i] * 6.0) * 2.0 * PI).sin();
                win.c[i] = grey(smoothstep(0.05, 0.95, v));
            }
        }
        F::ConteCrayon => {
            let (fl, bl) = (g("foregroundLevel") / 15.0, g("backgroundLevel") / 15.0);
            let mid = mix(fg, bg, 0.5);
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let l = lum(&win.c[i]);
                let sn = strokes(x, y, (1, -1), 7.0, 1.0, 141) - 0.5;
                let cf = clamp01((0.6 - l) * 2.0 * fl * 1.4 + sn * 0.5);
                let cb = clamp01((l - 0.4) * 2.0 * bl * 1.4 - sn * 0.5);
                win.c[i] = mix(mix(mid, fg, cf), bg, cb * (1.0 - cf));
            }
            texturize(win, e.choice("texture"), g("scaling"), g("relief"), e.choice("light"), e.flag("invert"));
        }
        F::GraphicPen => {
            let r = (g("strokeLength") / 2.0) as usize;
            let bal = g("lightDarkBalance") / 100.0;
            let l = win.lum();
            let mut n: Vec<f32> = (0..win.c.len())
                .map(|i| {
                    let (x, y) = win.xy(i);
                    hash01(x, y, 0, 151)
                })
                .collect();
            line_blur(&mut n, 1, w, h, direction(e.choice("strokeDirection")), r);
            let sd = 0.2887 / ((2 * r + 1) as f32).sqrt();
            for i in 0..win.c.len() {
                let z = (n[i] - 0.5) / sd;
                let t = l[i] + z * 0.16;
                win.c[i] = mix(bg, fg, smoothstep(bal + 0.04, bal - 0.04, t));
            }
        }
        F::HalftonePattern => {
            let size = g("size");
            let con = g("contrast") / 50.0;
            let cell = 3.0 + size * 2.0;
            let mut l = win.lum();
            blur1(&mut l, w, h, 0.5 + size * 0.3);
            let b = ctx.bounds;
            let (cx, cy) = ((b.x0 + b.x1) as f32 / 2.0, (b.y0 + b.y1) as f32 / 2.0);
            let ty = e.choice("patternType");
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
                let d = clamp01((1.0 - l[i] - 0.5) * (1.0 + con * 3.0) + 0.5);
                let aa = 1.0 / cell;
                let ink = match ty {
                    "circle" => {
                        let r = ((fx - cx).powi(2) + (fy - cy).powi(2)).sqrt() / cell;
                        let q = (r.fract() - 0.5).abs() * 2.0;
                        smoothstep(d + aa, d - aa, q)
                    }
                    "line" => {
                        let q = ((fy / cell).fract() - 0.5).abs() * 2.0;
                        smoothstep(d + aa, d - aa, q)
                    }
                    _ => {
                        let (u, v) = ((fx / cell).fract() - 0.5, (fy / cell).fract() - 0.5);
                        let q = (u * u + v * v).sqrt() * 2.0;
                        let rr = d.sqrt() * 1.15;
                        smoothstep(rr + aa, rr - aa, q)
                    }
                };
                win.c[i] = mix(bg, fg, ink);
            }
        }
        F::NotePaper => {
            let (bal, gr, rel) = (g("imageBalance") / 50.0, g("graininess") / 20.0, g("relief") / 25.0);
            let mut l = win.lum();
            blur1(&mut l, w, h, 1.0);
            let mut m: Vec<f32> = l.iter().map(|&v| smoothstep(bal - 0.05, bal + 0.05, v)).collect();
            blur1(&mut m, w, h, 1.2);
            let (gx, gy) = sobel(&m, w, h);
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let n = (hash01(x, y, 0, 161) - 0.5) * gr * 0.3;
                let f = shade(gx[i], gy[i], light_dir("topLeft"), rel * 6.0).clamp(0.0, 2.0);
                win.c[i] = map(mix(fg, bg, m[i]), |v| (v + n) * f);
            }
        }
        F::Photocopy => {
            let (det, dk) = (g("detail"), g("darkness") / 50.0);
            let l = win.lum();
            let mut b = l.clone();
            blur1(&mut b, w, h, 1.0 + det * 0.5);
            for i in 0..win.c.len() {
                let hp = l[i] - b[i];
                let ink = smoothstep(0.0, 0.06, -hp * (1.0 + dk * 8.0)) + smoothstep(0.35, 0.05, l[i]) * dk * 1.5;
                win.c[i] = mix(bg, fg, clamp01(ink));
            }
        }
        F::Plaster => {
            let bal = g("imageBalance") / 50.0;
            let mut l = win.lum();
            blur1(&mut l, w, h, 0.5 + g("smoothness") * 0.8);
            let mut m: Vec<f32> = l.iter().map(|&v| smoothstep(bal + 0.1, bal - 0.1, v)).collect();
            blur1(&mut m, w, h, 1.5);
            let (gx, gy) = sobel(&m, w, h);
            let lt = light_dir(e.choice("light"));
            for i in 0..win.c.len() {
                let f = shade(gx[i], gy[i], lt, 6.0).clamp(0.0, 2.0);
                win.c[i] = map(mix(bg, fg, m[i] * 0.6), |v| v * f);
            }
        }
        F::Reticulation => {
            let (den, fl, bl) = (g("density") / 50.0, g("foregroundLevel") / 50.0, g("backgroundLevel") / 50.0);
            let ink_c = mix(bg, fg, 0.3 + fl * 0.7);
            let paper = mix(bg, fg, bl * 0.6);
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let n = 0.5 * hash01(x, y, 0, 171) + 0.5 * vnoise(x as f32 / 1.8, y as f32 / 1.8, 172);
                let t = (1.0 - lum(&win.c[i])) + (n - 0.5) * (0.4 + den * 1.2);
                win.c[i] = mix(paper, ink_c, smoothstep(0.45, 0.55, t));
            }
        }
        F::Stamp => {
            let bal = g("lightDarkBalance") / 50.0;
            let mut l = win.lum();
            blur1(&mut l, w, h, 0.3 + g("smoothness") * 0.3);
            for i in 0..win.c.len() {
                win.c[i] = mix(bg, fg, smoothstep(bal + 0.02, bal - 0.02, l[i]));
            }
        }
        F::TornEdges => {
            let (bal, sm, con) = (g("imageBalance") / 50.0, g("smoothness"), g("contrast"));
            let mut l = win.lum();
            blur1(&mut l, w, h, 1.0);
            let rough = (16.0 - sm) / 15.0 * 0.35;
            let aa = 0.25 / con;
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let t = l[i] + (fbm(x as f32 / 2.0, y as f32 / 2.0, 181) - 0.5) * rough;
                win.c[i] = mix(bg, fg, smoothstep(bal + aa, bal - aa, t));
            }
        }
        F::WaterPaper => {
            let fl = g("fiberLength");
            let (br, co) = (g("brightness") / 100.0, g("contrast") / 100.0);
            let r = (fl / 6.0) as usize;
            line_blur3(&mut win.c, w, h, (1, 0), r);
            line_blur3(&mut win.c, w, h, (0, 1), r);
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let fib = strokes(x, y, (1, 0), fl, 1.0, 191).max(strokes(x, y, (0, 1), fl, 1.0, 192));
                win.c[i] = map(win.c[i], |v| (v * (0.85 + 0.25 * fib) - 0.5) * (co * 1.6 + 0.2) + 0.5 + (br - 0.5) * 0.8);
            }
        }
        // ---------------- Stylize ----------------
        F::GlowingEdges => {
            blur3(&mut win.c, w, h, 0.3 + g("smoothness") * 0.25);
            let mut e3 = vec![[0.0f32; 3]; win.c.len()];
            for c in 0..3 {
                let plane: Vec<f32> = win.c.iter().map(|p| p[c]).collect();
                for (o, v) in e3.iter_mut().zip(grad_mag(&plane, w, h)) {
                    o[c] = v;
                }
            }
            blur3(&mut e3, w, h, g("edgeWidth") * 0.4);
            let k = 2.0 + g("edgeBrightness") * 1.2;
            for (p, q) in win.c.iter_mut().zip(&e3) {
                *p = map(*q, |v| clamp01(v * k));
            }
        }
        // ---------------- Texture ----------------
        F::Craquelure => {
            let (sp, dep, bri) = (g("crackSpacing"), g("crackDepth") / 10.0, g("crackBrightness") / 10.0);
            let cell = sp * 1.3 + 1.0;
            let width = 0.8 + dep * 0.8;
            let crack = |x: f32, y: f32| {
                let (d1, d2, _) = voronoi(x, y, cell, 201);
                1.0 - smoothstep(0.0, width, d2 - d1)
            };
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
                let c0 = crack(fx, fy);
                let (gx, gy) = (crack(fx + 1.0, fy) - c0, crack(fx, fy + 1.0) - c0);
                let f = shade(-gx, -gy, light_dir("topLeft"), dep * 2.0).clamp(0.0, 2.0);
                win.c[i] = map(win.c[i], |v| v * (0.55 + bri * 0.5) * (1.0 - c0 * dep * 0.6) * f);
            }
        }
        F::Grain => {
            let (it, co) = (g("intensity") / 100.0, g("contrast") / 100.0);
            let ty = e.choice("grainType");
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let hc = |c: u32| hash01(x, y, c, 211) - 0.5;
                let p = win.c[i];
                let q = match ty {
                    "soft" => [p[0] + hc(0) * it * 0.4, p[1] + hc(1) * it * 0.4, p[2] + hc(2) * it * 0.4],
                    "sprinkles" => {
                        if hash01(x, y, 3, 211) < it * 0.15 {
                            bg
                        } else {
                            p
                        }
                    }
                    "clumped" => map(p, |v| v + (vnoise(x as f32 / 2.0, y as f32 / 2.0, 212) - 0.5) * it * 1.2),
                    "contrasty" => map(p, |v| (v - 0.5) * (1.0 + it * 2.0) + 0.5 + hc(4) * it * 0.5),
                    "enlarged" => {
                        let (fx, fy) = (x as f32 / 3.0, y as f32 / 3.0);
                        [p[0] + (vnoise(fx, fy, 213) - 0.5) * it, p[1] + (vnoise(fx, fy, 214) - 0.5) * it, p[2] + (vnoise(fx, fy, 215) - 0.5) * it]
                    }
                    "stippled" => {
                        let st = if lum(&p) > hash01(x, y, 5, 211) { bg } else { fg };
                        mix(p, st, 0.3 + it * 0.7)
                    }
                    "horizontal" => map(p, |v| v + (vnoise(x as f32 / 25.0, y as f32, 216) - 0.5) * it),
                    "vertical" => map(p, |v| v + (vnoise(x as f32, y as f32 / 25.0, 217) - 0.5) * it),
                    "speckle" => {
                        if hash01(x, y, 6, 211) < it * 0.1 {
                            mix(p, fg, 0.85)
                        } else {
                            p
                        }
                    }
                    _ => [p[0] + hc(0) * it * 0.8, p[1] + hc(1) * it * 0.8, p[2] + hc(2) * it * 0.8],
                };
                win.c[i] = map(q, |v| (v - 0.5) * (0.5 + co) + 0.5);
            }
        }
        F::MosaicTiles => {
            let (ts, gw, lg) = (g("tileSize"), g("groutWidth"), g("lightenGrout") / 10.0);
            let edge = |x: f32, y: f32| {
                let jx = x + (vnoise(x / ts * 0.7, y / ts * 0.7, 221) - 0.5) * ts * 0.3;
                let jy = y + (vnoise(x / ts * 0.7, y / ts * 0.7, 222) - 0.5) * ts * 0.3;
                let (u, v) = (jx.rem_euclid(ts), jy.rem_euclid(ts));
                u.min(ts - u).min(v.min(ts - v))
            };
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
                let de = edge(fx, fy);
                let grout = smoothstep(gw * 0.5 + 0.7, gw * 0.5 - 0.3, de);
                let hgt = |xx: f32, yy: f32| smoothstep(gw * 0.5, gw * 0.5 + 2.5, edge(xx, yy));
                let s = (hgt(fx - 1.0, fy - 1.0) - hgt(fx + 1.0, fy + 1.0)) * 0.35;
                let l = lum(&win.c[i]);
                let gc = grey(clamp01(l * 0.5 + lg * 0.45));
                win.c[i] = mix(map(win.c[i], |v| v * (1.0 + s)), gc, grout);
            }
        }
        F::Patchwork => {
            let size = patch_size(e);
            let rel = g("relief") / 25.0;
            let src = Win { r: win.r, w, h, c: win.c.clone() };
            let hv = |x: f32, y: f32| {
                let (bx, by) = ((x / size).floor() as i32, (y / size).floor() as i32);
                let (u, v) = (x.rem_euclid(size), y.rem_euclid(size));
                let bevel = smoothstep(0.0, 1.5, u.min(size - u).min(v.min(size - v)));
                hash01(bx, by, 0, 231) * bevel
            };
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
                let (bx, by) = ((fx / size).floor(), (fy / size).floor());
                let c = src.at(((bx + 0.5) * size) as i32, ((by + 0.5) * size) as i32);
                let s = (hv(fx - 1.0, fy - 1.0) - hv(fx + 1.0, fy + 1.0)) * rel * 1.5;
                win.c[i] = map(c, |v| v * (1.0 + s));
            }
        }
        F::StainedGlass => {
            let cell = glass_cell(e);
            let bt = g("borderThickness");
            let li = g("lightIntensity") / 10.0;
            let b = ctx.bounds;
            let (cx, cy) = ((b.x0 + b.x1) as f32 / 2.0, (b.y0 + b.y1) as f32 / 2.0);
            let half = (b.width().max(b.height()) as f32 / 2.0).max(1.0);
            let src = Win { r: win.r, w, h, c: win.c.clone() };
            for i in 0..win.c.len() {
                let (x, y) = win.xy(i);
                let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
                let (d1, d2, p) = voronoi(fx, fy, cell, 241);
                let border = smoothstep(bt * 0.5 + 0.6, bt * 0.5 - 0.4, (d2 - d1) * 0.5);
                let dn = (((fx - cx).powi(2) + (fy - cy).powi(2)).sqrt() / half).min(1.0);
                let col = map(src.at(p.0 as i32, p.1 as i32), |v| v * (1.0 + li * 0.6 * (1.0 - dn)));
                win.c[i] = mix(col, fg, border);
            }
        }
        F::Texturizer => texturize(win, e.choice("texture"), g("scaling"), g("relief"), e.choice("light"), e.flag("invert")),
    }
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-9);
    [v[0] / l, v[1] / l, v[2] / l]
}

#[cfg(test)]
mod tests {
    use photocraft_color::{ColorMode, PixelFormat, SampleType};
    use photocraft_raster::Surface;

    use super::*;
    use crate::{FilterParams, apply_tiled};

    fn pattern(st: SampleType) -> Surface {
        let mut s = Surface::new(PixelFormat::new(ColorMode::Rgb, st, true));
        for y in 0..40 {
            for x in 0..56 {
                let v = ((x * 7 + y * 3) % 23) as f32 / 22.0;
                let px = if (x / 9 + y / 7) % 2 == 0 { [v, 0.3, 1.0 - v, 1.0] } else { [0.9, v * 0.5, 0.2, 1.0] };
                s.write_pixel(x, y, &px);
            }
        }
        s
    }

    #[test]
    fn every_effect_changes_pixels_and_is_tile_independent() {
        let b = Rect::new(0, 0, 56, 40);
        for st in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let s = pattern(st);
            let before = s.read_region(b);
            for f in crate::GalleryFilter::ALL {
                let fp = FilterParams::FilterGallery { effects: vec![GalleryEffect::new(*f)] };
                let a = apply_tiled(&s, &fp, b, b, None, 256, Some(b)).read_region(b);
                assert_ne!(a, before, "{} @{st:?} changed nothing", f.key());
                assert!(a.iter().all(|v| v.is_finite()), "{}", f.key());
                if st == SampleType::F32 {
                    let t = apply_tiled(&s, &fp, b, b, None, 16, Some(b)).read_region(b);
                    let worst = a.iter().zip(&t).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
                    assert!(worst < 1e-3, "{} differs across tilings by {worst}", f.key());
                }
            }
        }
    }

    #[test]
    fn rough_pastels_and_angled_strokes_respond_to_every_slider() {
        // #879: these smeared or drowned the image in texture, and some sliders did nothing.
        use crate::GalleryFilter as G;
        let b = Rect::new(0, 0, 56, 40);
        let s = pattern(SampleType::F32);
        let run = |e: &GalleryEffect, tile: i32| {
            let fp = FilterParams::FilterGallery { effects: vec![e.clone()] };
            apply_tiled(&s, &fp, b, b, None, tile, Some(b)).read_region(b)
        };
        for f in [G::RoughPastels, G::AngledStrokes] {
            let base = run(&GalleryEffect::new(f), 256);
            for prm in f.params() {
                let crate::GalleryParamKind::Range { min, max, .. } = prm.kind else { continue };
                for v in [min, max] {
                    let mut e = GalleryEffect::new(f);
                    e.set(prm.key, v);
                    let a = run(&e, 256);
                    assert_ne!(a, base, "{} {} = {v} changed nothing", f.key(), prm.key);
                    let t = run(&e, 16);
                    let worst = a.iter().zip(&t).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
                    assert!(worst < 1e-3, "{} {} = {v} differs across tilings by {worst}", f.key(), prm.key);
                }
            }
        }
    }

    #[test]
    fn params_parse_and_clamp() {
        for f in crate::GalleryFilter::ALL {
            assert!(!f.params().is_empty(), "{}", f.key());
            assert_eq!(crate::GalleryFilter::from_key(f.key()), Some(*f));
            assert_eq!(crate::GalleryFilter::from_key(f.command_id()), Some(*f));
        }
        assert_eq!(crate::GalleryFilter::ALL.len(), 47);
        let mut e = GalleryEffect::new(crate::GalleryFilter::Texturizer);
        assert_eq!(e.choice("texture"), "canvas");
        assert_eq!(e.choice("light"), "top");
        assert!(e.set_choice("texture", "brick"));
        assert_eq!(e.choice("texture"), "brick");
        e.set("relief", 500.0);
        assert_eq!(e.get("relief"), 50.0);
    }

    /// Release timing on 24 MP: `cargo test -p photocraft-algo --release bench_gallery -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_gallery() {
        let (w, h) = (6000, 4000);
        let mut s = Surface::new(PixelFormat::new(ColorMode::Rgb, SampleType::U8, true));
        let b = Rect::new(0, 0, w, h);
        s.fill_rect(b, &[0.5, 0.4, 0.3, 1.0]);
        for y in (0..h).step_by(9) {
            s.fill_rect(Rect::new(0, y, w, y + 3), &[0.9, 0.8, 0.2, 1.0]);
        }
        let t = std::time::Instant::now();
        std::hint::black_box(crate::apply_in(
            &s,
            &FilterParams::AddNoise { amount: 10.0, distribution: crate::Distribution::Uniform, monochromatic: true, seed: 1 },
            b,
            b,
            None,
            b,
        ));
        println!("{:>18}: {:>7.0} ms", "(addNoise baseline)", t.elapsed().as_secs_f64() * 1000.0);
        let mut total = 0.0;
        for f in crate::GalleryFilter::ALL {
            let fp = FilterParams::FilterGallery { effects: vec![GalleryEffect::new(*f)] };
            let t = std::time::Instant::now();
            std::hint::black_box(crate::apply_in(&s, &fp, b, b, None, b));
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            total += ms;
            println!("{:>18}: {ms:>7.0} ms", f.key());
        }
        println!("{:>18}: {total:>7.0} ms", "total");
    }

    /// Visual check: `GALLERY_RAW=src.raw GALLERY_W=480 GALLERY_H=322 cargo test -p photocraft-algo --release gallery_sheet -- --ignored`
    /// writes `<key>.raw` (RGB8) beside the input for every filter.
    #[test]
    #[ignore]
    fn gallery_sheet() {
        let Ok(path) = std::env::var("GALLERY_RAW") else { return };
        let w: i32 = std::env::var("GALLERY_W").unwrap().parse().unwrap();
        let h: i32 = std::env::var("GALLERY_H").unwrap().parse().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let mut s = Surface::new(PixelFormat::new(ColorMode::Rgb, SampleType::U8, true));
        let data: Vec<f32> = bytes.chunks(3).flat_map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0, 1.0]).collect();
        let b = Rect::new(0, 0, w, h);
        s.write_region(b, &data);
        let dir = std::path::Path::new(&path).parent().unwrap().to_path_buf();
        for f in crate::GalleryFilter::ALL {
            let fp = FilterParams::FilterGallery { effects: vec![GalleryEffect::new(*f)] };
            let out = crate::apply_in(&s, &fp, b, b, None, b).read_region(b);
            let raw: Vec<u8> = out.chunks(4).flat_map(|p| [p[0], p[1], p[2]].map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)).collect();
            std::fs::write(dir.join(format!("{}.raw", f.key())), raw).unwrap();
        }
    }
}
