//! Distortions (inverse mapping + bilinear resampling): Twirl, Pinch,
//! Spherize, Wave, Ripple, Polar Coordinates. All are centred on the
//! reference bounds.

use photocraft_geom::Rect;

use crate::image::{Edge, Image};
use crate::noise::hash01;
use crate::other::edge_of;
use crate::{Ctx, PolarMode, RippleSize, SpherizeMode, UndefinedAreas, WaveType};

pub(crate) fn remap(src: &Image, out: Rect, ctx: &Ctx, edge: Edge, f: impl Fn(f32, f32) -> (f32, f32)) -> Vec<f32> {
    let n = src.ch;
    let mut res = Vec::with_capacity(out.width() as usize * out.height() as usize * n);
    let mut tmp = vec![0.0f32; n];
    for y in out.y0..out.y1 {
        for x in out.x0..out.x1 {
            let (sx, sy) = f(x as f32 + 0.5, y as f32 + 0.5);
            src.sample(sx, sy, edge, ctx.bounds, ctx.alpha, &mut tmp);
            res.extend_from_slice(&tmp);
        }
    }
    res
}

pub(crate) fn centre(b: Rect) -> (f32, f32, f32) {
    let (w, h) = (b.width() as f32, b.height() as f32);
    (b.x0 as f32 + w / 2.0, b.y0 as f32 + h / 2.0, w.min(h) / 2.0)
}

pub(crate) fn twirl(src: &Image, out: Rect, ctx: &Ctx, angle: f32) -> Vec<f32> {
    let (cx, cy, rr) = centre(ctx.bounds);
    let a = angle.to_radians();
    remap(src, out, ctx, Edge::Repeat, |x, y| {
        let (dx, dy) = (x - cx, y - cy);
        let r = (dx * dx + dy * dy).sqrt();
        if r >= rr || rr <= 0.0 {
            return (x, y);
        }
        let t = a * (1.0 - r / rr);
        let (s, c) = t.sin_cos();
        (cx + dx * c - dy * s, cy + dx * s + dy * c)
    })
}

pub(crate) fn pinch(src: &Image, out: Rect, ctx: &Ctx, amount: f32) -> Vec<f32> {
    let (cx, cy, rr) = centre(ctx.bounds);
    let a = (amount / 100.0).clamp(-1.0, 1.0);
    remap(src, out, ctx, Edge::Repeat, |x, y| {
        let (dx, dy) = (x - cx, y - cy);
        let r = (dx * dx + dy * dy).sqrt();
        if r >= rr || r <= 0.0 {
            return (x, y);
        }
        let rs = rr * (r / rr).powf(1.0 - a * 0.5);
        (cx + dx * rs / r, cy + dy * rs / r)
    })
}

fn sphere_1d(rn: f32, a: f32) -> f32 {
    // rn in [0,1): positive amount magnifies the centre (samples closer in).
    let target = if a >= 0.0 { rn.asin() * 2.0 / std::f32::consts::PI } else { (rn * std::f32::consts::FRAC_PI_2).sin() };
    rn + a.abs() * (target - rn)
}

pub(crate) fn spherize(src: &Image, out: Rect, ctx: &Ctx, amount: f32, mode: SpherizeMode) -> Vec<f32> {
    let b = ctx.bounds;
    let (cx, cy, rr) = centre(b);
    let (hx, hy) = (b.width() as f32 / 2.0, b.height() as f32 / 2.0);
    let a = (amount / 100.0).clamp(-1.0, 1.0);
    remap(src, out, ctx, Edge::Repeat, |x, y| {
        let (dx, dy) = (x - cx, y - cy);
        match mode {
            SpherizeMode::Normal => {
                let r = (dx * dx + dy * dy).sqrt();
                if r >= rr || r <= 0.0 {
                    return (x, y);
                }
                let rs = rr * sphere_1d(r / rr, a);
                (cx + dx * rs / r, cy + dy * rs / r)
            }
            SpherizeMode::HorizontalOnly => {
                if dx.abs() >= hx {
                    return (x, y);
                }
                (cx + dx.signum() * hx * sphere_1d(dx.abs() / hx, a), y)
            }
            SpherizeMode::VerticalOnly => {
                if dy.abs() >= hy {
                    return (x, y);
                }
                (x, cy + dy.signum() * hy * sphere_1d(dy.abs() / hy, a))
            }
        }
    })
}

/// Wave generator settings.
pub(crate) struct WaveSpec {
    pub generators: u32,
    pub wavelength: (f32, f32),
    pub amplitude: (f32, f32),
    pub wave_type: WaveType,
    pub undefined: UndefinedAreas,
    pub seed: u32,
}

fn shape(t: WaveType, phase: f32) -> f32 {
    let p = phase.rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU;
    match t {
        WaveType::Sine => (phase).sin(),
        WaveType::Triangle => 1.0 - 4.0 * (p - 0.5).abs(),
        WaveType::Square => {
            if p < 0.5 {
                1.0
            } else {
                -1.0
            }
        }
    }
}

pub(crate) fn wave(src: &Image, out: Rect, ctx: &Ctx, w: WaveSpec) -> Vec<f32> {
    let n = w.generators.clamp(1, 64);
    let lerp = |(a, b): (f32, f32), t: f32| a + (b - a).max(0.0) * t;
    let gens: Vec<(f32, f32, f32, f32)> = (0..n)
        .map(|i| {
            let r = |k: u32| hash01(i as i32, k as i32, 7, w.seed);
            (lerp(w.wavelength, r(1)).max(1.0), lerp(w.amplitude, r(2)), r(3) * std::f32::consts::TAU, r(4) * std::f32::consts::TAU)
        })
        .collect();
    let b = ctx.bounds;
    remap(src, out, ctx, edge_of(w.undefined), |x, y| {
        let (lx, ly) = (x - b.x0 as f32, y - b.y0 as f32);
        let (mut dx, mut dy) = (0.0, 0.0);
        for &(len, amp, p1, p2) in &gens {
            dx += amp * shape(w.wave_type, std::f32::consts::TAU * ly / len + p1);
            dy += amp * shape(w.wave_type, std::f32::consts::TAU * lx / len + p2);
        }
        (x + dx / n as f32, y + dy / n as f32)
    })
}

pub(crate) fn ripple(src: &Image, out: Rect, ctx: &Ctx, amount: f32, size: RippleSize) -> Vec<f32> {
    let len = match size {
        RippleSize::Small => 8.0,
        RippleSize::Medium => 16.0,
        RippleSize::Large => 32.0,
    };
    let amp = amount / 100.0 * len / 8.0;
    let b = ctx.bounds;
    remap(src, out, ctx, Edge::Repeat, |x, y| {
        let (lx, ly) = (x - b.x0 as f32, y - b.y0 as f32);
        (x + amp * (std::f32::consts::TAU * ly / len).sin(), y + amp * (std::f32::consts::TAU * lx / len).sin())
    })
}

pub(crate) fn polar(src: &Image, out: Rect, ctx: &Ctx, mode: PolarMode) -> Vec<f32> {
    let b = ctx.bounds;
    let (w, h) = (b.width().max(1) as f32, b.height().max(1) as f32);
    let (cx, cy) = (b.x0 as f32 + w / 2.0, b.y0 as f32 + h / 2.0);
    let rmax = w.min(h) / 2.0;
    // Photoshop 27.11: the angle runs counter-clockwise from 12 o'clock, and points past the
    // source's edge repeat its edge pixels (the corners outside the circle are filled, not empty).
    // Photoshop evaluates the map at pixel corners and reads the source at the resulting index
    // (half a pixel from `remap`'s centres): within 2 levels on a whole test chart in both directions.
    remap(src, out, ctx, Edge::Repeat, |x, y| polar_map(x - 0.5, y - 0.5, mode, b, (w, h), (cx, cy), rmax))
}

fn polar_map(x: f32, y: f32, mode: PolarMode, b: Rect, (w, h): (f32, f32), (cx, cy): (f32, f32), rmax: f32) -> (f32, f32) {
    let (sx, sy) = match mode {
        PolarMode::RectangularToPolar => {
            // Output is polar: angle from 12 o'clock counter-clockwise → source x, radius → source y.
            let (dx, dy) = (x - cx, y - cy);
            // The 12 o'clock seam itself reads the source's right edge (360°, not 0°).
            let theta = (-dx).atan2(-dy);
            let theta = if theta <= 0.0 { theta + std::f32::consts::TAU } else { theta };
            let r = (dx * dx + dy * dy).sqrt();
            (b.x0 as f32 + theta / std::f32::consts::TAU * w, b.y0 as f32 + r / rmax * h)
        }
        PolarMode::PolarToRectangular => {
            // Photoshop's angle is one column ahead of its radius row (fitted exactly).
            let theta = (x + 1.0 - b.x0 as f32) / w * std::f32::consts::TAU;
            let r = (y - b.y0 as f32) / h * rmax;
            (cx - r * theta.sin(), cy - r * theta.cos())
        }
    };
    (sx + 0.5, sy + 0.5)
}
