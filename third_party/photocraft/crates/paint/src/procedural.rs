//! Procedural (generated, deterministic) texture patterns and sampled tips for the built-in presets.
//! Everything here is original: seeded value noise and simple geometry, no bundled brush files.

use crate::brush::PatternStyle;
use crate::rng::hash2;
use crate::tile::{GrayTile, PatternImage};

#[inline]
fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Tileable value noise with `period` lattice cells over the tile.
fn value_noise(x: f32, y: f32, period: i32, salt: u64) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (tx, ty) = (smooth(x - x0), smooth(y - y0));
    let (xi, yi) = (x0 as i32, y0 as i32);
    let h = |i: i32, j: i32| hash2(i.rem_euclid(period), j.rem_euclid(period), salt);
    let a = h(xi, yi) + (h(xi + 1, yi) - h(xi, yi)) * tx;
    let b = h(xi, yi + 1) + (h(xi + 1, yi + 1) - h(xi, yi + 1)) * tx;
    a + (b - a) * ty
}

/// Tileable fractal noise in 0..1 (`cells` lattice cells across at the base octave).
fn fbm(x: f32, y: f32, size: f32, cells: i32, octaves: u32, salt: u64) -> f32 {
    let (mut sum, mut amp, mut norm, mut c) = (0.0, 1.0, 0.0, cells);
    for o in 0..octaves {
        sum += value_noise(x / size * c as f32, y / size * c as f32, c, salt.wrapping_add(o as u64 * 7919)) * amp;
        norm += amp;
        amp *= 0.5;
        c *= 2;
    }
    sum / norm
}

/// Generate a seamless pattern tile.
pub fn pattern(style: PatternStyle, size: u32, seed: u32) -> PatternImage {
    let n = size.clamp(8, 1024) as usize;
    let s = n as f32;
    let salt = u64::from(seed) * 0x0100_0000_01B3 + 17;
    let mut data = vec![0.0f32; n * n];
    for y in 0..n {
        for x in 0..n {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let v = match style {
                PatternStyle::Noise => fbm(fx, fy, s, 8, 4, salt),
                PatternStyle::Paper => {
                    let base = fbm(fx, fy, s, 16, 4, salt);
                    let grain = hash2(x as i32, y as i32, salt ^ 0x55);
                    ((base - 0.5) * 2.2 + 0.5 + (grain - 0.5) * 0.35).clamp(0.0, 1.0)
                }
                PatternStyle::Canvas => {
                    // Over/under weave: threads every 8 px, alternating which direction is on top.
                    let period = 8.0;
                    let (cx, cy) = ((fx / period).floor() as i32, (fy / period).floor() as i32);
                    let (u, v) = ((fx / period).fract(), (fy / period).fract());
                    let warp = (std::f32::consts::PI * u).sin();
                    let weft = (std::f32::consts::PI * v).sin();
                    let top = if (cx + cy) % 2 == 0 { warp } else { weft };
                    (0.25 + 0.65 * top + (hash2(x as i32, y as i32, salt) - 0.5) * 0.15).clamp(0.0, 1.0)
                }
                PatternStyle::Dots => {
                    let period = 8.0;
                    let (u, v) = ((fx / period).fract() - 0.5, (fy / period).fract() - 0.5);
                    let d = (u * u + v * v).sqrt();
                    1.0 - smooth(((0.35 - d) / 0.1 + 0.5).clamp(0.0, 1.0))
                }
            };
            data[y * n + x] = v;
        }
    }
    PatternImage { width: n, height: n, data }
}

/// A chalk/charcoal tip: a disc with a ragged edge and grainy interior.
pub fn chalk_tip(n: u32, seed: u32) -> GrayTile {
    let salt = u64::from(seed) + 101;
    let c = n as f32 / 2.0;
    GrayTile::from_fn(n, n, |x, y| {
        let (dx, dy) = (x as f32 + 0.5 - c, y as f32 + 0.5 - c);
        let r = (dx * dx + dy * dy).sqrt() / c;
        let ang = dy.atan2(dx);
        let edge = 0.8 + 0.15 * value_noise((ang + std::f32::consts::PI) * 3.0, 0.5, 19, salt);
        let inside = ((edge - r) * 8.0).clamp(0.0, 1.0);
        let grain = hash2(x as i32, y as i32, salt);
        inside * if grain < 0.2 { 0.2 } else { 0.7 + 0.3 * grain }
    })
}

/// A spatter tip: a cluster of round droplets of different sizes.
pub fn spatter_tip(n: u32, seed: u32, drops: u32) -> GrayTile {
    let salt = u64::from(seed) + 202;
    let s = n as f32;
    let blobs: Vec<(f32, f32, f32)> = (0..drops)
        .map(|i| {
            let a = hash2(i as i32, 0, salt) * std::f32::consts::TAU;
            let d = hash2(i as i32, 1, salt).sqrt() * 0.38 * s;
            let r = (0.03 + 0.09 * hash2(i as i32, 2, salt).powi(2)) * s;
            (s / 2.0 + a.cos() * d, s / 2.0 + a.sin() * d, r)
        })
        .collect();
    GrayTile::from_fn(n, n, |x, y| {
        let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
        blobs.iter().fold(0.0f32, |acc, &(bx, by, r)| {
            let d = ((px - bx).powi(2) + (py - by).powi(2)).sqrt();
            acc.max((r + 0.5 - d).clamp(0.0, 1.0))
        })
    })
}

/// A bristle tip: parallel hair streaks of varying density inside an ellipse.
pub fn bristle_tip(n: u32, seed: u32) -> GrayTile {
    let salt = u64::from(seed) + 303;
    let c = n as f32 / 2.0;
    GrayTile::from_fn(n, n, |x, y| {
        let (dx, dy) = ((x as f32 + 0.5 - c) / c, (y as f32 + 0.5 - c) / c);
        let r = (dx * dx + (dy * 1.8).powi(2)).sqrt();
        if r > 1.0 {
            return 0.0;
        }
        let hair = hash2(0, (y as f32 / 1.5) as i32, salt);
        let density = if hair < 0.25 { 0.0 } else { 0.5 + 0.5 * hair };
        density * (1.0 - r.powi(4))
    })
}

/// A charcoal tip: a squat ellipse of broken horizontal streaks with a gritty edge.
pub fn charcoal_tip(n: u32, seed: u32) -> GrayTile {
    let salt = u64::from(seed) + 404;
    let c = n as f32 / 2.0;
    GrayTile::from_fn(n, n, |x, y| {
        let (dx, dy) = ((x as f32 + 0.5 - c) / c, (y as f32 + 0.5 - c) / c);
        let r = (dx * dx + (dy * 1.6).powi(2)).sqrt();
        let edge = 0.85 + 0.15 * value_noise(dx * 6.0 + 9.0, dy * 6.0 + 9.0, 64, salt);
        if r > edge {
            return 0.0;
        }
        // Streaks: per-row density modulated along x so they break up.
        let row = hash2(0, y as i32, salt);
        let along = value_noise(x as f32 / 6.0, y as f32 / 1.5, 1 << 12, salt ^ 0x77);
        let grit = hash2(x as i32, y as i32, salt ^ 0x13);
        let v = (0.35 + 0.65 * row) * (0.4 + 0.6 * along) * if grit < 0.15 { 0.2 } else { 1.0 };
        v * ((edge - r) * 6.0).clamp(0.0, 1.0)
    })
}

/// A leaf: a pointed lens shape with a lighter midrib, stem towards -x.
pub fn leaf_tip(n: u32) -> GrayTile {
    let c = n as f32 / 2.0;
    GrayTile::from_fn(n, n, |x, y| {
        let (u, v) = ((x as f32 + 0.5 - c) / c, (y as f32 + 0.5 - c) / c);
        // Half-width of the blade at u: widest a little behind the middle, pointed at both ends.
        let t = ((u + 1.0) / 2.0).clamp(0.0, 1.0);
        let half = 0.42 * (std::f32::consts::PI * t).sin() * (1.0 - 0.25 * t);
        let inside = ((half - v.abs()) * c).clamp(0.0, 1.0);
        let rib = 1.0 - 0.45 * (1.0 - (v.abs() * c / 1.2).clamp(0.0, 1.0));
        let veins = 1.0 - 0.2 * (1.0 - ((((u * 7.0 - v.abs() * 5.0).fract()) - 0.5).abs() * 8.0).clamp(0.0, 1.0));
        let stem = if u < -0.75 && v.abs() * c < 0.9 { 1.0 } else { 0.0 };
        (inside * rib * veins).max(stem)
    })
}

/// Grass: a tuft of tapered, slightly curved blades rising from the bottom edge.
pub fn grass_tip(n: u32, seed: u32, blades: u32) -> GrayTile {
    let salt = u64::from(seed) + 505;
    let s = n as f32;
    let bl: Vec<(f32, f32, f32, f32)> = (0..blades)
        .map(|i| {
            let base = (0.3 + 0.4 * hash2(i as i32, 0, salt)) * s;
            let height = (0.55 + 0.42 * hash2(i as i32, 1, salt)) * s;
            let lean = (hash2(i as i32, 2, salt) - 0.5) * 0.9;
            let width = (0.025 + 0.025 * hash2(i as i32, 3, salt)) * s;
            (base, height, lean, width)
        })
        .collect();
    GrayTile::from_fn(n, n, |x, y| {
        let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
        let up = s - py;
        bl.iter().fold(0.0f32, |acc, &(base, height, lean, width)| {
            if up < 0.0 || up > height {
                return acc;
            }
            let t = up / height;
            let cx = base + lean * height * t * t;
            let w = width * (1.0 - t) + 0.3;
            acc.max((w - (px - cx).abs()).clamp(0.0, 1.0))
        })
    })
}

/// A sea sponge: a disc riddled with holes of different sizes.
pub fn sponge_tip(n: u32, seed: u32) -> GrayTile {
    let salt = u64::from(seed) + 606;
    let c = n as f32 / 2.0;
    GrayTile::from_fn(n, n, |x, y| {
        let (dx, dy) = (x as f32 + 0.5 - c, y as f32 + 0.5 - c);
        let r = (dx * dx + dy * dy).sqrt() / c;
        let edge = 0.78 + 0.2 * value_noise((dy.atan2(dx) + std::f32::consts::PI) * 2.5, 0.5, 16, salt);
        let disc = ((edge - r) * 6.0).clamp(0.0, 1.0);
        let pores = fbm(x as f32, y as f32, n as f32, 6, 3, salt);
        disc * ((pores - 0.42) * 6.0).clamp(0.0, 1.0)
    })
}

/// A four-point sparkle with a soft glow.
pub fn star_tip(n: u32) -> GrayTile {
    let c = n as f32 / 2.0;
    GrayTile::from_fn(n, n, |x, y| {
        let (u, v) = (((x as f32 + 0.5 - c) / c).abs(), ((y as f32 + 0.5 - c) / c).abs());
        let ray = |a: f32, b: f32| ((1.0 - a) * (1.0 - (b / (0.08 * (1.0 - a) + 0.005)).min(1.0))).max(0.0);
        let glow = (1.0 - (u * u + v * v).sqrt() * 2.5).max(0.0).powi(2);
        ray(u, v).max(ray(v, u)).max(glow).min(1.0)
    })
}

/// A rake: a row of separate round bristles of slightly different sizes.
pub fn rake_tip(n: u32, seed: u32, bristles: u32) -> GrayTile {
    let salt = u64::from(seed) + 707;
    let s = n as f32;
    let k = bristles.max(1);
    let dots: Vec<(f32, f32, f32)> = (0..k)
        .map(|i| {
            let x = s * (i as f32 + 0.5) / k as f32;
            let y = s / 2.0 + (hash2(i as i32, 0, salt) - 0.5) * s * 0.08;
            let r = s / k as f32 * (0.22 + 0.18 * hash2(i as i32, 1, salt));
            (x, y, r)
        })
        .collect();
    GrayTile::from_fn(n, n, |x, y| {
        let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
        dots.iter().fold(0.0f32, |acc, &(dx, dy, r)| {
            let d = ((px - dx).powi(2) + (py - dy).powi(2)).sqrt();
            acc.max((r + 0.5 - d).clamp(0.0, 1.0) * 0.9)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patterns_are_seamless_and_in_range() {
        for st in [PatternStyle::Noise, PatternStyle::Paper, PatternStyle::Canvas, PatternStyle::Dots] {
            let p = pattern(st, 64, 3);
            assert!(p.data.iter().all(|v| (0.0..=1.0).contains(v)), "{st:?}");
            let var = p.data.iter().map(|v| (v - 0.5).abs()).sum::<f32>() / p.data.len() as f32;
            assert!(var > 0.02, "{st:?} is flat");
            // Deterministic.
            assert_eq!(p.data, pattern(st, 64, 3).data);
        }
        // Noise tiles: left and right columns are neighbours.
        let p = pattern(PatternStyle::Noise, 64, 1);
        let seam: f32 = (0..64).map(|y| (p.data[y * 64] - p.data[y * 64 + 63]).abs()).sum::<f32>() / 64.0;
        assert!(seam < 0.08, "{seam}");
    }

    #[test]
    fn tips_have_paint() {
        for t in [
            chalk_tip(48, 1),
            spatter_tip(48, 1, 12),
            bristle_tip(48, 1),
            charcoal_tip(48, 1),
            leaf_tip(48),
            grass_tip(48, 1, 9),
            sponge_tip(48, 1),
            star_tip(48),
            rake_tip(48, 1, 7),
        ] {
            assert!(t.is_valid());
            let sum: f32 = t.to_f32().iter().sum();
            assert!(sum > 48.0, "{sum}");
            assert_eq!(t.get(0, 0), 0.0);
        }
    }
}
