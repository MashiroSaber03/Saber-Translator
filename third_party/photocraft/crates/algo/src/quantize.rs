//! Colour reduction for Image › Mode › Indexed Color and Bitmap: palette construction (exact,
//! system, web, uniform, perceptual, selective, adaptive), forced colours, and dithering
//! (none, Floyd–Steinberg diffusion, ordered pattern, noise), plus 1-bit conversion methods.
//!
//! Buffers are straight RGBA in 0..=1, row-major. Median cut follows Heckbert (1982); k-means
//! refinement is Lloyd's algorithm; error diffusion is Floyd & Steinberg (1976).

use photocraft_color::convert::srgb_to_lab;
use serde::{Deserialize, Serialize};

mod nearest;
use nearest::Nearest;

pub type Rgb8 = [u8; 3];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaletteKind {
    /// Every colour of the image (fails above 256).
    Exact,
    SystemMac,
    SystemWindows,
    Web,
    Uniform,
    /// Lab median cut + k-means (perceptually weighted).
    Perceptual,
    /// Perceptual, with entries near web-safe colours snapped to them.
    Selective,
    /// RGB median cut weighted by frequency.
    Adaptive,
}

impl PaletteKind {
    pub fn from_id(s: &str) -> Option<Self> {
        Some(match s {
            "exact" => PaletteKind::Exact,
            "system" | "systemMac" | "systemMacOs" => PaletteKind::SystemMac,
            "systemWindows" => PaletteKind::SystemWindows,
            "web" => PaletteKind::Web,
            "uniform" => PaletteKind::Uniform,
            "perceptual" | "localPerceptual" => PaletteKind::Perceptual,
            "selective" | "localSelective" => PaletteKind::Selective,
            "adaptive" | "localAdaptive" => PaletteKind::Adaptive,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Forced {
    None,
    BlackWhite,
    Primaries,
    Web,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dither {
    None,
    Diffusion,
    Pattern,
    Noise,
}

fn to8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn rgb8(p: &[f32; 4]) -> Rgb8 {
    [to8(p[0]), to8(p[1]), to8(p[2])]
}

/// The 216-colour web-safe cube (0, 51, …, 255 per channel).
pub fn web_palette() -> Vec<Rgb8> {
    let mut v = Vec::with_capacity(216);
    for r in 0..6u8 {
        for g in 0..6u8 {
            for b in 0..6u8 {
                v.push([r * 51, g * 51, b * 51]);
            }
        }
    }
    v
}

/// Classic Mac OS system palette: the web cube plus 10-step red, green, blue and gray ramps.
pub fn mac_palette() -> Vec<Rgb8> {
    let mut v = web_palette();
    // Ramps use the values between cube steps (17-unit spacing, skipping multiples of 51).
    let steps: Vec<u8> = (1..=15u8).map(|k| k * 17).filter(|x| x % 51 != 0).collect();
    for &s in steps.iter().rev() {
        v.push([s, 0, 0]);
    }
    for &s in steps.iter().rev() {
        v.push([0, s, 0]);
    }
    for &s in steps.iter().rev() {
        v.push([0, 0, s]);
    }
    for &s in steps.iter().rev() {
        v.push([s, s, s]);
    }
    v.truncate(256);
    v
}

/// Windows default palette: the 20 static system colours plus the web cube and a gray ramp.
pub fn windows_palette() -> Vec<Rgb8> {
    let mut v: Vec<Rgb8> = vec![
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [192, 192, 192],
        [192, 220, 192],
        [166, 202, 240],
        [255, 251, 240],
        [160, 160, 164],
        [128, 128, 128],
        [255, 0, 0],
        [0, 255, 0],
        [255, 255, 0],
        [0, 0, 255],
        [255, 0, 255],
        [0, 255, 255],
        [255, 255, 255],
    ];
    for c in web_palette() {
        if !v.contains(&c) {
            v.push(c);
        }
    }
    let mut g = 8u8;
    while v.len() < 256 {
        if !v.contains(&[g, g, g]) {
            v.push([g, g, g]);
        }
        g = g.wrapping_add(12);
    }
    v
}

/// Up to `n` colours on an equal-step RGB grid.
pub fn uniform_palette(n: usize) -> Vec<Rgb8> {
    let k = ((n as f32).cbrt().floor() as usize).max(2);
    let lv = |i: usize| ((i as f32 / (k - 1) as f32) * 255.0).round() as u8;
    let mut v = Vec::with_capacity(k * k * k);
    for r in 0..k {
        for g in 0..k {
            for b in 0..k {
                v.push([lv(r), lv(g), lv(b)]);
            }
        }
    }
    v
}

fn forced_colors(f: Forced) -> Vec<Rgb8> {
    match f {
        Forced::None => Vec::new(),
        Forced::BlackWhite => vec![[0, 0, 0], [255, 255, 255]],
        Forced::Primaries => vec![[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255], [255, 255, 0]],
        Forced::Web => web_palette(),
    }
}

/// Unique 8-bit colours with counts (opaque pixels only).
fn histogram(px: &[[f32; 4]]) -> Vec<(Rgb8, u32)> {
    let mut m: std::collections::HashMap<Rgb8, u32> = std::collections::HashMap::new();
    for p in px.iter().filter(|p| p[3] >= 0.5) {
        *m.entry(rgb8(p)).or_default() += 1;
    }
    let mut v: Vec<(Rgb8, u32)> = m.into_iter().collect();
    v.sort_unstable();
    v
}

/// Median cut over `pts` (feature vectors + weights) into at most `n` boxes; returns box means.
fn median_cut(pts: &[([f32; 3], u32)], n: usize) -> Vec<[f32; 3]> {
    if pts.is_empty() || n == 0 {
        return Vec::new();
    }
    let mut boxes: Vec<Vec<([f32; 3], u32)>> = vec![pts.to_vec()];
    while boxes.len() < n {
        // Split the box with the largest weighted extent.
        let (bi, axis, score) = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.len() > 1)
            .map(|(i, b)| {
                let mut best = (0usize, -1.0f32);
                for a in 0..3 {
                    let (lo, hi) = b.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p.0[a]), hi.max(p.0[a])));
                    if hi - lo > best.1 {
                        best = (a, hi - lo);
                    }
                }
                let w: f32 = b.iter().map(|p| p.1 as f32).sum();
                (i, best.0, best.1 * w.sqrt())
            })
            .fold((usize::MAX, 0, -1.0f32), |acc, x| if x.2 > acc.2 { x } else { acc });
        if bi == usize::MAX || score <= 0.0 {
            break;
        }
        let mut b = boxes.swap_remove(bi);
        b.sort_by(|p, q| p.0[axis].total_cmp(&q.0[axis]));
        let total: u64 = b.iter().map(|p| u64::from(p.1)).sum();
        let mut acc = 0u64;
        let mut cut = 1;
        for (i, p) in b.iter().enumerate() {
            acc += u64::from(p.1);
            if acc * 2 >= total {
                cut = (i + 1).clamp(1, b.len() - 1);
                break;
            }
        }
        let rest = b.split_off(cut);
        boxes.push(b);
        boxes.push(rest);
    }
    boxes
        .iter()
        .map(|b| {
            let w: f64 = b.iter().map(|p| f64::from(p.1)).sum();
            std::array::from_fn(|a| (b.iter().map(|p| f64::from(p.0[a]) * f64::from(p.1)).sum::<f64>() / w) as f32)
        })
        .collect()
}

/// Lloyd refinement of `centers` over weighted points.
fn kmeans(pts: &[([f32; 3], u32)], centers: &mut [[f32; 3]], iters: usize) {
    if centers.is_empty() {
        // No room left (the forced colours fill the palette): nothing to refine.
        return;
    }
    for _ in 0..iters {
        let lookup = Nearest::new(centers.iter().copied());
        let mut sum = vec![[0.0f64; 3]; centers.len()];
        let mut cnt = vec![0.0f64; centers.len()];
        for (p, w) in pts {
            let i = lookup.find(*p);
            for a in 0..3 {
                sum[i][a] += f64::from(p[a]) * f64::from(*w);
            }
            cnt[i] += f64::from(*w);
        }
        for (i, c) in centers.iter_mut().enumerate() {
            if cnt[i] > 0.0 {
                *c = std::array::from_fn(|a| (sum[i][a] / cnt[i]) as f32);
            }
        }
    }
}

fn lab_to_rgb8(lab: [f32; 3]) -> Rgb8 {
    photocraft_color::convert::lab_to_srgb(lab).map(to8)
}

/// Builds the palette for `px`. `colors` is the requested count (2..=256) for the computed
/// kinds; forced colours are included first and count toward it.
pub fn build_palette(px: &[[f32; 4]], kind: PaletteKind, colors: usize, forced: Forced) -> Result<Vec<Rgb8>, String> {
    let colors = colors.clamp(2, 256);
    let mut pal = forced_colors(forced);
    let room = colors.saturating_sub(pal.len());
    // Fixed palettes, or computed palettes already filled by forced colours, need no scan.
    // Exact still counts every colour to enforce its 256-colour limit.
    let hist = if kind == PaletteKind::Exact || (room > 0 && matches!(kind, PaletteKind::Adaptive | PaletteKind::Perceptual | PaletteKind::Selective)) {
        histogram(px)
    } else {
        Vec::new()
    };
    let computed: Vec<Rgb8> = match kind {
        PaletteKind::Exact => {
            if hist.len() > 256 {
                return Err(format!("the image has {} colours; Exact needs 256 or fewer", hist.len()));
            }
            hist.iter().map(|h| h.0).collect()
        }
        PaletteKind::SystemMac => mac_palette(),
        PaletteKind::SystemWindows => windows_palette(),
        PaletteKind::Web => web_palette(),
        PaletteKind::Uniform => uniform_palette(room.max(8)),
        PaletteKind::Adaptive => {
            if hist.len() <= room {
                hist.iter().map(|h| h.0).collect()
            } else {
                let pts: Vec<([f32; 3], u32)> = hist.iter().map(|(c, n)| (c.map(f32::from), *n)).collect();
                let mut centers = median_cut(&pts, room);
                kmeans(&pts, &mut centers, 2);
                centers.iter().map(|c| c.map(|v| v.round().clamp(0.0, 255.0) as u8)).collect()
            }
        }
        PaletteKind::Perceptual | PaletteKind::Selective => {
            if hist.len() <= room {
                hist.iter().map(|h| h.0).collect()
            } else {
                let pts: Vec<([f32; 3], u32)> = hist.iter().map(|(c, n)| (srgb_to_lab(c.map(|v| f32::from(v) / 255.0)), *n)).collect();
                let mut centers = median_cut(&pts, room);
                kmeans(&pts, &mut centers, 4);
                let mut v: Vec<Rgb8> = centers.iter().map(|c| lab_to_rgb8(*c)).collect();
                if kind == PaletteKind::Selective {
                    // Favour web-safe colours: snap entries within ~10 levels of one.
                    for c in &mut v {
                        let snap = c.map(|x| ((f32::from(x) / 51.0).round() * 51.0) as u8);
                        if c.iter().zip(snap).all(|(a, b)| a.abs_diff(b) <= 10) {
                            *c = snap;
                        }
                    }
                }
                v
            }
        }
    };
    for c in computed {
        if pal.len() >= 256 {
            break;
        }
        if !pal.contains(&c) {
            pal.push(c);
        }
    }
    if matches!(kind, PaletteKind::Adaptive | PaletteKind::Perceptual | PaletteKind::Selective | PaletteKind::Uniform) {
        pal.truncate(colors.max(forced_colors(forced).len()));
    }
    if pal.is_empty() {
        pal.push([0, 0, 0]);
    }
    Ok(pal)
}

/// 8×8 Bayer threshold in -0.5..0.5.
pub fn bayer8(x: usize, y: usize) -> f32 {
    const M: [u8; 64] = [
        0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26, 12, 44, 4, 36, 14, 46, 6, 38, 60, 28, 52, 20, 62, 30, 54, 22, 3, 35, 11, 43, 1, 33, 9, 41,
        51, 19, 59, 27, 49, 17, 57, 25, 15, 47, 7, 39, 13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21,
    ];
    (f32::from(M[(y % 8) * 8 + x % 8]) + 0.5) / 64.0 - 0.5
}

fn hash01(x: usize, y: usize, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    (h & 0xffff) as f32 / 65535.0
}

/// Maps every opaque pixel to its palette entry (in place) with the chosen dither; returns the
/// index per pixel. Pixels with alpha < ½ become index `transparent` (and alpha 0) when given;
/// otherwise alpha is kept. `amount` (0..=1) scales diffusion / pattern / noise strength.
pub fn quantize(px: &mut [[f32; 4]], w: usize, pal: &[Rgb8], dither: Dither, amount: f32, transparent: Option<usize>) -> Vec<u8> {
    let lookup = Nearest::new(pal.iter().map(|c| c.map(f32::from)));
    quantize_with(px, w, pal, dither, amount, transparent, |c| lookup.find(c.map(|v| v * 255.0)))
}

fn quantize_with(
    px: &mut [[f32; 4]],
    w: usize,
    pal: &[Rgb8],
    dither: Dither,
    amount: f32,
    transparent: Option<usize>,
    nearest: impl Fn([f32; 3]) -> usize + Sync,
) -> Vec<u8> {
    let h = px.len().checked_div(w).unwrap_or(0);
    let mut idx = vec![0u8; px.len()];
    // Typical palette spacing sets the ordered/noise dither amplitude.
    let spread = (1.0 / (pal.len() as f32).cbrt().max(1.0)).min(0.5) * amount.clamp(0.0, 1.0);
    // None / Pattern / Noise dither have no cross-pixel dependency: map every pixel in parallel.
    // (Diffusion propagates error between pixels, so it stays serial below.)
    if dither != Dither::Diffusion {
        crate::photo_util::par_rows2(&mut idx, px, w, |y, idxrow, pxrow| {
            for (x, (ip, p)) in idxrow.iter_mut().zip(pxrow.iter_mut()).enumerate() {
                let pv = *p;
                if pv[3] < 0.5 {
                    if let Some(t) = transparent {
                        *ip = t as u8;
                        *p = [0.0, 0.0, 0.0, 0.0];
                        continue;
                    }
                    if pv[3] <= 0.0 {
                        continue;
                    }
                }
                let mut c = [pv[0], pv[1], pv[2]];
                match dither {
                    Dither::Pattern => c = c.map(|v| v + bayer8(x, y) * spread),
                    Dither::Noise => c = c.map(|v| v + (hash01(x, y, 7) - 0.5) * spread),
                    _ => {}
                }
                let c = c.map(|v| v.clamp(0.0, 1.0));
                let k = nearest(c);
                *ip = k as u8;
                let e = pal[k].map(|v| f32::from(v) / 255.0);
                *p = [e[0], e[1], e[2], if transparent.is_some() { 1.0 } else { pv[3] }];
            }
        });
        return idx;
    }
    let mut err = vec![[0.0f32; 3]; 2 * (w + 2)];
    for y in 0..h {
        if dither == Dither::Diffusion {
            // Rows ping-pong: `cur` is row y, `next` row y + 1 (offset by one for x = -1).
            let (cur, next) = err.split_at_mut(w + 2);
            cur.copy_from_slice(next);
            next.iter_mut().for_each(|e| *e = [0.0; 3]);
        }
        for x in 0..w {
            let i = y * w + x;
            let p = px[i];
            if p[3] < 0.5 {
                if let Some(t) = transparent {
                    idx[i] = t as u8;
                    px[i] = [0.0, 0.0, 0.0, 0.0];
                    continue;
                }
                if p[3] <= 0.0 {
                    continue;
                }
            }
            let mut c = [p[0], p[1], p[2]];
            match dither {
                Dither::None => {}
                Dither::Diffusion => {
                    for k in 0..3 {
                        c[k] += err[x + 1][k];
                    }
                }
                Dither::Pattern => {
                    let t = bayer8(x, y) * spread;
                    c = c.map(|v| v + t);
                }
                Dither::Noise => {
                    let t = (hash01(x, y, 7) - 0.5) * spread;
                    c = c.map(|v| v + t);
                }
            }
            let c = c.map(|v| v.clamp(0.0, 1.0));
            let k = nearest(c);
            idx[i] = k as u8;
            let e = pal[k].map(|v| f32::from(v) / 255.0);
            if dither == Dither::Diffusion {
                let a = amount.clamp(0.0, 1.0);
                let d: [f32; 3] = std::array::from_fn(|q| (c[q] - e[q]) * a);
                let (cur, next) = err.split_at_mut(w + 2);
                for q in 0..3 {
                    cur[x + 2][q] += d[q] * 7.0 / 16.0;
                    next[x][q] += d[q] * 3.0 / 16.0;
                    next[x + 1][q] += d[q] * 5.0 / 16.0;
                    next[x + 2][q] += d[q] / 16.0;
                }
            }
            px[i] = [e[0], e[1], e[2], if transparent.is_some() { 1.0 } else { p[3] }];
        }
    }
    idx
}

/// Image › Mode › Bitmap methods.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum BitmapMethod {
    Threshold,
    Pattern,
    Diffusion,
    /// Halftone screen: lines per pixel-cell size, angle in degrees, dot shape.
    Halftone {
        cell: f32,
        angle: f32,
        shape: HalftoneShape,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HalftoneShape {
    Round,
    Ellipse,
    Line,
    Square,
    Diamond,
    Cross,
}

impl HalftoneShape {
    pub fn from_id(s: &str) -> Option<Self> {
        Some(match s {
            "round" => HalftoneShape::Round,
            "ellipse" => HalftoneShape::Ellipse,
            "line" => HalftoneShape::Line,
            "square" => HalftoneShape::Square,
            "diamond" => HalftoneShape::Diamond,
            "cross" => HalftoneShape::Cross,
            _ => return None,
        })
    }
}

/// Spot function in 0..=1 for a position (u, v) in -1..1 within a halftone cell.
fn spot(shape: HalftoneShape, u: f32, v: f32) -> f32 {
    let s = match shape {
        HalftoneShape::Round => 1.0 - (u * u + v * v) / 2.0,
        HalftoneShape::Ellipse => 1.0 - (u * u + 0.6 * v * v) / 1.6,
        HalftoneShape::Line => 1.0 - v.abs(),
        HalftoneShape::Square => 1.0 - u.abs().max(v.abs()),
        HalftoneShape::Diamond => 1.0 - (u.abs() + v.abs()) / 2.0,
        HalftoneShape::Cross => 1.0 - u.abs().min(v.abs()),
    };
    s.clamp(0.0, 1.0)
}

/// Converts gray values (0..=1, row-major `w` wide) to 0/1 in place.
pub fn to_bitmap(gray: &mut [f32], w: usize, method: BitmapMethod) {
    let h = gray.len().checked_div(w).unwrap_or(0);
    match method {
        BitmapMethod::Threshold => gray.iter_mut().for_each(|v| *v = if *v >= 0.5 { 1.0 } else { 0.0 }),
        BitmapMethod::Pattern => {
            for (i, v) in gray.iter_mut().enumerate() {
                *v = if *v > bayer8(i % w, i / w) + 0.5 { 1.0 } else { 0.0 };
            }
        }
        BitmapMethod::Diffusion => {
            let mut buf: Vec<f32> = gray.to_vec();
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let old = buf[i];
                    let new = if old >= 0.5 { 1.0 } else { 0.0 };
                    let e = old - new;
                    gray[i] = new;
                    if x + 1 < w {
                        buf[i + 1] += e * 7.0 / 16.0;
                    }
                    if y + 1 < h {
                        if x > 0 {
                            buf[i + w - 1] += e * 3.0 / 16.0;
                        }
                        buf[i + w] += e * 5.0 / 16.0;
                        if x + 1 < w {
                            buf[i + w + 1] += e / 16.0;
                        }
                    }
                }
            }
        }
        BitmapMethod::Halftone { cell, angle, shape } => {
            let cell = cell.max(2.0);
            let (s, c) = angle.to_radians().sin_cos();
            // Rank the spot function over the cell so coverage is proportional to the tone.
            let mut table: Vec<f32> = (0..32 * 32).map(|k| spot(shape, ((k % 32) as f32 + 0.5) / 16.0 - 1.0, ((k / 32) as f32 + 0.5) / 16.0 - 1.0)).collect();
            table.sort_by(f32::total_cmp);
            let cdf = |x: f32| table.partition_point(|t| *t < x) as f32 / table.len() as f32;
            for (i, v) in gray.iter_mut().enumerate() {
                let (x, y) = ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
                let (rx, ry) = (x * c + y * s, -x * s + y * c);
                let u = (rx / cell).rem_euclid(1.0) * 2.0 - 1.0;
                let vv = (ry / cell).rem_euclid(1.0) * 2.0 - 1.0;
                // Ink (black) covers where the spot function exceeds the gray level.
                *v = if cdf(spot(shape, u, vv)) >= *v { 0.0 } else { 1.0 };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(w: usize, h: usize) -> Vec<[f32; 4]> {
        (0..w * h).map(|i| [(i % w) as f32 / (w - 1) as f32, (i / w) as f32 / (h - 1) as f32, 0.3, 1.0]).collect()
    }

    #[test]
    fn fixed_palettes_have_expected_sizes() {
        assert_eq!(web_palette().len(), 216);
        assert_eq!(mac_palette().len(), 256);
        assert_eq!(windows_palette().len(), 256);
        assert_eq!(uniform_palette(27).len(), 27);
        let mut m = mac_palette();
        m.sort_unstable();
        m.dedup();
        assert_eq!(m.len(), 256, "mac palette entries are unique");
    }

    #[test]
    fn exact_palette_and_failure() {
        let px = vec![[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]];
        assert_eq!(build_palette(&px, PaletteKind::Exact, 256, Forced::None).unwrap().len(), 2);
        assert!(build_palette(&gradient(64, 64), PaletteKind::Exact, 256, Forced::None).is_err());
    }

    #[test]
    fn adaptive_and_perceptual_respect_count_and_forced() {
        let px = gradient(40, 40);
        for kind in [PaletteKind::Adaptive, PaletteKind::Perceptual, PaletteKind::Selective] {
            let pal = build_palette(&px, kind, 16, Forced::BlackWhite).unwrap();
            assert!(pal.len() <= 16 && pal.len() >= 8, "{kind:?} {}", pal.len());
            assert_eq!(&pal[..2], &[[0, 0, 0], [255, 255, 255]]);
        }
    }

    #[test]
    fn forced_colours_filling_the_count_do_not_panic() {
        // 2 colours + Black and White (or 8 + Primaries) leaves no room for
        // computed entries; the median cut used to return no centres and
        // k-means indexed into the empty list.
        let px = vec![[1.0, 1.0, 1.0, 1.0]; 16];
        for kind in [PaletteKind::Adaptive, PaletteKind::Perceptual, PaletteKind::Selective] {
            let pal = build_palette(&px, kind, 2, Forced::BlackWhite).unwrap();
            assert_eq!(pal, vec![[0, 0, 0], [255, 255, 255]], "{kind:?}");
            let pal = build_palette(&gradient(8, 8), kind, 8, Forced::Primaries).unwrap();
            assert_eq!(pal.len(), 8, "{kind:?}");
        }
    }

    #[test]
    #[ignore = "24 MP release timing comparison; run with --release --ignored --nocapture"]
    fn indexed_diffusion_24mp_release_comparison() {
        let source = gradient(6000, 4000);
        let pal = web_palette();
        for run in 0..4 {
            let mut linear = source.clone();
            let mut accelerated = source.clone();
            let mut before = None;
            let mut after = None;
            // Reverse the order on alternating rounds; exclude the first round as warmup.
            for old in if run % 2 == 0 { [true, false] } else { [false, true] } {
                let start = std::time::Instant::now();
                if old {
                    let indices = quantize_with(&mut linear, 6000, &pal, Dither::Diffusion, 0.75, None, |c| linear_rgb_nearest(&pal, c));
                    before = Some((start.elapsed().as_secs_f64() * 1000.0, indices));
                } else {
                    let indices = quantize(&mut accelerated, 6000, &pal, Dither::Diffusion, 0.75, None);
                    after = Some((start.elapsed().as_secs_f64() * 1000.0, indices));
                }
            }
            let (before_ms, before_indices) = before.unwrap();
            let (after_ms, after_indices) = after.unwrap();
            assert_eq!(before_indices.iter().zip(&after_indices).position(|(a, b)| a != b), None, "first differing index");
            assert_eq!(linear.iter().zip(&accelerated).position(|(a, b)| a != b), None, "first differing pixel");
            println!(
                "{}",
                serde_json::json!({"run":run,"warmup":run==0,"pixels":24_000_000,"palette_entries":216,"linear_ms":before_ms,"accelerated_ms":after_ms})
            );
        }
    }

    fn linear_rgb_nearest(pal: &[Rgb8], c: [f32; 3]) -> usize {
        let mut best = (f32::MAX, 0);
        for (i, e) in pal.iter().enumerate() {
            let d: f32 = (0..3).map(|k| (c[k] * 255.0 - f32::from(e[k])).powi(2)).sum();
            if d < best.0 {
                best = (d, i);
            }
        }
        best.1
    }

    #[test]
    fn all_dithers_match_the_original_linear_palette_search_exactly() {
        let mut source = gradient(93, 61);
        for (i, p) in source.iter_mut().enumerate() {
            p[2] = (i % 71) as f32 / 70.0;
            p[3] = [0.0, 0.2, 0.7, 1.0][i % 4];
        }
        for count in [8, 16, 32, 64, 128, 256] {
            let pal = build_palette(&source, PaletteKind::Selective, count, Forced::BlackWhite).unwrap();
            for dither in [Dither::None, Dither::Diffusion, Dither::Pattern, Dither::Noise] {
                for transparent in [None, Some(1)] {
                    let mut actual = source.clone();
                    let mut expected = source.clone();
                    let indices = quantize(&mut actual, 93, &pal, dither, 0.75, transparent);
                    let old_indices = quantize_with(&mut expected, 93, &pal, dither, 0.75, transparent, |c| linear_rgb_nearest(&pal, c));
                    assert_eq!(indices, old_indices, "{count} {dither:?} {transparent:?}");
                    assert_eq!(actual, expected, "{count} {dither:?} {transparent:?}");
                }
            }
        }
    }

    #[test]
    fn quantize_maps_to_palette_with_each_dither() {
        let pal = vec![[0, 0, 0], [255, 255, 255]];
        for d in [Dither::None, Dither::Diffusion, Dither::Pattern, Dither::Noise] {
            let mut px: Vec<[f32; 4]> = (0..64)
                .map(|i| [i as f32 / 63.0; 4])
                .map(|mut p| {
                    p[3] = 1.0;
                    p
                })
                .collect();
            let idx = quantize(&mut px, 8, &pal, d, 1.0, None);
            assert!(px.iter().all(|p| p[0] == 0.0 || p[0] == 1.0), "{d:?}");
            assert_eq!(idx.len(), 64);
            // Mean brightness is preserved by dithering (not by plain thresholding).
            if d == Dither::Diffusion {
                let mean = px.iter().map(|p| p[0]).sum::<f32>() / 64.0;
                assert!((mean - 0.5).abs() < 0.08, "{mean}");
            }
        }
        let mut px = vec![[1.0, 1.0, 1.0, 0.2]];
        let idx = quantize(&mut px, 1, &pal, Dither::None, 1.0, Some(1));
        assert_eq!((idx[0], px[0][3]), (1, 0.0));
    }

    #[test]
    fn bitmap_methods_are_binary_and_track_tone() {
        for m in [
            BitmapMethod::Threshold,
            BitmapMethod::Pattern,
            BitmapMethod::Diffusion,
            BitmapMethod::Halftone { cell: 6.0, angle: 45.0, shape: HalftoneShape::Round },
        ] {
            let mut g = vec![0.25f32; 32 * 32];
            to_bitmap(&mut g, 32, m);
            assert!(g.iter().all(|v| *v == 0.0 || *v == 1.0));
            let mean = g.iter().sum::<f32>() / g.len() as f32;
            if m != BitmapMethod::Threshold {
                assert!((mean - 0.25).abs() < 0.15, "{m:?} {mean}");
            } else {
                assert_eq!(mean, 0.0);
            }
        }
    }
}
