//! Image › Adjustments › Shadows/Highlights, as Photoshop 25.4 does it.
//!
//! Measured on Photoshop as a black box (synthetic 16-bit images through the adjustment, results
//! read back exactly); no Photoshop code was read. The pieces:
//!
//! - **Neighbourhood brightness.** Per pixel, the mean of its brightest channel and its Rec. 601
//!   luma. Its neighbourhood value is an edge-aware average: a Gaussian of σ = 1.14 · Radius whose
//!   weights fall linearly with the difference from the pixel's own (lightly blurred) brightness,
//!   so pixels across a strong edge count less and halos stay small. It is quantised to 8 bits.
//! - **Tonal mask.** Tone sets which neighbourhoods are treated: shadows take the complement of a
//!   Beta distribution function of the neighbourhood brightness, from Beta(1.95, 23) at Tone 0 to
//!   1 − L at Tone 95 and over; highlights take the same of 1 − L.
//! - **Curve.** Amount sets a curve that screens the image with itself: 1 − (1 − v)^q, blended
//!   between q = 1, 2, 4 and 8 by a strength s = 3 at 100, 2 at 50 and 1 near 15. Shadows lift each
//!   channel by mask × (curve − v); highlights darken with the mirrored curve.
//! - **Colour.** Only the luma of that result is kept: the pixel's own colour differences are
//!   added back, scaled by the Color slider (by exp(h · ΔY / max Y) for positive values, by
//!   1 + h · ΔY / max Y for negative ones) and shortened where they would leave the gamut.
//! - **Midtone** is a global S-curve on the luma, k · (Y − ½) · (4Y(1 − Y))^p.

use crate::fxutil::gauss_blur_n;
use crate::photo_util::par_rows;

use super::ShadowsHighlights;

const W: [f32; 3] = [0.299, 0.587, 0.114];

fn luma(c: [f32; 3]) -> f32 {
    W[0] * c[0] + W[1] * c[1] + W[2] * c[2]
}

/// Neighbourhood σ per pixel of Radius.
const SPATIAL: f32 = 1.1424;
/// σ of the blur of the pixel's own brightness the edge weights compare with.
const GUIDE: f32 = 1.3559;
/// Brightness difference at which a neighbour's weight reaches zero.
const RANGE: f32 = 1.0152;
/// Brightness levels of the piecewise-linear edge-aware average.
const LEVELS: usize = 17;

/// Edge-aware neighbourhood brightness of `lum` (`w × h`) for Photoshop's Radius.
fn neighbourhood(lum: &[f32], w: usize, h: usize, radius: f32) -> Vec<f32> {
    let mut guide = lum.to_vec();
    gauss_blur_n(&mut guide, w, h, 1, GUIDE);
    let sigma = SPATIAL * radius;
    if sigma < 0.3 {
        return guide;
    }
    // The average at each of a few brightness levels, interpolated at the pixel's own level
    // (Durand and Dorsey, "Fast bilateral filtering", 2002). The averages are smooth, so they are
    // taken on a grid `d` times coarser, with σ ≥ 4 cells there.
    let d = ((sigma / 4.0).floor() as usize).max(1);
    let (rw, rh) = (w.div_ceil(d), h.div_ceil(d));
    let cells = rw * rh;
    let step = 1.0 / (LEVELS - 1) as f32;
    let mut avg = vec![0.0f32; LEVELS * cells];
    let mut buf = vec![0.0f32; 2 * cells];
    for (k, out) in avg.chunks_exact_mut(cells).enumerate() {
        let g = k as f32 * step;
        buf.fill(0.0);
        for (i, &l) in lum.iter().enumerate() {
            let wgt = (1.0 - (l - g).abs() / RANGE).max(0.0);
            let c = 2 * ((i / w) / d * rw + (i % w) / d);
            buf[c] += wgt * l;
            buf[c + 1] += wgt;
        }
        gauss_blur_n(&mut buf, rw, rh, 2, sigma / d as f32);
        for (o, b) in out.iter_mut().zip(buf.as_chunks::<2>().0) {
            *o = if b[1] > 1e-6 { b[0] / b[1] } else { g };
        }
    }
    let mut out = vec![0.0f32; lum.len()];
    par_rows(&mut out, w, 1, |y, row| {
        let fy = ((y as f32 + 0.5) / d as f32 - 0.5).clamp(0.0, (rh - 1) as f32);
        let (y0, ty) = (fy.floor() as usize, fy.fract());
        let y1 = (y0 + 1).min(rh - 1);
        for (x, o) in row.iter_mut().enumerate() {
            let fx = ((x as f32 + 0.5) / d as f32 - 0.5).clamp(0.0, (rw - 1) as f32);
            let (x0, tx) = (fx.floor() as usize, fx.fract());
            let x1 = (x0 + 1).min(rw - 1);
            let at = |k: usize| {
                let a = &avg[k * cells..(k + 1) * cells];
                let top = a[y0 * rw + x0] + (a[y0 * rw + x1] - a[y0 * rw + x0]) * tx;
                let bot = a[y1 * rw + x0] + (a[y1 * rw + x1] - a[y1 * rw + x0]) * tx;
                top + (bot - top) * ty
            };
            let t = (guide[y * w + x].clamp(0.0, 1.0) / step).min((LEVELS - 1) as f32 - 1e-4);
            let k = t.floor() as usize;
            let f = t - k as f32;
            *o = at(k) * (1.0 - f) + at(k + 1) * f;
        }
    });
    out
}

/// Beta(α, β) parameters of the tonal mask for Tone 0, 5, …, 95 (fitted to Photoshop's masks
/// within 2/255).
const MASK_BETA: [(f32, f32); 20] = [
    (1.9468, 22.9366),
    (1.7770, 18.3471),
    (1.6403, 14.7559),
    (1.5270, 11.9346),
    (1.4269, 9.6467),
    (1.3470, 7.8430),
    (1.2807, 6.4038),
    (1.2240, 5.2384),
    (1.1772, 4.2967),
    (1.1373, 3.5350),
    (1.1021, 2.9109),
    (1.0755, 2.4186),
    (1.0508, 2.0189),
    (1.0307, 1.7034),
    (1.0140, 1.4583),
    (1.0015, 1.2758),
    (1.0, 1.1403),
    (1.0, 1.0510),
    (1.0, 1.0002),
    (1.0, 1.0),
];

/// The tonal mask for `tone` (0..=100) at the 256 neighbourhood levels: 1 − I_L(α, β).
fn tonal_mask(tone: f32) -> [f32; 256] {
    let t = (tone.clamp(0.0, 95.0)) / 5.0;
    let i = (t.floor() as usize).min(MASK_BETA.len() - 2);
    let f = t - i as f32;
    let (a0, b0) = MASK_BETA[i];
    let (a1, b1) = MASK_BETA[i + 1];
    let (a, b) = ((a0 + (a1 - a0) * f) as f64, (b0 + (b1 - b0) * f) as f64);
    // Midpoint sums of the Beta density, 16 per level.
    const SUB: usize = 16;
    let n = 255 * SUB;
    let dens = |x: f64| ((a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln()).exp();
    let mut cum = vec![0.0f64; 256];
    let mut acc = 0.0;
    for j in 0..n {
        acc += dens((j as f64 + 0.5) / n as f64);
        if (j + 1) % SUB == 0 {
            cum[(j + 1) / SUB] = acc;
        }
    }
    let total = acc.max(f64::MIN_POSITIVE);
    std::array::from_fn(|k| (1.0 - cum[k] / total) as f32)
}

/// Curve strength for Amount `a` (0..=100): 0 at 0, 1 near 15, 2 at 50, 3 at 100.
fn strength(a: f32) -> f32 {
    let u = (a.clamp(0.0, 100.0) / 100.0).powf(0.6216);
    if u <= 0.3 { u / 0.3 } else { (1.0 + 20.0 * u) / 7.0 }
}

/// The full-strength curve of strength `s`: 1 − (1 − v)^q between q = 2^⌊s⌋ and 2^(⌊s⌋+1).
fn lift(s: f32, v: f32) -> f32 {
    let n = (s.floor() as i32).clamp(0, 2);
    let t = (s - n as f32).clamp(0.0, 1.0);
    let c = (1.0 - v).clamp(0.0, 1.0);
    let lo = c.powi(1 << n);
    let hi = lo * lo;
    1.0 - ((1.0 - t) * lo + t * hi)
}

/// Linear interpolation in a table of `(x, y)` sorted by `x`.
fn interp(table: &[(f32, f32)], x: f32) -> f32 {
    let Some(&(x0, y0)) = table.first() else { return 0.0 };
    if x <= x0 {
        return y0;
    }
    for p in table.windows(2) {
        let ((xa, ya), (xb, yb)) = (p[0], p[1]);
        if x <= xb {
            return ya + (yb - ya) * (x - xa) / (xb - xa);
        }
    }
    table.last().map_or(0.0, |p| p.1)
}

/// Color: the h of the colour-difference gain for positive values (exp(h · F))…
const COLOR_UP: [(f32, f32); 12] = [
    (0.0, 0.0),
    (1.0, 0.1566),
    (2.0, 0.2271),
    (5.0, 0.3776),
    (10.0, 0.5653),
    (20.0, 0.8669),
    (30.0, 1.1299),
    (40.0, 1.3748),
    (50.0, 1.6093),
    (60.0, 1.8375),
    (75.0, 2.1721),
    (100.0, 2.7193),
];
/// … and for negative values (1 + h · F), by −Color.
const COLOR_DOWN: [(f32, f32); 9] =
    [(0.0, 0.0), (1.0, -0.0145), (5.0, -0.0709), (10.0, -0.1373), (20.0, -0.2570), (30.0, -0.3612), (50.0, -0.5312), (75.0, -0.6866), (100.0, -0.7968)];

/// Midtone: (value, k, p) of k · (Y − ½) · (4Y(1 − Y))^p.
const MIDTONE: [(f32, f32, f32); 11] = [
    (-100.0, -0.6955, 2.267),
    (-75.0, -0.5873, 2.362),
    (-50.0, -0.4429, 2.492),
    (-25.0, -0.2514, 2.667),
    (-10.0, -0.1088, 2.803),
    (0.0, 0.0, 2.9),
    (10.0, 0.1206, 3.016),
    (25.0, 0.3262, 3.216),
    (50.0, 0.7423, 3.625),
    (75.0, 1.2635, 4.160),
    (100.0, 1.9032, 4.850),
];

fn midtone(m: f32, y: f32) -> f32 {
    if m == 0.0 {
        return y;
    }
    let ks: Vec<(f32, f32)> = MIDTONE.iter().map(|r| (r.0, r.1)).collect();
    let ps: Vec<(f32, f32)> = MIDTONE.iter().map(|r| (r.0, r.2)).collect();
    let (k, p) = (interp(&ks, m), interp(&ps, m));
    (y + k * (y - 0.5) * (4.0 * y * (1.0 - y)).max(0.0).powf(p)).clamp(0.0, 1.0)
}

fn level(v: f32) -> usize {
    (v.clamp(0.0, 1.0) * 255.0).round() as usize
}

/// Image › Adjustments › Shadows/Highlights on straight RGBA `w × h`.
pub fn shadows_highlights(px: &mut [[f32; 4]], w: usize, h: usize, p: &ShadowsHighlights) {
    if px.is_empty() || px.len() != w * h {
        return;
    }
    let (sa, ha) = (p.shadow_amount, p.highlight_amount);
    let lum: Vec<f32> = px.iter().map(|q| (q[0].max(q[1]).max(q[2]) + luma([q[0], q[1], q[2]])) / 2.0).collect();
    let ls = (sa > 0.0).then(|| neighbourhood(&lum, w, h, p.shadow_radius));
    let lh = (ha > 0.0).then(|| match &ls {
        Some(l) if (p.highlight_radius - p.shadow_radius).abs() < 1e-3 => l.clone(),
        _ => neighbourhood(&lum, w, h, p.highlight_radius),
    });
    let (ms, mh) = (tonal_mask(p.shadow_tone), tonal_mask(p.highlight_tone));
    let (ss, hs) = (strength(sa), strength(ha));
    // The new luma of every pixel.
    let mut y2 = vec![0.0f32; px.len()];
    par_rows(&mut y2, w, 1, |y, row| {
        for (x, out) in row.iter_mut().enumerate() {
            let i = y * w + x;
            let q = px[i];
            let mut adj = [q[0], q[1], q[2]];
            if let Some(l) = &ls {
                let m = ms[level(l[i])];
                for v in &mut adj {
                    *v += m * (lift(ss, *v) - *v);
                }
            }
            if let Some(l) = &lh {
                let m = mh[255 - level(l[i])];
                for (v, &o) in adj.iter_mut().zip(&q[..3]) {
                    *v -= m * (lift(hs, 1.0 - o) - (1.0 - o));
                }
            }
            *out = midtone(p.midtone, luma(adj).clamp(0.0, 1.0));
        }
    });
    clip_levels(&mut y2, px, p.black_clip, p.white_clip);
    let gain = |f: f32| {
        if p.color >= 0.0 { (interp(&COLOR_UP, p.color) * f).exp() } else { 1.0 + interp(&COLOR_DOWN, -p.color) * f }
    };
    par_rows(px, w, 1, |y, row| {
        for (x, q) in row.iter_mut().enumerate() {
            if q[3] <= 0.0 {
                continue;
            }
            let y1 = luma([q[0], q[1], q[2]]);
            let yn = y2[y * w + x];
            let f = (yn - y1).abs() / yn.max(y1).max(1e-6);
            let g = gain(f);
            let d = [(q[0] - y1) * g, (q[1] - y1) * g, (q[2] - y1) * g];
            // Shorten the colour difference where it would leave 0..=1.
            let mut k = 1.0f32;
            for v in d {
                if v > 1e-9 {
                    k = k.min((1.0 - yn) / v);
                } else if v < -1e-9 {
                    k = k.min(yn / -v);
                }
            }
            let k = k.clamp(0.0, 1.0);
            for c in 0..3 {
                q[c] = (yn + k * d[c]).clamp(0.0, 1.0);
            }
        }
    });
}

/// Black and White Clip: stretches the new luma so `black`% of pixels reach 0 and `white`% reach 1.
fn clip_levels(y2: &mut [f32], px: &[[f32; 4]], black: f32, white: f32) {
    if black <= 0.0 && white <= 0.0 {
        return;
    }
    let mut hist = vec![0u32; 4096];
    let mut n = 0u32;
    for (&y, q) in y2.iter().zip(px) {
        if q[3] > 0.0 {
            hist[(y.clamp(0.0, 1.0) * 4095.0).round() as usize] += 1;
            n += 1;
        }
    }
    if n == 0 {
        return;
    }
    let find = |pct: f32, rev: bool| -> f32 {
        let target = (pct.clamp(0.0, 50.0) / 100.0 * n as f32) as u32;
        let mut acc = 0;
        for k in 0..4096 {
            let i = if rev { 4095 - k } else { k };
            acc += hist[i];
            if acc > target {
                return i as f32 / 4095.0;
            }
        }
        if rev { 0.0 } else { 1.0 }
    };
    let (lo, hi) = (if black > 0.0 { find(black, false) } else { 0.0 }, if white > 0.0 { find(white, true) } else { 1.0 });
    if hi - lo < 1e-3 {
        return;
    }
    for y in y2.iter_mut() {
        *y = ((*y - lo) / (hi - lo)).clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat `w × h` grey or colour image with one different pixel in the middle; returns that
    /// pixel after Shadows/Highlights.
    fn probe(bg: [f32; 3], px: [f32; 3], p: &ShadowsHighlights) -> [f32; 3] {
        let (w, h) = (121, 121);
        let mut img = vec![[bg[0], bg[1], bg[2], 1.0]; w * h];
        img[60 * w + 60] = [px[0], px[1], px[2], 1.0];
        shadows_highlights(&mut img, w, h, p);
        let q = img[60 * w + 60];
        [q[0], q[1], q[2]]
    }

    fn levels(v: [i32; 3]) -> [f32; 3] {
        v.map(|c| c as f32 / 32768.0)
    }

    fn off(a: [f32; 3], b: [f32; 3]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs() * 255.0).fold(0.0, f32::max)
    }

    /// Photoshop 25.4's results (16-bit levels of 32768) for grey pixels lifted by the default
    /// Shadows (35 %, Tone 50 %, Radius 30 px; Color 0), in black surroundings and in grey 0.5.
    #[test]
    fn grey_matches_photoshop() {
        let p = ShadowsHighlights { color: 0.0, black_clip: 0.0, white_clip: 0.0, ..Default::default() };
        for (bg, want) in [(0.0, [11389, 19422, 28451, 31931]), (0.5, [5211, 9909, 18229, 25700])] {
            for (v, w) in [0.125f32, 0.25, 0.5, 0.749969].into_iter().zip(want) {
                let got = probe([bg; 3], [v; 3], &p);
                assert!(off(got, levels([w; 3])) < 2.0, "bg {bg} v {v}: {} vs {}", got[0] * 32768.0, w);
            }
        }
    }

    /// Colour pixels keep their colour differences, scaled by Color (+20 here; Amount 35 %,
    /// Radius 10 px), in black, 0.15 and 0.3 grey surroundings: Photoshop's results.
    #[test]
    fn colour_matches_photoshop() {
        let p = ShadowsHighlights { shadow_radius: 10.0, black_clip: 0.0, white_clip: 0.0, ..Default::default() };
        for (bg, px, want) in [
            (0.0, [3277, 1638, 0], [8235, 5202, 2170]),
            (0.0, [8192, 6554, 4915], [19350, 16514, 13676]),
            (4915.0 / 32768.0, [4915, 3277, 1638], [10124, 7424, 4723]),
            (9830.0 / 32768.0, [11469, 9830, 8192], [16540, 14376, 12214]),
        ] {
            let got = probe([bg; 3], levels(px), &p);
            assert!(off(got, levels(want)) < 2.0, "{px:?} on {bg}: {:?} vs {want:?}", got.map(|c| (c * 32768.0).round()));
        }
    }

    /// Across a strong edge the neighbourhood barely leaks (Amount 100 %, Tone 100 %, Radius
    /// 10 px; 0.1 | 0.9): Photoshop's results 8, 2 and 1 pixels either side.
    #[test]
    fn edge_matches_photoshop() {
        let (w, h) = (200, 24);
        let (a, b) = (3277.0 / 32768.0, 29491.0 / 32768.0);
        let mut img: Vec<[f32; 4]> = (0..w * h).map(|i| if i % w < 100 { [a, a, a, 1.0] } else { [b, b, b, 1.0] }).collect();
        let p = ShadowsHighlights {
            shadow_amount: 100.0,
            shadow_tone: 100.0,
            shadow_radius: 10.0,
            color: 0.0,
            black_clip: 0.0,
            white_clip: 0.0,
            ..Default::default()
        };
        shadows_highlights(&mut img, w, h, &p);
        let row = &img[12 * w..13 * w];
        for (x, want) in [(92, 16370), (98, 14439), (99, 12146), (100, 30891), (101, 30403), (102, 30146), (108, 29979)] {
            let got = row[x][0] * 32768.0;
            assert!((got - want as f32).abs() / 32768.0 * 255.0 < 6.0, "x {x}: {got} vs {want}");
        }
    }

    #[test]
    fn strength_and_curve_follow_photoshop() {
        assert!((strength(50.0) - 2.0).abs() < 2e-3 && (strength(100.0) - 3.0).abs() < 1e-4);
        assert!((strength(15.0) - 1.02).abs() < 0.01 && (strength(1.0) - 0.19).abs() < 0.01);
        // Amount 50 is the image screened with itself twice: 1 − (1 − v)^4.
        for v in [0.1f32, 0.3, 0.6] {
            assert!((lift(2.0, v) - (1.0 - (1.0 - v).powi(4))).abs() < 1e-6);
        }
    }

    #[test]
    fn tonal_mask_spans_photoshop_tones() {
        let m95 = tonal_mask(95.0);
        assert!((m95[128] - (1.0 - 128.0 / 255.0)).abs() < 1e-3, "{}", m95[128]);
        assert_eq!(tonal_mask(100.0), m95);
        // Tone 0 still reaches about 7 % into the levels (Photoshop: half at level 18).
        let m0 = tonal_mask(0.0);
        assert!(m0[0] > 0.999 && (m0[18] - 0.5).abs() < 0.03 && m0[60] < 0.02, "{} {}", m0[18], m0[60]);
    }
}
