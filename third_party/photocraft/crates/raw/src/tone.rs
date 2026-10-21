//! Camera Raw's Shadows, Highlights, Clarity, Texture and Dehaze on raw files, applied while
//! developing (in linear light, before the tone curve), and its adaptive black point.
//!
//! Measured black-box from Photoshop 25.4 opening 29 Nikon D4 NEFs through Camera Raw (each slider
//! at ±100 and +50, exposure ±1), against PhotoCraft's own linear development of the same files:
//!
//! * The change is a gain in linear light: exposure −1 gives the same gains, so Camera Raw applies
//!   it before its tone curve.
//! * log₂ of the gain is `slider / 100 · C(B − anchor)`. `B` is an edge-preserving base of the log₂
//!   luminance (K. He's guided filter, radius 32 and ε 1 stop² on a copy 1232 px on its long
//!   side): whole areas move together and edges stay sharp. The change is linear in the slider;
//!   each direction has its own curve `C`.
//! * The anchor follows the image, so exposure changes nothing: for Shadows the log₂ power mean
//!   (p = 1.5) of the linear luminance, for Highlights the mean of the log₂ luminance.
//!
//! On frames left out of the fit the curves explain 80–90 % of the per-pixel change (a fixed,
//! absolute anchor: 60–75 %).
//!
//! The black point: Camera Raw's default rendering follows the frame's darkest tones. A third of
//! the 0.1th percentile of the linear luminance is subtracted; that halves the deep-shadow
//! error of a fixed curve. It comes before the Shadows / Highlights gain, as in Camera Raw:
//! the other order left lifted shadows 0.2 stop lighter and flatter than Photoshop's.
//!
//! Clarity, Texture and Dehaze were measured the same way (each at ±100 and +50, and +100 at
//! exposure −1): all three act before the tone curve (the scene-domain change at exposure −1
//! correlates 0.98–0.99 with the one at exposure 0) and follow the image, not absolute levels.
//!
//! * **Clarity** changes detail at every scale above a few pixels: log₂ luminance is split into
//!   bands between edge-preserving bases (guided filter, radii 2, 8, 32 and 128 on the work copy,
//!   ε 1 stop²), each band scaled by a gain that depends on its base's tone relative to the
//!   frame's mean, plus a tone term. +100 roughly doubles midtone detail and darkens darker areas;
//!   −100 flattens the detail of midtones and highlights and leaves the shadows' alone. On frames
//!   left out of the fit the change correlates 0.81 (+100) and 0.92 (−100) with Photoshop's;
//!   +50 is half of +100.
//! * **Texture** works on fine detail only: bands between Gaussians of σ 1…32 px (full size), each
//!   through its own measured response, so small detail is raised more than strong edges
//!   (computed at half size and carried to full size by guided upsampling).
//! * **Dehaze** follows K. He, J. Sun and X. Tang's dark channel prior (*Single Image Haze Removal
//!   Using Dark Channel Prior*, 2009): the airlight `A` is the mean colour of the 0.1 % of pixels
//!   with the brightest dark channel, the transmission `t = 1 − w · dark` from the dark channel
//!   (minimum over RGB of `I / A` and over a 17 px window on the work copy) smoothed by a guided
//!   filter, and each channel becomes `(I − A) / t + A`, with `w` 0.8 at +100 (0.7 fits the
//!   per-pixel change best, 0.85 the overall darkening). −100 adds a uniform veil,
//!   `I (1 − w) + A w`, `w` 0.22. The black point stays the default rendering's.

use crate::par;

mod tables;

/// Camera Raw's Basic-panel sliders that act on raw data before the tone curve, −100…100.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Tone {
    pub shadows: f32,
    pub highlights: f32,
    pub clarity: f32,
    pub texture: f32,
    pub dehaze: f32,
}

impl Tone {
    /// Every slider at 0.
    pub const NEUTRAL: Tone = Tone { shadows: 0.0, highlights: 0.0, clarity: 0.0, texture: 0.0, dehaze: 0.0 };

    pub fn is_neutral(&self) -> bool {
        self.gain_neutral() && self.dehaze == 0.0
    }

    /// No luminance gain: Shadows, Highlights, Clarity and Texture at 0.
    fn gain_neutral(&self) -> bool {
        self.work_neutral() && self.texture == 0.0
    }

    /// Nothing to compute on the work copy: Shadows, Highlights and Clarity at 0.
    fn work_neutral(&self) -> bool {
        self.shadows == 0.0 && self.highlights == 0.0 && self.clarity == 0.0
    }
}

/// Long side of the copy the base is computed on, and the guided filter's radius and ε there.
const WORK_SIDE: usize = 1232;
const BASE_RADIUS: f32 = 32.0;
const BASE_EPS: f32 = 1.0;
/// ε (stops²) of the guided upsampling that carries the change to full size.
const UPSAMPLE_EPS: f32 = 1.0;
/// The curves' first knot (stops from the anchor) and knot spacing.
const LO: f32 = -12.0;
const STEP: f32 = 0.25;

// log₂ gain at +100 / −100 against B − anchor, knots LO, LO + STEP, … (measured; flat or
// extended linearly beyond the measured range, never steeper than 0.8 stop per stop so tones
// keep their order).
const SHADOWS_UP: [f32; 73] = [
    2.142, 2.142, 2.142, 2.142, 2.142, 2.142, 2.142, 2.142, 2.142, 2.142, 2.142, 2.142, 2.142, 2.142, 2.142, 2.142, 2.142, 2.142, 2.142, 2.142, 2.142, 2.142,
    2.142, 2.142, 2.142, 2.142, 2.138, 2.117, 2.074, 2.010, 1.918, 1.797, 1.668, 1.546, 1.429, 1.315, 1.204, 1.093, 0.979, 0.860, 0.734, 0.604, 0.482, 0.378,
    0.288, 0.211, 0.157, 0.127, 0.104, 0.078, 0.056, 0.046, 0.044, 0.044, 0.044, 0.044, 0.044, 0.041, 0.030, 0.014, 0.003, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
];
const SHADOWS_DOWN: [f32; 73] = [
    -2.365, -2.365, -2.365, -2.365, -2.365, -2.365, -2.365, -2.365, -2.365, -2.365, -2.365, -2.365, -2.365, -2.365, -2.365, -2.365, -2.365, -2.365, -2.365,
    -2.365, -2.365, -2.365, -2.365, -2.346, -2.270, -2.155, -2.078, -2.054, -2.031, -1.973, -1.872, -1.739, -1.604, -1.483, -1.372, -1.266, -1.161, -1.053,
    -0.940, -0.821, -0.696, -0.570, -0.455, -0.356, -0.270, -0.197, -0.146, -0.117, -0.098, -0.076, -0.057, -0.048, -0.046, -0.046, -0.046, -0.046, -0.046,
    -0.043, -0.032, -0.014, -0.003, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
];
const HIGHLIGHTS_UP: [f32; 73] = [
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.002, 0.003, 0.004, 0.008, 0.017, 0.030, 0.043, 0.055, 0.070, 0.085, 0.097, 0.106,
    0.114, 0.120, 0.127, 0.142, 0.167, 0.199, 0.239, 0.301, 0.381, 0.470, 0.569, 0.677, 0.796, 0.925, 1.056, 1.176, 1.268, 1.319, 1.339, 1.347, 1.351, 1.357,
    1.364, 1.372, 1.380, 1.388, 1.396, 1.404, 1.412,
];
#[allow(clippy::approx_constant)] // measured values, one of them happens to be close to e
const HIGHLIGHTS_DOWN: [f32; 73] = [
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, -0.004, -0.020, -0.046, -0.068, -0.086, -0.104, -0.116, -0.118, -0.118, -0.118, -0.118, -0.118,
    -0.119, -0.120, -0.123, -0.127, -0.131, -0.144, -0.169, -0.204, -0.245, -0.301, -0.375, -0.452, -0.537, -0.639, -0.753, -0.872, -0.997, -1.132, -1.294,
    -1.475, -1.637, -1.751, -1.836, -1.936, -2.059, -2.191, -2.323, -2.454, -2.586, -2.718, -2.850,
];

/// Clarity: the work-copy radii of the bases between its bands, and their guided-filter ε.
const CLARITY_RADII: [f32; 4] = [2.0, 8.0, 32.0, 128.0];
const CLARITY_EPS: f32 = 1.0;
/// Texture: long side of its copy, the Gaussians' σ there, and the guided upsampling's ε (small:
/// the change must follow fine detail into the full-size pixels).
const TEXTURE_SIDE: usize = 2464;
const TEXTURE_SIGMAS: [f32; 6] = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
const TEXTURE_UPSAMPLE_EPS: f32 = 1e-4;
/// Dehaze: dark-channel window radius and guided smoothing radius (work copy), its ε, the
/// smallest transmission, and the haze weight at +100 and −100.
const HAZE_RADIUS: f32 = 8.0;
const HAZE_SMOOTH: f32 = 64.0;
const HAZE_EPS: f32 = 1e-3;
const HAZE_T0: f32 = 0.1;
const HAZE_UP: f32 = 0.8;
const HAZE_DOWN: f32 = 0.22;
/// Linear ProPhoto → linear Display P3 (both D50): the dark channel was measured in Photoshop's
/// Display P3 output.
const PRO_TO_P3: [[f32; 3]; 3] = [[1.632638, -0.379816, -0.252821], [-0.153846, 1.166859, -0.013013], [0.010511, -0.062886, 1.052375]];

/// A change of log₂ luminance computed on a reduced copy, carried to full size along the image's
/// own edges (guided upsampling): a local linear model `change = a · log₂ Y + b` per reduced pixel.
struct Upsampled {
    sw: usize,
    sh: usize,
    a: Vec<f32>,
    b: Vec<f32>,
    /// log₂ luminance floor (the reduced copy's 0.1th percentile).
    floor: f32,
}

impl Upsampled {
    /// The change at full-size pixel (`x`, `y`) of a `w × h` image, of linear luminance `lum`.
    fn change(&self, w: usize, h: usize, x: usize, y: usize, lum: f32) -> f32 {
        let l = if lum > 0.0 { lum.log2().max(self.floor) } else { self.floor };
        bilinear(&self.a, self.sw, self.sh, w, h, x, y) * l + bilinear(&self.b, self.sw, self.sh, w, h, x, y)
    }
}

/// `v` (`sw × sh`) at full-size pixel (`x`, `y`) of a `w × h` image, bilinearly.
fn bilinear(v: &[f32], sw: usize, sh: usize, w: usize, h: usize, x: usize, y: usize) -> f32 {
    let sx = ((x as f32 + 0.5) * sw as f32 / w as f32 - 0.5).clamp(0.0, (sw - 1) as f32);
    let sy = ((y as f32 + 0.5) * sh as f32 / h as f32 - 0.5).clamp(0.0, (sh - 1) as f32);
    let (x0, y0) = (sx as usize, sy as usize);
    let (x1, y1) = ((x0 + 1).min(sw - 1), (y0 + 1).min(sh - 1));
    let (fx, fy) = (sx - x0 as f32, sy - y0 as f32);
    let top = v[y0 * sw + x0] * (1.0 - fx) + v[y0 * sw + x1] * fx;
    let bottom = v[y1 * sw + x0] * (1.0 - fx) + v[y1 * sw + x1] * fx;
    top * (1.0 - fy) + bottom * fy
}

/// Box means of `f` over `k × k` cells, `k` the smallest factor that brings the long side to
/// `side` or less: (`sw`, `sh`, the means).
fn reduce<const N: usize>(w: usize, h: usize, side: usize, f: impl Fn(usize) -> [f32; N] + Sync) -> (usize, usize, Vec<[f32; N]>) {
    let k = w.max(h).div_ceil(side).max(1);
    let (sw, sh) = (w.div_ceil(k), h.div_ceil(k));
    let rows = par::map(sh, |sy| {
        let (y0, y1) = (sy * k, ((sy + 1) * k).min(h));
        (0..sw)
            .map(|sx| {
                let (x0, x1) = (sx * k, ((sx + 1) * k).min(w));
                let mut s = [0.0f64; N];
                for y in y0..y1 {
                    for x in x0..x1 {
                        let v = f(y * w + x);
                        for c in 0..N {
                            s[c] += f64::from(v[c].max(0.0));
                        }
                    }
                }
                let n = ((x1 - x0) * (y1 - y0)) as f64;
                s.map(|v| (v / n) as f32)
            })
            .collect::<Vec<_>>()
    });
    (sw, sh, rows.concat())
}

/// The luminance change of one image (Shadows, Highlights, Clarity on the work copy; Texture on
/// a half-size copy) and the black point.
pub(crate) struct ToneMap {
    w: usize,
    h: usize,
    work: Option<Upsampled>,
    texture: Option<Upsampled>,
    /// Linear black point to subtract (0: none).
    pub(crate) black: f32,
}

impl ToneMap {
    /// `lum(i)`: linear luminance of full-size pixel `i` (row-major, `w × h`). `black_point`: the
    /// camera profile's black-point factor (see [`crate::Sensor::black_point`]). `None` when
    /// nothing changes. Dehaze is not part of it (see [`Haze`]).
    pub(crate) fn new(w: usize, h: usize, lum: impl Fn(usize) -> f32 + Sync, tone: Tone, black_point: f32) -> Option<Self> {
        if w == 0 || h == 0 || (tone.gain_neutral() && black_point <= 0.0) {
            return None;
        }
        let (sw, sh, y) = reduce(w, h, WORK_SIDE, |i| [lum(i)]);
        let y: Vec<f32> = y.into_iter().map(|[v]| v).collect();
        let p01 = quantile(&y, 0.001).max(2f32.powi(-20));
        let black = if black_point > 0.0 { black_point * p01 } else { 0.0 };
        let work = (!tone.work_neutral()).then(|| {
            let l: Vec<f32> = y.iter().map(|v| v.max(p01).log2()).collect();
            let n = l.len() as f64;
            let geo = (l.iter().map(|v| f64::from(*v)).sum::<f64>() / n) as f32;
            let scale = sw.max(sh) as f32 / WORK_SIDE as f32;
            let mut change = vec![0.0f32; l.len()];
            if tone.shadows != 0.0 || tone.highlights != 0.0 {
                let pm = ((l.iter().map(|v| f64::from(v.exp2()).powf(1.5)).sum::<f64>() / n).log2() / 1.5) as f32;
                let r = ((BASE_RADIUS * scale).round() as usize).max(1);
                let base = guided(&l, &l, sw, sh, r, BASE_EPS);
                for (c, b) in change.iter_mut().zip(&base) {
                    *c += shadows(tone.shadows, b - pm) + highlights(tone.highlights, b - geo);
                }
            }
            if tone.clarity != 0.0 {
                clarity(&mut change, &l, sw, sh, geo, scale, tone.clarity);
            }
            // Fit change ≈ a · l + b locally, averaged; evaluated at full size. The large ε lets the
            // change follow strong edges only: inside texture it stays smooth, so fine detail keeps
            // its contrast as in Camera Raw (with ε 0.01 shadow detail lost a fifth of it; with ε 4 the change bled ~16 px over 3-stop edges).
            let (a, b) = linear_fit(&l, &change, sw, sh, 2, UPSAMPLE_EPS);
            Upsampled { sw, sh, a, b, floor: p01.log2() }
        });
        let texture = (tone.texture != 0.0).then(|| texture(w, h, &lum, tone.texture));
        Some(ToneMap { w, h, work, texture, black })
    }

    /// The gain at full-size pixel (`x`, `y`) of linear luminance `lum`.
    pub(crate) fn gain(&self, x: usize, y: usize, lum: f32) -> f32 {
        if self.work.is_none() && self.texture.is_none() {
            return 1.0;
        }
        let c = [&self.work, &self.texture].iter().filter_map(|u| u.as_ref()).map(|u| u.change(self.w, self.h, x, y, lum)).sum::<f32>();
        c.clamp(-6.0, 6.0).exp2()
    }

    /// Black-point subtraction of one linear value.
    #[inline]
    pub(crate) fn unblack(&self, v: f32) -> f32 {
        if self.black > 0.0 { ((v - self.black) / (1.0 - self.black)).max(0.0) } else { v }
    }
}

/// The black point of an image of luminance `lum` for the profile's factor (0: none): `factor`
/// times the 0.1th percentile of the work copy's linear luminance.
pub(crate) fn black_level(w: usize, h: usize, lum: impl Fn(usize) -> f32 + Sync, factor: f32) -> f32 {
    if w == 0 || h == 0 || factor <= 0.0 {
        return 0.0;
    }
    let (_, _, y) = reduce(w, h, WORK_SIDE, |i| [lum(i)]);
    let y: Vec<f32> = y.into_iter().map(|[v]| v).collect();
    factor * quantile(&y, 0.001).max(2f32.powi(-20))
}

/// The slider's table for its direction and its weight 0…1.
fn direction<T>(s: f32, up: &'static T, down: &'static T) -> (f32, &'static T) {
    let s = s.clamp(-100.0, 100.0) / 100.0;
    if s >= 0.0 { (s, up) } else { (-s, down) }
}

/// A table on evenly spaced knots `lo`, `lo + step`, …, read linearly, flat beyond its ends.
fn even<const N: usize>(c: &[f32; N], lo: f32, step: f32, x: f32) -> f32 {
    let t = ((x - lo) / step).clamp(0.0, (N - 1) as f32);
    let i = (t as usize).min(N - 2);
    c[i] + (c[i + 1] - c[i]) * (t - i as f32)
}

/// Adds Clarity `s` to `change`: bands of `l` between guided bases, each scaled by a gain that
/// follows its base's tone (relative to `geo`), and a tone term of the coarsest base.
fn clarity(change: &mut [f32], l: &[f32], sw: usize, sh: usize, geo: f32, scale: f32, s: f32) {
    let (k, t) = direction(s, &tables::CLARITY_UP, &tables::CLARITY_DOWN);
    let mut prev = l.to_vec();
    for (band, r) in CLARITY_RADII.iter().enumerate() {
        let base = guided(l, l, sw, sh, ((r * scale).round() as usize).max(1), CLARITY_EPS);
        for ((c, p), b) in change.iter_mut().zip(&prev).zip(&base) {
            *c += k * (p - b) * even(&t.bands[band], tables::CLARITY_TONE_LO, tables::CLARITY_TONE_STEP, b - geo);
        }
        prev = base;
    }
    for (c, b) in change.iter_mut().zip(&prev) {
        *c += k * (even(&t.base, tables::CLARITY_BASE_LO, tables::CLARITY_BASE_STEP, b - geo) + t.offset);
    }
}

/// Texture `s` on a half-size copy: each Gaussian band of log₂ luminance through its measured
/// response (zero for flat areas), carried to full size.
fn texture(w: usize, h: usize, lum: &(impl Fn(usize) -> f32 + Sync), s: f32) -> Upsampled {
    let (sw, sh, y) = reduce(w, h, TEXTURE_SIDE, |i| [lum(i)]);
    let y: Vec<f32> = y.into_iter().map(|[v]| v).collect();
    let p01 = quantile(&y, 0.001).max(2f32.powi(-20));
    let l: Vec<f32> = y.iter().map(|v| v.max(p01).log2()).collect();
    let scale = sw.max(sh) as f32 / TEXTURE_SIDE as f32;
    let (k, t) = direction(s, &tables::TEXTURE_UP, &tables::TEXTURE_DOWN);
    let mut change = vec![0.0f32; l.len()];
    let mut prev = l.clone();
    for (band, sigma) in TEXTURE_SIGMAS.iter().enumerate() {
        let base = gauss(&l, sw, sh, sigma * scale);
        let zero = knots(&t[band], 0.0);
        for ((c, p), b) in change.iter_mut().zip(&prev).zip(&base) {
            *c += k * (knots(&t[band], p - b) - zero);
        }
        prev = base;
    }
    let (a, b) = linear_fit(&l, &change, sw, sh, 2, TEXTURE_UPSAMPLE_EPS);
    Upsampled { sw, sh, a, b, floor: p01.log2() }
}

/// Texture's band response on the knots [`tables::TEXTURE_KNOTS`], read linearly, flat beyond.
fn knots(c: &[f32; 11], d: f32) -> f32 {
    let k = &tables::TEXTURE_KNOTS;
    let d = d.clamp(k[0], k[10]);
    let i = k.partition_point(|v| *v <= d).clamp(1, 10);
    let f = (d - k[i - 1]) / (k[i] - k[i - 1]);
    c[i - 1] + (c[i] - c[i - 1]) * f
}

/// Gaussian blur of `v` (`w × h`), σ `sigma`, radius ⌈3σ⌉, mirrored edges.
fn gauss(v: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma <= 0.05 {
        return v.to_vec();
    }
    let r = (3.0 * sigma).ceil() as isize;
    let mut k: Vec<f32> = (-r..=r).map(|x| (-(x * x) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let sum: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= sum);
    let rows = par::map(h, |y| {
        let src = &v[y * w..(y + 1) * w];
        (0..w as isize).map(|x| k.iter().enumerate().map(|(j, kv)| kv * src[mirror(x + j as isize - r, w)]).sum::<f32>()).collect::<Vec<f32>>()
    });
    let tmp = rows.concat();
    let cols = par::map(w, |x| {
        (0..h as isize).map(|y| k.iter().enumerate().map(|(j, kv)| kv * tmp[mirror(y + j as isize - r, h) * w + x]).sum::<f32>()).collect::<Vec<f32>>()
    });
    let mut out = vec![0.0f32; w * h];
    for (x, col) in cols.iter().enumerate() {
        for (y, v) in col.iter().enumerate() {
            out[y * w + x] = *v;
        }
    }
    out
}

/// Minimum over a `(2r + 1)²` window, edges clamped.
fn min_filter(v: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let rows = par::map(h, |y| {
        let src = &v[y * w..(y + 1) * w];
        (0..w).map(|x| src[x.saturating_sub(r)..=(x + r).min(w - 1)].iter().fold(f32::MAX, |a, b| a.min(*b))).collect::<Vec<f32>>()
    });
    let tmp = rows.concat();
    let cols =
        par::map(w, |x| (0..h).map(|y| (y.saturating_sub(r)..=(y + r).min(h - 1)).map(|yy| tmp[yy * w + x]).fold(f32::MAX, f32::min)).collect::<Vec<f32>>());
    let mut out = vec![0.0f32; w * h];
    for (x, col) in cols.iter().enumerate() {
        for (y, v) in col.iter().enumerate() {
            out[y * w + x] = *v;
        }
    }
    out
}

/// Camera Raw's Dehaze on linear ProPhoto RGB (see the module docs).
pub(crate) struct Haze {
    w: usize,
    h: usize,
    sw: usize,
    sh: usize,
    /// Transmission per work-copy pixel (Dehaze > 0); empty for the veil of Dehaze < 0.
    t: Vec<f32>,
    /// Airlight, linear ProPhoto.
    air: [f32; 3],
    /// Weight of the uniform veil (Dehaze < 0).
    veil: f32,
}

impl Haze {
    /// `px(i)`: linear ProPhoto RGB of full-size pixel `i`. `None` at 0.
    pub(crate) fn new(w: usize, h: usize, px: impl Fn(usize) -> [f32; 3] + Sync, amount: f32) -> Option<Self> {
        let amount = amount.clamp(-100.0, 100.0) / 100.0;
        if w == 0 || h == 0 || amount == 0.0 {
            return None;
        }
        let (sw, sh, small) = reduce(w, h, WORK_SIDE, px);
        let p3 = |v: &[f32; 3]| PRO_TO_P3.map(|row| (row[0] * v[0] + row[1] * v[1] + row[2] * v[2]).max(0.0));
        let small_p3: Vec<[f32; 3]> = small.iter().map(p3).collect();
        // Airlight: mean colour of the 0.1 % of pixels with the brightest dark channel.
        let dc: Vec<f32> = small_p3.iter().map(|c| c[0].min(c[1]).min(c[2])).collect();
        let top = (dc.len() / 1000).max(1);
        let mut idx: Vec<usize> = (0..dc.len()).collect();
        idx.select_nth_unstable_by(top - 1, |a, b| dc[*b].total_cmp(&dc[*a]));
        let mut air = [0.0f32; 3];
        for i in &idx[..top] {
            for c in 0..3 {
                air[c] += small[*i][c] / top as f32;
            }
        }
        let air = air.map(|v| v.max(1e-6));
        if amount < 0.0 {
            return Some(Haze { w, h, sw, sh, t: Vec::new(), air, veil: HAZE_DOWN * -amount });
        }
        let air_p3 = p3(&air).map(|v| v.max(1e-6));
        let scale = sw.max(sh) as f32 / WORK_SIDE as f32;
        let dark0: Vec<f32> = small_p3.iter().map(|c| (0..3).map(|k| c[k] / air_p3[k]).fold(f32::MAX, f32::min)).collect();
        let dark = min_filter(&dark0, sw, sh, ((HAZE_RADIUS * scale).round() as usize).max(1));
        let y = crate::color::rgb_to_xyz(crate::color::ROMM, crate::color::D50_XY)[1].map(|v| v as f32);
        let lum = |c: &[f32; 3]| y[0] * c[0] + y[1] * c[1] + y[2] * c[2];
        let air_y = lum(&air).max(1e-6);
        let guide: Vec<f32> = small.iter().map(|c| lum(c) / air_y).collect();
        let smooth = guided(&guide, &dark, sw, sh, ((HAZE_SMOOTH * scale).round() as usize).max(1), HAZE_EPS);
        let wgt = HAZE_UP * amount;
        let t = smooth.iter().map(|d| (1.0 - wgt * d).max(HAZE_T0)).collect();
        Some(Haze { w, h, sw, sh, t, air, veil: 0.0 })
    }

    /// Dehaze at full-size pixel (`x`, `y`) of linear ProPhoto `v`.
    pub(crate) fn apply(&self, x: usize, y: usize, v: [f32; 3]) -> [f32; 3] {
        if self.t.is_empty() {
            return [0, 1, 2].map(|c| v[c] * (1.0 - self.veil) + self.air[c] * self.veil);
        }
        let t = bilinear(&self.t, self.sw, self.sh, self.w, self.h, x, y).max(HAZE_T0);
        [0, 1, 2].map(|c| ((v[c] - self.air[c]) / t + self.air[c]).max(0.0))
    }
}

fn curve(c: &[f32; 73], d: f32) -> f32 {
    let t = ((d - LO) / STEP).clamp(0.0, (c.len() - 1) as f32);
    let i = (t as usize).min(c.len() - 2);
    let f = t - i as f32;
    c[i] + (c[i + 1] - c[i]) * f
}

/// log₂ gain of Shadows `s` (−100…100) at `d` stops from its anchor.
fn shadows(s: f32, d: f32) -> f32 {
    let s = s.clamp(-100.0, 100.0) / 100.0;
    if s > 0.0 {
        s * curve(&SHADOWS_UP, d)
    } else if s < 0.0 {
        -s * curve(&SHADOWS_DOWN, d)
    } else {
        0.0
    }
}

/// log₂ gain of Highlights `h` (−100…100) at `d` stops from its anchor.
fn highlights(h: f32, d: f32) -> f32 {
    let h = h.clamp(-100.0, 100.0) / 100.0;
    if h > 0.0 {
        h * curve(&HIGHLIGHTS_UP, d)
    } else if h < 0.0 {
        -h * curve(&HIGHLIGHTS_DOWN, d)
    } else {
        0.0
    }
}

/// The `q` quantile (0…1) of `v`.
fn quantile(v: &[f32], q: f32) -> f32 {
    let mut s = v.to_vec();
    let k = ((s.len() - 1) as f32 * q).round() as usize;
    let (_, m, _) = s.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
    *m
}

/// Index `i` mirrored into `0..n` (half-sample symmetric: … 1 0 | 0 1 … n−1 | n−1 n−2 …).
fn mirror(i: isize, n: usize) -> usize {
    let n = n as isize;
    let p = i.rem_euclid(2 * n);
    (if p < n { p } else { 2 * n - 1 - p }) as usize
}

/// Box mean of radius `r` with mirrored edges.
fn box_mean(v: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let k = (2 * r + 1) as f64;
    let ri = r as isize;
    let rows = par::map(h, |y| {
        let src = &v[y * w..(y + 1) * w];
        let mut acc: f64 = (-ri..=ri).map(|i| f64::from(src[mirror(i, w)])).sum();
        let mut out = vec![0.0f32; w];
        for (x, o) in out.iter_mut().enumerate() {
            *o = (acc / k) as f32;
            let x = x as isize;
            acc += f64::from(src[mirror(x + ri + 1, w)]) - f64::from(src[mirror(x - ri, w)]);
        }
        out
    });
    let tmp = rows.concat();
    let cols = par::map(w, |x| {
        let at = |y: isize| f64::from(tmp[mirror(y, h) * w + x]);
        let mut acc: f64 = (-ri..=ri).map(at).sum();
        let mut out = vec![0.0f32; h];
        for (y, o) in out.iter_mut().enumerate() {
            *o = (acc / k) as f32;
            let y = y as isize;
            acc += at(y + ri + 1) - at(y - ri);
        }
        out
    });
    let mut out = vec![0.0f32; w * h];
    for (x, col) in cols.iter().enumerate() {
        for (y, v) in col.iter().enumerate() {
            out[y * w + x] = *v;
        }
    }
    out
}

/// Local linear fit `p ≈ a · i + b` over boxes of radius `r` (regularized by `eps`), with `a`
/// and `b` box-averaged (K. He, J. Sun, X. Tang, *Guided Image Filtering*, 2010).
fn linear_fit(i: &[f32], p: &[f32], w: usize, h: usize, r: usize, eps: f32) -> (Vec<f32>, Vec<f32>) {
    let ip: Vec<f32> = i.iter().zip(p).map(|(a, b)| a * b).collect();
    let ii: Vec<f32> = i.iter().map(|a| a * a).collect();
    let (mi, mp, mip, mii) = (box_mean(i, w, h, r), box_mean(p, w, h, r), box_mean(&ip, w, h, r), box_mean(&ii, w, h, r));
    let a: Vec<f32> = (0..w * h).map(|k| (mip[k] - mi[k] * mp[k]) / ((mii[k] - mi[k] * mi[k]).max(0.0) + eps)).collect();
    let b: Vec<f32> = (0..w * h).map(|k| mp[k] - a[k] * mi[k]).collect();
    (box_mean(&a, w, h, r), box_mean(&b, w, h, r))
}

/// Guided filter of `p` with guide `i`.
fn guided(i: &[f32], p: &[f32], w: usize, h: usize, r: usize, eps: f32) -> Vec<f32> {
    let (a, b) = linear_fit(i, p, w, h, r, eps);
    a.iter().zip(&b).zip(i).map(|((a, b), i)| a * i + b).collect()
}

/// The raw-stage sliders ([`Tone`]) on an image [`develop`](crate::develop) already made (gamma-1.8 ProPhoto
/// RGBA, 0…1) with the profile tone `curve`, for a preview of the open-time setting: the curve is
/// undone, the change applied in linear light and the curve redone. The developed pixels already
/// carry the black point, so the change's map is computed from the luminance after it rather
/// than before; that only differs in the deepest shadows.
pub fn retone(px: &mut [[f32; 4]], w: usize, h: usize, curve: &[[f32; 2]], tone: Tone) {
    if tone.is_neutral() || w == 0 || h == 0 || px.len() < w * h {
        return;
    }
    let forward = |v: f32| crate::develop::tone(curve, v.clamp(0.0, 1.0)).clamp(0.0, 1.0).powf(1.0 / 1.8);
    // The inverse from a table over s = √v (the encoding is steep near 0).
    const N: usize = 4096;
    let table: Vec<f32> = (0..=N).map(|i| forward((i as f32 / N as f32).powi(2))).collect();
    let inverse = |e: f32| {
        let i = table.partition_point(|t| *t < e).clamp(1, N);
        let (a, b) = (table[i - 1], table[i]);
        let f = if b > a { ((e - a) / (b - a)).clamp(0.0, 1.0) } else { 0.0 };
        ((i - 1) as f32 + f) / N as f32
    };
    let mut lin: Vec<[f32; 3]> = px.iter().map(|p| [0, 1, 2].map(|k| inverse(p[k]).powi(2))).collect();
    if let Some(haze) = Haze::new(w, h, |i| lin[i], tone.dehaze) {
        for (i, l) in lin.iter_mut().enumerate() {
            *l = haze.apply(i % w, i / w, *l);
        }
    }
    let y = crate::color::rgb_to_xyz(crate::color::ROMM, crate::color::D50_XY)[1].map(|v| v as f32);
    let lum = |c: &[f32; 3]| y[0] * c[0] + y[1] * c[1] + y[2] * c[2];
    let map = ToneMap::new(w, h, |i| lum(&lin[i]), tone, 0.0);
    for (i, (p, l)) in px.iter_mut().zip(&lin).enumerate() {
        let g = map.as_ref().map_or(1.0, |m| m.gain(i % w, i / w, lum(l)));
        for k in 0..3 {
            p[k] = forward(l[k] * g);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHADOWS: Tone = Tone { shadows: 100.0, ..Tone::NEUTRAL };
    const HIGHLIGHTS: Tone = Tone { highlights: -100.0, ..Tone::NEUTRAL };

    /// Three vertical bands of linear luminance 2⁻⁷ | 2⁻⁴ | 2⁻¹, times `scale`.
    fn bands(w: usize, h: usize, scale: f32) -> Vec<f32> {
        (0..w * h).map(|i| scale * [2f32.powi(-7), 2f32.powi(-4), 0.5][(i % w) * 3 / w]).collect()
    }

    fn gains(y: &[f32], w: usize, h: usize, tone: Tone) -> Vec<f32> {
        let Some(m) = ToneMap::new(w, h, |i| y[i], tone, 0.0) else { return vec![1.0; w * h] };
        (0..w * h).map(|i| m.gain(i % w, i / w, y[i])).collect()
    }

    #[test]
    fn shadows_lift_the_dark_area_and_keep_the_bright_one() {
        let (w, h) = (300, 120);
        let g = gains(&bands(w, h, 1.0), w, h, SHADOWS);
        let (dark, mid, bright) = (g[60 * w + 50], g[60 * w + 150], g[60 * w + 250]);
        // 5 stops below the anchor: about +2 stops, as Camera Raw lifts such areas; 1 stop above it: almost nothing.
        assert!(dark.log2() > 1.8 && dark.log2() < 2.3, "{dark}");
        assert!(mid > 1.0 && mid < dark, "{mid}");
        assert!(bright.log2() < 0.06, "{bright}");
    }

    #[test]
    fn highlights_lower_the_bright_area_and_keep_the_dark_one() {
        let (w, h) = (300, 120);
        let g = gains(&bands(w, h, 1.0), w, h, HIGHLIGHTS);
        let (dark, bright) = (g[60 * w + 50], g[60 * w + 250]);
        // 3 stops above the anchor about −1 stop; 3 below a slight darkening (Camera Raw lowers a
        // high-contrast frame as a whole a little too).
        assert!(dark.log2() < 0.0 && dark.log2() > -0.15, "{dark}");
        assert!(bright.log2() < -0.8 && bright.log2() > -1.3, "{bright}");
    }

    #[test]
    fn exposure_changes_nothing() {
        let (w, h) = (300, 120);
        for tone in [SHADOWS, HIGHLIGHTS, Tone { shadows: -60.0, highlights: 40.0, clarity: 50.0, texture: -40.0, ..Tone::NEUTRAL }] {
            let (a, b) = (gains(&bands(w, h, 1.0), w, h, tone), gains(&bands(w, h, 0.25), w, h, tone));
            assert!(a.iter().zip(&b).all(|(a, b)| (a - b).abs() < 1e-3 * a), "{tone:?}");
        }
    }

    #[test]
    fn areas_move_as_a_whole_without_a_halo() {
        // Next to an edge the change stays between the two areas' changes (no over- or undershoot)
        // and close to the change inside the area.
        let (w, h) = (600, 120);
        let g: Vec<f32> = gains(&bands(w, h, 1.0), w, h, SHADOWS).iter().map(|g| g.log2()).collect();
        let row = &g[60 * w..61 * w];
        let (dark, mid) = (row[w / 6], row[w / 2]);
        for (x, v) in row.iter().enumerate().take(w / 2).skip(w / 6) {
            assert!(*v <= dark + 1e-3 && *v >= mid - 1e-3, "{x}: {v} outside {mid}…{dark}");
        }
        // Within 0.3 stop: near strong edges the change matches Photoshop as well as elsewhere.
        assert!(dark - row[w / 3 - 2] < 0.3 && row[w / 3 + 2] - mid < 0.3, "{} {} | {} {}", dark, row[w / 3 - 2], row[w / 3 + 2], mid);
    }

    #[test]
    fn the_result_does_not_depend_on_the_image_size() {
        let probe = |w: usize, h: usize| {
            let g = gains(&bands(w, h, 1.0), w, h, SHADOWS);
            [w / 6, w / 2, 5 * w / 6].map(|x| g[(h / 2) * w + x].log2())
        };
        let (a, b) = (probe(4928, 1640), probe(1232, 410));
        assert!(a.iter().zip(&b).all(|(a, b)| (a - b).abs() < 0.02), "{a:?} vs {b:?}");
    }

    #[test]
    fn the_black_point_follows_the_darkest_tones() {
        let (w, h) = (300, 120);
        let y = bands(w, h, 1.0);
        let m = ToneMap::new(w, h, |i| y[i], Tone::default(), 0.33).unwrap();
        assert!((m.black - 0.33 * 2f32.powi(-7)).abs() < 1e-7, "{}", m.black);
        assert!(m.unblack(m.black * 0.5) == 0.0 && (m.unblack(1.0) - 1.0).abs() < 1e-6);
        assert!(ToneMap::new(w, h, |i| y[i], Tone::default(), 0.0).is_none());
    }

    /// log₂ luminance of a `w × h` image: `base` plus a vertical sine of `amp` stops and period `period` px.
    fn stripes(w: usize, h: usize, base: f32, amp: f32, period: f32) -> Vec<f32> {
        (0..w * h).map(|i| (base + amp * ((i % w) as f32 * std::f32::consts::TAU / period).sin()).exp2()).collect()
    }

    /// Peak-to-peak log₂ luminance over the middle row's central half, after the gains.
    fn swing(y: &[f32], g: &[f32], w: usize, h: usize) -> f32 {
        let row = (h / 2) * w;
        let v: Vec<f32> = (w / 4..3 * w / 4).map(|x| (y[row + x] * g[row + x]).log2()).collect();
        v.iter().cloned().fold(f32::MIN, f32::max) - v.iter().cloned().fold(f32::MAX, f32::min)
    }

    #[test]
    fn clarity_raises_midtone_detail_and_minus_flattens_it() {
        let (w, h) = (1232, 200);
        let y = stripes(w, h, -2.5, 0.25, 40.0);
        let before = swing(&y, &vec![1.0; w * h], w, h);
        let up = swing(&y, &gains(&y, w, h, Tone { clarity: 100.0, ..Tone::NEUTRAL }), w, h);
        let down = swing(&y, &gains(&y, w, h, Tone { clarity: -100.0, ..Tone::NEUTRAL }), w, h);
        // Measured: +100 roughly doubles midtone detail, −100 flattens it by more than a third.
        assert!(up > 1.5 * before && up < 3.5 * before, "{before} → {up}");
        assert!(down < 0.8 * before && down > 0.1 * before, "{before} → {down}");
    }

    #[test]
    fn texture_raises_fine_detail_more_than_coarse_and_leaves_flat_areas() {
        let (w, h) = (2464, 160);
        let fine = stripes(w, h, -2.5, 0.1, 6.0);
        let coarse = stripes(w, h, -2.5, 0.1, 600.0);
        let gain = |y: &[f32]| swing(y, &gains(y, w, h, Tone { texture: 100.0, ..Tone::NEUTRAL }), w, h) / swing(y, &vec![1.0; w * h], w, h);
        let (f, c) = (gain(&fine), gain(&coarse));
        assert!(f > 1.15, "fine detail ×{f}");
        assert!(c < 1.05 && f > c + 0.1, "coarse ×{c}, fine ×{f}");
        let flat = vec![0.2f32; 400 * 300];
        assert!(gains(&flat, 400, 300, Tone { texture: 100.0, ..Tone::NEUTRAL }).iter().all(|g| (g - 1.0).abs() < 1e-4));
    }

    #[test]
    fn dehaze_lowers_the_haze_and_minus_adds_a_veil() {
        // A pale, hazy sky (top) over a darker, colourful ground.
        let (w, h) = (400, 300);
        let px: Vec<[f32; 3]> = (0..w * h).map(|i| if i / w < h / 2 { [0.55, 0.6, 0.7] } else { [0.12, 0.08, 0.03] }).collect();
        let up = Haze::new(w, h, |i| px[i], 100.0).unwrap();
        let (sky, ground) = (up.apply(200, 40, px[40 * w + 200]), up.apply(200, 260, px[260 * w + 200]));
        // The airlight itself stays; what is darker than it moves away from it.
        assert!(sky.iter().zip(&px[40 * w + 200]).all(|(a, b)| (a - b).abs() < 0.02), "{sky:?}");
        assert!(ground[2] < 0.03 && ground[0] <= 0.12, "{ground:?}");
        let down = Haze::new(w, h, |i| px[i], -100.0).unwrap();
        let veiled = down.apply(200, 260, px[260 * w + 200]);
        assert!((0..3).all(|c| veiled[c] > px[260 * w + 200][c]), "{veiled:?}");
        assert!(Haze::new(w, h, |i| px[i], 0.0).is_none());
    }

    #[test]
    fn retone_changes_nothing_at_zero_and_matches_the_develop_stage() {
        let (w, h) = (300, 120);
        let y = bands(w, h, 1.0);
        let curve = [[0.01, 0.02], [0.1, 0.25], [1.0, 1.0]];
        let enc = |v: f32| crate::develop::tone(&curve, v).powf(1.0 / 1.8);
        let px: Vec<[f32; 4]> = y.iter().map(|v| [enc(*v), enc(*v), enc(*v), 1.0]).collect();
        let mut same = px.clone();
        retone(&mut same, w, h, &curve, Tone::default());
        assert_eq!(same, px);
        let mut out = px.clone();
        retone(&mut out, w, h, &curve, SHADOWS);
        let g = gains(&y, w, h, SHADOWS);
        for i in [60 * w + 50, 60 * w + 150, 60 * w + 250] {
            assert!((out[i][0] - enc(y[i] * g[i])).abs() < 0.003, "{i}: {} vs {}", out[i][0], enc(y[i] * g[i]));
        }
    }
}
