//! Filter › Camera Raw Filter: a float develop pipeline over RGB pixels.
//!
//! Stages, in order (each skipped when its controls are neutral, so defaults are an identity):
//!
//! 1. **White balance and exposure** in linear light (the document's values decoded with the sRGB
//!    curve as a perceptual working encoding; encoding is undone exactly at the end).
//! 2. **Dehaze**: K. He, J. Sun, X. Tang, *Single Image Haze Removal Using Dark Channel Prior*,
//!    CVPR 2009 (dark channel, atmospheric light from its brightest 0.1 %, transmission refined
//!    with the guided filter); negative amounts add haze toward the atmospheric light.
//! 3. **Tone** on luminance with colour ratios kept: Highlights / Shadows as exposure changes
//!    weighted by an edge-preserving base layer (guided filter: K. He, J. Sun, X. Tang, *Guided
//!    Image Filtering*, ECCV 2010), so local contrast survives; Contrast is an S-curve about
//!    middle grey; Clarity (large radius, midtone weighted) and Texture (small radius) add
//!    band-pass local contrast. Whites / Blacks follow as Photoshop's Camera Raw applies them:
//!    image-adaptive global curves measured on Photoshop 25.4 (see `whites_blacks`).
//! 4. **Presence**: Vibrance (weighted toward less saturated colours) and Saturation.
//! 5. **Tone Curve**: parametric regions (Highlights, Lights, Darks, Shadows with movable splits)
//!    then point curves (master and per channel), monotone cubic interpolation (F. Fritsch,
//!    R. Carlson, *Monotone Piecewise Cubic Interpolation*, SIAM J. Numer. Anal. 1980).
//! 6. **Color Mixer (HSL)**: eight hue bands (red, orange, yellow, green, aqua, blue, purple,
//!    magenta) with hat-function weights on the hue circle.
//! 7. **Color Grading**: shadows / midtones / highlights / global wheels (hue, saturation,
//!    luminance) with Blending and Balance.
//! 8. **Detail**: sharpening on luminance (unsharp mask with Detail as halo control and Masking as
//!    an edge mask) and noise reduction (luminance by guided filter, colour by chroma blur).
//! 9. **Effects**: post-crop vignette (amount, midpoint, roundness, feather, highlights,
//!    style) and grain (amount, size, roughness; position-hashed, deterministic).

#![allow(clippy::needless_range_loop)] // index loops mirror the maths (pixels × channels)

use serde::{Deserialize, Serialize};

/// Most points a Camera Raw point curve accepts from commands and the control channel.
pub const MAX_CURVE_POINTS: usize = 16;

/// Checks a point curve from untrusted input: empty (linear) or 2..=16 finite points in 0..=255
/// with inputs increasing by at least one level. Deserialization stays lenient on purpose:
/// earlier editors saved curves this rejects, and [`curve_lut`] already sorts and de-duplicates
/// them, so stored Smart Filters keep rendering.
pub fn validate_curve(points: &[[f32; 2]]) -> Result<(), String> {
    if points.len() == 1 {
        return Err("a nonempty curve needs at least two points".into());
    }
    if points.len() > MAX_CURVE_POINTS {
        return Err(format!("at most {MAX_CURVE_POINTS} curve points"));
    }
    if !points.iter().flatten().all(|v| v.is_finite() && (0.0..=255.0).contains(v)) {
        return Err("curve coordinates must be finite and in 0..=255".into());
    }
    if points.windows(2).any(|w| w[1][0] - w[0][0] < 1.0) {
        return Err("curve inputs must increase by at least one level".into());
    }
    Ok(())
}

use crate::photo_util::{hash01, linear_to_srgb, par_rows, srgb_to_linear};

mod local_tone;
mod whites_blacks;

/// How the pixels encode light: the sRGB curve (gamma 0) or a pure power law (ProPhoto's 1.8,
/// Adobe RGB's 2.2). The tone controls work on light, so the document's own curve decodes it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Encoding(f32);

impl Encoding {
    pub(crate) fn of(gamma: f32) -> Self {
        Encoding(if gamma.is_finite() && gamma > 0.0 { gamma.clamp(0.5, 4.0) } else { 0.0 })
    }
    pub(crate) fn decode(self, v: f32) -> f32 {
        if self.0 > 0.0 { v.max(0.0).powf(self.0) } else { srgb_to_linear(v) }
    }
    pub(crate) fn encode(self, v: f32) -> f32 {
        if self.0 > 0.0 { v.max(0.0).powf(1.0 / self.0) } else { linear_to_srgb(v) }
    }
}

fn srgb_encode(v: f32) -> f32 {
    linear_to_srgb(v)
}

fn srgb_decode(v: f32) -> f32 {
    srgb_to_linear(v)
}

fn luma_linear(c: [f32; 3]) -> f32 {
    luma(c)
}

/// One colour grading wheel.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Wheel {
    /// Degrees 0..360.
    pub hue: f32,
    /// 0..=100.
    pub sat: f32,
    /// −100..=100.
    pub lum: f32,
}

impl Wheel {
    fn neutral(&self) -> bool {
        self.sat == 0.0 && self.lum == 0.0
    }
}

/// Camera Raw settings in Adobe Camera Raw's slider units.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct CameraRaw {
    // Basic
    pub temperature: f32,
    pub tint: f32,
    pub exposure: f32,
    pub contrast: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub whites: f32,
    pub blacks: f32,
    pub texture: f32,
    pub clarity: f32,
    pub dehaze: f32,
    pub vibrance: f32,
    pub saturation: f32,
    // Tone curve
    pub curve_highlights: f32,
    pub curve_lights: f32,
    pub curve_darks: f32,
    pub curve_shadows: f32,
    /// Region splits (percent).
    pub curve_splits: [f32; 3],
    /// Point curves as `[input, output]` in 0..=255 (empty = linear).
    pub point_curve: Vec<[f32; 2]>,
    pub point_curve_red: Vec<[f32; 2]>,
    pub point_curve_green: Vec<[f32; 2]>,
    pub point_curve_blue: Vec<[f32; 2]>,
    // Color mixer: red, orange, yellow, green, aqua, blue, purple, magenta.
    pub hsl_hue: [f32; 8],
    pub hsl_sat: [f32; 8],
    pub hsl_lum: [f32; 8],
    // Color grading
    pub grade_shadows: Wheel,
    pub grade_midtones: Wheel,
    pub grade_highlights: Wheel,
    pub grade_global: Wheel,
    pub grade_blending: f32,
    pub grade_balance: f32,
    // Detail
    pub sharpen_amount: f32,
    pub sharpen_radius: f32,
    pub sharpen_detail: f32,
    pub sharpen_masking: f32,
    pub noise_luminance: f32,
    pub noise_luminance_detail: f32,
    pub noise_color: f32,
    pub noise_color_detail: f32,
    // Effects
    pub grain_amount: f32,
    pub grain_size: f32,
    pub grain_roughness: f32,
    pub vignette_amount: f32,
    pub vignette_midpoint: f32,
    pub vignette_roundness: f32,
    pub vignette_feather: f32,
    pub vignette_highlights: f32,
    /// `highlightPriority | colorPriority | paintOverlay`.
    pub vignette_style: String,
    pub seed: u32,
    /// Pixel-size multiplier for previews on downsampled proxies (radii in px scale by it).
    pub pixel_scale: f32,
    /// The document's transfer curve: 0 for sRGB's, else the gamma of a power-law space (1.8
    /// for ProPhoto). Set from the document's profile; Whites and Blacks act on light.
    pub encoding_gamma: f32,
}

impl Default for CameraRaw {
    fn default() -> Self {
        CameraRaw {
            temperature: 0.0,
            tint: 0.0,
            exposure: 0.0,
            contrast: 0.0,
            highlights: 0.0,
            shadows: 0.0,
            whites: 0.0,
            blacks: 0.0,
            texture: 0.0,
            clarity: 0.0,
            dehaze: 0.0,
            vibrance: 0.0,
            saturation: 0.0,
            curve_highlights: 0.0,
            curve_lights: 0.0,
            curve_darks: 0.0,
            curve_shadows: 0.0,
            curve_splits: [25.0, 50.0, 75.0],
            point_curve: Vec::new(),
            point_curve_red: Vec::new(),
            point_curve_green: Vec::new(),
            point_curve_blue: Vec::new(),
            hsl_hue: [0.0; 8],
            hsl_sat: [0.0; 8],
            hsl_lum: [0.0; 8],
            grade_shadows: Wheel::default(),
            grade_midtones: Wheel::default(),
            grade_highlights: Wheel::default(),
            grade_global: Wheel::default(),
            grade_blending: 50.0,
            grade_balance: 0.0,
            sharpen_amount: 0.0,
            sharpen_radius: 1.0,
            sharpen_detail: 25.0,
            sharpen_masking: 0.0,
            noise_luminance: 0.0,
            noise_luminance_detail: 50.0,
            noise_color: 0.0,
            noise_color_detail: 50.0,
            grain_amount: 0.0,
            grain_size: 25.0,
            grain_roughness: 50.0,
            vignette_amount: 0.0,
            vignette_midpoint: 50.0,
            vignette_roundness: 0.0,
            vignette_feather: 50.0,
            vignette_highlights: 0.0,
            vignette_style: "highlightPriority".into(),
            seed: 0,
            pixel_scale: 1.0,
            encoding_gamma: 0.0,
        }
    }
}

/// Hue band centres of the Color Mixer (degrees).
pub const HSL_BANDS: [f32; 8] = [0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 270.0, 300.0];

fn luma(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Box mean of a single-channel buffer (radius `r`, edge clamped): rows in parallel, then
/// columns in parallel bands with running column sums (cache-friendly, no transposes).
fn box_mean(v: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    if r == 0 || w == 0 || h == 0 {
        return v.to_vec();
    }
    let n = (2 * r + 1) as f64;
    let mut tmp = vec![0.0f32; w * h];
    par_rows(&mut tmp, w, 1, |y, row| {
        let src = &v[y * w..(y + 1) * w];
        let at = |k: i64| src[k.clamp(0, w as i64 - 1) as usize] as f64;
        let mut acc: f64 = (-(r as i64)..=r as i64).map(at).sum();
        for (x, o) in row.iter_mut().enumerate() {
            *o = (acc / n) as f32;
            acc += at(x as i64 + r as i64 + 1) - at(x as i64 - r as i64);
        }
    });
    const BAND: usize = 256;
    let bands = crate::photo_util::par_map(w.div_ceil(BAND), |b| {
        let (x0, x1) = (b * BAND, ((b + 1) * BAND).min(w));
        let bw = x1 - x0;
        let row = |y: i64| &tmp[y.clamp(0, h as i64 - 1) as usize * w + x0..y.clamp(0, h as i64 - 1) as usize * w + x1];
        let mut acc = vec![0.0f64; bw];
        for y in -(r as i64)..=r as i64 {
            for (a, v) in acc.iter_mut().zip(row(y)) {
                *a += *v as f64;
            }
        }
        let mut out = vec![0.0f32; bw * h];
        for y in 0..h {
            for (o, a) in out[y * bw..(y + 1) * bw].iter_mut().zip(&acc) {
                *o = (*a / n) as f32;
            }
            let (add, sub) = (row(y as i64 + r as i64 + 1), row(y as i64 - r as i64));
            for ((a, p), m) in acc.iter_mut().zip(add).zip(sub) {
                *a += (*p - *m) as f64;
            }
        }
        out
    });
    let mut out = vec![0.0f32; w * h];
    for (b, band) in bands.iter().enumerate() {
        let (x0, x1) = (b * BAND, ((b + 1) * BAND).min(w));
        let bw = x1 - x0;
        for y in 0..h {
            out[y * w + x0..y * w + x1].copy_from_slice(&band[y * bw..(y + 1) * bw]);
        }
    }
    out
}

/// Approximate Gaussian (three box passes).
fn gauss(v: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma < 0.3 {
        return v.to_vec();
    }
    let mut out = v.to_vec();
    for r in crate::fxutil::gauss_box_radii(sigma) {
        out = box_mean(&out, w, h, r);
    }
    out
}

/// Fast guided filter (K. He, J. Sun, *Fast Guided Filter*, arXiv 2015): the linear
/// coefficients are computed on a `s×` subsampled copy and upsampled bilinearly, which is
/// visually equivalent for the large radii used by tone controls at a fraction of the cost.
fn guided_fast(i: &[f32], p: &[f32], w: usize, h: usize, r: usize, eps: f32) -> Vec<f32> {
    let s = (r / 8).clamp(1, 8);
    if s == 1 {
        return guided(i, p, w, h, r, eps);
    }
    let (sw, sh) = (w.div_ceil(s), h.div_ceil(s));
    let down = |v: &[f32]| -> Vec<f32> {
        let mut o = vec![0.0f32; sw * sh];
        let mut c = vec![0.0f32; sw * sh];
        for y in 0..h {
            for x in 0..w {
                let k = (y / s) * sw + x / s;
                o[k] += v[y * w + x];
                c[k] += 1.0;
            }
        }
        o.iter().zip(&c).map(|(a, b)| a / b.max(1.0)).collect()
    };
    let (si, sp) = (down(i), down(p));
    let rs = (r / s).max(1);
    let mi = box_mean(&si, sw, sh, rs);
    let mp = box_mean(&sp, sw, sh, rs);
    let ip: Vec<f32> = si.iter().zip(&sp).map(|(a, b)| a * b).collect();
    let ii: Vec<f32> = si.iter().map(|a| a * a).collect();
    let mip = box_mean(&ip, sw, sh, rs);
    let mii = box_mean(&ii, sw, sh, rs);
    let a: Vec<f32> = (0..sw * sh).map(|k| (mip[k] - mi[k] * mp[k]) / (mii[k] - mi[k] * mi[k] + eps)).collect();
    let b: Vec<f32> = (0..sw * sh).map(|k| mp[k] - a[k] * mi[k]).collect();
    let (ma, mb) = (box_mean(&a, sw, sh, rs), box_mean(&b, sw, sh, rs));
    let mut out = vec![0.0f32; w * h];
    let sf = s as f32;
    par_rows(&mut out, w, 1, |y, row| {
        let fy = (y as f32 + 0.5) / sf - 0.5;
        for (x, o) in row.iter_mut().enumerate() {
            let fx = (x as f32 + 0.5) / sf - 0.5;
            let av = crate::photo_util::bilinear(&ma, sw, sh, 1, 0, fx, fy);
            let bv = crate::photo_util::bilinear(&mb, sw, sh, 1, 0, fx, fy);
            *o = av * i[y * w + x] + bv;
        }
    });
    out
}

/// Guided filter of `p` guided by `i` (He et al. 2010).
fn guided(i: &[f32], p: &[f32], w: usize, h: usize, r: usize, eps: f32) -> Vec<f32> {
    let mi = box_mean(i, w, h, r);
    let mp = box_mean(p, w, h, r);
    let ip: Vec<f32> = i.iter().zip(p).map(|(a, b)| a * b).collect();
    let ii: Vec<f32> = i.iter().map(|a| a * a).collect();
    let mip = box_mean(&ip, w, h, r);
    let mii = box_mean(&ii, w, h, r);
    let a: Vec<f32> = (0..w * h).map(|k| (mip[k] - mi[k] * mp[k]) / (mii[k] - mi[k] * mi[k] + eps)).collect();
    let b: Vec<f32> = (0..w * h).map(|k| mp[k] - a[k] * mi[k]).collect();
    let ma = box_mean(&a, w, h, r);
    let mb = box_mean(&b, w, h, r);
    (0..w * h).map(|k| ma[k] * i[k] + mb[k]).collect()
}

/// Monotone cubic (Fritsch–Carlson) through `pts` (x ascending, 0..=1), evaluated into a LUT.
pub fn curve_lut(pts: &[[f32; 2]], n: usize) -> Vec<f32> {
    let mut p: Vec<[f32; 2]> = pts.iter().map(|q| [q[0] / 255.0, q[1] / 255.0]).collect();
    p.sort_by(|a, b| a[0].total_cmp(&b[0]));
    p.dedup_by(|a, b| (a[0] - b[0]).abs() < 1e-6);
    if p.len() < 2 {
        return (0..n).map(|k| k as f32 / (n - 1) as f32).collect();
    }
    let m = p.len();
    let d: Vec<f32> = (0..m - 1).map(|k| (p[k + 1][1] - p[k][1]) / (p[k + 1][0] - p[k][0]).max(1e-6)).collect();
    let mut t = vec![0.0f32; m];
    t[0] = d[0];
    t[m - 1] = d[m - 2];
    for k in 1..m - 1 {
        t[k] = if d[k - 1] * d[k] <= 0.0 { 0.0 } else { (d[k - 1] + d[k]) / 2.0 };
    }
    for k in 0..m - 1 {
        if d[k] == 0.0 {
            t[k] = 0.0;
            t[k + 1] = 0.0;
            continue;
        }
        let (a, b) = (t[k] / d[k], t[k + 1] / d[k]);
        let s = a * a + b * b;
        if s > 9.0 {
            let tau = 3.0 / s.sqrt();
            t[k] = tau * a * d[k];
            t[k + 1] = tau * b * d[k];
        }
    }
    (0..n)
        .map(|i| {
            let x = i as f32 / (n - 1) as f32;
            if x <= p[0][0] {
                return p[0][1];
            }
            if x >= p[m - 1][0] {
                return p[m - 1][1];
            }
            let k = p.windows(2).position(|wv| x <= wv[1][0]).unwrap_or(m - 2);
            let hh = p[k + 1][0] - p[k][0];
            let s = (x - p[k][0]) / hh;
            let (h00, h10, h01, h11) = (2.0 * s * s * s - 3.0 * s * s + 1.0, s * s * s - 2.0 * s * s + s, -2.0 * s * s * s + 3.0 * s * s, s * s * s - s * s);
            (h00 * p[k][1] + h10 * hh * t[k] + h01 * p[k + 1][1] + h11 * hh * t[k + 1]).clamp(0.0, 1.0)
        })
        .collect()
}

fn lut_eval(lut: &[f32], v: f32) -> f32 {
    if !(0.0..=1.0).contains(&v) {
        // Extend linearly outside (float documents).
        return if v < 0.0 { v + lut[0] } else { lut[lut.len() - 1] + (v - 1.0) };
    }
    let x = v * (lut.len() - 1) as f32;
    let i = (x as usize).min(lut.len() - 2);
    let t = x - i as f32;
    lut[i] + (lut[i + 1] - lut[i]) * t
}

/// Parametric tone curve as a LUT (regions split at `splits`, each slider ±100 moves its region
/// by up to a quarter of the range, smoothly).
fn parametric_lut(p: &CameraRaw, n: usize) -> Vec<f32> {
    let s = p.curve_splits.map(|v| v.clamp(5.0, 95.0) / 100.0);
    let centres = [s[0] / 2.0, (s[0] + s[1]) / 2.0, (s[1] + s[2]) / 2.0, (s[2] + 1.0) / 2.0];
    let amounts = [p.curve_shadows, p.curve_darks, p.curve_lights, p.curve_highlights];
    let widths = [s[0].max(0.1), (s[1] - s[0]).max(0.1), (s[2] - s[1]).max(0.1), (1.0 - s[2]).max(0.1)];
    let mut lut: Vec<f32> = (0..n)
        .map(|i| {
            let x = i as f32 / (n - 1) as f32;
            let mut y = x;
            for k in 0..4 {
                let d = (x - centres[k]) / (widths[k] * 1.2);
                let bump = (-d * d * 2.0).exp() * x * (1.0 - x) * 4.0;
                y += amounts[k] / 100.0 * 0.25 * bump;
            }
            y.clamp(0.0, 1.0)
        })
        .collect();
    // Keep it monotone.
    for i in 1..n {
        lut[i] = lut[i].max(lut[i - 1]);
    }
    lut
}

fn rgb_to_hsv(c: [f32; 3]) -> [f32; 3] {
    let (mx, mn) = (c[0].max(c[1]).max(c[2]), c[0].min(c[1]).min(c[2]));
    let d = mx - mn;
    let h = if d <= 1e-9 {
        0.0
    } else if mx == c[0] {
        60.0 * (((c[1] - c[2]) / d).rem_euclid(6.0))
    } else if mx == c[1] {
        60.0 * ((c[2] - c[0]) / d + 2.0)
    } else {
        60.0 * ((c[0] - c[1]) / d + 4.0)
    };
    [h, if mx > 1e-9 { d / mx } else { 0.0 }, mx]
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h = h.rem_euclid(360.0) / 60.0;
    let c = v * s;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r + m, g + m, b + m]
}

/// Weights of the eight HSL bands at hue `h` (hat functions between neighbouring centres).
pub fn band_weights(h: f32) -> [f32; 8] {
    let mut w = [0.0f32; 8];
    for k in 0..8 {
        let (c, prev, next) = (HSL_BANDS[k], HSL_BANDS[(k + 7) % 8], HSL_BANDS[(k + 1) % 8]);
        let diff = ((h - c + 540.0) % 360.0) - 180.0;
        let span = if diff >= 0.0 { (next - c).rem_euclid(360.0) } else { (c - prev).rem_euclid(360.0) };
        w[k] = (1.0 - diff.abs() / span.max(1e-3)).max(0.0);
    }
    w
}

impl CameraRaw {
    /// Rejects malformed point curves in new settings (see [`validate_curve`]).
    pub fn validate(&self) -> Result<(), String> {
        for (name, curve) in [
            ("pointCurve", &self.point_curve),
            ("pointCurveRed", &self.point_curve_red),
            ("pointCurveGreen", &self.point_curve_green),
            ("pointCurveBlue", &self.point_curve_blue),
        ] {
            validate_curve(curve).map_err(|e| format!("{name}: {e}"))?;
        }
        Ok(())
    }
    /// [`validate`](Self::validate) for an edit of `stored` settings: a curve left as stored is
    /// accepted, so editing another control of an older Smart Filter still works.
    pub fn validate_changes(&self, stored: &CameraRaw) -> Result<(), String> {
        for (name, curve, kept) in [
            ("pointCurve", &self.point_curve, &stored.point_curve),
            ("pointCurveRed", &self.point_curve_red, &stored.point_curve_red),
            ("pointCurveGreen", &self.point_curve_green, &stored.point_curve_green),
            ("pointCurveBlue", &self.point_curve_blue, &stored.point_curve_blue),
        ] {
            if curve != kept {
                validate_curve(curve).map_err(|e| format!("{name}: {e}"))?;
            }
        }
        Ok(())
    }
    fn wb_neutral(&self) -> bool {
        self.temperature == 0.0 && self.tint == 0.0 && self.exposure == 0.0
    }
    fn tone_neutral(&self) -> bool {
        self.local_tone_neutral() && self.whites == 0.0 && self.blacks == 0.0
    }
    /// Contrast, Highlights, Shadows, Texture and Clarity (the luminance stage).
    fn local_tone_neutral(&self) -> bool {
        [self.contrast, self.highlights, self.shadows, self.texture, self.clarity].iter().all(|v| *v == 0.0)
    }
    fn hsl_neutral(&self) -> bool {
        self.hsl_hue.iter().chain(&self.hsl_sat).chain(&self.hsl_lum).all(|v| *v == 0.0)
    }
    fn grade_neutral(&self) -> bool {
        self.grade_shadows.neutral() && self.grade_midtones.neutral() && self.grade_highlights.neutral() && self.grade_global.neutral()
    }
    fn curves_neutral(&self) -> bool {
        [self.curve_highlights, self.curve_lights, self.curve_darks, self.curve_shadows].iter().all(|v| *v == 0.0)
            && [&self.point_curve, &self.point_curve_red, &self.point_curve_green, &self.point_curve_blue].iter().all(|c| is_linear(c))
    }
    /// True when every control is neutral (the filter is an identity).
    pub fn is_identity(&self) -> bool {
        self.wb_neutral()
            && self.tone_neutral()
            && self.dehaze == 0.0
            && self.vibrance == 0.0
            && self.saturation == 0.0
            && self.curves_neutral()
            && self.hsl_neutral()
            && self.grade_neutral()
            && self.sharpen_amount == 0.0
            && self.noise_luminance == 0.0
            && self.noise_color == 0.0
            && self.grain_amount == 0.0
            && self.vignette_amount == 0.0
    }
}

fn is_linear(c: &[[f32; 2]]) -> bool {
    c.len() < 2 || c.iter().all(|p| (p[0] - p[1]).abs() < 1e-3)
}

/// Camera Raw's white-balance channel gains for Temperature / Tint (−100…100), relative to the
/// as-shot balance: warmer raises red and lowers blue (±35 % at the ends); a negative tint
/// raises green, a positive one lowers it toward magenta (±25 %). An observed approximation of
/// Adobe Camera Raw's sliders, not a colour-temperature model; shared by the filter (in linear
/// RGB) and the raw open (on the camera's white-balance multipliers) so both agree.
pub fn white_balance_gains(temperature: f32, tint: f32) -> [f32; 3] {
    let t = temperature / 100.0;
    let tn = tint / 100.0;
    [1.0 + 0.35 * t, 1.0 - 0.25 * tn, 1.0 - 0.35 * t]
}

/// Runs the Camera Raw pipeline on straight RGBA pixels (`w × h`, display-encoded RGB).
/// `float` keeps values above 1 (32-bit documents).
pub fn develop(px: &mut [[f32; 4]], w: usize, h: usize, p: &CameraRaw, float: bool) {
    if px.is_empty() || p.is_identity() {
        return;
    }
    let ps = p.pixel_scale.clamp(0.01, 1.0);
    let long = w.max(h) as f32;
    // 1. White balance + exposure in linear light.
    if !p.wb_neutral() {
        let mut g = white_balance_gains(p.temperature, p.tint);
        let norm = luma(g);
        g = g.map(|v| v / norm * 2f32.powf(p.exposure));
        par_rows(px, w, 1, |_, row| {
            for q in row.iter_mut() {
                for c in 0..3 {
                    q[c] = linear_to_srgb((srgb_to_linear(q[c]) * g[c]).max(0.0));
                }
            }
        });
    }
    // 2. Dehaze.
    if p.dehaze != 0.0 {
        dehaze(px, w, h, p.dehaze / 100.0);
    }
    // 3. Tone. Highlights and Shadows: a local Laplacian filter (see `local_tone`).
    if p.highlights != 0.0 || p.shadows != 0.0 {
        local_tone::apply(px, w, h, p.shadows, p.highlights, Encoding::of(p.encoding_gamma));
    }
    if !p.local_tone_neutral() && [p.contrast, p.texture, p.clarity].iter().any(|v| *v != 0.0) {
        let l: Vec<f32> = px.iter().map(|q| luma([q[0], q[1], q[2]]).max(0.0)).collect();
        let lc: Vec<f32> = l.iter().map(|v| v.min(1.0)).collect();
        let clar = (p.clarity != 0.0).then(|| guided_fast(&lc, &lc, w, h, ((long * 0.02) as usize).max(3), 0.005));
        let tex = (p.texture != 0.0).then(|| gauss(&lc, w, h, (long * 0.002).max(1.5 * ps)));
        let k_con = p.contrast / 100.0 * 0.6;
        let tau = std::f32::consts::TAU;
        par_rows(px, w, 1, |y, row| {
            for (x, q) in row.iter_mut().enumerate() {
                let i = y * w + x;
                let l0 = l[i];
                let mut lv = lc[i];
                if let Some(c) = &clar {
                    let mw = 1.0 - (2.0 * lv.clamp(0.0, 1.0) - 1.0).powi(2);
                    lv += p.clarity / 100.0 * 0.8 * (lc[i] - c[i]) * mw;
                }
                if let Some(t) = &tex {
                    lv += p.texture / 100.0 * 0.9 * (lc[i] - t[i]).clamp(-0.1, 0.1);
                }
                if k_con != 0.0 {
                    let xx = lv.clamp(0.0, 1.0);
                    lv += -k_con * (tau * xx).sin() / tau;
                }
                let lv = lv.max(0.0) + (l0 - lc[i]);
                let ratio = if l0 > 1e-5 { lv / l0 } else { 1.0 };
                for c in 0..3 {
                    q[c] = if l0 > 1e-5 { q[c] * ratio } else { q[c] + (lv - l0) };
                }
            }
        });
    }
    // 3b. Whites and Blacks: global curves adapted to the image's black and white points.
    if p.whites != 0.0 || p.blacks != 0.0 {
        let enc = Encoding::of(p.encoding_gamma);
        let (bp, wp) = luminance_points(px, w, h, enc);
        let (xw, xb) = whites_blacks::amounts(p.whites.clamp(-100.0, 100.0), p.blacks.clamp(-100.0, 100.0), bp, wp);
        whites_blacks::apply(px, w, xw, xb, enc);
    }
    // 4–7: per-pixel colour stages.
    let need_color = p.vibrance != 0.0 || p.saturation != 0.0 || !p.curves_neutral() || !p.hsl_neutral() || !p.grade_neutral();
    if need_color {
        let par = (!(p.curve_highlights == 0.0 && p.curve_lights == 0.0 && p.curve_darks == 0.0 && p.curve_shadows == 0.0)).then(|| parametric_lut(p, 1024));
        let pts = |c: &Vec<[f32; 2]>| (!is_linear(c)).then(|| curve_lut(c, 1024));
        let (master, cr, cg, cb) = (pts(&p.point_curve), pts(&p.point_curve_red), pts(&p.point_curve_green), pts(&p.point_curve_blue));
        let hsl = !p.hsl_neutral();
        let grade = !p.grade_neutral();
        let wheel_tint = |wh: &Wheel| -> [f32; 3] {
            let c = hsv_to_rgb(wh.hue, 1.0, 1.0);
            let l = luma(c);
            [c[0] - l, c[1] - l, c[2] - l].map(|v| v * wh.sat / 100.0 * 0.3)
        };
        let tints = [wheel_tint(&p.grade_shadows), wheel_tint(&p.grade_midtones), wheel_tint(&p.grade_highlights), wheel_tint(&p.grade_global)];
        let blend = 0.15 + p.grade_blending.clamp(0.0, 100.0) / 100.0 * 0.35;
        let bal = p.grade_balance.clamp(-100.0, 100.0) / 100.0 * 0.25;
        par_rows(px, w, 1, |_, row| {
            for q in row.iter_mut() {
                let mut c = [q[0], q[1], q[2]];
                if p.vibrance != 0.0 || p.saturation != 0.0 {
                    let l = luma(c);
                    let (mx, mn) = (c[0].max(c[1]).max(c[2]), c[0].min(c[1]).min(c[2]));
                    let sat = if mx > 1e-6 { (mx - mn) / mx } else { 0.0 };
                    // Vibrance protects skin-ish hues (oranges) a little, like ACR.
                    let hue = rgb_to_hsv(c)[0];
                    let skin = (1.0 - ((hue - 25.0).abs() / 25.0)).clamp(0.0, 1.0) * 0.5;
                    let k = 1.0 + p.saturation / 100.0 + p.vibrance / 100.0 * (1.0 - sat) * (1.0 - skin);
                    c = c.map(|v| l + (v - l) * k.max(0.0));
                }
                if let Some(lut) = &par {
                    c = c.map(|v| lut_eval(lut, v));
                }
                if let Some(lut) = &master {
                    c = c.map(|v| lut_eval(lut, v));
                }
                for (ch, lut) in [&cr, &cg, &cb].iter().enumerate() {
                    if let Some(lut) = lut {
                        c[ch] = lut_eval(lut, c[ch]);
                    }
                }
                if hsl {
                    let [hh, s, v] = rgb_to_hsv(c.map(|x| x.max(0.0)));
                    if s > 1e-4 {
                        let wts = band_weights(hh);
                        let (mut dh, mut ds, mut dl) = (0.0f32, 0.0f32, 0.0f32);
                        for k in 0..8 {
                            dh += wts[k] * p.hsl_hue[k];
                            ds += wts[k] * p.hsl_sat[k];
                            dl += wts[k] * p.hsl_lum[k];
                        }
                        let nh = hh + dh / 100.0 * 30.0;
                        let ns = (s * (1.0 + ds / 100.0)).clamp(0.0, 1.0);
                        let mut out = hsv_to_rgb(nh, ns, v);
                        // Keep the luminance, then apply the luminance shift weighted by saturation.
                        let (l0, l1) = (luma(c), luma(out));
                        if l1 > 1e-6 {
                            out = out.map(|x| x * l0 / l1);
                        }
                        let k = 2f32.powf(dl / 100.0 * 1.0 * s);
                        c = out.map(|x| x * k);
                    }
                }
                if grade {
                    let l = luma(c).clamp(0.0, 1.0);
                    let sh = 1.0 - smoothstep(0.0, 0.5 + bal + blend, l);
                    let hi = smoothstep(0.5 + bal - blend, 1.0, l);
                    let mid = (1.0 - sh - hi).max(0.0);
                    let masks = [sh, mid, hi, 1.0];
                    let lums = [p.grade_shadows.lum, p.grade_midtones.lum, p.grade_highlights.lum, p.grade_global.lum];
                    for k in 0..4 {
                        for ch in 0..3 {
                            c[ch] += masks[k] * tints[k][ch];
                        }
                        if lums[k] != 0.0 {
                            let f = 2f32.powf(masks[k] * lums[k] / 100.0 * 0.6);
                            c = c.map(|x| x * f);
                        }
                    }
                }
                q[0] = c[0];
                q[1] = c[1];
                q[2] = c[2];
            }
        });
    }
    // 8. Detail.
    if p.noise_luminance > 0.0 || p.noise_color > 0.0 || p.sharpen_amount > 0.0 {
        detail(px, w, h, p, ps);
    }
    // 9. Effects.
    if p.vignette_amount != 0.0 {
        vignette(px, w, h, p);
    }
    if p.grain_amount > 0.0 {
        let cell = 1.0 + p.grain_size / 25.0;
        let rough = p.grain_roughness / 100.0;
        let amt = p.grain_amount / 100.0 * 0.12;
        let seed = p.seed as u64;
        let pscale = 1.0 / ps;
        par_rows(px, w, 1, |y, row| {
            for (x, q) in row.iter_mut().enumerate() {
                // Sample in full-resolution coordinates so previews match the result.
                let (fx, fy) = (x as f32 * pscale / cell, y as f32 * pscale / cell);
                let n = value_noise(fx, fy, seed) * (1.0 - rough) + (value_noise(fx * 2.3, fy * 2.3, seed + 7) - 0.0) * rough;
                let l = luma([q[0], q[1], q[2]]).clamp(0.0, 1.0);
                let g = (n - 0.5) * 2.0 * amt * (0.4 + 2.4 * l * (1.0 - l));
                for c in 0..3 {
                    q[c] += g;
                }
            }
        });
    }
    if !float {
        for q in px.iter_mut() {
            for c in 0..3 {
                q[c] = q[c].clamp(0.0, 1.0);
            }
        }
    }
}

/// The image's black and white points as sRGB-encoded grey levels: the 3rd percentile of the
/// luminance and the 99.5th of the brightest channel, on a copy at most 256 pixels across (so a
/// preview proxy and the full image agree).
fn luminance_points(px: &[[f32; 4]], w: usize, h: usize, enc: Encoding) -> (f32, f32) {
    let f = w.max(h).div_ceil(256).max(1);
    let (sw, sh) = (w.div_ceil(f), h.div_ceil(f));
    let mut sum = vec![[0.0f64; 4]; sw * sh];
    for y in 0..h {
        for x in 0..w {
            let q = px[y * w + x];
            let s = &mut sum[(y / f) * sw + x / f];
            for c in 0..3 {
                s[c] += f64::from(enc.decode(q[c]).clamp(0.0, 1.0));
            }
            s[3] += 1.0;
        }
    }
    let cells: Vec<[f32; 3]> = sum.iter().filter(|s| s[3] > 0.0).map(|s| [0, 1, 2].map(|c| (s[c] / s[3]) as f32)).collect();
    if cells.is_empty() {
        return (0.0, 1.0);
    }
    let pct = |mut v: Vec<f32>, q: f32| {
        v.sort_by(f32::total_cmp);
        v[((v.len() - 1) as f32 * q).round() as usize]
    };
    let bp = pct(cells.iter().map(|c| luma(*c)).collect(), 0.03);
    let wp = pct(cells.iter().map(|c| c[0].max(c[1]).max(c[2])).collect(), 0.995);
    (linear_to_srgb(bp), linear_to_srgb(wp))
}

fn value_noise(x: f32, y: f32, seed: u64) -> f32 {
    let (xi, yi) = (x.floor(), y.floor());
    let (tx, ty) = (x - xi, y - yi);
    let (sx, sy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
    let h = |dx: i64, dy: i64| hash01(xi as i64 + dx, yi as i64 + dy, seed);
    let top = h(0, 0) + (h(1, 0) - h(0, 0)) * sx;
    let bot = h(0, 1) + (h(1, 1) - h(0, 1)) * sx;
    top + (bot - top) * sy
}

fn dehaze(px: &mut [[f32; 4]], w: usize, h: usize, amount: f32) {
    // Transmission on a ≤ 768 px proxy, upsampled.
    let k = w.max(h).div_ceil(768).max(1);
    let (sw, sh) = (w.div_ceil(k), h.div_ceil(k));
    let mut small = vec![[0.0f32; 3]; sw * sh];
    let mut cnt = vec![0f32; sw * sh];
    for y in 0..h {
        for x in 0..w {
            let i = (y / k) * sw + x / k;
            for c in 0..3 {
                small[i][c] += px[y * w + x][c].clamp(0.0, 1.0);
            }
            cnt[i] += 1.0;
        }
    }
    for (s, c) in small.iter_mut().zip(&cnt) {
        *s = s.map(|v| v / c.max(1.0));
    }
    let r = (sw.max(sh) / 60).max(2);
    let min_filter = |v: &[f32]| -> Vec<f32> {
        let mut a = vec![0.0f32; sw * sh];
        for y in 0..sh {
            for x in 0..sw {
                let (x0, x1) = (x.saturating_sub(r), (x + r).min(sw - 1));
                a[y * sw + x] = (x0..=x1).map(|xx| v[y * sw + xx]).fold(f32::MAX, f32::min);
            }
        }
        let mut b = vec![0.0f32; sw * sh];
        for y in 0..sh {
            for x in 0..sw {
                let (y0, y1) = (y.saturating_sub(r), (y + r).min(sh - 1));
                b[y * sw + x] = (y0..=y1).map(|yy| a[yy * sw + x]).fold(f32::MAX, f32::min);
            }
        }
        b
    };
    let dark = min_filter(&small.iter().map(|c| c[0].min(c[1]).min(c[2])).collect::<Vec<_>>());
    // Atmospheric light: mean colour of the brightest 0.1 % of the dark channel.
    let mut idx: Vec<usize> = (0..sw * sh).collect();
    idx.sort_by(|&a, &b| dark[b].total_cmp(&dark[a]));
    let top = (sw * sh / 1000).max(1);
    let mut air = [0.0f32; 3];
    for &i in &idx[..top] {
        for c in 0..3 {
            air[c] += small[i][c] / top as f32;
        }
    }
    let air = air.map(|v| v.max(0.2));
    let norm_dark = min_filter(&small.iter().map(|c| (c[0] / air[0]).min(c[1] / air[1]).min(c[2] / air[2])).collect::<Vec<_>>());
    let t_raw: Vec<f32> = norm_dark.iter().map(|d| 1.0 - 0.95 * d).collect();
    let guide: Vec<f32> = small.iter().map(|c| luma(*c)).collect();
    let t = guided(&guide, &t_raw, sw, sh, r * 2, 1e-3);
    let ks = k as f32;
    par_rows(px, w, 1, |y, row| {
        for (x, q) in row.iter_mut().enumerate() {
            let tv = crate::photo_util::bilinear(&t, sw, sh, 1, 0, (x as f32 + 0.5) / ks - 0.5, (y as f32 + 0.5) / ks - 0.5).clamp(0.1, 1.0);
            for c in 0..3 {
                let i = q[c];
                let j = (i - air[c]) / tv + air[c];
                q[c] = if amount >= 0.0 { i + (j - i) * amount } else { i + (air[c] - i) * (-amount) * 0.5 * (1.0 - tv * 0.5) };
            }
        }
    });
}

fn detail(px: &mut [[f32; 4]], w: usize, h: usize, p: &CameraRaw, ps: f32) {
    // Work in Y + chroma (BT.709 luma, B−Y, R−Y).
    let mut y: Vec<f32> = px.iter().map(|q| luma([q[0], q[1], q[2]])).collect();
    let mut cb: Vec<f32> = px.iter().zip(&y).map(|(q, l)| q[2] - l).collect();
    let mut cr: Vec<f32> = px.iter().zip(&y).map(|(q, l)| q[0] - l).collect();
    if p.noise_luminance > 0.0 {
        let n = p.noise_luminance / 100.0;
        let r = ((1.0 + 3.0 * n) * ps).round().max(1.0) as usize;
        let smooth = guided(&y, &y, w, h, r, (n * 0.06).powi(2) + 1e-6);
        let keep = p.noise_luminance_detail / 100.0 * 0.5;
        for i in 0..w * h {
            y[i] = smooth[i] + (y[i] - smooth[i]) * keep;
        }
    }
    if p.noise_color > 0.0 {
        let sigma = (p.noise_color / 100.0 * 6.0 * ps).max(0.3);
        let (bcb, bcr) = (gauss(&cb, w, h, sigma), gauss(&cr, w, h, sigma));
        let keep = p.noise_color_detail / 100.0 * 0.3;
        for i in 0..w * h {
            cb[i] = bcb[i] + (cb[i] - bcb[i]) * keep;
            cr[i] = bcr[i] + (cr[i] - bcr[i]) * keep;
        }
    }
    if p.sharpen_amount > 0.0 {
        let sigma = (p.sharpen_radius.clamp(0.5, 3.0) * ps).max(0.3);
        let blur = gauss(&y, w, h, sigma);
        // Masking: an edge mask from the gradient of a smoothed luminance.
        let mask: Option<Vec<f32>> = (p.sharpen_masking > 0.0).then(|| {
            let s = gauss(&y, w, h, 1.0 * ps.max(0.5));
            let t = p.sharpen_masking / 100.0 * 0.08;
            crate::photo_util::par_map(w * h, |i| {
                let (x, yy) = (i % w, i / w);
                let gx = s[yy * w + (x + 1).min(w - 1)] - s[yy * w + x.saturating_sub(1)];
                let gy = s[(yy + 1).min(h - 1) * w + x] - s[yy.saturating_sub(1) * w + x];
                smoothstep(t * 0.5, t + 1e-4, gx.hypot(gy))
            })
        });
        let amt = p.sharpen_amount / 100.0 * 1.2;
        let det = p.sharpen_detail / 100.0;
        for i in 0..w * h {
            let hp = y[i] - blur[i];
            // Low Detail damps large (halo-producing) differences.
            let limit = 0.03 + det * 0.3;
            let hp = hp.clamp(-limit, limit) * (0.5 + 0.5 * det);
            let m = mask.as_ref().map_or(1.0, |m| m[i]);
            y[i] += amt * hp * m;
        }
    }
    for i in 0..w * h {
        let (l, b, r) = (y[i], cb[i], cr[i]);
        let (rr, bb) = (r + l, b + l);
        let g = (l - 0.2126 * rr - 0.0722 * bb) / 0.7152;
        px[i][0] = rr;
        px[i][1] = g;
        px[i][2] = bb;
    }
}

fn vignette(px: &mut [[f32; 4]], w: usize, h: usize, p: &CameraRaw) {
    let amt = p.vignette_amount / 100.0;
    let mid = p.vignette_midpoint.clamp(0.0, 100.0) / 100.0;
    let feather = p.vignette_feather.clamp(0.0, 100.0) / 100.0;
    let round = p.vignette_roundness.clamp(-100.0, 100.0) / 100.0;
    let hl = p.vignette_highlights.clamp(0.0, 100.0) / 100.0;
    let style = p.vignette_style.as_str();
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    // Roundness: 0 follows the frame's aspect, +1 a circle, −1 a rounded rectangle.
    let (ax, ay) = if round >= 0.0 {
        let r = cx.min(cy) + (cx.max(cy) - cx.min(cy)) * (1.0 - round);
        (if cx >= cy { r } else { cx }, if cy > cx { r } else { cy })
    } else {
        (cx, cy)
    };
    let pw = 2.0 + (-round).max(0.0) * 6.0;
    let inner = 1.0 - mid * 0.8;
    let edge0 = inner - feather * 0.5 * inner;
    let edge1 = inner + feather * 0.5 * (1.4 - inner) + 0.02;
    par_rows(px, w, 1, |y, row| {
        for (x, q) in row.iter_mut().enumerate() {
            let (dx, dy) = (((x as f32 + 0.5 - cx) / ax).abs(), ((y as f32 + 0.5 - cy) / ay).abs());
            let r = (dx.powf(pw) + dy.powf(pw)).powf(1.0 / pw);
            let m = smoothstep(edge0, edge1, r);
            if m <= 0.0 {
                continue;
            }
            let l = luma([q[0], q[1], q[2]]).clamp(0.0, 1.0);
            match style {
                "paintOverlay" => {
                    let target = if amt < 0.0 { 0.0 } else { 1.0 };
                    let k = m * amt.abs();
                    for c in 0..3 {
                        q[c] += (target - q[c]) * k;
                    }
                }
                _ => {
                    let mut k = 2f32.powf(amt * 2.0 * m);
                    if amt < 0.0 && hl > 0.0 {
                        // Highlights: let bright areas punch through a darkening vignette.
                        let protect = smoothstep(0.6, 1.0, l) * hl;
                        k += (1.0 - k) * protect;
                    }
                    if style == "colorPriority" {
                        let nl = l * k;
                        for c in 0..3 {
                            q[c] += nl - l;
                        }
                    } else {
                        for c in 0..3 {
                            q[c] *= k;
                        }
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_curves_are_validated_but_saved_curves_still_load_and_render() {
        for field in ["pointCurve", "pointCurveRed", "pointCurveGreen", "pointCurveBlue"] {
            for points in [
                serde_json::json!([[128, 128]]),
                serde_json::json!([[255, 255], [0, 0]]),
                serde_json::json!([[0, 0], [0, 255]]),
                serde_json::json!([[0, -1], [255, 255]]),
                serde_json::json!([[0, 0], [256, 255]]),
                serde_json::json!([[0, 0], [1e30, 255]]),
                serde_json::json!((0..17).map(|i| [i * 15, i * 15]).collect::<Vec<_>>()),
            ] {
                let p: CameraRaw = serde_json::from_value(serde_json::json!({field: points})).unwrap();
                assert!(p.validate().is_err(), "{field}: {points}");
            }
            for points in
                [serde_json::json!([]), serde_json::json!([[0, 0], [255, 255]]), serde_json::json!((0..16).map(|i| [i * 17, i * 17]).collect::<Vec<_>>())]
            {
                let p: CameraRaw = serde_json::from_value(serde_json::json!({field: points})).unwrap();
                assert!(p.validate().is_ok(), "{field}: {points}");
            }
        }
        assert_eq!(serde_json::from_str::<CameraRaw>("{}").unwrap(), CameraRaw::default());
        // An older editor could stack two points on one input and exceed 16 points; such a
        // saved Smart Filter must still develop instead of silently disappearing.
        let mut legacy: Vec<[f32; 2]> = (0..20).map(|i| [i as f32 * 12.0, i as f32 * 12.0]).collect();
        legacy.insert(5, [60.0, 200.0]);
        let p: CameraRaw = serde_json::from_value(serde_json::json!({"pointCurve": legacy})).unwrap();
        assert!(p.validate().is_err());
        let edited = CameraRaw { exposure: 1.0, ..p.clone() };
        assert!(edited.validate_changes(&p).is_ok(), "an unchanged stored curve stays editable");
        let broken = CameraRaw { point_curve: vec![[10.0, 10.0]], ..p.clone() };
        assert!(broken.validate_changes(&p).is_err(), "a changed curve is validated");
        let mut px = img(4, 4);
        develop(&mut px, 4, 4, &p, false);
        assert!(px.iter().flatten().all(|v| v.is_finite()));
    }

    fn img(w: usize, h: usize) -> Vec<[f32; 4]> {
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 / w as f32, (i / w) as f32 / h as f32);
                [0.2 + 0.6 * x, 0.3 + 0.4 * y, 0.5 - 0.3 * x * y, 1.0]
            })
            .collect()
    }

    fn mean(px: &[[f32; 4]], c: usize) -> f32 {
        px.iter().map(|q| q[c]).sum::<f32>() / px.len() as f32
    }

    #[test]
    fn defaults_are_identity() {
        let mut a = img(40, 30);
        let b = a.clone();
        develop(&mut a, 40, 30, &CameraRaw::default(), false);
        assert_eq!(a, b);
    }

    #[test]
    fn basic_panel_moves_the_right_way() {
        let (w, h) = (64, 48);
        let base = img(w, h);
        let run = |p: CameraRaw| {
            let mut a = base.clone();
            develop(&mut a, w, h, &p, false);
            a
        };
        let warm = run(CameraRaw { temperature: 50.0, ..Default::default() });
        assert!(mean(&warm, 0) > mean(&base, 0) && mean(&warm, 2) < mean(&base, 2));
        let bright = run(CameraRaw { exposure: 1.0, ..Default::default() });
        assert!(mean(&bright, 1) > mean(&base, 1) + 0.1);
        let shadows = run(CameraRaw { shadows: 100.0, ..Default::default() });
        let dark_px = |a: &[[f32; 4]]| a[0][1];
        assert!(dark_px(&shadows) > dark_px(&base));
        let hl = run(CameraRaw { highlights: -100.0, ..Default::default() });
        assert!(hl[w * h - 1][0] < base[w * h - 1][0]);
        let con = run(CameraRaw { contrast: 100.0, ..Default::default() });
        // More contrast: darker darks, brighter brights.
        assert!(con[0][1] < base[0][1] && con[w * h - 1][0] > base[w * h - 1][0]);
        let desat = run(CameraRaw { saturation: -100.0, ..Default::default() });
        assert!(desat.iter().all(|q| (q[0] - q[1]).abs() < 1e-4 && (q[1] - q[2]).abs() < 1e-4));
        let vib = run(CameraRaw { vibrance: 100.0, ..Default::default() });
        let spread = |q: [f32; 4]| q[0].max(q[1]).max(q[2]) - q[0].min(q[1]).min(q[2]);
        assert!(spread(vib[w * h / 2]) > spread(base[w * h / 2]));
        for p in [
            CameraRaw { clarity: 80.0, ..Default::default() },
            CameraRaw { texture: 80.0, ..Default::default() },
            CameraRaw { dehaze: 60.0, ..Default::default() },
            CameraRaw { dehaze: -60.0, ..Default::default() },
            CameraRaw { whites: 50.0, blacks: -50.0, ..Default::default() },
        ] {
            let out = run(p.clone());
            assert!(out.iter().all(|q| q[..3].iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v))), "{p:?}");
            assert_ne!(out, base);
        }
    }

    #[test]
    fn curves_hsl_grading_detail_effects() {
        // Monotone cubic LUT through the points.
        let lut = curve_lut(&[[0.0, 0.0], [64.0, 40.0], [192.0, 220.0], [255.0, 255.0]], 256);
        assert!((lut[64] - 40.0 / 255.0).abs() < 0.01 && (lut[192] - 220.0 / 255.0).abs() < 0.01);
        assert!(lut.windows(2).all(|p| p[1] >= p[0]));
        // Band weights sum to one everywhere.
        for hh in (0..360).step_by(7) {
            let s: f32 = band_weights(hh as f32).iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "{hh}: {s}");
        }
        let (w, h) = (48, 32);
        let reds: Vec<[f32; 4]> = vec![[0.8, 0.2, 0.2, 1.0]; w * h];
        let mut a = reds.clone();
        let mut hs = [0.0; 8];
        hs[0] = -100.0;
        develop(&mut a, w, h, &CameraRaw { hsl_sat: hs, ..Default::default() }, false);
        assert!(a[0][0] - a[0][1] < 0.1, "reds desaturated: {:?}", a[0]);
        let blues: Vec<[f32; 4]> = vec![[0.2, 0.2, 0.8, 1.0]; w * h];
        let mut b = blues.clone();
        develop(&mut b, w, h, &CameraRaw { hsl_sat: hs, ..Default::default() }, false);
        assert!((b[0][2] - 0.8).abs() < 1e-4, "blues untouched");
        // Grading: blue shadows tint dark pixels, leave bright ones nearly alone.
        let mut g: Vec<[f32; 4]> = vec![[0.1, 0.1, 0.1, 1.0], [0.95, 0.95, 0.95, 1.0]];
        develop(&mut g, 2, 1, &CameraRaw { grade_shadows: Wheel { hue: 220.0, sat: 80.0, lum: 0.0 }, ..Default::default() }, false);
        assert!(g[0][2] > g[0][0] + 0.05 && (g[1][2] - g[1][0]).abs() < 0.02, "{g:?}");
        // Point curve + parametric.
        let mut c = img(w, h);
        develop(&mut c, w, h, &CameraRaw { point_curve: vec![[0.0, 30.0], [255.0, 255.0]], curve_shadows: 50.0, ..Default::default() }, false);
        assert!(c.iter().all(|q| q[1] >= 30.0 / 255.0 - 1e-4));
        // Noise reduction reduces variance; sharpening increases it; grain is deterministic.
        let noisy: Vec<[f32; 4]> = (0..w * h)
            .map(|i| {
                let n = hash01(i as i64, 0, 3) * 0.2;
                [0.4 + n, 0.4 + n, 0.4 + n, 1.0]
            })
            .collect();
        let var = |a: &[[f32; 4]]| {
            let m = mean(a, 1);
            a.iter().map(|q| (q[1] - m).powi(2)).sum::<f32>() / a.len() as f32
        };
        let mut nr = noisy.clone();
        develop(&mut nr, w, h, &CameraRaw { noise_luminance: 80.0, ..Default::default() }, false);
        assert!(var(&nr) < var(&noisy) * 0.5);
        let mut sh = noisy.clone();
        develop(&mut sh, w, h, &CameraRaw { sharpen_amount: 100.0, ..Default::default() }, false);
        assert!(var(&sh) > var(&noisy));
        let mut g1 = img(w, h);
        let mut g2 = img(w, h);
        let gp = CameraRaw { grain_amount: 50.0, seed: 4, ..Default::default() };
        develop(&mut g1, w, h, &gp, false);
        develop(&mut g2, w, h, &gp, false);
        assert_eq!(g1, g2);
        assert_ne!(g1, img(w, h));
        // Post-crop vignette darkens corners, leaves the centre.
        let flat = vec![[0.5f32, 0.5, 0.5, 1.0]; w * h];
        let mut v = flat.clone();
        develop(&mut v, w, h, &CameraRaw { vignette_amount: -80.0, ..Default::default() }, false);
        assert!(v[0][0] < 0.35 && (v[(h / 2) * w + w / 2][0] - 0.5).abs() < 0.01, "{:?}", v[0]);
    }

    #[test]
    fn float_documents_keep_overrange() {
        let mut a = vec![[2.0f32, 1.5, 0.5, 1.0]; 16];
        develop(&mut a, 4, 4, &CameraRaw { exposure: 0.5, ..Default::default() }, true);
        assert!(a[0][0] > 2.0);
    }
}
