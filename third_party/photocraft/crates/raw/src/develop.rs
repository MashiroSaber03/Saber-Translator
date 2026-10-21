//! Raw development: linearize, subtract black, scale to white, white
//! balance, clip highlights, demosaic, camera → XYZ (D50) → linear ProPhoto,
//! exposure, Dehaze, Shadows / Highlights / Clarity / Texture and the black
//! point (`tone.rs`), the profile's tone curve, gamma 1.8 encode to 16 bits,
//! orientation.

use crate::color::{self, Mat3};
use crate::demosaic::{Demosaic, PAD, Padded, Pattern, demosaic, demosaic_cfa};
use crate::error::{RawError, Result};
use crate::look::{CameraLook, ToneCurve};
use crate::sensor::Sensor;
use crate::tone::{Haze, Tone, ToneMap, black_level};
use crate::{Limits, RawFormat, par};

/// White balance choice.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum WhiteBalance {
    /// The camera's as-shot white balance when the file records one, else automatic.
    #[default]
    AsShot,
    /// Grey-world estimate from the image.
    Auto,
    /// Explicit R, G, B multipliers applied to the camera channels.
    Multipliers([f64; 3]),
}

/// Development settings.
#[derive(Debug, Clone, PartialEq)]
pub struct DevelopOptions {
    pub demosaic: Demosaic,
    pub white_balance: WhiteBalance,
    /// Exposure compensation in EV, added to the file's baseline exposure.
    pub exposure: f64,
    /// Apply the file's orientation (rotate / flip the pixels).
    pub orient: bool,
    /// Camera Raw's Shadows, Highlights, Clarity, Texture and Dehaze, applied in linear light
    /// before the tone curve.
    pub tone: Tone,
    pub limits: Limits,
    /// A colour rendering fitted to the camera's own JPEG ([`crate::fit_look`]). When set it
    /// replaces the file's colour calibration (or the neutral fallback) and adds its tone curve.
    pub camera_look: Option<CameraLook>,
}

impl Default for DevelopOptions {
    fn default() -> Self {
        DevelopOptions {
            demosaic: Demosaic::default(),
            white_balance: WhiteBalance::default(),
            exposure: 0.0,
            orient: true,
            tone: Tone::default(),
            limits: Limits::default(),
            camera_look: None,
        }
    }
}

/// What was developed and how.
#[derive(Debug, Clone, PartialEq)]
pub struct RawInfo {
    pub format: RawFormat,
    pub make: Option<String>,
    pub model: Option<String>,
    /// Sensor data size.
    pub sensor_width: usize,
    pub sensor_height: usize,
    /// CFA pattern at the crop origin: `"RGGB"` for Bayer; other patterns list
    /// their whole repeat row by row, rows separated by `/` (X-Trans:
    /// `"GGRGGB/GGBGGR/…"`); `None` for LinearRaw.
    pub cfa: Option<String>,
    /// White-balance multipliers used (R, G, B, normalized to a minimum of 1).
    pub wb_multipliers: [f64; 3],
    pub orientation: u16,
    pub baseline_exposure: f64,
}

/// A developed image: 16-bit RGB in ProPhoto RGB (ROMM primaries, D50, gamma 1.8).
#[derive(Debug, Clone)]
pub struct Developed {
    pub width: u32,
    pub height: u32,
    /// Interleaved RGB, `width * height * 3` samples.
    pub rgb: Vec<u16>,
    pub info: RawInfo,
    /// Notes about anything approximated or not applied.
    pub warnings: Vec<String>,
}

/// Linearization + black/white scaling for one sample (pre white balance), 0..1.
pub(crate) struct Scale<'a> {
    s: &'a Sensor,
    lin: Option<&'a [u16]>,
}

impl<'a> Scale<'a> {
    /// The scale for `s` (lazily: `lin` is the sensor's linearization table).
    pub(crate) fn new(s: &'a Sensor) -> Self {
        Scale { s, lin: s.linearization.as_deref() }
    }

    #[inline]
    pub(crate) fn value(&self, x: usize, y: usize, sample: usize) -> f32 {
        let s = self.s;
        let raw = s.data[(y * s.width + x) * s.samples + sample];
        let v = match self.lin {
            Some(l) => f32::from(l.get(usize::from(raw)).or_else(|| l.last()).copied().unwrap_or(raw)),
            None => f32::from(raw),
        };
        let black = s.black.at(x.saturating_sub(s.active.x), y.saturating_sub(s.active.y), sample, s.samples);
        let white = s.white[sample.min(2)];
        let range = white - black;
        if range.is_nan() || range < 1.0 {
            return 0.0;
        }
        let n = (v - black) / range;
        if n.is_finite() { n } else { 0.0 }
    }
}

/// Grey-world neutral (camera-space R, G, B) from unclipped mid-range samples.
fn grey_world(s: &Sensor, sc: &Scale) -> [f64; 3] {
    let c = s.crop;
    let step = (c.height / 512).max(1);
    let rows: Vec<usize> = (0..c.height).step_by(step).collect();
    let parts = par::map(rows.len(), |i| {
        let y = c.y + rows[i];
        let mut sum = [0.0f64; 3];
        let mut n = [0u64; 3];
        for x in c.x..c.x + c.width {
            for k in 0..s.samples.min(3) {
                let ch = match &s.cfa {
                    Some(cfa) => usize::from(cfa.color(x, y)),
                    None => k,
                };
                let v = sc.value(x, y, k);
                if v > 0.002 && v < 0.9 {
                    sum[ch] += f64::from(v);
                    n[ch] += 1;
                }
            }
        }
        (sum, n)
    });
    let mut sum = [0.0f64; 3];
    let mut n = [0u64; 3];
    for (s, k) in parts {
        for c in 0..3 {
            sum[c] += s[c];
            n[c] += k[c];
        }
    }
    let mean = [0, 1, 2].map(|c| if n[c] > 0 { sum[c] / n[c] as f64 } else { 0.0 });
    if mean.iter().any(|m| m.is_nan() || *m <= 1e-6) {
        return [1.0; 3];
    }
    let m = mean.iter().cloned().fold(f64::MIN, f64::max);
    mean.map(|v| v / m)
}

fn normalize(n: [f64; 3]) -> Option<[f64; 3]> {
    let m = n.iter().cloned().fold(f64::MIN, f64::max);
    (m > 0.0 && n.iter().all(|v| v.is_finite() && *v > 1e-4)).then(|| n.map(|v| v / m))
}

/// The camera-space neutral (max component 1) for `opts.white_balance`, as
/// [`develop_sensor`] uses it (the same white balance [`crate::fit_look`] balances with).
pub(crate) fn resolve_neutral(s: &Sensor, opts: &DevelopOptions, sc: &Scale, warnings: &mut Vec<String>) -> Result<[f64; 3]> {
    // Monochrome data has no colour to balance.
    if s.is_monochrome() {
        return Ok([1.0; 3]);
    }
    // White balance: camera-space neutral, max component 1.
    let as_shot = s
        .color
        .as_shot_neutral
        .or_else(|| s.color.as_shot_white_xy.and_then(|xy| s.color.xy_to_neutral(xy)))
        .or_else(|| s.camera_wb.map(|m| m.map(|v| 1.0 / v)))
        .and_then(normalize);
    Ok(match opts.white_balance {
        WhiteBalance::AsShot => match as_shot {
            Some(n) => n,
            None => {
                warnings.push("no as-shot white balance in the file; estimated automatically (grey world)".into());
                grey_world(s, sc)
            }
        },
        WhiteBalance::Auto => grey_world(s, sc),
        WhiteBalance::Multipliers(m) => {
            normalize(m.map(|v| 1.0 / v)).ok_or_else(|| RawError::Malformed("white-balance multipliers must be positive".into()))?
        }
    })
}

/// Develops sensor data.
pub fn develop_sensor(s: &Sensor, opts: &DevelopOptions) -> Result<Developed> {
    let c = s.crop;
    let samples = s.width.checked_mul(s.height).and_then(|n| n.checked_mul(s.samples));
    if c.is_empty()
        || c.x.checked_add(c.width).is_none_or(|x| x > s.width)
        || c.y.checked_add(c.height).is_none_or(|y| y > s.height)
        || samples.is_none_or(|n| s.data.len() < n)
    {
        return Err(RawError::malformed("crop lies outside the sensor data"));
    }
    // LinearRaw can be monochrome (one sample) or RGB (three); CFA data needs one.
    if s.samples != 1 && s.samples != 3 {
        return Err(RawError::unsupported(format!("{} samples per pixel", s.samples)));
    }
    if let Some(cfa) = &s.cfa
        && (s.samples != 1
            || cfa.width == 0
            || cfa.height == 0
            || cfa.width > 16
            || cfa.height > 16
            || cfa.colors.len() != cfa.width * cfa.height
            || cfa.colors.iter().any(|c| *c > 2))
    {
        return Err(RawError::malformed("invalid sensor CFA pattern"));
    }
    let bayer = s.cfa.as_ref().is_none_or(|cfa| cfa.is_bayer());
    let ahd_jobs = working_memory(c.width, c.height, s.cfa.is_some(), bayer, opts, par::workers())?;
    let mut warnings = s.warnings.clone();
    let sc = Scale::new(s);

    let neutral = resolve_neutral(s, opts, &sc, &mut warnings)?;
    let mult = neutral.map(|v| (1.0 / v) as f32);

    // Balanced camera → XYZ D50: a look fitted to the camera's JPEG, the file's calibration,
    // or the neutral fallback.
    // Monochrome data stays neutral grey: no camera look and no colour calibration.
    let mono = s.is_monochrome();
    let look = opts.camera_look.as_ref().filter(|l| l.is_usable() && !mono);
    if opts.camera_look.is_some() && look.is_none() && !mono {
        warnings.push("the supplied camera look is not usable (non-finite or singular); ignored".into());
    }
    let to_xyz: Mat3 = match (look, s.color.balanced_to_xyz_d50(neutral)) {
        (Some(l), _) => {
            warnings.push(l.note());
            l.to_xyz
        }
        (None, Some(m)) => m,
        (None, None) => {
            if s.format != RawFormat::Dng && !mono {
                warnings.push("no colour calibration for this camera; camera colours are treated as sRGB primaries".into());
            }
            color::srgb_to_xyz_d50()
        }
    };
    let gain = 2f64.powf(s.baseline_exposure + opts.exposure.clamp(-10.0, 10.0));
    let out_m = color::scale(&color::mul(&color::xyz_d50_to_prophoto(), &to_xyz), gain);
    let out_m: [[f32; 3]; 3] = out_m.map(|r| r.map(|v| v as f32));

    // Neutral grey through `out_m`: the gain alone (the matrix maps camera white to ProPhoto
    // white, up to rounding), so the three output channels stay exactly equal.
    let grey_gain = gain as f32;
    let (w, h) = (c.width, c.height);
    let band = par::band_rows(w);
    let mut rgb: Vec<f32>;
    let pattern: Option<String>;
    match &s.cfa {
        Some(cfa) => {
            // Normalized, balanced, clipped CFA plane.
            let mut plane = Padded::new(w, h);
            let stride = plane.stride;
            // Fast path: black varies at most per 2×2 position, so black and gain
            // repeat every `period` columns (the CFA width, made even).
            let bl = &s.black;
            let periodic = bl.delta_h.is_empty() && bl.delta_v.is_empty() && matches!(bl.rows, 1 | 2) && matches!(bl.cols, 1 | 2) && cfa.width <= 16;
            let period = if cfa.width % 2 == 0 { cfa.width.max(2) } else { cfa.width * 2 };
            plane.fill_rows(band, |y0, chunk| {
                for (r, prow) in chunk.chunks_exact_mut(stride).enumerate() {
                    let row = &mut prow[PAD..PAD + w];
                    let y = c.y + y0 + r;
                    if periodic {
                        // Per column of the period: black and the combined scale × white-balance gain.
                        let k: Vec<(f32, f32)> = (0..period)
                            .map(|p| {
                                let x = c.x + p;
                                let black = bl.at(x.saturating_sub(s.active.x), y.saturating_sub(s.active.y), 0, 1);
                                let range = s.white[0] - black;
                                let gain = if range >= 1.0 { mult[usize::from(cfa.color(x, y)).min(2)] / range } else { 0.0 };
                                (black, gain)
                            })
                            .collect();
                        let src = &s.data[y * s.width + c.x..y * s.width + c.x + w];
                        let ks = k.iter().cycle();
                        match &sc.lin {
                            None => {
                                for ((v, &raw), &(black, gain)) in row.iter_mut().zip(src).zip(ks) {
                                    *v = ((f32::from(raw) - black) * gain).clamp(0.0, 1.0);
                                }
                            }
                            Some(l) => {
                                for ((v, &raw), &(black, gain)) in row.iter_mut().zip(src).zip(ks) {
                                    let lin = f32::from(l.get(usize::from(raw)).or_else(|| l.last()).copied().unwrap_or(raw));
                                    *v = ((lin - black) * gain).clamp(0.0, 1.0);
                                }
                            }
                        }
                    } else {
                        for (i, v) in row.iter_mut().enumerate() {
                            let x = c.x + i;
                            let ch = usize::from(cfa.color(x, y)).min(2);
                            *v = (sc.value(x, y, 0) * mult[ch]).clamp(0.0, 1.0);
                        }
                    }
                    // Lens-shading gain maps, after the white-balance clip so clipped
                    // highlights stay neutral.
                    if let Some(g) = gain_row(s, y, c.x, w, 0) {
                        for (v, g) in row.iter_mut().zip(g) {
                            *v = (*v * g).clamp(0.0, 1.0);
                        }
                    }
                }
            });
            plane.fill_borders();
            let names = |p: &[u8]| p.iter().map(|c| ['R', 'G', 'B'][usize::from(*c).min(2)]).collect::<String>();
            if bayer {
                let p = cfa.phase(c.x, c.y);
                pattern = Some(names(&p));
                rgb = demosaic(&plane, p, opts.demosaic, &to_xyz, ahd_jobs);
            } else {
                let pat = Pattern::of(cfa, c.x, c.y);
                pattern = Some(pat.colors.chunks(pat.w.max(1)).map(names).collect::<Vec<_>>().join("/"));
                rgb = demosaic_cfa(&plane, &pat);
            }
            drop(plane);
        }
        None => {
            pattern = None;
            rgb = vec![0.0f32; w * h * 3];
            par::chunks_mut(&mut rgb, band * w * 3, |b, chunk| {
                for (r, row) in chunk.chunks_exact_mut(w * 3).enumerate() {
                    let y = c.y + b * band + r;
                    for i in 0..w {
                        for k in 0..3 {
                            // Monochrome: the one sample in all three channels.
                            let sample = k.min(s.samples.saturating_sub(1));
                            row[i * 3 + k] = (sc.value(c.x + i, y, sample) * mult[k]).clamp(0.0, 1.0);
                        }
                    }
                    for k in 0..s.samples.min(3) {
                        if let Some(g) = gain_row(s, y, c.x, w, k) {
                            for (i, g) in g.into_iter().enumerate() {
                                row[i * 3 + k] = (row[i * 3 + k] * g).clamp(0.0, 1.0);
                            }
                        }
                    }
                }
            });
        }
    }

    // Camera → ProPhoto, exposure, Dehaze, Shadows / Highlights / Clarity / Texture and the black
    // point, the look's (or the profile's) tone curve, gamma 1.8, 16 bits.
    let y_row = color::rgb_to_xyz(color::ROMM, color::D50_XY)[1].map(|v| v as f32);
    let lum = |v: &[f32; 3]| y_row[0] * v[0] + y_row[1] * v[1] + y_row[2] * v[2];
    // Monochrome: the one sample, grey; no colour matrix.
    let pro = |p: &[f32]| if mono { [grey_gain * p[0]; 3] } else { [0, 1, 2].map(|k| out_m[k][0] * p[0] + out_m[k][1] * p[1] + out_m[k][2] * p[2]) };
    let haze = Haze::new(w, h, |i| pro(&rgb[i * 3..i * 3 + 3]), opts.tone.dehaze);
    // Linear ProPhoto of pixel `i` after Dehaze.
    let linear = |i: usize, p: &[f32]| {
        let v = pro(p);
        haze.as_ref().map_or(v, |z| z.apply(i % w, i / w, v))
    };
    let mut map = ToneMap::new(w, h, |i| lum(&linear(i, &rgb[i * 3..i * 3 + 3])), opts.tone, s.black_point);
    if let (Some(m), Some(_)) = (&mut map, &haze) {
        // The black point belongs to the default rendering: Dehaze does not move it (taken after
        // Dehaze it fell to almost 0, and Dehaze darkened less than Photoshop's).
        m.black = black_level(w, h, |i| lum(&pro(&rgb[i * 3..i * 3 + 3])), s.black_point);
    }
    let mut out = vec![0u16; w * h * 3];
    let lut = match look {
        Some(l) => GammaLut::with_tone(l.tone.as_ref()),
        None => GammaLut::with_curve(&s.tone_curve),
    };
    par::chunks_mut(&mut out, band * w * 3, |b, chunk| {
        let start = b * band * w * 3;
        let src = &rgb[start..start + chunk.len()];
        for (i, (o, p)) in chunk.as_chunks_mut::<3>().0.iter_mut().zip(src.as_chunks::<3>().0).enumerate() {
            let mut v = linear(b * band * w + i, p);
            if let Some(m) = &map {
                let g = m.gain(i % w, b * band + i / w, lum(&v));
                // The black point first, as Camera Raw: lifting before it would flatten the
                // deepest shadows' contrast.
                v = v.map(|c| m.unblack(c) * g);
            }
            for k in 0..3 {
                o[k] = lut.encode(v[k]);
            }
        }
    });
    rgb.clear();
    rgb.shrink_to_fit();

    let (out, ow, oh) = if opts.orient { orient(out, w, h, s.orientation) } else { (out, w, h) };
    let nmax = mult.iter().cloned().fold(f32::MAX, f32::min).max(1e-6);
    Ok(Developed {
        width: ow as u32,
        height: oh as u32,
        rgb: out,
        info: RawInfo {
            format: s.format,
            make: s.make.clone(),
            model: s.model.clone(),
            sensor_width: s.width,
            sensor_height: s.height,
            cfa: pattern,
            wb_multipliers: mult.map(|m| f64::from(m / nmax)),
            orientation: s.orientation,
            baseline_exposure: s.baseline_exposure,
        },
        warnings,
    })
}

/// Checks the peak of the two development phases. The CFA and non-Bayer green
/// planes are freed before the u16 output is allocated; AHD scratch is bounded
/// by reducing simultaneous bands when needed. Sensor storage belongs to the
/// caller and is checked separately by the decoder.
fn working_memory(w: usize, h: usize, cfa: bool, bayer: bool, opts: &DevelopOptions, workers: usize) -> Result<usize> {
    let limits = &opts.limits;
    limits.check(w as u64, h as u64, 18)?;
    let pixels = (w as u64).saturating_mul(h as u64);
    let output = pixels.saturating_mul(18).saturating_add((GAMMA_STEPS as u64 + 2) * 4);
    let mut peak = output;
    let mut jobs = workers.max(1);
    if cfa {
        let plane = (w as u64).saturating_add(2 * PAD as u64).saturating_mul((h as u64).saturating_add(2 * PAD as u64)).saturating_mul(4);
        let mut mosaic = pixels.saturating_mul(if bayer { 12 } else { 16 }).saturating_add(plane);
        if bayer && opts.demosaic == Demosaic::Ahd {
            mosaic = mosaic.saturating_add(crate::demosaic::AHD_TABLE_BYTES);
            let scratch = crate::demosaic::ahd_scratch(w as u64, h as u64);
            let available = limits.max_alloc.saturating_sub(mosaic) / scratch.max(1);
            jobs = jobs.min(crate::demosaic::ahd_bands(h)).min(usize::try_from(available).unwrap_or(usize::MAX));
            // At least one band must fit, even when no parallelism is available.
            mosaic = mosaic.saturating_add(scratch.saturating_mul(jobs.max(1) as u64));
        }
        peak = peak.max(mosaic);
    }
    if peak > limits.max_alloc || usize::try_from(peak).is_err() {
        return Err(RawError::LimitExceeded(format!("{peak} bytes of RAW working memory exceed {}", limits.max_alloc)));
    }
    Ok(jobs.max(1))
}

/// Gain-map factors (DNG OpcodeList2 GainMap) for data row `y`, columns
/// `x0..x0 + w`, sample `k`; `None` when no map touches the row.
fn gain_row(s: &Sensor, y: usize, x0: usize, w: usize, k: usize) -> Option<Vec<f32>> {
    let a = s.active;
    if s.gain_maps.is_empty() || y < a.y || a.width == 0 || a.height == 0 {
        return None;
    }
    let ay = y - a.y;
    // Map points are placed in coordinates relative to the (active) image.
    let rv = (ay as f64 + 0.5) / a.height as f64;
    let mut out: Option<Vec<f32>> = None;
    for m in &s.gain_maps {
        if k < m.plane || k >= m.plane.saturating_add(m.planes) || ay < m.top || ay >= m.bottom || !(ay - m.top).is_multiple_of(m.row_pitch) {
            continue;
        }
        let row = m.row(rv, k - m.plane);
        let o = out.get_or_insert_with(|| vec![1.0; w]);
        for (i, f) in o.iter_mut().enumerate() {
            let x = x0 + i;
            if x >= a.x && m.covers(x - a.x, ay) {
                *f *= m.at(&row, ((x - a.x) as f64 + 0.5) / a.width as f64);
            }
        }
    }
    out
}

/// Linear 0..1 → gamma-1.8 16-bit code. Tabulated over `s = sqrt(v)`, where
/// the curve `s^(2/1.8)` is smooth down to 0, with linear interpolation
/// (error below 0.2 code values).
struct GammaLut {
    table: Vec<f32>,
}

const GAMMA_STEPS: usize = 4096;

impl GammaLut {
    #[cfg(test)]
    fn new() -> Self {
        Self::with_curve(&[])
    }

    /// The gamma-1.8 encoding after a tone curve (see [`Sensor::tone_curve`]; empty: none).
    fn with_curve(curve: &[[f32; 2]]) -> Self {
        let table = (0..=GAMMA_STEPS + 1)
            .map(|i| {
                let s = (i as f32 / GAMMA_STEPS as f32).min(1.0);
                tone(curve, s * s).clamp(0.0, 1.0).powf(1.0 / 1.8) * 65535.0
            })
            .collect();
        GammaLut { table }
    }

    /// The gamma-1.8 encoding after a fitted [`ToneCurve`] (`None`: none, plain gamma 1.8).
    fn with_tone(tone: Option<&ToneCurve>) -> Self {
        let table = (0..=GAMMA_STEPS + 1)
            .map(|i| {
                let s = (i as f64 / GAMMA_STEPS as f64).min(1.0);
                match tone {
                    Some(t) => (t.eval(s * s).powf(1.0 / 1.8) * 65535.0) as f32,
                    None => (s as f32).powf(2.0 / 1.8) * 65535.0,
                }
            })
            .collect();
        GammaLut { table }
    }

    #[inline]
    fn encode(&self, v: f32) -> u16 {
        let v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
        let x = v.sqrt() * GAMMA_STEPS as f32;
        let i = (x as usize).min(GAMMA_STEPS);
        let f = x - i as f32;
        let (a, b) = (self.table[i], self.table[i + 1]);
        (a + (b - a) * f + 0.5) as u16
    }
}

/// `curve` at linear `v`: linear interpolation between the points in log₂ of the input, a line
/// through 0 below the first point, the last output beyond the last.
pub(crate) fn tone(curve: &[[f32; 2]], v: f32) -> f32 {
    let (Some(first), Some(last)) = (curve.first(), curve.last()) else { return v };
    if v <= first[0] {
        return if first[0] > 0.0 { v * first[1] / first[0] } else { first[1] };
    }
    if v >= last[0] {
        return last[1];
    }
    let i = curve.partition_point(|p| p[0] <= v).clamp(1, curve.len() - 1);
    let ([x0, y0], [x1, y1]) = (curve[i - 1], curve[i]);
    let f = (v.log2() - x0.log2()) / (x1.log2() - x0.log2());
    y0 + (y1 - y0) * f
}

/// The source position in a `w × h` image of output pixel (`x2`, `y2`) after
/// applying TIFF orientation `o` (output in range when the input is).
pub(crate) fn source_xy(o: u16, w: usize, h: usize, x2: usize, y2: usize) -> (usize, usize) {
    let (wl, hl) = (w.saturating_sub(1), h.saturating_sub(1));
    match o {
        2 => (wl.saturating_sub(x2), y2),
        3 => (wl.saturating_sub(x2), hl.saturating_sub(y2)),
        4 => (x2, hl.saturating_sub(y2)),
        5 => (y2, x2),
        6 => (y2, hl.saturating_sub(x2)),
        7 => (wl.saturating_sub(y2), hl.saturating_sub(x2)),
        8 => (wl.saturating_sub(y2), x2),
        _ => (x2, y2),
    }
}

/// Applies a TIFF orientation (1–8) to interleaved RGB.
pub(crate) fn orient(src: Vec<u16>, w: usize, h: usize, o: u16) -> (Vec<u16>, usize, usize) {
    if !(2..=8).contains(&o) {
        return (src, w, h);
    }
    let swap = o >= 5;
    let (ow, oh) = if swap { (h, w) } else { (w, h) };
    let mut out = vec![0u16; src.len()];
    let band = par::band_rows(ow);
    par::chunks_mut(&mut out, band * ow * 3, |b, chunk| {
        for (r, row) in chunk.chunks_exact_mut(ow * 3).enumerate() {
            let y2 = b * band + r;
            for x2 in 0..ow {
                let (x, y) = match o {
                    2 => (w - 1 - x2, y2),
                    3 => (w - 1 - x2, h - 1 - y2),
                    4 => (x2, h - 1 - y2),
                    5 => (y2, x2),
                    6 => (y2, h - 1 - x2),
                    7 => (w - 1 - y2, h - 1 - x2),
                    _ => (w - 1 - y2, x2),
                };
                let i = (y * w + x) * 3;
                row[x2 * 3..x2 * 3 + 3].copy_from_slice(&src[i..i + 3]);
            }
        }
    });
    (out, ow, oh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hundred_megapixel_bayer_fits_without_raising_default_limits() {
        let mut opts = DevelopOptions::default();
        for method in [Demosaic::Bilinear, Demosaic::Mhc, Demosaic::Ahd] {
            opts.demosaic = method;
            for workers in [1, 10, 32, 128] {
                let jobs = working_memory(11648, 8736, true, true, &opts, workers).unwrap();
                assert!(jobs <= workers);
                if method == Demosaic::Ahd && workers >= 32 {
                    assert!(jobs < workers);
                }
            }
        }
        opts.limits.max_alloc = 1 << 30;
        assert!(working_memory(11648, 8736, true, true, &opts, 1).is_err());
    }

    #[test]
    fn working_memory_accounts_for_padding_and_at_least_one_ahd_band() {
        let mut opts = DevelopOptions { demosaic: Demosaic::Ahd, ..Default::default() };
        opts.limits.max_alloc = 100 * 100 * 22;
        assert!(working_memory(100, 100, true, true, &opts, 1).is_err());
        opts.demosaic = Demosaic::Mhc;
        assert!(working_memory(100, 100, true, true, &opts, 1).is_ok());
        opts.limits.max_alloc = 100 * 100 * 18;
        assert!(working_memory(100, 100, false, true, &opts, 1).is_err());
    }

    #[test]
    fn gamma_table_matches_powf() {
        let lut = GammaLut::new();
        for i in 0..=100_000 {
            let v = i as f32 / 100_000.0;
            let exact = v.powf(1.0 / 1.8) * 65535.0;
            assert!((f32::from(lut.encode(v)) - exact).abs() <= 1.0, "{v}: {} vs {exact}", lut.encode(v));
        }
        assert_eq!(lut.encode(-1.0), 0);
        assert_eq!(lut.encode(2.0), 65535);
        assert_eq!(lut.encode(f32::NAN), 0);
    }

    #[test]
    fn tone_curve_is_folded_into_the_gamma_table() {
        let code = |v: f32| v.powf(1.0 / 1.8) * 65535.0;
        let lut = GammaLut::with_curve(&[[0.25, 0.5], [1.0, 1.0]]);
        let near = |got: u16, want: f32| (f32::from(got) - want).abs() <= 2.0;
        // On a point, between points (linear in log2 of the input), on the line through 0 below the first.
        assert!(near(lut.encode(0.25), code(0.5)));
        assert!(near(lut.encode(0.5), code(0.75)), "{} vs {}", lut.encode(0.5), code(0.75));
        assert!(near(lut.encode(0.125), code(0.25)));
        assert_eq!(lut.encode(0.0), 0);
        assert_eq!(lut.encode(1.0), 65535);
        assert!((0..1000).map(|i| lut.encode(i as f32 / 999.0)).collect::<Vec<_>>().windows(2).all(|w| w[1] >= w[0]));
        // No curve: the plain gamma 1.8.
        assert!(near(GammaLut::with_curve(&[]).encode(0.25), code(0.25)));
    }

    #[test]
    fn orientations() {
        // 3×2 image, pixel value = index.
        let src: Vec<u16> = (0..6).flat_map(|i| [i, i, i]).collect();
        let px = |v: &[u16]| v.as_chunks::<3>().0.iter().map(|p| p[0]).collect::<Vec<_>>();
        assert_eq!(px(&orient(src.clone(), 3, 2, 1).0), vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(px(&orient(src.clone(), 3, 2, 2).0), vec![2, 1, 0, 5, 4, 3]);
        assert_eq!(px(&orient(src.clone(), 3, 2, 3).0), vec![5, 4, 3, 2, 1, 0]);
        assert_eq!(px(&orient(src.clone(), 3, 2, 4).0), vec![3, 4, 5, 0, 1, 2]);
        let (v, w, h) = orient(src.clone(), 3, 2, 6);
        assert_eq!((w, h), (2, 3));
        // Rotated 90° clockwise: the left column (0, 3) becomes the top row (3, 0).
        assert_eq!(px(&v), vec![3, 0, 4, 1, 5, 2]);
        assert_eq!(px(&orient(src.clone(), 3, 2, 8).0), vec![2, 5, 1, 4, 0, 3]);
        assert_eq!(px(&orient(src.clone(), 3, 2, 5).0), vec![0, 3, 1, 4, 2, 5]);
        assert_eq!(px(&orient(src, 3, 2, 7).0), vec![5, 2, 4, 1, 3, 0]);
    }
}
