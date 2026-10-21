//! Evaluation of adjustment layers on a composite buffer.
//!
//! Formulas are documented approximations of Photoshop behaviour. Exact matching is tuned in
//! milestone M7 against Photoshop-rendered PSD composites (the oracle in `testkit`).

use photocraft_color::convert::{D50, SRGB_TO_XYZ_D50, XYZ_D50_TO_SRGB, lab_to_srgb, mat3_mul, rgb_to_gray, srgb_to_lab, srgb_to_linear};
use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::Adjustment;
use photocraft_doc::adjust::{CurvePoint, HueRange, LevelsChannel, ToneSpace, lut3d_len};

use crate::Buffer;

/// The document tone curve used to linearize values for adjustments that
/// work in linear light (Exposure). Until ICC profiles are wired in (M8) this
/// approximates the Photoshop defaults: sRGB for colour documents and the
/// "Dot Gain 20%" grey profile (≈ gamma 1.73, fitted on the corpus) for
/// grayscale documents. 32-bit documents store linear light already.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Transfer {
    /// sRGB piecewise curve.
    Srgb,
    /// Pure power law with this exponent.
    Gamma(f32),
}

impl Transfer {
    /// Default transfer for a document's colour mode and depth. 32-bit samples are linear, and
    /// Photoshop's Exposure works on them as stored (photoshop corpus rgb32 and gray32
    /// exposure.psd: within 0.5/255, against 69/255 through the 2.2 curve).
    pub fn for_document(mode: photocraft_color::ColorMode, depth: photocraft_color::SampleType) -> Self {
        if depth == photocraft_color::SampleType::F32 {
            return Transfer::Gamma(1.0);
        }
        match mode {
            photocraft_color::ColorMode::Grayscale | photocraft_color::ColorMode::Duotone | photocraft_color::ColorMode::Bitmap => Transfer::Gamma(1.732),
            _ => Transfer::Srgb,
        }
    }
    /// The tone curve Exposure linearises through: RGB documents use a pure 2.2 power, not the
    /// sRGB curve (psd-tools adjustment_nested_composition_4: an offset of 0.1738 lifts 76 to 134,
    /// not 136; ag-psd masks), grey documents their dot-gain gamma.
    pub fn for_exposure(self) -> Self {
        match self {
            Transfer::Srgb => Transfer::Gamma(2.2),
            t => t,
        }
    }
    /// Tone-encoded value → linear light.
    pub fn decode(self, v: f32) -> f32 {
        match self {
            Transfer::Srgb => photocraft_color::convert::srgb_to_linear(v.max(0.0)),
            Transfer::Gamma(1.0) => v.max(0.0),
            Transfer::Gamma(g) => v.max(0.0).powf(g),
        }
    }
    /// Linear light → tone-encoded value.
    pub fn encode(self, v: f32) -> f32 {
        match self {
            Transfer::Srgb => photocraft_color::convert::linear_to_srgb(v.max(0.0)),
            Transfer::Gamma(1.0) => v.max(0.0),
            Transfer::Gamma(g) => v.max(0.0).powf(1.0 / g),
        }
    }
}

/// Applies an adjustment assuming an sRGB document.
pub fn apply(adj: &Adjustment, buf: &mut Buffer) {
    apply_with(adj, buf, Transfer::Srgb);
}

/// Applies an adjustment with the document's tone transfer.
pub fn apply_with(adj: &Adjustment, buf: &mut Buffer, transfer: Transfer) {
    apply_depth(adj, buf, transfer, None);
}

/// Applies an adjustment with the document's tone transfer, in a document of `depth` (`None`:
/// unrounded). Integer depths work on whole levels like Photoshop's (see [`levels_q`] and
/// `adjustment_quantum`), 32-bit documents get its float Levels (see [`levels_float`]).
pub fn apply_depth(adj: &Adjustment, buf: &mut Buffer, transfer: Transfer, depth: Option<SampleType>) {
    match adj {
        Adjustment::Invert => map_rgb(buf, |c| [1.0 - c[0], 1.0 - c[1], 1.0 - c[2]]),
        Adjustment::Threshold { level } => {
            // Compared on the 8-bit grid like Photoshop (avoids float ties).
            let t = (level * 255.0).round();
            map_rgb(buf, |c| {
                let v = if (rgb_to_gray(c) * 255.0).round() >= t { 1.0 } else { 0.0 };
                [v; 3]
            })
        }
        Adjustment::Posterize { levels } => map_rgb(buf, |c| c.map(|v| posterize(v, *levels))),
        Adjustment::BrightnessContrast { brightness, contrast, legacy: true } => {
            // Legacy: contrast scales around 128/255, with brightness added before it when
            // contrast is raised and after it when contrast is lowered. Measured on Photoshop
            // 27.11, brightness +40 with contrast +30 maps 128 to 185 (brightness first), while
            // the corpus's brightness -23 with contrast -26 (psd-tools) needs contrast first;
            // contrast 80 keeps 128 at 128 (a 0.5 pivot moved it to 130).
            let f = legacy_brightness_contrast(*brightness, *contrast);
            map_rgb(buf, |px| px.map(&f))
        }
        Adjustment::BrightnessContrast { brightness, contrast, .. } => {
            // Modern (CS3+) Brightness/Contrast, reverse-engineered from Photoshop ground truth
            // (a 0..255 ramp pushed through the real app; see log/devlog.md). Unlike the legacy
            // linear scale, both are smooth curves that pin pure black and white:
            //   • Brightness: a line of slope s = 1.375^(b/50) from the origin that rolls off to
            //     (1,1) via a `v^P` white-anchor term (P grows as |b| grows).
            //   • Contrast: a symmetric cubic-Hermite S-curve pivoting at 0.5, with endpoint slope
            //     1 - c/128 and centre slope 1 + c/128 (near-exact: ≤0.5/255 vs Photoshop).
            // Applied brightness-then-contrast. Exact at the sliders' zero and extremes still drift
            // on the brightness side (the real curve is a spline); contrast matches closely.
            let b = *brightness;
            let c = *contrast;
            map_rgb(buf, |px| px.map(|v| modern_contrast(modern_brightness(v, b), c).clamp(0.0, 1.0)))
        }
        Adjustment::Exposure { exposure, offset, gamma } => {
            // In linear light: (lin·2^exposure + offset)^(1/gamma), then back
            // through the document tone curve (fitted on the corpus).
            let m = 2f32.powf(*exposure);
            let g = gamma.max(0.01);
            let transfer = transfer.for_exposure();
            let hdr = depth == Some(SampleType::F32);
            map_rgb(buf, |c| {
                c.map(|v| {
                    if hdr {
                        // Float documents store linear samples. Photoshop's 32-bit merged
                        // Exposure oracle also retains negative results with signed gamma.
                        let lin = v * m + offset;
                        return if g == 1.0 { lin } else { lin.signum() * lin.abs().powf(1.0 / g) };
                    }
                    let lin = (transfer.decode(v) * m + offset).max(0.0);
                    let lin = if g == 1.0 { lin } else { lin.powf(1.0 / g) };
                    transfer.encode(lin).clamp(0.0, 1.0)
                })
            })
        }
        Adjustment::Levels { master, per_channel, space: ToneSpace::Rgb, .. } if depth == Some(SampleType::F32) => {
            // A 0..1 lookup table cannot represent signed, unbounded float samples.
            let master_identity = *master == LevelsChannel::default();
            let channel_identity = per_channel.each_ref().map(|ch| *ch == LevelsChannel::default());
            if !master_identity || channel_identity.iter().any(|&v| !v) {
                map_rgb(buf, |c| {
                    std::array::from_fn(|i| {
                        let v = if channel_identity[i] { c[i] } else { levels_float(&per_channel[i], c[i]) };
                        if master_identity { v } else { levels_float(master, v) }
                    })
                });
            }
        }
        Adjustment::Levels { space, .. } | Adjustment::Curves { space, .. } => {
            let luts = tone_luts_depth(adj, depth);
            let identity = float_tone_identity(adj, depth);
            match space {
                ToneSpace::Rgb => map_rgb(buf, |c| std::array::from_fn(|i| if identity[i] { c[i] } else { lut(&luts[i], c[i]) })),
                ToneSpace::Cmyk | ToneSpace::Lab => map_rgb(buf, |c| tone_in_space(*space, &luts, c)),
            }
        }
        Adjustment::HueSaturation { hue, saturation, lightness, colorize, ranges } => {
            let table = (!*colorize && ranges.iter().any(|r| !r.is_neutral())).then(|| hue_range_tables(ranges));
            map_rgb(buf, |c| {
                let (dh, ds, dl) = match &table {
                    Some(t) => {
                        // Range edits follow the pixel's original hue, in full however dull the colour
                        // (Photoshop's merged images, #4089). Greys stay grey: no hue to shift, no
                        // saturation to scale, and range Lightness moves towards a channel they share.
                        let h0 = rgb_to_hsl(c).0;
                        (lut(&t[0], h0), lut(&t[1], h0), lut(&t[2], h0))
                    }
                    None => (0.0, 0.0, 0.0),
                };
                let out = hue_saturation(c, *hue + dh, (*saturation / 100.0 + ds).clamp(-1.0, 1.0), (*lightness / 100.0).clamp(-1.0, 1.0), *colorize);
                range_lightness(out, dl)
            })
        }
        Adjustment::Vibrance { vibrance, saturation } => {
            let (v, s) = (*vibrance / 100.0, *saturation / 100.0);
            let hi = if depth == Some(SampleType::F32) { f32::INFINITY } else { 1.0 };
            map_rgb(buf, |c| vibrance_px(c, v, s, transfer, hi))
        }
        Adjustment::ChannelMixer { matrix, monochrome } => map_rgb(buf, |c| {
            let mix = |row: &[f32; 4]| (row[0] * c[0] + row[1] * c[1] + row[2] * c[2] + row[3]).clamp(0.0, 1.0);
            if *monochrome { [mix(&matrix[0]); 3] } else { [mix(&matrix[0]), mix(&matrix[1]), mix(&matrix[2])] }
        }),
        Adjustment::PhotoFilter { color, density, preserve_luminosity } => {
            let linear_doc = depth == Some(SampleType::F32);
            let m = photo_filter_matrix(*color, *density, *preserve_luminosity && linear_doc);
            let set_lum = *preserve_luminosity && !linear_doc;
            map_rgb(buf, |c| {
                let f = mat3_mul(&m, c.map(|v| transfer.decode(v))).map(|v| transfer.encode(v));
                let f = if set_lum { photocraft_color::blend::set_lum(f, photocraft_color::blend::lum(c)) } else { f };
                f.map(|v| v.clamp(0.0, 1.0))
            })
        }
        Adjustment::BlackWhite { weights, tint } => map_rgb(buf, |c| {
            let g = black_white_gray(c, weights);
            match tint {
                // Photoshop tints like the Color blend mode: the tint's hue and saturation at the
                // grey's luminosity (SetLum + ClipColor), measured in Photoshop 27.11.
                Some(t) => photocraft_color::blend::set_lum(*t, g).map(|v| v.clamp(0.0, 1.0)),
                None => [g; 3],
            }
        }),
        Adjustment::GradientMap { stops, reverse, dither } => {
            let (w, x0, y0) = (buf.rect.width().max(1) as usize, buf.rect.x0, buf.rect.y0);
            for (i, p) in buf.px.iter_mut().enumerate() {
                if p[3] <= 0.0 {
                    continue;
                }
                let mut t = rgb_to_gray([p[0], p[1], p[2]]);
                if *reverse {
                    t = 1.0 - t;
                }
                let mut o = gradient(stops, t);
                if *dither {
                    let d = bayer4(x0 + (i % w) as i32, y0 + (i / w) as i32) / 255.0;
                    o = o.map(|v| (v + d).clamp(0.0, 1.0));
                }
                p[..3].copy_from_slice(&o);
            }
        }
        Adjustment::ColorBalance { shadows, midtones, highlights, preserve_luminosity } => {
            map_rgb(buf, |c| std::array::from_fn(|ch| color_balance_channel([*shadows, *midtones, *highlights], *preserve_luminosity, ch, c[ch])))
        }
        Adjustment::SelectiveColor { relative, adjustments } => map_rgb(buf, |c| selective_color(c, *relative, adjustments)),
        Adjustment::ColorLookup { lut: Some(table), size, tetrahedral, dither, .. } if lut3d_len(*size).is_some_and(|len| table.len() >= len) => {
            let (n, w, x0, y0) = (*size as usize, buf.rect.width().max(1) as usize, buf.rect.x0, buf.rect.y0);
            for (i, p) in buf.px.iter_mut().enumerate() {
                if p[3] <= 0.0 {
                    continue;
                }
                let mut o = lut3d_sample(table, n, [p[0], p[1], p[2]], *tetrahedral);
                if *dither {
                    let d = bayer4(x0 + (i % w) as i32, y0 + (i / w) as i32) / 255.0;
                    o = o.map(|v| (v + d).clamp(0.0, 1.0));
                }
                p[..3].copy_from_slice(&o);
            }
        }
        // Not evaluated: identity (still round-trips through PSD).
        Adjustment::ColorLookup { .. } | Adjustment::Unsupported { .. } => {}
    }
}

/// Legacy Brightness/Contrast as one channel's tone curve (see [`apply_depth`]).
fn legacy_brightness_contrast(brightness: f32, contrast: f32) -> impl Fn(f32) -> f32 {
    let b = brightness / 255.0;
    let c = contrast.clamp(-100.0, 99.0);
    let k = if c >= 0.0 { 1.0 / (1.0 - c / 100.0) } else { 1.0 + c / 100.0 };
    let (pre, post) = if c >= 0.0 { (b, 0.0) } else { (0.0, b) };
    move |v: f32| ((v + pre - LEGACY_PIVOT) * k + LEGACY_PIVOT + post).clamp(0.0, 1.0)
}

/// Applies an adjustment in a document of colour `mode`: [`apply_depth`], except that in Lab
/// and CMYK documents the adjustments Photoshop evaluates on the document's own channels do so
/// (see [`lab_channel_adjustment`] and [`cmyk_channel_adjustment`]).
pub fn apply_in(adj: &Adjustment, buf: &mut Buffer, transfer: Transfer, depth: Option<SampleType>, mode: ColorMode) {
    if mode == ColorMode::Lab && lab_channel_adjustment(adj) {
        apply_lab(adj, buf, depth);
    } else if mode == ColorMode::Cmyk && cmyk_channel_adjustment(adj) {
        use photocraft_color::convert::{cmyk_to_rgb, rgb_to_cmyk};
        map_rgb(buf, |c| {
            let mut ink = rgb_to_cmyk(c);
            stored_channels(adj, mode, depth, &mut ink, rgb_to_gray(c));
            cmyk_to_rgb(ink)
        });
    } else {
        apply_depth(adj, buf, transfer, depth);
    }
}

/// Whether a Lab document evaluates `adj` on its L, a and b channels rather than on the screen
/// colour: Invert and Posterize (L, a and b alike), Brightness/Contrast (L only), Threshold
/// (split on L, neutral result) and Gradient Map (driven by L). Levels and Curves carry their own
/// Lab records ([`ToneSpace::Lab`]); everything else stays on the screen colour.
pub fn lab_channel_adjustment(adj: &Adjustment) -> bool {
    matches!(
        adj,
        Adjustment::Invert
            | Adjustment::Posterize { .. }
            | Adjustment::Threshold { .. }
            | Adjustment::BrightnessContrast { .. }
            | Adjustment::GradientMap { .. }
    )
}

/// Whether a CMYK document evaluates `adj` on its inks rather than on the screen colour: Invert
/// and Posterize (each ink, black included, so paper inverts to 400% ink) and Threshold (paper or
/// 100% black ink alone). Photoshop corpus cmyk8 and cmyk16 invert, posterize-4 and threshold-128
/// follow these rules on the files' own inks on every pixel; here the inks are the screen colour
/// separated through the document's CMYK profile, so neutrals carry the black that profile
/// generates rather than the file's.
pub fn cmyk_channel_adjustment(adj: &Adjustment) -> bool {
    stored_channel_adjustment(adj)
}

/// Whether [`stored_channels`] evaluates `adj`: Invert, Posterize and Threshold.
pub fn stored_channel_adjustment(adj: &Adjustment) -> bool {
    matches!(adj, Adjustment::Invert | Adjustment::Posterize { .. } | Adjustment::Threshold { .. })
}

/// Invert, Posterize or Threshold on one pixel's own channels in a Lab or CMYK document of
/// `depth` (`None`: unknown, treated as 8-bit): `ch` holds L, a, b normalised as [`lab_norm`]
/// gives them, or the C, M, Y, K inks (0: none); `gray` is the pixel's luma, which CMYK Threshold
/// compares. Lab Invert turns 8-bit bytes `v` into `255 - v` (a becomes -1 - a) and mirrors a and
/// b around neutral at 16 bits (photoshop corpus lab16 invert.psd: neutral stays neutral).
pub fn stored_channels(adj: &Adjustment, mode: ColorMode, depth: Option<SampleType>, ch: &mut [f32], gray: f32) {
    let lab = mode == ColorMode::Lab;
    match adj {
        Adjustment::Invert => {
            let mirror = if lab && depth.is_some_and(|d| d != SampleType::U8) { 256.0 / 255.0 } else { 1.0 };
            for (i, v) in ch.iter_mut().enumerate() {
                *v = if i > 0 { mirror } else { 1.0 } - v.clamp(0.0, 1.0);
            }
        }
        Adjustment::Posterize { levels } => ch.iter_mut().for_each(|v| *v = posterize(*v, *levels)),
        Adjustment::Threshold { level } => {
            // On the 8-bit grid, like the RGB threshold.
            let t = (level * 255.0).round();
            let out = if lab {
                let white = ch.first().is_some_and(|l| (l.clamp(0.0, 1.0) * 255.0).round() >= t);
                [if white { 1.0 } else { 0.0 }, 128.0 / 255.0, 128.0 / 255.0, 0.0]
            } else {
                [0.0, 0.0, 0.0, if (gray * 255.0).round() < t { 1.0 } else { 0.0 }]
            };
            ch.iter_mut().zip(out).for_each(|(v, o)| *v = o);
        }
        _ => {}
    }
}

/// Encoded sRGB → Lab normalised the way Photoshop's channels hold it: L/100, (a+128)/255,
/// (b+128)/255.
fn lab_norm(c: [f32; 3]) -> [f32; 3] {
    let l = srgb_to_lab(c);
    [l[0] / 100.0, (l[1] + 128.0) / 255.0, (l[2] + 128.0) / 255.0]
}

/// The inverse of [`lab_norm`].
fn lab_denorm(n: [f32; 3]) -> [f32; 3] {
    lab_to_srgb([n[0] * 100.0, n[1] * 255.0 - 128.0, n[2] * 255.0 - 128.0])
}

/// The Lab evaluation of a [`lab_channel_adjustment`] in a document of `depth` (no-op for any
/// other adjustment).
fn apply_lab(adj: &Adjustment, buf: &mut Buffer, depth: Option<SampleType>) {
    match adj {
        // L → 100 - L and a, b mirrored about neutral; each channel posterized (see `stored_channels`).
        Adjustment::Invert | Adjustment::Posterize { .. } => map_rgb(buf, |c| {
            let mut n = lab_norm(c);
            stored_channels(adj, ColorMode::Lab, depth, &mut n, 0.0);
            lab_denorm(n)
        }),
        Adjustment::Threshold { level } => {
            // Split on L alone (on the 8-bit grid); the result is neutral black or white.
            let t = (level * 255.0).round();
            map_rgb(buf, |c| [if (lab_norm(c)[0].clamp(0.0, 1.0) * 255.0).round() >= t { 1.0 } else { 0.0 }; 3])
        }
        Adjustment::BrightnessContrast { brightness, contrast, legacy } => {
            let legacy_curve = legacy_brightness_contrast(*brightness, *contrast);
            let curve = |v: f32| if *legacy { legacy_curve(v) } else { modern_contrast(modern_brightness(v, *brightness), *contrast).clamp(0.0, 1.0) };
            // The tone curve on L only. Added as the difference of two round trips (like
            // `tone_in_space`) so a and b, and colours a round trip moves, stay put.
            map_rgb(buf, |c| {
                let n = lab_norm(c);
                let l = curve(n[0]);
                if l == n[0] {
                    return c;
                }
                let (before, after) = (lab_denorm(n), lab_denorm([l, n[1], n[2]]));
                std::array::from_fn(|i| (c[i] + after[i] - before[i]).clamp(0.0, 1.0))
            })
        }
        Adjustment::GradientMap { stops, reverse, dither } => {
            let (w, x0, y0) = (buf.rect.width().max(1) as usize, buf.rect.x0, buf.rect.y0);
            for (i, p) in buf.px.iter_mut().enumerate() {
                if p[3] <= 0.0 {
                    continue;
                }
                // Driven by L: equal lightness maps to the same stop colour whatever the chroma.
                let l = lab_norm([p[0], p[1], p[2]])[0].clamp(0.0, 1.0);
                let mut o = gradient(stops, if *reverse { 1.0 - l } else { l });
                if *dither {
                    let d = bayer4(x0 + (i % w) as i32, y0 + (i / w) as i32) / 255.0;
                    o = o.map(|v| (v + d).clamp(0.0, 1.0));
                }
                p[..3].copy_from_slice(&o);
            }
        }
        _ => {}
    }
}

/// Float RGB curve channels with no edit must preserve samples outside the LUT domain.
/// Edited curves retain the existing spline/LUT behaviour until HDR extrapolation has an oracle.
pub fn float_tone_identity(adj: &Adjustment, depth: Option<SampleType>) -> [bool; 3] {
    match adj {
        Adjustment::Curves { master, per_channel, space: ToneSpace::Rgb, .. } if depth == Some(SampleType::F32) => {
            std::array::from_fn(|i| photocraft_doc::adjust::is_identity_curve(master) && photocraft_doc::adjust::is_identity_curve(&per_channel[i]))
        }
        _ => [false; 3],
    }
}

/// Per-channel LUTs of a Levels or Curves adjustment: each channel's record first, then the
/// master (composite) record, as Photoshop does (psd-tools levels_rgb.psd and curves_rgb.psd,
/// #975). Four rows: the three `per_channel` channels then black
/// (identity unless the space is CMYK). In Lab there is no composite record, so the master is ignored.
pub fn tone_luts(adj: &Adjustment) -> [Vec<f32>; 4] {
    tone_luts_q(adj, None)
}

/// [`tone_luts`] for samples with `quantum` steps per unit (Levels works on whole levels; see
/// [`levels_q`]).
pub fn tone_luts_q(adj: &Adjustment, quantum: Option<f32>) -> [Vec<f32>; 4] {
    let x = |k: usize| k as f32 / (LUT_SIZE - 1) as f32;
    match adj {
        Adjustment::Levels { master, per_channel, space, black } => {
            let ident = LevelsChannel::default();
            let m = if *space == ToneSpace::Lab { &ident } else { master };
            let row = |c: &LevelsChannel| (0..LUT_SIZE).map(|k| levels_q(m, levels_q(c, x(k), quantum), quantum)).collect();
            [row(&per_channel[0]), row(&per_channel[1]), row(&per_channel[2]), row(if *space == ToneSpace::Cmyk { black } else { &ident })]
        }
        Adjustment::Curves { master, per_channel, space, black, pencil } => {
            let m = if *space == ToneSpace::Lab { curve_lut(&[]) } else { curve_lut_mode(master, *pencil) };
            let row = |c: &[CurvePoint]| curve_lut_mode(c, *pencil).iter().map(|&v| lut(&m, v)).collect();
            [row(&per_channel[0]), row(&per_channel[1]), row(&per_channel[2]), row(if *space == ToneSpace::Cmyk { black } else { &[] })]
        }
        _ => std::array::from_fn(|_| (0..LUT_SIZE).map(x).collect()),
    }
}

/// [`tone_luts_q`] in a document of `depth` (`None`: unrounded). 32-bit Levels use
/// [`levels_float`], sampled on 0..1 and kept in 0..1 for LUT consumers. Float RGB rendering
/// evaluates the formula directly instead, so samples and results outside that domain survive.
pub fn tone_luts_depth(adj: &Adjustment, depth: Option<SampleType>) -> [Vec<f32>; 4] {
    match (adj, depth) {
        (Adjustment::Levels { master, per_channel, space: ToneSpace::Rgb, .. }, Some(SampleType::F32)) => {
            let x = |k: usize| k as f32 / (LUT_SIZE - 1) as f32;
            let row = |c: &LevelsChannel| (0..LUT_SIZE).map(|k| levels_float(master, levels_float(c, x(k))).clamp(0.0, 1.0)).collect();
            [row(&per_channel[0]), row(&per_channel[1]), row(&per_channel[2]), (0..LUT_SIZE).map(x).collect()]
        }
        _ => tone_luts_q(adj, depth.and_then(crate::adjustment_quantum)),
    }
}

/// Applies channel LUTs to a display-RGB colour in CMYK (ink brightness, 1 - ink) or Lab space,
/// through the conversions the document's surfaces use. The change is added as a difference of
/// two round trips, so channels a curve leaves alone (and out-of-gamut colours) stay exact.
fn tone_in_space(space: ToneSpace, luts: &[Vec<f32>; 4], c: [f32; 3]) -> [f32; 3] {
    use photocraft_color::convert::{cmyk_to_rgb, rgb_to_cmyk};
    let (before, after) = match space {
        ToneSpace::Cmyk => {
            let ink = rgb_to_cmyk(c);
            let out: [f32; 4] = std::array::from_fn(|i| 1.0 - lut(&luts[i], 1.0 - ink[i]));
            if out == ink {
                return c;
            }
            (cmyk_to_rgb(ink), cmyk_to_rgb(out))
        }
        ToneSpace::Lab => {
            let l = srgb_to_lab(c);
            let n = [l[0] / 100.0, (l[1] + 128.0) / 255.0, (l[2] + 128.0) / 255.0];
            let o: [f32; 3] = std::array::from_fn(|i| lut(&luts[i], n[i]));
            if o == n {
                return c;
            }
            let back = |v: [f32; 3]| lab_to_srgb([v[0] * 100.0, v[1] * 255.0 - 128.0, v[2] * 255.0 - 128.0]);
            (back(n), back(o))
        }
        ToneSpace::Rgb => return std::array::from_fn(|i| lut(&luts[i], c[i])),
    };
    std::array::from_fn(|i| (c[i] + after[i] - before[i]).clamp(0.0, 1.0))
}

/// Hue/Saturation's range edits as three hue-indexed tables (hue shift in degrees, saturation and
/// lightness as fractions), sampled at hue `k / (len - 1)` turns. Ranges add up where they overlap.
pub fn hue_range_tables(ranges: &[HueRange; 6]) -> [Vec<f32>; 3] {
    let at = |k: usize| {
        let deg = k as f32 / (LUT_SIZE - 1) as f32 * 360.0;
        ranges.iter().filter(|r| !r.is_neutral()).fold([0.0f32; 3], |acc, r| {
            let w = r.weight(deg);
            [acc[0] + w * r.hue, acc[1] + w * r.saturation / 100.0, acc[2] + w * r.lightness / 100.0]
        })
    };
    let rows: Vec<[f32; 3]> = (0..LUT_SIZE).map(at).collect();
    std::array::from_fn(|i| rows.iter().map(|r| r[i]).collect())
}

/// One Hue/Saturation evaluation: hue shift in degrees, saturation and lightness in -1..=1.
/// Colorize sets the hue and saturation instead of shifting them; at 0% it leaves only the
/// lightness, as in Photoshop (#2326). Ticking Colorize in the dialog starts saturation at 25%.
pub fn hue_saturation(c: [f32; 3], hue: f32, s: f32, l: f32, colorize: bool) -> [f32; 3] {
    let lighten = |v: f32| if l > 0.0 { v + (1.0 - v) * l } else { v * (1.0 + l) };
    if colorize {
        // Colorize builds the colour at the lightened (or darkened) lightness, so black with
        // Lightness +30 becomes a 30 % lightness colour, not grey (measured in Photoshop 27.11).
        let (_, _, ll) = rgb_to_hsl(c);
        return hsl_to_rgb(hue.rem_euclid(360.0) / 360.0, s.abs().min(1.0), lighten(ll));
    }
    // Photoshop applies Lightness first, then rotates the hue and scales the saturation of the
    // result: Hue +40, Saturation +25, Lightness -10 turns (240, 160, 30) into (173, 243, 0), not
    // (168, 230, 14) (Photoshop 27.11 corpus file adjustments/rgb8/hue-saturation).
    let (hh, ss, ll) = rgb_to_hsl(c.map(lighten));
    let hh = (hh + hue / 360.0).rem_euclid(1.0);
    // Raising saturation divides by 1 - s, lowering it multiplies by 1 + s: +50 % doubles it
    // (measured in Photoshop 27.11); +100 % saturates any colour that has some.
    let ss = if s > 0.0 { ss / (1.0 - s).max(1.0 / 512.0) } else { ss * (1.0 + s) }.clamp(0.0, 1.0);
    hsl_to_rgb(hh, ss, ll)
}

/// A colour range's Lightness, applied after the master edits. It moves each channel towards the
/// pixel's brightest channel when raised and towards its darkest when lowered, not towards white
/// or black: Blues +50 turns (0, 29, 115) into (57, 71, 115), Blues -50 turns (102, 128, 204) into
/// (102, 115, 153) (measured in Photoshop 27.11).
pub fn range_lightness(c: [f32; 3], l: f32) -> [f32; 3] {
    let l = l.clamp(-1.0, 1.0);
    let target = if l > 0.0 { c[0].max(c[1]).max(c[2]) } else { c[0].min(c[1]).min(c[2]) };
    c.map(|v| v + (target - v) * l.abs())
}

/// Black & White grey value. Each colour is the sum of a grey part (its smallest channel), a
/// secondary-colour part (yellow, cyan or magenta: middle − smallest) and a primary part (red,
/// green or blue: largest − middle); the sliders (percent, Photoshop defaults 40 60 40 60 20 80 for
/// reds, yellows, greens, cyans, blues, magentas) weight the two colour parts.
pub fn black_white_gray(c: [f32; 3], weights: &[f32; 6]) -> f32 {
    let (r, g, b) = (c[0], c[1], c[2]);
    let (max, min) = (r.max(g).max(b), r.min(g).min(b));
    let mid = r + g + b - max - min;
    // Primary: the largest channel; secondary: the two largest channels together.
    let primary = if r >= g && r >= b {
        0
    } else if g >= b {
        2
    } else {
        4
    };
    let secondary = if b <= r && b <= g {
        1 // red + green = yellow
    } else if r <= g {
        3 // green + blue = cyan
    } else {
        5 // red + blue = magenta
    };
    (min + (mid - min) * weights[secondary] / 100.0 + (max - mid) * weights[primary] / 100.0).clamp(0.0, 1.0)
}

/// Ordered-dither offset in -0.5..0.5 (4×4 Bayer matrix) for document pixel (x, y).
pub fn bayer4(x: i32, y: i32) -> f32 {
    const M: [f32; 16] = [0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0];
    (M[((y & 3) * 4 + (x & 3)) as usize] + 0.5) / 16.0 - 0.5
}

/// Samples a 3D LUT (`n`³ RGB triplets, red fastest) at `c`, trilinear or tetrahedral.
pub fn lut3d_sample(lut: &[f32], n: usize, c: [f32; 3], tetrahedral: bool) -> [f32; 3] {
    // A degenerate or short table can't be sampled: leave the colour alone.
    if n < 2 || n.checked_pow(3).and_then(|c| c.checked_mul(3)).is_none_or(|len| lut.len() < len) {
        return c;
    }
    let m = (n - 1) as f32;
    let pos = c.map(|v| v.clamp(0.0, 1.0) * m);
    let i0 = pos.map(|p| (p.floor() as usize).min(n - 2));
    let f: [f32; 3] = std::array::from_fn(|k| pos[k] - i0[k] as f32);
    let at = |dr: usize, dg: usize, db: usize| -> [f32; 3] {
        let idx = (((i0[2] + db) * n + i0[1] + dg) * n + i0[0] + dr) * 3;
        [lut[idx], lut[idx + 1], lut[idx + 2]]
    };
    let mix = |w: &[(f32, [f32; 3])]| -> [f32; 3] { std::array::from_fn(|k| w.iter().map(|(a, v)| a * v[k]).sum()) };
    let (fr, fg, fb) = (f[0], f[1], f[2]);
    if tetrahedral {
        let (c000, c111) = (at(0, 0, 0), at(1, 1, 1));
        if fr > fg {
            if fg > fb {
                mix(&[(1.0 - fr, c000), (fr - fg, at(1, 0, 0)), (fg - fb, at(1, 1, 0)), (fb, c111)])
            } else if fr > fb {
                mix(&[(1.0 - fr, c000), (fr - fb, at(1, 0, 0)), (fb - fg, at(1, 0, 1)), (fg, c111)])
            } else {
                mix(&[(1.0 - fb, c000), (fb - fr, at(0, 0, 1)), (fr - fg, at(1, 0, 1)), (fg, c111)])
            }
        } else if fb > fg {
            mix(&[(1.0 - fb, c000), (fb - fg, at(0, 0, 1)), (fg - fr, at(0, 1, 1)), (fr, c111)])
        } else if fb > fr {
            mix(&[(1.0 - fg, c000), (fg - fb, at(0, 1, 0)), (fb - fr, at(0, 1, 1)), (fr, c111)])
        } else {
            mix(&[(1.0 - fg, c000), (fg - fr, at(0, 1, 0)), (fr - fb, at(1, 1, 0)), (fb, c111)])
        }
    } else {
        let lerp = |a: [f32; 3], b: [f32; 3], t: f32| -> [f32; 3] { std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t) };
        let c00 = lerp(at(0, 0, 0), at(1, 0, 0), fr);
        let c10 = lerp(at(0, 1, 0), at(1, 1, 0), fr);
        let c01 = lerp(at(0, 0, 1), at(1, 0, 1), fr);
        let c11 = lerp(at(0, 1, 1), at(1, 1, 1), fr);
        lerp(lerp(c00, c10, fg), lerp(c01, c11, fg), fb)
    }
}

/// How far one Color Balance slider at ±100 moves the black or white point (Photoshop 27.11:
/// shadows −100 maps 99.6 of 255 to black, highlights +100 maps 155.5 to white).
const BALANCE_LEVELS: f32 = 0.39;

/// Color Balance on channel `ch` (0 red, 1 green, 2 blue) of `v` (0..1); `tones` are the
/// shadows, midtones and highlights slider triples (cyan–red, magenta–green, yellow–blue,
/// −100..100).
///
/// Reverse-engineered from Photoshop 27.11 (a gray ramp and pure colours through each slider at
/// ±25, ±50 and ±100, with Preserve Luminosity on and off): every slider is a levels-and-gamma
/// curve per channel, exact within 1 level for a single slider. With `a` = slider / 100:
/// - Preserve Luminosity off: a slider only moves its own channel. Midtones is gamma 2^−a.
///   Shadows is gamma 2^(−a/2) after a black point at 0.39·|a| when lowered; highlights is
///   gamma 2^(−a/2) after a white point at 1 − 0.39·a when raised.
/// - Preserve Luminosity on: midtones uses gamma 2^((max(a) + min(a))/2 − a_ch),
///   preserving the one-slider behaviour without summing offsets when multiple sliders move.
///   Shadows and highlights go by the differences between a tone's three sliders:
///   each channel gets a plain black point at 0.39·(max − its own) for shadows and a plain white
///   point at 1 − 0.39·(its own − min) for highlights (so one raised shadows slider darkens the
///   other two channels, and two raised together darken only the third).
///
/// Tones apply highlights, shadows, then midtones (measured with two and three tones on one
/// channel: within 1 level).
pub fn color_balance_channel(tones: [[f32; 3]; 3], preserve_luminosity: bool, ch: usize, v: f32) -> f32 {
    let [shadows, midtones, highlights] = tones;
    let black = |v: f32, b: f32| ((v - b) / (1.0 - b)).max(0.0);
    let white = |v: f32, w: f32| (v / w).min(1.0);
    let mut v = v.clamp(0.0, 1.0);
    for (tone, sliders) in [(2, highlights), (0, shadows), (1, midtones)] {
        let a: [f32; 3] = sliders.map(|s| (s / 100.0).clamp(-1.0, 1.0));
        if preserve_luminosity {
            let own = a.get(ch).copied().unwrap_or(0.0);
            let (lo, hi) = a.iter().fold((f32::MAX, f32::MIN), |m, x| (m.0.min(*x), m.1.max(*x)));
            v = match tone {
                0 => black(v, BALANCE_LEVELS * (hi - own)),
                1 => v.powf(2f32.powf((hi + lo) / 2.0 - own)),
                _ => white(v, 1.0 - BALANCE_LEVELS * (own - lo)),
            };
            continue;
        }
        for (src, slider) in sliders.iter().enumerate() {
            let a = (slider / 100.0).clamp(-1.0, 1.0);
            if a == 0.0 || src != ch {
                continue;
            }
            v = match tone {
                1 => v.powf(2f32.powf(-a)),
                0 if a < 0.0 => black(v, BALANCE_LEVELS * -a).powf(2f32.powf(-a / 2.0)),
                0 => v.powf(2f32.powf(-a / 2.0)),
                2 if a > 0.0 => white(v, 1.0 - BALANCE_LEVELS * a).powf(2f32.powf(-a / 2.0)),
                _ => v.powf(2f32.powf(-a / 2.0)),
            };
        }
    }
    v
}

/// How much a colour belongs to each Selective Color range: reds, yellows, greens, cyans,
/// blues, magentas (by chroma within the hue sector), whites, neutrals and blacks (by lightness).
pub fn selective_color_weights(c: [f32; 3]) -> [f32; 9] {
    let max = c[0].max(c[1]).max(c[2]);
    let min = c[0].min(c[1]).min(c[2]);
    let mid = c[0] + c[1] + c[2] - max - min;
    let top = |i: usize| if c[i] >= max { max - mid } else { 0.0 };
    let bottom = |i: usize| if c[i] <= min { mid - min } else { 0.0 };
    [
        top(0),
        bottom(2),
        top(1),
        bottom(0),
        top(2),
        bottom(1),
        ((min - 0.5) * 2.0).max(0.0),
        (1.0 - (max - 0.5).abs() - (min - 0.5).abs()).clamp(0.0, 1.0),
        ((0.5 - max) * 2.0).max(0.0),
    ]
}

/// Photoshop Selective Color: each range shifts the cyan, magenta and yellow ink (1 − R, G, B) by
/// its percentage, plus its black percentage on every ink, weighted by how much the colour belongs
/// to the range. Relative mode scales the shift by the ink already present (so it can't tint pure
/// white). Absolute mode scales it by the room the ink has: adding ink by min(1, 2 × (1 − ink)),
/// removing it by min(1, 2 × ink), so Neutrals Cyan +50 darkens red 64 by 32, not 64 (all 72
/// absolute range × ink cases within 1 level of Photoshop 27.11).
pub fn selective_color(c: [f32; 3], relative: bool, adj: &[[f32; 4]; 9]) -> [f32; 3] {
    let w = selective_color_weights(c);
    let mut delta = [0.0f32; 3];
    for (r, wr) in w.iter().enumerate() {
        if *wr <= 0.0 {
            continue;
        }
        let a = adj[r].map(|v| v / 100.0);
        for i in 0..3 {
            let ink = 1.0 - c[i];
            let d = a[i] + a[3];
            let d = if relative {
                d * ink
            } else if d > 0.0 {
                d * (2.0 * (1.0 - ink)).min(1.0)
            } else {
                d * (2.0 * ink).min(1.0)
            };
            delta[i] += d * wr;
        }
    }
    std::array::from_fn(|i| (c[i] - delta[i]).clamp(0.0, 1.0))
}

const LUT_SIZE: usize = 4096;

fn map_rgb(buf: &mut Buffer, f: impl Fn([f32; 3]) -> [f32; 3]) {
    for p in &mut buf.px {
        if p[3] <= 0.0 {
            continue;
        }
        let r = f([p[0], p[1], p[2]]);
        p[0] = r[0];
        p[1] = r[1];
        p[2] = r[2];
    }
}

#[inline]
fn lut(table: &[f32], v: f32) -> f32 {
    let x = v.clamp(0.0, 1.0) * (table.len() - 1) as f32;
    let i = x.floor() as usize;
    let j = (i + 1).min(table.len() - 1);
    let f = x - i as f32;
    table[i] * (1.0 - f) + table[j] * f
}

/// Photo Filter's linear-light RGB matrix. Photoshop multiplies the pixel's D50 XYZ, relative to
/// white, by the filter colour's (mixed in by `density`): a fixed linear map, so it is folded into
/// one matrix. With `normalize_y` the map keeps luminance Y (32-bit Preserve Luminosity; 8/16-bit
/// documents instead restore the encoded luminosity like the Luminosity blend mode). Fitted on the
/// photoshop corpus `photo-filter.psd`: rgb16 within 0.6/255 everywhere.
pub fn photo_filter_matrix(color: [f32; 3], density: f32, normalize_y: bool) -> [[f32; 3]; 3] {
    let d = if density.is_finite() { density.clamp(0.0, 1.0) } else { 0.0 };
    let xyz = mat3_mul(&SRGB_TO_XYZ_D50, color.map(|v| srgb_to_linear(v.clamp(0.0, 1.0))));
    let mut s: [f32; 3] = std::array::from_fn(|i| 1.0 - d + d * xyz[i] / D50[i]);
    if normalize_y && s[1] > 1e-6 {
        let y = s[1];
        s = s.map(|v| v / y);
    }
    // XYZ_D50_TO_SRGB · diag(s) · SRGB_TO_XYZ_D50
    std::array::from_fn(|r| std::array::from_fn(|c| (0..3).map(|k| XYZ_D50_TO_SRGB[r][k] * s[k] * SRGB_TO_XYZ_D50[k][c]).sum()))
}

/// Photoshop's Vibrance on one pixel (`v`, `s`: the sliders / 100; `hi`: the sample ceiling).
///
/// Measured black-box against Photoshop 25.4 on 7,400-colour charts (8- and 16-bit; sRGB, Adobe RGB
/// and ProPhoto documents). Both sliders work in linear light through the document's tone curve,
/// Vibrance first, then Saturation:
/// - Saturation moves each colour away from (or towards) the grey `0.288 R + 0.712 G` (blue has no
///   weight, in every profile) by `1 + s`, clipping each channel: within 0.07/255 at every `s`.
/// - Vibrance is an HSV change in linear light that keeps the hue: see [`vibrance_lin`].
pub fn vibrance_px(c: [f32; 3], v: f32, s: f32, transfer: Transfer, hi: f32) -> [f32; 3] {
    let mut l = c.map(|x| transfer.decode(x));
    if v != 0.0 {
        l = vibrance_lin(l, v);
    }
    if s != 0.0 {
        let g = VIBRANCE_GREY[0] * l[0] + VIBRANCE_GREY[1] * l[1];
        l = l.map(|x| (g + (1.0 + s) * (x - g)).clamp(0.0, hi));
    }
    l.map(|x| transfer.encode(x.clamp(0.0, hi)))
}

/// Weights of the grey Photoshop's Vibrance › Saturation desaturates to (R, G; blue has none).
pub const VIBRANCE_GREY: [f32; 2] = [0.288, 0.712];

/// Positive Vibrance's saturation boost, fitted to the measurements (see [`vibrance_lin`]).
pub const VIBRANCE_BOOST: [f32; 7] = [0.5001, 0.0887, 0.4374, 0.7573, -0.8113, 0.5154, 2.8026];

/// The Vibrance slider (`v` in [-1, 1]) on a linear-light pixel. With `t = 1 - min/max` (HSV
/// saturation) and `V = max`:
/// - below 0 (`a = -v`): `t' = t (1 - a/4) (1 - a + a t (1 + t) / 2)` and
///   `V' = V - a t (1 - t) (2 - t + t²) V (1 - V)`; exact (0.2/255 mean) except in the deepest
///   shadows (`V` under 0.05), which Photoshop desaturates less.
/// - above 0: `V' = V + v/4 t (1 - t) (2 - t + t²) V (1 - V)` (exact) and a saturation boost that
///   grows as `t` and `V` fall, damped on reds and skin (HSV hue 0–30°, fading out by 45° and in
///   from 300°): a fit, 0.4 (+10) to 5 (+100) /255 off on average.
pub fn vibrance_lin(c: [f32; 3], v: f32) -> [f32; 3] {
    let mx = c[0].max(c[1]).max(c[2]);
    let mn = c[0].min(c[1]).min(c[2]);
    if mx <= 0.0 || mx - mn <= 1e-9 {
        return c;
    }
    let t = 1.0 - mn / mx;
    let shape = t * (1.0 - t) * (2.0 - t + t * t) * mx * (1.0 - mx).max(0.0);
    let (t2, v2) = if v < 0.0 {
        let a = -v;
        (t * (1.0 - a / 4.0) * (1.0 - a + a * t * (1.0 + t) / 2.0), mx - a * shape)
    } else {
        let [c0, c1, p, q, r0, ra, rt] = VIBRANCE_BOOST;
        let w = vibrance_skin(hsv_hue(c, mx, mn));
        let a = v * (1.0 - w) + 0.64 * v.powf(1.6) * w;
        let boost = (c0 * a + c1 * a * a) * t.powf(p) * (1.0 - t).powf(q) * ((r0 + ra * a + rt * (1.0 - t)) * (1.0 - mx).max(0.0)).exp();
        ((t * (1.0 + boost)).min(1.0), mx + v / 4.0 * shape)
    };
    let m2 = v2 * (1.0 - t2);
    let k = (v2 - m2) / (mx - mn);
    c.map(|x| m2 + (x - mn) * k)
}

/// How much positive Vibrance treats a hue (degrees) as skin: 1 on 0–30°, fading to 0 by 45° and
/// in again from 300°.
fn vibrance_skin(h: f32) -> f32 {
    if h <= 30.0 {
        1.0
    } else if h < 45.0 {
        (45.0 - h) / 15.0
    } else if h < 300.0 {
        0.0
    } else {
        (h - 300.0) / 60.0
    }
}

/// HSV hue in degrees of `c` (its max `mx` and min `mn` given, `mx > mn`).
fn hsv_hue(c: [f32; 3], mx: f32, mn: f32) -> f32 {
    let d = mx - mn;
    let h = if mx == c[0] {
        ((c[1] - c[2]) / d).rem_euclid(6.0)
    } else if mx == c[1] {
        (c[2] - c[0]) / d + 2.0
    } else {
        (c[0] - c[1]) / d + 4.0
    };
    h * 60.0
}

/// Photoshop posterize: `n` equal input bins over 0..=255, output levels
/// `floor(k * 255 / (n - 1))` (exact on the corpus for 3, 7, 13 and 21 levels).
/// Modern Brightness curve (one channel, `brightness` in [-150, 150]). A line of slope
/// `s = 1.375^(b/50)` from the origin, rolled off to (1, 1) by a `v^P` white-anchor term. Pins pure
/// black and white; `b = 0` is the identity. Fit to Photoshop ground truth (see modern B/C above).
pub fn modern_brightness(v: f32, brightness: f32) -> f32 {
    if brightness == 0.0 {
        return v;
    }
    let s = 1.375f32.powf(brightness / 50.0);
    // Exponent of the white-anchor term grows with |b|; shapes the roll-off toward (1,1).
    let p = if brightness >= 0.0 { (4.5 - 0.013 * brightness).max(2.0) } else { 5.0 - 0.072 * brightness };
    let v = v.clamp(0.0, 1.0);
    (s * v + (1.0 - s) * v.powf(p)).clamp(0.0, 1.0)
}

/// Modern Contrast curve (one channel, `contrast` in [-50, 100]). A symmetric cubic-Hermite S-curve
/// pivoting at 0.5: endpoint slope `1 - c/128`, centre slope `1 + c/128`. Near-exact vs Photoshop
/// (≤0.5/255). `c = 0` is the identity; pins 0, 0.5 and 1.
pub fn modern_contrast(v: f32, contrast: f32) -> f32 {
    if contrast == 0.0 {
        return v;
    }
    let k = contrast / 128.0;
    let (end, mid) = (1.0 - k, 1.0 + k);
    // Cubic Hermite on [0, 0.5]: pinned (0,0) slope `end`, (0.5,0.5) slope `mid`. `t` in [0,1].
    let half = |t: f32| end * 0.5 * (t * t * t - 2.0 * t * t + t) + (-2.0 * t * t * t + 3.0 * t * t) * 0.5 + mid * 0.5 * (t * t * t - t * t);
    let v = v.clamp(0.0, 1.0);
    if v <= 0.5 { half(v / 0.5) } else { 1.0 - half((1.0 - v) / 0.5) }
}

pub fn posterize(v: f32, levels: u32) -> f32 {
    let n = levels.clamp(2, 255) as f32;
    let x = (v.clamp(0.0, 1.0) * 255.0).round();
    let bin = (x * n / 256.0).floor().min(n - 1.0);
    (bin * 255.0 / (n - 1.0)).floor() / 255.0
}

pub fn levels(ch: &LevelsChannel, v: f32) -> f32 {
    levels_q(ch, v, None)
}

/// [`levels`] on samples with `quantum` steps per unit (8-bit: 255). Photoshop works on whole
/// levels: the input is a level, and the input-range stretch is rounded (half up) to a whole
/// level before the midtone gamma, so a 44..214 range maps 45, 46, 47, 48 → 1.5, 3, 4.5, 6 →
/// 2, 3, 5, 6 (psd-tools levels_grayscale, where the gamma toe magnifies the steps). In a LUT
/// the result is a staircase that is exact at every level.
pub fn levels_q(ch: &LevelsChannel, v: f32, quantum: Option<f32>) -> f32 {
    let range = (ch.in_white - ch.in_black).max(1e-6);
    let u = match quantum.filter(|q| q.is_finite() && *q >= 1.0) {
        Some(q) => {
            let v = (v * q).round() / q;
            (((v - ch.in_black) / range).clamp(0.0, 1.0) * q + 0.5 + 1e-3).floor() / q
        }
        None => ((v - ch.in_black) / range).clamp(0.0, 1.0),
    };
    let g = ch.gamma.max(0.01);
    let t = if g > 1.0 {
        // A midtone gamma > 1 lifts shadows, and a pure `u^(1/g)` has infinite slope at black —
        // which Photoshop bounds, giving a soft shadow toe (initial slope ≈ 2^gamma, verified against
        // the real app). Soft-min (p-norm) of the power curve with that line; reduces to the exact
        // power curve in the body, so highlights are unchanged. gamma <= 1 is untouched. Slope and
        // sharpness fitted on whole-level input (`levels_q`): a 0..255 ramp at gamma 2.0 and
        // psd-tools levels_grayscale's gamma 1.78 (within one level of Photoshop on both).
        let power = u.powf(1.0 / g);
        let line = 0.93 * 2.0f32.powf(g) * u;
        if power <= 1e-6 || line <= 1e-6 {
            power.min(line)
        } else {
            let p = 10.0; // p-norm sharpness, fit to Photoshop ground truth
            (power.powf(-p) + line.powf(-p)).powf(-1.0 / p)
        }
    } else {
        u.powf(1.0 / g)
    };
    ch.out_black + t * (ch.out_white - ch.out_black)
}

/// [`levels`] in a 32-bit document. Neither the input is clipped to the input range nor the result
/// to the output range there, and the midtone gamma is a plain power curve, mirrored below the
/// black point (the oracle corpus' rgb32 and gray32 levels.psd: within 0.5/255, 10/255 off with
/// the clipped curve).
pub fn levels_float(ch: &LevelsChannel, v: f32) -> f32 {
    let t = (v - ch.in_black) / (ch.in_white - ch.in_black).max(1e-6);
    let t = if ch.gamma == 1.0 { t } else { t.signum() * t.abs().powf(1.0 / ch.gamma.max(0.01)) };
    ch.out_black + t * (ch.out_white - ch.out_black)
}

/// The LUT of one Curves curve: [`curve_map_lut`] for a pencil-drawn map, else [`curve_lut`].
pub fn curve_lut_mode(points: &[CurvePoint], pencil: bool) -> Vec<f32> {
    if pencil { curve_map_lut(points) } else { curve_lut(points) }
}

/// A pencil-drawn Curves map (see [`photocraft_doc::adjust::curve_map`]): straight segments
/// between its entries, constant outside the first/last, clamped to 0..=1, sampled into a LUT.
pub fn curve_map_lut(points: &[CurvePoint]) -> Vec<f32> {
    let mut pts: Vec<(f32, f32)> = points.iter().filter(|p| p.input.is_finite() && p.output.is_finite()).map(|p| (p.input, p.output)).collect();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    pts.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-6);
    let ident = |k: usize| k as f32 / (LUT_SIZE - 1) as f32;
    let (Some(&first), Some(&last)) = (pts.first(), pts.last()) else { return (0..LUT_SIZE).map(ident).collect() };
    let mut seg = 0;
    (0..LUT_SIZE)
        .map(|k| {
            let xv = ident(k);
            if xv <= first.0 {
                return first.1.clamp(0.0, 1.0);
            }
            if xv >= last.0 {
                return last.1.clamp(0.0, 1.0);
            }
            while pts.get(seg + 1).is_some_and(|p| p.0 < xv) {
                seg += 1;
            }
            let (Some(&(x0, y0)), Some(&(x1, y1))) = (pts.get(seg), pts.get(seg + 1)) else { return last.1.clamp(0.0, 1.0) };
            let t = if x1 > x0 { (xv - x0) / (x1 - x0) } else { 0.0 };
            (y0 + (y1 - y0) * t).clamp(0.0, 1.0)
        })
        .collect()
}

/// Photoshop curve: a natural cubic spline through the points (it may
/// overshoot between points, as Photoshop's does), constant outside the
/// first/last point, clamped to 0..=1, sampled into a LUT.
pub fn curve_lut(points: &[CurvePoint]) -> Vec<f32> {
    let mut pts: Vec<(f32, f32)> = points.iter().map(|p| (p.input, p.output)).collect();
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    pts.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-6);
    if pts.len() < 2 {
        return (0..LUT_SIZE).map(|i| i as f32 / (LUT_SIZE - 1) as f32).collect();
    }
    let n = pts.len();
    // Second derivatives of the natural spline (tridiagonal solve).
    let mut m2 = vec![0.0f64; n];
    if n > 2 {
        let x: Vec<f64> = pts.iter().map(|p| f64::from(p.0)).collect();
        let y: Vec<f64> = pts.iter().map(|p| f64::from(p.1)).collect();
        let mut c = vec![0.0f64; n];
        let mut d = vec![0.0f64; n];
        for i in 1..n - 1 {
            let (h0, h1) = (x[i] - x[i - 1], x[i + 1] - x[i]);
            let a = h0 / 6.0;
            let b = (h0 + h1) / 3.0;
            let cc = h1 / 6.0;
            let r = (y[i + 1] - y[i]) / h1 - (y[i] - y[i - 1]) / h0;
            let denom = b - a * c[i - 1];
            c[i] = cc / denom;
            d[i] = (r - a * d[i - 1]) / denom;
        }
        for i in (1..n - 1).rev() {
            m2[i] = d[i] - c[i] * m2[i + 1];
        }
    }
    (0..LUT_SIZE)
        .map(|k| {
            let xv = k as f32 / (LUT_SIZE - 1) as f32;
            if xv <= pts[0].0 {
                return pts[0].1.clamp(0.0, 1.0);
            }
            if xv >= pts[n - 1].0 {
                return pts[n - 1].1.clamp(0.0, 1.0);
            }
            let i = pts.windows(2).position(|w| xv <= w[1].0).unwrap_or(n - 2);
            let (x0, y0) = (f64::from(pts[i].0), f64::from(pts[i].1));
            let (x1, y1) = (f64::from(pts[i + 1].0), f64::from(pts[i + 1].1));
            let h = x1 - x0;
            let t = f64::from(xv);
            let a = (x1 - t) / h;
            let b = (t - x0) / h;
            let y = a * y0 + b * y1 + ((a * a * a - a) * m2[i] + (b * b * b - b) * m2[i + 1]) * h * h / 6.0;
            (y as f32).clamp(0.0, 1.0)
        })
        .collect()
}

/// A layer-style contour curve: [`curve_lut`], except that the curve bends sharply at corner
/// points (`corners[i]` true; Photoshop's Contour Editor "Corner"): each run between corners is
/// its own natural spline, so all-corner points make a polyline. Missing flags are smooth.
pub fn contour_curve_lut(points: &[CurvePoint], corners: &[bool]) -> Vec<f32> {
    let mut pts: Vec<(CurvePoint, bool)> = points.iter().enumerate().map(|(i, p)| (p.clone(), corners.get(i).copied().unwrap_or(false))).collect();
    if !pts.iter().any(|p| p.1) {
        return curve_lut(points);
    }
    pts.sort_by(|a, b| a.0.input.partial_cmp(&b.0.input).unwrap_or(std::cmp::Ordering::Equal));
    pts.dedup_by(|a, b| (a.0.input - b.0.input).abs() < 1e-6);
    let mut out: Vec<f32> = Vec::new();
    let mut start = 0usize;
    let last = pts.len().saturating_sub(1);
    for i in 1..=last {
        let corner = pts.get(i).is_some_and(|p| p.1);
        if !(corner || i == last) {
            continue;
        }
        let Some(seg) = pts.get(start..=i) else { break };
        let from = seg.first().map_or(0.0, |p| p.0.input);
        let lut = curve_lut(&seg.iter().map(|p| p.0.clone()).collect::<Vec<_>>());
        if out.is_empty() {
            out = lut;
        } else {
            for (k, (o, v)) in out.iter_mut().zip(lut).enumerate() {
                if k as f32 / (LUT_SIZE - 1) as f32 >= from {
                    *o = v;
                }
            }
        }
        start = i;
    }
    if out.is_empty() { curve_lut(points) } else { out }
}

fn gradient(stops: &[(f32, [f32; 3])], t: f32) -> [f32; 3] {
    match stops {
        [] => [t; 3],
        [s] => s.1,
        _ => {
            if t <= stops[0].0 {
                return stops[0].1;
            }
            for w in stops.windows(2) {
                if t <= w[1].0 {
                    let k = if w[1].0 > w[0].0 { (t - w[0].0) / (w[1].0 - w[0].0) } else { 0.0 };
                    return std::array::from_fn(|i| w[0].1[i] + (w[1].1[i] - w[0].1[i]) * k);
                }
            }
            stops[stops.len() - 1].1
        }
    }
}

pub fn rgb_to_hsl(c: [f32; 3]) -> (f32, f32, f32) {
    let max = c[0].max(c[1]).max(c[2]);
    let min = c[0].min(c[1]).min(c[2]);
    let l = (max + min) / 2.0;
    if (max - min).abs() < 1e-7 {
        return (0.0, 0.0, l);
    }
    let d = max - min;
    let s = if l > 0.5 { d / (2.0 - max - min) } else { d / (max + min) };
    let h = if max == c[0] {
        ((c[1] - c[2]) / d).rem_euclid(6.0)
    } else if max == c[1] {
        (c[2] - c[0]) / d + 2.0
    } else {
        (c[0] - c[1]) / d + 4.0
    };
    (h / 6.0, s, l)
}

pub fn hsl_to_rgb(h: f32, s: f32, l: f32) -> [f32; 3] {
    if s <= 0.0 {
        return [l; 3];
    }
    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;
    let f = |mut t: f32| {
        t = t.rem_euclid(1.0);
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };
    [f(h + 1.0 / 3.0), f(h), f(h - 1.0 / 3.0)]
}

#[cfg(test)]
mod lookup_tests {
    use super::*;

    fn identity(n: usize) -> Vec<f32> {
        let m = (n - 1) as f32;
        (0..n * n * n).flat_map(|i| [(i % n) as f32 / m, ((i / n) % n) as f32 / m, (i / (n * n)) as f32 / m]).collect()
    }

    #[test]
    fn identity_lut_is_identity_both_interpolations() {
        let t = identity(5);
        for c in [[0.1, 0.5, 0.9], [0.33, 0.77, 0.0], [1.0, 1.0, 1.0]] {
            for tet in [false, true] {
                let o = lut3d_sample(&t, 5, c, tet);
                for k in 0..3 {
                    assert!((o[k] - c[k]).abs() < 1e-5, "{c:?} {tet} {o:?}");
                }
            }
        }
    }

    /// Photoshop 27.11, measured: absolute Selective Color on greys (red channel after one range's
    /// cyan change).
    #[test]
    fn absolute_selective_color_matches_photoshop() {
        // (range index, cyan %, grey, Photoshop's red)
        let measured = [
            (7, 50.0, 64u8, 32u8),
            (7, 50.0, 192, 129),
            (7, -50.0, 192, 223),
            (7, -50.0, 64, 128),
            (6, -50.0, 192, 224),
            (6, -50.0, 240, 253),
            (8, 50.0, 64, 32),
            (8, 50.0, 16, 2),
        ];
        for (range, cyan, grey, want) in measured {
            let mut adj = [[0.0f32; 4]; 9];
            adj[range][0] = cyan;
            let g = grey as f32 / 255.0;
            let got = selective_color([g; 3], false, &adj)[0] * 255.0;
            assert!((got - want as f32).abs() <= 1.5, "range {range} cyan {cyan} grey {grey}: {got} vs {want}");
        }
    }

    #[test]
    fn selective_color_zero_is_identity_and_ranges_are_local() {
        let zero = [[0.0f32; 4]; 9];
        for c in [[0.9, 0.1, 0.1], [0.5, 0.5, 0.5], [0.2, 0.4, 0.8]] {
            assert_eq!(selective_color(c, true, &zero), c);
        }
        // +100 % cyan on reds only: red loses red, blue is untouched.
        let mut a = zero;
        a[0][0] = 100.0;
        let red = selective_color([0.9, 0.1, 0.1], false, &a);
        assert!(red[0] < 0.2, "{red:?}");
        assert_eq!(selective_color([0.1, 0.1, 0.9], false, &a), [0.1, 0.1, 0.9]);
        // Relative mode can't tint pure white; absolute can (via whites).
        let mut w = zero;
        w[6] = [0.0, 0.0, 50.0, 0.0];
        assert_eq!(selective_color([1.0; 3], true, &w), [1.0; 3]);
        assert!(selective_color([1.0; 3], false, &w)[2] < 0.6);
    }

    #[test]
    fn dither_offsets_are_balanced() {
        let sum: f32 = (0..4).flat_map(|y| (0..4).map(move |x| bayer4(x, y))).sum();
        assert!(sum.abs() < 1e-5);
        assert_eq!(bayer4(-4, -4), bayer4(0, 0));
    }
}

#[cfg(test)]
mod tone_tests {
    use super::*;
    use photocraft_geom::Rect;

    fn ramp() -> Buffer {
        let rect = Rect::new(0, 0, 8, 8);
        let px = (0..64).map(|i| [((i % 8) as f32) / 7.0, ((i / 8) as f32) / 7.0, 0.3, 1.0]).collect();
        Buffer { rect, px }
    }

    /// Photoshop 27.11, measured: Brightness/Contrast with Use Legacy on, over a gray ramp
    /// (x = 0, 16, …, 240, 255), one row per (brightness, contrast).
    #[test]
    fn legacy_brightness_contrast_matches_photoshop() {
        const X: [u8; 17] = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255];
        let measured: [(f32, f32, [u8; 17]); 7] = [
            (40.0, 0.0, [40, 56, 72, 88, 104, 120, 136, 152, 168, 184, 200, 216, 232, 248, 255, 255, 255]),
            (-40.0, 0.0, [0, 0, 0, 8, 24, 40, 56, 72, 88, 104, 120, 136, 152, 168, 184, 200, 215]),
            (0.0, 30.0, [0, 0, 0, 14, 37, 60, 82, 105, 128, 151, 174, 196, 219, 242, 255, 255, 255]),
            (0.0, -30.0, [38, 49, 61, 72, 83, 94, 106, 117, 128, 139, 150, 162, 173, 184, 195, 207, 217]),
            (40.0, 30.0, [3, 25, 48, 71, 94, 117, 139, 162, 185, 208, 231, 253, 255, 255, 255, 255, 255]),
            (100.0, 0.0, [100, 116, 132, 148, 164, 180, 196, 212, 228, 244, 255, 255, 255, 255, 255, 255, 255]),
            (0.0, 80.0, [0, 0, 0, 0, 0, 0, 0, 48, 128, 208, 255, 255, 255, 255, 255, 255, 255]),
        ];
        for (brightness, contrast, want) in measured {
            let rect = Rect::new(0, 0, 17, 1);
            let px = X.iter().map(|x| [f32::from(*x) / 255.0; 4]).map(|p| [p[0], p[1], p[2], 1.0]).collect();
            let mut b = Buffer { rect, px };
            apply(&Adjustment::BrightnessContrast { brightness, contrast, legacy: true }, &mut b);
            for (i, (p, w)) in b.px.iter().zip(want).enumerate() {
                let got = (p[0] * 255.0).round() as i32;
                assert!((got - i32::from(w)).abs() <= 1, "b {brightness} c {contrast} at {}: {got} vs Photoshop {w}", X[i]);
            }
        }
    }

    #[test]
    fn color_lookup_with_an_overflowing_stored_size_is_the_identity() {
        // A stored document's size is untrusted: (2²²)³ × 3 overflows, which once wrapped past
        // the table-length guard and indexed out of bounds.
        for size in [1 << 22, u32::MAX] {
            let a = Adjustment::ColorLookup { name: "x".into(), lut: Some(std::sync::Arc::new(vec![0.5; 24])), size, tetrahedral: false, dither: false };
            let mut b = ramp();
            apply(&a, &mut b);
            assert_eq!(b.px, ramp().px, "size {size}");
        }
    }

    /// Photoshop 27.11, measured: Color Balance with one cyan–red slider moved, over a gray ramp
    /// (x = 0, 16, …, 240, 255); red and green out (blue follows green on gray).
    #[test]
    fn color_balance_matches_photoshop() {
        const X: [u8; 17] = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255];
        type Row = (&'static str, f32, bool, [u8; 17], [u8; 17]);
        let measured: [Row; 6] = [
            ("midtones", 50.0, false, [0, 36, 59, 78, 96, 112, 128, 142, 156, 170, 183, 196, 209, 221, 233, 244, 255], X),
            ("shadows", -100.0, false, [0, 0, 0, 0, 0, 0, 0, 7, 23, 43, 67, 93, 122, 154, 186, 221, 255], X),
            ("highlights", 100.0, false, [0, 51, 84, 111, 136, 160, 182, 202, 223, 242, 255, 255, 255, 255, 255, 255, 255], X),
            ("shadows", 50.0, true, X, [0, 0, 0, 0, 17, 37, 57, 77, 97, 117, 137, 157, 177, 197, 216, 236, 255]),
            (
                "midtones",
                100.0,
                true,
                [0, 36, 59, 78, 96, 112, 128, 142, 156, 170, 183, 196, 209, 221, 233, 244, 255],
                [0, 5, 14, 24, 36, 50, 64, 80, 97, 114, 132, 151, 171, 191, 212, 234, 255],
            ),
            ("highlights", -100.0, true, X, [0, 26, 53, 79, 105, 132, 158, 184, 211, 237, 255, 255, 255, 255, 255, 255, 255]),
        ];
        // Two sliders in one tone, Preserve Luminosity on (measured): shadows red and blue +50 darken
        // only green; highlights red −50 with green +50 lift green by 100 and blue by 50.
        let two =
            |tones: [[f32; 3]; 3], x: f32| -> [i32; 3] { std::array::from_fn(|ch| (color_balance_channel(tones, true, ch, x / 255.0) * 255.0).round() as i32) };
        assert_eq!(two([[50.0, 0.0, 50.0], [0.0; 3], [0.0; 3]], 128.0), [128, 97, 128]);
        let hi = two([[0.0; 3], [0.0; 3], [-50.0, 50.0, 0.0]], 128.0);
        assert!((hi[0] - 128).abs() <= 1 && (hi[1] - 211).abs() <= 1 && (hi[2] - 159).abs() <= 1, "{hi:?}");
        for (tone, amount, preserve, red, green) in measured {
            let mut tones = [[0.0f32; 3]; 3];
            tones[["shadows", "midtones", "highlights"].iter().position(|t| *t == tone).unwrap()][0] = amount;
            for (i, x) in X.iter().enumerate() {
                let v = f32::from(*x) / 255.0;
                let got = |ch| (color_balance_channel(tones, preserve, ch, v) * 255.0).round() as i32;
                let (r, g) = (got(0), got(1));
                assert!((r - i32::from(red[i])).abs() <= 1, "{tone} {amount} pl {preserve} at {x}: red {r} vs Photoshop {}", red[i]);
                assert!((g - i32::from(green[i])).abs() <= 1, "{tone} {amount} pl {preserve} at {x}: green {g} vs Photoshop {}", green[i]);
            }
        }
    }

    #[test]
    fn color_balance_midtones_with_multiple_sliders_preserve_luminosity() {
        // A shared offset depends on the two extremes, not the sum of moved sliders.
        // The old per-slider product leaves red/blue unchanged for [30, 0, 30].
        let tones = [[0.0; 3], [30.0, 0.0, 30.0], [0.0; 3]];
        for input in [0.25_f32, 0.5, 0.75] {
            for (ch, offset) in [-0.15_f32, 0.15, -0.15].into_iter().enumerate() {
                let expected = input.powf(2f32.powf(offset));
                let actual = color_balance_channel(tones, true, ch, input);
                assert!((actual - expected).abs() < 1e-6, "channel {ch}, input {input}: {actual} vs {expected}");
            }
            // Preserve Luminosity off must still affect only the moved channels.
            assert_eq!(color_balance_channel(tones, false, 1, input), input);
            // One moved slider retains its previous half-strength compensation.
            let single = [[0.0; 3], [30.0, 0.0, 0.0], [0.0; 3]];
            assert!((color_balance_channel(single, true, 0, input) - input.powf(2f32.powf(-0.15))).abs() < 1e-6);
            assert!((color_balance_channel(single, true, 1, input) - input.powf(2f32.powf(0.15))).abs() < 1e-6);
        }
    }

    #[test]
    fn levels_and_curves_apply_each_channel_before_the_master() {
        // These two affine tone adjustments describe the same mapping:
        // channel(x) = 1 / 2 + x / 2, then master(x) = x / 2, so 0 maps to 0.25.
        // Levels used to apply the master first (0 mapped to 0.5).
        let line = |a: f32, b: f32| vec![CurvePoint { input: 0.0, output: a }, CurvePoint { input: 1.0, output: b }];
        let levels = Adjustment::Levels {
            master: LevelsChannel { out_white: 0.5, ..Default::default() },
            per_channel: std::array::from_fn(|_| LevelsChannel { out_black: 0.5, ..Default::default() }),
            space: ToneSpace::Rgb,
            black: Default::default(),
        };
        let curves = Adjustment::Curves {
            master: line(0.0, 0.5),
            per_channel: std::array::from_fn(|_| line(0.5, 1.0)),
            space: ToneSpace::Rgb,
            black: Vec::new(),
            pencil: false,
        };
        for depth in [None, Some(SampleType::U8), Some(SampleType::U16), Some(SampleType::F32)] {
            let ll = tone_luts_depth(&levels, depth);
            let cl = tone_luts_depth(&curves, depth);
            for channel in 0..3 {
                for index in [0, LUT_SIZE / 4, LUT_SIZE / 2, 3 * LUT_SIZE / 4, LUT_SIZE - 1] {
                    assert!(
                        (ll[channel][index] - cl[channel][index]).abs() < 0.01,
                        "{depth:?} channel {channel} sample {index}: Levels {} vs Curves {}",
                        ll[channel][index],
                        cl[channel][index]
                    );
                }
            }
            assert!((cl[0][0] - 0.25).abs() < 0.01, "{depth:?}: {}", cl[0][0]);

            // Exercise the compositor's real adjustment path, not only its LUT builder.
            for adj in [&levels, &curves] {
                let mut pixel = Buffer { rect: Rect::new(0, 0, 1, 1), px: vec![[0.0, 0.0, 0.0, 1.0]] };
                apply_depth(adj, &mut pixel, Transfer::Srgb, depth);
                for channel in 0..3 {
                    assert!((pixel.px[0][channel] - 0.25).abs() < 0.01, "{depth:?}: {adj:?} rendered {:?}", pixel.px[0]);
                }
            }
        }

        // Lab has no composite/master tone record.
        let mut lab = curves.clone();
        let mut lab_without_master = curves;
        if let Adjustment::Curves { space, .. } = &mut lab {
            *space = ToneSpace::Lab;
        }
        if let Adjustment::Curves { master, space, .. } = &mut lab_without_master {
            *space = ToneSpace::Lab;
            *master = line(0.0, 1.0);
        }
        assert_eq!(tone_luts(&lab), tone_luts(&lab_without_master));
    }

    #[test]
    fn identity_tone_in_every_space_is_identity() {
        for space in [ToneSpace::Rgb, ToneSpace::Cmyk, ToneSpace::Lab] {
            let mut a = Adjustment::identity_curves();
            if let Adjustment::Curves { space: s, .. } = &mut a {
                *s = space;
            }
            let mut b = ramp();
            apply(&a, &mut b);
            for (x, y) in b.px.iter().zip(ramp().px.iter()) {
                for k in 0..3 {
                    assert!((x[k] - y[k]).abs() < 0.02, "{space:?} {x:?} {y:?}");
                }
            }
        }
    }

    #[test]
    fn cmyk_black_curve_darkens_and_lab_lightness_lifts() {
        let line = |a: f32, b: f32| vec![CurvePoint { input: 0.0, output: a }, CurvePoint { input: 1.0, output: b }];
        // Black brightness 1 -> 0.5: more black ink everywhere.
        let a = Adjustment::Curves {
            master: line(0.0, 1.0),
            per_channel: [line(0.0, 1.0), line(0.0, 1.0), line(0.0, 1.0)],
            space: ToneSpace::Cmyk,
            black: line(0.0, 0.5),
            pencil: false,
        };
        let mut b = Buffer { rect: Rect::new(0, 0, 1, 1), px: vec![[0.8, 0.8, 0.8, 1.0]] };
        apply(&a, &mut b);
        assert!(b.px[0][0] < 0.7, "{:?}", b.px[0]);
        // Lab lightness 0..1 -> 0.2..1 lifts a dark grey; a/b untouched keeps it grey.
        let a = Adjustment::Curves {
            master: line(1.0, 0.0), // ignored in Lab
            per_channel: [line(0.2, 1.0), line(0.0, 1.0), line(0.0, 1.0)],
            space: ToneSpace::Lab,
            black: Vec::new(),
            pencil: false,
        };
        let mut b = Buffer { rect: Rect::new(0, 0, 1, 1), px: vec![[0.2, 0.2, 0.2, 1.0]] };
        apply(&a, &mut b);
        let p = b.px[0];
        assert!(p[0] > 0.3 && (p[0] - p[2]).abs() < 0.02, "{p:?}");
    }

    #[test]
    fn hue_ranges_only_touch_their_colours() {
        let mut ranges = HueRange::defaults();
        ranges[0].saturation = -100.0; // desaturate reds
        let a = Adjustment::HueSaturation { hue: 0.0, saturation: 0.0, lightness: 0.0, colorize: false, ranges };
        let mut b = Buffer { rect: Rect::new(0, 0, 3, 1), px: vec![[0.9, 0.1, 0.1, 1.0], [0.1, 0.1, 0.9, 1.0], [0.5, 0.5, 0.5, 1.0]] };
        apply(&a, &mut b);
        let red = b.px[0];
        assert!((red[0] - red[1]).abs() < 0.05, "red desaturated {red:?}");
        assert!((b.px[1][2] - 0.9).abs() < 1e-4 && (b.px[1][0] - 0.1).abs() < 1e-4, "blue untouched {:?}", b.px[1]);
        assert_eq!(b.px[2], [0.5, 0.5, 0.5, 1.0]);
    }

    #[test]
    fn colorize_at_zero_saturation_is_grey() {
        // Photoshop: Colorize at 0% saturation leaves no colour, only the pixel's lightness (#2326).
        let px = vec![[0.9, 0.2, 0.1, 1.0], [0.1, 0.6, 0.8, 1.0], [0.5, 0.5, 0.5, 1.0]];
        for depth in [None, Some(SampleType::U8), Some(SampleType::U16), Some(SampleType::F32)] {
            for lightness in [0.0, 30.0, -30.0] {
                let a = Adjustment::HueSaturation { hue: 200.0, saturation: 0.0, lightness, colorize: true, ranges: HueRange::defaults() };
                let mut b = Buffer { rect: Rect::new(0, 0, 3, 1), px: px.clone() };
                apply_depth(&a, &mut b, Transfer::Srgb, depth);
                for p in &b.px {
                    assert!((p[0] - p[1]).abs() < 1e-4 && (p[1] - p[2]).abs() < 1e-4, "{depth:?} lightness {lightness}: {p:?}");
                }
            }
            // Low saturations still tint: the old 25% floor no longer hides them.
            let tint = |saturation: f32| {
                let a = Adjustment::HueSaturation { hue: 0.0, saturation, lightness: 0.0, colorize: true, ranges: HueRange::defaults() };
                let mut b = Buffer { rect: Rect::new(0, 0, 1, 1), px: vec![[0.5, 0.5, 0.5, 1.0]] };
                apply_depth(&a, &mut b, Transfer::Srgb, depth);
                b.px[0][0] - b.px[0][2]
            };
            assert!(tint(10.0) > 0.05 && tint(10.0) < tint(25.0), "{depth:?}: 10% {} vs 25% {}", tint(10.0), tint(25.0));
        }
    }

    /// Photoshop 27.11, measured: Hue/Saturation raising saturation, and Colorize with Lightness.
    #[test]
    fn hue_saturation_matches_photoshop() {
        // (input, hue, saturation, lightness, colorize, Photoshop's output)
        type Case = ([u8; 3], f32, f32, f32, bool, [u8; 3]);
        let measured: [Case; 8] = [
            ([204, 102, 102], 0.0, 50.0, 0.0, false, [254, 51, 51]),
            ([204, 159, 102], 0.0, 50.0, 0.0, false, [254, 164, 51]),
            ([204, 102, 102], 0.0, -50.0, 0.0, false, [178, 127, 127]),
            ([0, 0, 0], 120.0, 100.0, 30.0, true, [0, 152, 0]),
            ([28, 28, 28], 120.0, 100.0, 30.0, true, [0, 192, 0]),
            ([160, 160, 160], 30.0, 25.0, -40.0, true, [120, 96, 73]),
            ([252, 252, 252], 30.0, 25.0, -40.0, true, [177, 151, 126]),
            ([255, 143, 0], 30.0, 25.0, -40.0, true, [95, 76, 58]),
        ];
        for (input, hue, saturation, lightness, colorize, want) in measured {
            let a = Adjustment::HueSaturation { hue, saturation, lightness, colorize, ranges: HueRange::defaults() };
            let c = input.map(|v| v as f32 / 255.0);
            let mut b = Buffer { rect: Rect::new(0, 0, 1, 1), px: vec![[c[0], c[1], c[2], 1.0]] };
            apply(&a, &mut b);
            for (i, w) in want.iter().enumerate() {
                let got = b.px[0][i] * 255.0;
                assert!((got - *w as f32).abs() <= 2.0, "{input:?} h{hue} s{saturation} l{lightness} colorize {colorize}, channel {i}: {got} vs {w}");
            }
        }
    }

    /// Photoshop 27.11, measured: a range's Lightness (Blues ±50) moves towards the brightest or darkest channel.
    #[test]
    fn range_lightness_matches_photoshop() {
        let measured = [
            ([0u8, 29, 115], 0.5, [57u8, 71, 115]),
            ([0, 64, 255], 0.5, [125, 157, 255]),
            ([102, 128, 204], 0.5, [152, 165, 204]),
            ([102, 128, 204], -0.5, [102, 115, 154]),
            ([0, 18, 115], -0.5, [0, 9, 58]),
        ];
        for (input, l, want) in measured {
            let got = range_lightness(input.map(|v| v as f32 / 255.0), l);
            for i in 0..3 {
                assert!((got[i] * 255.0 - want[i] as f32).abs() <= 2.5, "{input:?} channel {i}: {} vs {}", got[i] * 255.0, want[i]);
            }
        }
    }

    /// Photoshop's own render (merged image of psd-tools `adjustments/huesaturation_lightness_rgb.psd`,
    /// layer "Hue/Saturation 1"): a dull magenta gets the whole range edit, as a vivid one does (#4089).
    #[test]
    fn hue_ranges_apply_in_full_to_dull_colours() {
        let range = |lightness: f32, bounds: [f32; 4]| HueRange { hue: 0.0, saturation: 0.0, lightness, bounds };
        let ranges = [
            range(-11.0, [-57.0, 4.0, 82.0, 175.0]),
            range(100.0, [12.0, 45.0, 75.0, 105.0]),
            HueRange::neutral(2),
            HueRange::neutral(3),
            HueRange::neutral(4),
            range(100.0, [-117.0, -50.0, 39.0, 104.0]),
        ];
        let a = Adjustment::HueSaturation { hue: 0.0, saturation: 0.0, lightness: 17.0, colorize: false, ranges };
        let measured = [([126u8, 97, 120], [148u8, 148, 148]), ([131, 103, 125], [152, 152, 152]), ([95, 69, 90], [122, 122, 122])];
        for (input, want) in measured {
            let c = input.map(|v| v as f32 / 255.0);
            let mut b = Buffer { rect: Rect::new(0, 0, 1, 1), px: vec![[c[0], c[1], c[2], 1.0]] };
            apply(&a, &mut b);
            for (i, w) in want.iter().enumerate() {
                let got = b.px[0][i] * 255.0;
                assert!((got - *w as f32).abs() <= 2.0, "{input:?} channel {i}: {got} vs {w}");
            }
        }
        // Greys have no hue, so the ranges leave them alone: only the master Lightness moves them.
        let mut b = Buffer { rect: Rect::new(0, 0, 1, 1), px: vec![[0.4, 0.4, 0.4, 1.0]] };
        apply(&a, &mut b);
        let p = b.px[0];
        assert!((p[0] - p[1]).abs() < 1e-6 && (p[1] - p[2]).abs() < 1e-6 && (p[0] - (0.4 + 0.6 * 0.17)).abs() < 1e-4, "{p:?}");
    }

    #[test]
    fn black_white_weights() {
        let d = [40.0, 60.0, 40.0, 60.0, 20.0, 80.0];
        assert!((black_white_gray([0.5; 3], &d) - 0.5).abs() < 1e-6, "greys keep their value");
        assert!((black_white_gray([1.0, 0.0, 0.0], &d) - 0.4).abs() < 1e-6);
        assert!((black_white_gray([1.0, 1.0, 0.0], &d) - 0.6).abs() < 1e-6);
        let mut w = d;
        w[0] = 100.0;
        assert!(black_white_gray([1.0, 0.0, 0.0], &w) > black_white_gray([1.0, 0.0, 0.0], &d));
    }

    /// Photoshop 27.11, measured: Black & White (default weights) with Tint over greys. The tint
    /// is blended like the Color blend mode: its hue and saturation at the grey's luminosity.
    #[test]
    fn black_white_tint_matches_photoshop() {
        let weights = [40.0, 60.0, 40.0, 60.0, 20.0, 80.0];
        let measured: [([u8; 3], u8, [u8; 3]); 6] = [
            ([255, 0, 0], 24, [81, 1, 1]),
            ([255, 0, 0], 120, [255, 64, 64]),
            ([255, 0, 0], 240, [255, 235, 235]),
            ([0xe1, 0xd3, 0xb3], 120, [133, 119, 87]),
            ([0xe1, 0xd3, 0xb3], 24, [33, 24, 1]),
            ([0x40, 0x20, 0x10], 120, [144, 112, 96]),
        ];
        for (tint, grey, want) in measured {
            let tint = Some(tint.map(|v| v as f32 / 255.0));
            let mut b = Buffer { rect: Rect::new(0, 0, 1, 1), px: vec![[grey as f32 / 255.0; 4]] };
            b.px[0][3] = 1.0;
            apply(&Adjustment::BlackWhite { weights, tint }, &mut b);
            for (i, w) in want.iter().enumerate() {
                let got = b.px[0][i] * 255.0;
                assert!((got - *w as f32).abs() <= 2.5, "tint {tint:?} grey {grey} channel {i}: {got} vs {w}");
            }
        }
    }

    #[test]
    fn gradient_map_dither_stays_close() {
        let stops = vec![(0.0, [0.0; 3]), (1.0, [1.0; 3])];
        let mut a = ramp();
        let mut b = ramp();
        apply(&Adjustment::GradientMap { stops: stops.clone(), reverse: false, dither: false }, &mut a);
        apply(&Adjustment::GradientMap { stops, reverse: false, dither: true }, &mut b);
        assert!(a.px.iter().zip(&b.px).all(|(x, y)| (x[0] - y[0]).abs() <= 0.5 / 255.0 + 1e-6));
        assert!(a.px.iter().zip(&b.px).any(|(x, y)| x[0] != y[0]));
    }

    /// `adj` applied in a Lab document to one encoded sRGB colour, returned as (L, a, b).
    fn in_lab(adj: &Adjustment, c: [f32; 3]) -> [f32; 3] {
        let mut b = Buffer::filled(crate::Rect::new(0, 0, 1, 1), [c[0], c[1], c[2], 1.0]);
        apply_in(adj, &mut b, Transfer::Srgb, Some(SampleType::U8), ColorMode::Lab);
        srgb_to_lab([b.px[0][0], b.px[0][1], b.px[0][2]])
    }

    /// A muted, in-gamut colour from (L, a, b).
    fn lab(l: f32, a: f32, b: f32) -> [f32; 3] {
        lab_to_srgb([l, a, b])
    }

    fn near(got: [f32; 3], want: [f32; 3], tol: f32) -> bool {
        got.iter().zip(want).all(|(g, w)| (g - w).abs() <= tol)
    }

    #[test]
    fn lab_invert_flips_l_a_and_b() {
        let got = in_lab(&Adjustment::Invert, lab(30.0, 12.0, -18.0));
        // n -> 1 - n: L -> 100 - L, a -> -a - 1, b -> -b - 1 (mirrored on the 0..255 grid).
        assert!(near(got, [70.0, -13.0, 17.0], 0.6), "{got:?}");
        // On the screen colour it would not be a mirror in Lab.
        let mut rgb = Buffer::filled(crate::Rect::new(0, 0, 1, 1), [0.0, 0.0, 0.0, 1.0]);
        let c = lab(30.0, 12.0, -18.0);
        rgb.px[0] = [c[0], c[1], c[2], 1.0];
        apply(&Adjustment::Invert, &mut rgb);
        assert!(!near(srgb_to_lab([rgb.px[0][0], rgb.px[0][1], rgb.px[0][2]]), got, 1.0));
    }

    #[test]
    fn sixteen_bit_lab_invert_keeps_greys_neutral() {
        // 16-bit Lab Invert mirrors a and b around neutral (photoshop corpus lab16 invert.psd);
        // 8-bit bytes land one step off it.
        let mut b = Buffer::filled(crate::Rect::new(0, 0, 1, 1), [0.0, 0.0, 0.0, 1.0]);
        let c = lab(40.0, 0.0, 0.0);
        b.px[0] = [c[0], c[1], c[2], 1.0];
        apply_in(&Adjustment::Invert, &mut b, Transfer::Srgb, Some(SampleType::U16), ColorMode::Lab);
        let got = srgb_to_lab([b.px[0][0], b.px[0][1], b.px[0][2]]);
        assert!(near(got, [60.0, 0.0, 0.0], 0.05), "{got:?}");
        assert!(near(in_lab(&Adjustment::Invert, c), [60.0, -1.0, -1.0], 0.1));
    }

    /// `adj` applied in a CMYK document to one encoded sRGB colour.
    fn in_cmyk(adj: &Adjustment, c: [f32; 3]) -> [f32; 3] {
        let mut b = Buffer::filled(crate::Rect::new(0, 0, 1, 1), [c[0], c[1], c[2], 1.0]);
        apply_in(adj, &mut b, Transfer::Srgb, Some(SampleType::U8), ColorMode::Cmyk);
        [b.px[0][0], b.px[0][1], b.px[0][2]]
    }

    #[test]
    fn cmyk_invert_posterize_and_threshold_work_on_the_inks() {
        use photocraft_color::convert::{cmyk_to_rgb, rgb_to_cmyk};
        // Paper inverts to all four inks at 100%, not to RGB black (photoshop corpus cmyk8 invert.psd).
        assert!(near(in_cmyk(&Adjustment::Invert, [1.0; 3]), cmyk_to_rgb([1.0; 4]), 2e-3));
        let red = [0.9, 0.2, 0.1];
        assert!(near(in_cmyk(&Adjustment::Invert, red), cmyk_to_rgb(rgb_to_cmyk(red).map(|v| 1.0 - v)), 2e-3));
        let two = cmyk_to_rgb(rgb_to_cmyk(red).map(|v| if v >= 0.5 { 1.0 } else { 0.0 }));
        assert!(near(in_cmyk(&Adjustment::Posterize { levels: 2 }, red), two, 2e-3));
        // Threshold's black is the black ink alone (threshold-128.psd), its white the paper.
        let t = Adjustment::Threshold { level: 0.5 };
        assert!(near(in_cmyk(&t, [0.2; 3]), cmyk_to_rgb([0.0, 0.0, 0.0, 1.0]), 2e-3));
        assert!(near(in_cmyk(&t, [0.8; 3]), cmyk_to_rgb([0.0; 4]), 2e-3));
    }

    #[test]
    fn lab_posterize_steps_l_a_and_b() {
        let got = in_lab(&Adjustment::Posterize { levels: 2 }, lab(70.0, 20.0, -30.0));
        // Two levels: each normalised channel snaps to 0 or 1 (L 100, a and b at +127 / -128).
        assert!(near(got, srgb_to_lab(lab_denorm([1.0, 1.0, 0.0])), 1e-3), "{got:?}");
        let grey = in_lab(&Adjustment::Posterize { levels: 255 }, lab(50.0, 0.0, 0.0));
        assert!(near(grey, [50.0, 0.0, 0.0], 1.1), "255 levels is within a step of the identity: {grey:?}");
    }

    #[test]
    fn lab_threshold_splits_on_l_with_a_neutral_result() {
        let t = Adjustment::Threshold { level: 0.5 };
        // A saturated blue (L ≈ 32) and yellow (L ≈ 97): black and white, nothing in between.
        for (c, white) in [([0.0, 0.0, 1.0], false), ([1.0, 1.0, 0.0], true), (lab(52.0, 40.0, 30.0), true), (lab(48.0, -40.0, 30.0), false)] {
            let mut b = Buffer::filled(crate::Rect::new(0, 0, 1, 1), [c[0], c[1], c[2], 1.0]);
            apply_in(&t, &mut b, Transfer::Srgb, None, ColorMode::Lab);
            let v = if white { 1.0 } else { 0.0 };
            assert_eq!(b.px[0], [v, v, v, 1.0], "{c:?}");
        }
    }

    #[test]
    fn lab_brightness_contrast_moves_l_only() {
        for legacy in [false, true] {
            let adj = Adjustment::BrightnessContrast { brightness: 40.0, contrast: 30.0, legacy };
            let before = [45.0, 15.0, -20.0];
            let got = in_lab(&adj, lab(before[0], before[1], before[2]));
            assert!(got[0] > before[0] + 5.0, "{legacy}: L rises: {got:?}");
            assert!((got[1] - before[1]).abs() < 0.5 && (got[2] - before[2]).abs() < 0.5, "{legacy}: a, b kept: {got:?}");
        }
    }

    #[test]
    fn lab_gradient_map_follows_l() {
        let stops = vec![(0.0, [1.0, 0.0, 0.0]), (1.0, [0.0, 0.0, 1.0])];
        for reverse in [false, true] {
            let map = Adjustment::GradientMap { stops: stops.clone(), reverse, dither: false };
            // A grey and a saturated colour with the same L map to the same colour.
            let grey = in_lab(&map, lab(60.0, 0.0, 0.0));
            let green = in_lab(&map, lab(60.0, -50.0, 40.0));
            assert!(near(grey, green, 1e-3), "{reverse}: {grey:?} vs {green:?}");
            let t = if reverse { 0.4 } else { 0.6 };
            let want = srgb_to_lab([1.0 - t, 0.0, t]);
            assert!(near(grey, want, 0.05), "{reverse}: {grey:?} vs {want:?}");
        }
    }

    #[test]
    fn other_modes_and_adjustments_keep_the_screen_colour_path() {
        let c = [0.2, 0.5, 0.7, 1.0];
        let lab_only = [Adjustment::Invert, Adjustment::Threshold { level: 0.47 }, Adjustment::Posterize { levels: 4 }];
        for adj in
            lab_only.iter().chain([&Adjustment::HueSaturation { hue: 30.0, saturation: 0.0, lightness: 0.0, colorize: false, ranges: HueRange::defaults() }])
        {
            for mode in [ColorMode::Rgb, ColorMode::Cmyk, ColorMode::Grayscale, ColorMode::Lab] {
                let (mut a, mut b) = (Buffer::filled(crate::Rect::new(0, 0, 1, 1), c), Buffer::filled(crate::Rect::new(0, 0, 1, 1), c));
                apply_in(adj, &mut a, Transfer::Srgb, None, mode);
                apply_depth(adj, &mut b, Transfer::Srgb, None);
                let own = (mode == ColorMode::Lab && lab_channel_adjustment(adj)) || (mode == ColorMode::Cmyk && cmyk_channel_adjustment(adj));
                assert_eq!(a.px == b.px, !own, "{adj:?} in {mode:?}");
            }
        }
        // Transparent pixels are left alone and odd values never panic.
        let mut t = Buffer::filled(crate::Rect::new(0, 0, 2, 1), [f32::NAN, 2.0, -1.0, 0.0]);
        for adj in lab_only.iter().chain([&Adjustment::GradientMap { stops: vec![], reverse: true, dither: true }]) {
            for mode in [ColorMode::Lab, ColorMode::Cmyk] {
                apply_in(adj, &mut t, Transfer::Srgb, None, mode);
                let mut odd = Buffer::filled(crate::Rect::new(0, 0, 1, 1), [f32::NAN, f32::INFINITY, -3.0, 1.0]);
                apply_in(adj, &mut odd, Transfer::Srgb, Some(SampleType::U16), mode);
            }
        }
        assert!(t.px[0][1] == 2.0);
    }
}

/// Legacy Brightness/Contrast scales contrast around 128 of 255 (contrast 80 keeps 128 at 128,
/// measured on Photoshop 27.11; a 0.5 pivot moved it to 130).
pub const LEGACY_PIVOT: f32 = 128.0 / 255.0;

#[cfg(test)]
mod lut3d_guard_tests {
    use super::lut3d_sample;

    #[test]
    fn degenerate_or_short_tables_leave_the_colour_alone() {
        let c = [0.2, 0.4, 0.6];
        for tetra in [false, true] {
            assert_eq!(lut3d_sample(&[], 0, c, tetra), c);
            assert_eq!(lut3d_sample(&[0.0; 3], 1, c, tetra), c);
            assert_eq!(lut3d_sample(&[0.0; 3], 2, c, tetra), c);
            assert_eq!(lut3d_sample(&[0.0; 24], usize::MAX, c, tetra), c);
        }
    }
}
