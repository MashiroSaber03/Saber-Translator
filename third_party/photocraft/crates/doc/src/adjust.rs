//! Adjustment parameters (data only). Evaluation lives in `photocraft-compose`.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CurvePoint {
    pub input: f32,
    pub output: f32,
}

/// Whether `points` is the straight 0→0, 1→1 line (or too short to be a curve).
pub fn is_identity_curve(points: &[CurvePoint]) -> bool {
    match points {
        [a, b] => a.input == 0.0 && a.output == 0.0 && b.input == 1.0 && b.output == 1.0,
        p => p.len() < 2,
    }
}

/// Entries in a pencil-drawn Curves map (one per 8-bit level, as in PSD `curv` maps).
pub const CURVE_MAP_LEN: usize = 256;

/// A pencil-mode curve from its map: `outputs[k]` (0..=1) is the output at input `k / 255`.
/// Missing entries repeat the last one; extra entries are ignored.
pub fn curve_map(outputs: &[f32]) -> Vec<CurvePoint> {
    let last = (CURVE_MAP_LEN - 1) as f32;
    let mut prev = 0.0;
    (0..CURVE_MAP_LEN)
        .map(|k| {
            let out = outputs.get(k).copied().filter(|v| v.is_finite()).map_or(prev, |v| v.clamp(0.0, 1.0));
            prev = out;
            CurvePoint { input: k as f32 / last, output: out }
        })
        .collect()
}

/// The [`CURVE_MAP_LEN`] outputs (0..=1) of `points` read as a pencil-mode curve: straight
/// segments between the points, constant outside the first/last (identity without two points).
pub fn curve_map_outputs(points: &[CurvePoint]) -> Vec<f32> {
    let mut pts: Vec<(f32, f32)> = points.iter().filter(|p| p.input.is_finite() && p.output.is_finite()).map(|p| (p.input, p.output)).collect();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let last = (CURVE_MAP_LEN - 1) as f32;
    (0..CURVE_MAP_LEN)
        .map(|k| {
            let x = k as f32 / last;
            let y = match (pts.first(), pts.last()) {
                (Some(a), Some(b)) if pts.len() >= 2 => {
                    if x <= a.0 {
                        a.1
                    } else if x >= b.0 {
                        b.1
                    } else {
                        let i = pts.partition_point(|p| p.0 < x);
                        match (pts.get(i.wrapping_sub(1)), pts.get(i)) {
                            (Some(p0), Some(p1)) if p1.0 > p0.0 => p0.1 + (p1.1 - p0.1) * (x - p0.0) / (p1.0 - p0.0),
                            (_, Some(p1)) => p1.1,
                            _ => b.1,
                        }
                    }
                }
                _ => x,
            };
            y.clamp(0.0, 1.0)
        })
        .collect()
}

/// Number of values in the table of a `size`³ Color Lookup (`size³ × 3`). `None` for a size
/// below 2 or one whose table length overflows: a stored document's size is untrusted.
pub fn lut3d_len(size: u32) -> Option<usize> {
    if size < 2 {
        return None;
    }
    usize::try_from(size).ok()?.checked_pow(3)?.checked_mul(3)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LevelsChannel {
    pub in_black: f32,
    pub in_white: f32,
    pub gamma: f32,
    pub out_black: f32,
    pub out_white: f32,
}

impl Default for LevelsChannel {
    fn default() -> Self {
        Self { in_black: 0.0, in_white: 1.0, gamma: 1.0, out_black: 0.0, out_white: 1.0 }
    }
}

/// Which document channels the per-channel records of Levels and Curves address. Compositing
/// happens in display RGB; `Cmyk` and `Lab` records are applied to the pixel converted into that
/// space (the same conversions the document's surfaces use), so ink and Lab channel edits behave
/// like Photoshop's in CMYK and Lab documents. Grayscale documents use `Rgb` with the master record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToneSpace {
    /// `per_channel` = red, green, blue.
    #[default]
    Rgb,
    /// `per_channel` = cyan, magenta, yellow plus `black`; values are channel brightness
    /// (1 - ink), and the master record applies to all four inks.
    Cmyk,
    /// `per_channel` = lightness, a, b (stored 0..=1 like Lab surfaces); there is no composite
    /// record in Lab, so the master is ignored.
    Lab,
}

/// One of Hue/Saturation's six colour ranges (reds, yellows, greens, cyans, blues, magentas).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HueRange {
    /// Hue shift in degrees (-180..=180).
    pub hue: f32,
    /// Saturation change in percent (-100..=100).
    pub saturation: f32,
    /// Lightness change in percent (-100..=100).
    pub lightness: f32,
    /// Hue degrees where the range starts fading in, is fully on, starts fading out and ends
    /// (Photoshop's four range sliders). May wrap past 0/360.
    pub bounds: [f32; 4],
}

impl HueRange {
    /// The neutral range `i` (0 = reds … 5 = magentas) with Photoshop's default extent: fully on
    /// for 30° around its centre, fading over 30° each side.
    pub fn neutral(i: usize) -> Self {
        let c = 60.0 * i as f32;
        Self { hue: 0.0, saturation: 0.0, lightness: 0.0, bounds: [c - 45.0, c - 15.0, c + 15.0, c + 45.0] }
    }

    pub fn defaults() -> [HueRange; 6] {
        std::array::from_fn(Self::neutral)
    }

    /// The canonical form of four range sliders: increasing, within one turn, centre in
    /// -30..330 (so the default reds are -45, -15, 15, 45 whichever way they were written).
    pub fn canonical_bounds(b: [f32; 4]) -> [f32; 4] {
        if b.iter().any(|v| !v.is_finite()) {
            return [-45.0, -15.0, 15.0, 45.0];
        }
        let mut out = [b[0]; 4];
        for k in 1..4 {
            out[k] = (out[k - 1] + (b[k] - out[k - 1]).rem_euclid(360.0)).min(out[0] + 359.0);
        }
        let centre = (out[1] + out[2]) / 2.0;
        let shift = (centre + 30.0).rem_euclid(360.0) - 30.0 - centre;
        out.map(|v| v + shift)
    }

    pub fn is_neutral(&self) -> bool {
        self.hue == 0.0 && self.saturation == 0.0 && self.lightness == 0.0
    }

    /// How much a pixel of hue `deg` (degrees) belongs to this range, 0..=1.
    pub fn weight(&self, deg: f32) -> f32 {
        let [a, b, c, d] = self.bounds;
        // Measure from the range's start so wrap-around ranges (reds) work.
        let rel = |x: f32| if x.is_finite() { (x - a).rem_euclid(360.0) } else { 0.0 };
        let b = rel(b);
        let c = rel(c).max(b);
        let d = rel(d).max(c);
        let h = rel(deg);
        if h <= b {
            if b <= 0.0 { 1.0 } else { h / b }
        } else if h <= c {
            1.0
        } else if h <= d {
            if d <= c { 1.0 } else { (d - h) / (d - c) }
        } else {
            0.0
        }
    }
}

fn default_hue_ranges() -> [HueRange; 6] {
    HueRange::defaults()
}

/// Photoshop adjustment layers (Layer → New Adjustment Layer).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Adjustment {
    BrightnessContrast {
        brightness: f32,
        contrast: f32,
        legacy: bool,
    },
    /// R, G, B (or the channels `space` names) first, then the master (composite) record.
    /// In CMYK this also applies to `black`; in Lab the master is ignored.
    Levels {
        master: LevelsChannel,
        per_channel: [LevelsChannel; 3],
        #[serde(default)]
        space: ToneSpace,
        /// The black ink channel when `space` is [`ToneSpace::Cmyk`].
        #[serde(default)]
        black: LevelsChannel,
    },
    /// R, G, B curves (or the channels `space` names) first, then the master (composite) curve.
    /// In CMYK this also applies to `black`; in Lab the master is ignored.
    Curves {
        master: Vec<CurvePoint>,
        per_channel: [Vec<CurvePoint>; 3],
        #[serde(default)]
        space: ToneSpace,
        /// The black ink curve when `space` is [`ToneSpace::Cmyk`] (empty = identity).
        #[serde(default)]
        black: Vec<CurvePoint>,
        /// Drawn with the pencil (Photoshop's "Draw to modify the curve", PSD `curv` map form):
        /// every curve is a [`CURVE_MAP_LEN`]-entry map (see [`curve_map`]) evaluated
        /// piecewise-linearly instead of a spline through its points.
        #[serde(default)]
        pencil: bool,
    },
    Exposure {
        exposure: f32,
        offset: f32,
        gamma: f32,
    },
    Vibrance {
        vibrance: f32,
        saturation: f32,
    },
    /// Master hue / saturation / lightness (or the Colorize colour) plus six hue ranges.
    HueSaturation {
        hue: f32,
        saturation: f32,
        lightness: f32,
        colorize: bool,
        #[serde(default = "default_hue_ranges")]
        ranges: [HueRange; 6],
    },
    ColorBalance {
        shadows: [f32; 3],
        midtones: [f32; 3],
        highlights: [f32; 3],
        preserve_luminosity: bool,
    },
    BlackWhite {
        weights: [f32; 6],
        tint: Option<[f32; 3]>,
    },
    PhotoFilter {
        color: [f32; 3],
        density: f32,
        preserve_luminosity: bool,
    },
    ChannelMixer {
        matrix: [[f32; 4]; 3],
        monochrome: bool,
    },
    /// A 3D LUT: `size`³ RGB triplets, red varying fastest (`((b·size + g)·size + r)·3`), in 0..=1.
    /// `lut` is None for a lookup Photoshop stores as an ICC profile (identity here).
    ColorLookup {
        name: String,
        lut: Option<Arc<Vec<f32>>>,
        size: u32,
        /// Tetrahedral instead of trilinear interpolation.
        #[serde(default)]
        tetrahedral: bool,
        /// Ordered dither of ±½ an 8-bit step to hide banding (Photoshop's "Dither").
        #[serde(default)]
        dither: bool,
    },
    Invert,
    Posterize {
        levels: u32,
    },
    Threshold {
        level: f32,
    },
    GradientMap {
        stops: Vec<(f32, [f32; 3])>,
        reverse: bool,
        /// Ordered dither of ±½ an 8-bit step to hide banding.
        #[serde(default)]
        dither: bool,
    },
    /// Per range (reds, yellows, greens, cyans, blues, magentas, whites, neutrals, blacks) the
    /// cyan, magenta, yellow and black change in percent (-100..=100), as in Photoshop.
    SelectiveColor {
        relative: bool,
        adjustments: [[f32; 4]; 9],
    },
    /// A PSD adjustment we can't evaluate yet; preserved raw for round-trip.
    Unsupported {
        psd_key: String,
        raw: Vec<u8>,
    },
}

impl Adjustment {
    pub fn label(&self) -> &str {
        match self {
            Adjustment::BrightnessContrast { .. } => "Brightness/Contrast",
            Adjustment::Levels { .. } => "Levels",
            Adjustment::Curves { .. } => "Curves",
            Adjustment::Exposure { .. } => "Exposure",
            Adjustment::Vibrance { .. } => "Vibrance",
            Adjustment::HueSaturation { .. } => "Hue/Saturation",
            Adjustment::ColorBalance { .. } => "Color Balance",
            Adjustment::BlackWhite { .. } => "Black & White",
            Adjustment::PhotoFilter { .. } => "Photo Filter",
            Adjustment::ChannelMixer { .. } => "Channel Mixer",
            Adjustment::ColorLookup { .. } => "Color Lookup",
            Adjustment::Invert => "Invert",
            Adjustment::Posterize { .. } => "Posterize",
            Adjustment::Threshold { .. } => "Threshold",
            Adjustment::GradientMap { .. } => "Gradient Map",
            Adjustment::SelectiveColor { .. } => "Selective Color",
            Adjustment::Unsupported { .. } => "Adjustment",
        }
    }

    pub fn default_hue_saturation() -> Self {
        Adjustment::HueSaturation { hue: 0.0, saturation: 0.0, lightness: 0.0, colorize: false, ranges: HueRange::defaults() }
    }

    pub fn identity_curves() -> Self {
        let line = || vec![CurvePoint { input: 0.0, output: 0.0 }, CurvePoint { input: 1.0, output: 1.0 }];
        Adjustment::Curves { master: line(), per_channel: [line(), line(), line()], space: ToneSpace::Rgb, black: Vec::new(), pencil: false }
    }

    pub fn identity_levels() -> Self {
        Adjustment::Levels { master: LevelsChannel::default(), per_channel: Default::default(), space: ToneSpace::Rgb, black: LevelsChannel::default() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hue_range_weights() {
        let reds = HueRange::neutral(0);
        assert_eq!(reds.weight(0.0), 1.0);
        assert_eq!(reds.weight(350.0), 1.0);
        assert!((reds.weight(30.0) - 0.5).abs() < 1e-5);
        assert!((reds.weight(330.0) - 0.5).abs() < 1e-5);
        assert_eq!(reds.weight(60.0), 0.0);
        assert_eq!(reds.weight(180.0), 0.0);
        let blues = HueRange::neutral(4);
        assert_eq!(blues.weight(240.0), 1.0);
        assert_eq!(blues.weight(120.0), 0.0);
        // Degenerate bounds never divide by zero.
        let odd = HueRange { bounds: [10.0, 10.0, 10.0, 10.0], ..HueRange::neutral(0) };
        assert!(odd.weight(10.0).is_finite() && odd.weight(200.0).is_finite());
        let nan = HueRange { bounds: [f32::NAN; 4], ..HueRange::neutral(0) };
        assert!(nan.weight(10.0).is_finite());
    }

    #[test]
    fn old_files_get_neutral_new_fields() {
        let a: Adjustment = serde_json::from_str(r#"{"HueSaturation":{"hue":1.0,"saturation":2.0,"lightness":3.0,"colorize":false}}"#).unwrap();
        assert!(matches!(a, Adjustment::HueSaturation { ranges, .. } if ranges == HueRange::defaults()));
        let c: Adjustment = serde_json::from_str(r#"{"Curves":{"master":[],"per_channel":[[],[],[]]}}"#).unwrap();
        assert!(matches!(c, Adjustment::Curves { space: ToneSpace::Rgb, pencil: false, .. }));
        let g: Adjustment = serde_json::from_str(r#"{"GradientMap":{"stops":[],"reverse":true}}"#).unwrap();
        assert!(matches!(g, Adjustment::GradientMap { dither: false, .. }));
    }

    #[test]
    fn lut3d_len_rejects_tiny_and_overflowing_sizes() {
        assert_eq!(lut3d_len(0), None);
        assert_eq!(lut3d_len(1), None);
        assert_eq!(lut3d_len(2), Some(24));
        assert_eq!(lut3d_len(33), Some(33 * 33 * 33 * 3));
        assert_eq!(lut3d_len(1 << 22), None, "(2²²)³ overflows a 64-bit length");
        assert_eq!(lut3d_len(u32::MAX), None);
    }
}
