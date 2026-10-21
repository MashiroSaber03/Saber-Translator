//! Colour-mode side data: the Indexed Color table, Duotone inks, and smart-object stack modes.
//!
//! Indexed, Bitmap and Duotone documents keep their pixels expanded (RGB / Gray, see
//! [`crate::Document::pixel_format`]); these structs carry what the mode adds on top.

use serde::{Deserialize, Serialize};

use crate::adjust::CurvePoint;

/// Image › Mode › Color Table: the palette of an Indexed Color document (at most 256 entries).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ColorTable {
    /// sRGB 8-bit entries.
    pub colors: Vec<[u8; 3]>,
    /// Index of the entry shown as transparent (Indexed Color › Transparency).
    #[serde(default)]
    pub transparent: Option<u8>,
}

impl ColorTable {
    /// Nearest palette index for an RGB colour in 0..=1 (squared distance, ties to the lower index).
    pub fn nearest(&self, c: [f32; 3]) -> usize {
        let mut best = (f32::MAX, 0);
        for (i, e) in self.colors.iter().enumerate() {
            let d: f32 = (0..3).map(|k| (c[k] - f32::from(e[k]) / 255.0).powi(2)).sum();
            if d < best.0 {
                best = (d, i);
            }
        }
        best.1
    }

    /// Entry `i` as RGB in 0..=1 (black past the end).
    pub fn rgb(&self, i: usize) -> [f32; 3] {
        self.colors.get(i).map_or([0.0; 3], |e| e.map(|v| f32::from(v) / 255.0))
    }
}

/// One Duotone ink: its display colour and the curve mapping gray density to ink density.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DuotoneInk {
    pub name: String,
    /// Display colour of the solid ink (sRGB 0..=1).
    pub color: [f32; 3],
    /// Ink density curve, input = gray *density* (0 = paper, 1 = solid), output = ink density.
    pub curve: Vec<CurvePoint>,
}

impl DuotoneInk {
    pub fn new(name: impl Into<String>, color: [f32; 3]) -> Self {
        DuotoneInk { name: name.into(), color, curve: vec![CurvePoint { input: 0.0, output: 0.0 }, CurvePoint { input: 1.0, output: 1.0 }] }
    }
}

/// Image › Mode › Duotone: 1–4 inks (Monotone, Duotone, Tritone, Quadtone) printed over paper.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Duotone {
    pub inks: Vec<DuotoneInk>,
    /// The PSD colour-mode data this came from, written back unchanged while the inks are unedited.
    #[serde(default)]
    pub psd_raw: Option<Vec<u8>>,
}

impl DuotoneInk {
    /// Ink density for a gray density (0 = paper, 1 = solid) through the ink's curve.
    pub fn density(&self, gray_density: f32) -> f32 {
        curve_eval(&self.curve, gray_density.clamp(0.0, 1.0)).clamp(0.0, 1.0)
    }
}

impl Duotone {
    /// Simulated print colour of a gray value (0 = black, 1 = white): inks multiply over white paper.
    pub fn render(&self, gray: f32) -> [f32; 3] {
        let density = 1.0 - gray.clamp(0.0, 1.0);
        let mut out = [1.0f32; 3];
        for ink in &self.inks {
            let d = curve_eval(&ink.curve, density);
            for (o, c) in out.iter_mut().zip(ink.color) {
                // Each ink filters light: full density transmits the ink colour.
                *o *= 1.0 - d * (1.0 - c);
            }
        }
        out
    }
}

impl Duotone {
    /// A Gradient Map that turns the gray composite into the printed look; displays and flat
    /// exports put it on top of a copy of the document.
    pub fn display_adjustment(&self) -> crate::Adjustment {
        let stops = (0..=32).map(|i| {
            let t = i as f32 / 32.0;
            (t, self.render(t))
        });
        crate::Adjustment::GradientMap { stops: stops.collect(), reverse: false, dither: false }
    }
}

/// Piecewise-linear evaluation (Photoshop's duotone curves are 13-point linear transfer curves).
fn curve_eval(pts: &[CurvePoint], x: f32) -> f32 {
    match pts {
        [] => x,
        [p] => p.output,
        _ => {
            if x <= pts[0].input {
                return pts[0].output;
            }
            for w in pts.windows(2) {
                if x <= w[1].input {
                    let span = (w[1].input - w[0].input).max(1e-6);
                    return w[0].output + (w[1].output - w[0].output) * (x - w[0].input) / span;
                }
            }
            pts[pts.len() - 1].output
        }
    }
}

/// Layer › Smart Objects › Stack Mode: a per-pixel statistic over the smart object's layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StackMode {
    Entropy,
    Kurtosis,
    Maximum,
    Mean,
    Median,
    Minimum,
    Range,
    Skewness,
    StandardDeviation,
    Summation,
    Variance,
}

impl StackMode {
    pub const ALL: [StackMode; 11] = [
        StackMode::Entropy,
        StackMode::Kurtosis,
        StackMode::Maximum,
        StackMode::Mean,
        StackMode::Median,
        StackMode::Minimum,
        StackMode::Range,
        StackMode::Skewness,
        StackMode::StandardDeviation,
        StackMode::Summation,
        StackMode::Variance,
    ];

    /// The menu-id suffix (`layer.smartObjects.stackMode.<id>`).
    pub fn id(self) -> &'static str {
        match self {
            StackMode::Entropy => "entropy",
            StackMode::Kurtosis => "kurtosis",
            StackMode::Maximum => "maximum",
            StackMode::Mean => "mean",
            StackMode::Median => "median",
            StackMode::Minimum => "minimum",
            StackMode::Range => "range",
            StackMode::Skewness => "skewness",
            StackMode::StandardDeviation => "standardDeviation",
            StackMode::Summation => "summation",
            StackMode::Variance => "variance",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.id() == id)
    }

    pub fn label(self) -> &'static str {
        match self {
            StackMode::Entropy => "Entropy",
            StackMode::Kurtosis => "Kurtosis",
            StackMode::Maximum => "Maximum",
            StackMode::Mean => "Mean",
            StackMode::Median => "Median",
            StackMode::Minimum => "Minimum",
            StackMode::Range => "Range",
            StackMode::Skewness => "Skewness",
            StackMode::StandardDeviation => "Standard Deviation",
            StackMode::Summation => "Summation",
            StackMode::Variance => "Variance",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_table_nearest() {
        let t = ColorTable { colors: vec![[0, 0, 0], [255, 255, 255], [255, 0, 0]], transparent: None };
        assert_eq!(t.nearest([0.9, 0.1, 0.1]), 2);
        assert_eq!(t.nearest([0.6, 0.6, 0.6]), 1);
        assert_eq!(t.rgb(2), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn duotone_render_black_ink_is_gray() {
        let d = Duotone { inks: vec![DuotoneInk::new("Black", [0.0; 3])], psd_raw: None };
        for g in [0.0, 0.25, 1.0] {
            let c = d.render(g);
            assert!((c[0] - g).abs() < 1e-6 && c[0] == c[2]);
        }
        let d = Duotone { inks: vec![DuotoneInk::new("Black", [0.0; 3]), DuotoneInk::new("Orange", [1.0, 0.5, 0.0])], psd_raw: None };
        let c = d.render(0.5);
        assert!(c[0] > c[2], "{c:?}");
    }

    #[test]
    fn stack_mode_ids_roundtrip() {
        for m in StackMode::ALL {
            assert_eq!(StackMode::from_id(m.id()), Some(m));
        }
    }
}
