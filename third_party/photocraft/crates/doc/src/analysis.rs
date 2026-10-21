//! Image › Analysis data stored with a document (measurement scale, count groups, the ruler line)
//! and annotations (Notes). Pure data; the measuring itself lives in the engine.

use photocraft_color::Color;
use serde::{Deserialize, Serialize};

/// Image › Analysis › Set Measurement Scale: `pixel_length` pixels equal `logical_length`
/// `units`. The default is 1 pixel = 1 pixel.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MeasurementScale {
    pub pixel_length: f64,
    pub logical_length: f64,
    pub units: String,
}

impl Default for MeasurementScale {
    fn default() -> Self {
        Self { pixel_length: 1.0, logical_length: 1.0, units: "pixels".into() }
    }
}

impl MeasurementScale {
    /// Logical units per pixel.
    pub fn factor(&self) -> f64 {
        if self.pixel_length > 0.0 { self.logical_length / self.pixel_length } else { 1.0 }
    }
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
    /// Photoshop's "Scale" column text, e.g. `1 pixels = 1.0000 pixels`.
    pub fn describe(&self) -> String {
        format!("{} pixels = {:.4} {}", trim(self.pixel_length), self.logical_length, self.units)
    }
}

fn trim(v: f64) -> String {
    if v.fract() == 0.0 { format!("{}", v as i64) } else { format!("{v}") }
}

/// One Count tool group: its numbered markers (document pixel coordinates, in click order).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CountGroup {
    pub name: String,
    pub color: Color,
    /// Marker (dot) size and label font size, in screen pixels (Photoshop: 1–10 and 8–72).
    pub marker_size: u32,
    pub label_size: u32,
    pub visible: bool,
    pub points: Vec<[f64; 2]>,
}

impl Default for CountGroup {
    fn default() -> Self {
        Self { name: "Count Group 1".into(), color: Color::rgb(1.0, 0.0, 0.0), marker_size: 1, label_size: 8, visible: true, points: Vec::new() }
    }
}

/// The Ruler tool line, optionally with a protractor second arm (⌥-drag from an end point).
/// `start` is the vertex the angle is measured at when there is a protractor.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Ruler {
    pub start: [f64; 2],
    pub end: [f64; 2],
    #[serde(default)]
    pub protractor: Option<[f64; 2]>,
}

/// Measurement settings and marks of a document.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Measurement {
    pub scale: MeasurementScale,
    pub count_groups: Vec<CountGroup>,
    /// Index into `count_groups` that new markers go to.
    pub active_count_group: usize,
    pub ruler: Option<Ruler>,
}

impl Measurement {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
    /// Total markers over every group.
    pub fn count_total(&self) -> usize {
        self.count_groups.iter().map(|g| g.points.len()).sum()
    }
}

/// A Note tool annotation (PSD `Anno` text annotation).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Note {
    pub author: String,
    pub color: Color,
    pub text: String,
    /// Icon position (top-left, document pixels).
    pub position: [f64; 2],
    /// Popup rectangle [left, top, right, bottom] (document pixels), as stored in PSD.
    pub popup: [f64; 4],
    /// Popup shown open.
    pub open: bool,
    /// Modification date as PDF date text (`D:YYYYMMDDhhmmss…`); empty when unknown.
    pub modified: String,
}

impl Default for Note {
    fn default() -> Self {
        // Photoshop's default note colour (pale yellow).
        Self {
            author: String::new(),
            color: Color::rgb(1.0, 1.0, 0.51),
            text: String::new(),
            position: [0.0, 0.0],
            popup: [0.0, 0.0, 0.0, 0.0],
            open: false,
            modified: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_factor_and_text() {
        let s = MeasurementScale { pixel_length: 100.0, logical_length: 2.5, units: "cm".into() };
        assert!((s.factor() - 0.025).abs() < 1e-12);
        assert_eq!(s.describe(), "100 pixels = 2.5000 cm");
        assert!(MeasurementScale::default().is_default());
        assert_eq!(MeasurementScale { pixel_length: 0.0, ..Default::default() }.factor(), 1.0);
    }

    #[test]
    fn serde_defaults_fill_missing_fields() {
        let m: Measurement = serde_json::from_str("{}").unwrap();
        assert!(m.is_empty());
        let n: Note = serde_json::from_str(r#"{"text":"hi"}"#).unwrap();
        assert_eq!(n.text, "hi");
        let g = CountGroup { points: vec![[1.0, 2.0], [3.0, 4.0]], ..Default::default() };
        let m = Measurement { count_groups: vec![g.clone(), g], ..Default::default() };
        assert_eq!(m.count_total(), 4);
    }
}
