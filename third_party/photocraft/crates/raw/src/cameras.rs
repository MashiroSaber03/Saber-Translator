//! Camera profiles measured by PhotoCraft, for raw formats that carry no colour description of
//! their own (DNG does; Nikon NEF does not).
//!
//! Clean-room: every number here comes from our own measurements of the camera's output, never
//! from another converter's tables. A profile gives the image area inside the sensor data (the
//! camera's own JPEG crop), the saturation level, a ForwardMatrix (white-balanced camera → XYZ D50,
//! DNG semantics) and a default tone curve (like DNG ProfileToneCurve).
//!
//! **Nikon D4** (148 uncompressed NEFs from one body, 2023–2026, ISO 100–10000): the sensor data is
//! 4992 × 3292; the camera's full-size JPEG (4928 × 3280) sits at (8, 6), found by aligning the two
//! (2 px of optical black on the left, 50 + 8 on the right show as a purple stripe otherwise). The
//! data is already black-subtracted (the darkest image pixels reach 0). Green saturates at
//! 15546–15672 while red and blue reach 15687–16383, so one level of 15520 makes clipped
//! highlights white without shifting the white balance. Matrix, curve and black point copy
//! Photoshop Camera Raw's default rendering of these files, measured black-box: 29 NEFs opened
//! in Photoshop 25.4 against our own linear development of the same data (8 × 8 means, pixels
//! clipped in neither). The matrix is our first ForwardMatrix (fitted to the camera's embedded
//! JPEGs) times a linear-ProPhoto look matrix whose rows sum to 1; the curve applies per channel
//! after it; the black point follows the frame's darkest tones ([`crate::tone`]). Median
//! luminance difference to Camera Raw: 0.04 stop (0.29 with the camera-JPEG curve).
//!
//! Bodies whose full profile is not measured yet can still get their measured image area — the
//! same alignment method, see [`IMAGE_AREAS`] — so their frames crop like the camera's JPEG.

use crate::color::{Calibration, ColorInfo, IDENTITY, Mat3};
use crate::sensor::{Rect, Sensor};

const D4_FORWARD: Mat3 = [[0.639454, 0.260359, 0.0643991], [0.273094, 0.796492, -0.0695779], [0.00965321, -0.131427, 0.94699]];
/// A third of the 0.1th percentile of the luminance, as Camera Raw renders D4 NEFs by default.
const D4_BLACK_POINT: f32 = 0.33;
const D4_COLOR: Mat3 = [[0.895936, -0.218092, -0.185988], [-1.09736, 2.06898, -0.0131889], [-0.26977, 0.393884, 0.687829]];
const D4_TONE: [[f32; 2]; 53] = [
    [0.00012207031, 7.44377e-05],
    [0.00014516688, 8.44049e-05],
    [0.00017263349, 9.57067e-05],
    [0.00020529698, 0.000108523],
    [0.00024414062, 0.000123055],
    [0.00029033376, 0.000123202],
    [0.00034526698, 0.000123348],
    [0.00041059396, 0.000142387],
    [0.00048828125, 0.000164366],
    [0.0005806675, 0.00020246],
    [0.00069053395, 0.000249382],
    [0.0008211879, 0.000310083],
    [0.0009765625, 0.000385559],
    [0.001161335, 0.000474115],
    [0.0013810679, 0.00058301],
    [0.0016423758, 0.000717142],
    [0.001953125, 0.000882133],
    [0.00232267, 0.00109054],
    [0.0027621358, 0.00134818],
    [0.0032847517, 0.00163655],
    [0.00390625, 0.0019866],
    [0.00464534, 0.00235945],
    [0.0055242716, 0.00280228],
    [0.0065695033, 0.00342616],
    [0.0078125, 0.00418894],
    [0.00929068, 0.00540942],
    [0.011048543, 0.0069855],
    [0.013139007, 0.00959286],
    [0.015625, 0.0131734],
    [0.01858136, 0.0181471],
    [0.022097087, 0.0249987],
    [0.026278013, 0.0336416],
    [0.03125, 0.0452726],
    [0.03716272, 0.0591889],
    [0.044194173, 0.077383],
    [0.052556027, 0.0990869],
    [0.0625, 0.126878],
    [0.07432544, 0.161688],
    [0.088388346, 0.206049],
    [0.10511205, 0.256563],
    [0.125, 0.31946],
    [0.14865088, 0.382637],
    [0.17677669, 0.458308],
    [0.2102241, 0.531942],
    [0.25, 0.617405],
    [0.29730177, 0.683279],
    [0.35355338, 0.756182],
    [0.4204482, 0.809073],
    [0.5, 0.865664],
    [0.59460354, 0.900869],
    [std::f32::consts::FRAC_1_SQRT_2, 0.937507],
    [0.8408964, 0.956434],
    [1.0, 1.0],
];

struct Profile {
    make: &'static str,
    model: &'static str,
    /// Sensor data size the profile was measured on.
    size: (usize, usize),
    /// The image area: x, y, width, height.
    crop: (usize, usize, usize, usize),
    white: f32,
    /// The black level is known to be 0 (the data is black-subtracted in camera).
    black_zero: bool,
    forward: Mat3,
    color: Mat3,
    tone: &'static [[f32; 2]],
    /// Camera Raw's adaptive black-point factor ([`Sensor::black_point`]).
    black_point: f32,
}

const PROFILES: &[Profile] = &[Profile {
    make: "NIKON CORPORATION",
    model: "NIKON D4",
    size: (4992, 3292),
    crop: (8, 6, 4928, 3280),
    white: 15520.0,
    black_zero: true,
    forward: D4_FORWARD,
    color: D4_COLOR,
    tone: &D4_TONE,
    black_point: D4_BLACK_POINT,
}];

/// A measured image area of a body whose remaining profile has not been
/// measured yet. Measured the same way as the profiles: by aligning the decoded
/// plane with the camera's own JPEG.
///
/// Both origins are even, so the Bayer phase of the crop is unchanged.
/// * **D800E** (2 files): the camera's JPEG sits 8 x 6 pixels in from the left
///   and top of the 7424 x 4924 sensor data (56 and 4 pixels of margin on the
///   right and bottom), the same corner as the D4.
/// * **D5100** (3 files): 8 x 8 pixels in from the left and top of the
///   4992 x 3280 sensor data, giving the camera JPEG's 4928 x 3264 frame.
/// * **ILCE-7** (2 files): 12 x 12 pixels in from the left and top of the
///   6048 x 4024 cRAW data (36 and 12 pixels of margin on the right and bottom).
struct ImageArea {
    make: &'static str,
    model: &'static str,
    /// Sensor data size the measurement was taken on.
    size: (usize, usize),
    /// The image area: x, y, width, height.
    crop: (usize, usize, usize, usize),
}

const IMAGE_AREAS: &[ImageArea] = &[
    ImageArea { make: "NIKON CORPORATION", model: "NIKON D5100", size: (4992, 3280), crop: (8, 8, 4928, 3264) },
    ImageArea { make: "NIKON CORPORATION", model: "NIKON D800E", size: (7424, 4924), crop: (8, 6, 7360, 4912) },
    ImageArea { make: "SONY", model: "ILCE-7", size: (6048, 4024), crop: (12, 12, 6000, 4000) },
];

/// Applies the measured profile of the camera that made `s`, when there is one,
/// and its measured image area when the profile is not measured yet.
pub(crate) fn apply(s: &mut Sensor) {
    let (Some(make), Some(model)) = (s.make.as_deref(), s.model.as_deref()) else { return };
    let (make, model) = (make.trim(), model.trim());
    if let Some(a) = IMAGE_AREAS.iter().find(|a| a.make == make && a.model == model && a.size == (s.width, s.height) && s.crop == s.active) {
        let (x, y, w, h) = a.crop;
        s.crop = Rect::new(x, y, w, h);
    }
    let Some(p) = PROFILES.iter().find(|p| make == p.make && model == p.model && (s.width, s.height) == p.size) else { return };
    let (x, y, w, h) = p.crop;
    s.crop = Rect::new(x, y, w, h);
    s.white = [p.white; 3];
    if p.black_zero {
        s.warnings.retain(|w| !w.contains("black level not recorded"));
    }
    s.color = ColorInfo {
        // D50 daylight: the matrices are consistent at the camera's daylight neutral.
        calibrations: vec![Calibration { temperature: 5003.0, color_matrix: p.color, forward_matrix: Some(p.forward), camera_calibration: IDENTITY }],
        ..ColorInfo::default()
    };
    s.tone_curve = p.tone.to_vec();
    s.black_point = p.black_point;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RawFormat;
    use crate::color::apply as mat_apply;
    use crate::sensor::BlackLevels;

    fn sensor(make: &str, model: &str, width: usize, height: usize) -> Sensor {
        Sensor {
            format: RawFormat::Nef,
            make: Some(make.into()),
            model: Some(model.into()),
            width,
            height,
            samples: 1,
            data: Vec::new(),
            cfa: None,
            linearization: None,
            black: BlackLevels::uniform(0.0),
            white: [16383.0; 3],
            active: Rect::new(0, 0, width, height),
            crop: Rect::new(0, 0, width, height),
            color: ColorInfo::default(),
            camera_wb: Some([2.0, 1.0, 1.4]),
            orientation: 1,
            baseline_exposure: 0.0,
            gain_maps: Vec::new(),
            tone_curve: Vec::new(),
            black_point: 0.0,
            warnings: vec!["NEF: black level not recorded in a documented tag; assumed 0".into()],
        }
    }

    fn nef(model: &str, width: usize, height: usize) -> Sensor {
        sensor("NIKON CORPORATION", model, width, height)
    }

    #[test]
    fn a_d4_nef_gets_its_crop_levels_matrix_and_curve() {
        let mut s = nef("NIKON D4", 4992, 3292);
        apply(&mut s);
        assert_eq!(s.crop, Rect::new(8, 6, 4928, 3280));
        assert_eq!(s.white, [15520.0; 3]);
        assert_eq!(s.color.calibrations.len(), 1);
        assert_eq!(s.color.calibrations[0].forward_matrix, Some(D4_FORWARD));
        assert_eq!(s.tone_curve.len(), D4_TONE.len());
        assert_eq!(s.black_point, D4_BLACK_POINT);
        assert!(s.warnings.is_empty(), "the D4's black level is known: {:?}", s.warnings);
        // The profile gives a colour conversion for the as-shot white balance.
        assert!(s.color.balanced_to_xyz_d50([0.5, 1.0, 0.7]).is_some());
    }

    #[test]
    fn d5100_image_area_respects_explicit_crops_and_exact_sensor_size() {
        let mut s = nef("NIKON D5100", 4992, 3280);
        // The NEF reader trims eight columns of row padding before applying the model area.
        s.active = Rect::new(0, 0, 4984, 3280);
        s.crop = s.active;
        let wb = s.camera_wb;
        apply(&mut s);
        assert_eq!(s.crop, Rect::new(8, 8, 4928, 3264));
        assert_eq!(s.active, Rect::new(0, 0, 4984, 3280));
        assert_eq!(s.camera_wb, wb);
        assert!(s.color.calibrations.is_empty() && s.tone_curve.is_empty());
        assert_eq!(s.white, [16383.0; 3]);

        s.crop = Rect::new(16, 12, 4800, 3200);
        apply(&mut s);
        assert_eq!(s.crop, Rect::new(16, 12, 4800, 3200));
        for (model, w, h) in [("NIKON D5100", 2496, 1640), ("NIKON D5200", 4992, 3280)] {
            let mut other = nef(model, w, h);
            apply(&mut other);
            assert_eq!(other.crop, other.active);
        }
    }

    #[test]
    fn measured_image_areas_crop_the_borders_without_touching_the_rest() {
        let mut s = nef("NIKON D800E", 7424, 4924);
        apply(&mut s);
        assert_eq!(s.crop, Rect::new(8, 6, 7360, 4912));
        // Only the image area is measured: levels, colour and tone stay untouched.
        assert!(s.color.calibrations.is_empty() && s.tone_curve.is_empty());
        assert_eq!(s.white, [16383.0; 3]);

        let mut s = sensor("SONY", "ILCE-7", 6048, 4024);
        apply(&mut s);
        assert_eq!(s.crop, Rect::new(12, 12, 6000, 4000));

        // Another size or model of the same maker is left alone.
        let mut s = nef("NIKON D800E", 2496, 1646);
        apply(&mut s);
        assert_eq!(s.crop, Rect::new(0, 0, 2496, 1646));
        let mut s = nef("NIKON D800", 7424, 4924);
        apply(&mut s);
        assert_eq!(s.crop, Rect::new(0, 0, 7424, 4924));
    }

    #[test]
    fn other_cameras_and_sizes_are_left_alone() {
        for (model, w, h) in [("NIKON D4S", 4992, 3292), ("NIKON D4", 2496, 1646), ("NIKON D800", 7424, 4924)] {
            let mut s = nef(model, w, h);
            apply(&mut s);
            assert_eq!(s.crop, Rect::new(0, 0, w, h), "{model} {w}x{h}");
            assert!(s.tone_curve.is_empty() && s.color.calibrations.is_empty() && s.white == [16383.0; 3]);
            assert_eq!(s.warnings.len(), 1);
        }
    }

    #[test]
    fn d4_forward_matrix_maps_neutral_to_d50_and_curve_reaches_white() {
        let n = mat_apply(&D4_FORWARD, [1.0; 3]);
        assert!((n[0] - 0.9642).abs() < 1e-3 && (n[1] - 1.0).abs() < 1e-3 && (n[2] - 0.8252).abs() < 1e-3, "{n:?}");
        assert!(D4_TONE.windows(2).all(|w| w[1][0] > w[0][0] && w[1][1] >= w[0][1]), "increasing");
        assert_eq!(D4_TONE.last().map(|p| (p[0], p[1])), Some((1.0, 1.0)));
    }
}
