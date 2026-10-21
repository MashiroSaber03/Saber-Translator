//! Colour science for raw development: 3×3 matrix helpers, chromaticity
//! conversions, correlated colour temperature, Bradford adaptation and the
//! camera → XYZ (D50) → output-space mapping of the DNG specification
//! (Adobe DNG Specification 1.7, chapter 6 "Mapping Camera Color Space to
//! CIE XYZ Space").

pub type Mat3 = [[f64; 3]; 3];

pub const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// CIE D50 white (the ICC PCS illuminant), XYZ with Y = 1.
pub const D50: [f64; 3] = [0.9642, 1.0, 0.8249];
/// D50 chromaticity.
pub const D50_XY: [f64; 2] = [0.3457, 0.3585];
/// D65 chromaticity.
pub const D65_XY: [f64; 2] = [0.3127, 0.3290];

pub fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut o = [[0.0; 3]; 3];
    for (i, row) in o.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    o
}

pub fn apply(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

pub fn diag(v: [f64; 3]) -> Mat3 {
    [[v[0], 0.0, 0.0], [0.0, v[1], 0.0], [0.0, 0.0, v[2]]]
}

pub fn scale(m: &Mat3, k: f64) -> Mat3 {
    m.map(|r| r.map(|v| v * k))
}

/// Inverse, or `None` when (nearly) singular or not finite.
pub fn invert(m: &Mat3) -> Option<Mat3> {
    let [[a, b, c], [d, e, f], [g, h, i]] = *m;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if !det.is_finite() || det.abs() < 1e-12 {
        return None;
    }
    let r = 1.0 / det;
    let o = [
        [(e * i - f * h) * r, (c * h - b * i) * r, (b * f - c * e) * r],
        [(f * g - d * i) * r, (a * i - c * g) * r, (c * d - a * f) * r],
        [(d * h - e * g) * r, (b * g - a * h) * r, (a * e - b * d) * r],
    ];
    o.iter().flatten().all(|v| v.is_finite()).then_some(o)
}

/// A row-major 3×3 matrix from 9 values.
pub fn from_slice(v: &[f64]) -> Option<Mat3> {
    if v.len() < 9 || v[..9].iter().any(|x| !x.is_finite()) {
        return None;
    }
    Some([[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]])
}

pub fn xy_to_xyz(xy: [f64; 2]) -> [f64; 3] {
    let y = xy[1].max(1e-6);
    [xy[0] / y, 1.0, (1.0 - xy[0] - xy[1]) / y]
}

pub fn xyz_to_xy(v: [f64; 3]) -> Option<[f64; 2]> {
    let s = v[0] + v[1] + v[2];
    (s.is_finite() && s > 1e-9).then(|| [v[0] / s, v[1] / s])
}

/// Correlated colour temperature from chromaticity (McCamy 1992, "Correlated
/// color temperature as an explicit function of chromaticity coordinates",
/// Color Res. Appl. 17(2)); clamped to 2000–50000 K.
pub fn cct(xy: [f64; 2]) -> f64 {
    let n = (xy[0] - 0.3320) / (0.1858 - xy[1]);
    let t = 449.0 * n.powi(3) + 3525.0 * n.powi(2) + 6823.3 * n + 5520.33;
    if t.is_finite() { t.clamp(2000.0, 50000.0) } else { 5000.0 }
}

/// Approximate temperature of an EXIF `LightSource` code (the
/// CalibrationIlluminant tags use the EXIF 2.3 table). `None` for unknown.
pub fn illuminant_temperature(code: u32) -> Option<f64> {
    Some(match code {
        17 => 2856.0,        // Standard light A
        18 => 4874.0,        // Standard light B
        19 => 6774.0,        // Standard light C
        20 => 5503.0,        // D55
        21 => 6504.0,        // D65
        22 => 7504.0,        // D75
        23 => 5003.0,        // D50
        24 => 3200.0,        // ISO studio tungsten
        3 => 2850.0,         // Tungsten
        1 | 4 | 9 => 5500.0, // Daylight, flash, fine weather
        10 => 6500.0,        // Cloudy
        11 => 7500.0,        // Shade
        12 => 6430.0,        // Daylight fluorescent
        13 => 5000.0,        // Day white fluorescent
        14 => 4150.0,        // Cool white fluorescent
        15 => 3525.0,        // White fluorescent
        16 => 2925.0,        // Warm white fluorescent
        2 => 4200.0,         // Fluorescent
        _ => return None,
    })
}

/// Bradford chromatic adaptation from white `src` to white `dst` (XYZ).
pub fn bradford(src: [f64; 3], dst: [f64; 3]) -> Mat3 {
    const M: Mat3 = [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]];
    let Some(mi) = invert(&M) else { return IDENTITY };
    let s = apply(&M, src);
    let d = apply(&M, dst);
    if s.iter().any(|v| v.abs() < 1e-9) {
        return IDENTITY;
    }
    mul(&mi, &mul(&diag([d[0] / s[0], d[1] / s[1], d[2] / s[2]]), &M))
}

/// RGB → XYZ matrix for primaries and a white point (all as chromaticities),
/// white mapping to Y = 1.
pub fn rgb_to_xyz(prim: [[f64; 2]; 3], white: [f64; 2]) -> Mat3 {
    let cols = prim.map(xy_to_xyz);
    let p: Mat3 = [[cols[0][0], cols[1][0], cols[2][0]], [cols[0][1], cols[1][1], cols[2][1]], [cols[0][2], cols[1][2], cols[2][2]]];
    let Some(pi) = invert(&p) else { return IDENTITY };
    let s = apply(&pi, xy_to_xyz(white));
    mul(&p, &diag(s))
}

/// ROMM RGB (ProPhoto) primaries, ISO 22028-2; white D50.
pub const ROMM: [[f64; 2]; 3] = [[0.7347, 0.2653], [0.1596, 0.8404], [0.0366, 0.0001]];
/// sRGB / Rec. 709 primaries.
pub const REC709: [[f64; 2]; 3] = [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]];

/// XYZ (D50) → linear ProPhoto RGB.
pub fn xyz_d50_to_prophoto() -> Mat3 {
    invert(&rgb_to_xyz(ROMM, D50_XY)).unwrap_or(IDENTITY)
}

/// Linear sRGB (D65) → XYZ adapted to D50: the documented neutral camera
/// matrix used when a file carries no colour calibration.
pub fn srgb_to_xyz_d50() -> Mat3 {
    mul(&bradford(xy_to_xyz(D65_XY), xy_to_xyz(D50_XY)), &rgb_to_xyz(REC709, D65_XY))
}

/// One DNG calibration (ColorMatrixN with its companions).
#[derive(Debug, Clone, PartialEq)]
pub struct Calibration {
    /// Illuminant temperature in kelvin.
    pub temperature: f64,
    /// XYZ → reference camera (ColorMatrixN).
    pub color_matrix: Mat3,
    /// White-balanced camera → XYZ D50 (ForwardMatrixN).
    pub forward_matrix: Option<Mat3>,
    /// Reference camera → individual camera (CameraCalibrationN).
    pub camera_calibration: Mat3,
}

/// Colour description of a raw file.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ColorInfo {
    /// DNG calibrations (0, 1 or 2), sorted by temperature.
    pub calibrations: Vec<Calibration>,
    /// AnalogBalance.
    pub analog_balance: Option<[f64; 3]>,
    /// AsShotNeutral: camera coordinates of a neutral.
    pub as_shot_neutral: Option<[f64; 3]>,
    /// AsShotWhiteXY.
    pub as_shot_white_xy: Option<[f64; 2]>,
}

fn lerp_mat(a: &Mat3, b: &Mat3, g: f64) -> Mat3 {
    let mut o = *a;
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = a[i][j] * g + b[i][j] * (1.0 - g);
        }
    }
    o
}

impl ColorInfo {
    /// Weight of the first calibration at temperature `t` (inverse-temperature
    /// interpolation, DNG spec chapter 6).
    fn weight(&self, t: f64) -> f64 {
        match self.calibrations.as_slice() {
            [a, b] if (a.temperature - b.temperature).abs() > 1.0 => {
                let (t1, t2) = (a.temperature, b.temperature);
                if t <= t1 {
                    1.0
                } else if t >= t2 {
                    0.0
                } else {
                    ((1.0 / t - 1.0 / t2) / (1.0 / t1 - 1.0 / t2)).clamp(0.0, 1.0)
                }
            }
            _ => 1.0,
        }
    }

    /// The interpolated `(AB · CC · CM, ForwardMatrix, AB · CC)` at temperature `t`.
    fn at(&self, t: f64) -> Option<(Mat3, Option<Mat3>, Mat3)> {
        let first = self.calibrations.first()?;
        let g = self.weight(t);
        let pick = |f: &dyn Fn(&Calibration) -> Mat3| match self.calibrations.get(1) {
            Some(b) => lerp_mat(&f(first), &f(b), g),
            None => f(first),
        };
        let cm = pick(&|c| c.color_matrix);
        let cc = pick(&|c| c.camera_calibration);
        let fm = match (first.forward_matrix, self.calibrations.get(1).map(|c| c.forward_matrix)) {
            (Some(a), Some(Some(b))) => Some(lerp_mat(&a, &b, g)),
            (Some(a), None) => Some(a),
            _ => None,
        };
        let ab = diag(self.analog_balance.unwrap_or([1.0; 3]));
        let abcc = mul(&ab, &cc);
        Some((mul(&abcc, &cm), fm, abcc))
    }

    /// Chromaticity of the white whose camera coordinates are `neutral`,
    /// by fixed-point iteration over the temperature-dependent matrix.
    pub fn neutral_to_xy(&self, neutral: [f64; 3]) -> [f64; 2] {
        let mut xy = D50_XY;
        for _ in 0..30 {
            let Some((xyz_to_cam, _, _)) = self.at(cct(xy)) else { return xy };
            let Some(cam_to_xyz) = invert(&xyz_to_cam) else { return xy };
            let Some(next) = xyz_to_xy(apply(&cam_to_xyz, neutral)) else { return xy };
            if !(0.0..1.0).contains(&next[0]) || !(0.0..1.0).contains(&next[1]) {
                return xy;
            }
            let done = (next[0] - xy[0]).abs() + (next[1] - xy[1]).abs() < 1e-7;
            xy = next;
            if done {
                break;
            }
        }
        xy
    }

    /// Camera coordinates of a white with chromaticity `xy`, max component 1.
    pub fn xy_to_neutral(&self, xy: [f64; 2]) -> Option<[f64; 3]> {
        let (xyz_to_cam, _, _) = self.at(cct(xy))?;
        let n = apply(&xyz_to_cam, xy_to_xyz(xy));
        let m = n.iter().cloned().fold(f64::MIN, f64::max);
        (m > 1e-9 && n.iter().all(|v| v.is_finite() && *v > 0.0)).then(|| n.map(|v| v / m))
    }

    /// The matrix from white-balanced camera values (camera values divided by
    /// `neutral / max(neutral)`) to XYZ D50, scaled so that the balanced
    /// white (1, 1, 1) maps to D50 with Y = 1. `None` without calibrations.
    pub fn balanced_to_xyz_d50(&self, neutral: [f64; 3]) -> Option<Mat3> {
        let nmax = neutral.iter().cloned().fold(f64::MIN, f64::max);
        if nmax.is_nan() || nmax <= 0.0 || neutral.iter().any(|v| v.is_nan() || *v <= 0.0) {
            return None;
        }
        let xy = self.neutral_to_xy(neutral);
        let (xyz_to_cam, fm, abcc) = self.at(cct(xy))?;
        let unbalance = diag(neutral.map(|v| v / nmax));
        let m = match fm {
            Some(fm) => {
                // CameraToXYZ_D50 = FM · D · Inverse(AB · CC), D = Invert(diag(ReferenceNeutral)).
                let abcc_inv = invert(&abcc)?;
                let reference = apply(&abcc_inv, neutral);
                if reference.iter().any(|v| v.is_nan() || *v <= 0.0) {
                    return None;
                }
                let d = diag(reference.map(|v| 1.0 / v));
                mul(&mul(&fm, &mul(&d, &abcc_inv)), &unbalance)
            }
            None => {
                let cam_to_xyz = invert(&xyz_to_cam)?;
                let adapt = bradford(xy_to_xyz(xy), D50);
                mul(&mul(&adapt, &cam_to_xyz), &unbalance)
            }
        };
        // Normalize: balanced white → Y = 1.
        let y = apply(&m, [1.0; 3])[1];
        (y.is_finite() && y > 1e-9).then(|| scale(&m, 1.0 / y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn inverse_roundtrip() {
        let m = srgb_to_xyz_d50();
        let i = invert(&m).unwrap();
        let p = mul(&m, &i);
        for (r, row) in p.iter().enumerate() {
            for (c, v) in row.iter().enumerate() {
                assert!(close(*v, if r == c { 1.0 } else { 0.0 }, 1e-9));
            }
        }
        assert!(invert(&[[0.0; 3]; 3]).is_none());
    }

    #[test]
    fn white_points_map_to_d50() {
        let w = apply(&srgb_to_xyz_d50(), [1.0; 3]);
        assert!(close(w[0], 0.9642, 2e-3) && close(w[1], 1.0, 1e-9) && close(w[2], 0.8249, 2e-3), "{w:?}");
        let p = apply(&xyz_d50_to_prophoto(), xy_to_xyz(D50_XY));
        assert!(p.iter().all(|v| close(*v, 1.0, 1e-9)), "{p:?}");
    }

    #[test]
    fn mccamy_reference_points() {
        assert!(close(cct(D65_XY), 6504.0, 15.0));
        assert!(close(cct([0.44757, 0.40745]), 2856.0, 15.0)); // illuminant A
    }

    #[test]
    fn dng_color_matrix_roundtrip() {
        // A synthetic camera whose XYZ → camera matrix is the inverse of sRGB: with a
        // D65 neutral, the balanced pipeline must reproduce sRGB → XYZ D50.
        let cam_to_xyz = rgb_to_xyz(REC709, D65_XY);
        let cm = invert(&cam_to_xyz).unwrap();
        let info = ColorInfo {
            calibrations: vec![Calibration { temperature: 6504.0, color_matrix: cm, forward_matrix: None, camera_calibration: IDENTITY }],
            ..Default::default()
        };
        let neutral = info.xy_to_neutral(D65_XY).unwrap();
        assert!(neutral.iter().all(|v| close(*v, 1.0, 1e-3)), "{neutral:?}");
        let xy = info.neutral_to_xy(neutral);
        assert!(close(xy[0], D65_XY[0], 1e-4) && close(xy[1], D65_XY[1], 1e-4));
        let m = info.balanced_to_xyz_d50(neutral).unwrap();
        let w = apply(&m, [1.0; 3]);
        assert!(close(w[0], 0.9642, 3e-3) && close(w[1], 1.0, 1e-9) && close(w[2], 0.8249, 3e-3), "{w:?}");
    }
}
