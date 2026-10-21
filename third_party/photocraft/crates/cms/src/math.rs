//! Small colour-science helpers: 3×3 matrices, CIE XYZ ↔ L*a*b*, chromatic adaptation.
//!
//! All formulas are from public standards: CIE 15 (L*a*b*), ICC.1:2010 Annex A (PCS, D50),
//! and the Bradford cone response matrix published by Lam (1985).

/// The ICC profile connection space illuminant (D50), as encoded in s15Fixed16 in every
/// ICC header: X = 0.9642, Y = 1.0, Z = 0.8249.
pub const D50: [f64; 3] = [0.9642, 1.0, 0.8249];

/// CIE D65 white point as XYZ (Y = 1), from the chromaticity (0.3127, 0.3290) used by sRGB,
/// Display P3 and Adobe RGB-compatible spaces.
pub const D65_XY: [f64; 2] = [0.3127, 0.3290];

/// Row-major 3×3 matrix.
pub type Mat3 = [[f64; 3]; 3];

pub const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

pub fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut o = [[0.0; 3]; 3];
    for (i, row) in o.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    o
}

pub fn apply(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2], m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2], m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]]
}

pub fn det(m: &Mat3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Inverse, or `None` when (nearly) singular.
pub fn invert(m: &Mat3) -> Option<Mat3> {
    let d = det(m);
    if !d.is_finite() || d.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / d;
    Some([
        [(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv, (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv, (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv],
        [(m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv, (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv, (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv],
        [(m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv, (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv, (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv],
    ])
}

/// XYZ (Y = 1) of a chromaticity `(x, y)`.
pub fn xy_to_xyz(xy: [f64; 2]) -> [f64; 3] {
    let [x, y] = xy;
    [x / y, 1.0, (1.0 - x - y) / y]
}

/// Bradford cone response matrix (Lam 1985).
const BRADFORD: Mat3 = [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]];

/// Bradford chromatic adaptation matrix from white `src` to white `dst` (both XYZ).
pub fn bradford(src: [f64; 3], dst: [f64; 3]) -> Mat3 {
    // `BRADFORD` is a constant with determinant ~1.7 (see `bradford_maps_white`).
    #[allow(clippy::expect_used)]
    let inv = invert(&BRADFORD).expect("Bradford matrix is invertible");
    let s = apply(&BRADFORD, src);
    let d = apply(&BRADFORD, dst);
    let scale = [[d[0] / s[0], 0.0, 0.0], [0.0, d[1] / s[1], 0.0], [0.0, 0.0, d[2] / s[2]]];
    mul(&inv, &mul(&scale, &BRADFORD))
}

/// RGB → XYZ matrix (relative to `white`, Y = 1) for primaries given as chromaticities, or
/// `None` when the primaries are not independent.
pub fn rgb_to_xyz_matrix(r: [f64; 2], g: [f64; 2], b: [f64; 2], white: [f64; 3]) -> Option<Mat3> {
    let (xr, xg, xb) = (xy_to_xyz(r), xy_to_xyz(g), xy_to_xyz(b));
    let m = [[xr[0], xg[0], xb[0]], [xr[1], xg[1], xb[1]], [xr[2], xg[2], xb[2]]];
    let s = apply(&invert(&m)?, white);
    Some([[m[0][0] * s[0], m[0][1] * s[1], m[0][2] * s[2]], [m[1][0] * s[0], m[1][1] * s[1], m[1][2] * s[2]], [m[2][0] * s[0], m[2][1] * s[1], m[2][2] * s[2]]])
}

const EPS: f64 = 216.0 / 24389.0; // (6/29)^3
const KAPPA: f64 = 24389.0 / 27.0; // (29/3)^3

#[inline]
fn lab_f(t: f64) -> f64 {
    if t > EPS { t.cbrt() } else { (KAPPA * t + 16.0) / 116.0 }
}

#[inline]
fn lab_finv(f: f64) -> f64 {
    let f3 = f * f * f;
    if f3 > EPS { f3 } else { (116.0 * f - 16.0) / KAPPA }
}

/// CIE XYZ → L*a*b* relative to `white` (CIE 15).
pub fn xyz_to_lab(xyz: [f64; 3], white: [f64; 3]) -> [f64; 3] {
    let fx = lab_f(xyz[0] / white[0]);
    let fy = lab_f(xyz[1] / white[1]);
    let fz = lab_f(xyz[2] / white[2]);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// CIE L*a*b* → XYZ relative to `white`.
pub fn lab_to_xyz(lab: [f64; 3], white: [f64; 3]) -> [f64; 3] {
    let fy = (lab[0] + 16.0) / 116.0;
    let fx = fy + lab[1] / 500.0;
    let fz = fy - lab[2] / 200.0;
    [lab_finv(fx) * white[0], lab_finv(fy) * white[1], lab_finv(fz) * white[2]]
}

/// f32 versions for the hot paths (D50 PCS only).
#[inline]
pub fn xyz_to_lab_f32(xyz: [f32; 3]) -> [f32; 3] {
    const W: [f32; 3] = [0.9642, 1.0, 0.8249];
    const E: f32 = 216.0 / 24389.0;
    const K: f32 = 24389.0 / 27.0;
    let f = |t: f32| if t > E { t.cbrt() } else { (K * t + 16.0) / 116.0 };
    let fx = f(xyz[0] / W[0]);
    let fy = f(xyz[1] / W[1]);
    let fz = f(xyz[2] / W[2]);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

#[inline]
pub fn lab_to_xyz_f32(lab: [f32; 3]) -> [f32; 3] {
    const W: [f32; 3] = [0.9642, 1.0, 0.8249];
    const E: f32 = 216.0 / 24389.0;
    const K: f32 = 24389.0 / 27.0;
    let fy = (lab[0] + 16.0) / 116.0;
    let fx = fy + lab[1] / 500.0;
    let fz = fy - lab[2] / 200.0;
    let finv = |f: f32| {
        let f3 = f * f * f;
        if f3 > E { f3 } else { (116.0 * f - 16.0) / K }
    };
    [finv(fx) * W[0], finv(fy) * W[1], finv(fz) * W[2]]
}

/// CIE 1976 colour difference.
pub fn delta_e76(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Quantizes to ICC s15Fixed16Number and back (what a profile can actually store).
pub fn s15f16_round(v: f64) -> f64 {
    (v * 65536.0).round() / 65536.0
}

/// The D50 illuminant exactly as an ICC header stores it (s15Fixed16).
pub fn d50_quantized() -> [f64; 3] {
    [s15f16_round(D50[0]), s15f16_round(D50[1]), s15f16_round(D50[2])]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lab_roundtrip() {
        for xyz in [[0.2, 0.3, 0.1], [0.9642, 1.0, 0.8249], [0.001, 0.002, 0.0005]] {
            let back = lab_to_xyz(xyz_to_lab(xyz, D50), D50);
            for i in 0..3 {
                assert!((back[i] - xyz[i]).abs() < 1e-12);
            }
        }
        let w = xyz_to_lab(D50, D50);
        assert!((w[0] - 100.0).abs() < 1e-12 && w[1].abs() < 1e-12 && w[2].abs() < 1e-12);
    }

    #[test]
    fn bradford_maps_white() {
        let d65 = xy_to_xyz(D65_XY);
        let m = bradford(d65, D50);
        let w = apply(&m, d65);
        for i in 0..3 {
            assert!((w[i] - D50[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn invert_roundtrip() {
        let m = [[0.4, 0.3, 0.2], [0.2, 0.7, 0.1], [0.0, 0.1, 0.8]];
        let p = mul(&m, &invert(&m).unwrap());
        for (i, row) in p.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                assert!((v - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
    }
}
