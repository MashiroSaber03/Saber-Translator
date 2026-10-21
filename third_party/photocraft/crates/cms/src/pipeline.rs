//! Float evaluation pipelines: a list of stages mapping normalised device values to the
//! profile connection space (or back). This is the exact reference path; [`crate::Transform`]
//! builds faster lookup tables from it.

use crate::clut::Clut;
use crate::curve::Curve;
use crate::math;

/// Largest channel count flowing through a pipeline.
pub const MAX: usize = 16;

/// One processing element.
#[derive(Clone, Debug, PartialEq)]
pub enum Stage {
    /// Per-channel curves.
    Curves(Vec<Curve>),
    /// Per-channel inverse curves.
    InvCurves(Vec<Curve>),
    /// `out[r] = Σ m[r][c]·in[c] + offset[r]`, `rows × cols`.
    Matrix {
        rows: usize,
        cols: usize,
        m: Vec<f64>,
        offset: Vec<f64>,
    },
    Clut(Clut),
    /// Float CIE Lab (L 0–100) → XYZ (D50, Y = 1).
    LabToXyz,
    XyzToLab,
    /// ICC v4 / 8-bit Lab encoding (normalised) → float Lab, and back.
    DecodeLabV4,
    EncodeLabV4,
    /// ICC v2 legacy 16-bit Lab encoding (used by `lut16Type`) → float Lab, and back.
    DecodeLabV2,
    EncodeLabV2,
    /// u1Fixed15 XYZ encoding (normalised over 0..=0xFFFF) → XYZ, and back.
    DecodeXyz,
    EncodeXyz,
    /// Clamp every channel to `[0, 1]`.
    Clamp01,
}

const XYZ_SCALE: f32 = 65535.0 / 32768.0;
const V2_L: f32 = 100.0 * 65535.0 / 65280.0;
const V2_AB: f32 = 255.0 * 65535.0 / 65280.0;

impl Stage {
    /// Output channel count given `n` inputs.
    pub fn outputs(&self, n: usize) -> usize {
        match self {
            Stage::Matrix { rows, .. } => *rows,
            Stage::Clut(c) => c.outputs,
            _ => n,
        }
    }

    pub fn matrix3(m: &math::Mat3) -> Stage {
        Stage::Matrix { rows: 3, cols: 3, m: m.iter().flatten().copied().collect(), offset: vec![0.0; 3] }
    }

    #[inline]
    fn run(&self, v: &mut [f32; MAX], n: usize) -> usize {
        match self {
            Stage::Curves(cs) => {
                for (x, c) in v.iter_mut().zip(cs) {
                    *x = c.eval(*x);
                }
                n
            }
            Stage::InvCurves(cs) => {
                for (x, c) in v.iter_mut().zip(cs) {
                    *x = c.eval_inverse(*x);
                }
                n
            }
            Stage::Matrix { rows, cols, m, offset } => {
                let mut o = [0.0f32; MAX];
                for r in 0..*rows {
                    let mut s = offset[r];
                    for c in 0..*cols {
                        s += m[r * cols + c] * v[c] as f64;
                    }
                    o[r] = s as f32;
                }
                v[..*rows].copy_from_slice(&o[..*rows]);
                *rows
            }
            Stage::Clut(c) => {
                let mut o = [0.0f32; MAX];
                let mut inp = [0.0f32; MAX];
                inp[..n].copy_from_slice(&v[..n]);
                c.eval(&inp[..c.inputs], &mut o[..c.outputs]);
                v[..c.outputs].copy_from_slice(&o[..c.outputs]);
                c.outputs
            }
            Stage::LabToXyz => {
                let x = math::lab_to_xyz_f32([v[0], v[1], v[2]]);
                v[..3].copy_from_slice(&x);
                3
            }
            Stage::XyzToLab => {
                let l = math::xyz_to_lab_f32([v[0].max(0.0), v[1].max(0.0), v[2].max(0.0)]);
                v[..3].copy_from_slice(&l);
                3
            }
            Stage::DecodeLabV4 => {
                v[0] *= 100.0;
                v[1] = v[1] * 255.0 - 128.0;
                v[2] = v[2] * 255.0 - 128.0;
                3
            }
            Stage::EncodeLabV4 => {
                v[0] = (v[0] / 100.0).clamp(0.0, 1.0);
                v[1] = ((v[1] + 128.0) / 255.0).clamp(0.0, 1.0);
                v[2] = ((v[2] + 128.0) / 255.0).clamp(0.0, 1.0);
                3
            }
            Stage::DecodeLabV2 => {
                v[0] *= V2_L;
                v[1] = v[1] * V2_AB - 128.0;
                v[2] = v[2] * V2_AB - 128.0;
                3
            }
            Stage::EncodeLabV2 => {
                v[0] = (v[0] / V2_L).clamp(0.0, 1.0);
                v[1] = ((v[1] + 128.0) / V2_AB).clamp(0.0, 1.0);
                v[2] = ((v[2] + 128.0) / V2_AB).clamp(0.0, 1.0);
                3
            }
            Stage::DecodeXyz => {
                for x in &mut v[..3] {
                    *x *= XYZ_SCALE;
                }
                3
            }
            Stage::EncodeXyz => {
                for x in &mut v[..3] {
                    *x = (*x / XYZ_SCALE).clamp(0.0, 1.0);
                }
                3
            }
            Stage::Clamp01 => {
                for x in &mut v[..n] {
                    *x = x.clamp(0.0, 1.0);
                }
                n
            }
        }
    }
}

/// A sequence of stages with fixed input and output channel counts.
#[derive(Clone, Debug, PartialEq)]
pub struct Pipeline {
    pub inputs: usize,
    pub outputs: usize,
    pub stages: Vec<Stage>,
}

impl Pipeline {
    pub fn new(inputs: usize, stages: Vec<Stage>) -> Self {
        let outputs = stages.iter().fold(inputs, |n, s| s.outputs(n));
        Pipeline { inputs, outputs, stages }
    }

    /// Appends stages.
    pub fn extend(&mut self, stages: impl IntoIterator<Item = Stage>) {
        for s in stages {
            self.outputs = s.outputs(self.outputs);
            self.stages.push(s);
        }
    }

    /// Evaluates one pixel: `input` has `inputs` values, `out` receives `outputs` values.
    #[inline]
    pub fn eval(&self, input: &[f32], out: &mut [f32]) {
        let mut v = [0.0f32; MAX];
        v[..self.inputs].copy_from_slice(&input[..self.inputs]);
        let mut n = self.inputs;
        for s in &self.stages {
            n = s.run(&mut v, n);
        }
        out[..self.outputs].copy_from_slice(&v[..self.outputs]);
    }

    /// Removes no-op stages and cancels adjacent inverse pairs (Lab↔XYZ, encode/decode).
    pub fn optimize(&mut self) {
        let mut out: Vec<Stage> = Vec::with_capacity(self.stages.len());
        for s in self.stages.drain(..) {
            let noop = match &s {
                Stage::Curves(c) | Stage::InvCurves(c) => c.iter().all(Curve::is_identity),
                Stage::Matrix { rows: 3, cols: 3, m, offset } => {
                    offset.iter().all(|o| o.abs() < 1e-12) && m.iter().enumerate().all(|(i, v)| (v - if i % 4 == 0 { 1.0 } else { 0.0 }).abs() < 1e-12)
                }
                _ => false,
            };
            if noop {
                continue;
            }
            let cancels = matches!(
                (out.last(), &s),
                (Some(Stage::LabToXyz), Stage::XyzToLab)
                    | (Some(Stage::XyzToLab), Stage::LabToXyz)
                    | (Some(Stage::EncodeLabV4), Stage::DecodeLabV4)
                    | (Some(Stage::EncodeLabV2), Stage::DecodeLabV2)
                    | (Some(Stage::EncodeXyz), Stage::DecodeXyz)
            );
            // Decode→encode of the same encoding is only exact inside the gamut of the encoding,
            // which always holds for data coming out of a table, so those pairs cancel too.
            let cancels_dec = matches!(
                (out.last(), &s),
                (Some(Stage::DecodeLabV4), Stage::EncodeLabV4) | (Some(Stage::DecodeLabV2), Stage::EncodeLabV2) | (Some(Stage::DecodeXyz), Stage::EncodeXyz)
            );
            if cancels || cancels_dec {
                out.pop();
                continue;
            }
            // Merge consecutive 3×3 matrices.
            if let (Some(Stage::Matrix { rows: 3, cols: 3, m: a, offset: oa }), Stage::Matrix { rows: 3, cols: 3, m: b, offset: ob }) = (out.last(), &s) {
                let mut m = vec![0.0; 9];
                let mut off = vec![0.0; 3];
                for r in 0..3 {
                    for c in 0..3 {
                        m[r * 3 + c] = (0..3).map(|k| b[r * 3 + k] * a[k * 3 + c]).sum();
                    }
                    off[r] = (0..3).map(|k| b[r * 3 + k] * oa[k]).sum::<f64>() + ob[r];
                }
                out.pop();
                out.push(Stage::Matrix { rows: 3, cols: 3, m, offset: off });
                continue;
            }
            out.push(s);
        }
        self.stages = out;
    }
}
