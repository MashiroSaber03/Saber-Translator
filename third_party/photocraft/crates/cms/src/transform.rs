//! Colour transforms between profiles through the PCS, with rendering intents, black point
//! compensation, soft-proofing chains, and fast lookup-table evaluation of pixel buffers.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::clut::{Clut, MAX_CHANNELS};
use crate::curve::Curve;
use crate::math;
use crate::pipeline::{Pipeline, Stage};
use crate::profile::{ColorSpace, Pcs, Profile, ProfileClass};
use crate::{CmsError, Intent};

/// Sample storage of a byte buffer for [`Transform::convert_bytes_many`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleKind {
    U8,
    U16,
    F32,
}

/// Options for building a [`Transform`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TransformOptions {
    pub intent: Intent,
    /// Adobe-style black point compensation (ignored for absolute colorimetric).
    pub bpc: bool,
    /// `true`: float buffers go through the exact pipeline (default). `false`: float buffers use
    /// the same lookup tables as integer buffers (faster, for previews).
    pub precise_float: bool,
}

impl Default for TransformOptions {
    fn default() -> Self {
        Self { intent: Intent::RelativeColorimetric, bpc: true, precise_float: true }
    }
}

/// Grid sizes of the device-link tables built for integer (and fast float) evaluation.
pub const GRID_3D: usize = 33;
pub const GRID_4D: usize = 17;
const TABLE_1D: usize = 4096;
/// Resolution of output shaper tables (indexed by `x^(1/4)`).
const OUT_TABLE: usize = 4096;

/// ICC v4 perceptual reference medium black (ICC.1:2010 §6.3.4.4), XYZ.
const PERCEPTUAL_BLACK: [f64; 3] = [0.00336, 0.0034731, 0.00287];

/// Accelerated representation of a transform.
#[derive(Debug)]
enum Core {
    /// Input curves (sampled per input channel over `[0, 1]`, 4096+1 entries; 256 exact
    /// entries for 8-bit input) followed by an optional matrix.
    Shaper { curves8: Vec<[f32; 256]>, curves: Vec<Vec<f32>>, matrix: Option<(usize, usize, Vec<f32>, Vec<f32>)> },
    /// Device link table over the input space.
    Grid(Clut),
    /// No acceleration possible: use the exact pipeline.
    Exact,
}

/// Output inverse-curve tables indexed by `x^(1/4)` (accurate near black even for pure gamma
/// curves, whose slope is infinite at zero).
#[derive(Debug)]
struct Post {
    tables: Vec<Vec<f32>>,
}

impl Post {
    #[inline]
    fn eval(&self, c: usize, x: f32) -> f32 {
        let t = &self.tables[c];
        let u = x.clamp(0.0, 1.0).sqrt().sqrt() * (OUT_TABLE - 1) as f32;
        let i = (u as usize).min(OUT_TABLE - 2);
        let f = u - i as f32;
        t[i] + (t[i + 1] - t[i]) * f
    }
}

/// A colour transform from one profile's device space to another's.
///
/// Build once (it precomputes lookup tables) and apply to any number of buffers from any
/// thread. Buffers are interleaved; colour channels come first in each pixel and any further
/// channels (alpha, spot) are copied or left alone.
#[derive(Debug)]
pub struct Transform {
    inputs: usize,
    outputs: usize,
    opts: TransformOptions,
    pipeline: Pipeline,
    core: Core,
    post: Option<Post>,
    /// Pre-located grid coordinates per 8-bit code: (offset contribution, fraction) per input.
    grid8: Vec<[(u32, f32); 256]>,
}

impl Transform {
    /// Transform from `src` to `dst` with `intent`, optionally with black point compensation.
    pub fn new(src: &Profile, dst: &Profile, intent: Intent, bpc: bool) -> Result<Transform, CmsError> {
        Self::with_options(src, dst, TransformOptions { intent, bpc, ..Default::default() })
    }

    pub fn with_options(src: &Profile, dst: &Profile, opts: TransformOptions) -> Result<Transform, CmsError> {
        let stages = link_stages(src, dst, opts.intent, opts.bpc)?;
        let mut t = Self::from_pipeline(Pipeline::new(src.channels(), stages), opts);
        if opts.intent != Intent::AbsoluteColorimetric {
            t.fix_white(src.color_space, dst.color_space);
        }
        Ok(t)
    }

    /// Soft-proof transform `src → proof → dst` (Photoshop View › Proof Setup): `intent`/`bpc`
    /// apply to the document → proof step; the proof → display step is relative colorimetric
    /// with BPC, or absolute colorimetric without BPC when `simulate_paper` is set (paper
    /// colour and black ink simulated).
    pub fn proof(src: &Profile, proof: &Profile, dst: &Profile, intent: Intent, bpc: bool, simulate_paper: bool) -> Result<Transform, CmsError> {
        let mut stages = link_stages(src, proof, intent, bpc)?;
        stages.push(Stage::Clamp01);
        let (i2, b2) = if simulate_paper { (Intent::AbsoluteColorimetric, false) } else { (Intent::RelativeColorimetric, true) };
        stages.extend(link_stages(proof, dst, i2, b2)?);
        Ok(Self::from_pipeline(Pipeline::new(src.channels(), stages), TransformOptions { intent, bpc, precise_float: true }))
    }

    /// Builds a transform from an arbitrary pipeline over normalised values.
    pub fn from_pipeline(mut pipeline: Pipeline, opts: TransformOptions) -> Transform {
        pipeline.optimize();
        let (inputs, outputs) = (pipeline.inputs, pipeline.outputs);
        // Split trailing inverse curves into accurate output tables.
        let mut head = pipeline.stages.clone();
        let mut post = None;
        if let Some(Stage::InvCurves(cs)) = head.last()
            && cs.len() == outputs
        {
            let tables = cs
                .iter()
                .map(|c| {
                    (0..OUT_TABLE)
                        .map(|i| {
                            let u = i as f64 / (OUT_TABLE - 1) as f64;
                            c.eval_inverse64(u * u * u * u).clamp(0.0, 1.0) as f32
                        })
                        .collect()
                })
                .collect();
            post = Some(Post { tables });
            head.pop();
            if matches!(head.last(), Some(Stage::Clamp01)) {
                head.pop();
            }
        }
        let head_pipe = Pipeline::new(inputs, head.clone());
        let mid = head_pipe.outputs;
        let core = match head.as_slice() {
            [] => Core::Shaper { curves8: Vec::new(), curves: Vec::new(), matrix: None },
            [Stage::Curves(cs)] if cs.len() == inputs => shaper(cs, None),
            [Stage::Matrix { rows, cols, m, offset }] => shaper(&[], Some((*rows, *cols, m, offset))),
            [Stage::Curves(cs), Stage::Matrix { rows, cols, m, offset }] if cs.len() == inputs => shaper(cs, Some((*rows, *cols, m, offset))),
            // Purely analytic heads (Lab/XYZ maths, matrices, curves) are evaluated exactly:
            // a grid would amplify interpolation error through steep output curves.
            _ if !head.iter().any(|s| matches!(s, Stage::Clut(_))) => Core::Exact,
            _ => match inputs {
                1 => Core::Grid(Clut::sample(vec![TABLE_1D], mid, |i, o| head_pipe.eval(i, o))),
                3 => Core::Grid(Clut::sample(vec![GRID_3D; 3], mid, |i, o| head_pipe.eval(i, o))),
                4 => Core::Grid(Clut::sample(vec![GRID_4D; 4], mid, |i, o| head_pipe.eval(i, o))),
                2 => Core::Grid(Clut::sample(vec![GRID_3D; 2], mid, |i, o| head_pipe.eval(i, o))),
                _ => Core::Exact,
            },
        };
        let grid8 = match &core {
            Core::Grid(c) if c.inputs >= 2 => (0..c.inputs)
                .map(|d| {
                    let n = c.grid[d];
                    let stride = c.data.len() / c.outputs / c.grid[..=d].iter().product::<usize>() * c.outputs;
                    std::array::from_fn(|v| {
                        let p = v as f32 / 255.0 * (n - 1) as f32;
                        let i = (p as usize).min(n - 2);
                        ((i * stride) as u32, p - i as f32)
                    })
                })
                .collect(),
            _ => Vec::new(),
        };
        Transform { inputs, outputs, opts, pipeline, core, post, grid8 }
    }

    /// White preservation: the device-link node at the source's device white gets exactly the
    /// destination's device white, so paper/white survive table quantization (e.g. the PCS
    /// white is not a node of a legacy-encoded Lab grid).
    fn fix_white(&mut self, src: ColorSpace, dst: ColorSpace) {
        let (Some(sw), Some(mut dw)) = (device_white(src), device_white(dst)) else { return };
        if self.post.is_some() {
            // Output curves map 1 → 1 for additive spaces; the grid holds pre-curve values.
            if !matches!(dst, ColorSpace::Rgb | ColorSpace::Gray) {
                return;
            }
            dw.iter_mut().for_each(|v| *v = 1.0);
        }
        let Core::Grid(c) = &mut self.core else { return };
        if c.inputs != sw.len() || c.outputs != dw.len() {
            return;
        }
        let mut off = 0usize;
        let mut stride = c.outputs;
        for d in (0..c.inputs).rev() {
            let n = c.grid[d];
            let pos = sw[d] * (n - 1) as f32;
            if (pos - pos.round()).abs() > 1e-6 {
                return;
            }
            off += pos.round() as usize * stride;
            stride *= n;
        }
        c.data[off..off + c.outputs].copy_from_slice(&dw);
    }

    pub fn inputs(&self) -> usize {
        self.inputs
    }
    pub fn outputs(&self) -> usize {
        self.outputs
    }
    pub fn options(&self) -> TransformOptions {
        self.opts
    }
    /// The exact evaluation pipeline.
    pub fn pipeline(&self) -> &Pipeline {
        &self.pipeline
    }

    /// Exact evaluation of one colour (`inputs` → `outputs` normalised values).
    #[inline]
    pub fn eval(&self, input: &[f32], out: &mut [f32]) {
        self.pipeline.eval(input, out);
    }

    /// Lookup-table evaluation of one colour (what integer buffers use).
    #[inline]
    pub fn eval_fast(&self, input: &[f32], out: &mut [f32]) {
        let mut mid = [0.0f32; 16];
        match &self.core {
            Core::Exact => return self.pipeline.eval(input, out),
            Core::Grid(c) => {
                let mut inp = [0.0f32; 16];
                for (d, s) in inp.iter_mut().zip(&input[..self.inputs]) {
                    *d = s.clamp(0.0, 1.0);
                }
                c.eval(&inp[..self.inputs], &mut mid[..c.outputs]);
            }
            Core::Shaper { curves, matrix, .. } => {
                let mut v = [0.0f32; 16];
                for k in 0..self.inputs {
                    v[k] = if curves.is_empty() { input[k] } else { table(&curves[k], input[k]) };
                }
                apply_matrix(matrix, &v, &mut mid, self.inputs);
            }
        }
        self.finish(&mid, out);
    }

    #[inline]
    fn finish(&self, mid: &[f32], out: &mut [f32]) {
        match &self.post {
            Some(p) => {
                for k in 0..self.outputs {
                    out[k] = p.eval(k, mid[k]);
                }
            }
            None => out[..self.outputs].copy_from_slice(&mid[..self.outputs]),
        }
    }

    /// Converts float pixels. `src`/`dst` hold whole pixels of `src_stride`/`dst_stride`
    /// values; colour channels come first. Trailing channels are copied when `copy_extra`.
    pub fn convert_f32(&self, src: &[f32], src_stride: usize, dst: &mut [f32], dst_stride: usize, copy_extra: bool) {
        self.check(src_stride, dst_stride);
        let extra = extra_channels(self, src_stride, dst_stride, copy_extra);
        par_rows(src, src_stride, dst, dst_stride, |s, d| self.run_f32(s, src_stride, d, dst_stride, extra));
    }

    fn run_f32(&self, s: &[f32], ss: usize, d: &mut [f32], ds: usize, extra: usize) {
        let precise = self.opts.precise_float || matches!(self.core, Core::Exact);
        for (sp, dp) in s.chunks_exact(ss).zip(d.chunks_exact_mut(ds)) {
            if precise {
                self.pipeline.eval(sp, dp);
            } else {
                self.eval_fast(sp, dp);
            }
            for e in 0..extra {
                dp[self.outputs + e] = sp[self.inputs + e];
            }
        }
    }

    /// In-place float conversion; `stride ≥ max(inputs, outputs)`.
    pub fn apply(&self, buf: &mut [f32], stride: usize) {
        assert!(stride >= self.inputs.max(self.outputs), "stride too small");
        let precise = self.opts.precise_float || matches!(self.core, Core::Exact);
        let body = |chunk: &mut [f32]| {
            let mut out = [0.0f32; 16];
            for p in chunk.chunks_exact_mut(stride) {
                if precise {
                    self.pipeline.eval(p, &mut out);
                } else {
                    self.eval_fast(p, &mut out);
                }
                p[..self.outputs].copy_from_slice(&out[..self.outputs]);
            }
        };
        par_inplace(buf, stride, body);
    }

    /// Converts 8-bit pixels (see [`Transform::convert_f32`] for the layout).
    pub fn convert_u8(&self, src: &[u8], src_stride: usize, dst: &mut [u8], dst_stride: usize, copy_extra: bool) {
        self.check(src_stride, dst_stride);
        let extra = extra_channels(self, src_stride, dst_stride, copy_extra);
        par_rows(src, src_stride, dst, dst_stride, |s, d| self.run_u8(s, src_stride, d, dst_stride, extra));
    }

    fn run_u8(&self, s: &[u8], ss: usize, d: &mut [u8], ds: usize, extra: usize) {
        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let mut mid = [0.0f32; 16];
        let mut out = [0.0f32; 16];
        match &self.core {
            Core::Grid(c) if c.inputs == 3 => {
                let (g0, g1, g2) = (&self.grid8[0], &self.grid8[1], &self.grid8[2]);
                let s1 = c.outputs * c.grid[2];
                let s0 = s1 * c.grid[1];
                for (sp, dp) in s.chunks_exact(ss).zip(d.chunks_exact_mut(ds)) {
                    let (a, b, cc) = (g0[sp[0] as usize], g1[sp[1] as usize], g2[sp[2] as usize]);
                    c.tetra_pub((a.0 + b.0 + cc.0) as usize, [s0, s1, c.outputs], [a.1, b.1, cc.1], &mut mid);
                    self.finish(&mid, &mut out);
                    for k in 0..self.outputs {
                        dp[k] = q(out[k]);
                    }
                    for e in 0..extra {
                        dp[self.outputs + e] = sp[self.inputs + e];
                    }
                }
            }
            Core::Grid(c) if c.inputs == 4 => {
                let g = &self.grid8;
                let s3 = c.outputs;
                let s2 = s3 * c.grid[3];
                let s1 = s2 * c.grid[2];
                let s0 = s1 * c.grid[1];
                let mut hi = [0.0f32; 16];
                for (sp, dp) in s.chunks_exact(ss).zip(d.chunks_exact_mut(ds)) {
                    let (a, b, cc, k4) = (g[0][sp[0] as usize], g[1][sp[1] as usize], g[2][sp[2] as usize], g[3][sp[3] as usize]);
                    let base = (a.0 + b.0 + cc.0 + k4.0) as usize;
                    c.tetra_pub(base, [s1, s2, s3], [b.1, cc.1, k4.1], &mut mid);
                    if a.1 > 0.0 {
                        c.tetra_pub(base + s0, [s1, s2, s3], [b.1, cc.1, k4.1], &mut hi);
                        for k in 0..c.outputs {
                            mid[k] += (hi[k] - mid[k]) * a.1;
                        }
                    }
                    self.finish(&mid, &mut out);
                    for k in 0..self.outputs {
                        dp[k] = q(out[k]);
                    }
                    for e in 0..extra {
                        dp[self.outputs + e] = sp[self.inputs + e];
                    }
                }
            }
            Core::Shaper { curves8, matrix, .. } => {
                let mut v = [0.0f32; 16];
                for (sp, dp) in s.chunks_exact(ss).zip(d.chunks_exact_mut(ds)) {
                    for k in 0..self.inputs {
                        v[k] = if curves8.is_empty() { sp[k] as f32 / 255.0 } else { curves8[k][sp[k] as usize] };
                    }
                    apply_matrix(matrix, &v, &mut mid, self.inputs);
                    self.finish(&mid, &mut out);
                    for k in 0..self.outputs {
                        dp[k] = q(out[k]);
                    }
                    for e in 0..extra {
                        dp[self.outputs + e] = sp[self.inputs + e];
                    }
                }
            }
            _ => {
                let mut inp = [0.0f32; 16];
                for (sp, dp) in s.chunks_exact(ss).zip(d.chunks_exact_mut(ds)) {
                    for k in 0..self.inputs {
                        inp[k] = sp[k] as f32 / 255.0;
                    }
                    self.eval_fast(&inp, &mut out);
                    for k in 0..self.outputs {
                        dp[k] = q(out[k]);
                    }
                    for e in 0..extra {
                        dp[self.outputs + e] = sp[self.inputs + e];
                    }
                }
            }
        }
    }

    /// Converts 16-bit pixels (see [`Transform::convert_f32`] for the layout).
    pub fn convert_u16(&self, src: &[u16], src_stride: usize, dst: &mut [u16], dst_stride: usize, copy_extra: bool) {
        self.check(src_stride, dst_stride);
        let extra = extra_channels(self, src_stride, dst_stride, copy_extra);
        par_rows(src, src_stride, dst, dst_stride, |s, d| self.run_u16(s, src_stride, d, dst_stride, extra));
    }

    fn run_u16(&self, s: &[u16], ss: usize, d: &mut [u16], ds: usize, extra: usize) {
        let mut inp = [0.0f32; 16];
        let mut out = [0.0f32; 16];
        for (sp, dp) in s.chunks_exact(ss).zip(d.chunks_exact_mut(ds)) {
            for k in 0..self.inputs {
                inp[k] = sp[k] as f32 / 65535.0;
            }
            self.eval_fast(&inp, &mut out);
            for k in 0..self.outputs {
                dp[k] = (out[k].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
            }
            for e in 0..extra {
                dp[self.outputs + e] = sp[self.inputs + e];
            }
        }
    }

    /// Converts many independent buffers of native-endian samples in one parallel pass (e.g.
    /// all tiles of a surface): far cheaper than one parallel call per small buffer. Each job
    /// is `(src bytes, dst bytes)` holding whole pixels of `src_stride`/`dst_stride` samples.
    pub fn convert_bytes_many(&self, sample: SampleKind, jobs: Vec<(&[u8], &mut [u8])>, src_stride: usize, dst_stride: usize, copy_extra: bool) {
        self.check(src_stride, dst_stride);
        let extra = extra_channels(self, src_stride, dst_stride, copy_extra);
        let run = |(s, d): (&[u8], &mut [u8])| match sample {
            SampleKind::U8 => self.run_u8(s, src_stride, d, dst_stride, extra),
            SampleKind::U16 => {
                let a: Vec<u16> = s.as_chunks::<2>().0.iter().map(|b| u16::from_ne_bytes([b[0], b[1]])).collect();
                let mut o: Vec<u16> = d.as_chunks::<2>().0.iter().map(|b| u16::from_ne_bytes([b[0], b[1]])).collect();
                self.run_u16(&a, src_stride, &mut o, dst_stride, extra);
                for (b, v) in d.as_chunks_mut::<2>().0.iter_mut().zip(o) {
                    b.copy_from_slice(&v.to_ne_bytes());
                }
            }
            SampleKind::F32 => {
                let a: Vec<f32> = s.as_chunks::<4>().0.iter().map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]])).collect();
                let mut o: Vec<f32> = d.as_chunks::<4>().0.iter().map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]])).collect();
                self.run_f32(&a, src_stride, &mut o, dst_stride, extra);
                for (b, v) in d.as_chunks_mut::<4>().0.iter_mut().zip(o) {
                    b.copy_from_slice(&v.to_ne_bytes());
                }
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            jobs.into_par_iter().for_each(run);
        }
        #[cfg(target_arch = "wasm32")]
        jobs.into_iter().for_each(run);
    }

    fn check(&self, ss: usize, ds: usize) {
        assert!(ss >= self.inputs && ds >= self.outputs, "pixel stride smaller than the channel count");
    }
}

fn device_white(cs: ColorSpace) -> Option<Vec<f32>> {
    Some(match cs {
        ColorSpace::Rgb => vec![1.0; 3],
        ColorSpace::Gray => vec![1.0],
        ColorSpace::Cmyk => vec![0.0; 4],
        ColorSpace::Cmy => vec![0.0; 3],
        ColorSpace::Lab => vec![1.0, 0.5, 0.5],
        _ => return None,
    })
}

fn extra_channels(t: &Transform, ss: usize, ds: usize, copy: bool) -> usize {
    if copy { (ss - t.inputs).min(ds - t.outputs) } else { 0 }
}

fn shaper(cs: &[Curve], matrix: Option<(usize, usize, &Vec<f64>, &Vec<f64>)>) -> Core {
    let curves8 = cs.iter().map(|c| std::array::from_fn(|i| c.eval64(i as f64 / 255.0) as f32)).collect();
    let curves = cs.iter().map(|c| (0..=TABLE_1D).map(|i| c.eval64(i as f64 / TABLE_1D as f64) as f32).collect()).collect();
    let matrix = matrix.map(|(r, c, m, o)| (r, c, m.iter().map(|v| *v as f32).collect(), o.iter().map(|v| *v as f32).collect()));
    Core::Shaper { curves8, curves, matrix }
}

#[inline]
fn table(t: &[f32], x: f32) -> f32 {
    // Sampled over [0, 1] with TABLE_1D segments; values outside extrapolate linearly from
    // the end segments so HDR floats stay usable in the fast path.
    let n = t.len() - 1;
    let p = x * n as f32;
    let i = (p.floor().max(0.0) as usize).min(n - 1);
    let f = p - i as f32;
    t[i] + (t[i + 1] - t[i]) * f
}

#[inline]
fn apply_matrix(m: &Option<(usize, usize, Vec<f32>, Vec<f32>)>, v: &[f32; 16], out: &mut [f32; 16], n: usize) {
    match m {
        None => out[..n].copy_from_slice(&v[..n]),
        Some((rows, cols, m, off)) => {
            for r in 0..*rows {
                let mut s = off[r];
                for c in 0..*cols {
                    s += m[r * cols + c] * v[c];
                }
                out[r] = s;
            }
        }
    }
}

// ------------------------------------------------------------ parallel helpers

#[cfg(not(target_arch = "wasm32"))]
const PAR_MIN_PIXELS: usize = 16 * 1024;
#[cfg(not(target_arch = "wasm32"))]
const PAR_CHUNK_PIXELS: usize = 8 * 1024;

#[cfg(not(target_arch = "wasm32"))]
fn par_rows<S: Sync, D: Send>(src: &[S], ss: usize, dst: &mut [D], ds: usize, f: impl Fn(&[S], &mut [D]) + Sync) {
    use rayon::prelude::*;
    let n = (src.len() / ss).min(dst.len() / ds);
    let (src, dst) = (&src[..n * ss], &mut dst[..n * ds]);
    if n < PAR_MIN_PIXELS {
        return f(src, dst);
    }
    src.par_chunks(PAR_CHUNK_PIXELS * ss).zip(dst.par_chunks_mut(PAR_CHUNK_PIXELS * ds)).for_each(|(s, d)| f(s, d));
}

#[cfg(target_arch = "wasm32")]
fn par_rows<S: Sync, D: Send>(src: &[S], ss: usize, dst: &mut [D], ds: usize, f: impl Fn(&[S], &mut [D]) + Sync) {
    let n = (src.len() / ss).min(dst.len() / ds);
    f(&src[..n * ss], &mut dst[..n * ds]);
}

#[cfg(not(target_arch = "wasm32"))]
fn par_inplace(buf: &mut [f32], stride: usize, f: impl Fn(&mut [f32]) + Sync) {
    use rayon::prelude::*;
    if buf.len() / stride < PAR_MIN_PIXELS {
        return f(buf);
    }
    buf.par_chunks_mut(PAR_CHUNK_PIXELS * stride).for_each(&f);
}

#[cfg(target_arch = "wasm32")]
fn par_inplace(buf: &mut [f32], _stride: usize, f: impl Fn(&mut [f32]) + Sync) {
    f(buf)
}

// ------------------------------------------------------------ linking

/// Stages from `src` device values to `dst` device values.
pub(crate) fn link_stages(src: &Profile, dst: &Profile, intent: Intent, bpc: bool) -> Result<Vec<Stage>, CmsError> {
    for p in [src, dst] {
        // A Lab DeviceLink stands in for an abstract profile only when it also ends in Lab.
        let lab_to_lab = p.color_space == ColorSpace::Lab && p.a2b.iter().flatten().all(|l| l.outputs == 3) && p.b2a.iter().flatten().all(|l| l.inputs == 3);
        if matches!(p.class, ProfileClass::DeviceLink | ProfileClass::NamedColor | ProfileClass::Abstract) && !lab_to_lab {
            return Err(CmsError::Unsupported(format!("{:?} profiles cannot be used as a source or destination", p.class)));
        }
    }
    let (mut stages, s_pcs) = src.device_to_pcs(intent)?;
    let (d_stages, d_pcs) = dst.pcs_to_device(intent)?;
    let adjust = connection(src, dst, intent, bpc)?;
    match adjust {
        Some((scale, offset)) => {
            if s_pcs == Pcs::Lab {
                stages.push(Stage::LabToXyz);
            }
            stages.push(Stage::Matrix { rows: 3, cols: 3, m: vec![scale[0], 0.0, 0.0, 0.0, scale[1], 0.0, 0.0, 0.0, scale[2]], offset: offset.to_vec() });
            if d_pcs == Pcs::Lab {
                stages.push(Stage::XyzToLab);
            }
        }
        None => match (s_pcs, d_pcs) {
            (Pcs::Lab, Pcs::Xyz) => stages.push(Stage::LabToXyz),
            (Pcs::Xyz, Pcs::Lab) => stages.push(Stage::XyzToLab),
            _ => {}
        },
    }
    stages.extend(d_stages);
    Ok(stages)
}

/// Per-component XYZ scale and offset applied in the PCS: absolute colorimetric media-white
/// scaling or black point compensation. `None` = identity.
/// Per-component XYZ `(scale, offset)`.
type Affine3 = ([f64; 3], [f64; 3]);

fn connection(src: &Profile, dst: &Profile, intent: Intent, bpc: bool) -> Result<Option<Affine3>, CmsError> {
    if intent == Intent::AbsoluteColorimetric {
        let (ws, wd) = (src.media_white(), dst.media_white());
        // `wtpt` comes straight from the file. A zero, negative or non-finite component would
        // make the scale NaN or infinite and poison every colour the transform produces, so a
        // profile with such a media white is refused for this intent instead.
        let usable = |w: &[f64; 3]| w.iter().all(|v| v.is_finite() && *v > 0.0);
        if !usable(&ws) || !usable(&wd) {
            return Err(CmsError::Invalid(format!("absolute colorimetric needs a positive media white point (source {ws:?}, destination {wd:?})")));
        }
        let s = [ws[0] / wd[0], ws[1] / wd[1], ws[2] / wd[2]];
        if s.iter().all(|v| (v - 1.0).abs() < 1e-6) {
            return Ok(None);
        }
        return Ok(Some((s, [0.0; 3])));
    }
    if !bpc {
        return Ok(None);
    }
    let bs = black_point(src, intent, true)?;
    let bd = black_point(dst, intent, false)?;
    if (bs - bd).abs() < 1e-6 {
        return Ok(None);
    }
    // Neutral black points (luminance only, as in Adobe's BPC), mapped linearly in XYZ so that
    // the source black lands on the destination black and the white stays fixed.
    let w = math::D50;
    let mut scale = [1.0; 3];
    let mut offset = [0.0; 3];
    for i in 0..3 {
        let (s, d) = (bs * w[i], bd * w[i]);
        let a = (w[i] - d) / (w[i] - s);
        scale[i] = a;
        offset[i] = w[i] * (1.0 - a);
    }
    Ok(Some((scale, offset)))
}

/// Luminance (Y, D50-relative) of the profile's black point for `intent`.
///
/// Follows the structure of Adobe's published BPC algorithm: matrix/TRC profiles use device
/// zero; v4 LUT profiles in perceptual/saturation use the perceptual reference black; other LUT
/// profiles use the darkest colour the device reaches (Lab 0 round-tripped through BToA/AToB,
/// or device black for additive spaces). The destination-side curve fitting of the paper is
/// not implemented; the round-trip black is used directly.
pub fn black_point(p: &Profile, intent: Intent, _input: bool) -> Result<f64, CmsError> {
    if matches!(p.color_space, ColorSpace::Lab | ColorSpace::Xyz) {
        return Ok(0.0);
    }
    let lut_based = p.a2b.iter().any(Option::is_some);
    if lut_based && p.version.0 >= 4 && matches!(intent, Intent::Perceptual | Intent::Saturation) {
        return Ok(PERCEPTUAL_BLACK[1]);
    }
    let pcs_y = |st: Vec<Stage>, pcs: Pcs, dev: &[f32]| -> f64 {
        let pipe = Pipeline::new(p.channels(), st);
        let mut o = [0.0f32; 16];
        pipe.eval(dev, &mut o);
        let y = match pcs {
            Pcs::Xyz => o[1] as f64,
            Pcs::Lab => math::lab_to_xyz([o[0] as f64, 0.0, 0.0], math::D50)[1],
        };
        y.clamp(0.0, 0.5)
    };
    let (to_pcs, pcs) = p.device_to_pcs(intent)?;
    let additive = matches!(p.color_space, ColorSpace::Rgb | ColorSpace::Gray);
    if additive || !lut_based {
        let zero = vec![0.0f32; p.channels()];
        return Ok(pcs_y(to_pcs, pcs, &zero));
    }
    // Subtractive LUT profile: darkest reachable colour.
    let colorimetric = if intent == Intent::Saturation { Intent::Saturation } else { Intent::RelativeColorimetric };
    match p.pcs_to_device(colorimetric) {
        Ok((from_pcs, fpcs)) => {
            let mut st = Vec::new();
            if fpcs == Pcs::Xyz {
                st.push(Stage::LabToXyz);
            }
            st.extend(from_pcs);
            let pipe = Pipeline::new(3, st);
            let mut dev = [0.0f32; 16];
            pipe.eval(&[0.0, 0.0, 0.0], &mut dev);
            Ok(pcs_y(to_pcs, pcs, &dev[..p.channels()]))
        }
        Err(_) => {
            let ones = vec![1.0f32; p.channels()];
            Ok(pcs_y(to_pcs, pcs, &ones))
        }
    }
}

// ------------------------------------------------------------ cache

type Key = (u64, u64, TransformOptions);

/// Process-wide cache of transforms keyed by profile content and options.
pub fn cached(src: &Profile, dst: &Profile, opts: TransformOptions) -> Result<Arc<Transform>, CmsError> {
    static CACHE: OnceLock<Mutex<HashMap<Key, Arc<Transform>>>> = OnceLock::new();
    let key = (src.content_hash(), dst.content_hash(), opts);
    let cache = CACHE.get_or_init(Default::default);
    if let Some(t) = cache.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return Ok(t.clone());
    }
    let t = Arc::new(Transform::with_options(src, dst, opts)?);
    let mut c = cache.lock().unwrap_or_else(|e| e.into_inner());
    if c.len() > 64 {
        c.clear();
    }
    c.insert(key, t.clone());
    Ok(t)
}

impl Clut {
    /// Tetrahedral interpolation in one cube (crate-internal fast paths).
    #[inline]
    pub(crate) fn tetra_pub(&self, base: usize, s: [usize; 3], r: [f32; 3], out: &mut [f32; 16]) {
        let d = &self.data;
        let [rx, ry, rz] = r;
        let [sx, sy, sz] = s;
        let (p1, p2, p3, w1, w2, w3) = if rx >= ry {
            if ry >= rz {
                (sx, sx + sy, sx + sy + sz, rx, ry, rz)
            } else if rx >= rz {
                (sx, sx + sz, sx + sy + sz, rx, rz, ry)
            } else {
                (sz, sx + sz, sx + sy + sz, rz, rx, ry)
            }
        } else if rx >= rz {
            (sy, sx + sy, sx + sy + sz, ry, rx, rz)
        } else if ry >= rz {
            (sy, sy + sz, sx + sy + sz, ry, rz, rx)
        } else {
            (sz, sy + sz, sx + sy + sz, rz, ry, rx)
        };
        let o = self.outputs.min(MAX_CHANNELS);
        for k in 0..o {
            let v0 = d[base + k];
            let v1 = d[base + p1 + k];
            let v2 = d[base + p2 + k];
            let v3 = d[base + p3 + k];
            out[k] = v0 + (v1 - v0) * w1 + (v2 - v1) * w2 + (v3 - v2) * w3;
        }
    }
}
