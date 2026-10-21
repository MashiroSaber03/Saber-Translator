//! Multi-dimensional colour lookup tables (ICC CLUTs) with tetrahedral interpolation.
//!
//! Grid layout follows ICC.1: the first input channel varies slowest, output values of one
//! node are stored together. 3-input tables use tetrahedral interpolation (Sakamoto 1980 /
//! Kasson et al. 1995); 4-input tables interpolate linearly along the first input between
//! two tetrahedral evaluations; other input counts use multilinear interpolation.

/// Maximum number of input or output channels of a CLUT.
pub const MAX_CHANNELS: usize = 15;

#[derive(Clone, Debug, PartialEq)]
pub struct Clut {
    pub inputs: usize,
    pub outputs: usize,
    /// Grid points per input dimension (each ≥ 2, or 1 for a degenerate dimension).
    pub grid: Vec<usize>,
    /// Normalised node values, `outputs` per node.
    pub data: Vec<f32>,
    strides: Vec<usize>,
}

impl Clut {
    /// Builds a table from node data. Panics if `data` has the wrong length.
    pub fn new(grid: Vec<usize>, outputs: usize, data: Vec<f32>) -> Self {
        let inputs = grid.len();
        assert!((1..=MAX_CHANNELS).contains(&inputs) && (1..=MAX_CHANNELS).contains(&outputs), "bad CLUT dimensions");
        let mut strides = vec![0usize; inputs];
        let mut s = outputs;
        for i in (0..inputs).rev() {
            strides[i] = s;
            s *= grid[i].max(1);
        }
        assert_eq!(data.len(), s, "CLUT data length");
        Clut { inputs, outputs, grid, data, strides }
    }

    /// Number of nodes.
    pub fn nodes(&self) -> usize {
        self.data.len() / self.outputs
    }

    /// Builds a table by sampling `f` at every node (inputs normalised to `[0, 1]`).
    pub fn sample(grid: Vec<usize>, outputs: usize, mut f: impl FnMut(&[f32], &mut [f32])) -> Self {
        let inputs = grid.len();
        let total: usize = grid.iter().product();
        let mut data = vec![0.0f32; total * outputs];
        let mut idx = vec![0usize; inputs];
        let mut inp = vec![0.0f32; inputs];
        for node in 0..total {
            let mut rem = node;
            for d in (0..inputs).rev() {
                idx[d] = rem % grid[d];
                rem /= grid[d];
            }
            for d in 0..inputs {
                inp[d] = if grid[d] > 1 { idx[d] as f32 / (grid[d] - 1) as f32 } else { 0.0 };
            }
            f(&inp, &mut data[node * outputs..(node + 1) * outputs]);
        }
        Clut::new(grid, outputs, data)
    }

    #[inline]
    fn locate(&self, d: usize, v: f32) -> (usize, f32, usize) {
        let n = self.grid[d];
        if n < 2 {
            return (0, 0.0, 0);
        }
        let p = v.clamp(0.0, 1.0) * (n - 1) as f32;
        let i = (p as usize).min(n - 2);
        (i * self.strides[d], p - i as f32, self.strides[d])
    }

    /// Interpolates at `input` (`inputs` values) into `out` (`outputs` values).
    #[inline]
    pub fn eval(&self, input: &[f32], out: &mut [f32]) {
        match self.inputs {
            1 => self.eval1(input[0], out),
            3 => {
                let (b0, r0, s0) = self.locate(0, input[0]);
                let (b1, r1, s1) = self.locate(1, input[1]);
                let (b2, r2, s2) = self.locate(2, input[2]);
                self.tetra(b0 + b1 + b2, [s0, s1, s2], [r0, r1, r2], out);
            }
            4 => {
                let (b0, r0, s0) = self.locate(0, input[0]);
                let (b1, r1, s1) = self.locate(1, input[1]);
                let (b2, r2, s2) = self.locate(2, input[2]);
                let (b3, r3, s3) = self.locate(3, input[3]);
                let base = b0 + b1 + b2 + b3;
                let mut lo = [0.0f32; MAX_CHANNELS];
                self.tetra(base, [s1, s2, s3], [r1, r2, r3], &mut lo[..self.outputs]);
                if r0 > 0.0 && s0 > 0 {
                    let mut hi = [0.0f32; MAX_CHANNELS];
                    self.tetra(base + s0, [s1, s2, s3], [r1, r2, r3], &mut hi[..self.outputs]);
                    for (o, (l, h)) in out.iter_mut().zip(lo.iter().zip(&hi)) {
                        *o = l + (h - l) * r0;
                    }
                } else {
                    out[..self.outputs].copy_from_slice(&lo[..self.outputs]);
                }
            }
            _ => self.eval_multilinear(input, out),
        }
    }

    fn eval1(&self, v: f32, out: &mut [f32]) {
        let (b, r, s) = self.locate(0, v);
        for (k, o) in out.iter_mut().enumerate().take(self.outputs) {
            let a = self.data[b + k];
            let c = if s > 0 { self.data[b + s + k] } else { a };
            *o = a + (c - a) * r;
        }
    }

    /// Tetrahedral interpolation inside one cube of three dimensions.
    #[inline]
    fn tetra(&self, base: usize, s: [usize; 3], r: [f32; 3], out: &mut [f32]) {
        let d = &self.data;
        let [rx, ry, rz] = r;
        let [sx, sy, sz] = s;
        let c000 = base;
        let c100 = base + sx;
        let c010 = base + sy;
        let c001 = base + sz;
        let c110 = base + sx + sy;
        let c101 = base + sx + sz;
        let c011 = base + sy + sz;
        let c111 = base + sx + sy + sz;
        // Pick the tetrahedron containing (rx, ry, rz) and the path through the cube corners.
        let (p1, p2, p3, w1, w2, w3) = if rx >= ry {
            if ry >= rz {
                (c100, c110, c111, rx, ry, rz)
            } else if rx >= rz {
                (c100, c101, c111, rx, rz, ry)
            } else {
                (c001, c101, c111, rz, rx, ry)
            }
        } else if rx >= rz {
            (c010, c110, c111, ry, rx, rz)
        } else if ry >= rz {
            (c010, c011, c111, ry, rz, rx)
        } else {
            (c001, c011, c111, rz, ry, rx)
        };
        for (k, o) in out.iter_mut().enumerate().take(self.outputs) {
            let v0 = d[c000 + k];
            let v1 = d[p1 + k];
            let v2 = d[p2 + k];
            let v3 = d[p3 + k];
            *o = v0 + (v1 - v0) * w1 + (v2 - v1) * w2 + (v3 - v2) * w3;
        }
    }

    /// Trilinear interpolation (3 inputs), kept as a reference for accuracy comparisons.
    pub fn eval_trilinear(&self, input: &[f32], out: &mut [f32]) {
        assert_eq!(self.inputs, 3);
        self.eval_multilinear(input, out);
    }

    /// Multilinear interpolation over all 2^n cube corners.
    pub fn eval_multilinear(&self, input: &[f32], out: &mut [f32]) {
        let n = self.inputs;
        let mut base = 0usize;
        let mut r = [0.0f32; MAX_CHANNELS];
        let mut s = [0usize; MAX_CHANNELS];
        for d in 0..n {
            let (b, rr, ss) = self.locate(d, input[d]);
            base += b;
            r[d] = rr;
            s[d] = ss;
        }
        let o = self.outputs;
        out[..o].iter_mut().for_each(|v| *v = 0.0);
        for corner in 0..(1usize << n) {
            let mut w = 1.0f32;
            let mut off = base;
            for d in 0..n {
                if corner & (1 << d) != 0 {
                    w *= r[d];
                    off += s[d];
                } else {
                    w *= 1.0 - r[d];
                }
            }
            if w == 0.0 {
                continue;
            }
            for (k, v) in out[..o].iter_mut().enumerate() {
                *v += w * self.data[off + k];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f3(p: &[f32], o: &mut [f32]) {
        o[0] = p[0] * 0.5 + p[1] * 0.3 + p[2] * 0.2;
        o[1] = p[0] * p[1];
        o[2] = (p[2] * 3.0).sin() * 0.5 + 0.5;
    }

    #[test]
    fn exact_on_nodes_and_linear_functions() {
        let c = Clut::sample(vec![5, 5, 5], 3, f3);
        let mut o = [0.0; 3];
        c.eval(&[0.25, 0.5, 0.75], &mut o);
        let mut w = [0.0; 3];
        f3(&[0.25, 0.5, 0.75], &mut w);
        for i in 0..3 {
            assert!((o[i] - w[i]).abs() < 1e-6);
        }
        // Linear function is reproduced exactly anywhere.
        c.eval(&[0.13, 0.77, 0.41], &mut o);
        assert!((o[0] - (0.13 * 0.5 + 0.77 * 0.3 + 0.41 * 0.2)).abs() < 1e-6);
    }

    #[test]
    fn four_d_and_multilinear_agree_on_linear() {
        let f = |p: &[f32], o: &mut [f32]| o[0] = 0.1 * p[0] + 0.2 * p[1] + 0.3 * p[2] + 0.4 * p[3];
        let c = Clut::sample(vec![3, 4, 5, 6], 1, f);
        let mut a = [0.0];
        let mut b = [0.0];
        for p in [[0.1f32, 0.9, 0.3, 0.7], [1.0, 0.0, 0.5, 0.5], [0.33, 0.66, 0.99, 0.01]] {
            c.eval(&p, &mut a);
            c.eval_multilinear(&p, &mut b);
            let want = 0.1 * p[0] + 0.2 * p[1] + 0.3 * p[2] + 0.4 * p[3];
            assert!((a[0] - want).abs() < 1e-5 && (b[0] - want).abs() < 1e-5);
        }
    }

    #[test]
    fn one_and_two_d() {
        let c = Clut::sample(vec![3], 2, |p, o| {
            o[0] = p[0];
            o[1] = 1.0 - p[0];
        });
        let mut o = [0.0; 2];
        c.eval(&[0.25], &mut o);
        assert!((o[0] - 0.25).abs() < 1e-6 && (o[1] - 0.75).abs() < 1e-6);
        let c = Clut::sample(vec![2, 2], 1, |p, o| o[0] = p[0] + p[1]);
        c.eval(&[0.5, 0.25], &mut o);
        assert!((o[0] - 0.75).abs() < 1e-6);
    }
}
