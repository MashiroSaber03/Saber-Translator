//! Small numeric helpers shared by the computational-photography modules (panorama, HDR, lens,
//! wide angle, Camera Raw): dense and sparse least squares, parallel row loops, sampling.

/// Solves the dense `n × n` system `a · x = b` (row-major) by Gaussian elimination with partial
/// pivoting. `None` when singular.
pub(crate) fn solve_dense(n: usize, mut a: Vec<f64>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i * n + col].abs().total_cmp(&a[j * n + col].abs()))?;
        if a[piv * n + col].abs() < 1e-14 {
            return None;
        }
        if piv != col {
            for k in 0..n {
                a.swap(col * n + k, piv * n + k);
            }
            b.swap(col, piv);
        }
        let d = a[col * n + col];
        for r in col + 1..n {
            let f = a[r * n + col] / d;
            if f == 0.0 {
                continue;
            }
            for k in col..n {
                a[r * n + k] -= f * a[col * n + k];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|k| a[r * n + k] * x[k]).sum();
        x[r] = (b[r] - s) / a[r * n + r];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Accumulates weighted rows into normal equations `AᵀA x = Aᵀb` for a dense least-squares
/// problem with `n` unknowns.
pub(crate) struct Normal {
    pub n: usize,
    pub ata: Vec<f64>,
    pub atb: Vec<f64>,
}

impl Normal {
    pub fn new(n: usize) -> Self {
        Normal { n, ata: vec![0.0; n * n], atb: vec![0.0; n] }
    }
    /// Adds the sparse row `Σ coef·x[idx] = rhs` with weight `w` (applied squared).
    pub fn add(&mut self, row: &[(usize, f64)], rhs: f64, w: f64) {
        let w2 = w * w;
        for &(i, a) in row {
            self.atb[i] += w2 * a * rhs;
            for &(j, b) in row {
                self.ata[i * self.n + j] += w2 * a * b;
            }
        }
    }
    pub fn solve(self) -> Option<Vec<f64>> {
        solve_dense(self.n, self.ata, self.atb)
    }
}

/// A sparse least-squares problem `min Σ w²(row·x − rhs)²`, solved by conjugate gradients on the
/// normal equations (CGLS). Used for mesh warps with thousands of unknowns.
#[derive(Default)]
pub(crate) struct SparseLs {
    pub n: usize,
    rows: Vec<(Vec<(usize, f64)>, f64)>,
}

impl SparseLs {
    pub fn new(n: usize) -> Self {
        SparseLs { n, rows: Vec::new() }
    }
    pub fn add(&mut self, row: Vec<(usize, f64)>, rhs: f64, w: f64) {
        self.rows.push((row.into_iter().map(|(i, c)| (i, c * w)).collect(), rhs * w));
    }
    /// CGLS from `x0`; at most `iters` iterations.
    pub fn solve(&self, x0: &[f64], iters: usize) -> Vec<f64> {
        let n = self.n;
        let mut x = x0.to_vec();
        let ax = |x: &[f64]| -> Vec<f64> { self.rows.iter().map(|(r, _)| r.iter().map(|&(i, c)| c * x[i]).sum()).collect() };
        let atv = |v: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0; n];
            for ((r, _), vi) in self.rows.iter().zip(v) {
                for &(i, c) in r {
                    out[i] += c * vi;
                }
            }
            out
        };
        let mut r: Vec<f64> = self.rows.iter().zip(ax(&x)).map(|((_, b), a)| b - a).collect();
        let mut s = atv(&r);
        let mut p = s.clone();
        let mut gamma: f64 = s.iter().map(|v| v * v).sum();
        let g0 = gamma.max(1e-30);
        for _ in 0..iters {
            if gamma <= g0 * 1e-20 {
                break;
            }
            let q = ax(&p);
            let qq: f64 = q.iter().map(|v| v * v).sum();
            if qq <= 1e-300 {
                break;
            }
            let alpha = gamma / qq;
            for i in 0..n {
                x[i] += alpha * p[i];
            }
            for (ri, qi) in r.iter_mut().zip(&q) {
                *ri -= alpha * qi;
            }
            s = atv(&r);
            let g1: f64 = s.iter().map(|v| v * v).sum();
            let beta = g1 / gamma;
            gamma = g1;
            for i in 0..n {
                p[i] = s[i] + beta * p[i];
            }
        }
        x
    }
}

/// Runs `f(y, row)` for every row of a `w × h × ch` buffer, in parallel on native targets.
pub(crate) fn par_rows<T: Send + Sync>(buf: &mut [T], w: usize, ch: usize, f: impl Fn(usize, &mut [T]) + Sync + Send) {
    let stride = (w * ch).max(1);
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        buf.par_chunks_mut(stride).enumerate().for_each(|(y, row)| f(y, row));
    }
    #[cfg(target_arch = "wasm32")]
    buf.chunks_mut(stride).enumerate().for_each(|(y, row)| f(y, row));
}

/// Like [`par_rows`] but over two row-aligned buffers in lockstep (e.g. index + pixels).
pub(crate) fn par_rows2<A: Send, B: Send>(a: &mut [A], b: &mut [B], w: usize, f: impl Fn(usize, &mut [A], &mut [B]) + Sync + Send) {
    let s = w.max(1);
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        a.par_chunks_mut(s).zip(b.par_chunks_mut(s)).enumerate().for_each(|(y, (ar, br))| f(y, ar, br));
    }
    #[cfg(target_arch = "wasm32")]
    a.chunks_mut(s).zip(b.chunks_mut(s)).enumerate().for_each(|(y, (ar, br))| f(y, ar, br));
}

/// Maps `f` over `0..n` (in parallel on native targets), keeping order.
pub(crate) fn par_map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        (0..n).into_par_iter().map(f).collect()
    }
    #[cfg(target_arch = "wasm32")]
    (0..n).map(f).collect()
}

/// Deterministic 64-bit generator (SplitMix64).
pub(crate) struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

/// Integer hash → `0..1` (position-keyed noise, independent of tiling).
pub(crate) fn hash01(x: i64, y: i64, seed: u64) -> f32 {
    let mut z = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ seed.wrapping_mul(0x1656_67B1_9E37_79F9);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 40) as f32 / (1u64 << 24) as f32
}

/// sRGB transfer function, decode (extended linearly below 0 and as a power above 1).
pub(crate) fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

/// sRGB transfer function, encode.
pub(crate) fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

/// Bilinear sample of channel `c` of a `w × h × ch` buffer, clamped to the edges.
pub(crate) fn bilinear(buf: &[f32], w: usize, h: usize, ch: usize, c: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (tx, ty) = (x - x0 as f32, y - y0 as f32);
    let p = |xx: usize, yy: usize| buf[(yy * w + xx) * ch + c];
    let top = p(x0, y0) + (p(x1, y0) - p(x0, y0)) * tx;
    let bot = p(x0, y1) + (p(x1, y1) - p(x0, y1)) * tx;
    top + (bot - top) * ty
}

/// Catmull-Rom weights for a fractional offset `t`.
pub(crate) fn catmull_rom(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_and_sparse_solvers_agree() {
        // Overdetermined line fit y = 2x + 1.
        let mut ne = Normal::new(2);
        let mut sp = SparseLs::new(2);
        for x in 0..10 {
            let y = 2.0 * x as f64 + 1.0;
            ne.add(&[(0, x as f64), (1, 1.0)], y, 1.0);
            sp.add(vec![(0, x as f64), (1, 1.0)], y, 1.0);
        }
        let a = ne.solve().unwrap();
        let b = sp.solve(&[0.0, 0.0], 50);
        assert!((a[0] - 2.0).abs() < 1e-9 && (a[1] - 1.0).abs() < 1e-9);
        assert!((b[0] - 2.0).abs() < 1e-6 && (b[1] - 1.0).abs() < 1e-6, "{b:?}");
        assert!(solve_dense(2, vec![1.0, 2.0, 2.0, 4.0], vec![1.0, 2.0]).is_none());
    }

    #[test]
    fn srgb_round_trip_and_sampling() {
        for v in [0.0f32, 0.01, 0.2, 0.5, 1.0] {
            assert!((linear_to_srgb(srgb_to_linear(v)) - v).abs() < 1e-5);
        }
        let buf = [0.0f32, 1.0, 2.0, 3.0];
        assert!((bilinear(&buf, 2, 2, 1, 0, 0.5, 0.5) - 1.5).abs() < 1e-6);
        let w = catmull_rom(0.3);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let a = hash01(3, 4, 1);
        assert!((0.0..1.0).contains(&a) && a == hash01(3, 4, 1) && a != hash01(4, 3, 1));
    }
}
