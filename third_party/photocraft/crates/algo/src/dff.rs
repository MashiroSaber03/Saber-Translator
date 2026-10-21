//! Depth from focus for Edit › Auto-Blend Layers › Stack Images: a dense, sub-layer depth map
//! from the aligned stack, the index of the layer each pixel is sharpest in.
//!
//! The pipeline is the non-learned "focus volume" recipe, written from the papers:
//!
//! 1. **Focus measure** per layer on luminance at full resolution: the *ring difference filter*
//!    of Jeon, Surh, Im & Kweon, *Ring Difference Filter for Fast and Noise Robust Depth From
//!    Focus*, IEEE TIP 29 (2019), the magnitude of the difference between the mean of a small
//!    disk and the mean of the ring around it, a band-pass whose support averages sensor noise
//!    away while staying local. Block-averaged to a working grid (half resolution).
//! 2. **Cost aggregation**: each slice of the focus volume is filtered with the *guided filter*
//!    (He, Sun & Tang, PAMI 2013) using the all-in-focus luminance as guide, edge-aware
//!    aggregation as in fast cost-volume filtering (Hosni et al., PAMI 2013).
//! 3. **Peak search** per pixel over the layer axis, one slice at a time: the global peak and its
//!    two neighbours for the *Gaussian interpolation* of Nayar & Nakagawa, *Shape from Focus*,
//!    PAMI 1994 (sub-layer depth), the second-best local maximum for a peak-ratio confidence,
//!    the profile mean for a prominence term and the profile minimum for a noise gate.
//! 4. **Regularisation**: the sub-layer depth is smoothed and holes filled by an edge-aware
//!    weighted least squares energy (Farbman, Fattal, Lischinski & Szeliski, SIGGRAPH 2008)
//!    whose data term is the confidence, solved by conjugate gradients preconditioned by a
//!    multigrid V-cycle from the separable sweeps of the *fast global smoother* (Min, Choi, Lu,
//!    Ham, Sohn & Do, IEEE TIP 2014) as the start, with one Huber reweighting so isolated
//!    outliers stop pulling.
//! 5. **Upsampling** to full resolution with the guided filter's linear coefficients.
//!
//! Everything is deterministic; the slices are folded one at a time, so memory is one working
//! grid per layer plus a few planes.

use crate::photo_util::{par_map, par_rows, par_rows2};

/// Depth-from-focus settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthParams {
    /// The working grid is `1 / 2^scale` of full resolution (1 = half).
    pub scale: usize,
    /// Ring difference filter radii: the disk `r ≤ r_in`, the ring `r_in < r ≤ r_out`.
    pub r_in: usize,
    pub r_out: usize,
    /// Guided-filter aggregation radius (working-grid pixels) and regulariser.
    pub agg_radius: usize,
    pub agg_eps: f32,
    /// Smoothness weight and guide-edge sensitivity (luminance units) of the regularisation.
    pub lambda: f32,
    pub sigma_c: f32,
    /// Noise gate: full confidence needs a peak ≥ `(1 + gate) ×` the noise floor (the median
    /// over pixels of the profile minimum). 0 = off.
    pub gate: f32,
    /// Robust data term: after the first solve the data weights are scaled by the Huber factor
    /// `min(1, robust / |d − u|)` (in layers) and the system is solved once more. 0 = off.
    pub robust: f32,
}

impl Default for DepthParams {
    fn default() -> Self {
        DepthParams { scale: 1, r_in: 1, r_out: 3, agg_radius: 3, agg_eps: 1e-4, lambda: 3.0, sigma_c: 0.04, gate: 1.0, robust: 1.0 }
    }
}

/// The result of [`depth_from_focus`].
#[derive(Clone, Debug, PartialEq)]
pub struct DepthMap {
    /// Full-resolution fractional layer index per pixel, in `0 ..= n − 1`.
    pub depth: Vec<f32>,
    /// Full-resolution confidence in `0 ..= 1`.
    pub conf: Vec<f32>,
    pub w: usize,
    pub h: usize,
}

/// Mean over `k × k` blocks (partial blocks at the far edges use their own count).
pub fn block_mean(src: &[f32], w: usize, h: usize, k: usize) -> (Vec<f32>, usize, usize) {
    let k = k.max(1);
    let (dw, dh) = (w.div_ceil(k), h.div_ceil(k));
    if k == 1 {
        return (src.to_vec(), dw, dh);
    }
    let mut out = vec![0.0f32; dw * dh];
    par_rows(&mut out, dw, 1, |oy, row| {
        let (y0, y1) = (oy * k, (oy * k + k).min(h));
        for y in y0..y1 {
            let s = &src[y * w..y * w + w];
            for (ox, o) in row.iter_mut().enumerate() {
                *o += s[ox * k..(ox * k + k).min(w)].iter().sum::<f32>();
            }
        }
        for (ox, o) in row.iter_mut().enumerate() {
            *o /= ((y1 - y0) * ((ox * k + k).min(w) - ox * k)) as f32;
        }
    });
    (out, dw, dh)
}

/// Bilinear resampling of a working-grid plane (block size `k`, samples at block centres) to
/// `w × h`.
pub fn upsample_bilinear(g: &[f32], dw: usize, dh: usize, w: usize, h: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    if dw == 0 || dh == 0 {
        return out;
    }
    let inv = 1.0 / k.max(1) as f32;
    par_rows(&mut out, w, 1, |y, row| {
        let fy = ((y as f32 + 0.5) * inv - 0.5).clamp(0.0, (dh - 1) as f32);
        let y0 = fy as usize;
        let y1 = (y0 + 1).min(dh - 1);
        let ty = fy - y0 as f32;
        let (r0, r1) = (&g[y0 * dw..y0 * dw + dw], &g[y1 * dw..y1 * dw + dw]);
        for (x, o) in row.iter_mut().enumerate() {
            let fx = ((x as f32 + 0.5) * inv - 0.5).clamp(0.0, (dw - 1) as f32);
            let x0 = fx as usize;
            let x1 = (x0 + 1).min(dw - 1);
            let tx = fx - x0 as f32;
            let top = r0[x0] + (r0[x1] - r0[x0]) * tx;
            let bot = r1[x0] + (r1[x1] - r1[x0]) * tx;
            *o = top + (bot - top) * ty;
        }
    });
    out
}

/// Box means over a `(2r + 1)²` window clamped to the plane (partial windows at the borders use
/// their own count): a horizontal moving sum per row in parallel, then a vertical moving sum
/// carried down the rows as one accumulator row.
struct BoxFilter {
    w: usize,
    h: usize,
    r: usize,
    inv_count: Vec<f32>,
}

impl BoxFilter {
    fn new(w: usize, h: usize, r: usize) -> BoxFilter {
        let cnt = |n: usize, i: usize| ((i + r + 1).min(n) - i.saturating_sub(r)) as f32;
        let mut inv_count = vec![0.0f32; w * h];
        par_rows(&mut inv_count, w, 1, |y, row| {
            let cy = cnt(h, y);
            for (x, o) in row.iter_mut().enumerate() {
                *o = 1.0 / (cy * cnt(w, x));
            }
        });
        BoxFilter { w, h, r, inv_count }
    }

    fn mean(&self, src: &[f32]) -> Vec<f32> {
        let (w, h, r) = (self.w, self.h, self.r);
        let mut tmp = vec![0.0f32; w * h];
        par_rows(&mut tmp, w, 1, |y, row| {
            let s = &src[y * w..y * w + w];
            let mut acc: f64 = s[..(r + 1).min(w)].iter().map(|v| f64::from(*v)).sum();
            for x in 0..w {
                row[x] = acc as f32;
                if x + r + 1 < w {
                    acc += f64::from(s[x + r + 1]);
                }
                if x >= r {
                    acc -= f64::from(s[x - r]);
                }
            }
        });
        let mut out = vec![0.0f32; w * h];
        let mut acc = vec![0.0f64; w];
        for y in 0..(r + 1).min(h) {
            for (a, v) in acc.iter_mut().zip(&tmp[y * w..y * w + w]) {
                *a += f64::from(*v);
            }
        }
        for y in 0..h {
            for ((o, a), c) in out[y * w..y * w + w].iter_mut().zip(&acc).zip(&self.inv_count[y * w..y * w + w]) {
                *o = *a as f32 * c;
            }
            if y + r + 1 < h {
                for (a, v) in acc.iter_mut().zip(&tmp[(y + r + 1) * w..(y + r + 2) * w]) {
                    *a += f64::from(*v);
                }
            }
            if y >= r {
                for (a, v) in acc.iter_mut().zip(&tmp[(y - r) * w..(y - r + 1) * w]) {
                    *a -= f64::from(*v);
                }
            }
        }
        out
    }
}

/// Guided filter (He, Sun & Tang 2013) with a fixed grayscale guide: made once per guide and
/// radius, it keeps the guide's statistics so a slice costs only the arithmetic.
struct GuidedFilter {
    guide: Vec<f32>,
    mean_i: Vec<f32>,
    var_i: Vec<f32>,
    bf: BoxFilter,
    eps: f32,
}

impl GuidedFilter {
    fn new(guide: Vec<f32>, w: usize, h: usize, r: usize, eps: f32) -> GuidedFilter {
        let bf = BoxFilter::new(w, h, r);
        let mean_i = bf.mean(&guide);
        let ii: Vec<f32> = guide.iter().map(|v| v * v).collect();
        let var_i: Vec<f32> = bf.mean(&ii).iter().zip(&mean_i).map(|(v, m)| (v - m * m).max(0.0)).collect();
        GuidedFilter { guide, mean_i, var_i, bf, eps }
    }

    /// The box-averaged linear coefficients `(ā, b̄)` such that `q ≈ ā·I + b̄`.
    fn coeffs(&self, p: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mean_p = self.bf.mean(p);
        let ip: Vec<f32> = self.guide.iter().zip(p).map(|(i, p)| i * p).collect();
        let corr_ip = self.bf.mean(&ip);
        let (mut a, mut b) = (vec![0.0f32; p.len()], vec![0.0f32; p.len()]);
        for k in 0..p.len() {
            let cov = corr_ip[k] - self.mean_i[k] * mean_p[k];
            a[k] = cov / (self.var_i[k] + self.eps);
            b[k] = mean_p[k] - a[k] * self.mean_i[k];
        }
        (self.bf.mean(&a), self.bf.mean(&b))
    }

    fn filter(&self, p: &[f32]) -> Vec<f32> {
        let (a, b) = self.coeffs(p);
        (0..p.len()).map(|k| a[k] * self.guide[k] + b[k]).collect()
    }
}

/// Ring difference filter taps `(dy, dx, weight)`: `+1/|disk|` for `r ≤ r_in`, `−1/|ring|` for
/// `r_in < r ≤ r_out`.
fn rdf_taps(r_in: usize, r_out: usize) -> Vec<(i64, i64, f32)> {
    let (ri2, ro2) = ((r_in * r_in) as i64, (r_out * r_out) as i64);
    let (mut disk, mut ring) = (Vec::new(), Vec::new());
    let r = r_out as i64;
    for dy in -r..=r {
        for dx in -r..=r {
            let d2 = dx * dx + dy * dy;
            if d2 <= ri2 {
                disk.push((dy, dx));
            } else if d2 <= ro2 {
                ring.push((dy, dx));
            }
        }
    }
    let (wd, wr) = (1.0 / disk.len().max(1) as f32, -1.0 / ring.len().max(1) as f32);
    disk.into_iter().map(|(y, x)| (y, x, wd)).chain(ring.into_iter().map(|(y, x)| (y, x, wr))).collect()
}

/// Per-pixel focus measure of a luminance plane: the magnitude of the ring difference filter,
/// clamped borders.
pub fn focus_measure(y: &[f32], w: usize, h: usize, r_in: usize, r_out: usize) -> Vec<f32> {
    let taps = rdf_taps(r_in, r_out.max(r_in + 1));
    let mut out = vec![0.0f32; w * h];
    if w == 0 || h == 0 {
        return out;
    }
    par_rows(&mut out, w, 1, |yy, row| {
        for &(dy, dx, wt) in &taps {
            let sy = (yy as i64 + dy).clamp(0, h as i64 - 1) as usize;
            let s = &y[sy * w..sy * w + w];
            for (x, o) in row.iter_mut().enumerate() {
                *o += wt * s[(x as i64 + dx).clamp(0, w as i64 - 1) as usize];
            }
        }
        row.iter_mut().for_each(|v| *v = v.abs());
    });
    out
}

/// Per-pixel state of the focus profile, updated one slice at a time.
#[derive(Clone, Copy)]
struct Px {
    /// Best local maximum (value, left neighbour, right neighbour; −1 = none) and its index.
    c1: f32,
    l1: f32,
    r1: f32,
    i1: u32,
    /// Second-best local maximum (−1 = none).
    c2: f32,
    /// The last two slice values.
    prev: f32,
    prev2: f32,
    sum: f32,
    cmin: f32,
}

impl Px {
    fn register(&mut self, val: f32, idx: u32, l: f32, r: f32) {
        if val > self.c1 {
            if self.c1 >= 0.0 {
                self.c2 = self.c1;
            }
            (self.c1, self.i1, self.l1, self.r1) = (val, idx, l, r);
        } else if val > self.c2 {
            self.c2 = val;
        }
    }
}

/// The streaming peak search over the layer axis.
struct PeakTracker {
    px: Vec<Px>,
    m: usize,
}

impl PeakTracker {
    fn new(n: usize) -> PeakTracker {
        PeakTracker { px: vec![Px { c1: -1.0, l1: -1.0, r1: -1.0, i1: 0, c2: -1.0, prev: 0.0, prev2: 0.0, sum: 0.0, cmin: f32::INFINITY }; n], m: 0 }
    }

    /// Fold the next slice (aggregated focus values ≥ 0).
    fn push(&mut self, c: &[f32]) {
        let m = self.m;
        for (p, &v) in self.px.iter_mut().zip(c) {
            if m >= 1 && (m == 1 || p.prev >= p.prev2) && p.prev > v {
                let l = if m >= 2 { p.prev2 } else { -1.0 };
                p.register(p.prev, (m - 1) as u32, l, v);
            }
            p.sum += v;
            p.cmin = p.cmin.min(v);
            (p.prev2, p.prev) = (p.prev, v);
        }
        self.m += 1;
    }

    /// Close the profiles: the sub-layer depth and the raw confidence per pixel.
    fn finish(mut self, gate: f32) -> (Vec<f32>, Vec<f32>) {
        let n = self.m;
        if n == 0 {
            return (vec![0.0; self.px.len()], vec![0.0; self.px.len()]);
        }
        for p in self.px.iter_mut() {
            if n == 1 || p.prev >= p.prev2 {
                let l = if n >= 2 { p.prev2 } else { -1.0 };
                p.register(p.prev, (n - 1) as u32, l, -1.0);
            }
        }
        // The noise floor: the median of the per-pixel profile minimum (subsampled).
        let mut mins: Vec<f32> = self.px.iter().step_by(7).map(|p| if p.cmin.is_finite() { p.cmin } else { 0.0 }).collect();
        let floor = if mins.is_empty() {
            0.0
        } else {
            let mid = mins.len() / 2;
            *mins.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1
        };
        let inv_n = 1.0 / n as f32;
        let out: Vec<(f32, f32)> = par_map(self.px.len(), |i| {
            let p = &self.px[i];
            let (c1, l, r) = (p.c1, p.l1, p.r1);
            let mut delta = 0.0f32;
            if l >= 0.0 && r >= 0.0 && c1 > 0.0 {
                // Gaussian interpolation (Nayar & Nakagawa 1994): a parabola in the log domain.
                let (ll, lc, lr) = (l.max(1e-12).ln(), c1.ln(), r.max(1e-12).ln());
                let den = ll - 2.0 * lc + lr;
                if den < 0.0 {
                    delta = (0.5 * (ll - lr) / den).clamp(-0.5, 0.5);
                }
            }
            let conf = if c1 > 0.0 {
                let prom = (1.0 - p.sum * inv_n / c1).clamp(0.0, 1.0);
                // A rival local maximum of similar height means an ambiguous profile.
                let pkr = if p.c2 >= 0.0 { (1.0 - p.c2 / c1).clamp(0.0, 1.0) } else { 1.0 };
                let g = if gate > 0.0 && floor > 0.0 { ((c1 - floor) / (gate * floor)).clamp(0.0, 1.0) } else { 1.0 };
                prom * pkr * g
            } else {
                0.0
            };
            (p.i1 as f32 + delta, conf)
        });
        out.into_iter().unzip()
    }
}

/// Scale the confidence so its 90th percentile maps to 1: `lambda` then means the same on every
/// stack.
fn normalize_conf(conf: &mut [f32]) {
    let mut sample: Vec<f32> = conf.iter().step_by(7).copied().collect();
    if sample.is_empty() {
        return;
    }
    let k = (sample.len() * 9 / 10).min(sample.len() - 1);
    let p90 = *sample.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1;
    if p90 > 1e-6 {
        conf.iter_mut().for_each(|c| *c = (*c / p90).min(1.0));
    }
}

/// 3×3 median, clamped borders.
fn median3(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    par_rows(&mut out, w, 1, |y, row| {
        let ys = [y.saturating_sub(1), y, (y + 1).min(h - 1)];
        let mut v = [0.0f32; 9];
        for (x, o) in row.iter_mut().enumerate() {
            let xs = [x.saturating_sub(1), x, (x + 1).min(w - 1)];
            for (k, (yy, xx)) in ys.iter().flat_map(|yy| xs.iter().map(move |xx| (*yy, *xx))).enumerate() {
                v[k] = src[yy * w + xx];
            }
            v.sort_unstable_by(|a, b| a.total_cmp(b));
            *o = v[4];
        }
    });
    out
}

/// Edge weights between horizontal and vertical neighbours: `exp(−|ΔI| / σ)`; `ax[i]` sits
/// between `(x, y)` and `(x + 1, y)`, `ay[i]` between `(x, y)` and `(x, y + 1)`.
fn edge_weights(guide: &[f32], w: usize, h: usize, sigma: f32) -> (Vec<f32>, Vec<f32>) {
    let inv = -1.0 / sigma.max(1e-6);
    let (mut ax, mut ay) = (vec![0.0f32; w * h], vec![0.0f32; w * h]);
    par_rows2(&mut ax, &mut ay, w, |y, rx, ry| {
        let g = &guide[y * w..y * w + w];
        for x in 0..w.saturating_sub(1) {
            rx[x] = ((g[x] - g[x + 1]).abs() * inv).exp();
        }
        if y + 1 < h {
            let g1 = &guide[(y + 1) * w..(y + 2) * w];
            for x in 0..w {
                ry[x] = ((g[x] - g1[x]).abs() * inv).exp();
            }
        }
    });
    (ax, ay)
}

/// One 1-D weighted-least-squares line `(wd_i + λ(a_{i−1} + a_i)) u_i − λ a_{i−1} u_{i−1} −
/// λ a_i u_{i+1} = wd_i f_i` by the Thomas algorithm; `a[i]` is the weight between `i` and
/// `i + 1`; `scratch` holds the sweep's two coefficient lines.
fn solve_line(u: &mut [f32], f: &[f32], wd: &[f32], a: &[f32], lam: f32, scratch: &mut [f32]) {
    let n = u.len();
    if n == 0 || scratch.len() < 2 * n {
        return;
    }
    let (cp, dp) = scratch.split_at_mut(n);
    let mut prev_a = 0.0f32;
    for i in 0..n {
        let ai = if i + 1 < n { a[i] } else { 0.0 };
        let diag = wd[i] + lam * (prev_a + ai);
        let lower = -lam * prev_a;
        let upper = -lam * ai;
        let (c0, d0) = if i > 0 { (cp[i - 1], dp[i - 1]) } else { (0.0, 0.0) };
        let m = diag - lower * c0;
        let m = if m.abs() < 1e-12 { 1e-12 } else { m };
        cp[i] = upper / m;
        dp[i] = (wd[i] * f[i] - lower * d0) / m;
        prev_a = ai;
    }
    u[n - 1] = dp[n - 1];
    for i in (0..n - 1).rev() {
        u[i] = dp[i] - cp[i] * u[i + 1];
    }
}

/// The row sweeps of the fast global smoother, in parallel over rows.
fn solve_rows(u: &mut [f32], f: &[f32], wd: &[f32], ax: &[f32], w: usize, lam: f32) {
    par_rows(u, w, 1, |y, row| {
        let o = y * w;
        let mut scratch = vec![0.0f32; 2 * w];
        solve_line(row, &f[o..o + w], &wd[o..o + w], &ax[o..o + w], lam, &mut scratch);
    });
}

/// The column sweeps: the planes are transposed, swept as rows, and transposed back.
fn solve_cols(u: &mut [f32], f: &[f32], wd: &[f32], ay: &[f32], w: usize, h: usize, lam: f32) {
    let t = |p: &[f32]| -> Vec<f32> {
        let mut o = vec![0.0f32; w * h];
        par_rows(&mut o, h, 1, |x, col| {
            for (y, v) in col.iter_mut().enumerate() {
                *v = p[y * w + x];
            }
        });
        o
    };
    let (ft, wdt, ayt) = (t(f), t(wd), t(ay));
    let mut ut = vec![0.0f32; w * h];
    solve_rows(&mut ut, &ft, &wdt, &ayt, h, lam);
    par_rows(u, w, 1, |y, row| {
        for (x, v) in row.iter_mut().enumerate() {
            *v = ut[x * h + y];
        }
    });
}

/// The fast global smoother's three separable passes with the `λ_t` schedule of Min et al.
/// 2014, the first pass carrying the data weights so holes are filled by interpolation, the
/// later ones the previous output: the initial guess of [`wls_solve`].
fn sweeps(d: &[f32], wd: &[f32], ex: &[f32], ey: &[f32], w: usize, h: usize, lambda: f32) -> Vec<f32> {
    const T: i32 = 3;
    let ones = vec![1.0f32; w * h];
    let mut u = vec![0.0f32; w * h];
    let mut f = d.to_vec();
    for t in 1..=T {
        let lam_t = 1.5 * lambda * 4f32.powi(T - t) / (4f32.powi(T) - 1.0);
        let wt: &[f32] = if t == 1 { wd } else { &ones };
        solve_rows(&mut u, &f, wt, ex, w, lam_t);
        f.copy_from_slice(&u);
        solve_cols(&mut u, &f, &ones, ey, w, h, lam_t);
        f.copy_from_slice(&u);
    }
    u
}

/// The WLS operator `A = W + Σ λ a_pq` on one grid of the multigrid hierarchy, with the planes
/// a V-cycle needs there.
struct MgLevel {
    w: usize,
    h: usize,
    /// Data weight per cell; on the coarse grids the sum over the block.
    wd: Vec<f32>,
    /// `λ·a` between `(x, y)` and `(x + 1, y)`, and between `(x, y)` and `(x, y + 1)`; on the
    /// coarse grids the sum of the fine edges the block boundary cuts.
    ax: Vec<f32>,
    ay: Vec<f32>,
    /// `1 / diag(A)`.
    dinv: Vec<f32>,
    x: Vec<f32>,
    b: Vec<f32>,
    r: Vec<f32>,
}

/// Damped-Jacobi weight (4/5 is the smoothing optimum of the 5-point stencil).
const MG_OMEGA: f32 = 0.8;
/// Smoothing sweeps before and after the coarse correction, and on the coarsest grid (where
/// they are the solve).
const MG_PRE: usize = 1;
const MG_POST: usize = 1;
const MG_COARSE: usize = 32;
/// Coarsen until the grid is this small on its longer side.
const MG_MIN: usize = 16;

impl MgLevel {
    fn new(w: usize, h: usize) -> MgLevel {
        let n = w * h;
        MgLevel { w, h, wd: vec![0.0; n], ax: vec![0.0; n], ay: vec![0.0; n], dinv: vec![0.0; n], x: vec![0.0; n], b: vec![0.0; n], r: vec![0.0; n] }
    }

    /// `out_i = f(i, (A x)_i)` over the grid.
    fn apply<F: Fn(usize, f32) -> f32 + Sync + Send>(&self, x: &[f32], out: &mut [f32], f: F) {
        let (w, h) = (self.w, self.h);
        let (wd, ax, ay) = (&self.wd, &self.ax, &self.ay);
        par_rows(out, w, 1, |y, row| {
            let o = y * w;
            for (i, r) in row.iter_mut().enumerate() {
                let k = o + i;
                let xk = x[k];
                let mut v = wd[k] * xk;
                if i > 0 {
                    v += ax[k - 1] * (xk - x[k - 1]);
                }
                if i + 1 < w {
                    v += ax[k] * (xk - x[k + 1]);
                }
                if y > 0 {
                    v += ay[k - w] * (xk - x[k - w]);
                }
                if y + 1 < h {
                    v += ay[k] * (xk - x[k + w]);
                }
                *r = f(k, v);
            }
        });
    }

    fn set_dinv(&mut self) {
        let (w, h) = (self.w, self.h);
        let (wd, ax, ay) = (&self.wd, &self.ax, &self.ay);
        par_rows(&mut self.dinv, w, 1, |y, row| {
            for (i, r) in row.iter_mut().enumerate() {
                let k = y * w + i;
                let mut v = wd[k];
                if i > 0 {
                    v += ax[k - 1];
                }
                if i + 1 < w {
                    v += ax[k];
                }
                if y > 0 {
                    v += ay[k - w];
                }
                if y + 1 < h {
                    v += ay[k];
                }
                *r = 1.0 / v.max(1e-12);
            }
        });
    }

    /// One damped-Jacobi sweep on `x` for `b`: `x += ω D⁻¹ (b − A x)`.
    fn jacobi(&mut self, zero_start: bool) {
        if zero_start {
            for ((x, b), d) in self.x.iter_mut().zip(&self.b).zip(&self.dinv) {
                *x = MG_OMEGA * d * b;
            }
            return;
        }
        let mut r = std::mem::take(&mut self.r);
        {
            let (x, b, dinv) = (&self.x, &self.b, &self.dinv);
            self.apply(x, &mut r, |k, ax| x[k] + MG_OMEGA * dinv[k] * (b[k] - ax));
        }
        self.r = std::mem::replace(&mut self.x, r);
    }

    /// The 2×2 aggregation of this grid: the Galerkin operator of a piecewise-constant
    /// prolongation (data weights summed over the block, the fine edges cut by a block boundary
    /// summed into the coarse edge).
    fn coarsen(&self) -> MgLevel {
        let (w, h) = (self.w, self.h);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let mut c = MgLevel::new(cw, ch);
        self.coarsen_wd_into(&mut c);
        let (ax, ay) = (&self.ax, &self.ay);
        par_rows(&mut c.ax, cw, 1, |yy, row| {
            for y in 2 * yy..(2 * yy + 2).min(h) {
                for (xx, o) in row.iter_mut().enumerate() {
                    if xx + 1 < cw {
                        *o += ax[y * w + 2 * xx + 1];
                    }
                }
            }
        });
        par_rows(&mut c.ay, cw, 1, |yy, row| {
            if yy + 1 < ch {
                let y = 2 * yy + 1;
                for (xx, o) in row.iter_mut().enumerate() {
                    *o = ay[y * w..y * w + w][2 * xx..(2 * xx + 2).min(w)].iter().sum::<f32>();
                }
            }
        });
        c
    }

    /// The data weights of the coarse grid `c` from this grid's (after a reweighting; the edges
    /// do not change).
    fn coarsen_wd_into(&self, c: &mut MgLevel) {
        let (w, h, cw) = (self.w, self.h, c.w);
        let wd = &self.wd;
        par_rows(&mut c.wd, cw, 1, |yy, row| {
            row.fill(0.0);
            for y in 2 * yy..(2 * yy + 2).min(h) {
                for (xx, o) in row.iter_mut().enumerate() {
                    *o += wd[y * w..y * w + w][2 * xx..(2 * xx + 2).min(w)].iter().sum::<f32>();
                }
            }
        });
    }
}

/// `x += P x_c`: the coarse correction injected (piecewise constant).
fn prolong_add(x: &mut [f32], w: usize, xc: &[f32], cw: usize) {
    par_rows(x, w, 1, |y, row| {
        let c = &xc[(y / 2) * cw..(y / 2) * cw + cw];
        for (i, o) in row.iter_mut().enumerate() {
            *o += c[i / 2];
        }
    });
}

/// `b_c = Pᵀ r`: the residual summed over each block.
fn restrict(r: &[f32], w: usize, h: usize, bc: &mut [f32], cw: usize) {
    par_rows(bc, cw, 1, |yy, row| {
        row.fill(0.0);
        for y in 2 * yy..(2 * yy + 2).min(h) {
            let s = &r[y * w..y * w + w];
            for (xx, o) in row.iter_mut().enumerate() {
                *o += s[2 * xx..(2 * xx + 2).min(w)].iter().sum::<f32>();
            }
        }
    });
}

/// One V-cycle from `levels[0]` (whose `b` is set) into its `x`.
fn vcycle(levels: &mut [MgLevel]) {
    let Some((top, rest)) = levels.split_first_mut() else { return };
    let Some(next) = rest.first_mut() else {
        for k in 0..MG_COARSE {
            top.jacobi(k == 0);
        }
        return;
    };
    for k in 0..MG_PRE {
        top.jacobi(k == 0);
    }
    {
        let mut r = std::mem::take(&mut top.r);
        let (x, b) = (&top.x, &top.b);
        top.apply(x, &mut r, |k, ax| b[k] - ax);
        top.r = r;
    }
    restrict(&top.r, top.w, top.h, &mut next.b, next.w);
    vcycle(rest);
    let Some(next) = rest.first() else { return };
    prolong_add(&mut top.x, top.w, &next.x, next.w);
    for _ in 0..MG_POST {
        top.jacobi(false);
    }
}

/// Edge-aware WLS: `argmin_u Σ w_p (u_p − d_p)² + λ Σ a_pq (u_p − u_q)²`, the 2-D system
/// `(W + λL) u = W d` solved by conjugate gradients from the separable sweeps' guess,
/// preconditioned by one multigrid V-cycle (2×2 aggregation, damped Jacobi), so the low modes
/// converge where the confidence is low over a large area (a flat wall). The solver keeps its
/// hierarchy between solves, so a reweighting costs no rebuild.
struct WlsSolver {
    w: usize,
    h: usize,
    lambda: f32,
    /// Edge weights `exp(−|ΔI| / σ)` (without λ), for the separable sweeps.
    ex: Vec<f32>,
    ey: Vec<f32>,
    levels: Vec<MgLevel>,
}

impl WlsSolver {
    fn new(guide: &[f32], w: usize, h: usize, lambda: f32, sigma_c: f32) -> WlsSolver {
        let (ex, ey) = edge_weights(guide, w, h, sigma_c);
        let mut top = MgLevel::new(w, h);
        for (a, e) in top.ax.iter_mut().zip(&ex) {
            *a = lambda * e;
        }
        for (a, e) in top.ay.iter_mut().zip(&ey) {
            *a = lambda * e;
        }
        let mut levels = vec![top];
        while levels.last().is_some_and(|l| l.w.max(l.h) > MG_MIN) {
            let Some(last) = levels.last() else { break };
            let c = last.coarsen();
            levels.push(c);
        }
        WlsSolver { w, h, lambda, ex, ey, levels }
    }

    /// The data weights: the confidence (plus a floor so the system is definite where nothing
    /// is known), aggregated down the hierarchy.
    fn set_weights(&mut self, conf: &[f32]) {
        const EPS_DATA: f32 = 1e-4;
        if let Some(top) = self.levels.first_mut() {
            for (w, c) in top.wd.iter_mut().zip(conf) {
                *w = c.max(0.0) + EPS_DATA;
            }
        }
        for l in 0..self.levels.len() {
            if l > 0 {
                let (fine, coarse) = self.levels.split_at_mut(l);
                if let (Some(f), Some(c)) = (fine.last(), coarse.first_mut()) {
                    f.coarsen_wd_into(c);
                }
            }
            self.levels[l].set_dinv();
        }
    }

    /// Solve for the data `d`: the solution and its final relative residual.
    fn solve(&mut self, d: &[f32], max_iters: usize) -> (Vec<f32>, f32) {
        let (w, h, lambda) = (self.w, self.h, self.lambda);
        let n = w * h;
        let Some(top) = self.levels.first() else { return (vec![0.0; n], 0.0) };
        let mut u = sweeps(d, &top.wd, &self.ex, &self.ey, w, h, lambda);
        if w == 1 || h == 1 || n == 0 {
            return (u, 0.0);
        }
        let dot = |a: &[f32], b: &[f32]| -> f64 { a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum() };
        let b: Vec<f32> = top.wd.iter().zip(d).map(|(w, d)| w * d).collect();
        let bnorm = dot(&b, &b).sqrt().max(1e-30);
        let mut r = vec![0.0f32; n];
        top.apply(&u, &mut r, |k, au| b[k] - au);
        let precond = |levels: &mut [MgLevel], r: &[f32], z: &mut [f32]| {
            if let Some(top) = levels.first_mut() {
                top.b.copy_from_slice(r);
            }
            vcycle(levels);
            if let Some(top) = levels.first() {
                z.copy_from_slice(&top.x);
            }
        };
        let mut z = vec![0.0f32; n];
        precond(&mut self.levels, &r, &mut z);
        let mut p = z.clone();
        let mut ap = vec![0.0f32; n];
        let mut rz = dot(&r, &z);
        let mut rel = (dot(&r, &r).sqrt() / bnorm) as f32;
        for _ in 0..max_iters {
            if rel < 1e-5 {
                break;
            }
            let Some(top) = self.levels.first() else { break };
            top.apply(&p, &mut ap, |_, v| v);
            let pap = dot(&p, &ap);
            if pap <= 0.0 {
                break;
            }
            let alpha = (rz / pap) as f32;
            for (u, p) in u.iter_mut().zip(&p) {
                *u += alpha * p;
            }
            for (r, ap) in r.iter_mut().zip(&ap) {
                *r -= alpha * ap;
            }
            rel = (dot(&r, &r).sqrt() / bnorm) as f32;
            if rel < 1e-5 {
                break;
            }
            precond(&mut self.levels, &r, &mut z);
            let rz_new = dot(&r, &z);
            let beta = (rz_new / rz) as f32;
            rz = rz_new;
            for (p, z) in p.iter_mut().zip(&z) {
                *p = z + beta * *p;
            }
        }
        (u, rel)
    }
}

/// Conjugate-gradient iteration cap of the regularisation.
const CG_ITERS: usize = 200;

/// Depth from focus over the `n` layers' luminance planes `luma` (`w × h`, aligned), guided by
/// the all-in-focus luminance `guide` of the same size. Without layers, or when `guide` or a
/// plane is not `w × h` long, the map is empty: depth 0 and confidence 0 everywhere.
pub fn depth_from_focus(luma: &[&[f32]], guide: &[f32], w: usize, h: usize, p: &DepthParams) -> DepthMap {
    let n = luma.len();
    let empty = n == 0 || w == 0 || h == 0 || guide.len() != w * h || luma.iter().any(|l| l.len() != w * h);
    if empty {
        return DepthMap { depth: vec![0.0; w * h], conf: vec![0.0; w * h], w, h };
    }
    let k = 1usize << p.scale.min(8);
    let (g, dw, dh) = block_mean(guide, w, h, k);
    let gf = GuidedFilter::new(g.clone(), dw, dh, p.agg_radius, p.agg_eps);
    let mut tracker = PeakTracker::new(dw * dh);
    for l in luma {
        let slice = block_mean(&focus_measure(l, w, h, p.r_in, p.r_out), w, h, k).0;
        if p.agg_radius > 0 {
            // The guided filter can undershoot; the profile statistics assume ≥ 0.
            let agg: Vec<f32> = gf.filter(&slice).into_iter().map(|v| v.max(0.0)).collect();
            tracker.push(&agg);
        } else {
            tracker.push(&slice);
        }
    }
    let (depth_w, mut conf) = tracker.finish(p.gate);
    let depth_w = median3(&depth_w, dw, dh);
    normalize_conf(&mut conf);
    let depth_w = if p.lambda > 0.0 {
        let mut solver = WlsSolver::new(&g, dw, dh, p.lambda, p.sigma_c);
        solver.set_weights(&conf);
        let (u, _) = solver.solve(&depth_w, CG_ITERS);
        if p.robust > 0.0 {
            // One reweighted step with a Huber loss on the data residual.
            let wd2: Vec<f32> = conf.iter().zip(&depth_w).zip(&u).map(|((c, d), u)| c * (p.robust / (d - u).abs().max(1e-6)).min(1.0)).collect();
            solver.set_weights(&wd2);
            solver.solve(&depth_w, CG_ITERS).0
        } else {
            u
        }
    } else {
        depth_w
    };
    let max_d = (n - 1) as f32;
    let up = GuidedFilter::new(g, dw, dh, 2, 1e-2);
    let depth: Vec<f32> = if k == 1 {
        up.filter(&depth_w)
    } else {
        let (a, b) = up.coeffs(&depth_w);
        let a = upsample_bilinear(&a, dw, dh, w, h, k);
        let b = upsample_bilinear(&b, dw, dh, w, h, k);
        (0..w * h).map(|i| a[i] * guide[i] + b[i]).collect()
    };
    let depth = depth.into_iter().map(|v| v.clamp(0.0, max_d)).collect();
    let conf = if k == 1 { conf } else { upsample_bilinear(&conf, dw, dh, w, h, k) };
    DepthMap { depth, conf, w, h }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) % 10000) as f32 / 10000.0
    }

    #[test]
    fn box_filter_matches_the_naive_window() {
        let (w, h, r) = (23, 17, 3);
        let mut s = 5u64;
        let src: Vec<f32> = (0..w * h).map(|_| lcg(&mut s)).collect();
        let m = BoxFilter::new(w, h, r).mean(&src);
        for y in 0..h {
            for x in 0..w {
                let (mut acc, mut cnt) = (0.0, 0.0);
                for yy in y.saturating_sub(r)..(y + r + 1).min(h) {
                    for xx in x.saturating_sub(r)..(x + r + 1).min(w) {
                        acc += src[yy * w + xx];
                        cnt += 1.0;
                    }
                }
                assert!((m[y * w + x] - acc / cnt).abs() < 1e-4, "({x},{y})");
            }
        }
    }

    #[test]
    fn guided_filter_with_a_flat_guide_is_a_double_box_mean() {
        let (w, h) = (40, 30);
        let mut s = 9u64;
        let p: Vec<f32> = (0..w * h).map(|_| lcg(&mut s)).collect();
        let q = GuidedFilter::new(vec![0.5; w * h], w, h, 2, 1e-3).filter(&p);
        let bf = BoxFilter::new(w, h, 2);
        let m = bf.mean(&bf.mean(&p));
        assert!(q.iter().zip(&m).all(|(a, b)| (a - b).abs() < 1e-4));
    }

    #[test]
    fn ring_difference_is_zero_mean_and_sees_texture() {
        let taps = rdf_taps(1, 3);
        assert!(taps.iter().map(|t| t.2).sum::<f32>().abs() < 1e-6);
        let (w, h) = (32, 32);
        assert!(focus_measure(&vec![0.4; w * h], w, h, 1, 3).iter().all(|v| v.abs() < 1e-6));
        let check: Vec<f32> = (0..w * h).map(|i| if ((i % w) / 2 + (i / w) / 2).is_multiple_of(2) { 0.2 } else { 0.8 }).collect();
        assert!(focus_measure(&check, w, h, 1, 3).iter().cloned().fold(0.0, f32::max) > 0.1);
        let (m, dw, dh) = block_mean(&check, w, h, 2);
        assert_eq!((dw, dh, m.len()), (16, 16, 256));
        assert!(m.iter().all(|v| (v - 0.2).abs() < 1e-6 || (v - 0.8).abs() < 1e-6));
        let up = upsample_bilinear(&m, dw, dh, w, h, 2);
        assert_eq!(up.len(), w * h);
    }

    #[test]
    fn peak_search_interpolates_between_layers() {
        // A profile peaking between layers 2 and 3 (log-parabola), one flat profile.
        let mut t = PeakTracker::new(2);
        for m in 0..6 {
            let v = (-((m as f32 - 2.4).powi(2)) / 2.0).exp();
            t.push(&[v, 0.1]);
        }
        let (d, c) = t.finish(0.0);
        assert!((d[0] - 2.4).abs() < 0.05, "{d:?}");
        assert!(c[0] > 0.5 && c[1] < 1e-6, "{c:?}");
    }

    #[test]
    fn smoothing_fills_holes_and_respects_edges() {
        let (w, h) = (40, 20);
        // A guide with a hard edge at x = 20; data known only at two pixels, one per side.
        let guide: Vec<f32> = (0..w * h).map(|i| if i % w < 20 { 0.0 } else { 1.0 }).collect();
        let mut d = vec![0.0f32; w * h];
        let mut wd = vec![1e-4f32; w * h];
        (d[10 * w + 5], wd[10 * w + 5]) = (1.0, 1.0);
        (d[10 * w + 35], wd[10 * w + 35]) = (5.0, 1.0);
        let mut solver = WlsSolver::new(&guide, w, h, 3.0, 0.04);
        solver.set_weights(&wd);
        let (u, rel) = solver.solve(&d, CG_ITERS);
        assert!(rel < 1e-4, "residual {rel}");
        // Each side follows its own sample (the data floor on the unknown pixels pulls a little
        // toward zero), and the edge keeps them apart.
        assert!((u[3 * w + 15] - 1.0).abs() < 0.15, "left side follows its sample: {}", u[3 * w + 15]);
        assert!((u[15 * w + 25] - 5.0).abs() < 0.6, "right side follows its sample: {}", u[15 * w + 25]);
        assert!(u[10 * w + 19] < 1.5 && u[10 * w + 20] > 4.0, "the edge holds: {} | {}", u[10 * w + 19], u[10 * w + 20]);
        // The sweeps alone leave the holes near zero; the solve fills them.
        let guess = sweeps(&d, &wd, &solver.ex, &solver.ey, w, h, 3.0);
        assert!(guess[3 * w + 15] < 0.5);
        assert_eq!(median3(&[0.5; 9], 3, 3)[4], 0.5);
    }

    /// Three layers: the left third sharp in layer 0, the middle in 1, the right in 2. The depth
    /// map says so, sub-layer in between, with high confidence on the texture.
    #[test]
    fn depth_from_focus_finds_the_sharp_layer() {
        let (w, h) = (96, 48);
        let texture = |x: usize, y: usize| if ((x / 2) + (y / 2)).is_multiple_of(2) { 0.2 } else { 0.8 };
        let flat = 0.5;
        let layers: Vec<Vec<f32>> = (0..3).map(|k| (0..w * h).map(|i| if (i % w) / 32 == k { texture(i % w, i / w) } else { flat }).collect()).collect();
        let guide: Vec<f32> = (0..w * h).map(|i| texture(i % w, i / w)).collect();
        let refs: Vec<&[f32]> = layers.iter().map(Vec::as_slice).collect();
        let dm = depth_from_focus(&refs, &guide, w, h, &DepthParams::default());
        assert_eq!((dm.w, dm.h, dm.depth.len()), (w, h, w * h));
        for (x, want) in [(8, 0.0), (48, 1.0), (88, 2.0)] {
            let d = dm.depth[24 * w + x];
            assert!((d - want).abs() < 0.35, "x {x}: depth {d}, want {want}");
            assert!(dm.conf[24 * w + x] > 0.3, "x {x}: conf {}", dm.conf[24 * w + x]);
        }
        assert!(dm.depth.iter().all(|v| (0.0..=2.0).contains(v)));
        // Degenerate inputs are fine.
        assert_eq!(depth_from_focus(&[], &guide, w, h, &DepthParams::default()).depth, vec![0.0; w * h]);
        // A guide or a plane of the wrong length is no input: the empty map, never a panic.
        let empty = DepthMap { depth: vec![0.0; w * h], conf: vec![0.0; w * h], w, h };
        assert_eq!(depth_from_focus(&refs, &guide[1..], w, h, &DepthParams::default()), empty);
        let short = &refs[0][..w * h - 1];
        assert_eq!(depth_from_focus(&[refs[0], short], &guide, w, h, &DepthParams::default()), empty);
        assert_eq!(depth_from_focus(&refs, &guide, w + 1, h, &DepthParams::default()).w, w + 1);
        let one = depth_from_focus(&refs[..1], &guide, w, h, &DepthParams::default());
        assert!(one.depth.iter().all(|v| *v == 0.0));
    }
}
