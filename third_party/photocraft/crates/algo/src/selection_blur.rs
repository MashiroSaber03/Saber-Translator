//! Fast mask blurs behind Select › Modify › Smooth and Feather (#211).
//!
//! Both are separable and give the same numbers as the direct definitions (see the oracle tests):
//!
//! * **Smooth** is a box average with edge-replicated borders: a running sum per line, `O(1)` per
//!   pixel whatever the radius.
//! * **Feather** is a Gaussian (`σ = radius`, as Photoshop's, kernel truncated at `⌈3σ⌉` and normalized, zero
//!   outside the canvas). Small kernels are convolved directly. Wide ones are written as a short
//!   cosine series over the kernel window, `k(j) ≈ a₀ + Σ aₖ cos(ωₖ j)`, and every term is a
//!   sliding windowed sum `Σ m[x+j]·e^{iωₖj}` updated in `O(1)` per pixel, so the cost doesn't grow
//!   with the radius. The series is fitted per call by least squares and its error is checked
//!   (≤ 0.3/255 per pass); a fit that misses falls back to the direct convolution.
//!
//! Work is confined to the mask's non-zero bounding box grown by the kernel radius (pixels
//! farther away stay exactly 0). Rows run in parallel (`par_rows`/`par_map`, serial on wasm);
//! the vertical pass walks strips of 128 columns row by row (vectorizable, no transpose). Windows
//! lying entirely in a constant run (inside or outside the selection) are written without
//! arithmetic.

use crate::photo_util::{par_map, par_rows};

/// Cosine terms in the wide-kernel series (5 gives ≈ 0.1/255 per pass; see the module docs).
const TERMS: usize = 5;
/// Kernels up to this radius are convolved directly.
const DIRECT_MAX_R: usize = 12;
/// Largest error (in mask units, per pass) a cosine fit may have before falling back.
const FIT_TOL: f64 = 0.3 / 255.0;
/// σ cap: far beyond any canvas, keeps the kernel arithmetic finite for absurd radii.
const MAX_SIGMA: f32 = 100_000.0;
/// Box radius cap for the same reason.
const MAX_BOX_R: usize = 1 << 24;
/// Columns per task in the vertical pass.
const STRIP: usize = 64;

/// A window of the mask: columns `x0..x1`, rows `y0..y1`.
#[derive(Clone, Copy, Debug)]
struct Win {
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
}

impl Win {
    fn w(&self) -> usize {
        self.x1 - self.x0
    }
    fn h(&self) -> usize {
        self.y1 - self.y0
    }
}

/// Bounding box of the non-zero samples (NaN counts as non-zero), or `None` when all are 0.
fn nonzero_bounds(m: &[f32], w: usize, h: usize) -> Option<Win> {
    let rows: Vec<Option<(usize, usize)>> = par_map(h, |y| {
        let row = m.get(y * w..(y + 1) * w)?;
        let first = row.iter().position(|v| *v != 0.0)?;
        let last = row.iter().rposition(|v| *v != 0.0)?;
        Some((first, last + 1))
    });
    let mut b: Option<Win> = None;
    for (y, r) in rows.into_iter().enumerate() {
        if let Some((a, z)) = r {
            b = Some(match b {
                None => Win { x0: a, x1: z, y0: y, y1: y + 1 },
                Some(b) => Win { x0: b.x0.min(a), x1: b.x1.max(z), y0: b.y0, y1: y + 1 },
            });
        }
    }
    b
}

/// `b` grown by `g` on every side, clamped to `w × h`.
fn grow(b: Win, g: usize, w: usize, h: usize) -> Win {
    Win { x0: b.x0.saturating_sub(g), x1: b.x1.saturating_add(g).min(w), y0: b.y0.saturating_sub(g), y1: b.y1.saturating_add(g).min(h) }
}

/// The horizontal pass's output as the vertical pass reads it: `bh` rows of `gw` samples placed
/// at window rows `off..off + bh` of a window `gh` rows tall (other rows are zero).
struct Rows<'a> {
    tmp: &'a [f32],
    gw: usize,
    bh: usize,
    off: usize,
    gh: usize,
}

impl Rows<'_> {
    /// Columns `c0..c0 + sw` of window row `yy`, or `None` for a zero row.
    fn get(&self, yy: usize, c0: usize, sw: usize) -> Option<&[f32]> {
        let r = yy.checked_sub(self.off).filter(|r| *r < self.bh)?;
        self.tmp.get(r * self.gw + c0..r * self.gw + c0 + sw)
    }
}

/// Runs the vertical pass `f(c0, sw)` on strips of `STRIP` columns in parallel; each returns its
/// `gh × sw` result, row-major.
fn strips(gw: usize, f: impl Fn(usize, usize) -> Vec<f32> + Sync + Send) -> Vec<Vec<f32>> {
    par_map(gw.div_ceil(STRIP), |s| {
        let c0 = s * STRIP;
        f(c0, STRIP.min(gw - c0))
    })
}

/// Writes the strips into a fresh `w × h` mask through `f`; outside the window it is `outside`.
fn assemble(strips: &[Vec<f32>], win: Win, w: usize, h: usize, outside: f32, f: impl Fn(f32) -> f32 + Sync + Send) -> Vec<f32> {
    let mut out = vec![outside; w * h];
    par_rows(&mut out, w, 1, |y, row| {
        let Some(yy) = y.checked_sub(win.y0).filter(|_| y < win.y1) else { return };
        for (s, strip) in strips.iter().enumerate() {
            let c0 = s * STRIP;
            let sw = STRIP.min(win.w().saturating_sub(c0));
            if let (Some(src), Some(dst)) = (strip.get(yy * sw..(yy + 1) * sw), row.get_mut(win.x0 + c0..win.x0 + c0 + sw)) {
                for (d, v) in dst.iter_mut().zip(src) {
                    *d = f(*v);
                }
            }
        }
    });
    out
}

/// Adds `k ×` window row `yy` (columns `c0..c0 + acc.len()`) into `acc`.
fn add_row(acc: &mut [f64], rows: &Rows, yy: usize, c0: usize, k: f64) {
    if k == 0.0 {
        return;
    }
    if let Some(row) = rows.get(yy, c0, acc.len()) {
        for (a, v) in acc.iter_mut().zip(row) {
            *a += k * f64::from(*v);
        }
    }
}

// ---- Smooth: box average, edges replicated -------------------------------------------------

/// Running-sum box average of one line (radius `r`, edges replicated) into `dst`.
fn box_line(src: &[f32], dst: &mut [f32], r: usize) {
    let n = src.len().min(dst.len());
    let (Some(&first), Some(&last)) = (src.first(), src.get(n.wrapping_sub(1))) else { return };
    if r == 0 {
        dst[..n].copy_from_slice(&src[..n]);
        return;
    }
    let at = |i: i64| -> f64 { f64::from(src.get(i.clamp(0, n as i64 - 1) as usize).copied().unwrap_or(0.0)) };
    // Window at x = 0: r copies of the first sample, then src[0..=r], then copies of the last.
    let inside = r.min(n - 1);
    let mut acc = r as f64 * f64::from(first) + src[..=inside].iter().map(|v| f64::from(*v)).sum::<f64>() + (r - inside) as f64 * f64::from(last);
    let norm = 1.0 / (2 * r + 1) as f64;
    let ri = r as i64;
    for (x, d) in dst[..n].iter_mut().enumerate() {
        *d = (acc * norm) as f32;
        let x = x as i64;
        acc += at(x + ri + 1) - at(x - ri);
    }
}

/// Select › Modify › Smooth: box average of radius `round(r)` (edges replicated), then a hard
/// threshold at ½, as in Photoshop: straight edges stay exactly where they were, corners round
/// off and specks drop (measured on Photoshop 27.11, Smooth 8 on a 32 px square: every sampled
/// pixel 0 or 255, corner cut where the 17×17 average crosses ½). The contrast curve
/// `(v − ½)·4 + ½` used before left grey, half-selected pixels along every edge.
pub(crate) fn smooth(m: &[f32], w: usize, h: usize, r: f32) -> Vec<f32> {
    let curve = |v: f32| if v >= 0.5 { 1.0 } else { 0.0 };
    let n = w.saturating_mul(h);
    if m.len() != n || n == 0 {
        return m.iter().map(|v| curve(*v)).collect();
    }
    let r = if r.is_finite() {
        (r.max(0.0).round() as usize).min(MAX_BOX_R)
    } else if r > 0.0 {
        MAX_BOX_R
    } else {
        0
    };
    let Some(b) = nonzero_bounds(m, w, h) else {
        return vec![curve(0.0); n];
    };
    // Grown one past the radius, so a window edge inside the canvas sits on zeros and edge
    // replication there equals the zeros beyond it.
    let win = grow(b, r.saturating_add(1), w, h);
    let (gw, gh) = (win.w(), win.h());
    // Rows: only the bounding box's rows are non-zero.
    let mut tmp = vec![0.0f32; b.h() * gw];
    par_rows(&mut tmp, gw, 1, |y, row| {
        let o = (b.y0 + y) * w + win.x0;
        if let Some(s) = m.get(o..o + gw) {
            box_line(s, row, r);
        }
    });
    // Columns: a running sum of rows per strip of columns.
    let rows = Rows { tmp: &tmp, gw, bh: b.h(), off: b.y0 - win.y0, gh };
    let res = strips(gw, |c0, sw| box_strip(&rows, c0, sw, r));
    drop(tmp);
    assemble(&res, win, w, h, curve(0.0), curve)
}

/// The vertical box pass over columns `c0..c0 + sw` (edges replicated).
fn box_strip(rows: &Rows, c0: usize, sw: usize, r: usize) -> Vec<f32> {
    let gh = rows.gh;
    let mut out = vec![0.0f32; gh * sw];
    if gh == 0 || sw == 0 {
        return out;
    }
    let mut acc = vec![0.0f64; sw];
    // Window at row 0: r copies of row 0, rows 0..=r, then copies of the last row.
    let inside = r.min(gh - 1);
    add_row(&mut acc, rows, 0, c0, r as f64);
    for j in 0..=inside {
        add_row(&mut acc, rows, j, c0, 1.0);
    }
    add_row(&mut acc, rows, gh - 1, c0, (r - inside) as f64);
    let norm = 1.0 / (2 * r + 1) as f64;
    for (yy, dst) in out.chunks_exact_mut(sw).enumerate() {
        for (d, a) in dst.iter_mut().zip(&acc) {
            *d = (a * norm) as f32;
        }
        add_row(&mut acc, rows, (yy + r + 1).min(gh - 1), c0, 1.0);
        add_row(&mut acc, rows, yy.saturating_sub(r), c0, -1.0);
    }
    out
}

// ---- Feather: truncated Gaussian, zero outside -----------------------------------------------

/// Sliding cosine-series form of a symmetric kernel over `-r..=r`.
struct CosKernel {
    r: usize,
    a0: f64,
    a: [f64; TERMS],
    /// `e^{-iω}`: shifts the window by one sample.
    rot: [(f64, f64); TERMS],
    /// `e^{-iωr}` (leaving sample) and `e^{iωr}` (entering sample).
    out_w: [(f64, f64); TERMS],
    in_w: [(f64, f64); TERMS],
    /// `Σ_{|j|≤r} cos(ωj)`: a window full of ones.
    full: [f64; TERMS],
    omega: [f64; TERMS],
}

enum Kernel {
    /// Taps for `-r..=r`.
    Direct(Vec<f32>),
    Cos(Box<CosKernel>),
}

#[allow(clippy::needless_range_loop)] // textbook elimination reads clearer with indices
/// Solves the small dense system `a·x = b` (partial pivoting); `None` when singular.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|i, j| a[*i][c].abs().total_cmp(&a[*j][c].abs()))?;
        if a[p][c].abs() < 1e-12 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for i in c + 1..n {
            let f = a[i][c] / a[c][c];
            for j in c..n {
                a[i][j] -= f * a[c][j];
            }
            b[i] -= f * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|j| a[i][j] * x[j]).sum();
        x[i] = (b[i] - s) / a[i][i];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

impl Kernel {
    /// The feather kernel (σ, truncation radius `r`) as applied along lines of length `n`.
    fn new(sigma: f64, r: usize, n: usize) -> Kernel {
        // Normalized over the full truncated window, as the direct definition does.
        let g = |j: f64| (-(j * j) / (2.0 * sigma * sigma)).exp();
        let total: f64 = 1.0 + 2.0 * (1..=r).map(|j| g(j as f64)).sum::<f64>();
        // Only offsets inside the line matter.
        let jr = r.min(n.saturating_sub(1));
        if jr == 0 {
            // One-sample line: the centre tap alone.
            return Kernel::Direct(vec![(1.0 / total) as f32]);
        }
        let taps: Vec<f64> = (0..=jr).map(|j| g(j as f64) / total).collect();
        let direct = || Kernel::Direct((0..=2 * jr).map(|i| taps[i.abs_diff(jr)] as f32).collect());
        if jr <= DIRECT_MAX_R {
            return direct();
        }
        // Least squares over j = -jr..=jr (symmetric: weight 2 for j > 0).
        let l = (jr + 1) as f64;
        let omega: [f64; TERMS] = std::array::from_fn(|k| std::f64::consts::PI * (k + 1) as f64 / l);
        let basis = |q: usize, j: usize| if q == 0 { 1.0 } else { (omega[q - 1] * j as f64).cos() };
        let wt = |j: usize| if j == 0 { 1.0 } else { 2.0 };
        let mut a = vec![vec![0.0; TERMS + 1]; TERMS + 1];
        let mut b = vec![0.0; TERMS + 1];
        for (j, t) in taps.iter().enumerate() {
            let phi: Vec<f64> = (0..=TERMS).map(|q| basis(q, j)).collect();
            for p in 0..=TERMS {
                b[p] += wt(j) * phi[p] * t;
                for q in 0..=TERMS {
                    a[p][q] += wt(j) * phi[p] * phi[q];
                }
            }
        }
        let Some(c) = solve(a, b) else { return direct() };
        // Error bound per pass: half the L1 distance between the series and the kernel.
        let err: f64 = taps.iter().enumerate().map(|(j, t)| wt(j) * ((0..=TERMS).map(|q| c[q] * basis(q, j)).sum::<f64>() - t).abs()).sum::<f64>() / 2.0;
        if !err.is_finite() || err > FIT_TOL {
            return direct();
        }
        let jf = jr as f64;
        let full: [f64; TERMS] = std::array::from_fn(|k| 1.0 + 2.0 * (1..=jr).map(|j| (omega[k] * j as f64).cos()).sum::<f64>());
        Kernel::Cos(Box::new(CosKernel {
            r: jr,
            a0: c[0],
            a: std::array::from_fn(|k| c[k + 1]),
            rot: std::array::from_fn(|k| (omega[k].cos(), -omega[k].sin())),
            out_w: std::array::from_fn(|k| ((omega[k] * jf).cos(), -(omega[k] * jf).sin())),
            in_w: std::array::from_fn(|k| ((omega[k] * jf).cos(), (omega[k] * jf).sin())),
            full,
            omega,
        }))
    }

    /// Convolves one line (zero outside it) into `dst`.
    fn line(&self, src: &[f32], dst: &mut [f32], runs: &mut Vec<usize>) {
        let n = src.len().min(dst.len());
        match self {
            Kernel::Direct(taps) => {
                let r = taps.len() / 2;
                dst[..n].fill(0.0);
                for (i, k) in taps.iter().enumerate() {
                    // Output x reads src[x + i - r].
                    let (d0, s0) = if i >= r { (0, i - r) } else { (r - i, 0) };
                    if d0 >= n || s0 >= n {
                        continue;
                    }
                    let len = (n - d0).min(n - s0);
                    for (d, s) in dst[d0..d0 + len].iter_mut().zip(&src[s0..s0 + len]) {
                        *d += k * s;
                    }
                }
            }
            Kernel::Cos(k) => k.line(&src[..n], &mut dst[..n], runs),
        }
    }

    /// The vertical pass over columns `c0..c0 + sw` (zero outside the window).
    fn strip(&self, rows: &Rows, c0: usize, sw: usize) -> Vec<f32> {
        match self {
            Kernel::Direct(taps) => {
                let r = taps.len() / 2;
                let mut out = vec![0.0f32; rows.gh * sw];
                if sw == 0 {
                    return out;
                }
                for (yy, dst) in out.chunks_exact_mut(sw).enumerate() {
                    for (i, k) in taps.iter().enumerate() {
                        let Some(src) = (yy + i).checked_sub(r).and_then(|y| rows.get(y, c0, sw)) else { continue };
                        for (d, s) in dst.iter_mut().zip(src) {
                            *d += k * s;
                        }
                    }
                }
                out
            }
            Kernel::Cos(k) => k.strip(rows, c0, sw),
        }
    }
}

impl CosKernel {
    #[allow(clippy::needless_range_loop)] // `x` and `k` index several arrays in lockstep
    fn line(&self, src: &[f32], dst: &mut [f32], runs: &mut Vec<usize>) {
        let n = src.len();
        let r = self.r;
        // runs[i]: end (exclusive) of the constant run containing i.
        runs.clear();
        runs.resize(n, n);
        for i in (0..n.saturating_sub(1)).rev() {
            runs[i] = if src[i] == src[i + 1] { runs[i + 1] } else { i + 1 };
        }
        let at = |i: usize| f64::from(src.get(i).copied().unwrap_or(0.0));
        // Window sums for the last position handled (valid when `have`).
        let mut have = false;
        let mut sb = 0.0f64;
        let mut s = [(0.0f64, 0.0f64); TERMS];
        for x in 0..n {
            let lo = x.saturating_sub(r);
            let hi = (x + r + 1).min(n);
            let c = src[lo];
            if runs[lo] >= hi && (c == 0.0 || (x >= r && x + r < n)) {
                // The whole window is one value (zeros may run off the line; ones may not).
                dst[x] = c;
                let cf = f64::from(c);
                sb = cf * (2 * r + 1) as f64;
                for (sk, fk) in s.iter_mut().zip(&self.full) {
                    *sk = (cf * fk, 0.0);
                }
                have = true;
                continue;
            }
            if have {
                // Slide from x - 1: drop src[x - 1 - r], take src[x + r].
                let gone = if x > r { at(x - 1 - r) } else { 0.0 };
                let come = at(x + r);
                sb += come - gone;
                for k in 0..TERMS {
                    let (re, im) = (s[k].0 - gone * self.out_w[k].0, s[k].1 - gone * self.out_w[k].1);
                    let (cr, ci) = self.rot[k];
                    s[k] = (re * cr - im * ci + come * self.in_w[k].0, re * ci + im * cr + come * self.in_w[k].1);
                }
            } else {
                // Direct sums over the in-line part of the window.
                sb = 0.0;
                s = [(0.0, 0.0); TERMS];
                for (i, v) in src.iter().enumerate().take(hi).skip(lo) {
                    let v = f64::from(*v);
                    if v == 0.0 {
                        continue;
                    }
                    let j = i as f64 - x as f64;
                    sb += v;
                    for (sk, w) in s.iter_mut().zip(&self.omega) {
                        sk.0 += v * (w * j).cos();
                        sk.1 += v * (w * j).sin();
                    }
                }
                have = true;
            }
            let v = self.a0 * sb + s.iter().zip(&self.a).map(|(sk, a)| a * sk.0).sum::<f64>();
            dst[x] = v as f32;
        }
    }
}

impl CosKernel {
    /// [`CosKernel::line`] down columns `c0..c0 + sw`, all columns of a row at once.
    #[allow(clippy::needless_range_loop)] // `k` indexes several term arrays in lockstep
    fn strip(&self, rows: &Rows, c0: usize, sw: usize) -> Vec<f32> {
        let gh = rows.gh;
        let r = self.r;
        let mut out = vec![0.0f32; gh * sw];
        if sw == 0 {
            return out;
        }
        // Rows whose strip segment is one value, and the runs of equal such rows.
        let flag: Vec<Option<f32>> = (0..gh)
            .map(|yy| match rows.get(yy, c0, sw) {
                None => Some(0.0),
                Some(s) => {
                    let c = s.first().copied().unwrap_or(0.0);
                    s.iter().all(|v| *v == c).then_some(c)
                }
            })
            .collect();
        let mut runs = vec![gh; gh];
        for i in (0..gh.saturating_sub(1)).rev() {
            runs[i] = match (flag[i], flag[i + 1]) {
                (Some(a), Some(b)) if a == b => runs[i + 1],
                _ => i + 1,
            };
        }
        let zeros = vec![0.0f32; sw];
        let mut sb = vec![0.0f64; sw];
        let mut re = vec![vec![0.0f64; sw]; TERMS];
        let mut im = vec![vec![0.0f64; sw]; TERMS];
        let mut acc = vec![0.0f64; sw];
        let mut have = false;
        for (yy, dst) in out.chunks_exact_mut(sw).enumerate() {
            let lo = yy.saturating_sub(r);
            let hi = (yy + r + 1).min(gh);
            // The whole window is one value (zero rows may run off the window; others may not).
            if let Some(c) = flag[lo]
                && runs[lo] >= hi
                && (c == 0.0 || (yy >= r && yy + r < gh))
            {
                dst.fill(c);
                let cf = f64::from(c);
                sb.fill(cf * (2 * r + 1) as f64);
                for k in 0..TERMS {
                    re[k].fill(cf * self.full[k]);
                    im[k].fill(0.0);
                }
                have = true;
                continue;
            }
            if have {
                // Slide from yy - 1: drop row yy - 1 - r, take row yy + r.
                let gone = if yy > r { rows.get(yy - 1 - r, c0, sw) } else { None }.unwrap_or(&zeros);
                let come = rows.get(yy + r, c0, sw).unwrap_or(&zeros);
                for ((s, g), c) in sb.iter_mut().zip(gone).zip(come) {
                    *s += f64::from(*c) - f64::from(*g);
                }
                for k in 0..TERMS {
                    let (cr, ci) = self.rot[k];
                    let (o0, o1) = self.out_w[k];
                    let (i0, i1) = self.in_w[k];
                    for (((sr, si), g), c) in re[k].iter_mut().zip(im[k].iter_mut()).zip(gone).zip(come) {
                        let (g, c) = (f64::from(*g), f64::from(*c));
                        let (a, b) = (*sr - g * o0, *si - g * o1);
                        *sr = a * cr - b * ci + c * i0;
                        *si = a * ci + b * cr + c * i1;
                    }
                }
            } else {
                // Direct sums over the in-window rows.
                sb.fill(0.0);
                for k in 0..TERMS {
                    re[k].fill(0.0);
                    im[k].fill(0.0);
                }
                for i in lo..hi {
                    let Some(row) = rows.get(i, c0, sw) else { continue };
                    let j = i as f64 - yy as f64;
                    add_row(&mut sb, rows, i, c0, 1.0);
                    for k in 0..TERMS {
                        let (cs, sn) = ((self.omega[k] * j).cos(), (self.omega[k] * j).sin());
                        for ((sr, si), v) in re[k].iter_mut().zip(im[k].iter_mut()).zip(row) {
                            *sr += f64::from(*v) * cs;
                            *si += f64::from(*v) * sn;
                        }
                    }
                }
                have = true;
            }
            for (a, s) in acc.iter_mut().zip(&sb) {
                *a = self.a0 * s;
            }
            for k in 0..TERMS {
                for (a, s) in acc.iter_mut().zip(&re[k]) {
                    *a += self.a[k] * s;
                }
            }
            for (d, a) in dst.iter_mut().zip(&acc) {
                *d = *a as f32;
            }
        }
        out
    }
}

/// Select › Modify › Feather: Gaussian blur of the mask with `σ = radius`, the kernel truncated
/// at `⌈3σ⌉` and normalized, zero beyond the canvas. Photoshop's feather radius is the Gaussian's
/// σ (measured on Photoshop 27.11; the same mapping the mask feather uses, fitted on the
/// psd-tools corpus in `photocraft_compose::masks::feather_sigma`). It was `radius / 2`: half
/// as wide as Photoshop's.
pub(crate) fn feather(m: &[f32], w: usize, h: usize, radius: f32) -> Vec<f32> {
    let n = w.saturating_mul(h);
    let sigma = if radius.is_finite() {
        radius.max(0.0)
    } else if radius > 0.0 {
        MAX_SIGMA
    } else {
        0.0
    };
    if sigma < 0.1 || m.len() != n || n == 0 {
        return m.to_vec();
    }
    let sigma = sigma.min(MAX_SIGMA);
    let r = (sigma * 3.0).ceil() as usize;
    let Some(b) = nonzero_bounds(m, w, h) else {
        return m.to_vec();
    };
    let win = grow(b, r, w, h);
    let (gw, gh) = (win.w(), win.h());
    let s = f64::from(sigma);
    let (kh, kv) = (Kernel::new(s, r, gw), Kernel::new(s, r, gh));
    // Rows: only the bounding box's rows are non-zero.
    // The run table is scratch: one buffer per worker thread, reused across rows instead of
    // one allocation per row (thousands of 8-byte-per-sample Vecs on a 24 MP mask).
    thread_local! { static RUNS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) } }
    let mut tmp = vec![0.0f32; b.h() * gw];
    par_rows(&mut tmp, gw, 1, |y, row| {
        let o = (b.y0 + y) * w + win.x0;
        if let Some(s) = m.get(o..o + gw) {
            RUNS.with(|runs| kh.line(s, row, &mut runs.borrow_mut()));
        }
    });
    // Columns, a strip of columns at a time.
    let rows = Rows { tmp: &tmp, gw, bh: b.h(), off: b.y0 - win.y0, gh };
    let res = strips(gw, |c0, sw| kv.strip(&rows, c0, sw));
    drop(tmp);
    // Rounding residue of the sliding sums is not coverage.
    assemble(&res, win, w, h, 0.0, |v| if v.abs() < 1e-7 { 0.0 } else { v })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::photo_util::Rng;

    /// The previous direct implementations, kept as the oracle.
    mod oracle {
        pub fn box_blur(m: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
            if r == 0 {
                return m.to_vec();
            }
            let mut tmp = vec![0.0f32; m.len()];
            let win = (2 * r + 1) as f32;
            for y in 0..h {
                for x in 0..w {
                    let mut s = 0.0;
                    for k in x as i64 - r as i64..=x as i64 + r as i64 {
                        s += m[y * w + k.clamp(0, w as i64 - 1) as usize];
                    }
                    tmp[y * w + x] = s / win;
                }
            }
            let mut out = vec![0.0f32; m.len()];
            for y in 0..h {
                for x in 0..w {
                    let mut s = 0.0;
                    for k in y as i64 - r as i64..=y as i64 + r as i64 {
                        s += tmp[k.clamp(0, h as i64 - 1) as usize * w + x];
                    }
                    out[y * w + x] = s / win;
                }
            }
            out
        }

        pub fn smooth(m: &[f32], w: usize, h: usize, r: f32) -> Vec<f32> {
            box_blur(m, w, h, r.max(0.0).round() as usize).into_iter().map(|v| if v >= 0.5 { 1.0 } else { 0.0 }).collect()
        }

        pub fn feather(m: &[f32], w: usize, h: usize, radius: f32) -> Vec<f32> {
            let sigma = radius.max(0.0);
            if sigma < 0.1 {
                return m.to_vec();
            }
            let r = (sigma * 3.0).ceil() as i64;
            let k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
            let s: f32 = k.iter().sum();
            let k: Vec<f32> = k.iter().map(|v| v / s).collect();
            let mut tmp = vec![0.0f32; m.len()];
            for y in 0..h {
                for x in 0..w {
                    let mut a = 0.0;
                    for (i, kv) in k.iter().enumerate() {
                        let xx = x as i64 + i as i64 - r;
                        if xx >= 0 && xx < w as i64 {
                            a += m[y * w + xx as usize] * kv;
                        }
                    }
                    tmp[y * w + x] = a;
                }
            }
            let mut out = vec![0.0f32; m.len()];
            for y in 0..h {
                for x in 0..w {
                    let mut a = 0.0;
                    for (i, kv) in k.iter().enumerate() {
                        let yy = y as i64 + i as i64 - r;
                        if yy >= 0 && yy < h as i64 {
                            a += tmp[yy as usize * w + x] * kv;
                        }
                    }
                    out[y * w + x] = a;
                }
            }
            out
        }
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    /// Random masks of several kinds: binary noise, soft noise, blobs with soft edges, a sparse
    /// scatter and an off-centre rectangle.
    fn masks(w: usize, h: usize, seed: u64) -> Vec<(&'static str, Vec<f32>)> {
        let mut rng = Rng(seed);
        let mut u = move || (rng.next() >> 11) as f32 / (1u64 << 53) as f32;
        let binary: Vec<f32> = (0..w * h).map(|_| if u() < 0.5 { 1.0 } else { 0.0 }).collect();
        let soft: Vec<f32> = (0..w * h).map(|_| u()).collect();
        let (cx, cy, rad) = (w as f32 * u(), h as f32 * u(), (w.min(h) as f32 * 0.4).max(1.0));
        let blob: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 + 0.5, (i / w) as f32 + 0.5);
                (rad - ((x - cx).powi(2) + (y - cy).powi(2)).sqrt()).clamp(0.0, 1.0)
            })
            .collect();
        let sparse: Vec<f32> = (0..w * h).map(|_| if u() < 0.02 { 1.0 } else { 0.0 }).collect();
        let rect: Vec<f32> = (0..w * h).map(|i| if (i % w) * 3 >= w && (i / w) * 4 < h * 3 { 1.0 } else { 0.0 }).collect();
        vec![("binary", binary), ("soft", soft), ("blob", blob), ("sparse", sparse), ("rect", rect)]
    }

    /// Photoshop 27.11, measured: a 32×32 rectangle at (16,16) in 64×64, Select › Modify ›
    /// Feather 4 px, filled white. Row y = 32 for x = 8..=24, then (16,16), (18,18), (20,20).
    #[test]
    fn feather_matches_photoshop() {
        const ROW: [u8; 17] = [7, 13, 22, 34, 50, 70, 92, 115, 140, 163, 185, 205, 221, 233, 242, 248, 252];
        const DIAG: [(usize, u8); 3] = [(16, 76), (18, 134), (20, 190)];
        let (w, h) = (64, 64);
        let m: Vec<f32> = (0..w * h).map(|i| if (16..48).contains(&(i % w)) && (16..48).contains(&(i / w)) { 1.0 } else { 0.0 }).collect();
        let f = feather(&m, w, h, 4.0);
        let v = |x: usize, y: usize| (f[y * w + x] * 255.0).round() as i32;
        for (i, want) in ROW.iter().enumerate() {
            let x = 8 + i;
            assert!((v(x, 32) - i32::from(*want)).abs() <= 3, "x {x}: {} vs Photoshop {want}", v(x, 32));
        }
        // Photoshop's corners come out a few levels darker than a true 2-D Gaussian's; the row
        // is the defining measurement.
        for (d, want) in DIAG {
            assert!((v(d, d) - i32::from(want)).abs() <= 4, "({d},{d}): {} vs Photoshop {want}", v(d, d));
        }
    }

    #[test]
    fn feather_matches_direct_gaussian() {
        for (w, h, seed) in [(97, 61, 1), (40, 150, 2), (13, 7, 3)] {
            for (kind, m) in masks(w, h, seed) {
                for radius in [0.1, 0.3, 1.0, 2.5, 8.0, 9.0, 17.0, 25.0, 50.0, 120.0, 1000.0] {
                    let d = max_diff(&feather(&m, w, h, radius), &oracle::feather(&m, w, h, radius));
                    assert!(d <= 1.0 / 255.0, "{kind} {w}x{h} r {radius}: {d}");
                }
            }
        }
    }

    /// Photoshop 27.11, measured: a 32×32 rectangle at (16,16) in 64×64, Select › Modify ›
    /// Smooth 8 px, filled white. The straight edge stays hard at x = 16; only the corner rounds.
    #[test]
    fn smooth_matches_photoshop() {
        let (w, h) = (64, 64);
        let m: Vec<f32> = (0..w * h).map(|i| if (16..48).contains(&(i % w)) && (16..48).contains(&(i / w)) { 1.0 } else { 0.0 }).collect();
        let s = smooth(&m, w, h, 8.0);
        let row: Vec<f32> = (8..=24).map(|x| s[32 * w + x]).collect();
        let want_row: Vec<f32> = (8..=24).map(|x| if x >= 16 { 1.0 } else { 0.0 }).collect();
        assert_eq!(row, want_row);
        let diag: Vec<f32> = (16..=24).map(|d| s[d * w + d]).collect();
        assert_eq!(diag, [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn smooth_matches_direct_box() {
        for (w, h, seed) in [(97, 61, 4), (40, 150, 5), (13, 7, 6)] {
            for (kind, m) in masks(w, h, seed) {
                for radius in [0.0, 0.4, 1.0, 2.0, 5.0, 16.0, 50.0, 500.0] {
                    // The threshold is hard, so a pixel may only differ where the box average sits
                    // within float rounding of ½ (the running sum adds in another order).
                    let (fast, slow) = (smooth(&m, w, h, radius), oracle::smooth(&m, w, h, radius));
                    let avg = oracle::box_blur(&m, w, h, radius.max(0.0).round() as usize);
                    for (i, (a, b)) in fast.iter().zip(&slow).enumerate() {
                        assert!(a == b || (avg[i] - 0.5).abs() < 1e-4, "{kind} {w}x{h} r {radius} pixel {i}: {a} vs {b} (average {})", avg[i]);
                    }
                }
            }
        }
    }

    #[test]
    fn edge_cases() {
        // Empty selection stays empty; nothing selected far from a selection stays exactly 0.
        let z = vec![0.0; 50 * 40];
        assert_eq!(feather(&z, 50, 40, 30.0), z);
        assert_eq!(smooth(&z, 50, 40, 10.0), z);
        let mut dot = z.clone();
        dot[20 * 50 + 25] = 1.0;
        let f = feather(&dot, 50, 40, 4.0);
        assert_eq!(f[0], 0.0);
        assert!(f[20 * 50 + 25] > 0.0);
        // A full canvas: smooth keeps it; feather fades only at the canvas edge, as before.
        let one = vec![1.0; 50 * 40];
        assert_eq!(smooth(&one, 50, 40, 25.0), one);
        let f = feather(&one, 50, 40, 6.0);
        assert!((f[20 * 50 + 25] - 1.0).abs() < 1e-5);
        let f30 = feather(&one, 50, 40, 30.0);
        assert!(max_diff(&f30, &oracle::feather(&one, 50, 40, 30.0)) <= 1.0 / 255.0);
        assert!(f[0] < 0.5);
        assert!(max_diff(&f, &oracle::feather(&one, 50, 40, 6.0)) <= 1.0 / 255.0);
        // Radius 0 is identity (feather) / the bare contrast curve (smooth).
        let m: Vec<f32> = (0..50 * 40).map(|i| (i % 7) as f32 / 6.0).collect();
        assert_eq!(feather(&m, 50, 40, 0.0), m);
        assert_eq!(smooth(&m, 50, 40, 0.0), oracle::smooth(&m, 50, 40, 0.0));
        // 1×1 canvas and empty canvases.
        for r in [0.0, 1.0, 50.0, 1000.0] {
            assert!((feather(&[1.0], 1, 1, r)[0] - oracle::feather(&[1.0], 1, 1, r)[0]).abs() <= 1.0 / 255.0);
            assert_eq!(smooth(&[1.0], 1, 1, r), vec![1.0]);
        }
        assert!(feather(&[], 0, 0, 5.0).is_empty());
        assert!(smooth(&[], 0, 5, 5.0).is_empty());
        // Non-finite or absurd radii and a mismatched buffer return without panicking.
        for r in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -5.0, 1e30] {
            let f = feather(&dot, 50, 40, r);
            assert_eq!(f.len(), dot.len());
            assert!(f.iter().all(|v| v.is_finite()), "{r}");
            let s = smooth(&dot, 50, 40, r);
            assert_eq!(s.len(), dot.len());
        }
        assert_eq!(feather(&dot, 50, 41, 5.0).len(), dot.len());
        assert_eq!(smooth(&dot, 49, 40, 5.0).len(), dot.len());
    }

    /// Timings on a 24 MP ellipse (run with `--release -- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn bench_24mp() {
        let (w, h) = (6000usize, 4000usize);
        let m: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 / w as f32 - 0.5, (i / w) as f32 / h as f32 - 0.5);
                if x * x + y * y < 0.11 { 1.0 } else { 0.0 }
            })
            .collect();
        // `SEL_BENCH=feather:250` runs one case (e.g. under `time` for its CPU cost).
        let only = std::env::var("SEL_BENCH").ok();
        for (name, r) in [("smooth", 50.0), ("feather", 50.0), ("feather", 250.0), ("feather", 1000.0)] {
            if only.as_ref().is_some_and(|o| *o != format!("{name}:{r}")) {
                continue;
            }
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                let out = if name == "smooth" { smooth(&m, w, h, r) } else { feather(&m, w, h, r) };
                best = best.min(t.elapsed().as_secs_f64() * 1000.0);
                assert_eq!(out.len(), m.len());
            }
            eprintln!("{name} r {r}: {best:.1} ms (best of 5)");
        }
    }

    #[test]
    fn cosine_fit_is_used_and_accurate() {
        for sigma in [7.0, 25.0, 125.0] {
            let r = (sigma * 3.0f64).ceil() as usize;
            match Kernel::new(sigma, r, 4000) {
                Kernel::Cos(_) => {}
                _ => panic!("σ {sigma}: expected the sliding cosine kernel"),
            }
        }
    }
}
