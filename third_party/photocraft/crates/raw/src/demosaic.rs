//! Bayer demosaicing.
//!
//! * **Bilinear**: each missing colour is the mean of the same-colour
//!   neighbours in the 3×3 window.
//! * **Malvar–He–Cutler**: H. S. Malvar, L.-W. He, R. Cutler, "High-quality
//!   linear interpolation for demosaicing of Bayer-pattern color images",
//!   ICASSP 2004: bilinear plus a gradient-correction term (5×5 kernels).
//! * **AHD**: K. Hirakawa, T. W. Parks, "Adaptive homogeneity-directed
//!   demosaicing algorithm", IEEE Trans. Image Processing 14(3), 2005:
//!   horizontally and vertically interpolated candidates, chosen per pixel by
//!   their local homogeneity in CIELAB.
//!
//! Input is a [`Padded`] plane of normalized CFA samples (mirror-reflected
//! borders, which keep the CFA parity, so every kernel reads in bounds);
//! `phase` gives the colours (0 = R, 1 = G, 2 = B) of the 2×2 cell at the
//! image origin, row major. Output is interleaved RGB. Work is split in row
//! bands run in parallel.
//!
//! Other periodic CFAs (Fujifilm X-Trans' 6×6 pattern) go through
//! [`demosaic_cfa`], a pattern-agnostic edge-directed interpolation:
//!
//! 1. Green at non-green sites: per axis (horizontal, vertical), the mean of
//!    the green neighbours on that axis, weighted by the inverse square of the
//!    local gradient along it (mean absolute difference of same-colour sample
//!    pairs on the axis within a 5×5 window), as in gradient-weighted
//!    interpolation (e.g. W. Lu, Y.-P. Tan, "Color filter array demosaicking:
//!    new method and performance measures", IEEE TIP 12(10), 2003).
//! 2. Red and blue everywhere from colour differences: the green estimate plus
//!    the weighted mean of (sample − green) at the same-colour sites of the 5×5
//!    window, weighted by inverse squared distance and by green similarity (so
//!    differences are not carried across edges).
//!
//! The pattern breaks under mirror padding when its period is not 2, so this
//! path reads only interior samples and drops taps outside the image.

use crate::color::Mat3;
use crate::par;

/// Demosaicing algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Demosaic {
    /// Fastest; soft, with colour fringes on fine detail.
    Bilinear,
    /// Malvar–He–Cutler gradient-corrected linear interpolation: fast and sharp.
    Mhc,
    /// Adaptive homogeneity-directed (Hirakawa–Parks): best edges, slower.
    #[default]
    Ahd,
}

impl Demosaic {
    pub fn id(self) -> &'static str {
        match self {
            Demosaic::Bilinear => "bilinear",
            Demosaic::Mhc => "mhc",
            Demosaic::Ahd => "ahd",
        }
    }

    pub fn from_id(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "bilinear" | "fast" => Some(Demosaic::Bilinear),
            "mhc" | "malvar" => Some(Demosaic::Mhc),
            "ahd" | "best" => Some(Demosaic::Ahd),
            _ => None,
        }
    }
}

/// Border width of a [`Padded`] plane (even, so the CFA parity is unchanged).
pub(crate) const PAD: usize = 6;

/// Mirror-reflects `i` into `0..n` (n ≥ 1); clamps when `n` is tiny.
#[inline]
fn reflect(i: isize, n: usize) -> usize {
    let n = n as isize;
    if n <= 1 {
        return 0;
    }
    let mut i = i;
    if i < 0 {
        i = -i;
    }
    if i >= n {
        i = 2 * (n - 1) - i;
    }
    i.clamp(0, n - 1) as usize
}

/// A `w × h` plane stored with [`PAD`] extra pixels on every side.
pub(crate) struct Padded {
    pub data: Vec<f32>,
    pub w: usize,
    pub h: usize,
    pub stride: usize,
}

impl Padded {
    pub fn new(w: usize, h: usize) -> Self {
        let stride = w + 2 * PAD;
        Padded { data: vec![0.0; stride * (h + 2 * PAD)], w, h, stride }
    }

    /// From an unpadded plane (tests and small inputs).
    #[cfg(test)]
    pub fn from_plane(plane: &[f32], w: usize, h: usize) -> Self {
        let mut p = Padded::new(w, h);
        for y in 0..h {
            let at = (y + PAD) * p.stride + PAD;
            p.data[at..at + w].copy_from_slice(&plane[y * w..(y + 1) * w]);
        }
        p.fill_borders();
        p
    }

    /// Mutable interior rows, `rows` at a time, for parallel filling: calls
    /// `f(first_row, rows_slice)` where each row is `stride` long and the
    /// image pixels start at offset [`PAD`].
    pub fn fill_rows(&mut self, rows: usize, f: impl Fn(usize, &mut [f32]) + Sync + Send) {
        let (s, h) = (self.stride, self.h);
        let interior = &mut self.data[PAD * s..(PAD + h) * s];
        par::chunks_mut(interior, rows.max(1) * s, |b, chunk| f(b * rows.max(1), chunk));
    }

    /// Fills the borders by mirror reflection of the interior.
    pub fn fill_borders(&mut self) {
        let (w, h, s) = (self.w, self.h, self.stride);
        if w == 0 || h == 0 {
            return;
        }
        for y in PAD..PAD + h {
            let row = &mut self.data[y * s..(y + 1) * s];
            for px in (0..PAD).chain(PAD + w..s) {
                row[px] = row[PAD + reflect(px as isize - PAD as isize, w)];
            }
        }
        for py in (0..PAD).chain(PAD + h..h + 2 * PAD) {
            let src = PAD + reflect(py as isize - PAD as isize, h);
            self.data.copy_within(src * s..(src + 1) * s, py * s);
        }
    }
}

/// Per 2×2 position, per output colour: kernel taps as (offset, weight).
fn kernels(phase: [u8; 4], method: Demosaic, stride: usize) -> [[Vec<(isize, f32)>; 3]; 4] {
    let color = |x: isize, y: isize| phase[((y & 1) * 2 + (x & 1)) as usize];
    let s = stride as isize;
    std::array::from_fn(|pos| {
        let (px, py) = ((pos & 1) as isize, (pos >> 1) as isize);
        let own = color(px, py);
        std::array::from_fn(|c| {
            let c = c as u8;
            let taps: Vec<(isize, isize, f32)> = if c == own {
                vec![(0, 0, 1.0)]
            } else if method == Demosaic::Bilinear {
                let t: Vec<(isize, isize)> =
                    (-1..=1).flat_map(|dy| (-1..=1).map(move |dx| (dx, dy))).filter(|&(dx, dy)| color(px + dx, py + dy) == c).collect();
                let w = 1.0 / t.len().max(1) as f32;
                t.into_iter().map(|(dx, dy)| (dx, dy, w)).collect()
            } else if c == 1 {
                // Malvar–He–Cutler (weights / 8): G at R or B.
                vec![(0, 0, 4.0), (-1, 0, 2.0), (1, 0, 2.0), (0, -1, 2.0), (0, 1, 2.0), (-2, 0, -1.0), (2, 0, -1.0), (0, -2, -1.0), (0, 2, -1.0)]
            } else if own == 1 {
                // R or B at G: from the row neighbours or the column neighbours.
                let (ax, ay) = if color(px + 1, py) == c { (1, 0) } else { (0, 1) };
                vec![
                    (0, 0, 5.0),
                    (-ax, -ay, 4.0),
                    (ax, ay, 4.0),
                    (-1, -1, -1.0),
                    (1, -1, -1.0),
                    (-1, 1, -1.0),
                    (1, 1, -1.0),
                    (-2 * ax, -2 * ay, -1.0),
                    (2 * ax, 2 * ay, -1.0),
                    (-2 * ay, -2 * ax, 0.5),
                    (2 * ay, 2 * ax, 0.5),
                ]
            } else {
                // R at B or B at R.
                vec![(0, 0, 6.0), (-1, -1, 2.0), (1, -1, 2.0), (-1, 1, 2.0), (1, 1, 2.0), (-2, 0, -1.5), (2, 0, -1.5), (0, -2, -1.5), (0, 2, -1.5)]
            };
            let norm = if method == Demosaic::Bilinear || c == own { 1.0 } else { 1.0 / 8.0 };
            taps.into_iter().map(|(dx, dy, w)| (dy * s + dx, w * norm)).collect()
        })
    })
}

/// Demosaics a padded CFA plane into interleaved RGB (`w × h × 3`).
pub(crate) fn demosaic(p: &Padded, phase: [u8; 4], method: Demosaic, to_xyz: &Mat3, ahd_jobs: usize) -> Vec<f32> {
    let (w, h, s) = (p.w, p.h, p.stride);
    let mut out = vec![0.0f32; w * h * 3];
    if w == 0 || h == 0 || p.data.len() < s * (h + 2 * PAD) {
        return out;
    }
    if method == Demosaic::Ahd {
        ahd(p, phase, to_xyz, &mut out, ahd_jobs);
        return out;
    }
    let ks = kernels(phase, method, s);
    let rows = par::band_rows(w);
    let d = &p.data;
    par::chunks_mut(&mut out, rows * w * 3, |band, chunk| {
        let y0 = band * rows;
        for (r, row) in chunk.chunks_exact_mut(w * 3).enumerate() {
            let y = y0 + r;
            let base = (y + PAD) * s + PAD;
            for (x, o) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                let pos = (y & 1) * 2 + (x & 1);
                let i = base + x;
                if method == Demosaic::Bilinear {
                    // Direct Bayer bilinear (the kernels' result without the tap loops).
                    let own = usize::from(phase[pos]).min(2);
                    let (n, so, e, wv) = (d[i - s], d[i + s], d[i + 1], d[i - 1]);
                    if own == 1 {
                        let hc = usize::from(phase[(y & 1) * 2 + ((x + 1) & 1)]).min(2);
                        o[1] = d[i];
                        o[hc] = (e + wv) * 0.5;
                        o[2 - hc] = (n + so) * 0.5;
                    } else {
                        o[own] = d[i];
                        o[1] = (n + so + e + wv) * 0.25;
                        o[2 - own] = (d[i - s - 1] + d[i - s + 1] + d[i + s - 1] + d[i + s + 1]) * 0.25;
                    }
                    continue;
                }
                for (c, k) in ks[pos].iter().enumerate() {
                    let mut v = 0.0f32;
                    for &(off, wt) in k {
                        v += wt * d[(i as isize + off) as usize];
                    }
                    o[c] = v.clamp(0.0, 1.0);
                }
            }
        }
    });
    out
}

/// A CFA repeat as seen from the plane's origin: colour of plane pixel
/// (x, y) is `colors[(y % h) * w + x % w]`.
#[derive(Debug, Clone)]
pub(crate) struct Pattern {
    pub w: usize,
    pub h: usize,
    pub colors: Vec<u8>,
}

impl Pattern {
    /// The repeat of `cfa` starting at data coordinates (`x0`, `y0`).
    pub fn of(cfa: &crate::sensor::Cfa, x0: usize, y0: usize) -> Self {
        let (w, h) = (cfa.width.clamp(1, 16), cfa.height.clamp(1, 16));
        Pattern { w, h, colors: (0..w * h).map(|i| cfa.color(x0 + i % w, y0 + i / w).min(2)).collect() }
    }

    fn at(&self, x: isize, y: isize) -> u8 {
        let (w, h) = (self.w.max(1) as isize, self.h.max(1) as isize);
        self.colors.get((y.rem_euclid(h) * w + x.rem_euclid(w)) as usize).copied().unwrap_or(1)
    }
}

/// Tap radius of the generic demosaic (a 5×5 window).
const RADIUS: isize = 2;

type Off = (isize, isize);

/// One interpolation axis at a non-green site.
#[derive(Default)]
struct Axis {
    /// Green neighbours at distance 1 on the axis.
    near: Vec<Off>,
    /// Same-colour sample pairs along the axis (a, b, 1 / distance).
    pairs: Vec<(Off, Off, f32)>,
}

/// The taps of one CFA phase.
struct Taps {
    own: u8,
    axes: [Axis; 2],
    /// Greens of the 3×3 (or 5×5) window, when no axis has a green neighbour.
    greens: Vec<Off>,
    /// Per colour, its sites in the window with weight 1 / distance².
    sites: [Vec<(Off, f32)>; 3],
}

fn phase_taps(pat: &Pattern, px: isize, py: isize) -> Taps {
    let col = |dx: isize, dy: isize| pat.at(px + dx, py + dy);
    let window: Vec<Off> = (-RADIUS..=RADIUS).flat_map(|dy| (-RADIUS..=RADIUS).map(move |dx| (dx, dy))).collect();
    let axes = [(1isize, 0isize), (0, 1)].map(|(ax, ay)| {
        let near = [(ax, ay), (-ax, -ay)].into_iter().filter(|&(dx, dy)| col(dx, dy) == 1).collect();
        let mut pairs = Vec::new();
        for &(dx, dy) in &window {
            // Pairs on the axis line or the lines next to it.
            if (ax == 1 && dy.abs() > 1) || (ay == 1 && dx.abs() > 1) {
                continue;
            }
            for k in 1..=2 {
                let (bx, by) = (dx + k * ax, dy + k * ay);
                if bx.abs() <= RADIUS && by.abs() <= RADIUS && col(dx, dy) == col(bx, by) {
                    pairs.push(((dx, dy), (bx, by), 1.0 / k as f32));
                }
            }
        }
        Axis { near, pairs }
    });
    let ring = |r: isize| -> Vec<Off> {
        window.iter().copied().filter(|&(dx, dy)| dx.abs() <= r && dy.abs() <= r && (dx, dy) != (0, 0) && col(dx, dy) == 1).collect()
    };
    let greens = Some(ring(1)).filter(|g| !g.is_empty()).unwrap_or_else(|| ring(RADIUS));
    let sites = [0u8, 1, 2].map(|c| {
        window.iter().filter(|&&(dx, dy)| (dx, dy) != (0, 0) && col(dx, dy) == c).map(|&(dx, dy)| ((dx, dy), 1.0 / (dx * dx + dy * dy) as f32)).collect()
    });
    Taps { own: col(0, 0), axes, greens, sites }
}

/// Demosaics a padded plane with any periodic CFA (see the module docs) into
/// interleaved RGB. There is one algorithm for every [`Demosaic`] choice:
/// without the edge-directed weights it is no faster and leaves zipper
/// artefacts along every edge of an X-Trans image.
pub(crate) fn demosaic_cfa(p: &Padded, pat: &Pattern) -> Vec<f32> {
    let (w, h, s) = (p.w, p.h, p.stride);
    let mut out = vec![0.0f32; w * h * 3];
    if w == 0 || h == 0 || pat.w == 0 || pat.h == 0 || pat.colors.len() < pat.w * pat.h || p.data.len() < s * (h + 2 * PAD) {
        return out;
    }
    let taps: Vec<Taps> = (0..pat.w * pat.h).map(|i| phase_taps(pat, (i % pat.w) as isize, (i / pat.w) as isize)).collect();
    let tap_at = |x: usize, y: usize| &taps[(y % pat.h) * pat.w + x % pat.w];
    let d = &p.data;
    // Sample at (x + dx, y + dy), or None outside the image.
    let get = |plane: &[f32], stride: usize, pad: usize, x: usize, y: usize, (dx, dy): Off| -> Option<f32> {
        let (xx, yy) = (x.checked_add_signed(dx)?, y.checked_add_signed(dy)?);
        (xx < w && yy < h).then(|| plane.get((yy + pad) * stride + pad + xx).copied()).flatten()
    };
    let raw = |x: usize, y: usize, o: Off| get(d, s, PAD, x, y, o);
    let band = par::band_rows(w);

    // 1. Green everywhere.
    let mut green = vec![0.0f32; w * h];
    par::chunks_mut(&mut green, band * w, |b, chunk| {
        for (r, row) in chunk.chunks_exact_mut(w).enumerate() {
            let y = b * band + r;
            for (x, g) in row.iter_mut().enumerate() {
                let t = tap_at(x, y);
                let v = raw(x, y, (0, 0)).unwrap_or(0.0);
                if t.own == 1 {
                    *g = v;
                    continue;
                }
                let (mut num, mut den) = (0.0f32, 0.0f32);
                for a in &t.axes {
                    let (mut sum, mut n) = (0.0f32, 0u32);
                    for &o in &a.near {
                        if let Some(v) = raw(x, y, o) {
                            sum += v;
                            n += 1;
                        }
                    }
                    if n == 0 {
                        continue;
                    }
                    let (mut grad, mut m) = (0.0f32, 0.0f32);
                    for &(oa, ob, k) in &a.pairs {
                        if let (Some(va), Some(vb)) = (raw(x, y, oa), raw(x, y, ob)) {
                            grad += (va - vb).abs() * k;
                            m += 1.0;
                        }
                    }
                    let grad = if m > 0.0 { grad / m } else { 0.0 };
                    let wt = 1.0 / ((GRAD_EPS + grad) * (GRAD_EPS + grad));
                    num += wt * sum / n as f32;
                    den += wt;
                }
                *g = if den > 0.0 {
                    num / den
                } else {
                    let (mut sum, mut n) = (0.0f32, 0u32);
                    for &o in &t.greens {
                        if let Some(v) = raw(x, y, o) {
                            sum += v;
                            n += 1;
                        }
                    }
                    if n > 0 { sum / n as f32 } else { v }
                };
            }
        }
    });

    // 2. Red and blue from colour differences.
    let gref = &green;
    let gat = |x: usize, y: usize, o: Off| get(gref, w, 0, x, y, o);
    par::chunks_mut(&mut out, band * w * 3, |b, chunk| {
        for (r, row) in chunk.chunks_exact_mut(w * 3).enumerate() {
            let y = b * band + r;
            for (x, o) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                let t = tap_at(x, y);
                let g = gref.get(y * w + x).copied().unwrap_or(0.0);
                let v = raw(x, y, (0, 0)).unwrap_or(0.0);
                o[1] = g;
                for c in [0usize, 2] {
                    if usize::from(t.own) == c {
                        o[c] = v;
                        continue;
                    }
                    let (mut num, mut den) = (0.0f32, 0.0f32);
                    for &(off, base) in &t.sites[c] {
                        if let (Some(vq), Some(gq)) = (raw(x, y, off), gat(x, y, off)) {
                            let wt = base / (DIFF_EPS + (gq - g).abs());
                            num += wt * (vq - gq);
                            den += wt;
                        }
                    }
                    o[c] = if den > 0.0 { g + num / den } else { g };
                }
                for v in o.iter_mut() {
                    *v = v.clamp(0.0, 1.0);
                }
            }
        }
    });
    out
}

/// Regularizes the inverse-gradient weights (normalized units, about 1/256 of full scale).
const GRAD_EPS: f32 = 1.0 / 256.0;
/// Regularizes the green-similarity weights of the colour differences.
const DIFF_EPS: f32 = 1.0 / 64.0;

/// CIE f(t) for L*a*b*, tabulated on 0..2 with linear interpolation.
struct LabF {
    table: Vec<f32>,
}

const LABF_STEPS: usize = 1 << 14;
const LABF_RANGE: f32 = 2.0;

impl LabF {
    fn new() -> Self {
        LabF { table: (0..=LABF_STEPS + 1).map(|i| Self::exact(i as f32 * LABF_RANGE / LABF_STEPS as f32)).collect() }
    }

    fn exact(t: f32) -> f32 {
        if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 }
    }

    #[inline]
    fn f(&self, t: f32) -> f32 {
        if !(0.0..LABF_RANGE).contains(&t) {
            return Self::exact(t.max(0.0));
        }
        let x = t * (LABF_STEPS as f32 / LABF_RANGE);
        let i = (x as usize).min(LABF_STEPS);
        let a = self.table[i];
        a + (self.table[i + 1] - a) * (x - i as f32)
    }
}

/// Output rows per AHD band.
const AHD_BAND: usize = 32;
/// Extra band rows above and below the output rows.
const M: usize = 3;

/// Maximum scratch allocation of one AHD band: two green, RGB and Lab float
/// planes and two byte homogeneity planes, including their border rows.
pub(crate) fn ahd_scratch(w: u64, h: u64) -> u64 {
    w.saturating_add(2 * PAD as u64).saturating_mul(h.min(AHD_BAND as u64).saturating_add(2 * M as u64)).saturating_mul(58)
}

pub(crate) fn ahd_bands(h: usize) -> usize {
    h.div_ceil(AHD_BAND)
}

pub(crate) const AHD_TABLE_BYTES: u64 = (LABF_STEPS as u64 + 2) * 4;

fn ahd(p: &Padded, phase: [u8; 4], to_xyz: &Mat3, out: &mut [f32], jobs: usize) {
    let (w, s) = (p.w, p.stride);
    let d = &p.data;
    // Colour at padded coordinates (PAD is even, so the phase is unchanged).
    let color = |px: usize, py: usize| phase[(py & 1) * 2 + (px & 1)];
    let white = [0.9642f32, 1.0, 0.8249];
    let m: [[f32; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| to_xyz[i][j] as f32 / white[i]));
    let labf = LabF::new();
    par::chunks_mut_limited(out, AHD_BAND * w * 3, jobs, |band, chunk| {
        let rows = chunk.len() / (w * 3);
        let n = rows + 2 * M;
        // Band row b ↔ padded row py(b); padded columns are used as is.
        let py = |b: usize| band * AHD_BAND + PAD - M + b;
        // 1. Green interpolated horizontally (0) and vertically (1), columns 2..s-2.
        let mut g = [vec![0.0f32; n * s], vec![0.0f32; n * s]];
        for b in 0..n {
            let y = py(b);
            for x in 2..s - 2 {
                let i = y * s + x;
                let c = d[i];
                let k = b * s + x;
                if color(x, y) == 1 {
                    g[0][k] = c;
                    g[1][k] = c;
                    continue;
                }
                let (l, r) = (d[i - 1], d[i + 1]);
                let gh = (l + r) * 0.5 + (2.0 * c - d[i - 2] - d[i + 2]) * 0.25;
                g[0][k] = gh.clamp(l.min(r), l.max(r));
                let (u, dn) = (d[i - s], d[i + s]);
                let gv = (u + dn) * 0.5 + (2.0 * c - d[i - 2 * s] - d[i + 2 * s]) * 0.25;
                g[1][k] = gv.clamp(u.min(dn), u.max(dn));
            }
        }
        // 2. RGB candidates and their Lab, band rows 1..n-1, columns 3..s-3.
        let mut rgb = [vec![0.0f32; n * s * 3], vec![0.0f32; n * s * 3]];
        let mut lab = [vec![0.0f32; n * s * 3], vec![0.0f32; n * s * 3]];
        for dir in 0..2 {
            let gd = &g[dir];
            // Colour difference (sample − green) at band row b, column x.
            let diff = |b: usize, x: usize| d[py(b) * s + x] - gd[b * s + x];
            let (rgb_d, lab_d) = (&mut rgb[dir], &mut lab[dir]);
            for b in 1..n - 1 {
                let y = py(b);
                for x in 3..s - 3 {
                    let own = color(x, y);
                    let gv = gd[b * s + x];
                    let mut px = [gv; 3];
                    if own == 1 {
                        let hc = usize::from(color(x + 1, y)).min(2);
                        px[hc] = gv + 0.5 * (diff(b, x - 1) + diff(b, x + 1));
                        px[2 - hc] = gv + 0.5 * (diff(b - 1, x) + diff(b + 1, x));
                    } else {
                        let o = usize::from(own).min(2);
                        px[o] = d[y * s + x];
                        px[2 - o] = gv + 0.25 * (diff(b - 1, x - 1) + diff(b - 1, x + 1) + diff(b + 1, x - 1) + diff(b + 1, x + 1));
                    }
                    let px = px.map(|v| v.clamp(0.0, 1.0));
                    let i = (b * s + x) * 3;
                    rgb_d[i..i + 3].copy_from_slice(&px);
                    let xyz: [f32; 3] = std::array::from_fn(|k| m[k][0] * px[0] + m[k][1] * px[1] + m[k][2] * px[2]);
                    let (fx, fy, fz) = (labf.f(xyz[0]), labf.f(xyz[1]), labf.f(xyz[2]));
                    lab_d[i] = 116.0 * fy - 16.0;
                    lab_d[i + 1] = 500.0 * (fx - fy);
                    lab_d[i + 2] = 200.0 * (fy - fz);
                }
            }
        }
        // 3. Homogeneity maps, band rows 2..n-2, columns 4..s-4.
        let mut hom = [vec![0u8; n * s], vec![0u8; n * s]];
        let at = |l: &[f32], k: usize| (l[k * 3], l[k * 3 + 1], l[k * 3 + 2]);
        let dist = |a: (f32, f32, f32), c: (f32, f32, f32)| ((a.0 - c.0).abs(), (a.1 - c.1) * (a.1 - c.1) + (a.2 - c.2) * (a.2 - c.2));
        for b in 2..n - 2 {
            for x in 4..s - 4 {
                let k = b * s + x;
                let nb = [k - 1, k + 1, k - s, k + s];
                let (ch, cv) = (at(&lab[0], k), at(&lab[1], k));
                let dh: [(f32, f32); 4] = nb.map(|j| dist(ch, at(&lab[0], j)));
                let dv: [(f32, f32); 4] = nb.map(|j| dist(cv, at(&lab[1], j)));
                // The horizontal candidate judged along the row, the vertical along the column.
                let eps_l = dh[0].0.max(dh[1].0).min(dv[2].0.max(dv[3].0));
                let eps_c = dh[0].1.max(dh[1].1).min(dv[2].1.max(dv[3].1));
                hom[0][k] = dh.iter().filter(|(l, c)| *l <= eps_l && *c <= eps_c).count() as u8;
                hom[1][k] = dv.iter().filter(|(l, c)| *l <= eps_l && *c <= eps_c).count() as u8;
            }
        }
        // 4. Per pixel, the candidate with the larger 3×3 homogeneity sum.
        for r in 0..rows {
            let b = r + M;
            for x in 0..w {
                let px = x + PAD;
                let (mut sh, mut sv) = (0u32, 0u32);
                for bb in b - 1..=b + 1 {
                    for xx in px - 1..=px + 1 {
                        sh += u32::from(hom[0][bb * s + xx]);
                        sv += u32::from(hom[1][bb * s + xx]);
                    }
                }
                let i = (b * s + px) * 3;
                let o = &mut chunk[(r * w + x) * 3..(r * w + x) * 3 + 3];
                for k in 0..3 {
                    o[k] = match sh.cmp(&sv) {
                        std::cmp::Ordering::Greater => rgb[0][i + k],
                        std::cmp::Ordering::Less => rgb[1][i + k],
                        std::cmp::Ordering::Equal => 0.5 * (rgb[0][i + k] + rgb[1][i + k]),
                    };
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::IDENTITY;

    #[test]
    fn ahd_band_concurrency_does_not_change_pixels() {
        let (w, h) = (53, 103);
        let cfa: Vec<f32> = (0..w * h).map(|i| ((i * 97 + i / w * 31) % 1009) as f32 / 1009.0).collect();
        let p = Padded::from_plane(&cfa, w, h);
        let one = demosaic(&p, [0, 1, 1, 2], Demosaic::Ahd, &IDENTITY, 1);
        for jobs in [2, 3, 4, 128] {
            assert_eq!(one, demosaic(&p, [0, 1, 1, 2], Demosaic::Ahd, &IDENTITY, jobs));
        }
    }

    /// Mosaics an RGB image with the given phase.
    fn mosaic(rgb: &[[f32; 3]], w: usize, h: usize, phase: [u8; 4]) -> Vec<f32> {
        (0..w * h).map(|i| rgb[i][phase[((i / w) & 1) * 2 + ((i % w) & 1)] as usize]).collect()
    }

    fn run(cfa: &[f32], w: usize, h: usize, phase: [u8; 4], m: Demosaic) -> Vec<f32> {
        demosaic(&Padded::from_plane(cfa, w, h), phase, m, &IDENTITY, par::workers())
    }

    fn psnr(a: &[f32], b: &[[f32; 3]], w: usize, h: usize, border: usize) -> f64 {
        let mut se = 0.0f64;
        let mut n = 0usize;
        for y in border..h - border {
            for x in border..w - border {
                for c in 0..3 {
                    let d = f64::from(a[(y * w + x) * 3 + c] - b[y * w + x][c]);
                    se += d * d;
                    n += 1;
                }
            }
        }
        10.0 * (1.0 / (se / n as f64).max(1e-12)).log10()
    }

    #[test]
    fn padding_reflects() {
        let p = Padded::from_plane(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let row = |y: usize| p.data[(y + PAD) * p.stride + PAD - 2..(y + PAD) * p.stride + PAD + 5].to_vec();
        assert_eq!(row(0), vec![3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]);
        // Row −1 mirrors row 1.
        let r = PAD - 1;
        assert_eq!(p.data[r * p.stride + PAD..r * p.stride + PAD + 3].to_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn flat_colour_is_exact() {
        let (w, h) = (16, 12);
        let rgb = vec![[0.2f32, 0.5, 0.8]; w * h];
        for phase in [[0, 1, 1, 2], [2, 1, 1, 0], [1, 0, 2, 1], [1, 2, 0, 1]] {
            for m in [Demosaic::Bilinear, Demosaic::Mhc, Demosaic::Ahd] {
                let out = run(&mosaic(&rgb, w, h, phase), w, h, phase, m);
                for (i, v) in out.as_chunks::<3>().0.iter().enumerate() {
                    for c in 0..3 {
                        assert!((v[c] - rgb[i][c]).abs() < 1e-5, "{m:?} {phase:?} px {i} c {c}: {}", v[c]);
                    }
                }
            }
        }
    }

    #[test]
    fn smooth_gradient_quality() {
        // A smooth colour field: every method should be accurate.
        let (w, h) = (64, 48);
        let rgb: Vec<[f32; 3]> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 / w as f32, (i / w) as f32 / h as f32);
                [0.2 + 0.6 * x, 0.3 + 0.4 * (x * y), 0.8 - 0.5 * y]
            })
            .collect();
        let phase = [0, 1, 1, 2];
        let cfa = mosaic(&rgb, w, h, phase);
        let p: Vec<f64> = [Demosaic::Bilinear, Demosaic::Mhc, Demosaic::Ahd].iter().map(|&m| psnr(&run(&cfa, w, h, phase, m), &rgb, w, h, 2)).collect();
        assert!(p.iter().all(|&v| v > 40.0), "{p:?}");
    }

    #[test]
    fn edges_ahd_beats_bilinear() {
        // Vertical and horizontal black/white stripes of width 3: AHD's
        // directional choice must reduce error versus bilinear.
        let (w, h) = (48, 48);
        let rgb: Vec<[f32; 3]> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let v = if y < h / 2 { ((x / 3) % 2) as f32 } else { ((y / 3) % 2) as f32 };
                [0.1 + 0.8 * v, 0.1 + 0.8 * v, 0.1 + 0.8 * v]
            })
            .collect();
        let phase = [0, 1, 1, 2];
        let cfa = mosaic(&rgb, w, h, phase);
        let bil = psnr(&run(&cfa, w, h, phase, Demosaic::Bilinear), &rgb, w, h, 3);
        let ahd = psnr(&run(&cfa, w, h, phase, Demosaic::Ahd), &rgb, w, h, 3);
        let mhc = psnr(&run(&cfa, w, h, phase, Demosaic::Mhc), &rgb, w, h, 3);
        assert!(ahd > bil + 3.0, "ahd {ahd:.1} dB vs bilinear {bil:.1} dB");
        assert!(mhc > bil, "mhc {mhc:.1} dB vs bilinear {bil:.1} dB");
    }

    #[test]
    fn bilinear_fast_path_matches_kernels() {
        // The direct bilinear code must equal the generic kernel evaluation.
        let (w, h) = (20, 14);
        let cfa: Vec<f32> = (0..w * h).map(|i| ((i * 7919) % 1000) as f32 / 1000.0).collect();
        for phase in [[0, 1, 1, 2], [1, 2, 0, 1]] {
            let p = Padded::from_plane(&cfa, w, h);
            let fast = demosaic(&p, phase, Demosaic::Bilinear, &IDENTITY, par::workers());
            let ks = kernels(phase, Demosaic::Bilinear, p.stride);
            for y in 0..h {
                for x in 0..w {
                    let i = (y + PAD) * p.stride + PAD + x;
                    for (c, k) in ks[(y & 1) * 2 + (x & 1)].iter().enumerate() {
                        let v: f32 = k.iter().map(|&(off, wt)| wt * p.data[(i as isize + off) as usize]).sum();
                        assert!((fast[(y * w + x) * 3 + c] - v).abs() < 1e-6, "({x},{y}) c{c}");
                    }
                }
            }
        }
    }

    #[test]
    fn lab_table_is_accurate() {
        let l = LabF::new();
        for i in 0..10000 {
            let t = i as f32 / 4000.0;
            assert!((l.f(t) - LabF::exact(t)).abs() < 2e-4, "{t}");
        }
    }

    /// The X-Trans repeat as observed at a sensor's data origin.
    const XTRANS: [&str; 6] = ["GGRGGB", "GGBGGR", "BRGRBG", "GGBGGR", "GGRGGB", "RBGBRG"];

    fn pattern(rows: &[&str]) -> Pattern {
        let colors = rows
            .concat()
            .chars()
            .map(|c| match c {
                'R' => 0,
                'G' => 1,
                _ => 2,
            })
            .collect();
        Pattern { w: rows[0].len(), h: rows.len(), colors }
    }

    fn mosaic_pat(rgb: &[[f32; 3]], w: usize, pat: &Pattern) -> Vec<f32> {
        (0..rgb.len()).map(|i| rgb[i][usize::from(pat.at((i % w) as isize, (i / w) as isize))]).collect()
    }

    #[test]
    fn xtrans_flat_colour_is_exact() {
        let (w, h) = (25, 19);
        let rgb = vec![[0.2f32, 0.5, 0.8]; w * h];
        let pat = pattern(&XTRANS);
        // Every phase of the pattern at the plane origin.
        for (sx, sy) in [(0, 0), (1, 0), (2, 3), (5, 5)] {
            let shifted = Pattern { colors: (0..36).map(|i| pat.at((i % 6 + sx) as isize, (i / 6 + sy) as isize)).collect(), ..pat.clone() };
            let out = demosaic_cfa(&Padded::from_plane(&mosaic_pat(&rgb, w, &shifted), w, h), &shifted);
            for (i, v) in out.as_chunks::<3>().0.iter().enumerate() {
                for c in 0..3 {
                    assert!((v[c] - rgb[i][c]).abs() < 1e-5, "shift ({sx},{sy}) px {i} c {c}: {}", v[c]);
                }
            }
        }
    }

    #[test]
    fn xtrans_quality() {
        let pat = pattern(&XTRANS);
        // A smooth colour field.
        let (w, h) = (66, 48);
        let rgb: Vec<[f32; 3]> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 / w as f32, (i / w) as f32 / h as f32);
                [0.2 + 0.6 * x, 0.3 + 0.4 * (x * y), 0.8 - 0.5 * y]
            })
            .collect();
        let out = demosaic_cfa(&Padded::from_plane(&mosaic_pat(&rgb, w, &pat), w, h), &pat);
        let smooth = psnr(&out, &rgb, w, h, 2);
        assert!(smooth > 40.0, "smooth field {smooth:.1} dB");
        // Grey stripes 3 pixels wide, vertical then horizontal: the edge-directed
        // green must beat a plain neighbour average by a wide margin.
        let (w, h) = (48, 48);
        let rgb: Vec<[f32; 3]> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let v = if y < h / 2 { ((x / 3) % 2) as f32 } else { ((y / 3) % 2) as f32 };
                [0.1 + 0.8 * v; 3]
            })
            .collect();
        let cfa = mosaic_pat(&rgb, w, &pat);
        let adaptive = psnr(&demosaic_cfa(&Padded::from_plane(&cfa, w, h), &pat), &rgb, w, h, 3);
        // Plain average of the 3×3 greens, colour differences unweighted.
        let plain: Vec<f32> = (0..w * h)
            .flat_map(|i| {
                let (x, y) = ((i % w) as isize, (i / w) as isize);
                let mean = |c: u8| {
                    let v: Vec<f32> = (-1..=1)
                        .flat_map(|dy| (-1..=1).map(move |dx| (x + dx, y + dy)))
                        .filter(|&(xx, yy)| xx >= 0 && yy >= 0 && xx < w as isize && yy < h as isize && pat.at(xx, yy) == c)
                        .map(|(xx, yy)| cfa[yy as usize * w + xx as usize])
                        .collect();
                    if v.is_empty() { 0.5 } else { v.iter().sum::<f32>() / v.len() as f32 }
                };
                [mean(0), mean(1), mean(2)]
            })
            .collect();
        let plain = psnr(&plain, &rgb, w, h, 3);
        assert!(adaptive > plain + 6.0, "adaptive {adaptive:.1} dB vs plain {plain:.1} dB");
    }

    #[test]
    fn odd_patterns_do_not_panic() {
        // A 3×3 pattern without blue, a 1×1 all-green one, and a 2×2 non-Bayer one.
        for rows in [&["RGG", "GGR", "GRG"][..], &["G"][..], &["RG", "BR"][..]] {
            let pat = pattern(rows);
            for (w, h) in [(1, 1), (2, 3), (7, 5), (13, 1)] {
                let out = demosaic_cfa(&Padded::from_plane(&vec![0.5; w * h], w, h), &pat);
                assert_eq!(out.len(), w * h * 3);
                assert!(out.iter().all(|v| v.is_finite()));
            }
        }
        // A degenerate pattern is rejected, not indexed out of range.
        let bad = Pattern { w: 6, h: 6, colors: vec![1; 4] };
        assert!(demosaic_cfa(&Padded::from_plane(&[0.5; 4], 2, 2), &bad).iter().all(|v| *v == 0.0));
    }

    #[test]
    fn tiny_planes_do_not_panic() {
        for (w, h) in [(1, 1), (2, 1), (1, 3), (2, 2), (3, 3), (7, 9), (40, 1)] {
            let cfa = vec![0.5f32; w * h];
            for m in [Demosaic::Bilinear, Demosaic::Mhc, Demosaic::Ahd] {
                assert_eq!(run(&cfa, w, h, [0, 1, 1, 2], m).len(), w * h * 3);
            }
        }
    }
}
