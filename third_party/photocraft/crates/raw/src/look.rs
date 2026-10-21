//! A per-file colour rendering ("camera look") fitted to the camera's own
//! embedded JPEG, for raw files without colour calibration (NEF, ARW, CR2,
//! RW2, ORF…), where the neutral fallback (camera channels treated as sRGB
//! primaries, no tone curve) renders flat, grey and slightly green.
//!
//! Method (after LightCraft's prose description in `docs/camera-preview-colour.md`,
//! storytold/lightcraft#165; this is an independent implementation against this
//! crate's pipeline): both the sensor data and the camera JPEG are reduced to a
//! common grid of at most [`PROXY_EDGE`] cells on the long edge, over the whole
//! (upright) frame. Each cell pairs the white-balanced camera RGB `c` (the same
//! white balance [`crate::develop`] applies) with the JPEG's colour in linear
//! ProPhoto RGB `y`. The model is the develop pipeline itself:
//!
//! ```text
//! y ≈ f(M · c)     M: 3×3, scaled so the balanced white keeps the exposure gain's luminance
//!                  f: one monotone tone curve, applied per channel to linear ProPhoto
//! ```
//!
//! fitted by alternating: `f` by binned (square-root domain), pool-adjacent-
//! violators isotonic regression of `y` on `M · c`, through (0, 0) and (1, 1);
//! `M` by weighted least squares of `f⁻¹(y)` on `c`, weighted so the residual
//! approximates a CIE L* difference, ridge-pulled toward the neutral fallback,
//! with Tukey biweights against misaligned edges and local tone mapping. The
//! ridge starts weak and is strengthened when the matrix comes out implausible
//! (scenes with little colour keep the fallback's saturation while the neutral
//! axis, the camera's rendering of white, still follows the JPEG). Cells with a
//! clipped sensor sample, clipped or near-black JPEG pixels are left out of the
//! fit; cells on strong edges get less weight.
//! A bounded centered alignment (90–100% of the raw frame's dimensions) can
//! account for a camera JPEG's tighter framing. It changes only the fit proxy,
//! requires correlation ≥ 0.9 and an improvement ≥ 0.05, and preserves the
//! developed crop. Alignment uses training cells; held-out cells stay out of it.
//!
//! The reference must show the same picture (lightness correlation ≥ 0.7 with
//! the raw) in colour (≥ 2 % of the cells with chroma above 4). One cell in
//! three is held out. The look is accepted only when its matrix is plausible
//! (relative to the gain: diagonal 0.2–3, off-diagonals within ±2, rows summing
//! to 0.6–1.6, determinant above 0.05) and it lowers the held-out mean CIE76 ΔE
//! (Lab D50) to at most 85 % of the neutral fallback's. This is a per-file
//! estimate of the camera's rendering, not a measured calibration: picture
//! styles, local tone mapping (Active D-Lighting and the like) and
//! hue-dependent rendering are approximated by one matrix and one curve.

use std::fmt;

use crate::color::{self, Mat3};
use crate::develop::{DevelopOptions, Scale, resolve_neutral, source_xy};
use crate::par;
use crate::sensor::Sensor;

/// Long edge of the fit grid, in cells.
pub const PROXY_EDGE: usize = 256;

/// The camera's rendering of the picture: upright, interleaved 8-bit sRGB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reference<'a> {
    pub width: u32,
    pub height: u32,
    /// `width * height * 3` samples.
    pub rgb: &'a [u8],
}

/// A monotone tone curve on linear values 0..1, piecewise linear between knots
/// in the square-root domain. Always passes through (0, 0) and (1, 1).
#[derive(Debug, Clone, PartialEq)]
pub struct ToneCurve {
    /// Knots `(sqrt x, sqrt y)`, x strictly and y weakly increasing, from (0, 0) to (1, 1).
    knots: Vec<[f64; 2]>,
}

impl ToneCurve {
    /// Most knots a curve may have.
    pub const MAX_KNOTS: usize = 256;

    /// A curve through linear-light knots `(x, y)`: 2 to [`Self::MAX_KNOTS`] finite
    /// points in 0..1, x strictly increasing, y not decreasing, starting at (0, 0)
    /// and ending at (1, 1). `None` otherwise.
    pub fn new(knots: &[[f64; 2]]) -> Option<ToneCurve> {
        if knots.len() < 2 || knots.len() > Self::MAX_KNOTS {
            return None;
        }
        let ok = knots.iter().all(|k| k.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)))
            && knots.windows(2).all(|w| w[1][0] > w[0][0] && w[1][1] >= w[0][1])
            && knots.first().is_some_and(|k| k[0] == 0.0 && k[1] == 0.0)
            && knots.last().is_some_and(|k| k[0] == 1.0 && k[1] == 1.0);
        ok.then(|| ToneCurve { knots: knots.iter().map(|k| [k[0].sqrt(), k[1].sqrt()]).collect() })
    }

    /// The knots in linear light.
    pub fn knots(&self) -> Vec<[f64; 2]> {
        self.knots.iter().map(|k| [k[0] * k[0], k[1] * k[1]]).collect()
    }

    /// `f(x)` for linear `x` (clamped to 0..1).
    pub fn eval(&self, x: f64) -> f64 {
        let t = if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0).sqrt() };
        let s = interp(&self.knots, t, 0, 1);
        s * s
    }

    /// `f⁻¹(y)` (the lowest `x` where a flat part repeats `y`).
    #[cfg(test)]
    fn inverse(&self, y: f64) -> f64 {
        self.inverse_and_slope(y).0
    }

    /// `(x, f'(x))` for `x = f⁻¹(y)`, from one segment lookup: with `sx = sqrt x`,
    /// `sy = sqrt y` and segment slope `k = dsy/dsx`, `f'(x) = k · sy / sx`.
    fn inverse_and_slope(&self, y: f64) -> (f64, f64) {
        let t = if y.is_nan() { 0.0 } else { y.clamp(0.0, 1.0).sqrt() };
        let k = &self.knots;
        let i = k.partition_point(|p| p[1] < t).clamp(1, k.len().saturating_sub(1).max(1));
        let (Some(a), Some(b)) = (k.get(i - 1), k.get(i)) else { return (y, 1.0) };
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        if dy <= 0.0 || dx <= 0.0 {
            return (a[0] * a[0], 0.0);
        }
        let sx = a[0] + dx * ((t - a[1]) / dy).clamp(0.0, 1.0);
        let slope = dy / dx;
        let d = if sx > 1e-9 { slope * t / sx } else { slope * slope };
        (sx * sx, d)
    }
}

/// Piecewise-linear interpolation of coordinate `to` at `t` on coordinate `from`
/// (both increasing; the first segment that reaches `t` wins).
fn interp(k: &[[f64; 2]], t: f64, from: usize, to: usize) -> f64 {
    let i = k.partition_point(|p| p[from] < t).clamp(1, k.len().saturating_sub(1).max(1));
    match (k.get(i - 1), k.get(i)) {
        (Some(a), Some(b)) => {
            let d = b[from] - a[from];
            if d <= 0.0 { a[to] } else { a[to] + (b[to] - a[to]) * ((t - a[from]) / d).clamp(0.0, 1.0) }
        }
        _ => t,
    }
}

/// A colour rendering fitted to the camera's JPEG: use it through
/// [`DevelopOptions::camera_look`].
#[derive(Debug, Clone, PartialEq)]
pub struct CameraLook {
    /// White-balanced camera RGB → XYZ (D50), white mapping to D50 with Y = 1.
    pub to_xyz: Mat3,
    /// Tone curve applied per channel to linear ProPhoto RGB after exposure.
    pub tone: Option<ToneCurve>,
    /// Mean CIE76 ΔE against the camera JPEG on the fit grid, with this look.
    pub delta_e: f64,
    /// The same with the neutral fallback (no calibration, no tone curve).
    pub delta_e_fallback: f64,
}

impl CameraLook {
    /// Finite, bounded and invertible, with a positive white luminance.
    pub(crate) fn is_usable(&self) -> bool {
        let m = &self.to_xyz;
        m.iter().flatten().all(|v| v.is_finite() && v.abs() < 64.0) && color::invert(m).is_some() && color::apply(m, [1.0; 3])[1] > 1e-3
    }

    /// The note `develop` reports.
    pub(crate) fn note(&self) -> String {
        format!("colour fitted to the camera's embedded JPEG (mean ΔE {:.1} against it; {:.1} with the neutral fallback)", self.delta_e, self.delta_e_fallback)
    }
}

/// Why no look was fitted (the neutral fallback stays).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LookRejected(pub String);

impl fmt::Display for LookRejected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "colour not fitted to the camera JPEG: {}", self.0)
    }
}

impl std::error::Error for LookRejected {}

fn reject<T>(why: impl Into<String>) -> Result<T, LookRejected> {
    Err(LookRejected(why.into()))
}

/// One grid cell.
#[derive(Debug, Clone, Copy, Default)]
struct Cell {
    /// White-balanced camera RGB, 0..1.
    cam: [f64; 3],
    /// Any sampled sensor value at or near clipping.
    cam_clipped: bool,
    /// The JPEG's colour, linear ProPhoto RGB.
    rgb: [f64; 3],
    /// Luminance (Y, D50) of `rgb`.
    lum: f64,
    /// Fraction of the cell's JPEG pixels with a channel at 250 or more.
    ref_clipped: f64,
    /// Weight from the JPEG's local contrast (edges count less).
    edge_weight: f64,
}

/// Fits a [`CameraLook`] so that developing `s` with `opts` (white balance,
/// baseline exposure) matches the camera's `reference` rendering of it.
pub fn fit_look(s: &Sensor, opts: &DevelopOptions, reference: &Reference) -> Result<CameraLook, LookRejected> {
    let (gw, mut cells) = grid(s, opts, reference)?;
    align_preview(&mut cells, gw, s.baseline_exposure);
    fit_cells(&cells, gw, s.baseline_exposure)
}

/// Camera JPEGs can show a slightly tighter frame (lens correction, for example).
/// Align only the fit proxy, never the developed crop. Search a bounded centered
/// crop of at most 10%, and require a strong, substantially improved match.
fn align_preview(cells: &mut [Cell], w: usize, exposure: f64) {
    if w == 0 || cells.len() < w || !cells.len().is_multiple_of(w) {
        return;
    }
    let h = cells.len() / w;
    let gain = 2f64.powf(if exposure.is_finite() { exposure.clamp(-10.0, 10.0) } else { 0.0 });
    let fallback = color::scale(&color::mul(&color::xyz_d50_to_prophoto(), &color::srgb_to_xyz_d50()), gain);
    let step = (w / 64).max(1);
    let score = |scale: f64| {
        let mut pairs = Vec::new();
        for y in (0..h).step_by(step) {
            for x in (0..w).step_by(step) {
                if (x + 2 * y) % 3 == 0 {
                    continue;
                }
                let Some(c) = cells.get(y * w + x) else { continue };
                let Some((cam, clipped)) = aligned_camera(cells, w, h, x, y, scale) else { continue };
                if clipped || c.ref_clipped >= 0.01 || c.lum <= 0.002 || c.edge_weight <= 0.05 || cam.iter().any(|v| !v.is_finite() || *v <= 1e-4) {
                    continue;
                }
                let lum = color::apply(&PRO_TO_XYZ_Y_ROW, color::apply(&fallback, cam))[0];
                pairs.push((lum.max(0.0).sqrt(), c.lum.sqrt()));
            }
        }
        if pairs.len() < 100 { f64::NAN } else { correlation(&pairs) }
    };
    let original = score(1.0);
    if !original.is_finite() {
        return;
    }
    let (mut best, mut scale) = (original, 1.0);
    for i in 0..5 {
        let s = 0.9 + 0.02 * i as f64;
        let r = score(s);
        if r > best {
            (best, scale) = (r, s);
        }
    }
    if scale < 1.0 {
        let center = scale;
        for i in -7..=7 {
            let s = (center + 0.0025 * f64::from(i)).clamp(0.9, 1.0);
            let r = score(s);
            if r > best {
                (best, scale) = (r, s);
            }
        }
    }
    if best < 0.9 || best < original + 0.05 || scale >= 1.0 {
        return;
    }
    let aligned: Vec<_> = (0..cells.len()).map(|i| aligned_camera(cells, w, h, i % w, i / w, scale)).collect();
    for (c, aligned) in cells.iter_mut().zip(aligned) {
        if let Some((cam, clipped)) = aligned {
            c.cam = cam;
            c.cam_clipped = clipped;
        }
    }
}

/// Bilinear interpolation on cell centers, retaining clipping information from
/// every participating cell. Coordinates remain inside the bounded fit grid.
fn aligned_camera(cells: &[Cell], w: usize, h: usize, x: usize, y: usize, scale: f64) -> Option<([f64; 3], bool)> {
    let coord = |i: usize, n: usize| ((i as f64 + 0.5 - n as f64 / 2.0) * scale + n as f64 / 2.0 - 0.5).clamp(0.0, n.saturating_sub(1) as f64);
    let (sx, sy) = (coord(x, w), coord(y, h));
    let (x0, y0) = (sx.floor() as usize, sy.floor() as usize);
    let (fx, fy) = (sx - x0 as f64, sy - y0 as f64);
    let mut cam = [0.0; 3];
    let mut clipped = false;
    for (dx, dy, weight) in [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
        if weight <= 0.0 {
            continue;
        }
        let index = y0.saturating_add(dy).min(h.saturating_sub(1)).checked_mul(w)?.checked_add(x0.saturating_add(dx).min(w.saturating_sub(1)))?;
        let c = cells.get(index)?;
        clipped |= c.cam_clipped;
        for (out, v) in cam.iter_mut().zip(c.cam) {
            *out += weight * v;
        }
    }
    Some((cam, clipped))
}

/// The upright fit grid: (width in cells, cells row-major).
fn grid(s: &Sensor, opts: &DevelopOptions, r: &Reference) -> Result<(usize, Vec<Cell>), LookRejected> {
    let c = s.crop;
    let in_data = c.x.checked_add(c.width).is_some_and(|v| v <= s.width) && c.y.checked_add(c.height).is_some_and(|v| v <= s.height);
    let need = s.width.checked_mul(s.height).and_then(|v| v.checked_mul(s.samples));
    if c.is_empty() || !in_data || need.is_none_or(|n| s.data.len() < n) {
        return reject("the crop lies outside the sensor data");
    }
    if !(s.samples == 3 || (s.samples == 1 && s.cfa.is_some())) {
        return reject("unsupported sensor layout");
    }
    let (rw, rh) = (r.width as usize, r.height as usize);
    if rw < 32 || rh < 32 || rw.checked_mul(rh).and_then(|n| n.checked_mul(3)) != Some(r.rgb.len()) {
        return reject("the reference image is too small or its buffer does not match its size");
    }
    let swap = (5..=8).contains(&s.orientation);
    let (uw, uh) = if swap { (c.height, c.width) } else { (c.width, c.height) };
    let aspect = (rw as f64 / rh as f64) / (uw as f64 / uh as f64);
    if !(aspect.is_finite() && (aspect.ln()).abs() <= 0.03) {
        return reject(format!("the reference ({rw}x{rh}) does not have the raw frame's shape ({uw}x{uh})"));
    }
    // Cells at least 4 sensor pixels and 2 reference pixels across.
    let long = uw.max(uh);
    let edge = PROXY_EDGE.min(long / 4).min(rw.max(rh) / 2);
    let gw_up = (uw * edge).div_ceil(long).max(1);
    let gh_up = (uh * edge).div_ceil(long).max(1);
    if gw_up < 16 || gh_up < 16 {
        return reject("the image is too small to fit");
    }
    let mut warnings = Vec::new();
    let neutral = resolve_neutral(s, opts, &Scale::new(s), &mut warnings).map_err(|e| LookRejected(e.to_string()))?;
    let mult = neutral.map(|v| 1.0 / v);

    // Sensor-orientation grid.
    let (gw, gh) = if swap { (gh_up, gw_up) } else { (gw_up, gh_up) };
    let sc = Scale::new(s);
    let rows = par::map(gh, |j| {
        let (y0, y1) = (c.y + j * c.height / gh, c.y + (j + 1) * c.height / gh);
        (0..gw)
            .map(|i| {
                let (x0, x1) = (c.x + i * c.width / gw, c.x + (i + 1) * c.width / gw);
                sensor_cell(s, &sc, mult, x0, x1, y0, y1)
            })
            .collect::<Vec<_>>()
    });
    let sensor_cells: Vec<Option<([f64; 3], bool)>> = rows.into_iter().flatten().collect();

    // Reference cells, upright; then pair with the oriented sensor cells.
    let lut: Vec<f64> = (0..256).map(|v| srgb_to_linear(v as f64 / 255.0)).collect();
    let to_pro = color::mul(&color::xyz_d50_to_prophoto(), &color::srgb_to_xyz_d50());
    let pro_to_xyz = color::rgb_to_xyz(color::ROMM, color::D50_XY);
    let ref_rows = par::map(gh_up, |j| {
        let (y0, y1) = (j * rh / gh_up, ((j + 1) * rh / gh_up).max(j * rh / gh_up + 1));
        (0..gw_up)
            .map(|i| {
                let (x0, x1) = (i * rw / gw_up, ((i + 1) * rw / gw_up).max(i * rw / gw_up + 1));
                reference_cell(r, &lut, x0, x1.min(rw), y0, y1.min(rh))
            })
            .collect::<Vec<_>>()
    });
    let mut cells = Vec::with_capacity(gw_up * gh_up);
    for (j, row) in ref_rows.into_iter().enumerate() {
        for (i, rc) in row.into_iter().enumerate() {
            let (sx, sy) = source_xy(s.orientation, gw, gh, i, j);
            let Some(Some((cam, cam_clipped))) = sensor_cells.get(sy * gw + sx).copied() else {
                cells.push(Cell { cam_clipped: true, ..Default::default() });
                continue;
            };
            let Some((srgb, ref_clipped, edge_weight)) = rc else {
                cells.push(Cell { cam_clipped: true, ..Default::default() });
                continue;
            };
            let rgb = color::apply(&to_pro, srgb);
            let lum = color::apply(&pro_to_xyz, rgb)[1];
            cells.push(Cell { cam, cam_clipped, rgb, lum, ref_clipped, edge_weight });
        }
    }
    Ok((gw_up, cells))
}

/// Mean white-balanced camera RGB over `x0..x1 × y0..y1` (sampling at most
/// about 8 × 8 Bayer quads), and whether any sample was near clipping.
fn sensor_cell(s: &Sensor, sc: &Scale, mult: [f64; 3], x0: usize, x1: usize, y0: usize, y1: usize) -> Option<([f64; 3], bool)> {
    let step = |a: usize, b: usize| b.saturating_sub(a).div_ceil(16).max(1) * 2;
    let (sx, sy) = (step(x0, x1), step(y0, y1));
    let mut sum = [0.0f64; 3];
    let mut n = [0u32; 3];
    let mut clipped = false;
    let mut add = |ch: usize, v: f32| {
        let v = f64::from(v);
        let b = v * mult.get(ch).copied().unwrap_or(1.0);
        clipped |= v >= 0.98 || b >= 0.98;
        if let (Some(s), Some(k)) = (sum.get_mut(ch), n.get_mut(ch)) {
            *s += b.clamp(0.0, 1.0);
            *k += 1;
        }
    };
    let mut y = y0;
    while y < y1 {
        let mut x = x0;
        while x < x1 {
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let (px, py) = (x + dx, y + dy);
                if px >= x1 || py >= y1 {
                    continue;
                }
                match &s.cfa {
                    Some(cfa) if s.samples == 1 => add(usize::from(cfa.color(px, py)).min(2), sc.value(px, py, 0)),
                    _ => (0..3).for_each(|k| add(k, sc.value(px, py, k))),
                }
            }
            x += sx;
        }
        y += sy;
    }
    (n.iter().all(|k| *k > 0)).then(|| ([0, 1, 2].map(|c| sum[c] / f64::from(n[c])), clipped))
}

/// Mean linear sRGB of a reference box, the fraction of its pixels with a
/// clipped channel and an edge weight from its luminance spread.
fn reference_cell(r: &Reference, lut: &[f64], x0: usize, x1: usize, y0: usize, y1: usize) -> Option<([f64; 3], f64, f64)> {
    let rw = r.width as usize;
    let mut sum = [0.0f64; 3];
    let (mut l1, mut l2, mut n, mut clipped) = (0.0f64, 0.0f64, 0usize, 0usize);
    for y in y0..y1 {
        let row = r.rgb.get((y * rw + x0) * 3..(y * rw + x1) * 3)?;
        for p in row.as_chunks::<3>().0 {
            let v = p.map(|b| lut.get(usize::from(b)).copied().unwrap_or(0.0));
            for k in 0..3 {
                sum[k] += v[k];
            }
            let l = 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2];
            l1 += l;
            l2 += l * l;
            n += 1;
            clipped += usize::from(p.iter().any(|b| *b >= 250));
        }
    }
    if n == 0 {
        return None;
    }
    let nf = n as f64;
    let mean = l1 / nf;
    let sd = (l2 / nf - mean * mean).max(0.0).sqrt();
    let cv = sd / (mean + 0.01);
    Some((sum.map(|v| v / nf), clipped as f64 / nf, 1.0 / (1.0 + (cv / 0.15).powi(2))))
}

fn srgb_to_linear(v: f64) -> f64 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

/// A training / evaluation sample.
#[derive(Debug, Clone, Copy)]
struct Sample {
    cam: [f64; 3],
    rgb: [f64; 3],
    /// Luminance (Y, D50) of `rgb` and its [`perceptual`] scale.
    lum: f64,
    perceptual: f64,
    weight: f64,
}

/// CIE L*a*b* (D50) of linear ProPhoto RGB.
struct Lab {
    to_xyz: Mat3,
    white: [f64; 3],
}

impl Lab {
    fn new() -> Self {
        let to_xyz = color::rgb_to_xyz(color::ROMM, color::D50_XY);
        Lab { white: color::apply(&to_xyz, [1.0; 3]), to_xyz }
    }

    fn lab(&self, rgb: [f64; 3]) -> [f64; 3] {
        let xyz = color::apply(&self.to_xyz, rgb);
        let f = |t: f64| if t > 216.0 / 24389.0 { t.cbrt() } else { (24389.0 / 27.0 * t + 16.0) / 116.0 };
        let [fx, fy, fz] = [0, 1, 2].map(|k| f(xyz[k] / self.white[k]));
        [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
    }

    fn delta_e(&self, a: [f64; 3], b: [f64; 3]) -> f64 {
        let (p, q) = (self.lab(a), self.lab(b));
        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
    }
}

/// The develop pipeline's output for one sample (clipped to 0..1 like the 16-bit encode).
fn render(m: &Mat3, tone: Option<&ToneCurve>, cam: [f64; 3]) -> [f64; 3] {
    color::apply(m, cam).map(|v| {
        let v = if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
        tone.map_or(v, |t| t.eval(v))
    })
}

fn mean_delta_e(lab: &Lab, samples: &[&Sample], m: &Mat3, tone: Option<&ToneCurve>) -> f64 {
    if samples.is_empty() {
        return f64::INFINITY;
    }
    let parts = par::map(samples.len().div_ceil(CHUNK), |ci| {
        let lo = ci * CHUNK;
        let chunk = samples.get(lo..(lo + CHUNK).min(samples.len())).unwrap_or(&[]);
        chunk.iter().map(|s| lab.delta_e(render(m, tone, s.cam), s.rgb)).sum::<f64>()
    });
    parts.iter().sum::<f64>() / samples.len() as f64
}

/// Error-weight scale turning a linear difference near luminance `lum` into
/// roughly L* units (dL*/dY = 116/3 · Y^(-2/3)).
fn perceptual(lum: f64) -> f64 {
    (116.0 / 3.0) * (lum.max(0.0) + 0.008).powf(-2.0 / 3.0)
}

fn fit_cells(cells: &[Cell], gw: usize, baseline_exposure: f64) -> Result<CameraLook, LookRejected> {
    let gain = 2f64.powf(if baseline_exposure.is_finite() { baseline_exposure.clamp(-10.0, 10.0) } else { 0.0 });
    let fallback = color::scale(&color::mul(&color::xyz_d50_to_prophoto(), &color::srgb_to_xyz_d50()), gain);
    let lab = Lab::new();

    let mut train = Vec::new();
    let mut test = Vec::new();
    let mut all = Vec::new();
    for (k, c) in cells.iter().enumerate() {
        if !c.cam.iter().chain(&c.rgb).all(|v| v.is_finite()) {
            continue;
        }
        let s = Sample { cam: c.cam, rgb: c.rgb, lum: c.lum, perceptual: perceptual(c.lum), weight: c.edge_weight };
        all.push(s);
        let usable = !c.cam_clipped && c.ref_clipped < 0.01 && c.lum > 0.002 && c.cam.iter().all(|v| *v > 1e-4) && c.edge_weight > 0.05;
        if !usable {
            continue;
        }
        let (x, y) = (k % gw.max(1), k / gw.max(1));
        if (x + 2 * y) % 3 == 0 { test.push(s) } else { train.push(s) }
    }
    if train.len() < 300 || test.len() < 100 {
        return reject(format!("too few usable cells ({} + {} held out)", train.len(), test.len()));
    }

    // The reference must show this picture (its lightness follows the raw's) in colour.
    let usable: Vec<&Sample> = train.iter().chain(&test).collect();
    let pairs: Vec<(f64, f64)> =
        usable.iter().map(|s| (color::apply(&PRO_TO_XYZ_Y_ROW, color::apply(&fallback, s.cam))[0].max(0.0).sqrt(), s.lum.max(0.0).sqrt())).collect();
    let r = correlation(&pairs);
    if r.is_nan() || r < 0.7 {
        return reject(format!("the reference does not match the raw picture (lightness correlation {r:.2})"));
    }
    let coloured = usable.iter().filter(|s| {
        let l = lab.lab(s.rgb);
        l[1].hypot(l[2]) > 4.0
    });
    if coloured.count() * 50 < usable.len() {
        return reject("the reference is (nearly) colourless");
    }

    // Weakest ridge first; a stronger pull toward the fallback when the matrix comes out
    // implausible (little colour in the scene, or a rendering a matrix can't follow).
    let test_refs: Vec<&Sample> = test.iter().collect();
    let held_fallback = mean_delta_e(&lab, &test_refs, &fallback, None);
    let mut why = String::new();
    let mut found = None;
    for ridge in RIDGES {
        let (m, tone) = fit_model(&train, &fallback, gain, ridge)?;
        let rel = color::scale(&m, 1.0 / gain);
        if !plausible(&rel) {
            why = format!("implausible matrix {rel:?}");
            continue;
        }
        let held_fit = mean_delta_e(&lab, &test_refs, &m, Some(&tone));
        if held_fit.is_nan() || held_fit > 0.85 * held_fallback {
            why = format!("the fit does not improve enough on the neutral fallback (held-out ΔE {held_fit:.1} vs {held_fallback:.1})");
            continue;
        }
        found = Some((m, rel, tone));
        break;
    }
    let Some((m, rel, tone)) = found else { return reject(why) };
    let all_refs: Vec<&Sample> = all.iter().collect();
    let delta_e = mean_delta_e(&lab, &all_refs, &m, Some(&tone));
    let delta_e_fallback = mean_delta_e(&lab, &all_refs, &fallback, None);

    let to_xyz = color::mul(&color::rgb_to_xyz(color::ROMM, color::D50_XY), &rel);
    let look = CameraLook { to_xyz, tone: Some(tone), delta_e, delta_e_fallback };
    if !look.is_usable() {
        return reject("the fitted matrix is not usable");
    }
    Ok(look)
}

/// Ridge strengths tried in turn (see [`fit_matrix`]).
const RIDGES: [f64; 3] = [0.01, 0.05, 0.15];

/// Alternates tone-curve and matrix fits with Tukey reweighting.
fn fit_model(train: &[Sample], fallback: &Mat3, gain: f64, ridge: f64) -> Result<(Mat3, ToneCurve), LookRejected> {
    let no_curve = || LookRejected("no tone curve could be fitted".into());
    let mut m = *fallback;
    let mut robust = vec![1.0f64; train.len()];
    let mut tone = fit_curve(train, &robust, &m).ok_or_else(no_curve)?;
    for _ in 0..6 {
        m = fit_matrix(train, &robust, &tone, fallback, gain, ridge);
        tone = fit_curve(train, &robust, &m).ok_or_else(no_curve)?;
        update_robust(train, &mut robust, &m, &tone);
    }
    Ok((m, tone))
}

/// A plausible white-balanced camera → ProPhoto matrix (exposure removed): moderate
/// diagonal and cross-talk, rows summing to about 1 (neutrals stay near neutral), not
/// mirrored or collapsed.
fn plausible(rel: &Mat3) -> bool {
    let entries = (0..3).all(|i| {
        (0..3).all(|j| {
            let v = rel[i][j];
            if i == j { (0.2..=3.0).contains(&v) } else { v.abs() <= 2.0 }
        })
    });
    let [[a, b, c], [d, e, f], [g, h, i]] = *rel;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    let sums = rel.iter().all(|r| (0.6..=1.6).contains(&(r[0] + r[1] + r[2])));
    entries && sums && det > 0.05
}

/// Pearson correlation of `(a, b)` pairs (NaN without spread).
fn correlation(p: &[(f64, f64)]) -> f64 {
    let n = p.len() as f64;
    let (ma, mb) = (p.iter().map(|v| v.0).sum::<f64>() / n, p.iter().map(|v| v.1).sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (a, b) in p {
        sab += (a - ma) * (b - mb);
        saa += (a - ma).powi(2);
        sbb += (b - mb).powi(2);
    }
    sab / (saa * sbb).sqrt()
}

/// Bins of the tone-curve fit, uniform in `sqrt(x)`.
const CURVE_BINS: usize = 32;

/// Fits the tone curve to `(M · cam, rgb)` pairs of all channels, by weighted
/// bin means in the square-root domain made monotone (pool adjacent violators).
fn fit_curve(samples: &[Sample], robust: &[f64], m: &Mat3) -> Option<ToneCurve> {
    let mut bins = [[0.0f64; 3]; CURVE_BINS]; // weight, Σ w·sqrt(x), Σ w·sqrt(y)
    let mut total = 0.0;
    for (s, r) in samples.iter().zip(robust) {
        let x = color::apply(m, s.cam);
        let w = s.weight * r;
        if w <= 0.0 {
            continue;
        }
        for (&xv, &yv) in x.iter().zip(&s.rgb) {
            if !(xv > 1e-5 && xv < 1.0 && yv.is_finite()) {
                continue;
            }
            let (sx, sy) = (xv.sqrt(), yv.clamp(0.0, 1.0).sqrt());
            let b = ((sx * CURVE_BINS as f64) as usize).min(CURVE_BINS - 1);
            if let Some(bin) = bins.get_mut(b) {
                bin[0] += w;
                bin[1] += w * sx;
                bin[2] += w * sy;
                total += w;
            }
        }
    }
    if total.is_nan() || total <= 0.0 {
        return None;
    }
    // (weight, sqrt x, sqrt y) of the populated bins, then PAVA on y.
    let mut pts: Vec<[f64; 3]> = bins.iter().filter(|b| b[0] > total * 2e-4).map(|b| [b[0], b[1] / b[0], b[2] / b[0]]).collect();
    if pts.len() < 4 {
        return None;
    }
    let mut blocks: Vec<(f64, f64, usize)> = Vec::new(); // (weight, mean y, count)
    for p in &pts {
        blocks.push((p[0], p[2], 1));
        while blocks.len() >= 2 {
            let (Some(&b), Some(&a)) = (blocks.last(), blocks.get(blocks.len() - 2)) else { break };
            if a.1 <= b.1 {
                break;
            }
            blocks.truncate(blocks.len() - 2);
            let w = a.0 + b.0;
            blocks.push((w, (a.0 * a.1 + b.0 * b.1) / w, a.2 + b.2));
        }
    }
    let mut i = 0;
    for (_, y, n) in blocks {
        for p in pts.iter_mut().skip(i).take(n) {
            p[2] = y;
        }
        i += n;
    }
    let mut knots = vec![[0.0, 0.0]];
    for p in &pts {
        let (x, y) = (p[1] * p[1], (p[2] * p[2]).clamp(0.0, 1.0));
        let Some(&[px, py]) = knots.last() else { continue };
        if x > px + 1e-6 && x < 1.0 - 1e-6 {
            knots.push([x, y.max(py)]);
        }
    }
    knots.push([1.0, 1.0]);
    ToneCurve::new(&knots)
}

/// Samples per parallel chunk of the least-squares accumulation.
const CHUNK: usize = 4096;

/// Weighted least squares for `M` from `f⁻¹(rgb)` on `cam`, ridge-pulled toward
/// `prior`, then scaled so the balanced white keeps luminance `gain` (a matrix
/// and a curve could otherwise trade exposure).
///
/// The ridge makes colour differences of `ridge` × the level weigh as much as the
/// prior, so scenes with little colour keep the neutral fallback's saturation while
/// the well-determined neutral axis (the camera's rendering of white) follows the data.
fn fit_matrix(samples: &[Sample], robust: &[f64], tone: &ToneCurve, prior: &Mat3, gain: f64, ridge: f64) -> Mat3 {
    // Per row: normal matrix, right-hand side, level.
    type Normal = [([[f64; 3]; 3], [f64; 3], f64); 3];
    let parts: Vec<Normal> = par::map(samples.len().div_ceil(CHUNK), |ci| {
        let mut acc: Normal = [([[0.0; 3]; 3], [0.0; 3], 0.0); 3];
        let lo = ci * CHUNK;
        let chunk = samples.get(lo..(lo + CHUNK).min(samples.len())).unwrap_or(&[]);
        let rob = robust.get(lo..(lo + CHUNK).min(robust.len())).unwrap_or(&[]);
        for (s, r) in chunk.iter().zip(rob) {
            let c = s.cam;
            let level = ((c[0] + c[1] + c[2]) / 3.0).powi(2);
            for (i, (a, b, l)) in acc.iter_mut().enumerate() {
                let y = s.rgb[i];
                if !(y > 0.0 && y < 0.995) {
                    continue;
                }
                let (t, slope) = tone.inverse_and_slope(y);
                let q = slope * s.perceptual;
                let w = s.weight * r * q * q;
                if !(w.is_finite() && w > 0.0) {
                    continue;
                }
                *l += w * level;
                for j in 0..3 {
                    for k in 0..3 {
                        a[j][k] += w * c[j] * c[k];
                    }
                    b[j] += w * c[j] * t;
                }
            }
        }
        acc
    });
    let mut out = *prior;
    for (i, row) in out.iter_mut().enumerate() {
        let (mut a, mut b, mut level) = ([[0.0f64; 3]; 3], [0.0f64; 3], 0.0);
        for p in &parts {
            let (pa, pb, pl) = &p[i];
            for j in 0..3 {
                for k in 0..3 {
                    a[j][k] += pa[j][k];
                }
                b[j] += pb[j];
            }
            level += pl;
        }
        let lambda = ridge * ridge * level + 1e-12;
        for j in 0..3 {
            a[j][j] += lambda;
            b[j] += lambda * prior[i][j];
        }
        if let Some(inv) = color::invert(&a) {
            let x = color::apply(&inv, b);
            if x.iter().all(|v| v.is_finite()) {
                *row = x;
            }
        }
    }
    let white_y = color::apply(&PRO_TO_XYZ_Y_ROW, color::apply(&out, [1.0; 3]))[0];
    if white_y.is_finite() && white_y > 1e-6 { color::scale(&out, gain / white_y) } else { *prior }
}

/// The Y row of linear ProPhoto → XYZ (other rows zero), for luminance.
const PRO_TO_XYZ_Y_ROW: Mat3 = [[0.288_040, 0.711_874, 0.000_086], [0.0; 3], [0.0; 3]];

/// Tukey biweights from each sample's L*-scaled residual.
fn update_robust(samples: &[Sample], robust: &mut [f64], m: &Mat3, tone: &ToneCurve) {
    let err: Vec<f64> = par::map(samples.len().div_ceil(CHUNK), |ci| {
        let lo = ci * CHUNK;
        let chunk = samples.get(lo..(lo + CHUNK).min(samples.len())).unwrap_or(&[]);
        chunk
            .iter()
            .map(|s| {
                let o = render(m, Some(tone), s.cam);
                ((0..3).map(|k| (o[k] - s.rgb[k]).powi(2)).sum::<f64>()).sqrt() * s.perceptual
            })
            .collect::<Vec<_>>()
    })
    .into_iter()
    .flatten()
    .collect();
    let mut sorted: Vec<f64> = err.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return;
    }
    let mid = sorted.len() / 2;
    let (_, median, _) = sorted.select_nth_unstable_by(mid, f64::total_cmp);
    let c = 4.685 * (1.4826 * *median).max(0.5);
    for (r, e) in robust.iter_mut().zip(&err) {
        let u = e / c;
        *r = if u.is_finite() && u < 1.0 { (1.0 - u * u).powi(2) } else { 0.0 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tone_curve_validates_and_interpolates() {
        assert!(ToneCurve::new(&[[0.0, 0.0]]).is_none());
        assert!(ToneCurve::new(&[[0.0, 0.0], [0.5, 0.4], [0.5, 0.6], [1.0, 1.0]]).is_none());
        assert!(ToneCurve::new(&[[0.0, 0.0], [0.5, 0.6], [0.7, 0.4], [1.0, 1.0]]).is_none());
        assert!(ToneCurve::new(&[[0.0, 0.0], [0.5, f64::NAN], [1.0, 1.0]]).is_none());
        assert!(ToneCurve::new(&[[0.1, 0.0], [1.0, 1.0]]).is_none());
        let t = ToneCurve::new(&[[0.0, 0.0], [0.04, 0.16], [1.0, 1.0]]).unwrap();
        assert!((t.eval(0.04) - 0.16).abs() < 1e-12);
        assert!((t.eval(0.01) - 0.04).abs() < 1e-12); // sqrt-domain line through the origin
        assert_eq!(t.eval(-1.0), 0.0);
        assert_eq!(t.eval(2.0), 1.0);
        assert_eq!(t.eval(f64::NAN), 0.0);
        assert!((t.inverse(0.16) - 0.04).abs() < 1e-12);
        let (x, d) = t.inverse_and_slope(0.5);
        assert!((t.eval(x) - 0.5).abs() < 1e-12);
        let num = (t.eval(x + 1e-6) - t.eval(x - 1e-6)) / 2e-6;
        assert!((d - num).abs() < 1e-6 * num.max(1.0), "{d} vs {num}");
        let k = t.knots();
        assert!((k[1][0] - 0.04).abs() < 1e-12 && (k[1][1] - 0.16).abs() < 1e-12);
    }

    /// Camera-space patches with a tonal ramp inside each patch.
    fn patches(w: usize, h: usize) -> Vec<[f32; 3]> {
        let mut seed = 12345u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let (pw, ph) = (12, 8);
        let colours: Vec<[f32; 3]> = (0..pw * ph).map(|_| [0.15 + 0.45 * rnd(), 0.15 + 0.45 * rnd(), 0.15 + 0.45 * rnd()]).collect();
        (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let (px, py) = (x * pw / w, y * ph / h);
                let ramp = 0.15 + 0.85 * ((x * pw) % w) as f32 / w as f32;
                colours[py * pw + px].map(|v| v * ramp)
            })
            .collect()
    }

    const WB: [f64; 3] = [1.5, 1.0, 1.25];

    /// A NEF-like raw of `cam` (balanced by `WB`) and its JPEG-like rendering through
    /// `pro = f(m · balanced)` in linear ProPhoto, encoded as 8-bit sRGB.
    fn shot(w: usize, h: usize, m: &Mat3) -> (Sensor, Vec<u8>) {
        let cam: Vec<[f32; 3]> = patches(w, h).iter().map(|p| [0, 1, 2].map(|k| p[k] / WB[k] as f32)).collect();
        let data = crate::testgen::mosaic(&cam, w, [0, 1, 1, 2], 0, 4095);
        let b = crate::testgen::tiff_ep("NIKON CORPORATION", w, h, &data, [0, 1, 1, 2], 12, vec![]);
        let s = crate::decode(&b, &crate::Limits::default()).unwrap();
        let to_srgb = color::invert(&color::srgb_to_xyz_d50()).unwrap();
        let pro_to_srgb = color::mul(&to_srgb, &color::rgb_to_xyz(color::ROMM, color::D50_XY));
        let f = |x: f64| 1.6 * x / (1.0 + 0.6 * x);
        let enc = |v: f64| {
            let v = v.clamp(0.0, 1.0);
            let e = if v <= 0.0031308 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
            (e * 255.0).round() as u8
        };
        let rgb = cam
            .iter()
            .flat_map(|p| {
                let bal = [0, 1, 2].map(|k| f64::from(p[k]) * WB[k]);
                let pro = color::apply(m, bal).map(|v| f(v.clamp(0.0, 1.0)));
                color::apply(&pro_to_srgb, pro).map(enc)
            })
            .collect();
        (s, rgb)
    }

    fn wb_opts() -> DevelopOptions {
        DevelopOptions { white_balance: crate::WhiteBalance::Multipliers(WB), ..Default::default() }
    }

    /// The true balanced camera → ProPhoto matrix of the synthetic camera: sRGB primaries
    /// with a saturating twist (rows sum to 1).
    fn true_matrix() -> Mat3 {
        let k = [[1.35, -0.25, -0.1], [-0.15, 1.25, -0.1], [0.0, -0.3, 1.3]];
        color::mul(&color::mul(&color::xyz_d50_to_prophoto(), &color::srgb_to_xyz_d50()), &k)
    }

    #[test]
    fn recovers_a_known_matrix_and_curve() {
        let (w, h) = (480, 320);
        let m = true_matrix();
        let (s, rgb) = shot(w, h, &m);
        let look = fit_look(&s, &wb_opts(), &Reference { width: w as u32, height: h as u32, rgb: &rgb }).unwrap();
        let fitted = color::mul(&color::xyz_d50_to_prophoto(), &look.to_xyz);
        for i in 0..3 {
            for j in 0..3 {
                assert!((fitted[i][j] - m[i][j]).abs() < 0.06, "{fitted:?} vs {m:?}");
            }
        }
        let tone = look.tone.as_ref().unwrap();
        for x in [0.02, 0.1, 0.3, 0.6] {
            let want = 1.6 * x / (1.0 + 0.6 * x);
            assert!((tone.eval(x) - want).abs() < 0.03, "f({x}) = {} vs {want}", tone.eval(x));
        }
        assert!(look.delta_e < 1.5 && look.delta_e < 0.2 * look.delta_e_fallback, "{look:?}");

        // Developing with the look replaces the fallback warning by the note.
        let d = crate::develop_sensor(&s, &DevelopOptions { camera_look: Some(look), ..wb_opts() }).unwrap();
        assert!(d.warnings.iter().any(|w| w.contains("fitted to the camera's embedded JPEG")), "{:?}", d.warnings);
        assert!(!d.warnings.iter().any(|w| w.contains("sRGB primaries")), "{:?}", d.warnings);
        let plain = crate::develop_sensor(&s, &wb_opts()).unwrap();
        assert!(plain.warnings.iter().any(|w| w.contains("sRGB primaries")));
        assert_ne!(d.rgb, plain.rgb);
    }

    #[test]
    fn tighter_camera_preview_aligns_for_color_without_changing_the_raw_crop() {
        let (w, h) = (480, 320);
        let (s, rgb) = shot(w, h, &true_matrix());
        let (x0, y0, cw, ch) = (24, 16, 432, 288);
        let mut tight = Vec::with_capacity(cw * ch * 3);
        for y in y0..y0 + ch {
            tight.extend_from_slice(&rgb[(y * w + x0) * 3..(y * w + x0 + cw) * 3]);
        }
        let reference = Reference { width: cw as u32, height: ch as u32, rgb: &tight };
        let (gw, plain) = grid(&s, &wb_opts(), &reference).unwrap();
        let before = fit_cells(&plain, gw, s.baseline_exposure);
        let look = fit_look(&s, &wb_opts(), &reference).unwrap();
        assert!(look.delta_e < 3.0, "{look:?}");
        if let Ok(before) = before {
            assert!(look.delta_e < before.delta_e * 0.5, "aligned {look:?}, before {before:?}");
        }
        let d = crate::develop_sensor(&s, &DevelopOptions { camera_look: Some(look), ..wb_opts() }).unwrap();
        assert_eq!((d.width as usize, d.height as usize), (w, h));
        assert_eq!(s.crop, crate::Rect::new(0, 0, w, h));
    }

    #[test]
    fn follows_the_raw_orientation() {
        let (w, h) = (480, 320);
        let m = true_matrix();
        let (mut s, rgb) = shot(w, h, &m);
        // Stored sideways (orientation 6: rotate 90° clockwise to view); the JPEG is upright.
        s.orientation = 6;
        let mut up = vec![0u8; rgb.len()];
        for y2 in 0..w {
            for x2 in 0..h {
                let (x, y) = source_xy(6, w, h, x2, y2);
                let (o, i) = ((y2 * h + x2) * 3, (y * w + x) * 3);
                up[o..o + 3].copy_from_slice(&rgb[i..i + 3]);
            }
        }
        let look = fit_look(&s, &wb_opts(), &Reference { width: h as u32, height: w as u32, rgb: &up }).unwrap();
        assert!(look.delta_e < 1.5, "{look:?}");
        // The un-rotated JPEG has the wrong shape.
        assert!(fit_look(&s, &wb_opts(), &Reference { width: w as u32, height: h as u32, rgb: &rgb }).is_err());
    }

    #[test]
    fn unusable_references_keep_the_fallback() {
        let (w, h) = (480, 320);
        let (s, rgb) = shot(w, h, &true_matrix());
        let opts = wb_opts();
        let fit = |r: Reference| fit_look(&s, &opts, &r);
        // Noise unrelated to the picture.
        let mut seed = 7u32;
        let noise: Vec<u8> = (0..rgb.len())
            .map(|_| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                (seed >> 16) as u8
            })
            .collect();
        assert!(fit(Reference { width: w as u32, height: h as u32, rgb: &noise }).is_err());
        // A flat grey picture.
        let grey = vec![118u8; rgb.len()];
        assert!(fit(Reference { width: w as u32, height: h as u32, rgb: &grey }).is_err());
        // Another shape, a short buffer, a tiny or empty image, absurd sizes.
        assert!(fit(Reference { width: 320, height: 320, rgb: &rgb[..320 * 320 * 3] }).is_err());
        assert!(fit(Reference { width: w as u32, height: h as u32, rgb: &rgb[..rgb.len() - 1] }).is_err());
        assert!(fit(Reference { width: 12, height: 8, rgb: &rgb[..12 * 8 * 3] }).is_err());
        assert!(fit(Reference { width: 0, height: 0, rgb: &[] }).is_err());
        assert!(fit(Reference { width: u32::MAX, height: u32::MAX, rgb: &rgb }).is_err());
        // A sensor whose crop lies outside its data, or a tiny one.
        let mut bad = s.clone();
        bad.crop.width += 10;
        assert!(fit_look(&bad, &opts, &Reference { width: w as u32, height: h as u32, rgb: &rgb }).is_err());
        let mut short = s.clone();
        short.data.truncate(100);
        assert!(fit_look(&short, &opts, &Reference { width: w as u32, height: h as u32, rgb: &rgb }).is_err());
    }

    #[test]
    fn develop_ignores_an_unusable_look() {
        let (s, _) = shot(64, 48, &true_matrix());
        for to_xyz in [[[f64::NAN; 3]; 3], [[0.0; 3]; 3], [[1e9, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]] {
            let look = CameraLook { to_xyz, tone: None, delta_e: 1.0, delta_e_fallback: 2.0 };
            let d = crate::develop_sensor(&s, &DevelopOptions { camera_look: Some(look), ..wb_opts() }).unwrap();
            assert!(d.warnings.iter().any(|w| w.contains("not usable")), "{:?}", d.warnings);
            assert!(d.warnings.iter().any(|w| w.contains("sRGB primaries")), "{:?}", d.warnings);
        }
    }

    #[test]
    fn identity_curve_and_fallback_have_zero_error_on_their_own_rendering() {
        let lab = Lab::new();
        assert!(lab.delta_e([0.2, 0.3, 0.4], [0.2, 0.3, 0.4]) < 1e-12);
        let w = lab.lab([1.0; 3]);
        assert!((w[0] - 100.0).abs() < 1e-9 && w[1].abs() < 1e-9 && w[2].abs() < 1e-9, "{w:?}");
    }
}
