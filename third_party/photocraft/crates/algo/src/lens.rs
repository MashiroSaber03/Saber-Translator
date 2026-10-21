//! Filter › Lens Correction (and File › Automate › Lens Correction): geometric distortion,
//! chromatic aberration, vignetting and perspective / rotation / scale, with Photoshop's edge
//! modes.
//!
//! The model is the classical one:
//!
//! * **Distortion**: Brown–Conrady radial terms (D. C. Brown, *Decentering Distortion of Lenses*,
//!   Photogrammetric Engineering 1966; A. Conrady 1919): an output point at normalised radius
//!   `r` (1 = half the frame diagonal) samples the source at `r·(1 + k1·r² + k2·r⁴)`, plus the
//!   tangential `p1, p2` terms. Photoshop's *Remove Distortion* slider adds to `k1`
//!   (positive values correct barrel distortion).
//! * **Chromatic aberration** (lateral): the red and blue planes are scaled radially about the
//!   centre relative to green (*Fix Red/Cyan Fringe*, *Fix Blue/Yellow Fringe*).
//! * **Vignetting**: the lens profile's `1 + v1·r² + v2·r⁴ + v3·r⁶` falloff is divided out, then
//!   the *Vignette Amount / Midpoint* sliders apply an exposure change ramping in from the
//!   midpoint.
//! * **Transform**: vertical / horizontal perspective as a keystone homography about the frame
//!   centre, then the angle and the scale.
//!
//! The geometry runs as an inverse map per output pixel (bicubic, premultiplied when there is
//! alpha), per output tile in parallel. A few generic profiles stand in for manufacturer data
//! (which is proprietary): [`generic_profile`] interpolates typical distortion and vignetting
//! over the 35 mm-equivalent focal length.

use photocraft_color::{ColorMode, PixelFormat};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde::{Deserialize, Serialize};

use crate::photo_util::{catmull_rom, par_map};

/// What fills the output where the corrected image doesn't reach.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum EdgeMode {
    /// Repeat the edge pixels.
    Extension,
    #[default]
    Transparency,
    /// A solid colour (straight RGBA in the document's model via `to_rgba` conventions).
    Color([f32; 4]),
}

/// A lens profile (distortion, vignetting and lateral CA coefficients).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct LensProfile {
    pub name: String,
    pub k1: f64,
    pub k2: f64,
    pub p1: f64,
    pub p2: f64,
    /// Vignetting falloff `1 + v1 r² + v2 r⁴ + v3 r⁶`.
    pub vignette: [f64; 3],
    /// Radial scale of red / blue relative to green (`1 + ca`).
    pub ca_red: f64,
    pub ca_blue: f64,
}

/// Generic profile for a 35 mm-equivalent focal length (typical zoom-lens behaviour: barrel
/// distortion and strong vignetting at the wide end, mild pincushion at the long end).
pub fn generic_profile(focal_35mm: f64) -> LensProfile {
    // (focal, k1, k2, v1, v2)
    const TABLE: [(f64, f64, f64, f64, f64); 7] = [
        (12.0, -0.16, 0.04, -0.60, 0.15),
        (16.0, -0.11, 0.025, -0.50, 0.12),
        (24.0, -0.06, 0.012, -0.38, 0.08),
        (35.0, -0.025, 0.004, -0.28, 0.05),
        (50.0, 0.0, 0.0, -0.22, 0.04),
        (85.0, 0.012, 0.0, -0.16, 0.02),
        (200.0, 0.02, 0.0, -0.10, 0.01),
    ];
    let f = focal_35mm.clamp(TABLE[0].0, TABLE[6].0);
    let k = TABLE.windows(2).position(|w| f <= w[1].0).unwrap_or(5);
    let (a, b) = (TABLE[k], TABLE[k + 1]);
    // Interpolate on log focal length.
    let t = (f.ln() - a.0.ln()) / (b.0.ln() - a.0.ln());
    let l = |x: f64, y: f64| x + (y - x) * t;
    LensProfile {
        name: format!("Generic Lens ({focal_35mm:.0} mm equivalent)"),
        k1: l(a.1, b.1),
        k2: l(a.2, b.2),
        vignette: [l(a.3, b.3), l(a.4, b.4), 0.0],
        ..Default::default()
    }
}

/// Lens Correction parameters in Photoshop's dialog units.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct LensCorrection {
    /// Profile applied first (Auto Correction tab).
    pub profile: Option<LensProfile>,
    pub correct_distortion: bool,
    pub correct_vignette: bool,
    pub correct_ca: bool,
    /// Remove Distortion −100..=100.
    pub distortion: f64,
    /// Fix Red/Cyan Fringe −100..=100.
    pub red_cyan: f64,
    /// Fix Blue/Yellow Fringe −100..=100.
    pub blue_yellow: f64,
    /// Vignette Amount −100..=100 (positive lightens the corners).
    pub vignette_amount: f64,
    /// Vignette Midpoint 0..=100.
    pub vignette_midpoint: f64,
    /// Vertical Perspective −100..=100.
    pub vertical: f64,
    /// Horizontal Perspective −100..=100.
    pub horizontal: f64,
    /// Angle in degrees (positive turns the image clockwise).
    pub angle: f64,
    /// Scale in percent (50..=150).
    pub scale: f64,
    pub edge: EdgeMode,
}

impl Default for LensCorrection {
    fn default() -> Self {
        LensCorrection {
            profile: None,
            correct_distortion: true,
            correct_vignette: true,
            correct_ca: true,
            distortion: 0.0,
            red_cyan: 0.0,
            blue_yellow: 0.0,
            vignette_amount: 0.0,
            vignette_midpoint: 50.0,
            vertical: 0.0,
            horizontal: 0.0,
            angle: 0.0,
            scale: 100.0,
            edge: EdgeMode::Transparency,
        }
    }
}

/// Where an output pixel samples the source: green (and alpha / non-RGB channels) at `p`,
/// red at `pr`, blue at `pb`, then the colour is multiplied by `gain`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub p: [f64; 2],
    pub pr: [f64; 2],
    pub pb: [f64; 2],
    pub gain: f32,
}

/// The inverse map of a [`LensCorrection`] over a frame.
pub struct LensMap {
    c: [f64; 2],
    /// Half diagonal (radius normaliser).
    r: f64,
    k1: f64,
    k2: f64,
    p1: f64,
    p2: f64,
    ca_r: f64,
    ca_b: f64,
    vig_profile: [f64; 3],
    vig_amount: f64,
    vig_mid: f64,
    /// Inverse keystone (normalised coords).
    hinv: [f64; 9],
    rot: (f64, f64),
    scale: f64,
}

impl LensMap {
    pub fn new(lc: &LensCorrection, frame: Rect) -> LensMap {
        let c = [(frame.x0 + frame.x1) as f64 / 2.0, (frame.y0 + frame.y1) as f64 / 2.0];
        let r = (frame.width() as f64).hypot(frame.height() as f64).max(2.0) / 2.0;
        let prof = lc.profile.clone().unwrap_or_default();
        let (pk1, pk2, pp1, pp2) = if lc.correct_distortion { (prof.k1, prof.k2, prof.p1, prof.p2) } else { (0.0, 0.0, 0.0, 0.0) };
        // Keystone x' = x / (1 + h x + g y), y' = y / (1 + h x + g y) (normalised by r).
        let g = -lc.vertical / 100.0 * 0.5;
        let h = -lc.horizontal / 100.0 * 0.5;
        // Inverse of [[1,0,0],[0,1,0],[h,g,1]] is [[1,0,0],[0,1,0],[-h,-g,1]].
        let hinv = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -h, -g, 1.0];
        let a = lc.angle.to_radians();
        LensMap {
            c,
            r,
            k1: pk1 - lc.distortion / 100.0 * 0.25,
            k2: pk2,
            p1: pp1,
            p2: pp2,
            ca_r: if lc.correct_ca { prof.ca_red } else { 0.0 } - lc.red_cyan * 4e-5,
            ca_b: if lc.correct_ca { prof.ca_blue } else { 0.0 } - lc.blue_yellow * 4e-5,
            vig_profile: if lc.correct_vignette { prof.vignette } else { [0.0; 3] },
            vig_amount: lc.vignette_amount / 100.0,
            vig_mid: lc.vignette_midpoint.clamp(0.0, 100.0) / 100.0,
            hinv,
            rot: (a.sin(), a.cos()),
            scale: (lc.scale / 100.0).clamp(0.05, 20.0),
        }
    }

    /// Output px → where to sample (`None` when the keystone sends it to infinity).
    pub fn sample(&self, x: f64, y: f64) -> Option<Sample> {
        // Undo scale and rotation (about the centre).
        let (dx, dy) = ((x - self.c[0]) / self.scale, (y - self.c[1]) / self.scale);
        let (s, co) = self.rot;
        let (rx, ry) = (co * dx + s * dy, -s * dx + co * dy);
        // Undo the keystone.
        let (nx, ny) = (rx / self.r, ry / self.r);
        let m = &self.hinv;
        let w = m[6] * nx + m[7] * ny + m[8];
        if w <= 0.05 {
            return None;
        }
        let (ux, uy) = (nx / w, ny / w);
        // Lens distortion (Brown–Conrady).
        let r2 = ux * ux + uy * uy;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r2 * r2;
        let sx = ux * radial + 2.0 * self.p1 * ux * uy + self.p2 * (r2 + 2.0 * ux * ux);
        let sy = uy * radial + self.p1 * (r2 + 2.0 * uy * uy) + 2.0 * self.p2 * ux * uy;
        let to_px = |k: f64| [self.c[0] + sx * k * self.r, self.c[1] + sy * k * self.r];
        // Vignetting on the source radius.
        let rs2 = sx * sx + sy * sy;
        let [v1, v2, v3] = self.vig_profile;
        let falloff = (1.0 + v1 * rs2 + v2 * rs2 * rs2 + v3 * rs2 * rs2 * rs2).max(0.05);
        let start = (1.0 - self.vig_mid) * 0.9;
        let t = ((rs2.sqrt() - start) / (1.0 - start + 1e-9)).clamp(0.0, 1.0);
        let ramp = t * t * (3.0 - 2.0 * t);
        let gain = 2f64.powf(self.vig_amount * 1.5 * ramp) / falloff;
        Some(Sample { p: to_px(1.0), pr: to_px(1.0 + self.ca_r), pb: to_px(1.0 + self.ca_b), gain: gain as f32 })
    }
}

/// Smallest scale (percent) at which the corrected image covers the whole frame (Auto Scale).
pub fn auto_scale(lc: &LensCorrection, frame: Rect) -> f64 {
    let covers = |scale: f64| {
        let m = LensMap::new(&LensCorrection { scale, ..lc.clone() }, frame);
        let (x0, y0, x1, y1) = (frame.x0 as f64, frame.y0 as f64, frame.x1 as f64, frame.y1 as f64);
        (0..=40).all(|i| {
            let t = i as f64 / 40.0;
            [(x0 + t * (x1 - x0), y0), (x0 + t * (x1 - x0), y1), (x0, y0 + t * (y1 - y0)), (x1, y0 + t * (y1 - y0))]
                .iter()
                .all(|&(x, y)| m.sample(x, y).is_some_and(|s| s.p[0] >= x0 - 0.5 && s.p[0] <= x1 + 0.5 && s.p[1] >= y0 - 0.5 && s.p[1] <= y1 + 0.5))
        })
    };
    if covers(lc.scale) {
        // Shrink while it still covers (never below the user's 100 % baseline when it covers).
        return lc.scale;
    }
    let (mut lo, mut hi) = (lc.scale, lc.scale);
    while !covers(hi) && hi < 400.0 {
        hi *= 1.25;
    }
    for _ in 0..30 {
        let mid = (lo + hi) / 2.0;
        if covers(mid) { hi = mid } else { lo = mid }
    }
    hi
}

/// Angle (degrees, for [`LensCorrection::angle`]) that makes the line `a → b` horizontal or
/// vertical, whichever it is closer to (the Straighten tool).
pub fn straighten_angle(a: [f64; 2], b: [f64; 2]) -> f64 {
    let deg = (b[1] - a[1]).atan2(b[0] - a[0]).to_degrees();
    // Nearest multiple of 90°.
    let target = (deg / 90.0).round() * 90.0;
    -(deg - target)
}

const TILE: i32 = 128;

/// Resamples `src` through an inverse map over `out_area`. `frame` is the image the map refers
/// to: samples clamp to it, and pixels mapping outside it take the edge mode. `map` returns where
/// each output pixel centre samples (`None` = outside).
pub fn remap(src: &Surface, frame: Rect, out_area: Rect, edge: EdgeMode, map: &(dyn Fn(f64, f64) -> Option<Sample> + Sync)) -> Surface {
    remap_impl(src, frame, out_area, edge, map, true)
}

fn remap_impl(src: &Surface, frame: Rect, out_area: Rect, edge: EdgeMode, map: &(dyn Fn(f64, f64) -> Option<Sample> + Sync), shared: bool) -> Surface {
    let sfmt = src.format();
    let needs_alpha = matches!(edge, EdgeMode::Transparency) && !sfmt.alpha;
    let ofmt = if needs_alpha { PixelFormat::new(sfmt.mode, sfmt.sample, true) } else { sfmt };
    let mut out = Surface::new(ofmt);
    if out_area.is_empty() || frame.is_empty() {
        return out;
    }
    let n_in = sfmt.channels();
    let n_out = ofmt.channels();
    let cc = sfmt.mode.color_channels();
    let rgb = matches!(sfmt.mode, ColorMode::Rgb | ColorMode::Indexed) && cc == 3;
    let subtractive = sfmt.mode == ColorMode::Cmyk;
    let lab = sfmt.mode == ColorMode::Lab;
    let edge_px: Vec<f32> = match edge {
        EdgeMode::Color(c) => photocraft_raster::from_rgba(&ofmt, c),
        _ => vec![0.0; n_out],
    };
    let mut tiles = Vec::new();
    let mut ty = out_area.y0;
    while ty < out_area.y1 {
        let mut tx = out_area.x0;
        while tx < out_area.x1 {
            tiles.push(Rect::new(tx, ty, (tx + TILE).min(out_area.x1), (ty + TILE).min(out_area.y1)));
            tx += TILE;
        }
        ty += TILE;
    }
    let results = par_map(tiles.len(), |k| {
        let t = tiles[k];
        let (tw, th) = (t.width() as usize, t.height() as usize);
        let samples: Vec<Option<Sample>> = (0..tw * th).map(|i| map(t.x0 as f64 + (i % tw) as f64 + 0.5, t.y0 as f64 + (i / tw) as f64 + 0.5)).collect();
        // Source window: bbox of every sample, clamped to the frame.
        let (mut bx0, mut by0, mut bx1, mut by1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for s in samples.iter().flatten() {
            for p in [s.p, s.pr, s.pb] {
                bx0 = bx0.min(p[0]);
                by0 = by0.min(p[1]);
                bx1 = bx1.max(p[0]);
                by1 = by1.max(p[1]);
            }
        }
        let mut data = vec![0.0f32; tw * th * n_out];
        let win = if bx0 > bx1 {
            Rect::EMPTY
        } else {
            let clamp_x = |v: f64| (v as i32).clamp(frame.x0, frame.x1);
            let clamp_y = |v: f64| (v as i32).clamp(frame.y0, frame.y1);
            Rect::new(clamp_x(bx0.floor() - 3.0), clamp_y(by0.floor() - 3.0), clamp_x(bx1.ceil() + 4.0), clamp_y(by1.ceil() + 4.0))
        };
        let (ww, wh) = (win.width() as usize, win.height() as usize);
        let mut buf = if win.is_empty() { Vec::new() } else { src.read_region(win) };
        // Premultiply colour by alpha so transparent pixels don't bleed.
        if sfmt.alpha {
            for px in buf.chunks_exact_mut(n_in) {
                let a = px[n_in - 1];
                for v in &mut px[..n_in - 1] {
                    *v *= a;
                }
            }
        }
        let fx0 = frame.x0 as f64;
        let fy0 = frame.y0 as f64;
        let fx1 = frame.x1 as f64;
        let fy1 = frame.y1 as f64;
        let bicubic = |c: usize, p: [f64; 2]| -> f32 {
            // Pixel centres at +0.5; clamp to the frame (edge extension).
            let x = (p[0] - 0.5).clamp(fx0, fx1 - 1.0);
            let y = (p[1] - 0.5).clamp(fy0, fy1 - 1.0);
            let (xi, yi) = (x.floor(), y.floor());
            let (wx, wy) = (catmull_rom((x - xi) as f32), catmull_rom((y - yi) as f32));
            let mut acc = 0.0f32;
            for (j, wyv) in wy.iter().enumerate() {
                let yy = ((yi as i32 + j as i32 - 1).clamp(win.y0, win.y1 - 1) - win.y0) as usize;
                for (i, wxv) in wx.iter().enumerate() {
                    let xx = ((xi as i32 + i as i32 - 1).clamp(win.x0, win.x1 - 1) - win.x0) as usize;
                    acc += wxv * wyv * buf[(yy * ww + xx) * n_in + c];
                }
            }
            acc
        };
        for (i, s) in samples.iter().enumerate() {
            let o = &mut data[i * n_out..(i + 1) * n_out];
            let Some(s) = s.filter(|_| ww > 0 && wh > 0) else {
                o.copy_from_slice(&edge_px);
                continue;
            };
            // Inside-frame weight (1 px soft edge) unless edges extend.
            let inside = if matches!(edge, EdgeMode::Extension) {
                1.0
            } else {
                let fx = ((s.p[0] - fx0).min(fx1 - s.p[0]) + 0.5).clamp(0.0, 1.0);
                let fy = ((s.p[1] - fy0).min(fy1 - s.p[1]) + 0.5).clamp(0.0, 1.0);
                (fx * fy) as f32
            };
            // Color channels and coverage usually sample the same location. Prepare the 16
            // tap addresses and weights once; each channel keeps its original summation order.
            let center = if shared && (!rgb || (s.pr == s.p && s.pb == s.p)) { sample_channels(&buf, win, frame, n_in, s.p) } else { None };
            let sample = |c: usize, p: [f64; 2]| {
                let cached = if p == s.p { center.as_ref() } else { None };
                cached.and_then(|v| v.get(c)).copied().unwrap_or_else(|| bicubic(c, p))
            };
            let alpha = if sfmt.alpha { sample(n_in - 1, s.p).clamp(0.0, 1.0) } else { 1.0 };
            let mut col = [0.0f32; 8];
            for (c, v) in col.iter_mut().enumerate().take(cc) {
                let p = if rgb && c == 0 {
                    s.pr
                } else if rgb && c == 2 {
                    s.pb
                } else {
                    s.p
                };
                let pm = sample(c, p);
                let straight = if sfmt.alpha { if alpha > 1e-6 { pm / alpha } else { 0.0 } } else { pm };
                let g = s.gain;
                *v = if subtractive {
                    1.0 - (1.0 - straight) * g
                } else if lab {
                    if c == 0 { straight * g } else { straight }
                } else {
                    straight * g
                };
                if sfmt.sample != photocraft_color::SampleType::F32 {
                    *v = v.clamp(0.0, 1.0);
                }
            }
            let a_out = alpha * inside;
            match edge {
                EdgeMode::Color(_) if inside < 1.0 => {
                    // Blend towards the edge colour by the outside share.
                    let ea = if ofmt.alpha { edge_px[n_out - 1] } else { 1.0 };
                    let oa = alpha * inside + ea * (1.0 - inside);
                    for c in 0..cc {
                        let mixed = col[c] * alpha * inside + edge_px[c] * ea * (1.0 - inside);
                        o[c] = if oa > 1e-6 { mixed / oa } else { 0.0 };
                    }
                    if ofmt.alpha {
                        o[n_out - 1] = oa;
                    }
                }
                _ => {
                    o[..cc].copy_from_slice(&col[..cc]);
                    if ofmt.alpha {
                        o[n_out - 1] = a_out;
                    }
                }
            }
        }
        (t, data)
    });
    for (t, data) in results {
        out.write_region(t, &data);
    }
    out.prune();
    out
}

fn sample_channels(buf: &[f32], win: Rect, frame: Rect, n: usize, p: [f64; 2]) -> Option<[f32; 8]> {
    match n {
        1 => sample_channels_impl::<1>(buf, win, frame, p),
        2 => sample_channels_impl::<2>(buf, win, frame, p),
        3 => sample_channels_impl::<3>(buf, win, frame, p),
        4 => sample_channels_impl::<4>(buf, win, frame, p),
        5 => sample_channels_impl::<5>(buf, win, frame, p),
        8 => sample_channels_impl::<8>(buf, win, frame, p),
        _ => None,
    }
}
fn sample_channels_impl<const N: usize>(buf: &[f32], win: Rect, frame: Rect, p: [f64; 2]) -> Option<[f32; 8]> {
    if N == 0 || N > 8 || win.is_empty() || frame.is_empty() || !p.iter().all(|v| v.is_finite()) {
        return None;
    }
    let x = (p[0] - 0.5).clamp(frame.x0 as f64, frame.x1 as f64 - 1.0);
    let y = (p[1] - 0.5).clamp(frame.y0 as f64, frame.y1 as f64 - 1.0);
    let (xi, yi) = (x.floor(), y.floor());
    let (wx, wy) = (catmull_rom((x - xi) as f32), catmull_rom((y - yi) as f32));
    let mut acc = [0.0f32; 8];
    for (j, wyv) in wy.iter().enumerate() {
        let yy = ((yi as i32 + j as i32 - 1).clamp(win.y0, win.y1 - 1) - win.y0) as usize;
        for (i, wxv) in wx.iter().enumerate() {
            let xx = ((xi as i32 + i as i32 - 1).clamp(win.x0, win.x1 - 1) - win.x0) as usize;
            let start = yy.checked_mul(win.width() as usize)?.checked_add(xx)?.checked_mul(N)?;
            let px = buf.get(start..start.checked_add(N)?)?.first_chunk::<N>()?;
            let weight = wxv * wyv;
            for (a, v) in acc.iter_mut().zip(px) {
                *a += weight * *v;
            }
        }
    }
    Some(acc)
}

/// Applies a lens correction to `src` over `frame` (normally the canvas).
pub fn correct(src: &Surface, frame: Rect, lc: &LensCorrection) -> Surface {
    let map = LensMap::new(lc, frame);
    remap(src, frame, frame, lc.edge, &|x, y| map.sample(x, y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::SampleType;

    fn grid_surface(fmt: PixelFormat, w: i32, h: i32) -> Surface {
        let mut s = Surface::new(fmt);
        let n = fmt.channels();
        let mut row = vec![0.0f32; w as usize * n];
        for y in 0..h {
            for x in 0..w {
                let v = if x % 16 < 2 || y % 16 < 2 { 0.9 } else { 0.3 };
                let px = photocraft_raster::from_rgba(&fmt, [v, v * 0.8, v * 0.6, 1.0]);
                row[x as usize * n..(x as usize + 1) * n].copy_from_slice(&px);
            }
            s.write_region(Rect::new(0, y, w, y + 1), &row);
        }
        s
    }

    #[test]
    fn identity_is_lossless_at_all_depths() {
        for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let fmt = PixelFormat::new(ColorMode::Rgb, sample, true);
            let s = grid_surface(fmt, 96, 64);
            let out = correct(&s, Rect::new(0, 0, 96, 64), &LensCorrection::default());
            let (a, b) = (s.read_region(Rect::new(0, 0, 96, 64)), out.read_region(Rect::new(0, 0, 96, 64)));
            let err = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            assert!(err < 2.0 / 255.0, "{sample:?}: {err}");
        }
    }

    #[test]
    fn distortion_round_trip_and_edges() {
        let frame = Rect::new(0, 0, 200, 100);
        // A barrel-distorting profile then its correction map back the centre exactly.
        let lc = LensCorrection { distortion: 50.0, ..Default::default() };
        let m = LensMap::new(&lc, frame);
        let s = m.sample(100.0, 50.0).unwrap();
        assert!((s.p[0] - 100.0).abs() < 1e-9 && (s.p[1] - 50.0).abs() < 1e-9);
        // Positive Remove Distortion samples corners closer to the centre (pincushion correction).
        let c = m.sample(0.0, 0.0).unwrap();
        assert!(c.p[0] > 5.0 && c.p[1] > 2.0, "{c:?}");
        // Transparency leaves the uncovered corners clear; extension and colour fill them.
        let src = grid_surface(PixelFormat::new(ColorMode::Rgb, SampleType::U8, false), 200, 100);
        let neg = LensCorrection { distortion: -60.0, ..Default::default() };
        let t = correct(&src, frame, &neg);
        assert!(t.format().alpha);
        assert!(t.pixel(0, 0)[3] < 0.01 && t.pixel(100, 50)[3] > 0.99);
        let e = correct(&src, frame, &LensCorrection { edge: EdgeMode::Extension, ..neg.clone() });
        assert!(!e.format().alpha);
        let col = correct(&src, frame, &LensCorrection { edge: EdgeMode::Color([1.0, 0.0, 0.0, 1.0]), ..neg });
        let p = col.pixel(0, 0);
        assert!(p[0] > 0.99 && p[1] < 0.01, "{p:?}");
    }

    #[test]
    fn vignette_ca_perspective_and_straighten() {
        let frame = Rect::new(0, 0, 300, 200);
        let m = LensMap::new(&LensCorrection { vignette_amount: 100.0, ..Default::default() }, frame);
        assert!((m.sample(150.0, 100.0).unwrap().gain - 1.0).abs() < 1e-6);
        assert!(m.sample(1.0, 1.0).unwrap().gain > 2.0);
        let d = LensMap::new(&LensCorrection { vignette_amount: -100.0, ..Default::default() }, frame);
        assert!(d.sample(1.0, 1.0).unwrap().gain < 0.5);
        // Lateral CA scales red outward relative to green.
        let ca = LensMap::new(&LensCorrection { red_cyan: -100.0, ..Default::default() }, frame).sample(10.0, 10.0).unwrap();
        assert!(ca.pr[0] < ca.p[0]);
        // Vertical perspective −: the top edge samples a narrower source span (widens the top).
        let v = LensMap::new(&LensCorrection { vertical: -50.0, ..Default::default() }, frame);
        let (tl, tr) = (v.sample(0.0, 0.0).unwrap().p, v.sample(300.0, 0.0).unwrap().p);
        let (bl, br) = (v.sample(0.0, 200.0).unwrap().p, v.sample(300.0, 200.0).unwrap().p);
        assert!(tr[0] - tl[0] < br[0] - bl[0], "top {} bottom {}", tr[0] - tl[0], br[0] - bl[0]);
        // Straighten a line tilted by 5° → −5° correction, and Auto Scale hides the corners.
        let a = straighten_angle([0.0, 0.0], [100.0, 100.0 * 5f64.to_radians().tan()]);
        assert!((a + 5.0).abs() < 1e-9, "{a}");
        assert!((straighten_angle([0.0, 0.0], [3.0, 100.0]) - 1.718).abs() < 0.01);
        let lc = LensCorrection { angle: 5.0, ..Default::default() };
        let s = auto_scale(&lc, frame);
        assert!(s > 105.0 && s < 130.0, "{s}");
        let rot = LensMap::new(&LensCorrection { scale: s, ..lc }, frame);
        assert!((0.0..=300.0).contains(&rot.sample(0.0, 0.0).unwrap().p[0]));
    }

    #[test]
    fn generic_profiles_and_cmyk() {
        let wide = generic_profile(16.0);
        let tele = generic_profile(135.0);
        assert!(wide.k1 < -0.05 && tele.k1 > 0.0 && wide.vignette[0] < tele.vignette[0]);
        let mid = generic_profile(30.0);
        assert!(mid.k1 < -0.025 && mid.k1 > -0.06);
        // CMYK vignette lightening reduces ink.
        let fmt = PixelFormat::new(ColorMode::Cmyk, SampleType::U8, false);
        let mut s = Surface::new(fmt);
        s.fill_rect(Rect::new(0, 0, 40, 40), &[0.5, 0.5, 0.5, 0.5]);
        let out = correct(&s, Rect::new(0, 0, 40, 40), &LensCorrection { vignette_amount: 100.0, edge: EdgeMode::Extension, ..Default::default() });
        assert!(out.pixel(0, 0)[0] < 0.4 && (out.pixel(20, 20)[0] - 0.5).abs() < 0.02, "{:?}", out.pixel(0, 0));
    }
}

#[cfg(test)]
mod shared_sample_tests;
