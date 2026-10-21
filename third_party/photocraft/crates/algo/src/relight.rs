//! Photographic relight: recover shading with a blur (Retinex) and replace it with a
//! Lambertian lamp whose direction comes from angle / elevation.

use photocraft_geom::Rect;

use crate::Ctx;
use crate::fxutil::{MAXC, rgba, set_rgba, xy};
use crate::image::Image;

/// Softness maps onto this σ range; the halo is derived from the σ actually used.
pub(crate) const SIGMA_MIN: f32 = 2.0;
pub(crate) const SIGMA_MAX: f32 = 96.0;
/// Finite-difference step is `σ`, but never more than this many pixels.
const STEP_MAX: f32 = 32.0;
/// Luma field allocated for one kernel call; larger `out` rects are split into overlapping bands.
const MAX_LUMA_PIXELS: usize = 8_000_000;

/// Dialog units (degrees / 0–100), matching the command params.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Params {
    pub angle: f32,
    pub elevation: f32,
    pub intensity: f32,
    pub ambient: f32,
    pub warmth: f32,
    pub softness: f32,
}

/// Blur σ from softness and the reference bounds, clamped to `[SIGMA_MIN, SIGMA_MAX]`.
pub(crate) fn sigma(softness: f32, bounds: Rect) -> f32 {
    let side = bounds.width().min(bounds.height()) as f32;
    let s = (softness.clamp(1.0, 100.0) / 100.0) * 0.08 * side.max(1.0);
    s.clamp(SIGMA_MIN, SIGMA_MAX)
}

/// Finite-difference step (pixels) at this σ.
pub(crate) fn step_for_sigma(sig: f32) -> i32 {
    sig.round().clamp(1.0, STEP_MAX) as i32
}

/// Pixels read outside an output tile: `3σ + 1` of the shading blur plus the finite-difference step.
pub(crate) fn read_radius(sig: f32) -> i32 {
    let blur = (sig.max(0.0) * 3.0).ceil() as i32 + 1;
    blur.saturating_add(step_for_sigma(sig))
}

/// Halo for a concrete document size (falls back to [`halo_radius_max`] when `bounds` is empty).
pub(crate) fn halo_radius(softness: f32, bounds: Rect) -> i32 {
    if bounds.is_empty() {
        return halo_radius_max();
    }
    read_radius(sigma(softness, bounds))
}

/// Conservative halo when the document size is unknown: cover [`SIGMA_MAX`].
pub(crate) fn halo_radius_max() -> i32 {
    read_radius(SIGMA_MAX)
}

/// Infinite-light direction: 0° from the right, 90° from above (image space, y down).
fn light_dir(angle: f32, elevation: f32) -> [f32; 3] {
    let (s, co) = angle.to_radians().sin_cos();
    let e = elevation.clamp(0.0, 90.0).to_radians();
    norm3([co * e.cos(), -s * e.cos(), e.sin()])
}

fn norm3(v: [f32; 3]) -> [f32; 3] {
    let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if !m.is_finite() || m < 1e-9 { [0.0, 0.0, 1.0] } else { [v[0] / m, v[1] / m, v[2] / m] }
}

fn light_color(warmth: f32) -> [f32; 3] {
    let w = warmth.clamp(-100.0, 100.0) / 100.0;
    if w >= 0.0 {
        [1.0 + 0.15 * w, 1.0 - 0.05 * w, 1.0 - 0.25 * w]
    } else {
        let t = -w;
        [1.0 - 0.20 * t, 1.0 - 0.10 * t, 1.0 + 0.15 * t]
    }
}

fn rec709_y(c: [f32; 4]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

fn finite_all(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

fn sample_y(buf: &[f32], w: usize, h: usize, rect: Rect, x: i32, y: i32) -> f32 {
    if rect.is_empty() || w == 0 || h == 0 {
        return 0.04;
    }
    let xx = x.clamp(rect.x0, rect.x1.saturating_sub(1));
    let yy = y.clamp(rect.y0, rect.y1.saturating_sub(1));
    let ox = (xx - rect.x0) as usize;
    let oy = (yy - rect.y0) as usize;
    if ox >= w || oy >= h {
        return 0.04;
    }
    buf.get(oy.saturating_mul(w).saturating_add(ox)).copied().filter(|v| v.is_finite()).unwrap_or(0.04)
}

/// Approximate Gaussian of a single-channel field (three box passes; cost independent of σ).
/// Non-finite samples are held out via a parallel weight channel so they do not poison neighbours.
fn blur_channel(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let n = w.saturating_mul(h);
    if n == 0 || src.len() < n {
        return vec![0.04; n];
    }
    let Some(pack_len) = n.checked_mul(2) else {
        return vec![0.04; n];
    };
    let mut packed = vec![0.0f32; pack_len];
    for (i, &v) in src.iter().take(n).enumerate() {
        if !v.is_finite() {
            continue;
        }
        let o = i.saturating_mul(2);
        if let Some(slot) = packed.get_mut(o) {
            *slot = v;
        }
        if let Some(slot) = packed.get_mut(o.saturating_add(1)) {
            *slot = 1.0;
        }
    }
    crate::fxutil::gauss_blur_n(&mut packed, w, h, 2, sigma);
    let mut out = vec![0.5f32; n];
    for i in 0..n {
        let o = i.saturating_mul(2);
        let Some(&acc) = packed.get(o) else { continue };
        let Some(&wgt) = packed.get(o.saturating_add(1)) else { continue };
        if let Some(slot) = out.get_mut(i) {
            *slot = if wgt > 1e-6 && acc.is_finite() && wgt.is_finite() { acc / wgt } else { 0.5 };
        }
    }
    out
}

fn read_rgba(src: &Image, ctx: &Ctx, x: i32, y: i32) -> [f32; 4] {
    let n = src.ch.min(MAXC);
    let mut tmp = [0.0f32; MAXC];
    for c in 0..n {
        if let Some(slot) = tmp.get_mut(c) {
            *slot = src.get(x, y, c);
        }
    }
    rgba(ctx, &tmp[..n])
}

/// Relight `out` of `src`. Identity when `intensity` is 0 (or not finite).
pub(crate) fn relight(src: &Image, out: Rect, ctx: &Ctx, p: Params) -> Vec<f32> {
    let n = src.ch;
    if out.is_empty() || n == 0 {
        return src.crop(out);
    }
    if !p.intensity.is_finite() || p.intensity == 0.0 {
        return src.crop(out);
    }
    let sig = sigma(p.softness, ctx.bounds);
    let step = step_for_sigma(sig);
    let reach = read_radius(sig);
    if let Some(band_h) = luma_band_rows(out, reach) {
        return relight_bands(src, out, ctx, p, sig, step, band_h);
    }
    relight_rect(src, out, ctx, p, sig, step)
}

/// Horizontal strip height so the inflated luma field stays within [`MAX_LUMA_PIXELS`].
fn luma_band_rows(out: Rect, reach: i32) -> Option<i32> {
    let margin = 2usize.saturating_mul(reach.max(0) as usize);
    let yw = (out.width() as usize).saturating_add(margin);
    let yh = (out.height() as usize).saturating_add(margin);
    if yw == 0 {
        return None;
    }
    let n = match yw.checked_mul(yh) {
        Some(n) if n > 0 => n,
        Some(_) => return None,
        None => usize::MAX,
    };
    if n <= MAX_LUMA_PIXELS {
        return None;
    }
    let max_yh = (MAX_LUMA_PIXELS / yw).max(1);
    let inner = max_yh.saturating_sub(margin).max(1).min(i32::MAX as usize) as i32;
    if inner >= out.height() as i32 { None } else { Some(inner.max(1)) }
}

fn relight_bands(src: &Image, out: Rect, ctx: &Ctx, p: Params, sig: f32, step: i32, band_h: i32) -> Vec<f32> {
    let n = src.ch;
    let mut res = src.crop(out);
    let mut y = out.y0;
    let mut guard = 0u32;
    while y < out.y1 && guard < 1_000_000 {
        guard = guard.saturating_add(1);
        let y1 = y.saturating_add(band_h).min(out.y1);
        let band = Rect::new(out.x0, y, out.x1, y1);
        if band.is_empty() {
            break;
        }
        let part = relight_rect(src, band, ctx, p, sig, step);
        copy_band(&mut res, out, &part, band, n);
        y = y1;
    }
    res
}

fn copy_band(dst: &mut [f32], dst_rect: Rect, src: &[f32], src_rect: Rect, n: usize) {
    let dw = dst_rect.width() as usize;
    let sw = src_rect.width() as usize;
    if n == 0 || dw == 0 || sw == 0 {
        return;
    }
    for y in src_rect.y0..src_rect.y1 {
        let sy = (y - src_rect.y0) as usize;
        let dy = (y - dst_rect.y0) as usize;
        let so = sy.saturating_mul(sw).saturating_mul(n);
        let d_off = dy.saturating_mul(dw).saturating_mul(n);
        let len = sw.saturating_mul(n);
        let Some(d) = dst.get_mut(d_off..d_off.saturating_add(len)) else { continue };
        let Some(s) = src.get(so..so.saturating_add(len)) else { continue };
        if d.len() == s.len() {
            d.copy_from_slice(s);
        }
    }
}

fn relight_rect(src: &Image, out: Rect, ctx: &Ctx, p: Params, sig: f32, step: i32) -> Vec<f32> {
    let n = src.ch;
    let mut res = src.crop(out);
    // Finite differences at the shading scale (`σ`), not 1 px: a 1 px step vanishes on
    // large documents and the lamp collapses to a flat multiply.
    const SCALE: f32 = 2.0;
    let y_rect = out.inflate(read_radius(sig));
    if y_rect.is_empty() {
        return res;
    }
    let yw = y_rect.width() as usize;
    let yh = y_rect.height() as usize;
    let Some(n_px) = yw.checked_mul(yh).filter(|&count| count > 0) else {
        return res;
    };
    let mut ybuf = vec![0.5f32; n_px];
    for y in y_rect.y0..y_rect.y1 {
        for x in y_rect.x0..y_rect.x1 {
            let ox = (x - y_rect.x0) as usize;
            let oy = (y - y_rect.y0) as usize;
            let Some(slot) = ybuf.get_mut(oy.saturating_mul(yw).saturating_add(ox)) else {
                continue;
            };
            let c = read_rgba(src, ctx, x, y);
            if !finite_all(&c) {
                *slot = f32::NAN;
                continue;
            }
            let a = if ctx.alpha { c[3] } else { 1.0 };
            *slot = if a > 0.0 && a.is_finite() { rec709_y(c) } else { f32::NAN };
        }
    }
    let sbuf = blur_channel(&ybuf, yw, yh, sig);
    let ldir = light_dir(p.angle, p.elevation);
    let lcol = light_color(p.warmth);
    let amb = (if p.ambient.is_finite() { p.ambient } else { 0.0 }).clamp(0.0, 100.0) / 100.0;
    let inten = p.intensity.clamp(0.0, 100.0) / 100.0;
    for (i, px) in res.chunks_exact_mut(n).enumerate() {
        let (x, y) = xy(out, i);
        if !finite_all(px) {
            continue;
        }
        let a = if ctx.alpha { px[n - 1] } else { 1.0 };
        if ctx.alpha && a <= 0.0 {
            continue;
        }
        let c = rgba(ctx, px);
        if !finite_all(&c) {
            continue;
        }
        let s0 = sample_y(&sbuf, yw, yh, y_rect, x, y).max(0.04);
        let nx = (sample_y(&sbuf, yw, yh, y_rect, x - step, y) - sample_y(&sbuf, yw, yh, y_rect, x + step, y)) * SCALE;
        let ny = (sample_y(&sbuf, yw, yh, y_rect, x, y - step) - sample_y(&sbuf, yw, yh, y_rect, x, y + step)) * SCALE;
        let nrm = norm3([nx, ny, 1.0]);
        let ndl = (nrm[0] * ldir[0] + nrm[1] * ldir[1] + nrm[2] * ldir[2]).max(0.0);
        let shade = amb + inten * ndl;
        let mut out_c = c;
        for (k, lc) in lcol.iter().enumerate() {
            let Some(ch) = c.get(k).copied() else { continue };
            let Some(slot) = out_c.get_mut(k) else { continue };
            *slot = (ch / s0) * shade * *lc;
        }
        if !finite_all(&out_c) {
            continue;
        }
        set_rgba(ctx, px, out_c);
        if ctx.alpha
            && let Some(slot) = px.get_mut(n - 1)
        {
            *slot = a;
        }
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FilterParams, Halo, apply, apply_tiled, output_area};
    use photocraft_color::{ColorMode, PixelFormat, SampleType};
    use photocraft_raster::Surface;

    fn fmt(s: SampleType) -> PixelFormat {
        PixelFormat::new(ColorMode::Rgb, s, true)
    }

    fn gray_fmt(s: SampleType) -> PixelFormat {
        PixelFormat::new(ColorMode::Grayscale, s, true)
    }

    fn params(intensity: f32, angle: f32, warmth: f32, softness: f32) -> FilterParams {
        FilterParams::Relight { angle, elevation: 40.0, intensity, ambient: 55.0, warmth, softness }
    }

    fn relight_p(angle: f32, elevation: f32, intensity: f32, ambient: f32, warmth: f32, softness: f32) -> FilterParams {
        FilterParams::Relight { angle, elevation, intensity, ambient, warmth, softness }
    }

    fn run(s: &Surface, p: &FilterParams, bounds: Rect) -> Surface {
        let area = output_area(p, s.content_bounds(), bounds, None);
        apply(s, p, area, bounds, None)
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    fn luma(px: &[f32]) -> f32 {
        0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2]
    }

    fn shaded_sphere(st: SampleType, size: i32) -> Surface {
        let r = Rect::new(0, 0, size, size);
        let mut surf = Surface::new(fmt(st));
        let cx = size as f32 / 2.0;
        let rad = size as f32 * 0.42;
        let mut v = Vec::new();
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 + 0.5 - cx;
                let dy = y as f32 + 0.5 - cx;
                let d2 = dx * dx + dy * dy;
                if d2 <= rad * rad {
                    let nz = (rad * rad - d2).max(0.0).sqrt() / rad;
                    let sh = 0.22 + 0.78 * nz;
                    v.extend_from_slice(&[0.55 * sh, 0.55 * sh, 0.55 * sh, 1.0]);
                } else {
                    v.extend_from_slice(&[0.18, 0.18, 0.18, 1.0]);
                }
            }
        }
        surf.write_region(r, &v);
        surf
    }

    fn sphere_half_mean(s: &Surface, size: i32, left: bool) -> f32 {
        let cx = size as f32 / 2.0;
        let rad = size as f32 * 0.42;
        let mut sum = 0.0;
        let mut n: f32 = 0.0;
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 + 0.5 - cx;
                let dy = y as f32 + 0.5 - cx;
                if dx * dx + dy * dy > rad * rad {
                    continue;
                }
                if left && (x as f32) >= cx - 4.0 {
                    continue;
                }
                if !left && (x as f32) < cx + 4.0 {
                    continue;
                }
                sum += luma(&s.pixel(x, y));
                n += 1.0;
            }
        }
        sum / n.max(1.0)
    }

    fn pattern(st: SampleType, r: Rect) -> Surface {
        let mut surf = Surface::new(fmt(st));
        let mut v = Vec::new();
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                v.extend_from_slice(&[((x * 7 + y * 3) % 64) as f32 / 63.0, ((x * x + y) % 50) as f32 / 49.0, (y % 9) as f32 / 8.0, 1.0]);
            }
        }
        surf.write_region(r, &v);
        surf
    }

    fn flat(st: SampleType, r: Rect, px: [f32; 4]) -> Surface {
        let mut surf = Surface::new(fmt(st));
        surf.fill_rect(r, &px);
        surf
    }

    #[test]
    fn halo_is_zero_at_identity_and_covers_sigma_otherwise() {
        assert_eq!(params(0.0, 45.0, 0.0, 25.0).halo(), Halo::Radius(0));
        assert_eq!(params(40.0, 45.0, 0.0, 25.0).halo(), Halo::Radius(halo_radius_max()));
        assert_eq!(halo_radius_max(), read_radius(SIGMA_MAX));
        // Short side 1600 px, default softness 25: σ = 32, read 3σ+1+step = 129, which
        // the old constant halo of 97 did not cover.
        let wide = Rect::new(0, 0, 1600, 1600);
        let sig = sigma(25.0, wide);
        assert!((sig - 32.0).abs() < 1e-4, "σ {sig}");
        let reach = read_radius(sig);
        assert_eq!(reach, 96 + 1 + 32);
        assert!(reach > 97);
        assert_eq!(halo_radius(25.0, wide), reach);
        assert_eq!(params(40.0, 45.0, 0.0, 25.0).halo_for(wide), Halo::Radius(reach));
    }

    #[test]
    fn identity_at_zero_intensity_all_depths() {
        let r = Rect::new(0, 0, 24, 16);
        let p = params(0.0, 120.0, 80.0, 90.0);
        for st in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let s = pattern(st, r);
            let out = run(&s, &p, r);
            assert_eq!(out.read_region(r), s.read_region(r), "{st:?}");
        }
        let mut g = Surface::new(gray_fmt(SampleType::U8));
        g.fill_rect(r, &[0.4, 1.0]);
        let out = run(&g, &p, r);
        assert_eq!(out.read_region(r), g.read_region(r));
        let mut g32 = Surface::new(gray_fmt(SampleType::F32));
        g32.fill_rect(r, &[0.4, 1.0]);
        let out = run(&g32, &p, r);
        assert_eq!(out.read_region(r), g32.read_region(r));
        let mut g16 = Surface::new(gray_fmt(SampleType::U16));
        g16.fill_rect(r, &[0.4, 1.0]);
        let out = run(&g16, &p, r);
        assert_eq!(out.read_region(r), g16.read_region(r));
    }

    fn assert_direction_flips(size: i32) {
        let bounds = Rect::new(0, 0, size, size);
        let s = shaded_sphere(SampleType::F32, size);
        let left_lit = run(&s, &relight_p(180.0, 30.0, 90.0, 12.0, 0.0, 15.0), bounds);
        let right_lit = run(&s, &relight_p(0.0, 30.0, 90.0, 12.0, 0.0, 15.0), bounds);
        let l_left = sphere_half_mean(&left_lit, size, true);
        let l_right = sphere_half_mean(&left_lit, size, false);
        let r_left = sphere_half_mean(&right_lit, size, true);
        let r_right = sphere_half_mean(&right_lit, size, false);
        assert!(l_left - l_right >= 0.05, "light from the left ({size}px): {l_left} vs {l_right}");
        assert!(r_right - r_left >= 0.05, "light from the right ({size}px): {r_left} vs {r_right}");
    }

    #[test]
    fn direction_flips_on_a_shaded_sphere() {
        assert_direction_flips(64);
    }

    #[test]
    fn direction_flips_on_a_large_shaded_sphere() {
        // A 1 px gradient vanishes at this size unless the kernel steps by σ.
        assert_direction_flips(256);
    }

    #[test]
    fn warmth_tints_a_neutral_field() {
        let r = Rect::new(0, 0, 32, 32);
        let s = flat(SampleType::F32, r, [0.5, 0.5, 0.5, 1.0]);
        let warm = run(&s, &relight_p(45.0, 40.0, 50.0, 55.0, 80.0, 25.0), r);
        let cool = run(&s, &relight_p(45.0, 40.0, 50.0, 55.0, -80.0, 25.0), r);
        let w = warm.pixel(16, 16);
        let c = cool.pixel(16, 16);
        assert!(w[0] > w[2], "warmth should raise R over B: {w:?}");
        assert!(c[2] > c[0], "cool light should raise B over R: {c:?}");
    }

    #[test]
    fn high_ambient_keeps_white_finite() {
        let r = Rect::new(0, 0, 16, 16);
        let s = flat(SampleType::F32, r, [1.0, 1.0, 1.0, 1.0]);
        let out = run(&s, &relight_p(45.0, 40.0, 40.0, 100.0, 0.0, 25.0), r);
        let px = out.pixel(8, 8);
        assert!(finite_all(&px), "{px:?}");
    }

    #[test]
    fn alpha_is_preserved() {
        let r = Rect::new(0, 0, 16, 16);
        let mut s = Surface::new(fmt(SampleType::F32));
        s.fill_rect(r, &[0.4, 0.4, 0.5, 1.0]);
        s.write_pixel(3, 3, &[0.4, 0.4, 0.5, 0.0]);
        s.write_pixel(4, 4, &[0.2, 0.3, 0.4, 0.5]);
        let out = run(&s, &params(50.0, 180.0, 20.0, 25.0), r);
        assert_eq!(out.pixel(3, 3)[3], 0.0);
        assert!((out.pixel(5, 5)[3] - 1.0).abs() < 1e-6);
        assert!((out.pixel(4, 4)[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn tiles_match_a_full_apply() {
        let size = 64;
        let bounds = Rect::new(0, 0, size, size);
        let s = shaded_sphere(SampleType::F32, size);
        let p = relight_p(180.0, 40.0, 60.0, 40.0, 0.0, 25.0);
        let area = output_area(&p, s.content_bounds(), bounds, None);
        let full = apply_tiled(&s, &p, area, bounds, None, 256, None);
        let tiled = apply_tiled(&s, &p, area, bounds, None, 32, None);
        let inner = Rect::new(0, 0, size, size);
        let d = max_diff(&full.read_region(inner), &tiled.read_region(inner));
        assert!(d <= 1e-4, "tile seam {d}");
    }

    #[test]
    fn tiles_match_a_full_apply_past_the_old_halo() {
        // Default softness on a 1600 px short side: σ = 32, reach 129 px. The previous
        // 97 px halo left a gap, so 128 px tiles seamed. Tile 128 vs one full-frame tile.
        let size = 1600;
        let bounds = Rect::new(0, 0, size, size);
        let s = shaded_sphere(SampleType::F32, size);
        let p = relight_p(180.0, 40.0, 60.0, 40.0, 0.0, 25.0);
        assert!(read_radius(sigma(25.0, bounds)) > 97);
        let area = output_area(&p, s.content_bounds(), bounds, None);
        let full = apply_tiled(&s, &p, area, bounds, None, 4096, None);
        let tiled = apply_tiled(&s, &p, area, bounds, None, 128, None);
        let inner = Rect::new(0, 0, size, size);
        let d = max_diff(&full.read_region(inner), &tiled.read_region(inner));
        assert!(d <= 1e-4, "tile seam {d}");
    }

    #[test]
    fn layers_above_8mp_are_relit() {
        // 4096×2048 = 8.39 MP. A single full-frame kernel call inflates past 8 MP of luma,
        // which used to return the input unchanged. Force one tile so that path is taken.
        let (w, h) = (4096, 2048);
        let bounds = Rect::new(0, 0, w, h);
        let mut s = Surface::new(PixelFormat::RGBA8);
        let bytes: Vec<u8> = (0..h)
            .flat_map(|y| {
                let v = ((y * 200) / h.max(1)) as u8;
                (0..w).flat_map(move |x| {
                    let u = ((x * 200) / w.max(1)) as u8;
                    [80 + u / 4, 40 + v / 3, 90, 255]
                })
            })
            .collect();
        s.write_interleaved(bounds, &bytes);
        let p = relight_p(180.0, 30.0, 90.0, 12.0, 0.0, 25.0);
        let area = output_area(&p, s.content_bounds(), bounds, None);
        let before = s.pixel(w / 2, h / 2);
        let out = apply_tiled(&s, &p, area, bounds, None, 8192, None);
        let after = out.pixel(w / 2, h / 2);
        assert!(after.iter().all(|c| c.is_finite()), "{after:?}");
        let d = max_diff(&before, &after);
        assert!(d > 1e-4, "8 MP+ layer was left unchanged (max diff {d})");
        assert!(max_diff(&s.pixel(8, 8), &out.pixel(8, 8)) > 1e-4);
    }

    #[test]
    fn hostile_sizes_do_not_panic() {
        let p = params(40.0, 45.0, 0.0, 25.0);
        for (w, h) in [(1, 1), (2, 2), (1, 2), (0, 0)] {
            let r = Rect::new(0, 0, w, h);
            let mut s = Surface::new(fmt(SampleType::F32));
            if w > 0 && h > 0 {
                s.fill_rect(r, &[0.4, 0.5, 0.6, 1.0]);
            }
            let _ = run(&s, &p, if r.is_empty() { Rect::new(0, 0, 1, 1) } else { r });
            let _ = apply(&s, &p, Rect::EMPTY, Rect::new(0, 0, 4, 4), None);
        }
    }

    #[test]
    fn nan_and_inf_pixels_pass_through() {
        let r = Rect::new(0, 0, 24, 24);
        let mut s = pattern(SampleType::F32, r);
        s.write_pixel(5, 5, &[f32::NAN, 0.4, 0.4, 1.0]);
        s.write_pixel(6, 8, &[f32::INFINITY, 0.2, 0.2, 1.0]);
        let before_n = s.pixel(7, 5);
        let out = run(&s, &params(50.0, 180.0, 0.0, 20.0), r);
        assert!(out.pixel(5, 5)[0].is_nan());
        assert!(out.pixel(6, 8)[0].is_infinite());
        let n = out.pixel(7, 5);
        assert!(finite_all(&n));
        assert_ne!(n, before_n);
    }
}
