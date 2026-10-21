//! Paint helpers: bucket fill and the gradient tool.

use photocraft_color::blend::{self, BlendMode};
use photocraft_geom::Rect;
use photocraft_raster::{Surface, from_rgba_into};
use serde::{Deserialize, Serialize};

/// Gradient tool shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum GradientShape {
    #[default]
    Linear,
    Radial,
    Angle,
    Reflected,
    Diamond,
}

/// Gradient tool parameter `t` for a drag from `from` to `to`.
pub fn tool_gradient_t(shape: GradientShape, from: (f32, f32), to: (f32, f32), x: f32, y: f32) -> f32 {
    let (dx, dy) = (to.0 - from.0, to.1 - from.1);
    let len = (dx * dx + dy * dy).sqrt().max(1e-6);
    let (ux, uy) = (dx / len, dy / len);
    let (px, py) = (x - from.0, y - from.1);
    let along = px * ux + py * uy;
    let across = -px * uy + py * ux;
    let t = match shape {
        GradientShape::Linear => along / len,
        GradientShape::Radial => (px * px + py * py).sqrt() / len,
        GradientShape::Reflected => along.abs() / len,
        GradientShape::Diamond => (along.abs() + across.abs()) / len,
        GradientShape::Angle => (across.atan2(along) / std::f32::consts::TAU).rem_euclid(1.0),
    };
    t.clamp(0.0, 1.0)
}

/// Samples RGBA stops (sorted by position) at `t`.
pub fn sample_stops(stops: &[(f32, [f32; 4])], t: f32) -> [f32; 4] {
    match stops {
        [] => [0.0; 4],
        [s] => s.1,
        _ => {
            if t <= stops[0].0 {
                return stops[0].1;
            }
            for w in stops.windows(2) {
                if t <= w[1].0 {
                    let k = if w[1].0 > w[0].0 { (t - w[0].0) / (w[1].0 - w[0].0) } else { 0.0 };
                    return std::array::from_fn(|i| w[0].1[i] + (w[1].1[i] - w[0].1[i]) * k);
                }
            }
            stops[stops.len() - 1].1
        }
    }
}

/// Composites `color(i)` over `area` of the surface with per-pixel coverage.
fn composite_area(
    s: &mut Surface,
    area: Rect,
    blend_mode: BlendMode,
    coverage: impl Fn(i32, i32) -> f32 + Sync,
    color: impl Fn(i32, i32) -> [f32; 4] + Sync,
    lock_transparency: bool,
    inks: Option<[f32; 4]>,
) {
    // Composite tile by tile (in parallel on native); tiles with zero coverage are left untouched,
    // so their copy-on-write storage (and GPU residency) survives.
    let fmt = s.format();
    let n = fmt.channels();
    let tiles: Vec<Rect> = area.tiles().map(|tc| tc.rect().intersect(&area)).filter(|r| !r.is_empty()).collect();
    let src: &Surface = s;
    let work = |r: &Rect| -> Option<(Rect, Vec<f32>)> {
        let w = r.width() as usize;
        let mut any = false;
        let mut region: Vec<f32> = Vec::new();
        let mut enc = [0.0f32; 8];
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                let k = coverage(x, y);
                if k <= 0.0 {
                    continue;
                }
                if !any {
                    region = src.read_region(*r);
                    any = true;
                }
                let i = ((y - r.y0) as usize * w + (x - r.x0) as usize) * n;
                let px = &mut region[i..i + n];
                let d = photocraft_raster::to_rgba(&fmt, px);
                let mut o = blend::composite(blend_mode, d, color(x, y), k);
                if lock_transparency {
                    o[3] = d[3];
                }
                from_rgba_into(&fmt, o, &mut enc);
                // A colour picked as CMYK inks replaces fully covered pixels of a CMYK target as it is.
                if let Some(i) = inks
                    && fmt.mode == photocraft_color::ColorMode::Cmyk
                    && blend_mode == BlendMode::Normal
                    && k >= 1.0
                {
                    enc[..4].copy_from_slice(&i);
                }
                px.copy_from_slice(&enc[..n]);
            }
        }
        any.then_some((*r, region))
    };
    #[cfg(not(target_arch = "wasm32"))]
    let done: Vec<(Rect, Vec<f32>)> = {
        use rayon::prelude::*;
        tiles.par_iter().filter_map(work).collect()
    };
    #[cfg(target_arch = "wasm32")]
    let done: Vec<(Rect, Vec<f32>)> = tiles.iter().filter_map(work).collect();
    for (r, region) in done {
        s.write_region(r, &region);
    }
}

/// Gradient tool over `area` (the selection bounds or canvas).
#[allow(clippy::too_many_arguments)]
pub fn paint_gradient(
    s: &mut Surface,
    area: Rect,
    from: (f32, f32),
    to: (f32, f32),
    shape: GradientShape,
    stops: &[(f32, [f32; 4])],
    reverse: bool,
    opacity: f32,
    blend_mode: BlendMode,
    dither: bool,
    selection: Option<&Surface>,
    lock_transparency: bool,
) {
    composite_area(
        s,
        area,
        blend_mode,
        |x, y| opacity * selection.map_or(1.0, |m| m.sample_channel(x, y, 0)),
        |x, y| {
            let t = tool_gradient_t(shape, from, to, x as f32 + 0.5, y as f32 + 0.5);
            let mut c = sample_stops(stops, if reverse { 1.0 - t } else { t });
            if dither {
                // One quantisation step of monochromatic noise breaks 8-bit banding without speckle.
                let n = (photocraft_color::dither_noise(x, y) - 0.5) / 255.0;
                for ch in c.iter_mut().take(3) {
                    *ch = (*ch + n).clamp(0.0, 1.0);
                }
            }
            c
        },
        lock_transparency,
        None,
    );
}

/// Paint bucket: fills the region similar to the seed pixel (on the layer's
/// own pixels) within `area`, respecting the selection.
#[allow(clippy::too_many_arguments)]
pub fn bucket_fill(
    s: &mut Surface,
    area: Rect,
    seed: (i32, i32),
    tolerance: f32,
    contiguous: bool,
    anti_alias: bool,
    color: [f32; 4],
    opacity: f32,
    selection: Option<&Surface>,
    lock_transparency: bool,
) -> bool {
    bucket_fill_src(s, area, seed, tolerance, contiguous, anti_alias, opacity, selection, lock_transparency, |_, _| color)
}

/// Like [`bucket_fill`] but the fill colour comes from `src(x, y)` — e.g. a pattern sampled at the
/// canvas position — so the Paint Bucket can fill with a pattern as well as a solid colour.
#[allow(clippy::too_many_arguments)]
pub fn bucket_fill_src(
    s: &mut Surface,
    area: Rect,
    seed: (i32, i32),
    tolerance: f32,
    contiguous: bool,
    anti_alias: bool,
    opacity: f32,
    selection: Option<&Surface>,
    lock_transparency: bool,
    src: impl Fn(i32, i32) -> [f32; 4] + Sync,
) -> bool {
    bucket_fill_inks(s, area, seed, tolerance, contiguous, anti_alias, opacity, selection, lock_transparency, src, None)
}

/// [`bucket_fill_src`] where `inks` (CMYK, 0..1) are what a fully covered pixel of a CMYK target
/// gets instead of `src`'s sRGB stand-in (a colour that was picked as CMYK).
#[allow(clippy::too_many_arguments)]
pub fn bucket_fill_inks(
    s: &mut Surface,
    area: Rect,
    seed: (i32, i32),
    tolerance: f32,
    contiguous: bool,
    anti_alias: bool,
    opacity: f32,
    selection: Option<&Surface>,
    lock_transparency: bool,
    src: impl Fn(i32, i32) -> [f32; 4] + Sync,
    inks: Option<[f32; 4]>,
) -> bool {
    if !area.contains(seed.0, seed.1) {
        return false;
    }
    let img = crate::selection::rgba8_image(s, area);
    let Some(region) = crate::selection::wand_region(&img, area, seed, tolerance, contiguous, anti_alias) else { return false };
    drop(img);
    bucket_fill_region_inks(s, &region, opacity, selection, lock_transparency, src, inks);
    true
}

/// Paints a bucket region sampled before the edit, so its boundary can come from the visible
/// composite while only `s` receives the colour or pattern. Selection coverage still clips it.
pub fn bucket_fill_region_src(
    s: &mut Surface,
    region: &crate::selection::Region,
    opacity: f32,
    selection: Option<&Surface>,
    lock_transparency: bool,
    src: impl Fn(i32, i32) -> [f32; 4] + Sync,
) {
    bucket_fill_region_inks(s, region, opacity, selection, lock_transparency, src, None);
}

/// [`bucket_fill_region_src`] with the CMYK `inks` of [`bucket_fill_inks`].
pub fn bucket_fill_region_inks(
    s: &mut Surface,
    region: &crate::selection::Region,
    opacity: f32,
    selection: Option<&Surface>,
    lock_transparency: bool,
    src: impl Fn(i32, i32) -> [f32; 4] + Sync,
    inks: Option<[f32; 4]>,
) {
    // Composite only over the filled region's box.
    let b = region.bbox;
    composite_area(
        s,
        b,
        BlendMode::Normal,
        |x, y| region.at(x, y) * opacity * selection.map_or(1.0, |m| m.sample_channel(x, y, 0)),
        src,
        lock_transparency,
        inks,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::PixelFormat;

    #[test]
    fn gradient_t_shapes() {
        let (f, t) = ((0.0, 0.0), (10.0, 0.0));
        assert_eq!(tool_gradient_t(GradientShape::Linear, f, t, 5.0, 3.0), 0.5);
        assert_eq!(tool_gradient_t(GradientShape::Linear, f, t, -5.0, 0.0), 0.0);
        assert_eq!(tool_gradient_t(GradientShape::Radial, f, t, 0.0, 5.0), 0.5);
        assert_eq!(tool_gradient_t(GradientShape::Reflected, f, t, -5.0, 0.0), 0.5);
        assert_eq!(tool_gradient_t(GradientShape::Diamond, f, t, 2.5, 2.5), 0.5);
        assert!((tool_gradient_t(GradientShape::Angle, f, t, 0.0, 5.0) - 0.25).abs() < 1e-5);
    }

    #[test]
    fn gradient_paints_ramp_within_selection() {
        let mut s = Surface::new(PixelFormat::RGBA8);
        let a = Rect::new(0, 0, 11, 2);
        let mut sel = Surface::new(PixelFormat::GRAY8);
        sel.fill_rect(Rect::new(0, 0, 11, 1), &[1.0]);
        paint_gradient(
            &mut s,
            a,
            (0.5, 0.0),
            (10.5, 0.0),
            GradientShape::Linear,
            &[(0.0, [0.0, 0.0, 0.0, 1.0]), (1.0, [1.0, 1.0, 1.0, 1.0])],
            false,
            1.0,
            BlendMode::Normal,
            false,
            Some(&sel),
            false,
        );
        assert!(s.pixel(0, 0)[0] < 0.01 && s.pixel(10, 0)[0] > 0.99);
        assert!((s.pixel(5, 0)[0] - 0.5).abs() < 0.01);
        assert_eq!(s.pixel(5, 1)[3], 0.0, "outside the selection");
    }

    #[test]
    fn bucket_fills_contiguous_region() {
        let mut s = Surface::new(PixelFormat::RGBA8);
        let a = Rect::new(0, 0, 10, 10);
        s.fill_rect(a, &[1.0, 1.0, 1.0, 1.0]);
        s.fill_rect(Rect::new(5, 0, 6, 10), &[0.0, 0.0, 0.0, 1.0]);
        assert!(bucket_fill(&mut s, a, (1, 1), 10.0, true, false, [1.0, 0.0, 0.0, 1.0], 1.0, None, false));
        assert_eq!(s.pixel(2, 2), vec![1.0, 0.0, 0.0, 1.0]);
        assert_eq!(s.pixel(8, 2), vec![1.0, 1.0, 1.0, 1.0]);
        assert!(!bucket_fill(&mut s, a, (50, 1), 10.0, true, false, [1.0, 0.0, 0.0, 1.0], 1.0, None, false));
    }

    #[test]
    fn bucket_region_uses_document_coordinates_and_multiplies_coverage() {
        let mut s = Surface::new(PixelFormat::RGBA8);
        let region = crate::selection::Region { bbox: Rect::new(-3, 4, 0, 5), mask: vec![255, 128, 255] };
        let mut selection = Surface::new(PixelFormat::GRAY8);
        selection.fill_rect(Rect::new(-3, 4, -1, 5), &[0.5]);
        bucket_fill_region_src(&mut s, &region, 0.5, Some(&selection), false, |_, _| [1.0, 0.0, 0.0, 1.0]);
        assert!((s.rgba(-3, 4)[3] - 0.25).abs() < 0.005);
        assert!((s.rgba(-2, 4)[3] - 0.125).abs() < 0.005);
        assert_eq!(s.rgba(-1, 4), [0.0; 4], "outside the selection");
        assert_eq!(s.rgba(-4, 4), [0.0; 4], "outside the region");
    }
}
