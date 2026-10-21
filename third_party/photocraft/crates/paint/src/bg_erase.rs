//! Background Eraser: along a stroke, sample the colour under the brush hotspot and erase pixels
//! of similar colour inside the brush to transparency.
//!
//! Colours are compared in the surface's own colour channels (RGB, CMYK, Gray, Lab… at any depth),
//! so nothing is converted through RGB; only alpha changes.

use photocraft_geom::Rect;
use photocraft_raster::{Surface, from_rgba};
use serde::{Deserialize, Serialize};

use crate::Stroke;
use crate::replace::{Limits, Sampling};
use crate::retouch::{Region, alpha_index, apply_dab_stroke};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct BgEraseSettings {
    pub sampling: Sampling,
    pub limits: Limits,
    /// 0..1: largest per-channel difference from the sampled colour that is erased.
    pub tolerance: f32,
    /// Keep pixels that match the foreground colour.
    pub protect_foreground: bool,
    /// Straight RGBA foreground colour (for Protect Foreground Color).
    pub foreground: [f32; 4],
    /// Straight RGBA background swatch (for [`Sampling::BackgroundSwatch`]).
    pub background: [f32; 4],
}

impl Default for BgEraseSettings {
    fn default() -> Self {
        Self {
            sampling: Sampling::Continuous,
            limits: Limits::Contiguous,
            tolerance: 0.5,
            protect_foreground: false,
            foreground: [0.0, 0.0, 0.0, 1.0],
            background: [1.0; 4],
        }
    }
}

/// Largest per-channel difference over the colour channels.
fn dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// Erase along `stroke`. The brush opacity is ignored (Photoshop's Background Eraser has none);
/// the selection limits the effect. Returns the damaged rectangle; empty when the surface has no
/// alpha channel.
pub fn apply_background_eraser(target: &mut Surface, stroke: &Stroke, bs: &BgEraseSettings, selection: Option<&Surface>) -> Rect {
    let fmt = target.format();
    let Some(a) = alpha_index(&fmt) else { return Rect::EMPTY };
    let nc = a;
    let tol = if bs.tolerance.is_finite() { bs.tolerance.clamp(0.0, 1.0) } else { 0.0 };
    let soft = bs.limits != Limits::FindEdges;
    let native = |c: [f32; 4]| -> Vec<f32> { from_rgba(&fmt, [c[0], c[1], c[2], 1.0])[..nc].to_vec() };
    let swatch = native(bs.background);
    let fg = native(bs.foreground);
    let mut once: Option<Option<Vec<f32>>> = None;
    let match_w = |d: f32| -> f32 {
        if d > tol {
            0.0
        } else if soft && tol > 0.0 {
            // Soft falloff over the last quarter of the tolerance.
            ((tol - d) / (tol * 0.25)).clamp(0.0, 1.0)
        } else {
            1.0
        }
    };
    let mut stroke = stroke.clone();
    stroke.brush.opacity = 1.0;
    stroke.brush.erase = false;
    // The pixels before the stroke, taken at the first dab. Sampling and matching read these, so a
    // dab landing on pixels earlier dabs already erased still samples and matches (#1280).
    let mut orig: Option<Region> = None;
    apply_dab_stroke(target, &stroke, selection, false, 0, |work: &mut Region, fp| {
        let r = fp.rect.intersect(&work.rect);
        if r.is_empty() {
            return;
        }
        let orig = &*orig.get_or_insert_with(|| work.clone());
        let (cx, cy) = (fp.dab.center.x.floor() as i32, fp.dab.center.y.floor() as i32);
        // The hotspot's colour (None where it was transparent before the stroke or is off the
        // working area).
        let hotspot = || -> Option<Vec<f32>> {
            if !orig.rect.contains(cx, cy) {
                return None;
            }
            let p = orig.px(cx, cy);
            (p[a] > 0.0).then(|| p[..nc].to_vec())
        };
        let sampled = match bs.sampling {
            Sampling::Continuous => hotspot(),
            Sampling::Once => once.get_or_insert_with(hotspot).clone(),
            Sampling::BackgroundSwatch => Some(swatch.clone()),
        };
        let Some(sampled) = sampled else { return };
        let (w, h) = (r.width() as usize, r.height() as usize);
        // Per-pixel match weight from the pre-stroke colours: the contiguous fill passes through
        // pixels this stroke already erased.
        let mut weight = vec![0.0f32; w * h];
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                if fp.at(x, y) <= 0.0 {
                    continue;
                }
                let p = orig.px(x, y);
                if p[a] <= 0.0 {
                    continue;
                }
                let c = &p[..nc];
                if bs.protect_foreground && dist(c, &fg) <= tol {
                    continue;
                }
                weight[(y - r.y0) as usize * w + (x - r.x0) as usize] = match_w(dist(c, &sampled));
            }
        }
        if bs.limits != Limits::Discontiguous {
            keep_connected(&mut weight, w, h, (cx.clamp(r.x0, r.x1 - 1) - r.x0) as usize, (cy.clamp(r.y0, r.y1 - 1) - r.y0) as usize);
        }
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                let k = fp.at(x, y) * weight[(y - r.y0) as usize * w + (x - r.x0) as usize];
                if k <= 0.0 {
                    continue;
                }
                let p = work.px_mut(x, y);
                p[a] *= 1.0 - k.min(1.0);
                if p[a] <= 0.0 {
                    p.fill(0.0);
                }
            }
        }
    })
}

/// Zeroes the weights not 4-connected to `(sx, sy)` through non-zero weights.
fn keep_connected(weight: &mut [f32], w: usize, h: usize, sx: usize, sy: usize) {
    let mut keep = vec![false; w * h];
    let si = sy * w + sx;
    if weight.get(si).is_some_and(|v| *v > 0.0) {
        let mut stack = vec![si];
        keep[si] = true;
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            let mut nb = |j: usize| {
                if !keep[j] && weight[j] > 0.0 {
                    keep[j] = true;
                    stack.push(j);
                }
            };
            if x > 0 {
                nb(i - 1);
            }
            if x + 1 < w {
                nb(i + 1);
            }
            if y > 0 {
                nb(i - w);
            }
            if y + 1 < h {
                nb(i + w);
            }
        }
    }
    for (wv, k) in weight.iter_mut().zip(&keep) {
        if !k {
            *wv = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BrushSettings, StrokePoint};
    use photocraft_color::{ColorMode, PixelFormat, SampleType};

    const GREEN: [f32; 4] = [0.2, 0.6, 0.2, 1.0];
    const BLUE: [f32; 4] = [0.1, 0.1, 0.5, 1.0];

    fn setup(fmt: PixelFormat) -> Surface {
        let mut s = Surface::new(fmt);
        s.fill_rect(Rect::new(0, 0, 40, 40), &from_rgba(&fmt, GREEN));
        s.fill_rect(Rect::new(40, 0, 80, 40), &from_rgba(&fmt, BLUE));
        s
    }

    fn stroke(points: &[(f64, f64)]) -> Stroke {
        Stroke {
            brush: BrushSettings { size: 30.0, hardness: 1.0, spacing: 0.25, pressure_size: false, ..Default::default() },
            points: points.iter().map(|&(x, y)| StrokePoint::new(x, y, 1.0)).collect(),
        }
    }

    fn alpha(s: &Surface, x: i32, y: i32) -> f32 {
        s.rgba(x, y)[3]
    }

    #[test]
    fn once_erases_only_the_sampled_colour_in_every_format() {
        for mode in [ColorMode::Rgb, ColorMode::Cmyk, ColorMode::Grayscale] {
            for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
                let fmt = PixelFormat::new(mode, sample, true);
                let mut s = setup(fmt);
                let bs = BgEraseSettings { sampling: Sampling::Once, limits: Limits::Discontiguous, tolerance: 0.05, ..Default::default() };
                let d = apply_background_eraser(&mut s, &stroke(&[(20.0, 20.0), (60.0, 20.0)]), &bs, None);
                assert!(!d.is_empty());
                assert_eq!(alpha(&s, 20, 20), 0.0, "{fmt:?} green under the hotspot erased");
                assert_eq!(alpha(&s, 45, 20), 1.0, "{fmt:?} blue kept");
                assert_eq!(alpha(&s, 20, 39), 1.0, "{fmt:?} outside the brush kept");
            }
        }
    }

    #[test]
    fn continuous_follows_the_hotspot_and_swatch_matches_background() {
        let fmt = PixelFormat::RGBA8;
        let mut s = setup(fmt);
        let bs = BgEraseSettings { sampling: Sampling::Continuous, limits: Limits::Discontiguous, tolerance: 0.05, ..Default::default() };
        apply_background_eraser(&mut s, &stroke(&[(20.0, 20.0), (60.0, 20.0)]), &bs, None);
        assert_eq!(alpha(&s, 20, 20), 0.0);
        assert_eq!(alpha(&s, 60, 20), 0.0, "blue erased once the hotspot reached it");
        // Background swatch = blue: only blue goes, wherever the hotspot is.
        let mut t = setup(fmt);
        let bs = BgEraseSettings { sampling: Sampling::BackgroundSwatch, background: BLUE, ..bs };
        apply_background_eraser(&mut t, &stroke(&[(30.0, 20.0), (50.0, 20.0)]), &bs, None);
        assert_eq!(alpha(&t, 30, 20), 1.0);
        assert_eq!(alpha(&t, 50, 20), 0.0);
    }

    #[test]
    fn contiguous_and_protect_foreground() {
        let fmt = PixelFormat::RGBA32F;
        let mut s = Surface::new(fmt);
        s.fill_rect(Rect::new(0, 0, 60, 40), &GREEN);
        // A blue wall separates a green island at x 30..40.
        s.fill_rect(Rect::new(26, 0, 30, 40), &BLUE);
        for (limits, island_erased) in [(Limits::Contiguous, false), (Limits::FindEdges, false), (Limits::Discontiguous, true)] {
            let mut t = s.clone();
            let bs = BgEraseSettings { sampling: Sampling::Once, limits, tolerance: 0.05, ..Default::default() };
            apply_background_eraser(&mut t, &stroke(&[(20.0, 20.0)]), &bs, None);
            assert_eq!(alpha(&t, 31, 20) == 0.0, island_erased, "{limits:?}");
            assert_eq!(alpha(&t, 20, 20), 0.0);
        }
        // Protecting a foreground equal to the sampled colour erases nothing.
        let mut t = s.clone();
        let bs = BgEraseSettings { sampling: Sampling::Once, tolerance: 0.05, protect_foreground: true, foreground: GREEN, ..Default::default() };
        apply_background_eraser(&mut t, &stroke(&[(20.0, 20.0)]), &bs, None);
        assert_eq!(t, s);
    }

    #[test]
    fn one_pass_clears_the_whole_stroke_when_dabs_overlap_erased_pixels() {
        // #1280: later dabs land their hotspot on pixels earlier dabs already erased.
        let fmt = PixelFormat::RGBA8;
        let mut s = Surface::new(fmt);
        s.fill_rect(Rect::new(0, 0, 60, 30), &from_rgba(&fmt, GREEN));
        let mut st = stroke(&[(10.0, 15.0), (50.0, 15.0)]);
        st.brush.size = 20.0;
        for sampling in [Sampling::Continuous, Sampling::Once] {
            let mut t = s.clone();
            let bs = BgEraseSettings { sampling, limits: Limits::Contiguous, ..Default::default() };
            apply_background_eraser(&mut t, &st, &bs, None);
            // Everything within radius − 1 of the line.
            for y in 6..24 {
                for x in 10..51 {
                    assert_eq!(alpha(&t, x, y), 0.0, "{sampling:?} ({x}, {y}) left behind");
                }
            }
        }
    }

    #[test]
    fn selection_limits_and_degenerate_inputs() {
        let fmt = PixelFormat::RGBA8;
        let mut s = setup(fmt);
        let mut sel = Surface::new(PixelFormat::GRAY8);
        sel.fill_rect(Rect::new(0, 0, 20, 40), &[1.0]);
        let bs = BgEraseSettings { sampling: Sampling::Once, limits: Limits::Discontiguous, tolerance: 0.05, ..Default::default() };
        apply_background_eraser(&mut s, &stroke(&[(20.0, 20.0)]), &bs, Some(&sel));
        assert_eq!(alpha(&s, 15, 20), 0.0);
        assert_eq!(alpha(&s, 25, 20), 1.0, "outside the selection kept");
        // Sampling a transparent hotspot erases nothing; NaN tolerance acts as 0; no alpha → no-op.
        let mut t = Surface::new(fmt);
        assert!(apply_background_eraser(&mut t, &stroke(&[(5.0, 5.0)]), &bs, None).width() > 0);
        assert_eq!(t.content_bounds(), Rect::EMPTY);
        let mut u = setup(fmt);
        let nan = BgEraseSettings { tolerance: f32::NAN, ..bs.clone() };
        apply_background_eraser(&mut u, &stroke(&[(20.0, 20.0)]), &nan, None);
        assert_eq!(alpha(&u, 20, 20), 0.0, "exact match still erased at tolerance 0");
        let mut g = Surface::new(PixelFormat::GRAY8);
        assert!(apply_background_eraser(&mut g, &stroke(&[(5.0, 5.0)]), &bs, None).is_empty());
    }
}
