//! Color Replacement: recolour pixels similar to a sampled colour using a non-separable blend
//! (Hue, Saturation, Color or Luminosity) with the foreground colour.

use photocraft_color::BlendMode;
use photocraft_color::blend::blend_rgb;
use photocraft_geom::Rect;
use photocraft_raster::{Surface, from_rgba_into, to_rgba};
use serde::{Deserialize, Serialize};

use crate::Stroke;
use crate::retouch::{Footprint, Region, apply_dab_stroke};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ReplaceMode {
    Hue,
    Saturation,
    #[default]
    Color,
    Luminosity,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Sampling {
    /// Resample under the brush centre at every dab.
    #[default]
    Continuous,
    /// Sample once, where the stroke starts.
    Once,
    /// Replace only colours near the background swatch.
    BackgroundSwatch,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Limits {
    /// Matching pixels anywhere under the brush.
    Discontiguous,
    /// Matching pixels connected to the brush centre.
    #[default]
    Contiguous,
    /// Contiguous, and stop at edges (no soft tolerance falloff).
    FindEdges,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ReplaceSettings {
    pub mode: ReplaceMode,
    pub sampling: Sampling,
    pub limits: Limits,
    /// 0..1 (max per-channel RGB distance).
    pub tolerance: f32,
    pub anti_alias: bool,
    /// Replacement (foreground) colour, straight RGBA.
    pub color: [f32; 4],
    /// Background swatch for [`Sampling::BackgroundSwatch`].
    pub background: [f32; 4],
}

impl Default for ReplaceSettings {
    fn default() -> Self {
        Self {
            mode: ReplaceMode::Color,
            sampling: Sampling::Continuous,
            limits: Limits::Contiguous,
            tolerance: 0.3,
            anti_alias: true,
            color: [1.0, 0.0, 0.0, 1.0],
            background: [1.0; 4],
        }
    }
}

fn dist(a: &[f32; 4], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).abs().max((a[1] - b[1]).abs()).max((a[2] - b[2]).abs())
}

/// Replace colours along a stroke. Returns the damaged rectangle.
pub fn apply_color_replacement(target: &mut Surface, stroke: &Stroke, rs: &ReplaceSettings, selection: Option<&Surface>, lock_transparency: bool) -> Rect {
    let effect = color_replacement_effect(target.clone(), rs.clone());
    apply_dab_stroke(target, stroke, selection, lock_transparency, 0, effect)
}

/// One Color Replacement dab's edit of a working copy, matching against `pre` (the target before
/// the stroke). Shared by [`apply_color_replacement`] and live previews that run the dabs as they
/// come, so both recolour the same pixels.
pub fn color_replacement_effect(pre: Surface, rs: ReplaceSettings) -> impl FnMut(&mut Region, &Footprint) {
    let fmt = pre.format();
    let n = fmt.channels();
    let mode = match rs.mode {
        ReplaceMode::Hue => BlendMode::Hue,
        ReplaceMode::Saturation => BlendMode::Saturation,
        ReplaceMode::Color => BlendMode::Color,
        ReplaceMode::Luminosity => BlendMode::Luminosity,
    };
    let fg = [rs.color[0], rs.color[1], rs.color[2]];
    let tol = rs.tolerance.clamp(0.0, 1.0);
    let soft = rs.anti_alias && rs.limits != Limits::FindEdges;
    let mut once: Option<[f32; 3]> = None;
    let match_w = move |d: f32| -> f32 {
        if d > tol {
            0.0
        } else if soft && tol > 0.0 {
            // Soft falloff over the last quarter of the tolerance.
            ((tol - d) / (tol * 0.25)).clamp(0.0, 1.0)
        } else {
            1.0
        }
    };
    move |work: &mut Region, fp: &Footprint| {
        let (cx, cy) = (fp.dab.center.x.floor() as i32, fp.dab.center.y.floor() as i32);
        let sampled = match rs.sampling {
            Sampling::Continuous => {
                let c = pre.rgba(cx, cy);
                [c[0], c[1], c[2]]
            }
            Sampling::Once => *once.get_or_insert_with(|| {
                let c = pre.rgba(cx, cy);
                [c[0], c[1], c[2]]
            }),
            Sampling::BackgroundSwatch => [rs.background[0], rs.background[1], rs.background[2]],
        };
        let r = fp.rect.intersect(&work.rect);
        if r.is_empty() {
            return;
        }
        let (w, h) = (r.width() as usize, r.height() as usize);
        // Per-pixel match weight.
        let mut weight = vec![0.0f32; w * h];
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                if fp.at(x, y) <= 0.0 {
                    continue;
                }
                // Match against the pre-stroke colours, so pixels recoloured by earlier dabs of
                // this stroke still count as the sampled colour.
                let c = pre.rgba(x, y);
                if c[3] <= 0.0 {
                    continue;
                }
                weight[(y - r.y0) as usize * w + (x - r.x0) as usize] = match_w(dist(&c, sampled));
            }
        }
        if rs.limits != Limits::Discontiguous {
            // Keep only the region connected to the centre (4-connected flood fill).
            let mut keep = vec![false; w * h];
            let start = (cx.clamp(r.x0, r.x1 - 1), cy.clamp(r.y0, r.y1 - 1));
            let si = (start.1 - r.y0) as usize * w + (start.0 - r.x0) as usize;
            if weight[si] > 0.0 {
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
        let mut enc = [0.0f32; 8];
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                let k = fp.at(x, y) * weight[(y - r.y0) as usize * w + (x - r.x0) as usize];
                if k <= 0.0 {
                    continue;
                }
                let px = work.px_mut(x, y);
                let c = to_rgba(&fmt, px);
                let b = blend_rgb(mode, [c[0], c[1], c[2]], fg);
                let o = [c[0] + (b[0] - c[0]) * k, c[1] + (b[1] - c[1]) * k, c[2] + (b[2] - c[2]) * k, c[3]];
                from_rgba_into(&fmt, o, &mut enc);
                px.copy_from_slice(&enc[..n]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BrushSettings, StrokePoint};
    use photocraft_color::PixelFormat;
    use photocraft_color::blend::lum;

    fn setup(fmt: PixelFormat) -> Surface {
        let mut s = Surface::new(fmt);
        // Left: mid green; right: dark blue.
        s.fill_rect(Rect::new(0, 0, 40, 40), &photocraft_raster::from_rgba(&fmt, [0.2, 0.6, 0.2, 1.0]));
        s.fill_rect(Rect::new(40, 0, 80, 40), &photocraft_raster::from_rgba(&fmt, [0.1, 0.1, 0.5, 1.0]));
        s
    }

    fn stroke() -> Stroke {
        Stroke {
            brush: BrushSettings { size: 30.0, hardness: 1.0, spacing: 0.25, pressure_size: false, ..Default::default() },
            points: vec![StrokePoint::new(20.0, 20.0, 1.0), StrokePoint::new(60.0, 20.0, 1.0)],
        }
    }

    #[test]
    fn modes_follow_the_nonseparable_definitions() {
        let fg = [1.0, 0.0, 0.0, 1.0];
        for mode in [ReplaceMode::Hue, ReplaceMode::Saturation, ReplaceMode::Color, ReplaceMode::Luminosity] {
            let mut s = setup(PixelFormat::RGBA32F);
            let rs = ReplaceSettings { mode, sampling: Sampling::Once, limits: Limits::Discontiguous, tolerance: 0.1, color: fg, ..Default::default() };
            apply_color_replacement(&mut s, &stroke(), &rs, None, false);
            let c = s.rgba(20, 20);
            let green = [0.2, 0.6, 0.2];
            match mode {
                // Hue/Color keep the luminosity of the original.
                ReplaceMode::Hue | ReplaceMode::Color => {
                    assert!((lum([c[0], c[1], c[2]]) - lum(green)).abs() < 1e-3, "{mode:?} {c:?}");
                    assert!(c[0] > c[1], "{mode:?} turned red-ish {c:?}");
                }
                ReplaceMode::Saturation => assert!((lum([c[0], c[1], c[2]]) - lum(green)).abs() < 1e-3 && c[1] > c[0], "{c:?}"),
                ReplaceMode::Luminosity => assert!((lum([c[0], c[1], c[2]]) - lum([1.0, 0.0, 0.0])).abs() < 1e-3, "{c:?}"),
            }
            // Sampling once on green: the blue half is outside the tolerance and untouched.
            let b = s.rgba(60, 20);
            assert!((b[2] - 0.5).abs() < 1e-3 && (b[0] - 0.1).abs() < 1e-3, "{mode:?} {b:?}");
        }
    }

    #[test]
    fn continuous_sampling_follows_the_cursor_and_swatch_limits() {
        let mut s = setup(PixelFormat::RGBA8);
        let rs = ReplaceSettings {
            mode: ReplaceMode::Color,
            sampling: Sampling::Continuous,
            limits: Limits::Contiguous,
            tolerance: 0.1,
            color: [1.0, 0.0, 0.0, 1.0],
            ..Default::default()
        };
        apply_color_replacement(&mut s, &stroke(), &rs, None, false);
        assert!(s.rgba(20, 20)[0] > 0.3 && s.rgba(70, 20)[0] > s.rgba(70, 20)[1], "both halves recoloured {:?} {:?}", s.rgba(20, 20), s.rgba(70, 20));
        // Background swatch = white: nothing matches.
        let mut s2 = setup(PixelFormat::RGBA8);
        let rs2 = ReplaceSettings { sampling: Sampling::BackgroundSwatch, background: [1.0; 4], ..rs };
        apply_color_replacement(&mut s2, &stroke(), &rs2, None, false);
        assert_eq!(s2.rgba(20, 20), setup(PixelFormat::RGBA8).rgba(20, 20));
    }

    #[test]
    fn contiguous_stops_at_disconnected_matches() {
        let fmt = PixelFormat::RGBA32F;
        let mut s = Surface::new(fmt);
        s.fill_rect(Rect::new(0, 0, 60, 40), &[0.2, 0.6, 0.2, 1.0]);
        // A blue wall column separates a green island at x 34..40.
        s.fill_rect(Rect::new(30, 0, 34, 40), &[0.0, 0.0, 1.0, 1.0]);
        let st = Stroke { points: vec![StrokePoint::new(20.0, 20.0, 1.0)], ..stroke() };
        for (limits, island_changed) in [(Limits::Contiguous, false), (Limits::Discontiguous, true)] {
            let mut t = s.clone();
            let rs = ReplaceSettings {
                mode: ReplaceMode::Color,
                sampling: Sampling::Once,
                limits,
                tolerance: 0.05,
                color: [1.0, 0.0, 0.0, 1.0],
                ..Default::default()
            };
            apply_color_replacement(&mut t, &st, &rs, None, false);
            let changed = (t.rgba(34, 20)[0] - 0.2).abs() > 0.05;
            assert_eq!(changed, island_changed, "{limits:?}");
            assert!(t.rgba(20, 20)[0] > 0.3);
        }
    }
}
