//! Gradient fill layer geometry.
//!
//! Photoshop lays a gradient fill layer's Linear and Reflected gradients out between whole-pixel
//! end points: the chord through the frame's centre along the angle (as [`effects::gradient_t`])
//! ends at points truncated to the pixel grid. On large frames that is invisible, on small ones it
//! changes the effective angle: a 30° Reflected fill on a 4 × 4 canvas ends at the corner
//! `(4, 0)` instead of `(4, 0.845)` and renders as a 45° one (fitted on the psd-tools `colormodes`
//! files, whose 16/32-bit versions store no fill pixels).
//!
//! The snapped end points are expressed as the angle / scale / centre offset that
//! [`effects::gradient_t`] (and the GPU's `gradient_t`) already take, so both compositors share
//! one layout.

use photocraft_doc::GradientStyle;
use photocraft_geom::Rect;

use crate::effects;

/// Effective `(angle, scale, centre offset)` for a gradient fill with `style`, `angle` (degrees)
/// and `scale` laid out in `frame`.
pub fn fill_gradient_layout(style: GradientStyle, angle: f32, scale: f32, frame: Rect) -> (f32, f32, (f32, f32)) {
    gradient_layout(style, angle, scale, (0.0, 0.0), frame)
}

/// [`fill_gradient_layout`] for a gradient centred `offset` (a fraction of the frame) off the
/// frame's centre: layer-effect gradients (overlays, gradient glows and strokes) snap the same
/// way (psd-tools layer_effects: an 87° overlay on a 600 × 60 text line runs along (3, 60), i.e.
/// 87.14°).
pub fn gradient_layout(style: GradientStyle, angle: f32, scale: f32, offset: (f32, f32), frame: Rect) -> (f32, f32, (f32, f32)) {
    let unchanged = (angle, scale, offset);
    if !matches!(style, GradientStyle::Linear | GradientStyle::Reflected)
        || !angle.is_finite()
        || !scale.is_finite()
        || !offset.0.is_finite()
        || !offset.1.is_finite()
    {
        return unchanged;
    }
    let w = f64::from(frame.width().max(1));
    let h = f64::from(frame.height().max(1));
    let (cx, cy) = (f64::from(frame.x0) + w / 2.0 + f64::from(offset.0) * w, f64::from(frame.y0) + h / 2.0 + f64::from(offset.1) * h);
    // Unscaled chord length along `a` (radians), as `effects::gradient_t`.
    let chord_of = |a: f64| {
        let (s, c) = a.sin_cos();
        (w / c.abs().max(1e-6)).min(h / s.abs().max(1e-6)).max(1.0)
    };
    let a = f64::from(angle).to_radians();
    let scale64 = f64::from(scale.max(1e-3));
    let half = chord_of(a) * scale64 / 2.0;
    let (s, c) = a.sin_cos();
    let (dx, dy) = (c * half, -s * half);
    // Truncate to the pixel grid (with a little slack for rounding error in exact cases).
    let snap = |v: f64| (v + 1e-4).floor();
    let end = (snap(cx + dx), snap(cy + dy));
    let (start, mid) = match style {
        GradientStyle::Reflected => ((cx, cy), (cx, cy)),
        _ => {
            let s0 = (snap(cx - dx), snap(cy - dy));
            (s0, ((s0.0 + end.0) / 2.0, (s0.1 + end.1) / 2.0))
        }
    };
    let v = (end.0 - start.0, end.1 - start.1);
    let len = v.0.hypot(v.1);
    if len < 0.5 {
        return unchanged;
    }
    let a2 = (-v.1).atan2(v.0);
    // Reflected spans half the chord from the centre; Linear the whole chord.
    let span = if style == GradientStyle::Reflected { 2.0 * len } else { len };
    let scale2 = span / chord_of(a2);
    let shift = (((mid.0 - cx) / w) as f32, ((mid.1 - cy) / h) as f32);
    (a2.to_degrees() as f32, scale2 as f32, (offset.0 + shift.0, offset.1 + shift.1))
}

/// Gradient parameter `t` of a gradient fill at pixel centre `(x, y)` (see [`fill_gradient_layout`]).
pub fn fill_gradient_t(style: GradientStyle, angle: f32, scale: f32, reverse: bool, frame: Rect, x: f32, y: f32) -> f32 {
    let (a, s, o) = fill_gradient_layout(style, angle, scale, frame);
    effects::gradient_t(style, a, s, reverse, o, frame, x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(style: GradientStyle, angle: f32, frame: Rect) -> Vec<f32> {
        let mut v = Vec::new();
        for y in frame.y0..frame.y1 {
            for x in frame.x0..frame.x1 {
                v.push(fill_gradient_t(style, angle, 1.0, false, frame, x as f32 + 0.5, y as f32 + 0.5));
            }
        }
        v
    }

    #[test]
    fn small_reflected_30_degrees_snaps_to_the_corner() {
        // Photoshop's 4 × 4 30° Reflected fill: t = |x − y| / 4.
        let t = grid(GradientStyle::Reflected, 30.0, Rect::new(0, 0, 4, 4));
        for y in 0..4 {
            for x in 0..4 {
                let want = (x as f32 - y as f32).abs() / 4.0;
                assert!((t[y * 4 + x] - want).abs() < 1e-4, "({x},{y}) {} vs {want}", t[y * 4 + x]);
            }
        }
    }

    #[test]
    fn exact_layouts_are_unchanged() {
        for (style, angle, frame) in [
            (GradientStyle::Linear, 90.0, Rect::new(0, 0, 33, 20)),
            (GradientStyle::Linear, 0.0, Rect::new(-5, 3, 40, 17)),
            (GradientStyle::Reflected, 45.0, Rect::new(0, 0, 32, 32)),
            (GradientStyle::Linear, -90.0, Rect::new(10, 10, 210, 210)),
        ] {
            for y in [frame.y0, (frame.y0 + frame.y1) / 2, frame.y1 - 1] {
                for x in [frame.x0, (frame.x0 + frame.x1) / 2, frame.x1 - 1] {
                    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                    let a = fill_gradient_t(style, angle, 1.0, false, frame, px, py);
                    let b = effects::gradient_t(style, angle, 1.0, false, (0.0, 0.0), frame, px, py);
                    assert!((a - b).abs() < 1e-3, "{style:?} {angle} {frame:?} ({x},{y}): {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn offset_centres_snap_like_the_frame_centre() {
        // A zero offset is the plain layout exactly.
        let f = Rect::new(0, 0, 7, 5);
        for style in [GradientStyle::Linear, GradientStyle::Reflected] {
            assert_eq!(gradient_layout(style, 30.0, 1.0, (0.0, 0.0), f), fill_gradient_layout(style, 30.0, 1.0, f));
        }
        // Moving the centre by whole pixels moves the snapped layout by the same pixels.
        let big = Rect::new(0, 0, 40, 20);
        let (a0, s0, o0) = gradient_layout(GradientStyle::Linear, 30.0, 0.5, (0.0, 0.0), big);
        let (a1, s1, o1) = gradient_layout(GradientStyle::Linear, 30.0, 0.5, (0.25, -0.1), big);
        assert!((a0 - a1).abs() < 1e-4 && (s0 - s1).abs() < 1e-4, "{a0} {a1} {s0} {s1}");
        assert!((o1.0 - o0.0 - 0.25).abs() < 1e-4 && (o1.1 - o0.1 + 0.1).abs() < 1e-4, "{o0:?} {o1:?}");
        // Non-snapping styles and bad offsets pass through.
        assert_eq!(gradient_layout(GradientStyle::Radial, 30.0, 1.0, (0.2, 0.1), big), (30.0, 1.0, (0.2, 0.1)));
        let (_, _, o) = gradient_layout(GradientStyle::Linear, 30.0, 1.0, (f32::NAN, 0.0), big);
        assert!(o.0.is_nan());
    }

    #[test]
    fn large_frames_barely_move_and_bad_input_is_harmless() {
        let f = Rect::new(0, 0, 3000, 2000);
        let (a, s, o) = fill_gradient_layout(GradientStyle::Linear, 30.0, 1.0, f);
        assert!((a - 30.0).abs() < 0.05 && (s - 1.0).abs() < 1e-3 && o.0.abs() < 1e-3 && o.1.abs() < 1e-3);
        for (ang, sc) in [(f32::NAN, 1.0), (30.0, f32::INFINITY), (30.0, 0.0), (1e30, 1.0)] {
            let t = fill_gradient_t(GradientStyle::Linear, ang, sc, false, Rect::new(0, 0, 4, 4), 1.5, 1.5);
            assert!(t.is_nan() || (0.0..=1.0).contains(&t));
        }
        assert_eq!(fill_gradient_layout(GradientStyle::Radial, 30.0, 1.0, f), (30.0, 1.0, (0.0, 0.0)));
    }
}
