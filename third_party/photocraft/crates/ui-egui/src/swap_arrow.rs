//! Compact colour-swap glyph above-right of the toolbar swatches. The shaft starts horizontally
//! and bends clockwise into a vertical end. Filled heads point left and down, not diagonally.
//! Geometry stays within the existing 13x13 icon box; this module never changes toolbar layout.

use egui::{Color32, Painter, Pos2, Shape, Stroke, Vec2, pos2, vec2};

/// Shaft endpoints on the 13x13 icon grid.
pub const START: Pos2 = pos2(4.0, 4.0);
pub const END: Pos2 = pos2(9.0, 9.0);
/// The control point keeps the starting tangent horizontal and the ending tangent vertical.
pub const CONTROL: Pos2 = pos2(9.0, 4.0);
/// A slim shaft leaves the two small heads distinct at toolbar size.
pub const WIDTH: f32 = 1.6;
/// Filled head length and half-width, in logical pixels.
pub const HEAD_LEN: f32 = 3.0;
pub const HEAD_HALF: f32 = 2.8;

/// The curve's tangent at the start (false) or the end (true), pointing into the curve.
fn tangent(at_end: bool) -> Vec2 {
    if at_end { vec2(END.x - CONTROL.x, END.y - CONTROL.y) } else { vec2(CONTROL.x - START.x, CONTROL.y - START.y) }
}

/// Points sampled along the curve, `steps` segments.
pub fn curve_points(steps: usize) -> Vec<Pos2> {
    let n = steps.max(1);
    (0..=n)
        .map(|i| {
            let t = i as f32 / n as f32;
            let (a, b) = (1.0 - t, t);
            pos2(a * a * START.x + 2.0 * a * b * CONTROL.x + b * b * END.x, a * a * START.y + 2.0 * a * b * CONTROL.y + b * b * END.y)
        })
        .collect()
}

/// The filled head at `pt` pointing outward along the unit vector `out`.
fn head_triangle(pt: Pos2, out: Vec2) -> [Pos2; 3] {
    let tip = pt + out * HEAD_LEN;
    let side = vec2(-out.y, out.x);
    [tip, pt + side * HEAD_HALF, pt - side * HEAD_HALF]
}

/// Paint the glyph with its top-left at `o`, in ink colour `ink`.
pub fn paint(p: &Painter, o: Pos2, ink: Color32) {
    let stroke = Stroke::new(WIDTH, ink);
    let curve = curve_points(10).iter().map(|q| o + q.to_vec2()).collect::<Vec<_>>();
    p.add(Shape::line(curve, stroke));
    for (pt, out) in [(START, -tangent(false).normalized()), (END, tangent(true).normalized())] {
        let head = head_triangle(pt, out).iter().map(|q| o + q.to_vec2()).collect::<Vec<_>>();
        p.add(Shape::convex_polygon(head, ink, Stroke::NONE));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every part of the glyph stays inside the 13x13 icon box: the curve and both heads.
    #[test]
    fn the_glyph_stays_inside_the_13px_icon_box() {
        let heads = [head_triangle(START, -tangent(false).normalized()), head_triangle(END, tangent(true).normalized())];
        for q in curve_points(200).into_iter().chain(heads.iter().flatten().copied()) {
            assert!((0.0..=13.0).contains(&q.x), "x out of the icon box at {q:?}");
            assert!((0.0..=13.0).contains(&q.y), "y out of the icon box at {q:?}");
        }
    }

    #[test]
    fn rendered_glyph_has_a_left_head_and_a_lower_right_head() {
        let mut h = egui_kittest::Harness::builder().with_size(vec2(32.0, 32.0)).wgpu().build_ui(|ui| {
            paint(ui.painter(), pos2(8.0, 8.0), Color32::WHITE);
        });
        h.run_steps(2);
        let image = h.render().expect("render swap glyph");
        let has_ink = |x0, y0, x1, y1| (y0..y1).any(|y| (x0..x1).any(|x| image.get_pixel(x, y).0[0] > 180));
        assert!(has_ink(9, 10, 12, 14), "missing left-pointing head");
        assert!(has_ink(15, 17, 20, 21), "missing downward-pointing head");
        assert!(!has_ink(9, 17, 13, 21), "wrong diagonal runs into the lower-left corner");
    }

    /// Only the swap icon above-right of the swatches is the reference, not the pencil above
    /// it. Its curve leaves horizontally, turns clockwise and finishes vertically downward.
    #[test]
    fn swap_heads_point_left_and_down_not_on_a_diagonal() {
        let (t0, t1) = (tangent(false).normalized(), tangent(true).normalized());
        assert!(t0.x > 0.99 && t0.y.abs() < 0.01, "the left head must point left: {t0:?}");
        assert!(t1.y > 0.99 && t1.x.abs() < 0.01, "the lower head must point down: {t1:?}");
        const { assert!(START.x < END.x && START.y < END.y, "the curve must turn toward the bottom-right") };
    }
}
