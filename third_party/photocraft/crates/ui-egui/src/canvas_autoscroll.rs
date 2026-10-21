//! Auto-scroll while a tool drag runs past the canvas edge (#4183). As in Photoshop, dragging a
//! marquee, lasso, brush stroke or crop off the visible part of a zoomed image scrolls the view
//! toward the pointer, so a selection can be larger than what is on screen. The view stops at the
//! scroll limits (it never auto-scrolls an image that already fits into empty pasteboard).

use crate::canvas::ViewXform;
use crate::state::View;
use egui::{Pos2, Rect, Vec2, vec2};

/// Slowest and fastest auto-scroll, in egui points per second.
const MIN_SPEED: f32 = 120.0;
const MAX_SPEED: f32 = 2400.0;
/// Extra speed per point the pointer is past the edge.
const GAIN: f32 = 12.0;
/// A long frame moves at most this many seconds' worth, so a stall doesn't jump the view.
const MAX_DT: f32 = 0.05;

/// How far, in screen points, to scroll this frame for a drag whose pointer is at `pointer` over a
/// canvas of `rect`. Zero while the pointer is inside the canvas. Each axis points toward the
/// pointer and speeds up the further past the edge it is.
pub fn edge_step(pointer: Pos2, rect: Rect, dt: f32) -> Vec2 {
    if !pointer.x.is_finite() || !pointer.y.is_finite() || !dt.is_finite() || dt <= 0.0 || rect.width() <= 0.0 || rect.height() <= 0.0 {
        return Vec2::ZERO;
    }
    let past = |p: f32, lo: f32, hi: f32| {
        if p < lo {
            p - lo
        } else if p > hi {
            p - hi
        } else {
            0.0
        }
    };
    let dt = dt.min(MAX_DT);
    let axis = |d: f32| if d == 0.0 { 0.0 } else { d.signum() * (MIN_SPEED + GAIN * d.abs()).min(MAX_SPEED) * dt };
    vec2(axis(past(pointer.x, rect.left(), rect.right())), axis(past(pointer.y, rect.top(), rect.bottom())))
}

/// Move the camera by `step` in screen points, never past the image's on-screen bounds in the
/// direction of travel. A view the user already panned past a limit (Overscroll on) is not pulled
/// back: it just doesn't scroll further out. True when the view moved.
pub fn scroll(view: &mut View, step: Vec2, xf: ViewXform) -> bool {
    if !step.is_finite() || step == Vec2::ZERO || !xf.zoom.is_finite() || xf.zoom <= 0.0 {
        return false;
    }
    let image = xf.image_bounds(view.doc_size);
    if !image.is_finite() || !xf.rect.is_finite() {
        return false;
    }
    // Compare screen axes before unrotating: a document axis that fits before rotation may
    // be clipped afterwards, and a vertical screen drag must remain vertical at any angle.
    let axis = |step: f32, lo: f32, hi: f32, v0: f32, v1: f32| {
        if hi - lo <= v1 - v0 {
            0.0
        } else if step > 0.0 {
            step.min((hi - v1).max(0.0))
        } else {
            step.max((lo - v0).min(0.0))
        }
    };
    let step = vec2(
        axis(step.x, image.left(), image.right(), xf.rect.left(), xf.rect.right()),
        axis(step.y, image.top(), image.bottom(), xf.rect.top(), xf.rect.bottom()),
    );
    let d = xf.unmap_vec(step) / xf.zoom;
    let moved = d.is_finite() && (d.x.abs() > 1e-3 || d.y.abs() > 1e-3);
    if moved {
        view.center[0] += d.x;
        view.center[1] += d.y;
    }
    moved
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::pos2;

    // Existing unrotated cases express their requested movement in document pixels.
    fn scroll(view: &mut View, doc_step: Vec2, size: Vec2, ppp: f32) -> bool {
        let xf = ViewXform {
            rect: Rect::from_min_size(Pos2::ZERO, size),
            zoom: view.zoom / ppp,
            center: view.center,
            flip: false,
            rotation: view.rotation,
            aspect: 1.0,
        };
        super::scroll(view, doc_step * xf.zoom, xf)
    }

    fn rect() -> Rect {
        Rect::from_min_max(pos2(100.0, 50.0), pos2(500.0, 350.0))
    }

    #[test]
    fn inside_the_canvas_does_not_scroll() {
        assert_eq!(edge_step(pos2(300.0, 200.0), rect(), 1.0 / 60.0), Vec2::ZERO);
        assert_eq!(edge_step(pos2(100.0, 350.0), rect(), 1.0 / 60.0), Vec2::ZERO, "the edge itself is inside");
    }

    #[test]
    fn past_an_edge_scrolls_toward_the_pointer() {
        let dt = 1.0 / 60.0;
        let right = edge_step(pos2(520.0, 200.0), rect(), dt);
        assert!(right.x > 0.0 && right.y == 0.0, "{right:?}");
        let left = edge_step(pos2(80.0, 200.0), rect(), dt);
        assert!(left.x < 0.0 && left.y == 0.0, "{left:?}");
        let corner = edge_step(pos2(90.0, 360.0), rect(), dt);
        assert!(corner.x < 0.0 && corner.y > 0.0, "{corner:?}");
    }

    #[test]
    fn further_out_is_faster_up_to_a_cap() {
        let dt = 1.0 / 60.0;
        let near = edge_step(pos2(505.0, 200.0), rect(), dt).x;
        let far = edge_step(pos2(600.0, 200.0), rect(), dt).x;
        let very_far = edge_step(pos2(50_000.0, 200.0), rect(), dt).x;
        assert!(near < far, "{near} < {far}");
        assert!((very_far - MAX_SPEED * dt).abs() < 1e-3, "capped: {very_far}");
    }

    #[test]
    fn hostile_input_does_not_scroll() {
        let dt = 1.0 / 60.0;
        assert_eq!(edge_step(pos2(f32::NAN, 200.0), rect(), dt), Vec2::ZERO);
        assert_eq!(edge_step(pos2(f32::INFINITY, 200.0), rect(), dt), Vec2::ZERO);
        assert_eq!(edge_step(pos2(600.0, 200.0), rect(), f32::NAN), Vec2::ZERO);
        assert_eq!(edge_step(pos2(600.0, 200.0), rect(), -1.0), Vec2::ZERO);
        assert_eq!(edge_step(pos2(600.0, 200.0), Rect::NOTHING, dt), Vec2::ZERO);
        // A multi-second stall moves no more than one capped frame.
        assert!(edge_step(pos2(600.0, 200.0), rect(), 10.0).x <= MAX_SPEED * MAX_DT + 1e-3);
    }

    fn view(zoom: f32, center: [f32; 2]) -> View {
        View { zoom, center, doc_size: [2000, 1000], ..Default::default() }
    }

    #[test]
    fn a_zoomed_view_scrolls_and_stops_at_the_limit() {
        // 2000 × 1000 at 100 % in a 400 × 300 canvas: the centre can range over 200..1800 in x.
        let size = vec2(400.0, 300.0);
        let mut v = view(1.0, [1000.0, 500.0]);
        assert!(scroll(&mut v, vec2(50.0, 0.0), size, 1.0));
        assert_eq!(v.center, [1050.0, 500.0]);
        let mut v = view(1.0, [1790.0, 500.0]);
        assert!(scroll(&mut v, vec2(50.0, 0.0), size, 1.0));
        assert_eq!(v.center, [1800.0, 500.0], "stops at the right limit");
        assert!(!scroll(&mut v, vec2(50.0, 0.0), size, 1.0), "and stays there");
    }

    #[test]
    fn an_image_that_fits_does_not_scroll() {
        let mut v = view(0.1, [1000.0, 500.0]);
        assert!(!scroll(&mut v, vec2(50.0, 50.0), vec2(400.0, 300.0), 1.0));
        assert_eq!(v.center, [1000.0, 500.0]);
        // Wider than the view but not taller: only x scrolls.
        let mut v = view(0.25, [1000.0, 500.0]);
        assert!(scroll(&mut v, vec2(50.0, 50.0), vec2(400.0, 300.0), 1.0));
        assert_eq!(v.center, [1050.0, 500.0]);
    }

    #[test]
    fn an_overscrolled_view_is_not_pulled_back() {
        // Already panned past the right limit (Overscroll on): scrolling further right does
        // nothing, scrolling left moves normally.
        let size = vec2(400.0, 300.0);
        let mut v = view(1.0, [1900.0, 500.0]);
        assert!(!scroll(&mut v, vec2(50.0, 0.0), size, 1.0));
        assert_eq!(v.center, [1900.0, 500.0]);
        assert!(scroll(&mut v, vec2(-50.0, 0.0), size, 1.0));
        assert_eq!(v.center, [1850.0, 500.0]);
    }

    #[test]
    fn non_finite_steps_are_ignored() {
        let mut v = view(1.0, [1000.0, 500.0]);
        assert!(!scroll(&mut v, vec2(f32::NAN, 0.0), vec2(400.0, 300.0), 1.0));
        assert_eq!(v.center, [1000.0, 500.0]);
    }

    #[test]
    fn quarter_turn_limits_follow_the_displayed_image_with_hidpi_and_pixel_aspect() {
        for rotation in [-90.0, 90.0] {
            for flip in [false, true] {
                let mut v = View { zoom: 2.0, center: [300.0, 600.0], doc_size: [600, 1200], rotation, ..Default::default() };
                let xf = ViewXform { rect: Rect::from_min_size(Pos2::ZERO, vec2(800.0, 400.0)), zoom: 1.0, center: v.center, flip, rotation, aspect: 2.0 };
                assert!(super::scroll(&mut v, vec2(0.0, 10_000.0), xf));
                // At 2:1 pixel aspect the document's 600-pixel width is 1200 screen points
                // tall after rotation. The 400-point viewport exposes 200 document pixels.
                let want = if rotation > 0.0 { 500.0 } else { 100.0 };
                assert!((v.center[0] - want).abs() < 0.01, "{rotation}, flip={flip}: {:?}", v.center);
                assert!((v.center[1] - 600.0).abs() < 0.01);
                assert!(!crate::scrollbars::clamp_view_with_aspect(&mut v, xf.rect.size(), 2.0, 2.0), "the next frame preserves the auto-scroll limit");
                let xf = ViewXform { center: v.center, ..xf };
                assert!(!super::scroll(&mut v, vec2(0.0, 20.0), xf));
                v.center = [300.0, 600.0];
                let xf = ViewXform { center: v.center, ..xf };
                assert!(super::scroll(&mut v, vec2(10_000.0, 0.0), xf));
                let want_y = if (rotation > 0.0) == flip { 800.0 } else { 400.0 };
                assert!((v.center[1] - want_y).abs() < 0.01 && (v.center[0] - 300.0).abs() < 0.01, "mirrored horizontal limit: {:?}", v.center);
                assert!(!crate::scrollbars::clamp_view_with_aspect(&mut v, xf.rect.size(), 2.0, 2.0));
            }
        }
    }

    #[test]
    fn oblique_rotation_keeps_a_vertical_scroll_vertical_and_a_fitting_axis_still() {
        for flip in [false, true] {
            let mut v = View { zoom: 1.0, center: [300.0, 300.0], doc_size: [600, 600], rotation: 45.0, ..Default::default() };
            let xf =
                ViewXform { rect: Rect::from_min_size(Pos2::ZERO, vec2(900.0, 400.0)), zoom: 1.0, center: v.center, flip, rotation: v.rotation, aspect: 1.0 };
            // A 600-pixel square at 45 degrees is about 849 points across: it fits horizontally
            // but is clipped vertically. Gating document axes would produce a diagonal drift.
            assert!(!super::scroll(&mut v, vec2(30.0, 0.0), xf));
            assert!(super::scroll(&mut v, vec2(0.0, 30.0), xf));
            let after = ViewXform { center: v.center, ..xf };
            let displacement = after.to_screen(300.0, 300.0) - xf.to_screen(300.0, 300.0);
            assert!(displacement.x.abs() < 0.01 && (displacement.y + 30.0).abs() < 0.01, "{displacement:?}");
            assert!(!crate::scrollbars::clamp_view_with_aspect(&mut v, xf.rect.size(), 1.0, 1.0));
        }
    }
}
