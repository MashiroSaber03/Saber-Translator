//! Fullscreen status-bar scrubbing, with absolute cursor-warp movement removed from the value.

use egui::{Pos2, Rect, Response, Ui, Vec2};

#[derive(Clone, Copy)]
struct Scrub {
    precise: f64,
    pending: Option<(Pos2, Pos2)>,
}

pub(crate) fn active(ui: &Ui) -> bool {
    !cfg!(target_arch = "wasm32") && ui.input(|i| i.focused && i.viewport().fullscreen == Some(true) && !i.any_touches())
}

fn wrap_target(pos: Pos2, viewport: Rect) -> Option<Pos2> {
    if !viewport.is_finite() || viewport.width() < 32.0 || !pos.is_finite() || pos.y < viewport.top() || pos.y > viewport.bottom() {
        return None;
    }
    if pos.x <= viewport.left() + 1.0 {
        Some(Pos2::new(viewport.right() - 3.0, pos.y))
    } else if pos.x >= viewport.right() - 1.0 {
        Some(Pos2::new(viewport.left() + 3.0, pos.y))
    } else {
        None
    }
}

fn unwarped(delta: Vec2, pos: Pos2, pending: &mut Option<(Pos2, Pos2)>) -> Vec2 {
    if let Some((from, to)) = *pending {
        let middle = (from.x + to.x) / 2.0;
        if (to.x > from.x && pos.x > middle) || (to.x < from.x && pos.x < middle) {
            *pending = None;
            return delta - (to - from);
        }
    }
    delta
}

pub(crate) fn update(ui: &Ui, resp: &Response, value: &mut f32, before: f32, range: std::ops::RangeInclusive<f32>, speed: f64, wrapping: bool) {
    let key = resp.id.with("wrapped-scrub");
    if !wrapping || !resp.dragged_by(egui::PointerButton::Primary) || !ui.input(|i| i.pointer.primary_down()) {
        ui.data_mut(|d| d.remove::<Scrub>(key));
        return;
    }
    let (pos, delta, viewport, slow, aim) = ui.input(|i| (i.pointer.latest_pos(), i.pointer.delta(), i.viewport_rect(), i.modifiers.shift, i.aim_radius()));
    let Some(pos) = pos else { return };
    let mut scrub = ui.data(|d| d.get_temp::<Scrub>(key)).unwrap_or(Scrub { precise: f64::from(before), pending: None });
    let delta = unwarped(delta, pos, &mut scrub.pending);
    let speed = if slow { speed / 10.0 } else { speed };
    let change = f64::from(delta.x - delta.y) * speed;
    let (lo, hi) = (f64::from(*range.start()), f64::from(*range.end()));
    if change.is_finite() && change != 0.0 && lo <= hi {
        scrub.precise = (scrub.precise + change).clamp(lo, hi);
        let aim = f64::from(aim) * speed;
        let rounded = egui::emath::smart_aim::best_in_range_f64(scrub.precise - aim, scrub.precise + aim);
        *value = egui::emath::round_to_decimals(rounded, 3).clamp(lo, hi) as f32;
    }
    if scrub.pending.is_none()
        && let Some(to) = wrap_target(pos, viewport)
    {
        ui.ctx().send_viewport_cmd(egui::ViewportCommand::CursorPosition(to));
        scrub.pending = Some((pos, to));
        ui.ctx().request_repaint();
    }
    ui.ctx().accesskit_node_builder(resp.id, |node| node.set_numeric_value_step(speed));
    ui.data_mut(|d| d.insert_temp(key, scrub));
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{pos2, vec2};

    #[test]
    fn warp_preserves_real_motion_in_the_acknowledgement_frame() {
        for (from, to, motion) in [(pos2(0.5, 600.0), pos2(997.0, 600.0), vec2(-8.0, 2.0)), (pos2(999.5, 600.0), pos2(3.0, 600.0), vec2(8.0, -2.0))] {
            let mut pending = Some((from, to));
            assert_eq!(unwarped(Vec2::ZERO, from, &mut pending), Vec2::ZERO, "wait for an actual warp acknowledgement");
            assert!(pending.is_some());
            assert_eq!(unwarped(to - from + motion, to + motion, &mut pending), motion);
            assert!(pending.is_none());
            assert_eq!(unwarped(motion, to + motion * 2.0, &mut pending), motion, "ordinary movement after the warp");
        }
    }

    #[test]
    fn warp_uses_logical_viewport_edges_even_when_motion_crosses_the_edge() {
        let rect = Rect::from_min_size(pos2(20.0, 30.0), vec2(800.0, 600.0));
        assert_eq!(wrap_target(pos2(20.5, 500.0), rect), Some(pos2(817.0, 500.0)));
        assert_eq!(wrap_target(pos2(819.5, 500.0), rect), Some(pos2(23.0, 500.0)));
        assert_eq!(wrap_target(pos2(19.0, 500.0), rect), Some(pos2(817.0, 500.0)));
        assert_eq!(wrap_target(pos2(20.5, 631.0), rect), None);
        assert_eq!(wrap_target(rect.center(), rect), None);
        assert_eq!(wrap_target(pos2(0.5, 10.0), Rect::from_min_size(Pos2::ZERO, vec2(8.0, 20.0))), None);
    }
}
