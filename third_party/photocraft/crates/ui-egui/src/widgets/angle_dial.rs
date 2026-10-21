//! Angle dial: a small wheel you click or drag to point in a direction, next to a degree field.
//! Drop Shadow's light (#882), Motion Blur and the other direction angles of the filter
//! dialogs (#3152) share it, as Photoshop's dialogs do.

use super::*;

/// `Label (dial) [field°]` for a direction in document angles: counter-clockwise from three
/// o'clock, y up. The dial points anywhere in -180..=180; the result is clamped to `range`, and
/// the field accepts the whole of `range` (Motion Blur takes -360..=360, so a typed or
/// remembered 270° stays 270°).
pub fn angle_dial_row(ui: &mut Ui, label: &str, angle: &mut f32, range: std::ops::RangeInclusive<f32>) -> Response {
    let t = Tokens::get(ui.ctx());
    // The dial clamps with max/min, not `f32::clamp`, which panics if a malformed spec
    // reverses the range.
    let (lo, hi) = (*range.start(), *range.end());
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(tl!(label)).color(t.text_dim));
        let (rect, mut dial) = ui.allocate_exact_size(vec2(42.0, 42.0), Sense::click_and_drag());
        dial.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Slider, ui.is_enabled(), tl!(label)));
        if (dial.clicked() || dial.dragged())
            && let Some(pos) = dial.interact_pointer_pos()
        {
            let delta = pos - rect.center();
            if delta.length_sq() > 1.0 {
                *angle = (-delta.y).atan2(delta.x).to_degrees().round().max(lo).min(hi);
                dial.mark_changed();
            }
        }
        let painter = ui.painter();
        let radius = rect.width() / 2.0 - 2.0;
        painter.circle(rect.center(), radius, t.field, Stroke::new(1.0, t.field_border));
        let radians = angle.to_radians();
        let tip = rect.center() + vec2(radians.cos(), -radians.sin()) * (radius - 4.0);
        painter.line_segment([rect.center(), tip], Stroke::new(2.0, t.accent));
        painter.circle_filled(rect.center(), 2.0, t.text_dim);
        if dial.has_focus() {
            painter.circle_stroke(rect.center(), radius + 2.0, Stroke::new(1.0, t.accent));
            let step = ui.input(|i| {
                i.num_presses(egui::Key::ArrowRight) as f32 + i.num_presses(egui::Key::ArrowUp) as f32
                    - i.num_presses(egui::Key::ArrowLeft) as f32
                    - i.num_presses(egui::Key::ArrowDown) as f32
            });
            if step != 0.0 {
                // Wrap round the circle only when the range is the dial's own half turn each
                // way; a wider range (Motion Blur's -360..360) steps on, so 270° goes to 271°.
                let stepped = if lo >= -180.0 && hi <= 180.0 { (*angle + step + 180.0).rem_euclid(360.0) - 180.0 } else { *angle + step };
                *angle = stepped.max(lo).min(hi);
                dial.mark_changed();
            }
        }
        // The dialogs' own number field: it takes the opening focus (#1757), Tab and arrow steps.
        let field = value_field(ui, angle, range, "°", 64.0);
        if field.changed() {
            dial.mark_changed();
        }
        dial | field
    })
    .inner
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui_kittest::kittest::Queryable;

    fn click(h: &mut egui_kittest::Harness<'_, f32>, delta: Vec2) {
        let rect = h.query_all_by_label("Angle").last().unwrap().rect();
        let pos = rect.center() + delta;
        h.event(egui::Event::PointerMoved(pos));
        h.event(egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed: true, modifiers: egui::Modifiers::NONE });
        h.step();
        h.event(egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed: false, modifiers: egui::Modifiers::NONE });
        h.run();
    }

    #[test]
    fn a_wide_range_keeps_angles_past_half_a_turn() {
        let mut h = egui_kittest::Harness::builder().with_size(vec2(300.0, 100.0)).build_ui_state(
            |ui, angle| {
                angle_dial_row(ui, "Angle", angle, -360.0..=360.0);
            },
            270.0_f32,
        );
        h.run();
        assert_eq!(*h.state(), 270.0, "the field does not clamp 270° to the dial's -180..180");
        click(&mut h, vec2(0.0, -15.0));
        assert_eq!(*h.state(), 90.0);
    }

    /// Focuses the dial (its accessibility node is the Slider labelled "Angle") and presses `key`.
    fn press_on_dial(h: &mut egui_kittest::Harness<'_, f32>, key: egui::Key) {
        h.get_by_role_and_label(egui::accesskit::Role::Slider, "Angle").focus();
        h.run();
        h.key_press(key);
        h.run();
    }

    #[test]
    fn arrow_keys_step_past_half_a_turn_on_a_wide_range() {
        let mut h = egui_kittest::Harness::builder().with_size(vec2(300.0, 100.0)).build_ui_state(
            |ui, angle| {
                angle_dial_row(ui, "Angle", angle, -360.0..=360.0);
            },
            270.0_f32,
        );
        h.run();
        press_on_dial(&mut h, egui::Key::ArrowRight);
        assert_eq!(*h.state(), 271.0, "an arrow key steps 270° on, not wrapped to -89°");
    }

    #[test]
    fn arrow_keys_wrap_round_a_half_turn_range() {
        let mut h = egui_kittest::Harness::builder().with_size(vec2(300.0, 100.0)).build_ui_state(
            |ui, angle| {
                angle_dial_row(ui, "Angle", angle, -180.0..=180.0);
            },
            180.0_f32,
        );
        h.run();
        press_on_dial(&mut h, egui::Key::ArrowRight);
        assert_eq!(*h.state(), -179.0, "Drop Shadow's dial still wraps past 180°");
    }

    #[test]
    fn dial_output_is_clamped_to_a_narrow_range() {
        let mut h = egui_kittest::Harness::builder().with_size(vec2(300.0, 100.0)).build_ui_state(
            |ui, angle| {
                angle_dial_row(ui, "Angle", angle, -90.0..=90.0);
            },
            0.0_f32,
        );
        h.run();
        click(&mut h, vec2(-15.0, -2.0));
        assert_eq!(*h.state(), 90.0, "pointing left (about 172°) stops at the range's end");
    }
}
