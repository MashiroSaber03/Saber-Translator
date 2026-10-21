//! Integer-only Indexed Color field and a linear slider with common palette sizes.

use super::*;

const STOPS: [u32; 6] = [8, 16, 32, 64, 128, 256];

fn parse_count(text: &str) -> Option<f64> {
    super::parse_num(text).filter(|v| *v >= 0.0 && *v <= f64::from(u32::MAX) && v.fract() == 0.0)
}

fn position(track: Rect, count: u32) -> f32 {
    track.left() + count.saturating_sub(2) as f32 / 254.0 * track.width()
}

fn at_pointer(track: Rect, x: f32) -> u32 {
    // Use the nearest marker: low counts are close together on the linear track.
    if let Some(stop) = STOPS.iter().copied().min_by(|&a, &b| (position(track, a) - x).abs().total_cmp(&(position(track, b) - x).abs()))
        && (position(track, stop) - x).abs() <= 6.0
    {
        return stop;
    }
    let fraction = ((x - track.left()) / track.width().max(1.0)).clamp(0.0, 1.0);
    (2.0 + fraction * 254.0).round() as u32
}

/// A count in 2..=256. Direct entry allows every integer; only slider gestures snap to markers.
pub fn color_count_row(ui: &mut Ui, label: &str, count: &mut u32) -> Response {
    let t = Tokens::get(ui.ctx());
    *count = (*count).clamp(2, 256);
    let before = *count;
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(tl!(label)).color(t.text_dim));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let (rect, _) = ui.allocate_exact_size(vec2(74.0, 24.0), Sense::hover());
            surface(ui, rect, t.field, false);
            if !t.bevel {
                ui.painter().rect_stroke(rect, t.radius_sm, Stroke::new(1.0, t.field_border), StrokeKind::Inside);
            }
            let mut child = ui.new_child(egui::UiBuilder::new().max_rect(rect.shrink2(vec2(4.0, 2.0))));
            child.style_mut().visuals.widgets.inactive.bg_fill = Color32::TRANSPARENT;
            child.style_mut().visuals.widgets.inactive.weak_bg_fill = Color32::TRANSPARENT;
            child.style_mut().visuals.widgets.inactive.bg_stroke = Stroke::NONE;
            child.style_mut().visuals.widgets.hovered.bg_stroke = Stroke::NONE;
            child.style_mut().override_font_id = Some(theme::mono(12.0));
            child.add(egui::DragValue::new(count).range(2..=256).speed(1.0).max_decimals(0).custom_parser(parse_count));
        });
    });
    let (rect, mut response) = ui.allocate_exact_size(vec2(ui.available_width().max(60.0), 36.0), Sense::click_and_drag());
    let track = Rect::from_min_max(pos2(rect.left() + 7.0, rect.top() + 7.5), pos2(rect.right() - 7.0, rect.top() + 10.5));
    if (response.clicked() || response.dragged())
        && let Some(pointer) = response.interact_pointer_pos()
    {
        *count = at_pointer(track, pointer.x).clamp(2, 256);
        response.request_focus();
    }
    if response.has_focus() {
        ui.input_mut(|input| {
            for key in [egui::Key::ArrowLeft, egui::Key::ArrowDown] {
                if input.consume_key(egui::Modifiers::NONE, key) {
                    *count = count.saturating_sub(1).max(2);
                }
            }
            for key in [egui::Key::ArrowRight, egui::Key::ArrowUp] {
                if input.consume_key(egui::Modifiers::NONE, key) {
                    *count = count.saturating_add(1).min(256);
                }
            }
        });
    }
    response.widget_info(|| egui::WidgetInfo::slider(ui.is_enabled(), f64::from(*count), label));
    let x = position(track, *count);
    let painter = ui.painter();
    painter.rect_filled(track, t.radius_sm, t.field_border);
    painter.rect_filled(Rect::from_min_max(track.min, pos2(x, track.bottom())), t.radius_sm, t.accent);
    for stop in STOPS {
        let sx = position(track, stop);
        painter.line_segment([pos2(sx, track.bottom() + 3.0), pos2(sx, track.bottom() + 6.0)], Stroke::new(1.0, t.text_faint));
        painter.text(
            pos2(sx, track.bottom() + 9.0),
            if stop == 256 { Align2::RIGHT_TOP } else { Align2::CENTER_TOP },
            stop.to_string(),
            theme::mono(10.0),
            t.text_faint,
        );
    }
    let knob = pos2(x, track.center().y);
    if t.bevel {
        surface(ui, Rect::from_center_size(knob, vec2(10.0, 16.0)), t.card, true);
    } else {
        painter.circle_filled(knob, 6.0, t.text);
        if response.hovered() || response.dragged() {
            painter.circle_stroke(knob, 8.0, Stroke::new(2.0, t.accent_soft));
        }
    }
    if *count != before {
        response.mark_changed();
    }
    ui.add_space(4.0);
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui_kittest::{Harness, kittest::Queryable};

    #[test]
    fn typed_color_counts_reject_fractions_and_allow_arbitrary_integers() {
        for (text, expected) in [("17", 17), ("256", 256), ("17.5", 64), ("17,5", 64), ("1e2", 100), ("(8+8)*2", 32), ("2^8", 256), ("5/2", 64)] {
            let mut harness = Harness::builder().with_size(vec2(320.0, 100.0)).build_ui_state(
                |ui, count: &mut u32| {
                    color_count_row(ui, "Colors", count);
                },
                64,
            );
            harness.get_by_role(egui::accesskit::Role::SpinButton).click();
            harness.run();
            harness.event(egui::Event::Text(text.into()));
            harness.key_press(egui::Key::Enter);
            harness.run();
            assert_eq!(*harness.state(), expected, "{text}");
        }
    }

    #[test]
    fn slider_clicks_snap_to_every_marker_but_allow_intermediate_counts() {
        for count in STOPS {
            let mut harness = Harness::builder().with_size(vec2(320.0, 100.0)).build_ui_state(
                |ui, count: &mut u32| {
                    color_count_row(ui, "Colors", count);
                },
                2,
            );
            let rect = harness.get_by_role(egui::accesskit::Role::Slider).rect();
            let track = rect.shrink2(vec2(7.0, 0.0));
            let point = pos2(position(track, count) - 4.0, rect.top() + 9.0);
            harness.event(egui::Event::PointerMoved(point));
            for pressed in [true, false] {
                harness.event(egui::Event::PointerButton { pos: point, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE });
                harness.run();
            }
            assert_eq!(*harness.state(), count);
            harness.key_press(egui::Key::ArrowRight);
            harness.run();
            assert_eq!(*harness.state(), (count + 1).min(256));
        }
        let track = Rect::from_min_size(Pos2::ZERO, vec2(300.0, 3.0));
        assert_eq!(at_pointer(track, position(track, 23)), 23);
        assert_eq!(position(track, 129), track.center().x);
        assert!(((position(track, 128) - position(track, 64)) - (position(track, 96) - position(track, 32))).abs() < 0.001);
    }
}

#[cfg(test)]
mod arithmetic_tests {
    #[test]
    fn count_expression_preserves_integer_validation() {
        assert_eq!(super::parse_count("2^8"), Some(256.0));
        assert_eq!(super::parse_count("(8+8)*2"), Some(32.0));
        assert_eq!(super::parse_count("5/2"), None);
        assert_eq!(super::parse_count("1/0"), None);
        assert_eq!(super::parse_count("-2"), None);
    }
}
