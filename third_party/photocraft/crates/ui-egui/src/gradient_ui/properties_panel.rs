//! Compact live-gradient Properties controls, sharing the editor's stop interaction.

use super::*;
use crate::{icons, props_layout, theme};

const STYLES: [(GradientStyle, &str); 5] = [
    (GradientStyle::Linear, "Linear Gradient"),
    (GradientStyle::Radial, "Radial Gradient"),
    (GradientStyle::Angle, "Angle Gradient"),
    (GradientStyle::Reflected, "Reflected Gradient"),
    (GradientStyle::Diamond, "Diamond Gradient"),
];

pub(super) fn show(app: &mut PhotocraftApp, ui: &mut egui::Ui, layer: &Layer) {
    let document = app.session.active().map(|s| s.doc.id);
    ui.push_id(("gradient-properties", document, layer.id.0), |ui| {
        layer_controls(app, ui, layer);
    });
}

pub(super) fn editor(app: &mut PhotocraftApp, ui: &mut egui::Ui, layer: &Layer) {
    let document = app.session.active().map(|s| s.doc.id);
    ui.push_id(("gradient-editor-layer", document, layer.id.0), |ui| {
        layer_controls(app, ui, layer);
    });
}

fn layer_controls(app: &mut PhotocraftApp, ui: &mut egui::Ui, layer: &Layer) {
    if props_layout::section(ui, "gradient-fill-stops", "Gradient")
        && let Some(fill) = current_fill(app, layer.id)
    {
        stops_editor_for(app, ui, Sink::Layer(layer.id), fill);
    }
    options(app, ui, layer);
}

fn options(app: &mut PhotocraftApp, ui: &mut egui::Ui, layer: &Layer) {
    let Some(Fill::Gradient { angle, scale, style, reverse, dither, align, .. }) = fill_of(layer).cloned() else { return };
    if props_layout::section(ui, "gradient-fill-options", "Gradient Options") {
        ui.horizontal(|ui| {
            row_label(ui, "Style");
            ui.spacing_mut().item_spacing.x = 3.0;
            for (value, name) in STYLES {
                if style_button(ui, value, value == style, name).clicked() {
                    let _ = app.run(cmds::SET, json!({"layer": layer.id.0, "style": cmds::style_name(value)}));
                }
            }
        });
        if let Some(value) = numeric_row(ui, "angle", "Angle", angle, -180.0..=180.0, "°") {
            let _ = app.run(cmds::SET, json!({"layer": layer.id.0, "angle": value}));
        }
        if let Some(value) = numeric_row(ui, "scale", "Scale", scale * 100.0, 10.0..=150.0, "%") {
            let _ = app.run(cmds::SET, json!({"layer": layer.id.0, "scale": value}));
        }
        ui.horizontal_wrapped(|ui| {
            let mut value = reverse;
            if widgets::checkbox(ui, &mut value, "Reverse").changed() {
                let _ = app.run(cmds::SET, json!({"layer": layer.id.0, "reverse": value}));
            }
            let mut value = dither;
            if widgets::checkbox(ui, &mut value, "Dither").changed() {
                let _ = app.run(cmds::SET, json!({"layer": layer.id.0, "dither": value}));
            }
        });
        let mut value = align;
        if widgets::checkbox(ui, &mut value, "Align with layer").changed() {
            let _ = app.run(cmds::SET, json!({"layer": layer.id.0, "align": value}));
        }
        if widgets::secondary_button(ui, "Reset Alignment", ui.available_width()).on_hover_text(tl!("Centre the gradient (offset 0, 0)")).clicked() {
            let _ = app.run(cmds::SET, json!({"layer": layer.id.0, "offset": [0, 0]}));
        }
    }
}

fn current_fill(app: &PhotocraftApp, id: LayerId) -> Option<Fill> {
    fill_of(app.session.active()?.doc.layer(id)?).cloned()
}

fn row_label(ui: &mut egui::Ui, name: &str) -> egui::Response {
    let t = Tokens::get(ui.ctx());
    ui.add_sized(vec2(72.0, 24.0), egui::Label::new(egui::RichText::new(tl!(name)).color(t.text_dim)).halign(egui::Align::Min))
}

/// Scrubbing/typing commits once, as the existing Properties fields do.
fn numeric_row(ui: &mut egui::Ui, id: &str, name: &str, current: f32, range: std::ops::RangeInclusive<f32>, suffix: &str) -> Option<f32> {
    ui.push_id(id, |ui| {
        ui.horizontal(|ui| {
            let label = row_label(ui, name);
            let key = ui.id().with("draft");
            let mut value = ui.data(|d| d.get_temp::<f32>(key)).unwrap_or(current);
            let response = widgets::value_field(ui, &mut value, range, suffix, ui.available_width().min(88.0)).labelled_by(label.id);
            if response.dragged() || response.has_focus() {
                ui.data_mut(|d| d.insert_temp(key, value));
                return None;
            }
            ui.data_mut(|d| d.remove::<f32>(key));
            (response.drag_stopped() || response.lost_focus() || response.changed()).then_some(value).filter(|v| (*v - current).abs() > 1e-3)
        })
        .inner
    })
    .inner
}

/// Small procedural previews: no proprietary icon assets, colours follow the theme.
fn style_button(ui: &mut egui::Ui, style: GradientStyle, selected: bool, name: &str) -> egui::Response {
    let t = Tokens::get(ui.ctx());
    let (rect, response) = ui.allocate_exact_size(vec2(25.0, 25.0), Sense::click());
    icons::button_chrome(ui, rect, selected, response.hovered());
    let preview = rect.shrink(5.0);
    // One mesh avoids antialiased seams between the small preview's cells.
    let mut mesh = egui::Mesh::default();
    const N: usize = 16;
    for y in 0..N {
        for x in 0..N {
            let (u, v) = ((x as f32 + 0.5) / N as f32, (y as f32 + 0.5) / N as f32);
            let (dx, dy) = ((u - 0.5) * 2.0, (v - 0.5) * 2.0);
            let amount = match style {
                GradientStyle::Linear => u,
                GradientStyle::Radial => dx.hypot(dy).min(1.0),
                GradientStyle::Angle => (dy.atan2(dx) / std::f32::consts::TAU + 1.0).fract(),
                GradientStyle::Reflected => dx.abs(),
                GradientStyle::Diamond => (dx.abs() + dy.abs()).min(1.0),
            };
            let channel = |a: u8, b: u8| (f32::from(a) + (f32::from(b) - f32::from(a)) * amount) as u8;
            let color = Color32::from_rgb(channel(t.field.r(), t.text.r()), channel(t.field.g(), t.text.g()), channel(t.field.b(), t.text.b()));
            let min = preview.min + vec2(x as f32, y as f32) * preview.size() / N as f32;
            mesh.add_colored_rect(Rect::from_min_size(min, preview.size() / N as f32), color);
        }
    }
    ui.painter().add(egui::Shape::mesh(mesh));
    ui.painter().rect_stroke(preview, 0.0, Stroke::new(1.0, t.text_dim), StrokeKind::Outside);
    response.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::SelectableLabel, ui.is_enabled(), selected, tl!(name)));
    response.on_hover_text(tl!(name))
}

pub(super) fn stop_fields(app: &mut PhotocraftApp, ui: &mut egui::Ui, sink: Sink) {
    let Some(Fill::Gradient { mut stops, mut opacity_stops, midpoints, .. }) = sink.fill(app) else { return };
    stops.sort_by(|a, b| a.0.total_cmp(&b.0));
    if opacity_stops.is_empty() {
        opacity_stops = vec![(0.0, 1.0), (1.0, 1.0)];
    }
    opacity_stops.sort_by(|a, b| a.0.total_cmp(&b.0));
    let key = strip_key(app, sink).with("selection");
    let selected = ui.data(|d| d.get_temp::<Marker>(key)).unwrap_or(Marker::Color(0));
    let (index, location, count) = match selected {
        Marker::Color(i) => {
            let Some((location, color)) = stops.get(i) else {
                ui.data_mut(|d| d.remove::<Marker>(key));
                return;
            };
            ui.horizontal(|ui| {
                row_label(ui, "Color");
                let (rect, response) = ui.allocate_exact_size(vec2(88.0, 24.0), Sense::click());
                let rgb = color.to_rgb();
                ui.painter().rect_filled(rect.shrink(2.0), Tokens::get(ui.ctx()).radius_sm, c32([rgb[0], rgb[1], rgb[2], 1.0]));
                ui.painter().rect_stroke(rect, Tokens::get(ui.ctx()).radius_sm, Stroke::new(1.0, Tokens::get(ui.ctx()).field_border), StrokeKind::Inside);
                response.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), tl!("Color")));
                if response.on_hover_text(tl!("Color")).clicked() {
                    match sink {
                        Sink::Layer(id) => edit_stop_color(app, id, i),
                        Sink::Preset => preset_stop_color(app, i),
                    }
                }
            });
            (i, *location, stops.len())
        }
        Marker::Opacity(i) => {
            let Some((location, opacity)) = opacity_stops.get(i) else {
                ui.data_mut(|d| d.remove::<Marker>(key));
                return;
            };
            if let Some(value) = numeric_row(ui, &format!("opacity-{i}"), "Opacity", *opacity * 100.0, 0.0..=100.0, "%") {
                let _ = app.run(sink.cmd(), sink.params(json!({"action": "opacity", "kind": "opacity", "index": i, "opacity": value})));
            }
            (i, *location, opacity_stops.len())
        }
        Marker::Mid(i) => {
            if stops.get(i).is_none() || stops.get(i.saturating_add(1)).is_none() {
                ui.data_mut(|d| d.remove::<Marker>(key));
                return;
            }
            if let Some(value) = numeric_row(ui, &format!("midpoint-{i}"), "Midpoint", midpoints.get(i).copied().unwrap_or(0.5) * 100.0, 5.0..=95.0, "%") {
                let _ = app.run(sink.cmd(), sink.params(json!({"action": "midpoint", "index": i, "location": value / 100.0})));
            }
            return;
        }
    };
    let kind = if matches!(selected, Marker::Opacity(_)) { "opacity" } else { "color" };
    if let Some(value) = numeric_row(ui, &format!("{kind}-location-{index}"), "Location", location * 100.0, 0.0..=100.0, "%") {
        let p = sink.params(json!({"action": "move", "kind": kind, "index": index, "location": value / 100.0}));
        let selection = sink.fill(app).and_then(|f| selection_after_edit(&f, &p, Some(selected)));
        if app.run(sink.cmd(), p).is_ok() {
            if let Some(marker) = selection {
                ui.data_mut(|d| d.insert_temp(key, marker));
            }
            // Index belongs to the old ordering; render the remaining controls next frame.
            return;
        }
    }
    ui.horizontal(|ui| {
        ui.add_space(72.0 + ui.spacing().item_spacing.x);
        ui.add_enabled_ui(count > 2, |ui| {
            let response = icons::button(ui, "trash", 24.0, false, "Delete");
            response.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), tl!("Delete")));
            if response.clicked() {
                let _ = app.run(sink.cmd(), sink.params(json!({"action": "delete", "kind": kind, "index": index})));
                ui.data_mut(|d| d.remove::<Marker>(key));
            }
        });
    });
    ui.add_space(theme::ROW_GAP);
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui_kittest::{
        Harness,
        kittest::{NodeT, Queryable},
    };

    fn app() -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 200, "height": 120})).unwrap();
        app.run(cmds::CREATE, json!({"from": [20, 60], "to": [180, 60], "stops": [[0, "#ff0000"], [1, "#0000ff"]], "transparency": [[0, 20], [1, 100]]}))
            .unwrap();
        app
    }

    fn panel(ui: &mut egui::Ui, app: &mut PhotocraftApp) {
        if ui.ctx().cumulative_pass_nr() == 0 {
            // Font definitions take effect on the next pass, before section headers draw.
            PhotocraftApp::setup_context(ui.ctx(), theme::ThemeKind::ProMedium);
            return;
        }
        let state = app.session.active().unwrap();
        let layer = state.doc.layer(state.active_layer.unwrap()).unwrap().clone();
        show(app, ui, &layer);
    }

    fn fill(app: &PhotocraftApp) -> Fill {
        let state = app.session.active().unwrap();
        current_fill(app, state.active_layer.unwrap()).unwrap()
    }

    fn click_strip(h: &mut Harness<'_, PhotocraftApp>, location: f32, y: f32) {
        let rect = h.get_by_label("Gradient stops").rect();
        let pos = pos2(rect.left() + 8.0 + (rect.width() - 16.0) * location, rect.top() + y);
        h.event(egui::Event::PointerMoved(pos));
        h.run_steps(1);
        h.event(egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed: true, modifiers: egui::Modifiers::NONE });
        h.run_steps(1);
        h.event(egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed: false, modifiers: egui::Modifiers::NONE });
        h.run_steps(3);
    }

    #[test]
    fn style_icons_edit_the_selected_layer_with_one_undo_step() {
        let mut h = Harness::builder().with_size(vec2(280.0, 800.0)).build_ui_state(panel, app());
        PhotocraftApp::setup_context(&h.ctx, theme::ThemeKind::ProMedium);
        h.run_steps(3);
        for (style, name) in STYLES.into_iter().skip(1) {
            let before = fill(h.state());
            let count = h.state().session.active().unwrap().history.past_len();
            h.get_by_label(name).click();
            h.run_steps(3);
            let after = fill(h.state());
            assert!(matches!(&after, Fill::Gradient { style: actual, .. } if *actual == style));
            assert_eq!(h.state().session.active().unwrap().doc.layers.len(), 2);
            assert_eq!(h.state().session.active().unwrap().history.past_len(), count + 1);
            h.state_mut().run("edit.undo", json!({})).unwrap();
            assert_eq!(fill(h.state()), before);
            h.state_mut().run("edit.redo", json!({})).unwrap();
            assert_eq!(fill(h.state()), after);
            h.run_steps(2);
        }
    }

    #[test]
    fn combined_strip_adds_only_the_clicked_kind_and_keeps_color_picker_working() {
        let mut h = Harness::builder().with_size(vec2(280.0, 800.0)).build_ui_state(panel, app());
        PhotocraftApp::setup_context(&h.ctx, theme::ThemeKind::ProMedium);
        h.run_steps(3);
        assert!(h.query_by_label("Color stops").is_none());
        assert!(h.query_by_label("Opacity stops").is_none());
        assert!(h.get_by_label("Gradient stops").rect().bottom() < h.get_by_label("Gradient Options").rect().top());
        for opacity in [false, true] {
            let before = fill(h.state());
            let count = h.state().session.active().unwrap().history.past_len();
            click_strip(&mut h, 0.5, if opacity { 22.0 } else { 34.0 });
            let after = fill(h.state());
            let Fill::Gradient { stops: old_colors, opacity_stops: old_opacity, .. } = &before else { panic!() };
            let Fill::Gradient { stops: colors, opacity_stops, .. } = &after else { panic!() };
            if opacity {
                assert_eq!(colors, old_colors);
                assert_eq!(opacity_stops.len(), old_opacity.len() + 1);
            } else {
                assert_eq!(opacity_stops, old_opacity);
                assert_eq!(colors.len(), old_colors.len() + 1);
            }
            assert_eq!(h.state().session.active().unwrap().history.past_len(), count + 1);
            h.state_mut().run("edit.undo", json!({})).unwrap();
            assert_eq!(fill(h.state()), before);
            h.state_mut().run("edit.redo", json!({})).unwrap();
            assert_eq!(fill(h.state()), after);
            h.run_steps(2);
        }
        // The selected opacity's fields are replaced when a bottom color handle is selected.
        assert!(h.query_all_by_label("Color").all(|n| n.rect().width() < 80.0));
        let before = fill(h.state());
        let count = h.state().session.active().unwrap().history.past_len();
        click_strip(&mut h, 0.0, 50.0);
        assert_eq!(fill(h.state()), before);
        assert_eq!(h.state().session.active().unwrap().history.past_len(), count);
        h.query_all_by_label("Color").find(|n| n.rect().width() > 80.0).unwrap().click();
        h.run_steps(2);
        assert!(!h.state().ui.dialogs.is_empty());
    }

    #[test]
    fn opacity_numeric_edit_commits_once_and_undo_restores_the_gradient() {
        let mut h = Harness::builder().with_size(vec2(280.0, 800.0)).build_ui_state(panel, app());
        PhotocraftApp::setup_context(&h.ctx, theme::ThemeKind::ProMedium);
        h.run_steps(3);
        click_strip(&mut h, 0.0, 6.0);
        let before = fill(h.state());
        let count = h.state().session.active().unwrap().history.past_len();
        // Selected opacity and location precede angle and scale.
        h.get_all_by_role(egui::accesskit::Role::SpinButton).next().unwrap().click();
        h.run_steps(2);
        h.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::A);
        h.event(egui::Event::Text("35".into()));
        h.run_steps(2);
        assert_eq!(fill(h.state()), before, "typing waits until committed");
        h.key_press(egui::Key::Enter);
        h.run_steps(3);
        assert!(matches!(fill(h.state()), Fill::Gradient { opacity_stops, .. } if (opacity_stops[0].1 - 0.35).abs() < 1e-5));
        assert_eq!(h.state().session.active().unwrap().history.past_len(), count + 1);
        h.state_mut().run("edit.undo", json!({})).unwrap();
        assert_eq!(fill(h.state()), before);
    }

    #[test]
    fn opacity_delete_is_one_undo_step_and_preserves_two_stop_minimum() {
        let mut app = app();
        app.run(cmds::STOP, json!({"action": "add", "kind": "opacity", "location": 0.5})).unwrap();
        let mut h = Harness::builder().with_size(vec2(280.0, 800.0)).build_ui_state(panel, app);
        PhotocraftApp::setup_context(&h.ctx, theme::ThemeKind::ProMedium);
        h.run_steps(3);
        assert!(h.get_all_by_label("Delete").next().unwrap().accesskit_node().is_disabled(), "two colors cannot be reduced further");
        click_strip(&mut h, 0.0, 6.0);
        assert!(!h.get_all_by_label("Delete").next().unwrap().accesskit_node().is_disabled(), "three opacity points can be reduced");
        let before = fill(h.state());
        let count = h.state().session.active().unwrap().history.past_len();
        h.get_all_by_label("Delete").next().unwrap().click();
        h.run_steps(3);
        let after = fill(h.state());
        let Fill::Gradient { stops: old_colors, opacity_stops: old_opacity, .. } = &before else { panic!() };
        let Fill::Gradient { stops: colors, opacity_stops, .. } = &after else { panic!() };
        assert_eq!(colors, old_colors);
        assert_eq!(opacity_stops, &old_opacity[1..]);
        assert_eq!(h.state().session.active().unwrap().history.past_len(), count + 1);
        click_strip(&mut h, 0.5, 6.0);
        assert!(h.get_all_by_label("Delete").next().unwrap().accesskit_node().is_disabled(), "two opacity points cannot be reduced further");
        h.state_mut().run("edit.undo", json!({})).unwrap();
        assert_eq!(fill(h.state()), before);
        h.run_steps(3);
        assert!(!h.get_all_by_label("Delete").next().unwrap().accesskit_node().is_disabled());
        h.state_mut().run("edit.redo", json!({})).unwrap();
        assert_eq!(fill(h.state()), after);
    }

    #[test]
    fn controls_fit_a_narrow_panel_in_every_theme() {
        assert!(icons::exists("trash"), "the stop deletion button has an embedded icon");
        for theme in theme::ThemeKind::ALL {
            let mut h = Harness::builder().with_size(vec2(260.0, 820.0)).build_ui_state(panel, app());
            PhotocraftApp::setup_context(&h.ctx, theme);
            h.run_steps(3);
            for (_, name) in STYLES {
                let rect = h.get_by_label(name).rect();
                assert!(rect.right() <= 260.0 && rect.left() >= 0.0, "{theme:?}: {name} {rect:?}");
            }
            let rect = h.get_by_label("Gradient stops").rect();
            assert!(rect.right() <= 260.0, "{theme:?}: {rect:?}");
        }
    }

    #[test]
    fn selection_follows_sorted_stops_including_equal_positions() {
        let f = Fill::gradient(
            vec![(0.0, photocraft_color::Color::BLACK), (0.5, photocraft_color::Color::WHITE), (1.0, photocraft_color::Color::WHITE)],
            0.0,
            1.0,
            GradientStyle::Linear,
            false,
        );
        assert_eq!(selection_after_edit(&f, &json!({"action": "move", "index": 0, "location": 0.5}), None), Some(Marker::Color(0)));
        assert_eq!(selection_after_edit(&f, &json!({"action": "move", "index": 0, "location": 0.75}), None), Some(Marker::Color(1)));
        assert_eq!(selection_after_edit(&f, &json!({"action": "add", "location": 0.5}), None), Some(Marker::Color(2)));
        assert_eq!(selection_after_edit(&f, &json!({"action": "move", "kind": "opacity", "index": 0, "location": 1.0}), None), Some(Marker::Opacity(0)));
    }

    #[test]
    fn dragging_a_stop_across_another_keeps_selection_and_commits_once() {
        let mut app = app();
        app.run(cmds::STOP, json!({"action": "add", "location": 0.4})).unwrap();
        let id = app.session.active().unwrap().active_layer.unwrap();
        let mut h = Harness::builder().with_size(vec2(280.0, 800.0)).build_ui_state(panel, app);
        PhotocraftApp::setup_context(&h.ctx, theme::ThemeKind::ProMedium);
        h.run_steps(3);
        let before = fill(h.state());
        let count = h.state().session.active().unwrap().history.past_len();
        let rect = h.get_by_label("Gradient stops").rect();
        let from = pos2(rect.left() + 8.0, rect.top() + 50.0);
        let to = pos2(rect.left() + 8.0 + (rect.width() - 16.0) * 0.7, from.y);
        h.event(egui::Event::PointerMoved(from));
        h.run_steps(1);
        h.event(egui::Event::PointerButton { pos: from, button: egui::PointerButton::Primary, pressed: true, modifiers: egui::Modifiers::NONE });
        h.run_steps(1);
        h.event(egui::Event::PointerMoved(to));
        h.run_steps(2);
        assert_eq!(fill(h.state()), before, "only preview changes during dragging");
        assert!(h.state().gradient.panel.is_some(), "large first move still grabbed the original stop");
        h.event(egui::Event::PointerButton { pos: to, button: egui::PointerButton::Primary, pressed: false, modifiers: egui::Modifiers::NONE });
        h.run_steps(3);
        assert_eq!(h.state().session.active().unwrap().history.past_len(), count + 1);
        assert!(matches!(fill(h.state()), Fill::Gradient { stops, .. } if (stops[1].0 - 0.7).abs() < 1e-5));
        let key = strip_key(h.state(), Sink::Layer(id)).with("selection");
        assert_eq!(h.ctx.data(|d| d.get_temp::<Marker>(key)), Some(Marker::Color(1)));
        h.get_all_by_label("Delete").next().unwrap().click();
        h.run_steps(3);
        assert!(matches!(fill(h.state()), Fill::Gradient { stops, .. } if stops.len() == 2 && (stops[0].0 - 0.4).abs() < 1e-5));
        h.state_mut().run("edit.undo", json!({})).unwrap();
        h.state_mut().run("edit.undo", json!({})).unwrap();
        assert_eq!(fill(h.state()), before);
    }
}
