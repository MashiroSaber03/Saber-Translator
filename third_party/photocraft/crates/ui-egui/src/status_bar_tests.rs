//! The status bar's zoom field and Fit button sit inside the bar with an even margin above and
//! below, in every theme (#2823).

use egui::{Rect, pos2, vec2};
use egui_kittest::{Harness, kittest::Queryable};
use serde_json::json;

use crate::PhotocraftApp;
use crate::theme::{ThemeKind, Tokens};

const SIZE: egui::Vec2 = vec2(900.0, 200.0);

fn bottom_id() -> egui::Id {
    egui::Id::new("status-bar-test-bottom")
}

fn harness(kind: ThemeKind) -> Harness<'static, PhotocraftApp> {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 1080, "height": 1080, "resolution": 72, "background": "white"})).unwrap();
    app.sync_views();
    let mut h = Harness::builder().with_size(SIZE).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            // The first frame runs before `setup_context` installs the theme's fonts.
            if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                // Where the bar goes: the bottom of the harness's (margined) root Ui.
                let bottom = ui.max_rect().bottom();
                ui.ctx().data_mut(|d| d.insert_temp(bottom_id(), bottom));
                crate::panels::status_bar(app, ui);
            }
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, kind);
    h.run_steps(3);
    h
}

#[test]
fn status_bar_controls_sit_inside_the_bar_with_even_margins() {
    for kind in ThemeKind::ALL {
        let h = harness(kind);
        let t = Tokens::for_kind(kind);
        let (bar_h, control_h) = crate::panels::status_bar_heights(&t);
        let margin = (bar_h - control_h) / 2.0;
        assert!(margin >= 3.0, "{kind:?}: margin {margin}");
        let bottom = h.ctx.data(|d| d.get_temp::<f32>(bottom_id())).unwrap();
        let bar = Rect::from_min_max(pos2(0.0, bottom - bar_h), pos2(SIZE.x, bottom));
        // Half a point of slack for egui's pixel rounding of the centred row.
        let inner = bar.shrink2(vec2(margin, margin - 0.5));
        // The zoom number (its box is painted around it, `control_h` tall and centred).
        let number = h.get_by_role(egui::accesskit::Role::SpinButton).rect();
        assert!(inner.contains_rect(number), "{kind:?}: bar {bar:?}, number {number:?}");
        // Pro (Photoshop) has no Fit button in the status bar.
        if t.pro {
            assert!(h.query_by_label("Fit").is_none(), "{kind:?}");
            continue;
        }
        let fit = h.get_by_label("Fit").rect();
        assert!(inner.contains_rect(fit), "{kind:?}: bar {bar:?}, Fit {fit:?}");
        assert!((fit.center().y - bar.center().y).abs() < 0.51, "{kind:?}: bar {bar:?}, Fit {fit:?}");
        assert!((fit.height() - control_h).abs() < 0.01, "{kind:?}: Fit {fit:?}");
    }
}

fn cursor_warp(h: &Harness<'_, PhotocraftApp>) -> Option<egui::Pos2> {
    h.output().viewport_output.values().flat_map(|v| &v.commands).find_map(|c| match c {
        egui::ViewportCommand::CursorPosition(p) => Some(*p),
        _ => None,
    })
}

fn fullscreen(h: &mut Harness<'_, PhotocraftApp>, on: bool) {
    h.input_mut().viewports.entry(egui::ViewportId::ROOT).or_default().fullscreen = Some(on);
    h.state_mut().ui.views[0].zoom = 10.0;
    h.run_steps(2);
}

fn begin_zoom_drag(h: &mut Harness<'_, PhotocraftApp>) -> egui::Pos2 {
    let start = h.get_by_role(egui::accesskit::Role::SpinButton).rect().center();
    h.hover_at(start);
    h.run_steps(1);
    h.drag_at(start);
    h.run_steps(1);
    start
}

/// #3344: both status-bar layouts can scrub across either fullscreen edge. Simulate the native
/// CursorPosition acknowledgement as a PointerMoved, then keep dragging in the same direction.
#[test]
fn fullscreen_status_zoom_wraps_without_turning_the_warp_into_a_value_change() {
    for kind in ThemeKind::ALL {
        for left in [true, false] {
            let mut h = harness(kind);
            fullscreen(&mut h, true);
            let start = begin_zoom_drag(&mut h);
            let edge = pos2(if left { 0.5 } else { SIZE.x - 0.5 }, start.y);
            h.hover_at(edge);
            h.run_steps(1);
            let target = cursor_warp(&h).unwrap_or_else(|| panic!("{kind:?}: wrap at {edge:?}"));
            assert_eq!(target.y, edge.y, "{kind:?}: preserve Y");
            assert_eq!(target.x, if left { SIZE.x - 3.0 } else { 3.0 });
            let at_edge = h.state().ui.views[0].zoom;
            h.hover_at(target);
            h.run_steps(1);
            assert_eq!(h.state().ui.views[0].zoom, at_edge, "{kind:?}: cursor teleport must not change zoom");
            assert!(cursor_warp(&h).is_none(), "{kind:?}: landing margin must not bounce");
            let next = target + vec2(if left { -40.0 } else { 40.0 }, 0.0);
            h.hover_at(next);
            h.run_steps(1);
            let continued = h.state().ui.views[0].zoom;
            assert!((continued - at_edge - if left { -0.2 } else { 0.2 }).abs() < 0.015, "{kind:?}: {at_edge} -> {continued}");
            h.drop_at(next);
            h.run_steps(2);
            h.hover_at(edge);
            h.run_steps(1);
            assert_eq!(h.state().ui.views[0].zoom, continued, "{kind:?}: pointer movement after release is not scrubbing");
            assert!(cursor_warp(&h).is_none());
        }
    }
}

#[test]
fn status_zoom_does_not_warp_for_windowed_scrubbing_or_hovering() {
    for kind in [ThemeKind::ProMedium, ThemeKind::Studio] {
        let mut h = harness(kind);
        fullscreen(&mut h, true);
        h.hover_at(pos2(0.5, SIZE.y - 10.0));
        h.run_steps(1);
        assert!(cursor_warp(&h).is_none(), "hovering the edge");
        fullscreen(&mut h, false);
        let start = begin_zoom_drag(&mut h);
        let edge = pos2(0.5, start.y);
        h.hover_at(edge);
        h.run_steps(1);
        assert!(cursor_warp(&h).is_none(), "ordinary windowed DragValue");
        assert!(h.state().ui.views[0].zoom < 10.0);
        h.drop_at(edge);
        h.run_steps(2);
    }
}

#[test]
fn fullscreen_status_zoom_keeps_typing_and_clears_a_pending_warp_on_release() {
    let mut h = harness(ThemeKind::ProMedium);
    fullscreen(&mut h, true);
    let start = begin_zoom_drag(&mut h);
    h.hover_at(pos2(0.5, start.y));
    h.run_steps(1);
    let target = cursor_warp(&h).expect("wrap");
    let before = h.state().ui.views[0].zoom;
    h.drop_at(target);
    h.run_steps(2);
    assert_eq!(h.state().ui.views[0].zoom, before);
    h.get_by_role(egui::accesskit::Role::SpinButton).click();
    h.run_steps(2);
    h.event(egui::Event::Text("250%".into()));
    h.key_press(egui::Key::Enter);
    h.run_steps(3);
    assert_eq!(h.state().ui.views[0].zoom, 2.5);
    assert!(cursor_warp(&h).is_none());
    h.get_by_role(egui::accesskit::Role::SpinButton).click();
    h.run_steps(2);
    h.key_press(egui::Key::ArrowUp);
    h.run_steps(2);
    assert!((h.state().ui.views[0].zoom - 2.51).abs() < 0.0001, "keep keyboard stepping in fullscreen");
}

#[test]
fn fullscreen_status_zoom_preserves_shift_precision_and_recovers_from_the_minimum() {
    let mut h = harness(ThemeKind::ProMedium);
    fullscreen(&mut h, true);
    let start = begin_zoom_drag(&mut h);
    h.event(egui::Event::ModifiersChanged(egui::Modifiers::SHIFT));
    let to = start + vec2(40.0, 0.0);
    h.hover_at(to);
    h.run_steps(1);
    assert!((h.state().ui.views[0].zoom - 10.02).abs() < 0.005, "Shift reduces sensitivity by ten");
    h.drop_at(to);
    h.event(egui::Event::ModifiersChanged(egui::Modifiers::NONE));
    h.run_steps(2);
    h.state_mut().ui.views[0].zoom = 0.01;
    h.run_steps(2);
    let start = begin_zoom_drag(&mut h);
    h.hover_at(pos2(0.5, start.y));
    h.run_steps(1);
    let target = cursor_warp(&h).expect("wrap");
    let minimum = crate::zoom_levels::min(h.state().ui.views[0].doc_size);
    assert_eq!(h.state().ui.views[0].zoom, minimum);
    h.hover_at(target);
    h.run_steps(1);
    let next = target - vec2(100.0, 0.0);
    h.hover_at(next);
    h.run_steps(1);
    assert_eq!(h.state().ui.views[0].zoom, minimum);
    h.hover_at(next + vec2(20.0, 0.0));
    h.run_steps(1);
    assert!(h.state().ui.views[0].zoom > minimum, "reversing direction responds immediately after reaching a bound");
}

/// #5016: toggling appearance between dark and light with default settings preserves the Pro
/// layout and does not make a Fit button appear.
#[test]
fn appearance_toggle_preserves_pro_layout_and_has_no_fit_button() {
    let mut h = harness(ThemeKind::ProMedium);
    assert!(h.query_by_label("Fit").is_none());
    assert!(Tokens::get(&h.ctx).pro);

    // Switch to Light mode (top-bar appearance button).
    let ctx = h.ctx.clone();
    crate::prefs_ui::cycle_appearance(h.state_mut(), &ctx);
    if h.state().session.prefs().interface.appearance_mode == photocraft_engine::prefs::AppearanceMode::Auto {
        crate::prefs_ui::cycle_appearance(h.state_mut(), &ctx);
    }
    crate::prefs_ui::tick(h.state_mut(), &ctx);
    h.run_steps(3);
    assert_eq!(h.state().session.prefs().interface.appearance_mode, photocraft_engine::prefs::AppearanceMode::Light);
    assert_eq!(h.state().ui.theme, ThemeKind::ProLight);
    assert!(Tokens::get(&h.ctx).pro);
    assert!(h.query_by_label("Fit").is_none(), "Fit button must not appear in light mode (#5016)");
}
