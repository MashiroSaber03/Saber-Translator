//! #1747: the scroll wheel over the Layers panel's Blend Mode dropdown steps through the modes,
//! as in Photoshop: down picks the next mode, up the previous one, one history step per notch.

use egui::accesskit::Role;
use egui::{Pos2, Rect, vec2};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use photocraft_doc::BlendMode;
use serde_json::json;

use crate::PhotocraftApp;

/// The app with a new document after `setup` (its last layer is the active one).
fn harness(setup: impl FnOnce(&mut PhotocraftApp) + 'static) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(vec2(1440.0, 900.0)).with_max_steps(64).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 64, "height": 48})).unwrap();
        setup(&mut app);
        app
    });
    h.run_steps(8);
    h
}

/// A filled pixel layer above the Background, active.
fn pixel_layer(app: &mut PhotocraftApp) {
    app.run("layer.new.layer", json!({})).unwrap();
    app.run("edit.fill", json!({"color": "#cc3366"})).unwrap();
}

fn blend(h: &Harness<'_, PhotocraftApp>) -> BlendMode {
    let st = h.state().session.active().unwrap();
    st.doc.layer(st.active_layer.unwrap()).unwrap().blend
}

fn steps(h: &Harness<'_, PhotocraftApp>) -> usize {
    h.state().session.active().unwrap().history.past_len()
}

/// The Layers panel's dropdown: the lowest one (the options bar and the title bar have some too).
fn dropdown(h: &Harness<'_, PhotocraftApp>) -> Rect {
    h.query_all_by_role(Role::ComboBox).map(|n| n.rect()).max_by(|a, b| a.top().total_cmp(&b.top())).unwrap()
}

/// `lines` wheel lines at `at` in one event: negative is down.
fn wheel(h: &mut Harness<'_, PhotocraftApp>, at: Pos2, lines: f32) {
    h.hover_at(at);
    h.run_steps(1);
    h.event(egui::Event::MouseWheel {
        unit: egui::MouseWheelUnit::Line,
        delta: vec2(0.0, lines),
        phase: egui::TouchPhase::Move,
        modifiers: Default::default(),
    });
    h.run_steps(3);
}

#[test]
fn each_wheel_notch_over_the_blend_mode_dropdown_steps_one_mode_and_one_history_step() {
    let mut h = harness(pixel_layer);
    let at = dropdown(&h).center();
    let undo = steps(&h);
    assert_eq!(blend(&h), BlendMode::Normal);
    for (n, mode) in [BlendMode::Dissolve, BlendMode::Darken, BlendMode::Multiply].into_iter().enumerate() {
        wheel(&mut h, at, -1.0);
        assert_eq!(blend(&h), mode, "notch {} down", n + 1);
        assert_eq!(steps(&h), undo + n + 1, "one history step per notch");
    }
    wheel(&mut h, at, 1.0);
    assert_eq!(blend(&h), BlendMode::Darken, "a notch up goes back one");
    assert_eq!(steps(&h), undo + 4);
}

#[test]
fn several_notches_in_one_wheel_event_record_one_history_step_per_notch() {
    let mut h = harness(pixel_layer);
    let at = dropdown(&h).center();
    let undo = steps(&h);
    wheel(&mut h, at, -3.0);
    assert_eq!(blend(&h), BlendMode::Multiply);
    assert_eq!(steps(&h), undo + 3, "one history step per notch");
    h.state_mut().run("edit.undo", json!({})).unwrap();
    h.run_steps(2);
    assert_eq!(blend(&h), BlendMode::Darken, "undo goes back one notch");
}

#[test]
fn the_wheel_stops_at_the_first_and_last_blend_mode() {
    let mut h = harness(|app| {
        pixel_layer(app);
        let id = app.session.active().unwrap().active_layer.unwrap().0;
        app.run("layer.setProps", json!({"layer": id, "blend": "Luminosity"})).unwrap();
    });
    let at = dropdown(&h).center();
    let undo = steps(&h);
    wheel(&mut h, at, -1.0);
    assert_eq!((blend(&h), steps(&h)), (BlendMode::Luminosity, undo), "down on the last mode");
    let id = h.state().session.active().unwrap().active_layer.unwrap().0;
    h.state_mut().run("layer.setProps", json!({"layer": id, "blend": "Normal"})).unwrap();
    h.run_steps(2);
    let undo = steps(&h);
    wheel(&mut h, at, 1.0);
    assert_eq!((blend(&h), steps(&h)), (BlendMode::Normal, undo), "up on the first mode");
}

#[test]
fn the_wheel_steps_a_group_through_its_list_with_pass_through() {
    let mut h = harness(|app| {
        app.run("layer.new.group", json!({})).unwrap();
    });
    assert_eq!(blend(&h), BlendMode::PassThrough);
    let at = dropdown(&h).center();
    wheel(&mut h, at, -1.0);
    assert_eq!(blend(&h), BlendMode::Normal);
    wheel(&mut h, at, 1.0);
    assert_eq!(blend(&h), BlendMode::PassThrough);
    let undo = steps(&h);
    wheel(&mut h, at, 1.0);
    assert_eq!((blend(&h), steps(&h)), (BlendMode::PassThrough, undo), "Pass Through is the first");
}

#[test]
fn the_wheel_leaves_the_greyed_background_blend_mode_alone() {
    let mut h = harness(|_| {});
    let st = h.state().session.active().unwrap();
    assert!(crate::doc_props_ui::is_background(&st.doc, st.doc.layer(st.active_layer.unwrap()).unwrap()));
    let at = dropdown(&h).center();
    let undo = steps(&h);
    wheel(&mut h, at, -1.0);
    wheel(&mut h, at, -1.0);
    assert_eq!((blend(&h), steps(&h)), (BlendMode::Normal, undo));
}

fn click(h: &mut Harness<'_, PhotocraftApp>, at: Pos2) {
    h.hover_at(at);
    h.run_steps(1);
    for pressed in [true, false] {
        h.event(egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() });
        h.run_steps(1);
    }
    h.run_steps(2);
}

/// Where the entry `label` of the open list is (also when it is scrolled out of sight).
fn entry(h: &Harness<'_, PhotocraftApp>, label: &str) -> Rect {
    h.get_by_label(label).rect()
}

#[test]
fn the_wheel_over_the_open_list_steps_modes_instead_of_scrolling_it() {
    let mut h = harness(pixel_layer);
    let undo = steps(&h);
    let at = dropdown(&h).center();
    click(&mut h, at);
    let (last, darken) = (entry(&h, "Luminosity"), entry(&h, "Darken").center());
    wheel(&mut h, darken, -1.0);
    h.run_steps(10);
    assert_eq!((blend(&h), steps(&h)), (BlendMode::Dissolve, undo + 1), "one mode, one step");
    assert_eq!(entry(&h, "Luminosity"), last, "the list stays open and doesn't scroll");
}

#[test]
fn wheel_steps_in_the_open_list_keep_the_chosen_mode_in_view() {
    let mut h = harness(pixel_layer);
    let at = dropdown(&h).center();
    click(&mut h, at);
    // Unscrolled, the list shows its first entry at the top and is at most 420 pt tall.
    let (top, darken) = (entry(&h, "Normal").top(), entry(&h, "Darken").center());
    for _ in 0..22 {
        wheel(&mut h, darken, -1.0);
    }
    h.run_steps(30);
    assert_eq!(blend(&h), BlendMode::Divide);
    let in_view = |r: Rect| r.top() >= top - 0.5 && r.bottom() <= top + 420.0 + 0.5;
    let chosen = entry(&h, "Divide");
    assert!(in_view(chosen), "{chosen:?} in view below {top}");
    // The wheel over the button scrolls the open list along too.
    for _ in 0..21 {
        wheel(&mut h, at, 1.0);
    }
    h.run_steps(30);
    assert_eq!(blend(&h), BlendMode::Dissolve);
    let chosen = entry(&h, "Dissolve");
    assert!(in_view(chosen), "{chosen:?} in view below {top}");
}

/// The mode the canvas previews on the active layer, if any (#970).
fn previewed(h: &mut Harness<'_, PhotocraftApp>) -> Option<BlendMode> {
    let st = h.state().session.active().unwrap();
    let id = st.active_layer.unwrap();
    crate::blend_preview::display_doc(h.state_mut(), 0).map(|(d, _)| d.layer(id).unwrap().blend)
}

#[test]
fn a_wheel_step_ends_the_hover_preview_until_the_pointer_moves() {
    let mut h = harness(pixel_layer);
    let at = dropdown(&h).center();
    click(&mut h, at);
    let screen = entry(&h, "Screen").center();
    h.hover_at(screen);
    h.run_steps(3);
    assert_eq!(previewed(&mut h), Some(BlendMode::Screen));
    wheel(&mut h, screen, -1.0);
    assert_eq!(blend(&h), BlendMode::Dissolve);
    assert_eq!(previewed(&mut h), None, "the chosen mode shows, not the one under the pointer");
    h.run_steps(10);
    assert_eq!(previewed(&mut h), None, "still, while the pointer rests");
    let multiply = entry(&h, "Multiply").center();
    h.hover_at(multiply);
    h.run_steps(3);
    assert_eq!(previewed(&mut h), Some(BlendMode::Multiply), "a moved pointer previews again");
}
