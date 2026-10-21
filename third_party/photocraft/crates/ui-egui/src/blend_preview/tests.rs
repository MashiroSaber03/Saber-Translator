//! #970: hovering a blend mode in the Layers panel previews it on the canvas; only a choice
//! records a history step.

use egui::accesskit::Role;
use egui::{PointerButton, Pos2, vec2};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use photocraft_color::BlendMode;
use photocraft_doc::LayerId;
use serde_json::json;

use super::{damage, display_doc, hover};
use crate::PhotocraftApp;

/// Background and a filled layer `top` (active).
fn session() -> (photocraft_engine::Session, LayerId) {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
    let top = s.execute("layer.new.layer", json!({"name": "top"})).unwrap()["layer"].as_u64().unwrap();
    s.execute("select.rect", json!({"x": 8, "y": 8, "width": 16, "height": 16})).unwrap();
    s.execute("edit.fill", json!({"color": "#cc3366"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    (s, LayerId(top))
}

fn blend(app: &PhotocraftApp, id: LayerId) -> BlendMode {
    app.session.active().unwrap().doc.layer(id).unwrap().blend
}

fn steps(app: &PhotocraftApp) -> usize {
    app.session.active().unwrap().history.past_len()
}

#[test]
fn a_hovered_mode_shows_on_the_canvas_until_the_pointer_leaves() {
    let (s, top) = session();
    let mut app = PhotocraftApp::new(s, crate::Services::default());
    let (undo, revision) = (steps(&app), app.session.active().unwrap().revision);
    hover(&mut app, top, Some(BlendMode::Multiply));
    let (shown, key) = display_doc(&mut app, 0).expect("previewed");
    assert_eq!(shown.layer(top).unwrap().blend, BlendMode::Multiply);
    let (again, same) = display_doc(&mut app, 0).unwrap();
    assert!(std::sync::Arc::ptr_eq(&shown, &again) && key == same, "one document per hovered mode, not one per frame");
    assert_eq!((blend(&app, top), steps(&app)), (BlendMode::Normal, undo), "hovering records nothing");
    // From the document to the preview and between previews: only the layer's area redraws.
    let doc = app.session.active().unwrap().doc.id;
    assert_eq!(damage(&app, doc, revision, 0, key).map(|r| r.width()), Some(16));
    hover(&mut app, top, Some(BlendMode::Screen));
    let (_, screen) = display_doc(&mut app, 0).unwrap();
    assert_ne!(screen, key);
    assert!(damage(&app, doc, revision, key, screen).is_some());
    // The current mode, or no mode, shows the document; and so does a list no longer drawn.
    hover(&mut app, top, Some(BlendMode::Normal));
    assert!(display_doc(&mut app, 0).is_none());
    hover(&mut app, top, None);
    assert!(display_doc(&mut app, 0).is_none());
    assert!(damage(&app, doc, revision, screen, 0).is_some(), "back to the document: the layer's area");
    hover(&mut app, top, Some(BlendMode::Multiply));
    app.frame += 2;
    assert!(display_doc(&mut app, 0).is_none(), "a stale hover previews nothing");
}

fn harness(session: photocraft_engine::Session) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(vec2(1440.0, 900.0)).with_max_steps(64).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        PhotocraftApp::new(session, crate::Services::default())
    });
    h.run_steps(8);
    h
}

fn click(h: &mut Harness<'_, PhotocraftApp>, at: Pos2) {
    h.hover_at(at);
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: at, button: PointerButton::Primary, pressed: true, modifiers: Default::default() });
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: at, button: PointerButton::Primary, pressed: false, modifiers: Default::default() });
    h.run_steps(3);
}

fn previewed(h: &mut Harness<'_, PhotocraftApp>) -> Option<BlendMode> {
    display_doc(h.state_mut(), 0).map(|(d, _)| d.layers.last().unwrap().blend)
}

#[test]
fn hovering_the_layers_panel_blend_list_previews_and_a_click_commits() {
    let (s, top) = session();
    let mut h = harness(s);
    let undo = steps(h.state());
    // The Layers panel's dropdown: the lowest one (the options bar and the title bar have some too).
    let dropdown = h.query_all_by_role(Role::ComboBox).map(|n| n.rect()).max_by(|a, b| a.top().total_cmp(&b.top())).unwrap();
    click(&mut h, dropdown.center());
    let multiply = h.get_by_label("Multiply").rect().center();
    h.hover_at(multiply);
    h.run_steps(3);
    assert_eq!(previewed(&mut h), Some(BlendMode::Multiply), "the hovered mode shows");
    assert_eq!((blend(h.state(), top), steps(h.state())), (BlendMode::Normal, undo), "no edit while hovering");
    h.hover_at(Pos2::new(400.0, 400.0));
    h.run_steps(3);
    assert_eq!(previewed(&mut h), None, "off the list: the document again");
    let screen = h.get_by_label("Screen").rect().center();
    click(&mut h, screen);
    assert_eq!(blend(h.state(), top), BlendMode::Screen);
    assert_eq!(steps(h.state()), undo + 1, "choosing records one step");
    assert_eq!(previewed(&mut h), None);
}

#[test]
fn moving_between_modes_keeps_a_preview_across_the_gaps_between_them() {
    // #2553: the pointer between two rows used to hover neither, flashing the document back.
    let (s, _) = session();
    let mut h = harness(s);
    let dropdown = h.query_all_by_role(Role::ComboBox).map(|n| n.rect()).max_by(|a, b| a.top().total_cmp(&b.top())).unwrap();
    click(&mut h, dropdown.center());
    let (dissolve, darken) = (h.get_by_label("Dissolve").rect(), h.get_by_label("Darken").rect());
    assert!(darken.top() > dissolve.bottom(), "the rows have a gap between them");
    let multiply = h.get_by_label("Multiply").rect().center();
    let mut y = dissolve.center().y;
    while y <= multiply.y {
        h.hover_at(Pos2::new(multiply.x, y));
        h.run_steps(1);
        assert!(previewed(&mut h).is_some(), "a mode previews at y = {y}");
        y += 0.5;
    }
    assert_eq!(previewed(&mut h), Some(BlendMode::Multiply));
}
