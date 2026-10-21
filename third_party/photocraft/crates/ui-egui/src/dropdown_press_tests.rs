//! #2735: a dropdown takes one press-drag-release, like the menus (#775): press the box, drag onto
//! an option, release to choose it.

use egui::accesskit::Role;
use egui_kittest::Harness;
use egui_kittest::kittest::{NodeT, Queryable};
use photocraft_doc::BlendMode;
use serde_json::json;

use crate::PhotocraftApp;
use crate::state::Tool;

fn app_with_a_layer() -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(egui::vec2(1400.0, 900.0)).with_max_steps(64).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 64, "height": 64})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        // No tool options bar Mode dropdown showing "Normal" too.
        app.ui.tool = Tool::Move;
        app
    });
    h.run_steps(6);
    h
}

fn blend(h: &Harness<'_, PhotocraftApp>) -> BlendMode {
    let st = h.state().session.active().unwrap();
    st.doc.layer(st.active_layer.unwrap()).unwrap().blend
}

/// The Layers panel's Blend Mode dropdown (showing `value`).
fn blend_box(h: &Harness<'_, PhotocraftApp>, value: &str) -> egui::Rect {
    let named = |n: &egui_kittest::Node<'_>| {
        let a = n.accesskit_node();
        a.value().as_deref() == Some(value) || a.label().as_deref() == Some(value)
    };
    h.query_all_by_role(Role::ComboBox).find(|n| named(n)).expect("the Layers panel's Blend Mode dropdown").rect()
}

#[test]
fn press_drag_release_picks_a_blend_mode() {
    let mut h = app_with_a_layer();
    assert_eq!(blend(&h), BlendMode::Normal);
    let at = blend_box(&h, "Normal").center();
    h.hover_at(at);
    h.run_steps(1);
    h.drag_at(at);
    h.run_steps(3);
    let multiply = h.query_by_label("Multiply").expect("the list opened on the press").rect().center();
    for k in 1..=6 {
        h.hover_at(at + (multiply - at) * (k as f32 / 6.0));
        h.run_steps(1);
    }
    h.drop_at(multiply);
    h.run_steps(3);
    assert_eq!(blend(&h), BlendMode::Multiply, "releasing on Multiply picked it");
    assert!(h.query_by_label("Color Burn").is_none(), "and closed the list");
}

#[test]
fn a_click_still_opens_the_list_for_a_second_click() {
    let mut h = app_with_a_layer();
    let at = blend_box(&h, "Normal").center();
    h.hover_at(at);
    h.run_steps(1);
    h.drag_at(at);
    h.run_steps(1);
    h.drop_at(at);
    h.run_steps(3);
    assert_eq!(blend(&h), BlendMode::Normal, "the click chose nothing");
    let screen = h.query_by_label("Screen").expect("the list stays open after a click");
    screen.click();
    h.run_steps(3);
    assert_eq!(blend(&h), BlendMode::Screen, "a second click picks");
}
