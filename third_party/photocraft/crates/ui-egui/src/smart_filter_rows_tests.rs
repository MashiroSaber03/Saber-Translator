//! Layers panel Smart Filters rows (#4271): a click selects a filter (and its smart object), a
//! drag onto another smart object moves it there, ⌥-drag copies it, and a drag onto Delete
//! removes it, as in Photoshop.

use egui::accesskit::{Role, Toggled};
use egui::{Modifiers, Pos2};
use egui_kittest::kittest::{NodeT, Queryable};
use photocraft_doc::{LayerContent, LayerId};
use serde_json::json;

use crate::PhotocraftApp;

/// Smart objects "Smart A" (Gaussian Blur and Offset) and "Smart B" (no filters), and a pixel
/// layer "Pixels" on top: (harness, A, B, pixels).
fn panel() -> (egui_kittest::Harness<'static, PhotocraftApp>, u64, u64, u64) {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 16, "height": 16})).unwrap();
    let smart = |s: &mut photocraft_engine::Session, name: &str| {
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("edit.fill", json!({"contents": "black"})).unwrap();
        let id = s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap()["layer"].as_u64().unwrap();
        s.execute("layer.setProps", json!({"layer": id, "name": name})).unwrap();
        id
    };
    let a = smart(&mut s, "Smart A");
    s.execute("filter.blur.gaussianBlur", json!({"radius": 1})).unwrap();
    s.execute("filter.other.offset", json!({"horizontal": 2})).unwrap();
    let b = smart(&mut s, "Smart B");
    let c = s.execute("layer.new.layer", json!({})).unwrap()["layer"].as_u64().unwrap();
    s.execute("layer.setProps", json!({"layer": c, "name": "Pixels"})).unwrap();
    // 60 fps steps, so a double-click's two clicks fall inside egui's 0.3 s window.
    let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(1200.0, 900.0)).with_max_steps(64).with_step_dt(1.0 / 60.0).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        PhotocraftApp::new(s, crate::Services::default())
    });
    h.run_steps(4);
    (h, a, b, c)
}

fn filters(h: &egui_kittest::Harness<'static, PhotocraftApp>, id: u64) -> Vec<String> {
    match &h.state().session.active().unwrap().doc.layer(LayerId(id)).unwrap().content {
        LayerContent::Smart(sm) => sm.smart_filters.iter().map(|f| f.command.clone()).collect(),
        _ => Vec::new(),
    }
}

/// The centre of the Layers panel row (or footer button) labelled `label`: rows are selectable
/// labels, which AccessKit calls buttons, unlike the same name drawn elsewhere.
fn at(h: &egui_kittest::Harness<'static, PhotocraftApp>, label: &str) -> Pos2 {
    h.query_all_by_label(label).find(|n| n.accesskit_node().role() == Role::Button).unwrap_or_else(|| panic!("`{label}` drawn")).rect().center()
}

fn selected(h: &egui_kittest::Harness<'static, PhotocraftApp>, label: &str) -> bool {
    h.get_by_role_and_label(Role::Button, label).accesskit_node().toggled() == Some(Toggled::True)
}

/// Press at `from`, move to `to` in steps and release there, with `modifiers` held on release.
fn drag(h: &mut egui_kittest::Harness<'static, PhotocraftApp>, from: Pos2, to: Pos2, modifiers: Modifiers) {
    h.hover_at(from);
    h.run_steps(1);
    h.drag_at(from);
    h.run_steps(1);
    for i in 1..=6 {
        h.hover_at(from + (to - from) * (i as f32 / 6.0));
        h.run_steps(1);
    }
    h.event(egui::Event::ModifiersChanged(modifiers));
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: to, button: egui::PointerButton::Primary, pressed: false, modifiers });
    h.run_steps(4);
    h.event(egui::Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(2);
}

#[test]
fn a_click_selects_the_filter_and_its_smart_object() {
    let (mut h, a, _, c) = panel();
    assert_eq!(h.state().session.active().unwrap().active_layer, Some(LayerId(c)));
    h.get_by_role_and_label(Role::Button, "Gaussian Blur").click();
    h.run_steps(4);
    assert_eq!(h.state().session.active().unwrap().active_layer, Some(LayerId(a)), "the smart object becomes active");
    assert!(selected(&h, "Gaussian Blur"));
    assert!(!selected(&h, "Offset"));
    // Another layer: the filter is no longer shown selected.
    let pixels = at(&h, "Pixels");
    h.hover_at(pixels);
    h.run_steps(1);
    h.drag_at(pixels);
    h.run_steps(1);
    h.drop_at(pixels);
    h.run_steps(4);
    assert_eq!(h.state().session.active().unwrap().active_layer, Some(LayerId(c)));
    assert!(!selected(&h, "Gaussian Blur"));
}

#[test]
fn dragging_a_filter_onto_another_smart_object_moves_it_and_alt_copies_it() {
    let (mut h, a, b, _) = panel();
    let (blur, offset) = ("filter.blur.gaussianBlur", "filter.other.offset");
    let (from, to) = (at(&h, "Gaussian Blur"), at(&h, "Smart B"));
    drag(&mut h, from, to, Modifiers::ALT);
    assert_eq!(filters(&h, b), [blur], "⌥-drag copies");
    assert_eq!(filters(&h, a), [blur, offset]);
    let (from, to) = (at(&h, "Offset"), at(&h, "Smart B"));
    drag(&mut h, from, to, Modifiers::NONE);
    assert_eq!(filters(&h, b), [blur, offset], "a drag moves");
    assert_eq!(filters(&h, a), [blur]);
}

#[test]
fn a_pixel_layer_does_not_take_smart_filters() {
    let (mut h, a, _, _) = panel();
    let (from, to) = (at(&h, "Gaussian Blur"), at(&h, "Pixels"));
    drag(&mut h, from, to, Modifiers::NONE);
    assert_eq!(filters(&h, a).len(), 2);
    assert!(!h.state().ui.status_error, "{}", h.state().ui.status);
}

#[test]
fn dragging_filters_onto_delete_removes_them() {
    let (mut h, a, _, _) = panel();
    let (from, to) = (at(&h, "Offset"), at(&h, "Delete layer"));
    drag(&mut h, from, to, Modifiers::NONE);
    assert_eq!(filters(&h, a), ["filter.blur.gaussianBlur"]);
    let (from, to) = (at(&h, "Smart Filters"), at(&h, "Delete layer"));
    drag(&mut h, from, to, Modifiers::NONE);
    assert!(filters(&h, a).is_empty());
    assert_eq!(h.state().session.active().unwrap().doc.layers.len(), 4, "no layer was deleted");
}

/// The report's case: the Camera Raw Filter is ⌥-dragged to another smart object with its
/// settings, and double-clicking the copy opens Camera Raw on that smart object's filter.
#[test]
fn a_camera_raw_filter_copies_and_its_copy_opens_on_the_target() {
    let (mut h, a, b, _) = panel();
    h.state_mut().run("layer.select", json!({"layer": a})).unwrap();
    h.state_mut().run("filter.cameraRaw", json!({"exposure": 1.0, "contrast": 20})).unwrap();
    h.run_steps(4);
    let (from, to) = (at(&h, "Camera Raw Filter"), at(&h, "Smart B"));
    drag(&mut h, from, to, Modifiers::ALT);
    let raw = |h: &egui_kittest::Harness<'static, PhotocraftApp>, id: u64| match &h.state().session.active().unwrap().doc.layer(LayerId(id)).unwrap().content {
        LayerContent::Smart(sm) => sm.smart_filters.iter().find(|f| f.command == "filter.cameraRaw").map(|f| f.params.clone()),
        _ => None,
    };
    assert_eq!(raw(&h, b), raw(&h, a), "copied with its settings");
    assert_eq!(raw(&h, b).unwrap()["exposure"], 1.0);
    // B is above A in the panel, so its copy is the upper of the two Camera Raw rows.
    let copy = h
        .query_all_by_label("Camera Raw Filter")
        .filter(|n| n.accesskit_node().role() == Role::Button)
        .map(|n| n.rect().center())
        .min_by(|p, q| p.y.total_cmp(&q.y))
        .unwrap();
    for _ in 0..2 {
        h.hover_at(copy);
        h.drag_at(copy);
        h.run_steps(1);
        h.drop_at(copy);
        h.run_steps(1);
    }
    h.run_steps(3);
    let dialog = h.state().camera_raw.as_ref().expect("Camera Raw opened");
    assert_eq!(dialog.layer, LayerId(b));
}

/// Dragging a filter up or down its own stack reorders it (Photoshop), which changes the result:
/// the bottom filter is applied first (`smart_cmds::tests::filter_stack_order_blend_disable_and_clear`). A drop on the upper half of a row lands above it, on the
/// lower half below it, on the Smart Filters row on top; the moved filter stays selected.
#[test]
fn dragging_a_filter_up_or_down_its_stack_reorders_it() {
    let (mut h, a, _, _) = panel();
    let (blur, offset) = ("filter.blur.gaussianBlur", "filter.other.offset");
    let half = |p: Pos2, up: bool| p + egui::vec2(0.0, if up { -6.0 } else { 6.0 });
    // Gaussian Blur (bottom) onto the upper half of Offset's row: blur now on top.
    let (from, to) = (at(&h, "Gaussian Blur"), at(&h, "Offset"));
    drag(&mut h, from, half(to, true), Modifiers::NONE);
    assert_eq!(filters(&h, a), [offset, blur]);
    assert!(selected(&h, "Gaussian Blur"), "the moved filter stays selected");
    assert!(!h.state().ui.status_error, "{}", h.state().ui.status);
    // Its lower half: back under it.
    let (from, to) = (at(&h, "Gaussian Blur"), at(&h, "Offset"));
    drag(&mut h, from, half(to, false), Modifiers::NONE);
    assert_eq!(filters(&h, a), [blur, offset]);
    // The Smart Filters row: on top again.
    let (from, to) = (at(&h, "Gaussian Blur"), at(&h, "Smart Filters"));
    drag(&mut h, from, to, Modifiers::NONE);
    assert_eq!(filters(&h, a), [offset, blur]);
    // Each drag is one undo step.
    h.state_mut().run("edit.undo", json!({})).unwrap();
    h.run_steps(4);
    assert_eq!(filters(&h, a), [blur, offset]);
}

fn click_at(h: &mut egui_kittest::Harness<'static, PhotocraftApp>, p: Pos2) {
    h.hover_at(p);
    h.run_steps(1);
    h.drag_at(p);
    h.run_steps(1);
    h.drop_at(p);
    h.run_steps(4);
}

/// #4751: with a filter's row selected, clicking Delete removes that filter, not the smart object.
#[test]
fn clicking_delete_removes_the_selected_filter() {
    let (mut h, a, _, _) = panel();
    let blur = at(&h, "Gaussian Blur");
    click_at(&mut h, blur);
    let trash = at(&h, "Delete layer");
    click_at(&mut h, trash);
    assert_eq!(filters(&h, a), ["filter.other.offset"]);
    assert_eq!(h.state().session.active().unwrap().doc.layers.len(), 4, "no layer was deleted");
    // The deleted row's selection doesn't carry over to Offset, now at its index.
    click_at(&mut h, trash);
    assert_eq!(filters(&h, a), ["filter.other.offset"]);
}

/// Clicking the smart object's own row leaves its filters: Delete then deletes the layer as usual.
#[test]
fn clicking_the_smart_object_row_deselects_its_filter() {
    let (mut h, a, _, _) = panel();
    let offset = at(&h, "Offset");
    click_at(&mut h, offset);
    let row = at(&h, "Smart A");
    click_at(&mut h, row);
    assert_eq!(h.state().session.active().unwrap().active_layer, Some(LayerId(a)));
    assert!(!selected(&h, "Offset"));
    let trash = at(&h, "Delete layer");
    click_at(&mut h, trash);
    assert_eq!(filters(&h, a).len(), 2, "the filters stay");
}
