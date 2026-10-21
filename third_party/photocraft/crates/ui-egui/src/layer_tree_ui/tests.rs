//! #126: group disclosure triangles in the real Layers panel (whole app, real pointer input).

use egui::{Modifiers, PointerButton, Pos2, vec2};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use photocraft_doc::{Document, LayerContent, LayerId};
use serde_json::json;

use super::display_rows;
use crate::PhotocraftApp;

/// Background, then `outer { in-outer, inner { deep } }` with a pixel layer above it all.
fn layered() -> photocraft_engine::Session {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
    s.execute("layer.new.layer", json!({"name": "deep"})).unwrap();
    let inner = s.execute("layer.groupLayers", json!({"name": "inner"})).unwrap()["layer"].as_u64().unwrap();
    let outer = s.execute("layer.groupLayers", json!({"layer": inner, "name": "outer"})).unwrap()["layer"].as_u64().unwrap();
    s.execute("layer.select", json!({"layer": inner})).unwrap();
    // `layer.new.layer` lands above the active layer, inside `outer`.
    s.execute("layer.new.layer", json!({"name": "in-outer"})).unwrap();
    s.execute("layer.select", json!({"layer": outer})).unwrap();
    s.execute("layer.new.layer", json!({"name": "top"})).unwrap();
    s
}

fn harness_with(session: photocraft_engine::Session, ppp: f32) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(vec2(1440.0, 900.0)).with_pixels_per_point(ppp).with_max_steps(64).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        PhotocraftApp::new(session, crate::Services::default())
    });
    h.run_steps(8);
    h
}

fn group(h: &Harness<'_, PhotocraftApp>, name: &str) -> (LayerId, bool) {
    let doc = &h.state().session.active().unwrap().doc;
    doc.walk()
        .into_iter()
        .find_map(|(_, _, l)| match &l.content {
            LayerContent::Group(g) if l.name == name => Some((l.id, g.expanded)),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no group {name}"))
}

/// The on-screen centre of a group's triangle, scrolled into view in the Layers panel first (the
/// dock gives Layers a fixed height, so rows can sit below its fold). `None` when the row isn't
/// listed (inside a closed group).
fn triangle(h: &mut Harness<'_, PhotocraftApp>, name: &str) -> Option<Pos2> {
    let verb = if group(h, name).1 { "Collapse" } else { "Expand" };
    let label = format!("{verb} group {name}");
    h.query_by_label(&label)?.scroll_to_me();
    h.run_steps(4);
    h.query_by_label(&label).map(|n| n.rect().center())
}

/// A real click (press and release on separate frames), modifiers held throughout.
fn click(h: &mut Harness<'_, PhotocraftApp>, at: Pos2, modifiers: Modifiers) {
    h.event(egui::Event::ModifiersChanged(modifiers));
    h.hover_at(at);
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: at, button: PointerButton::Primary, pressed: true, modifiers });
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: at, button: PointerButton::Primary, pressed: false, modifiers });
    h.run_steps(1);
    h.event(egui::Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(3);
}

#[test]
fn rows_list_groups_above_their_contents_and_hide_closed_ones() {
    let s = layered();
    let doc: &Document = &s.active().unwrap().doc;
    let names = |doc: &Document, all| display_rows(doc, all).iter().map(|(d, l)| format!("{d}:{}", l.name)).collect::<Vec<_>>();
    assert_eq!(names(doc, false), ["0:top", "0:outer", "1:in-outer", "1:inner", "2:deep", "0:Background"]);
    let mut closed = doc.clone();
    let id = closed.walk().into_iter().find(|(_, _, l)| l.name == "inner").unwrap().2.id;
    if let Some(LayerContent::Group(g)) = closed.layer_mut(id).map(|l| &mut l.content) {
        g.expanded = false;
    }
    assert_eq!(names(&closed, false), ["0:top", "0:outer", "1:in-outer", "1:inner", "0:Background"]);
    assert_eq!(names(&closed, true).len(), 6, "a kind filter still finds layers in closed groups");
}

#[test]
fn clicking_a_triangle_toggles_its_group_at_every_ui_scale() {
    for ppp in [1.0, 1.5, 2.0] {
        let mut h = harness_with(layered(), ppp);
        let active = h.state().session.active().unwrap().active_layer;
        let undo = h.state().session.active().unwrap().history.entries().len();
        let p = triangle(&mut h, "inner").expect("inner's triangle shows");
        click(&mut h, p, Modifiers::NONE);
        assert!(!group(&h, "inner").1, "@{ppp}x: click collapses");
        assert!(group(&h, "outer").1, "@{ppp}x: only that group");
        // The click opened/closed the group; it didn't select it or start a drag-reorder.
        assert_eq!(h.state().session.active().unwrap().active_layer, active, "@{ppp}x: selection unchanged");
        assert_eq!(h.state().session.active().unwrap().history.entries().len(), undo, "@{ppp}x: not an undo step");
        let p = triangle(&mut h, "inner").expect("still shows, now pointing right");
        click(&mut h, p, Modifiers::NONE);
        assert!(group(&h, "inner").1, "@{ppp}x: click expands again");
        // Nested: closing the outer group hides the inner group's row (and so its triangle).
        let p = triangle(&mut h, "outer").unwrap();
        click(&mut h, p, Modifiers::NONE);
        assert!(!group(&h, "outer").1);
        assert!(triangle(&mut h, "inner").is_none(), "@{ppp}x: inner row hidden in a closed group");
    }
}

#[test]
fn alt_click_toggles_every_group() {
    let mut h = harness_with(layered(), 1.0);
    let p = triangle(&mut h, "outer").unwrap();
    click(&mut h, p, Modifiers::ALT);
    assert!(!group(&h, "outer").1 && !group(&h, "inner").1, "⌥-click closes all groups");
    let p = triangle(&mut h, "outer").unwrap();
    click(&mut h, p, Modifiers::ALT);
    assert!(group(&h, "outer").1 && group(&h, "inner").1, "⌥-click opens all groups");
    assert!(triangle(&mut h, "inner").is_some());
}

#[test]
fn state_survives_psd_export_and_import_and_imported_groups_toggle() {
    let mut h = harness_with(layered(), 2.0);
    let p = triangle(&mut h, "inner").unwrap();
    click(&mut h, p, Modifiers::NONE);
    let doc = (*h.state().session.active().unwrap().doc).clone();
    let bytes = photocraft_io::export(&doc, "groups.psd", &Default::default()).unwrap().bytes;
    let back = photocraft_io::import("groups.psd", &bytes).unwrap().document;
    let mut s = photocraft_engine::Session::new();
    s.add_document(back, None);
    let mut h = harness_with(s, 2.0);
    assert!(group(&h, "outer").1 && !group(&h, "inner").1, "open/closed state round-trips through PSD");
    // A PSD-imported group toggles like any other.
    let p = triangle(&mut h, "inner").expect("imported group has a triangle");
    click(&mut h, p, Modifiers::NONE);
    assert!(group(&h, "inner").1);
    let p = triangle(&mut h, "outer").unwrap();
    click(&mut h, p, Modifiers::NONE);
    assert!(!group(&h, "outer").1);
}
