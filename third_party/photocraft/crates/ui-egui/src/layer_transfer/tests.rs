//! #589: layers dragged onto another document's tab, then dropped on its canvas, are copied there.

use egui::accesskit::Role;
use egui::{Modifiers, PointerButton, Pos2, Rect, vec2};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use photocraft_doc::LayerId;
use serde_json::json;

use crate::PhotocraftApp;
use crate::canvas::ViewXform;
use crate::state::Tool;

/// "source-doc" (40×30) with a red 10×10 square at (4, 6) on layer "paint", and "dest-doc"
/// (60×50). The source is active.
fn harness() -> (Harness<'static, PhotocraftApp>, LayerId) {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 40, "height": 30, "name": "source-doc"})).unwrap();
    let paint = LayerId(s.execute("layer.new.layer", json!({"name": "paint"})).unwrap()["layer"].as_u64().unwrap());
    s.execute("select.rect", json!({"x": 4, "y": 6, "width": 10, "height": 10})).unwrap();
    s.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    s.execute("file.new", json!({"width": 60, "height": 50, "name": "dest-doc"})).unwrap();
    s.set_active(0);
    let mut h = Harness::builder().with_size(vec2(1440.0, 900.0)).with_max_steps(64).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        PhotocraftApp::new(s, crate::Services::default())
    });
    h.run_steps(8);
    (h, paint)
}

/// A document tab, found by its accessible title ("name @ 100% (…)", see `pro_tabs`).
fn tab(h: &Harness<'_, PhotocraftApp>, name: &str) -> Pos2 {
    h.get_by_label_contains(&format!("{name} @")).rect().center()
}

/// Press at `path[0]`, move through the rest (a few frames at each) and release at the last
/// point with `release` held.
fn drag_along(h: &mut Harness<'_, PhotocraftApp>, path: &[Pos2], release: Modifiers) {
    let first = path[0];
    h.event(egui::Event::PointerMoved(first));
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: first, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    h.run_steps(1);
    for w in path.windows(2) {
        for k in 1..=4 {
            h.event(egui::Event::PointerMoved(w[0] + (w[1] - w[0]) * (k as f32 / 4.0)));
            h.run_steps(1);
        }
        h.run_steps(2);
    }
    let last = path[path.len() - 1];
    h.event(egui::Event::ModifiersChanged(release));
    h.event(egui::Event::PointerButton { pos: last, button: PointerButton::Primary, pressed: false, modifiers: release });
    h.run_steps(3);
    h.event(egui::Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(1);
}

/// The copy in "dest-doc" (its active layer) and its bounds.
fn copy(h: &Harness<'_, PhotocraftApp>) -> (String, Rect) {
    let st = h.state().session.active().unwrap();
    assert_eq!(st.doc.name, "dest-doc", "the destination is shown");
    let l = st.active_layer.and_then(|id| st.doc.layer(id)).unwrap();
    let b = photocraft_engine::layer_multi_cmds::layer_bounds(l).unwrap();
    (l.name.clone(), Rect::from_min_max(egui::pos2(b.x0 as f32, b.y0 as f32), egui::pos2(b.x1 as f32, b.y1 as f32)))
}

fn source_square(h: &Harness<'_, PhotocraftApp>, paint: LayerId) -> photocraft_geom::Rect {
    let doc = &h.state().session.documents()[0].doc;
    photocraft_engine::layer_multi_cmds::layer_bounds(doc.layer(paint).unwrap()).unwrap()
}

#[test]
fn a_layers_panel_drag_onto_a_tab_drops_the_layer_where_it_is_released() {
    let (mut h, _) = harness();
    let row = h.get_by_role_and_label(Role::Button, "paint").rect().center();
    let over_tab = tab(&h, "dest-doc");
    let drop = h.state().last_canvas_rect.center();
    drag_along(&mut h, &[row, over_tab, drop], Modifiers::NONE);
    let at = ViewXform::active(h.state()).unwrap().to_doc(drop);
    let (name, r) = copy(&h);
    assert_eq!(name, "paint");
    assert!((r.center().x - at[0] as f32).abs() <= 1.0 && (r.center().y - at[1] as f32).abs() <= 1.0, "centred on the drop point {at:?}: {r:?}");
    assert_eq!(h.state().session.documents()[0].doc.layers.len(), 2, "the source keeps its layers");
    assert_eq!(h.state().session.active().unwrap().doc.layers.len(), 2);
}

#[test]
fn shift_centres_the_copy_when_the_documents_differ_in_size() {
    let (mut h, _) = harness();
    let row = h.get_by_role_and_label(Role::Button, "paint").rect().center();
    let over_tab = tab(&h, "dest-doc");
    let drop = h.state().last_canvas_rect.center() + vec2(80.0, 60.0);
    drag_along(&mut h, &[row, over_tab, drop], Modifiers::SHIFT);
    assert_eq!(copy(&h).1, Rect::from_min_max(egui::pos2(25.0, 20.0), egui::pos2(35.0, 30.0)));
}

#[test]
fn a_move_tool_drag_onto_a_tab_copies_the_layer_under_the_pointer_and_leaves_the_source() {
    let (mut h, paint) = harness();
    h.state_mut().ui.tool = Tool::Move;
    h.run_steps(2);
    let before = source_square(&h, paint);
    let steps = h.state().session.documents()[0].history.entries().len();
    let grab = ViewXform::active(h.state()).unwrap().to_screen(9.0, 11.0);
    let grab_doc = ViewXform::active(h.state()).unwrap().to_doc(grab);
    let over_tab = tab(&h, "dest-doc");
    let drop = h.state().last_canvas_rect.center();
    drag_along(&mut h, &[grab, over_tab, drop], Modifiers::NONE);
    let at = ViewXform::active(h.state()).unwrap().to_doc(drop);
    let (name, r) = copy(&h);
    assert_eq!(name, "paint");
    // The grabbed pixel lands under the pointer.
    let want = egui::pos2((9.0 + at[0] - grab_doc[0]) as f32, (11.0 + at[1] - grab_doc[1]) as f32);
    assert!((r.center() - want).length() <= 1.5, "{r:?} around {want:?}");
    assert_eq!(source_square(&h, paint), before, "the source layer did not move");
    assert_eq!(h.state().session.documents()[0].history.entries().len(), steps, "no Move step in the source");
}

#[test]
fn dropping_back_on_the_source_copies_nothing() {
    let (mut h, _) = harness();
    let row = h.get_by_role_and_label(Role::Button, "paint").rect().center();
    let (dest_tab, source_tab) = (tab(&h, "dest-doc"), tab(&h, "source-doc"));
    let drop = h.state().last_canvas_rect.center();
    drag_along(&mut h, &[row, dest_tab, source_tab, drop], Modifiers::NONE);
    let docs = h.state().session.documents();
    assert_eq!((docs[0].doc.layers.len(), docs[1].doc.layers.len()), (2, 1));
    assert_eq!(h.state().session.active_index(), Some(0));
}
