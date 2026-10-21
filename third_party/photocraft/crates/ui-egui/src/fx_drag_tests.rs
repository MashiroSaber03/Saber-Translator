//! #1719: layer effects are dragged onto another layer in the Layers panel, as in Photoshop: an
//! effect row or the fx badge moves its effects there, ⌥-drag copies them.

use egui::{Modifiers, PointerButton, Pos2, Rect, accesskit::Role, vec2};
use egui_kittest::Harness;
use egui_kittest::kittest::{NodeT, Queryable};
use photocraft_doc::LayerId;
use serde_json::json;

use crate::PhotocraftApp;
use crate::layer_row_ui::{Indicator, RowRects, recorded};

/// `a` with a Drop Shadow and a Color Overlay, and `b` above it without effects.
fn harness() -> (Harness<'static, PhotocraftApp>, LayerId, LayerId) {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
    let a = s.execute("layer.new.layer", json!({"name": "a"})).unwrap()["layer"].as_u64().unwrap();
    s.execute("layer.layerStyle.dropShadow", json!({})).unwrap();
    s.execute("layer.layerStyle.colorOverlay", json!({"color": "#ff8000"})).unwrap();
    let b = s.execute("layer.new.layer", json!({"name": "b"})).unwrap()["layer"].as_u64().unwrap();
    let mut h = Harness::builder().with_size(vec2(1440.0, 900.0)).with_max_steps(64).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        PhotocraftApp::new(s, crate::Services::default())
    });
    h.run_steps(8);
    (h, LayerId(a), LayerId(b))
}

fn row(h: &Harness<'_, PhotocraftApp>, id: LayerId) -> RowRects {
    recorded(&h.ctx).into_iter().find(|r| r.layer == id.0).unwrap()
}

/// The effect row `name` under layer `id`'s row.
fn effect_row(h: &Harness<'_, PhotocraftApp>, id: LayerId, name: &str) -> Rect {
    let layer = row(h, id).row;
    h.query_all_by_role(Role::Button)
        .filter(|n| n.accesskit_node().label().as_deref() == Some(name))
        .map(|n| n.rect())
        .find(|r| r.top() >= layer.bottom() - 1.0 && r.left() < layer.center().x && r.right() > layer.center().x)
        .unwrap_or_else(|| panic!("no {name} row under the layer"))
}

fn effects(h: &Harness<'_, PhotocraftApp>, id: LayerId) -> Vec<&'static str> {
    h.state().session.active().unwrap().doc.layer(id).unwrap().effects.items.iter().map(photocraft_doc::Effect::label).collect()
}

/// A real press, drag and release, `modifiers` held throughout.
fn drag(h: &mut Harness<'_, PhotocraftApp>, from: Pos2, to: Pos2, modifiers: Modifiers) {
    h.event(egui::Event::ModifiersChanged(modifiers));
    h.hover_at(from);
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: from, button: PointerButton::Primary, pressed: true, modifiers });
    h.run_steps(1);
    for i in 1..=6 {
        h.hover_at(from + (to - from) * (i as f32 / 6.0));
        h.run_steps(1);
    }
    h.event(egui::Event::PointerButton { pos: to, button: PointerButton::Primary, pressed: false, modifiers });
    h.run_steps(1);
    h.event(egui::Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(3);
}

#[test]
fn dragging_effects_onto_a_layer_moves_them_and_alt_copies() {
    let (mut h, a, b) = harness();
    assert_eq!(effects(&h, a), ["Drop Shadow", "Color Overlay"]);
    let steps = h.state().session.active().unwrap().history.past_len();

    // ⌥-drag one effect row onto `b`: just that effect is copied.
    let from = effect_row(&h, a, "Color Overlay").center();
    let to = row(&h, b).row.center();
    drag(&mut h, from, to, Modifiers::ALT);
    assert_eq!(effects(&h, a), ["Drop Shadow", "Color Overlay"]);
    assert_eq!(effects(&h, b), ["Color Overlay"]);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), steps + 1, "one history step");

    // Drag the fx badge onto `b`: all of `a`'s effects move there, replacing `b`'s.
    let badge = row(&h, a).indicators.into_iter().find(|(k, _)| *k == Indicator::Fx).unwrap().1;
    let to = row(&h, b).row.center();
    drag(&mut h, badge.center(), to, Modifiers::NONE);
    assert!(effects(&h, a).is_empty(), "the effects moved off `a`");
    assert_eq!(effects(&h, b), ["Drop Shadow", "Color Overlay"]);
    // The drag moved effects, not the layer: `b` is still above `a`.
    assert!(row(&h, b).row.top() < row(&h, a).row.top());
}

/// The Layers panel footer's Delete button.
fn trash_button(h: &Harness<'_, PhotocraftApp>) -> Pos2 {
    h.query_all_by_role(Role::Button).find(|n| n.accesskit_node().label().as_deref() == Some("Delete layer")).unwrap().rect().center()
}

/// #3102: an effect row dropped on the footer's Delete button deletes that effect, the "Effects"
/// row all of them; the layer stays.
#[test]
fn dropping_effects_on_the_trash_deletes_them() {
    let (mut h, a, _) = harness();
    let layers = h.state().session.active().unwrap().doc.layers.len();
    let steps = h.state().session.active().unwrap().history.past_len();
    let from = effect_row(&h, a, "Drop Shadow").center();
    let trash = trash_button(&h);
    drag(&mut h, from, trash, Modifiers::NONE);
    assert_eq!(effects(&h, a), ["Color Overlay"]);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), steps + 1, "one history step");
    let from = effect_row(&h, a, "Effects").center();
    let trash = trash_button(&h);
    drag(&mut h, from, trash, Modifiers::NONE);
    assert!(effects(&h, a).is_empty());
    assert_eq!(h.state().session.active().unwrap().doc.layers.len(), layers, "no layer was deleted");
}

/// #2846: right-clicking an effect row opens Photoshop's layer style menu for that layer; Create
/// Layer turns its effects into layers of their own.
#[test]
fn right_clicking_an_effect_row_offers_create_layer_for_that_layer() {
    let (mut h, a, b) = harness();
    assert_eq!(h.state().session.active().unwrap().active_layer, Some(b), "`b` is active, not the clicked layer");
    let at = effect_row(&h, a, "Color Overlay").center();
    h.hover_at(at);
    h.run_steps(1);
    for pressed in [true, false] {
        h.event(egui::Event::PointerButton { pos: at, button: PointerButton::Secondary, pressed, modifiers: Modifiers::NONE });
        h.run_steps(1);
    }
    h.run_steps(3);
    for label in ["Blending Options…", "Copy Layer Style", "Paste Layer Style", "Clear Layer Style", "Global Light…", "Hide All Effects", "Scale Effects…"]
    {
        h.get_by_label(label);
    }
    h.get_by_label("Create Layer").click();
    h.run_steps(4);
    let st = h.state().session.active().unwrap();
    assert_eq!(st.active_layer, Some(a), "the menu acts on the right-clicked layer");
    assert!(effects(&h, a).is_empty(), "the effects became layers");
    let names: Vec<&str> = st.doc.layers.iter().map(|l| l.name.as_str()).collect();
    assert!(names.contains(&"a's Drop Shadow") && names.contains(&"a's Color Overlay"), "{names:?}");
    assert!(h.query_by_label("Create Layer").is_none(), "the menu closed");
}
