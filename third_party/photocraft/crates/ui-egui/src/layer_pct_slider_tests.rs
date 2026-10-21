//! The Layers panel's Opacity and Fill fields open Photoshop's pop-up slider from their ▾; a drag
//! on it, a press-and-drag from the ▾ and a scrub of the number are each one history step.

use egui::{Pos2, Rect, accesskit::Role, vec2};
use egui_kittest::Harness;
use egui_kittest::kittest::{NodeT, Queryable};
use serde_json::json;

use crate::PhotocraftApp;

fn harness() -> Harness<'static, PhotocraftApp> {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 200, "height": 120})).unwrap();
    s.execute("layer.new.layer", json!({"name": "Paint"})).unwrap();
    let mut h = Harness::builder().with_size(vec2(1440.0, 1000.0)).with_max_steps(64).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        PhotocraftApp::new(s, crate::Services::default())
    });
    h.run_steps(8);
    h
}

fn layer(h: &Harness<'_, PhotocraftApp>) -> photocraft_doc::Layer {
    let st = h.state().session.active().unwrap();
    st.doc.layer(st.active_layer.unwrap()).unwrap().clone()
}

fn steps(h: &Harness<'_, PhotocraftApp>) -> usize {
    h.state().session.active().unwrap().history.entries().len()
}

fn by_role(h: &Harness<'_, PhotocraftApp>, name: &str, role: Role) -> Option<Rect> {
    h.query_all_by_role(role).find(|n| n.accesskit_node().label().as_deref() == Some(name)).map(|n| n.rect())
}

fn click(h: &mut Harness<'_, PhotocraftApp>, at: Pos2) {
    h.hover_at(at);
    h.run_steps(1);
    h.drag_at(at);
    h.run_steps(1);
    h.drop_at(at);
    h.run_steps(3);
}

fn drag(h: &mut Harness<'_, PhotocraftApp>, from: Pos2, to: Pos2) {
    h.hover_at(from);
    h.run_steps(1);
    h.drag_at(from);
    h.run_steps(1);
    for i in 1..=6 {
        h.hover_at(from + (to - from) * (i as f32 / 6.0));
        h.run_steps(1);
    }
    h.drop_at(to);
    h.run_steps(3);
}

#[test]
fn opacity_and_fill_sliders_pop_up_and_drag_in_one_step() {
    for (name, value) in [("Opacity", (|l: &photocraft_doc::Layer| l.opacity) as fn(&photocraft_doc::Layer) -> f32), ("Fill", |l| l.fill_opacity)] {
        let mut h = harness();
        assert!(by_role(&h, name, Role::Slider).is_none(), "{name}: closed until its ▾ is clicked");
        let arrow = by_role(&h, name, Role::Button).unwrap_or_else(|| panic!("{name}: no ▾"));
        click(&mut h, arrow.center());
        let slider = by_role(&h, name, Role::Slider).unwrap_or_else(|| panic!("{name}: the slider didn't open"));
        assert!(slider.top() >= arrow.bottom() || slider.bottom() <= arrow.top(), "{name}: the slider covers its field");

        // A drag from the right end towards the left: live, and one undoable step.
        let before = steps(&h);
        // The knob at 100% sits at the track's right end, 7 pt in from the slider's.
        drag(&mut h, slider.right_center() - vec2(7.0, 0.0), slider.center());
        let v = value(&layer(&h));
        assert!((0.4..0.6).contains(&v), "{name}: dragged to the middle, got {v}");
        assert_eq!(steps(&h), before + 1, "{name}: a drag is one history step");
        h.state_mut().session.undo();
        h.run_steps(3);
        assert!((value(&layer(&h)) - 1.0).abs() < 1e-6, "{name}: undo restores 100%");

        // A click outside closes it.
        click(&mut h, Pos2::new(700.0, 500.0));
        assert!(by_role(&h, name, Role::Slider).is_none(), "{name}: still open after a click outside");
    }
}

#[test]
fn pressing_the_arrow_and_dragging_sets_the_value_and_closes_on_release() {
    for (name, value) in [("Opacity", (|l: &photocraft_doc::Layer| l.opacity) as fn(&photocraft_doc::Layer) -> f32), ("Fill", |l| l.fill_opacity)] {
        let mut h = harness();
        let arrow = by_role(&h, name, Role::Button).unwrap().center();
        let before = steps(&h);
        h.hover_at(arrow);
        h.run_steps(1);
        h.drag_at(arrow);
        h.run_steps(1);
        // Half the slider's track to the left: 100% → 50%.
        let to = arrow - vec2(73.0, 0.0);
        for i in 1..=6 {
            h.hover_at(arrow + (to - arrow) * (i as f32 / 6.0));
            h.run_steps(1);
        }
        assert!(by_role(&h, name, Role::Slider).is_some(), "{name}: the slider shows while the ▾ is dragged");
        h.drop_at(to);
        h.run_steps(3);
        let v = value(&layer(&h));
        assert!((v - 0.5).abs() < 0.02, "{name}: got {v}");
        assert_eq!(steps(&h), before + 1, "{name}: a press-and-drag is one history step");
        assert!(by_role(&h, name, Role::Slider).is_none(), "{name}: release closes the slider");
    }
}

#[test]
fn scrubbing_the_number_is_one_history_step() {
    let mut h = harness();
    let arrow = by_role(&h, "Opacity", Role::Button).unwrap();
    // The number is the spin button just left of the ▾.
    let field = h
        .query_all_by_role(Role::SpinButton)
        .map(|n| n.rect())
        .find(|r| r.y_range().contains(arrow.center().y) && r.right() <= arrow.left() + 1.0 && r.right() >= arrow.left() - 30.0)
        .expect("the Opacity number");
    let before = steps(&h);
    drag(&mut h, field.center(), field.center() - vec2(40.0, 0.0));
    let v = layer(&h).opacity;
    assert!(v < 0.95, "scrubbed down, got {v}");
    assert_eq!(steps(&h), before + 1, "a scrub is one history step");
}

#[test]
fn the_background_layer_has_no_pop_up_slider() {
    let mut h = harness();
    let bg = h.state().session.active().unwrap().doc.layers[0].id.0;
    h.state_mut().run("layer.select", json!({"layer": bg})).unwrap();
    h.run_steps(8);
    let arrow = by_role(&h, "Opacity", Role::Button).unwrap();
    click(&mut h, arrow.center());
    assert!(by_role(&h, "Opacity", Role::Slider).is_none(), "Photoshop greys Opacity for the Background");
}

#[test]
fn dragging_opacity_and_fill_labels_scrubs_percentages_in_one_history_step() {
    for (name, value) in [("Opacity", (|l: &photocraft_doc::Layer| l.opacity) as fn(&photocraft_doc::Layer) -> f32), ("Fill", |l| l.fill_opacity)] {
        let mut h = harness();
        // The label, not the adjacent spin button or popup arrow, is the drag target.
        // The popup arrow is named after its field; the text label sits on the same row (egui
        // exposes a label's text as its value, not its name).
        let arrow = by_role(&h, name, Role::Button).unwrap_or_else(|| panic!("missing {name} popup arrow"));
        let with_colon = format!("{name}:");
        let rect = h
            .query_all_by_role(Role::Label)
            .filter(|node| {
                let text = node.accesskit_node().value();
                (text.as_deref() == Some(name) || text.as_deref() == Some(with_colon.as_str())) && (node.rect().center().y - arrow.center().y).abs() < 10.0
            })
            .map(|node| node.rect())
            .next()
            .unwrap_or_else(|| panic!("missing {name} text label"));
        let before = steps(&h);
        drag(&mut h, rect.center(), rect.center() - vec2(40.0, 0.0));
        assert!((value(&layer(&h)) - 0.8).abs() < 0.03, "{name}: label scrub changed the value");
        assert_eq!(steps(&h), before + 1, "{name}: a label scrub is one history step");
        h.state_mut().session.undo();
        h.run_steps(3);
        assert!((value(&layer(&h)) - 1.0).abs() < 1e-6, "{name}: undo restores the original value");
    }
}
