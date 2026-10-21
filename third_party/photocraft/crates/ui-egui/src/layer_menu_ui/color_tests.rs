use super::*;
use egui::{Event, Modifiers, PointerButton, Pos2, pos2, vec2};
use egui_kittest::kittest::NodeT;
use egui_kittest::{Harness, kittest::Queryable};
use photocraft_doc::LayerId;

fn harness(kind: crate::theme::ThemeKind, ppp: f32) -> (Harness<'static, crate::PhotocraftApp>, LayerId, LayerId) {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
    let red = LayerId(s.execute("layer.new.layer", json!({"name": "Red layer"})).unwrap()["layer"].as_u64().unwrap());
    s.execute("layer.setLabelColor", json!({"color": "red"})).unwrap();
    let other = LayerId(s.execute("layer.new.layer", json!({"name": "Other layer"})).unwrap()["layer"].as_u64().unwrap());
    let mut h = Harness::builder().with_size(vec2(1440.0, 900.0)).with_pixels_per_point(ppp).with_max_steps(64).build_eframe(move |cc| {
        crate::PhotocraftApp::setup_context(&cc.egui_ctx, kind);
        let mut app = crate::PhotocraftApp::new(s, crate::Services::default());
        app.ui.theme = kind;
        app
    });
    h.run_steps(8);
    (h, red, other)
}

fn click(h: &mut Harness<'_, crate::PhotocraftApp>, at: Pos2, button: PointerButton, modifiers: Modifiers) {
    h.event(Event::ModifiersChanged(modifiers));
    h.hover_at(at);
    h.step();
    for pressed in [true, false] {
        h.event(Event::PointerButton { pos: at, button, pressed, modifiers });
        h.step();
    }
    h.run_steps(3);
}

fn row(h: &Harness<'_, crate::PhotocraftApp>, id: LayerId) -> egui::Rect {
    crate::layer_row_ui::recorded(&h.ctx).into_iter().find(|r| r.layer == id.0).unwrap().row
}

#[test]
fn layer_color_context_menu_edits_the_clicked_layer_and_preserves_multi_selection() {
    for ppp in [1.0, 2.0] {
        let (mut h, red, other) = harness(crate::theme::ThemeKind::Pro, ppp);
        let at = row(&h, red).center();
        click(&mut h, at, PointerButton::Secondary, Modifiers::NONE);
        let at = h.get_by_label("Color ⏵").rect().center();
        click(&mut h, at, PointerButton::Primary, Modifiers::NONE);
        let at = h.get_by_label("Seafoam").rect().center();
        click(&mut h, at, PointerButton::Primary, Modifiers::NONE);
        let st = h.state().session.active().unwrap();
        assert_eq!(st.active_layer, Some(red));
        assert_eq!(st.doc.layer(red).unwrap().label, LabelColor::Seafoam);
        assert_eq!(st.doc.layer(other).unwrap().label, LabelColor::None);
        h.state_mut().run("layer.select", json!({"layer": other.0, "mode": "add"})).unwrap();
        h.run_steps(4);
        let clicked = h.state().session.active().unwrap().doc.layer(red).unwrap();
        assert_eq!(common_color(h.state(), clicked, true), None);
        let at = row(&h, red).center();
        click(&mut h, at, PointerButton::Secondary, Modifiers::NONE);
        let at = h.get_by_label("Color ⏵").rect().center();
        click(&mut h, at, PointerButton::Primary, Modifiers::NONE);
        let at = h.get_by_label("Indigo").rect().center();
        click(&mut h, at, PointerButton::Primary, Modifiers::NONE);
        let st = h.state().session.active().unwrap();
        assert_eq!(st.selected_layers().len(), 2);
        assert_eq!(st.doc.layer(red).unwrap().label, LabelColor::Indigo);
        assert_eq!(st.doc.layer(other).unwrap().label, LabelColor::Indigo);
        assert_eq!(common_color(h.state(), st.doc.layer(red).unwrap(), true), Some(LabelColor::Indigo));
    }
}

#[test]
fn layer_color_background_is_confined_to_the_eye_column_in_every_theme() {
    fn rects(shape: &egui::Shape, color: egui::Color32, out: &mut Vec<egui::Rect>) {
        match shape {
            egui::Shape::Rect(r) if r.fill == color => out.push(r.rect),
            egui::Shape::Vec(shapes) => {
                for s in shapes {
                    rects(s, color, out);
                }
            }
            _ => {}
        }
    }
    fn has_icon(shape: &egui::Shape, rect: egui::Rect) -> bool {
        match shape {
            egui::Shape::Rect(r) => r.brush.is_some() && r.rect == rect,
            egui::Shape::Vec(shapes) => shapes.iter().any(|s| has_icon(s, rect)),
            _ => false,
        }
    }
    for kind in crate::theme::ThemeKind::ALL {
        let (mut h, red, _) = harness(kind, 1.0);
        for visible in [true, false] {
            h.state_mut().run("layer.setProps", json!({"layer": red.0, "visible": visible})).unwrap();
            h.run_steps(4);
            let r = row(&h, red);
            let t = crate::theme::Tokens::for_kind(kind);
            let (_, bg) = t.layer_label_colors(LabelColor::Red).unwrap();
            let mut fills = Vec::new();
            for s in &h.output().shapes {
                rects(&s.shape, bg, &mut fills);
            }
            assert_eq!(fills, vec![egui::Rect::from_min_max(r.left_top(), pos2(r.left() + 30.0, r.bottom()))], "{kind:?}");
            use egui::emath::GuiRounding;
            let icon = egui::Rect::from_center_size(pos2(r.left() + 15.0, r.center().y), vec2(15.0, 15.0)).round_to_pixels(h.ctx.pixels_per_point());
            assert_eq!(h.output().shapes.iter().any(|s| has_icon(&s.shape, icon)), visible, "centred eye in {kind:?}");
        }
        let eye = pos2(row(&h, red).left() + 17.0, row(&h, red).center().y);
        click(&mut h, eye, PointerButton::Primary, Modifiers::NONE);
        assert!(h.state().session.active().unwrap().doc.layer(red).unwrap().visible);
        assert_eq!(h.state().session.active().unwrap().doc.layer(red).unwrap().label, LabelColor::Red);
        click(&mut h, eye, PointerButton::Primary, Modifiers::ALT);
        assert!(h.state().session.active().unwrap().doc.layer(red).unwrap().visible);
    }
}

#[test]
fn right_clicking_eye_opens_color_choices_without_toggling_visibility() {
    let (mut h, red, other) = harness(crate::theme::ThemeKind::Pro, 1.0);
    let r = row(&h, red);
    let eye = pos2(r.left() + 17.0, r.center().y);
    click(&mut h, eye, PointerButton::Secondary, Modifiers::NONE);
    let blue = h.get_by_label("Blue").rect().center();
    click(&mut h, blue, PointerButton::Primary, Modifiers::NONE);

    let st = h.state().session.active().unwrap();
    assert!(st.doc.layer(red).unwrap().visible);
    assert_eq!(st.doc.layer(red).unwrap().label, LabelColor::Blue);
    assert_eq!(st.doc.layer(other).unwrap().label, LabelColor::None);
}

#[test]
fn layer_color_menu_marks_only_the_common_color() {
    let mut h = Harness::new_ui(|ui| {
        for c in LabelColor::ALL {
            color_button(ui, c, c == LabelColor::Blue);
        }
    });
    h.run();
    assert_eq!(h.get_by_label("Blue").accesskit_node().toggled(), Some(egui::accesskit::Toggled::True));
    assert_eq!(h.get_by_label("Red").accesskit_node().toggled(), Some(egui::accesskit::Toggled::False));
}
