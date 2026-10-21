use egui::{Event, Modifiers, PointerButton, Pos2, pos2, vec2};
use egui_kittest::Harness;
use photocraft_doc::{Fill, LayerContent, LayerId};
use serde_json::json;

use super::*;

fn double_click(h: &mut Harness<'_, PhotocraftApp>, at: Pos2) {
    h.run_steps(40);
    for _ in 0..2 {
        h.hover_at(at);
        h.step();
        for pressed in [true, false] {
            h.event(Event::PointerButton { pos: at, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
            h.step();
        }
        h.run_steps(3);
    }
}

fn fill(app: &PhotocraftApp, id: LayerId) -> Option<Fill> {
    match &app.session.active().unwrap().doc.layer(id).unwrap().content {
        LayerContent::Shape(sh) => sh.fill.clone(),
        _ => None,
    }
}

/// #2812: double-clicking a shape layer's thumbnail opens the Color Picker on its fill; the
/// canvas previews the pick, Cancel changes nothing and OK is one undoable `shape.edit`.
#[test]
fn shape_thumbnail_double_click_picks_the_fill_colour() {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
    let id = LayerId(s.execute("shape.create", json!({"rect": [8, 8, 40, 30], "fill": "#336699"})).unwrap()["layer"].as_u64().unwrap());
    s.execute("layer.new.layer", json!({"name": "Other"})).unwrap();
    let mut h = Harness::builder().with_size(vec2(1440.0, 900.0)).with_step_dt(1.0 / 60.0).with_max_steps(64).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        PhotocraftApp::new(s, crate::Services::default())
    });
    let ctx = h.ctx.clone();
    let (r, _) = crate::control::ControlRequest::new("ui.set", json!({"dock": {"collapsed": ["color", "properties", "history", "navigator"]}}));
    crate::control::handle(h.state_mut(), &ctx, &r);
    h.run_steps(8);
    let row = crate::layer_row_ui::recorded(&h.ctx).into_iter().find(|r| r.layer == id.0).unwrap().row;
    double_click(&mut h, pos2(row.left() + 46.0, row.center().y));

    let app = h.state_mut();
    assert_eq!(app.session.active().unwrap().active_layer, Some(id));
    let before = app.session.active().unwrap().doc.clone();
    let steps = app.session.active().unwrap().history.entries().len();
    let dialog = app.ui.dialogs.last().filter(|d| owns(&d.fields)).expect("Color Picker opened on the shape fill");
    assert_eq!(dialog.fields["color"], "#336699");
    let dialog = dialog.id;

    // Picking previews on the canvas without touching the document; Cancel drops it.
    app.ui.dialog_mut(dialog).unwrap().fields.insert("color".into(), json!("#ff8800"));
    let (shown, _) = display_doc(app, 0).unwrap();
    let LayerContent::Shape(sh) = &shown.layer(id).unwrap().content else { panic!() };
    assert_eq!(sh.fill, Some(Fill::Solid(Color::rgb(1.0, 136.0 / 255.0, 0.0))));
    app.ui.close_dialog(dialog);
    assert!(display_doc(app, 0).is_none());
    assert!(Arc::ptr_eq(&before, &app.session.active().unwrap().doc), "Cancel leaves the shape alone");

    // OK commits the colour as one undo step.
    let dialog = open(app).unwrap();
    app.ui.dialog_mut(dialog).unwrap().fields.insert("color".into(), json!("#ff8800"));
    crate::dialogs::confirm(app, dialog).unwrap();
    assert_eq!(fill(app, id), Some(Fill::Solid(Color::rgb(1.0, 136.0 / 255.0, 0.0))));
    assert_eq!(app.session.active().unwrap().history.entries().len(), steps + 1);
    app.run("edit.undo", json!({})).unwrap();
    assert_eq!(*app.session.active().unwrap().doc, *before);
}
