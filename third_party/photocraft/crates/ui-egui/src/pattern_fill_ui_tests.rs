use egui::{Event, Modifiers, PointerButton, Pos2, pos2, vec2};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use photocraft_doc::{Fill, LayerContent, LayerId};
use serde_json::json;

use super::*;

fn app() -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 40, "height": 30, "depth": 16})).unwrap();
    app
}

fn pattern_id(app: &PhotocraftApp, name: &str) -> String {
    app.session.patterns.items.iter().find(|p| p.name == name).unwrap().id.clone()
}

fn steps(app: &PhotocraftApp) -> usize {
    app.session.active().unwrap().history.entries().len()
}

fn set(app: &mut PhotocraftApp, dialog: u64, values: serde_json::Value) {
    let d = app.ui.dialog_mut(dialog).unwrap();
    for (k, v) in values.as_object().unwrap() {
        d.fields.insert(k.clone(), v.clone());
    }
}

#[test]
fn new_pattern_fill_opens_the_dialog_previews_and_ok_is_one_step() {
    let mut app = app();
    let ctx = egui::Context::default();
    let before = app.session.active().unwrap().doc.clone();
    let count = steps(&app);
    let dialog = crate::menus::invoke(&mut app, &ctx, cmds::CREATE, json!({})).unwrap()["dialog"].as_u64().unwrap();
    let d = app.ui.dialogs.last().unwrap();
    assert!(owns(&d.fields));
    assert_eq!(crate::dialogs::title(d), "Pattern Fill");
    assert_eq!((d.fields["scale"].clone(), d.fields["angle"].clone(), d.fields["link"].clone()), (json!(100.0), json!(0.0), json!(true)));
    assert!(Arc::ptr_eq(&before, &app.session.active().unwrap().doc), "no layer until OK");
    let bricks = pattern_id(&app, "Bricks");
    set(&mut app, dialog, json!({"pattern": bricks, "scale": 200.0, "angle": 45.0, "link": false}));
    let (shown, key) = display_doc(&mut app, 0).unwrap();
    let (again, again_key) = display_doc(&mut app, 0).unwrap();
    assert!(Arc::ptr_eq(&shown, &again) && key == again_key, "the preview is cached");
    assert_eq!(shown.layers.len(), before.layers.len() + 1, "the preview shows the new layer");
    assert!(Arc::ptr_eq(&before, &app.session.active().unwrap().doc), "previewing does not edit");
    assert_eq!(steps(&app), count);
    crate::dialogs::confirm(&mut app, dialog).unwrap();
    assert_eq!(steps(&app), count + 1);
    let st = app.session.active().unwrap();
    let LayerContent::Fill(Fill::Pattern { id, scale, angle, link, .. }) = &st.doc.layer(st.active_layer.unwrap()).unwrap().content else {
        panic!("no pattern fill")
    };
    assert_eq!((id.as_str(), *scale, *angle, *link), (bricks.as_str(), 2.0, 45.0, false));
    assert!(display_doc(&mut app, 0).is_none() && app.pattern_fill_preview.is_none());
    app.run("edit.undo", json!({})).unwrap();
    assert_eq!(*app.session.active().unwrap().doc, *before);
}

#[test]
fn cancelling_a_new_pattern_fill_adds_nothing() {
    let mut app = app();
    let ctx = egui::Context::default();
    let before = app.session.active().unwrap().doc.clone();
    let count = steps(&app);
    let dialog = crate::menus::invoke(&mut app, &ctx, cmds::CREATE, json!({})).unwrap()["dialog"].as_u64().unwrap();
    set(&mut app, dialog, json!({"scale": 300.0}));
    assert!(display_doc(&mut app, 0).is_some());
    crate::dialogs::cancel(&mut app, dialog).unwrap();
    assert!(display_doc(&mut app, 0).is_none());
    assert!(Arc::ptr_eq(&before, &app.session.active().unwrap().doc));
    assert_eq!(steps(&app), count);
    // Explicit params (Actions, CLI, MCP) still create the layer directly.
    let r = crate::menus::invoke(&mut app, &ctx, cmds::CREATE, json!({"pattern": "Grid"})).unwrap();
    assert!(r["layer"].as_u64().is_some());
    // No document: an error, not a dialog or a panic.
    let mut empty = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    assert!(crate::menus::invoke(&mut empty, &ctx, cmds::CREATE, json!({})).is_err());
    assert!(empty.ui.dialogs.is_empty());
}

#[test]
fn content_options_edit_the_layer_and_cancel_restores() {
    let mut app = app();
    let ctx = egui::Context::default();
    let id = app.run(cmds::CREATE, json!({"pattern": "Checkerboard", "scale": 150, "angle": 10})).unwrap()["layer"].as_u64().unwrap();
    let before = app.session.active().unwrap().doc.clone();
    let count = steps(&app);
    let dialog = crate::menus::invoke(&mut app, &ctx, "layer.layerContentOptions", serde_json::Value::Null).unwrap()["dialog"].as_u64().unwrap();
    let checkerboard = pattern_id(&app, "Checkerboard");
    let f = &app.ui.dialog_mut(dialog).unwrap().fields;
    assert!(owns(f));
    assert_eq!((f["pattern"].clone(), f["scale"].clone(), f["angle"].clone()), (json!(checkerboard), json!(150.0), json!(10.0)));
    assert_eq!(f["__params"]["layer"], id);
    // Unchanged: no preview and OK is no History step.
    assert!(display_doc(&mut app, 0).is_none());
    crate::dialogs::confirm(&mut app, dialog).unwrap();
    assert_eq!(steps(&app), count);
    // Changed then cancelled: previewed, but the layer keeps its fill.
    let dialog = open(&mut app).unwrap();
    let dots = pattern_id(&app, "Dots");
    set(&mut app, dialog, json!({"pattern": dots, "phase": [5.0, 5.0]}));
    let (shown, _) = display_doc(&mut app, 0).unwrap();
    assert!(matches!(&shown.layer(LayerId(id)).unwrap().content, LayerContent::Fill(Fill::Pattern { phase, .. }) if *phase == (5.0, 5.0)));
    crate::dialogs::cancel(&mut app, dialog).unwrap();
    assert!(Arc::ptr_eq(&before, &app.session.active().unwrap().doc));
    assert!(display_doc(&mut app, 0).is_none());
    // A stale dialog never edits another document.
    let dialog = open(&mut app).unwrap();
    set(&mut app, dialog, json!({"scale": 50.0}));
    app.run("file.new", json!({"width": 4, "height": 4})).unwrap();
    assert!(display_doc(&mut app, 1).is_none());
    assert!(crate::dialogs::confirm(&mut app, dialog).is_err());
    // Gradient and Solid Color fills keep their own editors.
    app.run("layer.newFillLayer.gradient", json!({})).unwrap();
    assert!(open(&mut app).is_none());
}

fn harness() -> (Harness<'static, PhotocraftApp>, LayerId) {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
    let id = LayerId(s.execute(cmds::CREATE, json!({"pattern": "Checkerboard"})).unwrap()["layer"].as_u64().unwrap());
    s.execute("layer.new.layer", json!({"name": "Other"})).unwrap();
    let mut h = Harness::builder().with_size(vec2(1440.0, 900.0)).with_step_dt(1.0 / 60.0).with_max_steps(64).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        PhotocraftApp::new(s, crate::Services::default())
    });
    let ctx = h.ctx.clone();
    let (r, _) = crate::control::ControlRequest::new("ui.set", json!({"dock": {"collapsed": ["color", "properties", "history", "navigator"]}}));
    crate::control::handle(h.state_mut(), &ctx, &r);
    h.run_steps(8);
    (h, id)
}

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

#[test]
fn double_clicking_the_thumbnail_opens_the_dialog_and_a_picked_pattern_is_applied() {
    let (mut h, id) = harness();
    let r = crate::layer_row_ui::recorded(&h.ctx).into_iter().find(|r| r.layer == id.0).unwrap().row;
    let count = steps(h.state());
    double_click(&mut h, pos2(r.left() + 46.0, r.center().y));
    let d = h.state().ui.dialogs.last().expect("the Pattern Fill dialog opened").clone();
    assert!(owns(&d.fields));
    assert_eq!(d.fields["__params"]["layer"], id.0);
    h.run_steps(3);
    h.get_by_role_and_label(egui::accesskit::Role::Button, "Patterns").click();
    h.run_steps(3);
    h.get_by_label("Bricks").click();
    h.run_steps(3);
    let bricks = pattern_id(h.state(), "Bricks");
    assert_eq!(h.state().ui.dialogs.iter().find(|x| x.id == d.id).unwrap().fields["pattern"], json!(bricks));
    h.get_by_label("OK").click();
    h.run_steps(3);
    assert!(!h.state().ui.dialogs.iter().any(|x| owns(&x.fields)));
    assert_eq!(steps(h.state()), count + 1);
    let content = h.state().session.active().unwrap().doc.layer(id).unwrap().content.clone();
    assert!(matches!(content, LayerContent::Fill(Fill::Pattern { id: ref p, .. }) if *p == bricks));
}
