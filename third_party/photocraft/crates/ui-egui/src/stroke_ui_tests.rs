use serde_json::json;

use super::*;

fn app() -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
    app.run("file.new", json!({"width": 40, "height": 30, "background": "transparent"})).unwrap();
    app.run("tools.setColors", json!({"foreground": "#ff0000", "background": "#0000ff"})).unwrap();
    app
}

fn pixel(app: &PhotocraftApp, x: i32, y: i32) -> [f32; 4] {
    let st = app.session.active().unwrap();
    st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().rgba(x, y)
}

#[test]
fn menu_and_canvas_invocations_open_options_but_explicit_params_run_directly() {
    let mut app = app();
    let ctx = egui::Context::default();
    let opened = crate::menus::invoke(&mut app, &ctx, COMMAND, json!({})).unwrap();
    assert!(opened["dialog"].is_u64());
    assert_eq!(app.ui.dialogs.len(), 1);
    assert!(owns(&app.ui.dialogs[0].fields));
    assert_eq!(app.session.active().unwrap().history.past_len(), 0);
    app.ui.close_dialog(opened["dialog"].as_u64().unwrap());
    let id = crate::dialogs::open_command_dialog(&mut app, COMMAND, "Stroke…");
    assert!(owns(&app.ui.dialogs.last().unwrap().fields));
    app.ui.close_dialog(id);

    app.run("select.rect", json!({"x": 10, "y": 10, "width": 10, "height": 10})).unwrap();
    crate::menus::invoke(&mut app, &ctx, COMMAND, json!({"width": 3, "location": "inside", "color": "#00ff00"})).unwrap();
    assert!(app.ui.dialogs.is_empty(), "explicit params must not show a modal dialog");
    assert!(pixel(&app, 10, 15)[1] > 0.98);

    let mut empty = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
    assert!(crate::menus::invoke(&mut empty, &ctx, COMMAND, json!({})).is_err());
    assert!(empty.ui.dialogs.is_empty());
}

#[test]
fn stroke_choices_apply_on_confirm_and_cancel_does_not_modify_document() {
    let mut app = app();
    app.run("select.rect", json!({"x": 10, "y": 10, "width": 10, "height": 10})).unwrap();
    let initial = app.session.active().unwrap().history.past_len();
    let cancelled = open(&mut app);
    let dialog = app.ui.dialog_mut(cancelled).unwrap();
    dialog.fields.insert("color".into(), json!("#00ff00"));
    dialog.fields.insert("width".into(), json!(4.0));
    app.ui.close_dialog(cancelled);
    assert_eq!(app.session.active().unwrap().history.past_len(), initial);
    assert_eq!(pixel(&app, 10, 15)[3], 0.0);

    let applied = open(&mut app);
    let d = app.ui.dialog_mut(applied).unwrap();
    d.fields.insert("color".into(), json!("#00ff00"));
    d.fields.insert("width".into(), json!(4.0));
    d.fields.insert("location".into(), json!("inside"));
    d.fields.insert("mode".into(), json!("normal"));
    crate::dialogs::confirm(&mut app, applied).unwrap();
    assert!(pixel(&app, 10, 15)[1] > 0.98, "inside edge receives selected green stroke");
    assert_eq!(pixel(&app, 15, 15)[3], 0.0, "unselected interior unchanged");
    assert_eq!(pixel(&app, 9, 15)[3], 0.0, "outside unchanged for Inside location");
    assert_eq!(app.session.active().unwrap().history.past_len(), initial + 1);
}

#[test]
fn preferences_remember_all_choices_and_reject_corrupt_values() {
    let mut app = app();
    let defaults = fields(&app);
    assert_eq!(defaults["width"], 1.0);
    assert_eq!(defaults["location"], "center");
    assert_eq!(defaults["mode"], "normal");
    assert_eq!(defaults["opacity"], 100.0);
    assert_eq!(defaults["preserveTransparency"], false);

    app.run("select.rect", json!({"x": 5, "y": 5, "width": 15, "height": 12})).unwrap();
    let id = open(&mut app);
    let d = app.ui.dialog_mut(id).unwrap();
    for (key, value) in json!({
        "width": 5,
        "color": "#112233",
        "location": "outside",
        "mode": "multiply",
        "opacity": 42,
        "preserveTransparency": true
    })
    .as_object()
    .unwrap()
    {
        d.fields.insert(key.clone(), value.clone());
    }
    crate::dialogs::confirm(&mut app, id).unwrap();
    let saved = app.session.prefs_to_json();
    let mut restored = photocraft_engine::Session::new();
    restored.load_prefs_json(&saved).unwrap();
    let mut again = PhotocraftApp::new(restored, Default::default());
    again.run("file.new", json!({"width": 40, "height": 30, "background": "transparent"})).unwrap();
    again.run("tools.setColors", json!({"foreground": "#ff0000"})).unwrap();
    let remembered = fields(&again);
    assert_eq!(remembered["width"], 5);
    assert_eq!(remembered["color"], "#112233");
    assert_eq!(remembered["location"], "outside");
    assert_eq!(remembered["mode"], "multiply");
    assert_eq!(remembered["opacity"], 42);
    assert_eq!(remembered["preserveTransparency"], true);

    again.session.prefs.edit(|p| {
        p.dialogs.insert(
            COMMAND.into(),
            json!({
                "width": -5, "color": "#nan", "location": "sideways",
                "mode": "unrecognized", "opacity": "a lot", "preserveTransparency": 7
            }),
        )
    });
    let fallback = fields(&again);
    for (key, value) in [
        ("width", json!(1.0)),
        ("color", json!("#ff0000")),
        ("location", json!("center")),
        ("mode", json!("normal")),
        ("opacity", json!(100.0)),
        ("preserveTransparency", json!(false)),
    ] {
        assert_eq!(fallback[key], value, "invalid persisted value {key} must be discarded");
    }
}

#[test]
fn command_params_do_not_leak_private_dialog_fields() {
    let app = app();
    let fields = fields(&app);
    let out = params(&fields);
    assert_eq!(out.as_object().unwrap().len(), 6);
    assert!(out.get("__stroke").is_none());
    assert!(out.get("__label").is_none());
    assert_eq!(out["location"], "center");
    assert_eq!(out["color"], "#ff0000");
}

#[test]
fn stroke_dialog_labels_are_translated_in_every_catalog() {
    const LABELS: &[&str] =
        &["Stroke", "Width:", "Color:", "Location", "Location:", "Inside", "Center", "Outside", "Blending", "Mode:", "Opacity:", "Preserve Transparency"];
    for lang in crate::i18n::Lang::all().filter(|lang| lang.complete_menus()) {
        for label in LABELS {
            assert!(crate::i18n::has(lang, label), "{} is missing Stroke dialog label: {label}", lang.code());
        }
    }
}

/// The Stroke dialog's colour swatch opens PhotoCraft's Color Picker on the stroke colour (as in
/// Photoshop), not egui's compact popup; OK sets the stroke colour, not the foreground, and the
/// stroke is drawn in it.
#[test]
fn the_color_swatch_opens_the_color_picker() {
    use egui_kittest::Harness;
    use egui_kittest::kittest::Queryable;
    let mut h = Harness::builder().with_size(egui::vec2(1280.0, 800.0)).with_max_steps(64).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        app()
    });
    h.run_steps(6);
    h.state_mut().run("select.rect", json!({"x": 10, "y": 10, "width": 10, "height": 10})).unwrap();
    let ctx = h.ctx.clone();
    let id = crate::menus::invoke(h.state_mut(), &ctx, COMMAND, json!({})).unwrap()["dialog"].as_u64().unwrap();
    h.state_mut().ui.dialog_mut(id).unwrap().fields.insert("color".into(), json!("#20a040"));
    h.run_steps(3);
    assert_eq!(crate::color_picker_ui::top(h.state()), None);
    h.get_by_label("Stroke color").click();
    h.run_steps(4);
    let picker = crate::color_picker_ui::top(h.state()).expect("the swatch opens the Color Picker");
    let field = |h: &Harness<'_, PhotocraftApp>, d: u64, k: &str| h.state().ui.dialogs.iter().find(|x| x.id == d).and_then(|x| x.fields.get(k).cloned());
    assert_eq!(field(&h, picker, "color"), Some(json!("#20a040")));
    assert_eq!(field(&h, picker, "__label"), Some(json!("Color Picker (Stroke Color)")));
    assert!(!egui::Popup::is_any_open(&h.ctx), "no egui colour popup");
    h.state_mut().ui.dialog_mut(picker).unwrap().fields.insert("color".into(), json!("#0000ff"));
    crate::dialogs::confirm(h.state_mut(), picker).unwrap();
    h.run_steps(3);
    assert_eq!(crate::color_picker_ui::top(h.state()), None);
    assert_eq!(field(&h, id, "color"), Some(json!("#0000ff")));
    assert_eq!(h.state().session.tools.foreground, [1.0, 0.0, 0.0, 1.0], "the foreground is untouched");
    crate::dialogs::confirm(h.state_mut(), id).unwrap();
    let p = pixel(h.state(), 10, 15);
    assert!(p[2] > 0.98 && p[0] < 0.02, "stroked in the picked colour: {p:?}");
}
