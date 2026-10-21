use serde_json::json;

use super::*;

fn app() -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
    app.run("file.new", json!({"width": 16, "height": 16})).unwrap();
    app.run("select.rect", json!({"x": 2, "y": 2, "width": 8, "height": 8})).unwrap();
    app.run("select.saveSelection", json!({})).unwrap();
    app
}

fn channel(app: &PhotocraftApp) -> photocraft_doc::AlphaChannel {
    app.session.active().unwrap().doc.channels[0].clone()
}

#[test]
fn saved_selection_becomes_a_spot_channel_with_its_ink_and_solidity() {
    let mut app = app();
    assert!(channel(&app).spot.is_none());
    let id = open(&mut app, &json!(0)).unwrap();
    let f = &app.ui.dialogs.last().unwrap().fields;
    assert!(owns(f));
    assert_eq!(f["indicates"], json!("masked"));
    assert_eq!(f["name"], json!(channel(&app).name));
    let d = app.ui.dialog_mut(id).unwrap();
    d.fields.insert("name".into(), json!("PMS 185"));
    d.fields.insert("indicates".into(), json!("spot"));
    d.fields.insert("color".into(), json!("#ff0000"));
    d.fields.insert("opacity".into(), json!(40.0));
    crate::dialogs::confirm(&mut app, id).unwrap();
    assert!(app.ui.dialogs.is_empty());
    let ch = channel(&app);
    assert_eq!(ch.name, "PMS 185");
    let (ink, solidity) = ch.spot.expect("a spot channel");
    assert_eq!(ink.to_rgb(), [1.0, 0.0, 0.0]);
    assert!((solidity - 0.4).abs() < 1e-6);

    // Reopened, the dialog shows the spot settings; switching back makes it an alpha channel.
    let id = open(&mut app, &json!(0)).unwrap();
    let f = &app.ui.dialog_mut(id).unwrap().fields;
    assert_eq!((f["indicates"].clone(), f["color"].clone(), f["opacity"].clone()), (json!("spot"), json!("#ff0000"), json!(40.0)));
    app.ui.dialog_mut(id).unwrap().fields.insert("indicates".into(), json!("selected"));
    crate::dialogs::confirm(&mut app, id).unwrap();
    let ch = channel(&app);
    assert!(ch.spot.is_none());
    assert_eq!(ch.indicates, ColorIndicates::SelectedAreas);
}

#[test]
fn cancel_leaves_the_channel_alone() {
    let mut app = app();
    let before = app.session.active().unwrap().history.past_len();
    let id = open(&mut app, &json!(0)).unwrap();
    app.ui.dialog_mut(id).unwrap().fields.insert("indicates".into(), json!("spot"));
    app.ui.close_dialog(id);
    assert!(channel(&app).spot.is_none());
    assert_eq!(app.session.active().unwrap().history.past_len(), before);
}

#[test]
fn spot_params_send_solidity_and_alpha_params_send_opacity() {
    let mut f = Map::new();
    f.insert(MARK.into(), json!(true));
    f.insert("channel".into(), json!(0));
    f.insert("indicates".into(), json!("spot"));
    f.insert("opacity".into(), json!(25.0));
    assert_eq!(params(&f), json!({"channel": 0, "indicates": "spot", "solidity": 25.0}));
    f.insert("indicates".into(), json!("masked"));
    assert_eq!(params(&f), json!({"channel": 0, "indicates": "masked", "opacity": 25.0}));
}

#[test]
fn menu_opens_the_dialog_on_the_targeted_channel_and_refuses_otherwise() {
    let ctx = egui::Context::default();
    let mut empty = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
    assert!(crate::menus::invoke(&mut empty, &ctx, COMMAND, json!({})).is_err());

    let mut app = app();
    // The composite is targeted: nothing to edit.
    assert!(crate::menus::invoke(&mut app, &ctx, COMMAND, json!({})).is_err());
    assert!(app.ui.dialogs.is_empty());
    app.run("channel.target", json!({"channel": 0})).unwrap();
    let r = crate::menus::invoke(&mut app, &ctx, COMMAND, json!({})).unwrap();
    assert!(r["dialog"].is_u64());
    assert_eq!(app.ui.dialogs.last().unwrap().fields["channel"], json!(0));
    // Explicit params run the engine directly, no dialog.
    app.ui.dialogs.clear();
    crate::menus::invoke(&mut app, &ctx, COMMAND, json!({"channel": 0, "indicates": "spot"})).unwrap();
    assert!(app.ui.dialogs.is_empty());
    assert!(channel(&app).spot.is_some());
}

#[test]
fn quick_mask_options_have_no_name_or_spot_and_bad_channels_open_nothing() {
    let mut app = app();
    let f = fields(&app, &json!("quickMask")).unwrap();
    assert!(!f.contains_key("name"));
    assert_eq!(f["__label"], json!("Quick Mask Options…"));
    assert!(fields(&app, &json!(7)).is_none());
    assert!(fields(&app, &json!("nope")).is_none());
    assert!(open(&mut app, &json!(-1)).is_none());
    assert!(app.ui.dialogs.is_empty());
}

#[test]
fn dialog_body_renders_for_alpha_spot_and_quick_mask() {
    let mut app = app();
    app.run("channel.options", json!({"channel": 0, "indicates": "spot"})).unwrap();
    let ctx = egui::Context::default();
    crate::theme::install_fonts(&ctx);
    for channel in [json!(0), json!("quickMask")] {
        let mut f = fields(&app, &channel).unwrap();
        ctx.run_ui(Default::default(), |ui| body(ui, &mut f)).textures_delta.clear();
        assert!(owns(&f));
    }
}

#[test]
fn new_spot_channel_dialog_creates_the_ink_from_the_selection_and_remembers_it() {
    let mut app = app();
    app.run("select.rect", json!({"x": 2, "y": 2, "width": 8, "height": 8})).unwrap();
    let id = open_new_spot(&mut app).unwrap();
    let f = &app.ui.dialogs.last().unwrap().fields;
    assert!(owns_new_spot(f) && !owns(f));
    assert_eq!((f["name"].clone(), f["color"].clone(), f["solidity"].clone()), (json!("Spot Color 1"), json!("#ff0000"), json!(0.0)));
    let d = app.ui.dialog_mut(id).unwrap();
    d.fields.insert("name".into(), json!("PMS 300"));
    d.fields.insert("color".into(), json!("#0050ff"));
    d.fields.insert("solidity".into(), json!(60.0));
    crate::dialogs::confirm(&mut app, id).unwrap();
    let doc = &app.session.active().unwrap().doc;
    let ch = doc.channels.last().unwrap();
    assert_eq!(ch.name, "PMS 300");
    let (ink, solidity) = ch.spot.expect("a spot channel");
    assert_eq!(ink.to_rgba8()[..3], [0x00, 0x50, 0xff]);
    assert!((solidity - 0.6).abs() < 1e-6);
    // Inked where the selection was, empty outside it.
    let (inside, outside) = (ch.surface.rgba(4, 4)[0], ch.surface.rgba(14, 14)[0]);
    assert!(inside > 0.99 && outside < 0.01, "{inside} {outside}");

    // The next one gets a fresh name and the last ink and solidity.
    let f = new_spot_fields(&app).unwrap();
    assert_eq!((f["name"].clone(), f["color"].clone(), f["solidity"].clone()), (json!("Spot Color 1"), json!("#0050ff"), json!(60.0)));
}

#[test]
fn new_spot_channel_cancel_menu_routing_and_no_document() {
    let ctx = egui::Context::default();
    let mut empty = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
    assert!(open_new_spot(&mut empty).is_none());
    assert!(crate::menus::invoke(&mut empty, &ctx, NEW_SPOT, json!({})).is_err());

    let mut app = app();
    let n = app.session.active().unwrap().doc.channels.len();
    let r = crate::menus::invoke(&mut app, &ctx, NEW_SPOT, json!({})).unwrap();
    let id = r["dialog"].as_u64().unwrap();
    assert!(owns_new_spot(&app.ui.dialog_mut(id).unwrap().fields));
    app.ui.close_dialog(id);
    assert_eq!(app.session.active().unwrap().doc.channels.len(), n, "Cancel adds nothing");
    // Explicit params run the engine directly, no dialog.
    crate::menus::invoke(&mut app, &ctx, NEW_SPOT, json!({"name": "Direct"})).unwrap();
    assert!(app.ui.dialogs.is_empty());
    assert_eq!(app.session.active().unwrap().doc.channels.last().unwrap().name, "Direct");
}

#[test]
fn new_spot_body_renders_and_bad_saved_prefs_fall_back() {
    let mut app = app();
    app.session.prefs.edit(|p| p.dialogs.insert(NEW_SPOT.into(), json!({"color": "nope", "solidity": 900})));
    let mut f = new_spot_fields(&app).unwrap();
    assert_eq!((f["color"].clone(), f["solidity"].clone()), (json!("#ff0000"), json!(0.0)));
    let ctx = egui::Context::default();
    crate::theme::install_fonts(&ctx);
    ctx.run_ui(Default::default(), |ui| new_spot_body(ui, &mut f)).textures_delta.clear();
    assert!(owns_new_spot(&f));
}
