//! Photoshop's layer and selection keys reach their commands: ⌘, toggles visibility, ⌘\ targets
//! the mask, / toggles Lock Transparent Pixels, ⌘⌥2 loads luminosity, ⌘⌥⌫ fills from history.

use egui::{Key, Modifiers};
use serde_json::json;

use crate::PhotocraftApp;

fn app() -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 20, "height": 10})).unwrap();
    app.run("layer.new.layer", json!({})).unwrap();
    app
}

fn press(app: &mut PhotocraftApp, key: Key, modifiers: Modifiers) {
    let ctx = egui::Context::default();
    let raw = egui::RawInput {
        events: vec![egui::Event::ModifiersChanged(modifiers), egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers }],
        ..Default::default()
    };
    ctx.begin_pass(raw);
    crate::shortcuts::handle(app, &ctx);
    ctx.end_pass().textures_delta.clear();
}

fn active(app: &PhotocraftApp) -> &photocraft_doc::Layer {
    let st = app.session.active().unwrap();
    st.doc.layer(st.active_layer.unwrap()).unwrap()
}

#[test]
fn command_comma_toggles_the_layers_visibility() {
    let mut app = app();
    press(&mut app, Key::Comma, Modifiers::COMMAND);
    assert!(!active(&app).visible);
    press(&mut app, Key::Comma, Modifiers::COMMAND);
    assert!(active(&app).visible);
}

#[test]
fn slash_toggles_lock_transparent_pixels() {
    let mut app = app();
    press(&mut app, Key::Slash, Modifiers::NONE);
    assert!(active(&app).locks.transparency);
    press(&mut app, Key::Slash, Modifiers::NONE);
    assert!(!active(&app).locks.transparency);
}

#[test]
fn command_backslash_targets_the_layer_mask() {
    let mut app = app();
    app.run("layer.layerMask.revealAll", json!({})).unwrap();
    app.run("channel.target.composite", json!({})).unwrap();
    press(&mut app, Key::Backslash, Modifiers::COMMAND);
    assert_eq!(app.session.active().unwrap().channel_view.target, photocraft_engine::channel_cmds::ChannelTarget::LayerMask);
    assert!(app.ui.mask_target && app.layer_mask_targeted(), "the Layers panel targets the mask too");
    // Ctrl+2 goes back to the layer's pixels, in the Layers panel and for the tools.
    press(&mut app, Key::Num2, Modifiers::COMMAND);
    assert_eq!(app.session.active().unwrap().channel_view.target, photocraft_engine::channel_cmds::ChannelTarget::Composite);
    assert!(!app.ui.mask_target && !app.layer_mask_targeted());
    // The same after targeting the mask by its thumbnail (the panel's own flag).
    app.ui.mask_target = true;
    app.sync_mask_targets();
    press(&mut app, Key::Num2, Modifiers::COMMAND);
    assert!(!app.ui.mask_target && !app.layer_mask_targeted());
}

#[test]
fn command_alt_2_loads_luminosity() {
    let mut app = app();
    press(&mut app, Key::Num2, Modifiers::COMMAND | Modifiers::ALT);
    assert!(app.session.active().unwrap().doc.selection.is_some(), "the white image's luminosity selects it all");
}

#[test]
fn command_alt_backspace_fills_from_history() {
    // The Background exists in the oldest state, so its pixels come back from there.
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 20, "height": 10})).unwrap();
    app.run("tools.setColors", json!({"foreground": "#ff0000"})).unwrap();
    app.run("edit.fillForeground", json!({})).unwrap();
    assert_eq!(active(&app).surface().unwrap().pixel(3, 3)[1], 0.0, "red");
    press(&mut app, Key::Backspace, Modifiers::COMMAND | Modifiers::ALT);
    assert_eq!(active(&app).surface().unwrap().pixel(3, 3)[1], 1.0, "white again");
}

/// ⌥-click a layer's thumbnail: select it and fit its content on screen.
#[test]
fn alt_click_on_a_layer_thumbnail_fits_the_layer() {
    use egui::{Event, PointerButton, pos2, vec2};
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 400, "height": 300})).unwrap();
    let small = s.execute("layer.new.layer", json!({"name": "Small"})).unwrap()["layer"].as_u64().unwrap();
    s.execute("select.rect", json!({"x": 300, "y": 200, "width": 20, "height": 10})).unwrap();
    s.execute("edit.fill", json!({"contents": "black"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    s.execute("layer.new.layer", json!({"name": "Top"})).unwrap();
    let mut h = egui_kittest::Harness::builder().with_size(vec2(1440.0, 900.0)).with_step_dt(1.0 / 60.0).with_max_steps(64).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        PhotocraftApp::new(s, crate::Services::default())
    });
    let ctx = h.ctx.clone();
    let (r, _) = crate::control::ControlRequest::new("ui.set", json!({"dock": {"collapsed": ["color", "properties", "history", "navigator"]}}));
    crate::control::handle(h.state_mut(), &ctx, &r);
    h.run_steps(8);
    let before = h.state().ui.views[0].clone();
    let row = crate::layer_row_ui::recorded(&h.ctx).into_iter().find(|r| r.layer == small).unwrap().row;
    let at = pos2(row.left() + 46.0, row.center().y);
    h.hover_at(at);
    h.event(Event::ModifiersChanged(Modifiers::ALT));
    h.run_steps(1);
    h.event(Event::PointerButton { pos: at, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::ALT });
    h.run_steps(1);
    h.event(Event::PointerButton { pos: at, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::ALT });
    h.run_steps(1);
    h.event(Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(3);
    let app = h.state();
    assert_eq!(app.session.active().unwrap().active_layer, Some(photocraft_doc::LayerId(small)));
    let v = &app.ui.views[0];
    assert_eq!(v.center, [310.0, 205.0], "centred on the layer's content");
    assert!(v.zoom > before.zoom * 4.0, "zoomed in on 20 × 10 px: {} -> {}", before.zoom, v.zoom);
}
