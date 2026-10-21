use egui::{Key, Modifiers, vec2};
use egui_kittest::Harness;
use serde_json::{Value, json};

use super::*;

fn app() -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
    app.run("file.new", json!({"width": 40, "height": 30, "background": "transparent"})).unwrap();
    app.run("tools.setColors", json!({"foreground": "#ff0000", "background": "#0000ff"})).unwrap();
    app
}

fn px(app: &PhotocraftApp, x: i32, y: i32) -> [f32; 4] {
    let st = app.session.active().unwrap();
    st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().rgba(x, y)
}

fn close(a: [f32; 4], b: [f32; 4]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1.5 / 255.0)
}

/// Open the dialog, set `fields`, press OK.
fn fill(app: &mut PhotocraftApp, fields: Value) -> Result<Value, String> {
    let id = open(app);
    let d = app.ui.dialog_mut(id).unwrap();
    for (k, v) in fields.as_object().unwrap() {
        d.fields.insert(k.clone(), v.clone());
    }
    crate::dialogs::confirm(app, id)
}

#[test]
fn every_contents_option_fills() {
    let mut app = app();
    for (contents, want) in [
        ("foreground", [1.0, 0.0, 0.0, 1.0]),
        ("background", [0.0, 0.0, 1.0, 1.0]),
        ("black", [0.0, 0.0, 0.0, 1.0]),
        ("gray", [128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0, 1.0]),
        ("white", [1.0; 4]),
    ] {
        fill(&mut app, json!({ "contents": contents })).unwrap();
        assert!(close(px(&app, 3, 3), want), "{contents}: {:?}", px(&app, 3, 3));
    }
    fill(&mut app, json!({"contents": "color", "color": "#00ff00"})).unwrap();
    assert!(close(px(&app, 3, 3), [0.0, 1.0, 0.0, 1.0]));
    // History: the default source is the opening state (the empty layer).
    fill(&mut app, json!({"contents": "history"})).unwrap();
    assert_eq!(px(&app, 3, 3)[3], 0.0);
    let pattern = app.session.patterns.items.first().map(|p| p.id.clone()).unwrap();
    fill(&mut app, json!({"contents": "pattern", "pattern": pattern})).unwrap();
    assert!(px(&app, 3, 3)[3] > 0.0);
    // Content-Aware fills a selected hole from around it.
    fill(&mut app, json!({"contents": "color", "color": "#336699"})).unwrap();
    app.run("select.rect", json!({"x": 15, "y": 10, "width": 6, "height": 6})).unwrap();
    fill(&mut app, json!({"contents": "white"})).unwrap();
    fill(&mut app, json!({"contents": "contentAware", "colorAdaptation": false})).unwrap();
    assert!(close(px(&app, 17, 12), [0.2, 0.4, 0.6, 1.0]), "{:?}", px(&app, 17, 12));
    // Each fill is one history step.
    assert_eq!(app.session.active().unwrap().history.past_len(), 12);
}

#[test]
fn mode_opacity_and_preserve_transparency() {
    let mut app = app();
    app.run("select.rect", json!({"x": 0, "y": 0, "width": 20, "height": 30})).unwrap();
    fill(&mut app, json!({"contents": "white"})).unwrap();
    app.run("select.deselect", json!({})).unwrap();
    fill(&mut app, json!({"contents": "black", "opacity": 50, "preserveTransparency": true})).unwrap();
    assert!(close(px(&app, 5, 5), [0.5, 0.5, 0.5, 1.0]), "{:?}", px(&app, 5, 5));
    assert_eq!(px(&app, 30, 5)[3], 0.0, "transparent pixels stay transparent");
    fill(&mut app, json!({"contents": "white", "opacity": 100, "mode": "multiply", "preserveTransparency": true})).unwrap();
    assert!(close(px(&app, 5, 5), [0.5, 0.5, 0.5, 1.0]));
}

#[test]
fn alt_reset_restores_photoshop_defaults_not_last_used() {
    let mut app = app();
    // Remember a non-default choice, then open (which restores it) and Reset.
    fill(&mut app, json!({"contents": "white", "opacity": 40})).unwrap();
    let id = open(&mut app);
    assert_eq!(app.ui.dialog_mut(id).unwrap().fields["opacity"].as_f64(), Some(40.0));
    let mut fields = app.ui.dialog_mut(id).unwrap().fields.clone();
    reset(&app, &mut fields);
    app.ui.dialog_mut(id).unwrap().fields = fields;
    let f = &app.ui.dialog_mut(id).unwrap().fields;
    assert_eq!(f["opacity"].as_f64(), Some(100.0));
    assert_eq!(f["contents"], json!("foreground"));
    assert!(app.ui.dialogs.iter().any(|d| d.id == id));
}

#[test]
fn choices_persist_across_restarts() {
    let mut app = app();
    // Photoshop's defaults the first time.
    let f = fields(&app);
    assert_eq!((f["contents"].as_str(), f["mode"].as_str(), f["opacity"].as_f64()), (Some("foreground"), Some("normal"), Some(100.0)));
    fill(&mut app, json!({"contents": "gray", "mode": "multiply", "opacity": 40, "preserveTransparency": true})).unwrap();
    // A fresh app with the saved preferences opens the dialog with the same choices.
    let saved = app.session.prefs_to_json();
    let mut s = photocraft_engine::Session::new();
    s.load_prefs_json(&saved).unwrap();
    let mut again = PhotocraftApp::new(s, Default::default());
    again.run("file.new", json!({"width": 10, "height": 10})).unwrap();
    let f = fields(&again);
    assert_eq!(f["contents"], json!("gray"));
    assert_eq!(f["mode"], json!("multiply"));
    assert_eq!(f["opacity"].as_f64(), Some(40.0));
    assert_eq!(f["preserveTransparency"], json!(true));
    // A corrupt remembered value falls back to the default.
    again.session.prefs.edit(|p| p.dialogs.insert(COMMAND.into(), json!({"contents": "plaid", "opacity": "lots", "mode": 7})));
    let f = fields(&again);
    assert_eq!((f["contents"].as_str(), f["mode"].as_str(), f["opacity"].as_f64()), (Some("foreground"), Some("normal"), Some(100.0)));
    again.session.prefs.edit(|p| p.dialogs.insert(COMMAND.into(), json!("not an object")));
    assert_eq!(fields(&again)["contents"], json!("foreground"));
}

#[test]
fn bad_fields_fail_gracefully() {
    let mut app = app();
    for f in [json!({"contents": "plaid"}), json!({"mode": "sideways"}), json!({"opacity": "x"}), json!({"contents": "contentAware"})] {
        assert!(fill(&mut app, f.clone()).is_err(), "{f}");
    }
    assert!(app.ui.dialogs.is_empty(), "OK closes the dialog even when the fill fails");
}

fn harness() -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(vec2(1280.0, 800.0)).with_max_steps(64).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 200, "height": 120})).unwrap();
        app
    });
    h.run_steps(6);
    h
}

fn press(h: &mut Harness<'_, PhotocraftApp>, key: Key, m: Modifiers) {
    let c = h.state().last_canvas_rect.center();
    h.hover_at(c);
    h.run_steps(1);
    h.event(egui::Event::ModifiersChanged(m));
    h.event(egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers: m });
    h.event(egui::Event::Key { key, physical_key: None, pressed: false, repeat: false, modifiers: m });
    h.event(egui::Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(3);
}

#[test]
fn shift_f5_and_shift_backspace_open_the_dialog() {
    for key in [Key::F5, Key::Backspace] {
        let mut h = harness();
        press(&mut h, key, Modifiers::SHIFT);
        let open: Vec<_> = h.state().ui.dialogs.iter().filter(|d| owns(&d.fields)).map(|d| d.id).collect();
        assert_eq!(open.len(), 1, "{key:?} opens the Fill dialog");
        // Nothing is filled until OK.
        assert_eq!(h.state().session.active().unwrap().history.past_len(), 0);
        let id = open[0];
        crate::dialogs::confirm(h.state_mut(), id).unwrap();
        assert_eq!(h.state().session.active().unwrap().history.past_len(), 1);
    }
}

#[test]
fn menu_opens_the_dialog_and_params_skip_it() {
    let mut h = harness();
    let ctx = h.ctx.clone();
    let r = crate::menus::invoke(h.state_mut(), &ctx, COMMAND, json!({})).unwrap();
    assert!(r["dialog"].is_u64());
    // With params (agents, actions), the engine fills directly.
    crate::menus::invoke(h.state_mut(), &ctx, COMMAND, json!({"contents": "black"})).unwrap();
    assert_eq!(h.state().session.active().unwrap().history.past_len(), 1);
    // No document: an error, not a dialog.
    let mut empty = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
    assert!(crate::menus::invoke(&mut empty, &ctx, COMMAND, json!({})).is_err());
    assert!(empty.ui.dialogs.is_empty());
}

/// The open Fill dialog's id and the top Color Picker, if one is open.
fn fill_and_picker(h: &Harness<'_, PhotocraftApp>) -> (u64, Option<u64>) {
    let app = h.state();
    let fill = app.ui.dialogs.iter().find(|d| owns(&d.fields)).map(|d| d.id).expect("the Fill dialog is open");
    (fill, crate::color_picker_ui::top(app))
}

fn field(h: &Harness<'_, PhotocraftApp>, id: u64, key: &str) -> Value {
    h.state().ui.dialogs.iter().find(|d| d.id == id).and_then(|d| d.fields.get(key).cloned()).unwrap_or(Value::Null)
}

/// Contents › Color… opens PhotoCraft's Color Picker on the dialog's colour (as Photoshop's Fill
/// does), and its OK sets the Fill colour, not the foreground; Fill then fills with it.
#[test]
fn choosing_color_opens_the_color_picker() {
    use egui_kittest::kittest::Queryable;
    let mut h = harness();
    h.state_mut().run("tools.setColors", json!({"foreground": "#ff0000", "background": "#0000ff"})).unwrap();
    let ctx = h.ctx.clone();
    crate::menus::invoke(h.state_mut(), &ctx, COMMAND, json!({})).unwrap();
    h.run_steps(3);
    assert_eq!(fill_and_picker(&h).1, None, "no picker before Color… is chosen");
    // The Contents dropdown shows its choice as the combo box's value.
    h.get_by_value("Foreground Color").click();
    h.run_steps(3);
    h.get_by_label("Color…").click();
    h.run_steps(4);
    let (fill, picker) = fill_and_picker(&h);
    assert_eq!(field(&h, fill, "contents"), json!("color"));
    let picker = picker.expect("Color… opens the Color Picker");
    assert_eq!(field(&h, picker, "color"), json!("#ff0000"), "on the dialog's colour (the foreground at first)");
    assert_eq!(field(&h, picker, "__label"), json!("Color Picker (Fill Color)"));
    // OK sets the Fill colour; the Fill dialog stays open and the tool colours are untouched.
    h.state_mut().ui.dialog_mut(picker).unwrap().fields.insert("color".into(), json!("#336699"));
    crate::dialogs::confirm(h.state_mut(), picker).unwrap();
    h.run_steps(3);
    assert_eq!(fill_and_picker(&h), (fill, None));
    assert_eq!(field(&h, fill, "color"), json!("#336699"));
    assert_eq!(h.state().session.tools.foreground, [1.0, 0.0, 0.0, 1.0]);
    crate::dialogs::confirm(h.state_mut(), fill).unwrap();
    let st = h.state().session.active().unwrap();
    let c = st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().rgba(5, 5);
    assert!(close(c, [0x33 as f32 / 255.0, 0x66 as f32 / 255.0, 0x99 as f32 / 255.0, 1.0]), "{c:?}");
}

/// The Fill dialog keeps its colour swatch (Photoshop shows none), and clicking it opens the real
/// Color Picker, not a small popup. Cancel keeps Contents on Color… and the colour unchanged.
#[test]
fn the_color_swatch_opens_the_color_picker() {
    use egui_kittest::kittest::Queryable;
    let mut h = harness();
    let ctx = h.ctx.clone();
    let id = crate::menus::invoke(h.state_mut(), &ctx, COMMAND, json!({})).unwrap()["dialog"].as_u64().unwrap();
    {
        let f = &mut h.state_mut().ui.dialog_mut(id).unwrap().fields;
        f.insert("contents".into(), json!("color"));
        f.insert("color".into(), json!("#20a040"));
    }
    h.run_steps(3);
    assert_eq!(fill_and_picker(&h).1, None, "setting the field (automation) opens nothing");
    h.get_by_label("Fill color").click();
    h.run_steps(4);
    let picker = fill_and_picker(&h).1.expect("the swatch opens the Color Picker");
    assert_eq!(field(&h, picker, "color"), json!("#20a040"));
    assert!(!egui::Popup::is_any_open(&h.ctx), "no egui colour popup");
    // Cancel: the Fill dialog is unchanged.
    h.state_mut().ui.dialog_mut(picker).unwrap().fields.insert("color".into(), json!("#000000"));
    h.state_mut().ui.close_dialog(picker);
    h.run_steps(3);
    assert_eq!(fill_and_picker(&h), (id, None));
    assert_eq!(field(&h, id, "contents"), json!("color"));
    assert_eq!(field(&h, id, "color"), json!("#20a040"));
}

#[test]
fn the_pattern_is_picked_from_its_swatches() {
    // The Custom Pattern was a dropdown of names; Photoshop shows the pattern itself and opens a grid
    // of swatches.
    use egui_kittest::kittest::Queryable;
    let mut h = harness();
    {
        let app = h.state_mut();
        app.run("edit.fill", json!({"color": "#cc2200"})).unwrap();
        app.run("edit.definePattern", json!({"name": "Brick", "rect": [0, 0, 8, 8]})).unwrap();
        app.run("edit.fill", json!({"color": "#2266cc"})).unwrap();
        app.run("edit.definePattern", json!({"name": "Sky", "rect": [0, 0, 8, 8]})).unwrap();
    }
    let ctx = h.ctx.clone();
    let id = crate::menus::invoke(h.state_mut(), &ctx, COMMAND, json!({})).unwrap()["dialog"].as_u64().unwrap();
    h.state_mut().ui.dialog_mut(id).unwrap().fields.insert("contents".into(), json!("pattern"));
    h.run_steps(3);
    let pattern = |h: &Harness<'_, PhotocraftApp>| h.state().ui.dialogs.iter().find(|d| d.id == id).unwrap().fields["pattern"].clone();
    let ids: Vec<String> = h.state().session.patterns.items.iter().map(|p| p.id.clone()).collect();
    let sky = ids[h.state().session.patterns.items.iter().position(|p| p.name == "Sky").unwrap()].clone();
    assert_ne!(pattern(&h), json!(sky), "starts on another pattern");
    assert!(h.query_by_label("Sky").is_none(), "the grid opens on a click");
    h.get_by_role_and_label(egui::accesskit::Role::Button, "Patterns").click();
    h.run_steps(3);
    h.get_by_label("Sky").click();
    h.run_steps(3);
    assert_eq!(pattern(&h), json!(sky));
    assert!(h.query_by_label("Brick").is_none(), "the grid closes after a pick");
}
