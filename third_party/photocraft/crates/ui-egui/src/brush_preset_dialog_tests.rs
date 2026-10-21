use egui::vec2;
use egui_kittest::Harness;
use egui_kittest::kittest::{NodeT, Queryable};

use super::*;

fn harness() -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(vec2(1440.0, 900.0)).with_max_steps(64).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default())
    });
    h.state_mut().run("file.new", json!({"width": 400, "height": 300})).unwrap();
    h.state_mut().ui.tool = crate::state::Tool::Brush;
    h.run_steps(4);
    h
}

/// The open New Brush Preset dialog, if there is one (and never two).
fn dialog(h: &Harness<'_, PhotocraftApp>) -> Option<u64> {
    let open: Vec<u64> = h.state().ui.dialogs.iter().filter(|d| owns(&d.fields)).map(|d| d.id).collect();
    assert!(open.len() <= 1, "one dialog at a time: {open:?}");
    open.first().copied()
}

fn field<'a>(h: &'a Harness<'_, PhotocraftApp>, id: u64, key: &str) -> &'a Value {
    &h.state().ui.dialogs.iter().find(|d| d.id == id).unwrap().fields[key]
}

/// Every way of making a preset opens the dialog, and Cancel saves nothing.
#[test]
fn every_new_preset_button_opens_the_dialog_and_cancel_saves_nothing() {
    let mut h = harness();
    let before = h.state().session.tools.presets.clone();
    let brush = h.state().session.tools.brush.clone();
    let mut opened = 0;
    let mut cancel = |h: &mut Harness<'_, PhotocraftApp>| {
        dialog(h).expect("the dialog opened");
        h.run_steps(2);
        h.get_by_label("Cancel").click();
        h.run_steps(3);
        assert!(dialog(h).is_none(), "Cancel closes it");
        assert_eq!(h.state().session.tools.presets, before, "and saves nothing");
        assert_eq!(h.state().session.tools.brush, brush, "nor touches the brush");
        opened += 1;
    };
    // The picker's [+] and its gear menu.
    h.state_mut().ui.brush_picker = Some([300.0, 120.0]);
    h.run_steps(3);
    h.get_by_label("New Brush Preset").click();
    h.run_steps(3);
    cancel(&mut h);
    h.state_mut().ui.brush_picker = Some([300.0, 120.0]);
    h.run_steps(3);
    h.get_by_label("Brush Preset Options").click();
    h.run_steps(2);
    h.get_by_label("New Brush Preset…").click();
    h.run_steps(3);
    cancel(&mut h);
    // Brush Settings' [+], then the Brushes tab's.
    h.state_mut().ui.brush_picker = None;
    h.state_mut().ui.panels.brush_settings = true;
    h.state_mut().ui.brush_tab = 0;
    h.run_steps(3);
    h.get_by_label("Create new brush from these settings").click();
    h.run_steps(3);
    cancel(&mut h);
    h.state_mut().ui.brush_tab = 1;
    h.run_steps(3);
    h.get_by_label("Create new brush from the current settings").click();
    h.run_steps(3);
    cancel(&mut h);
    assert_eq!(opened, 4);
}

/// The dialog suggests the current brush's name and the next free number, never a name in use; OK
/// saves under the typed name with the boxes as set, and the next one suggests the next number.
#[test]
fn ok_saves_under_the_name_with_the_boxes_as_set() {
    let mut h = harness();
    h.state_mut().run("tools.setBrush", json!({"preset": "Chalk"})).unwrap();
    let id = open(h.state_mut(), None);
    h.run_steps(3);
    assert_eq!(field(&h, id, "name"), "Chalk 1");
    assert_eq!((field(&h, id, "captureSize"), field(&h, id, "toolSettings"), field(&h, id, "color")), (&json!(true), &json!(false), &json!(false)));
    // Include Color waits for Include Tool Settings.
    assert!(h.get_by_label("Include Color").accesskit_node().is_disabled(), "Include Color is greyed out");
    h.get_by_label("Include Tool Settings").click();
    h.run_steps(2);
    assert!(!h.get_by_label("Include Color").accesskit_node().is_disabled());
    h.get_by_label("Include Color").click();
    h.get_by_label("Capture Brush Size in Preset").click();
    h.run_steps(2);
    h.get_by_label("OK").click();
    h.run_steps(3);
    assert!(dialog(&h).is_none());
    let p = h.state().session.tools.presets.iter().find(|p| p.name == "Chalk 1").expect("saved");
    assert_eq!(p.captures, photocraft_paint::PresetCaptures { size: false, tool_settings: true, color: true });
    assert!(h.state().session.tools.presets.iter().any(|p| p.name == "Chalk"), "the preset it came from is still there");
    assert_eq!(h.state().session.tools.current_preset.as_deref(), Some("Chalk 1"));

    // Again: the next number, and a name in use greys OK out.
    let id = open(h.state_mut(), None);
    h.run_steps(3);
    assert_eq!(field(&h, id, "name"), "Chalk 1 1", "named after the current brush, now Chalk 1");
    h.state_mut().ui.dialog_mut(id).unwrap().fields.insert("name".into(), json!("chalk"));
    h.run_steps(2);
    assert!(h.get_by_label("OK").accesskit_node().is_disabled(), "Chalk exists");
    assert!(h.query_by_label("A brush preset with this name already exists").is_some());
    assert!(crate::dialogs::confirm(h.state_mut(), id).is_err(), "automation can't overwrite it either");
}

/// Include Color unchecked with it ticked earlier: the preset doesn't take the colour.
#[test]
fn include_color_needs_include_tool_settings() {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    let id = open(&mut app, None);
    let f = &mut app.ui.dialog_mut(id).unwrap().fields;
    f.insert("name".into(), json!("Mine"));
    f.insert("color".into(), json!(true));
    crate::dialogs::confirm(&mut app, id).unwrap();
    let p = app.session.tools.presets.iter().find(|p| p.name == "Mine").unwrap();
    assert_eq!(p.captures, photocraft_paint::PresetCaptures { size: true, tool_settings: false, color: false });
}

/// The dialog opens on the name, all of it selected: typing replaces it, and Enter is OK.
#[test]
fn typing_replaces_the_suggested_name_and_enter_is_ok() {
    let mut h = harness();
    let id = open(h.state_mut(), None);
    h.run_steps(4);
    h.event(egui::Event::Text("Inky".into()));
    h.run_steps(2);
    assert_eq!(field(&h, id, "name"), "Inky");
    h.key_press(egui::Key::Enter);
    h.run_steps(3);
    assert!(dialog(&h).is_none(), "Enter is OK");
    assert_eq!(h.state().session.tools.current_preset.as_deref(), Some("Inky"));
}
