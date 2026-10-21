//! The Layers panel New Layer button: a click makes a layer at once, ⌥-click opens the New Layer
//! dialog first, as in Photoshop. Its shortcut is the same pair the other way round: ⇧⌘N asks
//! first, ⌥⇧⌘N makes the layer at once (#4794).

use egui::Modifiers;
use egui_kittest::kittest::Queryable;
use serde_json::json;

use crate::PhotocraftApp;

fn panel() -> egui_kittest::Harness<'static, PhotocraftApp> {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
    let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(1200.0, 800.0)).with_max_steps(64).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        PhotocraftApp::new(s, crate::Services::default())
    });
    h.run_steps(4);
    h
}

fn layers(h: &egui_kittest::Harness<'static, PhotocraftApp>) -> usize {
    h.state().session.active().unwrap().doc.layers.len()
}

#[test]
fn click_makes_a_layer_without_asking() {
    let mut h = panel();
    h.get_by_label("Create a new layer").click();
    h.run_steps(4);
    assert_eq!(layers(&h), 2);
    assert!(!h.state().ui.dialogs.iter().any(|d| crate::new_layer_ui::owns(&d.fields)));
}

#[test]
fn alt_click_opens_the_new_layer_dialog() {
    let mut h = panel();
    h.get_by_label("Create a new layer").click_modifiers(Modifiers::ALT);
    h.run_steps(4);
    assert_eq!(layers(&h), 1, "nothing made until OK");
    let id = h.state().ui.dialogs.iter().find(|d| crate::new_layer_ui::owns(&d.fields)).map(|d| d.id).expect("the New Layer dialog");
    crate::dialogs::confirm(h.state_mut(), id).unwrap();
    assert_eq!(layers(&h), 2);
}

fn dialog_open(h: &egui_kittest::Harness<'static, PhotocraftApp>) -> bool {
    h.state().ui.dialogs.iter().any(|d| crate::new_layer_ui::owns(&d.fields))
}

#[test]
fn shortcut_asks_first_and_alt_shortcut_makes_a_layer_at_once() {
    let mut h = panel();
    h.key_press_modifiers(Modifiers::COMMAND | Modifiers::SHIFT, egui::Key::N);
    h.run_steps(4);
    assert_eq!(layers(&h), 1, "⇧⌘N asks first");
    assert!(dialog_open(&h), "the New Layer dialog");

    let mut h = panel();
    h.key_press_modifiers(Modifiers::COMMAND | Modifiers::SHIFT | Modifiers::ALT, egui::Key::N);
    h.run_steps(4);
    assert_eq!(layers(&h), 2, "⌥⇧⌘N makes the layer");
    assert!(!dialog_open(&h));
    h.key_press_modifiers(Modifiers::COMMAND | Modifiers::SHIFT | Modifiers::ALT, egui::Key::N);
    h.run_steps(4);
    assert_eq!(layers(&h), 3, "once per press");
}

#[test]
fn alt_shortcut_follows_a_custom_new_layer_shortcut_and_yields_to_a_bound_command() {
    // Rebound to F2: ⌥F2 makes the layer at once.
    let mut h = panel();
    h.state_mut().session.prefs.edit(|p| p.shortcuts.insert("layer.new.layer".into(), "F2".into()));
    h.key_press_modifiers(Modifiers::ALT, egui::Key::F2);
    h.run_steps(4);
    assert_eq!(layers(&h), 2);
    assert!(!dialog_open(&h));
    // A command bound to ⌥⇧⌘N itself keeps the key.
    let mut h = panel();
    h.state_mut().session.prefs.edit(|p| p.shortcuts.insert("layer.new.layerViaCopy".into(), "Cmd+Alt+Shift+N".into()));
    h.key_press_modifiers(Modifiers::COMMAND | Modifiers::SHIFT | Modifiers::ALT, egui::Key::N);
    h.run_steps(4);
    assert_eq!(layers(&h), 2);
    assert!(!dialog_open(&h));
    let doc = &h.state().session.active().unwrap().doc;
    let top = doc.layers.last().unwrap().surface().unwrap();
    assert!(!top.content_bounds().is_empty(), "Layer via Copy ran (the white background copied), not New Layer");
}
