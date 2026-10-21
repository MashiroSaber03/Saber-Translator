//! Stamp Visible / Stamp Down (#217) through the real key handling: the default shortcuts fire,
//! Edit › Keyboard Shortcuts lists and rebinds them, and ⌥ + the Merge menu items stamp.

use egui::{Event, Key, Modifiers, vec2};
use egui_kittest::Harness;
use serde_json::json;

use crate::PhotocraftApp;

fn harness() -> Harness<'static, PhotocraftApp> {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 120, "height": 80})).unwrap();
    app.run("layer.new.layer", json!({"name": "Paint"})).unwrap();
    app.run("edit.fill", json!({"color": "#3366cc"})).unwrap();
    app.sync_views();
    let mut h = Harness::builder().with_size(vec2(900.0, 600.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            let ctx = ui.ctx().clone();
            if !ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            crate::shortcuts::handle(app, &ctx);
            egui::CentralPanel::default().show(ui, |ui| crate::canvas::document_area(app, ui));
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
    h.run_steps(4);
    h
}

/// ⌘ as the platform reports it (⌘ on the Mac, Ctrl elsewhere).
fn platform(m: Modifiers) -> Modifiers {
    let mut out = Modifiers { alt: m.alt, shift: m.shift, ctrl: m.ctrl, ..Default::default() };
    if m.command {
        out.command = true;
        if cfg!(target_os = "macos") {
            out.mac_cmd = true;
        } else {
            out.ctrl = true;
        }
    }
    out
}

fn tap(h: &mut Harness<'static, PhotocraftApp>, k: Key, m: Modifiers) {
    let m = platform(m);
    h.event(Event::ModifiersChanged(m));
    for pressed in [true, false] {
        h.event(Event::Key { key: k, physical_key: None, pressed, repeat: false, modifiers: m });
        h.run_steps(1);
    }
    h.event(Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(1);
}

fn layers(h: &Harness<'static, PhotocraftApp>) -> usize {
    h.state().session.active().unwrap().doc.walk().len()
}

const STAMP: Modifiers = Modifiers { alt: true, shift: true, command: true, ctrl: false, mac_cmd: false };

#[test]
fn cmd_alt_shift_e_stamps_visible_and_can_be_rebound() {
    let mut h = harness();
    assert_eq!(layers(&h), 2);
    tap(&mut h, Key::E, STAMP);
    assert_eq!(layers(&h), 3, "⌘⌥⇧E stamped the visible layers");
    let top = h.state().session.active().unwrap().doc.layers.last().unwrap().name.clone();
    assert!(top.starts_with("Layer "), "{top}");
    // Rebound to F9: the default no longer fires, the new key does.
    h.state_mut().run("edit.keyboardShortcuts", json!({"set": {"layer.stampVisible": "F9"}})).unwrap();
    tap(&mut h, Key::E, STAMP);
    assert_eq!(layers(&h), 3, "the old key is free");
    tap(&mut h, Key::F9, Modifiers::NONE);
    assert_eq!(layers(&h), 4);
}

#[test]
fn cmd_alt_e_stamps_down() {
    let mut h = harness();
    // Stamp Down into the Background: pixels change there, no layer is added.
    tap(&mut h, Key::E, Modifiers { alt: true, command: true, ..Default::default() });
    assert_eq!(layers(&h), 2);
    assert!(h.state().session.journal.iter().any(|(id, _)| id == "layer.stampDown"));
    let st = h.state().session.active().unwrap();
    let px = st.doc.layers[0].surface().unwrap().rgba(5, 5);
    assert!((px[2] - 0.8).abs() < 0.01, "the copy was merged into the Background: {px:?}");
    // On the Background there is nothing below: a status message, no crash.
    let bg = h.state().session.active().unwrap().doc.layers[0].id.0;
    h.state_mut().run("layer.select", json!({"layer": bg})).unwrap();
    tap(&mut h, Key::E, Modifiers { alt: true, command: true, ..Default::default() });
    assert!(h.state().ui.status.contains("below"), "{}", h.state().ui.status);
}

#[test]
fn keyboard_shortcuts_lists_the_stamps_with_layer() {
    let app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    let items = crate::prefs_ui::shortcut_items(&app);
    let find = |id: &str| items.iter().position(|i| i.0 == id);
    let (Some(v), Some(d)) = (find("layer.stampVisible"), find("layer.stampDown")) else { panic!("listed") };
    assert_eq!(items[v].2, ["Layer"]);
    assert_eq!(items[v].3.as_deref(), Some("Cmd+Alt+Shift+E"));
    assert_eq!(items[d].3.as_deref(), Some("Cmd+Alt+E"));
    // Inside one contiguous Layer section.
    let is_layer = |i: usize| items[i].2.first().map(String::as_str) == Some("Layer");
    let first = (0..items.len()).find(|i| is_layer(*i)).unwrap();
    let last = (0..items.len()).rev().find(|i| is_layer(*i)).unwrap();
    assert!((first..=last).all(is_layer), "one Layer section");
    assert!(v <= last && d <= last);
}

#[test]
fn alt_click_on_merge_items_stamps() {
    use crate::menus::alt_click;
    assert_eq!(alt_click("layer.mergeVisible".into(), true), "layer.stampVisible");
    assert_eq!(alt_click("layer.mergeDown".into(), true), "layer.stampDown");
    assert_eq!(alt_click("layer.mergeLayers".into(), true), "layer.stampDown");
    assert_eq!(alt_click("layer.mergeVisible".into(), false), "layer.mergeVisible");
    assert_eq!(alt_click("file.open".into(), true), "file.open");
}
