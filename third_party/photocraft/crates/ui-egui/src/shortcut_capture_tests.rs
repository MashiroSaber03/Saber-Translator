//! #292: Edit › Keyboard Shortcuts records the key pressed with the modifiers, not the
//! modifier key itself (egui reports Ctrl, Shift, Alt and ⌘ as key presses too), so Ctrl+F and
//! any other non-system combination can be bound; a shortcut already in use is reported.

use egui::{Event, Key, Modifiers};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use serde_json::{Value, json};

use crate::PhotocraftApp;

fn key(h: &mut Harness<'_, PhotocraftApp>, key: Key, pressed: bool, modifiers: Modifiers) {
    h.event(Event::Key { key, physical_key: None, pressed, repeat: false, modifiers });
}

/// The Keyboard Shortcuts dialog with `command` selected, filtered to it.
fn dialog(ppp: f32, command: &str, filter: &str) -> (Harness<'static, PhotocraftApp>, u64) {
    let mut h = Harness::builder().with_size(egui::vec2(1100.0, 760.0)).with_pixels_per_point(ppp).with_max_steps(64).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default())
    });
    let ctx = h.ctx.clone();
    let id = crate::menus::invoke(h.state_mut(), &ctx, "edit.keyboardShortcuts", json!({})).unwrap()["dialog"].as_u64().unwrap();
    let f = &mut h.state_mut().ui.dialog_mut(id).unwrap().fields;
    f.insert("filter".into(), json!(filter));
    f.insert("selected".into(), json!(command));
    h.run_steps(4);
    (h, id)
}

fn field(h: &Harness<'_, PhotocraftApp>, id: u64, name: &str) -> Value {
    h.state().ui.dialogs.iter().find(|d| d.id == id).unwrap().fields.get(name).cloned().unwrap_or(Value::Null)
}

#[test]
fn toolbar_shortcut_can_be_captured_and_restored_on_restart() {
    let (mut h, id) = dialog(1.0, "tools.select.pencil", "Pencil Tool");
    h.get_by_role_and_label(egui::accesskit::Role::Button, "B").click();
    h.run_steps(2);
    h.key_press(Key::F6);
    h.run_steps(2);
    assert_eq!(field(&h, id, "overrides")["tools.select.pencil"], json!("F6"));
    crate::dialogs::confirm(h.state_mut(), id).unwrap();
    h.run_steps(2);
    h.key_press(Key::F6);
    h.run_steps(2);
    assert_eq!(h.state().ui.tool, crate::state::Tool::Pencil);

    let saved = h.state().session.prefs_to_json();
    let mut session = photocraft_engine::Session::new();
    session.load_prefs_json(&saved).unwrap();
    let mut restarted = PhotocraftApp::new(session, crate::Services::default());
    let ctx = egui::Context::default();
    ctx.begin_pass(egui::RawInput {
        events: vec![Event::Key { key: Key::F6, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE }],
        ..Default::default()
    });
    crate::shortcuts::handle(&mut restarted, &ctx);
    ctx.end_pass().textures_delta.clear();
    assert_eq!(restarted.ui.tool, crate::state::Tool::Pencil);
}

#[test]
fn toolbar_shortcut_overrides_remove_old_keys_and_skip_removed_group_members() {
    use crate::state::Tool;
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    let ctx = egui::Context::default();
    let press = |app: &mut PhotocraftApp, key, modifiers| {
        ctx.begin_pass(egui::RawInput {
            // A tap: press and release, so the next press isn't a held key's repeat.
            events: vec![
                Event::ModifiersChanged(modifiers),
                Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers },
                Event::Key { key, physical_key: None, pressed: false, repeat: false, modifiers },
            ],
            ..Default::default()
        });
        crate::shortcuts::handle(app, &ctx);
        ctx.end_pass().textures_delta.clear();
    };
    app.run("edit.keyboardShortcuts", json!({"set": {"tools.select.move": "F6", "tools.select.pencil": ""}, "allowUnknown": true})).unwrap();
    app.ui.tool = Tool::Zoom;
    press(&mut app, Key::V, Modifiers::NONE);
    assert_eq!(app.ui.tool, Tool::Artboard, "the old key only selects unchanged tools");
    press(&mut app, Key::F6, Modifiers::NONE);
    assert_eq!(app.ui.tool, Tool::Move);

    app.ui.tool = Tool::Pencil;
    press(&mut app, Key::V, Modifiers::NONE);
    press(&mut app, Key::B, Modifiers::NONE);
    assert_eq!(app.ui.tool, Tool::Brush, "a removed tool cannot return through group memory");
    press(&mut app, Key::B, Modifiers::SHIFT);
    assert_eq!(app.ui.tool, Tool::MixerBrush, "Shift cycling skips a removed tool");

    app.run("edit.keyboardShortcuts", json!({"reset": true})).unwrap();
    app.ui.tool = Tool::Brush;
    press(&mut app, Key::B, Modifiers::SHIFT);
    assert_eq!(app.ui.tool, Tool::Pencil, "reset restores the default group");
    app.session.edit_prefs(|p| p.tools.use_shift_key_for_tool_switch = false);
    press(&mut app, Key::B, Modifiers::NONE);
    assert_eq!(app.ui.tool, Tool::MixerBrush, "plain-key cycling still follows the preference");
}

#[test]
fn toolbar_shortcuts_participate_in_conflict_resolution() {
    let (mut h, id) = dialog(1.0, "edit.search", "search…");
    capture(&mut h, &crate::shortcuts::pretty("Cmd+F"), Key::ControlLeft, Modifiers::COMMAND, Key::B);
    // Remove Ctrl from the captured binding: a plain B belongs to the Brush group by default.
    h.state_mut().ui.dialog_mut(id).unwrap().fields["overrides"]["edit.search"] = json!("B");
    crate::dialogs::confirm(h.state_mut(), id).unwrap();
    for tool in ["brush", "pencil", "mixerBrush", "colorReplacement"] {
        assert_eq!(h.state().session.prefs().shortcuts.get(&format!("tools.select.{tool}")).map(String::as_str), Some(""));
    }
    let ctx = h.ctx.clone();
    h.state_mut().ui.tool = crate::state::Tool::Move;
    key(&mut h, Key::B, true, Modifiers::NONE);
    h.run_steps(2);
    assert!(h.state().ui.palette_open, "B now runs Search");
    assert_eq!(h.state().ui.tool, crate::state::Tool::Move);
    assert_eq!(crate::shortcut_dispatch::take_log(&ctx).last().map(|i| i.0.as_str()), Some("edit.search"));
}

/// Click the selected command's shortcut button, then press `modifier` and `k` like a keyboard:
/// the modifier key goes down first (its own key event), then the key with the modifier held.
fn capture(h: &mut Harness<'_, PhotocraftApp>, button: &str, modifier: Key, held: Modifiers, k: Key) {
    h.get_by_label(button).click();
    h.run_steps(2);
    key(h, modifier, true, held);
    h.run_steps(1);
    key(h, k, true, held);
    h.run_steps(1);
    key(h, k, false, held);
    key(h, modifier, false, Modifiers::NONE);
    h.run_steps(2);
}

#[test]
fn ctrl_k_is_recorded_as_ctrl_k_not_the_control_key() {
    for ppp in [1.0, 2.0] {
        let (mut h, id) = dialog(ppp, "edit.search", "search…");
        // Search's default is ⌘F (Photoshop); ⌘K is Preferences, so this also reports a conflict.
        let current = crate::shortcuts::pretty("Cmd+F");
        // On Windows and Linux the Ctrl key is the command modifier.
        capture(&mut h, &current, Key::ControlLeft, Modifiers::COMMAND, Key::K);
        assert_eq!(field(&h, id, "overrides")["edit.search"], json!("Cmd+K"), "@{ppp}x");
        assert_eq!(field(&h, id, "capture"), json!(false));
        let message = field(&h, id, "message");
        assert!(!message.as_str().unwrap().contains("ontrol"), "@{ppp}x: {message}");
        // The row shows the new shortcut.
        h.get_by_label(&crate::shortcuts::pretty("Cmd+K"));
        h.render().ok();
    }
}

#[test]
fn every_modifier_key_waits_for_the_real_key() {
    for (modifier, held) in [
        (Key::ControlRight, Modifiers::COMMAND),
        (Key::ShiftLeft, Modifiers::SHIFT),
        (Key::AltRight, Modifiers::ALT),
        (Key::SuperLeft, Modifiers::COMMAND),
        (Key::ShiftRight, Modifiers::COMMAND | Modifiers::SHIFT),
    ] {
        let (mut h, id) = dialog(1.0, "edit.search", "search…");
        h.get_by_label(&crate::shortcuts::pretty("Cmd+F")).click();
        h.run_steps(2);
        key(&mut h, modifier, true, held);
        h.run_steps(2);
        assert_eq!(field(&h, id, "capture"), json!(true), "{modifier:?} alone keeps capturing");
        assert!(field(&h, id, "overrides").get("edit.search").is_none(), "{modifier:?} alone assigns nothing");
        key(&mut h, Key::J, true, held);
        h.run_steps(2);
        let want = crate::prefs_ui::shortcut_text(Key::J, held).unwrap();
        assert_eq!(field(&h, id, "overrides")["edit.search"], json!(want), "{modifier:?}");
    }
    for k in [Key::ControlLeft, Key::ShiftLeft, Key::AltLeft, Key::SuperRight] {
        assert_eq!(crate::prefs_ui::shortcut_text(k, Modifiers::COMMAND), None, "{k:?}");
    }
}

#[test]
fn a_shortcut_in_use_is_reported_and_moved_on_ok() {
    // ⌘J is Layer › New › Layer via Copy: giving it to Search warns, and OK moves it.
    let (mut h, id) = dialog(1.0, "edit.search", "search…");
    capture(&mut h, &crate::shortcuts::pretty("Cmd+F"), Key::ControlLeft, Modifiers::COMMAND, Key::J);
    let message = field(&h, id, "message");
    assert!(message.as_str().unwrap().contains("already in use"), "{message}");
    crate::dialogs::confirm(h.state_mut(), id).unwrap();
    h.run_steps(2);
    let app = h.state();
    assert_eq!(app.session.prefs().shortcut("edit.search", Some("Cmd+F")), Some("Cmd+J"));
    let bound: Vec<String> =
        crate::shortcut_dispatch::bindings(app).into_iter().filter(|(_, sc)| *sc == crate::shortcuts::parse("Cmd+J").unwrap()).map(|(id, _)| id).collect();
    assert_eq!(bound, ["edit.search"], "one owner for ⌘J");
}

#[test]
fn a_function_key_moved_off_a_panel_toggle_runs_its_new_command() {
    // #1272: F5 given to Select › Modify › Feather… is taken from Window › Brush Settings, whose
    // row in the dialog is Photoshop's `window.panel.brushSettings` while the key ran the shell's
    // `window.toggle.brushSettings`.
    let (mut h, id) = dialog(1.0, "select.modify.feather", "feather");
    capture(&mut h, &crate::shortcuts::pretty("Shift+F6"), Key::ShiftLeft, Modifiers::NONE, Key::F5);
    let message = field(&h, id, "message");
    assert!(message.as_str().unwrap().contains("already in use"), "{message}");
    crate::dialogs::confirm(h.state_mut(), id).unwrap();
    h.run_steps(2);
    let f5 = crate::shortcuts::parse("F5").unwrap();
    let bound: Vec<String> = crate::shortcut_dispatch::bindings(h.state()).into_iter().filter(|(_, sc)| *sc == f5).map(|(id, _)| id).collect();
    assert_eq!(bound, ["select.modify.feather"], "one owner for F5");
    // Pressing F5 opens Feather, not Brush Settings.
    let ctx = h.ctx.clone();
    crate::shortcut_dispatch::set_dry_run(&ctx, true);
    key(&mut h, Key::F5, true, Modifiers::NONE);
    h.run_steps(1);
    key(&mut h, Key::F5, false, Modifiers::NONE);
    h.run_steps(1);
    let log = crate::shortcut_dispatch::take_log(&ctx);
    assert!(log.iter().any(|(id, _)| id == "select.modify.feather"), "{log:?}");
    assert!(!h.state().ui.panels.brush_settings);
}

#[test]
fn a_user_shortcut_wins_over_a_default_it_was_not_moved_from() {
    // A key assigned by hand (here through the prefs file, so no row was cleared) still reaches
    // the command it was given to rather than the built-in owner.
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("edit.keyboardShortcuts", json!({"set": {"select.modify.feather": "F7"}, "allowUnknown": true, "removeConflicts": false})).unwrap();
    let f7 = crate::shortcuts::parse("F7").unwrap();
    let bound: Vec<String> = crate::shortcut_dispatch::bindings(&app).into_iter().filter(|(_, sc)| *sc == f7).map(|(id, _)| id).collect();
    assert_eq!(bound, ["select.modify.feather"]);
}
