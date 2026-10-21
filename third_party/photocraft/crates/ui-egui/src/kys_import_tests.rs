//! Photoshop `.kys` sets import into Edit › Keyboard Shortcuts (#1163): rows match our
//! commands by label, one shortcut per command, with Photoshop's menu order breaking ties.

use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use serde_json::{Value, json};

use super::{KysCommand, parse, plan};
use crate::PhotocraftApp;
use crate::file_dialog::FileDialogAnswer;

/// A cut-down set in Photoshop's own layout: BOM, a dynamic (filter) row, static rows with
/// their ids, a row with two shortcuts, an entity, scripts, and the tool keys.
const SAMPLE: &str = "\u{feff}<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<photoshop-keyboard-shortcuts version=\"4\" filename=\"$$$/FileName/Presets/KeyboardCustomization/texcuts=texcuts\" modified=\"0\" multi-undo=\"2\">
\t<command kind=\"dynamic\" name=\"Script_6F17BFA7-EFC8-40EA-B850-7B95ED8EA713\">
\t\t<shortcut>Control+Opt+Cmd+4</shortcut>
\t</command>
\t<command kind=\"dynamic\" name=\"Gaussian Blur...\">
\t\t<shortcut>Control+Opt+Cmd+G</shortcut>
\t</command>
\t<command kind=\"static\" name=\"Undo\" id=\"101\">
\t\t<shortcut>Cmd+Z</shortcut>
\t\t<shortcut>F1</shortcut>
\t</command>
\t<command kind=\"static\" name=\"Zoom In\" id=\"1004\">
\t\t<shortcut>Cmd++</shortcut>
\t\t<shortcut>Cmd+=</shortcut>
\t</command>
\t<command kind=\"static\" name=\"Levels...\" id=\"1801\">
\t\t<shortcut>Cmd+L</shortcut>
\t</command>
\t<command kind=\"static\" name=\"All\" id=\"1017\">
\t\t<shortcut>Cmd+A</shortcut>
\t</command>
\t<command kind=\"static\" name=\"Layer Via Copy\" id=\"2970\">
\t\t<shortcut>Cmd+J</shortcut>
\t</command>
\t<command kind=\"static\" name=\"Black &amp; White...\" id=\"1824\">
\t\t<shortcut>Opt+Shift+Cmd+B</shortcut>
\t</command>
\t<command kind=\"static\" name=\"General...\" id=\"2311\">
\t\t<shortcut>Cmd+K</shortcut>
\t</command>
\t<command kind=\"static\" name=\"Last Filter\" id=\"1019\">
\t\t<shortcut>Cmd+F</shortcut>
\t</command>
\t<command kind=\"static\" name=\"Hide Photoshop\" id=\"7001\">
\t\t<shortcut>Cmd+H</shortcut>
\t</command>
\t<tool name=\"Move Tool\" type=\"1\" key=\"1819113074\">V</tool>
\t<tool name=\"Default Foreground/Background Colors\" type=\"2\"></tool>
</photoshop-keyboard-shortcuts>
";

fn app() -> PhotocraftApp {
    PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default())
}

#[test]
fn parses_rows_shortcuts_entities_and_tool_keys() {
    let set = parse(SAMPLE).unwrap();
    assert_eq!(set.commands.len(), 11);
    assert_eq!(set.commands[2], KysCommand { name: "Undo".into(), shortcuts: vec!["Cmd+Z".into(), "F1".into()] });
    assert_eq!(set.commands[7].name, "Black & White...");
    assert_eq!(set.tool_keys, 1, "the empty Default Colors row carries no key");
}

#[test]
fn rejects_other_xml_and_text() {
    assert!(parse("<svg/>").unwrap_err().contains("not a keyboard shortcut set (.kys)"));
    assert!(parse("Cmd+K\tPreferences").is_err());
    assert_eq!(parse("<photoshop-keyboard-shortcuts/>").unwrap(), super::KysSet::default());
}

#[test]
fn rows_match_our_commands_by_label_with_photoshop_tie_breaks() {
    let app = app();
    let p = plan(&app, &parse(SAMPLE).unwrap());
    // Photoshop's "..." is our "…"; the dynamic (filter) row is a filter here.
    assert_eq!(p.set.get("filter.blur.gaussianBlur").map(String::as_str), Some("Cmd+Ctrl+Alt+G"));
    // Levels… is Image › Adjustments and Layer › New Adjustment Layer: the one whose default is
    // the imported key wins. All is Select, Edit › Purge, View › Show…: Select › All has the key.
    assert_eq!(p.set.get("image.adjustments.levels").map(String::as_str), Some("Cmd+L"));
    assert!(!p.set.contains_key("layer.newAdjustmentLayer.levels"));
    assert_eq!(p.set.get("select.all").map(String::as_str), Some("Cmd+A"));
    assert!(!p.set.contains_key("edit.purge.all"));
    // Undo is Edit › Undo, not Edit › Purge › Undo; only its first shortcut counts.
    assert_eq!(p.set.get("edit.undo").map(String::as_str), Some("Cmd+Z"));
    // Zoom In lists Cmd++ first, but Cmd+= is our default and is among its keys: it stays.
    assert_eq!(p.set.get("view.zoomIn").map(String::as_str), Some("Cmd+="));
    assert_eq!(p.alternates, 2);
    // Case does not matter; an entity decodes; the menu's Settings… is the engine's General….
    assert_eq!(p.set.get("layer.new.layerViaCopy").map(String::as_str), Some("Cmd+J"));
    assert_eq!(p.set.get("image.adjustments.blackWhite").map(String::as_str), Some("Cmd+Alt+Shift+B"));
    assert_eq!(p.set.get("edit.preferences.general").map(String::as_str), Some("Cmd+K"));
    assert_eq!(p.set.get("filter.lastFilter").map(String::as_str), Some("Cmd+F"));
    // Scripts and the macOS application menu have no command here.
    assert_eq!(p.unknown, vec!["Script_6F17BFA7-EFC8-40EA-B850-7B95ED8EA713".to_string(), "Hide Photoshop".to_string()]);
    assert!(p.unreadable.is_empty());
    assert_eq!(p.set.len(), 9);
}

#[test]
fn an_unreadable_shortcut_is_reported_not_imported() {
    let app = app();
    let set = super::KysSet { name: None, commands: vec![KysCommand { name: "Undo".into(), shortcuts: vec!["Cmd+Shift+Ctrl".into()] }], tool_keys: 0 };
    let p = plan(&app, &set);
    assert!(p.set.is_empty());
    assert_eq!(p.unreadable, vec!["Undo".to_string()]);
}

/// The dialog's Import Shortcuts… button: the chosen file fills the overrides (a key
/// equal to the default clears its override), reports what it did, and OK applies the set like
/// hand-made changes, taking a moved key from its old owner.
#[test]
fn import_button_fills_the_dialog_and_ok_applies_the_set() {
    let answers = vec![Some(FileDialogAnswer::Contents("/sets/texcuts.kys".into(), SAMPLE.as_bytes().to_vec()))];
    let services = crate::Services { file_dialog: Some(crate::file_dialog::fake(answers).0), ..Default::default() };
    let mut h = Harness::builder().with_size(egui::vec2(1100.0, 760.0)).with_max_steps(64).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        PhotocraftApp::new(photocraft_engine::Session::new(), services)
    });
    let ctx = h.ctx.clone();
    // A hand-made override the file agrees with the default on: the import clears it.
    h.state_mut().run("edit.keyboardShortcuts", json!({"set": {"edit.undo": "Cmd+Shift+Z"}})).unwrap();
    let id = crate::menus::invoke(h.state_mut(), &ctx, "edit.keyboardShortcuts", json!({})).unwrap()["dialog"].as_u64().unwrap();
    // A short list keeps the button row on screen.
    h.state_mut().ui.dialog_mut(id).unwrap().fields.insert("filter".into(), json!("gaussian"));
    h.state_mut().ui.dialog_mut(id).unwrap().fields.insert("messageWarning".into(), json!(true));
    h.run_steps(2);
    h.get_by_label("Import Shortcuts…").click();
    h.run_steps(3);
    let field = |h: &Harness<'_, PhotocraftApp>, name: &str| -> Value {
        h.state().ui.dialogs.iter().find(|d| d.id == id).unwrap().fields.get(name).cloned().unwrap_or(Value::Null)
    };
    assert_eq!(h.state().ui.status, "", "the import failed");
    let overrides = field(&h, "overrides");
    assert_eq!(overrides["filter.blur.gaussianBlur"], json!("Cmd+Ctrl+Alt+G"));
    assert!(overrides.get("edit.preferences.general").is_none(), "Cmd+K is Preferences' default: {overrides}");
    assert_eq!(overrides["filter.lastFilter"], json!("Cmd+F"));
    assert!(overrides.get("edit.undo").is_none(), "Cmd+Z is the default again: {overrides}");
    assert!(overrides.get("select.all").is_none(), "already the default: {overrides}");
    assert!(overrides.get("view.zoomIn").is_none(), "Cmd+= is among the keys, so the default stays: {overrides}");
    let message = field(&h, "message");
    assert_eq!(message, json!("Imported 9 shortcuts from texcuts.kys. Click OK to keep them. 2 commands are not in PhotoCraft."));
    assert_eq!(field(&h, "capture"), json!(false));
    assert_eq!(field(&h, "messageWarning"), json!(false), "an import clears the earlier shortcut conflict warning");
    // Nothing is applied until OK.
    assert_eq!(crate::shortcuts::effective_shortcut(h.state(), "edit.undo", Some("Cmd+Z")).as_deref(), Some("Cmd+Shift+Z"));
    crate::dialogs::confirm(h.state_mut(), id).unwrap();
    let app = h.state();
    assert_eq!(crate::shortcuts::effective_shortcut(app, "edit.undo", Some("Cmd+Z")).as_deref(), Some("Cmd+Z"));
    assert_eq!(crate::shortcuts::effective_shortcut(app, "filter.blur.gaussianBlur", None).as_deref(), Some("Cmd+Ctrl+Alt+G"));
    assert_eq!(crate::shortcuts::effective_shortcut(app, "edit.preferences.general", Some("Cmd+K")).as_deref(), Some("Cmd+K"));
    // Cmd+F went to Last Filter, so Search (Cmd+F by default) lost its key.
    assert_eq!(crate::shortcuts::effective_shortcut(app, "edit.search", Some("Cmd+F")), None);
    assert_eq!(crate::shortcuts::effective_shortcut(app, "filter.lastFilter", Some("Cmd+Alt+F")).as_deref(), Some("Cmd+F"));
}

/// Opening or dropping a `.kys` opens the dialog with the set loaded, so it can be reviewed.
#[test]
fn an_opened_kys_file_opens_the_dialog_filled() {
    let mut app = app();
    assert!(app.ui.dialogs.is_empty());
    let r = crate::preset_files_ui::open(&mut app, "texcuts.kys", SAMPLE.as_bytes()).unwrap();
    r.unwrap();
    let d = app.ui.dialogs.iter().find(|d| d.fields.get("__prefsui").and_then(Value::as_str) == Some("shortcuts")).unwrap();
    assert_eq!(d.fields["tab"], json!(0));
    assert_eq!(d.fields["overrides"]["filter.blur.gaussianBlur"], json!("Cmd+Ctrl+Alt+G"));
    // A second file lands in the same dialog, on top of the first.
    let one = "<photoshop-keyboard-shortcuts><command kind=\"static\" name=\"Copy\"><shortcut>F3</shortcut></command></photoshop-keyboard-shortcuts>";
    crate::preset_files_ui::open(&mut app, "more.kys", one.as_bytes()).unwrap().unwrap();
    assert_eq!(app.ui.dialogs.len(), 1);
    let d = &app.ui.dialogs[0];
    assert_eq!(d.fields["overrides"]["edit.copy"], json!("F3"));
    assert_eq!(d.fields["overrides"]["filter.blur.gaussianBlur"], json!("Cmd+Ctrl+Alt+G"));
    assert_eq!(d.fields["message"], json!("Imported 1 shortcuts from more.kys. Click OK to keep them."));
    let bad = crate::preset_files_ui::open(&mut app, "notes.kys", b"hello").unwrap().unwrap_err();
    assert!(bad.contains("not a keyboard shortcut set (.kys)"), "{bad}");
}

type Store = std::sync::Arc<std::sync::Mutex<Option<String>>>;

/// An app beside a Photoshop whose live set is `set`; `saved_prefs` is this app's own saved
/// preferences file, and the store receives what it writes back.
fn with_photoshop(set: &'static str, saved_prefs: Option<&'static str>) -> (PhotocraftApp, Store) {
    let store: Store = Default::default();
    let written = store.clone();
    let services = crate::Services {
        photoshop_shortcuts: Some(Box::new(move || {
            Some(("/Users/me/Library/Preferences/Adobe Photoshop 2026 Settings/Keyboard Shortcuts.psp".into(), set.into()))
        })),
        load_prefs: Some(Box::new(move || saved_prefs.map(str::to_string))),
        save_prefs: Some(Box::new(move |text: &str| {
            *written.lock().unwrap() = Some(text.to_string());
            Ok(())
        })),
        ..Default::default()
    };
    (PhotocraftApp::new(photocraft_engine::Session::new(), services), store)
}

/// First launch beside Photoshop: its live set is offered, not applied. Import applies it,
/// a notice says so, and the preference remembers it so the next launch does not ask again.
#[test]
fn first_launch_offers_photoshops_live_set_and_import_applies_it() {
    let (mut app, store) = with_photoshop(SAMPLE, None);
    assert!(app.session.prefs().shortcuts.is_empty(), "nothing applied unasked");
    assert!(!app.session.prefs().dialogs.contains_key(super::IMPORTED_PREF), "not answered yet");
    let offer = app.ui.kys_offer.clone().unwrap();
    assert_eq!(offer.set, "texcuts");
    assert!(offer.changes > 0);
    assert_eq!(crate::control::inspect(&app, &egui::Context::default())["kysOffer"]["set"], "texcuts");
    super::answer_offer(&mut app, true);
    assert!(app.ui.kys_offer.is_none());
    assert!(app.ui.dialogs.is_empty(), "applied without the dialog");
    assert_eq!(crate::shortcuts::effective_shortcut(&app, "filter.blur.gaussianBlur", None).as_deref(), Some("Cmd+Ctrl+Alt+G"));
    assert_eq!(crate::shortcuts::effective_shortcut(&app, "filter.lastFilter", Some("Cmd+Alt+F")).as_deref(), Some("Cmd+F"));
    assert_eq!(crate::shortcuts::effective_shortcut(&app, "edit.search", Some("Cmd+F")), None, "taken by Last Filter");
    let answer = &app.session.prefs().dialogs[super::IMPORTED_PREF];
    assert_eq!(answer["imported"], json!(true));
    assert_eq!(answer["source"].as_str().map(|s| s.ends_with("Keyboard Shortcuts.psp")), Some(true));
    let notice = app.ui.notices.last().unwrap();
    assert_eq!(notice.title, "Keyboard shortcuts imported");
    let line = notice.text(&notice.lines[0]);
    assert!(line.starts_with("Your keyboard shortcut set texcuts is in use here: "), "{line}");
    assert!(!app.ui.status_error, "{}", app.ui.status);
    // The answer moved the preferences past the loaded revision, so the next frame saves it.
    crate::prefs_ui::tick(&mut app, &egui::Context::default());
    let saved = store.lock().unwrap().clone().unwrap();
    assert!(saved.contains("Cmd+Ctrl+Alt+G") && saved.contains(super::IMPORTED_PREF), "{saved}");
    // Reset, then a second launch: the preference says it was answered.
    app.run("edit.keyboardShortcuts", json!({"reset": true})).unwrap();
    super::offer_import(&mut app);
    assert!(app.ui.kys_offer.is_none());
}

/// Don't Import changes nothing and is remembered too.
#[test]
fn dont_import_keeps_the_defaults_and_is_not_asked_again() {
    let (mut app, _) = with_photoshop(SAMPLE, None);
    super::answer_offer(&mut app, false);
    assert!(app.session.prefs().shortcuts.is_empty());
    assert!(app.ui.notices.is_empty());
    assert_eq!(app.session.prefs().dialogs[super::IMPORTED_PREF]["imported"], json!(false));
    super::offer_import(&mut app);
    assert!(app.ui.kys_offer.is_none());
}

/// The prompt on screen: it names the set, and its Import button applies it.
#[test]
fn the_prompt_shows_the_set_and_its_import_button_applies_it() {
    let mut h = Harness::builder().with_size(egui::vec2(1100.0, 760.0)).with_max_steps(64).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        with_photoshop(SAMPLE, None).0
    });
    h.run_steps(2);
    h.get_by_label("Import keyboard shortcuts?");
    assert!(h.query_by_label_contains("keyboard shortcut set texcuts").is_some());
    h.get_by_label("Import").click();
    h.run_steps(2);
    assert!(h.query_by_label("Import keyboard shortcuts?").is_none());
    assert_eq!(crate::shortcuts::effective_shortcut(h.state(), "filter.blur.gaussianBlur", None).as_deref(), Some("Cmd+Ctrl+Alt+G"));
}

/// Shortcuts the user already changed are theirs: nothing is offered, and the launch is
/// remembered, so the set is never offered over them later either.
#[test]
fn a_user_with_their_own_shortcuts_is_not_asked() {
    let saved = r#"{"shortcuts": {"edit.undo": "F1"}}"#;
    let (app, _) = with_photoshop(SAMPLE, Some(saved));
    assert_eq!(crate::shortcuts::effective_shortcut(&app, "edit.undo", Some("Cmd+Z")).as_deref(), Some("F1"));
    assert!(app.ui.kys_offer.is_none());
    assert!(app.session.prefs().dialogs.contains_key(super::IMPORTED_PREF));
    // A set that is just Photoshop's defaults changes nothing, so it is not offered.
    let defaults = "<photoshop-keyboard-shortcuts><command kind=\"static\" name=\"Undo\"><shortcut>Cmd+Z</shortcut></command></photoshop-keyboard-shortcuts>";
    let (quiet, _) = with_photoshop(defaults, None);
    assert!(quiet.ui.kys_offer.is_none());
    assert!(quiet.session.prefs().dialogs.contains_key(super::IMPORTED_PREF));
    // A file that is not a set is reported, not offered, and not retried.
    let (bad, _) = with_photoshop("binary junk", None);
    assert!(bad.ui.kys_offer.is_none());
    assert!(bad.ui.status_error, "{}", bad.ui.status);
    assert!(bad.ui.status.starts_with("The keyboard shortcut set was not imported: not a keyboard shortcut set (.kys)"), "{}", bad.ui.status);
    assert!(bad.session.prefs().dialogs.contains_key(super::IMPORTED_PREF));
}

/// A file far larger than any set is refused before it is read as text.
#[test]
fn an_oversized_file_is_refused() {
    let mut app = app();
    let big = vec![b' '; super::MAX_BYTES + 1];
    let e = super::open_bytes(&mut app, "huge.kys", &big).unwrap_err();
    assert!(e.contains("larger than 4 MiB"), "{e}");
    assert!(app.ui.dialogs.is_empty());
}
