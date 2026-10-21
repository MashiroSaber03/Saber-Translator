//! `edit.keyboardShortcuts {"importKys": <xml>}`: agents, the CLI and MCP import a Photoshop
//! `.kys` set without the app's dialog (#1163).
use photocraft_engine::Session;
use photocraft_engine::prefs::normalize_shortcut;
use serde_json::json;

const SET: &str = "\u{feff}<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<photoshop-keyboard-shortcuts version=\"4\" filename=\"$$$/FileName/Presets/KeyboardCustomization/texcuts=texcuts\">
\t<command kind=\"dynamic\" name=\"Gaussian Blur...\"><shortcut>Control+Opt+Cmd+G</shortcut></command>
\t<command kind=\"static\" name=\"Undo\" id=\"101\"><shortcut>Cmd+Z</shortcut></command>
\t<command kind=\"static\" name=\"Layer Via Copy\" id=\"2970\"><shortcut>F7</shortcut></command>
\t<command kind=\"static\" name=\"Hide Photoshop\" id=\"7001\"><shortcut>Cmd+H</shortcut></command>
\t<tool name=\"Move Tool\" type=\"1\">V</tool>
</photoshop-keyboard-shortcuts>
";

#[test]
fn import_kys_applies_the_set_and_reports_it() {
    let mut s = Session::new();
    let r = s.execute("edit.keyboardShortcuts", json!({"importKys": SET})).unwrap();
    let import = &r["import"];
    assert_eq!(import["name"], "texcuts");
    assert_eq!(import["imported"], 3);
    assert_eq!(import["unknown"], json!(["Hide Photoshop"]));
    assert_eq!(import["toolKeys"], 1);
    let prefs = s.prefs();
    assert_eq!(prefs.shortcuts.get("filter.blur.gaussianBlur").and_then(|k| normalize_shortcut(k)), normalize_shortcut("Ctrl+Alt+Cmd+G"));
    assert_eq!(prefs.shortcuts.get("layer.new.layerViaCopy").map(String::as_str), Some("F7"));
    // Undo's key is its default, so it gets no override.
    assert!(!prefs.shortcuts.contains_key("edit.undo"), "{:?}", prefs.shortcuts);
}

#[test]
fn import_kys_keys_given_in_set_win_and_bad_input_is_an_error() {
    let mut s = Session::new();
    s.execute("edit.keyboardShortcuts", json!({"importKys": SET, "set": {"layer.new.layerViaCopy": "F8"}})).unwrap();
    assert_eq!(s.prefs().shortcuts.get("layer.new.layerViaCopy").map(String::as_str), Some("F8"));
    let before = s.prefs().shortcuts.clone();
    assert!(s.execute("edit.keyboardShortcuts", json!({"importKys": "<other/>"})).is_err());
    assert!(s.execute("edit.keyboardShortcuts", json!({"importKys": 7})).is_err());
    assert_eq!(s.prefs().shortcuts, before, "a refused set changes nothing");
}
