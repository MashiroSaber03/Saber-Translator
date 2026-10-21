//! Regression tests for issue #719: a bulk `edit.keyboardShortcuts` set that
//! assigns the same key to two commands must not silently unbind both; the
//! clash inside the call must be reported in `conflicts`.
use photocraft_engine::Session;
use serde_json::{Value, json};

fn shortcut_of(v: &Value, id: &str) -> Option<String> {
    v["commands"].as_array()?.iter().find(|c| c["id"].as_str() == Some(id))?.get("shortcut").and_then(|s| s.as_str()).map(str::to_string)
}

#[test]
fn bulk_set_same_key_twice_reports_conflict_and_keeps_bindings() {
    let mut s = Session::new();
    // edit.undo's default is already Cmd+Z; the bulk set also gives Cmd+Z to
    // edit.fill. On main the conflict pass made each entry blank the other:
    // both ended up unbound and conflicts was empty.
    let r = s.execute("edit.keyboardShortcuts", json!({"set": {"edit.undo": "Cmd+Z", "edit.fill": "Cmd+Z"}, "list": true})).unwrap();
    let fill = shortcut_of(&r, "edit.fill").unwrap_or_default();
    let undo = shortcut_of(&r, "edit.undo").unwrap_or_default();
    assert_eq!(fill, "Cmd+Z", "edit.fill must keep the key it was just given (empty on main)");
    assert_eq!(undo, "Cmd+Z", "edit.undo must keep the key it was just given (empty on main)");
    let conflicts = r["conflicts"].as_array().unwrap();
    let clash = conflicts.iter().any(|c| c["shortcut"].as_str() == Some("Cmd+Z") && c["commands"].as_array().is_some_and(|ids| ids.len() == 2));
    assert!(clash, "the in-call clash must appear in conflicts, got {}", r["conflicts"]);
}

#[test]
fn single_set_still_steals_from_old_owner() {
    let mut s = Session::new();
    // Control: the documented single-assignment behaviour must be unchanged —
    // moving Cmd+Z to edit.fill takes it away from edit.undo.
    let r = s.execute("edit.keyboardShortcuts", json!({"set": {"edit.fill": "Cmd+Z"}, "list": true})).unwrap();
    assert_eq!(shortcut_of(&r, "edit.fill").as_deref(), Some("Cmd+Z"));
    let undo = shortcut_of(&r, "edit.undo").unwrap_or_default();
    assert_ne!(undo, "Cmd+Z", "edit.undo must lose the key when it moves to edit.fill");
}
