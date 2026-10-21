//! Regression for #1547: a successful save back to the document's own PSD or PSB file must
//! record the current revision as saved, so the session stops reporting `dirty` (as `.pcraft`
//! already did). A flat export stays a copy and leaves the dirty state and path alone.

use photocraft_automation::Headless;
use serde_json::json;

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "pc-layered-save-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The issue's scenario: create `name`, reopen it, edit, then save with no path. The save writes
/// the document's own file and must clear `dirty`.
fn pathless_save_clears_dirty(dir: &std::path::Path, name: &str) {
    let path = dir.join(name);
    let mut headless = Headless::trusted_local();
    headless.handle("doc.new", json!({"width": 8, "height": 8, "background": "white"})).unwrap();
    headless.handle("doc.save", json!({"path": path.to_str().unwrap()})).unwrap();
    assert!(path.is_file(), "{name} was written");

    // Reopen the saved file and edit it, so the pathless save below writes back over its own file.
    headless.handle("doc.open", json!({"path": path.to_str().unwrap()})).unwrap();
    headless.handle("engine.execute", json!({"command": "image.adjustments.invert", "params": {}})).unwrap();
    assert!(headless.session.active().unwrap().is_dirty(), "{name}: the edit dirties");

    // The reported symptom: the save succeeds but the session still reports `dirty`.
    headless.handle("doc.save", json!({})).unwrap();
    assert!(!headless.session.active().unwrap().is_dirty(), "{name}: a pathless save is clean");
    let revision = headless.session.active().unwrap().revision;
    assert_eq!(headless.session.active().unwrap().saved_revision, revision, "{name}: the current revision is recorded");
    assert_eq!(headless.session.active().unwrap().path.as_deref(), path.to_str(), "{name}: the file is still the path");

    // A second save, with no further edit, stays clean.
    headless.handle("doc.save", json!({})).unwrap();
    assert!(!headless.session.active().unwrap().is_dirty(), "{name}: a repeat save stays clean");

    let bytes = std::fs::read(&path).unwrap();
    if name.ends_with("psd") {
        assert!(photocraft_io::is_psd(&bytes), "{name}: PSD magic");
    } else {
        assert_eq!(&bytes[..4], b"8BPS", "{name}: PSB magic");
    }
}

#[test]
fn pathless_psd_and_psb_saves_clear_dirty() {
    let dir = tmp("clean");
    pathless_save_clears_dirty(&dir, "a.psd");
    pathless_save_clears_dirty(&dir, "a.psb");
    std::fs::remove_dir_all(dir).unwrap();
}

/// The control from the issue: exporting a flat copy is not a document save, so it must neither
/// clear `dirty` nor retarget the document's path. A `.pcraft` save sets the path on both the old
/// and the new code, so this test isolates the flat-export behaviour.
#[test]
fn a_flat_export_is_a_copy_and_leaves_the_document_dirty() {
    let dir = tmp("copy");
    let native = dir.join("doc.pcraft");
    let png = dir.join("copy.png");
    let mut headless = Headless::trusted_local();
    headless.handle("doc.new", json!({"width": 8, "height": 8, "background": "white"})).unwrap();
    headless.handle("doc.save", json!({"path": native.to_str().unwrap()})).unwrap();
    headless.handle("engine.execute", json!({"command": "image.adjustments.invert", "params": {}})).unwrap();
    assert!(headless.session.active().unwrap().is_dirty(), "the edit dirties");

    headless.handle("doc.save", json!({"path": png.to_str().unwrap()})).unwrap();
    assert!(png.is_file(), "the PNG copy was written");
    assert!(headless.session.active().unwrap().is_dirty(), "a flat export leaves the document dirty");
    assert_eq!(headless.session.active().unwrap().path.as_deref(), native.to_str(), "the path is not retargeted to the copy");

    // ...and the document can still be saved back to its own native file.
    headless.handle("doc.save", json!({})).unwrap();
    assert!(!headless.session.active().unwrap().is_dirty());
    std::fs::remove_dir_all(dir).unwrap();
}
