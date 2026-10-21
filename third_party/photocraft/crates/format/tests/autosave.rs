//! Background autosave and crash recovery.

mod common;
use std::sync::Arc;

use common::*;
use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::{DocId, Document};
use photocraft_format::*;

#[test]
fn autosave_then_recover() {
    let dir = temp_dir("recovery");
    let doc = Arc::new(rich_doc(ColorMode::Rgb, SampleType::U16));
    let saver = Autosaver::new(&dir, "doc-1");
    saver.request(doc.clone(), 7, Some("/work/a.pcraft".into()), SaveOptions::default());
    let r = saver.flush().expect("a save ran");
    let stats = r.unwrap();
    assert!(stats.tiles_written > 0);
    let entries = list_recovery(&dir);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].info.revision, 7);
    assert_eq!(entries[0].info.original_path.as_deref(), Some("/work/a.pcraft"));
    assert_eq!(entries[0].info.document_name, "Rich");
    assert_eq!(recover(&entries[0]).unwrap(), *doc);
    discard_recovery(&dir, &entries[0]).unwrap();
    assert!(list_recovery(&dir).is_empty());
    std::fs::remove_dir_all(dir).unwrap();
}

/// A queued snapshot can fail later; callers must observe and retry that revision.
#[test]
fn background_save_outcomes_report_failure_then_recovery() {
    let root = temp_dir("retry-failed-write");
    let blocked = root.join("recovery");
    std::fs::write(&blocked, b"not a directory").unwrap();
    let saver = Autosaver::new(&blocked, "doc");
    let doc = Arc::new(rich_doc(ColorMode::Rgb, SampleType::U8));
    saver.request_checked(doc.clone(), 7, None, SaveOptions::default()).unwrap();

    let receive = |saver: &Autosaver| {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while std::time::Instant::now() < deadline {
            if let Some(outcome) = saver.take_completed().into_iter().next() {
                return outcome;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("autosave worker did not report its completed write within 30 seconds");
    };
    let (revision, result) = receive(&saver);
    assert_eq!(revision, 7);
    assert!(result.is_err(), "the recovery location is not a directory");

    std::fs::remove_file(&blocked).unwrap();
    std::fs::create_dir(&blocked).unwrap();
    saver.request_checked(doc.clone(), 7, None, SaveOptions::default()).unwrap();
    let (revision, result) = receive(&saver);
    assert_eq!(revision, 7);
    result.unwrap();
    assert_eq!(list_recovery(&blocked).len(), 1);
    assert_eq!(recover(&list_recovery(&blocked)[0]).unwrap(), *doc);
    drop(saver);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn repeated_autosaves_are_incremental_and_coalesced() {
    let dir = temp_dir("coalesce");
    let saver = Autosaver::new(&dir, "k");
    let mut doc = rich_doc(ColorMode::Rgb, SampleType::U8);
    for i in 0..5 {
        let id = doc.layers[1].id;
        doc.layer_mut(id).unwrap().surface_mut().unwrap().write_pixel(i, 0, &[1.0, 0.0, 0.0, 1.0]);
        saver.request(Arc::new(doc.clone()), i as u64, None, SaveOptions::default());
    }
    saver.flush().unwrap().unwrap();
    let e = list_recovery(&dir);
    assert_eq!(e[0].info.revision, 4, "newest snapshot wins");
    assert_eq!(recover(&e[0]).unwrap(), doc);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn discard_removes_everything() {
    let dir = temp_dir("discard");
    let saver = Autosaver::new(&dir, "x y/z");
    saver.request(Arc::new(rich_doc(ColorMode::Grayscale, SampleType::U8)), 1, None, SaveOptions::default());
    let path = saver.bundle_path();
    assert!(path.file_name().unwrap().to_string_lossy().starts_with("x_y_z"));
    saver.discard().unwrap();
    assert!(list_recovery(&dir).is_empty());
    assert!(!path.exists());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn list_ignores_junk() {
    let dir = temp_dir("junk");
    std::fs::write(dir.join("bogus.json"), b"{}").unwrap();
    std::fs::write(dir.join("other.txt"), b"x").unwrap();
    assert!(list_recovery(&dir).is_empty());
    assert!(list_recovery(&dir.join("missing")).is_empty());
    std::fs::remove_dir_all(dir).unwrap();
}

/// `doc` as document `id` with pixel (x, 0) of its paint layer set, so versions are told apart.
fn edit(mut doc: Document, id: u64, x: i32) -> Document {
    doc.id = DocId(id);
    let layer = doc.layers[1].id;
    doc.layer_mut(layer).unwrap().surface_mut().unwrap().write_pixel(x, 0, &[1.0, 0.0, 0.0, 1.0]);
    doc
}

fn version(id: u64, x: i32, mode: ColorMode) -> Document {
    edit(rich_doc(mode, SampleType::U8), id, x)
}

/// Save one snapshot under `key` and wait for it.
fn save_once(dir: &std::path::Path, key: &str, doc: Document) {
    let saver = Autosaver::new(dir, key);
    saver.request(Arc::new(doc), 1, None, SaveOptions::default());
    saver.flush().unwrap().unwrap();
}

/// The recovered documents by entry key, ids cleared (they're reassigned on open).
fn recovered(store: &RecoveryStore) -> std::collections::BTreeMap<String, Document> {
    store
        .recover()
        .into_iter()
        .map(|(e, mut doc)| {
            doc.id = DocId(0);
            (e.info.key, doc)
        })
        .collect()
}

fn same(mut doc: Document) -> Document {
    doc.id = DocId(0);
    doc
}

#[test]
fn recovered_entries_survive_a_second_crash_until_saved_or_closed() {
    let dir = temp_dir("second-crash");
    let (a, b) = (version(1, 3, ColorMode::Rgb), version(2, 5, ColorMode::Grayscale));
    // Launch 1 autosaves two unsaved documents, then crashes (dropping waits for the writes).
    {
        let mut launch = RecoveryStore::new(&dir);
        launch.autosave(&Arc::new(a.clone()), 4, None);
        launch.autosave(&Arc::new(b.clone()), 6, Some("/work/b.pcraft".into()));
    }
    assert_eq!(list_recovery(&dir).len(), 2);
    // Launch 2 recovers both and crashes before its first autosave: nothing may be lost.
    let keys: Vec<String> = {
        let mut launch = RecoveryStore::new(&dir);
        let docs = recovered(&launch);
        assert_eq!(docs.len(), 2);
        assert_eq!(list_recovery(&dir).len(), 2, "recovering must not delete the entries");
        for (i, key) in docs.keys().enumerate() {
            launch.adopt(10 + i as u64, key);
        }
        docs.into_keys().collect()
    };
    // Launch 3 still has both, pixels intact; one is edited and autosaved, the other closed.
    let edited = edit(a.clone(), 20, 9);
    {
        let mut launch = RecoveryStore::new(&dir);
        let docs = recovered(&launch);
        assert_eq!(docs.values().cloned().collect::<Vec<_>>(), vec![same(a.clone()), same(b.clone())]);
        launch.adopt(20, &keys[0]);
        launch.adopt(21, &keys[1]);
        launch.autosave(&Arc::new(edited.clone()), 9, None);
        launch.discard(21).unwrap();
        assert_eq!(list_recovery(&dir).len(), 1, "closing removes the adopted entry");
    }
    // The autosave replaced the adopted entry in place: no duplicate, newest content.
    let entries = list_recovery(&dir);
    assert_eq!(entries.len(), 1);
    assert_eq!((entries[0].info.key.as_str(), entries[0].info.revision), (keys[0].as_str(), 9));
    // Launch 4 recovers it once more and saves it: the entry goes.
    let mut launch = RecoveryStore::new(&dir);
    assert_eq!(recovered(&launch).into_values().collect::<Vec<_>>(), vec![same(edited)]);
    launch.adopt(30, &keys[0]);
    launch.discard(30).unwrap();
    assert!(list_recovery(&dir).is_empty());
    drop(launch);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn new_documents_never_overwrite_surviving_entries() {
    let dir = temp_dir("collision");
    // An earlier launch (including builds that keyed entries by bare document id) left `doc-1`.
    let old = version(1, 3, ColorMode::Rgb);
    save_once(&dir, "doc-1", old.clone());
    // A store from another launch left an entry for its own document 1 too.
    let other = version(1, 6, ColorMode::Grayscale);
    RecoveryStore::new(&dir).autosave(&Arc::new(other.clone()), 2, None);
    assert_eq!(list_recovery(&dir).len(), 2);
    // This launch adopts `doc-1` for its document 5, and its own new document 1 autosaves.
    let fresh = version(1, 8, ColorMode::Rgb);
    {
        let mut launch = RecoveryStore::new(&dir);
        launch.adopt(5, "doc-1");
        launch.autosave(&Arc::new(fresh.clone()), 3, None);
    }
    let docs = recovered(&RecoveryStore::new(&dir));
    assert_eq!(docs.len(), 3, "the new document got an entry of its own");
    assert_eq!(docs.get("doc-1"), Some(&same(old)), "the adopted entry is untouched");
    assert!(docs.values().any(|d| *d == same(other.clone())));
    assert!(docs.values().any(|d| *d == same(fresh.clone())));
    // Closing the new document removes only its own entry.
    let mut launch = RecoveryStore::new(&dir);
    launch.adopt(5, "doc-1");
    launch.autosave(&Arc::new(fresh), 3, None);
    launch.discard(1).unwrap();
    assert_eq!(list_recovery(&dir).len(), 3);
    launch.discard(5).unwrap();
    assert_eq!(list_recovery(&dir).len(), 2);
    drop(launch);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn sidecars_cannot_point_at_other_bundles() {
    let dir = temp_dir("sidecar-key");
    let inner = dir.join("Recovery");
    save_once(&dir, "outside", rich_doc(ColorMode::Rgb, SampleType::U8));
    save_once(&inner, "k", rich_doc(ColorMode::Rgb, SampleType::U8));
    let info = std::fs::read_to_string(inner.join("k.json")).unwrap();
    // A copied sidecar (its name isn't its key) and one whose key leaves the directory.
    std::fs::write(inner.join("copy.json"), &info).unwrap();
    std::fs::write(inner.join("escape.json"), info.replace("\"k\"", "\"../outside\"")).unwrap();
    let entries = list_recovery(&inner);
    assert_eq!(entries.iter().map(|e| e.info.key.as_str()).collect::<Vec<_>>(), ["k"]);
    std::fs::remove_dir_all(dir).unwrap();
}

/// Once the manifest is committed the save has succeeded: an unreferenced object that cannot be
/// removed (here a folder squatting on a tile name, like a locked or vanished file) must not turn
/// the directory save (autosave's path) into a reported failure.
#[test]
fn failed_cleanup_after_commit_does_not_fail_the_save() {
    let dir = temp_dir("cleanup-best-effort").join("doc.pcraft");
    std::fs::create_dir_all(&dir).unwrap();
    let doc = rich_doc(ColorMode::Rgb, SampleType::U8);
    let mut w = PcraftWriter::new();
    w.save_dir(&doc, &dir, &SaveOptions::default()).unwrap();
    let stuck = dir.join("tiles").join("0000000000000000000000000000000000000000000000000000000000000000.zst");
    std::fs::create_dir_all(stuck.join("inner")).unwrap();
    let stats = w.save_dir(&doc, &dir, &SaveOptions::default()).expect("the committed save reports success");
    assert_eq!(stats.objects_removed, 0, "nothing was actually removed");
    assert_eq!(load_path(&dir).unwrap(), doc);
    std::fs::remove_dir_all(dir.parent().unwrap()).unwrap();
}
