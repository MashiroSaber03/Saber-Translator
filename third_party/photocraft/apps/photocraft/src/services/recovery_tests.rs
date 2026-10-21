use super::*;
use photocraft_color::Color;
use photocraft_format::{Autosaver, RecoveryEntry, SaveOptions, list_recovery};
use std::sync::atomic::{AtomicUsize, Ordering};

fn snapshot(tag: &str) -> (PathBuf, Document, RecoveryEntry) {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!("photocraft-deferred-recovery-{tag}-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    let _ = std::fs::remove_dir_all(&dir);
    let mut doc = Document::with_background("Recovered drawing", Size::new(4, 3), ColorMode::Rgb, SampleType::U16, Color::WHITE);
    doc.layers[0].surface_mut().unwrap().write_pixel(1, 2, &[1.0, 0.0, 0.0, 1.0]);
    let saver = Autosaver::new(&dir, "previous-session");
    saver.request_checked(Arc::new(doc.clone()), 7, Some(dir.join("original.pcraft").to_string_lossy().into_owned()), SaveOptions::default()).unwrap();
    assert!(saver.flush().unwrap().unwrap().tiles_written > 0);
    let mut entries = list_recovery(&dir);
    assert_eq!(entries.len(), 1);
    (dir, doc, entries.pop().unwrap())
}

#[test]
fn recovery_discovers_metadata_before_decoding_and_keeps_a_failed_snapshot() {
    let (dir, doc, entry) = snapshot("corrupt");
    let tile = std::fs::read_dir(entry.bundle.join("tiles")).unwrap().next().unwrap().unwrap().path();
    std::fs::write(&tile, b"corrupt compressed tile").unwrap();
    let sidecar = dir.join(format!("{}.json", entry.info.key));
    let metadata = std::fs::read(&sidecar).unwrap();
    let mut services = recovery_services(Some(dir.clone()));

    let mut found = services.recover.as_mut().unwrap()();
    assert_eq!(found.len(), 1, "discovery lists metadata without decoding the corrupt pixels");
    let pending = found.pop().unwrap();
    assert_eq!(pending.key, entry.info.key);
    assert_eq!(pending.name, doc.name);
    assert_eq!(pending.path, entry.info.original_path);
    assert!((pending.load)().is_err(), "decoding is deferred until the loader runs");

    assert_eq!(std::fs::read(&tile).unwrap(), b"corrupt compressed tile");
    assert_eq!(std::fs::read(&sidecar).unwrap(), metadata);
    assert_eq!(list_recovery(&dir), vec![entry], "a failed recovery must leave the snapshot available");
    drop(services);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn deferred_recovery_preserves_pixels_and_metadata_until_the_adopted_entry_is_discarded() {
    let (dir, doc, entry) = snapshot("valid");
    let mut services = recovery_services(Some(dir.clone()));
    let mut found = services.recover.as_mut().unwrap()();
    assert_eq!(found.len(), 1);
    let pending = found.pop().unwrap();
    assert_eq!(pending.key, entry.info.key);
    assert_eq!(pending.name, doc.name);
    assert_eq!(pending.path, entry.info.original_path);

    // The loader can move to a worker without taking the main thread's recovery store with it.
    let restored = std::thread::spawn(pending.load).join().unwrap().unwrap();
    assert_eq!(restored, doc);
    assert_eq!(restored.layers[0].surface().unwrap().pixel(1, 2), vec![1.0, 0.0, 0.0, 1.0]);
    services.adopt_autosave.as_mut().unwrap()(restored.id.0, &pending.key);
    assert_eq!(list_recovery(&dir), vec![entry.clone()], "loading and adopting retain the recovery files");

    services.discard_autosave.as_mut().unwrap()(restored.id.0);
    assert!(list_recovery(&dir).is_empty());
    assert!(!entry.bundle.exists());
    assert!(!dir.join(format!("{}.json", pending.key)).exists());
    drop(services);
    std::fs::remove_dir_all(dir).unwrap();
}
