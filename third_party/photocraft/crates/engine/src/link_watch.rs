//! Linked smart objects follow their files while the document is open, as in Photoshop
//! ("Linked Smart Objects update automatically when the Photoshop document is open, provided the
//! source file has changed", <https://helpx.adobe.com/photoshop/desktop/create-manage-layers/smart-objects/update-linked-smart-objects.html>):
//! a file another app saves (VectorCraft, LightCraft, Illustrator) updates its layers without
//! Layer › Smart Objects › Update Modified Content.
//!
//! The active document's linked files are compared by size and modification time with what they
//! were when last seen; the layers of a file whose stamp moved are re-rendered from it, one
//! Update Modified Content undo step per file. The desktop app takes the stamps every two seconds
//! and when its window comes back to the front, on a worker thread so a file on a slow network
//! share never holds up the interface ([`Session::start_link_scan`], [`Session::poll_link_scan`]);
//! `layer.smartObjects.updateChanged` takes them on the spot (CLI, agents).
//!
//! Stamps are kept per document: one in the background is looked at when it comes to the front.
//! A file seen for the first time is only remembered (one changed while its document was closed
//! keeps waiting for Update Modified Content, as in Photoshop); one that can't be read (another app
//! is still writing it) is tried again once it changes again, not on every look. Only smart objects
//! of the document itself follow their files, not ones nested in another smart object's contents.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex, PoisonError};

use photocraft_doc::{Document, LayerContent, LayerId, SmartSource};
use serde_json::{Value, json};

use crate::{EngineError, Result, Session};

/// A file's size and modification time (nanoseconds since 1970).
pub type Stamp = (u64, u128);

/// Stamps of a document's linked files taken on a worker thread: (document id, [(file, layer
/// ids, stamp)]).
pub type LinkScan = Arc<Mutex<Option<(u64, Vec<(String, Vec<u64>, Option<Stamp>)>)>>>;

#[cfg(not(target_arch = "wasm32"))]
fn stamp(path: &str) -> Option<Stamp> {
    let meta = std::fs::metadata(path).ok()?;
    let modified = meta.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?;
    Some((meta.len(), modified.as_nanos()))
}

#[cfg(target_arch = "wasm32")]
fn stamp(_: &str) -> Option<Stamp> {
    None
}

/// The files of `d`'s linked smart objects, with the layers showing each.
pub fn linked_files(d: &Document) -> Vec<(String, Vec<u64>)> {
    let mut by_file: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    for (_, _, l) in d.walk() {
        if let LayerContent::Smart(sm) = &l.content
            && let SmartSource::Linked { path } = &sm.source
        {
            by_file.entry(path.clone()).or_default().push(l.id.0);
        }
    }
    by_file.into_iter().collect()
}

/// `layer.smartObjects.updateChanged`: stamp the active document's linked files now and update
/// the layers of the ones that changed.
pub(crate) fn update_changed(s: &mut Session, _: &Value) -> Result<Value> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let (doc, files) = (st.doc.id.0, linked_files(&st.doc));
    let now = files.into_iter().map(|(p, ids)| (p.clone(), ids, stamp(&p))).collect();
    s.apply_link_stamps(doc, now)
}

impl Session {
    /// Start stamping the active document's linked files on a worker thread, unless a look is
    /// still running (or the build has no files to look at: the web app).
    pub fn start_link_scan(&mut self) {
        if self.link_scan.is_some() || cfg!(target_arch = "wasm32") {
            return;
        }
        let Some(st) = self.active() else { return };
        let (doc, files) = (st.doc.id.0, linked_files(&st.doc));
        if files.is_empty() {
            return;
        }
        let slot: LinkScan = Arc::default();
        let out = slot.clone();
        let spawned = std::thread::Builder::new().name("link-watch".into()).spawn(move || {
            let stamps = files.into_iter().map(|(p, ids)| (p.clone(), ids, stamp(&p))).collect();
            *out.lock().unwrap_or_else(PoisonError::into_inner) = Some((doc, stamps));
        });
        if spawned.is_ok() {
            self.link_scan = Some(slot);
        }
    }

    /// Apply a finished look (`None` while there is none, or it is still running).
    pub fn poll_link_scan(&mut self) -> Option<Result<Value>> {
        let (doc, stamps) = self.link_scan.as_ref()?.lock().unwrap_or_else(PoisonError::into_inner).take()?;
        self.link_scan = None;
        Some(self.apply_link_stamps(doc, stamps))
    }

    /// Compare the stamps of document `doc`'s linked files with the ones last seen and update the
    /// layers of the changed ones. Returns `{updated: [layer ids]}`.
    fn apply_link_stamps(&mut self, doc: u64, now: Vec<(String, Vec<u64>, Option<Stamp>)>) -> Result<Value> {
        // The document went to the back meanwhile: it's looked at when it's in front again.
        if self.active().map(|d| d.doc.id.0) != Some(doc) {
            return Ok(json!({"updated": []}));
        }
        // This document's last stamps; those of closed documents are dropped.
        let open: HashSet<u64> = self.documents().iter().map(|d| d.doc.id.0).collect();
        let mut before: HashMap<String, Stamp> = HashMap::new();
        self.link_stamps.retain(|(d, p), s| {
            if *d == doc {
                before.insert(p.clone(), *s);
                false
            } else {
                open.contains(d)
            }
        });
        let mut changed = Vec::new();
        for (path, ids, stamp) in now {
            match (before.get(&path), stamp) {
                // Not there right now (an app saving by delete and rename): keep the last stamp.
                (Some(old), None) => {
                    self.link_stamps.insert((doc, path), *old);
                }
                (old, Some(new)) => {
                    if old.is_some_and(|o| *o != new) {
                        changed.push((path.clone(), ids));
                    }
                    self.link_stamps.insert((doc, path), new);
                }
                (None, None) => {}
            }
        }
        let mut updated = Vec::new();
        for (path, ids) in changed {
            let r = self.edit("Update Modified Content", |d, _| {
                for id in &ids {
                    if !crate::smart_cmds::refresh(d, LayerId(*id))? {
                        return Err(EngineError::Other(format!("{path} can't be read")));
                    }
                }
                Ok(())
            });
            // Not readable (still being written, or not a format smart objects read): the edit left
            // nothing behind, and the file is tried again once it changes again, not on every look.
            if r.is_ok() {
                updated.extend(ids);
            }
        }
        Ok(json!({"updated": updated}))
    }
}

#[cfg(test)]
mod tests {
    use photocraft_geom::Rect;
    use serde_json::json;

    use crate::Session;

    fn dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("photocraft-link-watch-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// The bytes of a solid-colour `w × h` PNG (made in a folder of its own: tests run in
    /// parallel).
    fn png(w: u32, h: u32, color: &str) -> Vec<u8> {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let d = dir(&format!("png-{}-{n}", color.trim_start_matches('#')));
        let path = d.join("x.png").to_string_lossy().into_owned();
        let mut s = Session::new();
        s.execute("file.new", json!({"width": w, "height": h, "background": color})).unwrap();
        s.execute("file.saveACopy", json!({"path": path, "layers": false})).unwrap();
        std::fs::read(&path).unwrap()
    }

    /// Write `bytes` to `path` so its modification time moves even on coarse file systems.
    fn save(path: &std::path::Path, bytes: &[u8]) {
        std::thread::sleep(std::time::Duration::from_millis(30));
        std::fs::write(path, bytes).unwrap();
    }

    /// A 40×40 white document with the file at `path` placed linked → (session, layer id).
    fn placed(path: &std::path::Path) -> (Session, u64) {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 40, "height": 40})).unwrap();
        s.execute("file.placeLinked", json!({"path": path.to_string_lossy()})).unwrap();
        let id = s.active().unwrap().active_layer.unwrap().0;
        (s, id)
    }

    fn centre(s: &Session) -> [f32; 3] {
        let p = photocraft_compose::render(&s.active().unwrap().doc, Rect::new(20, 20, 21, 21)).px[0];
        [p[0], p[1], p[2]]
    }

    fn is(c: [f32; 3], rgb: [f32; 3]) -> bool {
        c.iter().zip(rgb).all(|(a, b)| (a - b).abs() < 0.02)
    }

    const RED: [f32; 3] = [1.0, 0.0, 0.0];
    const BLUE: [f32; 3] = [0.0, 0.0, 1.0];

    #[test]
    fn a_linked_smart_object_follows_its_file() {
        let d = dir("follow");
        let path = d.join("art.png");
        std::fs::write(&path, png(10, 10, "#ff0000")).unwrap();
        let (mut s, id) = placed(&path);
        assert!(is(centre(&s), RED));
        // First look: remembered, nothing to update.
        assert_eq!(s.execute("layer.smartObjects.updateChanged", json!({})).unwrap(), json!({"updated": []}));
        save(&path, &png(10, 10, "#0000ff"));
        assert_eq!(s.execute("layer.smartObjects.updateChanged", json!({})).unwrap(), json!({"updated": [id]}));
        assert!(is(centre(&s), BLUE), "{:?}", centre(&s));
        // An Update Modified Content step: undoing it is respected until the file changes again.
        assert!(s.undo());
        assert!(is(centre(&s), RED));
        assert_eq!(s.execute("layer.smartObjects.updateChanged", json!({})).unwrap(), json!({"updated": []}));
        assert!(is(centre(&s), RED));
    }

    #[test]
    fn a_file_still_being_written_is_tried_again() {
        let d = dir("partial");
        let path = d.join("art.png");
        std::fs::write(&path, png(10, 10, "#ff0000")).unwrap();
        let (mut s, id) = placed(&path);
        s.execute("layer.smartObjects.updateChanged", json!({})).unwrap();
        let undo = s.active().unwrap().history.past_len();
        // Another app has only written the start of the file so far.
        let full = png(10, 10, "#0000ff");
        save(&path, &full[..16]);
        assert_eq!(s.execute("layer.smartObjects.updateChanged", json!({})).unwrap(), json!({"updated": []}));
        assert!(is(centre(&s), RED), "the last good pixels stay");
        assert_eq!(s.active().unwrap().history.past_len(), undo, "no step for a failed update");
        save(&path, &full);
        assert_eq!(s.execute("layer.smartObjects.updateChanged", json!({})).unwrap(), json!({"updated": [id]}));
        assert!(is(centre(&s), BLUE));
    }

    #[test]
    fn a_document_in_the_back_catches_up_in_front() {
        let d = dir("back");
        let path = d.join("art.png");
        std::fs::write(&path, png(10, 10, "#ff0000")).unwrap();
        let (mut s, id) = placed(&path);
        s.execute("layer.smartObjects.updateChanged", json!({})).unwrap();
        s.execute("file.new", json!({"width": 10, "height": 10})).unwrap();
        save(&path, &png(10, 10, "#0000ff"));
        assert_eq!(s.execute("layer.smartObjects.updateChanged", json!({})).unwrap(), json!({"updated": []}));
        s.set_active(0);
        assert_eq!(s.execute("layer.smartObjects.updateChanged", json!({})).unwrap(), json!({"updated": [id]}));
        assert!(is(centre(&s), BLUE));
    }

    #[test]
    fn a_background_look_updates_changed_files() {
        let d = dir("scan");
        let path = d.join("art.png");
        std::fs::write(&path, png(10, 10, "#ff0000")).unwrap();
        let (mut s, id) = placed(&path);
        let look = |s: &mut Session| {
            s.start_link_scan();
            let t0 = std::time::Instant::now();
            loop {
                if let Some(r) = s.poll_link_scan() {
                    return r.unwrap();
                }
                assert!(t0.elapsed().as_secs() < 10, "the look never finished");
                std::thread::yield_now();
            }
        };
        assert_eq!(look(&mut s), json!({"updated": []}));
        save(&path, &png(10, 10, "#0000ff"));
        assert_eq!(look(&mut s), json!({"updated": [id]}));
        assert!(is(centre(&s), BLUE));
        assert!(s.link_scan.is_none());
    }
}
