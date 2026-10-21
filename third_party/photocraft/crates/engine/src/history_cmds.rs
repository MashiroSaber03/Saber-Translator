//! History panel actions beyond stepping (#1117): Delete (the trash button) and New Document (the
//! "Create new document from current state" button). Stepping to a state is Edit › Undo / Redo.
//!
//! States are indexed like [`photocraft_ops::History::entries`]: 0 is the oldest state still
//! held, the last index is the current document.
//!
//! Snapshots (#2966, the camera button) are named copies of the document that outlive the
//! states: `history.newSnapshot` takes one, `history.selectSnapshot` brings it back as a new,
//! undoable state. They live on [`crate::DocState::snapshots`] for the session only, as in
//! Photoshop, and are addressed by an id that is never reused. Their tiles are shared
//! copy-on-write with the document and its history, and are not counted in the history's pixel
//! budget.

use std::sync::Arc;

use photocraft_doc::{Document, Layer, LayerContent};
use photocraft_ops::LayerTarget;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, int};
use crate::image_cmds::reidentify_copied_layers;
use crate::{EngineError, Result, Session};

/// A History panel snapshot: the document as it was when taken, and the layers it targeted.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub id: u64,
    pub name: String,
    pub doc: Arc<Document>,
    pub layers: LayerTarget,
}

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

/// The `state` param as an index into the active document's history entries, defaulting to the
/// current state.
fn state_index(s: &Session, cmd: &str, p: &Value) -> Result<(usize, usize)> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let current = st.history.past_len();
    let i = match p.get("state") {
        None | Some(Value::Null) => current,
        Some(_) => {
            let v = int(p, "state").ok_or_else(|| bad(cmd, "`state` must be a number"))?;
            usize::try_from(v).ok().filter(|i| *i <= current).ok_or_else(|| bad(cmd, format!("`state` = {v} is not a state (0..={current})")))?
        }
    };
    Ok((i, current))
}

/// History panel › Delete: drop the state and every state after it, as Photoshop does in linear
/// history. The document returns to the state before it, and the deleted steps can't be redone.
/// The oldest state (the document as opened) can't be deleted.
fn delete_state(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "history.deleteState";
    let (i, current) = state_index(s, CMD, p)?;
    if i == 0 {
        return Err(bad(CMD, "the oldest state can't be deleted"));
    }
    // Step back to the state before `i`; the steps from `i` on are then redo states.
    for _ in i..=current {
        if !s.undo() {
            return Err(EngineError::Other("the history can't step back right now".into()));
        }
    }
    let st = s.active_mut().ok_or(EngineError::NoDocument)?;
    st.history.clear_redo();
    let entries = st.history.entries();
    Ok(json!({ "states": entries.len(), "current": entries.last() }))
}

/// History panel › New Document: a new untitled document holding the document as it was at a
/// state (the current one by default), named after that state, with a history of its own.
fn new_document(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "history.newDocument";
    let (i, current) = state_index(s, CMD, p)?;
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let src = if i == current { Some(st.doc.clone()) } else { st.history.state(i) };
    let src = src.ok_or_else(|| bad(CMD, format!("state {i} is not held")))?;
    let label = st.history.entries().get(i).cloned().unwrap_or_default();
    let mut doc = (*src).clone();
    doc.id = photocraft_doc::DocId::fresh();
    doc.name = match p.get("name").and_then(Value::as_str) {
        Some(n) if !n.trim().is_empty() => n.to_string(),
        _ if !label.trim().is_empty() => label,
        _ => format!("{} copy", doc.name),
    };
    reidentify_copied_layers(&mut doc);
    let name = doc.name.clone();
    let index = s.add_document(doc, None);
    Ok(json!({ "document": index, "name": name }))
}

/// The `snapshot` param: the index of the active document's snapshot with that id.
fn snapshot_index(s: &Session, cmd: &str, p: &Value) -> Result<usize> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let id = p.get("snapshot").and_then(Value::as_u64).ok_or_else(|| bad(cmd, "`snapshot` must be a snapshot id"))?;
    st.snapshots.iter().position(|snap| snap.id == id).ok_or_else(|| bad(cmd, format!("no snapshot {id}")))
}

/// A non-blank `name` param, trimmed; `None` when absent.
fn name_param(cmd: &str, p: &Value) -> Result<Option<String>> {
    match p.get("name") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(n)) if !n.trim().is_empty() => Ok(Some(n.trim().to_string())),
        Some(_) => Err(bad(cmd, "`name` must be a non-empty string")),
    }
}

/// History panel › Create new snapshot (the camera button): a named copy of the document,
/// "Snapshot N" by default. `from` is the New Snapshot dialog's From: the full document (default),
/// its layers merged into one, or only the active layer. Not a history state.
fn new_snapshot(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "history.newSnapshot";
    let name = name_param(CMD, p)?;
    let from = match p.get("from") {
        None | Some(Value::Null) => "fullDocument",
        Some(Value::String(f)) if matches!(f.as_str(), "fullDocument" | "mergedLayers" | "currentLayer") => f.as_str(),
        Some(_) => return Err(bad(CMD, "`from` must be \"fullDocument\", \"mergedLayers\" or \"currentLayer\"")),
    };
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let one_layer = |doc: &Document, layer: Layer| {
        let mut doc = doc.clone();
        let id = layer.id;
        doc.layers = vec![layer];
        (Arc::new(doc), LayerTarget { active: Some(id), selected: vec![id] })
    };
    let (doc, layers) = match from {
        "mergedLayers" => {
            let fmt = st.doc.pixel_format();
            one_layer(&st.doc, Layer::new("Merged Layers", LayerContent::Raster(crate::file_cmds::flattened(&st.doc, fmt))))
        }
        "currentLayer" => {
            let layer = st.active_layer.and_then(|id| st.doc.layer(id)).ok_or_else(|| bad(CMD, "no active layer"))?;
            one_layer(&st.doc, layer.clone())
        }
        _ => (st.doc.clone(), st.layer_target()),
    };
    let st = s.active_mut().ok_or(EngineError::NoDocument)?;
    let id = st.next_snapshot;
    st.next_snapshot = id.saturating_add(1);
    let name = name.unwrap_or_else(|| format!("Snapshot {id}"));
    st.snapshots.push(Snapshot { id, name: name.clone(), doc, layers });
    Ok(json!({ "snapshot": id, "name": name }))
}

/// Clicking a snapshot: the document returns to it as one new, undoable state named after it.
fn select_snapshot(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "history.selectSnapshot";
    let i = snapshot_index(s, CMD, p)?;
    // A background job is computing from the current state: it must not move under it.
    if s.active_job().is_some() {
        return Err(EngineError::Other("the document is busy with a background job".into()));
    }
    let snap = s.active().and_then(|st| st.snapshots.get(i).cloned()).ok_or(EngineError::NoDocument)?;
    s.edit(&snap.name, |doc, active| {
        // Keep the identity (colour state, views and caches are keyed by it) and the tab name.
        let (id, name) = (doc.id, doc.name.clone());
        *doc = (*snap.doc).clone();
        doc.id = id;
        doc.name = name;
        *active = snap.layers.active.filter(|l| doc.layer(*l).is_some()).or_else(|| doc.top_layer());
        Ok(())
    })?;
    let st = s.active_mut().ok_or(EngineError::NoDocument)?;
    if st.active_layer == snap.layers.active {
        st.selected_layers.clone_from(&snap.layers.selected);
        crate::fix_selection(st);
    }
    Ok(json!({ "snapshot": snap.id, "name": snap.name }))
}

/// Renaming a snapshot (double-click its name).
fn rename_snapshot(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "history.renameSnapshot";
    let i = snapshot_index(s, CMD, p)?;
    let name = name_param(CMD, p)?.ok_or_else(|| bad(CMD, "`name` is required"))?;
    let st = s.active_mut().ok_or(EngineError::NoDocument)?;
    let snap = st.snapshots.get_mut(i).ok_or_else(|| bad(CMD, "no such snapshot"))?;
    snap.name.clone_from(&name);
    Ok(json!({ "snapshot": snap.id, "name": name }))
}

/// Deleting a snapshot; the document and its history are unchanged.
fn delete_snapshot(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "history.deleteSnapshot";
    let i = snapshot_index(s, CMD, p)?;
    let st = s.active_mut().ok_or(EngineError::NoDocument)?;
    let snap = st.snapshots.remove(i);
    Ok(json!({ "snapshot": snap.id, "snapshots": st.snapshots.len() }))
}

fn can_delete(s: &Session) -> std::result::Result<(), String> {
    s.active().filter(|d| d.history.can_undo()).map(|_| ()).ok_or_else(|| "the oldest state can't be deleted".into())
}

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

/// `history.undo` / `history.redo` enable predicates, matching `edit.undo` / `edit.redo`.
fn can_undo(s: &Session) -> std::result::Result<(), String> {
    s.active().filter(|d| d.history.can_undo()).map(|_| ()).ok_or_else(|| "nothing to undo".into())
}

fn can_redo(s: &Session) -> std::result::Result<(), String> {
    s.active().filter(|d| d.history.can_redo()).map(|_| ()).ok_or_else(|| "nothing to redo".into())
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        // `history.*` aliases of Edit › Undo / Redo, so a client reaching for the History namespace
        // finds them (`edit.undo` / `edit.redo` remain the menu items and own the shortcuts, #3675).
        CommandSpec {
            id: "history.undo",
            label: "Undo",
            menu: &[],
            shortcut: None,
            params: "{} (the same action as `edit.undo`)",
            enabled: can_undo,
            run: |s, _| Ok(json!(s.undo())),
            journal: false,
        },
        CommandSpec {
            id: "history.redo",
            label: "Redo",
            menu: &[],
            shortcut: None,
            params: "{} (the same action as `edit.redo`)",
            enabled: can_redo,
            run: |s, _| Ok(json!(s.redo())),
            journal: false,
        },
        CommandSpec {
            id: "history.deleteState",
            label: "Delete State",
            menu: &[],
            shortcut: None,
            params: r##"{"state":index?} (History entry, 0 = oldest; default the current state) → {states,current}: removes the state and every later one"##,
            enabled: can_delete,
            run: delete_state,
            journal: true,
        },
        CommandSpec {
            id: "history.newDocument",
            label: "New Document from State",
            menu: &[],
            shortcut: None,
            params: r##"{"state":index?,"name":str?} (History entry, 0 = oldest; default the current state) → {document,name}"##,
            enabled: has_doc,
            run: new_document,
            journal: true,
        },
        CommandSpec {
            id: "history.newSnapshot",
            label: "New Snapshot",
            menu: &[],
            shortcut: None,
            params: r##"{"name":str?="Snapshot N","from":"fullDocument"|"mergedLayers"|"currentLayer"="fullDocument"} → {snapshot,name}: a named copy of the document for the session (not a history state)"##,
            enabled: has_doc,
            run: new_snapshot,
            journal: true,
        },
        CommandSpec {
            id: "history.selectSnapshot",
            label: "Select Snapshot",
            menu: &[],
            shortcut: None,
            params: r##"{"snapshot":id} → {snapshot,name}: the document returns to the snapshot as a new history state named after it"##,
            enabled: has_doc,
            run: select_snapshot,
            journal: true,
        },
        CommandSpec {
            id: "history.renameSnapshot",
            label: "Rename Snapshot",
            menu: &[],
            shortcut: None,
            params: r##"{"snapshot":id,"name":str} → {snapshot,name}"##,
            enabled: has_doc,
            run: rename_snapshot,
            journal: true,
        },
        CommandSpec {
            id: "history.deleteSnapshot",
            label: "Delete Snapshot",
            menu: &[],
            shortcut: None,
            params: r##"{"snapshot":id} → {snapshot,snapshots}: the document and its history are unchanged"##,
            enabled: has_doc,
            run: delete_snapshot,
            journal: true,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A document with three steps after Open: entries Open, New Layer, Fill, Fill.
    fn session() -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 20, "height": 10})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
        s.execute("edit.fill", json!({"color": "#00ff00"})).unwrap();
        s
    }

    fn entries(s: &Session) -> Vec<String> {
        s.active().unwrap().history.entries()
    }

    #[test]
    fn history_undo_and_redo_alias_edit_undo_and_redo() {
        let mut s = session();
        let n = s.active().unwrap().history.past_len();
        s.execute("history.undo", json!({})).unwrap();
        assert_eq!(s.active().unwrap().history.past_len(), n - 1, "history.undo steps back like edit.undo");
        s.execute("history.redo", json!({})).unwrap();
        assert_eq!(s.active().unwrap().history.past_len(), n, "history.redo steps forward like edit.redo");
    }

    #[test]
    fn delete_current_state_steps_back_and_drops_it() {
        let mut s = session();
        let before = entries(&s);
        let prev_doc = s.active().unwrap().history.state(before.len() - 2).unwrap();
        let r = s.execute("history.deleteState", json!({})).unwrap();
        assert_eq!(r["states"], json!(before.len() - 1));
        assert_eq!(entries(&s), before[..before.len() - 1].to_vec());
        assert!(std::sync::Arc::ptr_eq(&s.active().unwrap().doc, &prev_doc), "the document is the state before");
        assert!(!s.active().unwrap().history.can_redo(), "the deleted state can't be redone");
        assert!(s.execute("edit.redo", json!({})).is_err());
    }

    #[test]
    fn delete_an_earlier_state_drops_it_and_every_later_one() {
        let mut s = session();
        let first = s.active().unwrap().history.state(0).unwrap();
        s.execute("history.deleteState", json!({"state": 1})).unwrap();
        assert_eq!(entries(&s), vec!["Open".to_string()]);
        assert!(std::sync::Arc::ptr_eq(&s.active().unwrap().doc, &first));
        assert!(!s.active().unwrap().history.can_redo());
        // Redo states after an undo go too.
        let mut s = session();
        s.execute("edit.undo", json!({})).unwrap();
        s.execute("history.deleteState", json!({})).unwrap();
        assert_eq!(entries(&s).len(), 2);
        assert!(!s.active().unwrap().history.can_redo());
    }

    #[test]
    fn delete_state_refuses_the_oldest_state_and_bad_params() {
        let mut s = Session::new();
        assert!(s.execute("history.deleteState", json!({})).is_err(), "no document");
        s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
        assert!(!s.is_enabled("history.deleteState"), "only the opening state");
        assert!(s.execute("history.deleteState", json!({})).is_err());
        let mut s = session();
        let before = entries(&s);
        for p in [json!({"state": 0}), json!({"state": -1}), json!({"state": 99}), json!({"state": "x"}), json!({"state": f64::MAX}), json!({"state": [1]})] {
            assert!(s.execute("history.deleteState", p.clone()).is_err(), "{p}");
            assert_eq!(entries(&s), before, "{p} changed nothing");
        }
    }

    #[test]
    fn new_document_from_a_state_copies_it_into_a_fresh_document() {
        let mut s = session();
        let src = s.active().unwrap().doc.clone();
        let label = entries(&s).last().cloned().unwrap();
        let r = s.execute("history.newDocument", json!({})).unwrap();
        assert_eq!(s.documents().len(), 2);
        assert_eq!(r["document"], json!(1));
        assert_eq!(r["name"], json!(label), "named after the state");
        let new = s.active().unwrap();
        assert_eq!(new.doc.size, src.size);
        assert_eq!(new.doc.layers.len(), src.layers.len());
        assert_ne!(new.doc.id, src.id);
        assert_eq!(new.history.entries().len(), 1, "a fresh history");
        // An earlier state, with a chosen name; the source document is untouched.
        s.set_active(0);
        let before = entries(&s);
        let open = s.active().unwrap().history.state(0).unwrap();
        s.execute("history.newDocument", json!({"state": 0, "name": "Start"})).unwrap();
        let new = s.active().unwrap();
        assert_eq!(new.doc.name, "Start");
        assert_eq!(new.doc.layers.len(), open.layers.len());
        s.set_active(0);
        assert_eq!(entries(&s), before);
    }

    /// The top layer's pixel at (1, 1).
    fn top_pixel(s: &Session) -> [f32; 4] {
        s.active().unwrap().doc.layers.last().unwrap().surface().unwrap().rgba(1, 1)
    }

    fn snapshots(s: &Session) -> Value {
        crate::inspect::document(s.active().unwrap())["snapshots"].clone()
    }

    #[test]
    fn new_snapshot_keeps_the_document_and_is_not_a_history_state() {
        let mut s = session();
        let before = entries(&s);
        let r = s.execute("history.newSnapshot", json!({})).unwrap();
        assert_eq!(r, json!({"snapshot": 1, "name": "Snapshot 1"}));
        assert_eq!(entries(&s), before, "taking a snapshot is not a step");
        let r = s.execute("history.newSnapshot", json!({"name": "Before green"})).unwrap();
        assert_eq!(r, json!({"snapshot": 2, "name": "Before green"}));
        assert_eq!(snapshots(&s), json!([{"id": 1, "name": "Snapshot 1"}, {"id": 2, "name": "Before green"}]));
    }

    #[test]
    fn select_snapshot_returns_to_it_as_one_undoable_state() {
        let mut s = session();
        s.execute("history.newSnapshot", json!({})).unwrap();
        let green = top_pixel(&s);
        let id = s.active().unwrap().doc.id;
        s.execute("edit.fill", json!({"color": "#0000ff"})).unwrap();
        let blue = top_pixel(&s);
        let before = entries(&s);
        s.execute("history.selectSnapshot", json!({"snapshot": 1})).unwrap();
        assert_eq!(top_pixel(&s), green);
        assert_eq!(entries(&s).len(), before.len() + 1, "one new state");
        assert_eq!(entries(&s).last().map(String::as_str), Some("Snapshot 1"));
        assert_eq!(s.active().unwrap().doc.id, id, "the same document");
        assert!(s.active().unwrap().active_layer.is_some());
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(top_pixel(&s), blue, "undo leaves the snapshot");
        assert_eq!(snapshots(&s).as_array().unwrap().len(), 1, "snapshots are not history");
    }

    #[test]
    fn snapshots_from_merged_layers_or_the_current_layer() {
        let mut s = session();
        assert_eq!(s.active().unwrap().doc.layers.len(), 2);
        s.execute("history.newSnapshot", json!({"from": "mergedLayers"})).unwrap();
        s.execute("history.newSnapshot", json!({"from": "currentLayer"})).unwrap();
        s.execute("history.newSnapshot", json!({"from": "fullDocument"})).unwrap();
        s.execute("history.selectSnapshot", json!({"snapshot": 1})).unwrap();
        let doc = s.active().unwrap().doc.clone();
        assert_eq!(doc.layers.len(), 1, "merged into one layer");
        assert_eq!(doc.layers[0].surface().unwrap().rgba(1, 1), [0.0, 1.0, 0.0, 1.0], "the visible result");
        s.execute("history.selectSnapshot", json!({"snapshot": 2})).unwrap();
        assert_eq!(s.active().unwrap().doc.layers.len(), 1, "only the active layer");
        s.execute("history.selectSnapshot", json!({"snapshot": 3})).unwrap();
        assert_eq!(s.active().unwrap().doc.layers.len(), 2);
    }

    #[test]
    fn rename_and_delete_snapshots_keep_the_other_ids() {
        let mut s = session();
        for _ in 0..3 {
            s.execute("history.newSnapshot", json!({})).unwrap();
        }
        s.execute("history.renameSnapshot", json!({"snapshot": 2, "name": "Lesson start"})).unwrap();
        s.execute("history.deleteSnapshot", json!({"snapshot": 1})).unwrap();
        assert_eq!(snapshots(&s), json!([{"id": 2, "name": "Lesson start"}, {"id": 3, "name": "Snapshot 3"}]));
        let r = s.execute("history.newSnapshot", json!({})).unwrap();
        assert_eq!(r["snapshot"], json!(4), "ids are never reused");
    }

    #[test]
    fn snapshot_commands_refuse_bad_params() {
        let mut s = Session::new();
        for (cmd, p) in [
            ("history.newSnapshot", json!({})),
            ("history.selectSnapshot", json!({"snapshot": 1})),
            ("history.renameSnapshot", json!({"snapshot": 1, "name": "x"})),
            ("history.deleteSnapshot", json!({"snapshot": 1})),
        ] {
            assert!(s.execute(cmd, p).is_err(), "{cmd}: no document");
        }
        let mut s = session();
        s.execute("history.newSnapshot", json!({})).unwrap();
        let (before, snaps) = (entries(&s), snapshots(&s));
        for (cmd, p) in [
            ("history.newSnapshot", json!({"from": "everything"})),
            ("history.newSnapshot", json!({"from": 3})),
            ("history.newSnapshot", json!({"name": 7})),
            ("history.selectSnapshot", json!({})),
            ("history.selectSnapshot", json!({"snapshot": 99})),
            ("history.selectSnapshot", json!({"snapshot": -1})),
            ("history.selectSnapshot", json!({"snapshot": "1"})),
            ("history.selectSnapshot", json!({"snapshot": f64::MAX})),
            ("history.renameSnapshot", json!({"snapshot": 1})),
            ("history.renameSnapshot", json!({"snapshot": 1, "name": "  "})),
            ("history.renameSnapshot", json!({"snapshot": 99, "name": "x"})),
            ("history.deleteSnapshot", json!({"snapshot": 99})),
            ("history.deleteSnapshot", json!({"snapshot": [1]})),
        ] {
            assert!(s.execute(cmd, p.clone()).is_err(), "{cmd} {p}");
            assert_eq!((entries(&s), snapshots(&s)), (before.clone(), snaps.clone()), "{cmd} {p} changed nothing");
        }
        // Current Layer needs an active layer.
        s.execute("select.deselectLayers", json!({})).unwrap();
        assert!(s.execute("history.newSnapshot", json!({"from": "currentLayer"})).is_err());
    }

    #[test]
    fn new_document_refuses_bad_params() {
        let mut s = Session::new();
        assert!(s.execute("history.newDocument", json!({})).is_err(), "no document");
        let mut s = session();
        for p in [json!({"state": -1}), json!({"state": 99}), json!({"state": "x"}), json!({"state": f64::NAN.to_string()})] {
            assert!(s.execute("history.newDocument", p.clone()).is_err(), "{p}");
        }
        assert_eq!(s.documents().len(), 1);
    }
}
