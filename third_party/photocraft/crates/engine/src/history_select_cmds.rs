//! History panel › click a state: `history.selectState` returns the document to that state in one
//! step (#2966), as Photoshop does. The panel used to run Edit › Undo / Redo once per state: each
//! bumped the document revision, so the canvas could no longer tell what the jump changed and
//! recomposited (and re-uploaded) the whole document, plus N rounds of per-command bookkeeping.
//!
//! States are numbered as the History panel lists them, like `history.deleteState`: 0 is the
//! oldest state still held, `past` (the number of undo states) is the current one, and the redo
//! states (dimmed in the panel) follow it. Jumping back leaves the skipped states redoable;
//! jumping forward redoes them. Nothing is discarded until the next edit.

use serde_json::{Value, json};

use crate::commands::{CommandSpec, int};
use crate::{EngineError, Result, Session};

const CMD: &str = "history.selectState";

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: CMD.into(), msg: msg.into() }
}

fn select_state(s: &mut Session, p: &Value) -> Result<Value> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let past = st.history.past_len();
    let last = past.saturating_add(st.history.redo_labels().count());
    let v = match p.get("state") {
        None | Some(Value::Null) => return Err(bad("missing `state` (the History entry to return to, 0 = oldest)")),
        Some(_) => int(p, "state").ok_or_else(|| bad("`state` must be a number"))?,
    };
    let target = usize::try_from(v).ok().filter(|i| *i <= last).ok_or_else(|| bad(format!("`state` = {v} is not a state (0..={last})")))?;
    // A background job is computing from the current state: it must not move under it.
    if s.active_job().is_some() {
        return Err(EngineError::Other("the history can't move while a command is still running on this document".into()));
    }
    let st = s.active_mut().ok_or(EngineError::NoDocument)?;
    if target == past {
        return Ok(json!({ "state": target, "changed": false }));
    }
    st.coalesce = None;
    // Step the history itself (pointer moves, no pixels touched), then make the reached state
    // current once: one revision, one damage rect for the canvas.
    let start = st.doc.clone();
    let mut doc = start.clone();
    let mut layers = None;
    let mut at = past;
    while at != target {
        let step = if target < at { st.history.undo(doc.clone()) } else { st.history.redo(doc.clone()) };
        let Some((d, l)) = step else { break };
        doc = d;
        layers = Some(l);
        at = if target < at { at - 1 } else { at + 1 };
    }
    let Some(layers) = layers else {
        return Err(EngineError::Other("the history can't step right now".into()));
    };
    // Save As changes file identity outside history, as Edit › Undo keeps it.
    if doc.name != start.name {
        std::sync::Arc::make_mut(&mut doc).name.clone_from(&start.name);
    }
    let damage = crate::layer_multi_cmds::step_damage(&start, &doc);
    st.doc = doc;
    crate::restore_target(st, layers);
    st.revision += 1;
    st.last_damage = damage;
    if at != target {
        // Not reached (the bounds were checked): the document is consistent at the state reached.
        return Err(EngineError::Other(format!("the history stopped at state {at}")));
    }
    Ok(json!({ "state": target, "changed": true, "current": st.history.entries().last() }))
}

fn can_select(s: &Session) -> std::result::Result<(), String> {
    let st = s.active().ok_or_else(|| "no document open".to_string())?;
    if st.history.can_undo() || st.history.can_redo() { Ok(()) } else { Err("the history holds only one state".into()) }
}

pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Select History State",
        menu: &[],
        shortcut: None,
        params: r##"{"state":index} (History entry as the panel lists them: 0 = oldest, then the current state, then the redo states) → {state,changed,current?}: returns the document to that state in one step; the states after it stay redoable"##,
        enabled: can_select,
        run: select_state,
        journal: true,
    }]
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    /// A document with four steps after Open, each targeting different layers.
    fn session() -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 64, "height": 32})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("edit.fill", json!({"color": "#00ff00"})).unwrap();
        s
    }

    type Look = (Arc<photocraft_doc::Document>, Option<photocraft_doc::LayerId>, Vec<photocraft_doc::LayerId>, Vec<String>, Vec<String>);

    /// What the panel can see of a session: the document, the targeted layers, the entries and
    /// the redo states.
    fn look(s: &Session) -> Look {
        let st = s.active().unwrap();
        (st.doc.clone(), st.active_layer, st.selected_layers.clone(), st.history.entries(), st.history.redo_labels().map(str::to_string).collect())
    }

    /// Step from the current state to `target` with Edit › Undo / Redo, note what that gives, and
    /// step back.
    fn stepped(s: &mut Session, target: usize) -> Look {
        let current = s.active().unwrap().history.past_len();
        let (there, back) = if target < current { ("edit.undo", "edit.redo") } else { ("edit.redo", "edit.undo") };
        let n = target.abs_diff(current);
        for _ in 0..n {
            s.execute(there, json!({})).unwrap();
        }
        let seen = look(s);
        for _ in 0..n {
            s.execute(back, json!({})).unwrap();
        }
        seen
    }

    fn same(a: &Look, b: &Look) {
        assert!(Arc::ptr_eq(&a.0, &b.0), "the same document");
        assert_eq!((&a.1, &a.2, &a.3, &a.4), (&b.1, &b.2, &b.3, &b.4));
    }

    #[test]
    fn jumping_back_and_forward_matches_stepping() {
        for target in 0..=4usize {
            for from_back in [0usize, 2, 4] {
                // Start `from_back` states before the end, then jump to `target`.
                let mut s = session();
                for _ in 0..from_back {
                    s.execute("edit.undo", json!({})).unwrap();
                }
                let current = 4 - from_back;
                let expected = stepped(&mut s, target);
                let rev = s.active().unwrap().revision;
                let r = s.execute(CMD, json!({"state": target})).unwrap();
                assert_eq!(r["state"], json!(target));
                assert_eq!(r["changed"], json!(target != current));
                same(&look(&s), &expected);
                assert_eq!(s.active().unwrap().history.past_len(), target);
                let bumps = s.active().unwrap().revision - rev;
                assert_eq!(bumps, u64::from(target != current), "{from_back} → {target}: one revision for the jump");
            }
        }
    }

    #[test]
    fn a_jump_reports_only_the_pixels_it_changed() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 400, "height": 300})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("select.rect", json!({"x": 10, "y": 10, "width": 20, "height": 20})).unwrap();
        s.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
        s.execute("select.rect", json!({"x": 40, "y": 30, "width": 10, "height": 10})).unwrap();
        s.execute("edit.fill", json!({"color": "#0000ff"})).unwrap();
        let past = s.active().unwrap().history.past_len();
        s.execute(CMD, json!({"state": past - 4})).unwrap();
        let damage = s.active().unwrap().last_damage.expect("a bounded damage rect, not a full recomposite");
        assert!(damage.width() < 400 && damage.height() < 300, "{damage:?}");
        // And forward again.
        s.execute(CMD, json!({"state": past})).unwrap();
        assert!(s.active().unwrap().last_damage.is_some());
    }

    #[test]
    fn redo_states_survive_a_jump_and_go_with_the_next_edit() {
        let mut s = session();
        let all = s.active().unwrap().history.entries();
        s.execute(CMD, json!({"state": 1})).unwrap();
        let (_, _, _, entries, redo) = look(&s);
        assert_eq!(entries, all[..2].to_vec());
        assert_eq!(redo, all[2..].to_vec(), "the later states are dimmed, next one first");
        assert!(s.is_enabled(CMD));
        s.execute("edit.fill", json!({"color": "#0000ff"})).unwrap();
        assert!(!s.active().unwrap().history.can_redo(), "a new step discards them");
    }

    #[test]
    fn select_state_restores_the_layers_the_state_targeted() {
        let mut s = session();
        let bottom = s.active().unwrap().doc.layers.first().unwrap().id;
        // Target the bottom layer, then jump back to the first fill: its layer comes back, as
        // with Undo.
        s.select_layer(bottom).unwrap();
        let expected = stepped(&mut s, 2);
        s.select_layer(bottom).unwrap();
        s.execute(CMD, json!({"state": 2})).unwrap();
        same(&look(&s), &expected);
        assert_ne!(s.active().unwrap().active_layer, Some(bottom));
    }

    #[test]
    fn a_floating_piece_goes_back_and_the_jump_still_happens() {
        let mut s = session();
        s.execute("select.rect", json!({"x": 2, "y": 2, "width": 8, "height": 8})).unwrap();
        s.execute("select.float", json!({"dx": 5, "dy": 0})).unwrap();
        let all = s.active().unwrap().history.entries();
        let doc = s.active().unwrap().doc.clone();
        assert!(s.active().unwrap().floating.is_some());
        s.execute(CMD, json!({"state": 1})).unwrap();
        let st = s.active().unwrap();
        assert!(st.floating.is_none(), "the piece went back, it wasn't dropped");
        assert_eq!(st.history.past_len(), 1);
        let redo: Vec<String> = st.history.redo_labels().map(str::to_string).collect();
        assert_eq!(redo, all[2..].to_vec(), "no Drop step was added");
        // Forward to the state the piece was floated on: the document as it was, without the piece.
        s.execute(CMD, json!({"state": all.len() - 1})).unwrap();
        assert!(Arc::ptr_eq(&s.active().unwrap().doc, &doc));
    }

    #[test]
    fn select_state_refuses_bad_params_and_changes_nothing() {
        let mut s = Session::new();
        assert!(s.execute(CMD, json!({"state": 0})).is_err(), "no document");
        assert!(!s.is_enabled(CMD));
        s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
        assert!(!s.is_enabled(CMD), "only the opening state");
        let mut s = session();
        s.execute("edit.undo", json!({})).unwrap();
        let before = look(&s);
        let rev = s.active().unwrap().revision;
        for p in [
            json!({}),
            json!({"state": null}),
            json!({"state": -1}),
            json!({"state": 5}),
            json!({"state": 99}),
            json!({"state": i64::MAX}),
            json!({"state": f64::MAX}),
            json!({"state": "x"}),
            json!({"state": "1"}),
            json!({"state": [1]}),
            json!({"state": {"i": 1}}),
            json!({"state": true}),
            json!([1]),
        ] {
            assert!(s.execute(CMD, p.clone()).is_err(), "{p}");
            let now = look(&s);
            assert!(Arc::ptr_eq(&now.0, &before.0) && now.3 == before.3 && now.4 == before.4, "{p} changed nothing");
        }
        assert_eq!(s.active().unwrap().revision, rev);
        // The last redo state is reachable.
        assert!(s.execute(CMD, json!({"state": 4})).is_ok());
    }
}
