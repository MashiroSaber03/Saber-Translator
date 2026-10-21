//! Layers panel disclosure state: expanding and collapsing layer groups (#126).
//!
//! A group's open/closed state is document data ([`photocraft_doc::Group::expanded`], the PSD
//! section divider's open/closed folder type, saved in `.pcraft` too) but, as in Photoshop,
//! toggling it is a view change: no history step, and a clean document stays clean.

use std::sync::Arc;

use photocraft_doc::{Layer, LayerContent, LayerId};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

const ID: &str = "layer.setExpanded";

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: ID.into(), msg: msg.into() }
}

fn has_group(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    if d.doc.walk().iter().any(|(_, _, l)| l.is_group()) { Ok(()) } else { Err("the document has no layer groups".into()) }
}

fn opt_bool(p: &Value, key: &str) -> Result<Option<bool>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v.as_bool().map(Some).ok_or_else(|| bad(format!("`{key}` must be true or false"))),
    }
}

fn set_all(layers: &mut [Layer], expanded: bool) -> usize {
    let mut n = 0;
    for l in layers {
        if let LayerContent::Group(g) = &mut l.content {
            g.expanded = expanded;
            n += 1 + set_all(&mut g.children, expanded);
        }
    }
    n
}

/// `{"layer":id?,"expanded":bool?,"all":bool?}`: open or close a group (no `expanded`: toggle).
/// With `all`, every group in the document takes the new state (Photoshop's ⌥-click on a
/// disclosure triangle); without a layer, `all` toggles from "any group open".
fn set_expanded(s: &mut Session, p: &Value) -> Result<Value> {
    let want = opt_bool(p, "expanded")?;
    let all = opt_bool(p, "all")?.unwrap_or(false);
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let layer = match p.get("layer") {
        None | Some(Value::Null) if all => None,
        None | Some(Value::Null) => Some(st.active_layer.ok_or_else(|| bad("no layer given and no active layer"))?),
        Some(v) => Some(LayerId(v.as_u64().ok_or_else(|| bad("`layer` must be a layer id"))?)),
    };
    let current = match layer {
        Some(id) => match &st.doc.layer(id).ok_or(EngineError::NoLayer(id))?.content {
            LayerContent::Group(g) => g.expanded,
            _ => return Err(bad(format!("layer {} is not a group", id.0))),
        },
        None => st.doc.walk().iter().any(|(_, _, l)| matches!(&l.content, LayerContent::Group(g) if g.expanded)),
    };
    let expanded = want.unwrap_or(!current);
    let st = s.active_mut().ok_or(EngineError::NoDocument)?;
    // Copy-on-write: history snapshots keep their own state (layers share tiles, so this is cheap).
    let doc = Arc::make_mut(&mut st.doc);
    let changed = if all {
        set_all(&mut doc.layers, expanded)
    } else {
        let id = layer.ok_or_else(|| bad("no layer"))?;
        match doc.layer_mut(id).map(|l| &mut l.content) {
            Some(LayerContent::Group(g)) => {
                g.expanded = expanded;
                1
            }
            _ => return Err(EngineError::NoLayer(id)),
        }
    };
    Ok(json!({"expanded": expanded, "groups": changed}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: ID,
        label: "Expand/Collapse Group",
        menu: &[],
        shortcut: None,
        params: r##"{"layer":id?,"expanded":bool?,"all":bool?} (no expanded: toggle; all: every group; not an undo step)"##,
        enabled: has_group,
        run: set_expanded,
        journal: false,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> (Session, u64, u64) {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 32, "height": 32})).unwrap();
        s.execute("layer.new.layer", json!({"name": "deep"})).unwrap();
        let inner = s.execute("layer.groupLayers", json!({"name": "inner"})).unwrap()["layer"].as_u64().unwrap();
        let outer = s.execute("layer.groupLayers", json!({"layer": inner, "name": "outer"})).unwrap()["layer"].as_u64().unwrap();
        (s, inner, outer)
    }

    fn expanded(s: &Session, id: u64) -> bool {
        matches!(&s.active().unwrap().doc.layer(LayerId(id)).unwrap().content, LayerContent::Group(g) if g.expanded)
    }

    #[test]
    fn toggles_one_group_without_history_or_dirtying() {
        let (mut s, inner, outer) = session();
        let st = s.active_mut().unwrap();
        st.saved_revision = st.revision;
        let undo_steps = s.active().unwrap().history.entries().len();
        assert!(expanded(&s, inner) && expanded(&s, outer));
        assert_eq!(s.execute(ID, json!({"layer": inner})).unwrap()["expanded"], false);
        assert!(!expanded(&s, inner) && expanded(&s, outer));
        assert_eq!(s.active().unwrap().history.entries().len(), undo_steps, "no history step");
        assert_eq!(s.active().unwrap().saved_revision, s.active().unwrap().revision, "still clean");
        s.execute(ID, json!({"layer": inner, "expanded": true})).unwrap();
        assert!(expanded(&s, inner));
        // The inspect JSON reports it.
        s.execute(ID, json!({"layer": outer, "expanded": false})).unwrap();
        assert_eq!(s.execute("document.inspect", json!({})).unwrap()["layers"][0]["expanded"], false);
    }

    #[test]
    fn all_sets_every_group() {
        let (mut s, inner, outer) = session();
        s.execute(ID, json!({"layer": outer, "all": true})).unwrap();
        assert!(!expanded(&s, inner) && !expanded(&s, outer));
        assert_eq!(s.execute(ID, json!({"all": true})).unwrap()["groups"], 2);
        assert!(expanded(&s, inner) && expanded(&s, outer));
    }

    #[test]
    fn bad_params_fail_gracefully() {
        let (mut s, _, _) = session();
        let bg = s.active().unwrap().doc.layers[0].id.0;
        for p in [
            json!({"layer": bg}),
            json!({"layer": u64::MAX}),
            json!({"layer": "x"}),
            json!({"layer": -1}),
            json!({"expanded": "yes", "all": true}),
            json!({"all": 3}),
        ] {
            assert!(s.execute(ID, p.clone()).is_err(), "{p}");
        }
        let mut empty = Session::new();
        assert!(empty.execute(ID, json!({})).is_err());
        empty.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
        assert!(empty.execute(ID, json!({"all": true})).is_err(), "no groups: disabled");
    }
}
