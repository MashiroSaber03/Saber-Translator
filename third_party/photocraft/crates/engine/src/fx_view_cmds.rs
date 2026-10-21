//! Layers panel: showing or hiding a layer's effects list under its row (the triangle beside
//! the fx badge, #144). View state ([`crate::DocState::fx_collapsed`]): no history step, and a
//! clean document stays clean.

use photocraft_doc::LayerId;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

const ID: &str = "layer.setEffectsExpanded";

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: ID.into(), msg: msg.into() }
}

fn has_effects(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    if d.doc.walk().iter().any(|(_, _, l)| !l.effects.items.is_empty()) { Ok(()) } else { Err("no layer has effects".into()) }
}

fn opt_bool(p: &Value, key: &str) -> Result<Option<bool>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v.as_bool().map(Some).ok_or_else(|| bad(format!("`{key}` must be true or false"))),
    }
}

/// `{"layer":id?,"expanded":bool?,"all":bool?}`: show or hide a layer's effects list (no
/// `expanded`: toggle). With `all`, every layer with effects takes the new state (⌥-click).
fn set_expanded(s: &mut Session, p: &Value) -> Result<Value> {
    let want = opt_bool(p, "expanded")?;
    let all = opt_bool(p, "all")?.unwrap_or(false);
    let st = s.active_mut().ok_or(EngineError::NoDocument)?;
    let layer = match p.get("layer") {
        None | Some(Value::Null) if all => None,
        None | Some(Value::Null) => Some(st.active_layer.ok_or_else(|| bad("no layer given and no active layer"))?),
        Some(v) => Some(LayerId(v.as_u64().ok_or_else(|| bad("`layer` must be a layer id"))?)),
    };
    let with_fx: Vec<LayerId> = st.doc.walk().iter().filter(|(_, _, l)| !l.effects.items.is_empty()).map(|(_, _, l)| l.id).collect();
    let current = match layer {
        Some(id) => {
            let l = st.doc.layer(id).ok_or(EngineError::NoLayer(id))?;
            if l.effects.items.is_empty() {
                return Err(bad(format!("layer {} has no effects", id.0)));
            }
            !st.fx_collapsed.contains(&id)
        }
        None => with_fx.iter().any(|id| !st.fx_collapsed.contains(id)),
    };
    let expanded = want.unwrap_or(!current);
    let targets = if all { with_fx } else { layer.into_iter().collect() };
    // Forget ids no longer in the document so the list can't grow without bound.
    let doc = st.doc.clone();
    st.fx_collapsed.retain(|id| doc.layer(*id).is_some() && !targets.contains(id));
    if !expanded {
        st.fx_collapsed.extend(targets.iter().copied());
    }
    Ok(json!({"expanded": expanded, "layers": targets.len()}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: ID,
        label: "Expand/Collapse Effects",
        menu: &[],
        shortcut: None,
        params: r##"{"layer":id?,"expanded":bool?,"all":bool?} (no expanded: toggle; all: every layer with effects; view state, not an undo step)"##,
        enabled: has_effects,
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
        let a = s.execute("layer.new.layer", json!({"name": "a"})).unwrap()["layer"].as_u64().unwrap();
        s.execute("edit.fill", json!({"contents": "color", "color": "#ff0000"})).unwrap();
        s.execute("layer.layerStyle.dropShadow", json!({"layer": a})).unwrap();
        let b = s.execute("layer.new.layer", json!({"name": "b"})).unwrap()["layer"].as_u64().unwrap();
        s.execute("edit.fill", json!({"contents": "color", "color": "#00ff00"})).unwrap();
        s.execute("layer.layerStyle.stroke", json!({})).unwrap();
        (s, a, b)
    }

    fn open(s: &Session, id: u64) -> bool {
        !s.active().unwrap().fx_collapsed.contains(&LayerId(id))
    }

    #[test]
    fn toggles_one_layer_or_all_without_history() {
        let (mut s, a, b) = session();
        let st = s.active_mut().unwrap();
        st.saved_revision = st.revision;
        let steps = s.active().unwrap().history.entries().len();
        assert!(open(&s, a) && open(&s, b), "effects lists start open");
        assert_eq!(s.execute(ID, json!({"layer": a})).unwrap()["expanded"], false);
        assert!(!open(&s, a) && open(&s, b));
        assert_eq!(s.active().unwrap().history.entries().len(), steps, "no history step");
        assert_eq!(s.active().unwrap().saved_revision, s.active().unwrap().revision, "still clean");
        s.execute(ID, json!({"layer": a, "all": true})).unwrap();
        assert!(open(&s, a) && open(&s, b), "toggling an open-and-closed set from a closed layer opens all");
        s.execute(ID, json!({"all": true, "expanded": false})).unwrap();
        assert!(!open(&s, a) && !open(&s, b));
        s.execute(ID, json!({"layer": b, "expanded": true})).unwrap();
        assert!(!open(&s, a) && open(&s, b));
        assert_eq!(s.active().unwrap().fx_collapsed.len(), 1, "no duplicates");
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
        assert!(empty.execute(ID, json!({"all": true})).is_err(), "no effects: disabled");
    }
}
