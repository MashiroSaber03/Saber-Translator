//! Layers panel: the eyes on a layer's effects rows (#1622). The eye on the "Effects" row shows or
//! hides every effect of that layer; the eye on one effect's row shows or hides that effect. The
//! effects keep their settings either way, and each toggle is one history step.

use photocraft_doc::LayerId;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

const ID: &str = "layer.setEffectsVisible";

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: ID.into(), msg: msg.into() }
}

fn has_effects(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    if d.doc.walk().iter().any(|(_, _, l)| !l.effects.items.is_empty()) { Ok(()) } else { Err("no layer has effects".into()) }
}

/// `{"layer":id?,"index":n?,"visible":bool?}`: show or hide a layer's effects (no `visible`:
/// toggle). Without `index` it is the whole list (the "Effects" row); with it, the effect at that
/// position in the layer's list.
fn set_visible(s: &mut Session, p: &Value) -> Result<Value> {
    let want = match p.get("visible") {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.as_bool().ok_or_else(|| bad("`visible` must be true or false"))?),
    };
    let index = match p.get("index") {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.as_u64().and_then(|i| usize::try_from(i).ok()).ok_or_else(|| bad("`index` must be an effect position (0, 1, …)"))?),
    };
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let id = match p.get("layer") {
        None | Some(Value::Null) => st.active_layer.ok_or_else(|| bad("no layer given and no active layer"))?,
        Some(v) => LayerId(v.as_u64().ok_or_else(|| bad("`layer` must be a layer id"))?),
    };
    let l = st.doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    if l.effects.items.is_empty() {
        return Err(bad(format!("layer {} has no effects", id.0)));
    }
    let (current, name) = match index {
        Some(i) => {
            let e = l.effects.items.get(i).ok_or_else(|| bad(format!("layer {} has no effect at index {i}", id.0)))?;
            (e.enabled(), e.label())
        }
        None => (l.effects.enabled, "Layer Effects"),
    };
    let visible = want.unwrap_or(!current);
    if visible != current {
        let label = format!("{} {name}", if visible { "Show" } else { "Hide" });
        s.edit(&label, |doc, _| {
            let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
            match index {
                Some(i) => l.effects.items.get_mut(i).ok_or_else(|| bad(format!("layer {} has no effect at index {i}", id.0)))?.set_enabled(visible),
                None => l.effects.enabled = visible,
            }
            Ok(())
        })?;
    }
    Ok(json!({"layer": id.0, "index": index, "visible": visible}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: ID,
        label: "Show/Hide Effects",
        menu: &[],
        shortcut: None,
        params: r##"{"layer":id?,"index":n?,"visible":bool?} (no index: all of the layer's effects, the "Effects" row; index: the effect at that position in the layer's list; no visible: toggle)"##,
        enabled: has_effects,
        run: set_visible,
        journal: true,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A red layer with a drop shadow and a blue colour overlay, and the background (no effects).
    fn session() -> (Session, u64, u64) {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 32, "height": 32})).unwrap();
        let bg = s.active().unwrap().doc.layers[0].id.0;
        let a = s.execute("layer.new.layer", json!({"name": "a"})).unwrap()["layer"].as_u64().unwrap();
        s.execute("edit.fill", json!({"contents": "color", "color": "#ff0000"})).unwrap();
        s.execute("layer.layerStyle.dropShadow", json!({"layer": a})).unwrap();
        s.execute("layer.layerStyle.colorOverlay", json!({"layer": a, "color": "#0000ff"})).unwrap();
        (s, a, bg)
    }

    /// (the whole list is on, each effect is on).
    fn state(s: &Session, id: u64) -> (bool, Vec<bool>) {
        let d = s.active().unwrap();
        let fx = &d.doc.layer(LayerId(id)).unwrap().effects;
        (fx.enabled, fx.items.iter().map(|e| e.enabled()).collect())
    }

    fn steps(s: &Session) -> usize {
        s.active().unwrap().history.past_len()
    }

    fn flat(s: &Session) -> photocraft_compose::Buffer {
        photocraft_compose::flatten(&s.active().unwrap().doc)
    }

    #[test]
    fn the_effects_row_eye_hides_and_shows_the_whole_list() {
        let (mut s, a, _) = session();
        let n = steps(&s);
        assert_eq!(s.execute(ID, json!({"layer": a})).unwrap(), json!({"layer": a, "index": null, "visible": false}));
        assert_eq!(state(&s, a), (false, vec![true, true]), "each effect keeps its own state");
        assert_eq!(steps(&s), n + 1, "one history step");
        assert_eq!(s.active().unwrap().history.undo_label(), Some("Hide Layer Effects"));
        // The active layer is the default target.
        assert_eq!(s.execute(ID, json!({})).unwrap()["visible"], true);
        assert_eq!(state(&s, a), (true, vec![true, true]));
        assert_eq!(s.active().unwrap().history.undo_label(), Some("Show Layer Effects"));
    }

    #[test]
    fn one_effect_eye_hides_only_that_effect_and_undoes() {
        let (mut s, a, _) = session();
        assert_eq!(s.execute(ID, json!({"layer": a, "index": 1})).unwrap()["visible"], false);
        assert_eq!(state(&s, a), (true, vec![true, false]));
        assert_eq!(s.active().unwrap().history.undo_label(), Some("Hide Color Overlay"));
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(state(&s, a), (true, vec![true, true]));
        s.execute("edit.redo", json!({})).unwrap();
        assert_eq!(state(&s, a), (true, vec![true, false]));
        s.execute(ID, json!({"layer": a, "index": 1, "visible": true})).unwrap();
        assert_eq!(state(&s, a), (true, vec![true, true]));
    }

    #[test]
    fn hiding_changes_the_composite_and_keeps_the_settings() {
        let (mut s, a, _) = session();
        let before = s.active().unwrap().doc.layer(LayerId(a)).unwrap().effects.items.clone();
        let shown = flat(&s);
        s.execute(ID, json!({"layer": a, "visible": false})).unwrap();
        assert_ne!(flat(&s), shown, "hidden effects are not drawn");
        s.execute(ID, json!({"layer": a, "visible": true})).unwrap();
        assert_eq!(flat(&s), shown);
        assert_eq!(s.active().unwrap().doc.layer(LayerId(a)).unwrap().effects.items, before);
    }

    #[test]
    fn asking_for_the_current_state_adds_no_history_step() {
        let (mut s, a, _) = session();
        let n = steps(&s);
        assert_eq!(s.execute(ID, json!({"layer": a, "visible": true})).unwrap()["visible"], true);
        assert_eq!(s.execute(ID, json!({"layer": a, "index": 0, "visible": true})).unwrap()["visible"], true);
        assert_eq!(steps(&s), n);
    }

    #[test]
    fn bad_params_fail_gracefully() {
        let (mut s, a, bg) = session();
        let before = state(&s, a);
        for p in [
            json!({"layer": bg}),
            json!({"layer": u64::MAX}),
            json!({"layer": "x"}),
            json!({"layer": -1}),
            json!({"layer": a, "index": 2}),
            json!({"layer": a, "index": -1}),
            json!({"layer": a, "index": 1.5}),
            json!({"layer": a, "index": "0"}),
            json!({"layer": a, "index": u64::MAX}),
            json!({"layer": a, "visible": "yes"}),
            json!({"layer": a, "visible": 1}),
        ] {
            assert!(s.execute(ID, p.clone()).is_err(), "{p}");
        }
        assert_eq!(state(&s, a), before, "a refused call changes nothing");
        let mut empty = Session::new();
        assert!(empty.execute(ID, json!({})).is_err());
        empty.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
        assert!(empty.execute(ID, json!({})).is_err(), "no effects: disabled");
    }
}
