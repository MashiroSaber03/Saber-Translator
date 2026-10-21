//! Editing a Pattern Fill layer: Photoshop's Pattern Fill dialog (pattern, Angle, Scale, Link with
//! Layer, Snap to Origin) commits one undoable content edit, independent of the layer's masks.

use photocraft_doc::{Document, Fill, LayerContent, Pattern};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, check_pixels_unlocked, layer_param};
use crate::{EngineError, Result, Session};

pub const SET: &str = "layer.fill.pattern.set";
/// Layer › New Fill Layer › Pattern… (`pattern_cmds`).
pub const CREATE: &str = "layer.newFillLayer.pattern";

/// Photoshop's Scale range in the Pattern Fill dialog (percent).
pub const SCALE_RANGE: (f64, f64) = (1.0, 1000.0);
/// Largest phase (origin offset) accepted, in pixels.
const MAX_PHASE: f64 = 1.0e7;

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: SET.into(), msg: msg.into() }
}

/// The pattern fill `p` asks for, each param optional and defaulting to `base`. Returns the new
/// fill and the pattern to copy into the document (when `pattern` was given).
fn edited(s: &Session, base: &Fill, p: &Value) -> Result<(Fill, Option<Pattern>)> {
    let Fill::Pattern { name, scale, id, angle, link, phase } = base else { return Err(bad("the target must be a Pattern Fill layer")) };
    let num = |k: &str, (min, max): (f64, f64)| -> Result<Option<f32>> {
        match p.get(k) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => v
                .as_f64()
                .filter(|v| v.is_finite() && (min..=max).contains(v))
                .map(|v| Some(v as f32))
                .ok_or_else(|| bad(format!("`{k}` must be a number within {min}..{max}"))),
        }
    };
    let (mut scale, mut angle, mut link, mut phase, mut pat) = (*scale, *angle, *link, *phase, None);
    if let Some(v) = p.get("pattern").filter(|v| !v.is_null()) {
        let key = v.as_str().filter(|k| !k.is_empty()).ok_or_else(|| bad("`pattern` is a pattern id or name"))?;
        pat = Some(crate::pattern_cmds::resolve(s, key).ok_or_else(|| bad(format!("no pattern \"{key}\" (see pattern.list)")))?);
    }
    if let Some(v) = num("scale", SCALE_RANGE)? {
        scale = v / 100.0;
    }
    if let Some(v) = num("angle", (-180.0, 180.0))? {
        angle = v;
    }
    match p.get("link") {
        None | Some(Value::Null) => {}
        Some(v) => link = v.as_bool().ok_or_else(|| bad("`link` is true or false"))?,
    }
    match p.get("phase") {
        None | Some(Value::Null) => {}
        Some(v) => {
            let a = v.as_array().filter(|a| a.len() == 2).ok_or_else(|| bad("`phase` is [x, y]"))?;
            let c = |i: usize| a.get(i).and_then(Value::as_f64).filter(|v| v.is_finite() && v.abs() <= MAX_PHASE).map(|v| v as f32);
            let (Some(x), Some(y)) = (c(0), c(1)) else { return Err(bad(format!("`phase` needs finite coordinates within ±{MAX_PHASE}"))) };
            phase = (x, y);
        }
    }
    let (name, id) = match &pat {
        Some(pat) => (pat.name.clone(), pat.id.clone()),
        None => (name.clone(), id.clone()),
    };
    Ok((Fill::Pattern { name, scale, id, angle, link, phase }, pat))
}

/// The target layer and the fill `p` gives it, checked against the active document.
fn target(s: &Session, p: &Value) -> Result<(photocraft_doc::LayerId, Fill, Fill, Option<Pattern>)> {
    if !p.is_object() || p.get("layer").is_some_and(|v| v.as_u64().is_none()) {
        return Err(bad("params must be an object and layer an unsigned id"));
    }
    let st = s.active().ok_or(EngineError::NoDocument)?;
    if let Some(document) = p.get("document")
        && document.as_u64() != Some(st.doc.id.0)
    {
        return Err(bad("the target document is no longer active"));
    }
    let id = layer_param(s, p)?;
    let layer = st.doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    let LayerContent::Fill(original @ Fill::Pattern { .. }) = &layer.content else { return Err(bad("the target must be a Pattern Fill layer")) };
    let (fill, pat) = edited(s, original, p)?;
    Ok((id, original.clone(), fill, pat))
}

fn set(s: &mut Session, p: &Value) -> Result<Value> {
    let (id, original, fill, pat) = target(s, p)?;
    if fill != original {
        let st = s.active().ok_or(EngineError::NoDocument)?;
        check_pixels_unlocked(&st.doc, id)?;
        s.edit("Pattern Fill", |doc, _| {
            if let Some(pat) = &pat {
                crate::pattern_cmds::ensure_in_doc(doc, pat);
            }
            doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?.content = LayerContent::Fill(fill.clone());
            Ok(())
        })?;
    }
    Ok(info(id, &fill))
}

fn info(layer: photocraft_doc::LayerId, fill: &Fill) -> Value {
    match fill {
        Fill::Pattern { scale, id, angle, link, phase, .. } => {
            json!({"layer": layer.0, "pattern": id, "scale": scale * 100.0, "angle": angle, "link": link, "phase": [phase.0, phase.1]})
        }
        _ => json!({"layer": layer.0}),
    }
}

/// The active document as [`SET`] or [`CREATE`] with `p` would leave it, without a history
/// step: the Pattern Fill dialog's live preview, so it shows exactly what OK commits.
pub fn preview(s: &Session, cmd: &str, p: &Value) -> Result<Document> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let mut doc = (*st.doc).clone();
    match cmd {
        SET => {
            let (id, _, fill, pat) = target(s, p)?;
            if let Some(pat) = &pat {
                crate::pattern_cmds::ensure_in_doc(&mut doc, pat);
            }
            doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?.content = LayerContent::Fill(fill);
        }
        CREATE => {
            let pat = crate::pattern_cmds::resolve_param(s, CREATE, p)?;
            let mut active = st.active_layer;
            crate::pattern_cmds::insert_fill_layer(&mut doc, &mut active, &pat, p)?;
        }
        other => return Err(EngineError::BadParams { cmd: other.into(), msg: "no Pattern Fill preview for this command".into() }),
    }
    Ok(doc)
}

pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: SET,
        label: "Pattern Fill…",
        menu: &[],
        shortcut: None,
        params: r##"{"layer":id? (default active; a Pattern Fill layer),"document":id? (reject a stale dialog),"pattern":id|name?,"scale":1..1000?,"angle":-180..180?,"link":bool?,"phase":[x,y]? (Snap to Origin: [0,0])}; omitted values keep the layer's"##,
        enabled: |s| s.active().map(|_| ()).ok_or_else(|| "no document open".into()),
        run: set,
        journal: true,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_doc::LayerId;
    use std::sync::Arc;

    fn session(depth: u64) -> (Session, LayerId) {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 40, "height": 30, "depth": depth})).unwrap();
        s.execute("select.rect", json!({"x": 2, "y": 2, "width": 20, "height": 10})).unwrap();
        let id = s.execute(CREATE, json!({"pattern": "Checkerboard"})).unwrap()["layer"].as_u64().unwrap();
        (s, LayerId(id))
    }

    fn fill(s: &Session, id: LayerId) -> Fill {
        let LayerContent::Fill(f) = &s.active().unwrap().doc.layer(id).unwrap().content else { panic!("not a fill") };
        f.clone()
    }

    #[test]
    fn edit_keeps_masks_is_one_step_and_undoes_at_every_depth() {
        for depth in [8, 16, 32] {
            let (mut s, id) = session(depth);
            let before = s.active().unwrap().doc.clone();
            let count = s.active().unwrap().history.entries().len();
            let r = s.execute(SET, json!({"layer": id.0, "pattern": "Bricks", "scale": 250, "angle": -45, "link": false, "phase": [3, -4]})).unwrap();
            assert_eq!(r["scale"], json!(250.0));
            let bricks = crate::pattern_cmds::resolve(&s, "Bricks").unwrap();
            assert_eq!(
                fill(&s, id),
                Fill::Pattern { name: bricks.name.clone(), scale: 2.5, id: bricks.id.clone(), angle: -45.0, link: false, phase: (3.0, -4.0) }
            );
            let st = s.active().unwrap();
            assert_eq!(st.history.entries().len(), count + 1);
            assert!(st.doc.patterns.iter().any(|p| p.id == bricks.id), "the new pattern is copied into the document");
            assert_eq!(st.doc.layer(id).unwrap().mask, before.layer(id).unwrap().mask, "the selection mask is kept");
            // Omitted values keep the layer's; Snap to Origin zeroes the phase.
            s.execute(SET, json!({"phase": [0, 0]})).unwrap();
            let Fill::Pattern { scale, angle, phase, link, .. } = fill(&s, id) else { panic!() };
            assert_eq!((scale, angle, phase, link), (2.5, -45.0, (0.0, 0.0), false));
            s.undo();
            s.undo();
            assert_eq!(*s.active().unwrap().doc, *before);
            s.redo();
            assert_eq!(fill(&s, id), Fill::Pattern { name: bricks.name.clone(), scale: 2.5, id: bricks.id, angle: -45.0, link: false, phase: (3.0, -4.0) });
            let count = s.active().unwrap().history.entries().len();
            s.execute(SET, json!({"scale": 250, "angle": -45})).unwrap();
            assert_eq!(s.active().unwrap().history.entries().len(), count, "an unchanged fill is not a history step");
        }
    }

    #[test]
    fn preview_matches_what_the_command_commits() {
        let (mut s, id) = session(8);
        let p = json!({"layer": id.0, "pattern": "Dots", "scale": 50, "angle": 30});
        let (before, count) = (s.active().unwrap().doc.clone(), s.active().unwrap().history.entries().len());
        let shown = preview(&s, SET, &p).unwrap();
        assert!(Arc::ptr_eq(&before, &s.active().unwrap().doc));
        assert_eq!(s.active().unwrap().history.entries().len(), count, "previewing is not an edit");
        s.execute(SET, p).unwrap();
        assert_eq!(shown, *s.active().unwrap().doc);
        let p = json!({"pattern": "Grid", "scale": 300, "angle": 90, "link": false});
        let shown = preview(&s, CREATE, &p).unwrap();
        s.execute(CREATE, p).unwrap();
        // Layer ids are allocated per process, so the preview's new layer has its own id: compare
        // what the documents look like.
        let all = photocraft_geom::Rect::new(0, 0, 40, 30);
        assert_eq!(photocraft_compose::render(&shown, all).px, photocraft_compose::render(&s.active().unwrap().doc, all).px);
        assert_eq!(shown.layers.len(), s.active().unwrap().doc.layers.len());
        assert!(preview(&s, "edit.fill", &json!({})).is_err());
    }

    #[test]
    fn bad_params_and_targets_are_errors_not_panics() {
        let (mut s, id) = session(8);
        let before = s.active().unwrap().doc.clone();
        for p in [
            json!(null),
            json!([1, 2]),
            json!({"layer": -1}),
            json!({"layer": "x"}),
            json!({"layer": u64::MAX}),
            json!({"document": 999_999}),
            json!({"pattern": "nope"}),
            json!({"pattern": ""}),
            json!({"pattern": 5}),
            json!({"scale": 0}),
            json!({"scale": 1001}),
            json!({"scale": "big"}),
            json!({"angle": 181}),
            json!({"angle": -1e300}),
            json!({"link": "yes"}),
            json!({"phase": [1]}),
            json!({"phase": [1, "x"]}),
            json!({"phase": [1e30, 0]}),
            json!({"phase": {"x": 1}}),
        ] {
            assert!(s.execute(SET, p.clone()).is_err(), "{p}");
            assert!(preview(&s, SET, &p).is_err(), "{p}");
        }
        s.execute("layer.new.layer", json!({})).unwrap();
        assert!(s.execute(SET, json!({"scale": 50})).is_err(), "a pixel layer is not a Pattern Fill");
        s.execute("layer.newFillLayer.solidColor", json!({})).unwrap();
        assert!(s.execute(SET, json!({"scale": 50})).is_err(), "a Solid Color Fill is not a Pattern Fill");
        assert_eq!(s.active().unwrap().doc.layer(id), before.layer(id));
        assert!(Session::new().execute(SET, json!({})).is_err(), "no document");
    }

    #[test]
    fn locked_layer_is_refused() {
        let (mut s, id) = session(8);
        s.execute("layer.lockLayers", json!({"pixels": true})).unwrap();
        assert!(s.execute(SET, json!({"layer": id.0, "scale": 50})).is_err());
    }
}
