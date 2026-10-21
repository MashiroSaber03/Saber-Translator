//! Window › Layer Comps: capture, apply and manage layer comps (`layerComp.*`), plus
//! File › Export › Layer Comps to Files.
//!
//! A comp records every layer's visibility, position and appearance; its three options choose
//! which of them applying restores (see [`photocraft_doc::comps`]). Comps live in the document,
//! so every change here is one undoable history step. Commands that act on "the" comp take
//! `"comp": id | name` and default to the comp applied (or created) last.

use photocraft_doc::comps::{capture_states, layer_position, next_comp_id};
use photocraft_doc::{CompLayerState, Document, LayerComp, LayerId};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::file_cmds::{SaveOpts, join, native_doc, sanitize, save_doc, stem, str_param};
use crate::{EngineError, Result, Session};

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

fn has_comps(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    if d.doc.layer_comps.is_empty() { Err("the document has no layer comps".into()) } else { Ok(()) }
}

fn has_last_state(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    if d.doc.last_document_state.is_some() { Ok(()) } else { Err("no comp has been applied over the document state".into()) }
}

fn export_comps(s: &Session) -> std::result::Result<(), String> {
    native_doc(s)?;
    has_comps(s)
}

/// The comp a command acts on: `"comp"` as an id or a name, else the last applied comp.
fn comp_param(doc: &Document, p: &Value, cmd: &str) -> Result<u32> {
    match p.get("comp") {
        Some(v @ Value::Number(_)) => {
            let id = crate::commands::u32_id(cmd, "comp", v)?;
            doc.comp(id).map(|c| c.id).ok_or_else(|| bad(cmd, format!("no layer comp with id {id}")))
        }
        Some(Value::String(name)) => {
            doc.layer_comps.iter().find(|c| c.name == *name).map(|c| c.id).ok_or_else(|| bad(cmd, format!("no layer comp named \"{name}\"")))
        }
        Some(_) => Err(bad(cmd, "\"comp\" must be a comp id or name")),
        None => doc
            .last_applied_comp
            .filter(|id| doc.comp(*id).is_some())
            .or_else(|| doc.layer_comps.first().map(|c| c.id))
            .ok_or_else(|| bad(cmd, "the document has no layer comps")),
    }
}

fn index_of(doc: &Document, id: u32) -> usize {
    doc.layer_comps.iter().position(|c| c.id == id).unwrap_or(0)
}

fn flag(p: &Value, key: &str, default: bool) -> bool {
    p.get(key).and_then(Value::as_bool).unwrap_or(default)
}

/// Restores `comp`'s recorded state (only what its options select; `all` forces everything).
/// Layers deleted since the capture are skipped. Position-locked layers still move, as in
/// Photoshop, because the comp is not a user drag.
pub fn apply_comp(doc: &mut Document, comp: &LayerComp, all: bool) {
    let (vis, pos, app) = if all { (true, true, true) } else { (comp.apply_visibility, comp.apply_position, comp.apply_appearance) };
    // Walk order (parents before children): a moved artboard carries its children, which are
    // then placed absolutely, so the result does not depend on how far the parent moved.
    let order: Vec<LayerId> = doc.walk().into_iter().map(|(_, _, l)| l.id).collect();
    for id in order {
        let Some(st) = comp.state(id) else { continue };
        if pos
            && let Some((x, y)) = st.position
            && let Some((cx, cy)) = doc.layer(id).and_then(layer_position)
            && (x, y) != (cx, cy)
            // Recorded positions can come from files; a move that does not fit an `i32` is
            // skipped rather than wrapped onto some other position (#1017).
            && let (Some(dx), Some(dy)) = (x.checked_sub(cx), y.checked_sub(cy))
        {
            let snapshot = doc.clone();
            if let Some(l) = doc.layer_mut(id) {
                crate::commands::translate_layer(&snapshot, l, dx, dy);
                crate::vector_cmds::translate_vectors(&snapshot, l, f64::from(dx), f64::from(dy));
            }
        }
        let Some(l) = doc.layer_mut(id) else { continue };
        if vis && let Some(v) = st.visible {
            l.visible = v;
        }
        if app && let Some(a) = &st.appearance {
            l.blend = a.blend;
            l.opacity = a.opacity;
            l.fill_opacity = a.fill_opacity;
            l.effects = a.effects.clone();
        }
    }
}

fn new_comp(s: &mut Session, p: &Value) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let name = p.get("name").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| {
        let names: Vec<&str> = d.doc.layer_comps.iter().map(|c| c.name.as_str()).collect();
        // `names.len() + 1` candidates always include a free one.
        (1..=names.len() + 1).map(|n| format!("Layer Comp {n}")).find(|n| !names.contains(&n.as_str())).unwrap_or_else(|| "Layer Comp".into())
    });
    let comment = p.get("comment").and_then(Value::as_str).unwrap_or_default().to_string();
    let (v, pos, a) = (flag(p, "visibility", true), flag(p, "position", true), flag(p, "appearance", true));
    let id = s.edit("New Layer Comp", |doc, _| {
        let id = next_comp_id(doc);
        let comp = LayerComp { id, name, comment, apply_visibility: v, apply_position: pos, apply_appearance: a, states: capture_states(doc) };
        doc.layer_comps.push(comp);
        doc.last_applied_comp = Some(id);
        Ok(id)
    })?;
    Ok(json!({"comp": id}))
}

fn update_comp(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "layerComp.update";
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let what = p.get("what").and_then(Value::as_str).unwrap_or("all");
    if !matches!(what, "all" | "visibility" | "position" | "appearance") {
        return Err(bad(cmd, format!("unknown \"what\" `{what}` (all|visibility|position|appearance)")));
    }
    let every = p.get("comp").and_then(Value::as_str) == Some("*");
    let ids: Vec<u32> = if every { d.doc.layer_comps.iter().map(|c| c.id).collect() } else { vec![comp_param(&d.doc, p, cmd)?] };
    let label = match what {
        "visibility" => "Update Layer Comp Visibility",
        "position" => "Update Layer Comp Position",
        "appearance" => "Update Layer Comp Appearance",
        _ => "Update Layer Comp",
    };
    s.edit(label, |doc, _| {
        let now = capture_states(doc);
        for id in &ids {
            let Some(c) = doc.layer_comps.iter_mut().find(|c| c.id == *id) else { continue };
            if what == "all" {
                c.states = now.clone();
                continue;
            }
            // One property: refresh it on every live layer, keep the others as recorded.
            let mut states: Vec<CompLayerState> = Vec::with_capacity(now.len());
            for n in &now {
                let mut st = c.state(n.layer).cloned().unwrap_or(CompLayerState { layer: n.layer, visible: None, position: None, appearance: None });
                match what {
                    "visibility" => st.visible = n.visible,
                    "position" => st.position = n.position,
                    _ => st.appearance = n.appearance.clone(),
                }
                states.push(st);
            }
            c.states = states;
        }
        Ok(())
    })?;
    Ok(json!({"updated": ids}))
}

/// Whether the live layout still matches the properties the last applied comp restores.
/// Compare only on comp application: ordinary edits must not scan pixel bounds each frame.
fn matches_last_comp(doc: &Document, now: &[CompLayerState]) -> bool {
    let Some(comp) = doc.last_applied_comp.and_then(|id| doc.comp(id)) else { return false };
    let saved: std::collections::HashMap<_, _> = comp.states.iter().map(|st| (st.layer, st)).collect();
    now.len() == saved.len()
        && now.iter().all(|live| {
            saved.get(&live.layer).is_some_and(|saved| {
                (!comp.apply_visibility || saved.visible.is_none() || saved.visible == live.visible)
                    && (!comp.apply_position || saved.position.is_none() || saved.position == live.position)
                    && (!comp.apply_appearance || saved.appearance.is_none() || saved.appearance == live.appearance)
            })
        })
}

fn apply(s: &mut Session, id: u32, label: &str) -> Result<Value> {
    s.edit(label, |doc, _| {
        let comp = doc.comp(id).cloned().ok_or_else(|| EngineError::Other(format!("no layer comp with id {id}")))?;
        // Ordinary edits can leave last_applied_comp pointing at an outdated layout (#919).
        // Preserve the edited layout, but keep the original backup while cycling unedited comps.
        let states = capture_states(doc);
        if !matches_last_comp(doc, &states) {
            doc.last_document_state = Some(LayerComp {
                id: 0,
                name: "Last Document State".into(),
                comment: String::new(),
                apply_visibility: true,
                apply_position: true,
                apply_appearance: true,
                states,
            });
        }
        apply_comp(doc, &comp, false);
        doc.last_applied_comp = Some(id);
        Ok(())
    })?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let missing = d.doc.comp(id).map(|c| c.missing_layers(&d.doc).len()).unwrap_or(0);
    Ok(json!({"comp": id, "missingLayers": missing}))
}

fn apply_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let id = comp_param(&d.doc, p, "layerComp.apply")?;
    apply(s, id, "Apply Layer Comp")
}

fn step(s: &mut Session, delta: i64) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let n = d.doc.layer_comps.len() as i64;
    let cur = d.doc.last_applied_comp.and_then(|id| d.doc.layer_comps.iter().position(|c| c.id == id));
    let next = match cur {
        Some(i) => (i as i64 + delta).rem_euclid(n),
        None if delta > 0 => 0,
        None => n - 1,
    };
    let id = d.doc.layer_comps[next as usize].id;
    apply(s, id, "Apply Layer Comp")
}

fn delete(s: &mut Session, p: &Value) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let id = comp_param(&d.doc, p, "layerComp.delete")?;
    s.edit("Delete Layer Comp", |doc, _| {
        doc.layer_comps.retain(|c| c.id != id);
        if doc.last_applied_comp == Some(id) {
            doc.last_applied_comp = None;
        }
        Ok(())
    })?;
    Ok(json!({"deleted": id}))
}

fn duplicate(s: &mut Session, p: &Value) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let id = comp_param(&d.doc, p, "layerComp.duplicate")?;
    let new = s.edit("Duplicate Layer Comp", |doc, _| {
        let i = index_of(doc, id);
        let mut c = doc.layer_comps[i].clone();
        c.id = next_comp_id(doc);
        c.name = format!("{} copy", c.name);
        let nid = c.id;
        doc.layer_comps.insert(i + 1, c);
        Ok(nid)
    })?;
    Ok(json!({"comp": new}))
}

fn rename(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "layerComp.rename";
    let name = str_param(p, "name", cmd)?.to_string();
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let id = comp_param(&d.doc, p, cmd)?;
    s.edit("Rename Layer Comp", |doc, _| {
        let i = index_of(doc, id);
        doc.layer_comps[i].name = name;
        Ok(())
    })?;
    Ok(json!({"comp": id}))
}

fn set_comment(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "layerComp.setComment";
    let comment = p.get("comment").and_then(Value::as_str).ok_or_else(|| bad(cmd, "missing \"comment\""))?.to_string();
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let id = comp_param(&d.doc, p, cmd)?;
    s.edit("Layer Comp Options", |doc, _| {
        let i = index_of(doc, id);
        doc.layer_comps[i].comment = comment;
        Ok(())
    })?;
    Ok(json!({"comp": id}))
}

fn set_options(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "layerComp.setOptions";
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let id = comp_param(&d.doc, p, cmd)?;
    let get = |k: &str| p.get(k).and_then(Value::as_bool);
    let (v, pos, a, name) = (get("visibility"), get("position"), get("appearance"), p.get("name").and_then(Value::as_str).map(str::to_string));
    if v.is_none() && pos.is_none() && a.is_none() && name.is_none() && p.get("comment").is_none() {
        return Err(bad(cmd, "give \"visibility\", \"position\", \"appearance\", \"name\" or \"comment\""));
    }
    let comment = p.get("comment").and_then(Value::as_str).map(str::to_string);
    s.edit("Layer Comp Options", |doc, _| {
        let i = index_of(doc, id);
        let c = &mut doc.layer_comps[i];
        if let Some(v) = v {
            c.apply_visibility = v;
        }
        if let Some(v) = pos {
            c.apply_position = v;
        }
        if let Some(v) = a {
            c.apply_appearance = v;
        }
        if let Some(n) = name {
            c.name = n;
        }
        if let Some(n) = comment {
            c.comment = n;
        }
        Ok(())
    })?;
    Ok(json!({"comp": id}))
}

fn restore_last(s: &mut Session) -> Result<Value> {
    s.edit("Restore Last Document State", |doc, _| {
        let last = doc.last_document_state.clone().ok_or_else(|| EngineError::Other("no Last Document State".into()))?;
        apply_comp(doc, &last, true);
        doc.last_applied_comp = None;
        Ok(())
    })?;
    Ok(Value::Null)
}

/// Comps that refer to deleted layers (Photoshop's warning triangle). `"clear": true` drops
/// those stale states as one history step: from `"comp"` only when given (Clear Layer Comp
/// Warning), else from every comp (Clear All Layer Comp Warnings). `"comp"` also limits the
/// report to that comp; an unknown comp is an error and changes nothing.
fn update_warnings(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "layerComp.updateWarnings";
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let only = match p.get("comp") {
        None | Some(Value::Null) => None,
        Some(_) => Some(comp_param(&d.doc, p, cmd)?),
    };
    let warnings: Vec<Value> = d
        .doc
        .layer_comps
        .iter()
        .filter(|c| only.is_none_or(|id| c.id == id))
        .filter_map(|c| {
            let missing = c.missing_layers(&d.doc);
            (!missing.is_empty()).then(|| json!({"comp": c.id, "name": c.name, "missingLayers": missing.iter().map(|l| l.0).collect::<Vec<_>>()}))
        })
        .collect();
    let n = warnings.len();
    if flag(p, "clear", false) && n > 0 {
        let label = if only.is_some() { "Clear Layer Comp Warning" } else { "Clear All Layer Comp Warnings" };
        s.edit(label, |doc, _| {
            let live: std::collections::HashSet<LayerId> = doc.walk().into_iter().map(|(_, _, l)| l.id).collect();
            for c in doc.layer_comps.iter_mut().filter(|c| only.is_none_or(|id| c.id == id)) {
                c.states.retain(|st| live.contains(&st.layer));
            }
            Ok(())
        })?;
    }
    Ok(json!({"warnings": warnings, "count": n}))
}

fn list(s: &mut Session) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let comps: Vec<Value> = d
        .doc
        .layer_comps
        .iter()
        .map(|c| {
            json!({
                "id": c.id, "name": c.name, "comment": c.comment,
                "visibility": c.apply_visibility, "position": c.apply_position, "appearance": c.apply_appearance,
                "layers": c.states.len(), "missingLayers": c.missing_layers(&d.doc).len(),
            })
        })
        .collect();
    Ok(json!({"comps": comps, "lastApplied": d.doc.last_applied_comp, "hasLastDocumentState": d.doc.last_document_state.is_some()}))
}

/// File › Export › Layer Comps to Files: one file per comp, named
/// `<prefix>_<index>_<comp name>.<format>`.
fn comps_to_files(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "file.export.layerCompsToFiles";
    let dir = str_param(p, "dir", cmd).or_else(|_| str_param(p, "output", cmd))?.to_string();
    let format = p.get("format").and_then(Value::as_str).unwrap_or("png").trim_start_matches('.').to_ascii_lowercase();
    let save = SaveOpts::from_params(s, p, cmd)?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let prefix = p.get("prefix").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| stem(&d.doc.name));
    let doc = d.doc.clone();
    let chosen: Vec<u32> = match p.get("comps") {
        Some(Value::Array(a)) => a.iter().map(|v| crate::commands::u32_id(cmd, "comps", v)).collect::<Result<Vec<_>>>()?,
        _ if flag(p, "selectedOnly", false) => vec![comp_param(&doc, &json!({}), cmd)?],
        _ => doc.layer_comps.iter().map(|c| c.id).collect(),
    };
    let mut files = Vec::new();
    for (i, c) in doc.layer_comps.iter().enumerate().filter(|(_, c)| chosen.contains(&c.id)) {
        let mut one = (*doc).clone();
        apply_comp(&mut one, c, false);
        one.last_applied_comp = Some(c.id);
        let path = join(&dir, &format!("{}_{:04}_{}.{format}", sanitize(&prefix), i, sanitize(&c.name)));
        save_doc(&one, &path, save)?;
        files.push(path);
    }
    if files.is_empty() {
        return Err(bad(cmd, "no matching layer comps"));
    }
    Ok(json!({"files": files}))
}

pub fn specs() -> Vec<CommandSpec> {
    macro_rules! spec {
        ($id:expr, $label:expr, $menu:expr, $params:expr, $enabled:expr, $run:expr) => {
            CommandSpec { id: $id, label: $label, menu: $menu, shortcut: None, params: $params, enabled: $enabled, journal: true, run: $run }
        };
    }
    vec![
        spec!(
            "layerComp.new",
            "New Layer Comp…",
            &[],
            r##"{"name":str="Layer Comp N","comment":str="","visibility":bool=true,"position":bool=true,"appearance":bool=true} → {comp}"##,
            has_doc,
            new_comp
        ),
        spec!(
            "layerComp.update",
            "Update Layer Comp",
            &[],
            r##"{"comp":id|name|"*"? (default: last applied; "*" = all),"what":"all|visibility|position|appearance"="all"}"##,
            has_comps,
            update_comp
        ),
        spec!("layerComp.apply", "Apply Layer Comp", &[], r##"{"comp":id|name? (default: last applied)} → {comp, missingLayers}"##, has_comps, apply_cmd),
        spec!("layerComp.previous", "Apply Previous Layer Comp", &[], "{}", has_comps, |s, _| step(s, -1)),
        spec!("layerComp.next", "Apply Next Layer Comp", &[], "{}", has_comps, |s, _| step(s, 1)),
        spec!("layerComp.delete", "Delete Layer Comp", &[], r##"{"comp":id|name?}"##, has_comps, delete),
        spec!("layerComp.duplicate", "Duplicate Layer Comp", &[], r##"{"comp":id|name?} → {comp}"##, has_comps, duplicate),
        spec!("layerComp.rename", "Rename Layer Comp", &[], r##"{"comp":id|name?,"name":str}"##, has_comps, rename),
        spec!("layerComp.setComment", "Layer Comp Comment", &[], r##"{"comp":id|name?,"comment":str}"##, has_comps, set_comment),
        spec!(
            "layerComp.setOptions",
            "Layer Comp Options…",
            &[],
            r##"{"comp":id|name?,"visibility":bool?,"position":bool?,"appearance":bool?,"name":str?,"comment":str?}"##,
            has_comps,
            set_options
        ),
        spec!("layerComp.restoreLastDocumentState", "Restore Last Document State", &[], "{}", has_last_state, |s, _| restore_last(s)),
        spec!(
            "layerComp.updateWarnings",
            "Layer Comp Warnings",
            &[],
            r##"{"comp":id|name? (default: every comp; limits the report and the clear to that comp),"clear":bool=false (drop states of deleted layers)} → {warnings:[{comp,name,missingLayers}],count}"##,
            has_doc,
            update_warnings
        ),
        CommandSpec {
            journal: false,
            ..spec!(
                "layerComp.list",
                "List Layer Comps",
                &[],
                "{} → {comps:[{id,name,comment,visibility,position,appearance,layers,missingLayers}],lastApplied,hasLastDocumentState}",
                has_doc,
                |s, _| list(s)
            )
        },
        spec!(
            "file.export.layerCompsToFiles",
            "Layer Comps to Files…",
            &["File", "Export"],
            r##"{"dir":folder,"format":"png|jpg|psd|tiff|…"="png","prefix":str=document name,"selectedOnly":bool=false (only the last applied comp),"comps":[id]?,"quality":0..12?} → {files}"##,
            export_comps,
            comps_to_files
        ),
    ]
}

#[cfg(test)]
#[path = "comps_cmds/tests.rs"]
mod tests;
