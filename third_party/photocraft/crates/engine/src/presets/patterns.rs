//! Window › Patterns: folders over the pattern library (`pattern.*` owns the patterns
//! themselves), the selected pattern, and Pattern Fill layers made from it.
//!
//! Groups hold pattern **ids**. Library patterns that are in no group (newly defined or imported)
//! are shown in a trailing "Patterns" group; ids whose pattern was deleted are dropped.

use photocraft_doc::{Fill, LayerContent};
use serde_json::{Value, json};

use super::{Group, always, bad, edit_groups, group_index, req_str, str_param};
use crate::commands::CommandSpec;
use crate::{Result, Session};

const UNGROUPED: &str = "Patterns";

/// Built-in folders for the built-in procedural patterns.
pub fn builtin_groups() -> Vec<Group<String>> {
    let lib = crate::pattern_cmds::builtin();
    let ids = |names: &[&str]| -> Vec<String> { names.iter().filter_map(|n| lib.iter().find(|p| p.name == *n).map(|p| p.id.clone())).collect() };
    vec![Group::new("Geometric", ids(&["Checkerboard", "Diagonal Lines", "Dots", "Grid"])), Group::new("Textures", ids(&["Bricks", "Weave", "Paper Noise"]))]
}

/// Bring the groups in line with the library (see the module docs).
pub fn sync(s: &mut Session) {
    let lib: Vec<String> = s.patterns.items.iter().map(|p| p.id.clone()).collect();
    let groups = &mut s.presets.pattern_groups;
    for g in groups.iter_mut() {
        g.items.retain(|id| lib.contains(id));
    }
    let loose: Vec<String> = lib.iter().filter(|id| !groups.iter().any(|g| g.items.contains(id))).cloned().collect();
    if !loose.is_empty() {
        let gi = group_index(groups, Some(UNGROUPED));
        groups[gi].items.extend(loose);
    }
}

/// The selected pattern, if it is still in the library.
pub fn current(s: &Session) -> Option<&photocraft_doc::Pattern> {
    let id = s.presets.pattern.as_deref()?;
    s.patterns.items.iter().find(|p| p.id == id)
}

fn list(s: &mut Session, _: &Value) -> Result<Value> {
    sync(s);
    let groups: Vec<Value> = s
        .presets
        .pattern_groups
        .iter()
        .map(|g| {
            let pats: Vec<Value> = g
                .items
                .iter()
                .filter_map(|id| s.patterns.items.iter().find(|p| &p.id == id))
                .map(|p| json!({"id": p.id, "name": p.display_name(), "width": p.width, "height": p.height}))
                .collect();
            json!({"name": g.name, "patterns": pats})
        })
        .collect();
    Ok(json!({"groups": groups, "current": current(s).map(|p| p.id.clone())}))
}

fn pattern_key(s: &Session, p: &Value, cmd: &str) -> Result<photocraft_doc::Pattern> {
    let key = str_param(p, "pattern").unwrap_or("");
    crate::pattern_cmds::resolve(s, key).ok_or_else(|| bad(cmd, format!("no pattern \"{key}\" (see pattern.presets.list)")))
}

fn active_pattern_fill(s: &Session) -> Option<photocraft_doc::LayerId> {
    let d = s.active()?;
    let id = d.active_layer?;
    matches!(d.doc.layer(id)?.content, LayerContent::Fill(Fill::Pattern { .. })).then_some(id)
}

fn select(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "pattern.presets.select";
    let pat = pattern_key(s, p, CMD)?;
    s.presets.pattern = Some(pat.id.clone());
    s.presets.rev += 1;
    let mut out = json!({"pattern": pat.id, "name": pat.display_name()});
    if p.get("applyToLayer").and_then(Value::as_bool).unwrap_or(true)
        && let Some(id) = active_pattern_fill(s)
    {
        s.edit("Change Pattern Fill", |doc, _| {
            crate::pattern_cmds::ensure_in_doc(doc, &pat);
            if let Some(l) = doc.layer_mut(id)
                && let LayerContent::Fill(Fill::Pattern { name, id: pid, .. }) = &mut l.content
            {
                *name = pat.name.clone();
                *pid = pat.id.clone();
            }
            Ok(())
        })?;
        out["layer"] = json!(id.0);
    }
    Ok(out)
}

fn apply(s: &mut Session, p: &Value) -> Result<Value> {
    let pat = pattern_key(s, p, "pattern.presets.apply")?;
    let mut q = p.clone();
    q["pattern"] = json!(pat.id);
    super::call(s, "layer.newFillLayer.pattern", q)
}

fn new_preset(s: &mut Session, p: &Value) -> Result<Value> {
    let r = super::call(s, "edit.definePattern", p.clone())?;
    let id = r.get("pattern").and_then(Value::as_str).unwrap_or_default().to_string();
    sync(s);
    if let Some(g) = str_param(p, "group") {
        for gr in &mut s.presets.pattern_groups {
            gr.items.retain(|x| *x != id);
        }
        let gi = group_index(&mut s.presets.pattern_groups, Some(g));
        s.presets.pattern_groups[gi].items.push(id.clone());
    }
    s.presets_changed();
    Ok(json!({"pattern": id}))
}

fn edit(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "pattern.presets.edit";
    sync(s);
    let action = req_str(p, "action", CMD)?.to_string();
    // Rename/delete act on the library pattern (by id or name).
    let r = match action.as_str() {
        "rename" => super::call(s, "pattern.rename", json!({"pattern": req_str(p, "preset", CMD)?, "name": req_str(p, "name", CMD)?}))?,
        "delete" => {
            let keys: Vec<String> = match p.get("preset") {
                Some(Value::Array(a)) => a
                    .iter()
                    .map(|v| v.as_str().map(str::to_string).ok_or_else(|| bad(CMD, "`preset` must contain only pattern ids or names")))
                    .collect::<Result<_>>()?,
                _ => vec![req_str(p, "preset", CMD)?.to_string()],
            };
            if keys.is_empty() {
                return Err(bad(CMD, "missing `preset`"));
            }
            // Resolve every reference first. Deleting as we go left the library partly
            // removed when a later name was missing, and Undo cannot put a preset back (#4103).
            let mut ids = Vec::new();
            for k in &keys {
                // Only library patterns can be deleted here; a document-only pattern must
                // fail the whole batch up front rather than midway through it.
                let id = photocraft_doc::pattern::find(&s.patterns.items, k, k)
                    .map(|q| q.id.clone())
                    .ok_or_else(|| bad(CMD, format!("no library pattern \"{k}\"")))?;
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
            for id in &ids {
                super::call(s, "pattern.delete", json!({"pattern": id}))?;
            }
            json!({"deleted": ids.len()})
        }
        "deleteGroup" => {
            // Deleting a folder deletes its patterns, as in Photoshop.
            let g = req_str(p, "group", CMD)?;
            let i = s.presets.pattern_groups.iter().position(|x| x.name == g).ok_or_else(|| bad(CMD, format!("no group \"{g}\"")))?;
            let gone = s.presets.pattern_groups.remove(i);
            s.patterns.items.retain(|q| !gone.items.contains(&q.id));
            json!({"group": g, "deleted": gone.items.len()})
        }
        "move" => {
            let key = req_str(p, "preset", CMD)?;
            let id = crate::pattern_cmds::resolve(s, key).map(|q| q.id).ok_or_else(|| bad(CMD, format!("no pattern \"{key}\"")))?;
            let mut q = p.clone();
            q["preset"] = json!(id);
            edit_groups(&mut s.presets.pattern_groups, "move", &q, CMD)?
        }
        a => edit_groups(&mut s.presets.pattern_groups, a, p, CMD)?,
    };
    sync(s);
    s.presets_changed();
    Ok(r)
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "pattern.presets.list",
            label: "Pattern Presets",
            menu: &[],
            shortcut: None,
            params: "{} → {groups:[{name,patterns:[{id,name,width,height}]}],current}",
            enabled: always,
            run: list,
            journal: false,
        },
        CommandSpec {
            id: "pattern.presets.select",
            label: "Select Pattern",
            menu: &[],
            shortcut: None,
            params: r##"{"pattern":id|name,"applyToLayer":bool=true (also changes a selected Pattern Fill layer)} (Fill and new Pattern Fill layers default to it)"##,
            enabled: always,
            run: select,
            journal: true,
        },
        CommandSpec {
            id: "pattern.presets.apply",
            label: "New Pattern Fill Layer from Preset",
            menu: &[],
            shortcut: None,
            params: r##"{"pattern":id|name?=selected,"scale":1..1000=100,"angle":deg=0} → {layer}"##,
            enabled: super::has_doc,
            run: apply,
            journal: true,
        },
        CommandSpec {
            id: "pattern.presets.new",
            label: "New Pattern Preset",
            menu: &[],
            shortcut: None,
            params: r##"{"name":str?,"group":name?,"rect":[x0,y0,x1,y1]?} (Define Pattern from the selection/canvas into a group)"##,
            enabled: super::has_doc,
            run: new_preset,
            journal: true,
        },
        CommandSpec {
            id: "pattern.presets.edit",
            label: "Edit Pattern Presets",
            menu: &[],
            shortcut: None,
            params: super::GROUP_EDIT_PARAMS,
            enabled: always,
            run: edit,
            journal: true,
        },
    ]
}
