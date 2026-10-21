//! Organising brush presets (the Brushes panel's drag and drop and context menus): rename, move a
//! preset within or between groups and folders, move, rename and delete whole groups and folders.
//!
//! Groups are the panel's top-level folders ([`BrushPreset::group`]); a group can hold nested
//! folders ([`BrushPreset::folder`], a path inside the group, e.g. the folders of an imported
//! `.abr` file). The group commands take an optional `folder` path to act on a nested folder
//! instead of the whole group, and `moveGroup` can move a group or folder `into` another folder
//! or group, or out to the top level (dragging a folder onto a folder in Photoshop's panel). A
//! folder exists while a preset is in it (or below it).
//!
//! The library order is the panel order: groups appear in order of their first preset, and a
//! group's folders and presets in order of their first preset too. Every
//! change syncs to the preset store ([`crate::preset_store`]), whose index keeps the group order
//! and the order of every preset, built-ins included. A built-in that is renamed or moved to
//! another group becomes the user's preset (built-ins are regenerated with their own group).

use photocraft_paint::BrushPreset;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}
fn always(_: &Session) -> std::result::Result<(), String> {
    Ok(())
}

/// Longest preset or group name accepted.
pub const MAX_NAME: usize = 255;
/// Deepest folder path accepted inside a group (matches the `.abr` reader's limit).
pub const MAX_FOLDER_DEPTH: usize = photocraft_psd::abr::MAX_FOLDER_DEPTH;

fn text(p: &Value, k: &str, cmd: &str) -> Result<String> {
    let v = p.get(k).and_then(Value::as_str).map(str::trim).filter(|n| !n.is_empty()).ok_or_else(|| bad(cmd, format!("missing `{k}`")))?;
    if v.chars().count() > MAX_NAME {
        return Err(bad(cmd, format!("`{k}` is longer than {MAX_NAME} characters")));
    }
    Ok(v.to_string())
}

/// A group name param: may be empty (the ungrouped presets), never absurdly long.
fn group_param(p: &Value, k: &str, cmd: &str) -> Result<Option<String>> {
    match p.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(g)) if g.chars().count() <= MAX_NAME => Ok(Some(g.trim().to_string())),
        Some(_) => Err(bad(cmd, format!("`{k}` must be a group name (string, at most {MAX_NAME} characters)"))),
    }
}

/// A folder path param (`["Outer", "Inner"]`, inside the group): `None` when absent or null.
fn folder_param(p: &Value, cmd: &str) -> Result<Option<Vec<String>>> {
    let err = || bad(cmd, format!("`folder` must be a list of at most {MAX_FOLDER_DEPTH} folder names (non-empty strings of at most {MAX_NAME} characters)"));
    match p.get("folder") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(a)) if a.len() <= MAX_FOLDER_DEPTH => a
            .iter()
            .map(|v| v.as_str().map(str::trim).filter(|n| !n.is_empty() && n.chars().count() <= MAX_NAME).map(str::to_string).ok_or_else(err))
            .collect::<Result<Vec<_>>>()
            .map(Some),
        Some(_) => Err(err()),
    }
}

/// Is `x` in folder `f` of group `g`, directly or below it?
fn in_folder(x: &BrushPreset, g: &str, f: &[String]) -> bool {
    x.group == g && x.folder.starts_with(f)
}

/// The subfolders directly inside folder `parent` of group `g`, in panel order.
pub fn subfolder_order(presets: &[BrushPreset], g: &str, parent: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for x in presets.iter().filter(|x| in_folder(x, g, parent)) {
        if let Some(n) = x.folder.get(parent.len())
            && !out.contains(n)
        {
            out.push(n.clone());
        }
    }
    out
}

fn index_of(presets: &[BrushPreset], name: &str) -> Option<usize> {
    presets.iter().position(|x| x.name.eq_ignore_ascii_case(name))
}

fn find(s: &Session, p: &Value, k: &str, cmd: &str) -> Result<usize> {
    let name = text(p, k, cmd)?;
    index_of(&s.tools.presets, &name).ok_or_else(|| bad(cmd, format!("no brush preset named `{name}`")))
}

/// Groups in panel order (first appearance).
pub fn group_order(presets: &[BrushPreset]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for p in presets {
        if !out.contains(&p.group) {
            out.push(p.group.clone());
        }
    }
    out
}

fn position_in_group(presets: &[BrushPreset], i: usize) -> usize {
    presets.get(i).map_or(0, |pr| presets[..i].iter().filter(|x| x.group == pr.group).count())
}

fn rename(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "brush.presets.rename";
    let i = find(s, p, "name", cmd)?;
    let new = text(p, "newName", cmd)?;
    if index_of(&s.tools.presets, &new).is_some_and(|j| j != i) {
        return Err(bad(cmd, format!("a brush preset named `{new}` already exists")));
    }
    let pr = s.tools.presets.get_mut(i).ok_or_else(|| bad(cmd, "no such preset"))?;
    if pr.name != new {
        let was_current = s.tools.current_preset.as_ref().is_some_and(|c| c.eq_ignore_ascii_case(&pr.name));
        pr.name = new.clone();
        pr.builtin = false;
        if was_current {
            s.tools.current_preset = Some(new.clone());
            s.tools.brush_changed();
        }
        s.brush_presets_changed();
    }
    Ok(json!({ "name": new }))
}

fn move_preset(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "brush.presets.move";
    let i = find(s, p, "name", cmd)?;
    let before = match p.get("before") {
        None | Some(Value::Null) => None,
        Some(_) => {
            let b = find(s, p, "before", cmd)?;
            if b == i {
                return Err(bad(cmd, "`before` names the preset being moved"));
            }
            Some(b)
        }
    };
    let index = match p.get("index") {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.as_u64().ok_or_else(|| bad(cmd, "`index` must be a non-negative integer"))?),
    };
    let folder = folder_param(p, cmd)?;
    let group = group_param(p, "group", cmd)?;
    let lib = &mut s.tools.presets;
    let (Some(moving), before_pr) = (lib.get(i), before.and_then(|b| lib.get(b))) else {
        return Err(bad(cmd, "no such preset"));
    };
    let target = match (&group, before_pr) {
        (Some(g), _) => g.clone(),
        (None, Some(b)) => b.group.clone(),
        (None, None) => moving.group.clone(),
    };
    // The folder: as given, else `before`'s, else its own when it stays in its group.
    let target_folder = match (folder, before_pr) {
        (Some(f), _) => f,
        (None, Some(b)) => b.folder.clone(),
        (None, None) if moving.group == target => moving.folder.clone(),
        (None, None) => Vec::new(),
    };
    if let Some(b) = before_pr {
        if b.group != target {
            return Err(bad(cmd, format!("`before` is not in group `{target}`")));
        }
        if b.folder != target_folder {
            return Err(bad(cmd, "`before` is not in that folder"));
        }
    }
    let before_name = before_pr.map(|b| b.name.clone());
    let mut pr = lib.remove(i);
    if pr.group != target || pr.folder != target_folder {
        pr.group = target.clone();
        pr.folder = target_folder.clone();
        // A built-in moved elsewhere is the user's now (built-ins regenerate in their own group).
        pr.builtin = false;
    }
    let members: Vec<usize> = lib.iter().enumerate().filter(|(_, x)| x.group == target).map(|(k, _)| k).collect();
    // Appended: after the folder's last preset (or the group's, for a new folder).
    let end = lib.iter().rposition(|x| in_folder(x, &target, &target_folder)).or(members.last().copied()).map_or(lib.len(), |k| k + 1);
    let pos = match (before_name, index) {
        (Some(n), _) => index_of(lib, &n).unwrap_or(lib.len()),
        (None, Some(n)) => match members.get(n.min(usize::MAX as u64) as usize) {
            Some(&k) => k,
            None => members.last().map_or(lib.len(), |k| k + 1),
        },
        (None, None) => end,
    };
    let pos = pos.min(lib.len());
    let name = pr.name.clone();
    lib.insert(pos, pr);
    let at = position_in_group(lib, pos);
    s.brush_presets_changed();
    Ok(json!({ "name": name, "group": target, "folder": target_folder, "index": at }))
}

fn group_block(s: &Session, g: &str, cmd: &str) -> Result<()> {
    if s.tools.presets.iter().any(|x| x.group == g) { Ok(()) } else { Err(bad(cmd, format!("no brush preset group named `{g}`"))) }
}

/// The group and (possibly empty) folder path a group command acts on; both must exist.
fn target_folder(s: &Session, p: &Value, cmd: &str) -> Result<(String, Vec<String>)> {
    let g = group_param(p, "group", cmd)?.ok_or_else(|| bad(cmd, "missing `group`"))?;
    group_block(s, &g, cmd)?;
    let f = folder_param(p, cmd)?.unwrap_or_default();
    if !s.tools.presets.iter().any(|x| in_folder(x, &g, &f)) {
        return Err(bad(cmd, format!("no folder `{}` in brush preset group `{g}`", f.join("/"))));
    }
    Ok((g, f))
}

/// Move folder `f` (non-empty) of group `g` among its sibling folders, as a block.
fn move_folder(s: &mut Session, p: &Value, g: String, f: Vec<String>, cmd: &str) -> Result<Value> {
    let Some((leaf, parent)) = f.split_last() else { return Err(bad(cmd, "missing `folder`")) };
    let siblings = subfolder_order(&s.tools.presets, &g, parent);
    let before = match p.get("before") {
        None | Some(Value::Null) => None,
        Some(Value::String(b)) if b == leaf => return Err(bad(cmd, "`before` names the folder being moved")),
        Some(Value::String(b)) if siblings.contains(b) => Some(b.clone()),
        Some(_) => return Err(bad(cmd, "`before` must name a folder next to the one being moved")),
    };
    let index = match p.get("index") {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.as_u64().ok_or_else(|| bad(cmd, "`index` must be a non-negative integer"))?),
    };
    let lib = std::mem::take(&mut s.tools.presets);
    // Where the block sits among the other presets (its place when nothing else is in the parent).
    let start = lib.iter().position(|x| in_folder(x, &g, &f)).unwrap_or(0);
    let (block, mut rest): (Vec<BrushPreset>, Vec<BrushPreset>) = lib.into_iter().partition(|x| in_folder(x, &g, &f));
    let sibling = |name: &str| -> Vec<String> { parent.iter().cloned().chain(std::iter::once(name.to_string())).collect() };
    let first = |name: &str, rest: &[BrushPreset]| rest.iter().position(|x| in_folder(x, &g, &sibling(name))).unwrap_or(rest.len());
    let others: Vec<&String> = siblings.iter().filter(|n| *n != leaf).collect();
    let parent_end = |rest: &[BrushPreset]| rest.iter().rposition(|x| in_folder(x, &g, parent)).map_or(start, |k| k + 1);
    let pos = match (&before, index) {
        (Some(b), _) => first(b, &rest),
        (None, Some(n)) => match others.get(n.min(usize::MAX as u64) as usize) {
            Some(name) => first(name, &rest),
            None => parent_end(&rest),
        },
        (None, None) => parent_end(&rest),
    };
    let tail = rest.split_off(pos.min(rest.len()));
    rest.extend(block);
    rest.extend(tail);
    s.tools.presets = rest;
    s.brush_presets_changed();
    let at = subfolder_order(&s.tools.presets, &g, parent).iter().position(|x| x == leaf).unwrap_or(0);
    Ok(json!({ "group": g, "folder": f, "index": at }))
}

/// Where `into` puts a folder: `Some(None)` = the top level (it becomes a group), `Some(Some((group,
/// folder)))` = inside that folder of that group (`[]`: directly in the group). `None` without `into`.
type Dest = Option<(String, Vec<String>)>;

fn into_param(s: &Session, p: &Value, cmd: &str) -> Result<Option<Dest>> {
    let into = match p.get("into") {
        None | Some(Value::Null) => return Ok(None),
        Some(v @ Value::Object(_)) => v,
        Some(_) => return Err(bad(cmd, "`into` must be an object {group?, folder?}")),
    };
    let Some(g) = group_param(into, "group", cmd)? else {
        if folder_param(into, cmd)?.is_some_and(|f| !f.is_empty()) {
            return Err(bad(cmd, "`into.folder` needs `into.group`"));
        }
        return Ok(Some(None));
    };
    group_block(s, &g, cmd)?;
    let f = folder_param(into, cmd)?.unwrap_or_default();
    if !s.tools.presets.iter().any(|x| in_folder(x, &g, &f)) {
        return Err(bad(cmd, format!("no folder `{}` in brush preset group `{g}`", f.join("/"))));
    }
    Ok(Some(Some((g, f))))
}

/// Move folder `f` of group `g` (the whole group when `f` is empty) into `dest`, keeping its
/// content and the order of its presets: Photoshop's drag of a folder onto another folder. It
/// lands at the end of the destination, or before the sibling folder (group at the top level)
/// named by `before`.
fn move_into(s: &mut Session, p: &Value, g: String, f: Vec<String>, dest: Dest, cmd: &str) -> Result<Value> {
    if p.get("index").is_some_and(|v| !v.is_null()) {
        return Err(bad(cmd, "`index` can't be combined with `into` (use `before`)"));
    }
    let leaf = match f.last() {
        Some(l) => l.clone(),
        None if g.is_empty() => return Err(bad(cmd, "the ungrouped presets are not a folder that can be moved")),
        None => g.clone(),
    };
    let moving = |x: &BrushPreset| in_folder(x, &g, &f);
    let rest_of = |x: &BrushPreset| x.folder.get(f.len()..).unwrap_or_default().to_vec();
    let before = match p.get("before") {
        None | Some(Value::Null) => None,
        Some(Value::String(b)) => Some(b.trim().to_string()),
        Some(_) => return Err(bad(cmd, "`before` must be a folder name")),
    };
    // The destination's own folders and groups, without the one being moved.
    let others: Vec<BrushPreset> = s.tools.presets.iter().filter(|x| !moving(x)).cloned().collect();
    let (new_group, new_prefix, siblings) = match &dest {
        None => {
            if f.is_empty() {
                return Err(bad(cmd, "`into`: the group is already at the top level"));
            }
            (leaf.clone(), Vec::new(), group_order(&others))
        }
        Some((dg, df)) => {
            if *dg == g && df.starts_with(&f) {
                return Err(bad(cmd, "`into` names the folder being moved or one of its own folders"));
            }
            let mut prefix = df.clone();
            prefix.push(leaf.clone());
            (dg.clone(), prefix, subfolder_order(&others, dg, df))
        }
    };
    if siblings.contains(&leaf) {
        return Err(bad(cmd, format!("a folder named `{leaf}` already exists there")));
    }
    if let Some(b) = &before
        && !siblings.contains(b)
    {
        return Err(bad(cmd, format!("`before`: no folder named `{b}` there")));
    }
    let deepest = s.tools.presets.iter().filter(|x| moving(x)).map(|x| new_prefix.len() + rest_of(x).len()).max().unwrap_or(0);
    if deepest > MAX_FOLDER_DEPTH {
        return Err(bad(cmd, format!("folders can be nested at most {MAX_FOLDER_DEPTH} deep")));
    }
    let lib = std::mem::take(&mut s.tools.presets);
    let (mut block, mut rest): (Vec<BrushPreset>, Vec<BrushPreset>) = lib.into_iter().partition(|x| in_folder(x, &g, &f));
    for x in &mut block {
        let tail = rest_of(x);
        x.group.clone_from(&new_group);
        x.folder = new_prefix.iter().cloned().chain(tail).collect();
        x.builtin = false;
    }
    let pos = match (&dest, &before) {
        (None, Some(b)) => rest.iter().position(|x| x.group == *b),
        (None, None) => None,
        (Some((dg, df)), Some(b)) => {
            let sib: Vec<String> = df.iter().cloned().chain(std::iter::once(b.clone())).collect();
            rest.iter().position(|x| in_folder(x, dg, &sib))
        }
        (Some((dg, df)), None) => rest.iter().rposition(|x| in_folder(x, dg, df)).map(|k| k + 1),
    };
    let tail = rest.split_off(pos.unwrap_or(rest.len()).min(rest.len()));
    rest.extend(block);
    rest.extend(tail);
    s.tools.presets = rest;
    s.brush_presets_changed();
    let folder = new_prefix.get(..new_prefix.len().saturating_sub(1)).unwrap_or_default();
    let at = match &dest {
        None => group_order(&s.tools.presets).iter().position(|x| *x == new_group),
        Some(_) => subfolder_order(&s.tools.presets, &new_group, folder).iter().position(|x| *x == leaf),
    };
    Ok(json!({ "group": new_group, "folder": new_prefix, "index": at.unwrap_or(0) }))
}

fn move_group(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "brush.presets.moveGroup";
    let (g, f) = target_folder(s, p, cmd)?;
    if let Some(dest) = into_param(s, p, cmd)? {
        return move_into(s, p, g, f, dest, cmd);
    }
    if !f.is_empty() {
        return move_folder(s, p, g, f, cmd);
    }
    let before = group_param(p, "before", cmd)?;
    if let Some(b) = &before {
        group_block(s, b, cmd)?;
        if *b == g {
            return Err(bad(cmd, "`before` names the group being moved"));
        }
    }
    let index = match p.get("index") {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.as_u64().ok_or_else(|| bad(cmd, "`index` must be a non-negative integer"))?),
    };
    let lib = std::mem::take(&mut s.tools.presets);
    let (block, mut rest): (Vec<BrushPreset>, Vec<BrushPreset>) = lib.into_iter().partition(|x| x.group == g);
    let order = group_order(&rest);
    let first = |name: &str, rest: &[BrushPreset]| rest.iter().position(|x| x.group == name).unwrap_or(rest.len());
    let pos = match (&before, index) {
        (Some(b), _) => first(b, &rest),
        (None, Some(n)) => order.get(n.min(usize::MAX as u64) as usize).map_or(rest.len(), |name| first(name, &rest)),
        (None, None) => rest.len(),
    };
    let tail = rest.split_off(pos.min(rest.len()));
    rest.extend(block);
    rest.extend(tail);
    s.tools.presets = rest;
    s.brush_presets_changed();
    let at = group_order(&s.tools.presets).iter().position(|x| *x == g).unwrap_or(0);
    Ok(json!({ "group": g, "index": at }))
}

fn rename_group(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "brush.presets.renameGroup";
    let (g, f) = target_folder(s, p, cmd)?;
    let new = text(p, "newName", cmd)?;
    if let Some((leaf, parent)) = f.split_last() {
        if new != *leaf {
            if subfolder_order(&s.tools.presets, &g, parent).contains(&new) {
                return Err(bad(cmd, format!("a folder named `{new}` already exists there")));
            }
            let depth = parent.len();
            for x in s.tools.presets.iter_mut().filter(|x| in_folder(x, &g, &f)) {
                if let Some(n) = x.folder.get_mut(depth) {
                    n.clone_from(&new);
                }
                x.builtin = false;
            }
            s.brush_presets_changed();
        }
        let mut renamed = parent.to_vec();
        renamed.push(new);
        return Ok(json!({ "group": g, "folder": renamed }));
    }
    if new != g && s.tools.presets.iter().any(|x| x.group == new) {
        return Err(bad(cmd, format!("a group named `{new}` already exists")));
    }
    if new != g {
        for x in s.tools.presets.iter_mut().filter(|x| x.group == g) {
            x.group = new.clone();
            x.builtin = false;
        }
        s.brush_presets_changed();
    }
    Ok(json!({ "group": new }))
}

fn delete_group(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "brush.presets.deleteGroup";
    let (g, f) = target_folder(s, p, cmd)?;
    let before = s.tools.presets.len();
    s.tools.presets.retain(|x| !in_folder(x, &g, &f));
    if s.tools.current_preset.as_ref().is_some_and(|c| !s.tools.presets.iter().any(|x| x.name.eq_ignore_ascii_case(c))) {
        s.tools.current_preset = None;
        s.tools.brush_changed();
    }
    s.brush_presets_changed();
    Ok(json!({ "deleted": before - s.tools.presets.len(), "count": s.tools.presets.len() }))
}

macro_rules! spec {
    ($id:literal, $label:literal, $params:literal, $run:expr) => {
        CommandSpec { id: $id, label: $label, menu: &[], shortcut: None, params: $params, enabled: always, run: $run, journal: true }
    };
}

/// Brush preset organisation specs.
pub fn specs() -> Vec<CommandSpec> {
    vec![
        spec!("brush.presets.rename", "Rename Brush", r##"{"name":string,"newName":string} → {name}"##, rename),
        spec!(
            "brush.presets.move",
            "Move Brush",
            r##"{"name":string,"group":string?=its group ("" = ungrouped),"folder":[name]? (nested folder names in the group, outermost first; [] = the group itself)=before's folder, else its own if it stays in its group, else [],"before":name? (preset to land before, in that folder) | "index":n? (position in the group)=end of the folder} → {name, group, folder, index}"##,
            move_preset
        ),
        spec!(
            "brush.presets.moveGroup",
            "Move Brush Group",
            r##"{"group":string,"folder":[name]? (move this nested folder instead),"into":{"group":string?,"folder":[name]?}? (move it inside that folder of that group, [] = directly in the group; no group = out to the top level as a group of its own; the folder keeps its content),"before":group? (or sibling folder name, in its new place with `into`) | "index":n? (group or sibling folder position, without `into`)=end} → {group, folder?, index}"##,
            move_group
        ),
        spec!(
            "brush.presets.renameGroup",
            "Rename Brush Group",
            r##"{"group":string,"folder":[name]? (rename this nested folder instead),"newName":string} → {group, folder?}"##,
            rename_group
        ),
        spec!(
            "brush.presets.deleteGroup",
            "Delete Brush Group",
            r##"{"group":string,"folder":[name]? (delete this nested folder instead)} → {deleted, count}"##,
            delete_group
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(s: &Session, g: &str) -> Vec<String> {
        s.tools.presets.iter().filter(|x| x.group == g).map(|x| x.name.clone()).collect()
    }

    fn with_groups() -> Session {
        let mut s = Session::new();
        for (n, g) in [("A1", "Alpha"), ("A2", "Alpha"), ("A3", "Alpha"), ("B1", "Beta"), ("B2", "Beta")] {
            s.tools.presets.push(BrushPreset {
                name: n.into(),
                brush: Default::default(),
                builtin: false,
                group: g.into(),
                folder: Vec::new(),
                captures: Default::default(),
            });
        }
        s
    }

    #[test]
    fn move_within_and_between_groups() {
        let mut s = with_groups();
        let r = s.execute("brush.presets.move", json!({"name": "A3", "before": "A1"})).unwrap();
        assert_eq!(r, json!({"name": "A3", "group": "Alpha", "folder": [], "index": 0}));
        assert_eq!(names(&s, "Alpha"), ["A3", "A1", "A2"]);
        s.execute("brush.presets.move", json!({"name": "A3", "index": 99})).unwrap();
        assert_eq!(names(&s, "Alpha"), ["A1", "A2", "A3"]);
        s.execute("brush.presets.move", json!({"name": "A1", "index": 1})).unwrap();
        assert_eq!(names(&s, "Alpha"), ["A2", "A1", "A3"]);
        // Into another group, before a member (the group comes from `before`).
        s.execute("brush.presets.move", json!({"name": "A1", "before": "B2"})).unwrap();
        assert_eq!((names(&s, "Alpha"), names(&s, "Beta")), (vec!["A2".to_string(), "A3".into()], vec!["B1".to_string(), "A1".into(), "B2".into()]));
        // Into a group by name (appended), and into a new group.
        s.execute("brush.presets.move", json!({"name": "A2", "group": "Beta"})).unwrap();
        assert_eq!(names(&s, "Beta"), ["B1", "A1", "B2", "A2"]);
        s.execute("brush.presets.move", json!({"name": "A3", "group": "Gamma"})).unwrap();
        assert_eq!(group_order(&s.tools.presets).last().map(String::as_str), Some("Gamma"));
        assert!(names(&s, "Alpha").is_empty());
        // A built-in moved to another group becomes the user's.
        let b = s.tools.presets.iter().find(|x| x.builtin).unwrap().name.clone();
        s.execute("brush.presets.move", json!({"name": b, "group": "Beta", "index": 0})).unwrap();
        let moved = photocraft_paint::presets::find(&s.tools.presets, &b).unwrap();
        assert!(!moved.builtin && moved.group == "Beta");
        assert_eq!(names(&s, "Beta")[0], b);
    }

    #[test]
    fn groups_move_rename_and_delete() {
        let mut s = with_groups();
        let order = group_order(&s.tools.presets);
        let n = order.len();
        assert_eq!(&order[n - 2..], ["Alpha", "Beta"]);
        let r = s.execute("brush.presets.moveGroup", json!({"group": "Beta", "index": 0})).unwrap();
        assert_eq!(r["index"], 0);
        assert_eq!(group_order(&s.tools.presets)[0], "Beta");
        assert_eq!(names(&s, "Beta"), ["B1", "B2"], "a group moves as a block, in order");
        s.execute("brush.presets.moveGroup", json!({"group": "Alpha", "before": "Beta"})).unwrap();
        assert_eq!(&group_order(&s.tools.presets)[..2], ["Alpha", "Beta"]);
        s.execute("brush.presets.moveGroup", json!({"group": "Alpha"})).unwrap();
        assert_eq!(group_order(&s.tools.presets).last().map(String::as_str), Some("Alpha"));
        s.execute("brush.presets.renameGroup", json!({"group": "Alpha", "newName": "First"})).unwrap();
        assert_eq!(names(&s, "First"), ["A1", "A2", "A3"]);
        assert!(s.execute("brush.presets.renameGroup", json!({"group": "First", "newName": "Beta"})).is_err());
        let r = s.execute("brush.presets.deleteGroup", json!({"group": "First"})).unwrap();
        assert_eq!(r["deleted"], 3);
        assert!(names(&s, "First").is_empty());
    }

    #[test]
    fn rename_checks_names() {
        let mut s = with_groups();
        s.execute("brush.presets.rename", json!({"name": "a1", "newName": "Alpha One"})).unwrap();
        assert!(photocraft_paint::presets::find(&s.tools.presets, "Alpha One").is_some());
        assert!(s.execute("brush.presets.rename", json!({"name": "A2", "newName": "alpha one"})).is_err());
        // Case-only renames of itself are fine.
        s.execute("brush.presets.rename", json!({"name": "Alpha One", "newName": "ALPHA ONE"})).unwrap();
        let b = s.tools.presets.iter().find(|x| x.builtin).unwrap().name.clone();
        s.execute("brush.presets.rename", json!({"name": b, "newName": "My Round"})).unwrap();
        assert!(!photocraft_paint::presets::find(&s.tools.presets, "My Round").unwrap().builtin);
    }

    #[test]
    fn bad_params_fail_without_changes() {
        let mut s = with_groups();
        let before = s.tools.presets.clone();
        let long = "x".repeat(MAX_NAME + 1);
        for (id, p) in [
            ("brush.presets.move", json!({})),
            ("brush.presets.move", json!({"name": "Nope"})),
            ("brush.presets.move", json!({"name": "A1", "before": "A1"})),
            ("brush.presets.move", json!({"name": "A1", "before": "Nope"})),
            ("brush.presets.move", json!({"name": "A1", "group": "Alpha", "before": "B1"})),
            ("brush.presets.move", json!({"name": "A1", "index": -1})),
            ("brush.presets.move", json!({"name": "A1", "index": "first"})),
            ("brush.presets.move", json!({"name": "A1", "group": 7})),
            ("brush.presets.move", json!({"name": "A1", "group": long})),
            ("brush.presets.rename", json!({"name": "A1"})),
            ("brush.presets.rename", json!({"name": "A1", "newName": "  "})),
            ("brush.presets.rename", json!({"name": "A1", "newName": long})),
            ("brush.presets.moveGroup", json!({"group": "Nope"})),
            ("brush.presets.moveGroup", json!({"group": "Alpha", "before": "Alpha"})),
            ("brush.presets.moveGroup", json!({"group": "Alpha", "index": 1.5})),
            ("brush.presets.renameGroup", json!({"group": "Alpha"})),
            ("brush.presets.deleteGroup", json!({"group": null})),
            ("brush.presets.deleteGroup", json!({"group": "Nope"})),
        ] {
            assert!(s.execute(id, p.clone()).is_err(), "{id} {p}");
        }
        assert_eq!(s.tools.presets, before);
        // Huge indices clamp to the end.
        s.execute("brush.presets.move", json!({"name": "A1", "index": u64::MAX})).unwrap();
        s.execute("brush.presets.moveGroup", json!({"group": "Alpha", "index": u64::MAX})).unwrap();
    }

    /// Group "Set": S0 at its top level, folders Ink (I1, I2; subfolder Fine: F1) and Dry (D1).
    fn with_folders() -> Session {
        let mut s = Session::new();
        for (n, f) in [("S0", &[][..]), ("I1", &["Ink"][..]), ("F1", &["Ink", "Fine"][..]), ("I2", &["Ink"][..]), ("D1", &["Dry"][..])] {
            let folder = f.iter().map(|x| x.to_string()).collect();
            s.tools.presets.push(BrushPreset {
                name: n.into(),
                brush: Default::default(),
                builtin: false,
                group: "Set".into(),
                folder,
                captures: Default::default(),
            });
        }
        s
    }

    fn folder_of(s: &Session, n: &str) -> Vec<String> {
        photocraft_paint::presets::find(&s.tools.presets, n).unwrap().folder.clone()
    }

    #[test]
    fn presets_move_between_folders() {
        let mut s = with_folders();
        // `before` brings its folder.
        let r = s.execute("brush.presets.move", json!({"name": "D1", "before": "I2"})).unwrap();
        assert_eq!(r["folder"], json!(["Ink"]));
        assert_eq!(folder_of(&s, "D1"), ["Ink"]);
        // An explicit folder, appended at the end of it.
        s.execute("brush.presets.move", json!({"name": "S0", "folder": ["Ink", "Fine"]})).unwrap();
        assert_eq!(folder_of(&s, "S0"), ["Ink", "Fine"]);
        let pos = |n: &str| s.tools.presets.iter().position(|x| x.name == n).unwrap();
        assert_eq!(pos("S0"), pos("F1") + 1);
        // Within its group without a folder keeps its folder; to another group without one lands
        // at that group's top level; `folder: []` is the group's top level.
        s.execute("brush.presets.move", json!({"name": "S0", "index": 0})).unwrap();
        assert_eq!(folder_of(&s, "S0"), ["Ink", "Fine"]);
        s.execute("brush.presets.move", json!({"name": "S0", "group": "Other"})).unwrap();
        assert!(folder_of(&s, "S0").is_empty());
        s.execute("brush.presets.move", json!({"name": "I1", "folder": []})).unwrap();
        assert!(folder_of(&s, "I1").is_empty());
        // `before` in another folder than the one named.
        assert!(s.execute("brush.presets.move", json!({"name": "I1", "folder": ["Dry"], "before": "I2"})).is_err());
        assert_eq!(subfolder_order(&s.tools.presets, "Set", &[]), ["Ink"]);
    }

    #[test]
    fn folders_move_rename_and_delete() {
        let mut s = with_folders();
        assert_eq!(subfolder_order(&s.tools.presets, "Set", &[]), ["Ink", "Dry"]);
        // Move Dry before Ink: the whole folder moves as a block.
        let r = s.execute("brush.presets.moveGroup", json!({"group": "Set", "folder": ["Dry"], "before": "Ink"})).unwrap();
        assert_eq!(r, json!({"group": "Set", "folder": ["Dry"], "index": 0}));
        assert_eq!(subfolder_order(&s.tools.presets, "Set", &[]), ["Dry", "Ink"]);
        s.execute("brush.presets.moveGroup", json!({"group": "Set", "folder": ["Dry"]})).unwrap();
        assert_eq!(subfolder_order(&s.tools.presets, "Set", &[]), ["Ink", "Dry"]);
        s.execute("brush.presets.moveGroup", json!({"group": "Set", "folder": ["Dry"], "index": 0})).unwrap();
        assert_eq!(subfolder_order(&s.tools.presets, "Set", &[]), ["Dry", "Ink"]);
        let ink: Vec<&str> = s.tools.presets.iter().filter(|x| x.folder.first().is_some_and(|f| f == "Ink")).map(|x| x.name.as_str()).collect();
        assert_eq!(ink, ["I1", "F1", "I2"], "presets keep their order inside a moved folder");
        // Rename a nested folder: every preset below it follows.
        let r = s.execute("brush.presets.renameGroup", json!({"group": "Set", "folder": ["Ink"], "newName": "Inks"})).unwrap();
        assert_eq!(r["folder"], json!(["Inks"]));
        assert_eq!(folder_of(&s, "F1"), ["Inks", "Fine"]);
        assert!(s.execute("brush.presets.renameGroup", json!({"group": "Set", "folder": ["Inks"], "newName": "Dry"})).is_err());
        // Renaming the group keeps the folders.
        s.execute("brush.presets.renameGroup", json!({"group": "Set", "newName": "Kit"})).unwrap();
        assert_eq!(folder_of(&s, "F1"), ["Inks", "Fine"]);
        // Delete a nested folder: only the presets in it (and below) go.
        let r = s.execute("brush.presets.deleteGroup", json!({"group": "Kit", "folder": ["Inks"]})).unwrap();
        assert_eq!(r["deleted"], 3);
        assert!(photocraft_paint::presets::find(&s.tools.presets, "S0").is_some());
        assert!(photocraft_paint::presets::find(&s.tools.presets, "D1").is_some());
    }

    #[test]
    fn bad_folder_params_fail_without_changes() {
        let mut s = with_folders();
        let before = s.tools.presets.clone();
        let long = "x".repeat(MAX_NAME + 1);
        let deep: Vec<&str> = vec!["a"; MAX_FOLDER_DEPTH + 1];
        for (id, p) in [
            ("brush.presets.move", json!({"name": "S0", "folder": "Ink"})),
            ("brush.presets.move", json!({"name": "S0", "folder": [""]})),
            ("brush.presets.move", json!({"name": "S0", "folder": [7]})),
            ("brush.presets.move", json!({"name": "S0", "folder": [long]})),
            ("brush.presets.move", json!({"name": "S0", "folder": deep})),
            ("brush.presets.moveGroup", json!({"group": "Set", "folder": ["Nope"]})),
            ("brush.presets.moveGroup", json!({"group": "Set", "folder": ["Ink"], "before": "Ink"})),
            ("brush.presets.moveGroup", json!({"group": "Set", "folder": ["Ink"], "before": "Nope"})),
            ("brush.presets.moveGroup", json!({"group": "Set", "folder": ["Ink"], "before": 3})),
            ("brush.presets.moveGroup", json!({"group": "Set", "folder": ["Ink"], "index": -2})),
            ("brush.presets.renameGroup", json!({"group": "Set", "folder": ["Ink"], "newName": ""})),
            ("brush.presets.renameGroup", json!({"group": "Set", "folder": ["Ink", "Nope"], "newName": "x"})),
            ("brush.presets.deleteGroup", json!({"group": "Set", "folder": ["Dry", "Nope"]})),
            ("brush.presets.deleteGroup", json!({"group": "Nope", "folder": ["Dry"]})),
        ] {
            assert!(s.execute(id, p.clone()).is_err(), "{id} {p}");
        }
        assert_eq!(s.tools.presets, before);
        s.execute("brush.presets.moveGroup", json!({"group": "Set", "folder": ["Ink", "Fine"], "index": u64::MAX})).unwrap();
    }

    fn paths(s: &Session) -> Vec<String> {
        s.tools.presets.iter().filter(|x| !x.builtin).map(|x| format!("{}:{}:{}", x.group, x.folder.join("/"), x.name)).collect()
    }

    /// `with_folders` plus group "Other" holding O1, and folder Box (B1) in it.
    fn with_two_groups() -> Session {
        let mut s = with_folders();
        for (n, f) in [("O1", &[][..]), ("B1", &["Box"][..])] {
            let folder = f.iter().map(|x| x.to_string()).collect();
            s.tools.presets.push(BrushPreset {
                name: n.into(),
                brush: Default::default(),
                builtin: false,
                group: "Other".into(),
                folder,
                captures: Default::default(),
            });
        }
        s
    }

    #[test]
    fn folders_move_into_folders_and_groups() {
        let mut s = with_two_groups();
        // A nested folder into a sibling folder: its subtree follows, order kept.
        let r = s.execute("brush.presets.moveGroup", json!({"group": "Set", "folder": ["Dry"], "into": {"group": "Set", "folder": ["Ink", "Fine"]}})).unwrap();
        assert_eq!(r, json!({"group": "Set", "folder": ["Ink", "Fine", "Dry"], "index": 0}));
        assert_eq!(folder_of(&s, "D1"), ["Ink", "Fine", "Dry"]);
        // Into another group, before one of its folders.
        let r = s.execute("brush.presets.moveGroup", json!({"group": "Set", "folder": ["Ink"], "into": {"group": "Other"}, "before": "Box"})).unwrap();
        assert_eq!(r["folder"], json!(["Ink"]));
        assert_eq!(paths(&s), ["Set::S0", "Other::O1", "Other:Ink:I1", "Other:Ink/Fine:F1", "Other:Ink/Fine/Dry:D1", "Other:Ink:I2", "Other:Box:B1"]);
        // Out to the top level: the folder becomes a group of its own.
        let r = s.execute("brush.presets.moveGroup", json!({"group": "Other", "folder": ["Ink", "Fine"], "into": {}})).unwrap();
        assert_eq!(r["group"], "Fine");
        assert_eq!(folder_of(&s, "D1"), ["Dry"]);
        assert_eq!(photocraft_paint::presets::find(&s.tools.presets, "F1").unwrap().group, "Fine");
        // A whole group into another group's folder.
        s.execute("brush.presets.moveGroup", json!({"group": "Fine", "into": {"group": "Set"}})).unwrap();
        assert!(names(&s, "Fine").is_empty());
        assert_eq!(folder_of(&s, "D1"), ["Fine", "Dry"]);
        assert_eq!(subfolder_order(&s.tools.presets, "Set", &[]), ["Fine"]);
    }

    #[test]
    fn bad_moves_into_folders_fail_without_changes() {
        let mut s = with_two_groups();
        // A chain 31 deep under Set/Deep.
        let deep: Vec<String> = std::iter::once("Deep".to_string()).chain((0..30).map(|i| format!("d{i}"))).collect();
        s.tools.presets.push(BrushPreset {
            name: "Z".into(),
            brush: Default::default(),
            builtin: false,
            group: "Set".into(),
            folder: deep.clone(),
            captures: Default::default(),
        });
        s.tools.presets.push(BrushPreset {
            name: "Y".into(),
            brush: Default::default(),
            builtin: false,
            group: "Other".into(),
            folder: vec!["Dry".into()],
            captures: Default::default(),
        });
        let before = s.tools.presets.clone();
        let deep_json = json!(deep);
        for p in [
            // Into itself or its own descendant.
            json!({"group": "Set", "folder": ["Ink"], "into": {"group": "Set", "folder": ["Ink"]}}),
            json!({"group": "Set", "folder": ["Ink"], "into": {"group": "Set", "folder": ["Ink", "Fine"]}}),
            json!({"group": "Set", "into": {"group": "Set", "folder": ["Dry"]}}),
            json!({"group": "Set", "into": {"group": "Set"}}),
            // A group already at the top level; the ungrouped presets.
            json!({"group": "Set", "into": {}}),
            // A name clash at the destination.
            json!({"group": "Set", "folder": ["Dry"], "into": {"group": "Other"}}),
            json!({"group": "Other", "folder": ["Box"], "into": {"group": "Set", "folder": ["Ink"]}, "before": "Nope"}),
            json!({"group": "Set", "folder": ["Dry"], "into": {"group": "Set", "folder": ["Ink"]}, "index": 0}),
            // Missing destinations and bad shapes.
            json!({"group": "Set", "folder": ["Dry"], "into": {"group": "Nope"}}),
            json!({"group": "Set", "folder": ["Dry"], "into": {"group": "Other", "folder": ["Nope"]}}),
            json!({"group": "Set", "folder": ["Dry"], "into": {"folder": ["Box"]}}),
            json!({"group": "Set", "folder": ["Dry"], "into": "Other"}),
            json!({"group": "Set", "folder": ["Dry"], "into": {"group": 5}}),
            json!({"group": "Set", "folder": ["Dry"], "into": {"group": "Other"}, "before": 3}),
            // Deeper than MAX_FOLDER_DEPTH once moved (Ink/Fine is 2 deep: 2 + 31 > 32).
            json!({"group": "Set", "folder": ["Deep"], "into": {"group": "Set", "folder": ["Ink", "Fine"]}}),
            json!({"group": "Set", "folder": ["Ink"], "into": {"group": "Set", "folder": deep_json}}),
        ] {
            assert!(s.execute("brush.presets.moveGroup", p.clone()).is_err(), "{p}");
        }
        assert_eq!(s.tools.presets, before);
        // Ink/Fine into Set itself: a clash with nothing, so it moves up a level.
        s.execute("brush.presets.moveGroup", json!({"group": "Set", "folder": ["Ink", "Fine"], "into": {"group": "Set"}})).unwrap();
        assert_eq!(folder_of(&s, "F1"), ["Fine"]);
        // An ungrouped preset's group can't move.
        s.tools.presets.push(BrushPreset {
            name: "U".into(),
            brush: Default::default(),
            builtin: false,
            group: String::new(),
            folder: Vec::new(),
            captures: Default::default(),
        });
        assert!(s.execute("brush.presets.moveGroup", json!({"group": "", "into": {"group": "Set"}})).is_err());
    }
}
