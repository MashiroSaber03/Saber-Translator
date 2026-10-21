//! Preset panels: Window › Gradients, Patterns, Styles, Shapes, Tool Presets and Clone Source.
//!
//! Photoshop keeps presets in named groups (folders) that the panels show as thumbnail grids or
//! lists. The engine owns that data so every frontend (GUI, CLI, MCP) sees and edits the same
//! sets through `<kind>.presets.*` commands:
//!
//! - gradients ([`gradients`]): colour + transparency stops; the selected one drives the
//!   Gradient tool (`paint.gradient` without `colors`) and Gradient Fill layers;
//! - patterns ([`patterns`]): groups over the existing pattern library (`pattern.*`);
//! - styles ([`styles`]): layer effects (plus optional blending) applied to the selected layers;
//! - shapes ([`shapes`]): vector shapes for the Custom Shape tool, placed as shape layers;
//! - tool presets ([`tools`]): a tool plus its options (opaque JSON the shell fills in);
//! - clone sources ([`clone_source`]): five source slots with offset, scale, rotation and flip
//!   that `paint.cloneStamp` / `paint.healingBrush` follow;
//! - swatches ([`swatches`]): named colours in their own colour model (commands in
//!   `crate::swatch_cmds`, `.aco`/`.ase` import and export).
//!
//! The built-in groups are our own (clean-room): colours, styles and shapes were designed here,
//! not copied from Photoshop's presets. User edits persist with the preferences document: see
//! [`PresetState::to_json`] (saved under `"presets"` by [`Session::prefs_to_json`]).

pub mod clone_source;
pub mod gradients;
pub mod patterns;
pub mod shapes;
pub mod styles;
pub mod swatches;
pub mod tools;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

/// A named folder of presets.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Group<T> {
    pub name: String,
    pub items: Vec<T>,
}

impl<T> Group<T> {
    pub fn new(name: &str, items: Vec<T>) -> Self {
        Group { name: name.into(), items }
    }
}

/// Anything with a display name (presets, pattern references).
pub trait Named {
    fn name(&self) -> &str;
    fn set_name(&mut self, n: String);
}

impl Named for String {
    fn name(&self) -> &str {
        self
    }
    fn set_name(&mut self, n: String) {
        *self = n;
    }
}

/// All preset sets plus the panel state the engine needs.
#[derive(Clone, Debug)]
pub struct PresetState {
    pub gradients: Vec<Group<gradients::GradientPreset>>,
    /// The Gradient tool's current gradient (Gradients panel selection / gradient picker).
    pub gradient: gradients::GradientPreset,
    pub styles: Vec<Group<styles::StylePreset>>,
    pub shapes: Vec<Group<shapes::ShapePreset>>,
    /// Pattern groups: pattern names per folder (patterns themselves live in the library).
    pub pattern_groups: Vec<Group<String>>,
    /// The pattern the Patterns panel selected (Fill, Pattern fill layers default to it).
    pub pattern: Option<String>,
    pub tool_presets: Vec<tools::ToolPreset>,
    /// File › New: named snapshots, separate from the name of the document being created.
    pub documents: Vec<crate::document_preset_cmds::DocumentPreset>,
    /// File › New › Recent: the settings of documents recently created there, newest first.
    pub recent_documents: Vec<crate::document_preset_cmds::DocumentSettings>,
    pub clone: clone_source::CloneSources,
    /// The user defaults "Make Default" in the Layer Style dialog saves
    /// (effect kind → param set); "Reset to Default" restores them.
    pub layer_defaults: BTreeMap<String, Value>,
    /// Swatches panel library.
    pub swatches: Vec<Group<swatches::Swatch>>,
    /// Bumped on every preset change (UIs key thumbnail caches on it).
    pub rev: u64,
}

impl Default for PresetState {
    fn default() -> Self {
        PresetState {
            gradients: gradients::builtin(),
            gradient: gradients::GradientPreset::foreground_to_background(),
            styles: styles::builtin(),
            shapes: shapes::builtin(),
            pattern_groups: patterns::builtin_groups(),
            pattern: None,
            tool_presets: tools::builtin(),
            documents: Vec::new(),
            recent_documents: Vec::new(),
            clone: Default::default(),
            layer_defaults: BTreeMap::new(),
            swatches: swatches::builtin(),
            rev: 0,
        }
    }
}

/// What persists (user-visible preset sets, not transient panel state).
#[derive(Serialize, Deserialize)]
struct Persisted {
    #[serde(default)]
    gradients: Option<Vec<Group<gradients::GradientPreset>>>,
    #[serde(default)]
    gradient: Option<gradients::GradientPreset>,
    #[serde(default)]
    styles: Option<Vec<Group<styles::StylePreset>>>,
    #[serde(default)]
    shapes: Option<Vec<Group<shapes::ShapePreset>>>,
    #[serde(default)]
    pattern_groups: Option<Vec<Group<String>>>,
    // Read entry by entry (`tools::load`): a bad entry must not discard the rest.
    #[serde(default)]
    tool_presets: Option<Value>,
    // Read this set independently: a bad entry must not discard the other preset kinds.
    #[serde(default)]
    documents: Option<Value>,
    #[serde(default)]
    recent_documents: Option<Value>,
    #[serde(default)]
    custom_shapes: Option<Vec<crate::edit_menu_cmds::CustomShape>>,
    #[serde(default)]
    layer_defaults: Option<BTreeMap<String, Value>>,
    swatches: Option<Vec<Group<swatches::Swatch>>>,
}

impl PresetState {
    /// JSON saved with the preferences (`"presets"` key).
    pub fn to_json(&self, s: &Session) -> Value {
        let p = Persisted {
            gradients: Some(self.gradients.clone()),
            gradient: Some(self.gradient.clone()),
            styles: Some(self.styles.clone()),
            shapes: Some(self.shapes.clone()),
            pattern_groups: Some(self.pattern_groups.clone()),
            tool_presets: Some(json!(self.tool_presets)),
            documents: Some(json!(self.documents)),
            recent_documents: Some(json!(self.recent_documents)),
            custom_shapes: Some(s.edit_state.custom_shapes.clone()),
            layer_defaults: Some(self.layer_defaults.clone()),
            swatches: Some(self.swatches.clone()),
        };
        serde_json::to_value(p).unwrap_or(Value::Null)
    }
}

impl Session {
    /// Restore presets saved by [`PresetState::to_json`]. Missing sets keep their defaults; a
    /// malformed blob is ignored (presets are never worth refusing to start over).
    pub fn load_presets_json(&mut self, v: Value) {
        let Ok(p) = serde_json::from_value::<Persisted>(v) else { return };
        let st = &mut self.presets;
        if let Some(g) = p.gradients {
            st.gradients = g;
        }
        if let Some(g) = p.gradient {
            st.gradient = g;
        }
        if let Some(g) = p.styles {
            st.styles = g;
        }
        if let Some(g) = p.shapes {
            st.shapes = g;
        }
        if let Some(g) = p.pattern_groups {
            st.pattern_groups = g;
        }
        if let Some(t) = p.tool_presets {
            st.tool_presets = tools::load(t);
        }
        if let Some(d) = p.documents {
            st.documents = crate::document_preset_cmds::load(d);
        }
        if let Some(r) = p.recent_documents {
            st.recent_documents = crate::document_preset_cmds::load_recent(r);
        }
        if let Some(c) = p.custom_shapes {
            self.edit_state.custom_shapes = c;
        }
        if let Some(d) = p.layer_defaults {
            st.layer_defaults = d;
        }
        if let Some(mut g) = p.swatches {
            // A hand-edited preferences file can't make the library unbounded.
            crate::swatch_cmds::cap_library(&mut g);
            self.presets.swatches = g;
        }
        self.presets.rev += 1;
    }

    /// Record a preset change: bumps the panel revision and marks the preferences document dirty
    /// so the shell persists it.
    pub(crate) fn presets_changed(&mut self) {
        self.presets.rev += 1;
        self.prefs.edit(|_| ());
    }
}

// ------------------------------------------------------------------ shared helpers

pub(crate) fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

pub(crate) fn always(_: &Session) -> std::result::Result<(), String> {
    Ok(())
}

pub(crate) fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

pub(crate) fn has_layer(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    d.active_layer.filter(|id| d.doc.layer(*id).is_some()).map(|_| ()).ok_or_else(|| "no active layer".into())
}

pub(crate) fn str_param<'a>(p: &'a Value, k: &str) -> Option<&'a str> {
    p.get(k).and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty())
}

pub(crate) fn req_str<'a>(p: &'a Value, k: &str, cmd: &str) -> Result<&'a str> {
    str_param(p, k).ok_or_else(|| bad(cmd, format!("missing `{k}`")))
}

/// Locate a preset by name (optionally within `group`): (group index, item index).
pub(crate) fn find<T: Named>(groups: &[Group<T>], name: &str, group: Option<&str>) -> Option<(usize, usize)> {
    groups
        .iter()
        .enumerate()
        .filter(|(_, g)| group.is_none_or(|n| g.name == n))
        .find_map(|(gi, g)| g.items.iter().position(|i| i.name() == name).map(|ii| (gi, ii)))
}

/// `base`, or `base 2`, `base 3`… so names stay unique across all groups (Photoshop allows
/// duplicates, but agents address presets by name).
pub(crate) fn unique_name<T: Named>(groups: &[Group<T>], base: &str) -> String {
    let taken = |n: &str| groups.iter().any(|g| g.items.iter().any(|i| i.name() == n));
    if !taken(base) {
        return base.to_string();
    }
    (2..).map(|i| format!("{base} {i}")).find(|n| !taken(n)).unwrap_or_else(|| base.to_string())
}

/// Index of `group` (created at the end when missing), or the first group (created as
/// "Presets" when there is none).
pub(crate) fn group_index<T>(groups: &mut Vec<Group<T>>, group: Option<&str>) -> usize {
    match group {
        Some(n) => groups.iter().position(|g| g.name == n).unwrap_or_else(|| {
            groups.push(Group::new(n, Vec::new()));
            groups.len() - 1
        }),
        None => {
            if groups.is_empty() {
                groups.push(Group::new("Presets", Vec::new()));
            }
            0
        }
    }
}

/// Generic group/item edits shared by every preset kind: `rename`, `delete`, `move`,
/// `newGroup`, `renameGroup`, `deleteGroup`.
pub(crate) fn edit_groups<T: Named>(groups: &mut Vec<Group<T>>, action: &str, p: &Value, cmd: &str) -> Result<Value> {
    let group = str_param(p, "group");
    match action {
        "rename" => {
            let name = req_str(p, "preset", cmd)?;
            let to = req_str(p, "name", cmd)?.to_string();
            let (gi, ii) = find(groups, name, group).ok_or_else(|| bad(cmd, format!("no preset \"{name}\"")))?;
            if name != to && find(groups, &to, None).is_some() {
                return Err(bad(cmd, format!("a preset named \"{to}\" already exists")));
            }
            groups[gi].items[ii].set_name(to.clone());
            Ok(json!({"name": to}))
        }
        "delete" => {
            let names: Vec<String> = match p.get("preset") {
                Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).map(str::to_string).collect(),
                _ => vec![req_str(p, "preset", cmd)?.to_string()],
            };
            // All or nothing: a missing name deletes none of the others.
            if let Some(name) = names.iter().find(|n| find(groups, n, group).is_none()) {
                return Err(bad(cmd, format!("no preset \"{name}\"")));
            }
            let mut names = names;
            names.sort();
            names.dedup();
            for name in &names {
                if let Some((gi, ii)) = find(groups, name, group)
                    && let Some(g) = groups.get_mut(gi)
                    && ii < g.items.len()
                {
                    g.items.remove(ii);
                }
            }
            Ok(json!({"deleted": names.len()}))
        }
        "move" => {
            let name = req_str(p, "preset", cmd)?;
            let to = req_str(p, "to", cmd)?;
            let (gi, ii) = find(groups, name, group).ok_or_else(|| bad(cmd, format!("no preset \"{name}\"")))?;
            let item = groups[gi].items.remove(ii);
            let ti = group_index(groups, Some(to));
            let at = p.get("index").and_then(Value::as_u64).map_or(groups[ti].items.len(), |i| (i as usize).min(groups[ti].items.len()));
            groups[ti].items.insert(at, item);
            Ok(json!({"group": to}))
        }
        "newGroup" => {
            let base = str_param(p, "name").unwrap_or("Group");
            let taken = |n: &str| groups.iter().any(|g| g.name == n);
            let name = if taken(base) { (1..).map(|i| format!("{base} {i}")).find(|n| !taken(n)).unwrap_or_default() } else { base.to_string() };
            groups.push(Group::new(&name, Vec::new()));
            Ok(json!({"group": name}))
        }
        "renameGroup" => {
            let g = req_str(p, "group", cmd)?;
            let to = req_str(p, "name", cmd)?.to_string();
            let i = groups.iter().position(|x| x.name == g).ok_or_else(|| bad(cmd, format!("no group \"{g}\"")))?;
            groups[i].name = to.clone();
            Ok(json!({"group": to}))
        }
        "deleteGroup" => {
            let g = req_str(p, "group", cmd)?;
            let i = groups.iter().position(|x| x.name == g).ok_or_else(|| bad(cmd, format!("no group \"{g}\"")))?;
            let removed = groups.remove(i);
            Ok(json!({"group": g, "deleted": removed.items.len()}))
        }
        _ => Err(bad(cmd, format!("unknown action {action}"))),
    }
}

/// Params doc for the generic group commands.
pub(crate) const GROUP_EDIT_PARAMS: &str = r##"{"action":"rename|delete|move|newGroup|renameGroup|deleteGroup","preset":name|[names] (rename/delete/move),"group":name? (narrows the lookup; the group for renameGroup/deleteGroup),"name":str (rename: new name; newGroup/renameGroup: group name),"to":group (move),"index":n? (move)}"##;

/// Every preset-panel command.
pub fn specs() -> Vec<CommandSpec> {
    let mut v = gradients::specs();
    v.extend(patterns::specs());
    v.extend(styles::specs());
    v.extend(shapes::specs());
    v.extend(tools::specs());
    v.extend(clone_source::specs());
    v
}

/// Run another command's implementation from inside a preset command (checks `enabled`, but
/// doesn't journal it: the outer command is what an action replays).
pub(crate) fn call(s: &mut Session, id: &str, p: Value) -> Result<Value> {
    let spec = crate::commands::find(id).ok_or_else(|| EngineError::UnknownCommand(id.to_string()))?;
    (spec.enabled)(s).map_err(|why| EngineError::Disabled(id.to_string(), why))?;
    (spec.run)(s, &p)
}
