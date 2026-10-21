//! Window › Tool Presets: a tool plus its options under a name.
//!
//! The engine doesn't know the shell's options-bar fields, so a preset's `options` are opaque
//! JSON that the frontend writes and reads back (the desktop shell stores its `ToolOptions` under
//! `"toolOptions"`). Two keys the engine understands itself: `"brush"` (brush settings, applied
//! to the session brush like `tools.setBrush`) and `"foreground"` (Include Color).
//!
//! The Crop tool's options-bar presets (New / Delete Crop Preset…) are tool presets of the
//! `"crop"` tool, as in Photoshop, so they also show in the Tool Presets panel.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{Group, Named, always, bad, edit_groups, req_str, str_param};
use crate::commands::CommandSpec;
use crate::{Result, Session};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolPreset {
    pub name: String,
    /// Tool name as the shell spells it (`"brush"`, `"eraser"`, `"gradient"`, …), case-insensitive.
    pub tool: String,
    #[serde(default)]
    pub options: Value,
}

impl Named for ToolPreset {
    fn name(&self) -> &str {
        &self.name
    }
    fn set_name(&mut self, n: String) {
        self.name = n;
    }
}

/// The most tool presets kept (saving more is refused; a larger preferences file loads the first).
pub const MAX_PRESETS: usize = 1000;

/// A preset name: trimmed, 1–[`MAX_NAME`](crate::brush_preset_cmds::MAX_NAME) characters.
fn checked_name(name: &str, cmd: &str) -> Result<String> {
    let name = name.trim();
    if name.is_empty() {
        return Err(bad(cmd, "the preset name is empty"));
    }
    if name.chars().count() > crate::brush_preset_cmds::MAX_NAME {
        return Err(bad(cmd, format!("preset names are at most {} characters", crate::brush_preset_cmds::MAX_NAME)));
    }
    Ok(name.to_string())
}

/// Read saved tool presets entry by entry, so a damaged entry (wrong type, empty or overlong name,
/// a duplicate name) is skipped without discarding the others.
pub(crate) fn load(value: Value) -> Vec<ToolPreset> {
    let Some(entries) = value.as_array() else { return Vec::new() };
    let mut presets: Vec<ToolPreset> = Vec::new();
    for entry in entries {
        if presets.len() >= MAX_PRESETS {
            break;
        }
        let Ok(mut preset) = serde_json::from_value::<ToolPreset>(entry.clone()) else { continue };
        let Ok(name) = checked_name(&preset.name, "") else { continue };
        if preset.tool.trim().is_empty() || presets.iter().any(|p| p.name == name) {
            continue;
        }
        if !preset.options.is_object() {
            preset.options = json!({});
        }
        preset.name = name;
        presets.push(preset);
    }
    presets
}

fn norm(t: &str) -> String {
    t.to_ascii_lowercase().replace([' ', '_', '-'], "").trim_end_matches("tool").to_string()
}

/// A few starting presets (our own).
pub fn builtin() -> Vec<ToolPreset> {
    let brush = |size: f32, hardness: f32| json!({"brush": {"size": size, "hardness": hardness}});
    vec![
        ToolPreset { name: "Soft Round 100 px".into(), tool: "brush".into(), options: brush(100.0, 0.0) },
        ToolPreset { name: "Hard Round 12 px".into(), tool: "brush".into(), options: brush(12.0, 1.0) },
        ToolPreset { name: "Soft Eraser 60 px".into(), tool: "eraser".into(), options: brush(60.0, 0.0) },
        ToolPreset { name: "Radial Gradient".into(), tool: "gradient".into(), options: json!({"toolOptions": {"gradient_style": "radial"}}) },
        ToolPreset { name: "Clone Soft 40 px".into(), tool: "cloneStamp".into(), options: brush(40.0, 0.2) },
    ]
}

fn list(s: &mut Session, p: &Value) -> Result<Value> {
    let filter = str_param(p, "tool").map(norm);
    let items: Vec<Value> = s
        .presets
        .tool_presets
        .iter()
        .filter(|t| filter.as_ref().is_none_or(|f| norm(&t.tool) == *f))
        .map(|t| json!({"name": t.name, "tool": t.tool}))
        .collect();
    Ok(json!({"presets": items}))
}

fn new_preset(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "tool.presets.new";
    let tool = req_str(p, "tool", CMD)?.to_string();
    let mut options = p.get("options").cloned().unwrap_or_else(|| json!({}));
    if !options.is_object() {
        return Err(bad(CMD, "`options` must be an object"));
    }
    // Brush-based tools carry the current brush unless the caller sent one.
    let brushy = [
        "brush",
        "pencil",
        "eraser",
        "clonestamp",
        "healing",
        "spothealing",
        "historybrush",
        "blur",
        "sharpen",
        "smudge",
        "dodge",
        "burn",
        "sponge",
        "mixerbrush",
        "colorreplacement",
    ];
    if options.get("brush").is_none() && brushy.contains(&norm(&tool).as_str()) {
        options["brush"] = serde_json::to_value(&s.tools.brush).unwrap_or(Value::Null);
    }
    if p.get("includeColor").and_then(Value::as_bool).unwrap_or(false) {
        let f = s.tools.foreground;
        options["foreground"] = json!(f);
    }
    if s.presets.tool_presets.len() >= MAX_PRESETS {
        return Err(bad(CMD, format!("at most {MAX_PRESETS} tool presets can be saved; delete some first")));
    }
    let base = match p.get("name") {
        None | Some(Value::Null) => tool.clone(),
        Some(Value::String(n)) if n.trim().is_empty() => tool.clone(),
        Some(Value::String(n)) => checked_name(n, CMD)?,
        Some(_) => return Err(bad(CMD, "`name` must be a string")),
    };
    let groups = [Group::new("", s.presets.tool_presets.clone())];
    let name = super::unique_name(&groups, &base);
    s.presets.tool_presets.push(ToolPreset { name: name.clone(), tool: tool.clone(), options });
    s.presets_changed();
    Ok(json!({"name": name, "tool": tool}))
}

fn select(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "tool.presets.select";
    let name = req_str(p, "preset", CMD)?;
    let t = s
        .presets
        .tool_presets
        .iter()
        .find(|t| t.name == name)
        .cloned()
        .ok_or_else(|| bad(CMD, format!("no tool preset \"{name}\" (see tool.presets.list)")))?;
    if let Some(b) = t.options.get("brush").filter(|b| b.is_object()) {
        super::call(s, "tools.setBrush", json!({"brush": b}))?;
    }
    if let Some(c) = t.options.get("foreground").and_then(Value::as_array).filter(|a| a.len() >= 3) {
        let v: Vec<f32> = c.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
        s.tools.foreground = [v[0], v[1], v[2], v.get(3).copied().unwrap_or(1.0)];
    }
    Ok(json!({"name": t.name, "tool": t.tool, "options": t.options}))
}

fn edit(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "tool.presets.edit";
    let action = req_str(p, "action", CMD)?.to_string();
    if !matches!(action.as_str(), "rename" | "delete") {
        return Err(bad(CMD, "tool presets support rename and delete"));
    }
    if action == "rename" {
        checked_name(req_str(p, "name", CMD)?, CMD)?;
    }
    let mut groups = vec![Group::new("", std::mem::take(&mut s.presets.tool_presets))];
    let r = edit_groups(&mut groups, &action, p, CMD);
    s.presets.tool_presets = groups.pop().map(|g| g.items).unwrap_or_default();
    let r = r?;
    s.presets_changed();
    Ok(r)
}

fn reset(s: &mut Session, _: &Value) -> Result<Value> {
    s.presets.tool_presets = builtin();
    s.presets_changed();
    Ok(json!({"presets": s.presets.tool_presets.len()}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "tool.presets.list",
            label: "Tool Presets",
            menu: &[],
            shortcut: None,
            params: r##"{"tool":name? (Current Tool Only)} → {presets:[{name,tool}]}"##,
            enabled: always,
            run: list,
            journal: false,
        },
        CommandSpec {
            id: "tool.presets.new",
            label: "New Tool Preset…",
            menu: &[],
            shortcut: None,
            params: r##"{"name":str?=tool (at most 255 characters; a taken name gets " 2", " 3"…),"tool":name ("crop" for Crop presets),"options":{…}? (opaque; "brush" defaults to the current brush for painting tools),"includeColor":bool=false}"##,
            enabled: always,
            run: new_preset,
            journal: true,
        },
        CommandSpec {
            id: "tool.presets.select",
            label: "Select Tool Preset",
            menu: &[],
            shortcut: None,
            params: r##"{"preset":name} → {name,tool,options} (applies "brush" and "foreground"; the shell switches tool and options)"##,
            enabled: always,
            run: select,
            journal: true,
        },
        CommandSpec {
            id: "tool.presets.edit",
            label: "Edit Tool Presets",
            menu: &[],
            shortcut: None,
            params: r##"{"action":"rename|delete","preset":name|[names],"name":str (rename)}"##,
            enabled: always,
            run: edit,
            journal: true,
        },
        CommandSpec {
            id: "tool.presets.reset",
            label: "Reset Tool Presets",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: always,
            run: reset,
            journal: true,
        },
    ]
}
