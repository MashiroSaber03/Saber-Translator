//! `plugin.*` commands: sandboxed WebAssembly filter plug-ins (`photocraft-plugins`).
//!
//! Plug-ins are installed per process (`plugin.install`, or the Plug-ins preference folder),
//! listed with `plugin.list` and run with `plugin.run` on the active pixel layer, a targeted
//! channel, or a smart object (recorded as a smart filter, re-run on re-render while the plug-in
//! is installed). Runs respect the selection and are one undo step.

use std::sync::Mutex;

use photocraft_doc::{LayerContent, SmartFilter};
use photocraft_plugins::{Plugin, registry};
use photocraft_raster::Surface;
use serde_json::{Map, Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

/// The smart-filter command id plug-in runs are recorded under.
pub const RUN: &str = "plugin.run";

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn err(e: photocraft_plugins::Error) -> EngineError {
    EngineError::Other(e.to_string())
}

fn always(_: &Session) -> std::result::Result<(), String> {
    Ok(())
}

/// JSON description of an installed plug-in.
pub fn describe(p: &Plugin) -> Value {
    let m = p.manifest();
    json!({
        "id": m.id, "name": m.name, "version": m.version, "kind": m.kind, "author": m.author, "description": m.description,
        "params": m.params_notation(), "paramsSchema": serde_json::to_value(m).ok().and_then(|v| v.get("params").cloned()),
        "overlap": m.overlap, "area": m.area, "size": p.size(), "source": p.source(),
    })
}

/// The plug-in's own parameters from command params: `params` if given, else the top-level keys
/// minus the ones the command itself uses.
fn plugin_params(p: &Value) -> Value {
    if let Some(v) = p.get("params") {
        return v.clone();
    }
    match p {
        Value::Object(m) => Value::Object(
            m.iter()
                .filter(|(k, _)| !matches!(k.as_str(), "id" | "layer" | "target" | "channel") && !k.starts_with("__"))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        ),
        _ => Value::Object(Map::new()),
    }
}

fn plugin_by_id(cmd: &str, p: &Value) -> Result<std::sync::Arc<Plugin>> {
    let id = p.get("id").and_then(Value::as_str).ok_or_else(|| bad(cmd, "missing `id` (see plugin.list)"))?;
    registry::get(id).ok_or_else(|| err(photocraft_plugins::Error::NotFound(id.into())))
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    let plugin = plugin_by_id(RUN, p)?;
    let params = plugin_params(p);
    // Bad parameters fail before anything is touched.
    let resolved = Value::Object(plugin.manifest().resolve_params(&params).map_err(|e| bad(RUN, e.to_string()))?);
    let layer = crate::commands::layer_param(s, p)?;
    let label = plugin.manifest().name.clone();
    let id = plugin.id().to_string();
    s.edit(&label, |doc, _| {
        let selection = doc.selection.clone();
        let canvas = doc.bounds();
        if let Some(surf) = crate::channel_cmds::channel_surface_for_filter(doc, Some(layer), p)? {
            *surf = plugin.apply(surf, canvas, selection.as_ref(), &resolved).map_err(err)?;
            return Ok(());
        }
        let l = doc.layer_mut(layer).ok_or(EngineError::NoLayer(layer))?;
        match &mut l.content {
            LayerContent::Raster(surf) => {
                *surf = plugin.apply(surf, canvas, selection.as_ref(), &resolved).map_err(err)?;
                Ok(())
            }
            LayerContent::Smart(_) => {
                let sf = SmartFilter {
                    command: RUN.into(),
                    params: json!({"id": id, "params": resolved}),
                    blend: photocraft_color::BlendMode::Normal,
                    opacity: 1.0,
                    visible: true,
                };
                crate::smart_cmds::add_smart_filter(doc, layer, sf, selection.as_ref())
            }
            other => Err(EngineError::Other(format!("plug-in filters need a pixel layer (active layer is {} {} layer)", other.article(), other.kind_name()))),
        }
    })?;
    Ok(json!({"layer": layer.0, "plugin": id}))
}

/// Smart-filter re-render of a recorded `plugin.run` (see `filters::apply_filter_to_surface`).
/// `None` when `command` isn't a plug-in run or the plug-in isn't installed (the smart filter is
/// then skipped, as for other unknown filters) or fails.
pub(crate) fn apply_to_surface(command: &str, params: &Value, surf: &Surface, selection: Option<&Surface>, canvas: photocraft_geom::Rect) -> Option<Surface> {
    if command != RUN {
        return None;
    }
    let plugin = registry::get(params.get("id")?.as_str()?)?;
    plugin.apply(surf, canvas, selection, params.get("params").unwrap_or(&Value::Null)).ok()
}

/// Largest base64 payload `plugin.install` accepts (the module limit, encoded).
const MAX_DATA_CHARS: usize = (32 << 20) / 3 * 4 + 4;

fn b64_decode(s: &str) -> Option<Vec<u8>> {
    let val = |c: u8| -> Option<u32> {
        Some(u32::from(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => return None,
        }))
    };
    let clean: Vec<u8> = s.bytes().filter(|c| !c.is_ascii_whitespace() && *c != b'=').collect();
    if clean.len() % 4 == 1 {
        return None;
    }
    let mut out = Vec::with_capacity(clean.len() / 4 * 3 + 2);
    for chunk in clean.chunks(4) {
        let mut n = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            n |= val(c)? << (18 - 6 * i);
        }
        let bytes = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        out.extend_from_slice(bytes.get(..chunk.len() - 1)?);
    }
    Some(out)
}

fn install(_s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "plugin.install";
    let plugin = if let Some(data) = p.get("data") {
        let data = data.as_str().ok_or_else(|| bad(CMD, "`data` must be a base64 string"))?;
        if data.len() > MAX_DATA_CHARS {
            return Err(bad(CMD, "`data` is larger than the module limit"));
        }
        let bytes = b64_decode(data).ok_or_else(|| bad(CMD, "`data` is not valid base64"))?;
        Plugin::load(&bytes, photocraft_plugins::Limits::default()).map_err(err)?
    } else if let Some(path) = p.get("path") {
        let path = path.as_str().filter(|s| !s.is_empty()).ok_or_else(|| bad(CMD, "`path` must be a file path"))?;
        load_path(path)?
    } else {
        return Err(bad(CMD, "give `path` (a .wasm file) or `data` (the module as base64)"));
    };
    let replace = p.get("replace").and_then(Value::as_bool).unwrap_or(true);
    if !replace && registry::get(plugin.id()).is_some() {
        return Err(bad(CMD, format!("a plug-in with id {:?} is already installed", plugin.id())));
    }
    Ok(describe(&registry::install(plugin)))
}

#[cfg(not(target_arch = "wasm32"))]
fn load_path(path: &str) -> Result<Plugin> {
    registry::load_file(std::path::Path::new(path), photocraft_plugins::Limits::default()).map_err(err)
}

#[cfg(target_arch = "wasm32")]
fn load_path(_: &str) -> Result<Plugin> {
    Err(EngineError::Other("installing from a path is not available on the web; pass the module as base64 `data`".into()))
}

/// The plug-in folder last loaded from the preferences, and what happened.
static FOLDER: Mutex<Option<(String, Value)>> = Mutex::new(None);

/// Loads the Plug-ins preference folder (`*.wasm`) when it is enabled and has changed since the
/// last load. Called whenever preferences are applied (startup, `prefs.set`). Native only.
pub(crate) fn sync_prefs(s: &Session) {
    let pi = &s.prefs().plug_ins;
    let folder = pi.additional_plugins_folder.trim();
    if !pi.use_additional_plugins_folder || folder.is_empty() {
        return;
    }
    let mut last = FOLDER.lock().unwrap_or_else(|e| e.into_inner());
    if last.as_ref().is_some_and(|(f, _)| f == folder) {
        return;
    }
    *last = Some((folder.to_string(), load_folder(folder)));
}

#[cfg(not(target_arch = "wasm32"))]
fn load_folder(folder: &str) -> Value {
    match registry::load_folder(std::path::Path::new(folder)) {
        Ok(r) => json!({"folder": folder, "loaded": r.loaded, "failed": r.failed.iter().map(|(f, e)| json!({"file": f, "error": e})).collect::<Vec<_>>()}),
        Err(e) => json!({"folder": folder, "error": e.to_string()}),
    }
}

#[cfg(target_arch = "wasm32")]
fn load_folder(folder: &str) -> Value {
    json!({"folder": folder, "error": "plug-in folders are not available on the web"})
}

fn reload(s: &mut Session, p: &Value) -> Result<Value> {
    let folder = match p.get("path") {
        Some(v) => v.as_str().filter(|f| !f.trim().is_empty()).ok_or_else(|| bad("plugin.reload", "`path` must be a folder"))?.trim().to_string(),
        None => s.prefs().plug_ins.additional_plugins_folder.trim().to_string(),
    };
    if folder.is_empty() {
        return Err(bad("plugin.reload", "no folder: pass `path` or set Preferences › Plug-ins › Additional Plug-ins Folder"));
    }
    let report = load_folder(&folder);
    if let Some(e) = report.get("error").and_then(Value::as_str) {
        return Err(EngineError::Other(e.to_string()));
    }
    *FOLDER.lock().unwrap_or_else(|e| e.into_inner()) = Some((folder, report.clone()));
    Ok(report)
}

/// The plug-in command specs.
pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "plugin.list",
            label: "List Plug-ins",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: always,
            run: |_, _| {
                let folder = FOLDER.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(|(_, r)| r.clone());
                Ok(json!({"plugins": registry::list().iter().map(|p| describe(p)).collect::<Vec<_>>(), "folder": folder}))
            },
            journal: false,
        },
        CommandSpec {
            id: "plugin.install",
            label: "Install Plug-in…",
            menu: &["Filter", "Plug-ins"],
            shortcut: None,
            params: r##"{"path":text,"data":json,"replace":bool=true} (path: a .wasm file, native only; data: the module as base64)"##,
            enabled: always,
            run: install,
            journal: false,
        },
        CommandSpec {
            id: "plugin.remove",
            label: "Remove Plug-in",
            menu: &[],
            shortcut: None,
            params: r##"{"id":text}"##,
            enabled: always,
            run: |_, p| {
                let id = p.get("id").and_then(Value::as_str).ok_or_else(|| bad("plugin.remove", "missing `id`"))?;
                if registry::remove(id) { Ok(json!({"removed": id})) } else { Err(err(photocraft_plugins::Error::NotFound(id.into()))) }
            },
            journal: false,
        },
        CommandSpec {
            id: "plugin.reload",
            label: "Reload Plug-ins",
            menu: &[],
            shortcut: None,
            params: r##"{"path":text} (a folder of .wasm plug-ins; default: the Plug-ins preference folder)"##,
            enabled: always,
            run: reload,
            journal: false,
        },
        CommandSpec {
            id: RUN,
            label: "Run Plug-in",
            menu: &[],
            shortcut: None,
            params: r##"{"id":text,"params":json} (plug-in parameters under "params" or at the top level; see plugin.list)"##,
            enabled: crate::filters::has_filterable_layer,
            run,
            journal: true,
        },
    ]
}

#[cfg(test)]
#[path = "plugin_cmds/tests.rs"]
mod tests;
