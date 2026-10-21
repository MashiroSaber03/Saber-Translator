//! File › Automate › Contact Sheet II and Create Droplet, File › Scripts › Statistics, Browse
//! and Script Events Manager, plus the session state the File-menu commands share
//! ([`FileMenuState`]).
//!
//! "Scripts" are PhotoCraft action scripts: JSON (`[[id, params], …]`, `{"steps": …}` or a
//! droplet) or a plain-text list with one `command.id {json params}` per line (`#` comments).
//! Droplets are JSON files (`.pcdroplet`) holding an action plus batch options; `photocraft-cli
//! droplet <file> <inputs…>` runs one, and on Unix a `.command` shim makes it
//! double-clickable / drop-target-able from the shell.
//!
//! Script events (Start Application, New / Open / Save / Close Document, Print, Export) are
//! bound in Preferences (`scriptEvents`) and fired by [`fire_event`]: the engine fires New,
//! Close, Print and Export itself; the app shell fires Start, Open and Save.

use photocraft_doc::{DocId, Layer, LayerContent};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::file_cmds::{file_name, import, join, list_images, read_file, stem, write_file};
use crate::{EngineError, Result, Session};

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

/// Session state of the File-menu commands (not saved, not history).
#[derive(Clone, Debug, Default)]
pub struct FileMenuState {
    /// View › Lock Slices.
    pub slices_locked: bool,
    /// Documents with File › Generate › Image Assets on.
    pub image_assets: Vec<DocId>,
    /// File › Print settings used last (Print One Copy repeats them).
    pub last_print: Option<Value>,
    /// Save for Web settings used last.
    pub last_web: Option<Value>,
    /// Results of the most recent script event (for inspection), newest last, capped.
    pub event_log: Vec<Value>,
    /// Set while event scripts run, so their commands don't fire events recursively.
    pub firing: bool,
}

// ---------- scripts ----------

/// Parses an action script: JSON (array of steps, `{"steps"|"action": …}`, a droplet) or one
/// `command.id {json}` per line.
pub fn parse_script(text: &str) -> Result<Vec<(String, Value)>> {
    let cmd = "file.scripts.browse";
    let t = text.trim_start_matches('\u{feff}').trim();
    if t.starts_with('[') || t.starts_with('{') {
        let v: Value = serde_json::from_str(t).map_err(|e| bad(cmd, format!("script JSON: {e}")))?;
        return parse_action(&v, cmd);
    }
    let mut out = Vec::new();
    for (n, line) in t.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }
        let (id, rest) = line.split_once(char::is_whitespace).map_or((line, ""), |(a, b)| (a, b.trim()));
        let params = if rest.is_empty() { json!({}) } else { serde_json::from_str(rest).map_err(|e| bad(cmd, format!("line {}: {e}", n + 1)))? };
        out.push((id.to_string(), params));
    }
    Ok(out)
}

/// The steps of a JSON action: an array of steps (`[id, params]`, `{"command", "params"}` or a
/// bare id), `{"steps"|"action": …}` (a recorded action) or a droplet. `cmd` names the caller in
/// errors.
pub fn parse_action(v: &Value, cmd: &str) -> Result<Vec<(String, Value)>> {
    let steps = match v {
        Value::Array(_) => v,
        Value::Object(o) => {
            o.get("steps").or_else(|| o.get("action").and_then(|a| a.get("steps").or(Some(a)))).ok_or_else(|| bad(cmd, "an action object needs \"steps\""))?
        }
        _ => return Err(bad(cmd, "an action must be an array of steps or an object")),
    };
    parse_steps(steps, cmd)
}

fn parse_steps(v: &Value, cmd: &str) -> Result<Vec<(String, Value)>> {
    let arr = v.as_array().ok_or_else(|| bad(cmd, "steps must be an array"))?;
    arr.iter()
        .map(|st| match st {
            Value::Array(a) if !a.is_empty() => {
                Ok((a[0].as_str().ok_or_else(|| bad(cmd, "step id must be a string"))?.to_string(), a.get(1).cloned().unwrap_or(json!({}))))
            }
            Value::Object(o) => {
                let id = o.get("command").or_else(|| o.get("id")).and_then(Value::as_str).ok_or_else(|| bad(cmd, "step needs \"command\""))?;
                Ok((id.to_string(), o.get("params").cloned().unwrap_or(json!({}))))
            }
            Value::String(id) => Ok((id.clone(), json!({}))),
            _ => Err(bad(cmd, "each step is [id, params], {\"command\", \"params\"} or an id")),
        })
        .collect()
}

fn check_known(steps: &[(String, Value)], cmd: &str) -> Result<()> {
    match steps.iter().find(|(id, _)| crate::commands::find(id).is_none()) {
        Some((id, _)) => Err(bad(cmd, format!("unknown command `{id}` in the script"))),
        None => Ok(()),
    }
}

fn run_steps(s: &mut Session, steps: &[(String, Value)]) -> Vec<Value> {
    let mut results = Vec::new();
    for (id, params) in steps {
        match s.execute(id, params.clone()) {
            // A called action reports a failed step inside an `Ok`; it stops the script too.
            Ok(r) => match crate::actions_cmds::nested_failure(id, &r) {
                Some(error) => {
                    results.push(json!({"command": id, "result": r, "error": error}));
                    break;
                }
                None => results.push(json!({"command": id, "result": r})),
            },
            Err(e) => {
                results.push(json!({"command": id, "error": e.to_string()}));
                break;
            }
        }
    }
    results
}

/// File › Scripts › Browse…: run a script file on the session.
fn browse(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "file.scripts.browse";
    let steps = match (p.get("path").and_then(Value::as_str), p.get("script").and_then(Value::as_str), p.get("steps")) {
        (Some(path), _, _) => parse_script(&String::from_utf8_lossy(&read_file(path)?))?,
        (_, Some(text), _) => parse_script(text)?,
        (_, _, Some(v)) => parse_steps(v, cmd)?,
        _ => return Err(bad(cmd, "give \"path\" (a script file), \"script\" (its text) or \"steps\"")),
    };
    check_known(&steps, cmd)?;
    let results = run_steps(s, &steps);
    let ok = results.iter().all(|r| r.get("error").is_none());
    Ok(json!({"steps": steps.len(), "ok": ok, "results": results}))
}

// ---------- Script Events Manager ----------

/// Events a script can be bound to (Photoshop's Script Events Manager list).
pub const EVENTS: [(&str, &str); 8] = [
    ("startApplication", "Start Application"),
    ("newDocument", "New Document"),
    ("openDocument", "Open Document"),
    ("saveDocument", "Save Document"),
    ("closeDocument", "Close Document"),
    ("print", "Print Document"),
    ("export", "Export Document"),
    ("everything", "Everything"),
];

/// One event → script binding.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ScriptBinding {
    pub event: String,
    /// A script file (see [`parse_script`]).
    pub script: Option<String>,
    /// Or inline steps (an action).
    pub steps: Option<Value>,
    /// Display name (the action's name).
    pub name: String,
}

/// Preferences › Script Events Manager.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ScriptEvents {
    /// "Enable Events to Run Scripts/Actions".
    pub enabled: bool,
    pub bindings: Vec<ScriptBinding>,
}

fn script_events(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "file.scripts.scriptEventsManager";
    let mut ev = s.prefs().script_events.clone();
    if let Some(b) = p.get("enabled").and_then(Value::as_bool) {
        ev.enabled = b;
    }
    if p.get("removeAll").and_then(Value::as_bool) == Some(true) {
        ev.bindings.clear();
    }
    if let Some(i) = crate::commands::int(p, "remove").filter(|v| *v >= 0) {
        if (i as usize) >= ev.bindings.len() {
            return Err(bad(cmd, format!("no binding {i}")));
        }
        ev.bindings.remove(i as usize);
    }
    // The dialog's flat form: "event" + "script" at the top level means add.
    let flat = p
        .get("event")
        .and_then(Value::as_str)
        .filter(|_| p.get("script").and_then(Value::as_str).is_some_and(|v| !v.is_empty()))
        .map(|e| json!({"event": e, "script": p["script"], "name": p.get("name").cloned().unwrap_or(Value::Null)}));
    if let Some(a) = p.get("add").cloned().or(flat).as_ref() {
        let event = a.get("event").and_then(Value::as_str).ok_or_else(|| bad(cmd, "add needs \"event\""))?;
        if !EVENTS.iter().any(|(id, _)| *id == event) {
            return Err(bad(cmd, format!("unknown event `{event}` ({})", EVENTS.map(|e| e.0).join("|"))));
        }
        let script = a.get("script").and_then(Value::as_str).map(str::to_string);
        let steps = a.get("steps").cloned();
        if script.is_none() && steps.is_none() {
            return Err(bad(cmd, "add needs \"script\" (a file) or \"steps\" (an action)"));
        }
        if let Some(st) = &steps {
            check_known(&parse_steps(st, cmd)?, cmd)?;
        }
        let name = a
            .get("name")
            .and_then(Value::as_str)
            .filter(|n| !n.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| script.as_deref().map(file_name).unwrap_or_else(|| "Action".into()));
        ev.bindings.push(ScriptBinding { event: event.into(), script, steps, name });
        if a.get("enable").and_then(Value::as_bool).unwrap_or(true) {
            ev.enabled = true;
        }
    }
    let changed = ev != s.prefs().script_events;
    if changed {
        s.edit_prefs(|pr| pr.script_events = ev.clone());
    }
    let events: Vec<Value> = EVENTS.iter().map(|(id, label)| json!({"id": id, "label": label})).collect();
    Ok(json!({"enabled": ev.enabled, "bindings": ev.bindings, "events": events}))
}

/// Runs the scripts bound to `event` (when Script Events are enabled). Returns how many ran.
pub fn fire_event(s: &mut Session, event: &str) -> usize {
    if s.file_menu.firing {
        return 0;
    }
    let ev = &s.prefs().script_events;
    if !ev.enabled {
        return 0;
    }
    let jobs: Vec<ScriptBinding> = ev.bindings.iter().filter(|b| b.event == event || b.event == "everything").cloned().collect();
    if jobs.is_empty() {
        return 0;
    }
    s.file_menu.firing = true;
    for b in &jobs {
        let steps = match (&b.steps, &b.script) {
            (Some(st), _) => parse_steps(st, "scriptEvent"),
            (None, Some(path)) => read_file(path).and_then(|bytes| parse_script(&String::from_utf8_lossy(&bytes))),
            _ => Ok(Vec::new()),
        };
        let entry = match steps {
            Ok(steps) => json!({"event": event, "binding": b.name, "results": run_steps(s, &steps)}),
            Err(e) => json!({"event": event, "binding": b.name, "error": e.to_string()}),
        };
        s.file_menu.event_log.push(entry);
    }
    let n = s.file_menu.event_log.len();
    if n > 32 {
        s.file_menu.event_log.drain(..n - 32);
    }
    s.file_menu.firing = false;
    jobs.len()
}

/// Called by `Session::execute` after every successful command: keeps layer-based slices on
/// their layers and fires the engine-side script events.
pub(crate) fn after_command(s: &mut Session, id: &str) {
    crate::slice_cmds::refresh(s);
    let event = match id {
        "file.new" => "newDocument",
        "file.close" | "file.closeAll" | "file.closeOthers" => "closeDocument",
        _ => return,
    };
    fire_event(s, event);
}

/// App shell hooks for the events only it sees.
pub fn document_opened(s: &mut Session) {
    fire_event(s, "openDocument");
}

/// After the shell saved document `index`: Save Document scripts and Image Assets.
pub fn document_saved(s: &mut Session, index: usize) -> Option<Value> {
    fire_event(s, "saveDocument");
    crate::web_cmds::on_saved(s, index)
}

// ---------- Create Droplet ----------

fn create_droplet(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "file.automate.createDroplet";
    let path = p.get("path").and_then(Value::as_str).filter(|v| !v.is_empty()).ok_or_else(|| bad(cmd, "missing \"path\" (where to save the droplet)"))?;
    let steps_v = p.get("steps").or_else(|| p.get("action")).ok_or_else(|| bad(cmd, "missing \"steps\" (the action to run)"))?;
    let mut steps = parse_steps(steps_v, cmd)?;
    // The droplet's output folder replaces recorded saves, and view steps don't apply, as in batch.
    steps.retain(|(id, _)| !crate::actions_cmds::shell_save_command(id) && !crate::actions_cmds::shell_view_command(id));
    check_known(&steps, cmd)?;
    // Play Action steps need their actions when the droplet runs on its own (#2794). Not under
    // `actions`: the CLI reads that key as an action set.
    let called = crate::actions_cmds::called_actions(&s.actions.list, &mut steps);
    let path = if path.ends_with(".pcdroplet") || path.ends_with(".json") { path.to_string() } else { format!("{path}.pcdroplet") };
    let name = p.get("name").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| stem(&path));
    let steps_json: Vec<Value> = steps.iter().map(|(id, pr)| json!([id, pr])).collect();
    let mut options = serde_json::Map::new();
    for k in ["output", "format", "quality", "suffix", "overrideOpen"] {
        if let Some(v) = p.get(k) {
            options.insert(k.into(), v.clone());
        }
    }
    let mut droplet = json!({"photocraftDroplet": 1, "name": name, "action": {"name": name, "steps": steps_json}, "options": options});
    if !called.is_empty() {
        droplet["calledActions"] = json!(called);
    }
    let text = serde_json::to_string_pretty(&droplet).unwrap_or_default();
    write_file(&path, text.as_bytes())?;
    let shim = p.get("shim").and_then(Value::as_bool).unwrap_or(cfg!(unix));
    let shim_path = if shim { Some(write_shim(&path)?) } else { None };
    Ok(json!({"path": path, "shim": shim_path, "steps": steps.len()}))
}

/// A tiny shell script next to the droplet: drop files on it (or pass them) to run the droplet.
#[cfg(not(target_arch = "wasm32"))]
fn write_shim(droplet: &str) -> Result<String> {
    let abs = std::fs::canonicalize(droplet).map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|_| droplet.to_string());
    let base = abs.trim_end_matches(".pcdroplet").trim_end_matches(".json");
    let shim = format!("{base}.command");
    // Single quotes keep shell expansions inert; embedded quotes need to end the
    // quoted segment, emit a literal quote, then resume it.
    let quoted_abs = format!("'{}'", abs.replace('\'', "'\\''"));
    let body = format!(
        "#!/bin/sh\n# PhotoCraft droplet: runs the action on the files given (or dropped).\nexec \"${{PHOTOCRAFT_CLI:-photocraft-cli}}\" droplet {quoted_abs} \"$@\"\n"
    );
    crate::file_cmds::write_file(&shim, body.as_bytes())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&shim, std::fs::Permissions::from_mode(0o755));
    }
    Ok(shim)
}
#[cfg(target_arch = "wasm32")]
fn write_shim(_: &str) -> Result<String> {
    Err(EngineError::Other("no file system on the web".into()))
}

/// Runs a droplet file on inputs (files and/or folders).
fn run_droplet(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "file.automate.runDroplet";
    let path = p.get("droplet").or_else(|| p.get("path")).and_then(Value::as_str).ok_or_else(|| bad(cmd, "missing \"droplet\""))?;
    let v: Value = serde_json::from_slice(&read_file(path)?).map_err(|e| bad(cmd, format!("{path}: {e}")))?;
    if v.get("photocraftDroplet").is_none() {
        return Err(bad(cmd, format!("{path} is not a PhotoCraft droplet")));
    }
    let steps = v.get("action").and_then(|a| a.get("steps")).cloned().ok_or_else(|| bad(cmd, "droplet has no action"))?;
    let opts = v.get("options").cloned().unwrap_or(json!({}));
    let mut inputs: Vec<String> = Vec::new();
    let raw: Vec<String> = match p.get("input").or_else(|| p.get("files")) {
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).map(str::to_string).collect(),
        Some(Value::String(one)) => vec![one.clone()],
        _ => return Err(bad(cmd, "missing \"input\" (files or folders)")),
    };
    for i in raw {
        if std::path::Path::new(&i).is_dir() { inputs.extend(list_images(&i)?) } else { inputs.push(i) }
    }
    if inputs.is_empty() {
        return Err(bad(cmd, "no input images"));
    }
    let output = p.get("output").or_else(|| opts.get("output")).and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| {
        let first = std::path::Path::new(&inputs[0]);
        join(&first.parent().map(|d| d.to_string_lossy().into_owned()).unwrap_or_default(), "droplet-output")
    });
    let mut bp = json!({"steps": steps, "input": inputs, "output": output, "format": opts.get("format").cloned().unwrap_or(json!("same"))});
    if let Some(q) = opts.get("quality") {
        bp["quality"] = q.clone();
    }
    if let Some(called) = v.get("calledActions") {
        bp["calledActions"] = called.clone();
    }
    let r = s.execute("file.automate.batch", bp)?;
    Ok(json!({"droplet": path, "output": output, "files": r["files"], "errors": r["errors"]}))
}

// ---------- Statistics ----------

fn statistics(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "file.scripts.statistics";
    let mode = p.get("mode").or_else(|| p.get("stackMode")).and_then(Value::as_str).unwrap_or("median");
    let sm = photocraft_doc::StackMode::ALL
        .into_iter()
        .find(|m| m.id().eq_ignore_ascii_case(mode))
        .ok_or_else(|| bad(cmd, format!("unknown stack mode `{mode}`")))?;
    let input = p.get("input").or_else(|| p.get("paths")).cloned().ok_or_else(|| bad(cmd, "missing \"input\" (files or a folder)"))?;
    let align = p.get("align").and_then(Value::as_bool).unwrap_or(false);
    let r = s.execute("file.scripts.loadFilesIntoStack", json!({"paths": input, "align": align, "createSmartObject": true}))?;
    let n = r["layers"].as_u64().unwrap_or(0);
    let id = s.active().and_then(|d| d.doc.layers.first().map(|l| l.id)).ok_or(EngineError::NoDocument)?;
    s.select_layer(id)?;
    s.execute(&format!("layer.smartObjects.stackMode.{}", sm.id()), json!({"layer": id.0}))?;
    Ok(json!({"document": s.active_index(), "layer": id.0, "images": n, "mode": sm.id()}))
}

// ---------- Contact Sheet II ----------

fn units_px(v: f64, unit: &str, res: f64) -> f64 {
    match unit {
        "cm" => v * res / 2.54,
        "mm" => v * res / 25.4,
        "px" | "pixels" => v,
        _ => v * res,
    }
}

fn contact_sheet(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "file.automate.contactSheetII";
    let files: Vec<String> = match p.get("input").or_else(|| p.get("files")) {
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).map(str::to_string).collect(),
        Some(Value::String(dir)) => list_images(dir)?,
        _ => return Err(bad(cmd, "missing \"input\" (a folder or an array of files)")),
    };
    if files.is_empty() {
        return Err(bad(cmd, "no images to place"));
    }
    let f = |k: &str, d: f64| p.get(k).and_then(Value::as_f64).filter(|v| v.is_finite() && *v > 0.0).unwrap_or(d);
    let unit = p.get("units").and_then(Value::as_str).unwrap_or("inches");
    let res = f("resolution", 300.0).clamp(1.0, 10_000.0);
    let pw = units_px(f("width", 8.0), unit, res).round().clamp(16.0, 30_000.0) as u32;
    let ph = units_px(f("height", 10.0), unit, res).round().clamp(16.0, 30_000.0) as u32;
    let cols = f("columns", 5.0).round().clamp(1.0, 100.0) as u32;
    let rows = f("rows", 6.0).round().clamp(1.0, 100.0) as u32;
    let auto = p.get("autoSpacing").and_then(Value::as_bool).unwrap_or(true);
    let gap = (f64::from(pw.min(ph)) * 0.014).max(2.0);
    let hs = if auto { gap } else { p.get("horizontal").and_then(Value::as_f64).map_or(gap, |v| units_px(v, unit, res)) };
    let vs = if auto { gap } else { p.get("vertical").and_then(Value::as_f64).map_or(gap, |v| units_px(v, unit, res)) };
    let across = p.get("placeAcrossFirst").and_then(Value::as_bool).unwrap_or(true);
    let rotate = p.get("rotateForBestFit").and_then(Value::as_bool).unwrap_or(false);
    let caption = p.get("caption").and_then(Value::as_bool).unwrap_or(true);
    let font = p.get("font").and_then(Value::as_str).unwrap_or(photocraft_text::fonts::DEFAULT_FAMILY).to_string();
    let font_pt = f("fontSize", 12.0);
    let flatten = p.get("flatten").and_then(Value::as_bool).unwrap_or(false);
    let mode = p.get("mode").and_then(Value::as_str).unwrap_or("rgb").to_string();
    let depth = crate::commands::int(p, "depth").unwrap_or(8);
    let caption_px = if caption { font_pt * res / 72.0 * 1.5 } else { 0.0 };
    let cell_w = ((f64::from(pw) - f64::from(cols + 1) * hs) / f64::from(cols)).max(1.0);
    let cell_h = ((f64::from(ph) - f64::from(rows + 1) * vs) / f64::from(rows)).max(1.0);
    let box_h = (cell_h - caption_px).max(1.0);
    let per_page = (cols * rows) as usize;
    let mut docs = Vec::new();
    let mut placed = 0usize;
    let mut errors = Vec::new();
    for (page, chunk) in files.chunks(per_page).enumerate() {
        let name = if files.len() > per_page { format!("ContactSheet-{:03}", page + 1) } else { "ContactSheet-001".into() };
        s.execute("file.new", json!({"width": pw, "height": ph, "mode": mode, "depth": depth, "background": "white", "name": name}))?;
        let idx = s.active_index().ok_or(EngineError::NoDocument)?;
        if let Some(st) = s.active_mut() {
            std::sync::Arc::make_mut(&mut st.doc).resolution_dpi = res as f32;
        }
        let fmt = s.active().ok_or(EngineError::NoDocument)?.doc.pixel_format();
        for (k, path) in chunk.iter().enumerate() {
            let k = k as u32;
            let (c, r) = if across { (k % cols, k / cols) } else { (k / rows, k % rows) };
            let x0 = hs + f64::from(c) * (cell_w + hs);
            let y0 = vs + f64::from(r) * (cell_h + vs);
            let thumb = (|| -> Result<photocraft_doc::Surface> {
                let bytes = read_file(path)?;
                let src = import(&file_name(path), &bytes)?;
                let mut t = Session::new();
                t.add_document(src, None);
                let (w, h) = t.active().map(|d| (f64::from(d.doc.size.width), f64::from(d.doc.size.height))).unwrap_or((1.0, 1.0));
                let fit = |w: f64, h: f64| (cell_w / w).min(box_h / h);
                if rotate && fit(h, w) > fit(w, h) * 1.001 {
                    t.execute("image.imageRotation.90cw", json!({}))?;
                }
                let (w, h) = t.active().map(|d| (f64::from(d.doc.size.width), f64::from(d.doc.size.height))).unwrap_or((1.0, 1.0));
                let k = fit(w, h);
                let (tw, th) = ((w * k).round().max(1.0), (h * k).round().max(1.0));
                t.execute("image.imageSize", json!({"width": tw, "height": th, "resample": "bicubic"}))?;
                let d = &t.active().ok_or(EngineError::NoDocument)?.doc;
                let buf = photocraft_compose::flatten(d);
                let ox = (x0 + (cell_w - tw) / 2.0).round() as i32;
                let oy = (y0 + (box_h - th) / 2.0).round() as i32;
                let n = fmt.channels();
                let mut data = vec![0.0f32; buf.px.len() * n];
                for (px, out) in buf.px.iter().zip(data.chunks_exact_mut(n)) {
                    photocraft_raster::from_rgba_into(&fmt, *px, out);
                }
                let mut surf = photocraft_doc::Surface::new(fmt);
                surf.write_region(photocraft_geom::Rect::new(ox, oy, ox + tw as i32, oy + th as i32), &data);
                Ok(surf)
            })();
            let surf = match thumb {
                Ok(s) => s,
                Err(e) => {
                    errors.push(json!({"file": path, "error": e.to_string()}));
                    continue;
                }
            };
            let label = stem(path);
            s.edit("Contact Sheet", |doc, active| {
                let l = Layer::new(label.clone(), LayerContent::Raster(surf));
                *active = Some(l.id);
                doc.layers.push(l);
                Ok(())
            })?;
            placed += 1;
            if caption {
                let size_pt = font_pt;
                let baseline = y0 + box_h + caption_px * 0.72;
                s.execute("type.create", json!({"x": x0 + cell_w / 2.0, "y": baseline, "text": label, "font": font, "size": size_pt, "align": "center", "color": [0.0, 0.0, 0.0], "name": label}))?;
            }
        }
        if flatten {
            s.execute("layer.flattenImage", json!({}))?;
        }
        docs.push(idx);
    }
    Ok(json!({"documents": docs, "pages": docs.len(), "images": placed, "errors": errors, "width": pw, "height": ph}))
}

pub fn specs() -> Vec<CommandSpec> {
    macro_rules! spec {
        ($id:expr, $label:expr, $menu:expr, $params:expr, $enabled:expr, $run:expr) => {
            CommandSpec { id: $id, label: $label, menu: $menu, shortcut: None, params: $params, enabled: $enabled, journal: true, run: $run }
        };
    }
    let native = crate::file_cmds::native;
    vec![
        spec!(
            "file.automate.contactSheetII",
            "Contact Sheet II…",
            &["File", "Automate"],
            r##"{"input":folder|[paths],"units":"inches|cm|mm|pixels"="inches","width":8,"height":10,"resolution":ppi=300,"mode":"rgb|gray|cmyk|lab"="rgb","depth":8|16=8,"columns":5,"rows":6,"placeAcrossFirst":bool=true,"autoSpacing":bool=true,"horizontal":units?,"vertical":units?,"rotateForBestFit":bool=false,"caption":bool=true (file name as caption),"font":family?,"fontSize":pt=12,"flatten":bool=false} → {documents, pages, images}"##,
            native,
            contact_sheet
        ),
        spec!(
            "file.automate.createDroplet",
            "Create Droplet…",
            &["File", "Automate"],
            r##"{"path":str (.pcdroplet),"steps":[[id,params]…] (the action),"name":str?,"output":folder?,"format":"same|png|jpg|…"?,"quality":0..12?,"shim":bool=true on Unix (writes <name>.command calling `photocraft-cli droplet`)} → {path, shim} (the actions its Play Action steps call are saved in the droplet)"##,
            native,
            create_droplet
        ),
        spec!(
            "file.automate.runDroplet",
            "Run Droplet",
            &[],
            r##"{"droplet":path,"input":[files or folders],"output":folder? (default: droplet's, else <input folder>/droplet-output)} → {files, errors} (an input whose output name was already written in the run goes to errors)"##,
            native,
            run_droplet
        ),
        spec!(
            "file.scripts.statistics",
            "Statistics…",
            &["File", "Scripts"],
            r##"{"mode":"mean|median|maximum|minimum|range|summation|variance|standardDeviation|skewness|kurtosis|entropy"="median","input":folder|[paths],"align":bool=false (Auto-Align first)} → new document with one stack-mode smart object"##,
            native,
            statistics
        ),
        spec!(
            "file.scripts.browse",
            "Browse…",
            &["File", "Scripts"],
            r##"{"path":script file | "script":text | "steps":[[id,params]…]} (JSON action, or one `command.id {json}` per line) → {steps, ok, results}"##,
            |_| Ok(()),
            browse
        ),
        spec!(
            "file.scripts.scriptEventsManager",
            "Script Events Manager…",
            &["File", "Scripts"],
            r##"{"enabled":bool?,"add":{"event":"startApplication|newDocument|openDocument|saveDocument|closeDocument|print|export|everything","script":path?|"steps":[…]?,"name":str?}?,"remove":index?,"removeAll":bool?} → {enabled, bindings, events}"##,
            |_| Ok(()),
            script_events
        ),
    ]
}

#[cfg(test)]
#[path = "automate_cmds/tests.rs"]
mod tests;
