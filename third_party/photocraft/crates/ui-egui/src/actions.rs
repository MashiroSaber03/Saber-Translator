//! Actions panel: record and replay command sequences.
//!
//! The list and the recording flag live on the engine session (`actions.record` / `stop` /
//! `play` / `list` / `get` / `delete` / `move` / `rename`), so the panel, the CLI and MCP share them. This module
//! keeps only which row is selected and which rows are expanded.

use egui::{Align2, Color32, Rect, Sense, Stroke, pos2, vec2};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use photocraft_engine::actions_cmds::{self, Action};

use crate::PhotocraftApp;
use crate::theme::Tokens;

/// Which row is selected and which rows are expanded. The action list itself is
/// [`photocraft_engine::Session::actions`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ActionsUi {
    pub selected: Option<usize>,
    pub selected_step: Option<usize>,
    pub expanded: Vec<bool>,
    /// Set folders shown collapsed (Photoshop's set disclosure triangles).
    pub collapsed_sets: Vec<String>,
    #[serde(skip)]
    reveal_recording: Option<(usize, usize)>,
    /// The action whose name is being edited in place (double-click on its name).
    #[serde(skip)]
    renaming: Option<RenameAction>,
}

#[derive(Clone, Debug, PartialEq)]
struct RenameAction {
    index: usize,
    text: String,
    focused: bool,
}

pub(crate) fn label_of(id: &str) -> String {
    photocraft_engine::commands::find(id)
        .map(|c| c.label)
        .or_else(|| crate::menus::UI_COMMANDS.iter().find(|c| c.0 == id).map(|c| c.1))
        // Shell menu items recorded as steps, like Save a Copy (#4373).
        .or_else(|| crate::menu_catalog::CATALOG.iter().find(|c| c.3 == id).map(|c| c.1))
        .map(|label| label.trim_end_matches('…').to_string())
        .unwrap_or_else(|| id.to_string())
}

/// The selected action, or the first one when nothing is selected.
pub fn selected_action(app: &PhotocraftApp) -> Option<&Action> {
    let list = &app.session.actions.list;
    let idx = app.ui.actions.selected.filter(|i| *i < list.len());
    idx.and_then(|i| list.get(i)).or_else(|| list.first())
}

/// Index of the playback target. Photoshop's Play button uses the first action when
/// the panel has actions but none of its rows is selected.
fn playback_selection(app: &PhotocraftApp) -> Option<usize> {
    let n = app.session.actions.list.len();
    if n == 0 { None } else { Some(app.ui.actions.selected.filter(|i| *i < n).unwrap_or(0)) }
}

/// Steps in the `[[id, params], …]` shape batch and droplets already accept, without the
/// steps switched off in the panel.
pub fn action_steps(action: &Action) -> Vec<Value> {
    action.enabled_steps().map(|(_, (id, p))| json!([id, p])).collect()
}

/// Named action bindings use the existing persisted shortcut overrides. Keeping the
/// name in the key means selecting or deleting another row cannot retarget F6.
pub(crate) const SHORTCUT_PREFIX: &str = actions_cmds::PLAY_SHORTCUT_PREFIX;

pub(crate) fn shortcut_id(name: &str) -> String {
    format!("{SHORTCUT_PREFIX}{name}")
}

pub(crate) fn assign_shortcut(app: &mut PhotocraftApp, name: &str, shortcut: &str) -> Result<Value, String> {
    let id = shortcut_id(name);
    let mut set = serde_json::Map::new();
    // A function key belongs to only one action. Ordinary menu bindings remain
    // intact and become available again when the action binding is removed.
    for (other, assigned) in &app.session.prefs().shortcuts {
        if other.starts_with(SHORTCUT_PREFIX) && other != &id && assigned == shortcut && !shortcut.is_empty() {
            set.insert(other.clone(), json!(""));
        }
    }
    set.insert(id, json!(shortcut));
    app.run("edit.keyboardShortcuts", json!({"set": set, "allowUnknown": true, "removeConflicts": false}))
}

/// Replay edits synchronously, with view steps handled by the shell in the same order.
/// Engine steps keep the headless action semantics (no dialogs or background jobs).
pub(crate) fn play(app: &mut PhotocraftApp, params: &Value) -> Result<Value, String> {
    let (action, from) = actions_cmds::playback_plan(&app.session, params).map_err(|e| e.to_string())?;
    let recording = actions_cmds::begin_playback(&mut app.session, &action).map_err(|e| e.to_string())?;
    let mut ran = 0u64;
    let mut failed = None;
    let mut paused = None;
    for (i, (id, p)) in action.enabled_steps().skip_while(|(i, _)| *i < from) {
        // The dialog toggle: show the step's dialog with its recorded values and continue
        // from the next step when it is confirmed (`resume`). Agents play headlessly.
        if action.dialogs.contains(&i) && !app.automation_input && (crate::filter_dialog::has_dialog(id) || crate::adjust_dialog::has_dialog(id)) {
            let label = label_of(id);
            let dialog = crate::dialogs::open_command_dialog(app, id, &label);
            if let Some(d) = app.ui.dialog_mut(dialog) {
                if let Some(o) = p.as_object() {
                    d.fields.extend(o.iter().map(|(k, v)| (k.clone(), v.clone())));
                }
                d.fields.insert(RESUME.into(), json!({"action": action.name, "from": i + 1}));
            }
            paused = Some(json!({"step": i, "id": id, "dialog": dialog}));
            break;
        }
        let result = (|| {
            if let Some(auth) = app.session.authorize {
                auth(id, p).map_err(|e| e.to_string())?;
            }
            if id == "actions.play" {
                play(app, p)
            } else if actions_cmds::shell_save_command(id) {
                // Write synchronously, so the next step doesn't meet a document locked by a
                // background save.
                let background = std::mem::replace(&mut app.background_jobs, false);
                let r = app.replay_save(id, p);
                app.background_jobs = background;
                r
            } else if actions_cmds::shell_view_command(id) {
                app.sync_views();
                app.run(id, p.clone())
            } else {
                app.session.execute(id, p.clone()).map_err(|e| e.to_string())
            }
        })();
        match result {
            Ok(value) => {
                if let Some(error) = actions_cmds::nested_failure(id, &value) {
                    failed = Some(json!({"step": i, "id": id, "error": error}));
                    break;
                }
                if let Some(inner) = value.get("paused").filter(|p| p.is_object()) {
                    // A called action paused for a dialog; it resumes alone when confirmed.
                    paused = Some(inner.clone());
                    break;
                }
                ran += 1;
            }
            Err(error) => {
                failed = Some(json!({"step": i, "id": id, "error": error}));
                break;
            }
        }
    }
    actions_cmds::finish_playback(&mut app.session, recording, &action, from);
    Ok(match (failed, paused) {
        (Some(f), _) => json!({"action": action.name, "ran": ran, "failed": f}),
        (None, Some(p)) => json!({"action": action.name, "ran": ran, "paused": p}),
        (None, None) => json!({"action": action.name, "ran": ran}),
    })
}

/// Execute the actual menu command, not a synthetic keystroke. Fit is deferred
/// until canvas layout, so it uses the dimensions after all preceding action steps.
pub(crate) fn run_view(app: &mut PhotocraftApp, id: &str, params: Value) -> Result<Value, String> {
    app.sync_views();
    let i = app.session.active_index().ok_or("no document")?;
    let v = app.ui.views.get_mut(i).ok_or("no document view")?;
    v.fill_pending = false;
    match id {
        "view.fitOnScreen" => v.fit_pending = true,
        "view.zoomIn" => {
            v.fit_pending = false;
            v.zoom = crate::zoom_levels::step(v.zoom, 1, v.doc_size);
        }
        "view.zoomOut" => {
            v.fit_pending = false;
            v.zoom = crate::zoom_levels::step(v.zoom, -1, v.doc_size);
        }
        "view.actualPixels" => {
            v.fit_pending = false;
            v.zoom = 1.0;
        }
        _ => return Err(format!("unsupported view command: {id}")),
    }
    app.session.journal.push((id.into(), params));
    Ok(Value::Null)
}

fn begin_recording(app: &mut PhotocraftApp, append: bool) {
    if app.session.actions.recording.is_some() {
        return;
    }
    let params = if append { app.ui.actions.selected.map_or(json!({}), |i| json!({"action": i})) } else { json!({}) };
    let Ok(v) = app.run("actions.record", params) else { return };
    let Some(i) = v.get("index").and_then(Value::as_u64).and_then(|n| usize::try_from(n).ok()) else { return };
    app.ui.actions.selected = Some(i);
    app.ui.actions.selected_step = None;
    if app.ui.actions.expanded.len() <= i {
        app.ui.actions.expanded.resize(i + 1, false);
    }
    app.ui.actions.expanded[i] = true;
}

pub(crate) fn report_play(app: &mut PhotocraftApp, v: &Value) {
    if let Some(paused) = v.get("paused").filter(|p| p.is_object()) {
        let step = paused.get("step").and_then(Value::as_u64).unwrap_or(0);
        let template = "Step {step} is waiting for its dialog";
        app.ui.status = crate::i18n::fmt(tl!(template), &[("step", &(step + 1).to_string())]);
        app.ui.status_error = false;
        return;
    }
    let Some(failed) = v.get("failed").filter(|f| f.is_object()) else {
        let ran = v.get("ran").and_then(Value::as_u64).unwrap_or(0);
        app.ui.status = crate::i18n::trn(crate::i18n::current(), ran, "Played {n} step", "Played {n} steps");
        app.ui.status_error = false;
        return;
    };
    let step = failed.get("step").and_then(Value::as_u64).unwrap_or(0);
    let id = failed.get("id").and_then(Value::as_str).unwrap_or("");
    let error = failed.get("error").and_then(Value::as_str).unwrap_or("");
    let template = "Step {step} ({id}) failed: {error}";
    app.ui.status = crate::i18n::fmt(tl!(template), &[("step", &(step + 1).to_string()), ("id", id), ("error", error)]);
    app.ui.status_error = true;
}

fn visible_steps(app: &PhotocraftApp, action: usize) -> Vec<(String, Value)> {
    app.session.actions.list.get(action).into_iter().flat_map(|a| a.steps.iter()).chain(actions_cmds::pending_steps(&app.session, action)).cloned().collect()
}

/// "3 steps", or "2/3 steps" when some are switched off (Photoshop shows a red partial check).
fn step_count(steps: usize, disabled: usize) -> String {
    if disabled == 0 {
        crate::i18n::trn(crate::i18n::current(), steps as u64, "{n} step", "{n} steps")
    } else {
        let count = crate::i18n::trn(crate::i18n::current(), steps as u64, "{n} step", "{n} steps");
        format!("{}/{count}", steps.saturating_sub(disabled))
    }
}

fn paint_check(ui: &egui::Ui, rect: Rect, on: bool, t: &Tokens) {
    let b = Rect::from_center_size(rect.center(), vec2(11.0, 11.0));
    ui.painter().rect_stroke(b, 2.0, Stroke::new(1.0, t.text_faint), egui::StrokeKind::Inside);
    if on {
        let pts = vec![pos2(b.left() + 2.5, b.center().y), pos2(b.left() + 4.5, b.bottom() - 3.0), pos2(b.right() - 2.5, b.top() + 2.5)];
        ui.painter().add(egui::Shape::line(pts, Stroke::new(1.5, t.text)));
    }
}

fn step_label(id: &str, params: &Value) -> String {
    if id == "actions.play" {
        let target = params.get("action").map(|v| v.as_str().map(str::to_owned).unwrap_or_else(|| v.to_string())).unwrap_or_default();
        format!("{}: {target}", label_of(id))
    } else if id == "actions.conditional" {
        // "If hasLayers: FS8-Layers / FS8-No-Layers"
        let cond = params.get("if").and_then(Value::as_str).unwrap_or("?");
        let branch = |k: &str| params.get(k).and_then(Value::as_str).unwrap_or("–").to_string();
        format!("{}: {cond} → {} / {}", label_of(id), branch("then"), branch("else"))
    } else if id == photocraft_engine::atn_cmds::STOP_MESSAGE {
        format!("{}: {}", label_of(id), params.get("message").and_then(Value::as_str).unwrap_or_default())
    } else if id == photocraft_engine::atn_cmds::PHOTOSHOP_STEP {
        // A step from a .atn file PhotoCraft can't play yet: its Photoshop name, marked.
        let name = params.get("name").and_then(Value::as_str).filter(|n| !n.is_empty()).unwrap_or("?");
        crate::i18n::fmt(tl!("{name} (Photoshop)"), &[("name", name)])
    } else {
        label_of(id)
    }
}

/// Command params for a picked file: a real path keeps the journal small; browser files arrive
/// as bytes.
fn source(name: &str, bytes: &[u8]) -> Value {
    #[cfg(not(target_arch = "wasm32"))]
    if std::path::Path::new(name).is_absolute() && std::path::Path::new(name).is_file() {
        return json!({"path": name});
    }
    let _ = name;
    json!({"data": photocraft_engine::paint::tile::b64_encode(bytes)})
}

/// Load a `.atn` file's bytes into the Actions panel (the panel menu's Load Actions…).
pub fn load_bytes(app: &mut PhotocraftApp, name: &str, bytes: &[u8]) -> Result<Value, String> {
    let r = app.run("actions.load", source(name, bytes))?;
    let n = r["actions"].as_array().map_or(0, Vec::len) as u64;
    let unsupported = r["unsupported"].as_array().map_or(0, Vec::len);
    let shown = crate::file_open::display_name(name);
    let loaded =
        crate::i18n::fmt(&crate::i18n::trn(crate::i18n::current(), n, "Loaded {n} action from {file}", "Loaded {n} actions from {file}"), &[("file", &shown)]);
    app.ui.status_error = false;
    app.ui.status = if unsupported > 0 {
        let note = crate::i18n::trn(
            crate::i18n::current(),
            unsupported as u64,
            "{n} step has no PhotoCraft equivalent yet",
            "{n} steps have no PhotoCraft equivalent yet",
        );
        let template = "{message} ({note})";
        crate::i18n::fmt(tl!(template), &[("message", &loaded), ("note", &note)])
    } else {
        loaded
    };
    // Show what arrived: select the first loaded action, expanded.
    let list = &app.session.actions.list;
    if let Some(first) = r["actions"][0].as_str()
        && let Some(i) = list.iter().position(|a| a.name == first)
    {
        app.ui.actions.selected = Some(i);
        app.ui.actions.selected_step = None;
        app.ui.actions.expanded.resize(list.len(), false);
        if let Some(e) = app.ui.actions.expanded.get_mut(i) {
            *e = true;
        }
    }
    app.ui.panels.history = true;
    Ok(r)
}

fn load_dialog(app: &mut PhotocraftApp) -> Result<Value, String> {
    app.pick_file_bytes_filtered(&["atn"], |app, name, bytes| load_bytes(app, &name, &bytes))
}

/// Save Actions…: the selected action's set (every action of it), or every action when none
/// came from a set, as a `.atn` file Photoshop loads.
fn save_dialog(app: &mut PhotocraftApp, set: Option<String>) -> Result<Value, String> {
    let set = set.or_else(|| selected_action(app).and_then(|a| a.set.clone()));
    let params = match &set {
        Some(set) => json!({"set": set, "skipUnsupported": true}),
        None => json!({"skipUnsupported": true}),
    };
    let r = app.run("actions.save", params)?;
    let bytes = r["data"].as_str().and_then(photocraft_engine::paint::tile::b64_decode).ok_or("the export returned no data")?;
    let n = r["actions"].as_array().map_or(0, Vec::len) as u64;
    let skipped = r["skipped"].as_array().map_or(0, Vec::len);
    let file = format!("{}.atn", r["set"].as_str().unwrap_or("Actions"));
    app.pick_save(&file, move |app, path| {
        let write = app.services.write.as_mut().ok_or("no writer configured")?;
        write(&path, &bytes)?;
        let shown = crate::file_open::display_name(&path);
        let saved =
            crate::i18n::fmt(&crate::i18n::trn(crate::i18n::current(), n, "Saved {n} action to {file}", "Saved {n} actions to {file}"), &[("file", &shown)]);
        app.ui.status_error = false;
        app.ui.status = if skipped > 0 {
            let note = crate::i18n::trn(
                crate::i18n::current(),
                skipped as u64,
                "{n} step with no Photoshop equivalent left out",
                "{n} steps with no Photoshop equivalent left out",
            );
            let template = "{message} ({note})";
            crate::i18n::fmt(tl!(template), &[("message", &saved), ("note", &note)])
        } else {
            saved
        };
        Ok(json!({"path": path, "actions": n, "skipped": skipped}))
    })
}

const INSERT_STOP: &str = "__insertStop";

pub fn owns_insert_stop(fields: &serde_json::Map<String, Value>) -> bool {
    fields.contains_key(INSERT_STOP)
}

/// Insert Stop…: a message step after the selected step (or at the end of the selected action).
fn insert_stop_dialog(app: &mut PhotocraftApp) -> Result<Value, String> {
    let action = app.ui.actions.selected.filter(|i| *i < app.session.actions.list.len()).ok_or_else(|| tl!("Select an action first").to_string())?;
    let len = app.session.actions.list[action].steps.len() + actions_cmds::pending_steps(&app.session, action).count();
    let at = app.ui.actions.selected_step.map_or(len, |s| (s + 1).min(len));
    let fields =
        json!({INSERT_STOP: true, "__command": "actions.insert", "__label": "Insert Stop", "action": action, "at": at, "message": "", "allowContinue": false});
    let id = app.ui.open_dialog(crate::state::DialogKind::Command, fields.as_object().cloned().unwrap_or_default());
    Ok(json!({"dialog": id}))
}

pub fn insert_stop_body(ui: &mut egui::Ui, fields: &mut serde_json::Map<String, Value>) {
    ui.label(tl!("Message:"));
    let mut message = fields.get("message").and_then(Value::as_str).unwrap_or_default().to_string();
    ui.add(egui::TextEdit::multiline(&mut message).desired_rows(4).desired_width(320.0));
    fields.insert("message".into(), json!(message));
    ui.add_space(6.0);
    let mut allow = fields.get("allowContinue").and_then(Value::as_bool).unwrap_or(false);
    ui.checkbox(&mut allow, tl!("Allow Continue"));
    fields.insert("allowContinue".into(), json!(allow));
}

pub fn insert_stop_confirm(app: &mut PhotocraftApp, fields: &serde_json::Map<String, Value>) -> Result<Value, String> {
    let action = fields.get("action").and_then(Value::as_u64).ok_or("the dialog names no action")?;
    let params = json!({
        "action": action,
        "at": fields.get("at").cloned().unwrap_or(Value::Null),
        "id": photocraft_engine::atn_cmds::STOP_MESSAGE,
        "params": {"message": fields.get("message").cloned().unwrap_or(json!("")), "allowContinue": fields.get("allowContinue").cloned().unwrap_or(json!(false))},
    });
    let r = app.run("actions.insert", params)?;
    if let (Some(a), Some(step)) = (usize::try_from(action).ok(), r["step"].as_u64().and_then(|s| usize::try_from(s).ok())) {
        app.ui.actions.selected = Some(a);
        app.ui.actions.selected_step = Some(step);
        app.ui.actions.expanded.resize(app.session.actions.list.len(), false);
        if let Some(e) = app.ui.actions.expanded.get_mut(a) {
            *e = true;
        }
    }
    Ok(r)
}

/// The key a paused playback leaves in its dialog's fields: `{action, from}`.
pub const RESUME: &str = "__resumeAction";

/// A step dialog confirmed: continue the action after it (`dialogs::confirm` calls this).
pub fn resume(app: &mut PhotocraftApp, fields: &serde_json::Map<String, Value>) {
    let Some(r) = fields.get(RESUME) else { return };
    // The dialog's command may run as a background job; the next step needs its result.
    if let Some(doc) = app.session.active().map(|d| d.doc.id)
        && let Some(job) = app.session.job_on(doc)
    {
        let _ = app.session.wait_job(job.id);
    }
    let params = json!({"action": r.get("action").cloned().unwrap_or(Value::Null), "from": r.get("from").cloned().unwrap_or(Value::Null)});
    match play(app, &params) {
        Ok(v) => report_play(app, &v),
        Err(e) => {
            let template = "Action playback failed: {error}";
            app.ui.status = crate::i18n::fmt(tl!(template), &[("error", &e)]);
            app.ui.status_error = true;
        }
    }
}

/// A step dialog cancelled: the action stops there, as in Photoshop.
pub fn stopped(app: &mut PhotocraftApp, fields: &serde_json::Map<String, Value>) {
    if fields.contains_key(RESUME) {
        app.ui.status = tl!("Action playback stopped").to_string();
        app.ui.status_error = false;
    }
}

/// The Actions entries of the panel group's menu ("…").
pub fn panel_menu(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let mut act: Option<&str> = None;
    if ui.button(tl!("Load Actions…")).clicked() {
        act = Some("load");
    }
    let any = !app.session.actions.list.is_empty();
    if ui.add_enabled(any, egui::Button::new(tl!("Save Actions…"))).clicked() {
        act = Some("save");
    }
    ui.separator();
    let selected = app.ui.actions.selected.is_some_and(|i| i < app.session.actions.list.len());
    if ui.add_enabled(selected, egui::Button::new(tl!("Insert Stop…"))).clicked() {
        act = Some("stop");
    }
    let Some(a) = act else { return };
    ui.close();
    let r = match a {
        "load" => load_dialog(app),
        "stop" => insert_stop_dialog(app),
        _ => save_dialog(app, None),
    };
    if let Err(e) = r
        && e != "cancelled"
    {
        app.ui.status = e;
        app.ui.status_error = true;
    }
}

#[derive(Clone, Copy)]
struct DragRow {
    action: usize,
    step: Option<usize>,
}

/// Drops use an insertion line, converted to the final index expected by actions.move.
fn drag_row(ui: &egui::Ui, response: &egui::Response, row: DragRow, t: &Tokens) -> Option<(DragRow, usize)> {
    response.dnd_set_drag_payload(row);
    let source = response.dnd_hover_payload::<DragRow>()?;
    if source.step.is_some() != row.step.is_some() || (source.step.is_some() && source.action != row.action) {
        return None;
    }
    let after = ui.input(|i| i.pointer.hover_pos().is_some_and(|p| p.y >= response.rect.center().y));
    let index = row.step.unwrap_or(row.action);
    let from = source.step.unwrap_or(source.action);
    let slot = index.saturating_add(usize::from(after));
    let to = if from < slot { slot.saturating_sub(1) } else { slot };
    let y = if after { response.rect.bottom() } else { response.rect.top() };
    ui.painter().line_segment([pos2(response.rect.left() + 22.0, y), pos2(response.rect.right(), y)], Stroke::new(2.0, t.accent));
    response.dnd_release_payload::<DragRow>().map(|source| (*source, to))
}

fn move_row(app: &mut PhotocraftApp, source: DragRow, to: usize) {
    let mut params = json!({"action": source.action, "to": to});
    if let Some(step) = source.step {
        params["step"] = json!(step);
    }
    if app.run("actions.move", params).is_err() {
        return;
    }
    if source.step.is_some() {
        app.ui.actions.selected = Some(source.action);
        app.ui.actions.selected_step = Some(to);
    } else {
        app.ui.actions.expanded.resize(app.session.actions.list.len(), false);
        if source.action < app.ui.actions.expanded.len() && to < app.ui.actions.expanded.len() {
            let expanded = app.ui.actions.expanded.remove(source.action);
            app.ui.actions.expanded.insert(to, expanded);
        }
        app.ui.actions.selected = Some(to);
        app.ui.actions.selected_step = None;
        // Dropped among another set's actions: the action joins that set (the row above wins).
        let list = &app.session.actions.list;
        let neighbour = to.checked_sub(1).and_then(|j| list.get(j)).or_else(|| list.get(to + 1)).and_then(|a| a.set.clone());
        if list.get(to).is_some_and(|a| a.set != neighbour) {
            let _ = app.run("actions.assignSet", json!({"action": to, "set": neighbour}));
        }
    }
}

fn delete_selection(app: &mut PhotocraftApp) {
    let Some(action) = app.ui.actions.selected else { return };
    let step = app.ui.actions.selected_step;
    let mut params = json!({"action": action});
    if let Some(step) = step {
        params["step"] = json!(step);
    }
    if app.run("actions.delete", params).is_err() {
        return;
    }
    // Clear selection after deletion so a second click cannot delete a different
    // step or unexpectedly fall back to deleting the entire action.
    app.ui.actions.selected = None;
    app.ui.actions.selected_step = None;
    if step.is_none() && action < app.ui.actions.expanded.len() {
        app.ui.actions.expanded.remove(action);
    }
}

/// The in-place name field of the action being renamed. Enter, Tab or clicking elsewhere
/// returns the new name to commit; Escape cancels and keeps the old name.
fn rename_field(app: &mut PhotocraftApp, ui: &mut egui::Ui, index: usize, rect: Rect) -> Option<String> {
    let mut r = app.ui.actions.renaming.take().filter(|r| r.index == index)?;
    let id = egui::Id::new(("action-rename-field", index));
    let te = ui.put(rect, egui::TextEdit::singleline(&mut r.text).id(id).font(egui::FontId::proportional(12.0)));
    if !r.focused {
        te.request_focus();
        // Like Photoshop, the whole name starts selected so typing replaces it.
        if let Some(mut state) = egui::TextEdit::load_state(ui.ctx(), id) {
            let all = egui::text::CCursorRange::two(egui::text::CCursor::new(0), egui::text::CCursor::new(r.text.chars().count()));
            state.cursor.set_char_range(Some(all));
            state.store(ui.ctx(), id);
        }
        r.focused = true;
        app.ui.actions.renaming = Some(r);
        return None;
    }
    let (enter, tab, esc) = ui.input(|i| (i.key_pressed(egui::Key::Enter), i.key_pressed(egui::Key::Tab), i.key_pressed(egui::Key::Escape)));
    if esc {
        None
    } else if enter || tab || te.lost_focus() || !te.has_focus() {
        Some(r.text)
    } else {
        app.ui.actions.renaming = Some(r);
        None
    }
}

fn rename_action(app: &mut PhotocraftApp, index: usize, name: String) {
    if let Err(e) = app.run("actions.rename", json!({"action": index, "name": name})) {
        let template = "Could not rename the action: {error}";
        app.ui.status = crate::i18n::fmt(tl!(template), &[("error", &e)]);
        app.ui.status_error = true;
    }
}

/// One action's row data for a frame.
struct Row {
    name: String,
    steps: Vec<(String, Value)>,
    disabled: std::collections::BTreeSet<usize>,
    dialogs: std::collections::BTreeSet<usize>,
    set: Option<String>,
}

/// The panel's order: actions outside any set first, then each set folder (in order of first
/// appearance) with its actions. Indices are into the engine's flat list.
fn display_order(rows: &[Row]) -> Vec<usize> {
    let mut order: Vec<usize> = rows.iter().enumerate().filter(|(_, r)| r.set.is_none()).map(|(i, _)| i).collect();
    let mut sets: Vec<&str> = Vec::new();
    for r in rows {
        if let Some(set) = r.set.as_deref()
            && !sets.contains(&set)
        {
            sets.push(set);
        }
    }
    for set in sets {
        order.extend(rows.iter().enumerate().filter(|(_, r)| r.set.as_deref() == Some(set)).map(|(i, _)| i));
    }
    order
}

pub fn panel(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let recording = app.session.actions.recording;
    let rows: Vec<Row> = app
        .session
        .actions
        .list
        .iter()
        .enumerate()
        .map(|(i, a)| Row { name: a.name.clone(), steps: visible_steps(app, i), disabled: a.disabled.clone(), dialogs: a.dialogs.clone(), set: a.set.clone() })
        .collect();
    if app
        .ui
        .actions
        .selected
        .is_some_and(|i| i >= rows.len() || app.ui.actions.selected_step.is_some_and(|step| rows.get(i).is_none_or(|row| step >= row.steps.len())))
    {
        app.ui.actions.selected = None;
        app.ui.actions.selected_step = None;
    }
    if app.ui.actions.expanded.len() > rows.len() {
        app.ui.actions.expanded.truncate(rows.len());
    }
    if app.ui.actions.renaming.as_ref().is_some_and(|r| r.index >= rows.len()) {
        app.ui.actions.renaming = None;
    }
    let mut renamed = None;
    let mut toggled = None;
    let mut dialog_toggled = None;
    let mut set_toggled: Option<String> = None;
    let mut save_set: Option<String> = None;
    let max_h = (ui.available_height() - 38.0).max(0.0);
    let mut play_idx = None;
    let mut moved = None;
    let recording_row = recording.and_then(|(_, i)| rows.get(i).map(|r| (i, r.steps.len())));
    let reveal = recording_row.filter(|row| app.ui.actions.reveal_recording != Some(*row));
    app.ui.actions.reveal_recording = recording_row;
    if let Some((i, _)) = reveal {
        app.ui.actions.expanded.resize(rows.len(), false);
        if let Some(expanded) = app.ui.actions.expanded.get_mut(i) {
            *expanded = true;
        }
    }
    egui::ScrollArea::vertical().id_salt("actions-rows").max_height(max_h).min_scrolled_height(max_h).auto_shrink([false, false]).show(ui, |ui| {
        if rows.is_empty() {
            ui.label(egui::RichText::new(tl!("Record ● a sequence of edits, then play ▶ it on any document.")).color(t.text_faint).size(11.5));
        }
        if egui::DragAndDrop::has_payload_of_type::<DragRow>(ui.ctx()) {
            let clip = ui.clip_rect();
            if let Some(p) = ui.input(|i| i.pointer.hover_pos()).filter(|p| clip.contains(*p)) {
                let delta = if p.y < clip.top() + 22.0 {
                    8.0
                } else if p.y > clip.bottom() - 22.0 {
                    -8.0
                } else {
                    0.0
                };
                if delta != 0.0 {
                    ui.scroll_with_delta(vec2(0.0, delta));
                    ui.ctx().request_repaint();
                }
            }
        }
        let mut shown_set: Option<&str> = None;
        for i in display_order(&rows) {
            let Row { name, steps, disabled, dialogs, set } = &rows[i];
            // A set folder row before the first of its actions; collapsed folders hide them.
            if let Some(set) = set.as_deref()
                && shown_set != Some(set)
            {
                shown_set = Some(set);
                let collapsed = app.ui.actions.collapsed_sets.iter().any(|c| c == set);
                let (hr, hresp) = ui.allocate_exact_size(vec2(ui.available_width(), 24.0), Sense::click());
                hresp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::SelectableLabel, true, !collapsed, set));
                if hresp.hovered() {
                    ui.painter().rect_filled(hr, 0.0, t.hover.gamma_multiply(0.5));
                }
                let tri = Rect::from_center_size(pos2(hr.left() + 12.0, hr.center().y), vec2(8.0, 8.0));
                let pts = if collapsed {
                    vec![tri.left_top(), tri.right_center(), tri.left_bottom()]
                } else {
                    vec![tri.left_top(), tri.right_top(), tri.center_bottom()]
                };
                ui.painter().add(egui::Shape::convex_polygon(pts, t.text_dim, Stroke::NONE));
                crate::icons::paint(
                    ui,
                    Rect::from_center_size(pos2(hr.left() + 30.0, hr.center().y), vec2(16.0, 16.0)),
                    if collapsed { "folder" } else { "folder-open" },
                    12.0,
                    t.icon,
                );
                ui.painter().text(pos2(hr.left() + 44.0, hr.center().y), Align2::LEFT_CENTER, set, egui::FontId::proportional(12.0), t.text);
                let count = rows.iter().filter(|r| r.set.as_deref() == Some(set)).count() as u64;
                let summary = crate::i18n::trn(crate::i18n::current(), count, "{n} action", "{n} actions");
                ui.painter().text(pos2(hr.right() - 8.0, hr.center().y), Align2::RIGHT_CENTER, summary, egui::FontId::proportional(11.0), t.text_faint);
                if hresp.clicked() {
                    set_toggled = Some(set.to_string());
                }
                hresp.context_menu(|ui| {
                    if ui.button(tl!("Save Set…")).clicked() {
                        save_set = Some(set.to_string());
                        ui.close();
                    }
                });
            }
            if set.as_deref().is_some_and(|set| app.ui.actions.collapsed_sets.iter().any(|c| c == set)) {
                continue;
            }
            let indent = if set.is_some() { 14.0 } else { 0.0 };
            let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 26.0), Sense::click_and_drag());
            let x0 = rect.left() + indent;
            if reveal == Some((i, 0)) {
                ui.scroll_to_rect(rect, Some(egui::Align::BOTTOM));
            }
            let sel = app.ui.actions.selected == Some(i) && app.ui.actions.selected_step.is_none();
            resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::SelectableLabel, true, sel, name));
            if sel {
                ui.painter().rect_filled(rect, 0.0, t.row_selected);
            } else if resp.hovered() {
                ui.painter().rect_filled(rect, 0.0, t.hover.gamma_multiply(0.5));
            }
            if let Some(change) = drag_row(ui, &resp, DragRow { action: i, step: None }, &t) {
                moved = Some(change);
            }
            let expanded = app.ui.actions.expanded.get(i).copied().unwrap_or(false);
            let tri = Rect::from_center_size(pos2(x0 + 12.0, rect.center().y), vec2(8.0, 8.0));
            let pts =
                if expanded { vec![tri.left_top(), tri.right_top(), tri.center_bottom()] } else { vec![tri.left_top(), tri.right_center(), tri.left_bottom()] };
            ui.painter().add(egui::Shape::convex_polygon(pts, t.text_dim, Stroke::NONE));
            crate::icons::paint(ui, Rect::from_center_size(pos2(x0 + 30.0, rect.center().y), vec2(16.0, 16.0)), "play", 11.0, t.icon);
            let recording_this = recording.is_some_and(|(_, r)| r == i);
            let nsteps = steps.len();
            let name_rect = if app.ui.actions.renaming.as_ref().is_some_and(|r| r.index == i) {
                let field = Rect::from_min_max(pos2(x0 + 40.0, rect.top() + 3.0), pos2((rect.right() - 80.0).max(x0 + 120.0), rect.bottom() - 3.0));
                if let Some(text) = rename_field(app, ui, i, field) {
                    renamed = Some((i, text));
                }
                field
            } else {
                ui.painter().text(pos2(x0 + 44.0, rect.center().y), Align2::LEFT_CENTER, name, egui::FontId::proportional(12.0), t.text)
            };
            ui.painter().text(
                pos2(rect.right() - 8.0, rect.center().y),
                Align2::RIGHT_CENTER,
                app.session
                    .prefs()
                    .shortcuts
                    .get(&shortcut_id(name))
                    .filter(|s| !s.is_empty())
                    .map_or_else(|| step_count(nsteps, disabled.len()), |key| format!("{key} · {}", step_count(nsteps, disabled.len()))),
                egui::FontId::proportional(11.0),
                t.text_faint,
            );
            if recording_this {
                ui.painter().circle_filled(pos2(rect.right() - 64.0, rect.center().y), 4.0, Color32::from_rgb(230, 60, 60));
            }
            if resp.clicked() {
                app.ui.actions.selected = Some(i);
                app.ui.actions.selected_step = None;
                if resp.interact_pointer_pos().is_some_and(|p| p.x < x0 + 20.0) {
                    if app.ui.actions.expanded.len() <= i {
                        app.ui.actions.expanded.resize(i + 1, false);
                    }
                    app.ui.actions.expanded[i] = !expanded;
                }
            }
            resp.context_menu(|ui| {
                ui.label(tl!("Keyboard Shortcuts"));
                let current = app.session.prefs().shortcuts.get(&shortcut_id(name)).cloned().unwrap_or_default();
                for key in std::iter::once(String::new()).chain((1..=12).map(|n| format!("F{n}"))) {
                    let label = if key.is_empty() { tl!("None").to_string() } else { key.clone() };
                    if ui.selectable_label(current == key, label).clicked() {
                        let _ = assign_shortcut(app, name, &key);
                        ui.close();
                    }
                }
            });
            if resp.double_clicked() {
                // Photoshop renames in place when the name itself is double-clicked.
                if resp.interact_pointer_pos().is_some_and(|p| name_rect.expand2(vec2(4.0, 13.0)).contains(p)) {
                    app.ui.actions.renaming = Some(RenameAction { index: i, text: name.clone(), focused: false });
                } else {
                    play_idx = Some(i);
                }
            }
            if expanded {
                for (step, (id, params)) in steps.iter().enumerate() {
                    let (r, response) = ui.allocate_exact_size(vec2(ui.available_width(), 20.0), Sense::click_and_drag());
                    if reveal == Some((i, step + 1)) {
                        ui.scroll_to_rect(r, Some(egui::Align::BOTTOM));
                    }
                    let label = step_label(id, params);
                    let selected = app.ui.actions.selected == Some(i) && app.ui.actions.selected_step == Some(step);
                    response.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::SelectableLabel, true, selected, &label));
                    if selected {
                        ui.painter().rect_filled(r, 0.0, t.row_selected);
                    } else if response.hovered() {
                        ui.painter().rect_filled(r, 0.0, t.hover.gamma_multiply(0.5));
                    }
                    if let Some(change) = drag_row(ui, &response, DragRow { action: i, step: Some(step) }, &t) {
                        moved = Some(change);
                    }
                    let on = !disabled.contains(&step);
                    let color = if on { t.text_dim } else { t.text_faint };
                    ui.painter().text(pos2(x0 + 62.0, r.center().y), Align2::LEFT_CENTER, &label, egui::FontId::proportional(11.5), color);
                    // Photoshop's check column: unchecked steps are skipped by playback, Batch and droplets.
                    let check = ui.interact(
                        Rect::from_center_size(pos2(x0 + 30.0, r.center().y), vec2(16.0, 16.0)),
                        ui.id().with(("action-step-check", i, step)),
                        Sense::click(),
                    );
                    check.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Checkbox, true, on, format!("{}: {label}", tl!("Include in playback"))));
                    paint_check(ui, check.rect, on, &t);
                    // Photoshop's dialog column: playback shows the step's dialog, seeded with its
                    // recorded values, and continues when it is confirmed.
                    let with_dialog = dialogs.contains(&step);
                    let dialog = ui.interact(
                        Rect::from_center_size(pos2(x0 + 46.0, r.center().y), vec2(16.0, 16.0)),
                        ui.id().with(("action-step-dialog", i, step)),
                        Sense::click(),
                    );
                    // A toggle button, not a check box, so the check column stays the only CheckBox role.
                    dialog.widget_info(|| {
                        egui::WidgetInfo::selected(egui::WidgetType::Button, true, with_dialog, format!("{}: {label}", tl!("Show dialog during playback")))
                    });
                    crate::icons::paint(ui, dialog.rect, "message-square", 11.0, if with_dialog { t.icon } else { t.text_faint.gamma_multiply(0.45) });
                    let dialog = dialog.on_hover_text(tl!("Show dialog during playback"));
                    if check.on_hover_text(tl!("Include in playback")).clicked() {
                        toggled = Some((i, step));
                    } else if dialog.clicked() {
                        dialog_toggled = Some((i, step));
                    } else if response.clicked() {
                        app.ui.actions.selected = Some(i);
                        app.ui.actions.selected_step = Some(step);
                    }
                }
            }
        }
    });
    if let Some((action, step)) = toggled {
        let _ = app.run("actions.toggle", json!({"action": action, "step": step}));
    }
    if let Some((action, step)) = dialog_toggled {
        let _ = app.run("actions.dialog", json!({"action": action, "step": step}));
    }
    if let Some(set) = set_toggled {
        match app.ui.actions.collapsed_sets.iter().position(|c| *c == set) {
            Some(i) => {
                app.ui.actions.collapsed_sets.remove(i);
            }
            None => app.ui.actions.collapsed_sets.push(set),
        }
    }
    if let Some(set) = save_set
        && let Err(e) = save_dialog(app, Some(set))
        && e != "cancelled"
    {
        app.ui.status = e;
        app.ui.status_error = true;
    }
    if let Some((index, name)) = renamed {
        rename_action(app, index, name);
    }
    if let Some((source, to)) = moved {
        move_row(app, source, to);
    }
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        let recording_now = app.session.actions.recording.is_some();
        if crate::icons::button(ui, "square", 24.0, false, tl!("Stop playing/recording")).clicked() && recording_now {
            let _ = app.run("actions.stop", json!({}));
        }
        let (r, rec) = ui.allocate_exact_size(vec2(24.0, 24.0), Sense::click());
        if rec.hovered() {
            ui.painter().rect_filled(r, 3.0, t.hover);
        }
        ui.painter().circle_filled(r.center(), 5.5, if recording_now { Color32::from_rgb(230, 60, 60) } else { t.icon });
        if rec.on_hover_text(tl!("Begin recording")).clicked() && !recording_now {
            begin_recording(app, true);
        }
        let play = crate::icons::button(ui, "play", 24.0, false, tl!("Play selection"));
        play.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, tl!("Play selection")));
        if play.clicked() {
            play_idx = playback_selection(app);
        }
        if crate::icons::button(ui, "plus", 24.0, false, tl!("Create new action")).clicked() && !recording_now {
            begin_recording(app, false);
        }
        let can_delete = app.ui.actions.selected.is_some() && (!recording_now || app.ui.actions.selected_step.is_some());
        ui.add_enabled_ui(can_delete, |ui| {
            let response = crate::icons::button(ui, "trash", 24.0, false, tl!("Delete"));
            response.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, can_delete, tl!("Delete")));
            if response.clicked() {
                delete_selection(app);
            }
        });
    });
    if let Some(i) = play_idx {
        match app.run("actions.play", json!({"action": i})) {
            Ok(v) => report_play(app, &v),
            Err(e) => {
                let template = "Action playback failed: {error}";
                app.ui.status = crate::i18n::fmt(tl!(template), &[("error", &e)]);
                app.ui.status_error = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actions_list_grows_with_panel_and_keeps_footer_visible() {
        use egui_kittest::{Harness, kittest::Queryable};
        let bottom = |height: f32| {
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
            for i in 0..50 {
                app.session.actions.list.push(Action { name: format!("Action {i}"), steps: vec![], disabled: Default::default(), ..Default::default() });
            }
            let mut h = Harness::builder().with_size(vec2(320.0, height)).build_ui_state(|ui, app| panel(app, ui), app);
            h.run_steps(3);
            let rect = h.get_by_label("Play selection").rect();
            assert!(rect.bottom() <= height, "footer must fit: {rect:?}");
            rect.top()
        };
        assert!(bottom(700.0) - bottom(300.0) > 350.0, "list uses the additional panel height instead of stopping at 320 px");
        assert!(crate::dock::Group::History.scrolls_itself(1));
    }

    #[test]
    fn double_clicking_an_action_name_renames_it_in_place() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 32, "height": 32})).unwrap();
        app.session.actions.list.push(Action {
            name: "Action 1".into(),
            steps: vec![("layer.new.layer".into(), json!({}))],
            disabled: Default::default(),
            ..Default::default()
        });
        let layers = app.session.active().unwrap().doc.layers.len();
        // 60 fps steps, so two clicks a frame apart count as a double-click.
        let mut h = Harness::builder().with_size(vec2(320.0, 260.0)).with_step_dt(1.0 / 60.0).build_ui_state(|ui, app| panel(app, ui), app);
        h.run_steps(3);
        let rename = |h: &mut Harness<'_, PhotocraftApp>, text: &str| {
            let name = h.get_by_label("Action 1").rect().left_center() + vec2(52.0, 0.0);
            h.event(egui::Event::PointerMoved(name));
            // Wait out the double-click window of the previous attempt.
            h.run_steps(30);
            for pressed in [true, false, true, false] {
                h.event(egui::Event::PointerButton { pos: name, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE });
                h.step();
            }
            h.run_steps(3);
            h.event(egui::Event::Text(text.into()));
            h.run_steps(1);
            h.key_press(egui::Key::Enter);
            h.run_steps(3);
        };
        // Whitespace keeps the old name and reports why.
        rename(&mut h, "  ");
        assert_eq!(h.state().session.actions.list[0].name, "Action 1");
        assert!(h.state().ui.status_error);
        rename(&mut h, "Sepia");
        assert_eq!(h.state().session.actions.list[0].name, "Sepia");
        assert_eq!(h.state().session.active().unwrap().doc.layers.len(), layers, "renaming must not play the action");
    }

    #[test]
    fn play_button_targets_first_action_if_nothing_is_selected() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 32, "height": 32})).unwrap();
        assert_eq!(playback_selection(&app), None);
        app.session.actions.list.push(Action {
            name: "Make Layer".into(),
            steps: vec![("layer.new.layer".into(), json!({}))],
            disabled: Default::default(),
            ..Default::default()
        });
        app.session.actions.list.push(Action {
            name: "Fit".into(),
            steps: vec![("view.fitOnScreen".into(), json!({}))],
            disabled: Default::default(),
            ..Default::default()
        });
        assert_eq!(playback_selection(&app), Some(0));
        app.ui.actions.selected = Some(1);
        assert_eq!(playback_selection(&app), Some(1));
        app.ui.actions.selected = Some(999);
        assert_eq!(playback_selection(&app), Some(0));

        app.ui.actions.selected = None;
        use egui_kittest::{Harness, kittest::Queryable};
        let mut h = Harness::builder().with_size(vec2(400.0, 320.0)).build_ui_state(|ui, app| panel(app, ui), app);
        h.run_steps(3);
        let before = h.state().session.active().unwrap().doc.layers.len();
        h.get_by_label("Play selection").click();
        h.run_steps(3);
        assert_eq!(h.state().session.active().unwrap().doc.layers.len(), before + 1);
        assert!(!h.state().ui.status_error);
    }

    #[test]
    fn step_check_box_excludes_the_step_from_playback_and_batch() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 16, "height": 16})).unwrap();
        let steps = ["A", "B", "C"].map(|n| ("layer.new.layer".to_string(), json!({"name": n})));
        app.session.actions.list.push(Action::new("Three", steps.to_vec()));
        app.ui.actions.expanded = vec![true];
        let mut h = Harness::builder().with_size(vec2(320.0, 300.0)).build_ui_state(|ui, app| panel(app, ui), app);
        h.run_steps(3);
        let checks = h.query_all_by_role(egui::accesskit::Role::CheckBox).count();
        assert_eq!(checks, 3);
        h.query_all_by_role(egui::accesskit::Role::CheckBox).nth(1).unwrap().click();
        h.run_steps(3);
        let action = &h.state().session.actions.list[0];
        assert_eq!(action.disabled.iter().copied().collect::<Vec<_>>(), [1]);
        assert_eq!(action_steps(action).len(), 2, "Batch and droplets get only the checked steps");
        assert_eq!(step_count(3, action.disabled.len()), "2/3 steps");
        h.get_by_label("Play selection").click();
        h.run_steps(3);
        let names: Vec<_> = h.state().session.active().unwrap().doc.layers.iter().skip(1).map(|l| l.name.clone()).collect();
        assert_eq!(names, ["A", "C"]);
    }

    #[test]
    fn playback_errors_are_visible_in_status_bar() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 32, "height": 32})).unwrap();
        app.session.actions.list.push(Action {
            name: "Bad action".into(),
            steps: vec![("not.a.command".into(), json!({}))],
            disabled: Default::default(),
            ..Default::default()
        });
        let mut h = Harness::builder().with_size(vec2(400.0, 320.0)).build_ui_state(|ui, app| panel(app, ui), app);
        h.run_steps(3);
        h.get_by_label("Play selection").click();
        h.run_steps(3);
        assert!(h.state().ui.status_error);
        assert!(h.state().ui.status.contains("not.a.command") || h.state().ui.status.contains("unknown command"));
    }

    #[test]
    fn recording_reveals_new_action_and_latest_step_in_a_long_panel() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width":4,"height":4})).unwrap();
        for i in 0..30 {
            app.session.actions.list.push(Action { name: format!("Existing {i}"), steps: vec![], disabled: Default::default(), ..Default::default() });
        }
        let mut h = Harness::builder().with_size(vec2(320.0, 260.0)).build_ui_state(|ui, app| panel(app, ui), app);
        h.run_steps(4);
        begin_recording(h.state_mut(), false);
        h.run_steps(20);
        let rect = h.get_by_label("Action 31").rect();
        assert!(rect.top() >= 0.0 && rect.bottom() <= h.get_by_label("Play selection").rect().top(), "new action visible: {rect:?}");
        for _ in 0..20 {
            h.state_mut().run("view.zoomOut", json!({})).unwrap();
        }
        h.state_mut().run("view.fitOnScreen", json!({})).unwrap();
        h.run_steps(20);
        let rect = h.get_by_label("Fit on Screen").rect();
        assert!(rect.top() >= 0.0 && rect.bottom() <= h.get_by_label("Play selection").rect().top(), "latest step visible: {rect:?}");
    }

    #[test]
    fn drag_steps_and_action_headers_reorders_and_keeps_selection() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.actions.list.push(Action {
            name: "First".into(),
            steps: vec![("view.zoomOut".into(), json!({})), ("view.fitOnScreen".into(), json!({})), ("view.actualPixels".into(), json!({}))],
            disabled: Default::default(),
            ..Default::default()
        });
        app.session.actions.list.push(Action { name: "Second".into(), steps: vec![], disabled: Default::default(), ..Default::default() });
        app.ui.actions.expanded = vec![true, false];
        let mut h = Harness::builder().with_size(vec2(320.0, 400.0)).build_ui_state(|ui, app| panel(app, ui), app);
        h.run_steps(4);
        fn drag(h: &mut Harness<'_, PhotocraftApp>, from: egui::Pos2, to: egui::Pos2) {
            h.event(egui::Event::PointerMoved(from));
            h.run_steps(1);
            h.event(egui::Event::PointerButton { pos: from, button: egui::PointerButton::Primary, pressed: true, modifiers: egui::Modifiers::NONE });
            h.run_steps(1);
            h.event(egui::Event::PointerMoved(from + vec2(10.0, 0.0)));
            h.run_steps(2);
            h.event(egui::Event::PointerMoved(to));
            h.run_steps(2);
            h.event(egui::Event::PointerButton { pos: to, button: egui::PointerButton::Primary, pressed: false, modifiers: egui::Modifiers::NONE });
            h.run_steps(3);
        }
        let from = h.get_by_label("Zoom Out").rect().center();
        let target = h.get_by_label("100%").rect();
        drag(&mut h, from, pos2(target.center().x, target.bottom() - 2.0));
        assert_eq!(
            h.state().session.actions.list[0].steps.iter().map(|st| st.0.as_str()).collect::<Vec<_>>(),
            ["view.fitOnScreen", "view.actualPixels", "view.zoomOut"]
        );
        assert_eq!(h.state().ui.actions.selected_step, Some(2));
        let from = h.get_by_label("Second").rect().center();
        let target = h.get_by_label("First").rect();
        drag(&mut h, from, pos2(target.center().x, target.top() + 2.0));
        assert_eq!(h.state().session.actions.list[0].name, "Second");
        assert_eq!(h.state().ui.actions.expanded, [false, true]);
        assert_eq!(h.state().ui.actions.selected, Some(0));
    }

    #[test]
    fn nested_shell_actions_include_view_steps_and_propagate_failures() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width":40,"height":40})).unwrap();
        app.session.actions.list.push(Action {
            name: "Child".into(),
            steps: vec![("layer.new.layer".into(), json!({})), ("view.fitOnScreen".into(), json!({}))],
            disabled: Default::default(),
            ..Default::default()
        });
        app.run("actions.record", json!({"name":"Parent"})).unwrap();
        app.run("actions.play", json!({"action":"Child"})).unwrap();
        app.run("actions.stop", json!({})).unwrap();
        assert_eq!(app.session.actions.list[1].steps, [("actions.play".into(), json!({"action":"Child"}))]);
        assert_eq!(step_label("actions.play", &json!({"action":"Child"})), "Play Action: Child");
        app.run("view.actualPixels", json!({})).unwrap();
        let r = app.run("actions.play", json!({"action":"Parent"})).unwrap();
        assert!(r.get("failed").is_none(), "{r}");
        assert!(app.ui.views[0].fit_pending);
        app.session.authorize =
            Some(|id, _| if id == "view.fitOnScreen" { Err(photocraft_engine::EngineError::Other("denied nested view".into())) } else { Ok(()) });
        let r = app.run("actions.play", json!({"action":"Parent"})).unwrap();
        assert!(r["failed"]["error"].as_str().unwrap().contains("denied nested view"));
        assert!(app.session.actions.playback_stack.is_empty());
        assert_eq!(app.session.actions.playing, 0);
    }

    #[test]
    fn panel_shows_pending_steps_and_trash_deletes_only_the_clicked_step() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 40, "height": 40})).unwrap();
        begin_recording(&mut app, false);
        app.run("view.fitOnScreen", json!({})).unwrap();
        app.run("view.zoomOut", json!({})).unwrap();
        let mut h = Harness::builder().with_size(vec2(320.0, 400.0)).build_ui_state(|ui, app| panel(app, ui), app);
        h.run_steps(2);
        assert!(h.state().session.actions.recording.is_some());
        h.get_by_label("Fit on Screen");
        h.get_by_label("Zoom Out").click();
        h.run_steps(2);
        assert_eq!(h.state().ui.actions.selected_step, Some(1));
        h.get_by_label("Delete").click();
        h.run_steps(2);
        assert_eq!(visible_steps(h.state(), 0).iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), ["view.fitOnScreen"]);
        assert_eq!(h.state().session.actions.list.len(), 1);
        assert!(h.state().session.actions.recording.is_some());
        assert!(h.state().ui.actions.selected.is_none());
        delete_selection(h.state_mut());
        assert_eq!(h.state().session.actions.list.len(), 1, "repeated trash cannot delete the action");
        h.state_mut().run("view.zoomIn", json!({})).unwrap();
        h.run_steps(2);
        h.get_by_label("Zoom In");
        h.state_mut().run("actions.stop", json!({})).unwrap();
        h.run_steps(2);
        assert_eq!(visible_steps(h.state(), 0).iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), ["view.fitOnScreen", "view.zoomIn"]);
        h.get_by_label("Zoom In").click();
        h.run_steps(2);
        h.get_by_label("Delete").click();
        h.run_steps(2);
        assert_eq!(visible_steps(h.state(), 0).iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), ["view.fitOnScreen"]);
    }

    #[test]
    fn view_menu_commands_record_append_and_replay_after_resize() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width": 160, "height": 100})).unwrap();
        begin_recording(&mut app, false);
        app.run("image.canvasSize", json!({"width": 80, "height": 60})).unwrap();
        app.run("actions.stop", json!({})).unwrap();
        begin_recording(&mut app, true);
        crate::menus::invoke(&mut app, &ctx, "view.fitOnScreen", json!({})).unwrap();
        app.run("actions.stop", json!({})).unwrap();
        assert_eq!(app.session.actions.list.len(), 1);
        assert_eq!(app.session.actions.list[0].steps.iter().map(|s| s.0.as_str()).collect::<Vec<_>>(), ["image.canvasSize", "view.fitOnScreen"]);
        assert_eq!(label_of("view.fitOnScreen"), "Fit on Screen");
        // Persisted actions retain the menu step, and playback fits the new document.
        let saved = serde_json::to_string(&app.session.actions.list).unwrap();
        app.session.actions.list = serde_json::from_str(&saved).unwrap();
        app.run("file.new", json!({"width": 200, "height": 140})).unwrap();
        app.run("view.actualPixels", json!({})).unwrap();
        let result = app.run("actions.play", json!({"action": 0})).unwrap();
        assert_eq!(result["ran"], 2);
        assert!(result.get("failed").is_none(), "{result}");
        let idx = app.session.active_index().unwrap();
        assert!(app.ui.views[idx].fit_pending);
        let doc = &app.session.active().unwrap().doc;
        crate::canvas::fit_view(&mut app.ui.views[idx], doc, vec2(100.0, 100.0), 1.0, 1.0);
        assert_eq!(app.ui.views[idx].zoom, 0.75);
        assert_eq!(app.ui.views[idx].center, [40.0, 30.0]);
        assert!(!app.ui.views[idx].fit_pending);
    }

    #[test]
    fn view_action_failure_resume_and_authorization() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.actions.list.push(Action {
            name: "Fit".into(),
            steps: vec![("view.fitOnScreen".into(), json!({})), ("view.actualPixels".into(), json!({}))],
            disabled: Default::default(),
            ..Default::default()
        });
        let result = app.run("actions.play", json!({"action": "Fit"})).unwrap();
        assert_eq!(result["ran"], 0);
        assert_eq!(result["failed"]["step"], 0);
        assert_eq!(app.session.actions.playing, 0);
        app.run("file.new", json!({"width": 40, "height": 40})).unwrap();
        app.session.authorize =
            Some(
                |id, _| {
                    if id == "view.fitOnScreen" { Err(photocraft_engine::EngineError::BadParams { cmd: id.into(), msg: "denied view".into() }) } else { Ok(()) }
                },
            );
        let result = app.run("actions.play", json!({"action": 0})).unwrap();
        assert_eq!(result["ran"], 0);
        assert!(result["failed"]["error"].as_str().unwrap().contains("denied view"));
        let result = app.run("actions.play", json!({"action": 0, "from": 1})).unwrap();
        assert_eq!(result["ran"], 1);
        assert!(!app.ui.views[0].fit_pending);
        assert!(app.run("actions.play", json!({"action": 0, "from": -1})).is_err());
        app.session.actions.playing = 1;
        assert!(app.run("actions.play", json!({"action": 0})).is_err());
    }

    #[test]
    fn record_and_replay_on_another_document() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 40, "height": 40})).unwrap();
        app.sync_views();
        app.run("actions.record", json!({})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        app.run("select.rect", json!({"x": 0, "y": 0, "width": 10, "height": 10})).unwrap();
        app.run("edit.fill", json!({"color": "#ff0000"})).unwrap();
        let _ = app.run("document.pixel", json!({"x": 1, "y": 1}));
        app.run("actions.stop", json!({})).unwrap();
        let a = &app.session.actions.list[0];
        assert_eq!(a.steps.iter().map(|s| s.0.as_str()).collect::<Vec<_>>(), ["layer.new.layer", "select.rect", "edit.fill"]);
        app.run("file.new", json!({"width": 40, "height": 40})).unwrap();
        app.sync_views();
        let played = app.run("actions.play", json!({"action": 0})).unwrap();
        assert_eq!(played["ran"], 3);
        assert!(played.get("failed").is_none(), "{played}");
        let d = &app.session.active().unwrap().doc;
        assert_eq!(d.layers.len(), 2);
        assert_eq!(d.layers[1].surface().unwrap().pixel(5, 5), vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn recorded_save_as_and_save_replay_and_write_the_file() {
        let written = std::rc::Rc::new(std::cell::RefCell::new(Vec::<String>::new()));
        let log = written.clone();
        let services = crate::Services {
            export: Some(Box::new(|doc, path, _| {
                photocraft_io::export(doc, path, &Default::default()).map(|r| (r.bytes, r.warnings)).map_err(|e| e.to_string())
            })),
            write: Some(Box::new(move |path, _| {
                log.borrow_mut().push(path.to_string());
                Ok(())
            })),
            ..Default::default()
        };
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width": 16, "height": 16})).unwrap();
        app.run("actions.record", json!({})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        crate::menus::invoke(&mut app, &ctx, "file.saveAs", json!({"path": "out/result.psd"})).unwrap();
        crate::menus::invoke(&mut app, &ctx, "file.save", json!({})).unwrap();
        app.run("actions.stop", json!({})).unwrap();
        let steps = &app.session.actions.list[0].steps;
        assert_eq!(steps, &[("layer.new.layer".into(), json!({})), ("file.saveAs".into(), json!({"path": "out/result.psd"})), ("file.save".into(), json!({}))]);
        written.borrow_mut().clear();
        app.run("file.new", json!({"width": 16, "height": 16})).unwrap();
        let played = app.run("actions.play", json!({"action": 0})).unwrap();
        assert!(played.get("failed").is_none(), "{played}");
        assert_eq!(*written.borrow(), ["out/result.psd", "out/result.psd"]);
        assert_eq!(app.session.active().unwrap().path.as_deref(), Some("out/result.psd"));
    }

    #[test]
    fn set_folders_group_actions_and_collapse() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let mut sharpen = Action::new("Sharpen", vec![("filter.sharpen.sharpen".into(), json!({}))]);
        sharpen.set = Some("Retouch".into());
        let mut soften = Action::new("Soften", vec![("filter.blur.blur".into(), json!({}))]);
        soften.set = Some("Retouch".into());
        app.session.actions.list.push(Action::new("Loose", vec![]));
        app.session.actions.list.push(sharpen);
        app.session.actions.list.push(Action::new("Also loose", vec![]));
        app.session.actions.list.push(soften);
        let mut h = Harness::builder().with_size(vec2(360.0, 400.0)).build_ui_state(|ui, app| panel(app, ui), app);
        h.run_steps(3);
        // The folder row shows the set with its count; its actions sit under it, loose ones first.
        let folder = h.get_by_label("Retouch").rect();
        assert!(h.get_by_label("Loose").rect().top() < folder.top());
        assert!(h.get_by_label("Also loose").rect().top() < folder.top());
        assert!(h.get_by_label("Sharpen").rect().top() > folder.top());
        assert!(h.get_by_label("Soften").rect().top() > h.get_by_label("Sharpen").rect().top());
        assert_eq!(display_order(&[]), Vec::<usize>::new());
        // Clicking the folder collapses it; its actions disappear and the state is remembered.
        h.get_by_label("Retouch").click();
        h.run_steps(3);
        assert_eq!(h.state().ui.actions.collapsed_sets, ["Retouch"]);
        assert!(h.query_by_label("Sharpen").is_none());
        assert!(h.query_by_label("Loose").is_some());
        h.get_by_label("Retouch").click();
        h.run_steps(3);
        assert!(h.state().ui.actions.collapsed_sets.is_empty());
        assert!(h.query_by_label("Sharpen").is_some());
    }

    #[test]
    fn moving_an_action_next_to_another_set_joins_it() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let mut a = Action::new("A", vec![]);
        a.set = Some("One".into());
        let mut b = Action::new("B", vec![]);
        b.set = Some("Two".into());
        app.session.actions.list.push(a);
        app.session.actions.list.push(b);
        app.session.actions.list.push(Action::new("C", vec![]));
        // C dropped below B: it joins B's set, "Two".
        move_row(&mut app, DragRow { action: 2, step: None }, 2);
        assert_eq!(app.session.actions.list[2].set.as_deref(), Some("Two"));
        // B dropped to the top, above A: the row below decides when there is none above.
        move_row(&mut app, DragRow { action: 1, step: None }, 0);
        assert_eq!((app.session.actions.list[0].name.as_str(), app.session.actions.list[0].set.as_deref()), ("B", Some("One")));
        // C dropped between A (One) and nothing: the row above wins.
        move_row(&mut app, DragRow { action: 2, step: None }, 2);
        assert_eq!((app.session.actions.list[2].name.as_str(), app.session.actions.list[2].set.as_deref()), ("C", Some("One")));
    }

    #[test]
    fn insert_stop_dialog_adds_a_stop_after_the_selected_step() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 8, "height": 8})).unwrap();
        app.session.actions.list.push(Action::new("A", vec![("select.all".into(), json!({})), ("select.inverse".into(), json!({}))]));
        assert!(insert_stop_dialog(&mut app).is_err(), "needs a selected action");
        app.ui.actions.selected = Some(0);
        app.ui.actions.selected_step = Some(0);
        let id = insert_stop_dialog(&mut app).unwrap()["dialog"].as_u64().unwrap();
        let d = app.ui.dialog_mut(id).unwrap();
        assert!(owns_insert_stop(&d.fields));
        assert_eq!(d.fields["at"], 1);
        d.fields.insert("message".into(), json!("Check the mask"));
        d.fields.insert("allowContinue".into(), json!(true));
        crate::dialogs::confirm(&mut app, id).unwrap();
        let steps = &app.session.actions.list[0].steps;
        assert_eq!(steps[1], (photocraft_engine::atn_cmds::STOP_MESSAGE.to_string(), json!({"message": "Check the mask", "allowContinue": true})));
        assert_eq!((app.ui.actions.selected, app.ui.actions.selected_step), (Some(0), Some(1)));
        assert_eq!(step_label(&steps[1].0, &steps[1].1), "Stop: Check the mask");
        // Without a selected step the stop goes last; the panel's menu wiring is exercised by name.
        app.ui.actions.selected_step = None;
        let id = insert_stop_dialog(&mut app).unwrap()["dialog"].as_u64().unwrap();
        assert_eq!(app.ui.dialog_mut(id).unwrap().fields["at"], 3);
        crate::dialogs::cancel(&mut app, id).unwrap();
        assert_eq!(app.session.actions.list[0].steps.len(), 3);
    }

    #[test]
    fn a_dialog_toggled_step_pauses_playback_and_resumes_on_ok() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 16, "height": 16})).unwrap();
        let mut a = Action::new(
            "Blur",
            vec![("layer.new.layer".into(), json!({})), ("filter.blur.gaussianBlur".into(), json!({"radius": 3})), ("select.all".into(), json!({}))],
        );
        a.dialogs.insert(1);
        app.session.actions.list.push(a);
        let v = play(&mut app, &json!({"action": "Blur"})).unwrap();
        assert_eq!(v["ran"], 1, "{v}");
        assert_eq!(v["paused"]["step"], 1);
        let id = v["paused"]["dialog"].as_u64().unwrap();
        let d = app.ui.dialog_mut(id).unwrap();
        assert_eq!(d.fields["radius"], 3, "the dialog opens on the recorded values");
        assert_eq!(d.fields[RESUME], json!({"action": "Blur", "from": 2}));
        assert!(app.session.active().unwrap().doc.selection.is_none(), "nothing after the dialog ran yet");
        report_play(&mut app, &v);
        assert!(app.ui.status.contains("dialog"), "{}", app.ui.status);
        // OK runs the blur with the dialog's values and plays the rest.
        app.ui.dialog_mut(id).unwrap().fields.insert("radius".into(), json!(5));
        crate::dialogs::confirm(&mut app, id).unwrap();
        assert!(app.ui.dialogs.is_empty());
        assert!(app.session.active().unwrap().doc.selection.is_some(), "the step after the dialog ran");
        assert!(app.ui.status.starts_with("Played"), "{}", app.ui.status);
        // Cancel stops the action there.
        app.run("select.deselect", json!({})).unwrap();
        let v = play(&mut app, &json!({"action": "Blur"})).unwrap();
        let id = v["paused"]["dialog"].as_u64().unwrap();
        crate::dialogs::cancel(&mut app, id).unwrap();
        assert!(app.session.active().unwrap().doc.selection.is_none());
        assert!(app.ui.status.contains("stopped"), "{}", app.ui.status);
        // With background jobs (the desktop default) the resumed steps wait for the dialog's
        // filter to land instead of meeting a locked document.
        app.background_jobs = true;
        let v = play(&mut app, &json!({"action": "Blur"})).unwrap();
        let id = v["paused"]["dialog"].as_u64().unwrap();
        crate::dialogs::confirm(&mut app, id).unwrap();
        assert!(app.session.active().unwrap().doc.selection.is_some(), "the step after the background blur ran");
        assert!(app.ui.status.starts_with("Played"), "{}", app.ui.status);
        app.background_jobs = false;
        // Agents play headlessly: no pause, the recorded values run.
        app.automation_input = true;
        let v = play(&mut app, &json!({"action": "Blur"})).unwrap();
        assert_eq!(v["ran"], 3, "{v}");
        assert!(v.get("paused").is_none());
    }
}
