//! Programmatic control of the running app, for agents, tests and (soon) MCP.
//!
//! Transport-agnostic: a transport thread (TCP in the desktop app, a channel in tests) sends
//! [`ControlRequest`]s; the UI thread handles them between frames and replies with JSON.
//!
//! Methods:
//! - `engine.execute {command, params}`: run any engine or UI command by id
//! - `engine.commands`: list commands with enablement
//! - `ui.inspect`: full UI state (tool, panels, views, dialogs, windows, window size); the menu
//!   tree is `ui.menu.list`
//! - `ui.set {tool?, panels?, dock?, dockTabs?, dockWidth?, colorPanel?, maskTarget?, vectorMaskTarget?, selectionMode?, zoom?, center?, rotation?, fit?, theme?, brushSection?, brushTab?, brushesView?, brushPicker?, brushPickerName?, brushPickerStroke?, brushPickerTip?, brushPickerScale?, brushSize?}`:
//!   change UI state; any other field is an error ([`UI_SET_FIELDS`])
//! - `ui.dialog.open {kind, fields?}` (kinds: newDocument, about, layerStyle {effect?}, colorPicker {target: foreground|background}, command {command}) / `ui.dialog.set {dialog, field, value}` / `ui.dialog.confirm {dialog, wait?}` / `ui.dialog.cancel {dialog}`
//! - `ui.dialog.apply {dialog}`: commit Preferences changes without closing the dialog
//! - `ui.window.open {document?}` / `ui.window.close {window}`: extra document windows
//! - `ui.pointer {events: [{kind: down|move|up, x, y, pressure?, tiltX?, tiltY?, rotation?}], tool?, modifiers?, button?}`: drive the active tool in document coordinates (`button: "secondary"` opens the tool's canvas context menu (the pasteboard colour menu outside the image) or Brush Preset picker, or erases with Preferences › Tools › Right-click with painting tools = erase)
//! - `ui.click {x, y, button?, count?}` / `ui.move {x, y}`: synthetic pointer input in screen points
//!   (`count` at most [`MAX_CLICKS`])
//! - `ui.key {key, command?, shift?, alt?, ctrl?}` / `ui.type {text}`: synthetic keyboard input
//! - The synthetic input methods reject unknown fields, wrong types, unknown enum values and
//!   out-of-range numbers before acting, naming the parameter (#2449)
//! - `ui.resize {width, height}`: resize the main window
//! - `ui.gpu.simulateLoss {error?}`: act as if the wgpu device was lost (or, with `error: true`,
//!   reported an error): the app switches to the CPU renderer for the rest of the session, as on a
//!   real loss. For testing the fallback; returns whether a GPU canvas was active
//! - `ui.screenshot {path?, focus?}`: capture the main window (PNG). Raises the window first (default)
//!   because occluded macOS windows stop rendering
//! - `ui.focus`: bring the main window to the front
//! - `app.open {path}` / `app.save {path}`: relative file I/O under the automation roots; reply with `warnings`
//! - `app.quit`
//! - `jobs.list` / `jobs.cancel {job?}`: background jobs (#210) with progress; cancel one (or all).
//!   `engine.execute`, `ui.menu.invoke` and `ui.dialog.confirm` wait for a command that runs as a
//!   job unless `wait: false` (then the reply is `{job, pending: true}`)

use std::sync::mpsc::Sender;

use base64::Engine as _;
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::canvas::{ToolEvent, tool_event};
use crate::state::{DialogKind, Tool};

pub type ControlResponse = Value;

/// The control server's per-request reply deadline.
pub const REPLY_DEADLINE: std::time::Duration = std::time::Duration::from_secs(60);

/// The error the transport thread returns when a dispatched command gave no reply within
/// [`REPLY_DEADLINE`]. It warns that the command may still apply: re-sending blindly could apply
/// an edit twice (#3662). Built from the deadline constant, so the text cannot drift from it.
pub fn timeout_error() -> String {
    format!(
        "timeout: no reply in {}s — the command may still be running and may still apply, or the app was not drawing frames (display asleep, window fully occluded, or a long operation blocked the frame loop)",
        REPLY_DEADLINE.as_secs()
    )
}

/// The error for a request still queued when its deadline passed, from the UI thread or the
/// transport (see [`Dispatch`]). It was dropped before dispatch, so re-sending it is safe (#3662). Same `timeout` prefix as
/// [`timeout_error`], which machines match on.
pub fn expired_error() -> String {
    format!(
        "timeout: not run — the request was still queued when its {}s deadline passed, so it was dropped before dispatch and is safe to re-send; the app was not drawing frames (display asleep, window fully occluded, or a long operation blocked the frame loop)",
        REPLY_DEADLINE.as_secs()
    )
}

pub struct ControlRequest {
    pub method: String,
    pub params: Value,
    pub reply: Sender<ControlResponse>,
    /// Requests still queued at this instant are rejected; already dispatched work continues.
    pub deadline: Option<std::time::Instant>,
    /// Shared with the transport, so that when it stops waiting it knows whether the command ran.
    pub dispatch: Dispatch,
}

impl ControlRequest {
    pub fn new(method: impl Into<String>, params: Value) -> (Self, std::sync::mpsc::Receiver<ControlResponse>) {
        let (tx, rx) = std::sync::mpsc::channel();
        (Self { method: method.into(), params, reply: tx, deadline: None, dispatch: Dispatch::default() }, rx)
    }
}

/// Whether a request was dispatched, settled once between the UI thread (about to run it) and
/// the transport (giving up on its reply). The transport's clock runs out no later than the UI
/// thread's, so without this it could only ever say the command may still apply (#3662).
#[derive(Clone, Default)]
pub struct Dispatch(std::sync::Arc<std::sync::atomic::AtomicU8>);

const QUEUED: u8 = 0;
const DISPATCHED: u8 = 1;
const ABANDONED: u8 = 2;

impl Dispatch {
    /// The UI thread claims the request to run it: false once the transport gave up on it. A
    /// request the UI thread rejects as expired stays queued, which `give_up` reads as not run.
    pub(crate) fn start(&self) -> bool {
        self.settle(DISPATCHED) == DISPATCHED
    }

    /// The transport stops waiting: the error to reply with, [`expired_error`] when the command
    /// can no longer run and [`timeout_error`] when it was dispatched and may still apply.
    pub fn give_up(&self) -> String {
        if self.settle(ABANDONED) == DISPATCHED { timeout_error() } else { expired_error() }
    }

    /// Moves a queued request to `to`; returns the state it ends in.
    fn settle(&self, to: u8) -> u8 {
        use std::sync::atomic::Ordering;
        self.0.compare_exchange(QUEUED, to, Ordering::AcqRel, Ordering::Acquire).map_or_else(|state| state, |_| to)
    }
}

pub enum Outcome {
    Done(Value),
    Screenshot {
        token: u64,
        path: Option<String>,
    },
    /// Synthetic input queued: reply once the app has processed all of it (so a following
    /// `ui.inspect`/`engine.execute` observes the effect).
    AfterInput,
    /// A command started a background job and the caller waits for it: reply with its result
    /// once it has been applied (or with its error / "cancelled").
    AfterJob(photocraft_engine::jobs::JobId),
}

/// The fields `ui.set` reads. Anything else is rejected before a field is applied, and every
/// field's value is validated before the first one is applied, so a typo, an unknown field, a
/// bad value or a bad nested key can't reply with success while nothing — or only half of it —
/// changed (#412).
pub const UI_SET_FIELDS: [&str; 35] = [
    "tool",
    "panels",
    "dock",
    "dockTabs",
    "dockWidth",
    "colorPanel",
    "maskTarget",
    "vectorMaskTarget",
    "selectionMode",
    "zoom",
    "center",
    "rotation",
    "fit",
    "theme",
    "brushSection",
    "brushTab",
    "brushesView",
    "brushPicker",
    "brushPickerName",
    "brushPickerStroke",
    "brushPickerTip",
    "brushPickerScale",
    "brushSize",
    "gradientBlendMode",
    "gradientClassic",
    "eyedropperSampleSize",
    "eyedropperSample",
    "eyedropperRing",
    "cropOverlay",
    "cropOverlayShow",
    "cropOverlayOrientation",
    "cropShield",
    "shapeStroke",
    "camTransformOnDrop",
    "tileGrid",
];

/// Most clicks one `ui.click` may queue (#982). Each click is a press and a release that the app
/// feeds in its own frame before replying, so `count` sizes both the input queue and the wait;
/// the same ceiling as the steps of one batch request.
pub const MAX_CLICKS: u64 = 256;

fn ok(v: Value) -> Outcome {
    Outcome::Done(json!({"ok": true, "result": v}))
}
fn err(e: impl std::fmt::Display) -> Outcome {
    Outcome::Done(json!({"ok": false, "error": e.to_string()}))
}

/// A `ui.set` boolean field: `Ok(None)` when absent, an error when present with another type.
fn bool_field(p: &Value, key: &str) -> std::result::Result<Option<bool>, String> {
    match p.get(key) {
        None => Ok(None),
        Some(v) => v.as_bool().map(Some).ok_or_else(|| format!("{key} must be a boolean")),
    }
}

/// A `ui.set` integer field: `Ok(None)` when absent; whole-number floats (`12.0`, as UIs send
/// them) are accepted, anything else is an error.
fn uint_field(p: &Value, key: &str) -> std::result::Result<Option<u64>, String> {
    match p.get(key) {
        None => Ok(None),
        Some(v) if v.is_u64() => Ok(v.as_u64()),
        Some(v) => v.as_f64().filter(|f| f.fract() == 0.0 && *f >= 0.0).map(|f| Some(f as u64)).ok_or_else(|| format!("{key} must be a non-negative integer")),
    }
}

/// A `ui.set` numeric field: `Ok(None)` when absent, finite numbers only.
fn num_field(p: &Value, key: &str) -> std::result::Result<Option<f64>, String> {
    match p.get(key) {
        None => Ok(None),
        Some(v) => v.as_f64().filter(|f| f.is_finite()).map(Some).ok_or_else(|| format!("{key} must be a finite number")),
    }
}

/// A `ui.set` object field merged into the current state (e.g. `panels`): the patch's keys are
/// merged into the serialized current value and parsed back, so a non-object value or an
/// unknown nested key is an error — a nested typo must not reply ok while changing nothing.
fn merged_object<T: serde::Serialize + serde::de::DeserializeOwned>(current: &T, patch: Option<&Value>, field: &str) -> std::result::Result<Option<T>, String> {
    let Some(patch) = patch else { return Ok(None) };
    let Some(patch) = patch.as_object() else { return Err(format!("{field} must be an object")) };
    let mut cur = serde_json::to_value(current).map_err(|e| e.to_string())?;
    let known: std::collections::HashSet<String> = cur.as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default();
    for (k, v) in patch {
        if !known.contains(k) {
            return Err(format!("unknown {field} field `{k}`"));
        }
        if let Some(c) = cur.as_object_mut() {
            c.insert(k.clone(), v.clone());
        }
    }
    serde_json::from_value(cur).map(Some).map_err(|e| format!("{field}: {e}"))
}

/// A `ui.set` object field applied by whole-value replacement (e.g. `dock`, `colorPanel`):
/// unknown nested keys are rejected like [`merged_object`] does, but the value replaces the
/// state instead of merging into it.
fn whole_object<T: serde::Serialize + serde::de::DeserializeOwned>(current: &T, v: Option<&Value>, field: &str) -> std::result::Result<Option<T>, String> {
    let Some(v) = v else { return Ok(None) };
    let Some(patch) = v.as_object() else { return Err(format!("{field} must be an object")) };
    let cur = serde_json::to_value(current).map_err(|e| e.to_string())?;
    let known: std::collections::HashSet<&str> = cur.as_object().map(|o| o.keys().map(String::as_str).collect()).unwrap_or_default();
    for k in patch.keys() {
        if !known.contains(k.as_str()) {
            return Err(format!("unknown {field} field `{k}`"));
        }
    }
    serde_json::from_value(v.clone()).map(Some).map_err(|e| format!("{field}: {e}"))
}

/// The buttons `ui.pointer` and `ui.click` accept (`right` and `secondary` are the same one).
const BUTTONS: [&str; 4] = ["left", "right", "secondary", "middle"];
/// The modifier flags `ui.key` reads, top-level or under `modifiers`; `ui.pointer` adds `space`.
const KEY_MODIFIERS: [&str; 4] = ["shift", "alt", "command", "ctrl"];
const POINTER_MODIFIERS: [&str; 5] = ["shift", "alt", "command", "ctrl", "space"];

/// A synthetic input method's params (or one nested object, named `what`): an object whose keys
/// are all in `fields`, returned with its null fields dropped so they mean "omitted". A typo must
/// not reply ok after acting on a default (#2449), as `ui.set` already promises (#412).
fn input_object(v: &Value, what: Option<&str>, fields: &[&str]) -> std::result::Result<Value, String> {
    let object = match (v, what) {
        (Value::Null, None) => return Ok(json!({})),
        (Value::Object(o), _) => o,
        (_, what) => return Err(format!("{} must be an object", what.unwrap_or("params"))),
    };
    if let Some(key) = object.keys().find(|k| !fields.contains(&k.as_str())) {
        let field = what.map(|w| format!("{w} field")).unwrap_or_else(|| "field".into());
        return Err(format!("unknown {field} `{key}` (fields: {})", fields.join(", ")));
    }
    Ok(Value::Object(object.iter().filter(|(_, v)| !v.is_null()).map(|(k, v)| (k.clone(), v.clone())).collect()))
}

/// An optional finite number named `name` in errors, within `range` when given.
fn input_number(v: &Value, key: &str, name: &str, range: Option<(f64, f64)>) -> std::result::Result<Option<f64>, String> {
    let n = num_field(v, key).map_err(|_| format!("{name} must be a finite number"))?;
    match (n, range) {
        (Some(n), Some((lo, hi))) if !(lo..=hi).contains(&n) => Err(format!("{name} must be between {lo} and {hi} (got {n})")),
        _ => Ok(n),
    }
}

/// An optional string named `name` in errors, one of `accepted` when given.
fn input_choice<'a>(v: &'a Value, key: &str, name: &str, accepted: &[&str]) -> std::result::Result<Option<&'a str>, String> {
    match v.get(key) {
        None => Ok(None),
        Some(Value::String(s)) if accepted.is_empty() || accepted.contains(&s.as_str()) => Ok(Some(s)),
        Some(Value::String(s)) => Err(format!("unknown {name} `{s}` ({})", accepted.join(", "))),
        Some(_) => Err(format!("{name} must be a string")),
    }
}

/// The modifier flags of `ui.pointer` / `ui.key`: grouped under `modifiers` when present (the
/// top-level flags are then ignored, but still type-checked), else top-level.
fn input_modifiers(p: &Value, flags: &[&str]) -> std::result::Result<Value, String> {
    for flag in flags {
        bool_field(p, flag)?;
    }
    let Some(m) = p.get("modifiers") else { return Ok(p.clone()) };
    let m = input_object(m, Some("modifiers"), flags)?;
    for flag in flags {
        bool_field(&m, flag).map_err(|e| format!("modifiers.{e}"))?;
    }
    Ok(m)
}

fn wrap(r: Result<Value, String>) -> Outcome {
    match r {
        Ok(v) => ok(v),
        Err(e) => err(e),
    }
}

/// Run a command the way automation does (script events off). Long commands may run as
/// background jobs: by default the reply waits for the job's result (backward compatible); with
/// `wait` false it is `{job, pending: true}` at once.
fn run_waiting(app: &mut PhotocraftApp, wait: bool, run: impl FnOnce(&mut PhotocraftApp) -> Result<Value, String>) -> Outcome {
    let events_enabled = app.session.prefs().script_events.enabled;
    if events_enabled {
        app.session.edit_prefs(|prefs| prefs.script_events.enabled = false);
    }
    app.jobs.last_started = None;
    let result = run(app);
    if events_enabled {
        app.session.edit_prefs(|prefs| prefs.script_events.enabled = true);
    }
    match (result, app.jobs.last_started.take()) {
        (Ok(_), Some(job)) if wait => Outcome::AfterJob(job),
        (result, _) => wrap(result),
    }
}

/// Screen position of document point (x, y) on the main canvas, where a `ui.pointer` right-click
/// opens its menu (the mouse's opens at the pointer); the canvas centre when it can't be mapped.
fn screen_point(app: &PhotocraftApp, x: f64, y: f64) -> [f32; 2] {
    let p = crate::canvas::ViewXform::active(app)
        .map(|xf| xf.to_screen(x as f32, y as f32))
        .filter(|p| p.is_finite())
        .unwrap_or_else(|| app.last_canvas_rect.center());
    [p.x, p.y]
}

/// Modifier flags of a request, top-level or grouped under `modifiers`.
///
/// Off the Mac, Ctrl is the shortcut key, so egui wants `ctrl` and `command` to carry the same
/// value (`egui::Modifiers::command`: "On Windows and Linux, set this to the same value as
/// `ctrl`") and shortcuts are matched on `command`. A real Ctrl press arrives with both set, so
/// a request that sets only one of them gets both here and reaches the same shortcuts. On the Mac
/// they are different keys and stay apart: `command` is the ⌘ key, `ctrl` the Control key.
fn modifiers_from(p: &Value) -> egui::Modifiers {
    let m = p.get("modifiers").unwrap_or(p);
    let flag = |k: &str| m.get(k).and_then(Value::as_bool).unwrap_or(false);
    let (ctrl, command) = (flag("ctrl"), flag("command"));
    egui::Modifiers {
        shift: flag("shift"),
        alt: flag("alt"),
        command: if cfg!(target_os = "macos") { command } else { command || ctrl },
        mac_cmd: cfg!(target_os = "macos") && command,
        ctrl: if cfg!(target_os = "macos") { ctrl } else { command || ctrl },
    }
}

/// The `dialog` id of the shell's own dialog (`workspace_ui`) in `ui.inspect` and `ui.dialog.*`.
const SHELL_DIALOG: &str = "shell";

pub fn handle(app: &mut PhotocraftApp, ctx: &egui::Context, req: &ControlRequest) -> Outcome {
    let saved = app.session.authorize;
    if let Some(gate) = app.services.automation_authorize {
        app.session.authorize = Some(gate);
    }
    let outcome = dispatch(app, ctx, req);
    app.session.authorize = saved;
    outcome
}

/// The open canvas context menu's rows for `ui.state`: the spelling suggestions first (type edit
/// menu over a misspelled word), then the menu's own rows.
fn context_entries(app: &PhotocraftApp, menu: &crate::canvas_tool_menu::CanvasToolMenu) -> Vec<Value> {
    use crate::canvas_tool_menu::{entry_enabled, rows, spelling_rows};
    let spelling = spelling_rows(menu);
    let mut entries: Vec<Value> = spelling.iter().map(|(label, id)| json!({"label": label, "id": id, "enabled": entry_enabled(app, menu, id)})).collect();
    if !spelling.is_empty() {
        entries.push(json!({"separator": true}));
    }
    entries.extend(rows(menu).iter().map(|row| match row {
        Some((label, id)) => json!({"label": label, "id": id, "enabled": entry_enabled(app, menu, id)}),
        None => json!({"separator": true}),
    }));
    entries
}

fn dispatch(app: &mut PhotocraftApp, ctx: &egui::Context, req: &ControlRequest) -> Outcome {
    let p = &req.params;
    let s = |k: &str| p.get(k).and_then(Value::as_str);
    let u = |k: &str| p.get(k).and_then(Value::as_u64);
    let wait = p.get("wait").and_then(Value::as_bool).unwrap_or(true);
    match req.method.as_str() {
        "ui.context.choose" => {
            let Some(id) = s("id") else { return err("missing `id`") };
            let Some(menu) = app.ui.canvas_tool_menu.as_ref() else { return err("no canvas context menu is open") };
            if !crate::canvas_tool_menu::available(app, menu, id) {
                return err("context action is unavailable");
            }
            // A pasteboard row changes a preference: the policy sees the `prefs.set` it runs.
            let (command, params) = crate::canvas_tool_menu::engine_call(menu, id);
            if let Some(authorize) = app.services.automation_command.as_ref()
                && let Err(error) = authorize(&command, &params)
            {
                return err(error);
            }
            crate::canvas_tool_menu::choose(app, ctx, id);
            ok(json!({"command": id, "dialog": app.ui.dialogs.last().map(|d| d.id)}))
        }
        "engine.execute" | "ui.menu.invoke" => {
            let Some(id) = s("command").or(s("id")) else { return err("missing `command`") };
            let params = p.get("params").cloned().unwrap_or(json!({}));
            // A control-channel menu click must respect the same modal gate as the native menu.
            // Camera Raw's documented `ui` requests operate its already-open dialog rather
            // than click a menu. Only that continuation may pass its own modal gate.
            // engine.execute is deliberately not subject to the UI's menu-click semantics.
            let camera_raw_continuation = id == "filter.cameraRaw"
                && app.camera_raw.is_some()
                && params.as_object().is_some_and(|fields| fields.len() == 1 && fields.get("ui").is_some_and(Value::is_object))
                && app.ui.dialogs.is_empty()
                && app.discard.is_none()
                && !crate::crop_ui::blocks(app, id);
            if req.method == "ui.menu.invoke" && !camera_raw_continuation && !crate::menus::modal_allows(app, id) {
                return err("menu command is unavailable while a modal dialog is open");
            }
            if let Some(authorize) = app.services.automation_command.as_ref()
                && let Err(error) = authorize(id, &params)
            {
                return err(error);
            }
            // `engine.execute` is programmatic: engine commands run directly with their default
            // params and never open a dialog (an agent would otherwise get a modal instead of a
            // result). `ui.menu.invoke` behaves like a menu click, so it may open the dialog.
            if req.method == "engine.execute" && photocraft_engine::commands::find(id).is_some() {
                return run_waiting(app, wait, |app| app.run(id, params));
            }
            run_waiting(app, wait, |app| crate::menus::invoke(app, ctx, id, params))
        }
        "engine.commands" => wrap(app.run("command.list", json!({}))),
        // Background jobs (#210): running ones with progress, then the last few that ended.
        "jobs.list" => wrap(app.session.execute("jobs.list", json!({})).map_err(|e| e.to_string())),
        "jobs.cancel" => {
            let all = p.get("job").is_none_or(Value::is_null);
            match p.get("job").and_then(Value::as_u64) {
                Some(j) if app.session.job(photocraft_engine::jobs::JobId(j)).is_some() => {
                    let cancelled = crate::jobs_ui::cancel(app, photocraft_engine::jobs::JobId(j));
                    ok(json!({"cancelled": usize::from(cancelled)}))
                }
                Some(j) => err(format!("no running job {j}")),
                None if all => {
                    let ids: Vec<_> = app.session.jobs().into_iter().map(|j| j.id).collect();
                    let cancelled = ids.into_iter().filter(|j| crate::jobs_ui::cancel(app, *j)).count();
                    ok(json!({"cancelled": cancelled}))
                }
                None => err("`job` must be a job id"),
            }
        }
        "ui.menu.list" => ok(serde_json::to_value(crate::menus::menu_items(app)).unwrap_or_default()),
        "ui.inspect" => ok(inspect(app, ctx)),
        "ui.set" => {
            if let Some(field) = p.as_object().and_then(|o| o.keys().find(|k| !UI_SET_FIELDS.contains(&k.as_str()))) {
                return err(format!("unknown field `{field}` (fields: {})", UI_SET_FIELDS.join(", ")));
            }
            let gradient_blend = if let Some(value) = p.get("gradientBlendMode") {
                let Some(name) = value.as_str() else { return err("gradientBlendMode must be a blend mode name") };
                let Some(mode) = photocraft_engine::commands::blend_from_str(name).filter(|m| photocraft_color::BlendMode::LAYER_MODES.contains(m)) else {
                    return err(format!("unknown gradient blend mode `{name}`"));
                };
                Some(mode)
            } else {
                None
            };
            // Everything is validated and materialized before the first field is applied, so a
            // rejected call applies none of its fields (#412's guarantee, down to the values
            // and nested keys).
            let applied = (|| -> std::result::Result<Value, String> {
                let shape_stroke = match p.get("shapeStroke") {
                    Some(v) => {
                        if let Some(object) = v.as_object() {
                            let keys = ["width", "opacity", "align", "cap", "join", "miterLimit", "dashes", "dashOffset"];
                            if let Some(key) = object.keys().find(|key| !keys.contains(&key.as_str())) {
                                return Err(format!("unknown shapeStroke field `{key}`"));
                            }
                        }
                        let o = &app.ui.tool_options;
                        let base = o.shape_stroke.stroke(o.stroke_width, photocraft_doc::Fill::Solid(photocraft_doc::Color::BLACK));
                        Some(photocraft_engine::vector_cmds::parse_stroke(v, Some(base), photocraft_doc::Color::BLACK)?)
                    }
                    None => None,
                };
                let tool = match s("tool") {
                    Some(t) => Tool::from_name(t).map(Some).ok_or_else(|| format!("unknown tool `{t}`"))?,
                    None => None,
                };
                let gradient_classic = bool_field(p, "gradientClassic")?;
                // The Eyedropper's options bar (#1649): Sample Size, Sample, Show Sampling Ring.
                let eyedropper_size = match p.get("eyedropperSampleSize") {
                    Some(v) => Some(
                        photocraft_engine::sample_cmds::size_param("ui.set", &json!({"size": v}))
                            .map_err(|_| format!("eyedropperSampleSize must be \"point\" or one of {:?}", photocraft_engine::sample_cmds::SAMPLE_SIZES))?,
                    ),
                    None => None,
                };
                let eyedropper_sample = match p.get("eyedropperSample") {
                    Some(v) => {
                        let ids: Vec<&str> = photocraft_engine::sample_cmds::SampleLayers::ALL.iter().map(|(_, n)| *n).collect();
                        let id = v.as_str().filter(|id| ids.contains(id)).ok_or_else(|| format!("eyedropperSample must be one of {}", ids.join(", ")))?;
                        Some(id.to_string())
                    }
                    None => None,
                };
                let eyedropper_ring = bool_field(p, "eyedropperRing")?;
                // Content-Aware Move's Transform On Drop checkbox.
                let cam_transform_on_drop = bool_field(p, "camTransformOnDrop")?;
                // Developer overlay of the active layer's storage tiles (tile_overlay.rs).
                let tile_grid = bool_field(p, "tileGrid")?;
                // The Crop tool's overlay menu (#1919).
                let crop_overlay = match p.get("cropOverlay") {
                    Some(v) => Some(v.as_str().and_then(crate::crop_overlay::CropOverlay::from_id).ok_or_else(|| {
                        let ids: Vec<&str> = crate::crop_overlay::CropOverlay::ALL.iter().map(|k| k.id()).collect();
                        format!("cropOverlay must be one of {}", ids.join(", "))
                    })?),
                    None => None,
                };
                let crop_overlay_show = match p.get("cropOverlayShow") {
                    Some(v) => Some(
                        v.as_str()
                            .and_then(crate::crop_overlay::OverlayShow::from_id)
                            .ok_or_else(|| "cropOverlayShow must be one of auto, always, never".to_string())?,
                    ),
                    None => None,
                };
                let crop_overlay_orientation = match uint_field(p, "cropOverlayOrientation")? {
                    Some(o) if o < 4 => Some(o as u8),
                    Some(_) => return Err("cropOverlayOrientation must be 0, 1, 2 or 3".into()),
                    None => None,
                };
                // The Crop tool's gear menu: Show Cropped Area and the shield (#1919).
                let crop_shield = merged_object(&app.ui.tool_options.crop_shield, p.get("cropShield"), "cropShield")?;
                if crop_shield.as_ref().is_some_and(|s| !(0.0..=100.0).contains(&s.opacity)) {
                    return Err("cropShield.opacity must be 0..100".into());
                }
                let panels = merged_object(&app.ui.panels, p.get("panels"), "panels")?;
                let mask_target = bool_field(p, "maskTarget")?;
                let vector_mask_target = bool_field(p, "vectorMaskTarget")?;
                let selection_mode = uint_field(p, "selectionMode")?;
                let dock_tabs = merged_object(&app.ui.dock_tabs, p.get("dockTabs"), "dockTabs")?;
                let dock = whole_object(&app.ui.dock, p.get("dock"), "dock")?;
                // Which chip the Color panel edits.
                let color_panel = whole_object(&app.ui.color_panel, p.get("colorPanel"), "colorPanel")?;
                let dock_width = num_field(p, "dockWidth")?;
                let zoom = num_field(p, "zoom")?;
                let rotation = num_field(p, "rotation")?;
                if rotation.is_some() && app.session.active_index().is_none() {
                    return Err("no document open".into());
                }
                let center = match p.get("center") {
                    Some(v) => {
                        let a = v.as_array().ok_or_else(|| "center must be [x, y]".to_string())?;
                        if a.len() != 2 {
                            return Err("center must be exactly [x, y]".into());
                        }
                        let (Some(x), Some(y)) = (a.first().and_then(Value::as_f64), a.get(1).and_then(Value::as_f64)) else {
                            return Err("center must be [x, y] of two numbers".into());
                        };
                        Some([x as f32, y as f32])
                    }
                    None => None,
                };
                let fit = bool_field(p, "fit")? == Some(true);
                let theme = match s("theme") {
                    Some(name) => {
                        let kind = crate::theme::ThemeKind::from_name(name).ok_or_else(|| {
                            let names: Vec<_> = crate::theme::ThemeKind::ALL.iter().map(|k| k.id()).collect();
                            format!("unknown theme `{name}` ({})", names.join(", "))
                        })?;
                        Some(kind)
                    }
                    None => None,
                };
                let brush_section = uint_field(p, "brushSection")?;
                let brush_tab = uint_field(p, "brushTab")?;
                let brushes_view = match p.get("brushesView") {
                    Some(v) => serde_json::from_value(v.clone()).map(Some).map_err(|e| format!("brushesView: {e} (list, grid)"))?,
                    None => None,
                };
                // The three parts of the picker's cards. At least one stays on (a rejected call
                // leaves them all alone).
                let brush_picker_name = bool_field(p, "brushPickerName")?;
                let brush_picker_stroke = bool_field(p, "brushPickerStroke")?;
                let brush_picker_tip = bool_field(p, "brushPickerTip")?;
                if [brush_picker_name, brush_picker_stroke, brush_picker_tip].iter().any(Option::is_some) {
                    let on = [
                        brush_picker_name.unwrap_or(app.ui.brush_picker_list.show_name),
                        brush_picker_stroke.unwrap_or(app.ui.brush_picker_list.show_stroke),
                        brush_picker_tip.unwrap_or(app.ui.brush_picker_list.show_tip),
                    ];
                    if !on.iter().any(|b| *b) {
                        return Err("brushPickerName, brushPickerStroke and brushPickerTip: at least one must stay on".into());
                    }
                }
                // The picker footer slider: the preset cards' size scale (1 standard; 0.30 and
                // below the tips drop their size numbers).
                let brush_picker_scale = match num_field(p, "brushPickerScale")? {
                    Some(s) if !(0.15..=2.0).contains(&s) => return Err("brushPickerScale must be between 0.15 and 2".into()),
                    Some(s) => Some(s as f32),
                    None => None,
                };
                // `Some(None)` is an explicit null, which closes the picker (a missing field
                // leaves it alone).
                let brush_picker = match p.get("brushPicker") {
                    Some(v) => {
                        let at = serde_json::from_value::<Option<[f32; 2]>>(v.clone())
                            .map_err(|_| "brushPicker must be [x, y] in screen points, or null to close it".to_string())?;
                        if at.is_some_and(|[x, y]| !x.is_finite() || !y.is_finite()) {
                            return Err("brushPicker must be [x, y] in screen points, or null to close it".into());
                        }
                        Some(at)
                    }
                    None => None,
                };
                // brushSize rides the journaled `tools.setBrush` command (#744). Numeric values
                // are validated here so the command cannot fail after other fields were
                // applied; non-numeric values keep the API's historical silent no-op.
                let brush_size: Option<f64> = match p.get("brushSize").and_then(Value::as_f64) {
                    Some(size) => {
                        if !size.is_finite() || size > f64::from(photocraft_paint::MAX_BRUSH_SIZE) {
                            return Err(format!("brushSize must be finite and at most {} px", photocraft_paint::MAX_BRUSH_SIZE));
                        }
                        Some(size)
                    }
                    None => None,
                };

                let bridge_pro = matches!(theme.unwrap_or(app.ui.theme), crate::theme::ThemeKind::Pro | crate::theme::ThemeKind::ProMedium);
                let legacy_update = crate::panel_docking::prepare_legacy_update(app, p, panels.as_ref(), dock_tabs.as_ref(), dock.as_ref(), bridge_pro)?;

                // Apply (nothing below can fail).
                if let Some(stroke) = shape_stroke {
                    if let Some(stroke) = stroke {
                        app.ui.tool_options.stroke_width = stroke.width;
                        app.ui.tool_options.shape_stroke = crate::shape_stroke_ui::StrokeOptions::from(&stroke);
                    } else {
                        app.ui.tool_options.stroke_width = 0.0;
                    }
                }
                if let Some(t) = tool {
                    app.ui.tool = t;
                    // Each tool keeps its own brush (#218), so switch it in before `brushSize`
                    // below sets the new tool's size.
                    crate::paint_mouse::sync_tool_brush(app);
                }
                let gradient_before = app.ui.tool_options.clone();
                if let Some(mode) = gradient_blend {
                    app.ui.tool_options.gradient_blend_mode = mode;
                }
                if let Some(classic) = gradient_classic {
                    app.ui.tool_options.gradient_classic = classic;
                }
                if gradient_blend.is_some() {
                    crate::gradient_ui::options_changed(app, &gradient_before);
                }
                if let Some(size) = eyedropper_size {
                    app.ui.tool_options.eyedropper_size = size;
                }
                if let Some(id) = eyedropper_sample {
                    app.ui.tool_options.eyedropper_sample = id;
                }
                if let Some(ring) = eyedropper_ring {
                    app.ui.tool_options.eyedropper_ring = ring;
                }
                if let Some(on) = cam_transform_on_drop {
                    app.ui.tool_options.cam_transform_on_drop = on;
                }
                if let Some(on) = tile_grid {
                    app.ui.view.tile_grid = on;
                }
                if let Some(k) = crop_overlay {
                    app.ui.tool_options.crop_overlay = k;
                }
                if let Some(v) = crop_overlay_show {
                    app.ui.tool_options.crop_overlay_show = v;
                }
                if let Some(o) = crop_overlay_orientation {
                    app.ui.tool_options.crop_overlay_orientation = o;
                }
                if let Some(s) = crop_shield {
                    app.ui.tool_options.crop_shield = s;
                }
                if let Some(v) = panels {
                    app.ui.panels = v;
                }
                if let Some(m) = mask_target {
                    app.ui.mask_target = m;
                    app.ui.properties_layer = None;
                    app.ui.vector_mask_target &= !m;
                    app.sync_mask_targets();
                }
                if let Some(m) = vector_mask_target {
                    app.ui.vector_mask_target = m;
                    app.ui.mask_target &= !m;
                    app.sync_mask_targets();
                }
                // Selection tools' options-bar mode: 0 New, 1 Add, 2 Subtract, 3 Intersect.
                if let Some(m) = selection_mode {
                    app.ui.selection_mode = m.min(3) as u8;
                }
                if let Some(k) = theme {
                    app.set_theme(ctx, k);
                }
                if let Some(v) = dock_tabs {
                    app.ui.dock_tabs = v;
                }
                // Dock group order, heights and collapsed groups (see `dock::DockLayout`).
                if let Some(v) = dock {
                    app.ui.dock = v;
                }
                if let Some(update) = legacy_update {
                    crate::panel_docking::commit_legacy_update(app, update, p);
                }
                if let Some(v) = color_panel {
                    app.ui.color_panel = v;
                }
                // Right dock width in points (clamped to the dock's 250..=520 range), applied next frame.
                if let Some(w) = dock_width {
                    crate::panels::request_dock_width(ctx, w as f32);
                }
                if let Some(i) = app.session.active_index() {
                    if let Some(z) = zoom {
                        let size = app.session.documents().get(i).map_or([0, 0], |st| [st.doc.size.width, st.doc.size.height]);
                        app.ui.views[i].zoom = crate::zoom_levels::clamp(z as f32, size);
                        app.ui.views[i].fit_pending = false;
                        app.ui.views[i].fill_pending = false;
                    }
                    if let Some(c) = center {
                        app.ui.views[i].center = c;
                        app.ui.views[i].fit_pending = false;
                        app.ui.views[i].fill_pending = false;
                    }
                    if fit {
                        app.ui.views[i].fit_pending = true;
                        app.ui.views[i].fill_pending = false;
                    }
                    if let Some(r) = rotation {
                        app.ui.views[i].rotation = crate::rotate_view::wrap_deg(r as f32);
                    }
                }
                if let Some(i) = brush_section {
                    app.ui.brush_section = (i as usize).min(crate::brush_panel::SECTIONS.len() - 1);
                }
                if let Some(i) = brush_tab {
                    app.ui.brush_tab = (i as usize).min(1);
                }
                if let Some(v) = brushes_view {
                    app.ui.brushes_panel.view = v;
                }
                if let Some(v) = brush_picker_name {
                    app.ui.brush_picker_list.show_name = v;
                }
                if let Some(v) = brush_picker_stroke {
                    app.ui.brush_picker_list.show_stroke = v;
                }
                if let Some(v) = brush_picker_tip {
                    app.ui.brush_picker_list.show_tip = v;
                }
                if let Some(v) = brush_picker_scale {
                    app.ui.brush_picker_list.scale = v;
                }
                if let Some(at) = brush_picker {
                    app.ui.brush_picker = at;
                }
                // Which chip the Color panel edits.
                if let Some(c) = color_panel {
                    app.ui.color_panel = c;
                }
                if let Some(size) = brush_size {
                    app.run("tools.setBrush", json!({"brush": {"size": size}})).map_err(|e| e.to_string())?;
                }
                Ok(Value::Null)
            })();
            wrap(applied)
        }
        "ui.dialog.open" => {
            let kind = match s("kind").unwrap_or("") {
                "newDocument" | "NewDocument" => DialogKind::NewDocument,
                "about" | "About" => DialogKind::About,
                "layerStyle" | "LayerStyle" => {
                    return match crate::layer_style::open(app, s("effect")) {
                        Some(id) => ok(json!({"dialog": id})),
                        None => err("no active layer"),
                    };
                }
                "colorPicker" | "ColorPicker" => {
                    if let Some(style) = p.get("textStyle") {
                        if ["parent", "effect", "field", "target"].iter().any(|key| p.get(*key).is_some()) {
                            return err("textStyle cannot be combined with another picker target");
                        }
                        let Some(style) = style.as_object().filter(|s| s.keys().all(|k| k == "kind" || k == "id")) else {
                            return err("textStyle needs kind and id only");
                        };
                        let Some(kind) = style.get("kind").cloned().and_then(|v| serde_json::from_value::<crate::type_panels_ui::color_picker::Kind>(v).ok())
                        else {
                            return err("textStyle kind must be character or paragraph");
                        };
                        let Some(id) = style.get("id").and_then(Value::as_u64).and_then(|id| u32::try_from(id).ok()) else {
                            return err("textStyle id must be a u32 integer");
                        };
                        return match crate::type_panels_ui::color_picker::open(app, kind, id) {
                            Ok(id) => ok(json!({"dialog": id})),
                            Err(error) => err(error),
                        };
                    }
                    if p.get("parent").is_some() {
                        let (Some(parent), Some(effect), Some(field)) = (u("parent"), s("effect"), s("field")) else {
                            return err("need integer `parent` and string `effect` and `field`");
                        };
                        return match crate::layer_style::color_picker::open(app, parent, effect, field) {
                            Ok(id) => ok(json!({"dialog": id})),
                            Err(error) => err(error),
                        };
                    }
                    if p.get("effect").is_some() || p.get("field").is_some() {
                        return err("Layer Style color pickers need `parent`");
                    }
                    let target = if s("target") == Some("background") { "background" } else { "foreground" };
                    return ok(json!({"dialog": crate::color_picker_ui::open(app, target)}));
                }
                // The Crop tool's New / Delete Crop Preset… (#1919).
                "newCropPreset" => {
                    return match crate::crop_presets::open_new(app) {
                        Some(id) => ok(json!({"dialog": id})),
                        None => err("nothing to save: set W and H (W x H x Resolution) or a ratio first"),
                    };
                }
                "deleteCropPreset" => {
                    return match crate::crop_presets::open_delete(app) {
                        Some(id) => ok(json!({"dialog": id})),
                        None => err("no crop presets are saved"),
                    };
                }
                "command" | "Command" => {
                    let Some(cmd) = s("command") else { return err("command dialogs need `command`") };
                    let label = photocraft_engine::commands::find(cmd).map(|c| c.label).unwrap_or(cmd);
                    return ok(json!({"dialog": crate::dialogs::open_command_dialog(app, cmd, label)}));
                }
                other => return err(format!("unknown dialog kind `{other}`")),
            };
            let mut fields = if kind == DialogKind::NewDocument { app.new_document_fields() } else { Default::default() };
            if let Some(f) = p.get("fields").and_then(Value::as_object) {
                fields.extend(f.clone());
            }
            ok(json!({"dialog": app.ui.open_dialog(kind, fields)}))
        }
        // The shell's own dialog (Window › Workspace, View › Pixel Aspect Ratio › Custom, Show
        // Extras Options, 32-bit Preview Options), listed by ui.inspect with the id "shell".
        "ui.dialog.set" if s("dialog") == Some(SHELL_DIALOG) => {
            let Some(field) = s("field") else { return err("need `dialog` and `field`") };
            let value = p.get("value").cloned().unwrap_or(Value::Null);
            match app.ui.shell.dialog.as_mut() {
                Some((_, fields)) => {
                    fields.insert(field.to_string(), value);
                    ok(Value::Null)
                }
                None => err("no shell dialog is open"),
            }
        }
        "ui.dialog.confirm" if s("dialog") == Some(SHELL_DIALOG) => {
            let Some((kind, fields)) = app.ui.shell.dialog.clone() else { return err("no shell dialog is open") };
            if let Some((command, params)) = crate::workspace_ui::dialog_command(&kind, &fields)
                && let Some(authorize) = app.services.automation_command.as_ref()
                && let Err(error) = authorize(command, &params)
            {
                return err(error);
            }
            wrap(crate::workspace_ui::confirm(app, ctx))
        }
        "ui.dialog.apply" if s("dialog") == Some(SHELL_DIALOG) => err("`ui.dialog.apply` is for Preferences; use `ui.dialog.confirm`"),
        // The unsaved-changes prompt (closing, Revert, quitting), listed by ui.inspect with the
        // id "discard" (#4920): confirm takes a `choice` (save, dontSave, cancel; default save).
        "ui.dialog.confirm" if s("dialog") == Some(crate::discard_ui::CONTROL_ID) => {
            let choice = s("choice").unwrap_or("save");
            if choice != "cancel"
                && let Some((command, params)) = crate::discard_ui::pending_command(app)
                && let Some(authorize) = app.services.automation_command.as_ref()
                && let Err(error) = authorize(&command, &params)
            {
                return err(error);
            }
            wrap(crate::discard_ui::answer_control(app, ctx, choice))
        }
        "ui.dialog.cancel" if s("dialog") == Some(crate::discard_ui::CONTROL_ID) => wrap(crate::discard_ui::answer_control(app, ctx, "cancel")),
        "ui.dialog.set" | "ui.dialog.apply" if s("dialog") == Some(crate::discard_ui::CONTROL_ID) => {
            err("the unsaved-changes prompt has no fields; use `ui.dialog.confirm` with a `choice`")
        }
        "ui.dialog.cancel" if s("dialog") == Some(SHELL_DIALOG) => match app.ui.shell.dialog.take() {
            Some(_) => ok(Value::Null),
            None => err("no such dialog"),
        },
        "ui.dialog.set" => {
            let (Some(id), Some(field)) = (u("dialog"), s("field")) else { return err("need `dialog` and `field`") };
            let value = p.get("value").cloned().unwrap_or(Value::Null);
            match app.ui.dialog_mut(id) {
                Some(d) => {
                    d.fields.insert(field.to_string(), value);
                    ok(Value::Null)
                }
                None => err(format!("no dialog {id}")),
            }
        }
        "ui.dialog.confirm" | "ui.dialog.apply" => match u("dialog") {
            Some(id) => {
                let command = app.ui.dialogs.iter().find(|dialog| dialog.id == id).and_then(|dialog| {
                    if req.method == "ui.dialog.apply" {
                        dialog.fields.get("values").map(|values| ("prefs.set".to_string(), json!({"path": "", "value": values})))
                    } else {
                        dialog.fields.get("__command").and_then(Value::as_str).map(|command| (command.to_string(), Value::Object(dialog.fields.clone())))
                    }
                });
                if let Some((command, params)) = command
                    && let Some(authorize) = app.services.automation_command.as_ref()
                    && let Err(error) = authorize(&command, &params)
                {
                    return err(error);
                }
                let apply = req.method == "ui.dialog.apply";
                run_waiting(app, wait, |app| if apply { crate::prefs_ui::apply(app, id) } else { crate::dialogs::confirm(app, id) })
            }
            None => err("missing `dialog`"),
        },
        "ui.dialog.cancel" => match u("dialog") {
            Some(id) => match crate::dialogs::cancel(app, id) {
                Ok(value) => ok(value),
                Err(error) => err(error),
            },
            None => err("missing `dialog`"),
        },
        "ui.window.open" => {
            if let Some(d) = u("document")
                && !app.session.set_active(d as usize)
            {
                return err(format!("no document {d}"));
            }
            wrap(crate::menus::invoke(app, ctx, "window.newWindowForDocument", json!({})))
        }
        "ui.window.close" => {
            let Some(id) = u("window") else { return err("missing `window`") };
            let before = app.ui.windows.len();
            app.ui.windows.retain(|w| w.id != id);
            if app.ui.windows.len() < before { ok(Value::Null) } else { err(format!("no window {id}")) }
        }
        "ui.pointer" => {
            // Every field and event is validated before the first event runs, so a rejected call
            // changes nothing (#2449).
            let parsed = (|| -> std::result::Result<_, String> {
                let fields = ["events", "tool", "modifiers", "button", "shift", "alt", "command", "ctrl", "space"];
                let p = input_object(p, None, &fields)?;
                let tool = match input_choice(&p, "tool", "tool", &[])? {
                    Some(t) => Some(Tool::from_name(t).ok_or_else(|| format!("unknown tool `{t}`"))?),
                    None => None,
                };
                let button = input_choice(&p, "button", "button", &BUTTONS)?.unwrap_or("left").to_string();
                // Modifier flags may be top-level or grouped under "modifiers".
                let m = input_modifiers(&p, &POINTER_MODIFIERS)?;
                let Some(events) = p.get("events") else { return Err("missing `events`".into()) };
                let Some(events) = events.as_array() else { return Err("events must be an array".into()) };
                let mut parsed = Vec::with_capacity(events.len());
                for (i, e) in events.iter().enumerate() {
                    let name = format!("events[{i}]");
                    let e = input_object(e, Some(&name), &["kind", "x", "y", "pressure", "tiltX", "tiltY", "rotation"])?;
                    let num = |k: &str, range| input_number(&e, k, &format!("{name}.{k}"), range);
                    let (x, y) = (num("x", None)?.unwrap_or(0.0), num("y", None)?.unwrap_or(0.0));
                    let pr = num("pressure", Some((0.0, 1.0)))?.unwrap_or(1.0) as f32;
                    let ev = match input_choice(&e, "kind", &format!("{name}.kind"), &["down", "move", "up"])?.unwrap_or("move") {
                        "down" => ToolEvent::Down { x, y, pressure: pr },
                        "up" => ToolEvent::Up { x, y },
                        _ => ToolEvent::Move { x, y, pressure: pr },
                    };
                    let tilt = |k: &str, range| num(k, Some(range)).map(|v| v.map(|v| v as f32));
                    let pen = (tilt("tiltX", (-90.0, 90.0))?, tilt("tiltY", (-90.0, 90.0))?, tilt("rotation", (0.0, 360.0))?);
                    parsed.push((ev, x, y, pr, pen));
                }
                Ok((tool, button, m, parsed))
            })();
            let (tool, button, m, events) = match parsed {
                Ok(parsed) => parsed,
                Err(e) => return err(e),
            };
            let secondary = matches!(button.as_str(), "secondary" | "right");
            let mods = modifiers_from(&m);
            if let Some(t) = tool {
                app.ui.tool = t;
            }
            // Space held: the Crop tool moves the frame being drawn, a marquee, lasso or shape
            // being drawn moves instead of growing (hold_keys.rs). Like a modifier flag, it may be
            // top-level or grouped under "modifiers".
            let space = p.get("modifiers").unwrap_or(p).get("space").and_then(Value::as_bool).unwrap_or(false);
            crate::crop_ui::set_space(app, space);
            for (ev, x, y, pr, pen) in events {
                // With the Color Picker on top the image is its eyedropper, as for the mouse.
                if crate::color_picker_ui::top(app).is_some() {
                    if !matches!(ev, ToolEvent::Up { .. }) {
                        crate::color_picker_ui::sample_at(app, x, y);
                    }
                    continue;
                }
                // Curves' modal eyedroppers use document coordinates here, matching the native
                // canvas path after its ViewXform conversion. Sample on press only.
                if !secondary && crate::adjust_dialog::picker_armed(app) {
                    if matches!(ev, ToolEvent::Down { .. }) {
                        crate::adjust_dialog::sample_at(app, x, y);
                    }
                    continue;
                }
                // A Properties Levels/Curves eyedropper owns the primary button the same way.
                if app.ui.dialogs.is_empty() && !matches!(s("button"), Some("secondary" | "right")) && crate::layer_picker::armed(app).is_some() {
                    if matches!(ev, ToolEvent::Down { .. }) {
                        crate::layer_picker::sample_at(app, x, y);
                    }
                    continue;
                }
                // The dialog owns all pointer events, including Up: no underlying brush stroke
                // or context menu may leak through while inspecting a Color Range preview.
                if app.ui.dialogs.last().is_some_and(|d| crate::color_range_ui::owns(&d.fields)) {
                    let down = matches!(ev, ToolEvent::Down { .. });
                    let up = matches!(ev, ToolEvent::Up { .. });
                    let held = app.ui.dialogs.last().and_then(|d| d.fields.get("__pointerDown")).and_then(Value::as_bool).unwrap_or(false);
                    if let Some(d) = app.ui.dialogs.last_mut() {
                        d.fields.insert("__pointerDown".into(), json!(!up && (down || held)));
                    }
                    if !up && (down || held) && !space && !secondary && button != "middle" {
                        crate::color_range_ui::pick_top(app, [x, y], mods);
                    }
                    continue;
                }
                if secondary {
                    let down = matches!(ev, ToolEvent::Down { .. });
                    // Right-click while transforming: the box's menu, as on the canvas.
                    if app.ui.transform.as_ref().is_some_and(|t| t.warp.is_none()) {
                        if down {
                            crate::canvas_tool_menu::open_transform(app, ctx, screen_point(app, x, y));
                        }
                        continue;
                    }
                    // Right-click with the Move tool, or ⌘/Ctrl+right-click: list the layers there.
                    if crate::layer_pick_ui::is_gesture(app.ui.tool, mods) {
                        if down {
                            app.ui.canvas_tool_menu = None;
                            crate::layer_pick_ui::open(app, ctx, screen_point(app, x, y), x, y);
                        }
                        continue;
                    }
                    if crate::canvas_tool_menu::pasteboard_at(app, app.ui.tool, [x, y]) {
                        if down {
                            crate::canvas_tool_menu::open_pasteboard(app, ctx, app.ui.tool, screen_point(app, x, y));
                        }
                        continue;
                    }
                    if crate::canvas_tool_menu::is_view_tool(app.ui.tool) {
                        if down {
                            crate::canvas_tool_menu::open_view(app, ctx, app.ui.tool, screen_point(app, x, y), [x, y]);
                        }
                        continue;
                    }
                    if crate::canvas_tool_menu::applies(app.ui.tool) || crate::canvas_tool_menu::editing_text(app, app.ui.tool) {
                        if down {
                            crate::canvas_tool_menu::open_at(app, ctx, app.ui.tool, screen_point(app, x, y), Some([x, y]));
                        }
                        continue;
                    }
                    if !crate::paint_mouse::pointer_secondary(app, ctx, down, mods, screen_point(app, x, y)) {
                        continue;
                    }
                }
                // A simulated pen: tilt/rotation reach the stroke like a real stylus's (see `stylus`).
                if pen != (None, None, None) {
                    let (tilt_x, tilt_y, rotation) = (pen.0.unwrap_or(0.0), pen.1.unwrap_or(0.0), pen.2.unwrap_or(0.0));
                    app.stylus.feed.set(Some(crate::stylus::PenSample { pressure: pr, tilt_x, tilt_y, rotation, eraser: false }));
                }
                if let Some(d) = app.drag.as_mut().filter(|d| crate::hold_keys::repositions(d.tool)) {
                    d.reposition = space;
                }
                // A Crop drag in the default mode turns and pans the view with the image: its
                // points are read as the view showed them at the press (`crop_mode`).
                let ev = match ev {
                    ToolEvent::Move { x, y, pressure } => {
                        let [x, y] = crate::crop_mode::control_point(app, [x, y]);
                        ToolEvent::Move { x, y, pressure }
                    }
                    ToolEvent::Up { x, y } => {
                        let [x, y] = crate::crop_mode::control_point(app, [x, y]);
                        ToolEvent::Up { x, y }
                    }
                    down => down,
                };
                tool_event(app, ev, mods);
            }
            app.stylus.feed.set(None);
            ok(json!({"status": app.ui.status}))
        }
        "ui.click" | "ui.move" => {
            let click = req.method == "ui.click";
            let fields: &[&str] = if click { &["x", "y", "button", "count"] } else { &["x", "y"] };
            let parsed = (|| -> std::result::Result<_, String> {
                let p = input_object(p, None, fields)?;
                // Screen coordinates in points (as reported by ui.inspect window size).
                let x = input_number(&p, "x", "x", None)?.unwrap_or(0.0) as f32;
                let y = input_number(&p, "y", "y", None)?.unwrap_or(0.0) as f32;
                let button = match input_choice(&p, "button", "button", &BUTTONS)?.unwrap_or("left") {
                    "right" | "secondary" => egui::PointerButton::Secondary,
                    "middle" => egui::PointerButton::Middle,
                    _ => egui::PointerButton::Primary,
                };
                let clicks = if click { uint_field(&p, "count")?.unwrap_or(1) } else { 0 };
                if clicks > MAX_CLICKS {
                    return Err(format!("`count` must be at most {MAX_CLICKS} (got {clicks})"));
                }
                Ok((egui::pos2(x, y), button, clicks))
            })();
            let (pos, button, clicks) = match parsed {
                Ok(parsed) => parsed,
                Err(e) => return err(e),
            };
            app.synthetic.push(egui::Event::PointerMoved(pos));
            for _ in 0..clicks {
                app.synthetic.push(egui::Event::PointerButton { pos, button, pressed: true, modifiers: Default::default() });
                app.synthetic.push(egui::Event::PointerButton { pos, button, pressed: false, modifiers: Default::default() });
            }
            ctx.request_repaint();
            Outcome::AfterInput
        }
        "ui.key" => {
            let parsed = (|| -> std::result::Result<_, String> {
                let p = input_object(p, None, &["key", "modifiers", "shift", "alt", "command", "ctrl"])?;
                let Some(name) = input_choice(&p, "key", "key", &[])? else { return Err("missing `key`".into()) };
                let key = egui::Key::from_name(name).ok_or_else(|| format!("unknown key `{name}`"))?;
                // Modifier flags may be top-level or grouped under "modifiers".
                Ok((key, input_modifiers(&p, &KEY_MODIFIERS)?))
            })();
            let (key, m) = match parsed {
                Ok(parsed) => parsed,
                Err(e) => return err(e),
            };
            let modifiers = modifiers_from(&m);
            app.synthetic.push(egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers });
            app.synthetic.push(egui::Event::Key { key, physical_key: None, pressed: false, repeat: false, modifiers });
            ctx.request_repaint();
            Outcome::AfterInput
        }
        "ui.type" => {
            let Some(text) = s("text") else { return err("missing `text`") };
            app.synthetic.push(egui::Event::Text(text.to_string()));
            ctx.request_repaint();
            Outcome::AfterInput
        }
        "ui.resize" => {
            let (w, h) = (p.get("width").and_then(Value::as_f64).unwrap_or(1280.0), p.get("height").and_then(Value::as_f64).unwrap_or(800.0));
            ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(w as f32, h as f32)));
            ok(Value::Null)
        }
        "ui.gpu.simulateLoss" => {
            let error = p.get("error").and_then(Value::as_bool).unwrap_or(false);
            let fault = if error {
                photocraft_gpu::Fault::Error("simulated error (ui.gpu.simulateLoss)".into())
            } else {
                photocraft_gpu::Fault::Lost("simulated (ui.gpu.simulateLoss)".into())
            };
            let active = match app.gpu_health() {
                Some(h) => {
                    h.mark(fault);
                    true
                }
                None => false,
            };
            app.check_gpu(ctx);
            ctx.request_repaint();
            ok(json!({"wasActive": active, "gpuInfo": app.perf.gpu_info}))
        }
        "ui.focus" => {
            ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
            ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(false));
            ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
            ctx.request_repaint();
            ok(Value::Null)
        }
        "ui.screenshot" => {
            let parsed = input_object(p, None, &["path", "focus"])
                .and_then(|p| Ok((input_choice(&p, "path", "path", &[])?.map(str::to_string), bool_field(&p, "focus")?)));
            let (path, focus) = match parsed {
                Ok(parsed) => parsed,
                Err(e) => return err(e),
            };
            // Occluded windows don't render on macOS, so the screenshot would never arrive: raise first.
            if focus.unwrap_or(true) {
                ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
            }
            // The capture itself is issued by `PhotocraftApp::issue_screenshots` once open/close
            // animations (modals, popups) have settled.
            let token = app.ui.alloc_id();
            ctx.request_repaint();
            Outcome::Screenshot { token, path }
        }
        "app.open" => match s("path") {
            Some(path) => {
                // Directory .pcraft (the native on-disk form) loads through a capability-scoped
                // source; the single-file (zip) form falls back to the bytes path, as do all
                // other formats.
                if path.rsplit_once('.').is_some_and(|(_, ext)| ext.eq_ignore_ascii_case(photocraft_format::EXTENSION))
                    && let Some(open) = app.services.automation_open_pcraft.as_mut()
                {
                    match open(path) {
                        Ok(Some((name, doc))) => {
                            app.session.add_document(doc, None);
                            // Recent files, but no save-back target: the capability writer
                            // replaces files, never directories.
                            app.push_recent(path);
                            app.sync_views();
                            app.ui.status = crate::i18n::fmt(tl!("Opened {name}"), &[("name", &name)]);
                            app.ui.status_error = false;
                            return ok(json!({"path": path, "name": name, "warnings": []}));
                        }
                        Ok(None) => {} // a regular file: the bytes path reads the zip form
                        Err(e) => return err(e),
                    }
                }
                let opened = match app.services.automation_read.as_mut() {
                    Some(read) => read(path),
                    None => Err("automation read authority is not configured".into()),
                };
                wrap(opened.and_then(|(name, bytes)| {
                    let name = app.open_name(&name);
                    let warnings = app.open_automation_bytes(&name, &bytes)?;
                    // Brushes/gradients go to the preset libraries: no document, no Open Recent entry.
                    if !crate::preset_files_ui::is_preset_file(&name) {
                        app.opened_from(path);
                    }
                    Ok(json!({"path": path, "name": name, "warnings": warnings}))
                }))
            }
            None => err("missing `path`"),
        },
        "app.save" => match crate::targa_options_ui::bits_param(p, "app.save") {
            Ok(tga_bits) => {
                let path = s("path").map(str::to_string);
                run_waiting(app, p.get("wait").and_then(Value::as_bool).unwrap_or(true), move |app| app.save_automation_with(path, tga_bits))
            }
            Err(e) => err(e),
        },
        "app.quit" => {
            app.allow_close = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            ok(Value::Null)
        }
        other => err(format!("unknown method `{other}`")),
    }
}

/// Snapshot of everything on screen, addressable by id.
pub fn inspect(app: &PhotocraftApp, ctx: &egui::Context) -> Value {
    let screen = ctx.content_rect();
    let mut dialogs: Vec<Value> =
        app.ui.dialogs.iter().map(|d| json!({"id": d.id, "kind": d.kind, "title": crate::dialogs::title(d), "fields": d.fields})).collect();
    if let Some((kind, fields)) = &app.ui.shell.dialog {
        dialogs.push(json!({"id": SHELL_DIALOG, "kind": kind, "title": crate::workspace_ui::title(kind), "fields": fields}));
    }
    dialogs.extend(crate::discard_ui::inspect(app));
    json!({
        "window": {"width": screen.width(), "height": screen.height(), "pixelsPerPoint": ctx.pixels_per_point()},
        "tool": app.ui.tool,
        "toolOptions": app.ui.tool_options,
        "perspectiveCrop": app.ui.perspective_crop,
        "strokeEditor": app.ui.stroke_editor,
        "magnetic": app.ui.magnetic,
        "textEdit": app.ui.text_edit,
        "typeTransform": app.ui.type_transform,
        "camTransform": app.ui.cam_transform,
        "layerMenu": app.ui.layer_menu,
        "layerPicker": app.ui.layer_picker,
        "brushPicker": app.ui.brush_picker.map(|pos| json!({
            "pos": pos,
            // The picker's content size once its corner grip was dragged (null: the default).
            "size": app.ui.brush_picker_size,
            "list": {
                "collapsed": app.ui.brush_picker_list.collapsed,
                "filter": app.ui.brush_picker_list.filter,
                "renaming": app.ui.brush_picker_list.renaming,
                // The card's parts (`view` is the Brushes panel's listing and doesn't apply).
                "showName": app.ui.brush_picker_list.show_name,
                "showStroke": app.ui.brush_picker_list.show_stroke,
                "showTip": app.ui.brush_picker_list.show_tip,
                // The footer slider: the cards' width scale.
                "scale": app.ui.brush_picker_list.scale,
            },
        })),
        "canvasToolMenu": app.ui.canvas_tool_menu.as_ref().map(|menu| {
            json!({
                "pos": menu.pos,
                "tool": menu.tool,
                "pasteboard": menu.pasteboard,
                "entries": context_entries(app, menu)
            })
        }),
        "panels": app.ui.panels,
        "dockTabs": app.ui.dock_tabs,
        "dock": app.ui.dock,
        "docking": app.ui.docking,
        "dockingHidden": app.ui.docking_hidden,
        "dockingCollapsed": app.ui.docking_collapsed,
        "view": app.ui.view,
        "views": app.ui.views,
        "dialogs": dialogs,
        "windows": app.ui.windows,
        "theme": app.ui.theme,
        "status": app.ui.status,
        "statusError": app.ui.status_error,
        "notices": app.ui.notices,
        "gpuFallbackNotice": app.ui.gpu_fallback_notice,
        "kysOffer": app.ui.kys_offer,
        "frame": app.frame,
        "session": photocraft_engine::inspect::session(&app.session),
        "document": app.session.active().map(photocraft_engine::inspect::document),
        "perf": {"fps": app.fps, "timings": app.perf, "strokeTrace": app.stroke_trace.last},
        "brush": {"size": app.session.tools.brush.size, "hardness": app.session.tools.brush.hardness, "opacity": app.session.tools.brush.opacity},
        "distort": app.distort.describe(),
        "jobs": crate::jobs_ui::inspect(app),
        "cameraRaw": app.camera_raw.as_ref().map(|d| d.describe(&app.ui.camera_raw_scope, &app.ui.camera_raw_preview)),
    })
}

pub fn save_screenshot(app: &mut PhotocraftApp, image: &egui::ColorImage, path: Option<&str>) -> Value {
    let [w, h] = image.size;
    let rgba: Vec<u8> = image.pixels.iter().flat_map(|c| c.to_array()).collect();
    let png = match app.services.encode_png.as_ref() {
        Some(enc) => enc(w as u32, h as u32, &rgba),
        None => Err("no PNG encoder configured".into()),
    };
    let Some(path) = path else {
        return match png {
            Ok(bytes) => json!({
                "ok": true,
                "result": {
                    "base64": base64::engine::general_purpose::STANDARD.encode(bytes),
                    "mime": "image/png",
                    "width": w,
                    "height": h,
                }
            }),
            Err(error) => json!({"ok": false, "error": error}),
        };
    };
    let result = png.and_then(|bytes| match app.services.automation_write.as_mut() {
        Some(write) => write(path, &bytes),
        None => Err("automation write authority is not configured".into()),
    });
    match result {
        Ok(()) => json!({"ok": true, "result": {"path": path, "width": w, "height": h}}),
        Err(e) => json!({"ok": false, "error": e}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(app: &mut PhotocraftApp, ctx: &egui::Context, method: &str, params: Value) -> Value {
        let (req, _rx) = ControlRequest::new(method, params);
        match handle(app, ctx, &req) {
            Outcome::Done(v) => v,
            _ => panic!("{method}: expected an immediate reply"),
        }
    }

    #[test]
    fn camera_raw_control_updates_and_commits_or_cancels_an_open_raw() {
        use photocraft_raw::testgen::{DngSpec, mosaic, scene};
        let (w, h) = (48, 32);
        let mut spec = DngSpec::cfa(w, h, mosaic(&scene(w, h), w, [0, 1, 1, 2], 0, 30000));
        spec.as_shot_neutral = Some([0.5, 1.0, 0.7]);
        let dng = spec.build();
        for finish in ["commit", "cancel"] {
            let services = crate::Services {
                import: Some(Box::new(|name, bytes, depth| {
                    photocraft_io::import_with_svg_group_depth(name, bytes, depth).map(|r| (r.document, r.warnings)).map_err(|e| e.to_string())
                })),
                ..Default::default()
            };
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
            let ctx = egui::Context::default();
            PhotocraftApp::setup_context(&ctx, Default::default());
            app.open_bytes("control.dng", &dng).unwrap();
            ctx.run_ui(Default::default(), |ui| crate::camera_raw_ui::show(&mut app, ui.ctx())).textures_delta.clear();
            assert!(app.camera_raw.is_some());
            // A developed raw has no embedded profile to ask about (`open_prompt`), so no profile
            // prompt stands between the open and the Camera Raw dialog.
            assert!(app.ui.dialogs.is_empty());
            let update = json!({"id":"filter.cameraRaw","params":{"ui":{"set":{"exposure":0.5}}}});
            let original = app.session.active().unwrap().doc.clone();
            let changed = call(&mut app, &ctx, "ui.menu.invoke", update);
            assert_eq!(changed["ok"], true, "{changed}");
            assert_eq!(changed["result"]["params"]["exposure"], 0.5);
            assert!(std::sync::Arc::ptr_eq(&app.session.active().unwrap().doc, &original), "a preview must not commit");
            assert!(!crate::menus::modal_allows(&app, "filter.cameraRaw"), "native menu clicks stay blocked");
            let finished = call(&mut app, &ctx, "ui.menu.invoke", json!({"id":"filter.cameraRaw","params":{"ui":{finish:true}}}));
            assert_eq!(finished["ok"], true, "{finished}");
            assert!(app.camera_raw.is_none());
            if finish == "commit" {
                assert_eq!(finished["result"]["redeveloped"], true, "{finished}");
                assert_eq!(app.session.documents().len(), 1);
                let st = app.session.active().unwrap();
                assert_ne!(st.doc.id, original.id);
                assert_eq!(st.doc.depth, photocraft_color::SampleType::U16);
                assert_eq!(st.saved_revision, st.revision);
            } else {
                assert_eq!(finished["result"]["cancelled"], true);
                assert!(app.session.documents().is_empty());
            }
        }
    }

    #[test]
    fn camera_raw_control_continuation_does_not_bypass_other_modal_or_command_gates() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width":16,"height":16})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        app.run("edit.fill", json!({"color":"#808080"})).unwrap();
        let opened = call(&mut app, &ctx, "ui.menu.invoke", json!({"id":"filter.cameraRaw"}));
        assert_eq!(opened["ok"], true, "{opened}");
        let original = app.session.active().unwrap().doc.clone();
        let cancel = json!({"id":"filter.cameraRaw","params":{"ui":{"cancel":true}}});
        for request in [
            json!({"id":"file.new"}),
            json!({"id":"view.zoomIn"}),
            json!({"id":"filter.cameraRaw"}),
            json!({"id":"filter.cameraRaw","params":{"exposure":1}}),
            json!({"id":"filter.cameraRaw","params":{"ui":null}}),
            json!({"id":"filter.cameraRaw","params":{"ui":{},"smartFilter":{"layer":1,"index":0}}}),
        ] {
            let blocked = call(&mut app, &ctx, "ui.menu.invoke", request);
            assert_eq!(blocked["ok"], false, "{blocked}");
        }
        app.ui.open_dialog(DialogKind::About, Default::default());
        assert_eq!(call(&mut app, &ctx, "ui.menu.invoke", cancel.clone())["ok"], false);
        app.ui.dialogs.clear();
        assert!(crate::discard_ui::intercept(&mut app, "file.close", &json!({})));
        assert_eq!(call(&mut app, &ctx, "ui.menu.invoke", cancel.clone())["ok"], false);
        app.discard = None;
        app.ui.tool = Tool::Crop;
        app.ui.crop_rect = Some([1.0, 1.0, 10.0, 10.0]);
        assert!(crate::crop_ui::blocks(&app, "filter.cameraRaw"));
        assert_eq!(call(&mut app, &ctx, "ui.menu.invoke", cancel.clone())["ok"], false);
        app.ui.crop_rect = None;
        app.services.automation_command = Some(Box::new(|id, _| if id == "filter.cameraRaw" { Err("Camera Raw denied".into()) } else { Ok(()) }));
        for ui in [json!({"set":{"exposure":1}}), json!({"commit":true}), json!({"cancel":true})] {
            let denied = call(&mut app, &ctx, "ui.menu.invoke", json!({"id":"filter.cameraRaw","params":{"ui":ui}}));
            assert_eq!(denied["ok"], false, "{denied}");
            assert_eq!(denied["error"], "Camera Raw denied");
        }
        assert_eq!(app.camera_raw.as_ref().unwrap().params.exposure, 0.0);
        assert!(std::sync::Arc::ptr_eq(&app.session.active().unwrap().doc, &original));
        app.services.automation_command = None;
        assert_eq!(call(&mut app, &ctx, "ui.menu.invoke", cancel)["ok"], true);
        assert!(app.camera_raw.is_none());
    }

    #[test]
    fn layer_style_picker_control_binds_previews_confirms_and_cancels() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width":16,"height":16})).unwrap();
        let parent = call(&mut app, &ctx, "ui.dialog.open", json!({"kind":"layerStyle","effect":"gradientOverlay"}))["result"]["dialog"].as_u64();
        let parent = parent.unwrap();
        let before = app.ui.dialogs[0].fields.clone();
        for field in ["from", "to"] {
            let reply = call(&mut app, &ctx, "ui.dialog.open", json!({"kind":"colorPicker","parent":parent,"effect":"fx1","field":field}));
            assert_eq!(reply["ok"], true, "{reply}");
            let picker = app.ui.dialogs.last().unwrap().id;
            let inspected = call(&mut app, &ctx, "ui.inspect", json!({}));
            assert!(inspected.to_string().contains("__layerStyleColor"));
            assert_eq!(call(&mut app, &ctx, "ui.dialog.confirm", json!({"dialog":parent}))["ok"], false);
            assert_eq!(call(&mut app, &ctx, "ui.dialog.set", json!({"dialog":picker,"field":"color","value":"#123456"}))["ok"], true);
            assert_eq!(call(&mut app, &ctx, "ui.dialog.confirm", json!({"dialog":picker}))["ok"], true);
            assert_eq!(app.ui.dialogs[0].fields["effects"][0]["params"][field], "#123456");
        }
        assert_ne!(app.ui.dialogs[0].fields, before);
        call(&mut app, &ctx, "ui.dialog.open", json!({"kind":"colorPicker","parent":parent,"effect":"fx1","field":"to"}));
        assert_eq!(call(&mut app, &ctx, "ui.dialog.cancel", json!({"dialog":parent}))["ok"], true);
        assert!(app.ui.dialogs.is_empty());
    }

    #[test]
    fn text_style_picker_control_uses_bound_target_and_rejects_invalid_requests_atomically() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width":32,"height":32})).unwrap();
        let id = app.run("type.characterStyle.new", json!({"fromSelection":false,"attrs":{"color":"#123456"}})).unwrap()["id"].as_u64().unwrap();
        for style in [
            Value::Null,
            json!({"kind":"wrong","id":id}),
            json!({"kind":"character","id":"bad"}),
            json!({"kind":"paragraph","id":u64::MAX}),
            json!({"kind":"character","id":0}),
            json!({"kind":"character","id":id,"extra":1}),
        ] {
            let panels = app.ui.type_panels.clone();
            assert_eq!(call(&mut app, &ctx, "ui.dialog.open", json!({"kind":"colorPicker","textStyle":style}))["ok"], false);
            assert_eq!(app.ui.type_panels, panels);
            assert!(app.ui.dialogs.is_empty());
        }
        assert_eq!(call(&mut app, &ctx, "ui.dialog.open", json!({"kind":"colorPicker","textStyle":{"kind":"character","id":id},"parent":1}))["ok"], false);
        let doc = app.session.active().unwrap().doc.clone();
        let r = call(&mut app, &ctx, "ui.dialog.open", json!({"kind":"colorPicker","textStyle":{"kind":"character","id":id}}));
        assert_eq!(r["ok"], true);
        let picker = r["result"]["dialog"].as_u64().unwrap();
        assert_eq!(app.ui.dialogs[0].fields["__textStyleColor"]["id"], id);
        assert_eq!(call(&mut app, &ctx, "ui.dialog.set", json!({"dialog":picker,"field":"color","value":"#00ff00"}))["ok"], true);
        assert_eq!(call(&mut app, &ctx, "ui.dialog.cancel", json!({"dialog":picker}))["ok"], true);
        assert!(std::sync::Arc::ptr_eq(&app.session.active().unwrap().doc, &doc));
        let r = call(&mut app, &ctx, "ui.dialog.open", json!({"kind":"colorPicker","textStyle":{"kind":"character","id":id}}));
        let picker = r["result"]["dialog"].as_u64().unwrap();
        call(&mut app, &ctx, "ui.dialog.set", json!({"dialog":picker,"field":"color","value":"#00ff00"}));
        assert_eq!(call(&mut app, &ctx, "ui.dialog.confirm", json!({"dialog":picker}))["ok"], true);
        assert_ne!(app.session.active().unwrap().doc.text_styles, doc.text_styles);
    }

    #[test]
    fn layer_style_picker_control_rejects_bad_targets_without_opening_tool_picker() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width":16,"height":16})).unwrap();
        let parent = crate::layer_style::open(&mut app, Some("colorOverlay")).unwrap();
        for params in [
            json!({"kind":"colorPicker","parent":null,"effect":"fx1","field":"color"}),
            json!({"kind":"colorPicker","parent":"bad","effect":"fx1","field":"color"}),
            json!({"kind":"colorPicker","parent":parent,"effect":3,"field":"color"}),
            json!({"kind":"colorPicker","parent":parent,"effect":"missing","field":"color"}),
            json!({"kind":"colorPicker","parent":parent,"effect":"fx1","field":"opacity"}),
            json!({"kind":"colorPicker","parent":parent,"effect":"fx1","field":"to"}),
            json!({"kind":"colorPicker","effect":"fx1","field":"color"}),
        ] {
            let before = app.ui.dialogs.clone();
            assert_eq!(call(&mut app, &ctx, "ui.dialog.open", params)["ok"], false);
            assert_eq!(app.ui.dialogs, before);
        }
    }

    #[test]
    fn shape_stroke_defaults_are_drivable_and_validate_atomically() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        let result = call(
            &mut app,
            &ctx,
            "ui.set",
            json!({"tool":"Triangle", "shapeStroke":{"width":5,"cap":"round","join":"bevel","dashes":[0,2,4,1],"dashOffset":-0.5}}),
        );
        assert_eq!(result["ok"], true);
        assert_eq!(app.ui.tool_options.stroke_width, 5.0);
        assert_eq!(app.ui.tool_options.shape_stroke.dashes, [0.0, 2.0, 4.0, 1.0]);
        let before = app.ui.clone();
        let bad = call(&mut app, &ctx, "ui.set", json!({"tool":"Rectangle","shapeStroke":{"dashes":[0,0]}}));
        assert_eq!(bad["ok"], false);
        assert_eq!(app.ui, before);
        assert_eq!(call(&mut app, &ctx, "ui.set", json!({"shapeStroke":{"colour":"#ff0000"}}))["ok"], false);
        assert_eq!(call(&mut app, &ctx, "ui.set", json!({"shapeStroke":null}))["ok"], true);
        assert_eq!(app.ui.tool_options.stroke_width, 0.0);
    }

    #[test]
    fn the_unsaved_changes_prompt_is_listed_and_answered_through_ui_dialog() {
        // #4920: the prompt shown on quit or close with unsaved changes wasn't in ui.inspect, and a
        // control client couldn't answer it.
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        for _ in 0..2 {
            app.run("file.new", json!({"width": 8, "height": 8})).unwrap();
            app.run("layer.new.layer", json!({})).unwrap();
        }
        let prompt = |app: &mut PhotocraftApp| {
            let v = call(app, &ctx, "ui.inspect", json!({}));
            v["result"]["dialogs"].as_array().unwrap().iter().find(|d| d["id"] == "discard").cloned()
        };
        assert!(prompt(&mut app).is_none());
        // Closing a dirty document: listed with its choices; a bad choice and Cancel keep the document.
        crate::menus::invoke(&mut app, &ctx, "file.close", json!({})).unwrap();
        let d = prompt(&mut app).expect("the prompt is listed");
        assert_eq!(d["fields"]["command"], "file.close");
        assert_eq!(d["fields"]["choices"], json!(["save", "dontSave", "cancel"]));
        assert_eq!(call(&mut app, &ctx, "ui.dialog.confirm", json!({"dialog": "discard", "choice": "maybe"}))["ok"], false);
        assert_eq!(call(&mut app, &ctx, "ui.dialog.set", json!({"dialog": "discard", "field": "x", "value": 1}))["ok"], false);
        assert_eq!(call(&mut app, &ctx, "ui.dialog.cancel", json!({"dialog": "discard"}))["ok"], true);
        assert!(prompt(&mut app).is_none() && app.session.documents().len() == 2);
        assert_eq!(call(&mut app, &ctx, "ui.dialog.cancel", json!({"dialog": "discard"}))["ok"], false, "nothing left to answer");
        // Don't Save closes it.
        crate::menus::invoke(&mut app, &ctx, "file.close", json!({})).unwrap();
        assert_eq!(call(&mut app, &ctx, "ui.dialog.confirm", json!({"dialog": "discard", "choice": "dontSave"}))["ok"], true);
        assert_eq!(app.session.documents().len(), 1);
        // Quitting asks about each dirty document, then lets the window close.
        app.run("file.new", json!({"width": 8, "height": 8})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        assert!(crate::discard_ui::intercept(&mut app, "file.exit", &Value::Null));
        let d = prompt(&mut app).expect("the quit prompt is listed");
        assert_eq!((d["fields"]["command"].as_str(), d["fields"]["remaining"].as_u64()), (Some("file.exit"), Some(2)));
        assert_eq!(call(&mut app, &ctx, "ui.dialog.confirm", json!({"dialog": "discard", "choice": "dontSave"}))["result"]["open"], true);
        assert_eq!(prompt(&mut app).unwrap()["fields"]["remaining"], 1);
        assert_eq!(call(&mut app, &ctx, "ui.dialog.confirm", json!({"dialog": "discard", "choice": "dontSave"}))["result"]["open"], false);
        assert!(app.allow_close && app.session.documents().is_empty());
    }

    #[test]
    fn shell_dialogs_are_listed_and_driven_through_ui_dialog() {
        // #1004: Window › Workspace and View dialogs live in `ui.shell.dialog`, and ui.inspect and
        // ui.dialog.* only knew `ui.dialogs`.
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        let shell = |app: &mut PhotocraftApp| {
            let v = call(app, &ctx, "ui.inspect", json!({}));
            v["result"]["dialogs"].as_array().unwrap().iter().find(|d| d["id"] == "shell").cloned()
        };
        // Listed, then cancelled.
        crate::menus::invoke(&mut app, &ctx, "window.workspace.newWorkspace", json!({})).unwrap();
        let d = shell(&mut app).expect("the open shell dialog is listed");
        assert_eq!((d["kind"].as_str(), d["title"].as_str()), (Some("newWorkspace"), Some("New Workspace")));
        assert_eq!(call(&mut app, &ctx, "ui.dialog.cancel", json!({"dialog": "shell"}))["ok"], true);
        assert!(app.ui.shell.dialog.is_none() && shell(&mut app).is_none());
        assert_eq!(call(&mut app, &ctx, "ui.dialog.cancel", json!({"dialog": "shell"}))["ok"], false, "nothing left to cancel");
        // A field set over control reaches OK.
        crate::menus::invoke(&mut app, &ctx, "view.show.showExtrasOptions", json!({})).unwrap();
        assert!(app.ui.view.show.notes);
        assert_eq!(call(&mut app, &ctx, "ui.dialog.set", json!({"dialog": "shell", "field": "notes", "value": false}))["ok"], true);
        assert_eq!(shell(&mut app).unwrap()["fields"]["notes"], false);
        assert_eq!(call(&mut app, &ctx, "ui.dialog.confirm", json!({"dialog": "shell"}))["ok"], true);
        assert!(!app.ui.view.show.notes && app.ui.shell.dialog.is_none());
        // An OK that fails keeps the dialog open and says why.
        crate::menus::invoke(&mut app, &ctx, "view.show.showExtrasOptions", json!({})).unwrap();
        call(&mut app, &ctx, "ui.dialog.set", json!({"dialog": "shell", "field": "bogus", "value": true}));
        let r = call(&mut app, &ctx, "ui.dialog.confirm", json!({"dialog": "shell"}));
        assert_eq!(r["ok"], false);
        assert!(r["error"].as_str().unwrap_or("").contains("bogus"), "{r}");
        assert!(app.ui.shell.dialog.is_some());
        // Apply is Preferences-only; numeric ids still address ordinary dialogs only.
        assert_eq!(call(&mut app, &ctx, "ui.dialog.apply", json!({"dialog": "shell"}))["ok"], false);
        assert_eq!(call(&mut app, &ctx, "ui.dialog.cancel", json!({"dialog": 0}))["ok"], false);
        assert!(app.ui.shell.dialog.is_some());
    }

    #[test]
    fn opening_new_document_through_control_matches_clipboard_image_size() {
        // #2034: every way of opening New Document uses the clipboard-aware initial fields.
        let clipboard = std::sync::Arc::new(std::sync::Mutex::new(Some((100, 200, vec![255; 100 * 200 * 4]))));
        let image = clipboard.clone();
        let mut app = PhotocraftApp::new(
            photocraft_engine::Session::new(),
            crate::Services { clipboard_get_image: Some(Box::new(move || image.lock().ok()?.clone())), ..Default::default() },
        );
        let ctx = egui::Context::default();
        let opened = call(&mut app, &ctx, "ui.dialog.open", json!({"kind":"newDocument"}));
        assert_eq!(opened["ok"], true);
        let id = opened["result"]["dialog"].as_u64().expect("dialog id is returned");
        let fields = &app.ui.dialog_mut(id).expect("dialog was opened").fields;
        assert_eq!((fields["width"].as_u64(), fields["height"].as_u64()), (Some(100), Some(200)));
        assert_eq!(fields["__preset"], "Clipboard");
    }

    #[test]
    fn expired_queued_edit_does_not_run_and_a_live_retry_runs_once() {
        use std::time::{Duration, Instant};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 8, "height": 8})).unwrap();
        let before = app.session.active().unwrap();
        let layers = before.doc.layers.len();
        let steps = before.history.past_len();
        let (tx, rx) = std::sync::mpsc::channel();
        app = app.with_control(rx);
        let (mut expired, expired_reply) = ControlRequest::new("engine.execute", json!({"command": "layer.new.layer"}));
        expired.deadline = Some(Instant::now() - Duration::from_secs(1));
        tx.send(expired).unwrap();
        let (mut retry, retry_reply) = ControlRequest::new("engine.execute", json!({"command": "layer.new.layer"}));
        retry.deadline = Some(Instant::now() + Duration::from_secs(60));
        tx.send(retry).unwrap();
        app.drain_control(&egui::Context::default());
        let expired_error = expired_reply.try_recv().unwrap()["error"].as_str().unwrap().to_string();
        assert_eq!(retry_reply.try_recv().unwrap()["ok"], true);
        let after = app.session.active().unwrap();
        assert_eq!(after.doc.layers.len(), layers + 1, "only the live retry adds a layer");
        assert_eq!(after.history.past_len(), steps + 1, "the expired edit adds no undo step");
        // #3662: a caller must be able to tell this reply, safe to re-send, from the
        // transport's, sent while the command may still apply.
        assert!(expired_error.starts_with("timeout"), "machines match on the prefix: {expired_error}");
        assert!(expired_error.contains("not run"), "the expired request says it never ran: {expired_error}");
        let transport_error = super::timeout_error();
        assert!(!transport_error.contains("not run"), "the transport can't know it didn't run: {transport_error}");
        assert!(transport_error.contains("may still apply"), "the transport warns against a blind re-send: {transport_error}");
    }

    #[test]
    fn a_transport_that_gives_up_learns_whether_the_command_can_still_run() {
        // #3662: the transport's clock runs out no later than the UI thread's, so it is the
        // one that answers; it must still tell a request that will never run from one that may.
        use std::time::{Duration, Instant};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 8, "height": 8})).unwrap();
        let layers = app.session.active().unwrap().doc.layers.len();
        let (tx, rx) = std::sync::mpsc::channel();
        app = app.with_control(rx);
        let ctx = egui::Context::default();
        let request = |deadline: Instant| {
            let (mut req, reply) = ControlRequest::new("engine.execute", json!({"command": "layer.new.layer"}));
            req.deadline = Some(deadline);
            (req.dispatch.clone(), req, reply)
        };

        let (dispatch, queued, _) = request(Instant::now() + Duration::from_secs(60));
        tx.send(queued).unwrap();
        assert_eq!(dispatch.give_up(), super::expired_error(), "still queued: it will not run");
        app.drain_control(&ctx);
        assert_eq!(app.session.active().unwrap().doc.layers.len(), layers, "a request given up on never runs");

        let (dispatch, expired, _) = request(Instant::now() - Duration::from_secs(1));
        tx.send(expired).unwrap();
        app.drain_control(&ctx);
        assert_eq!(dispatch.give_up(), super::expired_error(), "rejected by the UI thread: it did not run");

        let (dispatch, live, reply) = request(Instant::now() + Duration::from_secs(60));
        tx.send(live).unwrap();
        app.drain_control(&ctx);
        assert_eq!(reply.try_recv().unwrap()["ok"], true);
        assert_eq!(app.session.active().unwrap().doc.layers.len(), layers + 1);
        assert_eq!(dispatch.give_up(), super::timeout_error(), "dispatched: it may still apply");
    }

    #[test]
    fn timeout_errors_name_the_likely_causes() {
        // #2489: the reply must tell "wait" from "check the machine", and its number
        // must come from the deadline it reports.
        let error = super::timeout_error();
        assert!(error.starts_with("timeout"), "machines match on the prefix");
        assert!(error.contains(&format!("{}s", super::REPLY_DEADLINE.as_secs())), "the text matches the deadline: {error}");
        assert!(error.contains("may still be running"), "the wait-it-out cause");
        assert!(error.contains("display asleep"), "the check-the-display cause");
        assert!(error.contains("frame"), "the frame-loop cause");
    }

    #[test]
    fn deadline_free_requests_run_even_after_the_receiver_is_dropped() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 8, "height": 8})).unwrap();
        let layers = app.session.active().unwrap().doc.layers.len();
        let (tx, rx) = std::sync::mpsc::channel();
        app = app.with_control(rx);
        let (req, reply) = ControlRequest::new("engine.execute", json!({"command": "layer.new.layer"}));
        assert!(req.deadline.is_none());
        drop(reply);
        tx.send(req).unwrap();
        app.drain_control(&egui::Context::default());
        assert_eq!(app.session.active().unwrap().doc.layers.len(), layers + 1);
    }

    #[test]
    fn right_pointer_opens_agent_visible_selection_menu() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width": 32, "height": 32})).unwrap();
        app.run("select.rect", json!({"x": 1, "y": 1, "width": 8, "height": 8})).unwrap();
        let result = call(
            &mut app,
            &ctx,
            "ui.pointer",
            json!({
                "tool": "RectMarquee", "button": "secondary",
                "events": [{"kind": "down", "x": 4, "y": 4}, {"kind": "up", "x": 4, "y": 4}]
            }),
        );
        assert_eq!(result.get("ok"), Some(&Value::Bool(true)));
        let inspected = call(&mut app, &ctx, "ui.inspect", json!({}));
        let entries = inspected.pointer("/result/canvasToolMenu/entries").and_then(Value::as_array).unwrap();
        assert!(entries.iter().any(|entry| entry.get("id") == Some(&json!("select.inverse")) && entry.get("enabled") == Some(&json!(true))));
        assert!(app.session.active().unwrap().doc.selection.is_some(), "right-click must not edit selection");
    }

    #[test]
    fn agent_right_click_with_the_zoom_tool_opens_and_runs_the_view_menu() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width": 32, "height": 32})).unwrap();
        app.sync_views();
        let result = call(
            &mut app,
            &ctx,
            "ui.pointer",
            json!({"tool": "Zoom", "button": "secondary", "events": [{"kind":"down","x":8,"y":8},{"kind":"up","x":8,"y":8}]}),
        );
        assert_eq!(result["ok"], true);
        assert_eq!(app.ui.canvas_tool_menu.as_ref().and_then(|m| m.anchor), Some([8.0, 8.0]));
        let inspected = call(&mut app, &ctx, "ui.inspect", json!({}));
        let entries = inspected.pointer("/result/canvasToolMenu/entries").and_then(Value::as_array).unwrap();
        let ids: Vec<_> = entries.iter().filter_map(|row| row.get("id").and_then(Value::as_str)).collect();
        assert_eq!(
            ids,
            ["view.zoomIn", "view.zoomOut", "view.fitOnScreen", "view.fillScreen", "view.actualPixels", "view.twoHundredPercent", "view.printSize"]
        );
        assert_eq!(call(&mut app, &ctx, "ui.context.choose", json!({"id":"select.all"}))["ok"], false);
        assert_eq!(call(&mut app, &ctx, "ui.context.choose", json!({"id":"view.twoHundredPercent"}))["ok"], true);
        assert_eq!(app.ui.views[0].zoom, 2.0);
        assert!(app.ui.canvas_tool_menu.is_none());
    }

    #[test]
    fn agent_can_inspect_and_choose_pen_make_selection_from_full_context_menu() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width": 32, "height": 32})).unwrap();
        app.run("path.set", json!({"name":"work","path":{"subpaths":[{"closed":true,"knots":[[2,2],[20,2],[20,20]]}]}})).unwrap();
        let expected_pos = screen_point(&app, 8.0, 8.0);
        let result = call(
            &mut app,
            &ctx,
            "ui.pointer",
            json!({
                "tool": "Pen", "button": "secondary", "events": [{"kind":"down","x":8,"y":8},{"kind":"up","x":8,"y":8}]
            }),
        );
        assert_eq!(result["ok"], true);
        let inspected = call(&mut app, &ctx, "ui.inspect", json!({}));
        assert_eq!(app.ui.canvas_tool_menu.as_ref().unwrap().pos, expected_pos);
        let entries = inspected.pointer("/result/canvasToolMenu/entries").and_then(Value::as_array).unwrap();
        assert_eq!(entries.iter().filter(|row| row.get("separator") == Some(&json!(true))).count(), 9);
        assert!(entries.iter().any(|row| row.get("id") == Some(&json!("path.toSelection")) && row.get("enabled") == Some(&json!(true))));
        assert_eq!(call(&mut app, &ctx, "ui.context.choose", json!({"id":"file.new"}))["ok"], false);
        let chosen = call(&mut app, &ctx, "ui.context.choose", json!({"id":"path.toSelection"}));
        assert_eq!(chosen["ok"], true);
        let dialog = chosen["result"]["dialog"].as_u64().unwrap();
        assert_eq!(call(&mut app, &ctx, "ui.dialog.confirm", json!({"dialog":dialog}))["ok"], true);
        assert!(app.session.active().unwrap().doc.selection.is_some());
    }

    #[test]
    fn ui_inspect_reports_the_view_preferences() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        let before = call(&mut app, &ctx, "ui.inspect", json!({}));
        assert_eq!(before.pointer("/result/view/show/selection_edges"), Some(&json!(true)));
        assert_eq!(before.pointer("/result/view/screen_mode"), Some(&json!("standard")));
        assert_eq!(call(&mut app, &ctx, "ui.menu.invoke", json!({"id": "view.show.selectionEdges"}))["ok"], true);
        let after = call(&mut app, &ctx, "ui.inspect", json!({}));
        assert_eq!(after.pointer("/result/view/show/selection_edges"), Some(&json!(false)));
        assert_eq!(after["result"]["view"], json!(app.ui.view));
    }

    #[test]
    fn engine_execute_runs_with_defaults_but_menu_invoke_opens_the_dialog() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width": 32, "height": 32})).unwrap();
        app.run("edit.fill", json!({"color": "#808080"})).unwrap();
        let rev = app.session.active().unwrap().revision;
        let r = call(&mut app, &ctx, "engine.execute", json!({"command": "filter.blur.gaussianBlur"}));
        assert!(!r.to_string().contains("\"dialog\""), "engine.execute opened a dialog: {r}");
        assert!(app.session.active().unwrap().revision > rev, "engine.execute didn't run the filter");
        let rev = app.session.active().unwrap().revision;
        let r = call(&mut app, &ctx, "ui.menu.invoke", json!({"id": "filter.blur.gaussianBlur"}));
        assert!(r.to_string().contains("dialog"), "ui.menu.invoke should open the dialog: {r}");
        assert_eq!(app.session.active().unwrap().revision, rev, "opening a dialog must not edit the document");
    }

    #[test]
    fn menu_invoke_cannot_edit_document_behind_an_open_dialog() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width": 400, "height": 300})).unwrap();

        let opened = call(&mut app, &ctx, "ui.menu.invoke", json!({"id": "image.imageSize"}));
        assert_eq!(opened["ok"], true, "{opened}");
        let dialog = opened["result"]["dialog"].as_u64().unwrap();
        let revision = app.session.active().unwrap().revision;
        let before = call(&mut app, &ctx, "ui.inspect", json!({}));
        let fields = before["result"]["dialogs"][0]["fields"].clone();

        for params in [json!({"id": "image.imageRotation.90cw"}), json!({"id": "image.imageRotation.90cw", "params": {}})] {
            let blocked = call(&mut app, &ctx, "ui.menu.invoke", params);
            assert_eq!(blocked["ok"], false, "{blocked}");
            assert_eq!(app.session.active().unwrap().revision, revision);
            let snapshot = call(&mut app, &ctx, "ui.inspect", json!({}));
            assert_eq!(snapshot["result"]["dialogs"][0]["id"], dialog);
            assert_eq!(snapshot["result"]["dialogs"][0]["fields"], fields);
        }

        // View navigation stays permitted by the same policy as native menus and shortcuts.
        assert!(crate::menus::modal_allows(&app, "view.zoomIn"));
        assert!(!crate::menus::modal_allows(&app, "image.imageRotation.90cw"));

        assert_eq!(call(&mut app, &ctx, "ui.dialog.cancel", json!({"dialog": dialog}))["ok"], true);
        let rotated = call(&mut app, &ctx, "ui.menu.invoke", json!({"id": "image.imageRotation.90cw"}));
        assert_eq!(rotated["ok"], true, "{rotated}");
        assert!(app.session.active().unwrap().revision > revision);
    }

    #[test]
    fn ui_set_rejects_unknown_fields_before_changing_anything() {
        use crate::theme::ThemeKind;
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        app.set_theme(&ctx, ThemeKind::StudioLight);
        // `dark` was advertised to MCP clients but never read; a typo looked like success too.
        for params in [json!({"dark": true}), json!({"thme": "classic"}), json!({"tool": "move", "dark": true})] {
            let r = call(&mut app, &ctx, "ui.set", params.clone());
            assert_eq!(r["ok"], false, "{params}: {r}");
            assert!(r["error"].as_str().unwrap().contains("unknown field"), "{r}");
        }
        assert_eq!(app.ui.theme, ThemeKind::StudioLight);
        assert_eq!(app.ui.tool, Tool::Brush, "a rejected call applies none of its fields");
        // Every field the method reads gets past the check (a bad value is its own error).
        for field in UI_SET_FIELDS {
            let r = call(&mut app, &ctx, "ui.set", json!({ field: null }));
            assert!(!r.to_string().contains("unknown field"), "{field}: {r}");
        }
        assert_eq!(call(&mut app, &ctx, "ui.set", Value::Null)["ok"], true);
    }

    #[test]
    fn ui_set_gradient_blend_mode_validates_and_updates_options() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        let good = call(&mut app, &ctx, "ui.set", json!({"tool": "gradient", "gradientBlendMode": "Difference", "gradientClassic": true}));
        assert_eq!(good["ok"], true, "{good}");
        assert_eq!(app.ui.tool_options.gradient_blend_mode, photocraft_color::BlendMode::Difference);
        assert!(app.ui.tool_options.gradient_classic);
        let bad = call(&mut app, &ctx, "ui.set", json!({"gradientBlendMode": "nonsense", "gradientClassic": false}));
        assert_eq!(bad["ok"], false, "{bad}");
        assert!(app.ui.tool_options.gradient_classic, "invalid mode must not change options");
    }

    #[test]
    fn ui_set_toggles_the_developer_tile_grid() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        assert!(!app.ui.view.tile_grid, "off by default");
        assert_eq!(call(&mut app, &ctx, "ui.set", json!({"tileGrid": true}))["ok"], true);
        assert!(app.ui.view.tile_grid);
        assert_eq!(call(&mut app, &ctx, "ui.inspect", json!({}))["result"]["view"]["tile_grid"], true);
        // Not a View › Show item: Extras doesn't hide it, and Show Extras Options doesn't list it.
        crate::menus::invoke(&mut app, &ctx, "view.extras", json!({})).unwrap();
        assert!(!app.ui.view.extras && app.ui.view.tile_grid);
        assert!(serde_json::to_value(app.ui.view.show).unwrap().get("tile_grid").is_none());
        let bad = call(&mut app, &ctx, "ui.set", json!({"tileGrid": "yes"}));
        assert_eq!(bad["ok"], false, "{bad}");
        assert!(app.ui.view.tile_grid, "a rejected value changes nothing");
        assert_eq!(call(&mut app, &ctx, "ui.set", json!({"tileGrid": false}))["ok"], true);
        assert!(!app.ui.view.tile_grid);
    }

    #[test]
    fn ui_set_drives_the_eyedropper_options() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        let good = call(
            &mut app,
            &ctx,
            "ui.set",
            json!({"tool": "eyedropper", "eyedropperSampleSize": 11, "eyedropperSample": "currentAndBelow", "eyedropperRing": false}),
        );
        assert_eq!(good["ok"], true, "{good}");
        let o = &app.ui.tool_options;
        assert_eq!((o.eyedropper_size, o.eyedropper_sample.as_str(), o.eyedropper_ring), (11, "currentAndBelow", false));
        assert_eq!(call(&mut app, &ctx, "ui.set", json!({"eyedropperSampleSize": "point"}))["ok"], true);
        assert_eq!(app.ui.tool_options.eyedropper_size, 1);
        for bad in [
            json!({"eyedropperSampleSize": 4}),
            json!({"eyedropperSampleSize": "big"}),
            json!({"eyedropperSample": "below"}),
            json!({"eyedropperSample": 2}),
            json!({"eyedropperRing": "yes"}),
        ] {
            let r = call(&mut app, &ctx, "ui.set", bad.clone());
            assert_eq!(r["ok"], false, "{bad}: {r}");
        }
        // A rejected call applies none of its fields.
        let r = call(&mut app, &ctx, "ui.set", json!({"eyedropperSampleSize": 101, "eyedropperSample": "nope"}));
        assert_eq!(r["ok"], false, "{r}");
        assert_eq!(app.ui.tool_options.eyedropper_size, 1);
    }

    /// #1046: the Color Sampler tool is selectable over the control channel and reported as active.
    #[test]
    fn ui_set_selects_the_color_sampler_tool() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        assert_eq!(call(&mut app, &ctx, "ui.set", json!({"tool": "colorSampler"}))["ok"], true);
        assert_eq!(app.ui.tool, Tool::ColorSampler);
        assert_eq!(call(&mut app, &ctx, "ui.inspect", json!({}))["result"]["tool"], "ColorSampler");
        // The `I` key group cycles through the eyedropper tools, so the sampler is on the same key.
        assert_eq!(Tool::ColorSampler.key(), 'I');
    }

    /// #1919: the Crop tool's overlay menu over the control channel, reported by `ui.inspect`.
    #[test]
    fn ui_set_drives_the_crop_overlay_options() {
        use crate::crop_overlay::{CropOverlay, OverlayShow};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        let good =
            call(&mut app, &ctx, "ui.set", json!({"tool": "crop", "cropOverlay": "goldenSpiral", "cropOverlayShow": "always", "cropOverlayOrientation": 2}));
        assert_eq!(good["ok"], true, "{good}");
        let o = &app.ui.tool_options;
        assert_eq!((o.crop_overlay, o.crop_overlay_show, o.crop_overlay_orientation), (CropOverlay::GoldenSpiral, OverlayShow::Always, 2));
        let seen = call(&mut app, &ctx, "ui.inspect", json!({}));
        let t = &seen["result"]["toolOptions"];
        assert_eq!(
            (t["crop_overlay"].as_str(), t["crop_overlay_show"].as_str(), t["crop_overlay_orientation"].as_u64()),
            (Some("goldenSpiral"), Some("always"), Some(2))
        );
        for bad in [
            json!({"cropOverlay": "spiral"}),
            json!({"cropOverlay": 1}),
            json!({"cropOverlayShow": "sometimes"}),
            json!({"cropOverlayOrientation": 4}),
            json!({"cropOverlayOrientation": -1}),
            json!({"cropOverlayOrientation": "1"}),
            json!({"cropOverlay": "grid", "cropOverlayShow": "nope"}),
        ] {
            let r = call(&mut app, &ctx, "ui.set", bad.clone());
            assert_eq!(r["ok"], false, "{bad}: {r}");
        }
        assert_eq!(app.ui.tool_options.crop_overlay, CropOverlay::GoldenSpiral, "a rejected call applies none of its fields");
    }

    /// #1919: the Crop tool's gear menu (Show Cropped Area, crop shield) over the control channel.
    #[test]
    fn ui_set_drives_the_crop_shield_options() {
        use crate::crop_shield::{CropShield, ShieldColor};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        let good = call(&mut app, &ctx, "ui.set", json!({"cropShield": {"color": "custom", "custom_color": [255, 0, 0], "opacity": 40}}));
        assert_eq!(good["ok"], true, "{good}");
        let want = CropShield { color: ShieldColor::Custom, custom_color: [255, 0, 0], opacity: 40.0, ..Default::default() };
        assert_eq!(app.ui.tool_options.crop_shield, want);
        // A patch: the other fields stay.
        assert_eq!(call(&mut app, &ctx, "ui.set", json!({"cropShield": {"show_cropped_area": false}}))["ok"], true);
        assert_eq!(app.ui.tool_options.crop_shield, CropShield { show_cropped_area: false, ..want.clone() });
        let t = &call(&mut app, &ctx, "ui.inspect", json!({}))["result"]["toolOptions"]["crop_shield"];
        assert_eq!((t["color"].as_str(), t["opacity"].as_f64(), t["show_cropped_area"].as_bool()), (Some("custom"), Some(40.0), Some(false)));
        for bad in [
            json!({"cropShield": true}),
            json!({"cropShield": {"opacity": 101}}),
            json!({"cropShield": {"opacity": -1}}),
            json!({"cropShield": {"opacity": "50"}}),
            json!({"cropShield": {"color": "red"}}),
            json!({"cropShield": {"custom_color": [256, 0, 0]}}),
            json!({"cropShield": {"shield": true}}),
            json!({"cropShield": {"enabled": false}, "cropOverlay": "nope"}),
        ] {
            let r = call(&mut app, &ctx, "ui.set", bad.clone());
            assert_eq!(r["ok"], false, "{bad}: {r}");
        }
        assert_eq!(app.ui.tool_options.crop_shield, CropShield { show_cropped_area: false, ..want }, "a rejected call applies none of its fields");
    }

    #[test]
    fn ui_set_opens_the_brush_preset_picker_and_sets_its_cards() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        let list = &app.ui.brush_picker_list;
        assert!(list.show_name && list.show_stroke && list.show_tip, "every card part is on by default");
        let r = call(&mut app, &ctx, "ui.set", json!({"tool": "brush", "brushPicker": [120, 80], "brushPickerStroke": false}));
        assert_eq!(r["ok"], true, "{r}");
        assert_eq!(app.ui.brush_picker, Some([120.0, 80.0]));
        assert!(!app.ui.brush_picker_list.show_stroke);
        for bad in [
            json!({"brushPicker": [1]}),
            json!({"brushPicker": "here"}),
            json!({"brushPickerStroke": "nope"}),
            json!({"brushPickerName": false, "brushPickerStroke": false, "brushPickerTip": false}),
        ] {
            assert_eq!(call(&mut app, &ctx, "ui.set", bad.clone())["ok"], false, "{bad}");
        }
        assert_eq!(app.ui.brush_picker, Some([120.0, 80.0]), "a bad value leaves the picker alone");
        assert!(app.ui.brush_picker_list.show_name, "a rejected call leaves the card parts alone");
        assert_eq!(call(&mut app, &ctx, "ui.set", json!({"brushPicker": null}))["ok"], true);
        assert_eq!(app.ui.brush_picker, None);
    }

    #[test]
    fn ui_inspect_tracks_brush_picker_open_and_closed() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width": 128, "height": 128})).unwrap();

        let closed = call(&mut app, &ctx, "ui.inspect", json!({}));
        assert!(closed["result"]["brushPicker"].is_null());

        // A right-click on the canvas with Brush opens the same picker as the options bar.
        assert_eq!(call(&mut app, &ctx, "ui.set", json!({"tool": "brush"}))["ok"], true);
        let click = call(
            &mut app,
            &ctx,
            "ui.pointer",
            json!({
                "button": "right",
                "events": [{"kind": "down", "x": 64, "y": 64}, {"kind": "up", "x": 64, "y": 64}]
            }),
        );
        assert_eq!(click["ok"], true, "{click}");
        let open = call(&mut app, &ctx, "ui.inspect", json!({}));
        assert!(open["result"]["brushPicker"]["pos"].is_array(), "{open}");
        assert_eq!(open["result"]["brushPicker"]["list"]["showName"], true);
        assert_eq!(open["result"]["brushPicker"]["list"]["showStroke"], true);
        assert_eq!(open["result"]["brushPicker"]["list"]["showTip"], true);

        assert_eq!(call(&mut app, &ctx, "ui.set", json!({"brushPickerTip": false}))["ok"], true);
        let changed = call(&mut app, &ctx, "ui.inspect", json!({}));
        assert_eq!(changed["result"]["brushPicker"]["list"]["showTip"], false);

        assert_eq!(call(&mut app, &ctx, "ui.set", json!({"brushPicker": null}))["ok"], true);
        let closed_again = call(&mut app, &ctx, "ui.inspect", json!({}));
        assert!(closed_again["result"]["brushPicker"].is_null());
    }

    #[test]
    fn ui_set_applies_all_fields_or_none() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        let before = app.ui.tool;
        let panels_before = serde_json::to_value(&app.ui.panels).unwrap_or_default();
        // A late invalid field must not leave the earlier valid one applied - including the
        // journaled brush command, which is validated up front for exactly this reason.
        for params in
            [json!({"tool": "move", "theme": "nope"}), json!({"tool": "move", "brushSize": 1e300}), json!({"tool": "move", "colorPanel": {"nope": true}})]
        {
            let r = call(&mut app, &ctx, "ui.set", params.clone());
            assert_eq!(r["ok"], false, "{params}: {r}");
            assert_eq!(app.ui.tool, before, "{params}: a rejected call applies none of its fields");
        }
        // Wrong types and unknown nested keys are errors, not silent no-ops with ok:true.
        for params in [
            json!({"panels": "x"}),
            json!({"panels": {"nope": true}}),
            json!({"dockTabs": {"nope": 1}}),
            json!({"dock": {"nope": 1}}),
            json!({"center": [1]}),
            json!({"center": [1, 2, 3]}),
            json!({"center": "here"}),
            json!({"colorPanel": "x"}),
            json!({"selectionMode": "add"}),
        ] {
            let r = call(&mut app, &ctx, "ui.set", params.clone());
            assert_eq!(r["ok"], false, "{params}: {r}");
        }
        assert_eq!(serde_json::to_value(&app.ui.panels).unwrap_or_default(), panels_before, "nothing was applied");
        // Non-numeric brushSize keeps the API's historical silent no-op (see the dispatch test).
        assert_eq!(call(&mut app, &ctx, "ui.set", json!({"brushSize": "large"}))["ok"], true);
    }

    #[test]
    fn ui_set_theme_keeps_explicit_dock_tab_indices() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        for theme in ["adwaitaDark", "pro"] {
            let r = call(&mut app, &ctx, "ui.set", json!({"theme": theme, "dockTabs": {"color": 0}}));
            assert_eq!(r["ok"], true, "{r}");
            assert_eq!(app.ui.dock_tabs.color, 0, "{theme}: explicit tab index changed");
        }
    }

    #[test]
    fn ui_set_color_panel_picks_the_edited_chip() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        assert_eq!(call(&mut app, &ctx, "ui.set", json!({"colorPanel": {"background": true}}))["ok"], true);
        assert!(app.ui.color_panel.background);
        assert_eq!(call(&mut app, &ctx, "ui.set", json!({"colorPanel": {"background": "yes"}}))["ok"], false);
        assert!(app.ui.color_panel.background, "a bad value changes nothing");
    }

    #[test]
    fn ui_set_unknown_theme_error_names_every_theme_and_each_name_works() {
        use crate::theme::ThemeKind;
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        let r = call(&mut app, &ctx, "ui.set", json!({"theme": "nope"}));
        assert_eq!(r["ok"], false);
        let error = r["error"].as_str().unwrap();
        for kind in ThemeKind::ALL {
            assert!(error.contains(kind.id()), "{error} lacks {}", kind.id());
            assert_eq!(call(&mut app, &ctx, "ui.set", json!({"theme": kind.id()}))["ok"], true);
            assert_eq!(app.ui.theme, kind);
            assert_eq!(ThemeKind::from_name(kind.id()), Some(kind));
        }
    }

    #[test]
    fn ui_window_open_rejects_an_unknown_document_index() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width": 8, "height": 8})).unwrap();
        // An out-of-range index must error instead of silently switching the active document
        // and opening the window for whatever is active now.
        let before = app.session.active_index();
        let r = call(&mut app, &ctx, "ui.window.open", json!({"document": 99}));
        assert_eq!(r["ok"], false, "{r}");
        assert!(r["error"].as_str().unwrap().contains("no document 99"), "{r}");
        assert_eq!(app.session.active_index(), before, "the active document is untouched");
        // A valid index still opens the window.
        assert_eq!(call(&mut app, &ctx, "ui.window.open", json!({"document": 0}))["ok"], true);
    }

    #[test]
    fn ui_set_brush_size_dispatches_a_journaled_brush_command() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();

        let r = call(&mut app, &ctx, "ui.set", json!({"brushSize": 42.5}));
        assert_eq!(r["ok"], true, "{r}");
        assert_eq!(app.session.tools.brush.size, 42.5);
        let (id, params) = app.session.journal.last().cloned().expect("brush change is journaled");
        assert_eq!(id, "tools.setBrush");
        assert_eq!(params, json!({"brush": {"size": 42.5}}));

        // The control API has historically ignored non-numeric optional values.
        for params in [json!({}), json!({"brushSize": null}), json!({"brushSize": "large"}), json!({"brushSize": true})] {
            let journal_len = app.session.journal.len();
            let r = call(&mut app, &ctx, "ui.set", params);
            assert_eq!(r["ok"], true, "{r}");
            assert_eq!(app.session.tools.brush.size, 42.5);
            assert_eq!(app.session.journal.len(), journal_len);
        }

        // Each tool keeps its own brush (#218), so `tool` + `brushSize` in one call sets the new
        // tool's size rather than the one it was carrying.
        call(&mut app, &ctx, "ui.set", json!({"tool": "eraser", "brushSize": 12.0}));
        assert_eq!(app.session.tools.brush.size, 12.0, "the Eraser's own size");
        call(&mut app, &ctx, "ui.set", json!({"tool": "brush"}));
        assert_eq!(app.session.tools.brush.size, 42.5, "the Brush gets its own back");
        call(&mut app, &ctx, "ui.set", json!({"tool": "eraser"}));
        assert_eq!(app.session.tools.brush.size, 12.0);
        call(&mut app, &ctx, "ui.set", json!({"tool": "brush", "brushSize": 42.5}));

        // Values that cannot be represented by BrushSettings must report the command error and
        // leave both the brush and journal unchanged instead of mutating tool state directly.
        let journal_len = app.session.journal.len();
        let r = call(&mut app, &ctx, "ui.set", json!({"brushSize": f64::MAX}));
        assert_eq!(r["ok"], false, "{r}");
        assert!(r["error"].as_str().is_some(), "{r}");
        assert_eq!(app.session.tools.brush.size, 42.5);
        assert_eq!(app.session.journal.len(), journal_len);
    }

    #[test]
    fn preferences_apply_is_available_over_control_without_closing() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        let r = call(&mut app, &ctx, "ui.menu.invoke", json!({"id": "edit.preferences.interface"}));
        let id = r["result"]["dialog"].as_u64().unwrap();
        let mut values = app.ui.dialog_mut(id).unwrap().fields["values"].clone();
        values["interface"]["uiScale"] = json!("200");
        assert_eq!(call(&mut app, &ctx, "ui.dialog.set", json!({"dialog": id, "field": "values", "value": values}))["ok"], true);
        assert_eq!(call(&mut app, &ctx, "ui.dialog.apply", json!({"dialog": id}))["ok"], true);
        assert_eq!(app.session.prefs().interface.ui_scale, photocraft_engine::prefs::UiScale::P200);
        assert!(app.ui.dialog_mut(id).is_some());
        assert_eq!(call(&mut app, &ctx, "ui.dialog.cancel", json!({"dialog": id}))["ok"], true);
        assert_eq!(app.session.prefs().interface.ui_scale, photocraft_engine::prefs::UiScale::P200);
        for params in [json!({}), json!({"dialog": "invalid"}), json!({"dialog": u64::MAX})] {
            assert_eq!(call(&mut app, &ctx, "ui.dialog.apply", params)["ok"], false);
        }
    }

    #[test]
    fn preferences_apply_respects_the_automation_command_policy() {
        let services = crate::Services {
            automation_command: Some(Box::new(|id, _| if id == "prefs.set" { Err("preference changes denied".into()) } else { Ok(()) })),
            ..Default::default()
        };
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
        let ctx = egui::Context::default();
        let before = app.session.prefs().to_json();
        let id = crate::prefs_ui::open_preferences(&mut app, "interface");
        app.ui.dialog_mut(id).unwrap().fields.get_mut("values").unwrap()["interface"]["uiScale"] = json!("200");
        let r = call(&mut app, &ctx, "ui.dialog.apply", json!({"dialog": id}));
        assert_eq!(r["ok"], false);
        assert_eq!(r["error"], "preference changes denied");
        assert_eq!(app.session.prefs().to_json(), before);
        assert!(app.ui.dialog_mut(id).is_some());
    }

    #[test]
    fn synthetic_shortcuts_use_the_automation_command_policy() {
        let services = crate::Services {
            automation_command: Some(Box::new(|id, _| if id.starts_with("file.") { Err("filesystem command denied".into()) } else { Ok(()) })),
            ..Default::default()
        };
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
        let ctx = egui::Context::default();
        app.automation_input = true;
        let error = crate::menus::invoke(&mut app, &ctx, "file.open", json!({})).unwrap_err();
        assert_eq!(error, "filesystem command denied");
    }

    /// `app.save` takes the Targa bits per pixel (`tgaBits`) and never stops at Targa Options.
    #[test]
    fn automation_save_passes_targa_bits_without_asking() {
        use std::cell::RefCell;
        use std::rc::Rc;
        // The bits each export was asked for, and each path written.
        let asked: Rc<RefCell<Vec<Option<u8>>>> = Rc::default();
        let written: Rc<RefCell<Vec<String>>> = Rc::default();
        let (a, w) = (asked.clone(), written.clone());
        let services = crate::Services {
            export: Some(Box::new(move |_d: &photocraft_doc::Document, _p: &str, s: &crate::ExportSettings| {
                a.borrow_mut().push(s.tga_bits.map(photocraft_io::TgaBits::bits));
                Ok((b"out".to_vec(), Vec::new()))
            })),
            automation_write: Some(Box::new(move |path: &str, _b: &[u8]| {
                w.borrow_mut().push(path.to_string());
                Ok(())
            })),
            ..Default::default()
        };
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width": 8, "height": 8})).unwrap();
        for params in [json!({"path": "a.tga"}), json!({"path": "b.tga", "tgaBits": 24}), json!({"path": "c.tga", "tgaBits": 32})] {
            let r = call(&mut app, &ctx, "app.save", params.clone());
            assert_eq!(r["result"]["path"], params["path"], "{r}");
            assert!(app.targa_options.is_none(), "automation saves never prompt");
        }
        assert_eq!(*asked.borrow(), [None, Some(24), Some(32)]);
        assert_eq!(*written.borrow(), ["a.tga", "b.tga", "c.tga"]);
        let r = call(&mut app, &ctx, "app.save", json!({"path": "d.tga", "tgaBits": 16}));
        assert!(r["error"].as_str().unwrap().contains("tgaBits"), "{r}");
        assert_eq!(written.borrow().len(), 3, "a bad tgaBits writes nothing");
    }

    #[test]
    fn automation_open_and_save_use_the_roots_and_reply_with_warnings() {
        use photocraft_color::{ColorMode, SampleType};
        use photocraft_doc::Document;
        use photocraft_geom::Size;
        use std::cell::RefCell;
        use std::rc::Rc;

        // Without granted roots both fail closed, even with an exporter and an ambient writer.
        let services = crate::Services {
            export: Some(Box::new(|_d: &Document, _p: &str, _s: &crate::ExportSettings| Ok((b"out".to_vec(), Vec::new())))),
            write: Some(Box::new(|_p: &str, _b: &[u8]| Err("ambient writer used".into()))),
            ..Default::default()
        };
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
        let ctx = egui::Context::default();
        let r = call(&mut app, &ctx, "app.open", json!({"path": "in/warn.psd"}));
        assert!(r.to_string().contains("automation read authority is not configured"), "{r}");
        app.run("file.new", json!({"width": 8, "height": 8})).unwrap();
        let r = call(&mut app, &ctx, "app.save", json!({"path": "out.png"}));
        assert!(r.to_string().contains("automation write authority is not configured"), "{r}");

        let written: Rc<RefCell<Vec<String>>> = Rc::default();
        let w = written.clone();
        let services = crate::Services {
            import: Some(Box::new(|name: &str, _b: &[u8], _depth: usize| {
                Ok((Document::new(name, Size::new(4, 4), ColorMode::Rgb, SampleType::U8), vec!["Adjustment layer flattened".to_string()]))
            })),
            export: Some(Box::new(|_d: &Document, _p: &str, _s: &crate::ExportSettings| Ok((b"out".to_vec(), vec!["Layers were flattened".to_string()])))),
            automation_read: Some(Box::new(|path: &str| Ok((path.rsplit('/').next().unwrap_or(path).to_string(), b"x".to_vec())))),
            automation_write: Some(Box::new(move |path: &str, _b: &[u8]| {
                w.borrow_mut().push(path.to_string());
                Ok(())
            })),
            ..Default::default()
        };
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
        let r = call(&mut app, &ctx, "app.open", json!({"path": "in/warn.psd"}));
        assert_eq!(r["result"]["warnings"], json!(["Adjustment layer flattened"]), "{r}");
        assert_eq!(r["result"]["name"], "warn.psd");
        assert_eq!(app.session.active().unwrap().path.as_deref(), Some("in/warn.psd"));
        assert_eq!(app.ui.recent_files, vec!["in/warn.psd".to_string()]);
        let r = call(&mut app, &ctx, "app.save", json!({"path": "out.png"}));
        assert_eq!(r["result"], json!({"path": "out.png", "warnings": ["Layers were flattened"]}), "{r}");
        assert_eq!(*written.borrow(), vec!["out.png".to_string()]);
        // A flat export is a copy: the document keeps its name and file (#2579).
        let st = app.session.active().unwrap();
        assert_eq!((st.doc.name.as_str(), st.path.as_deref()), ("warn.psd", Some("in/warn.psd")));
        // Without `path`, only a layered file is written back, like File › Save (#416).
        call(&mut app, &ctx, "app.open", json!({"path": "in/flat.jpg"}));
        let r = call(&mut app, &ctx, "app.save", json!({}));
        assert!(r["error"].as_str().unwrap().contains("pass `path`"), "{r}");
        assert_eq!(written.borrow().len(), 1, "nothing written over the JPEG");
        call(&mut app, &ctx, "app.open", json!({"path": "in/layered.psd"}));
        let r = call(&mut app, &ctx, "app.save", json!({}));
        assert_eq!(r["result"]["path"], "in/layered.psd", "{r}");
        assert_eq!(written.borrow().last().map(String::as_str), Some("in/layered.psd"));
        // A template opens untitled, so a save without `path` never writes over it.
        let r = call(&mut app, &ctx, "app.open", json!({"path": "in/card.psdt"}));
        assert_eq!(r["result"]["name"], "Untitled-1", "{r}");
        assert_eq!(app.session.active().unwrap().path, None);
        let r = call(&mut app, &ctx, "app.save", json!({}));
        assert!(r["error"].as_str().unwrap().contains("pass `path`"), "{r}");
    }

    fn deny_ambient_file(id: &str, _: &serde_json::Value) -> photocraft_engine::Result<()> {
        if id.starts_with("file.") && id != "file.new" {
            Err(photocraft_engine::EngineError::Other(format!("automation command `{id}` is disabled")))
        } else {
            Ok(())
        }
    }

    #[test]
    fn app_open_opens_a_directory_pcraft_and_falls_back_for_the_zip_form() {
        use photocraft_doc::Document;
        use serde_json::json;
        let services = crate::Services {
            automation_open_pcraft: Some(Box::new(|path: &str| {
                if path == "native.pcraft" {
                    let name = path.rsplit('/').next().unwrap_or(path).to_string();
                    let doc = Document::new("Native", photocraft_doc::Size::new(16, 12), photocraft_doc::ColorMode::Rgb, photocraft_doc::SampleType::U8);
                    Ok(Some((name, doc)))
                } else if path == "zipped.pcraft" {
                    Ok(None) // a regular file: the bytes path reads the zip form
                } else {
                    Err(format!("corrupt bundle: {path}"))
                }
            })),
            import: Some(Box::new(|name: &str, _b: &[u8], _depth: usize| {
                Ok((Document::new(name, photocraft_doc::Size::new(4, 4), photocraft_doc::ColorMode::Rgb, photocraft_doc::SampleType::U8), Vec::new()))
            })),
            automation_read: Some(Box::new(|path: &str| {
                if path == "zipped.pcraft" {
                    Ok((path.rsplit('/').next().unwrap_or(path).to_string(), b"zip-bytes".to_vec()))
                } else {
                    Err(format!("no such path: {path}"))
                }
            })),
            ..Default::default()
        };
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
        let ctx = egui::Context::default();

        let r = call(&mut app, &ctx, "app.open", json!({"path": "native.pcraft"}));
        assert_eq!(r["result"], json!({"path": "native.pcraft", "name": "native.pcraft", "warnings": []}), "{r}");
        assert_eq!(app.session.active().unwrap().doc.name, "Native");
        assert_eq!(app.ui.recent_files, vec!["native.pcraft".to_string()], "the open is recorded in recent files");
        assert_eq!(app.session.active().unwrap().path, None, "no save-back target: the automation writer replaces files, never directories");
        let r = call(&mut app, &ctx, "app.save", json!({}));
        assert!(r["error"].as_str().unwrap().contains("pass `path`"), "an in-place save is refused up front: {r}");

        // The single-file form takes the bytes path through the same `.pcraft` extension.
        let r = call(&mut app, &ctx, "app.open", json!({"path": "zipped.pcraft"}));
        assert_eq!(r["result"]["name"], "zipped.pcraft", "{r}");
        assert_eq!(app.session.active().unwrap().doc.name, "zipped.pcraft", "the import served the zip form");

        // A real opener error propagates instead of being masked by the fallback.
        let r = call(&mut app, &ctx, "app.open", json!({"path": "gone.pcraft"}));
        assert!(r["error"].as_str().unwrap().contains("corrupt bundle"), "{r}");
    }

    #[test]
    fn app_save_runs_as_a_background_job_and_the_final_reply_carries_the_path() {
        use photocraft_engine::automate_cmds::ScriptBinding;
        use serde_json::json;
        let saved: std::sync::Arc<std::sync::Mutex<Vec<String>>> = Default::default();
        let record = saved.clone();
        let services = crate::Services {
            automation_save: Some(std::sync::Arc::new(move |_doc, path: &str, _settings, _ctl| {
                record.lock().unwrap().push(path.to_string());
                Ok(vec!["Layers were flattened".to_string()])
            })),
            ..Default::default()
        };
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
        app.background_jobs = true;
        app.run("file.new", json!({"width": 8, "height": 8})).unwrap();
        // A "Save Document" binding that would leave a trace: automation saves never fire it.
        app.session.edit_prefs(|prefs| {
            prefs.script_events.enabled = true;
            prefs.script_events.bindings = vec![ScriptBinding {
                event: "saveDocument".into(),
                script: None,
                steps: Some(json!([["layer.new.layer", {"name": "From the binding"}]])),
                name: "Binding".into(),
            }];
        });
        let ctx = egui::Context::default();
        let (tx, rx) = std::sync::mpsc::channel();
        let app = app.with_control(rx);
        let mut app = app;

        let (mut req, reply) = ControlRequest::new("app.save", json!({"path": "out.psd"}));
        req.deadline = None;
        tx.send(req).unwrap();
        app.drain_control(&ctx);
        // Started in the background: no reply yet, the job is on the document.
        assert!(reply.try_recv().is_err(), "the reply waits for the save job");
        assert!(app.saving(), "the save runs as a background job");
        for _ in 0..100 {
            crate::jobs_ui::tick(&mut app, &ctx);
            if !app.saving() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!app.saving(), "the save finished");
        assert_eq!(*saved.lock().unwrap(), vec!["out.psd".to_string()]);
        assert_eq!(app.session.active().unwrap().path.as_deref(), Some("out.psd"), "the save was recorded");
        assert_eq!(app.session.active().unwrap().doc.layer_count(), 1, "the saveDocument binding did not run for an automation save");
        let r = reply.recv().unwrap();
        assert_eq!(r["result"], json!({"path": "out.psd", "warnings": ["Layers were flattened"]}), "{r}");

        // An interactive save (File › Save's worker) still fires the binding.
        let save = std::sync::Arc::new(|_doc: &photocraft_doc::Document, path: &str, _settings: &crate::ExportSettings, _ctl: &crate::jobs_ui::SaveCtl| {
            let _ = path;
            Ok(Vec::new())
        }) as crate::SaveFileFn;
        crate::jobs_ui::start_save(&mut app, "user.psd".into(), crate::ExportSettings::default(), false, save, false).unwrap();
        for _ in 0..100 {
            crate::jobs_ui::tick(&mut app, &ctx);
            if !app.saving() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!app.saving(), "the interactive save finished");
        assert_eq!(app.session.active().unwrap().doc.layer_count(), 2, "the binding runs for an interactive save");
    }

    #[test]
    fn ui_inspect_replies_while_a_save_job_encodes() {
        // #2488: the control channel must keep answering while a large save runs.
        use serde_json::json;
        let gate = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let wait_for_gate = gate.clone();
        let saved: std::sync::Arc<std::sync::Mutex<Vec<String>>> = Default::default();
        let record = saved.clone();
        let services = crate::Services {
            automation_save: Some(std::sync::Arc::new(move |_doc, path: &str, _settings, ctl| {
                while !wait_for_gate.load(std::sync::atomic::Ordering::Acquire) {
                    ctl.check()?;
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
                record.lock().unwrap().push(path.to_string());
                Ok(Vec::new())
            })),
            ..Default::default()
        };
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
        app.background_jobs = true;
        app.run("file.new", json!({"width": 8, "height": 8})).unwrap();
        let ctx = egui::Context::default();
        let (tx, rx) = std::sync::mpsc::channel();
        let app = app.with_control(rx);
        let mut app = app;

        let (mut save_req, save_reply) = ControlRequest::new("app.save", json!({"path": "out.png", "wait": false}));
        save_req.deadline = None;
        tx.send(save_req).unwrap();
        let (mut inspect_req, inspect_reply) = ControlRequest::new("ui.inspect", json!({}));
        inspect_req.deadline = None;
        tx.send(inspect_req).unwrap();
        app.drain_control(&ctx);

        let pending = save_reply.recv_timeout(std::time::Duration::from_secs(5)).expect("the unwaited save replies at once");
        assert_eq!(pending["result"]["pending"], true, "{pending}");
        assert_eq!(pending["result"]["path"], "out.png", "{pending}");
        assert!(pending["result"]["job"].is_u64(), "the caller can poll or wait: {pending}");
        // While the save is still encoding, the channel answers other requests.
        assert!(saved.lock().unwrap().is_empty(), "the save is still encoding");
        let inspected = inspect_reply.recv_timeout(std::time::Duration::from_secs(5)).expect("ui.inspect is not blocked by the save");
        assert_eq!(inspected["ok"], true, "{inspected}");

        gate.store(true, std::sync::atomic::Ordering::Release);
        for _ in 0..500 {
            crate::jobs_ui::tick(&mut app, &ctx);
            if !app.saving() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(*saved.lock().unwrap(), vec!["out.png".to_string()], "the save completed after the gate opened");
    }

    fn deny_smart_object_paths(id: &str, params: &Value) -> photocraft_engine::Result<()> {
        // The production policy lives in the automation crate. This gate exercises the control
        // session's state-derived path check without adding that dependency to the UI crate.
        if matches!(id, "layer.smartObjects.editContents" | "layer.smartObjects.convertToLayers" | "layer.smartObjects.saveContents")
            && params.get("path").and_then(Value::as_str).is_some()
        {
            Err(photocraft_engine::EngineError::Other("ambient smart-object path denied".into()))
        } else {
            Ok(())
        }
    }

    fn smart_control_app() -> (PhotocraftApp, egui::Context, photocraft_doc::LayerId) {
        let services = crate::Services {
            automation_authorize: Some(deny_smart_object_paths),
            automation_command: Some(Box::new(|id, params| deny_smart_object_paths(id, params).map_err(|e| e.to_string()))),
            ..Default::default()
        };
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
        app.run("file.new", json!({"width": 8, "height": 6, "background": "transparent"})).unwrap();
        app.run("edit.fill", json!({"color": "#ff0000"})).unwrap();
        let id = app.run("layer.smartObjects.convertToSmartObject", json!({})).unwrap()["layer"].as_u64().unwrap();
        (app, egui::Context::default(), photocraft_doc::LayerId(id))
    }

    #[test]
    fn control_edits_nested_embedded_smart_objects_and_converts_an_explicit_target() {
        use photocraft_doc::{LayerContent, SmartSource};
        use std::cell::RefCell;
        use std::rc::Rc;
        use std::sync::Arc;

        let (mut app, ctx, smart) = smart_control_app();
        // Embed this one-smart-layer document inside the outer object: two Edit Contents
        // requests must reach the raster layer, as in a nested product-photo template.
        let nested = photocraft_engine::smart_cmds::encode_source(&app.session.active().unwrap().doc).unwrap();
        app.session
            .edit("Nested fixture", |doc, _| {
                let LayerContent::Smart(sm) = &mut doc.layer_mut(smart).unwrap().content else { panic!("smart fixture") };
                sm.source = SmartSource::Embedded { file_name: "product.pcraft".into(), bytes: Arc::new(nested) };
                Ok(())
            })
            .unwrap();
        let other = app.run("layer.new.layer", json!({"name": "Unrelated"})).unwrap()["layer"].as_u64().unwrap();
        assert_ne!(app.session.active().unwrap().active_layer, Some(smart));
        for expected_document in [1, 2] {
            let r = call(&mut app, &ctx, "engine.execute", json!({"command": "layer.smartObjects.editContents", "params": {"layer": smart.0}}));
            assert_eq!(r["ok"], true, "{r}");
            assert_eq!(r["result"]["document"], expected_document, "{r}");
            assert!(app.session.active().unwrap().path.is_none(), "contents open in memory");
            assert!(app.session.authorize.is_none(), "the request gate must not remain installed");
        }
        let r = call(&mut app, &ctx, "engine.execute", json!({"command": "edit.fill", "params": {"color": "#0000ff"}}));
        assert_eq!(r["ok"], true, "{r}");
        for expected_documents in [2, 1] {
            let r = call(&mut app, &ctx, "engine.execute", json!({"command": "layer.smartObjects.saveContents"}));
            assert_eq!(r["ok"], true, "{r}");
            let r = call(&mut app, &ctx, "engine.execute", json!({"command": "file.close"}));
            assert_eq!(r["ok"], true, "{r}");
            assert_eq!(app.session.documents().len(), expected_documents);
        }
        assert_eq!(app.session.active().unwrap().active_layer, Some(photocraft_doc::LayerId(other)));
        assert_eq!(photocraft_compose::flatten(&app.session.active().unwrap().doc).px[0], [0.0, 0.0, 1.0, 1.0]);
        let r = call(&mut app, &ctx, "engine.execute", json!({"command": "layer.smartObjects.convertToLayers", "params": {"layer": smart.0}}));
        assert_eq!(r["ok"], true, "{r}");
        assert!(app.session.active().unwrap().doc.layer(photocraft_doc::LayerId(other)).is_some(), "the unrelated active layer remains");
        assert!(app.session.authorize.is_none());

        let written = Rc::new(RefCell::new(Vec::new()));
        let output = written.clone();
        app.services.export =
            Some(Box::new(|doc, _, _| photocraft_engine::smart_cmds::encode_source(doc).map(|bytes| (bytes, Vec::new())).map_err(|e| e.to_string())));
        app.services.automation_write = Some(Box::new(move |path, bytes| {
            output.borrow_mut().push((path.to_string(), bytes.to_vec()));
            Ok(())
        }));
        app.services.write = Some(Box::new(|_, _| Err("ambient writer used".into())));
        let r = call(&mut app, &ctx, "app.save", json!({"path": "out/template.pcraft"}));
        assert_eq!(r["ok"], true, "{r}");
        let written = written.borrow();
        assert_eq!(written.len(), 1);
        assert_eq!(written[0].0, "out/template.pcraft");
        let saved = photocraft_engine::smart_cmds::decode_source(&written[0].0, &written[0].1).unwrap();
        assert_eq!(photocraft_compose::flatten(&saved).px[0], [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn control_refuses_linked_smart_objects_in_commands_and_actions_but_local_editing_still_works() {
        use photocraft_doc::{LayerContent, SmartSource};
        use std::sync::Arc;

        let (mut app, ctx, smart) = smart_control_app();
        let path = std::env::temp_dir().join(format!("pc-control-smart-linked-{}.pcraft", std::process::id()));
        let st = app.session.active().unwrap();
        let LayerContent::Smart(sm) = &st.doc.layer(smart).unwrap().content else { panic!("smart fixture") };
        let SmartSource::Embedded { bytes, .. } = &sm.source else { panic!("embedded fixture") };
        std::fs::write(&path, bytes.as_slice()).unwrap();
        app.session
            .edit("Linked fixture", |doc, _| {
                let LayerContent::Smart(sm) = &mut doc.layer_mut(smart).unwrap().content else { panic!("smart fixture") };
                sm.source = SmartSource::Linked { path: path.to_string_lossy().into_owned() };
                Ok(())
            })
            .unwrap();
        app.run("layer.new.layer", json!({"name": "Unrelated"})).unwrap();
        let before = app.session.active().unwrap().doc.clone();
        let active = app.session.active().unwrap().active_layer;
        for command in ["layer.smartObjects.editContents", "layer.smartObjects.convertToLayers"] {
            let r = call(&mut app, &ctx, "engine.execute", json!({"command": command, "params": {"layer": smart.0}}));
            assert_eq!(r["ok"], false, "{r}");
            assert!(r["error"].as_str().unwrap().contains("ambient smart-object path denied"), "{r}");
            app.session.actions.list = vec![photocraft_engine::actions_cmds::Action {
                name: "Linked contents".into(),
                steps: vec![(command.into(), json!({"layer": smart.0}))],
                disabled: Default::default(),
                ..Default::default()
            }];
            let r = call(&mut app, &ctx, "engine.execute", json!({"command": "actions.play", "params": {"action": "Linked contents"}}));
            assert_eq!(r["ok"], true, "{r}");
            assert_eq!(r["result"]["ran"], 0, "{r}");
            assert_eq!(r["result"]["failed"]["id"], command, "{r}");
            assert_eq!(app.session.documents().len(), 1);
            assert!(Arc::ptr_eq(&app.session.active().unwrap().doc, &before), "a refused operation must not edit the document");
            assert_eq!(app.session.active().unwrap().active_layer, active);
            assert!(app.session.authorize.is_none(), "a refused request must restore the local session");
        }
        // A synthetic thumbnail action uses the same temporary authorization as control calls.
        app.automation_input = true;
        let error = crate::menus::invoke(&mut app, &ctx, "layer.smartObjects.editContents", json!({"layer": smart.0})).unwrap_err();
        assert!(error.contains("ambient smart-object path denied"), "{error}");
        assert!(app.session.authorize.is_none());
        app.automation_input = false;
        app.run("layer.smartObjects.editContents", json!({"layer": smart.0})).unwrap();
        assert_eq!(app.session.documents().len(), 2, "a human can still open the existing linked file");
        assert_eq!(photocraft_compose::flatten(&app.session.active().unwrap().doc).px[0], [1.0, 0.0, 0.0, 1.0]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn actions_play_checks_each_step_over_control() {
        let services = crate::Services { automation_authorize: Some(deny_ambient_file), ..Default::default() };
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
        app.session.actions.list.push(photocraft_engine::actions_cmds::Action {
            name: "Open".into(),
            steps: vec![("file.open".into(), json!({"path": "/etc/passwd"})), ("layer.new.layer".into(), json!({}))],
            disabled: Default::default(),
            ..Default::default()
        });
        let ctx = egui::Context::default();
        let r = call(&mut app, &ctx, "engine.execute", json!({"command": "actions.play", "params": {"action": "Open"}}));
        assert_eq!(r["ok"], true, "{r}");
        assert_eq!(r["result"]["ran"], 0, "{r}");
        assert_eq!(r["result"]["failed"]["id"], "file.open", "{r}");
        assert!(r["result"]["failed"]["error"].as_str().unwrap_or("").contains("disabled"), "{r}");
        assert!(app.session.documents().is_empty());
        assert!(app.session.authorize.is_none(), "the per-step gate is only installed for the request");
    }

    #[test]
    fn actions_play_checks_each_step_for_synthetic_input() {
        let services = crate::Services { automation_authorize: Some(deny_ambient_file), ..Default::default() };
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
        app.session.actions.list.push(photocraft_engine::actions_cmds::Action {
            name: "Open".into(),
            steps: vec![("file.open".into(), json!({"path": "/etc/passwd"})), ("layer.new.layer".into(), json!({}))],
            disabled: Default::default(),
            ..Default::default()
        });
        app.automation_input = true;
        let r = app.run("actions.play", json!({"action": "Open"})).unwrap();
        assert_eq!(r["ran"], 0, "{r}");
        assert_eq!(r["failed"]["id"], "file.open", "{r}");
        assert!(app.session.authorize.is_none(), "the per-step gate is only installed for the command");

        // A local play (no automation input) still runs recorded steps.
        app.automation_input = false;
        app.session.actions.list[0].steps.remove(0);
        app.session.execute("file.new", json!({"width": 4, "height": 4})).unwrap();
        let r = app.run("actions.play", json!({"action": "Open"})).unwrap();
        assert_eq!(r["ran"], 1, "{r}");
    }

    #[test]
    fn huge_click_count_is_rejected() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        // #982: a huge count used to queue a press and a release per click before replying.
        for count in [json!(MAX_CLICKS + 1), json!(1_000_000_000u64), json!(u64::MAX)] {
            let r = call(&mut app, &ctx, "ui.click", json!({"x": 10, "y": 10, "count": count}));
            assert_eq!(r["ok"], false, "{r}");
            assert!(r["error"].as_str().unwrap().contains("`count` must be at most 256"), "{r}");
            assert!(app.synthetic.is_empty(), "a rejected click queues nothing");
        }
        // Single, double and the largest allowed clicks still queue a move plus a press and a release each.
        for (count, events) in [(None, 3), (Some(2), 5), (Some(MAX_CLICKS), 2 * MAX_CLICKS as usize + 1)] {
            let params = match count {
                Some(n) => json!({"x": 10, "y": 10, "count": n}),
                None => json!({"x": 10, "y": 10}),
            };
            let (req, _rx) = ControlRequest::new("ui.click", params);
            assert!(matches!(handle(&mut app, &ctx, &req), Outcome::AfterInput));
            assert_eq!(app.synthetic.len(), events);
            app.synthetic.clear();
        }
    }

    /// #2449: the error of a rejected synthetic input call, after checking it changed nothing.
    fn rejected(app: &mut PhotocraftApp, ctx: &egui::Context, method: &str, params: Value, names: &str) -> String {
        let (tool, revision) = (app.ui.tool, app.session.active().map(|d| d.revision));
        let r = call(app, ctx, method, params.clone());
        assert_eq!(r["ok"], false, "{method} {params}: {r}");
        let error = r["error"].as_str().unwrap().to_string();
        assert!(error.contains(names), "{method} {params}: `{error}` should name `{names}`");
        assert!(app.synthetic.is_empty(), "{method} {params}: a rejected call queues nothing");
        assert_eq!(app.ui.tool, tool, "{method} {params}: a rejected call keeps the tool");
        assert_eq!(app.session.active().map(|d| d.revision), revision, "{method} {params}: a rejected call edits nothing");
        error
    }

    #[test]
    fn ui_pointer_rejects_bad_params_before_acting() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width": 64, "height": 64})).unwrap();
        let stroke = json!([{"kind": "down", "x": 5, "y": 5}, {"kind": "move", "x": 30, "y": 30}, {"kind": "up", "x": 30, "y": 30}]);
        // The same unknown tool is the same error as `ui.set`'s.
        let set = call(&mut app, &ctx, "ui.set", json!({"tool": "noSuchTool"}));
        let error = rejected(&mut app, &ctx, "ui.pointer", json!({"tool": "noSuchTool", "events": stroke}), "noSuchTool");
        assert_eq!(json!(error), set["error"]);
        let event = |extra: Value| {
            let mut e = json!({"kind": "down", "x": 5, "y": 5});
            e.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            // A valid stroke first: a later bad event must not let the earlier ones run.
            json!({"tool": "move", "events": [stroke[0], stroke[1], stroke[2], e]})
        };
        let cases = [
            // Unknown fields, top-level, under `modifiers` and in an event.
            (json!({"tol": "move", "events": stroke}), "tol"),
            (json!({"modifiers": {"shfit": true}, "events": stroke}), "shfit"),
            (event(json!({"presure": 0.5})), "presure"),
            // Wrong JSON types.
            (json!({"tool": 3, "events": stroke}), "tool"),
            (json!({"events": "down"}), "events"),
            (json!({}), "events"),
            (json!({"events": [5]}), "events[0]"),
            (event(json!({"x": "abc"})), "x"),
            (event(json!({"y": true})), "y"),
            (event(json!({"kind": 1})), "kind"),
            (json!({"shift": "yes", "events": stroke}), "shift"),
            (json!({"modifiers": {"alt": 1}, "events": stroke}), "alt"),
            (json!({"modifiers": true, "events": stroke}), "modifiers"),
            (json!({"button": 2, "events": stroke}), "button"),
            (json!([1, 2]), "params"),
            // Values outside the accepted set.
            (json!({"button": "ritgh", "events": stroke}), "button"),
            (event(json!({"kind": "dwon"})), "kind"),
            // Numbers outside their documented range.
            (event(json!({"pressure": 99})), "pressure"),
            (event(json!({"pressure": -0.1})), "pressure"),
            (event(json!({"tiltX": 91})), "tiltX"),
            (event(json!({"tiltY": -90.5})), "tiltY"),
            (event(json!({"rotation": 361})), "rotation"),
            (event(json!({"rotation": -1})), "rotation"),
        ];
        for (params, names) in cases {
            rejected(&mut app, &ctx, "ui.pointer", params, names);
        }
        let error = rejected(&mut app, &ctx, "ui.pointer", json!({"button": "ritgh", "events": stroke}), "button");
        assert!(error.contains("right") && error.contains("secondary"), "the error lists the buttons: {error}");
        let error = rejected(&mut app, &ctx, "ui.pointer", event(json!({"kind": "dwon"})), "kind");
        assert!(error.contains("down") && error.contains("move") && error.contains("up"), "the error lists the kinds: {error}");

        // Valid calls still run: an omitted kind is a move, null is an omitted field, and the
        // documented range ends are accepted.
        let before = app.session.active().unwrap().revision;
        let pen = json!({"kind": "down", "x": 5, "y": 5, "pressure": 0, "tiltX": -90, "tiltY": 90, "rotation": 360});
        let valid = json!({"tool": "brush", "button": null, "modifiers": {"shift": false}, "events": [pen, {"x": 20, "y": 20, "pressure": 1}, {"kind": "up", "x": 20, "y": 20}]});
        let r = call(&mut app, &ctx, "ui.pointer", valid);
        assert_eq!(r["ok"], true, "{r}");
        assert_eq!(app.ui.tool, Tool::Brush);
        assert!(app.session.active().unwrap().revision > before, "the stroke painted");
        for button in ["left", "middle", "right", "secondary"] {
            let r = call(
                &mut app,
                &ctx,
                "ui.pointer",
                json!({"tool": "move", "button": button, "events": [{"kind": "down", "x": 5, "y": 5}, {"kind": "up", "x": 5, "y": 5}]}),
            );
            assert_eq!(r["ok"], true, "{button}: {r}");
        }
        assert!(app.ui.layer_menu.is_some(), "a right click with Move still opens the layer menu");
    }

    #[test]
    fn ui_click_and_move_reject_bad_params_before_queueing_input() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        let cases = [
            ("ui.click", json!({"x": 10, "y": 10, "buton": "right"}), "buton"),
            ("ui.move", json!({"x": 10, "y": 10, "count": 2}), "count"),
            ("ui.click", json!({"x": "abc", "y": 10}), "x"),
            ("ui.move", json!({"x": 10, "y": [10]}), "y"),
            ("ui.click", json!({"x": 10, "y": 10, "count": "dos"}), "count"),
            ("ui.click", json!({"x": 10, "y": 10, "count": 1.5}), "count"),
            ("ui.click", json!({"x": 10, "y": 10, "count": -1}), "count"),
            ("ui.click", json!({"x": 10, "y": 10, "button": true}), "button"),
            ("ui.click", json!({"x": 10, "y": 10, "button": "ritgh"}), "button"),
            ("ui.click", json!({"x": 10, "y": 10, "count": MAX_CLICKS + 1}), "count"),
            ("ui.click", json!("click"), "params"),
        ];
        for (method, params, names) in cases {
            rejected(&mut app, &ctx, method, params, names);
        }
        let error = rejected(&mut app, &ctx, "ui.click", json!({"x": 10, "y": 10, "button": "ritgh"}), "button");
        assert!(error.contains("left") && error.contains("right") && error.contains("middle"), "the error lists the buttons: {error}");
        // Valid calls queue what they did before; an omitted button is a left click.
        for (method, params, button, events) in [
            ("ui.click", json!({"x": 10, "y": 10}), egui::PointerButton::Primary, 3),
            ("ui.click", json!({"x": 10, "y": 10, "button": "right", "count": 2}), egui::PointerButton::Secondary, 5),
            ("ui.click", json!({"x": 10, "y": 10, "button": "secondary"}), egui::PointerButton::Secondary, 3),
            ("ui.click", json!({"x": 10.5, "y": 10, "button": "middle", "count": null}), egui::PointerButton::Middle, 3),
            ("ui.move", json!({"x": 10, "y": 10}), egui::PointerButton::Primary, 1),
        ] {
            let (req, _rx) = ControlRequest::new(method, params.clone());
            assert!(matches!(handle(&mut app, &ctx, &req), Outcome::AfterInput), "{method} {params}");
            assert_eq!(app.synthetic.len(), events, "{method} {params}");
            if let Some(egui::Event::PointerButton { button: queued, .. }) = app.synthetic.get(1) {
                assert_eq!(*queued, button, "{method} {params}");
            }
            app.synthetic.clear();
        }
    }

    #[test]
    fn ui_key_rejects_bad_params_before_queueing_input() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        let cases = [
            (json!({"key": "N", "ctl": true}), "ctl"),
            (json!({"key": "N", "modifiers": {"space": true}}), "space"),
            (json!({"key": "N", "ctrl": "yes"}), "ctrl"),
            (json!({"key": "N", "modifiers": {"shift": 1}}), "shift"),
            (json!({"key": "N", "modifiers": "ctrl"}), "modifiers"),
            (json!({"key": 5}), "key"),
            (json!({}), "key"),
            (json!({"key": "NoSuchKey"}), "NoSuchKey"),
        ];
        for (params, names) in cases {
            rejected(&mut app, &ctx, "ui.key", params, names);
        }
        for params in [json!({"key": "N", "ctrl": true}), json!({"key": "N", "shift": false, "modifiers": {"ctrl": true}})] {
            let (req, _rx) = ControlRequest::new("ui.key", params.clone());
            assert!(matches!(handle(&mut app, &ctx, &req), Outcome::AfterInput), "{params}");
            assert!(matches!(app.synthetic.first(), Some(egui::Event::Key { key: egui::Key::N, modifiers, .. }) if modifiers.ctrl), "{params}");
            app.synthetic.clear();
        }
    }

    #[test]
    fn ui_screenshot_rejects_bad_params_before_capturing() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        for (params, names) in [(json!({"focus": "yes"}), "focus"), (json!({"path": 5}), "path"), (json!({"pth": "shot.png"}), "pth")] {
            rejected(&mut app, &ctx, "ui.screenshot", params, names);
        }
        for (params, path) in [(json!({}), None), (Value::Null, None), (json!({"path": "shot.png", "focus": false}), Some("shot.png"))] {
            let (req, _rx) = ControlRequest::new("ui.screenshot", params.clone());
            let Outcome::Screenshot { path: queued, .. } = handle(&mut app, &ctx, &req) else { panic!("{params}: expected a capture") };
            assert_eq!(queued.as_deref(), path, "{params}");
        }
    }

    /// A key event's modifiers, as `ui.key` queued them.
    fn key_modifiers(app: &mut PhotocraftApp, ctx: &egui::Context, params: Value) -> egui::Modifiers {
        app.synthetic.clear();
        let (req, _rx) = ControlRequest::new("ui.key", params.clone());
        assert!(matches!(handle(app, ctx, &req), Outcome::AfterInput), "{params}");
        match app.synthetic.first() {
            Some(egui::Event::Key { modifiers, .. }) => *modifiers,
            other => panic!("{params}: expected a key event, got {other:?}"),
        }
    }

    #[test]
    fn ui_key_ctrl_reaches_shortcuts_like_a_real_ctrl_press() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        let ctrl = key_modifiers(&mut app, &ctx, json!({"key":"N","ctrl":true}));
        let command = key_modifiers(&mut app, &ctx, json!({"key":"N","command":true}));
        let grouped = key_modifiers(&mut app, &ctx, json!({"key":"N","modifiers":{"ctrl":true}}));
        assert_eq!(ctrl, grouped, "a grouped flag must match the top-level one");
        if cfg!(target_os = "macos") {
            // Two different keys: Control is not ⌘, and only ⌘ matches a shortcut.
            assert!(ctrl.ctrl && !ctrl.command && !ctrl.mac_cmd, "{ctrl:?}");
            assert!(command.command && command.mac_cmd && !command.ctrl, "{command:?}");
        } else {
            // One key under two names, and shortcuts are matched on `command`, so asking for
            // either must reach them: Ctrl+N has to open New Document, not do nothing.
            assert_eq!(ctrl, command, "ctrl and command are the same key off the Mac");
            assert!(ctrl.command && ctrl.ctrl && !ctrl.mac_cmd, "{ctrl:?}");
        }
        let plain = key_modifiers(&mut app, &ctx, json!({"key":"N"}));
        assert_eq!(plain, egui::Modifiers::default(), "{plain:?}");
        let shift_alt = key_modifiers(&mut app, &ctx, json!({"key":"N","shift":true,"alt":true}));
        assert!(shift_alt.shift && shift_alt.alt && !shift_alt.command && !shift_alt.ctrl, "{shift_alt:?}");
    }
}
