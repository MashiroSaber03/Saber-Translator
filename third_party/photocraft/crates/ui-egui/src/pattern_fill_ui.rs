//! Photoshop's Pattern Fill dialog (#2908): Layer › New Fill Layer › Pattern… and double-clicking a
//! Pattern Fill layer's thumbnail open it. Pattern (swatch picker), Snap to Origin, Angle, Scale
//! and Link with Layer, previewed live on the canvas from a document snapshot. OK runs one engine
//! command (`layer.newFillLayer.pattern` for a new layer, `layer.fill.pattern.set` for an existing
//! one), so it is one History step; Cancel changes nothing. The fields are the command's params,
//! so `ui.dialog.set` / `ui.dialog.confirm` drive it like any dialog.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use photocraft_doc::{DocId, Document, Fill, LayerContent};
use photocraft_engine::pattern_fill_cmds as cmds;
use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::theme::Tokens;

const MARK: &str = "__patternFill";
/// The fields that are command params (besides the target in `__params`).
const KEYS: [&str; 5] = ["pattern", "scale", "angle", "link", "phase"];

pub(crate) struct Preview {
    document: DocId,
    revision: u64,
    dialog: u64,
    params: Value,
    shown: Arc<Document>,
    key: u64,
}

/// Is this dialog the Pattern Fill dialog?
pub(crate) fn owns(f: &Map<String, Value>) -> bool {
    f.contains_key(MARK)
}

fn fields(command: &str, target: Value, values: Value) -> Map<String, Value> {
    let mut f = Map::new();
    f.insert(MARK.into(), json!(true));
    f.insert("__command".into(), json!(command));
    f.insert("__label".into(), json!("Pattern Fill"));
    f.insert("__params".into(), target);
    for k in KEYS {
        if let Some(v) = values.get(k) {
            f.insert(k.into(), v.clone());
        }
    }
    f.insert("__orig".into(), values);
    f
}

/// Edit the active Pattern Fill layer (its thumbnail, Layer › Layer Content Options…).
pub(crate) fn open(app: &mut PhotocraftApp) -> Option<u64> {
    let st = app.session.active()?;
    let layer = st.active_layer?;
    let LayerContent::Fill(Fill::Pattern { scale, id, angle, link, phase, .. }) = &st.doc.layer(layer)?.content else { return None };
    let values = json!({"pattern": id, "scale": scale * 100.0, "angle": angle, "link": link, "phase": [phase.0, phase.1]});
    let f = fields(cmds::SET, json!({"layer": layer.0, "document": st.doc.id.0}), values);
    Some(app.ui.open_dialog(crate::state::DialogKind::Command, f))
}

/// Layer › New Fill Layer › Pattern…: the dialog in front of creating the layer, starting from
/// the Patterns panel's pattern at 100 %, 0° and linked, as in Photoshop.
pub(crate) fn open_new(app: &mut PhotocraftApp) -> Result<u64, String> {
    if let Some(Err(why)) = photocraft_engine::commands::find(cmds::CREATE).map(|c| (c.enabled)(&app.session)) {
        return Err(why);
    }
    let st = app.session.active().ok_or("no document open")?;
    let pattern = photocraft_engine::pattern_cmds::resolve(&app.session, "").ok_or("no patterns")?;
    let values = json!({"pattern": pattern.id, "scale": 100.0, "angle": 0.0, "link": true, "phase": [0.0, 0.0]});
    let f = fields(cmds::CREATE, json!({"document": st.doc.id.0}), values);
    Ok(app.ui.open_dialog(crate::state::DialogKind::Command, f))
}

/// The command and its params: the target plus the dialog's values.
fn command(app: &PhotocraftApp, f: &Map<String, Value>) -> Option<(&'static str, Value)> {
    let cmd = match f.get("__command").and_then(Value::as_str)? {
        cmds::SET => cmds::SET,
        cmds::CREATE => cmds::CREATE,
        _ => return None,
    };
    let mut params = f.get("__params").filter(|p| p.is_object()).cloned()?;
    for k in KEYS {
        if let Some(v) = f.get(k) {
            params[k] = v.clone();
        }
    }
    // A new fill layer is masked by the path selected in the Paths panel, as when it runs.
    let params = if cmd == cmds::CREATE {
        params.as_object_mut()?.remove("document");
        crate::vector_ui::with_active_path(app, cmd, params)
    } else {
        params
    };
    Some((cmd, params))
}

/// An edit dialog whose values are still the layer's.
fn unchanged(f: &Map<String, Value>) -> bool {
    f.get("__command").and_then(Value::as_str) == Some(cmds::SET) && KEYS.iter().all(|k| f.get(*k) == f.get("__orig").and_then(|o| o.get(*k)))
}

/// The document the dialog targets, if it is still the active one.
fn target_is_active(app: &PhotocraftApp, f: &Map<String, Value>) -> bool {
    let target = f.get("__params").and_then(|p| p.get("document")).and_then(Value::as_u64);
    app.session.active().is_some_and(|st| target == Some(st.doc.id.0))
}

/// OK: one engine command. A stale document never receives the edit.
pub(crate) fn confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    if !target_is_active(app, f) {
        return Err("the target document is no longer active".into());
    }
    if unchanged(f) {
        return Ok(json!({"unchanged": true}));
    }
    let (cmd, params) = command(app, f).ok_or("invalid Pattern Fill dialog")?;
    app.run(cmd, params)
}

/// The canvas preview while the dialog is open: the document as OK would leave it.
pub(crate) fn display_doc(app: &mut PhotocraftApp, idx: usize) -> Option<(Arc<Document>, u64)> {
    let dialog = app.ui.dialogs.iter().rev().find(|d| owns(&d.fields));
    let Some(d) = dialog else {
        app.pattern_fill_preview = None;
        return None;
    };
    let st = app.session.documents().get(idx)?;
    if !target_is_active(app, &d.fields) || app.session.active_index() != Some(idx) || unchanged(&d.fields) {
        return None;
    }
    let (cmd, params) = command(app, &d.fields)?;
    if let Some(p) = &app.pattern_fill_preview
        && p.document == st.doc.id
        && p.revision == st.revision
        && p.dialog == d.id
        && p.params == params
    {
        return Some((p.shown.clone(), p.key));
    }
    // Out-of-range values typed by automation show the committed document; OK reports them.
    let doc = cmds::preview(&app.session, cmd, &params).ok()?;
    // Include the dialog identity: cancelling and reopening at the same document revision must
    // not reuse an old preview's pixels in the canvas caches.
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    (st.doc.id.0, st.revision, d.id, params.to_string()).hash(&mut hash);
    let key = (1 << 47) | (hash.finish() & ((1 << 36) - 1));
    let (document, revision, dialog) = (st.doc.id, st.revision, d.id);
    let shown = Arc::new(photocraft_engine::mode_cmds::display_document(&doc).unwrap_or(doc));
    app.pattern_fill_preview = Some(Preview { document, revision, dialog, params, shown: shown.clone(), key });
    Some((shown, key))
}

fn num(f: &Map<String, Value>, k: &str, d: f64) -> f32 {
    f.get(k).and_then(Value::as_f64).unwrap_or(d) as f32
}

/// The dialog body, in Photoshop's layout: the pattern with Snap to Origin beside it, then
/// Angle, Scale and Link with Layer.
pub(crate) fn body(app: &PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    ui.spacing_mut().item_spacing.y = 6.0;
    ui.horizontal(|ui| {
        let pat = f.get("pattern").and_then(Value::as_str).unwrap_or_default().to_string();
        if app.session.patterns.items.is_empty() {
            ui.label(egui::RichText::new(tl!("No patterns")).color(Tokens::get(ui.ctx()).text_faint));
        } else if let Some(id) = crate::preset_panels::pattern_picker(app, ui, &pat) {
            f.insert("pattern".into(), json!(id));
        }
        ui.add_space(8.0);
        // Photoshop: Snap to Origin puts the pattern's origin back at the layer's (or canvas's).
        if crate::widgets::secondary_button(ui, tl!("Snap to Origin"), 110.0).clicked() {
            f.insert("phase".into(), json!([0.0, 0.0]));
        }
    });
    let mut angle = num(f, "angle", 0.0);
    if crate::widgets::angle_dial_row(ui, tl!("Angle:"), &mut angle, -180.0..=180.0).changed() {
        f.insert("angle".into(), json!(angle.clamp(-180.0, 180.0)));
    }
    let (min, max) = cmds::SCALE_RANGE;
    let mut scale = num(f, "scale", 100.0);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(tl!("Scale:")).color(Tokens::get(ui.ctx()).text_dim));
        if crate::widgets::value_field(ui, &mut scale, min as f32..=max as f32, "%", 70.0).changed() {
            f.insert("scale".into(), json!(scale.clamp(min as f32, max as f32)));
        }
    });
    let mut link = f.get("link").and_then(Value::as_bool).unwrap_or(true);
    if crate::widgets::checkbox(ui, &mut link, tl!("Link with Layer")).changed() {
        f.insert("link".into(), json!(link));
    }
}

#[cfg(test)]
#[path = "pattern_fill_ui_tests.rs"]
mod tests;
