//! Photoshop keyboard shortcut sets (`.kys`, Edit › Keyboard Shortcuts › Save Set) imported
//! into Edit › Keyboard Shortcuts (#1163): the dialog's Import Shortcuts… button, File › Open
//! and a drop on the window. Parsing and matching rows to commands are the engine's
//! ([`photocraft_engine::kys`], also behind `edit.keyboardShortcuts {"importKys": …}`); here
//! the candidates are the app's menus, in Photoshop's menu order, so ties go its way.
//!
//! The import fills the open dialog's overrides, so OK applies them like hand-made changes
//! (a shortcut moved to a command is taken from its old owner) and Cancel discards them.
//!
//! At first launch beside Photoshop, its live set is offered, never applied unasked: a prompt
//! names the set, and Import applies it ([`offer_import`], [`show_offer`]).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::PhotocraftApp;
pub use photocraft_engine::kys::{KysCommand, KysSet, MAX_BYTES, Plan, parse};

/// Match the rows to our commands: the menus' items first, then engine commands the menus list
/// under another label (Preferences › General… is the Settings… item).
pub fn plan(app: &PhotocraftApp, set: &KysSet) -> Plan {
    use photocraft_engine::kys::Candidate;
    let items = crate::menus::menu_items(app);
    let menu = items.iter().filter(|i| i.label != "---" && crate::menus::is_live(&i.id)).map(|i| Candidate {
        id: i.id.clone(),
        label: i.label.clone(),
        default: crate::shortcuts::default_shortcut(&i.id),
    });
    let specs = photocraft_engine::kys::command_candidates().into_iter().map(|c| Candidate { default: crate::shortcuts::default_shortcut(&c.id), ..c });
    photocraft_engine::kys::plan(menu.chain(specs), set)
}

/// Put the plan into the open Keyboard Shortcuts dialog (opening one when none is), as
/// overrides the OK button applies: a key equal to the command's default clears its override.
/// Returns the dialog id and the plan as JSON.
pub fn import_text(app: &mut PhotocraftApp, file_name: &str, text: &str) -> Result<Value, String> {
    let set = parse(text).map_err(|e| format!("{file_name}: {e}"))?;
    let plan = plan(app, &set);
    let id = match app.ui.dialogs.iter().find(|d| d.fields.get("__prefsui").and_then(Value::as_str) == Some("shortcuts")) {
        Some(d) => d.id,
        None => crate::prefs_ui::open_shortcuts(app, 0),
    };
    let f = &mut app.ui.dialog_mut(id).ok_or("the Keyboard Shortcuts dialog is gone")?.fields;
    let mut overrides: BTreeMap<String, String> = f.get("overrides").and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default();
    for (cmd, sc) in &plan.set {
        let default = crate::shortcuts::default_shortcut(cmd).and_then(|d| photocraft_engine::prefs::normalize_shortcut(&d));
        if default.as_deref() == Some(sc.as_str()) {
            overrides.remove(cmd);
        } else {
            overrides.insert(cmd.clone(), sc.clone());
        }
    }
    let shown = std::path::Path::new(file_name).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| file_name.to_string());
    let n = plan.set.len().to_string();
    let mut message = crate::i18n::fmt(tl!("Imported {n} shortcuts from {file}. Click OK to keep them."), &[("n", &n), ("file", &shown)]);
    if !plan.unknown.is_empty() {
        let u = plan.unknown.len().to_string();
        message.push(' ');
        message.push_str(&crate::i18n::fmt(tl!("{n} commands are not in PhotoCraft."), &[("n", &u)]));
    }
    f.insert("tab".into(), json!(0));
    f.insert("overrides".into(), json!(overrides));
    f.insert("selected".into(), json!(""));
    f.insert("capture".into(), json!(false));
    f.insert("message".into(), json!(message));
    f.insert("messageWarning".into(), json!(false));
    Ok(json!({
        "dialog": id,
        "name": set.name,
        "imported": plan.set.len(),
        "set": plan.set,
        "unknown": plan.unknown,
        "unreadable": plan.unreadable,
        "alternates": plan.alternates,
        "toolKeys": set.tool_keys,
    }))
}

/// A `.kys` opened like a document (File › Open, a drop): its bytes are text.
pub fn open_bytes(app: &mut PhotocraftApp, name: &str, bytes: &[u8]) -> Result<Value, String> {
    if bytes.len() > MAX_BYTES {
        return Err(format!("{name}: not a keyboard shortcut set (.kys): larger than 4 MiB"));
    }
    let text = std::str::from_utf8(bytes).map_err(|_| format!("{name}: not a keyboard shortcut set (.kys): not UTF-8"))?;
    import_text(app, name, text)
}

/// The dialog's Import Shortcuts… button: ask for the file, then fill the dialog.
pub fn import_dialog(app: &mut PhotocraftApp) -> Result<Value, String> {
    app.pick_file_bytes(|app, name, bytes| open_bytes(app, &name, &bytes))
}

/// Preferences › `dialogs` key that records the first-launch offer (its value names the source
/// and whether it was imported), so it is made once.
pub const IMPORTED_PREF: &str = "photoshopShortcuts";

/// A set found at first launch, waiting for the user's answer ([`show_offer`]).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Offer {
    /// Where the set was found (Photoshop's live `Keyboard Shortcuts.psp`).
    pub source: String,
    /// The set's name, else its file name.
    pub set: String,
    /// Shortcuts the import would change.
    pub changes: usize,
    #[serde(skip)]
    text: String,
}

fn remember(app: &mut PhotocraftApp, source: &str, imported: bool) {
    app.session.prefs.edit(|p| {
        p.dialogs.insert(IMPORTED_PREF.into(), json!({"source": source, "imported": imported}));
    });
}

fn not_imported(app: &mut PhotocraftApp, e: &str) {
    app.ui.status = crate::i18n::fmt(tl!("The keyboard shortcut set was not imported: {error}"), &[("error", e)]);
    app.ui.status_error = true;
}

/// First launch on a machine with Photoshop: its live set ([`crate::Services::photoshop_shortcuts`])
/// is offered once. Someone who already changed a shortcut keeps their own set and is not asked;
/// a set that changes nothing (Photoshop's defaults are ours too) is not offered either. The
/// preference remembers each outcome, so a later launch never asks again.
pub fn offer_import(app: &mut PhotocraftApp) {
    if app.session.prefs().dialogs.contains_key(IMPORTED_PREF) {
        return;
    }
    let Some((source, text)) = app.services.photoshop_shortcuts.as_mut().and_then(|f| f()) else { return };
    if !app.session.prefs().shortcuts.is_empty() {
        return remember(app, &source, false);
    }
    let set = match parse(&text) {
        Ok(set) => set,
        Err(e) => {
            remember(app, &source, false);
            return not_imported(app, &e);
        }
    };
    let plan = plan(app, &set);
    let default = |id: &str| crate::shortcuts::default_shortcut(id).and_then(|d| photocraft_engine::prefs::normalize_shortcut(&d));
    let changes = plan.set.iter().filter(|(id, sc)| default(id).as_deref() != Some(sc.as_str())).count();
    if changes == 0 {
        return remember(app, &source, false);
    }
    let name = set.name.unwrap_or_else(|| std::path::Path::new(&source).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default());
    app.ui.kys_offer = Some(Offer { source, set: name, changes, text });
}

/// The offer's answer: Import applies the set and says so; either answer is remembered.
pub fn answer_offer(app: &mut PhotocraftApp, import: bool) {
    let Some(offer) = app.ui.kys_offer.take() else { return };
    remember(app, &offer.source, import);
    if !import {
        return;
    }
    let applied = import_text(app, &offer.source, &offer.text).and_then(|r| {
        let id = r["dialog"].as_u64().ok_or("no dialog")?;
        crate::dialogs::confirm(app, id)
    });
    if let Err(e) = applied {
        return not_imported(app, &e);
    }
    let changed = app.session.prefs().shortcuts.len();
    let id = crate::notices::post(
        app,
        "Keyboard shortcuts imported",
        vec!["Your keyboard shortcut set {set} is in use here: {n} shortcuts differ from the defaults. Edit › Keyboard Shortcuts shows them; Reset All to Defaults undoes this.".to_owned()],
        false,
        None,
    );
    if let Some(n) = app.ui.notices.iter_mut().find(|n| n.id == id) {
        n.args = vec![("set".to_owned(), offer.set), ("n".to_owned(), changed.to_string())];
    }
}

/// The first-launch prompt: which set was found, and Import / Don't Import.
pub fn show_offer(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let Some(offer) = &app.ui.kys_offer else { return };
    let n = offer.changes.to_string();
    let line = crate::i18n::fmt(
        tl!(
            "PhotoCraft found your keyboard shortcut set {set} from another image editor on this computer: {n} of its shortcuts differ from PhotoCraft's. Import it?"
        ),
        &[("set", &offer.set), ("n", &n)],
    );
    let mut answer = None;
    egui::Window::new(tl!("Import keyboard shortcuts?"))
        .id(egui::Id::new("kys-offer"))
        .collapsible(false)
        .resizable(false)
        .default_width(440.0)
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .show(ctx, |ui| {
            ui.label(line);
            ui.label(tl!("Edit › Keyboard Shortcuts › Reset All to Defaults undoes an import."));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                use crate::widgets::{ButtonRole, DialogButton};
                let buttons =
                    [DialogButton::new(ButtonRole::Default, tl!("Import"), 100.0), DialogButton::new(ButtonRole::Alternate, tl!("Don't Import"), 120.0)];
                if let Some(role) = crate::widgets::dialog_buttons(ui, &buttons) {
                    answer = Some(role == ButtonRole::Default);
                }
            });
        });
    if let Some(import) = answer {
        answer_offer(app, import);
    }
}

#[cfg(test)]
#[path = "kys_import_tests.rs"]
mod tests;
