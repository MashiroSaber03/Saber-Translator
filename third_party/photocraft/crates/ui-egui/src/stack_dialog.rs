//! File › Scripts › Load Files into Stack… and Statistics: shared Source Files controls
//! with Use (Files or Folder), the list of what will be loaded, Browse…, Remove, Add Open Files and
//! Sort By Name, then Attempt to Automatically Align Source Images and Create Smart Object after
//! Loading Layers. Statistics adds Stack Mode and always creates a smart object. OK runs the
//! corresponding engine command with the list or the folder.
use egui::{RichText, vec2};
use serde_json::{Map, Value, json};

use crate::file_dialog::{FileDialogAnswer, FileDialogRequest};
use crate::{PhotocraftApp, state::DialogKind, theme::Tokens, widgets};

const MARK: &str = "__loadStack";
pub const COMMAND: &str = "file.scripts.loadFilesIntoStack";
const STATISTICS: &str = "file.scripts.statistics";

pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.contains_key(MARK)
}

/// Open the dialog with an empty Files list. Returns its id.
pub fn open(app: &mut PhotocraftApp) -> u64 {
    open_for(app, COMMAND, "Load Files into Stack…")
}

pub fn open_statistics(app: &mut PhotocraftApp) -> u64 {
    open_for(app, STATISTICS, "Image Statistics")
}

fn statistics(f: &Map<String, Value>) -> bool {
    f.get("__command").and_then(Value::as_str) == Some(STATISTICS)
}

fn open_for(app: &mut PhotocraftApp, command: &str, label: &str) -> u64 {
    let mut f = Map::new();
    f.insert(MARK.into(), json!(true));
    f.insert("__command".into(), json!(command));
    f.insert("__label".into(), json!(label));
    f.insert("use".into(), json!("files"));
    f.insert("files".into(), json!([]));
    f.insert("folder".into(), json!(""));
    f.insert("align".into(), json!(false));
    f.insert("smartObject".into(), json!(false));
    if command == STATISTICS {
        f.insert("mode".into(), json!("median"));
    }
    app.ui.open_dialog(DialogKind::Command, f)
}

fn folder_mode(f: &Map<String, Value>) -> bool {
    f.get("use").and_then(Value::as_str) == Some("folder")
}

fn files(f: &Map<String, Value>) -> Vec<String> {
    f.get("files").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default()
}

fn folder(f: &Map<String, Value>) -> &str {
    f.get("folder").and_then(Value::as_str).unwrap_or("")
}

fn selected(f: &Map<String, Value>) -> Option<usize> {
    f.get("selected").and_then(Value::as_u64).map(|i| i as usize)
}

/// Add `paths` to the Files list, skipping those already in it, in their order.
fn add(f: &mut Map<String, Value>, paths: impl IntoIterator<Item = String>) {
    let mut list = files(f);
    for p in paths {
        if !list.contains(&p) {
            list.push(p);
        }
    }
    f.insert("files".into(), json!(list));
}

fn file_name(p: &str) -> &str {
    std::path::Path::new(p).file_name().and_then(|n| n.to_str()).unwrap_or(p)
}

/// The open documents that have a file, in tab order (what Add Open Files adds).
fn open_files(app: &PhotocraftApp) -> Vec<String> {
    app.session.documents().iter().filter_map(|d| d.path.clone()).collect()
}

/// OK needs files to load: some in the list, or a chosen folder.
pub fn can_confirm(f: &Map<String, Value>) -> bool {
    !owns(f) || if folder_mode(f) { !folder(f).is_empty() } else { !files(f).is_empty() }
}

/// Browse…: files to add (Files) or the folder to load (Folder), into dialog `id` once the file
/// dialog answers.
fn browse(app: &mut PhotocraftApp, id: u64, folder: bool) {
    let r = if folder {
        app.ask_folder(move |app, path| {
            let d = app.ui.dialog_mut(id).ok_or("the dialog was closed")?;
            d.fields.insert("folder".into(), json!(path));
            Ok(Value::Null)
        })
    } else {
        let initial_dir = crate::file_dialog::last_used_dir(&app.ui.recent_files);
        app.ask_file(FileDialogRequest::Open { multiple: true, initial_dir, extensions: None }, move |app, answer| {
            let FileDialogAnswer::Paths(paths) = answer else { return Err("Load Files into Stack reads files from disk".into()) };
            let d = app.ui.dialog_mut(id).ok_or("the dialog was closed")?;
            add(&mut d.fields, paths);
            Ok(Value::Null)
        })
    };
    if let Err(e) = r.map(|_| ()).or_else(|e| if e == crate::file_dialog::CANCELLED { Ok(()) } else { Err(e) }) {
        app.ui.status = e;
    }
}

/// The dialog's body (dialog `id`); OK and Cancel are the standard ones below it.
pub fn body(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>, id: u64) {
    let t = Tokens::get(ui.ctx());
    if statistics(f) {
        ui.horizontal(|ui| {
            ui.label(tl!("Choose Stack Mode:"));
            let mut mode = f.get("mode").and_then(Value::as_str).unwrap_or("median").to_string();
            let options: Vec<_> = photocraft_doc::StackMode::ALL.iter().map(|m| (m.id().to_string(), m.label())).collect();
            if widgets::dropdown(ui, "statistics-stack-mode", &mut mode, &options, 180.0) {
                f.insert("mode".into(), json!(mode));
            }
        });
        ui.add_space(8.0);
    }
    ui.label(RichText::new(tl!("Source Files")).font(crate::theme::semibold(13.0)));
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label(tl!("Use:"));
        let mut mode = usize::from(folder_mode(f));
        let options = [(0, tl!("Files")), (1, tl!("Folder"))];
        if widgets::dropdown(ui, "load-stack-use", &mut mode, &options, 120.0) {
            f.insert("use".into(), json!(if mode == 1 { "folder" } else { "files" }));
            f.remove("selected");
        }
    });
    ui.add_space(6.0);
    let folder_mode = folder_mode(f);
    let mut browse_now = false;
    ui.horizontal_top(|ui| {
        // The list: the chosen files, or the images in the chosen folder.
        let shown: Vec<String> = if folder_mode {
            if folder(f).is_empty() { Vec::new() } else { photocraft_engine::file_cmds::list_images(folder(f)).unwrap_or_default() }
        } else {
            files(f)
        };
        let sel = selected(f);
        egui::Frame::NONE.fill(t.field).stroke(egui::Stroke::new(1.0, t.field_border)).corner_radius(t.radius_sm).inner_margin(4.0).show(ui, |ui| {
            ui.set_min_size(vec2(300.0, 160.0));
            ui.set_max_width(300.0);
            egui::ScrollArea::vertical().id_salt("load-stack-files").max_height(160.0).auto_shrink([false, false]).show(ui, |ui| {
                ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                    if folder_mode && !folder(f).is_empty() {
                        ui.label(RichText::new(folder(f)).color(t.text_dim));
                    }
                    for (i, p) in shown.iter().enumerate() {
                        let row = ui.add(egui::Button::selectable(!folder_mode && sel == Some(i), file_name(p))).on_hover_text(p.as_str());
                        if row.clicked() && !folder_mode {
                            f.insert("selected".into(), json!(i));
                        }
                    }
                });
            });
        });
        ui.add_space(8.0);
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = 4.0;
            if widgets::secondary_button(ui, tl!("Browse…"), 120.0).clicked() {
                browse_now = true;
            }
            let can_remove = !folder_mode && sel.is_some_and(|i| i < files(f).len());
            if ui.add_enabled_ui(can_remove, |ui| widgets::secondary_button(ui, tl!("Remove"), 120.0)).inner.clicked()
                && let Some(i) = sel
            {
                let mut list = files(f);
                list.remove(i);
                f.insert("files".into(), json!(list));
                f.remove("selected");
            }
            let open = open_files(app);
            if ui.add_enabled_ui(!folder_mode && !open.is_empty(), |ui| widgets::secondary_button(ui, tl!("Add Open Files"), 120.0)).inner.clicked() {
                add(f, open);
            }
            if ui.add_enabled_ui(!folder_mode && files(f).len() > 1, |ui| widgets::secondary_button(ui, tl!("Sort By Name"), 120.0)).inner.clicked() {
                let mut list = files(f);
                list.sort_by_key(|p| file_name(p).to_lowercase());
                f.insert("files".into(), json!(list));
                f.remove("selected");
            }
        });
    });
    ui.add_space(8.0);
    for (key, label) in [("align", tl!("Attempt to Automatically Align Source Images")), ("smartObject", tl!("Create Smart Object after Loading Layers"))] {
        if statistics(f) && key == "smartObject" {
            continue;
        }
        let mut on = f.get(key).and_then(Value::as_bool).unwrap_or(false);
        if widgets::checkbox(ui, &mut on, label).changed() {
            f.insert(key.into(), json!(on));
        }
    }
    if browse_now {
        // The file dialog's answer lands in the dialog's own fields: save this frame's edits first.
        if let Some(d) = app.ui.dialog_mut(id) {
            d.fields = f.clone();
        }
        browse(app, id, folder_mode);
    }
}

/// OK: load the list (or the folder) into a new document.
pub fn confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    let paths = if folder_mode(f) {
        let dir = folder(f);
        if dir.is_empty() {
            return Err("Choose a folder to load".into());
        }
        json!(dir)
    } else {
        let list = files(f);
        if list.is_empty() {
            return Err("Add the files to load".into());
        }
        json!(list)
    };
    let align = f.get("align").and_then(Value::as_bool).unwrap_or(false);
    let smart = f.get("smartObject").and_then(Value::as_bool).unwrap_or(false);
    let r = if statistics(f) {
        let mode = f.get("mode").and_then(Value::as_str).unwrap_or("median");
        app.run(STATISTICS, json!({"paths": paths, "align": align, "mode": mode}))
    } else {
        app.run(COMMAND, json!({"paths": paths, "align": align, "createSmartObject": smart}))
    };
    if r.is_ok() {
        app.sync_views();
    }
    r
}

#[cfg(test)]
#[path = "stack_dialog_tests.rs"]
mod tests;
