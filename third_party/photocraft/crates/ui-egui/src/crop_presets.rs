//! The Crop tool's saved presets (#1919): New Crop Preset… and Delete Crop Preset… at the end of
//! the options bar's preset dropdown, as in Photoshop.
//!
//! A crop preset is a tool preset of the `"crop"` tool (`tool.presets.*`, persisted with the
//! preferences like every tool preset), the way Photoshop keeps them: they show in this dropdown
//! and in Window › Tool Presets alike. A preset's options hold `{"toolOptions": {crop_ratio, …}}`:
//! W x H x Resolution saves `crop_ratio` = "whr", W, H, the resolution and its unit; a ratio saves
//! its `crop_ratio` ("2:3").
//!
//! - New Crop Preset… opens "New Crop Preset" with a Name (made from the values, "4 x 5 in 300 ppi"
//!   or "2 : 3"); OK saves it (`tool.presets.new`). It is greyed out when there is nothing to save
//!   (W or H empty, no ratio).
//! - Delete Crop Preset… lists the saved crop presets; OK deletes the selected ones
//!   (`tool.presets.edit` delete). Photoshop's built-in size presets ([`crop_size::PRESETS`]) are
//!   not tool presets here and can't be deleted. Greyed out with no saved crop preset.
//! - The dropdown shows the name of the preset whose values the fields hold, so a preset just saved
//!   or chosen reads as selected, and the mode again once the fields change or it is deleted.

use egui::vec2;
use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::crop_size::{self, PX_PER_CM, PX_PER_IN, WHR};
use crate::state::{DialogKind, Tool, ToolOptions};

/// Dialog field marking New Crop Preset.
const NEW_MARK: &str = "__cropPresetNew";
/// Dialog field marking Delete Crop Preset.
const DELETE_MARK: &str = "__cropPresetDelete";
/// The tool name crop presets are saved under.
pub const TOOL: &str = "crop";
/// The longest preset name (the engine's limit for tool preset names).
pub const MAX_NAME: usize = 255;
/// The `ToolOptions` fields a crop preset saves.
const KEYS: [&str; 5] = ["crop_ratio", "crop_width", "crop_height", "crop_resolution", "crop_resolution_unit"];

/// One entry of the preset dropdown.
#[derive(Clone, Debug, PartialEq)]
enum Choice {
    /// A built-in key (`crop_ratio` value, size preset or Front Image), see [`crop_size::chosen`].
    Builtin(String),
    /// A saved crop preset, by name.
    User(String),
    New,
    Delete,
}

/// The saved crop presets: (name, crop fields of its `toolOptions`), in saved order.
pub fn user_presets(app: &PhotocraftApp) -> Vec<(String, Map<String, Value>)> {
    app.session
        .presets
        .tool_presets
        .iter()
        .filter(|p| Tool::from_name(&p.tool) == Some(Tool::Crop))
        .map(|p| (p.name.clone(), crop_fields(p.options.get("toolOptions").unwrap_or(&Value::Null))))
        .collect()
}

/// The crop fields of a `toolOptions` object (string values only; anything else is dropped).
fn crop_fields(v: &Value) -> Map<String, Value> {
    KEYS.iter().filter_map(|k| v.get(*k).filter(|x| x.is_string()).map(|x| (k.to_string(), x.clone()))).collect()
}

/// The fields of `o` a preset compares against.
fn current_fields(o: &ToolOptions) -> Map<String, Value> {
    let v = json!({"crop_ratio": o.crop_ratio, "crop_width": o.crop_width, "crop_height": o.crop_height,
        "crop_resolution": o.crop_resolution, "crop_resolution_unit": o.crop_resolution_unit});
    v.as_object().cloned().unwrap_or_default()
}

/// Whether the fields hold a preset's values (each crop field it saved, `crop_ratio` required).
fn matches(o: &ToolOptions, fields: &Map<String, Value>) -> bool {
    let cur = current_fields(o);
    fields.contains_key("crop_ratio") && fields.iter().all(|(k, v)| cur.get(k) == Some(v))
}

/// The number of a length or resolution as the preset name shows it ("8.5", "300").
fn number(v: f64, unit: photocraft_engine::prefs::Unit) -> String {
    let s = crop_size::format_length(v, unit);
    s.split_once(' ').map_or(s.clone(), |(n, _)| n.to_string())
}

/// What New Crop Preset would save from `o`: the crop fields and the name it suggests. `None` when
/// there is nothing to save (W x H x Resolution without both W and H, or no numeric ratio).
pub fn savable(o: &ToolOptions) -> Option<(Map<String, Value>, String)> {
    use photocraft_engine::prefs::Unit;
    if o.crop_ratio == WHR {
        let (w, wu) = crop_size::parse_length(&o.crop_width, Unit::Pixels)?;
        let (h, hu) = crop_size::parse_length(&o.crop_height, Unit::Pixels)?;
        let mut name = if wu == hu {
            format!("{} x {} {}", number(w, wu), number(h, hu), wu.suffix())
        } else {
            format!("{} x {}", crop_size::format_length(w, wu), crop_size::format_length(h, hu))
        };
        if let Some(r) = crate::widgets::parse_num(&o.crop_resolution.trim().replace(',', ".")).filter(|r| r.is_finite() && *r > 0.0) {
            let unit = if o.crop_resolution_unit == PX_PER_CM { "px/cm" } else { "ppi" };
            name = format!("{name} {} {unit}", crate::widgets::fmt_num2(r));
        }
        let mut fields = current_fields(o);
        if fields.get("crop_resolution_unit").and_then(Value::as_str).is_none_or(|u| u != PX_PER_CM) {
            fields.insert("crop_resolution_unit".into(), json!(PX_PER_IN));
        }
        return Some((fields, name));
    }
    if o.crop_ratio == "original" {
        return None;
    }
    let (a, b) = crate::chrome_ui::crop_ratio(&o.crop_ratio, 0.0, 0.0)?;
    if !(a.is_finite() && b.is_finite() && a > 0.0 && b > 0.0) {
        return None;
    }
    let name = format!("{} : {}", crate::widgets::fmt_num(a), crate::widgets::fmt_num(b));
    let mut fields = Map::new();
    fields.insert("crop_ratio".into(), json!(o.crop_ratio));
    Some((fields, name))
}

/// The dropdown's button text: the saved preset (latest first) or built-in size preset the fields
/// hold, else the mode or ratio entry, else "Ratio" for a typed ratio.
fn button_label(o: &ToolOptions, builtin: &[(String, &'static str)], users: &[(String, Map<String, Value>)]) -> String {
    if let Some((name, _)) = users.iter().rev().find(|(_, f)| matches(o, f)) {
        return name.clone();
    }
    if o.crop_ratio == WHR
        && let Some((_, label, ..)) = crop_size::PRESETS
            .iter()
            .find(|(_, _, w, h, r)| o.crop_width == *w && o.crop_height == *h && o.crop_resolution == *r && o.crop_resolution_unit == PX_PER_IN)
    {
        return tl!(label).to_string();
    }
    let label = builtin.iter().find(|(k, _)| *k == o.crop_ratio).or_else(|| builtin.first()).map_or("", |(_, l)| *l);
    tl!(label).to_string()
}

/// The options bar's preset dropdown: the built-in entries ([`crop_size::dropdown_options`]), the
/// saved crop presets before Front Image, then New Crop Preset… and Delete Crop Preset….
pub fn dropdown(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let builtin = crop_size::dropdown_options();
    let users = user_presets(app);
    let o = &app.ui.tool_options;
    let label = button_label(o, &builtin, &users);
    let current_user = users.iter().rev().find(|(_, f)| matches(o, f)).map(|(n, _)| n.clone());
    let can_new = savable(o).is_some();
    let mut chosen: Option<Choice> = None;
    egui::ComboBox::from_id_salt("crop-ratio").selected_text(label).width(140.0).height(420.0).icon(crate::widgets::chevron_icon).show_ui(ui, |ui| {
        let (before, after): (Vec<_>, Vec<_>) = builtin.iter().partition(|(k, _)| k != crop_size::FRONT);
        for (k, l) in &before {
            if ui.selectable_label(current_user.is_none() && *k == o.crop_ratio, tl!(l)).clicked() {
                chosen = Some(Choice::Builtin(k.clone()));
            }
        }
        for (name, _) in &users {
            if ui.selectable_label(current_user.as_deref() == Some(name), name.as_str()).clicked() {
                chosen = Some(Choice::User(name.clone()));
            }
        }
        for (k, l) in &after {
            if ui.selectable_label(false, tl!(l)).clicked() {
                chosen = Some(Choice::Builtin(k.clone()));
            }
        }
        ui.separator();
        if ui.add_enabled(can_new, egui::Button::selectable(false, tl!("New Crop Preset…"))).clicked() {
            chosen = Some(Choice::New);
        }
        if ui.add_enabled(!users.is_empty(), egui::Button::selectable(false, tl!("Delete Crop Preset…"))).clicked() {
            chosen = Some(Choice::Delete);
        }
    });
    match chosen {
        Some(Choice::Builtin(k)) => {
            app.ui.tool_options.crop_ratio = k;
            // A size preset or Front Image fills W x H x Resolution (#2443).
            crop_size::chosen(app);
            sync_custom_ratio_fields(app);
        }
        Some(Choice::User(name)) => {
            select(app, &name);
            sync_custom_ratio_fields(app);
        }
        Some(Choice::New) => {
            open_new(app);
        }
        Some(Choice::Delete) => {
            open_delete(app);
        }
        None => {}
    }
}

/// A newly chosen ratio fills the editable custom W/H fields (free Ratio clears them, #3188).
fn sync_custom_ratio_fields(app: &mut PhotocraftApp) {
    if app.ui.tool_options.crop_ratio != crop_size::WHR {
        crop_size::select_custom_ratio(&mut app.ui.tool_options);
    }
}

/// Choose a saved crop preset: its fields fill the options bar (`tool.presets.select`, the same
/// as clicking it in the Tool Presets panel).
pub fn select(app: &mut PhotocraftApp, name: &str) {
    crate::preset_panels::select_tool_preset(app, name);
}

/// Open New Crop Preset with the name made from the current values. `None` (nothing opens) when
/// there is nothing to save; an open one is reused.
pub fn open_new(app: &mut PhotocraftApp) -> Option<u64> {
    let (fields, name) = savable(&app.ui.tool_options)?;
    if let Some(d) = app.ui.dialogs.iter().find(|d| d.fields.contains_key(NEW_MARK)) {
        return Some(d.id);
    }
    let f = json!({NEW_MARK: true, "__label": "New Crop Preset", "name": name, "options": fields});
    Some(app.ui.open_dialog(DialogKind::Command, f.as_object().cloned().unwrap_or_default()))
}

/// Open Delete Crop Preset. `None` when no crop preset is saved; an open one is reused.
pub fn open_delete(app: &mut PhotocraftApp) -> Option<u64> {
    if user_presets(app).is_empty() {
        return None;
    }
    if let Some(d) = app.ui.dialogs.iter().find(|d| d.fields.contains_key(DELETE_MARK)) {
        return Some(d.id);
    }
    let f = json!({DELETE_MARK: true, "__label": "Delete Crop Preset", "selected": []});
    Some(app.ui.open_dialog(DialogKind::Command, f.as_object().cloned().unwrap_or_default()))
}

pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.contains_key(NEW_MARK) || fields.contains_key(DELETE_MARK)
}

/// The trimmed name a New Crop Preset dialog would save, if it is one (1–[`MAX_NAME`] characters).
fn new_name(fields: &Map<String, Value>) -> Option<String> {
    let n = fields.get("name").and_then(Value::as_str)?.trim();
    (!n.is_empty() && n.chars().count() <= MAX_NAME).then(|| n.to_string())
}

/// The names a Delete Crop Preset dialog selected that are saved crop presets.
fn delete_names(app: &PhotocraftApp, fields: &Map<String, Value>) -> Vec<String> {
    let saved = user_presets(app);
    let picked: Vec<&str> = fields.get("selected").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    saved.into_iter().map(|(n, _)| n).filter(|n| picked.contains(&n.as_str())).collect()
}

/// Whether OK (and Enter) may confirm the dialog: a name to save, a preset to delete. True for any
/// other dialog.
pub fn can_confirm(app: &PhotocraftApp, fields: &Map<String, Value>) -> bool {
    if fields.contains_key(NEW_MARK) {
        new_name(fields).is_some()
    } else if fields.contains_key(DELETE_MARK) {
        !delete_names(app, fields).is_empty()
    } else {
        true
    }
}

pub fn body(app: &PhotocraftApp, ui: &mut egui::Ui, fields: &mut Map<String, Value>) {
    if fields.contains_key(NEW_MARK) {
        new_body(ui, fields);
    } else {
        delete_body(app, ui, fields);
    }
}

fn new_body(ui: &mut egui::Ui, fields: &mut Map<String, Value>) {
    let mut name = fields.get("name").and_then(Value::as_str).unwrap_or_default().to_string();
    let id = ui.id().with("crop-preset-name");
    ui.horizontal(|ui| {
        ui.label(tl!("Name:"));
        let r = ui.add(egui::TextEdit::singleline(&mut name).id(id).char_limit(MAX_NAME).desired_width(260.0));
        // Photoshop focuses the name with all of it selected, so typing replaces it.
        let once = id.with("focused");
        if !ui.ctx().data(|d| d.get_temp::<bool>(once).unwrap_or(false)) {
            r.request_focus();
            if let Some(mut state) = egui::TextEdit::load_state(ui.ctx(), id) {
                let all = egui::text::CCursorRange::two(egui::text::CCursor::new(0), egui::text::CCursor::new(name.chars().count()));
                state.cursor.set_char_range(Some(all));
                state.store(ui.ctx(), id);
                ui.ctx().data_mut(|d| d.insert_temp(once, true));
            }
        }
    });
    fields.insert("name".into(), json!(name));
}

fn delete_body(app: &PhotocraftApp, ui: &mut egui::Ui, fields: &mut Map<String, Value>) {
    let saved = user_presets(app);
    let mut selected: Vec<String> =
        fields.get("selected").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default();
    ui.label(tl!("Select the crop presets to delete:"));
    ui.add_space(4.0);
    let toggle = ui.input(|i| i.modifiers.command || i.modifiers.shift);
    egui::Frame::NONE.stroke(egui::Stroke::new(1.0, crate::theme::Tokens::get(ui.ctx()).field_border)).inner_margin(2.0).show(ui, |ui| {
        egui::ScrollArea::vertical().id_salt("crop-preset-delete").max_height(220.0).auto_shrink([false, true]).show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            // Full-width rows, names on the left.
            ui.with_layout(egui::Layout::top_down_justified(egui::Align::Min), |ui| {
                for (name, _) in &saved {
                    let on = selected.contains(name);
                    let r = ui.add(egui::Button::selectable(on, name.as_str()).min_size(vec2(0.0, 22.0)));
                    if r.clicked() {
                        // Click picks one; Ctrl/Cmd or Shift adds or removes it.
                        if !toggle {
                            selected.clear();
                            selected.push(name.clone());
                        } else if on {
                            selected.retain(|n| n != name);
                        } else {
                            selected.push(name.clone());
                        }
                    }
                }
            });
        });
    });
    fields.insert("selected".into(), json!(selected));
}

/// OK: save the preset, or delete the selected ones.
pub fn confirm(app: &mut PhotocraftApp, fields: &Map<String, Value>) -> Result<Value, String> {
    if fields.contains_key(NEW_MARK) {
        let name = new_name(fields).ok_or(tl!("Enter a name for the crop preset"))?;
        let options = crop_fields(fields.get("options").unwrap_or(&Value::Null));
        if !options.contains_key("crop_ratio") {
            return Err(tl!("There is no crop size or ratio to save").into());
        }
        return app.run("tool.presets.new", json!({"name": name, "tool": TOOL, "options": {"toolOptions": options}}));
    }
    let names = delete_names(app, fields);
    if names.is_empty() {
        return Err(tl!("Select the crop presets to delete").into());
    }
    app.run("tool.presets.edit", json!({"action": "delete", "preset": names}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_doc::{Color, ColorMode, Document, SampleType, Size};
    use std::{cell::RefCell, rc::Rc};

    fn app_with(services: crate::Services) -> PhotocraftApp {
        let mut doc = Document::with_background("crop", Size::new(400, 300), ColorMode::Rgb, SampleType::U8, Color::WHITE);
        doc.resolution_dpi = 72.0;
        let mut s = photocraft_engine::Session::new();
        s.add_document(doc, None);
        let mut app = PhotocraftApp::new(s, services);
        app.ui.tool = Tool::Crop;
        app
    }

    fn app() -> PhotocraftApp {
        app_with(crate::Services::default())
    }

    fn whr(app: &mut PhotocraftApp, w: &str, h: &str, r: &str, unit: &str) {
        let o = &mut app.ui.tool_options;
        o.crop_ratio = WHR.into();
        o.crop_width = w.into();
        o.crop_height = h.into();
        o.crop_resolution = r.into();
        o.crop_resolution_unit = unit.into();
    }

    fn names(app: &PhotocraftApp) -> Vec<String> {
        user_presets(app).into_iter().map(|(n, _)| n).collect()
    }

    #[test]
    fn suggested_names_follow_the_values() {
        let mut app = app();
        whr(&mut app, "4 in", "5 in", "300", PX_PER_IN);
        assert_eq!(savable(&app.ui.tool_options).unwrap().1, "4 x 5 in 300 ppi");
        whr(&mut app, "1024 px", "768 px", "", PX_PER_IN);
        assert_eq!(savable(&app.ui.tool_options).unwrap().1, "1024 x 768 px");
        whr(&mut app, "10 cm", "200 px", "118.11", PX_PER_CM);
        assert_eq!(savable(&app.ui.tool_options).unwrap().1, "10 cm x 200 px 118.11 px/cm");
        app.ui.tool_options.crop_ratio = "2:3".into();
        let (f, n) = savable(&app.ui.tool_options).unwrap();
        assert_eq!((n.as_str(), f.len()), ("2 : 3", 1));
    }

    #[test]
    fn nothing_to_save_greys_out_new_and_opens_nothing() {
        let mut app = app();
        for (ratio, w, h) in [(WHR, "", "5 in"), (WHR, "4 in", ""), (WHR, "junk", "5 in"), ("", "", ""), ("original", "", ""), ("0:0", "", ""), ("a:b", "", "")]
        {
            whr(&mut app, w, h, "300", PX_PER_IN);
            app.ui.tool_options.crop_ratio = ratio.into();
            assert!(savable(&app.ui.tool_options).is_none(), "{ratio} {w} {h}");
            assert_eq!(open_new(&mut app), None);
        }
        // No saved crop preset: Delete is greyed out too (the built-in Brush presets don't count).
        assert!(names(&app).is_empty() && !app.session.presets.tool_presets.is_empty());
        assert_eq!(open_delete(&mut app), None);
        assert!(app.ui.dialogs.is_empty());
    }

    /// New Crop Preset → saved → kept across a restart → listed in the dropdown → choosing it fills
    /// the fields; Delete Crop Preset removes it, and the dropdown falls back to the mode.
    #[test]
    fn save_reload_select_and_delete() {
        let saved = Rc::new(RefCell::new(String::new()));
        let copy = saved.clone();
        let mut app = app_with(crate::Services {
            save_prefs: Some(Box::new(move |text| {
                *copy.borrow_mut() = text.to_string();
                Ok(())
            })),
            ..Default::default()
        });
        whr(&mut app, "6 in", "9 in", "240", PX_PER_IN);
        let id = open_new(&mut app).unwrap();
        assert_eq!(open_new(&mut app), Some(id), "one dialog at a time");
        assert_eq!(app.ui.dialogs[0].fields["name"], "6 x 9 in 240 ppi");
        assert_eq!(crate::dialogs::title(&app.ui.dialogs[0]), "New Crop Preset");
        app.ui.dialog_mut(id).unwrap().fields.insert("name".into(), json!("  Poster  "));
        crate::dialogs::confirm(&mut app, id).unwrap();
        assert_eq!(names(&app), ["Poster"]);
        let builtin = crop_size::dropdown_options();
        assert_eq!(button_label(&app.ui.tool_options, &builtin, &user_presets(&app)), "Poster", "the new preset reads as selected");
        // A ratio preset too.
        app.ui.tool_options.crop_ratio = "3:2".into();
        let id = open_new(&mut app).unwrap();
        crate::dialogs::confirm(&mut app, id).unwrap();
        assert_eq!(names(&app), ["Poster", "3 : 2"]);
        crate::prefs_ui::save_preferences(&mut app).unwrap();

        // Restart.
        let text = saved.borrow().clone();
        assert!(text.contains("Poster"));
        let mut fresh = app_with(crate::Services { load_prefs: Some(Box::new(move || Some(text.clone()))), ..Default::default() });
        assert_eq!(names(&fresh), ["Poster", "3 : 2"]);
        whr(&mut fresh, "", "", "", PX_PER_CM);
        select(&mut fresh, "Poster");
        let o = &fresh.ui.tool_options;
        assert_eq!(
            (o.crop_ratio.as_str(), o.crop_width.as_str(), o.crop_height.as_str(), o.crop_resolution.as_str(), o.crop_resolution_unit.as_str()),
            (WHR, "6 in", "9 in", "240", PX_PER_IN)
        );
        assert_eq!(fresh.ui.tool, Tool::Crop);
        select(&mut fresh, "3 : 2");
        assert_eq!(fresh.ui.tool_options.crop_ratio, "3:2");
        select(&mut fresh, "Poster");

        // Delete the selected one: the fields stay, the dropdown shows the mode again.
        let id = open_delete(&mut fresh).unwrap();
        assert!(!can_confirm(&fresh, &fresh.ui.dialogs[0].fields), "nothing selected yet");
        fresh.ui.dialog_mut(id).unwrap().fields.insert("selected".into(), json!(["Poster", "not a preset"]));
        assert!(can_confirm(&fresh, &fresh.ui.dialogs[0].fields));
        crate::dialogs::confirm(&mut fresh, id).unwrap();
        assert_eq!(names(&fresh), ["3 : 2"]);
        assert_eq!(fresh.ui.tool_options.crop_width, "6 in");
        let label = button_label(&fresh.ui.tool_options, &crop_size::dropdown_options(), &user_presets(&fresh));
        assert_eq!(label, "W x H x Resolution");
        // Other tool presets are untouched.
        assert!(fresh.session.presets.tool_presets.iter().any(|p| p.name == "Soft Round 100 px"));
    }

    #[test]
    fn bad_dialog_input_is_refused_without_a_crash() {
        let mut app = app();
        whr(&mut app, "4 in", "5 in", "", PX_PER_IN);
        for bad in [json!(""), json!("   "), json!(42), json!(null), json!("x".repeat(MAX_NAME + 1))] {
            let id = open_new(&mut app).unwrap();
            app.ui.dialog_mut(id).unwrap().fields.insert("name".into(), bad.clone());
            assert!(!can_confirm(&app, &app.ui.dialogs[0].fields), "{bad}");
            assert!(crate::dialogs::confirm(&mut app, id).is_err(), "{bad}");
        }
        // Tampered options: not an object, or without a ratio.
        for bad in [json!(7), json!({"crop_width": "4 in"}), json!({"crop_ratio": 3})] {
            let id = open_new(&mut app).unwrap();
            app.ui.dialog_mut(id).unwrap().fields.insert("options".into(), bad.clone());
            assert!(crate::dialogs::confirm(&mut app, id).is_err(), "{bad}");
        }
        assert!(names(&app).is_empty());
        // Duplicate names get a number (the engine keeps names unique); a long name fits.
        for _ in 0..2 {
            let id = open_new(&mut app).unwrap();
            crate::dialogs::confirm(&mut app, id).unwrap();
        }
        let id = open_new(&mut app).unwrap();
        app.ui.dialog_mut(id).unwrap().fields.insert("name".into(), json!("é".repeat(MAX_NAME)));
        crate::dialogs::confirm(&mut app, id).unwrap();
        assert_eq!(names(&app)[..2], ["4 x 5 in", "4 x 5 in 2"]);
        // Delete with junk selections.
        for bad in [json!(null), json!("4 x 5 in"), json!([1, 2]), json!([])] {
            let id = open_delete(&mut app).unwrap();
            app.ui.dialog_mut(id).unwrap().fields.insert("selected".into(), bad.clone());
            assert!(crate::dialogs::confirm(&mut app, id).is_err(), "{bad}");
        }
        assert_eq!(names(&app).len(), 3);
        // A preset from a hand-edited file with odd options is listed and selecting it is harmless.
        app.session.presets.tool_presets.push(photocraft_engine::presets::tools::ToolPreset {
            name: "Odd".into(),
            tool: "crop".into(),
            options: json!({"toolOptions": {"crop_ratio": 5, "crop_width": []}}),
        });
        let before = app.ui.tool_options.clone();
        select(&mut app, "Odd");
        assert_eq!(app.ui.tool_options, before);
        let _ = button_label(&app.ui.tool_options, &crop_size::dropdown_options(), &user_presets(&app));
    }

    #[test]
    fn control_channel_opens_the_dialogs() {
        use crate::control::{ControlRequest, Outcome, handle};
        let mut app = app();
        let ctx = egui::Context::default();
        let call = |app: &mut PhotocraftApp, m: &str, p: Value| {
            let (r, _rx) = ControlRequest::new(m, p);
            match handle(app, &ctx, &r) {
                Outcome::Done(v) => v,
                _ => Value::Null,
            }
        };
        assert_eq!(call(&mut app, "ui.dialog.open", json!({"kind": "newCropPreset"}))["ok"], false);
        assert_eq!(call(&mut app, "ui.dialog.open", json!({"kind": "deleteCropPreset"}))["ok"], false);
        app.ui.tool_options.crop_ratio = "16:10".into();
        let id = call(&mut app, "ui.dialog.open", json!({"kind": "newCropPreset"}))["result"]["dialog"].as_u64().unwrap();
        assert_eq!(call(&mut app, "ui.dialog.set", json!({"dialog": id, "field": "name", "value": "Wide"}))["ok"], true);
        assert_eq!(call(&mut app, "ui.dialog.confirm", json!({"dialog": id}))["ok"], true);
        assert_eq!(names(&app), ["Wide"]);
        let id = call(&mut app, "ui.dialog.open", json!({"kind": "deleteCropPreset"}))["result"]["dialog"].as_u64().unwrap();
        call(&mut app, "ui.dialog.set", json!({"dialog": id, "field": "selected", "value": ["Wide"]}));
        assert_eq!(call(&mut app, "ui.dialog.confirm", json!({"dialog": id}))["ok"], true);
        assert!(names(&app).is_empty());
    }

    /// The dropdown in the real options bar: New Crop Preset… is greyed out with nothing to save,
    /// and opens the dialog otherwise.
    #[test]
    fn options_bar_dropdown_offers_new_and_delete() {
        use egui_kittest::kittest::{NodeT, Queryable};
        let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(1400.0, 800.0)).with_max_steps(64).build_eframe(|cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            let mut app = app();
            whr(&mut app, "", "", "", PX_PER_IN);
            app
        });
        h.run_steps(4);
        h.get_by_value("W x H x Resolution").click();
        h.run_steps(4);
        assert!(h.get_by_label("New Crop Preset…").accesskit_node().is_disabled(), "nothing to save");
        assert!(h.get_by_label("Delete Crop Preset…").accesskit_node().is_disabled(), "nothing saved");
        h.get_by_value("W x H x Resolution").click();
        h.run_steps(4);
        assert!(h.query_by_label("New Crop Preset…").is_none(), "the list closed");
        whr(h.state_mut(), "4 in", "6 in", "300", PX_PER_IN);
        h.run_steps(4);
        h.get_by_value("4 x 6 in 300 ppi").click();
        h.run_steps(4);
        assert!(!h.get_by_label("New Crop Preset…").accesskit_node().is_disabled());
        h.get_by_label("New Crop Preset…").click_accesskit();
        h.run_steps(6);
        assert!(h.state().ui.dialogs.iter().any(|d| d.fields.contains_key(NEW_MARK)), "{}", h.state().ui.status);
        h.get_by_label("OK").click();
        h.run_steps(6);
        assert!(h.state().ui.dialogs.is_empty());
        assert_eq!(names(h.state()), ["4 x 6 in 300 ppi"]);
    }
}
