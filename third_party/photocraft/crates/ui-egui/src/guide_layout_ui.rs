//! Photoshop-style guide layout dialog. Its preview is a pure guide overlay, never an edit.

use egui::{Color32, Stroke};
use photocraft_doc::{Document, Guides};
use serde_json::{Map, Value, json};

use crate::{PhotocraftApp, theme::Tokens, widgets};

const COMMAND: &str = "view.newGuideLayout";
const MARK: &str = "__guideLayout";

/// A `"#rrggbb"` colour as drawn.
fn rgb(hex: &str) -> Option<Color32> {
    photocraft_engine::prefs::parse_hex(hex).map(|[r, g, b]| Color32::from_rgb(r, g, b))
}

/// The Color list: New Guide's colours (`GUIDE_COLORS`) by their `"#rrggbb"`, then `current` as
/// Custom when it's none of them.
fn color_options(current: &str) -> Vec<(String, &'static str)> {
    let mut colors: Vec<(String, &'static str)> = photocraft_engine::edit_cmds::GUIDE_COLORS
        .iter()
        .map(|(name, [r, g, b])| (format!("#{r:02x}{g:02x}{b:02x}"), crate::new_guide_ui::color_label(name)))
        .collect();
    if !colors.iter().any(|(v, _)| v == current) {
        colors.push((current.to_string(), "Custom"));
    }
    colors
}

pub fn owns(f: &Map<String, Value>) -> bool {
    f.contains_key(MARK)
}

/// Alt+Cancel (Reset): factory New Guide Layout values, not the last-used remembered layout.
pub fn reset(app: &PhotocraftApp, f: &mut Map<String, Value>) {
    if !owns(f) {
        return;
    }
    let Value::Object(defaults) = crate::view_cmds::default_guide_layout() else { return };
    // Drop layout params and interactive toggles; keep dialog chrome (`__command`, …).
    f.retain(|k, _| {
        k.starts_with("__")
            && !matches!(
                k.as_str(),
                "__preset" | "__columnsEnabled" | "__rowsEnabled" | "__marginEnabled" | "__previewGuides" | "__previewParams" | "__previewSource"
            )
    });
    f.extend(defaults);
    f.insert("color".into(), json!(app.session.prefs().guides_grid_and_slices.guide_color));
    f.insert("target".into(), json!("document"));
    f.insert("__preset".into(), json!("custom"));
    f.insert("width".into(), Value::Null);
    f.insert("height".into(), Value::Null);
    let margin = f.get("margin").cloned().unwrap_or(json!(0));
    if !margin.is_array() {
        f.insert("margin".into(), json!([margin.clone(), margin.clone(), margin.clone(), margin]));
    }
    f.insert("__preview".into(), json!(true));
}

fn on(f: &Map<String, Value>, key: &str, fallback: bool) -> bool {
    f.get(key).and_then(Value::as_bool).unwrap_or(fallback)
}
fn positive(f: &Map<String, Value>, key: &str) -> bool {
    f.get(key).and_then(Value::as_f64).is_some_and(|n| n > 0.0)
}

pub fn open(app: &mut PhotocraftApp) -> u64 {
    let mut f = app.ui.view.guide_layout.as_object().cloned().unwrap_or_default();
    f.insert(MARK.into(), json!(true));
    f.insert("__command".into(), json!(COMMAND));
    f.insert("__label".into(), json!("New Guide Layout…"));
    f.insert("__document".into(), app.session.active().map(|s| json!(s.doc.id)).unwrap_or(Value::Null));
    f.insert("__preview".into(), json!(true));
    f.insert("color".into(), json!(app.session.prefs().guides_grid_and_slices.guide_color));
    f.entry("target").or_insert_with(|| json!("document"));
    if f.get("target").and_then(Value::as_str) == Some("selectedArtboards") && photocraft_engine::guide_layout::selected_artboards(&app.session).is_empty() {
        f.insert("target".into(), json!("document"));
    }
    f.entry("__preset").or_insert_with(|| json!("custom"));
    f.entry("width").or_insert(Value::Null);
    f.entry("height").or_insert(Value::Null);
    let margin = f.get("margin").cloned().unwrap_or(json!(0));
    if !margin.is_array() {
        f.insert("margin".into(), json!([margin.clone(), margin.clone(), margin.clone(), margin]));
    }
    for key in ["__previewGuides", "__previewParams", "__previewSource"] {
        f.remove(key);
    }
    app.ui.open_dialog(crate::state::DialogKind::Command, f)
}

/// Preserve legacy command parameter names for automation and remembered defaults.
pub fn params(f: &Map<String, Value>) -> Value {
    let mut p: Map<String, Value> = ["columns", "rows", "width", "height", "gutter", "rowGutter", "margin", "centerColumns", "clearExisting", "target"]
        .into_iter()
        .filter_map(|k| f.get(k).cloned().map(|v| (k.to_string(), v)))
        .collect();
    for (toggle, key) in [("__columnsEnabled", "columns"), ("__rowsEnabled", "rows")] {
        if !on(f, toggle, positive(f, key)) {
            p.insert(key.into(), json!(0));
        }
    }
    if !on(f, "__marginEnabled", true) {
        p.insert("margin".into(), json!(0));
    }
    for key in ["width", "height"] {
        if let Some(Value::String(s)) = p.get(key) {
            let value = if s.trim().is_empty() {
                Value::Null
            } else {
                widgets::parse_num(s).and_then(serde_json::Number::from_f64).map(Value::Number).unwrap_or_else(|| json!(s))
            };
            p.insert(key.into(), value);
        }
    }
    // A disabled axis must not be rejected for a stale width or gap typed before disabling it.
    for (key, size, gap) in [("columns", "width", "gutter"), ("rows", "height", "rowGutter")] {
        if p.get(key).and_then(Value::as_u64) == Some(0) {
            p.insert(size.into(), Value::Null);
            p.insert(gap.into(), json!(0));
        }
    }
    Value::Object(p)
}

fn source(app: &PhotocraftApp) -> Value {
    app.session.active().map(|st| json!([st.doc.id, st.revision, st.active_layer, st.selected_layers])).unwrap_or(Value::Null)
}

fn calculate(app: &PhotocraftApp, f: &Map<String, Value>) -> Result<Guides, String> {
    let st = app.session.active().ok_or("no document")?;
    if f.get("__document") != Some(&json!(st.doc.id)) {
        return Err("the guide layout belongs to another document".into());
    }
    let params = params(f);
    // Painting an unchanged dialog must not repeat the duplicate scans every frame.
    if f.get("__previewParams") == Some(&params)
        && f.get("__previewSource") == Some(&source(app))
        && let Some(guides) = f.get("__previewGuides").and_then(|v| serde_json::from_value(v.clone()).ok())
    {
        return Ok(guides);
    }
    photocraft_engine::guide_layout::calculate(&app.session, &params).map(|l| l.guides).map_err(|e| e.to_string())
}

pub fn validate(app: &PhotocraftApp, f: &Map<String, Value>) -> Result<(), String> {
    let color = f.get("color").and_then(Value::as_str).unwrap_or("#4affff");
    rgb(color).ok_or("invalid guide color")?;
    calculate(app, f).map(|_| ())
}

pub fn confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    validate(app, f)?;
    let color = f.get("color").and_then(Value::as_str).unwrap_or("#4affff");
    // Colour is the existing global guide display preference, not document data.
    let color = rgb(color).ok_or("invalid guide color")?;
    let hex = format!("#{:02x}{:02x}{:02x}", color.r(), color.g(), color.b());
    let result = app.run(COMMAND, params(f))?;
    // The preview is visible even when guides were hidden. Keep a committed layout visible too.
    app.ui.view.extras = true;
    app.ui.view.show.canvas_guides = true;
    app.ui.extras.guides = true;
    app.session.prefs.edit(|prefs| prefs.guides_grid_and_slices.guide_color = hex);
    let mut remembered = f.clone();
    for key in [MARK, "__command", "__label", "__document", "__previewGuides", "__previewParams", "__previewSource"] {
        remembered.remove(key);
    }
    app.ui.view.guide_layout = Value::Object(remembered);
    Ok(result)
}

/// Full replacement guide set for the dialog's own document. Cancel/removal drops it immediately.
pub fn preview(app: &PhotocraftApp, doc: &Document) -> Option<(Guides, Color32)> {
    let dialog = app.ui.dialogs.iter().rev().find(|d| owns(&d.fields))?;
    let f = &dialog.fields;
    if !on(f, "__preview", true) || f.get("__document") != Some(&json!(doc.id)) {
        return None;
    }
    let guides = calculate(app, f).ok().or_else(|| f.get("__previewGuides").and_then(|v| serde_json::from_value(v.clone()).ok()))?;
    let color = f.get("color").and_then(Value::as_str).and_then(rgb).unwrap_or(Color32::from_rgb(74, 255, 255));
    Some((guides, color))
}

fn check(ui: &mut egui::Ui, f: &mut Map<String, Value>, key: &str, label: &str, fallback: bool) -> bool {
    let mut value = on(f, key, fallback);
    if widgets::checkbox(ui, &mut value, label).changed() {
        f.insert(key.into(), json!(value));
    }
    value
}

fn number(ui: &mut egui::Ui, f: &mut Map<String, Value>, key: &str, count: bool) {
    let mut value = f.get(key).and_then(Value::as_f64).unwrap_or(0.0) as f32;
    if widgets::value_field(ui, &mut value, (if count { 1.0 } else { 0.0 })..=if count { 1000.0 } else { 300_000.0 }, if count { "" } else { "px" }, 68.0)
        .changed()
    {
        f.insert(key.into(), if count { json!(value.round() as u64) } else { json!(value) });
    }
}

fn optional(ui: &mut egui::Ui, f: &mut Map<String, Value>, key: &str) {
    let mut value = match f.get(key) {
        Some(Value::String(s)) => s.clone(),
        Some(v) if v.is_number() => v.to_string(),
        _ => String::new(),
    };
    let t = Tokens::get(ui.ctx());
    let response = ui.add(egui::TextEdit::singleline(&mut value).id_salt(key).desired_width(60.0).font(crate::theme::mono(12.0)).background_color(t.field));
    if ui.is_enabled() {
        crate::field_tab::register(ui.ctx(), response.id);
    }
    if response.gained_focus() {
        crate::field_tab::select_all(ui.ctx(), response.id, &value);
    }
    if response.changed() {
        f.insert(key.into(), json!(value));
    }
    response.on_hover_text(tl!("Leave empty for automatic size"));
}

fn group<R>(ui: &mut egui::Ui, content: impl FnOnce(&mut egui::Ui) -> R) -> R {
    let t = Tokens::get(ui.ctx());
    egui::Frame::new().stroke(Stroke::new(1.0, t.field_border)).inner_margin(8).show(ui, |ui| ui.vertical(content).inner).inner
}

fn axis(ui: &mut egui::Ui, f: &mut Map<String, Value>, rows: bool) {
    let (toggle, count, size, gap, title, size_label) = if rows {
        ("__rowsEnabled", "rows", "height", "rowGutter", tl!("Rows"), tl!("Height"))
    } else {
        ("__columnsEnabled", "columns", "width", "gutter", tl!("Columns"), tl!("Width"))
    };
    group(ui, |ui| {
        let enabled = check(ui, f, toggle, title, positive(f, count));
        if enabled && f.get(count).and_then(Value::as_f64) == Some(0.0) {
            f.insert(count.into(), json!(1));
        }
        ui.add_space(6.0);
        ui.add_enabled_ui(enabled, |ui| {
            egui::Grid::new(count).spacing([8.0, 6.0]).show(ui, |ui| {
                ui.label(tl!("Number"));
                number(ui, f, count, true);
                ui.end_row();
                ui.label(size_label);
                optional(ui, f, size);
                ui.end_row();
                ui.label(tl!("Spacing"));
                number(ui, f, gap, false);
                ui.end_row();
            });
        });
    });
}

fn main_fields(app: &PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    let before: Map<String, Value> = f.iter().filter(|(key, _)| !key.starts_with("__preview")).map(|(k, v)| (k.clone(), v.clone())).collect();
    egui::Grid::new("guide-header").spacing([8.0, 6.0]).show(ui, |ui| {
        ui.label(tl!("Preset:"));
        let mut preset = f.get("__preset").and_then(Value::as_str).unwrap_or("custom").to_string();
        if widgets::dropdown(
            ui,
            "guide-preset",
            &mut preset,
            &[("custom".into(), "Custom"), ("five".into(), "5 Columns, 3 Rows"), ("eight".into(), "8 Columns"), ("twelve".into(), "12 Columns")],
            222.0,
        ) {
            f.insert("__preset".into(), json!(preset));
            match preset.as_str() {
                "five" | "eight" | "twelve" => {
                    let (cols, rows, gap) = match preset.as_str() {
                        "five" => (5, 3, 0),
                        "eight" => (8, 0, 20),
                        _ => (12, 0, 24),
                    };
                    for (k, v) in [
                        ("columns", json!(cols)),
                        ("rows", json!(rows)),
                        ("gutter", json!(gap)),
                        ("rowGutter", json!(0)),
                        ("width", Value::Null),
                        ("height", Value::Null),
                        ("margin", json!([0, 0, 0, 0])),
                        ("__columnsEnabled", json!(true)),
                        ("__rowsEnabled", json!(rows > 0)),
                        ("__marginEnabled", json!(true)),
                        ("centerColumns", json!(false)),
                    ] {
                        f.insert(k.into(), v);
                    }
                }
                _ => {}
            }
        }
        ui.end_row();
        ui.label(tl!("Target:"));
        let mut target = f.get("target").and_then(Value::as_str).unwrap_or("document").to_string();
        let mut options = vec![("document".to_string(), "Document")];
        if !photocraft_engine::guide_layout::selected_artboards(&app.session).is_empty() {
            options.push(("selectedArtboards".into(), "Selected Artboards"));
        }
        if widgets::dropdown(ui, "guide-target", &mut target, &options, 222.0) {
            f.insert("target".into(), json!(target));
        }
        ui.end_row();
    });
    ui.add_space(10.0);
    group(ui, |ui| {
        ui.label(tl!("Color"));
        ui.horizontal(|ui| {
            ui.label(tl!("Color:"));
            let mut color = f.get("color").and_then(Value::as_str).unwrap_or("#4affff").to_string();
            let colors = color_options(&color);
            if widgets::dropdown(ui, "guide-color", &mut color, &colors, 174.0) {
                f.insert("color".into(), json!(color));
            }
            let shown = rgb(&color).unwrap_or(Tokens::get(ui.ctx()).accent);
            if widgets::color_swatch_button(ui, shown, tl!("Guide color")).clicked() {
                crate::color_picker_ui::request(f, tl!("Guide color"));
            }
        });
    });
    ui.add_space(10.0);
    ui.horizontal_top(|ui| {
        axis(ui, f, false);
        ui.add_space(4.0);
        axis(ui, f, true);
    });
    ui.add_space(10.0);
    group(ui, |ui| {
        let enabled = check(ui, f, "__marginEnabled", tl!("Margin"), true);
        let mut margins = f.get("margin").and_then(Value::as_array).cloned().unwrap_or_else(|| vec![json!(0); 4]);
        ui.add_space(5.0);
        ui.add_enabled_ui(enabled, |ui| {
            ui.horizontal(|ui| {
                for (value, label) in margins.iter_mut().zip([tl!("Top:"), tl!("Left:"), tl!("Bottom:"), tl!("Right:")]) {
                    ui.vertical(|ui| {
                        ui.label(label);
                        let mut n = value.as_f64().unwrap_or(0.0) as f32;
                        if widgets::value_field(ui, &mut n, 0.0..=300_000.0, "px", 64.0).changed() {
                            *value = json!(n);
                        }
                    });
                }
            });
        });
        f.insert("margin".into(), Value::Array(margins));
    });
    ui.add_space(8.0);
    check(ui, f, "centerColumns", tl!("Center Columns"), false);
    check(ui, f, "clearExisting", tl!("Clear Existing Guides"), false);
    // A typed edit makes a named preset custom; changing target/colour/preview does not.
    if ["columns", "rows", "width", "height", "gutter", "rowGutter", "margin", "centerColumns", "__columnsEnabled", "__rowsEnabled", "__marginEnabled"]
        .iter()
        .any(|k| before.get(*k) != f.get(*k))
        && before.get("__preset") == f.get("__preset")
    {
        f.insert("__preset".into(), json!("custom"));
    }
}

/// Own the action rail as well as the body, so OK/Cancel/Preview follow Photoshop's arrangement.
pub fn body(app: &PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>, interactive: bool) -> Option<bool> {
    let before = (params(f), on(f, "__preview", true), f.get("color").cloned());
    let mut outcome = None;
    ui.horizontal_top(|ui| {
        ui.vertical(|ui| {
            main_fields(app, ui, f);
        });
        ui.add_space(12.0);
        ui.vertical(|ui| {
            let valid = validate(app, f).is_ok();
            if ui.add_enabled_ui(valid, |ui| widgets::primary_button(ui, tl!("OK"), 96.0)).inner.clicked() {
                outcome = Some(true);
            }
            ui.add_space(4.0);
            if widgets::secondary_button(ui, tl!("Cancel"), 96.0).clicked() {
                outcome = Some(false);
            }
            ui.add_space(6.0);
            check(ui, f, "__preview", tl!("Preview"), true);
        });
    });
    match calculate(app, f) {
        Ok(guides) => {
            f.insert("__previewGuides".into(), json!(guides));
            f.insert("__previewParams".into(), params(f));
            f.insert("__previewSource".into(), source(app));
        }
        Err(error) => {
            ui.colored_label(Tokens::get(ui.ctx()).text_dim, tl!("Invalid guide layout")).on_hover_text(error);
        }
    }
    if outcome.is_none() && interactive && validate(app, f).is_ok() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
        outcome = Some(true);
    }
    // The canvas paints before the dialog. Schedule the frame that displays this draft edit.
    if before != (params(f), on(f, "__preview", true), f.get("color").cloned()) {
        ui.ctx().request_repaint();
    }
    outcome
}

#[cfg(test)]
mod tests;
