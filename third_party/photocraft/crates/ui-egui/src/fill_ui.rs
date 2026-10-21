//! Edit › Fill… (Shift+F5, Shift+Backspace): Photoshop's Fill dialog in front of `edit.fill`.
//!
//! Contents (Foreground, Background, Color…, Content-Aware, Pattern, History, Black, 50% Gray,
//! White) with the content's own options, then Blending: Mode, Opacity and Preserve
//! Transparency. The fields are the command's params, so `ui.dialog.set` / `ui.dialog.confirm`
//! drive it like any dialog. OK remembers the choices in the preferences (`dialogs["edit.fill"]`),
//! which are saved to disk, so the dialog opens with them again after a restart.

use egui::{Align2, Sense, pos2, vec2};
use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::theme::Tokens;

pub const COMMAND: &str = "edit.fill";
const MARK: &str = "__fill";

/// The Contents menu: (param, label), in Photoshop's order.
pub const CONTENTS: [(&str, &str); 9] = [
    ("foreground", "Foreground Color"),
    ("background", "Background Color"),
    ("color", "Color…"),
    ("contentAware", "Content-Aware"),
    ("pattern", "Pattern"),
    ("history", "History"),
    ("black", "Black"),
    ("gray", "50% Gray"),
    ("white", "White"),
];

/// Params the dialog edits and remembers.
const KEYS: [&str; 7] = ["contents", "color", "pattern", "colorAdaptation", "mode", "opacity", "preserveTransparency"];

/// Is this dialog the Fill dialog?
pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.contains_key(MARK)
}

fn hex(c: [f32; 4]) -> String {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", b(c[0]), b(c[1]), b(c[2]))
}

/// A remembered value, if it is one the dialog can show (a corrupt preference is ignored).
fn valid(app: &PhotocraftApp, key: &str, v: &Value) -> bool {
    match key {
        "contents" => v.as_str().is_some_and(|c| CONTENTS.iter().any(|(k, _)| *k == c)),
        "mode" | "opacity" | "preserveTransparency" => crate::dialog_blend_ui::valid(key, v),
        "color" => v.as_str().is_some_and(|h| h.len() == 7 && h.starts_with('#') && h.get(1..).is_some_and(|d| d.chars().all(|c| c.is_ascii_hexdigit()))),
        "pattern" => v.as_str().is_some_and(|p| app.session.patterns.items.iter().any(|q| q.id == p)),
        "colorAdaptation" => v.is_boolean(),
        _ => false,
    }
}

/// Photoshop's Fill defaults (no last-used prefs). Used by open and Alt+Reset.
pub fn default_fields(app: &PhotocraftApp) -> Map<String, Value> {
    let mut f = Map::new();
    f.insert(MARK.into(), json!(true));
    f.insert("__command".into(), json!(COMMAND));
    f.insert("__label".into(), json!("Fill"));
    f.insert("contents".into(), json!("foreground"));
    f.insert("color".into(), json!(hex(app.session.tools.foreground)));
    if let Some(p) = app.session.patterns.items.first() {
        f.insert("pattern".into(), json!(p.id));
    }
    f.insert("colorAdaptation".into(), json!(true));
    f.insert("mode".into(), json!("normal"));
    f.insert("opacity".into(), json!(100.0));
    f.insert("preserveTransparency".into(), json!(false));
    f
}

/// The dialog's fields: Photoshop's defaults, overridden by the last choices.
pub fn fields(app: &PhotocraftApp) -> Map<String, Value> {
    let mut f = default_fields(app);
    if let Some(Value::Object(saved)) = app.session.prefs().dialogs.get(COMMAND) {
        for k in KEYS {
            if let Some(v) = saved.get(k).filter(|v| valid(app, k, v)) {
                f.insert(k.into(), v.clone());
            }
        }
    }
    f
}

/// Alt+Cancel (Reset): Photoshop Fill defaults, not last-used prefs. Dialog stays open.
pub fn reset(app: &PhotocraftApp, fields: &mut Map<String, Value>) {
    if !owns(fields) {
        return;
    }
    let defaults = default_fields(app);
    for k in KEYS {
        if let Some(v) = defaults.get(k) {
            fields.insert(k.into(), v.clone());
        }
    }
}

/// Open the Fill dialog; returns its id.
pub fn open(app: &mut PhotocraftApp) -> u64 {
    let f = fields(app);
    app.ui.open_dialog(crate::state::DialogKind::Command, f)
}

/// The `edit.fill` params of the dialog's fields: only the options of the chosen contents.
pub fn params(f: &Map<String, Value>) -> Value {
    let contents = f.get("contents").and_then(Value::as_str).unwrap_or("foreground");
    let mut p = json!({ "contents": contents });
    for k in ["mode", "opacity", "preserveTransparency"] {
        if let Some(v) = f.get(k) {
            p[k] = v.clone();
        }
    }
    let extra = match contents {
        "color" => Some("color"),
        "pattern" => Some("pattern"),
        "contentAware" => Some("colorAdaptation"),
        _ => None,
    };
    if let Some(k) = extra
        && let Some(v) = f.get(k)
    {
        p[k] = v.clone();
    }
    p
}

/// OK: remember the choices, then fill.
pub fn confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    let remembered: Map<String, Value> = KEYS.iter().filter_map(|k| f.get(*k).map(|v| (k.to_string(), v.clone()))).collect();
    app.session.prefs.edit(|p| p.dialogs.insert(COMMAND.into(), Value::Object(remembered)));
    app.run(COMMAND, params(f))
}

fn label(ui: &mut egui::Ui, text: &str) {
    let t = Tokens::get(ui.ctx());
    let (r, _) = ui.allocate_exact_size(vec2(112.0, 22.0), Sense::hover());
    ui.painter().text(pos2(r.right() - 6.0, r.center().y), Align2::RIGHT_CENTER, text, egui::FontId::proportional(12.0), t.text_dim);
}

fn get_str(f: &Map<String, Value>, k: &str, d: &str) -> String {
    f.get(k).and_then(Value::as_str).unwrap_or(d).to_string()
}

fn parse_hex(h: &str) -> [u8; 3] {
    let d = |i: usize| h.get(i..i + 2).and_then(|x| u8::from_str_radix(x, 16).ok()).unwrap_or(0);
    [d(1), d(3), d(5)]
}

/// The dialog body.
pub fn body(app: &PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    ui.spacing_mut().item_spacing.y = 6.0;
    let mut contents = get_str(f, "contents", "foreground");
    ui.horizontal(|ui| {
        label(ui, tl!("Contents:"));
        let opts: Vec<(String, &str)> = CONTENTS.iter().map(|(k, l)| (k.to_string(), *l)).collect();
        if crate::widgets::dropdown(ui, "fill-contents", &mut contents, &opts, 170.0) {
            f.insert("contents".into(), json!(contents));
            // Choosing Color… opens the Color Picker, as in Photoshop.
            if contents == "color" {
                crate::color_picker_ui::request(f, tl!("Color Picker (Fill Color)"));
            }
        }
    });
    match contents.as_str() {
        "color" => {
            ui.horizontal(|ui| {
                label(ui, tl!("Color:"));
                // Photoshop shows no swatch; this one reopens the Color Picker on the colour.
                let rgb = parse_hex(&get_str(f, "color", "#000000"));
                if crate::widgets::color_swatch_button(ui, egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]), tl!("Fill color")).clicked() {
                    crate::color_picker_ui::request(f, tl!("Color Picker (Fill Color)"));
                }
            });
        }
        "pattern" => {
            ui.horizontal(|ui| {
                label(ui, tl!("Custom Pattern:"));
                let pat = get_str(f, "pattern", "");
                if app.session.patterns.items.is_empty() {
                    ui.label(egui::RichText::new(tl!("No patterns")).color(Tokens::get(ui.ctx()).text_faint));
                } else {
                    // Photoshop shows the pattern itself and opens a grid of swatches.
                    if let Some(id) = crate::preset_panels::pattern_picker(app, ui, &pat) {
                        f.insert("pattern".into(), json!(id));
                    }
                    let name = app.session.patterns.items.iter().find(|p| p.id == get_str(f, "pattern", "")).map(|p| p.display_name().to_string());
                    ui.label(egui::RichText::new(name.unwrap_or_default()).color(Tokens::get(ui.ctx()).text_dim));
                }
            });
        }
        "contentAware" => {
            ui.horizontal(|ui| {
                label(ui, "");
                let mut on = f.get("colorAdaptation").and_then(Value::as_bool).unwrap_or(true);
                if crate::widgets::checkbox(ui, &mut on, "Color Adaptation").changed() {
                    f.insert("colorAdaptation".into(), json!(on));
                }
            });
        }
        _ => {}
    }
    crate::dialog_blend_ui::body(ui, f, "fill");
}

#[cfg(test)]
#[path = "fill_ui_tests.rs"]
mod tests;
