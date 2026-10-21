//! Channels panel › Channel Options… (double-click an alpha or spot channel's thumbnail, or the
//! Quick Mask row): name, Color Indicates Masked Areas / Selected Areas / Spot Color, the overlay
//! or ink colour, and opacity (solidity for spot channels). OK runs the engine's
//! `channel.options`, which agents can also call directly with params (#3761).

use egui::{Align2, Sense, pos2, vec2};
use photocraft_doc::ColorIndicates;
use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::theme::Tokens;

pub const COMMAND: &str = "channel.options";
const MARK: &str = "__channelOptions";

pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.contains_key(MARK)
}

fn hex(rgb: [f32; 3]) -> String {
    crate::color_picker_ui::hex(rgb)
}

/// The dialog's fields for `channel` (an alpha/spot channel index, or `"quickMask"`), or `None`
/// when there's no document or no such channel.
pub fn fields(app: &PhotocraftApp, channel: &Value) -> Option<Map<String, Value>> {
    let doc = &app.session.active()?.doc;
    let kind = |i: ColorIndicates| if i == ColorIndicates::SelectedAreas { "selected" } else { "masked" };
    let (name, indicates, color, percent) = if channel.as_str() == Some("quickMask") {
        let o = app.session.quick_mask_options;
        (None, kind(o.indicates), o.color, o.opacity)
    } else {
        let ch = doc.channels.get(usize::try_from(channel.as_u64()?).ok()?)?;
        match ch.spot {
            Some((ink, solidity)) => (Some(ch.name.clone()), "spot", ink, solidity),
            None => (Some(ch.name.clone()), kind(ch.indicates), ch.color, ch.opacity),
        }
    };
    let mut f = Map::new();
    f.insert(MARK.into(), json!(true));
    f.insert("__command".into(), json!(COMMAND));
    f.insert("__label".into(), json!(if name.is_some() { "Channel Options…" } else { "Quick Mask Options…" }));
    f.insert("channel".into(), channel.clone());
    if let Some(n) = name {
        f.insert("name".into(), json!(n));
    }
    f.insert("indicates".into(), json!(indicates));
    f.insert("color".into(), json!(hex(color.to_rgb())));
    f.insert("opacity".into(), json!((percent.clamp(0.0, 1.0) * 100.0).round()));
    Some(f)
}

/// Opens Channel Options on `channel`, or `None` if it doesn't exist.
pub fn open(app: &mut PhotocraftApp, channel: &Value) -> Option<u64> {
    let f = fields(app, channel)?;
    Some(app.ui.open_dialog(crate::state::DialogKind::Command, f))
}

/// The `channel.options` params for the dialog's fields: spot channels take `solidity`, the
/// others `opacity`.
pub fn params(f: &Map<String, Value>) -> Value {
    let spot = f.get("indicates").and_then(Value::as_str) == Some("spot");
    let mut p = Map::new();
    for key in ["channel", "name", "indicates", "color"] {
        if let Some(v) = f.get(key) {
            p.insert(key.into(), v.clone());
        }
    }
    if let Some(v) = f.get("opacity") {
        p.insert(if spot { "solidity" } else { "opacity" }.into(), v.clone());
    }
    Value::Object(p)
}

pub fn confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    app.run(COMMAND, params(f))
}

fn label(ui: &mut egui::Ui, text: &str) {
    let t = Tokens::get(ui.ctx());
    let (r, _) = ui.allocate_exact_size(vec2(84.0, 22.0), Sense::hover());
    ui.painter().text(pos2(r.right() - 6.0, r.center().y), Align2::RIGHT_CENTER, text, egui::FontId::proportional(12.0), t.text_dim);
}

pub fn body(ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    ui.spacing_mut().item_spacing.y = 6.0;
    let quick = f.get("channel").and_then(Value::as_str) == Some("quickMask");
    if let Some(mut name) = f.get("name").and_then(Value::as_str).map(str::to_string) {
        ui.horizontal(|ui| {
            label(ui, tl!("Name:"));
            if ui.add(egui::TextEdit::singleline(&mut name).desired_width(200.0)).changed() {
                f.insert("name".into(), json!(name));
            }
        });
    }
    crate::widgets::section_label(ui, tl!("Color Indicates:"));
    crate::widgets::hairline(ui);
    let mut indicates = f.get("indicates").and_then(Value::as_str).unwrap_or("masked").to_string();
    let before = indicates.clone();
    ui.indent("channel-options-indicates", |ui| {
        ui.radio_value(&mut indicates, "masked".to_string(), tl!("Masked Areas"));
        ui.radio_value(&mut indicates, "selected".to_string(), tl!("Selected Areas"));
        // Quick Mask can't be a spot channel.
        if !quick {
            ui.radio_value(&mut indicates, "spot".to_string(), tl!("Spot Color"));
        }
    });
    if indicates != before {
        f.insert("indicates".into(), json!(indicates));
    }
    let spot = indicates == "spot";
    ui.add_space(4.0);
    crate::widgets::section_label(ui, tl!("Color"));
    crate::widgets::hairline(ui);
    ui.horizontal(|ui| {
        label(ui, "");
        let current = f.get("color").and_then(Value::as_str).and_then(crate::color_picker_ui::parse_hex).unwrap_or([1.0, 0.0, 0.0]);
        let c = current.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
        let what = if spot { tl!("Spot Color") } else { tl!("Channel Color") };
        if crate::widgets::color_swatch_button(ui, egui::Color32::from_rgb(c[0], c[1], c[2]), what).clicked() {
            crate::color_picker_ui::request(f, &crate::color_picker_ui::title_for(what));
        }
        ui.add_space(12.0);
        ui.label(if spot { tl!("Solidity:") } else { tl!("Opacity:") });
        let mut pct = f.get("opacity").and_then(Value::as_f64).unwrap_or(50.0) as f32;
        if crate::widgets::value_field(ui, &mut pct, 0.0..=100.0, "%", 64.0).changed() {
            f.insert("opacity".into(), json!(pct.round()));
        }
    });
}

// ---------------------------------------------------------------------------------------------
// New Spot Channel…: Name and Ink Characteristics (Color, Solidity). OK runs `channel.newSpot`
// (from the selection, as Photoshop does) and remembers the ink and solidity for next time.

pub const NEW_SPOT: &str = "channel.newSpot";
const NEW_SPOT_MARK: &str = "__newSpot";

pub fn owns_new_spot(fields: &Map<String, Value>) -> bool {
    fields.contains_key(NEW_SPOT_MARK)
}

/// The New Spot Channel fields: the next "Spot Color N" name and the last ink and solidity
/// (red and 0% at first), or `None` without a document.
pub fn new_spot_fields(app: &PhotocraftApp) -> Option<Map<String, Value>> {
    let doc = &app.session.active()?.doc;
    let saved = app.session.prefs().dialogs.get(NEW_SPOT).cloned().unwrap_or_default();
    let color = saved.get("color").and_then(Value::as_str).filter(|h| crate::color_picker_ui::parse_hex(h).is_some()).unwrap_or("#ff0000");
    let solidity = saved.get("solidity").and_then(Value::as_f64).filter(|v| (0.0..=100.0).contains(v)).unwrap_or(0.0);
    let mut f = Map::new();
    f.insert(NEW_SPOT_MARK.into(), json!(true));
    f.insert("__command".into(), json!(NEW_SPOT));
    f.insert("__label".into(), json!("New Spot Channel…"));
    f.insert("name".into(), json!(photocraft_engine::channel_cmds::next_alpha_name(doc, "Spot Color")));
    f.insert("color".into(), json!(color));
    f.insert("solidity".into(), json!(solidity));
    Some(f)
}

/// Opens New Spot Channel, or `None` without a document.
pub fn open_new_spot(app: &mut PhotocraftApp) -> Option<u64> {
    let f = new_spot_fields(app)?;
    Some(app.ui.open_dialog(crate::state::DialogKind::Command, f))
}

pub fn new_spot_params(f: &Map<String, Value>) -> Value {
    let mut p = Map::new();
    for key in ["name", "color", "solidity"] {
        if let Some(v) = f.get(key) {
            p.insert(key.into(), v.clone());
        }
    }
    Value::Object(p)
}

pub fn confirm_new_spot(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    let p = new_spot_params(f);
    let result = app.run(NEW_SPOT, p.clone());
    if result.is_ok() {
        let remembered = json!({ "color": p.get("color"), "solidity": p.get("solidity") });
        app.session.prefs.edit(|prefs| prefs.dialogs.insert(NEW_SPOT.into(), remembered));
    }
    result
}

pub fn new_spot_body(ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    ui.spacing_mut().item_spacing.y = 6.0;
    let mut name = f.get("name").and_then(Value::as_str).unwrap_or_default().to_string();
    ui.horizontal(|ui| {
        label(ui, tl!("Name:"));
        if ui.add(egui::TextEdit::singleline(&mut name).desired_width(200.0)).changed() {
            f.insert("name".into(), json!(name));
        }
    });
    crate::widgets::section_label(ui, tl!("Ink Characteristics"));
    crate::widgets::hairline(ui);
    ui.horizontal(|ui| {
        label(ui, tl!("Color:"));
        let current = f.get("color").and_then(Value::as_str).and_then(crate::color_picker_ui::parse_hex).unwrap_or([1.0, 0.0, 0.0]);
        let c = current.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
        if crate::widgets::color_swatch_button(ui, egui::Color32::from_rgb(c[0], c[1], c[2]), tl!("Spot Color")).clicked() {
            crate::color_picker_ui::request(f, &crate::color_picker_ui::title_for(tl!("Spot Color")));
        }
        ui.add_space(12.0);
        ui.label(tl!("Solidity:"));
        let mut pct = f.get("solidity").and_then(Value::as_f64).unwrap_or(0.0) as f32;
        if crate::widgets::value_field(ui, &mut pct, 0.0..=100.0, "%", 64.0).changed() {
            f.insert("solidity".into(), json!(pct.round()));
        }
    });
}

#[cfg(test)]
#[path = "channel_options_ui_tests.rs"]
mod tests;
