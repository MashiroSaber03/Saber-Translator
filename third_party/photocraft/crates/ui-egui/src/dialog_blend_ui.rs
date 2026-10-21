//! Blending controls shared by Edit › Fill and Edit › Stroke.
//!
//! Both commands support the same layer blend modes, opacity and transparency lock.
//! Keeping one widget and validation implementation prevents their options diverging.

use egui::{Align2, Sense, pos2, vec2};
use serde_json::{Map, Value, json};

use crate::theme::Tokens;

/// Validate a persisted blending choice before restoring it from Preferences.
pub fn valid(key: &str, v: &Value) -> bool {
    match key {
        "mode" => v.as_str().and_then(photocraft_engine::commands::blend_from_str).is_some_and(|m| m != photocraft_color::BlendMode::PassThrough),
        "opacity" => v.as_f64().is_some_and(|o| o.is_finite() && (0.0..=100.0).contains(&o)),
        "preserveTransparency" => v.is_boolean(),
        _ => false,
    }
}

/// UI for the identical Blending section of Fill and Stroke.
pub fn body(ui: &mut egui::Ui, fields: &mut Map<String, Value>, prefix: &str) {
    ui.add_space(4.0);
    crate::widgets::section_label(ui, tl!("Blending"));
    crate::widgets::hairline(ui);
    ui.horizontal(|ui| {
        label(ui, tl!("Mode:"));
        let mut mode = fields.get("mode").and_then(Value::as_str).unwrap_or("normal").to_string();
        let opts: Vec<(String, &str)> = photocraft_color::BlendMode::LAYER_MODES.iter().map(|m| (mode_key(*m), m.label())).collect();
        if crate::widgets::dropdown(ui, &format!("{prefix}-mode"), &mut mode, &opts, 170.0) {
            fields.insert("mode".into(), json!(mode));
        }
    });
    ui.horizontal(|ui| {
        label(ui, tl!("Opacity:"));
        let mut opacity = fields.get("opacity").and_then(Value::as_f64).unwrap_or(100.0) as f32;
        if crate::widgets::value_field(ui, &mut opacity, 0.0..=100.0, "%", 70.0).changed() {
            fields.insert("opacity".into(), json!(opacity.round()));
        }
    });
    ui.horizontal(|ui| {
        label(ui, "");
        let mut preserve = fields.get("preserveTransparency").and_then(Value::as_bool).unwrap_or(false);
        if crate::widgets::checkbox(ui, &mut preserve, tl!("Preserve Transparency")).changed() {
            fields.insert("preserveTransparency".into(), json!(preserve));
        }
    });
}

fn label(ui: &mut egui::Ui, text: &str) {
    let t = Tokens::get(ui.ctx());
    let (r, _) = ui.allocate_exact_size(vec2(112.0, 22.0), Sense::hover());
    ui.painter().text(pos2(r.right() - 6.0, r.center().y), Align2::RIGHT_CENTER, text, egui::FontId::proportional(12.0), t.text_dim);
}

/// The `mode` param naming a blend mode ("normal", "colorBurn"…), parsed by the engine.
fn mode_key(m: photocraft_color::BlendMode) -> String {
    let s = format!("{m:?}");
    let mut c = s.chars();
    c.next().map(|f| f.to_ascii_lowercase().to_string() + c.as_str()).unwrap_or_default()
}
