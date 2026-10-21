//! The Set Black / Neutral Gray / White Point eyedroppers of a Levels or Curves adjustment
//! layer's Properties panel (UI-217-15, #3116), as Photoshop has them next to the histogram or
//! curve.
//!
//! Arming one turns a click on the image into one `layer.setAdjustment` with the `eyedropper`
//! param (one undoable, recordable step, the same thing an agent sends), which samples the image
//! the layer adjusts. Unlike the modal Image › Adjustments › Levels and Curves dialogs (`adjust_dialog`), the
//! panel is not modal, so the picker stays armed until Esc, another tool, another layer, or a
//! second click on its button.

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::PhotocraftApp;
use crate::state::Tool;
use photocraft_doc::{Adjustment, LayerContent, LayerId};
use photocraft_engine::adjust_params::CurvesEyedropper;

/// An armed Properties eyedropper: on `layer`, for `point` ("black", "gray" or "white"), armed
/// while `tool` was the current tool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerPicker {
    pub layer: u64,
    pub point: String,
    pub tool: Tool,
}

const POINTS: [(&str, CurvesEyedropper); 3] = [("black", CurvesEyedropper::Black), ("gray", CurvesEyedropper::NeutralGray), ("white", CurvesEyedropper::White)];

fn eyedropper(point: &str) -> Option<CurvesEyedropper> {
    POINTS.iter().find(|(id, _)| *id == point).map(|(_, e)| *e)
}

/// The armed picker if it still applies: same tool, the layer still the active Levels or Curves
/// layer, and the point supported in the document's colour mode.
pub fn armed(app: &PhotocraftApp) -> Option<(LayerId, &str)> {
    let picker = app.ui.layer_picker.as_ref()?;
    let st = app.session.active()?;
    let layer = LayerId(picker.layer);
    let tonal = st.doc.layer(layer).is_some_and(|l| matches!(l.content, LayerContent::Adjustment(Adjustment::Levels { .. } | Adjustment::Curves { .. })));
    let supported = eyedropper(&picker.point).is_some_and(|e| crate::adjust_dialog::picker_supported(st.doc.mode, e));
    (picker.tool == app.ui.tool && st.active_layer == Some(layer) && tonal && supported).then_some((layer, picker.point.as_str()))
}

/// Drops a picker that no longer applies, so returning to its tool or layer doesn't re-arm it.
pub fn prune(app: &mut PhotocraftApp) {
    if app.ui.layer_picker.is_some() && armed(app).is_none() {
        app.ui.layer_picker = None;
    }
}

/// Esc: disarms the picker. Returns whether there was one.
pub fn cancel(app: &mut PhotocraftApp) -> bool {
    app.ui.layer_picker.take().is_some()
}

/// Sets the armed point from the image at document point (x, y): one `layer.setAdjustment`.
/// Returns whether a picker was armed (and so owns the click), even if the sample failed (off
/// the image, transparent): that is reported in the status bar and the picker stays armed.
pub fn sample_at(app: &mut PhotocraftApp, x: f64, y: f64) -> bool {
    let Some((layer, point)) = armed(app).map(|(l, p)| (l, p.to_string())) else { return false };
    let size = app.ui.tool_options.eyedropper_size;
    let p = json!({"layer": layer.0, "eyedropper": {"point": point, "at": [x, y], "size": size}});
    if let Err(err) = app.run("layer.setAdjustment", p) {
        app.ui.status = err;
        app.ui.status_error = true;
    }
    true
}

/// The three eyedropper buttons above the Levels or Curves editor of adjustment layer `layer`.
pub fn controls(app: &mut PhotocraftApp, ui: &mut egui::Ui, layer: LayerId) {
    let Some(mode) = app.session.active().map(|s| s.doc.mode) else { return };
    let current = armed(app).filter(|(l, _)| *l == layer).map(|(_, p)| p.to_string());
    let mut clicked = None;
    ui.horizontal(|ui| {
        // The swatch is the tone the picker maps the sample to, as in the dialog.
        for ((id, e), (label, tone)) in POINTS.iter().zip([(tl!("Set Black Point"), 0), (tl!("Set Neutral Gray Point"), 128), (tl!("Set White Point"), 255)]) {
            let enabled = crate::adjust_dialog::picker_supported(mode, *e);
            let selected = current.as_deref() == Some(*id);
            if crate::adjust_dialog::picker_button(ui, selected, enabled, label, egui::Color32::from_gray(tone)).clicked() {
                clicked = Some((*id, selected));
            }
        }
    });
    match clicked {
        Some((_, true)) => app.ui.layer_picker = None,
        Some((id, false)) => app.ui.layer_picker = Some(LayerPicker { layer: layer.0, point: id.into(), tool: app.ui.tool }),
        None => {}
    }
}
