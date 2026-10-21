//! Non-destructive layer-mask controls (pixel and vector), shared by docked and floating Properties.

use photocraft_doc::Layer;
use serde_json::json;

use crate::props_layout::COL_GAP;
use crate::theme::{self, Tokens};
use crate::{PhotocraftApp, widgets};

/// Image › Adjustments › Invert, which inverts a targeted layer mask in place (#780).
const INVERT: &str = "image.adjustments.invert";

/// The Layers panel targets the active layer's pixel mask (not its vector mask) and Properties
/// shows it: a just-created adjustment or fill layer shows its own settings instead (#3369).
pub fn targeted(app: &PhotocraftApp, layer: &Layer) -> bool {
    app.ui.mask_target && !app.ui.vector_mask_target && layer.mask.is_some() && app.ui.properties_layer != Some(layer.id.0)
}

/// The Layers panel targets the active layer's vector mask.
pub fn vector_targeted(app: &PhotocraftApp, layer: &Layer) -> bool {
    app.ui.vector_mask_target && layer.vector_mask.is_some()
}

/// The pixel-mask section: Density, Feather and the Refine row. Pixels stay intact; the two
/// properties ride on the mask and are applied by the compositor.
pub fn properties(app: &mut PhotocraftApp, ui: &mut egui::Ui, layer: &Layer) {
    let Some(mask) = &layer.mask else { return };
    let Some(document) = app.session.active().map(|s| s.doc.id.0) else { return };
    ui.push_id((document, layer.id, "pixel-mask-properties"), |ui| {
        if !crate::props_layout::section(ui, "pixel-mask", "Layer Mask") {
            return;
        }
        let (density, feather) = (mask.density, mask.feather);
        sliders(app, ui, layer.id.0, document, "layer.layerMask.edit", "pixel-mask", density, feather);
        refine_row(app, ui);
    });
}

/// The vector-mask section: Density and Feather, likewise non-destructive.
pub fn vector_properties(app: &mut PhotocraftApp, ui: &mut egui::Ui, layer: &Layer) {
    let Some(vm) = &layer.vector_mask else { return };
    let Some(document) = app.session.active().map(|s| s.doc.id.0) else { return };
    ui.push_id((document, layer.id, "vector-mask-properties"), |ui| {
        if !crate::props_layout::section(ui, "vector-mask", "Vector Mask") {
            return;
        }
        let (density, feather) = (vm.density, vm.feather);
        sliders(app, ui, layer.id.0, document, "layer.vectorMask.edit", "vector-mask", density, feather);
    });
}

/// Density (%) and Feather (px) sliders writing one undoable edit per drag: the value lives on the
/// mask, not in its pixels. `command` is the mask's edit command and `prefix` keys its coalescing.
#[allow(clippy::too_many_arguments)]
fn sliders(app: &mut PhotocraftApp, ui: &mut egui::Ui, layer: u64, document: u64, command: &str, prefix: &str, density: f32, feather: f32) {
    // Like brush settings, track pointer presses even on frames without a value change.
    // slider_row also merges numeric scrubbing, whose drag flags are not in its Response.
    let key = ui.id().with("gesture");
    let pass = ui.ctx().cumulative_pass_nr();
    let pressed = ui.input(|i| i.pointer.any_pressed());
    let gesture = ui.data_mut(|d| {
        let g = d.get_temp_mut_or_default::<(u64, u64)>(key);
        if pressed && g.1 != pass {
            g.0 = g.0.wrapping_add(1);
            g.1 = pass;
        }
        g.0
    });
    for (field, label, mut value, max, unit) in [("density", tl!("Density"), density * 100.0, 100.0, "%"), ("feather", tl!("Feather"), feather, 1000.0, "px")] {
        let response = widgets::slider_row(ui, label, &mut value, 0.0..=max, unit, None);
        response.widget_info(|| egui::WidgetInfo::slider(ui.is_enabled(), f64::from(value), label));
        if response.changed() {
            let mut params = json!({"document": document, "layer": layer,
                "coalesce": format!("{prefix}:{document}:{layer}:{field}:{gesture}")});
            params[field] = json!(value);
            if let Err(error) = app.run(command, params) {
                app.ui.status = error;
                app.ui.status_error = true;
            }
            if !ui.input(|i| i.pointer.any_down()) {
                ui.data_mut(|d| {
                    let g = d.get_temp_mut_or_default::<(u64, u64)>(key);
                    g.0 = g.0.wrapping_add(1);
                });
            }
        }
    }
}

/// Photoshop's Refine row below Density and Feather: Select and Mask… and Color Range… open their
/// dialogs (the mask is Select and Mask's input when nothing is selected), and Invert flips the
/// mask. Invert names its target itself (`"target":"mask"`) rather than leaning on the shell's
/// target routing, so it edits the mask and never the layer's pixels (#1137).
fn refine_row(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let refine: [(&str, &str); 3] = [(tl!("Select and Mask…"), "select.selectAndMask"), (tl!("Color Range…"), "select.colorRange"), (tl!("Invert"), INVERT)];
    ui.add_space(theme::ROW_GAP / 2.0);
    ui.label(egui::RichText::new(tl!("Refine")).color(Tokens::get(ui.ctx()).text_dim));
    ui.add_space(theme::ROW_GAP / 2.0);
    let avail = ui.available_width();
    // Two buttons to a row in a wide panel, as Quick Actions do; a narrow one stacks them.
    let cols = if avail >= 360.0 { 2 } else { 1 };
    for row in refine.chunks(cols) {
        // A short last row (or a lone button) still spans the panel.
        let w = ((avail - (row.len() - 1) as f32 * COL_GAP) / row.len() as f32).floor();
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = COL_GAP;
            for &(label, id) in row {
                // Greyed when its command can't run, by the same rule as the menu item.
                let enabled = crate::menus::is_enabled(app, id);
                if ui.add_enabled_ui(enabled, |ui| widgets::secondary_button(ui, label, w).clicked()).inner {
                    let params = if id == INVERT { json!({"target": "mask"}) } else { json!({}) };
                    if let Err(error) = crate::menus::invoke(app, ui.ctx(), id, params) {
                        app.ui.status = error;
                        app.ui.status_error = true;
                    }
                }
            }
        });
        ui.add_space(theme::ROW_GAP);
    }
}
