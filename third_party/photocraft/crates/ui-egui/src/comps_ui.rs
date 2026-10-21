//! Window › Layer Comps panel (a tab of the History card).
//!
//! Rows: the Last Document State, then each comp with its apply marker, name, warning triangle
//! (layers deleted since capture) and the three apply toggles (visibility, position,
//! appearance). Footer: previous / next comp, update, new, delete. Every action is a
//! `layerComp.*` engine command; the selection is `UiState::layer_comp_selected`.

use egui::{Align2, Rect, Sense, Stroke, pos2, vec2};
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::theme::Tokens;

/// The comp footer and row actions act on: the panel selection if still valid, else the last
/// applied comp.
pub fn target(app: &PhotocraftApp) -> Option<u32> {
    let doc = &app.session.active()?.doc;
    app.ui.layer_comp_selected.filter(|id| doc.comp(*id).is_some()).or(doc.last_applied_comp.filter(|id| doc.comp(*id).is_some()))
}

/// The row's warning triangle: Clear Layer Comp Warning for that comp only (Photoshop keeps
/// Clear All Layer Comp Warnings as a separate command).
fn clear_warning_action(comp: u32) -> (&'static str, Value) {
    ("layerComp.updateWarnings", json!({"clear": true, "comp": comp}))
}

fn run(app: &mut PhotocraftApp, id: &str, p: Value) {
    if let Err(e) = app.run(id, p) {
        app.ui.status = e;
        app.ui.status_error = true;
    }
}

pub fn panel(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let Some(st) = app.session.active() else {
        ui.label(egui::RichText::new(tl!("No document")).color(t.text_faint).size(11.5));
        return;
    };
    let doc = st.doc.clone();
    let selected = target(app);
    let applied = doc.last_applied_comp;
    let max_h = (ui.available_height() - 70.0).clamp(80.0, 320.0);
    let mut action: Option<(&'static str, Value)> = None;
    let mut select: Option<u32> = None;
    let row_h = 26.0;
    egui::ScrollArea::vertical().id_salt("comps-rows").max_height(max_h).auto_shrink([false, true]).show(ui, |ui| {
        // Last Document State.
        let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), row_h), Sense::click());
        if applied.is_none() {
            ui.painter().rect_filled(rect, 0.0, t.row_selected);
        } else if resp.hovered() {
            ui.painter().rect_filled(rect, 0.0, t.hover.gamma_multiply(0.5));
        }
        let marker = Rect::from_center_size(pos2(rect.left() + 14.0, rect.center().y), vec2(12.0, 12.0));
        ui.painter().rect_stroke(marker, 2.0, Stroke::new(1.0, t.text_faint), egui::StrokeKind::Inside);
        if applied.is_none() {
            crate::icons::paint(ui, marker, "check", 10.0, t.icon);
        }
        ui.painter().text(
            pos2(rect.left() + 30.0, rect.center().y),
            Align2::LEFT_CENTER,
            tl!("Last Document State"),
            egui::FontId::proportional(12.0),
            t.text_dim,
        );
        if resp.clicked() && doc.last_document_state.is_some() && applied.is_some() {
            action = Some(("layerComp.restoreLastDocumentState", json!({})));
        }
        ui.painter().line_segment([rect.left_bottom(), rect.right_bottom()], Stroke::new(1.0, t.separator));

        if doc.layer_comps.is_empty() {
            ui.add_space(6.0);
            ui.label(
                egui::RichText::new(tl!("Capture the visibility, position and style of every layer with + below, then switch between versions."))
                    .color(t.text_faint)
                    .size(11.5),
            );
        }
        for c in &doc.layer_comps {
            let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), row_h), Sense::click());
            if selected == Some(c.id) {
                ui.painter().rect_filled(rect, 0.0, t.row_selected);
            } else if resp.hovered() {
                ui.painter().rect_filled(rect, 0.0, t.hover.gamma_multiply(0.5));
            }
            // Apply marker (click to apply).
            let marker = Rect::from_center_size(pos2(rect.left() + 14.0, rect.center().y), vec2(12.0, 12.0));
            ui.painter().rect_stroke(marker, 2.0, Stroke::new(1.0, t.text_faint), egui::StrokeKind::Inside);
            if applied == Some(c.id) {
                crate::icons::paint(ui, marker, "check", 10.0, t.accent);
            }
            // The three apply toggles, right-aligned.
            let toggles = [
                ("eye", tl!("Visibility"), c.apply_visibility, "visibility"),
                ("move", tl!("Position"), c.apply_position, "position"),
                ("sparkles", tl!("Appearance (Layer Style)"), c.apply_appearance, "appearance"),
            ];
            let mut x = rect.right() - 14.0;
            let mut toggle_hit = false;
            for (icon, tip, on, key) in toggles.iter().rev() {
                let r = Rect::from_center_size(pos2(x, rect.center().y), vec2(20.0, 20.0));
                let hov = ui.rect_contains_pointer(r);
                if hov {
                    ui.painter().rect_filled(r, 3.0, t.hover);
                }
                crate::icons::paint(ui, r, icon, 13.0, if *on { t.icon } else { t.text_faint.gamma_multiply(0.45) });
                if hov && resp.clicked() {
                    toggle_hit = true;
                    action = Some(("layerComp.setOptions", json!({"comp": c.id, *key: !on})));
                }
                if hov {
                    let template = "Apply {option}: {state}";
                    let state = if *on { "on" } else { "off" };
                    resp.clone().on_hover_text(crate::i18n::fmt(tl!(template), &[("option", tip), ("state", tl!(state))]));
                }
                x -= 22.0;
            }
            let missing = c.missing_layers(&doc).len();
            let mut name_x = rect.left() + 30.0;
            if missing > 0 {
                let r = Rect::from_center_size(pos2(name_x + 7.0, rect.center().y), vec2(14.0, 14.0));
                crate::icons::paint(ui, r, "triangle-alert", 12.0, t.warning);
                if ui.rect_contains_pointer(r) {
                    resp.clone().on_hover_text(crate::i18n::trn(
                        crate::i18n::current(),
                        missing as u64,
                        "{n} layer recorded by this comp was deleted. Update the comp, or clear the warning.",
                        "{n} layers recorded by this comp were deleted. Update the comp, or clear the warning.",
                    ));
                    if resp.clicked() {
                        toggle_hit = true;
                        action = Some(clear_warning_action(c.id));
                    }
                }
                name_x += 18.0;
            }
            let galley_w = (x - name_x).max(20.0);
            let name = ui.painter().layout(c.name.clone(), egui::FontId::proportional(12.0), t.text, galley_w);
            ui.painter().galley(pos2(name_x, rect.center().y - name.size().y / 2.0), name, t.text);
            if !c.comment.is_empty() && resp.hovered() {
                resp.clone().on_hover_text(&c.comment);
            }
            if resp.clicked() && !toggle_hit {
                select = Some(c.id);
                if resp.interact_pointer_pos().is_some_and(|p| p.x < rect.left() + 24.0) {
                    action = Some(("layerComp.apply", json!({"comp": c.id})));
                }
            }
            if resp.double_clicked() && !toggle_hit {
                action = Some(("layerComp.apply", json!({"comp": c.id})));
            }
            ui.painter().line_segment([rect.left_bottom(), rect.right_bottom()], Stroke::new(1.0, t.separator.gamma_multiply(0.5)));
        }
    });
    // Footer: apply previous, apply next | update, new, delete (Photoshop's order).
    ui.add_space(4.0);
    let has = !doc.layer_comps.is_empty();
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        if crate::icons::button(ui, "chevrons-left", 24.0, false, tl!("Apply previous selected layer comp")).clicked() && has {
            action = Some(("layerComp.previous", json!({})));
        }
        if crate::icons::button(ui, "chevrons-right", 24.0, false, tl!("Apply next selected layer comp")).clicked() && has {
            action = Some(("layerComp.next", json!({})));
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if crate::icons::button(ui, "trash", 24.0, false, tl!("Delete layer comp")).clicked()
                && let Some(id) = selected
            {
                action = Some(("layerComp.delete", json!({"comp": id})));
            }
            if crate::icons::button(ui, "file-plus", 24.0, false, tl!("Create new layer comp")).clicked() {
                action = Some(("layerComp.new", json!({})));
            }
            if crate::icons::button(ui, "rotate-cw", 24.0, false, tl!("Update layer comp (visibility, position and appearance)")).clicked()
                && let Some(id) = selected
            {
                action = Some(("layerComp.update", json!({"comp": id})));
            }
        });
    });
    if let Some(id) = select {
        app.ui.layer_comp_selected = Some(id);
    }
    if let Some((cmd, p)) = action {
        let created = cmd == "layerComp.new";
        run(app, cmd, p);
        if created {
            app.ui.layer_comp_selected = app.session.active().and_then(|d| d.doc.last_applied_comp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_engine::Session;

    #[test]
    fn warning_icon_clears_only_its_own_comp() {
        // Regression for #4197.
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 16, "height": 16})).unwrap();
        let layer = s.execute("layer.new.layer", json!({})).unwrap()["layer"].as_u64().unwrap();
        let c1 = s.execute("layerComp.new", json!({"name": "One"})).unwrap()["comp"].as_u64().unwrap() as u32;
        let c2 = s.execute("layerComp.new", json!({"name": "Two"})).unwrap()["comp"].as_u64().unwrap() as u32;
        s.execute("layer.delete", json!({"layer": layer})).unwrap();
        let (id, p) = clear_warning_action(c1);
        assert_eq!(p["comp"], c1);
        s.execute(id, p).unwrap();
        let doc = &s.active().unwrap().doc;
        assert!(doc.comp(c1).unwrap().missing_layers(doc).is_empty());
        assert_eq!(doc.comp(c2).unwrap().missing_layers(doc).iter().map(|l| l.0).collect::<Vec<_>>(), vec![layer]);
    }
}
