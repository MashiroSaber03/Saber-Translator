//! Properties panel for pixel (and other bounded) layers, Photoshop 2026 style: a Transform section
//! (W/H/X/Y of the layer content, editable), Align and Distribute, the kind's sections and Quick Actions. Edits dispatch
//! `edit.transform`, `layer.translate`, `layer.align.*` and selection commands.

use egui::{Rect, Sense, Stroke, pos2, vec2};
use photocraft_doc::{Layer, LayerContent};
use serde_json::{Value, json};

use crate::props_layout::{LABEL_GAP, LABEL_W, quick_actions_ui, section};
use crate::theme::{ROW_GAP, Tokens};
use crate::{PhotocraftApp, widgets};

/// `edit.transform` params that scale the content box `b` = [x0, y0, x1, y1] to `w` x `h`, keeping
/// the top-left corner (Photoshop's Properties W/H fields), or `layer.translate` params for X/Y.
pub fn transform_params(layer: u64, b: [i32; 4], w: Option<f32>, h: Option<f32>, linked: bool) -> Option<Value> {
    let (bw, bh) = ((b[2] - b[0]) as f32, (b[3] - b[1]) as f32);
    if bw <= 0.0 || bh <= 0.0 {
        return None;
    }
    let (mut nw, mut nh) = (w.unwrap_or(bw).max(1.0), h.unwrap_or(bh).max(1.0));
    if linked {
        if w.is_some() {
            nh = (nw * bh / bw).max(1.0);
        } else if h.is_some() {
            nw = (nh * bw / bh).max(1.0);
        }
    }
    if (nw - bw).abs() < 0.5 && (nh - bh).abs() < 0.5 {
        return None;
    }
    let (x0, y0) = (b[0] as f32, b[1] as f32);
    let quad = [[x0, y0], [x0 + nw, y0], [x0 + nw, y0 + nh], [x0, y0 + nh]];
    Some(json!({"layer": layer, "rect": b, "quad": quad}))
}

/// A number field that reports a value once committed (drag released, Enter, focus lost).
fn field(ui: &mut egui::Ui, id: &str, label: &str, current: f32, width: f32, u: LenUnit, extent: f64) -> Option<f32> {
    let t = Tokens::get(ui.ctx());
    let (r, _) = ui.allocate_exact_size(vec2(LABEL_W, 22.0), Sense::hover());
    ui.painter().text(pos2(r.right() - 2.0, r.center().y), egui::Align2::RIGHT_CENTER, tl!(label), egui::FontId::proportional(12.0), t.text_dim);
    let key = ui.make_persistent_id(("layer-props-field", id));
    let mut v = ui.data(|d| d.get_temp::<f32>(key)).unwrap_or_else(|| u.shown(f64::from(current), extent));
    let resp = widgets::value_field(ui, &mut v, -300_000.0..=300_000.0, u.unit.suffix(), width);
    if resp.dragged() || resp.has_focus() {
        ui.data_mut(|d| d.insert_temp(key, v));
        return None;
    }
    ui.data_mut(|d| d.remove::<f32>(key));
    if !(resp.drag_stopped() || resp.lost_focus() || resp.changed()) {
        return None;
    }
    let px = u.px(v, extent) as f32;
    ((px - current).abs() >= 0.5).then_some(px)
}

/// Lengths in the Units & Rulers unit, as Photoshop's Properties panel shows them (cm when the
/// rulers are in centimeters): document px ↔ that unit at the document's resolution.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LenUnit {
    pub unit: photocraft_engine::prefs::Unit,
    pub dpi: f64,
    pub ppi: f64,
}

impl LenUnit {
    pub(crate) fn of(app: &PhotocraftApp) -> Self {
        let ur = &app.session.prefs().units_and_rulers;
        let dpi = app.session.active().map_or(72.0, |s| f64::from(s.doc.resolution_dpi));
        LenUnit { unit: ur.rulers, dpi, ppi: ur.point_size.per_inch() }
    }
    /// `px` shown in the unit (`extent` is 100% for percent).
    pub(crate) fn shown(&self, px: f64, extent: f64) -> f32 {
        self.unit.from_px(px, self.dpi, extent, self.ppi) as f32
    }
    /// A value typed in the unit, in document px.
    pub(crate) fn px(&self, v: f32, extent: f64) -> f64 {
        self.unit.to_px(f64::from(v), self.dpi, extent, self.ppi)
    }
}

/// Align glyph: a reference line and two bars (Photoshop's align icons).
fn align_glyph(ui: &egui::Ui, r: Rect, kind: &str, color: egui::Color32) {
    let p = ui.painter();
    let c = r.center();
    let s = Stroke::new(1.2, color);
    let vertical_ref = matches!(kind, "leftEdges" | "horizontalCenters" | "rightEdges");
    if vertical_ref {
        let x = match kind {
            "leftEdges" => c.x - 6.0,
            "rightEdges" => c.x + 6.0,
            _ => c.x,
        };
        p.line_segment([pos2(x, c.y - 7.0), pos2(x, c.y + 7.0)], s);
        for (dy, w) in [(-3.0, 10.0), (3.0, 6.0)] {
            let x0 = match kind {
                "leftEdges" => x,
                "rightEdges" => x - w,
                _ => x - w / 2.0,
            };
            p.rect_filled(Rect::from_min_size(pos2(x0, c.y + dy - 1.5), vec2(w, 3.0)), 0.0, color);
        }
    } else {
        let y = match kind {
            "topEdges" => c.y - 6.0,
            "bottomEdges" => c.y + 6.0,
            _ => c.y,
        };
        p.line_segment([pos2(c.x - 7.0, y), pos2(c.x + 7.0, y)], s);
        for (dx, h) in [(-3.0, 10.0), (3.0, 6.0)] {
            let y0 = match kind {
                "topEdges" => y,
                "bottomEdges" => y - h,
                _ => y - h / 2.0,
            };
            p.rect_filled(Rect::from_min_size(pos2(c.x + dx - 1.5, y0), vec2(3.0, h)), 0.0, color);
        }
    }
}

/// Width of the link toggle between the W and H fields (the X/Y row leaves the same gap).
const LINK_W: f32 = 20.0;

/// Pro Properties body for a non-adjustment layer: Transform, Align and Distribute, the kind's own
/// sections (Character/Paragraph/Type Options for type, Appearance/Shape for shapes), then Quick
/// Actions, all with the same collapsible headers (#155).
pub fn properties(app: &mut PhotocraftApp, ui: &mut egui::Ui, layer: &Layer) {
    let mut run: Vec<(String, Value)> = Vec::new();
    if let Some(s) = layer.surface() {
        let b = app.cached_bounds(layer.id.0, s);
        let u = LenUnit::of(app);
        let (dw, dh) = app.session.active().map_or((1.0, 1.0), |s| (f64::from(s.doc.size.width), f64::from(s.doc.size.height)));
        if section(ui, "transform", tl!("Transform")) {
            let link_key = egui::Id::new("layer-props-link");
            let linked = ui.data(|d| d.get_temp::<bool>(link_key)).unwrap_or(true);
            let bb = [b.x0, b.y0, b.x1, b.y1];
            // Two label+field columns with the link toggle between them, filling the panel.
            let w = ((ui.available_width() - 2.0 * (LABEL_W + LABEL_GAP) - LINK_W - 2.0 * LABEL_GAP) / 2.0).max(36.0);
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = LABEL_GAP;
                if let Some(v) = field(ui, "w", "W", b.width() as f32, w, u, dw) {
                    run.extend(transform_params(layer.id.0, bb, Some(v), None, linked).map(|p| ("edit.transform".to_string(), p)));
                }
                if crate::icons::button(ui, if linked { "link" } else { "unlink" }, LINK_W, linked, tl!("Link width and height")).clicked() {
                    ui.data_mut(|d| d.insert_temp(link_key, !linked));
                }
                if let Some(v) = field(ui, "h", "H", b.height() as f32, w, u, dh) {
                    run.extend(transform_params(layer.id.0, bb, None, Some(v), linked).map(|p| ("edit.transform".to_string(), p)));
                }
            });
            ui.add_space(ROW_GAP);
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = LABEL_GAP;
                if let Some(v) = field(ui, "x", "X", b.x0 as f32, w, u, dw) {
                    run.push(("layer.translate".into(), json!({"layer": layer.id.0, "dx": (v - b.x0 as f32).round() as i32, "dy": 0})));
                }
                ui.add_space(LINK_W + LABEL_GAP);
                if let Some(v) = field(ui, "y", "Y", b.y0 as f32, w, u, dh) {
                    run.push(("layer.translate".into(), json!({"layer": layer.id.0, "dx": 0, "dy": (v - b.y0 as f32).round() as i32})));
                }
            });
            ui.add_space(ROW_GAP);
        }
    }
    // Gradient fills: their gradient first, as in Photoshop's Properties panel.
    let gradient = matches!(layer.content, LayerContent::Fill(photocraft_doc::Fill::Gradient { .. }));
    if gradient {
        crate::gradient_ui::properties(app, ui, layer);
    }
    if !gradient {
        align_and_distribute(app, ui, &mut run);
    }
    for (id, p) in run.drain(..) {
        if let Err(e) = app.run(&id, p) {
            app.ui.status = e;
            app.ui.status_error = true;
        }
    }
    match &layer.content {
        LayerContent::Text(_) => crate::type_tool::type_properties(app, ui),
        LayerContent::Shape(_) => crate::vector_ui::shape_properties(app, ui, layer.id),
        _ => {}
    }
    // Re-read: a section above may have changed the layer (e.g. point to paragraph type).
    let content = app.session.active().and_then(|s| s.doc.layer(layer.id)).map_or_else(|| layer.content.clone(), |l| l.content.clone());
    if let Some(id) = quick_actions_ui(app, ui, &content) {
        let ctx = ui.ctx().clone();
        if let Err(e) = crate::menus::invoke(app, &ctx, id, json!({})) {
            app.ui.status = e;
            app.ui.status_error = true;
        }
    }
}

/// The Align and Distribute section: its buttons run `layer.align.*` and `layer.distribute.*` on
/// the selected layers (pushed onto `run`), greyed where the engine says they can't.
fn align_and_distribute(app: &PhotocraftApp, ui: &mut egui::Ui, run: &mut Vec<(String, Value)>) {
    let t = Tokens::get(ui.ctx());
    if !section(ui, "align", tl!("Align and Distribute")) {
        return;
    }
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        for (i, kind) in ["leftEdges", "horizontalCenters", "rightEdges", "topEdges", "verticalCenters", "bottomEdges"].into_iter().enumerate() {
            if i == 3 {
                ui.add_space(8.0);
            }
            let id = format!("layer.align.{kind}");
            let on = app.session.is_enabled(&id);
            let label = photocraft_engine::commands::find(&id).map_or(kind, |c| c.label);
            let (r, resp) = ui.allocate_exact_size(vec2(24.0, 24.0), if on { Sense::click() } else { Sense::hover() });
            if resp.hovered() && on {
                ui.painter().rect_filled(r, t.radius_sm, t.hover);
            }
            align_glyph(ui, r, kind, if on { t.icon } else { t.text_faint });
            let tip = crate::i18n::fmt(tl!("Align {what}"), &[("what", tl!(&label))]);
            resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, on, &tip));
            if resp.on_hover_text(tip).clicked() {
                run.push((id, json!({})));
            }
        }
    });
    // The engine already supports equal-gap distribution in both axes. Make those
    // existing commands discoverable here as well as through Layer > Distribute (#2349).
    // Three or more selected layers are required; the engine remains authoritative.
    ui.horizontal_wrapped(|ui| {
        for (kind, text) in [("horizontally", tl!("Horizontally")), ("vertically", tl!("Vertically"))] {
            let id = format!("layer.distribute.{kind}");
            if ui.add_enabled(app.session.is_enabled(&id), egui::Button::new(text)).clicked() {
                run.push((id, json!({})));
            }
        }
    });
    ui.add_space(ROW_GAP);
}

/// Several layers selected with the Background active: Photoshop's Properties keeps Align and
/// Distribute for the selection there, above the Document properties (#4608).
pub fn selection_align(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let mut run = Vec::new();
    align_and_distribute(app, ui, &mut run);
    for (id, p) in run {
        if let Err(e) = app.run(&id, p) {
            app.ui.status = e;
            app.ui.status_error = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switching_between_gradient_and_raster_keeps_alignment_on_raster_only() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 200, "height": 120})).unwrap();
        let mut h = egui_kittest::Harness::builder().with_size(vec2(280.0, 900.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                if ui.ctx().cumulative_pass_nr() == 0 {
                    PhotocraftApp::setup_context(ui.ctx(), crate::theme::ThemeKind::ProMedium);
                    return;
                }
                let state = app.session.active().unwrap();
                let layer = state.doc.layer(state.active_layer.unwrap()).unwrap().clone();
                properties(app, ui, &layer);
            },
            app,
        );
        for gradient in [false, true, false] {
            if gradient {
                h.state_mut().run("gradient.fill.create", json!({"from": [20, 60], "to": [180, 60]})).unwrap();
            } else {
                h.state_mut().run("layer.new.layer", json!({})).unwrap();
            }
            h.run_steps(3);
            let sections = crate::props_layout::sections_drawn(&h.ctx);
            assert_eq!(sections.iter().any(|(title, _)| title == "Align and Distribute"), !gradient);
            assert_eq!(sections.iter().any(|(title, _)| title == "Gradient"), gradient);
            assert_eq!(sections.iter().any(|(title, _)| title == "Gradient Options"), gradient);
            assert!(!sections.iter().any(|(title, _)| title == "Color stops" || title == "Opacity stops"));
        }
    }

    #[test]
    fn transform_fields_show_and_take_the_ruler_unit() {
        use photocraft_engine::prefs::Unit;
        // 72 dpi: 1644 px = 58 cm; typing 60 cm sets 1700.8 px
        let u = LenUnit { unit: Unit::Centimeters, dpi: 72.0, ppi: 72.0 };
        assert!((u.shown(1644.0, 1.0) - 58.0).abs() < 0.01);
        assert!((u.px(60.0, 1.0) - 1700.787).abs() < 0.01);
        let px = LenUnit { unit: Unit::Pixels, dpi: 150.0, ppi: 72.0 };
        assert_eq!(px.shown(170.0, 1.0), 170.0);
        assert_eq!(px.px(170.0, 1.0), 170.0);
        let pc = LenUnit { unit: Unit::Percent, dpi: 72.0, ppi: 72.0 };
        assert_eq!(pc.shown(822.0, 1644.0), 50.0);
    }

    #[test]
    fn width_edit_scales_from_the_top_left() {
        let p = transform_params(7, [10, 20, 110, 70], Some(200.0), None, true).unwrap();
        assert_eq!(p["layer"], 7);
        assert_eq!(p["quad"], json!([[10.0, 20.0], [210.0, 20.0], [210.0, 120.0], [10.0, 120.0]]));
        let p = transform_params(7, [10, 20, 110, 70], None, Some(25.0), false).unwrap();
        assert_eq!(p["quad"][2], json!([110.0, 45.0]));
        assert!(transform_params(7, [10, 20, 110, 70], Some(100.0), None, true).is_none());
        assert!(transform_params(7, [0, 0, 0, 0], Some(5.0), None, true).is_none());
    }

    #[test]
    fn transform_params_resize_a_layer_in_the_engine() {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 200, "height": 200, "background": "transparent"})).unwrap();
        s.execute("select.rect", json!({"x": 10, "y": 20, "width": 100, "height": 50})).unwrap();
        s.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
        s.execute("select.deselect", json!({})).unwrap();
        let st = s.active().unwrap();
        let id = st.active_layer.unwrap();
        let p = transform_params(id.0, [10, 20, 110, 70], Some(50.0), None, true).unwrap();
        s.execute("edit.transform", p).unwrap();
        let b = s.active().unwrap().doc.layer(id).unwrap().surface().unwrap().content_bounds();
        assert_eq!((b.x0, b.y0, b.width(), b.height()), (10, 20, 50, 25));
    }
}
