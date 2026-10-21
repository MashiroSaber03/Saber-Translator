//! The Solid Color Fill thumbnail opens the shared Color Picker. Preview a shallow document
//! snapshot; Cancel changes nothing and OK commits exactly one engine command.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use egui::{Color32, Rect, Stroke, StrokeKind, pos2, vec2};
use photocraft_color::Color;
use photocraft_doc::{DocId, Document, Fill, LayerContent, LayerId};
use photocraft_engine::solid_fill_cmds as cmds;
use serde_json::{Map, Value, json};

use crate::{PhotocraftApp, color_picker_ui, theme::Tokens};

pub(crate) struct Preview {
    document: DocId,
    revision: u64,
    dialog: u64,
    layer: LayerId,
    color: Color,
    shown: Arc<Document>,
    key: u64,
}

pub(crate) fn open(app: &mut PhotocraftApp) -> Option<u64> {
    let st = app.session.active()?;
    let layer = st.active_layer?;
    let LayerContent::Fill(Fill::Solid(color)) = st.doc.layer(layer)?.content else { return None };
    let params = json!({"layer": layer.0, "document": st.doc.id.0});
    Some(color_picker_ui::open_for_command(app, "Color Picker (Solid Color)", color.to_rgb(), cmds::SET, params))
}

pub(crate) fn owns(f: &Map<String, Value>) -> bool {
    f.get("__command").and_then(Value::as_str) == Some(cmds::SET)
}

pub(crate) fn thumbnail_color(app: &PhotocraftApp, document: DocId, layer: LayerId, original: Color) -> Color {
    app.ui
        .dialogs
        .iter()
        .rev()
        .filter(|d| owns(&d.fields))
        .find_map(|d| {
            let params = d.fields.get("__params")?;
            if params.get("document")?.as_u64()? != document.0 || params.get("layer")?.as_u64()? != layer.0 || d.fields.get("color") == d.fields.get("__orig") {
                return None;
            }
            cmds::color_param(d.fields.get("color")?, original, app.session.documents().iter().find(|st| st.doc.id == document)?.doc.mode).ok()
        })
        .unwrap_or(original)
}

/// No-op OK preserves imported high-precision / non-RGB colour values rather than quantising
/// them to the picker's display hex. A stale document must never receive the edit.
pub(crate) fn confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    let mut params = f.get("__params").cloned().ok_or("missing Solid Color target")?;
    let st = app.session.active().ok_or("no document open")?;
    if params.get("document").and_then(Value::as_u64) != Some(st.doc.id.0) {
        return Err("the target document is no longer active".into());
    }
    let layer = LayerId(params.get("layer").and_then(Value::as_u64).ok_or("missing Solid Color layer")?);
    if !st.doc.layer(layer).is_some_and(|l| matches!(l.content, LayerContent::Fill(Fill::Solid(_)))) {
        return Err("the target Solid Color Fill layer no longer exists".into());
    }
    if f.get("color") == f.get("__orig") {
        return Ok(json!({"layer": layer.0, "unchanged": true}));
    }
    let color = f.get("color").cloned().ok_or("missing color")?;
    let object = params.as_object_mut().ok_or("invalid Solid Color target")?;
    object.insert("color".into(), color);
    app.run(cmds::SET, params)
}

pub(crate) fn display_doc(app: &mut PhotocraftApp, idx: usize) -> Option<(Arc<Document>, u64)> {
    let dialog = app.ui.dialogs.iter().rev().find(|d| owns(&d.fields));
    let Some(d) = dialog else {
        app.solid_fill_preview = None;
        return None;
    };
    let st = app.session.documents().get(idx)?;
    let params = d.fields.get("__params")?;
    if params.get("document")?.as_u64()? != st.doc.id.0 || d.fields.get("color") == d.fields.get("__orig") {
        return None;
    }
    let layer = LayerId(params.get("layer")?.as_u64()?);
    let LayerContent::Fill(Fill::Solid(original)) = st.doc.layer(layer)?.content else { return None };
    let color = cmds::color_param(d.fields.get("color")?, original, st.doc.mode).ok()?;
    if let Some(p) = &app.solid_fill_preview
        && p.document == st.doc.id
        && p.revision == st.revision
        && p.dialog == d.id
        && p.layer == layer
        && p.color == color
    {
        return Some((p.shown.clone(), p.key));
    }
    let mut doc = (*st.doc).clone();
    doc.layer_mut(layer)?.content = LayerContent::Fill(Fill::Solid(color));
    // Include the dialog identity: cancelling and reopening at the same document revision
    // must not reuse an old preview's pixels in the canvas caches.
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    (st.doc.id.0, st.revision, d.id, layer.0).hash(&mut hash);
    color.mode.hash(&mut hash);
    for component in color.c.into_iter().chain([color.alpha]) {
        component.to_bits().hash(&mut hash);
    }
    let key = (1 << 36) | (hash.finish() & ((1 << 36) - 1));
    let shown = Arc::new(photocraft_engine::mode_cmds::display_document(&doc).unwrap_or(doc));
    app.solid_fill_preview = Some(Preview { document: st.doc.id, revision: st.revision, dialog: d.id, layer, color, shown: shown.clone(), key });
    Some((shown, key))
}

/// Clean-room drawing of Photoshop's Solid Color thumbnail: an inset colour sample with the
/// small fill marker underneath, rather than a raster thumbnail covering the whole tile.
pub(crate) fn paint_thumbnail(ui: &egui::Ui, color: Color, rect: Rect) {
    thumbnail_frame(ui, rect, |swatch| {
        let [r, g, b, a] = color.to_rgba8();
        crate::widgets::checker(ui.painter(), swatch, 3.0);
        ui.painter().rect_filled(swatch, 0.0, Color32::from_rgba_unmultiplied(r, g, b, a));
    });
}

/// Shared fill-layer thumbnail frame and marker, drawn around the inset contents.
pub(crate) fn thumbnail_frame(ui: &egui::Ui, rect: Rect, contents: impl FnOnce(Rect)) {
    let t = Tokens::get(ui.ctx());
    let p = ui.painter();
    p.rect_filled(rect, if t.pro { 0.0 } else { t.radius_sm }, t.field);
    let size = rect.size() * 0.7;
    let sample = Rect::from_center_size(rect.center() - vec2(0.0, size.y * 0.08), size);
    let bar_height = (size.y * 0.17).max(2.0);
    let swatch = Rect::from_min_max(sample.min, pos2(sample.right(), sample.bottom() - bar_height));
    contents(swatch);
    p.rect_stroke(sample, 0.0, Stroke::new(1.0, t.text), StrokeKind::Outside);
    let y = swatch.bottom();
    p.line_segment([pos2(sample.left(), y), pos2(sample.right(), y)], Stroke::new(1.0, t.text));
    let x = sample.center().x;
    p.add(egui::Shape::convex_polygon(
        vec![pos2(x, y + 1.0), pos2(x - bar_height * 0.6, sample.bottom()), pos2(x + bar_height * 0.6, sample.bottom())],
        t.text,
        Stroke::NONE,
    ));
}

#[cfg(test)]
mod tests;
