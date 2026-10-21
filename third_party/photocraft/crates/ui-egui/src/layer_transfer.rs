//! Dragging layers to another document (#589), as in Photoshop: drag layers from the Layers panel,
//! or with the Move tool from the canvas, onto another document's tab. The tab shows its document
//! while the pointer is over it (spring-loaded), and releasing over that document's canvas copies
//! the layers there with `layer.copyToDocument` (one history step in the destination).
//!
//! Where the copies land: a Move-tool drag puts the pixel that was grabbed under the pointer, and
//! a Layers panel drag centres the layers on it. With ⇧, or when released on the tab itself, they
//! keep their place when both documents have the same pixel size and are centred otherwise
//! (Photoshop's ⇧-drag). With Window › Arrange tiling several documents, releasing over another
//! document's tile drops there.
//!
//! A Move-tool drag starting inside a selection or moving a floating piece stays a move: Photoshop
//! would copy just the selected pixels, which is not supported yet.

use photocraft_doc::{DocId, LayerId};
use serde_json::json;

use crate::PhotocraftApp;
use crate::state::Tool;

/// The layers being dragged and where from.
#[derive(Clone, Debug)]
struct Transfer {
    source: DocId,
    layers: Vec<u64>,
    /// The document point a Move-tool drag grabbed (`None` for a Layers panel drag).
    grab: Option<[f64; 2]>,
    /// What the ghost by the pointer says.
    label: String,
}

fn key() -> egui::Id {
    egui::Id::new("pc-layer-transfer")
}

fn get(ctx: &egui::Context) -> Option<Transfer> {
    ctx.data(|d| d.get_temp::<Transfer>(key()))
}

fn clear(ctx: &egui::Context) {
    ctx.data_mut(|d| d.remove::<Transfer>(key()));
}

/// The layers `id` stands for: the selection when it is one of the selected layers, else itself.
fn begin(app: &PhotocraftApp, ctx: &egui::Context, id: LayerId, grab: Option<[f64; 2]>) -> bool {
    let Some(st) = app.session.active() else { return false };
    let layers = if st.is_layer_selected(id) { st.selected_layers() } else { vec![id] };
    let label = match layers.as_slice() {
        [one] => st.doc.layer(*one).map_or_else(String::new, |l| l.name.clone()),
        many => crate::i18n::trn(crate::i18n::current(), many.len() as u64, "{n} layer", "{n} layers"),
    };
    let t = Transfer { source: st.doc.id, layers: layers.iter().map(|l| l.0).collect(), grab, label };
    ctx.data_mut(|d| d.insert_temp(key(), t));
    true
}

/// A Layers panel row drag began on `layer`: it can be dropped on another document.
pub fn begin_from_panel(app: &PhotocraftApp, ctx: &egui::Context, layer: LayerId) {
    begin(app, ctx, layer, None);
}

/// A Move-tool drag of whole layers (not a selection or a floating piece) under way.
fn move_drag(app: &PhotocraftApp) -> Option<[f64; 2]> {
    let d = app.drag.as_ref().filter(|d| d.tool == Tool::Move && d.sel_move.is_none())?;
    let st = app.session.active()?;
    let plain = app.ui.transform.is_none() && app.guide_drag.is_none() && st.doc.selection.is_none() && st.floating.is_none();
    plain.then_some(d.start)
}

/// The pointer, while layers are dragged (or a Move drag could become a transfer): tabs only
/// hit-test it then.
pub fn pointer_if_armed(app: &PhotocraftApp, ctx: &egui::Context) -> Option<egui::Pos2> {
    if get(ctx).is_none() && move_drag(app).is_none() {
        return None;
    }
    ctx.input(|i| i.pointer.latest_pos())
}

/// The pointer is over document tab `index` while layers are dragged: a Move drag becomes a
/// transfer (the source layers stay where they were), the tab's document is shown, and a release
/// right here drops the layers into it.
pub fn over_tab(app: &mut PhotocraftApp, ctx: &egui::Context, index: usize) {
    if get(ctx).is_none() {
        let Some(grab) = move_drag(app) else { return };
        let Some(active) = app.session.active().and_then(|st| st.active_layer) else { return };
        if app.session.active_index() == Some(index) {
            return;
        }
        // The Move drag ends without moving anything; an ⌥-drag's copy is taken back.
        app.drag = None;
        app.move_preview = None;
        crate::move_mods::abandon(app);
        let active = app.session.active().and_then(|st| st.active_layer).unwrap_or(active);
        if !begin(app, ctx, active, Some(grab)) {
            return;
        }
    }
    if app.session.active_index() != Some(index) {
        app.session.set_active(index);
        app.jobs.focus = None;
        // The Layers panel now lists the other document: its rows don't take the drag.
        ctx.data_mut(|d| d.remove::<u64>(egui::Id::new("layer-drag")));
    }
    if ctx.input(|i| i.pointer.any_released()) {
        drop(app, ctx, index, None);
    }
}

/// Every frame after the document area: a release over another document's canvas drops the
/// layers there; while the button is held a ghost follows the pointer.
pub fn finish(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let Some(t) = get(ctx) else { return };
    let (pos, released, down) = ctx.input(|i| (i.pointer.latest_pos(), i.pointer.any_released(), i.pointer.primary_down()));
    let target = pos.and_then(|p| target_at(app, p));
    if !released {
        if !down {
            clear(ctx);
        } else if let Some(p) = pos
            && app.session.active().is_some_and(|st| st.doc.id != t.source)
        {
            ghost(ctx, p, &t.label);
            if target.is_some() {
                ctx.set_cursor_icon(egui::CursorIcon::Copy);
            }
        }
        return;
    }
    if let Some((dest, at)) = target {
        drop(app, ctx, dest, at);
    }
    clear(ctx);
}

/// The document under screen point `p` that isn't the source, and the document point there
/// (`None` in a tile: tiles keep their own views).
fn target_at(app: &PhotocraftApp, p: egui::Pos2) -> Option<(usize, Option<[f64; 2]>)> {
    if !app.last_canvas_rect.contains(p) {
        return None;
    }
    match crate::canvas::arranged_cells(app, app.last_canvas_rect) {
        Some(cells) => cells.into_iter().find(|(_, r)| r.contains(p)).map(|(d, _)| (d, None)),
        None => Some((app.session.active_index()?, crate::canvas::ViewXform::active(app).map(|xf| xf.to_doc(p)))),
    }
}

/// Copy the dragged layers into document `dest`, at document point `at` (see the module docs).
fn drop(app: &mut PhotocraftApp, ctx: &egui::Context, dest: usize, at: Option<[f64; 2]>) {
    let Some(t) = get(ctx) else { return };
    clear(ctx);
    let docs = app.session.documents();
    let Some(src) = docs.iter().position(|d| d.doc.id == t.source) else { return };
    let Some(to) = docs.get(dest).filter(|_| dest != src) else { return };
    let same_size = to.doc.size == docs[src].doc.size;
    let mut p = json!({"document": dest, "source": src, "layers": t.layers});
    match at.filter(|_| !ctx.input(|i| i.modifiers.shift)) {
        Some(at) => match t.grab {
            Some(g) => p["offset"] = json!([(at[0] - g[0]).round(), (at[1] - g[1]).round()]),
            None => p["at"] = json!(at),
        },
        None => p["center"] = json!(!same_size),
    }
    if let Err(e) = app.run("layer.copyToDocument", p) {
        app.ui.status = e;
        app.ui.status_error = true;
    }
}

/// The dragged layers' name next to the pointer (also the Layers panel's reorder drag).
pub fn ghost(ctx: &egui::Context, p: egui::Pos2, label: &str) {
    let t = crate::theme::Tokens::get(ctx);
    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("layer-transfer-ghost")));
    let g = painter.layout_no_wrap(label.to_string(), egui::FontId::proportional(12.0), t.text);
    let r = egui::Rect::from_min_size(p + egui::vec2(12.0, -10.0), g.size() + egui::vec2(16.0, 8.0));
    painter.rect_filled(r, t.radius_sm, t.card.gamma_multiply(0.95));
    painter.rect_stroke(r, t.radius_sm, egui::Stroke::new(1.0, t.accent), egui::StrokeKind::Inside);
    painter.galley(r.min + egui::vec2(8.0, 4.0), g, t.text);
}

#[cfg(test)]
mod tests;
