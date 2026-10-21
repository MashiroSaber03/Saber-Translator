//! Layers panel tree: the rows in display order (top of the stack first, each group directly
//! above its contents) and the groups' disclosure triangles (#126).
//!
//! Clicking a triangle opens or closes the group; ⌥-click sets every group in the document to
//! the new state (Photoshop). The state is the group's `expanded` flag, document data saved in
//! `.pcraft` and as the PSD section divider's open/closed folder type, toggled through
//! `layer.setExpanded` (a view change: no history step).

use egui::{Rect, Sense, Shape, Stroke, pos2, vec2};
use photocraft_doc::{Document, Layer, LayerContent};
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::theme::Tokens;

/// Width the disclosure triangle takes in a group row.
pub const TRIANGLE_W: f32 = 14.0;

/// Rows as the Layers panel lists them: topmost layer first, a group directly above its
/// children, and the contents of closed groups left out (unless `show_all`, used while the
/// panel is filtered by kind so matches inside closed groups still show).
pub fn display_rows(doc: &Document, show_all: bool) -> Vec<(usize, &Layer)> {
    fn rec<'a>(layers: &'a [Layer], depth: usize, show_all: bool, out: &mut Vec<(usize, &'a Layer)>) {
        for l in layers.iter().rev() {
            out.push((depth, l));
            if let LayerContent::Group(g) = &l.content
                && (g.expanded || show_all)
            {
                rec(&g.children, depth + 1, show_all, out);
            }
        }
    }
    let mut out = Vec::new();
    rec(&doc.layers, 0, show_all, &mut out);
    out
}

/// The Layers panel's rows for `app`: [`display_rows`] (every row while filtered) without a Type
/// Mask session's working text, which is no layer to the user: Photoshop shows only the mask
/// overlay while you type (#3036).
pub fn panel_rows<'a>(app: &PhotocraftApp, doc: &'a Document) -> Vec<(usize, &'a Layer)> {
    let mut rows = display_rows(doc, !app.ui.layer_filter.is_empty());
    if let Some(working) = app.ui.text_edit.as_ref().filter(|ed| ed.mask).map(|ed| ed.layer) {
        rows.retain(|(_, l)| l.id.0 != working);
    }
    rows
}

/// The hit area of a group row's triangle starting at `x`.
pub fn triangle_rect(row: Rect, x: f32) -> Rect {
    Rect::from_min_size(pos2(x - 2.0, row.top()), vec2(TRIANGLE_W + 2.0, row.height()))
}

/// Paint and handle a group row's disclosure triangle at `*x` (advancing it). Returns true when
/// it was clicked this frame, so the row doesn't also treat the click as a selection.
pub fn disclosure(ui: &mut egui::Ui, row: Rect, x: &mut f32, l: &Layer, actions: &mut Vec<(String, Value)>) -> bool {
    let LayerContent::Group(g) = &l.content else { return false };
    let hit = triangle_rect(row, *x);
    let resp = ui.interact(hit, ui.id().with(("disclosure", l.id.0)), Sense::click());
    let t = Tokens::get(ui.ctx());
    let c = pos2(*x + TRIANGLE_W / 2.0 - 1.0, row.center().y);
    let s = 3.5;
    let pts = if g.expanded {
        vec![pos2(c.x - s, c.y - s * 0.6), pos2(c.x + s, c.y - s * 0.6), pos2(c.x, c.y + s * 0.6)]
    } else {
        vec![pos2(c.x - s * 0.6, c.y - s), pos2(c.x + s * 0.6, c.y), pos2(c.x - s * 0.6, c.y + s)]
    };
    let color = if resp.hovered() { t.text } else { t.icon };
    ui.painter().add(Shape::convex_polygon(pts, color, Stroke::NONE));
    *x += TRIANGLE_W;
    let clicked = resp.clicked();
    if clicked {
        let all = ui.input(|i| i.modifiers.alt);
        actions.push(("layer.setExpanded".into(), json!({"layer": l.id.0, "expanded": !g.expanded, "all": all})));
    }
    let (verb, name) = (if g.expanded { tl!("Collapse") } else { tl!("Expand") }, l.name.clone());
    let tip = if g.expanded { tl!("Collapse group  ({key}-click: all groups)") } else { tl!("Expand group  ({key}-click: all groups)") };
    let resp = resp.on_hover_text(crate::i18n::fmt(tip, &[("key", &crate::shortcuts::pretty("Alt"))]));
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, format!("{verb} group {name}")));
    clicked
}

#[cfg(test)]
mod tests;
