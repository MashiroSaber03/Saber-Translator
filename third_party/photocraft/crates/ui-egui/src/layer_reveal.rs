//! Reveal the active layer in the Layers panel (#152), like Photoshop: whenever the active
//! layer changes, by whatever means (a click on the canvas with Auto-Select, Select › layer
//! commands, ⌥[ / ⌥], undo, automation), its closed parent groups open (`layer.setExpanded`)
//! and the panel scrolls just enough to show its row, unless the row is already in view.
//!
//! Only a change of the active layer reveals: closing the group that holds it, or scrolling
//! away, is left alone until the next change.

use egui::{Rect, pos2};
use photocraft_doc::{LayerContent, LayerId};
use serde_json::json;

use crate::PhotocraftApp;

/// Last active layer the panel saw: (document index, layer).
#[derive(Clone, Copy, PartialEq)]
struct Seen(usize, Option<LayerId>);

fn seen_id() -> egui::Id {
    egui::Id::new("layers-reveal-seen")
}

/// Called once per Layers panel frame, before its rows are listed. When the active layer
/// changed since the last frame, opens its closed parent groups and returns it: the row to
/// scroll into view (see [`scroll_to_row`]). The first frame of a document only records it.
pub fn track(app: &mut PhotocraftApp, ctx: &egui::Context) -> Option<LayerId> {
    let index = app.session.active_index()?;
    let active = app.session.active()?.active_layer;
    let now = Seen(index, active);
    let before = ctx.data(|d| d.get_temp::<Seen>(seen_id()));
    ctx.data_mut(|d| d.insert_temp(seen_id(), now));
    match before {
        Some(Seen(i, prev)) if i == index && prev != active => {}
        _ => return None,
    }
    let id = active?;
    for group in closed_ancestors(app, id) {
        // A view change (no history step); a failure just leaves the group closed.
        let _ = app.session.execute("layer.setExpanded", json!({"layer": group.0, "expanded": true}));
    }
    Some(id)
}

/// The groups containing `id` that are closed, outermost first.
fn closed_ancestors(app: &PhotocraftApp, id: LayerId) -> Vec<LayerId> {
    let Some(doc) = app.session.active().map(|s| &s.doc) else { return Vec::new() };
    let Some(path) = doc.path_of(id) else { return Vec::new() };
    (1..path.len()).filter_map(|n| doc.layer_at(path.get(..n)?)).filter(|l| matches!(&l.content, LayerContent::Group(g) if !g.expanded)).map(|l| l.id).collect()
}

/// After drawing the row that started at `top` in the Layers scroll area, scroll it into view
/// if any of it is hidden (no animation: Photoshop jumps).
pub fn scroll_to_row(ui: &egui::Ui, top: f32) {
    let bottom = ui.min_rect().bottom().max(top);
    let row = Rect::from_min_max(pos2(ui.min_rect().left(), top), pos2(ui.min_rect().right(), bottom));
    let view = ui.clip_rect();
    if row.top() < view.top() - 0.5 || row.bottom() > view.bottom() + 0.5 {
        ui.scroll_to_rect_animation(row, None, egui::style::ScrollAnimation::none());
    }
}

#[cfg(test)]
#[path = "layer_reveal_tests.rs"]
mod tests;
