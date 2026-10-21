//! Photoshop's "Could not use the move tool because the layer is locked." A Move-tool press (the
//! Move tool itself, or ⌘ with a selection tool, `canvas::command_moves_layer`) on a layer it
//! can't move opens that message in a dialog instead
//! of silently doing nothing: the Background and any position-locked layer when there is no
//! selection, and a layer whose selected pixels can't float
//! (`photocraft_engine::float_cmds::locked_for_float`) when there is one. As in Photoshop the
//! message comes when the pointer starts to drag; a plain click says nothing. OK closes it.

use serde_json::json;

use crate::PhotocraftApp;
use crate::state::{DialogKind, Tool};

pub const MESSAGE: &str = "Could not use the move tool because the layer is locked.";

/// Would a press at document point `p` with `tool` in effect and `mods` held be a move that
/// fails because of a lock?
pub fn blocked(app: &PhotocraftApp, tool: Tool, p: [f64; 2], mods: egui::Modifiers) -> bool {
    let Some(st) = app.session.active() else { return false };
    let floating = photocraft_engine::float_cmds::floating(st).is_some();
    // The selected pixels: only a fresh cut can be refused.
    let pixels_locked =
        || !floating && st.active_layer.and_then(|id| st.doc.layer(id)).is_some_and(|l| photocraft_engine::float_cmds::locked_for_float(&st.doc, l));
    // The layers: any of them locked in place refuses the whole move.
    let layers_locked = || {
        photocraft_engine::layer_multi_cmds::move_targets(&st.doc, &st.selected_layers()).iter().any(|id| {
            let locks = st.doc.effective_locks(*id);
            locks.position || locks.all
        })
    };
    if tool == Tool::Move {
        // `moves_selected_pixels_with` is false for a layer whose pixels can't float: then the Move tool
        // moves the layer, which its position lock refuses too.
        if crate::move_ui::moves_selected_pixels_with(app, tool) {
            return pixels_locked();
        }
        return !floating && st.doc.selection.is_some() && pixels_locked() || layers_locked();
    }
    if crate::canvas::command_moves_layer(app, tool, p, mods) {
        return layers_locked();
    }
    crate::canvas::selection_drag_kind(app, tool, p, mods) == Some(true) && mods.command && pixels_locked()
}

/// Opens the message. The dialog is ordinary UI state (`ui.dialog.confirm` closes it).
pub fn prompt(app: &mut PhotocraftApp) {
    app.ui.open_dialog(DialogKind::Error, json!({"message": tl!(MESSAGE)}).as_object().cloned().unwrap_or_default());
}
