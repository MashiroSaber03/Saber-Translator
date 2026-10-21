//! Held keys: Photoshop's temporary tools (#249).
//!
//! - Space: the Hand tool while held.
//! - ⌘Space (Ctrl+Space off the Mac): Zoom In while held; a drag is a scrubby zoom.
//! - ⌥Space (Alt+Space off the Mac): Zoom Out while held. ⌥ added to a held Zoom In (⌘⌥Space)
//!   zooms out too, as ⌥ does with the Zoom tool.
//! - The keys can be pressed in any order. Photoshop users usually hold Space first: on the Mac
//!   that keeps ⌘Space from opening Spotlight, and many Linux desktops take ⌥Space for the window
//!   menu when ⌥ comes first.
//! - Space while a marquee, lasso or shape is being dragged repositions it; releasing Space
//!   goes back to sizing it (the Crop tool does the same for its frame, `crop_ui`).
//! - ⌘ (Ctrl off the Mac): the Move tool while held, with the painting, retouching, eraser,
//!   gradient, eyedropper and other non-selection tools ([`cmd_moves`]); so ⌘-drag with the
//!   Brush moves the layer, ⌘⌥-drag duplicates it first (`move_mods`), and a ⌘-click picks the
//!   layer under the pointer, as with the Move tool itself; ⌘-arrows nudge ([`cmd_nudges`]). The
//!   selection tools do their own ⌘ handling in `canvas` (⌘ inside the selection cuts the
//!   selected pixels, outside it moves the layer, #896), and the Hand, Zoom, Crop, Slice, Path
//!   Selection, shape, Pen and Type tools keep ⌘ for themselves.
//! - `~` (the backtick key) with the Brush or Pencil: a stroke started while it is held erases,
//!   with that tool's own brush ([`erase_held`], #3439).
//!
//! The current tool is never changed, so releasing the key gives the previous tool back. A
//! temporary tool that started a drag lasts until the button is released, as in Photoshop.
//!
//! The keys are bindings, not commands: they come from
//! [`photocraft_engine::prefs::TEMPORARY_TOOLS`] with Edit › Keyboard Shortcuts overrides on top
//! (Tools › Temporary), and the reposition key is whatever the Hand key is.

use egui::{InputState, KeyboardShortcut};

use crate::PhotocraftApp;
use crate::shortcuts::parse;
use crate::state::Tool;

/// A tool in effect only while its key is held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Temporary {
    Hand,
    ZoomIn,
    ZoomOut,
    /// ⌘ held with a tool that [`cmd_moves`]: a modifier, not a rebindable key.
    Move,
}

impl Temporary {
    pub fn id(self) -> &'static str {
        match self {
            Temporary::Hand => "tools.temporary.hand",
            Temporary::ZoomIn => "tools.temporary.zoomIn",
            Temporary::ZoomOut => "tools.temporary.zoomOut",
            Temporary::Move => "tools.temporary.move",
        }
    }

    pub fn tool(self) -> Tool {
        match self {
            Temporary::Hand => Tool::Hand,
            Temporary::ZoomIn | Temporary::ZoomOut => Tool::Zoom,
            Temporary::Move => Tool::Move,
        }
    }

    /// The key-bound temporary tools (`prefs::TEMPORARY_TOOLS`).
    const BOUND: [Temporary; 3] = [Temporary::Hand, Temporary::ZoomIn, Temporary::ZoomOut];
}

/// Tools on which holding ⌘ (Ctrl off the Mac) is the Move tool. Photoshop excepts the Hand,
/// Rotate View, Zoom, Slice, Path and Direct Selection, shape, Pen and Type tools (⌘ means
/// something else to each) and the Crop tool's frame; a Free Transform in progress keeps ⌘ for its
/// distort handles. The selection tools are left out too: for them ⌘ is handled per press in
/// `canvas::tool_event` (`selection_drag_kind`, `command_moves_layer`), which needs the selection
/// tool to stay in effect.
pub fn cmd_moves(t: Tool) -> bool {
    !t.is_type()
        && !matches!(
            t,
            Tool::Move
                | Tool::Hand
                | Tool::RotateView
                | Tool::Zoom
                | Tool::Crop
                | Tool::PerspectiveCrop
                | Tool::Slice
                | Tool::SliceSelect
                | Tool::PathSelection
                | Tool::DirectSelection
                | Tool::Pen
                | Tool::FreeformPen
                | Tool::CurvaturePen
                | Tool::AddAnchorPoint
                | Tool::DeleteAnchorPoint
                | Tool::ConvertPoint
                | Tool::RectMarquee
                | Tool::EllipseMarquee
                | Tool::Lasso
                | Tool::PolygonLasso
                | Tool::MagicWand
        )
        && !crate::vector_ui::is_shape_tool(t)
}

/// Do ⌘-arrows (Ctrl off the Mac) nudge as the Move tool's arrows do (#2474)? Wherever ⌘ is the
/// Move tool ([`cmd_moves`]), with the Move tool itself, and with the selection tools: in
/// Photoshop their ⌘ is the Move tool too, so ⌘-arrows move the selected pixels (or the layer)
/// while the plain arrows move the outline. Not while a polygon or Magnetic Lasso border is in
/// progress, during Free Transform or while editing text.
pub fn cmd_nudges(app: &PhotocraftApp) -> bool {
    let t = app.ui.tool;
    let selecting = crate::tool_feedback::is_selection_tool(t);
    app.ui.transform.is_none()
        && app.ui.text_edit.is_none()
        && (cmd_moves(t) || t == Tool::Move || selecting)
        && !(selecting && (!app.ui.polygon.is_empty() || app.ui.magnetic.active()))
}

/// Is ⌘ alone (no other temporary key) making the Move tool now?
fn cmd_move_held(app: &PhotocraftApp, i: &InputState) -> bool {
    i.modifiers.command && cmd_moves(app.ui.tool) && app.ui.transform.is_none() && app.ui.text_edit.is_none()
}

/// Is `id` a held temporary tool rather than a command?
pub fn is_temporary(id: &str) -> bool {
    photocraft_engine::prefs::TEMPORARY_TOOLS.iter().any(|t| t.0 == id)
}

/// The key binding of a temporary tool in effect (`None` when removed or unparsable).
pub fn binding(app: &PhotocraftApp, t: Temporary) -> Option<KeyboardShortcut> {
    binding_of(app, t.id())
}

/// [`binding`] by id.
fn binding_of(app: &PhotocraftApp, id: &str) -> Option<KeyboardShortcut> {
    let def = photocraft_engine::prefs::TEMPORARY_TOOLS.iter().find(|x| x.0 == id).map(|x| x.2);
    parse(app.session.prefs().shortcut(id, def)?)
}

/// Is `sc` held: its key down with at least its modifiers (more may be held, so ⇧ still squares
/// a marquee while Space is down).
fn held(i: &InputState, sc: &KeyboardShortcut) -> bool {
    let (want, m) = (sc.modifiers, i.modifiers);
    i.key_down(sc.logical_key)
        && (!want.alt || m.alt)
        && (!want.shift || m.shift)
        && (!want.command || m.command)
        && (!want.ctrl || m.ctrl)
        && (!want.mac_cmd || m.mac_cmd)
}

/// The temporary tool whose key is held now, the most specific binding first (⌘Space before
/// Space). Never while a text field has the keyboard, or while editing type on the canvas
/// (Space must insert a space, not grab the Hand tool).
pub fn held_tool(app: &PhotocraftApp, ctx: &egui::Context) -> Option<Temporary> {
    if ctx.text_edit_focused() || app.ui.text_edit.is_some() {
        return None;
    }
    let count = |sc: &KeyboardShortcut| {
        let m = sc.modifiers;
        m.alt as u8 + m.shift as u8 + (m.command || m.mac_cmd) as u8 + (m.ctrl && !m.command) as u8
    };
    let mut best: Option<(u8, Temporary, KeyboardShortcut)> = None;
    let mut cmd_move = false;
    let mut alt = false;
    ctx.input(|i| {
        for t in Temporary::BOUND {
            if let Some(sc) = binding(app, t)
                && held(i, &sc)
                && best.is_none_or(|(n, ..)| count(&sc) > n)
            {
                best = Some((count(&sc), t, sc));
            }
        }
        cmd_move = cmd_move_held(app, i);
        alt = i.modifiers.alt;
    });
    // ⌥ turns a held Zoom In around, as it does the Zoom tool: ⌘⌥Space zooms out (and wins the
    // tie with ⌥Space), unless ⌥ is part of the Zoom In key itself.
    let held = best.map(|(_, t, sc)| if t == Temporary::ZoomIn && alt && !sc.modifiers.alt { Temporary::ZoomOut } else { t });
    // A bound key (⌘Space is Zoom In) wins over ⌘ alone.
    held.or(cmd_move.then_some(Temporary::Move))
}

/// The binding id of the Brush and Pencil erase key (`prefs::TEMPORARY_TOOLS`).
const ERASE: &str = "tools.temporary.erase";

/// Tools whose strokes the erase key turns into erasing strokes with their own brush.
pub fn erases_while_held(t: Tool) -> bool {
    matches!(t, Tool::Brush | Tool::Pencil)
}

/// Is the erase key (`~` by default) held with a tool that [`erases_while_held`]? Never while a
/// text field has the keyboard.
pub fn erase_held(app: &PhotocraftApp, ctx: &egui::Context) -> bool {
    erases_while_held(app.ui.tool) && !ctx.text_edit_focused() && binding_of(app, ERASE).is_some_and(|sc| ctx.input(|i| held(i, &sc)))
}

/// Is the reposition key (the Hand key, any modifiers) down?
pub fn reposition_held(app: &PhotocraftApp, ctx: &egui::Context) -> bool {
    !ctx.text_edit_focused() && app.ui.text_edit.is_none() && binding(app, Temporary::Hand).is_some_and(|sc| ctx.input(|i| i.key_down(sc.logical_key)))
}

/// Tools whose in-progress drag the reposition key moves.
pub fn repositions(t: Tool) -> bool {
    matches!(t, Tool::RectMarquee | Tool::EllipseMarquee | Tool::Lasso | Tool::ObjectSelection) || crate::vector_ui::is_shape_tool(t)
}

fn latch_id() -> egui::Id {
    egui::Id::new("pc-temporary-tool")
}

/// The temporary tool for this canvas frame: the held one, except while a selection is being
/// drawn (the key repositions it instead); a temporary tool that was in effect when the button
/// went down stays until the button comes up, release included (the Move tool's release is
/// what commits the move).
pub fn for_frame(app: &PhotocraftApp, ctx: &egui::Context, drawing: bool) -> Option<Temporary> {
    let (down, released) = ctx.input(|i| (i.pointer.primary_down(), i.pointer.primary_released()));
    let latched = ctx.data(|d| d.get_temp::<Option<Temporary>>(latch_id())).flatten();
    let t = if (down || released) && latched.is_some() {
        latched
    } else if drawing {
        None
    } else {
        held_tool(app, ctx)
    };
    ctx.data_mut(|d| d.insert_temp(latch_id(), if down { t } else { None }));
    t
}

#[cfg(test)]
mod tests;
