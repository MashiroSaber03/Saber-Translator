//! An ⌥-click the desktop took (#1658, #1659).
//!
//! Many Linux desktops move a window with Alt+drag: Cinnamon (Linux Mint's default), MATE, Xfce,
//! and KDE before Plasma 6. Their window manager grabs every Alt+left press, so the click never
//! reaches PhotoCraft: the Brush's ⌥ Eyedropper samples nothing, the Clone Stamp gets no source
//! point and the Zoom tool does not zoom out (#3137), while the ⌥ cursor still shows (the keyboard is not grabbed until the press). Nothing in
//! the app can win that grab back, but its signature is clear: the pointer "leaves" the window from
//! well inside the canvas while ⌥ is held with a tool whose ⌥-click matters (X11 and Wayland report
//! a grab as the pointer leaving), and comes back to a window that still has the keyboard focus
//! when the button is released. The status bar then says what happened and how to get ⌥-click
//! back. An Alt+Tab away leaves the window without focus, so it never shows the message.

use egui::{Context, Event, Id, Pos2, Rect};

use crate::PhotocraftApp;
use crate::state::Tool;

/// How far inside the canvas the pointer must be for a "leave" to count as a grab rather than
/// the pointer moving off the window.
const MARGIN: f32 = 8.0;

/// Per canvas view: where ⌥ was last held over it, and whether a grab is waiting for the pointer
/// to come back.
#[derive(Clone, Copy, Debug, Default)]
struct Watch {
    armed: Option<Pos2>,
    pending: bool,
}

/// Tools whose ⌥-click does something: the ⌥ Eyedropper, the clone source point and Zoom Out.
pub(crate) fn alt_click_tool(tool: Tool, mods: egui::Modifiers) -> bool {
    crate::canvas::alt_samples(tool, mods) || (mods.alt && !mods.ctrl && matches!(tool, Tool::CloneStamp | Tool::Healing | Tool::Zoom))
}

/// The status-bar message for an ⌥-click the desktop took.
pub(crate) fn message() -> &'static str {
    tl!("Alt-click went to the desktop, which moves windows with Alt-drag. Set its window-move key to Super, or latch Alt in Window › Modifier Keys.")
}

/// Watch one canvas view's frame for an ⌥-click the window manager grabbed. `rect` is the view's
/// canvas area, `tool` the tool in effect, `active` false while a dialog owns the canvas.
pub(crate) fn watch(app: &mut PhotocraftApp, ctx: &Context, id: Id, rect: Rect, tool: Tool, active: bool) {
    let (gone, hover, mods, down, focused) =
        ctx.input(|i| (i.events.iter().any(|e| matches!(e, Event::PointerGone)), i.pointer.hover_pos(), i.modifiers, i.pointer.any_down(), i.focused));
    let mut w = ctx.data(|d| d.get_temp::<Watch>(id)).unwrap_or_default();
    if gone && w.armed.is_some() {
        w.pending = true;
    }
    if w.pending
        && let Some(p) = hover
        && rect.contains(p)
    {
        w.pending = false;
        if focused && active {
            app.ui.status = message().into();
            app.ui.status_error = true;
        }
    }
    // Only the physical key: a latched Window › Modifier Keys ⌥ isn't held on the keyboard.
    w.armed = hover.filter(|p| active && !gone && !down && rect.shrink(MARGIN).contains(*p) && alt_click_tool(tool, mods) && !mods.command);
    ctx.data_mut(|d| d.insert_temp(id, w));
}
