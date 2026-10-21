//! Right-to-left shell layout (Arabic, Hebrew, Persian, Urdu).
//!
//! The direction follows the UI language ([`crate::i18n::Lang::dir`]). In a right-to-left
//! language the window's chrome mirrors, as in right-to-left macOS and Windows apps: the menu
//! bar starts at the right, the toolbox docks on the right, the panel dock and its rail on the
//! left, and the title and status bars read from the right. Panel interiors, dialogs, document
//! tabs and the canvas keep their left-to-right layout for now (`docs/localization-parity.md`).
//!
//! egui has no global direction, so call sites choose sides and rows through these helpers.
//! Rows nested in a [`row`] follow it on their own: egui starts a `ui.horizontal*` row from the
//! right when its parent layout runs right-to-left.
//!
//! `PHOTOCRAFT_UI_DIRECTION=rtl` (or `ltr`) forces a direction whatever the language, so the
//! mirrored layout can be checked before a right-to-left catalog is installed. It is read once
//! per process; the web build has no environment and always follows the language.

use std::cell::Cell;
use std::sync::OnceLock;

use egui::{Align, Layout};

pub use crate::i18n::Dir;

thread_local! {
    // Like the drawing language, per thread: test harnesses run side by side.
    static FORCED: Cell<Option<Dir>> = const { Cell::new(None) };
}

/// The direction the UI is drawn in on this thread.
pub fn current() -> Dir {
    FORCED.get().or_else(env_override).unwrap_or_else(|| crate::i18n::current().dir())
}

/// Is the UI drawn right-to-left?
pub fn is_rtl() -> bool {
    current() == Dir::Rtl
}

/// Draw in `dir` whatever the language until the guard drops on this thread (tests).
#[must_use]
pub fn direction_scope(dir: Dir) -> impl Drop {
    struct Restore {
        previous: Option<Dir>,
        _thread: std::marker::PhantomData<std::rc::Rc<()>>,
    }
    impl Drop for Restore {
        fn drop(&mut self) {
            FORCED.set(self.previous);
        }
    }
    let restore = Restore { previous: FORCED.get(), _thread: std::marker::PhantomData };
    FORCED.set(Some(dir));
    restore
}

fn env_override() -> Option<Dir> {
    static ENV: OnceLock<Option<Dir>> = OnceLock::new();
    *ENV.get_or_init(|| parse_dir(&std::env::var("PHOTOCRAFT_UI_DIRECTION").ok()?))
}

/// `rtl` / `ltr`, any case; anything else (including empty) follows the language.
fn parse_dir(value: &str) -> Option<Dir> {
    match value.trim().to_ascii_lowercase().as_str() {
        "rtl" => Some(Dir::Rtl),
        "ltr" => Some(Dir::Ltr),
        _ => None,
    }
}

/// A side panel on the reading-start side: left, or right in a right-to-left UI (the toolbox).
pub fn start_panel(id: impl Into<egui::Id>) -> egui::Panel {
    if is_rtl() { egui::Panel::right(id) } else { egui::Panel::left(id) }
}

/// A side panel on the reading-end side: right, or left in a right-to-left UI (the dock).
pub fn end_panel(id: impl Into<egui::Id>) -> egui::Panel {
    if is_rtl() { egui::Panel::left(id) } else { egui::Panel::right(id) }
}

/// A vertically centred row in reading order.
pub fn row() -> Layout {
    if is_rtl() { Layout::right_to_left(Align::Center) } else { Layout::left_to_right(Align::Center) }
}

/// A vertically centred row that starts from the reading end: for groups pinned to the far end
/// of a bar, such as the title bar's controls or the status bar's Fit button.
pub fn trailing() -> Layout {
    if is_rtl() { Layout::left_to_right(Align::Center) } else { Layout::right_to_left(Align::Center) }
}

/// [`egui::Ui::horizontal_centered`] in reading order: a row filling the space left, with its
/// items centred vertically, starting from the right in a right-to-left UI.
pub fn horizontal_centered<R>(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui) -> R) -> egui::InnerResponse<R> {
    let size = ui.available_size_before_wrap();
    ui.allocate_ui_with_layout(size, row(), add)
}

/// The popup alignment for a menu opening below its button: under its reading-start edge.
pub fn below() -> egui::RectAlign {
    if is_rtl() { egui::RectAlign::BOTTOM_END } else { egui::RectAlign::BOTTOM_START }
}

#[cfg(test)]
#[path = "direction_tests.rs"]
mod tests;
