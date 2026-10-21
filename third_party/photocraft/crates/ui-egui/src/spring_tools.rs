//! Spring-loaded tool keys (#2672): hold a tool's letter for a temporary tool.
//!
//! A tool letter switches tools when it is pressed, as before (`shortcuts::handle`). If the key is
//! then held for [`SPRING_HOLD_SECS`] or longer, or the canvas is used while it is down, the switch
//! was only temporary: releasing the key gives back the tool in effect before the press, with its
//! options. A quick tap with no canvas use stays a permanent switch.
//!
//! - ⇧+letter (cycling the group) and ⌥+letter never spring; nor do letters while typing in a
//!   field, editing type or under a modal dialog, which never reach the tool keys.
//! - Pressing the current group's own letter cycles the group, as a tap does, and doesn't spring.
//! - A canvas drag still in progress when the key comes up finishes with the temporary tool; the
//!   previous tool returns once the button is up (as `hold_keys` latches Space).
//! - Another letter pressed while one is held wins (newest first); releasing every held letter
//!   returns the original tool. Holding two letters at once counts as a hold.
//! - Key repeats are not presses: `shortcuts::handle` drops them for tool letters.
//!
//! Z is one case: hold Z to zoom, add ⌥ to zoom out, as with the Zoom tool itself (and as
//! `hold_keys` does for ⌥ with a held Zoom In).
//!
//! The state is [`crate::state::UiState::spring`], readable over the control channel.

use egui::Key;
use serde::{Deserialize, Serialize};

use crate::PhotocraftApp;
use crate::state::Tool;

/// How long (seconds) a tool letter must stay down for its tool to be temporary.
pub const SPRING_HOLD_SECS: f64 = 0.3;

/// A tool switch made by a letter that is still down (or whose drag is still finishing).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Spring {
    /// The tool to give back.
    pub previous: Tool,
    /// The tool the held letter picked. If the tool changes any other way while it's held (the
    /// toolbar, a menu, automation), that choice stands and nothing springs back.
    pub tool: Tool,
    /// The tool letters held, oldest first.
    pub keys: Vec<char>,
    /// `egui` input time of the first press.
    pub since: f64,
    /// The canvas was pressed (or a second letter pressed) while held: the switch is temporary
    /// however short the hold.
    pub used: bool,
    /// Every letter is up and the switch is temporary, but a drag is still in progress.
    pub released: bool,
}

/// The egui key of a tool letter (`None` for the tools without one).
pub fn key_of(letter: char) -> Option<Key> {
    if letter == '\0' {
        return None;
    }
    Key::from_name(&letter.to_string())
}

/// Is `k` some tool's single-key shortcut?
pub fn is_tool_key(k: Key) -> bool {
    Tool::ALL.iter().any(|t| key_of(t.key()) == Some(k))
}

/// A tool letter switched the tool from `before` (the switch is already made). `plain` is false
/// when ⇧ or ⌥ was held: those switches are permanent. `now` is the input time.
pub fn pressed(app: &mut PhotocraftApp, letter: char, before: Tool, plain: bool, now: f64) {
    let active = app.ui.spring.as_ref().is_some_and(|s| !s.released);
    if !plain {
        app.ui.spring = None;
        return;
    }
    if active {
        if let Some(s) = app.ui.spring.as_mut() {
            s.keys.retain(|k| *k != letter);
            s.keys.push(letter);
            s.used = true;
            s.tool = app.ui.tool;
        }
        return;
    }
    // The current group's own key cycles it (or keeps it), as a tap always has.
    let tool = app.ui.tool;
    app.ui.spring = (before.key() != letter).then(|| Spring { previous: before, tool, keys: vec![letter], since: now, used: false, released: false });
}

/// The canvas was pressed: a held letter's tool becomes temporary.
pub fn note_canvas_press(app: &mut PhotocraftApp) {
    if let Some(s) = app.ui.spring.as_mut().filter(|s| !s.released) {
        s.used = true;
    }
}

/// Once a frame, before the key handling: notice released letters and give the previous tool
/// back when the switch was temporary and no drag is in progress.
pub fn tick(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let Some(mut s) = app.ui.spring.take() else { return };
    // Type editing begun with the temporary Type tool keeps it: the edit owns the tool now. So
    // does a tool picked another way while the letter is down.
    if app.ui.text_edit.is_some() || app.ui.tool != s.tool {
        return;
    }
    let (now, button) = ctx.input(|i| (i.time, i.pointer.any_down()));
    let dragging = app.drag.is_some() || button;
    if !s.released {
        if app.drag.is_some() {
            s.used = true;
        }
        s.keys.retain(|k| key_of(*k).is_some_and(|k| ctx.input(|i| i.key_down(k))));
        if !s.keys.is_empty() {
            app.ui.spring = Some(s);
            return;
        }
        if !s.used && now - s.since < SPRING_HOLD_SECS {
            // A tap: the switch stands.
            return;
        }
        s.released = true;
    }
    if dragging {
        app.ui.spring = Some(s);
        ctx.request_repaint();
        return;
    }
    app.ui.tool = s.previous;
}

#[cfg(test)]
mod tests;
