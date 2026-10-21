//! Keyboard-layout independent keys for the brush shortcuts and ⇧+ / ⇧− (#3102, #3109).
//!
//! Photoshop's `[` `]` (brush size) and ⇧`[` ⇧`]` (hardness) sit on keys that many layouts don't
//! have: on German, French, Spanish... they need AltGr or Option. Photoshop therefore picks other
//! keys per language, and so do we, in two steps (neither needs an OS layout API: egui gives us
//! the typed text next to each key press, plus the logical and the physical key):
//!
//! 1. [`LAYOUTS`] / [`COMMON`]: a small data table, symbol typed -> key it stands for, for the
//!    layouts whose Photoshop keys we know. The layout is inferred from the typed symbols.
//! 2. Fallback for every other layout: when the physical keys at the US `[` / `]` positions type
//!    something that is not a bracket, they act as `[` / `]` (US layout: those keys *are* the
//!    brackets, nothing changes). Dead keys (AZERTY `^`) and symbols egui has no name for (German
//!    `ü`) already arrive as the physical key; this adds the ones egui names by their character
//!    (German/Spanish/Italian `+`, Dvorak `/`).
//!
//! The fallback stays off once a real bracket was typed (Dvorak types `[` elsewhere) or a table
//! layout's symbol was seen, and never applies with ⌘/Ctrl/Alt held. Edit › Keyboard Shortcuts
//! overrides are unaffected: this only decides which key a press is, the shortcut is looked up
//! after it.
//!
//! Adding a layout: append a [`Layout`] to [`LAYOUTS`] with the `(typed text, ⇧ held, key)` rows
//! Photoshop uses in that language (state them from Photoshop itself, not guesses), and add a
//! test next to `german_*` in `shortcuts.rs`. Layouts without a row work through the fallback.

use egui::{Event, Key};

/// One keyboard layout's symbol keys: `(text typed, ⇧ held, key the press stands for)`.
pub struct Layout {
    pub keys: &'static [(&'static str, bool, Key)],
}

/// Rows valid on every layout. ⇧+ / ⇧− (#3102): on layouts with a key of its own for `+`
/// (German, Italian, Spanish...) ⇧+ types `*` and ⇧− types `_`, which egui has no keys for, so
/// they arrive as the physical `]` / `/`. They become `+` / `-`, as ⇧= arrives as `+` on a US
/// layout. A US ⇧] still arrives as `}` and ⇧8 (`*` too) as a digit.
pub const COMMON: Layout = Layout { keys: &[("*", true, Key::Plus), ("_", true, Key::Minus)] };

/// Layouts with Photoshop-specific brush keys (#3109). German: Ö / # are `[` / `]` (smaller,
/// bigger) and ⇧Ö / ⇧# the hardness; `ö` and `#` arrive as the physical `;` and `\`, ⇧# types `'`.
/// French AZERTY and others have no verified Photoshop keys yet and use the fallback.
pub const LAYOUTS: &[Layout] = &[Layout {
    // German
    keys: &[("ö", false, Key::OpenBracket), ("Ö", true, Key::OpenBracket), ("#", false, Key::CloseBracket), ("'", true, Key::CloseBracket)],
}];

fn lookup(layout: &Layout, text: &str, shift: bool) -> Option<Key> {
    layout.keys.iter().find(|(t, s, _)| *t == text && *s == shift).map(|r| r.2)
}

/// What the app has learnt about the layout this session.
#[derive(Clone, Copy, Default)]
struct Learnt {
    /// A real `[` / `]` was typed: the fallback would only steal keys.
    brackets: bool,
    /// A table layout's own symbol was seen: its keys are the brush keys, not the fallback's.
    table_layout: bool,
}

fn is_digit(key: Key) -> bool {
    matches!(key, Key::Num0 | Key::Num1 | Key::Num2 | Key::Num3 | Key::Num4 | Key::Num5 | Key::Num6 | Key::Num7 | Key::Num8 | Key::Num9)
}

/// Gives this frame's key presses the keys the layout table and the fallback assign, so every
/// shortcut handler after it sees them. Only a press whose key egui took from the physical key
/// (it had no name for the symbol) or from that very text is changed by the table, and never a
/// digit: US ⇧8 types `*` and stays ⇧8 for the opacity keys.
///
/// What was typed comes as the text event next to the key press: right after it natively, right
/// before it on the web.
pub fn apply(ctx: &egui::Context) {
    let id = egui::Id::new("photocraft_layout_keys");
    let mut learnt: Learnt = ctx.data(|d| d.get_temp(id)).unwrap_or_default();
    let before = (learnt.brackets, learnt.table_layout);
    ctx.input_mut(|i| {
        for at in 0..i.events.len() {
            let text_at = if cfg!(target_arch = "wasm32") { at.checked_sub(1) } else { at.checked_add(1) };
            let text = match text_at.and_then(|t| i.events.get(t)) {
                Some(Event::Text(text)) => text.clone(),
                _ => String::new(),
            };
            let Some(Event::Key { key, physical_key, pressed: true, modifiers, .. }) = i.events.get_mut(at) else { continue };
            if modifiers.command || modifiers.ctrl || modifiers.alt {
                continue;
            }
            if matches!(text.as_str(), "[" | "]" | "{" | "}") {
                learnt.brackets = true;
                continue;
            }
            let from_physical = *physical_key == Some(*key) || Key::from_name(&text) == Some(*key);
            if !text.is_empty() && physical_key.is_some() && from_physical && !is_digit(*key) {
                if let Some(k) = lookup(&COMMON, &text, modifiers.shift) {
                    *key = k;
                    continue;
                }
                if let Some(k) = LAYOUTS.iter().find_map(|l| lookup(l, &text, modifiers.shift)) {
                    learnt.table_layout = true;
                    *key = k;
                    continue;
                }
            }
            // Fallback: the keys at the US bracket positions, when they don't type brackets.
            if !learnt.brackets
                && !learnt.table_layout
                && let Some(p @ (Key::OpenBracket | Key::CloseBracket)) = *physical_key
            {
                *key = p;
            }
        }
    });
    if before != (learnt.brackets, learnt.table_layout) {
        ctx.data_mut(|d| d.insert_temp(id, learnt));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PhotocraftApp, state::Tool};
    use egui::Modifiers;
    use serde_json::json;

    fn app() -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 32, "height": 32})).unwrap();
        app.ui.tool = Tool::Brush;
        app
    }

    /// One key press the way a layout reports it: logical `key`, physical position, typed text.
    fn press(app: &mut PhotocraftApp, ctx: &egui::Context, key: Key, physical: Key, text: &str) {
        let modifiers = Modifiers::NONE;
        let press = Event::Key { key, physical_key: Some(physical), pressed: true, repeat: false, modifiers };
        let raw = egui::RawInput { events: vec![press, Event::Text(text.into())], ..Default::default() };
        ctx.begin_pass(raw);
        crate::shortcuts::handle(app, ctx);
        ctx.end_pass().textures_delta.clear();
    }

    /// A layout that types `+` on the US `]` key (Spanish, Italian...) and has no table row:
    /// that key is `]` through the fallback, and the US `]` and a Dvorak `[` keep working.
    #[test]
    fn bracket_positions_are_the_brush_keys_on_other_layouts() {
        let (mut a, ctx) = (app(), egui::Context::default());
        let size = a.session.tools.brush.size;
        press(&mut a, &ctx, Key::Plus, Key::CloseBracket, "+");
        assert!(a.session.tools.brush.size > size, "the `+` key at the `]` position is ]");
        // AZERTY `^` is a dead key at the `[` position: it arrives as the physical key.
        press(&mut a, &ctx, Key::OpenBracket, Key::OpenBracket, "^");
        assert_eq!(a.session.tools.brush.size, size, "the key at the `[` position is [");
    }

    #[test]
    fn real_brackets_switch_the_fallback_off() {
        // Dvorak: `[` is typed on the US `-` key, the US `]` position types `=`.
        let (mut a, ctx) = (app(), egui::Context::default());
        press(&mut a, &ctx, Key::OpenBracket, Key::Minus, "[");
        let size = a.session.tools.brush.size;
        press(&mut a, &ctx, Key::Equals, Key::CloseBracket, "=");
        assert_eq!(a.session.tools.brush.size, size, "= is not stolen once brackets exist");
    }

    #[test]
    fn german_keys_do_not_also_use_the_fallback() {
        let (mut a, ctx) = (app(), egui::Context::default());
        press(&mut a, &ctx, Key::Semicolon, Key::Semicolon, "ö");
        let size = a.session.tools.brush.size;
        press(&mut a, &ctx, Key::Plus, Key::CloseBracket, "+");
        assert_eq!(a.session.tools.brush.size, size, "German + is not a brush key");
    }
}
