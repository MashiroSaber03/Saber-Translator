//! ⇧+ and ⇧− step through the blend modes, as in Photoshop (#3102): the Brush's or Pencil's mode
//! while one of them is active, the Gradient's with the Gradient tool, and otherwise (Move, the
//! selection tools…) the active layer's. The list wraps at either end. Each press is one call of
//! an engine command (`tools.setBrush`, `layer.setProps`), so it records and undoes like a pick
//! from the dropdown.

use egui::{Event, Key};
use photocraft_color::BlendMode;
use serde_json::json;

use crate::PhotocraftApp;
use crate::state::Tool;

/// The mode `step` places after (or before, when negative) `current` in `modes`, wrapping around.
/// A mode that isn't listed steps from the top of the list.
pub fn stepped(modes: &[BlendMode], current: BlendMode, step: i32) -> Option<BlendMode> {
    let n = i32::try_from(modes.len()).ok().filter(|n| *n > 0)?;
    let at = modes.iter().position(|m| *m == current).and_then(|i| i32::try_from(i).ok());
    let next = match at {
        Some(i) => (i + step).rem_euclid(n),
        None if step > 0 => 0,
        None => n - 1,
    };
    modes.get(usize::try_from(next).ok()?).copied()
}

/// Consumes this frame's ⇧+ / ⇧− presses (⇧= on a US layout is `+`; on a German one
/// `layout_keys::apply` has already turned ⇧+ and ⇧- into `+` and `-`) and applies them.
pub fn handle(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let steps: Vec<i32> = ctx.input_mut(|i| {
        let mut out = Vec::new();
        i.events.retain(|e| match e {
            Event::Key { key, pressed: true, modifiers, .. } if modifiers.shift && !modifiers.command && !modifiers.ctrl && !modifiers.alt => {
                let step = match key {
                    Key::Plus | Key::Equals => 1,
                    Key::Minus => -1,
                    _ => return true,
                };
                out.push(step);
                false
            }
            _ => true,
        });
        out
    });
    for step in steps {
        press(app, step);
    }
}

/// One ⇧+ (`step` 1) or ⇧− (`step` -1) press.
pub fn press(app: &mut PhotocraftApp, step: i32) {
    match app.ui.tool {
        Tool::Brush | Tool::Pencil => {
            let current = app.session.tools.brush.mode;
            if let Some(mode) = stepped(&BlendMode::LAYER_MODES, current, step)
                && let Ok(mode) = serde_json::to_value(mode)
            {
                let _ = app.run("tools.setBrush", json!({ "brush": { "mode": mode } }));
            }
        }
        Tool::Gradient => {
            let before = app.ui.tool_options.clone();
            if let Some(mode) = stepped(&BlendMode::LAYER_MODES, before.gradient_blend_mode, step) {
                app.ui.tool_options.gradient_blend_mode = mode;
                // Live mode: a selected gradient fill layer follows, as with the dropdown.
                crate::gradient_ui::options_changed(app, &before);
            }
        }
        // Their blend mode isn't the brush's (the Eraser has none): leave them alone.
        t if t.is_brushlike() || t == Tool::MagicEraser || t == Tool::PaintBucket => {}
        _ => {
            let Some(st) = app.session.active() else { return };
            let Some(layer) = st.active_layer.and_then(|id| st.doc.layer(id)) else { return };
            // The locked Background has no blend mode.
            if crate::doc_props_ui::is_background(&st.doc, layer) {
                return;
            }
            let modes: Vec<BlendMode> = std::iter::once(BlendMode::PassThrough).filter(|_| layer.is_group()).chain(BlendMode::LAYER_MODES).collect();
            if let Some(mode) = stepped(&modes, layer.blend, step) {
                let id = layer.id.0;
                let _ = app.run("layer.setProps", json!({ "layer": id, "blend": mode.label() }));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app(tool: Tool) -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 32, "height": 32})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        app.ui.tool = tool;
        app
    }

    fn layer_blend(app: &PhotocraftApp) -> BlendMode {
        let st = app.session.active().unwrap();
        st.doc.layer(st.active_layer.unwrap()).unwrap().blend
    }

    /// Feeds one key press with `modifiers` through the app's shortcut handling.
    fn key(app: &mut PhotocraftApp, ctx: &egui::Context, key: Key, modifiers: egui::Modifiers) {
        let raw = egui::RawInput {
            events: vec![egui::Event::ModifiersChanged(modifiers), egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers }],
            ..Default::default()
        };
        ctx.begin_pass(raw);
        crate::shortcuts::handle(app, ctx);
        ctx.end_pass().textures_delta.clear();
    }

    /// #3102: on a German layout ⇧+ types `*`, which egui reports as the physical `]`, and ⇧- types
    /// `_` as the physical `/`. With the Brush they step its mode, not its hardness; a US ⇧] (`}`)
    /// still raises the hardness.
    #[test]
    fn german_shift_plus_and_minus_step_the_mode_and_us_shift_bracket_keeps_the_hardness() {
        let ctx = egui::Context::default();
        let mut a = app(Tool::Brush);
        a.session.tools.brush.hardness = 0.5;
        let shift = egui::Modifiers::SHIFT;
        let typed = |a: &mut PhotocraftApp, key: Key, physical_key: Key, text: &str| {
            let press = egui::Event::Key { key, physical_key: Some(physical_key), pressed: true, repeat: false, modifiers: shift };
            let raw = egui::RawInput { events: vec![egui::Event::ModifiersChanged(shift), press, egui::Event::Text(text.into())], ..Default::default() };
            ctx.begin_pass(raw);
            crate::shortcuts::handle(a, &ctx);
            ctx.end_pass().textures_delta.clear();
        };
        typed(&mut a, Key::CloseBracket, Key::CloseBracket, "*");
        assert_eq!(a.session.tools.brush.mode, BlendMode::Dissolve);
        assert_eq!(a.session.tools.brush.hardness, 0.5, "not Increase Brush Hardness");
        typed(&mut a, Key::Slash, Key::Slash, "_");
        assert_eq!(a.session.tools.brush.mode, BlendMode::Normal);
        // US: ⇧] arrives as `}` from the physical `]`.
        typed(&mut a, Key::CloseCurlyBracket, Key::CloseBracket, "}");
        assert!(a.session.tools.brush.hardness > 0.5);
        assert_eq!(a.session.tools.brush.mode, BlendMode::Normal);
    }

    #[test]
    fn stepping_wraps_at_both_ends() {
        let m = BlendMode::LAYER_MODES;
        assert_eq!(stepped(&m, BlendMode::Normal, 1), Some(BlendMode::Dissolve));
        assert_eq!(stepped(&m, BlendMode::Normal, -1), m.last().copied());
        assert_eq!(stepped(&m, m[m.len() - 1], 1), Some(BlendMode::Normal));
        assert_eq!(stepped(&[], BlendMode::Normal, 1), None);
    }

    /// #3102: with the Move tool ⇧+ / ⇧− step the active layer's mode, one undo step each; with
    /// the Brush they step the brush's mode and leave the layer alone.
    #[test]
    fn shift_plus_and_minus_step_the_layer_or_the_brush_mode() {
        let ctx = egui::Context::default();
        let mut a = app(Tool::Move);
        let undo = a.session.active().unwrap().history.past_len();
        key(&mut a, &ctx, Key::Plus, egui::Modifiers::SHIFT);
        assert_eq!(layer_blend(&a), BlendMode::Dissolve);
        key(&mut a, &ctx, Key::Plus, egui::Modifiers::SHIFT);
        assert_eq!(layer_blend(&a), BlendMode::Darken);
        key(&mut a, &ctx, Key::Minus, egui::Modifiers::SHIFT);
        assert_eq!(layer_blend(&a), BlendMode::Dissolve);
        assert_eq!(a.session.active().unwrap().history.past_len(), undo + 3, "one undo step per press");
        // Unshifted or ⌘ presses are not blend-mode keys.
        key(&mut a, &ctx, Key::Minus, egui::Modifiers::NONE);
        key(&mut a, &ctx, Key::Plus, egui::Modifiers::COMMAND | egui::Modifiers::SHIFT);
        assert_eq!(layer_blend(&a), BlendMode::Dissolve);
        a.ui.tool = Tool::Brush;
        key(&mut a, &ctx, Key::Minus, egui::Modifiers::SHIFT);
        assert_eq!(a.session.tools.brush.mode, BlendMode::LAYER_MODES[BlendMode::LAYER_MODES.len() - 1]);
        assert_eq!(layer_blend(&a), BlendMode::Dissolve, "the layer is untouched");
        // The Background keeps Normal; no document is a no-op.
        a.ui.tool = Tool::Move;
        let bg = a.session.active().unwrap().doc.layers[0].id;
        a.run("layer.select", json!({"layer": bg.0})).unwrap();
        key(&mut a, &ctx, Key::Plus, egui::Modifiers::SHIFT);
        assert_eq!(a.session.active().unwrap().doc.layers[0].blend, BlendMode::Normal);
        let mut none = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        press(&mut none, 1);
    }
}
