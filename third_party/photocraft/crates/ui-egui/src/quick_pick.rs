//! ⌘⌥⌃-click on the canvas, with any tool: select the topmost visible layer with pixels under the
//! pointer (`layer.pickAt`), without switching to the Move tool. The whole press–drag–release is
//! swallowed so the active tool never sees it; it runs before the ⌃⌥ brush-resize gesture, which
//! would otherwise claim it.
//!
//! Platforms: the gesture is macOS-only, because ⌘ is the only key that tells it apart from the
//! ⌃⌥ brush resize. egui never reports `mac_cmd` on Windows or Linux, so there it is a no-op and
//! ⌃⌥-drag keeps resizing the brush; the Move tool's Auto-Select picks layers on every platform.
//! A failed pick is shown as a status-bar error and never panics.

use serde_json::json;

use crate::PhotocraftApp;
use crate::canvas::ToolEvent;

/// ⌘ + ⌥ + ⌃ held together. Always false off macOS, where egui never sets `mac_cmd`.
pub fn is_gesture(mods: egui::Modifiers) -> bool {
    mods.mac_cmd && mods.alt && mods.ctrl
}

/// Handles one canvas pointer event; true when it was consumed.
pub fn pointer(app: &mut PhotocraftApp, ev: ToolEvent, mods: egui::Modifiers) -> bool {
    match ev {
        ToolEvent::Down { x, y, .. } => {
            if app.drag.is_some() || app.session.active().is_none() || !is_gesture(mods) {
                return false;
            }
            if let Err(e) = app.run("layer.pickAt", json!({"x": x, "y": y, "target": "layer", "mode": "replace"})) {
                app.ui.status = e;
                app.ui.status_error = true;
            }
            app.quick_pick = true;
            true
        }
        ToolEvent::Move { .. } => app.quick_pick,
        ToolEvent::Up { .. } => std::mem::take(&mut app.quick_pick),
    }
}

#[cfg(test)]
mod tests {
    use egui::Modifiers;
    use photocraft_geom::Rect;
    use serde_json::json;

    use super::*;
    use crate::canvas::tool_event;
    use crate::state::Tool;

    const PICK: Modifiers = Modifiers { alt: true, ctrl: true, shift: false, mac_cmd: true, command: true };

    /// A 64×64 document with A painted at (8..24)² and B at (40..56)²; B is active.
    fn app() -> (PhotocraftApp, photocraft_doc::LayerId) {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
        app.sync_views();
        let mut first = None;
        for at in [8, 40] {
            app.session.execute("layer.new.layer", json!({})).unwrap();
            app.session
                .edit("paint", |doc, a| {
                    doc.layer_mut(a.unwrap()).unwrap().surface_mut().unwrap().fill_rect(Rect::new(at, at, at + 16, at + 16), &[1.0, 0.0, 0.0, 1.0]);
                    Ok(())
                })
                .unwrap();
            first.get_or_insert(app.session.active().unwrap().active_layer.unwrap());
        }
        (app, first.unwrap())
    }

    #[test]
    fn picks_the_layer_under_the_pointer_with_a_painting_tool() {
        let (mut app, a) = app();
        app.ui.tool = Tool::Brush;
        let before = app.session.active().unwrap().history.past_len();
        tool_event(&mut app, ToolEvent::Down { x: 12.0, y: 12.0, pressure: 1.0 }, PICK);
        tool_event(&mut app, ToolEvent::Move { x: 20.0, y: 20.0, pressure: 1.0 }, PICK);
        tool_event(&mut app, ToolEvent::Up { x: 20.0, y: 20.0 }, PICK);
        let st = app.session.active().unwrap();
        assert_eq!(st.active_layer, Some(a));
        assert!(app.drag.is_none() && app.brush_resize.is_none(), "neither painted nor resized the brush");
        assert_eq!(st.history.past_len(), before, "no paint stroke was recorded");
    }

    #[test]
    fn needs_all_three_modifiers() {
        let (mut app, a) = app();
        app.ui.tool = Tool::RectMarquee;
        let no_ctrl = Modifiers { ctrl: false, ..PICK };
        assert!(!pointer(&mut app, ToolEvent::Down { x: 12.0, y: 12.0, pressure: 1.0 }, no_ctrl));
        assert_ne!(app.session.active().unwrap().active_layer, Some(a));
    }

    /// Windows and Linux report Ctrl+Alt (+Shift) but never `mac_cmd`: the click falls through to
    /// the tool and the ⌃⌥ brush resize instead of picking a layer.
    #[test]
    fn is_a_no_op_without_the_mac_command_key() {
        let (mut app, a) = app();
        app.ui.tool = Tool::Brush;
        let pc = Modifiers { alt: true, ctrl: true, shift: false, mac_cmd: false, command: true };
        for mods in [pc, Modifiers { shift: true, ..pc }] {
            assert!(!pointer(&mut app, ToolEvent::Down { x: 12.0, y: 12.0, pressure: 1.0 }, mods));
            assert!(!app.quick_pick);
            assert_ne!(app.session.active().unwrap().active_layer, Some(a));
        }
    }

    #[test]
    fn a_failed_pick_is_a_status_error_and_swallows_the_click() {
        let (mut app, _) = app();
        // Far outside the canvas no layer has pixels: the pick fails gracefully.
        assert!(pointer(&mut app, ToolEvent::Down { x: 1e9, y: -1e9, pressure: 1.0 }, PICK));
        assert!(pointer(&mut app, ToolEvent::Up { x: 1e9, y: -1e9 }, PICK));
        assert!(!app.quick_pick);
    }
}
