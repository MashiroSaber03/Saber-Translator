//! Move tool marquee (#2844), as in Photoshop: with Auto-Select on (or ⌘ held with it off), a drag
//! that starts where no layer would be picked (an empty, transparent spot, or the pasteboard
//! outside the canvas) draws a box instead of moving anything. Releasing selects every layer whose
//! content touches the box (`layer.selectInRect`; Auto-Select: Group selects their outermost
//! groups), ⇧ adds them to the layers already selected. A click there selects nothing.

use serde_json::json;

use crate::PhotocraftApp;
use crate::canvas::{ToolEvent, ViewXform};

/// The marquee being dragged, in document pixels.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerBox {
    start: [f64; 2],
    end: [f64; 2],
    /// ⇧ at the press: add to the layer selection.
    add: bool,
}

/// A Move-tool press that Auto-Select found nothing under: start the marquee at `p`. The press
/// started a Move gesture's snapping and ⇧/⌥ handling, which a marquee has none of.
pub(crate) fn begin(app: &mut PhotocraftApp, p: [f64; 2], add: bool) {
    app.layer_box = Some(LayerBox { start: p, end: p, add });
    app.prefs_rt.snap = None;
    app.prefs_rt.snap_lines.clear();
    app.move_mods = Default::default();
}

/// No visible layer has pixels at document point `p` (locked ones included).
pub(crate) fn empty_at(app: &mut PhotocraftApp, p: [f64; 2]) -> bool {
    app.run("layer.pickAt", json!({"x": p[0], "y": p[1], "list": true})).is_ok_and(|v| v["layers"].as_array().is_some_and(Vec::is_empty))
}

/// The pointer moves or is released while a marquee is dragged (unsnapped, unconstrained): true
/// when the event was the marquee's.
pub(crate) fn pointer(app: &mut PhotocraftApp, ev: ToolEvent) -> bool {
    let Some(b) = app.layer_box.as_mut() else { return false };
    match ev {
        // A press never continues a marquee (one left over from a lost release).
        ToolEvent::Down { .. } => {
            app.layer_box = None;
            false
        }
        ToolEvent::Move { x, y, .. } => {
            b.end = [x, y];
            true
        }
        ToolEvent::Up { x, y } => {
            let b = LayerBox { end: [x, y], ..*b };
            app.layer_box = None;
            let (w, h) = (b.end[0] - b.start[0], b.end[1] - b.start[1]);
            if w == 0.0 || h == 0.0 {
                return true;
            }
            let target = app.ui.tool_options.move_target.clone();
            let mode = if b.add { "add" } else { "replace" };
            let p = json!({"x": b.start[0], "y": b.start[1], "width": w, "height": h, "target": target, "mode": mode});
            if let Err(e) = app.run("layer.selectInRect", p) {
                app.ui.status = e;
                app.ui.status_error = true;
            }
            true
        }
    }
}

/// The marquee as marching ants.
pub(crate) fn draw(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    let Some(b) = app.layer_box else { return };
    let r = egui::Rect::from_two_pos(xf.to_screen(b.start[0] as f32, b.start[1] as f32), xf.to_screen(b.end[0] as f32, b.end[1] as f32));
    crate::tool_feedback::draw_ants(painter, &[r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom()], true);
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::PhotocraftApp;
    use crate::canvas::{ToolEvent, tool_event};
    use crate::state::Tool;

    fn drag(app: &mut PhotocraftApp, from: [f64; 2], to: [f64; 2], m: egui::Modifiers) {
        tool_event(app, ToolEvent::Down { x: from[0], y: from[1], pressure: 1.0 }, m);
        tool_event(app, ToolEvent::Move { x: to[0], y: to[1], pressure: 1.0 }, m);
        tool_event(app, ToolEvent::Up { x: to[0], y: to[1] }, m);
    }

    /// #2844: a drag from the pasteboard selects the layers its box touches (⇧ adds) and moves
    /// nothing.
    #[test]
    fn a_drag_from_an_empty_spot_selects_the_layers_it_touches() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
        app.sync_views();
        let mut ids = Vec::new();
        for x in [8, 40] {
            app.session.execute("layer.new.layer", json!({})).unwrap();
            app.session.execute("select.rect", json!({"x": x, "y": 8, "width": 16, "height": 16})).unwrap();
            app.session.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
            ids.push(app.session.active().unwrap().active_layer.unwrap());
        }
        app.session.execute("select.deselect", json!({})).unwrap();
        app.ui.tool = Tool::Move;
        app.ui.tool_options.move_auto_select = true;
        drag(&mut app, [-10.0, -10.0], [10.0, 10.0], egui::Modifiers::NONE);
        let st = app.session.active().unwrap();
        assert_eq!(st.selected_layers(), vec![ids[0]], "the box touched the first square only");
        assert_eq!(st.doc.layer(ids[1]).unwrap().surface().unwrap().content_bounds().x0, 40, "nothing moved");
        assert!(app.layer_box.is_none());
        // ⇧ adds the second square, dragged up and left from beyond the right edge.
        drag(&mut app, [70.0, 30.0], [50.0, 20.0], egui::Modifiers::SHIFT);
        let mut sel = app.session.active().unwrap().selected_layers();
        sel.sort_unstable_by_key(|l| l.0);
        assert_eq!(sel, ids);
    }

    /// #4058: a drag that starts on the locked Background draws the layer marquee. It does not
    /// open the move-lock dialog, and releasing selects the unlocked layer the box touches.
    #[test]
    fn a_drag_on_the_locked_background_selects_instead_of_erroring() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 80, "height": 60})).unwrap();
        app.sync_views();
        app.run("layer.new.layer", json!({})).unwrap();
        app.run("select.rect", json!({"x": 40, "y": 8, "width": 16, "height": 16})).unwrap();
        app.run("edit.fill", json!({"color": "#ff0000"})).unwrap();
        app.run("select.deselect", json!({})).unwrap();
        let paint = app.session.active().unwrap().active_layer.unwrap();
        app.ui.tool = Tool::Move;
        app.ui.tool_options.move_auto_select = true;
        drag(&mut app, [8.0, 8.0], [50.0, 20.0], egui::Modifiers::NONE);
        assert!(app.ui.dialogs.iter().all(|d| d.kind != crate::state::DialogKind::Error), "no lock dialog");
        assert!(app.layer_box.is_none());
        assert_eq!(app.session.active().unwrap().selected_layers(), vec![paint]);
    }
}
