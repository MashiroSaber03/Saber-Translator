//! Free Transform on type layers (#2630): the Edit menu offers it, a selection doesn't narrow it
//! (type can't be partly transformed, so the whole layer moves), and Distort / Perspective are
//! greyed as in Photoshop until the type is rasterized.

use serde_json::json;

use crate::PhotocraftApp;
use crate::state::Tool;

/// A 200 × 120 document with a type layer "Hello" active and the Rectangular Marquee chosen.
fn app() -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 200, "height": 120})).unwrap();
    app.sync_views();
    app.run("type.create", json!({"x": 20, "y": 60, "text": "Hello", "size": 36})).unwrap();
    app.ui.tool = Tool::RectMarquee;
    app
}

fn type_bounds(app: &PhotocraftApp) -> photocraft_geom::Rect {
    let st = app.session.active().unwrap();
    st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().content_bounds()
}

fn select_left_half(app: &mut PhotocraftApp) {
    let b = type_bounds(app);
    app.run("select.rect", json!({"x": b.x0, "y": b.y0, "width": b.width() / 2, "height": b.height()})).unwrap();
}

#[test]
fn type_layers_offer_free_transform_but_not_distort_or_perspective() {
    let mut app = app();
    for selection in [false, true] {
        if selection {
            select_left_half(&mut app);
        }
        for id in ["edit.freeTransform", "edit.transform.scale", "edit.transform.rotate", "edit.transform.skew"] {
            assert!(crate::menus::is_enabled(&app, id), "{id} (selection: {selection})");
        }
        for id in ["edit.transform.distort", "edit.transform.perspective"] {
            assert!(!crate::menus::is_enabled(&app, id), "{id} is greyed for editable type (selection: {selection})");
        }
    }
    // The same rows in the marquee's canvas context menu.
    let menu = crate::canvas_tool_menu::CanvasToolMenu {
        pos: [10.0, 10.0],
        tool: Tool::RectMarquee,
        has_selection: true,
        has_path: false,
        path_name: None,
        transform: false,
        pasteboard: false,
        anchor: None,
        text: false,
        spelling: None,
    };
    assert!(crate::canvas_tool_menu::available(&app, &menu, "edit.freeTransform"));
    assert!(!crate::canvas_tool_menu::available(&app, &menu, "edit.transform.distort"));
    assert!(!crate::canvas_tool_menu::available(&app, &menu, "edit.transform.perspective"));
    // Rasterized, the type is pixels and can be distorted.
    app.run("layer.rasterize.type", json!({})).unwrap();
    assert!(crate::menus::is_enabled(&app, "edit.transform.distort") && crate::menus::is_enabled(&app, "edit.transform.perspective"));
}

#[test]
fn free_transform_with_a_selection_frames_and_moves_the_whole_type_layer() {
    let mut app = app();
    let whole = type_bounds(&app);
    select_left_half(&mut app);
    let ctx = egui::Context::default();
    crate::menus::invoke(&mut app, &ctx, "edit.freeTransform", json!({})).unwrap();
    let t = app.ui.transform.as_ref().unwrap();
    assert_eq!(t.rect, [whole.x0 as f64, whole.y0 as f64, whole.x1 as f64, whole.y1 as f64], "the box frames the whole text");
    // The preview moves the whole layer (it is hidden under the box), not half of its pixels.
    let pv = app.transform_preview.as_ref().unwrap();
    let id = app.session.active().unwrap().active_layer.unwrap();
    let under = pv.doc.layer(id).unwrap();
    assert!(!under.visible, "the type layer is hidden while its preview moves");
    assert_eq!(under.surface().unwrap().content_bounds(), whole, "nothing lifted out of the type");
    // Move the box 10 px right: the text moves 10 px, unscaled.
    if let Some(t) = app.ui.transform.as_mut() {
        t.quad = t.quad.map(|q| [q[0] + 10.0, q[1]]);
    }
    crate::transform_tool::commit(&mut app);
    assert_eq!(app.ui.status, "");
    let moved = type_bounds(&app);
    assert!(moved.x0.abs_diff(whole.x0 + 10) <= 1 && moved.width().abs_diff(whole.width()) <= 1, "{whole:?} -> {moved:?}");
    let st = app.session.active().unwrap();
    assert!(matches!(st.doc.layer(id).unwrap().content, photocraft_doc::LayerContent::Text(_)), "still editable type");
}

/// ⌘-drag (distort) and ⌘⌥⇧-drag (perspective) of a corner on editable type scale like a plain
/// corner drag (#3010): the box stays a parallelogram and the commit keeps the type live, instead
/// of a distorted box the engine then refuses.
#[test]
fn distort_and_perspective_corner_drags_scale_editable_type() {
    use crate::canvas::ToolEvent;
    let ctx = egui::Context::default();
    let cmd = egui::Modifiers::COMMAND;
    for mods in [cmd, cmd | egui::Modifiers::ALT | egui::Modifiers::SHIFT] {
        let mut app = app();
        crate::menus::invoke(&mut app, &ctx, "edit.freeTransform", json!({})).unwrap();
        let q0 = app.ui.transform.as_ref().unwrap().quad;
        let [x, y] = q0[1];
        crate::transform_tool::pointer(&mut app, ToolEvent::Down { x, y, pressure: 1.0 }, mods);
        crate::transform_tool::pointer(&mut app, ToolEvent::Move { x: x + 15.0, y: y - 10.0, pressure: 1.0 }, mods);
        crate::transform_tool::pointer(&mut app, ToolEvent::Up { x: x + 15.0, y: y - 10.0 }, mods);
        let q = app.ui.transform.as_ref().unwrap().quad;
        assert_ne!(q, q0, "the drag scaled the box ({mods:?})");
        let skew = (0..2).map(|k| (q[0][k] + q[2][k] - q[1][k] - q[3][k]).abs()).fold(0.0, f64::max);
        assert!(skew < 1e-6, "the box stays a parallelogram ({mods:?}): {q:?}");
        crate::transform_tool::commit(&mut app);
        assert_eq!(app.ui.status, "", "{mods:?}");
        let st = app.session.active().unwrap();
        assert!(matches!(st.doc.layer(st.active_layer.unwrap()).unwrap().content, photocraft_doc::LayerContent::Text(_)), "still editable type");
    }
}
