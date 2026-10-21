//! #571: tools leave hidden layers alone, as in Photoshop. Auto-Select never picks a hidden layer,
//! and painting on or moving a hidden target layer is refused at the press with Photoshop's
//! "the target layer is hidden" instead of changing pixels nobody can see.

use photocraft_doc::LayerId;
use photocraft_geom::Rect;
use serde_json::json;

use crate::PhotocraftApp;
use crate::canvas::{ToolEvent, tool_event};
use crate::state::Tool;

/// A 64×64 document with a painted layer covering (8..24)², hidden and active.
fn app(tool: Tool) -> (PhotocraftApp, LayerId) {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.session.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
    app.sync_views();
    app.session.execute("layer.new.layer", json!({})).unwrap();
    app.session
        .edit("paint", |doc, a| {
            doc.layer_mut(a.unwrap()).unwrap().surface_mut().unwrap().fill_rect(Rect::new(8, 8, 24, 24), &[1.0, 0.0, 0.0, 1.0]);
            Ok(())
        })
        .unwrap();
    let id = app.session.active().unwrap().active_layer.unwrap();
    app.session.execute("layer.setProps", json!({"layer": id.0, "visible": false})).unwrap();
    app.ui.tool = tool;
    app.ui.extras.snap = false;
    (app, id)
}

fn drag(app: &mut PhotocraftApp, from: [f64; 2], to: [f64; 2]) {
    let m = egui::Modifiers::NONE;
    tool_event(app, ToolEvent::Down { x: from[0], y: from[1], pressure: 1.0 }, m);
    tool_event(app, ToolEvent::Move { x: to[0], y: to[1], pressure: 1.0 }, m);
    tool_event(app, ToolEvent::Up { x: to[0], y: to[1] }, m);
}

fn steps(app: &PhotocraftApp) -> usize {
    app.session.active().unwrap().history.entries().len()
}

fn bounds(app: &PhotocraftApp, id: LayerId) -> Rect {
    app.session.active().unwrap().doc.layer(id).unwrap().surface().unwrap().content_bounds()
}

#[test]
fn auto_select_skips_hidden_layers() {
    let (mut app, id) = app(Tool::Move);
    let bg = app.session.active().unwrap().doc.layers[0].id;
    drag(&mut app, [16.0, 16.0], [30.0, 16.0]);
    assert_eq!(app.session.active().unwrap().active_layer, Some(bg), "the visible layer under the pointer was picked");
    assert_eq!(bounds(&app, id), Rect::new(8, 8, 24, 24), "the hidden layer stayed put");
}

#[test]
fn moving_a_hidden_target_layer_is_refused() {
    let (mut app, id) = app(Tool::Move);
    app.ui.tool_options.move_auto_select = false;
    let n = steps(&app);
    drag(&mut app, [16.0, 16.0], [30.0, 16.0]);
    assert!(app.ui.status_error && app.ui.status.contains("target layer is hidden"), "{}", app.ui.status);
    assert_eq!((bounds(&app, id), steps(&app)), (Rect::new(8, 8, 24, 24), n));
}

#[test]
fn painting_a_hidden_layer_is_refused_at_the_press() {
    for tool in [Tool::Brush, Tool::Eraser, Tool::ColorReplacement, Tool::PaintBucket, Tool::Gradient, Tool::CloneStamp] {
        let (mut app, id) = app(tool);
        // The live Gradient makes a new Gradient Fill layer; the classic one paints the target.
        app.ui.tool_options.gradient_classic = true;
        let n = steps(&app);
        tool_event(&mut app, ToolEvent::Down { x: 30.0, y: 30.0, pressure: 1.0 }, egui::Modifiers::NONE);
        assert!(app.drag.is_none() && app.live_stroke.is_none(), "{tool:?}: no stroke started");
        assert!(app.ui.status.contains("target layer is hidden"), "{tool:?}: {}", app.ui.status);
        tool_event(&mut app, ToolEvent::Up { x: 40.0, y: 40.0 }, egui::Modifiers::NONE);
        assert_eq!((bounds(&app, id), steps(&app)), (Rect::new(8, 8, 24, 24), n), "{tool:?}");
        // Shown again, the same press paints.
        app.session.execute("layer.setProps", json!({"layer": id.0, "visible": true})).unwrap();
        app.ui.status.clear();
        tool_event(&mut app, ToolEvent::Down { x: 30.0, y: 30.0, pressure: 1.0 }, egui::Modifiers::NONE);
        assert!(!app.ui.status.contains("hidden"), "{tool:?}");
    }
}

#[test]
fn the_live_gradient_still_works_over_a_hidden_layer() {
    // The live Gradient adds a Gradient Fill layer rather than painting the target, so a hidden
    // target doesn't stop it; the hidden layer's pixels stay as they were.
    let (mut app, id) = app(Tool::Gradient);
    app.ui.tool_options.gradient_classic = false;
    let layers = app.session.active().unwrap().doc.layers.len();
    drag(&mut app, [4.0, 30.0], [60.0, 30.0]);
    assert!(!app.ui.status.contains("hidden"), "{}", app.ui.status);
    assert_eq!(app.session.active().unwrap().doc.layers.len(), layers + 1, "a Gradient Fill layer was added");
    assert_eq!(bounds(&app, id), Rect::new(8, 8, 24, 24));
}
