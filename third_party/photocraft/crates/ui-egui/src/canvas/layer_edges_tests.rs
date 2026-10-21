//! Layer Edges must frame the pixels shown during a Move drag, rather than the last commit.

use super::*;
use serde_json::json;

const ORIGINAL: DRect = DRect::new(100, 100, 300, 260);

fn app(depth: u32) -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 600, "height": 400, "depth": depth, "background": "transparent"})).unwrap();
    app.run("select.rect", json!({"x": 100, "y": 100, "width": 200, "height": 160})).unwrap();
    app.run("edit.fill", json!({"color": "#e05030"})).unwrap();
    app.run("select.deselect", json!({})).unwrap();
    app.sync_views();
    app.ui.tool = Tool::Move;
    app.ui.tool_options.move_auto_select = false;
    app.ui.extras.rulers = false;
    app.ui.extras.snap = false;
    app.ui.view.show.smart_guides = false;
    app.ui.view.show.layer_edges = true;
    app
}

/// Inspect the actual painted Layer Edges rectangle, distinct from the snap/transform box.
fn assert_edges(app: &mut PhotocraftApp, ctx: &egui::Context, idx: usize, primary: bool, expected: DRect) {
    let rect = Rect::from_min_size(Pos2::ZERO, vec2(600.0, 400.0));
    let view = View { zoom: 0.75, center: [300.0, 200.0], doc_size: [600, 400], fit_pending: false, ..Default::default() };
    let mut output = ctx.run_ui(egui::RawInput { screen_rect: Some(rect), ..Default::default() }, |ui| {
        canvas_view(app, ui, idx, rect, view.clone(), primary);
    });
    output.textures_delta.clear();
    let xf = ViewXform { rect, zoom: view.zoom / ctx.pixels_per_point(), center: view.center, flip: false, rotation: 0.0, aspect: 1.0 };
    let edges: Vec<_> = output
        .shapes
        .iter()
        .filter_map(|s| match &s.shape {
            egui::Shape::Rect(r) if r.stroke.color == LAYER_EDGES && r.stroke_kind == egui::StrokeKind::Outside => Some(r.rect),
            _ => None,
        })
        .collect();
    assert_eq!(edges, vec![xf.doc_rect(expected)]);
}

fn context(ppp: f32) -> egui::Context {
    let ctx = egui::Context::default();
    PhotocraftApp::setup_context(&ctx, crate::theme::ThemeKind::ALL[0]);
    ctx.set_pixels_per_point(ppp);
    ctx
}

#[test]
fn layer_edges_follow_live_move_then_release_and_undo() {
    for depth in [8, 16, 32] {
        for ppp in [1.0, 2.0] {
            let mut app = app(depth);
            let ctx = context(ppp);
            app.ui.tool_options.move_show_transform = true;
            assert_edges(&mut app, &ctx, 0, true, ORIGINAL);
            let before = app.session.active().unwrap().history.past_len();
            tool_event(&mut app, ToolEvent::Down { x: 180.0, y: 180.0, pressure: 1.0 }, egui::Modifiers::NONE);
            for (x, y) in [(210.0, 160.0), (260.0, 220.0)] {
                tool_event(&mut app, ToolEvent::Move { x, y, pressure: 1.0 }, egui::Modifiers::NONE);
                assert_edges(&mut app, &ctx, 0, true, ORIGINAL.translate((x - 180.0) as i32, (y - 180.0) as i32));
                let st = app.session.active().unwrap();
                assert_eq!(st.history.past_len(), before, "preview must not add an undo step");
                assert_eq!(st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().content_bounds(), ORIGINAL);
            }
            tool_event(&mut app, ToolEvent::Up { x: 260.0, y: 220.0 }, egui::Modifiers::NONE);
            assert_edges(&mut app, &ctx, 0, true, ORIGINAL.translate(80, 40));
            assert_eq!(app.session.active().unwrap().history.past_len(), before + 1);
            app.run("edit.undo", json!({})).unwrap();
            assert_edges(&mut app, &ctx, 0, true, ORIGINAL);
        }
    }
}

#[test]
fn layer_edges_follow_the_displayed_document_in_secondary_views() {
    let mut app = app(8);
    let ctx = context(1.0);
    app.run("file.new", json!({"width": 600, "height": 400, "background": "transparent"})).unwrap();
    app.run("select.rect", json!({"x": 100, "y": 100, "width": 200, "height": 160})).unwrap();
    app.run("edit.fill", json!({"color": "#3060e0"})).unwrap();
    app.run("select.deselect", json!({})).unwrap();
    app.sync_views();
    tool_event(&mut app, ToolEvent::Down { x: 180.0, y: 180.0, pressure: 1.0 }, egui::Modifiers::NONE);
    tool_event(&mut app, ToolEvent::Move { x: 260.0, y: 220.0, pressure: 1.0 }, egui::Modifiers::NONE);
    assert_edges(&mut app, &ctx, 1, false, ORIGINAL.translate(80, 40));
    assert_edges(&mut app, &ctx, 0, false, ORIGINAL);
}

#[test]
fn layer_edges_stay_with_unmoved_pixels_when_live_preview_is_disabled() {
    let mut app = app(8);
    let ctx = context(1.0);
    app.session.edit_prefs(|p| p.interface.show_bounding_box_when_dragging_layer = true);
    tool_event(&mut app, ToolEvent::Down { x: 180.0, y: 180.0, pressure: 1.0 }, egui::Modifiers::NONE);
    tool_event(&mut app, ToolEvent::Move { x: 260.0, y: 220.0, pressure: 1.0 }, egui::Modifiers::NONE);
    assert_edges(&mut app, &ctx, 0, true, ORIGINAL);
    tool_event(&mut app, ToolEvent::Up { x: 260.0, y: 220.0 }, egui::Modifiers::NONE);
    assert_edges(&mut app, &ctx, 0, true, ORIGINAL.translate(80, 40));
}
