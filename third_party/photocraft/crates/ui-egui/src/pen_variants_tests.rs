//! The Freeform Pen and Curvature Pen (#1054): gestures through the canvas commit the same paths
//! the Pen does.

use egui::Modifiers;
use photocraft_doc::{LayerContent, Path};
use serde_json::json;

use crate::PhotocraftApp;
use crate::canvas::{ToolEvent, tool_event};
use crate::state::Tool;

fn app(tool: Tool) -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
    app.run("file.new", json!({"width": 300, "height": 200})).unwrap();
    app.ui.tool = tool;
    app
}

fn work_path(app: &PhotocraftApp) -> Path {
    app.session.active().unwrap().doc.work_path.clone().expect("a work path")
}

fn click(app: &mut PhotocraftApp, x: f64, y: f64) {
    tool_event(app, ToolEvent::Down { x, y, pressure: 1.0 }, Modifiers::NONE);
    tool_event(app, ToolEvent::Up { x, y }, Modifiers::NONE);
}

#[test]
fn freeform_pen_drag_is_fitted_into_one_work_path_or_shape() {
    assert_eq!(Tool::from_name("freeformPen"), Some(Tool::FreeformPen));
    assert_eq!(Tool::FreeformPen.key(), 'P');
    let mut app = app(Tool::FreeformPen);
    let before = app.session.active().unwrap().history.entries().len();
    // A zig-zag with two right-angle corners, sampled every 2 px: 90 pointer samples.
    let corners = [[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [140.0, 80.0]];
    let trace: Vec<[f64; 2]> = corners
        .windows(2)
        .flat_map(|w| (0..30).map(move |k| [w[0][0] + (w[1][0] - w[0][0]) * k as f64 / 30.0, w[0][1] + (w[1][1] - w[0][1]) * k as f64 / 30.0]))
        .chain([corners[3]])
        .collect();
    tool_event(&mut app, ToolEvent::Down { x: 20.0, y: 20.0, pressure: 1.0 }, Modifiers::NONE);
    for p in &trace[1..] {
        tool_event(&mut app, ToolEvent::Move { x: p[0], y: p[1], pressure: 1.0 }, Modifiers::NONE);
    }
    tool_event(&mut app, ToolEvent::Up { x: 140.0, y: 80.0 }, Modifiers::NONE);
    let path = work_path(&app);
    assert_eq!(path.subpaths.len(), 1);
    let sp = &path.subpaths[0];
    assert!(!sp.closed);
    let anchors: Vec<[f64; 2]> = sp.knots.iter().map(|k| [k.anchor.x, k.anchor.y]).collect();
    assert_eq!(anchors, corners, "the fit keeps the corners, not the 90 samples");
    assert_eq!(app.session.active().unwrap().history.entries().len(), before + 1, "one History step");

    // Shape mode: a closed loop (released on its start) becomes a filled shape layer.
    app.ui.tool_options.vector_mode = "shape".into();
    tool_event(&mut app, ToolEvent::Down { x: 200.0, y: 100.0, pressure: 1.0 }, Modifiers::NONE);
    for i in 1..=90 {
        let a = (i as f64 * 4.0).to_radians();
        tool_event(&mut app, ToolEvent::Move { x: 160.0 + 40.0 * a.cos(), y: 100.0 + 40.0 * a.sin(), pressure: 1.0 }, Modifiers::NONE);
    }
    tool_event(&mut app, ToolEvent::Up { x: 200.0, y: 100.0 }, Modifiers::NONE);
    let st = app.session.active().unwrap();
    let LayerContent::Shape(sh) = &st.doc.layer(st.active_layer.unwrap()).unwrap().content else { panic!("no shape layer") };
    assert!(sh.path.subpaths[0].closed && sh.fill.is_some());
    assert!(sh.path.subpaths[0].knots.len() <= 8, "{} anchors for a circle", sh.path.subpaths[0].knots.len());
}

#[test]
fn curvature_pen_clicks_make_a_smooth_path_through_every_point() {
    assert_eq!(Tool::from_name("Curvature Pen Tool"), Some(Tool::CurvaturePen));
    let mut app = app(Tool::CurvaturePen);
    let pts = [[20.0, 150.0], [100.0, 40.0], [180.0, 150.0], [260.0, 60.0]];
    for p in pts {
        click(&mut app, p[0], p[1]);
    }
    assert!(app.session.active().unwrap().doc.work_path.is_none(), "clicks are not History steps");
    // Dragging an existing point moves it.
    tool_event(&mut app, ToolEvent::Down { x: 260.0, y: 60.0, pressure: 1.0 }, Modifiers::NONE);
    tool_event(&mut app, ToolEvent::Move { x: 250.0, y: 70.0, pressure: 1.0 }, Modifiers::NONE);
    tool_event(&mut app, ToolEvent::Up { x: 250.0, y: 70.0 }, Modifiers::NONE);
    crate::vector_ui::pen_commit(&mut app, false); // ↩
    let path = work_path(&app);
    let knots = &path.subpaths[0].knots;
    let anchors: Vec<[f64; 2]> = knots.iter().map(|k| [k.anchor.x, k.anchor.y]).collect();
    assert_eq!(anchors, [pts[0], pts[1], pts[2], [250.0, 70.0]], "the path passes through every point");
    for k in &knots[1..3] {
        // Handles are a third of a chord of at least 100 px, never zero length; collinear to
        // 1e-9 (f64 rounding of the same direction scaled twice).
        let (i, o) = ([k.anchor.x - k.in_ctrl.x, k.anchor.y - k.in_ctrl.y], [k.out_ctrl.x - k.anchor.x, k.out_ctrl.y - k.anchor.y]);
        let (li, lo) = (i[0].hypot(i[1]), o[0].hypot(o[1]));
        assert!(li > 10.0 && lo > 10.0);
        assert!(k.smooth && ((i[0] * o[1] - i[1] * o[0]) / (li * lo)).abs() < 1e-9 && i[0] * o[0] + i[1] * o[1] > 0.0);
    }

    // Double-clicking a point makes it a corner; clicking the first point closes the path.
    app.run("path.delete", json!({"name": "work"})).unwrap();
    for p in &pts[..3] {
        click(&mut app, p[0], p[1]);
    }
    crate::vector_ui::curvature_toggle(&mut app, 100.0, 40.0);
    click(&mut app, 20.0, 150.0);
    let sp = &work_path(&app).subpaths[0];
    assert!(sp.closed && sp.knots.len() == 3);
    assert!(!sp.knots[1].smooth && sp.knots[1].in_ctrl == sp.knots[1].anchor, "a corner point");
    assert!(sp.knots[0].smooth && sp.knots[2].smooth);
}

fn ev(app: &mut PhotocraftApp, kind: &str, x: f64, y: f64, mods: Modifiers) {
    let e = match kind {
        "down" => ToolEvent::Down { x, y, pressure: 1.0 },
        "up" => ToolEvent::Up { x, y },
        _ => ToolEvent::Move { x, y, pressure: 1.0 },
    };
    tool_event(app, e, mods);
}

#[test]
fn freeform_pen_alt_clicks_draw_straight_segments_between_freehand_runs() {
    // Photoshop: ⌥ held at the release keeps the path open, each ⌥-click adds a straight segment,
    // releasing ⌥ with the button down goes on freehand.
    let mut app = app(Tool::FreeformPen);
    app.sync_views();
    let before = app.session.active().unwrap().history.entries().len();
    ev(&mut app, "down", 20.0, 20.0, Modifiers::NONE);
    for x in (22..=80).step_by(2) {
        ev(&mut app, "move", f64::from(x), 20.0, Modifiers::NONE);
    }
    ev(&mut app, "move", 80.0, 20.0, Modifiers::ALT);
    ev(&mut app, "up", 80.0, 20.0, Modifiers::ALT);
    assert!(app.session.active().unwrap().doc.work_path.is_none(), "⌥ at the release keeps the path open");
    // Hovering only moves the rubber band; the click adds a straight segment.
    ev(&mut app, "move", 120.0, 40.0, Modifiers::ALT);
    ev(&mut app, "down", 80.0, 100.0, Modifiers::ALT);
    // ⌥ released with the button down: freehand again, a half circle down to (20, 100).
    for i in 0..=36 {
        let a = (f64::from(i) * 5.0).to_radians();
        ev(&mut app, "move", 50.0 + 30.0 * a.cos(), 100.0 + 30.0 * a.sin(), Modifiers::NONE);
    }
    ev(&mut app, "up", 20.0, 100.0, Modifiers::NONE);
    let path = work_path(&app);
    let knots = &path.subpaths[0].knots;
    let anchors: Vec<[f64; 2]> = knots.iter().map(|k| [k.anchor.x.round(), k.anchor.y.round()]).collect();
    assert_eq!(anchors[..3], [[20.0, 20.0], [80.0, 20.0], [80.0, 100.0]], "{anchors:?}");
    assert_eq!(anchors.last(), Some(&[20.0, 100.0]));
    assert!(!anchors.contains(&[120.0, 40.0]), "the hover is not a point");
    // The ⌥-click segment is straight: no handles on either end of it.
    assert_eq!((knots[1].out_ctrl, knots[2].in_ctrl), (knots[1].anchor, knots[2].anchor));
    assert!(!knots[1].smooth && !knots[2].smooth);
    // The freehand run after it curves.
    assert!(knots[2..].iter().any(|k| k.out_ctrl != k.anchor), "{knots:?}");
    assert_eq!(app.session.active().unwrap().history.entries().len(), before + 1, "one History step");
}

#[test]
fn new_pens_with_command_edit_paths_as_direct_selection() {
    for tool in [Tool::FreeformPen, Tool::CurvaturePen] {
        let mut app = app(Tool::Pen);
        click(&mut app, 30.0, 30.0);
        click(&mut app, 90.0, 30.0);
        crate::vector_ui::pen_commit(&mut app, false);
        app.ui.tool = tool;
        // ⌘-drag (Ctrl off the Mac) on an anchor moves it; the tool stays.
        ev(&mut app, "down", 90.0, 30.0, Modifiers::COMMAND);
        ev(&mut app, "move", 90.0, 70.0, Modifiers::COMMAND);
        ev(&mut app, "up", 90.0, 70.0, Modifiers::COMMAND);
        let wp = work_path(&app);
        assert_eq!(wp.subpaths.len(), 1, "{tool:?} drew nothing");
        assert_eq!([wp.subpaths[0].knots[1].anchor.x, wp.subpaths[0].knots[1].anchor.y], [90.0, 70.0], "{tool:?}");
        assert_eq!(app.ui.tool, tool);
    }
}
