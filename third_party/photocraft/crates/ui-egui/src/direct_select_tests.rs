//! #790: paths can be reshaped after they're drawn, as in Photoshop: the Direct Selection tool
//! (A), and the Pen with ⌘/Ctrl (Direct Selection) or ⌥/Alt (Convert Point). #3204: Convert Point
//! is a tool of its own, in the Pen's flyout.

use egui::Modifiers;
use photocraft_doc::vector::Path;
use photocraft_geom::Point;
use serde_json::json;

use super::{Drag, PathRef};
use crate::PhotocraftApp;
use crate::canvas::{ToolEvent, tool_event};
use crate::state::Tool;

fn app(tool: Tool) -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.session.execute("file.new", json!({"width": 200, "height": 200})).unwrap();
    app.sync_views();
    app.ui.tool = tool;
    app
}

/// A square work path; knot 1 at (120, 40) is smooth with horizontal handles.
fn with_path(app: &mut PhotocraftApp) {
    let path = json!({"subpaths": [{"closed": true, "knots": [
        [40, 40],
        {"anchor": [120, 40], "in": [100, 40], "out": [140, 40], "smooth": true},
        [120, 120],
        [40, 120],
    ]}]});
    app.run("path.set", json!({"name": "work", "path": path})).unwrap();
}

fn work(app: &PhotocraftApp) -> Path {
    app.session.active().unwrap().doc.work_path.clone().unwrap()
}

fn anchor(app: &PhotocraftApp, k: usize) -> Point {
    work(app).subpaths[0].knots[k].anchor
}

fn steps(app: &PhotocraftApp) -> usize {
    app.session.active().unwrap().history.entries().len()
}

fn drag(app: &mut PhotocraftApp, from: [f64; 2], to: [f64; 2], mods: Modifiers) {
    tool_event(app, ToolEvent::Down { x: from[0], y: from[1], pressure: 1.0 }, mods);
    let mid = [(from[0] + to[0]) / 2.0, (from[1] + to[1]) / 2.0];
    tool_event(app, ToolEvent::Move { x: mid[0], y: mid[1], pressure: 1.0 }, mods);
    tool_event(app, ToolEvent::Move { x: to[0], y: to[1], pressure: 1.0 }, mods);
    tool_event(app, ToolEvent::Up { x: to[0], y: to[1] }, mods);
}

fn click(app: &mut PhotocraftApp, at: [f64; 2], mods: Modifiers) {
    drag(app, at, at, mods);
}

#[test]
fn dragging_an_anchor_moves_it_in_one_undoable_step() {
    let mut app = app(Tool::DirectSelection);
    with_path(&mut app);
    let (before, n) = (work(&app), steps(&app));
    drag(&mut app, [40.5, 120.5], [50.5, 130.5], Modifiers::NONE);
    assert_eq!(anchor(&app, 3), Point::new(50.0, 130.0));
    assert_eq!(anchor(&app, 0), Point::new(40.0, 40.0), "only the dragged anchor moved");
    assert_eq!(steps(&app), n + 1, "one history step for the whole drag");
    assert_eq!((app.ui.direct_selection.target, app.ui.direct_selection.anchors.clone()), (Some(PathRef::Work), vec![[0, 3]]));
    assert!(app.session.undo());
    assert_eq!(work(&app), before);
}

#[test]
fn the_preview_is_what_the_drag_commits() {
    let mut app = app(Tool::DirectSelection);
    with_path(&mut app);
    tool_event(&mut app, ToolEvent::Down { x: 80.0, y: 120.0, pressure: 1.0 }, Modifiers::NONE);
    tool_event(&mut app, ToolEvent::Move { x: 80.0, y: 150.0, pressure: 1.0 }, Modifiers::NONE);
    let Some(Drag::Edit { preview: Some(shown), .. }) = app.ui.direct_selection.drag.clone() else { panic!("a segment drag with a preview") };
    assert_ne!(shown, work(&app), "the document changes only on release");
    tool_event(&mut app, ToolEvent::Up { x: 80.0, y: 150.0 }, Modifiers::NONE);
    assert_eq!(work(&app), shown);
    // The straight bottom segment moved with both its anchors.
    assert_eq!((anchor(&app, 2), anchor(&app, 3)), (Point::new(120.0, 150.0), Point::new(40.0, 150.0)));
}

#[test]
fn a_smooth_points_handles_turn_together_and_alt_breaks_them() {
    let mut app = app(Tool::DirectSelection);
    with_path(&mut app);
    click(&mut app, [120.0, 40.0], Modifiers::NONE);
    assert_eq!(app.ui.direct_selection.anchors, vec![[0, 1]], "a click selects the anchor and shows its handles");
    drag(&mut app, [140.0, 40.0], [120.0, 60.0], Modifiers::NONE);
    let k = work(&app).subpaths[0].knots[1];
    assert_eq!(k.out_ctrl, Point::new(120.0, 60.0));
    assert!((k.in_ctrl.x - 120.0).abs() < 1e-9 && (k.in_ctrl.y - 20.0).abs() < 1e-9, "the opposite handle turned with it: {:?}", k.in_ctrl);
    drag(&mut app, [120.0, 20.0], [100.0, 20.0], Modifiers::ALT);
    let k = work(&app).subpaths[0].knots[1];
    assert_eq!((k.in_ctrl, k.out_ctrl, k.smooth), (Point::new(100.0, 20.0), Point::new(120.0, 60.0), false));
}

#[test]
fn marquee_shift_and_empty_clicks_select_like_photoshop() {
    let mut app = app(Tool::DirectSelection);
    with_path(&mut app);
    drag(&mut app, [100.0, 100.0], [180.0, 180.0], Modifiers::NONE);
    assert_eq!(app.ui.direct_selection.anchors, vec![[0, 2]]);
    click(&mut app, [40.0, 120.0], Modifiers::SHIFT);
    assert_eq!(app.ui.direct_selection.anchors, vec![[0, 2], [0, 3]], "⇧-click adds");
    // Dragging one selected anchor moves them all.
    drag(&mut app, [40.0, 120.0], [40.0, 110.0], Modifiers::NONE);
    assert_eq!((anchor(&app, 2), anchor(&app, 3)), (Point::new(120.0, 110.0), Point::new(40.0, 110.0)));
    click(&mut app, [40.0, 110.0], Modifiers::SHIFT);
    assert_eq!(app.ui.direct_selection.anchors, vec![[0, 2]], "⇧-click on a selected anchor deselects it");
    click(&mut app, [80.0, 80.0], Modifiers::NONE);
    assert!(app.ui.direct_selection.anchors.is_empty(), "a click on empty canvas deselects");
    click(&mut app, [80.0, 40.0], Modifiers::ALT);
    assert_eq!(app.ui.direct_selection.anchors.len(), 4, "⌥-click selects the whole subpath");
}

#[test]
fn pen_with_command_ends_the_path_being_drawn_when_pressed_off_it() {
    let mut app = app(Tool::Pen);
    for p in [[30.0, 30.0], [90.0, 30.0]] {
        click(&mut app, p, Modifiers::NONE);
    }
    assert_eq!(app.ui.pen.as_ref().map(|p| p.knots.len()), Some(2));
    // ⌘-click (Ctrl-click off the Mac) on empty canvas: the path is finished, open.
    click(&mut app, [170.0, 170.0], Modifiers::COMMAND);
    assert!(app.ui.pen.is_none());
    assert_eq!(app.ui.tool, Tool::Pen, "the Pen stays the tool");
    assert!(!work(&app).subpaths[0].closed);
    // ⌘-drag then edits the finished path.
    drag(&mut app, [90.0, 30.0], [90.0, 70.0], Modifiers::COMMAND);
    assert_eq!(anchor(&app, 1), Point::new(90.0, 70.0));
    // Without ⌘ the Pen draws again.
    click(&mut app, [150.0, 150.0], Modifiers::NONE);
    assert_eq!(app.ui.pen.as_ref().map(|p| p.knots.len()), Some(1));
}

/// #4022: ⌘/Ctrl-dragging an anchor of the path being drawn moves it in place; the path stays
/// unfinished, outside the document and its history, and the Pen carries on from its last anchor.
#[test]
fn pen_with_command_edits_the_path_being_drawn_in_place() {
    let mut app = app(Tool::Pen);
    for p in [[30.0, 30.0], [90.0, 30.0]] {
        click(&mut app, p, Modifiers::NONE);
    }
    let before = steps(&app);
    drag(&mut app, [90.0, 30.0], [90.0, 70.0], Modifiers::COMMAND);
    let pen = app.ui.pen.clone().expect("still drawing");
    assert_eq!(pen.knots.len(), 2);
    assert_eq!(pen.knots[1][0], [90.0, 70.0], "the anchor moved");
    assert_eq!(pen.knots[0][0], [30.0, 30.0]);
    assert!(app.session.active().unwrap().doc.work_path.is_none(), "not committed");
    assert_eq!(steps(&app), before, "no history step until the path is finished");
    // Let go of ⌘: the next click adds an anchor after the moved one.
    click(&mut app, [150.0, 150.0], Modifiers::NONE);
    assert_eq!(app.ui.pen.as_ref().map(|p| p.knots.len()), Some(3));
    crate::vector_ui::pen_commit(&mut app, false);
    let wp = work(&app);
    assert_eq!(wp.subpaths[0].knots.len(), 3);
    assert_eq!(wp.subpaths[0].knots[1].anchor, Point::new(90.0, 70.0));
    assert_eq!(app.ui.direct_selection.target, None, "the Pen selection is gone with the Pen path");
}

/// ⌘/Ctrl-click selects an anchor of the path being drawn, showing its handles; ⌘-dragging one
/// turns it and the other follows (a smooth point), still without committing.
#[test]
fn pen_with_command_turns_a_handle_of_the_path_being_drawn() {
    let mut app = app(Tool::Pen);
    click(&mut app, [30.0, 30.0], Modifiers::NONE);
    drag(&mut app, [90.0, 30.0], [110.0, 30.0], Modifiers::NONE);
    assert_eq!(app.ui.pen.as_ref().unwrap().knots[1], [[90.0, 30.0], [70.0, 30.0], [110.0, 30.0]]);
    click(&mut app, [90.0, 30.0], Modifiers::COMMAND);
    assert_eq!(app.ui.direct_selection.target, Some(PathRef::Pen));
    assert_eq!(app.ui.direct_selection.anchors, vec![[0, 1]]);
    drag(&mut app, [110.0, 30.0], [110.0, 50.0], Modifiers::COMMAND);
    let k = app.ui.pen.as_ref().expect("still drawing").knots[1];
    assert_eq!(k[2], [110.0, 50.0], "the outgoing handle turned");
    let (o, i) = ([k[2][0] - k[0][0], k[2][1] - k[0][1]], [k[1][0] - k[0][0], k[1][1] - k[0][1]]);
    assert!((o[0] * i[1] - o[1] * i[0]).abs() < 1e-6 && o[0] * i[0] + o[1] * i[1] < 0.0, "the incoming handle stays opposite: {k:?}");
    assert!(app.session.active().unwrap().doc.work_path.is_none());
}

/// The Curvature Pen too: ⌘/Ctrl-dragging an anchor of its path in progress moves it, and the
/// curve through the anchors follows.
#[test]
fn curvature_pen_with_command_moves_an_anchor_of_the_path_being_drawn() {
    let mut app = app(Tool::CurvaturePen);
    for p in [[30.0, 30.0], [90.0, 30.0], [150.0, 60.0]] {
        click(&mut app, p, Modifiers::NONE);
    }
    let before = app.ui.pen.clone().unwrap();
    drag(&mut app, [90.0, 30.0], [90.0, 10.0], Modifiers::COMMAND);
    let pen = app.ui.pen.clone().expect("still drawing");
    assert!(pen.curvature && pen.knots.len() == 3);
    assert_eq!(pen.knots[1][0], [90.0, 10.0]);
    assert_ne!(pen.knots[1][1], before.knots[1][1], "the handles follow the moved anchor");
    assert!(app.session.active().unwrap().doc.work_path.is_none());
}

#[test]
fn pen_with_alt_converts_points() {
    let mut app = app(Tool::Pen);
    with_path(&mut app);
    click(&mut app, [120.0, 40.0], Modifiers::ALT);
    assert_eq!(work(&app).subpaths[0].knots[1], photocraft_doc::Knot::corner(120.0, 40.0), "⌥-click makes a corner");
    drag(&mut app, [40.0, 120.0], [40.0, 140.0], Modifiers::ALT);
    let k = work(&app).subpaths[0].knots[3];
    assert_eq!((k.in_ctrl, k.out_ctrl, k.smooth), (Point::new(40.0, 100.0), Point::new(40.0, 140.0), true), "⌥-drag pulls out smooth handles");
    assert!(app.ui.pen.is_none(), "no pen path was started");
    // ⌥ off the path is a plain Pen click.
    click(&mut app, [170.0, 170.0], Modifiers::ALT);
    assert!(app.ui.pen.is_some());
}

#[test]
fn shape_layer_paths_are_edited_too() {
    let mut app = app(Tool::DirectSelection);
    let id = app.run("shape.create", json!({"kind": "rect", "rect": [20, 20, 60, 60], "fill": "#3070c0"})).unwrap()["layer"].as_u64().unwrap();
    drag(&mut app, [80.0, 80.0], [100.0, 90.0], Modifiers::NONE);
    assert_eq!(app.ui.direct_selection.target, Some(PathRef::Layer(id)));
    let info = app.run("shape.info", json!({"layer": id})).unwrap();
    assert_eq!(info["path"]["subpaths"][0]["knots"][2]["anchor"], json!([100.0, 90.0]));
    let b: [i64; 4] = serde_json::from_value(info["bounds"].clone()).unwrap();
    assert!(b[0] + b[2] >= 100 && b[1] + b[3] >= 90, "the shape re-rendered: {b:?}");
}

#[test]
fn the_convert_point_tool_converts_points_without_a_modifier() {
    let mut app = app(Tool::ConvertPoint);
    with_path(&mut app);
    let n = steps(&app);
    click(&mut app, [120.0, 40.0], Modifiers::NONE);
    assert_eq!(work(&app).subpaths[0].knots[1], photocraft_doc::Knot::corner(120.0, 40.0), "a click makes a corner");
    assert_eq!(steps(&app), n + 1);
    assert_eq!(app.ui.direct_selection.anchors, vec![[0, 1]], "the converted anchor is selected");
    drag(&mut app, [40.0, 120.0], [40.0, 140.0], Modifiers::NONE);
    let k = work(&app).subpaths[0].knots[3];
    assert_eq!((k.in_ctrl, k.out_ctrl, k.smooth), (Point::new(40.0, 100.0), Point::new(40.0, 140.0), true), "a drag pulls out smooth handles");
    assert_eq!(steps(&app), n + 2, "one history step per press");
    assert_eq!(app.ui.direct_selection.anchors, vec![[0, 3]]);
    assert!(app.ui.pen.is_none(), "the tool never starts a pen path");
    assert_eq!(app.ui.tool, Tool::ConvertPoint);
    assert!(app.session.undo() && app.session.undo());
    assert!(work(&app).subpaths[0].knots[1].smooth, "undo restores the smooth point");
}

#[test]
fn the_convert_point_tool_moves_a_handle_alone_and_ignores_empty_canvas_and_segments() {
    let mut app = app(Tool::DirectSelection);
    with_path(&mut app);
    // Select the smooth anchor with Direct Selection so its handles show.
    click(&mut app, [120.0, 40.0], Modifiers::NONE);
    app.ui.tool = Tool::ConvertPoint;
    drag(&mut app, [140.0, 40.0], [140.0, 70.0], Modifiers::NONE);
    let k = work(&app).subpaths[0].knots[1];
    assert_eq!((k.out_ctrl, k.in_ctrl, k.smooth), (Point::new(140.0, 70.0), Point::new(100.0, 40.0), false), "the handle moved without its twin");
    // Nothing happens on empty canvas or on a segment, and the tool keeps the event.
    let (before, n) = (work(&app), steps(&app));
    click(&mut app, [80.0, 80.0], Modifiers::NONE);
    drag(&mut app, [120.0, 80.0], [120.0, 100.0], Modifiers::NONE);
    assert!(app.ui.direct_selection.drag.is_none(), "no marquee");
    assert_eq!(app.ui.direct_selection.anchors, vec![[0, 1]], "the selection stays");
    assert_eq!((work(&app), steps(&app)), (before, n));
    assert!(app.ui.pen.is_none() && app.drag.is_none(), "the press was not handed to another tool");
}

#[test]
fn the_convert_point_tool_with_command_is_direct_selection() {
    let mut app = app(Tool::ConvertPoint);
    with_path(&mut app);
    drag(&mut app, [40.5, 120.5], [50.5, 130.5], Modifiers::COMMAND);
    assert_eq!(anchor(&app, 3), Point::new(50.0, 130.0), "⌘-drag moves the anchor");
    assert!(work(&app).subpaths[0].knots[3].in_ctrl == work(&app).subpaths[0].knots[3].anchor, "and does not pull out handles");
    // A handle drag with ⌘ turns the smooth pair together, like Direct Selection.
    click(&mut app, [120.0, 40.0], Modifiers::COMMAND);
    assert_eq!(app.ui.direct_selection.anchors, vec![[0, 1]]);
    drag(&mut app, [140.0, 40.0], [120.0, 60.0], Modifiers::COMMAND);
    let k = work(&app).subpaths[0].knots[1];
    assert!((k.in_ctrl.x - 120.0).abs() < 1e-9 && (k.in_ctrl.y - 20.0).abs() < 1e-9, "the twin handle turned with it: {:?}", k.in_ctrl);
}

#[test]
fn the_convert_point_tool_shows_the_path_overlay() {
    let mut app = app(Tool::ConvertPoint);
    assert!(super::shows(&app, Modifiers::NONE), "anchors are drawn before any press");
    app.ui.tool = Tool::Pen;
    assert!(!super::shows(&app, Modifiers::NONE));
    assert!(super::shows(&app, Modifiers::COMMAND));
}

/// One key press through the canvas arrow keys, as egui delivers it.
fn press(app: &mut PhotocraftApp, key: egui::Key, mods: Modifiers) -> bool {
    let ctx = egui::Context::default();
    ctx.input_mut(|i| {
        i.modifiers = mods;
        i.events.push(egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers: mods });
    });
    crate::move_mods::arrow_keys(app, &ctx)
}

/// #4486: the arrow keys nudge the selected anchors 1 px (⇧ 10 px), one history step per press,
/// as in Photoshop. With no anchor selected they leave the path alone.
#[test]
fn arrow_keys_nudge_the_selected_anchors() {
    let mut app = app(Tool::DirectSelection);
    with_path(&mut app);
    // The path stays targeted after a click on empty canvas, but no anchor is selected.
    click(&mut app, [40.0, 120.0], Modifiers::NONE);
    click(&mut app, [80.0, 80.0], Modifiers::NONE);
    assert_eq!(app.ui.direct_selection.target, Some(PathRef::Work));
    let n = steps(&app);
    assert!(!press(&mut app, egui::Key::ArrowRight, Modifiers::NONE), "no anchor selected: not Direct Selection's key");
    assert_eq!(steps(&app), n);
    click(&mut app, [40.0, 120.0], Modifiers::NONE);
    click(&mut app, [120.0, 120.0], Modifiers::SHIFT);
    let (before, n) = (work(&app), steps(&app));
    assert!(press(&mut app, egui::Key::ArrowRight, Modifiers::NONE));
    assert_eq!((anchor(&app, 2), anchor(&app, 3)), (Point::new(121.0, 120.0), Point::new(41.0, 120.0)));
    assert!(press(&mut app, egui::Key::ArrowDown, Modifiers::SHIFT));
    assert_eq!((anchor(&app, 2), anchor(&app, 3)), (Point::new(121.0, 130.0), Point::new(41.0, 130.0)), "⇧ nudges 10 px");
    assert_eq!(anchor(&app, 0), Point::new(40.0, 40.0), "unselected anchors stay");
    assert_eq!(steps(&app), n + 2, "one history step per press");
    assert!(app.session.undo() && app.session.undo());
    assert_eq!(work(&app), before);
}
