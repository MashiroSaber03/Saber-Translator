//! Polygonal Lasso through the real canvas: a double-click closes the polygon, and a
//! new-selection polygon hides the selection it replaces while it is drawn (⇧ or ⌥ at the first
//! click keep it, to add to or subtract from), replacing it in one history step.

use egui::{Modifiers, PointerButton, Pos2, vec2};
use egui_kittest::Harness;
use photocraft_geom::Rect;
use serde_json::json;

use crate::PhotocraftApp;
use crate::canvas::ViewXform;
use crate::state::Tool;

fn harness() -> Harness<'static, PhotocraftApp> {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 400, "height": 300})).unwrap();
    app.sync_views();
    app.ui.extras.rulers = false;
    app.ui.tool = Tool::PolygonLasso;
    // 60 fps, so a double-click's two clicks fall inside egui's 0.3 s window.
    let mut h = Harness::builder().with_size(vec2(1000.0, 700.0)).with_step_dt(1.0 / 60.0).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            let ctx = ui.ctx().clone();
            if !ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            crate::shortcuts::handle(app, &ctx);
            egui::CentralPanel::default().show(ui, |ui| crate::canvas::document_area(app, ui));
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
    h.run_steps(4);
    // 100 %: one document pixel per point.
    let v = &mut h.state_mut().ui.views[0];
    v.zoom = 1.0;
    v.center = [200.0, 150.0];
    v.fit_pending = false;
    h.run_steps(2);
    h
}

fn screen(h: &Harness<'static, PhotocraftApp>, x: f32, y: f32) -> Pos2 {
    let app = h.state();
    let v = &app.ui.views[0];
    let xf = ViewXform {
        rect: crate::rulers::content_rect(app, app.last_canvas_rect),
        zoom: v.zoom,
        center: v.center,
        flip: app.ui.view.flip_horizontal,
        rotation: v.rotation,
        aspect: app.ui.view.display_aspect(),
    };
    xf.to_screen(x, y)
}

fn button(h: &mut Harness<'static, PhotocraftApp>, p: Pos2, down: bool, m: Modifiers) {
    h.event(egui::Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: down, modifiers: m });
    h.run_steps(1);
}

/// A click at document point `(x, y)`, the pointer still between press and release.
fn click(h: &mut Harness<'static, PhotocraftApp>, x: f32, y: f32, m: Modifiers) {
    let p = screen(h, x, y);
    h.event(egui::Event::PointerMoved(p));
    h.run_steps(1);
    button(h, p, true, m);
    button(h, p, false, m);
}

fn double_click(h: &mut Harness<'static, PhotocraftApp>, x: f32, y: f32) {
    click(h, x, y, Modifiers::NONE);
    click(h, x, y, Modifiers::NONE);
    h.run_steps(1);
}

fn selection(h: &Harness<'static, PhotocraftApp>) -> Option<Rect> {
    h.state().session.active().unwrap().doc.selection.as_ref().map(|s| s.content_bounds())
}

fn near(r: Rect, x0: i32, y0: i32, x1: i32, y1: i32) -> bool {
    r.x0.abs_diff(x0) <= 1 && r.y0.abs_diff(y0) <= 1 && r.x1.abs_diff(x1) <= 1 && r.y1.abs_diff(y1) <= 1
}

#[test]
fn switching_tools_discards_unfinished_polygon_without_changing_selection() {
    let mut app = polygon_app();
    app.run("select.rect", json!({"x": 200, "y": 150, "width": 50, "height": 50})).unwrap();
    let before = app.session.active().unwrap().doc.selection.as_ref().unwrap().content_bounds();
    let history = app.session.active().unwrap().history.past_len();

    ev(&mut app, "down", 20.0, 20.0, Modifiers::NONE);
    ev(&mut app, "up", 20.0, 20.0, Modifiers::NONE);
    assert_eq!(app.ui.polygon.len(), 1);
    assert_eq!(app.ui.polygon_mode, "replace");
    assert!(crate::canvas::polygon_replaces_selection(&app));

    app.ui.tool = Tool::QuickSelection;
    app.sync_views();
    assert!(app.ui.polygon.is_empty());
    assert!(app.ui.polygon_mode.is_empty());
    assert!(!crate::canvas::polygon_replaces_selection(&app));
    assert_eq!(app.session.active().unwrap().doc.selection.as_ref().unwrap().content_bounds(), before);
    assert_eq!(app.session.active().unwrap().history.past_len(), history);

    app.ui.tool = Tool::PolygonLasso;
    app.sync_views();
    assert!(app.ui.polygon.is_empty(), "returning to the tool must start a fresh outline");
}

#[test]
fn double_click_closes_the_polygon_at_the_last_point() {
    let mut h = harness();
    click(&mut h, 50.0, 50.0, Modifiers::NONE);
    h.run_steps(30); // Well past egui's double-click delay: separate clicks.
    click(&mut h, 150.0, 50.0, Modifiers::NONE);
    h.run_steps(30);
    double_click(&mut h, 150.0, 150.0);
    assert!(h.state().ui.polygon.is_empty(), "the polygon closed: {:?}", h.state().ui.polygon);
    let r = selection(&h).expect("a selection");
    assert!(near(r, 50, 50, 150, 150), "the triangle's bounds: {r:?}");
    // Only one vertex was added at the double-click, not a duplicate starting a new polygon.
    h.run_steps(30);
    assert!(h.state().ui.polygon.is_empty());
}

/// Vertices placed quickly: egui counts the double-click's second click as a triple click (the
/// vertex before it was under 0.6 s earlier). It still closes the polygon.
#[test]
fn double_click_closes_the_polygon_after_quickly_placed_vertices() {
    let mut h = harness();
    click(&mut h, 50.0, 50.0, Modifiers::NONE);
    h.run_steps(12);
    click(&mut h, 150.0, 50.0, Modifiers::NONE);
    h.run_steps(12);
    double_click(&mut h, 150.0, 150.0);
    assert!(h.state().ui.polygon.is_empty(), "{:?}", h.state().ui.polygon);
    assert!(selection(&h).is_some_and(|r| near(r, 50, 50, 150, 150)), "{:?}", selection(&h));
}

/// Double-clicking on the first vertex: the first click closes the polygon, and the second must
/// not start a new one (which would deselect what was just made).
#[test]
fn double_click_on_the_first_vertex_keeps_the_new_selection() {
    let mut h = harness();
    for (x, y) in [(50.0, 50.0), (150.0, 50.0), (150.0, 150.0)] {
        click(&mut h, x, y, Modifiers::NONE);
        h.run_steps(30);
    }
    double_click(&mut h, 50.0, 50.0);
    assert!(h.state().ui.polygon.is_empty());
    assert!(selection(&h).is_some_and(|r| near(r, 50, 50, 150, 150)), "{:?}", selection(&h));
}

/// Three clicks and Enter: a triangle with corners at the first three points.
fn triangle(h: &mut Harness<'static, PhotocraftApp>, x: f32, y: f32, m: Modifiers) {
    h.event(egui::Event::ModifiersChanged(m));
    h.run_steps(1);
    for (dx, dy) in [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)] {
        click(h, x + dx, y + dy, m);
        h.run_steps(30); // Not a double-click with the click before.
    }
    h.event(egui::Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(1);
    h.key_press(egui::Key::Enter);
    h.run_steps(2);
}

fn steps(h: &Harness<'static, PhotocraftApp>) -> usize {
    h.state().session.active().unwrap().history.entries().len()
}

/// A new polygon hides the selection it will replace; Esc brings it back untouched.
#[test]
fn a_new_polygon_hides_the_selection_and_esc_restores_it() {
    let mut h = harness();
    h.state_mut().run("select.rect", json!({"x": 200, "y": 150, "width": 100, "height": 100})).unwrap();
    h.run_steps(1);
    let before = steps(&h);
    click(&mut h, 20.0, 20.0, Modifiers::NONE);
    assert_eq!(h.state().ui.polygon.len(), 1);
    assert!(crate::canvas::polygon_replaces_selection(h.state()), "the old outline is hidden");
    assert_eq!(steps(&h), before, "starting a polygon records nothing");
    h.key_press(egui::Key::Escape);
    h.run_steps(2);
    assert!(h.state().ui.polygon.is_empty());
    assert!(!crate::canvas::polygon_replaces_selection(h.state()));
    assert!(selection(&h).is_some_and(|r| near(r, 200, 150, 300, 250)), "{:?}", selection(&h));
    assert_eq!(steps(&h), before);
}

/// Replacing a selection is one history step, and one Undo brings the old selection back.
#[test]
fn replacing_a_selection_is_one_history_step() {
    let mut h = harness();
    h.state_mut().run("select.rect", json!({"x": 200, "y": 150, "width": 100, "height": 100})).unwrap();
    h.run_steps(1);
    let before = steps(&h);
    triangle(&mut h, 20.0, 20.0, Modifiers::NONE);
    assert!(selection(&h).is_some_and(|r| near(r, 20, 20, 120, 120)), "{:?}", selection(&h));
    assert_eq!(steps(&h), before + 1);
    h.state_mut().run("edit.undo", json!({})).unwrap();
    assert!(selection(&h).is_some_and(|r| near(r, 200, 150, 300, 250)), "{:?}", selection(&h));
}

/// ⇧ (add) and ⌥ (subtract) at the first click keep the selection in view, and the polygon
/// combines in that mode even when it is closed with a bare Enter.
#[test]
fn modifiers_at_the_first_click_keep_and_combine_the_selection() {
    let mut h = harness();
    h.state_mut().run("select.rect", json!({"x": 200, "y": 150, "width": 100, "height": 100})).unwrap();
    h.run_steps(1);
    for m in [Modifiers::SHIFT, Modifiers::ALT] {
        h.event(egui::Event::ModifiersChanged(m));
        h.run_steps(1);
        click(&mut h, 20.0, 20.0, m);
        h.event(egui::Event::ModifiersChanged(Modifiers::NONE));
        h.run_steps(1);
        assert_eq!(h.state().ui.polygon.len(), 1, "{m:?}");
        assert!(!crate::canvas::polygon_replaces_selection(h.state()), "{m:?}: the selection stays in view");
        h.key_press(egui::Key::Escape);
        h.run_steps(30);
    }
    triangle(&mut h, 20.0, 20.0, Modifiers::SHIFT);
    assert!(selection(&h).is_some_and(|r| near(r, 20, 20, 300, 250)), "added: {:?}", selection(&h));
}

/// A new polygon over no selection records one step: the selection it makes.
#[test]
fn a_new_polygon_without_a_selection_adds_one_step() {
    let mut h = harness();
    let before = steps(&h);
    click(&mut h, 20.0, 20.0, Modifiers::NONE);
    assert_eq!(h.state().ui.polygon.len(), 1);
    assert_eq!(steps(&h), before);
    h.run_steps(30);
    h.key_press(egui::Key::Escape);
    h.run_steps(30);
    triangle(&mut h, 20.0, 20.0, Modifiers::NONE);
    assert_eq!(steps(&h), before + 1);
}

/// The options bar's Feather softens the Lasso and Polygonal Lasso; the Patch Tool, which has no
/// Feather, keeps a hard edge.
#[test]
fn lasso_tools_apply_the_options_bar_feather() {
    use crate::canvas::{ToolEvent, tool_event};
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 200, "height": 150})).unwrap();
    app.sync_views();
    app.ui.tool_options.feather = 4.0;
    let corners = [(40.0, 30.0), (160.0, 30.0), (160.0, 120.0), (40.0, 120.0)];
    let coverage = |app: &PhotocraftApp, x: i32| app.session.active().unwrap().doc.selection.as_ref().map_or(0.0, |m| m.sample_channel(x, 75, 0));
    let drag = |app: &mut PhotocraftApp, tool: Tool| {
        app.run("select.deselect", json!({})).ok();
        app.ui.tool = tool;
        tool_event(app, ToolEvent::Down { x: 40.0, y: 30.0, pressure: 1.0 }, Modifiers::NONE);
        for (x, y) in corners.iter().skip(1) {
            tool_event(app, ToolEvent::Move { x: *x, y: *y, pressure: 1.0 }, Modifiers::NONE);
        }
        tool_event(app, ToolEvent::Up { x: 40.0, y: 120.0 }, Modifiers::NONE);
    };
    let soft = |app: &PhotocraftApp| coverage(app, 37) > 0.05 && coverage(app, 41) < 0.95;
    drag(&mut app, Tool::Lasso);
    assert!(soft(&app), "lasso: {} {}", coverage(&app, 37), coverage(&app, 41));
    app.ui.tool = Tool::PolygonLasso;
    for (x, y) in corners {
        tool_event(&mut app, ToolEvent::Down { x, y, pressure: 1.0 }, Modifiers::NONE);
        tool_event(&mut app, ToolEvent::Up { x, y }, Modifiers::NONE);
    }
    crate::canvas::commit_polygon(&mut app);
    assert!(soft(&app), "polygonal: {} {}", coverage(&app, 37), coverage(&app, 41));
    drag(&mut app, Tool::Patch);
    assert_eq!((coverage(&app, 37), coverage(&app, 41)), (0.0, 1.0), "patch: a hard edge");
}

fn polygon_app() -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 400, "height": 300, "background": "transparent"})).unwrap();
    app.sync_views();
    app.ui.extras.snap = false;
    app.ui.tool = Tool::PolygonLasso;
    app
}

fn ev(app: &mut PhotocraftApp, kind: &str, x: f64, y: f64, m: Modifiers) {
    use crate::canvas::{ToolEvent, tool_event};
    let e = match kind {
        "down" => ToolEvent::Down { x, y, pressure: 1.0 },
        "up" => ToolEvent::Up { x, y },
        _ => ToolEvent::Move { x, y, pressure: 1.0 },
    };
    tool_event(app, e, m);
}

/// ⌘-drag inside a selection with the Polygonal Lasso cuts and floats the pixels, as with the
/// marquee; a plain press there still places a point.
#[test]
fn cmd_drag_inside_the_selection_floats_it_with_the_polygonal_lasso() {
    let mut app = polygon_app();
    let layer = app.session.active().unwrap().active_layer.unwrap();
    app.session
        .edit("paint", |doc, _| {
            doc.layer_mut(layer).unwrap().surface_mut().unwrap().fill_rect(Rect::new(10, 10, 30, 30), &[1.0, 0.0, 0.0, 1.0]);
            Ok(())
        })
        .unwrap();
    app.run("select.rect", json!({"x": 10, "y": 10, "width": 20, "height": 20})).unwrap();
    ev(&mut app, "down", 20.0, 20.0, Modifiers::COMMAND);
    ev(&mut app, "move", 40.0, 20.0, Modifiers::COMMAND);
    ev(&mut app, "up", 40.0, 20.0, Modifiers::COMMAND);
    let f = photocraft_engine::float_cmds::floating(app.session.active().unwrap()).map(|f| f.offset);
    assert_eq!(f, Some((20, 0)));
    assert!(app.ui.polygon.is_empty(), "no polygon point was placed");
    app.run("select.drop", json!({})).unwrap();
    // A plain press inside the selection starts a polygon.
    ev(&mut app, "down", 35.0, 20.0, Modifiers::NONE);
    ev(&mut app, "up", 35.0, 20.0, Modifiers::NONE);
    assert_eq!(app.ui.polygon.len(), 1);
}

#[test]
fn backspace_delete_and_right_click_remove_the_last_vertex() {
    // #1229: ⌫ took Edit › Clear and wiped the layer while a polygon was being drawn.
    let mut h = harness();
    for (x, y) in [(50.0, 50.0), (150.0, 50.0), (150.0, 120.0)] {
        click(&mut h, x, y, Modifiers::NONE);
        // Past egui's double-click window, so each click adds a vertex.
        h.run_steps(20);
    }
    assert_eq!(h.state().ui.polygon.len(), 3);
    let st = h.state().session.active().unwrap();
    let (layer, steps) = (st.active_layer.unwrap(), st.history.past_len());
    for key in [egui::Key::Backspace, egui::Key::Delete] {
        h.event(egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE });
        h.run_steps(2);
    }
    assert_eq!(h.state().ui.polygon.len(), 1);
    // A right-click removes the last one and cancels the polygon, without a context menu.
    let p = screen(&h, 90.0, 90.0);
    h.event(egui::Event::PointerMoved(p));
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: p, button: PointerButton::Secondary, pressed: true, modifiers: Modifiers::NONE });
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: p, button: PointerButton::Secondary, pressed: false, modifiers: Modifiers::NONE });
    h.run_steps(2);
    assert!(h.state().ui.polygon.is_empty() && h.state().ui.polygon_mode.is_empty());
    assert!(h.state().ui.canvas_tool_menu.is_none());
    let st = h.state().session.active().unwrap();
    assert!(st.doc.layer(layer).is_some(), "the layer survives");
    assert_eq!(st.history.past_len(), steps, "nothing was edited");
}

/// ⌥ held: a drag draws freehand into the polygon; releasing ⌥ goes back to straight segments,
/// and the polygon stays open (a new selection, even though ⌥ was held at the first click).
#[test]
fn alt_drags_draw_freehand_and_releasing_alt_keeps_the_polygon_open() {
    let mut app = polygon_app();
    let alt = Modifiers::ALT;
    ev(&mut app, "down", 50.0, 50.0, alt);
    for x in [70.0, 90.0, 110.0, 130.0, 150.0] {
        ev(&mut app, "move", x, 50.0 + (x - 50.0) / 4.0, alt);
    }
    ev(&mut app, "up", 150.0, 75.0, alt);
    assert!(app.ui.polygon.len() >= 6, "freehand points: {}", app.ui.polygon.len());
    assert_eq!(app.ui.polygon_mode, "replace", "⌥ with nothing selected doesn't subtract");
    // ⌥ released: clicks add straight segments, nothing closes.
    ev(&mut app, "down", 150.0, 200.0, Modifiers::NONE);
    ev(&mut app, "up", 150.0, 200.0, Modifiers::NONE);
    assert!(!app.ui.polygon.is_empty() && app.session.active().unwrap().doc.selection.is_none());
    crate::canvas::commit_polygon(&mut app);
    assert!(app.session.active().unwrap().doc.selection.as_ref().unwrap().sample_channel(110, 100, 0) > 0.5);
}

#[test]
fn shift_constrains_polygonal_lasso_segments_but_unmodified_clicks_remain_free() {
    let mut app = polygon_app();
    ev(&mut app, "down", 100.0, 100.0, Modifiers::NONE);
    ev(&mut app, "up", 100.0, 100.0, Modifiers::NONE);
    assert_eq!(app.ui.polygon, vec![[100.0, 100.0]]);

    // Near a diagonal: Shift constrains to exactly 45° from the previous fixed point.
    ev(&mut app, "down", 140.0, 120.0, Modifiers::SHIFT);
    ev(&mut app, "up", 140.0, 120.0, Modifiers::SHIFT);
    let diagonal = app.ui.polygon[1];
    assert!((diagonal[0] - diagonal[1]).abs() < 1e-8, "45-degree direction: {diagonal:?}");
    assert!((diagonal[0] - 100.0 - 40.0_f64.hypot(20.0) / 2.0_f64.sqrt()).abs() < 1e-8);
    // Releasing Shift restores free placement rather than constraining subsequent points.
    ev(&mut app, "down", 155.0, 175.0, Modifiers::NONE);
    ev(&mut app, "up", 155.0, 175.0, Modifiers::NONE);
    assert_eq!(app.ui.polygon[2], [155.0, 175.0]);
}

#[test]
fn shift_polygonal_lasso_snap_has_all_eight_directions_and_handles_zero_length() {
    let origin = [200.0, 150.0];
    let directions = [[40.0, 1.0], [40.0, 25.0], [1.0, 40.0], [-40.0, 25.0], [-40.0, -1.0], [-40.0, -25.0], [-1.0, -40.0], [40.0, -25.0]];
    for delta in directions {
        let end = crate::canvas::constrained_polygon_point(origin, [origin[0] + delta[0], origin[1] + delta[1]]);
        let dx = end[0] - origin[0];
        let dy = end[1] - origin[1];
        let angle = dy.atan2(dx) / std::f64::consts::FRAC_PI_4;
        assert!((angle - angle.round()).abs() < 1e-9, "{delta:?} → {end:?}");
        assert!((dx.hypot(dy) - delta[0].hypot(delta[1])).abs() < 1e-8);
    }
    assert_eq!(crate::canvas::constrained_polygon_point(origin, origin), origin);
}

/// Two vertices placed with the Polygonal Lasso, well apart in time so they aren't a double-click.
fn two_vertices(h: &mut Harness<'static, PhotocraftApp>) {
    click(h, 50.0, 50.0, Modifiers::NONE);
    h.run_steps(30);
    click(h, 150.0, 50.0, Modifiers::NONE);
    h.run_steps(30);
    assert_eq!(h.state().ui.polygon.len(), 2);
}

/// The polygon, its mode and its document are all gone, and the selection it would have replaced
/// is shown again.
fn assert_polygon_dropped(h: &Harness<'static, PhotocraftApp>, what: &str) {
    let app = h.state();
    assert!(app.ui.polygon.is_empty(), "{what}: the outline is gone: {:?}", app.ui.polygon);
    assert!(app.ui.polygon_mode.is_empty(), "{what}: the mode is gone");
    assert!(app.ui.polygon_doc.is_none(), "{what}: the document is forgotten");
    assert!(!crate::canvas::polygon_replaces_selection(app), "{what}: the old outline is shown again");
}

/// Every other tool drops the polygon in progress, not just the Quick Selection tool, and coming
/// back to the Polygonal Lasso starts a new one.
#[test]
fn every_other_tool_drops_the_polygon_in_progress() {
    let tools = [
        Tool::QuickSelection,
        Tool::ObjectSelection,
        Tool::MagicWand,
        Tool::RectMarquee,
        Tool::EllipseMarquee,
        Tool::Lasso,
        Tool::MagneticLasso,
        Tool::Move,
        Tool::Brush,
    ];
    for tool in tools {
        let mut h = harness();
        h.state_mut().run("select.rect", json!({"x": 200, "y": 150, "width": 100, "height": 100})).unwrap();
        h.run_steps(1);
        let before = steps(&h);
        two_vertices(&mut h);
        assert!(crate::canvas::polygon_replaces_selection(h.state()), "{tool:?}: the old outline is hidden while drawing");
        h.state_mut().ui.tool = tool;
        h.state_mut().sync_views();
        assert_polygon_dropped(&h, &format!("{tool:?}"));
        assert!(selection(&h).is_some_and(|r| near(r, 200, 150, 300, 250)), "{tool:?}: {:?}", selection(&h));
        assert_eq!(steps(&h), before, "{tool:?}: nothing was recorded");
        h.state_mut().ui.tool = Tool::PolygonLasso;
        h.state_mut().sync_views();
        assert!(h.state().ui.polygon.is_empty(), "{tool:?}: coming back starts a new polygon");
    }
}

/// The polygon belongs to the document it was started on: another document becoming active drops
/// it, while the tool stays the Polygonal Lasso (its outline would otherwise show on the other
/// document).
#[test]
fn switching_documents_drops_the_polygon_in_progress() {
    let mut h = harness();
    two_vertices(&mut h);
    let first = h.state().session.active().map(|st| st.doc.id);
    assert!(first.is_some());
    assert_eq!(h.state().ui.polygon_doc, first, "the polygon remembers its document");
    h.state_mut().run("file.new", json!({"width": 200, "height": 100})).unwrap();
    assert_ne!(h.state().session.active().map(|st| st.doc.id), first, "another document is active");
    assert_polygon_dropped(&h, "new document");
    assert_eq!(h.state().ui.tool, Tool::PolygonLasso, "the tool stays");
    // Back on the first document, nothing is left of it.
    h.state_mut().run("document.activate", json!({"document": 0})).unwrap();
    h.state_mut().sync_views();
    assert!(h.state().ui.polygon.is_empty());
}

/// A held temporary tool (Space is the Hand, ⌘ the Move tool) doesn't change the selected tool, so
/// the polygon being drawn stays.
#[test]
fn a_held_temporary_tool_keeps_the_polygon() {
    let mut h = harness();
    two_vertices(&mut h);
    h.event(egui::Event::Key { key: egui::Key::Space, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE });
    h.run_steps(3);
    h.state_mut().sync_views();
    assert_eq!(h.state().ui.tool, Tool::PolygonLasso);
    assert_eq!(h.state().ui.polygon.len(), 2, "Space held");
    h.event(egui::Event::Key { key: egui::Key::Space, physical_key: None, pressed: false, repeat: false, modifiers: Modifiers::NONE });
    h.run_steps(3);
    h.event(egui::Event::ModifiersChanged(Modifiers::COMMAND));
    h.run_steps(3);
    h.state_mut().sync_views();
    assert_eq!(h.state().ui.tool, Tool::PolygonLasso);
    assert_eq!(h.state().ui.polygon.len(), 2, "⌘ held");
    h.event(egui::Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(3);
    assert_eq!(h.state().ui.polygon.len(), 2, "released");
}
