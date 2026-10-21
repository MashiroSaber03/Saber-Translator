use super::*;
use crate::canvas::tool_event;
use serde_json::json;

fn app() -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
    app.run("file.new", json!({"width": 400, "height": 300})).unwrap();
    app.sync_views();
    app.ui.tool = Tool::Lasso;
    app
}

fn event(app: &mut PhotocraftApp, kind: &str, x: f64, y: f64, mods: Modifiers) {
    let ev = match kind {
        "down" => ToolEvent::Down { x, y, pressure: 1.0 },
        "up" => ToolEvent::Up { x, y },
        _ => ToolEvent::Move { x, y, pressure: 1.0 },
    };
    tool_event(app, ev, mods);
}

fn begin(app: &mut PhotocraftApp, mods: Modifiers) {
    event(app, "down", 50.0, 50.0, mods);
    event(app, "move", 90.0, 40.0, mods);
    event(app, "move", 120.0, 50.0, mods);
}

#[test]
fn alt_clicks_add_straight_segments_and_releasing_alt_closes_the_outline() {
    let mut app = app();
    begin(&mut app, Modifiers::NONE);
    event(&mut app, "up", 120.0, 50.0, Modifiers::ALT);
    assert!(app.session.active().unwrap().doc.selection.is_none(), "Alt held at the release keeps it open");
    // Hover with Alt only moves the rubber band.
    event(&mut app, "move", 170.0, 10.0, Modifiers::ALT);
    event(&mut app, "move", 180.0, 100.0, Modifiers::ALT);
    assert_eq!(app.drag.as_ref().unwrap().points.len(), 3);
    // An Alt click adds a straight segment to it.
    event(&mut app, "down", 180.0, 100.0, Modifiers::ALT);
    event(&mut app, "up", 180.0, 100.0, Modifiers::ALT);
    event(&mut app, "move", 60.0, 140.0, Modifiers::ALT);
    event(&mut app, "down", 60.0, 140.0, Modifiers::ALT);
    event(&mut app, "up", 60.0, 140.0, Modifiers::ALT);
    assert_eq!(app.drag.as_ref().unwrap().points.len(), 5);
    assert_eq!(crate::tool_feedback::badge(&app, Tool::Lasso, Modifiers::ALT), None);
    // Releasing Alt with the button up closes the outline and makes the selection.
    event(&mut app, "move", 70.0, 140.0, Modifiers::NONE);
    assert!(!active(&app));
    let sel = app.session.active().unwrap().doc.selection.as_ref().unwrap();
    assert!(sel.sample_channel(100, 80, 0) > 0.9);
    assert!(sel.sample_channel(160, 20, 0) < 0.1, "hover detour must stay outside");
    app.run("edit.undo", json!({})).unwrap();
    assert!(app.session.active().unwrap().doc.selection.is_none(), "one selection undo");
}

#[test]
fn a_drag_with_alt_held_draws_freehand() {
    let mut app = app();
    begin(&mut app, Modifiers::NONE);
    // Alt pressed during the drag: still freehand while the button is down.
    event(&mut app, "move", 180.0, 100.0, Modifiers::ALT);
    event(&mut app, "move", 170.0, 130.0, Modifiers::ALT);
    let pts = &app.drag.as_ref().unwrap().points;
    assert_eq!(&pts[3..], &[[180.0, 100.0, 1.0], [170.0, 130.0, 1.0]]);
    // Released with Alt held: the outline stays open for straight segments...
    event(&mut app, "up", 170.0, 130.0, Modifiers::ALT);
    assert!(active(&app));
    // ...and an Alt drag adds a freehand stretch to it.
    event(&mut app, "down", 120.0, 150.0, Modifiers::ALT);
    event(&mut app, "move", 90.0, 155.0, Modifiers::ALT);
    event(&mut app, "move", 60.0, 140.0, Modifiers::ALT);
    event(&mut app, "up", 60.0, 140.0, Modifiers::ALT);
    assert_eq!(app.drag.as_ref().unwrap().points.len(), 8);
    event(&mut app, "move", 60.0, 140.0, Modifiers::NONE);
    assert!(app.session.active().unwrap().doc.selection.is_some());
}

#[test]
fn alt_at_the_first_press_without_a_selection_starts_straight_segments() {
    let mut app = app();
    assert_eq!(crate::tool_feedback::badge(&app, Tool::Lasso, Modifiers::ALT), None, "no − badge: nothing to subtract");
    // Alt-click three corners of a triangle, then let go of Alt.
    for [x, y] in [[50.0, 50.0], [250.0, 50.0], [150.0, 200.0]] {
        event(&mut app, "down", x, y, Modifiers::ALT);
        event(&mut app, "up", x, y, Modifiers::ALT);
        assert!(active(&app), "Alt held: the outline stays open");
    }
    event(&mut app, "move", 150.0, 200.0, Modifiers::NONE);
    assert!(!active(&app));
    let sel = app.session.active().unwrap().doc.selection.as_ref().unwrap();
    assert!(sel.sample_channel(150, 100, 0) > 0.9, "a new selection, nothing subtracted");
    assert!(sel.sample_channel(60, 180, 0) < 0.1);
}

#[test]
fn initial_alt_still_subtracts_and_initial_shift_still_adds() {
    for initial in [Modifiers::ALT, Modifiers::SHIFT, Modifiers::SHIFT | Modifiers::ALT] {
        let mut app = app();
        app.run("select.rect", json!({"x": 0, "y": 0, "width": 150, "height": 200})).unwrap();
        begin(&mut app, initial);
        if initial.alt {
            event(&mut app, "move", 120.0, 50.0, Modifiers::NONE);
        }
        event(&mut app, "up", 120.0, 50.0, Modifiers::ALT);
        event(&mut app, "down", 180.0, 150.0, Modifiers::ALT);
        event(&mut app, "up", 180.0, 150.0, Modifiers::ALT);
        commit(&mut app);
        let sel = app.session.active().unwrap().doc.selection.as_ref().unwrap();
        assert_eq!(sel.sample_channel(120, 90, 0) > 0.5, initial.shift);
        assert_eq!(sel.sample_channel(10, 10, 0) > 0.5, !(initial.shift && initial.alt));
    }
}

#[test]
fn tool_or_document_switch_discards_pending_outline() {
    let mut app = app();
    begin(&mut app, Modifiers::NONE);
    event(&mut app, "up", 120.0, 50.0, Modifiers::ALT);
    app.ui.tool = Tool::Brush;
    app.sync_views();
    assert!(!active(&app));
    app.ui.tool = Tool::Lasso;
    begin(&mut app, Modifiers::NONE);
    event(&mut app, "up", 120.0, 50.0, Modifiers::ALT);
    app.run("file.new", json!({"width": 200, "height": 200})).unwrap();
    commit(&mut app);
    assert!(!active(&app));
    assert!(app.session.active().unwrap().doc.selection.is_none());
}

fn harness() -> egui_kittest::Harness<'static, PhotocraftApp> {
    let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(1000.0, 700.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            let ctx = ui.ctx().clone();
            if !ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            crate::shortcuts::handle(app, &ctx);
            egui::CentralPanel::default().show(ui, |ui| crate::canvas::document_area(app, ui));
        },
        app(),
    );
    PhotocraftApp::setup_context(&h.ctx, Default::default());
    h.run_steps(4);
    let v = &mut h.state_mut().ui.views[0];
    v.zoom = 1.0;
    v.center = [200.0, 150.0];
    v.fit_pending = false;
    h.run_steps(2);
    h
}

fn mouse(h: &mut egui_kittest::Harness<'static, PhotocraftApp>, kind: &str, x: f32, y: f32, mods: Modifiers) {
    let app = h.state();
    let v = &app.ui.views[0];
    let xf = ViewXform {
        rect: crate::rulers::content_rect(app, app.last_canvas_rect),
        zoom: v.zoom,
        center: v.center,
        flip: false,
        rotation: v.rotation,
        aspect: 1.0,
    };
    let pos = xf.to_screen(x, y);
    h.event(Event::ModifiersChanged(mods));
    h.event(Event::PointerMoved(pos));
    if kind != "move" {
        h.event(Event::PointerButton { pos, button: PointerButton::Primary, pressed: kind == "down", modifiers: mods });
    }
    h.run_steps(2);
}

#[test]
fn real_canvas_alt_clicks_enter_escape_and_resume() {
    for finish in ["enter", "escape", "resume", "close"] {
        let mut h = harness();
        mouse(&mut h, "down", 50.0, 50.0, Modifiers::NONE);
        mouse(&mut h, "move", 120.0, 50.0, Modifiers::NONE);
        mouse(&mut h, "up", 120.0, 50.0, Modifiers::ALT);
        assert!(active(h.state()), "Alt release must keep gesture");
        mouse(&mut h, "move", 180.0, 100.0, Modifiers::ALT);
        mouse(&mut h, "down", 180.0, 100.0, Modifiers::ALT);
        mouse(&mut h, "up", 180.0, 100.0, Modifiers::ALT);
        assert_eq!(h.state().drag.as_ref().unwrap().points.len(), 3);
        match finish {
            "resume" => {
                mouse(&mut h, "down", 150.0, 150.0, Modifiers::NONE);
                mouse(&mut h, "move", 90.0, 130.0, Modifiers::NONE);
                mouse(&mut h, "up", 50.0, 120.0, Modifiers::NONE);
            }
            "close" => {
                mouse(&mut h, "down", 50.0, 50.0, Modifiers::ALT);
                mouse(&mut h, "up", 50.0, 50.0, Modifiers::ALT);
            }
            _ => {
                h.event(Event::Key {
                    key: if finish == "escape" { egui::Key::Escape } else { egui::Key::Enter },
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: Modifiers::ALT,
                });
                h.run_steps(2);
            }
        }
        assert!(!active(h.state()), "{finish}");
        assert_eq!(h.state().session.active().unwrap().doc.selection.is_some(), finish != "escape", "{finish}");
    }
}

/// A lasso press inside the selection still drags the selection (`canvas::selection_drag_kind`),
/// both through tool events and through the real canvas.
#[test]
fn a_drag_inside_the_selection_moves_it() {
    let mut app = app();
    app.ui.extras.snap = false;
    app.run("select.rect", json!({"x": 100, "y": 100, "width": 50, "height": 40})).unwrap();
    let bounds = |app: &PhotocraftApp| app.session.active().unwrap().doc.selection.as_ref().map(|s| s.content_bounds());
    event(&mut app, "down", 120.0, 120.0, Modifiers::NONE);
    event(&mut app, "move", 130.0, 125.0, Modifiers::NONE);
    event(&mut app, "up", 130.0, 125.0, Modifiers::NONE);
    assert!(!active(&app));
    assert_eq!(bounds(&app), Some(photocraft_geom::Rect::new(110, 105, 160, 145)));

    let mut h = harness();
    h.state_mut().ui.extras.snap = false;
    h.state_mut().run("select.rect", json!({"x": 100, "y": 100, "width": 50, "height": 40})).unwrap();
    h.run_steps(1);
    mouse(&mut h, "down", 120.0, 120.0, Modifiers::NONE);
    mouse(&mut h, "move", 125.0, 122.0, Modifiers::NONE);
    mouse(&mut h, "move", 130.0, 125.0, Modifiers::NONE);
    mouse(&mut h, "up", 130.0, 125.0, Modifiers::NONE);
    assert!(h.state().drag.is_none());
    assert_eq!(bounds(h.state()), Some(photocraft_geom::Rect::new(110, 105, 160, 145)));
}

#[test]
fn retracting_polygonal_vertices_keeps_the_document_unchanged() {
    let mut app = app();
    begin(&mut app, Modifiers::NONE);
    event(&mut app, "up", 120.0, 50.0, Modifiers::ALT);
    assert!(waiting_for_vertex(&app));
    let layer = app.session.active().unwrap().active_layer.unwrap();
    assert_eq!(app.drag.as_ref().unwrap().points.len(), 3);
    assert!(undo_last_vertex(&mut app));
    assert_eq!(app.drag.as_ref().unwrap().points.len(), 2);
    assert!(undo_last_vertex(&mut app));
    assert_eq!(app.drag.as_ref().unwrap().points.len(), 1);
    assert!(undo_last_vertex(&mut app));
    assert!(!active(&app));
    assert!(!undo_last_vertex(&mut app));
    let st = app.session.active().unwrap();
    assert!(st.doc.layer(layer).is_some());
    assert!(st.doc.selection.is_none());
}

#[test]
fn real_canvas_backspace_and_right_click_retract_polygonal_points() {
    let mut h = harness();
    mouse(&mut h, "down", 50.0, 50.0, Modifiers::NONE);
    mouse(&mut h, "move", 120.0, 50.0, Modifiers::NONE);
    mouse(&mut h, "up", 120.0, 50.0, Modifiers::ALT);
    mouse(&mut h, "down", 180.0, 100.0, Modifiers::ALT);
    mouse(&mut h, "up", 180.0, 100.0, Modifiers::ALT);
    assert_eq!(h.state().drag.as_ref().unwrap().points.len(), 3);
    let layer = h.state().session.active().unwrap().active_layer.unwrap();

    h.event(Event::Key { key: egui::Key::Backspace, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::ALT });
    h.run_steps(2);
    assert_eq!(h.state().drag.as_ref().unwrap().points.len(), 2);
    assert!(h.state().session.active().unwrap().doc.layer(layer).is_some());

    let state = h.state();
    let view = &state.ui.views[0];
    let xf = ViewXform {
        rect: crate::rulers::content_rect(state, state.last_canvas_rect),
        zoom: view.zoom,
        center: view.center,
        flip: false,
        rotation: 0.0,
        aspect: 1.0,
    };
    let pos = xf.to_screen(155.0, 100.0);
    h.event(Event::PointerMoved(pos));
    h.event(Event::PointerButton { pos, button: PointerButton::Secondary, pressed: true, modifiers: Modifiers::ALT });
    h.run_steps(1);
    h.event(Event::PointerButton { pos, button: PointerButton::Secondary, pressed: false, modifiers: Modifiers::ALT });
    h.run_steps(2);
    assert_eq!(h.state().drag.as_ref().unwrap().points.len(), 1);
    assert!(h.state().ui.canvas_tool_menu.is_none());
    assert!(h.state().session.active().unwrap().doc.layer(layer).is_some());
    assert!(h.state().session.active().unwrap().doc.selection.is_none());
}

#[test]
fn clicking_the_last_point_again_closes_the_outline() {
    let mut app = app();
    for [x, y] in [[50.0, 50.0], [250.0, 50.0], [150.0, 200.0]] {
        event(&mut app, "down", x, y, Modifiers::ALT);
        event(&mut app, "up", x, y, Modifiers::ALT);
    }
    assert!(active(&app));
    // Alt still held: a second click on the last point (however long after) closes it.
    event(&mut app, "down", 151.0, 199.0, Modifiers::ALT);
    assert!(!active(&app));
    assert!(app.session.active().unwrap().doc.selection.as_ref().unwrap().sample_channel(150, 100, 0) > 0.9);
}

/// ⌘ (Ctrl) with the Lasso is the Move tool for the drag: through the real canvas, a ⌘-drag outside
/// the selection (here: without one) moves the layer and draws no outline (`canvas::command_moves_layer`).
#[test]
fn real_canvas_cmd_drag_outside_the_selection_moves_the_layer() {
    let mut h = harness();
    h.state_mut().ui.extras.snap = false;
    h.state_mut().run("layer.new.layer", json!({})).unwrap();
    let layer = h.state().session.active().unwrap().active_layer.unwrap();
    h.state_mut()
        .session
        .edit("paint", |doc, _| {
            doc.layer_mut(layer).unwrap().surface_mut().unwrap().fill_rect(photocraft_geom::Rect::new(100, 100, 140, 140), &[1.0, 0.0, 0.0, 1.0]);
            Ok(())
        })
        .unwrap();
    h.run_steps(1);
    mouse(&mut h, "down", 120.0, 120.0, Modifiers::COMMAND);
    mouse(&mut h, "move", 130.0, 120.0, Modifiers::COMMAND);
    mouse(&mut h, "move", 150.0, 120.0, Modifiers::COMMAND);
    mouse(&mut h, "up", 150.0, 120.0, Modifiers::COMMAND);
    let app = h.state();
    let alpha = |x, y| app.session.active().unwrap().doc.layer(layer).unwrap().surface().unwrap().rgba(x, y)[3];
    assert!(app.drag.is_none() && !active(app), "no outline in progress");
    assert!(app.session.active().unwrap().doc.selection.is_none(), "no selection was made");
    assert_eq!((alpha(105, 120), alpha(165, 120)), (0.0, 1.0), "the layer moved 30 px to the right");
}
