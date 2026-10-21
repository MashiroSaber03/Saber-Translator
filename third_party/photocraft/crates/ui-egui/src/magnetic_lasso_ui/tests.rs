//! The Magnetic Lasso through `tool_event` (mouse and `ui.pointer` alike), and through the real
//! canvas for hover, keys and double-click.

use egui::{Modifiers, PointerButton, Pos2, vec2};
use egui_kittest::Harness;
use photocraft_geom::Rect;
use serde_json::json;

use super::*;
use crate::canvas::{ToolEvent, ViewXform, tool_event};

const SQUARE: Rect = Rect::new(50, 40, 130, 110);

/// A black 80×70 square on a white 200×150 document, the Magnetic Lasso, 100 % zoom.
fn app() -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 200, "height": 150})).unwrap();
    app.sync_views();
    app.session
        .edit("paint", |doc, _| {
            let bg = doc.layers[0].surface_mut().unwrap();
            bg.fill_rect(Rect::new(0, 0, 200, 150), &[1.0, 1.0, 1.0, 1.0]);
            bg.fill_rect(SQUARE, &[0.0, 0.0, 0.0, 1.0]);
            Ok(())
        })
        .unwrap();
    app.ui.tool = Tool::MagneticLasso;
    app
}

fn click(app: &mut PhotocraftApp, x: f64, y: f64, m: Modifiers) {
    tool_event(app, ToolEvent::Down { x, y, pressure: 1.0 }, m);
    tool_event(app, ToolEvent::Up { x, y }, m);
}

/// The pointer moves (button up) along the polyline `pts` in 2 px steps, like a hand on a mouse.
fn hover_along(app: &mut PhotocraftApp, pts: &[[f64; 2]]) {
    for w in pts.windows(2) {
        let n = (dist(w[0], w[1]) / 2.0).ceil().max(1.0) as usize;
        for k in 1..=n {
            let t = k as f64 / n as f64;
            let (x, y) = (w[0][0] + (w[1][0] - w[0][0]) * t, w[0][1] + (w[1][1] - w[0][1]) * t);
            tool_event(app, ToolEvent::Move { x, y, pressure: 1.0 }, Modifiers::NONE);
        }
    }
}

/// Around the square, 3 px outside its edges, back to just past the start.
const AROUND: [[f64; 2]; 6] = [[52.0, 37.0], [133.0, 37.0], [133.0, 113.0], [47.0, 113.0], [47.0, 37.0], [52.0, 37.0]];

fn coverage(app: &PhotocraftApp, x: i32, y: i32) -> f32 {
    app.session.active().unwrap().doc.selection.as_ref().map_or(0.0, |m| m.sample_channel(x, y, 0))
}

fn selected_area(app: &PhotocraftApp) -> f32 {
    (0..150).flat_map(|y| (0..200).map(move |x| (x, y))).map(|(x, y)| coverage(app, x, y)).sum()
}

fn last_step(app: &PhotocraftApp) -> Option<String> {
    app.session.active().unwrap().history.entries().last().cloned()
}

/// Whether `p` lies on the square's outline (within `tol`).
fn on_square(p: [f64; 2], tol: f64) -> bool {
    let (x0, y0, x1, y1) = (f64::from(SQUARE.x0), f64::from(SQUARE.y0), f64::from(SQUARE.x1), f64::from(SQUARE.y1));
    let inside = (x0 - tol..=x1 + tol).contains(&p[0]) && (y0 - tol..=y1 + tol).contains(&p[1]);
    let near = (p[0] - x0).abs() <= tol || (p[0] - x1).abs() <= tol || (p[1] - y0).abs() <= tol || (p[1] - y1).abs() <= tol;
    inside && near
}

#[test]
fn tracing_around_a_shape_selects_it() {
    let mut app = app();
    click(&mut app, AROUND[0][0], AROUND[0][1], Modifiers::NONE);
    let m = &app.ui.magnetic;
    assert!(m.active() && m.anchors == vec![0] && m.mode == "replace", "{m:?}");
    assert!(on_square(m.path[0], 0.01), "the first point snaps onto the edge: {:?}", m.path[0]);
    hover_along(&mut app, &AROUND[..5]);
    let m = &app.ui.magnetic;
    // Points fastened by themselves on the way (about every 47 px at Frequency 57).
    assert!(m.anchors.len() >= 5, "{:?}", m.anchors);
    for p in m.path.iter().chain(&m.live) {
        assert!(on_square(*p, 1.0), "{p:?} is off the square's edge");
    }
    // No selection until the border closes, and nothing in the history.
    assert_eq!(last_step(&app).as_deref(), Some("paint"));
    // Back at the first point, a click closes the border along the edges.
    hover_along(&mut app, &AROUND[4..]);
    click(&mut app, AROUND[5][0], AROUND[5][1], Modifiers::NONE);
    assert!(!app.ui.magnetic.active());
    let area = selected_area(&app);
    assert!((area - 5600.0).abs() < 120.0, "the square's 80 × 70: {area}");
    assert_eq!((coverage(&app, 90, 75), coverage(&app, 47, 75), coverage(&app, 90, 113)), (1.0, 0.0, 0.0));
    assert_eq!(last_step(&app).as_deref(), Some("Magnetic Lasso"));
}

#[test]
fn enter_closes_along_the_edges_and_alt_closes_straight() {
    let mut app = app();
    // Over the top and right edges only; the rest closes along the edges from the last point.
    click(&mut app, 52.0, 37.0, Modifiers::NONE);
    hover_along(&mut app, &[[52.0, 37.0], [133.0, 37.0], [133.0, 108.0]]);
    close(&mut app, false);
    let magnetic = selected_area(&app);
    assert!(magnetic > 2000.0, "{magnetic}");
    // Straight: the closing segment is the diagonal from the last point to the first.
    click(&mut app, 52.0, 37.0, Modifiers::NONE);
    hover_along(&mut app, &[[52.0, 37.0], [133.0, 37.0], [133.0, 108.0]]);
    close(&mut app, true);
    let straight = selected_area(&app);
    assert!((straight - 2700.0).abs() < 300.0, "half the square: {straight}");
}

#[test]
fn backspace_removes_fastening_points_back_to_the_first() {
    let mut app = app();
    click(&mut app, 52.0, 37.0, Modifiers::NONE);
    hover_along(&mut app, &[[52.0, 37.0], [133.0, 37.0], [133.0, 70.0]]);
    let n = app.ui.magnetic.anchors.len();
    assert!(n >= 3, "{n}");
    remove_last_point(&mut app);
    let m = &app.ui.magnetic;
    assert_eq!(m.anchors.len(), n - 1);
    assert_eq!(m.path.len(), m.anchors[n - 2] + 1, "the border ends at the point before");
    assert!(!m.live.is_empty(), "the live segment runs on to the pointer");
    // Without moving, ⌫ keeps removing (nothing re-fastens) down to the first point, then cancels.
    for _ in 0..n - 2 {
        remove_last_point(&mut app);
    }
    assert_eq!(app.ui.magnetic.anchors, vec![0]);
    remove_last_point(&mut app);
    assert!(!app.ui.magnetic.active());
    assert!(app.session.active().unwrap().doc.selection.is_none());
}

#[test]
fn clicks_fasten_and_alt_draws_straight_and_freehand_segments() {
    let mut app = app();
    click(&mut app, 20.0, 20.0, Modifiers::NONE);
    // A click on a flat area fastens right where it is.
    hover_along(&mut app, &[[20.0, 20.0], [30.0, 20.0]]);
    click(&mut app, 30.0, 20.0, Modifiers::NONE);
    assert_eq!(app.ui.magnetic.path.last(), Some(&[30.0, 20.0]));
    let anchors = app.ui.magnetic.anchors.len();
    // ⌥-click: straight to the point, wherever the edges are.
    click(&mut app, 180.0, 140.0, Modifiers::ALT);
    let m = &app.ui.magnetic;
    assert_eq!((m.path.last(), m.anchors.len(), m.freehand), (Some(&[180.0, 140.0]), anchors + 1, false));
    let before = m.path.len();
    assert_eq!(m.path[before - 2], [30.0, 20.0], "one straight segment");
    // ⌥-drag: the pointer's path is the border.
    tool_event(&mut app, ToolEvent::Down { x: 180.0, y: 140.0, pressure: 1.0 }, Modifiers::ALT);
    for (x, y) in [(170.0, 142.0), (160.0, 138.0), (150.0, 141.0)] {
        tool_event(&mut app, ToolEvent::Move { x, y, pressure: 1.0 }, Modifiers::ALT);
    }
    tool_event(&mut app, ToolEvent::Up { x: 150.0, y: 141.0 }, Modifiers::ALT);
    let m = &app.ui.magnetic;
    assert_eq!(&m.path[before..], &[[170.0, 142.0], [160.0, 138.0], [150.0, 141.0]]);
    assert_eq!(m.anchors.last(), Some(&(m.path.len() - 1)));
    assert!(!m.freehand);
    // ⌫ takes the freehand segment back in one step.
    remove_last_point(&mut app);
    assert_eq!(app.ui.magnetic.path.last(), Some(&[180.0, 140.0]));
}

#[test]
fn the_first_click_sets_the_mode_and_hides_a_replaced_selection() {
    let mut app = app();
    app.run("select.rect", json!({"x": 150, "y": 10, "width": 30, "height": 30})).unwrap();
    click(&mut app, 52.0, 37.0, Modifiers::NONE);
    assert!(crate::canvas::polygon_replaces_selection(&app));
    cancel(&mut app);
    assert!(!crate::canvas::polygon_replaces_selection(&app));
    // ⇧ at the first click adds; later the modifier keys draw segments, not modes.
    click(&mut app, 52.0, 37.0, Modifiers::SHIFT);
    assert_eq!(app.ui.magnetic.mode, "add");
    assert!(!crate::canvas::polygon_replaces_selection(&app));
    hover_along(&mut app, &AROUND);
    click(&mut app, AROUND[5][0], AROUND[5][1], Modifiers::NONE);
    assert_eq!((coverage(&app, 160, 20), coverage(&app, 90, 75)), (1.0, 1.0), "added to the selection");
}

#[test]
fn agents_see_the_border_in_progress() {
    let mut app = app();
    let ctx = egui::Context::default();
    click(&mut app, 52.0, 37.0, Modifiers::NONE);
    hover_along(&mut app, &[[52.0, 37.0], [133.0, 37.0]]);
    let v = crate::control::inspect(&app, &ctx);
    let m = &v["magnetic"];
    assert_eq!(m["mode"], json!("replace"));
    assert_eq!(m["anchors"].as_array().map(Vec::len), Some(app.ui.magnetic.anchors.len()));
    assert!(m["path"].as_array().is_some_and(|p| p.len() >= 2), "{m}");
    assert!(m["live"].as_array().is_some_and(|p| !p.is_empty()), "{m}");
    assert_eq!(v["tool"], json!("MagneticLasso"));
}

#[test]
fn switching_tools_or_documents_drops_the_border() {
    let mut app = app();
    click(&mut app, 52.0, 37.0, Modifiers::NONE);
    frame(&mut app);
    assert!(app.ui.magnetic.active());
    app.ui.tool = Tool::Lasso;
    frame(&mut app);
    assert!(!app.ui.magnetic.active());
    app.ui.tool = Tool::MagneticLasso;
    click(&mut app, 52.0, 37.0, Modifiers::NONE);
    app.run("file.new", json!({"width": 50, "height": 50})).unwrap();
    app.sync_views();
    frame(&mut app);
    assert!(!app.ui.magnetic.active());
}

#[test]
fn a_document_edit_while_drawing_is_traced_on() {
    let mut app = app();
    click(&mut app, 52.0, 37.0, Modifiers::NONE);
    hover_along(&mut app, &[[52.0, 37.0], [90.0, 37.0]]);
    // The square grows 10 px taller upwards mid-border: the border follows the new edge.
    app.session
        .edit("grow", |doc, _| {
            doc.layers[0].surface_mut().unwrap().fill_rect(Rect::new(50, 30, 130, 40), &[0.0, 0.0, 0.0, 1.0]);
            Ok(())
        })
        .unwrap();
    hover_along(&mut app, &[[90.0, 37.0], [100.0, 27.0], [125.0, 27.0]]);
    let live_end = app.ui.magnetic.live.last().copied().unwrap();
    assert!((live_end[1] - 30.0).abs() < 0.01, "on the new edge: {live_end:?}");
}

#[test]
fn width_follows_pen_pressure_only_when_asked_and_only_with_a_pen() {
    let mut app = app();
    app.ui.tool_options.magnetic_width = 21.0;
    assert_eq!(settings(&app, 1.0).width, 21.0);
    app.ui.tool_options.magnetic_pressure = true;
    // A mouse has no pressure: the width stays.
    assert_eq!(settings(&app, 1.0).width, 21.0);
    app.stylus.use_pressure = true;
    app.stylus.feed.set(Some(crate::stylus::PenSample { pressure: 0.5, ..Default::default() }));
    assert_eq!(settings(&app, 0.5).width, 11.0);
    assert_eq!(settings(&app, 1.0).width, 1.0);
    assert_eq!(settings(&app, f32::NAN).width, 21.0);
}

#[test]
fn hostile_input_never_panics() {
    let mut app = app();
    for (x, y) in [(f64::NAN, 1.0), (1.0, f64::INFINITY)] {
        click(&mut app, x, y, Modifiers::NONE);
        assert!(!app.ui.magnetic.active());
    }
    // Far off the canvas: kept on it.
    click(&mut app, -1e12, 1e300, Modifiers::NONE);
    tool_event(&mut app, ToolEvent::Move { x: 1e12, y: -1e12, pressure: 1.0 }, Modifiers::NONE);
    tool_event(&mut app, ToolEvent::Move { x: 10.0, y: 1e15, pressure: f32::NAN }, Modifiers::NONE);
    let m = &app.ui.magnetic;
    assert!(m.path.iter().chain(&m.live).all(|p| (0.0..=200.0).contains(&p[0]) && (0.0..=150.0).contains(&p[1])), "{m:?}");
    close(&mut app, false);
    // Out-of-range options are clamped.
    for (w, c, f) in [(0.0, 0.0, -5.0), (1e9, 1e9, 1e9), (f32::NAN, f32::NAN, f32::NAN)] {
        let o = &mut app.ui.tool_options;
        (o.magnetic_width, o.magnetic_contrast, o.magnetic_frequency) = (w, c, f);
        click(&mut app, 52.0, 37.0, Modifiers::NONE);
        hover_along(&mut app, &[[52.0, 37.0], [90.0, 37.0], [90.0, 60.0]]);
        close(&mut app, false);
    }
    // A border too short to select, closing, removing and cancelling with nothing drawn.
    click(&mut app, 10.0, 10.0, Modifiers::NONE);
    close(&mut app, false);
    close(&mut app, true);
    remove_last_point(&mut app);
    cancel(&mut app);
    // No document at all.
    let mut empty = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    empty.ui.tool = Tool::MagneticLasso;
    click(&mut empty, 5.0, 5.0, Modifiers::NONE);
    hover_along(&mut empty, &[[5.0, 5.0], [60.0, 5.0], [60.0, 60.0]]);
    close(&mut empty, false);
    assert!(!empty.ui.magnetic.active());
}

// ---------------------------------------------------------------------------- real canvas

fn harness() -> Harness<'static, PhotocraftApp> {
    let mut app = app();
    app.ui.extras.rulers = false;
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
    let v = &mut h.state_mut().ui.views[0];
    v.zoom = 1.0;
    v.center = [100.0, 75.0];
    v.fit_pending = false;
    h.run_steps(2);
    h
}

fn screen(h: &Harness<'static, PhotocraftApp>, x: f64, y: f64) -> Pos2 {
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
    xf.to_screen(x as f32, y as f32)
}

fn ui_click(h: &mut Harness<'static, PhotocraftApp>, x: f64, y: f64) {
    let p = screen(h, x, y);
    h.event(egui::Event::PointerMoved(p));
    h.run_steps(1);
    for down in [true, false] {
        h.event(egui::Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: down, modifiers: Modifiers::NONE });
        h.run_steps(1);
    }
}

fn ui_hover(h: &mut Harness<'static, PhotocraftApp>, pts: &[[f64; 2]]) {
    for w in pts.windows(2) {
        let n = (dist(w[0], w[1]) / 3.0).ceil().max(1.0) as usize;
        for k in 1..=n {
            let t = k as f64 / n as f64;
            let p = screen(h, w[0][0] + (w[1][0] - w[0][0]) * t, w[0][1] + (w[1][1] - w[0][1]) * t);
            h.event(egui::Event::PointerMoved(p));
            h.run_steps(1);
        }
    }
}

#[test]
fn on_the_canvas_the_border_follows_the_hovering_pointer_and_keys_work() {
    let mut h = harness();
    ui_click(&mut h, 52.0, 37.0);
    assert!(h.state().ui.magnetic.active());
    // No button held: the live segment follows the pointer along the edge.
    ui_hover(&mut h, &[[52.0, 37.0], [100.0, 36.0]]);
    let m = &h.state().ui.magnetic;
    let end = m.live.last().or(m.path.last()).copied().unwrap();
    assert!((end[0] - 100.0).abs() < 1.5 && (end[1] - 40.0).abs() < 0.01, "{end:?}");
    // [ and ] change the width; ⌫ removes a point and never clears pixels (Edit › Clear).
    h.key_press(egui::Key::CloseBracket);
    h.run_steps(1);
    assert_eq!(h.state().ui.tool_options.magnetic_width, 11.0);
    h.key_press(egui::Key::OpenBracket);
    h.key_press(egui::Key::OpenBracket);
    h.run_steps(1);
    assert_eq!(h.state().ui.tool_options.magnetic_width, 9.0);
    let revision = h.state().session.active().unwrap().revision;
    let anchors = h.state().ui.magnetic.anchors.len();
    h.key_press(egui::Key::Backspace);
    h.run_steps(1);
    assert_eq!(h.state().ui.magnetic.anchors.len(), anchors - 1);
    assert_eq!(h.state().session.active().unwrap().revision, revision);
    // Esc cancels; ↩ closes.
    h.key_press(egui::Key::Escape);
    h.run_steps(1);
    assert!(!h.state().ui.magnetic.active());
    ui_click(&mut h, 52.0, 37.0);
    h.run_steps(30);
    ui_hover(&mut h, &AROUND[..4]);
    h.key_press(egui::Key::Enter);
    h.run_steps(2);
    assert!(!h.state().ui.magnetic.active());
    let area = selected_area(h.state());
    assert!((area - 5600.0).abs() < 120.0, "{area}");
}

#[test]
fn on_the_canvas_a_double_click_closes_the_border() {
    let mut h = harness();
    ui_click(&mut h, 52.0, 37.0);
    h.run_steps(30);
    ui_hover(&mut h, &AROUND[..4]);
    ui_click(&mut h, 47.0, 113.0);
    ui_click(&mut h, 47.0, 113.0);
    h.run_steps(2);
    assert!(!h.state().ui.magnetic.active(), "{:?}", h.state().ui.magnetic);
    let area = selected_area(h.state());
    assert!((area - 5600.0).abs() < 120.0, "{area}");
    assert_eq!(last_step(h.state()).as_deref(), Some("Magnetic Lasso"));
}
