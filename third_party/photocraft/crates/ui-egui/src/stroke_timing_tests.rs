//! Brush strokes through the real canvas: the moves before egui recognises a drag reach the
//! stroke, and stroke points carry the times the engine's time-based features need.

use egui::{Modifiers, PointerButton, Pos2, vec2};
use egui_kittest::Harness;
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::canvas::ViewXform;
use crate::state::Tool;

fn harness() -> Harness<'static, PhotocraftApp> {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 200, "height": 100, "background": "white"})).unwrap();
    app.sync_views();
    app.ui.tool = Tool::Brush;
    let mut h = Harness::builder().with_size(vec2(900.0, 600.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            let ctx = ui.ctx().clone();
            if !ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            egui::CentralPanel::default().show(ui, |ui| crate::canvas::document_area(app, ui));
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
    h.run_steps(4);
    let v = &mut h.state_mut().ui.views[0];
    v.zoom = 1.0;
    v.center = [100.0, 50.0];
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

fn button(h: &mut Harness<'static, PhotocraftApp>, x: f32, y: f32, down: bool) {
    let p = screen(h, x, y);
    h.event(egui::Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: down, modifiers: Modifiers::NONE });
    h.run_steps(1);
}

fn move_to(h: &mut Harness<'static, PhotocraftApp>, x: f32, y: f32) {
    let p = screen(h, x, y);
    h.event(egui::Event::PointerMoved(p));
    h.run_steps(1);
}

/// The points of the committed `paint.stroke`.
fn committed_points(h: &Harness<'static, PhotocraftApp>) -> Vec<Vec<f64>> {
    let (_, p) = h.state().session.journal.iter().rev().find(|(id, _)| id == "paint.stroke").cloned().expect("a committed stroke");
    p["points"].as_array().expect("points").iter().map(|v| v.as_array().expect("point").iter().filter_map(Value::as_f64).collect()).collect()
}

#[test]
fn moves_before_the_drag_threshold_reach_the_stroke() {
    let mut h = harness();
    move_to(&mut h, 20.0, 50.0);
    button(&mut h, 20.0, 50.0, true);
    // Small steps, each well inside egui's click distance from the press.
    for x in [21.0, 22.0, 23.0] {
        move_to(&mut h, x, 50.0);
    }
    for x in [40.0, 60.0] {
        move_to(&mut h, x, 50.0);
    }
    button(&mut h, 60.0, 50.0, false);
    h.run_steps(2);
    let xs: Vec<f64> = committed_points(&h).iter().map(|p| p[0]).collect();
    for x in [21.0, 22.0, 23.0] {
        assert!(xs.iter().any(|v| (v - x).abs() < 0.01), "the move to {x} is part of the stroke: {xs:?}");
    }
}

#[test]
fn stroke_points_carry_increasing_times_from_zero() {
    let mut h = harness();
    move_to(&mut h, 20.0, 50.0);
    button(&mut h, 20.0, 50.0, true);
    for x in [30.0, 40.0, 50.0, 60.0] {
        move_to(&mut h, x, 50.0);
    }
    button(&mut h, 60.0, 50.0, false);
    h.run_steps(2);
    let pts = committed_points(&h);
    assert!(pts.len() >= 4, "{pts:?}");
    let times: Vec<f64> = pts.iter().map(|p| *p.get(6).expect("timeMs")).collect();
    assert_eq!(times.first().copied(), Some(0.0), "{times:?}");
    assert!(times.windows(2).all(|w| w[1] >= w[0]), "times never go back: {times:?}");
    assert!(times.last().copied().unwrap_or(0.0) > 0.0, "time advances over the frames: {times:?}");
}

/// Points of the drag in progress after holding the pointer still for a while.
fn held_points(h: &mut Harness<'static, PhotocraftApp>) -> (usize, usize) {
    move_to(h, 20.0, 50.0);
    button(h, 20.0, 50.0, true);
    move_to(h, 30.0, 50.0);
    let before = h.state().drag.as_ref().map_or(0, |d| d.points.len());
    h.run_steps(12);
    let after = h.state().drag.as_ref().map_or(0, |d| d.points.len());
    button(h, 30.0, 50.0, false);
    (before, after)
}

#[test]
fn a_held_airbrush_keeps_the_stroke_clock_running() {
    let mut h = harness();
    h.state_mut().run("tools.setBrush", json!({"buildUp": true, "smoothing": {"amount": 0.0}})).unwrap();
    let (before, after) = held_points(&mut h);
    assert!(after > before, "a held airbrush adds timed repeats of its point ({before} → {after})");
}

#[test]
fn a_held_plain_brush_adds_nothing() {
    let mut h = harness();
    h.state_mut().run("tools.setBrush", json!({"buildUp": false, "smoothing": {"amount": 0.0}})).unwrap();
    let (before, after) = held_points(&mut h);
    assert_eq!(after, before, "nothing changes with time, so nothing is repeated");
}

#[test]
fn spread_ms_spaces_a_frames_samples_after_the_last_point() {
    use crate::canvas::spread_ms;
    assert_eq!((0..4).map(|k| spread_ms(100.0, 116.0, k, 4)).collect::<Vec<_>>(), vec![104.0, 108.0, 112.0, 116.0]);
    assert_eq!(spread_ms(116.0, 116.0, 0, 3), 116.0, "a press frame's moves share its time");
    assert_eq!(spread_ms(120.0, 116.0, 0, 1), 116.0, "a clock that went back never goes below now");
    assert_eq!(spread_ms(f64::NAN, 116.0, 0, 1), 116.0);
}

#[test]
fn moves_in_one_frame_take_the_pressures_the_pen_reported_between_them() {
    use crate::stylus::PenSample;
    let mut h = harness();
    let pen = |p: f32| Some(PenSample { pressure: p, ..Default::default() });
    h.state().stylus.feed.set(pen(0.2));
    move_to(&mut h, 20.0, 50.0);
    button(&mut h, 20.0, 50.0, true);
    // One frame: three moves while the pen reported 0.4, 0.6 and 0.8. They go straight into the
    // next frame's raw input, as moves piling up during a slow frame do (kittest's `event`
    // delivers queued pointer moves one per frame).
    for (x, p) in [(30.0, 0.4), (40.0, 0.6), (50.0, 0.8)] {
        h.state().stylus.feed.set(pen(p));
        let q = screen(&h, x, 50.0);
        h.input_mut().events.push(egui::Event::PointerMoved(q));
    }
    h.run_steps(1);
    button(&mut h, 50.0, 50.0, false);
    h.run_steps(2);
    let pts = committed_points(&h);
    let tail: Vec<f64> = pts.iter().filter(|p| p[0] >= 29.0).map(|p| (p[2] * 100.0).round() / 100.0).collect();
    assert_eq!(tail.get(..3), Some(&[0.4, 0.6, 0.8][..]), "each move keeps its own pressure: {pts:?}");
}

#[test]
fn the_pressure_curve_shapes_the_committed_stroke() {
    use crate::stylus::PenSample;
    let mut h = harness();
    h.state_mut().run("prefs.set", json!({"path": "tools.pressureCurve", "value": [[0.0, 0.0], [0.3, 0.6], [1.0, 1.0]]})).unwrap();
    h.state().stylus.feed.set(Some(PenSample { pressure: 0.3, ..Default::default() }));
    move_to(&mut h, 20.0, 50.0);
    button(&mut h, 20.0, 50.0, true);
    for x in [40.0, 60.0] {
        h.state().stylus.feed.set(Some(PenSample { pressure: 0.3, ..Default::default() }));
        move_to(&mut h, x, 50.0);
    }
    button(&mut h, 60.0, 50.0, false);
    h.run_steps(2);
    let pts = committed_points(&h);
    assert!(pts.iter().all(|p| (p[2] - 0.6).abs() < 1e-3), "pen 0.3 through the curve is 0.6: {pts:?}");
}

/// A Brush whose size follows pen pressure, painting on a new empty layer.
fn pressure_sized_brush(h: &mut Harness<'static, PhotocraftApp>) {
    let app = h.state_mut();
    app.run("layer.new.layer", json!({})).unwrap();
    let brush = json!({"size": 40, "hardness": 1.0, "smoothing": {"amount": 0.0}, "shapeDynamics": {"enabled": true, "size": {"control": "penPressure"}}});
    app.run("tools.setBrush", json!({ "brush": brush })).unwrap();
    h.run_steps(1);
}

/// Width in pixels of the paint on the active layer along row `y`.
fn painted_width(h: &Harness<'static, PhotocraftApp>, y: i32) -> usize {
    let st = h.state().session.active().expect("a document");
    let s = st.doc.layer(st.active_layer.expect("an active layer")).and_then(|l| l.surface()).expect("a raster layer");
    (0..200).filter(|x| s.rgba(*x, y)[3] > 0.5).count()
}

/// Press and release at (`x`, 50) in one frame, the pen reporting `samples` (`None`: cleared, as
/// a lifting pen or a mouse leaves it) before the UI runs: a tap shorter than a frame.
fn tap(h: &mut Harness<'static, PhotocraftApp>, x: f32, samples: &[Option<f32>]) {
    use crate::stylus::PenSample;
    move_to(h, x, 50.0);
    for p in samples {
        h.state().stylus.feed.set(p.map(|pressure| PenSample { pressure, ..Default::default() }));
    }
    let q = screen(h, x, 50.0);
    for pressed in [true, false] {
        h.input_mut().events.push(egui::Event::PointerButton { pos: q, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
    }
    h.run_steps(3);
}

#[test]
fn a_tap_shorter_than_a_frame_paints_at_the_pens_pressure() {
    // #1798: the pen lifted (its sample cleared) before the UI ran, so the tap read as a mouse
    // and painted a full-size dab whatever the pressure.
    let mut h = harness();
    pressure_sized_brush(&mut h);
    tap(&mut h, 50.0, &[Some(0.2), None]);
    assert!((committed_points(&h)[0][2] - 0.2).abs() < 1e-3, "{:?}", committed_points(&h));
    tap(&mut h, 150.0, &[Some(0.9), None]);
    assert!((committed_points(&h)[0][2] - 0.9).abs() < 1e-3, "{:?}", committed_points(&h));
    let st = h.state().session.active().unwrap();
    let s = st.doc.layer(st.active_layer.unwrap()).and_then(|l| l.surface()).unwrap();
    let width = |x0: i32, x1: i32| (x0..x1).filter(|x| s.rgba(*x, 50)[3] > 0.5).count() as f64;
    let (light, firm) = (width(0, 100), width(100, 200));
    assert!((light / 40.0 - 0.2).abs() < 0.1, "a light tap is a small dab: {light} px");
    assert!((firm / 40.0 - 0.9).abs() < 0.1, "a firm tap is a large dab: {firm} px");
}

#[test]
fn a_tap_never_paints_at_the_lifts_zero_pressure() {
    // X11 and a hovering macOS pen report pressure 0 around the contact.
    let mut h = harness();
    pressure_sized_brush(&mut h);
    tap(&mut h, 50.0, &[Some(0.0), Some(0.6), Some(0.0)]);
    assert!((committed_points(&h)[0][2] - 0.6).abs() < 1e-3, "{:?}", committed_points(&h));
}

#[test]
fn a_tap_over_two_frames_paints_at_the_pressure_it_reached() {
    use crate::stylus::PenSample;
    let mut h = harness();
    pressure_sized_brush(&mut h);
    let pen = |p: f32| Some(PenSample { pressure: p, ..Default::default() });
    move_to(&mut h, 50.0, 50.0);
    h.state().stylus.feed.set(pen(0.1));
    button(&mut h, 50.0, 50.0, true);
    h.state().stylus.feed.set(pen(0.5));
    h.state().stylus.feed.set(pen(0.0));
    button(&mut h, 50.0, 50.0, false);
    h.run_steps(2);
    let pts = committed_points(&h);
    assert_eq!(pts.len(), 1, "{pts:?}");
    assert!((pts[0][2] - 0.5).abs() < 1e-3, "the tap's peak, not its first touch: {pts:?}");
}

#[test]
fn a_mouse_tap_paints_a_full_dab() {
    let mut h = harness();
    pressure_sized_brush(&mut h);
    tap(&mut h, 50.0, &[]);
    assert_eq!(committed_points(&h)[0][2], 1.0);
    let w = painted_width(&h, 50) as f64;
    assert!((w - 40.0).abs() <= 3.0, "full size: {w} px");
}

#[test]
fn a_stroke_ends_at_the_last_pressure_the_pen_drew_with() {
    use crate::stylus::PenSample;
    // The lift either clears the pen (macOS proximity, a non-tablet button-up) or reports 0
    // (X11) before the UI runs the frame with the stroke's last moves.
    for lift in [None, Some(0.0)] {
        let mut h = harness();
        pressure_sized_brush(&mut h);
        let pen = |p: f32| Some(PenSample { pressure: p, ..Default::default() });
        h.state().stylus.feed.set(pen(0.4));
        move_to(&mut h, 20.0, 50.0);
        button(&mut h, 20.0, 50.0, true);
        for x in [30.0, 40.0] {
            h.state().stylus.feed.set(pen(0.4));
            move_to(&mut h, x, 50.0);
        }
        // One frame: the last moves, then the lift.
        for x in [50.0, 60.0] {
            h.state().stylus.feed.set(pen(0.4));
            let q = screen(&h, x, 50.0);
            h.input_mut().events.push(egui::Event::PointerMoved(q));
        }
        h.state().stylus.feed.set(lift.map(|p| PenSample { pressure: p, ..Default::default() }));
        let q = screen(&h, 60.0, 50.0);
        h.input_mut().events.push(egui::Event::PointerButton { pos: q, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
        h.run_steps(3);
        let pts = committed_points(&h);
        assert!(pts.iter().all(|p| (p[2] - 0.4).abs() < 1e-3), "lift {lift:?}: no full or zero pressure dab at the end: {pts:?}");
    }
}

#[test]
fn releasing_a_stroke_commits_the_live_pixels_and_replays_identically() {
    let mut h = harness();
    h.state_mut().run("tools.setBrush", json!({"brush": {"size": 14, "hardness": 0.2}, "smoothing": {"amount": 0.3}})).unwrap();
    move_to(&mut h, 20.0, 50.0);
    button(&mut h, 20.0, 50.0, true);
    for (x, y) in [(40.0, 30.0), (70.0, 60.0), (110.0, 40.0), (150.0, 70.0)] {
        move_to(&mut h, x, y);
    }
    button(&mut h, 150.0, 70.0, false);
    h.run_steps(2);
    let app = h.state();
    let st = app.session.active().unwrap();
    let committed = app.committed_stroke.as_ref().expect("the commit took the live stroke's pixels");
    assert_eq!((committed.doc, committed.revision), (st.doc.id, st.revision));
    // Replaying the journaled command from scratch gives the same pixels.
    let (_, p) = app.session.journal.iter().rev().find(|(id, _)| id == "paint.stroke").cloned().unwrap();
    let mut fresh = photocraft_engine::Session::new();
    fresh.execute("file.new", json!({"width": 200, "height": 100, "background": "white"})).unwrap();
    fresh.execute("tools.setBrush", json!({"brush": {"size": 14, "hardness": 0.2}, "smoothing": {"amount": 0.3}})).unwrap();
    fresh.execute("paint.stroke", p).unwrap();
    let a = photocraft_compose::render(&st.doc, st.doc.bounds());
    let b = photocraft_compose::render(&fresh.active().unwrap().doc, st.doc.bounds());
    assert_eq!(a.px.len(), b.px.len());
    assert!(a.px.iter().zip(&b.px).all(|(x, y)| x == y), "live commit and replay differ");
}

#[test]
fn a_committed_stroke_refreshes_only_what_each_cache_had_not_shown() {
    use crate::canvas::CommittedStroke;
    let r = |x: i32| photocraft_geom::Rect::new(x, 0, x + 10, 10);
    let c = CommittedStroke::for_test(100, vec![r(0), r(20), r(40)]);
    assert_eq!(c.since(103), Some(photocraft_geom::Rect::EMPTY), "showed the last step: nothing");
    assert_eq!(c.since(102), Some(r(40)), "one step behind: that step");
    assert_eq!(c.since(0), Some(r(0).union(&r(20)).union(&r(40))), "never showed the stroke: all of it");
    assert_eq!(c.since(99), None, "not a key of this stroke");
}

#[test]
fn the_stroke_trace_times_each_frame_of_a_stroke_and_ends_on_release() {
    // The app's hooks around the canvas (`raw_input_hook` starts a frame, `ui` ends it); the
    // canvas times the live stroke and the release's commit itself.
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 200, "height": 100, "background": "white"})).unwrap();
    app.sync_views();
    app.ui.tool = Tool::Brush;
    app.stroke_trace.output = crate::stroke_trace::Output::Stderr;
    let mut h = Harness::builder().with_size(vec2(900.0, 600.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            let ctx = ui.ctx().clone();
            if !ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            let t0 = crate::gpu_canvas::now_ms();
            let moves = ctx.input(|i| i.raw.events.iter().filter(|e| matches!(e, egui::Event::PointerMoved(_))).count());
            app.stroke_trace.begin_frame(t0, moves);
            app.stroke_trace.cpu_prev(Some(0.004));
            egui::CentralPanel::default().show(ui, |ui| crate::canvas::document_area(app, ui));
            let stroking = app.live_stroke.is_some();
            app.stroke_trace.end_frame(crate::gpu_canvas::now_ms() - t0, stroking, || "test".into());
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
    h.run_steps(4);
    move_to(&mut h, 20.0, 50.0);
    button(&mut h, 20.0, 50.0, true);
    for x in [40.0, 60.0, 80.0, 100.0, 120.0] {
        move_to(&mut h, x, 50.0);
    }
    assert!(h.state().stroke_trace.last.is_none(), "the stroke is still being drawn");
    button(&mut h, 120.0, 50.0, false);
    h.run_steps(2);
    let s = h.state().stroke_trace.last.clone().expect("releasing ends the stroke's trace");
    assert!(s.frames >= 6, "{s:?}");
    assert!(s.stroke[1] > 0.0 && s.ui[0] >= s.stroke[0], "{s:?}");
    assert_eq!(s.moves[1], 1.0, "one move per test frame: {s:?}");
    assert_eq!(s.backend, "test");
}

/// A Clone Stamp stroke through the canvas renders the points its commit gets (position and
/// pressure, no tilt or times), so the commit can take the live pixels; they are what replaying
/// the journaled command paints.
#[test]
fn releasing_a_clone_stamp_stroke_commits_what_a_replay_paints() {
    let setup = |run: &mut dyn FnMut(&str, Value)| {
        run("tools.setBrush", json!({"brush": {"size": 16, "hardness": 0.4}}));
        run("paint.stroke", json!({"points": [[10, 20], [190, 80]], "size": 30, "hardness": 0.5, "color": "#2060c0"}));
        run("cloneSource.set", json!({"source": [40, 30]}));
    };
    let mut h = harness();
    setup(&mut |id, p| {
        h.state_mut().run(id, p).unwrap();
    });
    h.state_mut().ui.tool = Tool::CloneStamp;
    h.run_steps(1);
    move_to(&mut h, 60.0, 50.0);
    button(&mut h, 60.0, 50.0, true);
    for (x, y) in [(80.0, 40.0), (110.0, 60.0), (140.0, 45.0), (170.0, 65.0)] {
        move_to(&mut h, x, y);
    }
    let shown = crate::canvas::live_retouch_points(h.state()).expect("a live Clone Stamp stroke");
    button(&mut h, 170.0, 65.0, false);
    h.run_steps(2);
    let app = h.state();
    let (_, p) = app.session.journal.iter().rev().find(|(id, _)| id == "paint.cloneStamp").cloned().expect("a committed Clone Stamp stroke");
    let committed: Vec<_> = p["points"]
        .as_array()
        .unwrap()
        .iter()
        .map(|q| photocraft_engine::paint::StrokePoint::new(q[0].as_f64().unwrap(), q[1].as_f64().unwrap(), q[2].as_f64().unwrap() as f32))
        .collect();
    assert!(shown.len() > 1 && committed.starts_with(&shown), "live {shown:?} vs committed {committed:?}");
    let mut fresh = photocraft_engine::Session::new();
    fresh.execute("file.new", json!({"width": 200, "height": 100, "background": "white"})).unwrap();
    setup(&mut |id, p| {
        fresh.execute(id, p).unwrap();
    });
    fresh.execute("paint.cloneStamp", p).unwrap();
    let st = app.session.active().unwrap();
    let a = photocraft_compose::render(&st.doc, st.doc.bounds());
    let b = photocraft_compose::render(&fresh.active().unwrap().doc, st.doc.bounds());
    assert_eq!(a.px.len(), b.px.len());
    assert!(a.px.iter().zip(&b.px).all(|(x, y)| x == y), "live commit and replay differ");
}
