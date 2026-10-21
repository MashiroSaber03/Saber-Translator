//! Marquee modifiers through the real canvas (#208): ⇧ during the drag draws a square or circle,
//! ⌥ draws from the centre (both together work too), a "W × H px" readout follows the cursor, and
//! modifiers held before the drag still pick add/subtract (#188) instead of constraining.

use egui::{Modifiers, PointerButton, Pos2, pos2, vec2};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use photocraft_geom::Rect;
use serde_json::json;

use crate::PhotocraftApp;
use crate::canvas::ViewXform;
use crate::state::Tool;

fn harness(tool: Tool) -> Harness<'static, PhotocraftApp> {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 400, "height": 300})).unwrap();
    app.sync_views();
    app.ui.extras.rulers = false;
    app.ui.tool = tool;
    let mut h = Harness::builder().with_size(vec2(1000.0, 700.0)).build_ui_state(
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
    // 100 %: one document pixel per point, so drags land on whole pixels.
    let v = &mut h.state_mut().ui.views[0];
    v.zoom = 1.0;
    v.center = [200.0, 150.0];
    v.fit_pending = false;
    h.run_steps(2);
    h
}

fn view_xform(h: &Harness<'static, PhotocraftApp>) -> ViewXform {
    let app = h.state();
    let v = &app.ui.views[0];
    ViewXform {
        rect: crate::rulers::content_rect(app, app.last_canvas_rect),
        zoom: v.zoom,
        center: v.center,
        flip: app.ui.view.flip_horizontal,
        rotation: v.rotation,
        aspect: app.ui.view.display_aspect(),
    }
}

fn screen(h: &Harness<'static, PhotocraftApp>, x: f32, y: f32) -> Pos2 {
    view_xform(h).to_screen(x, y)
}

fn mods(h: &mut Harness<'static, PhotocraftApp>, m: Modifiers) {
    h.event(egui::Event::ModifiersChanged(m));
    h.run_steps(1);
}

fn button(h: &mut Harness<'static, PhotocraftApp>, p: Pos2, down: bool, m: Modifiers) {
    h.event(egui::Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: down, modifiers: m });
    h.run_steps(1);
}

/// Move the pointer to document point `(x, y)` in a few steps.
fn move_to(h: &mut Harness<'static, PhotocraftApp>, x: f32, y: f32) {
    let p = screen(h, x, y);
    h.event(egui::Event::PointerMoved(p));
    h.run_steps(2);
}

fn press_at(h: &mut Harness<'static, PhotocraftApp>, x: f32, y: f32, m: Modifiers) {
    let p = screen(h, x, y);
    h.event(egui::Event::PointerMoved(p));
    h.run_steps(1);
    button(h, p, true, m);
    // Past egui's click distance, so the drag starts at the press point.
    move_to(h, x + 8.0, y + 6.0);
}

fn release_at(h: &mut Harness<'static, PhotocraftApp>, x: f32, y: f32, m: Modifiers) {
    move_to(h, x, y);
    button(h, screen(h, x, y), false, m);
    h.run_steps(2);
}

fn selection(h: &Harness<'static, PhotocraftApp>) -> Rect {
    h.state().session.active().unwrap().doc.selection.as_ref().map_or(Rect::EMPTY, |s| s.content_bounds())
}

/// The two-row readout shows `W: {w} px` and `H: {ht} px` (Photoshop's format).
fn readout(h: &Harness<'static, PhotocraftApp>, w: i32, ht: i32) -> bool {
    let labels = h.query_by_label("W:").is_some() && h.query_by_label("H:").is_some();
    let count = |v: i32| h.query_all_by_label(&format!("{v} px")).count();
    let values = if w == ht { count(w) == 2 } else { count(w) == 1 && count(ht) == 1 };
    labels && values && h.query_all_by_label_contains(" px").count() == 2
}

#[test]
fn marquee_intent_is_captured_before_the_drag_threshold() {
    for tool in [Tool::RectMarquee, Tool::EllipseMarquee] {
        for at_press in [Modifiers::NONE, Modifiers::SHIFT] {
            let mut h = harness(tool);
            h.state_mut().ui.extras.snap = false;
            h.state_mut().run("select.rect", json!({"x": 10, "y": 10, "width": 20, "height": 20})).unwrap();
            mods(&mut h, at_press);
            move_to(&mut h, 100.0, 100.0);
            let start = screen(&h, 100.0, 100.0);
            button(&mut h, start, true, at_press);
            assert!(h.state().drag.is_none(), "still below the drag threshold");
            // Change Shift while the pointer is still at the press, before egui starts a drag.
            let during = if at_press.shift { Modifiers::NONE } else { Modifiers::SHIFT };
            mods(&mut h, during);
            move_to(&mut h, 180.0, 130.0);
            assert!(readout(&h, 80, if at_press.shift { 30 } else { 80 }), "{tool:?}, Shift at press: {}", at_press.shift);
            release_at(&mut h, 180.0, 130.0, during);
            let st = h.state().session.active().unwrap();
            let sel = st.doc.selection.as_ref().unwrap();
            if at_press.shift {
                assert!(sel.sample_channel(20, 20, 0) > 0.99, "Shift at the press keeps Add mode");
                assert_eq!(selection(&h), Rect::new(10, 10, 180, 130));
            } else {
                assert!(sel.sample_channel(20, 20, 0) < 0.01, "Shift after the press constrains a replacement");
                assert_eq!(selection(&h), Rect::new(100, 100, 180, 180));
            }
        }
    }
}

#[test]
fn shift_during_the_drag_draws_a_square_and_the_readout_follows() {
    let mut h = harness(Tool::RectMarquee);
    press_at(&mut h, 50.0, 50.0, Modifiers::NONE);
    move_to(&mut h, 150.0, 90.0);
    assert!(readout(&h, 100, 40), "free drag: 100 × 40");
    mods(&mut h, Modifiers::SHIFT);
    move_to(&mut h, 150.0, 91.0);
    assert!(readout(&h, 100, 100), "⇧: the larger extent wins");
    release_at(&mut h, 150.0, 91.0, Modifiers::SHIFT);
    mods(&mut h, Modifiers::NONE);
    assert_eq!(selection(&h), Rect::new(50, 50, 150, 150));
    assert!(h.query_by_label("W:").is_none() && h.query_by_label_contains(" px").is_none(), "the readout goes away with the drag");
}

/// #204: Preferences › Tools › Show Transformation Values off hides the readout, not the marquee.
#[test]
fn show_transformation_values_off_hides_the_readout() {
    for shown in [false, true] {
        let mut h = harness(Tool::RectMarquee);
        h.state_mut().run("prefs.set", json!({"path": "tools.showTransformationValues", "value": shown})).unwrap();
        press_at(&mut h, 50.0, 50.0, Modifiers::NONE);
        move_to(&mut h, 150.0, 90.0);
        assert_eq!(readout(&h, 100, 40), shown, "shown: {shown}");
        if !shown {
            assert!(h.query_by_label("W:").is_none() && h.query_by_label_contains(" px").is_none());
        }
        release_at(&mut h, 150.0, 90.0, Modifiers::NONE);
        assert_eq!(selection(&h), Rect::new(50, 50, 150, 90), "the marquee still selects");
    }
}

/// #204: the Free Transform readout (#1767) follows Show Transformation Values too.
#[test]
fn show_transformation_values_off_hides_the_free_transform_readout() {
    use crate::canvas::{ToolEvent, tool_event};
    for shown in [false, true] {
        let mut h = harness(Tool::Move);
        let app = h.state_mut();
        app.run("prefs.set", json!({"path": "tools.showTransformationValues", "value": shown})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        app.run("select.all", json!({})).unwrap();
        app.run("edit.fill", json!({"color": "#3366cc"})).unwrap();
        app.run("select.deselect", json!({})).unwrap();
        app.ui.extras.snap = false;
        crate::menus::invoke(app, &egui::Context::default(), "edit.freeTransform", json!({})).unwrap();
        // Move the box by dragging inside it (away from the centre pivot), then hover there.
        tool_event(app, ToolEvent::Down { x: 100.0, y: 100.0, pressure: 1.0 }, Modifiers::NONE);
        tool_event(app, ToolEvent::Move { x: 130.0, y: 120.0, pressure: 1.0 }, Modifiers::NONE);
        assert!(crate::transform_tool::live_readout(h.state()).is_some(), "a live readout to show");
        let at = screen(&h, 130.0, 120.0);
        h.event(egui::Event::PointerMoved(at));
        h.run_steps(2);
        assert_eq!(h.query_by_label("ΔX:").is_some(), shown, "shown: {shown}");
    }
}

#[test]
fn alt_draws_from_the_centre_and_shift_alt_a_centred_circle() {
    let mut h = harness(Tool::RectMarquee);
    press_at(&mut h, 200.0, 150.0, Modifiers::NONE);
    mods(&mut h, Modifiers::ALT);
    move_to(&mut h, 240.0, 170.0);
    assert!(readout(&h, 80, 40));
    release_at(&mut h, 240.0, 170.0, Modifiers::ALT);
    mods(&mut h, Modifiers::NONE);
    assert_eq!(selection(&h), Rect::new(160, 130, 240, 170), "centred on the press point");
    // Elliptical, ⇧⌥: a circle centred on the press point (replacing the selection).
    let mut h = harness(Tool::EllipseMarquee);
    press_at(&mut h, 200.0, 150.0, Modifiers::NONE);
    mods(&mut h, Modifiers::SHIFT | Modifiers::ALT);
    move_to(&mut h, 230.0, 160.0);
    assert!(readout(&h, 60, 60));
    release_at(&mut h, 230.0, 160.0, Modifiers::SHIFT | Modifiers::ALT);
    mods(&mut h, Modifiers::NONE);
    let r = selection(&h);
    assert!(r.x0.abs_diff(170) <= 1 && r.x1.abs_diff(230) <= 1 && r.y0.abs_diff(120) <= 1 && r.y1.abs_diff(180) <= 1, "{r:?}");
}

#[test]
fn the_live_outline_snaps_to_pixels_and_matches_the_committed_selection() {
    for tool in [Tool::RectMarquee, Tool::EllipseMarquee] {
        let mut h = harness(tool);
        // Zoomed in, the pointer lands between pixel edges.
        let v = &mut h.state_mut().ui.views[0];
        v.zoom = 8.0;
        v.center = [20.0, 15.0];
        h.run_steps(2);
        press_at(&mut h, 10.3, 10.3, Modifiers::NONE);
        move_to(&mut h, 20.6, 15.4);
        let app = h.state();
        let live = app.drag.as_ref().and_then(|d| crate::canvas::marquee_preview_px(&app.ui.tool_options, d, app.point_zoom())).unwrap();
        assert_eq!(live, [10.0, 10.0, 21.0, 16.0], "{tool:?}: whole pixels while dragging");
        release_at(&mut h, 20.6, 15.4, Modifiers::NONE);
        assert_eq!(selection(&h), Rect::new(10, 10, 21, 16), "{tool:?}: the commit is what was shown");
    }
}

/// #2512: the hinting must follow the same transform as the release. With a rotated view the
/// marquee projects onto a rotated rectangle, and an axis-aligned screen box drawn through its
/// two diagonal corners is a different shape somewhere else entirely.
#[test]
fn the_live_outline_matches_the_committed_selection_in_a_rotated_view() {
    for tool in [Tool::RectMarquee, Tool::EllipseMarquee] {
        let mut h = harness(tool);
        let v = &mut h.state_mut().ui.views[0];
        v.rotation = 30.0;
        h.run_steps(2);
        press_at(&mut h, 50.0, 50.0, Modifiers::NONE);
        move_to(&mut h, 150.0, 90.0);
        let xf = view_xform(&h);
        let (live, pts) = {
            let app = h.state();
            let d = app.drag.as_ref().expect("the drag is on");
            let live = crate::canvas::marquee_preview_px(&app.ui.tool_options, d, app.point_zoom()).expect("a marquee is dragged");
            // Exactly what the canvas paints for this drag.
            (live, crate::canvas::marquee_drag_ants(d, &xf, Some(live)))
        };
        release_at(&mut h, 150.0, 90.0, Modifiers::NONE);
        let sel = selection(&h);
        // The pointer round-trips through f32 screen points, so a corner can land a hair over a
        // pixel edge; what matters is that the rotation moves neither the selection nor the
        // hinting, and that the release commits what was shown.
        assert!(
            (sel.x0 as f64 - live[0]).abs() <= 1.0
                && (sel.y0 as f64 - live[1]).abs() <= 1.0
                && (sel.x1 as f64 - live[2]).abs() <= 1.0
                && (sel.y1 as f64 - live[3]).abs() <= 1.0,
            "{tool:?}: the release commits what the hinting showed: {sel:?} vs {live:?}"
        );
        if tool == Tool::RectMarquee {
            // Each corner is a corner of the dragged rectangle, mapped: an axis-aligned box
            // through the two diagonal corners puts all four elsewhere.
            let snap = |p: Pos2| pos2(p.x.round() + 0.5, p.y.round() + 0.5);
            let want = [[live[0], live[1]], [live[2], live[1]], [live[2], live[3]], [live[0], live[3]]].map(|[x, y]| snap(xf.to_screen(x as f32, y as f32)));
            for (i, (p, w)) in pts.iter().zip(want).enumerate() {
                assert!((p.x - w.x).abs() < 0.01 && (p.y - w.y).abs() < 0.01, "{tool:?}: corner {i} at {p:?}, the dragged rectangle's corner is at {w:?}");
            }
        } else {
            // Every point maps back onto the ellipse the drag drew.
            let (cx, cy, rx, ry) = ((live[0] + live[2]) * 0.5, (live[1] + live[3]) * 0.5, (live[2] - live[0]) * 0.5, (live[3] - live[1]) * 0.5);
            for p in &pts {
                let [x, y] = xf.to_doc(*p);
                let e = ((x - cx) / rx).powi(2) + ((y - cy) / ry).powi(2);
                assert!((e - 1.0).abs() < 0.08, "{tool:?}: {p:?} maps back to ({x},{y}), off the dragged ellipse (e = {e})");
            }
        }
    }
}

/// 800 %, snapping off, the view centred on (30, 30): the pointer lands between pixel edges.
fn zoomed(tool: Tool) -> Harness<'static, PhotocraftApp> {
    let mut h = harness(tool);
    h.state_mut().ui.extras.snap = false;
    let v = &mut h.state_mut().ui.views[0];
    v.zoom = 8.0;
    v.center = [30.0, 30.0];
    h.run_steps(2);
    h
}

/// Press at `from` and move to `to` in six ordinary pointer moves, the button still held.
fn drag_to(h: &mut Harness<'static, PhotocraftApp>, from: [f32; 2], to: [f32; 2]) {
    move_to(h, from[0], from[1]);
    button(h, screen(h, from[0], from[1]), true, Modifiers::NONE);
    for i in 1..=6 {
        let t = i as f32 / 6.0;
        move_to(h, from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t);
    }
}

/// #2391: a one-pixel column or row shown while dragging (ants and readout) is the selection the
/// release keeps, also when the drag starts outside a selection it replaces (Photoshop). A click
/// without a drag still deselects.
#[test]
fn a_one_pixel_column_or_row_is_kept_on_release() {
    for tool in [Tool::RectMarquee, Tool::EllipseMarquee] {
        for (name, end, want) in [("column", [10.6, 50.4], Rect::new(10, 10, 11, 51)), ("row", [50.6, 10.8], Rect::new(10, 10, 51, 11))] {
            for existing in [false, true] {
                let mut h = zoomed(tool);
                if existing {
                    h.state_mut().run("select.rect", json!({"x": 100, "y": 100, "width": 10, "height": 10})).unwrap();
                }
                drag_to(&mut h, [10.2, 10.4], end);
                let app = h.state();
                let live = app.drag.as_ref().and_then(|d| crate::canvas::marquee_preview_px(&app.ui.tool_options, d, app.point_zoom()));
                let shown = [want.x0, want.y0, want.x1, want.y1].map(f64::from);
                assert_eq!(live, Some(shown), "{tool:?} {name} (existing {existing}): the outline while dragging");
                assert!(readout(&h, want.width() as i32, want.height() as i32), "{tool:?} {name}: W/H readout");
                let p = screen(&h, end[0], end[1]);
                button(&mut h, p, false, Modifiers::NONE);
                h.run_steps(2);
                assert_eq!(selection(&h), want, "{tool:?} {name} (existing {existing}): the release keeps what was shown");
                assert!(h.query_by_label("W:").is_none(), "the readout goes away with the drag");
            }
        }
        // A plain click outside the selection deselects.
        let mut h = zoomed(tool);
        h.state_mut().run("select.rect", json!({"x": 100, "y": 100, "width": 10, "height": 10})).unwrap();
        move_to(&mut h, 10.2, 10.4);
        let p = screen(&h, 10.2, 10.4);
        button(&mut h, p, true, Modifiers::NONE);
        button(&mut h, p, false, Modifiers::NONE);
        h.run_steps(2);
        assert!(h.state().session.active().unwrap().doc.selection.is_none(), "{tool:?}: a click deselects");
    }
}

#[test]
fn modifiers_held_before_the_drag_pick_the_mode_not_the_shape() {
    // #188: ⇧ held at the press adds (no square); released and pressed again it constrains.
    let mut h = harness(Tool::RectMarquee);
    h.state_mut().run("select.rect", json!({"x": 10, "y": 10, "width": 20, "height": 20})).unwrap();
    mods(&mut h, Modifiers::SHIFT);
    press_at(&mut h, 100.0, 100.0, Modifiers::SHIFT);
    move_to(&mut h, 180.0, 130.0);
    assert!(readout(&h, 80, 30), "not squared by the add-mode ⇧");
    release_at(&mut h, 180.0, 130.0, Modifiers::SHIFT);
    mods(&mut h, Modifiers::NONE);
    assert_eq!(selection(&h), Rect::new(10, 10, 180, 130), "added to the first selection");
    let st = h.state().session.active().unwrap();
    let sel = st.doc.selection.as_ref().unwrap();
    assert!(sel.sample_channel(20, 20, 0) > 0.99 && sel.sample_channel(150, 120, 0) > 0.99 && sel.sample_channel(60, 60, 0) < 0.01);
    // ⇧ at the press, released, pressed again: add mode and a square.
    let mut h = harness(Tool::RectMarquee);
    h.state_mut().run("select.rect", json!({"x": 10, "y": 10, "width": 20, "height": 20})).unwrap();
    mods(&mut h, Modifiers::SHIFT);
    press_at(&mut h, 100.0, 100.0, Modifiers::SHIFT);
    mods(&mut h, Modifiers::NONE);
    move_to(&mut h, 180.0, 130.0);
    mods(&mut h, Modifiers::SHIFT);
    move_to(&mut h, 180.0, 131.0);
    assert!(readout(&h, 80, 80));
    release_at(&mut h, 180.0, 131.0, Modifiers::SHIFT);
    mods(&mut h, Modifiers::NONE);
    let st = h.state().session.active().unwrap();
    let sel = st.doc.selection.as_ref().unwrap();
    assert!(sel.sample_channel(20, 20, 0) > 0.99, "still added");
    assert!(sel.sample_channel(170, 170, 0) > 0.99 && sel.sample_channel(170, 185, 0) < 0.01, "an 80 px square");
    // ⌥ held at the press subtracts.
    let mut h = harness(Tool::RectMarquee);
    h.state_mut().run("select.rect", json!({"x": 0, "y": 0, "width": 400, "height": 300})).unwrap();
    mods(&mut h, Modifiers::ALT);
    press_at(&mut h, 100.0, 100.0, Modifiers::ALT);
    release_at(&mut h, 150.0, 120.0, Modifiers::ALT);
    mods(&mut h, Modifiers::NONE);
    let st = h.state().session.active().unwrap();
    let sel = st.doc.selection.as_ref().unwrap();
    assert!(sel.sample_channel(120, 110, 0) < 0.01 && sel.sample_channel(90, 110, 0) > 0.99, "subtracted, not centred");
}

/// #208: `⇧⌥` held at the press intersects, the third selection mode after add (`⇧`) and subtract
/// (`⌥`). Like those, the modifier at the press picks the mode; here the keys are released once the
/// press has fixed the mode, so the drag stays a plain rectangle and only the overlap of the
/// existing selection and the dragged rectangle survives. The three sample points below are each in
/// exactly one of {overlap, first-only, drag-only}, so they pass only for intersect, not for
/// replace, add or subtract.
#[test]
fn shift_alt_before_the_drag_intersects_the_selection() {
    let mut h = harness(Tool::RectMarquee);
    // A: the existing selection, 50..200 on both axes.
    h.state_mut().run("select.rect", json!({"x": 50, "y": 50, "width": 150, "height": 150})).unwrap();
    let both = Modifiers::SHIFT | Modifiers::ALT;
    mods(&mut h, both);
    press_at(&mut h, 120.0, 120.0, both);
    // Mode is fixed at the press; releasing the keys keeps intersect mode but leaves the drag a
    // plain rectangle (no square/centre constraint), so B is roughly 120..300 on both axes.
    mods(&mut h, Modifiers::NONE);
    move_to(&mut h, 300.0, 300.0);
    release_at(&mut h, 300.0, 300.0, Modifiers::NONE);
    let st = h.state().session.active().unwrap();
    let sel = st.doc.selection.as_ref().expect("a selection survives the intersect");
    assert!(sel.sample_channel(160, 160, 0) > 0.99, "the overlap of the two rectangles stays selected");
    assert!(sel.sample_channel(60, 60, 0) < 0.01, "a point only in the existing selection is dropped (not add)");
    assert!(sel.sample_channel(260, 260, 0) < 0.01, "a point only in the dragged rectangle is dropped (not replace)");
}

/// `cargo test --release -p photocraft-ui-egui marquee_drag_bench -- --ignored --nocapture`
#[test]
#[ignore]
fn marquee_drag_bench() {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 6000, "height": 4000})).unwrap();
    app.run("filter.render.clouds", json!({})).unwrap();
    app.sync_views();
    app.ui.tool = Tool::RectMarquee;
    let mut h = Harness::builder().with_size(vec2(1600.0, 1000.0)).build_ui_state(
        |ui, app: &mut PhotocraftApp| {
            if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                return;
            }
            egui::CentralPanel::default().show(ui, |ui| crate::canvas::document_area(app, ui));
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
    h.run_steps(6);
    press_at(&mut h, 1000.0, 1000.0, Modifiers::NONE);
    mods(&mut h, Modifiers::SHIFT | Modifiers::ALT);
    let n = 60;
    let t0 = std::time::Instant::now();
    for i in 0..n {
        let p = screen(&h, 1500.0 + i as f32 * 20.0, 1300.0 + i as f32 * 7.0);
        h.event(egui::Event::PointerMoved(p));
        h.step();
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(n);
    eprintln!("marquee drag on 6000x4000 (shift+alt, readout): {ms:.2} ms per frame");
    assert!(h.query_by_label("W:").is_some());
}

/// An 80×60 transparent document, red over (10..30)², that square selected, Rectangular Marquee.
fn painted() -> (PhotocraftApp, photocraft_doc::LayerId) {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 80, "height": 60, "background": "transparent"})).unwrap();
    app.sync_views();
    app.ui.extras.snap = false;
    app.ui.tool = Tool::RectMarquee;
    let layer = app.session.active().unwrap().active_layer.unwrap();
    app.session
        .edit("paint", |doc, _| {
            doc.layer_mut(layer).unwrap().surface_mut().unwrap().fill_rect(Rect::new(10, 10, 30, 30), &[1.0, 0.0, 0.0, 1.0]);
            Ok(())
        })
        .unwrap();
    app.run("select.rect", json!({"x": 10, "y": 10, "width": 20, "height": 20})).unwrap();
    (app, layer)
}

fn drag(app: &mut PhotocraftApp, from: [f64; 2], to: [f64; 2], m: Modifiers) {
    use crate::canvas::{ToolEvent, tool_event};
    tool_event(app, ToolEvent::Down { x: from[0], y: from[1], pressure: 1.0 }, m);
    tool_event(app, ToolEvent::Move { x: to[0], y: to[1], pressure: 1.0 }, m);
    tool_event(app, ToolEvent::Up { x: to[0], y: to[1] }, m);
}

/// A drag inside the ants moves the outline (`select.transformSelection`); ⇧ still draws (adds);
/// a click inside deselects.
#[test]
fn drag_inside_the_selection_moves_the_outline() {
    let (mut app, layer) = painted();
    let sel = |app: &PhotocraftApp| app.session.active().unwrap().doc.selection.as_ref().map(|s| s.content_bounds());
    drag(&mut app, [20.0, 20.0], [30.0, 25.0], Modifiers::NONE);
    assert_eq!(sel(&app), Some(Rect::new(20, 15, 40, 35)));
    assert_eq!(app.session.active().unwrap().doc.layer(layer).unwrap().surface().unwrap().rgba(12, 12)[3], 1.0, "pixels stay put");
    drag(&mut app, [30.0, 20.0], [60.0, 50.0], Modifiers::SHIFT);
    assert_eq!(sel(&app), Some(Rect::new(20, 15, 60, 50)));
    drag(&mut app, [30.0, 20.0], [30.0, 20.0], Modifiers::NONE);
    assert_eq!(sel(&app), None);
}

/// ⌘-drag cuts the selected pixels into a floating piece (`select.float`): shown at the pointer
/// while dragging, moved again by plain drags, put back by Undo, dropped by Deselect.
#[test]
fn cmd_drag_floats_the_selected_pixels() {
    use crate::canvas::{ToolEvent, tool_event};
    let (mut app, layer) = painted();
    let alpha = |d: &photocraft_doc::Document, x, y| d.layer(layer).unwrap().surface().unwrap().rgba(x, y)[3];
    let doc = |app: &PhotocraftApp| app.session.active().unwrap().doc.clone();
    let offset = |app: &PhotocraftApp| photocraft_engine::float_cmds::floating(app.session.active().unwrap()).map(|f| f.offset);
    let steps = app.session.active().unwrap().history.past_len();
    tool_event(&mut app, ToolEvent::Down { x: 20.0, y: 20.0, pressure: 1.0 }, Modifiers::COMMAND);
    tool_event(&mut app, ToolEvent::Move { x: 35.0, y: 20.0, pressure: 1.0 }, Modifiers::COMMAND);
    let (shown, _) = crate::move_ui::display_doc(&mut app, 0).expect("the piece shows at the pointer while dragging");
    assert!(alpha(&shown, 12, 20) == 0.0 && alpha(&shown, 40, 20) == 1.0);
    tool_event(&mut app, ToolEvent::Up { x: 35.0, y: 20.0 }, Modifiers::COMMAND);
    assert_eq!(offset(&app), Some((15, 0)));
    assert_eq!(alpha(&doc(&app), 12, 20), 1.0, "the document waits for the drop");
    // A plain drag on the piece moves it again.
    drag(&mut app, [30.0, 20.0], [30.0, 30.0], Modifiers::NONE);
    assert_eq!(offset(&app), Some((15, 10)));
    assert_eq!(app.session.active().unwrap().history.past_len(), steps);
    // Undo puts it back.
    crate::menus::invoke(&mut app, &egui::Context::default(), "edit.undo", json!({})).unwrap();
    assert!(offset(&app).is_none() && alpha(&doc(&app), 12, 20) == 1.0);
    // Float again and deselect: dropped (one history step) where it was shown, then deselected.
    drag(&mut app, [20.0, 20.0], [35.0, 30.0], Modifiers::COMMAND);
    crate::menus::invoke(&mut app, &egui::Context::default(), "select.deselect", json!({})).unwrap();
    let d = doc(&app);
    assert!(d.selection.is_none() && offset(&app).is_none());
    assert!(alpha(&d, 12, 12) == 0.0 && alpha(&d, 26, 21) == 1.0 && alpha(&d, 44, 39) == 1.0);
    let labels: Vec<String> = app.session.active().unwrap().history.entries().into_iter().skip(steps + 1).map(|e| e.to_string()).collect();
    assert_eq!(labels.first().map(String::as_str), Some("Move"), "{labels:?}");
}

/// Through the real canvas (mouse events): a drag inside the ants moves the selection.
#[test]
fn mouse_drag_inside_the_selection_moves_it() {
    let mut h = harness(Tool::RectMarquee);
    press_at(&mut h, 100.0, 80.0, Modifiers::NONE);
    release_at(&mut h, 200.0, 160.0, Modifiers::NONE);
    assert_eq!(selection(&h), Rect::new(100, 80, 200, 160), "drawn");
    press_at(&mut h, 150.0, 120.0, Modifiers::NONE);
    release_at(&mut h, 170.0, 130.0, Modifiers::NONE);
    assert_eq!(selection(&h), Rect::new(120, 90, 220, 170), "moved by (20, 10)");
}

/// #1428: hovering inside the ants with a marquee shows the move cursor (a press there drags the
/// outline); outside it is the marquee's own cursor.
#[test]
fn hovering_inside_the_selection_shows_the_move_cursor() {
    for tool in [Tool::RectMarquee, Tool::EllipseMarquee] {
        let mut h = harness(tool);
        press_at(&mut h, 100.0, 80.0, Modifiers::NONE);
        release_at(&mut h, 200.0, 160.0, Modifiers::NONE);
        move_to(&mut h, 150.0, 120.0);
        assert_eq!(h.output().platform_output.cursor_icon, egui::CursorIcon::Move, "{tool:?} inside");
        move_to(&mut h, 300.0, 250.0);
        assert_ne!(h.output().platform_output.cursor_icon, egui::CursorIcon::Move, "{tool:?} outside");
    }
}

/// #1428: with a marquee, the arrow keys nudge the selection outline 1 px, ⇧ 10 px (Photoshop);
/// each press is one undoable step.
#[test]
fn arrow_keys_nudge_the_selection_outline() {
    use egui::Key;
    for tool in [Tool::RectMarquee, Tool::EllipseMarquee] {
        let mut h = harness(tool);
        press_at(&mut h, 100.0, 80.0, Modifiers::NONE);
        release_at(&mut h, 200.0, 160.0, Modifiers::NONE);
        let drawn = selection(&h);
        assert!(!drawn.is_empty(), "{tool:?} drew");
        let shifted = |dx: i32, dy: i32| Rect::new(drawn.x0 + dx, drawn.y0 + dy, drawn.x1 + dx, drawn.y1 + dy);
        let steps = h.state().session.active().unwrap().history.past_len();
        h.key_press(Key::ArrowRight);
        h.run_steps(1);
        assert_eq!(selection(&h), shifted(1, 0), "{tool:?} right 1 px");
        h.key_press(Key::ArrowUp);
        h.run_steps(1);
        assert_eq!(selection(&h), shifted(1, -1), "{tool:?} up 1 px");
        h.key_press_modifiers(Modifiers::SHIFT, Key::ArrowDown);
        h.run_steps(1);
        assert_eq!(selection(&h), shifted(1, 9), "{tool:?} shift-down 10 px");
        h.key_press_modifiers(Modifiers::SHIFT, Key::ArrowLeft);
        h.run_steps(1);
        assert_eq!(selection(&h), shifted(-9, 9), "{tool:?} shift-left 10 px");
        assert_eq!(h.state().session.active().unwrap().history.past_len(), steps + 4, "one step per press");
        h.state_mut().session.undo();
        assert_eq!(selection(&h), shifted(1, 9), "{tool:?} undo takes back one nudge");
    }
}

/// Arrow keys with a marquee but no selection change nothing and report no error.
#[test]
fn arrow_keys_without_a_selection_do_nothing() {
    let mut h = harness(Tool::RectMarquee);
    let steps = h.state().session.active().unwrap().history.past_len();
    h.key_press(egui::Key::ArrowRight);
    h.run_steps(1);
    assert_eq!(selection(&h), Rect::EMPTY);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), steps);
    assert!(!h.state().ui.status_error);
}

/// ⌘⌥-drag copies the selected pixels instead of cutting them: the original stays.
#[test]
fn cmd_alt_drag_floats_a_copy() {
    let (mut app, layer) = painted();
    let alpha = |d: &photocraft_doc::Document, x, y| d.layer(layer).unwrap().surface().unwrap().rgba(x, y)[3];
    let cmd_alt = Modifiers { alt: true, ..Modifiers::COMMAND };
    drag(&mut app, [20.0, 20.0], [45.0, 20.0], cmd_alt);
    app.run("select.drop", json!({})).unwrap();
    let d = app.session.active().unwrap().doc.clone();
    assert!(alpha(&d, 12, 20) == 1.0, "the original stays");
    assert!(alpha(&d, 50, 20) == 1.0, "the copy dropped 25 px right");
}

#[test]
fn alt_with_nothing_selected_draws_a_new_selection() {
    // #1106: with no selection, ⌥ has nothing to subtract from, so a marquee drawn with it held
    // still selects (Photoshop: ⌥ then only draws from the centre). With a selection it subtracts.
    let mut h = harness(Tool::RectMarquee);
    mods(&mut h, Modifiers::ALT);
    press_at(&mut h, 100.0, 100.0, Modifiers::ALT);
    move_to(&mut h, 140.0, 130.0);
    release_at(&mut h, 140.0, 130.0, Modifiers::ALT);
    mods(&mut h, Modifiers::NONE);
    assert!(h.state().session.active().unwrap().doc.selection.as_ref().is_some_and(|s| !s.content_bounds().is_empty()), "a selection was made");
    // Now ⌥ subtracts from it.
    let before = selection(&h);
    mods(&mut h, Modifiers::ALT);
    press_at(&mut h, 0.0, 0.0, Modifiers::ALT);
    move_to(&mut h, 400.0, 400.0);
    release_at(&mut h, 400.0, 400.0, Modifiers::ALT);
    mods(&mut h, Modifiers::NONE);
    let after = h.state().session.active().unwrap().doc.selection.as_ref().map(|s| s.content_bounds());
    assert!(after.is_none_or(|r| r.is_empty() || r != before), "⌥ subtracted: {before:?} -> {after:?}");
}

fn error_dialogs(app: &PhotocraftApp) -> Vec<String> {
    app.ui.dialogs.iter().filter(|d| d.kind == crate::state::DialogKind::Error).filter_map(|d| d.fields.get("message")?.as_str().map(String::from)).collect()
}

/// Photoshop's "Could not use the move tool because the layer is locked." when a Move-tool drag
/// (the Move tool, or ⌘ with a marquee) starts on a layer it can't move: the Background with no
/// selection, a position-locked layer; a plain click says nothing, and the drag never starts.
#[test]
fn move_on_the_background_without_a_selection_shows_photoshops_lock_message() {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 80, "height": 60})).unwrap();
    app.sync_views();
    app.ui.tool = Tool::Move;
    app.ui.tool_options.move_auto_select = false;
    let steps = app.session.active().unwrap().history.past_len();
    use crate::canvas::{ToolEvent, tool_event};
    // A click: no message.
    tool_event(&mut app, ToolEvent::Down { x: 20.0, y: 20.0, pressure: 1.0 }, Modifiers::NONE);
    tool_event(&mut app, ToolEvent::Up { x: 20.0, y: 20.0 }, Modifiers::NONE);
    assert!(error_dialogs(&app).is_empty() && !app.move_blocked);
    // A drag: the message once, nothing moved, no drag in progress.
    tool_event(&mut app, ToolEvent::Down { x: 20.0, y: 20.0, pressure: 1.0 }, Modifiers::NONE);
    tool_event(&mut app, ToolEvent::Move { x: 30.0, y: 20.0, pressure: 1.0 }, Modifiers::NONE);
    tool_event(&mut app, ToolEvent::Move { x: 40.0, y: 20.0, pressure: 1.0 }, Modifiers::NONE);
    tool_event(&mut app, ToolEvent::Up { x: 40.0, y: 20.0 }, Modifiers::NONE);
    assert_eq!(error_dialogs(&app), [crate::move_lock::MESSAGE]);
    assert!(app.drag.is_none(), "no drag started");
    assert_eq!(app.session.active().unwrap().history.past_len(), steps);
    app.ui.dialogs.clear();
    // ⌘-drag with a marquee (the Move tool for the drag, #896): the same.
    app.ui.tool = Tool::RectMarquee;
    drag(&mut app, [20.0, 20.0], [40.0, 20.0], Modifiers::COMMAND);
    assert_eq!(error_dialogs(&app), [crate::move_lock::MESSAGE]);
    app.ui.dialogs.clear();
    // A position-locked layer: the same; unlocked, it moves.
    app.run("layer.new.layer", json!({})).unwrap();
    app.session
        .edit("lock", |doc, a| {
            doc.layer_mut(a.unwrap()).unwrap().locks.position = true;
            Ok(())
        })
        .unwrap();
    drag(&mut app, [20.0, 20.0], [40.0, 20.0], Modifiers::COMMAND);
    assert_eq!(error_dialogs(&app).len(), 1);
    app.ui.dialogs.clear();
    app.session
        .edit("unlock", |doc, a| {
            doc.layer_mut(a.unwrap()).unwrap().locks.position = false;
            Ok(())
        })
        .unwrap();
    drag(&mut app, [20.0, 20.0], [40.0, 20.0], Modifiers::COMMAND);
    assert!(error_dialogs(&app).is_empty());
}

/// With a selection the Background's pixels do move (its lock is partial: transparency and
/// position), so no message; a layer locked all over gets it, with the marquee's ⌘-drag inside the
/// selection as with the Move tool.
#[test]
fn selected_pixels_move_off_the_background_but_not_off_a_fully_locked_layer() {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 80, "height": 60})).unwrap();
    app.sync_views();
    app.ui.extras.snap = false;
    app.ui.view.show.smart_guides = false;
    app.ui.tool = Tool::RectMarquee;
    app.run("select.rect", json!({"x": 10, "y": 10, "width": 20, "height": 20})).unwrap();
    drag(&mut app, [20.0, 20.0], [50.0, 20.0], Modifiers::COMMAND);
    assert!(error_dialogs(&app).is_empty());
    assert_eq!(photocraft_engine::float_cmds::floating(app.session.active().unwrap()).map(|f| f.offset), Some((30, 0)));
    let (mut app, layer) = painted();
    app.session
        .edit("lock", |doc, _| {
            doc.layer_mut(layer).unwrap().locks.all = true;
            Ok(())
        })
        .unwrap();
    // ⌘ with the Brush is the Move tool too (#2248).
    for (tool, m) in [(Tool::RectMarquee, Modifiers::COMMAND), (Tool::Move, Modifiers::NONE), (Tool::Brush, Modifiers::COMMAND)] {
        app.ui.tool = tool;
        drag(&mut app, [20.0, 20.0], [40.0, 20.0], m);
        assert_eq!(error_dialogs(&app), [crate::move_lock::MESSAGE], "{tool:?}");
        assert!(photocraft_engine::float_cmds::floating(app.session.active().unwrap()).is_none(), "{tool:?}");
        app.ui.dialogs.clear();
    }
}

/// Photoshop's cursor over a selection: the move cursor where a marquee press moves the outline,
/// scissors where ⌘ (or the Move tool) cuts the selected pixels, a double arrow where ⌥ copies
/// them, a hollow arrowhead while they are dragged, and a plain arrow over the floating piece.
#[test]
fn selection_cursor_follows_the_modifiers_and_the_floating_piece() {
    use crate::canvas::{SelCursor, selection_cursor};
    let (mut app, _) = painted();
    let (inside, outside) = ([20.0, 20.0], [60.0, 50.0]);
    let none = Modifiers::NONE;
    let cmd_alt = Modifiers::COMMAND | Modifiers::ALT;
    assert_eq!(selection_cursor(&app, Tool::RectMarquee, inside, none), Some(SelCursor::Outline));
    assert_eq!(selection_cursor(&app, Tool::RectMarquee, outside, none), None);
    assert_eq!(selection_cursor(&app, Tool::RectMarquee, inside, Modifiers::SHIFT), None);
    assert_eq!(selection_cursor(&app, Tool::RectMarquee, inside, Modifiers::COMMAND), Some(SelCursor::Cut));
    assert_eq!(selection_cursor(&app, Tool::RectMarquee, inside, cmd_alt), Some(SelCursor::Copy));
    assert_eq!(selection_cursor(&app, Tool::RectMarquee, inside, Modifiers::ALT), None, "⌥ alone subtracts");
    assert_eq!(selection_cursor(&app, Tool::Move, inside, none), Some(SelCursor::Cut));
    assert_eq!(selection_cursor(&app, Tool::Move, outside, none), Some(SelCursor::Cut), "the Move tool moves them from anywhere");
    assert_eq!(selection_cursor(&app, Tool::Move, inside, Modifiers::ALT), Some(SelCursor::Copy));
    assert_eq!(selection_cursor(&app, Tool::Brush, inside, none), None);
    // While the selected pixels are dragged, then once they float.
    use crate::canvas::{ToolEvent, tool_event};
    tool_event(&mut app, ToolEvent::Down { x: 20.0, y: 20.0, pressure: 1.0 }, Modifiers::COMMAND);
    tool_event(&mut app, ToolEvent::Move { x: 35.0, y: 20.0, pressure: 1.0 }, Modifiers::COMMAND);
    assert_eq!(selection_cursor(&app, Tool::RectMarquee, [35.0, 20.0], Modifiers::COMMAND), Some(SelCursor::Dragging));
    tool_event(&mut app, ToolEvent::Up { x: 35.0, y: 20.0 }, Modifiers::COMMAND);
    assert_eq!(selection_cursor(&app, Tool::RectMarquee, [35.0, 20.0], none), Some(SelCursor::Piece));
    assert_eq!(selection_cursor(&app, Tool::Move, [35.0, 20.0], none), Some(SelCursor::Piece));
    assert_eq!(selection_cursor(&app, Tool::RectMarquee, [35.0, 20.0], cmd_alt), Some(SelCursor::Copy), "a copy of the piece");
    assert_eq!(selection_cursor(&app, Tool::RectMarquee, inside, none), None, "where it was cut from");
    assert_eq!(selection_cursor(&app, Tool::Brush, [35.0, 20.0], none), None);
    // The outline drag keeps the move cursor (#1428).
    let (mut app, _) = painted();
    tool_event(&mut app, ToolEvent::Down { x: 20.0, y: 20.0, pressure: 1.0 }, none);
    tool_event(&mut app, ToolEvent::Move { x: 30.0, y: 20.0, pressure: 1.0 }, none);
    assert_eq!(selection_cursor(&app, Tool::RectMarquee, [30.0, 20.0], none), Some(SelCursor::Outline));
}

/// A click with `tool` at document point `(x, y)`, as the canvas sends it.
fn click_with(app: &mut PhotocraftApp, tool: Tool, x: f64, y: f64, m: Modifiers) {
    use crate::canvas::{ToolEvent, tool_event};
    app.ui.tool = tool;
    tool_event(app, ToolEvent::Down { x, y, pressure: 1.0 }, m);
    tool_event(app, ToolEvent::Up { x, y }, m);
}

/// Single Row / Single Column Marquee (#2837): a click selects one pixel row (column) across the
/// whole canvas; ⇧ adds, ⌥ subtracts, as with the other marquees.
#[test]
fn single_row_and_column_marquee_select_one_pixel_lines() {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("file.new", json!({"width": 200, "height": 100})).unwrap();
    app.ui.extras.snap = false;
    let bounds = |app: &PhotocraftApp| app.session.active().unwrap().doc.selection.as_ref().map_or(Rect::EMPTY, |s| s.content_bounds());
    let at = |app: &PhotocraftApp, x: i32, y: i32| app.session.active().unwrap().doc.selection.as_ref().map_or(0.0, |s| s.sample_channel(x, y, 0));

    click_with(&mut app, Tool::SingleRowMarquee, 120.3, 37.6, Modifiers::NONE);
    assert_eq!(bounds(&app), Rect::new(0, 37, 200, 38));
    click_with(&mut app, Tool::SingleRowMarquee, 15.0, 60.2, Modifiers::SHIFT);
    assert_eq!(bounds(&app), Rect::new(0, 37, 200, 61));
    assert!(at(&app, 5, 37) > 0.99 && at(&app, 199, 60) > 0.99 && at(&app, 100, 50) < 0.01, "two rows, nothing between");
    let step = |app: &PhotocraftApp| app.session.active().unwrap().history.undo_label().map(str::to_string);
    assert_eq!(step(&app).as_deref(), Some("Single Row Marquee"), "History names the step after the tool (#3031)");

    click_with(&mut app, Tool::SingleColumnMarquee, 42.9, 80.0, Modifiers::NONE);
    assert_eq!(bounds(&app), Rect::new(42, 0, 43, 100));
    assert_eq!(step(&app).as_deref(), Some("Single Column Marquee"));
    click_with(&mut app, Tool::SingleColumnMarquee, 150.5, 3.0, Modifiers::SHIFT);
    assert_eq!(bounds(&app), Rect::new(42, 0, 151, 100));
    click_with(&mut app, Tool::SingleColumnMarquee, 42.0, 10.0, Modifiers::ALT);
    assert_eq!(bounds(&app), Rect::new(150, 0, 151, 100), "⌥ subtracts the column");
}

/// The two single-line marquees sit in the Marquee flyout but M (⇧M) cycles only the
/// Rectangular and Elliptical Marquee, as in Photoshop; automation picks them by name.
#[test]
fn m_cycles_only_the_rectangular_and_elliptical_marquee() {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    let ctx = egui::Context::default();
    app.ui.tool = Tool::SingleRowMarquee;
    let mut seen = Vec::new();
    for _ in 0..4 {
        let raw = egui::RawInput {
            events: vec![
                egui::Event::Key { key: egui::Key::M, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::SHIFT },
                egui::Event::Key { key: egui::Key::M, physical_key: None, pressed: false, repeat: false, modifiers: Modifiers::SHIFT },
            ],
            ..Default::default()
        };
        ctx.begin_pass(raw);
        crate::shortcuts::handle(&mut app, &ctx);
        ctx.end_pass().textures_delta.clear();
        seen.push(app.ui.tool);
    }
    assert_eq!(seen, [Tool::RectMarquee, Tool::EllipseMarquee, Tool::RectMarquee, Tool::EllipseMarquee]);
    assert_eq!(Tool::from_name("SingleRowMarquee"), Some(Tool::SingleRowMarquee));
    assert_eq!(Tool::from_name("Single Column Marquee Tool"), Some(Tool::SingleColumnMarquee));
}

#[test]
fn size_readout_follows_the_ruler_unit() {
    use photocraft_engine::prefs::{Unit, UnitsAndRulers};
    let mut ur = UnitsAndRulers::default();
    // 1644 × 2659 px at 72 dpi: 58 cm wide
    let r = [0.0, 0.0, 1644.0, 2659.0];
    ur.rulers = Unit::Pixels;
    assert_eq!(crate::canvas::readout_in(&ur, 72.0, [1644.0, 2659.0], r), ["1644 px".to_string(), "2659 px".to_string()]);
    ur.rulers = Unit::Centimeters;
    assert_eq!(crate::canvas::readout_in(&ur, 72.0, [1644.0, 2659.0], r), ["58 cm".to_string(), "93.8 cm".to_string()]);
    ur.rulers = Unit::Millimeters;
    assert_eq!(crate::canvas::readout_in(&ur, 150.0, [1644.0, 2659.0], [0.0, 0.0, 3425.0, 100.0])[0], "579.97 mm");
}
