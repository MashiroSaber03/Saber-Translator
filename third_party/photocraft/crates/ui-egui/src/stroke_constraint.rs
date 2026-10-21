//! ⇧ with painting tools and the Gradient tool (#178), as in Photoshop:
//!
//! - ⇧ while dragging a painting stroke keeps it on a straight line at 0°, 45° or 90° from where
//!   the constraint began (the stroke's start, or the point where ⇧ went down mid-stroke). The
//!   direction locks once the pointer has moved a few screen points.
//! - ⇧-click after a stroke paints a straight line from the end of the previous stroke to the
//!   click. The line is part of the new stroke, so it is one `paint.stroke` and one undo step.
//!   The endpoint remembers pen pressure so size and opacity interpolate between the strokes.
//! - The Gradient tool's ⇧ snaps the gradient angle to 45° increments.
//!
//! Points go through the normal drag/`LiveStroke`/commit path, so the live preview, pen dynamics
//! and Smoothing treat them like any other stroke points.

use crate::state::Tool;

/// Screen points the pointer must travel before a ⇧-constrained stroke locks its direction.
const LOCK_DISTANCE: f64 = 3.0;

/// A ⇧ constraint in progress: its origin and, once locked, the unit direction of the line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Axis {
    pub origin: [f64; 2],
    pub dir: Option<[f64; 2]>,
}

/// Tools whose drag ⇧ constrains: every painting tool and the Gradient tool.
pub fn applies(tool: Tool) -> bool {
    tool.is_brushlike() || tool == Tool::Gradient
}

/// Tools that ⇧-click connects to the previous stroke with a straight line.
pub fn connects(tool: Tool) -> bool {
    tool.is_brushlike()
}

/// The unit direction from `origin` towards `p`, snapped to the nearest multiple of `step`
/// radians; None when the points coincide (or aren't finite).
pub fn snapped_dir(origin: [f64; 2], p: [f64; 2], step: f64) -> Option<[f64; 2]> {
    let (dx, dy) = (p[0] - origin[0], p[1] - origin[1]);
    if !(dx.is_finite() && dy.is_finite()) || dx.hypot(dy) < 1e-9 || step <= 0.0 {
        return None;
    }
    let a = (dy.atan2(dx) / step).round() * step;
    Some([a.cos(), a.sin()])
}

/// `p` projected onto the line through `origin` along unit `dir`.
pub fn project(origin: [f64; 2], dir: [f64; 2], p: [f64; 2]) -> [f64; 2] {
    let t = (p[0] - origin[0]) * dir[0] + (p[1] - origin[1]) * dir[1];
    [origin[0] + dir[0] * t, origin[1] + dir[1] * t]
}

/// `p` moved onto the nearest 45° ray from `origin` (the Gradient tool's ⇧).
pub fn snap45(origin: [f64; 2], p: [f64; 2]) -> [f64; 2] {
    match snapped_dir(origin, p, std::f64::consts::FRAC_PI_4) {
        Some(d) => project(origin, d, p),
        None => p,
    }
}

/// Constrain the pointer at `p` for a drag of `tool` that started at `start` and has reached
/// `last`. `axis` carries the stroke's lock between events; `zoom` converts the lock distance
/// to document pixels. Without ⇧ the point passes through and the lock is dropped.
pub fn constrain(tool: Tool, axis: &mut Option<Axis>, start: [f64; 2], last: [f64; 2], p: [f64; 2], shift: bool, zoom: f32) -> [f64; 2] {
    if !shift || !applies(tool) {
        *axis = None;
        return p;
    }
    if tool == Tool::Gradient {
        return snap45(start, p);
    }
    let a = axis.get_or_insert(Axis { origin: last, dir: None });
    if a.dir.is_none() {
        let reach = LOCK_DISTANCE / f64::from(zoom.max(1e-3));
        if (p[0] - a.origin[0]).hypot(p[1] - a.origin[1]) < reach {
            return a.origin;
        }
        a.dir = snapped_dir(a.origin, p, std::f64::consts::FRAC_PI_4);
    }
    match a.dir {
        Some(d) => project(a.origin, d, p),
        None => a.origin,
    }
}

/// Where the ⇧-click line preview starts (#257): the end of the last stroke on the active
/// document, while ⇧ is held with a painting tool and nothing is being painted.
pub fn line_preview_start(app: &crate::PhotocraftApp, tool: Tool, shift: bool) -> Option<[f64; 2]> {
    if !shift || !connects(tool) || app.drag.is_some() {
        return None;
    }
    let active = app.session.active()?.doc.id;
    app.last_stroke_end.filter(|(doc, _)| *doc == active).map(|(_, p)| [p[0], p[1]])
}

/// Photoshop's rubber band: while ⇧ is held after a stroke, a thin line from where that stroke
/// ended to the pointer shows the line a click will paint, with the brush footprint at the
/// pointer (drawn here only when the painting cursor doesn't already show the tip). Two line
/// segments and at most one circle: no measurable cost per frame. Returns the drawn segment.
pub fn draw_line_preview(
    app: &crate::PhotocraftApp,
    painter: &egui::Painter,
    xf: &crate::canvas::ViewXform,
    pointer: egui::Pos2,
    tool: Tool,
    shift: bool,
) -> Option<[egui::Pos2; 2]> {
    use egui::{Color32, Stroke};
    let a = line_preview_start(app, tool, shift)?;
    let a = xf.to_screen(a[0] as f32, a[1] as f32);
    if !(a.x.is_finite() && a.y.is_finite()) {
        return None;
    }
    for (w, c) in [(2.5, Color32::from_black_alpha(120)), (1.0, Color32::from_white_alpha(230))] {
        painter.line_segment([a, pointer], Stroke::new(w, c));
    }
    use photocraft_engine::prefs::PaintingCursor;
    if matches!(app.session.prefs().cursors.painting, PaintingCursor::Standard | PaintingCursor::Precise) {
        let r = (app.session.tools.brush.size / 2.0 * xf.zoom).max(1.0);
        painter.circle_stroke(pointer, r + 0.5, Stroke::new(1.0, Color32::from_black_alpha(140)));
        painter.circle_stroke(pointer, r, Stroke::new(1.0, Color32::from_white_alpha(220)));
    }
    Some([a, pointer])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PhotocraftApp;
    use crate::canvas::{ToolEvent, tool_event};
    use serde_json::json;

    fn close(a: [f64; 2], b: [f64; 2]) -> bool {
        (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6
    }

    #[test]
    fn snapping_angles() {
        let o = [10.0, 10.0];
        // Gradient: 45° increments.
        assert!(close(snap45(o, [50.0, 13.0]), [50.0, 10.0]));
        assert!(close(snap45(o, [12.0, -40.0]), [10.0, -40.0]));
        let d = snap45(o, [40.0, 37.0]);
        assert!((d[0] - d[1]).abs() < 1e-9 && d[0] > 30.0, "45°: {d:?}");
        assert!(close(snap45(o, o), o));
        assert!(snap45(o, [f64::NAN, 1.0])[0].is_nan(), "non-finite input passes through");
        // Painting: the direction locks once the pointer has moved a few points.
        let mut axis = None;
        let p = |q, ax: &mut Option<Axis>| constrain(Tool::Brush, ax, o, o, q, true, 1.0);
        assert!(close(p([11.0, 11.0], &mut axis), o), "inside the lock distance");
        assert!(close(p([30.0, 12.0], &mut axis), [30.0, 10.0]), "horizontal");
        // Locked: wandering towards 45° stays on the horizontal line.
        assert!(close(p([60.0, 45.0], &mut axis), [60.0, 10.0]));
        // Releasing ⇧ frees the stroke and drops the lock.
        assert!(close(constrain(Tool::Brush, &mut axis, o, o, [70.0, 50.0], false, 1.0), [70.0, 50.0]) && axis.is_none());
        // Vertical and diagonal locks.
        let mut axis = None;
        assert!(close(p([9.0, 60.0], &mut axis), [10.0, 60.0]));
        let mut axis = None;
        let q = p([40.0, 43.0], &mut axis);
        assert!((q[0] - 10.0 - (q[1] - 10.0)).abs() < 1e-9);
        // Other tools pass through.
        assert!(close(constrain(Tool::Move, &mut None, o, o, [30.0, 12.0], true, 1.0), [30.0, 12.0]));
    }

    fn app() -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 120, "height": 120, "background": "transparent"})).unwrap();
        app.run("tools.setBrush", json!({"brush": {"size": 4, "hardness": 1.0}})).unwrap();
        app.session.tools.brush.smoothing.amount = 0.0;
        app
    }

    fn last_stroke_points(app: &PhotocraftApp) -> Vec<[f64; 2]> {
        let p = app.session.journal.iter().rev().find(|(id, _)| id == "paint.stroke").map(|(_, p)| p.clone()).unwrap();
        p["points"].as_array().unwrap().iter().map(|q| [q[0].as_f64().unwrap(), q[1].as_f64().unwrap()]).collect()
    }

    fn alpha(app: &PhotocraftApp, x: i32, y: i32) -> f32 {
        app.session.documents()[0].doc.layers[0].surface().unwrap().rgba(x, y)[3]
    }

    #[test]
    fn shift_drag_paints_a_straight_constrained_stroke() {
        for tool in [Tool::Brush, Tool::Eraser] {
            let mut app = app();
            if tool == Tool::Eraser {
                app.run("edit.fill", json!({"contents": "foreground"})).unwrap();
            }
            app.ui.tool = tool;
            let shift = egui::Modifiers::SHIFT;
            tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 60.0, pressure: 1.0 }, shift);
            for (x, y) in [(30.0, 63.0), (50.0, 55.0), (80.0, 68.0), (100.0, 52.0)] {
                tool_event(&mut app, ToolEvent::Move { x, y, pressure: 1.0 }, shift);
            }
            tool_event(&mut app, ToolEvent::Up { x: 105.0, y: 70.0 }, shift);
            let pts = last_stroke_points(&app);
            assert!(pts.len() >= 2 && pts.iter().all(|p| p[1] == 60.0), "{tool:?}: {pts:?}");
            assert_eq!(pts.last().unwrap()[0], 105.0);
            // The pixels follow the line: painted (or erased) along y = 60, untouched at y = 68.
            let on = alpha(&app, 70, 60);
            let off = alpha(&app, 80, 68);
            if tool == Tool::Brush {
                assert!(on > 0.9 && off == 0.0, "{on} {off}");
            } else {
                assert!(on < 0.1 && off > 0.9, "{on} {off}");
            }
            assert_eq!(app.session.documents()[0].history.past_len(), 1 + usize::from(tool == Tool::Eraser));
        }
    }

    #[test]
    fn shift_click_draws_a_line_from_the_last_stroke_in_one_step() {
        let mut app = app();
        app.ui.tool = Tool::Brush;
        let none = egui::Modifiers::NONE;
        // A first (free) stroke ending at (20, 20).
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 10.0, pressure: 1.0 }, none);
        tool_event(&mut app, ToolEvent::Move { x: 20.0, y: 20.0, pressure: 1.0 }, none);
        tool_event(&mut app, ToolEvent::Up { x: 20.0, y: 20.0 }, none);
        let before = app.session.documents()[0].history.past_len();
        // ⇧-click at (100, 100): a straight line from (20, 20).
        let shift = egui::Modifiers::SHIFT;
        tool_event(&mut app, ToolEvent::Down { x: 100.0, y: 100.0, pressure: 1.0 }, shift);
        tool_event(&mut app, ToolEvent::Up { x: 100.0, y: 100.0 }, shift);
        assert_eq!(last_stroke_points(&app), vec![[20.0, 20.0], [100.0, 100.0]]);
        assert_eq!(app.session.documents()[0].history.past_len(), before + 1, "one undo step per line");
        assert!(alpha(&app, 60, 60) > 0.9 && alpha(&app, 60, 70) == 0.0);
        // And the next ⇧-click continues from there.
        tool_event(&mut app, ToolEvent::Down { x: 100.0, y: 20.0, pressure: 1.0 }, shift);
        tool_event(&mut app, ToolEvent::Up { x: 100.0, y: 20.0 }, shift);
        assert_eq!(last_stroke_points(&app), vec![[100.0, 100.0], [100.0, 20.0]]);
        assert!(alpha(&app, 100, 60) > 0.9);
        // Undo removes just the last line.
        app.session.undo();
        assert!(alpha(&app, 100, 60) == 0.0 && alpha(&app, 60, 60) > 0.9);
        // The line start survives the undo (as in Photoshop). With smoothing on, the line still
        // runs from (100, 20) exactly to the click.
        app.session.tools.brush.smoothing.amount = 0.6;
        tool_event(&mut app, ToolEvent::Down { x: 20.0, y: 100.0, pressure: 1.0 }, shift);
        tool_event(&mut app, ToolEvent::Up { x: 20.0, y: 100.0 }, shift);
        assert!(alpha(&app, 20, 100) > 0.9 && alpha(&app, 40, 80) > 0.9 && alpha(&app, 90, 30) > 0.9);
    }

    #[test]
    fn shift_click_interpolates_from_previous_strokes_endpoint_pressure() {
        let mut app = app();
        app.ui.tool = Tool::Brush;
        app.session.tools.brush.pressure_size = true;
        app.session.tools.brush.pressure_opacity = true;
        let none = egui::Modifiers::NONE;
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 10.0, pressure: 0.2 }, none);
        tool_event(&mut app, ToolEvent::Move { x: 20.0, y: 20.0, pressure: 0.35 }, none);
        tool_event(&mut app, ToolEvent::Up { x: 20.0, y: 20.0 }, none);
        let before = app.session.documents()[0].history.past_len();

        let shift = egui::Modifiers::SHIFT;
        tool_event(&mut app, ToolEvent::Down { x: 100.0, y: 100.0, pressure: 0.85 }, shift);
        tool_event(&mut app, ToolEvent::Up { x: 100.0, y: 100.0 }, shift);
        let (_, stroke) = app.session.journal.iter().rev().find(|(id, _)| id == "paint.stroke").expect("connecting stroke committed");
        let points = stroke["points"].as_array().expect("stroke points array");
        assert_eq!(points.len(), 2, "one connecting segment, one stroke");
        assert_eq!(last_stroke_points(&app), vec![[20.0, 20.0], [100.0, 100.0]]);
        let pressure = |i: usize| points[i][2].as_f64().expect("pen pressure in stroke point");
        assert!((pressure(0) - 0.35).abs() < 1e-6, "line starts at previous pen pressure, not the new click's");
        assert!((pressure(1) - 0.85).abs() < 1e-6, "line ends at the new click's pen pressure");
        assert_eq!(app.session.documents()[0].history.past_len(), before + 1, "connecting line is one undo step");

        // Another Shift-click uses the just-completed endpoint, not the first stroke's.
        tool_event(&mut app, ToolEvent::Down { x: 80.0, y: 30.0, pressure: 0.55 }, shift);
        tool_event(&mut app, ToolEvent::Up { x: 80.0, y: 30.0 }, shift);
        let (_, next) = app.session.journal.iter().rev().find(|(id, _)| id == "paint.stroke").expect("next connecting stroke");
        let next = next["points"].as_array().expect("next stroke points");
        assert!((next[0][2].as_f64().unwrap() - 0.85).abs() < 1e-6);
        assert!((next[1][2].as_f64().unwrap() - 0.55).abs() < 1e-6);
    }

    #[test]
    fn shift_click_without_a_previous_stroke_is_a_dab() {
        let mut app = app();
        app.ui.tool = Tool::Brush;
        let shift = egui::Modifiers::SHIFT;
        tool_event(&mut app, ToolEvent::Down { x: 50.0, y: 50.0, pressure: 1.0 }, shift);
        tool_event(&mut app, ToolEvent::Up { x: 50.0, y: 50.0 }, shift);
        assert_eq!(last_stroke_points(&app), vec![[50.0, 50.0]]);
        // A new document doesn't connect to another document's stroke.
        app.run("file.new", json!({"width": 60, "height": 60, "background": "transparent"})).unwrap();
        tool_event(&mut app, ToolEvent::Down { x: 5.0, y: 5.0, pressure: 1.0 }, shift);
        tool_event(&mut app, ToolEvent::Up { x: 5.0, y: 5.0 }, shift);
        assert_eq!(last_stroke_points(&app), vec![[5.0, 5.0]]);
    }

    #[test]
    fn shift_gradient_snaps_to_45_degrees() {
        let mut app = app();
        app.ui.tool = Tool::Gradient;
        // The destructive drag (live gradients snap in gradient_ui.rs, with the same helper).
        app.ui.tool_options.gradient_classic = true;
        let shift = egui::Modifiers::SHIFT;
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 10.0, pressure: 1.0 }, shift);
        tool_event(&mut app, ToolEvent::Move { x: 60.0, y: 52.0, pressure: 1.0 }, shift);
        tool_event(&mut app, ToolEvent::Up { x: 90.0, y: 85.0 }, shift);
        let p = app.session.journal.iter().rev().find(|(id, _)| id == "paint.gradient").map(|(_, p)| p.clone()).unwrap();
        let to = [p["to"][0].as_f64().unwrap(), p["to"][1].as_f64().unwrap()];
        assert!((to[0] - to[1]).abs() < 1e-9 && to[0] > 80.0, "45°: {to:?}");
        // Horizontal.
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 10.0, pressure: 1.0 }, shift);
        tool_event(&mut app, ToolEvent::Up { x: 90.0, y: 22.0 }, shift);
        let p = app.session.journal.iter().rev().find(|(id, _)| id == "paint.gradient").map(|(_, p)| p.clone()).unwrap();
        assert_eq!(p["to"], json!([90.0, 10.0]));
    }

    #[test]
    fn line_preview_needs_shift_a_previous_stroke_and_no_stroke_in_progress() {
        let mut app = app();
        app.ui.tool = Tool::Brush;
        assert_eq!(line_preview_start(&app, Tool::Brush, true), None, "no previous stroke");
        let none = egui::Modifiers::NONE;
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 10.0, pressure: 1.0 }, none);
        tool_event(&mut app, ToolEvent::Move { x: 30.0, y: 20.0, pressure: 1.0 }, none);
        assert_eq!(line_preview_start(&app, Tool::Brush, true), None, "while painting");
        tool_event(&mut app, ToolEvent::Up { x: 30.0, y: 20.0 }, none);
        assert_eq!(line_preview_start(&app, Tool::Brush, true), Some([30.0, 20.0]));
        assert_eq!(line_preview_start(&app, Tool::Brush, false), None, "⇧ released");
        assert_eq!(line_preview_start(&app, Tool::Eraser, true), Some([30.0, 20.0]), "any painting tool");
        assert_eq!(line_preview_start(&app, Tool::Move, true), None);
        // Another document has no line start.
        app.run("file.new", json!({"width": 60, "height": 60})).unwrap();
        assert_eq!(line_preview_start(&app, Tool::Brush, true), None);
    }

    /// Every line segment the last frame painted.
    fn segments(h: &egui_kittest::Harness<'static, PhotocraftApp>) -> Vec<[egui::Pos2; 2]> {
        fn walk(s: &egui::Shape, out: &mut Vec<[egui::Pos2; 2]>) {
            match s {
                egui::Shape::LineSegment { points, .. } => out.push(*points),
                egui::Shape::Vec(v) => v.iter().for_each(|s| walk(s, out)),
                _ => {}
            }
        }
        let mut out = Vec::new();
        for c in &h.output().shapes {
            walk(&c.shape, &mut out);
        }
        out
    }

    #[test]
    fn shift_shows_a_rubber_band_from_the_last_stroke_to_the_pointer() {
        use egui::{Event, Modifiers, PointerButton, vec2};
        let mut a = app();
        a.ui.tool = Tool::Brush;
        a.sync_views();
        let mut h = egui_kittest::Harness::builder().with_size(vec2(1000.0, 700.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                let ctx = ui.ctx().clone();
                if !ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    return;
                }
                egui::CentralPanel::default().show(ui, |ui| crate::canvas::document_area(app, ui));
            },
            a,
        );
        crate::PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::default());
        h.run_steps(4);
        let c = h.state().last_canvas_rect.center();
        let target = c + vec2(120.0, 90.0);
        let near = |p: egui::Pos2, q: egui::Pos2| p.distance(q) < 1.5;
        let band = |h: &egui_kittest::Harness<'static, PhotocraftApp>, from: egui::Pos2| segments(h).iter().any(|s| near(s[0], from) && near(s[1], target));
        // ⇧ with no previous stroke: nothing.
        h.event(Event::ModifiersChanged(Modifiers::SHIFT));
        h.event(Event::PointerMoved(target));
        h.run_steps(2);
        assert!(!segments(&h).iter().any(|s| near(s[1], target) && s[0].distance(s[1]) > 20.0), "no band without a previous stroke");
        // A stroke ending at c.
        h.event(Event::ModifiersChanged(Modifiers::NONE));
        let start = c - vec2(60.0, 0.0);
        h.event(Event::PointerMoved(start));
        h.run();
        h.event(Event::PointerButton { pos: start, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
        h.run();
        for k in 1..=6 {
            h.event(Event::PointerMoved(start + vec2(10.0 * k as f32, 0.0)));
            h.run();
        }
        h.event(Event::PointerButton { pos: c, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
        h.run_steps(2);
        let end = h.state().last_stroke_end.map(|(_, p)| p).expect("stroke recorded");
        let xf = {
            let st = h.state();
            let v = st.ui.views[st.session.active_index().unwrap()].clone();
            crate::canvas::ViewXform { rect: st.last_canvas_rect, zoom: v.zoom, center: v.center, flip: false, rotation: v.rotation, aspect: 1.0 }
        };
        let from = xf.to_screen(end[0] as f32, end[1] as f32);
        h.event(Event::PointerMoved(target));
        h.run_steps(2);
        assert!(!band(&h, from), "no band without ⇧");
        h.event(Event::ModifiersChanged(Modifiers::SHIFT));
        h.run_steps(2);
        assert!(band(&h, from), "⇧ shows the line from the last stroke's end to the pointer");
        h.event(Event::ModifiersChanged(Modifiers::NONE));
        h.run_steps(2);
        assert!(!band(&h, from), "releasing ⇧ hides it");
        // Previewing never edits: no stroke, no undo step.
        let n = h.state().session.journal.iter().filter(|(id, _)| id == "paint.stroke").count();
        assert_eq!(n, 1);
    }
}
