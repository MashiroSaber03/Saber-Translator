//! Preferences › Tools › Pressure Curve: the pen's pressure response, edited as points on a curve
//! (`tools.pressureCurve`, read by the stylus through `photocraft_engine::prefs::PressureCurve`).
//! Drag a point to move it, click empty space to add one, right-click a point to remove it.

use egui::{Pos2, Sense, Stroke, Vec2, pos2};
use photocraft_engine::prefs::PressureCurve;
use serde_json::{Value, json};

use crate::theme::Tokens;

/// Side of the editing square, in points.
const SIZE: f32 = 140.0;
/// egui memory key holding the editing square's rectangle.
pub(crate) const RECT_KEY: &str = "pressure-curve-rect";
/// How close (in points) a press must land to grab a point.
const GRAB: f32 = 8.0;

/// Preset curves: Linear, Soft (light pressure already gives a lot) and Hard (it takes a firm
/// press to reach full pressure).
const PRESETS: [(&str, &[[f32; 2]]); 3] =
    [("Linear", &[[0.0, 0.0], [1.0, 1.0]]), ("Soft", &[[0.0, 0.0], [0.3, 0.6], [1.0, 1.0]]), ("Hard", &[[0.0, 0.0], [0.6, 0.3], [1.0, 1.0]])];

/// The points a preferences value holds (anything malformed is skipped; `PressureCurve` sanitises
/// the rest).
pub(crate) fn points_of(v: &Value) -> Vec<[f32; 2]> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|p| {
                    let p = p.as_array()?;
                    Some([p.first()?.as_f64()? as f32, p.get(1)?.as_f64()? as f32])
                })
                .collect()
        })
        .unwrap_or_default()
}

fn value_of(points: &[[f32; 2]]) -> Value {
    // Two decimals: the stored file stays readable and a drag doesn't write float noise.
    json!(points.iter().map(|p| [round2(p[0]), round2(p[1])]).collect::<Vec<_>>())
}

fn round2(x: f32) -> f64 {
    (f64::from(x.clamp(0.0, 1.0)) * 100.0).round() / 100.0
}

/// The editor. Returns the new value when the curve changed.
pub(crate) fn editor(ui: &mut egui::Ui, value: &Value) -> Option<Value> {
    let t = Tokens::get(ui.ctx());
    let mut points = points_of(value);
    if points.len() < 2 {
        points = vec![[0.0, 0.0], [1.0, 1.0]];
    }
    let mut changed = false;
    // The square with the presets beside it, so the control fits the preferences page's height.
    ui.horizontal_top(|ui| {
        let (rect, resp) = ui.allocate_exact_size(Vec2::splat(SIZE), Sense::click_and_drag());
        // Where the square is this frame (tests and automation point at curve coordinates).
        ui.data_mut(|d| d.insert_temp(egui::Id::new(RECT_KEY), rect));
        let to_screen = |p: [f32; 2]| pos2(rect.left() + p[0] * rect.width(), rect.bottom() - p[1] * rect.height());
        let to_curve = |s: Pos2| [((s.x - rect.left()) / rect.width()).clamp(0.0, 1.0), ((rect.bottom() - s.y) / rect.height()).clamp(0.0, 1.0)];
        let painter = ui.painter_at(rect.expand(4.0));
        painter.rect_filled(rect, 2.0, t.field);
        for i in 1..4 {
            let f = i as f32 / 4.0;
            let grid = Stroke::new(1.0, t.field_border);
            painter.line_segment([pos2(rect.left() + f * rect.width(), rect.top()), pos2(rect.left() + f * rect.width(), rect.bottom())], grid);
            painter.line_segment([pos2(rect.left(), rect.top() + f * rect.height()), pos2(rect.right(), rect.top() + f * rect.height())], grid);
        }
        painter.rect_stroke(rect, 2.0, Stroke::new(1.0, t.field_border), egui::StrokeKind::Inside);
        painter.line_segment([rect.left_bottom(), rect.right_top()], Stroke::new(1.0, t.text_dim.gamma_multiply(0.35)));

        // The point under a press is dragged until release (kept in the egui memory).
        let drag_id = resp.id.with("dragging");
        let nearest = |at: Pos2, pts: &[[f32; 2]]| {
            pts.iter().enumerate().map(|(i, p)| (i, to_screen(*p).distance(at))).filter(|(_, d)| *d <= GRAB).min_by(|a, b| a.1.total_cmp(&b.1)).map(|(i, _)| i)
        };
        // The point under the press, not under the pointer when the drag is recognised (by then
        // it has moved past egui's drag threshold, off the point).
        if resp.drag_started()
            && let Some(at) = ui.input(|i| i.pointer.press_origin()).or(resp.interact_pointer_pos())
        {
            let grabbed = nearest(at, &points);
            ui.data_mut(|d| d.insert_temp(drag_id, grabbed));
        }
        let dragging: Option<usize> = ui.data(|d| d.get_temp(drag_id)).flatten();
        if let (Some(i), Some(at)) = (dragging, resp.interact_pointer_pos().filter(|_| resp.dragged()))
            && let Some(p) = points.get_mut(i)
        {
            *p = to_curve(at);
            changed = true;
        }
        if resp.drag_stopped() {
            ui.data_mut(|d| d.insert_temp::<Option<usize>>(drag_id, None));
        }
        if resp.clicked()
            && let Some(at) = resp.interact_pointer_pos()
            && nearest(at, &points).is_none()
            && points.len() < PressureCurve::MAX_POINTS
        {
            points.push(to_curve(at));
            changed = true;
        }
        if resp.secondary_clicked()
            && let Some(at) = resp.interact_pointer_pos()
            && let Some(i) = nearest(at, &points)
            && points.len() > 2
        {
            points.remove(i);
            changed = true;
        }

        let curve = PressureCurve::new(&points);
        let line: Vec<Pos2> = (0..=64).map(|i| i as f32 / 64.0).map(|x| to_screen([x, curve.eval(x)])).collect();
        painter.add(egui::Shape::line(line, Stroke::new(2.0, t.accent)));
        for (i, p) in points.iter().enumerate() {
            let active = dragging == Some(i);
            painter.circle(to_screen(*p), if active { 5.0 } else { 4.0 }, if active { t.accent } else { t.field }, Stroke::new(1.5, t.accent));
        }
        resp.on_hover_text(tl!("Click to add a point, right-click to remove"));

        ui.vertical(|ui| {
            for (name, preset) in PRESETS {
                if ui.small_button(tl!(name)).clicked() {
                    points = preset.to_vec();
                    changed = true;
                }
            }
        });
    });
    changed.then(|| value_of(&points))
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui_kittest::Harness;
    use egui_kittest::kittest::Queryable;

    fn harness(start: Value) -> Harness<'static, Value> {
        let mut h = Harness::builder().with_size(egui::vec2(300.0, 260.0)).build_ui_state(
            |ui, v: &mut Value| {
                if let Some(next) = editor(ui, v) {
                    *v = next;
                }
            },
            start,
        );
        h.run_steps(2);
        h
    }

    /// The screen position of curve point `p` (the editor's square starts at the panel's origin).
    fn at(h: &Harness<'static, Value>, p: [f32; 2]) -> Pos2 {
        let r: egui::Rect = h.ctx.data(|d| d.get_temp(egui::Id::new(RECT_KEY))).expect("the editor ran");
        pos2(r.left() + p[0] * r.width(), r.bottom() - p[1] * r.height())
    }

    fn press(h: &mut Harness<'static, Value>, p: Pos2, button: egui::PointerButton, release: Pos2) {
        h.event(egui::Event::PointerMoved(p));
        h.run_steps(1);
        h.event(egui::Event::PointerButton { pos: p, button, pressed: true, modifiers: Default::default() });
        h.run_steps(1);
        if release != p {
            h.event(egui::Event::PointerMoved(pos2((p.x + release.x) / 2.0, (p.y + release.y) / 2.0)));
            h.run_steps(1);
            h.event(egui::Event::PointerMoved(release));
            h.run_steps(1);
        }
        h.event(egui::Event::PointerButton { pos: release, button, pressed: false, modifiers: Default::default() });
        h.run_steps(2);
    }

    #[test]
    fn click_adds_right_click_removes_drag_moves_and_presets_replace() {
        let mut h = harness(json!([[0.0, 0.0], [1.0, 1.0]]));
        // A click on empty space adds a point there.
        let p = at(&h, [0.5, 0.2]);
        press(&mut h, p, egui::PointerButton::Primary, p);
        let pts = points_of(h.state());
        assert_eq!(pts.len(), 3, "{pts:?}");
        assert!(pts.iter().any(|q| (q[0] - 0.5).abs() < 0.03 && (q[1] - 0.2).abs() < 0.03), "{pts:?}");
        // Dragging it moves it.
        let (from, to) = (at(&h, [0.5, 0.2]), at(&h, [0.4, 0.7]));
        press(&mut h, from, egui::PointerButton::Primary, to);
        let pts = points_of(h.state());
        assert!(pts.iter().any(|q| (q[0] - 0.4).abs() < 0.03 && (q[1] - 0.7).abs() < 0.03), "{pts:?}");
        // A right-click on it removes it; the last two points can't be removed.
        let p = at(&h, [0.4, 0.7]);
        press(&mut h, p, egui::PointerButton::Secondary, p);
        assert_eq!(points_of(h.state()).len(), 2);
        let p = at(&h, [1.0, 1.0]);
        press(&mut h, p, egui::PointerButton::Secondary, p);
        assert_eq!(points_of(h.state()).len(), 2, "a curve keeps at least two points");
        // A preset replaces the curve.
        h.get_by_label("Soft").click();
        h.run_steps(2);
        assert_eq!(points_of(h.state()), vec![[0.0, 0.0], [0.3, 0.6], [1.0, 1.0]]);
        h.get_by_label("Linear").click();
        h.run_steps(2);
        assert_eq!(points_of(h.state()), vec![[0.0, 0.0], [1.0, 1.0]]);
    }

    #[test]
    fn malformed_stored_values_still_edit() {
        let mut h = harness(json!([[0.5], "x", [2.0, -1.0]]));
        h.get_by_label("Hard").click();
        h.run_steps(2);
        assert_eq!(points_of(h.state()), vec![[0.0, 0.0], [0.6, 0.3], [1.0, 1.0]]);
        assert!(points_of(&json!("nope")).is_empty());
    }
}
