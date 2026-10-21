//! The Perspective Crop tool (C, in the Crop flyout), as in Photoshop:
//!
//! - drag a box, or click four corners, to make a quad (`UiState::perspective_crop`, TL, TR, BR,
//!   BL in document px; clicked corners are put in that order);
//! - drag a corner to move it alone, onto an edge of a skewed object (a photographed page, a
//!   facade); drag inside the quad to move it;
//! - the quad shows its perspective grid (Show Grid) and turns red while it isn't a convex quad
//!   wound like its box, which can't be committed;
//! - ↵ (or ✓) commits: `image.perspectiveCrop` maps the quad onto an upright rectangle, the whole
//!   document in one undo step. Its size comes from the options bar's W x H (and the resolution),
//!   which are the Crop tool's fields (`crop_size`), or with them empty from the quad's mean side
//!   lengths. Esc (or ⊘) cancels.
//!
//! A pending quad belongs to the document it was drawn on, like the Crop tool's frame.

use egui::{Color32, Pos2, Stroke};
use photocraft_algo::transform::{Homography, convex_quad};
use photocraft_engine::prefs::Unit;
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::canvas::{ToolEvent, ViewXform};
use crate::state::Tool;

/// A drag shorter than this (screen px) is a click: it places a corner.
const CLICK_PX: f64 = 3.0;

#[derive(Clone, Copy, Debug)]
pub enum Drag {
    /// Drawing a box from `start`.
    New { start: [f64; 2], cur: [f64; 2] },
    /// Moving one corner; `offset` is the press's distance from it.
    Corner { index: usize, offset: [f64; 2] },
    /// Moving the whole quad.
    Move { start: [f64; 2], quad: [[f64; 2]; 4] },
}

#[derive(Default)]
pub struct PerspectiveCrop {
    pub drag: Option<Drag>,
    /// Corners clicked so far, before the fourth makes the quad.
    pub clicks: Vec<[f64; 2]>,
    /// The document the pending quad belongs to.
    doc: Option<photocraft_doc::DocId>,
}

fn active(app: &PhotocraftApp) -> bool {
    app.ui.tool == Tool::PerspectiveCrop && app.session.active().is_some()
}

/// A quad or corners clicked or a drag in progress.
pub fn pending(app: &PhotocraftApp) -> bool {
    app.ui.perspective_crop.is_some() || !app.pcrop.clicks.is_empty() || app.pcrop.drag.is_some()
}

pub fn cancel(app: &mut PhotocraftApp) {
    app.ui.perspective_crop = None;
    app.pcrop.drag = None;
    app.pcrop.clicks.clear();
}

/// Another document became active: its pending quad is dropped.
fn cancel_stale(app: &mut PhotocraftApp) {
    let doc = app.session.active().map(|st| st.doc.id);
    if pending(app) && app.pcrop.doc.is_some() && app.pcrop.doc != doc {
        cancel(app);
    }
    app.pcrop.doc = doc;
}

/// Four points in quad order: clockwise on screen around their centre, starting top-left.
pub fn ordered(points: [[f64; 2]; 4]) -> [[f64; 2]; 4] {
    let c = [points.iter().map(|p| p[0]).sum::<f64>() / 4.0, points.iter().map(|p| p[1]).sum::<f64>() / 4.0];
    let mut v = points;
    // y grows downwards, so increasing atan2 runs clockwise on screen.
    v.sort_by(|a, b| (a[1] - c[1]).atan2(a[0] - c[0]).total_cmp(&(b[1] - c[1]).atan2(b[0] - c[0])));
    let first = (0..4).min_by(|&i, &j| (v[i][0] + v[i][1]).total_cmp(&(v[j][0] + v[j][1]))).unwrap_or(0);
    v.rotate_left(first);
    v
}

fn inside(q: &[[f64; 2]; 4], p: [f64; 2]) -> bool {
    (0..4).all(|i| {
        let (a, b) = (q[i], q[(i + 1) % 4]);
        (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0.0
    })
}

/// Pointer events (document px) with the tool. True when consumed.
pub fn pointer(app: &mut PhotocraftApp, ev: ToolEvent) -> bool {
    if app.ui.tool != Tool::PerspectiveCrop {
        app.pcrop.drag = None;
        return false;
    }
    let p = match ev {
        ToolEvent::Down { x, y, .. } | ToolEvent::Move { x, y, .. } | ToolEvent::Up { x, y } => [x, y],
    };
    if app.session.active().is_none() || !p[0].is_finite() || !p[1].is_finite() {
        return true;
    }
    cancel_stale(app);
    let tol = crate::distort_ui::tolerance(app);
    match ev {
        ToolEvent::Down { .. } => {
            app.pcrop.drag = match app.ui.perspective_crop {
                Some(q) => {
                    let near = (0..4)
                        .filter(|&i| (q[i][0] - p[0]).hypot(q[i][1] - p[1]) <= tol)
                        .min_by(|&i, &j| (q[i][0] - p[0]).hypot(q[i][1] - p[1]).total_cmp(&(q[j][0] - p[0]).hypot(q[j][1] - p[1])));
                    match near {
                        Some(index) => Some(Drag::Corner { index, offset: [p[0] - q[index][0], p[1] - q[index][1]] }),
                        None if inside(&q, p) => Some(Drag::Move { start: p, quad: q }),
                        None => None,
                    }
                }
                None => Some(Drag::New { start: p, cur: p }),
            };
        }
        ToolEvent::Move { .. } => drag_to(app, p),
        ToolEvent::Up { .. } => {
            drag_to(app, p);
            if let Some(Drag::New { start, cur }) = app.pcrop.drag {
                let click = tol * CLICK_PX / 8.0;
                if (cur[0] - start[0]).abs() <= click && (cur[1] - start[1]).abs() <= click {
                    app.pcrop.clicks.push(start);
                    if let [a, b, c, d] = app.pcrop.clicks[..] {
                        app.ui.perspective_crop = Some(ordered([a, b, c, d]));
                        app.pcrop.clicks.clear();
                    }
                } else {
                    app.pcrop.clicks.clear();
                    let (x0, y0, x1, y1) = (start[0].min(cur[0]), start[1].min(cur[1]), start[0].max(cur[0]), start[1].max(cur[1]));
                    app.ui.perspective_crop = Some([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
                }
            }
            app.pcrop.drag = None;
        }
    }
    true
}

fn drag_to(app: &mut PhotocraftApp, p: [f64; 2]) {
    match app.pcrop.drag {
        Some(Drag::New { start, .. }) => app.pcrop.drag = Some(Drag::New { start, cur: p }),
        Some(Drag::Corner { index, offset }) => {
            if let Some(c) = app.ui.perspective_crop.as_mut().and_then(|q| q.get_mut(index)) {
                *c = [p[0] - offset[0], p[1] - offset[1]];
            }
        }
        Some(Drag::Move { start, quad }) => {
            let (dx, dy) = (p[0] - start[0], p[1] - start[1]);
            app.ui.perspective_crop = Some(quad.map(|c| [c[0] + dx, c[1] + dy]));
        }
        None => {}
    }
}

/// `image.perspectiveCrop`'s size params from W, H and the resolution (the Crop tool's fields):
/// lengths in physical units convert at the typed resolution, else the document's.
fn size_params(app: &PhotocraftApp, p: &mut Value) {
    let o = &app.ui.tool_options;
    let res = crate::crop_size::resolution_ppi(o);
    let dpi = res.unwrap_or_else(|| app.session.active().map_or(72.0, |st| f64::from(st.doc.resolution_dpi)));
    let px = |text: &str| {
        let (v, u) = crate::crop_size::parse_length(text, Unit::Pixels)?;
        let px = u.to_px(v, dpi, 0.0, 72.0).round();
        (px.is_finite() && px >= 1.0).then_some(px)
    };
    if let Some(w) = px(&o.crop_width) {
        p["width"] = json!(w);
    }
    if let Some(h) = px(&o.crop_height) {
        p["height"] = json!(h);
    }
    if let Some(r) = res {
        p["resolution"] = json!(r);
    }
}

/// ↵: crops to the pending quad. An invalid quad stays for fixing, with the engine's reason.
pub fn commit(app: &mut PhotocraftApp) {
    cancel_stale(app);
    let Some(q) = app.ui.perspective_crop else { return };
    let mut p = json!({"corners": q});
    size_params(app, &mut p);
    if app.run("image.perspectiveCrop", p).is_ok() {
        cancel(app);
    }
}

/// ↵ commits and Esc cancels while the tool has something pending. True when a key was taken.
pub fn keys(app: &mut PhotocraftApp, ctx: &egui::Context) -> bool {
    if !active(app) || !pending(app) {
        return false;
    }
    use egui::{Key, Modifiers};
    if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Enter)) {
        commit(app);
        return true;
    }
    if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape)) {
        cancel(app);
        return true;
    }
    false
}

/// Grid lines per side for a quad about `side` screen px across (Photoshop's grid gets denser as
/// the quad grows).
fn grid_steps(side: f32) -> usize {
    ((side / 48.0).round() as usize).clamp(2, 12)
}

/// The quad, its perspective grid and corner handles; corners clicked so far; the box being drawn.
pub fn draw_overlay(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    if app.ui.tool != Tool::PerspectiveCrop {
        return;
    }
    let scr = |p: [f64; 2]| xf.to_screen(p[0] as f32, p[1] as f32);
    let accent = crate::theme::Tokens::get(painter.ctx()).accent;
    let quad = match app.pcrop.drag {
        Some(Drag::New { start, cur }) => {
            let (x0, y0, x1, y1) = (start[0].min(cur[0]), start[1].min(cur[1]), start[0].max(cur[0]), start[1].max(cur[1]));
            (x1 > x0 && y1 > y0).then_some([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        }
        _ => app.ui.perspective_crop,
    };
    if let Some(q) = quad.filter(|q| q.iter().flatten().all(|v| v.is_finite())) {
        let ok = convex_quad(&q);
        let color = if ok { accent } else { Color32::from_rgb(0xe5, 0x3e, 0x3e) };
        let pts: Vec<Pos2> = q.iter().map(|c| scr(*c)).collect();
        if ok
            && app.ui.tool_options.perspective_crop_grid
            && let Some(h) = Homography::rect_to_quad([0.0, 0.0, 1.0, 1.0], q)
        {
            let side = pts.iter().zip(pts.iter().skip(1)).map(|(a, b)| a.distance(*b)).fold(0.0, f32::max);
            let n = grid_steps(side);
            let thin = Stroke::new(0.75, Color32::from_rgba_unmultiplied(255, 255, 255, 150));
            let at = |u: f64, v: f64| {
                let (x, y) = h.apply(u, v);
                scr([x, y])
            };
            for k in 1..n {
                let t = k as f64 / n as f64;
                painter.line_segment([at(t, 0.0), at(t, 1.0)], thin);
                painter.line_segment([at(0.0, t), at(1.0, t)], thin);
            }
        }
        painter.add(egui::Shape::closed_line(pts.clone(), Stroke::new(1.5, color)));
        if !matches!(app.pcrop.drag, Some(Drag::New { .. })) {
            for c in pts {
                let r = egui::Rect::from_center_size(c, egui::vec2(8.0, 8.0));
                painter.rect_filled(r, 0.0, Color32::WHITE);
                painter.rect_stroke(r, 0.0, Stroke::new(1.25, color), egui::StrokeKind::Inside);
            }
        }
    }
    let clicks: Vec<Pos2> = app.pcrop.clicks.iter().map(|c| scr(*c)).collect();
    if clicks.len() > 1 {
        painter.add(egui::Shape::line(clicks.clone(), Stroke::new(1.5, accent)));
    }
    for c in clicks {
        painter.circle_filled(c, 4.0, Color32::WHITE);
        painter.circle_stroke(c, 4.0, Stroke::new(1.25, accent));
    }
}

/// Options bar: W, ⇄, H, the resolution (the Crop tool's fields), Front Image, Clear, Show Grid,
/// and cancel / commit.
pub fn options_bar(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    crate::crop_size::fields(&mut app.ui.tool_options, ui);
    if crate::widgets::secondary_button(ui, tl!("Front Image"), 0.0).clicked()
        && let Some((size, dpi)) = app.session.active().map(|st| (st.doc.size, f64::from(st.doc.resolution_dpi)))
    {
        let o = &mut app.ui.tool_options;
        o.crop_width = crate::crop_size::format_length(f64::from(size.width), Unit::Pixels);
        o.crop_height = crate::crop_size::format_length(f64::from(size.height), Unit::Pixels);
        let res = if o.crop_resolution_unit == crate::crop_size::PX_PER_CM { dpi / 2.54 } else { dpi };
        o.crop_resolution = crate::widgets::fmt_num2(res);
    }
    if crate::widgets::secondary_button(ui, tl!("Clear"), 0.0).clicked() {
        crate::crop_size::clear(&mut app.ui.tool_options);
    }
    crate::widgets::vline(ui, 22.0);
    crate::widgets::checkbox(ui, &mut app.ui.tool_options.perspective_crop_grid, tl!("Show Grid"));
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        let commit_tip = crate::i18n::fmt(tl!("Commit current crop operation  ({key})"), &[("key", &crate::shortcuts::pretty("Enter"))]);
        if crate::icons::button(ui, "check", 26.0, false, &commit_tip).clicked() {
            commit(app);
        }
        if crate::icons::button(ui, "ban", 26.0, false, tl!("Cancel current crop operation  (Esc)")).clicked() {
            cancel(app);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_doc::{Color, ColorMode, Document, SampleType, Size};

    /// A box drag makes the quad, corners drag alone, and ↵'s commit crops the document to the
    /// quad's size; four clicks in any order make a quad in TL, TR, BR, BL order.
    #[test]
    fn drag_a_box_move_a_corner_and_commit() {
        let doc = Document::with_background("p", Size::new(200, 150), ColorMode::Rgb, SampleType::U8, Color::WHITE);
        let mut s = photocraft_engine::Session::new();
        s.add_document(doc, None);
        let mut app = PhotocraftApp::new(s, crate::Services::default());
        app.ui.tool = Tool::PerspectiveCrop;
        let ev = |app: &mut PhotocraftApp, e| assert!(pointer(app, e));
        ev(&mut app, ToolEvent::Down { x: 20.0, y: 10.0, pressure: 1.0 });
        ev(&mut app, ToolEvent::Move { x: 120.0, y: 90.0, pressure: 1.0 });
        ev(&mut app, ToolEvent::Up { x: 120.0, y: 90.0 });
        assert_eq!(app.ui.perspective_crop, Some([[20.0, 10.0], [120.0, 10.0], [120.0, 90.0], [20.0, 90.0]]));
        ev(&mut app, ToolEvent::Down { x: 121.0, y: 89.0, pressure: 1.0 });
        ev(&mut app, ToolEvent::Up { x: 141.0, y: 99.0 });
        assert_eq!(app.ui.perspective_crop.unwrap()[2], [140.0, 100.0]);
        commit(&mut app);
        assert!(app.ui.perspective_crop.is_none());
        let size = app.session.active().unwrap().doc.size;
        assert_eq!((size.width, size.height), (110, 86), "the quad's mean side lengths");

        assert_eq!(ordered([[10.0, 90.0], [100.0, 5.0], [0.0, 0.0], [90.0, 100.0]]), [[0.0, 0.0], [100.0, 5.0], [90.0, 100.0], [10.0, 90.0]]);
    }
}
