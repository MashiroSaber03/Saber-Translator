//! Frame tool (K): drag on the canvas to draw a rectangular or elliptical frame (the options bar
//! picks the shape). ⇧ makes it a square or circle, ⌥ draws it from the centre. Drawn over the
//! selected pixel layer's content, the frame takes that layer in, as in Photoshop; else it is an
//! empty placeholder. Each frame is one engine command (`layer.new.frame`), so one undo step; this
//! module only turns the drag into it.

use egui::{Pos2, Stroke};
use photocraft_doc::LayerContent;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::PhotocraftApp;
use crate::canvas::{ToolEvent, ViewXform};
use crate::state::Tool;
use crate::theme::Tokens;

/// A press that moves less than this (screen points) is a click, not a drawn frame.
const CLICK_PX: f64 = 3.0;

/// Frame tool state (serde: readable and settable through the control channel).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FrameToolUi {
    /// Options bar: draw elliptical frames (else rectangular).
    pub ellipse: bool,
    /// The drag in progress: press and current point (document pixels), ⇧ and ⌥ at the last event.
    #[serde(skip)]
    pub drag: Option<FrameDrag>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameDrag {
    pub start: [f64; 2],
    pub cur: [f64; 2],
    pub square: bool,
    pub centred: bool,
}

/// The frame a drag leaves, `[x0, y0, x1, y1]` on whole pixels.
fn drawn(d: &FrameDrag) -> [i64; 4] {
    let (mut dx, mut dy) = (d.cur[0] - d.start[0], d.cur[1] - d.start[1]);
    if d.square {
        let side = dx.abs().max(dy.abs());
        (dx, dy) = (side.copysign(dx), side.copysign(dy));
    }
    let (a, b) =
        if d.centred { ([d.start[0] - dx, d.start[1] - dy], [d.start[0] + dx, d.start[1] + dy]) } else { (d.start, [d.start[0] + dx, d.start[1] + dy]) };
    [a[0].min(b[0]), a[1].min(b[1]), a[0].max(b[0]), a[1].max(b[1])].map(|v| v.round() as i64)
}

/// The selected pixel layer the frame `r` takes in: one whose content it overlaps and that can
/// move (not the Background or a position-locked layer).
fn content_layer(app: &PhotocraftApp, r: [i64; 4]) -> Option<u64> {
    let st = app.session.active()?;
    let l = st.doc.layer(st.active_layer?)?;
    if !matches!(l.content, LayerContent::Raster(_)) || l.locks.position || l.locks.all {
        return None;
    }
    let b = l.surface()?.content_bounds();
    let overlaps = i64::from(b.x0) < r[2] && r[0] < i64::from(b.x1) && i64::from(b.y0) < r[3] && r[1] < i64::from(b.y1);
    (!b.is_empty() && overlaps).then_some(l.id.0)
}

/// Pointer input for the Frame tool. Returns true when the event was its own.
pub fn pointer(app: &mut PhotocraftApp, ev: ToolEvent, mods: egui::Modifiers) -> bool {
    if app.ui.tool != Tool::Frame {
        return false;
    }
    let (square, centred) = (mods.shift, mods.alt);
    match ev {
        ToolEvent::Down { x, y, .. } => {
            if app.session.active().is_some() {
                app.ui.frame_tool.drag = Some(FrameDrag { start: [x, y], cur: [x, y], square, centred });
            }
        }
        ToolEvent::Move { x, y, .. } => {
            if let Some(d) = &mut app.ui.frame_tool.drag {
                *d = FrameDrag { cur: [x, y], square, centred, ..*d };
            }
        }
        ToolEvent::Up { x, y } => {
            let Some(mut d) = app.ui.frame_tool.drag.take() else { return true };
            d.cur = [x, y];
            let screen = (d.cur[0] - d.start[0]).hypot(d.cur[1] - d.start[1]) * f64::from(app.point_zoom());
            let r = drawn(&d);
            if screen < CLICK_PX || r[2] <= r[0] || r[3] <= r[1] {
                return true;
            }
            let mut p = json!({
                "x": r[0], "y": r[1], "width": r[2] - r[0], "height": r[3] - r[1],
                "shape": if app.ui.frame_tool.ellipse { "ellipse" } else { "rectangle" },
            });
            if let Some(id) = content_layer(app, r) {
                p["layer"] = json!(id);
            }
            if let Err(e) = app.run("layer.new.frame", p) {
                app.ui.status = e;
                app.ui.status_error = true;
            }
        }
    }
    true
}

/// The frame a drag in progress would leave.
pub fn draw_overlay(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    let Some(d) = app.ui.frame_tool.drag.filter(|_| app.ui.tool == Tool::Frame) else { return };
    let t = Tokens::get(painter.ctx());
    let r = drawn(&d).map(|v| v as f32);
    let stroke = Stroke::new(1.5, t.accent);
    let points: Vec<Pos2> = if app.ui.frame_tool.ellipse {
        let (cx, cy, rx, ry) = ((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0, (r[2] - r[0]) / 2.0, (r[3] - r[1]) / 2.0);
        (0..=64).map(|i| i as f32 * std::f32::consts::TAU / 64.0).map(|a| xf.to_screen(cx + rx * a.cos(), cy + ry * a.sin())).collect()
    } else {
        [(r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3]), (r[0], r[1])].into_iter().map(|(x, y)| xf.to_screen(x, y)).collect()
    };
    painter.add(egui::Shape::line(points, stroke));
}

/// The Frame tool's options bar: Create a new rectangular frame / elliptical frame.
pub fn options_bar(app: &mut PhotocraftApp, ui: &mut egui::Ui, tool: Tool) -> bool {
    if tool != Tool::Frame {
        return false;
    }
    let ellipse = app.ui.frame_tool.ellipse;
    if crate::icons::button(ui, "square", 24.0, !ellipse, tl!("Create a new rectangular frame")).clicked() {
        app.ui.frame_tool.ellipse = false;
    }
    if crate::icons::button(ui, "circle", 24.0, ellipse, tl!("Create a new elliptical frame")).clicked() {
        app.ui.frame_tool.ellipse = true;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::tool_event;
    use photocraft_doc::LayerId;

    fn drag(app: &mut PhotocraftApp, from: [f64; 2], to: [f64; 2]) {
        let m = egui::Modifiers::NONE;
        tool_event(app, ToolEvent::Down { x: from[0], y: from[1], pressure: 1.0 }, m);
        tool_event(app, ToolEvent::Move { x: to[0], y: to[1], pressure: 1.0 }, m);
        tool_event(app, ToolEvent::Up { x: to[0], y: to[1] }, m);
    }

    /// K picks the Frame tool; a rectangular drag makes one frame (one undo step) clipped to the
    /// drawn rect; with the elliptical button the frame's corners are hidden.
    #[test]
    fn frame_tool_draws_rectangular_and_elliptical_frames() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 400, "height": 300})).unwrap();
        app.sync_views();
        app.ui.extras.snap = false;
        let ctx = egui::Context::default();
        let raw = egui::RawInput {
            events: vec![egui::Event::Key { key: egui::Key::K, physical_key: None, pressed: true, repeat: false, modifiers: egui::Modifiers::NONE }],
            ..Default::default()
        };
        ctx.begin_pass(raw);
        crate::shortcuts::handle(&mut app, &ctx);
        ctx.end_pass().textures_delta.clear();
        assert_eq!(app.ui.tool, Tool::Frame);

        let frame = |app: &PhotocraftApp| {
            let st = app.session.active().unwrap();
            let l = st.doc.layer(st.active_layer.unwrap()).unwrap();
            assert!(matches!(l.content, LayerContent::Group(_)) && l.name.starts_with("Frame"), "the new frame is selected");
            (l.id, l.mask.clone().unwrap())
        };
        let layers = app.session.active().unwrap().doc.layers.len();
        drag(&mut app, [20.0, 30.0], [120.0, 110.0]);
        let (_, m) = frame(&app);
        assert_eq!((m.value(20, 30), m.value(119, 109), m.value(19, 30), m.value(120, 109)), (1.0, 1.0, 0.0, 0.0));
        app.run("edit.undo", json!({})).unwrap();
        assert_eq!(app.session.active().unwrap().doc.layers.len(), layers, "one undo takes the frame back");

        app.ui.frame_tool.ellipse = true;
        drag(&mut app, [20.0, 30.0], [120.0, 110.0]);
        let (id, m): (LayerId, _) = frame(&app);
        assert_eq!((m.value(70, 70), m.value(20, 30)), (1.0, 0.0), "centre shown, corner hidden");
        // A click draws nothing.
        drag(&mut app, [300.0, 250.0], [300.0, 250.0]);
        assert_eq!(frame(&app).0, id);
    }
}
