//! Gradient tool, live mode (Photoshop 2023+ "Gradient" vs "Classic gradient"): a drag makes a
//! Gradient Fill layer (`gradient.fill.create`), and with a gradient fill layer selected the canvas
//! shows its widget: start and end handles (drag to move, ⇧ snaps the angle to 45°), the colour
//! stops along the line (drag to move, drag off the line to delete, double-click for the Color
//! Picker), midpoint diamonds between them, a click on the line adds a stop, and dragging the line
//! moves the whole gradient. A drag elsewhere redraws the selected gradient. Radial and Angle
//! gradients show their circle, Diamond its square, Reflected its mirrored half.
//!
//! Every gesture previews live through the regular canvas path (a shallow copy of the document
//! with the edited fill, so the GPU canvas just re-runs its gradient pass) and commits one command
//! on release: one history step. The Properties panel's stops editor ([`properties`]) previews
//! through the same path.

use std::sync::Arc;

use egui::emath::GuiRounding;
use egui::{Color32, Pos2, Rect, Sense, Stroke, StrokeKind, pos2, vec2};
use photocraft_compose::gradient_fill as gf;
use photocraft_doc::{Document, Fill, GradientStyle, Layer, LayerContent, LayerId};
use photocraft_engine::gradient_fill_cmds as cmds;
use photocraft_engine::presets::gradients as gradient_presets;
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::canvas::{ToolEvent, ViewXform};
use crate::state::Tool;
use crate::theme::Tokens;
use crate::widgets;

#[cfg(test)]
mod editor_tests;
mod properties_panel;

/// Id of the temporary layer previewing a new gradient while it is drawn.
pub const PREVIEW_LAYER: LayerId = LayerId(u64::MAX - 78);
/// Tag of this module's canvas preview keys.
const TAG: u64 = 1 << 58;
/// Screen distance (points) within which a handle, stop or the line is grabbed.
const GRAB: f32 = 8.0;
/// Screen distance from the line past which a dragged stop is deleted.
const TEAR_OFF: f32 = 36.0;
/// Seconds between two clicks of a double-click.
const DOUBLE_CLICK: f64 = 0.45;

/// What a pointer press grabbed on the widget.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Grab {
    Start,
    End,
    /// A colour stop (index in sorted order).
    Stop(usize),
    /// The midpoint of the segment after colour stop `i`.
    Mid(usize),
    /// The line at gradient location `t`.
    Line(f32),
}

#[derive(Clone, Debug)]
enum Drag {
    /// Drawing a new gradient, or redrawing the selected one.
    Draw { from: [f32; 2], to: [f32; 2], redraw: Option<LayerId> },
    /// Editing the selected gradient: the command (and params) to commit on release.
    Edit { layer: LayerId, grab: Grab, down: [f32; 2], moved: bool, commit: Option<(&'static str, Value)> },
}

/// Live-gradient interaction state (not saved).
#[derive(Default)]
pub struct LiveGradient {
    drag: Option<Drag>,
    last_click: Option<(Grab, f64)>,
    /// Pending edit from the Properties panel's stops editor (previewed like a canvas drag).
    panel: Option<(LayerId, &'static str, Value)>,
    /// Pending edit from the Gradient Editor window's current-gradient strip
    /// (`gradient.presets.stop`, previewed the same way).
    preset_stop: Option<Value>,
    /// Cached preview document by key.
    preview: Option<(u64, Arc<Document>)>,
}

/// Whether the Gradient tool is in its live (Gradient Fill layer) mode.
/// Painting into a layer mask, an alpha channel or the Quick Mask stays classic (pixels), except
/// that a selected gradient fill layer is always edited live (selecting it targets its mask).
pub fn live_mode(app: &PhotocraftApp) -> bool {
    if app.active_tool() != Tool::Gradient || app.ui.tool_options.gradient_classic {
        return false;
    }
    match crate::canvas::paint_target(app).as_str() {
        Some("pixels") => true,
        Some("mask") => active_gradient(app).is_some(),
        _ => false,
    }
}

/// The active layer when it is a gradient fill layer: (layer, its frame, the canvas).
fn active_gradient(app: &PhotocraftApp) -> Option<(Layer, Rect32, photocraft_geom::Rect)> {
    let st = app.session.active()?;
    let l = st.doc.layer(st.active_layer?)?;
    if !matches!(l.content, LayerContent::Fill(Fill::Gradient { .. })) || !l.visible {
        return None;
    }
    let canvas = st.doc.bounds();
    Some((l.clone(), photocraft_compose::fill_frame(l, canvas), canvas))
}

type Rect32 = photocraft_geom::Rect;

fn fill_of(l: &Layer) -> Option<&Fill> {
    match &l.content {
        LayerContent::Fill(f @ Fill::Gradient { .. }) => Some(f),
        _ => None,
    }
}

/// The fill an in-progress edit shows (`None` when it can't apply, e.g. a stale index).
fn edited_fill(app: &PhotocraftApp, layer: &Layer, canvas: Rect32, cmd: &str, p: &Value) -> Option<Fill> {
    let f = fill_of(layer)?;
    let (fg, bg) = (app.session.tools.foreground, app.session.tools.background);
    let edited = match cmd {
        cmds::SET => cmds::apply_set(layer, f, canvas, p, fg, bg).ok(),
        cmds::STOP => cmds::apply_stop(f, p, fg, bg).ok(),
        _ => None,
    }?;
    // As the command commits it: colours in the document's mode.
    Some(match app.session.active() {
        Some(st) => edited.in_mode(st.doc.mode),
        None => edited,
    })
}

/// Options-bar params of a new live gradient.
fn create_params(app: &PhotocraftApp, from: [f32; 2], to: [f32; 2]) -> Value {
    let o = &app.ui.tool_options;
    json!({"from": from, "to": to, "style": o.gradient_style, "reverse": o.gradient_reverse, "dither": o.gradient_dither, "opacity": o.fill_opacity, "mode": o.gradient_blend_mode.label()})
}

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// Position on the segment `s`→`e` at fraction `u`.
fn lerp(s: [f32; 2], e: [f32; 2], u: f32) -> [f32; 2] {
    [s[0] + (e[0] - s[0]) * u, s[1] + (e[1] - s[1]) * u]
}

/// Fraction along `s`→`e` of `p`'s projection (clamped) and `p`'s distance from the line.
fn project(s: [f32; 2], e: [f32; 2], p: [f32; 2]) -> (f32, f32) {
    let (dx, dy) = (e[0] - s[0], e[1] - s[1]);
    let l2 = dx * dx + dy * dy;
    if l2 <= 1e-9 {
        return (0.0, dist(s, p));
    }
    let u = (((p[0] - s[0]) * dx + (p[1] - s[1]) * dy) / l2).clamp(0.0, 1.0);
    (u, dist(lerp(s, e, u), p))
}

/// The widget's geometry for a gradient fill: handles, stop positions, midpoint positions.
struct Widget {
    style: GradientStyle,
    start: [f32; 2],
    end: [f32; 2],
    reverse: bool,
    /// Sorted colour stops: (location, rgba).
    stops: Vec<(f32, [f32; 4])>,
    mids: Vec<f32>,
}

impl Widget {
    fn new(f: &Fill, frame: Rect32) -> Option<Widget> {
        let Fill::Gradient { stops, midpoints, reverse, style, .. } = f else { return None };
        let (start, end) = gf::fill_handles(f, frame)?;
        let mut st: Vec<(f32, [f32; 4])> = stops
            .iter()
            .map(|(t, c)| {
                let rgb = c.to_rgb();
                (*t, [rgb[0], rgb[1], rgb[2], c.alpha])
            })
            .collect();
        st.sort_by(|a, b| a.0.total_cmp(&b.0));
        Some(Widget { style: *style, start, end, reverse: *reverse, stops: st, mids: midpoints.clone() })
    }

    /// Fraction along the line of gradient location `t`.
    fn u_of(&self, t: f32) -> f32 {
        if self.reverse { 1.0 - t } else { t }
    }

    fn at(&self, t: f32) -> [f32; 2] {
        lerp(self.start, self.end, self.u_of(t))
    }

    fn mid_location(&self, i: usize) -> Option<f32> {
        let (a, b) = (self.stops.get(i)?.0, self.stops.get(i + 1)?.0);
        (b - a > 1e-4).then(|| a + (b - a) * self.mids.get(i).copied().unwrap_or(0.5))
    }

    /// What `p` (document pixels) grabs, with `tol` the grab distance in document pixels.
    fn hit(&self, p: [f32; 2], tol: f32) -> Option<Grab> {
        if dist(p, self.end) <= tol {
            return Some(Grab::End);
        }
        if dist(p, self.start) <= tol {
            return Some(Grab::Start);
        }
        let near = |q: [f32; 2]| dist(p, q) <= tol;
        if let Some(i) = self.stops.iter().position(|s| near(self.at(s.0))) {
            return Some(Grab::Stop(i));
        }
        if let Some(i) = (0..self.stops.len().saturating_sub(1)).find(|i| self.mid_location(*i).is_some_and(|t| near(self.at(t)))) {
            return Some(Grab::Mid(i));
        }
        let (u, d) = project(self.start, self.end, p);
        (d <= tol).then_some(Grab::Line(if self.reverse { 1.0 - u } else { u }))
    }
}

/// Pointer input for the Gradient tool in live mode. Returns true when consumed.
pub fn pointer(app: &mut PhotocraftApp, ev: ToolEvent, mods: egui::Modifiers) -> bool {
    if !live_mode(app) {
        app.gradient.drag = None;
        return false;
    }
    let zoom = app.point_zoom().max(0.01);
    match ev {
        ToolEvent::Down { x, y, .. } => {
            let p = [x as f32, y as f32];
            let active = active_gradient(app);
            let grab = active.as_ref().and_then(|(l, frame, _)| Widget::new(fill_of(l)?, *frame)?.hit(p, GRAB / zoom));
            app.gradient.drag = Some(match (grab, active) {
                (Some(grab), Some((l, ..))) => Drag::Edit { layer: l.id, grab, down: p, moved: false, commit: None },
                (_, active) => Drag::Draw { from: p, to: p, redraw: active.map(|a| a.0.id) },
            });
        }
        ToolEvent::Move { x, y, .. } => drag_to(app, [x as f32, y as f32], mods),
        ToolEvent::Up { x, y } => {
            drag_to(app, [x as f32, y as f32], mods);
            finish(app);
        }
    }
    true
}

fn drag_to(app: &mut PhotocraftApp, p: [f32; 2], mods: egui::Modifiers) {
    let zoom = app.point_zoom().max(0.01);
    let active = active_gradient(app);
    let Some(drag) = app.gradient.drag.as_mut() else { return };
    // ⇧: 45° gradient angles, the same snap as the classic drag (stroke_constraint.rs).
    let snap = |from: [f32; 2], to: [f32; 2]| -> [f32; 2] {
        if !mods.shift {
            return to;
        }
        let c = crate::stroke_constraint::snap45([f64::from(from[0]), f64::from(from[1])], [f64::from(to[0]), f64::from(to[1])]);
        [c[0] as f32, c[1] as f32]
    };
    match drag {
        Drag::Draw { from, to, .. } => *to = snap(*from, p),
        Drag::Edit { layer, grab, down, moved, commit } => {
            if dist(*down, p) * zoom >= 2.0 {
                *moved = true;
            }
            if !*moved {
                return;
            }
            let Some(w) = active.as_ref().filter(|a| a.0.id == *layer).and_then(|(l, frame, _)| Widget::new(fill_of(l)?, *frame)) else { return };
            let id = layer.0;
            *commit = match *grab {
                Grab::Start => Some((cmds::SET, json!({"layer": id, "from": snap(w.end, p)}))),
                Grab::End => Some((cmds::SET, json!({"layer": id, "to": snap(w.start, p)}))),
                Grab::Line(_) => {
                    let (dx, dy) = (p[0] - down[0], p[1] - down[1]);
                    Some((cmds::SET, json!({"layer": id, "from": [w.start[0] + dx, w.start[1] + dy], "to": [w.end[0] + dx, w.end[1] + dy]})))
                }
                Grab::Stop(i) => {
                    let (u, d) = project(w.start, w.end, p);
                    if d * zoom > TEAR_OFF && w.stops.len() > 2 {
                        Some((cmds::STOP, json!({"layer": id, "action": "delete", "index": i})))
                    } else {
                        Some((cmds::STOP, json!({"layer": id, "action": "move", "index": i, "location": w.u_of(u)})))
                    }
                }
                Grab::Mid(i) => {
                    let t = w.u_of(project(w.start, w.end, p).0);
                    match (w.stops.get(i), w.stops.get(i + 1)) {
                        (Some(a), Some(b)) if b.0 - a.0 > 1e-4 => {
                            Some((cmds::STOP, json!({"layer": id, "action": "midpoint", "index": i, "location": ((t - a.0) / (b.0 - a.0)).clamp(0.05, 0.95)})))
                        }
                        _ => None,
                    }
                }
            };
        }
    }
}

fn finish(app: &mut PhotocraftApp) {
    let Some(drag) = app.gradient.drag.take() else { return };
    let zoom = app.point_zoom().max(0.01);
    match drag {
        Drag::Draw { from, to, redraw } => {
            if dist(from, to) * zoom < 2.0 {
                return;
            }
            let _ = match redraw {
                Some(id) => app.run(cmds::SET, json!({"layer": id.0, "from": from, "to": to})),
                None => {
                    let p = create_params(app, from, to);
                    app.run(cmds::CREATE, p)
                }
            };
        }
        Drag::Edit { commit: Some((cmd, p)), moved: true, .. } => {
            let _ = app.run(cmd, p);
        }
        Drag::Edit { layer, grab, moved: false, .. } => click(app, layer, grab),
        Drag::Edit { .. } => {}
    }
}

/// A click (no drag) on the widget: the line adds a stop; a double-click on a stop (or on an end
/// handle, for the stop there) opens the Color Picker.
fn click(app: &mut PhotocraftApp, layer: LayerId, grab: Grab) {
    let now = app.last_frame_time;
    let double = app.gradient.last_click.is_some_and(|(g, t)| g == grab && now - t <= DOUBLE_CLICK);
    app.gradient.last_click = Some((grab, now));
    match grab {
        Grab::Line(t) => {
            let _ = app.run(cmds::STOP, json!({"layer": layer.0, "action": "add", "location": t}));
        }
        Grab::Stop(i) if double => edit_stop_color(app, layer, i),
        Grab::Start | Grab::End if double => {
            let Some((l, frame, _)) = active_gradient(app) else { return };
            let Some(w) = fill_of(&l).and_then(|f| Widget::new(f, frame)) else { return };
            // The stop at that end of the line.
            let at_start = (grab == Grab::Start) != w.reverse;
            let i = if at_start { 0 } else { w.stops.len().saturating_sub(1) };
            edit_stop_color(app, layer, i);
        }
        _ => {}
    }
}

/// Opens the Color Picker on colour stop `i`; OK runs `gradient.fill.stop` (one history step).
pub fn edit_stop_color(app: &mut PhotocraftApp, layer: LayerId, i: usize) {
    let Some(st) = app.session.active() else { return };
    let Some(Fill::Gradient { stops, .. }) = st.doc.layer(layer).and_then(fill_of) else { return };
    let mut sorted = stops.clone();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    let Some((_, c)) = sorted.get(i) else { return };
    app.gradient.last_click = None;
    crate::color_picker_ui::open_for_command(
        app,
        "Color Picker (Stop Color)",
        c.to_rgb(),
        cmds::STOP,
        json!({"layer": layer.0, "action": "color", "index": i}),
    );
}

fn preview_key(revision: u64, what: &str) -> u64 {
    let h = what.bytes().fold(revision.wrapping_mul(0x9e37_79b9_7f4a_7c15), |h, b| h.wrapping_mul(31).wrapping_add(u64::from(b)));
    (h & (TAG - 1)) | TAG
}

/// The document to show while a gradient is drawn or edited (`canvas::display_doc`).
pub fn display_doc(app: &mut PhotocraftApp, idx: usize) -> Option<(Arc<Document>, u64)> {
    if app.session.active_index() != Some(idx) {
        return None;
    }
    let st = app.session.active()?;
    let (doc, revision) = (st.doc.clone(), st.revision);
    let edit: Option<(LayerId, &'static str, Value)> = match &app.gradient.drag {
        Some(Drag::Draw { from, to, redraw }) if live_mode(app) && dist(*from, *to) >= 1.0 => match redraw {
            Some(id) => Some((*id, cmds::SET, json!({"from": from, "to": to}))),
            None => Some((PREVIEW_LAYER, cmds::CREATE, create_params(app, *from, *to))),
        },
        Some(Drag::Edit { layer, commit: Some((cmd, p)), moved: true, .. }) if live_mode(app) => Some((*layer, *cmd, p.clone())),
        _ => app.gradient.panel.clone(),
    };
    let Some((layer, cmd, p)) = edit else {
        app.gradient.preview = None;
        return None;
    };
    let key = preview_key(revision, &format!("{}{cmd}{p}", layer.0));
    if let Some((k, d)) = &app.gradient.preview
        && *k == key
    {
        return Some((d.clone(), key));
    }
    let mut shown = (*doc).clone();
    if cmd == cmds::CREATE {
        let mut l = cmds::new_layer(&app.session, &doc, &p).ok()?;
        l.id = PREVIEW_LAYER;
        let above = app.session.active().and_then(|s| s.active_layer);
        shown.insert_above(above, l);
    } else {
        let l = doc.layer(layer)?;
        let f = edited_fill(app, l, doc.bounds(), cmd, &p)?;
        shown.layer_mut(layer)?.content = LayerContent::Fill(f);
    }
    let shown = Arc::new(shown);
    app.gradient.preview = Some((key, shown.clone()));
    Some((shown, key))
}

/// The fill and frame the widget shows: the one being drawn or edited, else the selected one.
fn shown_fill(app: &PhotocraftApp) -> Option<(Fill, Rect32, Option<Grab>)> {
    let st = app.session.active()?;
    let canvas = st.doc.bounds();
    match &app.gradient.drag {
        Some(Drag::Draw { from, to, redraw: None }) => {
            let l = cmds::new_layer(&app.session, &st.doc, &create_params(app, *from, *to)).ok()?;
            let f = fill_of(&l)?.clone();
            Some((f, photocraft_compose::fill_frame(&l, canvas), Some(Grab::End)))
        }
        Some(Drag::Draw { from, to, redraw: Some(id) }) => {
            let l = st.doc.layer(*id)?;
            let f = edited_fill(app, l, canvas, cmds::SET, &json!({"from": from, "to": to}))?;
            Some((f, photocraft_compose::fill_frame(l, canvas), Some(Grab::End)))
        }
        Some(Drag::Edit { layer, grab, commit, .. }) => {
            let l = st.doc.layer(*layer)?;
            let f = match commit {
                Some((cmd, p)) => match edited_fill(app, l, canvas, cmd, p) {
                    Some(f) => f,
                    None => fill_of(l)?.clone(),
                },
                None => fill_of(l)?.clone(),
            };
            // A torn-off stop no longer exists: highlight nothing.
            let deleting = commit.as_ref().is_some_and(|(_, p)| p["action"] == "delete");
            Some((f, photocraft_compose::fill_frame(l, canvas), (!deleting).then_some(*grab)))
        }
        None => {
            let (l, frame, _) = active_gradient(app)?;
            Some((fill_of(&l)?.clone(), frame, None))
        }
    }
}

fn c32(c: [f32; 4]) -> Color32 {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    Color32::from_rgb(q(c[0]), q(c[1]), q(c[2]))
}

fn line(painter: &egui::Painter, a: Pos2, b: Pos2) {
    painter.line_segment([a, b], Stroke::new(3.0, Color32::from_black_alpha(90)));
    painter.line_segment([a, b], Stroke::new(1.5, Color32::WHITE));
}

fn diamond(painter: &egui::Painter, c: Pos2, r: f32, fill: Color32, outline: Color32) {
    let pts = vec![pos2(c.x, c.y - r), pos2(c.x + r, c.y), pos2(c.x, c.y + r), pos2(c.x - r, c.y)];
    painter.add(egui::Shape::convex_polygon(pts, fill, Stroke::new(1.0, outline)));
}

/// Draws the live gradient widget over the canvas.
pub fn draw_overlay(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    if !live_mode(app) {
        return;
    }
    let Some((fill, frame, grabbed)) = shown_fill(app) else { return };
    let Some(w) = Widget::new(&fill, frame) else { return };
    let accent = Tokens::get(painter.ctx()).accent;
    let sp = |p: [f32; 2]| xf.to_screen(p[0], p[1]);
    let (s, e) = (sp(w.start), sp(w.end));
    let r = s.distance(e);
    let shadow = Stroke::new(3.0, Color32::from_black_alpha(90));
    let white = Stroke::new(1.5, Color32::WHITE);
    match w.style {
        GradientStyle::Radial | GradientStyle::Angle => {
            painter.circle_stroke(s, r, shadow);
            painter.circle_stroke(s, r, white);
        }
        GradientStyle::Diamond => {
            let d = e - s;
            let n = vec2(-d.y, d.x);
            let pts = vec![s + d, s + n, s - d, s - n, s + d];
            painter.add(egui::Shape::line(pts.clone(), shadow));
            painter.add(egui::Shape::line(pts, white));
        }
        GradientStyle::Reflected => {
            let m = s - (e - s);
            painter.add(egui::Shape::dashed_line(&[s, m], Stroke::new(1.5, Color32::WHITE), 5.0, 4.0));
        }
        GradientStyle::Linear => {}
    }
    line(painter, s, e);
    // Midpoints.
    for i in 0..w.stops.len().saturating_sub(1) {
        if let Some(t) = w.mid_location(i) {
            let sel = grabbed == Some(Grab::Mid(i));
            diamond(painter, sp(w.at(t)), 4.5, if sel { accent } else { Color32::WHITE }, Color32::from_black_alpha(160));
        }
    }
    // End handles, then the colour stops (a stop at an end sits inside its handle).
    for (p, g) in [(s, Grab::Start), (e, Grab::End)] {
        let ring = if grabbed == Some(g) { accent } else { Color32::WHITE };
        painter.circle_stroke(p, 8.0, Stroke::new(3.5, Color32::from_black_alpha(90)));
        painter.circle_stroke(p, 8.0, Stroke::new(2.0, ring));
    }
    for (i, (t, c)) in w.stops.iter().enumerate() {
        let p = sp(w.at(*t));
        let sel = grabbed == Some(Grab::Stop(i));
        painter.circle_filled(p, 5.0, c32(*c));
        painter.circle_stroke(p, 5.5, Stroke::new(1.5, if sel { accent } else { Color32::WHITE }));
        painter.circle_stroke(p, 6.8, Stroke::new(1.0, Color32::from_black_alpha(110)));
    }
}

// ---------------------------------------------------------------- options bar

/// A ramp of RGBA stops over a checkerboard in `r`.
fn paint_ramp(p: &egui::Painter, r: Rect, sample: impl Fn(f32) -> [f32; 4]) {
    // The tessellator snaps the checker's rects to pixels but not the mesh: snap both alike, or a
    // half-pixel offset leaves a row of checker showing under the ramp (#4577).
    let r = r.round_to_pixels(p.pixels_per_point());
    widgets::checker(p, r, 5.0);
    let n = 64;
    let mut mesh = egui::Mesh::default();
    for k in 0..=n {
        let u = k as f32 / n as f32;
        let c = sample(u);
        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        let col = Color32::from_rgba_unmultiplied(q(c[0]), q(c[1]), q(c[2]), q(c[3]));
        let x = r.left() + u * r.width();
        mesh.colored_vertex(pos2(x, r.top()), col);
        mesh.colored_vertex(pos2(x, r.bottom()), col);
        if k > 0 {
            let i = (2 * k) as u32;
            mesh.add_triangle(i - 2, i - 1, i);
            mesh.add_triangle(i - 1, i + 1, i);
        }
    }
    p.add(mesh);
}

/// Side, in pixels, of a gradient fill layer's thumbnail texture.
const THUMB_PX: u32 = 64;

/// Pixels of a gradient fill's Layers-panel thumbnail: the fill as it lays out (style, angle,
/// scale, offset, reverse, stops, midpoints and opacity) in a square, as Photoshop shows it.
/// `None` for other fills.
pub fn thumbnail_image(f: &Fill) -> Option<egui::ColorImage> {
    let Fill::Gradient { .. } = f else { return None };
    let n = THUMB_PX as i32;
    let square = photocraft_geom::Rect::new(0, 0, n, n);
    // No dither: it is invisible at this size and would only add noise.
    let mut plain = f.clone();
    if let Fill::Gradient { dither, .. } = &mut plain {
        *dither = false;
    }
    let px = gf::render(&plain, square, square);
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let rgba: Vec<Color32> = px.iter().map(|c| Color32::from_rgba_unmultiplied(q(c[0]), q(c[1]), q(c[2]), q(c[3]))).collect();
    Some(egui::ColorImage::new([THUMB_PX as usize; 2], rgba))
}

/// Paints a gradient fill layer's inset thumbnail and fill marker into `rect`, over a
/// checkerboard (opacity stops show through). Cached until its fill changes; `false` for other fills.
pub fn paint_thumbnail(ui: &egui::Ui, layer: LayerId, f: &Fill, rect: Rect) -> bool {
    if !matches!(f, Fill::Gradient { .. }) {
        return false;
    }
    let ctx = ui.ctx();
    let key = egui::Id::new(("gradient-thumb", layer.0));
    let cached: Option<(Fill, egui::TextureHandle)> = ctx.data(|d| d.get_temp(key));
    let tex = match cached {
        Some((cf, t)) if cf == *f => t,
        _ => {
            let Some(img) = thumbnail_image(f) else { return false };
            let t = ctx.load_texture(format!("gradient-thumb-{}", layer.0), img, egui::TextureOptions::LINEAR);
            ctx.data_mut(|d| d.insert_temp(key, (f.clone(), t.clone())));
            t
        }
    };
    if ui.is_rect_visible(rect) {
        let p = ui.painter();
        crate::solid_fill_ui::thumbnail_frame(ui, rect, |swatch| {
            widgets::checker(p, swatch, 3.0);
            p.image(tex.id(), swatch, Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
        });
    }
    true
}

/// Options-bar swatch of the current gradient. Both live and classic modes use the same
/// engine-owned preset (#1651). Clicking it opens the Gradient Editor window, as Photoshop's
/// swatch does. Returns whether it was clicked.
pub fn preset_swatch(app: &mut PhotocraftApp, ui: &mut egui::Ui) -> bool {
    let t = Tokens::get(ui.ctx());
    let stops = app.session.presets.gradient.resolve(app.session.tools.foreground, app.session.tools.background);
    let (r, resp) = ui.allocate_exact_size(vec2(96.0, 20.0), Sense::click());
    paint_ramp(ui.painter(), r, |u| photocraft_algo::paint::sample_stops(&stops, u));
    ui.painter().rect_stroke(r, 0.0, Stroke::new(1.0, t.field_border), StrokeKind::Outside);
    // Name the control for accessibility (and so tests and agents can find it).
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, tl!("Gradient picker")));
    resp.on_hover_text(tl!("Click to edit the gradient")).clicked()
}

/// The selected layer when it is a gradient fill layer. Unlike [`active_gradient`] it does not
/// require the layer to be visible: the editor edits it either way.
fn selected_gradient(app: &PhotocraftApp) -> Option<Layer> {
    let st = app.session.active()?;
    let l = st.doc.layer(st.active_layer?)?;
    fill_of(l)?;
    Some(l.clone())
}

/// The Gradient Editor window the options-bar swatch opens, standing in for Photoshop's Gradient
/// Editor. A selected gradient fill layer gets a combined ramp, with opacity above and colour
/// below, selected-stop fields and Gradient Options. Without a gradient fill layer, the same
/// stop controls edit the tool's current gradient and offer saving it as a preset.
pub fn editor_window(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if !app.ui.panels.gradient_editor {
        return;
    }
    let mut open = true;
    let mut show_panel = false;
    // Resolved before the closure so `app` is borrowed only once inside it.
    let layer = selected_gradient(app);
    egui::Window::new(tl!("Gradient Editor"))
        .id(egui::Id::new("gradient-editor"))
        .open(&mut open)
        .collapsible(false)
        .resizable(true)
        .default_width(330.0)
        .show(ctx, |ui| match &layer {
            Some(l) => properties_panel::editor(app, ui, l),
            None => {
                // Photoshop's editor edits the gradient the tool paints with, so with no gradient
                // fill layer around the strip edits the current gradient in place.
                let t = Tokens::get(ui.ctx());
                ui.label(egui::RichText::new(tl!("The gradient the Gradient tool paints with")).color(t.text_dim));
                ui.add_space(6.0);
                preset_stops_editor(app, ui);
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if widgets::secondary_button(ui, tl!("Save As Preset"), 0.0)
                        .on_hover_text(tl!("Adds the current gradient to the Gradients panel"))
                        .clicked()
                    {
                        let _ = app.run("gradient.presets.new", json!({}));
                    }
                    show_panel = widgets::secondary_button(ui, tl!("Gradients"), 0.0).clicked();
                });
            }
        });
    if show_panel {
        let _ = crate::preset_panels::menu(app, "window.panel.gradients", &Value::Null);
    }
    if !open {
        app.ui.panels.gradient_editor = false;
    }
}

/// Live mode: style, reverse, dither and blend mode changes in the options bar also edit the selected
/// gradient fill layer (one history step each), as in Photoshop.
pub fn options_changed(app: &mut PhotocraftApp, before: &crate::state::ToolOptions) {
    let o = app.ui.tool_options.clone();
    if app.ui.tool != Tool::Gradient || o.gradient_classic {
        return;
    }
    let Some((l, ..)) = active_gradient(app) else { return };
    if o.gradient_blend_mode != before.gradient_blend_mode {
        let _ = app.run("layer.setProps", json!({"layer": l.id.0, "blend": o.gradient_blend_mode.label()}));
    }
    if o.gradient_style == before.gradient_style && o.gradient_reverse == before.gradient_reverse && o.gradient_dither == before.gradient_dither {
        return;
    }
    let mut p = json!({"layer": l.id.0});
    if o.gradient_style != before.gradient_style {
        p["style"] = json!(o.gradient_style);
    }
    if o.gradient_reverse != before.gradient_reverse {
        p["reverse"] = json!(o.gradient_reverse);
    }
    if o.gradient_dither != before.gradient_dither {
        p["dither"] = json!(o.gradient_dither);
    }
    let _ = app.run(cmds::SET, p);
}

/// Live mode: the options bar's Opacity sets a selected gradient fill layer's opacity, as a
/// drag's `opacity` does, live while it is scrubbed; `drag` numbers the scrub so it is one undo
/// step, as in the Layers panel (#5012).
pub fn opacity_changed(app: &mut PhotocraftApp, drag: Option<u64>) {
    let o = &app.ui.tool_options;
    if app.ui.tool != Tool::Gradient || o.gradient_classic {
        return;
    }
    let opacity = o.fill_opacity / 100.0;
    let Some((l, ..)) = active_gradient(app) else { return };
    if (l.opacity - opacity).abs() > 1e-6 {
        let mut p = json!({"layer": l.id.0, "opacity": opacity});
        if let Some(n) = drag {
            p["coalesce"] = json!(format!("gradient-tool-opacity:{n}"));
        }
        let _ = app.run("layer.setProps", p);
    }
}

// ---------------------------------------------------------------- Properties panel

/// Properties panel for a live gradient. Every change uses the existing gradient commands.
pub fn properties(app: &mut PhotocraftApp, ui: &mut egui::Ui, layer: &Layer) {
    properties_panel::show(app, ui, layer);
}

/// Which marker of the stops editor is being dragged.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Marker {
    Color(usize),
    Opacity(usize),
    Mid(usize),
}

fn strip_key(app: &PhotocraftApp, sink: Sink) -> egui::Id {
    // Layer IDs can repeat in different documents. Tool presets belong to the session.
    let document = match sink {
        Sink::Layer(_) => app.session.active().map(|s| s.doc.id),
        Sink::Preset => None,
    };
    sink.key().with(document)
}

/// What a stops editor instance edits: a Gradient Fill layer (`gradient.fill.stop`) or the
/// Gradient tool's current gradient (`gradient.presets.stop`, the Gradient Editor window's
/// no-layer mode). Presets keep midpoints centred, so midpoint markers are layer-only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Sink {
    Layer(LayerId),
    Preset,
}

impl Sink {
    fn fill(self, app: &PhotocraftApp) -> Option<Fill> {
        match self {
            Sink::Layer(id) => fill_of(app.session.active()?.doc.layer(id)?).cloned(),
            Sink::Preset => preset_fill(app),
        }
    }
    /// Uniqueness for the drag state.
    fn key(self) -> egui::Id {
        match self {
            Sink::Layer(id) => egui::Id::new(("gradient-stops-drag", id.0)),
            Sink::Preset => egui::Id::new(("gradient-stops-drag", "preset")),
        }
    }
    fn cmd(self) -> &'static str {
        match self {
            Sink::Layer(_) => cmds::STOP,
            Sink::Preset => gradient_presets::STOP,
        }
    }
    /// Preset gradients keep midpoints at the centre (they are not stored), so only a layer's
    /// editor shows draggable midpoint diamonds.
    fn midpoints(self) -> bool {
        matches!(self, Sink::Layer(_))
    }
    /// The edit params with the layer added (the preset grammar has no `layer`).
    fn params(self, mut p: Value) -> Value {
        if let Sink::Layer(id) = self {
            p["layer"] = json!(id.0);
        }
        p
    }
}

/// The Gradient Editor window's no-layer mode: the strip edits the Gradient tool's current
/// gradient in place (`gradient.presets.stop`), as Photoshop's editor edits the tool gradient.
fn preset_stops_editor(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    if let Some(f) = preset_fill(app) {
        stops_editor_for(app, ui, Sink::Preset, f);
    }
}

/// The current gradient as a fill for the strip (foreground/background resolved now, the preset's
/// own opacity stops, empty = opaque, midpoints centred).
fn preset_fill(app: &PhotocraftApp) -> Option<Fill> {
    let (fg, bg) = (app.session.tools.foreground, app.session.tools.background);
    let (cs, os) = app.session.presets.gradient.fill_stops(fg, bg);
    Some(Fill::Gradient {
        stops: cs,
        angle: 0.0,
        scale: 1.0,
        style: GradientStyle::Linear,
        reverse: false,
        opacity_stops: os,
        midpoints: Vec::new(),
        offset: (0.0, 0.0),
        dither: false,
        align: true,
    })
}

/// Opens the Color Picker on the current gradient's colour stop `i`; OK runs
/// `gradient.presets.stop` (one journal step).
fn preset_stop_color(app: &mut PhotocraftApp, i: usize) {
    let Some(f) = preset_fill(app) else { return };
    let Fill::Gradient { stops, .. } = &f else { return };
    let mut sorted = stops.clone();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    let Some((_, c)) = sorted.get(i) else { return };
    let r = c.to_rgb();
    app.gradient.preset_stop = None;
    crate::color_picker_ui::open_for_command(app, "Color Picker (Stop Color)", r, gradient_presets::STOP, json!({"action": "color", "index": i}));
}

fn stops_editor_for(app: &mut PhotocraftApp, ui: &mut egui::Ui, sink: Sink, f: Fill) {
    let key = strip_key(app, sink).with("selection");
    if ui.data(|d| d.get_temp::<Marker>(key)).is_none() {
        ui.data_mut(|d| d.insert_temp(key, Marker::Color(0)));
    }
    stop_strip(app, ui, sink, f);
    properties_panel::stop_fields(app, ui, sink);
}

/// Keep selection on a stop after the engine sorts its new position.
fn selection_after_edit(f: &Fill, p: &Value, previous: Option<Marker>) -> Option<Marker> {
    let Fill::Gradient { stops, opacity_stops, .. } = f else { return previous };
    let opacity = p.get("kind").and_then(Value::as_str) == Some("opacity");
    let mut locations: Vec<f32> = if opacity {
        if opacity_stops.is_empty() { vec![0.0, 1.0] } else { opacity_stops.iter().map(|s| s.0).collect() }
    } else {
        stops.iter().map(|s| s.0).collect()
    };
    locations.sort_by(f32::total_cmp);
    let action = p.get("action").and_then(Value::as_str);
    let index = p.get("index").and_then(Value::as_u64).and_then(|i| usize::try_from(i).ok());
    let at = match action {
        Some("move" | "add") => {
            let location = p.get("location").and_then(Value::as_f64)? as f32;
            locations
                .iter()
                .enumerate()
                .filter(|(i, value)| Some(*i) != index && (**value < location || (**value == location && index.is_none_or(|old| *i < old))))
                .count()
        }
        Some("delete") => 0,
        _ => return previous,
    };
    Some(if opacity { Marker::Opacity(at) } else { Marker::Color(at) })
}

fn stop_strip(app: &mut PhotocraftApp, ui: &mut egui::Ui, sink: Sink, f: Fill) {
    let Fill::Gradient { stops, .. } = &f else { return };
    let t = Tokens::get(ui.ctx());
    let w = ui.available_width().max(120.0);
    let (outer, resp) = ui.allocate_exact_size(vec2(w, 58.0), Sense::click_and_drag());
    // Name the strip for accessibility (and so tests and agents can find it).
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), tl!("Gradient stops")));
    let top = outer.top() + 16.0;
    let bar = Rect::from_min_max(pos2(outer.left() + 8.0, top), pos2(outer.right() - 8.0, top + 24.0));
    let x_of = |loc: f32| bar.left() + loc.clamp(0.0, 1.0) * bar.width();
    let loc_of = |x: f32| ((x - bar.left()) / bar.width().max(1.0)).clamp(0.0, 1.0);
    // The ramp being shown (with a pending drag's edit applied).
    let pending: Option<Value> = match sink {
        Sink::Layer(id) => app.gradient.panel.clone().filter(|(l, ..)| *l == id).map(|(_, _, p)| p),
        Sink::Preset => app.gradient.preset_stop.clone(),
    };
    let (fg, bg) = (app.session.tools.foreground, app.session.tools.background);
    let shown = pending.as_ref().and_then(|p| cmds::apply_stop(&f, p, fg, bg).ok()).unwrap_or_else(|| f.clone());
    let mut track_fill = shown.clone();
    if let Fill::Gradient { reverse, .. } = &mut track_fill {
        // All markers address un-reversed positions; Reverse applies to the canvas output.
        *reverse = false;
    }
    if let Some(ramp) = gf::Ramp::new(&track_fill) {
        paint_ramp(ui.painter(), bar, |u| ramp.sample(u));
    }
    ui.painter().rect_stroke(bar, 0.0, Stroke::new(1.0, t.field_border), StrokeKind::Outside);
    let (ss, os, mids) = match &shown {
        Fill::Gradient { stops, opacity_stops, midpoints, .. } => (stops.clone(), opacity_stops.clone(), midpoints.clone()),
        _ => (Vec::new(), Vec::new(), Vec::new()),
    };
    let mut sorted = ss.clone();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut ops: Vec<(f32, f32)> = if os.is_empty() { vec![(0.0, 1.0), (1.0, 1.0)] } else { os.clone() };
    ops.sort_by(|a, b| a.0.total_cmp(&b.0));
    let key = strip_key(app, sink);
    let selection_key = key.with("selection");
    let selected: Option<Marker> = ui.data(|d| d.get_temp(selection_key));
    let dragging: Option<Marker> = ui.data(|d| d.get_temp(key));
    // Markers.
    let p = ui.painter();
    for (i, (loc, c)) in sorted.iter().enumerate() {
        let x = x_of(*loc);
        let sel = dragging.or(selected) == Some(Marker::Color(i));
        let tip = pos2(x, bar.bottom() + 1.0);
        let body = Rect::from_center_size(pos2(x, bar.bottom() + 10.0), vec2(10.0, 10.0));
        let rgb = c.to_rgb();
        let color = c32([rgb[0], rgb[1], rgb[2], 1.0]);
        p.add(egui::Shape::convex_polygon(
            vec![tip, pos2(x + 5.0, body.top()), pos2(x - 5.0, body.top())],
            if sel { t.accent } else { t.text_dim },
            Stroke::NONE,
        ));
        p.rect_filled(body, 1.0, color);
        p.rect_stroke(body, 1.0, Stroke::new(1.0, if sel { t.accent } else { t.text_dim }), StrokeKind::Outside);
    }
    if sink.midpoints() {
        for i in 0..sorted.len().saturating_sub(1) {
            let (a, b) = (sorted[i].0, sorted[i + 1].0);
            if b - a > 1e-3 {
                let m = mids.get(i).copied().unwrap_or(0.5);
                let sel = dragging.or(selected) == Some(Marker::Mid(i));
                diamond(p, pos2(x_of(a + (b - a) * m), bar.bottom() + 6.0), 3.0, if sel { t.accent } else { t.text_faint }, t.text_faint);
            }
        }
    }
    for (i, (loc, a)) in ops.iter().enumerate() {
        let x = x_of(*loc);
        let sel = dragging.or(selected) == Some(Marker::Opacity(i));
        let y = bar.top() - 10.0;
        let body = Rect::from_center_size(pos2(x, y), vec2(10.0, 10.0));
        let color = Color32::from_gray((a.clamp(0.0, 1.0) * 255.0) as u8);
        p.add(egui::Shape::convex_polygon(
            vec![pos2(x, bar.top() - 1.0), pos2(x + 5.0, body.bottom()), pos2(x - 5.0, body.bottom())],
            if sel { t.accent } else { t.text_dim },
            Stroke::NONE,
        ));
        p.rect_filled(body, 1.0, color);
        p.rect_stroke(body, 1.0, Stroke::new(1.0, if sel { t.accent } else { t.text_dim }), StrokeKind::Outside);
    }
    // Interaction.
    let hit = |pos: Pos2| -> Option<Marker> {
        if pos.y > bar.bottom() {
            if let Some(i) = sorted.iter().position(|(l, _)| (x_of(*l) - pos.x).abs() <= 6.0 && pos.y >= bar.bottom() + 3.0) {
                return Some(Marker::Color(i));
            }
            if !sink.midpoints() {
                return None;
            }
            return (0..sorted.len().saturating_sub(1))
                .find(|i| {
                    let (a, b) = (sorted[*i].0, sorted[*i + 1].0);
                    (x_of(a + (b - a) * mids.get(*i).copied().unwrap_or(0.5)) - pos.x).abs() <= 4.0 && b - a > 1e-3
                })
                .map(Marker::Mid);
        }
        if pos.y < bar.top() {
            return ops.iter().position(|(l, _)| (x_of(*l) - pos.x).abs() <= 6.0).map(Marker::Opacity);
        }
        None
    };
    if resp.drag_started()
        && let Some(pos) = ui.input(|i| i.pointer.press_origin())
        && let Some(m) = hit(pos)
    {
        ui.data_mut(|d| d.insert_temp(key, m));
        ui.data_mut(|d| d.insert_temp(selection_key, m));
    }
    let dragging: Option<Marker> = ui.data(|d| d.get_temp(key));
    if let (Some(m), Some(pos)) = (dragging, resp.interact_pointer_pos()) {
        let off = if pos.y > bar.bottom() + 28.0 || pos.y < bar.top() - 26.0 { 1.0 } else { 0.0 };
        let loc = loc_of(pos.x);
        let edit = match m {
            Marker::Color(i) if off > 0.0 && stops.len() > 2 => json!({"action": "delete", "index": i}),
            Marker::Color(i) => json!({"action": "move", "index": i, "location": loc}),
            Marker::Opacity(i) if off > 0.0 && ops.len() > 2 => json!({"action": "delete", "kind": "opacity", "index": i}),
            Marker::Opacity(i) => json!({"action": "move", "kind": "opacity", "index": i, "location": loc}),
            Marker::Mid(i) => {
                let (a, b) = (sorted.get(i).map_or(0.0, |s| s.0), sorted.get(i + 1).map_or(1.0, |s| s.0));
                json!({"action": "midpoint", "index": i, "location": ((loc - a) / (b - a).max(1e-4)).clamp(0.05, 0.95)})
            }
        };
        if resp.dragged() {
            match sink {
                Sink::Layer(id) => app.gradient.panel = Some((id, cmds::STOP, sink.params(edit))),
                Sink::Preset => app.gradient.preset_stop = Some(edit),
            }
        }
    }
    if resp.drag_stopped() {
        ui.data_mut(|d| d.remove::<Marker>(key));
        match sink {
            Sink::Layer(_) => {
                if let Some((_, cmd, p)) = app.gradient.panel.take() {
                    let selection = selection_after_edit(&f, &p, selected);
                    if app.run(cmd, p).is_ok()
                        && let Some(marker) = selection
                    {
                        ui.data_mut(|d| d.insert_temp(selection_key, marker));
                    }
                }
            }
            Sink::Preset => {
                if let Some(p) = app.gradient.preset_stop.take() {
                    let selection = selection_after_edit(&f, &p, selected);
                    if app.run(sink.cmd(), p).is_ok()
                        && let Some(marker) = selection
                    {
                        ui.data_mut(|d| d.insert_temp(selection_key, marker));
                    }
                }
            }
        }
        return;
    }
    if resp.double_clicked()
        && let Some(pos) = resp.interact_pointer_pos()
        && let Some(Marker::Color(i)) = hit(pos)
    {
        ui.data_mut(|d| d.insert_temp(selection_key, Marker::Color(i)));
        match sink {
            Sink::Layer(id) => edit_stop_color(app, id, i),
            Sink::Preset => preset_stop_color(app, i),
        }
        return;
    }
    if resp.clicked()
        && let Some(pos) = resp.interact_pointer_pos()
    {
        if let Some(marker) = hit(pos) {
            ui.data_mut(|d| d.insert_temp(selection_key, marker));
            return;
        }
        let loc = loc_of(pos.x);
        let opacity = pos.y < bar.center().y;
        let p = if opacity {
            sink.params(json!({"action": "add", "kind": "opacity", "location": loc}))
        } else {
            sink.params(json!({"action": "add", "location": loc}))
        };
        let selection = selection_after_edit(&f, &p, selected);
        if app.run(sink.cmd(), p).is_ok()
            && let Some(marker) = selection
        {
            ui.data_mut(|d| d.insert_temp(selection_key, marker));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app_with_gradient(style: &str) -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 200, "height": 120})).unwrap();
        app.ui.tool = Tool::Gradient;
        app.ui.tool_options.gradient_style = style.into();
        app
    }

    fn mods() -> egui::Modifiers {
        egui::Modifiers::NONE
    }

    fn drag(app: &mut PhotocraftApp, a: [f64; 2], b: [f64; 2]) {
        crate::canvas::tool_event(app, ToolEvent::Down { x: a[0], y: a[1], pressure: 1.0 }, mods());
        crate::canvas::tool_event(app, ToolEvent::Move { x: (a[0] + b[0]) / 2.0, y: (a[1] + b[1]) / 2.0, pressure: 1.0 }, mods());
        crate::canvas::tool_event(app, ToolEvent::Move { x: b[0], y: b[1], pressure: 1.0 }, mods());
        crate::canvas::tool_event(app, ToolEvent::Up { x: b[0], y: b[1] }, mods());
    }

    fn get(app: &mut PhotocraftApp) -> Value {
        app.run(cmds::GET, json!({})).unwrap()
    }

    #[test]
    fn thumbnails_show_the_gradient() {
        use photocraft_color::Color;
        // Black to white, left to right: the left edge is dark, the right light, top = bottom.
        let f = Fill::gradient(vec![(0.0, Color::BLACK), (1.0, Color::WHITE)], 0.0, 1.0, GradientStyle::Linear, false);
        let img = thumbnail_image(&f).unwrap();
        let n = img.size[0];
        let at = |x: usize, y: usize| img.pixels[y * n + x];
        assert!(at(0, n / 2).r() < 10 && at(n - 1, n / 2).r() > 245, "{:?} {:?}", at(0, n / 2), at(n - 1, n / 2));
        assert_eq!(at(n / 3, 0), at(n / 3, n - 1));
        // Reverse flips colour and opacity alike; a transparent stop lets the checkerboard through.
        let mut r = f.clone();
        if let Fill::Gradient { reverse, opacity_stops, .. } = &mut r {
            *reverse = true;
            *opacity_stops = vec![(0.0, 1.0), (1.0, 0.0)];
        }
        let img = thumbnail_image(&r).unwrap();
        let (left, right) = (img.pixels[n / 2 * n], img.pixels[n / 2 * n + n - 1]);
        assert!(left.a() < 10 && right.r() < 10 && right.a() > 245, "{left:?} {right:?}");
        // A radial thumbnail is centred.
        let rad = Fill::gradient(vec![(0.0, Color::BLACK), (1.0, Color::WHITE)], 0.0, 1.0, GradientStyle::Radial, false);
        let img = thumbnail_image(&rad).unwrap();
        assert!(img.pixels[n / 2 * n + n / 2].r() < img.pixels[0].r());
        assert!(thumbnail_image(&Fill::Solid(Color::WHITE)).is_none());
    }

    #[test]
    fn options_bar_swatch_opens_the_editor_and_reaches_the_gradients_panel() {
        use egui_kittest::{Harness, kittest::Queryable};
        let app = app_with_gradient("linear");
        let mut h = Harness::builder().with_size(vec2(440.0, 560.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                if preset_swatch(app, ui) {
                    app.ui.panels.gradient_editor = true;
                }
                let ctx = ui.ctx().clone();
                editor_window(app, &ctx);
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Pro);
        h.run_steps(2);
        // The swatch opens the Gradient Editor window (Photoshop's behaviour), not a popup.
        h.get_by_label("Gradient picker").click();
        h.run_steps(2);
        assert!(h.state().ui.panels.gradient_editor, "the swatch opened the editor window");
        // Quick preset picking lives in the editor: its Gradients button shows the Gradients panel.
        h.get_by_label("Gradients").click();
        h.run_steps(2);
        assert!(h.state().ui.panels.color, "the Gradients panel opened");
        assert_eq!(h.state().ui.dock_tabs.color, 2, "on the Gradients tab");
        assert_eq!(h.state().session.active().unwrap().doc.layers.len(), 1, "no document layers edited");
    }

    #[test]
    fn live_drag_creates_a_fill_layer_and_handles_edit_it() {
        let mut app = app_with_gradient("linear");
        drag(&mut app, [20.0, 60.0], [180.0, 60.0]);
        assert_eq!(app.session.active().unwrap().doc.layers.len(), 2, "a Gradient Fill layer");
        let g = get(&mut app);
        assert!((g["to"][0].as_f64().unwrap() - 180.0).abs() < 1e-3);
        // Drag the end handle: one history step.
        let before = app.session.active().unwrap().history.past_len();
        drag(&mut app, [180.0, 60.0], [150.0, 100.0]);
        let g = get(&mut app);
        assert!((g["to"][0].as_f64().unwrap() - 150.0).abs() < 1e-3 && (g["to"][1].as_f64().unwrap() - 100.0).abs() < 1e-3, "{g}");
        assert_eq!(app.session.active().unwrap().history.past_len(), before + 1);
        // Click on the line adds a stop; dragging it off the line deletes it.
        let (s, e) = ([20.0, 60.0], [150.0, 100.0]);
        // A third of the way (the midpoint diamond sits half-way).
        let mid = [s[0] + (e[0] - s[0]) / 3.0, s[1] + (e[1] - s[1]) / 3.0];
        crate::canvas::tool_event(&mut app, ToolEvent::Down { x: mid[0], y: mid[1], pressure: 1.0 }, mods());
        crate::canvas::tool_event(&mut app, ToolEvent::Up { x: mid[0], y: mid[1] }, mods());
        assert_eq!(get(&mut app)["stops"].as_array().unwrap().len(), 3);
        drag(&mut app, mid, [mid[0], mid[1] - 80.0]);
        assert_eq!(get(&mut app)["stops"].as_array().unwrap().len(), 2);
        // A drag away from the widget redraws the selected gradient (no new layer).
        drag(&mut app, [100.0, 10.0], [100.0, 110.0]);
        assert_eq!(app.session.active().unwrap().doc.layers.len(), 2);
        assert!((get(&mut app)["from"][1].as_f64().unwrap() - 10.0).abs() < 1e-3);
    }

    #[test]
    fn dragging_the_midpoint_diamond_moves_it() {
        let mut app = app_with_gradient("linear");
        drag(&mut app, [20.0, 60.0], [180.0, 60.0]);
        // The untouched midpoint sits at the segment centre (doc x = 100). Drag it to a quarter.
        drag(&mut app, [100.0, 60.0], [60.0, 60.0]);
        let g = get(&mut app);
        assert_eq!(g["stops"].as_array().map(Vec::len), Some(2), "a midpoint drag must not add a stop: {g}");
        let m = g["midpoints"][0].as_f64().unwrap_or(f64::NAN);
        assert!((m - 0.25).abs() < 1e-3, "the midpoint did not move: {g}");
    }

    #[test]
    fn shift_snaps_live_drags_to_45_degrees() {
        let mut app = app_with_gradient("linear");
        let shift = egui::Modifiers::SHIFT;
        crate::canvas::tool_event(&mut app, ToolEvent::Down { x: 20.0, y: 20.0, pressure: 1.0 }, shift);
        crate::canvas::tool_event(&mut app, ToolEvent::Move { x: 70.0, y: 62.0, pressure: 1.0 }, shift);
        crate::canvas::tool_event(&mut app, ToolEvent::Up { x: 90.0, y: 84.0 }, shift);
        let g = get(&mut app);
        let (fx, fy, tx, ty) = (g["from"][0].as_f64().unwrap(), g["from"][1].as_f64().unwrap(), g["to"][0].as_f64().unwrap(), g["to"][1].as_f64().unwrap());
        assert!(((tx - fx) - (ty - fy)).abs() < 1e-2 && tx > 80.0, "45°: {g}");
        // Dragging the end handle with ⇧ snaps too (horizontal here).
        crate::canvas::tool_event(&mut app, ToolEvent::Down { x: tx, y: ty, pressure: 1.0 }, shift);
        crate::canvas::tool_event(&mut app, ToolEvent::Up { x: 120.0, y: 26.0 }, shift);
        let g = get(&mut app);
        assert!((g["to"][1].as_f64().unwrap() - fy).abs() < 1e-2, "horizontal: {g}");
    }

    #[test]
    fn classic_mode_paints_pixels() {
        let mut app = app_with_gradient("linear");
        app.ui.tool_options.gradient_classic = true;
        drag(&mut app, [20.0, 60.0], [180.0, 60.0]);
        assert_eq!(app.session.active().unwrap().doc.layers.len(), 1, "no fill layer in classic mode");
    }

    #[test]
    fn classic_drag_preserves_preset_transparency_at_all_depths() {
        for depth in [8, 16, 32] {
            for reverse in [false, true] {
                let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
                app.run("file.new", json!({"width": 200, "height": 120, "depth": depth, "background": "transparent"})).unwrap();
                app.ui.tool = Tool::Gradient;
                app.ui.tool_options.gradient_classic = true;
                app.ui.tool_options.gradient_reverse = reverse;
                app.ui.tool_options.gradient_dither = false;
                app.ui.tool_options.fill_opacity = 50.0;
                app.session.tools.foreground = [1.0, 0.0, 0.0, 1.0];
                app.run("gradient.presets.select", json!({"preset": "Foreground to Transparent"})).unwrap();
                let before = app.session.active().unwrap().history.past_len();
                drag(&mut app, [20.0, 60.0], [180.0, 60.0]);
                let st = app.session.active().unwrap();
                let surface = st.doc.layers[0].surface().unwrap();
                for x in [0, 100, 199] {
                    let u = ((x as f32 - 20.0) / 160.0).clamp(0.0, 1.0);
                    let expected = 0.5 * if reverse { u } else { 1.0 - u };
                    let px = surface.pixel(x, 60);
                    assert!((px[3] - expected).abs() < 0.01, "depth {depth}, reverse {reverse}, x {x}: {px:?}");
                    if expected > 0.01 {
                        assert!(px[0] > 0.99 && px[1] < 0.01 && px[2] < 0.01, "preset colour: {px:?}");
                    }
                }
                assert_eq!(st.history.past_len(), before + 1);
                app.run("edit.undo", json!({})).unwrap();
                assert_eq!(app.session.active().unwrap().doc.layers[0].surface().unwrap().pixel(100, 60)[3], 0.0);
                app.run("gradient.presets.select", json!({"stops": [[0, "#00ff00"], [1, "#00ff00"]], "transparency": [[0, 60], [1, 20]]})).unwrap();
                drag(&mut app, [20.0, 60.0], [180.0, 60.0]);
                let px = app.session.active().unwrap().doc.layers[0].surface().unwrap().pixel(100, 60);
                assert!((px[3] - 0.2).abs() < 0.01 && px[1] > 0.99 && px[0] < 0.01, "custom semi-transparent colour: {px:?}");
            }
        }
    }

    #[test]
    fn sidebar_transparent_preset_reaches_classic_canvas_over_existing_art() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut app = app_with_gradient("linear");
        app.run("paint.gradient", json!({"from": [0, 0], "to": [199, 0], "colors": ["#ff0000", "#ff0000"]})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        app.ui.tool_options.gradient_classic = true;
        app.ui.tool_options.gradient_dither = false;
        let mut h = Harness::builder().with_size(vec2(420.0, 600.0)).build_ui_state(|ui, app| crate::preset_panels::gradients_panel(app, ui), app);
        h.get_by_label("Foreground to Transparent").click();
        h.run();
        assert_eq!(h.state().session.presets.gradient.name, "Foreground to Transparent");
        drag(h.state_mut(), [20.0, 60.0], [180.0, 60.0]);
        let st = h.state().session.active().unwrap();
        assert_eq!(st.doc.layers.len(), 2);
        let surface = st.doc.layers[1].surface().unwrap();
        assert!(surface.pixel(0, 60)[3] > 0.99);
        assert!((surface.pixel(100, 60)[3] - 0.5).abs() < 0.01);
        assert!(surface.pixel(199, 60)[3] < 0.01, "transparent end reveals existing art");
        assert_eq!(st.doc.layers[0].surface().unwrap().pixel(100, 60), vec![1.0, 0.0, 0.0, 1.0], "existing art is unchanged");
    }

    #[test]
    fn gradient_blend_mode_reaches_live_and_classic_drags() {
        use photocraft_color::BlendMode;
        for mode in [BlendMode::Difference, BlendMode::Multiply, BlendMode::Screen, BlendMode::Exclusion] {
            let mut live = app_with_gradient("linear");
            live.ui.tool_options.gradient_blend_mode = mode;
            drag(&mut live, [20.0, 60.0], [180.0, 60.0]);
            let layer = live.session.active().unwrap().doc.layers.last().unwrap();
            assert_eq!(layer.blend, mode, "live {mode:?}");

            let mut classic = app_with_gradient("linear");
            classic.ui.tool_options.gradient_classic = true;
            classic.ui.tool_options.gradient_blend_mode = mode;
            classic.session.tools.foreground = [1.0, 0.0, 0.0, 1.0];
            classic.session.tools.background = [1.0, 0.0, 0.0, 1.0];
            drag(&mut classic, [20.0, 60.0], [180.0, 60.0]);
            let st = classic.session.active().unwrap();
            let px = st.doc.layers[0].surface().unwrap().pixel(60, 60);
            let expected = match mode {
                BlendMode::Difference | BlendMode::Exclusion => [0.0, 1.0, 1.0],
                BlendMode::Multiply => [1.0, 0.0, 0.0],
                BlendMode::Screen => [1.0, 1.0, 1.0],
                _ => [1.0, 1.0, 1.0],
            };
            for (actual, want) in px.iter().zip(expected) {
                assert!((actual - want).abs() < 0.02, "classic {mode:?}: {px:?}");
            }
        }
    }

    #[test]
    fn changing_live_gradient_mode_updates_selected_fill_layer() {
        use photocraft_color::BlendMode;
        let mut app = app_with_gradient("linear");
        drag(&mut app, [20.0, 60.0], [180.0, 60.0]);
        let before = app.ui.tool_options.clone();
        app.ui.tool_options.gradient_blend_mode = BlendMode::Difference;
        options_changed(&mut app, &before);
        let layer = app.session.active().unwrap().doc.layers.last().unwrap();
        assert_eq!(layer.blend, BlendMode::Difference);
    }

    #[test]
    fn changing_live_gradient_opacity_updates_selected_fill_layer() {
        let mut app = app_with_gradient("linear");
        drag(&mut app, [20.0, 60.0], [180.0, 60.0]);
        let past = app.session.active().unwrap().history.past_len();
        // A scrub is live on every frame and one undo step.
        for pct in [70.0, 55.0, 40.0] {
            app.ui.tool_options.fill_opacity = pct;
            opacity_changed(&mut app, Some(7));
        }
        opacity_changed(&mut app, None);
        let st = app.session.active().unwrap();
        assert!((st.doc.layers.last().unwrap().opacity - 0.4).abs() < 1e-6);
        assert_eq!(st.history.past_len(), past + 1, "one undo step, none for an unchanged value");
        // Classic mode paints pixels: the options bar leaves the layer alone.
        app.ui.tool_options.gradient_classic = true;
        app.ui.tool_options.fill_opacity = 80.0;
        opacity_changed(&mut app, None);
        assert!((app.session.active().unwrap().doc.layers.last().unwrap().opacity - 0.4).abs() < 1e-6);
    }

    #[test]
    fn preview_shows_the_drag_before_commit() {
        let mut app = app_with_gradient("radial");
        crate::canvas::tool_event(&mut app, ToolEvent::Down { x: 100.0, y: 60.0, pressure: 1.0 }, mods());
        crate::canvas::tool_event(&mut app, ToolEvent::Move { x: 140.0, y: 60.0, pressure: 1.0 }, mods());
        let (doc, key) = display_doc(&mut app, 0).expect("a preview while drawing");
        assert!(key & TAG != 0);
        assert!(doc.layer(PREVIEW_LAYER).is_some(), "the new layer previews");
        assert_eq!(app.session.active().unwrap().doc.layers.len(), 1, "nothing committed yet");
        crate::canvas::tool_event(&mut app, ToolEvent::Up { x: 140.0, y: 60.0 }, mods());
        assert!(display_doc(&mut app, 0).is_none());
    }

    /// Regression: the options bar's gradient swatch could not open anything. In live mode it
    /// sensed hover only, so `clicked()` was never true; in classic mode it sensed clicks and
    /// promised "Click to edit the gradient" but never read the response. Both now open the
    /// Gradient Editor window.
    #[test]
    fn the_options_bar_swatch_opens_the_gradient_editor_in_both_modes() {
        use egui_kittest::kittest::Queryable;

        for classic in [false, true] {
            let mut app = app_with_gradient("linear");
            app.ui.tool_options.gradient_classic = classic;
            let mut h = egui_kittest::Harness::builder().with_size(vec2(1400.0, 900.0)).build_ui_state(
                |ui, app: &mut PhotocraftApp| {
                    crate::panels::options_bar(app, ui);
                    let ctx = ui.ctx().clone();
                    editor_window(app, &ctx);
                },
                app,
            );
            PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
            h.run_steps(4);
            assert!(!h.state().ui.panels.gradient_editor, "classic={classic}: starts closed");
            h.get_by_label("Gradient picker").click();
            h.run_steps(3);
            assert!(h.state().ui.panels.gradient_editor, "classic={classic}: the swatch opened the editor");
        }
    }

    /// The editor itself renders in both states: with a gradient fill layer selected it shows the
    /// full stops editor, and with none selected it must still draw (and say what to do) instead of
    /// opening an empty window or panicking.
    #[test]
    fn the_editor_window_renders_with_and_without_a_gradient_layer() {
        let ctx = egui::Context::default();
        PhotocraftApp::setup_context(&ctx, crate::theme::ThemeKind::ALL[0]);
        // `run_ui` hands back a `FullOutput` whose texture deltas the caller has to consume.
        let run = |app: &mut PhotocraftApp| {
            let mut out = ctx.run_ui(egui::RawInput::default(), |ui| editor_window(app, ui.ctx()));
            out.textures_delta.clear();
        };

        // Nothing selected.
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.panels.gradient_editor = true;
        run(&mut app);
        assert!(app.ui.panels.gradient_editor, "the window stays open");

        // A committed live gradient fill layer: `selected_gradient` finds it, so the full editor
        // (combined stops and Gradient Options) is what gets built.
        let mut app = app_with_gradient("linear");
        drag(&mut app, [10.0, 10.0], [150.0, 90.0]);
        assert!(selected_gradient(&app).is_some(), "the drag left a gradient fill layer selected");
        app.ui.panels.gradient_editor = true;
        run(&mut app);
        assert!(app.ui.panels.gradient_editor);

        // Closed: the window draws nothing and leaves the flag alone.
        app.ui.panels.gradient_editor = false;
        run(&mut app);
        assert!(!app.ui.panels.gradient_editor);
    }

    /// With no gradient fill layer selected the editor edits the Gradient tool's current gradient
    /// in place (Photoshop's editor edits the tool gradient): a click on the strip adds a stop to
    /// it instead of only pointing at the Gradients panel's presets.
    #[test]
    fn the_editor_edits_the_current_gradient_without_a_layer() {
        use egui_kittest::kittest::Queryable;

        let app = app_with_gradient("linear");
        let before = app.session.presets.gradient.stops.len();
        let mut h = egui_kittest::Harness::builder().with_size(vec2(1400.0, 900.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                let ctx = ui.ctx().clone();
                editor_window(app, &ctx);
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
        // Open the window once the fonts are bound (the build's first frame runs without them).
        h.run_steps(2);
        h.state_mut().ui.panels.gradient_editor = true;
        h.run_steps(4);
        h.get_by_label("Gradient stops").click();
        h.run_steps(3);
        assert_eq!(h.state().session.presets.gradient.stops.len(), before + 1, "the click added a stop to the current gradient");
    }

    /// The editor's Save As Preset button files the current gradient into the Gradients panel
    /// under a fresh "Custom" name.
    #[test]
    fn the_editor_saves_the_current_gradient_as_a_preset() {
        use egui_kittest::kittest::Queryable;

        let app = app_with_gradient("linear");
        let has_custom = |s: &PhotocraftApp| s.session.presets.gradients.iter().any(|g| g.items.iter().any(|p| p.name == "Custom"));
        let mut h = egui_kittest::Harness::builder().with_size(vec2(1400.0, 900.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                let ctx = ui.ctx().clone();
                editor_window(app, &ctx);
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
        h.run_steps(2);
        h.state_mut().ui.panels.gradient_editor = true;
        h.run_steps(4);
        assert!(!has_custom(h.state()), "no Custom preset yet");
        h.get_by_label("Save As Preset").click();
        h.run_steps(3);
        assert!(has_custom(h.state()), "the button saved the current gradient as a preset");
    }

    /// A hidden gradient fill layer is still editable from the editor (the Properties panel edits
    /// it whether or not it is drawn), so the window must not require visibility.
    #[test]
    fn the_editor_finds_a_hidden_gradient_layer() {
        let mut app = app_with_gradient("linear");
        drag(&mut app, [10.0, 10.0], [150.0, 90.0]);
        let id = app.session.active().unwrap().active_layer.unwrap();
        app.run("layer.setProps", json!({"layer": id.0, "visible": false})).unwrap();
        assert!(active_gradient(&app).is_none(), "the canvas widget skips a hidden layer");
        assert!(selected_gradient(&app).is_some(), "the editor still edits it");
    }

    /// #4577: the tessellator snaps the checkerboard's rects to whole pixels but leaves the ramp's
    /// mesh where it was laid out, so at a half-pixel offset the ramp stopped a pixel short of the
    /// checker and a row of it showed under the gradient. The ramp must cover the checker exactly.
    #[test]
    fn the_ramp_covers_its_checkerboard_at_half_pixel_offsets() {
        let ctx = egui::Context::default();
        for ppp in [1.0, 2.0] {
            ctx.set_pixels_per_point(ppp);
            let r = Rect::from_min_size(pos2(10.0 + 0.5 / ppp, 7.0 + 0.5 / ppp), vec2(96.0, 20.0));
            // The new scale applies from the next frame.
            ctx.run_ui(egui::RawInput::default(), |_| {}).textures_delta.clear();
            let mut painted_at = 0.0;
            let mut out = ctx.run_ui(egui::RawInput::default(), |ui| {
                painted_at = ui.ctx().pixels_per_point();
                paint_ramp(ui.painter(), r, |u| [u, u, u, 1.0]);
            });
            out.textures_delta.clear();
            assert_eq!(painted_at, ppp);
            let checker = out
                .shapes
                .iter()
                .find_map(|s| match &s.shape {
                    egui::Shape::Rect(rs) if rs.fill == Color32::from_gray(250) => Some(rs.rect),
                    _ => None,
                })
                .expect("the checkerboard's base rect");
            let ramp = out
                .shapes
                .iter()
                .find_map(|s| match &s.shape {
                    egui::Shape::Mesh(m) => Some(m.calc_bounds()),
                    _ => None,
                })
                .expect("the ramp mesh");
            assert_eq!(ramp, checker.round_to_pixels(ppp), "{ppp}x: the ramp covers every pixel of the checker");
        }
    }
}
