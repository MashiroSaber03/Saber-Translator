//! Vector tools: Shape tools (U), Pen (P), Path Selection (A), path overlays and the Paths panel.
//! Direct Selection (A) lives in `direct_select`.
//! All edits go through the engine's `shape.*` / `path.*` commands.

use egui::{Color32, Pos2, Rect, Sense, Stroke, pos2, vec2};
use photocraft_doc::vector::Path;
use photocraft_doc::{Document, LayerContent};
use photocraft_geom::Affine;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::canvas::ViewXform;
use crate::state::{Tool, ToolOptions};
use crate::theme::Tokens;

/// Pen tool path under construction: knots as [anchor, in, out] (document px).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PenPath {
    pub knots: Vec<[[f64; 2]; 3]>,
    /// Indices of cusp knots whose incoming and outgoing handles are not linked.
    #[serde(default)]
    pub unlinked: Vec<usize>,
    #[serde(skip)]
    pub dragging: bool,
    /// Re-dragging the final anchor changes its outgoing control only; the incoming curve stays put.
    #[serde(skip)]
    pub adjusting_last: bool,
    /// Drawn with the Curvature Pen: handles follow the anchors (`curvature_refresh`) and
    /// `unlinked` lists its corner points.
    #[serde(default)]
    pub curvature: bool,
    /// Curvature Pen: the anchor being dragged.
    #[serde(skip)]
    pub moving: Option<usize>,
}

impl PenPath {
    /// Does `tool` keep drawing this path? Another tool finishes it.
    pub fn drawn_with(&self, tool: Tool) -> bool {
        tool == if self.curvature { Tool::CurvaturePen } else { Tool::Pen }
    }
}

pub fn is_shape_tool(t: Tool) -> bool {
    matches!(t, Tool::Rectangle | Tool::EllipseShape | Tool::Triangle | Tool::Polygon | Tool::Line | Tool::CustomShape)
}

fn rgb32(c: [f32; 4]) -> Color32 {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    Color32::from_rgb(b(c[0]), b(c[1]), b(c[2]))
}

fn hex(c: [f32; 4]) -> String {
    let c = rgb32(c);
    format!("#{:02x}{:02x}{:02x}", c.r(), c.g(), c.b())
}

fn stroke_param(app: &PhotocraftApp) -> Value {
    let o = &app.ui.tool_options;
    if o.stroke_width <= 0.0 {
        return Value::Null;
    }
    let mut p = o.shape_stroke.params();
    p["width"] = json!(o.stroke_width);
    p["color"] = json!(app.session.tools.background);
    p
}

/// Geometry of a Shape-tool drag as `shape.create` params, without fill and stroke (None when too
/// small): ⇧ constrains proportions, ⌥ draws from the centre. Custom shapes give their box as a `rect`.
fn shape_geometry(o: &ToolOptions, tool: Tool, start: [f64; 2], end: [f64; 2], mods: egui::Modifiers) -> Option<Value> {
    Some(if tool == Tool::Line {
        let (mut dx, mut dy) = (end[0] - start[0], end[1] - start[1]);
        if mods.shift {
            // Snap to 45°.
            let a = (dy.atan2(dx) / std::f64::consts::FRAC_PI_4).round() * std::f64::consts::FRAC_PI_4;
            let len = dx.hypot(dy);
            (dx, dy) = (len * a.cos(), len * a.sin());
        }
        if dx.hypot(dy) < 1.0 {
            return None;
        }
        json!({"kind": "line", "from": start, "to": [start[0] + dx, start[1] + dy], "weight": o.line_weight.max(1.0)})
    } else {
        let (mut w, mut h) = (end[0] - start[0], end[1] - start[1]);
        if mods.shift {
            let m = w.abs().max(h.abs());
            (w, h) = (m * w.signum(), m * h.signum());
        }
        let (x0, y0) = if mods.alt { (start[0] - w, start[1] - h) } else { (start[0], start[1]) };
        let (w, h) = if mods.alt { (w * 2.0, h * 2.0) } else { (w, h) };
        let rect = [x0.min(x0 + w).round(), y0.min(y0 + h).round(), w.abs().round(), h.abs().round()];
        if rect[2] < 1.0 || rect[3] < 1.0 {
            return None;
        }
        match tool {
            Tool::Rectangle if o.corner_radius > 0.0 => json!({"kind": "roundedRect", "rect": rect, "radii": vec![o.corner_radius; 4]}),
            Tool::Rectangle | Tool::CustomShape => json!({"kind": "rect", "rect": rect}),
            Tool::EllipseShape => json!({"kind": "ellipse", "rect": rect}),
            Tool::Triangle => json!({"kind": "polygon", "rect": rect, "sides": 3}),
            _ => json!({"kind": "polygon", "rect": rect, "sides": o.polygon_sides.max(3)}),
        }
    })
}

/// Finish a Shape-tool drag: ⇧ constrains proportions, ⌥ draws from the centre.
pub fn finish_shape(app: &mut PhotocraftApp, tool: Tool, start: [f64; 2], end: [f64; 2], mods: egui::Modifiers) {
    let Some(mut p) = shape_geometry(&app.ui.tool_options, tool, start, end, mods) else { return };
    if tool == Tool::CustomShape {
        // ⇧ keeps the shape's proportions (the rect is already squared).
        if let Ok(rect) = serde_json::from_value(p["rect"].take()) {
            let (fill, stroke) = (fill_param(app), stroke_param(app));
            crate::preset_panels::finish_custom_shape(app, rect, mods.shift, fill, stroke);
        }
        return;
    }
    if let Err(e) = create_shape(app, p) {
        app.ui.status = e;
        app.ui.status_error = true;
    }
}

fn fill_param(app: &PhotocraftApp) -> Value {
    if app.ui.tool_options.shape_fill { json!(app.session.tools.foreground) } else { Value::Null }
}

/// Create a shape layer from `shape.create` geometry with the options bar's fill and stroke: the
/// end of a drag, and the Create Rectangle / Ellipse / … dialogs (`shape_dialog`).
pub fn create_shape(app: &mut PhotocraftApp, mut geometry: Value) -> Result<Value, String> {
    geometry["fill"] = fill_param(app);
    geometry["stroke"] = stroke_param(app);
    app.run("shape.create", geometry)
}

/// The preview and command both use the full float tool colours, including transparent paint.
fn shape_preview(app: &PhotocraftApp, tool: Tool, start: [f64; 2], end: [f64; 2], mods: egui::Modifiers) -> Option<photocraft_doc::ShapeLayer> {
    let o = &app.ui.tool_options;
    let geometry = shape_geometry(o, tool, start, end, mods)?;
    let path = if tool == Tool::CustomShape {
        let rect = geometry.get("rect").and_then(|r| serde_json::from_value::<[f64; 4]>(r.clone()).ok())?;
        let groups = photocraft_engine::presets::shapes::all_groups(&app.session);
        let shape = groups.iter().flat_map(|g| &g.items).find(|s| s.name == app.ui.presets_ui.shape())?;
        photocraft_engine::presets::shapes::fit(&shape.path, rect, mods.shift)
    } else {
        photocraft_engine::vector_cmds::shape_path(&geometry).ok()?
    };
    Some(photocraft_doc::ShapeLayer {
        path,
        fill: o.shape_fill.then(|| solid_paint(app.session.tools.foreground)),
        stroke: (o.stroke_width > 0.0).then(|| o.shape_stroke.stroke(o.stroke_width, solid_paint(app.session.tools.background))),
        ..Default::default()
    })
}

/// Shape-tool drag preview using the same path and vector stroke geometry as the commit.
pub fn draw_shape_preview(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform, tool: Tool, start: [f64; 2], end: [f64; 2], mods: egui::Modifiers) {
    let Some(mut shape) = shape_preview(app, tool, start, end, mods) else { return };
    let path = shape.path.clone();
    let origin = xf.to_screen(0.0, 0.0);
    let x = xf.to_screen(1.0, 0.0) - origin;
    let y = xf.to_screen(0.0, 1.0) - origin;
    let transform = Affine { m: [x.x as f64, x.y as f64, y.x as f64, y.y as f64, origin.x as f64, origin.y as f64] };
    shape.path = shape.path.transform(&transform);
    if let Some(stroke) = &mut shape.stroke {
        stroke.width *= xf.zoom;
    }
    // This lightweight screen overlay retains the existing display-colour behaviour; document
    // rendering still goes through its format/profile path when the shape is committed.
    crate::shape_stroke_ui::paint_shape_preview(painter, &shape);
    let accent = Tokens::get(painter.ctx()).accent;
    for (pts, _) in path_lines(&path, &|q| xf.to_screen(q[0] as f32, q[1] as f32)) {
        // The accent path over a dark halo: visible on any pixels (#172).
        painter.extend(crate::tool_feedback::contrast_path(pts, true, accent));
    }
}

// ---------------------------------------------------------------------------------------------
// Pen

fn pen_to_json(pen: &PenPath, closed: bool) -> Value {
    let folded = fold_close(pen);
    let (pen, closed) = match &folded {
        Some(p) => (p, true),
        None => (pen, closed),
    };
    json!({"subpaths": [{"closed": closed, "knots": pen.knots.iter().enumerate().map(|(i, k)| json!({"anchor": k[0], "in": k[1], "out": k[2], "smooth": !pen.unlinked.contains(&i) && (k[1] != k[0] || k[2] != k[0])})).collect::<Vec<_>>()}]})
}

/// Is `(x, y)` on the first anchor of a Pen path that a click there would close (two anchors or
/// more)? `tol` is the anchors' hit radius in document pixels.
fn on_closing_anchor(pen: &PenPath, x: f64, y: f64, tol: f64) -> bool {
    pen.knots.len() >= 2 && pen.knots.first().is_some_and(|k| (k[0][0] - x).hypot(k[0][1] - y) < tol)
}

/// Would a click at `(x, y)` with `tool` close the path being drawn: the Pen over the first
/// anchor, or the Curvature Pen over the first of three anchors or more? The canvas shows the
/// close-path pointer there, as Photoshop does (#4596).
pub fn closes_path_at(app: &PhotocraftApp, tool: Tool, x: f64, y: f64) -> bool {
    let Some(pen) = app.ui.pen.as_ref().filter(|p| !p.dragging && p.moving.is_none()) else { return false };
    match tool {
        Tool::Pen => !pen.curvature && on_closing_anchor(pen, x, y, 6.0 / app.point_zoom().max(0.01) as f64),
        Tool::CurvaturePen => pen.curvature && pen.knots.len() >= 3 && curvature_hit(app, x, y) == Some(0),
        _ => false,
    }
}

/// Pen press: close on the first anchor, reshape the last anchor's outgoing handle,
/// or add a new anchor (dragging a new anchor pulls symmetrical handles).
pub fn pen_down(app: &mut PhotocraftApp, x: f64, y: f64) {
    let tol = 6.0 / app.point_zoom().max(0.01) as f64;
    let pen = app.ui.pen.get_or_insert_with(PenPath::default);
    if let Some(first) = pen.knots.first().map(|k| k[0])
        && on_closing_anchor(pen, x, y, tol)
    {
        // Pressing the first anchor closes the path on release. Until then a stand-in for it
        // takes the drag, which pulls out that anchor's handles (close-drag, #3306).
        pen.knots.push([first; 3]);
        pen.dragging = true;
        pen.adjusting_last = false;
        return;
    }
    // Clicking the last anchor breaks its outgoing handle without throwing away the incoming
    // curve. Dragging from that anchor then sets the outgoing handle independently (#1482).
    // This must precede adding a point, or a re-click creates a zero-length segment.
    if let Some((i, last)) = pen.knots.len().checked_sub(1).zip(pen.knots.last_mut())
        && (last[0][0] - x).hypot(last[0][1] - y) < tol
    {
        last[2] = last[0];
        if !pen.unlinked.contains(&i) {
            pen.unlinked.push(i);
        }
        pen.dragging = true;
        pen.adjusting_last = true;
        return;
    }
    pen.knots.push([[x, y]; 3]);
    pen.dragging = true;
    pen.adjusting_last = false;
}

pub fn pen_move(app: &mut PhotocraftApp, x: f64, y: f64) {
    if let Some(pen) = app.ui.pen.as_mut()
        && pen.dragging
    {
        let adjusting_last = pen.adjusting_last;
        if let Some(k) = pen.knots.last_mut() {
            let a = k[0];
            k[2] = [x, y];
            if !adjusting_last {
                k[1] = [2.0 * a[0] - x, 2.0 * a[1] - y];
            }
        }
    }
}

/// Undo the most recently placed Pen anchor while its path is still in progress.
/// These knots are editor-only gesture state, not document history until Enter or
/// clicking the first anchor commits the path. Therefore Cmd/Ctrl+Z must consume one
/// knot rather than undoing an unrelated, already committed document operation.
pub fn pen_undo_last_point(app: &mut PhotocraftApp) -> bool {
    let Some(pen) = app.ui.pen.as_mut() else { return false };
    if pen.knots.pop().is_none() {
        return false;
    }
    let n = pen.knots.len();
    let ds = &mut app.ui.direct_selection;
    if ds.target == Some(crate::direct_select::PathRef::Pen) {
        // A ⌘/Ctrl selection on the path being drawn loses the removed anchor.
        ds.anchors.retain(|a| a[1] < n);
        ds.segment = ds.segment.filter(|s| s[1] + 1 < n);
    }
    pen.unlinked.retain(|&i| i < pen.knots.len());
    if pen.curvature {
        pen.moving = None;
        curvature_refresh(pen, false);
    }
    pen.dragging = false;
    if pen.knots.is_empty() {
        app.ui.pen = None;
    }
    true
}

pub fn pen_up(app: &mut PhotocraftApp) {
    let jitter = 2.0 / f64::from(app.point_zoom().max(0.01));
    let Some(pen) = app.ui.pen.as_mut() else { return };
    pen.dragging = false;
    pen.adjusting_last = false;
    if closing(pen) {
        // A click on the first anchor closes the path and leaves that anchor's handles alone.
        if let Some(k) = pen.knots.last_mut()
            && (k[2][0] - k[0][0]).hypot(k[2][1] - k[0][1]) < jitter
        {
            k[1] = k[0];
            k[2] = k[0];
        }
        pen_commit(app, true);
    }
}

/// ⌥/Alt during a Pen drag breaks the pair of handles being pulled out: from then on only the
/// outgoing handle follows the pointer and the incoming curve stays as it is (#3306).
pub fn pen_break_handle(app: &mut PhotocraftApp) {
    let Some(pen) = app.ui.pen.as_mut().filter(|p| p.dragging && !p.adjusting_last) else { return };
    let Some(i) = pen.knots.len().checked_sub(1) else { return };
    if !pen.unlinked.contains(&i) {
        pen.unlinked.push(i);
    }
    pen.adjusting_last = true;
}

/// ⌥/Alt-press on the last anchor's outgoing handle while drawing grabs that handle alone: the
/// drag turns it and the incoming curve stays put (#3306). False when the press misses it.
pub fn pen_grab_handle(app: &mut PhotocraftApp, x: f64, y: f64) -> bool {
    let tol = 6.0 / f64::from(app.point_zoom().max(0.01));
    let Some(pen) = app.ui.pen.as_mut() else { return false };
    let Some((i, k)) = pen.knots.len().checked_sub(1).zip(pen.knots.last()) else { return false };
    let dist = |p: [f64; 2]| (p[0] - x).hypot(p[1] - y);
    if k[2] == k[0] || dist(k[2]) >= tol || dist(k[2]) >= dist(k[0]) {
        return false;
    }
    if !pen.unlinked.contains(&i) {
        pen.unlinked.push(i);
    }
    pen.dragging = true;
    pen.adjusting_last = true;
    true
}

/// Would a press at `at` close the Pen path: is the pointer on the first anchor of a path that
/// already has a segment, and no drag is under way? The canvas then shows the close cue (#4193).
pub(crate) fn over_first_anchor(pen: &PenPath, at: [f64; 2], tol: f64) -> bool {
    !pen.dragging && pen.moving.is_none() && pen.knots.len() >= 2 && pen.knots.first().is_some_and(|k| (k[0][0] - at[0]).hypot(k[0][1] - at[1]) < tol)
}

/// Is a close-drag in progress: does the path end in the stand-in for its first anchor?
fn closing(pen: &PenPath) -> bool {
    pen.knots.len() >= 3 && pen.knots.first().map(|k| k[0]) == pen.knots.last().map(|k| k[0])
}

/// The path with a close-drag's stand-in folded into the first knot, which takes the dragged
/// handles (smooth, or a cusp after ⌥/Alt); `None` when no close-drag is in progress.
fn fold_close(pen: &PenPath) -> Option<PenPath> {
    if !closing(pen) {
        return None;
    }
    let mut p = pen.clone();
    let stand_in = p.knots.pop()?;
    let broken = p.unlinked.contains(&p.knots.len());
    p.unlinked.retain(|&i| i < p.knots.len());
    if stand_in[1] != stand_in[0] || stand_in[2] != stand_in[0] {
        let first = p.knots.first_mut()?;
        first[1] = stand_in[1];
        first[2] = stand_in[2];
        p.unlinked.retain(|&i| i != 0);
        if broken {
            p.unlinked.push(0);
        }
    }
    Some(p)
}

/// The Pen path being drawn as a document path (one open subpath), so ⌘/Ctrl (Direct Selection)
/// can edit it in place while drawing (#4022).
pub(crate) fn pen_path(pen: &PenPath) -> Path {
    let pt = |q: [f64; 2]| photocraft_geom::Point::new(q[0], q[1]);
    let knots = pen
        .knots
        .iter()
        .enumerate()
        .map(|(i, k)| photocraft_doc::Knot {
            anchor: pt(k[0]),
            in_ctrl: pt(k[1]),
            out_ctrl: pt(k[2]),
            smooth: !pen.unlinked.contains(&i) && (k[1] != k[0] || k[2] != k[0]),
        })
        .collect();
    Path::new(vec![photocraft_doc::vector::Subpath { closed: false, knots, op: Default::default() }])
}

/// The Pen path being drawn becomes `path`'s first subpath, edited with ⌘/Ctrl (Direct Selection);
/// drawing carries on from its last anchor. A Curvature Pen path keeps following its anchors.
pub(crate) fn pen_replace(app: &mut PhotocraftApp, path: &Path) {
    let Some(pen) = app.ui.pen.as_mut() else { return };
    let knots = path.subpaths.first().map(|s| s.knots.as_slice()).unwrap_or_default();
    pen.knots = knots.iter().map(knot_points).collect();
    pen.unlinked = knots.iter().enumerate().filter(|(_, k)| !k.smooth).map(|(i, _)| i).collect();
    pen.dragging = false;
    pen.adjusting_last = false;
    pen.moving = None;
    if pen.curvature {
        curvature_refresh(pen, false);
    }
    if pen.knots.is_empty() {
        app.ui.pen = None;
    }
}

/// Finish the pen path: a work path (Path mode) or a new shape layer (Shape mode).
pub fn pen_commit(app: &mut PhotocraftApp, closed: bool) {
    let Some(mut pen) = app.ui.pen.take() else { return };
    if app.ui.direct_selection.target == Some(crate::direct_select::PathRef::Pen) {
        // Its anchors are no longer the Pen's: the committed path is another path.
        app.ui.direct_selection = Default::default();
    }
    if pen.curvature {
        curvature_refresh(&mut pen, closed);
    }
    commit_path(app, &pen, closed);
}

/// Commit a finished Pen, Freeform Pen or Curvature Pen path, as one History step: a new shape
/// layer (Shape mode), else a subpath of the targeted vector mask, else the work path.
fn commit_path(app: &mut PhotocraftApp, pen: &PenPath, closed: bool) {
    if pen.knots.len() < 2 {
        return;
    }
    let path = pen_to_json(pen, closed);
    let r = if app.ui.tool_options.vector_mode == "shape" {
        let fill = if closed { fill_param(app) } else { Value::Null };
        let stroke = if closed {
            stroke_param(app)
        } else {
            let mut stroke = app.ui.tool_options.shape_stroke.params();
            stroke["width"] = json!(app.ui.tool_options.stroke_width.max(1.0));
            stroke["color"] = json!(app.session.tools.foreground);
            stroke
        };
        app.run("shape.create", json!({"kind": "path", "path": path, "fill": fill, "stroke": stroke}))
    } else if let Some((id, existing)) = targeted_vector_mask(app) {
        // A targeted vector mask takes the new subpath (#196), as in Photoshop.
        let mut p = photocraft_engine::vector_cmds::path_json(&existing);
        if let (Some(subs), Some(new)) = (p.get_mut("subpaths").and_then(Value::as_array_mut), path.get("subpaths").and_then(Value::as_array)) {
            subs.extend(new.iter().cloned());
        }
        app.run("layer.vectorMask.edit", json!({"layer": id, "path": p}))
    } else {
        // Photoshop adds the new subpath to the path selected in the Paths panel, which is the
        // Work Path right after drawing one, so path after path builds up there (#4721). With no
        // path selected, a new Work Path replaces the old one. The drawn path ends up selected.
        let target = app.ui.selected_path.clone().filter(|n| app.session.active().is_some_and(|st| path_exists(&st.doc, n)));
        let name = target.clone().unwrap_or_else(|| "work".into());
        let mut params = json!({"name": name, "path": path});
        if target.is_some() {
            params["op"] = json!("combine");
        }
        app.run("path.set", params).inspect(|_| app.ui.selected_path = Some(name))
    };
    if let Err(e) = r {
        app.ui.status = e;
        app.ui.status_error = true;
    }
}

/// Does `doc` have the path the Paths panel calls `name`: "work", or a saved path's name?
fn path_exists(doc: &Document, name: &str) -> bool {
    if name == "work" { doc.work_path.is_some() } else { doc.paths.iter().any(|p| p.name == name) }
}

// ---------------------------------------------------------------------------------------------
// Freeform Pen and Curvature Pen

/// Freeform Pen release: fit the drag's trace within the options bar's Curve Fit and commit it
/// as the Pen does. Releasing on the start point closes the path; a click draws nothing. The
/// points at indices `straight` were ⌥-clicked: the segment into each one stays straight.
pub fn freeform_finish(app: &mut PhotocraftApp, points: &[[f64; 3]], straight: &[usize]) {
    let trace: Vec<[f64; 2]> = points.iter().map(|p| [p[0], p[1]]).collect();
    let (Some(first), Some(last)) = (trace.first(), trace.last()) else { return };
    let tol = 6.0 / app.point_zoom().max(0.01) as f64;
    let length: f64 = trace.windows(2).map(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1])).sum();
    let closed = (first[0] - last[0]).hypot(first[1] - last[1]) < tol && length > 4.0 * tol;
    let fit = f64::from(app.ui.tool_options.curve_fit);
    let pen = if straight.is_empty() {
        let Some(sub) = photocraft_vector::trace::fit_trace(&trace, fit, closed) else { return };
        let knots = sub.knots.iter().map(knot_points).collect();
        let unlinked = sub.knots.iter().enumerate().filter(|(_, k)| !k.smooth).map(|(i, _)| i).collect();
        (PenPath { knots, unlinked, ..Default::default() }, sub.closed)
    } else {
        fit_with_straight_segments(&trace, straight, fit, closed.then_some(tol))
    };
    commit_path(app, &pen.0, pen.1);
}

fn knot_points(k: &photocraft_doc::Knot) -> [[f64; 2]; 3] {
    [[k.anchor.x, k.anchor.y], [k.in_ctrl.x, k.in_ctrl.y], [k.out_ctrl.x, k.out_ctrl.y]]
}

/// A Freeform Pen trace with ⌥-clicked straight segments (into the points at `straight`): each
/// freehand run between them is fitted on its own, and the runs are joined by straight lines
/// between corner anchors. `close` (a reach) closes the path when it ends on its start point.
fn fit_with_straight_segments(trace: &[[f64; 2]], straight: &[usize], fit: f64, close: Option<f64>) -> (PenPath, bool) {
    let mut cuts: Vec<usize> = straight.iter().copied().filter(|&i| i > 0 && i < trace.len()).collect();
    cuts.sort_unstable();
    cuts.dedup();
    cuts.push(trace.len());
    let (mut knots, mut corners): (Vec<[[f64; 2]; 3]>, Vec<usize>) = (Vec::new(), Vec::new());
    let mut start = 0;
    for end in cuts {
        let run = trace.get(start..end).unwrap_or_default();
        let first = knots.len();
        match photocraft_vector::trace::fit_trace(run, fit, false) {
            Some(sub) => {
                corners.extend(sub.knots.iter().enumerate().filter(|(_, k)| !k.smooth).map(|(i, _)| first + i));
                knots.extend(sub.knots.iter().map(knot_points));
            }
            // A single click (or a run that never moved): one corner anchor.
            None => knots.extend(run.first().map(|p| [*p; 3])),
        }
        // Where a straight segment meets the run: a corner with no handle on that side.
        if let Some(k) = knots.get_mut(first).filter(|_| first > 0) {
            k[1] = k[0];
            corners.push(first);
        }
        if end < trace.len()
            && let Some(k) = knots.last_mut()
        {
            k[2] = k[0];
            corners.push(knots.len() - 1);
        }
        start = end;
    }
    // Ended on the start point: the last anchor is the first one.
    let ends_on_start = |k: &[[[f64; 2]; 3]], reach: f64| match (k.first(), k.last()) {
        (Some(a), Some(b)) => k.len() > 3 && (a[0][0] - b[0][0]).hypot(a[0][1] - b[0][1]) < reach,
        _ => false,
    };
    let closed = close.is_some_and(|reach| ends_on_start(&knots, reach));
    if closed && let Some(last) = knots.pop() {
        if let Some(k) = knots.first_mut() {
            k[1] = [last[1][0] - last[0][0] + k[0][0], last[1][1] - last[0][1] + k[0][1]];
        }
        corners.retain(|&i| i < knots.len());
        corners.push(0);
    }
    corners.sort_unstable();
    corners.dedup();
    (PenPath { knots, unlinked: corners, ..Default::default() }, closed)
}

/// Recompute a Curvature Pen path's handles from its anchors and corner points.
fn curvature_refresh(pen: &mut PenPath, closed: bool) {
    let anchors: Vec<[f64; 2]> = pen.knots.iter().map(|k| k[0]).collect();
    let corners: Vec<bool> = (0..anchors.len()).map(|i| pen.unlinked.contains(&i)).collect();
    pen.knots = photocraft_vector::edit::curvature_knots(&anchors, &corners, closed);
}

/// The Curvature Pen's anchor within a few screen pixels of (x, y).
fn curvature_hit(app: &PhotocraftApp, x: f64, y: f64) -> Option<usize> {
    let tol = 6.0 / app.point_zoom().max(0.01) as f64;
    let pen = app.ui.pen.as_ref().filter(|p| p.curvature)?;
    pen.knots
        .iter()
        .enumerate()
        .map(|(i, k)| (i, (k[0][0] - x).hypot(k[0][1] - y)))
        .filter(|(_, d)| *d < tol)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}

/// Curvature Pen press: clicking the first anchor closes the path, pressing another anchor
/// grabs it to move, anywhere else adds a smooth anchor (still draggable until the release).
pub fn curvature_down(app: &mut PhotocraftApp, x: f64, y: f64) {
    if app.ui.pen.as_ref().is_some_and(|p| !p.curvature) {
        pen_commit(app, false);
    }
    let hit = curvature_hit(app, x, y);
    if hit == Some(0) && app.ui.pen.as_ref().is_some_and(|p| p.knots.len() >= 3) {
        pen_commit(app, true);
        return;
    }
    let pen = app.ui.pen.get_or_insert_with(|| PenPath { curvature: true, ..Default::default() });
    if hit.is_none() {
        pen.knots.push([[x, y]; 3]);
    }
    pen.moving = hit.or(pen.knots.len().checked_sub(1));
    curvature_refresh(pen, false);
}

pub fn curvature_move(app: &mut PhotocraftApp, x: f64, y: f64) {
    if let Some(pen) = app.ui.pen.as_mut().filter(|p| p.curvature)
        && let Some(k) = pen.moving.and_then(|i| pen.knots.get_mut(i))
    {
        k[0] = [x, y];
        curvature_refresh(pen, false);
    }
}

pub fn curvature_up(app: &mut PhotocraftApp) {
    if let Some(pen) = app.ui.pen.as_mut() {
        pen.moving = None;
    }
}

/// Curvature Pen double-click: toggle the anchor under the pointer between smooth and corner
/// (double-clicking empty canvas therefore places a corner point, as in Photoshop).
pub fn curvature_toggle(app: &mut PhotocraftApp, x: f64, y: f64) {
    let Some(i) = curvature_hit(app, x, y) else { return };
    let Some(pen) = app.ui.pen.as_mut() else { return };
    if let Some(at) = pen.unlinked.iter().position(|&c| c == i) {
        pen.unlinked.remove(at);
    } else {
        pen.unlinked.push(i);
    }
    curvature_refresh(pen, false);
}

/// The path that path commands act on without a name: the one selected in the Paths panel, else
/// the work path, else the selected layer's shape path or vector mask; `None` when there's none.
pub fn active_path_name(app: &PhotocraftApp) -> Option<String> {
    let st = app.session.active()?;
    let rows = path_rows(&st.doc, st.active_layer);
    let key = |r: &PathEntry| match r.kind {
        PathRow::Work => "work".to_string(),
        PathRow::Layer => "layer".to_string(),
        PathRow::Saved => r.name.clone(),
    };
    let selected = app.ui.selected_path.as_deref().filter(|s| rows.iter().any(|r| key(r) == *s)).map(str::to_string);
    selected.or_else(|| rows.iter().find(|r| r.kind == PathRow::Work).map(key)).or_else(|| rows.iter().find(|r| r.kind == PathRow::Layer).map(key))
}

/// The commands that create a fill or adjustment layer and take a `"path"` as its vector mask.
pub fn takes_path_mask(id: &str) -> bool {
    id.starts_with("layer.newFillLayer.") || id.starts_with("layer.newAdjustmentLayer.")
}

/// The work or saved path explicitly selected in the Paths panel (not a layer's own path).
pub fn selected_document_path(app: &PhotocraftApp) -> Option<String> {
    let st = app.session.active()?;
    let sel = app.ui.selected_path.as_deref()?;
    let listed = path_rows(&st.doc, st.active_layer).iter().any(|r| match r.kind {
        PathRow::Work => sel == "work",
        PathRow::Saved => r.name == sel,
        PathRow::Layer => false,
    });
    listed.then(|| sel.to_string())
}

/// New fill and adjustment layers take the path selected in the Paths panel as their vector mask,
/// as in Photoshop (#1419): the command gets it as `"path"` unless the caller set one (`null`
/// opts out). Only an explicitly selected work or saved path counts; deselect it by clicking the
/// panel's empty area.
pub fn with_active_path(app: &PhotocraftApp, id: &str, params: Value) -> Value {
    if !takes_path_mask(id) || params.get("path").is_some() {
        return params;
    }
    let Some(sel) = selected_document_path(app) else { return params };
    match params {
        Value::Object(mut m) => {
            m.insert("path".into(), json!(sel));
            Value::Object(m)
        }
        _ => json!({ "path": sel }),
    }
}

/// ⌘↩ / Ctrl+Enter (#306): load a path as a selection, like Photoshop with a Pen or Path
/// Selection tool or a path selected in the Paths panel. A path being drawn with the Pen is
/// finished first and loads; otherwise [`active_path_name`]. `params` may set feather, mode…
pub fn path_to_selection(app: &mut PhotocraftApp, params: Value) -> Result<Value, String> {
    if app.ui.pen.as_ref().is_some_and(|p| p.knots.len() >= 2) {
        // The finished path becomes the work path, or a shape layer's path or a vector mask.
        let on_layer = app.ui.tool_options.vector_mode == "shape" || targeted_vector_mask(app).is_some();
        pen_commit(app, false);
        app.ui.selected_path = Some(if on_layer { "layer" } else { "work" }.into());
    }
    let name = active_path_name(app)
        .ok_or_else(|| tl!("No path to make a selection from: draw one with the Pen tool or select one in the Paths panel.").to_string())?;
    let mut p = params.as_object().cloned().unwrap_or_default();
    p.insert("name".into(), json!(name));
    app.run("path.toSelection", Value::Object(p))
}

// ---------------------------------------------------------------------------------------------
// Path Selection

/// What Path Selection edits.
enum PathTarget {
    Shape(u64),
    /// The active layer's vector mask, when its Layers thumbnail is targeted (#196).
    VectorMask(u64),
    Work,
}

/// The targeted vector mask of the active layer, if the Layers panel targets it.
pub(crate) fn targeted_vector_mask(app: &PhotocraftApp) -> Option<(u64, Path)> {
    let st = app.session.active()?;
    let l = st.active_layer.and_then(|id| st.doc.layer(id))?;
    let targeted = app.ui.vector_mask_target || st.channel_view.target == photocraft_engine::channel_cmds::ChannelTarget::VectorMask;
    (targeted && !matches!(l.content, LayerContent::Shape(_))).then_some(())?;
    l.vector_mask.as_ref().map(|m| (l.id.0, m.path.clone()))
}

/// The active shape layer's path, if the active layer is a shape layer.
pub(crate) fn active_shape_path(app: &PhotocraftApp) -> Option<(u64, Path)> {
    let st = app.session.active()?;
    let l = st.active_layer.and_then(|id| st.doc.layer(id))?;
    match &l.content {
        LayerContent::Shape(sh) => Some((l.id.0, sh.path.clone())),
        _ => None,
    }
}

/// The path Path Selection edits: the targeted vector mask, the active shape layer's path, else
/// the work path.
fn target_path(app: &PhotocraftApp) -> Option<(PathTarget, Path)> {
    if let Some((id, p)) = targeted_vector_mask(app) {
        return Some((PathTarget::VectorMask(id), p));
    }
    if let Some((id, p)) = active_shape_path(app) {
        return Some((PathTarget::Shape(id), p));
    }
    app.session.active()?.doc.work_path.clone().map(|p| (PathTarget::Work, p))
}

/// Edit › Free Transform Path (⌘T with Path Selection or Direct Selection): the path the
/// box transforms (the one Path Selection edits) and `path.transform`'s params that name it.
pub(crate) fn free_transform_path(app: &PhotocraftApp) -> Option<(Value, Path)> {
    if !matches!(app.ui.tool, Tool::PathSelection | Tool::DirectSelection) {
        return None;
    }
    let (target, path) = target_path(app)?;
    path.control_bounds()?;
    let params = match target {
        PathTarget::Shape(id) | PathTarget::VectorMask(id) => json!({"name": "layer", "layer": id}),
        PathTarget::Work => json!({"name": "work"}),
    };
    Some((params, path))
}

fn point_in_subpath(sp: &photocraft_doc::vector::Subpath, pt: [f64; 2]) -> bool {
    if !sp.closed || sp.knots.len() < 3 {
        return false;
    }
    let mut inside = false;
    let n = sp.knots.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let xi = sp.knots[i].anchor.x;
        let yi = sp.knots[i].anchor.y;
        let xj = sp.knots[j].anchor.x;
        let yj = sp.knots[j].anchor.y;
        let intersect = ((yi > pt[1]) != (yj > pt[1])) && (pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi);
        if intersect {
            inside = !inside;
        }
    }
    inside
}

/// The topmost visible shape layer under `at` (document coordinates):
/// hits the shape path or anchor within `tol`, or any non-transparent pixel of the shape's raster surface,
/// or point inside the shape's filled geometry.
pub fn pick_shape_at(doc: &Document, at: [f64; 2], tol: f64) -> Option<photocraft_doc::LayerId> {
    let (x, y) = (at[0].floor() as i32, at[1].floor() as i32);
    for (path, _, l) in doc.walk().iter().rev() {
        if !l.visible || !(1..=path.len()).all(|n| doc.layer_at(&path[..n]).is_some_and(|a| a.visible)) {
            continue;
        }
        let board = path.first().and_then(|i| doc.layers.get(*i)).and_then(|t| t.artboard());
        if board.is_some_and(|a| !a.rect.contains(x, y)) {
            continue;
        }
        if let LayerContent::Shape(sh) = &l.content {
            if photocraft_vector::edit::hit(&sh.path, photocraft_geom::Point::new(at[0], at[1]), tol, &[]).is_some() {
                return Some(l.id);
            }
            if sh.fill.is_some() {
                if let Some(surf) = l.surface() {
                    let mut px = [[0.0f32; 4]; 1];
                    surf.read_rgba_into(photocraft_doc::Rect::from_xywh(x, y, 1, 1), &mut px);
                    if px[0][3] > 0.0 {
                        return Some(l.id);
                    }
                }
                if let Some(photocraft_doc::LiveShape::Rect { rect, .. }) = &sh.live {
                    let x0 = rect[0].min(rect[0] + rect[2]);
                    let x1 = rect[0].max(rect[0] + rect[2]);
                    let y0 = rect[1].min(rect[1] + rect[3]);
                    let y1 = rect[1].max(rect[1] + rect[3]);
                    if at[0] >= x0 && at[0] <= x1 && at[1] >= y0 && at[1] <= y1 {
                        return Some(l.id);
                    }
                }
                if let Some(photocraft_doc::LiveShape::Ellipse { rect }) = &sh.live {
                    let rx = rect[2].abs() / 2.0;
                    let ry = rect[3].abs() / 2.0;
                    if rx > 1e-6 && ry > 1e-6 {
                        let cx = rect[0] + rect[2] / 2.0;
                        let cy = rect[1] + rect[3] / 2.0;
                        let dx = (at[0] - cx) / rx;
                        let dy = (at[1] - cy) / ry;
                        if dx * dx + dy * dy <= 1.0 {
                            return Some(l.id);
                        }
                    }
                }
                if sh.path.subpaths.iter().any(|sp| point_in_subpath(sp, at)) {
                    return Some(l.id);
                }
            }
        }
    }
    None
}

pub fn path_selection_finish(app: &mut PhotocraftApp, start: [f64; 2], end: [f64; 2]) {
    let (dx, dy) = (end[0] - start[0], end[1] - start[1]);
    let tol = 6.0 / f64::from(app.point_zoom().max(0.01));
    if dx.abs() + dy.abs() < 0.5 {
        let Some(st) = app.session.active() else { return };
        if let Some(id) = pick_shape_at(&st.doc, start, tol) {
            let _ = app.run("layer.select", json!({"layer": id.0}));
        }
        return;
    }
    if target_path(app).is_none() {
        let Some(st) = app.session.active() else { return };
        if let Some(id) = pick_shape_at(&st.doc, start, tol) {
            let _ = app.run("layer.select", json!({"layer": id.0}));
        }
    }
    let Some((target, path)) = target_path(app) else { return };
    let _ = match target {
        PathTarget::Shape(id) => app.run("shape.edit", json!({"layer": id, "move": [dx.round(), dy.round()]})),
        PathTarget::VectorMask(id) => {
            let moved = path.transform(&Affine::translate(dx.round(), dy.round()));
            app.run("layer.vectorMask.edit", json!({"layer": id, "path": photocraft_engine::vector_cmds::path_json(&moved)}))
        }
        PathTarget::Work => {
            let moved = path.transform(&Affine::translate(dx.round(), dy.round()));
            app.run("path.set", json!({"name": "work", "path": photocraft_engine::vector_cmds::path_json(&moved)}))
        }
    };
}

// ---------------------------------------------------------------------------------------------
// Drawing

fn bezier(p0: [f64; 2], c0: [f64; 2], c1: [f64; 2], p1: [f64; 2], t: f64) -> [f64; 2] {
    let u = 1.0 - t;
    let f = |a: f64, b: f64, c: f64, d: f64| u * u * u * a + 3.0 * u * u * t * b + 3.0 * u * t * t * c + t * t * t * d;
    [f(p0[0], c0[0], c1[0], p1[0]), f(p0[1], c0[1], c1[1], p1[1])]
}

/// `q` lies on the chord from `a` to `d` (within a thousandth of a pixel).
fn on_chord(a: [f64; 2], q: [f64; 2], d: [f64; 2]) -> bool {
    let (dx, dy, vx, vy) = (d[0] - a[0], d[1] - a[1], q[0] - a[0], q[1] - a[1]);
    let len = dx.hypot(dy);
    if len < 1e-9 {
        return vx.hypot(vy) < 1e-3;
    }
    let t = (vx * dx + vy * dy) / (len * len);
    (vx * dy - vy * dx).abs() / len < 1e-3 && (-1e-9..=1.0 + 1e-9).contains(&t)
}

/// How far (in screen pixels) a drawn chord may stray from the true curve.
const CURVE_TOLERANCE: f64 = 0.3;
/// Pieces per segment at most: 0.3 px holds for curves bending up to about 1.4 million px on screen.
const MAX_CURVE_PIECES: usize = 1024;

/// Screen points along one cubic segment (both ends included). The control points are mapped to
/// the screen first (`xf` is affine, so the curve maps exactly), and the number of chords follows
/// how much the curve bends there: a straight segment is one chord, and a curve zoomed in stays
/// smooth instead of showing a fixed 16 chords (#4084). Uses the bound
/// `max|B''| / 8n² <= tol` with `max|B''| <= 6·max(|p0−2c0+c1|, |c0−2c1+p1|)`.
pub(crate) fn curve_points(p0: [f64; 2], c0: [f64; 2], c1: [f64; 2], p1: [f64; 2], xf: &dyn Fn([f64; 2]) -> Pos2) -> Vec<Pos2> {
    let [a, b, c, d] = [p0, c0, c1, p1].map(|q| {
        let s = xf(q);
        [f64::from(s.x), f64::from(s.y)]
    });
    let second = |p: [f64; 2], q: [f64; 2], r: [f64; 2]| (p[0] - 2.0 * q[0] + r[0]).hypot(p[1] - 2.0 * q[1] + r[1]);
    let n = (0.75 * second(a, b, c).max(second(b, c, d)) / CURVE_TOLERANCE).sqrt().ceil();
    // A straight segment (controls on the chord, like the Pen's retracted handles) is one chord;
    // NaN (a non-finite point) falls back to one too.
    let n = if n.is_finite() && !(on_chord(a, b, d) && on_chord(a, c, d)) { (n as usize).clamp(1, MAX_CURVE_PIECES) } else { 1 };
    (0..=n)
        .map(|k| {
            let q = bezier(a, b, c, d, k as f64 / n as f64);
            pos2(q[0] as f32, q[1] as f32)
        })
        .collect()
}

/// Screen polylines for a path (one per subpath).
pub(crate) fn path_lines(path: &Path, xf: &dyn Fn([f64; 2]) -> Pos2) -> Vec<(Vec<Pos2>, bool)> {
    path.subpaths
        .iter()
        .map(|s| {
            let n = s.knots.len();
            let mut pts = Vec::new();
            let segs = if s.closed { n } else { n.saturating_sub(1) };
            for i in 0..segs {
                let (a, b) = (&s.knots[i], &s.knots[(i + 1) % n]);
                pts.extend(curve_points([a.anchor.x, a.anchor.y], [a.out_ctrl.x, a.out_ctrl.y], [b.in_ctrl.x, b.in_ctrl.y], [b.anchor.x, b.anchor.y], xf));
            }
            if pts.is_empty()
                && let Some(k) = s.knots.first()
            {
                pts.push(xf([k.anchor.x, k.anchor.y]));
            }
            (pts, s.closed)
        })
        .collect()
}

/// A path's outline in the accent colour.
pub(crate) fn draw_outline(painter: &egui::Painter, p: &Path, to_scr: &dyn Fn([f64; 2]) -> Pos2, accent: Color32) {
    for (pts, closed) in path_lines(p, to_scr) {
        if closed {
            painter.add(egui::Shape::closed_line(pts, Stroke::new(1.0, accent)));
        } else {
            painter.add(egui::Shape::line(pts, Stroke::new(1.0, accent)));
        }
    }
}

/// An anchor point's square: filled with the accent when selected, else hollow (white).
pub(crate) fn draw_anchor(painter: &egui::Painter, c: Pos2, selected: bool, accent: Color32) {
    let r = Rect::from_center_size(c, vec2(6.0, 6.0));
    painter.rect_filled(r, 0.0, if selected { accent } else { Color32::WHITE });
    painter.rect_stroke(r, 0.0, Stroke::new(1.0, accent), egui::StrokeKind::Inside);
}

/// A direction handle: the line from its anchor and the round handle end.
pub(crate) fn draw_handle(painter: &egui::Painter, anchor: Pos2, handle: Pos2, accent: Color32) {
    painter.line_segment([anchor, handle], Stroke::new(1.0, accent));
    painter.circle_filled(handle, 3.0, accent);
}

/// Work path / active shape path outlines, anchors, and the pen path in progress.
pub fn draw_overlay(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform, doc: &Document) {
    let tool = app.ui.tool;
    let vector_tool = matches!(tool, Tool::Pen | Tool::FreeformPen | Tool::CurvaturePen | Tool::PathSelection) || is_shape_tool(tool);
    let accent = Tokens::get(painter.ctx()).accent;
    let to_scr = |q: [f64; 2]| xf.to_screen(q[0] as f32, q[1] as f32);
    let draw_path = |p: &Path, anchors: bool| {
        draw_outline(painter, p, &to_scr, accent);
        if anchors {
            for k in p.subpaths.iter().flat_map(|s| &s.knots) {
                draw_anchor(painter, to_scr([k.anchor.x, k.anchor.y]), false, accent);
            }
        }
    };
    // View › Show › Target Path (under Extras) hides the paths; the Pen's path in progress stays.
    // Free Transform Path draws the path through its box instead (transform_tool).
    let paths = app.ui.view.shows(app.ui.view.show.target_path) && app.ui.transform.as_ref().is_none_or(|t| t.path.is_none());
    if crate::direct_select::shows(app, painter.ctx().input(|i| i.modifiers)) {
        // Direct Selection (or the Pen with ⌘/Ctrl held) draws the paths it edits (#790).
        crate::direct_select::draw_overlay(app, painter, &to_scr, accent, paths);
        if paths {
            crate::anchor_tools::draw_hover(app, painter, &to_scr, accent, painter.ctx().input(|i| i.modifiers));
        }
    } else if paths {
        if vector_tool {
            if let Some(wp) = &doc.work_path {
                draw_path(wp, tool == Tool::PathSelection);
            }
            let st = app.session.active();
            if let Some(l) = st.and_then(|s| s.active_layer.and_then(|id| s.doc.layer(id)))
                && let LayerContent::Shape(sh) = &l.content
            {
                draw_path(&sh.path, tool == Tool::PathSelection);
            }
        }
        // A targeted vector mask shows its path with any tool (#196).
        if let Some((_, p)) = targeted_vector_mask(app) {
            draw_path(&p, tool == Tool::PathSelection);
        }
        // The segment the Pen's Auto Add/Delete would add an anchor to, or the anchor it would delete.
        crate::anchor_tools::draw_pen_hover(app, painter, &to_scr, accent, painter.ctx().input(|i| i.modifiers));
    }
    // Pen path in progress, with handles of the last knot and a rubber band to the pointer. While
    // ⌘/Ctrl drags it, Direct Selection draws its preview instead.
    let edited = matches!(&app.ui.direct_selection.drag, Some(crate::direct_select::Drag::Edit { target: crate::direct_select::PathRef::Pen, .. }));
    if let Some(pen) = app.ui.pen.as_ref().filter(|_| !edited) {
        let mut pts = Vec::new();
        // A close-drag shows the first segment bending with the handle being pulled out.
        let mut knots = pen.knots.clone();
        if closing(pen)
            && let Some(&[a, _, out]) = pen.knots.last()
            && out != a
            && let Some(first) = knots.first_mut()
        {
            first[2] = out;
        }
        for w in knots.windows(2) {
            pts.extend(curve_points(w[0][0], w[0][2], w[1][1], w[1][0], &to_scr));
        }
        painter.add(egui::Shape::line(pts, Stroke::new(1.5, accent)));
        if let (Some(last), Some(h)) = (pen.knots.last(), app.hover_doc)
            && !pen.dragging
            && pen.moving.is_none()
        {
            if pen.curvature && pen.knots.len() >= 2 {
                // The Curvature Pen previews the curve the next click makes through the pointer.
                let mut next = pen.clone();
                next.knots.push([h; 3]);
                curvature_refresh(&mut next, false);
                if let [.., a, b] = next.knots.as_slice() {
                    let pts = curve_points(a[0], a[2], b[1], b[0], &to_scr);
                    painter.add(egui::Shape::dashed_line(&pts, Stroke::new(1.0, accent), 4.0, 3.0));
                }
            } else {
                painter.add(egui::Shape::dashed_line(&[to_scr(last[0]), to_scr(h)], Stroke::new(1.0, accent), 4.0, 3.0));
            }
        }
        for (i, k) in pen.knots.iter().enumerate() {
            let c = to_scr(k[0]);
            let last = i + 1 == pen.knots.len();
            if last && k[2] != k[0] && !pen.curvature {
                for hnd in [k[1], k[2]] {
                    draw_handle(painter, c, to_scr(hnd), accent);
                }
            }
            draw_anchor(painter, c, last, accent);
        }
        // Hovering the first anchor: Photoshop's close cue, a small ring beside the pointer.
        let tol = 6.0 / f64::from(app.point_zoom().max(0.01));
        if let Some(h) = app.hover_doc
            && over_first_anchor(pen, h, tol)
            && let Some(first) = pen.knots.first()
        {
            let c = to_scr(first[0]);
            painter.rect_filled(Rect::from_center_size(c, vec2(8.0, 8.0)), 0.0, accent);
            painter.circle_stroke(c + vec2(11.0, 11.0), 3.5, Stroke::new(1.5, accent));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Options bars

/// Options bar for vector tools; false for other tools.
pub fn options_bar(app: &mut PhotocraftApp, ui: &mut egui::Ui, tool: Tool) -> bool {
    if !(is_shape_tool(tool)
        || matches!(
            tool,
            Tool::Pen | Tool::FreeformPen | Tool::CurvaturePen | Tool::AddAnchorPoint | Tool::DeleteAnchorPoint | Tool::PathSelection | Tool::DirectSelection
        ))
    {
        return false;
    }
    let t = Tokens::get(ui.ctx());
    let lbl = |ui: &mut egui::Ui, s: &str| {
        ui.label(egui::RichText::new(s).color(t.text_dim).size(12.0));
    };
    let o = &mut app.ui.tool_options;
    let target = selected_shape(&app.session);
    if tool == Tool::PathSelection && target.is_none() {
        lbl(ui, if app.ui.vector_mask_target { "Drag to move the targeted vector mask" } else { tl!("Drag to move the active shape's path or the Work Path") });
        return true;
    }
    if tool == Tool::DirectSelection && target.is_none() {
        let (shift, alt) = (crate::shortcuts::pretty("Shift"), crate::shortcuts::pretty("Alt"));
        lbl(
            ui,
            &crate::i18n::fmt(
                tl!("Click: select anchor · Drag: move anchor, handle or segment · {shift}-click: add · {alt}-click: whole subpath"),
                &[("shift", &shift), ("alt", &alt)],
            ),
        );
        return true;
    }
    if matches!(tool, Tool::AddAnchorPoint | Tool::DeleteAnchorPoint) {
        let hint = if tool == Tool::AddAnchorPoint {
            tl!("Click a segment to add an anchor point · {alt}-click an anchor to delete it")
        } else {
            tl!("Click an anchor point to delete it · {alt}-click a segment to add one")
        };
        lbl(ui, &crate::i18n::fmt(hint, &[("alt", &crate::shortcuts::pretty("Alt"))]));
        return true;
    }
    if matches!(tool, Tool::Pen | Tool::FreeformPen | Tool::CurvaturePen) {
        let opts = [("path".to_string(), tl!("Path")), ("shape".to_string(), tl!("Shape"))];
        crate::widgets::dropdown(ui, "pen-mode", &mut o.vector_mode, &opts, 80.0);
        crate::widgets::vline(ui, 22.0);
        if o.vector_mode == "shape" {
            crate::widgets::value_field(ui, &mut o.stroke_width, 0.0..=288.0, "px", 58.0);
            if let Some(change) =
                crate::shape_stroke_ui::button(ui, "pen-stroke", &o.shape_stroke, &mut app.ui.stroke_editor, false, true, None, o.stroke_width > 0.0)
            {
                o.shape_stroke = change.options;
                o.stroke_width = enabled_stroke_width(o.stroke_width);
            }
        }
        if tool == Tool::Pen {
            crate::widgets::vline(ui, 22.0);
            crate::widgets::checkbox(ui, &mut o.pen_auto_add_delete, tl!("Auto Add/Delete"));
        }
        let enter = crate::shortcuts::pretty("Enter");
        match tool {
            Tool::FreeformPen => {
                crate::widgets::vline(ui, 22.0);
                lbl(ui, tl!("Curve Fit:"));
                crate::widgets::value_field(ui, &mut o.curve_fit, 0.5..=10.0, "px", 58.0);
                let alt = crate::shortcuts::pretty("Alt");
                lbl(ui, &crate::i18n::fmt(tl!("Drag: draw freehand · {alt}-click: straight segment · Release on the start point: close"), &[("alt", &alt)]));
            }
            Tool::CurvaturePen => lbl(
                ui,
                &crate::i18n::fmt(
                    tl!("Click: smooth point · Double-click: corner · Drag a point: move · Click first point: close · {key} finish"),
                    &[("key", &enter)],
                ),
            ),
            _ => lbl(
                ui,
                &crate::i18n::fmt(
                    tl!("Click: corner · Drag: smooth · {alt}-drag: break handle · Click or drag first point: close · {key} finish · Esc cancel"),
                    &[("alt", &crate::shortcuts::pretty("Alt")), ("key", &enter)],
                ),
            ),
        }
        return true;
    }
    let mut mode = "shape".to_string();
    crate::widgets::dropdown(ui, "shape-mode", &mut mode, &[("shape".to_string(), tl!("Shape"))], 80.0);
    crate::widgets::vline(ui, 22.0);
    // Fill and Stroke edit the selected shape layer too (#3347), and still set the next shape's.
    let (fg, bg) = (app.session.tools.foreground, app.session.tools.background);
    let (mut has_fill, fill, stroke) = match &target {
        Some(s) => (s.fill.is_some(), s.fill.clone(), s.stroke.clone()),
        None => (o.shape_fill, o.shape_fill.then(|| solid_paint(fg)), (o.stroke_width > 0.0).then(|| o.shape_stroke.stroke(o.stroke_width, solid_paint(bg)))),
    };
    let mut width = stroke.as_ref().map_or(0.0, |s| s.width);
    let style = stroke.as_ref().map_or_else(|| o.shape_stroke.clone(), crate::shape_stroke_ui::StrokeOptions::from);
    let key = |k: &str| target.as_ref().map(|s| format!("shape-options-{:?}-{}-{k}", s.doc, s.layer.0));
    let mut edit: Option<Value> = None;
    lbl(ui, tl!("Fill:"));
    if crate::widgets::checkbox(ui, &mut has_fill, "").changed() {
        o.shape_fill = has_fill;
        edit = Some(json!({"fill": if has_fill { json!(fg) } else { Value::Null }}));
    }
    let fill_color = swatch(ui, fill.as_ref(), fg, tl!("Fill color"));
    lbl(ui, tl!("Stroke:"));
    let stroke_color = swatch(ui, stroke.as_ref().map(|s| &s.paint), bg, tl!("Stroke color"));
    if crate::widgets::value_field(ui, &mut width, 0.0..=288.0, "px", 58.0).changed() {
        o.stroke_width = width;
        let mut p = json!({"width": width});
        if stroke.is_none() {
            p["color"] = json!(bg);
        }
        edit = Some(if width <= 0.0 { json!({"stroke": null}) } else { json!({"stroke": p, "coalesce": key("stroke-w")}) });
    }
    let preview = target.as_ref().filter(|s| s.stroke.is_some()).map(|s| (s.doc, s.layer));
    if let Some(change) =
        crate::shape_stroke_ui::button(ui, "tool-stroke", &style, &mut app.ui.stroke_editor, true, tool != Tool::EllipseShape, preview, stroke.is_some())
    {
        let mut p = change.options.params();
        if stroke.is_none() {
            p["width"] = json!(3.0);
            p["color"] = json!(bg);
        }
        edit = Some(json!({"stroke": p, "coalesce": change.coalesce}));
        o.shape_stroke = change.options;
        o.stroke_width = enabled_stroke_width(o.stroke_width);
    }
    match tool {
        Tool::Rectangle => {
            crate::widgets::vline(ui, 22.0);
            lbl(ui, tl!("Radius:"));
            let mut r = match target.as_ref().and_then(|s| s.live.as_ref()) {
                Some(photocraft_doc::LiveShape::Rect { radii, .. }) => radii[0] as f32,
                _ => o.corner_radius,
            };
            if crate::widgets::value_field(ui, &mut r, 0.0..=10000.0, "px", 62.0).changed() {
                o.corner_radius = r;
                let rf = r as f64;
                edit = Some(json!({"radii": [rf, rf, rf, rf], "coalesce": key("radius")}));
            }
        }
        Tool::Polygon => {
            crate::widgets::vline(ui, 22.0);
            lbl(ui, tl!("Sides:"));
            let mut s = o.polygon_sides as f32;
            if crate::widgets::value_field(ui, &mut s, 3.0..=100.0, "", 48.0).changed() {
                o.polygon_sides = s.round().clamp(3.0, 100.0) as u32;
            }
        }
        Tool::Line => {
            crate::widgets::vline(ui, 22.0);
            lbl(ui, tl!("Weight:"));
            crate::widgets::value_field(ui, &mut o.line_weight, 1.0..=1000.0, "px", 58.0);
        }
        Tool::CustomShape => {
            crate::widgets::vline(ui, 22.0);
            crate::preset_panels::shape_picker(app, ui);
        }
        _ if matches!(tool, Tool::PathSelection | Tool::DirectSelection) => {
            if let Some(photocraft_doc::LiveShape::Rect { radii, .. }) = target.as_ref().and_then(|s| s.live.as_ref()) {
                crate::widgets::vline(ui, 22.0);
                lbl(ui, tl!("Radius:"));
                let mut r = radii[0] as f32;
                if crate::widgets::value_field(ui, &mut r, 0.0..=10000.0, "px", 62.0).changed() {
                    o.corner_radius = r;
                    let rf = r as f64;
                    edit = Some(json!({"radii": [rf, rf, rf, rf], "coalesce": key("radius")}));
                }
            }
        }
        _ => {}
    }
    for (is_stroke, change) in [(false, fill_color), (true, stroke_color)] {
        let Some(change) = change else { continue };
        if let Err(e) = set_tool_paint(app, is_stroke, change) {
            app.ui.status = e;
            app.ui.status_error = true;
        }
        edit = Some(match (is_stroke, change) {
            (false, PaintChange::None) => json!({"fill": null}),
            (false, PaintChange::Color(color)) => json!({"fill": color, "coalesce": key("fill")}),
            (true, PaintChange::None) => json!({"stroke": null}),
            (true, PaintChange::Color(color)) => json!({"stroke": {"color": color, "width": enabled_stroke_width(width)}, "coalesce": key("stroke")}),
        });
    }
    if let (Some(mut p), Some(s)) = (edit, &target) {
        p["layer"] = json!(s.layer.0);
        if let Err(e) = app.run("shape.edit", p) {
            app.ui.status = e;
            app.ui.status_error = true;
        }
    }
    true
}

/// The shape layer the options bar edits.
struct SelectedShape {
    doc: photocraft_doc::DocId,
    layer: photocraft_doc::LayerId,
    fill: Option<photocraft_doc::Fill>,
    stroke: Option<photocraft_doc::ShapeStroke>,
    live: Option<photocraft_doc::LiveShape>,
}

fn selected_shape(session: &photocraft_engine::Session) -> Option<SelectedShape> {
    let st = session.active()?;
    let layer = st.active_layer?;
    match &st.doc.layer(layer)?.content {
        LayerContent::Shape(sh) => Some(SelectedShape { doc: st.doc.id, layer, fill: sh.fill.clone(), stroke: sh.stroke.clone(), live: sh.live.clone() }),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------
// Properties (shape layers)

fn solid_paint(c: [f32; 4]) -> photocraft_doc::Fill {
    photocraft_doc::Fill::Solid(photocraft_doc::Color::rgba(c[0], c[1], c[2], c[3]))
}

fn color_of(f: &photocraft_doc::Fill) -> Option<[f32; 4]> {
    match f {
        photocraft_doc::Fill::Solid(c) => {
            let [r, g, b] = c.to_rgb();
            Some([r, g, b, c.alpha])
        }
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum PaintChange {
    Color([f32; 4]),
    None,
}

fn enabled_stroke_width(width: f32) -> f32 {
    if width.is_finite() && width > 0.0 { width } else { 3.0 }
}

/// Tool-bar paint changes affect the next shape; Properties edits the selected shape instead.
fn set_tool_paint(app: &mut PhotocraftApp, stroke: bool, change: PaintChange) -> Result<(), String> {
    if let PaintChange::Color(color) = change {
        app.run("tools.setColors", if stroke { json!({"background": color}) } else { json!({"foreground": color}) })?;
    }
    if stroke {
        app.ui.tool_options.stroke_width = if change == PaintChange::None { 0.0 } else { enabled_stroke_width(app.ui.tool_options.stroke_width) };
    } else {
        app.ui.tool_options.shape_fill = change != PaintChange::None;
    }
    Ok(())
}

fn rgba32(c: [f32; 4]) -> Color32 {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    Color32::from_rgba_unmultiplied(b(c[0]), b(c[1]), b(c[2]), b(c[3]))
}

fn picker_color(c: [f32; 4]) -> egui::ecolor::Hsva {
    let linear = egui::ecolor::linear_from_gamma;
    let display = c.map(|v| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 });
    egui::ecolor::Hsva::from_rgba_unmultiplied(linear(display[0]), linear(display[1]), linear(display[2]), display[3])
}

fn picker_rgba(hsva: egui::ecolor::Hsva) -> [f32; 4] {
    let [r, g, b, a] = hsva.to_rgba_unmultiplied();
    let gamma = egui::ecolor::gamma_from_linear;
    [gamma(r), gamma(g), gamma(b), a]
}

/// The shared compact picker keeps the float RGB paint and its alpha. Color32 is only display
/// data: opening the picker must never quantize an existing high-depth colour or enable paint.
fn swatch(ui: &mut egui::Ui, fill: Option<&photocraft_doc::Fill>, fallback: [f32; 4], tip: &str) -> Option<PaintChange> {
    let t = Tokens::get(ui.ctx());
    let (r, resp) = ui.allocate_exact_size(vec2(30.0, 22.0), Sense::click());
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::ColorButton, ui.is_enabled(), tip));
    let current = fill.and_then(color_of);
    crate::widgets::checker(ui.painter(), r.shrink(1.0), 4.0);
    match (current, fill) {
        (Some(c), _) => {
            ui.painter().rect_filled(r.shrink(1.0), t.radius_sm, rgba32(c));
        }
        (None, Some(photocraft_doc::Fill::Gradient { stops, .. })) if !stops.is_empty() => {
            // Gradient fills preview as a left-to-right ramp through their stops.
            let rgb = |c: &photocraft_doc::Color| {
                let v = c.to_rgba8();
                Color32::from_rgba_unmultiplied(v[0], v[1], v[2], v[3])
            };
            let mut mesh = egui::Mesh::default();
            let mut ramp: Vec<(f32, Color32)> = stops.iter().map(|(p, c)| (p.clamp(0.0, 1.0), rgb(c))).collect();
            ramp.insert(0, (0.0, ramp[0].1));
            ramp.push((1.0, ramp[ramp.len() - 1].1));
            for (i, (p, c)) in ramp.iter().enumerate() {
                let x = r.left() + r.width() * p;
                mesh.colored_vertex(egui::pos2(x, r.top()), *c);
                mesh.colored_vertex(egui::pos2(x, r.bottom()), *c);
                if i > 0 {
                    let k = (i as u32) * 2;
                    mesh.add_triangle(k - 2, k - 1, k);
                    mesh.add_triangle(k - 1, k, k + 1);
                }
            }
            ui.painter().add(mesh);
        }
        (None, Some(_)) => {}
        (None, None) => {
            ui.painter().rect_filled(r.shrink(1.0), t.radius_sm, t.field);
            ui.painter().line_segment([r.shrink(3.0).left_bottom(), r.shrink(3.0).right_top()], Stroke::new(1.5, t.danger));
        }
    };
    let border = if resp.has_focus() {
        t.accent
    } else if resp.hovered() {
        t.text_dim
    } else {
        t.field_border
    };
    ui.painter().rect_stroke(r, t.radius_sm, Stroke::new(1.0, border), egui::StrokeKind::Inside);
    let resp = resp.on_hover_text(tip);
    let screen_id = resp.id.with("screen-color");
    let original = current.unwrap_or(fallback);
    let sampled = |rgb: [f32; 3]| PaintChange::Color([rgb[0], rgb[1], rgb[2], original[3]]);
    let mut out = crate::screen_picker::take(ui.ctx(), screen_id).map(sampled);
    let cache = resp.id.with("shape-paint-hsva");
    let mut hsva = ui
        .data(|d| d.get_temp::<([f32; 4], egui::ecolor::Hsva)>(cache))
        .filter(|(previous, _)| *previous == original)
        .map_or_else(|| picker_color(original), |(_, hsva)| hsva);
    if let Some(PaintChange::Color(color)) = out {
        hsva = picker_color(color);
    }
    crate::widgets::swatch_popup(&resp).show(|ui| {
        ui.horizontal(|ui| {
            if ui.selectable_label(fill.is_none(), tl!("No Color")).clicked() && fill.is_some() {
                out = Some(PaintChange::None);
            }
            if ui.selectable_label(current.is_some(), tl!("Solid")).clicked() && current.is_none() {
                out = Some(PaintChange::Color(original));
            }
        });
        ui.separator();
        let before = picker_rgba(hsva);
        if egui::color_picker::color_picker_hsva_2d(ui, &mut hsva, egui::color_picker::Alpha::OnlyBlend) {
            let color = picker_rgba(hsva);
            // egui's HSV gamma/linear round trip can move a float by an ulp without input.
            if color.iter().zip(before).any(|(a, b)| (a - b).abs() > 0.000001) {
                out = Some(PaintChange::Color(color));
            }
        }
        if let Some(rgb) = crate::screen_picker::button(ui, screen_id) {
            out = Some(sampled(rgb));
            hsva = picker_color([rgb[0], rgb[1], rgb[2], original[3]]);
        }
    });
    let cached_color = match out {
        Some(PaintChange::Color(color)) => color,
        _ => original,
    };
    ui.data_mut(|d| d.insert_temp(cache, (cached_color, hsva)));
    out
}

/// One corner of the live rectangle, preserving the other three radii.
/// The engine takes the full [top-left, top-right, bottom-right, bottom-left] array.
fn corner_radii_patch(radii: [f64; 4], corner: usize, radius: f32) -> Value {
    let mut next = radii;
    if let Some(r) = next.get_mut(corner) {
        *r = f64::from(radius.max(0.0));
    }
    json!({"radii": next})
}

/// Follow the existing brush/mask continuous-edit convention: one history state per press or
/// keyboard edit, with every value tick of a held-pointer gesture sharing its key.
fn paint_gesture(ui: &egui::Ui, key: egui::Id) -> u64 {
    let pass = ui.ctx().cumulative_pass_nr();
    let pressed = ui.input(|i| i.pointer.any_pressed());
    ui.data_mut(|d| {
        let gesture = d.get_temp_mut_or_default::<(u64, u64)>(key);
        if pressed && gesture.1 != pass {
            gesture.0 = gesture.0.wrapping_add(1);
            gesture.1 = pass;
        }
        gesture.0
    })
}

/// Properties panel for a shape layer: Appearance (fill, stroke) and live shape geometry.
pub fn shape_properties(app: &mut PhotocraftApp, ui: &mut egui::Ui, id: photocraft_doc::LayerId) {
    if app.session.active().is_some_and(|st| {
        let locks = st.doc.effective_locks(id);
        locks.all || locks.pixels
    }) {
        ui.disable();
    }
    let Some(sh) = app.session.active().and_then(|s| s.doc.layer(id)).and_then(|l| match &l.content {
        LayerContent::Shape(sh) => Some(sh.clone()),
        _ => None,
    }) else {
        return;
    };
    let t = Tokens::get(ui.ctx());
    let mut edit: Option<Value> = None;
    let doc_id = app.session.active().map(|st| st.doc.id);
    let key = |k: &str| format!("shape-{doc_id:?}-{}-{k}", id.0);
    let gesture_key = ui.id().with(key("paint-gesture"));
    let gesture = paint_gesture(ui, gesture_key);
    let coalesce = |field: &str| format!("{}:{}:{gesture}", key(field), gesture_key.value());
    // The shared collapsible section headers (#155).
    // Prefs "Appearance" is theme chrome (Giao diện); shape fill/stroke is Diện mạo.
    if crate::props_layout::section(ui, "appearance", crate::i18n::tr_ctx(crate::i18n::current(), "shape", "Appearance")) {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(tl!("Fill")).color(t.text_dim).size(12.0));
            if let Some(change) = swatch(ui, sh.fill.as_ref(), app.session.tools.foreground, tl!("Set shape fill type")) {
                let fill = match change {
                    PaintChange::None => Value::Null,
                    PaintChange::Color(color) => json!(color),
                };
                edit = Some(json!({"fill": fill, "coalesce": coalesce("fill")}));
            }
            ui.add_space(12.0);
            ui.label(egui::RichText::new(tl!("Stroke")).color(t.text_dim).size(12.0));
            if let Some(change) = swatch(ui, sh.stroke.as_ref().map(|s| &s.paint), app.session.tools.foreground, tl!("Set shape stroke type")) {
                let stroke = match change {
                    PaintChange::None => Value::Null,
                    PaintChange::Color(color) => json!({"color": color, "width": enabled_stroke_width(sh.stroke.as_ref().map_or(0.0, |s| s.width))}),
                };
                edit = Some(json!({"stroke": stroke, "coalesce": coalesce("stroke")}));
            }
            let mut w = sh.stroke.as_ref().map_or(0.0, |s| s.width);
            if crate::widgets::value_field(ui, &mut w, 0.0..=288.0, "px", 60.0).changed() {
                let stroke = if w <= 0.0 { Value::Null } else { json!({"width": w}) };
                edit = Some(json!({"stroke": stroke, "coalesce": coalesce("stroke-w")}));
            }
        });
        let style = sh.stroke.as_ref().map(crate::shape_stroke_ui::StrokeOptions::from).unwrap_or_default();
        let stroke_enabled = sh.stroke.as_ref().is_some_and(|stroke| stroke.width > 0.0);
        if let Some(change) = crate::shape_stroke_ui::button(
            ui,
            &key("style"),
            &style,
            &mut app.ui.stroke_editor,
            sh.path.subpaths.iter().all(|s| s.closed),
            !matches!(sh.live, Some(photocraft_doc::LiveShape::Ellipse { .. })),
            app.session.active().map(|st| (st.doc.id, id)),
            stroke_enabled,
        ) {
            let mut stroke = change.options.params();
            if !stroke_enabled {
                stroke["width"] = json!(3.0);
            }
            edit = Some(json!({"stroke": stroke, "coalesce": change.coalesce}));
        }
    }
    if let Some(live) = &sh.live
        && crate::props_layout::section(ui, "liveShape", tl!("Shape"))
    {
        let num = |ui: &mut egui::Ui, label: &str, v: &mut f32, range: std::ops::RangeInclusive<f32>, unit: &str| -> bool {
            ui.label(egui::RichText::new(tl!(&label)).color(t.text_dim).size(12.0));
            crate::widgets::value_field(ui, v, range, unit, 64.0).changed()
        };
        match live {
            photocraft_doc::vector::LiveShape::Rect { rect, .. }
            | photocraft_doc::vector::LiveShape::Ellipse { rect }
            | photocraft_doc::vector::LiveShape::Polygon { rect, .. } => {
                ui.horizontal(|ui| {
                    let (mut w, mut h) = (rect[2] as f32, rect[3] as f32);
                    let cw = num(ui, "W", &mut w, 1.0..=300000.0, "px");
                    let ch = num(ui, "H", &mut h, 1.0..=300000.0, "px");
                    if cw || ch {
                        edit = Some(json!({"rect": [rect[0], rect[1], w.round(), h.round()], "coalesce": key("wh")}));
                    }
                });
            }
            photocraft_doc::vector::LiveShape::Line { .. } => {}
        }
        ui.vertical(|ui| match live {
            photocraft_doc::vector::LiveShape::Rect { radii, .. } => {
                // Keep the uniform workflow available after independent corner edits too.
                ui.horizontal(|ui| {
                    let mut all = radii[0] as f32;
                    if num(ui, tl!("Corner radius"), &mut all, 0.0..=100000.0, "px") {
                        edit = Some(json!({"radii": all, "coalesce": key("radius-all")}));
                    }
                });
                // Separate rows also fit the narrow Properties dock. The labels use the
                // localized corner names, rather than untranslated abbreviations.
                for (index, label) in [(0, tl!("Top Left")), (1, tl!("Top Right")), (3, tl!("Bottom Left")), (2, tl!("Bottom Right"))] {
                    ui.horizontal(|ui| {
                        let mut radius = radii[index] as f32;
                        if num(ui, label, &mut radius, 0.0..=100000.0, "px") {
                            let mut patch = corner_radii_patch(*radii, index, radius);
                            patch["coalesce"] = json!(key(&format!("radius-{index}")));
                            edit = Some(patch);
                        }
                    });
                }
            }
            photocraft_doc::vector::LiveShape::Polygon { sides, star_ratio, .. } => {
                ui.horizontal(|ui| {
                    let mut n = *sides as f32;
                    if num(ui, tl!("Sides"), &mut n, 3.0..=100.0, "") {
                        edit = Some(json!({"sides": n.round() as u32, "coalesce": key("sides")}));
                    }
                    let mut sr = (*star_ratio * 100.0) as f32;
                    if num(ui, tl!("Star ratio"), &mut sr, 1.0..=100.0, "%") {
                        edit = Some(json!({"starRatio": sr as f64 / 100.0, "coalesce": key("star")}));
                    }
                })
                .inner
            }
            photocraft_doc::vector::LiveShape::Line { weight, .. } => {
                ui.horizontal(|ui| {
                    let mut w = *weight as f32;
                    if num(ui, tl!("Weight"), &mut w, 1.0..=10000.0, "px") {
                        edit = Some(json!({"weight": w, "coalesce": key("weight")}));
                    }
                })
                .inner
            }
            _ => {}
        });
    }
    if let Some(mut p) = edit {
        p["layer"] = json!(id.0);
        if let Err(e) = app.run("shape.edit", p) {
            app.ui.status = e;
            app.ui.status_error = true;
        }
        if !ui.input(|i| i.pointer.any_down()) {
            ui.data_mut(|d| {
                let gesture = d.get_temp_mut_or_default::<(u64, u64)>(gesture_key);
                gesture.0 = gesture.0.wrapping_add(1);
            });
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Paths panel

fn thumb(ui: &egui::Ui, r: Rect, path: &Path, doc: &Document) {
    let t = Tokens::get(ui.ctx());
    // At the document's aspect ratio, like the Channels thumbnails.
    let r = crate::channels_panel::fit_thumb(r, doc.size.width, doc.size.height).0;
    ui.painter().rect_filled(r, 0.0, Color32::from_gray(if t.pro { 222 } else { 240 }));
    let (w, h) = (doc.size.width.max(1) as f64, doc.size.height.max(1) as f64);
    let s = (r.width() as f64 / w).min(r.height() as f64 / h);
    let off = r.center() - vec2((w * s) as f32 / 2.0, (h * s) as f32 / 2.0);
    let map = |q: [f64; 2]| pos2(off.x + (q[0] * s) as f32, off.y + (q[1] * s) as f32);
    for (pts, closed) in path_lines(path, &map) {
        if closed && pts.len() >= 3 {
            ui.painter().add(egui::Shape::closed_line(pts, Stroke::new(1.0, Color32::from_gray(40))));
        } else {
            ui.painter().add(egui::Shape::line(pts, Stroke::new(1.0, Color32::from_gray(40))));
        }
    }
}

/// What a Paths panel row is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathRow {
    Saved,
    Work,
    /// The selected layer's shape path or vector mask (temporary, italic).
    Layer,
}

/// A Paths panel row.
#[derive(Clone, Debug)]
pub struct PathEntry {
    pub name: String,
    pub path: Path,
    pub kind: PathRow,
}

/// The Paths panel's rows, top to bottom: saved paths, the work path, then the selected layer's
/// shape path ("<Layer> Shape Path") or vector mask ("<Layer> Vector Mask"), like Photoshop.
pub fn path_rows(doc: &Document, active: Option<photocraft_doc::LayerId>) -> Vec<PathEntry> {
    let mut rows: Vec<_> = doc.paths.iter().map(|p| PathEntry { name: p.name.clone(), path: p.path.clone(), kind: PathRow::Saved }).collect();
    if let Some(wp) = &doc.work_path {
        rows.push(PathEntry { name: "Work Path".into(), path: wp.clone(), kind: PathRow::Work });
    }
    if let Some(l) = active.and_then(|id| doc.layer(id)) {
        let layer_path = match &l.content {
            LayerContent::Shape(sh) => Some((format!("{} Shape Path", l.name), sh.path.clone())),
            _ => l.vector_mask.as_ref().map(|v| (format!("{} Vector Mask", l.name), v.path.clone())),
        };
        if let Some((name, path)) = layer_path {
            rows.push(PathEntry { name, path, kind: PathRow::Layer });
        }
    }
    rows
}

fn footer_id() -> egui::Id {
    egui::Id::new("paths-footer")
}

fn ctx_data_footer(ctx: &egui::Context, r: Rect) {
    ctx.data_mut(|d| d.insert_temp(footer_id(), r));
}

/// Where the Paths panel's footer buttons were drawn last frame (the rect around them all).
pub fn paths_footer(ctx: &egui::Context) -> Option<Rect> {
    ctx.data(|d| d.get_temp(footer_id()))
}

/// Commands offered by a Paths row's context menu. The row's path is passed explicitly so a
/// right-click acts on that row even when another path is selected in the panel.
fn path_context_actions(entry: &PathEntry, doc: &Document) -> Vec<(&'static str, &'static str, Value)> {
    let key = match entry.kind {
        PathRow::Work => "work".to_string(),
        PathRow::Layer => "layer".to_string(),
        PathRow::Saved => entry.name.clone(),
    };
    let mut actions = vec![
        ("Make Selection", "path.toSelection", json!({"name": key})),
        ("Fill Path", "path.fill", json!({"name": key})),
        ("Stroke Path…", "path.stroke", json!({"name": key})),
    ];
    if entry.kind == PathRow::Work {
        let mut n = doc.paths.len().saturating_add(1);
        while doc.paths.iter().any(|p| p.name == format!("Path {n}")) {
            n = n.saturating_add(1);
            if n == usize::MAX {
                break;
            }
        }
        actions.push(("Save Path", "path.rename", json!({"name": "work", "to": format!("Path {n}")})));
    } else {
        let mut n = 1usize;
        while doc.paths.iter().any(|p| p.name == format!("{} copy {n}", entry.name)) {
            n = n.saturating_add(1);
            if n == usize::MAX {
                break;
            }
        }
        let copy_name = format!("{} copy {n}", entry.name);
        actions.push(("Duplicate Path", "path.set", json!({"name": copy_name, "path": photocraft_engine::vector_cmds::path_json(&entry.path)})));
    }
    if entry.kind != PathRow::Layer {
        actions.push(("Delete Path", "path.delete", json!({"name": key})));
    } else if entry.name.ends_with(" Vector Mask") {
        actions.push(("Delete Vector Mask", "layer.vectorMask.delete", json!({})));
    }
    actions
}

pub fn paths_panel(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let Some(st) = app.session.active() else {
        ui.label(egui::RichText::new(tl!("No document")).color(t.text_faint));
        return;
    };
    let doc = st.doc.clone();
    let rows = path_rows(&doc, st.active_layer);
    let has_layer_path = rows.iter().any(|r| r.kind == PathRow::Layer);
    let mut action: Option<(&str, Value)> = None;
    let mut stroke_dialog: Option<String> = None;
    // The buttons sit in a footer at the panel's bottom, like Photoshop's.
    let footer = crate::widgets::footer_height(ui) + ui.spacing().item_spacing.y;
    let fill = ui.available_height() > footer + 60.0;
    let rows_h = if fill { ui.available_height() - footer } else { f32::INFINITY };
    egui::ScrollArea::vertical().id_salt("path-rows").max_height(rows_h).min_scrolled_height(if fill { rows_h } else { 0.0 }).auto_shrink([false, !fill]).show(
        ui,
        |ui| {
            if rows.is_empty() {
                ui.label(egui::RichText::new(tl!("Draw with the Pen tool (P) or make a work path from a selection.")).color(t.text_faint).size(11.5));
            }
            for PathEntry { name, path, kind } in &rows {
                let (r, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 36.0), Sense::click());
                let key = match kind {
                    PathRow::Work => "work".to_string(),
                    PathRow::Layer => "layer".to_string(),
                    PathRow::Saved => name.clone(),
                };
                // Rows are painted: name them for screen readers and UI tests.
                resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::SelectableLabel, true, name));
                let sel = app.ui.selected_path.as_deref() == Some(key.as_str());
                if sel {
                    ui.painter().rect_filled(r, 0.0, t.row_selected);
                } else if resp.hovered() {
                    ui.painter().rect_filled(r, 0.0, t.hover.gamma_multiply(0.5));
                }
                let thumbnail = Rect::from_min_size(pos2(r.left() + 8.0, r.top() + 4.0), vec2(28.0, 28.0));
                thumb(ui, thumbnail, path, &doc);
                let mut job = egui::text::LayoutJob::default();
                // Temporary paths (the work path, the selected layer's shape path or vector mask) are italic.
                let italics = *kind != PathRow::Saved;
                job.append(
                    if *kind == PathRow::Work { tl!(name) } else { name },
                    0.0,
                    egui::TextFormat { font_id: egui::FontId::proportional(12.0), color: t.text, italics, ..Default::default() },
                );
                let g = ui.painter().layout_job(job);
                ui.painter().galley(pos2(r.left() + 46.0, r.center().y - g.size().y / 2.0), g, t.text);
                // Ctrl/Cmd-click loads only the thumbnail's path, without changing the panel target.
                let modifiers = ui.input(|i| i.modifiers);
                let load = resp.clicked() && modifiers.command && resp.interact_pointer_pos().is_some_and(|p| thumbnail.contains(p));
                if load {
                    let mode = match crate::channels_panel::load_operation(modifiers) {
                        "new" => "replace",
                        mode => mode,
                    };
                    action = Some(("path.toSelection", json!({"name": key, "mode": mode})));
                } else if resp.clicked() {
                    app.ui.selected_path = Some(key.clone());
                }
                // Double-clicking the Work Path saves it (Photoshop's "Save Path").
                if !load && resp.double_clicked() && *kind == PathRow::Work {
                    let n = doc.paths.len() + 1;
                    action = Some(("path.rename", json!({"name": "work", "to": format!("Path {n}")})));
                }
                resp.context_menu(|ui| {
                    crate::widgets::menu_scroll(ui, |ui| {
                        ui.set_min_width(190.0);
                        let entry = PathEntry { name: name.clone(), path: path.clone(), kind: *kind };
                        let can_paint = app.session.active().and_then(|s| s.active_layer.and_then(|id| s.doc.layer(id))).is_some_and(|l| l.surface().is_some());
                        for (label, cmd, params) in path_context_actions(&entry, &doc) {
                            let enabled = !matches!(cmd, "path.fill" | "path.stroke") || can_paint;
                            if ui.add_enabled(enabled, egui::Button::new(tl!(label))).clicked() {
                                if cmd == "path.stroke" {
                                    // Stroke Path… asks for the tool first, as in Photoshop (#2978).
                                    stroke_dialog = Some(params.get("name").and_then(Value::as_str).unwrap_or("work").to_string());
                                } else {
                                    action = Some((cmd, params));
                                }
                                ui.close();
                            }
                        }
                    });
                });
                ui.painter().line_segment([r.left_bottom(), r.right_bottom()], Stroke::new(1.0, t.separator));
            }
            // Clicking the empty area deselects the path, as in Photoshop: new fill and
            // adjustment layers then take the selection as their mask instead (#1419).
            let rest = ui.available_rect_before_wrap();
            let rest = rest.intersect(Rect::from_min_size(rest.min, vec2(rest.width(), rest.height().min(4000.0))));
            if rest.is_positive() && ui.interact(rest, ui.id().with("paths-empty"), Sense::click()).clicked() {
                app.ui.selected_path = None;
            }
        },
    );
    let sel = app.ui.selected_path.clone().filter(|s| s != "layer" || has_layer_path).unwrap_or_else(|| "work".into());
    let buttons = crate::widgets::panel_footer(ui, |ui| {
        let fg = hex(app.session.tools.foreground);
        let mut n = doc.paths.len() + 1;
        while doc.paths.iter().any(|p| p.name == format!("Path {n}")) {
            n += 1;
        }
        let items: [(&str, &str, &str, Value); 7] = [
            ("paint-bucket", "Fill path with foreground color", "path.fill", json!({"name": sel, "color": fg})),
            ("circle", "Stroke path with brush", "path.stroke", json!({"name": sel, "tool": "brush"})),
            ("square-dashed", "Load path as a selection", "path.toSelection", json!({"name": sel})),
            ("spline", "Make work path from selection", "select.toWorkPath", json!({"tolerance": 2.0})),
            ("square", "Add vector mask", "layer.vectorMask.fromPath", json!({"name": sel})),
            ("square-plus", "Create new path", "path.set", json!({"name": format!("Path {n}"), "path": {"subpaths": []}})),
            ("trash", "Delete path", "path.delete", json!({"name": sel})),
        ];
        // The layer's own path: deleting it deletes the vector mask (a shape keeps its path).
        let shape = rows.iter().any(|r| r.kind == PathRow::Layer && r.name.ends_with(" Shape Path"));
        let delete_layer_path = sel == "layer" && !shape;
        let items = items.map(|(icon, tip, cmd, p)| {
            if cmd == "path.delete" && delete_layer_path { (icon, "Delete vector mask", "layer.vectorMask.delete", json!({})) } else { (icon, tip, cmd, p) }
        });
        // Laid out from the right, so the last button goes first.
        let mut buttons = Rect::NOTHING;
        for (icon, tip, cmd, p) in items.into_iter().rev() {
            let icon = if crate::icons::exists(icon) { icon } else { "square" };
            let b = crate::icons::button(ui, icon, 26.0, false, tip);
            if b.clicked() {
                if cmd == "path.stroke" && ui.input(|i| i.modifiers.alt) {
                    // Alt-click opens the Stroke Path dialog, as in Photoshop (#2978).
                    stroke_dialog = Some(sel.clone());
                } else {
                    action = Some((cmd, p));
                }
            }
            buttons = buttons.union(b.rect);
        }
        buttons
    });
    ctx_data_footer(ui.ctx(), buttons);
    if let Some(name) = stroke_dialog {
        crate::canvas_tool_menu::stroke_path_dialog(app, &name);
    }
    if let Some((cmd, p)) = action
        && let Err(e) = app.run(cmd, p)
    {
        app.ui.status = e;
        app.ui.status_error = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #4084: zoomed in, a curve's chords stay within a fraction of a pixel of the true curve
    /// (16 fixed chords showed corners), and a straight segment stays one chord.
    #[test]
    fn path_curves_stay_smooth_at_any_zoom() {
        let (p0, c0, c1, p1) = ([0.0, 100.0], [20.0, -60.0], [80.0, -60.0], [100.0, 100.0]);
        for zoom in [0.25, 1.0, 8.0, 64.0, 320.0] {
            let xf = move |q: [f64; 2]| pos2((q[0] * zoom) as f32 + 10.0, (q[1] * zoom) as f32 - 5.0);
            let pts = curve_points(p0, c0, c1, p1, &xf);
            let n = pts.len() - 1;
            assert_eq!((pts[0], pts[n]), (xf(p0), xf(p1)), "the ends are the anchors");
            for (k, w) in pts.windows(2).enumerate() {
                let curve = xf(bezier(p0, c0, c1, p1, (k as f64 + 0.5) / n as f64));
                let chord = w[0] + (w[1] - w[0]) * 0.5;
                assert!((chord - curve).length() <= 0.35, "zoom {zoom}: chord {k} of {n} is {} px off", (chord - curve).length());
            }
            // Zoomed out it stays cheap.
            if zoom <= 1.0 {
                assert!(n <= 32, "zoom {zoom}: {n} chords");
            }
        }
        let xf = |q: [f64; 2]| pos2((q[0] * 320.0) as f32, (q[1] * 320.0) as f32);
        // Retracted handles, and handles along the line: one chord.
        assert_eq!(curve_points([0.0, 0.0], [0.0, 0.0], [50.0, 50.0], [50.0, 50.0], &xf).len(), 2);
        assert_eq!(curve_points([0.0, 0.0], [10.0, 10.0], [40.0, 40.0], [50.0, 50.0], &xf).len(), 2);
        // A handle past the end is a curve that doubles back, not a line.
        assert!(curve_points([0.0, 0.0], [0.0, 0.0], [80.0, 80.0], [50.0, 50.0], &xf).len() > 2);
        // Hostile coordinates: no panic, a bounded number of points.
        for bad in [f64::NAN, f64::INFINITY, 1e300] {
            assert!(curve_points([0.0, 0.0], [bad, 0.0], [0.0, bad], [1.0, 1.0], &xf).len() <= MAX_CURVE_PIECES + 1);
        }
        // Through path_lines too: a closed subpath of two curved segments.
        let mut path = Path::default();
        path.subpaths.push(photocraft_doc::vector::Subpath {
            knots: vec![
                photocraft_doc::vector::Knot {
                    anchor: photocraft_geom::Point::new(0.0, 0.0),
                    in_ctrl: photocraft_geom::Point::new(0.0, 40.0),
                    out_ctrl: photocraft_geom::Point::new(0.0, -40.0),
                    smooth: true,
                },
                photocraft_doc::vector::Knot {
                    anchor: photocraft_geom::Point::new(60.0, 0.0),
                    in_ctrl: photocraft_geom::Point::new(60.0, -40.0),
                    out_ctrl: photocraft_geom::Point::new(60.0, 40.0),
                    smooth: true,
                },
            ],
            closed: true,
            ..Default::default()
        });
        let lines = path_lines(&path, &xf);
        assert!(lines[0].1 && lines[0].0.len() > 2 * 17, "zoomed in, more than the old 16 chords per segment: {}", lines[0].0.len());
    }

    #[test]
    fn all_shape_tools_and_open_pen_use_the_complete_stroke_defaults() {
        for tool in [Tool::Rectangle, Tool::EllipseShape, Tool::Triangle, Tool::Polygon, Tool::Line, Tool::CustomShape] {
            let mut a = app();
            a.ui.tool_options.stroke_width = 5.0;
            a.ui.tool_options.shape_stroke = crate::shape_stroke_ui::StrokeOptions {
                cap: photocraft_doc::LineCap::Square,
                join: photocraft_doc::LineJoin::Bevel,
                dashes: vec![4.0, 2.0, 1.0, 3.0],
                dash_offset: -0.5,
                opacity: 0.4,
                ..Default::default()
            };
            let history = a.session.active().unwrap().history.past_len();
            finish_shape(&mut a, tool, [20.0, 20.0], [120.0, 100.0], Default::default());
            let info = a.run("shape.info", json!({})).unwrap();
            assert_eq!(info["stroke"]["dashes"], json!([4.0, 2.0, 1.0, 3.0]));
            assert_eq!(info["stroke"]["cap"], "square");
            assert_eq!(info["stroke"]["width"], 5.0);
            assert_eq!(a.session.active().unwrap().history.past_len(), history + 1);
            a.run("edit.undo", json!({})).unwrap();
            a.run("edit.redo", json!({})).unwrap();
            assert_eq!(a.run("shape.info", json!({})).unwrap(), info);
            a.ui.tool_options.vector_mode = "shape".into();
            a.ui.pen = Some(PenPath { knots: vec![[[30.0, 40.0]; 3], [[80.0, 40.0]; 3]], ..Default::default() });
            pen_commit(&mut a, false);
            let open = a.run("shape.info", json!({})).unwrap();
            assert_eq!(open["stroke"]["dashes"], info["stroke"]["dashes"]);
            assert_eq!(open["path"]["subpaths"][0]["closed"], false);
        }
    }

    fn app() -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": 200, "height": 200})).unwrap();
        app.sync_views();
        app
    }

    #[test]
    fn paths_context_menu_uses_the_clicked_row_and_dispatches_commands() {
        let mut app = app();
        let path = json!({"subpaths": [{"closed": true, "knots": [
            {"anchor": [10, 10], "in": [10, 10], "out": [10, 10]},
            {"anchor": [80, 10], "in": [80, 10], "out": [80, 10]},
            {"anchor": [80, 80], "in": [80, 80], "out": [80, 80]}
        ]}]});
        app.run("path.set", json!({"name": "First", "path": path})).unwrap();
        app.run("path.set", json!({"name": "Second", "path": path})).unwrap();
        let doc = &app.session.active().unwrap().doc;
        let first = path_rows(doc, None).into_iter().find(|r| r.name == "First").unwrap();
        let entries = path_context_actions(&first, doc);
        assert_eq!(entries[0].1, "path.toSelection");
        assert_eq!(entries[0].2["name"], "First");
        assert!(entries.iter().all(|(_, id, _)| photocraft_engine::commands::find(id).is_some()));
        let (_, id, params) = entries.into_iter().find(|(_, id, _)| *id == "path.toSelection").unwrap();
        app.run(id, params).unwrap();
        assert!(app.session.active().unwrap().doc.selection.is_some());
        assert_eq!(app.session.journal.last().map(|(id, _)| id.as_str()), Some("path.toSelection"));

        app.run("path.set", json!({"name": "First copy 1", "path": path})).unwrap();
        let doc = &app.session.active().unwrap().doc;
        let first = path_rows(doc, None).into_iter().find(|r| r.name == "First").unwrap();
        let (_, _, duplicate) = path_context_actions(&first, doc).into_iter().find(|(_, id, _)| *id == "path.set").unwrap();
        assert_eq!(duplicate["name"], "First copy 2", "duplicate must not overwrite an existing path");
    }

    #[test]
    fn work_path_context_save_and_layer_path_deletion_rules() {
        let mut app = app();
        let path = json!({"subpaths": []});
        app.run("path.set", json!({"name": "work", "path": path})).unwrap();
        let doc = &app.session.active().unwrap().doc;
        let work = path_rows(doc, None).into_iter().find(|r| r.kind == PathRow::Work).unwrap();
        let entries = path_context_actions(&work, doc);
        assert!(entries.iter().any(|(_, id, _)| *id == "path.rename"));
        assert!(!entries.iter().any(|(_, id, _)| *id == "path.set"));
        let layer = PathEntry { kind: PathRow::Layer, ..work };
        let entries = path_context_actions(&layer, doc);
        assert!(!entries.iter().any(|(_, id, _)| *id == "path.delete"));
    }

    /// How many shapes the path overlay paints for `app` this frame.
    fn overlay_shapes(app: &PhotocraftApp, ctx: &egui::Context) -> usize {
        let doc = app.session.active().unwrap().doc.clone();
        let mut out = ctx.run_ui(egui::RawInput::default(), |ui| {
            let rect = Rect::from_min_size(Pos2::ZERO, vec2(400.0, 400.0));
            let painter = ui.ctx().layer_painter(egui::LayerId::background()).with_clip_rect(rect);
            let xf = ViewXform { rect, zoom: 1.0, center: [100.0, 100.0], flip: false, rotation: 0.0, aspect: 1.0 };
            draw_overlay(app, &painter, &xf, &doc);
        });
        out.textures_delta.clear();
        out.shapes.len()
    }

    #[test]
    fn show_target_path_and_extras_hide_the_path_outlines() {
        // #1119: View › Show › Target Path only flipped its checkmark; the canvas kept drawing the
        // work path and the active shape's path.
        let ctx = egui::Context::default();
        let mut app = app();
        let path = json!({"subpaths": [{"closed": true, "knots": [
            {"anchor": [10, 10], "in": [10, 10], "out": [10, 10]},
            {"anchor": [80, 10], "in": [80, 10], "out": [80, 10]},
            {"anchor": [80, 80], "in": [80, 80], "out": [80, 80]}
        ]}]});
        app.run("path.set", json!({"path": path})).unwrap();
        assert!(app.session.active().unwrap().doc.work_path.is_some());
        finish_shape(&mut app, Tool::Rectangle, [100.0, 100.0], [160.0, 150.0], egui::Modifiers::NONE);
        let toggle = |app: &mut PhotocraftApp, id: &str| crate::menus::invoke(app, &ctx, id, Value::Null).unwrap();
        for tool in [Tool::Pen, Tool::PathSelection, Tool::DirectSelection, Tool::Rectangle] {
            app.ui.tool = tool;
            assert!(app.ui.view.show.target_path && app.ui.view.extras);
            assert!(overlay_shapes(&app, &ctx) > 0, "{tool:?}: the paths are drawn by default");
            toggle(&mut app, "view.show.targetPath");
            assert!(!app.ui.view.show.target_path);
            assert_eq!(overlay_shapes(&app, &ctx), 0, "{tool:?}: Target Path off");
            toggle(&mut app, "view.show.targetPath");
            toggle(&mut app, "view.extras");
            assert!(!app.ui.view.extras);
            assert_eq!(overlay_shapes(&app, &ctx), 0, "{tool:?}: Extras off");
            toggle(&mut app, "view.extras");
            assert!(overlay_shapes(&app, &ctx) > 0, "{tool:?}: shown again");
        }
        // The Pen's path in progress is a gesture, not the target path: it stays visible.
        app.ui.tool = Tool::Pen;
        app.ui.pen = Some(PenPath { knots: vec![[[20.0, 20.0]; 3], [[60.0, 40.0]; 3]], ..Default::default() });
        toggle(&mut app, "view.show.targetPath");
        assert!(overlay_shapes(&app, &ctx) > 0, "the pen path in progress is still drawn");
    }

    #[test]
    fn shape_drag_creates_layers_with_modifiers() {
        let mut app = app();
        finish_shape(&mut app, Tool::Rectangle, [10.0, 10.0], [60.0, 30.0], egui::Modifiers::SHIFT);
        let st = app.session.active().unwrap();
        let b = st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().content_bounds();
        assert_eq!((b.width(), b.height()), (50, 50), "shift = square");
        finish_shape(&mut app, Tool::EllipseShape, [100.0, 100.0], [120.0, 110.0], egui::Modifiers::ALT);
        let st = app.session.active().unwrap();
        let b = st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().content_bounds();
        assert!(b.x0.abs_diff(80) <= 1 && b.width().abs_diff(40) <= 1, "alt = from centre: {b:?}");
        finish_shape(&mut app, Tool::Line, [10.0, 150.0], [90.0, 152.0], egui::Modifiers::SHIFT);
        assert_eq!(app.session.active().unwrap().doc.layers.len(), 4);
    }

    #[test]
    fn shift_pressed_during_a_shape_drag_constrains_it() {
        // #668: ⇧ only counted when held at the press; Photoshop constrains when it is pressed
        // mid-drag (and stops when it is released).
        use crate::canvas::{ToolEvent, tool_event};
        for (tool, sides) in [(Tool::Polygon, 6), (Tool::Rectangle, 4), (Tool::EllipseShape, 0)] {
            let mut app = app();
            app.ui.tool = tool;
            app.ui.tool_options.polygon_sides = 6;
            let none = egui::Modifiers::NONE;
            tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 10.0, pressure: 1.0 }, none);
            tool_event(&mut app, ToolEvent::Move { x: 40.0, y: 30.0, pressure: 1.0 }, none);
            tool_event(&mut app, ToolEvent::Move { x: 70.0, y: 40.0, pressure: 1.0 }, egui::Modifiers::SHIFT);
            tool_event(&mut app, ToolEvent::Up { x: 70.0, y: 40.0 }, egui::Modifiers::SHIFT);
            let st = app.session.active().unwrap();
            let l = st.doc.layer(st.active_layer.unwrap()).unwrap();
            assert!(matches!(l.content, LayerContent::Shape(_)), "{tool:?} ({sides} sides): a shape layer");
            let b = l.surface().unwrap().content_bounds();
            // The drag is 60 × 30; constrained, the shape's box is 60 × 60 (a hexagon is narrower
            // than its box, so the height tells).
            assert!(b.height().abs_diff(60) <= 1 && b.width() <= 61, "{tool:?}: ⇧ mid-drag keeps proportions: {b:?}");
        }
    }

    #[test]
    fn shape_preview_is_the_committed_path() {
        let mut app = app();
        app.ui.tool_options.corner_radius = 8.0;
        let preview = |app: &PhotocraftApp, tool, end| {
            shape_geometry(&app.ui.tool_options, tool, [10.0, 10.0], end, egui::Modifiers::ALT)
                .and_then(|p| photocraft_engine::vector_cmds::shape_path(&p).ok())
        };
        for tool in [Tool::Rectangle, Tool::EllipseShape, Tool::Triangle, Tool::Polygon, Tool::Line] {
            let shown = preview(&app, tool, [60.0, 30.0]).unwrap();
            finish_shape(&mut app, tool, [10.0, 10.0], [60.0, 30.0], egui::Modifiers::ALT);
            let st = app.session.active().unwrap();
            let LayerContent::Shape(sh) = &st.doc.layer(st.active_layer.unwrap()).unwrap().content else { panic!("{tool:?}: no shape layer") };
            assert_eq!(sh.path, shown, "{tool:?}");
        }
        assert!(preview(&app, Tool::Rectangle, [10.2, 40.0]).is_none(), "too thin to draw");
        assert!(photocraft_engine::vector_cmds::shape_path(&json!({"kind": "nope", "rect": [0, 0, 5, 5]})).is_err());
        assert!(photocraft_engine::vector_cmds::shape_path(&json!({"kind": "rect"})).is_err());
    }

    #[test]
    fn pen_shows_the_close_cue_only_over_the_first_anchor_of_a_real_segment() {
        let mut app = app();
        app.ui.tool = Tool::Pen;
        pen_down(&mut app, 20.0, 20.0);
        pen_up(&mut app);
        let one = app.ui.pen.clone().unwrap();
        assert!(!over_first_anchor(&one, [20.0, 20.0], 6.0), "a single anchor has nothing to close");
        pen_down(&mut app, 120.0, 20.0);
        pen_up(&mut app);
        let two = app.ui.pen.clone().unwrap();
        assert!(over_first_anchor(&two, [22.0, 21.0], 6.0));
        assert!(!over_first_anchor(&two, [60.0, 60.0], 6.0));
    }

    #[test]
    fn pen_builds_and_closes_a_work_path() {
        let mut app = app();
        app.ui.tool = Tool::Pen;
        for (x, y) in [(20.0, 20.0), (120.0, 20.0)] {
            pen_down(&mut app, x, y);
            pen_up(&mut app);
        }
        pen_down(&mut app, 120.0, 120.0);
        pen_move(&mut app, 140.0, 140.0); // smooth knot
        pen_up(&mut app);
        pen_down(&mut app, 20.5, 20.5); // click the first anchor: close
        pen_up(&mut app);
        assert!(app.ui.pen.is_none());
        let wp = app.session.active().unwrap().doc.work_path.clone().unwrap();
        assert!(wp.subpaths[0].closed && wp.subpaths[0].knots.len() == 3);
        assert!(wp.subpaths[0].knots[2].smooth);
        // Path Selection moves it.
        path_selection_finish(&mut app, [50.0, 50.0], [60.0, 55.0]);
        let wp = app.session.active().unwrap().doc.work_path.clone().unwrap();
        assert_eq!((wp.subpaths[0].knots[0].anchor.x, wp.subpaths[0].knots[0].anchor.y), (30.0, 25.0));
    }

    /// #4721: a second Pen path joins the Work Path it was drawn after (still selected), as a new
    /// subpath, instead of replacing it; a saved path that is selected takes it instead; with no
    /// path selected, the new path replaces the Work Path, as in Photoshop.
    #[test]
    fn a_new_pen_path_joins_the_selected_path() {
        let mut app = app();
        app.ui.tool = Tool::Pen;
        let draw = |app: &mut PhotocraftApp, x: f64| {
            for (dx, dy) in [(0.0, 0.0), (40.0, 0.0), (40.0, 40.0)] {
                pen_down(app, x + dx, 20.0 + dy);
                pen_up(app);
            }
            pen_down(app, x + 0.5, 20.5); // close
            pen_up(app);
        };
        let work = |app: &PhotocraftApp| app.session.active().unwrap().doc.work_path.clone().unwrap();
        draw(&mut app, 20.0);
        assert_eq!(app.ui.selected_path.as_deref(), Some("work"));
        draw(&mut app, 120.0);
        let wp = work(&app);
        assert_eq!(wp.subpaths.len(), 2, "the first path stays");
        assert!(wp.subpaths.iter().all(|s| s.closed && s.knots.len() == 3));
        assert_eq!(wp.subpaths[0].knots[0].anchor.x, 20.0);
        assert_eq!(wp.subpaths[1].knots[0].anchor.x, 120.0);
        // One History step per path.
        app.session.undo();
        assert_eq!(work(&app).subpaths.len(), 1, "undo takes back only the second path");
        app.session.redo();
        // A selected saved path takes the next one.
        app.run("path.set", json!({"name": "Saved", "path": {"subpaths": [{"knots": [[2, 2], [12, 2], [12, 8]]}]}})).unwrap();
        app.ui.selected_path = Some("Saved".into());
        draw(&mut app, 220.0);
        let saved = app.session.active().unwrap().doc.paths.iter().find(|p| p.name == "Saved").unwrap().path.clone();
        assert_eq!(saved.subpaths.len(), 2);
        assert_eq!(work(&app).subpaths.len(), 2, "the Work Path is left alone");
        assert_eq!(app.ui.selected_path.as_deref(), Some("Saved"));
        // No path selected: a new Work Path replaces the old one.
        app.ui.selected_path = None;
        draw(&mut app, 320.0);
        let wp = work(&app);
        assert_eq!(wp.subpaths.len(), 1);
        assert_eq!(wp.subpaths[0].knots[0].anchor.x, 320.0);
        assert_eq!(app.ui.selected_path.as_deref(), Some("work"));
    }

    /// #4596: the close-path pointer shows where a click would close the path, and only there:
    /// over the first anchor once the Pen has two anchors (the Curvature Pen three), not while an
    /// anchor is being dragged.
    #[test]
    fn close_path_pointer_shows_where_a_click_closes() {
        let mut app = app();
        app.ui.tool = Tool::Pen;
        assert!(!closes_path_at(&app, Tool::Pen, 20.0, 20.0), "no path yet");
        pen_down(&mut app, 20.0, 20.0);
        pen_up(&mut app);
        assert!(!closes_path_at(&app, Tool::Pen, 20.0, 20.0), "one anchor: a click there doesn't close");
        pen_down(&mut app, 120.0, 20.0);
        assert!(!closes_path_at(&app, Tool::Pen, 20.0, 20.0), "not while dragging the new anchor");
        pen_up(&mut app);
        assert!(closes_path_at(&app, Tool::Pen, 21.0, 22.0), "over the first anchor");
        assert!(!closes_path_at(&app, Tool::Pen, 120.0, 20.0) && !closes_path_at(&app, Tool::Pen, 60.0, 60.0), "the last anchor and empty canvas");
        assert!(!closes_path_at(&app, Tool::CurvaturePen, 20.0, 20.0), "a Pen path isn't the Curvature Pen's");
        pen_down(&mut app, 21.0, 22.0);
        pen_up(&mut app);
        assert!(app.ui.pen.is_none() && !closes_path_at(&app, Tool::Pen, 20.0, 20.0), "the click closed it");

        app.ui.tool = Tool::CurvaturePen;
        for (x, y) in [(20.0, 20.0), (120.0, 20.0)] {
            curvature_down(&mut app, x, y);
            curvature_up(&mut app);
        }
        assert!(!closes_path_at(&app, Tool::CurvaturePen, 20.0, 20.0), "two anchors: the Curvature Pen moves the first");
        curvature_down(&mut app, 120.0, 120.0);
        curvature_up(&mut app);
        assert!(closes_path_at(&app, Tool::CurvaturePen, 20.0, 20.0));
        assert!(!closes_path_at(&app, Tool::CurvaturePen, 120.0, 120.0));
    }

    #[test]
    fn pen_point_undo_preserves_earlier_knots_and_document_history() {
        let mut app = app();
        app.ui.tool = Tool::Pen;
        app.run("shape.create", json!({"kind": "rect", "rect": [10, 10, 40, 40], "fill": "#ff0000"})).unwrap();
        let old_history = app.session.active().unwrap().history.past_len();

        pen_down(&mut app, 20.0, 20.0);
        pen_up(&mut app);
        pen_down(&mut app, 120.0, 20.0);
        pen_move(&mut app, 120.0, 60.0);
        pen_up(&mut app);
        pen_down(&mut app, 120.0, 120.0);
        pen_up(&mut app);

        let original_first_two = app.ui.pen.as_ref().unwrap().knots[..2].to_vec();
        assert!(pen_undo_last_point(&mut app));
        let pen = app.ui.pen.as_ref().unwrap();
        assert_eq!(pen.knots, original_first_two);
        assert!(!pen.dragging, "undo releases the current Pen drag");
        assert_eq!(app.session.active().unwrap().history.past_len(), old_history, "undo must not roll back previous document changes");
        assert!(app.session.active().unwrap().doc.work_path.is_none(), "path is not committed yet");

        pen_commit(&mut app, false);
        let work = app.session.active().unwrap().doc.work_path.as_ref().unwrap();
        assert_eq!(work.subpaths[0].knots.len(), 2);
        assert!(work.subpaths[0].knots[1].smooth, "retained smooth handles are unchanged");
    }

    #[test]
    fn pen_undo_of_first_anchor_cancels_empty_draft() {
        let mut app = app();
        assert!(!pen_undo_last_point(&mut app));
        pen_down(&mut app, 15.0, 17.0);
        pen_up(&mut app);
        assert!(pen_undo_last_point(&mut app));
        assert!(app.ui.pen.is_none());
        assert!(!pen_undo_last_point(&mut app));
        assert_eq!(app.session.active().unwrap().history.past_len(), 0);
        assert!(app.session.active().unwrap().doc.work_path.is_none());
    }

    /// #1419: Solid Color (and every new fill or adjustment layer) made while a path is selected
    /// in the Paths panel takes it as its vector mask; with the path deselected, the selection
    /// is its layer mask again.
    #[test]
    fn new_fill_layer_takes_the_selected_path_as_its_vector_mask() {
        let mut app = app();
        let layer = |app: &PhotocraftApp, r: Result<Value, String>| {
            let id = photocraft_doc::LayerId(r.unwrap()["layer"].as_u64().unwrap());
            app.session.active().unwrap().doc.layer(id).unwrap().clone()
        };
        // Drawing a work path with the Pen selects it.
        app.ui.tool = Tool::Pen;
        for (x, y) in [(20.0, 20.0), (120.0, 20.0), (120.0, 120.0)] {
            pen_down(&mut app, x, y);
            pen_up(&mut app);
        }
        pen_down(&mut app, 20.5, 20.5);
        pen_up(&mut app);
        assert_eq!(app.ui.selected_path.as_deref(), Some("work"));
        let wp = app.session.active().unwrap().doc.work_path.clone().unwrap();
        app.run("select.rect", json!({"x": 150, "y": 150, "width": 20, "height": 20})).unwrap();
        for id in ["layer.newFillLayer.solidColor", "layer.newFillLayer.gradient", "layer.newAdjustmentLayer.invert"] {
            app.ui.selected_path = Some("work".into());
            let r = app.run(id, json!({}));
            let l = layer(&app, r);
            assert_eq!(l.vector_mask.as_ref().map(|v| &v.path), Some(&wp), "{id}: the work path is the vector mask");
            assert!(l.mask.is_none(), "{id}: the path wins over the selection");
            // The new layer's vector mask is now the active path, not the work path.
            assert_eq!(app.ui.selected_path.as_deref(), Some("layer"), "{id}");
        }
        // So the next fill layer isn't masked by the work path again.
        let r = app.run("layer.newFillLayer.solidColor", json!({}));
        assert!(layer(&app, r).vector_mask.is_none());
        // A caller can opt out.
        let r = app.run("layer.newFillLayer.solidColor", json!({"path": null}));
        assert!(layer(&app, r).vector_mask.is_none());
        // A selected saved path is used.
        app.run("path.set", json!({"name": "Saved", "path": {"subpaths": [{"closed": true, "knots": [[0, 0], [9, 0], [9, 9]]}]}})).unwrap();
        app.ui.selected_path = Some("Saved".into());
        let saved = app.session.active().unwrap().doc.paths[0].path.clone();
        let r = app.run("layer.newFillLayer.solidColor", json!({}));
        assert_eq!(layer(&app, r).vector_mask.map(|v| v.path), Some(saved));
        // Deselected (or a stale name): the selection is the layer mask, no vector mask.
        for sel in [None, Some("Gone".to_string())] {
            app.ui.selected_path = sel;
            let r = app.run("layer.newFillLayer.solidColor", json!({}));
            let l = layer(&app, r);
            assert!(l.vector_mask.is_none() && l.mask.is_some());
        }
        // Make Work Path selects the new work path.
        app.run("select.toWorkPath", json!({"tolerance": 2.0})).unwrap();
        assert_eq!(app.ui.selected_path.as_deref(), Some("work"));
    }

    /// #1482: clicking an existing final Pen anchor breaks its outgoing handle without
    /// deleting the preceding curve or creating another knot; dragging changes only that handle.
    #[test]
    fn pen_last_anchor_can_be_broken_and_reshaped_while_drawing() {
        use crate::canvas::{ToolEvent, tool_event};
        let mut app = app();
        app.ui.tool = Tool::Pen;
        let down = |app: &mut PhotocraftApp, x, y| tool_event(app, ToolEvent::Down { x, y, pressure: 1.0 }, egui::Modifiers::NONE);
        let move_to = |app: &mut PhotocraftApp, x, y| tool_event(app, ToolEvent::Move { x, y, pressure: 1.0 }, egui::Modifiers::NONE);
        let up = |app: &mut PhotocraftApp, x, y| tool_event(app, ToolEvent::Up { x, y }, egui::Modifiers::NONE);

        down(&mut app, 20.0, 20.0);
        up(&mut app, 20.0, 20.0);
        down(&mut app, 100.0, 20.0);
        move_to(&mut app, 120.0, 40.0);
        up(&mut app, 120.0, 40.0);
        let original = app.ui.pen.as_ref().unwrap().knots[1];
        assert_eq!(original, [[100.0, 20.0], [80.0, 0.0], [120.0, 40.0]]);

        // Bare click breaks the outgoing handle, preserving the preceding segment's tangent.
        down(&mut app, 100.0, 20.0);
        up(&mut app, 100.0, 20.0);
        let pen = app.ui.pen.as_ref().unwrap();
        assert_eq!(pen.knots.len(), 2);
        assert_eq!(pen.knots[1], [[100.0, 20.0], [80.0, 0.0], [100.0, 20.0]]);
        assert!(!pen.dragging && !pen.adjusting_last);

        // Re-dragging the same final anchor moves only its outgoing handle (a cusp).
        down(&mut app, 100.0, 20.0);
        move_to(&mut app, 110.0, 60.0);
        up(&mut app, 110.0, 60.0);
        let pen = app.ui.pen.as_ref().unwrap();
        assert_eq!(pen.knots.len(), 2);
        assert_eq!(pen.knots[1], [[100.0, 20.0], [80.0, 0.0], [110.0, 60.0]]);

        // Drawing on still adds a separate next anchor and keeps the cusp in the work path.
        down(&mut app, 150.0, 100.0);
        up(&mut app, 150.0, 100.0);
        assert_eq!(app.ui.pen.as_ref().unwrap().knots.len(), 3);
        pen_commit(&mut app, false);
        let work = app.session.active().unwrap().doc.work_path.as_ref().unwrap();
        let k = &work.subpaths[0].knots[1];
        assert_eq!([k.in_ctrl.x, k.in_ctrl.y], [80.0, 0.0]);
        assert_eq!([k.out_ctrl.x, k.out_ctrl.y], [110.0, 60.0]);
        assert!(!k.smooth, "breaking the handle must persist as an unlinked PSD/vector knot");
    }

    #[test]
    fn pen_last_anchor_hit_tolerance_tracks_zoom_and_draft_round_trips() {
        for zoom in [0.25, 1.0, 4.0] {
            let mut app = app();
            app.ui.views[0].zoom = zoom;
            pen_down(&mut app, 20.0, 20.0);
            pen_up(&mut app);
            pen_down(&mut app, 100.0, 100.0);
            pen_move(&mut app, 110.0, 120.0);
            pen_up(&mut app);
            let incoming = app.ui.pen.as_ref().unwrap().knots[1][1];
            pen_down(&mut app, 100.0 + 5.0 / f64::from(zoom), 100.0);
            pen_up(&mut app);
            let pen = app.ui.pen.as_ref().unwrap();
            assert_eq!(pen.knots.len(), 2);
            assert_eq!(pen.knots[1][1], incoming);
            assert_eq!(pen.knots[1][2], pen.knots[1][0]);
            let saved = serde_json::to_value(pen).unwrap();
            let restored: PenPath = serde_json::from_value(saved).unwrap();
            assert_eq!(restored, *pen);
            assert_eq!(pen_to_json(&restored, false)["subpaths"][0]["knots"][1]["smooth"], false);
            pen_down(&mut app, 100.0 + 7.0 / f64::from(zoom), 100.0);
            assert_eq!(app.ui.pen.as_ref().unwrap().knots.len(), 3);
        }
        let legacy: PenPath = serde_json::from_value(json!({"knots": [[[10, 10], [10, 10], [10, 10]]]})).unwrap();
        assert!(legacy.unlinked.is_empty());
        assert!(!legacy.dragging && !legacy.adjusting_last);
    }

    /// #1520: changing one corner through the Properties command must keep the other three
    /// independent, remain a live rectangle, and undo/redo as one geometry edit.
    #[test]
    fn live_rectangle_corner_radius_edit_preserves_other_corners_and_history() {
        let mut app = app();
        let created = app.run("shape.create", json!({"kind": "roundedRect", "rect": [10, 10, 150, 100], "radii": [5, 10, 15, 20]})).unwrap();
        let id = created["layer"].as_u64().unwrap();
        let get_radii = |app: &PhotocraftApp| {
            let st = app.session.active().unwrap();
            let shape = st.doc.layer(photocraft_doc::LayerId(id)).unwrap();
            match &shape.content {
                LayerContent::Shape(sh) => match &sh.live {
                    Some(photocraft_doc::vector::LiveShape::Rect { radii, .. }) => *radii,
                    _ => panic!("rectangle must remain a live shape"),
                },
                _ => panic!("expected shape layer"),
            }
        };
        assert_eq!(get_radii(&app), [5.0, 10.0, 15.0, 20.0]);
        let mut patch = corner_radii_patch(get_radii(&app), 1, 30.0);
        patch["layer"] = json!(id);
        app.run("shape.edit", patch).unwrap();
        assert_eq!(get_radii(&app), [5.0, 30.0, 15.0, 20.0]);
        app.run("edit.undo", json!({})).unwrap();
        assert_eq!(get_radii(&app), [5.0, 10.0, 15.0, 20.0]);
        app.run("edit.redo", json!({})).unwrap();
        assert_eq!(get_radii(&app), [5.0, 30.0, 15.0, 20.0]);
        for corner in 0..4 {
            let before = get_radii(&app);
            let radius = 35.0 + corner as f32;
            let mut expected = before;
            expected[corner] = f64::from(radius);
            let mut patch = corner_radii_patch(before, corner, radius);
            patch["layer"] = json!(id);
            app.run("shape.edit", patch).unwrap();
            assert_eq!(get_radii(&app), expected);
            app.run("edit.undo", json!({})).unwrap();
            assert_eq!(get_radii(&app), before);
            app.run("edit.redo", json!({})).unwrap();
            assert_eq!(get_radii(&app), expected);
        }
        app.run("shape.edit", json!({"layer": id, "radii": 12})).unwrap();
        assert_eq!(get_radii(&app), [12.0; 4]);
    }

    /// #534: dragging in a shape's fill picker, opened from the Properties panel at the right edge
    /// of the window, keeps the picker in place (it flipped from side to side as its width
    /// followed the colour readouts, which moved it under the pointer).
    #[test]
    fn shape_colour_picker_stays_put_while_dragging() {
        use egui::{PointerButton, vec2};
        use egui_kittest::kittest::Queryable;
        let mut app = app();
        let id = app.run("shape.create", json!({"kind": "rect", "rect": [10, 10, 100, 60], "fill": "#3070c0"})).unwrap()["layer"].as_u64().unwrap();
        let mut h = egui_kittest::Harness::builder().with_size(vec2(1440.0, 900.0)).build_ui_state(
            move |ui, app: &mut PhotocraftApp| {
                // Fonts set up after the first frame only apply from the next one.
                if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    return;
                }
                egui::Panel::right("properties").resizable(false).exact_size(285.0).show(ui, |ui| shape_properties(app, ui, photocraft_doc::LayerId(id)));
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Pro);
        h.run_steps(4);
        let label = h.query_all(egui_kittest::kittest::by().label("Fill").include_labels()).next().expect("the Fill label").rect();
        // The swatch is right of its label.
        let swatch = egui::pos2(label.right() + 17.0, label.center().y);
        let press = |h: &mut egui_kittest::Harness<'_, PhotocraftApp>, pos, pressed| {
            h.event(egui::Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE });
            h.step();
        };
        h.hover_at(swatch);
        h.step();
        press(&mut h, swatch, true);
        press(&mut h, swatch, false);
        h.run_steps(3);
        let popup = |h: &egui_kittest::Harness<'_, PhotocraftApp>| {
            h.ctx.memory(|m| m.areas().visible_layer_ids().into_iter().filter(|l| l.order == egui::Order::Foreground).find_map(|l| m.area_rect(l.id)))
        };
        let opened = popup(&h).expect("the picker opened");
        let fill = |h: &egui_kittest::Harness<'_, PhotocraftApp>| {
            let st = h.state().session.active().unwrap();
            match &st.doc.layer(photocraft_doc::LayerId(id)).unwrap().content {
                LayerContent::Shape(sh) => sh.fill.clone(),
                _ => None,
            }
        };
        let before = fill(&h);
        // Drag across the picker's colour area.
        let start = opened.min + vec2(60.0, opened.height() * 0.5);
        h.hover_at(start);
        h.step();
        press(&mut h, start, true);
        for i in 1..=10 {
            h.event(egui::Event::PointerMoved(start + vec2(i as f32 * 5.0, i as f32)));
            h.step();
            let r = popup(&h).expect("the picker stays open");
            assert_eq!(r.left_top(), opened.left_top(), "step {i}: the picker moved from {opened:?} to {r:?}");
            assert!(h.ctx.content_rect().contains_rect(r), "step {i}: {r:?} runs off the window");
        }
        press(&mut h, start, false);
        assert_ne!(fill(&h), before, "the drag picked a colour");
    }

    /// The options bar edits the selected shape's fill and stroke.
    #[test]
    fn shape_options_bar_edits_the_selected_shape() {
        use egui::accesskit::Role;
        use egui_kittest::kittest::Queryable;
        let mut app = app();
        app.run("tools.setColors", json!({"foreground": "#ff0000", "background": "#0000ff"})).unwrap();
        finish_shape(&mut app, Tool::Rectangle, [20.0, 20.0], [120.0, 100.0], Default::default());
        let shape = |app: &PhotocraftApp| {
            let st = app.session.active().unwrap();
            match &st.doc.layer(st.active_layer.unwrap()).unwrap().content {
                LayerContent::Shape(sh) => (sh.fill.clone(), sh.stroke.clone()),
                _ => panic!("not a shape layer"),
            }
        };
        assert!(shape(&app).0.is_some() && shape(&app).1.is_none());
        let mut h = egui_kittest::Harness::builder().with_size(vec2(1100.0, 48.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    ui.horizontal_centered(|ui| options_bar(app, ui, Tool::Rectangle));
                }
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Studio);
        h.run_steps(4);
        let click_fill = |h: &mut egui_kittest::Harness<'_, PhotocraftApp>| {
            let label = h.query_all(egui_kittest::kittest::by().label("Fill:").include_labels()).next().expect("the Fill label").rect();
            let at = egui::pos2(label.right() + h.ctx.global_style().spacing.item_spacing.x + 7.0, label.center().y);
            h.hover_at(at);
            h.step();
            for pressed in [true, false] {
                h.event(egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE });
                h.step();
            }
            h.run_steps(2);
        };
        // Stroke width.
        h.get_all_by_role(Role::SpinButton).next().expect("the stroke width").click();
        h.run();
        for c in "6".chars() {
            h.event(egui::Event::Text(c.to_string()));
            h.run();
        }
        h.key_press(egui::Key::Enter);
        h.run_steps(2);
        let stroke = shape(h.state()).1.expect("the width field stroked the shape");
        assert_eq!(stroke.width, 6.0);
        assert_eq!(stroke.paint, photocraft_doc::Fill::Solid(photocraft_doc::Color::rgba(0.0, 0.0, 1.0, 1.0)));
        // Fill off.
        click_fill(&mut h);
        assert_eq!(shape(h.state()).0, None, "unchecking Fill removed the shape's fill");
        assert!(!h.state().ui.tool_options.shape_fill, "and the next shape has none either");
        h.state_mut().run("edit.undo", json!({})).unwrap();
        assert!(shape(h.state()).0.is_some());
        // No shape selected: no edit.
        h.state_mut().run("layer.new.layer", json!({})).unwrap();
        let history = h.state().session.active().unwrap().history.past_len();
        h.run_steps(2);
        click_fill(&mut h);
        assert_eq!(h.state().session.active().unwrap().history.past_len(), history);
    }

    /// #3915: Path Selection and Direct Selection options bars show Fill, Stroke, and Radius
    /// controls when a shape is selected, and editing them edits the selected shape.
    #[test]
    fn path_selection_options_bar_edits_the_selected_shape() {
        use egui::accesskit::Role;
        use egui_kittest::kittest::Queryable;
        let mut app = app();
        app.run("tools.setColors", json!({"foreground": "#ff0000", "background": "#0000ff"})).unwrap();
        finish_shape(&mut app, Tool::Rectangle, [20.0, 20.0], [120.0, 100.0], Default::default());
        let shape = |app: &PhotocraftApp| {
            let st = app.session.active().unwrap();
            match &st.doc.layer(st.active_layer.unwrap()).unwrap().content {
                LayerContent::Shape(sh) => (sh.fill.clone(), sh.stroke.clone()),
                _ => panic!("not a shape layer"),
            }
        };
        assert!(shape(&app).0.is_some() && shape(&app).1.is_none());
        let mut h = egui_kittest::Harness::builder().with_size(vec2(1100.0, 48.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    ui.horizontal_centered(|ui| options_bar(app, ui, Tool::PathSelection));
                }
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Studio);
        h.run_steps(4);
        let click_fill = |h: &mut egui_kittest::Harness<'_, PhotocraftApp>| {
            let label = h.query_all(egui_kittest::kittest::by().label("Fill:").include_labels()).next().expect("the Fill label").rect();
            let at = egui::pos2(label.right() + h.ctx.global_style().spacing.item_spacing.x + 7.0, label.center().y);
            h.hover_at(at);
            h.step();
            for pressed in [true, false] {
                h.event(egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE });
                h.step();
            }
            h.run_steps(2);
        };
        // Stroke width field is available on Path Selection bar too.
        h.get_all_by_role(Role::SpinButton).next().expect("the stroke width").click();
        h.run();
        for c in "6".chars() {
            h.event(egui::Event::Text(c.to_string()));
            h.run();
        }
        h.key_press(egui::Key::Enter);
        h.run_steps(2);
        let stroke = shape(h.state()).1.expect("the width field stroked the shape");
        assert_eq!(stroke.width, 6.0);
        assert_eq!(stroke.paint, photocraft_doc::Fill::Solid(photocraft_doc::Color::rgba(0.0, 0.0, 1.0, 1.0)));
        // Fill toggle.
        click_fill(&mut h);
        assert_eq!(shape(h.state()).0, None, "unchecking Fill removed the shape's fill");
        // No shape selected: renders hint text instead.
        h.state_mut().run("layer.new.layer", json!({})).unwrap();
        h.run_steps(2);
        assert!(h.query_all(egui_kittest::kittest::by().label("Fill:").include_labels()).next().is_none());
    }
}

#[cfg(test)]
#[path = "path_selection_tests.rs"]
mod path_selection_tests;

#[cfg(test)]
#[path = "pen_draw_tests.rs"]
mod pen_draw_tests;

#[cfg(test)]
#[path = "vector_paint_tests.rs"]
mod paint_tests;
