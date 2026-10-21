//! Image › Analysis and Notes in the shell: the Ruler, Count, Note and Color Sampler tools
//! (eyedropper group), their canvas overlays and options bars, Window › Measurement Log and
//! Window › Notes, and the Set Measurement Scale / Select Data Points / Place Scale Marker dialogs.
//!
//! Every change goes through engine commands (`image.analysis.*`, `count.*`, `notes.*`,
//! `view.colorSamplers.*`, `measurementLog.*`); this module only holds view state ([`AnalysisUi`],
//! part of the serialisable UI state so the control channel can read and drive it). The Color
//! Sampler points themselves live in the engine (`DocState.color_samplers`) so every front end
//! shares them; this module only maps pointer input and draws the numbered markers.

use egui::{Align2, Color32, CornerRadius, FontId, Pos2, RichText, Stroke, pos2, vec2};
use photocraft_doc::{Document, Ruler};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::canvas::{ToolEvent, ViewXform};
use crate::state::Tool;
use crate::theme::Tokens;

/// What a pointer drag is doing (not serialised).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Drag {
    /// New ruler line from a point.
    RulerNew,
    /// Moving a ruler handle: 0 start, 1 end, 2 protractor end.
    RulerHandle(u8),
    /// Moving count marker (group, index) from where it was.
    Count(usize, usize, [f64; 2]),
    /// Moving note `index`: grab offset from its position.
    Note(usize, [f64; 2]),
    /// Moving Color Sampler point `index` from where the press landed (a click, not a drag, does
    /// nothing; dragging off the canvas removes the point).
    Sampler(usize, [f64; 2]),
}

/// View state of the analysis tools and panels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AnalysisUi {
    /// Window › Measurement Log / Notes panels.
    pub measurement_log: bool,
    pub notes: bool,
    /// Measurement Log rows selected (row ids).
    pub log_selected: Vec<u64>,
    /// Notes panel: the note shown (index).
    pub note_selected: Option<usize>,
    /// Note tool options: author and colour of new notes.
    pub note_author: String,
    pub note_color: [f32; 3],
    /// Ruler options bar: show values in the measurement scale's units.
    pub use_measurement_scale: bool,
    /// Open dialog: (kind, fields). Kinds: "scale", "dataPoints", "scaleMarker".
    pub dialog: Option<(String, Map<String, Value>)>,
    #[serde(skip)]
    pub drag: Option<Drag>,
}

impl Default for AnalysisUi {
    fn default() -> Self {
        Self {
            measurement_log: false,
            notes: false,
            log_selected: Vec::new(),
            note_selected: None,
            note_author: String::new(),
            note_color: [1.0, 1.0, 0.51],
            use_measurement_scale: false,
            dialog: None,
            drag: None,
        }
    }
}

const PANELS: [&str; 2] = ["window.panel.measurementLog", "window.panel.notes"];

/// Menu ids handled here (live menu items).
pub fn handles(id: &str) -> bool {
    PANELS.contains(&id)
}

pub fn checked(app: &PhotocraftApp, id: &str) -> Option<bool> {
    match id {
        "window.panel.measurementLog" => Some(app.ui.analysis.measurement_log),
        "window.panel.notes" => Some(app.ui.analysis.notes),
        "image.analysis.rulerTool" => Some(app.ui.tool == Tool::Ruler),
        "image.analysis.countTool" => Some(app.ui.tool == Tool::Count),
        _ => {
            // View › Proof Setup simulations: checked while that proof is shown.
            // A check item with no document too: an item's kind never changes (native menus).
            let kind = id.strip_prefix("view.proofSetup.").and_then(photocraft_engine::proof_sim::ProofKind::from_id)?;
            let Some(d) = app.session.active() else { return Some(false) };
            let pv = app.session.color.proof(d.doc.id);
            Some(pv.enabled && pv.setup.kind == kind)
        }
    }
}

fn no_params(p: &Value) -> bool {
    p.as_object().is_none_or(|o| o.is_empty())
}

fn fields(v: Value) -> Map<String, Value> {
    v.as_object().cloned().unwrap_or_default()
}

/// Menu front ends: panel toggles, tool selection, dialogs and pickers in front of the engine's
/// analysis commands. `None` when `id` isn't ours (or the engine should run it directly).
pub fn menu(app: &mut PhotocraftApp, id: &str, params: &Value) -> Option<Result<Value, String>> {
    let a = &mut app.ui.analysis;
    match id {
        "window.panel.measurementLog" | "window.panel.notes" => {
            let slot = if id.ends_with("measurementLog") { &mut a.measurement_log } else { &mut a.notes };
            *slot = params.get("show").and_then(Value::as_bool).unwrap_or(!*slot);
            return Some(Ok(json!({"visible": *slot})));
        }
        _ => {}
    }
    if !no_params(params) {
        return None;
    }
    let doc = app.session.active().map(|d| d.doc.clone());
    Some(match id {
        "image.analysis.rulerTool" => {
            app.ui.tool = Tool::Ruler;
            Ok(json!({"tool": "Ruler"}))
        }
        "image.analysis.countTool" => {
            app.ui.tool = Tool::Count;
            Ok(json!({"tool": "Count"}))
        }
        "image.analysis.setMeasurementScale" => {
            let d = doc?;
            let sc = &d.measurement.scale;
            open(
                app,
                "scale",
                json!({"preset": if sc.is_default() { "default" } else { "custom" }, "pixelLength": sc.pixel_length, "logicalLength": sc.logical_length, "units": sc.units}),
            )
        }
        "image.analysis.selectDataPoints" => {
            let dp = serde_json::to_value(&app.session.analysis.data_points).unwrap_or_default();
            open(app, "dataPoints", dp)
        }
        "image.analysis.placeScaleMarker" => {
            let d = doc?;
            let units = d.measurement.scale.units.clone();
            open(
                app,
                "scaleMarker",
                json!({"length": suggested_length(&d), "units": units, "fontSize": 12.0, "displayText": true, "textPosition": "bottom", "color": "black"}),
            )
        }
        "image.analysis.recordMeasurements" => {
            // The current tool picks the source, as in Photoshop.
            let source = match app.ui.tool {
                Tool::Ruler => "ruler",
                Tool::Count => "count",
                _ => "auto",
            };
            let r = app.run(id, json!({"source": source}));
            if r.is_ok() {
                app.ui.analysis.measurement_log = true;
            }
            r
        }
        "file.import.notes" => match app.active_doc_id() {
            Ok(doc) => app.pick_file_bytes(move |app, name, bytes| {
                app.refocus(doc)?;
                let r = photocraft_engine::notes_cmds::import_notes_from(&mut app.session, &name, &bytes).map_err(|e| e.to_string());
                if r.is_ok() {
                    app.ui.analysis.notes = true;
                    app.sync_views();
                }
                r
            }),
            Err(e) => Err(e),
        },
        "measurementLog.export" => export_log(app, None),
        _ => return None,
    })
}

fn open(app: &mut PhotocraftApp, kind: &str, f: Value) -> Result<Value, String> {
    app.ui.analysis.dialog = Some((kind.to_string(), fields(f)));
    Ok(json!({"dialog": kind}))
}

/// A "nice" bar length near a fifth of the image width (same rule as the engine default).
fn suggested_length(d: &Document) -> f64 {
    let target = f64::from(d.size.width) / 5.0 * d.measurement.scale.factor();
    let e = 10f64.powf(target.log10().floor());
    [5.0, 2.0, 1.0].into_iter().map(|m| m * e).find(|v| *v <= target).unwrap_or(e)
}

/// Export the Measurement Log (or the given rows) as CSV through the save picker.
fn export_log(app: &mut PhotocraftApp, rows: Option<Vec<u64>>) -> Result<Value, String> {
    let p = rows.map_or(json!({}), |r| json!({"rows": r}));
    let csv = app.run("measurementLog.export", p)?;
    let text = csv["csv"].as_str().unwrap_or_default().to_string();
    app.pick_save("Measurements.csv", move |app, path| {
        let write = app.services.write.as_mut().ok_or("no writer configured")?;
        write(&path, text.as_bytes())?;
        Ok(json!({"path": path, "rows": csv["rows"]}))
    })
}

// ------------------------------------------------------------------ tools

fn tolerance(app: &PhotocraftApp) -> f64 {
    7.0 / f64::from(app.point_zoom().max(0.01))
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// Shift constrains to 45° steps around `from`.
fn constrain(from: [f64; 2], to: [f64; 2], shift: bool) -> [f64; 2] {
    if !shift {
        return to;
    }
    let (dx, dy) = (to[0] - from[0], to[1] - from[1]);
    let a = (dy.atan2(dx) / std::f64::consts::FRAC_PI_4).round() * std::f64::consts::FRAC_PI_4;
    let l = dx.hypot(dy);
    [from[0] + l * a.cos(), from[1] + l * a.sin()]
}

fn set_ruler(app: &mut PhotocraftApp, r: Ruler) {
    let _ = app.run("image.analysis.rulerTool", json!({"start": r.start, "end": r.end, "protractor": r.protractor}));
}

/// Pointer input for the Ruler, Count, Note and Color Sampler tools. Returns true when consumed.
pub fn pointer(app: &mut PhotocraftApp, ev: ToolEvent, mods: egui::Modifiers) -> bool {
    let tool = app.active_tool();
    if !matches!(tool, Tool::Ruler | Tool::Count | Tool::Note | Tool::ColorSampler) {
        return false;
    }
    let Some(doc) = app.session.active().map(|d| d.doc.clone()) else {
        // With no document the Color Sampler does nothing and says so, as the other tools are inert.
        if tool == Tool::ColorSampler && matches!(ev, ToolEvent::Down { .. }) {
            app.ui.status = tl!("No document open").into();
            app.ui.status_error = false;
        }
        return true;
    };
    let tol = tolerance(app);
    let samplers: Vec<[f64; 2]> = app.session.active().map(|d| d.color_samplers.clone()).unwrap_or_default();
    match (tool, ev) {
        (Tool::Ruler, ToolEvent::Down { x, y, .. }) => {
            let p = [x, y];
            let cur = doc.measurement.ruler;
            let near = |q: [f64; 2]| dist(p, q) <= tol;
            match cur {
                // ⌥-drag from an end point starts the protractor arm at that vertex.
                Some(mut r) if mods.alt && (near(r.start) || near(r.end)) => {
                    if near(r.end) && !near(r.start) {
                        std::mem::swap(&mut r.start, &mut r.end);
                    }
                    r.protractor = Some(p);
                    set_ruler(app, r);
                    app.ui.analysis.drag = Some(Drag::RulerHandle(2));
                }
                Some(r) if r.protractor.is_some_and(near) => app.ui.analysis.drag = Some(Drag::RulerHandle(2)),
                Some(r) if near(r.end) => app.ui.analysis.drag = Some(Drag::RulerHandle(1)),
                Some(r) if near(r.start) => app.ui.analysis.drag = Some(Drag::RulerHandle(0)),
                _ => {
                    set_ruler(app, Ruler { start: p, end: p, protractor: None });
                    app.ui.analysis.drag = Some(Drag::RulerNew);
                }
            }
        }
        (Tool::Ruler, ToolEvent::Move { x, y, .. }) => {
            let (Some(drag), Some(mut r)) = (app.ui.analysis.drag, doc.measurement.ruler) else { return true };
            match drag {
                Drag::RulerNew | Drag::RulerHandle(1) => r.end = constrain(r.start, [x, y], mods.shift),
                Drag::RulerHandle(0) => r.start = constrain(r.end, [x, y], mods.shift),
                Drag::RulerHandle(_) => r.protractor = Some(constrain(r.start, [x, y], mods.shift)),
                _ => return true,
            }
            set_ruler(app, r);
        }
        (Tool::Ruler, ToolEvent::Up { .. })
            if app.ui.analysis.drag.take() == Some(Drag::RulerNew) && doc.measurement.ruler.is_some_and(|r| dist(r.start, r.end) < 0.5) =>
        {
            let _ = app.run("image.analysis.rulerTool", json!({"clear": true}));
        }
        (Tool::Count, ToolEvent::Down { x, y, .. }) => {
            let hit = nearest_marker(&doc, [x, y], tol);
            match hit {
                Some((g, i)) if mods.alt => {
                    let _ = app.run("count.remove", json!({"group": g, "index": i}));
                }
                Some((g, i)) => app.ui.analysis.drag = Some(Drag::Count(g, i, [x, y])),
                None if !mods.alt => {
                    let _ = app.run("count.add", json!({"x": x, "y": y}));
                }
                None => {}
            }
        }
        (Tool::Count, ToolEvent::Up { x, y }) => {
            if let Some(Drag::Count(g, i, from)) = app.ui.analysis.drag.take()
                && dist(from, [x, y]) > 0.5
            {
                let _ = app.run("count.move", json!({"group": g, "index": i, "to": [x, y]}));
            }
        }
        (Tool::Note, ToolEvent::Down { x, y, .. }) => {
            let z = f64::from(app.point_zoom().max(0.01));
            let hit =
                doc.notes.iter().rposition(|n| x >= n.position[0] && x <= n.position[0] + 16.0 / z && y >= n.position[1] && y <= n.position[1] + 20.0 / z);
            match hit {
                Some(i) => {
                    app.ui.analysis.note_selected = Some(i);
                    app.ui.analysis.notes = true;
                    app.ui.analysis.drag = Some(Drag::Note(i, [x - doc.notes[i].position[0], y - doc.notes[i].position[1]]));
                }
                None => {
                    let a = &app.ui.analysis;
                    let p = json!({"x": x.round(), "y": y.round(), "author": a.note_author, "color": [a.note_color[0], a.note_color[1], a.note_color[2]]});
                    if let Ok(r) = app.run("notes.add", p) {
                        app.ui.analysis.note_selected = r["index"].as_u64().map(|v| v as usize);
                        app.ui.analysis.notes = true;
                    }
                }
            }
        }
        (Tool::Note, ToolEvent::Up { x, y }) => {
            if let Some(Drag::Note(i, off)) = app.ui.analysis.drag.take() {
                let to = [(x - off[0]).round(), (y - off[1]).round()];
                if doc.notes.get(i).is_some_and(|n| dist(n.position, to) >= 1.0) {
                    let _ = app.run("notes.set", json!({"index": i, "x": to[0], "y": to[1]}));
                }
            }
        }
        (Tool::ColorSampler, ToolEvent::Down { x, y, .. }) => {
            let p = [x, y];
            match nearest_sampler(&samplers, p, tol) {
                // ⌥-click a point removes it.
                Some(i) if mods.alt => {
                    let _ = app.run("view.colorSamplers.delete", json!({"index": i}));
                }
                // A plain click on a point starts a move (committed on release).
                Some(i) => app.ui.analysis.drag = Some(Drag::Sampler(i, p)),
                // ⌥-click on empty space does nothing; a plain click inside places a new point.
                None if mods.alt => {}
                None if (0.0..f64::from(doc.size.width)).contains(&x) && (0.0..f64::from(doc.size.height)).contains(&y) => {
                    let r = app.run("view.colorSamplers.add", json!({"x": x, "y": y}));
                    status_on_err(app, r);
                }
                // A click outside the canvas places nothing, as in Photoshop.
                None => {}
            }
        }
        (Tool::ColorSampler, ToolEvent::Up { x, y }) => {
            if let Some(Drag::Sampler(i, from)) = app.ui.analysis.drag.take() {
                if dist(from, [x, y]) <= 0.5 {
                    return true; // a click on a point, not a drag: nothing to do
                }
                // Dropped inside the canvas: move it. Dropped outside: remove just that point.
                let inside = (0.0..f64::from(doc.size.width)).contains(&x) && (0.0..f64::from(doc.size.height)).contains(&y);
                let r = if inside {
                    app.run("view.colorSamplers.move", json!({"index": i, "x": x, "y": y}))
                } else {
                    app.run("view.colorSamplers.delete", json!({"index": i}))
                };
                status_on_err(app, r);
            }
        }
        (Tool::ColorSampler, ToolEvent::Move { .. }) => {}
        _ => {}
    }
    true
}

/// The nearest Color Sampler point within `tol` document pixels, or `None`.
fn nearest_sampler(points: &[[f64; 2]], at: [f64; 2], tol: f64) -> Option<usize> {
    points.iter().enumerate().filter(|(_, q)| dist(**q, at) <= tol).min_by(|a, b| dist(*a.1, at).total_cmp(&dist(*b.1, at))).map(|(i, _)| i)
}

/// Show a command error in the status bar without treating the sample-limit message as a fault.
fn status_on_err(app: &mut PhotocraftApp, r: Result<Value, String>) {
    if let Err(e) = r {
        app.ui.status = e;
        app.ui.status_error = false;
    }
}

fn nearest_marker(d: &Document, at: [f64; 2], tol: f64) -> Option<(usize, usize)> {
    let mut best: Option<(f64, usize, usize)> = None;
    for (gi, g) in d.measurement.count_groups.iter().enumerate().filter(|(_, g)| g.visible) {
        for (pi, q) in g.points.iter().enumerate() {
            let dd = dist(*q, at);
            if dd <= tol && best.is_none_or(|b| dd < b.0) {
                best = Some((dd, gi, pi));
            }
        }
    }
    best.map(|b| (b.1, b.2))
}

fn color32(c: [f32; 3]) -> Color32 {
    Color32::from_rgb((c[0].clamp(0.0, 1.0) * 255.0) as u8, (c[1].clamp(0.0, 1.0) * 255.0) as u8, (c[2].clamp(0.0, 1.0) * 255.0) as u8)
}

fn handle(painter: &egui::Painter, p: Pos2) {
    for (w, c) in [(3.0, Color32::BLACK), (1.0, Color32::WHITE)] {
        painter.line_segment([p - vec2(5.0, 0.0), p + vec2(5.0, 0.0)], Stroke::new(w, c));
        painter.line_segment([p - vec2(0.0, 5.0), p + vec2(0.0, 5.0)], Stroke::new(w, c));
    }
}

/// Ruler line (while the Ruler tool is active), count markers and note icons.
pub fn draw_overlay(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    let Some(doc) = app.session.active().map(|d| d.doc.clone()) else { return };
    let v = &app.ui.view;
    if app.ui.tool == Tool::Ruler
        && let Some(r) = doc.measurement.ruler
    {
        let s = xf.to_screen(r.start[0] as f32, r.start[1] as f32);
        let e = xf.to_screen(r.end[0] as f32, r.end[1] as f32);
        let mut segs = vec![[s, e]];
        if let Some(q) = r.protractor {
            segs.push([s, xf.to_screen(q[0] as f32, q[1] as f32)]);
        }
        for seg in &segs {
            painter.line_segment(*seg, Stroke::new(3.0, Color32::from_black_alpha(160)));
            painter.line_segment(*seg, Stroke::new(1.0, Color32::WHITE));
            handle(painter, seg[1]);
        }
        handle(painter, s);
    }
    if v.shows(v.show.count) {
        for g in doc.measurement.count_groups.iter().filter(|g| g.visible) {
            let c = color32(g.color.to_rgb());
            for (i, p) in g.points.iter().enumerate() {
                let at = xf.to_screen(p[0] as f32, p[1] as f32);
                painter.circle_filled(at, 1.0 + g.marker_size as f32, c);
                let font = FontId::proportional(g.label_size as f32);
                painter.text(at + vec2(2.0 + g.marker_size as f32, -1.0), Align2::LEFT_BOTTOM, format!("{}", i + 1), font, c);
            }
        }
    }
    if v.shows(v.show.notes) {
        for (i, n) in doc.notes.iter().enumerate() {
            let p = xf.to_screen(n.position[0] as f32, n.position[1] as f32);
            let r = egui::Rect::from_min_size(p, vec2(16.0, 20.0));
            let fill = color32(n.color.to_rgb());
            painter.rect_filled(r, CornerRadius::same(2), fill);
            let sel = app.ui.analysis.note_selected == Some(i);
            painter.rect_stroke(
                r,
                CornerRadius::same(2),
                Stroke::new(if sel { 2.0 } else { 1.0 }, if sel { Color32::WHITE } else { Color32::from_black_alpha(200) }),
                egui::StrokeKind::Outside,
            );
            // Folded corner and text lines.
            let ink = Color32::from_black_alpha(140);
            painter.line_segment([r.right_top() + vec2(-5.0, 0.0), r.right_top() + vec2(0.0, 5.0)], Stroke::new(1.0, ink));
            for k in 0..3 {
                let y = r.top() + 7.0 + 4.0 * k as f32;
                painter.line_segment([pos2(r.left() + 3.0, y), pos2(r.right() - 3.0, y)], Stroke::new(1.0, ink));
            }
        }
    }
    // Color Sampler points (#1046): persistent numbered markers, drawn whatever the active tool is.
    for (i, p) in app.session.active().map(|d| d.color_samplers.clone()).unwrap_or_default().iter().enumerate() {
        let at = xf.to_screen(p[0] as f32, p[1] as f32);
        // A small target reticle, double-stroked so it reads on any pixel.
        for (w, c) in [(3.0, Color32::from_black_alpha(170)), (1.0, Color32::WHITE)] {
            painter.circle_stroke(at, 5.0, Stroke::new(w, c));
            painter.line_segment([at - vec2(8.0, 0.0), at - vec2(6.0, 0.0)], Stroke::new(w, c));
            painter.line_segment([at + vec2(6.0, 0.0), at + vec2(8.0, 0.0)], Stroke::new(w, c));
            painter.line_segment([at - vec2(0.0, 8.0), at - vec2(0.0, 6.0)], Stroke::new(w, c));
            painter.line_segment([at + vec2(0.0, 6.0), at + vec2(0.0, 8.0)], Stroke::new(w, c));
        }
        // The number, upper right so several points stay readable when they overlap.
        let font = FontId::proportional(11.0);
        let anchor = at + vec2(7.0, -7.0);
        painter.text(anchor + vec2(1.0, 1.0), Align2::LEFT_BOTTOM, format!("{}", i + 1), font.clone(), Color32::from_black_alpha(180));
        painter.text(anchor, Align2::LEFT_BOTTOM, format!("{}", i + 1), font, Color32::WHITE);
    }
}

// ------------------------------------------------------------------ options bars

fn readout(ui: &mut egui::Ui, label: &str, value: Option<f64>) {
    let t = Tokens::get(ui.ctx());
    ui.label(RichText::new(tl!(&label)).color(t.text_dim).size(11.0));
    ui.label(RichText::new(value.map_or(String::new(), |v| format!("{v:.2}"))).color(t.text).size(11.0).monospace());
    ui.add_space(6.0);
}

/// Options bar of the Ruler, Count, Note and Color Sampler tools. Returns true when drawn.
pub fn options_bar(app: &mut PhotocraftApp, ui: &mut egui::Ui, tool: Tool) -> bool {
    let Some(doc) = app.session.active().map(|d| d.doc.clone()) else {
        return matches!(tool, Tool::Ruler | Tool::Count | Tool::Note | Tool::ColorSampler);
    };
    match tool {
        Tool::Ruler => {
            let sc = doc.measurement.scale.clone();
            let f = if app.ui.analysis.use_measurement_scale { sc.factor() } else { 1.0 };
            let info = doc.measurement.ruler.map(|r| photocraft_engine::analysis_cmds::ruler_info(&r, &sc));
            let g = |k: &str| info.as_ref().and_then(|i| i[k].as_f64());
            readout(ui, "X:", g("x").map(|v| v * f));
            readout(ui, "Y:", g("y").map(|v| v * f));
            readout(ui, "W:", g("w").map(|v| v * f));
            readout(ui, "H:", g("h").map(|v| v * f));
            readout(ui, "A:", g("angle"));
            readout(ui, "L1:", g("l1").map(|v| v * f));
            readout(ui, "L2:", g("l2").map(|v| v * f));
            crate::widgets::vline(ui, 22.0);
            crate::widgets::checkbox(ui, &mut app.ui.analysis.use_measurement_scale, tl!("Use Measurement Scale"));
            let has = doc.measurement.ruler.is_some();
            if ui.add_enabled_ui(has, |ui| crate::widgets::secondary_button(ui, tl!("Straighten Layer"), 0.0)).inner.clicked() {
                let _ = app.run("image.analysis.straightenLayer", json!({}));
            }
            if ui.add_enabled_ui(has, |ui| crate::widgets::secondary_button(ui, tl!("Clear"), 0.0)).inner.clicked() {
                let _ = app.run("image.analysis.rulerTool", json!({"clear": true}));
            }
            true
        }
        Tool::Count => {
            let t = Tokens::get(ui.ctx());
            let m = &doc.measurement;
            ui.label(RichText::new(crate::i18n::fmt(tl!("Count: {n}"), &[("n", &m.count_total().to_string())])).color(t.text).size(11.0));
            crate::widgets::vline(ui, 22.0);
            if !m.count_groups.is_empty() {
                let mut gi = m.active_count_group.min(m.count_groups.len() - 1);
                let opts: Vec<(usize, &str)> = m.count_groups.iter().enumerate().map(|(i, g)| (i, g.name.as_str())).collect();
                if crate::widgets::dropdown(ui, "count-group", &mut gi, &opts, 140.0) {
                    let _ = app.run("count.setGroup", json!({"group": gi, "active": true}));
                }
                let g = &m.count_groups[gi];
                let eye = if g.visible { "eye" } else { "eye-off" };
                if crate::icons::button(ui, eye, 22.0, false, tl!("Toggle count group visibility")).clicked() {
                    let _ = app.run("count.setGroup", json!({"group": gi, "visible": !g.visible}));
                }
                let rgb = g.color.to_rgb();
                // PhotoCraft's Color Picker (sRGB), as everywhere else (#2144).
                if crate::widgets::color_swatch_button(ui, color32(rgb), tl!("Color")).clicked() {
                    crate::color_picker_ui::open_for_command(
                        app,
                        &crate::color_picker_ui::title_for(tl!("Color")),
                        rgb,
                        "count.setGroup",
                        json!({"group": gi}),
                    );
                }
                if crate::icons::button(ui, "trash", 22.0, false, tl!("Delete count group")).clicked() {
                    let _ = app.run("count.deleteGroup", json!({"group": gi}));
                }
            }
            if crate::icons::button(ui, "folder-plus", 22.0, false, tl!("Create a new count group")).clicked() {
                let _ = app.run("count.newGroup", json!({}));
            }
            if crate::widgets::secondary_button(ui, tl!("Clear"), 0.0).clicked() && m.count_total() > 0 {
                let _ = app.run("count.clear", json!({}));
            }
            if let Some(g) = m.count_groups.get(m.active_count_group) {
                crate::widgets::vline(ui, 22.0);
                let (mut ms, mut ls) = (g.marker_size as f32, g.label_size as f32);
                ui.label(RichText::new(tl!("Marker Size")).color(t.text_dim).size(11.0));
                let a = crate::widgets::value_field(ui, &mut ms, 1.0..=10.0, "", 40.0).changed();
                ui.label(RichText::new(tl!("Label Size")).color(t.text_dim).size(11.0));
                let b = crate::widgets::value_field(ui, &mut ls, 8.0..=72.0, "", 44.0).changed();
                if a || b {
                    let _ = app.run("count.setGroup", count_size_params(ms, ls));
                }
            }
            true
        }
        Tool::Note => {
            let t = Tokens::get(ui.ctx());
            ui.label(RichText::new(tl!("Author:")).color(t.text_dim).size(11.0));
            ui.add(egui::TextEdit::singleline(&mut app.ui.analysis.note_author).desired_width(120.0));
            ui.label(RichText::new(tl!("Color:")).color(t.text_dim).size(11.0));
            let rgb = app.ui.analysis.note_color;
            if crate::widgets::color_swatch_button(ui, color32(rgb), tl!("Color")).clicked() {
                crate::color_picker_ui::open_for_target(app, "noteColor", &crate::color_picker_ui::title_for(tl!("Color")), rgb);
            }
            if ui.add_enabled_ui(!doc.notes.is_empty(), |ui| crate::widgets::secondary_button(ui, tl!("Clear All"), 0.0)).inner.clicked() {
                let _ = app.run("notes.delete", json!({"all": true}));
                app.ui.analysis.note_selected = None;
            }
            if crate::icons::button(ui, "message-square", 22.0, app.ui.analysis.notes, tl!("Show or hide the Notes panel")).clicked() {
                app.ui.analysis.notes = !app.ui.analysis.notes;
            }
            true
        }
        Tool::ColorSampler => {
            let t = Tokens::get(ui.ctx());
            let n = app.session.active().map_or(0, |d| d.color_samplers.len());
            ui.label(
                RichText::new(crate::i18n::fmt(
                    tl!("{n} of {max} sample points"),
                    &[("n", &n.to_string()), ("max", &photocraft_engine::sampler_cmds::MAX_SAMPLERS.to_string())],
                ))
                .color(t.text)
                .size(11.0),
            );
            crate::widgets::vline(ui, 22.0);
            if ui.add_enabled_ui(n > 0, |ui| crate::widgets::secondary_button(ui, tl!("Clear All"), 0.0)).inner.clicked() {
                let _ = app.run("view.colorSamplers.clear", json!({}));
            }
            crate::widgets::vline(ui, 22.0);
            ui.label(RichText::new(tl!("Click to place · Alt-click or drag off-canvas to remove")).color(t.text_dim).size(11.0));
            true
        }
        _ => false,
    }
}

// ------------------------------------------------------------------ panels and dialogs

/// `count.setGroup` params for the options bar's Marker Size and Label Size fields, sent as JSON
/// integers (the fields edit `f32`s; `8.0` is a float on the wire).
fn count_size_params(marker: f32, label: f32) -> Value {
    let whole = |v: f32, lo: f32, hi: f32| if v.is_finite() { v.round().clamp(lo, hi) as u32 } else { lo as u32 };
    json!({"markerSize": whole(marker, 1.0, 10.0), "labelSize": whole(label, 8.0, 72.0)})
}

fn float_frame(t: &Tokens) -> egui::Frame {
    egui::Frame::NONE
        .fill(t.card)
        .stroke(Stroke::new(1.0, t.card_border))
        .corner_radius(CornerRadius::same(t.radius_lg as u8))
        .shadow(egui::Shadow { offset: [0, 10], blur: 30, spread: 0, color: t.shadow })
        .inner_margin(egui::Margin::same(8))
}

/// Title row with a close button; returns true when closed.
pub(crate) fn title_row(ui: &mut egui::Ui, title: &str) -> bool {
    let t = Tokens::get(ui.ctx());
    let mut close = false;
    ui.horizontal(|ui| {
        ui.label(RichText::new(tl!(&title)).color(t.text).size(12.0).strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if crate::icons::button(ui, "x", 20.0, false, tl!("Close")).clicked() {
                close = true;
            }
        });
    });
    crate::widgets::hairline(ui);
    ui.add_space(4.0);
    close
}

/// A floating panel window; `movable` follows Window › Workspace › Lock Workspace.
pub(crate) fn panel_window(app: &PhotocraftApp, ctx: &egui::Context, id: &str, title: &str, offset: egui::Vec2, width: f32, body: impl FnOnce(&mut egui::Ui)) {
    let t = Tokens::get(ctx);
    let canvas = app.last_canvas_rect;
    egui::Window::new(title)
        .id(egui::Id::new(id))
        .title_bar(false)
        .resizable(false)
        .movable(!app.session.prefs().workspace_locked)
        .min_width(width)
        .max_width(width)
        .frame(float_frame(&t))
        .default_pos(pos2((canvas.right() - width - 16.0 + offset.x).max(canvas.left() + 8.0), canvas.top() + offset.y))
        .show(ctx, |ui| {
            ui.set_width(width);
            ui.set_max_width(width);
            body(ui);
        });
}

/// Draws the open panels and dialogs.
pub fn windows(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if app.ui.analysis.measurement_log {
        let mut act: Option<(&str, Value)> = None;
        let mut close = false;
        let rows = app.session.analysis.log.clone();
        let mut sel = app.ui.analysis.log_selected.clone();
        let t = Tokens::get(ctx);
        panel_window(app, ctx, "measurement-log", tl!("Measurement Log"), vec2(-300.0, 470.0), 620.0, |ui| {
            close = title_row(ui, tl!("Measurement Log"));
            ui.horizontal(|ui| {
                if crate::widgets::secondary_button(ui, tl!("Record Measurements"), 0.0).clicked() {
                    act = Some(("record", Value::Null));
                }
                if crate::widgets::secondary_button(ui, tl!("Select All"), 0.0).clicked() {
                    sel = rows.iter().map(|r| r.id).collect();
                }
                if crate::widgets::secondary_button(ui, tl!("Deselect All"), 0.0).clicked() {
                    sel.clear();
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if crate::icons::button(ui, "trash", 22.0, false, tl!("Delete selected measurements")).clicked() && !sel.is_empty() {
                        act = Some(("measurementLog.delete", json!({"rows": sel.clone()})));
                    }
                    if crate::icons::button(ui, "file", 22.0, false, tl!("Export selected measurements (CSV)")).clicked() {
                        act = Some(("export", json!(sel.clone())));
                    }
                });
            });
            let cols: Vec<&str> = photocraft_engine::analysis_cmds::log_columns(&rows).into_iter().filter(|c| *c != "histogram").collect();
            egui::ScrollArea::both().max_height(220.0).max_width(620.0).show(ui, |ui| {
                egui::Grid::new("mlog-grid").striped(true).spacing(vec2(10.0, 2.0)).show(ui, |ui| {
                    for c in &cols {
                        let name = photocraft_engine::analysis_cmds::COLUMNS.iter().find(|x| x.0 == *c).map_or(*c, |x| x.1);
                        ui.label(RichText::new(tl!(name)).color(t.text_dim).size(10.5).strong());
                    }
                    ui.end_row();
                    for r in &rows {
                        let on = sel.contains(&r.id);
                        for (k, c) in cols.iter().enumerate() {
                            let v = r.values.get(*c).map(|v| match v {
                                Value::String(s) => s.clone(),
                                Value::Null => String::new(),
                                other => other.to_string(),
                            });
                            let txt = RichText::new(v.unwrap_or_default()).size(10.5).color(if on { t.accent_text } else { t.text });
                            let resp = if k == 0 { ui.selectable_label(on, txt) } else { ui.label(txt) };
                            if k == 0 && resp.clicked() {
                                if on {
                                    sel.retain(|x| *x != r.id);
                                } else {
                                    sel.push(r.id);
                                }
                            }
                        }
                        ui.end_row();
                    }
                });
            });
        });
        app.ui.analysis.log_selected = sel;
        if close {
            app.ui.analysis.measurement_log = false;
        }
        match act {
            Some(("record", _)) => {
                let _ = crate::menus::invoke(app, ctx, "image.analysis.recordMeasurements", json!({}));
            }
            Some(("export", rows)) => {
                let ids: Vec<u64> = rows.as_array().map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default();
                if let Err(e) = export_log(app, (!ids.is_empty()).then_some(ids))
                    && e != "cancelled"
                {
                    app.ui.status = e;
                }
            }
            Some((id, p)) => {
                let _ = app.run(id, p).inspect(|_| app.ui.analysis.log_selected.clear());
            }
            None => {}
        }
    }
    if app.ui.analysis.notes {
        notes_panel(app, ctx);
    }
    if app.ui.analysis.dialog.is_some() {
        dialog(app, ctx);
    }
}

fn notes_panel(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let notes = app.session.active().map(|d| d.doc.notes.clone()).unwrap_or_default();
    let n = notes.len();
    let mut sel = app.ui.analysis.note_selected.filter(|i| *i < n).or((n > 0).then_some(0));
    let mut close = false;
    let mut edit: Option<Value> = None;
    let mut delete = false;
    let t = Tokens::get(ctx);
    panel_window(app, ctx, "notes-panel", tl!("Notes"), vec2(-40.0, 300.0), 260.0, |ui| {
        close = title_row(ui, tl!("Notes"));
        match sel {
            None => {
                ui.label(RichText::new(tl!("Click with the Note tool to add a note.")).color(t.text_dim).size(11.0));
            }
            Some(i) => {
                let note = &notes[i];
                ui.label(RichText::new(if note.author.is_empty() { tl!("No author").to_string() } else { note.author.clone() }).color(t.text_dim).size(11.0));
                let mut text = note.text.clone();
                if ui.add(egui::TextEdit::multiline(&mut text).desired_rows(6).desired_width(f32::INFINITY)).changed() {
                    edit = Some(json!({"index": i, "text": text, "coalesce": format!("note-{i}")}));
                }
            }
        }
        crate::widgets::hairline(ui);
        ui.horizontal(|ui| {
            if ui.add_enabled_ui(n > 1, |ui| crate::icons::button(ui, "chevrons-left", 22.0, false, tl!("Previous note"))).inner.clicked() {
                sel = sel.map(|i| (i + n - 1) % n);
            }
            if ui.add_enabled_ui(n > 1, |ui| crate::icons::button(ui, "chevrons-right", 22.0, false, tl!("Next note"))).inner.clicked() {
                sel = sel.map(|i| (i + 1) % n);
            }
            let template = "{index} of {total}";
            let position = crate::i18n::fmt(tl!(template), &[("index", &sel.map_or(0, |i| i + 1).to_string()), ("total", &n.to_string())]);
            ui.label(RichText::new(position).color(t.text_dim).size(11.0));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.add_enabled_ui(sel.is_some(), |ui| crate::icons::button(ui, "trash", 22.0, false, tl!("Delete note"))).inner.clicked() {
                    delete = true;
                }
            });
        });
    });
    app.ui.analysis.note_selected = sel;
    if let Some(p) = edit {
        let _ = app.run("notes.set", p);
    }
    if delete
        && let Some(i) = sel
        && app.run("notes.delete", json!({"index": i})).is_ok()
    {
        app.ui.analysis.note_selected = (n > 1).then(|| i.min(n - 2));
    }
    if close {
        app.ui.analysis.notes = false;
    }
}

fn num_field(ui: &mut egui::Ui, f: &mut Map<String, Value>, key: &str, label: &str, range: std::ops::RangeInclusive<f32>, suffix: &str) {
    let t = Tokens::get(ui.ctx());
    ui.horizontal(|ui| {
        ui.add_sized(vec2(110.0, 18.0), egui::Label::new(RichText::new(tl!(&label)).color(t.text_dim).size(11.0)));
        let mut v = f.get(key).and_then(Value::as_f64).unwrap_or(0.0) as f32;
        if crate::widgets::value_field(ui, &mut v, range, suffix, 80.0).changed() {
            f.insert(key.into(), json!(v));
        }
    });
}

fn choice(ui: &mut egui::Ui, f: &mut Map<String, Value>, key: &str, label: &str, opts: &[(&str, &str)]) {
    let t = Tokens::get(ui.ctx());
    ui.horizontal(|ui| {
        ui.add_sized(vec2(110.0, 18.0), egui::Label::new(RichText::new(tl!(&label)).color(t.text_dim).size(11.0)));
        let mut cur = f.get(key).and_then(Value::as_str).unwrap_or(opts[0].0).to_string();
        let list: Vec<(String, &str)> = opts.iter().map(|(k, l)| (k.to_string(), *l)).collect();
        if crate::widgets::dropdown(ui, &format!("an-{key}"), &mut cur, &list, 120.0) {
            f.insert(key.into(), json!(cur));
        }
    });
}

/// The params the open dialog's OK runs with (engine command, params).
pub fn dialog_command(kind: &str, f: &Map<String, Value>) -> Option<(&'static str, Value)> {
    Some(match kind {
        "scale" => {
            if f.get("preset").and_then(Value::as_str) == Some("default") {
                ("image.analysis.setMeasurementScale", json!({"preset": "default"}))
            } else {
                (
                    "image.analysis.setMeasurementScale",
                    json!({"preset": "custom", "pixelLength": f.get("pixelLength"), "logicalLength": f.get("logicalLength"), "units": f.get("units")}),
                )
            }
        }
        "dataPoints" => ("image.analysis.selectDataPoints", json!({"selection": f.get("selection"), "ruler": f.get("ruler"), "count": f.get("count")})),
        "scaleMarker" => (
            "image.analysis.placeScaleMarker",
            json!({"length": f.get("length"), "fontSize": f.get("fontSize"), "displayText": f.get("displayText"), "textPosition": f.get("textPosition"), "color": f.get("color")}),
        ),
        _ => return None,
    })
}

fn dialog(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let Some((kind, mut f)) = app.ui.analysis.dialog.clone() else { return };
    let t = Tokens::get(ctx);
    let title = match kind.as_str() {
        "scale" => "Measurement Scale",
        "dataPoints" => "Select Data Points",
        _ => "Measurement Scale Marker",
    };
    let mut result: Option<bool> = None;
    egui::Window::new(title)
        .id(egui::Id::new("analysis-dialog"))
        .collapsible(false)
        .resizable(false)
        .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
        .frame(float_frame(&t))
        .show(ctx, |ui| {
            ui.label(RichText::new(tl!(&title)).color(t.text).size(13.0).strong());
            crate::widgets::hairline(ui);
            ui.add_space(6.0);
            match kind.as_str() {
                "scale" => {
                    choice(ui, &mut f, "preset", tl!("Presets:"), &[("custom", tl!("Custom")), ("default", tl!("Default"))]);
                    let custom = f.get("preset").and_then(Value::as_str) != Some("default");
                    ui.add_enabled_ui(custom, |ui| {
                        num_field(ui, &mut f, "pixelLength", tl!("Pixel Length:"), 0.001..=1e7, "");
                        num_field(ui, &mut f, "logicalLength", tl!("Logical Length:"), 0.0001..=1e7, "");
                        ui.horizontal(|ui| {
                            ui.add_sized(vec2(110.0, 18.0), egui::Label::new(RichText::new(tl!("Logical Units:")).color(t.text_dim).size(11.0)));
                            let mut u = f.get("units").and_then(Value::as_str).unwrap_or("pixels").to_string();
                            if ui.add(egui::TextEdit::singleline(&mut u).desired_width(80.0)).changed() {
                                f.insert("units".into(), json!(u));
                            }
                        });
                    });
                    let (pl, ll) = (f.get("pixelLength").and_then(Value::as_f64).unwrap_or(1.0), f.get("logicalLength").and_then(Value::as_f64).unwrap_or(1.0));
                    let u = f.get("units").and_then(Value::as_str).unwrap_or("pixels");
                    let text = if custom { format!("{pl} pixels = {ll:.4} {u}") } else { "1 pixels = 1.0000 pixels".into() };
                    ui.label(RichText::new(tl!(&text)).color(t.text_dim).size(11.0));
                }
                "dataPoints" => {
                    ui.horizontal_top(|ui| {
                        for (source, label) in [("selection", tl!("Selections")), ("ruler", tl!("Ruler Tool")), ("count", tl!("Count Tool"))] {
                            ui.vertical(|ui| {
                                ui.label(RichText::new(tl!(&label)).color(t.text).size(11.0).strong());
                                let mut list: Vec<String> = f
                                    .get(source)
                                    .and_then(Value::as_array)
                                    .map(|a| a.iter().filter_map(Value::as_str).map(String::from).collect())
                                    .unwrap_or_default();
                                let mut changed = false;
                                for key in photocraft_engine::analysis_cmds::available(source) {
                                    let name = photocraft_engine::analysis_cmds::COLUMNS.iter().find(|c| c.0 == key).map_or(key, |c| c.1);
                                    let mut on = list.iter().any(|x| x == key);
                                    if crate::widgets::checkbox(ui, &mut on, name).changed() {
                                        list.retain(|x| x != key);
                                        if on {
                                            list.push(key.to_string());
                                        }
                                        changed = true;
                                    }
                                }
                                if changed {
                                    f.insert(source.into(), json!(list));
                                }
                            });
                            ui.add_space(12.0);
                        }
                    });
                }
                _ => {
                    let units = f.get("units").and_then(Value::as_str).unwrap_or("pixels").to_string();
                    num_field(ui, &mut f, "length", tl!("Length:"), 0.0001..=1e7, &units);
                    num_field(ui, &mut f, "fontSize", tl!("Font Size:"), 1.0..=1000.0, "pt");
                    let mut show = f.get("displayText").and_then(Value::as_bool).unwrap_or(true);
                    if crate::widgets::checkbox(ui, &mut show, tl!("Display Text")).changed() {
                        f.insert("displayText".into(), json!(show));
                    }
                    choice(ui, &mut f, "textPosition", tl!("Text Position:"), &[("bottom", tl!("Bottom")), ("top", tl!("Top"))]);
                    choice(ui, &mut f, "color", tl!("Color:"), &[("black", tl!("Black")), ("white", tl!("White"))]);
                }
            }
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if let Some(role) = crate::widgets::dialog_buttons(
                        ui,
                        &[
                            crate::widgets::DialogButton::new(crate::widgets::ButtonRole::Default, tl!("OK"), 70.0),
                            crate::widgets::DialogButton::new(crate::widgets::ButtonRole::Cancel, tl!("Cancel"), 70.0),
                        ],
                    ) {
                        result = Some(role == crate::widgets::ButtonRole::Default);
                    }
                });
            });
        });
    match result {
        Some(true) => {
            app.ui.analysis.dialog = None;
            if let Some((id, p)) = dialog_command(&kind, &f) {
                let _ = app.run(id, p);
            }
        }
        Some(false) => app.ui.analysis.dialog = None,
        None => app.ui.analysis.dialog = Some((kind, f)),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn app() -> (PhotocraftApp, egui::Context) {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 200, "height": 100})).unwrap();
        (app, egui::Context::default())
    }

    /// Draw `f` for a few frames (fonts load on the first).
    pub(crate) fn render(app: &mut PhotocraftApp, ctx: &egui::Context, f: fn(&mut PhotocraftApp, &egui::Context)) {
        PhotocraftApp::setup_context(ctx, Default::default());
        for _ in 0..3 {
            let mut o = ctx.run_ui(Default::default(), |ui| f(app, ui.ctx()));
            o.textures_delta.clear();
        }
    }

    fn down(app: &mut PhotocraftApp, x: f64, y: f64, m: egui::Modifiers) {
        crate::canvas::tool_event(app, ToolEvent::Down { x, y, pressure: 1.0 }, m);
    }
    fn mv(app: &mut PhotocraftApp, x: f64, y: f64, m: egui::Modifiers) {
        crate::canvas::tool_event(app, ToolEvent::Move { x, y, pressure: 1.0 }, m);
    }
    fn up(app: &mut PhotocraftApp, x: f64, y: f64) {
        crate::canvas::tool_event(app, ToolEvent::Up { x, y }, egui::Modifiers::NONE);
    }

    fn ruler(app: &PhotocraftApp) -> Option<Ruler> {
        app.session.active().unwrap().doc.measurement.ruler
    }

    #[test]
    fn menu_selects_tools_and_toggles_panels() {
        let (mut app, ctx) = app();
        crate::menus::invoke(&mut app, &ctx, "image.analysis.rulerTool", json!({})).unwrap();
        assert_eq!(app.ui.tool, Tool::Ruler);
        assert_eq!(checked(&app, "image.analysis.rulerTool"), Some(true));
        crate::menus::invoke(&mut app, &ctx, "image.analysis.countTool", json!({})).unwrap();
        assert_eq!(app.ui.tool, Tool::Count);
        crate::menus::invoke(&mut app, &ctx, "window.panel.measurementLog", json!({})).unwrap();
        assert!(app.ui.analysis.measurement_log);
        crate::menus::invoke(&mut app, &ctx, "window.panel.notes", json!({"show": true})).unwrap();
        assert!(app.ui.analysis.notes);
        let items = crate::menus::menu_items(&app);
        for id in [
            "window.panel.measurementLog",
            "window.panel.notes",
            "image.analysis.setMeasurementScale",
            "image.analysis.placeScaleMarker",
            "file.import.notes",
            "view.proofSetup.workingCyanPlate",
        ] {
            assert!(items.iter().any(|i| i.id == id && i.enabled), "{id} live");
        }
    }

    #[test]
    fn arbitrary_rotation_starts_at_the_ruler_angle() {
        let (mut app, ctx) = app();
        let open = |app: &mut PhotocraftApp| {
            let r = crate::menus::invoke(app, &ctx, "image.rotation.arbitrary", json!({})).unwrap();
            let f = &app.ui.dialog_mut(r["dialog"].as_u64().unwrap()).unwrap().fields;
            (f["angle"].as_f64().unwrap(), f["direction"].clone())
        };
        assert_eq!(open(&mut app), (0.0, json!("cw")));
        // A line falling to the right straightens counter-clockwise, a near-vertical one to the y axis.
        for (end, dir) in [([150, 60], "ccw"), ([150, 40], "cw"), ([60, -50], "ccw")] {
            app.run("image.analysis.rulerTool", json!({"start": [50, 50], "end": end})).unwrap();
            let (a, d) = open(&mut app);
            assert!((a - 5.7106).abs() < 1e-3 && d == json!(dir), "{end:?}: {a} {d}");
        }
    }

    #[test]
    fn ruler_drag_protractor_and_clear() {
        let (mut app, _) = app();
        app.ui.tool = Tool::Ruler;
        let none = egui::Modifiers::NONE;
        down(&mut app, 10.0, 10.0, none);
        mv(&mut app, 60.0, 10.0, none);
        up(&mut app, 60.0, 10.0);
        assert_eq!(ruler(&app).map(|r| r.end), Some([60.0, 10.0]));
        // Drag the end point with ⇧: constrained to 45°.
        down(&mut app, 60.0, 10.0, none);
        mv(&mut app, 12.0, 52.0, egui::Modifiers::SHIFT);
        up(&mut app, 12.0, 52.0);
        let r = ruler(&app).unwrap();
        assert!((r.end[0] - 10.0).abs() < 1e-6, "vertical: {r:?}");
        // ⌥-drag from the start point: protractor arm.
        down(&mut app, 10.0, 10.0, egui::Modifiers::ALT);
        mv(&mut app, 60.0, 10.0, none);
        up(&mut app, 60.0, 10.0);
        let info = photocraft_engine::analysis_cmds::ruler_info(&ruler(&app).unwrap(), &Default::default());
        assert!((info["angle"].as_f64().unwrap() - 90.0).abs() < 1e-6, "{info}");
        // A click without a drag far away starts a new line and clears it.
        down(&mut app, 150.0, 80.0, none);
        up(&mut app, 150.0, 80.0);
        assert!(ruler(&app).is_none());
        assert_eq!(app.session.active().unwrap().history.past_len(), 0, "the ruler adds no history");
    }

    #[test]
    fn count_tool_add_move_remove() {
        let (mut app, _) = app();
        app.ui.tool = Tool::Count;
        down(&mut app, 20.0, 20.0, egui::Modifiers::NONE);
        up(&mut app, 20.0, 20.0);
        down(&mut app, 50.0, 50.0, egui::Modifiers::NONE);
        up(&mut app, 50.0, 50.0);
        let pts = |app: &PhotocraftApp| app.session.active().unwrap().doc.measurement.count_groups[0].points.clone();
        assert_eq!(pts(&app).len(), 2);
        // Drag marker 1.
        down(&mut app, 20.5, 20.5, egui::Modifiers::NONE);
        up(&mut app, 30.0, 25.0);
        assert_eq!(pts(&app)[0], [30.0, 25.0]);
        // ⌥-click removes.
        down(&mut app, 50.0, 50.0, egui::Modifiers::ALT);
        up(&mut app, 50.0, 50.0);
        assert_eq!(pts(&app).len(), 1);
    }

    #[test]
    fn note_tool_and_panel_state() {
        let (mut app, ctx) = app();
        app.ui.tool = Tool::Note;
        app.ui.analysis.note_author = "QA".into();
        down(&mut app, 30.0, 40.0, egui::Modifiers::NONE);
        up(&mut app, 30.0, 40.0);
        let n = app.session.active().unwrap().doc.notes.clone();
        assert_eq!(n.len(), 1);
        assert_eq!(n[0].author, "QA");
        assert!(app.ui.analysis.notes && app.ui.analysis.note_selected == Some(0));
        // Drag the icon.
        down(&mut app, 32.0, 42.0, egui::Modifiers::NONE);
        up(&mut app, 62.0, 52.0);
        assert_eq!(app.session.active().unwrap().doc.notes[0].position, [60.0, 50.0]);
        // The panel renders without panicking.
        render(&mut app, &ctx, windows);
    }

    #[test]
    fn dialogs_open_and_run_their_commands() {
        let (mut app, ctx) = app();
        let r = crate::menus::invoke(&mut app, &ctx, "image.analysis.setMeasurementScale", json!({})).unwrap();
        assert_eq!(r["dialog"], "scale");
        let (kind, mut f) = app.ui.analysis.dialog.clone().unwrap();
        f.insert("preset".into(), json!("custom"));
        f.insert("pixelLength".into(), json!(40));
        f.insert("logicalLength".into(), json!(1));
        f.insert("units".into(), json!("cm"));
        let (id, p) = dialog_command(&kind, &f).unwrap();
        app.run(id, p).unwrap();
        assert_eq!(app.session.active().unwrap().doc.measurement.scale.units, "cm");
        crate::menus::invoke(&mut app, &ctx, "image.analysis.placeScaleMarker", json!({})).unwrap();
        let (kind, f) = app.ui.analysis.dialog.clone().unwrap();
        assert_eq!(f["length"], 1.0);
        let (id, p) = dialog_command(&kind, &f).unwrap();
        app.run(id, p).unwrap();
        crate::menus::invoke(&mut app, &ctx, "image.analysis.selectDataPoints", json!({})).unwrap();
        let (kind, mut f) = app.ui.analysis.dialog.clone().unwrap();
        f.insert("ruler".into(), json!(["label", "length"]));
        let (id, p) = dialog_command(&kind, &f).unwrap();
        app.run(id, p).unwrap();
        assert_eq!(app.session.analysis.data_points.ruler, ["label", "length"]);
        // Record from the Count tool records counts and opens the log.
        app.ui.tool = Tool::Count;
        crate::menus::invoke(&mut app, &ctx, "image.analysis.recordMeasurements", json!({})).unwrap();
        assert!(app.ui.analysis.measurement_log);
        assert_eq!(app.session.analysis.log.last().unwrap().values["source"], "Count Tool");
        render(&mut app, &ctx, windows);
    }

    #[test]
    fn color_sampler_tool_places_moves_and_removes_points() {
        let (mut app, ctx) = app();
        app.ui.tool = Tool::ColorSampler;
        let pts = |app: &PhotocraftApp| app.session.active().unwrap().color_samplers.clone();
        down(&mut app, 20.0, 20.0, egui::Modifiers::NONE);
        up(&mut app, 20.0, 20.0);
        down(&mut app, 60.0, 40.0, egui::Modifiers::NONE);
        up(&mut app, 60.0, 40.0);
        assert_eq!(pts(&app), vec![[20.0, 20.0], [60.0, 40.0]]);
        // Drag the first point to a new pixel (committed on release).
        down(&mut app, 20.0, 20.0, egui::Modifiers::NONE);
        mv(&mut app, 30.0, 25.0, egui::Modifiers::NONE);
        up(&mut app, 30.0, 25.0);
        assert_eq!(pts(&app)[0], [30.0, 25.0]);
        // A click on a point, with no drag, leaves it where it is.
        down(&mut app, 30.0, 25.0, egui::Modifiers::NONE);
        up(&mut app, 30.0, 25.0);
        assert_eq!(pts(&app)[0], [30.0, 25.0]);
        // ⌥-click removes that one point.
        down(&mut app, 60.0, 40.0, egui::Modifiers::ALT);
        up(&mut app, 60.0, 40.0);
        assert_eq!(pts(&app).len(), 1);
        // Dragging a point off the canvas removes it.
        down(&mut app, 30.0, 25.0, egui::Modifiers::NONE);
        mv(&mut app, -50.0, 25.0, egui::Modifiers::NONE);
        up(&mut app, -50.0, 25.0);
        assert!(pts(&app).is_empty());
        assert_eq!(app.session.active().unwrap().history.past_len(), 0, "sampling adds no history");
        // The overlay and the options bar render without panicking.
        down(&mut app, 10.0, 10.0, egui::Modifiers::NONE);
        up(&mut app, 10.0, 10.0);
        render(&mut app, &ctx, windows);
        let xf = ViewXform {
            rect: egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(200.0, 100.0)),
            zoom: 1.0,
            center: [100.0, 50.0],
            flip: false,
            rotation: 0.0,
            aspect: 1.0,
        };
        let mut o = ctx.run_ui(Default::default(), |ui| draw_overlay(&app, ui.painter(), &xf));
        o.textures_delta.clear();
    }

    #[test]
    fn color_sampler_readout_and_crop_removal_report() {
        let (mut app, _) = app();
        app.ui.tool = Tool::ColorSampler;
        down(&mut app, 5.0, 5.0, egui::Modifiers::NONE);
        up(&mut app, 5.0, 5.0);
        down(&mut app, 150.0, 90.0, egui::Modifiers::NONE);
        up(&mut app, 150.0, 90.0);
        // The list reports both, numbered, with their position and RGB/CMYK readout.
        let l = app.run("view.colorSamplers.list", json!({})).unwrap();
        assert_eq!(l["samplers"].as_array().unwrap().len(), 2);
        assert_eq!(l["samplers"][0]["number"], 1);
        assert_eq!(l["samplers"][1]["position"], json!([150.0, 90.0]));
        assert!(l["samplers"][0]["rgb"].as_array().is_some_and(|a| a.len() == 3));
        // Cropping to the top-left drops the outside point and reports how many went.
        app.run("image.crop", json!({"x": 0, "y": 0, "width": 10, "height": 10})).unwrap();
        assert_eq!(app.session.active().unwrap().color_samplers, vec![[5.0, 5.0]]);
        assert!(app.ui.status.contains("removed"), "{}", app.ui.status);
        assert!(!app.ui.status_error);
    }

    #[test]
    fn color_sampler_is_per_document_and_inert_without_one() {
        let (mut app, _) = app();
        app.ui.tool = Tool::ColorSampler;
        down(&mut app, 10.0, 10.0, egui::Modifiers::NONE);
        up(&mut app, 10.0, 10.0);
        // A second document starts empty; the first keeps its own point.
        app.run("file.new", json!({"width": 50, "height": 50})).unwrap();
        assert!(app.session.active().unwrap().color_samplers.is_empty());
        down(&mut app, 20.0, 20.0, egui::Modifiers::NONE);
        up(&mut app, 20.0, 20.0);
        assert_eq!(app.session.active().unwrap().color_samplers, vec![[20.0, 20.0]]);
        app.run("document.activate", json!({"document": 0})).unwrap();
        assert_eq!(app.session.active().unwrap().color_samplers, vec![[10.0, 10.0]]);
        // With no document the tool says so (no dialog, no panic) and the commands error.
        let mut empty = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        empty.ui.tool = Tool::ColorSampler;
        down(&mut empty, 5.0, 5.0, egui::Modifiers::NONE);
        assert_eq!(empty.ui.status, "No document open");
        assert!(!empty.ui.status_error);
        assert!(empty.run("view.colorSamplers.add", json!({"x": 1, "y": 1})).is_err());
        assert!(empty.run("view.colorSamplers.list", json!({})).is_err());
    }

    #[test]
    fn count_size_fields_send_whole_numbers_the_engine_applies() {
        let p = count_size_params(7.6, 20.0);
        assert_eq!(p, json!({"markerSize": 8, "labelSize": 20}));
        assert!(p["markerSize"].is_u64() && p["labelSize"].is_u64());
        assert_eq!(count_size_params(f32::NAN, 500.0), json!({"markerSize": 1, "labelSize": 72}));
        let (mut app, _ctx) = app();
        app.run("count.add", json!({"x": 10, "y": 10})).unwrap();
        app.run("count.setGroup", count_size_params(8.0, 20.0)).unwrap();
        let g = &app.session.active().unwrap().doc.measurement.count_groups[0];
        assert_eq!((g.marker_size, g.label_size), (8, 20));
    }
}
