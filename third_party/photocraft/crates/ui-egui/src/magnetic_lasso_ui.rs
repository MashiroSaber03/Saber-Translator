//! Magnetic Lasso Tool (L): a selection border that snaps to edges as the pointer moves.
//!
//! Click to set the first fastening point, then move the pointer along an edge (with the button up
//! or held down): the border from the last fastening point to the pointer follows the most
//! prominent edges within the detection width, and points fasten by themselves as it goes, more
//! often at a higher Frequency. A click fastens a point; ⌥-click draws a straight segment and
//! ⌥-drag a freehand one. ⌫ or Delete removes the last fastening point. Clicking the first point,
//! a double-click or ↩ closes the border along the edges (⌥ with a straight segment); Esc cancels.
//! [ and ] narrow or widen the detection width by 1 px. As in Photoshop, the border follows the
//! active layer's pixels (the composite for a layer without pixels of its own).
//!
//! The border closes through `select.magneticLasso` with `"trace": false`: the selection is exactly
//! the outline drawn, in one history step, and an action replays that outline.

use egui::{Color32, Key, Modifiers, Pos2, Rect, Stroke, vec2};
use photocraft_algo::magnetic::{self, Settings};
use photocraft_engine::magnetic_cmds::EdgeSource;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::PhotocraftApp;
use crate::canvas::{ToolEvent, ViewXform};
use crate::state::Tool;

/// A border being drawn (document px).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct MagneticLasso {
    /// The fastened border so far, from the first fastening point to the last.
    pub path: Vec<[f64; 2]>,
    /// Indices into `path` of the fastening points (the first is 0).
    pub anchors: Vec<usize>,
    /// The live segment from the last fastening point to the edge near the pointer.
    pub live: Vec<[f64; 2]>,
    /// Where the pointer has been since the last fastening point: edges are looked for near it.
    pub trail: Vec<[f64; 2]>,
    /// The selection mode the first click set (`replace`, `add`, `subtract`, `intersect`).
    pub mode: String,
    /// The document the border is drawn on.
    pub doc: u64,
    /// An ⌥-drag is drawing a freehand segment.
    pub freehand: bool,
}

impl MagneticLasso {
    /// A border is being drawn.
    pub fn active(&self) -> bool {
        !self.path.is_empty()
    }
}

/// The pixels being traced: a snapshot of the document, kept while a border is drawn.
#[derive(Default)]
pub struct Runtime {
    source: Option<EdgeSource>,
    /// Pointer moves were batched (`PhotocraftApp::defer_live_stroke`): trace once at the end.
    pending: bool,
}

/// Detection settings from the options bar, for an event with pen `pressure`.
fn settings(app: &PhotocraftApp, pressure: f32) -> Settings {
    let o = &app.ui.tool_options;
    let mut width = f64::from(o.magnetic_width);
    // Stylus Pressure: pressing harder narrows the width (a pen only; a mouse has no pressure).
    if o.magnetic_pressure && app.stylus.sample().is_some() && pressure.is_finite() {
        width = 1.0 + (width - 1.0) * f64::from(1.0 - pressure.clamp(0.0, 1.0));
    }
    Settings::new(width, o.magnetic_contrast / 100.0)
}

/// How far the border runs before a point fastens by itself (document px): 8 screen points at
/// Frequency 100, 98 at 0, 47 at the default 57.
fn spacing(app: &PhotocraftApp) -> f64 {
    let f = app.ui.tool_options.magnetic_frequency;
    let f = if f.is_finite() { f64::from(f.clamp(0.0, 100.0)) } else { 57.0 };
    (8.0 + (100.0 - f) * 0.9) / f64::from(app.point_zoom().max(0.01))
}

/// Within this distance of the first point (document px), a click closes the border.
fn close_radius(app: &PhotocraftApp) -> f64 {
    8.0 / f64::from(app.point_zoom().max(0.01))
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// The edge source for the active document, refreshed when the document or layer changed.
fn source(app: &mut PhotocraftApp) -> Option<&mut EdgeSource> {
    if !app.magnetic.source.as_ref().is_some_and(|s| s.is_current(&app.session)) {
        app.magnetic.source = EdgeSource::new(&app.session);
    }
    app.magnetic.source.as_mut()
}

fn snap(app: &mut PhotocraftApp, p: [f64; 2], s: Settings) -> [f64; 2] {
    source(app).and_then(|src| src.snap(p, s)).unwrap_or(p)
}

fn trace(app: &mut PhotocraftApp, from: [f64; 2], to: [f64; 2], guide: &[[f64; 2]], s: Settings) -> Vec<[f64; 2]> {
    let t = source(app).map(|src| src.trace(from, to, guide, s)).unwrap_or_default();
    if t.is_empty() { vec![from, to] } else { t }
}

/// Pointer events while the Magnetic Lasso is the tool (it takes them all).
pub fn pointer(app: &mut PhotocraftApp, ev: ToolEvent, mods: Modifiers) -> bool {
    // ⌘ held makes it the Move tool (`hold_keys::cmd_moves`): nothing to take then.
    if app.active_tool() != Tool::MagneticLasso {
        return false;
    }
    match ev {
        ToolEvent::Down { x, y, pressure } if x.is_finite() && y.is_finite() => down(app, [x, y], pressure, mods),
        ToolEvent::Move { x, y, pressure } if x.is_finite() && y.is_finite() => moved(app, [x, y], pressure),
        ToolEvent::Up { .. } => end_freehand(app),
        _ => {}
    }
    true
}

fn down(app: &mut PhotocraftApp, p: [f64; 2], pressure: f32, mods: Modifiers) {
    let s = settings(app, pressure);
    if !app.ui.magnetic.active() {
        // The first fastening point; the modifiers held now set the selection mode.
        let mode = crate::canvas::selection_mode(app, mods).to_owned();
        app.magnetic = Runtime::default();
        let start = snap(app, p, s);
        let doc = app.session.active().map_or(0, |d| d.doc.id.0);
        app.ui.magnetic = MagneticLasso { path: vec![start], anchors: vec![0], trail: vec![p], mode, doc, ..Default::default() };
        return;
    }
    if near_start(app, p) {
        // On the first point: the border closes there.
        let m = &app.ui.magnetic;
        let (Some(&first), Some(&last)) = (m.path.first(), m.path.last()) else { return };
        let closing = if mods.alt {
            vec![last, first]
        } else {
            let guide = app.ui.magnetic.trail.clone();
            trace(app, last, first, &guide, s)
        };
        let m = &mut app.ui.magnetic;
        m.path.extend(closing.into_iter().skip(1));
        m.live.clear();
        commit(app);
        return;
    }
    if mods.alt {
        // ⌥-click: a straight segment to here; dragging on draws freehand.
        let m = &mut app.ui.magnetic;
        if m.path.last().is_none_or(|q| dist(*q, p) > 1e-9) {
            m.path.push(p);
            m.anchors.push(m.path.len() - 1);
        }
        m.live.clear();
        m.trail = vec![p];
        m.freehand = true;
        return;
    }
    // A click fastens a point at the edge near the pointer.
    follow(app, p, s, false);
    fasten_live(app);
}

fn moved(app: &mut PhotocraftApp, p: [f64; 2], pressure: f32) {
    if !app.ui.magnetic.active() {
        return;
    }
    if app.ui.magnetic.freehand {
        // ⌥-drag: the pointer's path is the border.
        let m = &mut app.ui.magnetic;
        if m.path.last().is_none_or(|q| dist(*q, p) >= 0.5) {
            m.path.push(p);
        }
        m.trail = vec![p];
        return;
    }
    let s = settings(app, pressure);
    let step = (spacing(app) / 2.0).max(1.0);
    let last = app.ui.magnetic.trail.last().copied().unwrap_or(p);
    // A long jump (a fast flick, automation) is walked in steps, so points keep fastening along it.
    let n = ((dist(last, p) / step).ceil() as usize).clamp(1, 256);
    for k in 1..=n {
        let t = k as f64 / n as f64;
        let q = [last[0] + (p[0] - last[0]) * t, last[1] + (p[1] - last[1]) * t];
        if k < n || !app.defer_live_stroke {
            follow(app, q, s, true);
        } else {
            // Batched moves (one frame's worth): note the pointer, trace once in `flush`.
            extend_trail(app, q, s);
            app.magnetic.pending = true;
        }
    }
}

/// Trace to the pointer once after a batch of pointer moves.
pub fn flush(app: &mut PhotocraftApp) {
    if std::mem::take(&mut app.magnetic.pending)
        && let Some(&p) = app.ui.magnetic.trail.last()
    {
        let s = settings(app, app.stylus.pressure());
        follow(app, p, s, true);
    }
}

/// Add the pointer position `q` to the trail, keeping its points a few pixels apart.
fn extend_trail(app: &mut PhotocraftApp, q: [f64; 2], s: Settings) {
    let gap = (s.width / 4.0).clamp(1.0, 8.0);
    let t = &mut app.ui.magnetic.trail;
    match t.len() {
        n if n >= 2 && dist(t[n - 2], q) < gap => t[n - 1] = q,
        _ => t.push(q),
    }
}

/// The pointer is at `q`: trace the live segment to the edge near it and, with `fasten`, fasten
/// points behind the pointer while the live segment is long enough.
fn follow(app: &mut PhotocraftApp, q: [f64; 2], s: Settings, fasten: bool) {
    extend_trail(app, q, s);
    let target = snap(app, q, s);
    let every = spacing(app);
    for _ in 0..64 {
        let Some(&from) = app.ui.magnetic.path.last() else { return };
        let guide = app.ui.magnetic.trail.clone();
        let live = trace(app, from, target, &guide, s);
        // The part of the border `spacing` behind its start no longer changes as the pointer
        // moves: fasten it.
        match cut(&live, every).filter(|_| fasten) {
            Some((head, point)) => {
                let m = &mut app.ui.magnetic;
                m.path.extend(head.into_iter().skip(1));
                m.anchors.push(m.path.len() - 1);
                // Edges are looked for along the pointer's path from where it passed the new point.
                let near = m.trail.iter().enumerate().min_by(|a, b| dist(*a.1, point).total_cmp(&dist(*b.1, point))).map_or(0, |(i, _)| i);
                let rest: Vec<[f64; 2]> = m.trail.iter().skip(near + 1).copied().collect();
                m.trail = std::iter::once(point).chain(rest).collect();
            }
            None => {
                app.ui.magnetic.live = live;
                return;
            }
        }
    }
}

/// The polyline `p` up to `len` along it (ending at that point) and the point, when more than a
/// pixel of it lies beyond.
fn cut(p: &[[f64; 2]], len: f64) -> Option<(Vec<[f64; 2]>, [f64; 2])> {
    if magnetic::length(p) < len + 1.0 {
        return None;
    }
    let mut head = Vec::new();
    let mut acc = 0.0;
    for w in p.windows(2) {
        let (a, b) = (w[0], w[1]);
        head.push(a);
        let d = dist(a, b);
        if acc + d >= len && d > 0.0 {
            let t = (len - acc) / d;
            let point = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
            head.push(point);
            return Some((head, point));
        }
        acc += d;
    }
    None
}

/// Fasten the live segment: its end becomes a fastening point.
fn fasten_live(app: &mut PhotocraftApp) {
    let m = &mut app.ui.magnetic;
    let live = std::mem::take(&mut m.live);
    let before = m.path.len();
    for q in live.into_iter().skip(1) {
        if m.path.last().is_none_or(|l| dist(*l, q) > 1e-9) {
            m.path.push(q);
        }
    }
    if m.path.len() > before {
        m.anchors.push(m.path.len() - 1);
    }
    if let Some(&end) = m.path.last() {
        m.trail = vec![end];
    }
}

/// An ⌥-drag ends: where it stopped is a fastening point.
fn end_freehand(app: &mut PhotocraftApp) {
    let m = &mut app.ui.magnetic;
    if std::mem::take(&mut m.freehand) && m.anchors.last().is_some_and(|&a| a + 1 < m.path.len()) {
        m.anchors.push(m.path.len() - 1);
    }
}

/// `p` is over the first point of a border long enough to close.
fn near_start(app: &PhotocraftApp, p: [f64; 2]) -> bool {
    let m = &app.ui.magnetic;
    let r = close_radius(app);
    m.path.first().is_some_and(|&first| dist(first, p) <= r) && magnetic::length(&m.path) + magnetic::length(&m.live) > 2.0 * r
}

/// Close the border and select it: the live segment fastens, then the last point joins the first
/// along the edges, or with a straight segment with `straight`.
pub fn close(app: &mut PhotocraftApp, straight: bool) {
    if !app.ui.magnetic.active() {
        return;
    }
    end_freehand(app);
    fasten_live(app);
    let m = &app.ui.magnetic;
    let (Some(&first), Some(&last)) = (m.path.first(), m.path.last()) else { return };
    if !straight {
        let s = settings(app, 0.0);
        let mut closing = trace(app, last, first, &[], s);
        // The border ends where it starts.
        closing.pop();
        app.ui.magnetic.path.extend(closing.into_iter().skip(1));
    }
    commit(app);
}

/// Select the border as drawn (in the mode the first click set) and finish.
fn commit(app: &mut PhotocraftApp) {
    let m = std::mem::take(&mut app.ui.magnetic);
    app.magnetic = Runtime::default();
    if m.path.len() < 3 {
        return;
    }
    let o = &app.ui.tool_options;
    let mode = if m.mode.is_empty() { "replace".to_owned() } else { m.mode };
    let p = json!({"points": m.path, "trace": false, "mode": mode, "antiAlias": o.anti_alias, "feather": o.feather});
    let _ = app.run("select.magneticLasso", p);
}

/// Esc: drop the border.
pub fn cancel(app: &mut PhotocraftApp) {
    app.ui.magnetic = MagneticLasso::default();
    app.magnetic = Runtime::default();
}

/// ⌫ / Delete: remove the last fastening point and the border back to the one before it; with only
/// the first point left, cancel.
pub fn remove_last_point(app: &mut PhotocraftApp) {
    end_freehand(app);
    let m = &mut app.ui.magnetic;
    if m.anchors.len() <= 1 {
        cancel(app);
        return;
    }
    m.anchors.pop();
    let keep = m.anchors.last().map_or(1, |&a| a + 1);
    m.path.truncate(keep);
    m.live.clear();
    let pointer = m.trail.last().copied();
    m.trail = m.path.last().copied().into_iter().collect();
    // The live segment runs from the earlier point to the pointer again; it fastens only when the
    // pointer moves on, so ⌫ can keep removing points.
    if let Some(p) = pointer {
        let s = settings(app, app.stylus.pressure());
        follow(app, p, s, false);
    }
}

/// A new-selection border is being drawn, so the selection it will replace is hidden.
pub fn replaces_selection(app: &PhotocraftApp) -> bool {
    app.ui.magnetic.active() && app.ui.magnetic.mode == "replace"
}

/// Once a frame: a border left behind by a tool or document switch is dropped.
pub fn frame(app: &mut PhotocraftApp) {
    let m = &app.ui.magnetic;
    if m.active() && (app.ui.tool != Tool::MagneticLasso || app.session.active().map(|d| d.doc.id.0) != Some(m.doc)) {
        cancel(app);
    }
}

/// Keys while the Magnetic Lasso is the tool: ↩ closes the border along the edges (⌥↩ straight),
/// Esc cancels it, ⌫ / Delete remove the last fastening point, and [ / ] change the detection width
/// by 1 px. Returns whether a key was taken.
pub fn keys(app: &mut PhotocraftApp, ctx: &egui::Context) -> bool {
    if app.ui.tool != Tool::MagneticLasso {
        return false;
    }
    let pressed = |m: Modifiers, k: Key| ctx.input_mut(|i| i.consume_key(m, k));
    if app.ui.magnetic.active() {
        if pressed(Modifiers::ALT, Key::Enter) {
            close(app, true);
            return true;
        }
        if pressed(Modifiers::NONE, Key::Enter) {
            close(app, false);
            return true;
        }
        if pressed(Modifiers::NONE, Key::Escape) {
            cancel(app);
            return true;
        }
        if pressed(Modifiers::NONE, Key::Backspace) || pressed(Modifiers::NONE, Key::Delete) {
            remove_last_point(app);
            return true;
        }
    }
    let step = if pressed(Modifiers::NONE, Key::OpenBracket) {
        -1.0
    } else if pressed(Modifiers::NONE, Key::CloseBracket) {
        1.0
    } else {
        return false;
    };
    let w = &mut app.ui.tool_options.magnetic_width;
    *w = if w.is_finite() { (w.round() + step).clamp(1.0, 256.0) } else { 10.0 };
    true
}

/// The border being drawn, its fastening points, and the closing mark when the pointer is over the
/// first point.
pub fn draw(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform, hover: Option<Pos2>) {
    let m = &app.ui.magnetic;
    if !m.active() {
        return;
    }
    let screen = |q: &[f64; 2]| xf.to_screen(q[0] as f32, q[1] as f32);
    let pts: Vec<Pos2> = m.path.iter().chain(m.live.iter().skip(1)).map(screen).collect();
    crate::tool_feedback::draw_ants(painter, &pts, false);
    for p in m.anchors.iter().filter_map(|&i| m.path.get(i)).map(screen) {
        let r = Rect::from_center_size(p, vec2(5.0, 5.0));
        painter.rect_filled(r, 0.0, Color32::WHITE);
        painter.rect_stroke(r, 0.0, Stroke::new(1.0, Color32::BLACK), egui::StrokeKind::Outside);
    }
    if let Some(h) = hover
        && near_start(app, xf.to_doc(h))
    {
        // Photoshop's closing cursor: a small circle beside the pointer.
        let c = h + vec2(12.0, 12.0);
        painter.circle_stroke(c, 4.0, Stroke::new(3.0, Color32::from_black_alpha(200)));
        painter.circle_stroke(c, 4.0, Stroke::new(1.2, Color32::WHITE));
    }
}

/// The cursor over the canvas: a crosshair; Preferences › Cursors › Other Cursors › Precise also
/// shows the detection width around it.
pub fn cursor(app: &PhotocraftApp, painter: &egui::Painter, p: Pos2, zoom: f32) -> egui::CursorIcon {
    if app.session.prefs().cursors.other == photocraft_engine::prefs::OtherCursor::Precise {
        let r = (app.ui.tool_options.magnetic_width.clamp(1.0, 256.0) * zoom).max(2.0);
        painter.circle_stroke(p, r + 0.5, Stroke::new(1.0, Color32::from_black_alpha(140)));
        painter.circle_stroke(p, r, Stroke::new(1.0, Color32::from_white_alpha(220)));
    }
    egui::CursorIcon::Crosshair
}

#[cfg(test)]
mod tests;
