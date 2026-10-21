//! Ctrl/Cmd while typing: an oriented frame and a temporary affine drag, without ending typing.
//! The existing Free Transform geometry is reused with Type's modifier rules (Ctrl exposes the
//! frame; it must never distort a corner). Only release dispatches `type.edit`.

use std::collections::VecDeque;
use std::sync::Arc;

use egui::{CursorIcon, Modifiers, Pos2, Stroke, vec2};
use photocraft_doc::{DocId, Document, LayerContent, LayerId, text::TextShape};
use photocraft_geom::{Affine, Point, Rect};
use serde_json::json;

use crate::PhotocraftApp;
use crate::canvas::{ToolEvent, ViewXform};
use crate::state::{TransformSession, TypeTransform};
use crate::transform_tool::{Gesture, Hit};

const PREVIEW_BASE: u64 = 1 << 37;
const ROTATE_PX: f64 = 28.0;

/// Cached rendered preview and bounded damage history. Unchanged pointer positions render nothing.
pub(crate) struct Preview {
    document: DocId,
    revision: u64,
    transform: Affine,
    shown: Option<Arc<Document>>,
    key: u64,
    original_area: Rect,
    areas: VecDeque<(u64, Rect)>,
}

pub(crate) fn active(app: &PhotocraftApp) -> bool {
    app.ui.type_transform.is_some()
}

pub(crate) fn visible(app: &PhotocraftApp, mods: Modifiers) -> bool {
    app.ui.tool.is_type() && app.ui.text_edit.is_some() && (mods.command || active(app))
}

/// Drop pointer capture when its source changed externally. Never apply to another document.
pub(crate) fn cancel_stale(app: &mut PhotocraftApp) {
    let Some(t) = &app.ui.type_transform else { return };
    let valid = app.ui.tool.is_type()
        && app.ui.text_edit.as_ref().is_some_and(|ed| ed.layer == t.frame.layer)
        && app.session.active().is_some_and(|st| {
            st.doc.id.0 == t.document
                && st.revision == t.revision
                && st.doc.layer(LayerId(t.frame.layer)).is_some_and(|l| !l.locks.all && !l.locks.position && matches!(l.content, LayerContent::Text(_)))
        });
    if !valid {
        let lost_layer = app.session.active().is_none_or(|st| st.doc.id.0 != t.document || st.doc.layer(LayerId(t.frame.layer)).is_none());
        cancel_drag(app);
        if lost_layer {
            app.ui.text_edit = None;
        }
    }
}

pub(crate) fn cancel_drag(app: &mut PhotocraftApp) {
    app.ui.type_transform = None;
    if let Some(p) = app.type_transform_preview.as_mut() {
        // A closed last document has no next canvas frame to release its preview's tiles.
        p.shown = None;
    }
    if let Some(ed) = app.ui.text_edit.as_mut() {
        ed.dragging = false;
        ed.resize = None;
    }
}

pub(crate) fn reset(app: &mut PhotocraftApp) {
    cancel_drag(app);
    app.ui.type_transform_pivot = None;
}

/// Logical text-space bounds, mapped by the layer's existing affine (including rotation/skew).
pub(crate) fn frame(app: &mut PhotocraftApp) -> Option<TransformSession> {
    if let Some(t) = &app.ui.type_transform {
        return Some(t.frame.clone());
    }
    let id = LayerId(app.ui.text_edit.as_ref()?.layer);
    let st = app.session.active()?;
    let layer = st.doc.layer(id)?;
    if layer.locks.all || layer.locks.position || !layer.visible {
        return None;
    }
    let LayerContent::Text(text) = &layer.content else { return None };
    let shape = text.shape.clone();
    let size = text.size_pt;
    let (layout, aff, _) = crate::type_tool::layout(app, id)?;
    if !aff.m.iter().all(|v| v.is_finite()) || aff.inverse().is_none() {
        return None;
    }
    let rect = match shape {
        TextShape::Box { x, y, width, height } => [x, y, x + width, y + height].map(f64::from),
        _ => {
            let mut r = layout.bounds().unwrap_or([0.0, -size * layout.px_per_pt, 1.0, 0.0]).map(f64::from);
            // Empty lines still have a caret and can be moved/scaled, without inventing glyphs.
            r[2] = r[2].max(r[0] + 1.0);
            r[3] = r[3].max(r[1] + 1.0);
            r
        }
    };
    if !rect.iter().all(|v| v.is_finite()) || rect[2] <= rect[0] || rect[3] <= rect[1] {
        return None;
    }
    let quad = [[rect[0], rect[1]], [rect[2], rect[1]], [rect[2], rect[3]], [rect[0], rect[3]]].map(|q| {
        let p = aff.apply(Point::new(q[0], q[1]));
        [p.x, p.y]
    });
    if !quad.iter().flatten().all(|v| v.is_finite()) {
        return None;
    }
    let pivot = app.ui.type_transform_pivot.unwrap_or([(quad[0][0] + quad[2][0]) / 2.0, (quad[0][1] + quad[2][1]) / 2.0]);
    Some(TransformSession {
        session: 0,
        layer: id.0,
        rect,
        quad,
        pivot,
        interpolation: "bicubic".into(),
        warp: None,
        selection: false,
        target: None,
        path: None,
        made: None,
        mode: Default::default(),
        content_aware: None,
        skew_v: 0.0,
    })
}

fn near_border(t: &TransformSession, p: [f64; 2], tol: f64) -> bool {
    t.quad.iter().zip(t.quad.iter().cycle().skip(1)).take(4).any(|(a, b)| {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = dx * dx + dy * dy;
        let f = if len > 0.0 { ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len } else { 0.0 }.clamp(0.0, 1.0);
        (p[0] - a[0] - f * dx).hypot(p[1] - a[1] - f * dy) <= tol
    })
}

fn hit(app: &PhotocraftApp, t: &TransformSession, p: [f64; 2]) -> Option<Hit> {
    let h = crate::transform_tool::hit(t, p, crate::transform_tool::handle_tolerance(app));
    (h != Hit::Outside || near_border(t, p, ROTATE_PX / f64::from(app.point_zoom().max(0.01)))).then_some(h)
}

/// Type uses legacy scaling: Shift constrains; Ctrl/Cmd is the mode key, never corner Distort.
fn gesture_mods(h: Hit, mut mods: Modifiers) -> Modifiers {
    mods.command = matches!(h, Hit::Edge(_));
    mods.ctrl = false;
    mods.mac_cmd = false;
    if matches!(h, Hit::Corner(_)) {
        mods.shift = !mods.shift;
    }
    mods
}

fn affine(t: &TypeTransform) -> Option<Affine> {
    if t.frame.quad == t.gesture.quad0 {
        return Some(t.original);
    }
    let [x0, y0, x1, y1] = t.frame.rect;
    let [a, b, _, d] = t.frame.quad;
    let (w, h) = (x1 - x0, y1 - y0);
    if w <= 0.0 || h <= 0.0 {
        return None;
    }
    let (ax, ay, bx, by) = ((b[0] - a[0]) / w, (b[1] - a[1]) / w, (d[0] - a[0]) / h, (d[1] - a[1]) / h);
    let out = Affine { m: [ax, ay, bx, by, a[0] - ax * x0 - bx * y0, a[1] - ay * x0 - by * y0] };
    let xs = t.frame.quad.map(|p| p[0]);
    let ys = t.frame.quad.map(|p| p[1]);
    let width = xs.into_iter().fold(f64::MIN, f64::max) - xs.into_iter().fold(f64::MAX, f64::min);
    let height = ys.into_iter().fold(f64::MIN, f64::max) - ys.into_iter().fold(f64::MAX, f64::min);
    // Reject collapsed/nonfinite geometry and bound input-sized text raster allocations.
    (out.m.iter().all(|v| v.is_finite())
        && out.determinant().abs() >= 1e-8
        && t.frame.quad.iter().flatten().all(|v| v.is_finite() && v.abs() <= 1e7)
        && width >= 0.05
        && height >= 0.05
        && width * height <= 64.0 * 1024.0 * 1024.0)
        .then_some(out)
}

pub(crate) fn current_transform(app: &PhotocraftApp, id: LayerId) -> Option<Affine> {
    affine(app.ui.type_transform.as_ref().filter(|t| t.frame.layer == id.0)?)
}

fn update(app: &mut PhotocraftApp, p: [f64; 2], mods: Modifiers) {
    if !p.iter().all(|v| v.is_finite() && v.abs() <= 1e7) {
        return;
    }
    let Some(t) = app.ui.type_transform.as_mut() else { return };
    let before = t.frame.clone();
    if p == t.gesture.start {
        t.frame.quad = t.gesture.quad0;
        t.frame.pivot = t.gesture.pivot0;
    } else {
        // A handle grabbed off-centre follows the pointer delta, without a jump to its hotspot.
        let target = match t.gesture.hit {
            Hit::Corner(i) => t.gesture.quad0.get(i).copied(),
            Hit::Pivot => Some(t.gesture.pivot0),
            _ => None,
        };
        let p = target.map_or(p, |q| [q[0] + p[0] - t.gesture.start[0], q[1] + p[1] - t.gesture.start[1]]);
        crate::transform_tool::apply_drag(&mut t.frame, t.gesture, p, gesture_mods(t.gesture.hit, mods));
    }
    if affine(t).is_none() {
        t.frame = before;
    }
}

pub(crate) fn finish(app: &mut PhotocraftApp) {
    cancel_stale(app);
    let Some(t) = app.ui.type_transform.take() else { return };
    cancel_drag(app);
    app.ui.type_transform_pivot = Some(t.frame.pivot);
    let Some(transform) = affine(&t) else { return };
    // Pivot-only and out-and-back gestures leave no history entry, even in an untouched session.
    if transform.m.iter().zip(t.original.m).all(|(a, b)| (a - b).abs() < 1e-9) {
        return;
    }
    let Some(ed) = app.ui.text_edit.as_ref() else { return };
    let params = json!({"layer": ed.layer, "transform": transform.m, "coalesce": ed.session});
    if app.run("type.edit", params).is_ok()
        && let Some(preview) = app.type_transform_preview.take().filter(|p| p.transform == transform)
    {
        crate::canvas::shown_as_document(app, preview.document, |key| key == preview.key);
    }
}

/// First refusal before snapping, caret selection, paragraph resizing or clicking out to commit.
pub(crate) fn pointer(app: &mut PhotocraftApp, ev: ToolEvent, mods: Modifiers) -> bool {
    cancel_stale(app);
    if !app.ui.tool.is_type() || app.ui.text_edit.is_none() {
        return false;
    }
    match ev {
        ToolEvent::Down { .. } if active(app) => true,
        ToolEvent::Down { x, y, .. } if mods.command => {
            let p = [x, y];
            if !p.iter().all(|v| v.is_finite()) {
                return true;
            }
            let Some(mut frame) = frame(app) else { return true };
            let Some(h) = hit(app, &frame, p) else { return true };
            let Some((_, original, _)) = crate::type_tool::layout(app, LayerId(frame.layer)) else { return true };
            let Some(st) = app.session.active() else { return true };
            let (document, revision) = (st.doc.id.0, st.revision);
            frame.session = app.ui.alloc_id();
            let gesture = Gesture { hit: h, start: p, quad0: frame.quad, pivot0: frame.pivot };
            app.ui.type_transform = Some(TypeTransform { document, revision, original, frame, gesture });
            if let Some(ed) = app.ui.text_edit.as_mut() {
                ed.dragging = false;
                ed.resize = None;
            }
            true
        }
        ToolEvent::Move { x, y, .. } | ToolEvent::Up { x, y } if active(app) => {
            update(app, [x, y], mods);
            if matches!(ev, ToolEvent::Up { .. }) {
                finish(app);
            }
            true
        }
        _ => false,
    }
}

/// Preview via the existing text renderer; all glyphs/styles/effects remain native document data.
pub(crate) fn display_doc(app: &mut PhotocraftApp, idx: usize) -> Option<(Arc<Document>, u64)> {
    cancel_stale(app);
    if !active(app) {
        if let Some(p) = app.type_transform_preview.as_mut() {
            // Retain damage bookkeeping for the return to the document, not its rendered tiles.
            p.shown = None;
        }
        return None;
    }
    let t = app.ui.type_transform.as_ref()?;
    let st = app.session.documents().get(idx)?;
    if st.doc.id.0 != t.document || st.revision != t.revision {
        return None;
    }
    let transform = affine(t)?;
    if transform == t.original {
        return None;
    }
    let (id, revision, source, layer) = (st.doc.id, st.revision, st.doc.clone(), LayerId(t.frame.layer));
    if let Some(p) = &app.type_transform_preview
        && p.document == id
        && p.revision == revision
        && p.transform == transform
        && let Some(shown) = &p.shown
    {
        return Some((shown.clone(), p.key));
    }
    let original_area = source.layer(layer)?.surface().map_or(Rect::EMPTY, |s| s.tile_bounds());
    let mut shown = (*source).clone();
    let LayerContent::Text(text) = &mut shown.layer_mut(layer)?.content else { return None };
    text.transform = transform;
    let started = crate::gpu_canvas::now_ms();
    photocraft_engine::type_cmds::refresh(&source, text);
    let area = text.cache.as_ref().map_or(Rect::EMPTY, |s| s.tile_bounds());
    let shown = Arc::new(photocraft_engine::mode_cmds::display_document(&shown).unwrap_or(shown));
    let key = PREVIEW_BASE | (app.ui.alloc_id() & (PREVIEW_BASE - 1));
    let mut areas = app.type_transform_preview.take().filter(|p| p.document == id && p.revision == revision).map(|p| p.areas).unwrap_or_default();
    if areas.len() >= 128 {
        areas.pop_front();
    }
    areas.push_back((key, area));
    app.type_transform_preview = Some(Preview { document: id, revision, transform, shown: Some(shown.clone()), key, original_area, areas });
    app.perf.span("type transform preview", crate::gpu_canvas::now_ms() - started);
    Some((shown, key))
}

pub(crate) fn damage(app: &PhotocraftApp, doc: DocId, revision: u64, before: u64, after: u64) -> Option<Rect> {
    let p = app.type_transform_preview.as_ref().filter(|p| p.document == doc && p.revision == revision)?;
    let area = |key| if key == 0 { Some(p.original_area) } else { p.areas.iter().find(|(k, _)| *k == key).map(|(_, r)| *r) };
    Some(area(before)?.union(&area(after)?))
}

pub(crate) fn draw(app: &mut PhotocraftApp, painter: &egui::Painter, xf: &ViewXform, mods: Modifiers) -> bool {
    if !visible(app, mods) {
        return false;
    }
    let Some(t) = frame(app) else { return false };
    draw_box(painter, xf, &t);
    true
}

/// A transform box: its outline, corner and edge handles and reference point. Shared with
/// Content-Aware Move's Transform On Drop.
pub(crate) fn draw_box(painter: &egui::Painter, xf: &ViewXform, t: &TransformSession) {
    let tokens = crate::theme::Tokens::get(painter.ctx());
    let pts = t.quad.map(|p| xf.to_screen(p[0] as f32, p[1] as f32));
    painter.add(egui::Shape::closed_line(pts.to_vec(), Stroke::new(3.0, tokens.shadow)));
    painter.add(egui::Shape::closed_line(pts.to_vec(), Stroke::new(1.0, tokens.accent)));
    let mids = [pts[0].lerp(pts[1], 0.5), pts[1].lerp(pts[2], 0.5), pts[2].lerp(pts[3], 0.5), pts[3].lerp(pts[0], 0.5)];
    for p in pts.iter().chain(mids.iter()) {
        let r = egui::Rect::from_center_size(*p, vec2(7.0, 7.0));
        painter.rect_filled(r, 0.0, tokens.accent_text);
        painter.rect_stroke(r, 0.0, Stroke::new(1.0, tokens.accent), egui::StrokeKind::Inside);
    }
    let c = xf.to_screen(t.pivot[0] as f32, t.pivot[1] as f32);
    painter.circle_stroke(c, 4.0, Stroke::new(2.0, tokens.shadow));
    painter.circle_stroke(c, 4.0, Stroke::new(1.0, tokens.accent_text));
    for axis in [vec2(6.0, 0.0), vec2(0.0, 6.0)] {
        painter.line_segment([c - axis, c + axis], Stroke::new(1.0, tokens.accent));
    }
}

pub(crate) fn resize_cursor(d: [f64; 2], flip: bool) -> CursorIcon {
    let angle = d[1].atan2(if flip { -d[0] } else { d[0] }).rem_euclid(std::f64::consts::PI);
    match (angle / std::f64::consts::FRAC_PI_4).round() as u8 % 4 {
        0 => CursorIcon::ResizeHorizontal,
        1 => CursorIcon::ResizeNwSe,
        2 => CursorIcon::ResizeVertical,
        _ => CursorIcon::ResizeNeSw,
    }
}

pub(crate) fn rotation_cursor(ctx: &egui::Context, p: Pos2) {
    let tokens = crate::theme::Tokens::get(ctx);
    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("type-rotate-cursor")));
    let first = -std::f32::consts::FRAC_PI_4;
    let last = std::f32::consts::PI + std::f32::consts::FRAC_PI_4;
    let points: Vec<Pos2> = (0..=24)
        .map(|i| {
            let angle = first + (last - first) * i as f32 / 24.0;
            p + vec2(angle.cos(), angle.sin()) * 7.0
        })
        .collect();
    painter.add(egui::Shape::line(points.clone(), Stroke::new(3.0, tokens.shadow)));
    painter.add(egui::Shape::line(points, Stroke::new(1.25, tokens.accent_text)));
    for (angle, sign) in [(first, -1.0), (last, 1.0)] {
        let tip = p + vec2(angle.cos(), angle.sin()) * 7.0;
        let tangent = vec2(-angle.sin(), angle.cos()) * sign;
        let normal = vec2(angle.cos(), angle.sin());
        painter.add(egui::Shape::convex_polygon(
            vec![tip + tangent * 3.0, tip - tangent * 3.0 + normal * 3.0, tip - tangent * 3.0 - normal * 3.0],
            tokens.accent_text,
            Stroke::new(1.0, tokens.shadow),
        ));
    }
}

/// The cursor and press share the exact same hit test, including a bounded outside rotation ring.
pub(crate) fn cursor(app: &mut PhotocraftApp, ctx: &egui::Context, xf: &ViewXform, p: Pos2, mods: Modifiers) -> Option<CursorIcon> {
    if !visible(app, mods) {
        let id = LayerId(app.ui.text_edit.as_ref()?.layer);
        let q = xf.to_doc(p);
        let handle = crate::type_tool::box_handle_at(app, id, q[0], q[1])?;
        let t = frame(app)?;
        let direction = if handle < 4 {
            let a = *t.quad.get(usize::from(handle))?;
            let b = *t.quad.get((usize::from(handle) + 2) % 4)?;
            [a[0] - b[0], a[1] - b[1]]
        } else {
            let b = if handle.is_multiple_of(2) { t.quad[3] } else { t.quad[1] };
            [b[0] - t.quad[0][0], b[1] - t.quad[0][1]]
        };
        return Some(resize_cursor(direction, xf.flip));
    }
    let Some(t) = frame(app) else { return Some(CursorIcon::Default) };
    let h = app.ui.type_transform.as_ref().map(|t| t.gesture.hit).or_else(|| hit(app, &t, xf.to_doc(p)));
    Some(match h {
        Some(Hit::Corner(i)) => {
            let q = *t.quad.get(i)?;
            let opposite = *t.quad.get((i + 2) % 4)?;
            resize_cursor([q[0] - opposite[0], q[1] - opposite[1]], xf.flip)
        }
        Some(Hit::Edge(_)) => {
            // Original arrowhead indicator, above the canvas, without a proprietary cursor asset.
            let tokens = crate::theme::Tokens::get(ctx);
            let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("type-skew-cursor")));
            painter.add(egui::Shape::convex_polygon(vec![p, p + vec2(14.0, 6.0), p + vec2(6.0, 14.0)], tokens.accent_text, Stroke::new(1.0, tokens.shadow)));
            CursorIcon::None
        }
        Some(Hit::Outside) => {
            rotation_cursor(ctx, p);
            CursorIcon::None
        }
        Some(Hit::Pivot) => CursorIcon::Crosshair,
        Some(Hit::Inside) => CursorIcon::Move,
        None => CursorIcon::Default,
    })
}
