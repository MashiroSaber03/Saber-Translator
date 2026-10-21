//! Content-Aware Move › Transform On Drop (the options-bar checkbox, as in Photoshop): after the
//! drop a transform box sits on the moved patch. Its handles are the Free Transform box's
//! ([`crate::transform_tool::hit`], [`crate::transform_tool::apply_drag`]), limited to moving,
//! scaling and rotating; ↩, ✓ or a double-click computes the blend with the transformed patch
//! (`paint.contentAwareMove` with `scaleX`, `scaleY` and `rotate`), Esc or ⊘ drops the whole move.
//! The document is untouched until then, so the move stays one History step.

use egui::{CursorIcon, Modifiers, Pos2};
use photocraft_algo::transform::Homography;

use crate::PhotocraftApp;
use crate::canvas::{ToolEvent, ViewXform};
use crate::state::{CamTransform, Tool, TransformSession};
use crate::transform_tool::{Gesture, Hit};

/// Where the patch lands: the drag offset (px) and the scale (%, per axis) and clockwise rotation
/// (degrees) about its centre, as `paint.contentAwareMove` takes them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Placement {
    pub offset: [i64; 2],
    pub scale: [f64; 2],
    pub rotate: f64,
}

impl Placement {
    /// The patch is only moved (or not even that).
    pub fn is_plain_move(&self) -> bool {
        self.scale == [100.0, 100.0] && self.rotate == 0.0
    }
}

pub(crate) fn active(app: &PhotocraftApp) -> bool {
    app.ui.cam_transform.is_some()
}

/// Put the box on the selection dropped `offset` away. False when there is no selection to box.
pub(crate) fn begin(app: &mut PhotocraftApp, offset: [i32; 2]) -> bool {
    let Some((document, revision, canvas, sel)) =
        app.session.active().and_then(|st| Some((st.doc.id.0, st.revision, st.doc.bounds(), st.doc.selection.clone()?)))
    else {
        return false;
    };
    let b = app.cached_bounds(u64::MAX - document, &sel).intersect(&canvas);
    if b.is_empty() {
        return false;
    }
    let r = [b.x0.saturating_add(offset[0]), b.y0.saturating_add(offset[1]), b.x1.saturating_add(offset[0]), b.y1.saturating_add(offset[1])].map(f64::from);
    let frame = TransformSession {
        session: app.ui.alloc_id(),
        layer: 0,
        rect: r,
        quad: [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]],
        pivot: [(r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0],
        interpolation: "bilinear".into(),
        warp: None,
        selection: false,
        target: None,
        path: None,
        made: None,
        mode: Default::default(),
        content_aware: None,
        skew_v: 0.0,
    };
    app.ui.cam_transform = Some(CamTransform { document, revision, offset, frame, gesture: None });
    app.ui.status = crate::i18n::fmt(tl!("Scale or rotate the moved area, then press {key}"), &[("key", &crate::shortcuts::pretty("Enter"))]);
    app.ui.status_error = false;
    true
}

/// The box as `paint.contentAwareMove`'s params: its rect → quad map split into a scale along the
/// box's own axes, a rotation and the move of its centre. `None` for a collapsed or absurd box.
pub(crate) fn placement(t: &CamTransform) -> Option<Placement> {
    let [x0, y0, x1, y1] = t.frame.rect;
    let (w, h) = (x1 - x0, y1 - y0);
    if !(w > 0.0 && h > 0.0) {
        return None;
    }
    let q = t.frame.quad;
    // The box's x axis (scaled by sx, turned) and y axis (scaled by sy, a negative sy flips).
    let u = [(q[1][0] - q[0][0]) / w, (q[1][1] - q[0][1]) / w];
    let v = [(q[3][0] - q[0][0]) / h, (q[3][1] - q[0][1]) / h];
    let angle = u[1].atan2(u[0]);
    let (sn, cs) = angle.sin_cos();
    let (sx, sy) = (u[0].hypot(u[1]), v[1] * cs - v[0] * sn);
    // Scale and rotation are about the selection's centre, then the offset moves it.
    let centre = [(q[0][0] + q[2][0]) / 2.0, (q[0][1] + q[2][1]) / 2.0];
    let origin = [(x0 + x1) / 2.0 - f64::from(t.offset[0]), (y0 + y1) / 2.0 - f64::from(t.offset[1])];
    let off = [(centre[0] - origin[0]).round(), (centre[1] - origin[1]).round()];
    let round = |v: f64| {
        let r = (v * 1e4).round() / 1e4;
        if r == 0.0 { 0.0 } else { r }
    };
    let (sx, sy, rotate) = (round(sx * 100.0), round(sy * 100.0), round(angle.to_degrees()));
    let sane = |s: f64| (1.0..=10_000.0).contains(&s.abs());
    let ok = rotate.is_finite() && sane(sx) && sane(sy) && off.iter().all(|v| v.is_finite() && v.abs() < 1e7);
    ok.then_some(Placement { offset: [off[0] as i64, off[1] as i64], scale: [sx, sy], rotate })
}

pub(crate) fn cancel(app: &mut PhotocraftApp) {
    if app.ui.cam_transform.take().is_some() && !app.ui.status_error {
        // The box's hint is done with.
        app.ui.status.clear();
    }
}

/// Drop the box when what it was made for changed: another tool, document or edit.
pub(crate) fn cancel_stale(app: &mut PhotocraftApp) {
    let Some(t) = &app.ui.cam_transform else { return };
    let valid = app.ui.tool == Tool::ContentAwareMove
        && app.session.active().is_some_and(|st| st.doc.id.0 == t.document && st.revision == t.revision && st.doc.selection.is_some());
    if !valid {
        cancel(app);
    }
}

/// `paint.contentAwareMove`'s params for the box's placement and the options bar.
pub(crate) fn params(app: &PhotocraftApp, place: &Placement) -> serde_json::Value {
    let mut p = crate::retouch_ui::content_aware_move_params(app, place.offset);
    if !place.is_plain_move() {
        p["scaleX"] = place.scale[0].into();
        p["scaleY"] = place.scale[1].into();
        p["rotate"] = place.rotate.into();
    }
    p
}

/// ↩ / ✓: the move with the transformed patch, as one background job and History step.
pub(crate) fn commit(app: &mut PhotocraftApp) {
    cancel_stale(app);
    let Some(t) = app.ui.cam_transform.clone() else { return };
    cancel(app);
    let Some(place) = placement(&t) else { return };
    if place.offset == [0, 0] && place.is_plain_move() {
        crate::retouch_ui::patch_hint(app);
        return;
    }
    let p = params(app, &place);
    if let Err(e) = app.run("paint.contentAwareMove", p) {
        app.ui.status = e;
        app.ui.status_error = true;
    }
}

/// Pointer input while the box is up: every press goes to it (the canvas is otherwise inert, as
/// during Free Transform). Returns false when there is no box.
pub(crate) fn pointer(app: &mut PhotocraftApp, ev: ToolEvent, mods: Modifiers) -> bool {
    cancel_stale(app);
    let tol = crate::transform_tool::handle_tolerance(app);
    let legacy = app.session.prefs().general.use_legacy_free_transform;
    let Some(t) = app.ui.cam_transform.as_mut() else { return false };
    match ev {
        ToolEvent::Down { x, y, .. } => {
            let p = [x, y];
            if !p.iter().all(|v| v.is_finite()) {
                return true;
            }
            let mut h = crate::transform_tool::hit(&t.frame, p, tol);
            // ⌥-click away from the handles moves the reference point there (and drags it).
            if mods.alt && matches!(h, Hit::Inside | Hit::Outside) {
                t.frame.pivot = p;
                h = Hit::Pivot;
            }
            t.gesture = Some(Gesture { hit: h, start: p, quad0: t.frame.quad, pivot0: t.frame.pivot });
        }
        ToolEvent::Move { x, y, .. } | ToolEvent::Up { x, y } => {
            if let Some(g) = t.gesture
                && [x, y].iter().all(|v| v.is_finite() && v.abs() <= 1e7)
            {
                let before = (t.frame.quad, t.frame.pivot);
                // Move, scale and rotate only: ⌘ would skew or distort, which the blend can't take.
                let plain = Modifiers { command: false, ctrl: false, mac_cmd: false, ..mods };
                crate::transform_tool::apply_drag(&mut t.frame, g, [x, y], crate::transform_tool::legacy_mods(legacy, g.hit, plain));
                if placement(t).is_none() {
                    (t.frame.quad, t.frame.pivot) = before;
                }
            }
            if matches!(ev, ToolEvent::Up { .. }) {
                t.gesture = None;
            }
        }
    }
    true
}

/// The selection's outline where the box puts it, and the box.
pub(crate) fn draw(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    let Some(t) = &app.ui.cam_transform else { return };
    if let Some((doc, _, segs)) = &app.outline_cache
        && doc.0 == t.document
        && let Some(h) = Homography::rect_to_quad(t.frame.rect, t.frame.quad)
    {
        let [ox, oy] = t.offset.map(f64::from);
        let time = painter.ctx().input(|i| i.time);
        crate::canvas::marching_ants_mapped(painter, segs, time, |p| {
            let (x, y) = h.apply(f64::from(p[0]) + ox, f64::from(p[1]) + oy);
            xf.to_screen(x as f32, y as f32)
        });
        painter.ctx().request_repaint_after(std::time::Duration::from_millis(100));
    }
    crate::type_transform::draw_box(painter, xf, &t.frame);
}

/// The Free Transform box's cursors over the box's handles.
pub(crate) fn cursor(app: &PhotocraftApp, ctx: &egui::Context, xf: &ViewXform, p: Pos2) -> Option<CursorIcon> {
    let t = app.ui.cam_transform.as_ref().filter(|_| app.ui.tool == Tool::ContentAwareMove)?;
    let q = t.frame.quad;
    let h = t.gesture.map_or_else(|| crate::transform_tool::hit(&t.frame, xf.to_doc(p), crate::transform_tool::handle_tolerance(app)), |g| g.hit);
    let mid = |i: usize| -> Option<[f64; 2]> {
        let (a, b) = (q.get(i % 4)?, q.get((i + 1) % 4)?);
        Some([(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0])
    };
    Some(match h {
        Hit::Corner(i) => {
            let (a, b) = (q.get(i)?, q.get((i + 2) % 4)?);
            crate::type_transform::resize_cursor([a[0] - b[0], a[1] - b[1]], xf.flip)
        }
        Hit::Edge(i) => {
            let (a, b) = (mid(i)?, mid(i + 2)?);
            crate::type_transform::resize_cursor([a[0] - b[0], a[1] - b[1]], xf.flip)
        }
        Hit::Outside => {
            crate::type_transform::rotation_cursor(ctx, p);
            CursorIcon::None
        }
        Hit::Pivot => CursorIcon::Crosshair,
        Hit::Inside => CursorIcon::Move,
    })
}
