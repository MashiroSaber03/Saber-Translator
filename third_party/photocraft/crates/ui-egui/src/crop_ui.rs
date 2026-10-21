//! Crop tool gestures. In Photoshop's "Classic Mode" (Use Classic Mode, P) the frame moves and
//! turns over a fixed image, as described below; in its default mode (`crop_mode`) the box stays
//! upright on screen and the same gestures inside and outside it move and turn the image behind
//! it instead (the view turns with the frame, so hit-testing in document space is unchanged):
//!
//! - with no frame, or inside the untouched default frame, a drag draws a new frame; ⇧ makes it
//!   square, ⌥ draws it from the centre, and holding Space while drawing repositions the frame
//!   being drawn;
//! - inside the frame, a drag moves it;
//! - on an edge or corner, a drag resizes it along the frame's own (possibly turned) axes; ⇧ keeps
//!   the frame's aspect ratio, ⌥ resizes about the centre, and an options-bar ratio preset always
//!   holds;
//! - outside the frame (off its handles) the pointer is the rotate cursor and a drag turns the
//!   frame about its centre, ⇧ in 15° steps, with the angle beside the pointer (#1792). As in
//!   Photoshop, a new frame is drawn after the pending one is committed or cancelled;
//! - while a frame is drawn or resized, its W × H (in px, as ↵ would crop it) shows beside the
//!   pointer, like the marquee's readout ([`sizing`]);
//! - with a pending frame and no gesture, the arrow keys nudge it 1 document px (⇧ 10 px) instead
//!   of the selection or layer, and X swaps it between portrait and landscape about its centre
//!   (keeping its angle, and swapping an options-bar ratio with it) instead of the colours
//!   ([`keys`], #1919). In the default mode the arrows move the image under the box instead.
//!
//! ↵ commits (`image.crop`, see `canvas::commit_crop`, with the turn as its `angle`: the document
//! is rotated so the frame is upright, then cropped to it) and Esc cancels. The pending frame lives
//! in `UiState::crop_rect` (the frame before its turn) and `UiState::crop_angle` (degrees,
//! clockwise on screen), so the control channel reads both. Both modes commit the same crop.
//!
//! As in Photoshop, from the first press until the crop is committed or cancelled the canvas also
//! shows the layers' pixels past its edges (kept by a crop with Delete Cropped Pixels off, or moved
//! out), with transparency out to the frame, under the shield ([`shows_beyond_canvas`]).

use egui::{CursorIcon, Modifiers};

use crate::PhotocraftApp;
use crate::canvas::ToolEvent;
use crate::state::Tool;

/// How far (screen points) from an edge the pointer still grabs it.
const HANDLE_PX: f64 = 8.0;
/// Smallest frame (document px) a gesture may leave behind.
const MIN_FRAME: f64 = 1.0;

/// Crop tool pointer state (not serialized: it only exists during a gesture).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CropState {
    pub drag: Option<CropDrag>,
    /// Space is held: drawing a new frame moves it instead of sizing it.
    pub space: bool,
    /// `UiState::crop_rect` is the untouched default frame ([`ensure_frame`]): a drag inside it
    /// draws a new frame (as in Photoshop) instead of moving it, and committing it does nothing.
    pub default_frame: bool,
    /// The document (index and size) the default frame was made for.
    frame_for: Option<(usize, u32, u32)>,
    /// The frame is being edited (pressed since it was made): the canvas shows what lies past it.
    pub editing: bool,
    /// The document the pending frame (or gesture) belongs to ([`cancel_stale`], #1918).
    doc: Option<photocraft_doc::DocId>,
    /// The options-bar Straighten button and ⌘-drag (`crop_straighten`).
    pub straighten: crate::crop_straighten::Straighten,
    /// The box's own Undo/Redo while a crop is pending (Photoshop keeps Edit › Undo live during a
    /// crop, so it must step the box, not the document — #3145): frame snapshots.
    undo: Vec<CropStep>,
    redo: Vec<CropStep>,
    /// The frame at the current gesture's press; pushed to `undo` on release when the box changed.
    press_step: Option<CropStep>,
}

/// One frame state of the crop box's own Undo/Redo.
#[derive(Clone, Copy, Debug, PartialEq)]
struct CropStep {
    rect: Option<[f64; 4]>,
    angle: f64,
    default_frame: bool,
}

/// A crop gesture in progress. Rects are `[x0, y0, x1, y1]` in document coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CropDrag {
    /// Drawing a new frame from `anchor`; `prev` is the frame it replaces, restored when the new
    /// one is too small (a click).
    Draw { anchor: [f64; 2], cur: [f64; 2], last: [f64; 2], prev: Option<[f64; 4]> },
    /// Moving the frame `rect` grabbed at `start`.
    Move { start: [f64; 2], rect: [f64; 4] },
    /// Dragging an edge or corner: `hx`/`hy` are -1 (left/top), 1 (right/bottom) or 0 (untouched).
    /// `view` is the view at the press when Auto Center Preview keeps the box's centre in place
    /// while it is resized (default mode, `crop_mode::live_resize`), else `None`.
    Resize { hx: i8, hy: i8, start: [f64; 2], rect: [f64; 4], view: Option<crate::crop_mode::PressView> },
    /// Turning the frame about `center`: the pointer's direction from it at the press (radians)
    /// and the frame's angle then (degrees).
    Rotate { center: [f64; 2], grab: f64, from: f64 },
    /// Default mode, inside the box: the image moves under it, so the document point grabbed at
    /// `start` stays under the pointer. `rect` is the frame and `view` the view at the press;
    /// pointer points are read in that view (`crop_mode::to_press`).
    MoveImage { start: [f64; 2], rect: [f64; 4], view: Option<crate::crop_mode::PressView> },
    /// Default mode, outside the box: the image turns about the frame's `center` (which stays put
    /// on screen). `grab` is the pointer's direction from it at the press (radians, in the press
    /// `view`), `from` the frame's angle then (degrees).
    TurnImage { center: [f64; 2], grab: f64, from: f64, view: Option<crate::crop_mode::PressView> },
}

/// What the pointer is over, relative to a crop frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hit {
    Handle(i8, i8),
    Inside,
    Outside,
}

/// Hit-test a frame with `tol` document px of reach (shrunk for tiny frames so they stay movable).
pub fn hit(r: [f64; 4], p: [f64; 2], tol: f64) -> Hit {
    let axis = |lo: f64, hi: f64, v: f64, other_in: bool| -> i8 {
        let t = tol.min((hi - lo) / 4.0).max(0.0);
        // Outside the frame the full reach applies; inside it, the shrunk one.
        let near = |e: f64, outward: f64| (v - e) * outward <= tol && (e - v) * outward <= t;
        match (other_in, near(lo, -1.0), near(hi, 1.0)) {
            (false, ..) => 0,
            (_, true, true) => {
                if (v - lo).abs() <= (v - hi).abs() {
                    -1
                } else {
                    1
                }
            }
            (_, true, false) => -1,
            (_, false, true) => 1,
            _ => 0,
        }
    };
    let in_x = p[0] >= r[0] - tol && p[0] <= r[2] + tol;
    let in_y = p[1] >= r[1] - tol && p[1] <= r[3] + tol;
    let (hx, hy) = (axis(r[0], r[2], p[0], in_y), axis(r[1], r[3], p[1], in_x));
    if hx != 0 || hy != 0 {
        Hit::Handle(hx, hy)
    } else if p[0] > r[0] && p[0] < r[2] && p[1] > r[1] && p[1] < r[3] {
        Hit::Inside
    } else {
        Hit::Outside
    }
}

/// The centre of `[x0, y0, x1, y1]`.
pub fn center(r: [f64; 4]) -> [f64; 2] {
    [(r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0]
}

/// `p` turned `deg` degrees clockwise on screen (y down) about `c`.
pub fn turn(p: [f64; 2], c: [f64; 2], deg: f64) -> [f64; 2] {
    let (s, k) = deg.to_radians().sin_cos();
    let (dx, dy) = (p[0] - c[0], p[1] - c[1]);
    [c[0] + k * dx - s * dy, c[1] + s * dx + k * dy]
}

/// The corners of frame `r` turned `deg` degrees about its centre: top-left, top-right,
/// bottom-right, bottom-left (of the upright frame).
pub fn corners(r: [f64; 4], deg: f64) -> [[f64; 2]; 4] {
    let c = center(r);
    [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]].map(|p| turn(p, c, deg))
}

/// The axis-aligned bounds `[x0, y0, x1, y1]` of frame `r` turned `deg` degrees.
pub fn turned_bounds(r: [f64; 4], deg: f64) -> [f64; 4] {
    if deg == 0.0 {
        return r;
    }
    corners(r, deg)
        .iter()
        .fold([f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY], |b, p| [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])])
}

/// [`hit`] for frame `r` turned `deg` degrees about its centre: handles and inside are found in
/// the frame's own axes.
pub fn hit_turned(r: [f64; 4], deg: f64, p: [f64; 2], tol: f64) -> Hit {
    if deg == 0.0 { hit(r, p, tol) } else { hit(r, turn(p, center(r), -deg), tol) }
}

/// An angle in degrees brought into (-180, 180]; 0 for anything not finite.
pub fn normalized_angle(deg: f64) -> f64 {
    if !deg.is_finite() {
        return 0.0;
    }
    let a = deg.rem_euclid(360.0);
    if a > 180.0 { a - 360.0 } else { a }
}

/// The frame's angle after turning it from `from` degrees by the pointer's sweep about `c` from
/// direction `grab` (radians) to `p`; ⇧ snaps it to 15° steps, as in Photoshop.
pub fn turned_angle(c: [f64; 2], grab: f64, from: f64, p: [f64; 2], snap: bool) -> f64 {
    let now = (p[1] - c[1]).atan2(p[0] - c[0]);
    let a = from + (now - grab).to_degrees();
    normalized_angle(if snap { (a / 15.0).round() * 15.0 } else { a })
}

/// [`resized`] for frame `r` turned `deg` degrees: the handle moves along the frame's own axes by
/// the document-space drag `d`, and the edges opposite stay where they are on the page.
pub fn resized_turned(r: [f64; 4], deg: f64, hx: i8, hy: i8, d: [f64; 2], ratio: Option<f64>, alt: bool) -> [f64; 4] {
    if deg == 0.0 {
        return resized(r, hx, hy, d, ratio, alt);
    }
    let c = center(r);
    // In the frame's axes (about its old centre), then the new centre back on the page.
    let n = resized(r, hx, hy, turn(d, [0.0, 0.0], -deg), ratio, alt);
    let nc = turn(center(n), c, deg);
    let (hw, hh) = ((n[2] - n[0]) / 2.0, (n[3] - n[1]) / 2.0);
    [nc[0] - hw, nc[1] - hh, nc[0] + hw, nc[1] + hh]
}

/// `[x0, y0, x1, y1]` with x0 ≤ x1, y0 ≤ y1.
fn normalized(a: [f64; 2], b: [f64; 2]) -> [f64; 4] {
    [a[0].min(b[0]), a[1].min(b[1]), a[0].max(b[0]), a[1].max(b[1])]
}

fn size_ok(r: [f64; 4]) -> bool {
    r[2] - r[0] >= MIN_FRAME && r[3] - r[1] >= MIN_FRAME
}

/// The options-bar ratio preset as width / height (W : H in W x H x Resolution mode).
pub(crate) fn preset_ratio(app: &PhotocraftApp) -> Option<f64> {
    if app.ui.tool_options.crop_ratio == crate::crop_size::WHR {
        let dpi = app.session.active().map_or(72.0, |s| f64::from(s.doc.resolution_dpi));
        return crate::crop_size::ratio(&app.ui.tool_options, dpi);
    }
    let size = app.session.active().map_or((1.0, 1.0), |s| (s.doc.size.width as f64, s.doc.size.height as f64));
    crate::chrome_ui::crop_ratio(&app.ui.tool_options.crop_ratio, size.0, size.1).map(|(w, h)| w / h).filter(|k| k.is_finite() && *k > 0.0)
}

/// Largest centred rectangle of `ratio` contained in the current frame; never adds area.
fn fitted_ratio(r: [f64; 4], ratio: f64) -> Option<[f64; 4]> {
    let (w, h) = (r[2] - r[0], r[3] - r[1]);
    if !r.iter().all(|v| v.is_finite()) || !w.is_finite() || !h.is_finite() || w <= 0.0 || h <= 0.0 || !ratio.is_finite() || ratio <= 0.0 {
        return None;
    }
    if ((w / h) / ratio - 1.0).abs() < 1e-12 {
        return Some(r);
    }
    let c = [r[0] + w / 2.0, r[1] + h / 2.0];
    let (w, h) = if w / h > ratio { (h * ratio, h) } else { (w, w / ratio) };
    let fitted = [c[0] - w / 2.0, c[1] - h / 2.0, c[0] + w / 2.0, c[1] + h / 2.0];
    (fitted.iter().all(|v| v.is_finite()) && fitted[2] > fitted[0] && fitted[3] > fitted[1]).then_some(fitted)
}

/// Apply committed options to the preview now, before a resize or confirmation is needed.
pub(crate) fn apply_ratio(app: &mut PhotocraftApp) {
    if app.ui.tool != Tool::Crop || app.crop.drag.is_some() {
        return;
    }
    frame(app);
    fit_frame(app);
    crate::crop_mode::sync_frame(app);
}

fn fit_frame(app: &mut PhotocraftApp) {
    let dpi = app.session.active().map_or(72.0, |s| f64::from(s.doc.resolution_dpi));
    if let Some(r) = app.ui.crop_rect
        && let Some(ratio) = preset_ratio(app)
        && let Some(fitted) = fitted_ratio(r, ratio)
        // W x H x Resolution can change output size even when the frame already fits.
        && (fitted != r || crate::crop_size::target(&app.ui.tool_options, dpi).is_some())
    {
        keyed(app, fitted);
    }
}

/// A new frame drawn from `anchor` to `cur`: ⇧ square, ⌥ from the centre, `ratio` held.
pub fn drawn(anchor: [f64; 2], cur: [f64; 2], ratio: Option<f64>, mods: Modifiers) -> [f64; 4] {
    let e = match ratio {
        Some(k) => crate::chrome_ui::marquee_end("fixedRatio", k, 1.0, false, anchor, cur),
        None => crate::chrome_ui::marquee_end("", 0.0, 0.0, mods.shift, anchor, cur),
    };
    if mods.alt { normalized([2.0 * anchor[0] - e[0], 2.0 * anchor[1] - e[1]], e) } else { normalized(anchor, e) }
}

/// Frame `r` with handle (`hx`, `hy`) dragged by `d`. `alt` resizes about the centre; `ratio`
/// (width / height) keeps the proportions.
pub fn resized(r: [f64; 4], hx: i8, hy: i8, d: [f64; 2], ratio: Option<f64>, alt: bool) -> [f64; 4] {
    // Per axis: the fixed point, the signed extent from it, and whether it is centred (the
    // extent is then a half-extent).
    let axis = |lo: f64, hi: f64, h: i8, d: f64| -> (f64, f64, bool) {
        let c = (lo + hi) / 2.0;
        match h {
            1 if alt => (c, hi + d - c, true),
            1 => (lo, hi + d - lo, false),
            -1 if alt => (c, lo + d - c, true),
            -1 => (hi, lo + d - hi, false),
            _ => (c, (hi - lo) / 2.0, true),
        }
    };
    let (fx, mut ex, cx) = axis(r[0], r[2], hx, d[0]);
    let (fy, mut ey, cy) = axis(r[1], r[3], hy, d[1]);
    if let Some(k) = ratio.filter(|k| k.is_finite() && *k > 0.0) {
        let full = |e: f64, c: bool| e.abs() * if c { 2.0 } else { 1.0 };
        let (w, h) = (full(ex, cx), full(ey, cy));
        let (w, h) = match (hx != 0, hy != 0) {
            // A corner: the larger relative extent wins.
            (true, true) => {
                let s = (w / k).max(h);
                (s * k, s)
            }
            (true, false) => (w, w / k),
            (false, true) => (h * k, h),
            (false, false) => (w, h),
        };
        let back = |e: f64, len: f64, c: bool| if e < 0.0 { -1.0 } else { 1.0 } * if c { len / 2.0 } else { len };
        ex = back(ex, w, cx);
        ey = back(ey, h, cy);
    }
    let span = |f: f64, e: f64, c: bool| if c { (f - e, f + e) } else { (f, f + e) };
    let ((ax, bx), (ay, by)) = (span(fx, ex, cx), span(fy, ey, cy));
    normalized([ax, ay], [bx, by])
}

/// Space is held (set by the canvas each frame, or by `ui.pointer`'s `space` flag).
pub fn set_space(app: &mut PhotocraftApp, down: bool) {
    app.crop.space = down;
}

/// As in Photoshop, the Crop tool always shows a frame: on picking the tool, and after a crop is
/// cancelled or committed, it frames the selection's bounds, or the whole canvas. In default mode
/// the view then follows the frame being edited (`crop_mode::sync`).
pub fn ensure_frame(app: &mut PhotocraftApp) {
    frame(app);
    crate::crop_mode::sync(app, None);
}

fn frame(app: &mut PhotocraftApp) {
    if app.ui.tool != Tool::Crop {
        app.crop.editing = false;
        crate::crop_straighten::reset(app);
        // Leaving the tool drops an untouched default frame and, as in Photoshop, commits a
        // drawn one (#3368).
        if app.crop.default_frame {
            app.ui.crop_rect = None;
            app.crop.default_frame = false;
        }
        if app.ui.crop_rect.is_some() && !keeps_pending(app) {
            commit_on_leave(app);
        }
        if app.ui.crop_rect.is_none() {
            app.ui.crop_angle = 0.0;
        }
        return;
    }
    // No frame (cancelled or committed): the next one starts upright.
    if app.ui.crop_rect.is_none() && app.crop.drag.is_none() {
        app.ui.crop_angle = 0.0;
    }
    let key = app.session.active_index().zip(app.session.active()).map(|(i, st)| (i, st.doc.size.width, st.doc.size.height));
    // Another document (or a resized one) gets its own default frame.
    if app.crop.default_frame && app.crop.frame_for != key && app.crop.drag.is_none() {
        app.ui.crop_rect = None;
    }
    if app.ui.crop_rect.is_some() || app.crop.drag.is_some() {
        return;
    }
    let Some(st) = app.session.active() else { return };
    let canvas = st.doc.bounds();
    // Computed once per new frame, not per frame drawn.
    let r = st.doc.selection.as_ref().map(|s| s.content_bounds().intersect(&canvas)).filter(|b| !b.is_empty()).unwrap_or(canvas);
    if r.is_empty() {
        return;
    }
    app.ui.crop_rect = Some([f64::from(r.x0), f64::from(r.y0), f64::from(r.x1), f64::from(r.y1)]);
    app.crop.default_frame = true;
    app.crop.frame_for = key;
    app.crop.editing = false;
    app.crop.undo.clear();
    app.crop.redo.clear();
    app.crop.press_step = None;
    claim(app);
    fit_frame(app);
}

/// Another tool is current but the drawn frame stays pending: a gesture on it is still in
/// progress, or the pen's eraser end stands in for the Crop tool until the tip comes back
/// (`stylus`). The held tools (Space, ⌘…, `hold_keys`) never change the current tool.
fn keeps_pending(app: &PhotocraftApp) -> bool {
    app.crop.drag.is_some() || app.stylus.tool_before_eraser == Some(Tool::Crop)
}

/// Commits the drawn frame on leaving the Crop tool as ↵ would with the tool still picked (so the
/// view goes upright too, `crop_mode::upright`), then gives the new tool back.
fn commit_on_leave(app: &mut PhotocraftApp) {
    let tool = std::mem::replace(&mut app.ui.tool, Tool::Crop);
    crate::canvas::commit_crop(app);
    app.ui.tool = tool;
}

/// A crop gesture is in progress: Space repositions the frame rather than panning.
pub fn active(app: &PhotocraftApp) -> bool {
    app.ui.tool == Tool::Crop && app.crop.drag.is_some()
}

/// Cancel the pending crop: the frame, any gesture on it and the default-frame state. With the
/// Crop tool still picked, [`ensure_frame`] then makes the active document a new default frame.
pub fn cancel(app: &mut PhotocraftApp) {
    crate::crop_mode::upright(app);
    app.ui.crop_rect = None;
    app.ui.crop_angle = 0.0;
    app.crop.drag = None;
    app.crop.default_frame = false;
    app.crop.frame_for = None;
    app.crop.editing = false;
    clear_history(app);
    crate::crop_straighten::reset(app);
}

/// Drop the box's own Undo/Redo (leaving the tool, cancelling or committing a crop).
pub fn clear_history(app: &mut PhotocraftApp) {
    app.crop.undo.clear();
    app.crop.redo.clear();
    app.crop.press_step = None;
}

/// The frame's current state, as an undo step.
fn step_of(app: &PhotocraftApp) -> CropStep {
    CropStep { rect: app.ui.crop_rect, angle: app.ui.crop_angle, default_frame: app.crop.default_frame }
}

/// Put a frame state back.
fn apply_step(app: &mut PhotocraftApp, s: CropStep) {
    app.ui.crop_rect = s.rect;
    app.ui.crop_angle = s.angle;
    app.crop.default_frame = s.default_frame;
}

/// While the Crop tool has box history, Undo/Redo step the *box*, not the document (Photoshop keeps
/// Edit › Undo live during a crop, #1918/#3145). `Some(enabled)` for those commands, else `None` so
/// the caller falls back to document history.
pub fn is_enabled(app: &PhotocraftApp, id: &str) -> Option<bool> {
    if app.ui.tool != Tool::Crop {
        return None;
    }
    match id {
        "edit.undo" if !app.crop.undo.is_empty() => Some(true),
        "edit.redo" if !app.crop.redo.is_empty() => Some(true),
        "edit.toggleLastState" if !app.crop.undo.is_empty() || !app.crop.redo.is_empty() => Some(false),
        _ => None,
    }
}

/// Intercept Undo/Redo/Toggle Last State while the Crop tool owns box history, for every caller
/// (menus, shortcuts, the control channel, MCP) — as `transform_tool::intercept` does.
pub fn intercept(app: &mut PhotocraftApp, id: &str) -> Option<Result<serde_json::Value, String>> {
    if app.ui.tool != Tool::Crop || (app.crop.undo.is_empty() && app.crop.redo.is_empty()) {
        return None;
    }
    match id {
        "edit.undo" => Some(Ok(step(app, false))),
        "edit.redo" => Some(Ok(step(app, true))),
        "edit.toggleLastState" => Some(Err("Commit or cancel the crop first".into())),
        _ => None,
    }
}

/// Undo (or redo) one box step.
fn step(app: &mut PhotocraftApp, redo: bool) -> serde_json::Value {
    let s = if redo { app.crop.redo.pop() } else { app.crop.undo.pop() };
    let Some(s) = s else { return serde_json::Value::Null };
    let now = step_of(app);
    apply_step(app, s);
    if redo {
        app.crop.undo.push(now);
    } else {
        app.crop.redo.push(now);
    }
    crate::crop_mode::sync_frame(app);
    serde_json::Value::Null
}

/// The pending frame belongs to the active document.
fn claim(app: &mut PhotocraftApp) {
    app.crop.doc = app.session.active().map(|st| st.doc.id);
}

/// A crop is pending: the Crop tool's frame on the active document has been set (drawn, moved,
/// resized or turned), not just the untouched default frame the tool shows.
pub fn pending(app: &PhotocraftApp) -> bool {
    app.ui.tool == Tool::Crop
        && app.ui.crop_rect.is_some()
        && !app.crop.default_frame
        && (app.crop.doc.is_none() || app.crop.doc == app.session.active().map(|st| st.doc.id))
}

/// Menu commands Photoshop greys out while a crop is pending (#1918), until it is committed or
/// cancelled. Checked item by item in Photoshop 27.11: everything except Close, Save, Save As,
/// Save a Copy; Undo, Redo, Toggle Last State, Search; Image › Crop (which commits the frame); the
/// View menu but Proof Setup, Pixel Aspect Ratio (Correction), 32-bit Preview Options, Flip
/// Horizontal and Screen Mode; the Window menu but Workspace and Adjustments; Help but System Info.
pub fn blocks(app: &PhotocraftApp, id: &str) -> bool {
    pending(app) && !available_while_pending(id)
}

fn available_while_pending(id: &str) -> bool {
    if let Some(v) = id.strip_prefix("view.") {
        let greyed = ["proofSetup.", "pixelAspectRatio", "thirtyTwoBitPreviewOptions", "flipHorizontal", "screenMode."];
        return !greyed.iter().any(|g| v.starts_with(g));
    }
    if let Some(w) = id.strip_prefix("window.") {
        return !(w.starts_with("workspace.") || w == "panel.adjustments");
    }
    if id.starts_with("help.") {
        return id != "help.systemInfo";
    }
    matches!(
        id,
        "file.close"
            | "file.save"
            | "file.saveAs"
            | "file.saveACopy"
            | "file.exit"
            | "edit.undo"
            | "edit.redo"
            | "edit.toggleLastState"
            | "edit.search"
            | "image.crop"
    )
}

/// A pending crop belongs to the document it was drawn on (#1918): when another document becomes
/// active (File › New, Open, a tab switch, closing the document) it is cancelled, so it never
/// shows on or crops another document. (As in Photoshop: switching to another document discards
/// the pending crop, and that document gets the tool's own frame.) Run on every frame and command
/// (`sync_views`) and before a commit.
pub fn cancel_stale(app: &mut PhotocraftApp) {
    let active = app.session.active().map(|st| st.doc.id);
    let pending = app.ui.crop_rect.is_some() || app.crop.drag.is_some();
    // A frame made before any document was claimed (control channel, tests) is the active one's.
    if pending && app.crop.doc.is_some() && app.crop.doc != active {
        cancel(app);
    }
    app.crop.doc = active;
}

/// The frame is being edited, so the canvas shows the pixels past its edges (Photoshop's crop
/// preview): from the first press with the tool until the crop is committed or cancelled.
pub fn shows_beyond_canvas(app: &PhotocraftApp) -> bool {
    app.ui.tool == Tool::Crop && app.ui.crop_rect.is_some() && (app.crop.editing || app.crop.drag.is_some())
}

/// The button went down on the canvas with the Crop tool: the frame is being edited from now on.
/// Returns true when that is new.
pub fn press(app: &mut PhotocraftApp) -> bool {
    let new = app.ui.tool == Tool::Crop && app.ui.crop_rect.is_some() && !app.crop.editing;
    if new {
        app.crop.editing = true;
    }
    new
}

pub(crate) fn tolerance(app: &PhotocraftApp) -> f64 {
    HANDLE_PX / (app.point_zoom() as f64).max(0.01)
}

/// The pending frame's angle (degrees), 0 when it isn't a finite number.
pub fn angle(app: &PhotocraftApp) -> f64 {
    normalized_angle(app.ui.crop_angle)
}

/// A press at document point `p` would turn the frame (Crop tool, outside the frame and its
/// handles) rather than draw, move or resize one.
pub fn turns_at(app: &PhotocraftApp, p: [f64; 2]) -> bool {
    app.ui.tool == Tool::Crop
        && app.ui.crop_rect.filter(|r| r.iter().all(|v| v.is_finite())).is_some_and(|r| hit_turned(r, angle(app), p, tolerance(app)) == Hit::Outside)
}

/// A double-click at document point `p` commits the crop (Photoshop: double-click inside the box,
/// like ↵): Crop tool, a pending frame, and `p` inside it (not on a handle, not outside).
pub fn commits_at(app: &PhotocraftApp, p: [f64; 2]) -> bool {
    app.ui.tool == Tool::Crop
        && app.crop.drag.is_none()
        && app.ui.crop_rect.filter(|r| r.iter().all(|v| v.is_finite())).is_some_and(|r| hit_turned(r, angle(app), p, tolerance(app)) == Hit::Inside)
}

/// The frame is being turned: the angle for the readout beside the pointer. Classic Mode shows
/// the frame's turn; the default mode the image's (clockwise positive, as Photoshop 25.1 does).
pub fn turning(app: &PhotocraftApp) -> Option<f64> {
    match app.crop.drag {
        Some(CropDrag::Rotate { .. }) => Some(angle(app)),
        Some(CropDrag::TurnImage { .. }) => Some(crate::crop_mode::image_turn(angle(app))),
        _ => None,
    }
}

/// The size ↵ would crop frame `r` to: its own width and height (before any turn) in whole
/// document px, at least 1; `None` when they aren't finite.
pub fn frame_size(r: [f64; 4]) -> Option<[f64; 2]> {
    let (w, h) = (r[2] - r[0], r[3] - r[1]);
    (w.is_finite() && h.is_finite()).then(|| [w.round().max(1.0), h.round().max(1.0)])
}

/// The frame is being drawn or resized: its W × H, for the readout beside the pointer (as the
/// marquee's, in px).
pub fn sizing(app: &PhotocraftApp) -> Option<[f64; 2]> {
    match app.crop.drag {
        // Until the pointer moves, the frame shown is still the previous one.
        Some(CropDrag::Draw { anchor, cur, .. }) if anchor != cur => {}
        Some(CropDrag::Resize { .. }) => {}
        _ => return None,
    }
    frame_size(app.ui.crop_rect?)
}

/// The Crop tool owns the arrow keys and X: a pending frame and no gesture in progress.
pub fn takes_keys(app: &PhotocraftApp) -> bool {
    app.ui.tool == Tool::Crop && app.crop.drag.is_none() && app.ui.transform.is_none() && app.ui.crop_rect.is_some_and(|r| r.iter().all(|v| v.is_finite()))
}

/// The frame now set by the keys: a real frame (no longer the default), being edited.
fn keyed(app: &mut PhotocraftApp, r: [f64; 4]) {
    if r.iter().all(|v| v.is_finite()) {
        app.ui.crop_rect = Some(r);
        app.crop.default_frame = false;
        app.crop.editing = true;
    }
}

/// Moves the pending frame by (`dx`, `dy`) document px (a turned frame moves on the page, keeping
/// its angle); in default mode the image moves that way on screen under the box instead. Returns
/// false when the Crop tool doesn't take the keys ([`takes_keys`]).
pub fn nudge(app: &mut PhotocraftApp, dx: f64, dy: f64) -> bool {
    let Some(r) = app.ui.crop_rect.filter(|_| takes_keys(app)) else { return false };
    if crate::crop_mode::moves_image(app) {
        if dx.is_finite() && dy.is_finite() {
            app.crop.default_frame = false;
            app.crop.editing = true;
            crate::crop_mode::sync_frame(app);
            let d = crate::crop_mode::screen_to_doc_vec(app, [dx, dy]);
            crate::crop_mode::shift_image(app, d);
        }
        return true;
    }
    if dx.is_finite() && dy.is_finite() {
        keyed(app, [r[0] + dx, r[1] + dy, r[2] + dx, r[3] + dy]);
    }
    true
}

/// Frame `r` with its width and height swapped about its centre.
pub fn swapped(r: [f64; 4]) -> [f64; 4] {
    let c = center(r);
    let (hw, hh) = ((r[2] - r[0]) / 2.0, (r[3] - r[1]) / 2.0);
    [c[0] - hh, c[1] - hw, c[0] + hh, c[1] + hw]
}

/// The options-bar ratio `key` with its sides swapped (what the bar's ⇄ button does), `None` when
/// no ratio is set; "original" becomes the document's size turned on its side.
pub fn swapped_ratio(key: &str, doc_w: f64, doc_h: f64) -> Option<String> {
    let (w, h) = crate::chrome_ui::crop_ratio(key, doc_w, doc_h).filter(|(w, h)| w.is_finite() && h.is_finite() && *w > 0.0 && *h > 0.0)?;
    Some(format!("{}:{}", crate::widgets::fmt_num(h), crate::widgets::fmt_num(w)))
}

/// X: swaps the pending frame between portrait and landscape about its centre, keeping its angle,
/// and swaps the options-bar ratio with it, as in Photoshop. Returns false when the Crop tool
/// doesn't take the keys ([`takes_keys`]), so X swaps the colours as usual.
pub fn swap_orientation(app: &mut PhotocraftApp) -> bool {
    let Some(r) = app.ui.crop_rect.filter(|_| takes_keys(app)) else { return false };
    let size = app.session.active().map_or((0.0, 0.0), |s| (f64::from(s.doc.size.width), f64::from(s.doc.size.height)));
    // W x H x Resolution swaps W and H instead of a ratio.
    if !crate::crop_size::swap(&mut app.ui.tool_options)
        && let Some(k) = swapped_ratio(&app.ui.tool_options.crop_ratio, size.0, size.1)
    {
        app.ui.tool_options.crop_ratio = k;
    }
    if preset_ratio(app).is_some() {
        apply_ratio(app);
    } else {
        keyed(app, swapped(r));
        crate::crop_mode::sync_frame(app);
        crate::crop_mode::auto_center(app);
    }
    true
}

/// Crop tool keys with a pending frame: arrows nudge it 1 px (⇧ 10 px) instead of the selection
/// or layer, and X swaps its orientation instead of the colours. Returns true when a key was used.
pub fn keys(app: &mut PhotocraftApp, ctx: &egui::Context) -> bool {
    if !takes_keys(app) {
        return false;
    }
    use egui::Key;
    let arrows = [(Key::ArrowLeft, -1.0, 0.0), (Key::ArrowRight, 1.0, 0.0), (Key::ArrowUp, 0.0, -1.0), (Key::ArrowDown, 0.0, 1.0)];
    for (key, ux, uy) in arrows {
        // ⇧ first: `consume_key` ignores an extra ⇧.
        for mods in [Modifiers::SHIFT, Modifiers::NONE] {
            if ctx.input_mut(|i| i.consume_key(mods, key)) {
                let k = if mods.shift { 10.0 } else { 1.0 };
                return nudge(app, ux * k, uy * k);
            }
        }
    }
    ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::X)) && swap_orientation(app)
}

/// Pointer input for the Crop tool. Returns true when the event was its.
pub fn pointer(app: &mut PhotocraftApp, ev: ToolEvent, mods: Modifiers) -> bool {
    if app.ui.tool != Tool::Crop {
        app.crop.drag = None;
        return false;
    }
    let p = match ev {
        ToolEvent::Down { x, y, .. } | ToolEvent::Move { x, y, .. } | ToolEvent::Up { x, y } => [x, y],
    };
    if app.session.active().is_none() || !p[0].is_finite() || !p[1].is_finite() {
        return true;
    }
    // The Straighten button, or ⌘ held: the drag draws a line to level the frame on.
    if crate::crop_straighten::pointer(app, ev, mods) {
        claim(app);
        // Default mode: the image turns so the line ends up level (the press keeps its point).
        match ev {
            ToolEvent::Down { .. } => crate::crop_mode::sync(app, Some(p)),
            _ => crate::crop_mode::sync_frame(app),
        }
        return true;
    }
    match ev {
        ToolEvent::Down { .. } => {
            claim(app);
            app.crop.editing = true;
            app.crop.press_step = Some(step_of(app));
            // Default mode: the view is upright for the frame (picking the tool already made it so;
            // this catches a frame set over the control channel), about the pressed point.
            crate::crop_mode::sync(app, Some(p));
            let frame = app.ui.crop_rect.filter(|r| r.iter().all(|v| v.is_finite()));
            let deg = angle(app);
            let image = crate::crop_mode::moves_image(app);
            app.crop.drag = Some(match frame.map(|r| (r, hit_turned(r, deg, p, tolerance(app)))) {
                Some((rect, Hit::Handle(hx, hy))) => {
                    let view = if crate::crop_mode::live_center(app) { crate::crop_mode::press_view(app) } else { None };
                    CropDrag::Resize { hx, hy, start: p, rect, view }
                }
                // Default mode: inside the box the image moves, the box staying where it is on
                // screen (Photoshop doesn't re-centre it). Inside the untouched frame a drag still
                // draws a new box.
                Some((rect, Hit::Inside)) if !app.crop.default_frame && image => {
                    CropDrag::MoveImage { start: p, rect, view: crate::crop_mode::press_view(app) }
                }
                Some((rect, Hit::Inside)) if !app.crop.default_frame => CropDrag::Move { start: p, rect },
                Some((rect, Hit::Outside)) if image => {
                    let c = center(rect);
                    CropDrag::TurnImage { center: c, grab: (p[1] - c[1]).atan2(p[0] - c[0]), from: deg, view: crate::crop_mode::press_view(app) }
                }
                // Outside the frame a drag turns it, as in Photoshop (#1792).
                Some((rect, Hit::Outside)) => {
                    let c = center(rect);
                    CropDrag::Rotate { center: c, grab: (p[1] - c[1]).atan2(p[0] - c[0]), from: deg }
                }
                _ => {
                    app.ui.crop_angle = 0.0;
                    CropDrag::Draw { anchor: p, cur: p, last: p, prev: app.ui.crop_rect }
                }
            });
        }
        ToolEvent::Move { .. } => update(app, p, mods),
        ToolEvent::Up { .. } => {
            update(app, p, mods);
            let Some(drag) = app.crop.drag.take() else { return true };
            let ok = app.ui.crop_rect.is_some_and(size_ok);
            // A click outside the untouched default frame (no turn) leaves it the default.
            let changed = !matches!(drag, CropDrag::Rotate { from, .. } | CropDrag::TurnImage { from, .. } if from == angle(app))
                && !matches!(drag, CropDrag::MoveImage { rect, .. } if app.ui.crop_rect == Some(rect));
            if ok && changed {
                app.crop.default_frame = false;
            }
            match drag {
                CropDrag::Draw { prev, .. } if !ok => app.ui.crop_rect = prev,
                CropDrag::Move { rect, .. } | CropDrag::Resize { rect, view: None, .. } if !ok => app.ui.crop_rect = Some(rect),
                // A live-centred resize collapsed: the frame and the view it moved go back.
                CropDrag::Resize { rect, view: Some(view), .. } if !ok => crate::crop_mode::offset_image(app, rect, Some(view), [0.0, 0.0]),
                // Auto Center Preview: a box drawn in default mode moves to the middle. (A resized
                // one stays where its centre was: `crop_mode::live_resize`.)
                CropDrag::Draw { .. } if ok => {
                    crate::crop_mode::sync_frame(app);
                    crate::crop_mode::auto_center(app);
                }
                _ => {}
            }
            if let Some(before) = app.crop.press_step.take()
                && step_of(app) != before
            {
                app.crop.undo.push(before);
                app.crop.redo.clear();
            }
        }
    }
    crate::crop_mode::sync_frame(app);
    true
}

/// The frame follows the pointer at `p`.
fn update(app: &mut PhotocraftApp, p: [f64; 2], mods: Modifiers) {
    let ratio = preset_ratio(app);
    let space = app.crop.space;
    let deg = angle(app);
    match app.crop.drag {
        Some(CropDrag::Rotate { center, grab, from }) => {
            app.ui.crop_angle = turned_angle(center, grab, from, p, mods.shift);
            return;
        }
        Some(CropDrag::TurnImage { center, grab, from, view }) => {
            // The image turns with the pointer's sweep on screen, so the frame turns against it.
            let q = crate::crop_mode::to_press(app, view, p);
            let a = from - ((q[1] - center[1]).atan2(q[0] - center[0]) - grab).to_degrees();
            if a.is_finite() {
                app.ui.crop_angle = normalized_angle(if mods.shift { (a / 15.0).round() * 15.0 } else { a });
                crate::crop_mode::sync_frame(app);
            }
            return;
        }
        Some(CropDrag::MoveImage { start, rect, view }) => {
            // The image follows the pointer: the frame moves the other way on it, and the view
            // with the frame, so the box stays put.
            let q = crate::crop_mode::to_press(app, view, p);
            crate::crop_mode::offset_image(app, rect, view, [q[0] - start[0], q[1] - start[1]]);
            return;
        }
        Some(CropDrag::Resize { hx, hy, start, rect, view: Some(view) }) => {
            // Auto Center Preview: the box stays centred where it was while the handle follows
            // the pointer.
            let q = crate::crop_mode::to_press(app, Some(view), p);
            let ratio = resize_ratio(app, rect, mods);
            crate::crop_mode::live_resize(app, crate::crop_mode::LiveResize { rect, deg, hx, hy, start, q, ratio, alt: mods.alt }, view);
            return;
        }
        _ => {}
    }
    let Some(drag) = app.crop.drag.as_mut() else { return };
    let rect = match drag {
        CropDrag::Draw { anchor, cur, last, .. } => {
            if space {
                // Space: the frame being drawn follows the pointer, keeping its size.
                let d = [p[0] - last[0], p[1] - last[1]];
                *anchor = [anchor[0] + d[0], anchor[1] + d[1]];
                *cur = [cur[0] + d[0], cur[1] + d[1]];
            } else {
                *cur = p;
            }
            *last = p;
            if *anchor == *cur {
                return;
            }
            drawn(*anchor, *cur, ratio, mods)
        }
        CropDrag::Move { start, rect } => {
            let (dx, dy) = (p[0] - start[0], p[1] - start[1]);
            [rect[0] + dx, rect[1] + dy, rect[2] + dx, rect[3] + dy]
        }
        CropDrag::Resize { hx, hy, start, rect, .. } => {
            let keep = || {
                let (w, h) = (rect[2] - rect[0], rect[3] - rect[1]);
                (w > 0.0 && h > 0.0).then(|| w / h)
            };
            let ratio = ratio.or_else(|| if mods.shift { keep() } else { None });
            resized_turned(*rect, deg, *hx, *hy, [p[0] - start[0], p[1] - start[1]], ratio, mods.alt)
        }
        CropDrag::Rotate { .. } | CropDrag::TurnImage { .. } | CropDrag::MoveImage { .. } => return,
    };
    app.ui.crop_rect = Some(rect);
}

/// Geometry and snapping must use the same ratio so constrained handles stay aligned.
pub(crate) fn resize_ratio(app: &PhotocraftApp, rect: [f64; 4], mods: Modifiers) -> Option<f64> {
    let keep = (rect[2] - rect[0] > 0.0 && rect[3] - rect[1] > 0.0).then(|| (rect[2] - rect[0]) / (rect[3] - rect[1]));
    preset_ratio(app).or(if mods.shift { keep } else { None })
}

/// Cursor over the canvas at document point `p`: resize arrows on the edges, move inside (the
/// crosshair inside the untouched default frame, where a drag draws a new one).
pub fn cursor(app: &PhotocraftApp, p: [f64; 2]) -> Option<CursorIcon> {
    if app.ui.tool != Tool::Crop {
        return None;
    }
    if crate::crop_straighten::mode(app) {
        return Some(CursorIcon::Crosshair);
    }
    let deg = angle(app);
    let h = match app.crop.drag {
        Some(CropDrag::Move { .. } | CropDrag::MoveImage { .. }) => Hit::Inside,
        Some(CropDrag::Resize { hx, hy, .. }) => Hit::Handle(hx, hy),
        Some(CropDrag::Draw { .. }) => return Some(CursorIcon::Crosshair),
        Some(CropDrag::Rotate { .. } | CropDrag::TurnImage { .. }) => Hit::Outside,
        None => hit_turned(app.ui.crop_rect?, deg, p, tolerance(app)),
    };
    Some(match h {
        // The arrow as the handle faces on screen: the frame's turn plus the view's.
        Hit::Handle(hx, hy) => resize_icon(hx, hy, deg + f64::from(crate::crop_mode::view_rotation(app))),
        // Inside the untouched default frame a drag draws a new one (`pointer`), so nothing moves (#4461).
        Hit::Inside if app.crop.default_frame => CursorIcon::Crosshair,
        Hit::Inside => CursorIcon::Move,
        // Photoshop's curved two-headed arrow (`draw_turn_cursor`, from `turn_cursor_dir`).
        Hit::Outside => CursorIcon::None,
    })
}

/// Where the turn cursor's arc bends at document point `p` (pointer outside the frame, or turning
/// it): towards the frame corner of `p`'s quadrant, along the diagonal of the frame's own axes.
/// As in Photoshop that gives four cursors, one per corner; they turn with the frame.
pub fn turn_cursor_dir(app: &PhotocraftApp, p: [f64; 2]) -> Option<[f64; 2]> {
    if app.ui.tool != Tool::Crop || crate::crop_straighten::mode(app) {
        return None;
    }
    let r = app.ui.crop_rect.filter(|r| r.iter().all(|v| v.is_finite()))?;
    let deg = angle(app);
    let turning = match app.crop.drag {
        Some(CropDrag::Rotate { .. } | CropDrag::TurnImage { .. }) => true,
        Some(_) => false,
        None => hit_turned(r, deg, p, tolerance(app)) == Hit::Outside,
    };
    if !(turning && p[0].is_finite() && p[1].is_finite()) {
        return None;
    }
    let c = center(r);
    let local = turn(p, c, -deg);
    let sx = if local[0] < c[0] { 1.0 } else { -1.0 };
    let sy = if local[1] < c[1] { 1.0 } else { -1.0 };
    Some(turn([sx * std::f64::consts::FRAC_1_SQRT_2, sy * std::f64::consts::FRAC_1_SQRT_2], [0.0, 0.0], deg))
}

/// Photoshop's turn cursor (Crop, and Free Transform's rotate zone) at screen point `p`: a short
/// arc with an arrowhead at each end, curving round the frame corner that lies in screen direction
/// `toward` (an original drawing, no proprietary cursor asset).
pub fn draw_turn_cursor(ctx: &egui::Context, p: egui::Pos2, toward: egui::Vec2) {
    let len = toward.length();
    if !len.is_finite() || len < 1e-6 {
        return;
    }
    let u = toward / len;
    let tokens = crate::theme::Tokens::get(ctx);
    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("crop-turn-cursor")));
    const R: f32 = 12.0;
    const HALF: f32 = 0.75;
    let c = p + u * R;
    let mid = (-u.y).atan2(-u.x);
    let at = |a: f32| c + egui::vec2(a.cos(), a.sin()) * R;
    let points: Vec<egui::Pos2> = (0..=16).map(|i| at(mid - HALF + 2.0 * HALF * i as f32 / 16.0)).collect();
    painter.add(egui::Shape::line(points.clone(), egui::Stroke::new(3.5, tokens.shadow)));
    painter.add(egui::Shape::line(points, egui::Stroke::new(1.5, tokens.accent_text)));
    for (a, sign) in [(mid - HALF, -1.0f32), (mid + HALF, 1.0)] {
        let tip = at(a) + egui::vec2(-a.sin(), a.cos()) * (sign * 2.0);
        let tangent = egui::vec2(-a.sin(), a.cos()) * sign;
        let normal = egui::vec2(a.cos(), a.sin());
        painter.add(egui::Shape::convex_polygon(
            vec![tip + tangent * 3.5, tip - tangent * 3.0 + normal * 3.5, tip - tangent * 3.0 - normal * 3.5],
            tokens.accent_text,
            egui::Stroke::new(1.0, tokens.shadow),
        ));
    }
}

/// The resize arrow for handle (`hx`, `hy`) of a frame turned `deg` degrees: the handle's outward
/// direction, turned with the frame, to the nearest of the four arrows.
fn resize_icon(hx: i8, hy: i8, deg: f64) -> CursorIcon {
    let d = turn([f64::from(hx), f64::from(hy)], [0.0, 0.0], deg);
    let a = d[1].atan2(d[0]).rem_euclid(std::f64::consts::PI);
    match ((a / std::f64::consts::FRAC_PI_4).round() as i64).rem_euclid(4) {
        0 => CursorIcon::ResizeHorizontal,
        1 => CursorIcon::ResizeNwSe,
        2 => CursorIcon::ResizeVertical,
        _ => CursorIcon::ResizeNeSw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::tool_event;
    use photocraft_doc::{Color, ColorMode, Document, SampleType, Size};
    use serde_json::json;

    const NONE: Modifiers = Modifiers::NONE;
    const SHIFT: Modifiers = Modifiers::SHIFT;

    fn app(depth: SampleType) -> PhotocraftApp {
        let doc = Document::with_background("crop", Size::new(200, 100), ColorMode::Rgb, depth, Color::WHITE);
        let mut s = photocraft_engine::Session::new();
        s.add_document(doc, None);
        let mut app = PhotocraftApp::new(s, crate::Services::default());
        app.ui.tool = Tool::Crop;
        // These tests pin Classic Mode (the frame moves and turns over the image); the default
        // mode's are in `crop_mode`.
        app.ui.tool_options.crop_shield.classic_mode = true;
        app.ui.extras.snap = false;
        app.ui.view.show.smart_guides = false;
        app
    }

    fn drag(app: &mut PhotocraftApp, pts: &[[f64; 2]], mods: Modifiers) {
        let (first, last) = (pts[0], pts[pts.len() - 1]);
        tool_event(app, ToolEvent::Down { x: first[0], y: first[1], pressure: 1.0 }, mods);
        for p in &pts[1..] {
            tool_event(app, ToolEvent::Move { x: p[0], y: p[1], pressure: 1.0 }, mods);
        }
        tool_event(app, ToolEvent::Up { x: last[0], y: last[1] }, mods);
    }

    #[test]
    fn fitting_a_ratio_keeps_the_centre_angle_and_area() {
        let mut app = app(SampleType::U8);
        app.ui.crop_rect = Some([20.0, 10.0, 180.0, 90.0]);
        app.ui.crop_angle = 25.0;
        for key in ["1:1", "16:9", "9:16", "", "0:0"] {
            let before = app.ui.crop_rect.unwrap();
            app.ui.tool_options.crop_ratio = key.into();
            crate::crop_size::chosen(&mut app);
            let after = app.ui.crop_rect.unwrap();
            assert_eq!(center(after), center(before));
            assert_eq!(app.ui.crop_angle, 25.0);
            assert!(after[0] >= before[0] && after[1] >= before[1] && after[2] <= before[2] && after[3] <= before[3]);
            if let Some(ratio) = preset_ratio(&app) {
                assert!(((after[2] - after[0]) / (after[3] - after[1]) - ratio).abs() < 1e-10);
            } else {
                assert_eq!(after, before);
            }
        }
        for ratio in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(fitted_ratio([0.0, 0.0, 200.0, 100.0], ratio).is_none());
        }
        assert!(fitted_ratio([-f64::MAX, 0.0, f64::MAX, 100.0], 1.0).is_none());
    }

    #[test]
    fn choosing_a_ratio_fits_and_commits_without_a_resize() {
        for classic in [false, true] {
            let mut app = app(SampleType::U8);
            app.ui.tool_options.crop_shield.classic_mode = classic;
            ensure_frame(&mut app);
            app.ui.tool_options.crop_ratio = "1:1".into();
            crate::crop_size::chosen(&mut app);
            assert_eq!(app.ui.crop_rect, Some([50.0, 0.0, 150.0, 100.0]));
            crate::canvas::commit_crop(&mut app);
            assert_eq!(app.session.active().unwrap().doc.size, Size::new(100, 100));
        }
    }

    #[test]
    fn entering_crop_with_a_saved_ratio_fits_the_selection_or_canvas() {
        let mut app = app(SampleType::U8);
        app.ui.tool_options.crop_ratio = "1:1".into();
        ensure_frame(&mut app);
        assert_eq!(app.ui.crop_rect, Some([50.0, 0.0, 150.0, 100.0]));
        cancel(&mut app);
        app.run("select.rect", json!({"x": 20, "y": 10, "width": 80, "height": 40})).unwrap();
        ensure_frame(&mut app);
        assert_eq!(app.ui.crop_rect, Some([40.0, 10.0, 80.0, 50.0]));
    }

    #[test]
    fn picking_the_tool_frames_the_canvas_or_the_selection() {
        // #668: the Crop tool started with no frame.
        let mut app = app(SampleType::U8);
        ensure_frame(&mut app);
        assert_eq!(app.ui.crop_rect, Some([0.0, 0.0, 200.0, 100.0]));
        // Committing the untouched frame crops nothing.
        let steps = app.session.active().unwrap().history.past_len();
        crate::canvas::commit_crop(&mut app);
        assert_eq!(app.session.active().unwrap().history.past_len(), steps);
        assert_eq!(app.session.active().unwrap().doc.size, Size::new(200, 100));
        // A drag inside the untouched frame draws a new one instead of moving it.
        ensure_frame(&mut app);
        drag(&mut app, &[[50.0, 20.0], [80.0, 40.0], [120.0, 70.0]], NONE);
        assert_eq!(app.ui.crop_rect, Some([50.0, 20.0, 120.0, 70.0]));
        // Now it's a real frame: a drag inside moves it.
        drag(&mut app, &[[60.0, 30.0], [70.0, 30.0]], NONE);
        assert_eq!(app.ui.crop_rect, Some([60.0, 20.0, 130.0, 70.0]));
        // Cancelling frames the selection's bounds when there is one.
        app.run("select.rect", json!({"x": 10, "y": 10, "width": 30, "height": 20})).unwrap();
        app.ui.crop_rect = None;
        ensure_frame(&mut app);
        assert_eq!(app.ui.crop_rect, Some([10.0, 10.0, 40.0, 30.0]));
        // Leaving the tool drops the untouched frame.
        app.ui.tool = Tool::Brush;
        ensure_frame(&mut app);
        assert_eq!(app.ui.crop_rect, None);
    }

    #[test]
    fn enter_on_the_selection_frame_crops_to_it_at_8_and_16_bit() {
        // #1789: selection, C, ↵ did nothing, as the untouched frame was always treated as a no-op.
        for depth in [SampleType::U8, SampleType::U16] {
            let mut app = app(depth);
            app.run("select.rect", json!({"x": 10, "y": 20, "width": 30, "height": 40})).unwrap();
            ensure_frame(&mut app);
            assert_eq!(app.ui.crop_rect, Some([10.0, 20.0, 40.0, 60.0]));
            crate::canvas::commit_crop(&mut app);
            let doc = &app.session.active().unwrap().doc;
            assert_eq!(doc.size, Size::new(30, 40), "{depth:?}");
            assert_eq!(doc.depth, depth);
            assert!(doc.selection.is_none(), "cropping to the selection deselects, as in Photoshop");
            // The new default frame is the whole (cropped) canvas: ↵ on it does nothing.
            ensure_frame(&mut app);
            assert_eq!(app.ui.crop_rect, Some([0.0, 0.0, 30.0, 40.0]));
            let steps = app.session.active().unwrap().history.past_len();
            crate::canvas::commit_crop(&mut app);
            let st = app.session.active().unwrap();
            assert_eq!((st.doc.size, st.history.past_len()), (Size::new(30, 40), steps), "{depth:?}");
        }
    }

    #[test]
    fn the_frame_is_edited_from_a_press_until_commit_cancel_or_another_tool() {
        let mut app = app(SampleType::U8);
        ensure_frame(&mut app);
        assert!(!shows_beyond_canvas(&app), "a new frame is not being edited");
        assert!(press(&mut app));
        assert!(shows_beyond_canvas(&app));
        assert!(!press(&mut app), "only the first press is news");
        // A click on the frame keeps it edited.
        drag(&mut app, &[[50.0, 50.0]], NONE);
        assert!(shows_beyond_canvas(&app));
        // Esc drops the frame: the new default one isn't edited.
        app.ui.crop_rect = None;
        ensure_frame(&mut app);
        assert!(!shows_beyond_canvas(&app));
        // A drawn frame is edited until it is committed.
        drag(&mut app, &[[10.0, 10.0], [60.0, 40.0]], NONE);
        assert!(shows_beyond_canvas(&app));
        crate::canvas::commit_crop(&mut app);
        ensure_frame(&mut app);
        assert_eq!(app.ui.crop_rect, Some([0.0, 0.0, 50.0, 30.0]));
        assert!(!shows_beyond_canvas(&app));
        // Or until another tool is picked.
        drag(&mut app, &[[5.0, 5.0], [20.0, 20.0]], NONE);
        app.ui.tool = Tool::Brush;
        ensure_frame(&mut app);
        assert!(!app.crop.editing && !shows_beyond_canvas(&app));
        assert!(!press(&mut app), "no frame without the Crop tool");
    }

    #[test]
    fn hit_test_finds_handles_inside_and_outside() {
        let r = [10.0, 10.0, 110.0, 60.0];
        assert_eq!(hit(r, [10.0, 10.0], 4.0), Hit::Handle(-1, -1));
        assert_eq!(hit(r, [112.0, 61.0], 4.0), Hit::Handle(1, 1));
        assert_eq!(hit(r, [60.0, 8.0], 4.0), Hit::Handle(0, -1));
        assert_eq!(hit(r, [108.0, 30.0], 4.0), Hit::Handle(1, 0));
        assert_eq!(hit(r, [60.0, 30.0], 4.0), Hit::Inside);
        assert_eq!(hit(r, [150.0, 30.0], 4.0), Hit::Outside);
        // A tiny frame keeps a movable middle.
        assert_eq!(hit([0.0, 0.0, 4.0, 4.0], [2.0, 2.0], 8.0), Hit::Inside);
    }

    #[test]
    fn dragging_inside_moves_the_frame() {
        let mut app = app(SampleType::U8);
        drag(&mut app, &[[20.0, 20.0], [80.0, 60.0]], NONE);
        assert_eq!(app.ui.crop_rect, Some([20.0, 20.0, 80.0, 60.0]));
        drag(&mut app, &[[50.0, 40.0], [60.0, 45.0], [70.0, 50.0]], NONE);
        assert_eq!(app.ui.crop_rect, Some([40.0, 30.0, 100.0, 70.0]), "moved, not redrawn");
        // A click outside keeps the frame (and leaves it upright).
        drag(&mut app, &[[5.0, 5.0]], NONE);
        assert_eq!(app.ui.crop_rect, Some([40.0, 30.0, 100.0, 70.0]));
        assert_eq!(app.ui.crop_angle, 0.0);
        // Outside, a drag turns the frame rather than drawing a new one (#1792).
        drag(&mut app, &[[150.0, 50.0], [190.0, 90.0]], NONE);
        assert_eq!(app.ui.crop_rect, Some([40.0, 30.0, 100.0, 70.0]));
        assert!(app.ui.crop_angle > 10.0, "{}", app.ui.crop_angle);
    }

    #[test]
    fn dragging_outside_turns_the_frame_about_its_centre() {
        let mut app = app(SampleType::U8);
        app.ui.crop_rect = Some([40.0, 30.0, 100.0, 70.0]);
        // Centre (70, 50): from straight right to straight below is a quarter turn clockwise.
        drag(&mut app, &[[150.0, 50.0], [140.0, 70.0], [70.0, 95.0]], NONE);
        assert!((app.ui.crop_angle - 90.0).abs() < 1e-9, "{}", app.ui.crop_angle);
        assert_eq!(app.ui.crop_rect, Some([40.0, 30.0, 100.0, 70.0]), "turning keeps the frame");
        // Back the other way, past -180 (normalised).
        drag(&mut app, &[[70.0, 120.0], [-10.0, 50.0], [70.0, -20.0]], NONE);
        assert!((app.ui.crop_angle - -90.0).abs() < 1e-9, "{}", app.ui.crop_angle);
        // ⇧ snaps the angle to 15° steps.
        app.ui.crop_angle = 0.0;
        let a = 22.0f64.to_radians();
        drag(&mut app, &[[150.0, 50.0], [70.0 + 80.0 * a.cos(), 50.0 + 80.0 * a.sin()]], SHIFT);
        assert_eq!(app.ui.crop_angle, 15.0);
        assert_eq!(cursor(&app, [150.0, 50.0]), Some(CursorIcon::None), "outside: the drawn turn cursor");
        // Turning the untouched default frame makes it a real frame; a click outside it doesn't.
        let mut app = super::tests::app(SampleType::U8);
        ensure_frame(&mut app);
        drag(&mut app, &[[250.0, 50.0]], NONE);
        assert!(app.crop.default_frame, "a click outside keeps the default frame");
        drag(&mut app, &[[250.0, 50.0], [250.0, 80.0]], NONE);
        assert!(!app.crop.default_frame && app.ui.crop_angle > 0.0);
        // Esc (no frame) starts the next one upright.
        app.ui.crop_rect = None;
        ensure_frame(&mut app);
        assert_eq!((app.ui.crop_rect, app.ui.crop_angle), (Some([0.0, 0.0, 200.0, 100.0]), 0.0));
    }

    #[test]
    fn hit_testing_a_turned_frame_uses_its_own_axes() {
        let r = [40.0, 30.0, 100.0, 70.0];
        // A quarter turn: the frame covers x 50..90, y 20..80 on the page.
        assert_eq!(hit_turned(r, 90.0, [70.0, 75.0], 2.0), Hit::Inside, "inside the turned frame only");
        assert_eq!(hit(r, [70.0, 75.0], 2.0), Hit::Outside);
        assert_eq!(hit_turned(r, 90.0, [45.0, 50.0], 2.0), Hit::Outside, "inside the upright frame only: the rotate zone");
        // The upright frame's top-left corner turns to the top-right of the page.
        assert_eq!(hit_turned(r, 90.0, [90.0, 20.0], 2.0), Hit::Handle(-1, -1));
        // Its right edge's middle is now at the bottom.
        assert_eq!(hit_turned(r, 90.0, [70.0, 80.0], 2.0), Hit::Handle(1, 0));
        let c = corners(r, 90.0);
        for (a, b) in c.iter().zip([[90.0, 20.0], [90.0, 80.0], [50.0, 80.0], [50.0, 20.0]]) {
            assert!((a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9, "{c:?}");
        }
        let b = turned_bounds(r, 90.0);
        assert!((b[0] - 50.0).abs() < 1e-9 && (b[3] - 80.0).abs() < 1e-9, "{b:?}");
        // Handle arrows turn with the frame.
        assert_eq!(resize_icon(1, 0, 0.0), CursorIcon::ResizeHorizontal);
        assert_eq!(resize_icon(1, 0, 90.0), CursorIcon::ResizeVertical);
        assert_eq!(resize_icon(1, 0, 45.0), CursorIcon::ResizeNwSe);
        assert_eq!(resize_icon(-1, -1, 90.0), CursorIcon::ResizeNeSw);
        assert_eq!(normalized_angle(540.0), 180.0);
        assert_eq!(normalized_angle(-190.0), 170.0);
        assert_eq!(normalized_angle(f64::NAN), 0.0);
    }

    #[test]
    fn resizing_a_turned_frame_moves_along_its_axes_and_keeps_the_opposite_edge() {
        let r = [40.0, 30.0, 100.0, 70.0];
        // At 90°, the right edge (local +x) faces down the page: dragging down 10 grows the width.
        let n = resized_turned(r, 90.0, 1, 0, [0.0, 10.0], None, false);
        assert!(((n[2] - n[0]) - 70.0).abs() < 1e-9 && ((n[3] - n[1]) - 40.0).abs() < 1e-9, "{n:?}");
        // The opposite (left) edge stays put on the page: at y = 20 with x 50..90.
        let (old, new) = (corners(r, 90.0), corners(n, 90.0));
        for (i, (a, b)) in [(0, 0), (3, 3)].map(|(i, j)| (old[i], new[j])).iter().enumerate() {
            assert!((a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9, "corner {i}: {a:?} vs {b:?}");
        }
        // A sideways drag does nothing to that handle.
        let same = resized_turned(r, 90.0, 1, 0, [25.0, 0.0], None, false);
        assert!(same.iter().zip(r).all(|(a, b)| (a - b).abs() < 1e-9), "{same:?}");
        // At 0° it is the plain resize.
        assert_eq!(resized_turned(r, 0.0, 1, 1, [5.0, 6.0], None, false), resized(r, 1, 1, [5.0, 6.0], None, false));
        // Through the pointer: grabbing the turned right-edge handle and dragging down 10.
        let mut app = app(SampleType::U8);
        app.ui.crop_rect = Some(r);
        app.ui.crop_angle = 90.0;
        drag(&mut app, &[[70.0, 80.0], [70.0, 90.0]], NONE);
        let got = app.ui.crop_rect.unwrap();
        assert!(got.iter().zip(n).all(|(a, b)| (a - b).abs() < 1e-9), "{got:?} vs {n:?}");
        // Moving a turned frame is a plain translation, angle kept.
        drag(&mut app, &[[70.0, 50.0], [80.0, 55.0]], NONE);
        let moved = app.ui.crop_rect.unwrap();
        assert!(moved.iter().zip([n[0] + 10.0, n[1] + 5.0, n[2] + 10.0, n[3] + 5.0]).all(|(a, b)| (a - b).abs() < 1e-9), "{moved:?}");
        assert_eq!(app.ui.crop_angle, 90.0);
    }

    /// ↵ on a turned frame crops the turned document (one step), at 8, 16 and 32 bits.
    #[test]
    fn commit_crops_to_the_turned_frame() {
        for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let mut app = app(depth);
            app.run("select.rect", json!({"x": 60, "y": 40, "width": 1, "height": 1})).unwrap();
            app.run("edit.fill", json!({"color": "#ff0000"})).unwrap();
            app.run("select.deselect", json!({})).unwrap();
            let past = app.session.active().unwrap().history.past_len();
            // A 40 × 20 frame centred on (60, 40), turned 90° by a drag outside it.
            app.ui.crop_rect = Some([40.0, 30.0, 80.0, 50.0]);
            app.crop.default_frame = false;
            drag(&mut app, &[[120.0, 40.0], [60.0, 100.0]], NONE);
            assert!((app.ui.crop_angle - 90.0).abs() < 1e-6, "{depth:?}: {}", app.ui.crop_angle);
            crate::canvas::commit_crop(&mut app);
            assert!(app.ui.crop_rect.is_none());
            assert_eq!(app.ui.crop_angle, 0.0, "the next frame starts upright");
            let st = app.session.active().unwrap();
            assert_eq!(st.history.past_len(), past + 1, "{depth:?}: one undo step");
            assert_eq!((st.doc.size.width, st.doc.size.height), (40, 20), "{depth:?}");
            assert_eq!(st.doc.depth, depth);
            // The red pixel was at the frame's centre: it is at the cropped image's centre.
            let surf = st.doc.layers[0].surface().unwrap();
            let red = (19..=21).flat_map(|x| (9..=11).map(move |y| (x, y))).map(|(x, y)| surf.rgba(x, y)).find(|px| px[0] > 0.6 && px[1] < 0.4);
            assert!(red.is_some(), "{depth:?}: no red at the centre");
            app.run("edit.undo", json!({})).unwrap();
            assert_eq!(app.session.active().unwrap().doc.size, Size::new(200, 100), "{depth:?}");
        }
    }

    #[test]
    fn moving_the_frame_snaps_its_edges() {
        let mut app = app(SampleType::U8);
        app.ui.extras.snap = true;
        app.ui.crop_rect = Some([10.0, 10.0, 50.0, 40.0]);
        drag(&mut app, &[[30.0, 25.0], [23.0, 25.0]], NONE);
        assert_eq!(app.ui.crop_rect, Some([0.0, 10.0, 40.0, 40.0]), "left edge snaps to the document edge");
    }

    /// #3145: Edit › Undo while a crop is pending steps the *box*, not the document (Photoshop keeps
    /// Undo live during a crop), so a moved frame comes back.
    #[test]
    fn edit_undo_steps_the_crop_box_not_the_document() {
        let mut app = app(SampleType::U8);
        app.ui.crop_rect = Some([40.0, 30.0, 100.0, 70.0]);
        app.crop.default_frame = false;
        let before = app.ui.crop_rect.unwrap();
        let doc_steps = app.session.active().unwrap().history.past_len();
        drag(&mut app, &[[70.0, 50.0], [90.0, 50.0]], NONE); // move the frame right 20
        let moved = app.ui.crop_rect.unwrap();
        assert!((moved[0] - 60.0).abs() < 1e-3, "{moved:?}");
        app.run("edit.undo", json!({})).unwrap();
        assert_eq!(app.ui.crop_rect, Some(before), "undo restores the box");
        assert_eq!(app.session.active().unwrap().history.past_len(), doc_steps, "the document is untouched");
        app.run("edit.redo", json!({})).unwrap();
        assert!((app.ui.crop_rect.unwrap()[0] - 60.0).abs() < 1e-3, "redo re-applies the move");
    }

    #[test]
    fn handles_resize_with_shift_and_alt() {
        let mut app = app(SampleType::U8);
        app.ui.crop_rect = Some([20.0, 20.0, 100.0, 60.0]);
        // Right edge.
        drag(&mut app, &[[100.0, 40.0], [120.0, 50.0]], NONE);
        assert_eq!(app.ui.crop_rect, Some([20.0, 20.0, 120.0, 60.0]));
        // Bottom-right corner, free.
        drag(&mut app, &[[120.0, 60.0], [140.0, 70.0]], NONE);
        assert_eq!(app.ui.crop_rect, Some([20.0, 20.0, 140.0, 70.0]));
        // ⇧ keeps 120:50 from the top-left corner: the larger relative extent wins.
        app.ui.crop_rect = Some([20.0, 20.0, 140.0, 70.0]);
        drag(&mut app, &[[20.0, 20.0], [10.0, 0.0]], SHIFT);
        let r = app.ui.crop_rect.unwrap();
        assert_eq!((r[2], r[3]), (140.0, 70.0), "opposite corner fixed");
        assert!(((r[2] - r[0]) / (r[3] - r[1]) - 120.0 / 50.0).abs() < 1e-9, "{r:?}");
        assert_eq!(r[3] - r[1], 70.0);
        // ⌥ resizes about the centre.
        app.ui.crop_rect = Some([40.0, 20.0, 80.0, 60.0]);
        drag(&mut app, &[[80.0, 40.0], [90.0, 40.0]], Modifiers::ALT);
        assert_eq!(app.ui.crop_rect, Some([30.0, 20.0, 90.0, 60.0]));
        // An options-bar ratio holds on an edge drag, centred on the other axis.
        app.ui.tool_options.crop_ratio = "1:1".into();
        app.ui.crop_rect = Some([40.0, 20.0, 80.0, 60.0]);
        drag(&mut app, &[[80.0, 40.0], [100.0, 40.0]], NONE);
        assert_eq!(app.ui.crop_rect, Some([40.0, 10.0, 100.0, 70.0]));
    }

    #[test]
    fn dragging_a_handle_past_the_other_side_flips_and_collapse_reverts() {
        let mut app = app(SampleType::U8);
        app.ui.crop_rect = Some([20.0, 20.0, 60.0, 60.0]);
        drag(&mut app, &[[60.0, 40.0], [10.0, 40.0]], NONE);
        assert_eq!(app.ui.crop_rect, Some([10.0, 20.0, 20.0, 60.0]));
        // Collapsing to nothing restores the frame the gesture started from.
        drag(&mut app, &[[20.0, 40.0], [10.0, 40.0]], NONE);
        assert_eq!(app.ui.crop_rect, Some([10.0, 20.0, 20.0, 60.0]));
    }

    #[test]
    fn space_repositions_the_frame_being_drawn() {
        let mut app = app(SampleType::U8);
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 10.0, pressure: 1.0 }, NONE);
        tool_event(&mut app, ToolEvent::Move { x: 50.0, y: 40.0, pressure: 1.0 }, NONE);
        assert!(active(&app));
        set_space(&mut app, true);
        tool_event(&mut app, ToolEvent::Move { x: 70.0, y: 50.0, pressure: 1.0 }, NONE);
        assert_eq!(app.ui.crop_rect, Some([30.0, 20.0, 70.0, 50.0]), "same size, moved");
        set_space(&mut app, false);
        tool_event(&mut app, ToolEvent::Move { x: 90.0, y: 60.0, pressure: 1.0 }, NONE);
        tool_event(&mut app, ToolEvent::Up { x: 90.0, y: 60.0 }, NONE);
        assert_eq!(app.ui.crop_rect, Some([30.0, 20.0, 90.0, 60.0]), "sizing resumes from the moved anchor");
        assert!(!active(&app));
    }

    /// Real egui input on the canvas: holding Space mid-draw moves the frame instead of panning.
    #[test]
    fn space_on_the_canvas_moves_the_frame_not_the_view() {
        use egui::{Event, Key, PointerButton, pos2};
        let mut a = app(SampleType::U8);
        let view = crate::state::View { zoom: 2.0, center: [100.0, 50.0], fit_pending: false, fill_pending: false, doc_size: [200, 100], rotation: 0.0 };
        a.ui.views = vec![view.clone()];
        let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(600.0, 400.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                let v = app.ui.views[0].clone();
                crate::canvas::canvas_view(app, ui, 0, ui.max_rect(), v, true);
            },
            a,
        );
        h.run_steps(2);
        let button = |pos, pressed| Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE };
        let space = |pressed| Event::Key { key: Key::Space, physical_key: None, pressed, repeat: false, modifiers: Modifiers::NONE };
        let p0 = pos2(200.0, 150.0);
        h.event(Event::PointerMoved(p0));
        h.event(button(p0, true));
        h.step();
        for p in [pos2(220.0, 170.0), pos2(260.0, 190.0)] {
            h.event(Event::PointerMoved(p));
            h.step();
        }
        let drawn = h.state().ui.crop_rect.unwrap();
        // The frame starts where the button went down (200, 150), not at the first move past
        // egui's drag threshold (#123).
        assert_eq!((drawn[2] - drawn[0], drawn[3] - drawn[1]), (30.0, 20.0), "60x40 screen px at 200%");
        h.event(space(true));
        h.step();
        for p in [pos2(280.0, 200.0), pos2(300.0, 210.0)] {
            h.event(Event::PointerMoved(p));
            h.step();
        }
        let moved = h.state().ui.crop_rect.unwrap();
        assert_eq!(moved, [drawn[0] + 20.0, drawn[1] + 10.0, drawn[2] + 20.0, drawn[3] + 10.0], "moved by 40x20 screen px");
        h.event(space(false));
        h.event(button(pos2(300.0, 210.0), false));
        h.step();
        let st = h.state();
        assert_eq!(st.ui.crop_rect, Some(moved));
        assert_eq!(st.ui.views[0].center, view.center, "the view did not pan");
        assert!(st.crop.drag.is_none());
    }

    /// #1918: a pending crop belongs to the document it was drawn on. File › New, opening a file
    /// or switching tabs cancels it; it never shows on, or crops, another document.
    /// A turned pending frame is dropped with its angle when another document becomes active, and
    /// Esc drops both too, so the next frame starts upright.
    #[test]
    fn cancelling_a_turned_crop_resets_its_angle() {
        let mut app = app(SampleType::U8);
        drag(&mut app, &[[10.0, 10.0], [50.0, 40.0]], NONE);
        app.ui.crop_angle = 30.0;
        app.run("file.new", json!({"width": 300, "height": 150})).unwrap();
        app.sync_views();
        assert_eq!(app.ui.crop_angle, 0.0);
        drag(&mut app, &[[10.0, 10.0], [50.0, 40.0]], NONE);
        app.ui.crop_angle = -12.5;
        cancel(&mut app);
        assert_eq!((app.ui.crop_rect, app.ui.crop_angle, app.crop.drag), (None, 0.0, None));
    }

    #[test]
    fn a_pending_crop_stays_with_its_document() {
        let size = |app: &PhotocraftApp| app.session.active().map(|st| st.doc.size);
        // File › New.
        let mut app = app(SampleType::U8);
        drag(&mut app, &[[10.0, 10.0], [50.0, 40.0]], NONE);
        assert_eq!(app.ui.crop_rect, Some([10.0, 10.0, 50.0, 40.0]));
        app.run("file.new", json!({"width": 300, "height": 150})).unwrap();
        assert_eq!(app.ui.crop_rect, None, "the first document's frame is gone");
        assert!(app.crop.drag.is_none() && !app.crop.editing && !app.crop.default_frame);
        crate::canvas::commit_crop(&mut app);
        assert_eq!(size(&app), Some(Size::new(300, 150)), "↵ crops nothing");
        // The new document gets its own default frame.
        ensure_frame(&mut app);
        assert_eq!(app.ui.crop_rect, Some([0.0, 0.0, 300.0, 150.0]));
        assert!(app.crop.default_frame);
        // Switching tabs (outside a command), even before the next frame's sync: ↵ never crops
        // the other document.
        drag(&mut app, &[[20.0, 20.0], [60.0, 60.0]], NONE);
        assert!(app.session.set_active(0));
        crate::canvas::commit_crop(&mut app);
        assert_eq!(size(&app), Some(Size::new(200, 100)));
        assert_eq!(app.session.documents()[1].doc.size, Size::new(300, 150));
        assert_eq!(app.ui.crop_rect, None);
        // The next frame's sync also drops a frame left on a tab switched away from.
        drag(&mut app, &[[20.0, 20.0], [60.0, 60.0]], NONE);
        assert!(app.session.set_active(1));
        app.sync_views();
        assert_eq!(app.ui.crop_rect, None);
        // Closing another document keeps the pending crop; closing its own cancels it.
        drag(&mut app, &[[20.0, 20.0], [60.0, 60.0]], NONE);
        app.session.close(0);
        app.sync_views();
        assert_eq!(app.ui.crop_rect, Some([20.0, 20.0, 60.0, 60.0]), "still its document");
        crate::canvas::commit_crop(&mut app);
        assert_eq!(size(&app), Some(Size::new(40, 40)));
        drag(&mut app, &[[5.0, 5.0], [25.0, 25.0]], NONE);
        app.session.close(0);
        app.sync_views();
        assert!(app.session.active().is_none());
        assert_eq!(app.ui.crop_rect, None);
        assert!(app.crop.drag.is_none());
        crate::canvas::commit_crop(&mut app);
        ensure_frame(&mut app);
        assert_eq!(app.ui.crop_rect, None, "no document, no frame");
    }

    /// A drag in progress when its document goes away is dropped too, not finished elsewhere.
    #[test]
    fn a_crop_drag_ends_with_its_document() {
        let mut app = app(SampleType::U8);
        tool_event(&mut app, ToolEvent::Down { x: 10.0, y: 10.0, pressure: 1.0 }, NONE);
        tool_event(&mut app, ToolEvent::Move { x: 50.0, y: 40.0, pressure: 1.0 }, NONE);
        app.run("file.new", json!({"width": 300, "height": 150})).unwrap();
        tool_event(&mut app, ToolEvent::Up { x: 60.0, y: 50.0 }, NONE);
        assert_eq!(app.ui.crop_rect, None);
        crate::canvas::commit_crop(&mut app);
        assert_eq!(app.session.active().map(|st| st.doc.size), Some(Size::new(300, 150)));
    }

    /// #1918: as in Photoshop, a pending crop (not the untouched default frame) greys almost every
    /// menu command; the ones Photoshop keeps stay available, and commit or cancel frees the rest.
    /// Walks the whole menu catalog.
    #[test]
    fn a_pending_crop_greys_the_menus_photoshop_greys() {
        let mut app = app(SampleType::U8);
        ensure_frame(&mut app);
        assert!(app.crop.default_frame);
        let ids: Vec<&str> = crate::menu_catalog::CATALOG.iter().map(|e| e.3).filter(|id| *id != "---").collect();
        for id in &ids {
            assert!(!blocks(&app, id), "default frame: {id} stays as it was");
        }
        drag(&mut app, &[[10.0, 10.0], [50.0, 40.0]], NONE);
        assert!(pending(&app));
        let kept = [
            "file.close",
            "file.save",
            "file.saveAs",
            "file.saveACopy",
            "edit.undo",
            "edit.redo",
            "edit.toggleLastState",
            "edit.search",
            "image.crop",
            "view.proofColors",
            "view.gamutWarning",
            "view.zoomIn",
            "view.zoomOut",
            "view.fitOnScreen",
            "view.actualPixels",
            "view.twoHundredPercent",
            "view.printSize",
            "view.extras",
            "view.show.grid",
            "view.rulers",
            "view.snap",
            "view.snapTo.guides",
            "view.lockSlices",
            "view.newGuide",
            "window.arrange.tile",
            "window.panel.layers",
            "help.about",
        ];
        let greyed = [
            "file.new",
            "file.open",
            "file.openAs",
            "file.openExrParts",
            "file.closeAll",
            "file.revert",
            "file.export.exportAs",
            "file.placeEmbedded",
            "file.automate.batch",
            "file.scripts.browse",
            "file.import.notes",
            "file.fileInfo",
            "file.print",
            "edit.cut",
            "edit.fill",
            "edit.freeTransform",
            "edit.colorSettings",
            "edit.preferences.general",
            "image.imageSize",
            "image.mode.grayscale",
            "layer.new.layer",
            "select.all",
            "filter.blur.gaussianBlur",
            "view.proofSetup.workingCmyk",
            "view.pixelAspectRatio.square",
            "view.pixelAspectRatioCorrection",
            "view.thirtyTwoBitPreviewOptions",
            "view.flipHorizontal",
            "view.screenMode.fullScreen",
            "window.workspace.essentials",
            "window.panel.adjustments",
            "help.systemInfo",
        ];
        for id in kept {
            assert!(!blocks(&app, id), "{id} stays available");
        }
        for id in greyed {
            assert!(blocks(&app, id), "{id} is greyed");
            assert!(!crate::menus::is_enabled(&app, id), "{id} greyed in the menus");
            assert!(!crate::menus::modal_allows(&app, id), "{id} refused over the menu gate");
        }
        assert!(
            ids.iter()
                .filter(|id| id.starts_with("layer.") || id.starts_with("type.") || id.starts_with("select.") || id.starts_with("filter."))
                .all(|id| blocks(&app, id))
        );
        // Committing still runs `image.crop` itself, and frees the menus again.
        crate::canvas::commit_crop(&mut app);
        assert_eq!(app.session.active().unwrap().doc.size, Size::new(40, 30));
        assert!(!pending(&app) && crate::menus::is_enabled(&app, "file.new"));
        // Cancelling does too.
        drag(&mut app, &[[5.0, 5.0], [20.0, 20.0]], NONE);
        assert!(blocks(&app, "file.new"));
        cancel(&mut app);
        assert!(!blocks(&app, "file.new"));
        // Another tool: nothing is pending.
        drag(&mut app, &[[5.0, 5.0], [20.0, 20.0]], NONE);
        app.ui.tool = Tool::Brush;
        assert!(!blocks(&app, "file.new"));
    }

    /// #3368: as in Photoshop, picking another tool (toolbar or its key) with a drawn frame
    /// commits the crop silently, and the new tool is current.
    #[test]
    fn another_tool_commits_a_drawn_frame_at_8_and_16_bit() {
        for depth in [SampleType::U8, SampleType::U16] {
            let mut app = app(depth);
            drag(&mut app, &[[10.0, 10.0], [50.0, 40.0]], NONE);
            assert!(pending(&app));
            app.ui.tool = Tool::Move;
            ensure_frame(&mut app);
            let st = app.session.active().unwrap();
            assert_eq!((st.doc.size, st.doc.depth), (Size::new(40, 30), depth));
            assert_eq!(st.history.undo_label(), Some("Crop"));
            assert_eq!(app.ui.tool, Tool::Move);
            assert_eq!((app.ui.crop_rect, app.ui.crop_angle), (None, 0.0));
            // Nothing is left to commit: back on the Crop tool, the new canvas gets a default frame.
            ensure_frame(&mut app);
            assert_eq!(app.session.active().unwrap().doc.size, Size::new(40, 30));
            app.ui.tool = Tool::Crop;
            ensure_frame(&mut app);
            assert!(app.crop.default_frame && app.ui.crop_rect == Some([0.0, 0.0, 40.0, 30.0]));
        }
    }

    /// #3368, default mode: a turned frame crops turned, and the view it turned is left upright.
    #[test]
    fn another_tool_commits_a_turned_frame_and_leaves_the_view_upright() {
        let mut app = app(SampleType::U8);
        app.sync_views();
        app.ui.tool_options.crop_shield.classic_mode = false;
        app.ui.crop_rect = Some([40.0, 30.0, 100.0, 70.0]);
        app.ui.crop_angle = 10.0;
        ensure_frame(&mut app);
        assert!(pending(&app) && crate::crop_mode::view_rotation(&app) != 0.0);
        app.ui.tool = Tool::Brush;
        ensure_frame(&mut app);
        assert_eq!(app.session.active().unwrap().doc.size, Size::new(60, 40));
        assert_eq!(crate::crop_mode::view_rotation(&app), 0.0);
        assert_eq!((app.ui.tool, app.ui.crop_rect), (Tool::Brush, None));
    }

    /// #3368: the untouched default frame (the canvas, or the selection's bounds) is just dropped
    /// on leaving the tool: nothing is cropped.
    #[test]
    fn another_tool_drops_an_untouched_default_frame() {
        for select in [false, true] {
            let mut app = app(SampleType::U8);
            if select {
                app.run("select.rect", json!({"x": 10, "y": 20, "width": 30, "height": 40})).unwrap();
            }
            ensure_frame(&mut app);
            assert!(app.crop.default_frame && app.ui.crop_rect.is_some());
            let steps = app.session.active().unwrap().history.past_len();
            app.ui.tool = Tool::Move;
            ensure_frame(&mut app);
            let st = app.session.active().unwrap();
            assert_eq!((st.doc.size, st.history.past_len()), (Size::new(200, 100), steps), "selection: {select}");
            assert_eq!(app.ui.crop_rect, None);
        }
    }

    /// The pen's eraser end stands in for the Crop tool only until the tip comes back (`stylus`),
    /// and a gesture still in progress isn't cut short: neither commits the frame.
    #[test]
    fn the_stylus_eraser_and_a_gesture_in_progress_keep_the_frame_pending() {
        let mut app = app(SampleType::U8);
        drag(&mut app, &[[10.0, 10.0], [50.0, 40.0]], NONE);
        app.stylus.tool_before_eraser = Some(Tool::Crop);
        app.ui.tool = Tool::Eraser;
        ensure_frame(&mut app);
        assert_eq!(app.session.active().unwrap().doc.size, Size::new(200, 100));
        app.ui.tool = app.stylus.tool_before_eraser.take().unwrap();
        ensure_frame(&mut app);
        assert!(pending(&app) && app.ui.crop_rect == Some([10.0, 10.0, 50.0, 40.0]));

        tool_event(&mut app, ToolEvent::Down { x: 20.0, y: 20.0, pressure: 1.0 }, NONE);
        assert!(app.crop.drag.is_some());
        app.ui.tool = Tool::Move;
        ensure_frame(&mut app);
        assert_eq!(app.session.active().unwrap().doc.size, Size::new(200, 100));
    }

    /// Image › Crop with a pending crop commits the frame (the one Image item Photoshop keeps).
    #[test]
    fn image_crop_menu_commits_a_pending_crop() {
        let mut app = app(SampleType::U8);
        drag(&mut app, &[[10.0, 10.0], [50.0, 40.0]], NONE);
        let ctx = egui::Context::default();
        crate::menus::invoke(&mut app, &ctx, "image.crop", json!({})).unwrap();
        assert_eq!(app.session.active().unwrap().doc.size, Size::new(40, 30));
        assert!(app.ui.crop_rect.is_none() || app.crop.default_frame);
    }

    #[test]
    fn commit_crops_to_the_moved_frame_at_8_and_16_bit() {
        for depth in [SampleType::U8, SampleType::U16] {
            let mut app = app(depth);
            app.run("select.rect", json!({"x": 60, "y": 40, "width": 1, "height": 1})).unwrap();
            app.run("edit.fill", json!({"color": "#ff0000"})).unwrap();
            app.run("select.deselect", json!({})).unwrap();
            drag(&mut app, &[[10.0, 10.0], [50.0, 40.0]], NONE);
            drag(&mut app, &[[30.0, 25.0], [70.0, 45.0]], NONE);
            assert_eq!(app.ui.crop_rect, Some([50.0, 30.0, 90.0, 60.0]));
            crate::canvas::commit_crop(&mut app);
            assert!(app.ui.crop_rect.is_none());
            let doc = &app.session.active().unwrap().doc;
            assert_eq!((doc.size.width, doc.size.height), (40, 30), "{depth:?}");
            assert_eq!(doc.depth, depth);
            let px = doc.layers[0].surface().unwrap().rgba(10, 10);
            assert!(px[0] > 0.99 && px[1] < 0.01, "{depth:?} {px:?}");
        }
    }

    #[test]
    fn degenerate_and_bad_input_never_panics() {
        let mut app = app(SampleType::U8);
        for r in [[0.0; 4], [5.0, 5.0, 5.0, 5.0], [10.0, 10.0, 0.0, 0.0], [f64::NAN, 0.0, 10.0, 10.0], [f64::INFINITY, 0.0, f64::MAX, 1e300]] {
            app.ui.crop_rect = Some(r);
            for m in [NONE, SHIFT, Modifiers::ALT, SHIFT | Modifiers::ALT] {
                drag(&mut app, &[[5.0, 5.0], [f64::NAN, 3.0], [1e300, -1e300], [6.0, 7.0]], m);
                let _ = cursor(&app, [5.0, 5.0]);
                app.ui.crop_rect = Some(r);
            }
        }
        // Odd angles (the control channel can set any) turn, resize and commit without panicking.
        for a in [f64::NAN, f64::INFINITY, -1e300, 1e-300, 720.0] {
            app.ui.crop_rect = Some([10.0, 10.0, 50.0, 40.0]);
            app.ui.crop_angle = a;
            for pts in [[[30.0, 25.0], [f64::NAN, 1.0]], [[30.0, 25.0], [1e300, 2.0]], [[50.0, 40.0], [60.0, 45.0]], [[90.0, 90.0], [-1e300, 1e300]]] {
                drag(&mut app, &pts, SHIFT);
                let _ = cursor(&app, [5.0, 5.0]);
            }
            assert!(app.ui.crop_angle.is_finite() || !a.is_finite());
            crate::canvas::commit_crop(&mut app);
            app.run("edit.undo", json!({})).ok();
        }
        app.ui.tool_options.crop_ratio = "0:0".into();
        app.ui.crop_rect = Some([0.0, 0.0, 10.0, 10.0]);
        drag(&mut app, &[[10.0, 10.0], [30.0, 30.0]], SHIFT);
        crate::canvas::commit_crop(&mut app);
        // No document: events are swallowed quietly.
        let mut empty = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        empty.ui.tool = Tool::Crop;
        drag(&mut empty, &[[1.0, 1.0], [9.0, 9.0]], NONE);
        assert!(empty.ui.crop_rect.is_none());
    }

    /// #1792: outside the frame the turn cursor bends towards the corner of the pointer's quadrant
    /// (four variants, as in Photoshop) and turns with the frame; no turn cursor inside or on a handle.
    #[test]
    fn turn_cursor_points_at_the_quadrant_corner() {
        let mut app = app(SampleType::U8);
        app.ui.crop_rect = Some([20.0, 20.0, 100.0, 60.0]);
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let near = |d: Option<[f64; 2]>, want: [f64; 2]| d.is_some_and(|d| (d[0] - want[0]).abs() < 1e-9 && (d[1] - want[1]).abs() < 1e-9);
        assert!(near(turn_cursor_dir(&app, [0.0, 0.0]), [h, h]), "top-left: towards the top-left corner");
        assert!(near(turn_cursor_dir(&app, [130.0, 5.0]), [-h, h]), "top-right");
        assert!(near(turn_cursor_dir(&app, [130.0, 90.0]), [-h, -h]), "bottom-right");
        assert!(near(turn_cursor_dir(&app, [5.0, 90.0]), [h, -h]), "bottom-left");
        assert!(near(turn_cursor_dir(&app, [150.0, 30.0]), [-h, h]), "beside the right edge, upper half");
        assert_eq!(turn_cursor_dir(&app, [60.0, 40.0]), None, "inside");
        assert_eq!(turn_cursor_dir(&app, [20.0, 20.0]), None, "on a handle");
        // A frame turned 45° turns the cursor with it: the diagonal becomes horizontal.
        app.ui.crop_angle = 45.0;
        assert!(near(turn_cursor_dir(&app, [0.0, 0.0]), [1.0, 0.0]), "{:?}", turn_cursor_dir(&app, [0.0, 0.0]));
        app.ui.crop_angle = f64::NAN;
        assert!(turn_cursor_dir(&app, [f64::NAN, 0.0]).is_none());
        app.ui.tool = Tool::Brush;
        assert_eq!(turn_cursor_dir(&app, [0.0, 0.0]), None);
    }

    /// A double-click inside the frame commits it, as ↵ does; on a handle or outside it doesn't.
    #[test]
    fn double_click_inside_the_frame_commits() {
        let mut app = app(SampleType::U8);
        app.ui.crop_rect = Some([20.0, 20.0, 100.0, 60.0]);
        assert!(commits_at(&app, [60.0, 40.0]));
        assert!(!commits_at(&app, [20.0, 20.0]), "on a handle");
        assert!(!commits_at(&app, [150.0, 40.0]), "outside: that turns the frame");
        app.ui.crop_angle = 30.0;
        assert!(commits_at(&app, [60.0, 40.0]), "inside a turned frame");
        assert!(!commits_at(&app, [25.0, 22.0]), "inside the upright frame but outside the turned one");
        app.ui.crop_angle = 0.0;
        app.ui.crop_rect = None;
        assert!(!commits_at(&app, [60.0, 40.0]), "no frame");
        app.ui.crop_rect = Some([f64::NAN, 0.0, 1.0, 1.0]);
        assert!(!commits_at(&app, [0.5, 0.5]));
        app.ui.crop_rect = Some([20.0, 20.0, 100.0, 60.0]);
        app.ui.tool = Tool::Brush;
        assert!(!commits_at(&app, [60.0, 40.0]));
    }

    #[test]
    fn cursor_follows_the_frame() {
        let mut app = app(SampleType::U8);
        assert_eq!(cursor(&app, [5.0, 5.0]), None, "no frame yet: the default crosshair");
        app.ui.crop_rect = Some([20.0, 20.0, 100.0, 60.0]);
        assert_eq!(cursor(&app, [20.0, 20.0]), Some(CursorIcon::ResizeNwSe));
        assert_eq!(cursor(&app, [100.0, 20.0]), Some(CursorIcon::ResizeNeSw));
        assert_eq!(cursor(&app, [60.0, 60.0]), Some(CursorIcon::ResizeVertical));
        assert_eq!(cursor(&app, [60.0, 40.0]), Some(CursorIcon::Move));
        assert_eq!(cursor(&app, [150.0, 40.0]), Some(CursorIcon::None), "outside: the drawn turn cursor");
        app.ui.tool = Tool::Brush;
        assert_eq!(cursor(&app, [60.0, 40.0]), None);
    }

    /// #4461: inside the untouched default frame a drag draws a new frame, so the pointer is the
    /// drawing crosshair there, not the move cursor; once the frame is changed a drag moves it.
    #[test]
    fn cursor_inside_the_default_frame_draws() {
        let mut app = app(SampleType::U8);
        ensure_frame(&mut app);
        assert!(app.crop.default_frame && app.ui.crop_rect == Some([0.0, 0.0, 200.0, 100.0]));
        assert_eq!(cursor(&app, [100.0, 50.0]), Some(CursorIcon::Crosshair));
        assert_eq!(cursor(&app, [200.0, 50.0]), Some(CursorIcon::ResizeHorizontal), "its handles still resize");
        drag(&mut app, &[[200.0, 50.0], [150.0, 50.0]], NONE);
        assert!(!app.crop.default_frame && app.ui.crop_rect == Some([0.0, 0.0, 150.0, 100.0]));
        assert_eq!(cursor(&app, [100.0, 50.0]), Some(CursorIcon::Move));
    }

    /// One key press through the app's keyboard handling (`shortcuts::handle`).
    fn key(app: &mut PhotocraftApp, key: egui::Key, modifiers: Modifiers) {
        let ctx = egui::Context::default();
        let raw = egui::RawInput {
            events: vec![egui::Event::ModifiersChanged(modifiers), egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers }],
            ..Default::default()
        };
        let mut out = ctx.run_ui(raw, |ui| crate::shortcuts::handle(app, ui.ctx()));
        out.textures_delta.clear();
    }

    fn layer_offset(app: &PhotocraftApp) -> (i32, i32) {
        let st = app.session.active().unwrap();
        let l = st.doc.layer(st.active_layer.unwrap()).unwrap();
        l.surface().map_or((0, 0), |s| (s.content_bounds().x0, s.content_bounds().y0))
    }

    /// #1919: with a pending frame the arrow keys nudge it 1 px (⇧ 10 px), turned or not, and
    /// never the selection or layer; without one they keep their usual job.
    #[test]
    fn arrow_keys_nudge_the_pending_frame() {
        use egui::Key;
        let mut app = app(SampleType::U8);
        app.run("select.rect", json!({"x": 10, "y": 10, "width": 30, "height": 20})).unwrap();
        app.ui.crop_rect = Some([20.0, 20.0, 100.0, 60.0]);
        let steps = app.session.active().unwrap().history.past_len();
        key(&mut app, Key::ArrowRight, NONE);
        assert_eq!(app.ui.crop_rect, Some([21.0, 20.0, 101.0, 60.0]));
        key(&mut app, Key::ArrowDown, SHIFT);
        assert_eq!(app.ui.crop_rect, Some([21.0, 30.0, 101.0, 70.0]));
        key(&mut app, Key::ArrowLeft, SHIFT);
        key(&mut app, Key::ArrowUp, NONE);
        assert_eq!(app.ui.crop_rect, Some([11.0, 29.0, 91.0, 69.0]));
        assert!(app.crop.editing && !app.crop.default_frame);
        assert_eq!(app.session.active().unwrap().history.past_len(), steps, "nothing in the document moved");
        assert_eq!(app_selection_bounds(&app), Some((10, 10, 40, 30)));
        assert_eq!(layer_offset(&app), (0, 0));
        // A turned frame moves on the page and keeps its angle and shape.
        app.ui.crop_angle = 30.0;
        let before = corners(app.ui.crop_rect.unwrap(), 30.0);
        key(&mut app, Key::ArrowRight, SHIFT);
        let after = corners(app.ui.crop_rect.unwrap(), angle(&app));
        for (a, b) in before.iter().zip(after) {
            assert!((b[0] - a[0] - 10.0).abs() < 1e-9 && (b[1] - a[1]).abs() < 1e-9);
        }
        assert_eq!(app.ui.crop_angle, 30.0);
        // The untouched default frame becomes a real one.
        app.ui.crop_rect = None;
        app.ui.crop_angle = 0.0;
        app.run("select.deselect", json!({})).unwrap();
        ensure_frame(&mut app);
        assert!(app.crop.default_frame);
        assert!(nudge(&mut app, 1.0, 0.0));
        assert_eq!(app.ui.crop_rect, Some([1.0, 0.0, 201.0, 100.0]));
        assert!(!app.crop.default_frame);
        // No frame: not the Crop tool's key. With a selection tool the arrows nudge the selection.
        app.ui.crop_rect = None;
        assert!(!nudge(&mut app, 1.0, 0.0));
        app.ui.tool = Tool::RectMarquee;
        app.run("select.rect", json!({"x": 10, "y": 10, "width": 30, "height": 20})).unwrap();
        key(&mut app, Key::ArrowRight, NONE);
        assert_eq!(app_selection_bounds(&app), Some((11, 10, 41, 30)));
        // Not mid-gesture, nor with another tool, nor for a non-finite frame or step.
        app.ui.tool = Tool::Crop;
        app.ui.crop_rect = Some([20.0, 20.0, 100.0, 60.0]);
        app.crop.drag = Some(CropDrag::Move { start: [0.0, 0.0], rect: [20.0, 20.0, 100.0, 60.0] });
        assert!(!nudge(&mut app, 1.0, 0.0));
        app.crop.drag = None;
        assert!(nudge(&mut app, f64::NAN, f64::INFINITY), "the key is the Crop tool's");
        assert_eq!(app.ui.crop_rect, Some([20.0, 20.0, 100.0, 60.0]), "but the frame stays put");
        app.ui.crop_rect = Some([f64::MAX, 0.0, f64::MAX, 1.0]);
        assert!(nudge(&mut app, f64::MAX, 0.0));
        assert_eq!(app.ui.crop_rect, Some([f64::MAX, 0.0, f64::MAX, 1.0]), "an overflow leaves it");
        app.ui.crop_rect = Some([f64::NAN, 0.0, 1.0, 1.0]);
        assert!(!nudge(&mut app, 1.0, 0.0));
        app.ui.crop_rect = Some([20.0, 20.0, 100.0, 60.0]);
        app.ui.tool = Tool::Brush;
        assert!(!nudge(&mut app, 1.0, 0.0));
    }

    fn app_selection_bounds(app: &PhotocraftApp) -> Option<(i32, i32, i32, i32)> {
        let b = app.session.active()?.doc.selection.as_ref()?.content_bounds();
        Some((b.x0, b.y0, b.x1, b.y1))
    }

    /// #1919: X swaps the pending frame between portrait and landscape about its centre (keeping
    /// its angle and swapping a ratio preset); without a frame X still swaps the colours.
    #[test]
    fn x_swaps_the_frame_orientation_not_the_colours() {
        use egui::Key;
        let mut app = app(SampleType::U16);
        app.run("tools.setColors", json!({"foreground": "#ff0000", "background": "#0000ff"})).unwrap();
        let fg = |app: &PhotocraftApp| app.session.tools.foreground;
        let red = fg(&app);
        app.ui.crop_rect = Some([20.0, 20.0, 100.0, 60.0]);
        app.ui.crop_angle = 25.0;
        key(&mut app, Key::X, NONE);
        assert_eq!(app.ui.crop_rect, Some([40.0, 0.0, 80.0, 80.0]), "40 x 80 about the centre (60, 40)");
        assert_eq!(app.ui.crop_angle, 25.0);
        assert_eq!(fg(&app), red, "the colours stay");
        key(&mut app, Key::X, NONE);
        assert_eq!(app.ui.crop_rect, Some([20.0, 20.0, 100.0, 60.0]), "and back");
        // A ratio preset swaps with the frame, as the options bar's ⇄ does.
        app.ui.tool_options.crop_ratio = "16:9".into();
        assert!(swap_orientation(&mut app));
        assert_eq!(app.ui.tool_options.crop_ratio, "9:16");
        app.ui.tool_options.crop_ratio = "original".into();
        assert!(swap_orientation(&mut app));
        assert_eq!(app.ui.tool_options.crop_ratio, "100:200", "the document's size on its side");
        app.ui.tool_options.crop_ratio = String::new();
        assert!(swap_orientation(&mut app));
        assert_eq!(app.ui.tool_options.crop_ratio, "", "no ratio stays none");
        assert_eq!(swapped_ratio("0:5", 1.0, 1.0), None);
        assert_eq!(swapped_ratio("nonsense", 1.0, 1.0), None);
        // No frame, or another tool: X is Switch Foreground and Background Colors.
        app.ui.crop_rect = None;
        key(&mut app, Key::X, NONE);
        assert_ne!(fg(&app), red, "swapped");
        app.ui.crop_rect = Some([20.0, 20.0, 100.0, 60.0]);
        app.ui.tool = Tool::Brush;
        key(&mut app, Key::X, NONE);
        assert_eq!(fg(&app), red, "swapped back");
        assert_eq!(app.ui.crop_rect, Some([20.0, 20.0, 100.0, 60.0]));
        // Huge or broken frames never panic and are never made non-finite.
        app.ui.tool = Tool::Crop;
        app.ui.crop_rect = Some([-f64::MAX, 0.0, f64::MAX, 1.0]);
        assert!(swap_orientation(&mut app));
        assert_eq!(app.ui.crop_rect, Some([-f64::MAX, 0.0, f64::MAX, 1.0]));
        app.ui.crop_rect = Some([f64::NAN, 0.0, 1.0, 1.0]);
        assert!(!swap_orientation(&mut app));
    }

    /// #1919: W × H beside the pointer while the frame is drawn or resized (its own size when
    /// turned), not while it moves or turns or between gestures.
    #[test]
    fn readout_shows_the_frame_size_while_drawing_or_resizing() {
        let mut app = app(SampleType::U8);
        let ev = |app: &mut PhotocraftApp, e: ToolEvent| tool_event(app, e, NONE);
        ev(&mut app, ToolEvent::Down { x: 10.0, y: 10.0, pressure: 1.0 });
        assert_eq!(sizing(&app), None, "nothing drawn yet");
        ev(&mut app, ToolEvent::Move { x: 70.4, y: 50.6, pressure: 1.0 });
        assert_eq!(sizing(&app), Some([60.0, 41.0]), "rounded as ↵ crops it");
        ev(&mut app, ToolEvent::Up { x: 70.4, y: 50.6 });
        assert_eq!(sizing(&app), None);
        // Resizing a turned frame by its right edge: the frame's own width grows.
        app.ui.crop_rect = Some([20.0, 20.0, 100.0, 60.0]);
        app.ui.crop_angle = 30.0;
        let grab = turn([100.0, 40.0], [60.0, 40.0], 30.0);
        let to = [grab[0] + 10.0 * 30f64.to_radians().cos(), grab[1] + 10.0 * 30f64.to_radians().sin()];
        ev(&mut app, ToolEvent::Down { x: grab[0], y: grab[1], pressure: 1.0 });
        ev(&mut app, ToolEvent::Move { x: to[0], y: to[1], pressure: 1.0 });
        assert!(matches!(app.crop.drag, Some(CropDrag::Resize { .. })));
        assert_eq!(sizing(&app), Some([90.0, 40.0]));
        ev(&mut app, ToolEvent::Up { x: to[0], y: to[1] });
        // Moving shows no size.
        ev(&mut app, ToolEvent::Down { x: 60.0, y: 40.0, pressure: 1.0 });
        ev(&mut app, ToolEvent::Move { x: 62.0, y: 40.0, pressure: 1.0 });
        assert!(matches!(app.crop.drag, Some(CropDrag::Move { .. })));
        assert_eq!(sizing(&app), None);
        ev(&mut app, ToolEvent::Up { x: 62.0, y: 40.0 });
        assert_eq!(frame_size([0.0, 0.0, 0.2, 0.4]), Some([1.0, 1.0]));
        assert_eq!(frame_size([f64::NAN, 0.0, 1.0, 1.0]), None);
        assert_eq!(frame_size([-f64::MAX, 0.0, f64::MAX, 1.0]), None);
    }
}
