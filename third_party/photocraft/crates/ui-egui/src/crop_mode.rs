//! The Crop tool's two modes (Photoshop's gear menu, "Set additional Crop options", #1919).
//!
//! - **Default mode** (Photoshop's default, Use Classic Mode off): the crop box stays upright on
//!   screen and the image turns behind it. A drag outside the box turns the image about the box's
//!   centre (⇧ in 15° steps; the readout beside the pointer is the image's turn, clockwise
//!   positive), a drag inside moves the image under the box (the box stays put), the arrow keys
//!   nudge the image, and the handles resize the box on screen as usual. With **Auto Center
//!   Preview** on (the default), a box being resized keeps its centre where it is on screen (also
//!   after a Space pan) while the handle follows the pointer, a drawn box moves to the middle of
//!   the canvas on release, and X re-centres it. A drag inside never re-centres the box
//!   (Photoshop 25.1).
//! - **Classic Mode** (P toggles it while the Crop tool shows a box): the box moves and turns over
//!   a fixed image (`crop_ui`). Show Cropped Area and Auto Center Preview are greyed out in it, and
//!   the area outside the box shows whatever Show Cropped Area says (Photoshop 25.1).
//!
//! The crop model is the same in both: `UiState::crop_rect` is the frame in document px and
//! `UiState::crop_angle` its turn, so ↵ commits the same `image.crop {…, angle}` (one undo step)
//! whichever mode drew it. Default mode only changes the view: while the Crop tool shows a frame,
//! the view's camera rotation is the frame's turn undone (`View::rotation = -crop_angle`), so the
//! box shows upright. In Classic Mode the camera is upright (rotation 0) and the frame shows with
//! its own turn on the image. As in Photoshop 25.1, picking the tool in either mode turns a Rotate
//! View (R) turn upright at once, about the view's centre (the frame stays on the same image
//! area), and the view stays upright after ↵ or Esc and when P switches modes (the R turn isn't
//! restored).
//!
//! Both options live in `ToolOptions::crop_shield` (`classic_mode`, `auto_center_preview`), which
//! `ui.inspect` reports and `ui.set {cropShield: {...}}` patches.

use egui::{Rect, vec2};

use crate::PhotocraftApp;
use crate::canvas::ViewXform;
use crate::state::Tool;

/// Use Classic Mode is on: the box turns and moves over the image.
pub fn classic(app: &PhotocraftApp) -> bool {
    app.ui.tool_options.crop_shield.classic_mode
}

fn finite_frame(app: &PhotocraftApp) -> Option<[f64; 4]> {
    app.ui.crop_rect.filter(|r| r.iter().all(|v| v.is_finite()))
}

/// The Crop tool drives the view: it shows a frame (either mode; [`sync`]).
pub fn turns_view(app: &PhotocraftApp) -> bool {
    app.ui.tool == Tool::Crop && finite_frame(app).is_some() && app.session.active().is_some()
}

/// Gestures move and turn the image rather than the box (default mode with the Crop tool).
pub fn moves_image(app: &PhotocraftApp) -> bool {
    app.ui.tool == Tool::Crop && !classic(app)
}

/// The view centre that keeps document point `c` where it is on screen when the camera rotation
/// goes from `from` to `to` degrees (zoom and flip unchanged). `center` itself when anything isn't
/// finite.
pub fn pivot_center(center: [f32; 2], c: [f64; 2], from: f32, to: f32, flip: bool) -> [f32; 2] {
    let a = ViewXform { rect: Rect::ZERO, zoom: 1.0, center, flip, rotation: from, aspect: 1.0 };
    // c's offset from the view centre on screen (at zoom 1), then the same offset in the new view.
    let off = a.to_screen(c[0] as f32, c[1] as f32).to_vec2();
    let u = ViewXform { rotation: to, ..a }.unmap_vec(off);
    let n = [c[0] as f32 - u.x, c[1] as f32 - u.y];
    if n.iter().all(|v| v.is_finite()) { n } else { center }
}

/// The active document's view rotation in degrees (0 without a view).
pub fn view_rotation(app: &PhotocraftApp) -> f32 {
    app.session.active_index().and_then(|i| app.ui.views.get(i)).map_or(0.0, |v| v.rotation)
}

/// Turns the active view so the frame shows upright while default mode drives it, pivoting about
/// document point `pivot` so that point stays put on screen (the view's own centre when `None`:
/// picking the tool, a commit or a cancel only straighten the view).
pub fn sync(app: &mut PhotocraftApp, pivot: Option<[f64; 2]>) {
    if !turns_view(app) {
        return;
    }
    let Some(i) = app.session.active_index() else { return };
    // Classic Mode: an upright camera, the frame turned on it. Default mode: the frame upright.
    let want = if classic(app) { 0.0 } else { crate::rotate_view::wrap_deg(-crate::crop_ui::angle(app) as f32) };
    let flip = app.ui.view.flip_horizontal;
    let Some(v) = app.ui.views.get_mut(i) else { return };
    if v.rotation != want {
        if let Some(p) = pivot.filter(|p| p.iter().all(|v| v.is_finite())) {
            v.center = pivot_center(v.center, p, v.rotation, want, flip);
        }
        v.rotation = want;
    }
}

/// [`sync`] about the frame's centre: the box stays put on screen while the image turns.
pub fn sync_frame(app: &mut PhotocraftApp) {
    let c = finite_frame(app).map(crate::crop_ui::center);
    sync(app, c);
}

/// A commit or cancel in default mode: the view is left upright (Photoshop 25.1 doesn't bring
/// back a Rotate View turn).
pub fn upright(app: &mut PhotocraftApp) {
    if app.ui.tool != Tool::Crop {
        return;
    }
    if let Some(v) = app.session.active_index().and_then(|i| app.ui.views.get_mut(i)) {
        v.rotation = 0.0;
    }
}

/// The image's turn the default mode shows beside the pointer: clockwise on screen is positive,
/// the frame's angle on the image undone.
pub fn image_turn(frame_angle: f64) -> f64 {
    let a = crate::crop_ui::normalized_angle(-frame_angle);
    // Never "-0.0°".
    if a == 0.0 { 0.0 } else { a }
}

/// A live-centred resize ([`live_resize`]): the frame `rect` and handle (`hx`, `hy`) at the press,
/// grabbed at document point `start`; the pointer now at `q` (in the press view); the frame's turn
/// `deg`, the aspect `ratio` to hold and ⌥ (resize about the centre).
#[derive(Clone, Copy, Debug)]
pub struct LiveResize {
    pub rect: [f64; 4],
    pub deg: f64,
    pub hx: i8,
    pub hy: i8,
    pub start: [f64; 2],
    pub q: [f64; 2],
    pub ratio: Option<f64>,
    pub alt: bool,
}

/// Auto Center Preview during a handle drag: the box stays centred on screen where it was at the
/// press and grows or shrinks symmetrically there, so the handle stays under the pointer, while on
/// the image the opposite edge stays put (the centre with ⌥), as in a plain resize. Everything is
/// computed from the press, so the view following the frame can't feed back into the drag.
/// Returns the proposed frame and the camera's document-space offset without changing the app.
pub(crate) fn resized_frame(g: LiveResize) -> Option<([f64; 4], [f64; 2])> {
    let LiveResize { rect, deg, hx, hy, start, q, ratio, alt } = g;
    let c0 = crate::crop_ui::center(rect);
    let (w0, h0) = (rect[2] - rect[0], rect[3] - rect[1]);
    // The pointer's offset from where it grabbed the handle, in the frame's own axes; the handle
    // moves by it on screen, so each half extent grows by it.
    let d = crate::crop_ui::turn([q[0] - start[0], q[1] - start[1]], [0.0, 0.0], -deg);
    let local = [-w0 / 2.0, -h0 / 2.0, w0 / 2.0, h0 / 2.0];
    let n = crate::crop_ui::resized(local, hx, hy, d, ratio, true);
    let (w1, h1) = (n[2] - n[0], n[3] - n[1]);
    // On the image the opposite edge stays put (with ⌥ the centre), so the centre moves.
    let shift = if alt { [0.0, 0.0] } else { [f64::from(hx) * (w1 - w0) / 2.0, f64::from(hy) * (h1 - h0) / 2.0] };
    let s = crate::crop_ui::turn(shift, [0.0, 0.0], deg);
    let c1 = [c0[0] + s[0], c0[1] + s[1]];
    let f = [c1[0] - w1 / 2.0, c1[1] - h1 / 2.0, c1[0] + w1 / 2.0, c1[1] + h1 / 2.0];
    f.iter().all(|v| v.is_finite()).then_some((f, s))
}

/// Move the camera with the resized frame to keep the crop box's centre stationary on screen.
pub fn live_resize(app: &mut PhotocraftApp, g: LiveResize, press: PressView) {
    let Some((f, s)) = resized_frame(g) else { return };
    app.ui.crop_rect = Some(f);
    // The view follows the centre, so the box stays where it was on screen.
    if let Some(v) = app.session.active_index().and_then(|i| app.ui.views.get_mut(i)) {
        let c = [press.center[0] + s[0] as f32, press.center[1] + s[1] as f32];
        if c.iter().all(|x| x.is_finite()) {
            v.center = c;
        }
    }
}

/// Auto Center Preview while a handle is dragged in default mode.
pub fn live_center(app: &PhotocraftApp) -> bool {
    moves_image(app) && app.ui.tool_options.crop_shield.auto_center_preview
}

/// Moves the image under the box by document vector `d` as the view shows it: the frame moves by
/// `-d` in the document and the view with it, so the box stays put on screen.
pub fn shift_image(app: &mut PhotocraftApp, d: [f64; 2]) {
    let Some(r) = finite_frame(app) else { return };
    if !(d[0].is_finite() && d[1].is_finite()) {
        return;
    }
    let n = [r[0] - d[0], r[1] - d[1], r[2] - d[0], r[3] - d[1]];
    if !n.iter().all(|v| v.is_finite()) {
        return;
    }
    app.ui.crop_rect = Some(n);
    if let Some(v) = app.session.active_index().and_then(|i| app.ui.views.get_mut(i)) {
        let c = [v.center[0] - d[0] as f32, v.center[1] - d[1] as f32];
        if c.iter().all(|x| x.is_finite()) {
            v.center = c;
        }
    }
}

/// The active view's camera (rotation and centre) when a default-mode drag began: the drag reads
/// its points in it, so it doesn't matter that the view turns and pans with the image meanwhile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PressView {
    rotation: f32,
    center: [f32; 2],
}

/// The active view's camera now.
pub fn press_view(app: &PhotocraftApp) -> Option<PressView> {
    app.session.active_index().and_then(|i| app.ui.views.get(i)).map(|v| PressView { rotation: v.rotation, center: v.center })
}

/// Document point `p` as seen through camera `from` re-read through camera `to`: the point that
/// `to` shows at the screen position where `from` shows `p` (zoom and flip are the same). `p`
/// itself when anything isn't finite.
fn remap(p: [f64; 2], from: PressView, to: PressView, flip: bool) -> [f64; 2] {
    let at = |v: PressView| ViewXform { rect: Rect::ZERO, zoom: 1.0, center: v.center, flip, rotation: v.rotation, aspect: 1.0 };
    let q = at(to).to_doc(at(from).to_screen(p[0] as f32, p[1] as f32));
    if q.iter().all(|v| v.is_finite()) { q } else { p }
}

/// A pointer point `p` the canvas mapped through the view as it is now, read in the press-time
/// view `press` (unchanged without one).
pub fn to_press(app: &PhotocraftApp, press: Option<PressView>, p: [f64; 2]) -> [f64; 2] {
    match (press, press_view(app)) {
        (Some(press), Some(now)) => remap(p, now, press, app.ui.view.flip_horizontal),
        _ => p,
    }
}

/// A press-view point mapped back into the current camera after snapping the crop handle.
pub(crate) fn from_press(app: &PhotocraftApp, press: PressView, p: [f64; 2]) -> [f64; 2] {
    match press_view(app) {
        Some(now) => remap(p, press, now, app.ui.view.flip_horizontal),
        None => p,
    }
}

/// The control channel's `ui.pointer` gives document points as the view showed them at the press
/// (it can't know how the view turns and pans with the image): during a default-mode drag, the
/// point the canvas would report for the same screen position now.
pub fn control_point(app: &PhotocraftApp, p: [f64; 2]) -> [f64; 2] {
    let press = match app.crop.drag {
        Some(
            crate::crop_ui::CropDrag::MoveImage { view, .. } | crate::crop_ui::CropDrag::TurnImage { view, .. } | crate::crop_ui::CropDrag::Resize { view, .. },
        ) => view,
        _ => None,
    };
    match (press, press_view(app)) {
        (Some(press), Some(now)) if app.ui.tool == Tool::Crop => remap(p, press, now, app.ui.view.flip_horizontal),
        _ => p,
    }
}

/// A default-mode drag inside the box: the image has moved by document vector `d` (in the press
/// view) since the press, when the frame was `rect` and the view `press`. The frame moves by `-d`
/// on the image and the view's centre with it, so the box stays put on screen.
pub fn offset_image(app: &mut PhotocraftApp, rect: [f64; 4], press: Option<PressView>, d: [f64; 2]) {
    let n = [rect[0] - d[0], rect[1] - d[1], rect[2] - d[0], rect[3] - d[1]];
    if !n.iter().all(|v| v.is_finite()) {
        return;
    }
    app.ui.crop_rect = Some(n);
    let (Some(press), Some(v)) = (press, app.session.active_index().and_then(|i| app.ui.views.get_mut(i))) else { return };
    let c = [press.center[0] - d[0] as f32, press.center[1] - d[1] as f32];
    if c.iter().all(|x| x.is_finite()) {
        v.center = c;
    }
}

/// A screen vector `s` (points, at the current zoom ignored) as a document vector of the same
/// length: undoes the view's flip and rotation.
pub fn screen_to_doc_vec(app: &PhotocraftApp, s: [f64; 2]) -> [f64; 2] {
    let xf = ViewXform { rect: Rect::ZERO, zoom: 1.0, center: [0.0, 0.0], flip: app.ui.view.flip_horizontal, rotation: view_rotation(app), aspect: 1.0 };
    let u = xf.unmap_vec(vec2(s[0] as f32, s[1] as f32));
    [f64::from(u.x), f64::from(u.y)]
}

/// Auto Center Preview: in default mode, pans the view so the box sits in the middle of the canvas
/// (the view turns about its centre, so the box stays upright).
pub fn auto_center(app: &mut PhotocraftApp) {
    if !turns_view(app) || classic(app) || !app.ui.tool_options.crop_shield.auto_center_preview {
        return;
    }
    let Some(c) = finite_frame(app).map(crate::crop_ui::center) else { return };
    let c = [c[0] as f32, c[1] as f32];
    if let (Some(v), true) = (app.session.active_index().and_then(|i| app.ui.views.get_mut(i)), c.iter().all(|x| x.is_finite())) {
        v.center = c;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::{ToolEvent, tool_event};
    use crate::crop_ui::{CropDrag, corners};
    use egui::{Modifiers, Pos2, pos2};
    use photocraft_doc::{Color, ColorMode, Document, SampleType, Size};
    use serde_json::json;

    const NONE: Modifiers = Modifiers::NONE;

    /// A 200 × 100 document in a 800 × 600 canvas at 100 %, centred, the Crop tool in default mode.
    fn app() -> PhotocraftApp {
        let doc = Document::with_background("crop", Size::new(200, 100), ColorMode::Rgb, SampleType::U8, Color::WHITE);
        let mut s = photocraft_engine::Session::new();
        s.add_document(doc, None);
        let mut app = PhotocraftApp::new(s, crate::Services::default());
        app.sync_views();
        app.ui.tool = Tool::Crop;
        app.ui.extras.snap = false;
        app.ui.extras.rulers = false;
        app.ui.view.show.smart_guides = false;
        app.last_canvas_rect = Rect::from_min_size(Pos2::ZERO, vec2(800.0, 600.0));
        let v = &mut app.ui.views[0];
        (v.zoom, v.center, v.fit_pending, v.rotation) = (1.0, [100.0, 50.0], false, 0.0);
        assert!(!classic(&app), "Photoshop's default mode is the default");
        app
    }

    fn xf(app: &PhotocraftApp) -> ViewXform {
        ViewXform::active(app).unwrap()
    }

    /// A drag through screen points, each mapped through the view as it is at that moment (as the
    /// canvas does every frame).
    fn screen_drag(app: &mut PhotocraftApp, pts: &[Pos2], mods: Modifiers) {
        let at = |app: &PhotocraftApp, p: Pos2| xf(app).to_doc(p);
        let d = at(app, pts[0]);
        tool_event(app, ToolEvent::Down { x: d[0], y: d[1], pressure: 1.0 }, mods);
        for p in &pts[1..] {
            let d = at(app, *p);
            tool_event(app, ToolEvent::Move { x: d[0], y: d[1], pressure: 1.0 }, mods);
        }
        let d = at(app, pts[pts.len() - 1]);
        tool_event(app, ToolEvent::Up { x: d[0], y: d[1] }, mods);
    }

    fn screen_box(app: &PhotocraftApp) -> [Pos2; 4] {
        let x = xf(app);
        corners(app.ui.crop_rect.unwrap(), crate::crop_ui::angle(app)).map(|p| x.to_screen(p[0] as f32, p[1] as f32))
    }

    fn assert_upright(q: [Pos2; 4]) {
        let ok = (q[0].y - q[1].y).abs() < 0.01 && (q[2].y - q[3].y).abs() < 0.01 && (q[0].x - q[3].x).abs() < 0.01 && (q[1].x - q[2].x).abs() < 0.01;
        assert!(ok && q[1].x > q[0].x && q[3].y > q[0].y, "not an upright box on screen: {q:?}");
    }

    fn close(a: [Pos2; 4], b: [Pos2; 4]) -> bool {
        a.iter().zip(b).all(|(p, q)| p.distance(q) < 0.02)
    }

    /// A real (not default) frame [40, 30, 100, 70], centred at (70, 50) = screen (370, 300).
    fn with_frame(app: &mut PhotocraftApp) {
        crate::crop_ui::ensure_frame(app);
        app.ui.crop_rect = Some([40.0, 30.0, 100.0, 70.0]);
        app.crop.default_frame = false;
    }

    #[test]
    fn a_live_resize_snaps_the_handle_without_feedback_from_the_camera() {
        for (hx, hy) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)] {
            for alt in [false, true] {
                for (zoom, ppp, flip, ratio) in [(1.0, 1.0, false, false), (0.5, 1.0, false, true), (3.0, 2.0, false, false), (3.0, 2.0, true, true)] {
                    let mut app = app();
                    app.ppp = ppp;
                    app.ui.views[0].zoom = zoom;
                    app.ui.view.flip_horizontal = flip;
                    if ratio {
                        app.ui.tool_options.crop_ratio = "original".into();
                    }
                    app.ui.extras.snap = true;
                    crate::crop_ui::ensure_frame(&mut app);
                    let box0 = screen_box(&app);
                    let c = mid_of(box0);
                    let from = xf(&app).to_screen(100.0 + hx as f32 * 100.0, 50.0 + hy as f32 * 50.0);
                    let direction = (from - c).normalized();
                    let mods = if alt { Modifiers::ALT } else { NONE };
                    let d = xf(&app).to_doc(from);
                    tool_event(&mut app, ToolEvent::Down { x: d[0], y: d[1], pressure: 1.0 }, mods);
                    // Leave the snap radius, then return just inside it. A still pointer must
                    // settle on the canvas edge instead of alternating on opposite sides of it.
                    for offset in [20.0, -2.0, -2.0, -2.0, -2.0, -2.0] {
                        let p = from + direction * offset;
                        let d = xf(&app).to_doc(p);
                        tool_event(&mut app, ToolEvent::Move { x: d[0], y: d[1], pressure: 1.0 }, mods);
                        if offset == -2.0 {
                            assert!(close(screen_box(&app), box0), "handle ({hx}, {hy}), Alt {alt}: {:?}", app.ui.crop_rect);
                        }
                    }
                    let d = xf(&app).to_doc(from + direction * -2.0);
                    tool_event(&mut app, ToolEvent::Up { x: d[0], y: d[1] }, mods);
                    assert!(close(screen_box(&app), box0), "release: {:?}", app.ui.crop_rect);
                }
            }
        }
    }

    #[test]
    fn dragging_outside_turns_the_image_and_the_box_stays_upright() {
        let mut app = app();
        with_frame(&mut app);
        let before = screen_box(&app);
        // From straight right of the box centre to straight below it: a quarter turn clockwise
        // on screen, so the image turns clockwise and the frame turns -90° on it.
        screen_drag(&mut app, &[pos2(450.0, 300.0), pos2(440.0, 320.0), pos2(370.0, 380.0)], NONE);
        assert!((crate::crop_ui::angle(&app) - -90.0).abs() < 0.01, "{}", app.ui.crop_angle);
        assert!((view_rotation(&app) - 90.0).abs() < 0.01, "the view turned with it");
        assert_upright(screen_box(&app));
        assert!(close(screen_box(&app), before), "the box doesn't move on screen: {:?} vs {before:?}", screen_box(&app));
        assert_eq!(app.ui.crop_rect, Some([40.0, 30.0, 100.0, 70.0]), "turning keeps the frame");
        // ⇧ steps of 15°, still upright.
        let a = 22.0f32.to_radians();
        screen_drag(&mut app, &[pos2(450.0, 300.0), pos2(370.0 + 80.0 * a.cos(), 300.0 + 80.0 * a.sin())], Modifiers::SHIFT);
        assert!((crate::crop_ui::angle(&app) - -105.0).abs() < 0.01, "{}", app.ui.crop_angle);
        assert_upright(screen_box(&app));
        assert!(close(screen_box(&app), before));
        assert!(crate::crop_ui::turning(&app).is_none());
    }

    /// Even with Auto Center Preview, a drag inside an off-centre box (after a pan, say) leaves the
    /// box where it is on screen: only the image moves (Photoshop 25.1).
    #[test]
    fn a_move_never_recentres_the_box() {
        let mut app = app();
        assert!(app.ui.tool_options.crop_shield.auto_center_preview);
        with_frame(&mut app);
        assert!(mid_of(screen_box(&app)).distance(pos2(370.0, 300.0)) < 0.01);
        screen_drag(&mut app, &[pos2(370.0, 300.0), pos2(380.0, 300.0), pos2(390.0, 300.0)], NONE);
        assert!(mid_of(screen_box(&app)).distance(pos2(370.0, 300.0)) < 0.01, "{:?}", mid_of(screen_box(&app)));
        let r = app.ui.crop_rect.unwrap();
        assert!(r.iter().zip([20.0, 30.0, 80.0, 70.0]).all(|(a, b)| (a - b).abs() < 1e-3), "{r:?}");
    }

    fn mid_of(q: [Pos2; 4]) -> Pos2 {
        pos2((q[0].x + q[2].x) / 2.0, (q[0].y + q[2].y) / 2.0)
    }

    #[test]
    fn dragging_inside_moves_the_image_under_the_box() {
        let mut app = app();
        with_frame(&mut app);
        let before = screen_box(&app);
        screen_drag(&mut app, &[pos2(370.0, 300.0), pos2(380.0, 305.0), pos2(390.0, 310.0)], NONE);
        // The image followed the pointer 20, 10 to the right and down: the frame lies that much
        // further up-left on it.
        let r = app.ui.crop_rect.unwrap();
        assert!(r.iter().zip([20.0, 20.0, 80.0, 60.0]).all(|(a, b)| (a - b).abs() < 1e-3), "{r:?}");
        assert!(close(screen_box(&app), before), "the box stays put on screen");
        // Turned 30° first: the move follows the pointer on screen, along the turned image.
        screen_drag(&mut app, &[pos2(450.0, 300.0), pos2(370.0 + 80.0 * 30f32.to_radians().cos(), 300.0 + 80.0 * 30f32.to_radians().sin())], NONE);
        assert!((crate::crop_ui::angle(&app) - -30.0).abs() < 0.01);
        let (before, c0) = (screen_box(&app), crate::crop_ui::center(app.ui.crop_rect.unwrap()));
        let img = |app: &PhotocraftApp| xf(app).to_screen(0.0, 0.0);
        let origin = img(&app);
        screen_drag(&mut app, &[pos2(370.0, 300.0), pos2(400.0, 300.0)], NONE);
        assert!(close(screen_box(&app), before), "box put");
        assert!((img(&app) - origin - vec2(30.0, 0.0)).length() < 0.05, "the image moved 30 right on screen");
        let c1 = crate::crop_ui::center(app.ui.crop_rect.unwrap());
        assert!(((c1[0] - c0[0]).hypot(c1[1] - c0[1]) - 30.0).abs() < 1e-3);
    }

    #[test]
    fn the_readout_is_the_images_turn_clockwise_positive() {
        let mut app = app();
        with_frame(&mut app);
        // A 30° clockwise sweep on screen, held mid-drag.
        let d = |app: &PhotocraftApp, p: Pos2| xf(app).to_doc(p);
        let a = 30f32.to_radians();
        let (p0, p1) = (pos2(450.0, 300.0), pos2(370.0 + 80.0 * a.cos(), 300.0 + 80.0 * a.sin()));
        let s = d(&app, p0);
        tool_event(&mut app, ToolEvent::Down { x: s[0], y: s[1], pressure: 1.0 }, NONE);
        let m = d(&app, p1);
        tool_event(&mut app, ToolEvent::Move { x: m[0], y: m[1], pressure: 1.0 }, NONE);
        let shown = crate::crop_ui::turning(&app).unwrap();
        assert!((shown - 30.0).abs() < 0.01, "clockwise is positive: {shown}");
        assert!((crate::crop_ui::angle(&app) - -30.0).abs() < 0.01, "the frame turns the other way on the image");
        let m = d(&app, p1);
        tool_event(&mut app, ToolEvent::Up { x: m[0], y: m[1] }, NONE);
        assert_eq!(image_turn(0.0).to_string(), "0", "never -0");
        // Classic Mode shows the frame's own turn, clockwise positive as before.
        let mut app = self::app();
        app.ui.tool_options.crop_shield.classic_mode = true;
        with_frame(&mut app);
        let s = d(&app, p0);
        tool_event(&mut app, ToolEvent::Down { x: s[0], y: s[1], pressure: 1.0 }, NONE);
        let m = d(&app, p1);
        tool_event(&mut app, ToolEvent::Move { x: m[0], y: m[1], pressure: 1.0 }, NONE);
        assert!((crate::crop_ui::turning(&app).unwrap() - 30.0).abs() < 0.01);
    }

    #[test]
    fn auto_center_keeps_the_box_centred_while_a_handle_is_dragged() {
        for auto in [true, false] {
            let mut app = app();
            app.ui.tool_options.crop_shield.auto_center_preview = auto;
            with_frame(&mut app);
            // Turn 90° clockwise on screen first: the frame is at -90° in the image.
            screen_drag(&mut app, &[pos2(450.0, 300.0), pos2(370.0, 380.0)], NONE);
            let q = screen_box(&app);
            assert_upright(q);
            let mid = |q: [Pos2; 4]| pos2((q[0].x + q[2].x) / 2.0, (q[0].y + q[2].y) / 2.0);
            // The box's right edge on screen, dragged 10 then 20 further right.
            let right = pos2(q[1].x, (q[1].y + q[2].y) / 2.0);
            assert_eq!(crate::crop_ui::cursor(&app, xf(&app).to_doc(right)), Some(egui::CursorIcon::ResizeHorizontal), "screen-space arrow");
            let at = |app: &PhotocraftApp, p: Pos2| xf(app).to_doc(p);
            let s = at(&app, right);
            tool_event(&mut app, ToolEvent::Down { x: s[0], y: s[1], pressure: 1.0 }, NONE);
            for dx in [10.0, 20.0] {
                let m = at(&app, right + vec2(dx, 0.0));
                tool_event(&mut app, ToolEvent::Move { x: m[0], y: m[1], pressure: 1.0 }, NONE);
                let n = screen_box(&app);
                assert_upright(n);
                assert!((n[1].x - (right.x + dx)).abs() < 0.02, "auto={auto} dx={dx}: the handle follows the pointer: {n:?}");
                if auto {
                    assert!(mid(n).distance(mid(q)) < 0.02, "dx={dx}: live, the box stays centred: {:?} vs {:?}", mid(n), mid(q));
                    assert!(((n[1].x - n[0].x) - (q[1].x - q[0].x) - 2.0 * dx).abs() < 0.02, "it grows on both sides");
                } else {
                    assert!((n[0].x - q[0].x).abs() < 0.02, "the left edge stays put");
                }
                assert!(((n[3].y - n[0].y) - (q[3].y - q[0].y)).abs() < 0.02);
            }
            let m = at(&app, right + vec2(20.0, 0.0));
            tool_event(&mut app, ToolEvent::Up { x: m[0], y: m[1] }, NONE);
            let n = screen_box(&app);
            if auto {
                assert!(mid(n).distance(mid(q)) < 0.05, "on release it stays where its centre was: {:?} vs {:?}", mid(n), mid(q));
                // On the image, the opposite edge stayed put: the frame grew 40 px from it.
                let r = app.ui.crop_rect.unwrap();
                assert!(((r[2] - r[0]) - 100.0).abs() < 1e-3 && ((r[3] - r[1]) - 40.0).abs() < 1e-3, "{r:?}");
                let left = corners(r, crate::crop_ui::angle(&app))[0];
                let was = corners([40.0, 30.0, 100.0, 70.0], -90.0)[0];
                assert!((left[0] - was[0]).abs() < 1e-3 && (left[1] - was[1]).abs() < 1e-3, "{left:?} vs {was:?}");
            } else {
                assert!(mid(n).distance(pos2(380.0, 300.0)) < 0.05, "{:?}", mid(n));
            }
        }
        // ⌥: about the centre, and a collapsed drag puts frame and view back.
        let mut app = app();
        with_frame(&mut app);
        let q = screen_box(&app);
        let center = app.ui.views[0].center;
        let right = pos2(q[1].x, (q[1].y + q[2].y) / 2.0);
        screen_drag(&mut app, &[right, right + vec2(10.0, 0.0)], Modifiers::ALT);
        let r = app.ui.crop_rect.unwrap();
        assert!(((r[0] + r[2]) / 2.0 - 70.0).abs() < 1e-3 && ((r[2] - r[0]) - 80.0).abs() < 1e-3, "{r:?}");
        let mut app = self::app();
        with_frame(&mut app);
        screen_drag(&mut app, &[right, pos2(370.0, 300.0)], NONE);
        assert_eq!(app.ui.crop_rect, Some([40.0, 30.0, 100.0, 70.0]), "too small: the frame is back");
        assert_eq!(app.ui.views[0].center, center, "and the view");
    }

    /// The same frame and angle commit the same crop in both modes, and in both the view stays
    /// upright after a commit or a cancel (Photoshop 25.1 doesn't restore a Rotate View turn).
    #[test]
    fn commit_is_the_same_in_both_modes_and_the_view_stays_upright() {
        let mut out = Vec::new();
        for classic in [true, false] {
            let mut app = app();
            app.run("select.rect", json!({"x": 70, "y": 50, "width": 2, "height": 2})).unwrap();
            app.run("edit.fill", json!({"color": "#ff0000"})).unwrap();
            app.run("select.deselect", json!({})).unwrap();
            app.ui.tool_options.crop_shield.classic_mode = classic;
            app.ui.views[0].rotation = 20.0;
            with_frame(&mut app);
            app.ui.crop_angle = 30.0;
            app.crop.editing = true;
            crate::crop_ui::ensure_frame(&mut app);
            let want = if classic { 0.0 } else { -30.0 };
            assert!((view_rotation(&app) - want).abs() < 1e-4, "classic={classic}: {}", view_rotation(&app));
            if !classic {
                assert_upright(screen_box(&app));
            }
            let past = app.session.active().unwrap().history.past_len();
            crate::canvas::commit_crop(&mut app);
            assert_eq!(view_rotation(&app), 0.0, "classic={classic}: after ↵");
            crate::crop_ui::ensure_frame(&mut app);
            assert_eq!(view_rotation(&app), 0.0, "classic={classic}: with the next frame");
            let st = app.session.active().unwrap();
            assert_eq!(st.history.past_len(), past + 1, "one undo step");
            let surf = st.doc.layers[0].surface().unwrap();
            let px: Vec<[f32; 4]> =
                (0..st.doc.size.width as i32).flat_map(|x| (0..st.doc.size.height as i32).map(move |y| (x, y))).map(|(x, y)| surf.rgba(x, y)).collect();
            out.push((st.doc.size, px));
            // Esc.
            let mut app = self::app();
            app.ui.tool_options.crop_shield.classic_mode = classic;
            app.ui.views[0].rotation = -45.0;
            with_frame(&mut app);
            screen_drag(&mut app, &[pos2(500.0, 300.0), pos2(370.0, 420.0)], NONE);
            crate::crop_ui::cancel(&mut app);
            crate::crop_ui::ensure_frame(&mut app);
            assert_eq!(view_rotation(&app), 0.0, "classic={classic}: Esc");
        }
        assert_eq!(out[0].0, out[1].0, "same size");
        assert!(out[0].1 == out[1].1, "same pixels");
    }

    #[test]
    fn picking_the_crop_tool_turns_a_rotated_view_upright() {
        let mut app = app();
        app.ui.tool = Tool::Brush;
        app.ui.views[0].rotation = 30.0;
        let center = app.ui.views[0].center;
        app.ui.tool = Tool::Crop;
        crate::crop_ui::ensure_frame(&mut app);
        assert!(app.crop.default_frame && !app.crop.editing, "no click needed");
        assert_eq!(view_rotation(&app), 0.0);
        assert_eq!(app.ui.views[0].center, center, "about the view's centre");
        // Leaving the tool doesn't bring the turn back.
        app.ui.tool = Tool::Brush;
        crate::crop_ui::ensure_frame(&mut app);
        assert_eq!(view_rotation(&app), 0.0);
        // Classic Mode too (Photoshop 25.1): the camera goes upright about the view's centre and
        // the frame stays on the same image area, keeping its own turn.
        let mut app = self::app();
        app.ui.tool_options.crop_shield.classic_mode = true;
        app.ui.tool = Tool::Brush;
        app.ui.views[0].rotation = 30.0;
        app.ui.crop_rect = Some([40.0, 30.0, 100.0, 70.0]);
        app.ui.crop_angle = 12.0;
        app.ui.tool = Tool::Crop;
        crate::crop_ui::ensure_frame(&mut app);
        assert_eq!(view_rotation(&app), 0.0);
        assert_eq!(app.ui.views[0].center, center, "about the view's centre");
        assert_eq!((app.ui.crop_rect, app.ui.crop_angle), (Some([40.0, 30.0, 100.0, 70.0]), 12.0), "the frame is unchanged");
        // P switches modes without bringing the R turn back: default mode turns the view with the
        // frame, Classic Mode puts the camera upright again.
        let press_p = |app: &mut PhotocraftApp| {
            let ctx = egui::Context::default();
            ctx.input_mut(|i| i.events.push(egui::Event::Key { key: egui::Key::P, physical_key: None, pressed: true, repeat: false, modifiers: NONE }));
            assert!(crate::crop_shield::keys(app, &ctx));
        };
        press_p(&mut app);
        assert!(!classic(&app));
        assert!((view_rotation(&app) - -12.0).abs() < 1e-4, "{}", view_rotation(&app));
        press_p(&mut app);
        assert!(classic(&app));
        assert_eq!(view_rotation(&app), 0.0);
        crate::crop_ui::cancel(&mut app);
        crate::crop_ui::ensure_frame(&mut app);
        assert_eq!(view_rotation(&app), 0.0, "Esc in Classic Mode: still upright");
    }

    fn key(app: &mut PhotocraftApp, key: egui::Key, modifiers: Modifiers) {
        let ctx = egui::Context::default();
        ctx.input_mut(|i| i.events.push(egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers }));
        crate::shortcuts::handle(app, &ctx);
    }

    #[test]
    fn arrows_nudge_the_image_shift_by_ten_and_x_recentres() {
        let mut app = app();
        with_frame(&mut app);
        app.crop.editing = true;
        app.ui.crop_angle = -90.0;
        crate::crop_ui::ensure_frame(&mut app);
        let before = screen_box(&app);
        let img = |app: &PhotocraftApp| xf(app).to_screen(0.0, 0.0);
        let origin = img(&app);
        key(&mut app, egui::Key::ArrowRight, NONE);
        assert!(close(screen_box(&app), before), "the box stays put");
        assert!((img(&app) - origin - vec2(1.0, 0.0)).length() < 0.02, "the image moved 1 px right on screen");
        key(&mut app, egui::Key::ArrowDown, Modifiers::SHIFT);
        assert!((img(&app) - origin - vec2(1.0, 10.0)).length() < 0.02, "⇧: 10 px");
        assert!(close(screen_box(&app), before));
        assert!(crate::crop_ui::swap_orientation(&mut app));
        assert_upright(screen_box(&app));
        let q = screen_box(&app);
        assert!(pos2((q[0].x + q[2].x) / 2.0, (q[0].y + q[2].y) / 2.0).distance(pos2(400.0, 300.0)) < 0.05, "Auto Center Preview");
    }

    /// Photoshop 25.1 greys out Show Cropped Area and Auto Center Preview in Classic Mode. Show
    /// Cropped Area then acts as on (an assumption) and H does nothing; the stored choice stays.
    #[test]
    fn classic_mode_shows_the_cropped_area_and_h_does_nothing() {
        let mut app = app();
        crate::crop_ui::ensure_frame(&mut app);
        app.ui.tool_options.crop_shield.show_cropped_area = false;
        assert!(crate::crop_shield::hides_outside(&app));
        app.ui.tool_options.crop_shield.classic_mode = true;
        assert!(!crate::crop_shield::hides_outside(&app));
        let canvas = egui::Color32::from_rgb(40, 40, 40);
        assert_ne!(crate::crop_shield::fill(&app.ui.tool_options.crop_shield, canvas, false), Some(canvas), "the shield, not the opaque canvas");
        key(&mut app, egui::Key::H, NONE);
        assert_eq!(app.ui.tool, Tool::Crop, "H is taken, not the Hand tool");
        assert!(!app.ui.tool_options.crop_shield.show_cropped_area, "and changes nothing");
        app.ui.tool_options.crop_shield.classic_mode = false;
        assert!(crate::crop_shield::hides_outside(&app), "the stored choice is back");
        key(&mut app, egui::Key::H, NONE);
        assert!(app.ui.tool_options.crop_shield.show_cropped_area);
    }

    #[test]
    fn p_toggles_classic_mode_while_the_crop_tool_shows_a_box() {
        let mut app = app();
        crate::crop_ui::ensure_frame(&mut app);
        let press = |app: &mut PhotocraftApp| {
            let ctx = egui::Context::default();
            ctx.input_mut(|i| i.events.push(egui::Event::Key { key: egui::Key::P, physical_key: None, pressed: true, repeat: false, modifiers: NONE }));
            crate::shortcuts::handle(app, &ctx);
        };
        press(&mut app);
        assert!(classic(&app) && app.ui.tool == Tool::Crop, "P doesn't pick the Pen tool");
        press(&mut app);
        assert!(!classic(&app));
        // Mid-drag P is left alone; without a box P is the Pen key.
        app.crop.drag = Some(CropDrag::Move { start: [0.0, 0.0], rect: [0.0, 0.0, 10.0, 10.0] });
        let ctx = egui::Context::default();
        ctx.input_mut(|i| i.events.push(egui::Event::Key { key: egui::Key::P, physical_key: None, pressed: true, repeat: false, modifiers: NONE }));
        assert!(!crate::crop_shield::keys(&mut app, &ctx));
        assert!(!classic(&app));
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.tool = Tool::Crop;
        press(&mut app);
        assert_ne!(app.ui.tool, Tool::Crop);
        assert!(!classic(&app));
    }

    #[test]
    fn straighten_turns_the_image_so_the_line_is_level_on_screen() {
        let mut app = app();
        crate::crop_ui::ensure_frame(&mut app);
        crate::crop_straighten::toggle(&mut app);
        // A horizon falling to the right, drawn on screen.
        let (a, b) = (pos2(320.0, 290.0), pos2(480.0, 310.0));
        let (da, db) = (xf(&app).to_doc(a), xf(&app).to_doc(b));
        screen_drag(&mut app, &[a, b], NONE);
        assert!(crate::crop_ui::angle(&app) > 5.0, "{}", app.ui.crop_angle);
        let (sa, sb) = (xf(&app).to_screen(da[0] as f32, da[1] as f32), xf(&app).to_screen(db[0] as f32, db[1] as f32));
        assert!((sa.y - sb.y).abs() < 0.01, "the line is level: {sa:?} {sb:?}");
        assert_upright(screen_box(&app));
    }

    /// `ui.pointer` points are read in the view at the press, so an `up` where the last `move` was
    /// doesn't move or turn the image a second time.
    #[test]
    fn ui_pointer_drives_the_default_mode_in_press_coordinates() {
        let mut app = app();
        app.ui.tool_options.crop_shield.auto_center_preview = false;
        with_frame(&mut app);
        let ctx = egui::Context::default();
        let pointer = |app: &mut PhotocraftApp, pts: [[f64; 2]; 2]| {
            let ev = json!([
                {"kind": "down", "x": pts[0][0], "y": pts[0][1]},
                {"kind": "move", "x": pts[1][0], "y": pts[1][1]},
                {"kind": "up", "x": pts[1][0], "y": pts[1][1]},
            ]);
            let (req, _rx) = crate::control::ControlRequest::new("ui.pointer", json!({"events": ev}));
            let crate::control::Outcome::Done(v) = crate::control::handle(app, &ctx, &req) else { panic!("ui.pointer didn't finish") };
            assert_eq!(v["ok"], true, "{v}");
        };
        pointer(&mut app, [[70.0, 50.0], [80.0, 55.0]]);
        let r = app.ui.crop_rect.unwrap();
        assert!(r.iter().zip([30.0, 25.0, 90.0, 65.0]).all(|(a, b)| (a - b).abs() < 1e-3), "{r:?}");
        assert!((app.ui.views[0].center[0] - 90.0).abs() < 1e-3 && (app.ui.views[0].center[1] - 45.0).abs() < 1e-3);
        // Outside: from right of the centre (60, 45) to below it, a quarter turn of the image.
        pointer(&mut app, [[140.0, 45.0], [60.0, 125.0]]);
        assert!((crate::crop_ui::angle(&app) - -90.0).abs() < 0.01, "{}", app.ui.crop_angle);
        assert_upright(screen_box(&app));
    }

    #[test]
    fn degenerate_input_never_panics() {
        let mut app = app();
        for frame in
            [[0.0, 0.0, 0.0, 0.0], [5.0, 5.0, 5.0, 5.0], [f64::NAN, 0.0, 10.0, 10.0], [f64::INFINITY, 0.0, 1e300, 1e300], [-1e300, -1e300, 1e300, 1e300]]
        {
            for angle in [0.0, f64::NAN, f64::INFINITY, 1e300, -720.5] {
                app.ui.crop_rect = Some(frame);
                app.ui.crop_angle = angle;
                app.crop.default_frame = false;
                app.crop.editing = true;
                crate::crop_ui::ensure_frame(&mut app);
                for p in [pos2(400.0, 300.0), pos2(10.0, 10.0), pos2(f32::NAN, 3.0), pos2(1e30, -1e30)] {
                    screen_drag(&mut app, &[p, p + vec2(15.0, 7.0)], NONE);
                    screen_drag(&mut app, &[p, p + vec2(15.0, 7.0)], Modifiers::SHIFT | Modifiers::ALT);
                }
                crate::crop_ui::nudge(&mut app, f64::NAN, 1.0);
                crate::crop_ui::nudge(&mut app, 1.0, 1.0);
                crate::crop_ui::swap_orientation(&mut app);
                auto_center(&mut app);
                shift_image(&mut app, [f64::NAN, f64::INFINITY]);
                let press = PressView { rotation: f32::NAN, center: [f32::INFINITY, 0.0] };
                let g = LiveResize { rect: frame, deg: angle, hx: 1, hy: -1, start: [0.0, f64::NAN], q: [1e300, -1e300], ratio: Some(f64::NAN), alt: false };
                live_resize(&mut app, g, press);
                assert!(app.ui.crop_rect.is_none_or(|r| r.iter().all(|v| v.is_finite()) || r == frame), "{frame:?} {angle}");
                let v = &app.ui.views[0];
                assert!(v.rotation.is_finite() && v.center.iter().all(|c| c.is_finite()), "{frame:?} {angle}: {v:?}");
                crate::crop_ui::cancel(&mut app);
            }
        }
        assert_eq!(pivot_center([1.0, 2.0], [f64::NAN, 0.0], 0.0, 90.0, false), [1.0, 2.0]);
        assert_eq!(pivot_center([1.0, 2.0], [0.0, 0.0], f32::NAN, 90.0, true), [1.0, 2.0]);
        // No view at all (no document).
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.tool = Tool::Crop;
        app.ui.crop_rect = Some([0.0, 0.0, 10.0, 10.0]);
        app.crop.editing = true;
        sync(&mut app, None);
        auto_center(&mut app);
        shift_image(&mut app, [1.0, 1.0]);
        upright(&mut app);
    }

    #[test]
    fn the_options_default_to_photoshops_and_older_settings_load() {
        let s = crate::crop_shield::CropShield::default();
        assert!(!s.classic_mode && s.auto_center_preview);
        let s: crate::crop_shield::CropShield = serde_json::from_str(r#"{"enabled": false}"#).unwrap();
        assert!(!s.classic_mode && s.auto_center_preview);
        let s: crate::crop_shield::CropShield = serde_json::from_str(r#"{"classic_mode": true, "auto_center_preview": false}"#).unwrap();
        assert!(s.classic_mode && !s.auto_center_preview);
    }

    #[test]
    fn pivoting_keeps_the_point_on_screen() {
        for flip in [false, true] {
            for (from, to) in [(0.0, 30.0), (45.0, -90.0), (170.0, -170.0)] {
                let center = [100.0f32, 50.0];
                let c = [70.0, 20.0];
                let a = ViewXform { rect: Rect::from_min_size(Pos2::ZERO, vec2(800.0, 600.0)), zoom: 2.0, center, flip, rotation: from, aspect: 1.0 };
                let n = pivot_center(center, c, from, to, flip);
                let b = ViewXform { center: n, rotation: to, ..a };
                let (p, q) = (a.to_screen(70.0, 20.0), b.to_screen(70.0, 20.0));
                assert!(p.distance(q) < 0.01, "{flip} {from} {to}: {p:?} vs {q:?}");
            }
        }
    }

    /// #1919: with Delete Cropped Pixels off, a second crop's handle dragged out toward the hidden
    /// pixels snaps exactly onto their edge (a layer edge), however far from the handle's centre
    /// the press grabbed it, in both modes, with and without Auto Center Preview: the committed
    /// crop has no transparent strip.
    #[test]
    fn a_recrop_handle_snaps_exactly_to_the_hidden_pixels_edge() {
        // The image 200 × 100 cropped to [40, 20, 140, 70]: its pixels stay at [-40, -20, 160, 80].
        let hidden = [-40.0, -20.0, 160.0, 80.0];
        for classic in [false, true] {
            for auto in [true, false] {
                for (hx, hy) in [(1i8, 0i8), (0, -1), (-1, -1), (1, 1)] {
                    // A corner also with the image's 2:1 ratio held, which both hidden corners keep.
                    for (grab, ratio) in [(-4.0f32, false), (0.0, false), (4.0, false), (-4.0, true), (4.0, true)] {
                        if ratio && (hx == 0 || hy == 0) {
                            continue;
                        }
                        let what = format!("Classic Mode {classic}, Auto Center {auto}, handle ({hx}, {hy}), grabbed {grab} px off, ratio {ratio}");
                        let mut app = app();
                        app.ui.tool_options.crop_delete = false;
                        app.ui.tool_options.crop_shield.classic_mode = classic;
                        app.ui.tool_options.crop_shield.auto_center_preview = auto;
                        crate::crop_ui::ensure_frame(&mut app);
                        app.ui.crop_rect = Some([40.0, 20.0, 140.0, 70.0]);
                        app.crop.default_frame = false;
                        crate::canvas::commit_crop(&mut app);
                        let v = &mut app.ui.views[0];
                        (v.zoom, v.center, v.fit_pending, v.rotation) = (1.0, [50.0, 25.0], false, 0.0);
                        app.ui.extras.snap = true;
                        crate::crop_ui::ensure_frame(&mut app);
                        assert_eq!(app.ui.crop_rect, Some([0.0, 0.0, 100.0, 50.0]));
                        if ratio {
                            app.ui.tool_options.crop_ratio = "original".into();
                        }
                        let grab = vec2(f32::from(hx), f32::from(hy)) * grab;
                        let from = xf(&app).to_screen(50.0 + f32::from(hx) * 50.0, 25.0 + f32::from(hy) * 25.0) + grab;
                        // Toward the hidden corner (or edge), a screen pixel at a time on the longer axis.
                        let gap = vec2(f32::from(hx) * if hx < 0 { 40.0 } else { 60.0 }, f32::from(hy) * if hy < 0 { 20.0 } else { 30.0 });
                        let dir = gap / gap.x.abs().max(gap.y.abs());
                        let d = xf(&app).to_doc(from);
                        tool_event(&mut app, ToolEvent::Down { x: d[0], y: d[1], pressure: 1.0 }, NONE);
                        // Drag outward a screen pixel at a time; let go once the moving edges show
                        // the snap to the hidden pixels' edges (as the user would).
                        let snapped = |app: &PhotocraftApp| {
                            let on = |vertical: bool, pos: f64| {
                                app.prefs_rt
                                    .snap_lines
                                    .iter()
                                    .any(|l| l.vertical == vertical && l.pos == pos && l.kind == photocraft_engine::snap::SnapKind::LayerEdge)
                            };
                            (hx == 0 || on(true, if hx < 0 { hidden[0] } else { hidden[2] }))
                                && (hy == 0 || on(false, if hy < 0 { hidden[1] } else { hidden[3] }))
                        };
                        let mut end = None;
                        for i in 1..=80 {
                            let d = xf(&app).to_doc(from + dir * i as f32);
                            tool_event(&mut app, ToolEvent::Move { x: d[0], y: d[1], pressure: 1.0 }, NONE);
                            if snapped(&app) {
                                end = Some(from + dir * i as f32);
                                break;
                            }
                        }
                        let end = end.unwrap_or_else(|| panic!("{what}: never snapped to the hidden edge"));
                        // Released where it stopped, read through the view as it is now.
                        let end = xf(&app).to_doc(end);
                        tool_event(&mut app, ToolEvent::Up { x: end[0], y: end[1] }, NONE);
                        let pick = |h: i8, lo: f64, hi: f64, a: f64, b: f64| match h {
                            -1 => [a, hi],
                            1 => [lo, b],
                            _ => [lo, hi],
                        };
                        let [x0, x1] = pick(hx, 0.0, 100.0, hidden[0], hidden[2]);
                        let [y0, y1] = pick(hy, 0.0, 50.0, hidden[1], hidden[3]);
                        assert_eq!(app.ui.crop_rect, Some([x0, y0, x1, y1]), "{what}: the frame lands exactly on the hidden edge");
                        crate::canvas::commit_crop(&mut app);
                        let st = app.session.active().unwrap();
                        assert_eq!((st.doc.size.width, st.doc.size.height), ((x1 - x0) as u32, (y1 - y0) as u32), "{what}");
                        let layer = st.doc.walk().into_iter().map(|(_, _, l)| l.id).next().unwrap();
                        let r = photocraft_engine::snap::layer_rect(&st.doc, layer).unwrap();
                        assert_eq!(r, [hidden[0] - x0, hidden[1] - y0, hidden[2] - x0, hidden[3] - y0], "{what}: no transparent strip");
                    }
                }
            }
        }
    }
}
