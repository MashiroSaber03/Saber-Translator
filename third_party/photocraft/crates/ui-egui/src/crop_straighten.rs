//! The Crop tool's Straighten (Photoshop's options-bar level button, #1919): draw a line along
//! something that should be level (a horizon) or plumb (a wall), and the crop frame turns so that
//! line ends up horizontal, or vertical when it is nearer vertical (the Ruler's Straighten
//! convention, `analysis_cmds::straighten_angle`).
//!
//! - The options-bar button arms it for one line, then disarms (one-shot); clicking it again
//!   disarms it without a line.
//! - Holding ⌘ (Ctrl off the Mac) with the Crop tool is a temporary Straighten: a ⌘-drag draws the
//!   line; releasing ⌘ gives the normal crop gestures back.
//!
//! While the line is drawn its angle shows beside the pointer. Straighten sets the frame's angle
//! and fits the frame to the image ([`fitted`]); in Classic Mode the frame then shows turned over
//! the image, in the default mode the image turns behind the upright box (`crop_mode`). It is only a pending frame edit:
//! ↵ commits it and Esc cancels it as any crop. A line shorter than [`MIN_LINE_PX`] screen points
//! does nothing.

use egui::{Modifiers, Pos2};

use crate::PhotocraftApp;
use crate::canvas::ToolEvent;
use crate::state::Tool;

/// The shortest line (screen points) that straightens.
pub const MIN_LINE_PX: f64 = 3.0;

/// Straighten state (not serialized: it only lasts for a line or while a key is held).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Straighten {
    /// The options-bar button is on: the next drag draws a straighten line.
    pub armed: bool,
    /// ⌘ (Ctrl) is held: set by the canvas every frame.
    pub held: bool,
    /// The line being drawn: start and end in document px.
    pub line: Option<[[f64; 2]; 2]>,
}

/// ⌘ (Ctrl) is held over the canvas.
pub fn set_held(app: &mut PhotocraftApp, down: bool) {
    app.crop.straighten.held = down;
}

/// A drag with the Crop tool now draws a straighten line (or one is being drawn): the canvas
/// shows the straighten cursor instead of the frame's.
pub fn mode(app: &PhotocraftApp) -> bool {
    let s = &app.crop.straighten;
    app.ui.tool == Tool::Crop && app.crop.drag.is_none() && (s.armed || s.held || s.line.is_some())
}

impl Straighten {
    /// The options-bar button: arms (or disarms) Straighten for one line.
    pub fn toggle(&mut self) {
        self.armed = !self.armed;
        self.line = None;
    }
}

/// [`Straighten::toggle`] for the app's Crop tool.
pub fn toggle(app: &mut PhotocraftApp) {
    app.crop.straighten.toggle();
}

/// Drops the armed button and any line (Esc, another tool).
pub fn reset(app: &mut PhotocraftApp) {
    app.crop.straighten.armed = false;
    app.crop.straighten.line = None;
}

/// Pointer input for Straighten, called by the Crop tool first (finite `p`, a document open).
/// Returns true when the event drew a straighten line.
pub fn pointer(app: &mut PhotocraftApp, ev: ToolEvent, mods: Modifiers) -> bool {
    match ev {
        ToolEvent::Down { x, y, .. } => {
            let s = app.crop.straighten;
            if app.crop.drag.is_some() || !(s.armed || mods.command) {
                return false;
            }
            app.crop.straighten.line = Some([[x, y], [x, y]]);
            app.crop.editing = app.ui.crop_rect.is_some();
            true
        }
        ToolEvent::Move { x, y, .. } => {
            let Some(l) = app.crop.straighten.line.as_mut() else { return false };
            l[1] = [x, y];
            true
        }
        ToolEvent::Up { x, y } => {
            let Some([a, _]) = app.crop.straighten.line.take() else { return false };
            // One line per click of the button.
            app.crop.straighten.armed = false;
            let zoom = f64::from(app.point_zoom());
            apply(app, a, [x, y], zoom);
            true
        }
    }
}

/// The frame's angle (degrees, clockwise on screen) that makes the line from `a` to `b` level, or
/// plumb when it is nearer vertical; `None` for a line shorter than `min_len` or not finite.
pub fn frame_angle(a: [f64; 2], b: [f64; 2], min_len: f64) -> Option<f64> {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len = dx.hypot(dy);
    if !len.is_finite() || len < min_len.max(f64::MIN_POSITIVE) {
        return None;
    }
    let rot = photocraft_engine::analysis_cmds::straighten_angle(&photocraft_doc::Ruler { start: a, end: b, protractor: None });
    // ↵ turns the document back by the frame's angle; turning it by `rot` levels the line.
    rot.is_finite().then(|| crate::crop_ui::normalized_angle(-rot) + 0.0)
}

/// The largest frame with the proportions of `w`×`h`, turned `deg` degrees, that stays inside a
/// `cw`×`ch` canvas, centred on the canvas. `None` when a size is not a positive finite number.
///
/// Kept apart on purpose: whether Photoshop's Classic Mode fits the frame exactly this way after
/// Straighten (keeping the frame's proportions, centred on the image) is not verified.
pub fn fitted(w: f64, h: f64, cw: f64, ch: f64, deg: f64) -> Option<[f64; 4]> {
    if ![w, h, cw, ch, deg].iter().all(|v| v.is_finite()) || w <= 0.0 || h <= 0.0 || cw <= 0.0 || ch <= 0.0 {
        return None;
    }
    let (s, c) = deg.to_radians().sin_cos();
    let (s, c) = (s.abs(), c.abs());
    // The turned frame's bounds, scaled by `k`, must fit the canvas on both axes.
    let k = (cw / (w * c + h * s)).min(ch / (w * s + h * c));
    if !k.is_finite() || k <= 0.0 {
        return None;
    }
    let (hw, hh, cx, cy) = (w * k / 2.0, h * k / 2.0, cw / 2.0, ch / 2.0);
    Some([cx - hw, cy - hh, cx + hw, cy + hh])
}

/// The line from `a` to `b` (document px, at `zoom` screen points per px) straightens the frame.
fn apply(app: &mut PhotocraftApp, a: [f64; 2], b: [f64; 2], zoom: f64) {
    let min_len = if zoom.is_finite() && zoom > 0.0 { MIN_LINE_PX / zoom } else { MIN_LINE_PX };
    let Some(deg) = frame_angle(a, b, min_len) else { return };
    let Some(size) = app.session.active().map(|st| (f64::from(st.doc.size.width), f64::from(st.doc.size.height))) else { return };
    // The frame keeps its proportions (an options-bar ratio is already the frame's); no frame
    // takes the canvas's.
    let (w, h) = app.ui.crop_rect.filter(|r| r.iter().all(|v| v.is_finite())).map_or(size, |r| (r[2] - r[0], r[3] - r[1]));
    let Some(r) = fitted(w, h, size.0, size.1, deg) else { return };
    if app.ui.crop_rect == Some(r) && crate::crop_ui::angle(app) == deg {
        return;
    }
    app.ui.crop_rect = Some(r);
    app.ui.crop_angle = deg;
    app.crop.default_frame = false;
    app.crop.editing = true;
}

/// The line being drawn and its angle beside the pointer. `to_screen` maps document px to the
/// screen.
pub fn draw(app: &PhotocraftApp, painter: &egui::Painter, to_screen: impl Fn([f64; 2]) -> Pos2, hover: Option<Pos2>) {
    let Some([a, b]) = app.crop.straighten.line.filter(|_| app.ui.tool == Tool::Crop) else { return };
    let t = crate::theme::Tokens::get(painter.ctx());
    let seg = [to_screen(a), to_screen(b)];
    painter.line_segment(seg, egui::Stroke::new(3.0, t.shadow));
    painter.line_segment(seg, egui::Stroke::new(1.0, t.accent_text));
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let deg = (-dy).atan2(dx).to_degrees();
    if let Some(h) = hover.filter(|_| deg.is_finite() && (dx != 0.0 || dy != 0.0)) {
        crate::canvas::draw_value_readout(app, painter.ctx(), "crop-straighten-readout", h, ["Angle:"], [format!("{deg:.1}°")]);
    }
}

/// The options-bar Straighten button (Photoshop's level icon), lit while armed.
pub fn options_button(s: &mut Straighten, ui: &mut egui::Ui) {
    if crate::icons::button(ui, "ruler", 24.0, s.armed, tl!("Straighten the image by drawing a line on it")).clicked() {
        s.toggle();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::tool_event;
    use crate::crop_ui::{CropDrag, ensure_frame};
    use photocraft_doc::{Color, ColorMode, Document, SampleType, Size};

    const NONE: Modifiers = Modifiers::NONE;
    const CMD: Modifiers = Modifiers::COMMAND;

    fn app(depth: SampleType) -> PhotocraftApp {
        let doc = Document::with_background("straighten", Size::new(200, 100), ColorMode::Rgb, depth, Color::WHITE);
        let mut s = photocraft_engine::Session::new();
        s.add_document(doc, None);
        let mut app = PhotocraftApp::new(s, crate::Services::default());
        app.ui.tool = Tool::Crop;
        // Classic Mode: the frame turns over the image (the default mode's Straighten is tested in
        // `crop_mode`).
        app.ui.tool_options.crop_shield.classic_mode = true;
        app.ui.extras.snap = false;
        app.ui.view.show.smart_guides = false;
        ensure_frame(&mut app);
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

    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    /// Every corner of frame `r` turned `deg` lies inside the `w`×`h` canvas.
    fn inside(r: [f64; 4], deg: f64, w: f64, h: f64) -> bool {
        crate::crop_ui::corners(r, deg).iter().all(|p| p[0] >= -1e-6 && p[1] >= -1e-6 && p[0] <= w + 1e-6 && p[1] <= h + 1e-6)
    }

    #[test]
    fn the_frame_turns_with_a_horizon_and_nearly_vertical_lines_turn_it_plumb() {
        // A horizon falling 10° to the right: the frame turns 10° clockwise, so ↵ levels it.
        let d = 10.0f64.to_radians();
        assert!(near(frame_angle([0.0, 0.0], [100.0 * d.cos(), 100.0 * d.sin()], 1.0).unwrap(), 10.0));
        // Rising to the right: counter-clockwise. Drawn right to left: the same line.
        assert!(near(frame_angle([0.0, 50.0], [100.0, 40.0], 1.0).unwrap(), -(10.0f64 / 100.0).atan().to_degrees()));
        assert!(near(frame_angle([100.0, 40.0], [0.0, 50.0], 1.0).unwrap(), -(10.0f64 / 100.0).atan().to_degrees()));
        // A wall leaning 5° (nearer vertical) is made plumb, not level.
        let a = frame_angle([50.0, 0.0], [50.0 + 100.0 * 5.0f64.to_radians().tan(), 100.0], 1.0).unwrap();
        assert!(near(a, -5.0), "{a}");
        // Level and plumb lines need no turn.
        assert_eq!(frame_angle([0.0, 0.0], [40.0, 0.0], 1.0), Some(0.0));
        assert_eq!(frame_angle([0.0, 0.0], [0.0, 40.0], 1.0), Some(0.0));
        // Too short, zero-length or not finite: nothing.
        assert_eq!(frame_angle([5.0, 5.0], [5.0, 5.0], 0.0), None);
        assert_eq!(frame_angle([5.0, 5.0], [6.0, 6.0], 3.0), None);
        assert_eq!(frame_angle([f64::NAN, 0.0], [10.0, 0.0], 1.0), None);
        assert_eq!(frame_angle([0.0, 0.0], [f64::INFINITY, 1.0], 1.0), None);
    }

    #[test]
    fn the_fitted_frame_is_the_largest_of_its_proportions_inside_the_canvas() {
        // Upright, the canvas's own proportions fill it.
        assert_eq!(fitted(200.0, 100.0, 200.0, 100.0, 0.0), Some([0.0, 0.0, 200.0, 100.0]));
        for deg in [-45.0, -10.0, 3.0, 10.0, 30.0, 45.0, 90.0] {
            let r = fitted(200.0, 100.0, 200.0, 100.0, deg).unwrap();
            assert!(inside(r, deg, 200.0, 100.0), "{deg}: {r:?}");
            assert!(near((r[2] - r[0]) / (r[3] - r[1]), 2.0), "keeps 2:1 at {deg}");
            assert!(near((r[0] + r[2]) / 2.0, 100.0) && near((r[1] + r[3]) / 2.0, 50.0), "centred at {deg}");
            // As large as possible: 1% larger would leave the canvas.
            let c = crate::crop_ui::center(r);
            let g = [c[0] - (c[0] - r[0]) * 1.01, c[1] - (c[1] - r[1]) * 1.01, c[0] + (r[2] - c[0]) * 1.01, c[1] + (r[3] - c[1]) * 1.01];
            assert!(!inside(g, deg, 200.0, 100.0), "{deg}: not the largest");
        }
        for bad in [
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, -1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, f64::NAN],
            [f64::INFINITY, 1.0, 1.0, 1.0, 0.0],
        ] {
            assert_eq!(fitted(bad[0], bad[1], bad[2], bad[3], bad[4]), None, "{bad:?}");
        }
    }

    #[test]
    fn the_button_straightens_one_line_then_disarms() {
        for depth in [SampleType::U8, SampleType::U16] {
            let mut app = app(depth);
            assert!(app.crop.default_frame && !mode(&app));
            toggle(&mut app);
            assert!(mode(&app), "armed");
            // A line inside the frame straightens instead of drawing a new frame.
            drag(&mut app, &[[20.0, 40.0], [100.0, 50.0], [180.0, 60.0]], NONE);
            let want = (20.0f64 / 160.0).atan().to_degrees();
            assert!(near(app.ui.crop_angle, want), "{depth:?}: {}", app.ui.crop_angle);
            let r = app.ui.crop_rect.unwrap();
            assert!(inside(r, want, 200.0, 100.0) && near((r[2] - r[0]) / (r[3] - r[1]), 2.0), "{r:?}");
            assert!(!app.crop.default_frame && app.crop.editing && crate::crop_ui::pending(&app));
            assert!(!app.crop.straighten.armed && app.crop.straighten.line.is_none(), "one-shot");
            // The next drag is a normal crop gesture again: inside the frame it moves it.
            let before = app.ui.crop_rect.unwrap();
            drag(&mut app, &[[100.0, 50.0], [105.0, 50.0]], NONE);
            let after = app.ui.crop_rect.unwrap();
            assert!(near(after[0] - before[0], 5.0) && near(app.ui.crop_angle, want), "{depth:?}");
            // ↵ commits the turn and the crop in one step.
            let steps = app.session.active().unwrap().history.past_len();
            crate::canvas::commit_crop(&mut app);
            let st = app.session.active().unwrap();
            assert_eq!(st.history.past_len(), steps + 1);
            assert_eq!((st.doc.depth, app.ui.crop_angle), (depth, 0.0));
        }
    }

    #[test]
    fn clicking_the_button_again_disarms_it_and_esc_drops_it() {
        let mut app = app(SampleType::U8);
        toggle(&mut app);
        toggle(&mut app);
        assert!(!app.crop.straighten.armed);
        toggle(&mut app);
        crate::crop_ui::cancel(&mut app);
        assert!(!app.crop.straighten.armed && !mode(&app));
        // Leaving the tool drops it too.
        toggle(&mut app);
        app.ui.tool = Tool::Brush;
        ensure_frame(&mut app);
        assert!(!app.crop.straighten.armed);
    }

    #[test]
    fn a_tiny_or_level_line_changes_nothing() {
        let mut app = app(SampleType::U8);
        toggle(&mut app);
        // A click (zero length), then a line under 3 screen points.
        drag(&mut app, &[[50.0, 50.0]], NONE);
        assert_eq!((app.ui.crop_rect, app.ui.crop_angle), (Some([0.0, 0.0, 200.0, 100.0]), 0.0));
        assert!(app.crop.default_frame, "the untouched frame stays the default");
        assert!(!app.crop.straighten.armed, "the click used the button");
        let zoom = f64::from(app.point_zoom());
        toggle(&mut app);
        drag(&mut app, &[[50.0, 50.0], [50.0 + 2.0 / zoom, 50.0 + 1.0 / zoom]], NONE);
        assert_eq!(app.ui.crop_angle, 0.0);
        assert!(app.ui.crop_rect.unwrap().iter().all(|v| v.is_finite()));
        // An already level line on the default frame leaves it as it is.
        toggle(&mut app);
        drag(&mut app, &[[10.0, 50.0], [190.0, 50.0]], NONE);
        assert_eq!((app.ui.crop_rect, app.ui.crop_angle), (Some([0.0, 0.0, 200.0, 100.0]), 0.0));
        // Non-finite pointer positions never reach it.
        toggle(&mut app);
        drag(&mut app, &[[f64::NAN, 50.0], [190.0, f64::INFINITY]], NONE);
        assert!(app.ui.crop_angle.is_finite() && app.ui.crop_rect.unwrap().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn holding_command_is_a_temporary_straighten() {
        let mut app = app(SampleType::U8);
        set_held(&mut app, true);
        assert!(mode(&app));
        assert_eq!(crate::crop_ui::cursor(&app, [100.0, 50.0]), Some(egui::CursorIcon::Crosshair));
        assert_eq!(crate::crop_ui::turn_cursor_dir(&app, [300.0, 50.0]), None, "no turn cursor while straightening");
        // A ⌘-drag outside the frame draws the line instead of turning the frame about the pointer.
        tool_event(&mut app, ToolEvent::Down { x: 250.0, y: 60.0, pressure: 1.0 }, CMD);
        tool_event(&mut app, ToolEvent::Move { x: 150.0, y: 0.0, pressure: 1.0 }, CMD);
        assert!(app.crop.straighten.line.is_some() && app.crop.drag.is_none());
        tool_event(&mut app, ToolEvent::Up { x: 0.0, y: 0.0 }, CMD);
        // From (250, 60) to (0, 0): nearer horizontal, rising to the left.
        let want = (60.0f64 / 250.0).atan().to_degrees();
        assert!(near(app.ui.crop_angle, want), "{}", app.ui.crop_angle);
        // Released: the same drag outside turns the frame again.
        set_held(&mut app, false);
        assert!(!mode(&app));
        drag(&mut app, &[[300.0, 50.0], [300.0, 90.0]], NONE);
        assert!(app.crop.drag.is_none() && app.ui.crop_angle > want, "{}", app.ui.crop_angle);
        // And a ⌘-drag in the middle of a crop gesture doesn't hijack it.
        app.crop.drag = Some(CropDrag::Move { start: [100.0, 50.0], rect: app.ui.crop_rect.unwrap() });
        assert!(!pointer(&mut app, ToolEvent::Down { x: 100.0, y: 50.0, pressure: 1.0 }, CMD));
    }
}
