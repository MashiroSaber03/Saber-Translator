//! Move-tool modifier keys, as in Photoshop (#90):
//!
//! - ⇧ while dragging locks the move to the dominant axis (horizontal or vertical). The axis is
//!   re-evaluated as the drag goes on, so moving far enough the other way switches it; a small
//!   hysteresis keeps it from flickering near the diagonal. Free Transform uses the same rule with
//!   eight directions ([`constrain`]).
//! - ⌥ held when the drag starts duplicates the selected layer(s) (groups included) once the
//!   pointer actually moves, and the drag then moves the copies. Duplicate and move land as one
//!   history step ("Duplicate + Move"). ⇧⌥ combines both.
//! - Arrow keys nudge the selected layers 1 px (⇧: 10 px); ⌥ duplicates first. While Free
//!   Transform is active they nudge the box instead.
//! - With a selection the drag and the arrow keys move the selected pixels (`move_ui`): ⇧ locks
//!   the drag to multiples of 45° and ⌥ copies the pixels instead of duplicating the layer.
//!
//! The hooks are small and local: [`filter_event`] rewrites pointer events before the Move tool
//! sees them and [`finish`] folds the history after the drag, so the Move drag pipeline itself is
//! untouched.

use serde_json::json;

use crate::PhotocraftApp;
use crate::canvas::ToolEvent;
use crate::state::Tool;

/// Directions ⇧ locks a Move-tool drag to: horizontal and vertical (Photoshop's Move tool).
pub const MOVE_DIRECTIONS: u32 = 4;
/// Directions ⇧ locks a Free Transform body drag, or a Move-tool drag of selected pixels, to: the
/// axes and the 45° diagonals.
pub const TRANSFORM_DIRECTIONS: u32 = 8;
/// Extra angle (degrees) the pointer must pass the half-way line by before the locked axis
/// switches, so it doesn't flicker when dragging near the diagonal.
const HYSTERESIS_DEG: f64 = 8.0;
/// ⌥-drag duplicates only once the pointer has moved this far (screen points), so an ⌥-click
/// without a drag doesn't make a copy (Photoshop).
const DUPLICATE_THRESHOLD: f64 = 2.0;
/// History label of an ⌥-drag / ⌥-nudge.
pub const DUPLICATE_MOVE_LABEL: &str = "Duplicate + Move";

/// Locks the drag offset `d` to the nearest of `n` evenly spaced directions (0° = right). `prev`
/// is the previously locked offset: its direction is kept until `d` is clearly nearer another one.
pub fn constrain(d: [f64; 2], prev: Option<[f64; 2]>, n: u32) -> [f64; 2] {
    let len = d[0].hypot(d[1]);
    if n == 0 || !len.is_finite() || len == 0.0 {
        return d;
    }
    let step = std::f64::consts::TAU / f64::from(n);
    let a = d[1].atan2(d[0]);
    let mut k = (a / step).round();
    if let Some(p) = prev.filter(|p| p[0].hypot(p[1]) > 0.0) {
        let kp = (p[1].atan2(p[0]) / step).round();
        // Angular distance from `d` to the previous direction, wrapped to [0, π].
        let off = (a - kp * step).rem_euclid(std::f64::consts::TAU);
        let off = off.min(std::f64::consts::TAU - off);
        if off < step / 2.0 + HYSTERESIS_DEG.to_radians() {
            k = kp;
        }
    }
    let (s, c) = (k * step).sin_cos();
    let t = (d[0] * c + d[1] * s).max(0.0);
    // Snap tiny float noise so axis-locked moves stay exactly on the axis.
    let clean = |v: f64| if v.abs() < 1e-9 { 0.0 } else { v };
    [clean(t * c), clean(t * s)]
}

/// Per-drag state (one Move-tool gesture at a time).
#[derive(Clone, Debug, Default)]
pub struct MoveDrag {
    /// Where the drag started (document pixels); `None` when no Move drag is in progress.
    start: Option<[f64; 2]>,
    /// ⌥ was held when the drag started.
    alt: bool,
    /// The last offset after ⇧ locking (keeps the locked axis stable).
    last: Option<[f64; 2]>,
    /// History length before the ⌥-drag duplicated (set once it has).
    dup_from: Option<usize>,
}

fn past_len(app: &PhotocraftApp) -> Option<usize> {
    app.session.active().map(|st| st.history.past_len())
}

fn moving(app: &PhotocraftApp) -> bool {
    app.drag.as_ref().is_some_and(|d| d.tool == Tool::Move)
}

/// Duplicates the selected layers (the copies become the selection), remembering the history
/// length before so [`fold_history`] can merge the copy and the move into one step.
fn duplicate(app: &mut PhotocraftApp) -> Option<usize> {
    let before = past_len(app)?;
    // In place: the copy follows the pointer from the original, artboards too (#1531).
    match app.run("layer.duplicate", json!({"inPlace": true})) {
        Ok(_) => Some(before),
        Err(e) => {
            app.ui.status = e;
            None
        }
    }
}

/// Merges every history step after `from` into one "Duplicate + Move" step.
fn fold_history(app: &mut PhotocraftApp, from: usize) {
    let Some(st) = app.session.active_mut() else { return };
    while st.history.past_len() > from + 1 {
        if !st.history.purge_last() {
            break;
        }
    }
    if st.history.past_len() == from + 1 {
        st.history.set_current_label(DUPLICATE_MOVE_LABEL);
    }
}

/// Rewrites a Move-tool pointer event for the held modifiers: ⇧ locks it to an axis, and the
/// first real movement of an ⌥-drag duplicates the layers being moved. Other tools pass through.
pub fn filter_event(app: &mut PhotocraftApp, ev: ToolEvent, mods: egui::Modifiers) -> ToolEvent {
    if app.active_tool() != Tool::Move || app.ui.transform.is_some() {
        app.move_mods = MoveDrag::default();
        return ev;
    }
    match ev {
        ToolEvent::Down { x, y, .. } => {
            app.move_mods = MoveDrag { start: Some([x, y]), alt: mods.alt, last: None, dup_from: None };
            ev
        }
        ToolEvent::Move { x, y, pressure } => {
            let [x, y] = drag_to(app, [x, y], mods);
            ToolEvent::Move { x, y, pressure }
        }
        ToolEvent::Up { x, y } => {
            let [x, y] = drag_to(app, [x, y], mods);
            ToolEvent::Up { x, y }
        }
    }
}

fn drag_to(app: &mut PhotocraftApp, p: [f64; 2], mods: egui::Modifiers) -> [f64; 2] {
    let Some(start) = app.move_mods.start.filter(|_| moving(app)) else { return p };
    // Selected pixels (`move_ui`): ⌥ copied them at the press.
    let pixels = app.drag.as_ref().is_some_and(|d| d.sel_move.is_some());
    let mut d = [p[0] - start[0], p[1] - start[1]];
    if mods.shift {
        d = constrain(d, app.move_mods.last, if pixels { TRANSFORM_DIRECTIONS } else { MOVE_DIRECTIONS });
    }
    app.move_mods.last = Some(d);
    let zoom = f64::from(app.point_zoom().max(0.01));
    if app.move_mods.alt && !pixels && app.move_mods.dup_from.is_none() && d[0].hypot(d[1]) * zoom >= DUPLICATE_THRESHOLD {
        app.move_mods.dup_from = duplicate(app);
        // Only once per drag, even if duplicating failed.
        app.move_mods.alt = false;
    }
    [start[0] + d[0], start[1] + d[1]]
}

/// Call after the Move tool finished its drag: an ⌥-drag's copy and move become one step.
pub fn finish(app: &mut PhotocraftApp) {
    let st = std::mem::take(&mut app.move_mods);
    if let Some(from) = st.dup_from {
        fold_history(app, from);
    }
}

/// Call when a Move drag ends without moving (its layers were dragged to another document): an
/// ⌥-drag's copy is taken back.
pub fn abandon(app: &mut PhotocraftApp) {
    if std::mem::take(&mut app.move_mods).dup_from.is_some() {
        app.session.undo();
    }
}

/// Arrow keys with the Move tool (or while Free Transform is active): nudge 1 px, ⇧ 10 px;
/// ⌥ duplicates the layers first. ⌘-arrows (Ctrl off the Mac) do the same from most other tools,
/// where ⌘ is the temporary Move tool (`hold_keys::cmd_nudges`, #2474). Returns true when a key
/// was used.
pub fn arrow_keys(app: &mut PhotocraftApp, ctx: &egui::Context) -> bool {
    use egui::{Key, Modifiers};
    let cmd = ctx.input(|i| i.modifiers.command) && crate::hold_keys::cmd_nudges(app);
    if app.ui.tool != Tool::Move && app.ui.transform.is_none() && !cmd {
        return selection_arrow_keys(app, ctx) || crate::direct_select::arrow_keys(app, ctx) || brush_arrow_keys(app, ctx);
    }
    // `consume_key` never ignores an extra ⌘, so it is part of every pattern while it moves.
    let base = if cmd { Modifiers::COMMAND } else { Modifiers::NONE };
    let keys = [(Key::ArrowLeft, -1.0, 0.0), (Key::ArrowRight, 1.0, 0.0), (Key::ArrowUp, 0.0, -1.0), (Key::ArrowDown, 0.0, 1.0)];
    for (key, ux, uy) in keys {
        // Most specific first: `consume_key` ignores extra ⇧ / ⌥ (`matches_logically`).
        for mods in [Modifiers::SHIFT | Modifiers::ALT, Modifiers::SHIFT, Modifiers::ALT, Modifiers::NONE] {
            if ctx.input_mut(|i| i.consume_key(base | mods, key)) {
                let k = if mods.shift { 10.0 } else { 1.0 };
                nudge(app, ux * k, uy * k, mods.alt);
                return true;
            }
        }
    }
    false
}

/// ← / → with a painting tool: turn the brush tip 1° (⇧ 15°), as Photoshop does; → turns it
/// clockwise. Returns true when a key was used.
fn brush_arrow_keys(app: &mut PhotocraftApp, ctx: &egui::Context) -> bool {
    use egui::{Key, Modifiers};
    // Only plain or ⇧ arrows (`consume_key` ignores an extra ⌥, which kerns type), and never
    // while type is being edited.
    let m = ctx.input(|i| i.modifiers);
    if !app.ui.tool.is_brushlike() || app.ui.text_edit.is_some() || m.alt || m.command || m.ctrl {
        return false;
    }
    for (key, sign) in [(Key::ArrowLeft, 1.0), (Key::ArrowRight, -1.0)] {
        // ⇧ first: `consume_key` ignores an extra ⇧.
        for (mods, step) in [(Modifiers::SHIFT, 15.0), (Modifiers::NONE, 1.0)] {
            if ctx.input_mut(|i| i.consume_key(mods, key)) {
                if let Err(e) = app.run("tools.rotateBrush", json!({"degrees": sign * step})) {
                    app.ui.status = e;
                }
                return true;
            }
        }
    }
    false
}

/// Moves the selected layers (or the Free Transform box, or the selected pixels) by `(dx, dy)`
/// pixels, as the Move tool (also when ⌘ makes it the temporary Move tool).
pub fn nudge(app: &mut PhotocraftApp, dx: f64, dy: f64, duplicate_first: bool) {
    if let Some(t) = app.ui.transform.as_mut() {
        if t.warp.is_none() {
            t.quad = t.quad.map(|q| [q[0] + dx, q[1] + dy]);
            t.pivot = [t.pivot[0] + dx, t.pivot[1] + dy];
        }
        return;
    }
    if app.drag.is_some() {
        return;
    }
    if crate::move_ui::moves_selected_pixels_with(app, Tool::Move) {
        crate::move_ui::float_selected(app, duplicate_first, dx, dy);
        return;
    }
    let from = if duplicate_first { duplicate(app) } else { None };
    if let Err(e) = app.run("layer.translate", json!({"dx": dx, "dy": dy})) {
        app.ui.status = e;
    }
    if let Some(from) = from {
        fold_history(app, from);
    }
}

/// Arrow keys with a selection tool (#1428), as in Photoshop: nudge the selection outline 1 px
/// (⇧ 10 px) through `select.transformSelection`, one history step per press; the pixels stay
/// put. A floating piece (⌘-dragged pixels) moves instead, like a plain drag on it. Not while a
/// drag, a polygon or a Magnetic Lasso border is in progress. Returns true when a key was used.
fn selection_arrow_keys(app: &mut PhotocraftApp, ctx: &egui::Context) -> bool {
    if !crate::tool_feedback::is_selection_tool(app.ui.tool) || app.drag.is_some() || !app.ui.polygon.is_empty() || app.ui.magnetic.active() {
        return false;
    }
    let floating = app.session.active().is_some_and(|st| photocraft_engine::float_cmds::floating(st).is_some());
    if !floating && !app.session.is_enabled("select.transformSelection") {
        return false;
    }
    use egui::{Key, Modifiers};
    let keys = [(Key::ArrowLeft, -1.0, 0.0), (Key::ArrowRight, 1.0, 0.0), (Key::ArrowUp, 0.0, -1.0), (Key::ArrowDown, 0.0, 1.0)];
    for (key, ux, uy) in keys {
        // ⇧ first: `consume_key` ignores an extra ⇧.
        for mods in [Modifiers::SHIFT, Modifiers::NONE] {
            if ctx.input_mut(|i| i.consume_key(mods, key)) {
                let k = if mods.shift { 10.0 } else { 1.0 };
                let (dx, dy) = (ux * k, uy * k);
                let id = if floating { "select.float" } else { "select.transformSelection" };
                if let Err(e) = app.run(id, json!({"dx": dx, "dy": dy})) {
                    app.ui.status = e;
                    app.ui.status_error = true;
                }
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: [f64; 2], b: [f64; 2]) -> bool {
        (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9
    }

    #[test]
    fn shift_picks_the_dominant_axis() {
        assert!(near(constrain([10.0, 3.0], None, 4), [10.0, 0.0]));
        assert!(near(constrain([-2.0, -9.0], None, 4), [0.0, -9.0]));
        assert!(near(constrain([0.0, 0.0], None, 4), [0.0, 0.0]));
        // Eight directions: a near-diagonal drag lands on 45°.
        let d = constrain([10.0, 9.0], None, 8);
        assert!((d[0] - d[1]).abs() < 1e-9 && d[0] > 9.0, "{d:?}");
        assert!(near(constrain([10.0, 1.0], None, 8), [10.0, 0.0]));
        // Hostile input never panics and stays finite or passes through.
        let _ = constrain([f64::NAN, 1.0], None, 4);
        let _ = constrain([f64::INFINITY, 1.0], Some([f64::NAN, 0.0]), 0);
    }

    #[test]
    fn shift_axis_switches_once_the_pointer_is_clearly_nearer_the_other_one() {
        let h = constrain([20.0, 2.0], None, 4);
        assert!(near(h, [20.0, 0.0]));
        // Just past the diagonal: hysteresis keeps horizontal.
        let still = constrain([20.0, 21.0], Some(h), 4);
        assert!(near(still, [20.0, 0.0]), "{still:?}");
        // Far enough the other way: switches to vertical.
        let v = constrain([20.0, 40.0], Some(h), 4);
        assert!(near(v, [0.0, 40.0]), "{v:?}");
        // And back.
        assert!(near(constrain([30.0, 20.0], Some(v), 4), [30.0, 0.0]));
    }

    fn app_with_layer() -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
        app.sync_views();
        app.session.execute("layer.new.layer", json!({})).unwrap();
        app.session
            .edit("paint", |doc, a| {
                doc.layer_mut(a.unwrap()).unwrap().surface_mut().unwrap().fill_rect(photocraft_geom::Rect::new(8, 8, 24, 24), &[1.0, 0.0, 0.0, 1.0]);
                Ok(())
            })
            .unwrap();
        app.ui.tool = Tool::Move;
        app.ui.tool_options.move_auto_select = false;
        // Snapping (to the document centre with a 1:1 test view) would shift the expected offsets.
        app.ui.extras.snap = false;
        app.ui.view.show.smart_guides = false;
        app
    }

    fn bounds(app: &PhotocraftApp, id: photocraft_doc::LayerId) -> photocraft_geom::Rect {
        app.session.active().unwrap().doc.layer(id).unwrap().surface().unwrap().content_bounds()
    }

    fn drag(app: &mut PhotocraftApp, pts: &[[f64; 2]], mods: egui::Modifiers) {
        crate::canvas::tool_event(app, ToolEvent::Down { x: pts[0][0], y: pts[0][1], pressure: 1.0 }, mods);
        for p in &pts[1..pts.len() - 1] {
            crate::canvas::tool_event(app, ToolEvent::Move { x: p[0], y: p[1], pressure: 1.0 }, mods);
        }
        let e = pts[pts.len() - 1];
        crate::canvas::tool_event(app, ToolEvent::Up { x: e[0], y: e[1] }, mods);
    }

    #[test]
    fn shift_drag_moves_along_one_axis_only() {
        let mut app = app_with_layer();
        let id = app.session.active().unwrap().active_layer.unwrap();
        drag(&mut app, &[[16.0, 16.0], [22.0, 18.0], [26.0, 19.0]], egui::Modifiers::SHIFT);
        assert_eq!(bounds(&app, id), photocraft_geom::Rect::new(18, 8, 34, 24));
        // Switching axis mid-drag: the end point decides.
        drag(&mut app, &[[16.0, 16.0], [22.0, 17.0], [18.0, 30.0], [17.0, 36.0]], egui::Modifiers::SHIFT);
        assert_eq!(bounds(&app, id), photocraft_geom::Rect::new(18, 28, 34, 44));
    }

    #[test]
    fn alt_drag_duplicates_and_moves_the_copy_in_one_step() {
        let mut app = app_with_layer();
        let orig = app.session.active().unwrap().active_layer.unwrap();
        let n0 = app.session.active().unwrap().doc.layers.len();
        let h0 = app.session.active().unwrap().history.past_len();
        drag(&mut app, &[[16.0, 16.0], [20.0, 16.0], [26.0, 21.0]], egui::Modifiers::ALT);
        let st = app.session.active().unwrap();
        assert_eq!(st.doc.layers.len(), n0 + 1, "one copy");
        let copy = st.active_layer.unwrap();
        assert_ne!(copy, orig);
        assert_eq!(bounds(&app, orig), photocraft_geom::Rect::new(8, 8, 24, 24), "original untouched");
        assert_eq!(bounds(&app, copy), photocraft_geom::Rect::new(18, 13, 34, 29));
        let st = app.session.active().unwrap();
        assert_eq!(st.history.past_len(), h0 + 1, "one undo step");
        assert_eq!(st.history.undo_label(), Some(DUPLICATE_MOVE_LABEL));
        assert!(app.session.undo());
        assert_eq!(app.session.active().unwrap().doc.layers.len(), n0);
        assert_eq!(bounds(&app, orig), photocraft_geom::Rect::new(8, 8, 24, 24));
        // ⌥-click without a drag makes no copy.
        drag(&mut app, &[[16.0, 16.0], [16.0, 16.0]], egui::Modifiers::ALT);
        assert_eq!(app.session.active().unwrap().doc.layers.len(), n0);
        // ⇧⌥: a copy moved along one axis.
        drag(&mut app, &[[16.0, 16.0], [30.0, 19.0]], egui::Modifiers::SHIFT | egui::Modifiers::ALT);
        let st = app.session.active().unwrap();
        assert_eq!(st.doc.layers.len(), n0 + 1);
        assert_eq!(bounds(&app, st.active_layer.unwrap()), photocraft_geom::Rect::new(22, 8, 38, 24));
    }

    #[test]
    fn alt_drag_duplicates_every_selected_layer() {
        let mut app = app_with_layer();
        let a = app.session.active().unwrap().active_layer.unwrap();
        app.session.execute("layer.new.layer", json!({})).unwrap();
        let b = app.session.active().unwrap().active_layer.unwrap();
        app.session.execute("layer.select", json!({"layer": a.0, "mode": "add"})).unwrap();
        let n0 = app.session.active().unwrap().doc.layers.len();
        let h0 = app.session.active().unwrap().history.past_len();
        drag(&mut app, &[[16.0, 16.0], [26.0, 16.0]], egui::Modifiers::ALT);
        let st = app.session.active().unwrap();
        assert_eq!(st.doc.layers.len(), n0 + 2);
        assert_eq!(st.history.past_len(), h0 + 1);
        assert_eq!(bounds(&app, a), photocraft_geom::Rect::new(8, 8, 24, 24));
        assert!(st.doc.layer(b).is_some());
    }

    #[test]
    fn nudge_moves_layers_or_the_transform_box() {
        let mut app = app_with_layer();
        let id = app.session.active().unwrap().active_layer.unwrap();
        nudge(&mut app, 10.0, 0.0, false);
        nudge(&mut app, 0.0, -1.0, false);
        assert_eq!(bounds(&app, id), photocraft_geom::Rect::new(18, 7, 34, 23));
        let n0 = app.session.active().unwrap().doc.layers.len();
        nudge(&mut app, 1.0, 0.0, true);
        assert_eq!(app.session.active().unwrap().doc.layers.len(), n0 + 1);
        assert_eq!(bounds(&app, id), photocraft_geom::Rect::new(18, 7, 34, 23));
        let ctx = egui::Context::default();
        crate::transform_tool::begin(&mut app, &ctx).unwrap();
        let q0 = app.ui.transform.as_ref().unwrap().quad;
        nudge(&mut app, 0.0, 10.0, false);
        assert_eq!(app.ui.transform.as_ref().unwrap().quad[0], [q0[0][0], q0[0][1] + 10.0]);
    }

    /// One key press through `arrow_keys`, as egui delivers it.
    fn press(app: &mut PhotocraftApp, key: egui::Key, mods: egui::Modifiers) -> bool {
        let ctx = egui::Context::default();
        ctx.input_mut(|i| {
            i.modifiers = mods;
            i.events.push(egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers: mods });
        });
        arrow_keys(app, &ctx)
    }

    /// Photoshop's ← / → with a painting tool turn the brush tip 1°, ⇧ 15°; → is clockwise. Other
    /// tools keep their arrows (the Hand has none, the marquee nudges the selection).
    #[test]
    fn arrows_turn_the_brush_tip_for_painting_tools_only() {
        let mut app = app_with_layer();
        for tool in [Tool::Brush, Tool::Eraser, Tool::CloneStamp] {
            app.ui.tool = tool;
            app.run("tools.setBrush", json!({"brush": {"angle": 0}})).unwrap();
            assert!(press(&mut app, egui::Key::ArrowLeft, egui::Modifiers::NONE), "{tool:?}");
            assert_eq!(app.session.tools.brush.angle, 1.0, "{tool:?}: ← anticlockwise");
            assert!(press(&mut app, egui::Key::ArrowRight, egui::Modifiers::SHIFT));
            assert_eq!(app.session.tools.brush.angle, -14.0, "{tool:?}: ⇧→ 15° clockwise");
        }
        app.run("tools.setBrush", json!({"brush": {"angle": 0}})).unwrap();
        // ⌥-arrows are not the brush's (they kern type, and ⌘ moves).
        assert!(!press(&mut app, egui::Key::ArrowLeft, egui::Modifiers::ALT));
        app.ui.tool = Tool::Hand;
        assert!(!press(&mut app, egui::Key::ArrowLeft, egui::Modifiers::NONE));
        assert_eq!(app.session.tools.brush.angle, 0.0);
    }

    /// Real key events: ⇧ nudges 10 px and ⌥ duplicates first (egui ignores extra ⇧ / ⌥ when
    /// matching, so the plain arrow must not be tried first).
    #[test]
    fn arrow_key_modifiers_reach_the_move_tool() {
        let mut app = app_with_layer();
        let id = app.session.active().unwrap().active_layer.unwrap();
        let n0 = app.session.active().unwrap().doc.layers.len();
        assert!(press(&mut app, egui::Key::ArrowRight, egui::Modifiers::NONE));
        assert_eq!(bounds(&app, id), photocraft_geom::Rect::new(9, 8, 25, 24));
        assert!(press(&mut app, egui::Key::ArrowDown, egui::Modifiers::SHIFT));
        assert_eq!(bounds(&app, id), photocraft_geom::Rect::new(9, 18, 25, 34));
        assert!(press(&mut app, egui::Key::ArrowLeft, egui::Modifiers::ALT));
        let st = app.session.active().unwrap();
        assert_eq!(st.doc.layers.len(), n0 + 1, "⌥ duplicated first");
        assert_eq!(bounds(&app, id), photocraft_geom::Rect::new(9, 18, 25, 34), "the original stays");
    }

    /// ⌘ as a Mac sends it, and Ctrl as Windows and Linux send it (`command` is Ctrl there).
    const MAC_CMD: egui::Modifiers = egui::Modifiers { alt: false, ctrl: false, shift: false, mac_cmd: true, command: true };
    const CTRL: egui::Modifiers = egui::Modifiers { alt: false, ctrl: true, shift: false, mac_cmd: false, command: true };

    /// #2474: ⌘ is the temporary Move tool, so ⌘-arrows nudge the layer from the Brush (⌘⇧ 10 px,
    /// ⌘⌥ duplicates first); the plain arrows still do nothing there.
    #[test]
    fn cmd_arrows_nudge_the_layer_from_the_brush() {
        let mut app = app_with_layer();
        app.ui.tool = Tool::Brush;
        let id = app.session.active().unwrap().active_layer.unwrap();
        let n0 = app.session.active().unwrap().doc.layers.len();
        // A plain arrow turns the Brush's tip (below), never the layer.
        assert!(press(&mut app, egui::Key::ArrowRight, egui::Modifiers::NONE));
        assert_eq!(bounds(&app, id), photocraft_geom::Rect::new(8, 8, 24, 24));
        assert_eq!(app.session.tools.brush.angle, -1.0);
        app.run("tools.setBrush", json!({"brush": {"angle": 0}})).unwrap();
        for cmd in [CTRL, MAC_CMD] {
            assert!(press(&mut app, egui::Key::ArrowRight, cmd));
        }
        assert_eq!(bounds(&app, id), photocraft_geom::Rect::new(10, 8, 26, 24), "1 px per press");
        assert!(press(&mut app, egui::Key::ArrowDown, CTRL | egui::Modifiers::SHIFT));
        assert_eq!(bounds(&app, id), photocraft_geom::Rect::new(10, 18, 26, 34), "⌘⇧: 10 px");
        assert!(press(&mut app, egui::Key::ArrowLeft, CTRL | egui::Modifiers::ALT));
        let st = app.session.active().unwrap();
        assert_eq!(st.doc.layers.len(), n0 + 1, "⌘⌥ duplicated first");
        assert_eq!(bounds(&app, id), photocraft_geom::Rect::new(10, 18, 26, 34), "the original stays");
        assert_eq!(bounds(&app, st.active_layer.unwrap()), photocraft_geom::Rect::new(9, 18, 25, 34), "the copy moved");
        assert_eq!(app.ui.tool, Tool::Brush);
        // The Move tool itself takes ⌘-arrows as plain ones.
        app.ui.tool = Tool::Move;
        assert!(press(&mut app, egui::Key::ArrowUp, CTRL));
        let copy = app.session.active().unwrap().active_layer.unwrap();
        assert_eq!(bounds(&app, copy), photocraft_geom::Rect::new(9, 17, 25, 33));
    }

    /// #2474: tools where ⌘ means something else in Photoshop (Pen and path tools, Hand, Zoom,
    /// Crop, Type) keep ⌘-arrows away from the layer, as does an active Free Transform.
    #[test]
    fn cmd_arrows_leave_the_layer_alone_where_cmd_is_not_the_move_tool() {
        let mut app = app_with_layer();
        let id = app.session.active().unwrap().active_layer.unwrap();
        for tool in [Tool::Pen, Tool::PathSelection, Tool::DirectSelection, Tool::Rectangle, Tool::Hand, Tool::Zoom, Tool::Crop, Tool::Type] {
            app.ui.tool = tool;
            assert!(!press(&mut app, egui::Key::ArrowRight, CTRL), "{tool:?}");
            assert!(!press(&mut app, egui::Key::ArrowRight, MAC_CMD | egui::Modifiers::SHIFT), "{tool:?}");
        }
        assert_eq!(bounds(&app, id), photocraft_geom::Rect::new(8, 8, 24, 24));
        app.ui.tool = Tool::Brush;
        crate::transform_tool::begin(&mut app, &egui::Context::default()).unwrap();
        let q0 = app.ui.transform.as_ref().unwrap().quad;
        assert!(!press(&mut app, egui::Key::ArrowRight, CTRL));
        assert_eq!(app.ui.transform.as_ref().unwrap().quad, q0);
    }

    /// #2474: with a selection tool ⌘-arrows move the selected pixels (the temporary Move tool), the
    /// plain arrows the outline.
    #[test]
    fn cmd_arrows_move_the_selected_pixels_with_a_selection_tool() {
        let mut app = app_with_layer();
        app.ui.tool = Tool::RectMarquee;
        app.session.execute("select.rect", json!({"x": 8, "y": 8, "width": 8, "height": 8})).unwrap();
        assert!(press(&mut app, egui::Key::ArrowRight, CTRL));
        let offset = |app: &PhotocraftApp| photocraft_engine::float_cmds::floating(app.session.active().unwrap()).map(|f| f.offset);
        assert_eq!(offset(&app), Some((1, 0)), "the selected pixels float, 1 px to the right");
        assert!(press(&mut app, egui::Key::ArrowDown, CTRL | egui::Modifiers::SHIFT));
        assert_eq!(offset(&app), Some((1, 10)), "⌘⇧: 10 px");
        assert_eq!(app.ui.tool, Tool::RectMarquee);
        // Not while a polygon is being drawn.
        app.ui.tool = Tool::PolygonLasso;
        app.ui.polygon = vec![[1.0, 1.0]];
        assert!(!press(&mut app, egui::Key::ArrowRight, CTRL));
    }

    /// #1428: with a selection tool the arrows nudge the selection outline (⇧ 10 px), one step
    /// each; the layer stays put. Without a selection they are left alone.
    #[test]
    fn arrow_keys_nudge_the_selection_with_a_selection_tool() {
        let mut app = app_with_layer();
        let id = app.session.active().unwrap().active_layer.unwrap();
        app.ui.tool = Tool::Lasso;
        assert!(!press(&mut app, egui::Key::ArrowRight, egui::Modifiers::NONE), "no selection: not used");
        app.session.execute("select.rect", json!({"x": 4, "y": 4, "width": 10, "height": 10})).unwrap();
        let sel = |app: &PhotocraftApp| app.session.active().unwrap().doc.selection.as_ref().unwrap().content_bounds();
        let h0 = app.session.active().unwrap().history.past_len();
        assert!(press(&mut app, egui::Key::ArrowRight, egui::Modifiers::NONE));
        assert!(press(&mut app, egui::Key::ArrowDown, egui::Modifiers::SHIFT));
        assert_eq!(sel(&app), photocraft_geom::Rect::new(5, 14, 15, 24));
        assert_eq!(app.session.active().unwrap().history.past_len(), h0 + 2);
        assert_eq!(bounds(&app, id), photocraft_geom::Rect::new(8, 8, 24, 24), "pixels stay put");
        // Not while a polygon is being drawn.
        app.ui.tool = Tool::PolygonLasso;
        app.ui.polygon = vec![[1.0, 1.0]];
        assert!(!press(&mut app, egui::Key::ArrowRight, egui::Modifiers::NONE));
    }
}
