//! Artboard tool (Move group, V / ⇧V cycles with the Move tool): drag on an empty spot to draw a
//! new artboard, click a board to select it, drag it to move it with its contents, and drag the
//! selected board's edge and corner handles to resize it (⇧ on a corner keeps its proportions).
//! The "+" buttons beside the selected board's sides add a board of the same size there (⌥: a copy
//! with its contents). The options bar sets the selected board's size (preset, W/H,
//! portrait/landscape) and adds a board of the same size beside the last one. Every change is one
//! engine command (`layer.new.artboard`, `layer.artboard.set`, `layer.artboard.addBeside`), so each
//! gesture is one undo step; this module only turns pointer gestures into them.

use egui::{CursorIcon, Pos2, Rect, Stroke, vec2};
use photocraft_doc::{Document, Layer, LayerId};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::canvas::{ToolEvent, ViewXform};
use crate::crop_ui::{Hit, hit};
use crate::state::Tool;
use crate::theme::Tokens;

/// How far from an edge (screen points) a press still grabs its handle.
const HANDLE_PX: f64 = 6.0;
/// A press that moves less than this (screen points) is a click, not a drawn board.
const CLICK_PX: f64 = 3.0;
/// How far the "+" buttons sit outside the selected board's edges (screen points, to their centre).
const PLUS_OFFSET_PX: f64 = 18.0;
/// The "+" buttons' radius (screen points).
const PLUS_RADIUS_PX: f64 = 8.0;
/// The sides the "+" buttons add a board on, as `layer.artboard.addBeside` names them, with the
/// direction from the board's centre.
const SIDES: [(&str, [f64; 2]); 4] = [("top", [0.0, -1.0]), ("right", [1.0, 0.0]), ("bottom", [0.0, 1.0]), ("left", [-1.0, 0.0])];

/// Artboard tool state (serde: readable through the control channel).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ArtboardToolUi {
    #[serde(skip)]
    pub drag: Option<ArtboardDrag>,
}

/// A gesture in progress. Board rects are `[x0, y0, x1, y1]` in document pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArtboardDrag {
    /// Drawing a new board from `start`.
    New { start: [f64; 2], cur: [f64; 2] },
    /// Moving board `id`, whose rect was `rect` at the press.
    Move { id: LayerId, rect: [i64; 4], start: [f64; 2], cur: [f64; 2] },
    /// Dragging handle (`hx`, `hy`) of board `id`: −1 the left/top edge, 1 the right/bottom, 0 neither.
    /// `keep`: ⇧ is down, so a corner keeps the board's proportions.
    Resize { id: LayerId, rect: [i64; 4], hx: i8, hy: i8, start: [f64; 2], cur: [f64; 2], keep: bool },
}

fn edges(r: photocraft_geom::Rect) -> [i64; 4] {
    [i64::from(r.x0), i64::from(r.y0), i64::from(r.x1), i64::from(r.y1)]
}

fn as_f64(r: [i64; 4]) -> [f64; 4] {
    r.map(|v| v as f64)
}

/// `[x, y, width, height]` for the engine's `rect` param.
fn xywh(r: [i64; 4]) -> [i64; 4] {
    [r[0], r[1], r[2].saturating_sub(r[0]), r[3].saturating_sub(r[1])]
}

/// The selected board: the active layer's artboard, with its edges.
fn active_board(app: &PhotocraftApp) -> Option<(LayerId, [i64; 4])> {
    let st = app.session.active()?;
    let id = st.doc.artboard_of(st.active_layer?)?;
    Some((id, edges(st.doc.layer(id).and_then(Layer::artboard)?.rect)))
}

/// The topmost board under a document point.
fn board_at(doc: &Document, p: [f64; 2]) -> Option<(LayerId, [i64; 4])> {
    let inside = |r: [f64; 4]| p[0] >= r[0] && p[0] < r[2] && p[1] >= r[1] && p[1] < r[3];
    doc.artboards().iter().rev().map(|b| (b.0, edges(b.2.rect))).find(|b| inside(as_f64(b.1)))
}

fn tolerance(app: &PhotocraftApp) -> f64 {
    HANDLE_PX / f64::from(app.point_zoom()).max(0.01)
}

/// The selected board's handle under a document point.
fn handle_at(app: &PhotocraftApp, p: [f64; 2]) -> Option<(LayerId, [i64; 4], i8, i8)> {
    let (id, r) = active_board(app)?;
    match hit(as_f64(r), p, tolerance(app)) {
        Hit::Handle(hx, hy) => Some((id, r, hx, hy)),
        _ => None,
    }
}

/// Each side's name and "+" button centre.
type PlusButtons = [(&'static str, [f64; 2]); 4];

/// The "+" buttons of the selected board: its id and each side's name and button centre (document
/// pixels).
fn plus_buttons(app: &PhotocraftApp) -> Option<(LayerId, PlusButtons)> {
    let (id, r) = active_board(app)?;
    let r = as_f64(r);
    let off = PLUS_OFFSET_PX / f64::from(app.point_zoom()).max(0.01);
    let (cx, cy) = ((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0);
    let (hw, hh) = ((r[2] - r[0]) / 2.0 + off, (r[3] - r[1]) / 2.0 + off);
    Some((id, SIDES.map(|(side, [dx, dy])| (side, [cx + dx * hw, cy + dy * hh]))))
}

/// The side of the "+" button under a document point.
fn plus_at(app: &PhotocraftApp, p: [f64; 2]) -> Option<(LayerId, &'static str)> {
    let (id, buttons) = plus_buttons(app)?;
    let r = PLUS_RADIUS_PX / f64::from(app.point_zoom()).max(0.01);
    buttons.iter().find(|(_, c)| (p[0] - c[0]).hypot(p[1] - c[1]) <= r).map(|(side, _)| (id, *side))
}

/// The board drawn from `a` to `b`, edges on whole pixels.
fn drawn(a: [f64; 2], b: [f64; 2]) -> [i64; 4] {
    [a[0].min(b[0]), a[1].min(b[1]), a[0].max(b[0]), a[1].max(b[1])].map(|v| v.round() as i64)
}

/// `d` rounded to whole pixels.
fn offset(start: [f64; 2], cur: [f64; 2]) -> [i64; 2] {
    [(cur[0] - start[0]).round() as i64, (cur[1] - start[1]).round() as i64]
}

/// `r` with handle (`hx`, `hy`) dragged by `d`: an edge dragged past the opposite one flips the
/// board, which never gets narrower than one pixel.
fn resized(r: [i64; 4], hx: i8, hy: i8, d: [i64; 2]) -> [i64; 4] {
    let mut e = r;
    for (h, lo, hi, dv) in [(hx, 0, 2, d[0]), (hy, 1, 3, d[1])] {
        match h {
            -1 => e[lo] = e[lo].saturating_add(dv),
            1 => e[hi] = e[hi].saturating_add(dv),
            _ => {}
        }
        let (a, b) = (e[lo].min(e[hi]), e[lo].max(e[hi]));
        (e[lo], e[hi]) = (a, b.max(a.saturating_add(1)));
    }
    e
}

/// `r` with corner (`hx`, `hy`) dragged by `d`, keeping its proportions (⇧): the opposite corner
/// stays, and the board scales by the larger of the two axes' factors.
fn resized_keep(r: [i64; 4], hx: i8, hy: i8, d: [i64; 2]) -> [i64; 4] {
    let (w, h) = ((r[2] - r[0]).max(1) as f64, (r[3] - r[1]).max(1) as f64);
    // The fixed corner, and the dragged one's offset from it.
    let (ax, ay) = (if hx > 0 { r[0] } else { r[2] }, if hy > 0 { r[1] } else { r[3] });
    let (cx, cy) = (if hx > 0 { r[2] } else { r[0] }.saturating_add(d[0]), if hy > 0 { r[3] } else { r[1] }.saturating_add(d[1]));
    let (dw, dh) = (cx.saturating_sub(ax) as f64, cy.saturating_sub(ay) as f64);
    let k = (dw.abs() / w).max(dh.abs() / h);
    let (nw, nh) = ((w * k).round().max(1.0) as i64, (h * k).round().max(1.0) as i64);
    let span = |a: i64, len: i64, toward: f64| if toward < 0.0 { [a.saturating_sub(len), a] } else { [a, a.saturating_add(len)] };
    let ([x0, x1], [y0, y1]) = (span(ax, nw, dw), span(ay, nh, dh));
    [x0, y0, x1, y1]
}

/// The board a gesture would leave, for the preview.
fn preview(d: &ArtboardDrag) -> [i64; 4] {
    match *d {
        ArtboardDrag::New { start, cur } => drawn(start, cur),
        ArtboardDrag::Move { rect, start, cur, .. } => {
            let o = offset(start, cur);
            [rect[0].saturating_add(o[0]), rect[1].saturating_add(o[1]), rect[2].saturating_add(o[0]), rect[3].saturating_add(o[1])]
        }
        ArtboardDrag::Resize { rect, hx, hy, start, cur, keep: true, .. } if hx != 0 && hy != 0 => resized_keep(rect, hx, hy, offset(start, cur)),
        ArtboardDrag::Resize { rect, hx, hy, start, cur, .. } => resized(rect, hx, hy, offset(start, cur)),
    }
}

fn report(app: &mut PhotocraftApp, r: Result<Value, String>) {
    if let Err(e) = r {
        app.ui.status = e;
        app.ui.status_error = true;
    }
}

/// Moves a gesture's pointer to `p`; ⇧ (held or released mid-drag) decides whether a corner keeps
/// the board's proportions.
fn follow(d: &mut ArtboardDrag, p: [f64; 2], mods: egui::Modifiers) {
    match d {
        ArtboardDrag::New { cur, .. } | ArtboardDrag::Move { cur, .. } => *cur = p,
        ArtboardDrag::Resize { cur, keep, .. } => (*cur, *keep) = (p, mods.shift),
    }
}

/// Pointer input for the Artboard tool. Returns true when the event was its own.
pub fn pointer(app: &mut PhotocraftApp, ev: ToolEvent, mods: egui::Modifiers) -> bool {
    if app.ui.tool != Tool::Artboard {
        return false;
    }
    let Some(doc) = app.session.active().map(|d| d.doc.clone()) else { return true };
    match ev {
        ToolEvent::Down { x, y, .. } => {
            let p = [x, y];
            // A "+" button: a board the size of the selected one on that side (⌥: a copy).
            if let Some((id, side)) = plus_at(app, p) {
                app.ui.artboard_tool.drag = None;
                let r = app.run("layer.artboard.addBeside", json!({"layer": id.0, "side": side, "duplicate": mods.alt}));
                report(app, r);
                return true;
            }
            app.ui.artboard_tool.drag = Some(if let Some((id, rect, hx, hy)) = handle_at(app, p) {
                ArtboardDrag::Resize { id, rect, hx, hy, start: p, cur: p, keep: mods.shift }
            } else if let Some((id, rect)) = board_at(&doc, p) {
                if app.session.active().and_then(|st| st.active_layer) != Some(id) {
                    let r = app.run("layer.select", json!({"layer": id.0}));
                    report(app, r);
                }
                ArtboardDrag::Move { id, rect, start: p, cur: p }
            } else {
                ArtboardDrag::New { start: p, cur: p }
            });
        }
        ToolEvent::Move { x, y, .. } => {
            if let Some(d) = &mut app.ui.artboard_tool.drag {
                follow(d, [x, y], mods);
            }
        }
        ToolEvent::Up { x, y } => {
            let Some(mut d) = app.ui.artboard_tool.drag.take() else { return true };
            follow(&mut d, [x, y], mods);
            let r = preview(&d);
            let result = match d {
                ArtboardDrag::New { start, cur } => {
                    let screen = (cur[0] - start[0]).hypot(cur[1] - start[1]) * f64::from(app.point_zoom());
                    if screen < CLICK_PX || r[2] <= r[0] || r[3] <= r[1] {
                        return true;
                    }
                    app.run("layer.new.artboard", json!({"rect": xywh(r)}))
                }
                ArtboardDrag::Move { id, rect, .. } if r != rect => app.run("layer.artboard.set", json!({"layer": id.0, "x": r[0], "y": r[1]})),
                // Resizing leaves the contents where they are, as in Photoshop.
                ArtboardDrag::Resize { id, rect, .. } if r != rect => {
                    app.run("layer.artboard.set", json!({"layer": id.0, "rect": xywh(r), "moveContents": false}))
                }
                _ => return true,
            };
            report(app, result);
        }
    }
    true
}

/// The resize pointer for handle (`hx`, `hy`), or none away from the selected board's handles.
pub fn cursor(app: &PhotocraftApp, p: [f64; 2]) -> Option<CursorIcon> {
    let (hx, hy) = match app.ui.artboard_tool.drag {
        Some(ArtboardDrag::Resize { hx, hy, .. }) => (hx, hy),
        Some(_) => return None,
        None if plus_at(app, p).is_some() => return Some(CursorIcon::PointingHand),
        None => handle_at(app, p).map(|h| (h.2, h.3))?,
    };
    Some(match (hx, hy) {
        (_, 0) => CursorIcon::ResizeHorizontal,
        (0, _) => CursorIcon::ResizeVertical,
        _ if hx == hy => CursorIcon::ResizeNwSe,
        _ => CursorIcon::ResizeNeSw,
    })
}

/// The selected board's handles, its "+" buttons, and the board a gesture in progress would leave.
pub fn draw_overlay(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    if app.ui.tool != Tool::Artboard {
        return;
    }
    let t = Tokens::get(painter.ctx());
    let pt = |x: i64, y: i64| xf.to_screen(x as f32, y as f32);
    let outline = |r: [i64; 4]| -> Vec<Pos2> { vec![pt(r[0], r[1]), pt(r[2], r[1]), pt(r[2], r[3]), pt(r[0], r[3]), pt(r[0], r[1])] };
    let drag = app.ui.artboard_tool.drag;
    if let Some(d) = drag.as_ref() {
        painter.add(egui::Shape::line(outline(preview(d)), Stroke::new(1.5, t.accent)));
    }
    let Some((_, r)) = active_board(app).filter(|_| !matches!(drag, Some(ArtboardDrag::New { .. }))) else { return };
    if drag.is_none()
        && let Some((_, buttons)) = plus_buttons(app)
    {
        let rad = PLUS_RADIUS_PX as f32;
        for (_, c) in buttons {
            let c = xf.to_screen(c[0] as f32, c[1] as f32);
            painter.circle(c, rad, t.card, Stroke::new(1.0, t.accent));
            let arm = rad * 0.5;
            let s = Stroke::new(1.5, t.accent);
            painter.line_segment([c - vec2(arm, 0.0), c + vec2(arm, 0.0)], s);
            painter.line_segment([c - vec2(0.0, arm), c + vec2(0.0, arm)], s);
        }
    }
    let r = drag.as_ref().map_or(r, preview);
    let (mx, my) = (r[0].saturating_add(r[2]) / 2, r[1].saturating_add(r[3]) / 2);
    for (x, y) in [(r[0], r[1]), (mx, r[1]), (r[2], r[1]), (r[2], my), (r[2], r[3]), (mx, r[3]), (r[0], r[3]), (r[0], my)] {
        let h = Rect::from_center_size(pt(x, y), vec2(7.0, 7.0));
        painter.rect_filled(h, 0.0, egui::Color32::WHITE);
        painter.rect_stroke(h, 0.0, Stroke::new(1.0, t.accent), egui::StrokeKind::Inside);
    }
}

/// The Artboard tool's options bar: the selected board's size preset, W/H, Make Portrait / Make
/// Landscape, and Add New Artboard.
pub fn options_bar(app: &mut PhotocraftApp, ui: &mut egui::Ui, tool: Tool) -> bool {
    if tool != Tool::Artboard {
        return false;
    }
    let t = Tokens::get(ui.ctx());
    let board = active_board(app);
    let preset = board.and_then(|(id, _)| app.session.active()?.doc.layer(id)?.artboard().map(|a| a.preset.clone()));
    let mut edit: Option<Value> = None;
    ui.add_enabled_ui(board.is_some(), |ui| {
        let (id, r) = board.unwrap_or((LayerId(0), [0, 0, 0, 0]));
        ui.label(egui::RichText::new(tl!("Size")).color(t.text_dim));
        let mut preset = preset.unwrap_or_default();
        let mut opts: Vec<(String, &str)> = vec![(String::new(), tl!("Custom"))];
        opts.extend(photocraft_engine::artboard_cmds::PRESETS.iter().map(|(n, _, _)| (n.to_string(), *n)));
        if crate::widgets::dropdown(ui, "artboard-tool-preset", &mut preset, &opts, 150.0) && !preset.is_empty() {
            edit = Some(json!({"layer": id.0, "preset": preset}));
        }
        let (mut w, mut h) = ((r[2] - r[0]) as f32, (r[3] - r[1]) as f32);
        ui.label(egui::RichText::new(tl!("W")).color(t.text_dim));
        let cw = crate::widgets::value_field(ui, &mut w, 1.0..=300_000.0, "px", 64.0).changed();
        ui.label(egui::RichText::new(tl!("H")).color(t.text_dim));
        let ch = crate::widgets::value_field(ui, &mut h, 1.0..=300_000.0, "px", 64.0).changed();
        if cw || ch {
            edit = Some(json!({"layer": id.0, "width": w.round(), "height": h.round(), "coalesce": format!("artboard-tool-wh-{}", id.0)}));
        }
        if crate::widgets::secondary_button(ui, tl!("Make Portrait"), 0.0).clicked() && w > h {
            edit = Some(json!({"layer": id.0, "width": h.round(), "height": w.round()}));
        }
        if crate::widgets::secondary_button(ui, tl!("Make Landscape"), 0.0).clicked() && h > w {
            edit = Some(json!({"layer": id.0, "width": h.round(), "height": w.round()}));
        }
    });
    if let Some(p) = edit {
        let r = app.run("layer.artboard.set", p);
        report(app, r);
    }
    crate::widgets::vline(ui, 22.0);
    let has_doc = app.session.active().is_some();
    if ui.add_enabled_ui(has_doc, |ui| crate::widgets::secondary_button(ui, tl!("Add New Artboard"), 0.0)).inner.clicked() {
        // A board the size of the selected one, right of the rightmost board (the canvas size
        // without a selected board).
        let p = board.map_or(json!({}), |(_, r)| json!({"width": r[2] - r[0], "height": r[3] - r[1]}));
        let r = app.run("layer.new.artboard", p);
        report(app, r);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::tool_event;

    fn app() -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 400, "height": 300})).unwrap();
        app.sync_views();
        app.ui.extras.snap = false;
        app.ui.view.show.smart_guides = false;
        app.ui.tool = Tool::Artboard;
        app
    }

    fn drag(app: &mut PhotocraftApp, from: [f64; 2], to: [f64; 2]) {
        let m = egui::Modifiers::NONE;
        tool_event(app, ToolEvent::Down { x: from[0], y: from[1], pressure: 1.0 }, m);
        tool_event(app, ToolEvent::Move { x: (from[0] + to[0]) / 2.0, y: (from[1] + to[1]) / 2.0, pressure: 1.0 }, m);
        tool_event(app, ToolEvent::Move { x: to[0], y: to[1], pressure: 1.0 }, m);
        tool_event(app, ToolEvent::Up { x: to[0], y: to[1] }, m);
    }

    fn boards(app: &PhotocraftApp) -> Vec<photocraft_geom::Rect> {
        app.session.active().unwrap().doc.artboards().iter().map(|b| b.2.rect).collect()
    }

    /// Drawing on empty canvas makes a board; its right-edge handle resizes it in one undo step;
    /// dragging it from inside moves it.
    #[test]
    fn artboard_tool_draws_resizes_and_moves_boards() {
        let mut app = app();
        drag(&mut app, [20.0, 30.0], [120.0, 110.0]);
        assert_eq!(boards(&app), vec![photocraft_geom::Rect::new(20, 30, 120, 110)]);
        let st = app.session.active().unwrap();
        assert_eq!(st.active_layer, Some(st.doc.artboards()[0].0));
        // The right edge's handle.
        drag(&mut app, [120.0, 70.0], [160.0, 75.0]);
        assert_eq!(boards(&app), vec![photocraft_geom::Rect::new(20, 30, 160, 110)]);
        app.run("edit.undo", json!({})).unwrap();
        assert_eq!(boards(&app), vec![photocraft_geom::Rect::new(20, 30, 120, 110)]);
        // From inside: the board moves.
        drag(&mut app, [60.0, 60.0], [70.0, 50.0]);
        assert_eq!(boards(&app), vec![photocraft_geom::Rect::new(30, 20, 130, 100)]);
        // A click on empty canvas draws nothing.
        drag(&mut app, [300.0, 250.0], [300.0, 250.0]);
        assert_eq!(boards(&app).len(), 1);
    }

    /// #3033: a click on the selected board's right "+" adds a board of its size a gap to the right;
    /// ⇧ on a corner keeps the board's proportions.
    #[test]
    fn plus_buttons_add_boards_and_shift_keeps_proportions() {
        let mut app = app();
        drag(&mut app, [20.0, 30.0], [120.0, 80.0]);
        let (_, buttons) = plus_buttons(&app).unwrap();
        let right = buttons.iter().find(|b| b.0 == "right").unwrap().1;
        assert!(right[0] > 120.0 && (right[1] - 55.0).abs() < 1e-9, "{right:?}");
        drag(&mut app, right, right);
        assert_eq!(boards(&app), vec![photocraft_geom::Rect::new(20, 30, 120, 80), photocraft_geom::Rect::new(220, 30, 320, 80)]);
        app.run("edit.undo", json!({})).unwrap();
        assert_eq!(boards(&app).len(), 1);
        // The 100×50 board's bottom-right corner, dragged 40 right and 5 down with ⇧: 140×70.
        let m = egui::Modifiers::SHIFT;
        tool_event(&mut app, ToolEvent::Down { x: 120.0, y: 80.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Move { x: 160.0, y: 85.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Up { x: 160.0, y: 85.0 }, m);
        assert_eq!(boards(&app), vec![photocraft_geom::Rect::new(20, 30, 160, 100)]);
    }

    /// V picks the Move group, ⇧V cycles Move ↔ Artboard as in Photoshop.
    #[test]
    fn shift_v_cycles_move_and_artboard() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        let ctx = egui::Context::default();
        let press = |app: &mut PhotocraftApp, modifiers| {
            let key = egui::Key::V;
            let raw = egui::RawInput {
                events: vec![
                    egui::Event::ModifiersChanged(modifiers),
                    egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers },
                    egui::Event::Key { key, physical_key: None, pressed: false, repeat: false, modifiers },
                ],
                ..Default::default()
            };
            ctx.begin_pass(raw);
            crate::shortcuts::handle(app, &ctx);
            ctx.end_pass().textures_delta.clear();
        };
        app.ui.tool = Tool::Brush;
        press(&mut app, egui::Modifiers::NONE);
        assert_eq!(app.ui.tool, Tool::Move);
        press(&mut app, egui::Modifiers::SHIFT);
        assert_eq!(app.ui.tool, Tool::Artboard);
        press(&mut app, egui::Modifiers::SHIFT);
        assert_eq!(app.ui.tool, Tool::Move);
    }
}
