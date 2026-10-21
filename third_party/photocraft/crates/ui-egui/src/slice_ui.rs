//! Slice tool and Slice Select tool (Crop group, C): drawing new user slices, selecting and
//! moving slices, the slice overlay (outlines and numbered badges in the Preferences › Guides,
//! Grid & Slices colour) and both tools' options bars. Every change is an engine command
//! (`slice.new`, `slice.set`, `slice.promote`, `slice.divide`, `slice.delete`); this module only
//! turns pointer gestures into them.

use egui::{Color32, Rect, Stroke, vec2};
use photocraft_doc::slices::{self, ResolvedSlice};
use photocraft_doc::{Document, SliceOrigin};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::canvas::{ToolEvent, ViewXform};
use crate::state::{DialogKind, Tool};

/// Slice tool state (serde: readable through the control channel).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SliceUi {
    /// The selected slice's display number (Slice Select tool).
    pub selected: Option<usize>,
    /// Slice Select options bar › Hide Auto Slices.
    pub hide_auto: bool,
    #[serde(skip)]
    pub drag: Option<SliceDrag>,
}

/// A gesture in progress.
#[derive(Clone, Debug, PartialEq)]
pub enum SliceDrag {
    New { start: [f64; 2], cur: [f64; 2] },
    Move { id: u32, start: [f64; 2], cur: [f64; 2], rect: photocraft_geom::Rect },
}

fn is_slice_tool(t: Tool) -> bool {
    matches!(t, Tool::Slice | Tool::SliceSelect)
}

/// The slice under a document point: stored slices win (the latest on top), then auto slices.
pub fn slice_at(doc: &Document, x: f64, y: f64) -> Option<ResolvedSlice> {
    let all = slices::resolve(doc);
    let (xi, yi) = (x.floor() as i32, y.floor() as i32);
    let order = |r: &ResolvedSlice| r.id.and_then(|id| doc.slices.list.iter().position(|s| s.id == id));
    all.iter().filter(|r| r.rect.contains(xi, yi)).max_by_key(|r| order(r).map_or(0, |i| i + 1)).cloned()
}

fn rect_from(a: [f64; 2], b: [f64; 2]) -> photocraft_geom::Rect {
    let (x0, x1) = (a[0].min(b[0]).floor() as i32, a[0].max(b[0]).ceil() as i32);
    let (y0, y1) = (a[1].min(b[1]).floor() as i32, a[1].max(b[1]).ceil() as i32);
    photocraft_geom::Rect::new(x0, y0, x1, y1)
}

/// Pointer input for the two slice tools. Returns true when the event was theirs.
pub fn pointer(app: &mut PhotocraftApp, ev: ToolEvent, _mods: egui::Modifiers) -> bool {
    let tool = app.ui.tool;
    if !is_slice_tool(tool) {
        return false;
    }
    let Some(doc) = app.session.active().map(|d| d.doc.clone()) else { return true };
    let locked = app.session.file_menu.slices_locked;
    match ev {
        ToolEvent::Down { x, y, .. } => {
            if tool == Tool::Slice {
                if !locked {
                    app.ui.slices.drag = Some(SliceDrag::New { start: [x, y], cur: [x, y] });
                }
            } else {
                let hit = slice_at(&doc, x, y);
                app.ui.slices.selected = hit.as_ref().map(|r| r.number);
                if let Some(r) = hit
                    && let Some(id) = r.id.filter(|_| !locked && r.origin == SliceOrigin::User)
                {
                    app.ui.slices.drag = Some(SliceDrag::Move { id, start: [x, y], cur: [x, y], rect: r.rect });
                }
            }
        }
        ToolEvent::Move { x, y, .. } => match &mut app.ui.slices.drag {
            Some(SliceDrag::New { cur, .. }) | Some(SliceDrag::Move { cur, .. }) => *cur = [x, y],
            None => {}
        },
        ToolEvent::Up { x, y } => match app.ui.slices.drag.take() {
            Some(SliceDrag::New { start, .. }) => {
                let r = rect_from(start, [x, y]);
                if r.width() >= 1 && r.height() >= 1 && (r.width() > 2 || r.height() > 2) {
                    if let Ok(v) = app.run("slice.new", json!({"rect": [r.x0, r.y0, r.width(), r.height()]})) {
                        app.ui.slices.selected = v.get("number").and_then(Value::as_u64).map(|n| n as usize);
                    }
                } else {
                    app.ui.slices.selected = slice_at(&doc, x, y).map(|r| r.number);
                }
            }
            Some(SliceDrag::Move { id, start, rect, .. }) => {
                let (dx, dy) = ((x - start[0]).round() as i32, (y - start[1]).round() as i32);
                if dx != 0 || dy != 0 {
                    let r = rect.translate(dx, dy);
                    if app.run("slice.set", json!({"slice": id, "rect": [r.x0, r.y0, r.width(), r.height()]})).is_ok() {
                        let d = app.session.active().map(|d| d.doc.clone());
                        app.ui.slices.selected = d.and_then(|d| slices::resolve(&d).into_iter().find(|s| s.id == Some(id)).map(|s| s.number));
                    }
                }
            }
            None => {}
        },
    }
    true
}

fn hex(s: &str, fallback: Color32) -> Color32 {
    let s = s.trim_start_matches('#');
    let b = |i: usize| s.get(i..i + 2).and_then(|h| u8::from_str_radix(h, 16).ok());
    match (s.len(), b(0), b(2), b(4)) {
        (6, Some(r), Some(g), Some(bl)) => Color32::from_rgb(r, g, bl),
        _ => fallback,
    }
}

/// Slice outlines and numbered badges (View › Show › Slices; always while a slice tool is on).
pub fn draw_overlay(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    let tool_on = is_slice_tool(app.ui.tool);
    let Some(doc) = app.session.active().map(|d| d.doc.clone()) else { return };
    if !(tool_on || (app.ui.view.shows(app.ui.view.show.slices) && !doc.slices.is_empty())) {
        return;
    }
    let prefs = &app.session.prefs().guides_grid_and_slices;
    let color = hex(&prefs.slice_color, Color32::from_rgb(0x38, 0xb5, 0xff));
    let numbers = prefs.show_slice_numbers;
    let selected_color = Color32::from_rgb(0xff, 0xb0, 0x00);
    let to_screen = |r: photocraft_geom::Rect| Rect::from_two_pos(xf.to_screen(r.x0 as f32, r.y0 as f32), xf.to_screen(r.x1 as f32, r.y1 as f32));
    let all = slices::resolve(&doc);
    // Auto slices first (dotted, underneath), then user and layer slices.
    for pass in [true, false] {
        for r in all.iter().filter(|r| (r.origin == SliceOrigin::Auto) == pass) {
            if pass && (!tool_on || app.ui.slices.hide_auto) {
                continue;
            }
            let sr = to_screen(r.rect);
            let selected = app.ui.slices.selected == Some(r.number);
            if pass {
                let pts = [sr.left_top(), sr.right_top(), sr.right_bottom(), sr.left_bottom(), sr.left_top()];
                painter.add(egui::Shape::dashed_line(&pts, Stroke::new(1.0, color.gamma_multiply(0.7)), 3.0, 3.0));
            } else {
                painter.rect_stroke(sr, 0.0, Stroke::new(1.0, color), egui::StrokeKind::Inside);
            }
            if selected && tool_on {
                painter.rect_stroke(sr, 0.0, Stroke::new(2.0, selected_color), egui::StrokeKind::Inside);
                if r.origin == SliceOrigin::User {
                    for c in [
                        sr.left_top(),
                        sr.right_top(),
                        sr.left_bottom(),
                        sr.right_bottom(),
                        sr.center_top(),
                        sr.center_bottom(),
                        sr.left_center(),
                        sr.right_center(),
                    ] {
                        painter.rect_filled(Rect::from_center_size(c, vec2(5.0, 5.0)), 0.0, selected_color);
                    }
                }
            }
            if numbers && sr.width() > 18.0 && sr.height() > 12.0 {
                let label = match r.origin {
                    SliceOrigin::Layer => format!("{:02} ▪", r.number),
                    _ => format!("{:02}", r.number),
                };
                let font = egui::FontId::proportional(10.0);
                let galley = painter.layout_no_wrap(label, font, Color32::WHITE);
                let badge = Rect::from_min_size(sr.left_top() + vec2(2.0, 2.0), galley.size() + vec2(6.0, 2.0));
                let bg = if pass {
                    Color32::from_gray(110)
                } else if selected && tool_on {
                    selected_color
                } else {
                    color
                };
                painter.rect_filled(badge, 2.0, bg);
                painter.galley(badge.min + vec2(3.0, 1.0), galley, Color32::WHITE);
            }
        }
    }
    match &app.ui.slices.drag {
        Some(SliceDrag::New { start, cur }) => {
            let sr = to_screen(rect_from(*start, *cur));
            painter.rect_stroke(sr, 0.0, Stroke::new(1.0, Color32::WHITE), egui::StrokeKind::Middle);
            let pts = [sr.left_top(), sr.right_top(), sr.right_bottom(), sr.left_bottom(), sr.left_top()];
            painter.add(egui::Shape::dashed_line(&pts, Stroke::new(1.0, color), 4.0, 4.0));
        }
        Some(SliceDrag::Move { start, cur, rect, .. }) => {
            let r = rect.translate((cur[0] - start[0]).round() as i32, (cur[1] - start[1]).round() as i32);
            painter.rect_stroke(to_screen(r), 0.0, Stroke::new(2.0, selected_color), egui::StrokeKind::Inside);
        }
        None => {}
    }
}

/// The selected slice's id, promoting an auto slice first when `promote`.
fn selected_id(app: &mut PhotocraftApp, promote: bool) -> Option<u32> {
    let n = app.ui.slices.selected?;
    let doc = app.session.active()?.doc.clone();
    let r = slices::resolve(&doc).into_iter().find(|r| r.number == n)?;
    match r.id {
        Some(id) => Some(id),
        None if promote => app.run("slice.promote", json!({"number": n})).ok()?.get("slice")?.as_u64().map(|v| v as u32),
        None => None,
    }
}

/// Slice Options dialog for the selected slice (auto slices are promoted on OK).
pub fn open_options(app: &mut PhotocraftApp) -> Result<Value, String> {
    let n = app.ui.slices.selected.ok_or("select a slice with the Slice Select tool")?;
    let st = app.session.active().ok_or("no document")?;
    let r = slices::resolve(&st.doc).into_iter().find(|r| r.number == n).ok_or("no such slice")?;
    let v = photocraft_engine::slice_cmds::resolved_json(&st.doc, &r);
    let mut f = serde_json::Map::new();
    f.insert("__command".into(), json!("slice.set"));
    f.insert("__label".into(), json!("Slice Options"));
    f.insert("__form".into(), json!(true));
    f.insert("__choices".into(), json!({"kind": ["image", "noImage"]}));
    f.insert("number".into(), json!(n));
    f.insert("kind".into(), v["kind"].clone());
    for k in ["name", "url", "target", "alt", "message", "cellText"] {
        f.insert(k.into(), v[k].clone());
    }
    f.insert("background".into(), json!(v["background"].as_str().unwrap_or("none")));
    Ok(json!({"dialog": app.ui.open_dialog(DialogKind::Command, f)}))
}

/// Divide Slice dialog.
fn open_divide(app: &mut PhotocraftApp) -> Result<Value, String> {
    let n = app.ui.slices.selected.ok_or("select a slice first")?;
    let mut f = serde_json::Map::new();
    f.insert("__command".into(), json!("slice.divide"));
    f.insert("__label".into(), json!("Divide Slice"));
    f.insert("__form".into(), json!(true));
    f.insert("__choices".into(), json!({}));
    f.insert("number".into(), json!(n));
    f.insert("horizontal".into(), json!(2));
    f.insert("vertical".into(), json!(1));
    Ok(json!({"dialog": app.ui.open_dialog(DialogKind::Command, f)}))
}

/// Options bars of the Slice and Slice Select tools.
pub fn options_bar(app: &mut PhotocraftApp, ui: &mut egui::Ui, tool: Tool) -> bool {
    if !is_slice_tool(tool) {
        return false;
    }
    let has_doc = app.session.active().is_some();
    let locked = app.session.file_menu.slices_locked;
    let t = crate::theme::Tokens::get(ui.ctx());
    if tool == Tool::Slice {
        ui.label(egui::RichText::new(tl!("Style: Normal")).color(t.text_dim));
        crate::widgets::vline(ui, 22.0);
        if ui.add_enabled_ui(has_doc && !locked, |ui| crate::widgets::secondary_button(ui, tl!("Slices From Guides"), 0.0)).inner.clicked() {
            let _ = app.run("slice.fromGuides", json!({}));
        }
    } else {
        let sel = app.ui.slices.selected.is_some();
        let stored =
            sel && app.session.active().is_some_and(|d| slices::resolve(&d.doc).iter().any(|r| Some(r.number) == app.ui.slices.selected && r.id.is_some()));
        if ui.add_enabled_ui(sel && !locked, |ui| crate::widgets::secondary_button(ui, tl!("Promote"), 0.0)).inner.clicked() {
            let _ = selected_id(app, true).map(|id| app.run("slice.promote", json!({"slice": id})));
        }
        if ui.add_enabled_ui(sel && !locked, |ui| crate::widgets::secondary_button(ui, tl!("Divide…"), 0.0)).inner.clicked() {
            let _ = open_divide(app);
        }
        if ui.add_enabled_ui(stored && !locked, |ui| crate::widgets::secondary_button(ui, tl!("Delete"), 0.0)).inner.clicked()
            && let Some(id) = selected_id(app, false)
            && app.run("slice.delete", json!({"slice": id})).is_ok()
        {
            app.ui.slices.selected = None;
        }
        crate::widgets::vline(ui, 22.0);
        crate::widgets::checkbox(ui, &mut app.ui.slices.hide_auto, tl!("Hide Auto Slices"));
        crate::widgets::vline(ui, 22.0);
        if ui.add_enabled_ui(sel && !locked, |ui| crate::widgets::secondary_button(ui, tl!("Slice Options…"), 0.0)).inner.clicked() {
            let _ = open_options(app);
        }
    }
    if locked {
        ui.label(egui::RichText::new(tl!("Slices are locked (View › Lock Slices)")).color(t.text_dim).size(11.0));
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width": 100, "height": 80})).unwrap();
        app
    }

    #[test]
    fn slice_tool_draws_and_select_tool_moves() {
        let mut app = app();
        app.ui.tool = Tool::Slice;
        let m = egui::Modifiers::NONE;
        assert!(pointer(&mut app, ToolEvent::Down { x: 10.0, y: 10.0, pressure: 1.0 }, m));
        pointer(&mut app, ToolEvent::Move { x: 40.0, y: 30.0, pressure: 1.0 }, m);
        pointer(&mut app, ToolEvent::Up { x: 40.0, y: 30.0 }, m);
        let doc = app.session.active().unwrap().doc.clone();
        assert_eq!(doc.slices.list.len(), 1);
        assert_eq!(doc.slices.list[0].rect, photocraft_geom::Rect::new(10, 10, 40, 30));
        let n = app.ui.slices.selected.unwrap();
        // Slice Select: drag it by (5, 5).
        app.ui.tool = Tool::SliceSelect;
        pointer(&mut app, ToolEvent::Down { x: 20.0, y: 20.0, pressure: 1.0 }, m);
        assert_eq!(app.ui.slices.selected, Some(n));
        pointer(&mut app, ToolEvent::Up { x: 25.0, y: 25.0 }, m);
        let doc = app.session.active().unwrap().doc.clone();
        assert_eq!(doc.slices.list[0].rect, photocraft_geom::Rect::new(15, 15, 45, 35));
        // Clicking an auto slice selects it; Slice Options promotes it.
        pointer(&mut app, ToolEvent::Down { x: 90.0, y: 75.0, pressure: 1.0 }, m);
        pointer(&mut app, ToolEvent::Up { x: 90.0, y: 75.0 }, m);
        let r = open_options(&mut app).unwrap();
        let id = r["dialog"].as_u64().unwrap();
        app.ui.dialog_mut(id).unwrap().fields.insert("name".into(), json!("footer"));
        crate::dialogs::confirm(&mut app, id).unwrap();
        let doc = app.session.active().unwrap().doc.clone();
        assert!(doc.slices.list.iter().any(|s| s.name == "footer"));
    }

    #[test]
    fn locked_slices_ignore_gestures() {
        let mut app = app();
        app.run("view.lockSlices", json!({"on": true})).unwrap();
        app.ui.tool = Tool::Slice;
        let m = egui::Modifiers::NONE;
        pointer(&mut app, ToolEvent::Down { x: 10.0, y: 10.0, pressure: 1.0 }, m);
        pointer(&mut app, ToolEvent::Up { x: 40.0, y: 30.0 }, m);
        assert!(app.session.active().unwrap().doc.slices.is_empty());
    }
}
