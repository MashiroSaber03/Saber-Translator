//! Painting symmetry menu, guides and an isolated, cancellable transform preview.
use crate::{
    PhotocraftApp,
    canvas::{ToolEvent, ViewXform},
    icons,
    state::Tool,
    theme::Tokens,
    widgets,
};
use egui::{Atom, Key, Stroke, vec2};
use photocraft_doc::Document;
use photocraft_engine::symmetry_cmds::PaintingSymmetry;
use photocraft_paint::symmetry::{PresetSymmetry, SymmetryMode};
use serde::Serialize;
use serde_json::json;
use std::sync::Arc;

#[derive(Clone, Debug, Serialize)]
pub struct Transform {
    pub original: PresetSymmetry,
    pub preview: PresetSymmetry,
    document: usize,
    tool: Tool,
    #[serde(skip)]
    source: Arc<Document>,
    #[serde(skip)]
    gesture: Option<Gesture>,
}
#[derive(Clone, Copy, Debug, PartialEq)]
enum Gesture {
    Move { start: [f64; 2], center: [f64; 2] },
    Rotate { angle: f64, rotation: f64 },
}

impl PartialEq for Transform {
    fn eq(&self, other: &Self) -> bool {
        self.original == other.original
            && self.preview == other.preview
            && self.document == other.document
            && self.tool == other.tool
            && self.gesture == other.gesture
            && Arc::ptr_eq(&self.source, &other.source)
    }
}

fn label(mode: SymmetryMode) -> &'static str {
    match mode {
        SymmetryMode::Vertical => tl!("Vertical"),
        SymmetryMode::Horizontal => tl!("Horizontal"),
        SymmetryMode::Dual => tl!("Dual Axis"),
        SymmetryMode::Diagonal => tl!("Diagonal"),
    }
}
fn preset(app: &PhotocraftApp) -> Option<PresetSymmetry> {
    app.session.active()?.symmetry.as_ref()?.preset()
}

pub fn can_transform(app: &PhotocraftApp) -> bool {
    matches!(app.ui.tool, Tool::Brush | Tool::Pencil | Tool::Eraser)
        && app.drag.is_none()
        && preset(app).is_some()
        && app.ui.transform.is_none()
        && app.ui.text_edit.is_none()
        && app.ui.dialogs.is_empty()
        && app.camera_raw.is_none()
        && app.distort.liquify.is_none()
        && app.distort.puppet.is_none()
        && app.distort.perspective.is_none()
        && crate::jobs_ui::job_in_view(app).is_none()
}
pub fn begin(app: &mut PhotocraftApp) -> Result<(), String> {
    if !can_transform(app) {
        return Err("no preset symmetry available to transform".into());
    }
    let original = preset(app).ok_or("no preset symmetry")?;
    let document = app.session.active_index().ok_or("no document")?;
    let source = app.session.active().ok_or("no document")?.doc.clone();
    app.ui.symmetry_transform = Some(Transform { original, preview: original, document, source, tool: app.ui.tool, gesture: None });
    Ok(())
}
pub fn commit(app: &mut PhotocraftApp) {
    if let Some(t) = app.ui.symmetry_transform.take() {
        let _ = app.run("paint.setSymmetry", json!({"mode":t.preview.mode.id(), "center":t.preview.center,"rotation":t.preview.rotation}));
    }
}
/// While the symmetry is being transformed, the options bar names the edit and gives it ✓ and ⊘
/// buttons, as Free Transform's bar does: Enter and Esc aren't the only ways out, which a pen or
/// touch user may have no key for (#4495). False (nothing drawn) when no transform is open.
pub fn options_bar(app: &mut PhotocraftApp, ui: &mut egui::Ui) -> bool {
    if app.ui.symmetry_transform.is_none() {
        return false;
    }
    ui.label(egui::RichText::new(tl!("Transform Symmetry")).color(Tokens::get(ui.ctx()).text_dim));
    widgets::vline(ui, 22.0);
    let tip = crate::i18n::fmt(tl!("Commit transform ({key})"), &[("key", &crate::shortcuts::pretty("Enter"))]);
    let commit_btn = icons::button(ui, "check", 26.0, false, &tip);
    commit_btn.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, "Commit transform"));
    if commit_btn.clicked() {
        commit(app);
    }
    let cancel = icons::button(ui, "ban", 26.0, false, tl!("Cancel transform (Esc)"));
    cancel.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, "Cancel transform"));
    if cancel.clicked() {
        app.ui.symmetry_transform = None;
    }
    true
}
/// The canvas cursor while the symmetry is being transformed: the move cursor over the centre,
/// the rotate cursor over the rotation handle and the arrow elsewhere, never the brush outline,
/// since a press there doesn't paint (#4495). `None` when no transform is open on this document.
pub fn cursor(app: &PhotocraftApp, xf: &ViewXform, p: [f64; 2]) -> Option<egui::CursorIcon> {
    let t = app.ui.symmetry_transform.as_ref().filter(|t| app.session.active_index() == Some(t.document))?;
    let size = app.session.active().map(|s| [f64::from(s.doc.size.width), f64::from(s.doc.size.height)]).unwrap_or([1.0, 1.0]);
    let tol = 8.0 / f64::from(xf.zoom.max(0.001));
    let distance = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
    Some(match t.gesture {
        Some(Gesture::Move { .. }) => egui::CursorIcon::Move,
        Some(Gesture::Rotate { .. }) => egui::CursorIcon::Alias,
        None if distance(p, t.preview.center) <= tol => egui::CursorIcon::Move,
        None if distance(p, rotation_handle(t.preview, size)) <= tol => egui::CursorIcon::Alias,
        None => egui::CursorIcon::Default,
    })
}
fn valid(app: &PhotocraftApp, t: &Transform) -> bool {
    app.session.active_index() == Some(t.document)
        && app.ui.tool == t.tool
        && can_transform(app)
        && preset(app) == Some(t.original)
        && app.session.active().is_some_and(|s| Arc::ptr_eq(&s.doc, &t.source))
}
pub fn track(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let Some(t) = &app.ui.symmetry_transform else {
        return;
    };
    if !valid(app, t) || !ctx.input(|i| i.focused) {
        app.ui.symmetry_transform = None;
        return;
    }
    // A popup owns Enter/Escape; dismissing it must not commit or cancel the transform.
    if crate::menu_nav::is_open(ctx) || egui::Popup::is_any_open(ctx) {
        return;
    }
    if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, Key::Escape)) {
        app.ui.symmetry_transform = None;
    } else if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, Key::Enter)) {
        commit(app);
    }
}

/// Highlight the active mode without reserving a separate checkmark column.
fn mode_button(ui: &mut egui::Ui, mode: SymmetryMode, checked: bool) -> egui::Response {
    let icon = ui.id().with((mode.id(), "icon"));
    let name = label(mode);
    let b = egui::Button::new((Atom::custom(icon, vec2(16.0, 16.0)), name, Atom::grow())).selected(checked).atom_ui(ui);
    let tint = Tokens::get(ui.ctx()).icon;
    if let Some(r) = b.rect(icon) {
        let glyph = match mode {
            SymmetryMode::Vertical => "symmetry-vertical",
            SymmetryMode::Horizontal => "symmetry-horizontal",
            SymmetryMode::Dual => "symmetry-dual",
            SymmetryMode::Diagonal => "symmetry-diagonal",
        };
        icons::paint(ui, r, glyph, 14.0, tint);
    }
    b.response.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Button, ui.is_enabled(), checked, name));
    b.response
}
pub fn menu(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let current = app.session.active().and_then(|s| s.symmetry.clone());
    let resp = icons::button_with_icon_size(ui, "painting-symmetry", 24.0, 14.0, current.is_some(), tl!("Set painting symmetry options"));
    let resp = crate::brush_picker::named(resp, "Set painting symmetry options");
    egui::Popup::menu(&resp).show(|ui| {
        ui.set_min_width(220.0);
        widgets::menu_scroll(ui, |ui| {
            let mut run = None;
            if ui
                .add_enabled(
                    current.is_some() && crate::menus::is_enabled(app, "paint.symmetryDisable"),
                    egui::Button::selectable(current.is_none(), tl!("Symmetry Off")),
                )
                .clicked()
            {
                run = Some(("paint.symmetryDisable", json!({})));
                ui.close();
            }
            ui.separator();
            for mode in SymmetryMode::ALL {
                let checked = current.as_ref().and_then(PaintingSymmetry::preset).is_some_and(|p| p.mode == mode);
                if ui.add_enabled_ui(crate::menus::is_enabled(app, "paint.setSymmetry"), |ui| mode_button(ui, mode, checked)).inner.clicked() {
                    run = Some(("paint.setSymmetry", json!({"mode":mode.id()})));
                    ui.close();
                }
            }
            if let Some(st) = app.session.active() {
                use crate::vector_ui::{PathRow, path_rows};
                let rows = path_rows(&st.doc, st.active_layer);
                if !rows.is_empty() {
                    ui.separator();
                }
                for row in rows {
                    let (name, text) = match row.kind {
                        PathRow::Saved => (row.name.clone(), row.name),
                        PathRow::Work => ("work".into(), tl!("Work Path").into()),
                        PathRow::Layer => ("layer".into(), row.name),
                    };
                    let checked = current.as_ref().and_then(PaintingSymmetry::path_source) == Some(name.as_str());
                    if ui.add_enabled(crate::menus::is_enabled(app, "paint.symmetryFromPath"), egui::Button::selectable(checked, text)).clicked() {
                        run = Some(("paint.symmetryFromPath", json!({"name":name})));
                        ui.close();
                    }
                }
            }
            ui.separator();
            if ui.add_enabled(can_transform(app), egui::Button::new(tl!("Transform Symmetry"))).clicked() {
                if let Err(e) = crate::menus::invoke(app, ui.ctx(), "ui.symmetryTransform", json!({})) {
                    app.ui.status = e;
                }
                ui.close();
            }
            if let Some((id, params)) = run {
                let _ = app.run(id, params);
            }
        });
    });
}

fn rotation_handle(p: PresetSymmetry, size: [f64; 2]) -> [f64; 2] {
    let d = p.directions().into_iter().next().unwrap_or([1.0, 0.0]);
    let r = (size[0].min(size[1]) * 0.35).max(1.0);
    [p.center[0] + r * d[0], p.center[1] + r * d[1]]
}
/// True throughout transform editing, including clicks outside a handle: never paint by accident.
pub fn pointer(app: &mut PhotocraftApp, ev: ToolEvent, mods: egui::Modifiers) -> bool {
    if app.ui.symmetry_transform.as_ref().is_some_and(|t| !valid(app, t)) {
        app.ui.symmetry_transform = None;
    }
    if app.ui.symmetry_transform.is_none() {
        return false;
    }
    let Some(xf) = ViewXform::active(app) else {
        return true;
    };
    let size = app.session.active().map(|s| [f64::from(s.doc.size.width), f64::from(s.doc.size.height)]).unwrap_or([1.0, 1.0]);
    let Some(t) = &mut app.ui.symmetry_transform else {
        return false;
    };
    let tol = 8.0 / f64::from(xf.zoom.max(0.001));
    let distance = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]);
    match ev {
        ToolEvent::Down { x, y, .. } => {
            let p = [x, y];
            t.gesture = if distance(p, t.preview.center) <= tol {
                Some(Gesture::Move { start: p, center: t.preview.center })
            } else if distance(p, rotation_handle(t.preview, size)) <= tol {
                Some(Gesture::Rotate { angle: (y - t.preview.center[1]).atan2(x - t.preview.center[0]), rotation: t.preview.rotation })
            } else {
                None
            };
        }
        ToolEvent::Move { x, y, .. } => {
            let (center, rotation) = match t.gesture {
                Some(Gesture::Move { start, center }) => ([center[0] + x - start[0], center[1] + y - start[1]], t.preview.rotation),
                Some(Gesture::Rotate { angle, rotation }) => {
                    let mut r = rotation + ((y - t.preview.center[1]).atan2(x - t.preview.center[0]) - angle).to_degrees();
                    if mods.shift {
                        r = (r / 15.0).round() * 15.0;
                    }
                    (t.preview.center, r)
                }
                None => return true,
            };
            if let Some(p) = PresetSymmetry::new(t.preview.mode, center, rotation) {
                t.preview = p;
            }
        }
        ToolEvent::Up { .. } => t.gesture = None,
    }
    true
}
pub fn draw(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    if !matches!(app.ui.tool, Tool::Brush | Tool::Pencil | Tool::Eraser) && app.ui.symmetry_transform.is_none() {
        return;
    }
    let Some(st) = app.session.active() else {
        return;
    };
    let edit = app.ui.symmetry_transform.as_ref().filter(|t| app.session.active_index() == Some(t.document));
    let preview = edit.map(|t| t.preview).or_else(|| preset(app));
    let Some(p) = preview else {
        return;
    };
    let size = [f64::from(st.doc.size.width), f64::from(st.doc.size.height)];
    let painter = painter.with_clip_rect(painter.clip_rect().intersect(xf.rect));
    let screen = |p: [f64; 2]| xf.to_screen(p[0] as f32, p[1] as f32);
    let t = Tokens::get(painter.ctx());
    for [a, b] in p.segments(size) {
        painter.line_segment([screen(a), screen(b)], Stroke::new(1.0, t.accent));
    }
    if edit.is_some() {
        let center = screen(p.center);
        crate::vector_ui::draw_anchor(&painter, center, true, t.accent);
        crate::vector_ui::draw_handle(&painter, center, screen(rotation_handle(p, size)), t.accent);
    }
}

#[cfg(test)]
mod tests;
