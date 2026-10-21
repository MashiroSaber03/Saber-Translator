//! Layer Style contours: the Contour row (thumbnail, preset picker, Anti-aliased) and the
//! Contour Editor. Everything edits the dialog's pending effect params, so the live preview
//! follows and OK applies it in the dialog's one `layer.layerStyle.replace`.
//!
//! Editor state lives in the dialog fields, so automation can drive it:
//! - `contourPicker`: `{effect, key}` while a preset picker is open.
//! - `contourEditor`: `{effect, key, original}` while the Contour Editor is open; `original` is
//!   the param Cancel restores.

use egui::{Color32, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, pos2, vec2};
use photocraft_doc::Contour;
use photocraft_doc::adjust::CurvePoint;
use photocraft_engine::layer_style::{CONTOURS, contour_from_param, contour_param, contour_to_shc, contours_from_shc};
use serde_json::{Map, Value, json};

use super::{entry, set_param};
use crate::state::{DialogKind, PointCurveState};
use crate::theme::Tokens;
use crate::{PhotocraftApp, point_curve, widgets};

pub(super) const PICKER: &str = "contourPicker";
pub(super) const EDITOR: &str = "contourEditor";

/// The contour a param names (Linear when it names none we know).
pub(super) fn contour_of(v: Option<&Value>) -> Contour {
    v.and_then(contour_from_param).unwrap_or_default()
}

/// A contour's points (`0..=255` levels, as the shared point editor works) and corner flags.
fn levels_of(c: &Contour) -> (Vec<[f32; 2]>, Vec<bool>) {
    match c {
        Contour::Linear => (vec![[0.0, 0.0], [255.0, 255.0]], vec![false, false]),
        Contour::Custom { points, .. } => {
            (points.iter().map(|p| [p.input * 255.0, p.output * 255.0]).collect(), (0..points.len()).map(|i| c.is_corner(i)).collect())
        }
    }
}

fn lut_of(c: &Contour) -> Vec<f32> {
    match c {
        Contour::Linear => (0..64).map(|i| i as f32 / 63.0).collect(),
        Contour::Custom { points, corners, .. } => photocraft_compose::adjust::contour_curve_lut(points, corners),
    }
}

fn sample(l: &[f32], x: f32) -> f32 {
    let i = (x.clamp(0.0, 1.0) * l.len().saturating_sub(1) as f32).round() as usize;
    l.get(i).copied().unwrap_or(0.0)
}

/// Draws `c` into `r` (field background, the curve, a border).
pub(super) fn draw(p: &egui::Painter, r: Rect, c: &Contour, line: Color32, t: &Tokens) {
    p.rect_filled(r, 0.0, t.field);
    let inner = r.shrink(2.0);
    let l = lut_of(c);
    let n = (inner.width() as usize).clamp(8, 512);
    let pts: Vec<Pos2> = (0..=n)
        .map(|k| {
            let x = k as f32 / n as f32;
            pos2(inner.left() + x * inner.width(), inner.bottom() - sample(&l, x) * inner.height())
        })
        .collect();
    p.add(egui::Shape::line(pts, Stroke::new(1.2, line)));
    p.rect_stroke(r, 0.0, Stroke::new(1.0, t.field_border), StrokeKind::Inside);
}

fn picker_target(effect: &str, key: &str) -> Value {
    json!({"effect": effect, "key": key})
}

/// Picks a preset for `effect`'s `key` (the picker's click).
pub(super) fn pick(f: &mut Map<String, Value>, effect: &str, key: &str, name: &str) {
    set_param(f, effect, key, json!(name));
    f.remove(PICKER);
}

/// The Contour row of an effect page: label, thumbnail (opens the Contour Editor), the preset
/// picker arrow and, when the effect has one, Anti-aliased.
pub(super) fn row(ui: &mut egui::Ui, f: &mut Map<String, Value>, effect: &str, key: &str, label: &str, aa_key: Option<&str>, disp: &mut Value) {
    let t = Tokens::get(ui.ctx());
    ui.horizontal(|ui| {
        if label.is_empty() {
            ui.add_space(16.0);
        } else {
            ui.label(RichText::new(tl!(label)).color(t.text_dim));
        }
        let c = contour_of(disp.get(key));
        let (r, thumb) = ui.allocate_exact_size(vec2(36.0, 36.0), Sense::click());
        draw(ui.painter(), r, &c, t.text, &t);
        if thumb.hovered() {
            ui.painter().rect_stroke(r, 0.0, Stroke::new(1.0, t.accent), StrokeKind::Outside);
        }
        if thumb.on_hover_text(tl!("Contour Editor")).clicked() {
            open_editor(f, effect, key);
        }
        let (ar, arrow) = ui.allocate_exact_size(vec2(14.0, 36.0), Sense::click());
        let (cx, s) = (ar.center(), 3.2);
        let col = if arrow.hovered() { t.text } else { t.text_dim };
        ui.painter().line_segment([cx + vec2(-s, -s * 0.5), cx + vec2(0.0, s * 0.5)], Stroke::new(1.3, col));
        ui.painter().line_segment([cx + vec2(0.0, s * 0.5), cx + vec2(s, -s * 0.5)], Stroke::new(1.3, col));
        let target = picker_target(effect, key);
        let was = f.get(PICKER) == Some(&target);
        let mut open = was;
        if arrow.clicked() {
            open = !open;
        }
        let mut chosen: Option<&str> = None;
        egui::Popup::from_response(&arrow).open_bool(&mut open).close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside).show(|ui| {
            ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
            egui::Grid::new(("contour-presets", effect, key)).spacing(vec2(4.0, 4.0)).show(ui, |ui| {
                for (i, (name, _)) in CONTOURS.iter().enumerate() {
                    let preset = photocraft_engine::layer_style::builtin_contour(name).unwrap_or_default();
                    let (r, resp) = ui.allocate_exact_size(vec2(40.0, 40.0), Sense::click());
                    draw(ui.painter(), r, &preset, t.text, &t);
                    if preset == c || resp.hovered() {
                        ui.painter().rect_stroke(r, 0.0, Stroke::new(1.5, t.accent), StrokeKind::Outside);
                    }
                    if resp.on_hover_text(tl!(name)).clicked() {
                        chosen = Some(name);
                    }
                    if i % 4 == 3 {
                        ui.end_row();
                    }
                }
            });
        });
        if let Some(name) = chosen {
            disp[key] = json!(name);
            pick(f, effect, key, name);
        } else if open != was {
            if open {
                f.insert(PICKER.into(), target);
            } else {
                f.remove(PICKER);
            }
        }
        if let Some(aa_key) = aa_key {
            let mut on = disp.get(aa_key).and_then(Value::as_bool).unwrap_or(false);
            if widgets::checkbox(ui, &mut on, tl!("Anti-aliased")).changed() {
                disp[aa_key] = json!(on);
                set_param(f, effect, aa_key, json!(on));
            }
        }
    });
}

/// Opens the Contour Editor on `effect`'s `key`, remembering the param Cancel restores.
pub(super) fn open_editor(f: &mut Map<String, Value>, effect: &str, key: &str) {
    let original = current_param(f, effect, key);
    f.remove(PICKER);
    f.insert(EDITOR.into(), json!({"effect": effect, "key": key, "original": original}));
}

fn current_param(f: &Map<String, Value>, effect: &str, key: &str) -> Value {
    entry(f, effect)
        .and_then(|e| e.get("params"))
        .and_then(|p| p.get(key))
        .cloned()
        .or_else(|| entry(f, effect).and_then(|e| e.get("kind")).and_then(Value::as_str).and_then(|k| super::defaults(k).get(key).cloned()))
        .unwrap_or_else(|| json!("Linear"))
}

fn editor_target(f: &Map<String, Value>) -> Option<(String, String)> {
    let ed = f.get(EDITOR)?;
    let effect = ed.get("effect")?.as_str()?.to_string();
    let key = ed.get("key")?.as_str()?.to_string();
    entry(f, &effect)?;
    Some((effect, key))
}

/// The edited curve: its points (`0..=255`), corner flags and name.
fn editor_curve(f: &Map<String, Value>) -> Option<(String, String, Contour)> {
    let (effect, key) = editor_target(f)?;
    let c = contour_of(Some(&current_param(f, &effect, &key)));
    Some((effect, key, c))
}

/// Stores an edited curve on the effect. Editing a built-in preset makes a "Custom" contour, so a
/// saved file doesn't call the edited curve by the preset's name.
fn store(f: &mut Map<String, Value>, effect: &str, key: &str, name: &str, pts: &[[f32; 2]], corners: &[bool]) -> bool {
    let name = if name.is_empty() || photocraft_engine::layer_style::builtin_contour(name).is_some() { "Custom" } else { name };
    let v = json!({
        "name": name,
        "points": pts.iter().map(|q| json!([q[0] / 255.0, q[1] / 255.0])).collect::<Vec<_>>(),
        "corners": corners,
    });
    let Some(c) = contour_from_param(&v) else { return false };
    set_param(f, effect, key, contour_param(&c));
    true
}

fn name_of(c: &Contour) -> &str {
    match c {
        Contour::Linear => "Linear",
        Contour::Custom { name, .. } => name,
    }
}

/// Contour Editor › Input/Output: moves point `i` to `input`/`output` percent.
pub(super) fn set_point(f: &mut Map<String, Value>, i: usize, input: f32, output: f32) -> bool {
    let Some((effect, key, c)) = editor_curve(f) else { return false };
    let (mut pts, corners) = levels_of(&c);
    point_curve::move_to(&mut pts, i, [input * 2.55, output * 2.55]) && store(f, &effect, &key, name_of(&c), &pts, &corners)
}

/// Contour Editor › Corner: whether point `i` is a corner.
pub(super) fn set_corner(f: &mut Map<String, Value>, i: usize, on: bool) -> bool {
    let Some((effect, key, c)) = editor_curve(f) else { return false };
    let (pts, mut corners) = levels_of(&c);
    let Some(flag) = corners.get_mut(i) else { return false };
    if *flag == on {
        return false;
    }
    *flag = on;
    store(f, &effect, &key, name_of(&c), &pts, &corners)
}

/// Closes the Contour Editor: OK keeps the edited curve, Cancel restores the original.
pub(super) fn close_editor(f: &mut Map<String, Value>, keep: bool) {
    if !keep
        && let Some((effect, key)) = editor_target(f)
        && let Some(original) = f.get(EDITOR).and_then(|e| e.get("original")).cloned()
    {
        set_param(f, &effect, &key, original);
    }
    f.remove(EDITOR);
}

/// Corner flags after the shared point editor added or removed a point: the flags follow their
/// points; a new point is smooth.
fn follow_corners(before: &[[f32; 2]], after: &[[f32; 2]], corners: &mut Vec<bool>) {
    let first_diff = |a: &[[f32; 2]], b: &[[f32; 2]]| (0..a.len()).find(|&i| a.get(i) != b.get(i)).unwrap_or(a.len());
    if after.len() == before.len() + 1 {
        let i = first_diff(after, before).min(corners.len());
        corners.insert(i, false);
    } else if after.len() + 1 == before.len() {
        let i = first_diff(before, after);
        if i < corners.len() {
            corners.remove(i);
        }
    }
    corners.resize(after.len(), false);
}

/// The Contour Editor (a modal over the Layer Style dialog), when one is open.
pub(super) fn editor(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    if !f.contains_key(EDITOR) {
        return;
    }
    let Some((effect, key, c)) = editor_curve(f) else {
        f.remove(EDITOR);
        return;
    };
    let t = Tokens::get(ui.ctx());
    let ctx = ui.ctx().clone();
    let mem = egui::Id::new(("contour-editor", effect.as_str(), key.as_str()));
    let mut st: PointCurveState = ctx.data(|d| d.get_temp(mem)).unwrap_or_default();
    let (mut pts, mut corners) = levels_of(&c);
    let mut close: Option<bool> = None;
    let mut load = false;
    let mut save = false;
    let mut preset: Option<String> = None;
    let modal = egui::Modal::new(mem.with("modal")).show(&ctx, |ui| {
        ui.label(RichText::new(tl!("Contour Editor")).font(crate::theme::semibold(14.0)).color(t.text));
        ui.add_space(6.0);
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(tl!("Preset:")).color(t.text_dim));
                    let mut cur = name_of(&c).to_string();
                    let opts: Vec<(String, &str)> = CONTOURS.iter().map(|(n, _)| (n.to_string(), *n)).collect();
                    if widgets::dropdown(ui, "contour-editor-preset", &mut cur, &opts, 190.0) {
                        preset = Some(cur);
                    }
                });
                ui.add_space(6.0);
                let side = 220.0;
                let (graph, resp) = ui.allocate_exact_size(vec2(side, side), Sense::click_and_drag());
                let before = pts.clone();
                let it = point_curve::interact(ui, &resp, graph, &mut pts, &mut st);
                if it.changed {
                    follow_corners(&before, &pts, &mut corners);
                    store(f, &effect, &key, name_of(&c), &pts, &corners);
                }
                let p = ui.painter_at(graph.expand(4.0));
                p.rect_filled(graph, 0.0, t.field);
                for i in 1..4 {
                    let k = i as f32 / 4.0;
                    let g = Stroke::new(1.0, t.separator);
                    p.line_segment([pos2(graph.left() + k * graph.width(), graph.top()), pos2(graph.left() + k * graph.width(), graph.bottom())], g);
                    p.line_segment([pos2(graph.left(), graph.top() + k * graph.height()), pos2(graph.right(), graph.top() + k * graph.height())], g);
                }
                let shown = Contour::Custom {
                    name: String::new(),
                    points: pts.iter().map(|q| CurvePoint { input: q[0] / 255.0, output: q[1] / 255.0 }).collect(),
                    corners: corners.clone(),
                };
                let l = lut_of(&shown);
                let line: Vec<Pos2> = (0..=256)
                    .map(|k| {
                        let x = k as f32 / 256.0;
                        pos2(graph.left() + x * graph.width(), graph.bottom() - sample(&l, x) * graph.height())
                    })
                    .collect();
                p.add(egui::Shape::line(line, Stroke::new(1.5, t.text)));
                p.rect_stroke(
                    graph,
                    0.0,
                    Stroke::new(if resp.has_focus() { 1.5 } else { 1.0 }, if resp.has_focus() { t.accent } else { t.field_border }),
                    StrokeKind::Outside,
                );
                for (i, q) in pts.iter().enumerate() {
                    let at = pos2(graph.left() + q[0] / 255.0 * graph.width(), graph.bottom() - q[1] / 255.0 * graph.height());
                    let r = Rect::from_center_size(at, vec2(7.0, 7.0));
                    let corner = corners.get(i).copied().unwrap_or(false);
                    if Some(i) == st.selected {
                        p.rect_filled(r, 0.0, t.text);
                    } else {
                        p.rect_filled(r, 0.0, t.field);
                        p.rect_stroke(r, if corner { 0.0 } else { 3.5 }, Stroke::new(1.0, t.text), StrokeKind::Inside);
                    }
                }
                ui.add_space(6.0);
                let sel = st.selected.filter(|i| *i < pts.len());
                ui.horizontal(|ui| {
                    let (mut vi, mut vo) = sel.and_then(|i| pts.get(i)).map_or((0.0, 0.0), |q| ((q[0] / 2.55).round(), (q[1] / 2.55).round()));
                    ui.label(RichText::new(tl!("Input:")).color(t.text_dim));
                    let ri = ui.add_enabled_ui(sel.is_some(), |ui| widgets::value_field(ui, &mut vi, 0.0..=100.0, "%", 52.0)).inner;
                    ui.label(RichText::new(tl!("Output:")).color(t.text_dim));
                    let ro = ui.add_enabled_ui(sel.is_some(), |ui| widgets::value_field(ui, &mut vo, 0.0..=100.0, "%", 52.0)).inner;
                    if let Some(i) = sel
                        && (ri.changed() || ro.changed())
                    {
                        set_point(f, i, vi, vo);
                    }
                    let mut corner = sel.and_then(|i| corners.get(i)).copied().unwrap_or(false);
                    let rc = ui.add_enabled_ui(sel.is_some(), |ui| widgets::checkbox(ui, &mut corner, tl!("Corner"))).inner;
                    if let Some(i) = sel
                        && rc.changed()
                    {
                        set_corner(f, i, corner);
                    }
                });
            });
            ui.add_space(10.0);
            ui.vertical(|ui| {
                if widgets::primary_button(ui, tl!("OK"), 84.0).clicked() {
                    close = Some(true);
                }
                ui.add_space(4.0);
                if widgets::secondary_button(ui, tl!("Cancel"), 84.0).clicked() {
                    close = Some(false);
                }
                ui.add_space(12.0);
                load = widgets::secondary_button(ui, tl!("Load…"), 84.0).clicked();
                ui.add_space(4.0);
                save = widgets::secondary_button(ui, tl!("Save…"), 84.0).clicked();
            });
        });
    });
    ctx.data_mut(|d| d.insert_temp(mem, st));
    if let Some(name) = preset {
        set_param(f, &effect, &key, json!(name));
    }
    if close.is_none() && modal.should_close() {
        close = Some(false);
    }
    if close.is_none() && ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Enter)) {
        close = Some(true);
    }
    if let Some(keep) = close {
        close_editor(f, keep);
        return;
    }
    if load {
        let r = app.pick_file_bytes_filtered(&["shc"], |app, name, bytes| {
            let list = contours_from_shc(&bytes).map_err(|e| format!("{}: {e}", crate::file_open::display_name(&name)))?;
            let first = list.into_iter().next().ok_or_else(|| format!("{}: no contours", crate::file_open::display_name(&name)))?;
            let d = app.ui.dialogs.iter_mut().find(|d| d.kind == DialogKind::LayerStyle && d.fields.contains_key(EDITOR)).ok_or("the Contour Editor closed")?;
            let (effect, key) = editor_target(&d.fields).ok_or("the Contour Editor closed")?;
            set_param(&mut d.fields, &effect, &key, first);
            Ok(Value::Null)
        });
        report(app, r);
    }
    if save {
        let r = contour_to_shc(&c).map_err(|e| e.to_string()).and_then(|bytes| {
            let file = format!("{}.shc", name_of(&c));
            app.pick_save(&file, move |app, path| {
                let write = app.services.write.as_mut().ok_or("no writer configured")?;
                write(&path, &bytes)?;
                Ok(json!({"path": path}))
            })
        });
        report(app, r);
    }
}

fn report(app: &mut PhotocraftApp, r: Result<Value, String>) {
    if let Err(e) = r
        && e != crate::file_dialog::CANCELLED
    {
        app.ui.status_error = true;
        app.ui.status = e;
    }
}
