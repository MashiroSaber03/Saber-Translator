//! Create Rectangle / Ellipse / Triangle / Polygon: a single click (no drag) with a Shape tool
//! opens Photoshop's creation dialog instead of drawing. OK runs the same `shape.create` command as
//! a drag, with the options bar's fill and stroke, at the clicked point (or centred on it).
//! Width and height are stored in document pixels (so automation can set them with
//! `ui.dialog.set`) and shown in the ruler unit.

use egui::{Align2, Sense, pos2, vec2};
use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::state::{DialogKind, Tool};
use crate::theme::Tokens;

const MARK: &str = "__createShape";
/// Largest side the dialog accepts, in pixels (Photoshop's document limit).
const MAX_PX: f64 = 300_000.0;
/// Remembered values (document pixels; sides and radii come from the options bar).
const KEYS: [&str; 4] = ["width", "height", "fromCenter", "starRatio"];

pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.contains_key(MARK)
}

/// The tool's name in the dialog fields and the title, for the tools that have a dialog.
fn kind_of(tool: Tool) -> Option<(&'static str, &'static str)> {
    Some(match tool {
        Tool::Rectangle => ("rectangle", "Create Rectangle"),
        Tool::EllipseShape => ("ellipse", "Create Ellipse"),
        Tool::Triangle => ("triangle", "Create Triangle"),
        Tool::Polygon => ("polygon", "Create Polygon"),
        _ => return None,
    })
}

fn pref_key(kind: &str) -> String {
    format!("shape.create.{kind}")
}

fn valid(key: &str, v: &Value) -> bool {
    match key {
        "width" | "height" => v.as_f64().is_some_and(|x| x.is_finite() && (1.0..=MAX_PX).contains(&x)),
        "starRatio" => v.as_f64().is_some_and(|x| x.is_finite() && (1.0..=100.0).contains(&x)),
        _ => v.is_boolean(),
    }
}

/// True for a press and release that never left `start` by more than a few screen pixels.
pub fn is_click(start: [f64; 2], points: &[[f64; 3]], zoom: f32) -> bool {
    let slop = 3.0 / f64::from(zoom.max(0.01));
    points.iter().all(|p| (p[0] - start[0]).hypot(p[1] - start[1]) <= slop)
}

/// Open the creation dialog of `tool` for a click at `at` (document px). `None` for a tool without
/// one (Line, Custom Shape).
pub fn open(app: &mut PhotocraftApp, tool: Tool, at: [f64; 2]) -> Option<u64> {
    let (kind, label) = kind_of(tool)?;
    if !(at[0].is_finite() && at[1].is_finite()) {
        return None;
    }
    let o = &app.ui.tool_options;
    let r = f64::from(o.corner_radius.max(0.0));
    let sides = o.polygon_sides.clamp(3, 100);
    let mut f = Map::new();
    f.insert(MARK.into(), json!(kind));
    f.insert("__command".into(), json!("shape.create"));
    f.insert("__label".into(), json!(label));
    f.insert("__at".into(), json!([at[0].round(), at[1].round()]));
    f.insert("width".into(), json!(100.0));
    f.insert("height".into(), json!(100.0));
    f.insert("fromCenter".into(), json!(false));
    match kind {
        "rectangle" => {
            f.insert("radii".into(), json!([r, r, r, r]));
            f.insert("linkRadii".into(), json!(true));
        }
        "polygon" => {
            f.insert("sides".into(), json!(sides));
            f.insert("starRatio".into(), json!(100.0));
        }
        _ => {}
    }
    if let Some(Value::Object(saved)) = app.session.prefs().dialogs.get(&pref_key(kind)) {
        for key in KEYS {
            if f.contains_key(key)
                && let Some(v) = saved.get(key).filter(|v| valid(key, v))
            {
                f.insert(key.into(), v.clone());
            }
        }
    }
    Some(app.ui.open_dialog(DialogKind::Command, f))
}

fn num(f: &Map<String, Value>, key: &str) -> Option<f64> {
    f.get(key).and_then(Value::as_f64).filter(|v| v.is_finite())
}

/// `shape.create` geometry for the dialog's values (no fill or stroke).
pub fn geometry(f: &Map<String, Value>) -> Result<Value, String> {
    let kind = f.get(MARK).and_then(Value::as_str).ok_or("not a Create Shape dialog")?;
    let at = f.get("__at").and_then(Value::as_array).ok_or("the dialog has no position")?;
    let (Some(x), Some(y)) = (at.first().and_then(Value::as_f64), at.get(1).and_then(Value::as_f64)) else {
        return Err("the dialog has no position".into());
    };
    let side = |key: &str, label: &str| -> Result<f64, String> {
        num(f, key).filter(|v| (1.0..=MAX_PX).contains(v)).ok_or_else(|| format!("{label} must be between 1 and {MAX_PX} pixels"))
    };
    let (w, h) = (side("width", "Width")?, side("height", "Height")?);
    let center = f.get("fromCenter").and_then(Value::as_bool).unwrap_or(false);
    if !(x.abs() <= MAX_PX && y.abs() <= MAX_PX) {
        return Err("the shape's position is out of range".into());
    }
    let (x0, y0) = if center { (x - w / 2.0, y - h / 2.0) } else { (x, y) };
    let rect = json!([x0, y0, w, h]);
    Ok(match kind {
        "rectangle" => {
            let mut radii = [0.0; 4];
            if let Some(a) = f.get("radii").and_then(Value::as_array) {
                for (r, v) in radii.iter_mut().zip(a) {
                    *r = v.as_f64().filter(|v| v.is_finite()).unwrap_or(0.0).clamp(0.0, w.min(h) / 2.0);
                }
            }
            if radii.iter().any(|r| *r > 0.0) { json!({"kind": "roundedRect", "rect": rect, "radii": radii}) } else { json!({"kind": "rect", "rect": rect}) }
        }
        "ellipse" => json!({"kind": "ellipse", "rect": rect}),
        "triangle" => json!({"kind": "polygon", "rect": rect, "sides": 3}),
        "polygon" => {
            let sides = num(f, "sides").map_or(5.0, |s| s.round().clamp(3.0, 100.0));
            let ratio = num(f, "starRatio").map_or(100.0, |r| r.clamp(1.0, 100.0));
            json!({"kind": "polygon", "rect": rect, "sides": sides, "starRatio": ratio / 100.0})
        }
        other => return Err(format!("unknown shape `{other}`")),
    })
}

/// OK: create the shape and remember the values for the next click.
pub fn confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    let result = geometry(f).and_then(|g| crate::vector_ui::create_shape(app, g));
    match &result {
        Ok(_) => {
            if let Some(kind) = f.get(MARK).and_then(Value::as_str) {
                let saved: Map<String, Value> = KEYS.iter().filter_map(|k| f.get(*k).map(|v| (k.to_string(), v.clone()))).collect();
                app.session.prefs.edit(|p| p.dialogs.insert(pref_key(kind), Value::Object(saved)));
            }
        }
        Err(e) => {
            app.ui.status = e.clone();
            app.ui.status_error = true;
        }
    }
    result
}

fn label(ui: &mut egui::Ui, text: &str) {
    let t = Tokens::get(ui.ctx());
    let (r, _) = ui.allocate_exact_size(vec2(96.0, 22.0), Sense::hover());
    ui.painter().text(pos2(r.right() - 6.0, r.center().y), Align2::RIGHT_CENTER, text, egui::FontId::proportional(12.0), t.text_dim);
}

pub fn body(app: &PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    ui.spacing_mut().item_spacing.y = 6.0;
    let kind = f.get(MARK).and_then(Value::as_str).unwrap_or("").to_string();
    // Width and height in the ruler unit (percent of the document's side).
    let ur = &app.session.prefs().units_and_rulers;
    let (unit, ppi) = (ur.rulers, ur.point_size.per_inch());
    let (dpi, size) = app
        .session
        .active()
        .map_or((72.0, [1.0, 1.0]), |d| (f64::from(d.doc.resolution_dpi.max(1.0)), [f64::from(d.doc.size.width), f64::from(d.doc.size.height)]));
    for (key, text, extent) in [("width", tl!("Width:"), size[0]), ("height", tl!("Height:"), size[1])] {
        ui.horizontal(|ui| {
            label(ui, text);
            let px = num(f, key).unwrap_or(100.0);
            let to_unit = |px: f64| unit.from_px(px, dpi, extent, ppi) as f32;
            let mut v = to_unit(px);
            let places = unit.decimals() as u32;
            if crate::widgets::value_field_prec(ui, &mut v, to_unit(1.0)..=to_unit(MAX_PX), unit.suffix(), 110.0, places).changed() {
                let px = unit.to_px(f64::from(v), dpi, extent, ppi).clamp(1.0, MAX_PX);
                f.insert(key.into(), json!((px * 1000.0).round() / 1000.0));
            }
        });
    }
    match kind.as_str() {
        "rectangle" => {
            let mut radii = [0.0f32; 4];
            if let Some(a) = f.get("radii").and_then(Value::as_array) {
                for (r, v) in radii.iter_mut().zip(a) {
                    *r = v.as_f64().unwrap_or(0.0) as f32;
                }
            }
            let mut linked = f.get("linkRadii").and_then(Value::as_bool).unwrap_or(true);
            let mut changed = None;
            ui.horizontal(|ui| {
                label(ui, tl!("Radius:"));
                ui.vertical(|ui| {
                    for row in [[0usize, 1], [3, 2]] {
                        ui.horizontal(|ui| {
                            for i in row {
                                if crate::widgets::value_field(ui, &mut radii[i], 0.0..=10000.0, "px", 72.0).changed() {
                                    changed = Some(i);
                                }
                            }
                        });
                    }
                });
            });
            ui.horizontal(|ui| {
                label(ui, "");
                if crate::widgets::checkbox(ui, &mut linked, tl!("Link radii")).changed() {
                    f.insert("linkRadii".into(), json!(linked));
                }
            });
            if let Some(i) = changed {
                if linked {
                    radii = [radii[i]; 4];
                }
                f.insert("radii".into(), json!(radii.map(|r| f64::from(r.max(0.0)))));
            }
        }
        "polygon" => {
            ui.horizontal(|ui| {
                label(ui, tl!("Sides:"));
                let mut s = num(f, "sides").unwrap_or(5.0) as f32;
                if crate::widgets::value_field(ui, &mut s, 3.0..=100.0, "", 72.0).changed() {
                    f.insert("sides".into(), json!(s.round().clamp(3.0, 100.0) as u32));
                }
            });
            ui.horizontal(|ui| {
                label(ui, tl!("Star Ratio:"));
                let mut r = num(f, "starRatio").unwrap_or(100.0) as f32;
                if crate::widgets::value_field(ui, &mut r, 1.0..=100.0, "%", 72.0).changed() {
                    f.insert("starRatio".into(), json!(f64::from(r.clamp(1.0, 100.0))));
                }
            });
        }
        _ => {}
    }
    ui.horizontal(|ui| {
        label(ui, "");
        let mut center = f.get("fromCenter").and_then(Value::as_bool).unwrap_or(false);
        if crate::widgets::checkbox(ui, &mut center, tl!("From Center")).changed() {
            f.insert("fromCenter".into(), json!(center));
        }
    });
}

#[cfg(test)]
#[path = "shape_dialog_tests.rs"]
mod tests;
