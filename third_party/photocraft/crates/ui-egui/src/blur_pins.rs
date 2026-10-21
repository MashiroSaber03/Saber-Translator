//! Filter › Blur Gallery › Field Blur… as Photoshop's workspace (#3373): pins on the image, each
//! with its own blur, interpolated between them. While the dialog is open the image under it takes
//! the pointer: a click on the image adds a pin (and selects it), a drag moves a pin, and a drag on
//! the selected pin's ring turns its blur up or down (one turn = the full 0–500 px; the arc shows
//! the amount). Delete or Backspace removes the selected pin. The dialog shows the selected pin's
//! Blur and the Preview switch; OK runs `filter.blurGallery.fieldBlur` with the `pins`.
//!
//! Everything lives in the dialog's fields, so automation drives it with `ui.dialog.set`: `pins`
//! (`[{x, y, blur}]`, x and y as fractions of the filter's bounds) and `__selected`. The bounds are
//! the engine's: the selection's when there is one, else the canvas.

use egui::{Color32, Pos2, Stroke, vec2};
use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::canvas::ViewXform;

pub const COMMAND: &str = "filter.blurGallery.fieldBlur";
/// Field Blur's range, in pixels.
const MAX_BLUR: f64 = 500.0;
/// Photoshop's first pin.
const DEFAULT_BLUR: f64 = 15.0;
/// More pins than anyone places by hand; keeps the params and the per-pixel loop bounded.
pub const MAX_PINS: usize = 64;
/// Screen radii (points): the pin's dot, how near a press must be to grab it, and its ring.
const DOT: f32 = 5.0;
const GRAB: f32 = 9.0;
const RING: f32 = 24.0;
const RING_BAND: f32 = 5.0;

pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.get("__command").and_then(Value::as_str) == Some(COMMAND)
}

/// Turn a freshly opened Field Blur dialog into the workspace: one pin in the middle of the
/// filter's bounds with the remembered blur, selected.
pub fn init(app: &mut PhotocraftApp, fields: &mut Map<String, Value>) {
    reset(fields);
    let bounds = app.session.active().map(|st| (st.doc.id, st.doc.bounds(), st.doc.selection.clone()));
    if let Some((doc, canvas, selection)) = bounds {
        let b = selection.map(|s| app.cached_bounds(u64::MAX - doc.0, &s)).filter(|b| !b.is_empty()).unwrap_or(canvas);
        fields.insert("__bounds".into(), json!([b.x0, b.y0, b.x1, b.y1]));
    }
}

/// One pin in the middle with the dialog's blur, selected. Alt+Cancel (Reset) runs this alone, so
/// the bounds captured on opening stay.
pub fn reset(fields: &mut Map<String, Value>) {
    let blur = fields.get("blur").and_then(Value::as_f64).filter(|b| b.is_finite()).unwrap_or(DEFAULT_BLUR).clamp(0.0, MAX_BLUR);
    fields.remove("centerX");
    fields.remove("centerY");
    fields.insert("blur".into(), json!(blur));
    fields.insert("pins".into(), json!([{"x": 0.5, "y": 0.5, "blur": blur}]));
    fields.insert("__selected".into(), json!(0));
}

/// Where the dialog first opens (offset from the window's top-left): at the right edge, beside the
/// image, as Photoshop docks its Blur Tools panel, so the pins under it stay in view. Moved, it
/// reopens where it was left, like any dialog.
pub fn first_place(fields: &Map<String, Value>, ctx: &egui::Context) -> Option<egui::Vec2> {
    owns(fields).then(|| vec2((ctx.content_rect().width() - 480.0).max(0.0), 96.0))
}

/// One pin: position as fractions of the bounds, blur in pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pin {
    pub x: f64,
    pub y: f64,
    pub blur: f64,
}

/// The pins in `fields`, cleaned (finite, in range, at most [`MAX_PINS`]).
pub fn pins(fields: &Map<String, Value>) -> Vec<Pin> {
    let Some(list) = fields.get("pins").and_then(Value::as_array) else { return Vec::new() };
    let num = |v: &Value, k: &str, d: f64| v.get(k).and_then(Value::as_f64).filter(|x| x.is_finite()).unwrap_or(d);
    list.iter()
        .take(MAX_PINS)
        .map(|v| Pin { x: num(v, "x", 0.5).clamp(0.0, 1.0), y: num(v, "y", 0.5).clamp(0.0, 1.0), blur: num(v, "blur", DEFAULT_BLUR).clamp(0.0, MAX_BLUR) })
        .collect()
}

fn set_pins(fields: &mut Map<String, Value>, pins: &[Pin]) {
    let list: Vec<Value> = pins.iter().map(|p| json!({"x": p.x, "y": p.y, "blur": p.blur})).collect();
    fields.insert("pins".into(), Value::Array(list));
}

/// The selected pin's index, if it exists.
pub fn selected(fields: &Map<String, Value>) -> Option<usize> {
    let n = pins(fields).len();
    fields.get("__selected").and_then(Value::as_u64).map(|i| i as usize).filter(|i| *i < n)
}

fn select(fields: &mut Map<String, Value>, i: Option<usize>) {
    fields.insert("__selected".into(), i.map_or(Value::Null, |i| json!(i)));
    // The dialog's Blur (remembered for the next Field Blur) follows the selected pin.
    if let Some(p) = i.and_then(|i| pins(fields).get(i).copied()) {
        fields.insert("blur".into(), json!(p.blur));
    }
}

/// The filter's bounds in document pixels `[x0, y0, x1, y1]`.
fn bounds(fields: &Map<String, Value>) -> Option<[f64; 4]> {
    let b = fields.get("__bounds")?.as_array()?;
    let v = |i: usize| b.get(i).and_then(Value::as_f64);
    let r = [v(0)?, v(1)?, v(2)?, v(3)?];
    (r[2] > r[0] && r[3] > r[1]).then_some(r)
}

fn to_doc(b: [f64; 4], p: Pin) -> [f64; 2] {
    [b[0] + p.x * (b[2] - b[0]), b[1] + p.y * (b[3] - b[1])]
}

fn from_doc(b: [f64; 4], d: [f64; 2]) -> (f64, f64) {
    (((d[0] - b[0]) / (b[2] - b[0])).clamp(0.0, 1.0), ((d[1] - b[1]) / (b[3] - b[1])).clamp(0.0, 1.0))
}

/// Remove the selected pin; the one before it (or the new first) is selected.
pub fn remove_selected(fields: &mut Map<String, Value>) -> bool {
    let Some(i) = selected(fields) else { return false };
    let mut list = pins(fields);
    list.remove(i);
    set_pins(fields, &list);
    select(fields, if list.is_empty() { None } else { Some(i.saturating_sub(1)) });
    true
}

/// The dialog body: the selected pin's Blur, how to place pins, and Preview.
pub fn body(ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    let t = crate::theme::Tokens::get(ui.ctx());
    match selected(f) {
        Some(i) => {
            let mut list = pins(f);
            if let Some(pin) = list.get_mut(i) {
                let mut v = pin.blur as f32;
                if crate::widgets::slider_row(ui, tl!("Blur"), &mut v, 0.0..=MAX_BLUR as f32, "px", None).changed() {
                    pin.blur = f64::from(v.round()).clamp(0.0, MAX_BLUR);
                    set_pins(f, &list);
                    select(f, Some(i));
                }
            }
        }
        None => {
            ui.label(egui::RichText::new(tl!("No pin selected")).color(t.text_dim));
        }
    }
    ui.add_space(4.0);
    ui.add(
        egui::Label::new(
            egui::RichText::new(tl!("Click the image to add a pin; drag a pin to move it, or its ring to change its blur. Delete removes the selected pin."))
                .color(t.text_dim)
                .size(11.5),
        )
        .wrap(),
    );
    // Delete removes the selected pin, unless a field is being typed in.
    let typing = ui.ctx().memory(|m| m.focused().is_some());
    if !typing && ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Delete) || i.consume_key(egui::Modifiers::NONE, egui::Key::Backspace)) {
        remove_selected(f);
        ui.ctx().request_repaint();
    }
    if let Some(mut preview) = f.get("__preview").and_then(Value::as_bool) {
        ui.add_space(4.0);
        crate::widgets::checkbox(ui, &mut preview, tl!("Preview"));
        f.insert("__preview".into(), json!(preview));
    }
}

/// What a press at screen point `at` grabs: the selected pin's ring, a pin, or nothing (a new pin).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Grab {
    Ring(usize),
    Pin(usize),
    New,
}

fn grab(fields: &Map<String, Value>, xf: &ViewXform, at: Pos2) -> Grab {
    let Some(b) = bounds(fields) else { return Grab::New };
    let list = pins(fields);
    let screen = |p: Pin| {
        let d = to_doc(b, p);
        xf.to_screen(d[0] as f32, d[1] as f32)
    };
    if let Some(i) = selected(fields)
        && let Some(p) = list.get(i)
        && (screen(*p).distance(at) - RING).abs() <= RING_BAND
    {
        return Grab::Ring(i);
    }
    list.iter()
        .enumerate()
        .map(|(i, p)| (i, screen(*p).distance(at)))
        .filter(|(_, d)| *d <= GRAB)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(Grab::New, |(i, _)| Grab::Pin(i))
}

/// The pointer on the free image under the dialog: `over` is where it is, `press` where the
/// primary button is held (`dialogs::free_press`), both in screen points. Returns whether the
/// workspace took it.
pub fn canvas(app: &mut PhotocraftApp, ctx: &egui::Context, xf: &ViewXform, over: Pos2, press: Option<Pos2>) -> bool {
    let Some(d) = app.ui.dialogs.last_mut().filter(|d| owns(&d.fields)) else { return false };
    let f = &mut d.fields;
    let Some(b) = bounds(f) else { return false };
    let (start, down) = ctx.input(|i| (i.pointer.primary_pressed(), i.pointer.primary_down()));
    ctx.set_cursor_icon(match grab(f, xf, over) {
        Grab::Ring(_) => egui::CursorIcon::Alias,
        Grab::Pin(_) => egui::CursorIcon::Grab,
        Grab::New => egui::CursorIcon::Crosshair,
    });
    if !down && press.is_none() {
        f.remove("__drag");
        return true;
    }
    let Some(at) = press else { return true };
    let doc = xf.to_doc(at);
    let mut list = pins(f);
    if start {
        let drag = match grab(f, xf, at) {
            Grab::Ring(i) => {
                let c = list.get(i).map(|p| to_doc(b, *p)).unwrap_or(doc);
                json!({"ring": i, "angle": (doc[1] - c[1]).atan2(doc[0] - c[0])})
            }
            Grab::Pin(i) => {
                select(f, Some(i));
                let c = list.get(i).map(|p| to_doc(b, *p)).unwrap_or(doc);
                json!({"pin": i, "dx": c[0] - doc[0], "dy": c[1] - doc[1]})
            }
            Grab::New if list.len() < MAX_PINS && (b[0]..=b[2]).contains(&doc[0]) && (b[1]..=b[3]).contains(&doc[1]) => {
                // A new pin takes the selected pin's blur, as the dialog's Blur shows it.
                let blur = f.get("blur").and_then(Value::as_f64).filter(|v| v.is_finite()).unwrap_or(DEFAULT_BLUR).clamp(0.0, MAX_BLUR);
                let (x, y) = from_doc(b, doc);
                list.push(Pin { x, y, blur });
                set_pins(f, &list);
                select(f, Some(list.len() - 1));
                json!({"pin": list.len() - 1, "dx": 0.0, "dy": 0.0})
            }
            Grab::New => return true,
        };
        f.insert("__drag".into(), drag);
        ctx.request_repaint();
        return true;
    }
    let Some(drag) = f.get("__drag").cloned() else { return true };
    if let Some(i) = drag.get("pin").and_then(Value::as_u64).map(|i| i as usize)
        && let Some(p) = list.get_mut(i)
    {
        let num = |k: &str| drag.get(k).and_then(Value::as_f64).unwrap_or(0.0);
        (p.x, p.y) = from_doc(b, [doc[0] + num("dx"), doc[1] + num("dy")]);
        set_pins(f, &list);
    } else if let Some(i) = drag.get("ring").and_then(Value::as_u64).map(|i| i as usize)
        && let Some(p) = list.get_mut(i)
    {
        // The angle turned since the last frame, the short way round: clockwise raises the blur.
        let c = to_doc(b, *p);
        let a = (doc[1] - c[1]).atan2(doc[0] - c[0]);
        let last = drag.get("angle").and_then(Value::as_f64).unwrap_or(a);
        let turn = (a - last + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
        p.blur = (p.blur + turn / std::f64::consts::TAU * MAX_BLUR).clamp(0.0, MAX_BLUR);
        set_pins(f, &list);
        select(f, Some(i));
        f.insert("__drag".into(), json!({"ring": i, "angle": a}));
    }
    ctx.request_repaint();
    true
}

/// Draw the pins over the image while the workspace is open: a dot per pin (filled when
/// selected) and, round the selected one, its ring with the blur as an arc from the top.
pub fn draw(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    let Some(d) = app.ui.dialogs.last().filter(|d| owns(&d.fields)) else { return };
    let Some(b) = bounds(&d.fields) else { return };
    let sel = selected(&d.fields);
    let (dark, light) = (Stroke::new(3.0, Color32::from_black_alpha(150)), Stroke::new(1.5, Color32::from_white_alpha(235)));
    for (i, p) in pins(&d.fields).into_iter().enumerate() {
        let c = to_doc(b, p);
        let c = xf.to_screen(c[0] as f32, c[1] as f32);
        if sel == Some(i) {
            painter.circle_stroke(c, RING, dark);
            painter.circle_stroke(c, RING, Stroke::new(1.0, Color32::from_white_alpha(140)));
            let frac = (p.blur / MAX_BLUR) as f32;
            let steps = (frac * 64.0).ceil().max(1.0) as usize;
            let arc: Vec<Pos2> = (0..=steps)
                .map(|k| {
                    let a = -std::f32::consts::FRAC_PI_2 + std::f32::consts::TAU * frac * k as f32 / steps as f32;
                    c + vec2(a.cos(), a.sin()) * RING
                })
                .collect();
            if frac > 0.0 {
                painter.add(egui::Shape::line(arc.clone(), Stroke::new(4.5, Color32::from_black_alpha(150))));
                painter.add(egui::Shape::line(arc, Stroke::new(2.5, Color32::WHITE)));
            }
            painter.circle(c, DOT, Color32::WHITE, Stroke::new(1.0, Color32::from_black_alpha(180)));
        } else {
            painter.circle_stroke(c, DOT, dark);
            painter.circle_stroke(c, DOT, light);
        }
    }
}

/// Preview proxies run at 1/`k` of the document: the pins' pixel blur scales with them.
pub fn scale_for_proxy(params: &mut Value, k: u32) {
    if k <= 1 {
        return;
    }
    let Some(list) = params.get_mut("pins").and_then(Value::as_array_mut) else { return };
    for pin in list {
        if let Some(b) = pin.get("blur").and_then(Value::as_f64) {
            pin["blur"] = json!(b / f64::from(k));
        }
    }
}
