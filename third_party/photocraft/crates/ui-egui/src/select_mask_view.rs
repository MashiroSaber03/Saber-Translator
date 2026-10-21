//! Select and Mask view modes (#1754): how the live preview shows the refined selection.
//!
//! Photoshop's View Mode list: Onion Skin (O), Marching Ants (M), Overlay (V), On Black (A),
//! On White (T), Black & White (K) and On Layers (Y), with an Opacity for the first four that
//! veil the unselected area. F cycles the views, holding X shows the image without one, J toggles
//! Show Edge (the band the edge detection radius refines) and P Show Original (the selection
//! before refinement).
//!
//! The view state lives in the dialog's fields (`__view`, `__opacity`, `__showEdge`,
//! `__showOriginal`, `__hideView`), so `ui.dialog.set` drives it like any other field. The preview
//! runs `select.refineEdge` on the proxy document with `output: selection`, then builds the view
//! from that proxy (veil fill layers, a layer mask, the refined selection for the ants) and the
//! normal filter-preview path composites it.

use std::sync::Arc;

use photocraft_color::{Color, PixelFormat};
use photocraft_doc::{Document, Fill, Layer, LayerContent, LayerId, LayerMask};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::theme::Tokens;

/// The engine command behind Select › Select and Mask….
pub const COMMAND: &str = "select.refineEdge";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    OnionSkin,
    MarchingAnts,
    Overlay,
    OnBlack,
    OnWhite,
    BlackWhite,
    OnLayers,
}

impl Mode {
    /// In menu order, which is also the order F cycles through.
    pub const ALL: [Mode; 7] = [Mode::OnionSkin, Mode::MarchingAnts, Mode::Overlay, Mode::OnBlack, Mode::OnWhite, Mode::BlackWhite, Mode::OnLayers];

    pub fn id(self) -> &'static str {
        match self {
            Mode::OnionSkin => "onionSkin",
            Mode::MarchingAnts => "marchingAnts",
            Mode::Overlay => "overlay",
            Mode::OnBlack => "onBlack",
            Mode::OnWhite => "onWhite",
            Mode::BlackWhite => "blackWhite",
            Mode::OnLayers => "onLayers",
        }
    }

    pub fn from_id(id: &str) -> Option<Mode> {
        Mode::ALL.into_iter().find(|m| m.id() == id)
    }

    fn label(self) -> &'static str {
        match self {
            Mode::OnionSkin => tl!("Onion Skin"),
            Mode::MarchingAnts => tl!("Marching Ants"),
            Mode::Overlay => tl!("Overlay"),
            Mode::OnBlack => tl!("On Black"),
            Mode::OnWhite => tl!("On White"),
            Mode::BlackWhite => tl!("Black & White"),
            Mode::OnLayers => tl!("On Layers"),
        }
    }

    /// The single-letter key that picks this view.
    fn key(self) -> egui::Key {
        match self {
            Mode::OnionSkin => egui::Key::O,
            Mode::MarchingAnts => egui::Key::M,
            Mode::Overlay => egui::Key::V,
            Mode::OnBlack => egui::Key::A,
            Mode::OnWhite => egui::Key::T,
            Mode::BlackWhite => egui::Key::K,
            Mode::OnLayers => egui::Key::Y,
        }
    }

    /// The view's opacity (percent) until it is changed; `None` for views without one.
    fn default_opacity(self) -> Option<f32> {
        match self {
            Mode::OnionSkin | Mode::Overlay => Some(50.0),
            Mode::OnBlack | Mode::OnWhite => Some(100.0),
            Mode::MarchingAnts | Mode::BlackWhite | Mode::OnLayers => None,
        }
    }

    fn next(self) -> Mode {
        let i = Mode::ALL.iter().position(|m| *m == self).unwrap_or(0);
        Mode::ALL.get(i + 1).copied().unwrap_or(Mode::OnionSkin)
    }
}

/// The view the dialog shows (Onion Skin, Photoshop's default, until another is picked).
pub fn mode(fields: &Map<String, Value>) -> Mode {
    fields.get("__view").and_then(Value::as_str).and_then(Mode::from_id).unwrap_or(Mode::OnionSkin)
}

fn flag(fields: &Map<String, Value>, key: &str) -> bool {
    fields.get(key).and_then(Value::as_bool).unwrap_or(false)
}

/// The opacity (percent) of `mode`: each view keeps its own, as in Photoshop.
fn opacity(fields: &Map<String, Value>, mode: Mode) -> f32 {
    let set = fields.get("__opacity").and_then(|o| o.get(mode.id())).and_then(Value::as_f64).filter(|v| v.is_finite());
    set.map(|v| v as f32).or(mode.default_opacity()).unwrap_or(100.0).clamp(0.0, 100.0)
}

fn set_opacity(fields: &mut Map<String, Value>, mode: Mode, value: f32) {
    let mut all = fields.get("__opacity").and_then(Value::as_object).cloned().unwrap_or_default();
    all.insert(mode.id().into(), json!(value));
    fields.insert("__opacity".into(), Value::Object(all));
}

/// Holding X shows the image without the view.
pub fn hides_view(fields: &Map<String, Value>) -> bool {
    flag(fields, "__hideView")
}

/// Whether `fields` belong to the Select and Mask dialog.
pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.get("__command").and_then(Value::as_str) == Some(COMMAND)
}

/// A view key pressed in the dialog: a view letter picks that view, F the next one, J toggles
/// Show Edge and P Show Original, R the Refine Edge Brush, and [ / ] change its size. Returns
/// whether the key was one of them.
pub fn apply_key(fields: &mut Map<String, Value>, key: egui::Key) -> bool {
    if key == egui::Key::F {
        fields.insert("__view".into(), json!(mode(fields).next().id()));
    } else if let Some(m) = Mode::ALL.into_iter().find(|m| m.key() == key) {
        fields.insert("__view".into(), json!(m.id()));
    } else if key == egui::Key::R {
        fields.insert("__brush".into(), json!(!brush_on(fields)));
    } else if key == egui::Key::OpenBracket || key == egui::Key::CloseBracket {
        let size = brush_size(fields);
        let next = if key == egui::Key::OpenBracket { (size * 0.9).min(size - 1.0) } else { (size * 1.1).max(size + 1.0) };
        fields.insert("__brushSize".into(), json!(next.round().clamp(BRUSH_SIZES.0, BRUSH_SIZES.1)));
    } else if key == egui::Key::J || key == egui::Key::P {
        let field = if key == egui::Key::J { "__showEdge" } else { "__showOriginal" };
        fields.insert(field.into(), json!(!flag(fields, field)));
    } else {
        return false;
    }
    true
}

/// The view keys, unless a text field or a popup (the View dropdown) has the keyboard.
fn keys(ctx: &egui::Context, fields: &mut Map<String, Value>) {
    if ctx.text_edit_focused() || egui::Popup::is_any_open(ctx) {
        fields.insert("__hideView".into(), json!(false));
        return;
    }
    let (pressed, hold_x) = ctx.input(|i| {
        let plain = !i.modifiers.command && !i.modifiers.ctrl && !i.modifiers.alt;
        let keys: Vec<egui::Key> = i
            .events
            .iter()
            .filter_map(|e| match e {
                egui::Event::Key { key, pressed: true, repeat: false, .. } if plain => Some(*key),
                _ => None,
            })
            .collect();
        (keys, plain && i.key_down(egui::Key::X))
    });
    for key in pressed {
        apply_key(fields, key);
    }
    fields.insert("__hideView".into(), json!(hold_x));
}

/// The View Mode controls at the top of the Select and Mask dialog.
pub fn controls(ui: &mut egui::Ui, fields: &mut Map<String, Value>) {
    keys(ui.ctx(), fields);
    let t = Tokens::get(ui.ctx());
    let mut current = mode(fields);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(tl!("View")).color(t.text_dim));
        let options: Vec<(Mode, &str)> = Mode::ALL.iter().map(|m| (*m, m.label())).collect();
        crate::widgets::dropdown(ui, "select-mask-view", &mut current, &options, 170.0);
    });
    fields.insert("__view".into(), json!(current.id()));
    if current.default_opacity().is_some() {
        let mut v = opacity(fields, current);
        crate::widgets::slider_row(ui, tl!("Opacity"), &mut v, 0.0..=100.0, "%", None);
        set_opacity(fields, current, v.round());
    }
    for (field, label) in [("__showEdge", tl!("Show Edge")), ("__showOriginal", tl!("Show Original"))] {
        let mut on = flag(fields, field);
        crate::widgets::checkbox(ui, &mut on, label);
        fields.insert(field.into(), json!(on));
    }
    ui.label(egui::RichText::new(tl!("F cycles the views; hold X to see the image")).color(t.text_dim).small());
    ui.add_space(4.0);
    crate::widgets::hairline(ui);
    ui.add_space(6.0);
    brush_controls(ui, fields);
}

/// The Refine Edge Brush (R): painting on the canvas sets where edge refinement happens.
fn brush_controls(ui: &mut egui::Ui, fields: &mut Map<String, Value>) {
    let t = Tokens::get(ui.ctx());
    let mut on = brush_on(fields);
    crate::widgets::checkbox(ui, &mut on, tl!("Refine Edge Brush (R)"));
    fields.insert("__brush".into(), json!(on));
    if on {
        let mut subtract = brush_subtract(fields);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(tl!("Mode")).color(t.text_dim));
            let options = [(false, tl!("Expand Detection Area")), (true, tl!("Restore Original Edge"))];
            crate::widgets::dropdown(ui, "select-mask-brush-mode", &mut subtract, &options, 170.0);
        });
        fields.insert("__brushMode".into(), json!(if subtract { "subtract" } else { "add" }));
        let mut size = brush_size(fields);
        crate::widgets::slider_row(ui, tl!("Size"), &mut size, BRUSH_SIZES.0..=BRUSH_SIZES.1, "px", None);
        fields.insert("__brushSize".into(), json!(size.round().clamp(BRUSH_SIZES.0, BRUSH_SIZES.1)));
        ui.label(egui::RichText::new(tl!("Paint over hair or fur on the image; Alt paints the other mode")).color(t.text_dim).small());
    }
    let painted = fields.get("refineStrokes").and_then(Value::as_array).is_some_and(|a| !a.is_empty());
    if painted && crate::widgets::secondary_button(ui, tl!("Clear Brush Strokes"), 0.0).clicked() {
        fields.remove("refineStrokes");
    }
    ui.add_space(4.0);
    crate::widgets::hairline(ui);
    ui.add_space(6.0);
}

/// The Refine Edge Brush's diameters (px) in the dialog.
const BRUSH_SIZES: (f32, f32) = (1.0, 1000.0);

/// Whether the Refine Edge Brush is the dialog's tool.
pub fn brush_on(fields: &Map<String, Value>) -> bool {
    flag(fields, "__brush")
}

fn brush_size(fields: &Map<String, Value>) -> f32 {
    let size = fields.get("__brushSize").and_then(Value::as_f64).filter(|v| v.is_finite()).map_or(30.0, |v| v as f32);
    size.clamp(BRUSH_SIZES.0, BRUSH_SIZES.1)
}

/// The brush restores the original edge (Photoshop's − mode) instead of expanding the area.
fn brush_subtract(fields: &Map<String, Value>) -> bool {
    fields.get("__brushMode").and_then(Value::as_str) == Some("subtract")
}

/// Paints the Refine Edge Brush at document point `at` into the dialog's `refineStrokes`:
/// `start` begins a new stroke (in `subtract` mode or not), otherwise the point extends the last
/// one. Points closer than an eighth of the brush to the previous one are skipped, and painting
/// stops at the command's stroke and point limits.
pub fn paint(fields: &mut Map<String, Value>, at: [f64; 2], start: bool, subtract: bool) {
    if !at[0].is_finite() || !at[1].is_finite() {
        return;
    }
    let size = f64::from(brush_size(fields));
    let mut strokes = fields.get("refineStrokes").and_then(Value::as_array).cloned().unwrap_or_default();
    let points = |s: &Value| s.get("points").and_then(Value::as_array).map_or(0, Vec::len);
    if strokes.iter().map(points).sum::<usize>() >= photocraft_engine::smartselect_cmds::MAX_STROKE_POINTS {
        return;
    }
    let round = |v: f64| (v * 10.0).round() / 10.0;
    let pt = json!([round(at[0]), round(at[1])]);
    let room = strokes.len() < photocraft_engine::smartselect_cmds::MAX_STROKES;
    let last = strokes.last_mut().and_then(|s| s.get_mut("points")).and_then(Value::as_array_mut);
    match last {
        Some(pts) if !start => {
            let prev = pts.last().and_then(|p| Some((p.get(0)?.as_f64()?, p.get(1)?.as_f64()?)));
            if prev.is_none_or(|(x, y)| (x - at[0]).hypot(y - at[1]) >= (size / 8.0).max(1.0)) {
                pts.push(pt);
            }
        }
        _ if room => {
            strokes.push(json!({"points": [pt], "size": size, "mode": if subtract { "subtract" } else { "add" }}));
        }
        _ => return,
    }
    fields.insert("refineStrokes".into(), Value::Array(strokes));
}

/// The canvas under a Select and Mask dialog with the Refine Edge Brush on: shows the brush at
/// `over` (zoom = screen points per document pixel) and paints at document point `press`. False
/// when the brush isn't the top dialog's tool.
pub fn canvas_brush(app: &mut PhotocraftApp, painter: &egui::Painter, over: egui::Pos2, press: Option<[f64; 2]>, zoom: f32) -> bool {
    let ctx = painter.ctx().clone();
    let Some(d) = app.ui.dialogs.last_mut().filter(|d| owns(&d.fields) && brush_on(&d.fields)) else { return false };
    let (alt, start) = ctx.input(|i| (i.modifiers.alt, i.pointer.primary_pressed()));
    // Alt paints in the other mode, as in Photoshop.
    let subtract = brush_subtract(&d.fields) != alt;
    let r = (brush_size(&d.fields) / 2.0 * zoom).max(1.5);
    let (dark, light) = (egui::Stroke::new(1.0, egui::Color32::from_black_alpha(140)), egui::Stroke::new(1.0, egui::Color32::from_white_alpha(220)));
    painter.circle_stroke(over, r + 0.5, dark);
    painter.circle_stroke(over, r, light);
    let arm = 3.0;
    for stroke in [egui::Stroke::new(2.5, egui::Color32::from_black_alpha(140)), egui::Stroke::new(1.0, egui::Color32::from_white_alpha(220))] {
        painter.line_segment([over - egui::vec2(arm, 0.0), over + egui::vec2(arm, 0.0)], stroke);
        if !subtract {
            painter.line_segment([over - egui::vec2(0.0, arm), over + egui::vec2(0.0, arm)], stroke);
        }
    }
    ctx.set_cursor_icon(egui::CursorIcon::None);
    if let Some(at) = press {
        paint(&mut d.fields, at, start, subtract);
        ctx.request_repaint();
    }
    true
}

/// The preview request: the command's params plus the view (`__view`), so a view change
/// recomputes the preview like a parameter change.
pub fn request_params(fields: &Map<String, Value>, mut params: Value) -> Value {
    let m = mode(fields);
    if let Some(o) = params.as_object_mut() {
        o.insert(
            "__view".into(),
            json!({
                "mode": m.id(),
                "opacity": opacity(fields, m) / 100.0,
                "showEdge": flag(fields, "__showEdge"),
                "showOriginal": flag(fields, "__showOriginal"),
            }),
        );
    }
    params
}

/// What the canvas outlines while Select and Mask previews `idx`: `None` = the document's own
/// selection (no Select and Mask preview); `Some(None)` = nothing (every view but Marching Ants
/// hides the ants, as in Photoshop); `Some(Some((proxy, k)))` = the refined selection of the
/// preview proxy, reduced by `k`.
pub(crate) fn ants(app: &PhotocraftApp, idx: usize) -> Option<Option<(Arc<Document>, u32)>> {
    if app.session.active_index() != Some(idx) {
        return None;
    }
    let d = app.ui.dialogs.iter().find(|d| d.fields.contains_key("__filter"))?;
    if !owns(&d.fields) || !flag(&d.fields, "__preview") || hides_view(&d.fields) {
        return None;
    }
    if mode(&d.fields) != Mode::MarchingAnts {
        return Some(None);
    }
    let p = app.filter_preview.as_ref().filter(|p| p.key.command == COMMAND && p.key.dialog == d.id)?;
    Some(Some((p.result.clone()?, p.key.k)))
}

/// A one-channel coverage over the canvas: `values` inside `area`, `outside` elsewhere.
struct Coverage {
    area: Rect,
    values: Vec<f32>,
    outside: f32,
}

impl Coverage {
    fn empty() -> Coverage {
        Coverage { area: Rect::EMPTY, values: Vec::new(), outside: 0.0 }
    }

    /// `s` (a selection or a layer mask) read over the part of `canvas` that isn't its default.
    fn read(s: Option<&Surface>, canvas: Rect) -> Coverage {
        let Some(s) = s else { return Coverage::empty() };
        let outside = s.default_pixel().first().copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let area = if outside == 0.0 { s.content_bounds().intersect(&canvas) } else { canvas };
        Coverage { area, values: photocraft_algo::selection::mask_from_surface(Some(s), area), outside }
    }

    /// The band `r` px either side of the edge of `s`: where grown and shrunk coverage differ.
    fn edge_band(s: Option<&Surface>, canvas: Rect, r: usize) -> Coverage {
        let Some(s) = s else { return Coverage::empty() };
        let reach = i32::try_from(r).unwrap_or(i32::MAX).saturating_add(1);
        let outside = s.default_pixel().first().copied().unwrap_or(0.0);
        let area = if outside == 0.0 { s.content_bounds().inflate(reach).intersect(&canvas) } else { canvas };
        let (w, h) = (area.width() as usize, area.height() as usize);
        let values = photocraft_algo::selection::mask_from_surface(Some(s), area);
        let grown = photocraft_algo::matting::morph(&values, w, h, r, true);
        let shrunk = photocraft_algo::matting::morph(&values, w, h, r, false);
        if grown.len() != values.len() || shrunk.len() != values.len() {
            return Coverage::empty();
        }
        Coverage { area, values: grown.iter().zip(&shrunk).map(|(g, s)| (g - s).clamp(0.0, 1.0)).collect(), outside: 0.0 }
    }

    /// Coverage at canvas pixel `(x, y)`.
    fn at(&self, x: i32, y: i32) -> f32 {
        if !self.area.contains(x, y) {
            return self.outside;
        }
        let i = (y - self.area.y0) as usize * self.area.width() as usize + (x - self.area.x0) as usize;
        self.values.get(i).copied().unwrap_or(self.outside)
    }

    /// The edge band as the Refine Edge Brush changes it: painted areas join it, restored ones
    /// leave it.
    fn with_brush(self, brush: Option<&photocraft_algo::matting::EdgeBrush>, canvas: Rect) -> Coverage {
        let Some(brush) = brush else { return self };
        let area = self.area.union(&brush.bounds()).intersect(&canvas);
        let values = (area.y0..area.y1)
            .flat_map(|y| (area.x0..area.x1).map(move |x| (x, y)))
            .map(|(x, y)| {
                let (v, b) = (self.at(x, y), brush.at(x, y));
                if b >= 0.0 { v.max(b) } else { v * (1.0 + b) }
            })
            .collect();
        Coverage { area, values, outside: self.outside }
    }

    /// A GRAY8 mask surface holding `f(coverage)`.
    fn surface(&self, f: impl Fn(f32) -> f32) -> Surface {
        let g = |v: f32| f(v).clamp(0.0, 1.0);
        let mut s = Surface::with_default(PixelFormat::GRAY8, &[g(self.outside)]);
        if !self.area.is_empty() && self.values.len() == self.area.width() as usize * self.area.height() as usize {
            let bytes: Vec<u8> = self.values.iter().map(|v| (g(*v) * 255.0).round() as u8).collect();
            s.write_interleaved(self.area, &bytes);
        }
        s
    }
}

/// The view parameters a preview request carries (see [`request_params`]).
struct View {
    mode: Mode,
    opacity: f32,
    show_edge: bool,
    show_original: bool,
}

impl View {
    fn of(params: &Value) -> View {
        let v = params.get("__view");
        let b = |key: &str| v.and_then(|v| v.get(key)).and_then(Value::as_bool).unwrap_or(false);
        View {
            mode: v.and_then(|v| v.get("mode")).and_then(Value::as_str).and_then(Mode::from_id).unwrap_or(Mode::OnionSkin),
            opacity: v.and_then(|v| v.get("opacity")).and_then(Value::as_f64).filter(|o| o.is_finite()).unwrap_or(0.5).clamp(0.0, 1.0) as f32,
            show_edge: b("showEdge"),
            show_original: b("showOriginal"),
        }
    }
}

/// A pixel length param (`radius`, `feather`) of `params`, reduced by `k`.
fn length(params: &Value, key: &str, k: u32) -> f64 {
    params.get(key).and_then(Value::as_f64).filter(|v| v.is_finite()).unwrap_or(0.0).max(0.0) / f64::from(k.max(1))
}

/// The preview of the Select and Mask dialog for `doc`: the proxy reduced by `k`, shown in the
/// requested view. `None` when the refinement fails or is cancelled.
pub fn preview_document(doc: &Document, active: Option<LayerId>, params: &Value, k: u32, cancel: Option<&photocraft_engine::jobs::JobCtx>) -> Option<Document> {
    let view = View::of(params);
    let k = k.max(1);
    let mut proxy = crate::proxy::proxy_document(doc, k);
    let active = active.filter(|id| proxy.layer(*id).is_some());
    // What Select and Mask starts from: the selection, else the active layer's mask.
    let original = proxy.selection.clone().or_else(|| active.and_then(|id| proxy.layer(id)).and_then(|l| l.mask.as_ref()).map(|m| m.surface.clone()));
    let canvas = Rect::new(0, 0, i32::try_from(proxy.size.width).unwrap_or(i32::MAX), i32::try_from(proxy.size.height).unwrap_or(i32::MAX));
    let coverage = if view.show_edge {
        let band = Coverage::edge_band(original.as_ref(), canvas, length(params, "radius", k).round().clamp(1.0, 250.0) as usize);
        band.with_brush(scaled_strokes(params, k).as_ref().and_then(|s| photocraft_algo::matting::EdgeBrush::new(s, canvas)).as_ref(), canvas)
    } else if view.show_original {
        Coverage::read(original.as_ref(), canvas)
    } else {
        Coverage::read(refine(&proxy, active, params, k, cancel)?.as_ref(), canvas)
    };
    show(&mut proxy, active, &view, &coverage);
    Some(proxy)
}

/// `select.refineEdge` on the proxy, its output forced to the selection: the refined coverage.
fn refine(proxy: &Document, active: Option<LayerId>, params: &Value, k: u32, cancel: Option<&photocraft_engine::jobs::JobCtx>) -> Option<Option<Surface>> {
    let mut s = photocraft_engine::Session::new();
    s.set_inline_job_ctx(cancel.cloned());
    s.add_document(proxy.clone(), None);
    if let Some(id) = active {
        s.select_layer(id).ok()?;
    }
    let mut p = params.as_object().cloned().unwrap_or_default();
    p.remove("__view");
    p.insert("output".into(), json!("selection"));
    p.insert("decontaminate".into(), json!(false));
    for key in ["radius", "feather"] {
        if p.contains_key(key) {
            p.insert(key.into(), json!(length(params, key, k)));
        }
    }
    if let Some(strokes) = p.get("refineStrokes").and_then(Value::as_array).cloned() {
        let k = f64::from(k.max(1));
        let scale = |v: &Value| v.as_f64().map_or(v.clone(), |n| json!(n / k));
        let scaled: Vec<Value> = strokes
            .into_iter()
            .map(|mut s| {
                if let Some(o) = s.as_object_mut() {
                    if let Some(size) = o.get("size").and_then(Value::as_f64) {
                        o.insert("size".into(), json!((size / k).max(1.0)));
                    }
                    if let Some(pts) = o.get_mut("points").and_then(Value::as_array_mut) {
                        for pt in pts.iter_mut() {
                            if let Some(xy) = pt.as_array_mut() {
                                xy.iter_mut().for_each(|v| *v = scale(v));
                            }
                        }
                    }
                }
                s
            })
            .collect();
        p.insert("refineStrokes".into(), Value::Array(scaled));
    }
    s.execute(COMMAND, Value::Object(p)).ok()?;
    Some(s.active()?.doc.selection.clone())
}

/// The Refine Edge Brush strokes of `params`, reduced by `k`, for Show Edge; `None` when there
/// are none (or they are malformed: the refinement itself reports those).
fn scaled_strokes(params: &Value, k: u32) -> Option<Vec<photocraft_algo::matting::EdgeStroke>> {
    let k = k.max(1) as f32;
    params
        .get("refineStrokes")?
        .as_array()?
        .iter()
        .map(|s| {
            let points = s
                .get("points")?
                .as_array()?
                .iter()
                .map(|p| Some((p.get(0)?.as_f64()? as f32 / k, p.get(1)?.as_f64()? as f32 / k)))
                .collect::<Option<Vec<_>>>()?;
            let size = (s.get("size").and_then(Value::as_f64).unwrap_or(30.0) as f32 / k).max(1.0);
            Some(photocraft_algo::matting::EdgeStroke { points, size, add: s.get("mode").and_then(Value::as_str) != Some("subtract") })
        })
        .collect()
}

/// A solid fill layer named for the view.
fn veil(name: &str, color: Color, opacity: f32, mask: Option<Surface>) -> Layer {
    let mut l = Layer::new(name, LayerContent::Fill(Fill::Solid(color)));
    l.opacity = opacity;
    l.mask = mask.map(|surface| LayerMask { surface, ..LayerMask::reveal_all() });
    l
}

/// Turn `doc` into `view` of `coverage`.
fn show(doc: &mut Document, active: Option<LayerId>, view: &View, coverage: &Coverage) {
    let o = view.opacity;
    doc.selection = Some(coverage.surface(|v| v));
    match view.mode {
        Mode::MarchingAnts => {}
        Mode::Overlay => doc.layers.push(veil("Overlay", Color::rgb(1.0, 0.0, 0.0), o, Some(coverage.surface(|v| 1.0 - v)))),
        Mode::OnBlack => doc.layers.push(veil("On Black", Color::BLACK, o, Some(coverage.surface(|v| 1.0 - v)))),
        Mode::OnWhite => doc.layers.push(veil("On White", Color::WHITE, o, Some(coverage.surface(|v| 1.0 - v)))),
        Mode::BlackWhite => {
            doc.layers.push(veil("Black", Color::BLACK, 1.0, None));
            doc.layers.push(veil("White", Color::WHITE, 1.0, Some(coverage.surface(|v| v))));
        }
        Mode::OnLayers | Mode::OnionSkin => {
            let Some(id) = active else { return };
            if view.mode == Mode::OnionSkin
                && let Some(path) = doc.path_of(id)
            {
                // Only the layer being masked (with its groups and contents) stays visible.
                let others: Vec<LayerId> =
                    doc.walk().into_iter().filter(|(p, _, _)| !path.starts_with(p) && !p.starts_with(&path)).map(|(_, _, l)| l.id).collect();
                for other in others {
                    if let Some(l) = doc.layer_mut(other) {
                        l.visible = false;
                    }
                }
            }
            let keep = if view.mode == Mode::OnionSkin { 1.0 - o } else { 0.0 };
            let surface = coverage.surface(|v| v + (1.0 - v) * keep);
            if let Some(l) = doc.layer_mut(id) {
                l.mask = Some(LayerMask { surface, ..LayerMask::reveal_all() });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::{ColorMode, SampleType};
    use photocraft_doc::Size;

    /// A white 40×20 document with its left half selected.
    fn half_selected() -> Document {
        let mut doc = Document::with_background("d", Size::new(40, 20), ColorMode::Rgb, SampleType::U8, Color::WHITE);
        let mut sel = Surface::new(PixelFormat::GRAY8);
        sel.fill_rect(Rect::new(0, 0, 20, 20), &[1.0]);
        doc.selection = Some(sel);
        doc
    }

    fn view(mode: Mode) -> Value {
        json!({"radius": 0, "__view": {"mode": mode.id(), "opacity": 0.5, "showEdge": false, "showOriginal": false}})
    }

    fn pixel(doc: &Document, x: i32, y: i32) -> Vec<f32> {
        let s = photocraft_compose::flatten_to_surface(doc, PixelFormat::RGBA8, None);
        s.pixel(x, y)
    }

    /// Every view shows the selected half as the image and marks the rest its own way.
    #[test]
    fn each_view_marks_the_unselected_area() {
        let doc = half_selected();
        let active = doc.layers.first().map(|l| l.id);
        let near = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(a, b)| (a - b).abs() < 0.02);
        let cases = [
            (Mode::Overlay, [1.0, 0.5, 0.5, 1.0]),
            (Mode::OnBlack, [0.5, 0.5, 0.5, 1.0]),
            (Mode::OnWhite, [1.0, 1.0, 1.0, 1.0]),
            (Mode::BlackWhite, [0.0, 0.0, 0.0, 1.0]),
            (Mode::OnLayers, [0.0, 0.0, 0.0, 0.0]),
            (Mode::OnionSkin, [1.0, 1.0, 1.0, 0.5]),
            (Mode::MarchingAnts, [1.0, 1.0, 1.0, 1.0]),
        ];
        for (mode, unselected) in cases {
            // Through the filter dialog's preview, which every Select and Mask preview takes.
            let shown = crate::filter_dialog::preview_document(&doc, active, COMMAND, &view(mode), 1).expect("a preview");
            let inside = pixel(&shown, 5, 10);
            assert!(near(&inside, &[1.0, 1.0, 1.0, 1.0]), "{mode:?}: selected area {inside:?}");
            let outside = pixel(&shown, 35, 10);
            assert!(near(&outside, &unselected), "{mode:?}: unselected area {outside:?}, want {unselected:?}");
            // The refined selection rides along for the canvas's marching ants.
            assert!(shown.selection.as_ref().is_some_and(|s| s.pixel(5, 10)[0] > 0.99 && s.pixel(35, 10)[0] < 0.01), "{mode:?}");
        }
    }

    #[test]
    fn view_keys_pick_cycle_and_toggle() {
        let mut f = Map::new();
        assert_eq!(mode(&f), Mode::OnionSkin, "Photoshop's default view");
        assert!(apply_key(&mut f, egui::Key::V));
        assert_eq!(mode(&f), Mode::Overlay);
        for want in [Mode::OnBlack, Mode::OnWhite, Mode::BlackWhite, Mode::OnLayers, Mode::OnionSkin, Mode::MarchingAnts] {
            apply_key(&mut f, egui::Key::F);
            assert_eq!(mode(&f), want, "F cycles");
        }
        apply_key(&mut f, egui::Key::J);
        apply_key(&mut f, egui::Key::P);
        assert!(flag(&f, "__showEdge") && flag(&f, "__showOriginal"));
        apply_key(&mut f, egui::Key::J);
        assert!(!flag(&f, "__showEdge"));
        assert!(!apply_key(&mut f, egui::Key::Q), "other keys are left alone");
        // Each view keeps its own opacity; hostile values fall back to the default.
        set_opacity(&mut f, Mode::Overlay, 80.0);
        assert_eq!((opacity(&f, Mode::Overlay), opacity(&f, Mode::OnBlack)), (80.0, 100.0));
        f.insert("__opacity".into(), json!({"overlay": "x"}));
        assert_eq!(opacity(&f, Mode::Overlay), 50.0);
        // The view travels with the preview request, and a hostile one still previews.
        let p = request_params(&f, json!({"radius": 2}));
        assert_eq!(p["__view"]["mode"], json!("marchingAnts"));
        let doc = half_selected();
        let bad = json!({"radius": -5, "feather": "x", "__view": {"mode": 7, "opacity": "nan"}});
        assert!(preview_document(&doc, None, &bad, 3, None).is_some());
    }

    /// Refine Edge Brush strokes painted in the dialog reach the (reduced) preview: strands
    /// beyond the band join the selection, and Show Edge includes the painted area.
    #[test]
    fn brush_strokes_reach_the_preview() {
        // Warm subject (x < 40) with warm strands every 4th column from x = 60, on blue.
        let strand = |x: u32| x < 40 || ((60..84).contains(&x) && x.is_multiple_of(4));
        let mut doc = Document::with_background("d", Size::new(120, 40), ColorMode::Rgb, SampleType::U8, Color::WHITE);
        if let Some(bg) = doc.layers.first_mut().and_then(|l| l.surface_mut()) {
            let n = bg.channels();
            let data: Vec<f32> = (0..40u32)
                .flat_map(|_| (0..120u32).flat_map(move |x| if strand(x) { [0.9, 0.7, 0.2, 1.0] } else { [0.1, 0.3, 0.8, 1.0] }.into_iter().take(n)))
                .collect();
            bg.write_region(Rect::new(0, 0, 120, 40), &data);
        }
        let mut sel = Surface::new(PixelFormat::GRAY8);
        sel.fill_rect(Rect::new(0, 0, 40, 40), &[1.0]);
        doc.selection = Some(sel);
        let active = doc.layers.first().map(|l| l.id);
        let mut f = Map::new();
        apply_key(&mut f, egui::Key::R);
        assert!(brush_on(&f), "R picks the brush");
        f.insert("__brushSize".into(), json!(30));
        paint(&mut f, [70.0, 2.0], true, false);
        paint(&mut f, [70.0, 2.5], false, false);
        paint(&mut f, [70.0, 38.0], false, false);
        let strokes = f.get("refineStrokes").cloned().unwrap_or_default();
        assert_eq!(strokes, json!([{"points": [[70.0, 2.0], [70.0, 38.0]], "size": 30.0, "mode": "add"}]), "near points are skipped");
        let mut params = json!({"radius": 4, "refineStrokes": strokes, "__view": {"mode": "blackWhite"}});
        let shown = preview_document(&doc, active, &params, 2, None).expect("a preview");
        assert!(pixel(&shown, 32, 10)[0] > 0.8, "a strand at full size x = 64 joins the selection");
        assert!(pixel(&shown, 50, 10)[0] < 0.01, "the background past the strands stays out");
        params["__view"]["showEdge"] = json!(true);
        let edge = preview_document(&doc, active, &params, 2, None).expect("a preview");
        assert!(pixel(&edge, 35, 10)[0] > 0.99 && pixel(&edge, 55, 10)[0] < 0.01, "Show Edge includes the painted area");
    }

    /// Show Edge shows the band around the selection's edge; Show Original the unrefined selection.
    #[test]
    fn show_edge_and_show_original() {
        let doc = half_selected();
        let active = doc.layers.first().map(|l| l.id);
        let edge = json!({"radius": 3, "smooth": 100, "__view": {"mode": "blackWhite", "opacity": 0.5, "showEdge": true}});
        let shown = preview_document(&doc, active, &edge, 1, None).expect("a preview");
        assert!(pixel(&shown, 20, 10)[0] > 0.99, "on the edge");
        assert!(pixel(&shown, 5, 10)[0] < 0.01 && pixel(&shown, 35, 10)[0] < 0.01, "away from it");
        let original = json!({"radius": 3, "shiftEdge": 100, "__view": {"mode": "blackWhite", "showOriginal": true}});
        let shown = preview_document(&doc, active, &original, 1, None).expect("a preview");
        assert!(pixel(&shown, 19, 10)[0] > 0.99 && pixel(&shown, 21, 10)[0] < 0.01, "the selection as it was");
    }
}
