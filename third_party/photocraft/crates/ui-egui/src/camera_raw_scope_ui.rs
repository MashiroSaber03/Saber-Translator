//! Camera Raw scope interaction. View state is in UiState; filter edits still commit through
//! the existing filter.cameraRaw command. Image analysis is cached per completed proxy revision.
use crate::{camera_raw_ui::CameraRawDialog, state::CameraRawScopeState, theme::Tokens, widgets};
use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, TextureHandle, pos2, vec2};
use photocraft_algo::{
    camera_raw::CameraRaw,
    histogram::clipping,
    vectorscope::{HUES, SATURATIONS, Vectorscope},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Warning combinations and Alt diagnostics are separate modes: both warnings must not
/// accidentally select the shadow-channel diagnostic (the old numeric value was shared).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClippingMode {
    None,
    Warnings(u8),
    ShadowChannels,
    HighlightChannels,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum ToneZone {
    Blacks,
    Shadows,
    Exposure,
    Highlights,
    Whites,
}
impl ToneZone {
    fn at(x: f32) -> Self {
        if x < 0.1 {
            Self::Blacks
        } else if x < 0.25 {
            Self::Shadows
        } else if x < 0.75 {
            Self::Exposure
        } else if x < 0.9 {
            Self::Highlights
        } else {
            Self::Whites
        }
    }
    /// English source label, shared with the Light slider.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Blacks => "Blacks",
            Self::Shadows => "Shadows",
            Self::Exposure => "Exposure",
            Self::Highlights => "Highlights",
            Self::Whites => "Whites",
        }
    }
    fn label(self) -> &'static str {
        crate::i18n::tr_ctx(crate::i18n::current(), "cameraRaw", self.name())
    }
    fn range(self) -> [f32; 2] {
        match self {
            Self::Blacks => [0.0, 0.1],
            Self::Shadows => [0.1, 0.25],
            Self::Exposure => [0.25, 0.75],
            Self::Highlights => [0.75, 0.9],
            Self::Whites => [0.9, 1.0],
        }
    }
    fn value(self, p: &mut CameraRaw) -> &mut f32 {
        match self {
            Self::Blacks => &mut p.blacks,
            Self::Shadows => &mut p.shadows,
            Self::Exposure => &mut p.exposure,
            Self::Highlights => &mut p.highlights,
            Self::Whites => &mut p.whites,
        }
    }
    pub(crate) fn limit(self) -> f32 {
        if self == Self::Exposure { 5.0 } else { 100.0 }
    }
    fn set(self, d: &mut CameraRawDialog, value: f32) {
        let value = value.clamp(-self.limit(), self.limit());
        let target = self.value(&mut d.params);
        if *target != value {
            *target = value;
            d.dirty = true;
        }
    }
}

/// Most colour samplers (probes) on the preview.
pub(crate) const MAX_SAMPLERS: usize = 9;
/// Vectorscope analysis samples at most this many proxy pixels.
const SCOPE_SAMPLES: usize = 8192;

/// Scope caches and gesture state of one open dialog; never filter parameters or history.
#[derive(Default)]
pub(crate) struct ScopeView {
    /// Vectorscope keyed by (preview revision, Before, selected region).
    pub(crate) cache: Option<(u64, bool, bool, Vectorscope)>,
    pub(crate) ms: f64,
    pub(crate) revision: u64,
    pub(crate) overlay_tex: Option<TextureHandle>,
    pub(crate) overlay_detail: bool,
    pub(crate) overlay_key: Option<(u64, bool, ClippingMode, crate::theme::ThemeKind)>,
    pub(crate) floating_dragged: bool,
    pub(crate) geometry_requested: bool,
    pub(crate) sampler_drag: Option<usize>,
    pub(crate) pointer_sample: Option<[f32; 2]>,
    /// Histogram drag: zone, pointer start x and the value at the start.
    pub(crate) tone_drag: Option<(ToneZone, f32, f32)>,
    pub(crate) hovered_zone: Option<ToneZone>,
    /// Tone slider being Alt-dragged in the settings panel. The panel draws after the preview,
    /// so the preview's diagnostics follow it one frame later.
    pub(crate) alt_tone: Option<ToneZone>,
    pub(crate) preview_rect: Option<Rect>,
    pub(crate) rect: Option<Rect>,
    pub(crate) vectorscope_rect: Option<Rect>,
}

/// One visible proxy pixel, in the histogram RGB domain and ICC-managed vectorscope domain.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct HoverSample {
    pub(crate) position: [f32; 2],
    pub(crate) rgb: [f32; 3],
    pub(crate) hue_saturation: Option<[f32; 2]>,
}

/// Floating panel bounds `[left, top, right, bottom]` in points: finite and 200×150..=4096².
fn valid_floating_rect(r: [f32; 4]) -> bool {
    let (w, h) = (r[2] - r[0], r[3] - r[1]);
    r.iter().all(|x| x.is_finite()) && (200.0..=4096.0).contains(&w) && (150.0..=4096.0).contains(&h)
}

impl CameraRawDialog {
    fn pixels(&self) -> &[[f32; 4]] {
        if self.show_before { &self.proxy } else { &self.processed }
    }
    fn pixel(&self, p: [f32; 2]) -> Option<[f32; 4]> {
        if !p.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)) || self.pw == 0 || self.ph == 0 {
            return None;
        }
        let x = ((p[0] * self.pw as f32) as usize).min(self.pw - 1);
        let y = ((p[1] * self.ph as f32) as usize).min(self.ph - 1);
        let q = self
            .detail
            .sample(p, self.preview_revision, self.show_before, &self.params)
            .or_else(|| self.pixels().get(y.checked_mul(self.pw)?.checked_add(x)?).copied())?;
        (q[3] > 0.0 && q.iter().all(|v| v.is_finite())).then_some(q)
    }
    pub(crate) fn readout(&self, p: [f32; 2], lab: bool) -> Option<Value> {
        let q = self.pixel(p)?;
        let rgb = [q[0], q[1], q[2]];
        let values = if lab {
            let mut l = [0.0; 3];
            self.lab_transform.eval(&rgb, &mut l);
            [l[0] * 100.0, l[1] * 255.0 - 128.0, l[2] * 255.0 - 128.0]
        } else {
            rgb.map(|v| v * 255.0)
        };
        Some(json!({"position":p,"space":if lab {"lab"}else{"rgb"},"values":values,"alpha":q[3]}))
    }
    pub(crate) fn hover_sample(&self, vectorscope: bool) -> Option<HoverSample> {
        let position = self.scope.pointer_sample?;
        let q = self.pixel(position)?;
        let rgb = [q[0], q[1], q[2]];
        let hue_saturation = if vectorscope {
            let mut scope_rgb = [0.0; 3];
            self.scope_transform.eval(&rgb, &mut scope_rgb);
            if scope_rgb.iter().all(|x| x.is_finite()) {
                let (h, s) = photocraft_algo::vectorscope::hue_saturation(scope_rgb.map(|x| x.clamp(0.0, 1.0)));
                Some([h, s])
            } else {
                None
            }
        } else {
            None
        };
        Some(HoverSample { position, rgb, hue_saturation })
    }
    pub(crate) fn ensure_scope(&mut self, selected: bool) {
        if self
            .scope
            .cache
            .as_ref()
            .is_some_and(|(revision, before, masked, _)| *revision == self.preview_revision && *before == self.show_before && *masked == selected)
        {
            return;
        }
        let start = crate::gpu_canvas::now_ms();
        // A fixed 2-D grid bounds CMS and analysis work independently of the image size. Rows
        // and columns step separately, so a stride can't alias with the row width into stripes.
        let step = ((self.pixels().len() as f64 / SCOPE_SAMPLES as f64).sqrt().ceil() as usize).max(1);
        let (pw, ph) = (self.pw, self.ph);
        let pixels: Vec<_> = (0..ph)
            .step_by(step)
            .flat_map(|y| (0..pw).step_by(step).map(move |x| y * pw + x))
            .filter_map(|i| self.pixels().get(i).map(|p| (i, p)))
            .filter(|(i, p)| {
                p[3] > 0.0 && p.iter().all(|v| v.is_finite()) && (!selected || self.selection.as_ref().and_then(|m| m.get(*i)).is_some_and(|v| *v > 0.0))
            })
            .map(|(_, p)| {
                let mut rgb = [0.0; 3];
                self.scope_transform.eval(&p[..3], &mut rgb);
                [rgb[0], rgb[1], rgb[2], p[3]]
            })
            .collect();
        let scope = Vectorscope::from_rgba(&pixels);
        self.scope.cache = Some((self.preview_revision, self.show_before, selected, scope));
        self.scope.ms = crate::gpu_canvas::now_ms() - start;
        self.scope.revision = self.scope.revision.saturating_add(1);
    }
}

/// Remember dialog presentation through the existing preferences store, after a gesture ends.
/// Probe positions are image-specific and never restored onto a different document.
pub(crate) fn persist(app: &mut crate::PhotocraftApp, ctx: &egui::Context) {
    if ctx.input(|i| i.pointer.any_down()) {
        return;
    }
    let mut view = app.ui.camera_raw_scope.clone();
    view.samplers.clear();
    view.vectorscope = false;
    let Ok(value) = serde_json::to_value(view) else {
        return;
    };
    const KEY: &str = "filter.cameraRaw.scope";
    if app.session.prefs().dialogs.get(KEY) != Some(&value) {
        app.session.prefs.edit(|p| p.dialogs.insert(KEY.into(), value));
    }
}

fn point(v: &Value) -> Result<[f32; 2], String> {
    let p: [f32; 2] = serde_json::from_value(v.clone()).map_err(|e| e.to_string())?;
    if !p.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x)) {
        return Err("sample coordinates must be normalized to 0..1".into());
    }
    Ok(p)
}

/// The same controller operations drive real gestures and automation. Validate the entire request
/// before changing state; invalid coordinates, huge vectors, and unknown properties are errors.
pub(crate) struct ScopeUpdate {
    view: CameraRawScopeState,
    sample: Option<Option<[f32; 2]>>,
    tone: Option<(ToneZone, f32)>,
    geometry_requested: bool,
}

impl ScopeUpdate {
    pub(crate) fn apply(self, d: &mut CameraRawDialog, view: &mut CameraRawScopeState) {
        d.scope.geometry_requested |= self.geometry_requested;
        *view = self.view;
        if let Some(s) = self.sample {
            d.scope.pointer_sample = s;
        }
        if let Some((z, v)) = self.tone {
            z.set(d, v);
        }
    }
}

pub(crate) fn prepare_control(params: &CameraRaw, has_selection: bool, view: &CameraRawScopeState, value: &Value) -> Result<ScopeUpdate, String> {
    let obj = value.as_object().ok_or("scope must be an object")?;
    if obj.get("samplers").is_some_and(|v| v.as_array().is_none_or(|a| a.len() > MAX_SAMPLERS)) {
        return Err(format!("at most {MAX_SAMPLERS} samplers"));
    }
    if obj.contains_key("samplers") && ["addSampler", "removeSampler", "clearSamplers"].iter().any(|k| obj.contains_key(*k)) {
        return Err("samplers replaces the list; don't combine it with addSampler, removeSampler or clearSamplers".into());
    }
    // Plain view fields first, then the sampler operations on the result (clear → remove → add).
    let mut base = serde_json::to_value(view).map_err(|e| e.to_string())?;
    let mut sample = None;
    let mut tone = None;
    for (key, v) in obj {
        match key.as_str() {
            "sample" => sample = Some(if v.is_null() { None } else { Some(point(v)?) }),
            "addSampler" | "removeSampler" | "clearSamplers" => {}
            "tone" => {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct Change {
                    zone: ToneZone,
                    delta: Option<f32>,
                    reset: Option<bool>,
                }
                let c: Change = serde_json::from_value(v.clone()).map_err(|e| e.to_string())?;
                if c.delta.is_some_and(|x| !x.is_finite()) || (c.delta.is_none() && c.reset != Some(true)) {
                    return Err("tone requires finite delta or reset=true".into());
                }
                let mut params = params.clone();
                tone = Some((c.zone, if c.reset == Some(true) { 0.0 } else { *c.zone.value(&mut params) + c.delta.unwrap_or(0.0) }));
            }
            _ => {
                if base.get(key).is_none() {
                    return Err(format!("unknown scope property {key}"));
                }
                base[key] = v.clone();
            }
        }
    }
    let mut next: CameraRawScopeState = serde_json::from_value(base).map_err(|e| e.to_string())?;
    if let Some(v) = obj.get("clearSamplers") {
        if v.as_bool() != Some(true) {
            return Err("clearSamplers must be true".into());
        }
        next.samplers.clear();
    }
    if let Some(v) = obj.get("removeSampler") {
        let i = v.as_u64().and_then(|v| usize::try_from(v).ok()).ok_or("bad sampler index")?;
        if i >= next.samplers.len() {
            return Err("unknown sampler".into());
        }
        next.samplers.remove(i);
    }
    if let Some(v) = obj.get("addSampler") {
        if next.samplers.len() >= MAX_SAMPLERS {
            return Err(format!("at most {MAX_SAMPLERS} samplers"));
        }
        next.samplers.push(point(v)?);
    }
    for p in &next.samplers {
        point(&json!(p))?;
    }
    if next.floating_rect.is_some_and(|r| !valid_floating_rect(r)) {
        return Err("invalid floating scope rectangle".into());
    }
    if next.selected_region && !has_selection {
        return Err("no selected region in this document".into());
    }
    Ok(ScopeUpdate { view: next, sample, tone, geometry_requested: obj.contains_key("floatingRect") })
}

fn readout_text(value: Option<Value>) -> String {
    let Some(v) = value else { return "—".into() };
    let a = v.get("values").and_then(Value::as_array);
    let get = |i| a.and_then(|a| a.get(i)).and_then(Value::as_f64).unwrap_or(0.0);
    if v.get("space").and_then(Value::as_str) == Some("lab") {
        format!("L {:.1}  a {:.1}  b {:.1}", get(0), get(1), get(2))
    } else {
        format!("R {:.0}  G {:.0}  B {:.0}", get(0), get(1), get(2))
    }
}

fn warning(ui: &mut egui::Ui, rect: Rect, mask: u8, on: &mut bool, label: &str) {
    let t = Tokens::get(ui.ctx());
    let response = ui.interact(rect, ui.id().with(label), Sense::click()).on_hover_text(label);
    response.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Checkbox, ui.is_enabled(), *on, label));
    let r = rect.shrink(3.0);
    ui.painter().add(egui::Shape::convex_polygon(
        vec![pos2(r.center().x, r.top()), r.left_bottom(), r.right_bottom()],
        if mask == 0 { t.histogram_background() } else { t.histogram_color(mask) },
        Stroke::new(1.0, if *on { t.text } else { t.text_faint }),
    ));
    if response.clicked() {
        *on = !*on;
    }
}

/// Scope display options. The menu stays open while toggling, so options that depend on
/// Show Vectorscope appear in place.
fn context_menu(response: &egui::Response, has_selection: bool, view: &mut CameraRawScopeState) {
    egui::Popup::context_menu(response).close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside).show(|ui| context(ui, has_selection, view));
}

fn context(ui: &mut egui::Ui, has_selection: bool, view: &mut CameraRawScopeState) {
    widgets::checkbox(ui, &mut view.lab, tl!("Show Lab Color Readouts"));
    widgets::checkbox(ui, &mut view.vectorscope, tl!("Show Vectorscope"));
    if view.vectorscope {
        if has_selection {
            widgets::checkbox(ui, &mut view.selected_region, tl!("Show Selected Region"));
        }
        widgets::checkbox(ui, &mut view.red_right, tl!("Show Red at 3 o'clock"));
        let mut skin = !view.hide_skin_line;
        widgets::checkbox(ui, &mut skin, tl!("Show Skin Tone Indicator"));
        view.hide_skin_line = !skin;
    }
}

fn graph(ui: &mut egui::Ui, d: &mut CameraRawDialog, view: &mut CameraRawScopeState, height: f32) {
    let t = Tokens::get(ui.ctx());
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width().max(1.0), height), Sense::click_and_drag());
    d.scope.rect = Some(rect);
    response.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Other, ui.is_enabled(), tl!("Tone Histogram")));
    ui.painter().rect_filled(rect, t.radius_sm, t.histogram_background());
    let plot = rect.shrink(4.0);
    crate::rgb_histogram::paint(ui.painter(), plot, d.histogram(), &t);
    let corner = 18.0;
    if let Some(sample) = d.hover_sample(false) {
        crate::rgb_histogram::paint_sample(ui.painter(), plot, sample.rgb, &t);
        let [r, g, b] = sample.rgb.map(|x| x * 255.0);
        let text = format!("R {r:.0}   G {g:.0}   B {b:.0}");
        let font = crate::theme::mono(11.0);
        let galley = ui.painter().layout_no_wrap(text, font, t.text);
        let width = (galley.size().x + 12.0).min((rect.width() - 2.0 * corner - 4.0).max(1.0));
        let badge = Rect::from_center_size(pos2(rect.center().x, rect.top() + 13.0), vec2(width, 20.0));
        ui.painter().rect_filled(badge, t.radius_sm, t.card.gamma_multiply(0.65));
        ui.painter().rect_stroke(badge, t.radius_sm, Stroke::new(0.7, t.field_border.gamma_multiply(0.55)), egui::StrokeKind::Inside);
        ui.painter().with_clip_rect(badge.shrink(2.0)).galley(badge.center() - galley.size() * 0.5, galley, t.text);
    }
    let sh = d.histogram().shadows.iter().enumerate().fold(0u8, |m, (c, n)| m | if *n > 0 { 1 << c } else { 0 });
    let hi = d.histogram().highlights.iter().enumerate().fold(0u8, |m, (c, n)| m | if *n > 0 { 1 << c } else { 0 });
    warning(ui, Rect::from_min_size(rect.min, vec2(corner, corner)), sh, &mut view.shadows, tl!("Shadow Clipping Warning (U)"));
    warning(
        ui,
        Rect::from_min_size(pos2(rect.right() - corner, rect.top()), vec2(corner, corner)),
        hi,
        &mut view.highlights,
        tl!("Highlight Clipping Warning (O)"),
    );
    let pos = response.interact_pointer_pos().or_else(|| response.hover_pos());
    d.scope.hovered_zone = pos.filter(|p| p.y > rect.top() + corner).map(|p| ToneZone::at(((p.x - plot.left()) / plot.width().max(1.0)).clamp(0.0, 1.0)));
    if let Some(z) = d.scope.tone_drag.map(|x| x.0).or(d.scope.hovered_zone) {
        let [lo, hi] = z.range();
        let region = Rect::from_min_max(pos2(plot.left() + lo * plot.width(), plot.top()), pos2(plot.left() + hi * plot.width(), plot.bottom()));
        ui.painter().rect_filled(region, 0.0, t.text.gamma_multiply(0.055));
        response.clone().on_hover_cursor(egui::CursorIcon::ResizeHorizontal);
        response.clone().on_hover_text(format!("{}  {:.2}", z.label(), *z.value(&mut d.params)));
        if response.double_clicked() {
            z.set(d, 0.0);
        }
        if response.drag_started_by(egui::PointerButton::Primary)
            && let Some(p) = pos
        {
            let delta = response.drag_delta().x;
            d.scope.tone_drag = Some((z, p.x - delta, *z.value(&mut d.params)));
        }
    }
    if response.dragged_by(egui::PointerButton::Primary)
        && let (Some(p), Some((z, start, initial))) = (pos, d.scope.tone_drag)
    {
        z.set(d, initial + (p.x - start) / plot.width().max(1.0) * z.limit() * 2.0);
    }
    if response.drag_stopped() {
        d.scope.tone_drag = None;
    }
    context_menu(&response, d.selection.is_some(), view);
    ui.painter().rect_stroke(rect, t.radius_sm, Stroke::new(1.0, t.field_border), egui::StrokeKind::Inside);
}

/// The cloud and pixel target use exactly the same orientation and saturation radius.
fn scope_point(rect: Rect, red_right: bool, hue: f32, saturation: f32) -> Pos2 {
    let radius = (rect.width().min(rect.height()) * 0.5 - 14.0).max(1.0);
    let offset = if red_right { 0.0 } else { -100.0f32.to_radians() };
    let angle = offset - hue * std::f32::consts::TAU;
    rect.center() + vec2(angle.cos(), angle.sin()) * saturation * radius
}

fn vectorscope(ui: &mut egui::Ui, d: &mut CameraRawDialog, view: &mut CameraRawScopeState, height: f32) {
    d.ensure_scope(view.selected_region);
    let t = Tokens::get(ui.ctx());
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), height), Sense::click());
    context_menu(&response, d.selection.is_some(), view);
    ui.painter().rect_filled(rect, t.radius_sm, t.histogram_background());
    d.scope.vectorscope_rect = Some(rect);
    let center = rect.center();
    let radius = (rect.width().min(rect.height()) * 0.5 - 14.0).max(1.0);
    let point = |h: f32, s: f32| scope_point(rect, view.red_right, h, s);
    for ring in [0.25, 0.5, 0.75, 1.0] {
        ui.painter().circle_stroke(center, radius * ring, Stroke::new(0.7, t.text_faint.gamma_multiply(0.45)));
    }
    let mut mesh = egui::Mesh::default();
    if let Some((_, _, _, data)) = &d.scope.cache {
        let peak = data.counts.iter().copied().max().unwrap_or(1).max(1) as f32;
        for (i, count) in data.counts.iter().enumerate().filter(|(_, c)| **c > 0) {
            let h = (i / SATURATIONS) as f32 / HUES as f32;
            let s = (i % SATURATIONS) as f32 / (SATURATIONS - 1) as f32;
            let p = point(h, s);
            let color = t.scope_hue(h, s).gamma_multiply(0.2 + 0.8 * (*count as f32 / peak).sqrt());
            let dot = (radius / (SATURATIONS - 1) as f32).clamp(0.7, 3.0);
            mesh.add_colored_rect(Rect::from_center_size(p, vec2(dot * 2.0, dot * 2.0)), color);
        }
    }
    ui.painter().add(mesh);
    for (i, label) in ["R", "Y", "G", "C", "B", "M"].iter().enumerate() {
        let h = i as f32 / 6.0;
        ui.painter().line_segment([center, point(h, 1.0)], Stroke::new(0.6, t.text_faint.gamma_multiply(0.4)));
        ui.painter().circle_filled(point(h, 1.0), 2.0, t.scope_hue(h, 1.0));
        ui.painter().text(point(h, 1.11), Align2::CENTER_CENTER, label, FontId::proportional(10.0), t.scope_hue(h, 1.0));
    }
    if !view.hide_skin_line {
        ui.painter().line_segment([center, point(23.0 / 360.0, 1.0)], Stroke::new(1.0, t.text_dim));
    }
    if let Some(sample) = d.hover_sample(true)
        && let Some([h, saturation]) = sample.hue_saturation
    {
        let at = point(h, saturation);
        let painter = ui.painter().with_clip_rect(rect);
        for stroke in [Stroke::new(3.0, t.histogram_background()), Stroke::new(1.2, t.histogram_color(7))] {
            painter.circle_stroke(at, 5.0, stroke);
            painter.line_segment([at - vec2(9.0, 0.0), at + vec2(9.0, 0.0)], stroke);
            painter.line_segment([at - vec2(0.0, 9.0), at + vec2(0.0, 9.0)], stroke);
        }
    }
}

pub(crate) fn header(ui: &mut egui::Ui, d: &mut CameraRawDialog, view: &mut CameraRawScopeState) {
    view.samplers.truncate(MAX_SAMPLERS);
    view.samplers.retain(|p| p.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x)));
    if view.floating_rect.is_some_and(|r| !valid_floating_rect(r)) {
        view.floating_rect = None;
    }
    if view.floating {
        let (_, response) = ui.allocate_exact_size(vec2(ui.available_width(), 48.0), Sense::click());
        context_menu(&response, d.selection.is_some(), view);
    } else {
        let height = (ui.available_height() * 0.18).clamp(48.0, 110.0);
        graph(ui, d, view, height);
        if view.vectorscope {
            vectorscope(ui, d, view, (ui.available_height() * 0.3).clamp(100.0, 220.0));
        }
    }
    ui.add_space(8.0);
}

pub(crate) fn floating(ctx: &egui::Context, d: &mut CameraRawDialog, view: &mut CameraRawScopeState, dock: Rect) {
    if !view.floating {
        return;
    }
    let r = view.floating_rect.unwrap_or([40.0, 60.0, 400.0, 440.0]);
    let content = ctx.content_rect();
    let start = pos2(
        r[0].clamp(content.left(), (content.right() - 200.0).max(content.left())),
        r[1].clamp(content.top(), (content.bottom() - 150.0).max(content.top())),
    );
    let window = egui::Window::new("")
        .id(egui::Id::new("cr-floating-scope"))
        .order(egui::Order::Foreground)
        .collapsible(false)
        .resizable(true)
        .default_pos(start)
        .default_size(vec2(r[2] - r[0], r[3] - r[1]))
        .min_size(vec2(200.0, 150.0));
    let window = if d.scope.geometry_requested {
        d.scope.geometry_requested = false;
        let size = vec2(r[2] - r[0], r[3] - r[1]).min(content.size().max(vec2(1.0, 1.0)));
        // egui's title-bar drag restores the previous pivot in this frame. Disable moving
        // only for the explicit controller geometry update, then restore normal dragging.
        window.movable(false).current_pos(start).min_size(size).max_size(size)
    } else {
        window
    };
    let response = window.show(ctx, |ui| {
        let height = if view.vectorscope { (ui.available_height() * 0.35).clamp(48.0, 150.0) } else { (ui.available_height() - 28.0).max(48.0) };
        graph(ui, d, view, height);
        if view.vectorscope {
            vectorscope(ui, d, view, (ui.available_height() - 12.0).max(100.0));
        }
    });
    if let Some(response) = response {
        let previous = pos2(r[0], r[1]);
        let r = response.response.rect;
        view.floating_rect = Some([r.left(), r.top(), r.right(), r.bottom()]);
        if ctx.input(|i| i.pointer.primary_down()) && r.min.distance(previous) > 0.5 {
            d.scope.floating_dragged = true;
        }
        if ctx.input(|i| i.pointer.any_released()) {
            if d.scope.floating_dragged && ctx.input(|i| i.pointer.interact_pos().is_some_and(|p| dock.contains(p))) {
                view.floating = false;
            }
            d.scope.floating_dragged = false;
        }
    }
}

pub(crate) fn preview(ui: &mut egui::Ui, d: &mut CameraRawDialog, view: &mut CameraRawScopeState, rect: Rect, resp: &egui::Response, panning: bool) {
    let t = Tokens::get(ui.ctx());
    if let Some(p) = resp.hover_pos().filter(|p| rect.contains(*p)) {
        d.scope.pointer_sample =
            Some([((p.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0), ((p.y - rect.top()) / rect.height().max(1.0)).clamp(0.0, 1.0)]);
    } else {
        d.scope.pointer_sample = None;
    }
    let to_screen = |p: [f32; 2]| pos2(rect.left() + p[0] * rect.width(), rect.top() + p[1] * rect.height());
    if view.sampler_tool && !panning {
        resp.clone().on_hover_cursor(egui::CursorIcon::Crosshair);
        if resp.clicked()
            && let Some(p) = d.scope.pointer_sample
        {
            let near = view.samplers.iter().position(|q| to_screen(*q).distance(to_screen(p)) < 12.0);
            if ui.input(|i| i.modifiers.alt) {
                if let Some(i) = near {
                    view.samplers.remove(i);
                }
            } else if near.is_none() && view.samplers.len() < MAX_SAMPLERS {
                view.samplers.push(p);
            }
        }
        if resp.drag_started() {
            let start = resp.interact_pointer_pos().map(|p| p - resp.drag_delta());
            d.scope.sampler_drag = view.samplers.iter().position(|q| start.is_some_and(|s| to_screen(*q).distance(s) < 16.0));
        }
        if resp.dragged()
            && let (Some(p), Some(i)) = (d.scope.pointer_sample, d.scope.sampler_drag)
            && let Some(q) = view.samplers.get_mut(i)
        {
            *q = p;
        }
        if resp.drag_stopped() {
            d.scope.sampler_drag = None;
        }
    }
    if panning {
        d.scope.sampler_drag = None;
    }
    let alt_zone = d.scope.tone_drag.map(|d| d.0).or(d.scope.alt_tone);
    let mode = if ui.input(|i| i.modifiers.alt && i.pointer.primary_down()) && alt_zone.is_some() {
        if matches!(alt_zone, Some(ToneZone::Blacks | ToneZone::Shadows)) { ClippingMode::ShadowChannels } else { ClippingMode::HighlightChannels }
    } else if view.shadows || view.highlights {
        ClippingMode::Warnings(u8::from(view.shadows) | (u8::from(view.highlights) << 1))
    } else {
        ClippingMode::None
    };
    let key = (d.preview_revision, d.show_before, mode, t.kind);
    if mode != ClippingMode::None && d.detail.overlay(ui, rect, d.preview_revision, mode) {
        d.scope.overlay_key = Some(key);
        d.scope.overlay_detail = true;
    } else if mode != ClippingMode::None {
        if d.scope.overlay_key != Some(key) || d.scope.overlay_detail {
            d.scope.overlay_detail = false;
            let colors = d
                .pixels()
                .iter()
                .map(|p| {
                    let (lo, hi) = clipping(*p);
                    if p[3] <= 0.0 || !p.iter().all(|v| v.is_finite()) {
                        return Color32::TRANSPARENT;
                    }
                    match mode {
                        ClippingMode::ShadowChannels => t.histogram_color(7 ^ lo),
                        ClippingMode::HighlightChannels => {
                            if hi == 0 {
                                t.histogram_background()
                            } else {
                                t.histogram_color(hi)
                            }
                        }
                        ClippingMode::Warnings(warnings) => {
                            if warnings & 2 != 0 && hi != 0 {
                                t.histogram_color(1)
                            } else if warnings & 1 != 0 && lo != 0 {
                                t.histogram_color(4)
                            } else {
                                Color32::TRANSPARENT
                            }
                        }
                        ClippingMode::None => Color32::TRANSPARENT,
                    }
                })
                .collect();
            let image = egui::ColorImage::new([d.pw, d.ph], colors);
            match &mut d.scope.overlay_tex {
                Some(t) => t.set(image, egui::TextureOptions::NEAREST),
                None => d.scope.overlay_tex = Some(ui.ctx().load_texture("cr-clipping", image, egui::TextureOptions::NEAREST)),
            }
            d.scope.overlay_key = Some(key);
        }
        if let Some(tex) = &d.scope.overlay_tex {
            ui.painter().image(tex.id(), rect, Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
        }
    }
    let visible = rect.intersect(ui.clip_rect());
    if !view.samplers.is_empty() {
        ui.painter().rect_filled(
            Rect::from_min_size(visible.min, vec2(visible.width().min(330.0), view.samplers.len().min(MAX_SAMPLERS) as f32 * 16.0 + 12.0)),
            t.radius_sm,
            t.dock.gamma_multiply(0.92),
        );
    }
    for (i, p) in view.samplers.iter().take(MAX_SAMPLERS).enumerate() {
        let q = to_screen(*p);
        ui.painter().circle_stroke(q, 5.0, Stroke::new(1.0, t.text));
        ui.painter().line_segment([q - vec2(8.0, 0.0), q + vec2(8.0, 0.0)], Stroke::new(1.0, t.text));
        ui.painter().line_segment([q - vec2(0.0, 8.0), q + vec2(0.0, 8.0)], Stroke::new(1.0, t.text));
        let text = format!("{}  {}", i + 1, readout_text(d.readout(*p, view.lab)));
        ui.painter().text(pos2(visible.left() + 6.0, visible.top() + 6.0 + i as f32 * 16.0), Align2::LEFT_TOP, text, crate::theme::mono(11.0), t.text);
        ui.painter().text(q + vec2(7.0, 7.0), Align2::LEFT_TOP, format!("{}", i + 1), crate::theme::mono(11.0), t.text);
    }
}

pub(crate) fn shortcuts(ctx: &egui::Context, view: &mut CameraRawScopeState) {
    if ctx.text_edit_focused() {
        return;
    }
    ctx.input_mut(|i| {
        if i.consume_key(egui::Modifiers::NONE, egui::Key::U) {
            view.shadows = !view.shadows;
        }
        if i.consume_key(egui::Modifiers::NONE, egui::Key::O) {
            view.highlights = !view.highlights;
        }
        if i.consume_key(egui::Modifiers::NONE, egui::Key::S) {
            view.sampler_tool = !view.sampler_tool;
        }
    });
}
