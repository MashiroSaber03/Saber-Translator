//! Select › Color Range… as Photoshop lays it out: the Select menu (Sampled Colors, the six hue
//! families, Highlights / Midtones / Shadows, Out Of Gamut), Fuzziness, Localized Color Clusters
//! with Range, the tonal range of Highlights / Midtones / Shadows, the three eyedroppers (sample,
//! add, subtract; they pick on the preview, Shift adds and Alt subtracts), a Selection / Image
//! preview, canvas Selection Preview modes and Invert. Controls a mode doesn't use are disabled, like Photoshop's.
//!
//! Everything lives in the dialog fields (`ui.dialog.set` drives it). The preview and OK run the
//! same engine command, `select.colorRange`, so the GUI, MCP and scripts select identically; the
//! preview compares a small proxy against samples from the original document; OK selects at
//! full resolution (one history step). Reduced previews approximate fine detail and effects.
//! Cancel never touches the document, so the previous selection stays as it was.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use egui::{Color32, Rect, Sense, Stroke, StrokeKind, pos2, vec2};
use photocraft_doc::{DocId, Document};
use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::state::DialogKind;
use crate::theme::Tokens;
use crate::widgets;

/// The engine command the dialog drives.
pub const COMMAND: &str = "select.colorRange";

/// Select menu entries in Photoshop's order (engine value, label).
pub const SELECTS: &[(&str, &str)] = &[
    ("sampledColors", "Sampled Colors"),
    ("reds", "Reds"),
    ("yellows", "Yellows"),
    ("greens", "Greens"),
    ("cyans", "Cyans"),
    ("blues", "Blues"),
    ("magentas", "Magentas"),
    ("highlights", "Highlights"),
    ("midtones", "Midtones"),
    ("shadows", "Shadows"),
    ("skinTones", "Skin Tones"),
    ("outOfGamut", "Out Of Gamut"),
];

/// Longest side of the preview thumbnail, in points.
const PREVIEW: u32 = 200;
/// Canvas overlays are view state; keep their allocation bounded on very large documents.
const CANVAS_PREVIEW: u32 = 1024;
const PREVIEWS: &[(&str, &str)] =
    &[("none", "None"), ("grayscale", "Grayscale"), ("blackMatte", "Black Matte"), ("whiteMatte", "White Matte"), ("quickMask", "Quick Mask")];

/// Which controls the current Select mode uses (the rest are disabled or hidden).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Controls {
    /// Fuzziness (Sampled Colors: colour distance; tones: falloff in % of the tonal scale).
    pub fuzziness: bool,
    /// The eyedroppers and Localized Color Clusters (Sampled Colors only).
    pub sampling: bool,
    /// The Range slider (Localized Color Clusters on).
    pub range: bool,
    /// The tonal range sliders (Highlights / Midtones / Shadows).
    pub tonal: bool,
}

fn s<'a>(f: &'a Map<String, Value>, k: &str, d: &'a str) -> &'a str {
    f.get(k).and_then(Value::as_str).unwrap_or(d)
}
fn num(f: &Map<String, Value>, k: &str, d: f32) -> f32 {
    f.get(k).and_then(Value::as_f64).map_or(d, |v| v as f32)
}
fn flag(f: &Map<String, Value>, k: &str) -> bool {
    f.get(k).and_then(Value::as_bool).unwrap_or(false)
}
fn points(f: &Map<String, Value>, k: &str) -> Vec<[f64; 2]> {
    let pt = |v: &Value| match v.as_array()?.as_slice() {
        [x, y] => Some([x.as_f64()?, y.as_f64()?]),
        _ => None,
    };
    f.get(k).and_then(Value::as_array).map(|a| a.iter().filter_map(pt).collect()).unwrap_or_default()
}

pub fn controls(f: &Map<String, Value>) -> Controls {
    let select = s(f, "select", "sampledColors");
    let sampled = select == "sampledColors";
    let tonal = matches!(select, "highlights" | "midtones" | "shadows");
    Controls { fuzziness: sampled || tonal || select == "skinTones", sampling: sampled, range: sampled && flag(f, "localized"), tonal }
}

/// Open the dialog with Photoshop's defaults (Sampled Colors, Fuzziness 40, the foreground colour
/// as the sample until the eyedropper picks one).
pub fn open(app: &mut PhotocraftApp) -> u64 {
    let label = photocraft_engine::commands::find(COMMAND).map_or(tl!("Color Range…"), |c| c.label);
    let mut f = Map::new();
    f.insert("__colorRange".into(), json!(true));
    f.insert("__label".into(), json!(label));
    f.insert("select".into(), json!("sampledColors"));
    f.insert("fuzziness".into(), json!(40.0));
    f.insert("localized".into(), json!(false));
    f.insert("range".into(), json!(100.0));
    f.insert("toneFuzziness".into(), json!(20.0));
    f.insert("shadowsLevel".into(), json!(65.0));
    f.insert("highlightsLevel".into(), json!(190.0));
    f.insert("midtonesLow".into(), json!(105.0));
    f.insert("midtonesHigh".into(), json!(150.0));
    f.insert("invert".into(), json!(false));
    f.insert("points".into(), json!([]));
    f.insert("subtractPoints".into(), json!([]));
    f.insert("__order".into(), json!([]));
    f.insert("__tool".into(), json!("sample"));
    f.insert("__view".into(), json!("selection"));
    f.insert("__selectionPreview".into(), json!("none"));
    app.color_range = None;
    app.ui.open_dialog(DialogKind::Command, f)
}

pub fn owns(f: &Map<String, Value>) -> bool {
    f.contains_key("__colorRange")
}

/// The `select.colorRange` params for the dialog's state: only what the Select mode uses.
pub fn params(f: &Map<String, Value>) -> Value {
    let select = s(f, "select", "sampledColors");
    let c = controls(f);
    let mut p = Map::new();
    p.insert("select".into(), json!(select));
    p.insert("invert".into(), json!(flag(f, "invert")));
    if c.sampling {
        p.insert("fuzziness".into(), json!(num(f, "fuzziness", 40.0)));
        let (add, sub) = (points(f, "points"), points(f, "subtractPoints"));
        let picked = !add.is_empty() || !sub.is_empty();
        if !add.is_empty() {
            p.insert("points".into(), json!(add));
        }
        if !sub.is_empty() {
            p.insert("subtractPoints".into(), json!(sub));
            if let Some(o) = order(f, add.len(), sub.len()) {
                p.insert("order".into(), o);
            }
        }
        // Localized clusters need a picked position; until then it is the plain colour range.
        if c.range && picked {
            p.insert("localized".into(), json!(true));
            p.insert("range".into(), json!(num(f, "range", 100.0)));
        }
    } else if select == "skinTones" {
        p.insert("fuzziness".into(), json!(num(f, "fuzziness", 40.0)));
    } else if c.tonal {
        p.insert("fuzziness".into(), json!(num(f, "toneFuzziness", 20.0)));
        let range = match select {
            "shadows" => json!(num(f, "shadowsLevel", 65.0)),
            "highlights" => json!(num(f, "highlightsLevel", 190.0)),
            _ => {
                let (lo, hi) = (num(f, "midtonesLow", 105.0), num(f, "midtonesHigh", 150.0));
                json!([lo.min(hi), lo.max(hi)])
            }
        };
        p.insert("tonalRange".into(), range);
    }
    Value::Object(p)
}

/// OK: run the command on the document (one history step).
pub fn confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    app.color_range = None;
    app.run(COMMAND, params(f))
}

/// Cached preview: the document proxy (per revision) and the textures drawn from it.
pub struct Preview {
    doc: DocId,
    revision: u64,
    /// Proxy scale: proxy pixel = `k` document pixels.
    k: u32,
    proxy: Arc<Document>,
    image: Option<egui::TextureHandle>,
    /// Hash of the params the mask texture shows.
    mask_key: u64,
    mask: Option<egui::TextureHandle>,
    /// Why the preview couldn't be drawn (shown in the dialog and logged, never silent: #145).
    error: Option<String>,
    canvas_proxy: Option<(u32, Arc<Document>)>,
    overlay: Option<egui::TextureHandle>,
    overlay_key: u64,
}

fn hash(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
}

fn query_key(app: &PhotocraftApp, params: &Value) -> u64 {
    let mut key = std::collections::hash_map::DefaultHasher::new();
    params.to_string().hash(&mut key);
    let foreground = app.session.tools.foreground;
    foreground.map(f32::to_bits).hash(&mut key);
    // This runs several times per frame, including mere pointer motion. Debug-formatting a
    // ProofView serializes the whole ICC profile/LUT; proof() also clones the default profile.
    // Borrow the state and use the profile's memoized content hash instead.
    let proof = app.session.active().and_then(|st| app.session.color.proof_ref(st.doc.id));
    proof.is_some().hash(&mut key);
    if let Some(proof) = proof {
        let setup = &proof.setup;
        (setup.profile.content_hash(), setup.intent, setup.bpc, setup.simulate_paper, setup.kind).hash(&mut key);
        (proof.enabled, proof.gamut_warning, proof.gamut_threshold.to_bits()).hash(&mut key);
    }
    key.finish().max(1)
}

/// Keep the accelerated preview paths in sync with the engine's Color Range replacement.
/// The full-resolution engine handles other combine modes itself.
fn scope_preview(proxy: &Document, params: &Value, mask: &mut [f32]) {
    if matches!(params.get("mode").and_then(Value::as_str), Some("add" | "subtract" | "intersect")) {
        return;
    }
    if let Some(selection) = proxy.selection.as_ref() {
        let prior = photocraft_algo::selection::mask_from_surface(Some(selection), proxy.bounds());
        for (value, bound) in mask.iter_mut().zip(prior) {
            *value = value.min(bound);
        }
    }
}

/// The selection mask `params` would make, on the proxy: the engine command run on a scratch
/// session (Out Of Gamut uses the document's own proof setup).
fn proxy_mask(app: &PhotocraftApp, proxy: &Document, k: u32, params: &Value) -> Result<Vec<f32>, String> {
    let area = proxy.bounds();
    if params.get("select").and_then(Value::as_str).unwrap_or("sampledColors") == "sampledColors" {
        // Pick on the original document, not the thumbnail's nearest retained pixel. The
        // prepared Lab query is the same one the command uses at full resolution.
        let query = photocraft_engine::selection_cmds::ColorRangeSamples::new(&app.session, params).map_err(|e| e.to_string())?;
        // Like the command (and OK): `sampleAllLayers: false` judges the active layer's own
        // pixels, otherwise the composite.
        let all_layers = params.get("sampleAllLayers").and_then(Value::as_bool).unwrap_or(true);
        let layer = app.session.active().and_then(|d| d.active_layer).and_then(|id| proxy.layer(id)).and_then(|l| l.surface());
        let px = match (all_layers, layer) {
            (false, Some(surf)) => {
                let mut px = vec![[0.0f32; 4]; area.width() as usize * area.height() as usize];
                surf.read_rgba_into(area, &mut px);
                px
            }
            _ => photocraft_compose::render(proxy, area).px,
        };
        let invert = params.get("invert").and_then(Value::as_bool).unwrap_or(false);
        let mut mask: Vec<f32> = query
            .coverage(&px, area.width() as usize, k as f32, [0.0, 0.0])
            .into_iter()
            .map(|v| {
                let v = if invert { 1.0 - v } else { v };
                // Match the command's GRAY8 mask rounding, before display interpolation.
                (v.clamp(0.0, 1.0) * 255.0 + 0.5).floor() / 255.0
            })
            .collect();
        scope_preview(proxy, params, &mut mask);
        return Ok(mask);
    }
    if params.get("select").and_then(Value::as_str) == Some("outOfGamut") {
        let pv = app.session.color.proof(proxy.id);
        let (m, _) = photocraft_engine::color_cmds::gamut_mask(proxy, &pv.setup, pv.gamut_threshold).map_err(|e| e.to_string())?;
        let invert = params.get("invert").and_then(Value::as_bool).unwrap_or(false);
        let mut mask: Vec<f32> = m.iter().map(|v| f32::from(*v) / 255.0).map(|v| if invert { 1.0 - v } else { v }).collect();
        scope_preview(proxy, params, &mut mask);
        return Ok(mask);
    }
    let mut p = params.clone();
    // Eyedropper points are in document pixels; the proxy is 1/k the size.
    let (w, h) = (f64::from(area.width().max(1)), f64::from(area.height().max(1)));
    if let Some(o) = p.as_object_mut() {
        for key in ["points", "subtractPoints"] {
            if let Some(Value::Array(a)) = o.get_mut(key) {
                for v in a.iter_mut() {
                    if let Some([x, y]) = v.as_array().map(Vec::as_slice).and_then(|s| match s {
                        [x, y] => Some([x.as_f64()?, y.as_f64()?]),
                        _ => None,
                    }) {
                        *v = json!([(x / f64::from(k)).floor().clamp(0.0, w - 1.0), (y / f64::from(k)).floor().clamp(0.0, h - 1.0)]);
                    }
                }
            }
        }
    }
    let mut s = photocraft_engine::Session::new();
    s.tools = app.session.tools.clone();
    // Preserve the proxy's pre-existing selection so preview and OK use the same scope.
    s.add_document(proxy.clone(), None);
    if let Some(id) = app.session.active().and_then(|d| d.active_layer) {
        let _ = s.select_layer(id);
    }
    s.execute(COMMAND, p).map_err(|e| e.to_string())?;
    let d = s.active().ok_or("no preview document")?;
    Ok(photocraft_algo::selection::mask_from_surface(d.doc.selection.as_ref(), area))
}

/// Record why the preview failed (logged once per new reason), or clear it.
fn report(app: &mut PhotocraftApp, error: Option<String>) {
    let Some(p) = app.color_range.as_mut() else { return };
    if p.error != error {
        if let Some(e) = &error {
            log::warn!("Color Range preview: {e}");
        }
        p.error = error;
    }
}

/// Refresh the cached proxy / textures for the active document and the dialog's params. Returns
/// (texture to show, its size in proxy pixels, k).
fn preview(app: &mut PhotocraftApp, ctx: &egui::Context, f: &Map<String, Value>) -> Option<(egui::TextureId, [usize; 2], u32)> {
    let (doc_id, revision, doc) = {
        let st = app.session.active()?;
        (st.doc.id, st.revision, st.doc.clone())
    };
    if !matches!(&app.color_range, Some(p) if p.doc == doc_id && p.revision == revision) {
        let side = doc.size.width.max(doc.size.height).max(1);
        let k = side.div_ceil(PREVIEW).max(1);
        let proxy = Arc::new(crate::proxy::proxy_document(&doc, k));
        app.color_range = Some(Preview {
            doc: doc_id,
            revision,
            k,
            proxy,
            image: None,
            mask_key: 0,
            mask: None,
            error: None,
            canvas_proxy: None,
            overlay: None,
            overlay_key: 0,
        });
    }
    let image_view = (s(f, "__view", "selection") == "image") ^ ctx.input(|i| i.modifiers.command || (!cfg!(target_os = "macos") && i.modifiers.ctrl));
    let params = params(f);
    let key = query_key(app, &params);
    let (proxy, k) = {
        let p = app.color_range.as_ref()?;
        (p.proxy.clone(), p.k)
    };
    let (w, h) = (proxy.size.width as usize, proxy.size.height as usize);
    if image_view {
        if app.color_range.as_ref().is_some_and(|p| p.image.is_none()) {
            let buf = photocraft_compose::thumbnail_buffer(&proxy, proxy.size.width.max(proxy.size.height));
            let thumb = match app.session.color.canvas_display(&proxy) {
                Ok(display) => display.to_rgba8(&buf),
                Err(e) => {
                    report(app, Some(e.to_string()));
                    return None;
                }
            };
            let size = [thumb.width as usize, thumb.height as usize];
            if size != [w, h] || thumb.pixels.len() != w * h * 4 {
                report(app, Some(format!("the image thumbnail is {}×{}, expected {w}×{h}", thumb.width, thumb.height)));
                return None;
            }
            let img = egui::ColorImage::from_rgba_unmultiplied(size, &thumb.pixels);
            let tex = ctx.load_texture("color-range-image", img, egui::TextureOptions::LINEAR);
            if let Some(p) = app.color_range.as_mut() {
                p.image = Some(tex);
            }
        }
        return app.color_range.as_ref()?.image.as_ref().map(|t| (t.id(), [w, h], k));
    }
    if app.color_range.as_ref().is_some_and(|p| p.mask_key != key || p.mask.is_none()) {
        let mask = match proxy_mask(app, &proxy, k, &params) {
            Ok(m) => {
                report(app, None);
                m
            }
            Err(e) => {
                report(app, Some(e));
                vec![0.0; w * h]
            }
        };
        if mask.len() != w * h {
            report(app, Some(format!("the selection preview has {} pixels, expected {}", mask.len(), w * h)));
            return None;
        }
        let px: Vec<Color32> = mask.iter().map(|v| Color32::from_gray((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)).collect();
        let img = egui::ColorImage::new([w, h], px);
        let p = app.color_range.as_mut()?;
        match &mut p.mask {
            Some(t) => t.set(img, egui::TextureOptions::LINEAR),
            None => p.mask = Some(ctx.load_texture("color-range-mask", img, egui::TextureOptions::LINEAR)),
        }
        p.mask_key = key;
    }
    app.color_range.as_ref()?.mask.as_ref().map(|t| (t.id(), [w, h], k))
}

/// Photoshop-style radio button.
fn radio(ui: &mut egui::Ui, on: bool, label: &str) -> egui::Response {
    let t = Tokens::get(ui.ctx());
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 5.0;
        let (r, resp) = ui.allocate_exact_size(vec2(14.0, 14.0), Sense::click());
        ui.painter().circle_stroke(r.center(), 5.5, Stroke::new(1.2, if on { t.accent } else { t.text_faint }));
        if on {
            ui.painter().circle_filled(r.center(), 3.0, t.accent);
        }
        let l = ui
            .add(egui::Label::new(egui::RichText::new(tl!(&label)).color(t.text_dim)).selectable(false).sense(Sense::click()))
            .on_hover_cursor(egui::CursorIcon::PointingHand);
        resp.union(l)
    })
    .inner
}

/// Eyedropper button: the pipette icon with a +/− badge; labelled for accessibility.
fn eyedropper(ui: &mut egui::Ui, badge: &str, selected: bool, label: &str) -> egui::Response {
    let t = Tokens::get(ui.ctx());
    let r = crate::icons::button(ui, "pipette", 26.0, selected, label);
    if !badge.is_empty() {
        let at = r.rect.right_bottom() - vec2(5.0, 6.0);
        ui.painter().text(at, egui::Align2::CENTER_CENTER, badge, crate::theme::semibold(12.0), if selected { t.accent_text } else { t.text });
    }
    r.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Button, ui.is_enabled(), selected, label));
    r
}

fn set_points(f: &mut Map<String, Value>, k: &str, pts: &[[f64; 2]]) {
    f.insert(k.into(), json!(pts));
}

/// Dialog body.
pub fn body(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    let t = Tokens::get(ui.ctx());
    let c = controls(f);
    let select = s(f, "select", "sampledColors").to_string();
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(tl!("Select:")).color(t.text_dim));
        let mut cur: &str = &select;
        if widgets::dropdown(ui, "color-range-select", &mut cur, SELECTS, 170.0) {
            f.insert("select".into(), json!(cur));
        }
    });
    ui.add_space(4.0);
    ui.add_enabled_ui(c.sampling, |ui| {
        let mut on = flag(f, "localized");
        if widgets::checkbox(ui, &mut on, tl!("Localized Color Clusters")).changed() {
            f.insert("localized".into(), json!(on));
        }
    });
    ui.add_space(4.0);
    ui.add_enabled_ui(c.fuzziness, |ui| {
        if c.tonal {
            let mut v = num(f, "toneFuzziness", 20.0);
            if widgets::slider_row(ui, tl!("Fuzziness:"), &mut v, 0.0..=100.0, "%", None).changed() {
                f.insert("toneFuzziness".into(), json!(v.round()));
            }
        } else {
            let mut v = num(f, "fuzziness", 40.0);
            if widgets::slider_row(ui, tl!("Fuzziness:"), &mut v, 0.0..=200.0, "", None).changed() {
                f.insert("fuzziness".into(), json!(v.round()));
            }
        }
    });
    if c.range {
        let mut v = num(f, "range", 100.0);
        if widgets::slider_row(ui, tl!("Range:"), &mut v, 0.0..=100.0, "%", None).changed() {
            f.insert("range".into(), json!(v.round()));
        }
    }
    if c.tonal {
        let mut level = |ui: &mut egui::Ui, key: &str, label: &str, d: f32| {
            let mut v = num(f, key, d);
            if widgets::slider_row(ui, label, &mut v, 0.0..=255.0, "", None).changed() {
                f.insert(key.into(), json!(v.round()));
            }
        };
        match select.as_str() {
            "shadows" => level(ui, "shadowsLevel", "Shadows up to:", 65.0),
            "highlights" => level(ui, "highlightsLevel", "Highlights from:", 190.0),
            _ => {
                level(ui, "midtonesLow", tl!("Midtones from:"), 105.0);
                level(ui, "midtonesHigh", tl!("Midtones to:"), 150.0);
            }
        }
    }
    ui.add_space(6.0);
    ui.horizontal_top(|ui| {
        // Preview: the selection as a grayscale mask, or the image to pick colours from.
        let box_size = vec2(PREVIEW as f32, PREVIEW as f32);
        let (frame, resp) = ui.allocate_exact_size(box_size, Sense::click_and_drag());
        ui.painter().rect_filled(frame, 0.0, t.canvas);
        let shown = preview(app, ui.ctx(), f);
        if let Some((tex, [w, h], k)) = shown {
            let scale = (box_size.x / w.max(1) as f32).min(box_size.y / h.max(1) as f32);
            let img_rect = Rect::from_center_size(frame.center(), vec2(w as f32 * scale, h as f32 * scale));
            ui.painter().image(tex, img_rect, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
            // Eyedropper on the preview (Photoshop also samples in the preview area).
            // Response::clicked may consume accessibility input (a write lock): query it
            // before reading PointerState, never from inside Ui::input's read lock.
            let clicked = resp.clicked();
            let held = resp.is_pointer_button_down_on();
            let sample =
                ui.input(|i| (clicked && i.pointer.primary_pressed()) || (held && (i.pointer.primary_pressed() || i.pointer.delta() != egui::Vec2::ZERO)));
            if let Some(p) = resp.interact_pointer_pos().filter(|p| c.sampling && sample && img_rect.contains(*p)) {
                let at = [
                    ((p.x - img_rect.left()) / scale).floor().max(0.0) as f64 * f64::from(k) + f64::from(k / 2),
                    ((p.y - img_rect.top()) / scale).floor().max(0.0) as f64 * f64::from(k) + f64::from(k / 2),
                ];
                pick(app, f, at, ui.input(|i| i.modifiers));
            }
        }
        // A preview that can't be drawn says why (#145) instead of staying blank.
        if let Some(e) = app.color_range.as_ref().and_then(|p| p.error.clone()) {
            ui.painter().rect_filled(frame, 0.0, t.canvas);
            let msg = format!(
                "Preview unavailable:
{e}"
            );
            let g = ui.painter().layout(msg, egui::FontId::proportional(11.0), t.text_faint, frame.width() - 16.0);
            ui.painter().galley(frame.center() - g.size() / 2.0, g, t.text_faint);
        }
        ui.painter().rect_stroke(frame, 0.0, Stroke::new(1.0, t.field_border), StrokeKind::Outside);
        let label = match app.color_range.as_ref().and_then(|p| p.error.as_deref()) {
            Some(e) => format!("Color Range preview unavailable: {e}"),
            None => tl!("Color Range preview").to_string(),
        };
        resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Image, ui.is_enabled(), &label));
        ui.add_space(10.0);
        ui.vertical(|ui| {
            ui.add_enabled_ui(c.sampling, |ui| {
                let tool = s(f, "__tool", "sample").to_string();
                ui.horizontal(|ui| {
                    for (id, badge, label) in
                        [("sample", "", tl!("Eyedropper")), ("add", "+", tl!("Add to Sample")), ("subtract", "−", tl!("Subtract from Sample"))]
                    {
                        if eyedropper(ui, badge, tool == id, label).clicked() {
                            f.insert("__tool".into(), json!(id));
                        }
                    }
                });
                let (add, sub) = (points(f, "points").len(), points(f, "subtractPoints").len());
                let what = match (add, sub) {
                    (0, 0) => tl!("Sample: foreground colour").to_string(),
                    (a, 0) => crate::i18n::trn(crate::i18n::current(), a as u64, "{n} sample", "{n} samples"),
                    (a, s) => crate::i18n::fmt(
                        tl!("{added}, {removed} subtracted"),
                        &[("added", &crate::i18n::trn(crate::i18n::current(), a as u64, "{n} sample", "{n} samples")), ("removed", &s.to_string())],
                    ),
                };
                ui.label(egui::RichText::new(what).size(11.0).color(t.text_faint));
                if add + sub > 0 && widgets::secondary_button(ui, tl!("Clear Samples"), 0.0).clicked() {
                    set_points(f, "points", &[]);
                    set_points(f, "subtractPoints", &[]);
                    f.insert("__order".into(), json!([]));
                }
            });
            ui.add_space(8.0);
            let mut inv = flag(f, "invert");
            // Selection invert (≠ Image › Adjustments › Invert → Đảo màu).
            if widgets::checkbox(ui, &mut inv, crate::i18n::tr_ctx(crate::i18n::current(), "colorRange", "Invert")).changed() {
                f.insert("invert".into(), json!(inv));
            }
        });
    });
    ui.add_space(4.0);
    let view = s(f, "__view", "selection").to_string();
    ui.horizontal(|ui| {
        if radio(ui, view == "selection", "Selection").clicked() {
            f.insert("__view".into(), json!("selection"));
        }
        ui.add_space(8.0);
        if radio(ui, view == "image", "Image").clicked() {
            f.insert("__view".into(), json!("image"));
        }
    });
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(tl!("Selection Preview:")).color(t.text_dim));
        let mode = s(f, "__selectionPreview", "none").to_string();
        let mut chosen = mode.as_str();
        if widgets::dropdown(ui, "color-range-selection-preview", &mut chosen, PREVIEWS, 170.0) {
            f.insert("__selectionPreview".into(), json!(chosen));
        }
    });
    if app.session.active().is_none() {
        ui.label(egui::RichText::new(tl!("Open a document to select a colour range.")).color(t.text_faint));
    }
}

/// A selection view is an overlay only: eyedroppers always sample the original image below it.
/// White = selected, black = unselected, gray = partial; matte/Quick Mask cover unselected pixels.
fn overlay_pixel(mode: &str, coverage: f32) -> Color32 {
    let coverage = coverage.clamp(0.0, 1.0);
    let q = |v: f32| (v * 255.0 + 0.5) as u8;
    match mode {
        "grayscale" => Color32::from_gray(q(coverage)),
        "blackMatte" => Color32::from_black_alpha(q(1.0 - coverage)),
        "whiteMatte" => Color32::from_rgba_unmultiplied(255, 255, 255, q(1.0 - coverage)),
        "quickMask" => Color32::from_rgba_unmultiplied(255, 0, 0, q(0.5 * (1.0 - coverage))),
        _ => Color32::TRANSPARENT,
    }
}

/// Photoshop's Selection Preview on the image, shared by CPU/GPU canvases and all platforms.
/// This never edits the real selection, image, channels or history; closing the dialog removes it.
pub(crate) fn paint_selection_preview(app: &mut PhotocraftApp, ctx: &egui::Context, painter: &egui::Painter, doc: DocId, image: Rect, flip: bool) {
    let Some(fields) = app.ui.dialogs.last().filter(|d| owns(&d.fields)).map(|d| d.fields.clone()) else { return };
    let mode = s(&fields, "__selectionPreview", "none");
    if mode == "none" || !app.session.active().is_some_and(|st| st.doc.id == doc) {
        return;
    }
    // Refresh the document/revision cache before borrowing its canvas proxy.
    let _ = preview(app, ctx, &fields);
    if app.color_range.as_ref().is_some_and(|p| p.canvas_proxy.is_none()) {
        let Some(original) = app.session.active().map(|st| st.doc.clone()) else { return };
        let k = original.size.width.max(original.size.height).max(1).div_ceil(CANVAS_PREVIEW);
        let proxy = Arc::new(crate::proxy::proxy_document(&original, k));
        if let Some(p) = app.color_range.as_mut() {
            p.canvas_proxy = Some((k, proxy));
        }
    }
    let params = params(&fields);
    let key = (query_key(app, &params) ^ hash(mode)).max(1);
    if app.color_range.as_ref().is_some_and(|p| p.overlay_key != key || p.overlay.is_none()) {
        let Some((k, proxy)) = app.color_range.as_ref().and_then(|p| p.canvas_proxy.clone()) else { return };
        let mask = match proxy_mask(app, &proxy, k, &params) {
            Ok(mask) => mask,
            Err(e) => {
                report(app, Some(e));
                return;
            }
        };
        let size = [proxy.size.width as usize, proxy.size.height as usize];
        let pixels = mask.into_iter().map(|v| overlay_pixel(mode, v)).collect();
        let img = egui::ColorImage::new(size, pixels);
        if let Some(p) = app.color_range.as_mut() {
            match &mut p.overlay {
                Some(t) => t.set(img, egui::TextureOptions::LINEAR),
                None => p.overlay = Some(ctx.load_texture("color-range-canvas-preview", img, egui::TextureOptions::LINEAR)),
            }
            p.overlay_key = key;
        }
    }
    if let Some(tex) = app.color_range.as_ref().and_then(|p| p.overlay.as_ref()) {
        let uv = if flip { Rect::from_min_max(pos2(1.0, 0.0), pos2(0.0, 1.0)) } else { Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)) };
        painter.image(tex.id(), image, uv, Color32::WHITE);
    }
}

/// The top dialog's fields when it is Color Range in a mode that takes eyedropper samples.
fn sampling_top(app: &PhotocraftApp) -> Option<Map<String, Value>> {
    app.ui.dialogs.last().map(|d| &d.fields).filter(|f| owns(f) && controls(f).sampling).cloned()
}

/// The dialog's eyedropper on the canvas: with Color Range › Sampled Colors as the top dialog,
/// picks the colour at document point `at` (Shift adds to the sample, Alt subtracts), as a click
/// on the image does in Photoshop. Returns whether it picked.
pub fn pick_top(app: &mut PhotocraftApp, at: [f64; 2], mods: egui::Modifiers) -> bool {
    let Some(mut f) = sampling_top(app) else { return false };
    let Some(doc) = app.session.active().map(|st| &st.doc) else { return false };
    if !at[0].is_finite() || !at[1].is_finite() || at[0] < 0.0 || at[1] < 0.0 || at[0] >= f64::from(doc.size.width) || at[1] >= f64::from(doc.size.height) {
        return false;
    }
    if doc.selection.as_ref().is_some_and(|selection| selection.sample_channel(at[0].floor() as i32, at[1].floor() as i32, 0) <= 0.0) {
        return false;
    }
    pick(app, &mut f, at, mods);
    match app.ui.dialogs.last_mut() {
        Some(d) => {
            d.fields = f;
            true
        }
        None => false,
    }
}

/// The canvas under an open Color Range dialog is its eyedropper (Photoshop): the pointer `over`
/// shows the dialog's eyedropper (+ / − while Shift / Alt add or subtract), and a press at document
/// point `press` takes a sample. False when Color Range isn't the top dialog.
pub fn canvas_eyedropper(app: &mut PhotocraftApp, ctx: &egui::Context, over: egui::Pos2, press: Option<[f64; 2]>) -> bool {
    let Some(f) = sampling_top(app) else { return false };
    let mods = ctx.input(|i| i.modifiers);
    let tool = if mods.shift {
        "add"
    } else if mods.alt {
        "subtract"
    } else {
        s(&f, "__tool", "sample")
    };
    if app.session.prefs().cursors.other == photocraft_engine::prefs::OtherCursor::Precise {
        ctx.set_cursor_icon(egui::CursorIcon::Crosshair);
    } else {
        // The tip of the icon's pipette is at (2, 22) of its 24-unit box (as the Color Picker's).
        crate::icons::cursor(ctx, "pipette", over, vec2(2.0, 22.0) / 24.0, 20.0);
        ctx.set_cursor_icon(egui::CursorIcon::None);
    }
    let badge = match tool {
        "add" => "+",
        "subtract" => "−",
        _ => "",
    };
    if !badge.is_empty() {
        let p = ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("pc-color-range-badge")));
        let at = over + vec2(20.0, -4.0);
        p.text(at + vec2(1.0, 1.0), egui::Align2::CENTER_CENTER, badge, crate::theme::semibold(13.0), Color32::from_black_alpha(200));
        p.text(at, egui::Align2::CENTER_CENTER, badge, crate::theme::semibold(13.0), Color32::WHITE);
    }
    if let Some(at) = press
        && pick_top(app, at, mods)
    {
        ctx.request_repaint();
    }
    true
}

/// An eyedropper click at document pixel `at`: the plain eyedropper replaces the samples, the
/// + one (or Shift) adds one, the − one (or Alt) subtracts one.
pub fn pick(app: &PhotocraftApp, f: &mut Map<String, Value>, at: [f64; 2], mods: egui::Modifiers) {
    let Some(st) = app.session.active() else { return };
    let (w, h) = (f64::from(st.doc.size.width.max(1)), f64::from(st.doc.size.height.max(1)));
    let [x, y] = at;
    let at = [x.floor().clamp(0.0, w - 1.0), y.floor().clamp(0.0, h - 1.0)];
    if st.doc.selection.as_ref().is_some_and(|selection| selection.sample_channel(at[0] as i32, at[1] as i32, 0) <= 0.0) {
        return;
    }
    let tool = if mods.shift {
        "add"
    } else if mods.alt {
        "subtract"
    } else {
        s(f, "__tool", "sample")
    };
    // The click order matters: Photoshop applies additions and subtractions one after another.
    let mut order: Vec<Value> = f.get("__order").and_then(Value::as_array).cloned().unwrap_or_default();
    match tool {
        "add" => {
            let mut p = points(f, "points");
            p.push(at);
            set_points(f, "points", &p);
            order.push(json!("+"));
        }
        "subtract" => {
            let mut p = points(f, "subtractPoints");
            p.push(at);
            set_points(f, "subtractPoints", &p);
            order.push(json!("-"));
        }
        _ => {
            set_points(f, "points", &[at]);
            set_points(f, "subtractPoints", &[]);
            order = vec![json!("+")];
        }
    }
    f.insert("__order".into(), Value::Array(order));
}

/// The click order for the command (`order`), when it is consistent with the point lists.
fn order(f: &Map<String, Value>, add: usize, sub: usize) -> Option<Value> {
    let o = f.get("__order").and_then(Value::as_array)?;
    let plus = o.iter().filter(|v| v.as_str() == Some("+")).count();
    let minus = o.iter().filter(|v| v.as_str() == Some("-")).count();
    (plus == add && minus == sub && plus + minus == o.len()).then(|| Value::Array(o.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui_kittest::{
        Harness,
        kittest::{NodeT, Queryable},
    };
    use photocraft_doc::{Color, ColorMode, SampleType, Size};
    use photocraft_geom::Rect as GRect;

    /// 40 × 30: red block (0..20, 0..15), blue block (20..40, 0..15), black and white below.
    fn app_with_doc() -> PhotocraftApp {
        let mut doc = Document::with_background("cr", Size::new(40, 30), ColorMode::Rgb, SampleType::U8, Color::WHITE);
        let bg = doc.layers[0].surface_mut().unwrap();
        bg.fill_rect(GRect::new(0, 0, 20, 15), &[1.0, 0.0, 0.0, 1.0]);
        bg.fill_rect(GRect::new(20, 0, 40, 15), &[0.0, 0.0, 1.0, 1.0]);
        bg.fill_rect(GRect::new(0, 15, 20, 30), &[0.0, 0.0, 0.0, 1.0]);
        let mut s = photocraft_engine::Session::new();
        s.add_document(doc, None);
        PhotocraftApp::new(s, crate::Services::default())
    }

    fn coverage(app: &PhotocraftApp, x: i32, y: i32) -> f32 {
        app.session.active().unwrap().doc.selection.as_ref().map_or(0.0, |s| s.sample_channel(x, y, 0))
    }

    fn harness(app: PhotocraftApp) -> Harness<'static, PhotocraftApp> {
        let mut h = Harness::builder().with_size(egui::vec2(1200.0, 900.0)).build_ui_state(
            |ui, app| {
                crate::menus::menu_bar(app, ui);
                crate::dialogs::show(app, ui.ctx());
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
        h.run_steps(2);
        h
    }

    fn dialog_id(app: &PhotocraftApp) -> u64 {
        app.ui.dialogs.iter().find(|d| owns(&d.fields)).map(|d| d.id).expect("Color Range dialog open")
    }

    fn set(h: &mut Harness<'static, PhotocraftApp>, key: &str, v: Value) {
        let id = dialog_id(h.state());
        h.state_mut().ui.dialog_mut(id).unwrap().fields.insert(key.into(), v);
        h.run_steps(3);
    }

    /// OK / Cancel are painted buttons without accessibility labels; the dialog's keys do the same.
    fn ok(h: &mut Harness<'static, PhotocraftApp>) {
        h.key_press(egui::Key::Enter);
    }

    fn click(h: &mut Harness<'static, PhotocraftApp>, at: egui::Pos2) {
        h.hover_at(at);
        h.run_steps(1);
        h.drag_at(at);
        h.run_steps(1);
        h.drop_at(at);
    }

    fn disabled(h: &Harness<'static, PhotocraftApp>, label: &str) -> bool {
        h.get_by_label(label).accesskit_node().is_disabled()
    }

    #[test]
    fn preview_and_eyedropper_keep_the_original_selection_boundary() {
        let mut app = app_with_doc();
        app.session.execute("select.rect", json!({"x": 0, "y": 0, "width": 20, "height": 30})).unwrap();
        let doc = app.session.active().unwrap().doc.clone();
        for params in [json!({"select": "sampledColors", "color": "#ff0000", "fuzziness": 0}), json!({"select": "reds"})] {
            let mask = proxy_mask(&app, &doc, 1, &params).unwrap();
            assert_eq!(mask[5 * 40 + 5], 1.0, "red inside the selection stays visible");
            assert_eq!(mask[5 * 40 + 25], 0.0, "the blue outside the selection stays excluded");
        }
        for params in [json!({"select": "sampledColors", "color": "#ffffff", "invert": true}), json!({"select": "outOfGamut", "invert": true})] {
            let mask = proxy_mask(&app, &doc, 1, &params).unwrap();
            assert_eq!(mask[5 * 40 + 25], 0.0, "even inverted previews stay within the selection");
        }

        open(&mut app);
        assert!(!pick_top(&mut app, [25.0, 5.0], egui::Modifiers::NONE));
        let fields = &app.ui.dialogs.last().unwrap().fields;
        assert!(points(fields, "points").is_empty(), "a click outside the selection is ignored");
        assert!(pick_top(&mut app, [5.0, 5.0], egui::Modifiers::NONE));
        assert_eq!(points(&app.ui.dialogs.last().unwrap().fields, "points"), vec![[5.0, 5.0]]);
    }

    #[test]
    fn preview_cache_refreshes_after_foreground_and_proof_changes() {
        let mut app = app_with_doc();
        let p = json!({"select": "outOfGamut"});
        let mut previous = query_key(&app, &p);
        app.session.tools.foreground = [0.2, 0.3, 0.4, 1.0];
        let key = query_key(&app, &p);
        assert_ne!(key, previous);
        previous = key;
        for setup in [
            json!({"profile": "srgb"}),
            json!({"profile": "linear-srgb"}),
            json!({"profile": "linear-srgb", "intent": "perceptual"}),
            json!({"profile": "linear-srgb", "intent": "perceptual", "bpc": false}),
            json!({"profile": "linear-srgb", "intent": "perceptual", "bpc": false, "simulatePaper": true}),
        ] {
            app.run("view.proofSetup", setup).unwrap();
            let key = query_key(&app, &p);
            assert_ne!(key, previous, "changed proof settings must invalidate the preview");
            assert_eq!(key, query_key(&app, &p), "an unchanged ICC profile must keep its cache key");
            previous = key;
        }
        app.run("view.gamutWarning", json!({"on": true, "threshold": 8})).unwrap();
        assert_ne!(previous, query_key(&app, &p));
        previous = query_key(&app, &p);
        app.run("view.gamutWarning", json!({"on": true, "threshold": 16})).unwrap();
        assert_ne!(previous, query_key(&app, &p));
        assert_ne!(query_key(&app, &p), query_key(&app, &json!({"select": "outOfGamut", "invert": true})));
    }

    #[test]
    fn eyedropper_hover_reuses_preview_textures_without_sampling() {
        let mut h = Harness::builder().with_size(vec2(1200.0, 900.0)).build_eframe(|cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            app_with_doc()
        });
        h.state_mut().ui.views[0].zoom = 20.0;
        h.state_mut().ui.views[0].center = [20.0, 15.0];
        h.state_mut().ui.views[0].fit_pending = false;
        h.run_steps(3);
        let id = open(h.state_mut());
        set(&mut h, "__selectionPreview", json!("quickMask"));
        let fields = h.state().ui.dialogs.iter().find(|d| d.id == id).unwrap().fields.clone();
        let cached = |app: &PhotocraftApp| {
            let p = app.color_range.as_ref().unwrap();
            (p.mask_key, p.overlay_key, p.mask.as_ref().unwrap().id(), p.overlay.as_ref().unwrap().id(), Arc::as_ptr(&p.proxy))
        };
        let before = cached(h.state());
        let st = h.state().session.active().unwrap();
        let document = (st.revision, st.history.past_len());
        for i in 0..12 {
            let at = crate::canvas::ViewXform::active(h.state()).unwrap().to_screen(4.0 + i as f32 * 0.5, 15.0);
            h.hover_at(at);
            h.run_steps(1);
            assert_eq!(cached(h.state()), before, "hover must reuse the mask, overlay and proxy");
        }
        assert_eq!(h.state_mut().ui.dialog_mut(id).unwrap().fields, fields, "hover must not pick a colour");
        let st = h.state().session.active().unwrap();
        assert_eq!((st.revision, st.history.past_len()), document);
    }

    #[test]
    fn opens_from_the_select_menu_without_touching_the_document() {
        let mut h = harness(app_with_doc());
        let rev = h.state().session.active().unwrap().revision;
        h.get_by_label("Select").click();
        h.run_steps(2);
        h.get_by_label("Color Range…").click();
        h.run_steps(3);
        assert!(h.state().ui.dialogs.iter().any(|d| owns(&d.fields)), "Select › Color Range… must open the dialog");
        for label in ["Color Range", "Fuzziness:", "Localized Color Clusters", "Invert", "Selection", "Color Range preview", "Add to Sample"] {
            assert!(h.query_by_label(label).is_some(), "missing {label}");
        }
        assert_eq!(h.query_all_by_label("Image").count(), 2, "the Image menu and the Image preview radio");
        assert_eq!(h.state().session.active().unwrap().revision, rev, "opening must not edit the document");
        // Automation and the generic command dialog get the same dialog.
        let ctx = egui::Context::default();
        let r = crate::menus::invoke(h.state_mut(), &ctx, COMMAND, json!({})).unwrap();
        assert!(r.get("dialog").is_some(), "{r}");
    }

    #[test]
    fn select_mode_switches_the_enabled_controls() {
        let mut h = harness(app_with_doc());
        open(h.state_mut());
        h.run_steps(4);
        // Sampled Colors: Fuzziness, eyedroppers and Localized Color Clusters; Range only when localized.
        assert!(!disabled(&h, "Fuzziness:") && !disabled(&h, "Localized Color Clusters") && !disabled(&h, "Add to Sample"));
        assert!(h.query_by_label("Range:").is_none());
        h.get_by_label("Localized Color Clusters").click();
        h.run_steps(2);
        assert!(h.query_by_label("Range:").is_some(), "Range shows with Localized Color Clusters");
        // A hue family: no Fuzziness, eyedroppers or clusters (Photoshop disables them).
        set(&mut h, "select", json!("reds"));
        assert!(disabled(&h, "Fuzziness:") && disabled(&h, "Localized Color Clusters") && disabled(&h, "Add to Sample"));
        assert!(h.query_by_label("Range:").is_none());
        assert!(!disabled(&h, "Invert"));
        // Tones: Fuzziness and the tonal range.
        set(&mut h, "select", json!("shadows"));
        assert!(!disabled(&h, "Fuzziness:") && disabled(&h, "Localized Color Clusters"));
        assert!(h.query_by_label("Shadows up to:").is_some());
        set(&mut h, "select", json!("midtones"));
        assert!(h.query_by_label("Midtones from:").is_some() && h.query_by_label("Midtones to:").is_some());
        set(&mut h, "select", json!("outOfGamut"));
        assert!(disabled(&h, "Fuzziness:") && h.query_by_label("Midtones from:").is_none());
        // Skin Tones: Fuzziness (the colour one), no eyedroppers or clusters.
        set(&mut h, "select", json!("skinTones"));
        assert!(!disabled(&h, "Fuzziness:") && disabled(&h, "Localized Color Clusters") && disabled(&h, "Add to Sample"));
        let skin: Map<String, Value> = serde_json::from_value(json!({"select": "skinTones", "fuzziness": 25.0, "points": [[1, 1]]})).unwrap();
        assert_eq!(params(&skin), json!({"select": "skinTones", "invert": false, "fuzziness": 25.0}));
        assert_eq!(
            controls(&serde_json::from_value(json!({"select": "highlights"})).unwrap()),
            Controls { fuzziness: true, sampling: false, range: false, tonal: true }
        );
    }

    #[test]
    fn dialog_shrinks_back_when_a_mode_needs_fewer_controls() {
        let mut h = harness(app_with_doc());
        open(h.state_mut());
        h.run_steps(4);
        let area = egui::Id::new(("dialog", dialog_id(h.state())));
        let rect = |h: &Harness<'static, PhotocraftApp>| h.ctx.memory(|m| m.area_rect(area)).unwrap_or(egui::Rect::NOTHING);
        set(&mut h, "select", json!("outOfGamut"));
        let short = rect(&h);
        set(&mut h, "select", json!("midtones"));
        let tall = rect(&h);
        assert!(tall.height() > short.height() + 40.0, "Midtones adds two sliders: {short:?} → {tall:?}");
        assert_eq!(tall.min, short.min, "the dialog grows downward, it doesn't re-centre");
        set(&mut h, "select", json!("outOfGamut"));
        let back = rect(&h);
        assert!((back.height() - short.height()).abs() < 1.0, "the dialog must shrink back, not keep an empty band above OK: {short:?} → {tall:?} → {back:?}");
        assert_eq!(back.min, short.min);
    }

    #[test]
    fn ok_applies_one_undo_step() {
        let mut h = harness(app_with_doc());
        // The prior selection covers the left red square: Color Range is scoped to it (#3313).
        h.state_mut().run("select.rect", json!({"x": 0, "y": 0, "width": 20, "height": 30})).unwrap();
        let past = h.state().session.active().unwrap().history.past_len();
        open(h.state_mut());
        h.run_steps(4);
        set(&mut h, "select", json!("reds"));
        ok(&mut h);
        h.run_steps(2);
        assert!(!h.state().ui.dialogs.iter().any(|d| owns(&d.fields)), "OK closes the dialog");
        let app = h.state();
        assert_eq!((coverage(app, 5, 5), coverage(app, 30, 5), coverage(app, 2, 22)), (1.0, 0.0, 0.0));
        assert_eq!(app.session.active().unwrap().history.past_len(), past + 1, "one history step");
        assert_eq!(app.session.active().unwrap().history.undo_label(), Some("Color Range"));
        h.state_mut().run("edit.undo", json!({})).unwrap();
        let app = h.state();
        assert_eq!((coverage(app, 5, 5), coverage(app, 2, 22), coverage(app, 30, 5)), (1.0, 1.0, 0.0), "undo restores the previous selection");
    }

    #[test]
    fn cancel_keeps_the_previous_selection() {
        let mut h = harness(app_with_doc());
        h.state_mut().run("select.rect", json!({"x": 0, "y": 20, "width": 5, "height": 5})).unwrap();
        let (rev, past) = {
            let st = h.state().session.active().unwrap();
            (st.revision, st.history.past_len())
        };
        open(h.state_mut());
        h.run_steps(4);
        set(&mut h, "select", json!("blues"));
        h.get_by_label("Invert").click();
        h.run_steps(2);
        h.key_press(egui::Key::Escape);
        h.run_steps(2);
        assert!(h.state().ui.dialogs.is_empty());
        let app = h.state();
        let st = app.session.active().unwrap();
        assert_eq!((st.revision, st.history.past_len()), (rev, past));
        assert_eq!((coverage(app, 2, 22), coverage(app, 30, 5), coverage(app, 5, 5)), (1.0, 0.0, 0.0));
    }

    #[test]
    fn invert_flips_the_selection() {
        let mut h = harness(app_with_doc());
        open(h.state_mut());
        h.run_steps(4);
        set(&mut h, "select", json!("reds"));
        h.get_by_label("Invert").click();
        h.run_steps(2);
        let id = dialog_id(h.state());
        assert_eq!(h.state_mut().ui.dialog_mut(id).unwrap().fields.get("invert"), Some(&json!(true)));
        ok(&mut h);
        h.run_steps(2);
        let app = h.state();
        assert_eq!((coverage(app, 5, 5), coverage(app, 30, 5), coverage(app, 5, 22), coverage(app, 30, 22)), (0.0, 1.0, 1.0, 1.0));
    }

    #[test]
    fn eyedroppers_pick_on_the_preview() {
        let mut h = harness(app_with_doc());
        open(h.state_mut());
        h.run_steps(4);
        // 40 × 30 at 200 pt: 5 pt per pixel; the image is centred vertically (150 pt tall).
        let at = |h: &Harness<'static, PhotocraftApp>, x: f32, y: f32| {
            h.get_by_label("Color Range preview").rect().left_top() + egui::vec2(x * 5.0 + 2.5, 25.0 + y * 5.0 + 2.5)
        };
        let p = at(&h, 30.0, 5.0);
        click(&mut h, p);
        h.run_steps(2);
        let f = |h: &Harness<'static, PhotocraftApp>| h.state().ui.dialogs.iter().find(|d| owns(&d.fields)).unwrap().fields.clone();
        assert_eq!(points(&f(&h), "points"), vec![[30.0, 5.0]]);
        // Add to Sample: the red block too; the selection then covers both.
        h.get_by_label("Add to Sample").click();
        h.run_steps(1);
        let p = at(&h, 5.0, 5.0);
        click(&mut h, p);
        h.run_steps(2);
        assert_eq!(points(&f(&h), "points").len(), 2);
        // Subtract from Sample: take the blue back out.
        h.get_by_label("Subtract from Sample").click();
        h.run_steps(1);
        let p = at(&h, 35.0, 10.0);
        click(&mut h, p);
        h.run_steps(2);
        assert_eq!(points(&f(&h), "subtractPoints"), vec![[35.0, 10.0]]);
        let p = params(&f(&h));
        assert_eq!(p["select"], "sampledColors");
        assert_eq!(p["order"], json!(["+", "+", "-"]));
        ok(&mut h);
        h.run_steps(2);
        let app = h.state();
        // The sampled red stays fully selected.
        assert_eq!(coverage(app, 5, 5), 1.0);
        // Like Photoshop, subtracting a colour on the edge of the samples' range only trims that
        // edge by a fifth of the falloff: the blue stays mostly selected (Photoshop: 232 of 255
        // for the same case), black never was.
        let blue = coverage(app, 30, 5);
        assert!(blue > 0.85 && blue < 0.95, "{blue}");
        assert_eq!(coverage(app, 5, 22), 0.0);
    }

    #[test]
    fn selection_preview_modes_cover_only_unselected_pixels() {
        assert_eq!(overlay_pixel("grayscale", 1.0), Color32::WHITE);
        assert_eq!(overlay_pixel("grayscale", 0.0), Color32::BLACK);
        assert_eq!(overlay_pixel("grayscale", 0.5), Color32::from_gray(128));
        for mode in ["blackMatte", "whiteMatte", "quickMask"] {
            assert_eq!(overlay_pixel(mode, 1.0).a(), 0, "{mode}: selected pixels stay visible");
        }
        assert_eq!(overlay_pixel("blackMatte", 0.0), Color32::BLACK);
        assert_eq!(overlay_pixel("whiteMatte", 0.0), Color32::WHITE);
        assert_eq!(overlay_pixel("quickMask", 0.0).a(), 128);
        assert_eq!(overlay_pixel("quickMask", 0.5).a(), 64);
        assert_eq!(overlay_pixel("none", 0.0), Color32::TRANSPARENT);
    }

    #[test]
    fn preview_mode_is_view_state_and_cancel_preserves_selection() {
        let mut h = harness(app_with_doc());
        h.state_mut().run("select.rect", json!({"x": 0, "y": 20, "width": 5, "height": 5})).unwrap();
        let original = h.state().session.active().unwrap();
        let past = original.history.past_len();
        let revision = original.revision;
        let id = open(h.state_mut());
        h.run_steps(3);
        for (mode, _) in PREVIEWS {
            set(&mut h, "__selectionPreview", json!(mode));
            let f = &h.state().ui.dialogs.last().unwrap().fields;
            assert!(params(f).get("__selectionPreview").is_none());
            let painter = h.ctx.layer_painter(egui::LayerId::background());
            let doc = h.state().session.active().unwrap().doc.id;
            let ctx = h.ctx.clone();
            paint_selection_preview(h.state_mut(), &ctx, &painter, doc, Rect::from_min_size(pos2(0.0, 0.0), vec2(400.0, 300.0)), false);
            let st = h.state().session.active().unwrap();
            assert_eq!((st.revision, st.history.past_len()), (revision, past));
            assert_eq!(coverage(h.state(), 2, 22), 1.0);
        }
        h.state_mut().ui.close_dialog(id);
        let painter = h.ctx.layer_painter(egui::LayerId::background());
        let ctx = h.ctx.clone();
        let doc = h.state().session.active().unwrap().doc.id;
        paint_selection_preview(h.state_mut(), &ctx, &painter, doc, Rect::EVERYTHING, false);
        assert_eq!(coverage(h.state(), 2, 22), 1.0);
    }

    #[test]
    fn repeated_subtraction_clicks_are_not_discarded() {
        let mut app = app_with_doc();
        let id = open(&mut app);
        let mut f = app.ui.dialog_mut(id).unwrap().fields.clone();
        pick(&app, &mut f, [2.0, 2.0], egui::Modifiers::NONE);
        pick(&app, &mut f, [30.0, 2.0], egui::Modifiers::SHIFT);
        pick(&app, &mut f, [30.0, 2.0], egui::Modifiers::ALT);
        pick(&app, &mut f, [30.0, 2.0], egui::Modifiers::ALT);
        assert_eq!(params(&f)["order"], json!(["+", "+", "-", "-"]));
    }

    #[test]
    fn control_pointer_drag_samples_and_never_paints_under_the_dialog() {
        let mut h = harness(app_with_doc());
        let app = h.state_mut();
        open(app);
        let doc = app.session.active().unwrap();
        let before = doc.revision;
        let history = doc.history.past_len();
        let ctx = h.ctx.clone();
        for (events, shift) in [
            (json!([{"kind":"down","x":2,"y":2},{"kind":"move","x":3,"y":2},{"kind":"up","x":3,"y":2}]), false),
            (json!([{"kind":"down","x":30,"y":2},{"kind":"move","x":31,"y":2},{"kind":"up","x":31,"y":2}]), true),
        ] {
            let (req, _) = crate::control::ControlRequest::new("ui.pointer", json!({"events":events,"shift":shift}));
            let _ = crate::control::handle(h.state_mut(), &ctx, &req);
        }
        let fields = &h.state().ui.dialogs.last().unwrap().fields;
        assert_eq!(points(fields, "points"), [[3.0, 2.0], [30.0, 2.0], [31.0, 2.0]]);
        let doc = h.state().session.active().unwrap();
        assert_eq!((doc.revision, doc.history.past_len()), (before, history));
        assert!(h.state().drag.is_none());
        assert!(!pick_top(h.state_mut(), [-1.0, 0.0], egui::Modifiers::NONE));
    }

    #[test]
    fn modifier_temporarily_swaps_image_and_selection_thumbnail() {
        let mut h = harness(app_with_doc());
        let id = open(h.state_mut());
        h.run_steps(3);
        let f = h.state().ui.dialogs.last().unwrap().fields.clone();
        let ctx = h.ctx.clone();
        let selection = preview(h.state_mut(), &ctx, &f).unwrap().0;
        h.event(egui::Event::ModifiersChanged(egui::Modifiers::COMMAND));
        h.run_steps(2);
        let image = preview(h.state_mut(), &ctx, &f).unwrap().0;
        assert_ne!(image, selection);
        assert_eq!(h.state().ui.dialogs.iter().find(|d| d.id == id).unwrap().fields["__view"], "selection");
        h.event(egui::Event::ModifiersChanged(egui::Modifiers::NONE));
        h.run_steps(2);
        assert_eq!(preview(h.state_mut(), &ctx, &f).unwrap().0, selection);
    }

    /// A sample between thumbnail pixels must use the original pixel, not its neighbour.
    #[test]
    fn reduced_preview_matches_the_original_eyedropper_sample() {
        let mut doc = Document::with_background("fine detail", Size::new(400, 4), ColorMode::Rgb, SampleType::U8, Color::rgb(1.0, 0.0, 0.0));
        let surface = doc.layers[0].surface_mut().unwrap();
        surface.fill_rect(GRect::new(200, 0, 400, 4), &[0.0, 0.0, 1.0, 1.0]);
        surface.fill_rect(GRect::new(101, 0, 102, 4), &[0.0, 0.0, 1.0, 1.0]);
        let mut session = photocraft_engine::Session::new();
        session.add_document(doc, None);
        let mut app = PhotocraftApp::new(session, crate::Services::default());
        let id = open(&mut app);
        let mut fields = app.ui.dialog_mut(id).unwrap().fields.clone();
        pick(&app, &mut fields, [101.0, 1.0], egui::Modifiers::NONE);
        fields.insert("fuzziness".into(), json!(0.0));
        let original = app.session.active().unwrap().doc.clone();
        let proxy = crate::proxy::proxy_document(&original, 2);
        let mut old = photocraft_engine::Session::new();
        old.add_document(proxy.clone(), None);
        old.execute(COMMAND, json!({"points": [[50, 0]], "fuzziness": 0})).unwrap();
        assert_eq!(
            old.active().unwrap().doc.selection.as_ref().unwrap().sample_channel(150, 0, 0),
            0.0,
            "sampling the thumbnail instead selects the red neighbour"
        );
        let shown = proxy_mask(&app, &proxy, 2, &params(&fields)).unwrap();
        confirm(&mut app, &fields).unwrap();
        assert_eq!(coverage(&app, 300, 0), 1.0);
        assert_eq!(shown[150], coverage(&app, 300, 0), "preview must show the same blue pixels as OK");
        assert_eq!(shown[0], coverage(&app, 0, 0), "preview must leave the red background unselected");
    }

    /// `sampleAllLayers: false` judges the active layer in the preview, as OK does.
    #[test]
    fn preview_honours_sample_all_layers_like_ok() {
        let mut app = app_with_doc();
        let doc = app.session.active().unwrap().doc.clone();
        let mut top = photocraft_doc::Layer::new("blue cover", photocraft_doc::LayerContent::Raster(doc.layers[0].surface().unwrap().clone()));
        top.surface_mut().unwrap().fill_rect(GRect::new(0, 0, 40, 30), &[0.0, 0.0, 1.0, 1.0]);
        app.session
            .edit("layers", |doc, _| {
                doc.layers.push(top);
                Ok(())
            })
            .unwrap();
        for all_layers in [false, true] {
            let p = json!({"select": "sampledColors", "points": [[5, 5]], "fuzziness": 0, "sampleAllLayers": all_layers});
            let proxy = crate::proxy::proxy_document(&app.session.active().unwrap().doc, 1);
            let shown = proxy_mask(&app, &proxy, 1, &p).unwrap();
            app.run(COMMAND, p).unwrap();
            for (x, y) in [(5, 5), (30, 5), (5, 20), (30, 20)] {
                assert_eq!(shown[y * 40 + x], coverage(&app, x as i32, y as i32), "all_layers={all_layers} at ({x}, {y})");
            }
        }
    }

    #[test]
    fn params_send_only_what_the_mode_uses() {
        let mut app = app_with_doc();
        let id = open(&mut app);
        let mut f = app.ui.dialog_mut(id).unwrap().fields.clone();
        assert_eq!(params(&f), json!({"select": "sampledColors", "invert": false, "fuzziness": 40.0}));
        // Localized needs a picked point.
        f.insert("localized".into(), json!(true));
        assert!(params(&f).get("localized").is_none());
        pick(&app, &mut f, [3.0, 4.0], egui::Modifiers::NONE);
        pick(&app, &mut f, [99.0, -4.0], egui::Modifiers::SHIFT);
        pick(&app, &mut f, [1.0, 1.0], egui::Modifiers::ALT);
        assert_eq!(
            params(&f),
            json!({"select": "sampledColors", "invert": false, "fuzziness": 40.0, "points": [[3.0, 4.0], [39.0, 0.0]], "subtractPoints": [[1.0, 1.0]], "order": ["+", "+", "-"], "localized": true, "range": 100.0})
        );
        f.insert("select".into(), json!("midtones"));
        f.insert("midtonesLow".into(), json!(160.0));
        assert_eq!(params(&f), json!({"select": "midtones", "invert": false, "fuzziness": 20.0, "tonalRange": [150.0, 160.0]}));
        f.insert("select".into(), json!("highlights"));
        assert_eq!(params(&f)["tonalRange"], json!(190.0));
        f.insert("select".into(), json!("cyans"));
        assert_eq!(params(&f), json!({"select": "cyans", "invert": false}));
        // Every Select entry is accepted by the engine.
        for (v, _) in SELECTS {
            f.insert("select".into(), json!(v));
            app.run(COMMAND, params(&f)).unwrap();
        }
    }

    /// #145: the whole app without a GPU (no wgpu render state: the CPU canvas, as on a Linux
    /// machine whose adapter can't run the GPU canvas), on a small window. Select › Color Range…
    /// from the menu opens the dialog on screen and draws both previews.
    #[test]
    fn opens_and_previews_in_the_full_app_without_a_gpu() {
        let mut h = Harness::builder().with_size(egui::vec2(1024.0, 600.0)).build_eframe(|cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            assert!(cc.wgpu_render_state.is_none(), "this test covers the no-GPU path");
            app_with_doc()
        });
        h.run_steps(6);
        assert!(h.state().gpu.is_none(), "the CPU canvas");
        h.get_all_by_label("Select").next().expect("the Select menu").click();
        h.run_steps(3);
        h.get_by_label("Color Range…").click();
        h.run_steps(6);
        let id = dialog_id(h.state());
        let area = h.ctx.memory(|m| m.area_rect(egui::Id::new(("dialog", id)))).expect("the dialog is drawn");
        assert!(h.ctx.content_rect().contains_rect(area), "the dialog is on screen: {area:?}");
        assert!(h.query_by_label("Color Range preview").is_some());
        let p = h.state().color_range.as_ref().expect("the preview is cached");
        assert!(p.mask.is_some() && p.error.is_none(), "the selection preview is drawn: {:?}", p.error);
        set(&mut h, "__view", json!("image"));
        let p = h.state().color_range.as_ref().unwrap();
        assert!(p.image.is_some() && p.error.is_none(), "the image preview is drawn: {:?}", p.error);
        set(&mut h, "select", json!("reds"));
        ok(&mut h);
        h.run_steps(3);
        assert_eq!(coverage(h.state(), 5, 5), 1.0);
    }

    #[test]
    fn a_preview_failure_is_shown_not_silent() {
        let mut h = harness(app_with_doc());
        open(h.state_mut());
        h.run_steps(3);
        // The engine refuses these params: the preview says so instead of staying blank.
        let id = dialog_id(h.state());
        h.state_mut().ui.dialog_mut(id).unwrap().fields.insert("select".into(), json!("noSuchMode"));
        h.run_steps(3);
        let err = h.state().color_range.as_ref().and_then(|p| p.error.clone());
        assert!(err.is_some(), "the failure is recorded");
        assert!(h.query_by_label_contains("Color Range preview unavailable").is_some(), "and shown");
        set(&mut h, "select", json!("reds"));
        assert!(h.state().color_range.as_ref().unwrap().error.is_none(), "and cleared once the preview works again");
    }
}
