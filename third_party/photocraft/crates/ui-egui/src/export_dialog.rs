//! File › Export › Export As… (and Quick Export as PNG): format, quality, transparency and scale,
//! with a preview and an estimated file size. The estimate encodes a small proxy and scales by
//! pixel count, so the dialog stays instant on 36 MP documents.

use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Mutex;

use photocraft_doc::Document;
use serde_json::{Map, Value, json};

use crate::state::DialogKind;
use crate::theme::Tokens;
use crate::{ExportSettings, PhotocraftApp};

const FORMATS: [(&str, &str); 7] = [("png", "PNG"), ("jpg", "JPG"), ("psd", "PSD"), ("webp", "WebP"), ("tif", "TIFF"), ("tga", "TGA"), ("ico", "ICO")];

/// Preferences ▸ remembered dialog settings: Export As writes the settings it was used with
/// last under its command id and seeds the next dialog from them, as Fill, Liquify and the
/// filter dialogs do.
pub(crate) const COMMAND: &str = "file.export.exportAs";

/// The dialog is seeded from the settings it was used with last (see [`COMMAND`]); a value that
/// no longer matches a control (a hand-edited preferences file) is left at its default.
fn seed_remembered(app: &PhotocraftApp, f: &mut Map<String, Value>) {
    let Some(Value::Object(saved)) = app.session.prefs().dialogs.get(COMMAND) else { return };
    if let Some(v) = saved.get("format").and_then(Value::as_str).filter(|v| FORMATS.iter().any(|(k, _)| k == v)) {
        f.insert("format".into(), json!(v));
    }
    if let Some(v) = saved.get("quality").and_then(Value::as_f64).filter(|v| (1.0..=100.0).contains(v)) {
        f.insert("quality".into(), json!(v));
    }
    if let Some(v) = saved.get("scale").and_then(Value::as_f64).filter(|v| (1.0..=1000.0).contains(v)) {
        f.insert("scale".into(), json!(v));
    }
    for key in ["lossless", "transparency"] {
        if let Some(v) = saved.get(key).and_then(Value::as_bool) {
            f.insert(key.into(), json!(v));
        }
    }
    if let Some(v) = saved.get("metadata").and_then(Value::as_str).filter(|v| ["none", "all"].contains(v)) {
        f.insert("metadata".into(), json!(v));
    }
    if let Some(v) = saved.get("tgaBits").and_then(Value::as_str).filter(|v| ["auto", "24", "32"].contains(v)) {
        f.insert("tgaBits".into(), json!(v));
    }
}

/// Record the settings for the next Export As (see [`seed_remembered`]).
fn remember(app: &mut PhotocraftApp, f: &Map<String, Value>) {
    const KEYS: [&str; 7] = ["format", "quality", "lossless", "transparency", "scale", "metadata", "tgaBits"];
    let saved: Map<String, Value> = KEYS.iter().filter_map(|k| f.get(*k).map(|v| ((*k).to_string(), v.clone()))).collect();
    app.session.prefs.edit(|p| {
        p.dialogs.insert(COMMAND.into(), Value::Object(saved));
    });
}

pub fn open(app: &mut PhotocraftApp) -> Result<u64, String> {
    let st = app.session.active().ok_or("no document")?;
    let mut f = Map::new();
    f.insert("__export".into(), json!(true));
    f.insert("__label".into(), json!("Export As"));
    f.insert("format".into(), json!("png"));
    f.insert("quality".into(), json!(app.session.prefs().export.jpeg_quality));
    f.insert("lossless".into(), json!(app.session.prefs().export.webp_lossless));
    f.insert("transparency".into(), json!(true));
    f.insert("scale".into(), json!(100));
    f.insert("metadata".into(), json!("none"));
    f.insert("tgaBits".into(), json!("auto"));
    f.insert("__w".into(), json!(st.doc.size.width));
    f.insert("__h".into(), json!(st.doc.size.height));
    seed_remembered(app, &mut f);
    Ok(app.ui.open_dialog(DialogKind::Command, f))
}

/// Layer › Export As…: the same dialog for just the active layer (trimmed to its pixels).
pub fn open_layer(app: &mut PhotocraftApp, layer: photocraft_doc::LayerId) -> Result<u64, String> {
    let st = app.session.active().ok_or("no document")?;
    let ldoc = photocraft_engine::layer_menu_cmds::layer_document(&st.doc, layer).map_err(|e| e.to_string())?;
    let id = open(app)?;
    if let Some(d) = app.ui.dialogs.iter_mut().find(|d| d.id == id) {
        d.fields.insert("__layer".into(), json!(layer.0));
        d.fields.insert("__label".into(), json!(crate::i18n::fmt(crate::i18n::t("Export As: {name}"), &[("name", &ldoc.name)])));
        d.fields.insert("__w".into(), json!(ldoc.size.width));
        d.fields.insert("__h".into(), json!(ldoc.size.height));
    }
    Ok(id)
}

/// Layer › Export As… with several layers (artboards, say) selected: the same dialog, then one
/// file per layer with its settings (#3629). The preview shows the first layer.
pub fn open_layers(app: &mut PhotocraftApp, layers: &[photocraft_doc::LayerId]) -> Result<u64, String> {
    let (first, n) = match layers {
        [] => return Err("no selected layers".into()),
        [one] => return open_layer(app, *one),
        [first, ..] => (*first, layers.len()),
    };
    let id = open_layer(app, first)?;
    if let Some(d) = app.ui.dialogs.iter_mut().find(|d| d.id == id) {
        d.fields.insert("__layers".into(), json!(layers.iter().map(|l| l.0).collect::<Vec<_>>()));
        let template = "Export As: {n} layers";
        d.fields.insert("__label".into(), json!(crate::i18n::fmt(tl!(template), &[("n", &n.to_string())])));
    }
    Ok(id)
}

/// Layer › Quick Export as PNG.
pub fn quick_export_layer_png(app: &mut PhotocraftApp, layer: photocraft_doc::LayerId) -> Result<Value, String> {
    let mut f = Map::new();
    f.insert("format".into(), json!("png"));
    f.insert("transparency".into(), json!(true));
    f.insert("scale".into(), json!(100));
    f.insert("__layer".into(), json!(layer.0));
    confirm(app, &f)
}

/// Export each selected layer separately, with one destination prompt for the batch.
/// The chosen path names the first file; subsequent files have numbered sibling names.
pub fn quick_export_selected_layers_png(app: &mut PhotocraftApp) -> Result<Value, String> {
    let st = app.session.active().ok_or("no document")?;
    let layers = st.selected_layers();
    if layers.len() == 1 {
        return quick_export_layer_png(app, layers[0]);
    }
    if layers.is_empty() {
        return Err("no selected layers".into());
    }
    // Capture all sources before showing the dialog: changing selection or document while it
    // is open cannot change which layers get exported.
    let docs: Vec<_> =
        layers.into_iter().map(|id| photocraft_engine::layer_menu_cmds::layer_document(&st.doc, id).map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
    let suggested = format!("{}.png", export_name(&st.doc.name));
    app.pick_save(&suggested, move |app, path| {
        if !path.to_ascii_lowercase().ends_with(".png") {
            return Err("Quick Export as PNG writes .png files".into());
        }
        let paths = batch_paths(&path, &docs);
        let mut exported = Vec::new();
        for (doc, path) in docs.iter().zip(paths) {
            let export = app.services.export.as_ref().ok_or("no exporter configured")?;
            let (bytes, warnings) = export(doc, &path, &ExportSettings { xmp_all: false, ..Default::default() })?;
            let write = app.services.write.as_mut().ok_or("no writer configured")?;
            write(&path, &bytes)?;
            crate::notices::io_warnings(app, &format!("Exported {}", crate::file_open::display_name(&path)), &warnings);
            exported.push(json!({"path": path, "bytes": bytes.len(), "warnings": warnings}));
        }
        app.ui.status = format!("Exported {} PNG files", exported.len());
        app.ui.status_error = false;
        Ok(json!({"files": exported}))
    })
}

fn export_name(name: &str) -> String {
    let name: String = name.chars().take(80).map(|c| if c.is_control() || "<>:\"/\\|?*".contains(c) { '_' } else { c }).collect();
    let name = name.trim().trim_matches('.');
    if name.is_empty() { "layer".into() } else { name.into() }
}

/// The files a multi-layer export writes: the chosen path, then numbered siblings named after
/// each layer, in the chosen path's format.
fn batch_paths(path: &str, docs: &[Document]) -> Vec<String> {
    let first = std::path::Path::new(path);
    let parent = first.parent().unwrap_or_else(|| std::path::Path::new(""));
    let stem = first.file_stem().and_then(|s| s.to_str()).unwrap_or("layers");
    let ext = first.extension().and_then(|e| e.to_str()).unwrap_or("png");
    docs.iter()
        .enumerate()
        .map(|(i, doc)| {
            if i == 0 {
                return path.to_string();
            }
            let name = export_name(&doc.name);
            let candidate = parent.join(format!("{stem}-{}-{name}.{ext}", i + 1));
            // The first file's overwrite is handled by the Save dialog; never overwrite sibling
            // files silently on desktop. Downloads on the web are named by the browser.
            #[cfg(not(target_arch = "wasm32"))]
            let candidate = {
                let mut candidate = candidate;
                let mut suffix = 1u64;
                while candidate.exists() {
                    candidate = parent.join(format!("{stem}-{}-{name}-{suffix}.{ext}", i + 1));
                    suffix = suffix.saturating_add(1);
                }
                candidate
            };
            candidate.to_string_lossy().into_owned()
        })
        .collect()
}

/// The document a dialog exports: the whole image, or one layer (`__layer`).
fn source_document_from(doc: Arc<Document>, f: &Map<String, Value>) -> Result<Arc<Document>, String> {
    match f.get("__layer").and_then(Value::as_u64) {
        Some(id) => photocraft_engine::layer_menu_cmds::layer_document(&doc, photocraft_doc::LayerId(id)).map(Arc::new).map_err(|e| e.to_string()),
        None => Ok(doc),
    }
}

fn source_document(app: &PhotocraftApp, f: &Map<String, Value>) -> Result<Arc<Document>, String> {
    source_document_from(app.session.active().ok_or("no document")?.doc.clone(), f)
}

fn s(f: &Map<String, Value>, k: &str) -> String {
    f.get(k).and_then(Value::as_str).unwrap_or_default().to_string()
}
fn n(f: &Map<String, Value>, k: &str, d: f64) -> f64 {
    f.get(k).and_then(Value::as_f64).unwrap_or(d)
}

/// The document as exported: scaled (engine Image Size, bicubic) and flattened over white when
/// transparency is off or the format has no alpha.
fn export_document(doc: &Document, f: &Map<String, Value>, max_side: Option<u32>) -> Result<Document, String> {
    let mut s = photocraft_engine::Session::new();
    s.add_document(doc.clone(), None);
    let mut scale = n(f, "scale", 100.0) / 100.0;
    if let Some(m) = max_side {
        let long = doc.size.width.max(doc.size.height) as f64 * scale;
        if long > m as f64 {
            scale *= m as f64 / long;
        }
    }
    // Both dimensions are sent: a scale can leave a 1 px width unchanged while halving the
    // height, and the engine would otherwise keep the aspect ratio from the width alone.
    let (w, h) = scaled_size(doc.size.width as f64, doc.size.height as f64, scale);
    if s_fmt(f) == "ico" && (w > 256.0 || h > 256.0) && max_side.is_none() {
        return Err("ICO export is limited to 256 × 256 pixels; reduce the export scale".into());
    }
    if (w - doc.size.width as f64).abs() >= 1.0 || (h - doc.size.height as f64).abs() >= 1.0 {
        s.execute("image.imageSize", json!({"width": w, "height": h, "resample": if max_side.is_some() { "bilinear" } else { "bicubic" }}))
            .map_err(|e| e.to_string())?;
    }
    let fmt = s_fmt(f);
    if fmt != "psd" && (!f.get("transparency").and_then(Value::as_bool).unwrap_or(true) || fmt == "jpg") {
        s.execute("layer.flattenImage", json!({})).map_err(|e| e.to_string())?;
    }
    let mut out = s.active().map(|d| (*d.doc).clone()).ok_or("export failed")?;
    if fmt == "psd" && f.get("metadata").and_then(Value::as_str) != Some("all") {
        out.metadata.text.clear();
    }
    Ok(out)
}

/// The exported pixel size: each side scaled and rounded on its own, never below 1 px. The
/// dialog's size label and `export_document` share it so the label names the file's real size.
fn scaled_size(w: f64, h: f64, scale: f64) -> (f64, f64) {
    ((w * scale).round().max(1.0), (h * scale).round().max(1.0))
}

/// Export As already supports lossy WebP; reuse the Quick Export defaults when
/// choosing that format so both flows expose the same quality and lossless options.
fn set_format_defaults(f: &mut Map<String, Value>, fmt: &str, prefs: &photocraft_engine::prefs::Export) {
    f.insert("format".into(), json!(fmt));
    match fmt {
        "jpg" => {
            f.insert("quality".into(), json!(prefs.jpeg_quality));
        }
        "webp" => {
            f.insert("quality".into(), json!(prefs.webp_quality));
            f.insert("lossless".into(), json!(prefs.webp_lossless));
        }
        "psd" => {
            f.insert("scale".into(), json!(100));
        }
        _ => {}
    }
}

fn s_fmt(f: &Map<String, Value>) -> String {
    let v = s(f, "format");
    if v.is_empty() { "png".into() } else { v }
}

fn lossless(f: &Map<String, Value>) -> bool {
    f.get("lossless").and_then(Value::as_bool).unwrap_or(false)
}

/// TGA bits per pixel: `"24"` or `"32"`; `"auto"` (the default) is 32 with an alpha channel or
/// transparency, else 24, as Export As wrote before it offered the choice.
fn tga_bits(f: &Map<String, Value>) -> Option<photocraft_io::TgaBits> {
    s(f, "tgaBits").parse().ok().and_then(photocraft_io::TgaBits::from_bits)
}

fn settings(f: &Map<String, Value>) -> ExportSettings {
    let fmt = s_fmt(f);
    let quality = n(f, "quality", 85.0).clamp(1.0, 100.0) as u8;
    ExportSettings {
        jpeg_quality: (fmt == "jpg").then_some(quality),
        webp_lossless: fmt != "webp" || lossless(f),
        webp_quality: (fmt == "webp" && !lossless(f)).then_some(quality),
        xmp_all: s(f, "metadata") == "all",
        tga_bits: if fmt == "tga" { tga_bits(f) } else { None },
        ..Default::default()
    }
}

/// The preview and size estimate share a single reduced document. Resizing every layer twice is
/// especially expensive for layered PSDs, even when the canvas itself is small.
fn preview_proxy(doc: &Document, f: &Map<String, Value>) -> Result<Document, String> {
    export_document(doc, f, Some(if s_fmt(f) == "ico" { 256 } else { 512 }))
}

fn scaled_estimate(bytes: usize, doc: &Document, proxy: &Document, f: &Map<String, Value>) -> u64 {
    let scale = n(f, "scale", 100.0) / 100.0;
    let full = doc.size.width as f64 * scale * doc.size.height as f64 * scale;
    let small = (proxy.size.width as f64 * proxy.size.height as f64).max(1.0);
    (bytes as f64 * full / small) as u64
}

/// Estimated size (bytes) from the already prepared proxy encode, scaled by pixel count.
fn estimate(app: &PhotocraftApp, doc: &Document, proxy: &Document, f: &Map<String, Value>) -> Option<u64> {
    let export = app.services.export.as_ref()?;
    let (bytes, _) = export(proxy, &format!("estimate.{}", s_fmt(f)), &settings(f)).ok()?;
    Some(scaled_estimate(bytes.len(), doc, proxy, f))
}

struct PreviewPixels {
    estimate: Option<u64>,
    width: usize,
    height: usize,
    rgba: Vec<u8>,
}

fn preview_pixels(proxy: &Document, estimate: Option<u64>) -> PreviewPixels {
    let img = photocraft_compose::thumbnail(proxy, 360);
    PreviewPixels { estimate, width: img.width as usize, height: img.height as usize, rgba: img.pixels }
}

fn preview_sync(app: &PhotocraftApp, doc: &Document, f: &Map<String, Value>) -> Result<PreviewPixels, String> {
    let proxy = preview_proxy(doc, f)?;
    Ok(preview_pixels(&proxy, estimate(app, doc, &proxy, f)))
}

#[derive(Clone)]
struct CachedPreview {
    signature: String,
    estimate: Option<u64>,
    texture: Option<Arc<egui::TextureHandle>>,
}

fn cache_preview(ui: &mut egui::Ui, key: egui::Id, signature: String, result: Result<PreviewPixels, String>) -> CachedPreview {
    let preview = match result {
        Ok(pixels) => {
            let image = egui::ColorImage::from_rgba_unmultiplied([pixels.width, pixels.height], &pixels.rgba);
            let texture = Arc::new(ui.ctx().load_texture("export-preview", image, egui::TextureOptions::LINEAR));
            CachedPreview { signature, estimate: pixels.estimate, texture: Some(texture) }
        }
        Err(error) => {
            log::warn!("Export As preview failed: {error}");
            CachedPreview { signature, estimate: None, texture: None }
        }
    };
    ui.data_mut(|d| d.insert_temp(key, preview.clone()));
    preview
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
struct PendingPreview {
    key: egui::Id,
    signature: String,
    result: Arc<Mutex<Option<Result<PreviewPixels, String>>>>,
}

#[cfg(not(target_arch = "wasm32"))]
fn background_preview_needed(doc: &Document) -> bool {
    doc.walk().len() > 32 || doc.size.area() > 16_000_000
}

/// Native app: keep the UI responsive while a many-layer document is reduced and encoded.
/// A single pending job per document also avoids starting a worker for every slider tick.
#[cfg(not(target_arch = "wasm32"))]
fn poll_or_start_preview(
    ui: &mut egui::Ui,
    key: egui::Id,
    signature: &str,
    doc: Arc<Document>,
    fields: &Map<String, Value>,
) -> Option<Result<PreviewPixels, String>> {
    let pending_key = egui::Id::new(("export-preview-pending", doc.id.0));
    let pending: Option<PendingPreview> = ui.data(|d| d.get_temp(pending_key));
    if let Some(pending) = pending {
        let result = pending.result.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take()?;
        ui.data_mut(|d| d.remove::<PendingPreview>(pending_key));
        if pending.key == key && pending.signature == signature {
            return Some(result);
        }
    }

    let result = Arc::new(Mutex::new(None));
    let shared = result.clone();
    let ctx = ui.ctx().clone();
    let fields = fields.clone();
    let started = std::thread::Builder::new().name("export-preview".into()).spawn(move || {
        let prepared = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let source = source_document_from(doc.clone(), &fields).unwrap_or(doc);
            let proxy = preview_proxy(&source, &fields)?;
            let estimate = estimate_native(&source, &proxy, &fields);
            Ok::<_, String>(preview_pixels(&proxy, estimate))
        }))
        .unwrap_or_else(|_| Err("preview worker panicked".into()));
        *shared.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(prepared);
        ctx.request_repaint();
    });
    match started {
        Ok(_) => {
            ui.data_mut(|d| d.insert_temp(pending_key, PendingPreview { key, signature: signature.to_string(), result }));
            None
        }
        Err(error) => Some(Err(format!("could not start preview worker: {error}"))),
    }
}

/// The native exporter uses this same IO encoder. Keep the entire estimate off the UI thread.
#[cfg(not(target_arch = "wasm32"))]
fn estimate_native(doc: &Document, proxy: &Document, f: &Map<String, Value>) -> Option<u64> {
    let settings = settings(f);
    let mut options = photocraft_io::ExportOptions::default();
    if let Some(quality) = settings.jpeg_quality {
        options.encode.jpeg_quality = quality;
    }
    options.encode.webp_lossless = settings.webp_lossless;
    if let Some(quality) = settings.webp_quality {
        options.encode.webp_quality = quality;
    }
    options.tiff_layers = settings.tiff_layers;
    options.xmp = if settings.xmp_all { photocraft_io::XmpEmbed::All } else { photocraft_io::XmpEmbed::None };
    options.tga_bits = settings.tga_bits;
    options.image_previews = settings.image_previews;
    options.psd_maximize_compatibility = settings.psd_maximize_compatibility;
    let bytes = photocraft_io::export(proxy, &format!("estimate.{}", s_fmt(f)), &options).ok()?.bytes;
    Some(scaled_estimate(bytes.len(), doc, proxy, f))
}

pub fn body(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    let t = Tokens::get(ui.ctx());
    let Some(doc) = app.session.active().map(|s| s.doc.clone()) else { return };
    ui.horizontal_top(|ui| {
        // Left: settings.
        ui.vertical(|ui| {
            ui.set_width(220.0);
            ui.label(egui::RichText::new(tl!("File Settings")).font(crate::theme::semibold(12.0)).color(t.text));
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(tl!("Format")).color(t.text_dim));
                let mut fmt = s_fmt(f);
                let opts: Vec<(String, &str)> = FORMATS.iter().map(|(k, l)| (k.to_string(), *l)).collect();
                if crate::widgets::dropdown(ui, "export-format", &mut fmt, &opts, 130.0) {
                    set_format_defaults(f, &fmt, &app.session.prefs().export);
                }
            });
            let fmt = s_fmt(f);
            if fmt == "psd" {
                ui.label(egui::RichText::new(tl!("Keeps layers and editable text. Install the same fonts in Photoshop. Text may reflow on first edit.")).color(t.text_dim).size(11.5));
            }
            if fmt == "webp" {
                let mut ll = lossless(f);
                crate::widgets::checkbox(ui, &mut ll, tl!("Lossless"));
                f.insert("lossless".into(), json!(ll));
            }
            if fmt == "jpg" || (fmt == "webp" && !lossless(f)) {
                let mut q = n(f, "quality", 85.0) as f32;
                crate::widgets::slider_row(ui, tl!("Quality"), &mut q, 1.0..=100.0, "%", None);
                f.insert("quality".into(), json!(q.round()));
            }
            if fmt == "tga" {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(tl!("Resolution")).color(t.text_dim));
                    let mut bits = s(f, "tgaBits");
                    let opts: Vec<(String, &str)> =
                        vec![("auto".into(), tl!("Auto")), ("24".into(), tl!("24 bits/pixel")), ("32".into(), tl!("32 bits/pixel"))];
                    if crate::widgets::dropdown(ui, "export-tga-bits", &mut bits, &opts, 130.0) {
                        f.insert("tgaBits".into(), json!(bits));
                    }
                });
            }
            if fmt != "jpg" && fmt != "psd" {
                let mut tr = f.get("transparency").and_then(Value::as_bool).unwrap_or(true);
                crate::widgets::checkbox(ui, &mut tr, tl!("Transparency"));
                f.insert("transparency".into(), json!(tr));
            }
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(tl!("Metadata")).color(t.text_dim));
                let mut m = s(f, "metadata");
                let opts: Vec<(String, &str)> = vec![("none".into(), tl!("None")), ("all".into(), tl!("All"))];
                if crate::widgets::dropdown(ui, "export-metadata", &mut m, &opts, 130.0) {
                    f.insert("metadata".into(), json!(m));
                }
            });
            ui.add_space(8.0);
            ui.label(egui::RichText::new(tl!("Image Size")).font(crate::theme::semibold(12.0)).color(t.text));
            let mut sc = n(f, "scale", 100.0) as f32;
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(tl!("Scale")).color(t.text_dim));
                crate::widgets::value_field(ui, &mut sc, 1.0..=1000.0, "%", 70.0);
            });
            f.insert("scale".into(), json!(sc.round()));
            let (bw, bh) = (n(f, "__w", f64::from(doc.size.width)), n(f, "__h", f64::from(doc.size.height)));
            let (w, h) = scaled_size(bw, bh, f64::from(sc) / 100.0);
            ui.label(egui::RichText::new(format!("{w} × {h} px")).color(t.text_dim).size(11.5));
            if fmt == "ico" && (w > 256.0 || h > 256.0) {
                ui.label(egui::RichText::new(tl!("ICO is limited to 256 × 256 pixels")).color(t.warning));
            }
        });
        ui.add_space(12.0);
        // Right: preview + estimated size.
        ui.vertical(|ui| {
            let layer = f.get("__layer").and_then(Value::as_u64);
            // Imported pages may share stored IDs and revisions; cache within this dialog.
            let key = ui.id().with(("export-preview", doc.id.0, app.session.active().map_or(0, |s| s.revision), layer));
            let sig = format!(
                "{}{}{}{}{}{}{}",
                s_fmt(f),
                n(f, "quality", 85.0),
                lossless(f),
                f.get("transparency").map(|v| v.to_string()).unwrap_or_default(),
                n(f, "scale", 100.0),
                s(f, "metadata"),
                s(f, "tgaBits")
            );
            let cached: Option<CachedPreview> = ui.data(|d| d.get_temp(key));
            let preview = match cached.filter(|cached| cached.signature == sig) {
                Some(cached) => Some(cached),
                None => {
                    #[cfg(not(target_arch = "wasm32"))]
                    let result = if background_preview_needed(&doc) {
                        poll_or_start_preview(ui, key, &sig, doc.clone(), f)
                    } else {
                        let source = source_document(app, f).unwrap_or_else(|_| doc.clone());
                        Some(preview_sync(app, &source, f))
                    };
                    #[cfg(target_arch = "wasm32")]
                    let result = {
                        let source = source_document(app, f).unwrap_or_else(|_| doc.clone());
                        Some(preview_sync(app, &source, f))
                    };
                    result.map(|result| cache_preview(ui, key, sig, result))
                }
            };
            let side = 300.0;
            let (r, _) = ui.allocate_exact_size(egui::vec2(side, side), egui::Sense::hover());
            crate::widgets::checker(ui.painter(), r, 8.0);
            if let Some(tex) = preview.as_ref().and_then(|preview| preview.texture.as_ref()) {
                let ts = tex.size_vec2();
                let k = (side / ts.x.max(ts.y)).min(1.0e3);
                let ir = egui::Rect::from_center_size(r.center(), ts * k);
                ui.painter().image(tex.id(), ir, egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
            } else if preview.is_none() {
                ui.put(egui::Rect::from_center_size(r.center(), egui::vec2(24.0, 24.0)), egui::Spinner::new());
            } else {
                ui.painter().text(r.center(), egui::Align2::CENTER_CENTER, tl!("Preview"), egui::FontId::proportional(12.0), t.text_dim);
            }
            ui.painter().rect_stroke(r, 0.0, egui::Stroke::new(1.0, t.separator), egui::StrokeKind::Outside);
            let est = preview.as_ref().and_then(|preview| preview.estimate).map_or("—".to_string(), |b| crate::sizing::human_bytes(b as f64));
            ui.label(egui::RichText::new(format!("{}  ≈ {est}", s_fmt(f).to_uppercase())).color(t.text_dim).size(11.5));
        });
    });
}

/// Export with the dialog's settings: choose a path, render, encode, write.
pub fn confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    remember(app, f);
    if let Some(layers) = f.get("__layers").and_then(Value::as_array) {
        let ids: Vec<_> = layers.iter().filter_map(Value::as_u64).map(photocraft_doc::LayerId).collect();
        return confirm_layers(app, f, &ids);
    }
    let doc = source_document(app, f)?;
    let stem = doc.name.rsplit_once('.').map_or(doc.name.as_str(), |(a, _)| a).to_string();
    let ext = s_fmt(f);
    let suggested = format!("{stem}.{ext}");
    let (f, settings) = (f.clone(), settings(f));
    app.pick_save(&suggested, move |app, path| {
        let out = export_document(&doc, &f, None)?;
        let export = app.services.export.as_ref().ok_or("no exporter configured")?;
        let (bytes, mut warnings) = export(&out, &path, &settings)?;
        let write = app.services.write.as_mut().ok_or("no writer configured")?;
        write(&path, &bytes)?;
        let template = "Exported {path} ({size})";
        app.ui.status = crate::i18n::fmt(tl!(template), &[("path", &path), ("size", &crate::sizing::human_bytes(bytes.len() as f64))]);
        if app.services.host_save.is_some() {
            // A flat download is a copy; the host still owns the layered page.
            warnings.retain(|warning| !warning.ends_with(" layer(s) flattened; layers, masks and blend modes are not kept"));
        }
        app.ui.status_error = false;
        let template = "Exported {path}";
        let what = crate::i18n::fmt(tl!(template), &[("path", &crate::file_open::display_name(&path))]);
        crate::notices::io_warnings(app, &what, &warnings);
        Ok(json!({"path": path, "bytes": bytes.len(), "warnings": warnings}))
    })
}

/// [`confirm`] for several layers: one destination prompt, then each layer exported with the
/// dialog's settings and named as Quick Export names a selection's files. The layers are captured
/// when asked, so changing the selection or document meanwhile can't change what is exported.
fn confirm_layers(app: &mut PhotocraftApp, f: &Map<String, Value>, ids: &[photocraft_doc::LayerId]) -> Result<Value, String> {
    let st = app.session.active().ok_or("no document")?;
    let docs: Vec<Document> =
        ids.iter().map(|id| photocraft_engine::layer_menu_cmds::layer_document(&st.doc, *id).map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
    if docs.is_empty() {
        return Err("no selected layers".into());
    }
    let stem = st.doc.name.rsplit_once('.').map_or(st.doc.name.as_str(), |(a, _)| a);
    let suggested = format!("{}.{}", export_name(stem), s_fmt(f));
    let (f, settings) = (f.clone(), settings(f));
    app.pick_save(&suggested, move |app, path| {
        let mut exported = Vec::new();
        for (doc, path) in docs.iter().zip(batch_paths(&path, &docs)) {
            let out = export_document(doc, &f, None)?;
            let export = app.services.export.as_ref().ok_or("no exporter configured")?;
            let (bytes, warnings) = export(&out, &path, &settings)?;
            let write = app.services.write.as_mut().ok_or("no writer configured")?;
            write(&path, &bytes)?;
            let template = "Exported {path}";
            let what = crate::i18n::fmt(tl!(template), &[("path", &crate::file_open::display_name(&path))]);
            crate::notices::io_warnings(app, &what, &warnings);
            exported.push(json!({"path": path, "bytes": bytes.len(), "warnings": warnings}));
        }
        let template = "Exported {n} files";
        app.ui.status = crate::i18n::fmt(tl!(template), &[("n", &exported.len().to_string())]);
        app.ui.status_error = false;
        Ok(json!({"files": exported}))
    })
}

/// File › Export › Quick Export as PNG: the format, quality, metadata, colour space and location
/// from File › Export › Export Preferences (engine `file.export.quickExport`). On the web (no
/// file system) it falls back to a PNG download through the export service.
pub fn quick_export_png(app: &mut PhotocraftApp) -> Result<Value, String> {
    if !cfg!(target_arch = "wasm32") {
        let prefs = app.session.prefs().export.clone();
        let fmt = serde_json::to_value(prefs.quick_export_format).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_else(|| "png".into());
        let same = serde_json::to_value(prefs.quick_export_location).ok().is_some_and(|v| v == "sameFolder");
        let saved = app.session.active().is_some_and(|d| d.path.is_some());
        if same && saved {
            return quick_export(app, json!({}));
        }
        let st = app.session.active().ok_or("no document")?;
        let stem = st.doc.name.rsplit_once('.').map_or(st.doc.name.as_str(), |(a, _)| a);
        let (suggested, doc) = (format!("{stem}.{fmt}"), st.doc.id);
        return app.pick_save(&suggested, move |app, path| {
            app.refocus(doc)?;
            quick_export(app, json!({"path": path}))
        });
    }
    let mut f = Map::new();
    f.insert("format".into(), json!("png"));
    f.insert("transparency".into(), json!(true));
    f.insert("scale".into(), json!(100));
    confirm(app, &f)
}

/// Runs Quick Export with `p` (an explicit `path`, or none for the document's folder).
fn quick_export(app: &mut PhotocraftApp, p: Value) -> Result<Value, String> {
    let r = app.run("file.export.quickExport", p)?;
    let template = "Exported {path}";
    app.ui.status = crate::i18n::fmt(tl!(template), &[("path", r["path"].as_str().unwrap_or_default())]);
    Ok(r)
}

#[cfg(test)]
mod tests {
    #[test]
    fn export_preview_does_not_reuse_a_closed_page_with_the_same_document_id() {
        use photocraft_doc::{Color, ColorMode, SampleType, Size};

        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        PhotocraftApp::setup_context(&ctx, crate::theme::ThemeKind::Pro);
        let mut page = Document::with_background("page", Size::new(32, 16), ColorMode::Rgb, SampleType::U8, Color::rgba(1.0, 0.0, 0.0, 1.0));
        let id = page.id;
        for (color, expected) in [(Color::rgba(1.0, 0.0, 0.0, 1.0), egui::Color32::RED), (Color::rgba(0.0, 1.0, 0.0, 1.0), egui::Color32::GREEN)] {
            app.ui.dialogs.clear();
            app.session.close(0);
            page = Document::with_background("page", Size::new(32, 16), ColorMode::Rgb, SampleType::U8, color);
            page.id = id;
            app.session.add_document(page.clone(), None);
            assert_eq!(app.session.active().unwrap().revision, 1);
            open(&mut app).unwrap();
            let mut output = ctx.run_ui(Default::default(), |ctx| crate::dialogs::show(&mut app, ctx));
            assert!(output.textures_delta.set.values().flatten().any(|delta| {
                let egui::ImageData::Color(image) = &delta.image;
                image.size == [32, 16] && image.pixels.iter().all(|pixel| *pixel == expected)
            }), "each page must upload its own preview, even when stored IDs and revisions match");
            output.textures_delta.clear();
        }
    }

    #[test]
    fn psd_export_keeps_editable_layers_despite_flat_image_settings() {
        use photocraft_doc::{Color, ColorMode, Layer, LayerContent, SampleType, Size, TextLayer};
        use photocraft_doc::text::{CharStyle, TextRun};

        let mut doc = Document::with_background("page", Size::new(160, 120), ColorMode::Rgb, SampleType::U8, Color::WHITE);
        let mut text = TextLayer {
            text: "AB".into(),
            runs: vec![TextRun { len: 1, style: CharStyle { size_pt: 24.0, ..Default::default() } },
                       TextRun { len: 1, style: CharStyle { size_pt: 18.0, ..Default::default() } }],
            ..Default::default()
        };
        text.sync_summary();
        photocraft_text::TextEngine::new().render_layer(&mut text, doc.resolution_dpi, doc.pixel_format());
        doc.layers.push(Layer::new("translation", LayerContent::Text(text)));
        doc.metadata.text.push(("application-data".into(), "private page data".into()));
        let before = doc.clone();
        let mut fields = Map::from_iter([("format".into(), json!("jpg")), ("scale".into(), json!(25)),
                                        ("transparency".into(), json!(false)), ("metadata".into(), json!("none"))]);
        set_format_defaults(&mut fields, "psd", &Default::default());
        assert!(FORMATS.iter().any(|(ext, _)| *ext == "psd"));
        assert_eq!(fields["scale"], 100, "PSD starts at the original page size");
        let output = export_document(&doc, &fields, None).unwrap();
        assert_eq!(output.layers, doc.layers, "PSD must not flatten hidden or editable layers");
        assert!(output.metadata.text.is_empty(), "Metadata: None excludes internal page data");
        let exported = photocraft_io::export(&output, "page.psd", &Default::default()).unwrap();
        assert!(exported.warnings.is_empty(), "{:?}", exported.warnings);
        let back = photocraft_io::import("page.psd", &exported.bytes).unwrap().document;
        assert_eq!(back.layers.len(), doc.layers.len());
        let LayerContent::Text(text) = &back.layers[1].content else { panic!("editable type layer") };
        assert_eq!(text.text, "AB");
        assert_eq!(text.char_runs().iter().map(|run| run.style.size_pt).collect::<Vec<_>>(), vec![24.0, 18.0]);
        assert_eq!(doc, before, "export must not mutate the current page or its metadata");
        fields.insert("metadata".into(), json!("all"));
        assert_eq!(export_document(&doc, &fields, None).unwrap().metadata.text, doc.metadata.text);
    }

    #[test]
    fn ico_export_is_available_and_rejects_oversize_output() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 300, "height": 200})).unwrap();
        let mut f = Map::new();
        f.insert("format".into(), json!("ico"));
        f.insert("scale".into(), json!(100));
        assert!(FORMATS.iter().any(|(ext, _)| *ext == "ico"));
        assert!(export_document(&app.session.active().unwrap().doc, &f, None).err().is_some_and(|e| e.contains("256")));
        f.insert("scale".into(), json!(50));
        let resized = export_document(&app.session.active().unwrap().doc, &f, None).unwrap();
        assert_eq!((resized.size.width, resized.size.height), (150, 100));
    }

    use super::*;

    /// Export As reopens with the settings it was used with last (Preferences ▸ remembered
    /// dialog settings), and a value no control accepts keeps its default.
    #[test]
    fn export_as_opens_with_the_settings_used_last() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 20, "height": 10})).unwrap();
        app.session.prefs.edit(|p| {
            p.dialogs.insert(
                COMMAND.into(),
                json!({"format": "jpg", "quality": 88, "scale": 37, "transparency": false, "metadata": "all", "tgaBits": "24", "lossless": true}),
            );
        });
        let id = open(&mut app).unwrap();
        let f = &app.ui.dialog_mut(id).unwrap().fields;
        assert_eq!(f["format"], "jpg");
        assert_eq!(f["quality"], json!(88.0));
        assert_eq!(f["scale"], json!(37.0));
        assert_eq!(f["transparency"], false);
        assert_eq!(f["metadata"], "all");
        assert_eq!(f["tgaBits"], "24");

        app.session.prefs.edit(|p| {
            p.dialogs.insert(COMMAND.into(), json!({"format": "nope", "scale": 0, "metadata": "sometimes"}));
        });
        let id = open(&mut app).unwrap();
        let f = &app.ui.dialog_mut(id).unwrap().fields;
        assert_eq!(f["format"], "png");
        assert_eq!(f["scale"], 100);
        assert_eq!(f["metadata"], "none");
    }

    /// Only the dialog's own settings are remembered: not the destination, the document size or
    /// the layer the dialog was opened with.
    #[test]
    fn remembering_keeps_only_the_dialog_settings() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let mut f = Map::new();
        f.insert("format".into(), json!("webp"));
        f.insert("scale".into(), json!(50));
        f.insert("path".into(), json!("/tmp/x.webp"));
        f.insert("__w".into(), json!(1200));
        f.insert("__layer".into(), json!(3));
        remember(&mut app, &f);
        let saved = app.session.prefs().dialogs.get(COMMAND).cloned().unwrap();
        assert_eq!(saved["format"], "webp");
        assert_eq!(saved["scale"], 50);
        assert!(saved.get("path").is_none() && saved.get("__w").is_none() && saved.get("__layer").is_none());
    }

    #[test]
    fn export_document_scales_height_when_rounded_width_is_unchanged() {
        use photocraft_doc::{Color, ColorMode, SampleType, Size};

        for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
            for (width, height, scale, expected) in [
                (1, 100, 50, (1, 50)),
                (1, 200, 50, (1, 100)),
                (10, 100, 99, (10, 99)),
                (10, 100, 101, (10, 101)),
                (400, 300, 50, (200, 150)),
                (400, 300, 100, (400, 300)),
            ] {
                let doc = Document::with_background("x", Size::new(width, height), ColorMode::Rgb, depth, Color::WHITE);
                let before = doc.clone();
                let mut f = Map::new();
                f.insert("format".into(), json!("png"));
                f.insert("scale".into(), json!(scale));
                let out = export_document(&doc, &f, None).unwrap();
                assert_eq!((out.size.width, out.size.height), expected, "{width} × {height} at {scale}%, {depth:?}");
                assert_eq!(out.depth, depth, "export preserves the document's depth");
                assert_eq!(doc, before, "export leaves the source document unchanged");
            }
        }
    }

    #[test]
    fn export_document_proxy_limits_both_dimensions_before_rounding() {
        use photocraft_doc::{Color, ColorMode, SampleType, Size};

        for (width, height, scale, max_side, expected) in
            [(1, 1000, 100, 360, (1, 360)), (1, 1000, 50, 360, (1, 360)), (1, 1000, 25, 360, (1, 250)), (13, 20, 50, 8, (5, 8)), (20, 13, 50, 8, (8, 5))]
        {
            let doc = Document::with_background("x", Size::new(width, height), ColorMode::Rgb, SampleType::U8, Color::WHITE);
            let before = doc.clone();
            let mut f = Map::new();
            f.insert("format".into(), json!("png"));
            f.insert("scale".into(), json!(scale));
            let out = export_document(&doc, &f, Some(max_side)).unwrap();
            assert_eq!((out.size.width, out.size.height), expected, "{width} × {height} at {scale}%, proxy limit {max_side}");
            assert!(out.size.width.max(out.size.height) <= max_side, "proxy respects its longest-side limit");
            assert_eq!(doc, before, "preview leaves the source document unchanged");
        }
    }

    #[test]
    fn export_document_scales_and_flattens() {
        let doc = Document::with_background(
            "x",
            photocraft_doc::Size::new(200, 100),
            photocraft_doc::ColorMode::Rgb,
            photocraft_doc::SampleType::U8,
            photocraft_doc::Color::WHITE,
        );
        let mut f = Map::new();
        f.insert("format".into(), json!("jpg"));
        f.insert("scale".into(), json!(50));
        let out = export_document(&doc, &f, None).unwrap();
        assert_eq!((out.size.width, out.size.height), (100, 50));
        assert_eq!(out.layers.len(), 1);
        let proxy = export_document(&doc, &f, Some(40)).unwrap();
        assert_eq!(proxy.size.width, 40);
        assert_eq!(settings(&f).jpeg_quality, Some(85));
    }

    /// #1353: a scale that changes only the height (the width rounds back to the original)
    /// used to be skipped, so the exported file kept the original size while the dialog
    /// labelled it as scaled.
    #[test]
    fn export_document_scales_when_only_the_height_changes() {
        let mut f = Map::new();
        f.insert("format".into(), json!("png"));
        f.insert("scale".into(), json!(50));
        for (src_h, want_h) in [(100u32, 50u32), (200, 100)] {
            let doc = Document::with_background(
                "x",
                photocraft_doc::Size::new(1, src_h),
                photocraft_doc::ColorMode::Rgb,
                photocraft_doc::SampleType::U8,
                photocraft_doc::Color::WHITE,
            );
            let out = export_document(&doc, &f, None).unwrap();
            assert_eq!((out.size.width, out.size.height), (1, want_h), "1 × {src_h} at 50%");
            // The preview proxy takes the same path.
            let proxy = export_document(&doc, &f, Some(40)).unwrap();
            assert_eq!((proxy.size.width, proxy.size.height), (1, 40), "proxy of 1 × {src_h}");
        }
        // The label and the export agree, and neither side drops below 1 px.
        assert_eq!(scaled_size(1.0, 100.0, 0.5), (1.0, 50.0));
        assert_eq!(scaled_size(10.0, 300.0, 0.01), (1.0, 3.0));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn layered_preview_runs_outside_the_ui_frame() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 32, "height": 16})).unwrap();
        for _ in 0..33 {
            app.run("layer.new.layer", json!({})).unwrap();
        }
        let doc = app.session.active().unwrap().doc.clone();
        assert!(background_preview_needed(&doc));
        let fields = Map::from_iter([("format".into(), json!("png")), ("scale".into(), json!(100))]);
        let ctx = egui::Context::default();
        let key = egui::Id::new("layered-preview-test");
        let mut result = None;
        let mut output = ctx.run_ui(Default::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                result = poll_or_start_preview(ui, key, "png100", doc.clone(), &fields);
            });
        });
        output.textures_delta.clear();
        assert!(result.is_none(), "the first UI frame must only start the worker");

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let mut output = ctx.run_ui(Default::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    result = poll_or_start_preview(ui, key, "png100", doc.clone(), &fields);
                });
            });
            output.textures_delta.clear();
            if let Some(preview) = result {
                let preview = preview.unwrap();
                assert_eq!((preview.width, preview.height), (32, 16));
                assert_eq!(preview.rgba.len(), 32 * 16 * 4);
                assert!(preview.estimate.is_some_and(|bytes| bytes > 0));
                break;
            }
            assert!(std::time::Instant::now() < deadline, "background preview did not complete");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let tiff_fields = Map::from_iter([("format".into(), json!("tif")), ("scale".into(), json!(100))]);
        let proxy = preview_proxy(&doc, &tiff_fields).unwrap();
        let options = photocraft_io::ExportOptions { tiff_layers: true, xmp: photocraft_io::XmpEmbed::None, ..Default::default() };
        let actual_bytes = photocraft_io::export(&proxy, "estimate.tif", &options).unwrap().bytes.len();
        assert_eq!(estimate_native(&doc, &proxy, &tiff_fields), Some(scaled_estimate(actual_bytes, &doc, &proxy, &tiff_fields)));
    }

    #[test]
    fn switching_export_formats_uses_independent_quality_preferences() {
        let prefs = photocraft_engine::prefs::Export { jpeg_quality: 43, webp_quality: 72, webp_lossless: false, ..Default::default() };
        let mut fields = Map::new();
        set_format_defaults(&mut fields, "webp", &prefs);
        assert_eq!(fields["quality"], 72);
        assert_eq!(fields["lossless"], false);
        let s = settings(&fields);
        assert!(!s.webp_lossless);
        assert_eq!(s.webp_quality, Some(72));
        assert_eq!(s.jpeg_quality, None);

        set_format_defaults(&mut fields, "jpg", &prefs);
        assert_eq!(fields["quality"], 43);
        assert_eq!(settings(&fields).jpeg_quality, Some(43));
        assert_eq!(settings(&fields).webp_quality, None);

        set_format_defaults(&mut fields, "webp", &photocraft_engine::prefs::Export::default());
        assert!(settings(&fields).webp_lossless, "default WebP Quick Export mode is lossless");
        assert_eq!(settings(&fields).webp_quality, None);
    }

    #[test]
    fn webp_settings_follow_the_lossless_switch() {
        let mut f = Map::new();
        f.insert("format".into(), json!("webp"));
        f.insert("quality".into(), json!(70));
        let s = settings(&f);
        assert!(!s.webp_lossless, "Export As writes lossy WebP unless asked");
        assert_eq!(s.webp_quality, Some(70));
        assert_eq!(s.jpeg_quality, None);
        f.insert("lossless".into(), json!(true));
        let s = settings(&f);
        assert!(s.webp_lossless);
        assert_eq!(s.webp_quality, None);
        f.insert("format".into(), json!("png"));
        assert!(settings(&f).webp_lossless, "other formats leave the WebP default alone");
    }

    /// TGA's Resolution (bits per pixel): Auto leaves the choice to the export; 24 and 32 force it, and only
    /// for TGA.
    #[test]
    fn tga_settings_follow_the_bits_per_pixel() {
        use photocraft_io::TgaBits;
        let mut f = Map::new();
        f.insert("format".into(), json!("tga"));
        assert_eq!(settings(&f).tga_bits, None, "no choice made");
        for (v, want) in [("auto", None), ("24", Some(TgaBits::Bits24)), ("32", Some(TgaBits::Bits32)), ("16", None)] {
            f.insert("tgaBits".into(), json!(v));
            assert_eq!(settings(&f).tga_bits, want, "{v}");
        }
        f.insert("format".into(), json!("png"));
        assert_eq!(settings(&f).tga_bits, None, "only TGA has bits per pixel");
    }
}
