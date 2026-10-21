//! Front ends of the File-menu commands added with slices: the Save for Web (Legacy) dialog
//! (Original / Optimized / 2-Up / 4-Up previews, GIF / PNG-8 / PNG-24 / JPEG / WBMP settings,
//! image size, per-slice output), the Print dialog (page preview with position, scale, marks;
//! colour handling), and form dialogs for Contact Sheet II, Create Droplet, Statistics, Script
//! Events Manager, Package and Paths to Illustrator. Every OK runs the engine command with the
//! dialog's fields, so the control channel can drive them with `ui.dialog.set` / `confirm`.

use std::sync::Arc;

use photocraft_engine::web_cmds::{self, Optimized, WebSettings};
use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::state::DialogKind;
use crate::theme::Tokens;

fn no_params(p: &Value) -> bool {
    p.as_object().is_none_or(|o| o.is_empty())
}

fn default_dir(app: &PhotocraftApp) -> String {
    app.session.active().and_then(|d| d.path.as_deref()).and_then(|p| p.rfind(['/', '\\']).map(|i| p[..i].to_string())).unwrap_or_else(|| ".".into())
}

/// A packaged macOS app does not start in a useful writable working directory. A saved
/// document keeps its own folder; otherwise prefer the user's Documents folder.
fn droplet_default_dir(app: &PhotocraftApp) -> String {
    let dir = default_dir(app);
    if dir != "." {
        return dir;
    }
    #[cfg(target_os = "macos")]
    if let Some(home) = std::env::var_os("HOME") {
        let home = std::path::PathBuf::from(home);
        let documents = home.join("Documents");
        return if documents.is_dir() { documents } else { home }.to_string_lossy().into_owned();
    }
    dir
}

fn doc_stem(app: &PhotocraftApp) -> String {
    app.session.active().map(|d| d.doc.name.rsplit_once('.').map_or(d.doc.name.clone(), |(a, _)| a.to_string())).unwrap_or_else(|| tl!("Untitled").into())
}

/// A generic form dialog for `command` (rendered by `view_cmds::form_body`).
fn form(app: &mut PhotocraftApp, command: &str, label: &str, fields: Value, choices: Value) -> Value {
    let mut f = Map::new();
    f.insert("__command".into(), json!(command));
    f.insert("__label".into(), json!(label));
    f.insert("__form".into(), json!(true));
    f.insert("__choices".into(), choices);
    if let Value::Object(m) = fields {
        f.extend(m);
    }
    json!({"dialog": app.ui.open_dialog(DialogKind::Command, f)})
}

/// File › Scripts › Browse… lists these script files (plus the JSON/text the engine parses).
const SCRIPT_EXTENSIONS: &[&str] = &["psjs", "ccjs", "jsx", "js", "jsxbin", "json", "txt"];

/// Menu items fronted here (dialogs before the engine command). `None` when not ours.
pub fn invoke(app: &mut PhotocraftApp, ctx: &egui::Context, id: &str, params: &Value) -> Option<Result<Value, String>> {
    if !no_params(params) {
        return None;
    }
    let dir = default_dir(app);
    let stem = doc_stem(app);
    Some(Ok(match id {
        "file.export.saveForWebLegacy" => {
            app.session.active()?;
            json!({"dialog": open_web(app)})
        }
        "file.print" => {
            app.session.active()?;
            json!({"dialog": open_print(app)})
        }
        "file.printOneCopy" if app.session.file_menu.last_print.is_none() => {
            app.session.active()?;
            json!({"dialog": open_print(app)})
        }
        "file.export.exportPreferences" => return crate::prefs_ui::invoke(app, ctx, "edit.preferences.export", &json!({})),
        "file.automate.contactSheetII" => form(
            app,
            id,
            tl!("Contact Sheet II"),
            json!({"input": dir, "units": "inches", "width": 8.0, "height": 10.0, "resolution": 300.0, "mode": "rgb", "depth": 8, "columns": 5, "rows": 6, "placeAcrossFirst": true, "autoSpacing": true, "rotateForBestFit": false, "caption": true, "font": photocraft_text::fonts::DEFAULT_FAMILY, "fontSize": 12.0, "flatten": false}),
            json!({"units": ["inches", "cm", "pixels"], "mode": ["rgb", "gray", "cmyk", "lab"]}),
        ),
        "file.automate.createDroplet" => {
            let dir = droplet_default_dir(app);
            let Some(action) = crate::actions::selected_action(app) else {
                return Some(Err("record an action in the Actions panel first".into()));
            };
            let steps = crate::actions::action_steps(action);
            let name = action.name.clone();
            form(
                app,
                id,
                tl!("Create Droplet"),
                json!({"path": format!("{dir}/{name}.pcdroplet"), "name": name, "steps": steps, "output": format!("{dir}/droplet-output"), "format": "same"}),
                json!({"format": ["same", "png", "jpg", "psd", "tiff"]}),
            )
        }
        "file.scripts.statistics" => json!({"dialog": crate::stack_dialog::open_statistics(app)}),
        "file.scripts.browse" => {
            return Some(app.pick_file_bytes_filtered(SCRIPT_EXTENSIONS, |app, name, bytes| {
                let r = app.run("file.scripts.browse", json!({"script": String::from_utf8_lossy(&bytes)}));
                if r.is_ok() {
                    let template = "Ran script {name}";
                    app.ui.status = crate::i18n::fmt(tl!(template), &[("name", &name)]);
                }
                r
            }));
        }
        "file.scripts.scriptEventsManager" => {
            let enabled = app.session.prefs().script_events.enabled;
            let events: Vec<&str> = photocraft_engine::automate_cmds::EVENTS.iter().map(|e| e.0).collect();
            form(app, id, "Script Events Manager", json!({"enabled": enabled, "event": "openDocument", "script": "", "name": ""}), json!({"event": events}))
        }
        "file.package" => form(app, id, "Package", json!({"dir": dir}), json!({})),
        "file.export.pathsToIllustrator" => {
            let mut names = vec!["all".to_string()];
            if let Some(d) = app.session.active() {
                if d.doc.work_path.is_some() {
                    names.push("work".into());
                }
                names.extend(d.doc.paths.iter().map(|p| p.name.clone()));
            }
            form(app, id, "Export Paths to File", json!({"path": format!("{dir}/{stem}.ai"), "paths": "all"}), json!({"paths": names}))
        }
        _ => return None,
    }))
}

/// Checked state of toggle items owned here.
pub fn checked(app: &PhotocraftApp, id: &str) -> Option<bool> {
    match id {
        "view.lockSlices" => Some(app.session.file_menu.slices_locked),
        "file.generate.imageAssets" => Some(app.session.active().is_some_and(|d| app.session.file_menu.image_assets.contains(&d.doc.id))),
        _ => None,
    }
}

/// Dialogs whose body and confirm live here.
pub fn owns(f: &Map<String, Value>) -> bool {
    f.contains_key("__web") || f.contains_key("__print")
}

pub fn dialog_width(f: &Map<String, Value>) -> Option<f32> {
    if f.contains_key("__web") {
        Some(900.0)
    } else if f.contains_key("__print") {
        Some(720.0)
    } else {
        None
    }
}

pub fn ok_label(f: &Map<String, Value>) -> Option<&'static str> {
    if f.contains_key("__web") {
        Some(tl!("Save…"))
    } else if f.contains_key("__print") {
        Some(tl!("Print"))
    } else {
        None
    }
}

pub fn body(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    if f.contains_key("__web") {
        web_body(app, ui, f);
    } else {
        print_body(app, ui, f);
    }
}

pub fn confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    if f.contains_key("__web") { web_confirm(app, f) } else { print_confirm(app, f) }
}

fn params(f: &Map<String, Value>) -> Value {
    Value::Object(f.iter().filter(|(k, _)| !k.starts_with("__")).map(|(k, v)| (k.clone(), v.clone())).collect())
}

fn s(f: &Map<String, Value>, k: &str, d: &str) -> String {
    f.get(k).and_then(Value::as_str).unwrap_or(d).to_string()
}
fn n(f: &Map<String, Value>, k: &str, d: f64) -> f64 {
    f.get(k).and_then(Value::as_f64).unwrap_or(d)
}
fn b(f: &Map<String, Value>, k: &str, d: bool) -> bool {
    f.get(k).and_then(Value::as_bool).unwrap_or(d)
}

// ---------- Save for Web ----------

#[path = "file_ui/web_preview.rs"]
mod web_preview;

const WEB_FORMATS: [(&str, &str); 5] = [("gif", "GIF"), ("png8", "PNG-8"), ("png24", "PNG-24"), ("jpeg", "JPEG"), ("wbmp", "WBMP")];
const WEB_COMMAND: &str = "file.export.saveForWebLegacy";

pub fn open_web(app: &mut PhotocraftApp) -> u64 {
    let mut f = Map::new();
    f.insert("__web".into(), json!(true));
    f.insert("__command".into(), json!(WEB_COMMAND));
    f.insert("__label".into(), json!("Save for Web (Legacy)"));
    f.insert("__view".into(), json!("2up"));
    // Remember encoding choices, but each document starts at its own full size. This session's
    // last export lives in `file_menu`; the last one before a restart comes from the
    // preferences' remembered dialog settings (`web_cmds::remember_web_settings`).
    let last = app
        .session
        .file_menu
        .last_web
        .clone()
        .or_else(|| app.session.prefs().dialogs.get(WEB_COMMAND).cloned())
        .unwrap_or_else(|| json!({"format": "jpeg", "quality": 60}));
    if let Value::Object(m) = last {
        for (k, v) in m {
            if !matches!(k.as_str(), "path" | "dir" | "width" | "height" | "percent" | "colorTable" | "colorTableEdits") {
                f.insert(k, v);
            }
        }
    }
    for (k, v) in [
        ("format", json!("jpeg")),
        ("quality", json!(60)),
        ("palette", json!("selective")),
        ("colors", json!(128)),
        ("dither", json!("diffusion")),
        ("ditherAmount", json!(88)),
        ("transparency", json!(true)),
        ("matte", json!("#ffffff")),
        ("interlaced", json!(false)),
        ("progressive", json!(false)),
        ("optimized", json!(true)),
        ("embedIcc", json!(false)),
        ("convertToSrgb", json!(true)),
        ("metadata", json!("copyright")),
        ("percent", json!(100.0)),
        ("webSnap", json!(0)),
    ] {
        f.entry(k.to_string()).or_insert(v);
    }
    app.ui.open_dialog(DialogKind::Command, f)
}

/// Settings variants shown in 4-Up (current, then two smaller alternatives).
fn variants(p: &Value) -> Vec<Value> {
    let mut out = vec![p.clone()];
    let fmt = p["format"].as_str().unwrap_or("jpeg");
    for k in [2.0, 4.0] {
        let mut v = p.clone();
        if let Some(m) = v.as_object_mut() {
            m.remove("colorTable");
            m.remove("colorTableEdits");
        }
        if fmt == "jpeg" {
            v["quality"] = json!((p["quality"].as_f64().unwrap_or(60.0) / k).round().max(1.0));
        } else if fmt == "gif" || fmt == "png8" {
            v["colors"] = json!((p["colors"].as_f64().unwrap_or(128.0) / k).round().max(2.0));
        } else {
            v["format"] = json!(if k == 2.0 { "png8" } else { "gif" });
            v["colors"] = json!(256 / k as u32);
        }
        out.push(v);
    }
    out
}

/// The flattened preview image: pixels, width, height, full-size / proxy pixel ratio.
type WebProxy = (Arc<Vec<[f32; 4]>>, u32, u32, f64, egui::Vec2);
type WebCache = (String, Arc<egui::TextureHandle>, Option<(usize, Option<usize>, &'static str)>);

/// Renders one preview pane: the original (`None`) or an optimised variant.
fn web_pane(app: &PhotocraftApp, ui: &mut egui::Ui, size: egui::Vec2, p: Option<&Value>, f: &mut Map<String, Value>, pane: usize, frame: Option<usize>) {
    let t = Tokens::get(ui.ctx());
    let Some(st) = app.session.active() else { return };
    let doc = st.doc.clone();
    let rev = st.revision;
    // A bounded proxy; navigation never recomposites or re-encodes it.
    let base = p.cloned().unwrap_or_else(|| json!({}));
    let size_sig = format!("{}|{}|{}|{}", base["percent"], base["width"], base["height"], base["convertToSrgb"]);
    let wkey = egui::Id::new(("web-proxy", doc.id.0, pane));
    let proxy_sig = format!("{rev}|{frame:?}|{size_sig}");
    let cached_proxy: Option<(String, WebProxy)> = ui.data(|d| d.get_temp(wkey));
    let proxy = cached_proxy.filter(|(sig, _)| sig == &proxy_sig).map(|(_, proxy)| proxy);
    let (px, w, h, ratio, source_size) = match proxy {
        Some(v) => v,
        None => {
            let (fw, fh) = match web_preview::dimensions(&doc, &base) {
                Ok(size) => size,
                Err(error) => {
                    ui.label(error);
                    return;
                }
            };
            let k = (2048.0 / fw.max(fh)).min(1.0);
            let st = WebSettings::from_params(&base, "preview").unwrap_or_default();
            let Ok(doc) = web_frame_document(&doc, frame) else { return };
            let Ok((wd, _, _)) =
                web_cmds::web_document(&doc, &json!({"width": (fw * k).round().max(1.0), "height": (fh * k).round().max(1.0), "resample": "bilinear"}), &st)
            else {
                return;
            };
            let buf = photocraft_compose::flatten(&wd);
            let v = (
                Arc::new(buf.px),
                wd.size.width,
                wd.size.height,
                (fw * fh) / (f64::from(wd.size.width) * f64::from(wd.size.height)).max(1.0),
                egui::vec2(fw as f32, fh as f32),
            );
            ui.data_mut(|d| d.insert_temp(wkey, (proxy_sig, v.clone())));
            v
        }
    };
    let sig = format!("{}|{rev}|{frame:?}|{size_sig}|{}", doc.id.0, p.map(|v| v.to_string()).unwrap_or_else(|| "original".into()));
    let key = egui::Id::new(("web-pane", doc.id.0, pane));
    let cached: Option<WebCache> = ui.data(|d| d.get_temp(key));
    let (tex, info) = match cached.filter(|c| c.0 == sig) {
        Some((_, tex, info)) => (tex, info),
        None => {
            let (rgba, info) = match p {
                None => (px.iter().flat_map(|q| q.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)).collect::<Vec<u8>>(), None),
                Some(p) => match web_preview::optimized(ui, key, &format!("{}|{rev}|{frame:?}|{size_sig}", doc.id.0), p, &px, w, h) {
                    Ok((preview, bytes, colors, ext)) => (preview, Some(((bytes as f64 * ratio) as usize, colors, ext))),
                    Err(error) => {
                        ui.label(error);
                        (vec![0; (w * h * 4) as usize], None)
                    }
                },
            };
            let img = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);
            let tex = Arc::new(ui.ctx().load_texture("web-pane", img, egui::TextureOptions::LINEAR));
            ui.data_mut(|d| d.insert_temp(key, (sig, tex.clone(), info)));
            (tex, info)
        }
    };
    ui.vertical(|ui| {
        ui.set_width(size.x);
        let (r, response) = ui.allocate_exact_size((size - egui::vec2(0.0, 42.0)).max(egui::vec2(1.0, 1.0)), egui::Sense::drag());
        let response = response.on_hover_cursor(egui::CursorIcon::Grab);
        if let Some(rects) = f.get_mut("__previewRects").and_then(Value::as_array_mut) {
            rects.push(json!([r.left(), r.top(), r.right(), r.bottom()]));
        }
        crate::widgets::checker(ui.painter(), r, 8.0);
        web_preview::paint(ui, r, &response, &tex, source_size, f);
        ui.painter().rect_stroke(r, 0.0, egui::Stroke::new(1.0, t.separator), egui::StrokeKind::Inside);
        let (l1, l2) = match (p, info) {
            (None, _) => ("Original".to_string(), format!("{}  ·  {} × {} px", doc.name, doc.size.width, doc.size.height)),
            (Some(_), Some((bytes, colors, ext))) => {
                // Photoshop's download estimate at 56.6 Kbps.
                let secs = (bytes as f64 * 8.0 / 56_600.0).ceil();
                (
                    format!("{}  {}", ext.to_uppercase(), crate::sizing::human_bytes(bytes as f64)),
                    format!("{secs} sec @ 56.6 Kbps{}", colors.map(|c| format!("  ·  {c} colors")).unwrap_or_default()),
                )
            }
            (Some(_), None) => ("—".into(), String::new()),
        };
        let l1 = if ratio > 1.001 { format!("{l1}  ·  Proxy preview") } else { l1 };
        ui.add(egui::Label::new(egui::RichText::new(l1).color(t.text).size(11.5)).truncate());
        ui.add(egui::Label::new(egui::RichText::new(l2).color(t.text_dim).size(11.0)).truncate());
    });
}

/// Print's Printer menu: the default printer (value ""), then every printer `file.printers`
/// listed. A saved name the spooler no longer lists stays selectable rather than vanishing.
/// CUPS queue names stand in for spaces with `_`; the labels show the spaces.
fn printer_options(list: Option<&Value>, current: &str) -> Vec<(String, String)> {
    let printers: Vec<(&str, bool)> = list
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|p| Some((p.get("name")?.as_str()?, p.get("default").and_then(Value::as_bool).unwrap_or(false)))).collect())
        .unwrap_or_default();
    let pretty = |n: &str| n.replace('_', " ");
    let default = match printers.iter().find(|p| p.1) {
        Some((n, _)) => format!("{} ({})", tl!("Default printer"), pretty(n)),
        None => tl!("Default printer").to_string(),
    };
    let mut opts = vec![(String::new(), default)];
    opts.extend(printers.iter().map(|(n, _)| (n.to_string(), pretty(n))));
    if !current.is_empty() && !printers.iter().any(|p| p.0 == current) {
        opts.push((current.to_string(), pretty(current)));
    }
    opts
}

fn dropdown_str(ui: &mut egui::Ui, id: &str, f: &mut Map<String, Value>, key: &str, opts: &[(&str, &str)], w: f32) {
    let mut v = s(f, key, opts.first().map_or("", |o| o.0));
    let o: Vec<(String, &str)> = opts.iter().map(|(k, l)| (k.to_string(), *l)).collect();
    if crate::widgets::dropdown(ui, id, &mut v, &o, w) {
        f.insert(key.into(), json!(v));
    }
}

fn check(ui: &mut egui::Ui, f: &mut Map<String, Value>, key: &str, label: &str, d: bool) {
    let mut v = b(f, key, d);
    crate::widgets::checkbox(ui, &mut v, label);
    f.insert(key.into(), json!(v));
}

fn number(ui: &mut egui::Ui, f: &mut Map<String, Value>, key: &str, label: &str, range: std::ops::RangeInclusive<f32>, unit: &str, d: f64) {
    let t = Tokens::get(ui.ctx());
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(tl!(&label)).color(t.text_dim));
        let mut v = n(f, key, d) as f32;
        crate::widgets::value_field(ui, &mut v, range, unit, 64.0);
        f.insert(key.into(), json!(v.round()));
    });
}

/// Advance a preview clock by whole elapsed frames, retaining fractional time.
/// The caller resets `last` on Play so paused wall time never advances the preview.
fn advance_preview(frame: usize, duration: usize, fps: f32, last: f64, now: f64) -> (usize, f64) {
    let duration = duration.max(1);
    let frame = frame.min(duration - 1);
    if !now.is_finite() {
        return (frame, last);
    }
    if !last.is_finite() || now < last {
        return (frame, now);
    }
    let fps = if fps.is_finite() && fps > 0.0 { f64::from(fps) } else { 30.0 };
    let elapsed_frames = (now - last) * fps;
    if !elapsed_frames.is_finite() {
        return (frame, now);
    }
    let steps = elapsed_frames.floor();
    if steps < 1.0 {
        return (frame, last);
    }
    let delta = (steps % duration as f64) as usize % duration;
    let remaining = duration - frame;
    let next = if delta >= remaining { delta - remaining } else { frame + delta };
    (next, now - (elapsed_frames - steps) / fps)
}

fn web_playback(ui: &mut egui::Ui, f: &mut Map<String, Value>, doc: &photocraft_doc::Document) -> Option<usize> {
    let mut timeline = doc.timeline.clone()?;
    timeline.clamp();
    if timeline.duration < 2 {
        return None;
    }
    let now = ui.input(|i| i.time);
    let mut frame = f.get("__web_frame").and_then(Value::as_u64).and_then(|v| usize::try_from(v).ok()).unwrap_or(timeline.current).min(timeline.duration - 1);
    let mut last = n(f, "__web_time", now);
    let mut playing = b(f, "__web_playing", false);
    if playing {
        (frame, last) = advance_preview(frame, timeline.duration, timeline.fps, last, now);
    }
    ui.horizontal(|ui| {
        if crate::icons::button(ui, if playing { "pause" } else { "play" }, 22.0, playing, if playing { tl!("Pause") } else { tl!("Play") }).clicked() {
            playing = !playing;
            last = now;
        }
        ui.label(format!("{} / {}", frame + 1, timeline.duration));
    });
    f.insert("__web_frame".into(), json!(frame));
    f.insert("__web_time".into(), json!(last));
    f.insert("__web_playing".into(), json!(playing));
    if playing {
        let remaining = (1.0 / f64::from(timeline.fps) - (now - last)).clamp(0.0, 1.0);
        ui.ctx().request_repaint_after(std::time::Duration::from_secs_f64(remaining));
    }
    Some(frame)
}

/// Seek a preview copy, preserving any edits to the source's currently displayed frame.
fn web_frame_document(doc: &photocraft_doc::Document, frame: Option<usize>) -> Result<photocraft_doc::Document, photocraft_engine::EngineError> {
    let Some(frame) = frame else { return Ok(doc.clone()) };
    let mut preview = photocraft_engine::Session::new();
    preview.add_document(doc.clone(), None);
    preview.execute("timeline.setFrame", json!({"frame": frame}))?;
    Ok((*preview.active().ok_or(photocraft_engine::EngineError::NoDocument)?.doc).clone())
}
fn web_body(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    let t = Tokens::get(ui.ctx());
    let Some(doc) = app.session.active().map(|d| d.doc.clone()) else { return };
    ui.horizontal(|ui| {
        let mut view = s(f, "__view", "2up");
        for (k, l) in [("original", tl!("Original")), ("optimized", tl!("Optimized")), ("2up", "2-Up"), ("4up", "4-Up")] {
            if ui.selectable_label(view == k, tl!(&l)).clicked() {
                view = k.into();
            }
        }
        f.insert("__view".into(), json!(view));
    });
    ui.add_space(4.0);
    let frame = web_playback(ui, f, &doc);
    web_preview::controls(ui, f);
    if let Err(error) = web_preview::prepare_table(app, f) {
        ui.colored_label(t.text_dim, error);
    }
    let p = params(f);
    f.insert("__previewRects".into(), json!([]));
    let area = egui::vec2((ui.available_width() - 270.0 - ui.spacing().item_spacing.x).max(260.0), (ui.available_height() - 48.0).max(180.0));
    ui.horizontal_top(|ui| {
        // Previews grow with the dialog; settings retain their compact width.
        ui.vertical(|ui| {
            ui.set_width(area.x);
            match s(f, "__view", "2up").as_str() {
                "original" => web_pane(app, ui, area, None, f, 0, frame),
                "optimized" => web_pane(app, ui, area, Some(&p), f, 1, frame),
                "2up" => {
                    ui.horizontal(|ui| {
                        web_pane(app, ui, egui::vec2(area.x / 2.0 - 4.0, area.y), None, f, 0, frame);
                        web_pane(app, ui, egui::vec2(area.x / 2.0 - 4.0, area.y), Some(&p), f, 1, frame);
                    });
                }
                _ => {
                    let v = variants(&p);
                    let cell = egui::vec2(area.x / 2.0 - 4.0, area.y / 2.0 - 4.0);
                    ui.horizontal(|ui| {
                        web_pane(app, ui, cell, None, f, 0, frame);
                        web_pane(app, ui, cell, Some(&v[0]), f, 1, frame);
                    });
                    ui.horizontal(|ui| {
                        web_pane(app, ui, cell, Some(&v[1]), f, 2, frame);
                        web_pane(app, ui, cell, Some(&v[2]), f, 3, frame);
                    });
                }
            }
        });
        ui.add_space(10.0);
        // Settings scroll independently on smaller windows.
        ui.vertical(|ui| {
            ui.set_width(250.0);
            egui::ScrollArea::vertical().id_salt("web-settings").max_height(area.y).show(ui, |ui| {
                ui.label(egui::RichText::new(tl!("Preset")).color(t.text_dim));
                let mut preset = s(f, "__preset", "");
                let mut opts: Vec<(String, &str)> = vec![("".into(), "[Unnamed]")];
                opts.extend(web_cmds::PRESETS.iter().map(|p| (p.to_string(), *p)));
                if crate::widgets::dropdown(ui, "web-preset", &mut preset, &opts, 230.0) && !preset.is_empty() {
                    if let Some(Value::Object(m)) = web_cmds::preset_settings(&preset).map(|st| st.to_params()) {
                        f.extend(m);
                    }
                    f.insert("__preset".into(), json!(preset));
                }
                ui.add_space(4.0);
                dropdown_str(ui, "web-format", f, "format", &WEB_FORMATS, 230.0);
                ui.add_space(4.0);
                let fmt = s(f, "format", "jpeg");
                match fmt.as_str() {
                    "gif" | "png8" => {
                        dropdown_str(
                            ui,
                            "web-palette",
                            f,
                            "palette",
                            &[
                                ("perceptual", tl!("Perceptual")),
                                ("selective", tl!("Selective")),
                                ("adaptive", tl!("Adaptive")),
                                ("restrictive", tl!("Restrictive (Web)")),
                                ("exact", tl!("Exact")),
                            ],
                            150.0,
                        );
                        number(ui, f, "colors", tl!("Colors"), 2.0..=256.0, "", 128.0);
                        dropdown_str(
                            ui,
                            "web-dither",
                            f,
                            "dither",
                            &[("none", tl!("No Dither")), ("diffusion", tl!("Diffusion")), ("pattern", tl!("Pattern")), ("noise", tl!("Noise"))],
                            150.0,
                        );
                        number(ui, f, "ditherAmount", tl!("Dither"), 0.0..=100.0, "%", 88.0);
                        check(ui, f, "transparency", tl!("Transparency"), true);
                        check(ui, f, "interlaced", tl!("Interlaced"), false);
                        number(ui, f, "webSnap", tl!("Web Snap"), 0.0..=100.0, "%", 0.0);
                    }
                    "jpeg" => {
                        number(ui, f, "quality", tl!("Quality"), 0.0..=100.0, "", 60.0);
                        check(ui, f, "progressive", tl!("Progressive"), false);
                        check(ui, f, "optimized", tl!("Optimized"), true);
                        check(ui, f, "embedIcc", tl!("Embed Color Profile"), false);
                    }
                    "wbmp" => {
                        dropdown_str(
                            ui,
                            "web-dither",
                            f,
                            "dither",
                            &[("none", tl!("No Dither")), ("diffusion", tl!("Diffusion")), ("pattern", tl!("Pattern"))],
                            150.0,
                        );
                    }
                    _ => {
                        check(ui, f, "transparency", tl!("Transparency"), true);
                        check(ui, f, "interlaced", tl!("Interlaced"), false);
                    }
                }
                if fmt != "png24" || !b(f, "transparency", true) {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(tl!("Matte")).color(t.text_dim));
                        let mut m = s(f, "matte", "#ffffff");
                        if ui.add(egui::TextEdit::singleline(&mut m).desired_width(80.0)).changed() {
                            f.insert("matte".into(), json!(m));
                        }
                    });
                }
                ui.add_space(6.0);
                check(ui, f, "convertToSrgb", tl!("Convert to sRGB"), true);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(tl!("Metadata")).color(t.text_dim));
                    dropdown_str(
                        ui,
                        "web-meta",
                        f,
                        "metadata",
                        &[
                            ("none", tl!("None")),
                            ("copyright", tl!("Copyright")),
                            ("copyrightAndContact", tl!("Copyright and Contact Info")),
                            ("all", tl!("All")),
                        ],
                        160.0,
                    );
                });
                if matches!(fmt.as_str(), "gif" | "png8") {
                    web_preview::table(ui, f);
                }
                ui.add_space(8.0);
                ui.label(egui::RichText::new(tl!("Image Size")).font(crate::theme::semibold(12.0)).color(t.text));
                number(ui, f, "percent", tl!("Percent"), 1.0..=1000.0, "%", 100.0);
                let pct = n(f, "percent", 100.0) / 100.0;
                ui.label(
                    egui::RichText::new(format!("W: {} px   H: {} px", (f64::from(doc.size.width) * pct).round(), (f64::from(doc.size.height) * pct).round()))
                        .color(t.text_dim)
                        .size(11.5),
                );
                let nslices = photocraft_doc::slices::resolve(&doc).len();
                if !doc.slices.is_empty() {
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new(crate::i18n::trn(crate::i18n::current(), nslices as u64, "{n} slice", "{n} slices")).color(t.text_dim));
                    check(ui, f, "html", tl!("Save HTML and Images"), true);
                    dropdown_str(ui, "web-slices", f, "slices", &[("all", tl!("All Slices")), ("user", tl!("All User Slices"))], 150.0);
                }
            });
        });
    });
}

fn web_confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    // Check while the control/input policy is still installed, before an asynchronous picker
    // can outlive the request. A browser download uses the interactive platform writer.
    let authorize = |p: &Value| -> Result<(), String> {
        if let Some(authorize) = app.session.authorize.or(if app.automation_input { app.services.automation_authorize } else { None }) {
            authorize(WEB_COMMAND, p).map_err(|e| e.to_string())?;
        }
        if app.automation_input
            && let Some(authorize) = app.services.automation_command.as_ref()
        {
            authorize(WEB_COMMAND, p)?;
        }
        Ok(())
    };
    let mut p = params(f);
    authorize(&p)?;
    // Confirmation may follow a field edit without an intervening frame (Enter or automation).
    let mut fields = f.clone();
    web_preview::prepare_table(app, &mut fields)?;
    let prepared = params(&fields);
    if prepared != p {
        // The policy must also approve the actual palette-bearing export parameters.
        authorize(&prepared)?;
        p = prepared;
    }
    if p.get("path").is_some() || p.get("dir").is_some() {
        return save_for_web(app, p);
    }
    let doc = app.session.active().ok_or("no document")?.doc.clone();
    let has_slices = !doc.slices.is_empty();
    let html = b(f, "html", true);
    let st = WebSettings::from_params(&p, "file.export.saveForWebLegacy").map_err(|e| e.to_string())?;
    let base = photocraft_doc::slices::base_name(&doc);
    let suggested = if has_slices && html { format!("{base}.html") } else { format!("{base}.{}", st.format.ext()) };
    app.pick_save(&suggested, move |app, path| {
        app.refocus(doc.id)?;
        if has_slices {
            let dir = path.rfind(['/', '\\']).map(|i| path[..i].to_string()).unwrap_or_else(|| ".".into());
            p["dir"] = json!(dir);
            p["html"] = json!(html);
        } else {
            p["path"] = json!(path);
        }
        save_for_web(app, p)
    })
}

/// Runs Save for Web with `p`, which says where to write.
fn save_for_web(app: &mut PhotocraftApp, p: Value) -> Result<Value, String> {
    #[cfg(target_arch = "wasm32")]
    let result = download_for_web(app, p);
    #[cfg(not(target_arch = "wasm32"))]
    let result = app.run(WEB_COMMAND, p);
    let r = result.inspect_err(|error| {
        app.ui.status = error.clone();
        app.ui.status_error = true;
    })?;
    app.sync_views();
    app.ui.status =
        crate::i18n::trn(crate::i18n::current(), r["files"].as_array().map_or(0, Vec::len) as u64, "Saved for Web: {n} file", "Saved for Web: {n} files");
    app.ui.status_error = false;
    Ok(r)
}

#[cfg(any(test, target_arch = "wasm32"))]
fn download_for_web(app: &mut PhotocraftApp, mut p: Value) -> Result<Value, String> {
    // Browser downloads have no directory hierarchy. Keep the HTML's image and spacer URLs
    // beside the downloaded page, matching the names handed to the platform service.
    p["imagesFolder"] = json!("");
    let write = app.services.write.as_mut().ok_or("no download service")?;
    web_cmds::save_for_web_with_writer(&mut app.session, &p, &mut |path, bytes| write(path, bytes).map_err(photocraft_engine::EngineError::Other))
        .map_err(|e| e.to_string())
}

// ---------- Print ----------

pub fn open_print(app: &mut PhotocraftApp) -> u64 {
    let mut f = Map::new();
    f.insert("__print".into(), json!(true));
    f.insert("__label".into(), json!("PhotoCraft Print Settings"));
    // Asked once per opening: the Printer menu lists what the spooler knows now.
    let printers = app.session.execute("file.printers", json!({})).unwrap_or_default();
    f.insert("__systemPrinters".into(), printers["printers"].clone());
    if let Some(Value::Object(m)) = app.session.file_menu.last_print.clone() {
        f.extend(m);
    }
    for (k, v) in [
        ("printer", json!("")),
        ("copies", json!(1)),
        ("paper", json!("letter")),
        ("orientation", json!("portrait")),
        ("colorHandling", json!("printerManages")),
        ("printerProfile", json!("coated-cmyk")),
        ("intent", json!("relative")),
        ("bpc", json!(true)),
        ("center", json!(true)),
        ("top", json!(0.0)),
        ("left", json!(0.0)),
        ("scale", json!(100.0)),
        ("scaleToFit", json!(false)),
        ("cornerCropMarks", json!(false)),
        ("centerCropMarks", json!(false)),
        ("registrationMarks", json!(false)),
        ("description", json!(false)),
        ("labels", json!(false)),
        ("emulsionDown", json!(false)),
        ("units", json!("cm")),
        ("output", json!("")),
    ] {
        f.entry(k.to_string()).or_insert(v);
    }
    // Windows: the system's printers, and the paper from the chosen printer's driver settings.
    if let Ok(list) = craft_print::rgb::printers() {
        f.insert("__printers".into(), json!(list.iter().map(|p| p.name.clone()).collect::<Vec<_>>()));
        let known = |n: &str| list.iter().any(|p| p.name == n);
        if !known(&s(&f, "printer", "")) {
            let fallback = list.iter().find(|p| p.default).or(list.first()).map(|p| p.name.clone()).unwrap_or_default();
            f.insert("printer".into(), json!(fallback));
            f.remove("printerSettings");
        }
    }
    app.ui.open_dialog(DialogKind::Command, f)
}

/// Takes the paper from the printer's driver settings (Windows), as Photoshop does.
fn driver_page(f: &mut Map<String, Value>) {
    let printer = s(f, "printer", "");
    let settings = f.get("printerSettings").and_then(Value::as_str).and_then(|h| craft_print::rgb::DevMode::from_hex(h).ok());
    match craft_print::rgb::page(&printer, settings.as_ref()) {
        Ok(pg) => {
            f.insert("paper".into(), json!([pg.paper.0, pg.paper.1]));
            f.insert("orientation".into(), json!("portrait"));
            f.remove("__pageError");
        }
        Err(e) => {
            f.insert("__pageError".into(), json!(e.to_string()));
        }
    }
}

/// Points per unit for the Position and Size fields.
fn unit_pt(units: &str) -> f64 {
    match units {
        "mm" => 72.0 / 25.4,
        "in" => 72.0,
        _ => 72.0 / 2.54,
    }
}

/// A number field in `units` for a value kept in points; true when edited.
fn length_field(ui: &mut egui::Ui, label: &str, pt: &mut f64, units: &str) -> bool {
    length_field_prec(ui, label, pt, units, 0)
}

fn length_field_prec(ui: &mut egui::Ui, label: &str, pt: &mut f64, units: &str, decimals: u32) -> bool {
    let t = Tokens::get(ui.ctx());
    let k = unit_pt(units);
    let mut v = (*pt / k) as f32;
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).color(t.text_dim));
        let before = v;
        crate::widgets::value_field_prec(ui, &mut v, 0.0..=10000.0, units, 70.0, decimals);
        if (v - before).abs() > 1e-4 && v.is_finite() {
            *pt = f64::from(v) * k;
            changed = true;
        }
    });
    changed
}

/// Unrotated paper dimensions in points, for either the current preset or a custom array.
fn print_paper_size(value: Option<&Value>) -> (f64, f64) {
    let default = (612.0, 792.0);
    match value {
        Some(Value::Array(a)) => (a.first().and_then(Value::as_f64).unwrap_or(default.0), a.get(1).and_then(Value::as_f64).unwrap_or(default.1)),
        Some(Value::String(name)) => photocraft_engine::print_cmds::PAPERS.iter().find(|p| p.0.eq_ignore_ascii_case(name)).map_or(default, |p| (p.1, p.2)),
        _ => default,
    }
}

fn print_paper_controls(ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    let previous = f.get("paper").cloned();
    let mut selected = if previous.as_ref().is_some_and(Value::is_array) { "custom".to_string() } else { s(f, "paper", "letter") };
    ui.horizontal(|ui| {
        let mut papers: Vec<_> = photocraft_engine::print_cmds::PAPERS.iter().map(|p| (p.0.to_string(), p.0)).collect();
        papers.push(("custom".to_string(), tl!("Custom")));
        if crate::widgets::dropdown(ui, "print-paper", &mut selected, &papers, 110.0) {
            if selected == "custom" {
                let (w, h) = print_paper_size(f.get("__customPaper").or(previous.as_ref()));
                f.insert("paper".into(), json!([w, h]));
            } else {
                if let Some(paper @ Value::Array(_)) = previous {
                    f.insert("__customPaper".into(), paper);
                }
                f.insert("paper".into(), json!(selected));
            }
        }
        dropdown_str(ui, "print-orient", f, "orientation", &[("portrait", tl!("Portrait")), ("landscape", tl!("Landscape"))], 110.0);
    });
    if f.get("paper").is_some_and(Value::is_array) {
        let (mut w, mut h) = print_paper_size(f.get("paper"));
        let units = s(f, "units", "cm");
        let changed = ui
            .horizontal(|ui| {
                let wc = length_field_prec(ui, tl!("Paper Width"), &mut w, &units, 2);
                let hc = length_field_prec(ui, tl!("Paper Height"), &mut h, &units, 2);
                wc || hc
            })
            .inner;
        if changed {
            // Keep points in the command, and let the engine apply orientation and validation.
            f.insert("paper".into(), json!([w, h]));
            f.insert("__customPaper".into(), json!([w, h]));
        }
    }
}

fn print_body(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    let t = Tokens::get(ui.ctx());
    let Some(doc) = app.session.active().map(|d| d.doc.clone()) else { return };
    // PDF exports and previews retain editable paper on Windows too. Only a real native
    // submission takes its media from the printer; entering that mode refreshes the driver
    // geometry after any export-only paper edits.
    let printers: Option<Vec<String>> = if !b(f, "dryRun", false) && s(f, "output", "").is_empty() {
        f.get("__printers").and_then(Value::as_array).map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
    } else {
        None
    };
    if printers.is_some() {
        if !b(f, "__driverPageInitialized", false) {
            driver_page(f);
            f.insert("__driverPageInitialized".into(), json!(true));
        }
    } else {
        f.remove("__driverPageInitialized");
    }
    let p = params(f);
    ui.horizontal_top(|ui| {
        // Page preview.
        ui.vertical(|ui| {
            let (area, _) = ui.allocate_exact_size(egui::vec2(330.0, 400.0), egui::Sense::hover());
            ui.painter().rect_filled(area, 0.0, t.canvas);
            match photocraft_engine::print_cmds::layout(&doc, &p, "file.print") {
                Ok(l) => {
                    let (pw, ph) = l.paper;
                    let k = ((area.width() - 20.0) / pw as f32).min((area.height() - 20.0) / ph as f32);
                    let page = egui::Rect::from_center_size(area.center(), egui::vec2(pw as f32 * k, ph as f32 * k));
                    ui.painter().rect_filled(page, 0.0, egui::Color32::WHITE);
                    ui.painter().rect_stroke(page, 0.0, egui::Stroke::new(1.0, t.separator), egui::StrokeKind::Outside);
                    let (x, y, w, h) = l.rect;
                    // Emulsion Down shows the page as it prints: mirrored.
                    let x = if l.emulsion_down { pw - x - w } else { x };
                    let img = egui::Rect::from_min_size(
                        egui::pos2(page.left() + x as f32 * k, page.bottom() - (y + h) as f32 * k),
                        egui::vec2(w as f32 * k, h as f32 * k),
                    );
                    let key = egui::Id::new(("print-thumb", doc.id.0, app.session.active().map_or(0, |d| d.revision)));
                    let tex: Option<Arc<egui::TextureHandle>> = ui.data(|d| d.get_temp(key));
                    let tex = tex.unwrap_or_else(|| {
                        let th = photocraft_compose::thumbnail(&doc, 400);
                        let ci = egui::ColorImage::from_rgba_unmultiplied([th.width as usize, th.height as usize], &th.pixels);
                        let tx = Arc::new(ui.ctx().load_texture("print-thumb", ci, egui::TextureOptions::LINEAR));
                        ui.data_mut(|d| d.insert_temp(key, tx.clone()));
                        tx
                    });
                    let clip = ui.painter().with_clip_rect(page);
                    let uv = if l.emulsion_down {
                        egui::Rect::from_min_max(egui::pos2(1.0, 0.0), egui::pos2(0.0, 1.0))
                    } else {
                        egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0))
                    };
                    clip.image(tex.id(), img, uv, egui::Color32::WHITE);
                    let mark = egui::Stroke::new(1.0, egui::Color32::BLACK);
                    if l.marks.corner_crop {
                        for (c, dx, dy) in
                            [(img.left_top(), -1.0, -1.0), (img.right_top(), 1.0, -1.0), (img.left_bottom(), -1.0, 1.0), (img.right_bottom(), 1.0, 1.0)]
                        {
                            clip.line_segment([c + egui::vec2(dx * 3.0, 0.0), c + egui::vec2(dx * 12.0, 0.0)], mark);
                            clip.line_segment([c + egui::vec2(0.0, dy * 3.0), c + egui::vec2(0.0, dy * 12.0)], mark);
                        }
                    }
                    if l.marks.registration {
                        for c in [
                            img.center_top() - egui::vec2(0.0, 9.0),
                            img.center_bottom() + egui::vec2(0.0, 9.0),
                            img.left_center() - egui::vec2(9.0, 0.0),
                            img.right_center() + egui::vec2(9.0, 0.0),
                        ] {
                            clip.circle_stroke(c, 3.5, mark);
                        }
                    }
                    let fits = x >= 0.0 && y >= 0.0 && x + w <= pw && y + h <= ph;
                    let note = if fits { String::new() } else { "  ·  larger than the paper".to_string() };
                    ui.label(
                        egui::RichText::new(format!("Scale {:.1}%  ·  {:.2} × {:.2} in{note}", l.scale * 100.0, w / 72.0, h / 72.0))
                            .color(if fits { t.text_dim } else { t.warning })
                            .size(11.0),
                    );
                }
                Err(e) => {
                    ui.painter().text(area.center(), egui::Align2::CENTER_CENTER, e.to_string(), egui::FontId::proportional(12.0), t.text_dim);
                }
            }
        });
        ui.add_space(12.0);
        ui.vertical(|ui| {
            ui.set_width(340.0);
            let head = |ui: &mut egui::Ui, s: &str| {
                ui.add_space(4.0);
                ui.label(egui::RichText::new(s).font(crate::theme::semibold(12.0)).color(t.text));
            };
            head(ui, tl!("Printer Setup"));
            if let Some(printers) = &printers {
                // Windows: Photoshop's Printer menu and Print Settings… (the driver's own dialog).
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(tl!("Printer")).color(t.text_dim));
                    let before = s(f, "printer", "");
                    let opts: Vec<(&str, &str)> = printers.iter().map(|n| (n.as_str(), n.as_str())).collect();
                    dropdown_str(ui, "print-printer", f, "printer", &opts, 220.0);
                    if s(f, "printer", "") != before {
                        f.remove("printerSettings");
                        driver_page(f);
                    }
                });
                ui.horizontal(|ui| {
                    number(ui, f, "copies", tl!("Copies"), 1.0..=999.0, "", 1.0);
                    if ui.button(tl!("Print Settings…")).clicked() {
                        let printer = s(f, "printer", "");
                        let current = f.get("printerSettings").and_then(Value::as_str).and_then(|h| craft_print::rgb::DevMode::from_hex(h).ok());
                        match craft_print::rgb::settings_dialog(&printer, current.as_ref(), None) {
                            Ok(Some(dm)) => {
                                f.insert("printerSettings".into(), json!(dm.to_hex()));
                                driver_page(f);
                            }
                            Ok(None) => {}
                            Err(e) => {
                                f.insert("__pageError".into(), json!(e.to_string()));
                            }
                        }
                    }
                });
                let paper = p.get("paper").and_then(Value::as_array).map(|a| (a.first().and_then(Value::as_f64), a.get(1).and_then(Value::as_f64)));
                let text = match paper {
                    Some((Some(w), Some(h))) => format!("{:.0} × {:.0} mm  ·  {}", w / 72.0 * 25.4, h / 72.0 * 25.4, tl!("set in Print Settings…")),
                    _ => String::new(),
                };
                ui.label(egui::RichText::new(text).color(t.text_dim).size(11.0));
                if let Some(e) = f.get("__pageError").and_then(Value::as_str) {
                    ui.label(egui::RichText::new(e).color(t.warning).size(11.0));
                }
            } else {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(tl!("Printer")).color(t.text_dim));
                    let mut pr = s(f, "printer", "");
                    let opts = printer_options(f.get("__systemPrinters"), &pr);
                    let o: Vec<(String, &str)> = opts.iter().map(|(k, l)| (k.clone(), l.as_str())).collect();
                    if crate::widgets::dropdown(ui, "print-printer", &mut pr, &o, 240.0) {
                        f.insert("printer".into(), json!(pr));
                    }
                });
                number(ui, f, "copies", tl!("Copies"), 1.0..=999.0, "", 1.0);
                print_paper_controls(ui, f);
            }
            head(ui, tl!("Color Management"));
            dropdown_str(
                ui,
                "print-color",
                f,
                "colorHandling",
                &[
                    ("printerManages", tl!("Printer Manages Colors")),
                    ("photocraftManages", tl!("PhotoCraft Manages Colors")),
                    ("noColorManagement", tl!("No Color Management")),
                ],
                240.0,
            );
            if s(f, "colorHandling", "") == "photocraftManages" {
                // Photoshop's Printer Profile menu: the built-ins, then every installed printer
                // (output class) profile by its description.
                let installed: Vec<_> = photocraft_engine::installed_profiles::installed()
                    .into_iter()
                    .filter(|p| p.is_destination() && p.class == photocraft_cms::ProfileClass::Output)
                    .collect();
                let mut profiles: Vec<(&str, &str)> = vec![
                    ("coated-cmyk", tl!("Coated CMYK")),
                    ("srgb", "sRGB IEC61966-2.1"),
                    ("adobe-rgb-compat", tl!("Adobe RGB (1998) compatible")),
                    ("display-p3", tl!("Display P3")),
                    ("gray-gamma-2.2", tl!("Gray Gamma 2.2")),
                ];
                profiles.extend(installed.iter().map(|p| (p.path.as_str(), p.description.as_str())));
                // A remembered profile that is no longer installed still shows, by its file name.
                let cur = s(f, "printerProfile", "coated-cmyk");
                let stem = std::path::Path::new(&cur).file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                if !profiles.iter().any(|(v, _)| *v == cur) {
                    profiles.push((cur.as_str(), stem.as_str()));
                }
                dropdown_str(ui, "print-profile", f, "printerProfile", &profiles, 240.0);
                ui.horizontal(|ui| {
                    dropdown_str(
                        ui,
                        "print-intent",
                        f,
                        "intent",
                        &[
                            ("perceptual", tl!("Perceptual")),
                            ("relative", tl!("Relative Colorimetric")),
                            ("saturation", tl!("Saturation")),
                            ("absolute", tl!("Absolute Colorimetric")),
                        ],
                        170.0,
                    );
                    check(ui, f, "bpc", tl!("Black Point Compensation"), true);
                });
                if printers.is_some() {
                    ui.label(
                        egui::RichText::new(tl!("Remember to turn off the printer's own colour management in Print Settings…")).color(t.warning).size(11.0),
                    );
                }
            }
            head(ui, tl!("Position and Size"));
            dropdown_str(ui, "print-units", f, "units", &[("cm", tl!("Centimeters")), ("mm", tl!("Millimeters")), ("in", tl!("Inches"))], 130.0);
            let units = s(f, "units", "cm");
            check(ui, f, "center", tl!("Center"), true);
            if !b(f, "center", true) {
                ui.horizontal(|ui| {
                    // Kept in inches (Photoshop's Position fields), shown in the chosen units.
                    for (key, label) in [("top", tl!("Top")), ("left", tl!("Left"))] {
                        let mut pt = n(f, key, 0.0) * 72.0;
                        if length_field(ui, label, &mut pt, &units) {
                            f.insert(key.into(), json!(pt / 72.0));
                        }
                    }
                });
            }
            check(ui, f, "scaleToFit", tl!("Scale to Fit Media"), false);
            // Scaled Print Size: Scale, Height and Width move together; Print Resolution follows.
            if let Ok(l) = photocraft_engine::print_cmds::layout(&doc, &params(f), "file.print") {
                let (_, _, w, h) = l.rect;
                let (mut w2, mut h2) = (w, h);
                let fit = b(f, "scaleToFit", false);
                if !fit {
                    // Two decimals: Height and Width set fractional scales.
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(tl!("Scale")).color(t.text_dim));
                        let mut v = n(f, "scale", 100.0) as f32;
                        crate::widgets::value_field(ui, &mut v, 1.0..=1000.0, "%", 64.0);
                        f.insert("scale".into(), json!((f64::from(v) * 100.0).round() / 100.0));
                    });
                }
                ui.horizontal(|ui| {
                    let hc = length_field(ui, tl!("Height"), &mut h2, &units);
                    let wc = length_field(ui, tl!("Width"), &mut w2, &units);
                    let k = if wc && w > 0.0 {
                        Some(w2 / w)
                    } else if hc && h > 0.0 {
                        Some(h2 / h)
                    } else {
                        None
                    };
                    if let Some(k) = k.filter(|k| k.is_finite() && *k > 0.0) {
                        f.insert("scale".into(), json!(((l.scale * k * 100.0) * 100.0).round() / 100.0));
                        f.insert("scaleToFit".into(), json!(false));
                    }
                });
                if w > 0.0 {
                    let ppi = f64::from(doc.size.width) / (w / 72.0);
                    ui.label(egui::RichText::new(format!("{} {ppi:.0} PPI", tl!("Print Resolution:"))).color(t.text_dim).size(11.0));
                }
            }
            head(ui, tl!("Printing Marks"));
            ui.horizontal(|ui| {
                check(ui, f, "cornerCropMarks", tl!("Corner Crop Marks"), false);
                check(ui, f, "centerCropMarks", tl!("Center Crop Marks"), false);
            });
            ui.horizontal(|ui| {
                check(ui, f, "registrationMarks", tl!("Registration Marks"), false);
                check(ui, f, "description", tl!("Description"), false);
                check(ui, f, "labels", tl!("Labels"), false);
            });
            head(ui, tl!("Functions"));
            check(ui, f, "emulsionDown", tl!("Emulsion Down"), false);
            if b(f, "emulsionDown", false) && printers.is_some() {
                ui.label(egui::RichText::new(tl!("Don't also tick Mirror Image in Print Settings…: the two flips cancel out.")).color(t.text_dim).size(11.0));
            }
            head(ui, tl!("Save as PDF"));
            let mut out = s(f, "output", "");
            if ui.add(egui::TextEdit::singleline(&mut out).hint_text(tl!("(print to the printer)")).desired_width(300.0)).changed() {
                f.insert("output".into(), json!(out));
            }
        });
    });
}

fn print_confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    let r = app.run("file.print", params(f))?;
    if r["pending"] != json!(true) {
        print_status(app, &r);
    }
    Ok(r)
}

pub(crate) fn print_status(app: &mut PhotocraftApp, r: &Value) {
    app.ui.status = if let Some(printer) = r["spooler"]["printer"].as_str() {
        crate::i18n::fmt(tl!("Sent to {printer}"), &[("printer", printer)])
    } else if r["sent"] == json!(true) {
        let template = "Sent to the printer ({spooler})";
        crate::i18n::fmt(tl!(template), &[("spooler", r["spooler"].as_str().unwrap_or(""))])
    } else {
        let template = "Printed to {path}";
        crate::i18n::fmt(tl!(template), &[("path", r["pdf"].as_str().unwrap_or(""))])
    };
    app.ui.status_error = false;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> (PhotocraftApp, egui::Context) {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 120, "height": 80, "name": "web"})).unwrap();
        (app, egui::Context::default())
    }

    #[test]
    fn droplet_folder_follows_the_document_or_uses_a_safe_fallback() {
        let (mut app, _) = app();
        let fallback = droplet_default_dir(&app);
        #[cfg(target_os = "macos")]
        if std::env::var_os("HOME").is_some() {
            assert!(std::path::Path::new(&fallback).is_absolute());
        }
        #[cfg(not(target_os = "macos"))]
        assert_eq!(fallback, ".");
        app.session.active_mut().unwrap().path = Some("/tmp/photocraft/drawing.pcraft".into());
        assert_eq!(droplet_default_dir(&app), "/tmp/photocraft");
    }

    #[test]
    fn save_for_web_dialog_runs_the_command() {
        let (mut app, ctx) = app();
        let r = crate::menus::invoke(&mut app, &ctx, "file.export.saveForWebLegacy", json!({})).unwrap();
        let id = r["dialog"].as_u64().unwrap();
        let dir = std::env::temp_dir().join(format!("pc-webui-{}", std::process::id()));
        let path = dir.join("web.gif").to_string_lossy().into_owned();
        let d = app.ui.dialog_mut(id).unwrap();
        d.fields.insert("format".into(), json!("gif"));
        d.fields.insert("path".into(), json!(path));
        let r = crate::dialogs::confirm(&mut app, id).unwrap();
        assert_eq!(r["files"][0], json!(path));
        assert!(std::fs::read(&path).unwrap().starts_with(b"GIF89a"));
        // The next dialog starts from those settings.
        let r = crate::menus::invoke(&mut app, &ctx, "file.export.saveForWebLegacy", json!({})).unwrap();
        let d = app.ui.dialog_mut(r["dialog"].as_u64().unwrap()).unwrap();
        assert_eq!(d.fields["format"], "gif");
        assert!(!d.fields.contains_key("path"));
    }

    /// A fresh session's dialog seeds from the preferences: the last export before a restart is
    /// remembered there (`web_cmds::remember_web_settings`), as `file_menu` is not saved.
    #[test]
    fn the_save_for_web_dialog_seeds_from_the_remembered_preferences() {
        let (mut app, _) = app();
        app.session.prefs.edit(|p| {
            p.dialogs.insert(WEB_COMMAND.into(), json!({"format": "png8", "colors": 64}));
        });
        let id = open_web(&mut app);
        let f = &app.ui.dialog_mut(id).unwrap().fields;
        assert_eq!(f["format"], "png8");
        assert_eq!(f["colors"], 64);
    }

    #[test]
    fn save_for_web_starts_at_the_current_document_size() {
        let (mut app, _) = app();
        let path = std::env::temp_dir().join(format!("pc-webui-size-{}.png", std::process::id())).to_string_lossy().into_owned();
        app.run("file.export.saveForWebLegacy", json!({"path": path, "format": "png24", "width": 60, "height": 40, "percent": 50})).unwrap();
        app.run("file.new", json!({"width": 240, "height": 160, "name": "next"})).unwrap();
        let id = open_web(&mut app);
        let f = &app.ui.dialog_mut(id).unwrap().fields;
        assert_eq!(f["format"], "png24");
        assert_eq!(f["percent"], 100.0);
        assert!(!f.contains_key("width"));
        assert!(!f.contains_key("height"));
    }

    #[test]
    fn save_for_web_downloads_images_and_html_through_the_platform_writer() {
        let (mut app, _) = app();
        let downloads = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let captured = downloads.clone();
        app.services.write = Some(Box::new(move |path, bytes| {
            captured.borrow_mut().push((path.to_string(), bytes.to_vec()));
            Ok(())
        }));
        download_for_web(&mut app, json!({"path": "web.png", "format": "png24"})).unwrap();
        assert_eq!(downloads.borrow()[0].0, "web.png");
        assert!(downloads.borrow()[0].1.starts_with(b"\x89PNG\r\n\x1a\n"));
        downloads.borrow_mut().clear();

        app.run("slice.new", json!({"rect": [0, 0, 60, 80], "name": "left"})).unwrap();
        download_for_web(&mut app, json!({"dir": ".", "format": "png24", "html": true, "imagesFolder": "images"})).unwrap();
        let files = downloads.borrow();
        let html = std::str::from_utf8(&files.iter().find(|(path, _)| path.ends_with(".html")).unwrap().1).unwrap();
        assert!(!html.contains("images/"), "downloaded files are siblings, not an images directory");
        for (path, bytes) in files.iter().filter(|(path, _)| !path.ends_with(".html")) {
            let name = std::path::Path::new(path).file_name().unwrap().to_str().unwrap();
            assert!(html.contains(&format!("src=\"{name}\"")), "missing downloaded image {name}");
            assert!(!bytes.is_empty());
        }
        assert!(files.iter().any(|(path, _)| path.ends_with("spacer.gif")));
        drop(files);

        app.services.write = Some(Box::new(|_, _| Err("download failed".into())));
        assert_eq!(download_for_web(&mut app, json!({"path": "fail.png", "format": "png24"})).unwrap_err(), "download failed");
        app.services.write = None;
        assert_eq!(download_for_web(&mut app, json!({"path": "missing.png"})).unwrap_err(), "no download service");
    }

    #[test]
    fn save_for_web_authorizes_before_opening_a_picker() {
        fn deny(_: &str, _: &Value) -> photocraft_engine::Result<()> {
            Err(photocraft_engine::EngineError::Other("export denied".into()))
        }
        for explicit_path in [false, true] {
            for synthetic_input in [false, true] {
                let (mut app, ctx) = app();
                let id = open_web(&mut app);
                if explicit_path {
                    app.ui.dialog_mut(id).unwrap().fields.insert("path".into(), json!("denied.png"));
                }
                app.services.automation_authorize = Some(deny);
                app.services.file_dialog = Some(Box::new(|_, _, _| panic!("denied export must not show a picker")));
                app.services.write = Some(Box::new(|_, _| panic!("denied export must not write")));
                let error = if synthetic_input {
                    app.automation_input = true;
                    crate::dialogs::confirm(&mut app, id).unwrap_err()
                } else {
                    let (request, _) = crate::control::ControlRequest::new("ui.dialog.confirm", json!({"dialog": id}));
                    let crate::control::Outcome::Done(reply) = crate::control::handle(&mut app, &ctx, &request) else { panic!("expected immediate denial") };
                    assert_eq!(reply["ok"], false);
                    reply["error"].as_str().unwrap().to_string()
                };
                assert_eq!(error, "export denied");
                assert!(!app.file_dialog_open());
            }
        }
    }

    #[test]
    fn save_for_web_authorizes_generated_palette_before_opening_a_picker() {
        fn deny_palette(_: &str, p: &Value) -> photocraft_engine::Result<()> {
            if p.get("colorTable").is_some() { Err(photocraft_engine::EngineError::Other("palette denied".into())) } else { Ok(()) }
        }
        for synthetic_input in [false, true] {
            let (mut app, _) = app();
            let id = open_web(&mut app);
            let fields = &mut app.ui.dialog_mut(id).unwrap().fields;
            fields.insert("format".into(), json!("gif"));
            assert!(!fields.contains_key("colorTable"));
            if synthetic_input {
                app.automation_input = true;
                app.services.automation_command = Some(Box::new(|cmd, p| deny_palette(cmd, p).map_err(|e| e.to_string())));
            } else {
                app.session.authorize = Some(deny_palette);
            }
            app.services.file_dialog = Some(Box::new(|_, _, _| panic!("denied palette must not show a picker")));
            app.services.write = Some(Box::new(|_, _| panic!("denied palette must not write")));
            assert_eq!(crate::dialogs::confirm(&mut app, id).unwrap_err(), "palette denied");
            assert!(!app.file_dialog_open());
        }
    }

    #[test]
    fn web_preview_grows_with_resize_and_accepts_zoom_and_pan() {
        let (mut app, _) = app();
        let id = open_web(&mut app);
        let revision = app.session.active().unwrap().revision;
        app.ui.dialog_mut(id).unwrap().fields.insert("__view".into(), json!("optimized"));
        let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(1600.0, 1100.0)).build_ui_state(
            |ui, app| {
                if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("semibold".into()))) {
                    crate::dialogs::show(app, ui.ctx());
                }
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, Default::default());
        h.run_steps(5);
        let pane = |h: &egui_kittest::Harness<'_, PhotocraftApp>| {
            let r = &h.state().ui.dialogs[0].fields["__previewRects"][0];
            egui::Rect::from_min_max(
                egui::pos2(r[0].as_f64().unwrap() as f32, r[1].as_f64().unwrap() as f32),
                egui::pos2(r[2].as_f64().unwrap() as f32, r[3].as_f64().unwrap() as f32),
            )
        };
        let before = pane(&h);
        let rects: Vec<egui::Rect> = h.ctx.data(|m| m.get_temp(egui::Id::new("pc-dialog-rects"))).unwrap();
        let corner = rects[0].max - egui::vec2(14.0, 14.0);
        h.event(egui::Event::PointerMoved(corner));
        h.step();
        h.event(egui::Event::PointerButton { pos: corner, button: egui::PointerButton::Primary, pressed: true, modifiers: egui::Modifiers::NONE });
        h.step();
        let end = corner + egui::vec2(180.0, 100.0);
        h.event(egui::Event::PointerMoved(end));
        h.step();
        h.event(egui::Event::PointerButton { pos: end, button: egui::PointerButton::Primary, pressed: false, modifiers: egui::Modifiers::NONE });
        h.run_steps(4);
        let after = pane(&h);
        assert!(after.width() > before.width() + 100.0, "before {before:?}, after {after:?}, corner {corner:?}");
        assert!(after.height() > before.height() + 50.0);
        // Zoom in enough that the small fixture can pan, then drag the real pane.
        h.state_mut().ui.dialogs[0].fields.insert("__zoom".into(), json!(16.0));
        h.run_steps(3);
        let at = pane(&h).center();
        h.event(egui::Event::PointerMoved(at));
        h.step();
        h.event(egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed: true, modifiers: egui::Modifiers::NONE });
        h.step();
        let end = at + egui::vec2(35.0, 20.0);
        h.event(egui::Event::PointerMoved(end));
        h.step();
        h.event(egui::Event::PointerButton { pos: end, button: egui::PointerButton::Primary, pressed: false, modifiers: egui::Modifiers::NONE });
        h.run_steps(3);
        assert_ne!(h.state().ui.dialogs[0].fields["__center"], json!([0.5, 0.5]));
        assert_eq!(h.state().session.active().unwrap().revision, revision);
        // Enter while editing a swatch must not export and close the parent dialog.
        h.state_mut().ui.dialogs[0].fields.insert("format".into(), json!("gif"));
        h.run_steps(4);
        use egui_kittest::kittest::Queryable;
        h.get_by_role(egui::accesskit::Role::ColorWell).click();
        h.run_steps(3);
        assert!(egui::Popup::is_any_open(&h.ctx));
        h.key_press(egui::Key::Enter);
        h.run_steps(3);
        assert_eq!(h.state().ui.dialogs.len(), 1);
    }

    #[test]
    fn palette_draft_survives_navigation_and_resets_when_colors_change() {
        let (app, _) = app();
        let mut f = serde_json::from_value::<Map<String, Value>>(json!({"format": "gif", "colors": 16})).unwrap();
        web_preview::prepare_table(&app, &mut f).unwrap();
        let table = f["colorTable"].clone();
        f.insert("colorTableEdits".into(), json!({"0": "#12ab34"}));
        f.insert("__zoom".into(), json!(2.0));
        f.insert("__center".into(), json!([0.25, 0.75]));
        web_preview::prepare_table(&app, &mut f).unwrap();
        assert_eq!(f["colorTable"], table);
        assert_eq!(f["colorTableEdits"]["0"], "#12ab34");
        f.insert("metadata".into(), json!("none"));
        f.insert("interlaced".into(), json!(true));
        f.insert("dither".into(), json!("pattern"));
        web_preview::prepare_table(&app, &mut f).unwrap();
        assert_eq!(f["colorTableEdits"]["0"], "#12ab34", "non-palette settings must retain edits");
        f.insert("colors".into(), json!(8));
        web_preview::prepare_table(&app, &mut f).unwrap();
        assert!(!f.contains_key("colorTableEdits"));
        assert!(f["colorTable"].as_array().unwrap().len() <= 8);
    }

    #[test]
    fn print_dialog_and_toggles() {
        let (mut app, ctx) = app();
        let r = crate::menus::invoke(&mut app, &ctx, "file.print", json!({})).unwrap();
        let id = r["dialog"].as_u64().unwrap();
        let out = std::env::temp_dir().join(format!("pc-printui-{}.pdf", std::process::id())).to_string_lossy().into_owned();
        app.ui.dialog_mut(id).unwrap().fields.insert("output".into(), json!(out));
        let r = crate::dialogs::confirm(&mut app, id).unwrap();
        assert_eq!(r["sent"], false);
        assert!(std::fs::read(&out).unwrap().starts_with(b"%PDF"));
        assert_eq!(checked(&app, "view.lockSlices"), Some(false));
        crate::menus::invoke(&mut app, &ctx, "view.lockSlices", json!({})).unwrap();
        assert_eq!(checked(&app, "view.lockSlices"), Some(true));
        crate::menus::invoke(&mut app, &ctx, "file.generate.imageAssets", json!({})).unwrap();
        assert_eq!(checked(&app, "file.generate.imageAssets"), Some(true));
        // Form dialogs open for the rest.
        for id in ["file.automate.contactSheetII", "file.scripts.statistics", "file.scripts.scriptEventsManager", "file.package"] {
            let r = crate::menus::invoke(&mut app, &ctx, id, json!({})).unwrap();
            assert!(r["dialog"].is_u64(), "{id}");
        }
    }

    #[test]
    fn printer_menu_lists_the_spooler_printers_after_the_default() {
        let list = json!([{"name": "HP_OfficeJet_3830_series", "default": true}, {"name": "DNP_DP_DS620", "default": false}]);
        let o = printer_options(Some(&list), "");
        let keys: Vec<&str> = o.iter().map(|p| p.0.as_str()).collect();
        assert_eq!(keys, ["", "HP_OfficeJet_3830_series", "DNP_DP_DS620"]);
        assert!(o[0].1.ends_with("(HP OfficeJet 3830 series)"), "{}", o[0].1);
        assert_eq!(o[2].1, "DNP DP DS620");
        // A remembered printer that's gone stays selectable; no list still offers the default.
        let o = printer_options(Some(&list), "Old_Queue");
        assert_eq!(o.last().unwrap().0, "Old_Queue");
        assert_eq!(printer_options(None, "").len(), 1);
        // The dialog asks the engine, and the list never reaches file.print's params.
        let (mut app, ctx) = app();
        let id = crate::menus::invoke(&mut app, &ctx, "file.print", json!({})).unwrap()["dialog"].as_u64().unwrap();
        let f = &app.ui.dialog_mut(id).unwrap().fields;
        assert!(f.get("__systemPrinters").is_some_and(Value::is_array));
        assert!(params(f).get("__systemPrinters").is_none());
        assert!(params(f).get("__printers").is_none());
    }

    #[test]
    fn print_dialog_does_not_inherit_scripted_send_override() {
        for send in [false, true] {
            let (mut app, ctx) = app();
            let out = std::env::temp_dir().join(format!("pc-printui-state-{}-{send}.pdf", std::process::id())).to_string_lossy().into_owned();
            app.session.execute("file.print", json!({"output": out, "send": send, "dryRun": true})).unwrap();
            assert!(app.session.file_menu.last_print.as_ref().unwrap().get("send").is_none());
            let r = crate::menus::invoke(&mut app, &ctx, "file.print", json!({})).unwrap();
            let id = r["dialog"].as_u64().unwrap();
            assert!(app.ui.dialog_mut(id).unwrap().fields.get("send").is_none());
            std::fs::remove_file(out).unwrap();
        }
    }
}

#[cfg(test)]
mod web_playback_tests {
    use super::*;
    use egui_kittest::{Harness, kittest::Queryable};

    fn animated_app(depth: u32) -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 32, "height": 24, "depth": depth})).unwrap();
        app.run("timeline.create", json!({"duration": 3, "fps": 2})).unwrap();
        app.run("layer.videoLayers.newBlankVideoLayer", json!({})).unwrap();
        for (frame, color) in ["#ff0000", "#00ff00", "#0000ff"].iter().enumerate() {
            app.run("timeline.setFrame", json!({"frame":frame})).unwrap();
            app.run("edit.fill", json!({"color":color})).unwrap();
        }
        app.run("timeline.setFrame", json!({"frame":0})).unwrap();
        app
    }

    fn harness(app: PhotocraftApp, fields: Map<String, Value>) -> Harness<'static, (PhotocraftApp, Map<String, Value>)> {
        let mut h = Harness::builder().with_size(egui::vec2(1000.0, 800.0)).with_step_dt(0.1).build_ui_state(
            |ui, state: &mut (PhotocraftApp, Map<String, Value>)| {
                if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    return;
                }
                web_body(&mut state.0, ui, &mut state.1);
            },
            (app, fields),
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
        h.run_steps(3);
        h
    }

    #[test]
    fn animated_save_for_web_offers_playback() {
        let mut app = animated_app(8);
        let id = open_web(&mut app);
        let mut fields = app.ui.dialog_mut(id).unwrap().fields.clone();
        fields.insert("__view".into(), json!("original"));
        fields.insert("format".into(), json!("gif"));
        let mut h = harness(app, fields);
        assert!(h.query_by_label("Play").is_some(), "an animated document needs a preview playback control");
        let original = h.state().0.session.active().unwrap().doc.clone();
        let revision = h.state().0.session.active().unwrap().revision;
        let history = h.state().0.session.active().unwrap().history.past_len();
        h.get_by_label("Play").click();
        h.run_steps(2);
        assert_eq!(h.state().1["__web_playing"], true);
        h.run_steps(6);
        assert_ne!(h.state().1["__web_frame"], 0);
        h.get_by_label("Pause").click();
        h.run_steps(2);
        assert_eq!(h.state().1["__web_playing"], false);
        let paused = h.state().1["__web_frame"].clone();
        h.run_steps(20);
        assert_eq!(h.state().1["__web_frame"], paused);
        h.get_by_label("Play").click();
        h.run_steps(2);
        assert_eq!(h.state().1["__web_playing"], true);
        assert_eq!(h.state().1["__web_frame"], paused, "resuming must not count time spent paused");
        h.run_steps(6);
        assert_ne!(h.state().1["__web_frame"], paused, "resuming must advance the preview again");
        let st = h.state().0.session.active().unwrap();
        assert!(Arc::ptr_eq(&st.doc, &original));
        assert_eq!(st.revision, revision);
        assert_eq!(st.history.past_len(), history);
        assert!(params(&h.state().1).as_object().unwrap().keys().all(|k| !k.starts_with("__")));
        crate::dialogs::cancel(&mut h.state_mut().0, id).unwrap();
        let next = open_web(&mut h.state_mut().0);
        let fields = &h.state_mut().0.ui.dialog_mut(next).unwrap().fields;
        assert!(!fields.contains_key("__web_playing"));
        assert!(!fields.contains_key("__web_frame"));
    }

    #[test]
    fn frame_previews_preserve_source_and_match_colors_across_depths() {
        for depth in [8, 16, 32] {
            let app = animated_app(depth);
            let source = app.session.active().unwrap().doc.clone();
            for (frame, expected) in [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]].iter().enumerate() {
                let preview = web_frame_document(&source, Some(frame)).unwrap();
                let (proxy, _, _) = web_cmds::web_document(&preview, &json!({"width":16,"height":12}), &WebSettings::default()).unwrap();
                let pixels = photocraft_compose::flatten(&proxy);
                for (actual, expected) in pixels.px[0].iter().zip(expected) {
                    assert!((actual - expected).abs() < 0.002, "depth={depth} frame={frame}");
                }
            }
            assert_eq!(source.timeline.as_ref().unwrap().current, 0);
            assert_eq!(photocraft_compose::flatten(&source).px[0], [1.0, 0.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn four_up_caches_follow_the_local_frame() {
        let mut app = animated_app(8);
        let doc_id = app.session.active().unwrap().doc.id.0;
        let id = open_web(&mut app);
        let mut fields = app.ui.dialog_mut(id).unwrap().fields.clone();
        fields.insert("__view".into(), json!("4up"));
        fields.insert("format".into(), json!("gif"));
        let mut h = harness(app, fields);
        for frame in [1, 2, 0] {
            h.state_mut().1.insert("__web_frame".into(), json!(frame));
            h.run_steps(2);
            for pane in 0usize..4 {
                let proxy: (String, WebProxy) = h.ctx.data(|d| d.get_temp(egui::Id::new(("web-proxy", doc_id, pane)))).unwrap();
                assert!(proxy.1.0[0][frame] > 0.99, "pane {pane} must show frame {frame}");
                let cache: WebCache = h.ctx.data(|d| d.get_temp(egui::Id::new(("web-pane", doc_id, pane)))).unwrap();
                assert!(cache.0.contains(&format!("Some({frame})")));
            }
        }
    }
    #[test]
    fn playback_refreshes_indexed_frames_without_resetting_palette_or_navigation() {
        let mut app = animated_app(8);
        let doc_id = app.session.active().unwrap().doc.id.0;
        let id = open_web(&mut app);
        let mut fields = app.ui.dialog_mut(id).unwrap().fields.clone();
        fields.insert("__view".into(), json!("optimized"));
        fields.insert("format".into(), json!("gif"));
        fields.insert("__zoom".into(), json!(2.0));
        let mut h = harness(app, fields);
        // Set the custom palette after the dialog has initialized its format defaults.
        let expected_palette = json!(["#ff0000", "#00ff00", "#0000ff"]);
        h.state_mut().1.insert("colorTable".into(), expected_palette.clone());
        h.run_steps(2);
        assert_eq!(h.state().1["colorTable"], expected_palette);
        let key = egui::Id::new(("web-pane", doc_id, 1usize)).with("indexed-source");
        let encoded = |ctx: &egui::Context| ctx.data(|d| d.get_temp::<(String, Arc<Optimized>)>(key)).unwrap().1;
        let first = encoded(&h.ctx);
        h.state_mut().1.insert("colorTableEdits".into(), json!({"0":"#ffff00"}));
        h.run_steps(2);
        assert!(Arc::ptr_eq(&first, &encoded(&h.ctx)), "palette edits reuse the current frame's indices");
        let palette = h.state().1["colorTable"].clone();
        let center = h.state().1["__center"].clone();
        let mut previous = first;
        for (frame, rgba) in [(1, [0, 255, 0, 255]), (2, [0, 0, 255, 255]), (0, [255, 0, 0, 255])] {
            h.state_mut().1.insert("__web_frame".into(), json!(frame));
            h.run_steps(2);
            let next = encoded(&h.ctx);
            assert!(!Arc::ptr_eq(&previous, &next), "a changed frame needs fresh indexed pixels");
            assert_eq!(&next.preview[..4], &rgba);
            assert_eq!(h.state().1["colorTable"], palette);
            assert_eq!(h.state().1["colorTableEdits"], json!({"0":"#ffff00"}));
            assert_eq!(h.state().1["__zoom"], 2.0);
            assert_eq!(h.state().1["__center"], center);
            assert_eq!(h.state().0.session.active().unwrap().doc.timeline.as_ref().unwrap().current, 0);
            previous = next;
        }
    }

    #[test]
    fn still_documents_have_no_playback_controls() {
        for duration in [None, Some(1)] {
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
            app.run("file.new", json!({"width":32,"height":24})).unwrap();
            if let Some(duration) = duration {
                app.run("timeline.create", json!({"duration":duration,"fps":2})).unwrap();
            }
            let id = open_web(&mut app);
            let fields = app.ui.dialog_mut(id).unwrap().fields.clone();
            let h = harness(app, fields);
            assert!(h.query_by_label("Play").is_none());
            assert!(!h.state().1.contains_key("__web_playing"));
        }
    }

    #[test]
    fn preview_keeps_current_frame_edits_without_storing_into_source() {
        let mut app = animated_app(8);
        app.run("edit.fill", json!({"color":"#ffff00"})).unwrap();
        let source = app.session.active().unwrap().doc.clone();
        let edited = web_frame_document(&source, Some(0)).unwrap();
        let other = web_frame_document(&source, Some(1)).unwrap();
        assert_eq!(photocraft_compose::flatten(&edited).px[0], [1.0, 1.0, 0.0, 1.0]);
        assert_eq!(photocraft_compose::flatten(&other).px[0], [0.0, 1.0, 0.0, 1.0]);
        assert!(Arc::ptr_eq(&source, &app.session.active().unwrap().doc));
        assert_eq!(source.timeline.as_ref().unwrap().current, 0);
    }

    #[test]
    fn optimized_gif_preview_matches_each_frame_at_every_depth() {
        let settings = WebSettings::from_params(&json!({"format":"gif","colors":16}), "preview").unwrap();
        for depth in [8, 16, 32] {
            let app = animated_app(depth);
            for (frame, expected) in [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]].iter().enumerate() {
                let doc = web_frame_document(&app.session.active().unwrap().doc, Some(frame)).unwrap();
                let (proxy, _, _) = web_cmds::web_document(&doc, &json!({"width":16,"height":12}), &settings).unwrap();
                let pixels = photocraft_compose::flatten(&proxy);
                let optimized = web_cmds::optimize(&pixels.px, 16, photocraft_geom::Rect::new(0, 0, 16, 12), &settings, None, None, 72.0, true).unwrap();
                assert_eq!(&optimized.preview[..4], expected, "depth={depth} frame={frame}");
            }
        }
    }
}

#[cfg(test)]
mod web_clock_tests {
    use super::*;
    #[test]
    fn frames_follow_elapsed_time_and_keep_fraction() {
        assert_eq!(advance_preview(0, 3, 2.0, 0.0, 0.49), (0, 0.0));
        assert_eq!(advance_preview(0, 3, 2.0, 0.0, 0.75), (1, 0.5));
        assert_eq!(advance_preview(1, 3, 2.0, 0.5, 1.0), (2, 1.0));
    }
    #[test]
    fn slow_frames_catch_up_and_loop() {
        assert_eq!(advance_preview(2, 3, 2.0, 0.0, 1.0), (1, 1.0));
        let (frame, last) = advance_preview(0, 3, 2.0, 0.0, 6.2);
        assert_eq!(frame, 0);
        assert!((last - 6.0).abs() < 1e-10);
    }
    #[test]
    fn resumed_clock_excludes_pause_time() {
        assert_eq!(advance_preview(1, 3, 2.0, 20.0, 20.25), (1, 20.0));
        assert_eq!(advance_preview(1, 3, 2.0, 20.0, 20.5), (2, 20.5));
    }
    #[test]
    fn fractional_frame_rates_work() {
        assert_eq!(advance_preview(0, 3, 0.5, 0.0, 2.0), (1, 2.0));
    }
    #[test]
    fn malformed_or_changed_timeline_is_bounded() {
        assert_eq!(advance_preview(usize::MAX, 0, f32::NAN, 0.0, 1.0), (0, 1.0));
        assert_eq!(advance_preview(usize::MAX - 2, usize::MAX, 2.0, 0.0, 1.0), (0, 1.0));
        assert_eq!(advance_preview(2, 3, 2.0, 10.0, 9.0), (2, 9.0));
    }
}
