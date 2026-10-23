//! The browser shell: web `Services`, drag-and-drop, and the eframe web runner.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use photocraft_codecs::{ChannelLayout, EncodeOptions, Image};
use photocraft_doc::Document;
use photocraft_ui_egui::i18n;
use photocraft_ui_egui::theme::ThemeKind;
use photocraft_ui_egui::{FileDialogAnswer, FileDialogRequest, PhotocraftApp, Services};
use wasm_bindgen::JsCast as _;

type Inbox = Arc<Mutex<Vec<(String, Vec<u8>)>>>;

/// Everything File › Open reads: PhotoCraft, Photoshop, OpenRaster, Krita and Affinity documents, flat images, and
/// Photoshop brushes (.abr), gradients (.grd) and swatches (.aco, .ase), which go to the preset libraries.
const OPEN_EXTS: &[&str] = &[
    "pcraft", "pdn", "ora", "kra", "krz", "psd", "psb", "psdt", "png", "jpg", "jpeg", "tif", "tiff", "webp", "gif", "bmp", "tga", "ico", "qoi", "exr", "hdr",
    "pbm", "pgm", "ppm", "pam", "pfm", "dds", "fits", "fit", "fts", "dpx", "heic", "heif", "hif", "dng", "cr2", "cr3", "nef", "nrw", "arw", "pef", "orf",
    "rw2", "raf", "abr", "grd", "svg", "svgz", "aco", "ase", "af", "afdesign", "afphoto", "afpub",
];
const SVG_EXTS: &[&str] = &["svg", "svgz"];
const CANVAS_ID: &str = "photocraft_canvas";

pub fn start() {
    eframe::WebLogger::init(log::LevelFilter::Info).ok();
    wasm_bindgen_futures::spawn_local(async {
        let Some(document) = web_sys::window().and_then(|w| w.document()) else {
            log::error!("no document");
            return;
        };
        let Some(canvas) = document.get_element_by_id(CANVAS_ID).and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok()) else {
            log::error!("missing <canvas id=\"{CANVAS_ID}\">");
            return;
        };
        let q = query();
        let force_cpu = q.contains("cpu");
        let mut options = eframe::WebOptions::default();
        photocraft_ui_egui::gpu_canvas::use_adapter_limits(&mut options.wgpu_options.wgpu_setup);
        if q.contains("webgl")
            && let eframe::egui_wgpu::WgpuSetup::CreateNew(create) = &mut options.wgpu_options.wgpu_setup
        {
            create.instance_descriptor.backends = eframe::wgpu::Backends::GL;
        }
        // Hydrate before constructing an interactive session: a late load must never undo a
        // user deletion/reorder. Storage denial or timeout still opens a usable session.
        let (database, presets, mut preset_warnings) = match crate::indexed_presets::load().await {
            Ok((db, bridge, warnings)) => (Some(db), bridge, warnings),
            Err(e) => {
                let bridge = crate::preset_bridge::Bridge::default();
                bridge.unavailable(e);
                (None, bridge, Vec::new())
            }
        };
        let pen_target = canvas.clone();
        let result = eframe::WebRunner::new()
            .start(
                canvas,
                options,
                Box::new(move |cc| {
                    cc.egui_ctx.options_mut(|options| options.sync_window_theme = false);
                    PhotocraftApp::setup_context(&cc.egui_ctx, ThemeKind::Pro);
                    let inbox: Inbox = Arc::default();
                    let (session, warnings) = crate::preset_bridge::session(presets.clone());
                    preset_warnings.extend(warnings);
                    let mut app = PhotocraftApp::new(session, services(inbox.clone()));
                    app.session.edit_prefs(|prefs| prefs.interface.language = "zh-hans".into());
                    if !preset_warnings.is_empty() {
                        photocraft_ui_egui::notices::post(&mut app, i18n::t("Some brush presets could not be loaded"), preset_warnings, true, None);
                    }
                    listen_pen(&pen_target, app.stylus.feed.clone());
                    app.set_theme(&cc.egui_ctx, ThemeKind::Pro);
                    if let Some(rs) = cc.wgpu_render_state.clone()
                        && !force_cpu
                    {
                        log::info!("photocraft-web: wgpu backend {:?}", rs.adapter.get_info().backend);
                        app.set_wgpu(rs);
                    }
                    let unsaved = Arc::new(AtomicBool::new(false));
                    guard_unload(unsaved.clone());
                    let host = crate::saber_bridge::Bridge::listen(cc.egui_ctx.clone());
                    app.services.host_save = Some(Box::new(|| crate::saber_bridge::emit(serde_json::json!({"event":"save"}), None)));
                    Ok(Box::new(WebShell { app, inbox, unsaved, database, presets, host }))
                }),
            )
            .await;
        let loading = document.get_element_by_id("photocraft_loading");
        match result {
            Ok(()) => {
                if let Some(el) = loading {
                    el.remove();
                }
            }
            Err(e) => {
                let error = format!("精细编辑启动失败：{e:?}。浏览器需要支持 WebGPU 或 WebGL2。");
                if let Some(el) = loading {
                    el.set_text_content(Some(&error));
                }
                crate::saber_bridge::emit(serde_json::json!({"event":"error","error":error}), None);
            }
        }
    });
}

/// Pen pressure, tilt, twist and the eraser button from Pointer Events (eframe forwards none of them for pens) into
/// the app's stylus feed. The sample is kept through `pointerup` so the stroke's last points keep
/// their pressure; hovering, a mouse, or leaving the canvas clears it.
fn listen_pen(target: &web_sys::HtmlCanvasElement, feed: photocraft_ui_egui::stylus::StylusFeed) {
    use photocraft_ui_egui::stylus::PenSample;
    use wasm_bindgen::closure::Closure;
    for kind in ["pointerdown", "pointermove", "pointerup", "pointercancel", "pointerleave"] {
        let feed = feed.clone();
        let cb = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(move |e: web_sys::PointerEvent| {
            let ty = e.type_();
            if ty == "pointerup" && e.pointer_type() == "pen" {
                return;
            }
            let pen = e.pointer_type() == "pen" && e.buttons() != 0 && ty != "pointercancel" && ty != "pointerleave";
            // W3C Pointer Events: `buttons` bit 5 (32) is the pen's eraser.
            let eraser = e.buttons() & 32 != 0;
            feed.set(pen.then(|| PenSample {
                pressure: e.pressure(),
                tilt_x: e.tilt_x() as f32,
                tilt_y: e.tilt_y() as f32,
                rotation: e.twist() as f32,
                eraser,
            }));
        });
        if target.add_event_listener_with_callback(kind, cb.as_ref().unchecked_ref()).is_ok() {
            cb.forget();
        }
    }
}

/// Closing or reloading the tab while a document has unsaved changes asks first, as closing the
/// desktop window does: the browser shows its own "Leave site?" prompt (#1380). `unsaved` is
/// refreshed every frame by [`WebShell`].
fn guard_unload(unsaved: Arc<AtomicBool>) {
    use wasm_bindgen::closure::Closure;
    let Some(window) = web_sys::window() else { return };
    let cb = Closure::<dyn FnMut(web_sys::BeforeUnloadEvent)>::new(move |e: web_sys::BeforeUnloadEvent| {
        if unsaved.load(Ordering::Relaxed) {
            e.prevent_default();
            // Older browsers show the prompt only when a return value is set.
            e.set_return_value("");
        }
    });
    if window.add_event_listener_with_callback("beforeunload", cb.as_ref().unchecked_ref()).is_ok() {
        cb.forget();
    }
}

fn query() -> String {
    web_sys::window().and_then(|w| w.location().search().ok()).unwrap_or_default()
}

/// Wraps the app to read dropped files asynchronously (browsers can't read them synchronously,
/// so the app's own drop path can't handle them) and feed them through the inbox.
struct WebShell {
    host: crate::saber_bridge::Bridge,
    app: PhotocraftApp,
    inbox: Inbox,
    /// Read by the `beforeunload` listener ([`guard_unload`]).
    unsaved: Arc<AtomicBool>,
    database: Option<web_sys::IdbDatabase>,
    presets: crate::preset_bridge::Bridge,
}

impl WebShell {
    fn save_presets(&mut self, ctx: &egui::Context) {
        if let Some(db) = self.database.clone()
            && let Some(batch) = self.presets.begin()
        {
            let presets = self.presets.clone();
            let wake = ctx.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let result = crate::indexed_presets::write(&db, &batch).await;
                if let Err(e) = &result {
                    log::error!("Could not save browser presets: {e}");
                }
                presets.finish(batch, result);
                wake.request_repaint();
            });
        }
        self.unsaved.store(self.app.has_unsaved_work() || self.presets.unsaved(), Ordering::Relaxed);
    }
}

impl eframe::App for WebShell {
    fn logic(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.host.tick(&mut self.app, ctx);
        let dropped = ctx.input_mut(|i| std::mem::take(&mut i.raw.dropped_files));
        for f in dropped {
            let inbox = self.inbox.clone();
            let ctx = ctx.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let name = f.path().file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "dropped".into());
                match f.bytes_async().await {
                    Ok(bytes) => {
                        inbox.lock().unwrap_or_else(|e| e.into_inner()).push((name, bytes));
                        ctx.request_repaint();
                    }
                    Err(e) => log::error!("couldn't read dropped file {name}: {e}"),
                }
            });
        }
        self.app.logic(ctx, frame);
        self.host.apply_theme(&mut self.app, ctx);
        self.save_presets(ctx);
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        // Only while a preset change is unsaved. Storage the browser denies from the start (a
        // sandboxed iframe, blocked site data) loses nothing until the first change, and that
        // change brings the banner with the reason.
        if self.presets.unsaved()
            && let Some(error) = self.presets.error()
        {
            egui::Panel::top("browser_preset_storage").show(ui, |ui| {
                let tokens = photocraft_ui_egui::theme::Tokens::for_kind(self.app.ui.theme);
                // Put retry first so floating brush windows cannot cover the control.
                ui.horizontal_wrapped(|ui| {
                    if self.database.is_some() && ui.button(i18n::t("Retry saving presets")).clicked() {
                        self.presets.retry();
                        ui.ctx().request_repaint();
                    }
                    ui.colored_label(tokens.warning, i18n::fmt(i18n::t("Brush preset changes are not saved: {error}"), &[("error", &error)]));
                });
                ui.label(if self.database.is_some() {
                    i18n::t("Keep this tab open to retain your brushes.")
                } else {
                    i18n::t("Browser storage is unavailable; brushes are session-only. Keep this tab open to retain them.")
                });
            });
        }
        self.app.ui(ui, frame);
        self.save_presets(ui.ctx());
    }
}

fn services(inbox: Inbox) -> Services {
    Services {
        screen_pick: screen_color_service(),
        import: Some(Box::new(|name: &str, bytes: &[u8], max_svg_group_depth: usize| {
            photocraft_io::import_with_svg_group_depth(name, bytes, max_svg_group_depth).map(|r| (r.document, r.warnings)).map_err(|e| e.to_string())
        })),
        export: Some(Box::new(|doc: &Document, path: &str, settings: &photocraft_ui_egui::ExportSettings| {
            let mut opts = photocraft_io::ExportOptions::default();
            if let Some(q) = settings.jpeg_quality {
                opts.encode.jpeg_quality = q;
            }
            opts.encode.webp_lossless = settings.webp_lossless;
            if let Some(q) = settings.webp_quality {
                opts.encode.webp_quality = q;
            }
            opts.tiff_layers = settings.tiff_layers;
            opts.xmp = if settings.xmp_all { photocraft_io::XmpEmbed::All } else { photocraft_io::XmpEmbed::None };
            opts.tga_bits = settings.tga_bits;
            opts.image_previews = settings.image_previews;
            opts.psd_maximize_compatibility = settings.psd_maximize_compatibility;
            photocraft_io::export(doc, path, &opts).map(|r| (r.bytes, r.warnings)).map_err(|e| e.to_string())
        })),
        file_dialog: Some(Box::new(|request, _parent, reply| match request {
            // The browser's file picker hands over the files' contents, not paths. File › Open
            // asks for several files (`multiple`), each opened in its own tab.
            FileDialogRequest::Open { multiple, extensions, .. } => wasm_bindgen_futures::spawn_local(async move {
                let dialog = if let Some(exts) = extensions {
                    rfd::AsyncFileDialog::new().add_filter("Supported Files", &exts)
                } else {
                    rfd::AsyncFileDialog::new().add_filter("All Formats", OPEN_EXTS).add_filter("SVG", SVG_EXTS)
                };
                let answer = if multiple {
                    match dialog.pick_files().await {
                        Some(picked) if !picked.is_empty() => {
                            let mut files = Vec::with_capacity(picked.len());
                            for file in picked {
                                files.push((file.file_name(), file.read().await));
                            }
                            Some(FileDialogAnswer::ContentsMany(files))
                        }
                        _ => None,
                    }
                } else {
                    match dialog.pick_file().await {
                        Some(file) => Some(FileDialogAnswer::Contents(file.file_name(), file.read().await)),
                        None => None,
                    }
                };
                reply.send(answer);
            }),
            // Browser folder handles cannot be represented as filesystem paths for engine commands.
            FileDialogRequest::Folder { .. } => reply.send(None),
            // No save dialog on the web: the suggested name becomes the download name.
            FileDialogRequest::Save { suggested } => {
                let name = std::path::Path::new(&suggested).file_name().map_or_else(|| suggested.clone(), |n| n.to_string_lossy().to_string());
                reply.send(Some(FileDialogAnswer::SaveTo(name)));
            }
        })),
        write: Some(Box::new(|path: &str, bytes: &[u8]| {
            let name = std::path::Path::new(path).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "image".into());
            crate::saber_bridge::emit(serde_json::json!({"event":"download","filename":name}), Some(bytes));
            Ok(())
        })),
        encode_png: Some(Box::new(|w, h, rgba| {
            let img = Image::from_u8(w, h, ChannelLayout::Rgba, rgba.to_vec()).map_err(|e| e.to_string())?;
            photocraft_codecs::encode(&img, photocraft_codecs::Format::Png, &EncodeOptions::default()).map_err(|e| e.to_string())
        })),
        inbox: Some(inbox),
        // Preferences live in the browser's localStorage.
        load_prefs: Some(Box::new(|| local_storage()?.get_item(PREFS_KEY).ok().flatten())),
        save_prefs: Some(Box::new(|text: &str| local_storage().ok_or("no localStorage")?.set_item(PREFS_KEY, text).map_err(|e| format!("{e:?}")))),
        ..Default::default()
    }
}

const PREFS_KEY: &str = "photocraft.preferences";

fn local_storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok().flatten()
}

/// Rust bindings to the browser's user-activated EyeDropper API; unsupported browsers keep
/// the document picker and its normal canvas zoom. The browser owns its screen magnifier.
fn screen_color_service() -> Option<photocraft_ui_egui::screen_picker::Service> {
    use js_sys::{Function, Reflect};
    use photocraft_ui_egui::screen_picker::{Capture, Pending};
    use wasm_bindgen::{JsValue, closure::Closure};
    let window = web_sys::window()?;
    let constructor = Reflect::get(&window, &JsValue::from_str("EyeDropper")).ok()?.dyn_into::<Function>().ok()?;
    Some(Box::new(move |ctx| {
        let (tx, rx) = std::sync::mpsc::channel();
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let started = (|| -> Result<_, String> {
            let eye = Reflect::construct(&constructor, &js_sys::Array::new()).map_err(|e| format!("Could not start browser eyedropper: {e:?}"))?;
            let controller = web_sys::AbortController::new().map_err(|e| format!("Could not create eyedropper cancellation: {e:?}"))?;
            let options = js_sys::Object::new();
            Reflect::set(&options, &JsValue::from_str("signal"), &controller.signal()).map_err(|e| format!("Could not set eyedropper options: {e:?}"))?;
            let open = Reflect::get(&eye, &JsValue::from_str("open"))
                .map_err(|e| format!("Browser eyedropper has no open method: {e:?}"))?
                .dyn_into::<Function>()
                .map_err(|_| "Invalid browser eyedropper method")?;
            let promise = open
                .call1(&eye, &options)
                .map_err(|e| format!("Could not open browser eyedropper: {e:?}"))?
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| "Browser eyedropper returned no promise")?;
            let stop = cancelled.clone();
            let abort = controller.clone();
            let timer = Closure::<dyn FnMut()>::new(move || {
                if stop.load(std::sync::atomic::Ordering::Relaxed) {
                    abort.abort();
                }
            });
            let interval = window.set_interval_with_callback_and_timeout_and_arguments_0(timer.as_ref().unchecked_ref(), 50).map_err(|e| {
                controller.abort();
                format!("Could not watch eyedropper cancellation: {e:?}")
            })?;
            Ok((promise, timer, interval))
        })();
        match started {
            Err(e) => {
                let _ = tx.send(Err(e));
            }
            Ok((promise, timer, interval)) => {
                let wake = ctx.clone();
                let window = window.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    let result = match wasm_bindgen_futures::JsFuture::from(promise).await {
                        Ok(value) => Reflect::get(&value, &JsValue::from_str("sRGBHex"))
                            .ok()
                            .and_then(|v| v.as_string())
                            .and_then(|h| photocraft_ui_egui::color_picker_ui::parse_hex(&h))
                            .map(|rgb| Capture::Color(Some(rgb)))
                            .ok_or_else(|| "Browser eyedropper returned an invalid color".into()),
                        Err(e) if Reflect::get(&e, &JsValue::from_str("name")).ok().and_then(|v| v.as_string()).as_deref() == Some("AbortError") => {
                            Ok(Capture::Color(None))
                        }
                        Err(e) => Err(format!("Browser screen color selection failed: {e:?}")),
                    };
                    window.clear_interval_with_handle(interval);
                    drop(timer);
                    let _ = tx.send(result);
                    wake.request_repaint();
                });
            }
        }
        Pending { receiver: rx, cancelled }
    }))
}
