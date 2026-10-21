//! Same-origin host bridge. The engine owns document state; the host owns persistence.
use photocraft_ui_egui::PhotocraftApp;
use photocraft_ui_egui::theme::{self, ThemeKind, Tokens};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use wasm_bindgen::{JsCast as _, JsValue, closure::Closure};

#[derive(Default)]
pub struct Bridge {
    incoming: Arc<Mutex<Vec<(Value, Vec<u8>)>>>,
    page: String,
    revision: Option<u64>,
    theme: Option<Tokens>,
}

pub fn emit(value: Value, bytes: Option<&[u8]>) {
    let Some(window) = web_sys::window() else { return };
    let Ok(Some(parent)) = window.parent() else { return };
    let Ok(origin) = window.location().origin() else { return };
    let Ok(object) = js_sys::JSON::parse(&value.to_string()) else { return };
    let _ = js_sys::Reflect::set(&object, &JsValue::from_str("saberEditor"), &JsValue::TRUE);
    if let Some(bytes) = bytes {
        let _ = js_sys::Reflect::set(&object, &JsValue::from_str("bytes"), &js_sys::Uint8Array::from(bytes));
    }
    let _ = parent.post_message(&object, &origin);
}

impl Bridge {
    pub fn apply_theme(&self, app: &mut PhotocraftApp, ctx: &egui::Context) {
        if let Some(tokens) = self.theme
            && Tokens::get(ctx) != tokens
        {
            app.ui.theme = tokens.kind;
            theme::apply_tokens(ctx, tokens);
        }
    }

    pub fn listen(ctx: egui::Context) -> Self {
        let this = Self::default();
        let Some(window) = web_sys::window() else { return this };
        let Ok(Some(parent)) = window.parent() else { return this };
        let Ok(origin) = window.location().origin() else { return this };
        let queue = this.incoming.clone();
        let callback = Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |event: web_sys::MessageEvent| {
            if event.origin() != origin || event.source().is_none_or(|s| !js_sys::Object::is(&s, &parent)) {
                return;
            }
            let data = event.data();
            let get = |key: &str| js_sys::Reflect::get(&data, &JsValue::from_str(key)).ok();
            if get("saberEditor").and_then(|v| v.as_bool()) != Some(true) {
                return;
            }
            let Some(payload) = get("payload").and_then(|v| js_sys::JSON::stringify(&v).ok()).and_then(|v| v.as_string()) else { return };
            let Ok(value) = serde_json::from_str::<Value>(&payload) else { return };
            let bytes = get("bytes").filter(|v| !v.is_undefined()).map(|v| js_sys::Uint8Array::new(&v).to_vec()).unwrap_or_default();
            queue.lock().unwrap_or_else(|e| e.into_inner()).push((value, bytes));
            ctx.request_repaint();
        });
        if window.add_event_listener_with_callback("message", callback.as_ref().unchecked_ref()).is_ok() {
            callback.forget();
        }
        emit(json!({"event":"ready"}), None);
        this
    }

    pub fn tick(&mut self, app: &mut PhotocraftApp, ctx: &egui::Context) {
        let requests = std::mem::take(&mut *self.incoming.lock().unwrap_or_else(|e| e.into_inner()));
        for (request, bytes) in requests {
            let result = self.handle(app, ctx, &request, &bytes);
            if let Err(error) = result {
                emit(json!({"requestId":request.get("requestId"),"error":error}), None);
            }
        }
        if let Some(st) = app.session.active() {
            if self.revision != Some(st.revision.saturating_sub(st.view_revision)) && !self.page.is_empty() {
                self.revision = Some(st.revision.saturating_sub(st.view_revision));
                emit(
                    json!({"event":"changed","pageId":self.page,"revision":st.revision.saturating_sub(st.view_revision),
                    "canUndo":st.history.can_undo(),"canRedo":st.history.can_redo()}),
                    None,
                );
            }
        }
    }

    fn handle(&mut self, app: &mut PhotocraftApp, ctx: &egui::Context, request: &Value, bytes: &[u8]) -> Result<(), String> {
        let id = request.get("requestId").cloned().unwrap_or(Value::Null);
        let mut response = json!({"requestId":id});
        match request.get("op").and_then(Value::as_str).unwrap_or("") {
            "theme" => {
                let color = |key: &str| -> Result<egui::Color32, String> {
                    let value = request.get("colors").and_then(|c| c.get(key)).and_then(Value::as_str).ok_or_else(|| format!("缺少主题颜色：{key}"))?;
                    let [r, g, b] = photocraft_engine::prefs::parse_hex(value).ok_or_else(|| format!("无效主题颜色：{key}"))?;
                    Ok(egui::Color32::from_rgb(r, g, b))
                };
                let mut t = Tokens::for_kind(ThemeKind::Pro);
                t.chrome = color("surface")?;
                t.card = t.chrome;
                t.tab_strip = t.chrome;
                t.canvas = color("canvas")?;
                t.canvas_dot = t.canvas;
                t.dock = color("background")?;
                t.field = color("field")?;
                t.histogram_bg = t.field;
                t.field_border = t.field;
                t.card_border = t.field_border;
                t.separator = t.field_border;
                t.hover = t.field;
                t.pressed = t.hover;
                t.text = color("text")?;
                t.text_dim = t.text.gamma_multiply(0.75);
                t.text_faint = t.text_dim;
                t.icon = t.text_dim;
                t.accent = color("accent")?;
                t.accent_border = t.accent;
                t.accent_soft = t.accent;
                t.accent_text = t.text;
                t.primary_bg = t.accent;
                t.primary_text = t.accent_text;
                t.row_selected = egui::Color32::from_rgba_unmultiplied(t.accent.r(), t.accent.g(), t.accent.b(), 60);
                t.danger = color("danger")?;
                t.warning = color("warning")?;
                if let Some(radius) = request.get("radius").and_then(Value::as_f64).filter(|r| r.is_finite() && (0.0..=24.0).contains(r)) {
                    t.radius = radius as f32;
                    t.radius_sm = (t.radius - 2.0).max(0.0);
                }
                if let Some(radius) = request.get("radiusLarge").and_then(Value::as_f64).filter(|r| r.is_finite() && (0.0..=32.0).contains(r)) {
                    t.radius_lg = radius as f32;
                }
                self.theme = Some(t);
                self.apply_theme(app, ctx);
            }
            "font" => {
                let face = photocraft_saber::register_font(bytes.to_vec())?;
                if request.get("ui").and_then(Value::as_bool) == Some(true) {
                    let mut definitions = ctx.fonts(|fonts| fonts.definitions().clone());
                    definitions.font_data.insert("saber-cjk".into(), Arc::new(egui::FontData::from_owned(bytes.to_vec())));
                    for families in definitions.families.values_mut() {
                        families.push("saber-cjk".into());
                    }
                    ctx.set_fonts(definitions);
                }
                response["font"] = face;
            }
            "open" => {
                let page = request.get("pageId").and_then(Value::as_str).ok_or("pageId required")?;
                // Validate before closing the previous page, so a failed load keeps the work.
                let mut replacement = photocraft_engine::Session::new();
                photocraft_saber::open(&mut replacement, bytes, page)?;
                photocraft_saber::refresh(&mut replacement);
                while !app.session.documents().is_empty() {
                    app.session.close(0);
                }
                // Separate files can reuse a persisted ID and revision; release the closed page's caches first.
                app.sync_views();
                // Reuse the engine's history budget; desktop defaults exceed a browser's address space.
                app.session.edit_prefs(|prefs| prefs.performance.memory_usage_mb = 256);
                if let Some(st) = replacement.active() {
                    app.session.add_document(st.doc.as_ref().clone(), Some("page.pcraft".into()));
                }
                app.session.type_overflow_visible = true;
                self.page = page.into();
                if let Some(st) = app.session.active_mut() {
                    st.saved_revision = st.revision;
                    self.revision = Some(st.revision.saturating_sub(st.view_revision));
                }
                app.ui.views.clear();
                app.ui.tool = photocraft_ui_egui::state::Tool::Move;
                if let Some(st) = app.session.active() {
                    let meta = photocraft_saber::metadata(&st.doc);
                    if let Some(family) = meta.pointer("/fonts/__default").and_then(|font| font.as_str().or_else(|| font.get("family").and_then(Value::as_str)))
                    {
                        app.ui.tool_options.type_font = family.into();
                    }
                    if let Some(text) = st
                        .doc
                        .walk()
                        .into_iter()
                        .find_map(|(_, _, layer)| if let photocraft_doc::LayerContent::Text(text) = &layer.content { Some(text) } else { None })
                    {
                        app.ui.tool_options.type_font = text.font_family.clone();
                        app.ui.tool_options.type_size = text.size_pt;
                    }
                }
            }
            "command" => {
                let command = request.get("command").and_then(Value::as_str).ok_or("command required")?;
                if command.starts_with("file.") {
                    return Err("请使用页面工具栏保存或导出".into());
                }
                response["result"] = app.run(command, request.get("params").cloned().unwrap_or(json!({})))?;
            }
            "snapshot" => {
                response["revision"] = app.session.active().map(|st| json!(st.revision.saturating_sub(st.view_revision))).unwrap_or(Value::Null);
                let bytes = photocraft_saber::snapshot(&app.session)?;
                emit(response, Some(&bytes));
                return Ok(());
            }
            "finish" => photocraft_ui_egui::type_tool::commit(app),
            "selection" => {
                let bytes = photocraft_saber::selection_png(&app.session)?;
                emit(response, Some(&bytes));
                return Ok(());
            }
            "replace" => {
                let mut replacement = photocraft_engine::Session::new();
                photocraft_saber::open(&mut replacement, bytes, &self.page)?;
                photocraft_saber::refresh(&mut replacement);
                let mut updated = replacement.active().ok_or("没有打开的文档")?.doc.as_ref().clone();
                photocraft_ui_egui::type_tool::commit(app);
                app.session
                    .edit(request.get("label").and_then(Value::as_str).unwrap_or("背景修复"), move |doc, _| {
                        let id = doc.id;
                        photocraft_saber::reuse_pixels(&mut updated, doc);
                        *doc = updated;
                        doc.id = id;
                        Ok(())
                    })
                    .map_err(|e| e.to_string())?;
            }
            "saved" => {
                if let (Some(st), Some(revision)) = (app.session.active_mut(), request.get("revision").and_then(Value::as_u64)) {
                    // The host acknowledges document edits. View-only changes made
                    // while the save was in flight must not leave a false dirty flag.
                    st.saved_revision = revision.saturating_add(st.view_revision);
                }
            }
            "presentation" => {
                let fine = request.get("fine").and_then(Value::as_bool) == Some(true);
                app.ui.panels.toolbar = fine;
                app.ui.panels.dock = fine;
                app.ui.panels.menu_bar = fine;
                app.ui.panels.options_bar = fine;
                app.ui.panels.status_bar = fine;
            }
            _ => return Err("未知编辑操作".into()),
        }
        response["revision"] = app.session.active().map(|st| json!(st.revision.saturating_sub(st.view_revision))).unwrap_or(Value::Null);
        if let Some(st) = app.session.active() {
            response["canUndo"] = json!(st.history.can_undo());
            response["canRedo"] = json!(st.history.can_redo());
        }
        emit(response, None);
        ctx.request_repaint();
        Ok(())
    }
}
