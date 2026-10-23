//! Saber host adapter. One document owns text, graphics and history in both workspaces.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use base64::Engine as _;
use photocraft_doc::{ColorMode, Document, Layer, LayerContent, LayerId, PixelFormat};
use photocraft_engine::Session;
use serde_json::{Value, json};
use skrifa::{MetadataProvider, string::StringId};

const KEY: &str = "saber.page";
type Result<T> = std::result::Result<T, String>;

pub fn metadata(doc: &Document) -> Value {
    doc.metadata.text.iter().find(|(key, _)| key == KEY).and_then(|(_, value)| serde_json::from_str(value).ok()).unwrap_or_else(|| json!({}))
}

fn put_metadata(doc: &mut Document, value: &Value) {
    doc.metadata.text.retain(|(key, _)| key != KEY);
    doc.metadata.text.push((KEY.into(), value.to_string()));
}

pub fn user_mask(session: &mut Session, bytes: Option<&[u8]>) -> Result<()> {
    let mut meta = metadata(active(session)?);
    meta["userMask"] = bytes.map(|v| json!(base64::engine::general_purpose::STANDARD.encode(v))).unwrap_or(Value::Null);
    session
        .edit("手动修复范围", |doc, _| {
            put_metadata(doc, &meta);
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}

pub fn register_font(bytes: Vec<u8>) -> Result<Value> {
    let postscript = skrifa::FontRef::from_index(&bytes, 0)
        .ok()
        .and_then(|font| font.localized_strings(StringId::POSTSCRIPT_NAME).english_or_first().map(|name| name.to_string()));
    let mut engine = photocraft_text::shared().lock().unwrap_or_else(|e| e.into_inner());
    let families = engine.fonts.register_font_data(bytes);
    let face = families.iter().flat_map(|family| engine.fonts.faces(family)).find(|face| face.postscript_name == postscript).ok_or("无法读取字体字形")?;
    Ok(json!({"family":face.family,"fontStyle":face.style,"postscriptName":face.postscript_name,"weight":face.weight,"italic":face.italic}))
}

fn font_family(font: &Value) -> Option<&str> {
    font.as_str().or_else(|| font.get("family").and_then(Value::as_str))
}

pub fn set_fonts(session: &mut Session, fonts: &Value) -> Result<()> {
    let mut meta = metadata(active(session)?);
    meta["fonts"] = fonts.clone();
    session
        .edit("字体目录", |doc, _| {
            put_metadata(doc, &meta);
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}

pub fn open(session: &mut Session, bytes: &[u8], page_id: &str) -> Result<()> {
    let mut imported = photocraft_io::import("page.pcraft", bytes).map_err(|e| e.to_string())?;
    let meta = metadata(&imported.document);
    if meta.get("pageId").and_then(Value::as_str) != Some(page_id) {
        return Err("编辑文档不属于当前页面".into());
    }
    rgb_document(&mut imported.document);
    normalize_background(&mut imported.document)?;
    // Old dev projects predate visible overflow. Only native text's drawing policy changes;
    // its frame, repair geometry and already-published image assets stay intact.
    for id in imported.document.walk().iter().map(|(_, _, layer)| layer.id).collect::<Vec<_>>() {
        if let Some(Layer { content: LayerContent::Text(text), .. }) = imported.document.layer_mut(id) {
            text.overflow_visible = true;
            // Previous dev documents kept this preference in host metadata and tried to
            // implement it with first-paragraph spacing, which the type engine ignores.
            if let Some(align) = meta
                .get("bubbles")
                .and_then(Value::as_object)
                .and_then(|entries| entries.values().find(|entry| entry["layer"].as_u64() == Some(id.0)))
                .and_then(|entry| entry.pointer("/payload/blockAlign"))
                .and_then(Value::as_str)
            {
                text.block_align = match align {
                    "center" => photocraft_doc::text::BlockAlign::Center,
                    "end" => photocraft_doc::text::BlockAlign::End,
                    _ => photocraft_doc::text::BlockAlign::Start,
                };
                if text.block_align != photocraft_doc::text::BlockAlign::Start {
                    photocraft_engine::type_cmds::style_paragraphs(text, 0, 0, &|style| style.space_before_pt = 0.0);
                }
                text.cache = None;
            }
        }
    }
    session.type_overflow_visible = true;
    session.add_document(imported.document, Some("page.pcraft".into()));
    Ok(())
}

/// A host replacement is another revision of this page, not a new pixel history.
pub fn reuse_pixels(updated: &mut Document, previous: &Document) {
    for id in updated.walk().iter().map(|(_, _, layer)| layer.id).collect::<Vec<_>>() {
        let (Some(layer), Some(old)) = (updated.layer_mut(id), previous.layer(id)) else { continue };
        let surface = match &mut layer.content {
            LayerContent::Raster(surface) => Some(surface),
            LayerContent::Text(text) => text.cache.as_mut(),
            LayerContent::Shape(shape) => shape.cache.as_mut(),
            LayerContent::Smart(smart) => smart.cache.as_mut(),
            _ => None,
        };
        if let (Some(surface), Some(old)) = (surface, old.surface()) {
            surface.reuse_tiles_from(old);
        }
        if let (Some(mask), Some(old)) = (&mut layer.mask, &old.mask) {
            mask.surface.reuse_tiles_from(&old.surface);
        }
    }
    if let (Some(selection), Some(old)) = (&mut updated.selection, &previous.selection) {
        selection.reuse_tiles_from(old);
    }
}

pub fn create(session: &mut Session, source: &[u8], clean: Option<&[u8]>, page_id: &str) -> Result<()> {
    let mut doc = photocraft_io::import("source.png", source).map_err(|e| e.to_string())?.document;
    doc.resolution_dpi = 72.0;
    doc.name = "漫画页面".into();
    if let Some(bytes) = clean {
        let clean_doc = photocraft_io::import("clean.png", bytes).map_err(|e| e.to_string())?.document;
        if clean_doc.size != doc.size {
            return Err("修复图尺寸与原图不一致".into());
        }
        doc.layers.extend(clean_doc.layers);
    }
    rgb_document(&mut doc);
    let background = doc.layers.first().ok_or("页面缺少背景图层")?.id;
    let pixels = doc.clone();
    replace_background(&mut doc, background, &pixels)?;
    doc.layers.truncate(1);
    put_metadata(&mut doc, &json!({"pageId":page_id, "background":background.0, "bubbles":{}, "fonts":{}}));
    session.type_overflow_visible = true;
    session.add_document(doc, Some("page.pcraft".into()));
    Ok(())
}

// Gray source pixels keep their values; editing still needs RGB for coloured text and paint.
fn rgb_document(doc: &mut Document) {
    if !matches!(doc.mode, ColorMode::Grayscale | ColorMode::Rgb) {
        return;
    }
    let grayscale = doc.mode == ColorMode::Grayscale;
    for id in doc.walk().into_iter().map(|(_, _, layer)| layer.id).collect::<Vec<_>>() {
        if let Some(layer) = doc.layer_mut(id) {
            let surface = match &mut layer.content {
                LayerContent::Raster(surface) => Some(surface),
                LayerContent::Text(text) => text.cache.as_mut(),
                LayerContent::Shape(shape) => shape.cache.as_mut(),
                LayerContent::Smart(smart) => smart.cache.as_mut(),
                _ => None,
            };
            for surface in surface.into_iter().chain(layer.fill_cache.as_mut().map(|cache| &mut cache.surface)) {
                if grayscale && surface.format().mode != ColorMode::Grayscale {
                    *surface = surface.convert(PixelFormat { mode: ColorMode::Grayscale, ..surface.format() });
                }
                if surface.format().mode == ColorMode::Grayscale {
                    *surface = surface.convert(PixelFormat { mode: ColorMode::Rgb, ..surface.format() });
                }
            }
        }
    }
    if doc.mode == ColorMode::Grayscale {
        doc.mode = ColorMode::Rgb;
        doc.icc_profile = None;
    }
}

fn active(session: &Session) -> Result<&Document> {
    session.active().map(|st| st.doc.as_ref()).ok_or_else(|| "没有打开的页面".into())
}

fn number(v: &Value, key: &str, fallback: f64) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap_or(fallback)
}

// Text, character runs and block alignment live only in native layers.
// Disabled stroke preferences still need to survive here.
fn text_entry(bubble: &Value) -> Value {
    let mut payload = serde_json::Map::new();
    for key in ["strokeColor", "strokeWidth"] {
        if let Some(value) = bubble.pointer(&format!("/payload/{key}")) {
            payload.insert(key.into(), value.clone());
        }
    }
    json!({"bubbleId":bubble["bubbleId"],"fontId":bubble["fontId"],"layer":bubble["layer"],"payload":payload})
}

/// Existing bubble mutations are translated into engine operations, never a second text store.
pub fn set_bubbles(session: &mut Session, bubbles: &[Value], fonts: &Value) -> Result<()> {
    let before = session.active().cloned().ok_or("没有打开的页面")?;
    if let Err(error) = set_bubbles_inner(session, bubbles, fonts) {
        if let Some(st) = session.active_mut() {
            *st = before;
        }
        return Err(error);
    }
    let after = session.active().cloned().ok_or("没有打开的页面")?;
    if let Some(st) = session.active_mut() {
        *st = before;
    }
    session
        .edit("修改文字", move |doc, active| {
            *doc = after.doc.as_ref().clone();
            *active = after.active_layer;
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn set_bubbles_inner(session: &mut Session, bubbles: &[Value], fonts: &Value) -> Result<()> {
    let mut meta = metadata(active(session)?);
    // Compare with the current native text, including edits made in the fine
    // workspace since the last save. Metadata supplies identity and preferences.
    let previous: serde_json::Map<String, Value> = project(session)?["bubbles"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|bubble| bubble["bubbleId"].as_str().map(|id| (id.to_owned(), bubble.clone())))
        .collect();
    let mut entries = serde_json::Map::new();
    for bubble in bubbles {
        let id = bubble.get("bubbleId").and_then(Value::as_str).ok_or("缺少文字对象 ID")?;
        let payload = bubble.get("payload").ok_or("缺少文字数据")?;
        let prior = previous.get(id);
        let existing =
            prior.and_then(|v| v.get("layer")).and_then(Value::as_u64).filter(|id| active(session).ok().and_then(|d| d.layer(LayerId(*id))).is_some());
        let coords = payload.get("coords").and_then(Value::as_array).filter(|v| v.len() == 4).ok_or("无效的文本框")?;
        let at = |i: usize| coords.get(i).and_then(Value::as_f64).unwrap_or(0.0);
        let x = at(0) + payload.pointer("/position/x").and_then(Value::as_f64).unwrap_or(0.0);
        let y = at(1) + payload.pointer("/position/y").and_then(Value::as_f64).unwrap_or(0.0);
        let w = (at(2) - at(0)).max(1.0);
        let h = (at(3) - at(1)).max(1.0);
        let font_id = bubble.get("fontId").and_then(Value::as_str).unwrap_or("");
        let font = fonts.get(font_id).or_else(|| fonts.get("__default")).unwrap_or(&Value::Null);
        let family = font_family(font).unwrap_or("Inter");
        let text = payload.get("translatedText").and_then(Value::as_str).unwrap_or("");
        let layer = if let Some(layer) = existing {
            layer
        } else {
            session
                .execute(
                    "type.create",
                    json!({"text":text,"box":[x,y,w,h],"font":family,
                "size":number(payload,"fontSize",24.0),"color":payload.get("textColor").cloned().unwrap_or(json!("#000000")),
                "orientation":payload.get("textDirection").cloned().unwrap_or(json!("horizontal")),
                "name":format!("译文 {}", entries.len()+1)}),
                )
                .map_err(|e| e.to_string())?
                .get("layer")
                .and_then(Value::as_u64)
                .ok_or("无法建立文字层")?
        };
        let old = prior.and_then(|v| v.get("payload")).unwrap_or(&Value::Null);
        let changed = |key: &str| {
            bubble.get("fields").and_then(Value::as_array).is_some_and(|fields| fields.iter().any(|field| field.as_str() == Some(key)))
                || old.get(key) != payload.get(key)
        };
        if changed("translatedText") && existing.is_some() {
            // Reuse the engine's range edit so unchanged characters keep their styles.
            let before = old.get("translatedText").and_then(Value::as_str).unwrap_or("");
            let prefix = before.chars().zip(text.chars()).take_while(|(a, b)| a == b).count();
            let (old_len, new_len) = (before.chars().count(), text.chars().count());
            let suffix = before.chars().rev().zip(text.chars().rev()).take(old_len.min(new_len) - prefix).take_while(|(a, b)| a == b).count();
            let inserted: String = text.chars().skip(prefix).take(new_len - prefix - suffix).collect();
            session.execute("type.edit", json!({"layer":layer,"replace":{"start":prefix,"end":old_len-suffix,"text":inserted}})).map_err(|e| e.to_string())?;
        }
        let rich = payload.get("richText").filter(|v| !v.is_null());
        if (rich != old.get("richText").filter(|v| !v.is_null()) || existing.is_none())
            && let Some(rich) = rich
        {
            let runs = serde_json::from_value(rich.get("runs").cloned().unwrap_or(json!([]))).map_err(|e| format!("无效字符样式：{e}"))?;
            let paragraphs = serde_json::from_value(rich.get("paragraphs").cloned().unwrap_or(json!([]))).map_err(|e| format!("无效段落样式：{e}"))?;
            session
                .edit("字符样式", |doc, _| {
                    if let Some(Layer { content: LayerContent::Text(t), .. }) = doc.layer_mut(LayerId(layer)) {
                        t.runs = runs;
                        t.paragraphs = paragraphs;
                        t.sync_summary();
                        t.cache = None;
                    }
                    Ok(())
                })
                .map_err(|e| e.to_string())?;
        }
        let mut style = json!({"layer":layer});
        for (field, param) in [("fontSize", "size"), ("textColor", "color")] {
            if changed(field) && (existing.is_some() || rich.is_none()) {
                style[param] = payload.get(field).cloned().unwrap_or(Value::Null);
            }
        }
        let font_changed = bubble.get("fields").and_then(Value::as_array).is_some_and(|fields| fields.iter().any(|field| field.as_str() == Some("fontId")))
            || prior.and_then(|v| v.get("fontId")) != bubble.get("fontId");
        if font_changed && (existing.is_some() || rich.is_none()) {
            style["font"] = json!(family);
            for key in ["fontStyle", "postscriptName", "weight", "italic"] {
                if let Some(value) = font.get(key) {
                    style[key] = value.clone();
                }
            }
        }
        if style.as_object().is_some_and(|s| s.len() > 1) {
            session.execute("type.setStyle", style).map_err(|e| e.to_string())?;
        }
        let geometry_changed = ["coords", "position", "rotationAngle"].iter().any(|key| changed(key)) || existing.is_none();
        if geometry_changed {
            let doc = active(session)?;
            let Some(Layer { content: LayerContent::Text(t), .. }) = doc.layer(LayerId(layer)) else { return Err("文字层不存在".into()) };
            let bounds = text_bounds(t, doc.resolution_dpi);
            let mut m = t.transform.m;
            let prior_coords = old.get("coords").and_then(Value::as_array);
            let prior_at = |i: usize, fallback: f64| prior_coords.and_then(|c| c.get(i)).and_then(Value::as_f64).unwrap_or(fallback);
            let (ox, oy) = (prior_at(0, x), prior_at(1, y));
            let (ow, oh) = ((prior_at(2, x + w) - ox).max(1.0), (prior_at(3, y + h) - oy).max(1.0));
            let resized = w != ow || h != oh;
            let angle = (number(payload, "rotationAngle", 0.0) - m[1].atan2(m[0]).to_degrees()).to_radians();
            let (sin, cos) = angle.sin_cos();
            let (a, b, c, d) = (m[0], m[1], m[2], m[3]);
            m[0] = cos * a - sin * b;
            m[1] = sin * a + cos * b;
            m[2] = cos * c - sin * d;
            m[3] = sin * c + cos * d;
            let sx = m[0].hypot(m[1]).max(0.001);
            let sy = m[2].hypot(m[3]).max(0.001);
            let mut params = json!({"layer":layer});
            let (mut lx, mut ly) = ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0);
            let original = t.transform.m;
            let cx = original[4] + original[0] * lx + original[2] * ly + x - ox + (w - ow) / 2.0;
            let cy = original[5] + original[1] * lx + original[3] * ly + y - oy + (h - oh) / 2.0;
            if resized && matches!(t.shape, photocraft_doc::text::TextShape::Box { .. }) {
                // Keep the native affine (including skew/scale). The ordinary panel only
                // changes the paragraph dimensions, angle and centre it exposes.
                let (nw, nh) = (w / sx, h / sy);
                params["box"] = json!([0, 0, nw, nh]);
                (lx, ly) = (nw / 2.0, nh / 2.0);
            } else if resized {
                m[0] *= w / ow;
                m[1] *= w / ow;
                m[2] *= h / oh;
                m[3] *= h / oh;
            }
            m[4] = cx - m[0] * lx - m[2] * ly;
            m[5] = cy - m[1] * lx - m[3] * ly;
            if !resized && angle == 0.0 {
                params["move"] = json!([x - ox, y - oy]);
            } else {
                params["transform"] = json!(m);
            }
            session.execute("type.edit", params).map_err(|e| e.to_string())?;
        }
        if changed("textDirection") || existing.is_none() {
            let command = if payload.get("textDirection").and_then(Value::as_str) == Some("vertical") {
                "type.orientation.vertical"
            } else {
                "type.orientation.horizontal"
            };
            session.execute(command, json!({"layer":layer})).map_err(|e| e.to_string())?;
        }
        if changed("inlineAlign") && (existing.is_some() || rich.is_none()) {
            let align = match payload.get("inlineAlign").and_then(Value::as_str) {
                Some("start") => "left",
                Some("end") => "right",
                _ => "center",
            };
            session.execute("type.setStyle", json!({"layer":layer,"align":align})).map_err(|e| e.to_string())?;
        }
        if (changed("lineSpacing") || changed("fontSize")) && (existing.is_some() || rich.is_none()) {
            let spacing = number(payload, "lineSpacing", 1.0) as f32;
            session
                .edit("行间距", |doc, _| {
                    if let Some(Layer { content: LayerContent::Text(t), .. }) = doc.layer_mut(LayerId(layer)) {
                        t.runs = t.char_runs();
                        for run in &mut t.runs {
                            run.style.leading_pt = Some(run.style.size_pt * spacing);
                        }
                        t.cache = None;
                    }
                    Ok(())
                })
                .map_err(|e| e.to_string())?;
        }
        if changed("blockAlign") || changed("richText") || existing.is_none() {
            session
                .edit("文本块位置", |doc, _| {
                    if let Some(Layer { content: LayerContent::Text(t), .. }) = doc.layer_mut(LayerId(layer)) {
                        t.block_align = match payload.get("blockAlign").and_then(Value::as_str) {
                            Some("center") => photocraft_doc::text::BlockAlign::Center,
                            Some("end") => photocraft_doc::text::BlockAlign::End,
                            _ => photocraft_doc::text::BlockAlign::Start,
                        };
                        if t.block_align != photocraft_doc::text::BlockAlign::Start {
                            photocraft_engine::type_cmds::style_paragraphs(t, 0, 0, &|style| style.space_before_pt = 0.0);
                        }
                        t.cache = None;
                    }
                    Ok(())
                })
                .map_err(|e| e.to_string())?;
        }
        if ["strokeEnabled", "strokeColor", "strokeWidth"].iter().any(|key| changed(key)) {
            session.execute("layer.select", json!({"layer":layer})).map_err(|e| e.to_string())?;
            let indices = active(session)?
                .layer(LayerId(layer))
                .map(|l| {
                    l.effects.items.iter().enumerate().filter_map(|(i, e)| matches!(e, photocraft_doc::Effect::Stroke(_)).then_some(i)).collect::<Vec<_>>()
                })
                .unwrap_or_default();
            for i in indices.into_iter().rev() {
                session.execute("layer.layerStyle.clear", json!({"layer":layer,"effect":i})).map_err(|e| e.to_string())?;
            }
            if payload.get("strokeEnabled").and_then(Value::as_bool) == Some(true) {
                session
                    .execute(
                        "layer.layerStyle.stroke",
                        json!({"layer":layer,"size":number(payload,"strokeWidth",1.0),
                    "color":payload.get("strokeColor"),"position":"outside"}),
                    )
                    .map_err(|e| e.to_string())?;
            }
        }
        let mut entry = bubble.clone();
        entry["layer"] = json!(layer);
        entries.insert(id.into(), text_entry(&entry));
    }
    {
        for (id, entry) in &previous {
            if !entries.contains_key(id)
                && let Some(layer) = entry.get("layer").and_then(Value::as_u64)
            {
                session.execute("layer.delete", json!({"layer":layer})).map_err(|e| e.to_string())?;
            }
        }
    }
    meta["bubbles"] = Value::Object(entries);
    meta["fonts"] = fonts.clone();
    session
        .edit("页面数据", |doc, _| {
            put_metadata(doc, &meta);
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    refresh(session);
    Ok(())
}

pub fn refresh(session: &mut Session) {
    let layers = session
        .active()
        .map(|st| st.doc.walk().into_iter().filter_map(|(_, _, l)| matches!(l.content, LayerContent::Text(_)).then_some((st.doc.id, l.id))).collect::<Vec<_>>())
        .unwrap_or_default();
    session.refresh_type_layers(&layers);
}

fn text_bounds(t: &photocraft_doc::TextLayer, dpi: f32) -> [f64; 4] {
    if let photocraft_doc::text::TextShape::Box { x, y, width, height } = t.shape {
        return [x as f64, y as f64, (x + width) as f64, (y + height) as f64];
    }
    photocraft_text::shared().lock().unwrap_or_else(|e| e.into_inner()).layout(t, dpi).bounds().unwrap_or([0.0, 0.0, 1.0, 1.0]).map(f64::from)
}

fn text_id(page: &str, layer: u64) -> String {
    let hash = blake3::hash(format!("{page}:{layer}").as_bytes()).to_hex().to_string();
    format!("{}-{}-4{}-a{}-{}", &hash[..8], &hash[8..12], &hash[13..16], &hash[17..20], &hash[20..32])
}

/// Read models for the existing APIs are generated from the authoritative text layers.
pub fn project(session: &Session) -> Result<Value> {
    let doc = active(session)?;
    let meta = metadata(doc);
    let page = meta.get("pageId").and_then(Value::as_str).unwrap_or("");
    let entries = meta.get("bubbles").and_then(Value::as_object);
    let fonts = meta.get("fonts").and_then(Value::as_object);
    let mut bubbles = Vec::new();
    for (_, _, layer) in doc.walk() {
        let LayerContent::Text(t) = &layer.content else { continue };
        let entry = entries.and_then(|e| e.values().find(|v| v.get("layer").and_then(Value::as_u64) == Some(layer.id.0)));
        let id = entry.and_then(|e| e.get("bubbleId")).and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| text_id(page, layer.id.0));
        let mut bubble = entry.map(text_entry).unwrap_or_else(|| json!({"bubbleId":id,"fontId":fonts.and_then(|f|f.get("__defaultId")),"payload":{}}));
        let p = &mut bubble["payload"];
        p["translatedText"] = json!(t.text);
        p["blockAlign"] = json!(t.block_align);
        let runs = t.char_runs();
        let style = runs.first().map(|r| &r.style);
        if let Some(style) = style {
            p["fontSize"] = json!(style.size_pt.round().max(1.0) as u32);
            let spacing = style
                .leading_pt
                .map(|leading| leading / style.size_pt.max(0.001))
                .unwrap_or_else(|| t.paragraph_runs().first().map(|p| p.style.auto_leading).unwrap_or(1.0));
            p["lineSpacing"] = json!((spacing.max(0.01) as f64 * 1_000_000.0).round() / 1_000_000.0);
            let c = style.color.to_rgba8();
            p["textColor"] = json!(format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2]));
            // Files can share a family: match the actual face, keeping old
            // family-only metadata readable until the next ordinary edit.
            let matches = |font: &Value| {
                font_family(font) == Some(&style.font_family)
                    && (font.is_string() || font.get("postscriptName").and_then(Value::as_str) == style.postscript_name.as_deref())
            };
            let current = bubble.get("fontId").and_then(Value::as_str).and_then(|id| fonts.and_then(|f| f.get(id)));
            if !current.is_some_and(matches)
                && let Some((font, _)) = fonts.and_then(|f| f.iter().find(|(k, v)| !k.starts_with("__") && matches(v)))
            {
                bubble["fontId"] = json!(font);
            }
        }
        let p = &mut bubble["payload"];
        let paragraphs = t.paragraph_runs();
        if let Some(paragraph) = paragraphs.first() {
            p["inlineAlign"] = json!(match paragraph.style.align {
                photocraft_doc::text::TextAlign::Left => "start",
                photocraft_doc::text::TextAlign::Right => "end",
                _ => "center",
            });
        }
        if let Some(photocraft_doc::Effect::Stroke(stroke)) = layer.effects.items.iter().find(|e| matches!(e, photocraft_doc::Effect::Stroke(_))) {
            p["strokeEnabled"] = json!(layer.effects.enabled && stroke.common.enabled);
            p["strokeWidth"] = json!((stroke.size as f64 * 1_000_000.0).round() / 1_000_000.0);
            if let photocraft_doc::FxPaint::Color(color) = stroke.paint {
                let c = color.to_rgba8();
                p["strokeColor"] = json!(format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2]));
            }
        } else {
            p["strokeEnabled"] = json!(false);
        }
        p["richText"] = json!({"runs":runs,"paragraphs":t.paragraph_runs()});
        let bounds = text_bounds(t, doc.resolution_dpi);
        let m = t.transform.m;
        let (lx, ly) = ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0);
        let (w, h) = ((bounds[2] - bounds[0]) * m[0].hypot(m[1]), (bounds[3] - bounds[1]) * m[2].hypot(m[3]));
        let center = (m[4] + m[0] * lx + m[2] * ly, m[5] + m[1] * lx + m[3] * ly);
        let x = (center.0 - w / 2.0).round() as i64;
        let y = (center.1 - h / 2.0).round() as i64;
        p["coords"] = json!([x, y, x + (w.round() as i64).max(1), y + (h.round() as i64).max(1)]);
        p["position"] = json!({"x":0,"y":0});
        p["rotationAngle"] = json!(m[1].atan2(m[0]).to_degrees());
        p["textDirection"] = json!(if matches!(t.orientation, photocraft_doc::text::Orientation::Vertical) { "vertical" } else { "horizontal" });
        bubble["layer"] = json!(layer.id.0);
        bubbles.push(bubble);
    }
    Ok(json!({"pageId":page,"width":doc.size.width,"height":doc.size.height,"bubbles":bubbles,"userMask":meta.get("userMask")}))
}

pub fn snapshot(session: &Session) -> Result<Vec<u8>> {
    let mut doc = active(session)?.clone();
    let mut meta = metadata(&doc);
    let projection = project(session)?;
    let mut entries = serde_json::Map::new();
    for bubble in projection.get("bubbles").and_then(Value::as_array).into_iter().flatten() {
        if let Some(id) = bubble.get("bubbleId").and_then(Value::as_str) {
            entries.insert(id.into(), text_entry(bubble));
        }
    }
    meta["bubbles"] = Value::Object(entries);
    put_metadata(&mut doc, &meta);
    photocraft_format::PcraftWriter::default().save_zip(&doc, &Default::default()).map(|(bytes, _)| bytes).map_err(|e| e.to_string())
}

pub fn png(session: &Session, background: bool) -> Result<Vec<u8>> {
    let mut doc = active(session)?.clone();
    if background {
        let ids = doc.walk().into_iter().filter_map(|(_, _, l)| matches!(l.content, LayerContent::Text(_)).then_some(l.id)).collect::<Vec<_>>();
        for id in ids {
            if let Some(layer) = doc.layer_mut(id) {
                layer.visible = false;
            }
        }
    }
    photocraft_io::export(&doc, "page.png", &Default::default()).map(|r| r.bytes).map_err(|e| e.to_string())
}

// Only the host-owned background is flattened; other layers stay editable.
fn normalize_background(doc: &mut Document) -> Result<()> {
    let id = metadata(doc)["background"].as_u64().map(LayerId).ok_or("页面缺少背景图层")?;
    let layer = doc.layer(id).ok_or("页面背景图层不存在")?;
    if !matches!(layer.content, LayerContent::Raster(_)) {
        let mut solo = doc.clone();
        solo.layers = vec![layer.clone()];
        replace_background(doc, id, &solo)?;
    }
    let layer = doc.layer_mut(id).ok_or("页面背景图层不存在")?;
    layer.name = "修复背景".into();
    // Standard Background locks let Auto-Select start its layer marquee on the backdrop,
    // instead of falling through to a previously selected text layer.
    layer.locks.all = false;
    layer.locks.position = true;
    layer.locks.transparency = true;
    Ok(())
}

fn replace_background(doc: &mut Document, id: LayerId, pixels: &Document) -> Result<()> {
    let mut surface = photocraft_compose::flatten_to_surface(pixels, doc.pixel_format(), None);
    if let Some(previous) = doc.layer(id).and_then(Layer::surface) {
        surface.reuse_tiles_from(previous);
    }
    let mut layer = Layer::new("修复背景", LayerContent::Raster(surface));
    layer.id = id;
    layer.locks.position = true;
    layer.locks.transparency = true;
    *doc.layer_mut(id).ok_or("页面背景图层不存在")? = layer;
    Ok(())
}

/// Commit a repair into the fixed background, retaining the engine's existing undo history.
pub fn repair(session: &mut Session, bytes: &[u8]) -> Result<()> {
    let mut patch = photocraft_io::import("repair.png", bytes).map_err(|e| e.to_string())?.document;
    rgb_document(&mut patch);
    if patch.size != active(session)?.size {
        return Err("修复区域尺寸与原图不一致".into());
    }
    let background = metadata(active(session)?)["background"].as_u64().map(LayerId).ok_or("页面缺少背景图层")?;
    session
        .edit("背景修复", move |doc, _| {
            let layer = doc.layer(background).ok_or_else(|| photocraft_engine::EngineError::Other("页面背景图层不存在".into()))?;
            let mut solo = doc.clone();
            solo.layers = vec![layer.clone()];
            solo.layers.extend(patch.layers);
            replace_background(doc, background, &solo).map_err(photocraft_engine::EngineError::Other)?;
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}

pub fn text_png(session: &Session) -> Result<Vec<u8>> {
    let mut doc = active(session)?.clone();
    let ids = doc
        .walk()
        .into_iter()
        .filter_map(|(_, _, l)| (!matches!(l.content, LayerContent::Text(_) | LayerContent::Group(_))).then_some(l.id))
        .collect::<Vec<_>>();
    for id in ids {
        if let Some(layer) = doc.layer_mut(id) {
            layer.visible = false;
        }
    }
    photocraft_io::export(&doc, "text.png", &Default::default()).map(|r| r.bytes).map_err(|e| e.to_string())
}

pub fn selection_png(session: &Session) -> Result<Vec<u8>> {
    let doc = active(session)?;
    let selection = doc.selection.as_ref().ok_or("请先使用选框、套索或魔棒选择要修复的区域")?;
    let length = (doc.size.width as usize).checked_mul(doc.size.height as usize).ok_or("选区尺寸过大")?;
    let mut pixels = Vec::with_capacity(length);
    for y in 0..doc.size.height {
        for x in 0..doc.size.width {
            pixels.push(if selection.sample_channel(x as i32, y as i32, 0) >= 0.5 { 255 } else { 0 });
        }
    }
    if !pixels.contains(&255) {
        return Err("当前选区为空".into());
    }
    let image =
        photocraft_codecs::Image::from_u8(doc.size.width, doc.size.height, photocraft_codecs::ChannelLayout::Gray, pixels).map_err(|e| e.to_string())?;
    photocraft_codecs::encode(&image, photocraft_codecs::Format::Png, &Default::default()).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> Session {
        let mut session = Session::new();
        create(&mut session, &image_png([0, 0, 0, 0]), None, "p").unwrap();
        session
    }

    fn image_png(rgba: [u8; 4]) -> Vec<u8> {
        let image = photocraft_codecs::Image::from_u8(200, 200, photocraft_codecs::ChannelLayout::Rgba, rgba.repeat(200 * 200)).unwrap();
        photocraft_codecs::encode(&image, photocraft_codecs::Format::Png, &Default::default()).unwrap()
    }
    fn bubble() -> Value {
        json!({"bubbleId":"text-1","fontId":"font-1","payload":{
            "translatedText":"A中😀B","coords":[10,10,150,150],"fontSize":18,"textDirection":"horizontal",
            "rotationAngle":0,"position":{"x":0,"y":0},"textColor":"#000000",
            "strokeEnabled":false,"strokeWidth":2,"strokeColor":"#ffffff","lineSpacing":1,"inlineAlign":"start","blockAlign":"start"
        }})
    }
    fn fonts() -> Value {
        json!({"font-1":"Inter"})
    }

    #[test]
    fn ordinary_reflow_keeps_block_alignment_in_local_text_space() {
        for direction in ["horizontal", "vertical"] {
            for align in ["center", "end"] {
                for field in ["fontSize", "lineSpacing", "coords", "textDirection", "translatedText"] {
                    let mut session = session();
                    let mut initial = bubble();
                    initial["payload"]["coords"] = json!([10, 10, 170, 150]);
                    initial["payload"]["translatedText"] = json!("AB\nCD");
                    initial["payload"]["textDirection"] = json!(direction);
                    initial["payload"]["blockAlign"] = json!(align);
                    set_bubbles(&mut session, &[initial], &fonts()).unwrap();
                    let layer = project(&session).unwrap()["bubbles"][0]["layer"].as_u64().unwrap();
                    session.execute("type.edit", json!({"layer":layer,"transform":[1.5,0,0,1.5,10,10]})).unwrap();
                    let mut changed = project(&session).unwrap()["bubbles"][0].clone();
                    changed["payload"][field] = match field {
                        "fontSize" => json!(30),
                        "lineSpacing" => json!(1.8),
                        "coords" => json!([10, 10, 280, 250]),
                        "textDirection" => json!(if direction == "horizontal" { "vertical" } else { "horizontal" }),
                        _ => json!("XY\nZZ\nEF"),
                    };
                    set_bubbles(&mut session, &[changed], &fonts()).unwrap();
                    let doc = active(&session).unwrap();
                    let LayerContent::Text(text) = &doc.layer(LayerId(layer)).unwrap().content else { panic!("text") };
                    let layout = photocraft_text::shared().lock().unwrap_or_else(|e| e.into_inner()).layout(text, doc.resolution_dpi);
                    let bounds = layout.line_bounds().unwrap();
                    let photocraft_doc::text::TextShape::Box { x, y, width, height } = text.shape else { panic!("box") };
                    let extent = if layout.vertical { width } else { height };
                    let origin = if layout.vertical { -(x + width) } else { y };
                    let expected = origin + (extent - (bounds[3] - bounds[1])).max(0.0) * if align == "center" { 0.5 } else { 1.0 };
                    assert!((bounds[1] - expected).abs() < 0.001, "{direction}/{align}/{field}: actual lines must align");
                    assert!(
                        text.paragraph_runs().iter().all(|p| p.style.space_before_pt == 0.0),
                        "{direction}/{align}/{field}: block alignment must not become paragraph spacing"
                    );
                }
            }
        }
    }

    #[test]
    fn host_replacements_share_pixels_and_keep_undo_redo() {
        let mut session = session();
        set_bubbles(&mut session, &[bubble()], &fonts()).unwrap();
        session
            .edit("pixels", |doc, _| {
                for id in doc.walk().iter().map(|(_, _, layer)| layer.id).collect::<Vec<_>>() {
                    if let Some(surface) = doc.layer_mut(id).and_then(Layer::surface_mut) {
                        surface.write_pixel(10, 10, &[1.0, 0.0, 0.0, 1.0]);
                    }
                }
                Ok(())
            })
            .unwrap();
        session.active_mut().unwrap().history.clear();
        let before = project(&session).unwrap();
        for index in 0..60 {
            let bytes = snapshot(&session).unwrap();
            let mut imported = Session::new();
            open(&mut imported, &bytes, "p").unwrap();
            let mut updated = imported.active().unwrap().doc.as_ref().clone();
            let mut meta = metadata(&updated);
            meta["hostRevision"] = json!(index);
            put_metadata(&mut updated, &meta);
            session
                .edit("host replace", move |doc, _| {
                    reuse_pixels(&mut updated, doc);
                    *doc = updated;
                    Ok(())
                })
                .unwrap();
            let st = session.active().unwrap();
            assert_eq!(st.history.unique_bytes(&st.doc), 0, "unchanged source pixels must not accumulate");
        }
        session.execute("edit.undo", json!({})).unwrap();
        assert_eq!(metadata(&session.active().unwrap().doc)["hostRevision"], 58);
        session.execute("edit.redo", json!({})).unwrap();
        assert_eq!(metadata(&session.active().unwrap().doc)["hostRevision"], 59);
        assert_eq!(project(&session).unwrap(), before);
    }

    #[test]
    fn unicode_runs_survive_native_roundtrip_and_ordinary_edits() {
        let mut session = session();
        set_bubbles(&mut session, &[bubble()], &fonts()).unwrap();
        let layer = project(&session).unwrap()["bubbles"][0]["layer"].as_u64().unwrap();
        session.execute("type.setStyle", json!({"layer":layer,"range":[1,3],"size":36,"color":"#ff0000"})).unwrap();
        let before = project(&session).unwrap();
        let bytes = snapshot(&session).unwrap();
        let mut reopened = Session::new();
        open(&mut reopened, &bytes, "p").unwrap();
        assert_eq!(before, project(&reopened).unwrap());
        let mut values = before["bubbles"].as_array().unwrap().clone();
        values[0]["payload"]["coords"] = json!([20, 20, 160, 160]);
        set_bubbles(&mut reopened, &values, &fonts()).unwrap();
        let moved = project(&reopened).unwrap();
        assert_eq!(before["bubbles"][0]["payload"]["richText"], moved["bubbles"][0]["payload"]["richText"]);
        values = moved["bubbles"].as_array().unwrap().clone();
        values[0]["payload"]["fontSize"] = json!(24);
        set_bubbles(&mut reopened, &values, &fonts()).unwrap();
        let after = project(&reopened).unwrap();
        let runs = after["bubbles"][0]["payload"]["richText"]["runs"].as_array().unwrap();
        assert_eq!(runs.iter().map(|r| r["len"].as_u64().unwrap()).sum::<u64>(), 9);
        assert!(runs.iter().all(|r| r["style"]["size_pt"] == 24.0));
        assert_eq!(after["bubbles"][0]["payload"]["lineSpacing"], moved["bubbles"][0]["payload"]["lineSpacing"]);
        assert!(runs.iter().any(|r| r["style"]["color"] != runs[0]["style"]["color"]));
        reopened.execute("edit.undo", json!({})).unwrap();
        assert_eq!(moved, project(&reopened).unwrap(), "an ordinary action is one history step");
    }

    #[test]
    fn failed_mutation_keeps_the_entire_document() {
        let mut session = session();
        set_bubbles(&mut session, &[bubble()], &fonts()).unwrap();
        let before = project(&session).unwrap();
        let mut invalid = bubble();
        invalid["payload"]["coords"] = json!([0]);
        assert!(set_bubbles(&mut session, &[invalid], &fonts()).is_err());
        assert_eq!(before, project(&session).unwrap());
        let bytes = snapshot(&session).unwrap();
        assert!(open(&mut session, &bytes, "another-page").is_err());
    }

    #[test]
    fn explicit_box_style_updates_every_character_when_the_summary_already_matches() {
        let mut session = session();
        set_bubbles(&mut session, &[bubble()], &fonts()).unwrap();
        let layer = project(&session).unwrap()["bubbles"][0]["layer"].as_u64().unwrap();
        session.execute("type.setStyle", json!({"layer":layer,"range":[1,3],"size":36,"color":"#ff0000"})).unwrap();
        let mut values = project(&session).unwrap()["bubbles"].as_array().unwrap().clone();
        assert_eq!(values[0]["payload"]["fontSize"], 18);
        values[0]["fields"] = json!(["fontSize"]);
        set_bubbles(&mut session, &values, &fonts()).unwrap();
        let projected = project(&session).unwrap();
        let runs = projected["bubbles"][0]["payload"]["richText"]["runs"].as_array().unwrap();
        assert!(runs.iter().all(|run| run["style"]["size_pt"] == 18.0));
        assert!(runs.iter().any(|run| run["style"]["color"] != runs[0]["style"]["color"]));
    }

    #[test]
    fn snapshots_keep_text_in_native_layers_and_not_in_host_metadata() {
        let mut session = session();
        let mut value = bubble();
        value["payload"]["originalText"] = json!("database business data");
        set_bubbles(&mut session, &[value], &fonts()).unwrap();
        let bytes = snapshot(&session).unwrap();
        let mut reopened = Session::new();
        open(&mut reopened, &bytes, "p").unwrap();
        let meta = metadata(active(&reopened).unwrap());
        let payload = &meta["bubbles"]["text-1"]["payload"];
        assert!(payload.get("translatedText").is_none());
        assert!(payload.get("richText").is_none());
        assert!(payload.get("blockAlign").is_none());
        assert!(payload.get("originalText").is_none());
        assert_eq!(project(&reopened).unwrap()["bubbles"][0]["payload"]["translatedText"], "A中😀B");
    }

    #[test]
    fn old_host_block_alignment_moves_into_native_text_without_moving_its_frame() {
        let mut session = session();
        set_bubbles(&mut session, &[bubble()], &fonts()).unwrap();
        let mut doc = active(&session).unwrap().clone();
        let mut meta = metadata(&doc);
        let layer = LayerId(meta["bubbles"]["text-1"]["layer"].as_u64().unwrap());
        meta["bubbles"]["text-1"]["payload"]["blockAlign"] = json!("center");
        put_metadata(&mut doc, &meta);
        let LayerContent::Text(text) = &mut doc.layer_mut(layer).unwrap().content else { panic!("text") };
        let frame = (text.shape.clone(), text.transform);
        photocraft_engine::type_cmds::style_paragraphs(text, 0, 0, &|style| style.space_before_pt = 42.0);
        let (bytes, _) = photocraft_format::PcraftWriter::default().save_zip(&doc, &Default::default()).unwrap();
        let mut reopened = Session::new();
        open(&mut reopened, &bytes, "p").unwrap();
        let LayerContent::Text(text) = &active(&reopened).unwrap().layer(layer).unwrap().content else { panic!("text") };
        assert_eq!((text.shape.clone(), text.transform), frame);
        assert_eq!(text.block_align, photocraft_doc::text::BlockAlign::Center);
        assert_eq!(text.paragraph_runs()[0].style.space_before_pt, 0.0);
        let saved = snapshot(&reopened).unwrap();
        let mut final_session = Session::new();
        open(&mut final_session, &saved, "p").unwrap();
        assert_eq!(project(&final_session).unwrap()["bubbles"][0]["payload"]["blockAlign"], "center");
        assert!(metadata(active(&final_session).unwrap())["bubbles"]["text-1"]["payload"].get("blockAlign").is_none());
    }

    #[test]
    fn projection_keeps_selected_font_file_when_families_overlap() {
        let fonts = json!({"font-a":"Inter","font-z":"Inter","font-other":"Other"});
        for selected in ["font-a", "font-z"] {
            let mut session = session();
            let mut value = bubble();
            value["fontId"] = json!(selected);
            set_bubbles(&mut session, &[value], &fonts).unwrap();
            let projected = project(&session).unwrap();
            assert_eq!(projected["bubbles"][0]["fontId"], selected);
            let bytes = snapshot(&session).unwrap();
            let mut reopened = Session::new();
            open(&mut reopened, &bytes, "p").unwrap();
            let mut values = projected["bubbles"].as_array().unwrap().clone();
            values[0]["payload"]["strokeWidth"] = json!(3);
            set_bubbles(&mut reopened, &values, &fonts).unwrap();
            assert_eq!(project(&reopened).unwrap()["bubbles"][0]["fontId"], selected);
            let layer = projected["bubbles"][0]["layer"].as_u64().unwrap();
            reopened.execute("type.setStyle", json!({"layer":layer,"font":"Other"})).unwrap();
            assert_eq!(project(&reopened).unwrap()["bubbles"][0]["fontId"], "font-other");
        }
    }

    #[test]
    fn font_file_selection_sets_the_exact_face_and_projects_native_changes() {
        let regular = register_font(photocraft_text::fonts::INTER_REGULAR.to_vec()).unwrap();
        let bold = register_font(photocraft_text::fonts::INTER_SEMIBOLD.to_vec()).unwrap();
        assert_eq!(regular["postscriptName"], "Inter-Regular");
        assert_eq!(bold["postscriptName"], "Inter-SemiBold");
        assert_eq!(bold["weight"], 600.0);
        assert!(register_font(vec![0; 16]).is_err());
        let fonts = json!({"regular":regular,"bold":bold});
        let mut session = session();
        let mut value = bubble();
        value["fontId"] = json!("regular");
        set_bubbles(&mut session, &[value], &fonts).unwrap();
        let mut values = project(&session).unwrap()["bubbles"].as_array().unwrap().clone();
        values[0]["fontId"] = json!("bold");
        set_bubbles(&mut session, &values, &fonts).unwrap();
        let projected = project(&session).unwrap();
        assert_eq!(projected["bubbles"][0]["fontId"], "bold");
        let style = &projected["bubbles"][0]["payload"]["richText"]["runs"][0]["style"];
        assert_eq!(style["postscript_name"], "Inter-SemiBold");
        assert_eq!(style["weight"], 600);
        let layer = projected["bubbles"][0]["layer"].as_u64().unwrap();
        session.execute("type.setStyle", json!({"layer":layer,"weight":400})).unwrap();
        assert_eq!(project(&session).unwrap()["bubbles"][0]["fontId"], "regular");
        let bytes = snapshot(&session).unwrap();
        let mut reopened = Session::new();
        open(&mut reopened, &bytes, "p").unwrap();
        assert_eq!(project(&reopened).unwrap()["bubbles"][0]["fontId"], "regular");
    }

    #[test]
    fn background_repair_stays_below_grouped_type_layers() {
        let mut session = session();
        set_bubbles(&mut session, &[bubble()], &fonts()).unwrap();
        session
            .edit("文字组", |doc, _| {
                let text = std::mem::take(&mut doc.layers);
                doc.layers.push(Layer::group("译文", text));
                Ok(())
            })
            .unwrap();
        let before = png(&session, false).unwrap();
        let patch = photocraft_codecs::Image::from_u8(200, 200, photocraft_codecs::ChannelLayout::Rgba, vec![255; 200 * 200 * 4]).unwrap();
        let patch_png = photocraft_codecs::encode(&patch, photocraft_codecs::Format::Png, &Default::default()).unwrap();
        repair(&mut session, &patch_png).unwrap();
        let doc = active(&session).unwrap();
        assert!(matches!(doc.layers.last().unwrap().content, LayerContent::Group(_)));
        let background = LayerId(metadata(doc)["background"].as_u64().unwrap());
        assert!(matches!(doc.layer(background).unwrap().content, LayerContent::Raster(_)));
        assert!(doc.layer(background).unwrap().locks.position);
        session.execute("edit.undo", json!({})).unwrap();
        assert_eq!(active(&session).unwrap().layers.len(), 1);
        assert_eq!(png(&session, false).unwrap(), before);
    }

    #[test]
    fn repeated_repairs_update_one_background_with_undo_and_keep_other_layers() {
        let mut session = Session::new();
        create(&mut session, &image_png([30, 60, 90, 255]), None, "p").unwrap();
        set_bubbles(&mut session, &[bubble()], &fonts()).unwrap();
        let text = project(&session).unwrap()["bubbles"].clone();
        let mut user = photocraft_io::import("draw.png", &image_png([0, 0, 0, 0])).unwrap().document.layers.remove(0);
        user.name = "手绘".into();
        let user_id = user.id;
        session
            .edit("手绘图层", |doc, _| {
                doc.layers.push(user);
                Ok(())
            })
            .unwrap();
        let background = LayerId(metadata(active(&session).unwrap())["background"].as_u64().unwrap());
        let mut pixels = [0_u8; 4].repeat(200 * 200);
        for y in 150..170 {
            for x in 150..170 {
                pixels[(y * 200 + x) * 4..(y * 200 + x + 1) * 4].copy_from_slice(&[200, 100, 50, 255]);
            }
        }
        let image = photocraft_codecs::Image::from_u8(200, 200, photocraft_codecs::ChannelLayout::Rgba, pixels).unwrap();
        let patch = photocraft_codecs::encode(&image, photocraft_codecs::Format::Png, &Default::default()).unwrap();
        for bytes in [&patch, &image_png([30, 60, 90, 255]), &patch] {
            let before = png(&session, true).unwrap();
            repair(&mut session, bytes).unwrap();
            let after = png(&session, true).unwrap();
            let doc = active(&session).unwrap();
            assert_eq!(doc.layers.len(), 3, "repairs must not create user-visible layers");
            assert_eq!(doc.layer(user_id).unwrap().name, "手绘");
            assert_eq!(project(&session).unwrap()["bubbles"], text);
            assert!(doc.layer(background).unwrap().locks.position);
            let surface = doc.layer(background).unwrap().surface().unwrap();
            assert!((surface.sample_channel(1, 1, 0) - 30.0 / 255.0).abs() < 0.001, "pixels outside a local patch stay unchanged");
            session.execute("edit.undo", json!({})).unwrap();
            assert_eq!(png(&session, true).unwrap(), before);
            session.execute("edit.redo", json!({})).unwrap();
            assert_eq!(png(&session, true).unwrap(), after);
        }
        assert!(session.execute("layer.translate", json!({"layer":background.0,"dx":20,"dy":10})).is_err());
        let picked = session.execute("layer.pickAt", json!({"x":1,"y":1,"target":"group"})).unwrap();
        assert_eq!(picked["layer"], json!(background.0), "the standard Background is picked but cannot move");
        repair(&mut session, &image_png([120, 130, 140, 255])).unwrap();
        assert_eq!(active(&session).unwrap().layers.len(), 3);
        let surface = active(&session).unwrap().layer(background).unwrap().surface().unwrap();
        assert!((surface.sample_channel(160, 160, 0) - 120.0 / 255.0).abs() < 0.001, "a whole-page result replaces previous repair pixels");
    }

    #[test]
    fn legacy_background_group_opens_as_one_locked_layer_without_changing_pixels_or_text() {
        let mut session = session();
        set_bubbles(&mut session, &[bubble()], &fonts()).unwrap();
        let background = LayerId(metadata(active(&session).unwrap())["background"].as_u64().unwrap());
        session
            .edit("旧背景", |doc, _| {
                let mut source = photocraft_io::import("source.png", &image_png([30, 60, 90, 255])).unwrap().document.layers.remove(0);
                source.name = "原图".into();
                source.locks.all = true;
                let mut patch = photocraft_io::import("patch.png", &image_png([100, 110, 120, 255])).unwrap().document.layers.remove(0);
                patch.name = "背景修复".into();
                let mut group = Layer::group("背景", vec![source, patch]);
                group.id = background;
                *doc.layer_mut(background).unwrap() = group;
                Ok(())
            })
            .unwrap();
        let before = png(&session, false).unwrap();
        let text = project(&session).unwrap()["bubbles"].clone();
        let mut reopened = Session::new();
        open(&mut reopened, &snapshot(&session).unwrap(), "p").unwrap();
        assert_eq!(png(&reopened, false).unwrap(), before);
        assert_eq!(project(&reopened).unwrap()["bubbles"], text);
        assert_eq!(active(&reopened).unwrap().layers.len(), 2);
        let base = active(&reopened).unwrap().layer(background).unwrap();
        assert!(matches!(base.content, LayerContent::Raster(_)));
        assert!(base.locks.position && base.locks.transparency && !base.locks.all);
        assert!(reopened.execute("layer.translate", json!({"layer":background.0,"dx":5})).is_err());
    }

    #[test]
    fn ordinary_move_keeps_fine_affine_shape_and_unsaved_character_styles() {
        for point in [false, true] {
            let mut session = session();
            set_bubbles(&mut session, &[bubble()], &fonts()).unwrap();
            let layer = project(&session).unwrap()["bubbles"][0]["layer"].as_u64().unwrap();
            if point {
                session.execute("type.edit", json!({"layer":layer,"point":[10,20]})).unwrap();
            }
            session.execute("type.edit", json!({"layer":layer,"transform":[1.234,0.0,0.3,1.573,10,20]})).unwrap();
            session.execute("type.setStyle", json!({"layer":layer,"range":[0,1],"size":36,"color":"#ff0000"})).unwrap();
            let before = project(&session).unwrap();
            let mut values = before["bubbles"].as_array().unwrap().clone();
            let coords = values[0]["payload"]["coords"].as_array_mut().unwrap();
            for (i, value) in coords.iter_mut().enumerate() {
                *value = json!(value.as_i64().unwrap() + if i % 2 == 0 { 8 } else { 12 });
            }
            set_bubbles(&mut session, &values, &fonts()).unwrap();
            let after = project(&session).unwrap();
            assert_eq!(before["bubbles"][0]["payload"]["richText"], after["bubbles"][0]["payload"]["richText"]);
            let Some(Layer { content: LayerContent::Text(t), .. }) = session.active().unwrap().doc.layer(LayerId(layer)) else { panic!("text missing") };
            assert_eq!(&t.transform.m[..4], &[1.234, 0.0, 0.3, 1.573]);
            assert_eq!(matches!(t.shape, photocraft_doc::text::TextShape::Point), point);
            if !point {
                assert!(matches!(t.shape, photocraft_doc::text::TextShape::Box { width, height, .. } if width == 140.0 && height == 140.0));
            }
        }
    }
}
