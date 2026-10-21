//! Save for Web navigation and indexed palette UI. Draft state is serialized in dialog fields.
use super::*;

/// Export dimensions, using the same percent/width/height precedence as web_document.
pub fn dimensions(doc: &photocraft_doc::Document, p: &Value) -> Result<(f64, f64), String> {
    let (w0, h0) = (f64::from(doc.size.width).max(1.0), f64::from(doc.size.height).max(1.0));
    let positive = |k: &str| p.get(k).and_then(Value::as_f64).filter(|v| v.is_finite() && *v > 0.0);
    let (w, h) = match (positive("percent"), positive("width"), positive("height")) {
        (Some(pct), _, _) => (w0 * pct / 100.0, h0 * pct / 100.0),
        (_, Some(w), Some(h)) => (w, h),
        (_, Some(w), None) => (w, h0 * w / w0),
        (_, None, Some(h)) => (w0 * h / h0, h),
        _ => (w0, h0),
    };
    if !w.is_finite() || !h.is_finite() || w > 300_000.0 || h > 300_000.0 {
        return Err("export dimensions exceed the 300,000 px per-side limit".into());
    }
    Ok((w.round().max(1.0), h.round().max(1.0)))
}

pub fn prepare_table(app: &PhotocraftApp, f: &mut Map<String, Value>) -> Result<(), String> {
    if !matches!(s(f, "format", "jpeg").as_str(), "gif" | "png8") {
        return Ok(());
    }
    let st = app.session.active().ok_or("no document")?;
    let mut source = params(f);
    let Some(m) = source.as_object_mut() else { return Err("invalid web settings".into()) };
    // Only palette-affecting options invalidate index-based edits. Metadata, interlacing,
    // dithering and output paths must not discard the user's replacement colours.
    m.retain(|k, _| {
        matches!(
            k.as_str(),
            "preset" | "format" | "palette" | "colors" | "transparency" | "matte" | "webSnap" | "convertToSrgb" | "width" | "height" | "percent" | "resample"
        )
    });
    let signature = format!("{}|{}|{}", st.doc.id.0, st.revision, source);
    if f.get("__tableSignature").and_then(Value::as_str) == Some(&signature) {
        return Ok(());
    }
    // A regenerated palette invalidates index-based edits. Explicit tables supplied through
    // ui.dialog.set remain usable until a palette/size setting changes.
    if f.contains_key("__tableSignature") {
        f.remove("colorTable");
        f.remove("colorTableEdits");
        f.remove("__selectedColor");
    }
    if !f.contains_key("colorTable") {
        let settings = WebSettings::from_params(&source, "preview").map_err(|e| e.to_string())?;
        // Bound palette preparation as well as the displayed image. The fixed table is then
        // passed unchanged to full-resolution export, so the swatches are the file's colours.
        let (w, h) = dimensions(&st.doc, &source)?;
        let scale = (2048.0 / w.max(h)).min(1.0);
        let mut proxy = source.clone();
        if let Some(m) = proxy.as_object_mut() {
            m.remove("percent");
            m.insert("width".into(), json!((w * scale).round().max(1.0)));
            m.insert("height".into(), json!((h * scale).round().max(1.0)));
            m.insert("resample".into(), json!("bilinear"));
        }
        let (doc, _, _) = web_cmds::web_document(&st.doc, &proxy, &settings).map_err(|e| e.to_string())?;
        f.insert("__paletteApproximate".into(), json!(scale < 1.0));
        let buf = photocraft_compose::flatten(&doc);
        let mut optimized =
            web_cmds::optimize(&buf.px, doc.size.width as usize, doc.bounds(), &settings, None, None, doc.resolution_dpi, false).map_err(|e| e.to_string())?;
        // Tiny transparent features may disappear in the proxy; keep room for their index.
        if settings.transparency {
            optimized.color_table.truncate(255);
        }
        f.insert("colorTable".into(), json!(optimized.color_table.iter().map(|c| web_cmds::rgb_hex(*c)).collect::<Vec<_>>()));
        f.insert("__transparentIndex".into(), json!(optimized.transparent_index));
    }
    f.insert("__tableSignature".into(), json!(signature));
    Ok(())
}

pub fn table(ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    let p = params(f);
    let Ok(settings) = WebSettings::from_params(&p, "preview") else { return };
    let Some(table) = settings.color_table else { return };
    ui.add_space(8.0);
    ui.label(egui::RichText::new(tl!("Color Table…").trim_end_matches('…')).font(crate::theme::semibold(12.0)));
    ui.scope(|ui| {
        ui.spacing_mut().interact_size = egui::vec2(14.0, 14.0);
        ui.spacing_mut().item_spacing = egui::vec2(1.0, 1.0);
        egui::Grid::new("web-color-table").spacing(egui::vec2(1.0, 1.0)).show(ui, |ui| {
            for (i, source) in table.iter().enumerate() {
                let mut color = settings.color_table_edits.get(&i).copied().unwrap_or(*source);
                let response = ui.push_id(i, |ui| crate::widgets::color_edit_button_srgb(ui, &mut color)).inner;
                response.clone().on_hover_text(format!("Index {i}: {}", web_cmds::rgb_hex(color)));
                if response.clicked() {
                    f.insert("__selectedColor".into(), json!(i));
                }
                if f.get("__selectedColor").and_then(Value::as_u64) == Some(i as u64) {
                    let t = Tokens::get(ui.ctx());
                    ui.painter().rect_stroke(response.rect, t.radius_sm, egui::Stroke::new(1.0, t.accent), egui::StrokeKind::Inside);
                }
                if response.changed() {
                    let edits = f.entry("colorTableEdits").or_insert_with(|| json!({}));
                    if let Some(edits) = edits.as_object_mut() {
                        edits.insert(i.to_string(), json!(web_cmds::rgb_hex(color)));
                    }
                }
                if i % 16 == 15 {
                    ui.end_row();
                }
            }
            if f.get("__transparentIndex").and_then(Value::as_u64).is_some() {
                let (rect, response) = ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
                crate::widgets::checker(ui.painter(), rect, 4.0);
                response.on_hover_text(tl!("Transparency"));
            }
        });
    });
    ui.horizontal(|ui| {
        ui.weak(format!("{}: {}", tl!("Colors"), table.len()));
        if ui.small_button(tl!("Reset")).clicked() {
            f.remove("colorTableEdits");
        }
    });
}

type EncodedPreview = (Vec<u8>, usize, Option<usize>, &'static str);

/// Palette replacements reuse the cached index buffer; picker drags do not re-quantize or
/// re-encode the proxy. The GIF payload length is unchanged by RGB table replacements.
pub fn optimized(ui: &mut egui::Ui, key: egui::Id, signature: &str, p: &Value, px: &[[f32; 4]], w: u32, h: u32) -> Result<EncodedPreview, String> {
    let settings = WebSettings::from_params(p, "preview").map_err(|e| e.to_string())?;
    let mut base = p.clone();
    if let Some(m) = base.as_object_mut() {
        m.remove("colorTableEdits");
    }
    let signature = format!("{signature}|{base}");
    let key = key.with("indexed-source");
    let cached: Option<(String, Arc<Optimized>)> = ui.data(|d| d.get_temp(key));
    let encoded = if let Some((_, encoded)) = cached.filter(|(sig, _)| sig == &signature) {
        encoded
    } else {
        let settings = WebSettings::from_params(&base, "preview").map_err(|e| e.to_string())?;
        let encoded = Arc::new(
            web_cmds::optimize(px, w as usize, photocraft_geom::Rect::new(0, 0, w as i32, h as i32), &settings, None, None, 72.0, true)
                .map_err(|e| e.to_string())?,
        );
        ui.data_mut(|d| d.insert_temp(key, (signature, encoded.clone())));
        encoded
    };
    let rgba = if settings.color_table_edits.is_empty() || !settings.format.indexed() {
        encoded.preview.clone()
    } else {
        encoded
            .indices
            .iter()
            .flat_map(|index| {
                let index = usize::from(*index);
                let color = settings.color_table_edits.get(&index).copied().or_else(|| encoded.color_table.get(index).copied()).unwrap_or([0; 3]);
                [color[0], color[1], color[2], if Some(index) == encoded.transparent_index { 0 } else { 255 }]
            })
            .collect()
    };
    Ok((rgba, encoded.bytes.len(), encoded.colors, encoded.ext))
}

#[derive(Clone, Copy, Debug)]
struct Navigation {
    zoom: Option<f32>,
    center: egui::Vec2,
}
impl Navigation {
    fn read(f: &Map<String, Value>) -> Self {
        let zoom = f.get("__zoom").and_then(Value::as_f64).filter(|z| z.is_finite() && *z > 0.0).map(|z| (z as f32).clamp(0.0001, 32.0));
        let center = f
            .get("__center")
            .and_then(Value::as_array)
            .and_then(|a| Some(egui::vec2(a.first()?.as_f64()? as f32, a.get(1)?.as_f64()? as f32)))
            .filter(|c| c.is_finite())
            .unwrap_or(egui::vec2(0.5, 0.5));
        Self { zoom, center }
    }
    fn write(self, f: &mut Map<String, Value>) {
        f.insert("__zoom".into(), json!(self.zoom));
        f.insert("__center".into(), json!([self.center.x, self.center.y]));
    }
    fn scale(self, viewport: egui::Vec2, source: egui::Vec2, ppp: f32) -> f32 {
        self.zoom.map_or((viewport.x / source.x).min(viewport.y / source.y), |z| z / ppp)
    }
    fn clamp(&mut self, viewport: egui::Vec2, image: egui::Vec2) {
        let half = (viewport / image * 0.5).min(egui::vec2(0.5, 0.5));
        self.center = self.center.clamp(half, egui::Vec2::splat(1.0) - half);
    }
    fn zoom_at(&mut self, factor: f32, at: egui::Vec2, viewport: egui::Vec2, source: egui::Vec2, ppp: f32) {
        if !factor.is_finite() || factor <= 0.0 {
            return;
        }
        let old = self.scale(viewport, source, ppp);
        self.zoom = Some((old * ppp * factor).clamp(0.0001, 32.0));
        let new = self.scale(viewport, source, ppp);
        self.center += at / source * (1.0 / old - 1.0 / new);
        self.clamp(viewport, source * new);
    }
}

pub fn controls(ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    let mut nav = Navigation::read(f);
    let effective = n(f, "__effectiveZoom", 1.0) as f32;
    ui.horizontal(|ui| {
        if ui.small_button(tl!("Fit")).clicked() {
            nav.zoom = None;
            nav.center = egui::vec2(0.5, 0.5);
        }
        if ui.small_button("−").clicked() {
            nav.zoom = Some((nav.zoom.unwrap_or(effective) / 1.25).max(0.0001));
        }
        if ui.small_button("+").clicked() {
            nav.zoom = Some((nav.zoom.unwrap_or(effective) * 1.25).min(32.0));
        }
        let mut percent = nav.zoom.unwrap_or(effective) * 100.0;
        if ui.add(egui::DragValue::new(&mut percent).range(0.01..=3200.0).suffix("%").speed(10.0)).changed() {
            nav.zoom = Some(percent / 100.0);
        }
        if ui.small_button(tl!("100%")).clicked() {
            nav.zoom = Some(1.0);
        }
        ui.add(egui::Label::new(egui::RichText::new(tl!("Drag to pan · Scroll to zoom · Double-click to fit")).weak()).truncate());
    });
    nav.write(f);
}

pub fn paint(ui: &mut egui::Ui, rect: egui::Rect, response: &egui::Response, tex: &egui::TextureHandle, source: egui::Vec2, f: &mut Map<String, Value>) {
    let mut nav = Navigation::read(f);
    let ppp = ui.ctx().pixels_per_point();
    if response.double_clicked() {
        nav.zoom = None;
        nav.center = egui::Vec2::splat(0.5);
    }
    let mut scale = nav.scale(rect.size(), source, ppp);
    if response.dragged() {
        nav.zoom = Some(scale * ppp);
        nav.center -= response.drag_delta() / (source * scale);
    }
    if response.hovered() {
        let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
        let factor = if pinch != 1.0 { pinch } else { (scroll * 0.005).exp() };
        if factor != 1.0 {
            let at = response.hover_pos().unwrap_or(rect.center()) - rect.center();
            nav.zoom_at(factor, at, rect.size(), source, ppp);
            scale = nav.scale(rect.size(), source, ppp);
            ui.ctx().request_repaint();
        }
    }
    nav.clamp(rect.size(), source * scale);
    let image = egui::Rect::from_min_size(rect.center() - nav.center * source * scale, source * scale);
    ui.painter().with_clip_rect(rect.intersect(ui.clip_rect())).image(
        tex.id(),
        image,
        egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );
    // Fit follows resizing; explicit zoom/pan shares the same normalized centre in every pane.
    if nav.zoom.is_some() {
        f.insert("__effectiveZoom".into(), json!(nav.zoom));
    } else {
        f.insert("__effectiveZoom".into(), json!(scale * ppp));
    }
    nav.write(f);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn anchored_zoom_keeps_pixel_under_pointer_and_pan_is_bounded() {
        let mut nav = Navigation { zoom: Some(1.0), center: egui::Vec2::splat(0.5) };
        let viewport = egui::vec2(200.0, 100.0);
        let source = egui::vec2(800.0, 400.0);
        let at = egui::vec2(30.0, 20.0);
        let before = nav.center + at / source;
        nav.zoom_at(2.0, at, viewport, source, 1.0);
        let after = nav.center + at / (source * 2.0);
        assert!((before - after).length() < 0.00001);
        nav.center = egui::vec2(-5.0, 50.0);
        nav.clamp(viewport, source * 2.0);
        assert_eq!(nav.center, egui::vec2(0.0625, 0.9375));
        nav.zoom = None;
        assert_eq!(nav.scale(viewport, source, 1.0), 0.25);
        assert_eq!(nav.scale(viewport * 2.0, source, 1.0), 0.5);
    }

    #[test]
    fn palette_picker_reuses_indices_and_updates_preview_pixels() {
        let key = egui::Id::new("test-web-preview");
        let p = json!({"format": "gif", "colorTable": ["#ff0000", "#0000ff"], "dither": "none"});
        let mut h = egui_kittest::Harness::builder().build_ui_state(
            move |ui, state: &mut (Value, Vec<u8>)| {
                state.1 = optimized(ui, key, "fixture", &state.0, &[[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]], 2, 1).unwrap().0;
            },
            (p, Vec::new()),
        );
        let cached = |ctx: &egui::Context| ctx.data(|d| d.get_temp::<(String, Arc<Optimized>)>(key.with("indexed-source"))).unwrap().1;
        let before = cached(&h.ctx);
        assert_eq!(h.state().1, [255, 0, 0, 255, 0, 0, 255, 255]);
        h.state_mut().0["colorTableEdits"] = json!({"0": "#00ff00"});
        h.run_steps(3);
        assert!(Arc::ptr_eq(&before, &cached(&h.ctx)), "picker drag must not re-encode");
        assert_eq!(h.state().1, [0, 255, 0, 255, 0, 0, 255, 255]);
    }

    #[test]
    fn absurd_dimensions_and_navigation_fields_are_safe() {
        let doc =
            photocraft_doc::Document::new("fixture", photocraft_geom::Size::new(100, 50), photocraft_color::ColorMode::Rgb, photocraft_color::SampleType::U8);
        assert!(dimensions(&doc, &json!({"percent": 1e308})).is_err());
        assert!(dimensions(&doc, &json!({"width": 1e300})).is_err());
        let f = serde_json::from_value::<Map<String, Value>>(json!({"__zoom": 1e300, "__center": [1e300, -1e300]})).unwrap();
        let nav = Navigation::read(&f);
        assert_eq!(nav.zoom, Some(32.0));
        assert_eq!(nav.center, egui::Vec2::splat(0.5));
        let mut nav = nav;
        for factor in [f32::NAN, f32::INFINITY, 0.0, -1.0] {
            nav.zoom_at(factor, egui::Vec2::ZERO, egui::vec2(200.0, 100.0), egui::vec2(400.0, 200.0), 1.0);
            assert_eq!(nav.zoom, Some(32.0));
            assert!(nav.center.is_finite());
        }
    }
}
