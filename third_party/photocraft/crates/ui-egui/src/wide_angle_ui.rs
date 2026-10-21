//! Filter › Adaptive Wide Angle…: a full-window dialog like Photoshop's. Drag on the image to
//! draw a constraint (the scene line between the two points, curved by the lens model); hold
//! Shift when releasing to make it horizontal or vertical; right-click a constraint to delete
//! it. The right panel holds the correction model, focal length, crop factor, scale and the
//! constraint list. The preview solves the mesh warp on a CPU proxy (≤ 800 px); OK runs
//! `filter.adaptiveWideAngle` with the constraints in document pixels.
//!
//! Control channel: `ui.menu.invoke {"id":"filter.adaptiveWideAngle","params":{"ui":{"set":{…},
//! "add":{"a":[x,y],"b":[x,y],"orientation":"…"}, "remove":i, "commit":true | "cancel":true}}}`.

use egui::{Align2, Color32, FontId, Pos2, Rect as ERect, Sense, Stroke, TextureHandle, pos2, vec2};
use photocraft_algo::transform::Interp;
use photocraft_algo::wideangle::{self, Camera, Constraint, Orientation, WideAngle, WideModel};
use photocraft_color::PixelFormat;
use photocraft_doc::LayerId;
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::edit_history::{EditHistory, Step};
use crate::theme::Tokens;
use crate::widgets;

const PROXY_SIDE: usize = 800;
const PANEL_W: f32 = 300.0;

pub struct WideAngleDialog {
    pub layer: LayerId,
    layer_name: String,
    pub params: WideAngle,
    /// Canvas frame (document px) and the proxy scale (proxy px per document px).
    frame: Rect,
    scale: f64,
    proxy: Surface,
    pw: usize,
    ph: usize,
    src_tex: Option<TextureHandle>,
    out_tex: Option<TextureHandle>,
    dirty: bool,
    pub preview: bool,
    drag: Option<([f64; 2], [f64; 2])>,
    pub residual: f64,
    pub render_ms: f64,
    history: EditHistory<WideAngle>,
}

fn tex_image(s: &Surface, w: usize, h: usize) -> egui::ColorImage {
    let mut row = vec![[0u8; 4]; w * h];
    s.read_rgba8_into(Rect::new(0, 0, w as i32, h as i32), &mut row);
    egui::ColorImage::new([w, h], row.iter().map(|p| Color32::from_rgba_unmultiplied(p[0], p[1], p[2], p[3])).collect())
}

impl WideAngleDialog {
    pub fn describe(&self) -> Value {
        json!({"layer": self.layer.0, "params": serde_json::to_value(&self.params).unwrap_or(Value::Null), "proxy": [self.pw, self.ph], "residual": self.residual, "renderMs": self.render_ms, "preview": self.preview})
    }

    fn proxy_params(&self) -> WideAngle {
        let k = self.scale;
        let mut p = self.params.clone();
        let (fx, fy) = (self.frame.x0 as f64, self.frame.y0 as f64);
        p.constraints = p
            .constraints
            .iter()
            .map(|c| Constraint { a: [(c.a[0] - fx) * k, (c.a[1] - fy) * k], b: [(c.b[0] - fx) * k, (c.b[1] - fy) * k], orientation: c.orientation })
            .collect();
        p
    }

    fn render(&mut self, ctx: &egui::Context) {
        let t0 = crate::gpu_canvas::now_ms();
        let pf = Rect::new(0, 0, self.pw as i32, self.ph as i32);
        let p = self.proxy_params();
        let mesh = wideangle::solve(&p, pf);
        self.residual = mesh.residual / self.scale;
        let out = photocraft_algo::warp::warp_triangles(&self.proxy, pf, &mesh.verts, &mesh.triangles(), Interp::Bilinear);
        let img = tex_image(&out, self.pw, self.ph);
        match &mut self.out_tex {
            Some(t) => t.set(img, egui::TextureOptions::LINEAR),
            None => self.out_tex = Some(ctx.load_texture("awa-out", img, egui::TextureOptions::LINEAR)),
        }
        if self.src_tex.is_none() {
            self.src_tex = Some(ctx.load_texture("awa-src", tex_image(&self.proxy, self.pw, self.ph), egui::TextureOptions::LINEAR));
        }
        self.render_ms = crate::gpu_canvas::now_ms() - t0;
        self.dirty = false;
    }

    pub(crate) fn step(&mut self, step: Step) {
        if self.history.step(&mut self.params, step) {
            self.drag = None;
            self.dirty = true;
        }
    }

    fn command_params(&self) -> Value {
        let mut v = serde_json::to_value(&self.params).unwrap_or(json!({}));
        v["layer"] = json!(self.layer.0);
        v
    }
}

pub fn open(app: &mut PhotocraftApp, ctx: &egui::Context) -> Result<(), String> {
    photocraft_engine::commands::find("filter.adaptiveWideAngle").map(|c| (c.enabled)(&app.session)).unwrap_or(Err("unknown command".into()))?;
    let (layer, surf, _) = crate::distort_ui::active_pixels(app)?;
    let st = app.session.active().ok_or("no document")?;
    let frame = st.doc.bounds();
    let name = st.doc.layer(layer).map(|l| l.name.clone()).unwrap_or_default();
    let (w, h) = (frame.width() as usize, frame.height() as usize);
    let k = w.max(h).div_ceil(PROXY_SIDE).max(1);
    let (pw, ph) = (w.div_ceil(k), h.div_ceil(k));
    let proxy = photocraft_algo::resample::resize_surface(
        &surf.convert(PixelFormat::RGBA8),
        1.0 / k as f64,
        1.0 / k as f64,
        photocraft_algo::resample::Resample::Bilinear,
    );
    // Normalise the proxy to start at the origin.
    let mut px = Surface::new(PixelFormat::RGBA8);
    let src_r = Rect::new(
        (frame.x0 as f64 / k as f64) as i32,
        (frame.y0 as f64 / k as f64) as i32,
        (frame.x0 as f64 / k as f64) as i32 + pw as i32,
        (frame.y0 as f64 / k as f64) as i32 + ph as i32,
    );
    px.write_region(Rect::new(0, 0, pw as i32, ph as i32), &proxy.read_region(src_r));
    // EXIF focal length / crop factor, as the engine would resolve them.
    let info = st.doc.metadata.exif.as_ref().map(|e| photocraft_algo::exif::read(e));
    let params = photocraft_engine::lens_cmds::wide_params("filter.adaptiveWideAngle", &json!({}), info.as_ref()).unwrap_or_default();
    let mut d = WideAngleDialog {
        layer,
        layer_name: name,
        history: EditHistory::new(&params),
        params,
        frame,
        scale: 1.0 / k as f64,
        proxy: px,
        pw,
        ph,
        src_tex: None,
        out_tex: None,
        dirty: true,
        preview: true,
        drag: None,
        residual: 0.0,
        render_ms: 0.0,
    };
    d.render(ctx);
    app.wide_angle = Some(d);
    Ok(())
}

fn parse_orientation(s: &str) -> Orientation {
    match s {
        "horizontal" => Orientation::Horizontal,
        "vertical" => Orientation::Vertical,
        _ => Orientation::Free,
    }
}

pub fn menu(app: &mut PhotocraftApp, ctx: &egui::Context, id: &str, params: &Value) -> Option<Result<Value, String>> {
    if id != "filter.adaptiveWideAngle" {
        return None;
    }
    if params.as_object().is_none_or(|o| o.is_empty()) {
        return Some(open(app, ctx).map(|_| app.wide_angle.as_ref().map(|d| d.describe()).unwrap_or(Value::Null)));
    }
    let ui = params.get("ui")?;
    if app.wide_angle.is_none()
        && let Err(e) = open(app, ctx)
    {
        return Some(Err(e));
    }
    let d = app.wide_angle.as_mut()?;
    if let Some(set) = ui.get("set") {
        let mut cur = serde_json::to_value(&d.params).unwrap_or(json!({}));
        if let (Value::Object(c), Value::Object(s)) = (&mut cur, set) {
            for (k, v) in s {
                c.insert(k.clone(), v.clone());
            }
        }
        match serde_json::from_value::<WideAngle>(cur) {
            Ok(p) => d.params = p,
            Err(e) => return Some(Err(format!("bad Adaptive Wide Angle settings: {e}"))),
        }
        d.dirty = true;
    }
    if let Some(a) = ui.get("add") {
        let pt = |k: &str| -> Option<[f64; 2]> { Some([a.get(k)?.get(0)?.as_f64()?, a.get(k)?.get(1)?.as_f64()?]) };
        let (Some(pa), Some(pb)) = (pt("a"), pt("b")) else { return Some(Err("`add` needs a and b points".into())) };
        d.params.constraints.push(Constraint { a: pa, b: pb, orientation: parse_orientation(a.get("orientation").and_then(Value::as_str).unwrap_or("free")) });
        d.dirty = true;
    }
    if let Some(i) = ui.get("remove").and_then(Value::as_u64)
        && (i as usize) < d.params.constraints.len()
    {
        d.params.constraints.remove(i as usize);
        d.dirty = true;
    }
    d.history.settle(&d.params, false);
    if d.dirty {
        d.render(ctx);
    }
    if ui.get("cancel").and_then(Value::as_bool) == Some(true) {
        app.wide_angle = None;
        return Some(Ok(json!({"cancelled": true})));
    }
    if ui.get("commit").and_then(Value::as_bool) == Some(true) {
        return Some(commit(app));
    }
    Some(Ok(app.wide_angle.as_ref().map(|d| d.describe()).unwrap_or(Value::Null)))
}

fn commit(app: &mut PhotocraftApp) -> Result<Value, String> {
    let d = app.wide_angle.take().ok_or(tl!("Adaptive Wide Angle isn't open"))?;
    app.run("filter.adaptiveWideAngle", d.command_params())
}

pub fn show(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if app.wide_angle.is_none() {
        return;
    }
    let t = Tokens::get(ctx);
    let screen = ctx.content_rect();
    let mut action: Option<&str> = None;
    let shift = ctx.input(|i| i.modifiers.shift);
    egui::Area::new(egui::Id::new("awa-dialog")).order(egui::Order::Foreground).fixed_pos(screen.min).show(ctx, |ui| {
        let Some(d) = app.wide_angle.as_mut() else { return };
        d.history.settle(&d.params, ctx.input(|i| i.pointer.any_down()));
        if d.dirty {
            d.render(ctx);
        }
        let (full, _) = ui.allocate_exact_size(screen.size(), Sense::hover());
        let painter = ui.painter().clone();
        painter.rect_filled(full, 0.0, t.chrome);
        let title = ERect::from_min_size(full.min, vec2(full.width(), 30.0));
        painter.rect_filled(title, 0.0, t.dock);
        painter.text(title.center(), Align2::CENTER_CENTER, format!("Adaptive Wide Angle ({})", d.layer_name), FontId::proportional(13.0), t.text);
        let footer_h = 48.0;
        let body = ERect::from_min_max(pos2(full.left(), title.bottom()), pos2(full.right(), full.bottom() - footer_h));
        let view = ERect::from_min_max(body.min, pos2(body.right() - PANEL_W, body.bottom())).shrink(16.0);
        painter.rect_filled(view, 0.0, t.canvas);
        let s = (view.width() / d.pw as f32).min(view.height() / d.ph as f32);
        let img_r = ERect::from_center_size(view.center(), vec2(d.pw as f32 * s, d.ph as f32 * s));
        // Screen ↔ document mapping.
        let k = d.scale;
        let (fx, fy) = (d.frame.x0 as f64, d.frame.y0 as f64);
        let to_screen = |p: [f64; 2]| pos2(img_r.left() + ((p[0] - fx) * k) as f32 * s, img_r.top() + ((p[1] - fy) * k) as f32 * s);
        let to_doc = |q: Pos2| [fx + ((q.x - img_r.left()) / s) as f64 / k, fy + ((q.y - img_r.top()) / s) as f64 / k];
        let resp = ui.interact(img_r, ui.id().with("awa-image"), Sense::click_and_drag());
        if let Some(p) = resp.interact_pointer_pos() {
            if resp.drag_started() {
                d.drag = Some((to_doc(p), to_doc(p)));
            } else if resp.dragged()
                && let Some(dr) = d.drag.as_mut()
            {
                dr.1 = to_doc(p);
            }
        }
        if resp.drag_stopped()
            && let Some((a, b)) = d.drag.take()
            && (a[0] - b[0]).hypot(a[1] - b[1]) * k * s as f64 > 6.0
        {
            let orientation = if shift {
                if (b[0] - a[0]).abs() >= (b[1] - a[1]).abs() { Orientation::Horizontal } else { Orientation::Vertical }
            } else {
                Orientation::Free
            };
            d.params.constraints.push(Constraint { a, b, orientation });
            d.dirty = true;
        }
        let cam = Camera::new(&d.params, d.frame);
        if resp.secondary_clicked()
            && let Some(p) = resp.interact_pointer_pos()
        {
            // Delete the constraint whose curve passes nearest the click.
            let near = d
                .params
                .constraints
                .iter()
                .enumerate()
                .map(|(i, c)| (i, cam.arc(c.a, c.b, 24).iter().map(|q| (to_screen(*q) - p).length()).fold(f32::MAX, f32::min)))
                .min_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((i, dist)) = near
                && dist < 10.0
            {
                d.params.constraints.remove(i);
                d.dirty = true;
            }
        }
        let show_src = !d.preview || d.drag.is_some();
        let tex = if show_src { d.src_tex.as_ref() } else { d.out_tex.as_ref() };
        widgets::checker(&painter, img_r, 8.0);
        if let Some(tex) = tex {
            painter.image(tex.id(), img_r, ERect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
        }
        // Constraints: curved over the source; straight end-to-end over the corrected preview.
        let color = |o: Orientation| match o {
            Orientation::Free => t.accent,
            Orientation::Horizontal => t.warning,
            Orientation::Vertical => t.danger,
        };
        let pf = Rect::new(0, 0, d.pw as i32, d.ph as i32);
        let mesh = (!show_src && !d.params.constraints.is_empty()).then(|| wideangle::solve(&d.proxy_params(), pf));
        for c in &d.params.constraints {
            let pts: Vec<Pos2> = match &mesh {
                Some(m) => cam
                    .arc(c.a, c.b, 24)
                    .iter()
                    .map(|q| {
                        let pq = m.map(pf, [(q[0] - fx) * k, (q[1] - fy) * k]);
                        to_screen([pq[0] / k + fx, pq[1] / k + fy])
                    })
                    .collect(),
                None => cam.arc(c.a, c.b, 24).iter().map(|q| to_screen(*q)).collect(),
            };
            if let (Some(a), Some(b)) = (pts.first(), pts.last()) {
                painter.circle_filled(*a, 3.5, color(c.orientation));
                painter.circle_filled(*b, 3.5, color(c.orientation));
            }
            painter.add(egui::Shape::line(pts, Stroke::new(2.0, color(c.orientation))));
        }
        if let Some((a, b)) = d.drag {
            let pts: Vec<Pos2> = cam.arc(a, b, 24).iter().map(|q| to_screen(*q)).collect();
            painter.add(egui::Shape::line(pts, Stroke::new(2.0, t.accent_soft)));
        }
        // Panel.
        let right = ERect::from_min_max(pos2(body.right() - PANEL_W, body.top()), body.max);
        painter.rect_filled(right, 0.0, t.dock);
        painter.line_segment([right.left_top(), right.left_bottom()], Stroke::new(1.0, t.separator));
        let mut props = ui.new_child(egui::UiBuilder::new().max_rect(right.shrink2(vec2(14.0, 10.0))));
        egui::ScrollArea::vertical().id_salt("awa-props").show(&mut props, |ui| {
            ui.spacing_mut().item_spacing.y = 5.0;
            widgets::section_label(ui, tl!("Correction"));
            let mut model = d.params.model;
            if widgets::dropdown(
                ui,
                "awa-model",
                &mut model,
                &[
                    (WideModel::Auto, tl!("Auto")),
                    (WideModel::Fisheye, tl!("Fisheye")),
                    (WideModel::Perspective, tl!("Perspective")),
                    (WideModel::FullSpherical, tl!("Full Spherical")),
                ],
                200.0,
            ) {
                d.params.model = model;
                d.dirty = true;
            }
            let mut v = d.params.scale as f32;
            if widgets::slider_row(ui, tl!("Scale"), &mut v, 50.0..=150.0, "%", None).changed() {
                d.params.scale = v as f64;
                d.dirty = true;
            }
            let mut v = d.params.focal_length as f32;
            if widgets::slider_row(ui, tl!("Focal Length"), &mut v, 0.0..=200.0, "mm", None).changed() {
                d.params.focal_length = v as f64;
                d.dirty = true;
            }
            let mut v = d.params.crop_factor as f32;
            if widgets::slider_row(ui, tl!("Crop Factor"), &mut v, 0.1..=10.0, "", None).changed() {
                d.params.crop_factor = v as f64;
                d.dirty = true;
            }
            widgets::hairline(ui);
            widgets::section_label(ui, &crate::i18n::fmt(tl!("Constraints ({n})"), &[("n", &d.params.constraints.len().to_string())]));
            let mut remove = None;
            for (i, c) in d.params.constraints.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(format!("{}", i + 1));
                    let mut o = c.orientation;
                    if widgets::dropdown(
                        ui,
                        &format!("awa-o-{i}"),
                        &mut o,
                        &[(Orientation::Free, tl!("Free")), (Orientation::Horizontal, tl!("Horizontal")), (Orientation::Vertical, tl!("Vertical"))],
                        120.0,
                    ) {
                        c.orientation = o;
                        d.dirty = true;
                    }
                    if widgets::secondary_button(ui, tl!("Delete"), 60.0).clicked() {
                        remove = Some(i);
                    }
                });
            }
            if let Some(i) = remove {
                d.params.constraints.remove(i);
                d.dirty = true;
            }
            widgets::hairline(ui);
            widgets::checkbox(ui, &mut d.preview, tl!("Preview"));
            ui.label(
                egui::RichText::new(tl!("Drag on the image to add a constraint; Shift for horizontal/vertical; right-click to delete."))
                    .color(t.text_faint)
                    .size(11.0),
            );
        });
        let foot = ERect::from_min_max(pos2(full.left(), full.bottom() - footer_h), full.max);
        painter.rect_filled(foot, 0.0, t.dock);
        let mut fu = ui.new_child(egui::UiBuilder::new().max_rect(foot.shrink2(vec2(16.0, 9.0))).layout(egui::Layout::right_to_left(egui::Align::Center)));
        if let Some(role) = widgets::dialog_buttons(
            &mut fu,
            &[
                widgets::DialogButton::new(widgets::ButtonRole::Default, tl!("OK"), 90.0),
                widgets::DialogButton::new(widgets::ButtonRole::Cancel, tl!("Cancel"), 90.0),
            ],
        ) {
            action = Some(if role == widgets::ButtonRole::Default { "ok" } else { "cancel" });
        }
        fu.label(egui::RichText::new(format!("{:.0} ms · straightness {:.1} px", d.render_ms, d.residual)).color(t.text_faint));
    });
    if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
        action = Some("cancel");
    }
    match action {
        Some("ok") => {
            if let Err(e) = commit(app) {
                app.ui.status = e;
            }
        }
        Some("cancel") => app.wide_angle = None,
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dialog_adds_constraints_and_commits() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width": 160, "height": 100})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        app.run("edit.fill", json!({"color": "#808080"})).unwrap();
        let r = menu(&mut app, &ctx, "filter.adaptiveWideAngle", &json!({})).unwrap().unwrap();
        assert_eq!(r["proxy"], json!([160, 100]));
        let r = menu(
            &mut app,
            &ctx,
            "filter.adaptiveWideAngle",
            &json!({"ui": {"set": {"model": "fisheye", "focalLength": 10.0}, "add": {"a": [20, 20], "b": [140, 20], "orientation": "horizontal"}}}),
        )
        .unwrap()
        .unwrap();
        assert_eq!(r["params"]["constraints"].as_array().unwrap().len(), 1);
        assert!(r["residual"].as_f64().unwrap() < 3.0, "{r}");
        let r = menu(&mut app, &ctx, "filter.adaptiveWideAngle", &json!({"ui": {"commit": true}})).unwrap().unwrap();
        assert_eq!(r["model"], "fisheye");
        assert!(app.wide_angle.is_none());
        menu(&mut app, &ctx, "filter.adaptiveWideAngle", &json!({})).unwrap().unwrap();
        assert!(menu(&mut app, &ctx, "filter.adaptiveWideAngle", &json!({"ui": {"add": {"a": [1, 1]}}})).unwrap().is_err());
        menu(&mut app, &ctx, "filter.adaptiveWideAngle", &json!({"ui": {"cancel": true}})).unwrap().unwrap();
        assert!(app.wide_angle.is_none());
    }

    #[test]
    fn command_z_steps_the_dialog_not_the_document() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width": 160, "height": 100})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        app.run("edit.fill", json!({"color": "#808080"})).unwrap();
        let steps = app.session.active().unwrap().history.past_len();
        menu(&mut app, &ctx, "filter.adaptiveWideAngle", &json!({})).unwrap().unwrap();
        for y in [20, 60] {
            menu(&mut app, &ctx, "filter.adaptiveWideAngle", &json!({"ui": {"add": {"a": [20, y], "b": [140, y]}}})).unwrap().unwrap();
        }
        crate::edit_history::tests::press_z(&mut app, &ctx, egui::Modifiers::COMMAND);
        assert_eq!(app.wide_angle.as_ref().unwrap().params.constraints.len(), 1, "⌘Z removes the last constraint");
        assert_eq!(app.session.active().unwrap().history.past_len(), steps, "the document beneath is untouched");
        crate::edit_history::tests::press_z(&mut app, &ctx, egui::Modifiers::COMMAND | egui::Modifiers::SHIFT);
        assert_eq!(app.wide_angle.as_ref().unwrap().params.constraints.len(), 2, "⇧⌘Z puts it back");
    }
}
