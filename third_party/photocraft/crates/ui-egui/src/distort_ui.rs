//! Interactive Liquify (full-window dialog), Puppet Warp, Perspective Warp and Vanishing Point
//! (on-canvas modes).
//!
//! All three are previews over plain data (strokes, pins, planes); committing runs the engine
//! command with that data (`filter.liquify`, `edit.puppetWarp`, `edit.perspectiveWarp`,
//! `filter.vanishingPoint`, or the smart-object twins), so the result is one history step and exactly replayable. This module
//! holds the shared runtime state and the hooks the shell calls (menus, pointer, keys, overlay,
//! options bar); the modes live in `liquify_ui`, `puppet_ui`, `perspective_ui` and `vp_ui`.

use std::sync::Arc;

use egui::{Color32, ColorImage};
use photocraft_doc::{Document, LayerContent, LayerId};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::canvas::{ToolEvent, ViewXform};

/// Runtime state of the distortion modes (not serialisable: textures, solvers).
#[derive(Default)]
pub struct Distort {
    pub liquify: Option<crate::liquify_ui::LiquifyDialog>,
    pub puppet: Option<crate::puppet_ui::PuppetSession>,
    pub perspective: Option<crate::perspective_ui::PerspSession>,
    pub vanishing: Option<crate::vp_ui::VpSession>,
    /// Filter Gallery dialog (gallery_ui).
    pub gallery: Option<crate::gallery_ui::GalleryDialog>,
}

impl Distort {
    pub fn active(&self) -> bool {
        self.liquify.is_some() || self.puppet.is_some() || self.perspective.is_some() || self.vanishing.is_some()
    }

    /// Short description for `ui.inspect`.
    pub fn describe(&self) -> Value {
        json!({
            "liquify": self.liquify.as_ref().map(|d| d.describe()),
            "puppet": self.puppet.as_ref().map(|p| p.describe()),
            "perspective": self.perspective.as_ref().map(|p| p.describe()),
            "vanishingPoint": self.vanishing.as_ref().map(|v| v.describe()),
            "gallery": self.gallery.as_ref().map(|g| g.describe()),
        })
    }
}

/// The active layer's pixels (a pixel layer, or a smart object's rendering) and its id.
pub(crate) fn active_pixels(app: &PhotocraftApp) -> Result<(LayerId, Surface, bool), String> {
    let st = app.session.active().ok_or("no document")?;
    let id = st.active_layer.ok_or("no active layer")?;
    let l = st.doc.layer(id).ok_or("no layer")?;
    match &l.content {
        LayerContent::Raster(s) => Ok((id, s.clone(), false)),
        LayerContent::Smart(sm) => sm.cache.clone().map(|c| (id, c, true)).ok_or_else(|| "the smart object has no pixels".to_string()),
        other => Err(format!("needs a pixel layer or smart object (active layer is {} {} layer)", other.article(), other.kind_name())),
    }
}

/// The document with layer `id` hidden (what shows behind a live preview of that layer).
pub(crate) fn without_layer(doc: &Document, id: LayerId) -> Arc<Document> {
    let mut d = doc.clone();
    if let Some(l) = d.layer_mut(id) {
        l.visible = false;
    }
    Arc::new(d)
}

/// A texture image of `surf` over `r`, downsampled so the longer side is at most `max_side`.
pub(crate) fn surface_image(surf: &Surface, r: Rect, max_side: usize) -> ColorImage {
    let (w, h) = (r.width().max(1) as usize, r.height().max(1) as usize);
    let k = w.max(h).div_ceil(max_side).max(1);
    let (tw, th) = (w.div_ceil(k), h.div_ceil(k));
    let mut px = vec![Color32::TRANSPARENT; tw * th];
    let mut row = vec![[0u8; 4]; w];
    for ty in 0..th {
        let y = r.y0 + (ty * k) as i32;
        surf.read_rgba8_into(Rect::new(r.x0, y, r.x1, y + 1), &mut row);
        for tx in 0..tw {
            let p = row[(tx * k).min(w - 1)];
            let a = if surf.format().alpha { p[3] } else { 255 };
            px[ty * tw + tx] = Color32::from_rgba_unmultiplied(p[0], p[1], p[2], a);
        }
    }
    ColorImage::new([tw, th], px)
}

/// The document shown on the canvas while Puppet or Perspective Warp previews a layer.
pub fn display_doc(app: &PhotocraftApp, idx: usize) -> Option<(Arc<Document>, u64)> {
    if app.session.active_index() != Some(idx) {
        return None;
    }
    if let Some(p) = &app.distort.puppet {
        return Some((p.preview_doc.clone(), (1 << 42) + p.key));
    }
    if let Some(p) = &app.distort.perspective {
        return Some((p.preview_doc.clone(), (1 << 42) + p.key));
    }
    None
}

/// Menu / control routing. `Some` when this module handles the invocation.
pub fn menu(app: &mut PhotocraftApp, ctx: &egui::Context, id: &str, params: &Value) -> Option<Result<Value, String>> {
    let empty = params.as_object().is_none_or(|o| o.is_empty());
    let ui = params.get("ui");
    match id {
        "filter.filterGallery" if empty => Some(crate::gallery_ui::open(app).map(|_| json!({"gallery": app.distort.describe()["gallery"]}))),
        "filter.filterGallery" if ui.is_some() && app.distort.gallery.is_some() => Some(crate::gallery_ui::control(app, ui.unwrap_or(&Value::Null))),
        "filter.liquify" if empty => Some(crate::liquify_ui::open(app, ctx).map(|_| json!({"liquify": app.distort.describe()["liquify"]}))),
        "filter.liquify" if ui.is_some() && app.distort.liquify.is_some() => Some(crate::liquify_ui::control(app, ui.unwrap_or(&Value::Null))),
        "edit.puppetWarp" | "layer.smartObjects.puppetWarp" if empty => {
            Some(crate::puppet_ui::begin(app, ctx, id).map(|_| json!({"puppet": app.distort.describe()["puppet"]})))
        }
        "edit.puppetWarp" | "layer.smartObjects.puppetWarp" if ui.is_some() && app.distort.puppet.is_some() => {
            Some(crate::puppet_ui::control(app, ui.unwrap_or(&Value::Null)))
        }
        "edit.perspectiveWarp" | "layer.smartObjects.perspectiveWarp" if empty => {
            Some(crate::perspective_ui::begin(app, ctx, id).map(|_| json!({"perspective": app.distort.describe()["perspective"]})))
        }
        "edit.perspectiveWarp" | "layer.smartObjects.perspectiveWarp" if ui.is_some() && app.distort.perspective.is_some() => {
            Some(crate::perspective_ui::control(app, ui.unwrap_or(&Value::Null)))
        }
        "filter.vanishingPoint" if empty => Some(crate::vp_ui::begin(app, ctx).map(|_| json!({"vanishingPoint": app.distort.describe()["vanishingPoint"]}))),
        "filter.vanishingPoint" if ui.is_some() && app.distort.vanishing.is_some() => Some(crate::vp_ui::control(app, ui.unwrap_or(&Value::Null))),
        _ => None,
    }
}

/// Pointer events (document coordinates) while a mode is active. True when consumed.
pub fn pointer(app: &mut PhotocraftApp, ev: ToolEvent, mods: egui::Modifiers) -> bool {
    if app.distort.liquify.is_some() {
        crate::liquify_ui::pointer(app, ev, mods);
        return true;
    }
    if app.distort.puppet.is_some() {
        crate::puppet_ui::pointer(app, ev, mods);
        return true;
    }
    if app.distort.perspective.is_some() {
        crate::perspective_ui::pointer(app, ev, mods);
        return true;
    }
    if app.distort.vanishing.is_some() {
        crate::vp_ui::pointer(app, ev, mods);
        return true;
    }
    false
}

/// Keyboard: ↩ commits, Esc cancels (and Liquify's own shortcuts). True when the key handling
/// of the rest of the app should be skipped.
pub fn keys(app: &mut PhotocraftApp, ctx: &egui::Context) -> bool {
    use egui::{Key, Modifiers};
    let enter = |ctx: &egui::Context| ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Enter));
    let esc = |ctx: &egui::Context| ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape));
    if app.distort.gallery.is_some() {
        if enter(ctx) {
            let _ = crate::gallery_ui::commit(app);
        } else if esc(ctx) {
            app.distort.gallery = None;
        }
        return true;
    }
    if app.distort.liquify.is_some() {
        if enter(ctx) {
            crate::liquify_ui::commit(app);
        } else if esc(ctx) {
            crate::liquify_ui::cancel(app);
        } else {
            crate::liquify_ui::keys(app, ctx);
        }
        return true;
    }
    if let Some(p) = app.distort.puppet.as_mut() {
        let cmd_shift = egui::Modifiers::COMMAND | egui::Modifiers::SHIFT;
        if enter(ctx) {
            crate::puppet_ui::commit(app);
        } else if esc(ctx) {
            app.distort.puppet = None;
        } else if ctx.input_mut(|i| i.consume_key(cmd_shift, egui::Key::Z)) {
            p.redo();
        } else if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Z)) {
            p.undo();
        }
        return true;
    }
    if app.distort.perspective.is_some() {
        if enter(ctx) {
            crate::perspective_ui::commit(app);
        } else if esc(ctx) {
            app.distort.perspective = None;
        }
        return true;
    }
    if app.distort.vanishing.is_some() {
        if enter(ctx) {
            let _ = crate::vp_ui::commit(app);
        } else if esc(ctx) {
            crate::vp_ui::escape(app);
        } else {
            crate::vp_ui::keys(app, ctx);
        }
        return true;
    }
    false
}

/// Canvas overlay for the on-canvas modes.
pub fn draw_overlay(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    if let Some(p) = &app.distort.puppet {
        crate::puppet_ui::draw(p, painter, xf);
    }
    if let Some(p) = &app.distort.perspective {
        crate::perspective_ui::draw(p, painter, xf);
    }
    if let Some(v) = &app.distort.vanishing {
        crate::vp_ui::draw(v, painter, xf);
    }
}

/// Options bar for the on-canvas modes. True when drawn.
pub fn options_bar(app: &mut PhotocraftApp, ui: &mut egui::Ui) -> bool {
    if app.distort.puppet.is_some() {
        crate::puppet_ui::options_bar(app, ui);
        return true;
    }
    if app.distort.perspective.is_some() {
        crate::perspective_ui::options_bar(app, ui);
        return true;
    }
    if app.distort.vanishing.is_some() {
        crate::vp_ui::options_bar(app, ui);
        return true;
    }
    false
}

/// Full-window dialogs (Liquify, Filter Gallery).
pub fn show(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if app.distort.gallery.is_some() {
        crate::gallery_ui::show(app, ctx);
    }
    if app.distort.liquify.is_some() {
        crate::liquify_ui::show(app, ctx);
    }
}

/// Screen distance (document px) that counts as "on" a handle at the current zoom.
pub(crate) fn tolerance(app: &PhotocraftApp) -> f64 {
    let z = app.point_zoom().max(0.01);
    8.0 / f64::from(z)
}
