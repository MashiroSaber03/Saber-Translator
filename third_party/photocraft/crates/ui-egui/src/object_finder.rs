//! Object Selection › Object Finder (#4087): with the Object Selection tool active, resting the
//! pointer on the canvas highlights the object a click there would select, as Photoshop's Object
//! Finder does. A click (without dragging) commits that object (`select.object` with `"point"`;
//! ⇧ adds, ⌥ subtracts, ⇧⌥ intersects); dragging keeps the rectangle mode.
//!
//! The lookup is `photocraft_engine::smartselect_cmds::find_object`, run after the pointer has
//! been still for [`DEBOUNCE`] seconds on a worker thread (inline on the web), so the UI never
//! waits for it. Results go into the session's `object_finder` cache, which the click reuses and
//! which serves every point of an object already found; it is keyed by the document's pixels,
//! so an edit drops it. The preview itself is keyed by document revision, sampled layer and
//! Sample All Layers, and clears when the pointer leaves the canvas, the tool changes, a button
//! goes down or the option is off. Quality of the object found is out of scope here (#2788).

use std::sync::Arc;

use egui::{Color32, TextureOptions, pos2};
use photocraft_algo::selection::Region;
use photocraft_doc::{DocId, Document, LayerId};
use photocraft_engine::smartselect_cmds::{find_object, sampled_target};

use crate::PhotocraftApp;
use crate::canvas::ViewXform;
use crate::state::Tool;

/// Seconds the pointer must rest before an object is looked up.
pub const DEBOUNCE: f64 = 0.12;
/// Longest side of the overlay texture (the overlay is a preview, not the selection).
const TEX_MAX: usize = 768;

/// A lookup running off the UI thread.
struct Job {
    at: [i32; 2],
    doc: Arc<Document>,
    rx: std::sync::mpsc::Receiver<Option<Region>>,
}

/// The hover state of the active document (`PhotocraftApp::object_finder`).
pub(crate) struct Finder {
    doc: DocId,
    revision: u64,
    target: Option<LayerId>,
    all_layers: bool,
    /// The pointer pixel and when it got there.
    at: [i32; 2],
    since: f64,
    job: Option<Job>,
    /// Points already looked up that found nothing (the background).
    misses: Vec<[i32; 2]>,
    shown: Option<Arc<Region>>,
    texture: Option<(usize, egui::TextureHandle)>,
}

/// What [`update`] decided.
#[derive(Clone, Debug, Default)]
pub(crate) struct Update {
    /// The object to highlight.
    pub shown: Option<Arc<Region>>,
    /// A lookup is pending or running: repaint soon to pick it up.
    pub busy: bool,
}

/// Whether the preview applies now (`pressed`: a pointer button is down).
fn active(app: &PhotocraftApp, pressed: bool) -> bool {
    app.ui.tool == Tool::ObjectSelection
        && app.ui.tool_options.object_finder
        && !pressed
        && app.drag.is_none()
        && app.ui.transform.is_none()
        && app.session.active().is_some()
}

/// Advances the hover preview with the pointer at `at` (document px; `None` = off the canvas)
/// at time `now` (seconds).
pub(crate) fn update(app: &mut PhotocraftApp, at: Option<[f64; 2]>, pressed: bool, now: f64) -> Update {
    let at = at.filter(|p| p[0].is_finite() && p[1].is_finite());
    let (Some(at), true) = (at, active(app, pressed)) else {
        app.object_finder = None;
        return Update::default();
    };
    let Some(st) = app.session.active() else { return Update::default() };
    let all_layers = app.ui.tool_options.sample_all_layers;
    let target = sampled_target(&st.doc, st.active_layer, all_layers);
    let (doc, revision, layer) = (st.doc.id, st.revision, st.active_layer);
    let px = [at[0].floor() as i32, at[1].floor() as i32];
    let fresh = app.object_finder.as_ref().is_some_and(|f| f.doc == doc && f.revision == revision && f.target == target && f.all_layers == all_layers);
    if !fresh {
        app.object_finder = Some(Finder { doc, revision, target, all_layers, at: px, since: now, job: None, misses: Vec::new(), shown: None, texture: None });
    }
    let doc_arc = Arc::clone(&st.doc);
    let Some(f) = app.object_finder.as_mut() else { return Update::default() };
    if f.at != px {
        f.at = px;
        f.since = now;
    }
    // A finished lookup: remember it (only for the pixels it was computed from).
    if let Some(job) = &f.job {
        match job.rx.try_recv() {
            Ok(found) => {
                if let Some(r) = found {
                    if photocraft_engine::layer_multi_cmds::same_pixels(&job.doc, &doc_arc) {
                        app.session.object_finder.insert(&doc_arc, target, Arc::new(r));
                    }
                } else {
                    f.misses.push(job.at);
                    if f.misses.len() > 64 {
                        f.misses.remove(0);
                    }
                }
                f.job = None;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => f.job = None,
        }
    }
    let inside = |r: &Region| r.at(px[0], px[1]) >= 0.5;
    if !f.shown.as_deref().is_some_and(inside) {
        f.shown = app.session.object_finder.lookup(&doc_arc, target, at[0] as f32, at[1] as f32);
    }
    let missed = f.misses.iter().any(|m| (m[0] - px[0]).abs() <= 2 && (m[1] - px[1]).abs() <= 2);
    if f.shown.is_none() && f.job.is_none() && !missed && now - f.since >= DEBOUNCE && doc_arc.bounds().contains(px[0], px[1]) {
        f.job = Some(spawn(Arc::clone(&doc_arc), layer, all_layers, px, at));
    }
    let busy = f.job.is_some() || (f.shown.is_none() && !missed && now - f.since < DEBOUNCE);
    Update { shown: f.shown.clone(), busy }
}

fn spawn(doc: Arc<Document>, layer: Option<LayerId>, all_layers: bool, px: [i32; 2], at: [f64; 2]) -> Job {
    let (tx, rx) = std::sync::mpsc::channel();
    let work = {
        let doc = Arc::clone(&doc);
        move || {
            let _ = tx.send(find_object(&doc, layer, all_layers, at[0] as f32, at[1] as f32));
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        let spawned = std::thread::Builder::new().name("object-finder".into()).spawn(work);
        if spawned.is_err() {
            // No thread: the receiver reports Disconnected and the next rest retries.
        }
    }
    #[cfg(target_arch = "wasm32")]
    work();
    Job { at: px, doc, rx }
}

/// Clears the preview (a tool switch or an edit made elsewhere is caught by [`update`] too).
pub(crate) fn clear(app: &mut PhotocraftApp) {
    app.object_finder = None;
}

/// The overlay colour: the theme accent, translucent (Photoshop tints the object it finds).
fn overlay(accent: Color32, cov: f32, edge: bool) -> Color32 {
    let a = if edge { 0.95 } else { 0.42 * cov.clamp(0.0, 1.0) };
    Color32::from_rgba_unmultiplied(accent.r(), accent.g(), accent.b(), (a * 255.0).round() as u8)
}

/// The overlay image of `r`: translucent fill with an opaque one-texel outline, at most
/// [`TEX_MAX`] texels on its longest side.
pub(crate) fn overlay_image(r: &Region, accent: Color32) -> egui::ColorImage {
    let (w, h) = (r.bbox.width().max(1) as usize, r.bbox.height().max(1) as usize);
    let step = w.max(h).div_ceil(TEX_MAX).max(1);
    let (tw, th) = (w.div_ceil(step), h.div_ceil(step));
    let cov = |tx: usize, ty: usize| r.at(r.bbox.x0 + (tx * step) as i32, r.bbox.y0 + (ty * step) as i32);
    let mut px = Vec::with_capacity(tw * th);
    for ty in 0..th {
        for tx in 0..tw {
            let c = cov(tx, ty);
            let on = c >= 0.5;
            let edge = on
                && (tx == 0
                    || ty == 0
                    || tx + 1 == tw
                    || ty + 1 == th
                    || cov(tx - 1, ty) < 0.5
                    || cov(tx + 1, ty) < 0.5
                    || cov(tx, ty - 1) < 0.5
                    || cov(tx, ty + 1) < 0.5);
            px.push(if c <= 0.0 { Color32::TRANSPARENT } else { overlay(accent, c, edge) });
        }
    }
    egui::ColorImage::new([tw, th], px)
}

/// Updates the preview for this frame and draws the highlight on the active document's canvas.
pub(crate) fn draw(app: &mut PhotocraftApp, painter: &egui::Painter, xf: &ViewXform, idx: usize, hover: Option<egui::Pos2>) {
    if app.session.active_index() != Some(idx) {
        return;
    }
    let ctx = painter.ctx().clone();
    let (pressed, now) = ctx.input(|i| (i.pointer.any_down(), i.time));
    let up = update(app, hover.map(|p| xf.to_doc(p)), pressed, now);
    if up.busy {
        ctx.request_repaint_after(std::time::Duration::from_millis(40));
    }
    let Some(r) = up.shown else { return };
    let Some(f) = app.object_finder.as_mut() else { return };
    let key = Arc::as_ptr(&r) as usize;
    if f.texture.as_ref().is_none_or(|(k, _)| *k != key) {
        let accent = crate::theme::Tokens::get(&ctx).accent;
        let tex = ctx.load_texture("object-finder", overlay_image(&r, accent), TextureOptions::LINEAR);
        f.texture = Some((key, tex));
    }
    let Some((_, tex)) = &f.texture else { return };
    let b = r.bbox;
    let mut mesh = egui::Mesh::with_texture(tex.id());
    let corners = [(b.x0, b.y0, 0.0, 0.0), (b.x1, b.y0, 1.0, 0.0), (b.x1, b.y1, 1.0, 1.0), (b.x0, b.y1, 0.0, 1.0)];
    for (x, y, u, v) in corners {
        mesh.vertices.push(egui::epaint::Vertex { pos: xf.to_screen(x as f32, y as f32), uv: pos2(u, v), color: Color32::WHITE });
    }
    mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
    painter.add(egui::Shape::mesh(mesh));
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// A 200 × 160 document: a red disc (centre 100, 80, radius 45) on blue.
    pub(crate) fn disc_app() -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": 200, "height": 160})).unwrap();
        app.session
            .edit("paint", |doc, a| {
                let s = doc.layer_mut(a.unwrap()).unwrap().surface_mut().unwrap();
                s.fill_rect(photocraft_geom::Rect::new(0, 0, 200, 160), &[0.2, 0.4, 0.8, 1.0]);
                for y in 0..160 {
                    for x in 0..200 {
                        let (dx, dy) = (x as f32 + 0.5 - 100.0, y as f32 + 0.5 - 80.0);
                        if dx * dx + dy * dy <= 45.0 * 45.0 {
                            s.fill_rect(photocraft_geom::Rect::new(x, y, x + 1, y + 1), &[0.9, 0.2, 0.1, 1.0]);
                        }
                    }
                }
                Ok(())
            })
            .unwrap();
        app.ui.tool = Tool::ObjectSelection;
        app
    }

    /// Hovers at `at` until the lookup settles (time advances past the debounce).
    pub(crate) fn hover(app: &mut PhotocraftApp, at: [f64; 2]) -> Update {
        let mut t = 100.0;
        loop {
            let u = update(app, Some(at), false, t);
            if !u.busy || t > 130.0 {
                return u;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
            t += 0.05;
        }
    }

    #[test]
    fn hovering_an_object_highlights_it_after_a_rest() {
        let mut app = disc_app();
        // Not before the pointer has rested.
        let u = update(&mut app, Some([100.0, 80.0]), false, 50.0);
        assert!(u.shown.is_none() && u.busy);
        let u = hover(&mut app, [100.0, 80.0]);
        let r = u.shown.expect("the disc is highlighted");
        assert!(r.at(100, 80) >= 0.5 && r.at(60, 80) >= 0.5 && r.at(5, 5) < 0.5);
        // Elsewhere on the same object: served from the cache, no new lookup.
        let u = update(&mut app, Some([120.0, 90.0]), false, 200.0);
        assert!(u.shown.is_some_and(|s| Arc::ptr_eq(&s, &r)) && !u.busy);
        assert_eq!(app.session.object_finder.len(), 1);
        // The background highlights nothing.
        assert!(hover(&mut app, [4.0, 4.0]).shown.is_none());
    }

    #[test]
    fn preview_clears_on_leave_tool_switch_button_option_and_edits() {
        let mut app = disc_app();
        assert!(hover(&mut app, [100.0, 80.0]).shown.is_some());
        assert!(update(&mut app, None, false, 300.0).shown.is_none());
        assert!(app.object_finder.is_none());
        assert!(hover(&mut app, [100.0, 80.0]).shown.is_some());
        assert!(update(&mut app, Some([100.0, 80.0]), true, 300.0).shown.is_none());
        app.ui.tool = Tool::Lasso;
        assert!(update(&mut app, Some([100.0, 80.0]), false, 300.0).shown.is_none());
        app.ui.tool = Tool::ObjectSelection;
        app.ui.tool_options.object_finder = false;
        assert!(update(&mut app, Some([100.0, 80.0]), false, 300.0).shown.is_none());
        app.ui.tool_options.object_finder = true;
        assert!(hover(&mut app, [100.0, 80.0]).shown.is_some());
        // A pixel edit: the cached object is dropped and the preview starts over.
        app.session.execute("paint.pencil", json!({"points": [[5, 5], [30, 5]], "size": 4, "color": "#000000"})).unwrap();
        let u = update(&mut app, Some([100.0, 80.0]), false, 400.0);
        assert!(u.shown.is_none() && u.busy);
    }

    #[test]
    fn click_commits_the_highlighted_object_with_the_modifiers() {
        use crate::canvas::{ToolEvent, tool_event};
        let mut app = disc_app();
        assert!(hover(&mut app, [100.0, 80.0]).shown.is_some());
        let click = |app: &mut PhotocraftApp, m: egui::Modifiers| {
            tool_event(app, ToolEvent::Down { x: 100.0, y: 80.0, pressure: 1.0 }, m);
            tool_event(app, ToolEvent::Up { x: 100.0, y: 80.0 }, m);
        };
        let cov = |app: &PhotocraftApp, x: i32, y: i32| app.session.active().unwrap().doc.selection.as_ref().map_or(0.0, |s| s.sample_channel(x, y, 0));
        click(&mut app, egui::Modifiers::NONE);
        assert_eq!(app.session.journal.last().map(|(id, _)| id.as_str()), Some("select.object"));
        assert!(cov(&app, 100, 80) >= 0.5 && cov(&app, 60, 80) >= 0.5 && cov(&app, 5, 5) < 0.5);
        app.run("select.all", json!({})).unwrap();
        click(&mut app, egui::Modifiers::ALT);
        assert!(cov(&app, 100, 80) < 0.5 && cov(&app, 5, 5) >= 0.5);
        click(&mut app, egui::Modifiers::SHIFT);
        assert!(cov(&app, 100, 80) >= 0.5 && cov(&app, 5, 5) >= 0.5);
        click(&mut app, egui::Modifiers { alt: true, shift: true, ..Default::default() });
        assert!(cov(&app, 100, 80) >= 0.5 && cov(&app, 5, 5) < 0.5);
        // Off: a click keeps the Quick Selection fallback.
        app.ui.tool_options.object_finder = false;
        click(&mut app, egui::Modifiers::NONE);
        assert_eq!(app.session.journal.last().map(|(id, _)| id.as_str()), Some("select.quick"));
    }

    #[test]
    fn overlay_image_tints_the_object_and_outlines_it() {
        let r = Region { bbox: photocraft_geom::Rect::new(0, 0, 4, 4), mask: vec![255; 16] };
        let img = overlay_image(&r, Color32::from_rgb(0, 120, 255));
        assert_eq!(img.size, [4, 4]);
        assert!(img.pixels[0].a() > 200, "edge");
        assert!(img.pixels[5].a() > 0 && img.pixels[5].a() < 200, "fill");
        let big = Region { bbox: photocraft_geom::Rect::new(0, 0, 3000, 10), mask: vec![255; 30_000] };
        assert!(overlay_image(&big, Color32::WHITE).size[0] <= TEX_MAX);
    }
}
