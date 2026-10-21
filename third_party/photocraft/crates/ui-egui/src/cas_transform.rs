//! Edit › Content-Aware Scale as Photoshop has it (#3096): no dialog, a transform box on the layer
//! (or the selection) that only scales and moves. The options bar has the reference point X/Y,
//! W/H, Amount, Protect (an alpha channel) and Protect Skin Tones. While the box changes, the
//! command's own seam carving runs on a reduced copy in the background and its latest result is
//! drawn through the box; ↩ runs `edit.contentAwareScale` with the box's size and position (one
//! history step) and Esc cancels.
//!
//! With params (actions, the CLI, MCP, the control channel) the command runs directly.

use std::sync::mpsc::{Receiver, TryRecvError};

use egui::ColorImage;
use photocraft_engine::edit_menu_cmds::ContentAwareScalePreview;
use photocraft_engine::jobs::JobCtx;
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::state::{ContentAwareOptions, TransformSession};
use crate::transform_tex::PreviewTextures;

pub const COMMAND: &str = "edit.contentAwareScale";

/// Long side of the live preview's reduced copy, px: carving it takes milliseconds, so the
/// preview follows a drag.
const PROXY_SIDE: u32 = 384;

/// How long a preview may run for a box that has changed since before it is cancelled. Shorter
/// would cancel every preview during a drag, so none would show until the drag stops (the filter
/// previews' settle time).
const STALE_MS: f64 = 150.0;

/// The preview being computed: what for, its result channel and its cancellation.
struct Running {
    key: Value,
    rx: Receiver<Result<ColorImage, String>>,
    job: JobCtx,
    started_ms: f64,
}

/// The live preview of an open Content-Aware Scale box (kept with the Free Transform preview).
#[derive(Default)]
pub(crate) struct Live {
    running: Option<Running>,
    /// The latest finished preview and what it was computed for.
    shown: Option<(Value, PreviewTextures)>,
    /// Texture names must differ from one result to the next.
    serial: u64,
}

impl Drop for Live {
    fn drop(&mut self) {
        // A closed box stops its preview at the next seam.
        if let Some(r) = &self.running {
            r.job.cancel();
        }
    }
}

impl Live {
    /// The latest result to draw through the box `t`, with how many document px one of its
    /// texels covers there (so the texture picks the right level). `None` until a preview has
    /// finished, and for an unscaled box (the full-resolution pixels show then).
    pub(crate) fn shown(&self, t: &TransformSession) -> Option<(&PreviewTextures, f32)> {
        let (_, tex) = self.shown.as_ref()?;
        let [w, h] = box_size(t);
        if [w, h] == source_size(t) {
            return None;
        }
        let [tw, th] = tex.size();
        let texel = (w / tw.max(1) as f64).max(h / th.max(1) as f64);
        Some((tex, if texel.is_finite() { texel as f32 } else { 1.0 }))
    }
}

/// Edit › Content-Aware Scale from the menu: open the box (`Err` says why it can't).
pub fn begin(app: &mut PhotocraftApp, ctx: &egui::Context) -> Result<Value, String> {
    if app.ui.transform.is_some() {
        return Err("Commit or cancel the transform first".into());
    }
    if let Some(Err(why)) = photocraft_engine::commands::find(COMMAND).map(|c| (c.enabled)(&app.session)) {
        return Err(why);
    }
    // The command's own checks and area (the selection's bounds, else the layer's content).
    let region = ContentAwareScalePreview::new(&app.session, &json!({}), PROXY_SIDE).map_err(|e| e.to_string())?.region();
    crate::transform_tool::begin_content_aware(app, ctx, region, Live::default())?;
    Ok(json!({"transform": app.ui.transform}))
}

/// The box's size, document px (its bounding box: the box only scales and moves).
fn box_size(t: &TransformSession) -> [f64; 2] {
    let (x0, y0, x1, y1) = bounds(t);
    [(x1 - x0).round().max(1.0), (y1 - y0).round().max(1.0)]
}

fn source_size(t: &TransformSession) -> [f64; 2] {
    [(t.rect[2] - t.rect[0]).round().max(1.0), (t.rect[3] - t.rect[1]).round().max(1.0)]
}

fn bounds(t: &TransformSession) -> (f64, f64, f64, f64) {
    t.quad.iter().fold((f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY), |(x0, y0, x1, y1), q| {
        (x0.min(q[0]), y0.min(q[1]), x1.max(q[0]), y1.max(q[1]))
    })
}

/// `edit.contentAwareScale`'s params for the box: its size and options, and with `place` its
/// position.
pub(crate) fn params(t: &TransformSession, o: &ContentAwareOptions, place: bool) -> Value {
    let [w, h] = box_size(t);
    let mut p = json!({
        "width": w,
        "height": h,
        "amount": o.amount.clamp(0.0, 100.0),
        "protect": o.protect.as_deref().unwrap_or("none"),
        "protectSkinTones": o.protect_skin_tones,
    });
    if place {
        let (x0, y0, _, _) = bounds(t);
        p["x"] = json!(x0.round());
        p["y"] = json!(y0.round());
    }
    p
}

/// ↩: run the command for the box. An untouched box changes nothing and adds no step.
pub(crate) fn commit(app: &mut PhotocraftApp, t: &TransformSession, o: &ContentAwareOptions) {
    if t.quad == [[t.rect[0], t.rect[1]], [t.rect[2], t.rect[1]], [t.rect[2], t.rect[3]], [t.rect[0], t.rect[3]]] {
        return;
    }
    if let Err(e) = app.run(COMMAND, params(t, o, true)) {
        app.ui.status = e;
        app.ui.status_error = true;
    }
}

/// Once per frame while the box is open: take a finished preview, cancel one gone stale and start
/// one for the box as it is now.
pub fn tick(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let Some(t) = app.ui.transform.as_ref() else { return };
    let Some(o) = t.content_aware.as_ref() else { return };
    let unscaled = box_size(t) == source_size(t);
    let want = params(t, o, false);
    let now = crate::gpu_canvas::now_ms();
    let Some(live) = app.transform_preview.as_mut().and_then(|pv| pv.content_aware.as_mut()) else { return };
    if let Some(r) = &live.running {
        let finished = match r.rx.try_recv() {
            Ok(result) => Some(result),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(Err("Content-Aware Scale preview stopped".into())),
        };
        if let Some(result) = finished {
            let key = live.running.take().map(|r| r.key).unwrap_or_default();
            // A failed or cancelled preview keeps the last one; the commit reports real errors.
            if let Ok(image) = result {
                live.serial += 1;
                live.shown = Some((key, PreviewTextures::new(ctx, format!("content-aware-scale-{}", live.serial), image, [1.0, 1.0])));
            }
        } else {
            if r.key != want && now - r.started_ms >= STALE_MS {
                r.job.cancel();
            }
            ctx.request_repaint();
            return;
        }
    }
    if unscaled || live.shown.as_ref().is_some_and(|(k, _)| *k == want) {
        return;
    }
    let preview = match ContentAwareScalePreview::new(&app.session, &want, PROXY_SIDE) {
        Ok(p) => p,
        Err(e) => {
            app.ui.status = e.to_string();
            return;
        }
    };
    let Some(live) = app.transform_preview.as_mut().and_then(|pv| pv.content_aware.as_mut()) else { return };
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    let job = JobCtx::new();
    let work = job.clone();
    let repaint = ctx.clone();
    let compute = move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let surf = preview.run(&work).map_err(|e| e.to_string())?;
            let [w, h] = preview.size();
            Ok(crate::transform_tex::read_surface(&surf, None, photocraft_geom::Rect::from_xywh(0, 0, w, h), usize::MAX).0)
        }))
        .unwrap_or_else(|_| Err("Content-Aware Scale preview failed".into()));
        let _ = tx.send(result);
        repaint.request_repaint();
    };
    // No threads (the browser): the reduced copy is small enough to carve inline.
    #[cfg(not(target_arch = "wasm32"))]
    {
        let spawned = std::thread::Builder::new().name("content-aware-scale-preview".into()).spawn(compute);
        if spawned.is_err() {
            return;
        }
    }
    #[cfg(target_arch = "wasm32")]
    compute();
    live.running = Some(Running { key: want, rx, job, started_ms: now });
    ctx.request_repaint();
}

/// The options bar while the box is open: X/Y, W/H, Amount, Protect and Protect Skin Tones.
pub fn fields(app: &mut PhotocraftApp, ui: &mut egui::Ui, t: &TransformSession) {
    let Some(o) = t.content_aware.clone() else { return };
    let tk = crate::theme::Tokens::get(ui.ctx());
    let lbl = |ui: &mut egui::Ui, s: &str| {
        ui.label(egui::RichText::new(s).color(tk.text_dim).size(12.0));
    };
    crate::transform_tool::position_size_fields(app, ui, t);
    crate::widgets::vline(ui, 22.0);
    let mut next = o.clone();
    lbl(ui, tl!("Amount:"));
    crate::widgets::value_field(ui, &mut next.amount, 0.0..=100.0, "%", 60.0);
    crate::widgets::vline(ui, 22.0);
    lbl(ui, tl!("Protect:"));
    let channels: Vec<String> = app.session.active().map(|st| st.doc.channels.iter().map(|c| c.name.clone()).collect()).unwrap_or_default();
    let mut options: Vec<(Option<String>, &str)> = vec![(None, tl!("None"))];
    options.extend(channels.iter().map(|n| (Some(n.clone()), n.as_str())));
    crate::widgets::dropdown(ui, "content-aware-scale-protect", &mut next.protect, &options, 120.0);
    crate::widgets::checkbox(ui, &mut next.protect_skin_tones, tl!("Protect skin tones"));
    if next != o
        && let Some(s) = app.ui.transform.as_mut()
    {
        s.content_aware = Some(next);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::ToolEvent;

    #[test]
    fn menu_opens_a_scale_box_and_enter_commits_one_step() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width": 64, "height": 48})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        app.run("select.rect", json!({"x": 8, "y": 8, "width": 40, "height": 24})).unwrap();
        app.run("edit.fill", json!({"color": "#c03020"})).unwrap();
        app.run("select.deselect", json!({})).unwrap();
        app.sync_views();
        let past = app.session.active().unwrap().history.past_len();
        // The menu opens the box, not a dialog.
        let r = crate::menus::invoke(&mut app, &ctx, COMMAND, json!({})).unwrap();
        assert!(r.get("dialog").is_none() && app.ui.dialogs.is_empty(), "{r}");
        let t = app.ui.transform.clone().unwrap();
        assert_eq!(t.rect, [8.0, 8.0, 48.0, 32.0]);
        assert!(t.content_aware.is_some());
        // Dragging outside the box doesn't rotate it; the right edge narrows it (⇧: that axis only).
        let shift = egui::Modifiers { shift: true, ..Default::default() };
        for (from, to) in [([60.0, 40.0], [62.0, 10.0]), ([48.0, 20.0], [32.0, 20.0])] {
            assert!(crate::transform_tool::pointer(&mut app, ToolEvent::Down { x: from[0], y: from[1], pressure: 1.0 }, shift));
            crate::transform_tool::pointer(&mut app, ToolEvent::Move { x: to[0], y: to[1], pressure: 1.0 }, shift);
            crate::transform_tool::pointer(&mut app, ToolEvent::Up { x: to[0], y: to[1] }, shift);
        }
        assert_eq!(app.ui.transform.as_ref().unwrap().quad, [[8.0, 8.0], [32.0, 8.0], [32.0, 32.0], [8.0, 32.0]]);
        // The live preview carves a reduced copy in the background.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while app.transform_preview.as_ref().and_then(|pv| pv.content_aware.as_ref()).is_none_or(|l| l.shown.is_none()) {
            assert!(std::time::Instant::now() < deadline, "no live preview");
            tick(&mut app, &ctx);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        // ↩ (the options bar's commit) runs the command: one step, the new size, in place.
        crate::transform_tool::commit(&mut app);
        assert!(app.ui.transform.is_none());
        let st = app.session.active().unwrap();
        assert_eq!(st.history.past_len(), past + 1);
        assert_eq!(st.history.undo_label(), Some("Content-Aware Scale"));
        let b = st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().content_bounds();
        assert_eq!([b.x0, b.y0, b.width() as i32, b.height() as i32], [8, 8, 24, 24]);
    }
}
