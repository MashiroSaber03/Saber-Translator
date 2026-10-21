//! Content-Aware Move drags and its Transform On Drop box shown with the result: once the pointer
//! (or the box) rests for a moment, the move is computed as a background job
//! ([`photocraft_engine::retouch_cmds::content_aware_move_preview`]: a smaller sampling window than
//! the command and only small selections) and the canvas shows the document with it. Any change of
//! the offset, scale or rotation cancels the job and starts the wait again, so the UI never waits
//! for the blend. Releasing (or Enter) runs the exact `paint.contentAwareMove`; the document is
//! untouched until then. Not available on wasm (no threads).

use std::sync::Arc;
use std::sync::mpsc::{Receiver, TryRecvError};

use photocraft_doc::{DocId, Document};
use photocraft_engine::jobs::JobCtx;
use serde_json::Value;

use crate::PhotocraftApp;
use crate::state::Tool;

/// Preview keys: `BASE + n`, one per result shown (see `canvas::display_doc`).
const BASE: u64 = 1 << 45;
/// How long the move must stay unchanged before a preview is computed.
const DEBOUNCE_MS: f64 = 150.0;
/// How often a running job is polled for its result.
const POLL_MS: u64 = 40;

struct Running {
    key: String,
    ctx: JobCtx,
    rx: Receiver<photocraft_engine::Result<Document>>,
}

pub(crate) struct CamPreview {
    doc: DocId,
    revision: u64,
    /// The params the move has now, and when they last changed (ms).
    key: String,
    since: f64,
    /// Params whose preview failed (too large a selection, say): not tried again.
    failed: Option<String>,
    running: Option<Running>,
    /// The latest finished preview, with the params it shows.
    shown: Option<(String, Arc<Document>)>,
    serial: u64,
}

impl CamPreview {
    fn stop(&mut self) {
        if let Some(r) = self.running.take() {
            r.ctx.cancel();
        }
    }
}

impl Drop for CamPreview {
    fn drop(&mut self) {
        self.stop();
    }
}

/// `paint.contentAwareMove`'s params for what the pointer or the box has now; `None` without a
/// move to preview.
fn wanted(app: &mut PhotocraftApp) -> Option<Value> {
    if app.ui.tool != Tool::ContentAwareMove || cfg!(target_arch = "wasm32") {
        return None;
    }
    if let Some(t) = &app.ui.cam_transform {
        let place = crate::cam_transform::placement(t)?;
        if place.offset == [0, 0] && place.is_plain_move() {
            return None;
        }
        return Some(crate::cam_transform::params(app, &place));
    }
    let d = app.drag.as_ref().filter(|d| d.tool == Tool::ContentAwareMove)?;
    let (start, mods) = (d.start, d.modifiers);
    let end = d.points.last().map_or(start, |p| [p[0], p[1]]);
    if !crate::retouch_ui::patch_drags_selection(app, start, mods) {
        return None;
    }
    let off = crate::retouch_ui::patch_offset(app, start, end);
    (off != [0, 0]).then(|| crate::retouch_ui::content_aware_move_params(app, off.map(i64::from)))
}

/// The document to show for the move on document `idx`: with the move applied, once its preview
/// has been computed for the current offset and transform.
pub(crate) fn display_doc(app: &mut PhotocraftApp, idx: usize) -> Option<(Arc<Document>, u64)> {
    if app.session.active_index() != Some(idx) {
        return None;
    }
    step(app, crate::gpu_canvas::now_ms())
}

fn step(app: &mut PhotocraftApp, now: f64) -> Option<(Arc<Document>, u64)> {
    let Some(params) = wanted(app) else {
        app.cam_preview = None;
        return None;
    };
    let st = app.session.active()?;
    let (doc, revision) = (st.doc.id, st.revision);
    let key = params.to_string();
    if app.cam_preview.as_ref().is_some_and(|p| p.doc != doc || p.revision != revision) {
        app.cam_preview = None;
    }
    let p = app.cam_preview.get_or_insert_with(|| CamPreview {
        doc,
        revision,
        key: String::new(),
        since: now,
        failed: None,
        running: None,
        shown: None,
        serial: 0,
    });
    if p.key != key {
        p.key = key.clone();
        p.since = now;
        // What is computing is already out of date.
        if p.running.as_ref().is_some_and(|r| r.key != key) {
            p.stop();
        }
    }
    if let Some(r) = &p.running {
        match r.rx.try_recv() {
            Ok(result) => {
                let r = p.running.take();
                match (result, r) {
                    (Ok(d), Some(r)) => {
                        // Duotone documents display through their inks.
                        let d = photocraft_engine::mode_cmds::display_document(&d).unwrap_or(d);
                        p.shown = Some((r.key, Arc::new(d)));
                        p.serial += 1;
                    }
                    (_, Some(r)) => p.failed = Some(r.key),
                    _ => {}
                }
            }
            Err(TryRecvError::Disconnected) => {
                if let Some(r) = p.running.take() {
                    p.failed = Some(r.key);
                }
            }
            Err(TryRecvError::Empty) => {}
        }
    }
    let settled = now - p.since >= DEBOUNCE_MS;
    let done = p.shown.as_ref().is_some_and(|s| s.0 == key);
    if settled && !done && p.running.is_none() && p.failed.as_deref() != Some(&key) {
        match start(app, &params, key.clone()) {
            Some(r) => app.cam_preview.as_mut()?.running = Some(r),
            None => app.cam_preview.as_mut()?.failed = Some(key.clone()),
        }
    }
    let p = app.cam_preview.as_ref()?;
    // Only a preview of the move as it is now: an older one would show a wrong place.
    let (_, d) = p.shown.as_ref().filter(|s| s.0 == key)?;
    Some((d.clone(), BASE + p.serial))
}

/// Compute the preview of `params` on a worker thread.
fn start(app: &PhotocraftApp, params: &Value, key: String) -> Option<Running> {
    let job = photocraft_engine::retouch_cmds::content_aware_move_preview(&app.session, params).ok()?;
    let ctx = JobCtx::new();
    let (tx, rx) = std::sync::mpsc::channel();
    let worker = ctx.clone();
    std::thread::Builder::new()
        .name("cam-preview".into())
        .spawn(move || {
            let _ = tx.send(job.run(&worker).map(|(d, _)| d));
        })
        .ok()?;
    Some(Running { key, ctx, rx })
}

/// Keep frames coming while a preview waits out its debounce or computes.
pub(crate) fn wake(app: &PhotocraftApp, ctx: &egui::Context) {
    let Some(p) = &app.cam_preview else { return };
    if p.running.is_some() {
        ctx.request_repaint_after(std::time::Duration::from_millis(POLL_MS));
    } else if p.shown.as_ref().is_none_or(|s| s.0 != p.key) && p.failed.as_deref() != Some(&p.key) {
        let left = (DEBOUNCE_MS - (crate::gpu_canvas::now_ms() - p.since)).max(1.0);
        ctx.request_repaint_after(std::time::Duration::from_millis(left as u64));
    }
}

/// Whether the canvas shows a preview.
pub(crate) fn showing(app: &PhotocraftApp) -> bool {
    app.cam_preview.as_ref().is_some_and(|p| p.shown.as_ref().is_some_and(|s| s.0 == p.key))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::{ToolEvent, tool_event};
    use serde_json::json;

    fn blue(d: &Document, x: i32, y: i32) -> bool {
        d.layers[0].surface().unwrap().rgba(x, y)[0] < 0.5
    }

    #[test]
    fn the_drag_previews_the_move_after_it_rests_and_never_touches_the_document() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 100, "height": 60})).unwrap();
        app.run("paint.pencil", json!({"points": [[16, 30], [28, 30]], "size": 4, "color": "#0000ff"})).unwrap();
        app.run("select.lasso", json!({"points": [[6, 14], [38, 14], [38, 46], [6, 46]]})).unwrap();
        app.ui.tool = Tool::ContentAwareMove;
        let m = egui::Modifiers::NONE;
        let steps = app.session.active().unwrap().history.past_len();

        // Nothing to preview before the patch moves.
        tool_event(&mut app, ToolEvent::Down { x: 22.0, y: 30.0, pressure: 1.0 }, m);
        assert!(step(&mut app, 0.0).is_none() && app.cam_preview.is_none());
        tool_event(&mut app, ToolEvent::Move { x: 72.0, y: 30.0, pressure: 1.0 }, m);
        // Still moving: the wait starts and nothing is computed or shown yet.
        assert!(step(&mut app, 1000.0).is_none());
        assert!(step(&mut app, 1000.0 + DEBOUNCE_MS / 2.0).is_none() && app.cam_preview.as_ref().unwrap().running.is_none());
        // At rest: the job runs, off this thread, and its result shows once it arrives.
        let mut now = 1000.0 + DEBOUNCE_MS;
        let mut shown = step(&mut app, now);
        for _ in 0..600 {
            if shown.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
            now += 10.0;
            shown = step(&mut app, now);
        }
        let (preview, key) = shown.expect("a preview once the pointer rested");
        assert!(key > BASE && showing(&app));
        assert!(blue(&preview, 72, 30) && !blue(&preview, 22, 30), "the preview shows the move");
        let doc = app.session.active().unwrap().doc.clone();
        assert!(blue(&doc, 22, 30) && !blue(&doc, 72, 30), "the document is untouched");
        assert_eq!(app.session.active().unwrap().history.past_len(), steps);

        // Moving on drops the stale preview at once and cancels what is not shown yet.
        tool_event(&mut app, ToolEvent::Move { x: 60.0, y: 30.0, pressure: 1.0 }, m);
        assert!(step(&mut app, now + 5.0).is_none() && !showing(&app));
        // Release commits the exact move; the preview is gone.
        tool_event(&mut app, ToolEvent::Up { x: 60.0, y: 30.0 }, m);
        assert!(step(&mut app, now + 10.0).is_none() && app.cam_preview.is_none());
    }

    #[test]
    fn the_transform_on_drop_box_previews_the_dropped_patch() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 100, "height": 60})).unwrap();
        app.run("paint.pencil", json!({"points": [[16, 30], [28, 30]], "size": 4, "color": "#0000ff"})).unwrap();
        app.run("select.lasso", json!({"points": [[6, 14], [38, 14], [38, 46], [6, 46]]})).unwrap();
        app.ui.tool = Tool::ContentAwareMove;
        app.ui.tool_options.cam_transform_on_drop = true;
        let m = egui::Modifiers::NONE;
        tool_event(&mut app, ToolEvent::Down { x: 22.0, y: 30.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Move { x: 72.0, y: 30.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Up { x: 72.0, y: 30.0 }, m);
        assert!(crate::cam_transform::active(&app));
        let (mut now, mut shown) = (0.0, step(&mut app, 0.0));
        for _ in 0..800 {
            if shown.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
            now += 10.0;
            shown = step(&mut app, now);
        }
        let (preview, _) = shown.expect("the box shows the result");
        assert!(blue(&preview, 72, 30) && !blue(&preview, 22, 30));
        // Cancelling the box drops the preview.
        crate::cam_transform::cancel(&mut app);
        assert!(step(&mut app, now + 1.0).is_none() && app.cam_preview.is_none());
    }

    #[test]
    fn a_selection_too_large_to_preview_is_not_retried() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 600, "height": 400})).unwrap();
        app.run("select.rect", json!({"x": 10, "y": 10, "width": 300, "height": 300})).unwrap();
        app.ui.tool = Tool::ContentAwareMove;
        let m = egui::Modifiers::NONE;
        tool_event(&mut app, ToolEvent::Down { x: 100.0, y: 100.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Move { x: 140.0, y: 100.0, pressure: 1.0 }, m);
        assert!(step(&mut app, 0.0).is_none());
        assert!(step(&mut app, DEBOUNCE_MS + 1.0).is_none());
        let p = app.cam_preview.as_ref().unwrap();
        assert!(p.running.is_none() && p.failed.as_deref() == Some(p.key.as_str()));
    }
}
