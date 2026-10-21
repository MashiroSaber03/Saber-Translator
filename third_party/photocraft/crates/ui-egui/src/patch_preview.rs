//! Patch Tool drags shown live: while the patch is dragged, the canvas shows the document healed
//! at the pointer's offset ([`photocraft_engine::retouch_cmds::patch_preview`]) and recomposites
//! only the healed area. The preview solves the colour fit coarser for large selections, so it
//! keeps up with the pointer; releasing commits the exact `paint.patch`, whose damage rect then
//! refreshes the same area.

use std::sync::Arc;

use photocraft_doc::{DocId, Document};
use photocraft_geom::Rect;
use serde_json::json;

use crate::PhotocraftApp;
use crate::state::Tool;

/// Preview keys of patch drags: `BASE + n`, one per offset shown (see `canvas::display_doc`).
const BASE: u64 = 1 << 38;
/// Cells of the preview's coarse solve: a 128² solve takes about 5 ms (`bench_retouch`), whatever
/// the selection's size.
const MAX_CELLS: u64 = 128 * 128;

pub(crate) struct PatchPreview {
    doc: DocId,
    revision: u64,
    mode: String,
    /// Offsets shown so far, by preview key (`BASE + 1 + index`), with the area each changed.
    offsets: Vec<([i32; 2], Rect)>,
    /// The document at the latest offset.
    shown: Option<Arc<Document>>,
}

/// The offset of the current Patch drag, if one is dragging the patch (not drawing a lasso).
fn drag_offset(app: &mut PhotocraftApp) -> Option<[i32; 2]> {
    // Content-Aware patches are background jobs: the canvas shows the result when it is done.
    if app.ui.tool_options.patch_content_aware {
        return None;
    }
    let d = app.drag.as_ref().filter(|d| d.tool == Tool::Patch)?;
    let (start, mods) = (d.start, d.modifiers);
    let end = d.points.last().map_or(start, |p| [p[0], p[1]]);
    if !crate::retouch_ui::patch_drags_selection(app, start, mods) {
        return None;
    }
    Some(crate::retouch_ui::patch_offset(app, start, end))
}

/// The document to show while the patch is dragged on document `idx`: healed at the pointer.
/// `None` without such a drag, at offset (0, 0), or when the patch can't be applied there.
pub(crate) fn display_doc(app: &mut PhotocraftApp, idx: usize) -> Option<(Arc<Document>, u64)> {
    if app.session.active_index() != Some(idx) {
        return None;
    }
    let Some(offset) = drag_offset(app) else {
        app.patch_preview = None;
        return None;
    };
    let st = app.session.documents().get(idx)?;
    let (doc_id, revision) = (st.doc.id, st.revision);
    let mode = app.ui.tool_options.patch_mode.clone();
    let fresh = app.patch_preview.as_ref().is_some_and(|p| p.doc == doc_id && p.revision == revision && p.mode == mode);
    if !fresh {
        app.patch_preview = Some(PatchPreview { doc: doc_id, revision, mode: mode.clone(), offsets: Vec::new(), shown: None });
    }
    if offset == [0, 0] {
        return None;
    }
    if app.patch_preview.as_ref()?.offsets.last().map(|o| o.0) != Some(offset) {
        let t0 = crate::gpu_canvas::now_ms();
        // Zoomed out, a finer colour fit than the screen shows is wasted.
        let min_step = (1.0 / app.point_zoom().max(1e-3)).floor().clamp(1.0, 64.0) as u32;
        let params = json!({"offset": offset, "mode": mode, "target": crate::canvas::paint_target(app)});
        let result = photocraft_engine::retouch_cmds::patch_preview(&app.session, &params, MAX_CELLS, min_step);
        let p = app.patch_preview.as_mut()?;
        match result {
            Ok((d, area)) => {
                // Duotone documents display through their inks.
                let d = photocraft_engine::mode_cmds::display_document(&d).unwrap_or(d);
                p.shown = Some(Arc::new(d));
                p.offsets.push((offset, area));
            }
            Err(_) => {
                p.shown = None;
                return None;
            }
        }
        app.perf.span("patch preview", crate::gpu_canvas::now_ms() - t0);
    }
    let p = app.patch_preview.as_ref()?;
    Some((p.shown.clone()?, BASE + p.offsets.len() as u64))
}

/// What changed between preview (or document) key `seen` and `now` of the current Patch drag on
/// document `doc` at `revision`: the areas both healed. `None` when either key isn't one of this
/// drag's (the canvas then recomposites everything).
pub(crate) fn damage(app: &PhotocraftApp, doc: DocId, revision: u64, seen: u64, now: u64) -> Option<Rect> {
    let p = app.patch_preview.as_ref().filter(|p| p.doc == doc && p.revision == revision)?;
    let ours = |k: u64| k == 0 || is_preview_key(k);
    if seen == now || !ours(seen) || !ours(now) {
        return None;
    }
    let area = |k: u64| if k == 0 { Some(Rect::EMPTY) } else { p.offsets.get(usize::try_from(k - BASE - 1).ok()?).map(|o| o.1) };
    let (a, b) = (area(seen)?, area(now)?);
    Some(if a.is_empty() {
        b
    } else if b.is_empty() {
        a
    } else {
        a.union(&b)
    })
}

/// Whether the canvas shows a Patch drag's healed preview.
pub(crate) fn showing(app: &PhotocraftApp) -> bool {
    app.patch_preview.as_ref().is_some_and(|p| p.shown.is_some())
}

/// Whether `key` is a Patch drag preview key.
pub(crate) fn is_preview_key(key: u64) -> bool {
    key > BASE && key < BASE << 1
}

/// Release: after `paint.patch` at `offset` succeeded. If the canvas showed this very offset, its
/// caches count the preview as the document, so the commit's damage rect (the same healed area,
/// now solved exactly) refreshes only that area instead of everything.
pub(crate) fn committed(app: &mut PhotocraftApp, preview: Option<PatchPreview>, offset: [i32; 2]) {
    let Some(p) = preview.filter(|p| p.shown.is_some() && p.offsets.last().map(|o| o.0) == Some(offset)) else { return };
    crate::canvas::shown_as_document(app, p.doc, is_preview_key);
}

/// Take the preview off the app (at release, before the commit changes the document).
pub(crate) fn take(app: &mut PhotocraftApp) -> Option<PatchPreview> {
    app.patch_preview.take()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::{ToolEvent, tool_event};

    fn red_at(d: &Document, x: i32, y: i32) -> bool {
        d.layers[0].surface().unwrap().rgba(x, y)[1] < 0.5
    }

    #[test]
    fn dragging_the_patch_previews_the_heal_and_release_commits_it() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 100, "height": 60})).unwrap();
        app.run("paint.pencil", json!({"points": [[68, 30], [74, 30]], "size": 6, "color": "#ff0000"})).unwrap();
        app.run("select.lasso", json!({"points": [[60, 20], [84, 20], [84, 40], [60, 40]]})).unwrap();
        app.ui.tool = Tool::Patch;
        let m = egui::Modifiers::NONE;
        let doc = |app: &PhotocraftApp| app.session.documents()[0].doc.clone();
        let (id, rev) = (doc(&app).id, app.session.documents()[0].revision);

        // Nothing to preview before the patch moves, or while a lasso is drawn.
        tool_event(&mut app, ToolEvent::Down { x: 70.0, y: 30.0, pressure: 1.0 }, m);
        assert!(display_doc(&mut app, 0).is_none());
        tool_event(&mut app, ToolEvent::Move { x: 50.0, y: 30.0, pressure: 1.0 }, m);
        let (shown, k1) = display_doc(&mut app, 0).expect("a preview while dragging");
        assert!(is_preview_key(k1) && showing(&app));
        assert!(!red_at(&shown, 71, 30), "the preview shows the blemish healed");
        assert!(red_at(&doc(&app), 71, 30), "the document is untouched until release");
        let a = damage(&app, id, rev, 0, k1).unwrap();
        assert!(a.contains(71, 30) && !a.contains(20, 30), "only the healed area recomposites: {a:?}");
        // Same offset: the same preview, no new solve.
        assert_eq!(display_doc(&mut app, 0).unwrap().1, k1);
        tool_event(&mut app, ToolEvent::Move { x: 30.0, y: 30.0, pressure: 1.0 }, m);
        let k2 = display_doc(&mut app, 0).unwrap().1;
        assert!(k2 != k1 && damage(&app, id, rev, k1, k2).is_some_and(|r| r.contains(71, 30)));
        assert!(damage(&app, id, rev, k1, 12345).is_none(), "foreign keys recomposite everything");

        tool_event(&mut app, ToolEvent::Up { x: 30.0, y: 30.0 }, m);
        assert!(!app.ui.status_error, "{}", app.ui.status);
        assert!(!red_at(&doc(&app), 71, 30), "release commits the patch");
        assert!(display_doc(&mut app, 0).is_none() && !showing(&app));

        // A drag starting outside the selection draws a lasso: no preview.
        tool_event(&mut app, ToolEvent::Down { x: 5.0, y: 5.0, pressure: 1.0 }, m);
        tool_event(&mut app, ToolEvent::Move { x: 30.0, y: 10.0, pressure: 1.0 }, m);
        assert!(display_doc(&mut app, 0).is_none());
    }
}
