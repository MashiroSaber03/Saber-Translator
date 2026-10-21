//! Move tool drags shown live (#128): while the pointer drags, the canvas shows the document with
//! the moving layers already at the pointer, exactly as `layer.translate` will leave them, and
//! recomposites only where they were and where they are. Releasing commits one `layer.translate`
//! (one history step) whose damage rect refreshes the same area.
//!
//! With a selection, the Move tool drags the selected pixels instead (#578): they float
//! (`select.float`, the canvas shows the piece at the pointer) until ↩, Deselect or any other
//! command drops them, as one "Move" history step.

use std::sync::Arc;

use photocraft_doc::{DocId, Document, LayerId};
use photocraft_geom::Rect;
use serde_json::json;

use crate::PhotocraftApp;
use crate::state::Tool;

/// Does the Move tool drag (and do the arrow keys nudge) the selected pixels of the active layer
/// rather than the whole layer, with `tool` in effect? With a selection, as in Photoshop; a layer
/// whose pixels can't float (type, shape, Smart Object, group, position-locked), a hidden layer or
/// several selected layers still move whole, as without one. The canvas cursor asks before the
/// press, and a pointer event passes `canvas::event_tool`, which also sees a ⌘ sent with the event
/// (#2768).
pub(crate) fn moves_selected_pixels_with(app: &PhotocraftApp, tool: Tool) -> bool {
    let Some(st) = app.session.active() else { return false };
    tool == Tool::Move
        && app.ui.transform.is_none()
        && st.doc.selection.is_some()
        && st.selected_layers().len() <= 1
        && st.active_layer.and_then(|id| st.doc.layer(id)).is_some_and(|l| l.visible)
        && app.session.is_enabled("select.float")
}

/// Floats the selected pixels and moves them by (dx, dy) (`select.float`): cuts them the first
/// time, or with `copy` (⌥) floats a duplicate and leaves them; `copy` on a piece already floating
/// drops it where it is and floats a copy of it. False when it failed (reported in the status bar).
pub(crate) fn float_selected(app: &mut PhotocraftApp, copy: bool, dx: f64, dy: f64) -> bool {
    let floating = app.session.active().is_some_and(|st| photocraft_engine::float_cmds::floating(st).is_some());
    let dropped = if copy && floating { app.run("select.drop", json!({})) } else { Ok(json!(null)) };
    match dropped.and_then(|_| app.run("select.float", json!({"dx": dx, "dy": dy, "copy": copy}))) {
        Ok(_) => true,
        Err(e) => {
            app.ui.status = e;
            app.ui.status_error = true;
            false
        }
    }
}

/// Preview keys of move drags: `BASE + n`, one per offset shown (see `canvas::display_doc`).
const BASE: u64 = 1 << 39;

pub(crate) struct MovePreview {
    doc: DocId,
    revision: u64,
    /// Layers that move ([`photocraft_engine::layer_multi_cmds::move_targets`]).
    ids: Vec<LayerId>,
    /// `ids` and every layer inside them: the layers shown at the drag's offset.
    moving: Vec<LayerId>,
    /// What moving them can change at offset (0, 0), not clipped to the canvas: a layer larger
    /// than the canvas brings its pixels from beyond the edge into view (`None` = anything).
    bounds: Option<Rect>,
    canvas: Rect,
    /// Offsets shown so far, by preview key (`BASE + index`).
    offsets: Vec<(i32, i32)>,
    /// The document at the latest offset.
    shown: Option<Arc<Document>>,
    /// Shows a floating selection (`select.float`): offsets are the piece's, from where it was cut.
    floating: bool,
}

impl MovePreview {
    fn key(&self) -> u64 {
        BASE + self.offsets.len() as u64
    }
    fn offset_of(&self, key: u64) -> Option<(i32, i32)> {
        if key == 0 {
            return Some((0, 0));
        }
        let i = usize::try_from(key.checked_sub(BASE + 1)?).ok()?;
        self.offsets.get(i).copied()
    }
    fn area(&self, d: (i32, i32)) -> Option<Rect> {
        let b = self.bounds?;
        Some(if b.is_empty() { b } else { b.translate(d.0, d.1).inflate(1).intersect(&self.canvas) })
    }
}

/// The whole-pixel offset of the current Move drag on document `idx`, if one is under way.
fn drag_offset(app: &PhotocraftApp) -> Option<(i32, i32)> {
    let d = app.drag.as_ref().filter(|d| d.tool == Tool::Move)?;
    let end = d.points.last().map_or(d.start, |p| [p[0], p[1]]);
    let dx = (end[0] - d.start[0]).round().clamp(-1e7, 1e7) as i32;
    let dy = (end[1] - d.start[1]).round().clamp(-1e7, 1e7) as i32;
    Some((dx, dy))
}

/// The document to show while a Move drag is under way on document `idx`: the moving layers at
/// the pointer. `None` without a drag (or when the layers can't move: locked, say; the drag then
/// shows its arrow and the release reports why).
pub(crate) fn display_doc(app: &mut PhotocraftApp, idx: usize) -> Option<(Arc<Document>, u64)> {
    if app.session.active_index() == Some(idx)
        && let Some(st) = app.session.documents().get(idx)
        && photocraft_engine::float_cmds::floating(st).is_some()
    {
        return floating_doc(app, idx);
    }
    // Preferences › Interface › Show bounding box when dragging layer: outline and arrow only.
    if app.session.active_index() != Some(idx) || app.session.prefs().interface.show_bounding_box_when_dragging_layer {
        return None;
    }
    let Some(offset) = drag_offset(app) else {
        app.move_preview = None;
        return None;
    };
    let mask_target = app.layer_mask_targeted();
    let st = app.session.documents().get(idx)?;
    let (doc_id, revision, doc) = (st.doc.id, st.revision, st.doc.clone());
    let fresh = app.move_preview.as_ref().is_some_and(|p| p.doc == doc_id && p.revision == revision);
    if !fresh {
        let ids = photocraft_engine::layer_multi_cmds::move_targets(&doc, &st.selected_layers());
        if ids.is_empty() {
            return None;
        }
        let all = Rect::new(i32::MIN / 2, i32::MIN / 2, i32::MAX / 2, i32::MAX / 2);
        let bounds = ids.iter().try_fold(Rect::EMPTY, |acc, id| Some(acc.union(&photocraft_compose::change_bounds(doc.layer(*id)?, all)?)));
        let canvas = doc.bounds();
        let moving = with_descendants(&doc, &ids);
        let bounds = bounds.map(|b| moving.iter().filter_map(|id| shape_reach(doc.layer(*id)?, canvas)).fold(b, |acc, r| acc.union(&r)));
        app.move_preview = Some(MovePreview { doc: doc_id, revision, ids, moving, bounds, canvas, offsets: Vec::new(), shown: None, floating: false });
    }
    let p = app.move_preview.as_mut()?;
    if offset == (0, 0) && p.offsets.is_empty() {
        return None;
    }
    if p.offsets.last() != Some(&offset) {
        let t0 = crate::gpu_canvas::now_ms();
        match photocraft_engine::layer_multi_cmds::moved_target(&doc, &p.ids, offset.0, offset.1, mask_target) {
            Ok(d) => {
                // Duotone documents display through their inks.
                let d = photocraft_engine::mode_cmds::display_document(&d).unwrap_or(d);
                p.shown = Some(Arc::new(d));
                p.offsets.push(offset);
            }
            Err(_) => {
                p.shown = None;
                return None;
            }
        }
        app.perf.span("move preview", crate::gpu_canvas::now_ms() - t0);
    }
    let p = app.move_preview.as_ref()?;
    Some((p.shown.clone()?, p.key()))
}

/// Where shape layer `l` paints, effects included, not cut at the canvas as its rendered pixels
/// are: a drag brings the rest of it into view (#3946). `None` for other layers.
fn shape_reach(l: &photocraft_doc::Layer, canvas: Rect) -> Option<Rect> {
    let photocraft_doc::LayerContent::Shape(sh) = &l.content else { return None };
    let b = photocraft_vector::CompiledShape::new(sh, photocraft_vector::DEFAULT_TOLERANCE, canvas).bounds()?;
    let m = if photocraft_compose::effects::has_effects(l) { photocraft_compose::effects::margin(l) } else { 0 };
    Some(b.inflate(m))
}

/// `ids` and every layer inside them (groups move their contents).
fn with_descendants(doc: &Document, ids: &[LayerId]) -> Vec<LayerId> {
    let walk = doc.walk();
    let roots: Vec<_> = walk.iter().filter(|(_, _, l)| ids.contains(&l.id)).map(|(p, _, _)| p.as_slice()).collect();
    walk.iter().filter(|(p, _, _)| roots.iter().any(|r| p.starts_with(r))).map(|(_, _, l)| l.id).collect()
}

/// Where the canvas shows layer `id` of document `doc` at `revision` relative to where it is: the
/// offset of the latest Move drag frame drawn ([`display_doc`]) when the layer moves with the drag,
/// so its Layer Edges outline moves in lockstep with its pixels (#2413). `None` when it shows
/// where it is.
pub(crate) fn shown_offset(app: &PhotocraftApp, doc: DocId, revision: u64, id: LayerId) -> Option<(i32, i32)> {
    let p = app.move_preview.as_ref().filter(|p| p.doc == doc && p.revision == revision && !p.floating && p.shown.is_some())?;
    if !p.moving.contains(&id) {
        return None;
    }
    p.offsets.last().copied()
}

/// A floating selection's offset plus a drag of it in progress (`canvas::selection_drag_delta`).
pub(crate) fn floating_offset(app: &PhotocraftApp) -> Option<(i32, i32)> {
    let f = photocraft_engine::float_cmds::floating(app.session.active()?)?;
    let d = crate::canvas::selection_drag_delta(app).unwrap_or((0, 0));
    Some((f.offset.0 + d.0, f.offset.1 + d.1))
}

/// The document shown while a selection floats: the cut piece at its offset, over the cut-out
/// layer. Only the piece's area changes between offsets, so each move costs the piece, not the
/// layer.
fn floating_doc(app: &mut PhotocraftApp, idx: usize) -> Option<(Arc<Document>, u64)> {
    let offset = floating_offset(app)?;
    let st = app.session.documents().get(idx)?;
    let f = photocraft_engine::float_cmds::floating(st)?;
    let (doc_id, revision) = (st.doc.id, st.revision);
    let fresh = app.move_preview.as_ref().is_some_and(|p| p.doc == doc_id && p.revision == revision && p.floating);
    if !fresh {
        let bounds = st.doc.selection.as_ref().map(|s| s.content_bounds());
        let canvas = st.doc.bounds();
        app.move_preview = Some(MovePreview {
            doc: doc_id,
            revision,
            ids: vec![f.layer],
            moving: Vec::new(),
            bounds,
            canvas,
            offsets: Vec::new(),
            shown: None,
            floating: true,
        });
    }
    let st = app.session.documents().get(idx)?;
    let shown = app.move_preview.as_ref()?.offsets.last() != Some(&offset);
    if shown {
        let t0 = crate::gpu_canvas::now_ms();
        let d = photocraft_engine::float_cmds::displayed(st, crate::canvas::selection_drag_delta(app).unwrap_or((0, 0)))?;
        let d = photocraft_engine::mode_cmds::display_document(&d).unwrap_or(d);
        let p = app.move_preview.as_mut()?;
        p.shown = Some(Arc::new(d));
        p.offsets.push(offset);
        app.perf.span("floating preview", crate::gpu_canvas::now_ms() - t0);
    }
    let p = app.move_preview.as_ref()?;
    Some((p.shown.clone()?, p.key()))
}

/// What changed between preview (or document) key `seen` and `now` of the current Move drag on
/// document `doc` at `revision`: where the moving layers were in both. `None` when either key
/// isn't one of this drag's (the canvas then recomposites everything).
pub(crate) fn damage(app: &PhotocraftApp, doc: DocId, revision: u64, seen: u64, now: u64) -> Option<Rect> {
    let p = app.move_preview.as_ref().filter(|p| p.doc == doc && p.revision == revision)?;
    if seen == now || (seen < BASE && seen != 0) || (now < BASE && now != 0) {
        return None;
    }
    let (a, b) = (p.area(p.offset_of(seen)?)?, p.area(p.offset_of(now)?)?);
    Some(if a.is_empty() {
        b
    } else if b.is_empty() {
        a
    } else {
        a.union(&b)
    })
}

/// Whether the canvas shows a Move drag's layers at the pointer.
pub(crate) fn showing(app: &PhotocraftApp) -> bool {
    app.move_preview.as_ref().is_some_and(|p| p.shown.is_some())
}

/// Whether `key` is a Move drag preview key.
pub(crate) fn is_preview_key(key: u64) -> bool {
    key > BASE && key < BASE << 1
}

/// Release: one `layer.translate` by the drag's offset. The canvas already shows the result, so
/// its caches are told it showed the document: the command's damage rect refreshes that area.
pub(crate) fn finish(app: &mut PhotocraftApp, dx: f64, dy: f64) {
    // Only when the canvas shows this very offset (else it recomposites everything once).
    let at = (dx.clamp(-1e7, 1e7) as i32, dy.clamp(-1e7, 1e7) as i32);
    let shown = app.move_preview.take().filter(|p| p.shown.is_some() && p.offsets.last() == Some(&at));
    if dx == 0.0 && dy == 0.0 {
        return;
    }
    if app.run("layer.translate", json!({"dx": dx, "dy": dy})).is_err() {
        return;
    }
    let Some(p) = shown else { return };
    crate::canvas::shown_as_document(app, p.doc, is_preview_key);
}

#[cfg(test)]
mod tests {
    use photocraft_geom::Rect;
    use serde_json::json;

    use crate::PhotocraftApp;
    use crate::canvas::ToolEvent;
    use crate::state::Tool;

    #[test]
    fn dragging_a_layer_larger_than_the_canvas_redraws_what_comes_in_from_beyond_the_edge() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
        app.sync_views();
        app.session.execute("layer.new.layer", json!({})).unwrap();
        app.session
            .edit("paint", |doc, a| {
                doc.layer_mut(a.unwrap()).unwrap().surface_mut().unwrap().fill_rect(Rect::new(-32, -32, 96, 96), &[1.0, 0.0, 0.0, 1.0]);
                Ok(())
            })
            .unwrap();
        app.ui.tool = Tool::Move;
        app.ui.extras.snap = false;
        app.ui.view.show.smart_guides = false;
        let m = egui::Modifiers::NONE;
        crate::canvas::tool_event(&mut app, ToolEvent::Down { x: 32.0, y: 32.0, pressure: 1.0 }, m);
        let mut keys = Vec::new();
        for x in [42.0, 52.0] {
            crate::canvas::tool_event(&mut app, ToolEvent::Move { x, y: 32.0, pressure: 1.0 }, m);
            keys.push(super::display_doc(&mut app, 0).unwrap().1);
        }
        let st = app.session.active().unwrap();
        // From +10 to +20 the strip x 0..10 shows pixels that were off the canvas a frame ago.
        let r = super::damage(&app, st.doc.id, st.revision, keys[0], keys[1]).unwrap();
        assert_eq!(r, Rect::new(0, 0, 64, 64));
    }

    /// A shape reaching past the canvas edge renders cut at the canvas, but its path goes on: a
    /// drag frame must redraw where the rest of it was, or the cut edge stays behind (#3946).
    #[test]
    fn dragging_a_shape_past_the_canvas_edge_leaves_no_trail() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": 64, "height": 64, "background": "transparent"})).unwrap();
        app.sync_views();
        let id = app.session.execute("shape.create", json!({"kind": "rect", "rect": [40, 16, 50, 32], "fill": "#ff0000"})).unwrap()["layer"].as_u64().unwrap();
        let id = photocraft_doc::LayerId(id);
        app.ui.tool = Tool::Move;
        app.ui.extras.snap = false;
        app.ui.view.show.smart_guides = false;
        let m = egui::Modifiers::NONE;
        crate::canvas::tool_event(&mut app, ToolEvent::Down { x: 50.0, y: 32.0, pressure: 1.0 }, m);
        let mut frames = Vec::new();
        for x in [40.0, 10.0] {
            crate::canvas::tool_event(&mut app, ToolEvent::Move { x, y: 32.0, pressure: 1.0 }, m);
            frames.push(super::display_doc(&mut app, 0).unwrap());
        }
        let st = app.session.active().unwrap();
        let r = super::damage(&app, st.doc.id, st.revision, frames[0].1, frames[1].1).unwrap();
        let alpha = |d: &photocraft_doc::Document, x, y| d.layer(id).and_then(|l| l.surface()).map_or(0.0, |s| s.pixel(x, y)[3]);
        for y in 0..64 {
            for x in 0..64 {
                if alpha(&frames[0].0, x, y) != alpha(&frames[1].0, x, y) {
                    assert!(r.contains(x, y), "({x}, {y}) changed outside the redrawn {r:?}");
                }
            }
        }
    }

    /// Dragging with an unlinked mask targeted moves the mask alone, shown live and committed;
    /// with the mask linked (or the pixels targeted) the layer moves.
    #[test]
    fn dragging_with_an_unlinked_mask_targeted_moves_the_mask_alone() {
        for (linked, mask_target) in [(false, true), (false, false), (true, true)] {
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
            app.session.execute("file.new", json!({"width": 64, "height": 64, "background": "transparent"})).unwrap();
            app.sync_views();
            app.session.execute("layer.new.layer", json!({})).unwrap();
            app.session.execute("select.rect", json!({"x": 8, "y": 8, "width": 16, "height": 16})).unwrap();
            app.session.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
            app.session.execute("select.deselect", json!({})).unwrap();
            app.session.execute("layer.layerMask.revealAll", json!({})).unwrap();
            app.session.execute("layer.layerMask.linked", json!({"linked": linked})).unwrap();
            app.ui.mask_target = mask_target;
            app.ui.tool = Tool::Move;
            app.ui.extras.snap = false;
            app.ui.view.show.smart_guides = false;
            let m = egui::Modifiers::NONE;
            crate::canvas::tool_event(&mut app, ToolEvent::Down { x: 16.0, y: 16.0, pressure: 1.0 }, m);
            crate::canvas::tool_event(&mut app, ToolEvent::Move { x: 36.0, y: 16.0, pressure: 1.0 }, m);
            let shown = super::display_doc(&mut app, 0).unwrap().0;
            let id = app.session.active().unwrap().active_layer.unwrap();
            let layer_moved = |d: &photocraft_doc::Document| d.layer(id).and_then(|l| l.surface()).is_some_and(|s| s.pixel(28, 16)[3] > 0.5);
            let alone = !linked && mask_target;
            assert_eq!(layer_moved(&shown), !alone, "preview, linked {linked} mask target {mask_target}");
            crate::canvas::tool_event(&mut app, ToolEvent::Up { x: 36.0, y: 16.0 }, m);
            let d = &app.session.active().unwrap().doc;
            assert_eq!(layer_moved(d), !alone, "commit, linked {linked} mask target {mask_target}");
        }
    }

    /// Mid-drag on a painted layer: is the layer shown at the pointer?
    fn live(outline: bool) -> bool {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
        app.sync_views();
        app.session.execute("layer.new.layer", json!({})).unwrap();
        app.session.execute("select.rect", json!({"x": 8, "y": 8, "width": 16, "height": 16})).unwrap();
        app.session.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
        app.session.execute("select.deselect", json!({})).unwrap();
        app.session.edit_prefs(|p| p.interface.show_bounding_box_when_dragging_layer = outline);
        app.ui.tool = Tool::Move;
        let m = egui::Modifiers::NONE;
        crate::canvas::tool_event(&mut app, ToolEvent::Down { x: 16.0, y: 16.0, pressure: 1.0 }, m);
        crate::canvas::tool_event(&mut app, ToolEvent::Move { x: 30.0, y: 20.0, pressure: 1.0 }, m);
        let shown = super::display_doc(&mut app, 0).is_some();
        assert_eq!(shown, super::showing(&app));
        shown
    }

    #[test]
    fn bounding_box_preference_turns_the_live_drag_off() {
        assert!(live(false), "by default the layer follows the pointer");
        assert!(!live(true), "with the preference on, only the outline and arrow move");
    }

    /// A `depth`-bit transparent document whose layer holds a red square (8..24)² and a blue one
    /// (40..56)×(8..24), the red one selected; the Move tool with Auto-Select on.
    fn selected_pixels(depth: u32) -> (PhotocraftApp, photocraft_doc::LayerId) {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": 64, "height": 64, "background": "transparent", "depth": depth})).unwrap();
        app.sync_views();
        for (x, color) in [(40, "#0000ff"), (8, "#ff0000")] {
            app.session.execute("select.rect", json!({"x": x, "y": 8, "width": 16, "height": 16})).unwrap();
            app.session.execute("edit.fill", json!({"color": color})).unwrap();
        }
        app.ui.tool = Tool::Move;
        app.ui.tool_options.move_auto_select = true;
        app.ui.extras.snap = false;
        app.ui.view.show.smart_guides = false;
        let id = app.session.active().unwrap().active_layer.unwrap();
        (app, id)
    }

    fn alpha(app: &PhotocraftApp, id: photocraft_doc::LayerId, x: i32, y: i32) -> f32 {
        app.session.active().unwrap().doc.layer(id).unwrap().surface().unwrap().rgba(x, y)[3]
    }

    fn offset(app: &PhotocraftApp) -> Option<(i32, i32)> {
        photocraft_engine::float_cmds::floating(app.session.active().unwrap()).map(|f| f.offset)
    }

    fn drag(app: &mut PhotocraftApp, from: [f64; 2], to: [f64; 2], m: egui::Modifiers) {
        crate::canvas::tool_event(app, ToolEvent::Down { x: from[0], y: from[1], pressure: 1.0 }, m);
        crate::canvas::tool_event(app, ToolEvent::Move { x: to[0], y: to[1], pressure: 1.0 }, m);
        crate::canvas::tool_event(app, ToolEvent::Up { x: to[0], y: to[1] }, m);
    }

    /// ↩ through the app's key handling.
    fn enter(app: &mut PhotocraftApp) {
        let ctx = egui::Context::default();
        let key = egui::Event::Key { key: egui::Key::Enter, physical_key: None, pressed: true, repeat: false, modifiers: egui::Modifiers::NONE };
        let input = egui::RawInput { events: vec![key], ..Default::default() };
        ctx.run_ui(input, |ui| crate::shortcuts::handle(app, ui.ctx())).textures_delta.clear();
    }

    /// #578: with a selection the Move tool drags the selected pixels (from anywhere, without an
    /// Auto-Select pick), not the layer; ↩ drops them as one "Move" step that one undo takes back.
    #[test]
    fn with_a_selection_the_move_tool_moves_only_the_selected_pixels() {
        for depth in [8, 16, 32] {
            let (mut app, id) = selected_pixels(depth);
            let steps = app.session.active().unwrap().history.past_len();
            drag(&mut app, [50.0, 40.0], [60.0, 45.0], egui::Modifiers::NONE);
            assert_eq!(offset(&app), Some((10, 5)), "{depth}-bit: floating");
            let shown = super::display_doc(&mut app, 0).unwrap().0;
            let shown_alpha = |x, y| shown.layer(id).unwrap().surface().unwrap().rgba(x, y)[3];
            assert!(shown_alpha(10, 10) == 0.0 && shown_alpha(30, 25) == 1.0, "{depth}-bit: shown moved");
            enter(&mut app);
            assert_eq!(offset(&app), None, "{depth}-bit: ↩ dropped it");
            assert_eq!(alpha(&app, id, 10, 10), 0.0, "{depth}-bit: the hole is transparent");
            assert_eq!(alpha(&app, id, 30, 25), 1.0, "{depth}-bit: the red square moved");
            assert_eq!(alpha(&app, id, 48, 16), 1.0, "{depth}-bit: the unselected blue one stayed");
            let st = app.session.active().unwrap();
            assert_eq!(st.doc.selection.as_ref().unwrap().content_bounds(), Rect::new(18, 13, 34, 29), "{depth}-bit");
            assert_eq!((st.history.past_len(), st.history.undo_label()), (steps + 1, Some("Move")), "{depth}-bit");
            assert!(app.session.undo());
            assert!(alpha(&app, id, 10, 10) == 1.0 && alpha(&app, id, 30, 25) == 0.0, "{depth}-bit: undone");
        }
    }

    /// ⌥ copies the selected pixels (no layer copy), ⇧ locks the drag to multiples of 45°.
    #[test]
    fn alt_copies_and_shift_constrains_the_selected_pixels() {
        for depth in [8, 16, 32] {
            let (mut app, id) = selected_pixels(depth);
            let layers = app.session.active().unwrap().doc.layers.len();
            drag(&mut app, [16.0, 16.0], [30.0, 28.0], egui::Modifiers::ALT | egui::Modifiers::SHIFT);
            assert_eq!(offset(&app), Some((13, 13)), "{depth}-bit: 45°");
            app.run("select.drop", json!({})).unwrap();
            assert_eq!(app.session.active().unwrap().doc.layers.len(), layers, "{depth}-bit: no layer copy");
            assert!(alpha(&app, id, 10, 10) == 1.0 && alpha(&app, id, 35, 35) == 1.0, "{depth}-bit: original and copy");
            assert!(app.session.undo());
            assert_eq!(alpha(&app, id, 35, 35), 0.0, "{depth}-bit: undone");
        }
    }

    /// ⌘-drag with a painting tool drags the selected pixels as the Move tool does, also when ⌘
    /// comes with the event (control channel, MCP) rather than as the held-key override.
    #[test]
    fn command_drag_with_a_painting_tool_moves_only_the_selected_pixels() {
        let (mut app, id) = selected_pixels(8);
        app.ui.tool = Tool::Brush;
        app.ui.tool_options.move_auto_select = false;
        let cmd = egui::Modifiers { mac_cmd: true, command: true, ..Default::default() };
        drag(&mut app, [50.0, 40.0], [60.0, 45.0], cmd);
        assert_eq!(offset(&app), Some((10, 5)), "the selected pixels float");
        drag(&mut app, [50.0, 40.0], [60.0, 45.0], cmd);
        assert_eq!(offset(&app), Some((20, 10)), "a second drag moves the same floating piece");
        app.run("select.drop", json!({})).unwrap();
        assert!(alpha(&app, id, 10, 10) == 0.0 && alpha(&app, id, 40, 30) == 1.0, "the red square moved");
        assert_eq!(alpha(&app, id, 48, 16), 1.0, "the unselected blue one stayed");
    }

    /// A click moves nothing and leaves nothing floating (Undo isn't spent putting it back).
    #[test]
    fn a_click_with_a_selection_moves_nothing() {
        let (mut app, id) = selected_pixels(8);
        let steps = app.session.active().unwrap().history.past_len();
        drag(&mut app, [16.0, 16.0], [16.0, 16.0], egui::Modifiers::NONE);
        assert_eq!(offset(&app), None);
        assert_eq!(app.session.active().unwrap().history.past_len(), steps);
        assert!(alpha(&app, id, 10, 10) == 1.0 && alpha(&app, id, 48, 16) == 1.0);
    }

    #[test]
    fn arrow_keys_nudge_the_selected_pixels() {
        let (mut app, id) = selected_pixels(16);
        crate::move_mods::nudge(&mut app, 10.0, 0.0, false);
        crate::move_mods::nudge(&mut app, 0.0, 1.0, false);
        assert_eq!(offset(&app), Some((10, 1)));
        // ⌥ leaves the piece where it is and nudges a copy of it.
        crate::move_mods::nudge(&mut app, 1.0, 0.0, true);
        assert_eq!(offset(&app), Some((1, 0)));
        app.run("select.drop", json!({})).unwrap();
        assert!(alpha(&app, id, 10, 10) == 0.0 && alpha(&app, id, 18, 20) == 1.0 && alpha(&app, id, 34, 20) == 1.0);
        assert_eq!(alpha(&app, id, 48, 16), 1.0, "the layer itself didn't move");
    }

    /// Without floatable pixels (a position-locked layer) the Move tool behaves as without a
    /// selection: it moves the layer, which the lock refuses.
    #[test]
    fn a_layer_that_cannot_float_moves_whole_as_before() {
        let (mut app, id) = selected_pixels(8);
        app.session.execute("layer.lockLayers", json!({"position": true})).unwrap();
        drag(&mut app, [16.0, 16.0], [26.0, 16.0], egui::Modifiers::NONE);
        assert_eq!(offset(&app), None);
        assert!(alpha(&app, id, 10, 10) == 1.0 && alpha(&app, id, 30, 16) == 0.0, "nothing moved");
        // Without a selection the layer moves whole.
        app.session.execute("layer.lockLayers", json!({"position": false})).unwrap();
        app.session.execute("select.deselect", json!({})).unwrap();
        drag(&mut app, [16.0, 16.0], [26.0, 16.0], egui::Modifiers::NONE);
        assert!(alpha(&app, id, 10, 10) == 0.0 && alpha(&app, id, 58, 16) == 1.0, "the whole layer moved");
    }
}
