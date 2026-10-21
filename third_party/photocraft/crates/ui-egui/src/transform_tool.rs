//! Free Transform (⌘T): bounding box with handles over the canvas, live preview, commit via the
//! engine's `edit.transform` (one quad for scale/rotate/skew/distort/perspective).
//!
//! Gestures follow Photoshop CC: corner and edge drags scale proportionally (⇧ for free: an edge
//! then stretches its one axis), ⌥ scales about the reference point, ⌘-drag a corner distorts
//! (⌘⌥⇧: perspective), ⌘-drag an edge skews (⇧ along the edge), drag outside rotates (⇧ snaps to
//! 15°), drag inside moves (⇧ locks to 8 directions), the reference point can be dragged and
//! ⌥-click puts it under the pointer. Arrow keys nudge the box (move_mods.rs). ↩ commits, Esc
//! cancels. Undo and Redo step through the session's own changes (`Steps`), not the document's
//! history.

use std::sync::Arc;

use egui::{Color32, CursorIcon, Pos2, Stroke, pos2, vec2};
use photocraft_algo::transform::Homography;
use photocraft_doc::{DocId, Document, LayerContent, LayerId};
use photocraft_engine::jobs::{JobEvent, JobId, JobOutcome};
use photocraft_geom::warp::{BezierMesh, Warp, WarpStyle};
use serde_json::json;

use crate::PhotocraftApp;
use crate::canvas::{ToolEvent, ViewXform};
use crate::state::{MadeLayer, Tool, TransformMode, TransformSession};

/// Which Split button is armed. The guide follows the pointer and the split is added on release.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SplitTool {
    Cross,
    Vertical,
    Horizontal,
}

impl SplitTool {
    fn command(self) -> &'static str {
        match self {
            SplitTool::Cross => "edit.transform.splitWarpCrosswise",
            SplitTool::Vertical => "edit.transform.splitWarpVertically",
            SplitTool::Horizontal => "edit.transform.splitWarpHorizontally",
        }
    }

    fn icon(self) -> &'static str {
        match self {
            SplitTool::Cross => "cross",
            SplitTool::Vertical => "vertical",
            SplitTool::Horizontal => "horizontal",
        }
    }
}

/// What a Warp-mode press grabbed. `pts0` is the mesh's control grid at the press.
enum WarpDrag {
    /// An anchor or handle: it follows the pointer (an anchor takes its handles along).
    Point { index: usize, start: [f64; 2], pts0: Vec<[f64; 2]> },
    /// Anywhere else on the grid: the surface under the pointer follows it.
    Surface { s: f64, t: f64, start: [f64; 2], pts0: Vec<[f64; 2]> },
}

/// Preview state that isn't serialisable: the document without the transformed pixels, and
/// full-resolution textures of those pixels (transform_tex.rs).
pub struct TransformPreview {
    pub session: u64,
    pub doc: Arc<Document>,
    pub texture: crate::transform_tex::PreviewTextures,
    pub opacity: f32,
    /// The layers stacked over the moving one, drawn over its pixels ([`Above`]).
    above: Option<Above>,
    /// A layer with a style: its pixels composited with the style as the box has them now,
    /// drawn instead of the bare pixels (transform_fx.rs, #3653).
    styled: Option<crate::transform_fx::StyledPreview>,
    gesture: Option<Gesture>,
    /// Warp-mode drag in progress.
    warp_drag: Option<WarpDrag>,
    /// Armed split tool. None while the buttons are off and control points drag as usual.
    split_tool: Option<SplitTool>,
    /// Document point the split guide is following (a drag, or the hover while placing).
    split_pointer: Option<[f64; 2]>,
    split_placing: bool,
    /// Option-click places a split without a button armed.
    split_quick: bool,
    steps: Steps,
    /// The tool the session began with: picking another one applies the transform.
    tool: Tool,
    /// Free Transform Path: the path as it was, drawn through the box instead of pixels.
    path: Option<photocraft_doc::vector::Path>,
    /// A layer's session keeps `doc` and `opacity` up to date with edits made while the box is
    /// open ([`follow_edits`]); `None` for a lone mask or channel and Transform Selection.
    follow: Option<Follow>,
    /// The commit running as a background job (#1015): the moving pixels stay on screen, without
    /// the box, until it lands; Esc cancels it. The session ends with the job.
    committing: Option<JobId>,
    /// Content-Aware Scale's live preview (cas_transform.rs).
    pub(crate) content_aware: Option<crate::cas_transform::Live>,
}

/// What [`follow_edits`] needs to rebuild the preview document.
#[derive(Clone, Copy, Debug)]
struct Follow {
    /// The document revision `doc` was built from.
    revision: u64,
    /// The moving layer is hidden in `doc` (nothing was lifted out of it).
    hidden: bool,
}

/// Photoshop's alert when Free Transform (⌘T, Edit › Transform, the Move tool's transform controls)
/// starts on a layer locked in place, by its own lock or a group's.
pub const LOCKED: &str = "Could not complete the Free Transform command because the layer is locked.";

/// Why Edit › Transform's modes and Warp refuse while Content-Aware Scale's box is open.
pub const CAS_ONLY_SCALES: &str = "Content-Aware Scale only scales: commit or cancel it first";

/// Grab radius of the box's handles, in screen points. Generous, so a corner is easy to catch;
/// just beyond it, outside the box, a drag rotates.
pub const HANDLE_PX: f64 = 12.0;

/// [`HANDLE_PX`] in document pixels at the current zoom.
pub fn handle_tolerance(app: &PhotocraftApp) -> f64 {
    HANDLE_PX / app.point_zoom().max(0.01) as f64
}

/// The box as one undo step restores it.
#[derive(Clone, Debug, PartialEq)]
struct Step {
    rect: [f64; 4],
    quad: [[f64; 2]; 4],
    pivot: [f64; 2],
    warp: Option<Warp>,
}

impl Step {
    fn of(t: &TransformSession) -> Self {
        Step { rect: t.rect, quad: t.quad, pivot: t.pivot, warp: t.warp.clone() }
    }

    fn restore(self, t: &mut TransformSession) {
        (t.rect, t.quad, t.pivot, t.warp) = (self.rect, self.quad, self.pivot, self.warp);
    }
}

/// The session's own history: inside Free Transform, Undo steps back through the
/// handle drags, nudges and option edits rather than the document's history.
#[derive(Default)]
struct Steps {
    /// The box when it last came to rest (set when the session starts).
    settled: Option<Step>,
    undo: Vec<Step>,
    redo: Vec<Step>,
}

impl Steps {
    /// Record a step if the box changed since it last came to rest.
    fn settle(&mut self, now: Step) {
        if self.settled.as_ref() == Some(&now) {
            return;
        }
        if let Some(prev) = self.settled.replace(now) {
            self.undo.push(prev);
            self.redo.clear();
        }
    }

    /// Something to undo, counting a change not yet recorded (a nudge this frame).
    fn can_undo(&self, now: &Step) -> bool {
        !self.undo.is_empty() || self.settled.as_ref().is_some_and(|p| p != now)
    }
}

fn corners(r: [f64; 4]) -> [[f64; 2]; 4] {
    [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]]
}

/// Start Free Transform on the active layer (or its selected pixels), or on the targeted unlinked
/// layer mask, alpha channel or Quick Mask.
pub fn begin(app: &mut PhotocraftApp, ctx: &egui::Context) -> Result<(), String> {
    if let Some((params, path)) = crate::vector_ui::free_transform_path(app) {
        return begin_path(app, ctx, params, path);
    }
    if app.session.active().is_some_and(|st| photocraft_engine::float_cmds::floating(st).is_some()) {
        app.run("select.drop", json!({}))?;
    }
    let target = crate::canvas::paint_target(app);
    let st = app.session.active().ok_or("no document")?;
    let (doc, revision) = (st.doc.clone(), st.revision);
    let lone = photocraft_engine::transform_cmds::lone_target(&doc, st.active_layer, &json!({ "target": target })).map_err(|e| e.to_string())?.cloned();
    if let Some(surf) = lone {
        return begin_lone(app, ctx, doc, surf, target);
    }
    let id = st.active_layer.ok_or("no active layer")?;
    let layer = doc.layer(id).ok_or("no layer")?;
    if photocraft_engine::transform_cmds::locked_for_transform(&doc, id) {
        // Photoshop refuses before the box opens rather than when it is committed (#2585).
        app.ui.open_dialog(crate::state::DialogKind::Error, json!({"message": tl!(LOCKED)}).as_object().cloned().unwrap_or_default());
        return Err(LOCKED.into());
    }
    if matches!(layer.content, LayerContent::Adjustment(_)) && layer.mask.is_none() {
        return Err("Adjustment layers have nothing to transform".into());
    }
    let b = photocraft_engine::transform_cmds::transform_bounds(&doc, layer);
    if b.is_empty() {
        return Err("Could not transform: the layer is empty".into());
    }
    crate::type_tool::commit(app);
    let rect = [b.x0 as f64, b.y0 as f64, b.x1 as f64, b.y1 as f64];
    let session = app.ui.alloc_id();
    // Preview document: the moving pixels removed (layer hidden, or the selection lifted out).
    let mut pd = (*doc).clone();
    let mut lifted = None;
    // Type, shapes and smart objects move whole, selection or not (#2630): only pixels are lifted.
    if let Some(sel) = doc.selection.clone().filter(|_| !photocraft_engine::transform_cmds::moves_whole(layer) && layer.surface().is_some()) {
        if let Some(surf) = pd.layer_mut(id).and_then(|l| l.surface_mut()) {
            let (l, rest) = photocraft_engine::transform_cmds::split_selected(surf, &sel).map_err(|e| e.to_string())?;
            *surf = rest;
            lifted = Some(l);
        }
    } else if let Some(l) = pd.layer_mut(id) {
        l.visible = false;
    }
    // egui reports the renderer's limit in the app; offscreen harnesses only through the device.
    let max_side = ctx.input(|i| i.max_texture_side).max(app.gpu.as_ref().map_or(1, |g| g.max_texture_side()));
    // The layers covering the moving pixels draw over them instead of under them.
    let above = Above::new(ctx, format!("transform-{session}-above"), &doc, id, max_side);
    above.hide_in(&mut pd.layers);
    // A style is drawn again from the pixels as the box moves them, so they are read without it.
    // Not for the selected part of a layer: its style belongs to that part and the rest together.
    let styled = lifted.is_none() && photocraft_compose::effects::has_effects(layer);
    let (image, uv) = preview_image(&doc, id, lifted.as_ref(), b, max_side, styled);
    let texture = crate::transform_tex::PreviewTextures::new(ctx, format!("transform-{session}"), image, uv);
    app.transform_preview = Some(TransformPreview {
        session,
        doc: Arc::new(pd),
        texture,
        opacity: layer.opacity * layer.fill_opacity,
        above: Some(above),
        styled: styled.then(|| crate::transform_fx::StyledPreview::new(format!("transform-{session}-styled"))),
        follow: Some(Follow { revision, hidden: lifted.is_none() }),
        committing: None,
        gesture: None,
        warp_drag: None,
        split_tool: None,
        split_pointer: None,
        split_placing: false,
        split_quick: false,
        steps: Steps::default(),
        tool: app.ui.tool,
        path: None,
        content_aware: None,
    });
    app.ui.transform = Some(TransformSession {
        session,
        layer: id.0,
        rect,
        quad: corners(rect),
        pivot: [(rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0],
        interpolation: "bicubic".into(),
        warp: None,
        selection: false,
        target: None,
        path: None,
        made: None,
        mode: Default::default(),
        content_aware: None,
        skew_v: 0.0,
    });
    start_steps(app);
    Ok(())
}

/// Free Transform on a copy (⌥⌘T, no menu item): duplicates the active layer (with a selection,
/// its selected pixels, as Layer via Copy) and transforms the copy (#352).
pub fn begin_copy(app: &mut PhotocraftApp, ctx: &egui::Context) -> Result<(), String> {
    let selection = app.session.active().is_some_and(|d| d.doc.selection.is_some());
    let params = if selection { json!({}) } else { json!({"inPlace": true}) };
    app.run(if selection { "layer.new.layerViaCopy" } else { "layer.duplicate" }, params)?;
    let made = app.session.active().and_then(|st| st.active_layer.map(|layer| (st.doc.id, layer)));
    if let Err(e) = begin(app, ctx) {
        if let Some((document, layer)) = made {
            take_back_made(app, document, layer);
        }
        return Err(e);
    }
    if let Some(t) = app.ui.transform.as_mut() {
        t.made = Some(MadeLayer::Copy);
    }
    Ok(())
}

/// Free Transform on the layer a file dropped on the canvas just placed, like the reference app's
/// Place: Esc takes the place back, ↩ makes the place and the transform one Place Embedded step.
pub fn begin_placed(app: &mut PhotocraftApp, ctx: &egui::Context) -> Result<(), String> {
    begin(app, ctx)?;
    if let Some(t) = app.ui.transform.as_mut() {
        t.made = Some(MadeLayer::Place);
    }
    Ok(())
}

/// Edit › Content-Aware Scale (#3096): a scale-only box over `region` of the active pixel layer
/// (the selection's bounds, else the layer's content), cleared under the box as the command
/// clears it. OK runs `edit.contentAwareScale` (cas_transform.rs) instead of `edit.transform`.
pub(crate) fn begin_content_aware(
    app: &mut PhotocraftApp,
    ctx: &egui::Context,
    region: photocraft_geom::Rect,
    live: crate::cas_transform::Live,
) -> Result<(), String> {
    let st = app.session.active().ok_or("no document")?;
    let (doc, revision) = (st.doc.clone(), st.revision);
    let id = st.active_layer.ok_or("no active layer")?;
    let layer = doc.layer(id).ok_or("no layer")?;
    let surf = layer.surface().ok_or("Content-Aware Scale needs a pixel layer")?;
    if region.is_empty() {
        return Err("Could not scale: the layer is empty".into());
    }
    crate::type_tool::commit(app);
    let rect = [region.x0 as f64, region.y0 as f64, region.x1 as f64, region.y1 as f64];
    let session = app.ui.alloc_id();
    // The command clears the area before writing the result (a layer without transparency keeps it).
    let mut pd = (*doc).clone();
    if let Some(s) = pd.layer_mut(id).and_then(|l| l.surface_mut()).filter(|s| s.format().alpha) {
        let clear = vec![0.0; s.format().channels()];
        s.fill_rect(region, &clear);
        s.prune();
    }
    let max_side = ctx.input(|i| i.max_texture_side).max(app.gpu.as_ref().map_or(1, |g| g.max_texture_side()));
    let above = Above::new(ctx, format!("transform-{session}-above"), &doc, id, max_side);
    above.hide_in(&mut pd.layers);
    let (image, uv) = crate::transform_tex::read_surface(surf, None, region, max_side);
    let texture = crate::transform_tex::PreviewTextures::new(ctx, format!("transform-{session}"), image, uv);
    app.transform_preview = Some(TransformPreview {
        session,
        doc: Arc::new(pd),
        texture,
        opacity: layer.opacity * layer.fill_opacity,
        above: Some(above),
        follow: Some(Follow { revision, hidden: false }),
        committing: None,
        gesture: None,
        warp_drag: None,
        split_tool: None,
        split_pointer: None,
        split_placing: false,
        split_quick: false,
        steps: Steps::default(),
        tool: app.ui.tool,
        path: None,
        content_aware: Some(live),
        styled: None,
    });
    app.ui.transform = Some(TransformSession {
        session,
        layer: id.0,
        rect,
        quad: corners(rect),
        pivot: [(rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0],
        interpolation: "bicubic".into(),
        warp: None,
        selection: false,
        target: None,
        path: None,
        made: None,
        mode: Default::default(),
        content_aware: Some(Default::default()),
        skew_v: 0.0,
    });
    start_steps(app);
    Ok(())
}

/// Undoes the layer a cancelled or failed session made (⌥⌘T's copy, a placed file), leaving
/// nothing to redo.
fn take_back_made(app: &mut PhotocraftApp, document: DocId, layer: LayerId) {
    // Tabs can move or close while transforming. Only undo the creation in its own document,
    // and only while it is still the latest step (it may already have been undone).
    let Some(index) = app.session.documents().iter().position(|st| {
        st.doc.id == document
            && st.doc.layer(layer).is_some()
            && st.history.past_len().checked_sub(1).and_then(|i| st.history.state(i)).is_some_and(|before| before.layer(layer).is_none())
    }) else {
        return;
    };
    let active = app.session.active_index();
    app.session.set_active(index);
    if app.session.undo()
        && let Some(st) = app.session.active_mut()
    {
        st.history.clear_redo();
    }
    if let Some(active) = active {
        app.session.set_active(active);
    }
    app.sync_views();
}

/// After the transform: the step that made the layer and the transform become one history step,
/// the transform's for a copy and Place Embedded for a placed file.
fn fold_made(app: &mut PhotocraftApp, document: DocId, made: MadeLayer) {
    // A background commit lands in its own document, which need not be the active one by then.
    let Some(index) = app.session.documents().iter().position(|st| st.doc.id == document) else { return };
    let active = app.session.active_index();
    app.session.set_active(index);
    if let Some(st) = app.session.active_mut() {
        st.history.purge_last();
        if made == MadeLayer::Place {
            st.history.set_current_label(photocraft_engine::file_cmds::PLACE_EMBEDDED);
        }
    }
    if let Some(active) = active {
        app.session.set_active(active);
    }
}

/// Free Transform of a targeted unlinked layer mask, alpha channel or Quick Mask by itself: the
/// box frames its content (within the selection) and previews the moving values in grey over the
/// document with them vacated.
fn begin_lone(
    app: &mut PhotocraftApp,
    ctx: &egui::Context,
    doc: Arc<Document>,
    surf: photocraft_raster::Surface,
    target: serde_json::Value,
) -> Result<(), String> {
    use photocraft_engine::transform_cmds as tc;
    let b = tc::target_bounds(&doc, &surf);
    if b.is_empty() {
        return Err("Could not transform: nothing is selected".into());
    }
    let whole;
    let sel = match &doc.selection {
        Some(s) => s,
        None => {
            let mut s = photocraft_raster::Surface::new(photocraft_color::PixelFormat::GRAY8);
            s.fill_rect(b, &[1.0]);
            whole = s;
            &whole
        }
    };
    let (lifted, rest) = tc::split_gray_selected(&surf, sel).map_err(|e| e.to_string())?.ok_or("this channel can't be transformed")?;
    crate::type_tool::commit(app);
    let layer = app.session.active().and_then(|st| st.active_layer).map_or(0, |l| l.0);
    let mut pd = (*doc).clone();
    let tp = json!({ "target": target });
    if let Ok(Some(s)) = tc::lone_target_mut(&mut pd, Some(LayerId(layer)), &tp) {
        *s = rest;
    }
    let rect = [b.x0 as f64, b.y0 as f64, b.x1 as f64, b.y1 as f64];
    let session = app.ui.alloc_id();
    let max_side = ctx.input(|i| i.max_texture_side).max(app.gpu.as_ref().map_or(1, |g| g.max_texture_side()));
    let (image, uv) = crate::transform_tex::read_surface(&lifted, None, b, max_side);
    let texture = crate::transform_tex::PreviewTextures::new(ctx, format!("transform-{session}"), image, uv);
    app.transform_preview = Some(TransformPreview {
        session,
        doc: Arc::new(pd),
        texture,
        opacity: 0.6,
        above: None,
        styled: None,
        follow: None,
        committing: None,
        gesture: None,
        warp_drag: None,
        split_tool: None,
        split_pointer: None,
        split_placing: false,
        split_quick: false,
        steps: Steps::default(),
        tool: app.ui.tool,
        path: None,
        content_aware: None,
    });
    app.ui.transform = Some(TransformSession {
        session,
        layer,
        rect,
        quad: corners(rect),
        pivot: [(rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0],
        interpolation: "bicubic".into(),
        warp: None,
        selection: false,
        target: Some(target),
        path: None,
        made: None,
        mode: Default::default(),
        content_aware: None,
        skew_v: 0.0,
    });
    start_steps(app);
    Ok(())
}

/// Edit › Free Transform Path: the box frames the path's anchors and handles and draws the
/// path through it; OK commits one `path.transform` step.
fn begin_path(app: &mut PhotocraftApp, ctx: &egui::Context, params: serde_json::Value, path: photocraft_doc::vector::Path) -> Result<(), String> {
    let (x0, y0, x1, y1) = path.control_bounds().ok_or("the path has no points to transform")?;
    let st = app.session.active().ok_or("no document")?;
    let doc = st.doc.clone();
    let layer = st.active_layer.or_else(|| doc.layers.first().map(|l| l.id)).ok_or("no layer")?;
    crate::type_tool::commit(app);
    // A straight horizontal or vertical path still gets a box to turn.
    let rect = [x0, y0, x1.max(x0 + 1.0), y1.max(y0 + 1.0)];
    let session = app.ui.alloc_id();
    let texture =
        crate::transform_tex::PreviewTextures::new(ctx, format!("transform-{session}"), egui::ColorImage::new([1, 1], vec![Color32::TRANSPARENT]), [1.0, 1.0]);
    app.transform_preview = Some(TransformPreview {
        session,
        doc,
        texture,
        opacity: 1.0,
        styled: None,
        gesture: None,
        warp_drag: None,
        split_tool: None,
        split_pointer: None,
        split_placing: false,
        split_quick: false,
        steps: Steps::default(),
        tool: app.ui.tool,
        path: Some(path),
        content_aware: None,
        above: None,
        follow: None,
        committing: None,
    });
    app.ui.transform = Some(TransformSession {
        session,
        layer: layer.0,
        rect,
        quad: corners(rect),
        pivot: [(rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0],
        interpolation: "bicubic".into(),
        warp: None,
        selection: false,
        target: None,
        path: Some(params),
        made: None,
        mode: Default::default(),
        content_aware: None,
        skew_v: 0.0,
    });
    start_steps(app);
    Ok(())
}

/// The affine map taking `rect`'s corners to `quad`'s; `None` when the box was distorted out of a
/// parallelogram (Distort, Perspective): the path then goes as a `quad`.
fn rect_to_quad_affine(r: [f64; 4], q: [[f64; 2]; 4]) -> Option<photocraft_geom::Affine> {
    let (w, h) = (r[2] - r[0], r[3] - r[1]);
    if w <= 0.0 || h <= 0.0 {
        return None;
    }
    let tol = 1e-3 * (1.0 + w.max(h));
    if (q[0][0] + q[2][0] - q[1][0] - q[3][0]).abs() > tol || (q[0][1] + q[2][1] - q[1][1] - q[3][1]).abs() > tol {
        return None;
    }
    let (a, b) = ((q[1][0] - q[0][0]) / w, (q[1][1] - q[0][1]) / w);
    let (c, d) = ((q[3][0] - q[0][0]) / h, (q[3][1] - q[0][1]) / h);
    let m = [a, b, c, d, q[0][0] - a * r[0] - c * r[1], q[0][1] - b * r[0] - d * r[1]];
    m.iter().all(|v| v.is_finite()).then_some(photocraft_geom::Affine { m })
}

/// Start Select › Transform Selection: the same box over the selection's bounds, previewing the
/// selection mask (the marching ants are hidden meanwhile); commits `select.transformSelection`.
pub fn begin_selection(app: &mut PhotocraftApp, ctx: &egui::Context) -> Result<(), String> {
    let st = app.session.active().ok_or("no document")?;
    let doc = st.doc.clone();
    let sel = doc.selection.clone().ok_or("no selection")?;
    let b = sel.content_bounds();
    if b.is_empty() {
        return Err("no selection".into());
    }
    let layer = st.active_layer.or_else(|| doc.layers.first().map(|l| l.id)).ok_or("no layer")?;
    crate::type_tool::commit(app);
    let rect = [b.x0 as f64, b.y0 as f64, b.x1 as f64, b.y1 as f64];
    let session = app.ui.alloc_id();
    let (w, h) = (b.width() as usize, b.height() as usize);
    let k = w.max(h).div_ceil(2048).max(1);
    let (tw, th) = (w.div_ceil(k), h.div_ceil(k));
    let mut px = vec![Color32::TRANSPARENT; tw * th];
    for ty in 0..th {
        for tx in 0..tw {
            let m = sel.sample_channel(b.x0 + (tx * k) as i32, b.y0 + (ty * k) as i32, 0).clamp(0.0, 1.0);
            px[ty * tw + tx] = Color32::from_rgba_unmultiplied(120, 170, 255, (m * 110.0) as u8);
        }
    }
    let uv = [w as f32 / (tw * k) as f32, h as f32 / (th * k) as f32];
    let texture = crate::transform_tex::PreviewTextures::new(ctx, format!("transform-{session}"), egui::ColorImage::new([tw, th], px), uv);
    let mut pd = (*doc).clone();
    pd.selection = None;
    app.transform_preview = Some(TransformPreview {
        session,
        doc: Arc::new(pd),
        texture,
        opacity: 1.0,
        above: None,
        styled: None,
        follow: None,
        committing: None,
        gesture: None,
        warp_drag: None,
        split_tool: None,
        split_pointer: None,
        split_placing: false,
        split_quick: false,
        steps: Steps::default(),
        tool: app.ui.tool,
        path: None,
        content_aware: None,
    });
    app.ui.transform = Some(TransformSession {
        session,
        layer: layer.0,
        rect,
        quad: corners(rect),
        pivot: [(rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0],
        interpolation: "bilinear".into(),
        warp: None,
        selection: true,
        target: None,
        path: None,
        made: None,
        mode: Default::default(),
        content_aware: None,
        skew_v: 0.0,
    });
    start_steps(app);
    Ok(())
}

/// Start (or switch an active Free Transform into) Warp mode.
pub fn begin_warp(app: &mut PhotocraftApp, ctx: &egui::Context) -> Result<(), String> {
    if app.ui.transform.as_ref().is_some_and(|t| t.content_aware.is_some()) {
        return Err(CAS_ONLY_SCALES.into());
    }
    let fresh = app.ui.transform.is_none();
    if fresh {
        begin(app, ctx)?;
    }
    enter_warp(app, ctx);
    // Started in Warp: that is where Undo stops. Switching an open transform to Warp is a step.
    if fresh {
        start_steps(app);
    }
    Ok(())
}

/// Switch to Warp mode: a smart object's existing warp (when the box is untouched), else a
/// custom mesh following the current box (so a transform made first carries over). Editable type
/// gets its own Warp Text style instead ([`enter_type_warp`]).
pub fn enter_warp(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if app.ui.transform.as_ref().is_some_and(|t| t.warp.is_none()) && warps_type(app) {
        return enter_type_warp(app, ctx);
    }
    let existing =
        app.session.active().and_then(|d| d.doc.layer(LayerId(app.ui.transform.as_ref()?.layer)).and_then(photocraft_engine::warp_cmds::smart_warp_doc_space));
    let Some(t) = app.ui.transform.as_mut() else { return };
    // Warp moves layers and paths (#3027): a lone mask or channel keeps the box.
    if t.warp.is_some() || t.target.is_some() {
        return;
    }
    if t.quad == corners(t.rect)
        && !t.selection
        && t.path.is_none()
        && let Some(w) = existing
    {
        t.rect = w.bounds;
        t.quad = corners(w.bounds);
        t.warp = Some(w);
        return;
    }
    let h = Homography::rect_to_quad(t.rect, t.quad).unwrap_or(Homography::IDENTITY);
    let mesh = BezierMesh::identity(t.rect, 1, 1).map_points(|p| {
        let (x, y) = h.apply(p[0], p[1]);
        [x, y]
    });
    t.warp = Some(Warp::custom(mesh, t.rect));
}

/// The open transform moves an editable type layer as a whole (not a lone mask, channel, path or
/// the selection). Its Warp is Warp Text: only the styles, and the type stays editable (#3004).
pub fn warps_type(app: &PhotocraftApp) -> bool {
    let Some(t) = &app.ui.transform else { return false };
    if t.selection || t.target.is_some() || t.path.is_some() {
        return false;
    }
    app.session.active().and_then(|st| st.doc.layer(LayerId(t.layer))).is_some_and(|l| matches!(l.content, LayerContent::Text(_)))
}

/// Warp on editable type, as in Photoshop: no rasterizing, no mesh to drag, just the warp styles
/// of the options bar (Warp Text's). A box scaled or rotated first is applied first (that keeps
/// type editable). The preview shows the type without its current Warp Text, under that style.
fn enter_type_warp(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if app.ui.transform.as_ref().is_some_and(|t| t.quad != corners(t.rect)) {
        commit(app);
        if app.ui.transform.is_some() {
            return; // refused: the box stays as it was
        }
        if let Err(e) = begin(app, ctx) {
            app.ui.status = e;
            return;
        }
    }
    let (Some(t), Some(st)) = (app.ui.transform.as_ref(), app.session.active()) else { return };
    let (id, session) = (LayerId(t.layer), t.session);
    let doc = st.doc.clone();
    let Some(LayerContent::Text(text)) = doc.layer(id).map(|l| &l.content) else { return };
    let current = text.warp.clone();
    let mut plain = (*doc).clone();
    if let Some(LayerContent::Text(tt)) = plain.layer_mut(id).map(|l| &mut l.content)
        && current.is_some()
    {
        tt.warp = None;
        photocraft_engine::type_cmds::refresh(&doc, tt);
    }
    let Some(b) = plain.layer(id).map(|l| photocraft_engine::transform_cmds::transform_bounds(&plain, l)).filter(|b| !b.is_empty()) else { return };
    let rect = [b.x0 as f64, b.y0 as f64, b.x1 as f64, b.y1 as f64];
    if current.is_some()
        && let Some(pv) = app.transform_preview.as_mut()
    {
        let max_side = ctx.input(|i| i.max_texture_side).max(app.gpu.as_ref().map_or(1, |g| g.max_texture_side()));
        // Read as the session's texels were: without the style when the box draws it itself.
        let (image, uv) = preview_image(&plain, id, None, b, max_side, pv.styled.is_some());
        pv.texture = crate::transform_tex::PreviewTextures::new(ctx, format!("transform-{session}"), image, uv);
    }
    let warp =
        current.as_ref().and_then(|w| WarpStyle::parse(&w.style).filter(|s| s.is_preset()).map(|s| (w, s))).map_or(Warp::none(rect), |(w, style)| Warp {
            style,
            bend: f64::from(w.value),
            h_distort: f64::from(w.horizontal_distortion),
            v_distort: f64::from(w.vertical_distortion),
            vertical: !w.horizontal,
            ..Warp::none(rect)
        });
    if let Some(t) = app.ui.transform.as_mut() {
        t.rect = rect;
        t.quad = corners(rect);
        t.pivot = [(rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0];
        t.warp = Some(warp);
    }
}

/// Leave Warp mode (back to the box; warp edits are dropped, like Photoshop's mode toggle on an
/// untouched warp).
pub fn leave_warp(app: &mut PhotocraftApp) {
    if let Some(t) = app.ui.transform.as_mut() {
        t.warp = None;
    }
    if let Some(pv) = app.transform_preview.as_mut() {
        pv.split_tool = None;
        pv.split_pointer = None;
        pv.split_placing = false;
        pv.split_quick = false;
    }
}

/// Texels of the moving pixels over `b`: full resolution unless the long side exceeds the GPU's
/// `max_side` (#91: this used to be capped at 2048 px, so big layers previewed blurry), plus the
/// uv extent covering `b`. `bare` leaves the layer's own style, opacity and fill out of a
/// flattened render: the styled preview applies them to the moved pixels itself.
fn preview_image(
    doc: &Document,
    id: LayerId,
    lifted: Option<&photocraft_raster::Surface>,
    b: photocraft_geom::Rect,
    max_side: usize,
    bare: bool,
) -> (egui::ColorImage, [f32; 2]) {
    let layer = doc.layer(id);
    // Linked masks move with the pixels, so the moving texels carry them (#205).
    let mask = layer.and_then(|l| preview_mask(l, b));
    // Groups and type/fill layers preview from a flattened render of just that layer.
    let rendered;
    let surf: Option<&photocraft_raster::Surface> = match (lifted, layer.and_then(|l| l.surface())) {
        (Some(s), _) => Some(s),
        (None, Some(s)) if layer.is_some_and(|l| matches!(l.content, LayerContent::Raster(_))) => Some(s),
        _ => {
            let mut solo = doc.clone();
            for l in solo.layers.iter_mut() {
                if l.id != id && doc.layer(id).is_some_and(|_| !contains(l, id)) {
                    l.visible = false;
                }
            }
            if let Some(l) = solo.layer_mut(id).filter(|_| bare) {
                l.effects.enabled = false;
                l.opacity = 1.0;
                l.fill_opacity = 1.0;
            }
            let buf = photocraft_compose::render(&solo, b);
            let mut s = photocraft_raster::Surface::new(photocraft_color::PixelFormat::RGBA8);
            let flat: Vec<f32> = buf.px.iter().flat_map(|p| *p).collect();
            s.write_region(b, &flat);
            rendered = s;
            Some(&rendered)
        }
    };
    let Some(surf) = surf else { return (egui::ColorImage::new([1, 1], vec![Color32::TRANSPARENT]), [1.0, 1.0]) };
    // A flattened render already has the masks applied.
    let raw = lifted.is_some() || layer.is_some_and(|l| matches!(l.content, LayerContent::Raster(_)));
    crate::transform_tex::read_surface(surf, mask.as_ref().filter(|_| raw), b, max_side)
}

/// The layer's masks as the moving texels carry them: the enabled masks when all of them are linked
/// (the vector mask and feathering folded in as the compositor applies them), else the linked
/// pixel mask alone. `None` when no enabled mask moves with the pixels.
fn preview_mask(l: &photocraft_doc::Layer, b: photocraft_geom::Rect) -> Option<crate::transform_tex::PreviewMask> {
    use crate::transform_tex::PreviewMask;
    let pixel = l.mask.as_ref().filter(|m| m.enabled);
    let vector = l.vector_mask.as_ref().filter(|v| v.enabled);
    let linked_pixel = pixel.filter(|m| m.linked);
    if pixel.is_none_or(|m| m.linked)
        && vector.is_none_or(|v| v.linked)
        && let Some(surface) = photocraft_compose::masks::combined_mask(l, b)
    {
        return Some(PreviewMask { surface, density: 1.0 });
    }
    linked_pixel.map(|m| PreviewMask { surface: m.surface.clone(), density: m.density })
}

/// The layers covering the moving one ([`covering`]), composited on their own and drawn over its
/// preview, so the layer keeps its place in the stack while the box is open (#2439): they cover
/// it, as in Photoshop, instead of the moving pixels covering the whole document. The canvas
/// under the box leaves them out ([`Above::hide_in`]). Built when the session starts and again
/// when an edit changes the document; the box itself moves without re-compositing.
struct Above {
    ctx: egui::Context,
    name: String,
    max_side: usize,
    /// The covering layers (whole layers or groups, at any depth).
    ids: Vec<LayerId>,
    /// Their composite and the canvas area it covers; `None` when nothing visible covers the layer.
    shown: Option<(crate::transform_tex::PreviewTextures, photocraft_geom::Rect)>,
}

impl Above {
    fn new(ctx: &egui::Context, name: String, doc: &Document, id: LayerId, max_side: usize) -> Self {
        let mut a = Self { ctx: ctx.clone(), name, max_side, ids: Vec::new(), shown: None };
        a.rebuild(doc, id);
        a
    }

    /// Composite what covers layer `id` of `doc` (the live document) again.
    fn rebuild(&mut self, doc: &Document, id: LayerId) {
        self.ids.clear();
        covering(&doc.layers, id, &mut self.ids);
        let mut up = doc.clone();
        show_only(&mut up.layers, &self.ids);
        let canvas = up.bounds();
        let rect = up
            .layers
            .iter()
            .filter(|l| l.visible)
            .fold(photocraft_geom::Rect::EMPTY, |r, l| r.union(&photocraft_compose::composite_bounds(l, canvas).unwrap_or(canvas)))
            .intersect(&canvas);
        self.shown = (!rect.is_empty()).then(|| {
            let (image, uv) = composite_image(&up, rect, self.max_side);
            (crate::transform_tex::PreviewTextures::new(&self.ctx, self.name.clone(), image, uv), rect)
        });
    }

    /// Leave the covering layers out of `layers` (the document under the box).
    fn hide_in(&self, layers: &mut [photocraft_doc::Layer]) {
        for l in layers.iter_mut() {
            if self.ids.contains(&l.id) {
                l.visible = false;
            } else if let LayerContent::Group(g) = &mut l.content {
                self.hide_in(&mut g.children);
            }
        }
    }

    /// Draw the composite over the moving pixels.
    fn draw(&self, painter: &egui::Painter, xf: &ViewXform) {
        let Some((texture, r)) = &self.shown else { return };
        let (tex, uv) = texture.pick(painter.ctx(), xf.zoom * painter.ctx().pixels_per_point(), 1.0, false);
        let mut mesh = egui::Mesh::with_texture(tex);
        let corners = [(r.x0, r.y0, 0.0, 0.0), (r.x1, r.y0, uv[0], 0.0), (r.x1, r.y1, uv[0], uv[1]), (r.x0, r.y1, 0.0, uv[1])];
        for (x, y, u, v) in corners {
            mesh.vertices.push(egui::epaint::Vertex { pos: xf.to_screen(x as f32, y as f32), uv: pos2(u, v), color: Color32::WHITE });
        }
        mesh.add_triangle(0, 1, 2);
        mesh.add_triangle(0, 2, 3);
        painter.add(mesh);
    }
}

/// `rect` of `doc`'s composite as premultiplied texels: full resolution when it fits in
/// `max_side`, composited in bands (no full-size float copy), else area-averaged down to fit.
fn composite_image(doc: &Document, rect: photocraft_geom::Rect, max_side: usize) -> (egui::ColorImage, [f32; 2]) {
    let (w, h) = (rect.width() as usize, rect.height() as usize);
    let to_color = |p: &[u8; 4]| Color32::from_rgba_unmultiplied(p[0], p[1], p[2], p[3]);
    if w.max(h) <= max_side.max(1) {
        let mut px = Vec::with_capacity(w * h);
        let _ = photocraft_compose::render_bands(doc, rect, 0, |band| {
            px.extend(band.to_rgba8().pixels.as_chunks::<4>().0.iter().map(to_color));
            Ok::<(), ()>(())
        });
        px.resize(w * h, Color32::TRANSPARENT);
        return (egui::ColorImage::new([w, h], px), [1.0, 1.0]);
    }
    let k = w.max(h).div_ceil(max_side.max(1));
    let (tw, th) = (w.div_ceil(k).max(1), h.div_ceil(k).max(1));
    let buf = photocraft_compose::render_reduced_rect(doc, rect, tw as u32, th as u32);
    let px: Vec<Color32> = buf.to_rgba8().pixels.as_chunks::<4>().0.iter().map(to_color).collect();
    let [bw, bh] = [buf.rect.width() as usize, buf.rect.height() as usize];
    if px.len() != bw * bh || bw == 0 || bh == 0 {
        return (egui::ColorImage::new([1, 1], vec![Color32::TRANSPARENT]), [1.0, 1.0]);
    }
    (egui::ColorImage::new([bw, bh], px), [1.0, 1.0])
}

/// A layer that composites over anything the same as over transparency, so it can be drawn over
/// the moving pixels on its own: a Normal layer that isn't an adjustment, or a Normal or Pass
/// Through group of such layers (hidden layers count, they draw nothing). Adjustment layers and
/// the other blend modes depend on what is under them.
fn plain(l: &photocraft_doc::Layer) -> bool {
    use photocraft_color::BlendMode;
    !l.visible
        || match &l.content {
            LayerContent::Adjustment(_) => false,
            LayerContent::Group(g) => matches!(l.blend, BlendMode::Normal | BlendMode::PassThrough) && g.children.iter().all(plain),
            _ => l.blend == BlendMode::Normal,
        }
}

/// The layers over `id` in `layers` (bottom to top, groups holding `id` searched too) that cover
/// it in the preview: from the top down, while they are [`plain`]. The first layer that isn't
/// stops the search, so it and everything under it stay in the document under the box as before
/// (an adjustment there keeps applying to the layers under it). A clipping group goes as a whole,
/// and the layers clipped to `id` (or to the group holding it) stay with it.
fn covering(layers: &[photocraft_doc::Layer], id: LayerId, out: &mut Vec<LayerId>) {
    let mut end = layers.len();
    while end > 0 {
        // The clipping group ending at `end`: its base and the layers clipped to it.
        let mut start = end - 1;
        while start > 0 && layers.get(start).is_some_and(|l| l.clipped) {
            start -= 1;
        }
        let unit = layers.get(start..end).unwrap_or_default();
        if unit.iter().any(|l| contains(l, id)) {
            // A group holding `id` with nothing clipped to it: go on inside it.
            if let [g] = unit
                && g.id != id
                && let LayerContent::Group(group) = &g.content
                && matches!(g.blend, photocraft_color::BlendMode::Normal | photocraft_color::BlendMode::PassThrough)
            {
                covering(&group.children, id, out);
            }
            return;
        }
        if !unit.iter().all(plain) {
            return;
        }
        out.extend(unit.iter().map(|l| l.id));
        end = start;
    }
}

/// Hide everything in `layers` but the `keep` layers and the groups holding them.
fn show_only(layers: &mut [photocraft_doc::Layer], keep: &[LayerId]) {
    for l in layers.iter_mut() {
        if keep.contains(&l.id) {
            continue;
        }
        match &mut l.content {
            LayerContent::Group(g) if g.children.iter().any(|c| keep.iter().any(|k| contains(c, *k))) => show_only(&mut g.children, keep),
            _ => l.visible = false,
        }
    }
}

fn contains(l: &photocraft_doc::Layer, id: LayerId) -> bool {
    l.id == id || matches!(&l.content, LayerContent::Group(g) if g.children.iter().any(|c| contains(c, id)))
}

/// Undo and Redo while the commit runs in the background.
const STILL_APPLYING: &str = "Free Transform is still being applied";

pub fn commit(app: &mut PhotocraftApp) {
    if committing(app).is_some() {
        return;
    }
    let document = app.transform_preview.as_ref().map(|pv| pv.doc.id);
    if document.is_some_and(|id| app.session.active().is_none_or(|st| st.doc.id != id)) {
        cancel(app);
        return;
    }
    let type_layer = warps_type(app);
    let Some(t) = app.ui.transform.take() else { return };
    let mut preview = app.transform_preview.take();
    if let Some(options) = &t.content_aware {
        crate::cas_transform::commit(app, &t, options);
        return;
    }
    if t.selection {
        let mut p = json!({"rect": t.rect, "interpolation": t.interpolation});
        match &t.warp {
            Some(w) if !w.is_identity() => p["warp"] = json!(w),
            Some(_) => return,
            None if t.quad == corners(t.rect) => return,
            None => p["quad"] = json!(t.quad),
        }
        if let Err(e) = app.run("select.transformSelection", p) {
            keep_open(app, t, preview, e);
        }
        return;
    }
    if let Some(target) = &t.path {
        let mut p = target.clone();
        match &t.warp {
            Some(w) if w.is_identity() => return,
            Some(w) => p["warp"] = json!(w),
            None if t.quad == corners(t.rect) => return,
            // Distort and Perspective map the path through the box (#3027).
            None => match rect_to_quad_affine(t.rect, t.quad) {
                Some(a) => p["matrix"] = json!(a.m),
                None => {
                    p["rect"] = json!(t.rect);
                    p["quad"] = json!(t.quad);
                }
            },
        }
        if let Err(e) = app.run("path.transform", p) {
            keep_open(app, t, preview, e);
        }
        return;
    }
    let made = t.made;
    let r = if let Some(w) = &t.warp {
        // Type with a Warp Text can be warped back to None; the engine skips an unchanged style.
        if w.is_identity() && !type_layer {
            return;
        }
        app.run("edit.transform.warp", json!({"layer": t.layer, "rect": t.rect, "warp": w, "interpolation": t.interpolation}))
    } else {
        if t.quad == corners(t.rect) {
            return; // untouched: nothing to do and no history step; a ⌥⌘T copy or a place stays
        }
        let mut p = json!({"layer": t.layer, "rect": t.rect, "quad": t.quad, "interpolation": t.interpolation});
        if let Some(target) = &t.target {
            p["target"] = target.clone();
        }
        app.run("edit.transform", p)
    };
    // A large layer's warp runs in the background: the session stays until the job ends.
    if let Ok(v) = &r
        && let Some(job) = v.get("pending").and_then(serde_json::Value::as_bool).filter(|p| *p).and(v.get("job")).and_then(serde_json::Value::as_u64)
        && let Some(pv) = preview.as_mut()
    {
        pv.committing = Some(JobId(job));
        app.ui.transform = Some(t);
        app.transform_preview = preview;
        return;
    }
    match r {
        Ok(_) => finish(app, document, LayerId(t.layer), made, true),
        Err(e) => keep_open(app, t, preview, e),
    }
}

/// After the transform landed or not: fold the layer the session made into its step, or take it
/// back.
fn finish(app: &mut PhotocraftApp, document: Option<DocId>, layer: LayerId, made: Option<MadeLayer>, landed: bool) {
    let (Some(made), Some(document)) = (made, document) else { return };
    if landed {
        fold_made(app, document, made);
    } else {
        take_back_made(app, document, layer);
    }
}

/// The job applying the transform, while it runs.
pub(crate) fn committing(app: &PhotocraftApp) -> Option<JobId> {
    app.transform_preview.as_ref().and_then(|pv| pv.committing)
}

/// A job ended (see [`crate::jobs_ui`]): when it was the commit, the session ends with it. Its
/// status and errors are reported like any job's.
pub fn on_job_event(app: &mut PhotocraftApp, e: &JobEvent) {
    if committing(app) != Some(e.id) {
        return;
    }
    let (Some(t), Some(mut pv)) = (app.ui.transform.take(), app.transform_preview.take()) else { return };
    // A refused commit (a result over the pixel budget, #3773) keeps the box open, as inline.
    if let JobOutcome::Failed(msg) = &e.outcome {
        pv.committing = None;
        keep_open(app, t, Some(pv), msg.clone());
        return;
    }
    finish(app, Some(pv.doc.id), LayerId(t.layer), t.made, matches!(e.outcome, JobOutcome::Done(_)));
}

/// A commit the engine refused (a result over the pixel budget, #3773): the box stays open as it
/// was, with the reason in the status bar, so the transform can be changed or cancelled. A copy or
/// a placed layer it made goes only when it is cancelled.
fn keep_open(app: &mut PhotocraftApp, t: TransformSession, preview: Option<TransformPreview>, e: String) {
    app.ui.transform = Some(t);
    app.transform_preview = preview;
    app.ui.status = e;
}

/// A transform whose layer or document went away (undo, close) ends silently; picking another tool
/// or selecting another layer (Layers panel, ⌘-click, the control channel) applies it, so the box
/// never stays on a layer that is no longer the active one (#1400). Checked every frame and before
/// each pointer event, so a press with a tool chosen just before it (`ui.pointer`'s `tool`) goes to
/// that tool.
pub fn end_if_left(app: &mut PhotocraftApp) {
    let Some(t) = &app.ui.transform else { return };
    if committing(app).is_some() {
        return;
    }
    let Some(st) = app.session.active() else { return cancel(app) };

    let layer = LayerId(t.layer);
    let made = t.made;
    let selection = t.selection;
    let preview_doc_id = app.transform_preview.as_ref().map(|pv| pv.doc.id);
    let switched_document = preview_doc_id.is_some_and(|id| id != st.doc.id);

    let editing_placed_smart_object = switched_document
        && made == Some(MadeLayer::Place)
        && preview_doc_id
            .is_some_and(|parent_id| app.session.smart_links.iter().any(|link| link.parent == parent_id && link.child == st.doc.id && link.layer == layer));

    if editing_placed_smart_object {
        app.ui.transform = None;
        app.transform_preview = None;
    } else if switched_document || st.doc.layer(layer).is_none() {
        cancel(app);
    } else if app.transform_preview.as_ref().is_some_and(|pv| pv.tool != app.ui.tool)
        || (!selection && st.active_layer.is_some_and(|active| active.0 != layer.0))
    {
        commit(app);
        // Picking another tool or layer can't leave a refused transform open: it ends, as Esc would.
        if app.ui.transform.is_some() {
            cancel(app);
        }
    }
}

/// Edits made while the box is open show in the preview at once, not only after the commit
/// (#1384): the moving layer's opacity, fill and visibility tint the moving pixels, and the
/// document under them (other layers, and the part of the layer a selection leaves behind) is
/// rebuilt from the current one. The moving pixels keep the session's. Called before the canvas
/// reads the preview, so it never caches a revision with the old preview.
pub fn follow_edits(app: &mut PhotocraftApp) {
    let (Some(t), Some(pv)) = (&app.ui.transform, app.transform_preview.as_mut()) else { return };
    let Some(follow) = pv.follow.as_mut() else { return };
    let Some(st) = app.session.active().filter(|st| st.doc.id == pv.doc.id && st.revision != follow.revision) else { return };
    let id = LayerId(t.layer);
    let (Some(live), Some(moving)) = (st.doc.layer(id), pv.doc.layer(id)) else { return };
    let mut pd = (*st.doc).clone();
    if let Some(l) = pd.layer_mut(id) {
        l.content = moving.content.clone();
        l.visible = l.visible && !follow.hidden;
    }
    if let Some(a) = pv.above.as_mut() {
        a.rebuild(&st.doc, id);
        a.hide_in(&mut pd.layers);
    }
    pv.opacity = if live.visible { live.opacity * live.fill_opacity } else { 0.0 };
    follow.revision = st.revision;
    pv.doc = Arc::new(pd);
}

pub fn cancel(app: &mut PhotocraftApp) {
    // Esc while the commit runs cancels the job; the session ends when it does.
    if let Some(job) = committing(app) {
        crate::jobs_ui::cancel(app, job);
        return;
    }
    let transform = app.ui.transform.take();
    let preview = app.transform_preview.take();
    if let (Some(t), Some(pv)) = (transform, preview)
        && t.made.is_some()
    {
        take_back_made(app, pv.doc.id, LayerId(t.layer));
    }
}

/// The session's starting box: where Undo stops.
fn start_steps(app: &mut PhotocraftApp) {
    if let (Some(t), Some(pv)) = (&app.ui.transform, app.transform_preview.as_mut()) {
        pv.steps = Steps { settled: Some(Step::of(t)), ..Steps::default() };
    }
}

/// Record the box as an undo step whenever it comes to rest in a new state: once per drag (on
/// release), nudge or option edit (once typing in its field ends). Call once per frame.
pub fn track_steps(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let (Some(t), Some(pv)) = (&app.ui.transform, app.transform_preview.as_mut()) else { return };
    if pv.gesture.is_some() || pv.warp_drag.is_some() || ctx.input(|i| i.pointer.any_down()) || ctx.text_edit_focused() {
        return;
    }
    pv.steps.settle(Step::of(t));
}

/// While transforming, Undo and Redo apply to the box: `Some(enabled)` for those commands (and
/// Toggle Last State, which would change the document under the box), `None` for the rest.
pub fn is_enabled(app: &PhotocraftApp, id: &str) -> Option<bool> {
    let t = app.ui.transform.as_ref()?;
    if committing(app).is_some() {
        return box_command(id).then_some(false);
    }
    let steps = app.transform_preview.as_ref().map(|pv| &pv.steps);
    match id {
        // Content-Aware Scale's box only scales: no switching it to another transform or Warp.
        "edit.freeTransform" | "edit.freeTransformCopy" if t.content_aware.is_some() => Some(false),
        i if i.starts_with("edit.transform.") && t.content_aware.is_some() => Some(false),
        "edit.undo" => Some(steps.is_some_and(|s| s.can_undo(&Step::of(t)))),
        "edit.redo" => Some(steps.is_some_and(|s| !s.redo.is_empty())),
        // No Toggle Last State or history jump under the box, and one box at a time.
        "edit.toggleLastState" | "history.selectState" | "edit.contentAwareScale" | "edit.puppetWarp" => Some(false),
        // Rotate and Flip turn the box (not while warping, where the mesh is the box).
        _ if box_preset(id).is_some() => Some(t.warp.is_none()),
        // Editable type can't be distorted (#2630, as #2701 rules for the menus outside a box).
        "edit.transform.distort" | "edit.transform.perspective" if moves_type(app, t) => Some(false),
        _ => None,
    }
}

/// Does the box move an editable type layer (not its selection outline, a lone mask or a path)?
fn moves_type(app: &PhotocraftApp, t: &TransformSession) -> bool {
    !t.selection
        && t.target.is_none()
        && t.path.is_none()
        && app.session.active().and_then(|st| st.doc.layer(LayerId(t.layer))).is_some_and(|l| matches!(l.content, LayerContent::Text(_)))
}

/// Edit › Transform's Rotate and Flip presets while a box is open: what they do to the box, as
/// `(kx, ky, angle)`: a scale along the box's own axes, then a rotation (radians, clockwise on
/// screen), both about the reference point.
fn box_preset(id: &str) -> Option<(f64, f64, f64)> {
    use std::f64::consts::{FRAC_PI_2, PI};
    Some(match id {
        "edit.transform.rotate180" => (1.0, 1.0, PI),
        "edit.transform.rotate90Cw" => (1.0, 1.0, FRAC_PI_2),
        "edit.transform.rotate90Ccw" => (1.0, 1.0, -FRAC_PI_2),
        "edit.transform.flipHorizontal" => (-1.0, 1.0, 0.0),
        "edit.transform.flipVertical" => (1.0, -1.0, 0.0),
        _ => return None,
    })
}

/// While transforming, the box owns the history for every caller (menus, shortcuts, the control
/// channel, MCP): Undo and Redo step through it, and Toggle Last State and History state jumps,
/// which would change the document under the box, are refused. Edit › Transform's Rotate and
/// Flip turn the box itself, as in Photoshop, instead of the layer under it; committing stays
/// with ↩ and the options bar.
/// `None` for other commands or when not transforming.
/// The commands the open box answers itself ([`is_enabled`], [`intercept`]).
fn box_command(id: &str) -> bool {
    matches!(id, "edit.undo" | "edit.redo" | "edit.toggleLastState" | "edit.contentAwareScale" | "edit.puppetWarp") || box_preset(id).is_some()
}

pub fn intercept(app: &mut PhotocraftApp, id: &str) -> Option<Result<serde_json::Value, String>> {
    let warping = app.ui.transform.as_ref()?.warp.is_some();
    if committing(app).is_some() {
        return box_command(id).then(|| Err(STILL_APPLYING.into()));
    }
    match id {
        "edit.undo" => Some(Ok(step(app, false))),
        "edit.redo" => Some(Ok(step(app, true))),
        "edit.toggleLastState" | "history.selectState" | "edit.contentAwareScale" | "edit.puppetWarp" => {
            Some(Err("Commit or cancel the transform first".into()))
        }
        _ => {
            let (kx, ky, angle) = box_preset(id)?;
            if warping {
                return Some(Err("Commit or cancel the transform first".into()));
            }
            scale_about_pivot(app, kx, ky);
            rotate_about_pivot(app, angle);
            Some(Ok(json!({"transform": app.ui.transform})))
        }
    }
}

/// Undo (or redo) one step of the open session; nothing mid-drag or with no step to take.
fn step(app: &mut PhotocraftApp, redo: bool) -> serde_json::Value {
    let Some(t) = app.ui.transform.as_mut() else { return serde_json::Value::Null };
    let Some(pv) = app.transform_preview.as_mut().filter(|pv| pv.gesture.is_none() && pv.warp_drag.is_none()) else {
        return serde_json::Value::Null;
    };
    let s = &mut pv.steps;
    s.settle(Step::of(t));
    let (from, to) = if redo { (&mut s.redo, &mut s.undo) } else { (&mut s.undo, &mut s.redo) };
    let Some(step) = from.pop() else { return serde_json::Value::Null };
    to.push(Step::of(t));
    s.settled = Some(step.clone());
    step.restore(t);
    json!({"transform": t})
}

/// Which part of the box a document point hits.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) enum Hit {
    Corner(usize),
    Edge(usize),
    Pivot,
    Inside,
    Outside,
}

/// The nearest handle within `tol` (on a small box their grab areas overlap), else inside or
/// outside the box.
pub(crate) fn hit(t: &TransformSession, p: [f64; 2], tol: f64) -> Hit {
    let d = |a: [f64; 2]| ((a[0] - p[0]).powi(2) + (a[1] - p[1]).powi(2)).sqrt();
    let mid = |i: usize| {
        let (a, b) = (t.quad[i], t.quad[(i + 1) % 4]);
        [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]
    };
    let handles = (0..4).map(|i| (t.quad[i], Hit::Corner(i))).chain((0..4).map(|i| (mid(i), Hit::Edge(i)))).chain([(t.pivot, Hit::Pivot)]);
    let nearest = handles.map(|(q, h)| (d(q), h)).filter(|(dist, _)| *dist < tol).min_by(|a, b| a.0.total_cmp(&b.0));
    match nearest {
        Some((_, h)) => h,
        None if inside(&t.quad, p) => Hit::Inside,
        None => Hit::Outside,
    }
}

fn inside(q: &[[f64; 2]; 4], p: [f64; 2]) -> bool {
    let mut c = false;
    for i in 0..4 {
        let (a, b) = (q[i], q[(i + 3) % 4]);
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0] {
            c = !c;
        }
    }
    c
}

/// Transient drag state (kept in egui memory; not part of the serialised session).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Gesture {
    pub(crate) hit: Hit,
    pub(crate) start: [f64; 2],
    pub(crate) quad0: [[f64; 2]; 4],
    pub(crate) pivot0: [f64; 2],
}

/// A click outside the box accepts the transform; a drag outside still rotates it.
/// Decide on release, so the rotation zone keeps working for drags. The screen-sized
/// click tolerance absorbs pointer jitter without accepting a real rotation gesture.
fn accepts_outside_click(g: Gesture, end: [f64; 2], tol: f64) -> bool {
    g.hit == Hit::Outside && (end[0] - g.start[0]).hypot(end[1] - g.start[1]) <= 2.0 * tol / HANDLE_PX
}

/// Pointer input while transforming. Returns false when no transform is active.
pub fn pointer(app: &mut PhotocraftApp, ev: ToolEvent, mods: egui::Modifiers) -> bool {
    let Some(t) = app.ui.transform.clone() else { return false };
    if committing(app).is_some() {
        return true;
    }
    let tol = handle_tolerance(app);
    if t.warp.is_some() {
        warp_pointer(app, ev, tol, mods);
        return true;
    }
    let no_distort = distort_refused(app, &t);
    let Some(pv) = app.transform_preview.as_mut() else { return false };
    match ev {
        ToolEvent::Down { x, y, .. } => {
            let mut h = hit(&t, [x, y], tol);
            // ⌥-click away from the handles moves the reference point there (and drags it).
            if mods.alt
                && matches!(h, Hit::Inside | Hit::Outside)
                && let Some(s) = app.ui.transform.as_mut()
            {
                s.pivot = [x, y];
                h = Hit::Pivot;
            }
            // Distort only moves corners (and the whole box from inside): no rotating, no edges.
            // Even in Distort mode, remember an outside press so a plain click can
            // accept the transform. A drag outside remains inert in that mode, in Skew, and in
            // Content-Aware Scale, which only scales and moves.
            pv.gesture = (mode_allows(t.mode, h) || h == Hit::Outside).then_some(Gesture { hit: h, start: [x, y], quad0: t.quad, pivot0: t.pivot });
        }
        ToolEvent::Move { x, y, .. } | ToolEvent::Up { x, y } => {
            let g = pv.gesture;
            if matches!(ev, ToolEvent::Up { .. }) {
                pv.gesture = None;
            }
            // Clicking an empty spot commits instead of silently arming rotation (#2851).
            // Dragging the rotation zone and Option-clicking to move the pivot
            // keep their existing behavior.
            if matches!(ev, ToolEvent::Up { .. }) && g.is_some_and(|g| accepts_outside_click(g, [x, y], tol)) {
                commit(app);
                return true;
            }
            let legacy = app.session.prefs().general.use_legacy_free_transform;
            let unlinked = app.ui.transform_unlinked;
            let type_layer = moves_type(app, &t);
            if let (Some(g), Some(s)) = (g, app.ui.transform.as_mut())
                && mode_allows(t.mode, g.hit)
                && (t.content_aware.is_none() || g.hit != Hit::Outside)
            {
                let (mode, mods) = if no_distort { scale_not_distort(t.mode, g.hit, mods) } else { (t.mode, mods) };
                if t.content_aware.is_some() {
                    let mut mods = mode_mods(mode, g.hit, legacy_mods(legacy, g.hit, unlinked_mods(unlinked, legacy, g.hit, mods)));
                    // Content-Aware Scale: ⌘ neither skews nor distorts.
                    mods.command = false;
                    apply_drag(s, g, [x, y], mods);
                } else {
                    apply_mode_drag(s, g, [x, y], mode, type_layer, legacy_mods(legacy, g.hit, unlinked_mods(unlinked, legacy, g.hit, mods)));
                }
            }
        }
    }
    true
}

/// Editable type can be scaled, rotated and skewed but not distorted (#2630): the rule that greys
/// Distort and Perspective in `enable_rules::disabled`, for the box's own layer. A box on a path,
/// on the selection outline or on a lone mask or channel moves no type.
fn distort_refused(app: &PhotocraftApp, t: &TransformSession) -> bool {
    t.path.is_none()
        && !t.selection
        && t.target.is_none()
        && app.session.active().and_then(|st| st.doc.layer(LayerId(t.layer))).is_some_and(|l| matches!(l.content, LayerContent::Text(_)))
}

/// Where Distort and Perspective are refused ([`distort_refused`]), ⌘-drag and ⌘⌥⇧-drag of a
/// corner scale like a plain corner drag (with ⌥ and ⇧ keeping their meaning), so the box never
/// takes a shape the commit would refuse (#3010).
fn scale_not_distort(mode: TransformMode, hit: Hit, mut mods: egui::Modifiers) -> (TransformMode, egui::Modifiers) {
    if matches!(hit, Hit::Corner(_)) {
        mods.command = false;
    }
    let mode = if matches!(mode, TransformMode::Distort | TransformMode::Perspective) { TransformMode::Free } else { mode };
    (mode, mods)
}

/// Preferences › General › Use Legacy Free Transform, Photoshop's scaling before CC 2019: corner
/// drags stretch freely and ⇧ keeps the proportions, edge drags stretch their one axis whatever
/// the keys. The default is proportional for both, with ⇧ freeing them.
pub(crate) fn legacy_mods(legacy: bool, hit: Hit, mut mods: egui::Modifiers) -> egui::Modifiers {
    if legacy && !mods.command {
        match hit {
            Hit::Corner(_) => mods.shift = !mods.shift,
            Hit::Edge(_) => mods.shift = true,
            _ => {}
        }
    }
    mods
}

/// The options bar's Maintain Aspect Ratio link turned off: corner and edge drags stretch freely
/// and ⇧ keeps the proportions, as in Photoshop CC 2019 and later (#4174). The legacy preference
/// already works that way, so it is left alone there.
pub(crate) fn unlinked_mods(unlinked: bool, legacy: bool, hit: Hit, mut mods: egui::Modifiers) -> egui::Modifiers {
    if unlinked && !legacy && !mods.command && matches!(hit, Hit::Corner(_) | Hit::Edge(_)) {
        mods.shift = !mods.shift;
    }
    mods
}

/// Distort / Perspective modes: a corner drag acts as if that gesture's keys were held (⌘-drag a
/// corner distorts, ⌘⌥⇧-drag a corner adds perspective). Skew has its own ([`apply_mode_drag`]).
fn mode_mods(mode: TransformMode, hit: Hit, mut mods: egui::Modifiers) -> egui::Modifiers {
    match (mode, hit) {
        (TransformMode::Distort, Hit::Corner(_)) => mods.command = true,
        (TransformMode::Perspective, Hit::Corner(_)) => {
            mods.command = true;
            mods.alt = true;
            mods.shift = true;
        }
        _ => {}
    }
    mods
}

/// A handle drag in Edit › Transform `mode` (`Free` for Free Transform). `type_layer`: the box
/// moves editable type, which only takes affine transforms.
pub(crate) fn apply_mode_drag(s: &mut TransformSession, g: Gesture, p: [f64; 2], mode: TransformMode, type_layer: bool, mut mods: egui::Modifiers) {
    if !mode_allows(mode, g.hit) {
        return;
    }
    match (mode, g.hit) {
        // Skew, as in Photoshop (#3437): a corner moves alone along one of its two edges,
        // whichever the drag mostly follows; ⌥ moves the opposite corner the other way.
        (TransformMode::Skew, Hit::Corner(i)) if i < 4 => {
            let (d, q0) = ([p[0] - g.start[0], p[1] - g.start[1]], g.quad0);
            // Corners 0..3 = TL, TR, BR, BL: the neighbour along the top/bottom edge, and along
            // the left/right one.
            let along = |j: usize| {
                let e = [q0[i][0] - q0[j][0], q0[i][1] - q0[j][1]];
                let l = e[0].hypot(e[1]);
                let u = if l > 0.0 { [e[0] / l, e[1] / l] } else { [0.0, 0.0] };
                let t = d[0] * u[0] + d[1] * u[1];
                [t * u[0], t * u[1]]
            };
            let (hn, vn) = ([1, 0, 3, 2][i], [3, 2, 1, 0][i]);
            let (h, v) = (along(hn), along(vn));
            let (m, partner) = if h[0].hypot(h[1]) >= v[0].hypot(v[1]) { (h, hn) } else { (v, vn) };
            // Editable type stays a parallelogram: the corner's whole edge slides with it.
            let moved: &[usize] = if type_layer { &[i, partner] } else { &[i] };
            let mut q = q0;
            for k in 0..4 {
                let sign = if moved.contains(&k) {
                    1.0
                } else if mods.alt && (if type_layer { true } else { k == (i + 2) % 4 }) {
                    // ⌥: the opposite corner (on type, the opposite edge) the other way.
                    -1.0
                } else {
                    continue;
                };
                if let (Some(dst), Some(src)) = (q.get_mut(k), q0.get(k)) {
                    *dst = [src[0] + sign * m[0], src[1] + sign * m[1]];
                }
            }
            s.quad = keep_convex(q0, q);
            s.pivot = carry_pivot(q0, s.quad, g.pivot0);
        }
        // Rotate (#3941): a corner or edge turns about the reference point, the same drag as the
        // zone just outside the box. Free Transform still scales from those handles.
        (TransformMode::Rotate, Hit::Corner(_) | Hit::Edge(_)) => {
            let mut g = g;
            g.hit = Hit::Outside;
            apply_drag(s, g, p, mods);
        }
        // Skew: an edge slides only along itself (⌘⇧ in Free Transform).
        (TransformMode::Skew, Hit::Edge(_)) => {
            mods.command = true;
            mods.shift = true;
            apply_drag(s, g, p, mods);
        }
        _ => apply_drag(s, g, p, mode_mods(mode, g.hit, mods)),
    }
}

/// What a drag can do in each Edit › Transform mode: in Distort only the corners (and moving the
/// box from inside); in Skew everything but rotating from outside the box (#3437). A click
/// outside still commits.
fn mode_allows(mode: TransformMode, h: Hit) -> bool {
    match mode {
        TransformMode::Distort => !matches!(h, Hit::Outside | Hit::Edge(_)),
        TransformMode::Skew => h != Hit::Outside,
        _ => true,
    }
}

/// The turn at each corner of a quad (cross products of consecutive edges): all the same sign
/// for a convex quad, mixed for a concave or self-intersecting one.
fn turns(q: &[[f64; 2]; 4]) -> [f64; 4] {
    std::array::from_fn(|k| {
        let (a, b, c) = (q[k], q[(k + 1) % 4], q[(k + 2) % 4]);
        (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
    })
}

/// Is `q` convex with the orientation `sign`, every corner turning by more than a sliver?
fn convex(q: &[[f64; 2]; 4], sign: f64, min_turn: f64) -> bool {
    turns(q).iter().all(|t| t * sign > min_turn)
}

/// Distort and Perspective stop a corner where the box would turn concave or fold over itself,
/// as Photoshop does (#1323): the result moves from `from` towards `to` as far as the quad stays
/// convex. A box that is not convex to begin with is left free.
fn keep_convex(from: [[f64; 2]; 4], to: [[f64; 2]; 4]) -> [[f64; 2]; 4] {
    let t0 = turns(&from);
    let sign = t0[0].signum();
    // A sliver: a thousandth of the starting box's smallest turn, so the stop is short of flat.
    let min_turn = t0.iter().map(|t| t.abs()).fold(f64::INFINITY, f64::min) * 1e-3;
    if !(min_turn.is_finite() && min_turn > 0.0 && convex(&from, sign, 0.0)) || convex(&to, sign, min_turn) {
        return to;
    }
    let lerp = |t: f64| -> [[f64; 2]; 4] { std::array::from_fn(|k| [from[k][0] + (to[k][0] - from[k][0]) * t, from[k][1] + (to[k][1] - from[k][1]) * t]) };
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..40 {
        let mid = (lo + hi) / 2.0;
        if convex(&lerp(mid), sign, min_turn) { lo = mid } else { hi = mid }
    }
    lerp(lo)
}

pub(crate) fn apply_drag(s: &mut TransformSession, g: Gesture, p: [f64; 2], mods: egui::Modifiers) {
    let (dx, dy) = (p[0] - g.start[0], p[1] - g.start[1]);
    match g.hit {
        Hit::Inside => {
            // ⇧ locks the move to the axes and diagonals; the current offset keeps the lock stable.
            let (dx, dy) = if mods.shift {
                let prev = [s.quad[0][0] - g.quad0[0][0], s.quad[0][1] - g.quad0[0][1]];
                let d = crate::move_mods::constrain([dx, dy], Some(prev), crate::move_mods::TRANSFORM_DIRECTIONS);
                (d[0], d[1])
            } else {
                (dx, dy)
            };
            s.quad = g.quad0.map(|q| [q[0] + dx, q[1] + dy]);
            s.pivot = [g.pivot0[0] + dx, g.pivot0[1] + dy];
        }
        Hit::Pivot => s.pivot = p,
        Hit::Outside => {
            let a0 = (g.start[1] - g.pivot0[1]).atan2(g.start[0] - g.pivot0[0]);
            let a1 = (p[1] - g.pivot0[1]).atan2(p[0] - g.pivot0[0]);
            let mut da = a1 - a0;
            if mods.shift {
                // Snap the resulting angle to 15° steps.
                let base = (g.quad0[1][1] - g.quad0[0][1]).atan2(g.quad0[1][0] - g.quad0[0][0]);
                let step = 15f64.to_radians();
                da = ((base + da) / step).round() * step - base;
            }
            let (sn, cs) = da.sin_cos();
            let c = g.pivot0;
            s.quad = g.quad0.map(|q| {
                let (x, y) = (q[0] - c[0], q[1] - c[1]);
                [c[0] + x * cs - y * sn, c[1] + x * sn + y * cs]
            });
        }
        Hit::Corner(i) if mods.command && mods.alt && mods.shift => {
            // Perspective: the corner moves along the dominant axis and the corner sharing that
            // side moves the opposite way.
            s.quad = g.quad0;
            let horizontal = dx.abs() >= dy.abs();
            // Corners 0..3 = TL, TR, BR, BL: horizontal drags pair along the top/bottom edge,
            // vertical ones along the left/right edge.
            let j = match (i, horizontal) {
                (0, true) => 1,
                (1, true) => 0,
                (2, true) => 3,
                (3, true) => 2,
                (0, false) => 3,
                (3, false) => 0,
                (1, false) => 2,
                _ => 1,
            };
            let (mx, my) = if horizontal { (dx, 0.0) } else { (0.0, dy) };
            let mut q = g.quad0;
            q[i] = [g.quad0[i][0] + mx, g.quad0[i][1] + my];
            q[j] = [g.quad0[j][0] - mx, g.quad0[j][1] - my];
            s.quad = keep_convex(g.quad0, q);
            s.pivot = carry_pivot(g.quad0, s.quad, g.pivot0);
        }
        Hit::Corner(i) if mods.command => {
            // Distort: move the corner freely, up to where the box would fold over (#1323).
            let mut q = g.quad0;
            q[i] = [g.quad0[i][0] + dx, g.quad0[i][1] + dy];
            s.quad = keep_convex(g.quad0, q);
            s.pivot = carry_pivot(g.quad0, s.quad, g.pivot0);
        }
        Hit::Edge(i) if mods.command => {
            // Skew: the edge's two corners move together (⇧: only along the edge).
            let (a, b) = (i, (i + 1) % 4);
            let (mut mx, mut my) = (dx, dy);
            if mods.shift {
                let e = [g.quad0[b][0] - g.quad0[a][0], g.quad0[b][1] - g.quad0[a][1]];
                let l2 = e[0] * e[0] + e[1] * e[1];
                let t = if l2 > 0.0 { (dx * e[0] + dy * e[1]) / l2 } else { 0.0 };
                (mx, my) = (t * e[0], t * e[1]);
            }
            s.quad = g.quad0;
            s.quad[a] = [g.quad0[a][0] + mx, g.quad0[a][1] + my];
            s.quad[b] = [g.quad0[b][0] + mx, g.quad0[b][1] + my];
            if mods.alt {
                // ⌥: the opposite edge skews the other way (about the centre).
                let (c, d) = ((i + 2) % 4, (i + 3) % 4);
                s.quad[c] = [g.quad0[c][0] - mx, g.quad0[c][1] - my];
                s.quad[d] = [g.quad0[d][0] - mx, g.quad0[d][1] - my];
            }
            s.pivot = carry_pivot(g.quad0, s.quad, g.pivot0);
        }
        Hit::Corner(_) | Hit::Edge(_) => {
            // Work in the box's own (unit) frame so rotated/skewed boxes scale along their axes.
            let Some(h) = Homography::rect_to_quad([0.0, 0.0, 1.0, 1.0], g.quad0) else { return };
            let Some(inv) = h.inverse() else { return };
            let (u, v) = inv.apply(p[0], p[1]);
            let (pu, pv) = inv.apply(g.pivot0[0], g.pivot0[1]);
            // Moving edges/corner in unit space: [u0, v0, u1, v1] starting as [0, 0, 1, 1].
            let mut r = [0.0f64, 0.0, 1.0, 1.0];
            let (mu, mv) = match g.hit {
                Hit::Corner(0) => ((true, false), (true, false)),
                Hit::Corner(1) => ((false, true), (true, false)),
                Hit::Corner(2) => ((false, true), (false, true)),
                Hit::Corner(3) => ((true, false), (false, true)),
                Hit::Edge(0) => ((false, false), (true, false)),
                Hit::Edge(1) => ((false, true), (false, false)),
                Hit::Edge(2) => ((false, false), (false, true)),
                _ => ((true, false), (false, false)),
            };
            if mu.0 {
                r[0] = u;
            }
            if mu.1 {
                r[2] = u;
            }
            if mv.0 {
                r[1] = v;
            }
            if mv.1 {
                r[3] = v;
            }
            // ⌥: symmetric about the reference point.
            if mods.alt {
                if mu.0 {
                    r[2] = 2.0 * pu - r[0];
                }
                if mu.1 {
                    r[0] = 2.0 * pu - r[2];
                }
                if mv.0 {
                    r[3] = 2.0 * pv - r[1];
                }
                if mv.1 {
                    r[1] = 2.0 * pv - r[3];
                }
            }
            // Corners and edges scale proportionally by default; ⇧ frees them (the legacy
            // preference swaps the two for corners and always stretches edges, `legacy_mods`).
            let corner = matches!(g.hit, Hit::Corner(_));
            if !corner && !mods.shift {
                // An edge sets the scale of its own axis, and the other axis follows by the same
                // factor about the middle of the box (⌥: about the reference point, as for
                // corners), so the opposite edge's handle stays put. Past the opposite edge only
                // the dragged axis flips.
                if mu.0 || mu.1 {
                    let k = (r[2] - r[0]).abs();
                    let c = if mods.alt { pv } else { 0.5 };
                    r[1] = c - k / 2.0;
                    r[3] = c + k / 2.0;
                } else {
                    let k = (r[3] - r[1]).abs();
                    let c = if mods.alt { pu } else { 0.5 };
                    r[0] = c - k / 2.0;
                    r[2] = c + k / 2.0;
                }
            }
            if corner && !mods.shift {
                let (sx, sy) = (r[2] - r[0], r[3] - r[1]);
                let k = if sx.abs() > sy.abs() { sx.abs() } else { sy.abs() };
                let (nx, ny) = (k * sx.signum(), k * sy.signum());
                let (ax, ay) = if mods.alt { (pu, pv) } else { (if mu.0 { r[2] } else { r[0] }, if mv.0 { r[3] } else { r[1] }) };
                let fx = if mods.alt {
                    0.5
                } else if mu.0 {
                    1.0
                } else {
                    0.0
                };
                let fy = if mods.alt {
                    0.5
                } else if mv.0 {
                    1.0
                } else {
                    0.0
                };
                r[0] = ax - nx * fx;
                r[2] = r[0] + nx;
                r[1] = ay - ny * fy;
                r[3] = r[1] + ny;
            }
            let map = |a: f64, b: f64| {
                let (x, y) = h.apply(a, b);
                [x, y]
            };
            s.quad = [map(r[0], r[1]), map(r[2], r[1]), map(r[2], r[3]), map(r[0], r[3])];
            // ⌥ scales about the reference point, so it stays put; otherwise it rides along with
            // the box, keeping its place in it (#1329).
            s.pivot = if mods.alt { g.pivot0 } else { carry_pivot(g.quad0, s.quad, g.pivot0) };
        }
    }
}

/// Where the reference point lands when the box goes from `quad0` to `quad`: the same place
/// within the box, as in Photoshop, where it is attached to the bounding box (#1329). A box
/// that has collapsed leaves it where it was.
fn carry_pivot(quad0: [[f64; 2]; 4], quad: [[f64; 2]; 4], pivot0: [f64; 2]) -> [f64; 2] {
    let unit = [0.0, 0.0, 1.0, 1.0];
    let (Some(h0), Some(h1)) = (Homography::rect_to_quad(unit, quad0), Homography::rect_to_quad(unit, quad)) else { return pivot0 };
    let Some(inv) = h0.inverse() else { return pivot0 };
    let (u, v) = inv.apply(pivot0[0], pivot0[1]);
    let (x, y) = h1.apply(u, v);
    if x.is_finite() && y.is_finite() { [x, y] } else { pivot0 }
}

/// An anchor or a handle on a section line. The inner control points of a patch are not shown.
fn on_section_edge(i: usize, j: usize) -> bool {
    i.is_multiple_of(3) || j.is_multiple_of(3)
}

/// Index of the warp control point within `tol` of `p` (nearest anchor or handle).
fn warp_hit(w: &Warp, p: [f64; 2], tol: f64) -> Option<usize> {
    let m = w.mesh.as_ref()?;
    let nx = m.nx();
    m.points
        .iter()
        .enumerate()
        .filter(|(i, _)| on_section_edge(i % nx, i / nx))
        .map(|(i, q)| (i, ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)).sqrt()))
        .filter(|(_, d)| *d < tol)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}

/// A 1×1 mesh (Grid › Default) draws rule-of-thirds guides. Denser presets are real patch
/// boundaries only — a 3×3 is three cells, not a third-line lattice inside every cell.
fn single_patch(mesh: &BezierMesh) -> bool {
    mesh.us.len() == 2 && mesh.vs.len() == 2
}

/// The surface parameters under `p` when it is inside the warped mesh, else None.
fn surface_at(mesh: &BezierMesh, p: [f64; 2]) -> Option<(f64, f64)> {
    let (s, t) = mesh.param_at(p);
    let q = mesh.eval(s, t);
    ((q[0] - p[0]).hypot(q[1] - p[1]) < 1.0).then_some((s, t))
}

/// Ctrl-click (or Option-click) picks the split: crosswise in the open, and the perpendicular
/// split when the pointer is close to an existing grid line.
fn quick_split_kind(mesh: &BezierMesh, bounds: [f64; 4], p: [f64; 2]) -> SplitTool {
    let (s, t) = mesh.param_at(p);
    let w = (bounds[2] - bounds[0]).abs().max(1.0);
    let h = (bounds[3] - bounds[1]).abs().max(1.0);
    let su = (8.0 / w).clamp(0.02, 0.12);
    let sv = (8.0 / h).clamp(0.02, 0.12);
    let thirds = single_patch(mesh);
    let near = |knots: &[f64], v: f64, tol: f64| {
        knots.iter().any(|k| *k > 0.0 && *k < 1.0 && (k - v).abs() < tol) || (thirds && [1.0 / 3.0, 2.0 / 3.0].iter().any(|k| (k - v).abs() < tol))
    };
    match (near(&mesh.us, s, su), near(&mesh.vs, t, sv)) {
        (true, false) => SplitTool::Horizontal,
        (false, true) => SplitTool::Vertical,
        _ => SplitTool::Cross,
    }
}

fn split_cursor(tool: SplitTool) -> CursorIcon {
    match tool {
        SplitTool::Vertical => CursorIcon::ResizeHorizontal,
        SplitTool::Horizontal => CursorIcon::ResizeVertical,
        SplitTool::Cross => CursorIcon::Crosshair,
    }
}

/// Armed split tool, or Option held: the guide tracks the pointer and the split lands on release.
/// Returns false when this event is an ordinary control-point drag.
fn split_gesture(app: &mut PhotocraftApp, ev: ToolEvent, mods: egui::Modifiers) -> bool {
    let armed = app.transform_preview.as_ref().and_then(|p| p.split_tool);
    let placing = app.transform_preview.as_ref().is_some_and(|p| p.split_placing);
    match ev {
        ToolEvent::Down { x, y, .. } if armed.is_some() || mods.alt || mods.command => {
            if let Some(pv) = app.transform_preview.as_mut() {
                pv.split_placing = true;
                pv.split_pointer = Some([x, y]);
                pv.split_quick = armed.is_none();
            }
            true
        }
        ToolEvent::Move { x, y, .. } if placing => {
            if let Some(pv) = app.transform_preview.as_mut() {
                pv.split_pointer = Some([x, y]);
            }
            true
        }
        ToolEvent::Up { x, y } if placing => {
            let quick = app.transform_preview.as_ref().is_some_and(|p| p.split_quick);
            if let Some(pv) = app.transform_preview.as_mut() {
                pv.split_placing = false;
                pv.split_pointer = None;
                pv.split_quick = false;
            }
            let kind = if let Some(tool) = armed {
                Some(tool)
            } else if quick {
                app.ui.transform.as_ref().and_then(|t| t.warp.as_ref()).map(|w| quick_split_kind(&w.to_mesh(1, 1), w.bounds, [x, y]))
            } else {
                None
            };
            if let Some(kind) = kind {
                // On an existing line, or a click the drag path also delivers, this is a no-op.
                let _ = split(app, kind.command(), Some([x, y]));
            }
            true
        }
        _ => false,
    }
}

/// Drags the warp grid. A press on an anchor or handle moves that point (an anchor carries its
/// handles along); a press anywhere else inside the grid pulls the surface under the pointer.
/// Preset warps turn into a custom mesh when the press grabs something. An armed split tool, or
/// Ctrl / Option held, places a split where the pointer is released instead.
fn warp_pointer(app: &mut PhotocraftApp, ev: ToolEvent, tol: f64, mods: egui::Modifiers) {
    // Type takes only the warp styles: nothing on the canvas to drag or split (#3004).
    if warps_type(app) {
        return;
    }
    if split_gesture(app, ev, mods) {
        return;
    }
    let (Some(t), Some(pv)) = (app.ui.transform.as_mut(), app.transform_preview.as_mut()) else { return };
    let Some(w) = t.warp.as_mut() else { return };
    match ev {
        ToolEvent::Down { x, y, .. } => {
            // A preset warp becomes a custom mesh only when the press grabs one of its points.
            let custom = if w.mesh.is_none() || w.style != WarpStyle::Custom { Warp::custom(w.to_mesh(1, 1), w.bounds) } else { w.clone() };
            let pts0 = custom.mesh.as_ref().map(|m| m.points.clone()).unwrap_or_default();
            pv.warp_drag = warp_hit(&custom, [x, y], tol).map(|index| WarpDrag::Point { index, start: [x, y], pts0: pts0.clone() }).or_else(|| {
                let (s, t) = surface_at(custom.mesh.as_ref()?, [x, y])?;
                Some(WarpDrag::Surface { s, t, start: [x, y], pts0 })
            });
            if pv.warp_drag.is_some() {
                *w = custom;
            }
        }
        ToolEvent::Move { x, y, .. } | ToolEvent::Up { x, y } => {
            match (&pv.warp_drag, w.mesh.as_mut()) {
                (Some(WarpDrag::Point { index, start, pts0 }), Some(m)) => m.drag_point(pts0, *index, [x - start[0], y - start[1]]),
                (Some(WarpDrag::Surface { s, t, start, pts0 }), Some(m)) => m.pull(pts0, *s, *t, [x - start[0], y - start[1]]),
                _ => {}
            }
            if matches!(ev, ToolEvent::Up { .. }) {
                pv.warp_drag = None;
            }
        }
    }
}

/// Runs a warp-edit command against the mesh in the open Warp session and writes the mesh back.
pub fn edit_session_warp(app: &mut PhotocraftApp, id: &str, params: &serde_json::Value) -> Result<(), String> {
    let Some(w) = app.ui.transform.as_ref().and_then(|t| t.warp.clone()) else { return Err("not warping".into()) };
    if warps_type(app) {
        return Err("Warp on type takes only the warp styles".into());
    }
    let mut p = if params.is_object() { params.clone() } else { json!({}) };
    if p.get("warp").is_none() {
        p["warp"] = serde_json::to_value(&w).map_err(|e| e.to_string())?;
    }
    let r = app.session.execute(id, p).map_err(|e| e.to_string())?;
    let raw = r.get("warp").cloned().ok_or_else(|| "warp command returned no mesh".to_string())?;
    let nw: Warp = serde_json::from_value(raw).map_err(|e| e.to_string())?;
    // A warp-edit command can replace the mesh while pointer input is split across control
    // requests. Any drag index captured from the old mesh is invalid once that happens.
    if let Some(pv) = app.transform_preview.as_mut() {
        pv.warp_drag = None;
    }
    if let Some(t) = app.ui.transform.as_mut() {
        t.warp = Some(nw);
    }
    Ok(())
}

/// Applies a split-warp command to the mesh being edited (at the box centre unless `at`).
pub fn split(app: &mut PhotocraftApp, id: &str, at: Option<[f64; 2]>) -> Result<(), String> {
    let mut p = json!({});
    if let Some(a) = at {
        p["at"] = json!(a);
    }
    edit_session_warp(app, id, &p)
}

/// True just outside a corner: rotate handle feedback lives here, not over the entire
/// empty canvas. The actual rotation gesture continues to work farther away.
fn near_rotate_corner(t: &TransformSession, p: [f64; 2], tol: f64) -> bool {
    t.warp.is_none()
        && t.mode != TransformMode::Distort
        && {
            let h = hit(t, p, tol);
            h == Hit::Outside || (t.mode == TransformMode::Rotate && matches!(h, Hit::Corner(_) | Hit::Edge(_)))
        }
        && t.quad.iter().any(|q| (p[0] - q[0]).hypot(p[1] - q[1]) <= tol * 3.0)
}

/// Clockwise degrees in canvas coordinates (positive Y points down).
fn rotation_degrees(quad: [[f64; 2]; 4]) -> f64 {
    let dx = quad[1][0] - quad[0][0];
    let dy = quad[1][1] - quad[0][1];
    if dx.is_finite() && dy.is_finite() && dx.hypot(dy) > 1e-9 { dy.atan2(dx).to_degrees() } else { 0.0 }
}

/// Where the rotate cursor's arc bends at document point `p`, or `None` outside the rotate zone:
/// just outside a corner, or anywhere while a rotate drag is on. It points into the box along the
/// bisector of the nearest corner, so each corner gets its own arrow and they turn with the box.
pub(crate) fn rotate_cursor_dir(app: &PhotocraftApp, p: [f64; 2]) -> Option<[f64; 2]> {
    rotate_dir(app.ui.transform.as_ref()?, app.transform_preview.as_ref(), p, handle_tolerance(app))
}

fn rotate_dir(t: &TransformSession, pv: Option<&TransformPreview>, p: [f64; 2], tol: f64) -> Option<[f64; 2]> {
    // Content-Aware Scale's box only scales, and Skew doesn't rotate: no rotate arrow.
    if !(p[0].is_finite() && p[1].is_finite())
        || t.warp.is_some()
        || t.content_aware.is_some()
        || matches!(t.mode, TransformMode::Distort | TransformMode::Skew)
    {
        return None;
    }
    let rotating = pv.is_some_and(|pv| pv.gesture.is_some_and(|g| g.hit == Hit::Outside));
    if !rotating && !near_rotate_corner(t, p, tol) {
        return None;
    }
    let dist = |q: [f64; 2]| (p[0] - q[0]).hypot(p[1] - q[1]);
    let i = (0..4).min_by(|&a, &b| dist(t.quad[a]).total_cmp(&dist(t.quad[b])))?;
    let unit = |to: [f64; 2]| {
        let d = [to[0] - t.quad[i][0], to[1] - t.quad[i][1]];
        let l = d[0].hypot(d[1]);
        (l.is_finite() && l > 1e-9).then(|| [d[0] / l, d[1] / l])
    };
    let (a, b) = (unit(t.quad[(i + 1) % 4])?, unit(t.quad[(i + 3) % 4])?);
    let s = [a[0] + b[0], a[1] + b[1]];
    let l = s[0].hypot(s[1]);
    (l.is_finite() && l > 1e-9).then(|| [s[0] / l, s[1] / l])
}

/// Cursor for hovering a document point while transforming. `quick_split` is Ctrl or Option,
/// which arms a quick split while a warp is active.
pub fn cursor(app: &PhotocraftApp, p: [f64; 2], quick_split: bool) -> Option<CursorIcon> {
    let t = app.ui.transform.as_ref()?;
    let tol = handle_tolerance(app);
    if let Some(w) = &t.warp {
        if warps_type(app) {
            return Some(CursorIcon::Default);
        }
        if let Some(tool) = app.transform_preview.as_ref().and_then(|pv| pv.split_tool) {
            return Some(split_cursor(tool));
        }
        if quick_split {
            return Some(split_cursor(quick_split_kind(&w.to_mesh(1, 1), w.bounds, p)));
        }
        if w.style != WarpStyle::Custom || warp_hit(w, p, tol).is_some() {
            return Some(CursorIcon::Crosshair);
        }
        return Some(if w.mesh.as_ref().and_then(|m| surface_at(m, p)).is_some() { CursorIcon::Grab } else { CursorIcon::Default });
    }
    let h = hit(t, p, tol);
    if !mode_allows(t.mode, h) || (t.content_aware.is_some() && h == Hit::Outside) {
        return Some(CursorIcon::Default);
    }
    if rotate_cursor_dir(app, p).is_some() {
        // The overlay draws the curved rotate arrow (`rotate_cursor_dir`), also during the drag.
        return Some(CursorIcon::None);
    }
    Some(match h {
        Hit::Corner(0 | 2) => CursorIcon::ResizeNwSe,
        Hit::Corner(_) => CursorIcon::ResizeNeSw,
        Hit::Edge(0 | 2) => CursorIcon::ResizeVertical,
        Hit::Edge(_) => CursorIcon::ResizeHorizontal,
        Hit::Pivot => CursorIcon::Crosshair,
        Hit::Inside => CursorIcon::Move,
        Hit::Outside => CursorIcon::Alias,
    })
}

/// For each vertex of the preview's `n` × `n` grid over the unit square (row by row), whether it
/// maps to the front side of `h`'s horizon. A Distort dragged past the horizon sends the rest of
/// the image to the far side, where its cells would wrap round infinity; the preview leaves them
/// out, as the committed warp does (#4863).
fn front_vertices(h: &Homography, n: usize) -> Vec<bool> {
    let front = h.front_side([0.0, 0.0, 1.0, 1.0]);
    (0..=n).flat_map(|j| (0..=n).map(move |i| (i as f64 / n as f64, j as f64 / n as f64))).map(|(u, v)| front * h.w(u, v) > 1e-9).collect()
}

/// Warped preview of the moving pixels plus the box, handles and reference point.
pub fn draw_overlay(app: &PhotocraftApp, painter: &egui::Painter, xf: &ViewXform) {
    let (Some(t), Some(pv)) = (&app.ui.transform, &app.transform_preview) else { return };
    let scr = |q: [f64; 2]| xf.to_screen(q[0] as f32, q[1] as f32);
    if let Some(w) = &t.warp {
        let over_canvas = painter.ctx().input(|i| i.pointer.hover_pos()).filter(|p| xf.rect.contains(*p)).map(|p| xf.to_doc(p));
        let at = pv.split_pointer.or(over_canvas);
        let alt = painter.ctx().input(|i| i.modifiers.alt || i.modifiers.command);
        let guide = at.filter(|_| !warps_type(app)).and_then(|p| {
            let tool = pv.split_tool.or_else(|| (alt || pv.split_quick).then(|| quick_split_kind(&w.to_mesh(1, 1), w.bounds, p)));
            tool.map(|tool| (tool, p))
        });
        draw_warp(painter, xf, t, w, pv, guide);
        return;
    }
    let ctx = painter.ctx();
    let styled = styled_frame(app, xf.zoom * ctx.pixels_per_point(), ctx.input(|i| i.pointer.any_down())).and_then(|(s, frame)| s.texture(ctx, &frame));
    if let Some((tex, r)) = styled {
        let mut mesh = egui::Mesh::with_texture(tex);
        for (x, y, u, v) in [(r[0], r[1], 0.0, 0.0), (r[2], r[1], 1.0, 0.0), (r[2], r[3], 1.0, 1.0), (r[0], r[3], 0.0, 1.0)] {
            mesh.vertices.push(egui::epaint::Vertex { pos: pos2(x, y), uv: pos2(u, v), color: Color32::WHITE });
        }
        mesh.add_triangle(0, 1, 2);
        mesh.add_triangle(0, 2, 3);
        painter.add(clip_to_canvas(mesh, canvas_size(pv), |p| xf.to_screen(p.x, p.y)));
    } else if let Some(h) = Homography::rect_to_quad([0.0, 0.0, 1.0, 1.0], t.quad) {
        // A 24×24 grid keeps perspective previews straight.
        let n = 24;
        let zoom = xf.zoom * painter.ctx().pixels_per_point();
        // Content-Aware Scale shows its latest live result once there is one.
        let (tex, uv) = match pv.content_aware.as_ref().and_then(|l| l.shown(t)) {
            Some((texture, texel)) => texture.pick(painter.ctx(), zoom, texel, false),
            None => pv.texture.pick(painter.ctx(), zoom, quad_scale(t), t.interpolation == "nearest"),
        };
        let mut mesh = egui::Mesh::with_texture(tex);
        let tint = Color32::from_white_alpha((pv.opacity.clamp(0.0, 1.0) * 255.0) as u8);
        let ahead = front_vertices(&h, n);
        for j in 0..=n {
            for i in 0..=n {
                let (u, v) = (i as f64 / n as f64, j as f64 / n as f64);
                let (x, y) = h.apply(u, v);
                mesh.vertices.push(egui::epaint::Vertex { pos: pos2(x as f32, y as f32), uv: pos2(u as f32 * uv[0], v as f32 * uv[1]), color: tint });
            }
        }
        for j in 0..n {
            for i in 0..n {
                let a = (j * (n + 1) + i) as u32;
                for tri in [[a, a + 1, a + n as u32 + 2], [a, a + n as u32 + 2, a + n as u32 + 1]] {
                    // A cell with a corner past the horizon would stretch across the canvas (#4863).
                    if tri.iter().all(|&k| ahead[k as usize]) {
                        mesh.add_triangle(tri[0], tri[1], tri[2]);
                    }
                }
            }
        }
        painter.add(clip_to_canvas(mesh, canvas_size(pv), |p| xf.to_screen(p.x, p.y)));
    }
    if let Some(a) = &pv.above {
        a.draw(painter, xf);
    }
    if pv.committing.is_some() {
        return;
    }
    let accent = crate::theme::Tokens::get(painter.ctx()).accent;
    if let (Some(path), Some(h)) = (&pv.path, Homography::rect_to_quad(t.rect, t.quad)) {
        crate::vector_ui::draw_outline(
            painter,
            path,
            &|q| {
                let (x, y) = h.apply(q[0], q[1]);
                xf.to_screen(x as f32, y as f32)
            },
            accent,
        );
    }
    let pts: Vec<Pos2> = t.quad.iter().map(|q| scr(*q)).collect();
    painter.add(egui::Shape::closed_line(pts.clone(), Stroke::new(1.0, accent)));
    let mids: Vec<Pos2> = (0..4).map(|i| pts[i].lerp(pts[(i + 1) % 4], 0.5)).collect();
    for p in pts.iter().chain(mids.iter()) {
        let r = egui::Rect::from_center_size(*p, vec2(7.0, 7.0));
        painter.rect_filled(r, 0.0, Color32::WHITE);
        painter.rect_stroke(r, 0.0, Stroke::new(1.0, accent), egui::StrokeKind::Inside);
    }
    // Reference point (Photoshop's crosshair circle).
    let c = scr(t.pivot);
    painter.circle_stroke(c, 5.0, Stroke::new(1.0, Color32::WHITE));
    painter.circle_stroke(c, 5.0, Stroke::new(0.5, Color32::BLACK));
    painter.line_segment([c - vec2(8.0, 0.0), c + vec2(8.0, 0.0)], Stroke::new(1.0, Color32::WHITE));
    painter.line_segment([c - vec2(0.0, 8.0), c + vec2(0.0, 8.0)], Stroke::new(1.0, Color32::WHITE));
    if let Some(at) = painter.ctx().pointer_hover_pos().filter(|p| xf.rect.contains(*p)) {
        let p = xf.to_doc(at);
        let tol = HANDLE_PX / f64::from(xf.zoom.max(0.01));
        if let Some(k) = rotate_dir(t, Some(pv), p, tol) {
            // Through the view, so the arc bends round the corner as it shows under a turned or
            // flipped view.
            let toward = xf.to_screen((p[0] + k[0] * 16.0) as f32, (p[1] + k[1] * 16.0) as f32) - at;
            crate::crop_ui::draw_turn_cursor(painter.ctx(), at, toward);
        }
        // Preferences › Tools › Show Transformation Values off hides it, as in Photoshop (#204).
        if let Some(r) = live_readout(app).filter(|_| app.session.prefs().tools.show_transformation_values) {
            draw_live_readout(painter.ctx(), at, r);
        }
    }
}

/// What the styled preview of the moving layer is drawn from (transform_fx.rs), at `zoom` screen
/// pixels per document pixel. `None` for a layer without a style (also once it is switched off
/// while the box is open) or a hidden one: the bare pixels are drawn instead. The style, opacity
/// and fill are the layer's as they are now, so edits made while the box is open show.
pub(crate) fn styled_frame(app: &PhotocraftApp, zoom: f32, dragging: bool) -> Option<(&crate::transform_fx::StyledPreview, crate::transform_fx::Frame<'_>)> {
    let (t, pv) = (app.ui.transform.as_ref()?, app.transform_preview.as_ref()?);
    let styled = pv.styled.as_ref().filter(|_| t.warp.is_none())?;
    let st = app.session.active().filter(|st| st.doc.id == pv.doc.id)?;
    let layer = st.doc.layer(LayerId(t.layer)).filter(|l| l.visible && photocraft_compose::effects::has_effects(l))?;
    let frame = crate::transform_fx::Frame {
        doc: &st.doc,
        layer,
        revision: st.revision,
        texels: &pv.texture,
        quad: t.quad,
        quad_scale: quad_scale(t),
        nearest: t.interpolation == "nearest",
        zoom,
        dragging,
    };
    Some((styled, frame))
}

/// The value shown beside the pointer while the box is dragged, like Photoshop's Show
/// Transformation Values: the box's angle while rotating, its scale while resizing, the offset
/// while moving.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Readout {
    /// Clockwise degrees in document space (`rotation_degrees`), whatever the view's turn or flip.
    Angle(f64),
    /// Width and height in percent of the original box.
    Scale([f64; 2]),
    /// Document pixels moved since the press.
    Offset([f64; 2]),
}

/// The readout for the drag in progress, or `None` when nothing is dragged (or the drag is the
/// reference point, a warp, or a distortion that has no single scale).
pub(crate) fn live_readout(app: &PhotocraftApp) -> Option<Readout> {
    let t = app.ui.transform.as_ref().filter(|t| t.warp.is_none())?;
    let g = app.transform_preview.as_ref()?.gesture?;
    match g.hit {
        Hit::Outside => Some(Readout::Angle(rotation_degrees(t.quad))),
        Hit::Inside => Some(Readout::Offset([t.quad[0][0] - g.quad0[0][0], t.quad[0][1] - g.quad0[0][1]])),
        Hit::Corner(_) | Hit::Edge(_) if parallelogram(&t.quad) => {
            let (w, h, _, _) = readout(t);
            (w.is_finite() && h.is_finite()).then_some(Readout::Scale([w, h]))
        }
        Hit::Corner(_) | Hit::Edge(_) | Hit::Pivot => None,
    }
}

/// True when `q` is still an affine image of the box (opposite sides parallel), so a W/H scale
/// describes it; a distorted or perspective box has none.
fn parallelogram(q: &[[f64; 2]; 4]) -> bool {
    let scale = q.iter().flatten().fold(1.0f64, |m, v| m.max(v.abs()));
    (0..2).all(|k| ((q[0][k] + q[2][k]) - (q[1][k] + q[3][k])).abs() <= scale * 1e-9)
}

/// [`live_readout`] as the rounded label beside the pointer (the marquee's and Crop's readout).
fn draw_live_readout(ctx: &egui::Context, at: Pos2, r: Readout) {
    // Round first so a hair below zero never reads "-0".
    let fixed = |v: f64, places: i32| {
        let k = 10f64.powi(places);
        let v = (v * k).round() / k;
        if v == 0.0 { 0.0 } else { v }
    };
    let id = "transform-readout";
    match r {
        Readout::Angle(a) => crate::canvas::draw_readout(ctx, id, at, ["Angle:"], [format!("{:.1}°", fixed(a, 1))]),
        Readout::Scale([w, h]) => crate::canvas::draw_readout(ctx, id, at, ["W:", "H:"], [w, h].map(|v| format!("{:.1}%", fixed(v, 1)))),
        Readout::Offset([dx, dy]) => crate::canvas::draw_readout(ctx, id, at, ["ΔX:", "ΔY:"], [dx, dy].map(|v| format!("{:.0} px", fixed(v, 0)))),
    }
}

/// Width and height of the document being transformed, in document pixels.
fn canvas_size(pv: &TransformPreview) -> [f32; 2] {
    [pv.doc.size.width as f32, pv.doc.size.height as f32]
}

/// Clips `mesh` (positions in document pixels) to the canvas `[0, 0, size]` and maps it to the
/// screen. Pixels past the canvas edge are never shown, also not while transforming (Photoshop;
/// #2027): only the box and its handles reach over the pasteboard. Clipping in document space
/// stays exact under a rotated or flipped view, where a screen clip rectangle would not.
fn clip_to_canvas(mut mesh: egui::Mesh, size: [f32; 2], to_screen: impl Fn(Pos2) -> Pos2) -> egui::Mesh {
    let inside = |p: Pos2| p.x >= 0.0 && p.y >= 0.0 && p.x <= size[0] && p.y <= size[1];
    if mesh.vertices.iter().all(|v| inside(v.pos)) {
        for v in &mut mesh.vertices {
            v.pos = to_screen(v.pos);
        }
        return mesh;
    }
    let mut out = egui::Mesh::with_texture(mesh.texture_id);
    for tri in mesh.indices.as_chunks::<3>().0 {
        let mut poly: Vec<egui::epaint::Vertex> = tri.iter().filter_map(|i| mesh.vertices.get(*i as usize).copied()).collect();
        // Sutherland–Hodgman against the four canvas edges (`dist` >= 0 is the canvas side).
        for edge in 0..4 {
            let dist = |v: &egui::epaint::Vertex| match edge {
                0 => v.pos.x,
                1 => size[0] - v.pos.x,
                2 => v.pos.y,
                _ => size[1] - v.pos.y,
            };
            let mut next = Vec::with_capacity(poly.len() + 2);
            for (k, a) in poly.iter().enumerate() {
                let Some(b) = poly.get((k + 1) % poly.len()) else { continue };
                let (da, db) = (dist(a), dist(b));
                if da >= 0.0 {
                    next.push(*a);
                }
                if (da >= 0.0) != (db >= 0.0) && da != db {
                    let t = da / (da - db);
                    next.push(egui::epaint::Vertex { pos: a.pos.lerp(b.pos, t), uv: a.uv.lerp(b.uv, t), color: a.color });
                }
            }
            poly = next;
        }
        if poly.len() < 3 {
            continue;
        }
        let base = out.vertices.len() as u32;
        out.vertices.extend(poly.iter().map(|v| egui::epaint::Vertex { pos: to_screen(v.pos), ..*v }));
        for k in 1..poly.len() as u32 - 1 {
            out.add_triangle(base, base + k, base + k + 1);
        }
    }
    out
}

/// Warp preview (the moving pixels on a fine textured mesh) plus the control mesh. Patch
/// boundaries are solid. A single patch (Grid › Default) also draws its rule-of-thirds guides;
/// 3×3, 4×4 and 5×5 are just those even cells, with an anchor at every intersection. `guide` is
/// the split line following the pointer.
/// How near the pointer (screen px) a handle's anchor or the handle itself must be for the
/// handle to show. Everything else stays out of the way: only anchors are always drawn.
const HANDLE_REVEAL_PX: f32 = 56.0;

/// The anchor a handle belongs to, as `(i, j)` in the control grid (an anchor is its own).
fn handle_anchor(i: usize, j: usize) -> (usize, usize) {
    if i.is_multiple_of(3) {
        let aj = match j % 3 {
            0 => j,
            1 => j - 1,
            _ => j + 1,
        };
        (i, aj)
    } else {
        (if i % 3 == 1 { i - 1 } else { i + 1 }, j)
    }
}
/// Anchors always, small and quiet. A handle and its arm show only while the pointer is close to
/// them or they are being dragged, so a dense grid does not bury the picture in dots.
fn draw_warp_points(painter: &egui::Painter, m: &BezierMesh, scr: impl Fn([f64; 2]) -> Pos2, pv: &TransformPreview, line: Stroke) {
    let (nx, ny) = (m.nx(), m.ny());
    let hover = painter.ctx().input(|i| i.pointer.hover_pos());
    let dragged = match &pv.warp_drag {
        Some(WarpDrag::Point { index, .. }) => Some(handle_anchor(index % nx, index / nx)),
        _ => None,
    };
    let near = |p: Pos2| hover.is_some_and(|h| h.distance(p) <= HANDLE_REVEAL_PX);
    let faint = Stroke::new(0.75, line.color.gamma_multiply(0.8));
    for j in 0..ny {
        for i in 0..nx {
            if !on_section_edge(i, j) {
                continue;
            }
            let p = scr(m.point(i, j));
            if i.is_multiple_of(3) && j.is_multiple_of(3) {
                let r = egui::Rect::from_center_size(p, vec2(5.0, 5.0));
                painter.rect_filled(r, 0.0, Color32::WHITE);
                painter.rect_stroke(r, 0.0, faint, egui::StrokeKind::Inside);
                continue;
            }
            let (ai, aj) = handle_anchor(i, j);
            let a = scr(m.point(ai, aj));
            if !(near(p) || near(a) || dragged == Some((ai, aj))) {
                continue;
            }
            painter.line_segment([a, p], faint);
            painter.circle_filled(p, 2.75, Color32::WHITE);
            painter.circle_stroke(p, 2.75, faint);
        }
    }
}

fn draw_warp(painter: &egui::Painter, xf: &ViewXform, t: &TransformSession, w: &Warp, pv: &TransformPreview, guide: Option<(SplitTool, [f64; 2])>) {
    let scr = |q: [f64; 2]| xf.to_screen(q[0] as f32, q[1] as f32);
    let r = t.rect;
    let m = w.to_mesh(1, 1);
    let (us, vs) = crate::warp_preview::samples(&m, xf.zoom as f64);
    let (tex, uv) = pv.texture.pick(painter.ctx(), xf.zoom * painter.ctx().pixels_per_point(), 1.0, t.interpolation == "nearest");
    let mut mesh = egui::Mesh::with_texture(tex);
    let tint = Color32::from_white_alpha((pv.opacity.clamp(0.0, 1.0) * 255.0) as u8);
    for v in &vs {
        for u in &us {
            let (x, y) = w.map(r[0] + u * (r[2] - r[0]), r[1] + v * (r[3] - r[1]));
            mesh.vertices.push(egui::epaint::Vertex { pos: pos2(x as f32, y as f32), uv: pos2(*u as f32 * uv[0], *v as f32 * uv[1]), color: tint });
        }
    }
    let pos: Vec<[f32; 2]> = mesh.vertices.iter().map(|v| [v.pos.x, v.pos.y]).collect();
    let rest: Vec<[f32; 2]> = vs.iter().flat_map(|v| us.iter().map(move |u| [(r[0] + u * (r[2] - r[0])) as f32, (r[1] + v * (r[3] - r[1])) as f32])).collect();
    mesh.indices = crate::warp_preview::triangle_order(&pos, &rest, us.len());
    painter.add(clip_to_canvas(mesh, canvas_size(pv), |p| xf.to_screen(p.x, p.y)));
    if let Some(a) = &pv.above {
        a.draw(painter, xf);
    }
    let accent = crate::theme::Tokens::get(painter.ctx()).accent;
    // Transform Path › Warp: the path drawn through the warp.
    if let Some(path) = &pv.path {
        crate::vector_ui::draw_outline(
            painter,
            path,
            &|q| {
                let (x, y) = w.map(q[0], q[1]);
                xf.to_screen(x as f32, y as f32)
            },
            accent,
        );
    }
    let line = Stroke::new(1.0, accent);
    let thin = Stroke::new(0.75, accent.gamma_multiply(0.7));
    let curve = |fixed_u: Option<f64>, fixed_v: Option<f64>| -> Vec<Pos2> {
        (0..=40)
            .map(|k| {
                let s = k as f64 / 40.0;
                let p = m.eval(fixed_u.unwrap_or(s), fixed_v.unwrap_or(s));
                scr(p)
            })
            .collect()
    };
    // Patch boundaries are the grid. Only Default (one patch) draws the rule-of-thirds guides;
    // a 3×3 preset is three even cells, not those guides repeated inside every cell.
    let thirds = single_patch(&m);
    for (k, win) in m.us.windows(2).enumerate() {
        if thirds {
            for third in [1.0, 2.0] {
                painter.add(egui::Shape::line(curve(Some(win[0] + (win[1] - win[0]) * third / 3.0), None), thin));
            }
        }
        if k == 0 {
            painter.add(egui::Shape::line(curve(Some(win[0]), None), line));
        }
        painter.add(egui::Shape::line(curve(Some(win[1]), None), line));
    }
    for (k, win) in m.vs.windows(2).enumerate() {
        if thirds {
            for third in [1.0, 2.0] {
                painter.add(egui::Shape::line(curve(None, Some(win[0] + (win[1] - win[0]) * third / 3.0)), thin));
            }
        }
        if k == 0 {
            painter.add(egui::Shape::line(curve(None, Some(win[0])), line));
        }
        painter.add(egui::Shape::line(curve(None, Some(win[1])), line));
    }
    if w.style == WarpStyle::Custom {
        draw_warp_points(painter, &m, scr, pv, line);
    }
    if let Some((tool, p)) = guide {
        let (s, t) = m.param_at(p);
        let under = Stroke::new(3.0, Color32::BLACK);
        let over = Stroke::new(1.25, Color32::WHITE);
        let mut strokes = Vec::new();
        if matches!(tool, SplitTool::Vertical | SplitTool::Cross) {
            strokes.push(curve(Some(s), None));
        }
        if matches!(tool, SplitTool::Horizontal | SplitTool::Cross) {
            strokes.push(curve(None, Some(t)));
        }
        for pts in strokes {
            painter.add(egui::Shape::line(pts.clone(), under));
            painter.add(egui::Shape::line(pts, over));
        }
    }
}

/// How much the box enlarges its pixels: the longest edge relative to the original (picks the
/// preview's texture level and filter).
fn quad_scale(t: &TransformSession) -> f32 {
    let (w0, h0) = ((t.rect[2] - t.rect[0]).max(1e-9), (t.rect[3] - t.rect[1]).max(1e-9));
    let len = |a: [f64; 2], b: [f64; 2]| (b[0] - a[0]).hypot(b[1] - a[1]);
    let q = &t.quad;
    let s = (len(q[0], q[1]) / w0).max(len(q[3], q[2]) / w0).max(len(q[0], q[3]) / h0).max(len(q[1], q[2]) / h0);
    if s.is_finite() { s as f32 } else { 1.0 }
}

/// W and H (%), the angle (°) and the horizontal skew (°) the options bar shows for the box.
fn readout(t: &TransformSession) -> (f64, f64, f64, f64) {
    let p = Parts::of(t);
    (p.sx * 100.0, p.sy.abs() * 100.0, p.angle.to_degrees(), p.skew_h.to_degrees())
}

/// The box's affine part read as Photoshop's options bar does: the frame scaled (W, H), skewed
/// (H, V) and then rotated about the reference point. Angles are radians; positive turns
/// clockwise on screen, so a positive H skew leans the box's sides like the angle field would
/// turn them, and a positive V skew its top and bottom.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Parts {
    sx: f64,
    /// Negative when the box is flipped vertically relative to its rotation.
    sy: f64,
    angle: f64,
    skew_h: f64,
    skew_v: f64,
}

/// The skew fields' reach: a skew of ±90° would flatten the box.
const SKEW_LIMIT_DEG: f32 = 89.0;

impl Parts {
    /// The box's columns: where the frame's unit X and Y edges went (from the top-left corner).
    fn columns(t: &TransformSession) -> Option<[[f64; 2]; 2]> {
        let (w0, h0) = (t.rect[2] - t.rect[0], t.rect[3] - t.rect[1]);
        if w0.abs() < 1e-9 || h0.abs() < 1e-9 {
            return None;
        }
        let q = &t.quad;
        let m = [[(q[1][0] - q[0][0]) / w0, (q[1][1] - q[0][1]) / w0], [(q[3][0] - q[0][0]) / h0, (q[3][1] - q[0][1]) / h0]];
        m.iter().flatten().all(|v| v.is_finite()).then_some(m)
    }

    /// The box as scale, skew and rotation, keeping the session's typed V skew (any box can be
    /// written with any V skew short of ±90°).
    fn of(t: &TransformSession) -> Parts {
        let skew_v = t.skew_v.to_radians();
        let none = Parts { sx: 1.0, sy: 1.0, angle: 0.0, skew_h: 0.0, skew_v };
        Parts::columns(t).map_or(none, |m| Parts::from_columns(m, skew_v))
    }

    fn from_columns([c0, c1]: [[f64; 2]; 2], skew_v: f64) -> Parts {
        // X edge = sx · R(angle) · (1, tan v): its direction is the angle plus v.
        let angle = c0[1].atan2(c0[0]) - skew_v;
        let sx = c0[0].hypot(c0[1]) * skew_v.cos();
        // Y edge = sy · R(angle) · (−tan h, 1).
        let (gx, gy) = unturn(c1, angle);
        let skew_h = if gy.abs() < 1e-12 { 0.0 } else { (-gx / gy).atan() };
        Parts { sx, sy: gy, angle, skew_h, skew_v }
    }

    fn columns_of(self) -> [[f64; 2]; 2] {
        let (sn, cs) = self.angle.sin_cos();
        let rot = |x: f64, y: f64| [x * cs - y * sn, x * sn + y * cs];
        let c0 = rot(self.sx, self.sx * self.skew_v.tan());
        let c1 = rot(-self.sy * self.skew_h.tan(), self.sy);
        [c0, c1]
    }
}

/// `c` turned back by `angle` (radians).
fn unturn(c: [f64; 2], angle: f64) -> (f64, f64) {
    let (sn, cs) = angle.sin_cos();
    (c[0] * cs + c[1] * sn, -c[0] * sn + c[1] * cs)
}

/// Sets the box's H and V skew (degrees), keeping its scale, angle and reference point: the
/// options bar's skew fields.
fn skew_about_pivot(app: &mut PhotocraftApp, skew_h: f64, skew_v: f64) {
    let Some(s) = app.ui.transform.as_mut() else { return };
    let Some(m) = Parts::columns(s) else { return };
    let limit = f64::from(SKEW_LIMIT_DEG);
    let (skew_h, skew_v) = (skew_h.clamp(-limit, limit), skew_v.clamp(-limit, limit));
    let from = Parts::from_columns(m, s.skew_v.to_radians());
    let to = Parts { skew_h: skew_h.to_radians(), skew_v: skew_v.to_radians(), ..from }.columns_of();
    // The map from the old box to the new one: to · m⁻¹.
    let det = m[0][0] * m[1][1] - m[1][0] * m[0][1];
    if det.abs() < 1e-12 {
        return;
    }
    let inv = [[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]];
    let l = |x: f64, y: f64| {
        let (u, v) = (inv[0][0] * x + inv[1][0] * y, inv[0][1] * x + inv[1][1] * y);
        [to[0][0] * u + to[1][0] * v, to[0][1] * u + to[1][1] * v]
    };
    let c = s.pivot;
    let quad = s.quad.map(|q| {
        let d = l(q[0] - c[0], q[1] - c[1]);
        [c[0] + d[0], c[1] + d[1]]
    });
    if quad.iter().flatten().all(|v| v.is_finite()) {
        s.quad = quad;
        s.skew_v = skew_v;
    }
}

/// Width of the mode, cancel and commit cluster kept on the right of the options bar.
const ACTIONS_W: f32 = 140.0;
/// Width of the action buttons when they follow the Warp fields (one more than the box bar: Reset).
const WARP_ACTIONS_W: f32 = 176.0;

/// Reach of the reference point X/Y fields: the largest document side (`image.canvasSize`'s limit).
const POSITION_LIMIT_PX: f32 = 300_000.0;
/// Reach of the W/H fields. The engine takes any scale; this lets a 30 px layer span the largest
/// document (#1267).
const SCALE_LIMIT_PCT: f32 = 1_000_000.0;

/// The W or H field's limit for a box side of `side` px: as far as the result stays within the
/// largest document side, so a typed scale can't ask for an image too large to allocate.
fn scale_limit_pct(side: f64) -> f32 {
    let side = side.abs();
    if !side.is_finite() || side < 1e-6 {
        return SCALE_LIMIT_PCT;
    }
    ((f64::from(POSITION_LIMIT_PX) / side * 100.0) as f32).clamp(100.0, SCALE_LIMIT_PCT)
}

/// Where Free Transform remembers the unit of its W and H fields, picked by right-clicking either
/// of them as in Photoshop (Percent until another is picked).
const SIZE_UNIT_KEY: &str = "edit.freeTransform";

/// The W/H unit, with the document's resolution and the points per inch it converts with.
fn size_unit(app: &PhotocraftApp) -> crate::layer_props_ui::LenUnit {
    use photocraft_engine::prefs::Unit;
    let unit = app.session.prefs().dialogs.get(SIZE_UNIT_KEY).and_then(|v| v.get("sizeUnit")).and_then(serde_json::Value::as_str).and_then(Unit::parse);
    crate::layer_props_ui::LenUnit { unit: unit.unwrap_or(Unit::Percent), ..crate::layer_props_ui::LenUnit::of(app) }
}

fn set_size_unit(app: &mut PhotocraftApp, name: &str) {
    app.session.prefs.edit(|p| {
        let entry = p.dialogs.entry(SIZE_UNIT_KEY.to_string()).or_insert_with(|| json!({}));
        if !entry.is_object() {
            *entry = json!({});
        }
        entry["sizeUnit"] = json!(name);
    });
}

/// A W or H value shown in `u`: the box side `side` px wide at the original size, scaled by `pct`.
/// Percent stays the scale itself (100 % is the original size), lengths are the side's length.
fn size_shown(u: &crate::layer_props_ui::LenUnit, pct: f64, side: f64) -> f32 {
    u.shown(pct / 100.0 * side, side)
}

/// The scale (%) a W or H value typed in `u` asks for, for a box side `side` px wide originally.
fn size_pct(u: &crate::layer_props_ui::LenUnit, v: f32, side: f64) -> f64 {
    let side = side.abs().max(1e-9);
    u.px(v, side) / side * 100.0
}

/// Options bar while transforming: the fields (reference point X/Y, W/H %, angle, interpolation,
/// or the warp controls), then the mode switch, cancel and commit right after them. A bar too
/// narrow for both pins the buttons to the right edge and clips the fields, so Commit and Cancel
/// are always reachable.
pub fn options_bar(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let Some(t) = app.ui.transform.clone() else { return };
    let bar = ui.available_rect_before_wrap();
    let warping = t.warp.is_some();
    let buttons_w = if warping { WARP_ACTIONS_W } else { ACTIONS_W };
    let split = (bar.right() - buttons_w).max(bar.left());
    let fields = egui::Rect::from_min_max(bar.min, egui::pos2(split, bar.bottom()));
    let mut row = ui.new_child(egui::UiBuilder::new().id_salt("transform-fields").max_rect(fields).layout(egui::Layout::left_to_right(egui::Align::Center)));
    row.set_clip_rect(fields);
    if let Some(w) = t.warp.clone() {
        warp_fields(app, &mut row, &w);
    } else if t.content_aware.is_some() {
        crate::cas_transform::fields(app, &mut row, &t);
    } else {
        transform_fields(app, &mut row, &t);
    }
    crate::widgets::vline(&mut row, 22.0);
    let start = row.min_rect().right().min(split);
    let actions = egui::Rect::from_min_max(egui::pos2(start, bar.top()), egui::pos2(start + buttons_w, bar.bottom()));
    let mut btns = ui.new_child(egui::UiBuilder::new().id_salt("transform-actions").max_rect(actions).layout(egui::Layout::right_to_left(egui::Align::Center)));
    transform_actions(app, &mut btns, warping, t.content_aware.is_none());
}

fn transform_fields(app: &mut PhotocraftApp, ui: &mut egui::Ui, t: &TransformSession) {
    let tk = crate::theme::Tokens::get(ui.ctx());
    let (_, _, angle, skew_h) = readout(t);
    let lbl = |ui: &mut egui::Ui, s: &str| {
        ui.label(egui::RichText::new(s).color(tk.text_dim).size(12.0));
    };
    position_size_fields(app, ui, t);
    crate::widgets::vline(ui, 22.0);
    let mut a = angle as f32;
    let (r, resp) = ui.allocate_exact_size(vec2(18.0, 22.0), egui::Sense::hover());
    crate::icons::paint(ui, r, "rotate-cw", 13.0, tk.text_dim);
    resp.on_hover_text(tl!("Rotate"));
    if crate::widgets::value_field(ui, &mut a, -180.0..=180.0, "°", 60.0).changed() {
        rotate_about_pivot(app, (a as f64 - angle).to_radians());
    }
    // Photoshop's H and V skew fields: the box's sides lean by H, its top and bottom by V.
    let (mut h, mut v) = (skew_h as f32, t.skew_v as f32);
    let lim = -SKEW_LIMIT_DEG..=SKEW_LIMIT_DEG;
    lbl(ui, "H:");
    let rh = crate::widgets::value_field(ui, &mut h, lim.clone(), "°", 56.0).on_hover_text(tl!("Set horizontal skew"));
    lbl(ui, "V:");
    let rv = crate::widgets::value_field(ui, &mut v, lim, "°", 56.0).on_hover_text(tl!("Set vertical skew"));
    if rh.changed() || rv.changed() {
        skew_about_pivot(app, f64::from(h), f64::from(v));
    }
    crate::widgets::vline(ui, 22.0);
    lbl(ui, tl!("Interpolation:"));
    let mut interp = t.interpolation.clone();
    let opts = [("bicubic".to_string(), tl!("Bicubic")), ("bilinear".to_string(), tl!("Bilinear")), ("nearest".to_string(), tl!("Nearest Neighbor"))];
    if crate::widgets::dropdown(ui, "transform-interp", &mut interp, &opts, 130.0)
        && let Some(s) = app.ui.transform.as_mut()
    {
        s.interpolation = interp;
    }
}

/// The reference point's X/Y and the W/H scale fields (with the aspect link) of the options bar.
pub(crate) fn position_size_fields(app: &mut PhotocraftApp, ui: &mut egui::Ui, t: &TransformSession) {
    let tk = crate::theme::Tokens::get(ui.ctx());
    let (sx, sy, _, _) = readout(t);
    let lbl = |ui: &mut egui::Ui, s: &str| {
        ui.label(egui::RichText::new(s).color(tk.text_dim).size(12.0));
    };
    let mut px = t.pivot[0] as f32;
    let mut py = t.pivot[1] as f32;
    lbl(ui, "X:");
    let rx = crate::widgets::value_field(ui, &mut px, -POSITION_LIMIT_PX..=POSITION_LIMIT_PX, "px", 72.0);
    lbl(ui, "Y:");
    let ry = crate::widgets::value_field(ui, &mut py, -POSITION_LIMIT_PX..=POSITION_LIMIT_PX, "px", 72.0);
    if (rx.changed() || ry.changed())
        && let Some(s) = app.ui.transform.as_mut()
    {
        let (dx, dy) = (px as f64 - s.pivot[0], py as f64 - s.pivot[1]);
        s.quad = s.quad.map(|q| [q[0] + dx, q[1] + dy]);
        s.pivot = [px as f64, py as f64];
    }
    crate::widgets::vline(ui, 22.0);
    // W and H in the unit picked from their right-click menu: the scale in percent, or the box's
    // length (Photoshop's units, at the document's resolution).
    let u = size_unit(app);
    let (side_w, side_h) = (t.rect[2] - t.rect[0], t.rect[3] - t.rect[1]);
    let (wlim, hlim) = (scale_limit_pct(side_w), scale_limit_pct(side_h));
    let (wlim_u, hlim_u) = (size_shown(&u, f64::from(wlim), side_w), size_shown(&u, f64::from(hlim), side_h));
    let (mut w, mut h) = (size_shown(&u, sx, side_w), size_shown(&u, sy, side_h));
    let (suffix, places) = (u.unit.suffix(), u.unit.decimals() as u32);
    // Lengths need more room than a percentage ("2116.67 mm").
    let size_w = if u.unit == photocraft_engine::prefs::Unit::Percent { 66.0 } else { 84.0 };
    lbl(ui, "W:");
    let rw = crate::widgets::value_field_prec(ui, &mut w, -wlim_u..=wlim_u, suffix, size_w, places);
    let link = !app.ui.transform_unlinked;
    if crate::icons::button(ui, if link { "link" } else { "unlink" }, 22.0, link, tl!("Maintain aspect ratio")).clicked() {
        app.ui.transform_unlinked = link;
    }
    lbl(ui, "H:");
    let rh = crate::widgets::value_field_prec(ui, &mut h, -hlim_u..=hlim_u, suffix, size_w, places);
    for r in [&rw, &rh] {
        r.context_menu(|ui| {
            if let Some(name) = crate::rulers::unit_menu(ui, u.unit) {
                set_size_unit(app, name);
            }
        });
    }
    let (w, h) = (size_pct(&u, w, side_w) as f32, size_pct(&u, h, side_h) as f32);
    if rw.changed() || rh.changed() {
        let (kx, ky) = if link {
            let k = if rw.changed() { w as f64 / sx.max(1e-9) } else { h as f64 / sy.max(1e-9) };
            // Linked, the other side follows: keep it within its own limit too.
            let k = k.min(f64::from(wlim) / sx.abs().max(1e-9)).min(f64::from(hlim) / sy.abs().max(1e-9));
            (k, k)
        } else {
            (w as f64 / sx.max(1e-9), h as f64 / sy.max(1e-9))
        };
        scale_about_pivot(app, kx, ky);
    }
}

/// Options bar fields in Warp mode: style, bend, the split icons and the grid menu.
fn warp_fields(app: &mut PhotocraftApp, ui: &mut egui::Ui, w: &Warp) {
    let tk = crate::theme::Tokens::get(ui.ctx());
    let lbl = |ui: &mut egui::Ui, s: &str| {
        ui.label(egui::RichText::new(s).color(tk.text_dim).size(12.0));
    };
    // Type takes only the styles (#3004): Split and Grid are greyed, and there is no Custom.
    let on_type = warps_type(app);
    // Split, Grid, then the warp style with its bend fields.
    ui.add_enabled_ui(!on_type, |ui| warp_mesh_fields(app, ui, w));
    crate::widgets::vline(ui, 22.0);
    lbl(ui, tl!("Warp:"));
    let mut style = w.style;
    let opts: Vec<(WarpStyle, &str)> = WarpStyle::all().filter(|s| !(on_type && *s == WarpStyle::Custom)).map(|s| (s, s.label())).collect();
    if crate::widgets::dropdown(ui, "warp-style", &mut style, &opts, 120.0)
        && let Some(t) = app.ui.transform.as_mut()
    {
        let b = t.rect;
        t.warp = Some(match style {
            WarpStyle::Custom => Warp::custom(w.to_mesh(1, 1), b),
            WarpStyle::None if on_type => Warp::none(b),
            WarpStyle::None => Warp::custom(BezierMesh::identity(b, 1, 1), b),
            s => Warp { style: s, bend: if w.style.is_preset() { w.bend } else { 50.0 }, ..Warp::none(b) },
        });
    }
    // Orientation and the bend fields are always there, and only act on a preset style.
    let preset = w.style.is_preset();
    ui.add_enabled_ui(preset, |ui| {
        let orient = crate::icons::button(ui, "arrow-left-right", 26.0, w.vertical, tl!("Vertical"));
        orient.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, preset, "Change warp orientation"));
        if orient.clicked()
            && let Some(Some(cur)) = app.ui.transform.as_mut().map(|t| t.warp.as_mut())
        {
            cur.vertical = !cur.vertical;
        }
        let fields: [(&str, f64); 3] = [(tl!("Bend:"), w.bend), ("H:", w.h_distort), ("V:", w.v_distort)];
        for (k, (name, v)) in fields.iter().enumerate() {
            lbl(ui, name);
            let mut f = *v as f32;
            if crate::widgets::value_field(ui, &mut f, -100.0..=100.0, "%", 58.0).changed()
                && let Some(Some(cur)) = app.ui.transform.as_mut().map(|t| t.warp.as_mut())
            {
                match k {
                    0 => cur.bend = f64::from(f),
                    1 => cur.h_distort = f64::from(f),
                    _ => cur.v_distort = f64::from(f),
                }
            }
        }
    });
}

/// The Split icons and the Grid menu of the Warp options bar.
fn warp_mesh_fields(app: &mut PhotocraftApp, ui: &mut egui::Ui, w: &Warp) {
    let tk = crate::theme::Tokens::get(ui.ctx());
    let lbl = |ui: &mut egui::Ui, s: &str| {
        ui.label(egui::RichText::new(s).color(tk.text_dim).size(12.0));
    };
    lbl(ui, tl!("Split:"));
    let armed = app.transform_preview.as_ref().and_then(|p| p.split_tool);
    for (tool, tip) in [
        (SplitTool::Cross, tl!("Split Warp Crosswise")),
        (SplitTool::Vertical, tl!("Split Warp Vertically")),
        (SplitTool::Horizontal, tl!("Split Warp Horizontally")),
    ] {
        let on = armed == Some(tool);
        if split_icon(ui, tool.icon(), tip, on).clicked()
            && let Some(pv) = app.transform_preview.as_mut()
        {
            pv.split_tool = if on { None } else { Some(tool) };
            pv.split_placing = false;
            pv.split_pointer = None;
            pv.split_quick = false;
        }
    }
    crate::widgets::vline(ui, 22.0);
    lbl(ui, tl!("Grid:"));
    let mesh = w.mesh.clone().unwrap_or_else(|| BezierMesh::identity(w.bounds, 1, 1));
    let mut grid = warp_grid_id(&mesh).to_string();
    let opts = [
        ("custom".to_string(), tl!("Custom")),
        ("default".to_string(), tl!("Default")),
        ("3".to_string(), tl!("3 x 3")),
        ("4".to_string(), tl!("4 x 4")),
        ("5".to_string(), tl!("5 x 5")),
    ];
    if crate::widgets::dropdown(ui, "warp-grid", &mut grid, &opts, 92.0)
        && let Some(n) = match grid.as_str() {
            "default" => Some(1),
            "3" => Some(3),
            "4" => Some(4),
            "5" => Some(5),
            _ => None,
        }
    {
        let _ = edit_session_warp(app, "edit.transform.warpGrid", &json!({ "size": n }));
    }
}

/// Back to the untouched box: the default one-patch grid over the box, no bend.
pub fn reset_warp(app: &mut PhotocraftApp) {
    let Some(t) = app.ui.transform.as_mut() else { return };
    if t.warp.is_none() {
        return;
    }
    let b = t.rect;
    t.warp = Some(Warp::custom(BezierMesh::identity(b, 1, 1), b));
    if warps_type(app)
        && let Some(t) = app.ui.transform.as_mut()
    {
        t.warp = Some(Warp::none(b));
    }
    if let Some(pv) = app.transform_preview.as_mut() {
        pv.warp_drag = None;
        pv.split_tool = None;
        pv.split_pointer = None;
        pv.split_placing = false;
        pv.split_quick = false;
    }
}

/// Grid menu readout: Default is 1×1, the numbered entries are uniform n×n patches, and anything
/// split by hand reads as Custom.
fn warp_grid_id(mesh: &BezierMesh) -> &'static str {
    let cols = mesh.us.len().saturating_sub(1);
    let rows = mesh.vs.len().saturating_sub(1);
    let even = |k: &[f64]| {
        let n = k.len().saturating_sub(1);
        n > 0 && k.iter().enumerate().all(|(i, v)| (v - i as f64 / n as f64).abs() < 1e-4)
    };
    if cols == rows && even(&mesh.us) && even(&mesh.vs) {
        match cols {
            1 => "default",
            3 => "3",
            4 => "4",
            5 => "5",
            _ => "custom",
        }
    } else {
        "custom"
    }
}

/// One of the three Split icons. `on` is the armed tool: the line then follows the pointer.
fn split_icon(ui: &mut egui::Ui, kind: &str, tip: &str, on: bool) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(vec2(22.0, 22.0), egui::Sense::click());
    let ink = crate::icons::button_chrome(ui, rect, on, resp.hovered());
    let c = rect.center();
    let stroke = Stroke::new(1.15, ink);
    let box_r = egui::Rect::from_center_size(c, vec2(13.0, 13.0));
    ui.painter().rect_stroke(box_r, 2.0, stroke, egui::StrokeKind::Inside);
    let h = 5.2;
    match kind {
        "vertical" => {
            ui.painter().line_segment([c - vec2(0.0, h), c + vec2(0.0, h)], stroke);
        }
        "horizontal" => {
            ui.painter().line_segment([c - vec2(h, 0.0), c + vec2(h, 0.0)], stroke);
        }
        _ => {
            ui.painter().line_segment([c - vec2(h, 0.0), c + vec2(h, 0.0)], stroke);
            ui.painter().line_segment([c - vec2(0.0, h), c + vec2(0.0, h)], stroke);
        }
    }
    if kind == "cross" {
        ui.painter().circle_filled(c, 1.5, ink);
    } else {
        ui.painter().circle_stroke(c, 2.1, stroke);
    }
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, tip));
    resp.on_hover_text(tip)
}

/// Warp switch, cancel and commit, kept on the right of the options bar.
/// `mode_switch`: the Free Transform / Warp switch (not for Content-Aware Scale, which only scales).
fn transform_actions(app: &mut PhotocraftApp, ui: &mut egui::Ui, warping: bool, mode_switch: bool) {
    ui.add_space(8.0);
    let commit_tip = if warping {
        crate::i18n::fmt(tl!("Commit warp ({key})"), &[("key", &crate::shortcuts::pretty("Enter"))])
    } else {
        crate::i18n::fmt(tl!("Commit transform ({key})"), &[("key", &crate::shortcuts::pretty("Enter"))])
    };
    let commit_btn = crate::icons::button(ui, "check", 26.0, false, &commit_tip);
    let label = "Commit transform";
    commit_btn.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, tl!(label)));
    if commit_btn.clicked() {
        commit(app);
    }
    let cancel_tip = if warping { tl!("Cancel warp (Esc)") } else { tl!("Cancel transform (Esc)") };
    let cancel_btn = crate::icons::button(ui, "ban", 26.0, false, cancel_tip);
    let label = "Cancel transform";
    cancel_btn.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, tl!(label)));
    if cancel_btn.clicked() {
        cancel(app);
    }
    if warping {
        let reset = crate::icons::button(ui, "undo-2", 26.0, false, tl!("Reset"));
        reset.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, "Reset warp"));
        if reset.clicked() {
            reset_warp(app);
        }
    }
    if !mode_switch {
        return;
    }
    crate::widgets::vline(ui, 22.0);
    let mode = crate::icons::button(ui, "grid-3x3", 26.0, warping, tl!("Switch between free transform and warp modes"));
    mode.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, "Switch between free transform and warp modes"));
    if mode.clicked() {
        if warping {
            leave_warp(app);
        } else {
            enter_warp(app, &ui.ctx().clone());
        }
    }
}

/// Scale the box along its own axes about the reference point.
fn scale_about_pivot(app: &mut PhotocraftApp, kx: f64, ky: f64) {
    let Some(s) = app.ui.transform.as_mut() else { return };
    let Some(h) = Homography::rect_to_quad([0.0, 0.0, 1.0, 1.0], s.quad) else { return };
    let Some(inv) = h.inverse() else { return };
    let (pu, pv) = inv.apply(s.pivot[0], s.pivot[1]);
    let m = |u: f64, v: f64| {
        let (x, y) = h.apply(pu + (u - pu) * kx, pv + (v - pv) * ky);
        [x, y]
    };
    s.quad = [m(0.0, 0.0), m(1.0, 0.0), m(1.0, 1.0), m(0.0, 1.0)];
}

fn rotate_about_pivot(app: &mut PhotocraftApp, da: f64) {
    let Some(s) = app.ui.transform.as_mut() else { return };
    let (sn, cs) = da.sin_cos();
    let c = s.pivot;
    s.quad = s.quad.map(|q| {
        let (x, y) = (q[0] - c[0], q[1] - c[1]);
        [c[0] + x * cs - y * sn, c[1] + x * sn + y * cs]
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> TransformSession {
        TransformSession {
            session: 1,
            layer: 1,
            rect: [0.0, 0.0, 100.0, 50.0],
            quad: corners([0.0, 0.0, 100.0, 50.0]),
            pivot: [50.0, 25.0],
            interpolation: "bicubic".into(),
            warp: None,
            selection: false,
            target: None,
            path: None,
            made: None,
            mode: Default::default(),
            content_aware: None,
            skew_v: 0.0,
        }
    }

    #[test]
    fn a_plain_click_far_outside_accepts_transform_but_a_rotation_drag_does_not() {
        let context = egui::Context::default();
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
        begin(&mut app, &context).unwrap();
        let before = app.session.active().unwrap().history.past_len();
        let t = app.ui.transform.as_mut().unwrap();
        t.quad = t.quad.map(|[x, y]| [x + 6.0, y]);
        let outside = [44.0, 44.0];
        let none = egui::Modifiers::NONE;
        assert!(pointer(&mut app, ToolEvent::Down { x: outside[0], y: outside[1], pressure: 1.0 }, none));
        assert!(app.ui.transform.is_some(), "press alone does not accept");
        assert!(pointer(&mut app, ToolEvent::Up { x: outside[0], y: outside[1] }, none));
        assert!(app.ui.transform.is_none(), "a release without a drag commits");
        assert_eq!(app.session.active().unwrap().history.past_len(), before + 1, "one undo step for the transform");
        let st = app.session.active().unwrap();
        let layer = st.doc.layer(st.active_layer.unwrap()).unwrap();
        assert_eq!(layer.surface().unwrap().content_bounds(), photocraft_geom::Rect::new(14, 8, 30, 24));

        // An outside drag remains a rotation gesture; releasing it does not commit.
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
        begin(&mut app, &context).unwrap();
        let start = [44.0, 44.0];
        let end = [44.0, 54.0];
        let original = app.ui.transform.as_ref().unwrap().quad;
        pointer(&mut app, ToolEvent::Down { x: start[0], y: start[1], pressure: 1.0 }, none);
        pointer(&mut app, ToolEvent::Move { x: end[0], y: end[1], pressure: 1.0 }, none);
        pointer(&mut app, ToolEvent::Up { x: end[0], y: end[1] }, none);
        assert!(app.ui.transform.is_some(), "rotation stays active after an outside drag");
        assert_ne!(app.ui.transform.as_ref().unwrap().quad, original);
    }

    #[test]
    fn a_click_near_rotation_handles_accepts_but_drags_and_option_pivot_do_not() {
        let t = session();
        let tol = 12.0;
        let near = Gesture { hit: Hit::Outside, start: [-18.0, -18.0], quad0: t.quad, pivot0: t.pivot };
        assert!(accepts_outside_click(near, near.start, tol), "a click in the rotation zone accepts; a drag still rotates");
        let far = Gesture { start: [500.0, 500.0], ..near };
        assert!(accepts_outside_click(far, far.start, tol));
        assert!(!accepts_outside_click(far, [512.0, 500.0], tol), "drag does not commit");
        let handle = Gesture { hit: Hit::Corner(0), ..near };
        assert!(!accepts_outside_click(handle, handle.start, tol), "resize handles stay interactive");

        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
        begin(&mut app, &egui::Context::default()).unwrap();
        let p = [44.0, 44.0];
        pointer(&mut app, ToolEvent::Down { x: p[0], y: p[1], pressure: 1.0 }, egui::Modifiers::ALT);
        pointer(&mut app, ToolEvent::Up { x: p[0], y: p[1] }, egui::Modifiers::ALT);
        assert!(app.ui.transform.is_some(), "Option-click moves the pivot, never accepts");
        assert_eq!(app.ui.transform.as_ref().unwrap().pivot, p);
    }

    #[test]
    fn rotate_feedback_only_appears_just_outside_corners() {
        let mut t = session();
        assert!(near_rotate_corner(&t, [-18.0, -18.0], 12.0));
        assert!(!near_rotate_corner(&t, [50.0, 25.0], 12.0), "inside moves the box");
        assert!(!near_rotate_corner(&t, [500.0, 500.0], 12.0), "far outside stays generic");
        assert!(!near_rotate_corner(&t, [2.0, 2.0], 12.0), "corner handle resizes");
        t.mode = TransformMode::Distort;
        assert!(!near_rotate_corner(&t, [-18.0, -18.0], 12.0));
    }

    #[test]
    fn angle_readout_tracks_rotated_quad_and_handles_degenerate_edges() {
        let mut t = session();
        assert_eq!(rotation_degrees(t.quad), 0.0);
        let pivot = t.pivot;
        t.quad = t.quad.map(|q| [pivot[0] - (q[1] - pivot[1]), pivot[1] + (q[0] - pivot[0])]);
        assert!((rotation_degrees(t.quad) - 90.0).abs() < 1e-9);
        t.quad[1] = t.quad[0];
        assert_eq!(rotation_degrees(t.quad), 0.0);
    }

    #[test]
    fn transform_bar_pins_warp_cancel_and_commit() {
        use egui_kittest::Harness;
        use egui_kittest::kittest::Queryable;
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.transform = Some(session());
        let width = 720.0;
        let mut h = Harness::builder().with_size(vec2(width, 48.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    return;
                }
                ui.horizontal_centered(|ui| options_bar(app, ui));
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Studio);
        h.run_steps(6);
        let commit = h.get_by_label("Commit transform").rect();
        let cancel = h.get_by_label("Cancel transform").rect();
        let mode = h.get_by_label("Switch between free transform and warp modes").rect();
        assert!(commit.right() > width - 36.0, "commit sits on the right: {commit:?}");
        assert!(cancel.right() < commit.left() + 1.0, "cancel is left of commit");
        assert!(mode.right() < cancel.left() + 1.0, "warp switch is left of cancel");
        h.get_by_label("Switch between free transform and warp modes").click();
        h.run_steps(2);
        assert!(h.state().ui.transform.as_ref().unwrap().warp.is_some());
        h.get_by_label("Switch between free transform and warp modes").click();
        h.run_steps(2);
        assert!(h.state().ui.transform.as_ref().unwrap().warp.is_none());
        h.get_by_label("Cancel transform").click();
        h.run_steps(2);
        assert!(h.state().ui.transform.is_none());
    }

    #[test]
    fn free_transform_bar_buttons_follow_the_fields_on_a_wide_bar() {
        use egui_kittest::Harness;
        use egui_kittest::kittest::Queryable;
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.transform = Some(session());
        let width = 2400.0;
        let mut h = Harness::builder().with_size(vec2(width, 48.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    return;
                }
                ui.horizontal_centered(|ui| options_bar(app, ui));
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Studio);
        h.run_steps(6);
        let commit = h.get_by_label("Commit transform").rect();
        assert!(commit.right() < width * 0.7, "commit sits next to the fields, not at the far right: {commit:?}");
    }

    #[test]
    fn warp_bar_buttons_follow_the_fields_on_a_wide_bar() {
        use egui_kittest::Harness;
        use egui_kittest::kittest::Queryable;
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.transform = Some(session());
        let width = 2000.0;
        let mut h = Harness::builder().with_size(vec2(width, 48.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    return;
                }
                ui.horizontal_centered(|ui| options_bar(app, ui));
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Studio);
        h.run_steps(6);
        h.get_by_label("Switch between free transform and warp modes").click();
        h.run_steps(4);
        assert!(h.state().ui.transform.as_ref().unwrap().warp.is_some());
        let commit = h.get_by_label("Commit transform").rect();
        assert!(commit.right() < width / 2.0, "commit sits next to the fields, not at the far right: {commit:?}");
        let reset = h.get_by_label("Reset warp").rect();
        assert!(reset.right() < commit.left() + 1.0, "reset is left of commit");
        h.get_by_label("Reset warp").click();
        h.run_steps(2);
        assert!(h.state().ui.transform.as_ref().unwrap().warp.as_ref().unwrap().is_identity());
    }

    fn eq6(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    /// Scale, skew and rotation read back exactly from the box they make, flips included.
    #[test]
    fn parts_round_trip() {
        for (sx, sy, angle, h, v) in [(1.0, 1.0, 0.0, 0.0, 0.0), (1.5, 0.5, 30.0, 20.0, 0.0), (0.8, -1.2, -120.0, -35.0, 15.0), (2.0, 1.0, 179.0, 0.0, -60.0)] {
            let p = Parts { sx, sy, angle: f64::to_radians(angle), skew_h: f64::to_radians(h), skew_v: f64::to_radians(v) };
            let back = Parts::from_columns(p.columns_of(), p.skew_v);
            for (a, b) in [(back.sx, p.sx), (back.sy, p.sy), (back.angle, p.angle), (back.skew_h, p.skew_h)] {
                assert!(eq6(a, b), "{p:?} read back as {back:?}");
            }
        }
    }

    /// A box sheared sideways keeps W, H and the angle; only H skew moves (it used to read as a
    /// taller box).
    #[test]
    fn a_sheared_box_reads_as_horizontal_skew() {
        let mut t = session();
        // The 100×50 frame with its bottom slid 50·tan 20° to the left: a positive H skew.
        let d = 50.0 * 20f64.to_radians().tan();
        t.quad = [[0.0, 0.0], [100.0, 0.0], [100.0 - d, 50.0], [-d, 50.0]];
        let (w, h, angle, skew) = readout(&t);
        assert!(eq6(w, 100.0) && eq6(h, 100.0) && eq6(angle, 0.0), "{w} {h} {angle}");
        assert!(eq6(skew, 20.0), "{skew}");
    }

    /// Typing H and V skews leans the box about the reference point; the angle, W and H fields keep
    /// their values, and V stays as typed.
    #[test]
    fn skew_fields_lean_the_box_about_the_reference_point() {
        let t = type_into_bar_field(5, "20");
        let (w, h, angle, skew) = readout(&t);
        assert!(eq6(skew, 20.0) && eq6(angle, 0.0) && eq6(w, 100.0) && eq6(h, 100.0), "{w} {h} {angle} {skew}");
        // The pivot (50, 25) stays put: the top edge slid right, the bottom left, by the same amount.
        let d = 25.0 * 20f64.to_radians().tan();
        assert!(eq6(t.quad[0][0], d) && eq6(t.quad[0][1], 0.0) && eq6(t.quad[3][0], -d) && eq6(t.quad[3][1], 50.0), "{:?}", t.quad);

        let t = type_into_bar_field(6, "-10");
        assert!(eq6(t.skew_v, -10.0));
        let (w, h, angle, skew) = readout(&t);
        assert!(eq6(skew, 0.0) && eq6(angle, 0.0) && eq6(w, 100.0) && eq6(h, 100.0), "{w} {h} {angle} {skew}");
        // The right side rose: tan(−10°) per pixel across, about the pivot.
        let d = 50.0 * (-10f64).to_radians().tan();
        assert!(eq6(t.quad[1][1], d) && eq6(t.quad[0][1], -d), "{:?}", t.quad);
        // A skew past ±90° would flatten the box: the field stops at the limit.
        let t = type_into_bar_field(5, "120");
        assert!(eq6(readout(&t).3, f64::from(SKEW_LIMIT_DEG)), "{}", readout(&t).3);
    }

    /// Types `text` + Enter into the `n`th numeric field of the transform bar (X, Y, W, H, angle…).
    fn type_into_bar_field(n: usize, text: &str) -> TransformSession {
        type_into_bar_field_with(|_| {}, n, text)
    }

    /// The transform bar over the test box, after `setup` (e.g. picking the W/H unit). Tall enough
    /// for the fields' unit menu to open below the bar unscrolled.
    fn bar_harness(setup: impl FnOnce(&mut PhotocraftApp)) -> egui_kittest::Harness<'static, PhotocraftApp> {
        use egui_kittest::Harness;
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.transform = Some(session());
        setup(&mut app);
        let mut h = Harness::builder().with_size(vec2(900.0, 360.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    return;
                }
                ui.horizontal_centered(|ui| options_bar(app, ui));
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Studio);
        h.run_steps(6);
        h
    }

    /// [`type_into_bar_field`] after `setup`.
    fn type_into_bar_field_with(setup: impl FnOnce(&mut PhotocraftApp), n: usize, text: &str) -> TransformSession {
        use egui_kittest::kittest::Queryable;
        let mut h = bar_harness(setup);
        h.get_all_by_role(egui::accesskit::Role::SpinButton).nth(n).unwrap().click();
        h.run();
        for c in text.chars() {
            h.event(egui::Event::Text(c.to_string()));
            h.run();
        }
        h.key_press(egui::Key::Enter);
        h.run_steps(2);
        h.state().ui.transform.clone().unwrap()
    }

    /// The values the W and H fields show (the 3rd and 4th numeric fields).
    fn shown_size(h: &egui_kittest::Harness<'static, PhotocraftApp>) -> Vec<String> {
        use egui_kittest::kittest::{NodeT, Queryable};
        h.get_all_by_role(egui::accesskit::Role::SpinButton).skip(2).take(2).map(|n| n.accesskit_node().value().unwrap_or_default().to_string()).collect()
    }

    #[test]
    fn size_fields_convert_between_percent_and_lengths() {
        use crate::layer_props_ui::LenUnit;
        use photocraft_engine::prefs::Unit;
        // A 300 px side at 300 ppi is one inch at 100 %.
        let at = |unit| LenUnit { unit, dpi: 300.0, ppi: 72.0 };
        for (unit, at_100) in [
            (Unit::Percent, 100.0),
            (Unit::Pixels, 300.0),
            (Unit::Inches, 1.0),
            (Unit::Centimeters, 2.54),
            (Unit::Millimeters, 25.4),
            (Unit::Points, 72.0),
            (Unit::Picas, 6.0),
        ] {
            let u = at(unit);
            let shown = size_shown(&u, 100.0, 300.0);
            assert!((f64::from(shown) - at_100).abs() < 1e-3, "{unit:?}: {shown}");
            assert!((size_shown(&u, 150.0, 300.0) - shown * 1.5).abs() < 1e-3, "{unit:?} scales with the box");
            assert!((size_pct(&u, shown * 2.0, 300.0) - 200.0).abs() < 1e-3, "{unit:?}: twice the length is 200 %");
            assert!((size_pct(&u, -shown, 300.0) + 100.0).abs() < 1e-3, "{unit:?}: a negative length flips");
        }
        // A degenerate side never divides by zero.
        assert!(size_pct(&at(Unit::Inches), 1.0, 0.0).is_finite());
    }

    #[test]
    fn w_and_h_take_lengths_in_the_unit_picked_for_them() {
        use photocraft_engine::prefs::Unit;
        // The test box is 100 × 50 px; with no document open, lengths convert at 72 ppi.
        let cm = |app: &mut PhotocraftApp| set_size_unit(app, Unit::Centimeters.name());
        let h = bar_harness(cm);
        assert_eq!(shown_size(&h), ["3.53", "1.76"], "the box's size in cm, not 100 %");
        // 7.06 cm is twice the width; linked, the height follows.
        let t = type_into_bar_field_with(cm, 2, "7.06");
        let (sx, sy, _, _) = readout(&t);
        assert!((sx - 200.0).abs() < 0.2 && (sy - 200.0).abs() < 0.2, "{sx} {sy}");
        // Pixels: an exact width.
        let t = type_into_bar_field_with(|app| set_size_unit(app, Unit::Pixels.name()), 2, "250");
        assert!((t.quad[1][0] - t.quad[0][0] - 250.0).abs() < 0.01, "{:?}", t.quad);
        // Percent is still the default.
        assert_eq!(shown_size(&bar_harness(|_| {})), ["100", "100"]);
    }

    #[test]
    fn right_clicking_w_or_h_picks_their_unit_and_remembers_it() {
        use egui_kittest::kittest::Queryable;
        let mut h = bar_harness(|_| {});
        h.get_all_by_role(egui::accesskit::Role::SpinButton).nth(2).unwrap().click_secondary();
        h.run_steps(2);
        h.get_by_label("Inches").click();
        h.run_steps(3);
        assert_eq!(size_unit(h.state()).unit, photocraft_engine::prefs::Unit::Inches);
        assert_eq!(h.state().session.prefs().dialogs[SIZE_UNIT_KEY]["sizeUnit"], "inches");
        assert_eq!(shown_size(&h), ["1.389", "0.694"], "100 × 50 px at 72 ppi");
        // H opens the same menu, with the current unit checked.
        h.get_all_by_role(egui::accesskit::Role::SpinButton).nth(3).unwrap().click_secondary();
        h.run_steps(2);
        assert!(h.query_by_label("✓ Inches").is_some(), "the current unit is checked");
        h.get_by_label("Percent").click();
        h.run_steps(3);
        assert_eq!(shown_size(&h), ["100", "100"]);
        // A hand-edited or stale preference falls back to Percent instead of failing.
        h.state_mut().session.prefs.edit(|p| {
            p.dialogs.insert(SIZE_UNIT_KEY.into(), json!({"sizeUnit": "furlongs"}));
        });
        assert_eq!(size_unit(h.state()).unit, photocraft_engine::prefs::Unit::Percent);
        h.state_mut().session.prefs.edit(|p| {
            p.dialogs.insert(SIZE_UNIT_KEY.into(), json!("not an object"));
        });
        assert_eq!(size_unit(h.state()).unit, photocraft_engine::prefs::Unit::Percent);
        set_size_unit(h.state_mut(), "points");
        assert_eq!(size_unit(h.state()).unit, photocraft_engine::prefs::Unit::Points, "a bad entry is replaced, not kept");
    }

    #[test]
    fn editing_the_contents_of_a_dropped_file_keeps_the_layer() {
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));

        app.session.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
        app.sync_views();

        app.session.execute("layer.duplicate", json!({"inPlace": true})).unwrap();
        app.sync_views();

        let parent_id = app.session.active().unwrap().doc.id;
        let placed_layer = app.session.active().unwrap().active_layer.unwrap();

        begin_placed(&mut app, &egui::Context::default()).unwrap();

        app.session.execute("layer.smartObjects.editContents", json!({})).unwrap();

        assert_ne!(app.session.active().unwrap().doc.id, parent_id, "Edit Contents should activate the child document");

        end_if_left(&mut app);

        let parent = app.session.documents().iter().find(|st| st.doc.id == parent_id);

        assert!(parent.is_some_and(|st| st.doc.layer(placed_layer).is_some()), "the placed Smart Object must remain in the parent document");
        assert!(app.ui.transform.is_none(), "the transform session should end");
        assert!(app.transform_preview.is_none(), "the transform preview should end");
    }

    /// Large documents need scales past 10000% and positions past 30000 px (#1267).
    #[test]
    fn transform_fields_reach_large_documents() {
        let t = type_into_bar_field(2, "15000");
        let (sx, sy, _, _) = readout(&t);
        assert!((sx - 15000.0).abs() < 0.5 && (sy - 15000.0).abs() < 0.5, "{sx} {sy}");
        assert!((t.quad[1][0] - t.quad[0][0] - 15_000.0).abs() < 1.0, "{:?}", t.quad);
        let t = type_into_bar_field(0, "250000");
        assert!((t.pivot[0] - 250_000.0).abs() < 1.0, "{:?}", t.pivot);
    }

    /// A typed scale stops where the box reaches the largest document side, not past what can be
    /// allocated: the 100 × 50 px test box tops out at 300000 % wide.
    #[test]
    fn typed_scales_stay_within_the_largest_document() {
        assert_eq!(scale_limit_pct(20.0), SCALE_LIMIT_PCT);
        assert_eq!(scale_limit_pct(3000.0), 10_000.0);
        assert_eq!(scale_limit_pct(1e9), 100.0);
        assert_eq!(scale_limit_pct(0.0), SCALE_LIMIT_PCT);
        assert_eq!(scale_limit_pct(f64::NAN), SCALE_LIMIT_PCT);
        let t = type_into_bar_field(2, "99999999");
        let (sx, sy, _, _) = readout(&t);
        let side = t.rect[2] - t.rect[0];
        assert!(sx <= f64::from(scale_limit_pct(side)) + 0.5 && sy <= f64::from(scale_limit_pct(t.rect[3] - t.rect[1])) + 0.5, "{sx} {sy}");
    }

    fn drag(s: &mut TransformSession, from: [f64; 2], to: [f64; 2], mods: egui::Modifiers) {
        let g = Gesture { hit: hit(s, from, 8.0), start: from, quad0: s.quad, pivot0: s.pivot };
        apply_drag(s, g, to, mods);
    }

    /// A handle drag in Edit › Transform `mode`, as the canvas applies it (on a pixel layer).
    fn drag_in(mode: TransformMode, from: [f64; 2], to: [f64; 2], mods: egui::Modifiers) -> [[f64; 2]; 4] {
        drag_on(mode, false, from, to, mods)
    }

    fn drag_on(mode: TransformMode, type_layer: bool, from: [f64; 2], to: [f64; 2], mods: egui::Modifiers) -> [[f64; 2]; 4] {
        let mut s = session();
        let g = Gesture { hit: hit(&s, from, 8.0), start: from, quad0: s.quad, pivot0: s.pivot };
        apply_mode_drag(&mut s, g, to, mode, type_layer, mods);
        s.quad
    }

    /// Skew on editable type, checked in Photoshop: a corner can't move alone (type stays a
    /// parallelogram), so it slides its whole edge: the top or bottom edge for a mostly
    /// horizontal drag, the left or right one for a mostly vertical drag. ⌥ moves the opposite
    /// edge the other way. Side handles are as on any layer.
    #[test]
    fn skew_mode_on_type_slides_the_corners_edge() {
        let (skew, none) = (TransformMode::Skew, egui::Modifiers::NONE);
        assert_eq!(drag_on(skew, true, [100.0, 0.0], [110.0, 3.0], none), [[10.0, 0.0], [110.0, 0.0], [100.0, 50.0], [0.0, 50.0]]);
        assert_eq!(drag_on(skew, true, [100.0, 0.0], [102.0, -9.0], none), [[0.0, 0.0], [100.0, -9.0], [100.0, 41.0], [0.0, 50.0]]);
        assert_eq!(drag_on(skew, true, [100.0, 0.0], [110.0, 3.0], egui::Modifiers::ALT), [[10.0, 0.0], [110.0, 0.0], [90.0, 50.0], [-10.0, 50.0]]);
        assert_eq!(drag_on(skew, true, [50.0, 0.0], [62.0, 7.0], none), [[12.0, 0.0], [112.0, 0.0], [100.0, 50.0], [0.0, 50.0]]);
    }

    /// Edit › Transform › Skew, checked in Photoshop (#3437): a corner moves alone, horizontally or
    /// vertically (whichever way the drag mostly goes); the top and bottom handles slide only
    /// horizontally, the left and right ones only vertically; ⌥ moves the opposite corner or edge
    /// the other way. Nothing scales.
    #[test]
    fn skew_mode_moves_a_corner_or_an_edge_along_one_axis() {
        let skew = TransformMode::Skew;
        let none = egui::Modifiers::NONE;
        // Corner (top right), mostly horizontal, then mostly vertical: only that corner moves.
        assert_eq!(drag_in(skew, [100.0, 0.0], [110.0, 3.0], none), [[0.0, 0.0], [110.0, 0.0], [100.0, 50.0], [0.0, 50.0]]);
        assert_eq!(drag_in(skew, [100.0, 0.0], [102.0, -9.0], none), [[0.0, 0.0], [100.0, -9.0], [100.0, 50.0], [0.0, 50.0]]);
        // Top edge: horizontal only, both its corners.
        assert_eq!(drag_in(skew, [50.0, 0.0], [62.0, 7.0], none), [[12.0, 0.0], [112.0, 0.0], [100.0, 50.0], [0.0, 50.0]]);
        // Left edge: vertical only.
        assert_eq!(drag_in(skew, [0.0, 25.0], [6.0, 33.0], none), [[0.0, 8.0], [100.0, 0.0], [100.0, 50.0], [0.0, 58.0]]);
        // ⌥: the opposite corner, and the opposite edge, go the other way.
        let alt = egui::Modifiers::ALT;
        assert_eq!(drag_in(skew, [100.0, 0.0], [110.0, 3.0], alt), [[0.0, 0.0], [110.0, 0.0], [100.0, 50.0], [-10.0, 50.0]]);
        assert_eq!(drag_in(skew, [50.0, 0.0], [62.0, 7.0], alt), [[12.0, 0.0], [112.0, 0.0], [88.0, 50.0], [-12.0, 50.0]]);
        // ⇧ changes nothing in Skew mode.
        let shift = egui::Modifiers::SHIFT;
        assert_eq!(drag_in(skew, [100.0, 0.0], [110.0, 3.0], shift), drag_in(skew, [100.0, 0.0], [110.0, 3.0], none));
        assert_eq!(drag_in(skew, [50.0, 0.0], [62.0, 7.0], shift), drag_in(skew, [50.0, 0.0], [62.0, 7.0], none));
        // A drag outside the box doesn't rotate it in Skew mode.
        assert_eq!(drag_in(skew, [130.0, 25.0], [120.0, 60.0], none), corners([0.0, 0.0, 100.0, 50.0]));
        // Free Transform is unchanged: a corner scales.
        assert_ne!(drag_in(TransformMode::Free, [100.0, 0.0], [110.0, 3.0], none)[2], [100.0, 50.0]);
    }

    fn close(a: [[f64; 2]; 4], b: [[f64; 2]; 4]) -> bool {
        a.iter().flatten().zip(b.iter().flatten()).all(|(x, y)| (x - y).abs() < 1e-6)
    }

    /// On a box smaller than the handles' grab areas the nearest handle wins, so the reference
    /// point at the centre doesn't take presses on the corners and edges.
    #[test]
    fn the_nearest_handle_wins_on_a_small_box() {
        let mut s = session();
        s.quad = corners([0.0, 0.0, 16.0, 16.0]);
        s.pivot = [8.0, 8.0];
        assert_eq!(hit(&s, [0.0, 0.0], 12.0), Hit::Corner(0));
        assert_eq!(hit(&s, [16.0, 17.0], 12.0), Hit::Corner(2));
        assert_eq!(hit(&s, [8.0, -1.0], 12.0), Hit::Edge(0));
        assert_eq!(hit(&s, [8.0, 8.0], 12.0), Hit::Pivot);
        assert_eq!(hit(&s, [40.0, 40.0], 12.0), Hit::Outside);
    }

    #[test]
    fn interior_patch_points_are_hidden_and_grid_anchors_stay() {
        assert!(on_section_edge(0, 0) && on_section_edge(1, 0) && on_section_edge(0, 2));
        assert!(!on_section_edge(1, 1) && !on_section_edge(2, 2));
        let b = [0.0, 0.0, 30.0, 30.0];
        let w = Warp::custom(BezierMesh::identity(b, 1, 1), b);
        assert!(warp_hit(&w, [10.0, 10.0], 4.0).is_none(), "the point inside the patch is not shown");
        assert_eq!(warp_hit(&w, [0.0, 0.0], 4.0), Some(0));
        let grid = Warp::custom(BezierMesh::identity(b, 3, 3), b);
        assert_eq!(warp_hit(&grid, [10.0, 10.0], 4.0), Some(33), "a grid-corner anchor stays draggable");
        assert!(warp_hit(&grid, [30.0 / 9.0, 30.0 / 9.0], 2.0).is_none());
    }

    #[test]
    fn grid_menu_reads_default_and_even_presets() {
        let b = [0.0, 0.0, 100.0, 80.0];
        let plain = BezierMesh::identity(b, 1, 1);
        assert_eq!(warp_grid_id(&plain), "default");
        assert_eq!(warp_grid_id(&BezierMesh::identity(b, 3, 3)), "3");
        assert_eq!(warp_grid_id(&BezierMesh::identity(b, 4, 4)), "4");
        assert_eq!(warp_grid_id(&BezierMesh::identity(b, 5, 5)), "5");
        assert_eq!(warp_grid_id(&BezierMesh::identity(b, 2, 2)), "custom");
        let mut split = plain.clone();
        assert!(split.split_u(0.5));
        assert_eq!(warp_grid_id(&split), "custom");
        assert!(single_patch(&plain));
        assert!(!single_patch(&BezierMesh::identity(b, 3, 3)));
    }

    #[test]
    fn split_tool_places_the_line_where_the_pointer_releases() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
        app.sync_views();
        app.session.execute("layer.new.layer", json!({})).unwrap();
        app.session
            .edit("paint", |doc, a| {
                doc.layer_mut(a.unwrap()).unwrap().surface_mut().unwrap().fill_rect(photocraft_geom::Rect::new(8, 8, 32, 32), &[1.0, 0.0, 0.0, 1.0]);
                Ok(())
            })
            .unwrap();
        begin_warp(&mut app, &egui::Context::default()).unwrap();
        app.transform_preview.as_mut().unwrap().split_tool = Some(SplitTool::Vertical);
        let knots = |app: &PhotocraftApp| app.ui.transform.as_ref().unwrap().warp.as_ref().unwrap().mesh.as_ref().unwrap().us.clone();
        assert_eq!(knots(&app).len(), 2);
        let mods = egui::Modifiers::NONE;
        warp_pointer(&mut app, ToolEvent::Down { x: 20.0, y: 20.0, pressure: 1.0 }, 2.0, mods);
        warp_pointer(&mut app, ToolEvent::Move { x: 14.0, y: 20.0, pressure: 1.0 }, 2.0, mods);
        assert_eq!(knots(&app).len(), 2, "the line is only a guide until release");
        warp_pointer(&mut app, ToolEvent::Up { x: 14.0, y: 20.0 }, 2.0, mods);
        let us = knots(&app);
        let has = |ks: &[f64], u: f64| ks.iter().any(|k| (k - u).abs() < 0.02);
        // The default grid's thirds stay; the release cuts the section that contains it.
        assert!(has(&us, 0.25) && has(&us, 1.0 / 3.0) && has(&us, 2.0 / 3.0), "{us:?}");
        let vs = app.ui.transform.as_ref().unwrap().warp.as_ref().unwrap().mesh.as_ref().unwrap().vs.clone();
        assert!(has(&vs, 1.0 / 3.0) && has(&vs, 2.0 / 3.0), "horizontal sections stay: {vs:?}");
        assert!(app.transform_preview.as_ref().unwrap().split_tool.is_some(), "the tool stays armed");
        // Option-click in the open places both lines, without a button armed.
        app.transform_preview.as_mut().unwrap().split_tool = None;
        let alt = egui::Modifiers::ALT;
        warp_pointer(&mut app, ToolEvent::Down { x: 20.0, y: 20.0, pressure: 1.0 }, 2.0, alt);
        warp_pointer(&mut app, ToolEvent::Up { x: 20.0, y: 20.0 }, 2.0, alt);
        let m = app.ui.transform.as_ref().unwrap().warp.as_ref().unwrap().mesh.as_ref().unwrap();
        assert!(m.us.len() == us.len() + 1 && has(&m.vs, 0.5), "crosswise cuts the open section: us {:?} vs {:?}", m.us, m.vs);
    }

    #[test]
    fn split_of_a_grid_cuts_one_section_and_keeps_the_rest() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
        app.sync_views();
        app.session.execute("layer.new.layer", json!({})).unwrap();
        app.session
            .edit("paint", |doc, a| {
                doc.layer_mut(a.unwrap()).unwrap().surface_mut().unwrap().fill_rect(photocraft_geom::Rect::new(8, 8, 32, 32), &[1.0, 0.0, 0.0, 1.0]);
                Ok(())
            })
            .unwrap();
        begin_warp(&mut app, &egui::Context::default()).unwrap();
        let rect = app.ui.transform.as_ref().unwrap().rect;
        let before = BezierMesh::identity(rect, 3, 3);
        app.ui.transform.as_mut().unwrap().warp = Some(Warp::custom(before.clone(), rect));
        // Middle of the centre column (s = 0.5), inside the top row of cells.
        let x = rect[0] + 0.5 * (rect[2] - rect[0]);
        let y = rect[1] + (1.0 / 6.0) * (rect[3] - rect[1]);
        app.transform_preview.as_mut().unwrap().split_tool = Some(SplitTool::Vertical);
        let mods = egui::Modifiers::NONE;
        warp_pointer(&mut app, ToolEvent::Down { x, y, pressure: 1.0 }, 2.0, mods);
        warp_pointer(&mut app, ToolEvent::Up { x, y }, 2.0, mods);
        let m = app.ui.transform.as_ref().unwrap().warp.as_ref().unwrap().mesh.as_ref().unwrap();
        assert_eq!(m.vs.len(), before.vs.len(), "horizontal sections are not rebuilt: {:?}", m.vs);
        for (a, b) in m.vs.iter().zip(&before.vs) {
            assert!((a - b).abs() < 1e-6, "{:?} vs {:?}", m.vs, before.vs);
        }
        assert_eq!(m.us.len(), before.us.len() + 1, "one cut: {:?}", m.us);
        for u in [0.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0] {
            assert!(m.us.iter().any(|k| (k - u).abs() < 1e-3), "missing {u} in {:?}", m.us);
        }
        let (nx0, nx1) = (before.nx(), m.nx());
        for j in 0..before.ny() {
            for i in 0..3 {
                let a = before.points[j * nx0 + i];
                let b = m.points[j * nx1 + i];
                assert!((a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6, "left section moved");
            }
        }
    }

    #[test]
    fn corner_drag_scales_proportionally_about_opposite_corner() {
        let mut s = session();
        drag(&mut s, [100.0, 50.0], [200.0, 60.0], egui::Modifiers::NONE);
        assert!(close(s.quad, corners([0.0, 0.0, 200.0, 100.0])), "{:?}", s.quad);
        let mut s = session();
        drag(&mut s, [100.0, 50.0], [200.0, 60.0], egui::Modifiers::SHIFT);
        assert!(close(s.quad, corners([0.0, 0.0, 200.0, 60.0])), "shift = free: {:?}", s.quad);
    }

    /// Photoshop CC 2019 and later: an edge scales proportionally too, about the middle of the
    /// opposite edge, and ⇧ stretches its one axis instead.
    #[test]
    fn edge_drag_scales_proportionally_about_the_opposite_edge() {
        // Right edge: 100 × 50 → 150 × 75; the left edge's middle (0, 25) stays, and so does the
        // reference point's place in the box (its centre).
        let mut s = session();
        drag(&mut s, [100.0, 25.0], [150.0, 25.0], egui::Modifiers::NONE);
        assert!(close(s.quad, corners([0.0, -12.5, 150.0, 62.5])), "{:?}", s.quad);
        assert!(near(s.pivot, [75.0, 25.0]), "{:?}", s.pivot);
        // Top edge to twice the height: twice the width, about x = 50.
        let mut s = session();
        drag(&mut s, [50.0, 0.0], [50.0, -50.0], egui::Modifiers::NONE);
        assert!(close(s.quad, corners([-50.0, -50.0, 150.0, 50.0])), "{:?}", s.quad);
        // Past the opposite edge: the dragged axis flips, the other only shrinks.
        let mut s = session();
        drag(&mut s, [100.0, 25.0], [-50.0, 25.0], egui::Modifiers::NONE);
        assert!(close(s.quad, corners([0.0, 12.5, -50.0, 37.5])), "{:?}", s.quad);
        // ⇧: one axis.
        let mut s = session();
        drag(&mut s, [50.0, 50.0], [50.0, 80.0], egui::Modifiers::SHIFT);
        assert!(close(s.quad, corners([0.0, 0.0, 100.0, 80.0])), "{:?}", s.quad);
    }

    #[test]
    fn alt_scales_edges_about_reference_point() {
        let mut s = session();
        drag(&mut s, [100.0, 25.0], [150.0, 25.0], egui::Modifiers::ALT);
        assert!(close(s.quad, corners([-50.0, -25.0, 150.0, 75.0])), "{:?}", s.quad);
        // ⌥⇧: one axis, about the reference point.
        let mut s = session();
        drag(&mut s, [100.0, 25.0], [150.0, 25.0], egui::Modifiers { alt: true, shift: true, ..Default::default() });
        assert!(close(s.quad, corners([-50.0, 0.0, 150.0, 50.0])), "{:?}", s.quad);
    }

    fn near(a: [f64; 2], b: [f64; 2]) -> bool {
        (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6
    }

    /// The reference point is attached to the box (#1329): scaling, skewing, distorting and
    /// perspective carry it along to the same place within the box; ⌥-scaling about it and
    /// rotating about it leave it put.
    #[test]
    fn the_reference_point_moves_with_the_box() {
        // Corner scale down: the centre stays the centre.
        let mut s = session();
        drag(&mut s, [100.0, 50.0], [50.0, 25.0], egui::Modifiers::NONE);
        assert!(close(s.quad, corners([0.0, 0.0, 50.0, 25.0])), "{:?}", s.quad);
        assert!(near(s.pivot, [25.0, 12.5]), "{:?}", s.pivot);
        // A reference point set to a corner stays at that corner; an edge drag carries it too.
        let mut s = session();
        s.pivot = [100.0, 0.0];
        drag(&mut s, [50.0, 50.0], [50.0, 80.0], egui::Modifiers::SHIFT);
        assert!(near(s.pivot, [100.0, 0.0]), "{:?}", s.pivot);
        drag(&mut s, [100.0, 40.0], [150.0, 40.0], egui::Modifiers::SHIFT);
        assert!(near(s.pivot, s.quad[1]), "{:?} {:?}", s.pivot, s.quad);
        drag(&mut s, [150.0, 40.0], [180.0, 40.0], egui::Modifiers::NONE);
        assert!(near(s.pivot, s.quad[1]), "proportional edge drag: {:?} {:?}", s.pivot, s.quad);
        // Skew: the centre follows the box's centre.
        let mut s = session();
        drag(&mut s, [50.0, 0.0], [70.0, 0.0], egui::Modifiers::COMMAND);
        assert!(near(s.pivot, [60.0, 25.0]), "{:?}", s.pivot);
        // Distort: a corner reference point stays on its corner.
        let mut s = session();
        s.pivot = [0.0, 50.0];
        drag(&mut s, [0.0, 50.0], [-20.0, 70.0], egui::Modifiers::COMMAND);
        assert!(near(s.pivot, [-20.0, 70.0]), "{:?}", s.pivot);
        // Perspective: the centre stays inside the box, on the box's own centre lines.
        let mut s = session();
        drag(&mut s, [100.0, 0.0], [120.0, 0.0], egui::Modifiers { command: true, alt: true, shift: true, ..Default::default() });
        let h = Homography::rect_to_quad([0.0, 0.0, 1.0, 1.0], s.quad).unwrap();
        let (x, y) = h.apply(0.5, 0.5);
        assert!(near(s.pivot, [x, y]), "{:?} vs {:?}", s.pivot, [x, y]);
        // ⌥ scales about the reference point: it stays where it is.
        let mut s = session();
        drag(&mut s, [100.0, 50.0], [150.0, 75.0], egui::Modifiers::ALT);
        assert!(near(s.pivot, [50.0, 25.0]), "{:?}", s.pivot);
        // Rotation turns about it.
        let mut s = session();
        drag(&mut s, [150.0, 25.0], [50.0, 125.0], egui::Modifiers::NONE);
        assert!(near(s.pivot, [50.0, 25.0]), "{:?}", s.pivot);
        // From a box collapsed to a point there is no place within it to keep: it stays put.
        assert_eq!(carry_pivot([[0.0, 0.0]; 4], corners([0.0, 0.0, 10.0, 10.0]), [5.0, 5.0]), [5.0, 5.0]);
    }

    #[test]
    fn move_rotate_and_distort() {
        let mut s = session();
        drag(&mut s, [40.0, 20.0], [50.0, 30.0], egui::Modifiers::NONE);
        assert!(close(s.quad, corners([10.0, 10.0, 110.0, 60.0])));
        assert_eq!(s.pivot, [60.0, 35.0]);
        // Rotate 90° about the centre (drag from right of the box to below it).
        let mut s = session();
        drag(&mut s, [150.0, 25.0], [50.0, 125.0], egui::Modifiers::NONE);
        let (_, _, angle, _) = readout(&s);
        assert!((angle - 90.0).abs() < 1e-6, "{angle}");
        // ⇧ snaps: a 50° drag lands on 45°.
        let mut s = session();
        let a = 50f64.to_radians();
        drag(&mut s, [150.0, 25.0], [50.0 + 100.0 * a.cos(), 25.0 + 100.0 * a.sin()], egui::Modifiers::SHIFT);
        assert!((readout(&s).2 - 45.0).abs() < 1e-6);
        // ⌘ corner = distort.
        let mut s = session();
        drag(&mut s, [100.0, 0.0], [120.0, -10.0], egui::Modifiers::COMMAND);
        assert_eq!(s.quad[1], [120.0, -10.0]);
        assert_eq!(s.quad[0], [0.0, 0.0]);
    }

    #[test]
    fn begin_and_commit_through_the_engine() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
        app.sync_views();
        app.session.execute("layer.new.layer", json!({})).unwrap();
        app.session
            .edit("paint", |doc, a| {
                doc.layer_mut(a.unwrap()).unwrap().surface_mut().unwrap().fill_rect(photocraft_geom::Rect::new(8, 8, 24, 24), &[1.0, 0.0, 0.0, 1.0]);
                Ok(())
            })
            .unwrap();
        let ctx = egui::Context::default();
        begin(&mut app, &ctx).unwrap();
        assert_eq!(app.ui.transform.as_ref().unwrap().rect, [8.0, 8.0, 24.0, 24.0]);
        scale_about_pivot(&mut app, 2.0, 2.0);
        commit(&mut app);
        let st = app.session.active().unwrap();
        let b = st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().content_bounds();
        assert!(b.width().abs_diff(32) <= 2 && b.x0.abs_diff(0) <= 1, "{b:?}");
        assert!(app.ui.transform.is_none() && app.transform_preview.is_none());
    }

    /// A commit the engine refuses (here a collapsed box; a result over the pixel budget, #3773, is
    /// refused the same way) shows the reason and keeps the box open, unchanged; picking another
    /// tool then ends it.
    #[test]
    fn a_refused_commit_keeps_the_box_open() {
        let ctx = egui::Context::default();
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
        let (revision, steps) = (app.session.active().unwrap().revision, app.session.active().unwrap().history.past_len());
        begin(&mut app, &ctx).unwrap();
        if let Some(t) = app.ui.transform.as_mut() {
            t.quad = [[16.0, 16.0]; 4];
        }
        commit(&mut app);
        let t = app.ui.transform.as_ref().expect("the box stays open");
        assert_eq!(t.quad, [[16.0, 16.0]; 4], "as the user left it");
        assert!(app.transform_preview.is_some(), "with its preview");
        assert!(app.ui.status.contains("collapses"), "{}", app.ui.status);
        let st = app.session.active().unwrap();
        assert_eq!((st.revision, st.history.past_len()), (revision, steps), "the document was not changed");
        app.ui.tool = if app.ui.tool == Tool::Brush { Tool::Move } else { Tool::Brush };
        end_if_left(&mut app);
        assert!(app.ui.transform.is_none() && app.transform_preview.is_none(), "another tool ends it");
    }

    /// #352: ⌥⌘T transforms a copy. OK leaves the original and a moved copy as one history step;
    /// Cancel leaves no copy and nothing to redo; with a selection the copy holds the selected
    /// pixels only.
    #[test]
    fn free_transform_a_copy() {
        let ctx = egui::Context::default();
        let bounds = |app: &PhotocraftApp, i: usize| app.session.active().unwrap().doc.layers[i].surface().unwrap().content_bounds();
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
        let (layers, steps) = (app.session.active().unwrap().doc.layers.len(), app.session.active().unwrap().history.past_len());
        crate::menus::invoke(&mut app, &ctx, "edit.freeTransformCopy", json!({})).unwrap();
        assert_eq!(app.ui.transform.as_ref().unwrap().made, Some(MadeLayer::Copy));
        assert!(!crate::menus::is_enabled(&app, "edit.freeTransformCopy"), "one transform at a time");
        // Esc: the copy goes too.
        cancel(&mut app);
        let st = app.session.active().unwrap();
        assert_eq!((st.doc.layers.len(), st.history.past_len()), (layers, steps));
        assert!(!st.history.can_redo(), "nothing to redo");
        // OK: original in place, a moved copy, one step.
        crate::menus::invoke(&mut app, &ctx, "edit.freeTransformCopy", json!({})).unwrap();
        if let Some(t) = app.ui.transform.as_mut() {
            t.quad = t.quad.map(|[x, y]| [x + 20.0, y]);
        }
        commit(&mut app);
        let st = app.session.active().unwrap();
        assert_eq!((st.doc.layers.len(), st.history.past_len()), (layers + 1, steps + 1));
        assert_eq!(bounds(&app, layers - 1), photocraft_geom::Rect::new(8, 8, 24, 24), "the original stays");
        assert_eq!(bounds(&app, layers), photocraft_geom::Rect::new(28, 8, 44, 24), "the copy moved");
        app.session.undo();
        assert_eq!(app.session.active().unwrap().doc.layers.len(), layers, "one undo removes copy and move");
        // With a selection: Layer via Copy, so only the selected pixels are copied.
        app.run("select.rect", json!({"x": 8, "y": 8, "width": 8, "height": 16})).unwrap();
        crate::menus::invoke(&mut app, &ctx, "edit.freeTransformCopy", json!({})).unwrap();
        assert_eq!(app.ui.transform.as_ref().unwrap().rect, [8.0, 8.0, 16.0, 24.0]);
        cancel(&mut app);
        assert_eq!(app.session.active().unwrap().doc.layers.len(), layers);
    }

    /// #1099: cancel, frame cleanup and a commit before the next frame all belong to the
    /// original document, including after tab reordering and with shared layer IDs.
    #[test]
    fn cancelling_a_copy_after_switching_documents_preserves_the_active_document() {
        for reorder in [false, true] {
            for finish in [cancel, end_if_left, commit] {
                let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
                let original = app.session.active().unwrap().doc.clone();
                let steps = app.session.active().unwrap().history.past_len();
                begin_copy(&mut app, &egui::Context::default()).unwrap();
                app.ui.transform.as_mut().unwrap().quad = corners([12.0, 8.0, 28.0, 24.0]);
                // Cloning a document deliberately shares layer IDs, but gets a fresh document ID.
                let other = app.session.active().unwrap().doc.as_ref().clone();
                app.session.add_document(other, None);
                app.run("layer.new.layer", json!({})).unwrap();
                app.run("layer.new.layer", json!({})).unwrap();
                assert!(app.session.undo());
                let st = app.session.active().unwrap();
                let (doc, history, redo, revision, layer) =
                    (st.doc.clone(), st.history.entries(), st.history.redo_labels().map(str::to_owned).collect::<Vec<_>>(), st.revision, st.active_layer);
                if reorder {
                    app.session.move_document(0, 1);
                }
                finish(&mut app);
                assert!(app.ui.transform.is_none() && app.transform_preview.is_none());
                let st = app.session.active().unwrap();
                assert!(Arc::ptr_eq(&st.doc, &doc), "the selected document's pixels are unchanged");
                assert_eq!((st.history.entries(), st.revision, st.active_layer), (history, revision, layer));
                assert_eq!(st.history.redo_labels().collect::<Vec<_>>(), redo);
                let origin = app.session.documents().iter().find(|st| st.doc.id == original.id).unwrap();
                assert!(Arc::ptr_eq(&origin.doc, &original), "only the originating copy is undone");
                assert_eq!(origin.history.past_len(), steps);
                assert!(!origin.history.can_redo());
            }
        }
    }

    #[test]
    fn cancelling_a_copy_after_its_document_closes_does_not_undo_another_document() {
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
        begin_copy(&mut app, &egui::Context::default()).unwrap();
        app.run("file.new", json!({"width": 64, "height": 64})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        let doc = app.session.active().unwrap().doc.clone();
        let steps = app.session.active().unwrap().history.past_len();
        app.session.close(0);
        end_if_left(&mut app);
        let st = app.session.active().unwrap();
        assert!(Arc::ptr_eq(&st.doc, &doc));
        assert_eq!(st.history.past_len(), steps);
        assert!(app.ui.transform.is_none() && app.transform_preview.is_none());
    }

    #[test]
    fn cancelling_an_already_undone_copy_preserves_history_and_redo() {
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
        begin_copy(&mut app, &egui::Context::default()).unwrap();
        assert!(app.session.undo());
        let st = app.session.active().unwrap();
        let (doc, history, redo) = (st.doc.clone(), st.history.entries(), st.history.redo_labels().map(str::to_owned).collect::<Vec<_>>());
        end_if_left(&mut app);
        let st = app.session.active().unwrap();
        assert!(Arc::ptr_eq(&st.doc, &doc));
        assert_eq!(st.history.entries(), history);
        assert_eq!(st.history.redo_labels().collect::<Vec<_>>(), redo);
        assert!(app.ui.transform.is_none() && app.transform_preview.is_none());
    }

    #[test]
    fn cancelling_a_copy_does_not_undo_a_later_edit() {
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
        begin_copy(&mut app, &egui::Context::default()).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        let st = app.session.active().unwrap();
        let (doc, history) = (st.doc.clone(), st.history.entries());
        cancel(&mut app);
        let st = app.session.active().unwrap();
        assert!(Arc::ptr_eq(&st.doc, &doc));
        assert_eq!(st.history.entries(), history);
    }

    /// #1384: the Layers panel's opacity, fill, visibility and blend edits to the moving layer
    /// and to the others show while the box is open, and survive the commit.
    #[test]
    fn edits_during_a_transform_show_in_the_preview() {
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
        app.ui.tool = Tool::Move;
        let bg = app.session.active().unwrap().doc.layers[0].id;
        let mut h = egui_kittest::Harness::builder().with_size(vec2(1200.0, 800.0)).with_max_steps(64).build_eframe(move |cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            app
        });
        h.run_steps(4);
        let ctx = h.ctx.clone();
        crate::menus::invoke(h.state_mut(), &ctx, "edit.freeTransform", json!({})).unwrap();
        if let Some(t) = h.state_mut().ui.transform.as_mut() {
            t.quad = t.quad.map(|[x, y]| [x + 20.0, y]);
        }
        h.run_steps(2);
        let layer = h.state().ui.transform.as_ref().unwrap().layer;
        assert_eq!(h.state().transform_preview.as_ref().unwrap().opacity, 1.0);

        h.state_mut().run("layer.setProps", json!({"layer": layer, "opacity": 0.5, "fill": 0.6, "blend": "Multiply"})).unwrap();
        h.state_mut().run("layer.setProps", json!({"layer": bg.0, "visible": false})).unwrap();
        h.run_steps(2);
        let app = h.state();
        assert!(app.ui.transform.is_some(), "the edits leave the box open");
        let pv = app.transform_preview.as_ref().unwrap();
        assert!((pv.opacity - 0.3).abs() < 1e-6, "the moving pixels take the new opacity × fill: {}", pv.opacity);
        assert!(!pv.doc.layer(bg).unwrap().visible, "the document under the box follows");
        let moving = pv.doc.layer(LayerId(layer)).unwrap();
        assert!(!moving.visible, "the moving layer stays out of the document under the box");
        assert_eq!(moving.blend, photocraft_color::BlendMode::Multiply);

        h.state_mut().run("layer.setProps", json!({"layer": layer, "visible": false})).unwrap();
        h.run_steps(2);
        assert_eq!(h.state().transform_preview.as_ref().unwrap().opacity, 0.0, "hiding the layer hides the moving pixels");
        h.state_mut().run("layer.setProps", json!({"layer": layer, "visible": true})).unwrap();

        commit(h.state_mut());
        let st = h.state().session.active().unwrap();
        let l = st.doc.layer(LayerId(layer)).unwrap();
        assert_eq!((l.opacity, l.fill_opacity, l.blend), (0.5, 0.6, photocraft_color::BlendMode::Multiply));
        assert_eq!(l.surface().unwrap().content_bounds(), photocraft_geom::Rect::new(28, 8, 44, 24));
    }

    /// #1384 with a selection: the pixels it leaves behind stay in the rebuilt preview without
    /// the lifted ones, and take the layer's new properties.
    #[test]
    fn edits_during_a_selection_transform_keep_the_lifted_pixels_out() {
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
        app.ui.tool = Tool::Move;
        app.session.execute("select.rect", json!({"x": 8, "y": 8, "width": 8, "height": 16})).unwrap();
        let mut h = egui_kittest::Harness::builder().with_size(vec2(1200.0, 800.0)).with_max_steps(64).build_eframe(move |cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            app
        });
        h.run_steps(4);
        let ctx = h.ctx.clone();
        crate::menus::invoke(h.state_mut(), &ctx, "edit.freeTransform", json!({})).unwrap();
        h.run_steps(2);
        let layer = LayerId(h.state().ui.transform.as_ref().unwrap().layer);
        // Alpha in the preview's copy of the layer: in the lifted half (x 8..16) and the rest (16..24).
        let left = |app: &PhotocraftApp| {
            let s = app.transform_preview.as_ref().unwrap().doc.layer(layer).unwrap().surface().unwrap().clone();
            (s.sample_channel(10, 12, 3), s.sample_channel(20, 12, 3))
        };
        assert_eq!(left(h.state()), (0.0, 1.0), "the selected half is lifted out");

        h.state_mut().run("layer.setProps", json!({"layer": layer.0, "opacity": 0.4})).unwrap();
        h.run_steps(2);
        let app = h.state();
        assert_eq!(left(app), (0.0, 1.0), "still without the lifted pixels");
        let moving = app.transform_preview.as_ref().unwrap().doc.layer(layer).unwrap();
        assert!(moving.visible && (moving.opacity - 0.4).abs() < 1e-6, "what stays behind is drawn with the new opacity");
        assert!((app.transform_preview.as_ref().unwrap().opacity - 0.4).abs() < 1e-6);
    }

    /// #670: picking another tool applies the open transform, as one history step.
    #[test]
    fn picking_another_tool_applies_the_transform() {
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
        app.ui.tool = Tool::Move;
        let steps = app.session.active().unwrap().history.past_len();
        let mut h = egui_kittest::Harness::builder().with_size(vec2(1200.0, 800.0)).with_max_steps(64).build_eframe(move |cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            app
        });
        h.run_steps(4);
        let ctx = h.ctx.clone();
        crate::menus::invoke(h.state_mut(), &ctx, "edit.freeTransform", json!({})).unwrap();
        if let Some(t) = h.state_mut().ui.transform.as_mut() {
            t.quad = t.quad.map(|[x, y]| [x + 20.0, y]);
        }
        h.run_steps(2);
        assert!(h.state().ui.transform.is_some(), "open while the tool stays");
        h.key_press(egui::Key::B);
        h.run_steps(2);
        let app = h.state();
        assert_eq!(app.ui.tool, Tool::Brush);
        assert!(app.ui.transform.is_none() && app.transform_preview.is_none());
        let st = app.session.active().unwrap();
        assert_eq!(st.history.past_len(), steps + 1);
        assert_eq!(st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().content_bounds(), photocraft_geom::Rect::new(28, 8, 44, 24));
    }

    /// #1400: selecting another layer while transforming applies the box to the layer it was on,
    /// and the next transform starts on the newly selected layer.
    #[test]
    fn selecting_another_layer_applies_the_transform_and_the_next_one_follows() {
        let ctx = egui::Context::default();
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
        let first = app.session.active().unwrap().active_layer.unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        app.session
            .edit("paint", |doc, a| {
                doc.layer_mut(a.unwrap()).unwrap().surface_mut().unwrap().fill_rect(photocraft_geom::Rect::new(40, 40, 56, 60), &[0.0, 0.0, 1.0, 1.0]);
                Ok(())
            })
            .unwrap();
        let second = app.session.active().unwrap().active_layer.unwrap();
        app.run("layer.select", json!({"layer": first.0})).unwrap();
        app.ui.tool = Tool::Move;
        crate::menus::invoke(&mut app, &ctx, "edit.freeTransform", json!({})).unwrap();
        if let Some(t) = app.ui.transform.as_mut() {
            t.quad = t.quad.map(|[x, y]| [x + 20.0, y]);
        }
        let steps = app.session.active().unwrap().history.past_len();
        app.run("layer.select", json!({"layer": second.0})).unwrap();
        end_if_left(&mut app);
        assert!(app.ui.transform.is_none() && app.transform_preview.is_none(), "the box left the old layer");
        let st = app.session.active().unwrap();
        assert_eq!(st.active_layer, Some(second));
        assert_eq!(st.history.past_len(), steps + 1, "applied as one step");
        let bounds = |id: LayerId| st.doc.layer(id).unwrap().surface().unwrap().content_bounds();
        assert_eq!(bounds(first), photocraft_geom::Rect::new(28, 8, 44, 24), "applied to the layer it was on");
        assert_eq!(bounds(second), photocraft_geom::Rect::new(40, 40, 56, 60), "the new layer is untouched");
        crate::menus::invoke(&mut app, &ctx, "edit.freeTransform", json!({})).unwrap();
        let t = app.ui.transform.as_ref().unwrap();
        assert_eq!((t.layer, t.rect), (second.0, [40.0, 40.0, 56.0, 60.0]), "the next box is on the selected layer");
    }

    /// Transform Selection moves the selection outline, not a layer: selecting a layer keeps it open.
    #[test]
    fn selecting_another_layer_keeps_a_transform_selection_open() {
        let ctx = egui::Context::default();
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
        let first = app.session.active().unwrap().active_layer.unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        app.run("select.all", json!({})).unwrap();
        begin_selection(&mut app, &ctx).unwrap();
        app.run("layer.select", json!({"layer": first.0})).unwrap();
        end_if_left(&mut app);
        assert!(app.ui.transform.as_ref().is_some_and(|t| t.selection));
    }

    /// A control-channel stroke that picks another tool applies the box first, then paints.
    #[test]
    fn a_pointer_request_with_another_tool_applies_the_transform_first() {
        let ctx = egui::Context::default();
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
        app.ui.tool = Tool::Move;
        crate::menus::invoke(&mut app, &ctx, "edit.freeTransform", json!({})).unwrap();
        if let Some(t) = app.ui.transform.as_mut() {
            t.quad = t.quad.map(|[x, y]| [x + 20.0, y]);
        }
        let steps = app.session.active().unwrap().history.past_len();
        let ev = |kind: &str, x: f64| json!({"kind": kind, "x": x, "y": 50.0});
        let events = json!([ev("down", 4.0), ev("move", 32.0), ev("move", 60.0), ev("up", 60.0)]);
        let (req, _rx) = crate::control::ControlRequest::new("ui.pointer", json!({"tool": "brush", "events": events}));
        let _ = crate::control::handle(&mut app, &ctx, &req);
        assert!(app.ui.transform.is_none());
        let st = app.session.active().unwrap();
        assert_eq!(st.history.past_len(), steps + 2, "the transform, then the stroke");
        let b = st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().content_bounds();
        assert!(b.x0 < 8 && b.x1 >= 44 && b.y0 == 8 && b.y1 > 50, "moved square and stroke: {b:?}");
    }

    /// Drags with real pointer events (snapping off so the drops land exactly).
    fn press_drag(app: &mut PhotocraftApp, from: [f64; 2], to: [f64; 2], m: egui::Modifiers) {
        crate::canvas::tool_event(app, ToolEvent::Down { x: from[0], y: from[1], pressure: 1.0 }, m);
        crate::canvas::tool_event(app, ToolEvent::Up { x: to[0], y: to[1] }, m);
    }

    #[test]
    fn distort_mode_moves_one_corner_and_ignores_rotate_and_edges() {
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
        app.ui.extras.snap = false;
        let ctx = egui::Context::default();
        crate::menus::invoke(&mut app, &ctx, "edit.transform.distort", json!({})).unwrap();
        let q0 = app.ui.transform.as_ref().unwrap().quad;
        assert_eq!(app.ui.transform.as_ref().unwrap().mode, TransformMode::Distort);
        // Outside (rotate) and an edge handle do nothing in Distort.
        press_drag(&mut app, [44.0, 44.0], [54.0, 10.0], egui::Modifiers::NONE);
        press_drag(&mut app, [24.0, 16.0], [34.0, 16.0], egui::Modifiers::NONE);
        assert_eq!(app.ui.transform.as_ref().unwrap().quad, q0);
        // A corner moves alone, with no keys held.
        press_drag(&mut app, q0[1], [q0[1][0] + 7.0, q0[1][1] - 5.0], egui::Modifiers::NONE);
        let q = app.ui.transform.as_ref().unwrap().quad;
        assert_eq!(q[1], [q0[1][0] + 7.0, q0[1][1] - 5.0]);
        assert_eq!([q[0], q[2], q[3]], [q0[0], q0[2], q0[3]]);
        // Free Transform from the menu switches the live box back.
        crate::menus::invoke(&mut app, &ctx, "edit.freeTransform", json!({})).unwrap();
        assert_eq!(app.ui.transform.as_ref().unwrap().mode, TransformMode::Free);
    }

    /// #3941: Edit › Transform › Rotate must rotate from a corner handle, not scale.
    #[test]
    fn rotate_mode_turns_the_box_from_a_corner_handle() {
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
        app.ui.extras.snap = false;
        let ctx = egui::Context::default();
        crate::menus::invoke(&mut app, &ctx, "edit.transform.rotate", json!({})).unwrap();
        assert_eq!(app.ui.transform.as_ref().unwrap().mode, TransformMode::Rotate);
        let q0 = app.ui.transform.as_ref().unwrap().quad;
        press_drag(&mut app, q0[1], [q0[1][0], q0[1][1] + 16.0], egui::Modifiers::NONE);
        let q = app.ui.transform.as_ref().unwrap().quad;
        let w0 = (q0[1][0] - q0[0][0]).hypot(q0[1][1] - q0[0][1]);
        let w = (q[1][0] - q[0][0]).hypot(q[1][1] - q[0][1]);
        assert!((w - w0).abs() < 1e-6, "side length stays {w0}, got {w}: {q:?}");
        assert!((q[0][1] - q[1][1]).abs() > 1.0, "the box rotated, not scaled: {q:?}");
        crate::menus::invoke(&mut app, &ctx, "edit.transform.scale", json!({})).unwrap();
        assert_eq!(app.ui.transform.as_ref().unwrap().mode, TransformMode::Free);
    }

    fn near2(a: Option<[f64; 2]>, b: [f64; 2]) -> bool {
        a.is_some_and(|a| (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9)
    }

    /// #1767: just outside a corner is the rotate zone: the OS pointer hides for the drawn curved
    /// arrow, which bends round that corner.
    #[test]
    fn the_preview_leaves_out_cells_past_the_horizon() {
        // #4863: the bottom corners pulled in and one dragged far out: a concave quad whose
        // source straddles the horizon. Its cells there are left out; an ordinary quad keeps all.
        let past = Homography::rect_to_quad([0.0, 0.0, 1.0, 1.0], [[0.0, 0.0], [300.0, 0.0], [60.0, 60.0], [0.0, 300.0]]).unwrap();
        let ahead = front_vertices(&past, 24);
        assert!(ahead.iter().any(|a| *a) && ahead.iter().any(|a| !*a), "some of the grid is past the horizon");
        // Three corners are on one side (drawn); the top-left one is past the horizon.
        assert!(ahead[24 * 25 + 24] && ahead[24] && ahead[24 * 25], "the three corners in front are drawn");
        assert!(!ahead[0], "the corner past the horizon is left out");
        let plain = Homography::rect_to_quad([0.0, 0.0, 1.0, 1.0], [[0.0, 0.0], [300.0, 0.0], [280.0, 260.0], [0.0, 300.0]]).unwrap();
        assert!(front_vertices(&plain, 24).iter().all(|a| *a));
    }

    #[test]
    fn hovering_just_outside_a_corner_asks_for_the_rotate_cursor() {
        let mut app = app_with_square(256, photocraft_geom::Rect::new(32, 32, 160, 160));
        crate::menus::invoke(&mut app, &egui::Context::default(), "edit.freeTransform", json!({})).unwrap();
        let tol = handle_tolerance(&app);
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let (tl, br) = ([32.0 - tol * 1.5, 32.0 - tol * 1.5], [160.0 + tol * 1.5, 160.0 + tol * 1.5]);
        assert_eq!(cursor(&app, tl, false), Some(CursorIcon::None), "the arrow is drawn instead");
        assert!(near2(rotate_cursor_dir(&app, tl), [h, h]), "{:?}", rotate_cursor_dir(&app, tl));
        assert!(near2(rotate_cursor_dir(&app, br), [-h, -h]), "bends round the bottom-right corner");
        assert_eq!(cursor(&app, [60.0, 60.0], false), Some(CursorIcon::Move));
        assert_eq!(rotate_cursor_dir(&app, [60.0, 60.0]), None, "inside moves");
        assert_eq!(rotate_cursor_dir(&app, [32.0, 32.0]), None, "the corner handle scales");
        assert_eq!(rotate_cursor_dir(&app, [f64::NAN, 32.0]), None);
    }

    /// #1767: the readout beside the pointer during a drag (Photoshop's transformation values):
    /// the angle while rotating, W/H while scaling, ΔX/ΔY while moving; nothing otherwise.
    #[test]
    fn the_live_readout_follows_the_drag() {
        let mut app = app_with_square(256, photocraft_geom::Rect::new(32, 32, 160, 160));
        app.ui.extras.snap = false;
        crate::menus::invoke(&mut app, &egui::Context::default(), "edit.freeTransform", json!({})).unwrap();
        assert_eq!(live_readout(&app), None, "not dragging");
        let ev = |app: &mut PhotocraftApp, e| crate::canvas::tool_event(app, e, egui::Modifiers::NONE);
        // A quarter turn about the centre (96, 96): from right of the box to below it.
        ev(&mut app, ToolEvent::Down { x: 220.0, y: 96.0, pressure: 1.0 });
        ev(&mut app, ToolEvent::Move { x: 96.0, y: 220.0, pressure: 1.0 });
        let Some(Readout::Angle(a)) = live_readout(&app) else { panic!("{:?}", live_readout(&app)) };
        assert!((a - 90.0).abs() < 1e-6, "{a}");
        ev(&mut app, ToolEvent::Up { x: 96.0, y: 220.0 });
        assert_eq!(live_readout(&app), None, "released");
        let mut app = app_with_square(256, photocraft_geom::Rect::new(32, 32, 160, 160));
        app.ui.extras.snap = false;
        crate::menus::invoke(&mut app, &egui::Context::default(), "edit.freeTransform", json!({})).unwrap();
        ev(&mut app, ToolEvent::Down { x: 60.0, y: 60.0, pressure: 1.0 });
        ev(&mut app, ToolEvent::Move { x: 65.0, y: 57.0, pressure: 1.0 });
        assert_eq!(live_readout(&app), Some(Readout::Offset([5.0, -3.0])));
        ev(&mut app, ToolEvent::Up { x: 65.0, y: 57.0 });
        // The right edge doubles the width (proportional by default: the height too).
        ev(&mut app, ToolEvent::Down { x: 165.0, y: 93.0, pressure: 1.0 });
        ev(&mut app, ToolEvent::Move { x: 293.0, y: 93.0, pressure: 1.0 });
        let Some(Readout::Scale([w, h])) = live_readout(&app) else { panic!("{:?}", live_readout(&app)) };
        assert!((w - 200.0).abs() < 1e-6 && (h - 200.0).abs() < 1e-6, "{w} {h}");
    }

    #[test]
    fn a_distort_corner_stops_before_the_box_folds_over() {
        // #1323: dragging a corner across the opposite diagonal stops short of it, so the box
        // stays convex instead of folding into a bow tie.
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
        app.ui.extras.snap = false;
        let ctx = egui::Context::default();
        crate::menus::invoke(&mut app, &ctx, "edit.transform.distort", json!({})).unwrap();
        let q0 = app.ui.transform.as_ref().unwrap().quad;
        // Top-left far past the bottom-right corner.
        press_drag(&mut app, q0[0], [60.0, 60.0], egui::Modifiers::NONE);
        let q = app.ui.transform.as_ref().unwrap().quad;
        let sign = turns(&q0)[0].signum();
        assert!(turns(&q).iter().all(|t| t * sign > 0.0), "still convex: {q:?}");
        assert_eq!([q[1], q[2], q[3]], [q0[1], q0[2], q0[3]], "the other corners stay put");
        // It went as far as it could along the drag: up to the TR–BL diagonal (x + y = 32).
        assert!(q[0][0] > 15.0 && q[0][0] < 16.0 && (q[0][0] - q[0][1]).abs() < 1e-6, "{q:?}");
        // A drag that keeps the box convex is not held back.
        press_drag(&mut app, q[0], [2.0, 4.0], egui::Modifiers::NONE);
        assert_eq!(app.ui.transform.as_ref().unwrap().quad[0], [2.0, 4.0]);
    }

    #[test]
    fn keep_convex_leaves_a_degenerate_start_alone() {
        let flat = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let to = [[5.0, 5.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        assert_eq!(keep_convex(flat, to), to);
        let nan = [[f64::NAN, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let _ = keep_convex(nan, to);
    }

    #[test]
    fn the_legacy_preference_swaps_corner_proportions_and_stretches_edges() {
        let none = egui::Modifiers::NONE;
        let corner = Hit::Corner(1);
        let edge = Hit::Edge(1);
        // Default: proportional (⇧ frees), so the drag keeps its keys.
        assert!(!legacy_mods(false, corner, none).shift);
        assert!(!legacy_mods(false, edge, none).shift);
        // Legacy: corners free by default, ⇧ keeps proportions; edges always stretch one axis;
        // ⌘ gestures are left alone.
        assert!(legacy_mods(true, corner, none).shift);
        assert!(!legacy_mods(true, corner, egui::Modifiers::SHIFT).shift);
        assert!(legacy_mods(true, edge, none).shift);
        assert!(legacy_mods(true, edge, egui::Modifiers::SHIFT).shift);
        assert!(!legacy_mods(true, corner, egui::Modifiers::COMMAND).shift);
        assert!(!legacy_mods(true, edge, egui::Modifiers::COMMAND).shift);
        // Through the box: a plain corner drag, then a plain right-edge drag, on a 16×16 square.
        for (legacy, proportional) in [(false, true), (true, false)] {
            for at_edge in [false, true] {
                let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
                app.ui.extras.snap = false;
                app.session.prefs.edit(|p| p.general.use_legacy_free_transform = legacy);
                begin(&mut app, &egui::Context::default()).unwrap();
                let q0 = app.ui.transform.as_ref().unwrap().quad;
                let from = if at_edge { [q0[1][0], (q0[1][1] + q0[2][1]) / 2.0] } else { q0[2] };
                press_drag(&mut app, from, [from[0] + 16.0, from[1]], none);
                let q = app.ui.transform.as_ref().unwrap().quad;
                let (w, h) = (q[2][0] - q[0][0], q[2][1] - q[0][1]);
                assert_eq!((w - h).abs() < 1e-6, proportional, "legacy {legacy}, edge {at_edge}: {w} × {h}");
            }
        }
    }

    /// #4174: turning off the options bar's Maintain Aspect Ratio link frees the handles too, as in
    /// Photoshop: a plain corner or edge drag stretches one way, ⇧ keeps the proportions. Linked
    /// (the default) is unchanged, and so is the legacy preference.
    #[test]
    fn the_aspect_link_decides_whether_handle_drags_keep_proportions() {
        let (none, shift) = (egui::Modifiers::NONE, egui::Modifiers::SHIFT);
        for (unlinked, legacy, mods, proportional) in
            [(false, false, none, true), (true, false, none, false), (true, false, shift, true), (true, true, none, false)]
        {
            for at_edge in [false, true] {
                let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
                app.ui.extras.snap = false;
                app.ui.transform_unlinked = unlinked;
                app.session.prefs.edit(|p| p.general.use_legacy_free_transform = legacy);
                begin(&mut app, &egui::Context::default()).unwrap();
                let q0 = app.ui.transform.as_ref().unwrap().quad;
                let from = if at_edge { [q0[1][0], (q0[1][1] + q0[2][1]) / 2.0] } else { q0[2] };
                press_drag(&mut app, from, [from[0] + 16.0, from[1]], mods);
                let q = app.ui.transform.as_ref().unwrap().quad;
                let (w, h) = (q[2][0] - q[0][0], q[2][1] - q[0][1]);
                let case = format!("unlinked {unlinked}, legacy {legacy}, shift {}, edge {at_edge}: {w} × {h}", mods.shift);
                if legacy && at_edge {
                    // Legacy edges always stretch their one axis.
                    assert!((w - 32.0).abs() < 1e-6 && (h - 16.0).abs() < 1e-6, "{case}");
                } else {
                    assert_eq!((w - h).abs() < 1e-6, proportional, "{case}");
                    assert!((w - 32.0).abs() < 1e-6, "{case}");
                }
            }
        }
        // ⌘ gestures (skew, distort) are left alone.
        assert!(!unlinked_mods(true, false, Hit::Corner(0), egui::Modifiers::COMMAND).shift);
        assert!(!unlinked_mods(true, false, Hit::Inside, none).shift);
    }

    #[test]
    fn right_click_while_transforming_switches_the_mode() {
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
        let ctx = egui::Context::default();
        begin(&mut app, &ctx).unwrap();
        assert!(crate::canvas_tool_menu::open_transform(&mut app, &ctx, [10.0, 10.0]));
        let menu = app.ui.canvas_tool_menu.clone().unwrap();
        let ids: Vec<&str> = crate::canvas_tool_menu::rows(&menu).iter().flatten().map(|e| e.1).collect();
        assert!(ids.contains(&"edit.transform.distort") && ids.contains(&"edit.freeTransform"));
        crate::canvas_tool_menu::choose(&mut app, &ctx, "edit.transform.distort");
        assert_eq!(app.ui.transform.as_ref().unwrap().mode, TransformMode::Distort);
    }

    #[test]
    fn right_click_menu_warp_row_enters_warp_mode() {
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
        let ctx = egui::Context::default();
        begin(&mut app, &ctx).unwrap();
        assert!(crate::canvas_tool_menu::open_transform(&mut app, &ctx, [10.0, 10.0]));
        let menu = app.ui.canvas_tool_menu.clone().unwrap();
        assert!(crate::canvas_tool_menu::rows(&menu).iter().flatten().any(|e| e.1 == "edit.transform.warp"));
        assert!(crate::canvas_tool_menu::available(&app, &menu, "edit.transform.warp"));
        // Only the current mode (Free Transform) is checked: not Scale, Rotate or Warp.
        let checked: Vec<&str> =
            crate::canvas_tool_menu::rows(&menu).iter().flatten().map(|e| e.1).filter(|id| crate::canvas_tool_menu::mode_checked(&app, &menu, id)).collect();
        assert_eq!(checked, ["edit.freeTransform"]);
        crate::canvas_tool_menu::choose(&mut app, &ctx, "edit.transform.warp");
        assert!(app.ui.transform.as_ref().unwrap().warp.is_some());
        // Already warping: the row is no longer offered.
        assert!(!crate::canvas_tool_menu::available(&app, &menu, "edit.transform.warp"));
    }

    #[test]
    fn right_click_menu_offers_warp_for_a_path() {
        // Photoshop's Transform Path menu has Warp too (#3027).
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
        let ctx = egui::Context::default();
        begin(&mut app, &ctx).unwrap();
        app.ui.transform.as_mut().unwrap().path = Some(json!({}));
        assert!(crate::canvas_tool_menu::open_transform(&mut app, &ctx, [10.0, 10.0]));
        let menu = app.ui.canvas_tool_menu.clone().unwrap();
        assert!(crate::canvas_tool_menu::available(&app, &menu, "edit.transform.warp"));
    }

    fn app_with_square(size: u32, fill: photocraft_geom::Rect) -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": size, "height": size})).unwrap();
        app.sync_views();
        app.session.execute("layer.new.layer", json!({})).unwrap();
        app.session
            .edit("paint", |doc, a| {
                doc.layer_mut(a.unwrap()).unwrap().surface_mut().unwrap().fill_rect(fill, &[1.0, 0.0, 0.0, 1.0]);
                Ok(())
            })
            .unwrap();
        app
    }

    /// #2439: a layer under others keeps its place in the stack while the box is open. The
    /// canvas under the box shows only the layers below it; the ones over it are composited on
    /// their own and drawn over the moving pixels.
    #[test]
    fn free_transform_keeps_the_layer_under_the_layers_over_it() {
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
        let red = app.session.active().unwrap().active_layer.unwrap();
        app.session.execute("layer.new.layer", json!({})).unwrap();
        app.session
            .edit("paint", |doc, a| {
                doc.layer_mut(a.unwrap()).unwrap().surface_mut().unwrap().fill_rect(photocraft_geom::Rect::new(16, 16, 40, 40), &[0.0, 0.0, 1.0, 1.0]);
                Ok(())
            })
            .unwrap();
        let blue = app.session.active().unwrap().active_layer.unwrap();
        let bg = app.session.active().unwrap().doc.layers[0].id;
        app.session.execute("layer.select", json!({"layer": red.0})).unwrap();
        let ctx = gpu_ctx();
        begin(&mut app, &ctx).unwrap();
        let pv = app.transform_preview.as_ref().unwrap();
        assert!(pv.doc.layer(bg).unwrap().visible, "the layers below show under the box");
        assert!(!pv.doc.layer(blue).unwrap().visible, "the layer over it is drawn over the box instead");
        let (texture, rect) = pv.above.as_ref().and_then(|a| a.shown.as_ref()).expect("something covers the layer");
        assert_eq!(*rect, photocraft_geom::Rect::new(16, 16, 40, 40));
        assert_eq!(texture.size(), [24, 24]);
        cancel(&mut app);

        // The top layer: nothing to draw over it.
        app.session.execute("layer.select", json!({"layer": blue.0})).unwrap();
        begin(&mut app, &ctx).unwrap();
        let pv = app.transform_preview.as_ref().unwrap();
        assert!(pv.above.as_ref().is_some_and(|a| a.shown.is_none()));
        assert!(pv.doc.layer(red).unwrap().visible && pv.doc.layer(bg).unwrap().visible);
        cancel(&mut app);
    }

    #[test]
    fn plain_layers_from_the_top_cover_the_moving_one() {
        use photocraft_color::BlendMode;
        use photocraft_doc::Layer;
        let px = |name: &str| Layer::raster(name, photocraft_color::PixelFormat::RGBA8);
        let covers = |layers: &[Layer], id: LayerId| {
            let mut out = Vec::new();
            covering(layers, id, &mut out);
            out
        };
        // In a group: the group's layers over it, then the ones over the group; never the layers
        // clipped to it.
        let (below, target, mut clip, inner_top, top) = (px("below"), px("target"), px("clip"), px("inner top"), px("top"));
        clip.clipped = true;
        let (t, c, inner, tp) = (target.id, clip.id, inner_top.id, top.id);
        let mut layers = vec![below, Layer::group("group", vec![target, clip, inner_top]), top];
        assert_eq!(covers(&layers, t), [tp, inner]);
        assert_eq!(covers(&layers, c), [tp, inner], "a clipped layer: covered by what is over its clipping group");

        // Under the box, the covering layers leave the document; over it, they alone show.
        let above = Above { ctx: egui::Context::default(), name: String::new(), max_side: 64, ids: vec![tp, inner], shown: None };
        let mut under = layers.clone();
        above.hide_in(&mut under);
        let vis = |ls: &[Layer]| [ls[0].visible, ls[1].visible, ls[2].visible];
        assert_eq!(vis(&under), [true, true, false]);
        let mut over = layers.clone();
        show_only(&mut over, &above.ids);
        assert_eq!(vis(&over), [false, true, true]);
        let inside: Vec<bool> = match &over[1].content {
            LayerContent::Group(g) => g.children.iter().map(|l| l.visible).collect(),
            _ => Vec::new(),
        };
        assert_eq!(inside, [false, false, true]);

        // A layer that blends with what is under it stops the search: it and everything under it
        // stay in the document under the box.
        layers[2].blend = BlendMode::Multiply;
        assert!(covers(&layers, t).is_empty());
        layers[2].blend = BlendMode::Normal;
        let mut multiply = px("multiply");
        multiply.blend = BlendMode::Multiply;
        layers.insert(2, multiply);
        assert_eq!(covers(&layers, t), [tp]);
        // A group that isn't Normal or Pass Through keeps what is inside it.
        layers[1].blend = BlendMode::Screen;
        layers.remove(2);
        assert_eq!(covers(&layers, t), [tp]);
        assert_eq!(covers(&layers, LayerId::fresh()), [tp], "not in the stack: down to the first layer that isn't plain");
    }

    /// A context that reports a real GPU's texture limit (the default reports 2048).
    fn gpu_ctx() -> egui::Context {
        let ctx = egui::Context::default();
        // Clear the font atlas delta: epaint panics on dropping unapplied deltas in debug builds.
        let mut out = ctx.run_ui(egui::RawInput { max_texture_side: Some(16384), ..Default::default() }, |_| {});
        out.textures_delta.clear();
        ctx
    }

    #[test]
    fn preview_texture_is_full_resolution() {
        // #91: the preview used to be sampled down to 2048 px, so big layers looked blurry at 100%.
        let mut app = app_with_square(3000, photocraft_geom::Rect::new(10, 20, 2810, 2420));
        let ctx = gpu_ctx();
        begin(&mut app, &ctx).unwrap();
        let pv = app.transform_preview.as_ref().unwrap();
        assert_eq!(pv.texture.size(), [2800, 2400], "one texel per layer pixel");
        // At 100% the texture has at least as many texels as the box covers screen pixels.
        let t = app.ui.transform.as_ref().unwrap();
        assert!(pv.texture.size()[0] as f64 >= t.rect[2] - t.rect[0]);
    }

    /// The textured meshes `draw_overlay` paints for the moving pixels (not the box or handles).
    fn preview_meshes(app: &PhotocraftApp, xf: &ViewXform) -> Vec<egui::Mesh> {
        let ctx = gpu_ctx();
        let mut out = ctx.run_ui(egui::RawInput { max_texture_side: Some(16384), ..Default::default() }, |ui| {
            let painter = ui.ctx().layer_painter(egui::LayerId::background());
            draw_overlay(app, &painter, xf);
        });
        out.textures_delta.clear();
        out.shapes
            .into_iter()
            .filter_map(|s| match s.shape {
                egui::Shape::Mesh(m) if m.texture_id != egui::TextureId::default() => Some((*m).clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn preview_pixels_past_the_canvas_are_hidden() {
        // #2027: a box scaled past the canvas drew its pixels over the pasteboard.
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 56, 56));
        begin(&mut app, &gpu_ctx()).unwrap();
        scale_about_pivot(&mut app, 2.0, 2.0);
        let q = app.ui.transform.as_ref().unwrap().quad;
        assert!(q[0][0] < -10.0 && q[2][1] > 74.0, "the box reaches past the canvas: {q:?}");
        let tol = 1e-3;
        for (rotation, flip) in [(0.0, false), (30.0, false), (-75.0, true)] {
            let xf = ViewXform {
                rect: egui::Rect::from_min_size(egui::Pos2::ZERO, vec2(400.0, 400.0)),
                zoom: 2.0,
                center: [32.0, 32.0],
                flip,
                rotation,
                aspect: 1.0,
            };
            let meshes = preview_meshes(&app, &xf);
            let verts: Vec<[f64; 2]> = meshes.iter().flat_map(|m| m.vertices.iter().map(|v| xf.to_doc(v.pos))).collect();
            assert!(!verts.is_empty(), "the preview still draws (rotation {rotation})");
            for d in &verts {
                assert!(d[0] >= -tol && d[1] >= -tol && d[0] <= 64.0 + tol && d[1] <= 64.0 + tol, "vertex {d:?} off the canvas (rotation {rotation})");
            }
            // The clipped mesh still covers the whole canvas: the corners lie on it.
            for corner in [[0.0, 0.0], [64.0, 0.0], [64.0, 64.0], [0.0, 64.0]] {
                assert!(verts.iter().any(|d| (d[0] - corner[0]).abs() < 1e-2 && (d[1] - corner[1]).abs() < 1e-2), "corner {corner:?} (rotation {rotation})");
            }
        }
        // Inside the canvas the mesh is drawn unchanged (24×24 grid).
        cancel(&mut app);
        begin(&mut app, &gpu_ctx()).unwrap();
        let xf = ViewXform {
            rect: egui::Rect::from_min_size(egui::Pos2::ZERO, vec2(200.0, 200.0)),
            zoom: 1.0,
            center: [32.0, 32.0],
            flip: false,
            rotation: 0.0,
            aspect: 1.0,
        };
        let meshes = preview_meshes(&app, &xf);
        assert_eq!(meshes.iter().map(|m| m.vertices.len()).sum::<usize>(), 25 * 25);
    }

    #[test]
    fn shift_body_drag_locks_to_eight_directions_and_switches() {
        let mut s = session();
        let g = Gesture { hit: Hit::Inside, start: [40.0, 20.0], quad0: s.quad, pivot0: s.pivot };
        apply_drag(&mut s, g, [60.0, 23.0], egui::Modifiers::SHIFT);
        assert!(close(s.quad, corners([20.0, 0.0, 120.0, 50.0])), "horizontal: {:?}", s.quad);
        apply_drag(&mut s, g, [60.0, 39.0], egui::Modifiers::SHIFT);
        assert!((s.quad[0][0] - s.quad[0][1]).abs() < 1e-9 && s.quad[0][0] > 15.0, "diagonal: {:?}", s.quad[0]);
        apply_drag(&mut s, g, [41.0, 60.0], egui::Modifiers::SHIFT);
        assert!(close(s.quad, corners([0.0, 40.0, 100.0, 90.0])), "vertical: {:?}", s.quad);
    }

    #[test]
    fn alt_click_moves_the_reference_point_and_edges_skew() {
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
        let ctx = egui::Context::default();
        begin(&mut app, &ctx).unwrap();
        assert!(pointer(&mut app, ToolEvent::Down { x: 40.0, y: 40.0, pressure: 1.0 }, egui::Modifiers::ALT));
        pointer(&mut app, ToolEvent::Up { x: 40.0, y: 40.0 }, egui::Modifiers::ALT);
        let t = app.ui.transform.as_ref().unwrap();
        assert_eq!(t.pivot, [40.0, 40.0]);
        assert_eq!(t.quad, corners(t.rect), "the box didn't move");
        // ⌘ on an edge skews: both corners of the top edge move.
        let mut s = session();
        drag(&mut s, [50.0, 0.0], [70.0, 5.0], egui::Modifiers::COMMAND);
        assert_eq!((s.quad[0], s.quad[1], s.quad[2]), ([20.0, 5.0], [120.0, 5.0], [100.0, 50.0]));
        // ⌘⇧: only along the edge.
        let mut s = session();
        drag(&mut s, [50.0, 0.0], [70.0, 5.0], egui::Modifiers::COMMAND | egui::Modifiers::SHIFT);
        assert_eq!((s.quad[0], s.quad[1]), ([20.0, 0.0], [120.0, 0.0]));
        // ⌘⌥⇧ on a corner: perspective (the paired corner mirrors).
        let mut s = session();
        drag(&mut s, [100.0, 0.0], [101.0, -10.0], egui::Modifiers::COMMAND | egui::Modifiers::ALT | egui::Modifiers::SHIFT);
        assert_eq!((s.quad[1], s.quad[2]), ([100.0, -10.0], [100.0, 60.0]));
    }

    /// `cargo test --release -p photocraft-ui-egui transform_preview_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn transform_preview_bench() {
        let (w, h) = (6000, 4000);
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": w, "height": h})).unwrap();
        app.sync_views();
        app.session.execute("layer.new.layer", json!({})).unwrap();
        app.session.execute("filter.render.clouds", json!({})).unwrap();
        let ctx = gpu_ctx();
        let t0 = std::time::Instant::now();
        begin(&mut app, &ctx).unwrap();
        let begin_ms = t0.elapsed().as_secs_f64() * 1e3;
        // The same with a linked layer mask applied to the texels (#205).
        cancel(&mut app);
        app.session.execute("layer.layerMask.revealAll", json!({})).unwrap();
        app.session.execute("paint.gradient", json!({"from": [0, 0], "to": [w, 0], "colors": ["#000000", "#ffffff"], "target": "mask"})).unwrap();
        let t0 = std::time::Instant::now();
        begin(&mut app, &ctx).unwrap();
        let masked_ms = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!("transform preview begin {w}x{h}: {begin_ms:.1} ms, with a linked mask {masked_ms:.1} ms");
        let size = app.transform_preview.as_ref().unwrap().texture.size();
        rotate_about_pivot(&mut app, 0.05);
        let frame = |zoom: f32| {
            let t0 = std::time::Instant::now();
            let _ = ctx.run_ui(egui::RawInput { max_texture_side: Some(16384), ..Default::default() }, |ui| {
                let painter = ui.ctx().layer_painter(egui::LayerId::background());
                let xf = ViewXform {
                    rect: egui::Rect::from_min_size(egui::Pos2::ZERO, vec2(1600.0, 1000.0)),
                    zoom,
                    center: [3000.0, 2000.0],
                    flip: false,
                    rotation: 0.0,
                    aspect: 1.0,
                };
                draw_overlay(&app, &painter, &xf);
            });
            t0.elapsed().as_secs_f64() * 1e3
        };
        let first: Vec<f64> = [1.0, 0.25, 3.0].iter().map(|z| frame(*z)).collect();
        let n = 50;
        let steady = (0..n).map(|i| frame([1.0, 0.25, 3.0][i % 3])).sum::<f64>() / n as f64;
        eprintln!(
            "transform preview {w}x{h}: texture {size:?}, begin {begin_ms:.1} ms, first frame per level (1x, 0.25x, 3x) {first:.1?} ms, steady frame {steady:.3} ms"
        );
    }

    #[test]
    fn transform_selection_moves_only_the_outline() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
        app.sync_views();
        app.session.execute("select.rect", json!({"x": 8, "y": 8, "width": 16, "height": 16})).unwrap();
        let px0 = app.session.active().unwrap().doc.layers[0].surface().unwrap().read_region(photocraft_geom::Rect::new(0, 0, 64, 64));
        let ctx = egui::Context::default();
        crate::menus::invoke(&mut app, &ctx, "select.transformSelection", json!({})).unwrap();
        let t = app.ui.transform.as_ref().unwrap();
        assert!(t.selection && t.rect == [8.0, 8.0, 24.0, 24.0]);
        assert!(app.transform_preview.as_ref().unwrap().doc.selection.is_none(), "ants hidden while transforming");
        scale_about_pivot(&mut app, 2.0, 2.0);
        commit(&mut app);
        let st = app.session.active().unwrap();
        let b = st.doc.selection.as_ref().unwrap().content_bounds();
        assert!(b.width().abs_diff(32) <= 2 && b.x0.abs_diff(0) <= 1, "{b:?}");
        assert_eq!(st.doc.layers[0].surface().unwrap().read_region(photocraft_geom::Rect::new(0, 0, 64, 64)), px0);
    }

    #[test]
    fn warp_mode_drags_anchor_with_handles_splits_and_commits() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
        app.sync_views();
        app.session.execute("layer.new.layer", json!({})).unwrap();
        app.session
            .edit("paint", |doc, a| {
                doc.layer_mut(a.unwrap()).unwrap().surface_mut().unwrap().fill_rect(photocraft_geom::Rect::new(8, 8, 32, 32), &[1.0, 0.0, 0.0, 1.0]);
                Ok(())
            })
            .unwrap();
        let ctx = egui::Context::default();
        begin_warp(&mut app, &ctx).unwrap();
        let w = app.ui.transform.as_ref().unwrap().warp.clone().unwrap();
        assert!(w.is_identity());
        // Drag the bottom-right anchor (point 15) by (+10, +6): its two handles follow.
        warp_pointer(&mut app, ToolEvent::Down { x: 32.0, y: 32.0, pressure: 1.0 }, 2.0, egui::Modifiers::NONE);
        warp_pointer(&mut app, ToolEvent::Up { x: 42.0, y: 38.0 }, 2.0, egui::Modifiers::NONE);
        let m = app.ui.transform.as_ref().unwrap().warp.as_ref().unwrap().mesh.clone().unwrap();
        assert_eq!(m.points[15], [42.0, 38.0]);
        let near = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9;
        assert!(near(m.points[14], [24.0 + 10.0, 38.0]), "{:?}", m.points[14]);
        assert!(near(m.points[11], [42.0, 24.0 + 6.0]), "{:?}", m.points[11]);
        split(&mut app, "edit.transform.splitWarpCrosswise", None).unwrap();
        let split_m = app.ui.transform.as_ref().unwrap().warp.as_ref().unwrap().mesh.as_ref().unwrap();
        assert_eq!(split_m.points.last().copied(), Some([42.0, 38.0]), "the corner survives the cut");
        assert!(split_m.us.iter().any(|u| (u - 1.0 / 3.0).abs() < 1e-6) && split_m.us.iter().any(|u| (u - 0.5).abs() < 1e-6));
        commit(&mut app);
        let st = app.session.active().unwrap();
        let b = st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().content_bounds();
        assert!(b.x1 >= 41 && b.y1 >= 37, "{b:?}");
        assert!(app.ui.transform.is_none());
        // Toggling back to the box drops the warp.
        begin_warp(&mut app, &ctx).unwrap();
        leave_warp(&mut app);
        assert!(app.ui.transform.as_ref().unwrap().warp.is_none());
    }

    fn warping_square() -> PhotocraftApp {
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 32, 32));
        begin_warp(&mut app, &egui::Context::default()).unwrap();
        app
    }

    fn warp_mesh(app: &PhotocraftApp) -> BezierMesh {
        app.ui.transform.as_ref().unwrap().warp.as_ref().unwrap().mesh.clone().unwrap()
    }

    #[test]
    fn warp_drag_from_inside_the_grid_pulls_the_surface_under_the_pointer() {
        let mut app = warping_square();
        let before = warp_mesh(&app);
        // (20, 20) is mid-cell, far from every anchor and handle.
        warp_pointer(&mut app, ToolEvent::Down { x: 20.0, y: 20.0, pressure: 1.0 }, 2.0, egui::Modifiers::NONE);
        assert!(matches!(app.transform_preview.as_ref().unwrap().warp_drag, Some(WarpDrag::Surface { .. })));
        warp_pointer(&mut app, ToolEvent::Move { x: 30.0, y: 14.0, pressure: 1.0 }, 2.0, egui::Modifiers::NONE);
        warp_pointer(&mut app, ToolEvent::Up { x: 30.0, y: 14.0 }, 2.0, egui::Modifiers::NONE);
        let after = warp_mesh(&app);
        let (s, t) = surface_at(&before, [20.0, 20.0]).unwrap();
        let moved = after.eval(s, t);
        assert!((moved[0] - 30.0).abs() < 1e-6 && (moved[1] - 14.0).abs() < 1e-6, "{moved:?}");
        assert!(app.transform_preview.as_ref().unwrap().warp_drag.is_none());
        // The picture keeps filling its frame: a pull in the middle leaves the box corners put.
        assert!([0, 3, 12, 15].iter().all(|k| after.points[*k] == before.points[*k]));
        assert!(after.points[5] != before.points[5], "the inside of the grid moved");
    }

    #[test]
    fn a_handle_belongs_to_the_anchor_it_hangs_off() {
        assert_eq!(handle_anchor(0, 1), (0, 0));
        assert_eq!(handle_anchor(0, 2), (0, 3));
        assert_eq!(handle_anchor(1, 0), (0, 0));
        assert_eq!(handle_anchor(5, 3), (6, 3));
        assert_eq!(handle_anchor(3, 3), (3, 3));
    }

    #[test]
    fn reset_warp_returns_to_the_untouched_grid_and_keeps_the_session() {
        let mut app = warping_square();
        warp_pointer(&mut app, ToolEvent::Down { x: 20.0, y: 20.0, pressure: 1.0 }, 2.0, egui::Modifiers::NONE);
        warp_pointer(&mut app, ToolEvent::Up { x: 30.0, y: 14.0 }, 2.0, egui::Modifiers::NONE);
        assert!(!app.ui.transform.as_ref().unwrap().warp.as_ref().unwrap().is_identity());
        reset_warp(&mut app);
        let w = app.ui.transform.as_ref().unwrap().warp.as_ref().unwrap();
        assert!(w.is_identity(), "back to the untouched box");
        assert_eq!(w.mesh.as_ref().map(|m| (m.us.len(), m.vs.len())), Some((2, 2)));
        assert!(app.transform_preview.as_ref().unwrap().warp_drag.is_none());
        // Not warping: nothing to reset.
        leave_warp(&mut app);
        reset_warp(&mut app);
        assert!(app.ui.transform.as_ref().unwrap().warp.is_none());
    }

    #[test]
    fn warp_press_outside_the_grid_grabs_nothing() {
        let mut app = warping_square();
        let before = warp_mesh(&app);
        warp_pointer(&mut app, ToolEvent::Down { x: 55.0, y: 55.0, pressure: 1.0 }, 2.0, egui::Modifiers::NONE);
        assert!(app.transform_preview.as_ref().unwrap().warp_drag.is_none());
        warp_pointer(&mut app, ToolEvent::Up { x: 60.0, y: 60.0 }, 2.0, egui::Modifiers::NONE);
        assert_eq!(warp_mesh(&app), before);
    }

    #[test]
    fn ctrl_click_cuts_the_grid_both_ways() {
        let mut app = warping_square();
        let ctrl = egui::Modifiers::COMMAND;
        warp_pointer(&mut app, ToolEvent::Down { x: 20.0, y: 20.0, pressure: 1.0 }, 2.0, ctrl);
        warp_pointer(&mut app, ToolEvent::Up { x: 20.0, y: 20.0 }, 2.0, ctrl);
        let m = warp_mesh(&app);
        // A cut each way at the pointer (the default grid's rule-of-thirds guides become real
        // lines too, as with the Cross split tool).
        assert!(m.us.iter().any(|u| (u - 0.5).abs() < 1e-6), "vertical cut: {:?}", m.us);
        assert!(m.vs.iter().any(|v| (v - 0.5).abs() < 1e-6), "horizontal cut: {:?}", m.vs);
    }

    #[test]
    fn replacing_the_warp_mesh_cancels_a_pending_point_drag() {
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 32, 32));
        let ctx = egui::Context::default();
        begin_warp(&mut app, &ctx).unwrap();
        for _ in 0..3 {
            split(&mut app, "edit.transform.splitWarpCrosswise", None).unwrap();
        }

        pointer(&mut app, ToolEvent::Down { x: 32.0, y: 32.0, pressure: 1.0 }, egui::Modifiers::NONE);
        assert!(matches!(app.transform_preview.as_ref().unwrap().warp_drag, Some(WarpDrag::Point { .. })), "the anchor drag is armed");

        split(&mut app, "edit.transform.removeWarpSplit", None).unwrap();
        assert!(app.transform_preview.as_ref().unwrap().warp_drag.is_none(), "the old mesh index is discarded");

        // A later control-channel move belongs to the old gesture and must be harmless.
        pointer(&mut app, ToolEvent::Move { x: 40.0, y: 40.0, pressure: 1.0 }, egui::Modifiers::NONE);
    }

    /// A press on a preset warp that misses its points leaves the preset alone (no invisible undo
    /// step); grabbing a point turns it into a custom mesh.
    #[test]
    fn a_press_on_a_preset_warp_converts_it_only_when_it_grabs_the_grid() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
        app.sync_views();
        app.session.execute("layer.new.layer", json!({})).unwrap();
        app.session
            .edit("paint", |doc, a| {
                doc.layer_mut(a.unwrap()).unwrap().surface_mut().unwrap().fill_rect(photocraft_geom::Rect::new(8, 8, 32, 32), &[1.0, 0.0, 0.0, 1.0]);
                Ok(())
            })
            .unwrap();
        begin_warp(&mut app, &egui::Context::default()).unwrap();
        let t = app.ui.transform.as_mut().unwrap();
        let preset = Warp::preset(WarpStyle::Arc, 50.0, t.rect);
        t.warp = Some(preset.clone());
        // Off the grid altogether: a miss.
        warp_pointer(&mut app, ToolEvent::Down { x: 60.0, y: 60.0, pressure: 1.0 }, 2.0, egui::Modifiers::NONE);
        warp_pointer(&mut app, ToolEvent::Up { x: 60.0, y: 60.0 }, 2.0, egui::Modifiers::NONE);
        assert_eq!(app.ui.transform.as_ref().unwrap().warp.as_ref(), Some(&preset), "a miss changes nothing");
        // Inside the grid, away from every point: the surface is grabbed, so the preset converts.
        warp_pointer(&mut app, ToolEvent::Down { x: 20.0, y: 20.0, pressure: 1.0 }, 2.0, egui::Modifiers::NONE);
        assert_eq!(app.ui.transform.as_ref().unwrap().warp.as_ref().unwrap().style, WarpStyle::Custom);
        warp_pointer(&mut app, ToolEvent::Up { x: 20.0, y: 20.0 }, 2.0, egui::Modifiers::NONE);
        app.ui.transform.as_mut().unwrap().warp = Some(preset.clone());
        let corner = preset.to_mesh(1, 1).points[15];
        warp_pointer(&mut app, ToolEvent::Down { x: corner[0], y: corner[1], pressure: 1.0 }, 2.0, egui::Modifiers::NONE);
        assert_eq!(app.ui.transform.as_ref().unwrap().warp.as_ref().unwrap().style, WarpStyle::Custom);
    }

    // ---- Layer masks (#205) ----

    /// `app_with_square` (red over 8..24) plus a reveal-all mask hiding x < 16.
    fn masked_app(linked: bool) -> PhotocraftApp {
        let mut app = app_with_square(64, photocraft_geom::Rect::new(8, 8, 24, 24));
        app.session.execute("layer.layerMask.revealAll", json!({})).unwrap();
        app.session
            .edit("mask", |doc, a| {
                let m = doc.layer_mut(a.unwrap()).unwrap().mask.as_mut().unwrap();
                m.surface.fill_rect(photocraft_geom::Rect::new(0, 0, 16, 64), &[0.0]);
                m.linked = linked;
                Ok(())
            })
            .unwrap();
        app
    }

    fn texel_alpha(img: &egui::ColorImage, x: usize, y: usize) -> u8 {
        img.pixels[y * img.size[0] + x].a()
    }

    #[test]
    fn preview_applies_the_linked_mask() {
        for linked in [true, false] {
            let app = masked_app(linked);
            let st = app.session.active().unwrap();
            let id = st.active_layer.unwrap();
            let b = photocraft_geom::Rect::new(8, 8, 24, 24);
            let (img, _) = preview_image(&st.doc, id, None, b, 4096, false);
            // Texel x = doc x − 8: x 10 is hidden by the mask, x 20 shows.
            assert_eq!(texel_alpha(&img, 12, 4), 255);
            assert_eq!(texel_alpha(&img, 2, 4), if linked { 0 } else { 255 }, "linked={linked}");
        }
        // With a selection the lifted pixels carry the mask too.
        let mut app = masked_app(true);
        app.session.execute("select.rect", json!({"x": 8, "y": 8, "width": 12, "height": 16})).unwrap();
        let ctx = egui::Context::default();
        begin(&mut app, &ctx).unwrap();
        let st = app.session.active().unwrap();
        let sel = st.doc.selection.clone().unwrap();
        let id = st.active_layer.unwrap();
        let (lifted, _) = photocraft_engine::transform_cmds::split_selected(st.doc.layer(id).unwrap().surface().unwrap(), &sel).unwrap();
        let b = photocraft_geom::Rect::new(8, 8, 20, 24);
        let (img, _) = preview_image(&st.doc, id, Some(&lifted), b, 4096, false);
        assert_eq!(texel_alpha(&img, 2, 4), 0, "masked");
        assert_eq!(texel_alpha(&img, 10, 4), 255, "revealed");
    }

    #[test]
    fn preview_feathers_the_linked_mask() {
        // A feathered pixel mask without a vector mask: the compositor softens the edge at x 16,
        // so the moving texels must be soft there too, not hard until the commit (#4560).
        let mut app = masked_app(true);
        app.session
            .edit("feather", |doc, a| {
                doc.layer_mut(a.unwrap()).unwrap().mask.as_mut().unwrap().feather = 4.0;
                Ok(())
            })
            .unwrap();
        let st = app.session.active().unwrap();
        let id = st.active_layer.unwrap();
        let b = photocraft_geom::Rect::new(8, 8, 24, 24);
        let (img, _) = preview_image(&st.doc, id, None, b, 4096, false);
        // Texel x 7 = doc x 15, just inside the hidden half; texel x 8 = doc x 16, just revealed.
        assert!(texel_alpha(&img, 7, 4) > 0, "the feather reaches into the hidden side");
        assert!(texel_alpha(&img, 8, 4) < 255, "the feather reaches into the revealed side");
        // The same values the compositor masks the committed layer with.
        let m = photocraft_compose::masks::combined_mask(st.doc.layer(id).unwrap(), b).unwrap();
        for x in [6usize, 7, 8, 9, 15] {
            let want = (m.sample_channel(8 + x as i32, 12, 0) * 255.0).round() as i32;
            assert!((i32::from(texel_alpha(&img, x, 4)) - want).abs() <= 1, "texel {x}");
        }
    }

    #[test]
    fn targeted_unlinked_mask_transforms_alone_through_the_box() {
        let mut app = masked_app(false);
        app.ui.mask_target = true;
        let ctx = egui::Context::default();
        begin(&mut app, &ctx).unwrap();
        let t = app.ui.transform.as_ref().unwrap();
        assert_eq!(t.target, Some(json!("mask")));
        assert_eq!(t.rect, [0.0, 0.0, 16.0, 64.0], "the box frames the mask's content");
        // The preview document shows the mask vacated (revealed).
        let pd = &app.transform_preview.as_ref().unwrap().doc;
        assert!(pd.layer(app.session.active().unwrap().active_layer.unwrap()).unwrap().mask.as_ref().unwrap().surface.sample_channel(4, 4, 0) > 0.99);
        // Move the box 20 px right and commit.
        let t = app.ui.transform.as_mut().unwrap();
        t.quad = t.quad.map(|[x, y]| [x + 20.0, y]);
        commit(&mut app);
        assert!(!app.ui.status_error, "{}", app.ui.status);
        let st = app.session.active().unwrap();
        let l = st.doc.layer(st.active_layer.unwrap()).unwrap();
        let m = &l.mask.as_ref().unwrap().surface;
        assert!(m.sample_channel(4, 30, 0) > 0.99 && m.sample_channel(30, 30, 0) < 0.01, "mask moved 20 px right");
        assert_eq!(l.surface().unwrap().content_bounds(), photocraft_geom::Rect::new(8, 8, 24, 24), "pixels untouched");
        // Warp mode doesn't apply to a lone mask.
        app.ui.mask_target = true;
        begin(&mut app, &ctx).unwrap();
        enter_warp(&mut app, &egui::Context::default());
        assert!(app.ui.transform.as_ref().unwrap().warp.is_none());
    }

    #[test]
    fn linked_mask_target_transforms_the_layer() {
        let mut app = masked_app(true);
        app.ui.mask_target = true;
        let ctx = egui::Context::default();
        begin(&mut app, &ctx).unwrap();
        assert_eq!(app.ui.transform.as_ref().unwrap().target, None);
        assert_eq!(app.ui.transform.as_ref().unwrap().rect, [8.0, 8.0, 24.0, 24.0]);
    }

    /// ⌘T with Path Selection and a path transforms the path, not the layer's pixels, as one
    /// undoable step; it works on the Background too.
    #[test]
    fn free_transform_moves_the_selected_path_not_pixels() {
        let ctx = egui::Context::default();
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
        app.sync_views();
        app.run("path.set", json!({"name": "work", "path": {"subpaths": [{"closed": true, "knots": [[10, 10], [30, 10], [30, 20], [10, 20]]}]}})).unwrap();
        app.ui.tool = Tool::PathSelection;
        assert!(crate::menus::is_enabled(&app, "edit.freeTransform"), "a path is transformable on the Background");
        let pixels = app.session.active().unwrap().doc.layers[0].surface().cloned();
        let steps = app.session.active().unwrap().history.past_len();
        crate::menus::invoke(&mut app, &ctx, "edit.freeTransform", json!({})).unwrap();
        let t = app.ui.transform.as_mut().unwrap();
        assert_eq!(t.rect, [10.0, 10.0, 30.0, 20.0], "the box frames the path");
        assert_eq!(t.path, Some(json!({"name": "work"})));
        // Twice as wide and 5 px down.
        t.quad = [[10.0, 15.0], [50.0, 15.0], [50.0, 25.0], [10.0, 25.0]];
        commit(&mut app);
        let st = app.session.active().unwrap();
        assert_eq!(st.history.past_len(), steps + 1, "one step");
        let wp = st.doc.work_path.as_ref().unwrap();
        let anchors: Vec<[f64; 2]> = wp.subpaths[0].knots.iter().map(|k| [k.anchor.x, k.anchor.y]).collect();
        assert_eq!(anchors, vec![[10.0, 15.0], [50.0, 15.0], [50.0, 25.0], [10.0, 25.0]]);
        assert_eq!(st.doc.layers[0].surface().cloned(), pixels, "the pixels stay put");

        // ⌘-drag of a corner (Distort) maps the path onto the box (#3027): one step.
        crate::menus::invoke(&mut app, &ctx, "edit.freeTransform", json!({})).unwrap();
        app.ui.transform.as_mut().unwrap().quad[2] = [70.0, 40.0];
        commit(&mut app);
        let st = app.session.active().unwrap();
        assert_eq!(st.history.past_len(), steps + 2, "one step: {}", app.ui.status);
        let anchors: Vec<[f64; 2]> = st.doc.work_path.as_ref().unwrap().subpaths[0].knots.iter().map(|k| [k.anchor.x, k.anchor.y]).collect();
        let want = [[10.0, 15.0], [50.0, 15.0], [70.0, 40.0], [10.0, 25.0]];
        assert!(anchors.iter().zip(want).all(|(a, b)| (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9), "{anchors:?}");

        // Edit › Transform Path › Warp bends it: one more step.
        crate::menus::invoke(&mut app, &ctx, "edit.transform.warp", json!({})).unwrap();
        let t = app.ui.transform.as_mut().unwrap();
        assert!(t.warp.is_some(), "Warp mode on a path");
        t.warp = Some(Warp { style: WarpStyle::Arc, bend: 50.0, ..Warp::none(t.rect) });
        commit(&mut app);
        let st = app.session.active().unwrap();
        assert_eq!(st.history.past_len(), steps + 3, "one step: {}", app.ui.status);
        assert!(st.doc.work_path.as_ref().unwrap().subpaths[0].knots.len() > 4, "the warp bends the edges");

        // Without a path tool, ⌘T is the pixel transform again.
        app.ui.tool = Tool::Move;
        app.run("layer.new.layer", json!({})).unwrap();
        assert!(crate::vector_ui::free_transform_path(&app).is_none());
    }

    /// Edit › Transform › Warp on editable type (#3004), as in Photoshop: no rasterize prompt, no
    /// mesh to drag, only the warp styles; the commit is the type's own Warp Text, one step, and
    /// the layer stays type.
    #[test]
    fn warp_on_type_takes_a_style_and_keeps_the_type_editable() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.execute("file.new", json!({"width": 200, "height": 100})).unwrap();
        app.sync_views();
        app.run("type.create", json!({"x": 10, "y": 60, "text": "Hello"})).unwrap();
        let ctx = egui::Context::default();
        let past = |app: &PhotocraftApp| app.session.active().unwrap().history.past_len();
        let before = past(&app);
        crate::menus::invoke(&mut app, &ctx, "edit.transform.warp", json!({})).unwrap();
        assert!(app.ui.dialogs.is_empty(), "no prompt");
        let w0 = app.ui.transform.as_ref().and_then(|t| t.warp.clone()).expect("Warp mode");
        assert_eq!(w0.style, WarpStyle::None);
        // A drag on the grid changes nothing: type takes only the styles.
        let c = [(w0.bounds[0] + w0.bounds[2]) / 2.0, (w0.bounds[1] + w0.bounds[3]) / 2.0];
        warp_pointer(&mut app, ToolEvent::Down { x: w0.bounds[2], y: w0.bounds[3], pressure: 1.0 }, 4.0, egui::Modifiers::NONE);
        warp_pointer(&mut app, ToolEvent::Up { x: w0.bounds[2] + 20.0, y: w0.bounds[3] + 15.0 }, 4.0, egui::Modifiers::NONE);
        warp_pointer(&mut app, ToolEvent::Down { x: c[0], y: c[1], pressure: 1.0 }, 4.0, egui::Modifiers::NONE);
        warp_pointer(&mut app, ToolEvent::Up { x: c[0], y: c[1] + 20.0 }, 4.0, egui::Modifiers::NONE);
        assert_eq!(app.ui.transform.as_ref().unwrap().warp.as_ref(), Some(&w0));
        assert!(split(&mut app, "edit.transform.splitWarpCrosswise", None).is_err());
        // Pick Flag in the options bar, then Enter.
        if let Some(w) = app.ui.transform.as_mut().and_then(|t| t.warp.as_mut()) {
            *w = Warp { style: WarpStyle::Flag, bend: 40.0, ..Warp::none(w.bounds) };
        }
        commit(&mut app);
        assert!(!app.ui.status_error, "{}", app.ui.status);
        let st = app.session.active().unwrap();
        let LayerContent::Text(t) = &st.doc.layer(st.active_layer.unwrap()).unwrap().content else { panic!("still type") };
        let tw = t.warp.as_ref().expect("Warp Text set");
        assert_eq!((tw.style.as_str(), tw.value, tw.horizontal), ("warpFlag", 40.0, true));
        assert_eq!(past(&app), before + 1, "one history step");
        // Warp again shows Flag; None takes it off in one more step.
        begin_warp(&mut app, &ctx).unwrap();
        assert_eq!(app.ui.transform.as_ref().unwrap().warp.as_ref().unwrap().style, WarpStyle::Flag);
        if let Some(t) = app.ui.transform.as_mut() {
            t.warp = Some(Warp::none(t.rect));
        }
        commit(&mut app);
        let st = app.session.active().unwrap();
        assert!(matches!(&st.doc.layer(st.active_layer.unwrap()).unwrap().content, LayerContent::Text(t) if t.warp.is_none()));
        assert_eq!(past(&app), before + 2);
    }
}
