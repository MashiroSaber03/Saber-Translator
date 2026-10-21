//! Multi-layer selection and the commands that act on it: ⌘/⇧-click selection, Select › All /
//! Deselect / Find Layers, Layer › Align and Distribute, Link Layers, Merge Layers, Group from
//! Layers, Arrange › Reverse, Lock Layers and Rename Layer.
//!
//! The selection itself lives in [`DocState::selected_layers`] (UI state, like the active layer):
//! changing it never adds a history step. Every structural change below is one history step.

use photocraft_color::PixelFormat;
use photocraft_doc::{Document, Layer, LayerContent, LayerId};
use photocraft_geom::Rect;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, int_i32};
use crate::{DocState, EngineError, Result, Session};

// ---------- selection state ----------

fn doc_state(s: &Session) -> std::result::Result<&DocState, String> {
    s.active().ok_or_else(|| "no document open".into())
}

/// Selected layers of the active document, bottom-to-top.
pub fn selected(s: &Session) -> Vec<LayerId> {
    s.active().map(DocState::selected_layers).unwrap_or_default()
}

/// The layers a layer command acts on: the `layer` param when given, else every selected layer
/// (bottom-to-top), else the active layer. A single layer behaves exactly as before.
pub(crate) fn targets(s: &Session, p: &serde_json::Value) -> Result<Vec<LayerId>> {
    if p.get("layer").is_some() {
        return Ok(vec![crate::commands::layer_param(s, p)?]);
    }
    match selected(s) {
        sel if sel.is_empty() => Ok(vec![crate::commands::layer_param(s, p)?]),
        sel => Ok(sel),
    }
}

/// [`targets`] for a command that adds a layer mask: with several layers selected, those that can
/// take one (no layer mask yet, not the Background); one layer is taken as given.
pub(crate) fn mask_targets(s: &Session, p: &serde_json::Value) -> Result<Vec<LayerId>> {
    let ids = targets(s, p)?;
    if ids.len() < 2 {
        return Ok(ids);
    }
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let ids: Vec<LayerId> = ids.into_iter().filter(|id| d.doc.layer(*id).is_some_and(|l| l.mask.is_none() && !d.doc.is_background(l.id))).collect();
    if ids.is_empty() {
        return Err(EngineError::Other("none of the selected layers can take a layer mask".into()));
    }
    Ok(ids)
}

/// Ids in `ids` that have no ancestor also in `ids` (a selected group already carries its
/// selected children), bottom-to-top.
pub(crate) fn top_level(doc: &Document, ids: &[LayerId]) -> Vec<LayerId> {
    let walk = doc.walk();
    let paths: Vec<_> = walk.iter().filter(|(_, _, l)| ids.contains(&l.id)).map(|(p, _, l)| (p.clone(), l.id)).collect();
    paths.iter().filter(|(p, _)| !paths.iter().any(|(q, _)| q.len() < p.len() && p.starts_with(q))).map(|(_, id)| *id).collect()
}

/// Replace the layer selection without a history step (like [`Session::select_layer`]).
pub fn set_selection(s: &mut Session, ids: Vec<LayerId>, active: Option<LayerId>, anchor: Option<LayerId>) -> Result<()> {
    let st = s.active_mut().ok_or(EngineError::NoDocument)?;
    if let Some(bad) = ids.iter().chain(active.iter()).find(|id| st.doc.layer(**id).is_none()) {
        return Err(EngineError::NoLayer(*bad));
    }
    let active = active.or_else(|| ids.last().copied());
    st.selected_layers = ids;
    if let Some(a) = active
        && !st.selected_layers.contains(&a)
    {
        st.selected_layers.push(a);
    }
    st.active_layer = active;
    st.layer_anchor = anchor.or(active);
    crate::fix_selection(st);
    // Selecting layers is not an edit: keep a clean document clean.
    let clean = st.saved_revision == st.revision;
    st.revision += 1;
    st.view_revision += 1;
    if clean {
        st.saved_revision = st.revision;
    }
    st.last_damage = Some(Rect::EMPTY);
    Ok(())
}

/// After a structural edit, make `ids` the selection (dropping any that no longer exist), which
/// is also what the edit's history state targets.
pub(crate) fn reselect(s: &mut Session, ids: Vec<LayerId>, active: Option<LayerId>) {
    if let Some(st) = s.active_mut() {
        st.selected_layers = ids;
        if let Some(a) = active {
            st.active_layer = Some(a);
            st.layer_anchor = Some(a);
        }
        crate::fix_selection(st);
        let layers = st.layer_target();
        st.history.set_current_layers(layers);
    }
}

/// Layer rows follow the Layers panel's visual order, not Document::walk's storage order.
/// Closed groups hide their children unless both endpoints were selected programmatically.
fn range_order(layers: &[Layer], include_collapsed: bool, out: &mut Vec<LayerId>) {
    for layer in layers.iter().rev() {
        out.push(layer.id);
        if let LayerContent::Group(group) = &layer.content
            && (group.expanded || include_collapsed)
        {
            range_order(&group.children, include_collapsed, out);
        }
    }
}

/// `layer.select`: `{"layer": id, "mode": "replace|toggle|range|add|remove"}`.
pub fn select(s: &mut Session, p: &Value) -> Result<Value> {
    let id = crate::commands::explicit_layer(s, p)?.ok_or_else(|| bad("layer.select", "missing `layer`"))?;
    let mode = p.get("mode").and_then(Value::as_str).unwrap_or("replace");
    let st = s.active().ok_or(EngineError::NoDocument)?;
    if st.doc.layer(id).is_none() {
        return Err(EngineError::NoLayer(id));
    }
    let mut sel = st.selected_layers();
    match mode {
        "replace" => s.select_layer(id)?,
        "add" => {
            if !sel.contains(&id) {
                sel.push(id);
            }
            set_selection(s, sel, Some(id), Some(id))?;
        }
        "toggle" => {
            if sel.contains(&id) {
                if sel.len() == 1 {
                    // Photoshop keeps at least the clicked layer targeted.
                    return Ok(json!({"selected": [id.0]}));
                }
                sel.retain(|x| *x != id);
                let active = if st.active_layer == Some(id) { sel.last().copied() } else { st.active_layer };
                set_selection(s, sel, active, active)?;
            } else {
                sel.push(id);
                set_selection(s, sel, Some(id), Some(id))?;
            }
        }
        // Toggle's removing half: a layer that isn't selected stays unselected (Photoshop's
        // `removeFromSelection`, recorded by actions).
        "remove" => {
            if sel.contains(&id) && sel.len() > 1 {
                sel.retain(|x| *x != id);
                let active = if st.active_layer == Some(id) { sel.last().copied() } else { st.active_layer };
                set_selection(s, sel, active, active)?;
            }
        }
        "range" => {
            let anchor = st.layer_anchor.filter(|a| st.doc.layer(*a).is_some()).or(st.active_layer).unwrap_or(id);
            let mut order = Vec::new();
            range_order(&st.doc.layers, false, &mut order);
            // Engine clients may target a child of a collapsed group that the panel cannot
            // click. Retain that capability by using expanded visual order for such calls.
            if !order.contains(&anchor) || !order.contains(&id) {
                order.clear();
                range_order(&st.doc.layers, true, &mut order);
            }
            let (a, b) = (order.iter().position(|x| *x == anchor).unwrap_or(0), order.iter().position(|x| *x == id).unwrap_or(0));
            let range = order[a.min(b)..=a.max(b)].to_vec();
            set_selection(s, range, Some(id), Some(anchor))?;
        }
        other => return Err(bad("layer.select", format!("unknown mode `{other}`"))),
    }
    Ok(json!({"selected": selected(s).iter().map(|l| l.0).collect::<Vec<_>>()}))
}

fn select_all_layers(s: &mut Session) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let ids: Vec<LayerId> = d.doc.walk().into_iter().filter(|(_, _, l)| !d.doc.is_background(l.id)).map(|(_, _, l)| l.id).collect();
    if ids.is_empty() {
        return Err(EngineError::Other("there are no layers to select".into()));
    }
    let active = d.active_layer.filter(|a| ids.contains(a)).or(ids.last().copied());
    let n = ids.len();
    set_selection(s, ids, active, None)?;
    Ok(json!({"selected": n}))
}

fn find_layers(s: &mut Session, p: &Value) -> Result<Value> {
    let needle = p.get("name").and_then(Value::as_str).ok_or_else(|| bad("select.findLayers", "missing `name`"))?.to_lowercase();
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let ids: Vec<LayerId> = d.doc.walk().into_iter().filter(|(_, _, l)| l.name.to_lowercase().contains(&needle)).map(|(_, _, l)| l.id).collect();
    if ids.is_empty() {
        return Err(EngineError::Other(format!("no layer name contains \"{needle}\"")));
    }
    let out = ids.iter().map(|l| l.0).collect::<Vec<_>>();
    set_selection(s, ids, None, None)?;
    Ok(json!({"selected": out}))
}

fn select_linked(s: &mut Session) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let groups = link_groups(&d.doc, &d.selected_layers());
    let ids: Vec<LayerId> = d.doc.walk().into_iter().filter(|(_, _, l)| l.link_group.is_some_and(|g| groups.contains(&g))).map(|(_, _, l)| l.id).collect();
    let active = d.active_layer.filter(|a| ids.contains(a)).or(ids.last().copied());
    let n = ids.len();
    set_selection(s, ids, active, active)?;
    Ok(json!({"selected": n}))
}

// ---------- enabled predicates ----------

fn has_layer(s: &Session) -> std::result::Result<(), String> {
    if selected(s).is_empty() { Err("no layer selected".into()) } else { Ok(()) }
}
fn has_doc(s: &Session) -> std::result::Result<(), String> {
    doc_state(s).map(|_| ())
}
fn two_plus(s: &Session) -> std::result::Result<(), String> {
    if selected(s).len() >= 2 { Ok(()) } else { Err("select two or more layers".into()) }
}
fn can_align(s: &Session) -> std::result::Result<(), String> {
    let d = doc_state(s)?;
    match d.selected_layers().len() {
        0 => Err("no layer selected".into()),
        // Photoshop aligns a single layer to the canvas (or to an active selection); 2+ align to each other.
        _ => Ok(()),
    }
}
fn can_distribute(s: &Session) -> std::result::Result<(), String> {
    if selected(s).len() >= 3 { Ok(()) } else { Err("select three or more layers to distribute".into()) }
}
fn can_link(s: &Session) -> std::result::Result<(), String> {
    let d = doc_state(s)?;
    let sel = d.selected_layers();
    match sel.len() {
        0 => Err("no layer selected".into()),
        1 if d.doc.layer(sel[0]).is_none_or(|l| l.link_group.is_none()) => Err("select two or more layers to link".into()),
        _ => Ok(()),
    }
}
fn has_linked(s: &Session) -> std::result::Result<(), String> {
    let d = doc_state(s)?;
    if link_groups(&d.doc, &d.selected_layers()).is_empty() { Err("the selected layers are not linked".into()) } else { Ok(()) }
}

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

// ---------- moving layers ----------

fn link_groups(doc: &Document, ids: &[LayerId]) -> Vec<u64> {
    let mut g: Vec<u64> = ids.iter().filter_map(|id| doc.layer(*id)?.link_group).collect();
    g.sort_unstable();
    g.dedup();
    g
}

/// `ids` plus every layer linked to one of them.
pub(crate) fn with_links(doc: &Document, ids: &[LayerId]) -> Vec<LayerId> {
    let groups = link_groups(doc, ids);
    let mut out = ids.to_vec();
    if !groups.is_empty() {
        for (_, _, l) in doc.walk() {
            if l.link_group.is_some_and(|g| groups.contains(&g)) && !out.contains(&l.id) {
                out.push(l.id);
            }
        }
    }
    out
}

/// Move layers by whole pixels the way the Move tool does: pixels, linked masks, type and
/// vector content (shapes, linked vector masks) all move together. `before` is `doc` as it was
/// before any of the moves (the edit's starting document), read when type and shapes re-render.
pub(crate) fn move_layers(doc: &mut Document, before: &Document, moves: &[(LayerId, i32, i32)]) -> Result<()> {
    for &(id, dx, dy) in moves {
        if dx == 0 && dy == 0 {
            continue;
        }
        let locks = doc.effective_locks(id);
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        if locks.position || locks.all {
            return Err(EngineError::Other(format!("layer \"{}\" is position-locked", l.name)));
        }
        crate::commands::translate_layer(before, l, dx, dy);
        crate::vector_cmds::translate_vectors(before, l, dx as f64, dy as f64);
    }
    Ok(())
}

/// Does a move aimed at the layer mask (`"target":"mask"`) move `l`'s mask alone? Only an
/// unlinked mask does (the chain off); a linked one moves with its layer, as in Photoshop.
pub fn moves_mask_alone(l: &Layer) -> bool {
    l.mask.as_ref().is_some_and(|m| !m.linked)
}

/// `layer.translate`: the explicit layer, or every selected layer, plus their linked layers.
/// With `"target":"mask"` a layer whose mask is unlinked moves that mask alone.
pub fn translate(s: &mut Session, p: &Value) -> Result<Value> {
    let (dx, dy) = match p.get("units").and_then(Value::as_str) {
        None | Some("px") => (int_i32("layer.translate", p, "dx")?.unwrap_or(0), int_i32("layer.translate", p, "dy")?.unwrap_or(0)),
        // Points (1/72 in), as a Photoshop action records a nudge: whole pixels at this
        // document's resolution.
        Some("pt") => {
            let dpi = s.active().map_or(72.0, |d| f64::from(d.doc.resolution_dpi));
            let to_px = |key: &str| -> Result<i32> {
                let pt = match p.get(key) {
                    None => 0.0,
                    Some(v) => v.as_f64().filter(|f| f.is_finite()).ok_or_else(|| bad("layer.translate", format!("`{key}` must be a number")))?,
                };
                let px = (pt * dpi / 72.0).round();
                if px.abs() > f64::from(i32::MAX) {
                    return Err(bad("layer.translate", format!("`{key}` = {pt} pt is outside the 32-bit coordinate range")));
                }
                Ok(px as i32)
            };
            (to_px("dx")?, to_px("dy")?)
        }
        Some(other) => return Err(bad("layer.translate", format!("unknown units `{other}` (px|pt)"))),
    };
    if dx == 0 && dy == 0 {
        return Ok(Value::Null);
    }
    let roots = match crate::commands::explicit_layer(s, p)? {
        Some(id) => vec![id],
        None => selected(s),
    };
    if roots.is_empty() {
        return Err(EngineError::Other("no active layer".into()));
    }
    let mask_target = p.get("target").and_then(Value::as_str) == Some("mask");
    let before = s.active().ok_or(EngineError::NoDocument)?.doc.clone();
    let ids = s.edit("Move", |doc, _| {
        let (alone, rest): (Vec<LayerId>, Vec<LayerId>) = roots.iter().partition(|id| mask_target && doc.layer(**id).is_some_and(moves_mask_alone));
        let mut ids = move_targets(doc, &rest);
        if ids.is_empty() && alone.is_empty() {
            return Err(EngineError::NoLayer(roots[0]));
        }
        let moves: Vec<_> = ids.iter().map(|&id| (id, dx, dy)).collect();
        move_layers(doc, &before, &moves)?;
        for id in alone {
            let locks = doc.effective_locks(id);
            let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
            if locks.position || locks.all {
                return Err(EngineError::Other(format!("layer \"{}\" is position-locked", l.name)));
            }
            if let Some(m) = &mut l.mask {
                m.surface = shift_surface(&m.surface, dx, dy);
            }
            ids.push(id);
        }
        // A board moved past any canvas edge grows the canvas, as a new one does (#1531, #3070).
        if ids.iter().any(|id| doc.layer(*id).is_some_and(|l| l.artboard().is_some())) {
            crate::artboard_cmds::fit_canvas(doc);
        }
        Ok(ids)
    })?;
    note_damage(s, &before, &ids);
    Ok(Value::Null)
}

/// The layers a move of `roots` takes along: their linked layers, without layers whose group
/// is also moving (the group carries them).
pub fn move_targets(doc: &Document, roots: &[LayerId]) -> Vec<LayerId> {
    top_level(doc, &with_links(doc, roots))
}

/// `doc` with the layers `ids` ([`move_targets`]) moved by whole pixels, as `layer.translate`
/// leaves them: the Move tool's live preview while it drags. Type, shape and smart-object pixels
/// shift as they are instead of re-rendering (the commit re-renders them), so a drag frame costs
/// copying the moving pixels only.
pub fn moved(doc: &Document, ids: &[LayerId], dx: i32, dy: i32) -> Result<Document> {
    moved_target(doc, ids, dx, dy, false)
}

/// [`moved`] for a drag aimed at the layer mask (`mask_target`): unlinked masks move alone.
pub fn moved_target(doc: &Document, ids: &[LayerId], dx: i32, dy: i32, mask_target: bool) -> Result<Document> {
    let mut out = doc.clone();
    for &id in ids {
        let locks = doc.effective_locks(id);
        let l = out.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        if locks.position || locks.all {
            return Err(EngineError::Other(format!("layer \"{}\" is position-locked", l.name)));
        }
        if mask_target && moves_mask_alone(l) {
            if let Some(m) = &mut l.mask {
                m.surface = shift_surface(&m.surface, dx, dy);
            }
            continue;
        }
        shift_shown(doc, l, dx, dy);
    }
    Ok(out)
}

/// `photocraft_algo::resample::translate_surface` with the content scan cached per tile (a Move
/// drag shifts the same layers every frame; scanning them each time cost more than the copy).
pub(crate) fn shift_surface(s: &photocraft_raster::Surface, dx: i32, dy: i32) -> photocraft_raster::Surface {
    s.translated(dx, dy, photocraft_compose::bounds::content_bounds(s))
}

/// [`crate::commands::translate_layer`] plus the vector side, without re-rendering anything.
fn shift_shown(doc: &Document, l: &mut Layer, dx: i32, dy: i32) {
    use self::shift_surface as translate_surface;
    let a = photocraft_geom::Affine::translate(f64::from(dx), f64::from(dy));
    if let Some(r) = &mut l.effects.reference {
        *r = (r.0 + f64::from(dx), r.1 + f64::from(dy));
    }
    if let Some(m) = l.mask.as_mut().filter(|m| m.linked) {
        m.surface = translate_surface(&m.surface, dx, dy);
    }
    if let Some(vm) = l.vector_mask.as_mut().filter(|v| v.linked) {
        vm.path = vm.path.transform(&a);
    }
    match &mut l.content {
        LayerContent::Raster(s) => *s = translate_surface(s, dx, dy),
        LayerContent::Text(t) => {
            t.transform = a.mul(&t.transform);
            if let Some(c) = &mut t.cache {
                *c = translate_surface(c, dx, dy);
            }
        }
        LayerContent::Shape(sh) => {
            // The rendered shape is cut at the canvas: one that reaches past it renders again.
            let canvas = doc.bounds();
            let was_inside = sh.cache.as_ref().is_some_and(|c| {
                let b = photocraft_compose::bounds::content_bounds(c);
                b.is_empty() || (b.x0 > canvas.x0 && b.y0 > canvas.y0 && b.x1 < canvas.x1 && b.y1 < canvas.y1)
            });
            crate::vector_cmds::transform_shape(sh, &a);
            let stroke_pad = sh.stroke.as_ref().filter(|s| s.opacity > 0.0).map_or(0.0, |s| f64::from(s.width.ceil()));
            let stays_inside = sh.path.control_bounds().is_some_and(|(x0, y0, x1, y1)| {
                (x0 - stroke_pad) > canvas.x0 as f64
                    && (y0 - stroke_pad) > canvas.y0 as f64
                    && (x1 + stroke_pad) < canvas.x1 as f64
                    && (y1 + stroke_pad) < canvas.y1 as f64
            });
            match &mut sh.cache {
                Some(c) if was_inside && stays_inside => *c = translate_surface(c, dx, dy),
                _ => crate::vector_cmds::refresh_shape(doc, sh),
            }
        }
        LayerContent::Smart(sm) => crate::smart_cmds::shift_smart(sm, dx, dy),
        LayerContent::Group(g) => {
            if let Some(ab) = &mut g.artboard {
                ab.rect = ab.rect.translate(dx, dy);
            }
            g.children.iter_mut().for_each(|c| shift_shown(doc, c, dx, dy));
        }
        LayerContent::Adjustment(_) | LayerContent::Fill(_) => {}
    }
}

/// The document pixels an edit of the layers `ids` (moved, shown, hidden or restyled; contents
/// otherwise as they were) can have changed from `before` to `after`, or `None` for anywhere.
pub fn layers_damage(before: &Document, after: &Document, ids: &[LayerId]) -> Option<Rect> {
    let mut out = Rect::EMPTY;
    for d in [before, after] {
        let canvas = d.bounds();
        for id in ids {
            // A layer gone from one side (never for a move) changes who knows what.
            let b = photocraft_compose::change_bounds(d.layer(*id)?, canvas)?;
            if !b.is_empty() {
                out = if out.is_empty() { b } else { out.union(&b) };
            }
        }
    }
    Some(out)
}

/// Report [`layers_damage`] as the active document's last change, so the canvas recomposites
/// only that (grown by the effect reach around it) instead of everything.
pub(crate) fn note_damage(s: &mut Session, before: &Document, ids: &[LayerId]) {
    let Some(st) = s.active_mut() else { return };
    if before.size != st.doc.size {
        return;
    }
    st.last_damage = layers_damage(before, &st.doc, ids);
}

/// Whether two surfaces hold identical pixels: same format and default pixel, and the same
/// copy-on-write tile pointers at the same coordinates. Both documents stay alive during the
/// comparison, so tile addresses cannot be reused under it.
fn surfaces_equal(a: &photocraft_raster::Surface, b: &photocraft_raster::Surface) -> bool {
    a.format() == b.format()
        && a.default_pixel() == b.default_pixel()
        && a.tile_count() == b.tile_count()
        && a.tiles().zip(b.tiles()).all(|((ca, ta), (cb, tb))| ca == cb && std::sync::Arc::ptr_eq(ta, tb))
}

/// Byte blobs kept in `Arc`s (PSD raw blocks, embedded smart-object sources): pointer-identical
/// or content-equal (they are small compared to pixels, and undo steps rarely re-parse them).
fn blob_equal(x: &std::sync::Arc<Vec<u8>>, y: &std::sync::Arc<Vec<u8>>) -> bool {
    std::sync::Arc::ptr_eq(x, y) || (x.len() == y.len() && x.as_slice() == y.as_slice())
}

fn blobs_equal(a: &Option<std::sync::Arc<Vec<u8>>>, b: &Option<std::sync::Arc<Vec<u8>>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => blob_equal(x, y),
        _ => false,
    }
}

/// Layer masks (also the smart-filter mask): scalars plus tile-pointer surface identity.
fn masks_equal(a: &Option<photocraft_doc::LayerMask>, b: &Option<photocraft_doc::LayerMask>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => {
            x.enabled == y.enabled && x.linked == y.linked && x.density == y.density && x.feather == y.feather && surfaces_equal(&x.surface, &y.surface)
        }
        _ => false,
    }
}

fn texts_equal(a: &photocraft_doc::TextLayer, b: &photocraft_doc::TextLayer) -> bool {
    a.text == b.text
        && a.font_family == b.font_family
        && a.size_pt == b.size_pt
        && a.color == b.color
        && a.transform == b.transform
        && a.runs == b.runs
        && a.paragraphs == b.paragraphs
        && a.shape == b.shape
        && a.orientation == b.orientation
        && a.antialias == b.antialias
        && a.warp == b.warp
        && match (&a.cache, &b.cache) {
            (None, None) => true,
            (Some(x), Some(y)) => surfaces_equal(x, y),
            _ => false,
        }
        && blobs_equal(&a.psd_raw, &b.psd_raw)
}

fn smarts_equal(a: &photocraft_doc::SmartObject, b: &photocraft_doc::SmartObject) -> bool {
    a.filters_enabled == b.filters_enabled
        && a.transform == b.transform
        && a.smart_filters == b.smart_filters
        && a.stack_mode == b.stack_mode
        && a.perspective == b.perspective
        && a.warp == b.warp
        && match (&a.source, &b.source) {
            (photocraft_doc::SmartSource::Embedded { file_name: x, bytes: xb }, photocraft_doc::SmartSource::Embedded { file_name: y, bytes: yb }) => {
                x == y && blob_equal(xb, yb)
            }
            (photocraft_doc::SmartSource::Linked { path: x }, photocraft_doc::SmartSource::Linked { path: y }) => x == y,
            _ => false,
        }
        && match (&a.cache, &b.cache) {
            (None, None) => true,
            (Some(x), Some(y)) => surfaces_equal(x, y),
            _ => false,
        }
        && blobs_equal(&a.psd_raw, &b.psd_raw)
        && masks_equal(&a.filter_mask, &b.filter_mask)
}

/// Extra alpha/spot channels (the image itself in Multichannel mode).
fn alpha_channels_equal(a: &photocraft_doc::AlphaChannel, b: &photocraft_doc::AlphaChannel) -> bool {
    a.name == b.name && a.spot == b.spot && a.color == b.color && a.opacity == b.opacity && a.indicates == b.indicates && surfaces_equal(&a.surface, &b.surface)
}

/// Whether two layers render to identical pixels in place. Everything the compositor reads is
/// compared structurally — no hashing, a hash collision would under-report damage; pixels are
/// identified by their copy-on-write tile pointers. Group children are deliberately *not*
/// compared here: the caller walks into them, so an edit inside a group damages only the
/// edited child's extent, not the whole group's. The destructure is exhaustive, so a new
/// `Layer` field fails to compile until it is classified: compared, or explicitly ignored
/// (name, locks, label, `psd_blocks`, `psd_id`, link group — none of them composite; video
/// bails out earlier in the walk).
fn layer_pixels_equal(a: &Layer, b: &Layer) -> bool {
    let Layer {
        id,
        name: _,
        visible,
        locks: _,
        blend,
        opacity,
        fill_opacity,
        clipped,
        mask,
        vector_mask,
        effects,
        label: _,
        content,
        psd_blocks: _,
        psd_id: _,
        fill_cache,
        link_group: _,
        excluded_channels,
        blend_if,
        advanced,
        video: _,
    } = a;
    *id == b.id
        && *visible == b.visible
        && *blend == b.blend
        && *opacity == b.opacity
        && *fill_opacity == b.fill_opacity
        && *clipped == b.clipped
        && masks_equal(mask, &b.mask)
        && *vector_mask == b.vector_mask
        && *effects == b.effects
        && *excluded_channels == b.excluded_channels
        && *blend_if == b.blend_if
        && *advanced == b.advanced
        && match (content, &b.content) {
            (LayerContent::Raster(x), LayerContent::Raster(y)) => surfaces_equal(x, y),
            // Children are walked separately; only the group's own clipping frame counts.
            (LayerContent::Group(x), LayerContent::Group(y)) => x.artboard == y.artboard,
            (LayerContent::Fill(x), LayerContent::Fill(y)) => x == y,
            (LayerContent::Adjustment(x), LayerContent::Adjustment(y)) => x == y,
            (LayerContent::Text(x), LayerContent::Text(y)) => texts_equal(x, y),
            (LayerContent::Shape(x), LayerContent::Shape(y)) => x == y,
            (LayerContent::Smart(x), LayerContent::Smart(y)) => smarts_equal(x, y),
            _ => false,
        }
        && match (fill_cache, &b.fill_cache) {
            (None, None) => true,
            (Some(x), Some(y)) => x.fill == y.fill && surfaces_equal(&x.surface, &y.surface),
            _ => false,
        }
}

/// Document-level fields the compositor reads: all must match for a bounded damage claim. The
/// destructure is exhaustive, so a new `Document` field fails to compile until it is classified
/// here — compared when it composites, or explicitly ignored (the pixel-irrelevant allowlist:
/// identity, name, selection, guides, paths, layer comps, annotations, automation state).
fn doc_pixels_equal(a: &Document, b: &Document) -> bool {
    let Document {
        id: _,
        name: _,
        size,
        resolution_dpi,
        mode,
        depth,
        icc_profile,
        layers: _,
        channels,
        guides: _,
        ruler_origin: _,
        selection: _,
        metadata: _,
        global_light,
        paths: _,
        work_path: _,
        clipping_path: _,
        quick_mask,
        quick_mask_unselected: _,
        patterns,
        color_table,
        duotone,
        layer_comps: _,
        last_applied_comp: _,
        last_document_state: _,
        measurement: _,
        notes: _,
        text_styles: _,
        slices: _,
        variables: _,
        timeline: _,
        file_encoding: _,
    } = a;
    *size == b.size
        && resolution_dpi.to_bits() == b.resolution_dpi.to_bits()
        && *mode == b.mode
        && *depth == b.depth
        && icc_profile.as_deref() == b.icc_profile.as_deref()
        && channels.len() == b.channels.len()
        && channels.iter().zip(&b.channels).all(|(x, y)| alpha_channels_equal(x, y))
        && *global_light == b.global_light
        && match (quick_mask, &b.quick_mask) {
            (None, None) => true,
            (Some(x), Some(y)) => alpha_channels_equal(x, y),
            _ => false,
        }
        && *patterns == b.patterns
        && *color_table == b.color_table
        && *duotone == b.duotone
}

/// Whether `a` and `b` composite to the same pixels: they differ at most in fields the
/// compositor never reads (selection, guides, paths, names, …), by the same classification as
/// [`step_damage`] but without computing any bounds. Cheap: pixels compare by tile pointer.
/// Conservative (`false`) past a nesting depth no real document reaches.
pub fn same_pixels(a: &Document, b: &Document) -> bool {
    fn walk(xs: &[Layer], ys: &[Layer], depth: usize) -> bool {
        depth < 256
            && xs.len() == ys.len()
            && xs.iter().zip(ys).all(|(x, y)| {
                x.video.is_none()
                    && y.video.is_none()
                    && layer_pixels_equal(x, y)
                    && match (x.children(), y.children()) {
                        (Some(xc), Some(yc)) => walk(xc, yc, depth + 1),
                        (None, None) => true,
                        _ => false,
                    }
            })
    }
    doc_pixels_equal(a, b) && walk(&a.layers, &b.layers, 0)
}

/// Pixels a single history step can have changed between `a` and `b` (the documents on either
/// side of an edit, its undo or its redo): the union of the change bounds of every layer whose
/// pixels differ, so a local edit recomposites only that area instead of the whole canvas. A
/// layer only one side has (a duplicate, a paste, a deleted layer) counts where it draws (see
/// [`one_sided_bounds`]); one that draws nothing ([`inert_at`]) changes nothing. `None`
/// (composite everything) when the step touched anything the layer walk cannot bound:
/// document-level fields the compositor reads (canvas size, mode, depth, profile, global light,
/// patterns, channels, palettes), layer order and kind changes, layers added or removed that
/// reach outside their own bounds, layer properties that do (clipping, excluded channels —
/// what `layer.setProps` already reports as unbounded), and video layers. Inserting a copy of a
/// 12 MP layer used to recomposite every layer of the document (#2883, P8/P9 in
/// `perf/budgets.toml`).
pub fn step_damage(a: &Document, b: &Document) -> Option<Rect> {
    if !doc_pixels_equal(a, b) {
        return None;
    }
    fn walk(xs: &[Layer], ys: &[Layer], a: &Document, b: &Document, out: &mut Rect) -> bool {
        // Pair the layers by id, stepping over layers only one side has: one that draws nothing
        // changes no pixels (a new empty layer or group, #1771), any other only where it draws.
        let one_sided = |layers: &[Layer], k: usize, canvas: Rect, out: &mut Rect| {
            let Some(r) = one_sided_bounds(layers, k, canvas) else { return false };
            if !r.is_empty() {
                *out = if out.is_empty() { r } else { out.union(&r) };
            }
            true
        };
        let (mut i, mut j) = (0, 0);
        loop {
            let (x, y) = match (xs.get(i), ys.get(j)) {
                (None, None) => return true,
                (Some(x), Some(y)) if x.id == y.id => (x, y),
                _ if inert_at(ys, j) => {
                    j += 1;
                    continue;
                }
                _ if inert_at(xs, i) => {
                    i += 1;
                    continue;
                }
                (_, Some(y)) if xs.iter().all(|x| x.id != y.id) => {
                    if !one_sided(ys, j, b.bounds(), out) {
                        return false;
                    }
                    j += 1;
                    continue;
                }
                (Some(x), _) if ys.iter().all(|y| y.id != x.id) => {
                    if !one_sided(xs, i, a.bounds(), out) {
                        return false;
                    }
                    i += 1;
                    continue;
                }
                // Both sides have the layer, in another order.
                _ => return false,
            };
            i += 1;
            j += 1;
            if x.video.is_some() || y.video.is_some() {
                return false;
            }
            // Clipping and channel exclusion change how *other* layers render, outside this
            // layer's own bounds; they have no bounded damage.
            if x.clipped != y.clipped || x.excluded_channels != y.excluded_channels {
                return false;
            }
            if !layer_pixels_equal(x, y) {
                // Both sides' extents: a move changes pixels where it left and where it landed.
                let (Some(bx), Some(by)) = (photocraft_compose::change_bounds(x, a.bounds()), photocraft_compose::change_bounds(y, b.bounds())) else {
                    return false;
                };
                *out = if out.is_empty() { bx.union(&by) } else { out.union(&bx).union(&by) };
            }
            match (x.children(), y.children()) {
                (Some(xc), Some(yc)) => {
                    if !walk(xc, yc, a, b, out) {
                        return false;
                    }
                }
                (None, None) => {}
                _ => return false,
            }
        }
    }
    let mut out = Rect::EMPTY;
    if walk(&a.layers, &b.layers, a, b, &mut out) { Some(out) } else { None }
}

/// The pixels adding or removing `layers[k]` can change, when that is bounded: its change bounds
/// (`photocraft_compose::change_bounds`: not for an adjustment, or a pass-through group holding
/// one), provided it isn't the clipping base of clipped layers above it (they would change base),
/// and neither it nor anything inside it is a video layer or excludes channels (both reach
/// outside the layer's bounds, see [`step_damage`]).
fn one_sided_bounds(layers: &[Layer], k: usize, canvas: Rect) -> Option<Rect> {
    fn contained(l: &Layer, depth: usize) -> bool {
        depth < 256 && l.video.is_none() && l.excluded_channels == 0 && l.children().is_none_or(|c| c.iter().all(|c| contained(c, depth + 1)))
    }
    let l = layers.get(k)?;
    let new_base = !l.clipped && layers.get(k + 1).is_some_and(|above| above.clipped);
    if new_base || !contained(l, 0) {
        return None;
    }
    photocraft_compose::change_bounds(l, canvas)
}

/// Whether `layers[k]` draws nothing wherever it sits, so adding or removing it leaves the
/// composite as it was: an empty pixel layer (no tiles, transparent default) or an empty group,
/// in a Normal or Pass Through blend with no mask, vector mask, effects or Advanced Blending.
/// A clipping base that would gain or lose the clipped layers above it is not inert.
fn inert_at(layers: &[Layer], k: usize) -> bool {
    let Some(l) = layers.get(k) else { return false };
    let empty = match &l.content {
        LayerContent::Raster(s) => s.tile_count() == 0 && s.format().alpha && s.default_bytes().iter().all(|b| *b == 0),
        LayerContent::Group(g) => g.children.is_empty() && g.artboard.is_none(),
        _ => false,
    };
    empty
        && matches!(l.blend, photocraft_doc::BlendMode::Normal | photocraft_doc::BlendMode::PassThrough)
        && l.mask.is_none()
        && l.vector_mask.is_none()
        && !photocraft_compose::effects::has_effects(l)
        && l.fill_cache.is_none()
        && l.video.is_none()
        && l.advanced == photocraft_doc::AdvancedBlending::default()
        && (l.clipped || layers.get(k + 1).is_none_or(|above| !above.clipped))
}

/// Report that adding layer `id` to the active document changed no pixels when it draws
/// nothing ([`inert_at`]), so the canvas doesn't recomposite every layer for it (#1771). Looks
/// only at the new layer and its neighbour, never at other layers' tiles.
pub(crate) fn note_inert_insert(s: &mut Session, id: LayerId) {
    let Some(st) = s.active_mut() else { return };
    let Some(path) = st.doc.path_of(id) else { return };
    let Some((&k, parent)) = path.split_last() else { return };
    let siblings = if parent.is_empty() { Some(st.doc.layers.as_slice()) } else { st.doc.layer_at(parent).and_then(Layer::children) };
    if siblings.is_some_and(|sib| inert_at(sib, k)) {
        st.last_damage = Some(Rect::EMPTY);
    }
}

/// Report the pixels an edit from `before` that added layers changed ([`step_damage`]), for
/// commands made of more than one edit, so the canvas recomposites only where the new layers
/// draw instead of the whole document.
pub(crate) fn note_insert(s: &mut Session, before: &Document) {
    let Some(st) = s.active_mut() else { return };
    st.last_damage = step_damage(before, &st.doc);
}

/// Content bounds used by Align/Distribute: the layer's pixels (type and shape layers use their
/// rendered pixels), or the union of a group's children. Fill and adjustment layers have none.
pub fn layer_bounds(l: &Layer) -> Option<Rect> {
    let r = match &l.content {
        LayerContent::Group(g) => g.children.iter().filter_map(layer_bounds).fold(Rect::EMPTY, |a, b| a.union(&b)),
        _ => l.surface().map(|s| s.content_bounds()).unwrap_or(Rect::EMPTY),
    };
    (!r.is_empty()).then_some(r)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Edge {
    Top,
    VCenter,
    Bottom,
    Left,
    HCenter,
    Right,
}

impl Edge {
    fn parse(k: &str) -> Option<Edge> {
        Some(match k {
            "topEdges" => Edge::Top,
            "verticalCenters" => Edge::VCenter,
            "bottomEdges" => Edge::Bottom,
            "leftEdges" => Edge::Left,
            "horizontalCenters" => Edge::HCenter,
            "rightEdges" => Edge::Right,
            _ => return None,
        })
    }
    fn vertical(self) -> bool {
        matches!(self, Edge::Top | Edge::VCenter | Edge::Bottom)
    }
    /// The edge's coordinate, doubled so centres stay integral.
    fn value2(self, r: &Rect) -> i64 {
        let (a, b) = if self.vertical() { (r.y0, r.y1) } else { (r.x0, r.x1) };
        match self {
            Edge::Top | Edge::Left => 2 * a as i64,
            Edge::Bottom | Edge::Right => 2 * b as i64,
            Edge::VCenter | Edge::HCenter => a as i64 + b as i64,
        }
    }
    fn delta(self, d: i32) -> (i32, i32) {
        if self.vertical() { (0, d) } else { (d, 0) }
    }
}

/// Selected top-level layers that have content bounds, bottom-to-top.
fn bounded_selection(doc: &Document, sel: &[LayerId]) -> Vec<(LayerId, Rect)> {
    top_level(doc, sel).into_iter().filter_map(|id| Some((id, layer_bounds(doc.layer(id)?)?))).collect()
}

fn align(s: &mut Session, p: &Value, kind: &str) -> Result<Value> {
    let cmd = format!("layer.align.{kind}");
    let edge = Edge::parse(kind).ok_or_else(|| bad(&cmd, "unknown edge"))?;
    let sel = selected(s);
    let to = p.get("to").and_then(Value::as_str).unwrap_or("auto").to_string();
    let before = s.active().ok_or(EngineError::NoDocument)?.doc.clone();
    let moved = s.edit("Align", |doc, _| {
        let items = bounded_selection(doc, &sel);
        if items.is_empty() {
            return Err(EngineError::Other("the selected layers have no content to align".into()));
        }
        let target = match to.as_str() {
            "canvas" => doc.bounds(),
            "selection" => doc.selection.as_ref().map(|m| m.content_bounds()).ok_or_else(|| EngineError::Other("there is no selection to align to".into()))?,
            "auto" | "layers" => match (&doc.selection, sel.len()) {
                // Photoshop: one layer plus an active selection aligns to the selection bounds.
                (Some(m), 1) if to == "auto" => m.content_bounds(),
                // One layer with no selection aligns to the canvas (otherwise it would align to itself).
                (None, 1) if to == "auto" => doc.bounds(),
                _ => items.iter().fold(Rect::EMPTY, |a, (_, b)| a.union(b)),
            },
            other => return Err(bad(&cmd, format!("`to` must be auto|layers|selection|canvas, not `{other}`"))),
        };
        if target.is_empty() {
            return Err(EngineError::Other("nothing to align to".into()));
        }
        let t2 = edge.value2(&target);
        let moves: Vec<(LayerId, i32, i32)> = items
            .iter()
            .flat_map(|(id, r)| {
                let d = (t2 - edge.value2(r)).div_euclid(2) as i32;
                let (dx, dy) = edge.delta(d);
                // Linked layers travel with the layer they are linked to.
                with_links(doc, &[*id]).into_iter().map(move |l| (l, dx, dy))
            })
            .collect();
        let moves = dedup_moves(doc, moves);
        move_layers(doc, &before, &moves)?;
        Ok(moves.iter().filter(|m| m.1 != 0 || m.2 != 0).count())
    })?;
    Ok(json!({"moved": moved}))
}

/// One move per layer (first wins), skipping layers nested in another moved layer.
fn dedup_moves(doc: &Document, moves: Vec<(LayerId, i32, i32)>) -> Vec<(LayerId, i32, i32)> {
    let mut out: Vec<(LayerId, i32, i32)> = Vec::new();
    for m in moves {
        if !out.iter().any(|o| o.0 == m.0) {
            out.push(m);
        }
    }
    let ids: Vec<LayerId> = out.iter().map(|m| m.0).collect();
    let keep = top_level(doc, &ids);
    out.retain(|m| keep.contains(&m.0));
    out
}

fn distribute(s: &mut Session, kind: &str) -> Result<Value> {
    let sel = selected(s);
    let kind = kind.to_string();
    let before = s.active().ok_or(EngineError::NoDocument)?.doc.clone();
    let moved = s.edit("Distribute", |doc, _| {
        let mut items = bounded_selection(doc, &sel);
        if items.len() < 3 {
            return Err(EngineError::Other("distributing needs three or more layers with content".into()));
        }
        let n = items.len();
        let mut moves = Vec::new();
        match kind.as_str() {
            "horizontally" | "vertically" => {
                // Equal gaps between neighbours; the outermost layers stay put.
                let v = kind == "vertically";
                let lo = |r: &Rect| if v { r.y0 } else { r.x0 } as i64;
                let hi = |r: &Rect| if v { r.y1 } else { r.x1 } as i64;
                items.sort_by_key(|(_, r)| (lo(r) + hi(r), lo(r)));
                let span = hi(&items[n - 1].1) - lo(&items[0].1);
                let total: i64 = items.iter().map(|(_, r)| hi(r) - lo(r)).sum();
                let gap = (span - total) as f64 / (n - 1) as f64;
                let mut cursor = lo(&items[0].1) as f64;
                for (id, r) in &items {
                    let want = cursor.round() as i64;
                    let d = (want - lo(r)) as i32;
                    moves.push(if v { (*id, 0, d) } else { (*id, d, 0) });
                    cursor += (hi(r) - lo(r)) as f64 + gap;
                }
            }
            k => {
                let edge = Edge::parse(k).ok_or_else(|| bad("layer.distribute", format!("unknown kind `{k}`")))?;
                items.sort_by_key(|(_, r)| edge.value2(r));
                let (first, last) = (edge.value2(&items[0].1) as f64, edge.value2(&items[n - 1].1) as f64);
                for (i, (id, r)) in items.iter().enumerate() {
                    let want2 = first + (last - first) * i as f64 / (n - 1) as f64;
                    let d = ((want2 - edge.value2(r) as f64) / 2.0).round() as i32;
                    let (dx, dy) = edge.delta(d);
                    moves.push((*id, dx, dy));
                }
            }
        }
        let moves: Vec<_> = moves.into_iter().flat_map(|(id, dx, dy)| with_links(doc, &[id]).into_iter().map(move |l| (l, dx, dy))).collect();
        let moves = dedup_moves(doc, moves);
        move_layers(doc, &before, &moves)?;
        Ok(moves.iter().filter(|m| m.1 != 0 || m.2 != 0).count())
    })?;
    Ok(json!({"moved": moved}))
}

/// Reorder one or multiple layers to a single destination in one undoable edit.
///
/// Explicit `layers` are ordered according to the source document, not caller ordering.
/// Selected descendants of a moved group remain inside their parent; only top-level
/// selected nodes are detached, and every reference is validated before editing.
pub fn move_to(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "layer.moveTo";
    let target = LayerId(p.get("target").and_then(Value::as_u64).ok_or_else(|| bad(CMD, "missing `target`"))?);
    let pos = p.get("position").and_then(Value::as_str).unwrap_or("above");
    if !matches!(pos, "above" | "below" | "into") {
        return Err(bad(CMD, "position must be above, below, or into"));
    }
    // ⌥-drag in the Layers panel: duplicates land at the target and the layers stay put.
    let copy = p.get("copy").and_then(Value::as_bool).unwrap_or(false);

    let batch = p.get("layers");
    if batch.is_some() && p.get("layer").is_some() {
        return Err(bad(CMD, "give either `layer` or `layers`, not both"));
    }
    let requested = if let Some(raw) = batch {
        let arr = raw.as_array().filter(|a| !a.is_empty()).ok_or_else(|| bad(CMD, "`layers` must be a nonempty array of ids"))?;
        let mut ids = Vec::with_capacity(arr.len());
        for item in arr {
            let id = LayerId(item.as_u64().ok_or_else(|| bad(CMD, "every layer id must be an unsigned integer"))?);
            if ids.contains(&id) {
                return Err(bad(CMD, "duplicate layer id"));
            }
            ids.push(id);
        }
        ids
    } else {
        vec![crate::commands::layer_param(s, p)?]
    };
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let doc = &st.doc;
    let target_path = doc.path_of(target).ok_or(EngineError::NoLayer(target))?;
    for &id in &requested {
        if doc.layer(id).is_none() {
            return Err(EngineError::NoLayer(id));
        }
    }
    let moved_ids = top_level(doc, &requested);
    if moved_ids.len() == 1 && moved_ids[0] == target && batch.is_none() && !copy {
        // Preserve single-layer drag-to-self semantics.
        return Ok(Value::Null);
    }
    // Also rejects dragging a selected group onto any of its descendants.
    for id in &moved_ids {
        let path = doc.path_of(*id).ok_or(EngineError::NoLayer(*id))?;
        // A copy may land beside its own source (not inside it).
        let beside_itself = copy && *id == target && pos != "into";
        if target_path.starts_with(&path) && !beside_itself {
            return Err(bad(CMD, "target cannot be one of the moved layers or its descendant"));
        }
    }
    if pos == "into" && doc.layer(target).is_none_or(|l| !l.is_group()) {
        return Err(bad(CMD, "target is not a group"));
    }

    let original_active = st.active_layer;
    let background = crate::background_cmds::background_id(doc);
    let (count, copies) = s.edit(if copy { "Duplicate Layer" } else { "Reorder Layers" }, |doc, active| {
        let before = if copy { Default::default() } else { crate::clip_group::Before::read(doc, &moved_ids) };
        let mut moved = Vec::with_capacity(moved_ids.len());
        for &id in &moved_ids {
            moved.push(if copy { crate::commands::layer_copy(doc, id)? } else { doc.remove(id).ok_or(EngineError::NoLayer(id))? });
        }
        let copies: Vec<LayerId> = if copy { moved.iter().map(|l| l.id).collect() } else { Vec::new() };
        // Look up the destination AFTER removing every source: its indices may have shifted.
        let path = doc.path_of(target).ok_or(EngineError::NoLayer(target))?;
        if pos == "into" {
            let children = doc.layer_at_mut(&path).and_then(Layer::children_mut).ok_or_else(|| bad(CMD, "target is not a group"))?;
            children.extend(moved);
        } else {
            let (&idx, parent) = path.split_last().ok_or(EngineError::NoLayer(target))?;
            let siblings = if parent.is_empty() {
                &mut doc.layers
            } else {
                doc.layer_at_mut(parent).and_then(Layer::children_mut).ok_or_else(|| bad(CMD, "target's parent is not a group"))?
            };
            let at = if pos == "below" { idx } else { idx + 1 };
            siblings.splice(at..at, moved);
        }
        check_group_depth(doc, "Reorder Layer")?;
        crate::background_cmds::keep_background_at_bottom(background, doc)?;
        crate::clip_group::settle(doc, if copy { &copies } else { &moved_ids }, &before, false);
        *active = if copy {
            copies.last().copied()
        } else if batch.is_some() {
            original_active.filter(|id| requested.contains(id)).or_else(|| requested.last().copied())
        } else {
            moved_ids.first().copied()
        };
        Ok((moved_ids.len(), copies))
    })?;
    if copy {
        let ids: Vec<u64> = copies.iter().map(|id| id.0).collect();
        if batch.is_some() {
            let active = copies.last().copied();
            reselect(s, copies, active);
            return Ok(json!({"moved": count, "layers": ids}));
        }
        return Ok(json!({"layer": ids.first()}));
    }
    if batch.is_some() {
        let active = original_active.filter(|id| requested.contains(id)).or_else(|| requested.last().copied());
        reselect(s, requested, active);
        Ok(json!({"moved": count}))
    } else {
        Ok(Value::Null)
    }
}

// ---------- structural commands ----------

fn link_layers(s: &mut Session) -> Result<Value> {
    let sel = selected(s);
    let linked = s.edit("Link Layers", |doc, _| {
        let groups: Vec<Option<u64>> = sel.iter().map(|id| doc.layer(*id).and_then(|l| l.link_group)).collect();
        // Already one link group (or a single linked layer): unlink, like Photoshop's toggle.
        let first = groups.first().copied().flatten();
        let unlink = first.is_some() && groups.iter().all(|g| *g == first);
        // A loaded file can store any id up to u64::MAX, so the new id must not be a bare
        // max + 1 (#961). Unlinking needs no new id at all.
        let next = if unlink { None } else { Some(doc.unused_link_group().ok_or_else(|| bad("layer.linkLayers", "no free link group id is left"))?) };
        for id in &sel {
            let l = doc.layer_mut(*id).ok_or(EngineError::NoLayer(*id))?;
            l.link_group = next;
        }
        // A group left with a single member is no longer a link.
        let mut counts = std::collections::HashMap::<u64, usize>::new();
        for (_, _, l) in doc.walk() {
            if let Some(g) = l.link_group {
                *counts.entry(g).or_default() += 1;
            }
        }
        let lonely: Vec<LayerId> =
            doc.walk().iter().filter(|(_, _, l)| l.link_group.is_some_and(|g| counts.get(&g).is_some_and(|&c| c < 2))).map(|(_, _, l)| l.id).collect();
        for id in lonely {
            if let Some(l) = doc.layer_mut(id) {
                l.link_group = None;
            }
        }
        Ok(!unlink)
    })?;
    Ok(json!({"linked": linked}))
}

fn lock_layers(s: &mut Session, p: &Value) -> Result<Value> {
    let sel = selected(s);
    let keys = ["transparency", "pixels", "position", "artboard", "all"];
    let given: Vec<(&str, bool)> = keys.iter().filter_map(|k| Some((*k, p.get(*k)?.as_bool()?))).collect();
    s.edit("Lock Layers", |doc, _| {
        // No explicit locks: toggle "lock all" (on unless every selected layer is already locked).
        let all_locked = sel.iter().all(|id| doc.layer(*id).is_some_and(|l| l.locks.all));
        let given: Vec<(&str, bool)> = if given.is_empty() { vec![("all", !all_locked)] } else { given.clone() };
        for id in &sel {
            let l = doc.layer_mut(*id).ok_or(EngineError::NoLayer(*id))?;
            for (k, v) in &given {
                l.locks.set(k, *v);
            }
        }
        Ok(())
    })?;
    Ok(json!({"layers": sel.len()}))
}

fn rename_layer(s: &mut Session, p: &Value) -> Result<Value> {
    let name = p.get("name").and_then(Value::as_str).filter(|n| !n.trim().is_empty()).ok_or_else(|| bad("layer.renameLayer", "missing `name`"))?.to_string();
    let id = crate::commands::layer_param(s, p)?;
    s.edit("Rename Layer", |doc, _| {
        doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?.name = name;
        Ok(())
    })?;
    Ok(Value::Null)
}

fn reverse(s: &mut Session) -> Result<Value> {
    let sel = selected(s);
    s.edit("Reverse", |doc, _| {
        let background = crate::background_cmds::background_id(doc);
        let ids = top_level(doc, &sel);
        if ids.len() < 2 {
            return Err(EngineError::Other("select two or more layers to reverse".into()));
        }
        let paths: Vec<Vec<usize>> = ids.iter().map(|id| doc.path_of(*id).ok_or(EngineError::NoLayer(*id))).collect::<Result<_>>()?;
        let layers: Vec<Layer> = ids.iter().map(|id| doc.layer(*id).cloned().ok_or(EngineError::NoLayer(*id))).collect::<Result<_>>()?;
        // Slots are disjoint (no layer is nested in another), so writing whole layers back to the
        // same paths never shifts another slot.
        for (path, l) in paths.iter().zip(layers.into_iter().rev()) {
            *doc.layer_at_mut(path).ok_or_else(|| EngineError::Other("layer tree changed".into()))? = l;
        }
        crate::background_cmds::keep_background_at_bottom(background, doc)
    })?;
    Ok(Value::Null)
}

/// Refuses an edit that left `doc` nested deeper than [`photocraft_doc::MAX_GROUP_DEPTH`]
/// groups. Called at the end of a `Session::edit` closure, so an `Err` leaves the document and
/// history untouched.
pub(crate) fn check_group_depth(doc: &Document, what: &str) -> Result<()> {
    if doc.max_group_depth() > photocraft_doc::MAX_GROUP_DEPTH {
        return Err(EngineError::Other(format!("{what} would nest layers deeper than {} groups", photocraft_doc::MAX_GROUP_DEPTH)));
    }
    Ok(())
}

/// Group Layers (⌘G) / Group from Layers: the explicit layer, or every selected layer, moves
/// into a new group placed where the top-most of them was. Bottom-to-top order is preserved.
pub fn group_layers(s: &mut Session, p: &Value) -> Result<Value> {
    let ids = match crate::commands::explicit_layer(s, p)? {
        Some(id) => vec![id],
        None => selected(s),
    };
    if ids.is_empty() {
        return Err(EngineError::Other("no layer selected".into()));
    }
    let name = p.get("name").and_then(Value::as_str).map(str::to_string);
    let gid = s.edit("Group Layers", |doc, active| {
        let ids = top_level(doc, &ids);
        let top = *ids.last().ok_or(EngineError::NoLayer(LayerId(0)))?;
        let name = name.unwrap_or_else(|| doc.next_layer_name("Group"));
        let gid = doc.insert_above(Some(top), Layer::group(name, vec![]));
        let mut children = Vec::with_capacity(ids.len());
        for id in &ids {
            children.push(doc.remove(*id).ok_or(EngineError::NoLayer(*id))?);
        }
        *doc.layer_mut(gid).and_then(Layer::children_mut).ok_or(EngineError::NoLayer(gid))? = children;
        check_group_depth(doc, "Group Layers")?;
        *active = Some(gid);
        Ok(gid)
    })?;
    reselect(s, vec![gid], Some(gid));
    Ok(json!({ "layer": gid.0 }))
}

fn merge_layers(s: &mut Session) -> Result<Value> {
    let sel = selected(s);
    if sel.len() < 2 {
        // ⌘E on a lone group is Photoshop's Merge Group (#3832); on a layer, Merge Down.
        if let Some(&id) = sel.first()
            && s.active().and_then(|d| d.doc.layer(id)).is_some_and(|l| matches!(l.content, LayerContent::Group(_)))
        {
            return s.execute("layer.mergeGroup", json!({ "layer": id.0 }));
        }
        return s.execute("layer.mergeDown", json!({}));
    }
    let mid = s.edit("Merge Layers", |doc, active| {
        let ids = top_level(doc, &sel);
        let top_id = *ids.last().ok_or_else(|| EngineError::Other("Merge Layers needs two or more layers".into()))?;
        let top = doc.layer(top_id).ok_or(EngineError::NoLayer(top_id))?.clone();
        let mut solo = doc.clone();
        solo.layers = ids.iter().filter_map(|id| doc.layer(*id)).filter(|l| l.visible).cloned().collect();
        if solo.layers.is_empty() {
            return Err(EngineError::Other("the selected layers are all hidden".into()));
        }
        let fmt = doc.pixel_format();
        let fmt = PixelFormat::new(fmt.mode, fmt.sample, true);
        let mut merged = Layer::raster(top.name.clone(), fmt);
        merged.locks = top.locks;
        *crate::pixels_mut(&mut merged)? = crate::pixels::composite_layers(&solo, fmt, None);
        let mid = merged.id;
        // Hidden selected layers are discarded, as in Photoshop.
        for id in &ids[..ids.len() - 1] {
            doc.remove(*id);
        }
        *doc.layer_mut(top_id).ok_or(EngineError::NoLayer(top_id))? = merged;
        *active = Some(mid);
        Ok(mid)
    })?;
    reselect(s, vec![mid], Some(mid));
    Ok(json!({ "layer": mid.0 }))
}

/// Next visible layer row below the deleted target, then rows above in reverse order.
/// Group headers precede their children in the UI, unlike Document::walk's storage order.
pub(crate) fn deletion_neighbours(doc: &Document, target: LayerId) -> Vec<LayerId> {
    let rows = |expand_all: bool| {
        let mut stack: Vec<&Layer> = doc.layers.iter().collect();
        let mut order = Vec::new();
        while let Some(layer) = stack.pop() {
            order.push(layer.id);
            if let LayerContent::Group(group) = &layer.content
                && (expand_all || group.expanded)
            {
                stack.extend(group.children.iter());
            }
        }
        order
    };
    let mut order = rows(false);
    if !order.contains(&target) {
        order = rows(true);
    }
    let Some(index) = order.iter().position(|id| *id == target) else {
        return Vec::new();
    };
    order.iter().skip(index + 1).chain(order.iter().take(index).rev()).copied().collect()
}

/// Delete Layer with several layers selected.
pub fn delete_selected(s: &mut Session) -> Result<Value> {
    let sel = selected(s);
    s.edit("Delete Layers", |doc, active| {
        let ids = top_level(doc, &sel);
        let neighbours = active.map(|id| deletion_neighbours(doc, id)).unwrap_or_default();
        let before = crate::clip_group::Before::read(doc, &ids);
        for id in &ids {
            doc.remove(*id).ok_or(EngineError::NoLayer(*id))?;
        }
        crate::clip_group::settle(doc, &[], &before, false);
        if doc.layers.is_empty() {
            return Err(EngineError::Other("a document must keep at least one layer".into()));
        }
        *active = neighbours.into_iter().find(|candidate| doc.layer(*candidate).is_some());
        Ok(())
    })?;
    Ok(json!({"deleted": sel.len()}))
}

/// Duplicate Layer with several layers selected: each copy goes above its original and the
/// copies become the selection. Artboard copies go beside their boards unless `in_place`
/// ([`crate::artboard_cmds::place_copy`]). `label` names the history step (Layer via Copy reuses
/// this path, #2777).
pub fn duplicate_selected(s: &mut Session, in_place: bool, label: &str) -> Result<Value> {
    let sel = selected(s);
    let before = s.active().ok_or(EngineError::NoDocument)?.doc.clone();
    let old_active = s.active().and_then(|d| d.active_layer);
    let (copies, active) = s.edit(label, |doc, active| {
        let mut copies = Vec::new();
        let mut new_active = None;
        for id in top_level(doc, &sel) {
            // Match the single-layer duplicate path: Background copies become
            // ordinary layers, while normal copies retain their original locks.
            let dup = crate::commands::layer_copy(doc, id)?;
            let nid = doc.insert_above(Some(id), dup);
            if !in_place {
                crate::artboard_cmds::place_copy(doc, nid)?;
            }
            if Some(id) == old_active {
                new_active = Some(nid);
            }
            copies.push(nid);
        }
        let a = new_active.or(copies.last().copied());
        *active = a;
        Ok((copies, a))
    })?;
    note_insert(s, &before);
    let out = copies.iter().map(|l| l.0).collect::<Vec<_>>();
    reselect(s, copies, active);
    Ok(json!({"layers": out, "layer": active.map(|l| l.0)}))
}

/// Show/hide every selected layer in one step.
pub fn set_visible_selected(s: &mut Session, visible: bool) -> Result<Value> {
    let sel = selected(s);
    s.edit(if visible { "Show Layers" } else { "Hide Layers" }, |doc, _| {
        for id in &sel {
            doc.layer_mut(*id).ok_or(EngineError::NoLayer(*id))?.visible = visible;
        }
        Ok(())
    })?;
    Ok(Value::Null)
}

/// Is more than one layer selected (so single-layer commands should act on the whole set)?
pub fn multi(s: &Session, p: &Value) -> bool {
    p.get("layer").is_none() && selected(s).len() > 1
}

// ---------- registry ----------

pub fn specs() -> Vec<CommandSpec> {
    macro_rules! spec {
        ($id:expr, $label:expr, $menu:expr, $sc:expr, $params:expr, $enabled:expr, $run:expr) => {
            CommandSpec { id: $id, label: $label, menu: $menu, shortcut: $sc, params: $params, enabled: $enabled, journal: true, run: $run }
        };
    }
    const ALIGN: &str = r##"{"to":"auto|layers|selection|canvas"="auto"} (auto: the selection bounds with one layer and an active selection, else the selected layers' bounds)"##;
    vec![
        spec!("select.allLayers", "All Layers", &["Select"], Some("Cmd+Alt+A"), "{}", has_doc, |s, _| select_all_layers(s)),
        spec!("select.deselectLayers", "Deselect Layers", &["Select"], None, "{}", has_doc, |s, _| {
            set_selection(s, Vec::new(), None, None)?;
            if let Some(st) = s.active_mut() {
                st.active_layer = None;
                st.selected_layers.clear();
            }
            Ok(Value::Null)
        }),
        spec!("select.findLayers", "Find Layers", &["Select"], Some("Cmd+Alt+Shift+F"), r##"{"name":str (case-insensitive substring)}"##, has_doc, find_layers),
        spec!("layer.selectLinkedLayers", "Select Linked Layers", &["Layer"], None, "{}", has_linked, |s, _| select_linked(s)),
        spec!("layer.align.topEdges", "Top Edges", &["Layer", "Align"], None, ALIGN, can_align, |s, p| align(s, p, "topEdges")),
        spec!("layer.align.verticalCenters", "Vertical Centers", &["Layer", "Align"], None, ALIGN, can_align, |s, p| align(s, p, "verticalCenters")),
        spec!("layer.align.bottomEdges", "Bottom Edges", &["Layer", "Align"], None, ALIGN, can_align, |s, p| align(s, p, "bottomEdges")),
        spec!("layer.align.leftEdges", "Left Edges", &["Layer", "Align"], None, ALIGN, can_align, |s, p| align(s, p, "leftEdges")),
        spec!("layer.align.horizontalCenters", "Horizontal Centers", &["Layer", "Align"], None, ALIGN, can_align, |s, p| align(s, p, "horizontalCenters")),
        spec!("layer.align.rightEdges", "Right Edges", &["Layer", "Align"], None, ALIGN, can_align, |s, p| align(s, p, "rightEdges")),
        spec!("layer.distribute.topEdges", "Top Edges", &["Layer", "Distribute"], None, "{}", can_distribute, |s, _| distribute(s, "topEdges")),
        spec!("layer.distribute.verticalCenters", "Vertical Centers", &["Layer", "Distribute"], None, "{}", can_distribute, |s, _| distribute(
            s,
            "verticalCenters"
        )),
        spec!("layer.distribute.bottomEdges", "Bottom Edges", &["Layer", "Distribute"], None, "{}", can_distribute, |s, _| distribute(s, "bottomEdges")),
        spec!("layer.distribute.leftEdges", "Left Edges", &["Layer", "Distribute"], None, "{}", can_distribute, |s, _| distribute(s, "leftEdges")),
        spec!("layer.distribute.horizontalCenters", "Horizontal Centers", &["Layer", "Distribute"], None, "{}", can_distribute, |s, _| distribute(
            s,
            "horizontalCenters"
        )),
        spec!("layer.distribute.rightEdges", "Right Edges", &["Layer", "Distribute"], None, "{}", can_distribute, |s, _| distribute(s, "rightEdges")),
        spec!(
            "layer.distribute.horizontally",
            "Horizontally",
            &["Layer", "Distribute"],
            None,
            "{} (equal horizontal gaps)",
            can_distribute,
            |s, _| distribute(s, "horizontally")
        ),
        spec!("layer.distribute.vertically", "Vertically", &["Layer", "Distribute"], None, "{} (equal vertical gaps)", can_distribute, |s, _| distribute(
            s,
            "vertically"
        )),
        spec!("layer.linkLayers", "Link Layers", &["Layer"], None, "{} (toggles: unlinks when the selection is already one link group)", can_link, |s, _| {
            link_layers(s)
        }),
        spec!("layer.mergeLayers", "Merge Layers", &["Layer"], Some("Cmd+E"), "{} (one layer selected: Merge Down)", has_layer, |s, _| merge_layers(s)),
        spec!("layer.new.groupFromLayers", "Group from Layers…", &["Layer", "New"], None, r##"{"name":str?}"##, has_layer, group_layers),
        spec!("layer.arrange.reverse", "Reverse", &["Layer", "Arrange"], None, "{}", two_plus, |s, _| reverse(s)),
        spec!(
            "layer.lockLayers",
            "Lock Layers…",
            &["Layer"],
            None,
            r##"{"transparency":bool?,"pixels":bool?,"position":bool?,"artboard":bool?,"all":bool?} (none given: toggle lock all)"##,
            has_layer,
            lock_layers
        ),
        spec!("layer.renameLayer", "Rename Layer", &["Layer"], None, r##"{"layer":id?,"name":str}"##, has_layer, rename_layer),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicates_are_numbered_not_stacked() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 20, "height": 20})).unwrap();
        s.execute("layer.new.layer", json!({"name": "Layer 1"})).unwrap();
        let name = |s: &Session| {
            let st = s.active().unwrap();
            st.doc.layer(st.active_layer.unwrap()).unwrap().name.clone()
        };
        s.execute("layer.duplicate", json!({})).unwrap();
        assert_eq!(name(&s), "Layer 1 copy");
        // Duplicating the copy (the active layer) numbers it rather than adding "copy" again.
        s.execute("layer.duplicate", json!({})).unwrap();
        assert_eq!(name(&s), "Layer 1 copy 2");
        s.execute("layer.duplicate", json!({})).unwrap();
        assert_eq!(name(&s), "Layer 1 copy 3");
        let doc = &s.active().unwrap().doc;
        assert_eq!(doc.copy_name("Layer 1"), "Layer 1 copy 4");
        // Names that merely contain "copy" keep it.
        assert_eq!(doc.copy_name("Copywriting"), "Copywriting copy");
        assert_eq!(doc.copy_name("A copyedit"), "A copyedit copy");
    }

    fn session(depth: u32) -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 100, "height": 80, "depth": depth})).unwrap();
        s
    }

    fn doc(s: &Session) -> &Document {
        &s.active().unwrap().doc
    }

    /// New pixel layer with an opaque red rectangle.
    fn rect_layer(s: &mut Session, r: Rect) -> LayerId {
        let id = LayerId(s.execute("layer.new.layer", json!({})).unwrap()["layer"].as_u64().unwrap());
        s.edit("paint", |doc, _| {
            doc.layer_mut(id).unwrap().surface_mut().unwrap().fill_rect(r, &[1.0, 0.0, 0.0, 1.0]);
            Ok(())
        })
        .unwrap();
        id
    }

    fn bounds(s: &Session, id: LayerId) -> Rect {
        layer_bounds(doc(s).layer(id).unwrap()).unwrap()
    }

    fn sel(s: &Session) -> Vec<LayerId> {
        selected(s)
    }

    #[test]
    fn layer_selection_keeps_document_revision_and_pending_edits() {
        let mut s = session(8);
        let a = rect_layer(&mut s, Rect::new(10, 10, 20, 20));
        let b = rect_layer(&mut s, Rect::new(30, 10, 40, 20));
        let revision = s.active().unwrap().revision;
        s.active_mut().unwrap().saved_revision = revision;
        for dirty in [false, true] {
            if dirty {
                s.execute("layer.translate", json!({"layer": a.0, "dx": 1, "dy": 0})).unwrap();
            }
            let st = s.active().unwrap();
            let document_revision = st.revision - st.view_revision;
            let history = st.history.entries().len();
            let document = st.doc.clone();
            for (id, mode) in [(a, "replace"), (b, "add"), (a, "range"), (b, "remove"), (b, "toggle")] {
                s.execute("layer.select", json!({"layer": id.0, "mode": mode})).unwrap();
                let st = s.active().unwrap();
                assert_eq!(st.revision - st.view_revision, document_revision);
                assert_eq!(st.history.entries().len(), history);
                assert_eq!(st.revision != st.saved_revision, dirty);
                assert!(std::sync::Arc::ptr_eq(&st.doc, &document));
            }
        }
    }

    fn damage(s: &Session) -> Option<Rect> {
        s.active().unwrap().last_damage
    }

    /// A Photoshop action's nudge in points moves whole pixels at the document's resolution:
    /// 0.36 pt is 1 px at 200 ppi and 1.5 → 2 px at 300 ppi.
    #[test]
    fn translate_in_points_follows_the_resolution() {
        for (res, moved) in [(200, 1), (300, 2), (72, 0)] {
            let mut s = Session::new();
            s.execute("file.new", json!({"width": 100, "height": 80, "resolution": res})).unwrap();
            let a = rect_layer(&mut s, Rect::new(10, 10, 20, 20));
            s.execute("layer.translate", json!({"dx": 0.36, "dy": -0.36, "units": "pt"})).unwrap();
            assert_eq!(bounds(&s, a), Rect::new(10 + moved, 10 - moved, 20 + moved, 20 - moved), "{res} ppi");
        }
        let mut s = session(8);
        rect_layer(&mut s, Rect::new(10, 10, 20, 20));
        for bad in [json!({"dx": 1, "units": "in"}), json!({"dx": "x", "units": "pt"}), json!({"dx": 1e300, "units": "pt"})] {
            assert!(s.execute("layer.translate", bad.clone()).is_err(), "{bad}");
        }
    }

    /// `remove` deselects a selected layer, leaves an unselected one alone and never empties the
    /// selection.
    #[test]
    fn select_remove_takes_a_layer_out_of_the_selection() {
        let mut s = session(8);
        let a = rect_layer(&mut s, Rect::new(10, 10, 20, 20));
        let b = rect_layer(&mut s, Rect::new(30, 10, 40, 20));
        let c = rect_layer(&mut s, Rect::new(50, 10, 60, 20));
        s.execute("layer.select", json!({"layer": a.0})).unwrap();
        s.execute("layer.select", json!({"layer": b.0, "mode": "add"})).unwrap();
        s.execute("layer.select", json!({"layer": c.0, "mode": "remove"})).unwrap();
        assert_eq!(sel(&s).len(), 2, "an unselected layer stays out");
        s.execute("layer.select", json!({"layer": b.0, "mode": "remove"})).unwrap();
        assert_eq!(sel(&s), vec![a]);
        s.execute("layer.select", json!({"layer": a.0, "mode": "remove"})).unwrap();
        assert_eq!(sel(&s), vec![a], "the last selected layer stays");
    }

    /// A mask with its chain off stays put while the layer moves; a linked one moves along.
    #[test]
    fn translate_leaves_an_unlinked_mask_in_place() {
        for depth in [8, 16] {
            let mut s = session(depth);
            let a = rect_layer(&mut s, Rect::new(10, 10, 20, 20));
            s.execute("layer.select", json!({"layer": a.0})).unwrap();
            s.execute("layer.layerMask.hideAll", json!({})).unwrap();
            s.edit("mask", |doc, _| {
                doc.layer_mut(a).unwrap().mask.as_mut().unwrap().surface.fill_rect(Rect::new(12, 12, 18, 18), &[1.0]);
                Ok(())
            })
            .unwrap();
            let mval = |s: &Session, x, y| doc(s).layer(a).unwrap().mask.as_ref().unwrap().surface.sample_channel(x, y, 0);
            s.execute("layer.layerMask.linked", json!({"linked": false})).unwrap();
            s.execute("layer.translate", json!({"dx": 30, "dy": 0})).unwrap();
            assert!(mval(&s, 14, 14) > 0.9, "{depth}-bit: unlinked mask stays");
            assert!(mval(&s, 44, 14) < 0.1);
            s.execute("edit.undo", json!({})).unwrap();
            s.execute("layer.layerMask.linked", json!({"linked": true})).unwrap();
            s.execute("layer.translate", json!({"dx": 30, "dy": 0})).unwrap();
            assert!(mval(&s, 44, 14) > 0.9, "{depth}-bit: linked mask moves");
        }
    }

    /// Aimed at the mask, a move shifts an unlinked mask alone and leaves a linked one with its layer.
    #[test]
    fn translate_with_the_mask_targeted_moves_an_unlinked_mask_alone() {
        for depth in [8, 16] {
            let mut s = session(depth);
            let a = rect_layer(&mut s, Rect::new(10, 10, 20, 20));
            s.execute("layer.select", json!({"layer": a.0})).unwrap();
            s.execute("layer.layerMask.hideAll", json!({})).unwrap();
            s.edit("mask", |doc, _| {
                doc.layer_mut(a).unwrap().mask.as_mut().unwrap().surface.fill_rect(Rect::new(12, 12, 18, 18), &[1.0]);
                Ok(())
            })
            .unwrap();
            let mval = |s: &Session, x, y| doc(s).layer(a).unwrap().mask.as_ref().unwrap().surface.sample_channel(x, y, 0);
            // Linked: the layer and its mask move together.
            s.execute("layer.translate", json!({"dx": 30, "dy": 0, "target": "mask"})).unwrap();
            assert_eq!(bounds(&s, a), Rect::new(40, 10, 50, 20), "{depth}-bit");
            assert!(mval(&s, 44, 14) > 0.9);
            s.execute("layer.layerMask.linked", json!({"linked": false})).unwrap();
            s.execute("layer.translate", json!({"dx": -30, "dy": 0, "target": "mask"})).unwrap();
            assert_eq!(bounds(&s, a), Rect::new(40, 10, 50, 20), "{depth}-bit: the layer stays");
            assert!(mval(&s, 14, 14) > 0.9, "the mask moved");
            assert!(mval(&s, 44, 14) < 0.1);
            s.execute("edit.undo", json!({})).unwrap();
            assert!(mval(&s, 44, 14) > 0.9, "undo puts the mask back");
        }
    }

    #[test]
    fn moves_and_layer_props_report_only_their_area() {
        let mut s = session(8);
        let a = rect_layer(&mut s, Rect::new(10, 10, 20, 20));
        s.execute("layer.select", json!({"layer": a.0})).unwrap();
        s.execute("layer.translate", json!({"dx": 5, "dy": -3})).unwrap();
        assert_eq!(damage(&s), Some(Rect::new(10, 7, 25, 20)), "where the layer was and is");
        s.execute("layer.setProps", json!({"layer": a.0, "visible": false})).unwrap();
        assert_eq!(damage(&s), Some(Rect::new(15, 7, 25, 17)));
        s.execute("layer.setProps", json!({"layer": a.0, "opacity": 0.5, "blend": "Multiply"})).unwrap();
        assert_eq!(damage(&s), Some(Rect::new(15, 7, 25, 17)));
        // Clipping changes reach the layers around: everything.
        s.execute("layer.setProps", json!({"layer": a.0, "clipped": true})).unwrap();
        assert_eq!(damage(&s), None);
        // An adjustment layer changes everything beneath it.
        let adj = s.execute("layer.newAdjustmentLayer.invert", json!({})).unwrap()["layer"].as_u64().unwrap();
        s.execute("layer.setProps", json!({"layer": adj, "visible": false})).unwrap();
        assert_eq!(damage(&s), None);
        // The Background (opaque everywhere): everything.
        let bg = doc(&s).layers[0].id;
        s.execute("layer.setProps", json!({"layer": bg.0, "opacity": 0.5})).unwrap();
        let all = doc(&s).bounds();
        assert!(damage(&s).is_none_or(|r| r.contains_rect(&all)), "{:?}", damage(&s));
        // Bad params still fail cleanly.
        assert!(s.execute("layer.setProps", json!({"layer": u64::MAX, "visible": true})).is_err());
        assert!(s.execute("layer.translate", json!({"layer": u64::MAX, "dx": 1})).is_err());
    }

    /// #1771: a new empty layer or group changes no pixels, so it (and undoing or redoing it)
    /// must not make the canvas recomposite every layer of the document.
    #[test]
    fn new_empty_layers_and_groups_damage_nothing() {
        for depth in [8, 16, 32] {
            let mut s = session(depth);
            let a = rect_layer(&mut s, Rect::new(10, 10, 20, 20));
            let composite = |s: &Session| photocraft_compose::render(doc(s), doc(s).bounds());
            let shown = composite(&s);
            s.execute("layer.new.layer", json!({})).unwrap();
            assert_eq!(damage(&s), Some(Rect::EMPTY), "{depth}-bit new layer");
            assert_eq!(composite(&s), shown, "an empty layer changes no pixels");
            s.execute("edit.undo", json!({})).unwrap();
            assert_eq!(damage(&s), Some(Rect::EMPTY), "{depth}-bit undo New Layer");
            s.execute("edit.redo", json!({})).unwrap();
            assert_eq!(damage(&s), Some(Rect::EMPTY), "{depth}-bit redo New Layer");
            s.execute("layer.new.group", json!({})).unwrap();
            assert_eq!(damage(&s), Some(Rect::EMPTY), "{depth}-bit new group");
            // Inside a group too.
            let child = s.execute("layer.new.layer", json!({})).unwrap()["layer"].as_u64().unwrap();
            let g = s.execute("layer.groupLayers", json!({})).unwrap()["layer"].as_u64().unwrap();
            s.execute("layer.select", json!({"layer": child})).unwrap();
            s.execute("layer.new.layer", json!({})).unwrap();
            assert_eq!(damage(&s), Some(Rect::EMPTY), "{depth}-bit new layer in a group");
            assert!(doc(&s).layer(LayerId(g)).and_then(Layer::children).is_some_and(|c| c.len() == 2));
            s.execute("edit.undo", json!({})).unwrap();
            assert_eq!(damage(&s), Some(Rect::EMPTY));
            // Undoing a step that painted still reports that area, not nothing.
            s.execute("layer.select", json!({"layer": a.0})).unwrap();
            s.execute("layer.translate", json!({"dx": 5, "dy": 0})).unwrap();
            s.execute("edit.undo", json!({})).unwrap();
            assert_eq!(damage(&s), Some(Rect::new(10, 10, 25, 20)));
        }
    }

    #[test]
    fn new_layers_that_change_clipping_recomposite_everything() {
        let mut s = session(8);
        let base = rect_layer(&mut s, Rect::new(10, 10, 20, 20));
        let clipped = rect_layer(&mut s, Rect::new(0, 0, 100, 80));
        s.edit("green", |doc, _| {
            doc.layer_mut(clipped).unwrap().surface_mut().unwrap().fill_rect(Rect::new(0, 0, 100, 80), &[0.0, 1.0, 0.0, 1.0]);
            Ok(())
        })
        .unwrap();
        s.execute("layer.setProps", json!({"layer": clipped.0, "clipped": true})).unwrap();
        // New Layer between a clipping base and its clipped layer joins their clip group (#2203),
        // so it draws nothing; an empty layer there that stayed unclipped would become their base.
        s.execute("layer.select", json!({"layer": base.0})).unwrap();
        let shown = photocraft_compose::render(doc(&s), doc(&s).bounds());
        let before = doc(&s).clone();
        s.execute("layer.new.layer", json!({})).unwrap();
        assert_eq!(damage(&s), Some(Rect::EMPTY));
        assert_eq!(photocraft_compose::render(doc(&s), doc(&s).bounds()), shown);
        let mut split = before.clone();
        split.insert_above(Some(base), Layer::raster("split", split.pixel_format()));
        assert_ne!(photocraft_compose::render(&split, split.bounds()), shown);
        assert_eq!(step_damage(&before, &split), None);
        assert_eq!(step_damage(&split, &before), None);
        s.execute("edit.undo", json!({})).unwrap();
        // A step that adds a layer with pixels (Layer via Copy, paste…) changes where it draws.
        let before = doc(&s).clone();
        let mut after = before.clone();
        let mut l = Layer::raster("painted", after.pixel_format());
        l.surface_mut().unwrap().fill_rect(Rect::new(1, 1, 2, 2), &[0.0, 1.0, 0.0, 1.0]);
        after.insert_above(None, l);
        assert_eq!(step_damage(&before, &after), Some(Rect::new(1, 1, 2, 2)));
        assert_eq!(step_damage(&after, &before), Some(Rect::new(1, 1, 2, 2)));
        // Unless it becomes the base of clipped layers above it.
        let mut based = before.clone();
        let mut l = Layer::raster("base", based.pixel_format());
        l.surface_mut().unwrap().fill_rect(Rect::new(1, 1, 2, 2), &[0.0, 1.0, 0.0, 1.0]);
        let under = based.layers.iter().position(|l| l.id == clipped).unwrap();
        based.layers.insert(under, l);
        assert_eq!(step_damage(&before, &based), None);
        assert_eq!(step_damage(&based, &before), None);
        // An adjustment reaches every pixel below it.
        let mut adjusted = before.clone();
        adjusted.insert_above(None, Layer::new("inv", LayerContent::Adjustment(photocraft_doc::Adjustment::Invert)));
        assert_eq!(step_damage(&before, &adjusted), None);
        assert_eq!(step_damage(&adjusted, &before), None);
        // Empty layers draw nothing, in any blend mode.
        let mut styled = before.clone();
        let mut l = Layer::raster("styled", styled.pixel_format());
        l.blend = photocraft_doc::BlendMode::Multiply;
        styled.insert_above(None, l);
        assert_eq!(step_damage(&before, &styled), Some(Rect::EMPTY));
        let mut plain = before.clone();
        plain.insert_above(None, Layer::raster("plain", plain.pixel_format()));
        assert_eq!(step_damage(&before, &plain), Some(Rect::EMPTY));
        assert_eq!(step_damage(&plain, &before), Some(Rect::EMPTY));
    }

    /// #2883 (P8/P9): a duplicate shares the source's tiles (no pixel copy) and recomposites only
    /// where it draws; undo brings back exactly the old pixels. A copy that becomes the clipping
    /// base of the layers above still recomposites everything.
    #[test]
    fn duplicate_shares_tiles_and_damages_only_the_copy() {
        let render = |s: &Session| photocraft_compose::render(doc(s), doc(s).bounds());
        let tiles =
            |s: &Session, id: LayerId| doc(s).layer(id).unwrap().surface().unwrap().tiles().map(|(c, t)| (*c, std::sync::Arc::as_ptr(t))).collect::<Vec<_>>();
        for depth in [8, 16, 32] {
            let mut s = session(depth);
            let a = rect_layer(&mut s, Rect::new(10, 10, 20, 20));
            s.execute("layer.setProps", json!({"layer": a.0, "opacity": 0.5})).unwrap();
            let shown = render(&s);
            let copy = LayerId(s.execute("layer.duplicate", json!({"layer": a.0})).unwrap()["layer"].as_u64().unwrap());
            assert_eq!(damage(&s), Some(Rect::new(10, 10, 20, 20)), "{depth}-bit: only where the copy draws");
            assert!(!tiles(&s, a).is_empty());
            assert_eq!(tiles(&s, copy), tiles(&s, a), "{depth}-bit: the copy shares the source's tiles");
            let now = render(&s);
            assert_ne!(now, shown, "the half-transparent copy darkens its area");
            let w = shown.rect.width() as usize;
            for (i, (p, q)) in shown.px.iter().zip(&now.px).enumerate() {
                if !Rect::new(10, 10, 20, 20).contains((i % w) as i32, (i / w) as i32) {
                    assert_eq!(p, q, "{depth}-bit: pixel {i} outside the damage changed");
                }
            }
            s.execute("edit.undo", json!({})).unwrap();
            assert!(doc(&s).layer(copy).is_none());
            assert_eq!(render(&s), shown);
            // A clipping base's copy goes between it and its clipped layer and becomes their base.
            let clipped = rect_layer(&mut s, Rect::new(0, 0, 50, 50));
            s.execute("layer.setProps", json!({"layer": clipped.0, "clipped": true})).unwrap();
            s.execute("layer.duplicate", json!({"layer": a.0})).unwrap();
            assert_eq!(damage(&s), None, "{depth}-bit: a new clipping base");
        }
    }

    /// The composite of `after` equals `before`'s outside `damage`.
    fn unchanged_outside(before: &Document, after: &Document, damage: Rect) -> bool {
        let (x, y) = (photocraft_compose::render(before, before.bounds()), photocraft_compose::render(after, after.bounds()));
        let w = before.bounds().width() as i32;
        x.px.iter().zip(&y.px).enumerate().all(|(i, (p, q))| p == q || damage.contains(i as i32 % w, i as i32 / w))
    }

    /// Duplicate, paste and delete, and their undo and redo, recomposite only where the layer
    /// draws; undoing a duplicate used to recomposite the whole document.
    #[test]
    fn duplicate_paste_and_delete_report_where_the_layer_draws() {
        let mut s = session(8);
        let a = rect_layer(&mut s, Rect::new(10, 10, 20, 20));
        let full = doc(&s).bounds();
        let step = |s: &mut Session, id: &str, p: serde_json::Value| {
            let before = doc(s).clone();
            s.execute(id, p).unwrap();
            let d = damage(s).unwrap_or_else(|| panic!("{id}: unbounded damage"));
            assert!(d.width() < full.width(), "{id}: {d:?}");
            assert!(unchanged_outside(&before, doc(s), d), "{id}: pixels changed outside {d:?}");
            s.execute("edit.undo", json!({})).unwrap();
            assert_eq!(damage(s), Some(d), "undo {id}");
            s.execute("edit.redo", json!({})).unwrap();
            assert_eq!(damage(s), Some(d), "redo {id}");
        };
        step(&mut s, "layer.duplicate", json!({"layer": a.0}));
        s.execute("layer.select", json!({"layer": a.0})).unwrap();
        s.execute("select.rect", json!({"x": 12, "y": 12, "width": 4, "height": 4})).unwrap();
        s.execute("edit.copy", json!({})).unwrap();
        s.execute("select.deselect", json!({})).unwrap();
        step(&mut s, "edit.paste", json!({}));
        step(&mut s, "layer.delete", json!({"layer": a.0}));
    }

    #[test]
    fn the_move_preview_matches_the_move() {
        let mut s = session(8);
        let a = rect_layer(&mut s, Rect::new(10, 10, 20, 20));
        let before = s.active().unwrap().doc.clone();
        let ids = move_targets(&before, &[a]);
        let shown = moved(&before, &ids, 7, 4).unwrap();
        s.execute("layer.translate", json!({"layer": a.0, "dx": 7, "dy": 4})).unwrap();
        assert_eq!(layer_bounds(shown.layer(a).unwrap()), Some(bounds(&s, a)));
        // Locked layers don't move in the preview either.
        let mut locked = (*before).clone();
        locked.layer_mut(a).unwrap().locks.position = true;
        assert!(moved(&locked, &ids, 1, 1).is_err());
        assert!(moved(&before, &[LayerId(u64::MAX)], 1, 1).is_err());
    }

    #[test]
    fn move_stroked_shape_across_canvas_edge_matches_fresh_render() {
        let mut s = session(8);
        let id = LayerId(
            s.execute("shape.create", json!({"kind": "rect", "rect": [10, 10, 30, 30], "stroke": {"width": 4, "color": "#ff0000"}})).unwrap()["layer"]
                .as_u64()
                .unwrap(),
        );
        let before = doc(&s).clone();
        let ids = move_targets(&before, &[id]);

        // Drag the stroked shape across the left canvas boundary:
        let moved_doc = moved(&before, &ids, -25, 0).unwrap();
        let moved_layer = moved_doc.layer(id).unwrap();
        let LayerContent::Shape(sh) = &moved_layer.content else { panic!("not a shape") };
        let cached = sh.cache.as_ref().expect("cached render");

        let mut fresh_sh = sh.clone();
        crate::vector_cmds::refresh_shape(&moved_doc, &mut fresh_sh);
        let fresh = fresh_sh.cache.as_ref().expect("fresh render");

        assert_eq!(cached, fresh, "dragged shape cached render must match fresh render across canvas edge");

        // Drag it back inside the canvas:
        let moved_back = moved(&moved_doc, &ids, 30, 0).unwrap();
        let back_layer = moved_back.layer(id).unwrap();
        let LayerContent::Shape(back_sh) = &back_layer.content else { panic!("not a shape") };
        let back_cached = back_sh.cache.as_ref().expect("cached render");

        let mut fresh_back = back_sh.clone();
        crate::vector_cmds::refresh_shape(&moved_back, &mut fresh_back);
        let fresh_back_render = fresh_back.cache.as_ref().expect("fresh render");

        assert_eq!(back_cached, fresh_back_render, "cached render must match fresh render when dragged back");
    }

    fn select_all(s: &mut Session, ids: &[LayerId]) {
        s.execute("layer.select", json!({"layer": ids[0].0})).unwrap();
        for id in &ids[1..] {
            s.execute("layer.select", json!({"layer": id.0, "mode": "toggle"})).unwrap();
        }
    }

    #[test]
    fn toggle_and_range_selection() {
        let mut s = session(8);
        let a = rect_layer(&mut s, Rect::new(0, 0, 5, 5));
        let b = rect_layer(&mut s, Rect::new(0, 0, 5, 5));
        let c = rect_layer(&mut s, Rect::new(0, 0, 5, 5));
        let d = rect_layer(&mut s, Rect::new(0, 0, 5, 5));
        assert_eq!(sel(&s), vec![d]);
        s.execute("layer.select", json!({"layer": b.0, "mode": "toggle"})).unwrap();
        assert_eq!(sel(&s), vec![b, d]);
        assert_eq!(s.active().unwrap().active_layer, Some(b));
        // Toggling the active layer off hands "active" to a remaining member.
        s.execute("layer.select", json!({"layer": b.0, "mode": "toggle"})).unwrap();
        assert_eq!(sel(&s), vec![d]);
        assert_eq!(s.active().unwrap().active_layer, Some(d));
        // The last selected layer can't be toggled off.
        s.execute("layer.select", json!({"layer": d.0, "mode": "toggle"})).unwrap();
        assert_eq!(sel(&s), vec![d]);
        // Range from the anchor (a) to c.
        s.execute("layer.select", json!({"layer": a.0})).unwrap();
        s.execute("layer.select", json!({"layer": c.0, "mode": "range"})).unwrap();
        assert_eq!(sel(&s), vec![a, b, c]);
        // A second ⇧-click re-ranges from the same anchor.
        s.execute("layer.select", json!({"layer": b.0, "mode": "range"})).unwrap();
        assert_eq!(sel(&s), vec![a, b]);
        assert!(s.execute("layer.select", json!({"layer": b.0, "mode": "bogus"})).is_err());
        // Selection changes are not edits.
        let history = s.active().unwrap().history.entries().len();
        s.execute("select.allLayers", json!({})).unwrap();
        assert_eq!(sel(&s), vec![a, b, c, d], "Select All Layers skips the locked Background");
        s.execute("select.deselectLayers", json!({})).unwrap();
        assert!(sel(&s).is_empty());
        assert_eq!(s.active().unwrap().history.entries().len(), history);
        // Nothing selected: a no-op, not an error (#2457).
        s.execute("select.deselectLayers", json!({})).unwrap();
        assert_eq!(s.active().unwrap().history.entries().len(), history);
        // An explicit layer needs no active layer.
        s.execute("layer.hideLayers", json!({"layer": a.0})).unwrap();
        assert!(!s.active().unwrap().doc.layer(a).unwrap().visible);
        s.execute("layer.renameLayer", json!({"layer": a.0, "name": "x"})).unwrap();
        assert_eq!(s.active().unwrap().doc.layer(a).unwrap().name, "x");
        s.execute("layer.showLayers", json!({"layer": a.0})).unwrap();
        assert!(s.active().unwrap().doc.layer(a).unwrap().visible);
        assert!(s.execute("layer.hideLayers", json!({"layer": u64::MAX})).is_err());
        assert!(s.execute("layer.hideLayers", json!({})).is_err());
        assert!(!s.is_enabled("layer.align.topEdges"));
    }

    #[test]
    fn find_layers_by_name() {
        let mut s = session(8);
        let a = rect_layer(&mut s, Rect::new(0, 0, 5, 5));
        let b = rect_layer(&mut s, Rect::new(0, 0, 5, 5));
        s.execute("layer.renameLayer", json!({"layer": a.0, "name": "Sky photo"})).unwrap();
        s.execute("layer.renameLayer", json!({"layer": b.0, "name": "Photo of sea"})).unwrap();
        assert_eq!(doc(&s).layer(a).unwrap().name, "Sky photo");
        let r = s.execute("select.findLayers", json!({"name": "PHOTO"})).unwrap();
        assert_eq!(r["selected"], json!([a.0, b.0]));
        assert!(s.execute("select.findLayers", json!({"name": "nothing"})).is_err());
        assert!(s.execute("layer.renameLayer", json!({})).is_err());
    }

    #[test]
    fn rename_layer_keeps_surrounding_spaces() {
        let mut s = session(8);
        let a = rect_layer(&mut s, Rect::new(0, 0, 5, 5));
        s.execute("layer.renameLayer", json!({"layer": a.0, "name": " black skin bw "})).unwrap();
        assert_eq!(doc(&s).layer(a).unwrap().name, " black skin bw ");
        assert!(s.execute("layer.renameLayer", json!({"layer": a.0, "name": "   "})).is_err());
        assert_eq!(doc(&s).layer(a).unwrap().name, " black skin bw ");
    }

    #[test]
    fn align_each_edge() {
        for depth in [8, 16, 32] {
            let cases: [(&str, [Rect; 2]); 6] = [
                ("topEdges", [Rect::new(10, 5, 20, 15), Rect::new(40, 5, 50, 25)]),
                ("bottomEdges", [Rect::new(10, 20, 20, 30), Rect::new(40, 10, 50, 30)]),
                // Union 5..30: centre 17.5; half-pixel offsets round towards the top.
                ("verticalCenters", [Rect::new(10, 12, 20, 22), Rect::new(40, 7, 50, 27)]),
                ("leftEdges", [Rect::new(10, 5, 20, 15), Rect::new(10, 10, 50, 30)]),
                ("rightEdges", [Rect::new(40, 5, 50, 15), Rect::new(10, 10, 50, 30)]),
                ("horizontalCenters", [Rect::new(25, 5, 35, 15), Rect::new(10, 10, 50, 30)]),
            ];
            for (kind, want) in cases {
                let mut s = session(depth);
                let a = rect_layer(&mut s, Rect::new(10, 5, 20, 15));
                let b = rect_layer(&mut s, Rect::new(40, 10, 50, 30));
                // For the horizontal cases make b wide so it defines the target bounds.
                if !matches!(kind, "topEdges" | "bottomEdges" | "verticalCenters") {
                    s.edit("wide", |doc, _| {
                        doc.layer_mut(b).unwrap().surface_mut().unwrap().fill_rect(Rect::new(10, 10, 50, 30), &[0.0, 1.0, 0.0, 1.0]);
                        Ok(())
                    })
                    .unwrap();
                }
                select_all(&mut s, &[a, b]);
                s.execute(&format!("layer.align.{kind}"), json!({})).unwrap();
                assert_eq!([bounds(&s, a), bounds(&s, b)], want, "{kind} at {depth}-bit");
                // Pixels moved, not just bounds.
                let r = want[0];
                assert_eq!(doc(&s).layer(a).unwrap().surface().unwrap().pixel(r.x0, r.y0)[..4], [1.0, 0.0, 0.0, 1.0]);
                // One undo restores the original positions.
                s.undo();
                assert_eq!(bounds(&s, a), Rect::new(10, 5, 20, 15));
            }
        }
    }

    #[test]
    fn align_single_layer_to_selection() {
        for depth in [8, 16, 32] {
            let mut s = session(depth);
            let a = rect_layer(&mut s, Rect::new(10, 10, 20, 20));
            assert!(s.is_enabled("layer.align.leftEdges"), "one layer aligns to the canvas");
            s.execute("select.rect", json!({"x": 50, "y": 40, "width": 30, "height": 30})).unwrap();
            assert!(s.is_enabled("layer.align.leftEdges"));
            s.execute("layer.align.rightEdges", json!({})).unwrap();
            assert_eq!(bounds(&s, a), Rect::new(70, 10, 80, 20));
            s.execute("layer.align.verticalCenters", json!({})).unwrap();
            assert_eq!(bounds(&s, a), Rect::new(70, 50, 80, 60));
            s.execute("layer.align.topEdges", json!({"to": "canvas"})).unwrap();
            assert_eq!(bounds(&s, a), Rect::new(70, 0, 80, 10));
        }
    }

    #[test]
    fn align_single_layer_to_canvas() {
        // Photoshop: a single selected layer with no active selection aligns to the canvas.
        let mut s = session(8); // 100 x 80
        let a = rect_layer(&mut s, Rect::new(10, 10, 20, 20)); // 10 x 10
        assert!(s.is_enabled("layer.align.horizontalCenters"));
        s.execute("layer.align.horizontalCenters", json!({})).unwrap();
        assert_eq!(bounds(&s, a), Rect::new(45, 10, 55, 20), "centred on the 100px-wide canvas");
        s.execute("layer.align.verticalCenters", json!({})).unwrap();
        assert_eq!(bounds(&s, a), Rect::new(45, 35, 55, 45), "centred on the 80px-tall canvas");
        s.execute("layer.align.leftEdges", json!({})).unwrap();
        assert_eq!(bounds(&s, a), Rect::new(0, 35, 10, 45), "left edge to the canvas");
    }

    #[test]
    fn distribute_uneven_layers() {
        for depth in [8, 16, 32] {
            let mut s = session(depth);
            let a = rect_layer(&mut s, Rect::new(0, 0, 10, 10));
            let b = rect_layer(&mut s, Rect::new(12, 0, 32, 10)); // 20 wide
            let c = rect_layer(&mut s, Rect::new(70, 0, 100, 10)); // 30 wide
            select_all(&mut s, &[a, b, c]);
            // Undo targets the layers its state had when it was created (#495): reselect.
            assert!(s.is_enabled("layer.distribute.leftEdges"));
            s.execute("layer.distribute.leftEdges", json!({})).unwrap();
            assert_eq!(bounds(&s, b), Rect::new(35, 0, 55, 10), "left edges 0, 35, 70");
            s.undo();
            select_all(&mut s, &[a, b, c]);
            s.execute("layer.distribute.horizontalCenters", json!({})).unwrap();
            // centres 5 and 85 → middle centre 45
            assert_eq!(bounds(&s, b), Rect::new(35, 0, 55, 10));
            s.undo();
            select_all(&mut s, &[a, b, c]);
            s.execute("layer.distribute.rightEdges", json!({})).unwrap();
            // right edges 10 and 100 → 55
            assert_eq!(bounds(&s, b), Rect::new(35, 0, 55, 10));
            s.undo();
            select_all(&mut s, &[a, b, c]);
            s.execute("layer.distribute.horizontally", json!({})).unwrap();
            // span 100, widths 60 → gaps of 20: b at 30..50
            assert_eq!(bounds(&s, b), Rect::new(30, 0, 50, 10));
            assert_eq!(bounds(&s, a), Rect::new(0, 0, 10, 10));
            assert_eq!(bounds(&s, c), Rect::new(70, 0, 100, 10));
            s.undo();
            assert_eq!(bounds(&s, b), Rect::new(12, 0, 32, 10));
        }
        // Vertical: tall layers with uneven heights.
        let mut s = session(8);
        let a = rect_layer(&mut s, Rect::new(0, 0, 10, 10));
        let b = rect_layer(&mut s, Rect::new(0, 50, 10, 54));
        let c = rect_layer(&mut s, Rect::new(0, 60, 10, 80));
        select_all(&mut s, &[a, b, c]);
        s.execute("layer.distribute.vertically", json!({})).unwrap();
        // span 80, heights 34 → gaps 23: b at 33..37
        assert_eq!(bounds(&s, b), Rect::new(0, 33, 10, 37));
        s.undo();
        select_all(&mut s, &[a, b, c]);
        s.execute("layer.distribute.topEdges", json!({})).unwrap();
        assert_eq!(bounds(&s, b), Rect::new(0, 30, 10, 34));
        s.undo();
        select_all(&mut s, &[a, b, c]);
        s.execute("layer.distribute.bottomEdges", json!({})).unwrap();
        assert_eq!(bounds(&s, b), Rect::new(0, 41, 10, 45));
        s.undo();
        select_all(&mut s, &[a, b, c]);
        s.execute("layer.distribute.verticalCenters", json!({})).unwrap();
        // centres 5 and 70 → 37.5, b is 4 tall → top 35.5 → rounds to 36 (35 for ties down)
        let y0 = bounds(&s, b).y0;
        assert!((35..=36).contains(&y0), "{y0}");
        // Two layers can't be distributed.
        select_all(&mut s, &[a, b]);
        assert!(!s.is_enabled("layer.distribute.leftEdges"));
    }

    #[test]
    fn merge_selected_into_top_most() {
        for depth in [8, 16, 32] {
            let mut s = session(depth);
            let a = rect_layer(&mut s, Rect::new(0, 0, 10, 10));
            let b = rect_layer(&mut s, Rect::new(20, 0, 30, 10));
            let c = rect_layer(&mut s, Rect::new(40, 0, 50, 10));
            s.execute("layer.renameLayer", json!({"layer": c.0, "name": "Top"})).unwrap();
            select_all(&mut s, &[a, c]);
            let before = doc(&s).layer_count();
            let m = LayerId(s.execute("layer.mergeLayers", json!({})).unwrap()["layer"].as_u64().unwrap());
            let d = doc(&s);
            assert_eq!(d.layer_count(), before - 1);
            let ml = d.layer(m).unwrap();
            assert_eq!(ml.name, "Top");
            assert_eq!(layer_bounds(ml).unwrap(), Rect::new(0, 0, 50, 10));
            assert_eq!(ml.surface().unwrap().pixel(5, 5)[..4], [1.0, 0.0, 0.0, 1.0]);
            assert_eq!(ml.surface().unwrap().pixel(25, 5)[3], 0.0, "b was not merged");
            assert!(d.layer(b).is_some() && d.layer(a).is_none());
            assert_eq!(d.layers.last().unwrap().id, m, "merged layer sits where the top-most was");
            assert_eq!(sel(&s), vec![m]);
            s.undo();
            assert_eq!(doc(&s).layer_count(), before);
            assert_eq!(sel(&s), vec![a, c], "undo selects the layers that were merged");
            // One layer selected: Merge Down.
            s.execute("layer.select", json!({"layer": c.0})).unwrap();
            s.execute("layer.mergeLayers", json!({})).unwrap();
            assert_eq!(doc(&s).layer_count(), before - 1);
        }
    }

    #[test]
    fn merging_group_children_preserves_their_composite() {
        for depth in [8, 16, 32] {
            let mut s = session(depth);
            let lower = rect_layer(&mut s, Rect::new(10, 12, 40, 42));
            let upper = rect_layer(&mut s, Rect::new(25, 20, 55, 50));
            select_all(&mut s, &[lower, upper]);
            let group = LayerId(s.execute("layer.groupLayers", json!({"name": "Merge test group"})).unwrap()["layer"].as_u64().unwrap());
            let doc_before = doc(&s);
            let expected = photocraft_compose::render(doc_before, doc_before.bounds());
            let children = doc_before.layer(group).unwrap().children().unwrap().iter().map(|layer| layer.id).collect::<Vec<_>>();

            select_all(&mut s, &children);
            s.execute("layer.mergeLayers", json!({})).unwrap();
            let doc_after = doc(&s);
            let actual = photocraft_compose::render(doc_after, doc_after.bounds());
            assert_eq!(actual, expected, "depth {depth}: group merge changed rendered pixels");
            assert_eq!(doc_after.layer(group).unwrap().children().unwrap().len(), 1);
        }
    }

    #[test]
    fn merging_keeps_layer_effects_outside_the_pixels() {
        // A drop shadow reaches past the layer's pixels; the merge must render that far.
        for depth in [8, 16, 32] {
            let mut s = session(depth);
            let lower = rect_layer(&mut s, Rect::new(10, 10, 30, 30));
            let upper = rect_layer(&mut s, Rect::new(40, 20, 60, 40));
            s.execute("layer.layerStyle.dropShadow", json!({"layer": upper.0, "blend": "normal", "distance": 12, "size": 4})).unwrap();
            let before = photocraft_compose::render(doc(&s), doc(&s).bounds());
            select_all(&mut s, &[lower, upper]);
            s.execute("layer.mergeLayers", json!({})).unwrap();
            let after = photocraft_compose::render(doc(&s), doc(&s).bounds());
            let worst = before.px.iter().zip(&after.px).map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)).fold(0.0f32, f32::max);
            assert!(worst <= 1.5 / 255.0, "depth {depth}: merge layers dropped the shadow (off by {worst})");
        }
    }

    #[test]
    fn merging_a_group_down_preserves_the_composite() {
        for depth in [8, 16, 32] {
            let mut s = session(depth);
            let lower = rect_layer(&mut s, Rect::new(10, 12, 50, 52));
            let child_a = rect_layer(&mut s, Rect::new(24, 20, 64, 60));
            let child_b = rect_layer(&mut s, Rect::new(35, 28, 75, 68));
            select_all(&mut s, &[child_a, child_b]);
            let group = LayerId(s.execute("layer.groupLayers", json!({"name": "Merge down group"})).unwrap()["layer"].as_u64().unwrap());
            let doc_before = doc(&s);
            let expected = photocraft_compose::render(doc_before, doc_before.bounds());

            s.execute("layer.mergeDown", json!({"layer": group.0})).unwrap();
            let doc_after = doc(&s);
            let actual = photocraft_compose::render(doc_after, doc_after.bounds());
            assert_eq!(actual, expected, "depth {depth}: group merge down changed rendered pixels");
            assert!(doc_after.layer(lower).is_some());
            assert!(doc_after.layer(group).is_none());
        }
    }

    #[test]
    fn a_locked_group_locks_its_contents() {
        let mut s = session(8);
        let a = rect_layer(&mut s, Rect::new(0, 0, 5, 5));
        let g = s.execute("layer.new.groupFromLayers", json!({})).unwrap()["layer"].as_u64().unwrap();
        s.execute("layer.setProps", json!({"layer": g, "locked": true})).unwrap();
        s.execute("layer.select", json!({"layer": a.0})).unwrap();
        for (cmd, p) in [
            ("layer.translate", json!({"dx": 1})),
            ("paint.stroke", json!({"points": [[2, 2], [4, 4]]})),
            ("edit.fill", json!({"color": "#00ff00"})),
            ("edit.transform.flipHorizontal", json!({})),
        ] {
            assert!(s.execute(cmd, p).is_err(), "{cmd} edited a layer in a locked group");
        }
        // Unlocking the group releases its contents.
        s.execute("layer.setProps", json!({"layer": g, "locked": false})).unwrap();
        s.execute("layer.translate", json!({"dx": 1})).unwrap();
        assert_eq!(bounds(&s, a), Rect::new(1, 0, 6, 5));
    }

    #[test]
    fn a_locked_group_keeps_a_grouped_background_locked() {
        let mut s = session(8);
        let bg = doc(&s).layers[0].id;
        s.execute("layer.select", json!({"layer": bg.0})).unwrap();
        let g = s.execute("layer.new.groupFromLayers", json!({})).unwrap()["layer"].as_u64().unwrap();
        s.execute("layer.setProps", json!({"layer": g, "locked": true})).unwrap();
        s.execute("layer.select", json!({"layer": bg.0})).unwrap();
        for (cmd, p) in [
            ("edit.transform.flipHorizontal", json!({})),
            ("edit.transform.warp", json!({"style": "flag", "bend": 50})),
            ("edit.puppetWarp", json!({"pins": [{"src": [30, 25], "dst": [36, 29]}]})),
        ] {
            let e = s.execute(cmd, p).unwrap_err().to_string();
            assert!(e.contains("locked"), "{cmd}: {e}");
            assert_eq!(doc(&s).layer(bg).unwrap().name, "Background", "{cmd}");
        }
    }

    #[test]
    fn group_from_layers_preserves_order() {
        let mut s = session(8);
        let a = rect_layer(&mut s, Rect::new(0, 0, 5, 5));
        let b = rect_layer(&mut s, Rect::new(0, 0, 5, 5));
        let c = rect_layer(&mut s, Rect::new(0, 0, 5, 5));
        let d = rect_layer(&mut s, Rect::new(0, 0, 5, 5));
        select_all(&mut s, &[c, a]);
        let g = LayerId(s.execute("layer.new.groupFromLayers", json!({"name": "Pair"})).unwrap()["layer"].as_u64().unwrap());
        let docu = doc(&s);
        let gl = docu.layer(g).unwrap();
        assert_eq!(gl.name, "Pair");
        assert_eq!(gl.children().unwrap().iter().map(|l| l.id).collect::<Vec<_>>(), vec![a, c]);
        // Root (bottom-to-top): Background, b, Pair (where c was), d.
        assert_eq!(docu.layers.iter().map(|l| l.id).skip(1).collect::<Vec<_>>(), vec![b, g, d]);
        assert_eq!(sel(&s), vec![g]);
        s.undo();
        assert_eq!(doc(&s).layers.len(), 5);
        // ⌘G with several selected groups them too.
        select_all(&mut s, &[b, d]);
        s.execute("layer.groupLayers", json!({})).unwrap();
        assert_eq!(doc(&s).layers.len(), 4);
    }

    #[test]
    fn grouping_is_capped_at_the_document_nesting_limit() {
        let mut s = session(8);
        let a = rect_layer(&mut s, Rect::new(0, 0, 5, 5));
        select_all(&mut s, &[a]);
        for _ in 0..photocraft_doc::MAX_GROUP_DEPTH {
            s.execute("layer.groupLayers", json!({})).unwrap();
        }
        assert_eq!(doc(&s).max_group_depth(), photocraft_doc::MAX_GROUP_DEPTH);
        // One more wrapping group would pass the cap: rejected, document untouched.
        let err = s.execute("layer.groupLayers", json!({})).unwrap_err();
        assert!(err.to_string().contains("deeper than 100"), "{err}");
        assert_eq!(doc(&s).max_group_depth(), photocraft_doc::MAX_GROUP_DEPTH);
        // The rejected call recorded no history step: undo still lands one grouping earlier.
        s.undo();
        assert_eq!(doc(&s).max_group_depth(), photocraft_doc::MAX_GROUP_DEPTH - 1);
    }

    #[test]
    fn select_linked_layers_excludes_an_unlinked_active_layer() {
        for depth in [8, 16, 32] {
            let mut s = session(depth);
            let a = rect_layer(&mut s, Rect::new(0, 0, 10, 10));
            let b = rect_layer(&mut s, Rect::new(20, 20, 30, 30));
            let other = rect_layer(&mut s, Rect::new(50, 50, 60, 60));
            select_all(&mut s, &[a, b]);
            s.execute("layer.linkLayers", json!({})).unwrap();
            s.execute("layer.select", json!({"layer": a.0})).unwrap();
            s.execute("layer.select", json!({"layer": other.0, "mode": "toggle"})).unwrap();
            assert_eq!(sel(&s), vec![a, other]);
            assert_eq!(s.active().unwrap().active_layer, Some(other));
            let past = s.active().unwrap().history.past_len();
            let dirty = s.active().unwrap().is_dirty();

            let result = s.execute("layer.selectLinkedLayers", json!({})).unwrap();
            assert_eq!(sel(&s), vec![a, b]);
            assert_eq!(result["selected"], sel(&s).len());
            assert_eq!(s.active().unwrap().active_layer, Some(b));
            assert_eq!(s.active().unwrap().layer_anchor, Some(b));
            assert_eq!(s.active().unwrap().history.past_len(), past);
            assert_eq!(s.active().unwrap().is_dirty(), dirty);

            // A following multi-layer delete must leave the unrelated layer intact.
            s.execute("layer.delete", json!({})).unwrap();
            assert!(doc(&s).layer(a).is_none() && doc(&s).layer(b).is_none());
            assert!(doc(&s).layer(other).is_some());
            s.undo();
            assert!(doc(&s).layer(a).is_some() && doc(&s).layer(b).is_some());
            assert!(doc(&s).layer(other).is_some());
        }
    }

    #[test]
    fn select_linked_layers_preserves_a_linked_active_layer() {
        for depth in [8, 16, 32] {
            let mut s = session(depth);
            let a = rect_layer(&mut s, Rect::new(0, 0, 10, 10));
            let b = rect_layer(&mut s, Rect::new(20, 20, 30, 30));
            select_all(&mut s, &[a, b]);
            s.execute("layer.linkLayers", json!({})).unwrap();
            for active in [a, b] {
                s.execute("layer.select", json!({"layer": active.0})).unwrap();
                let st = s.active_mut().unwrap();
                st.saved_revision = st.revision;
                let past = st.history.past_len();
                for _ in 0..2 {
                    let result = s.execute("layer.selectLinkedLayers", json!({})).unwrap();
                    assert_eq!(sel(&s), vec![a, b]);
                    assert_eq!(result["selected"], sel(&s).len());
                    assert_eq!(s.active().unwrap().active_layer, Some(active));
                    assert_eq!(s.active().unwrap().layer_anchor, Some(active));
                    assert_eq!(s.active().unwrap().history.past_len(), past);
                    assert!(!s.active().unwrap().is_dirty());
                }
            }
        }
    }

    #[test]
    fn select_linked_layers_requires_a_linked_selection() {
        let mut s = Session::new();
        assert!(!s.is_enabled("layer.selectLinkedLayers"));
        assert!(s.execute("layer.selectLinkedLayers", json!({})).is_err());
        let mut s = session(8);
        let before = sel(&s);
        assert!(!s.is_enabled("layer.selectLinkedLayers"));
        assert!(s.execute("layer.selectLinkedLayers", json!({})).is_err());
        assert_eq!(sel(&s), before);
    }

    #[test]
    fn translate_rejects_offsets_that_would_wrap() {
        let mut s = session(8);
        let a = rect_layer(&mut s, Rect::new(0, 0, 5, 5));
        select_all(&mut s, &[a]);
        // 2^32 + 50 wrapped to `dx = 50` and 3e9 to a negative offset through `as i32`.
        for dx in [4_294_967_346_i64, 3_000_000_000_i64] {
            let err = s.execute("layer.translate", json!({"dx": dx, "dy": 0})).unwrap_err();
            assert!(err.to_string().contains("32-bit"), "{err}");
        }
        assert_eq!(bounds(&s, a), Rect::new(0, 0, 5, 5), "the layer never moved");
        // Large in-range offsets still work.
        s.execute("layer.translate", json!({"dx": -200_000, "dy": 200_000})).unwrap();
    }

    #[test]
    fn link_layers_never_reuses_a_group_when_the_largest_id_is_u64_max() {
        let mut s = session(8);
        let a = rect_layer(&mut s, Rect::new(0, 0, 10, 10));
        let b = rect_layer(&mut s, Rect::new(20, 20, 30, 30));
        let c = rect_layer(&mut s, Rect::new(50, 50, 60, 60));
        let d = rect_layer(&mut s, Rect::new(70, 70, 80, 80));
        // A stored document can carry any id; `max + 1` once overflowed here.
        let st = s.active_mut().unwrap();
        for id in [c, d] {
            std::sync::Arc::make_mut(&mut st.doc).layer_mut(id).unwrap().link_group = Some(u64::MAX);
        }
        select_all(&mut s, &[a, b]);
        assert_eq!(s.execute("layer.linkLayers", json!({})).unwrap()["linked"], true);
        let g = doc(&s).layer(a).unwrap().link_group;
        assert_eq!(g, Some(1));
        assert_eq!(doc(&s).layer(b).unwrap().link_group, g);
        for id in [c, d] {
            assert_eq!(doc(&s).layer(id).unwrap().link_group, Some(u64::MAX), "the existing group is untouched");
        }
    }

    #[test]
    fn undo_and_redo_report_only_the_edited_area() {
        let mut s = session(8);
        let a = rect_layer(&mut s, Rect::new(10, 10, 20, 20));
        select_all(&mut s, &[a]);
        // A pixel edit on one layer is bounded by that layer's extent on both sides.
        s.edit("paint", |doc, _| {
            doc.layer_mut(a).unwrap().surface_mut().unwrap().fill_rect(Rect::new(12, 12, 14, 14), &[0.0, 0.0, 1.0, 1.0]);
            Ok(())
        })
        .unwrap();
        s.undo();
        assert_eq!(damage(&s), Some(Rect::new(10, 10, 20, 20)));
        s.redo();
        assert_eq!(damage(&s), Some(Rect::new(10, 10, 20, 20)));
        // A move reports where the layer was and where it landed.
        s.execute("layer.translate", json!({"dx": 5, "dy": 0})).unwrap();
        s.undo();
        assert_eq!(damage(&s), Some(Rect::new(10, 10, 25, 20)), "union of where the layer was and is");
        s.redo();
        assert_eq!(damage(&s), Some(Rect::new(10, 10, 25, 20)));
    }

    #[test]
    fn undo_of_document_wide_steps_still_composites_everything() {
        let mut s = session(8);
        let _ = rect_layer(&mut s, Rect::new(10, 10, 20, 20));
        // A canvas change has no bounded damage.
        s.execute("image.crop", json!({"x": 0, "y": 0, "width": 30, "height": 25})).unwrap();
        assert_ne!(doc(&s).size, photocraft_doc::Size::new(100, 80));
        s.undo();
        assert_eq!(damage(&s), None, "a size change recomposites the whole canvas");
        // A layer removed and restored changes only where it draws.
        let b = rect_layer(&mut s, Rect::new(0, 0, 5, 5));
        select_all(&mut s, &[b]);
        s.execute("layer.delete", json!({})).unwrap();
        s.undo();
        assert_eq!(damage(&s), Some(Rect::new(0, 0, 5, 5)));
    }

    #[test]
    fn undo_of_compositor_read_document_fields_composites_everything() {
        let mut s = session(8);
        let _ = rect_layer(&mut s, Rect::new(10, 10, 20, 20));
        // Global Light changes every effect's angle across the document.
        s.execute("layer.layerStyle.globalLight", json!({"angle": 90.0})).unwrap();
        s.undo();
        assert_eq!(damage(&s), None, "a global-light change recomposites everything");
        // A mode change that keeps the pixel format (RGB → Indexed maps to RGB) neither.
        s.execute("image.mode.indexedColor", json!({})).unwrap();
        s.undo();
        assert_eq!(damage(&s), None, "a mode change recomposites everything");
    }

    #[test]
    fn undo_of_clipping_or_channel_exclusion_composites_everything() {
        let mut s = session(8);
        let _ = rect_layer(&mut s, Rect::new(10, 10, 20, 20));
        let b = rect_layer(&mut s, Rect::new(12, 12, 30, 30));
        select_all(&mut s, &[b]);
        // Clipping reaches the layers clipped above, outside this layer's bounds.
        s.execute("layer.setProps", json!({"clipped": true})).unwrap();
        s.undo();
        assert_eq!(damage(&s), None, "a clipping change recomposites everything");
        s.execute("layer.setProps", json!({"channels": [true, false, true]})).unwrap();
        s.undo();
        assert_eq!(damage(&s), None, "an excluded-channel change recomposites everything");
    }

    #[test]
    fn linked_layers_move_together_and_undo() {
        for depth in [8, 16, 32] {
            let mut s = session(depth);
            let a = rect_layer(&mut s, Rect::new(0, 0, 10, 10));
            let b = rect_layer(&mut s, Rect::new(20, 20, 30, 30));
            let c = rect_layer(&mut s, Rect::new(50, 50, 60, 60));
            assert!(!s.is_enabled("layer.linkLayers"));
            select_all(&mut s, &[a, b]);
            assert_eq!(s.execute("layer.linkLayers", json!({})).unwrap()["linked"], true);
            let g = doc(&s).layer(a).unwrap().link_group;
            assert!(g.is_some() && doc(&s).layer(b).unwrap().link_group == g);
            // Move only a: b follows, c stays.
            s.execute("layer.select", json!({"layer": a.0})).unwrap();
            s.execute("layer.translate", json!({"dx": 5, "dy": 3})).unwrap();
            assert_eq!(bounds(&s, a), Rect::new(5, 3, 15, 13));
            assert_eq!(bounds(&s, b), Rect::new(25, 23, 35, 33));
            assert_eq!(bounds(&s, c), Rect::new(50, 50, 60, 60));
            s.undo();
            assert_eq!(bounds(&s, a), Rect::new(0, 0, 10, 10));
            assert_eq!(bounds(&s, b), Rect::new(20, 20, 30, 30));
            // Select Linked Layers from just a.
            s.execute("layer.selectLinkedLayers", json!({})).unwrap();
            assert_eq!(sel(&s), vec![a, b]);
            // Multi-selection translate moves all selected (c added, unlinked).
            s.execute("layer.select", json!({"layer": c.0, "mode": "toggle"})).unwrap();
            s.execute("layer.translate", json!({"dx": -1, "dy": 0})).unwrap();
            assert_eq!(bounds(&s, c), Rect::new(49, 50, 59, 60));
            assert_eq!(bounds(&s, a), Rect::new(-1, 0, 9, 10));
            // Toggle: linking the same group again unlinks.
            select_all(&mut s, &[a, b]);
            assert_eq!(s.execute("layer.linkLayers", json!({})).unwrap()["linked"], false);
            assert!(doc(&s).layer(a).unwrap().link_group.is_none());
        }
    }

    /// Plain layers carrying stored link groups, as a loaded `.pcraft` or PSD can hold (#961).
    fn stored_groups(s: &mut Session, groups: &[Option<u64>]) -> Vec<LayerId> {
        let ids: Vec<LayerId> = groups.iter().map(|_| rect_layer(s, Rect::new(0, 0, 4, 4))).collect();
        s.edit("load groups", |doc, _| {
            for (id, g) in ids.iter().zip(groups) {
                doc.layer_mut(*id).unwrap().link_group = *g;
            }
            Ok(())
        })
        .unwrap();
        ids
    }

    fn group(s: &Session, id: LayerId) -> Option<u64> {
        doc(s).layer(id).unwrap().link_group
    }

    /// Links `ids`, then checks they share one group no other layer uses (#961).
    fn link_fresh(s: &mut Session, ids: &[LayerId]) -> u64 {
        select_all(s, ids);
        let past = s.active().unwrap().history.past_len();
        assert_eq!(s.execute("layer.linkLayers", json!({})).unwrap()["linked"], true);
        assert_eq!(s.active().unwrap().history.past_len(), past + 1);
        let g = group(s, ids[0]).expect("linked");
        assert!(ids.iter().all(|id| group(s, *id) == Some(g)));
        let members = doc(s).walk().iter().filter(|(_, _, l)| l.link_group == Some(g)).count();
        assert_eq!(members, ids.len(), "group {g} is shared with unselected layers");
        g
    }

    #[test]
    fn link_layers_beside_a_maximal_stored_group() {
        // The issue's repro: one layer stored at u64::MAX, link it with an unlinked one.
        let mut s = session(8);
        let ids = stored_groups(&mut s, &[Some(u64::MAX), None]);
        assert_ne!(link_fresh(&mut s, &ids), u64::MAX);
        // Unlink and relink the pair: with no groups left, numbering starts over at 1.
        assert_eq!(s.execute("layer.linkLayers", json!({})).unwrap()["linked"], false);
        assert!(ids.iter().all(|id| group(&s, *id).is_none()));
        assert_eq!(link_fresh(&mut s, &ids), 1);

        // Two other layers linked beside an untouched maximal pair.
        let mut s = session(8);
        let ids = stored_groups(&mut s, &[Some(u64::MAX), Some(u64::MAX), None, None]);
        let g = link_fresh(&mut s, &ids[2..]);
        assert_ne!(g, u64::MAX);
        assert_eq!((group(&s, ids[0]), group(&s, ids[1])), (Some(u64::MAX), Some(u64::MAX)));
    }

    #[test]
    fn unlink_a_maximal_stored_group() {
        let mut s = session(8);
        let ids = stored_groups(&mut s, &[Some(u64::MAX), Some(u64::MAX), None]);
        select_all(&mut s, &ids[..2]);
        assert_eq!(s.execute("layer.linkLayers", json!({})).unwrap()["linked"], false);
        assert!(ids.iter().all(|id| group(&s, *id).is_none()));
        s.undo();
        assert_eq!((group(&s, ids[0]), group(&s, ids[1])), (Some(u64::MAX), Some(u64::MAX)));
        // Relinking all three gets a working group.
        link_fresh(&mut s, &ids);
    }

    #[test]
    fn link_layers_when_every_group_is_huge() {
        // Unselected groups at the top of the range, including the 0 a wrapped id would join.
        let top = [u64::MAX, u64::MAX - 1, u64::MAX - 2, 0, 1, 2];
        let stored: Vec<Option<u64>> = top.iter().flat_map(|g| [Some(*g), Some(*g)]).chain([None, None, None, None]).collect();
        let mut s = session(8);
        let ids = stored_groups(&mut s, &stored);
        let n = top.len() * 2;
        let g = link_fresh(&mut s, &ids[n..n + 2]);
        assert!(!top.contains(&g), "{g} collides with a stored group");
        // A second link gets yet another group, and existing groups never change.
        let h = link_fresh(&mut s, &ids[n + 2..]);
        assert!(!top.contains(&h) && h != g);
        for (i, t) in top.iter().enumerate() {
            assert_eq!((group(&s, ids[2 * i]), group(&s, ids[2 * i + 1])), (Some(*t), Some(*t)));
        }
        // Moving a newly linked layer leaves the group-0 pair alone.
        assert_eq!(with_links(doc(&s), &[ids[n]]), vec![ids[n], ids[n + 1]]);
    }

    #[test]
    fn link_layers_near_the_maximum() {
        // max + 1 still fits: the new group is u64::MAX itself.
        let mut s = session(8);
        let ids = stored_groups(&mut s, &[Some(u64::MAX - 1), Some(u64::MAX - 1), None, None, None, None]);
        assert_eq!(link_fresh(&mut s, &ids[2..4]), u64::MAX);
        // The next link no longer fits above the maximum and takes a free id instead.
        let g = link_fresh(&mut s, &ids[4..]);
        assert!(g != u64::MAX && g != u64::MAX - 1);
    }

    #[test]
    fn link_layers_assigns_working_groups() {
        // Control: ordinary documents keep getting max + 1, starting at 1.
        let mut s = session(8);
        let ids = stored_groups(&mut s, &[None, None, None, None]);
        assert_eq!(link_fresh(&mut s, &ids[..2]), 1);
        assert_eq!(link_fresh(&mut s, &ids[2..]), 2);
        assert_eq!(s.execute("layer.linkLayers", json!({})).unwrap()["linked"], false);
        assert_eq!((group(&s, ids[2]), group(&s, ids[3])), (None, None));
        assert_eq!((group(&s, ids[0]), group(&s, ids[1])), (Some(1), Some(1)));
    }

    #[test]
    fn selection_pruned_after_delete_and_undo() {
        let mut s = session(8);
        let a = rect_layer(&mut s, Rect::new(0, 0, 5, 5));
        let b = rect_layer(&mut s, Rect::new(0, 0, 5, 5));
        let c = rect_layer(&mut s, Rect::new(0, 0, 5, 5));
        select_all(&mut s, &[a, c]);
        // Deleting one layer explicitly prunes it from the set.
        s.execute("layer.delete", json!({"layer": a.0})).unwrap();
        assert_eq!(sel(&s), vec![c]);
        s.undo();
        assert!(sel(&s).iter().all(|id| doc(&s).layer(*id).is_some()));
        // Deleting with several selected removes them all, then selects the layer below.
        select_all(&mut s, &[a, c]);
        s.execute("layer.delete", json!({})).unwrap();
        assert!(doc(&s).layer(a).is_none() && doc(&s).layer(c).is_none());
        assert_eq!(doc(&s).layer_count(), 2);
        assert_eq!(sel(&s).len(), 1);
        assert!(s.active().unwrap().selected_layers.iter().all(|id| doc(&s).layer(*id).is_some()));
        s.undo();
        assert_eq!(doc(&s).layer_count(), 4);
        let _ = b;
        // Redo then undo again keeps the set valid.
        s.redo();
        assert!(sel(&s).iter().all(|id| doc(&s).layer(*id).is_some()));
    }

    #[test]
    fn duplicate_hide_lock_and_reverse_selected() {
        let mut s = session(8);
        let a = rect_layer(&mut s, Rect::new(0, 0, 5, 5));
        let b = rect_layer(&mut s, Rect::new(0, 0, 5, 5));
        let c = rect_layer(&mut s, Rect::new(0, 0, 5, 5));
        select_all(&mut s, &[a, c]);
        let r = s.execute("layer.duplicate", json!({})).unwrap();
        assert_eq!(r["layers"].as_array().unwrap().len(), 2);
        assert_eq!(doc(&s).layer_count(), 6);
        assert_eq!(sel(&s).len(), 2);
        assert!(sel(&s).iter().all(|id| *id != a && *id != c));
        s.undo();
        select_all(&mut s, &[a, b]);
        s.execute("layer.hideLayers", json!({})).unwrap();
        assert!(!doc(&s).layer(a).unwrap().visible && !doc(&s).layer(b).unwrap().visible && doc(&s).layer(c).unwrap().visible);
        s.execute("layer.showLayers", json!({})).unwrap();
        assert!(doc(&s).layer(a).unwrap().visible);
        s.execute("layer.lockLayers", json!({})).unwrap();
        assert!(doc(&s).layer(a).unwrap().locks.all && doc(&s).layer(b).unwrap().locks.all);
        assert!(s.execute("layer.translate", json!({"dx": 1})).is_err(), "locked layers don't move");
        s.execute("layer.lockLayers", json!({})).unwrap();
        assert!(!doc(&s).layer(a).unwrap().locks.all);
        s.execute("layer.lockLayers", json!({"position": true})).unwrap();
        assert!(doc(&s).layer(b).unwrap().locks.position && !doc(&s).layer(c).unwrap().locks.position);
        // Reverse a, b, c.
        select_all(&mut s, &[a, b, c]);
        s.execute("layer.arrange.reverse", json!({})).unwrap();
        assert_eq!(doc(&s).layers.iter().skip(1).map(|l| l.id).collect::<Vec<_>>(), vec![c, b, a]);
        assert_eq!(s.active().unwrap().history.entries().last().map(|e| e.as_str()), Some("Reverse"));
    }

    /// #4892: Lock Image Pixels includes Lock Transparent Pixels, so the two are never both on:
    /// locking the pixels clears the transparency lock, which stays off while they are locked,
    /// through both `layer.setProps` and `layer.lockLayers`.
    #[test]
    fn locking_pixels_clears_and_holds_off_the_transparency_lock() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
        let id = LayerId(s.execute("layer.new.layer", json!({})).unwrap()["layer"].as_u64().unwrap());
        let locks = |s: &Session| doc(s).layer(id).unwrap().locks;
        s.execute("layer.setProps", json!({"locks": {"transparency": true}})).unwrap();
        assert!(locks(&s).transparency);
        s.execute("layer.setProps", json!({"locks": {"pixels": true}})).unwrap();
        assert!(locks(&s).pixels && !locks(&s).transparency, "locking the pixels clears the transparency lock");
        s.execute("layer.setProps", json!({"locks": {"transparency": true}})).unwrap();
        assert!(!locks(&s).transparency, "and it stays off while the pixels are locked");
        s.execute("layer.setProps", json!({"locks": {"transparency": true, "pixels": true}})).unwrap();
        assert!(locks(&s).pixels && !locks(&s).transparency, "both at once: the pixel lock wins");
        s.execute("layer.lockLayers", json!({"pixels": false})).unwrap();
        s.execute("layer.lockLayers", json!({"transparency": true})).unwrap();
        assert!(locks(&s).transparency && !locks(&s).pixels, "with the pixels unlocked it can be set again");
        s.execute("layer.lockLayers", json!({"pixels": true})).unwrap();
        assert!(locks(&s).pixels && !locks(&s).transparency);
        assert!(s.execute("layer.setProps", json!({"locks": {"nope": true}})).is_err());
    }

    #[test]
    fn inspect_reports_selection() {
        let mut s = session(8);
        let a = rect_layer(&mut s, Rect::new(0, 0, 5, 5));
        let b = rect_layer(&mut s, Rect::new(0, 0, 5, 5));
        select_all(&mut s, &[a, b]);
        let v = s.execute("document.inspect", json!({})).unwrap();
        assert_eq!(v["selectedLayers"], json!([a.0, b.0]));
        assert_eq!(v["layers"][0]["selected"], true);
    }
}
