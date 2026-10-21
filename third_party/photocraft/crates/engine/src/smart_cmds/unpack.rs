//! Layer › Smart Objects › Convert to Layers: unpacks a smart object's contents back into layers
//! at its placement, as Photoshop does. Several layers land in a group named after the smart
//! object, which carries its opacity, blend mode, masks and style; a single layer replaces the
//! smart object directly when those fold into it without changing the look. Smart filters are
//! discarded (the result reports how many).

use std::sync::Arc;

use photocraft_algo::transform::Interp;
use photocraft_cms::Intent;
use photocraft_color::BlendMode;
use photocraft_doc::{Document, Group, Layer, LayerContent, LayerId, Metadata, SmartObject, SmartSource};
use serde_json::{Value, json};

use super::{authorize_source, authorized_source_bytes, decode_source, detach_psd, other, placement, shift_layer, smart};
use crate::commands::layer_param;
use crate::{EngineError, Result, Session};

/// The smart object's contents as layers in `doc`'s colour model, profile and depth, still in
/// source pixel coordinates. A Background becomes a normal layer (it can't sit above others).
fn contents(s: &Session, doc: &Document, sm: &SmartObject, intent: Intent, bpc: bool) -> Result<Vec<Layer>> {
    let (name, bytes) = authorized_source_bytes(s, "layer.smartObjects.convertToLayers", &doc.metadata, &sm.source)?
        .ok_or_else(|| other("the smart object's contents are unavailable (missing linked file?)"))?;
    let mut src = decode_source(&name, &bytes)?;
    let mode = doc.pixel_format().mode;
    if src.pixel_format().mode != mode || src.icc_profile != doc.icc_profile {
        crate::color_cmds::convert_document(&mut src, &crate::color_cmds::document_profile(doc), intent, bpc)?;
        if src.pixel_format().mode != mode {
            return Err(other(format!("can't convert the smart object's contents to the document's {:?} mode", doc.mode)));
        }
    }
    if src.depth != doc.depth {
        crate::image_cmds::convert_layers_depth(&mut src.layers, doc.depth);
    }
    let background_id = src.background().map(|l| l.id);
    let mut layers = std::mem::take(&mut src.layers);
    for l in &mut layers {
        if Some(l.id) == background_id {
            l.name = "Layer 0".into();
            l.locks = Default::default();
        }
        embed_nested(&src.metadata, l);
    }
    Ok(layers)
}

/// Nested PSD placed layers find their files in the contents' own global blocks, which stay
/// behind: embed those sources so the nested smart objects keep rendering in the document.
fn embed_nested(meta: &Metadata, l: &mut Layer) {
    let found = match &l.content {
        LayerContent::Smart(SmartObject { source: SmartSource::Linked { path }, .. }) => photocraft_io::linked::find_linked_file(meta, path),
        _ => None,
    };
    if let Some(f) = found {
        if let LayerContent::Smart(sm) = &mut l.content {
            sm.source = SmartSource::Embedded { file_name: f.file_name, bytes: Arc::new(f.bytes) };
        }
        detach_psd(l);
    }
    for c in l.children_mut().into_iter().flatten() {
        embed_nested(meta, c);
    }
}

/// Moves unpacked layers from source pixels to where the smart object placed them: an exact
/// shift for whole-pixel moves, else its transform (type, shapes and nested smart objects stay
/// live; Distort and Perspective need them rasterized, as Free Transform does).
fn place(s: &Session, doc: &Document, sm: &SmartObject, layers: &mut [Layer]) -> Result<()> {
    if sm.warp.as_ref().is_some_and(|w| !w.is_identity()) {
        return Err(other("a warped smart object can't be converted to layers; rasterize it instead (Layer › Smart Objects › Rasterize)"));
    }
    let [a, b, c, d, e, f] = sm.transform.m;
    let whole = |v: f64| (v - v.round()).abs() < 1e-9;
    if sm.perspective.is_none() && [a - 1.0, b, c, d - 1.0].iter().all(|v| v.abs() < 1e-9) && whole(e) && whole(f) {
        // Saturating casts: a placement that far out moves the layers off any canvas either way.
        layers.iter_mut().for_each(|l| shift_layer(l, e.round() as i32, f.round() as i32));
        return Ok(());
    }
    // Unlike a whole-pixel shift, this path re-renders nested smart objects from their sources.
    // Check every read before transforming anything; refresh_text keeps errors as fallback pixels.
    if s.authorize.is_some() {
        let mut pending: Vec<&Layer> = layers.iter().collect();
        while let Some(l) = pending.pop() {
            if let LayerContent::Smart(sm) = &l.content {
                authorize_source(s, "layer.smartObjects.convertToLayers", doc, &sm.source)?;
            }
            pending.extend(l.children().into_iter().flatten());
        }
    }
    let h = placement(sm);
    let affine = sm.perspective.is_none().then_some(sm.transform);
    for l in layers {
        released(l, |l| crate::transform_cmds::transform_layer(None, photocraft_doc::Locks::default(), l, &h, affine, Interp::Bicubic))?;
        crate::transform_cmds::refresh_text(doc, l);
    }
    Ok(())
}

/// Runs `f` with every lock and mask link in `l`'s tree released, so a transform moves all of
/// it (the contents move as one, whatever their locks), then restores them.
fn released(l: &mut Layer, f: impl FnOnce(&mut Layer) -> Result<()>) -> Result<()> {
    let mut saved = Vec::new();
    crate::image_cmds::for_each_layer(std::slice::from_mut(l), &mut |x| {
        saved.push((x.locks, x.mask.as_ref().map(|m| m.linked), x.vector_mask.as_ref().map(|m| m.linked)));
        x.locks = Default::default();
        x.mask.iter_mut().for_each(|m| m.linked = true);
        x.vector_mask.iter_mut().for_each(|m| m.linked = true);
    });
    let r = f(l);
    // A transform keeps the tree's shape, so the walk meets the layers in the same order.
    let mut saved = saved.into_iter();
    crate::image_cmds::for_each_layer(std::slice::from_mut(l), &mut |x| {
        let Some((locks, mask, vector)) = saved.next() else { return };
        x.locks = locks;
        if let (Some(m), Some(linked)) = (x.mask.as_mut(), mask) {
            m.linked = linked;
        }
        if let (Some(m), Some(linked)) = (x.vector_mask.as_mut(), vector) {
            m.linked = linked;
        }
    });
    r
}

/// Whether the smart layer's own properties fold into its one unpacked layer `c` without
/// changing the look: nothing on the smart layer acts on `c`'s composite (masks, style, fill
/// opacity, Blend If, channel exclusions), and `c` blends normally and doesn't look at what's
/// beneath it (alone in the smart object, it lands on transparency).
fn foldable(so: &Layer, c: &Layer) -> bool {
    so.mask.is_none()
        && so.vector_mask.is_none()
        && (!so.effects.enabled || so.effects.items.is_empty())
        && so.fill_opacity >= 1.0
        && so.blend_if.is_default()
        && so.excluded_channels == 0
        && !c.clipped
        && matches!(c.blend, BlendMode::Normal | BlendMode::PassThrough)
        && c.blend_if.is_default()
        && c.excluded_channels == 0
}

/// What replaces smart layer `so`: its one unpacked layer with `so`'s properties folded in, or a
/// group named after it that carries them (isolated like the smart object, never pass-through).
fn unpacked(so: &Layer, mut layers: Vec<Layer>) -> Layer {
    let blend = if so.blend == BlendMode::PassThrough { BlendMode::Normal } else { so.blend };
    if let [c] = layers.as_slice()
        && foldable(so, c)
        && let Some(mut l) = layers.pop()
    {
        l.name = so.name.clone();
        l.visible &= so.visible;
        l.opacity *= so.opacity;
        l.blend = blend;
        l.clipped = so.clipped;
        l.locks = so.locks;
        l.label = so.label;
        l.link_group = so.link_group;
        return l;
    }
    let mut g = so.clone();
    g.id = LayerId::fresh();
    g.psd_id = None;
    g.blend = blend;
    g.content = LayerContent::Group(Group { children: layers, expanded: true, artboard: None });
    detach_psd(&mut g);
    g
}

pub(super) fn convert_to_layers(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    let (intent, bpc) = (s.color.settings.intent(), s.color.settings.bpc);
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let doc = &st.doc;
    let so = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    let sm = smart(doc, id)?;
    let mut layers = contents(s, doc, sm, intent, bpc)?;
    place(s, doc, sm, &mut layers)?;
    let discarded = sm.smart_filters.len();
    let out = unpacked(so, layers);
    let (new_id, group) = (out.id, out.is_group());
    s.edit("Convert to Layers", |doc, active| {
        let path = doc.path_of(id).ok_or(EngineError::NoLayer(id))?;
        *doc.layer_at_mut(&path).ok_or(EngineError::NoLayer(id))? = out;
        *active = Some(new_id);
        Ok(())
    })?;
    Ok(json!({"layer": new_id.0, "group": group, "discardedSmartFilters": discarded}))
}
