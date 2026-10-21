//! Layers panel: hovering a mode in the open blend-mode list previews it on the active layer
//! (Photoshop's live blend-mode preview, #970). Hovering records nothing: leaving the list, or
//! closing it without a choice, shows the document again, and choosing a mode commits it through
//! `layer.setProps` (one history step). The canvas re-renders once per hovered mode, and only
//! where the layer's blending can change pixels.

use std::sync::Arc;

use photocraft_color::BlendMode;
use photocraft_doc::{DocId, Document, LayerId};
use photocraft_geom::Rect;

use crate::PhotocraftApp;

/// Preview keys (`canvas::display_doc`) are `BASE` plus a hash of the revision and the mode.
const BASE: u64 = 1 << 37;

/// A blend mode hovered for a layer of a document at a revision.
pub(crate) struct BlendPreview {
    doc: DocId,
    revision: u64,
    layer: LayerId,
    /// The mode under the pointer; `None` once it left the list (the rest stays for the
    /// canvas's damage on its way back to the document).
    mode: Option<BlendMode>,
    /// The frame the list was last drawn in: a list no longer drawn previews nothing.
    frame: u64,
    /// Where the layer's blending can change pixels; `None` for anywhere.
    area: Option<Rect>,
    /// The document shown for `mode`, with its key.
    shown: Option<(u64, Arc<Document>)>,
}

/// The Layers panel, each frame it draws `layer`'s blend-mode dropdown: the mode under the
/// pointer in its open list, if any.
pub(crate) fn hover(app: &mut PhotocraftApp, layer: LayerId, mode: Option<BlendMode>) {
    let frame = app.frame;
    let Some(st) = app.session.active() else { return };
    let (doc, revision) = (st.doc.id, st.revision);
    if let Some(p) = app.blend_preview.as_mut().filter(|p| p.doc == doc && p.revision == revision && p.layer == layer) {
        p.frame = frame;
        if mode.is_none() {
            p.shown = None;
        }
        p.mode = mode;
        return;
    }
    app.blend_preview = mode.and_then(|mode| {
        let area = photocraft_compose::change_bounds(st.doc.layer(layer)?, st.doc.bounds());
        Some(BlendPreview { doc, revision, layer, mode: Some(mode), frame, area, shown: None })
    });
}

/// The document to show while a blend mode is hovered in document `idx` (`canvas::display_doc`).
pub(crate) fn display_doc(app: &mut PhotocraftApp, idx: usize) -> Option<(Arc<Document>, u64)> {
    let frame = app.frame;
    let st = app.session.documents().get(idx)?;
    let p = app.blend_preview.as_mut().filter(|p| p.doc == st.doc.id && p.revision == st.revision && p.frame + 1 >= frame)?;
    let mode = p.mode?;
    let key = preview_key(p.revision, p.layer, mode);
    if let Some((k, shown)) = &p.shown
        && *k == key
    {
        return Some((shown.clone(), key));
    }
    let mut doc = (*st.doc).clone();
    let l = doc.layer_mut(p.layer).filter(|l| l.blend != mode)?;
    l.blend = mode;
    // Duotone documents display through their inks.
    let doc = Arc::new(photocraft_engine::mode_cmds::display_document(&doc).unwrap_or(doc));
    p.shown = Some((key, doc.clone()));
    Some((doc, key))
}

/// What changed between preview (or document) key `seen` and `now` of document `doc` at
/// `revision`: the hovered layer's area. `None` when either key is another kind of preview, or
/// the layer can change anything (the canvas then recomposites everything).
pub(crate) fn damage(app: &PhotocraftApp, doc: DocId, revision: u64, seen: u64, now: u64) -> Option<Rect> {
    let p = app.blend_preview.as_ref().filter(|p| p.doc == doc && p.revision == revision)?;
    let ours = |k: u64| k == 0 || is_preview_key(k);
    (seen != now && ours(seen) && ours(now) && (is_preview_key(seen) || is_preview_key(now))).then_some(p.area)?
}

fn is_preview_key(key: u64) -> bool {
    (BASE..BASE << 1).contains(&key)
}

fn preview_key(revision: u64, layer: LayerId, mode: BlendMode) -> u64 {
    let h = mode.label().bytes().fold(revision.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ layer.0, |h, b| h.wrapping_mul(31).wrapping_add(u64::from(b)));
    (h & (BASE - 1)) | BASE
}

#[cfg(test)]
mod tests;
