//! Live canvas preview for hovered fonts in the Type tool's font picker (#2315).

use std::collections::VecDeque;
use std::sync::Arc;

use photocraft_doc::{DocId, Document, LayerContent, LayerId};
use photocraft_geom::Rect;

use crate::PhotocraftApp;

const TAG: u64 = 1 << 60;

#[derive(Clone)]
pub struct TypeFontPreview {
    document: DocId,
    revision: u64,
    layer: LayerId,
    range: Option<[usize; 2]>,
    family: Option<String>,
    shown: Option<Arc<Document>>,
    key: u64,
    original_area: Rect,
    areas: VecDeque<(u64, Rect)>,
}

pub(crate) fn follow_hover(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let hover_id = egui::Id::new("type-font-hover");
    let family = ctx.data_mut(|d| {
        let family = d.get_temp::<String>(hover_id);
        d.remove::<String>(hover_id);
        family
    });
    if let Some(family) = family {
        set(app, family);
    } else if app.type_font_preview.is_some() {
        clear(app);
    }
}

pub(crate) fn display_doc(app: &mut PhotocraftApp, idx: usize) -> Option<(Arc<Document>, u64)> {
    let p = app.type_font_preview.as_ref()?.clone();
    let family = p.family.as_deref()?;
    let st = app.session.documents().get(idx)?;
    if st.doc.id != p.document || st.revision != p.revision {
        return None;
    }
    if !photocraft_text::shared().lock().unwrap_or_else(|e| e.into_inner()).fonts.has_family(family) {
        return None;
    }
    if let Some(shown) = &p.shown {
        return Some((shown.clone(), p.key));
    }

    // Clone the original document to avoid mutating the session's document
    let original_doc = st.doc.clone();
    let mut doc = (*st.doc).clone();
    let layer = p.layer;
    let family = family.to_string();
    let range = p.range;

    {
        let l = doc.layer_mut(layer)?;
        if let LayerContent::Text(t) = &mut l.content {
            let (start, end) = range.map_or((0, t.text.len()), |[a, b]| (photocraft_text::byte_index(&t.text, a), photocraft_text::byte_index(&t.text, b)));
            photocraft_engine::type_cmds::style_range(t, start, end, &|style| {
                style.font_family.clone_from(&family);
                style.postscript_name = None;
            });
            photocraft_engine::type_cmds::refresh(&original_doc, t);
        } else {
            return None;
        }
    }

    let area = doc.layer(layer)?.surface().map_or(Rect::EMPTY, |surface| surface.tile_bounds());
    let shown = Arc::new(photocraft_engine::mode_cmds::display_document(&doc).unwrap_or(doc));
    let key = p.key;
    if let Some(p) = app.type_font_preview.as_mut() {
        if p.areas.len() >= 128 {
            p.areas.pop_front();
        }
        p.areas.push_back((key, area));
        p.shown = Some(shown.clone());
    }
    Some((shown, key))
}

pub(crate) fn set(app: &mut PhotocraftApp, family: String) {
    let Some((layer, range)) = crate::type_tool::target(app) else {
        clear(app);
        return;
    };
    let layer = LayerId(layer);
    let Some(st) = app.session.active() else {
        clear(app);
        return;
    };
    if !matches!(st.doc.layer(layer).map(|l| &l.content), Some(LayerContent::Text(_))) {
        clear(app);
        return;
    }
    photocraft_text::served::request(&family);
    if app.type_font_preview.as_ref().is_some_and(|p| {
        p.family.as_deref() == Some(family.as_str()) && p.document == st.doc.id && p.revision == st.revision && p.layer == layer && p.range == range
    }) {
        return;
    }
    let doc_id = st.doc.id;
    let revision = st.revision;
    let original_area = st.doc.layer(layer).and_then(|l| l.surface()).map_or(Rect::EMPTY, |s| s.tile_bounds());
    let previous = app.type_font_preview.take().filter(|p| p.document == doc_id && p.revision == revision && p.layer == layer && p.range == range);
    let (areas, original_area) = previous.map_or((VecDeque::new(), original_area), |p| (p.areas, p.original_area));
    let key = preview_key(revision, layer, range, &family);
    app.type_font_preview = Some(TypeFontPreview { document: doc_id, revision, layer, range, family: Some(family), shown: None, key, original_area, areas });
}

pub(crate) fn clear(app: &mut PhotocraftApp) {
    if let Some(p) = app.type_font_preview.as_mut() {
        p.family = None;
        p.shown = None;
    }
}

pub(crate) fn damage(app: &PhotocraftApp, doc: DocId, revision: u64, before: u64, after: u64) -> Option<Rect> {
    if before == after {
        return None;
    }
    let p = app.type_font_preview.as_ref().filter(|p| p.document == doc && p.revision == revision)?;
    let area = |key| if key == 0 { Some(p.original_area) } else { p.areas.iter().find(|(k, _)| *k == key).map(|(_, r)| *r) };
    Some(area(before)?.union(&area(after)?))
}

fn preview_key(revision: u64, layer: LayerId, range: Option<[usize; 2]>, family: &str) -> u64 {
    let range_key = range.map_or(0, |[start, end]| (start as u64).rotate_left(17) ^ end as u64);
    let h = family.bytes().fold(revision.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ layer.0 ^ range_key, |h, b| h.wrapping_mul(31).wrapping_add(u64::from(b)));
    (h & (TAG - 1)) | TAG
}
