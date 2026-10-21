//! Copy layers into another open document (#589): what Photoshop does when layers are dragged from
//! the Layers panel, or with the Move tool from the canvas, onto another document's tab or window.
//! Photoshop has no menu item for it; `layer.copyToDocument` is the command the drag dispatches and
//! agents call.
//!
//! The copies keep their names, content, masks, styles and blending; they land above the
//! destination's active layer as one history step there, and the destination becomes the active
//! document. Pixels are converted to the destination's colour profile (Color Settings' intent and
//! black point compensation) and bit depth, as Photoshop converts dragged layers.

use std::collections::{HashMap, HashSet};

use photocraft_color::ColorMode;
use photocraft_doc::{Document, LayerContent, LayerId, SmartContentsId};
use photocraft_geom::Rect;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

const CMD: &str = "layer.copyToDocument";

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: CMD.into(), msg: msg.into() }
}

fn enabled(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    d.active_layer.ok_or("no active layer")?;
    if s.documents().len() < 2 { Err("no other document is open".into()) } else { Ok(()) }
}

/// A document index param (`None` when absent), checked against the open documents.
fn doc_index(s: &Session, p: &Value, key: &str) -> Result<Option<usize>> {
    let Some(v) = p.get(key) else { return Ok(None) };
    let i = v.as_u64().and_then(|i| usize::try_from(i).ok()).ok_or_else(|| bad(format!("`{key}` must be a document index")))?;
    if i >= s.documents().len() {
        return Err(bad(format!("no document {i} (there are {})", s.documents().len())));
    }
    Ok(Some(i))
}

/// An `[x, y]` param of finite numbers within ±10⁷ pixels.
fn point(p: &Value, key: &str) -> Result<Option<[f64; 2]>> {
    let Some(v) = p.get(key) else { return Ok(None) };
    let xy = v.as_array().filter(|a| a.len() == 2).and_then(|a| Some([a[0].as_f64()?, a[1].as_f64()?]));
    match xy {
        Some(xy) if xy.iter().all(|c| c.is_finite() && c.abs() <= 1e7) => Ok(Some(xy)),
        _ => Err(bad(format!("`{key}` must be [x, y] (pixels)"))),
    }
}

/// Modes whose layers convert through the colour engine.
pub(crate) fn layered(mode: ColorMode) -> bool {
    matches!(mode, ColorMode::Rgb | ColorMode::Grayscale | ColorMode::Cmyk | ColorMode::Lab)
}

fn center(r: Rect) -> [f64; 2] {
    [(f64::from(r.x0) + f64::from(r.x1)) / 2.0, (f64::from(r.y0) + f64::from(r.y1)) / 2.0]
}

/// Readies `copies` (a scratch document of the source's colour holding duplicated layers) for
/// landing in `dest`: smart objects get fresh contents ids, keeping only the sharing within the
/// batch (contents ids belong to a document, so the copies never join objects already in `dest`,
/// including another opening of the same file), and the pixels convert to `dest`'s profile and
/// bit depth.
pub(crate) fn adapt_copies(s: &Session, copies: &mut Document, dest: &Document) -> Result<()> {
    // Link-group IDs are scoped to a document. Two independently linked pairs can both
    // use group 1; copying one must not link it to the other's existing layers (#3045).
    // Remap each imported group once so links within the copied batch stay intact.
    let mut used: HashSet<u64> = dest.walk().into_iter().filter_map(|(_, _, l)| l.link_group).collect();
    let mut next_group = dest.unused_link_group().ok_or_else(|| EngineError::Other("no free linked-layer group id".into()))?;
    let groups: Vec<_> = copies.walk().into_iter().filter_map(|(_, _, l)| l.link_group.map(|group| (l.id, group))).collect();
    let mut remapped = HashMap::new();
    for (_, group) in &groups {
        if remapped.contains_key(group) {
            continue;
        }
        while used.contains(&next_group) {
            next_group = next_group.checked_add(1).unwrap_or(1);
        }
        remapped.insert(*group, next_group);
        used.insert(next_group);
        next_group = next_group.checked_add(1).unwrap_or(1);
    }
    for (id, group) in groups {
        if let Some(layer) = copies.layer_mut(id) {
            layer.link_group = remapped.get(&group).copied();
        }
    }
    let mut contents = HashMap::new();
    let smart_ids: Vec<_> = copies.walk().into_iter().filter_map(|(_, _, l)| matches!(l.content, LayerContent::Smart(_)).then_some(l.id)).collect();
    for id in smart_ids {
        if let Some(l) = copies.layer_mut(id)
            && let LayerContent::Smart(sm) = &mut l.content
        {
            sm.contents_id = *contents.entry(sm.contents_id).or_insert_with(SmartContentsId::fresh);
        }
    }
    if (copies.mode, &copies.icc_profile) != (dest.mode, &dest.icc_profile) {
        let profile = crate::color_cmds::document_profile(dest);
        crate::color_cmds::convert_document(copies, &profile, s.color.settings.intent(), s.color.settings.bpc)?;
    }
    if copies.depth != dest.depth {
        crate::image_cmds::for_each_surface(&mut copies.layers, true, &mut |surf, _| {
            let f = surf.format().with_sample(dest.depth);
            *surf = surf.convert(f);
        });
    }
    Ok(())
}

/// Moves every layer of `copies` by whole pixels (content, masks, vectors and smart placement).
pub(crate) fn translate_copies(copies: &mut Document, dx: i32, dy: i32) {
    if dx != 0 || dy != 0 {
        let snapshot = copies.clone();
        for l in &mut copies.layers {
            crate::commands::translate_layer(&snapshot, l, dx, dy);
            crate::vector_cmds::translate_vectors(&snapshot, l, f64::from(dx), f64::from(dy));
        }
    }
}

/// `layer.copyToDocument`.
fn copy_to_document(s: &mut Session, p: &Value) -> Result<Value> {
    let dest = doc_index(s, p, "document")?.ok_or_else(|| bad("missing `document` (the destination's index)"))?;
    let src = match doc_index(s, p, "source")? {
        Some(i) => i,
        None => s.active_index().ok_or(EngineError::NoDocument)?,
    };
    if src == dest {
        return Err(bad("the destination is the source document (use layer.duplicate)"));
    }
    let (from, to) = (&s.documents()[src], &s.documents()[dest]);
    let (sdoc, ddoc) = (from.doc.clone(), to.doc.clone());
    let ids: Vec<LayerId> = match p.get("layers") {
        None => from.selected_layers(),
        Some(v) => v
            .as_array()
            .map(|a| a.iter().map(|id| id.as_u64().map(LayerId)).collect::<Option<Vec<_>>>())
            .and_then(|ids| ids)
            .ok_or_else(|| bad("`layers` must be an array of layer ids"))?,
    };
    if let Some(missing) = ids.iter().find(|id| sdoc.layer(**id).is_none()) {
        return Err(EngineError::NoLayer(*missing));
    }
    let ids = crate::layer_multi_cmds::top_level(&sdoc, &ids);
    if ids.is_empty() {
        return Err(bad("no layers to copy"));
    }
    for (doc, role) in [(&sdoc, "source"), (&ddoc, "destination")] {
        if !layered(doc.mode) {
            return Err(EngineError::Other(format!("layers can't be copied with a {:?} {role} document", doc.mode)));
        }
    }
    // The copies, bottom to top, in a scratch document of the source's colour so they convert
    // the way a whole document does.
    let mut copies = Document::new("", sdoc.size, sdoc.mode, sdoc.depth);
    copies.icc_profile = sdoc.icc_profile.clone();
    // Type re-lays out at the source's resolution, so it keeps its size in pixels.
    copies.resolution_dpi = sdoc.resolution_dpi;
    for id in &ids {
        let src_layer = sdoc.layer(*id).ok_or(EngineError::NoLayer(*id))?;
        let mut l = src_layer.duplicate();
        // A dragged Background arrives as an ordinary layer (the destination keeps its own).
        if sdoc.is_background(*id) {
            l.name = ddoc.next_layer_name("Layer");
            l.locks = Default::default();
        }
        copies.layers.push(l);
    }
    if let (Some(name), [l]) = (p.get("name").and_then(Value::as_str).filter(|n| !n.is_empty()), copies.layers.as_mut_slice()) {
        l.name = name.to_string();
    }
    adapt_copies(s, &mut copies, &ddoc)?;
    // Placement: centred on the destination's canvas, centred on a point, or offset from where
    // the layers are in the source (default: the same canvas position).
    let bounds = copies.layers.iter().filter_map(crate::layer_multi_cmds::layer_bounds).fold(Rect::EMPTY, |a, b| a.union(&b));
    let content = if bounds.is_empty() { sdoc.bounds() } else { bounds };
    let (dx, dy) = if p.get("center").and_then(Value::as_bool).unwrap_or(false) {
        let (c, d) = (center(content), center(ddoc.bounds()));
        (d[0] - c[0], d[1] - c[1])
    } else if let Some(at) = point(p, "at")? {
        let c = center(content);
        (at[0] - c[0], at[1] - c[1])
    } else {
        point(p, "offset")?.map_or((0.0, 0.0), |o| (o[0], o[1]))
    };
    let (dx, dy) = (dx.round() as i32, dy.round() as i32);
    translate_copies(&mut copies, dx, dy);
    let patterns: Vec<_> = sdoc.patterns.iter().filter(|pat| !ddoc.patterns.iter().any(|d| d.id == pat.id)).cloned().collect();
    s.set_active(dest);
    let label = if ids.len() == 1 { "Duplicate Layer" } else { "Duplicate Layers" };
    let edited = s.edit(label, |doc, active| {
        doc.patterns.extend(patterns);
        let mut above = *active;
        let mut new = Vec::with_capacity(copies.layers.len());
        for l in copies.layers {
            let id = doc.insert_above(above, l);
            above = Some(id);
            new.push(id);
        }
        *active = above;
        Ok(new)
    });
    // Nothing changed: the source stays the active document.
    if edited.is_err() {
        s.set_active(src);
    }
    let new = edited?;
    let last = new.last().copied();
    crate::layer_multi_cmds::reselect(s, new.clone(), last);
    Ok(json!({"document": dest, "layers": new.iter().map(|l| l.0).collect::<Vec<_>>(), "offset": [dx, dy]}))
}

/// `layer.duplicate` with `"document": "new"`: Photoshop's Duplicate Layer… ›
/// Document: New. A new document the size of the source's canvas holds just the duplicated
/// layers at their canvas positions (a transparent background elsewhere); the source is
/// untouched and the new document becomes active. Mode, depth, resolution, profile,
/// patterns and global light come along so the copies render the same.
pub(crate) fn duplicate_to_new_document(s: &mut Session, p: &Value) -> Result<Value> {
    let src = s.active_index().ok_or(EngineError::NoDocument)?;
    let sdoc = s.documents().get(src).ok_or(EngineError::NoDocument)?.doc.clone();
    let ids = match crate::commands::explicit_layer(s, p)? {
        Some(id) => vec![id],
        None => s.active().ok_or(EngineError::NoDocument)?.selected_layers(),
    };
    if let Some(missing) = ids.iter().find(|id| sdoc.layer(**id).is_none()) {
        return Err(EngineError::NoLayer(*missing));
    }
    let ids = crate::layer_multi_cmds::top_level(&sdoc, &ids);
    if ids.is_empty() {
        return Err(bad("no layers to copy"));
    }
    if !layered(sdoc.mode) {
        return Err(EngineError::Other(format!("layers can't be duplicated with a {:?} document", sdoc.mode)));
    }
    let doc_name = p.get("documentName").and_then(Value::as_str).filter(|n| !n.is_empty()).map(str::to_string).unwrap_or_else(|| format!("{} copy", sdoc.name));
    let mut nd = Document::new(doc_name, sdoc.size, sdoc.mode, sdoc.depth);
    nd.resolution_dpi = sdoc.resolution_dpi;
    nd.icc_profile.clone_from(&sdoc.icc_profile);
    nd.patterns.clone_from(&sdoc.patterns);
    nd.global_light = sdoc.global_light;
    for id in &ids {
        let mut l = sdoc.layer(*id).ok_or(EngineError::NoLayer(*id))?.duplicate();
        if ids.len() == 1
            && let Some(name) = p.get("name").and_then(Value::as_str).filter(|n| !n.is_empty())
        {
            l.name = name.to_string();
        }
        nd.layers.push(l);
    }
    // Contents ids belong to a document: fresh ones keep the copies independent of the
    // source while preserving sharing within the batch (as `adapt_copies` does).
    let mut contents = HashMap::new();
    let smart_ids: Vec<_> = nd.walk().into_iter().filter_map(|(_, _, l)| matches!(l.content, LayerContent::Smart(_)).then_some(l.id)).collect();
    for id in smart_ids {
        if let Some(l) = nd.layer_mut(id)
            && let LayerContent::Smart(sm) = &mut l.content
        {
            sm.contents_id = *contents.entry(sm.contents_id).or_insert_with(SmartContentsId::fresh);
        }
    }
    let new_ids: Vec<LayerId> = nd.layers.iter().map(|l| l.id).collect();
    let index = s.add_document(nd, None);
    crate::layer_multi_cmds::reselect(s, new_ids.clone(), new_ids.last().copied());
    let mut out = json!({"document": index, "layers": new_ids.iter().map(|l| l.0).collect::<Vec<_>>()});
    if let [one] = new_ids.as_slice()
        && let Some(map) = out.as_object_mut()
    {
        map.insert("layer".into(), json!(one.0));
    }
    Ok(out)
}

pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Copy Layers to Document",
        menu: &[],
        shortcut: None,
        params: r##"{"document":index (destination),"layers":[id,…]?=the source's selected layers,"source":index?=active,"center":bool=false (centre on the destination's canvas),"at":[x,y]? (centre the copies on this point),"offset":[dx,dy]?=[0,0] (from their place in the source),"name":str? (one layer: the copy's name)}"##,
        enabled,
        run: copy_to_document,
        journal: true,
    }]
}

#[cfg(test)]
mod tests;
