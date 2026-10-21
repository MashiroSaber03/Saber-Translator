//! Layer comps and artboards ⇄ PSD.
//!
//! * Layer comps: image resource 1065 holds a `CompList` descriptor (`list` of `Comp` objects
//!   with `Nm  `, `compID`, `capturedInfo` bit flags and an optional `comment`, plus
//!   `lastAppliedComp`). Each layer's state per comp lives in its `shmd` block under the `cmls`
//!   key: `layerSettings`, one object per comp id list with `enab` (visibility) and `Ofst`
//!   (offset from the layer's saved position). Comp id 0 is the Last Document State.
//! * Artboards: the per-layer `artb` block (older files: `artd`/`abdd`) is an `artboard`
//!   descriptor with `artboardRect`, `guideIndeces`, `artboardPresetName`, `Clr ` and
//!   `artboardBackgroundType` (1 white, 2 black, 3 transparent, 4 other).
//!
//! Both are re-emitted verbatim while they still decode to the document's state, and are
//! regenerated otherwise (comp appearance is not written to PSD; `.pcraft` keeps it).

use photocraft_doc::comps::layer_position;
use photocraft_doc::{Artboard, ArtboardBackground, CompLayerState, Document, Guides, Layer, LayerComp};
use photocraft_geom::Rect;
use photocraft_psd::descriptor::{Descriptor, UnicodeString, Value, VersionedDescriptor};
use photocraft_psd::metadata::{MetadataItem, parse_shmd, write_shmd};

use crate::blocks::{color_from_desc, color_to_desc, num};

/// Image resource id of the layer comp list.
pub const LAYER_COMPS: u16 = 1065;
/// Per-layer artboard block keys, preferred first.
pub const ARTBOARD_KEYS: [&[u8; 4]; 3] = [b"artb", b"artd", b"abdd"];

fn nul_name() -> UnicodeString {
    UnicodeString(vec![0])
}

fn obj(class: &str) -> Descriptor {
    Descriptor { name: nul_name(), ..Descriptor::new(class) }
}

fn pad_even(mut v: Vec<u8>) -> Vec<u8> {
    if v.len() % 2 == 1 {
        v.push(0);
    }
    v
}

fn text(d: &Descriptor, key: &str) -> Option<String> {
    match d.get(key)? {
        Value::Text(t) => Some(t.to_string_lossy()),
        _ => None,
    }
}

fn int(d: &Descriptor, key: &str) -> Option<i64> {
    match d.get(key)? {
        Value::Integer(i) => Some(i64::from(*i)),
        Value::LargeInteger(i) => Some(*i),
        Value::Double(f) => Some(f.round() as i64),
        Value::UnitFloat { value, .. } => Some(value.round() as i64),
        _ => None,
    }
}

fn desc<'a>(d: &'a Descriptor, key: &str) -> Option<&'a Descriptor> {
    match d.get(key)? {
        Value::Descriptor(x) | Value::GlobalObject(x) => Some(x),
        _ => None,
    }
}

// ---------------------------------------------------------------- artboards

/// Decodes an `artb`/`artd`/`abdd` block.
pub fn parse_artboard(data: &[u8]) -> Option<Artboard> {
    let (vd, _) = VersionedDescriptor::parse_prefix(data).ok()?;
    let d = &vd.descriptor;
    let r = desc(d, "artboardRect")?;
    let g = |k: &str| num(r.get(k)).map(|v| v.round() as i32);
    let rect = Rect::new(g("Left")?, g("Top ")?, g("Rght")?, g("Btom")?);
    let background = match int(d, "artboardBackgroundType").unwrap_or(1) {
        2 => ArtboardBackground::Black,
        3 => ArtboardBackground::Transparent,
        4 => ArtboardBackground::Custom(desc(d, "Clr ").and_then(color_from_desc).unwrap_or(photocraft_doc::Color::WHITE)),
        _ => ArtboardBackground::White,
    };
    let preset = text(d, "artboardPresetName").unwrap_or_default();
    Some(Artboard { rect, background, preset })
}

/// Encodes an `artb` block. `guides` are indices into the document's guide list (resource 1032)
/// that belong to the board.
pub fn write_artboard(a: &Artboard, guides: &[i32]) -> Vec<u8> {
    let f = |v: i32| Value::Double(f64::from(v));
    let rect = Descriptor::new("classFloatRect").with("Top ", f(a.rect.y0)).with("Left", f(a.rect.x0)).with("Btom", f(a.rect.y1)).with("Rght", f(a.rect.x1));
    let color = match a.background {
        ArtboardBackground::Custom(c) => c,
        ArtboardBackground::Black => photocraft_doc::Color::BLACK,
        _ => photocraft_doc::Color::WHITE,
    };
    let d = Descriptor::new("artboard")
        .with("artboardRect", Value::Descriptor(rect))
        .with("guideIndeces", Value::List(guides.iter().map(|i| Value::Integer(*i)).collect()))
        .with("artboardPresetName", Value::Text(UnicodeString::new_nul(&a.preset)))
        .with("Clr ", Value::Descriptor(color_to_desc(&color)))
        .with("artboardBackgroundType", Value::Integer(a.background.psd_type()));
    pad_even(VersionedDescriptor::new(d).to_bytes())
}

/// Indices of the document guides (vertical first, then horizontal, as in resource 1032) that
/// lie inside `r`.
pub fn guides_in(guides: &Guides, r: Rect) -> Vec<i32> {
    let v = guides.vertical.iter().map(|x| *x > f64::from(r.x0) && *x < f64::from(r.x1));
    let h = guides.horizontal.iter().map(|y| *y > f64::from(r.y0) && *y < f64::from(r.y1));
    v.chain(h).enumerate().filter(|(_, inside)| *inside).map(|(i, _)| i as i32).collect()
}

/// Keeps, regenerates or removes the artboard block among a layer's raw blocks.
pub(crate) fn artboard_block(guides: &Guides, l: &Layer, raw: &mut Vec<([u8; 4], Vec<u8>)>) {
    let pos = raw.iter().position(|(k, _)| ARTBOARD_KEYS.contains(&k));
    match (l.artboard(), pos) {
        (None, Some(_)) => raw.retain(|(k, _)| !ARTBOARD_KEYS.contains(&k)),
        (None, None) => {}
        (Some(a), Some(i)) if parse_artboard(&raw[i].1).as_ref() == Some(a) => {}
        (Some(a), pos) => {
            let data = write_artboard(a, &guides_in(guides, a.rect));
            match pos {
                Some(i) => raw[i] = (*b"artb", data),
                None => raw.push((*b"artb", data)),
            }
        }
    }
}

// ---------------------------------------------------------------- layer comps

/// One `layerSettings` entry of a `cmls` descriptor.
#[derive(Debug, Clone, PartialEq)]
struct Setting {
    comps: Vec<u32>,
    visible: Option<bool>,
    /// Kept at the file's width: whether origin + offset fits an `i32` is decided per layer (#1017).
    offset: Option<(i64, i64)>,
}

fn parse_cmls(data: &[u8]) -> Option<Vec<Setting>> {
    let (vd, _) = VersionedDescriptor::parse_prefix(data).ok()?;
    let Value::List(items) = vd.descriptor.get("layerSettings")? else { return None };
    let mut out = Vec::new();
    for it in items {
        let Value::Descriptor(s) = it else { continue };
        let comps = match s.get("compList") {
            Some(Value::List(ids)) => ids.iter().filter_map(|v| if let Value::Integer(i) = v { Some(*i as u32) } else { None }).collect(),
            _ => Vec::new(),
        };
        let visible = match s.get("enab") {
            Some(Value::Boolean(b)) => Some(*b),
            _ => None,
        };
        let offset = desc(s, "Ofst").and_then(|o| Some((int(o, "Hrzn")?, int(o, "Vrtc")?)));
        out.push(Setting { comps, visible, offset });
    }
    Some(out)
}

/// The `cmls` item data of a layer's preserved `shmd` block.
fn layer_cmls(l: &Layer) -> Option<Vec<u8>> {
    let (_, shmd) = l.psd_blocks.iter().find(|(k, _)| k == b"shmd")?;
    parse_shmd(shmd).ok()?.into_iter().find(|it| &it.key == b"cmls").map(|it| it.data)
}

/// Absolute comp position of a layer at `here` moved by a file offset; None when it does not fit
/// an `i32` (the offsets come from the file, so the sum must not wrap or overflow, #1017).
fn offset_position((x, y): (i32, i32), (dx, dy): (i64, i64)) -> Option<(i32, i32)> {
    let add = |a: i32, d: i64| i64::from(a).checked_add(d).and_then(|v| i32::try_from(v).ok());
    Some((add(x, dx)?, add(y, dy)?))
}

/// Comps, last applied comp and Last Document State described by `resource` (1065 data) and the
/// layers' `cmls` metadata.
pub fn comps_from_psd(resource: Option<&[u8]>, doc: &Document) -> (Vec<LayerComp>, Option<u32>, Option<LayerComp>) {
    let (comps, last_applied, last, _) = decode_comps(resource, doc);
    (comps, last_applied, last)
}

/// [`comps_from_psd`] plus import warnings for comp data that could not be represented.
pub(crate) fn decode_comps(resource: Option<&[u8]>, doc: &Document) -> (Vec<LayerComp>, Option<u32>, Option<LayerComp>, Vec<String>) {
    let mut warnings = Vec::new();
    let Some(data) = resource else { return (Vec::new(), None, None, warnings) };
    let Ok((vd, _)) = VersionedDescriptor::parse_prefix(data) else { return (Vec::new(), None, None, warnings) };
    let d = &vd.descriptor;
    let mut comps: Vec<LayerComp> = Vec::new();
    if let Some(Value::List(items)) = d.get("list") {
        for it in items {
            let Value::Descriptor(c) = it else { continue };
            let Some(id) = int(c, "compID") else { continue };
            let mut comp = LayerComp {
                id: id as u32,
                name: text(c, "Nm  ").unwrap_or_default(),
                comment: text(c, "comment").unwrap_or_default(),
                apply_visibility: false,
                apply_position: false,
                apply_appearance: false,
                states: Vec::new(),
            };
            comp.set_captured_info(int(c, "capturedInfo").unwrap_or(0) as i32);
            comps.push(comp);
        }
    }
    let last_applied = int(d, "lastAppliedComp").map(|v| v as u32).filter(|id| comps.iter().any(|c| c.id == *id));
    let mut last = LayerComp {
        id: 0,
        name: "Last Document State".into(),
        comment: String::new(),
        apply_visibility: true,
        apply_position: true,
        apply_appearance: false,
        states: Vec::new(),
    };
    for (_, _, l) in doc.walk() {
        let Some(settings) = layer_cmls(l).and_then(|c| parse_cmls(&c)) else { continue };
        let here = layer_position(l);
        for s in settings {
            if s.visible.is_none() && s.offset.is_none() {
                continue;
            }
            let state = CompLayerState {
                layer: l.id,
                visible: s.visible,
                position: match (here, s.offset) {
                    (Some(h), Some(o)) => {
                        let p = offset_position(h, o);
                        if p.is_none() {
                            warnings.push(format!(
                                "layer \"{}\": layer comp position offset ({}, {}) is out of range; that position was not imported",
                                l.name, o.0, o.1
                            ));
                        }
                        p
                    }
                    _ => None,
                },
                appearance: None,
            };
            for id in &s.comps {
                let target = if *id == 0 { Some(&mut last) } else { comps.iter_mut().find(|c| c.id == *id) };
                if let Some(c) = target {
                    c.states.push(state.clone());
                }
            }
        }
    }
    let last = (!last.states.is_empty()).then_some(last);
    (comps, last_applied, last, warnings)
}

/// The 1065 resource data for `doc`'s comps (None when there are none).
pub fn write_comps_resource(doc: &Document) -> Option<Vec<u8>> {
    if doc.layer_comps.is_empty() {
        return None;
    }
    let list = doc
        .layer_comps
        .iter()
        .map(|c| {
            let mut d = Descriptor::new("Comp").with("Nm  ", Value::Text(UnicodeString::new_nul(&c.name)));
            if !c.comment.is_empty() {
                d = d.with("comment", Value::Text(UnicodeString::new_nul(&c.comment)));
            }
            Value::Descriptor(d.with("compID", Value::Integer(c.id as i32)).with("capturedInfo", Value::Integer(c.captured_info())))
        })
        .collect();
    let mut d = Descriptor::new("CompList").with("list", Value::List(list));
    if let Some(id) = doc.last_applied_comp {
        d = d.with("lastAppliedComp", Value::Integer(id as i32));
    }
    Some(VersionedDescriptor::new(d).to_bytes())
}

fn setting_value(ids: &[u32], st: Option<&CompLayerState>, here: Option<(i32, i32)>) -> Value {
    let mut s = obj("null");
    if let Some(st) = st {
        if let Some(v) = st.visible {
            s = s.with("enab", Value::Boolean(v));
        }
        // A position whose offset from the layer does not fit an `i32` (e.g. from a `.pcraft`
        // file) is left out rather than wrapped: the comp then records no position (#1017).
        if let (Some((x, y)), Some((hx, hy))) = (st.position, here)
            && let (Some(dx), Some(dy)) = (x.checked_sub(hx), y.checked_sub(hy))
        {
            s = s.with("Ofst", Value::Descriptor(obj("null").with("Hrzn", Value::Integer(dx)).with("Vrtc", Value::Integer(dy))));
        }
    }
    Value::Descriptor(s.with("compList", Value::List(ids.iter().map(|i| Value::Integer(*i as i32)).collect())))
}

/// The `cmls` item data for one layer (None when the document has no comps).
pub fn write_cmls(comps: &[LayerComp], last: Option<&LayerComp>, l: &Layer, psd_id: u32) -> Option<Vec<u8>> {
    if comps.is_empty() {
        return None;
    }
    let here = layer_position(l);
    let mut settings: Vec<Value> = comps.iter().map(|c| setting_value(&[c.id], c.state(l.id), here)).collect();
    settings.push(setting_value(&[0], last.and_then(|c| c.state(l.id)), here));
    let origin = obj("null").with("Hrzn", Value::Double(0.0)).with("Vrtc", Value::Double(0.0));
    let d = Descriptor::new("null")
        .with("origFXRefPoint", Value::Descriptor(origin))
        .with("LyrI", Value::Integer(psd_id as i32))
        .with("layerSettings", Value::List(settings));
    Some(VersionedDescriptor::new(d).to_bytes())
}

/// Whether `doc`'s comps still equal what its preserved PSD data (resource 1065 plus the layers'
/// `cmls`) decodes to, in which case all of it is written back verbatim.
pub fn comps_unchanged(doc: &Document) -> bool {
    let raw = doc.metadata.psd_resources.iter().find(|(id, _, _)| *id == LAYER_COMPS).map(|(_, _, d)| d.as_slice());
    let (comps, last_applied, last) = comps_from_psd(raw, doc);
    comps == doc.layer_comps && last_applied == doc.last_applied_comp && last == doc.last_document_state
}

/// Replaces (or removes, with `None`) the `cmls` item of a layer's `shmd` block among its raw
/// blocks, keeping the other metadata items.
pub(crate) fn set_cmls(raw: &mut Vec<([u8; 4], Vec<u8>)>, cmls: Option<Vec<u8>>) {
    let pos = raw.iter().position(|(k, _)| k == b"shmd");
    let mut items = pos.and_then(|i| parse_shmd(&raw[i].1).ok()).unwrap_or_default();
    match (items.iter().position(|it| &it.key == b"cmls"), cmls) {
        (Some(i), Some(data)) => items[i] = MetadataItem { data: pad_even(data), ..items[i].clone() },
        (Some(i), None) => {
            items.remove(i);
        }
        (None, Some(data)) => items.insert(0, MetadataItem::new(*b"cmls", data)),
        (None, None) => return,
    }
    match (pos, items.is_empty()) {
        (Some(i), true) => {
            raw.remove(i);
        }
        (Some(i), false) => raw[i].1 = write_shmd(&items),
        (None, false) => raw.push((*b"shmd", write_shmd(&items))),
        (None, true) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_doc::{Color, ColorMode, LayerContent, PixelFormat, SampleType, Size};

    #[test]
    fn artboard_roundtrip() {
        let a = Artboard { rect: Rect::new(10, 20, 110, 220), background: ArtboardBackground::Custom(Color::rgb(0.2, 0.4, 0.6)), preset: "Web".into() };
        let b = write_artboard(&a, &[0, 2]);
        assert_eq!(b.len() % 2, 0);
        let back = parse_artboard(&b).unwrap();
        assert_eq!(back.rect, a.rect);
        assert_eq!(back.preset, "Web");
        let ArtboardBackground::Custom(c) = back.background else { panic!("{back:?}") };
        assert!((c.c[1] - 0.4).abs() < 1e-3);
        for bg in [ArtboardBackground::White, ArtboardBackground::Black, ArtboardBackground::Transparent] {
            let a = Artboard { background: bg, ..Artboard::new(Rect::new(0, 0, 4, 4)) };
            assert_eq!(parse_artboard(&write_artboard(&a, &[])), Some(a));
        }
        assert_eq!(parse_artboard(&[1, 2, 3]), None);
    }

    #[test]
    fn cmls_setting_roundtrip_and_shmd_merge() {
        let mut d = Document::with_background("t", Size::new(32, 32), ColorMode::Rgb, SampleType::U8, Color::WHITE);
        let mut l = Layer::raster("A", PixelFormat::RGBA8);
        l.surface_mut().unwrap().fill_rect(Rect::new(4, 4, 8, 8), &[1.0, 0.0, 0.0, 1.0]);
        let lid = l.id;
        d.layers.push(l);
        d.layer_comps.push(LayerComp {
            id: 7,
            name: "One".into(),
            comment: "c".into(),
            apply_visibility: true,
            apply_position: true,
            apply_appearance: false,
            states: vec![CompLayerState { layer: lid, visible: Some(false), position: Some((10, 1)), appearance: None }],
        });
        d.last_applied_comp = Some(7);
        let res = write_comps_resource(&d).unwrap();
        // `list` is a length-4 string id, as Photoshop writes it, never a code.
        assert!(res.windows(8).any(|w| w == b"\0\0\0\x04list"));
        assert!(!res.windows(8).any(|w| w == b"\0\0\0\0list"));
        let cm = write_cmls(&d.layer_comps, None, d.layer(lid).unwrap(), 3).unwrap();
        // Attach as Photoshop would: shmd with cmls plus another item.
        let mut raw = vec![(*b"shmd", write_shmd(&[MetadataItem::new(*b"cust", vec![1, 2])]))];
        set_cmls(&mut raw, Some(cm));
        let items = parse_shmd(&raw[0].1).unwrap();
        assert_eq!(items.len(), 2);
        let layer = d.layer_mut(lid).unwrap();
        layer.psd_blocks = raw.iter().map(|(k, v)| (*k, std::sync::Arc::new(v.clone()))).collect();
        d.metadata.psd_resources.push((LAYER_COMPS, String::new(), std::sync::Arc::new(res)));
        let (comps, last_applied, last) = comps_from_psd(d.metadata.psd_resources.last().map(|r| r.2.as_slice()), &d);
        assert_eq!(comps, d.layer_comps);
        assert_eq!(last_applied, Some(7));
        assert_eq!(last, None);
        assert!(comps_unchanged(&d));
        // Moving the layer changes the decoded (relative) position: no longer unchanged.
        if let LayerContent::Raster(s) = &mut d.layer_mut(lid).unwrap().content {
            s.fill_rect(Rect::new(0, 0, 2, 2), &[0.0, 0.0, 0.0, 1.0]);
        }
        assert!(!comps_unchanged(&d));
        // Removing cmls keeps the other item; removing everything drops the block.
        set_cmls(&mut raw, None);
        assert_eq!(parse_shmd(&raw[0].1).unwrap().len(), 1);
        let mut only = vec![(*b"shmd", write_shmd(&[MetadataItem::new(*b"cmls", vec![0, 0])]))];
        set_cmls(&mut only, None);
        assert!(only.is_empty());
    }

    /// A layer at origin (4, 4) whose `shmd` carries a `cmls` item with one `layerSettings`
    /// entry for comp 1: visible, offset by `ofst`.
    fn doc_with_cmls_offset(ofst: (Value, Value)) -> (Document, photocraft_doc::LayerId, Vec<u8>) {
        let mut d = Document::with_background("t", Size::new(32, 32), ColorMode::Rgb, SampleType::U8, Color::WHITE);
        let mut l = Layer::raster("A", PixelFormat::RGBA8);
        l.surface_mut().unwrap().fill_rect(Rect::new(4, 4, 8, 8), &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(layer_position(&l), Some((4, 4)));
        let setting = obj("null")
            .with("enab", Value::Boolean(true))
            .with("compList", Value::List(vec![Value::Integer(1)]))
            .with("Ofst", Value::Descriptor(obj("null").with("Hrzn", ofst.0).with("Vrtc", ofst.1)));
        let cmls = VersionedDescriptor::new(Descriptor::new("null").with("layerSettings", Value::List(vec![Value::Descriptor(setting)]))).to_bytes();
        l.psd_blocks = vec![(*b"shmd", std::sync::Arc::new(write_shmd(&[MetadataItem::new(*b"cmls", cmls)])))];
        let lid = l.id;
        d.layers.push(l);
        let comp = Descriptor::new("Comp").with("Nm  ", Value::Text(nul_name())).with("compID", Value::Integer(1)).with("capturedInfo", Value::Integer(3));
        let res = VersionedDescriptor::new(Descriptor::new("CompList").with("list", Value::List(vec![Value::Descriptor(comp)]))).to_bytes();
        (d, lid, res)
    }

    #[test]
    fn unrepresentable_comp_offsets_are_dropped_not_wrapped() {
        // #1017: the layer origin plus a file-supplied offset left i32 range and panicked on
        // import (wrapped in release builds). The position is dropped with a warning; the
        // visibility of the same entry is kept.
        for (h, v) in [(i32::MAX, 0), (0, i32::MAX)] {
            let (d, lid, res) = doc_with_cmls_offset((Value::Integer(h), Value::Integer(v)));
            let (comps, _, _, warnings) = decode_comps(Some(&res), &d);
            let st = comps[0].state(lid).unwrap();
            assert_eq!((st.visible, st.position), (Some(true), None), "({h}, {v})");
            assert_eq!(warnings.len(), 1, "{warnings:?}");
            assert!(warnings[0].contains("layer comp position"), "{warnings:?}");
            assert_eq!(comps_from_psd(Some(&res), &d).0, comps);
        }
        // Offsets wider than i32 (64-bit integers) are dropped too, not truncated.
        let (d, lid, res) = doc_with_cmls_offset((Value::LargeInteger(i64::from(u32::MAX) + 2), Value::Integer(0)));
        let (comps, _, _, warnings) = decode_comps(Some(&res), &d);
        assert_eq!(comps[0].state(lid).unwrap().position, None);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        // Control: an ordinary offset still decodes to layer origin + offset, without warnings.
        let (d, lid, res) = doc_with_cmls_offset((Value::Integer(6), Value::Integer(-3)));
        let (comps, _, _, warnings) = decode_comps(Some(&res), &d);
        assert_eq!(comps[0].state(lid).unwrap().position, Some((10, 1)));
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn unrepresentable_comp_positions_are_written_without_offset() {
        // #1017: a stored position minus the layer origin left i32 range and panicked on export.
        // The offset is omitted (the comp then records no position for that layer).
        let mut l = Layer::raster("A", PixelFormat::RGBA8);
        l.surface_mut().unwrap().fill_rect(Rect::new(4, 4, 8, 8), &[1.0, 0.0, 0.0, 1.0]);
        let comp = |position| LayerComp {
            id: 1,
            name: "One".into(),
            comment: String::new(),
            apply_visibility: true,
            apply_position: true,
            apply_appearance: false,
            states: vec![CompLayerState { layer: l.id, visible: Some(false), position, appearance: None }],
        };
        for p in [(i32::MIN, 0), (0, i32::MIN)] {
            let cm = write_cmls(&[comp(Some(p))], None, &l, 1).unwrap();
            let s = parse_cmls(&cm).unwrap();
            assert_eq!(s[0], Setting { comps: vec![1], visible: Some(false), offset: None }, "{p:?}");
        }
        // Control: an ordinary position is written as its offset from the layer origin.
        let cm = write_cmls(&[comp(Some((10, 1)))], None, &l, 1).unwrap();
        assert_eq!(parse_cmls(&cm).unwrap()[0].offset, Some((6, -3)));
    }
}
