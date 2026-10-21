//! Displace smart filters (`Fltr` class `Dspl`) as observed in Photoshop-authored mockup PSDs.
//!
//! The filter descriptor holds the scales (`HrzS`, `VrtS`, percent), the fit (`DspM`: `StrF`
//! stretch or `Tile`), the undefined areas (`UndA`: `RptE` repeat edge pixels or `WrpA` wrap),
//! an alias to the map file on the author's disk (`DspF`, usually unavailable), `EmbF` (true) and
//! the map itself, embedded as raw data (`DspD`, type `tdta`):
//!
//! ```text
//! u32 LE height, u32 LE width, u16 LE mode (3 observed), then two 8-bit planes of
//! width * height bytes (horizontal map, vertical map), then padding
//! ```
//!
//! The whole `filterFX` item (with the map) is kept verbatim in the params as a template
//! ([`TEMPLATE_KEY`]), so the filter re-renders, survives `.pcraft` and is written back to PSD
//! with only the editable settings replaced. Anything else (another mode, a missing or truncated
//! map, unknown enum values) stays an opaque [`super::UNSUPPORTED_FILTER`].

use std::sync::Arc;

use photocraft_algo::Image;
use photocraft_doc::SmartFilter;
use photocraft_geom::Rect;
use photocraft_psd::descriptor::{Descriptor, Id, Value};
use serde_json::{Value as J, json};

use super::{blend_options, from_hex, get_desc, int, set, to_hex};
use crate::blocks::{enum_of, num};

pub(super) const CLASS: &str = "Dspl";
pub(super) const COMMAND: &str = "filter.distort.displace";
const FILTER_ID: i32 = super::code(b"Dspl");
pub(super) const TEMPLATE_KEY: &str = "__displacePsd";
/// The embedded map (raw data).
const MAP_KEY: &str = "DspD";
/// `tdta` header: height, width (u32 LE) and mode (u16 LE).
const HEADER: usize = 10;
/// The only map layout seen in Photoshop files: two 8-bit planes.
const MODE_TWO_PLANES: u16 = 3;
/// 8K × 8K: far beyond the observed document-sized maps, and bounds the f32 map we build.
const MAX_MAP_PIXELS: usize = 1 << 26;
/// The template is the item plus the map; a little room for the rest of the descriptor.
const MAX_TEMPLATE_BYTES: usize = 2 * MAX_MAP_PIXELS + (1 << 20);

const FITS: [(&[u8], &str); 2] = [(b"StrF", "stretch"), (b"Tile", "tile")];
const UNDEFINED: [(&[u8], &str); 2] = [(b"RptE", "repeat"), (b"WrpA", "wrap")];

/// A parsed embedded map: size and the two 8-bit planes (borrowed from the blob).
#[derive(Debug, PartialEq)]
pub(super) struct EmbeddedMap<'a> {
    pub width: usize,
    pub height: usize,
    pub horizontal: &'a [u8],
    pub vertical: &'a [u8],
}

/// Parses an embedded displacement map; `None` for any other mode, a zero or oversized map, or a
/// blob shorter than the header plus both planes.
pub(super) fn parse_map(data: &[u8]) -> Option<EmbeddedMap<'_>> {
    let u32_at = |i: usize| Some(u32::from_le_bytes(data.get(i..i + 4)?.try_into().ok()?));
    let height = usize::try_from(u32_at(0)?).ok()?;
    let width = usize::try_from(u32_at(4)?).ok()?;
    let mode = u16::from_le_bytes(data.get(8..HEADER)?.try_into().ok()?);
    let n = width.checked_mul(height)?;
    if mode != MODE_TWO_PLANES || n == 0 || n > MAX_MAP_PIXELS {
        return None;
    }
    let horizontal = data.get(HEADER..HEADER.checked_add(n)?)?;
    let vertical = data.get(HEADER + n..HEADER.checked_add(n.checked_mul(2)?)?)?;
    Some(EmbeddedMap { width, height, horizontal, vertical })
}

fn fltr_map(item: &Descriptor) -> Option<EmbeddedMap<'_>> {
    match get_desc(item, "Fltr")?.get(MAP_KEY)? {
        Value::RawData(b) => parse_map(b),
        _ => None,
    }
}

fn lookup<'a>(table: &[(&[u8], &'a str)], v: &[u8]) -> Option<&'a str> {
    table.iter().find(|(k, _)| *k == v).map(|(_, s)| *s)
}

/// Editable settings of a `Dspl` descriptor, `None` when one is missing or unknown.
fn settings(f: &Descriptor) -> Option<J> {
    let scale = |k| num(f.get(k)).filter(|v| v.is_finite() && (-999.0..=999.0).contains(v));
    Some(json!({
        "horizontal": scale("HrzS")?,
        "vertical": scale("VrtS")?,
        "fit": lookup(&FITS, enum_of(f, "DspM")?)?,
        "undefinedAreas": lookup(&UNDEFINED, enum_of(f, "UndA")?)?,
    }))
}

/// `filter.distort.displace` params for a Displace item with an embedded map.
pub(super) fn import_params(item: &Descriptor) -> Option<J> {
    let f = get_desc(item, "Fltr")?;
    if !f.class_id.is(CLASS) {
        return None;
    }
    let mut params = settings(f)?;
    fltr_map(item)?;
    // Neutral blending in the template: the model's own fields replace it on export.
    let mut template = item.clone();
    let neutral = SmartFilter { command: String::new(), params: J::Null, blend: photocraft_color::BlendMode::Normal, opacity: 1.0, visible: true };
    set(&mut template, "blendOptions", blend_options(&neutral));
    set(&mut template, "enab", Value::Boolean(true));
    params.as_object_mut()?.insert(TEMPLATE_KEY.into(), json!(to_hex(&template.to_bytes())));
    Some(params)
}

/// The template item of `params`, checked to be a Displace filter with a readable map.
fn template(params: &J) -> Result<Descriptor, String> {
    let raw = params.get(TEMPLATE_KEY).ok_or("a Displace map from a file or document is not written to PSD")?;
    let raw = raw.as_str().ok_or("the Displace PSD data must be hex text")?;
    if raw.len() > MAX_TEMPLATE_BYTES * 2 {
        return Err("the Displace PSD data is too large".into());
    }
    let bytes = from_hex(raw).ok_or("the Displace PSD data is not valid hex")?;
    let item = Descriptor::from_bytes(&bytes).map_err(|e| format!("the Displace PSD data is unreadable ({e})"))?;
    if !item.class_id.is("filterFX") || int(&item, "filterID") != Some(FILTER_ID) || fltr_map(&item).is_none() {
        return Err("the Displace PSD data is not a Displace filter with an embedded map".into());
    }
    Ok(item)
}

/// The `filterFXList` item: the template with the current scales, fit and undefined areas.
pub(super) fn export_item(params: &J) -> Result<Descriptor, String> {
    let object = params.as_object().ok_or("Displace parameters must be an object")?;
    if let Some(k) = object.keys().find(|k| !super::is_meta(k) && !["horizontal", "vertical", "fit", "undefinedAreas"].contains(&k.as_str())) {
        return Err(format!("Displace setting `{k}` has no PSD equivalent next to its embedded map"));
    }
    let mut item = template(params)?;
    let mut f = get_desc(&item, "Fltr").ok_or("the Displace PSD data has no filter")?.clone();
    let scale = |k: &str, d: f64| -> Result<Value, String> {
        let v = params.get(k).map_or(Some(d), J::as_f64).filter(|v| v.is_finite()).ok_or(format!("Displace `{k}` must be a number"))?;
        Ok(Value::Integer(v.round().clamp(-999.0, 999.0) as i32))
    };
    let choice = |k: &str, table: &[(&[u8], &str)], d: &str| -> Result<Id, String> {
        let v = params.get(k).map_or(Some(d), J::as_str).ok_or(format!("Displace `{k}` must be text"))?;
        let code = table.iter().find(|(_, s)| *s == v).map(|(c, _)| *c).ok_or(format!("Displace `{k}` is not one of the PSD values"))?;
        Ok(Id::new(std::str::from_utf8(code).map_err(|e| e.to_string())?))
    };
    // Keep the template's enum types; only the values change.
    let set_enum = |f: &mut Descriptor, key: &str, value: Id| {
        let type_id = match f.get(key) {
            Some(Value::Enumerated { type_id, .. }) => type_id.clone(),
            _ => Id::new(key),
        };
        set(f, key, Value::Enumerated { type_id, value });
    };
    set(&mut f, "HrzS", scale("horizontal", 10.0)?);
    set(&mut f, "VrtS", scale("vertical", 10.0)?);
    set_enum(&mut f, "DspM", choice("fit", &FITS, "stretch")?);
    set_enum(&mut f, "UndA", choice("undefinedAreas", &UNDEFINED, "repeat")?);
    set(&mut item, "Fltr", Value::Descriptor(f));
    Ok(item)
}

/// The displacement map embedded in Displace params imported from PSD (`None` when there is
/// none or it is unreadable). The two planes become the first two channels; identical planes
/// (every file seen so far) become one channel, which Displace reads for both directions.
pub fn embedded_map(params: &J) -> Option<Arc<Image>> {
    let item = template(params).ok()?;
    let m = fltr_map(&item)?;
    let rect = Rect::new(0, 0, i32::try_from(m.width).ok()?, i32::try_from(m.height).ok()?);
    let v = |b: u8| f32::from(b) / 255.0;
    let (ch, data) = if m.horizontal == m.vertical {
        (1, m.horizontal.iter().map(|&b| v(b)).collect())
    } else {
        (3, m.horizontal.iter().zip(m.vertical).flat_map(|(&h, &w)| [v(h), v(w), 0.5]).collect())
    };
    Some(Arc::new(Image { rect, ch, data }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smart_map::{UNSUPPORTED_FILTER, filter_from_item, item_for_filter};

    /// A `tdta` blob: `w`×`h`, horizontal plane `h_of(i)`, vertical `v_of(i)`, then `pad` zeros.
    fn tdta(w: u32, h: u32, h_of: impl Fn(usize) -> u8, v_of: impl Fn(usize) -> u8, pad: usize) -> Vec<u8> {
        let n = (w * h) as usize;
        let mut b = Vec::new();
        b.extend(h.to_le_bytes());
        b.extend(w.to_le_bytes());
        b.extend(MODE_TWO_PLANES.to_le_bytes());
        b.extend((0..n).map(h_of));
        b.extend((0..n).map(v_of));
        b.extend(std::iter::repeat_n(0, pad));
        b
    }

    /// A `filterFXList` item as Photoshop writes a Displace smart filter.
    fn item(map: Vec<u8>) -> Descriptor {
        let e = |t: &str, v: &str| Value::Enumerated { type_id: Id::new(t), value: Id::new(v) };
        Descriptor::new("filterFX")
            .with("Nm  ", Value::Text(photocraft_psd::descriptor::UnicodeString::new_nul("Displace...")))
            .with(
                "blendOptions",
                blend_options(&SmartFilter {
                    command: String::new(),
                    params: J::Null,
                    blend: photocraft_color::BlendMode::Normal,
                    opacity: 1.0,
                    visible: true,
                }),
            )
            .with("enab", Value::Boolean(true))
            .with("hasoptions", Value::Boolean(true))
            .with(
                "Fltr",
                Value::Descriptor(
                    Descriptor::new(CLASS)
                        .with("HrzS", Value::Integer(10))
                        .with("VrtS", Value::Integer(10))
                        .with("DspM", e("DspM", "StrF"))
                        .with("UndA", e("UndA", "RptE"))
                        .with("DspF", Value::Alias(b"Macintosh HD:maps:displace.psd".to_vec()))
                        .with("EmbF", Value::Boolean(true))
                        .with(MAP_KEY, Value::RawData(map)),
                ),
            )
            .with("filterID", Value::Integer(FILTER_ID))
    }

    #[test]
    fn filter_id_is_dspl() {
        assert_eq!(FILTER_ID, 1148416108);
    }

    #[test]
    fn parses_the_observed_layout() {
        let b = tdta(3, 2, |i| i as u8, |i| 100 + i as u8, 3);
        let m = parse_map(&b).unwrap();
        assert_eq!((m.width, m.height), (3, 2));
        assert_eq!(m.horizontal, &[0, 1, 2, 3, 4, 5]);
        assert_eq!(m.vertical, &[100, 101, 102, 103, 104, 105]);
    }

    #[test]
    fn hostile_maps_are_rejected_without_panicking() {
        let good = tdta(4, 4, |i| i as u8, |i| i as u8, 0);
        for len in 0..good.len() {
            assert!(parse_map(&good[..len]).is_none(), "truncated to {len}");
        }
        let header = |h: u32, w: u32, mode: u16| [h.to_le_bytes().as_slice(), &w.to_le_bytes(), &mode.to_le_bytes(), &[0; 64]].concat();
        assert!(parse_map(&header(u32::MAX, u32::MAX, 3)).is_none());
        assert!(parse_map(&header(1 << 16, 1 << 16, 3)).is_none());
        assert!(parse_map(&header(0, 4, 3)).is_none());
        assert!(parse_map(&header(2, 2, 2)).is_none());
        assert!(parse_map(&header(2, 2, 4)).is_none());
        assert!(parse_map(&header(2, 2, 3)).is_some());
        // Every possible short blob, and a few corrupted bytes of a good one.
        for seed in 0..512u32 {
            let mut b = good.clone();
            let i = (seed as usize * 7) % b.len();
            b[i] = b[i].wrapping_add(seed as u8 | 1);
            let _ = parse_map(&b);
            let _ = embedded_map(&json!({TEMPLATE_KEY: to_hex(&item(b).to_bytes())}));
        }
    }

    #[test]
    fn displace_items_map_to_the_command_and_round_trip() {
        let src = item(tdta(4, 3, |i| (i * 20) as u8, |i| (i * 20) as u8, 4));
        let f = filter_from_item(&src);
        assert_eq!(f.command, COMMAND);
        assert_eq!(
            (f.params["horizontal"].as_f64(), f.params["fit"].as_str(), f.params["undefinedAreas"].as_str()),
            (Some(10.0), Some("stretch"), Some("repeat"))
        );
        assert_eq!(item_for_filter(&f).unwrap(), src, "unchanged settings write the original item back");
        let map = embedded_map(&f.params).unwrap();
        assert_eq!((map.rect, map.ch), (Rect::new(0, 0, 4, 3), 1));
        assert_eq!(map.data[1], 20.0 / 255.0);

        let mut edited = f.clone();
        edited.params["horizontal"] = json!(-25.4);
        edited.params["fit"] = json!("tile");
        edited.params["undefinedAreas"] = json!("wrap");
        edited.opacity = 0.5;
        let back = filter_from_item(&item_for_filter(&edited).unwrap());
        assert_eq!(back.command, COMMAND);
        assert_eq!(
            (back.params["horizontal"].as_f64(), back.params["fit"].as_str(), back.params["undefinedAreas"].as_str()),
            (Some(-25.0), Some("tile"), Some("wrap"))
        );
        assert_eq!(back.opacity, 0.5);
        assert!(embedded_map(&back.params).is_some(), "the map survives");
    }

    #[test]
    fn different_planes_drive_each_direction() {
        let f = filter_from_item(&item(tdta(2, 1, |_| 255, |_| 0, 0)));
        let map = embedded_map(&f.params).unwrap();
        assert_eq!(map.ch, 3);
        assert_eq!(&map.data[..3], &[1.0, 0.0, 0.5]);
    }

    #[test]
    fn unreadable_displace_filters_stay_opaque() {
        let only_alias = {
            let mut i = item(Vec::new());
            let mut f = get_desc(&i, "Fltr").unwrap().clone();
            f.items.retain(|(k, _)| !k.is(MAP_KEY));
            set(&mut i, "Fltr", Value::Descriptor(f));
            i
        };
        let mut bad_enum = item(tdta(2, 2, |_| 0, |_| 0, 0));
        let mut f = get_desc(&bad_enum, "Fltr").unwrap().clone();
        set(&mut f, "DspM", Value::Enumerated { type_id: Id::new("DspM"), value: Id::new("Huh?") });
        set(&mut bad_enum, "Fltr", Value::Descriptor(f));
        let wrong_mode = item([tdta(2, 2, |_| 0, |_| 0, 0)[..8].to_vec(), 1u16.to_le_bytes().to_vec(), vec![0; 8]].concat());
        for src in [only_alias, bad_enum, wrong_mode, item(vec![1, 2, 3])] {
            let f = filter_from_item(&src);
            assert_eq!(f.command, UNSUPPORTED_FILTER);
            assert_eq!(item_for_filter(&f).unwrap(), src);
        }
    }

    #[test]
    fn bad_exports_report_errors() {
        let ok = filter_from_item(&item(tdta(2, 2, |_| 0, |_| 0, 0))).params;
        let with = |k: &str, v: J| {
            let mut p = ok.clone();
            p[k] = v;
            p
        };
        for p in [
            J::Null,
            json!({"horizontal": 10}),
            json!({TEMPLATE_KEY: "zz"}),
            json!({TEMPLATE_KEY: 12}),
            json!({TEMPLATE_KEY: to_hex(&Descriptor::new("filterFX").with("filterID", Value::Integer(FILTER_ID)).to_bytes())}),
            with("horizontal", json!("big")),
            with("fit", json!("squash")),
            with("mapPath", json!("/tmp/map.psd")),
        ] {
            assert!(export_item(&p).is_err(), "{p}");
        }
        assert!(embedded_map(&json!({TEMPLATE_KEY: "0".repeat(MAX_TEMPLATE_BYTES * 2 + 2)})).is_none());
    }
}
