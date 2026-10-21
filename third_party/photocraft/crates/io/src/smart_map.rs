//! Smart objects in PSD: the placed-layer blocks (`SoLd`, `PlLd`) and the smart filter stack
//! (`filterFX` inside `SoLd`).
//!
//! Layouts follow the Adobe PSD spec ("Placed Layer", "Placed Layer Data") and Photoshop-authored
//! files (`corpus/photoshop/smart-filters`): a `soLD` descriptor holds the linked-file id (`Idnt`,
//! the uuid of the `lnk2` item), the instance id (`placed`, which keys the `FEid` filter cache),
//! the transform quad (`Trnf`), the source size (`Sz  `), the warp, and when filters are applied a
//! `filterFX` object:
//!
//! ```text
//! filterFX (filterFXStyle): enab, validAtPosition, filterMaskEnable, filterMaskLinked,
//!   filterMaskExtendWithWhite, filterFXList = [ filterFX: Nm, blendOptions {Opct, Md},
//!   enab, hasoptions, FrgC, BckC, Fltr (the filter's own descriptor), filterID ]
//! ```
//!
//! The list is in application order (first = bottom, applied first), like
//! [`SmartObject::smart_filters`](photocraft_doc::SmartObject). Filters PhotoCraft implements map
//! to their command id and dialog parameters; any other filter becomes a
//! [`UNSUPPORTED_FILTER`] entry that keeps its descriptor verbatim (hex in `params.psd`), so it is
//! listed, can be hidden, reordered or deleted, and is written back unchanged.
//!
//! Camera Raw and Displace keep their whole item as a template next to the editable settings
//! (Displace's carries the embedded displacement map, see `displace_map.rs`).

use photocraft_color::{BlendMode, ColorMode, PixelFormat, SampleType};
use photocraft_doc::{LayerMask, SmartFilter};
use photocraft_geom::warp::{Warp, WarpStyle};
use photocraft_geom::{Affine, Rect as GeomRect};
use photocraft_psd::Compression;
use photocraft_psd::Rect as PsdRect;
use photocraft_psd::descriptor::{Class, Descriptor, Id, ObjectArray, ReferenceItem, UnicodeString, Value, VersionedDescriptor};
use photocraft_psd::filter_effects::{EffectsPlane, FilterEffectsItem};
use photocraft_raster::Surface;
use serde_json::{Map, Value as J, json};

use crate::blocks::{enum_of, get_desc, num};

#[path = "camera_raw_map.rs"]
mod camera_raw;
#[path = "displace_map.rs"]
mod displace;

pub use displace::embedded_map as displace_embedded_map;

/// Command id of a Photoshop smart filter PhotoCraft does not implement. Its params hold the
/// filter's name, Photoshop filter id and descriptor (`psd`, hex); it renders as a pass-through.
pub const UNSUPPORTED_FILTER: &str = "psd.unsupportedFilter";

/// The smart filter stack of a placed layer, as stored in `filterFX`.
#[derive(Clone, Debug, PartialEq)]
pub struct FilterStack {
    /// `enab`: Layer › Smart Filter › Disable Smart Filters clears it.
    pub enabled: bool,
    pub filters: Vec<SmartFilter>,
    /// `filterMaskEnable`.
    pub mask_enabled: bool,
    /// `filterMaskLinked`.
    pub mask_linked: bool,
}

impl Default for FilterStack {
    fn default() -> Self {
        FilterStack { enabled: true, filters: Vec::new(), mask_enabled: true, mask_linked: false }
    }
}

/// What a `soLD` descriptor says about a smart object.
#[derive(Clone, Debug)]
pub struct Placed {
    pub descriptor: Descriptor,
    /// `Idnt`: uuid of the embedded or linked file (`lnk2` item).
    pub idnt: String,
    /// `placed`: id of this instance (keys the `FEid` filter cache).
    pub placed: String,
    /// Source pixels → document pixels (from `Trnf` and `Sz  `).
    pub transform: Affine,
    /// `filterFX`, if any filters were ever applied.
    pub stack: Option<FilterStack>,
}

fn text_of(d: &Descriptor, key: &str) -> Option<String> {
    match d.get(key)? {
        Value::Text(t) => Some(t.to_string_lossy()),
        _ => None,
    }
}

fn bool_or(d: &Descriptor, key: &str, default: bool) -> bool {
    match d.get(key) {
        Some(Value::Boolean(b)) => *b,
        _ => default,
    }
}

/// The `soLD` descriptor of `SoLd`/`SoLE` block data.
pub fn sold_descriptor(data: &[u8]) -> Option<Descriptor> {
    if data.get(..4)? != b"soLD" {
        return None;
    }
    crate::blocks::parse_prefix_versioned(data.get(8..)?)
}

/// Parses `SoLd`/`SoLE` block data (best effort: `None` when it is not a `soLD` structure).
pub fn parse_sold(data: &[u8]) -> Option<Placed> {
    let d = sold_descriptor(data)?;
    let (idnt, transform) = crate::blocks::parse_smart(b"SoLd", data);
    let placed = text_of(&d, "placed").unwrap_or_default();
    let stack = get_desc(&d, "filterFX").map(filter_stack);
    Some(Placed { descriptor: d, idnt, placed, transform, stack })
}

/// Reads a `filterFX` object.
pub fn filter_stack(fx: &Descriptor) -> FilterStack {
    let filters = match fx.get("filterFXList") {
        Some(Value::List(items)) => items
            .iter()
            .filter_map(|v| match v {
                Value::Descriptor(d) => Some(filter_from_item(d)),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    FilterStack {
        enabled: bool_or(fx, "enab", true),
        filters,
        mask_enabled: bool_or(fx, "filterMaskEnable", true),
        mask_linked: bool_or(fx, "filterMaskLinked", false),
    }
}

// ---------- the filters we model ----------

/// Photoshop filter → PhotoCraft command, keyed by the `Fltr` class (and `filterID`).
struct Known {
    class: &'static str,
    filter_id: i32,
    /// `Nm  ` as Photoshop writes it (the menu item, with its ellipsis and mnemonic).
    name: &'static str,
    command: &'static str,
}

const fn code(c: &[u8; 4]) -> i32 {
    i32::from_be_bytes(*c)
}

const KNOWN: [Known; 14] = [
    Known { class: "GsnB", filter_id: code(b"GsnB"), name: "Gaussian Blur...", command: "filter.blur.gaussianBlur" },
    Known { class: "boxblur", filter_id: 840, name: "Box Blur...", command: "filter.blur.boxBlur" },
    Known { class: "MtnB", filter_id: code(b"MtnB"), name: "Motion Blur...", command: "filter.blur.motionBlur" },
    Known { class: "UnsM", filter_id: code(b"UnsM"), name: "Unsharp Mask...", command: "filter.sharpen.unsharpMask" },
    Known { class: "AdNs", filter_id: code(b"AdNs"), name: "Add Noise...", command: "filter.noise.addNoise" },
    Known { class: "Mdn ", filter_id: code(b"Mdn "), name: "Median...", command: "filter.noise.median" },
    Known { class: "HghP", filter_id: code(b"HghP"), name: "High Pass...", command: "filter.other.highPass" },
    Known { class: "Mxm ", filter_id: code(b"Mxm "), name: "Maximum...", command: "filter.other.maximum" },
    Known { class: "Mnm ", filter_id: code(b"Mnm "), name: "Minimum...", command: "filter.other.minimum" },
    Known { class: "Msc ", filter_id: code(b"Msc "), name: "Mosaic...", command: "filter.pixelate.mosaic" },
    Known { class: "Embs", filter_id: code(b"Embs"), name: "Emboss...", command: "filter.stylize.emboss" },
    Known { class: "Lvls", filter_id: code(b"Lvls"), name: "Levels...", command: "image.adjustments.levels" },
    Known { class: "Crvs", filter_id: code(b"Crvs"), name: "Curves...", command: "image.adjustments.curves" },
    // Shadows/Highlights at its defaults stores no `Fltr` (`hasoptions` false).
    Known { class: "adaptCorrect", filter_id: 781, name: "Shado&ws/Highlights...", command: "image.adjustments.shadowsHighlights" },
];

fn known_by_command(command: &str) -> Option<&'static Known> {
    KNOWN.iter().find(|k| k.command == command)
}

/// RGB channel references in Levels/Curves (`Chnl` enum) and their parameter keys ("" = the
/// composite: top-level keys).
const CHANNELS: [(&[u8; 4], &str); 4] = [(b"Cmps", ""), (b"Rd  ", "red"), (b"Grn ", "green"), (b"Bl  ", "blue")];

fn px(d: &Descriptor, key: &str) -> Option<f64> {
    num(d.get(key)).filter(|v| v.is_finite())
}

fn int(d: &Descriptor, key: &str) -> Option<i32> {
    match d.get(key)? {
        Value::Integer(i) => Some(*i),
        _ => None,
    }
}

fn channel_key(d: &Descriptor) -> Option<&'static str> {
    let Some(Value::Reference(items)) = d.get("Chnl") else { return None };
    let Some(ReferenceItem::Enumerated { value, .. }) = items.first() else { return None };
    CHANNELS.iter().find(|(c, _)| value.as_bytes() == &c[..]).map(|(_, k)| *k)
}

fn adjustment_list(d: &Descriptor) -> Option<Vec<&Descriptor>> {
    let Some(Value::List(items)) = d.get("Adjs") else { return None };
    items
        .iter()
        .map(|v| match v {
            Value::Descriptor(x) => Some(x),
            _ => None,
        })
        .collect()
}

fn levels_params(f: &Descriptor) -> Option<J> {
    let mut out = Map::new();
    for a in adjustment_list(f)? {
        let key = channel_key(a)?;
        let mut o = Map::new();
        let pair = |k: &str| match a.get(k) {
            Some(Value::List(v)) if v.len() == 2 => Some((int_value(&v[0])?, int_value(&v[1])?)),
            _ => None,
        };
        if let Some((lo, hi)) = pair("Inpt") {
            o.insert("inBlack".into(), json!(lo));
            o.insert("inWhite".into(), json!(hi));
        }
        if let Some(g) = px(a, "Gmm ") {
            o.insert("gamma".into(), json!(g));
        }
        if let Some((lo, hi)) = pair("Otpt") {
            o.insert("outBlack".into(), json!(lo));
            o.insert("outWhite".into(), json!(hi));
        }
        if key.is_empty() {
            out.extend(o);
        } else {
            out.insert(key.into(), J::Object(o));
        }
    }
    Some(J::Object(out))
}

fn int_value(v: &Value) -> Option<i32> {
    match v {
        Value::Integer(i) => Some(*i),
        _ => None,
    }
}

fn curves_params(f: &Descriptor) -> Option<J> {
    let mut out = Map::new();
    for a in adjustment_list(f)? {
        let key = channel_key(a)?;
        // A pencil-drawn curve is a mapping of 256 output levels (`Mpng`) instead of points.
        if let Some(Value::List(map)) = a.get("Mpng") {
            let map: Option<Vec<J>> = map.iter().map(|v| int_value(v).map(|o| json!(o.clamp(0, 255)))).collect();
            out.insert(if key.is_empty() { "points".into() } else { key.into() }, J::Array(map.filter(|m| m.len() == 256)?));
            out.insert("pencil".into(), J::Bool(true));
            continue;
        }
        let Some(Value::List(pts)) = a.get("Crv ") else { return None };
        let pts: Option<Vec<J>> = pts
            .iter()
            .map(|p| match p {
                Value::Descriptor(p) => Some(json!([px(p, "Hrzn")?, px(p, "Vrtc")?])),
                _ => None,
            })
            .collect();
        out.insert(if key.is_empty() { "points".into() } else { key.into() }, J::Array(pts?));
    }
    Some(J::Object(out))
}

/// Our command and parameters for a known Photoshop filter; `None` when the filter (or one of its
/// settings) is not modelled.
fn known_params(class: &str, f: Option<&Descriptor>) -> Option<(&'static str, J)> {
    let k = KNOWN.iter().find(|k| k.class == class)?;
    let params = match (class, f) {
        ("adaptCorrect", None) => json!({}),
        (_, None) => return None,
        ("GsnB" | "boxblur" | "Mdn " | "HghP" | "Mxm " | "Mnm ", Some(f)) => json!({"radius": px(f, "Rds ")?}),
        ("MtnB", Some(f)) => json!({"angle": int(f, "Angl")?, "distance": px(f, "Dstn")?}),
        ("UnsM", Some(f)) => json!({"amount": px(f, "Amnt")?, "radius": px(f, "Rds ")?, "threshold": int(f, "Thsh")?}),
        ("AdNs", Some(f)) => {
            let dist = match enum_of(f, "Dstr")? {
                b"Unfr" => "uniform",
                b"Gsn " => "gaussian",
                _ => return None,
            };
            let mono = match f.get("Mnch") {
                Some(Value::Boolean(b)) => *b,
                _ => return None,
            };
            json!({"amount": px(f, "Nose")?, "distribution": dist, "monochromatic": mono, "seed": int(f, "FlRs").unwrap_or(0)})
        }
        ("Msc ", Some(f)) => json!({"cellSize": px(f, "ClSz")?}),
        ("Embs", Some(f)) => json!({"angle": int(f, "Angl")?, "height": int(f, "Hght")?, "amount": int(f, "Amnt")?}),
        ("Lvls", Some(f)) => levels_params(f)?,
        ("Crvs", Some(f)) => curves_params(f)?,
        _ => return None,
    };
    Some((k.command, params))
}

/// Lowercase hex of `bytes`.
fn to_hex(bytes: &[u8]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from(H[usize::from(b >> 4)]));
        s.push(char::from(H[usize::from(b & 15)]));
    }
    s
}

fn from_hex(s: &str) -> Option<Vec<u8>> {
    let d = |c: u8| (c as char).to_digit(16).map(|v| v as u8);
    let b = s.as_bytes();
    if !b.len().is_multiple_of(2) {
        return None;
    }
    b.as_chunks::<2>().0.iter().map(|c| Some(d(c[0])? << 4 | d(c[1])?)).collect()
}

/// One `filterFXList` item → [`SmartFilter`] (an [`UNSUPPORTED_FILTER`] when not modelled).
pub fn filter_from_item(item: &Descriptor) -> SmartFilter {
    let opts = get_desc(item, "blendOptions");
    let opacity = opts.and_then(|o| px(o, "Opct")).map_or(1.0, |v| (v / 100.0).clamp(0.0, 1.0) as f32);
    let blend = opts.and_then(|o| enum_of(o, "Md  ")).and_then(crate::effects_map::blend_from_id).unwrap_or(BlendMode::Normal);
    let visible = bool_or(item, "enab", true);
    let fltr = get_desc(item, "Fltr");
    let class = fltr.map(|f| String::from_utf8_lossy(f.class_id.as_bytes()).into_owned());
    let filter_id = int(item, "filterID").unwrap_or(0);
    let class = class.or_else(|| KNOWN.iter().find(|k| k.filter_id == filter_id).map(|k| k.class.to_string()));
    let known = match class.as_deref() {
        Some(camera_raw::CLASS) => camera_raw::import_params(item).map(|p| (camera_raw::COMMAND, p)),
        Some(displace::CLASS) => displace::import_params(item).map(|p| (displace::COMMAND, p)),
        c => c.and_then(|c| known_params(c, fltr)),
    };
    let (command, params) = match known {
        Some((c, p)) => (c.to_string(), p),
        None => {
            let name = text_of(item, "Nm  ").unwrap_or_default();
            // The stored descriptor carries neutral blending (the model's own fields replace
            // them on export), so hiding or re-blending the filter leaves its params alone.
            let mut neutral = item.clone();
            for (k, v) in &mut neutral.items {
                if k.is("blendOptions") {
                    *v = blend_options(&SmartFilter { command: String::new(), params: J::Null, blend: BlendMode::Normal, opacity: 1.0, visible: true });
                } else if k.is("enab") {
                    *v = Value::Boolean(true);
                }
            }
            (UNSUPPORTED_FILTER.to_string(), json!({"name": name.replace('&', ""), "filterID": filter_id, "psd": to_hex(&neutral.to_bytes())}))
        }
    };
    SmartFilter { command, params, blend, opacity, visible }
}

// ---------- writing ----------

fn unit(u: &[u8; 4], v: f64) -> Value {
    Value::UnitFloat { unit: *u, value: v }
}

fn rgbc(v: f64) -> Value {
    Value::Descriptor(Descriptor::new("RGBC").with("Rd  ", Value::Double(v)).with("Grn ", Value::Double(v)).with("Bl  ", Value::Double(v)))
}

fn blend_options(f: &SmartFilter) -> Value {
    let opacity = if f.opacity.is_finite() { f64::from(f.opacity.clamp(0.0, 1.0)) * 100.0 } else { 100.0 };
    Value::Descriptor(Descriptor::new("blendOptions").with("Opct", unit(b"#Prc", opacity)).with("Md  ", crate::effects_map::blend_long_value(f.blend)))
}

/// Parameter keys that are bookkeeping, not filter settings.
fn is_meta(k: &str) -> bool {
    k == "layer" || k.starts_with("__")
}

fn pf(p: &J, key: &str, default: f64) -> f64 {
    p.get(key).and_then(J::as_f64).filter(|v| v.is_finite()).unwrap_or(default)
}

fn pi(p: &J, key: &str, default: f64) -> Value {
    Value::Integer(pf(p, key, default).round().clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32)
}

fn channel_ref(code: &[u8; 4]) -> Value {
    Value::Reference(vec![ReferenceItem::Enumerated {
        class: Class { name: UnicodeString::new_nul(""), class_id: Id::new("Chnl") },
        type_id: Id::new("Chnl"),
        value: Id::Code(*code),
    }])
}

/// The `Adjs` entries of a Levels or Curves filter: the composite first, then red, green, blue.
fn adjustments(p: &J, curves: bool) -> Result<Value, String> {
    let obj = p.as_object().ok_or("parameters are not an object")?;
    let allowed: &[&str] =
        if curves { &["points", "red", "green", "blue", "pencil"] } else { &["inBlack", "inWhite", "gamma", "outBlack", "outWhite", "red", "green", "blue"] };
    if let Some(k) = obj.keys().find(|k| !is_meta(k) && !allowed.contains(&k.as_str())) {
        return Err(format!("setting `{k}` has no PSD equivalent"));
    }
    let mut list = Vec::new();
    for (code, key) in CHANNELS {
        let o = if key.is_empty() { Some(p) } else { p.get(key) };
        let Some(o) = o else { continue };
        let mut a = Descriptor::new(if curves { "CrvA" } else { "LvlA" }).with("Chnl", channel_ref(code));
        if curves {
            let src = if key.is_empty() { p.get("points") } else { Some(o) };
            let Some(pts) = src.and_then(J::as_array) else { continue };
            if p.get("pencil").and_then(J::as_bool) == Some(true) {
                let map: Option<Vec<Value>> =
                    pts.iter().map(|o| o.as_f64().filter(|o| o.is_finite()).map(|o| Value::Integer(o.round().clamp(0.0, 255.0) as i32))).collect();
                a = a.with("Mpng", Value::List(map.filter(|m| m.len() == 256).ok_or("a pencil-drawn curve is not 256 output levels")?));
                list.push(Value::Descriptor(a));
                continue;
            }
            let pts: Option<Vec<Value>> = pts
                .iter()
                .map(|q| {
                    let x = q.get(0)?.as_f64().filter(|v| v.is_finite())?;
                    let y = q.get(1)?.as_f64().filter(|v| v.is_finite())?;
                    Some(Value::Descriptor(Descriptor::new("Pnt ").with("Hrzn", Value::Double(x)).with("Vrtc", Value::Double(y))))
                })
                .collect();
            a = a.with("Crv ", Value::List(pts.ok_or("a curve point is not [input, output]")?));
        } else {
            if key.is_empty() && !["inBlack", "inWhite", "gamma", "outBlack", "outWhite"].iter().any(|k| o.get(k).is_some()) {
                continue;
            }
            a = a.with("Inpt", Value::List(vec![pi(o, "inBlack", 0.0), pi(o, "inWhite", 255.0)])).with("Gmm ", Value::Double(pf(o, "gamma", 1.0)));
            if o.get("outBlack").is_some() || o.get("outWhite").is_some() {
                a = a.with("Otpt", Value::List(vec![pi(o, "outBlack", 0.0), pi(o, "outWhite", 255.0)]));
            }
        }
        list.push(Value::Descriptor(a));
    }
    Ok(Value::List(list))
}

/// The `Fltr` descriptor of a modelled filter (`None` = written without options).
fn fltr_for(k: &Known, p: &J) -> Result<Option<Descriptor>, String> {
    let d = Descriptor::new(k.class);
    Ok(Some(match k.class {
        "GsnB" | "boxblur" | "Mdn " | "HghP" | "Mxm " | "Mnm " => {
            if p.get("preserve").and_then(J::as_str).is_some_and(|s| s == "roundness") {
                return Err("“preserve roundness” has no PSD equivalent".into());
            }
            d.with("Rds ", unit(b"#Pxl", pf(p, "radius", 1.0)))
        }
        "MtnB" => d.with("Angl", pi(p, "angle", 0.0)).with("Dstn", unit(b"#Pxl", pf(p, "distance", 10.0))),
        "UnsM" => d.with("Amnt", unit(b"#Prc", pf(p, "amount", 50.0))).with("Rds ", unit(b"#Pxl", pf(p, "radius", 1.0))).with("Thsh", pi(p, "threshold", 0.0)),
        "AdNs" => {
            let gauss = p.get("distribution").and_then(J::as_str) == Some("gaussian");
            d.with("Dstr", Value::Enumerated { type_id: Id::new("Dstr"), value: Id::new(if gauss { "Gsn " } else { "Unfr" }) })
                .with("Nose", unit(b"#Prc", pf(p, "amount", 12.5)))
                .with("Mnch", Value::Boolean(p.get("monochromatic").and_then(J::as_bool).unwrap_or(false)))
                .with("FlRs", pi(p, "seed", 0.0))
        }
        "Msc " => d.with("ClSz", unit(b"#Pxl", pf(p, "cellSize", 10.0))),
        "Embs" => d.with("Angl", pi(p, "angle", 135.0)).with("Hght", pi(p, "height", 3.0)).with("Amnt", pi(p, "amount", 100.0)),
        "Lvls" => d.with("Adjs", adjustments(p, false)?),
        "Crvs" => d.with("Adjs", adjustments(p, true)?),
        "adaptCorrect" => {
            if p.as_object().is_some_and(|o| o.keys().any(|k| !is_meta(k))) {
                return Err("only default Shadows/Highlights settings are written to PSD".into());
            }
            return Ok(None);
        }
        _ => return Err("not a recognized filter".into()),
    }))
}

/// The `filterFXList` item for one smart filter, or why it can't be written.
pub fn item_for_filter(f: &SmartFilter) -> Result<Descriptor, String> {
    if f.command == camera_raw::COMMAND {
        let mut d = camera_raw::export_item(&f.params)?;
        set(&mut d, "blendOptions", blend_options(f));
        set(&mut d, "enab", Value::Boolean(f.visible));
        return Ok(d);
    }
    if f.command == displace::COMMAND {
        let mut d = displace::export_item(&f.params)?;
        set(&mut d, "blendOptions", blend_options(f));
        set(&mut d, "enab", Value::Boolean(f.visible));
        return Ok(d);
    }
    if f.command == UNSUPPORTED_FILTER {
        let raw = f.params.get("psd").and_then(J::as_str).and_then(from_hex).ok_or("its filter data is missing")?;
        let mut d = Descriptor::from_bytes(&raw).map_err(|e| format!("its filter data is unreadable ({e})"))?;
        set(&mut d, "blendOptions", blend_options(f));
        set(&mut d, "enab", Value::Boolean(f.visible));
        return Ok(d);
    }
    let k = known_by_command(&f.command).ok_or("this filter has no PSD equivalent")?;
    let fltr = fltr_for(k, &f.params)?;
    let mut d = Descriptor::new("filterFX")
        .with("Nm  ", Value::Text(UnicodeString::new_nul(k.name)))
        .with("blendOptions", blend_options(f))
        .with("enab", Value::Boolean(f.visible))
        .with("hasoptions", Value::Boolean(fltr.is_some()))
        .with("FrgC", rgbc(0.0))
        .with("BckC", rgbc(255.0));
    if let Some(x) = fltr {
        d = d.with("Fltr", Value::Descriptor(x));
    }
    Ok(d.with("filterID", Value::Integer(k.filter_id)))
}

/// Builds a `filterFX` object; filters that can't be written are left out, with a message each.
pub fn filter_fx(stack: &FilterStack, layer: &str, warnings: &mut Vec<String>) -> Descriptor {
    let mut list = Vec::new();
    for f in &stack.filters {
        match item_for_filter(f) {
            Ok(d) => list.push(Value::Descriptor(d)),
            Err(e) => warnings.push(format!("layer \"{layer}\": smart filter {} was not written to PSD: {e}", f.command)),
        }
    }
    Descriptor::new("filterFXStyle")
        .with("enab", Value::Boolean(stack.enabled))
        .with("validAtPosition", Value::Boolean(true))
        .with("filterMaskEnable", Value::Boolean(stack.mask_enabled))
        .with("filterMaskLinked", Value::Boolean(stack.mask_linked))
        .with("filterMaskExtendWithWhite", Value::Boolean(true))
        .with("filterFXList", Value::List(list))
}

/// Replaces item `key` in place, or appends it.
fn set(d: &mut Descriptor, key: &str, v: Value) {
    match d.items.iter_mut().find(|(k, _)| k.is(key)) {
        Some(e) => e.1 = v,
        None => d.items.push((Id::new(key), v)),
    }
}

/// Everything needed to write the placed-layer blocks of a smart object.
#[derive(Clone, Debug)]
pub struct PlacedSpec<'a> {
    pub idnt: &'a str,
    pub placed: &'a str,
    /// Source pixels → document pixels.
    pub transform: Affine,
    /// Distort / Perspective: the full projective map (row-major 3×3), overriding `transform`.
    pub perspective: Option<[f64; 9]>,
    /// Source size in pixels.
    pub size: (f64, f64),
    /// Source resolution.
    pub dpi: f64,
    pub warp: Option<&'a Warp>,
    /// `filterFX`, when the smart object has filters.
    pub filter_fx: Option<Descriptor>,
}

/// Where the frame corners land (top-left, top-right, bottom-right, bottom-left), as x, y pairs.
/// The frame is [`crate::blocks::placed_frame`] for an affine placement (the inverse of
/// `parse_smart`), the source rect for a projective one (as `parse_smart_perspective` reads it).
fn quad_points(s: &PlacedSpec<'_>) -> [f64; 8] {
    let (w, h) = s.size;
    let [a, b, c, d, e, f] = s.transform.m;
    let map = |x: f64, y: f64| match &s.perspective {
        Some(p) => photocraft_algo::transform::Homography(*p).apply(x, y),
        None => (a * x + c * y + e, b * x + d * y + f),
    };
    let [x0, y0, x1, y1] = if s.perspective.is_some() { [0.0, 0.0, w, h] } else { crate::blocks::placed_frame(s.warp, w, h) };
    let mut out = [0.0; 8];
    for (i, (x, y)) in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)].into_iter().enumerate() {
        (out[2 * i], out[2 * i + 1]) = map(x, y);
    }
    out
}

/// The transform quad: the source corners (top-left, top-right, bottom-right, bottom-left).
fn quad(s: &PlacedSpec<'_>) -> Value {
    Value::List(quad_points(s).into_iter().map(Value::Double).collect())
}

/// Has the placement moved from the quad a template descriptor stores (`Trnf`)?
fn quad_moved(template: &Descriptor, s: &PlacedSpec<'_>) -> bool {
    let Some(Value::List(pts)) = template.get("Trnf") else { return true };
    let old: Vec<f64> = pts.iter().filter_map(|v| crate::blocks::num(Some(v))).collect();
    let new = quad_points(s);
    old.len() != 8 || old.iter().zip(new).any(|(a, b)| (a - b).abs() > 1e-6 * (1.0 + a.abs()))
}

fn enumv(t: &str, v: &str) -> Value {
    Value::Enumerated { type_id: Id::new(t), value: Id::new(v) }
}

/// The `warp` descriptor (also the `PlLd` warp). Multi-patch meshes have no 1 × 1 envelope and
/// are written as no warp, with a message.
fn warp_desc(w: Option<&Warp>, size: (f64, f64), warnings: &mut Vec<String>) -> Value {
    let none = Warp::none([0.0, 0.0, size.0, size.1]);
    let mut w = w.unwrap_or(&none).clone();
    let mesh = match (&w.style, &w.mesh) {
        (WarpStyle::Custom, Some(m)) if m.points.len() == 16 => Some(m.clone()),
        (WarpStyle::Custom, _) => {
            warnings.push("a custom warp with more than one patch was written to PSD without its warp".into());
            w = none.clone();
            None
        }
        _ => None,
    };
    let [x0, y0, x1, y1] = w.bounds;
    let bounds = Descriptor::new("classFloatRect")
        .with("Top ", Value::Double(y0))
        .with("Left", Value::Double(x0))
        .with("Btom", Value::Double(y1))
        .with("Rght", Value::Double(x1));
    let mut d = Descriptor::new("warp")
        .with("warpStyle", enumv("warpStyle", w.style.psd_name()))
        .with("warpValue", Value::Double(w.bend))
        .with("warpPerspective", Value::Double(w.h_distort))
        .with("warpPerspectiveOther", Value::Double(w.v_distort))
        .with("warpRotate", enumv("Ornt", if w.vertical { "Vrtc" } else { "Hrzn" }))
        .with("bounds", Value::Descriptor(bounds))
        .with("uOrder", Value::Integer(4))
        .with("vOrder", Value::Integer(4));
    if let Some(m) = mesh {
        let body = Descriptor::new("rationalPoint")
            .with("Hrzn", Value::UnitFloats { unit: *b"#Pxl", values: m.points.iter().map(|p| p[0]).collect() })
            .with("Vrtc", Value::UnitFloats { unit: *b"#Pxl", values: m.points.iter().map(|p| p[1]).collect() });
        d = d.with(
            "customEnvelopeWarp",
            Value::Descriptor(Descriptor::new("customEnvelopeWarp").with("meshPoints", Value::ObjectArray(ObjectArray { prefix: 16, body }))),
        );
    }
    Value::Descriptor(d)
}

fn ratio(n: i32, d: i32) -> Value {
    Value::Descriptor(Descriptor::new("null").with("numerator", Value::Integer(n)).with("denominator", Value::Integer(d)))
}

fn pad4(mut v: Vec<u8>) -> Vec<u8> {
    while !v.len().is_multiple_of(4) {
        v.push(0);
    }
    v
}

/// `SoLd` block data. With `template` (the imported descriptor) every key it has is kept and only
/// the placed id (for a duplicate), transform (when it moved), warp and filter stack are rewritten;
/// otherwise a fresh descriptor in Photoshop's key order.
pub fn sold_bytes(template: Option<&Descriptor>, s: &PlacedSpec<'_>, warnings: &mut Vec<String>) -> Vec<u8> {
    let warp = warp_desc(s.warp, s.size, warnings);
    let d = match template {
        Some(t) => {
            let mut d = t.clone();
            if text_of(t, "placed").as_deref() != Some(s.placed) {
                set(&mut d, "placed", Value::Text(UnicodeString::new_nul(s.placed)));
            }
            if quad_moved(t, s) {
                set(&mut d, "Trnf", quad(s));
                set(&mut d, "nonAffineTransform", quad(s));
            }
            set(&mut d, "warp", warp);
            match &s.filter_fx {
                Some(fx) => set(&mut d, "filterFX", Value::Descriptor(fx.clone())),
                None => d.items.retain(|(k, _)| !k.is("filterFX")),
            }
            d
        }
        None => {
            let mut d = Descriptor::new("null")
                .with("Idnt", Value::Text(UnicodeString::new_nul(s.idnt)))
                .with("placed", Value::Text(UnicodeString::new_nul(s.placed)))
                .with("PgNm", Value::Integer(1))
                .with("totalPages", Value::Integer(1))
                .with("Crop", Value::Integer(1))
                .with("frameStep", ratio(0, 600))
                .with("duration", ratio(0, 600))
                .with("frameCount", Value::Integer(1))
                .with("Annt", Value::Integer(16))
                .with("Type", Value::Integer(2))
                .with("Trnf", quad(s))
                .with("nonAffineTransform", quad(s))
                .with("warp", warp)
                .with("Sz  ", Value::Descriptor(Descriptor::new("Pnt ").with("Wdth", Value::Double(s.size.0)).with("Hght", Value::Double(s.size.1))))
                .with("Rslt", unit(b"#Rsl", s.dpi));
            if let Some(fx) = &s.filter_fx {
                d = d.with("filterFX", Value::Descriptor(fx.clone()));
            }
            d.with("comp", Value::Integer(-1))
                .with("compInfo", Value::Descriptor(Descriptor::new("null").with("compID", Value::Integer(-1)).with("originalCompID", Value::Integer(-1))))
        }
    };
    let mut out = b"soLD".to_vec();
    out.extend_from_slice(&4u32.to_be_bytes());
    out.extend(VersionedDescriptor::new(d).to_bytes());
    pad4(out)
}

/// `PlLd` block data (`plcL` version 3): the legacy placed-layer record Photoshop writes next to
/// `SoLd`.
pub fn plld_bytes(s: &PlacedSpec<'_>, warnings: &mut Vec<String>) -> Vec<u8> {
    let mut out = b"plcL".to_vec();
    out.extend_from_slice(&3u32.to_be_bytes());
    let id = s.idnt.as_bytes();
    let n = id.len().min(255);
    out.push(n as u8);
    out.extend_from_slice(&id[..n]);
    for v in [1i32, 1, 16, 2] {
        out.extend_from_slice(&v.to_be_bytes());
    }
    if let Value::List(q) = quad(s) {
        for v in q {
            if let Value::Double(x) = v {
                out.extend_from_slice(&x.to_be_bytes());
            }
        }
    }
    out.extend_from_slice(&0u32.to_be_bytes());
    let Value::Descriptor(w) = warp_desc(s.warp, s.size, warnings) else { return pad4(out) };
    out.extend(VersionedDescriptor::new(w).to_bytes());
    pad4(out)
}

/// The source size a `soLD` descriptor stores (`Sz  `).
pub fn stored_size(d: &Descriptor) -> Option<(f64, f64)> {
    let sz = get_desc(d, "Sz  ")?;
    Some((num(sz.get("Wdth"))?, num(sz.get("Hght"))?)).filter(|(w, h)| *w > 0.0 && *h > 0.0)
}

// ---------- the filter mask (`FEid`) ----------

/// The filter mask stored in an `FEid` item, in document coordinates and white beyond its bounds
/// (`filterMaskExtendWithWhite`), as a grayscale mask of `sample` depth. `Ok(None)` when the
/// item has no mask or it is all white (no mask); `Err` when it can't be decoded.
pub fn mask_from_item(item: &FilterEffectsItem, sample: SampleType, stack: &FilterStack) -> Result<Option<LayerMask>, String> {
    if item.mask.is_none() {
        return Ok(None);
    }
    let (rect, plane) = item.decoded_mask().ok_or("the smart filter mask could not be decoded")?;
    if item.depth != u32::from(crate::pixels::psd_depth(sample)) {
        return Err(format!("the smart filter mask is {}-bit in a {:?} document", item.depth, sample));
    }
    let (w, h) = rect.size().map_err(|e| e.to_string())?;
    let fmt = PixelFormat::new(ColorMode::Grayscale, sample, false);
    let mut surface = Surface::with_default(fmt, &[1.0]);
    let n = w.checked_mul(h).ok_or("the smart filter mask is too large")?;
    let bytes = crate::pixels::interleave(&[Some(&plane)], &[crate::pixels::zero_sample(sample)], n, sample, &[false]);
    surface.write_interleaved(GeomRect::new(rect.left, rect.top, rect.right, rect.bottom), &bytes);
    surface.prune();
    if surface.content_bounds().is_empty() {
        return Ok(None);
    }
    Ok(Some(LayerMask { surface, enabled: stack.mask_enabled, linked: stack.mask_linked, density: 1.0, feather: 0.0 }))
}

/// An `FEid` item for a smart object with filters: the unfiltered placed pixels Photoshop
/// filters from (colour channels from slot 0, transparency in the last slot; in the document's
/// pixel format, CMYK inverted as in layer channels) and the filter mask over `bounds` (white
/// when `mask` is `None`). Photoshop needs the planes: re-rendering the filters of an item
/// without them crashes it. `None` when there are no unfiltered pixels.
pub fn feid_item(placed: &str, unfiltered: &Surface, mask: Option<&LayerMask>, bounds: GeomRect, doc_fmt: PixelFormat) -> Option<FilterEffectsItem> {
    let sample = doc_fmt.sample;
    let px = if unfiltered.format() == doc_fmt { unfiltered.clone() } else { unfiltered.convert(doc_fmt) };
    let content = px.content_bounds();
    if content.is_empty() {
        return None;
    }
    // Over the whole area the filters may draw into: Photoshop clips the filtered result to these
    // planes, so the content bounds alone would cut blurs off at the source's edge.
    let r = content.union(&bounds);
    let cc = doc_fmt.mode.color_channels();
    let depth = crate::pixels::psd_depth(sample);
    let compression = if sample == SampleType::F32 { Compression::ZipPrediction } else { Compression::Rle };
    let encode = |plane: &[u8], w: usize, h: usize| {
        EffectsPlane::encode(compression, plane, w, h, depth)
            .or_else(|_| EffectsPlane::encode(Compression::Raw, plane, w, h, depth))
            .unwrap_or(EffectsPlane { compression: Compression::Raw, data: plane.to_vec() })
    };
    let mut invert = vec![doc_fmt.mode == ColorMode::Cmyk; cc];
    invert.push(false);
    // Fully transparent pixels carry white, as in Photoshop's own cache: filters that look at
    // colour regardless of alpha (Shadows/Highlights) read it.
    let mut vals = px.read_region(r);
    let white = photocraft_raster::from_rgba(&doc_fmt, [1.0, 1.0, 1.0, 1.0]);
    for p in vals.chunks_exact_mut(cc + 1) {
        if p[cc] <= 0.0 {
            p[..cc].copy_from_slice(&white[..cc]);
        }
    }
    let mut filled = Surface::new(doc_fmt);
    filled.write_region(r, &vals);
    drop(vals);
    let mut bytes = filled.to_interleaved(r);
    if doc_fmt.mode == ColorMode::Lab && sample == SampleType::U16 {
        // 16-bit Lab a*/b* in Photoshop's scale, as in layer channels.
        crate::pixels::lab16_chroma(&mut bytes, cc + 1, false);
    }
    let planes = crate::pixels::deinterleave(&bytes, cc + 1, sample, &invert);
    let (w, h) = (r.width() as usize, r.height() as usize);
    let mut slots = vec![None; 26];
    for (i, p) in planes.iter().enumerate() {
        let slot = if i == cc { 25 } else { i };
        if let Some(s) = slots.get_mut(slot) {
            *s = Some(encode(p, w, h));
        }
    }
    let mut item = filter_mask_item(placed, mask, bounds, sample);
    item.rect = PsdRect { top: r.y0, left: r.x0, bottom: r.y1, right: r.x1 };
    item.slots = slots;
    Some(item)
}

fn filter_mask_item(placed: &str, mask: Option<&LayerMask>, bounds: GeomRect, sample: SampleType) -> FilterEffectsItem {
    let fmt = PixelFormat::new(ColorMode::Grayscale, sample, false);
    let area = match mask {
        Some(m) => m.surface.content_bounds().union(&bounds),
        None => bounds,
    };
    let area = if area.is_empty() { GeomRect::new(0, 0, 1, 1) } else { area };
    let rect = PsdRect { top: area.y0, left: area.x0, bottom: area.y1, right: area.x1 };
    let (w, h) = (area.width() as usize, area.height() as usize);
    let plane = match mask {
        Some(m) => {
            let s = if m.surface.format() == fmt { m.surface.clone() } else { m.surface.convert(fmt) };
            crate::pixels::deinterleave(&s.to_interleaved(area), 1, sample, &[false]).into_iter().next().unwrap_or_default()
        }
        None => {
            let mut white = Vec::with_capacity(w * h * sample.bytes());
            for _ in 0..w * h {
                crate::pixels::encode_be(1.0, sample, &mut white);
            }
            white
        }
    };
    let depth = crate::pixels::psd_depth(sample);
    let compression = if sample == SampleType::F32 { Compression::ZipPrediction } else { Compression::Rle };
    let plane = EffectsPlane::encode(compression, &plane, w, h, depth)
        .or_else(|_| EffectsPlane::encode(Compression::Raw, &plane, w, h, depth))
        .unwrap_or(EffectsPlane { compression: Compression::Raw, data: plane });
    FilterEffectsItem { id: placed.to_string(), version: 1, rect, depth: u32::from(depth), max_channels: 24, slots: vec![None; 26], mask: Some((rect, plane)) }
}

/// A deterministic uuid-shaped id (8-4-4-4-12 hex digits) derived from `seed`.
pub fn uuid_from(seed: &[u8]) -> String {
    let h = blake3::hash(seed);
    let x = to_hex(&h.as_bytes()[..16]);
    format!("{}-{}-{}-{}-{}", &x[..8], &x[8..12], &x[12..16], &x[16..20], &x[20..32])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count(hay: &[u8], needle: &[u8]) -> usize {
        hay.windows(needle.len()).filter(|w| *w == needle).count()
    }

    /// A freshly built placed layer writes `warp` (key and class) as a length-4 string, as
    /// Photoshop does; the code form is a different key to Photoshop.
    #[test]
    fn fresh_placed_warp_is_a_string_id() {
        let spec =
            PlacedSpec { idnt: "id", placed: "id", transform: Affine::IDENTITY, perspective: None, size: (10.0, 10.0), dpi: 72.0, warp: None, filter_fx: None };
        let mut warnings = Vec::new();
        let sold = sold_bytes(None, &spec, &mut warnings);
        let plld = plld_bytes(&spec, &mut warnings);
        // `SoLd`: the `warp` key and the warp descriptor's class; `PlLd`: the descriptor's class.
        for (bytes, n) in [(&sold, 2), (&plld, 1)] {
            assert_eq!(count(bytes, b"\0\0\0\x04warp"), n);
            assert_eq!(count(bytes, b"\0\0\0\0warp"), 0);
        }
        let d = sold_descriptor(&sold).unwrap();
        assert!(matches!(d.get("warp"), Some(Value::Descriptor(w)) if w.class_id == Id::Str(b"warp".to_vec())));
    }

    fn sf(command: &str, params: J) -> SmartFilter {
        SmartFilter { command: command.into(), params, blend: BlendMode::Normal, opacity: 1.0, visible: true }
    }

    #[test]
    fn modelled_filters_round_trip() {
        let filters = vec![
            sf("filter.blur.gaussianBlur", json!({"radius": 4.0})),
            sf("filter.blur.boxBlur", json!({"radius": 5.0})),
            sf("filter.blur.motionBlur", json!({"angle": -20, "distance": 10.0})),
            sf("filter.sharpen.unsharpMask", json!({"amount": 150.0, "radius": 2.0, "threshold": 0})),
            sf("filter.noise.addNoise", json!({"amount": 12.0, "distribution": "uniform", "monochromatic": false, "seed": 1234})),
            sf("filter.noise.median", json!({"radius": 3.0})),
            sf("filter.other.highPass", json!({"radius": 3.0})),
            sf("filter.other.maximum", json!({"radius": 2.0})),
            sf("filter.other.minimum", json!({"radius": 2.0})),
            sf("filter.pixelate.mosaic", json!({"cellSize": 8.0})),
            sf("filter.stylize.emboss", json!({"angle": 135, "height": 3, "amount": 100})),
            sf("image.adjustments.levels", json!({"inBlack": 20, "inWhite": 220, "gamma": 1.4, "red": {"inBlack": 0, "inWhite": 255, "gamma": 0.8}})),
            sf("image.adjustments.curves", json!({"points": [[0.0, 0.0], [64.0, 40.0], [255.0, 255.0]], "blue": [[0.0, 20.0], [255.0, 235.0]]})),
            sf("image.adjustments.shadowsHighlights", json!({})),
            // A pencil-drawn curve: 256 output levels (`Mpng`).
            sf("image.adjustments.curves", json!({"pencil": true, "points": (0..256).rev().collect::<Vec<i32>>()})),
        ];
        for mut f in filters {
            f.blend = BlendMode::Overlay;
            f.opacity = 0.75;
            f.visible = false;
            let d = item_for_filter(&f).unwrap();
            let bytes = d.to_bytes();
            let back = filter_from_item(&Descriptor::from_bytes(&bytes).unwrap());
            assert_eq!(back, f, "{}", f.command);
        }
    }

    #[test]
    fn unknown_filters_keep_their_descriptor() {
        let item = Descriptor::new("filterFX")
            .with("Nm  ", Value::Text(UnicodeString::new_nul("Twirl...")))
            .with("blendOptions", Value::Descriptor(Descriptor::new("blendOptions").with("Opct", unit(b"#Prc", 100.0)).with("Md  ", enumv("BlnM", "normal"))))
            .with("enab", Value::Boolean(true))
            .with("Fltr", Value::Descriptor(Descriptor::new("Twrl").with("Angl", Value::Integer(50))))
            .with("filterID", Value::Integer(code(b"Twrl")));
        let f = filter_from_item(&item);
        assert_eq!(f.command, UNSUPPORTED_FILTER);
        assert_eq!(f.params["name"], "Twirl...");
        assert_eq!(item_for_filter(&f).unwrap(), item);
        // Hidden and re-blended: only those keys change.
        let mut g = f.clone();
        g.visible = false;
        g.blend = BlendMode::Screen;
        let back = filter_from_item(&item_for_filter(&g).unwrap());
        assert_eq!((back.visible, back.blend, &back.params), (false, BlendMode::Screen, &f.params));
        // A known filter with a setting we don't model is kept verbatim too.
        let odd = Descriptor::new("filterFX").with("Fltr", Value::Descriptor(Descriptor::new("GsnB"))).with("filterID", Value::Integer(code(b"GsnB")));
        assert_eq!(filter_from_item(&odd).command, UNSUPPORTED_FILTER);
    }

    #[test]
    fn unwritable_filters_are_reported_not_fatal() {
        let stack = FilterStack {
            filters: vec![
                sf("filter.distort.twirl", json!({"angle": 50})),
                sf("filter.blur.gaussianBlur", json!({"radius": 2.0})),
                sf(UNSUPPORTED_FILTER, json!({"psd": "zz"})),
                sf("image.adjustments.levels", json!({"cyan": {"gamma": 2.0}})),
                sf("image.adjustments.shadowsHighlights", json!({"shadowAmount": 50})),
            ],
            ..Default::default()
        };
        let mut w = Vec::new();
        let d = filter_fx(&stack, "L", &mut w);
        assert_eq!(w.len(), 4, "{w:?}");
        let back = filter_stack(&d);
        assert_eq!(back.filters, vec![stack.filters[1].clone()]);
    }

    #[test]
    fn placed_blocks_parse_back() {
        let t = Affine { m: [0.5, 0.1, -0.1, 0.5, 10.0, 20.0] };
        let stack =
            FilterStack { enabled: false, filters: vec![sf("filter.blur.gaussianBlur", json!({"radius": 3.0}))], mask_enabled: false, mask_linked: true };
        let mut w = Vec::new();
        let spec = PlacedSpec {
            idnt: "id-1",
            placed: "pl-2",
            transform: t,
            perspective: None,
            size: (64.0, 32.0),
            dpi: 72.0,
            warp: None,
            filter_fx: Some(filter_fx(&stack, "L", &mut w)),
        };
        let sold = sold_bytes(None, &spec, &mut w);
        assert!(w.is_empty());
        assert_eq!(sold.len() % 4, 0);
        let p = parse_sold(&sold).unwrap();
        assert_eq!((p.idnt.as_str(), p.placed.as_str()), ("id-1", "pl-2"));
        for (a, b) in p.transform.m.iter().zip(t.m) {
            assert!((a - b).abs() < 1e-9);
        }
        assert_eq!(p.stack, Some(stack));
        let tb = photocraft_psd::TaggedBlock::new(*b"SoLd", sold.clone());
        tb.check_structure().unwrap();
        let pl = photocraft_psd::TaggedBlock::new(*b"PlLd", plld_bytes(&spec, &mut w));
        pl.check_structure().unwrap();
        // A template keeps its other keys; an unmoved transform keeps its exact quad.
        let mut tmpl = sold_descriptor(&sold).unwrap();
        tmpl.items.push((Id::new("extraKey"), Value::Integer(7)));
        let again = sold_bytes(Some(&tmpl), &PlacedSpec { filter_fx: None, ..spec }, &mut w);
        let d = sold_descriptor(&again).unwrap();
        assert_eq!(d.get("extraKey"), Some(&Value::Integer(7)));
        assert!(d.get("filterFX").is_none());
        assert_eq!(d.get("Trnf"), tmpl.get("Trnf"));
    }

    /// A Distort / Perspective placement goes out as its four corners (`Trnf`, `nonAffineTransform`)
    /// and comes back as the same projective map; an affine one reads back as no perspective.
    #[test]
    fn perspective_placement_round_trips_through_sold() {
        let h = photocraft_algo::transform::Homography::rect_to_quad([0.0, 0.0, 64.0, 32.0], [[10.0, 5.0], [80.0, 8.0], [70.0, 50.0], [12.0, 40.0]]).unwrap();
        let mut w = Vec::new();
        let mut spec = PlacedSpec {
            idnt: "id",
            placed: "pl",
            transform: Affine::IDENTITY,
            perspective: Some(h.0),
            size: (64.0, 32.0),
            dpi: 72.0,
            warp: None,
            filter_fx: None,
        };
        let sold = sold_bytes(None, &spec, &mut w);
        let back = crate::blocks::parse_smart_perspective(b"SoLd", &sold).expect("a projective placement");
        let n = |m: [f64; 9]| m.map(|v| v / m[8]);
        for (a, b) in n(back).iter().zip(n(h.0)) {
            assert!((a - b).abs() < 1e-9 * (1.0 + b.abs()), "{back:?} vs {:?}", h.0);
        }
        // Affine placements carry no perspective.
        spec.perspective = None;
        spec.transform = Affine { m: [2.0, 0.0, 0.0, 2.0, 5.0, 7.0] };
        let sold = sold_bytes(None, &spec, &mut w);
        assert_eq!(crate::blocks::parse_smart_perspective(b"SoLd", &sold), None);
        assert_eq!(crate::blocks::parse_smart_perspective(b"PlLd", &sold), None);
    }

    /// `quad_moved` compares the corners a template stores with the placement's: unchanged keeps
    /// the template's quad, a moved corner (affine or projective) or a missing quad rewrites it.
    #[test]
    fn quad_moved_detects_corner_changes() {
        let t = Affine { m: [1.0, 0.0, 0.0, 1.0, 3.0, 4.0] };
        let mut w = Vec::new();
        let mut spec = PlacedSpec { idnt: "id", placed: "pl", transform: t, perspective: None, size: (20.0, 10.0), dpi: 72.0, warp: None, filter_fx: None };
        let template = parse_sold(&sold_bytes(None, &spec, &mut w)).expect("parses").descriptor;
        assert!(!quad_moved(&template, &spec), "same placement");
        spec.transform = Affine { m: [1.0, 0.0, 0.0, 1.0, 3.5, 4.0] };
        assert!(quad_moved(&template, &spec), "moved half a pixel");
        spec.transform = t;
        spec.perspective =
            photocraft_algo::transform::Homography::rect_to_quad([0.0, 0.0, 20.0, 10.0], [[3.0, 4.0], [23.0, 4.0], [20.0, 12.0], [3.0, 14.0]]).map(|h| h.0);
        assert!(quad_moved(&template, &spec), "one corner moved (Distort)");
        assert!(quad_moved(&Descriptor::new("null"), &spec), "no stored quad");
    }

    /// Corrupted placed-layer data and filter descriptors never panic: they parse to something
    /// (unknown filters stay verbatim) or to nothing.
    #[test]
    fn mutated_placed_data_never_panics() {
        let stack = FilterStack {
            filters: vec![
                sf("filter.blur.gaussianBlur", json!({"radius": 3.0})),
                sf("image.adjustments.curves", json!({"points": [[0.0, 0.0], [255.0, 255.0]]})),
                sf("image.adjustments.levels", json!({"inBlack": 3, "inWhite": 250, "gamma": 1.1})),
            ],
            ..Default::default()
        };
        let mut w = Vec::new();
        let spec = PlacedSpec {
            idnt: "id",
            placed: "pl",
            transform: Affine::IDENTITY,
            perspective: None,
            size: (8.0, 8.0),
            dpi: 72.0,
            warp: None,
            filter_fx: Some(filter_fx(&stack, "L", &mut w)),
        };
        let sold = sold_bytes(None, &spec, &mut w);
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..3000 {
            let mut b = sold.clone();
            for _ in 0..1 + next() % 3 {
                let i = (next() as usize) % b.len();
                b[i] = next() as u8;
            }
            if let Some(p) = parse_sold(&b) {
                for f in p.stack.iter().flat_map(|s| &s.filters) {
                    let _ = item_for_filter(f);
                }
                let _ = sold_bytes(Some(&p.descriptor), &spec, &mut w);
            }
        }
    }

    #[test]
    fn warps_are_written_and_read_back() {
        use photocraft_geom::warp::BezierMesh;
        let bounds = [0.0, 0.0, 60.0, 30.0];
        let mut custom = BezierMesh::identity(bounds, 1, 1);
        custom.points[15] = [67.0, 30.0];
        let mut multi = Warp::preset(WarpStyle::Custom, 0.0, bounds);
        multi.mesh = Some(BezierMesh::identity(bounds, 2, 2));
        let mut arc = Warp::preset(WarpStyle::Arc, 40.0, bounds);
        arc.vertical = true;
        arc.h_distort = 10.0;
        let mut c = Warp::preset(WarpStyle::Custom, 0.0, bounds);
        c.mesh = Some(custom);
        for (w, back) in [(Some(arc.clone()), Some(arc)), (Some(c.clone()), Some(c)), (Some(multi), None), (None, None)] {
            let mut warnings = Vec::new();
            let spec = PlacedSpec {
                idnt: "i",
                placed: "p",
                transform: Affine::IDENTITY,
                perspective: None,
                size: (60.0, 30.0),
                dpi: 72.0,
                warp: w.as_ref(),
                filter_fx: None,
            };
            let sold = sold_bytes(None, &spec, &mut warnings);
            assert_eq!(crate::blocks::parse_placed_warp(b"SoLd", &sold), back);
            assert_eq!(warnings.len(), usize::from(w.is_some() && back.is_none()));
        }
    }

    /// Export writes `Trnf` as the image of the custom warp's control-point box, the inverse of
    /// import: a spilled mesh comes back with the same transform and the quad Photoshop expects.
    #[test]
    fn warp_control_box_quad_round_trips() {
        use photocraft_geom::warp::BezierMesh;
        let mut mesh = BezierMesh::identity([0.0, 0.0, 50.0, 25.0], 1, 1);
        mesh.points[0] = [-10.0, -5.0];
        mesh.points[15] = [60.0, 25.0];
        let mut warp = Warp::preset(WarpStyle::Custom, 0.0, [0.0, 0.0, 50.0, 25.0]);
        warp.mesh = Some(mesh);
        let transform = Affine { m: [2.0, 0.0, 0.0, 2.0, 120.0, 210.0] };
        let spec = PlacedSpec { idnt: "i", placed: "p", transform, perspective: None, size: (50.0, 25.0), dpi: 72.0, warp: Some(&warp), filter_fx: None };
        let sold = sold_bytes(None, &spec, &mut Vec::new());
        let p = parse_sold(&sold).expect("parses");
        let Some(Value::List(q)) = p.descriptor.get("Trnf") else { panic!("no Trnf") };
        let q: Vec<f64> = q.iter().filter_map(|v| num(Some(v))).collect();
        assert_eq!(q, [100.0, 200.0, 240.0, 200.0, 240.0, 260.0, 100.0, 260.0]);
        assert_eq!(p.transform, transform);
        // The imported quad reads as unmoved, so the template's `Trnf` is kept as stored.
        assert!(!quad_moved(&p.descriptor, &spec));
    }

    #[test]
    fn hex_and_uuid() {
        assert_eq!(from_hex(&to_hex(&[0, 1, 0xab, 0xff])), Some(vec![0, 1, 0xab, 0xff]));
        assert_eq!(from_hex("abc"), None);
        assert_eq!(from_hex("zz"), None);
        let u = uuid_from(b"x");
        assert_eq!(u.len(), 36);
        assert_eq!(u, uuid_from(b"x"));
        assert_ne!(u, uuid_from(b"y"));
    }
}
