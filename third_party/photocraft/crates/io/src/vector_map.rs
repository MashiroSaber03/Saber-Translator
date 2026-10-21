//! Vector data ⇄ PSD: path records (vector masks `vmsk`/`vsms`, saved paths 2000–2997, the
//! work path 1025), live-shape origination (`vogk`), shape fill content (`vscg`) and shape
//! strokes (`vstk`).
//!
//! Export follows the "keep verbatim while unchanged" rule used for fills and adjustments:
//! a preserved block is written back byte-identical while it still decodes to the layer's
//! current model; otherwise it is regenerated from the model.

use photocraft_doc::{Fill, FillRule, Knot, LineCap, LineJoin, LiveShape, Path, PathOp, ShapeLayer, ShapeStroke, StrokeAlign, Subpath, VectorMask};
use photocraft_geom::Point;
use photocraft_psd::descriptor::{Descriptor, Id, Value, VersionedDescriptor};
use photocraft_psd::path::{PathData, PsdKnot, PsdSubpath, VectorMaskBlock};

use crate::blocks::{bool_of, enum_of, fill_from_desc, fill_to_desc, get_desc, num, parse_prefix_versioned};

/// Saved path resource ids (Adobe spec: 2000–2997).
pub const SAVED_PATHS: std::ops::RangeInclusive<u16> = 2000..=2997;
/// Work path resource id.
pub const WORK_PATH: u16 = 1025;
/// Clipping path name resource id.
pub const CLIPPING_PATH: u16 = 2999;

fn op_from_psd(v: i16) -> PathOp {
    match v {
        0 => PathOp::Exclude,
        2 => PathOp::Subtract,
        3 => PathOp::Intersect,
        // Continues the previous subpath's shape component.
        -1 => PathOp::Join,
        _ => PathOp::Combine,
    }
}

fn op_to_psd(op: PathOp) -> i16 {
    match op {
        PathOp::Combine => 1,
        PathOp::Subtract => 2,
        PathOp::Intersect => 3,
        PathOp::Exclude => 0,
        PathOp::Join => -1,
    }
}

/// Path records → model path in document pixels (`w`×`h` canvas).
pub fn path_from_records(p: &PathData, w: u32, h: u32) -> Path {
    let (w, h) = (f64::from(w), f64::from(h));
    let pt = |q: (f64, f64)| Point::new(q.0 * w, q.1 * h);
    let subpaths = p
        .subpaths()
        .into_iter()
        .map(|s| Subpath {
            closed: s.closed,
            op: op_from_psd(s.operation),
            knots: s.knots.iter().map(|k| Knot { anchor: pt(k.anchor), in_ctrl: pt(k.pre), out_ctrl: pt(k.post), smooth: k.linked }).collect(),
        })
        .collect();
    Path { subpaths, fill_rule: FillRule::NonZero, inverted: p.initial_fill() }
}

/// Model path → path records.
pub fn path_to_records(p: &Path, w: u32, h: u32) -> PathData {
    let (w, h) = (f64::from(w.max(1)), f64::from(h.max(1)));
    let fr = |q: Point| (q.x / w, q.y / h);
    let subs: Vec<PsdSubpath> = p
        .subpaths
        .iter()
        .map(|s| PsdSubpath {
            closed: s.closed,
            operation: op_to_psd(s.op),
            knots: s.knots.iter().map(|k| PsdKnot { linked: k.smooth, pre: fr(k.in_ctrl), anchor: fr(k.anchor), post: fr(k.out_ctrl) }).collect(),
        })
        .collect();
    PathData::from_subpaths(&subs, p.inverted)
}

/// Path from a saved-path / work-path resource.
pub fn path_from_resource(data: &[u8], w: u32, h: u32) -> Option<Path> {
    PathData::from_bytes(data).ok().map(|p| path_from_records(&p, w, h))
}

/// Layer vector data (`vmsk`/`vsms`) → model path, with Photoshop's reading of the records
/// (matching its rendering of the psd-tools corpus):
/// - without subpaths, the initial-fill record alone decides coverage (1 = everything);
/// - with subpaths the initial-fill record is ignored; a first component that subtracts starts
///   from a full canvas (the shape is inverted);
/// - a subpath with operation -1 continues the previous component ([`PathOp::Join`]: filled
///   together with it, so subpaths wound the other way cut holes, as in psd-tools
///   stroke-effects' custom shapes).
///
/// The model applies the first subpath as "combine" and inverts the final result, so a leading
/// subtract becomes `inverted` with every later operation replaced by its complement dual
/// (combine ↔ subtract; exclude is self-dual; intersect has no dual and is kept as is).
fn layer_path_from_records(p: &PathData, w: u32, h: u32) -> Path {
    let mut path = path_from_records(p, w, h);
    if path.subpaths.is_empty() {
        return path;
    }
    // `path_from_records` maps -1 to `PathOp::Join` (the component's operation is its first
    // subpath's); a leading -1 acts as the first component.
    path.inverted = path.subpaths.first().is_some_and(|s| s.op == PathOp::Subtract);
    if path.inverted {
        for s in path.subpaths.iter_mut().skip(1) {
            s.op = match s.op {
                PathOp::Combine => PathOp::Subtract,
                PathOp::Subtract => PathOp::Combine,
                op => op,
            };
        }
        if let Some(first) = path.subpaths.first_mut() {
            first.op = PathOp::Combine;
        }
    }
    path
}

/// A `vmsk`/`vsms` block → (path with the invert flag applied, flags). Coverage semantics: an
/// empty path covers everything when `inverted` (shape layers; vector masks flip this, see
/// [`vector_mask_from_block`]).
pub fn path_from_vmsk(data: &[u8], w: u32, h: u32) -> Option<(Path, u32)> {
    let b = VectorMaskBlock::from_bytes(data).ok()?;
    let mut p = layer_path_from_records(&b.path, w, h);
    if b.flags & VectorMaskBlock::FLAG_INVERT != 0 {
        p.inverted = !p.inverted;
    }
    Some((p, b.flags))
}

/// A `vmsk` block for `path` with extra flags (not linked / disabled). An empty path is written
/// as the initial-fill record (Photoshop's form); otherwise inversion uses the invert flag.
pub fn vmsk_bytes(path: &Path, flags: u32, w: u32, h: u32) -> Vec<u8> {
    let mut p = path.clone();
    let mut flags = flags & !VectorMaskBlock::FLAG_INVERT;
    if p.inverted && !p.subpaths.is_empty() {
        flags |= VectorMaskBlock::FLAG_INVERT;
        p.inverted = false;
    }
    // The model draws the first subpath as "combine"; a stored leading subtract would read back
    // as an inverted shape.
    if let Some(first) = p.subpaths.first_mut()
        && first.op == PathOp::Subtract
    {
        first.op = PathOp::Combine;
    }
    VectorMaskBlock { version: 3, flags, path: path_to_records(&p, w, h) }.to_bytes()
}

/// A layer's vector mask from its `vmsk`/`vsms` block. A vector mask without subpaths reveals
/// everything unless inverted (see `photocraft_vector::vector_mask_values`), the opposite of a
/// shape's coverage, hence the flip.
pub fn vector_mask_from_block(data: &[u8], w: u32, h: u32) -> Option<VectorMask> {
    let (mut path, flags) = path_from_vmsk(data, w, h)?;
    if path.subpaths.is_empty() {
        path.inverted = !path.inverted;
    }
    Some(VectorMask {
        path,
        enabled: flags & VectorMaskBlock::FLAG_DISABLED == 0,
        linked: flags & VectorMaskBlock::FLAG_NOT_LINKED == 0,
        density: 1.0,
        feather: 0.0,
    })
}

/// `vmsk` data for a vector mask.
pub fn vector_mask_bytes(m: &VectorMask, w: u32, h: u32) -> Vec<u8> {
    let mut flags = 0;
    if !m.enabled {
        flags |= VectorMaskBlock::FLAG_DISABLED;
    }
    if !m.linked {
        flags |= VectorMaskBlock::FLAG_NOT_LINKED;
    }
    let mut path = m.path.clone();
    if path.subpaths.is_empty() {
        path.inverted = !path.inverted;
    }
    vmsk_bytes(&path, flags, w, h)
}

/// Compares the parts of a vector mask stored in the block (density/feather live elsewhere).
pub fn vector_mask_block_matches(data: &[u8], m: &VectorMask, w: u32, h: u32) -> bool {
    vector_mask_from_block(data, w, h).is_some_and(|b| b.path == m.path && b.enabled == m.enabled && b.linked == m.linked)
}

// ---------------------------------------------------------------------------
// vscg (shape fill content)
// ---------------------------------------------------------------------------

/// `vscg`: content key (`SoCo`/`GdFl`/`PtFl`) + version-16 descriptor.
pub fn fill_from_vscg(data: &[u8]) -> Option<Fill> {
    let key: [u8; 4] = data.get(..4)?.try_into().ok()?;
    let d = parse_prefix_versioned(data.get(4..)?)?;
    fill_from_desc(&key, &d)
}

pub fn vscg_bytes(f: &Fill) -> Vec<u8> {
    let (k, d) = fill_to_desc(f);
    let mut out = k.to_vec();
    out.extend(VersionedDescriptor::new(d).to_bytes());
    out
}

// ---------------------------------------------------------------------------
// vstk (shape stroke)
// ---------------------------------------------------------------------------

/// Decoded `vstk`: the stroke (None when disabled) and whether the fill is enabled.
#[derive(Clone, Debug, PartialEq)]
pub struct VstkInfo {
    pub stroke: Option<ShapeStroke>,
    pub fill_enabled: bool,
}

fn content_fill(d: &Descriptor) -> Option<Fill> {
    let key = match d.class_id.as_bytes() {
        b"solidColorLayer" => b"SoCo",
        b"gradientLayer" => b"GdFl",
        b"patternLayer" => b"PtFl",
        _ if d.get("Clr ").is_some() => b"SoCo",
        _ if d.get("Grad").is_some() => b"GdFl",
        _ => b"PtFl",
    };
    fill_from_desc(key, d)
}

fn unit_px(v: Option<&Value>, dpi: f32) -> Option<f64> {
    match v? {
        Value::UnitFloat { unit: [b'#', b'P', b'n', b't'], value } => Some(value * f64::from(dpi) / 72.0),
        Value::UnitFloat { unit: [b'#', b'M', b'l', b'm'], value } => Some(value * f64::from(dpi) / 25.4),
        other => num(Some(other)),
    }
}

pub fn parse_vstk(data: &[u8], dpi: f32) -> Option<VstkInfo> {
    let d = parse_prefix_versioned(data)?;
    let fill_enabled = !matches!(d.get("fillEnabled"), Some(Value::Boolean(false)));
    if !bool_of(&d, "strokeEnabled") {
        return Some(VstkInfo { stroke: None, fill_enabled });
    }
    let e = |k: &str| enum_of(&d, k).map(<[u8]>::to_vec).unwrap_or_default();
    let cap = match &e("strokeStyleLineCapType")[..] {
        b"strokeStyleRoundCap" => LineCap::Round,
        b"strokeStyleSquareCap" => LineCap::Square,
        _ => LineCap::Butt,
    };
    let join = match &e("strokeStyleLineJoinType")[..] {
        b"strokeStyleRoundJoin" => LineJoin::Round,
        b"strokeStyleBevelJoin" => LineJoin::Bevel,
        _ => LineJoin::Miter,
    };
    let align = match &e("strokeStyleLineAlignment")[..] {
        b"strokeStyleAlignInside" => StrokeAlign::Inside,
        b"strokeStyleAlignOutside" => StrokeAlign::Outside,
        _ => StrokeAlign::Center,
    };
    let dashes = match d.get("strokeStyleLineDashSet") {
        Some(Value::List(l)) => l.iter().filter_map(|v| num(Some(v))).map(|v| v as f32).collect(),
        _ => Vec::new(),
    };
    let paint = get_desc(&d, "strokeStyleContent").and_then(content_fill).unwrap_or(Fill::Solid(photocraft_color::Color::BLACK));
    let stroke = ShapeStroke {
        width: unit_px(d.get("strokeStyleLineWidth"), dpi).unwrap_or(1.0) as f32,
        paint,
        opacity: num(d.get("strokeStyleOpacity")).map_or(1.0, |v| v as f32 / 100.0),
        align,
        cap,
        join,
        miter_limit: num(d.get("strokeStyleMiterLimit")).unwrap_or(100.0) as f32,
        dashes,
        dash_offset: num(d.get("strokeStyleLineDashOffset")).unwrap_or(0.0) as f32,
    };
    Some(VstkInfo { stroke: Some(stroke), fill_enabled })
}

fn en(ty: &str, v: &str) -> Value {
    Value::Enumerated { type_id: Id::new(ty), value: Id::new(v) }
}

pub fn vstk_bytes(stroke: Option<&ShapeStroke>, fill_enabled: bool, dpi: f32) -> Vec<u8> {
    let default = ShapeStroke::default();
    let s = stroke.unwrap_or(&default);
    let (_, content) = fill_to_desc(&s.paint);
    let class = match s.paint {
        Fill::Solid(_) => "solidColorLayer",
        Fill::Gradient { .. } => "gradientLayer",
        Fill::Pattern { .. } => "patternLayer",
    };
    let content = Descriptor { class_id: Id::new(class), ..content };
    let d = Descriptor::new("strokeStyle")
        .with("strokeStyleVersion", Value::Integer(2))
        .with("strokeEnabled", Value::Boolean(stroke.is_some()))
        .with("fillEnabled", Value::Boolean(fill_enabled))
        .with("strokeStyleLineWidth", Value::UnitFloat { unit: *b"#Pxl", value: f64::from(s.width) })
        .with("strokeStyleLineDashOffset", Value::UnitFloat { unit: *b"#Pnt", value: f64::from(s.dash_offset) })
        .with("strokeStyleMiterLimit", Value::Double(f64::from(s.miter_limit)))
        .with(
            "strokeStyleLineCapType",
            en(
                "strokeStyleLineCapType",
                match s.cap {
                    LineCap::Butt => "strokeStyleButtCap",
                    LineCap::Round => "strokeStyleRoundCap",
                    LineCap::Square => "strokeStyleSquareCap",
                },
            ),
        )
        .with(
            "strokeStyleLineJoinType",
            en(
                "strokeStyleLineJoinType",
                match s.join {
                    LineJoin::Miter => "strokeStyleMiterJoin",
                    LineJoin::Round => "strokeStyleRoundJoin",
                    LineJoin::Bevel => "strokeStyleBevelJoin",
                },
            ),
        )
        .with(
            "strokeStyleLineAlignment",
            en(
                "strokeStyleLineAlignment",
                match s.align {
                    StrokeAlign::Inside => "strokeStyleAlignInside",
                    StrokeAlign::Center => "strokeStyleAlignCenter",
                    StrokeAlign::Outside => "strokeStyleAlignOutside",
                },
            ),
        )
        .with("strokeStyleScaleLock", Value::Boolean(false))
        .with("strokeStyleStrokeAdjust", Value::Boolean(false))
        .with("strokeStyleLineDashSet", Value::List(s.dashes.iter().map(|v| Value::UnitFloat { unit: *b"#Nne", value: f64::from(*v) }).collect()))
        .with("strokeStyleBlendMode", en("BlnM", "Nrml"))
        .with("strokeStyleOpacity", Value::UnitFloat { unit: *b"#Prc", value: f64::from(s.opacity * 100.0) })
        .with("strokeStyleContent", Value::Descriptor(content))
        .with("strokeStyleResolution", Value::Double(f64::from(dpi)));
    VersionedDescriptor::new(d).to_bytes()
}

// ---------------------------------------------------------------------------
// vogk (live shape origination)
// ---------------------------------------------------------------------------

fn rect_of(d: &Descriptor) -> Option<[f64; 4]> {
    let b = get_desc(d, "keyOriginShapeBBox")?;
    let (t, l, bt, r) = (num(b.get("Top "))?, num(b.get("Left"))?, num(b.get("Btom"))?, num(b.get("Rght"))?);
    Some([l, t, r - l, bt - t])
}

fn point_of(d: &Descriptor, key: &str) -> Option<[f64; 2]> {
    let p = get_desc(d, key)?;
    Some([num(p.get("Hrzn"))?, num(p.get("Vrtc"))?])
}

/// Live shape from `vogk` (version 1 + versioned descriptor). Only a single, valid,
/// untransformed origination maps to a [`LiveShape`].
pub fn live_from_vogk(data: &[u8]) -> Option<LiveShape> {
    let d = parse_prefix_versioned(data.get(4..)?)?;
    let Some(Value::List(list)) = d.get("keyDescriptorList") else { return None };
    let [Value::Descriptor(o)] = &list[..] else { return None };
    if bool_of(o, "keyShapeInvalidated") {
        return None;
    }
    if let Some(t) = get_desc(o, "Trnf") {
        let g = |k: &str, dv: f64| num(t.get(k)).unwrap_or(dv);
        let ident = [("xx", 1.0), ("xy", 0.0), ("yx", 0.0), ("yy", 1.0), ("tx", 0.0), ("ty", 0.0)].iter().all(|(k, v)| (g(k, *v) - v).abs() < 1e-9);
        if !ident {
            return None;
        }
    }
    let ty = match o.get("keyOriginType") {
        Some(Value::Integer(i)) => *i,
        _ => return None,
    };
    match ty {
        1 => Some(LiveShape::Rect { rect: rect_of(o)?, radii: [0.0; 4] }),
        2 => {
            let r = get_desc(o, "keyOriginRRectRadii")?;
            let g = |k: &str| num(r.get(k)).unwrap_or(0.0);
            Some(LiveShape::Rect { rect: rect_of(o)?, radii: [g("topLeft"), g("topRight"), g("bottomRight"), g("bottomLeft")] })
        }
        5 => Some(LiveShape::Ellipse { rect: rect_of(o)? }),
        4 => Some(LiveShape::Line {
            from: point_of(o, "keyOriginLineStart")?,
            to: point_of(o, "keyOriginLineEnd")?,
            weight: num(o.get("keyOriginLineWeight")).unwrap_or(1.0),
        }),
        _ => None,
    }
}

fn px(v: f64) -> Value {
    Value::UnitFloat { unit: *b"#Pxl", value: v }
}

/// `vogk` for a live shape (None for kinds Photoshop has no origination type for here).
pub fn vogk_bytes(s: &LiveShape, dpi: f32) -> Option<Vec<u8>> {
    let bbox = |r: [f64; 4]| {
        Value::Descriptor(
            Descriptor::new("unitRect")
                .with("unitValueQuadVersion", Value::Integer(1))
                .with("Top ", px(r[1]))
                .with("Left", px(r[0]))
                .with("Btom", px(r[1] + r[3]))
                .with("Rght", px(r[0] + r[2])),
        )
    };
    let base = |ty: i32| Descriptor::new("null").with("keyOriginType", Value::Integer(ty)).with("keyOriginResolution", Value::Double(f64::from(dpi)));
    let o = match s {
        LiveShape::Rect { rect, radii } if radii.iter().all(|r| *r == 0.0) => base(1).with("keyOriginShapeBBox", bbox(*rect)),
        LiveShape::Rect { rect, radii } => base(2)
            .with(
                "keyOriginRRectRadii",
                Value::Descriptor(
                    Descriptor::new("radii")
                        .with("unitValueQuadVersion", Value::Integer(1))
                        .with("topRight", px(radii[1]))
                        .with("topLeft", px(radii[0]))
                        .with("bottomLeft", px(radii[3]))
                        .with("bottomRight", px(radii[2])),
                ),
            )
            .with("keyOriginShapeBBox", bbox(*rect)),
        LiveShape::Ellipse { rect } => base(5).with("keyOriginShapeBBox", bbox(*rect)),
        LiveShape::Line { from, to, weight } => {
            let pnt = |p: [f64; 2]| Value::Descriptor(Descriptor::new("Pnt ").with("Hrzn", Value::Double(p[0])).with("Vrtc", Value::Double(p[1])));
            let r = [from[0].min(to[0]), from[1].min(to[1]), (from[0] - to[0]).abs(), (from[1] - to[1]).abs()];
            base(4)
                .with("keyOriginShapeBBox", bbox(r))
                .with("keyOriginLineEnd", pnt(*to))
                .with("keyOriginLineStart", pnt(*from))
                .with("keyOriginLineWeight", Value::Double(*weight))
                .with("keyOriginLineArrowSt", Value::Boolean(false))
                .with("keyOriginLineArrowEnd", Value::Boolean(false))
                .with("keyOriginLineArrWdth", Value::Double(0.0))
                .with("keyOriginLineArrLngth", Value::Double(0.0))
                .with("keyOriginLineArrConc", Value::Integer(0))
        }
        LiveShape::Polygon { .. } => return None,
    };
    let o = o.with("keyOriginIndex", Value::Integer(0));
    let d = Descriptor::new("null").with("keyDescriptorList", Value::List(vec![Value::Descriptor(o)]));
    let mut out = 1u32.to_be_bytes().to_vec();
    out.extend(VersionedDescriptor::new(d).to_bytes());
    Some(out)
}

// ---------------------------------------------------------------------------
// Shape layers
// ---------------------------------------------------------------------------

/// Fills the typed parts of a shape layer from its PSD blocks.
pub fn shape_from_blocks(sh: &mut ShapeLayer, block: &dyn Fn(&[u8; 4]) -> Option<Vec<u8>>, w: u32, h: u32, dpi: f32) {
    if let Some(d) = block(b"vsms").or_else(|| block(b"vmsk"))
        && let Some((p, _)) = path_from_vmsk(&d, w, h)
    {
        sh.path = p;
    }
    if let Some(f) = block(b"vscg").and_then(|d| fill_from_vscg(&d)) {
        sh.fill = Some(f);
    }
    if let Some(v) = block(b"vstk").and_then(|d| parse_vstk(&d, dpi)) {
        sh.stroke = v.stroke;
        if !v.fill_enabled {
            sh.fill = None;
        }
    }
    sh.live = block(b"vogk").and_then(|d| live_from_vogk(&d));
}

fn has_fill_block(raw: &[([u8; 4], Vec<u8>)]) -> bool {
    raw.iter().any(|(k, d)| match k {
        b"vscg" => fill_from_vscg(d).is_some(),
        b"SoCo" | b"GdFl" | b"PtFl" => crate::blocks::parse_fill(k, d).is_some(),
        _ => false,
    })
}

/// Rewrites the vector blocks of a shape layer in `raw` (key, data) to match the model:
/// blocks that still decode to the model are kept byte-identical.
pub fn shape_blocks(sh: &ShapeLayer, raw: &mut Vec<([u8; 4], Vec<u8>)>, w: u32, h: u32, dpi: f32) {
    let find = |raw: &Vec<([u8; 4], Vec<u8>)>, k: &[u8; 4]| raw.iter().position(|(key, _)| key == k);
    // Path.
    let pos = find(raw, b"vsms").or_else(|| find(raw, b"vmsk"));
    let from_raw = |d: &[u8]| path_from_vmsk(d, w, h).map(|(p, _)| p);
    let keep = |d: &[u8]| from_raw(d).as_ref() == Some(&sh.path);
    match pos {
        Some(i) if keep(&raw[i].1) => {}
        _ => {
            // The layer's own raw copy may still be valid (psd_raw is the principal block).
            let data = match &sh.psd_raw {
                Some(r) if keep(r) => r.to_vec(),
                _ => vmsk_bytes(&sh.path, 0, w, h),
            };
            match pos {
                Some(i) => raw[i].1 = data,
                None => raw.push((*b"vmsk", data)),
            }
        }
    }
    // Fill content.
    if let Some(f) = &sh.fill {
        match find(raw, b"vscg") {
            Some(i) if fill_from_vscg(&raw[i].1).as_ref() == Some(f) => {}
            Some(i) => raw[i].1 = vscg_bytes(f),
            None => {}
        }
    }
    // Photoshop shapes always carry fill content, even with the fill turned off.
    let present = raw.iter().any(|(k, _)| matches!(k, b"vscg" | b"SoCo" | b"GdFl" | b"PtFl"));
    if sh.fill.is_none() && !present {
        raw.push((*b"vscg", vscg_bytes(&Fill::Solid(photocraft_color::Color::BLACK))));
    }
    // Stroke (+ fill enabled flag). A fill we could not decode counts as unchanged.
    let undecodable_fill = sh.fill.is_none() && present && !has_fill_block(raw);
    let same = |v: &VstkInfo| v.stroke == sh.stroke && (v.fill_enabled == sh.fill.is_some() || (v.fill_enabled && undecodable_fill));
    match find(raw, b"vstk") {
        Some(i) if parse_vstk(&raw[i].1, dpi).as_ref().is_some_and(same) => {}
        Some(i) => raw[i].1 = vstk_bytes(sh.stroke.as_ref(), sh.fill.is_some(), dpi),
        // "No fill" needs a vstk only when a fill block would otherwise paint.
        None if sh.stroke.is_some() || (sh.fill.is_none() && has_fill_block(raw)) => {
            raw.push((*b"vstk", vstk_bytes(sh.stroke.as_ref(), sh.fill.is_some(), dpi)));
        }
        None => {}
    }
    // Live shape origination.
    let pos = find(raw, b"vogk");
    let current = pos.and_then(|i| live_from_vogk(&raw[i].1));
    if current != sh.live || (pos.is_none() && sh.live.is_some()) {
        let generated = sh.live.as_ref().and_then(|l| vogk_bytes(l, dpi));
        match (pos, generated) {
            (Some(i), Some(g)) => raw[i].1 = g,
            (Some(i), None) => {
                raw.remove(i);
            }
            (None, Some(g)) => raw.push((*b"vogk", g)),
            (None, None) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::Color;

    fn sample_path() -> Path {
        let mut p = Path::new(vec![
            Subpath {
                closed: true,
                op: PathOp::Combine,
                knots: vec![
                    Knot::corner(10.0, 10.0),
                    Knot::smooth(Point::new(50.0, 20.0), Point::new(40.0, 10.0), Point::new(60.0, 30.0)),
                    Knot::corner(20.0, 40.0),
                ],
            },
            Subpath::polyline(&[(1.0, 2.0), (3.0, 4.0)]).with_op(PathOp::Subtract),
        ]);
        p.inverted = false;
        p
    }

    #[test]
    fn path_records_roundtrip() {
        let p = sample_path();
        let rec = path_to_records(&p, 64, 64);
        let back = path_from_records(&rec, 64, 64);
        assert_eq!(back, p);
        for op in [PathOp::Combine, PathOp::Subtract, PathOp::Intersect, PathOp::Exclude, PathOp::Join] {
            assert_eq!(op_from_psd(op_to_psd(op)), op);
        }
        assert_eq!(op_from_psd(-1), PathOp::Join);
        assert_eq!(op_from_psd(7), PathOp::Combine);
    }

    fn vmsk_with(initial_fill: bool, ops: &[i16]) -> Vec<u8> {
        let square = |x: f64| PsdSubpath {
            closed: true,
            operation: 0,
            knots: [(x, 0.25), (x + 0.25, 0.25), (x + 0.25, 0.5), (x, 0.5)].iter().map(|&p| PsdKnot { linked: false, pre: p, anchor: p, post: p }).collect(),
        };
        let subs: Vec<PsdSubpath> = ops.iter().enumerate().map(|(i, &op)| PsdSubpath { operation: op, ..square(0.1 * i as f64) }).collect();
        VectorMaskBlock { version: 3, flags: 0, path: PathData::from_subpaths(&subs, initial_fill) }.to_bytes()
    }

    /// psd-tools vector-mask.psd / vector-mask2.psd / vector-mask3.psd: Photoshop's initial-fill
    /// record only matters without subpaths; a leading subtract inverts; -1 continues the
    /// previous component.
    #[test]
    fn layer_path_semantics_follow_photoshop() {
        // Initial fill with a combine subpath: a plain (not inverted) shape.
        let (p, _) = path_from_vmsk(&vmsk_with(true, &[1]), 100, 100).unwrap();
        assert!(!p.inverted);
        // Leading subtract: everything but the shape; later ops become their complements.
        let (p, _) = path_from_vmsk(&vmsk_with(true, &[2, 1, 2, 0]), 100, 100).unwrap();
        assert!(p.inverted);
        assert_eq!(p.subpaths.iter().map(|s| s.op).collect::<Vec<_>>(), [PathOp::Combine, PathOp::Subtract, PathOp::Combine, PathOp::Exclude]);
        // -1 continues the previous component (filled together with it, then subtracted).
        let (p, _) = path_from_vmsk(&vmsk_with(false, &[1, 2, -1]), 100, 100).unwrap();
        assert_eq!(p.subpaths[2].op, PathOp::Join);
        assert_eq!(p.components(), vec![0..1, 1..3]);
        // Under a leading subtract the joined subpaths stay with their (first) component.
        let (p, _) = path_from_vmsk(&vmsk_with(false, &[2, -1, 1]), 100, 100).unwrap();
        assert!(p.inverted);
        assert_eq!(p.subpaths.iter().map(|s| s.op).collect::<Vec<_>>(), [PathOp::Combine, PathOp::Join, PathOp::Subtract]);
        // No subpaths: a vector mask with initial fill reveals all, without hides all; a shape
        // with initial fill covers everything.
        let reveal = vector_mask_from_block(&vmsk_with(true, &[]), 100, 100).unwrap();
        assert_eq!(photocraft_vector::vector_mask_values(&reveal, photocraft_geom::Rect::new(0, 0, 2, 2)), vec![1.0; 4]);
        let hide = vector_mask_from_block(&vmsk_with(false, &[]), 100, 100).unwrap();
        assert_eq!(photocraft_vector::vector_mask_values(&hide, photocraft_geom::Rect::new(0, 0, 2, 2)), vec![0.0; 4]);
        assert!(path_from_vmsk(&vmsk_with(true, &[]), 100, 100).unwrap().0.inverted);
        // Our own writer round-trips each case.
        for m in [reveal, hide] {
            assert_eq!(vector_mask_from_block(&vector_mask_bytes(&m, 100, 100), 100, 100), Some(m));
        }
        let mut lead = sample_path();
        lead.subpaths[0].op = PathOp::Subtract;
        let back = path_from_vmsk(&vmsk_bytes(&lead, 0, 64, 64), 64, 64).unwrap().0;
        assert!(!back.inverted, "a stored leading subtract is drawn as combine by the model");
    }

    #[test]
    fn vector_mask_roundtrip_with_flags() {
        let mut m = VectorMask::new(sample_path());
        m.enabled = false;
        m.linked = false;
        m.path.inverted = true;
        let b = vector_mask_bytes(&m, 64, 64);
        let back = vector_mask_from_block(&b, 64, 64).unwrap();
        assert_eq!(back, m);
        assert!(vector_mask_block_matches(&b, &m, 64, 64));
    }

    #[test]
    fn vstk_roundtrip() {
        let s = ShapeStroke {
            width: 4.5,
            paint: Fill::Solid(Color::rgb(1.0, 0.0, 0.0)),
            opacity: 0.5,
            align: StrokeAlign::Outside,
            cap: LineCap::Round,
            join: LineJoin::Bevel,
            miter_limit: 10.0,
            dashes: vec![2.0, 1.0],
            dash_offset: 0.5,
        };
        let b = vstk_bytes(Some(&s), false, 72.0);
        let v = parse_vstk(&b, 72.0).unwrap();
        assert!(!v.fill_enabled);
        let got = v.stroke.unwrap();
        assert_eq!(got.width, s.width);
        assert_eq!((got.cap, got.join, got.align), (s.cap, s.join, s.align));
        assert_eq!(got.dashes, s.dashes);
        match got.paint {
            Fill::Solid(c) => assert!((c.to_rgb()[0] - 1.0).abs() < 1e-3),
            other => panic!("{other:?}"),
        }
        let off = parse_vstk(&vstk_bytes(None, true, 72.0), 72.0).unwrap();
        assert!(off.stroke.is_none() && off.fill_enabled);
    }

    #[test]
    fn vogk_roundtrip() {
        for s in [
            LiveShape::Rect { rect: [1.0, 2.0, 30.0, 40.0], radii: [0.0; 4] },
            LiveShape::Rect { rect: [1.0, 2.0, 30.0, 40.0], radii: [1.0, 2.0, 3.0, 4.0] },
            LiveShape::Ellipse { rect: [5.0, 6.0, 7.0, 8.0] },
            LiveShape::Line { from: [0.0, 10.0], to: [20.0, 0.0], weight: 3.0 },
        ] {
            let b = vogk_bytes(&s, 72.0).unwrap();
            assert_eq!(live_from_vogk(&b), Some(s));
        }
        assert!(vogk_bytes(&LiveShape::Polygon { rect: [0.0; 4], sides: 5, star_ratio: 1.0 }, 72.0).is_none());
    }

    #[test]
    fn shape_blocks_keep_unchanged_and_regenerate_changed() {
        let sh = ShapeLayer {
            path: sample_path(),
            fill: Some(Fill::Solid(Color::rgb(0.0, 1.0, 0.0))),
            stroke: Some(ShapeStroke::default()),
            live: Some(LiveShape::Ellipse { rect: [0.0, 0.0, 10.0, 10.0] }),
            ..Default::default()
        };
        let mut raw = Vec::new();
        shape_blocks(&sh, &mut raw, 64, 64, 72.0);
        let keys: Vec<&[u8; 4]> = raw.iter().map(|(k, _)| k).collect();
        assert_eq!(keys, [b"vmsk", b"vstk", b"vogk"]);
        let before = raw.clone();
        shape_blocks(&sh, &mut raw, 64, 64, 72.0);
        assert_eq!(raw, before);
        let mut back = ShapeLayer::default();
        let lookup = |k: &[u8; 4]| raw.iter().find(|(key, _)| key == k).map(|(_, d)| d.clone());
        shape_from_blocks(&mut back, &lookup, 64, 64, 72.0);
        assert_eq!(back.path, sh.path);
        assert_eq!(back.stroke, sh.stroke);
        assert_eq!(back.live, sh.live);
        // Edit: path changes → vmsk regenerated, live dropped when polygon.
        let mut edited = sh.clone();
        edited.path.subpaths.pop();
        edited.live = Some(LiveShape::Polygon { rect: [0.0; 4], sides: 3, star_ratio: 1.0 });
        shape_blocks(&edited, &mut raw, 64, 64, 72.0);
        assert!(!raw.iter().any(|(k, _)| k == b"vogk"));
        let (p, _) = path_from_vmsk(&raw[0].1, 64, 64).unwrap();
        assert_eq!(p, edited.path);
    }
}
