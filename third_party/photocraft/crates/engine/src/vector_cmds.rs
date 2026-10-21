//! Vector commands: shape layers (`shape.*`), document paths (`path.*`, Paths panel), selection ⇄
//! path, and layer vector masks (`layer.vectorMask.*`).
//!
//! Coordinates are document pixels. Paths in parameters and results use a compact JSON form:
//!
//! ```json
//! {"subpaths":[{"closed":true,"op":"combine|subtract|intersect|exclude",
//!               "knots":[[x,y], {"anchor":[x,y],"in":[x,y],"out":[x,y],"smooth":true}, …]}],
//!  "fillRule":"nonzero|evenodd", "inverted":false}
//! ```
//!
//! A knot given as `[x, y]` is a corner with retracted handles; `in`/`out` default to the
//! anchor. The serde form of `photocraft_doc::Path` is accepted too. Colours are `"#rrggbb"`,
//! `"#rrggbbaa"` or `[r,g,b,a]` (0..1); opacities are percentages (0..100).
//!
//! Every edit re-renders the shape layer's cache with `photocraft-vector` in one undoable step.
//! PSD blocks stay byte-identical while they still decode to the model (see `photocraft-io`).

use photocraft_algo::selection::{self as sel, SelectionMode};
use photocraft_color::{BlendMode, Color};
use photocraft_doc::{
    ClippingPath, Document, Fill, FillRule, GradientStyle, Knot, Layer, LayerContent, LayerId, LineCap, LineJoin, LiveShape, NamedPath, Path, PathOp,
    ShapeLayer, ShapeStroke, StrokeAlign, Subpath, VectorMask,
};
use photocraft_geom::{Affine, Point, Rect};
use photocraft_vector as vector;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, blend_from_str};
use crate::{EngineError, Result, Session};

/// Keys PSD uses for a shape layer's vector data; dropped when a shape is rasterized.
const SHAPE_BLOCKS: [&[u8; 4]; 8] = [b"vmsk", b"vsms", b"vogk", b"vstk", b"vscg", b"SoCo", b"GdFl", b"PtFl"];

pub(crate) fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

pub(crate) fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

fn has_layer(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    d.active_layer.filter(|id| d.doc.layer(*id).is_some()).map(|_| ()).ok_or_else(|| "no active layer".into())
}

fn has_selection(s: &Session) -> std::result::Result<(), String> {
    s.active().filter(|d| d.doc.selection.is_some()).map(|_| ()).ok_or_else(|| "no selection".into())
}

fn active_shape(s: &Session) -> std::result::Result<&ShapeLayer, String> {
    let d = s.active().ok_or("no document open")?;
    let id = d.active_layer.ok_or("no active layer")?;
    let layer = d.doc.layer(id).ok_or("no active layer")?;
    match &layer.content {
        LayerContent::Shape(shape) => Ok(shape),
        _ => Err("active layer is not a shape layer".into()),
    }
}

fn has_shape_fill(s: &Session) -> std::result::Result<(), String> {
    active_shape(s)?.fill.as_ref().map(|_| ()).ok_or_else(|| "shape has no fill".into())
}

fn has_shape_stroke(s: &Session) -> std::result::Result<(), String> {
    active_shape(s)?.stroke.as_ref().map(|_| ()).ok_or_else(|| "shape has no stroke".into())
}

fn can_paste_shape_fill(s: &Session) -> std::result::Result<(), String> {
    active_shape(s)?;
    s.path_fill_clipboard.as_ref().map(|_| ()).ok_or_else(|| "no shape fill copied".into())
}

fn can_paste_shape_stroke(s: &Session) -> std::result::Result<(), String> {
    active_shape(s)?;
    s.path_stroke_clipboard.as_ref().map(|_| ()).ok_or_else(|| "no shape stroke copied".into())
}

fn copy_shape_fill(s: &mut Session, _p: &Value) -> Result<Value> {
    let fill = active_shape(s).map_err(EngineError::Other)?.fill.clone().ok_or_else(|| EngineError::Other("shape has no fill".into()))?;
    s.path_fill_clipboard = Some(fill);
    Ok(Value::Null)
}

fn copy_shape_stroke(s: &mut Session, _p: &Value) -> Result<Value> {
    let stroke = active_shape(s).map_err(EngineError::Other)?.stroke.clone().ok_or_else(|| EngineError::Other("shape has no stroke".into()))?;
    s.path_stroke_clipboard = Some(stroke);
    Ok(Value::Null)
}

fn paste_shape_fill(s: &mut Session, p: &Value) -> Result<Value> {
    let fill = s.path_fill_clipboard.clone().ok_or_else(|| EngineError::Other("no shape fill copied".into()))?;
    let id = layer_id(s, p)?;
    with_shape(s, id, "Paste Fill", |shape, _| {
        shape.fill = Some(fill);
        Ok(())
    })?;
    shape_info(s, id)
}

fn paste_shape_stroke(s: &mut Session, p: &Value) -> Result<Value> {
    let stroke = s.path_stroke_clipboard.clone().ok_or_else(|| EngineError::Other("no shape stroke copied".into()))?;
    let id = layer_id(s, p)?;
    with_shape(s, id, "Paste Complete Stroke", |shape, _| {
        shape.stroke = Some(stroke);
        Ok(())
    })?;
    shape_info(s, id)
}

pub(crate) fn layer_id(s: &Session, p: &Value) -> Result<LayerId> {
    match p.get("layer").and_then(Value::as_u64) {
        Some(id) => Ok(LayerId(id)),
        None => s.active().and_then(|d| d.active_layer).ok_or(EngineError::Other("no active layer".into())),
    }
}

fn f64p(p: &Value, k: &str) -> Option<f64> {
    p.get(k).and_then(Value::as_f64)
}

pub(crate) fn pt(v: &Value) -> Option<Point> {
    match v {
        Value::Array(a) if a.len() >= 2 => Some(Point::new(a[0].as_f64()?, a[1].as_f64()?)),
        Value::Object(_) => Some(Point::new(v.get("x")?.as_f64()?, v.get("y")?.as_f64()?)),
        _ => None,
    }
}

pub(crate) fn nums<const N: usize>(p: &Value, k: &str) -> Option<[f64; N]> {
    let a = p.get(k)?.as_array()?;
    if a.len() < N {
        return None;
    }
    let mut out = [0.0; N];
    for (o, v) in out.iter_mut().zip(a) {
        *o = v.as_f64()?;
    }
    Some(out)
}

/// `[tl, tr, br, bl]` from an array or a single number.
fn radii(p: &Value) -> Option<[f64; 4]> {
    match p.get("radii").or_else(|| p.get("radius"))? {
        Value::Number(n) => n.as_f64().map(|r| [r; 4]),
        _ => nums::<4>(p, "radii"),
    }
}

pub(crate) fn color(v: &Value) -> Option<Color> {
    match v {
        Value::Array(a) if a.len() >= 3 => {
            let c: Vec<f32> = a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
            Some(Color::rgba(c[0], c[1], c[2], c.get(3).copied().unwrap_or(1.0)))
        }
        Value::String(s) => {
            let s = s.trim_start_matches('#');
            let h = |i: usize| s.get(i..i + 2).and_then(|x| u8::from_str_radix(x, 16).ok()).map(|v| f32::from(v) / 255.0);
            match s.len() {
                6 => Some(Color::rgb(h(0)?, h(2)?, h(4)?)),
                8 => Some(Color::rgba(h(0)?, h(2)?, h(4)?, h(6)?)),
                _ => None,
            }
        }
        _ => None,
    }
}

fn hex(c: &Color) -> String {
    let [r, g, b] = c.to_rgb();
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    if c.alpha < 1.0 { format!("#{:02x}{:02x}{:02x}{:02x}", q(r), q(g), q(b), q(c.alpha)) } else { format!("#{:02x}{:02x}{:02x}", q(r), q(g), q(b)) }
}

fn op_from(s: &str) -> Option<PathOp> {
    Some(match s.to_ascii_lowercase().as_str() {
        "combine" | "add" | "union" => PathOp::Combine,
        "subtract" | "minus" => PathOp::Subtract,
        "intersect" => PathOp::Intersect,
        "exclude" | "xor" => PathOp::Exclude,
        "join" => PathOp::Join,
        _ => return None,
    })
}

fn op_name(op: PathOp) -> &'static str {
    match op {
        PathOp::Combine => "combine",
        PathOp::Subtract => "subtract",
        PathOp::Intersect => "intersect",
        PathOp::Exclude => "exclude",
        PathOp::Join => "join",
    }
}

/// Parses a path from the compact JSON form (or the serde form).
pub fn parse_path(v: &Value) -> std::result::Result<Path, String> {
    let subs = v.get("subpaths").and_then(Value::as_array).ok_or("path needs `subpaths`")?;
    let mut out = Vec::with_capacity(subs.len());
    for (i, s) in subs.iter().enumerate() {
        let knots_v = s.get("knots").and_then(Value::as_array).ok_or(format!("subpath {i} needs `knots`"))?;
        let mut knots = Vec::with_capacity(knots_v.len());
        for (j, k) in knots_v.iter().enumerate() {
            let knot = if k.is_array() {
                let a = pt(k).ok_or(format!("knot {i}.{j}: expected [x, y]"))?;
                Knot::corner(a.x, a.y)
            } else {
                let a = k.get("anchor").and_then(pt).ok_or(format!("knot {i}.{j}: missing anchor"))?;
                let get = |keys: &[&str]| keys.iter().find_map(|key| k.get(*key).and_then(pt));
                Knot {
                    anchor: a,
                    in_ctrl: get(&["in", "inCtrl", "in_ctrl"]).unwrap_or(a),
                    out_ctrl: get(&["out", "outCtrl", "out_ctrl"]).unwrap_or(a),
                    smooth: k.get("smooth").and_then(Value::as_bool).unwrap_or(false),
                }
            };
            knots.push(knot);
        }
        let op = match s.get("op").and_then(Value::as_str) {
            Some(o) => op_from(o).ok_or(format!("subpath {i}: unknown op `{o}`"))?,
            None => PathOp::Combine,
        };
        out.push(Subpath { closed: s.get("closed").and_then(Value::as_bool).unwrap_or(true), knots, op });
    }
    let fill_rule = match v.get("fillRule").or_else(|| v.get("fill_rule")).and_then(Value::as_str).map(str::to_ascii_lowercase).as_deref() {
        Some("evenodd" | "even-odd" | "even_odd") => FillRule::EvenOdd,
        _ => FillRule::NonZero,
    };
    Ok(Path { subpaths: out, fill_rule, inverted: v.get("inverted").and_then(Value::as_bool).unwrap_or(false) })
}

/// The compact JSON form of a path.
pub fn path_json(p: &Path) -> Value {
    let a = |q: Point| json!([q.x, q.y]);
    json!({
        "subpaths": p.subpaths.iter().map(|s| json!({
            "closed": s.closed,
            "op": op_name(s.op),
            "knots": s.knots.iter().map(|k| json!({ "anchor": a(k.anchor), "in": a(k.in_ctrl), "out": a(k.out_ctrl), "smooth": k.smooth })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "fillRule": if p.fill_rule == FillRule::EvenOdd { "evenodd" } else { "nonzero" },
        "inverted": p.inverted,
    })
}

fn fill_json(f: &Fill) -> Value {
    match f {
        Fill::Solid(c) => json!(hex(c)),
        Fill::Gradient { stops, angle, scale, style, reverse, .. } => json!({ "gradient": {
            "stops": stops.iter().map(|(t, c)| json!([t, hex(c)])).collect::<Vec<_>>(),
            "angle": angle, "scale": scale * 100.0, "style": format!("{style:?}").to_ascii_lowercase(), "reverse": reverse } }),
        Fill::Pattern { name, scale, .. } => json!({ "pattern": name, "scale": scale * 100.0 }),
    }
}

/// `"#rrggbb"` / `[r,g,b,a]` / `{"gradient":{"stops":[[t,"#hex"],…],"angle","scale","style","reverse"}}` /
/// `{"pattern":name,"scale"}`. `null` → `Ok(None)` (no fill).
fn parse_fill(v: &Value) -> std::result::Result<Option<Fill>, String> {
    if v.is_null() {
        return Ok(None);
    }
    if let Some(c) = color(v) {
        return Ok(Some(Fill::Solid(c)));
    }
    if let Some(g) = v.get("gradient") {
        let stops: Vec<(f32, Color)> = g
            .get("stops")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(|s| Some((s.get(0)?.as_f64()? as f32, color(s.get(1)?)?))).collect())
            .unwrap_or_else(|| vec![(0.0, Color::BLACK), (1.0, Color::WHITE)]);
        let style = match g.get("style").and_then(Value::as_str).unwrap_or("linear") {
            "radial" => GradientStyle::Radial,
            "angle" => GradientStyle::Angle,
            "reflected" => GradientStyle::Reflected,
            "diamond" => GradientStyle::Diamond,
            _ => GradientStyle::Linear,
        };
        return Ok(Some(Fill::gradient(
            stops,
            f64p(g, "angle").unwrap_or(90.0) as f32,
            f64p(g, "scale").unwrap_or(100.0) as f32 / 100.0,
            style,
            g.get("reverse").and_then(Value::as_bool).unwrap_or(false),
        )));
    }
    if let Some(name) = v.get("pattern").and_then(Value::as_str) {
        return Ok(Some(Fill::Pattern {
            name: name.into(),
            scale: f64p(v, "scale").unwrap_or(100.0) as f32 / 100.0,
            id: String::new(),
            angle: 0.0,
            link: true,
            phase: (0.0, 0.0),
        }));
    }
    Err(format!("unrecognised fill {v}"))
}

fn stroke_json(s: &ShapeStroke) -> Value {
    json!({
        "width": s.width, "fill": fill_json(&s.paint), "opacity": s.opacity * 100.0,
        "align": format!("{:?}", s.align).to_ascii_lowercase(), "cap": format!("{:?}", s.cap).to_ascii_lowercase(),
        "join": format!("{:?}", s.join).to_ascii_lowercase(), "miterLimit": s.miter_limit, "dashes": s.dashes, "dashOffset": s.dash_offset,
    })
}

/// Applies stroke keys onto `base` (`null` → no stroke).
pub fn parse_stroke(v: &Value, base: Option<ShapeStroke>, fg: Color) -> std::result::Result<Option<ShapeStroke>, String> {
    if v.is_null() || v.as_bool() == Some(false) {
        return Ok(None);
    }
    if !v.is_object() {
        return Err("stroke must be an object or null".into());
    }
    for (key, lo, hi) in [("width", 0.0, 1_000_000.0), ("opacity", 0.0, 100.0), ("miterLimit", 1.0, 500.0), ("dashOffset", -1_000_000.0, 1_000_000.0)] {
        if let Some(value) = v.get(key)
            && !value.as_f64().is_some_and(|n| n.is_finite() && (lo..=hi).contains(&n))
        {
            return Err(format!("`{key}` must be a finite number between {lo} and {hi}"));
        }
    }
    for key in ["align", "cap", "join"] {
        if v.get(key).is_some_and(|value| !value.is_string()) {
            return Err(format!("`{key}` must be a string"));
        }
    }
    let mut s = base.unwrap_or(ShapeStroke { paint: Fill::Solid(fg), ..Default::default() });
    if let Some(w) = f64p(v, "width") {
        s.width = w as f32;
    }
    if let Some(c) = v.get("color").or_else(|| v.get("fill")) {
        s.paint = parse_fill(c)?.ok_or("stroke colour cannot be null")?;
    }
    if let Some(o) = f64p(v, "opacity") {
        s.opacity = (o / 100.0) as f32;
    }
    if let Some(a) = v.get("align").and_then(Value::as_str) {
        s.align = match a {
            "inside" => StrokeAlign::Inside,
            "outside" => StrokeAlign::Outside,
            "center" | "centre" => StrokeAlign::Center,
            o => return Err(format!("unknown align `{o}`")),
        };
    }
    if let Some(c) = v.get("cap").and_then(Value::as_str) {
        s.cap = match c {
            "butt" => LineCap::Butt,
            "round" => LineCap::Round,
            "square" => LineCap::Square,
            o => return Err(format!("unknown cap `{o}`")),
        };
    }
    if let Some(j) = v.get("join").and_then(Value::as_str) {
        s.join = match j {
            "miter" => LineJoin::Miter,
            "round" => LineJoin::Round,
            "bevel" => LineJoin::Bevel,
            o => return Err(format!("unknown join `{o}`")),
        };
    }
    if let Some(m) = f64p(v, "miterLimit") {
        s.miter_limit = m as f32;
    }
    if let Some(d) = v.get("dashes") {
        s.dashes = match d {
            Value::Null => Vec::new(),
            Value::Array(a) if a.len() <= 32 => a
                .iter()
                .map(|x| {
                    x.as_f64()
                        .filter(|n| n.is_finite() && (0.0..=10000.0).contains(n))
                        .map(|x| x as f32)
                        .ok_or_else(|| "dash lengths must be finite numbers between 0 and 10000 widths".to_string())
                })
                .collect::<std::result::Result<Vec<_>, _>>()?,
            _ => return Err("`dashes` must be an array (multiples of the width)".into()),
        };
    }
    if let Some(o) = f64p(v, "dashOffset") {
        s.dash_offset = o as f32;
    }
    s.validate()?;
    Ok(Some(s))
}

fn live_json(l: &LiveShape) -> Value {
    serde_json::to_value(l).unwrap_or(Value::Null)
}

/// Builds a live shape from `kind` + geometry params.
fn live_from_params(cmd: &str, kind: &str, p: &Value, prev: Option<&LiveShape>) -> Result<LiveShape> {
    let prev_rect = match prev {
        Some(LiveShape::Rect { rect, .. } | LiveShape::Ellipse { rect } | LiveShape::Polygon { rect, .. }) => Some(*rect),
        _ => None,
    };
    let rect = || nums::<4>(p, "rect").or(prev_rect).ok_or_else(|| bad(cmd, "`rect`: [x, y, w, h] is required"));
    Ok(match kind {
        "rect" | "rectangle" => {
            let r = match prev {
                Some(LiveShape::Rect { radii, .. }) => *radii,
                _ => [0.0; 4],
            };
            LiveShape::Rect { rect: rect()?, radii: radii(p).unwrap_or(r) }
        }
        "roundedRect" | "roundRect" => {
            let r = match prev {
                Some(LiveShape::Rect { radii, .. }) if radii.iter().any(|v| *v > 0.0) => *radii,
                _ => [10.0; 4],
            };
            LiveShape::Rect { rect: rect()?, radii: radii(p).unwrap_or(r) }
        }
        "ellipse" => LiveShape::Ellipse { rect: rect()? },
        "polygon" | "star" => {
            let (ps, pr) = match prev {
                Some(LiveShape::Polygon { sides, star_ratio, .. }) => (*sides, *star_ratio),
                _ => (5, if kind == "star" { 0.5 } else { 1.0 }),
            };
            let sides = f64p(p, "sides").map_or(ps, |v| v.round().clamp(3.0, 100.0) as u32);
            let ratio = f64p(p, "starRatio").map_or(if kind == "star" && pr >= 1.0 { 0.5 } else { pr }, |v| v.clamp(0.01, 1.0));
            LiveShape::Polygon { rect: rect()?, sides, star_ratio: if kind == "polygon" && f64p(p, "starRatio").is_none() { 1.0 } else { ratio } }
        }
        "line" => {
            let (pf, pto, pw) = match prev {
                Some(LiveShape::Line { from, to, weight }) => (Some(*from), Some(*to), *weight),
                _ => (None, None, 1.0),
            };
            LiveShape::Line {
                from: nums::<2>(p, "from").or(pf).ok_or_else(|| bad(cmd, "`from`: [x, y] is required"))?,
                to: nums::<2>(p, "to").or(pto).ok_or_else(|| bad(cmd, "`to`: [x, y] is required"))?,
                weight: f64p(p, "weight").unwrap_or(pw).max(0.0),
            }
        }
        o => return Err(bad(cmd, format!("unknown kind `{o}`"))),
    })
}

/// The path `shape.create` draws for live-shape params `p` (the Shape tools preview it while dragging).
pub fn shape_path(p: &Value) -> Result<Path> {
    let kind = p.get("kind").and_then(Value::as_str).unwrap_or("rect");
    Ok(vector::shapes::live_path(&live_from_params("shape.create", kind, p, None)?))
}

fn live_kind(l: &LiveShape) -> &'static str {
    match l {
        LiveShape::Rect { radii, .. } if radii.iter().any(|r| *r > 0.0) => "roundedRect",
        LiveShape::Rect { .. } => "rect",
        LiveShape::Ellipse { .. } => "ellipse",
        LiveShape::Polygon { star_ratio, .. } if *star_ratio < 1.0 => "star",
        LiveShape::Polygon { .. } => "polygon",
        LiveShape::Line { .. } => "line",
    }
}

/// Re-renders a shape layer's cache for `doc` (canvas-clipped, document pixel format).
pub fn refresh_shape(doc: &Document, sh: &mut ShapeLayer) {
    sh.cache = Some(vector::render_shape(sh, doc.pixel_format(), doc.bounds()));
}

/// A staged stroke edit rendered with the same validation and cache refresh as `shape.edit`.
/// The caller owns this snapshot; the session and its history remain untouched.
pub fn preview_shape_stroke(doc: &Document, id: LayerId, params: &Value) -> Result<Document> {
    let mut result = doc.clone();
    let layer = result.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
    let LayerContent::Shape(shape) = &mut layer.content else {
        return Err(EngineError::Other("preview target is not a shape layer".into()));
    };
    shape.stroke = parse_stroke(params, shape.stroke.clone(), Color::BLACK).map_err(|e| bad("shape.edit", e))?;
    vector::flatten::validate_shape(&shape.path, shape.stroke.as_ref()).map_err(|e| bad("shape.edit", e))?;
    refresh_shape(doc, shape);
    Ok(result)
}

/// Moves a layer's vector content by `(dx, dy)`: shape paths (re-rendered) and linked vector
/// masks, recursing into groups. For the Move tool / `layer.move`.
pub fn translate_vectors(doc: &Document, l: &mut Layer, dx: f64, dy: f64) {
    let a = Affine::translate(dx, dy);
    if let Some(vm) = l.vector_mask.as_mut()
        && vm.linked
    {
        vm.path = vm.path.transform(&a);
    }
    match &mut l.content {
        LayerContent::Shape(sh) => {
            transform_shape(sh, &a);
            refresh_shape(doc, sh);
        }
        LayerContent::Group(g) => {
            for c in &mut g.children {
                translate_vectors(doc, c, dx, dy);
            }
        }
        _ => {}
    }
}

/// Applies an affine transform to a shape's path, keeping the live shape when the transform
/// is a translation plus axis-aligned scale (otherwise the shape stops being parametric).
pub fn transform_shape(sh: &mut ShapeLayer, a: &Affine) {
    sh.path = sh.path.transform(a);
    let [m0, m1, m2, m3, tx, ty] = a.m;
    let axis_aligned = m1.abs() < 1e-12 && m2.abs() < 1e-12 && m0 > 0.0 && m3 > 0.0;
    sh.live = match (sh.live.take(), axis_aligned) {
        (Some(l), true) => {
            let r = |r: [f64; 4]| [r[0] * m0 + tx, r[1] * m3 + ty, r[2] * m0, r[3] * m3];
            let pt = |p: [f64; 2]| [p[0] * m0 + tx, p[1] * m3 + ty];
            Some(match l {
                LiveShape::Rect { rect, radii } => {
                    let k = m0.min(m3);
                    LiveShape::Rect { rect: r(rect), radii: radii.map(|v| v * k) }
                }
                LiveShape::Ellipse { rect } => LiveShape::Ellipse { rect: r(rect) },
                LiveShape::Polygon { rect, sides, star_ratio } => LiveShape::Polygon { rect: r(rect), sides, star_ratio },
                LiveShape::Line { from, to, weight } => {
                    // Live-shape edits regenerate the line's filled bar from its parameters.
                    // Keep the weight in sync with a positive scale of the path, or an
                    // unchanged endpoint edit unexpectedly restores the old thickness (#3063).
                    // A line's width scales perpendicular to its direction.
                    let (dx, dy) = (to[0] - from[0], to[1] - from[1]);
                    let old_length = dx.hypot(dy);
                    let new_length = (dx * m0).hypot(dy * m3);
                    let width_scale = if old_length > 0.0 && new_length > 0.0 { m0 * m3 * old_length / new_length } else { m0.min(m3) };
                    LiveShape::Line { from: pt(from), to: pt(to), weight: weight * width_scale }
                }
            })
        }
        // Flips and 90° turns keep rectangles, ellipses and lines live (Image Rotation): the new
        // live box is the mapped box. Polygons/stars would change orientation, so they don't.
        (Some(l), false) if (m1.abs() < 1e-12 && m2.abs() < 1e-12) || (m0.abs() < 1e-12 && m3.abs() < 1e-12) => {
            let pt = |p: [f64; 2]| [m0 * p[0] + m2 * p[1] + tx, m1 * p[0] + m3 * p[1] + ty];
            let r = |r: [f64; 4]| {
                let (p, q) = (pt([r[0], r[1]]), pt([r[0] + r[2], r[1] + r[3]]));
                [p[0].min(q[0]), p[1].min(q[1]), (q[0] - p[0]).abs(), (q[1] - p[1]).abs()]
            };
            let k = (m0.abs() + m1.abs()).min(m2.abs() + m3.abs());
            match l {
                LiveShape::Rect { rect, radii } if radii.iter().all(|v| (v - radii[0]).abs() < 1e-9) => {
                    Some(LiveShape::Rect { rect: r(rect), radii: radii.map(|v| v * k) })
                }
                LiveShape::Ellipse { rect } => Some(LiveShape::Ellipse { rect: r(rect) }),
                LiveShape::Line { from, to, weight } => Some(LiveShape::Line { from: pt(from), to: pt(to), weight: weight * k }),
                _ => None,
            }
        }
        _ => None,
    };
    sh.psd_raw = None;
}

/// Is every anchor and control point of `path` a finite point?
fn finite_path(path: &Path) -> bool {
    path.subpaths.iter().flat_map(|s| &s.knots).all(|k| [k.anchor, k.in_ctrl, k.out_ctrl].iter().all(|p| p.x.is_finite() && p.y.is_finite()))
}

/// `path` through Distort / Perspective (`h`, a projective map of its points: exact for straight
/// segments, the end tangents of curves exact) or, with `warp`, Warp (split where the warp bends
/// it). Refused when the map sends a point to infinity (past a perspective's horizon).
pub(crate) fn map_path(path: &Path, h: Option<&photocraft_algo::transform::Homography>, warp: Option<&photocraft_geom::warp::Warp>) -> Result<Path> {
    let out = match (h, warp) {
        (_, Some(w)) => vector::map::map_path(
            path,
            |p| {
                let (x, y) = w.map(p.x, p.y);
                Point::new(x, y)
            },
            0.1,
        ),
        (Some(h), None) => vector::map::map_points(path, |p| {
            let (x, y) = h.apply(p.x, p.y);
            Point::new(x, y)
        }),
        (None, None) => path.clone(),
    };
    if !finite_path(&out) {
        return Err(EngineError::Other("the distortion sends part of the path out of bounds".into()));
    }
    Ok(out)
}

/// Distort, Perspective or Warp of a shape layer (#3009): the shape stays a shape, its path
/// mapped ([`map_path`]), as in Photoshop. A live rectangle, ellipse, polygon or line becomes a
/// plain path, as its parameters can't describe the distorted outline.
pub(crate) fn map_shape(sh: &mut ShapeLayer, h: Option<&photocraft_algo::transform::Homography>, warp: Option<&photocraft_geom::warp::Warp>) -> Result<()> {
    sh.path = map_path(&sh.path, h, warp)?;
    sh.live = None;
    sh.psd_raw = None;
    Ok(())
}

fn shape_info(s: &Session, id: LayerId) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let l = d.doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    let LayerContent::Shape(sh) = &l.content else {
        return Err(EngineError::Other(format!("layer {} is {} {} layer, not a shape layer", id.0, l.content.article(), l.content.kind_name())));
    };
    let bounds = sh.cache.as_ref().map(|c| c.content_bounds()).filter(|r| !r.is_empty()).map(|r| json!([r.x0, r.y0, r.width(), r.height()]));
    Ok(json!({
        "layer": id.0,
        "name": l.name,
        "path": path_json(&sh.path),
        "fill": sh.fill.as_ref().map(fill_json),
        "stroke": sh.stroke.as_ref().map(stroke_json),
        "live": sh.live.as_ref().map(live_json),
        "kind": sh.live.as_ref().map_or("path", live_kind),
        "bounds": bounds,
    }))
}

pub(crate) fn with_shape<R>(s: &mut Session, id: LayerId, label: &str, f: impl FnOnce(&mut ShapeLayer, &mut Layer) -> Result<R>) -> Result<R> {
    if let Some(st) = s.active() {
        let locks = st.doc.effective_locks(id);
        if locks.all || locks.pixels {
            return Err(EngineError::Other("shape layer is locked".into()));
        }
    }
    s.edit(label, |doc, _| {
        let snapshot = doc.clone();
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        if !matches!(l.content, LayerContent::Shape(_)) {
            return Err(EngineError::Other(format!("layer {} is {} {} layer, not a shape layer", id.0, l.content.article(), l.content.kind_name())));
        }
        let LayerContent::Shape(mut sh) = std::mem::replace(&mut l.content, LayerContent::Fill(Fill::Solid(Color::BLACK))) else {
            return Err(EngineError::Other(format!("layer {} is not a shape layer", id.0)));
        };
        let r = f(&mut sh, l);
        if r.is_ok() {
            refresh_shape(&snapshot, &mut sh);
        }
        l.content = LayerContent::Shape(sh);
        r
    })
}

fn default_name(kind: &str) -> &'static str {
    match kind {
        "rect" | "rectangle" => "Rectangle",
        "roundedRect" | "roundRect" => "Rounded Rectangle",
        "ellipse" => "Ellipse",
        "polygon" => "Polygon",
        "star" => "Star",
        "line" => "Line",
        _ => "Shape",
    }
}

fn shape_create(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "shape.create";
    let kind = p.get("kind").and_then(Value::as_str).unwrap_or("rect");
    let (path, live) = if kind == "path" {
        let v = p.get("path").ok_or_else(|| bad(CMD, "kind \"path\" needs `path`"))?;
        (parse_path(v).map_err(|e| bad(CMD, e))?, None)
    } else {
        let l = live_from_params(CMD, kind, p, None)?;
        (vector::shapes::live_path(&l), Some(l))
    };
    let fg = s.tools.foreground;
    let fgc = Color::rgba(fg[0], fg[1], fg[2], fg[3]);
    let fill = match p.get("fill") {
        Some(v) => parse_fill(v).map_err(|e| bad(CMD, e))?,
        None => Some(Fill::Solid(fgc)),
    };
    let stroke = match p.get("stroke") {
        Some(v) => parse_stroke(v, None, fgc).map_err(|e| bad(CMD, e))?,
        None => None,
    };
    vector::flatten::validate_shape(&path, stroke.as_ref()).map_err(|e| bad(CMD, e))?;
    // Add to an existing shape layer with a path operation (Shape tool in combine/subtract… mode).
    if let Some(target) = p.get("addTo").and_then(Value::as_u64) {
        let op = p.get("op").and_then(Value::as_str).map_or(Some(PathOp::Combine), op_from).ok_or_else(|| bad(CMD, "unknown `op`"))?;
        let id = LayerId(target);
        with_shape(s, id, "Add Shape", |sh, _| {
            for mut sp in path.subpaths {
                sp.op = op;
                sh.path.subpaths.push(sp);
            }
            sh.live = None;
            sh.psd_raw = None;
            vector::flatten::validate_shape(&sh.path, sh.stroke.as_ref()).map_err(|e| bad(CMD, e))?;
            Ok(())
        })?;
        return shape_info(s, id);
    }
    let name = p.get("name").and_then(Value::as_str).map(str::to_string);
    let mut sh = ShapeLayer { path, fill, stroke, live, ..Default::default() };
    let id = s.edit("New Shape Layer", |doc, active| {
        refresh_shape(doc, &mut sh);
        let name = name.unwrap_or_else(|| doc.next_layer_name(default_name(kind)));
        let id = doc.insert_above(*active, Layer::new(name, LayerContent::Shape(sh)));
        *active = Some(id);
        Ok(id)
    })?;
    shape_info(s, id)
}

fn shape_edit(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "shape.edit";
    let id = layer_id(s, p)?;
    let fg = s.tools.foreground;
    let fgc = Color::rgba(fg[0], fg[1], fg[2], fg[3]);
    let name = p.get("name").and_then(Value::as_str).map(str::to_string);
    with_shape(s, id, "Edit Shape", |sh, layer| {
        if let Some(v) = p.get("path") {
            sh.path = parse_path(v).map_err(|e| bad(CMD, e))?;
            sh.live = None;
            sh.psd_raw = None;
        }
        let geom_keys = ["kind", "rect", "radii", "radius", "sides", "starRatio", "from", "to", "weight"];
        if geom_keys.iter().any(|k| p.get(*k).is_some()) {
            let kind = p.get("kind").and_then(Value::as_str).map(str::to_string).or_else(|| sh.live.as_ref().map(|l| live_kind(l).to_string()));
            let kind = kind.ok_or_else(|| bad(CMD, "this shape has no live-shape parameters; pass `kind` or `path`"))?;
            // Explicit radii on a plain rectangle keep it a (now rounded) rectangle.
            let kind = if kind == "rect" && radii(p).is_some_and(|r| r.iter().any(|v| *v > 0.0)) { "roundedRect".to_string() } else { kind };
            let l = live_from_params(CMD, &kind, p, sh.live.as_ref())?;
            // A transformed live shape's path may carry nonuniform corner scaling
            // that its scalar radii cannot represent. Re-entering unchanged values
            // must not regenerate the outline and silently discard that geometry.
            if sh.live.as_ref() != Some(&l) {
                sh.path = vector::shapes::live_path(&l);
                sh.live = Some(l);
                sh.psd_raw = None;
            }
        }
        if let Some(m) = nums::<6>(p, "transform") {
            transform_shape(sh, &Affine { m });
        }
        if let Some([dx, dy]) = nums::<2>(p, "move") {
            transform_shape(sh, &Affine::translate(dx, dy));
        }
        if let Some(op) = p.get("op").and_then(Value::as_str) {
            // Operation of the selected subpath(s) (`subpath`: index, default all but the first).
            let op = op_from(op).ok_or_else(|| bad(CMD, format!("unknown op `{op}`")))?;
            match p.get("subpath").and_then(Value::as_u64) {
                Some(i) => sh.path.subpaths.get_mut(i as usize).ok_or_else(|| bad(CMD, "no such subpath"))?.op = op,
                None => sh.path.subpaths.iter_mut().skip(1).for_each(|sp| sp.op = op),
            }
            sh.live = None;
        }
        if let Some(r) = p.get("fillRule").and_then(Value::as_str) {
            sh.path.fill_rule = if r.eq_ignore_ascii_case("evenodd") { FillRule::EvenOdd } else { FillRule::NonZero };
        }
        if let Some(v) = p.get("fill") {
            sh.fill = parse_fill(v).map_err(|e| bad(CMD, e))?;
        }
        if let Some(v) = p.get("stroke") {
            sh.stroke = parse_stroke(v, sh.stroke.clone(), fgc).map_err(|e| bad(CMD, e))?;
        }
        if let Some(n) = name {
            layer.name = n;
        }
        vector::flatten::validate_shape(&sh.path, sh.stroke.as_ref()).map_err(|e| bad(CMD, e))?;
        Ok(())
    })?;
    shape_info(s, id)
}

fn shape_rasterize(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_id(s, p)?;
    s.edit("Rasterize Shape", |doc, _| {
        let snapshot = doc.clone();
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        let LayerContent::Shape(sh) = &mut l.content else {
            return Err(EngineError::Other(format!("layer {} is not a shape layer", id.0)));
        };
        if sh.cache.is_none() {
            refresh_shape(&snapshot, sh);
        }
        let fmt = snapshot.pixel_format();
        let surface = sh.cache.take().unwrap_or_else(|| photocraft_raster::Surface::new(fmt));
        let surface = if surface.format() == fmt { surface } else { surface.convert(fmt) };
        l.content = LayerContent::Raster(surface);
        l.psd_blocks.retain(|(k, _)| !SHAPE_BLOCKS.contains(&k));
        Ok(())
    })?;
    Ok(json!({ "layer": id.0 }))
}

// ---------------------------------------------------------------------------
// Document paths
// ---------------------------------------------------------------------------

pub(crate) fn is_work(name: Option<&str>) -> bool {
    name.is_none_or(|n| n.is_empty() || n.eq_ignore_ascii_case("work") || n == "Work Path")
}

/// The path a command targets: `"work"`/absent → the work path (falling back to the active
/// layer's shape path or vector mask when there is none), `"layer"` → the active layer's,
/// otherwise a saved path by name.
fn resolve_path(s: &Session, name: Option<&str>) -> Result<Path> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let layer_path = || -> Option<Path> {
        let l = d.doc.layer(d.active_layer?)?;
        match &l.content {
            LayerContent::Shape(sh) => Some(sh.path.clone()),
            _ => l.vector_mask.as_ref().map(|v| v.path.clone()),
        }
    };
    if name == Some("layer") {
        return layer_path().ok_or_else(|| EngineError::Other("active layer has no shape path or vector mask".into()));
    }
    if is_work(name) {
        return d.doc.work_path.clone().or_else(layer_path).ok_or_else(|| EngineError::Other("no work path".into()));
    }
    let n = name.unwrap_or_default();
    d.doc.paths.iter().find(|p| p.name == n).map(|p| p.path.clone()).ok_or_else(|| EngineError::Other(format!("no path named \"{n}\"")))
}

/// The path an edit targets, except a shape layer's (edit those with [`with_shape`], which
/// re-renders them): `layer`'s vector mask when given, else the work path or a saved path.
pub(crate) fn path_mut<'a>(doc: &'a mut Document, name: &str, layer: Option<LayerId>) -> Result<&'a mut Path> {
    if let Some(id) = layer {
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        return l.vector_mask.as_mut().map(|vm| &mut vm.path).ok_or_else(|| EngineError::Other("active layer has no shape path or vector mask".into()));
    }
    if is_work(Some(name)) {
        return doc.work_path.as_mut().ok_or_else(|| EngineError::Other("no work path".into()));
    }
    doc.paths.iter_mut().find(|q| q.name == name).map(|q| &mut q.path).ok_or_else(|| EngineError::Other(format!("no path named \"{name}\"")))
}

fn paths_list(s: &Session) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let doc = &d.doc;
    let knots = |p: &Path| p.subpaths.iter().map(|s| s.knots.len()).sum::<usize>();
    let layer = d.active_layer.and_then(|id| doc.layer(id)).and_then(|l| match &l.content {
        LayerContent::Shape(sh) => Some(json!({ "kind": "shape", "layer": l.id.0, "name": format!("{} Shape Path", l.name), "knots": knots(&sh.path) })),
        _ => {
            l.vector_mask.as_ref().map(|v| json!({ "kind": "vectorMask", "layer": l.id.0, "name": format!("{} Vector Mask", l.name), "knots": knots(&v.path) }))
        }
    });
    Ok(json!({
        "paths": doc.paths.iter().map(|p| json!({ "name": p.name, "knots": knots(&p.path), "subpaths": p.path.subpaths.len() })).collect::<Vec<_>>(),
        "workPath": doc.work_path.as_ref().map(|p| json!({ "knots": knots(p), "subpaths": p.subpaths.len() })),
        "clippingPath": doc.clipping_path.as_ref().map(|c| json!({ "name": c.name, "flatness": c.flatness })),
        "layerPath": layer,
    }))
}

fn path_set(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "path.set";
    let path = parse_path(p.get("path").ok_or_else(|| bad(CMD, "missing `path`"))?).map_err(|e| bad(CMD, e))?;
    let name = p.get("name").and_then(Value::as_str).map(str::to_string);
    let append = p.get("op").and_then(Value::as_str).map(|o| op_from(o).ok_or_else(|| bad(CMD, format!("unknown op `{o}`")))).transpose()?;
    s.edit(if is_work(name.as_deref()) { "Work Path" } else { "Save Path" }, |doc, _| {
        let merge = |old: Option<&Path>, new: Path| match (old, append) {
            (Some(o), Some(op)) => {
                let mut o = o.clone();
                o.subpaths.extend(new.subpaths.into_iter().map(|mut sp| {
                    sp.op = op;
                    sp
                }));
                o
            }
            _ => new,
        };
        if is_work(name.as_deref()) {
            doc.work_path = Some(merge(doc.work_path.as_ref(), path));
        } else {
            let n = name.clone().unwrap_or_default();
            match doc.paths.iter_mut().find(|q| q.name == n) {
                Some(q) => {
                    q.path = merge(Some(&q.path), path);
                }
                None => doc.paths.push(NamedPath { name: n, path, psd_raw: None }),
            }
        }
        Ok(())
    })?;
    let path = resolve_path(s, name.as_deref())?;
    Ok(json!({ "name": name.unwrap_or_else(|| "work".into()), "path": path_json(&path) }))
}

fn path_delete(s: &mut Session, p: &Value) -> Result<Value> {
    let name = p.get("name").and_then(Value::as_str).map(str::to_string);
    s.edit("Delete Path", |doc, _| {
        if is_work(name.as_deref()) {
            return doc.work_path.take().map(|_| ()).ok_or_else(|| EngineError::Other("no work path".into()));
        }
        let n = name.clone().unwrap_or_default();
        let before = doc.paths.len();
        doc.paths.retain(|q| q.name != n);
        if doc.paths.len() == before {
            return Err(EngineError::Other(format!("no path named \"{n}\"")));
        }
        if doc.clipping_path.as_ref().is_some_and(|c| c.name == n) {
            doc.clipping_path = None;
        }
        Ok(())
    })?;
    Ok(Value::Null)
}

fn clipping_path_set(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "path.clippingPath.set";
    let name = p.get("name").and_then(Value::as_str).filter(|name| !name.is_empty()).ok_or_else(|| bad(CMD, "missing saved path `name`"))?.to_owned();
    let flatness = match p.get("flatness") {
        Some(value) => {
            value.as_f64().filter(|value| value.is_finite() && (0.0..=100.0).contains(value)).ok_or_else(|| bad(CMD, "`flatness` must be between 0 and 100"))?
                as f32
        }
        None => 0.0,
    };
    s.edit("Clipping Path", |doc, _| {
        if !doc.paths.iter().any(|path| path.name == name) {
            return Err(EngineError::Other(format!("no saved path named \"{name}\"")));
        }
        doc.clipping_path = Some(ClippingPath { name: name.clone(), flatness });
        Ok(())
    })?;
    Ok(json!({ "name": name, "flatness": flatness }))
}

fn clipping_path_clear(s: &mut Session, _p: &Value) -> Result<Value> {
    s.edit("Clear Clipping Path", |doc, _| {
        doc.clipping_path.take().ok_or_else(|| EngineError::Other("no clipping path".into()))?;
        Ok(())
    })?;
    Ok(Value::Null)
}

/// The path `path.transform` edits: `layer`'s shape path or vector mask, else the work path or a
/// saved path.
fn transform_target(s: &Session, name: &str, layer: Option<LayerId>) -> Result<Path> {
    let Some(id) = layer else { return resolve_path(s, Some(name)) };
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let l = d.doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    match &l.content {
        LayerContent::Shape(sh) => Ok(sh.path.clone()),
        _ => l.vector_mask.as_ref().map(|v| v.path.clone()).ok_or_else(|| EngineError::Other("target layer has no shape path or vector mask".into())),
    }
}

/// `n` numbers at `key` (`[x0, y0, x1, y1]` or `[[x, y] × 4]`, flattened): finite, magnitude at
/// most 1000000. `None` when `key` is absent.
fn coords(p: &Value, key: &str, n: usize) -> Result<Option<Vec<f64>>> {
    const CMD: &str = "path.transform";
    let Some(v) = p.get(key) else { return Ok(None) };
    let flat: Vec<&Value> = match v.as_array() {
        Some(a) if a.iter().all(Value::is_array) => a.iter().filter_map(Value::as_array).flatten().collect(),
        Some(a) => a.iter().collect(),
        None => Vec::new(),
    };
    let nums: Vec<f64> = flat.iter().filter_map(|v| v.as_f64()).filter(|x| x.is_finite() && x.abs() <= 1_000_000.0).collect();
    if flat.len() != n || nums.len() != n {
        return Err(bad(CMD, format!("`{key}` must hold {n} finite numbers with magnitude at most 1000000")));
    }
    Ok(Some(nums))
}

/// How `path.transform` moves the path.
enum PathMap {
    Affine(Affine),
    /// Distort / Perspective: the box's corners go to a quad.
    Projective(photocraft_algo::transform::Homography),
    Warp(Box<photocraft_geom::warp::Warp>),
}

fn path_transform(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "path.transform";
    let name = p.get("name").and_then(Value::as_str).unwrap_or("work").to_owned();
    let layer_id = if name == "layer" { Some(layer_id(s, p)?) } else { None };
    // The box Distort, Perspective and Warp frame: `rect`, else the path's control bounds.
    let frame = |s: &Session| -> Result<[f64; 4]> {
        let r = match coords(p, "rect", 4)? {
            Some(r) => [r[0], r[1], r[2], r[3]],
            None => {
                let (x0, y0, x1, y1) = transform_target(s, &name, layer_id)?.control_bounds().ok_or_else(|| bad(CMD, "path has no points to transform"))?;
                [x0, y0, x1, y1]
            }
        };
        if r[2] - r[0] <= 0.0 || r[3] - r[1] <= 0.0 {
            return Err(bad(CMD, "`rect` must have a positive width and height (a straight path needs a `rect`)"));
        }
        Ok(r)
    };
    let map = if let Some(values) = p.get("matrix") {
        let values = values.as_array().filter(|values| values.len() == 6).ok_or_else(|| bad(CMD, "`matrix` must contain six numbers"))?;
        let mut m = [0.0; 6];
        for (slot, value) in m.iter_mut().zip(values) {
            *slot = value
                .as_f64()
                .filter(|number| number.is_finite() && number.abs() <= 1_000_000.0)
                .ok_or_else(|| bad(CMD, "matrix entries must be finite numbers with magnitude at most 1000000"))?;
        }
        PathMap::Affine(Affine { m })
    } else if let Some(q) = coords(p, "quad", 8)? {
        // Distort and Perspective (#3027): where the frame's corners go, clockwise from top-left.
        let quad = [[q[0], q[1]], [q[2], q[3]], [q[4], q[5]], [q[6], q[7]]];
        let h = photocraft_algo::transform::Homography::rect_to_quad(frame(s)?, quad)
            .filter(|h| h.0.iter().all(|v| v.is_finite()) && h.inverse().is_some())
            .ok_or_else(|| bad(CMD, "degenerate quad"))?;
        let [a, b, c, d, e, f, g, hh, i] = h.0;
        if g.abs() < 1e-12 && hh.abs() < 1e-12 && i.abs() > 1e-12 {
            // A parallelogram is affine: a live shape stays live.
            PathMap::Affine(Affine { m: [a / i, d / i, b / i, e / i, c / i, f / i] })
        } else {
            PathMap::Projective(h)
        }
    } else if ["warp", "style", "mesh"].iter().any(|key| p.get(*key).is_some()) {
        let w = crate::warp_cmds::warp_from_params(CMD, p, frame(s)?)?;
        if [w.bend, w.h_distort, w.v_distort].iter().chain(w.bounds.iter()).any(|v| !v.is_finite() || v.abs() > 1_000_000.0) {
            return Err(bad(CMD, "warp values must be finite numbers with magnitude at most 1000000"));
        }
        PathMap::Warp(Box::new(w))
    } else {
        let number = |key: &str, default: f64| -> Result<f64> {
            match p.get(key) {
                Some(value) => value
                    .as_f64()
                    .filter(|n| n.is_finite() && n.abs() <= 1_000_000.0)
                    .ok_or_else(|| bad(CMD, format!("`{key}` must be a finite number with magnitude at most 1000000"))),
                None => Ok(default),
            }
        };
        if !["translateX", "translateY", "scaleX", "scaleY", "angle"].iter().any(|key| p.get(*key).is_some()) {
            return Err(bad(CMD, "pass `matrix`, `quad`, a warp (`style`, `mesh` or `warp`) or translate/scale/angle parameters"));
        }
        let (tx, ty) = (number("translateX", 0.0)?, number("translateY", 0.0)?);
        let (sx, sy) = (number("scaleX", 1.0)?, number("scaleY", 1.0)?);
        let angle = number("angle", 0.0)?;
        // An explicit `layer` parameter determines both the transformed path and its pivot (#3062).
        let path = transform_target(s, &name, layer_id)?;
        let (x0, y0, x1, y1) = path.control_bounds().ok_or_else(|| bad(CMD, "path has no points to transform"))?;
        let (cx, cy) = ((x0 + x1) * 0.5, (y0 + y1) * 0.5);
        PathMap::Affine(
            Affine::translate(-cx, -cy)
                .then(&Affine { m: [sx, 0.0, 0.0, sy, 0.0, 0.0] })
                .then(&Affine::rotate(angle.to_radians()))
                .then(&Affine::translate(cx + tx, cy + ty)),
        )
    };
    let out = match &map {
        PathMap::Affine(a) => json!({ "name": name, "matrix": a.m }),
        PathMap::Projective(h) => json!({ "name": name, "homography": h.0 }),
        PathMap::Warp(_) => json!({ "name": name, "warp": true }),
    };
    if let Some(id) = layer_id
        && s.active().and_then(|d| d.doc.layer(id)).is_some_and(|layer| matches!(layer.content, LayerContent::Shape(_)))
    {
        with_shape(s, id, "Transform Path", |shape, _| {
            match &map {
                PathMap::Affine(a) => transform_shape(shape, a),
                PathMap::Projective(h) => map_shape(shape, Some(h), None)?,
                PathMap::Warp(w) => map_shape(shape, None, Some(w))?,
            }
            shape.psd_raw = None;
            Ok(())
        })?;
        return Ok(out);
    }
    s.edit("Transform Path", |doc, _| {
        let path = path_mut(doc, &name, layer_id)?;
        *path = match &map {
            PathMap::Affine(a) => path.transform(a),
            PathMap::Projective(h) => map_path(path, Some(h), None)?,
            PathMap::Warp(w) => map_path(path, None, Some(w))?,
        };
        Ok(())
    })?;
    Ok(out)
}

fn path_rename(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "path.rename";
    let name = p.get("name").and_then(Value::as_str).map(str::to_string);
    let to = p.get("to").and_then(Value::as_str).filter(|t| !t.is_empty()).ok_or_else(|| bad(CMD, "missing `to`"))?.to_string();
    if is_work(Some(&to)) {
        return Err(bad(CMD, "`to` cannot name the work path"));
    }
    s.edit(if is_work(name.as_deref()) { "Save Path" } else { "Rename Path" }, |doc, _| {
        if doc.paths.iter().any(|q| q.name == to) {
            return Err(EngineError::Other(format!("a path named \"{to}\" already exists")));
        }
        if is_work(name.as_deref()) {
            // Photoshop: saving the work path turns it into a named path.
            let path = doc.work_path.take().ok_or_else(|| EngineError::Other("no work path".into()))?;
            doc.paths.push(NamedPath { name: to.clone(), path, psd_raw: None });
            return Ok(());
        }
        let n = name.clone().unwrap_or_default();
        let q = doc.paths.iter_mut().find(|q| q.name == n).ok_or_else(|| EngineError::Other(format!("no path named \"{n}\"")))?;
        q.name = to.clone();
        if let Some(c) = doc.clipping_path.as_mut()
            && c.name == n
        {
            c.name = to.clone();
        }
        Ok(())
    })?;
    Ok(json!({ "name": to }))
}

fn path_to_selection(s: &mut Session, p: &Value) -> Result<Value> {
    let path = resolve_path(s, p.get("name").and_then(Value::as_str))?;
    let feather = f64p(p, "feather").unwrap_or(0.0) as f32;
    let aa = p.get("antiAlias").and_then(Value::as_bool).unwrap_or(true);
    let mode = SelectionMode::parse(p.get("mode").and_then(Value::as_str).unwrap_or("replace"));
    let area = s.active().ok_or(EngineError::NoDocument)?.doc.bounds();
    let mut mask = vector::fill_rasterizer(&path, vector::DEFAULT_TOLERANCE).render(area);
    if !aa {
        for v in &mut mask {
            *v = if *v >= 0.5 { 1.0 } else { 0.0 };
        }
    }
    if feather > 0.0 {
        mask = sel::feather(&mask, area.width() as usize, area.height() as usize, feather);
    }
    let selected = s.edit("Make Selection", |doc, _| {
        doc.selection = sel::combine(doc.selection.as_ref(), &mask, area, mode);
        Ok(doc.selection.is_some())
    })?;
    Ok(json!({ "selected": selected }))
}

fn select_to_work_path(s: &mut Session, p: &Value) -> Result<Value> {
    let tol = f64p(p, "tolerance").unwrap_or(2.0).clamp(0.5, 10.0);
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let area = d.doc.bounds();
    let sel_surface = d.doc.selection.as_ref().ok_or_else(|| EngineError::Other("no selection".into()))?;
    let mask = sel::mask_from_surface(Some(sel_surface), area);
    let path = vector::trace::trace_mask(&mask, area, 0.0, tol);
    let knots: usize = path.subpaths.iter().map(|s| s.knots.len()).sum();
    let subpaths = path.subpaths.len();
    s.edit("Make Work Path", |doc, _| {
        doc.work_path = Some(path);
        Ok(())
    })?;
    Ok(json!({ "subpaths": subpaths, "knots": knots }))
}

fn path_fill(s: &mut Session, p: &Value) -> Result<Value> {
    let path = resolve_path(s, p.get("name").and_then(Value::as_str))?;
    let fg = s.tools.foreground;
    let c = p.get("color").and_then(color).unwrap_or(Color::rgba(fg[0], fg[1], fg[2], fg[3]));
    let rgb = c.to_rgb();
    let src = [rgb[0], rgb[1], rgb[2], c.alpha];
    let opacity = (f64p(p, "opacity").unwrap_or(100.0) / 100.0).clamp(0.0, 1.0) as f32;
    let mode = p.get("mode").and_then(Value::as_str).and_then(blend_from_str).unwrap_or(BlendMode::Normal);
    let feather = f64p(p, "feather").unwrap_or(0.0) as f32;
    let aa = p.get("antiAlias").and_then(Value::as_bool).unwrap_or(true);
    let id = layer_id(s, p)?;
    s.edit("Fill Path", |doc, _| {
        let area = doc.bounds();
        let r = vector::fill_rasterizer(&path, vector::DEFAULT_TOLERANCE);
        // The feather reaches 3σ = 3 × radius (σ = radius, as in Photoshop), plus a pixel.
        let feather_pad = (feather.ceil() as i32).saturating_mul(3).saturating_add(1);
        let area = r.pixel_bounds().map_or(area, |b| b.inflate(feather_pad).intersect(&area));
        if area.is_empty() {
            return Ok(());
        }
        let mut cov = r.render(area);
        if !aa {
            cov.iter_mut().for_each(|v| *v = if *v >= 0.5 { 1.0 } else { 0.0 });
        }
        if feather > 0.0 {
            cov = sel::feather(&cov, area.width() as usize, area.height() as usize, feather);
        }
        // An active selection limits the fill, soft edges included, as every other fill does (#2387).
        if let Some(selection) = doc.selection.as_ref() {
            let mask = sel::mask_from_surface(Some(selection), area);
            cov.iter_mut().zip(&mask).for_each(|(c, m)| *c *= m);
        }
        let locks = doc.effective_locks(id);
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        if locks.pixels || locks.all {
            return Err(EngineError::Other(format!("Could not complete your request because the layer \"{}\" is locked", l.name)));
        }
        let lock = locks.transparency;
        let surf = l.surface_mut().ok_or_else(|| EngineError::Other("Fill Path needs a pixel layer".into()))?;
        paint_coverage(surf, area, &cov, src, opacity, mode, lock);
        Ok(())
    })?;
    Ok(Value::Null)
}

/// Composites `src` × coverage onto a surface over `area` (any pixel format).
fn paint_coverage(surf: &mut photocraft_raster::Surface, area: Rect, cov: &[f32], src: [f32; 4], opacity: f32, mode: BlendMode, lock_alpha: bool) {
    let fmt = surf.format();
    let ch = fmt.channels();
    let mut vals = surf.read_region(area);
    for (px, c) in vals.chunks_exact_mut(ch).zip(cov) {
        if *c <= 0.0 {
            continue;
        }
        let dst = photocraft_raster::to_rgba(&fmt, px);
        let mut out = photocraft_compose::psblend::composite(mode, dst, [src[0], src[1], src[2], src[3] * c], opacity);
        if lock_alpha {
            out[3] = dst[3];
        }
        photocraft_raster::from_rgba_into(&fmt, out, px);
    }
    surf.write_region(area, &vals);
    surf.prune();
}

/// Stroke Path's brush-based tools besides the Brush, Pencil and Eraser it renders itself: each
/// paints the path with its own `paint.*` command (Photoshop's Stroke Path dialog Tool list).
pub const STROKE_PATH_TOOLS: &[(&str, &str)] = &[
    ("backgroundEraser", "paint.backgroundEraser"),
    ("cloneStamp", "paint.cloneStamp"),
    ("patternStamp", "paint.patternStamp"),
    ("healingBrush", "paint.healingBrush"),
    ("spotHealingBrush", "paint.spotHealing"),
    ("historyBrush", "paint.historyBrush"),
    ("blur", "paint.blur"),
    ("sharpen", "paint.sharpen"),
    ("smudge", "paint.smudge"),
    ("dodge", "paint.dodge"),
    ("burn", "paint.burn"),
    ("sponge", "paint.sponge"),
    ("colorReplacement", "paint.colorReplacement"),
    ("mixerBrush", "paint.mixerBrush"),
];

/// At most this many points per subpath when Simulate Pressure resamples it.
const MAX_PRESSURE_POINTS: f64 = 200_000.0;

/// The stroke points along one flattened subpath. With `simulate` (Stroke Path's Simulate
/// Pressure) the subpath is resampled about every pixel and the pressure rises from 0 at its start
/// to full at the middle and falls back to 0 at its end, so a brush whose size or opacity follows
/// pen pressure tapers at both ends, as in Photoshop. Otherwise every point has full pressure.
fn stroke_points(pts: &[(f64, f64)], closed: bool, simulate: bool) -> Vec<photocraft_paint::StrokePoint> {
    use photocraft_paint::StrokePoint;
    let mut line = pts.to_vec();
    if closed && let Some(&first) = pts.first() {
        line.push(first);
    }
    let full = |line: Vec<(f64, f64)>| line.into_iter().map(|(x, y)| StrokePoint::new(x, y, 1.0)).collect();
    if !simulate {
        return full(line);
    }
    let seg = |a: &(f64, f64), b: &(f64, f64)| (b.0 - a.0).hypot(b.1 - a.1);
    let total: f64 = line.iter().zip(line.iter().skip(1)).map(|(a, b)| seg(a, b)).sum();
    if !(total.is_finite() && total > 0.0) {
        return full(line);
    }
    let step = (total / MAX_PRESSURE_POINTS).max(1.0);
    let pressure = |d: f64| (d / total * std::f64::consts::PI).sin().clamp(0.0, 1.0) as f32;
    let mut out = Vec::new();
    let mut walked = 0.0;
    for (a, b) in line.iter().zip(line.iter().skip(1)) {
        let len = seg(a, b);
        let n = (len / step).ceil().clamp(1.0, MAX_PRESSURE_POINTS) as usize;
        for i in 0..n {
            let t = i as f64 / n as f64;
            out.push(StrokePoint::new(a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t, pressure(walked + len * t)));
        }
        walked += len;
    }
    if let Some(&(x, y)) = line.last() {
        out.push(StrokePoint::new(x, y, 0.0));
    }
    out
}

/// Stroke Path with a tool that has its own `paint.*` command: the command runs once per subpath,
/// with the caller's params (size, opacity, the tool's options) and the path's points, all in one
/// history step. Only `path.stroke` itself is journalled, so a recorded action replays it once.
fn stroke_with_command(s: &mut Session, p: &Value, cmd: &str, lines: &[vector::Polyline], simulate: bool) -> Result<Value> {
    let mut params = p.as_object().cloned().unwrap_or_default();
    for k in ["name", "tool", "simulatePressure", "coalesce"] {
        params.remove(k);
    }
    let journal = s.journal.len();
    let r = crate::analysis_cmds::compound(s, "Stroke Path", |s| {
        // The union of what each subpath's command changed; None when one of them didn't say.
        let mut dmg = Some(Rect::EMPTY);
        for pl in lines {
            let pts = stroke_points(&pl.pts, pl.closed, simulate);
            if pts.is_empty() {
                continue;
            }
            let mut call = params.clone();
            call.insert("points".into(), Value::Array(pts.iter().map(|q| json!([q.x, q.y, q.pressure])).collect()));
            s.execute(cmd, Value::Object(call))?;
            let r = s.active().and_then(|d| d.last_damage);
            dmg = dmg.zip(r).map(|(a, b)| {
                if a.is_empty() {
                    b
                } else if b.is_empty() {
                    a
                } else {
                    a.union(&b)
                }
            });
        }
        Ok(dmg)
    });
    s.journal.truncate(journal);
    let dmg = r?;
    if let Some(st) = s.active_mut() {
        st.last_damage = dmg;
    }
    let dmg = dmg.unwrap_or_else(|| s.active().map_or(Rect::EMPTY, |d| d.doc.bounds()));
    Ok(json!({ "damage": [dmg.x0, dmg.y0, dmg.width(), dmg.height()] }))
}

fn path_stroke(s: &mut Session, p: &Value) -> Result<Value> {
    let path = resolve_path(s, p.get("name").and_then(Value::as_str))?;
    let tool = p.get("tool").and_then(Value::as_str).unwrap_or("brush");
    let simulate = match p.get("simulatePressure") {
        None | Some(Value::Null) => false,
        Some(v) => v.as_bool().ok_or_else(|| bad("path.stroke", "`simulatePressure` must be true or false"))?,
    };
    if let Some(&(_, cmd)) = STROKE_PATH_TOOLS.iter().find(|(t, _)| *t == tool) {
        let lines = vector::flatten_path(&path, 0.1);
        return stroke_with_command(s, p, cmd, &lines, simulate);
    }
    let base = s.tools.brush.clone();
    let fg = s.tools.foreground;
    let mut brush = photocraft_paint::BrushSettings {
        size: f64p(p, "size").map_or(base.size, |v| v as f32),
        hardness: f64p(p, "hardness").map_or(base.hardness, |v| v as f32),
        opacity: f64p(p, "opacity").map_or(base.opacity, |v| (v / 100.0) as f32),
        color: p.get("color").and_then(color).map_or(fg, |c| {
            let v = c.to_rgb();
            [v[0], v[1], v[2], c.alpha]
        }),
        erase: tool == "eraser",
        ..base
    };
    // The path is the exact geometry: stroke smoothing (a hand-drawing aid) would cut its corners.
    brush.smoothing.amount = 0.0;
    match tool {
        "pencil" => brush.hardness = 1.0,
        "brush" | "eraser" => {}
        o => {
            let tools: Vec<&str> = ["brush", "pencil", "eraser"].into_iter().chain(STROKE_PATH_TOOLS.iter().map(|(t, _)| *t)).collect();
            return Err(bad("path.stroke", format!("unknown tool `{o}` ({})", tools.join("|"))));
        }
    }
    crate::brush_cmds::validate_brush(&brush, "path.stroke")?;
    let lines = vector::flatten_path(&path, 0.1);
    let id = layer_id(s, p)?;
    let bg = s.tools.background;
    let dmg = s.edit("Stroke Path", |doc, _| {
        let sel = doc.selection.clone();
        let locks = doc.effective_locks(id);
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        // Pixel and full locks (the layer's own or a parent group's) refuse the stroke, as Fill Path does (#2388).
        if locks.pixels || locks.all {
            return Err(EngineError::Other(format!("Could not complete your request because the layer \"{}\" is locked", l.name)));
        }
        let lock = locks.transparency;
        let surf = l.surface_mut().ok_or_else(|| EngineError::Other("Stroke Path needs a pixel layer".into()))?;
        let mut brush = brush.clone();
        if brush.erase && lock {
            // Locked transparency (e.g. the Background): the Eraser paints the background colour (#76).
            brush.erase = false;
            brush.color = bg;
        }
        let mut dmg = Rect::EMPTY;
        for pl in &lines {
            let pts = stroke_points(&pl.pts, pl.closed, simulate);
            if pts.is_empty() {
                continue;
            }
            let r = photocraft_paint::apply_stroke(surf, &photocraft_paint::Stroke { brush: brush.clone(), points: pts }, sel.as_ref(), lock);
            dmg = if dmg.is_empty() { r } else { dmg.union(&r) };
        }
        Ok(dmg)
    })?;
    Ok(json!({ "damage": [dmg.x0, dmg.y0, dmg.width(), dmg.height()] }))
}

// ---------------------------------------------------------------------------
// Vector masks
// ---------------------------------------------------------------------------

fn vector_mask_info(s: &Session, id: LayerId) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let l = d.doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    Ok(match &l.vector_mask {
        Some(v) => {
            json!({ "layer": id.0, "path": path_json(&v.path), "enabled": v.enabled, "linked": v.linked, "density": v.density * 100.0, "feather": v.feather })
        }
        None => json!({ "layer": id.0, "vectorMask": null }),
    })
}

fn vector_mask_add(s: &mut Session, p: &Value, from_path: bool) -> Result<Value> {
    const CMD: &str = "layer.vectorMask.add";
    let id = layer_id(s, p)?;
    let mut path = match p.get("path") {
        Some(v) => parse_path(v).map_err(|e| bad(CMD, e))?,
        None if from_path || p.get("name").is_some() => {
            let name = p.get("name").and_then(Value::as_str);
            // Never fall back to the layer's own path here.
            if is_work(name) {
                s.active().and_then(|d| d.doc.work_path.clone()).ok_or_else(|| EngineError::Other("no work path".into()))?
            } else {
                resolve_path(s, name)?
            }
        }
        None => Path::default(),
    };
    if p.get("hide").and_then(Value::as_bool).unwrap_or(false) {
        path.inverted = !path.inverted;
    }
    s.edit("Add Vector Mask", |doc, _| {
        let locked = doc.effective_locks(id).all;
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        if matches!(l.content, LayerContent::Shape(_)) {
            return Err(EngineError::Other("a shape layer's path is already its vector mask".into()));
        }
        if locked {
            return Err(EngineError::Other(format!("layer \"{}\" is locked", l.name)));
        }
        l.vector_mask = Some(VectorMask::new(path));
        Ok(())
    })?;
    vector_mask_info(s, id)
}

fn vector_mask_edit(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "layer.vectorMask.edit";
    let id = layer_id(s, p)?;
    let path = p.get("path").map(parse_path).transpose().map_err(|e| bad(CMD, e))?;
    // Same range as a pixel mask's feather (`layer.layerMask.edit`), checked before any edit (#4201).
    let feather = p
        .get("feather")
        .map(|v| {
            v.as_f64()
                .filter(|n| n.is_finite() && (0.0..=1000.0).contains(n))
                .map(|n| n as f32)
                .ok_or_else(|| bad(CMD, "`feather` must be a finite number within 0..1000"))
        })
        .transpose()?;
    s.edit("Edit Vector Mask", |doc, _| {
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        let vm = l.vector_mask.as_mut().ok_or_else(|| EngineError::Other("layer has no vector mask".into()))?;
        if let Some(pth) = path {
            vm.path = pth;
        }
        if let Some(b) = p.get("enabled").and_then(Value::as_bool) {
            vm.enabled = b;
        }
        if let Some(b) = p.get("linked").and_then(Value::as_bool) {
            vm.linked = b;
        }
        if let Some(d) = f64p(p, "density") {
            vm.density = (d / 100.0).clamp(0.0, 1.0) as f32;
        }
        if let Some(f) = feather {
            vm.feather = f;
        }
        if let Some(b) = p.get("invert").and_then(Value::as_bool)
            && b
        {
            vm.path.inverted = !vm.path.inverted;
        }
        Ok(())
    })?;
    vector_mask_info(s, id)
}

fn vector_mask_delete(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_id(s, p)?;
    s.edit("Delete Vector Mask", |doc, _| {
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        l.vector_mask.take().map(|_| ()).ok_or_else(|| EngineError::Other("layer has no vector mask".into()))
    })?;
    Ok(Value::Null)
}

/// Layer › Rasterize › Vector Mask: multiplies the vector mask into the pixel mask. When a feather
/// (either mask's) or the pixel mask's density shapes the rendered edge, the pixel mask instead
/// becomes exactly what the compositor showed, with those settings baked in (#992).
fn vector_mask_rasterize(s: &mut Session, p: &Value) -> Result<Value> {
    use photocraft_compose::masks::{combined_mask, feather_sigma};
    let id = layer_id(s, p)?;
    s.edit("Rasterize Vector Mask", |doc, _| {
        let area = doc.bounds();
        let depth = doc.depth;
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        let soft = l.vector_mask.as_ref().is_some_and(|v| v.enabled)
            && (l.vector_mask.as_ref().is_some_and(|v| feather_sigma(v.feather) > 0.0)
                || l.mask.as_ref().is_some_and(|m| m.enabled && (feather_sigma(m.feather) > 0.0 || m.density < 1.0)));
        let shown = if soft { combined_mask(l, area) } else { None };
        let vm = l.vector_mask.take().ok_or_else(|| EngineError::Other("layer has no vector mask".into()))?;
        let mut mask = l.mask.take().unwrap_or_else(|| {
            let mut m = photocraft_doc::LayerMask::reveal_all();
            m.surface =
                photocraft_raster::Surface::with_default(photocraft_color::PixelFormat::new(photocraft_color::ColorMode::Grayscale, depth, false), &[1.0]);
            m
        });
        if let Some(shown) = shown {
            // Past the canvas the pixel mask keeps its own (density-applied) default, as below.
            let d = mask.surface.default_pixel().first().copied().unwrap_or(1.0);
            let mut surface = photocraft_raster::Surface::with_default(mask.surface.format(), &[1.0 - mask.density * (1.0 - d)]);
            surface.write_region(area, &shown.read_region(area));
            surface.prune();
            (mask.surface, mask.feather, mask.density) = (surface, 0.0, 1.0);
            l.mask = Some(mask);
            return Ok(());
        }
        let vals = vector::vector_mask_values(&vm, area);
        // Outside the canvas the vector mask is 0 unless it is empty/inverted; keep the old default there.
        let old = mask.surface.read_region(area);
        let merged: Vec<f32> = old.iter().zip(&vals).map(|(a, b)| a * b).collect();
        mask.surface.write_region(area, &merged);
        mask.surface.prune();
        l.mask = Some(mask);
        Ok(())
    })?;
    Ok(json!({ "layer": id.0 }))
}

/// `p` with `key` set (for alias commands).
fn with(p: &Value, key: &str, v: Value) -> Value {
    let mut o = if p.is_object() { p.clone() } else { json!({}) };
    o[key] = v;
    o
}

fn toggle_vector_mask(s: &mut Session, p: &Value, key: &str) -> Result<Value> {
    let id = layer_id(s, p)?;
    let cur = {
        let d = s.active().ok_or(EngineError::NoDocument)?;
        let vm = d.doc.layer(id).ok_or(EngineError::NoLayer(id))?.vector_mask.as_ref().ok_or_else(|| EngineError::Other("layer has no vector mask".into()))?;
        if key == "enabled" { vm.enabled } else { vm.linked }
    };
    let v = p.get(key).and_then(Value::as_bool).unwrap_or(!cur);
    vector_mask_edit(s, &json!({ "layer": id.0, key: v }))
}

/// Traces the exact coverage of the shape's path back into plain outlines.
fn merge_components(s: &mut Session, p: &Value) -> Result<Value> {
    let tol = f64p(p, "tolerance").unwrap_or(0.1).clamp(0.05, 10.0);
    if let Some(ids) = combine_targets(s, p, &["layer"])? {
        return merge_shape_layers(s, &ids, None, "Merge Shape Components", Some(tol));
    }
    let id = layer_id(s, p)?;
    with_shape(s, id, "Merge Shape Components", |sh, _| bake_components(sh, tol))?;
    shape_info(s, id)
}

/// Bakes `sh`'s path operations into plain combined outlines, traced from the exact coverage.
fn bake_components(sh: &mut ShapeLayer, tol: f64) -> Result<()> {
    // Trace at 4× resolution so the fitted outline follows the anti-aliased edge closely.
    const K: f64 = 4.0;
    let up = sh.path.transform(&Affine::scale(K));
    let r = vector::fill_rasterizer(&up, vector::DEFAULT_TOLERANCE);
    let Some(b) = r.pixel_bounds() else {
        sh.path.subpaths.clear();
        return Ok(());
    };
    let b = b.inflate(2);
    if u64::from(b.width()) * u64::from(b.height()) > 400_000_000 {
        return Err(EngineError::Other("shape too large to merge".into()));
    }
    let cov = r.render(b);
    let traced = vector::trace::trace_mask(&cov, b, 0.0, tol * K);
    sh.path = traced.transform(&Affine::scale(1.0 / K));
    sh.live = None;
    sh.psd_raw = None;
    Ok(())
}

/// The shape layers Layer › Combine Shapes merges into one: with none of the `narrow` params
/// given and two or more shape layers selected, those layers bottom-to-top (other selected layers
/// are left alone). `None` keeps the single-layer behaviour (acting on one layer's components).
fn combine_targets(s: &Session, p: &Value, narrow: &[&str]) -> Result<Option<Vec<LayerId>>> {
    if narrow.iter().any(|k| p.get(*k).is_some()) {
        return Ok(None);
    }
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let sel = d.selected_layers();
    let ids = crate::layer_multi_cmds::top_level(&d.doc, &sel);
    let ids: Vec<LayerId> = ids.into_iter().filter(|id| d.doc.layer(*id).is_some_and(|l| matches!(l.content, LayerContent::Shape(_)))).collect();
    Ok((ids.len() > 1).then_some(ids))
}

/// Layer › Combine Shapes with several shape layers selected: Photoshop merges them into the
/// top-most one. The bottom layer's components keep their operations; every component of the
/// layers above takes `op` (front shapes subtract, intersect… in stacking order), or, with no
/// `op` (Merge Shape Components), keeps its own with each layer's first component combined, and
/// the result is then baked with tolerance `bake`. The other layers are removed; one history step.
fn merge_shape_layers(s: &mut Session, ids: &[LayerId], op: Option<PathOp>, label: &str, bake: Option<f64>) -> Result<Value> {
    let (&keep, _) = ids.split_last().ok_or_else(|| EngineError::Other("Combine Shapes needs two or more shape layers".into()))?;
    if let Some(st) = s.active() {
        for id in ids {
            let locks = st.doc.effective_locks(*id);
            if locks.all || locks.pixels {
                return Err(EngineError::Other("a selected shape layer is locked".into()));
            }
        }
    }
    s.edit(label, |doc, active| {
        let mut path: Option<Path> = None;
        for id in ids {
            let l = doc.layer(*id).ok_or(EngineError::NoLayer(*id))?;
            let LayerContent::Shape(sh) = &l.content else {
                return Err(EngineError::Other(format!("layer {} is not a shape layer", id.0)));
            };
            let Some(acc) = path.as_mut() else {
                path = Some(sh.path.clone());
                continue;
            };
            for (j, sp) in sh.path.subpaths.iter().enumerate() {
                let mut sp = sp.clone();
                sp.op = match op {
                    Some(op) => op,
                    None if j == 0 => PathOp::Combine,
                    None => sp.op,
                };
                acc.subpaths.push(sp);
            }
        }
        let path = path.ok_or_else(|| EngineError::Other("Combine Shapes needs two or more shape layers".into()))?;
        let snapshot = doc.clone();
        let l = doc.layer_mut(keep).ok_or(EngineError::NoLayer(keep))?;
        let LayerContent::Shape(sh) = &mut l.content else {
            return Err(EngineError::Other(format!("layer {} is not a shape layer", keep.0)));
        };
        let mut merged = ShapeLayer { path, live: None, psd_raw: None, cache: None, ..sh.clone() };
        if let Some(tol) = bake {
            bake_components(&mut merged, tol)?;
        }
        vector::flatten::validate_shape(&merged.path, merged.stroke.as_ref()).map_err(|e| bad(label, e))?;
        refresh_shape(&snapshot, &mut merged);
        *sh = merged;
        for id in ids.iter().filter(|id| **id != keep) {
            doc.remove(*id);
        }
        *active = Some(keep);
        Ok(())
    })?;
    crate::layer_multi_cmds::reselect(s, vec![keep], Some(keep));
    shape_info(s, keep)
}

/// Layer › Combine Shapes › Unite / Subtract Front Shape / Intersect / Exclude: merges the
/// selected shape layers with `op` ([`merge_shape_layers`]), or with one layer (or an explicit
/// `layer`/`subpath`) sets the operation of that layer's components (`shape.edit`'s `op`).
fn combine_shapes(s: &mut Session, p: &Value, op: &str, label: &str) -> Result<Value> {
    if let Some(ids) = combine_targets(s, p, &["layer", "subpath"])? {
        let op = op_from(op).ok_or_else(|| EngineError::Other(format!("unknown op `{op}`")))?;
        return merge_shape_layers(s, &ids, Some(op), label, None);
    }
    shape_edit(s, &with(p, "op", json!(op)))
}

fn selection_to_shape(s: &mut Session, p: &Value) -> Result<Value> {
    let tol = f64p(p, "tolerance").or_else(|| f64p(p, "simplifyTolerance")).unwrap_or(2.0).clamp(0.5, 10.0);
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let area = d.doc.bounds();
    let mask = sel::mask_from_surface(d.doc.selection.as_ref(), area);
    let path = vector::trace::trace_mask(&mask, area, 0.0, tol);
    let mut q = json!({ "kind": "path", "path": path_json(&path) });
    for k in ["fill", "name", "stroke"] {
        if let Some(v) = p.get(k) {
            q[k] = v.clone();
        }
    }
    shape_create(s, &q)
}

// ---------------------------------------------------------------------------
// Specs
// ---------------------------------------------------------------------------

/// Params of the Layer › Combine Shapes operations.
const COMBINE_PARAMS: &str = r##"{"layer":id?,"subpath":index? (default all but the first)} (no layer/subpath and two or more shape layers selected: merges them into the top-most, whose components above the bottom layer's take the operation)"##;

pub(crate) const PATH_FORM: &str = r##"path: {"subpaths":[{"closed":bool=true,"op":"combine|subtract|intersect|exclude","knots":[[x,y] | {"anchor":[x,y],"in":[x,y],"out":[x,y],"smooth":bool}]}],"fillRule":"nonzero|evenodd","inverted":bool}"##;

macro_rules! spec {
    ($id:literal, $label:literal, [$($m:literal),*], $params:expr, $en:expr, $run:expr) => {
        CommandSpec { id: $id, label: $label, menu: &[$($m),*], shortcut: None, params: $params, enabled: $en, run: $run, journal: true }
    };
}

pub(crate) fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

pub fn specs() -> Vec<CommandSpec> {
    let stroke = r##""stroke":{"width":px,"color":"#rrggbb"|fill,"opacity":0..100,"align":"inside|center|outside","cap":"butt|round|square","join":"miter|round|bevel","miterLimit":n,"dashes":[multiples of width],"dashOffset":n}|null"##;
    let fill = r##""fill":"#rrggbb"|[r,g,b,a]|{"gradient":{"stops":[[t,"#hex"]],"angle":deg,"scale":%,"style":"linear|radial|angle|reflected|diamond","reverse":bool}}|{"pattern":name}|null (no fill)"##;
    vec![
        spec!(
            "shape.create",
            "New Shape Layer",
            ["Layer", "New"],
            leak(format!(
                r##"{{"kind":"rect|roundedRect|ellipse|polygon|star|line|path"="rect","rect":[x,y,w,h] (rect/roundedRect/ellipse/polygon/star),"radii":[tl,tr,br,bl]|r (roundedRect=10),"sides":3..100=5,"starRatio":0..1 (star=0.5),"from":[x,y],"to":[x,y],"weight":px=1 (line),"path":{{…}} (kind path),{fill}=foreground,{stroke}=none,"name":str?,"addTo":layerId? + "op":"combine|subtract|intersect|exclude" (append to an existing shape layer)}} → shape.info. {PATH_FORM}"##
            )),
            has_doc,
            shape_create
        ),
        spec!(
            "shape.edit",
            "Edit Shape",
            [],
            leak(format!(
                r##"{{"layer":id?,"path":{{…}}? (replaces; drops live shape),"kind":str?,"rect":[x,y,w,h]?,"radii":[tl,tr,br,bl]|r?,"sides":n?,"starRatio":r?,"from":[x,y]?,"to":[x,y]?,"weight":px? (live-shape params regenerate the path),"move":[dx,dy]?,"transform":[a,b,c,d,e,f]?,"op":"combine|…"? + "subpath":index? (default all but the first),"fillRule":"nonzero|evenodd"?,{fill}?,{stroke}? (merged into the current stroke),"name":str?}} → shape.info"##
            )),
            has_layer,
            shape_edit
        ),
        CommandSpec {
            id: "shape.info",
            label: "Shape Layer Info",
            menu: &[],
            shortcut: None,
            params: r##"{"layer":id?} → {layer,name,kind,path,fill,stroke,live,bounds:[x,y,w,h]}"##,
            enabled: has_layer,
            run: |s, p| {
                let id = layer_id(s, p)?;
                shape_info(s, id)
            },
            journal: false,
        },
        spec!("shape.rasterize", "Rasterize Shape", ["Layer", "Rasterize"], r##"{"layer":id?}"##, has_layer, shape_rasterize),
        CommandSpec {
            id: "path.list",
            label: "List Paths",
            menu: &[],
            shortcut: None,
            params: r##"{} → {paths:[{name,knots,subpaths}],workPath,clippingPath,layerPath (active layer's shape path / vector mask)}"##,
            enabled: has_doc,
            run: |s, _| paths_list(s),
            journal: false,
        },
        CommandSpec {
            id: "path.info",
            label: "Path Info",
            menu: &[],
            shortcut: None,
            params: r##"{"name":str|"work"|"layer"="work"} → {name,path}"##,
            enabled: has_doc,
            run: |s, p| {
                let name = p.get("name").and_then(Value::as_str);
                let path = resolve_path(s, name)?;
                Ok(json!({ "name": name.unwrap_or("work"), "path": path_json(&path) }))
            },
            journal: false,
        },
        spec!(
            "path.set",
            "Set Path",
            [],
            leak(format!(
                r##"{{"name":str|"work"="work","path":{{…}},"op":"combine|subtract|intersect|exclude"? (append to the existing path with this op instead of replacing)}}. {PATH_FORM}"##
            )),
            has_doc,
            path_set
        ),
        spec!("path.delete", "Delete Path", [], r##"{"name":str|"work"="work"}"##, has_doc, path_delete),
        spec!(
            "path.transform",
            "Free Transform Path",
            [],
            r##"{"name":str|"work"|"layer"="work","layer":id? (with layer target),"matrix":[a,b,c,d,e,f] | "quad":[[x,y]×4] (Distort/Perspective: where the frame's corners go, clockwise from top-left) | "style"|"mesh"|"warp":… (Warp, as edit.transform.warp) | "translateX":px?,"translateY":px?,"scaleX":factor?,"scaleY":factor?,"angle":degrees?,"rect":[x0,y0,x1,y1]? (the quad's or warp's frame; default = path control bounds)} (numeric transforms pivot on path bounds center)"##,
            has_doc,
            path_transform
        ),
        spec!(
            "path.clippingPath.set",
            "Clipping Path",
            [],
            r##"{"name":savedPathName,"flatness":0..100=0} (PSD export clipping path)"##,
            has_doc,
            clipping_path_set
        ),
        spec!("path.clippingPath.clear", "Clear Clipping Path", [], r##"{}"##, has_doc, clipping_path_clear),
        spec!("path.style.copyFill", "Copy Fill", [], r##"{} (copies active shape layer fill)"##, has_shape_fill, copy_shape_fill),
        spec!("path.style.copyStroke", "Copy Complete Stroke", [], r##"{} (copies active shape layer stroke)"##, has_shape_stroke, copy_shape_stroke),
        spec!(
            "path.style.pasteFill",
            "Paste Fill",
            [],
            r##"{"layer":id?} (pastes copied fill onto active shape layer)"##,
            can_paste_shape_fill,
            paste_shape_fill
        ),
        spec!(
            "path.style.pasteStroke",
            "Paste Complete Stroke",
            [],
            r##"{"layer":id?} (pastes copied stroke onto active shape layer)"##,
            can_paste_shape_stroke,
            paste_shape_stroke
        ),
        spec!("path.rename", "Rename Path", [], r##"{"name":str|"work"="work","to":str} (renaming the work path saves it)"##, has_doc, path_rename),
        // ⌘↩ / Ctrl+Enter, as in Photoshop (the UI loads the path selected in the Paths panel, #306).
        CommandSpec {
            id: "path.toSelection",
            label: "Make Selection from Path",
            menu: &[],
            shortcut: Some("Cmd+Enter"),
            params: r##"{"name":str|"work"|"layer"="work","feather":px=0,"antiAlias":bool=true,"mode":"replace|add|subtract|intersect"="replace"}"##,
            enabled: has_doc,
            run: path_to_selection,
            journal: true,
        },
        spec!("select.toWorkPath", "Make Work Path", [], r##"{"tolerance":0.5..10 px=2} → {subpaths,knots}"##, has_selection, select_to_work_path),
        spec!(
            "path.fill",
            "Fill Path",
            [],
            r##"{"name":str|"work"|"layer"="work","layer":id? (pixel layer, default active),"color":"#rrggbb"=foreground,"opacity":0..100=100,"mode":"normal|multiply|…"="normal","feather":px=0,"antiAlias":bool=true}"##,
            has_layer,
            path_fill
        ),
        spec!(
            "path.stroke",
            "Stroke Path",
            [],
            r##"{"name":str|"work"|"layer"="work","layer":id? (pixel layer),"tool":"brush|pencil|eraser|backgroundEraser|cloneStamp|patternStamp|healingBrush|spotHealingBrush|historyBrush|blur|sharpen|smudge|dodge|burn|sponge|colorReplacement|mixerBrush"="brush","simulatePressure":bool=false (taper both ends when the brush follows pen pressure),"size":0.5..5000 px?,"hardness":0..1?,"opacity":0..100?,"color":"#rrggbb"=foreground} (current brush settings otherwise; tools past the Eraser run their paint.* command along the path and take its params)"##,
            has_layer,
            path_stroke
        ),
        spec!(
            "layer.vectorMask.add",
            "Add Vector Mask",
            ["Layer", "Vector Mask"],
            r##"{"layer":id?,"path":{…}? | "name":str|"work"? (copy a document path),"hide":bool=false (Hide All = inverted)} → vector mask info. Without a path the mask reveals all."##,
            has_layer,
            |s, p| vector_mask_add(s, p, false)
        ),
        spec!(
            "layer.vectorMask.fromPath",
            "Vector Mask from Current Path",
            ["Layer", "Vector Mask"],
            r##"{"layer":id?,"name":str|"work"="work"}"##,
            has_layer,
            |s, p| vector_mask_add(s, p, true)
        ),
        spec!(
            "layer.vectorMask.edit",
            "Edit Vector Mask",
            [],
            leak(format!(
                r##"{{"layer":id?,"path":{{…}}?,"enabled":bool?,"linked":bool?,"density":0..100?,"feather":0..1000? (pixels),"invert":true?}}. {PATH_FORM}"##
            )),
            has_layer,
            vector_mask_edit
        ),
        spec!("layer.vectorMask.delete", "Delete Vector Mask", ["Layer", "Vector Mask"], r##"{"layer":id?}"##, has_layer, vector_mask_delete),
        spec!(
            "layer.rasterize.vectorMask",
            "Rasterize Vector Mask",
            [],
            r##"{"layer":id?} (multiplied into the layer mask)"##,
            has_layer,
            vector_mask_rasterize
        ),
        // Menu ids of Photoshop's Layer › Vector Mask / Rasterize / Combine Shapes.
        spec!("layer.vectorMask.revealAll", "Vector Mask: Reveal All", [], r##"{"layer":id?}"##, has_layer, |s, p| vector_mask_add(
            s,
            &with(p, "hide", json!(false)),
            false
        )),
        spec!("layer.vectorMask.hideAll", "Vector Mask: Hide All", [], r##"{"layer":id?}"##, has_layer, |s, p| vector_mask_add(
            s,
            &with(p, "hide", json!(true)),
            false
        )),
        spec!("layer.vectorMask.currentPath", "Vector Mask: Current Path", [], r##"{"layer":id?,"name":str|"work"="work"}"##, has_layer, |s, p| {
            vector_mask_add(s, p, true)
        }),
        spec!("layer.vectorMask.enabled", "Enable Vector Mask", [], r##"{"layer":id?,"enabled":bool? (default: toggle)}"##, has_layer, |s, p| {
            toggle_vector_mask(s, p, "enabled")
        }),
        spec!("layer.vectorMask.linked", "Link Vector Mask", [], r##"{"layer":id?,"linked":bool? (default: toggle)}"##, has_layer, |s, p| toggle_vector_mask(
            s, p, "linked"
        )),
        // Without `layer`, every selected shape layer (the shared Rasterize logic calls back here with one).
        spec!("layer.rasterize.shape", "Rasterize Shape", [], r##"{"layer":id?} (no layer: every selected shape layer)"##, has_layer, |s, p| {
            if p.get("layer").is_some() { shape_rasterize(s, p) } else { crate::extra_cmds::rasterize(s, p, Some("shape")) }
        }),
        // With two or more shape layers selected (and no `layer`/`subpath`), these merge them into
        // the top-most one, as in Photoshop; with one, they set the operation of its components.
        spec!("layer.combineShapes.unite", "Unite Shapes", [], COMBINE_PARAMS, has_layer, |s, p| combine_shapes(s, p, "combine", "Unite Shapes")),
        spec!("layer.combineShapes.subtractFrontShape", "Subtract Front Shape", [], COMBINE_PARAMS, has_layer, |s, p| combine_shapes(
            s,
            p,
            "subtract",
            "Subtract Front Shape"
        )),
        spec!("layer.combineShapes.intersectShapeAreas", "Intersect Shape Areas", [], COMBINE_PARAMS, has_layer, |s, p| combine_shapes(
            s,
            p,
            "intersect",
            "Intersect Shape Areas"
        )),
        spec!("layer.combineShapes.excludeOverlappingShapes", "Exclude Overlapping Shapes", [], COMBINE_PARAMS, has_layer, |s, p| {
            combine_shapes(s, p, "exclude", "Exclude Overlapping Shapes")
        }),
        spec!(
            "layer.combineShapes.mergeShapeComponents",
            "Merge Shape Components",
            [],
            r##"{"layer":id?,"tolerance":px=0.1} (bakes the path operations into plain combined outlines, traced from the exact coverage; no layer and several shape layers selected: merges them into the top-most first)"##,
            has_layer,
            merge_components
        ),
        spec!("select.convertToShape", "Convert to Shape", [], r##"{"tolerance":0.5..10 px=2} → shape.info"##, has_selection, selection_to_shape),
        CommandSpec {
            id: "layer.vectorMask.info",
            label: "Vector Mask Info",
            menu: &[],
            shortcut: None,
            params: r##"{"layer":id?} → {layer,path,enabled,linked,density,feather}"##,
            enabled: has_layer,
            run: |s, p| {
                let id = layer_id(s, p)?;
                vector_mask_info(s, id)
            },
            journal: false,
        },
    ]
}

#[cfg(test)]
mod tests;
