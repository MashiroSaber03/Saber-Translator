//! The Type menu: anti-aliasing, orientation, point ⇄ paragraph conversion, Create Work Path,
//! Convert to Shape, Rasterize, Warp Text, the OpenType toggles, Paste Lorem Ipsum, Update All
//! Text Layers, missing-font replacement and the default type styles.
//!
//! Everything edits the typed [`TextLayer`] model and re-renders through `type_cmds::refresh`,
//! so the PSD `TySh` data (and with it warp, anti-aliasing and the OpenType flags) is regenerated
//! and the layer saves as editable text.

use photocraft_doc::text::{AntiAlias, FontFeature, Orientation, TextAlign, TextShape, TextWarp};
use photocraft_doc::vector::{Knot, Path, PathOp, Subpath};
use photocraft_doc::{Affine, Document, Fill, Layer, LayerContent, LayerId, ShapeLayer, TextLayer};
use photocraft_geom::Point;
use photocraft_text::render::PathEl;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, layer_param};
use crate::type_cmds::{
    check_text_unlocked, format_targets, has_format_target, is_auto_named, layer_name, refresh, replace_text, style_paragraphs, style_range, with_text_targets,
};
use crate::{EngineError, Result, Session};

/// Photoshop's Paste Lorem Ipsum filler.
pub const LOREM_IPSUM: &str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";

// ---------- predicates ----------

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

fn active_text(s: &Session) -> std::result::Result<&TextLayer, String> {
    let d = s.active().ok_or("no document open")?;
    match d.active_layer.and_then(|id| d.doc.layer(id)).map(|l| &l.content) {
        Some(LayerContent::Text(t)) => Ok(t),
        _ => Err("the active layer is not a type layer".into()),
    }
}

fn has_text(s: &Session) -> std::result::Result<(), String> {
    active_text(s).map(|_| ())
}

fn has_point_text(s: &Session) -> std::result::Result<(), String> {
    match active_text(s)?.shape {
        TextShape::Point => Ok(()),
        TextShape::Box { .. } => Err("the type layer is already paragraph text".into()),
        TextShape::Path { .. } => Err("detach the type layer from its path first".into()),
    }
}

fn has_box_text(s: &Session) -> std::result::Result<(), String> {
    match active_text(s)?.shape {
        TextShape::Box { .. } => Ok(()),
        TextShape::Point => Err("the type layer is already point text".into()),
        TextShape::Path { .. } => Err("the type layer is path text".into()),
    }
}

fn any_text(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    if d.doc.walk().iter().any(|(_, _, l)| matches!(l.content, LayerContent::Text(_))) { Ok(()) } else { Err("the document has no type layers".into()) }
}

fn has_defaults(s: &Session) -> std::result::Result<(), String> {
    has_format_target(s)?;
    s.type_defaults.as_ref().map(|_| ()).ok_or_else(|| "no default type styles have been saved".into())
}

// ---------- helpers ----------

/// Edits a type layer as one history step and re-renders it.
fn with_text<R>(s: &mut Session, p: &Value, label: &str, f: impl FnOnce(&mut TextLayer, &Document) -> Result<R>) -> Result<R> {
    let id = layer_param(s, p)?;
    check_text_unlocked(&s.active().ok_or(EngineError::NoDocument)?.doc, id)?;
    s.edit(label, |doc, _| {
        let snapshot = doc.clone();
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        let auto_named = is_auto_named(l);
        let LayerContent::Text(t) = &mut l.content else {
            return Err(EngineError::Other(format!("layer {} is {} {} layer, not a type layer", id.0, l.content.article(), l.content.kind_name())));
        };
        let r = f(t, &snapshot)?;
        refresh(&snapshot, t);
        let name = auto_named.then(|| layer_name(&t.text));
        if let Some(n) = name {
            l.name = n;
        }
        Ok(r)
    })
}

fn layout(doc: &Document, t: &TextLayer) -> photocraft_text::TextLayout {
    photocraft_text::shared().lock().unwrap_or_else(|e| e.into_inner()).layout(t, doc.resolution_dpi)
}

/// `m ∘ translate(tx, ty)`.
fn then_translate(m: &Affine, tx: f64, ty: f64) -> Affine {
    let [a, b, c, d, e, f] = m.m;
    Affine { m: [a, b, c, d, a * tx + c * ty + e, b * tx + d * ty + f] }
}

fn first_align(t: &TextLayer) -> TextAlign {
    t.paragraph_runs().first().map(|p| p.style.align).unwrap_or_default()
}

fn char_index(text: &str, byte: usize) -> usize {
    text[..byte.min(text.len())].chars().count()
}

fn byte_index(text: &str, ci: usize) -> usize {
    text.char_indices().nth(ci).map_or(text.len(), |(b, _)| b)
}

// ---------- anti-aliasing / orientation ----------

pub fn aa_name(a: AntiAlias) -> &'static str {
    match a {
        AntiAlias::None => "none",
        AntiAlias::Sharp => "sharp",
        AntiAlias::Crisp => "crisp",
        AntiAlias::Strong => "strong",
        AntiAlias::Smooth => "smooth",
        AntiAlias::Windows => "windows",
        AntiAlias::WindowsLcd => "windowsLcd",
    }
}

fn set_aa(s: &mut Session, p: &Value, aa: AntiAlias) -> Result<Value> {
    with_text_targets(s, p, "Anti-Alias", |t, _| {
        t.antialias = aa;
        Ok(())
    })?;
    Ok(json!({"antialias": aa_name(aa)}))
}

fn set_orientation(s: &mut Session, p: &Value, o: Orientation) -> Result<Value> {
    with_text_targets(s, p, "Change Text Orientation", |t, _| {
        t.orientation = o;
        Ok(())
    })?;
    Ok(json!({"orientation": if o == Orientation::Vertical { "vertical" } else { "horizontal" }}))
}

// ---------- point ⇄ paragraph ----------

fn to_paragraph(s: &mut Session, p: &Value) -> Result<Value> {
    let r = with_text(s, p, "Convert to Paragraph Text", |t, doc| {
        let l = layout(doc, t);
        let size_px = t.char_runs().iter().map(|r| r.style.size_pt).fold(t.size_pt, f32::max) * doc.resolution_dpi / 72.0;
        let ps = t.paragraph_runs().first().map(|p| p.style.clone()).unwrap_or_default();
        let indents = (ps.start_indent_pt + ps.end_indent_pt).max(0.0) * doc.resolution_dpi / 72.0;
        if l.vertical {
            // Line space: u along the columns, v across them (first column centred on v = 0).
            let [u0, v0, u1, v1] = l.line_bounds().unwrap_or([0.0, -size_px / 2.0, size_px, size_px / 2.0]);
            let height = (u1 - u0).max(1.0) + indents + (size_px * 0.5).max(4.0);
            let width = (v1 - v0).max(1.0) + size_px * 0.5;
            let y = match ps.align {
                TextAlign::Center | TextAlign::JustifyCenter => -height / 2.0,
                TextAlign::Right | TextAlign::JustifyRight => -height,
                _ => 0.0,
            };
            // The box's right edge is the first column's right edge, so the columns stay put.
            let x = -v0 - width;
            t.shape = TextShape::Box { x: 0.0, y: 0.0, width, height };
            t.transform = then_translate(&t.transform, f64::from(x), f64::from(y));
            return Ok(json!([x, y, width, height]));
        }
        let [x0, y0, x1, y1] = l.bounds().unwrap_or([0.0, -t.size_pt, t.size_pt, 0.0]);
        // A little slack so no line re-wraps; the box keeps the text where it was.
        let width = (x1 - x0).max(1.0) + indents + (size_px * 0.5).max(4.0);
        let height = (y1 - y0).max(1.0) + size_px * 0.5;
        let x = match ps.align {
            TextAlign::Center | TextAlign::JustifyCenter => -width / 2.0,
            TextAlign::Right | TextAlign::JustifyRight => -width,
            _ => 0.0,
        };
        let mut probe = t.clone();
        probe.shape = TextShape::Box { x, y: y0, width, height };
        // Box text puts the first baseline at the box top plus the ascender height: measure it and
        // shift the box so that baseline stays at the point-text anchor (y = 0).
        let b0 = layout(doc, &probe).lines.first().map_or(0.0, |ln| ln.baseline);
        let y = y0 - b0;
        t.shape = TextShape::Box { x: 0.0, y: 0.0, width, height };
        t.transform = then_translate(&t.transform, f64::from(x), f64::from(y));
        Ok(json!([x, y, width, height]))
    })?;
    Ok(json!({"box": r}))
}

fn to_point(s: &mut Session, p: &Value) -> Result<Value> {
    let r = with_text(s, p, "Convert to Point Text", |t, doc| {
        let TextShape::Box { x, y, width, height } = t.shape else { return Err(EngineError::Other("already point text".into())) };
        let l = layout(doc, t);
        // Vertical type: columns run along the box height.
        let (x, width) = if l.vertical { (y, height) } else { (x, width) };
        let baseline = l.lines.first().map_or(0.0, |ln| ln.baseline);
        // Photoshop inserts a hard return at every soft line break.
        let mut breaks: Vec<usize> = l.lines.windows(2).filter(|w| w[0].paragraph == w[1].paragraph).map(|w| w[1].range.start).collect();
        // Lines a box overflows are dropped from the layout; keep their text after a break.
        breaks.sort_unstable();
        breaks.dedup();
        for at in breaks.iter().rev() {
            replace_text(t, *at, *at, "\n");
        }
        let ax = match first_align(t) {
            TextAlign::Center | TextAlign::JustifyCenter => x + width / 2.0,
            TextAlign::Right | TextAlign::JustifyRight => x + width,
            _ => x,
        };
        t.shape = TextShape::Point;
        // `baseline` is in line space (it already includes the box edge).
        let (tx, ty) = l.to_text(ax, baseline);
        t.transform = then_translate(&t.transform, f64::from(tx), f64::from(ty));
        Ok(breaks.len())
    })?;
    Ok(json!({"inserted": r}))
}

// ---------- outlines → paths ----------

fn signed_area(knots: &[Knot]) -> f64 {
    let n = knots.len();
    (0..n)
        .map(|i| {
            let (a, b) = (knots[i].anchor, knots[(i + 1) % n].anchor);
            a.x * b.y - b.x * a.y
        })
        .sum::<f64>()
        / 2.0
}

/// One glyph's outline elements as closed subpaths.
fn contours(els: &[PathEl]) -> Vec<Vec<Knot>> {
    let mut out: Vec<Vec<Knot>> = Vec::new();
    let mut cur: Vec<Knot> = Vec::new();
    let p = |v: [f64; 2]| Point::new(v[0], v[1]);
    let finish = |cur: &mut Vec<Knot>, out: &mut Vec<Vec<Knot>>| {
        if cur.len() >= 2 {
            // The closing point duplicates the first anchor: fold its incoming handle in.
            let (f, l) = (cur[0], cur[cur.len() - 1]);
            if (f.anchor.x - l.anchor.x).abs() < 1e-9 && (f.anchor.y - l.anchor.y).abs() < 1e-9 {
                cur[0].in_ctrl = l.in_ctrl;
                cur.pop();
            }
        }
        if cur.len() >= 2 {
            out.push(std::mem::take(cur));
        } else {
            cur.clear();
        }
    };
    for el in els {
        match *el {
            PathEl::MoveTo(a) => {
                finish(&mut cur, &mut out);
                cur.push(Knot::corner(a[0], a[1]));
            }
            PathEl::LineTo(a) => cur.push(Knot::corner(a[0], a[1])),
            PathEl::QuadTo(c, a) => {
                // Exact quadratic → cubic elevation.
                let Some(last) = cur.last_mut() else { continue };
                let s = last.anchor;
                last.out_ctrl = Point::new(s.x + 2.0 / 3.0 * (c[0] - s.x), s.y + 2.0 / 3.0 * (c[1] - s.y));
                let c2 = Point::new(a[0] + 2.0 / 3.0 * (c[0] - a[0]), a[1] + 2.0 / 3.0 * (c[1] - a[1]));
                cur.push(Knot { anchor: p(a), in_ctrl: c2, out_ctrl: p(a), smooth: false });
            }
            PathEl::CurveTo(c1, c2, a) => {
                let Some(last) = cur.last_mut() else { continue };
                last.out_ctrl = p(c1);
                cur.push(Knot { anchor: p(a), in_ctrl: p(c2), out_ctrl: p(a), smooth: false });
            }
            PathEl::Close => finish(&mut cur, &mut out),
        }
    }
    finish(&mut cur, &mut out);
    out
}

/// The type layer's glyph outlines as a document-space path. Within each glyph, contours wound
/// like its largest contour combine and the others (counters such as the hole in "o") subtract,
/// which reproduces the nonzero fill under Photoshop's per-subpath path operations.
pub fn text_path(doc: &Document, t: &TextLayer) -> Path {
    let l = layout(doc, t);
    let warp = photocraft_text::render::layout_deform(&l, t);
    let mut subpaths = Vec::new();
    for glyph in photocraft_text::render::outlines(&l, &t.transform, warp.as_ref()) {
        let cs = contours(&glyph);
        let Some(outer) = cs.iter().map(|k| signed_area(k)).max_by(|a, b| a.abs().total_cmp(&b.abs())) else { continue };
        let (fills, holes): (Vec<_>, Vec<_>) = cs.into_iter().partition(|k| signed_area(k).signum() == outer.signum());
        subpaths.extend(fills.into_iter().map(|knots| Subpath { closed: true, knots, op: PathOp::Combine }));
        subpaths.extend(holes.into_iter().map(|knots| Subpath { closed: true, knots, op: PathOp::Subtract }));
    }
    Path::new(subpaths)
}

fn create_work_path(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let Some(LayerContent::Text(t)) = d.doc.layer(id).map(|l| &l.content) else { return Err(EngineError::Other("not a type layer".into())) };
    let path = text_path(&d.doc, t);
    let n = path.subpaths.len();
    s.edit("Create Work Path", |doc, _| {
        doc.work_path = Some(path);
        Ok(())
    })?;
    Ok(json!({"subpaths": n}))
}

fn convert_to_shape(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    let n = s.edit("Convert to Shape", |doc, _| {
        let snapshot = doc.clone();
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        let LayerContent::Text(t) = &l.content else { return Err(EngineError::Other("not a type layer".into())) };
        let color = t.char_runs().first().map_or(t.color, |r| r.style.color);
        let mut sh = ShapeLayer { path: text_path(&snapshot, t), fill: Some(Fill::Solid(color)), ..Default::default() };
        crate::vector_cmds::refresh_shape(&snapshot, &mut sh);
        let n = sh.path.subpaths.len();
        l.content = LayerContent::Shape(sh);
        l.psd_blocks.retain(|(k, _)| k != b"TySh");
        Ok(n)
    })?;
    Ok(json!({"layer": id.0, "subpaths": n}))
}

// ---------- type on a path ----------

/// Largest anchor count of a path that type can follow (a hostile document cannot make the
/// layout walk millions of segments).
const MAX_TYPE_PATH_KNOTS: usize = 100_000;

/// The subpath `type.attachToPath` follows, in document space: the work path (default) or a
/// shape layer's path (`"shape":id`), subpath `"subpath"` (default 0).
fn path_source(doc: &Document, p: &Value) -> Result<Subpath> {
    let bad = |m: &str| EngineError::BadParams { cmd: "type.attachToPath".into(), msg: m.into() };
    let path = match p.get("shape") {
        None | Some(Value::Null) => doc.work_path.as_ref().ok_or_else(|| bad("the document has no work path; draw a path or pass \"shape\":layerId"))?,
        Some(v) => {
            let id = v.as_u64().ok_or_else(|| bad("\"shape\" must be a layer id"))?;
            match doc.layer(LayerId(id)).map(|l| &l.content) {
                Some(LayerContent::Shape(sh)) => &sh.path,
                Some(_) => return Err(bad("\"shape\" is not a shape layer")),
                None => return Err(EngineError::NoLayer(LayerId(id))),
            }
        }
    };
    let index = match p.get("subpath") {
        None | Some(Value::Null) => 0,
        Some(v) => v.as_u64().and_then(|n| usize::try_from(n).ok()).ok_or_else(|| bad("\"subpath\" must be a non-negative index"))?,
    };
    let sub = path.subpaths.get(index).ok_or_else(|| bad("the path has no such subpath"))?;
    if sub.knots.len() < 2 {
        return Err(bad("the subpath needs at least two anchor points"));
    }
    if sub.knots.len() > MAX_TYPE_PATH_KNOTS {
        return Err(bad("the subpath has too many anchor points"));
    }
    Ok(sub.clone())
}

fn attach_to_path(s: &mut Session, p: &Value) -> Result<Value> {
    let bad = |m: &str| EngineError::BadParams { cmd: "type.attachToPath".into(), msg: m.into() };
    let start = match p.get("start") {
        None | Some(Value::Null) => 0.0,
        Some(v) => v.as_f64().filter(|n| n.is_finite()).ok_or_else(|| bad("\"start\" must be a finite number of pixels"))? as f32,
    };
    let flip = match p.get("flip") {
        None | Some(Value::Null) => false,
        Some(v) => v.as_bool().ok_or_else(|| bad("\"flip\" must be true or false"))?,
    };
    let id = layer_param(s, p)?;
    let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
    if doc.effective_locks(id).all {
        return Err(EngineError::Other("the type layer is locked".into()));
    }
    let sub = path_source(doc, p)?;
    with_text(s, p, "Type on a Path", |t, _| {
        // The path lives in document space; the type layer keeps it in its own text space, so
        // the type lands on the path where it is drawn.
        let inv = t.transform.inverse().ok_or_else(|| EngineError::Other("the type layer's transform cannot be inverted".into()))?;
        let map = |q: Point| inv.apply(q);
        let knots = sub.knots.iter().map(|k| Knot { anchor: map(k.anchor), in_ctrl: map(k.in_ctrl), out_ctrl: map(k.out_ctrl), smooth: k.smooth }).collect();
        let path = Subpath { closed: sub.closed, knots, op: PathOp::Combine };
        if !path.knots.iter().all(|k| [k.anchor, k.in_ctrl, k.out_ctrl].iter().all(|q| q.x.is_finite() && q.y.is_finite())) {
            return Err(EngineError::Other("the path cannot be mapped into the type layer".into()));
        }
        t.shape = TextShape::Path { path, start, flip };
        Ok(())
    })?;
    Ok(json!({"layer": id.0, "start": start, "flip": flip}))
}

fn is_path_text(s: &Session) -> std::result::Result<(), String> {
    match active_text(s)?.shape {
        TextShape::Path { .. } => Ok(()),
        _ => Err("the type layer is not on a path".into()),
    }
}

fn detach_from_path(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    with_text(s, p, "Release Type from Path", |t, _| {
        if !matches!(t.shape, TextShape::Path { .. }) {
            return Err(EngineError::Other("the type layer is not on a path".into()));
        }
        // Point text starts where the type started on the path.
        let origin = photocraft_text::warp::Warp::on_path(&t.shape).map_or((0.0, 0.0), |w| w.apply(0.0, 0.0));
        t.shape = TextShape::Point;
        t.transform = then_translate(&t.transform, origin.0, origin.1);
        Ok(())
    })?;
    Ok(json!({"layer": id.0}))
}

// ---------- warp ----------

fn warp_text(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "type.warpText";
    let style = p.get("style").and_then(Value::as_str).unwrap_or("arc");
    let psd =
        photocraft_text::warp::psd_style(style).ok_or_else(|| EngineError::BadParams { cmd: cmd.into(), msg: format!("unknown warp style \"{style}\"") })?;
    let f = |k: &str, d: f64| p.get(k).and_then(Value::as_f64).unwrap_or(d).clamp(-100.0, 100.0) as f32;
    let warp = (psd != "warpNone").then(|| TextWarp {
        style: psd.into(),
        value: f("bend", 50.0),
        horizontal_distortion: f("horizontalDistortion", 0.0),
        vertical_distortion: f("verticalDistortion", 0.0),
        horizontal: p.get("orientation").and_then(Value::as_str) != Some("vertical"),
    });
    let out = json!({"warp": warp});
    with_text_targets(s, p, "Warp Text", |t, _| {
        t.warp = warp.clone();
        Ok(())
    })?;
    Ok(out)
}

// ---------- OpenType toggles ----------

/// Is the OpenType toggle `tag` on for a character style? (`liga`/`dlig` have dedicated fields.)
pub fn feature_on(st: &photocraft_doc::text::CharStyle, tag: &str) -> bool {
    match tag {
        "liga" => st.ligatures,
        "dlig" => st.discretionary_ligatures,
        _ => st.features.iter().any(|f| f.tag == tag && f.value > 0),
    }
}

fn set_feature(st: &mut photocraft_doc::text::CharStyle, tag: &str, on: bool) {
    match tag {
        "liga" => st.ligatures = on,
        "dlig" => st.discretionary_ligatures = on,
        _ => {
            st.features.retain(|f| f.tag != tag);
            if on {
                st.features.push(FontFeature { tag: tag.into(), value: 1 });
            }
        }
    }
}

fn toggle_opentype(s: &mut Session, p: &Value, tag: &'static str) -> Result<Value> {
    let ids = format_targets(s, p)?;
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let representative = st.active_layer.filter(|id| ids.contains(id)).or_else(|| ids.first().copied());
    let cur = representative
        .and_then(|id| st.doc.layer(id))
        .is_some_and(|l| matches!(&l.content, LayerContent::Text(t) if t.char_runs().first().is_some_and(|r| feature_on(&r.style, tag))));
    let on = p.get("on").and_then(Value::as_bool).unwrap_or(!cur);
    with_text_targets(s, p, "Change OpenType Feature", |t, (a, b)| {
        style_range(t, a.min(b), a.max(b), &|st| set_feature(st, tag, on));
        Ok(())
    })?;
    Ok(json!({"feature": tag, "on": on}))
}

// ---------- lorem ipsum / update / fonts / defaults ----------

fn paste_lorem(s: &mut Session, p: &Value) -> Result<Value> {
    if has_text(s).is_ok() && p.get("new").and_then(Value::as_bool) != Some(true) {
        let caret = with_text(s, p, "Paste Lorem Ipsum", |t, _| {
            let at = p.get("at").and_then(Value::as_u64).map_or(t.text.len(), |c| byte_index(&t.text, c as usize));
            replace_text(t, at, at, LOREM_IPSUM);
            Ok(char_index(&t.text, at + LOREM_IPSUM.len()))
        })?;
        return Ok(json!({"caret": caret}));
    }
    // No type layer active: a paragraph text box over most of the canvas.
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let (w, h) = (d.doc.size.width as f64, d.doc.size.height as f64);
    let size = p.get("size").and_then(Value::as_f64).unwrap_or((h / 20.0 * 72.0 / f64::from(d.doc.resolution_dpi)).clamp(6.0, 72.0));
    let r = s.execute("type.create", json!({"box": [w * 0.1, h * 0.1, w * 0.8, h * 0.8], "text": LOREM_IPSUM, "size": size}))?;
    Ok(json!({"layer": r["layer"]}))
}

fn update_all(s: &mut Session) -> Result<Value> {
    let mut n = 0;
    s.edit("Update All Text Layers", |doc, _| {
        let snapshot = doc.clone();
        let ids: Vec<LayerId> = snapshot.walk().into_iter().filter(|(_, _, l)| matches!(l.content, LayerContent::Text(_))).map(|(_, _, l)| l.id).collect();
        for id in ids {
            if let Some(Layer { content: LayerContent::Text(t), .. }) = doc.layer_mut(id) {
                refresh(&snapshot, t);
                n += 1;
            }
        }
        Ok(())
    })?;
    Ok(json!({"updated": n}))
}

fn style_font_missing(fonts: &mut photocraft_text::FontDb, style: &photocraft_doc::text::CharStyle) -> bool {
    if style.font_family.is_empty() || fonts.has_family(&style.font_family) {
        return false;
    }
    // Match layout's choice: an exact PostScript face wins over the import-time family guess;
    // a heuristic family is usable only when the stored family is unavailable.
    if let Some(ps) = &style.postscript_name {
        let resolved = fonts.resolve_postscript(ps);
        if resolved.exact || fonts.has_family(&resolved.family) {
            return false;
        }
    }
    true
}

/// Font families used by type layers that the font database cannot supply.
pub fn missing_fonts(doc: &Document) -> Vec<String> {
    let mut engine = photocraft_text::shared().lock().unwrap_or_else(|e| e.into_inner());
    let mut out: Vec<String> = Vec::new();
    for (_, _, l) in doc.walk() {
        if let LayerContent::Text(t) = &l.content {
            for r in t.char_runs() {
                if style_font_missing(&mut engine.fonts, &r.style) && !out.contains(&r.style.font_family) {
                    out.push(r.style.font_family);
                }
            }
        }
    }
    out
}

/// Font families used by a type layer that the font database cannot supply.
pub fn missing_fonts_in_layer(t: &TextLayer) -> Vec<String> {
    let mut engine = photocraft_text::shared().lock().unwrap_or_else(|e| e.into_inner());
    let mut out: Vec<String> = Vec::new();
    for r in t.char_runs() {
        if style_font_missing(&mut engine.fonts, &r.style) && !out.contains(&r.style.font_family) {
            out.push(r.style.font_family);
        }
    }
    out
}

/// Warnings for missing font families in `doc`.
pub fn missing_font_warnings(doc: &Document) -> Vec<String> {
    fonts_to_warnings(missing_fonts(doc))
}

/// Warnings for missing font families in `layer`.
pub fn missing_font_warnings_for_layer(t: &TextLayer) -> Vec<String> {
    fonts_to_warnings(missing_fonts_in_layer(t))
}

fn fonts_to_warnings(fonts: Vec<String>) -> Vec<String> {
    fonts.into_iter().map(|font| format!("font '{font}' is not installed; text may render using a fallback face")).collect()
}

/// Replace missing families per `map` (missing → installed; unmapped ones → the default family).
fn replace_fonts(s: &mut Session, map: &serde_json::Map<String, Value>, all: bool) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let missing = missing_fonts(&d.doc);
    let pick = |f: &str| -> Option<String> {
        if !missing.iter().any(|m| m == f) {
            return None;
        }
        map.get(f).and_then(Value::as_str).map(str::to_string).or_else(|| all.then(|| photocraft_text::fonts::DEFAULT_FAMILY.to_string()))
    };
    let ids: Vec<LayerId> = d.doc.walk().into_iter().filter(|(_, _, l)| matches!(l.content, LayerContent::Text(_))).map(|(_, _, l)| l.id).collect();
    if missing.is_empty() {
        return Ok(json!({"missing": [], "replaced": 0}));
    }
    let mut n = 0;
    s.edit("Replace Missing Fonts", |doc, _| {
        let snapshot = doc.clone();
        for id in &ids {
            if let Some(Layer { content: LayerContent::Text(t), .. }) = doc.layer_mut(*id) {
                let mut changed = false;
                let runs = t.char_runs();
                let mut at = 0;
                for r in runs {
                    let missing_run = {
                        let mut engine = photocraft_text::shared().lock().unwrap_or_else(|e| e.into_inner());
                        style_font_missing(&mut engine.fonts, &r.style)
                    };
                    if missing_run && let Some(to) = pick(&r.style.font_family) {
                        style_range(t, at, at + r.len, &|st| {
                            st.font_family = to.clone();
                            st.postscript_name = None;
                        });
                        changed = true;
                    }
                    at += r.len;
                }
                if changed {
                    refresh(&snapshot, t);
                    n += 1;
                }
            }
        }
        Ok(())
    })?;
    Ok(json!({"missing": missing, "replaced": n}))
}

fn resolve_missing(s: &mut Session, p: &Value) -> Result<Value> {
    match p.get("map").and_then(Value::as_object) {
        Some(m) => replace_fonts(s, &m.clone(), false),
        None => Ok(json!({"missing": missing_fonts(&s.active().ok_or(EngineError::NoDocument)?.doc)})),
    }
}

fn save_defaults(s: &mut Session) -> Result<Value> {
    let t = active_text(s).map_err(EngineError::Other)?;
    let c = t.char_runs().first().map(|r| r.style.clone()).unwrap_or_default();
    let pp = t.paragraph_runs().first().map(|r| r.style.clone()).unwrap_or_default();
    let out = json!({"font": c.font_family, "size": c.size_pt});
    s.type_defaults = Some((c, pp));
    Ok(out)
}

fn load_defaults(s: &mut Session, p: &Value) -> Result<Value> {
    let (c, pp) = s.type_defaults.clone().ok_or(EngineError::Other("no default type styles have been saved".into()))?;
    with_text_targets(s, p, "Load Default Type Styles", |t, _| {
        let n = t.text.len();
        style_range(t, 0, n, &|st| *st = c.clone());
        style_paragraphs(t, 0, n, &|st| *st = pp.clone());
        Ok(())
    })?;
    Ok(json!({"font": c.font_family, "size": c.size_pt}))
}

// ---------- registry ----------

pub fn specs() -> Vec<CommandSpec> {
    macro_rules! spec {
        ($id:expr, $label:expr, $menu:expr, $params:expr, $enabled:expr, $run:expr) => {
            CommandSpec { id: $id, label: $label, menu: $menu, shortcut: None, params: $params, enabled: $enabled, journal: true, run: $run }
        };
    }
    const LP: &str = r##"{"layer":id?}"##;
    const FP: &str = r##"{"layer":id? or "layers":[id,…]? (default selected Type layers)}"##;
    const OT: &str = r##"{"layer":id? or "layers":[id,…]? (default selected Type layers),"on":bool? (default: toggle representative),"range":[startChar,endChar]? (single layer; default all)}"##;
    vec![
        spec!("type.antiAlias.none", "None", &["Type", "Anti-Alias"], FP, has_format_target, |s, p| set_aa(s, p, AntiAlias::None)),
        spec!("type.antiAlias.sharp", "Sharp", &["Type", "Anti-Alias"], FP, has_format_target, |s, p| set_aa(s, p, AntiAlias::Sharp)),
        spec!("type.antiAlias.crisp", "Crisp", &["Type", "Anti-Alias"], FP, has_format_target, |s, p| set_aa(s, p, AntiAlias::Crisp)),
        spec!("type.antiAlias.strong", "Strong", &["Type", "Anti-Alias"], FP, has_format_target, |s, p| set_aa(s, p, AntiAlias::Strong)),
        spec!("type.antiAlias.smooth", "Smooth", &["Type", "Anti-Alias"], FP, has_format_target, |s, p| set_aa(s, p, AntiAlias::Smooth)),
        spec!("type.antiAlias.windowsLcd", "Windows LCD", &["Type", "Anti-Alias"], FP, has_format_target, |s, p| set_aa(s, p, AntiAlias::WindowsLcd)),
        spec!("type.antiAlias.windows", "Windows", &["Type", "Anti-Alias"], FP, has_format_target, |s, p| set_aa(s, p, AntiAlias::Windows)),
        spec!("type.orientation.horizontal", "Horizontal", &["Type", "Orientation"], FP, has_format_target, |s, p| set_orientation(
            s,
            p,
            Orientation::Horizontal
        )),
        spec!("type.orientation.vertical", "Vertical", &["Type", "Orientation"], FP, has_format_target, |s, p| set_orientation(s, p, Orientation::Vertical)),
        spec!("type.openType.standardLigatures", "Standard Ligatures", &["Type", "OpenType"], OT, has_format_target, |s, p| toggle_opentype(s, p, "liga")),
        spec!("type.openType.contextualAlternates", "Contextual Alternates", &["Type", "OpenType"], OT, has_format_target, |s, p| toggle_opentype(
            s, p, "calt"
        )),
        spec!("type.openType.discretionaryLigatures", "Discretionary Ligatures", &["Type", "OpenType"], OT, has_format_target, |s, p| toggle_opentype(
            s, p, "dlig"
        )),
        spec!("type.openType.swash", "Swash", &["Type", "OpenType"], OT, has_format_target, |s, p| toggle_opentype(s, p, "swsh")),
        spec!("type.openType.oldstyle", "Oldstyle", &["Type", "OpenType"], OT, has_format_target, |s, p| toggle_opentype(s, p, "onum")),
        spec!("type.openType.stylisticAlternates", "Stylistic Alternates", &["Type", "OpenType"], OT, has_format_target, |s, p| toggle_opentype(s, p, "salt")),
        spec!("type.openType.titlingAlternates", "Titling Alternates", &["Type", "OpenType"], OT, has_format_target, |s, p| toggle_opentype(s, p, "titl")),
        spec!("type.openType.ornaments", "Ornaments", &["Type", "OpenType"], OT, has_format_target, |s, p| toggle_opentype(s, p, "ornm")),
        spec!("type.openType.ordinals", "Ordinals", &["Type", "OpenType"], OT, has_format_target, |s, p| toggle_opentype(s, p, "ordn")),
        spec!("type.openType.fractions", "Fractions", &["Type", "OpenType"], OT, has_format_target, |s, p| toggle_opentype(s, p, "frac")),
        spec!("type.createWorkPath", "Create Work Path", &["Type"], LP, has_text, create_work_path),
        spec!("type.convertToShape", "Convert to Shape", &["Type"], LP, has_text, convert_to_shape),
        spec!("type.rasterizeTypeLayer", "Rasterize Type Layer", &["Type"], LP, has_text, |s, p| s.execute("type.rasterize", p.clone())),
        spec!("type.convertToParagraphText", "Convert to Paragraph Text", &["Type"], LP, has_point_text, to_paragraph),
        spec!("type.convertToPointText", "Convert to Point Text", &["Type"], LP, has_box_text, to_point),
        spec!(
            "type.warpText",
            "Warp Text…",
            &["Type"],
            r##"{"layer":id? or "layers":[id,…]? (default selected Type layers),"style":"none|arc|arcLower|arcUpper|arch|bulge|shellLower|shellUpper|flag|wave|fish|rise|fisheye|inflate|squeeze|twist"="arc","orientation":"horizontal|vertical"="horizontal","bend":-100..100=50,"horizontalDistortion":-100..100=0,"verticalDistortion":-100..100=0}"##,
            has_format_target,
            warp_text
        ),
        spec!(
            "type.attachToPath",
            "Type on a Path",
            &[],
            r##"{"layer":id? (default active Type layer),"shape":layerId? (a shape layer; default the work path),"subpath":index=0,"start":px=0 (offset along the path),"flip":bool=false (run along the path the other way, on its other side)}"##,
            has_text,
            attach_to_path
        ),
        spec!("type.detachFromPath", "Release Type from Path", &[], LP, is_path_text, detach_from_path),
        spec!("type.updateAllTextLayers", "Update All Text Layers", &["Type"], "{}", any_text, |s, _| update_all(s)),
        spec!("type.replaceAllMissingFonts", "Replace All Missing Fonts", &["Type"], "{} (with the default family)", any_text, |s, _| replace_fonts(
            s,
            &serde_json::Map::new(),
            true
        )),
        spec!(
            "type.resolveMissingFonts",
            "Resolve Missing Fonts…",
            &["Type"],
            r##"{"map":{"Missing Family":"Installed Family"}?} (no map: list the missing families)"##,
            any_text,
            resolve_missing
        ),
        spec!(
            "type.pasteLoremIpsum",
            "Paste Lorem Ipsum",
            &["Type"],
            r##"{"layer":id?,"at":char? (default end),"new":bool=false (new paragraph text layer)}"##,
            has_doc,
            paste_lorem
        ),
        spec!(
            "type.saveDefaultTypeStyles",
            "Save Default Type Styles",
            &["Type"],
            "{} (from the active type layer; new type layers start from them)",
            has_text,
            |s, _| save_defaults(s)
        ),
        spec!("type.loadDefaultTypeStyles", "Load Default Type Styles", &["Type"], FP, has_defaults, load_defaults),
    ]
}

#[cfg(test)]
#[path = "type_extra_cmds/tests.rs"]
mod tests;
