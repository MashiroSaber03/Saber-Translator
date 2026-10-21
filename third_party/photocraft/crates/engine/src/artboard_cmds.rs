//! Artboards: Layer › New › Artboard (from scratch, a group or the selected layers), artboard
//! properties (board rect, background, preset), View › Clear Selected Artboard Guides, and
//! File › Export › Artboards to Files / to PDF.
//!
//! An artboard is a top-level group with [`photocraft_doc::Artboard`] set; the compositor clips
//! its children to the board and paints its background. Moving the group (Move tool,
//! `layer.translate`) moves the board with its contents. The canvas grows to fit every board,
//! in every direction, like Photoshop's auto-expanding artboard canvas ([`fit_canvas`]).

use photocraft_color::{Color, ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Artboard, ArtboardBackground, Document, Layer, LayerContent, LayerId, Size};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, color_param, int, int_i32};
use crate::file_cmds::{SaveOpts, encode, f64_param, join, native_doc, sanitize, save_doc, stem, str_param, write_file};
use crate::{EngineError, Result, Session};

/// Artboard size presets (name, width, height) in pixels, Photoshop's common devices and pages.
pub const PRESETS: &[(&str, u32, u32)] = &[
    ("iPhone 14", 390, 844),
    ("iPhone 14 Pro Max", 430, 932),
    ("Android 1080p", 360, 640),
    ("iPad Pro 12.9", 1024, 1366),
    ("Web 1280", 1280, 800),
    ("Web 1366", 1366, 768),
    ("Web 1920", 1920, 1080),
    ("Instagram Post", 1080, 1080),
    ("Letter", 612, 792),
    ("A4", 595, 842),
];

/// Gap Photoshop leaves between a new artboard and the rightmost existing one.
const GAP: i32 = 100;

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

fn has_artboards(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    if d.doc.has_artboards() { Ok(()) } else { Err("the document has no artboards".into()) }
}

fn export_artboards(s: &Session) -> std::result::Result<(), String> {
    native_doc(s)?;
    has_artboards(s)
}

/// The active layer's artboard (the layer itself or its top-level ancestor).
fn active_artboard(s: &Session) -> Option<LayerId> {
    let d = s.active()?;
    d.doc.artboard_of(d.active_layer?)
}

fn has_active_artboard(s: &Session) -> std::result::Result<(), String> {
    active_artboard(s).map(|_| ()).ok_or_else(|| "select an artboard (or a layer inside one)".into())
}

fn has_plain_group(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    let id = d.active_layer.ok_or("no active layer")?;
    let l = d.doc.layer(id).ok_or("no active layer")?;
    if !l.is_group() || l.artboard().is_some() {
        return Err("select a layer group".into());
    }
    if !d.doc.layers.iter().any(|t| t.id == id) {
        return Err("artboards must be top-level: select a group that is not nested".into());
    }
    Ok(())
}

fn has_free_layers(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    let sel = d.selected_layers();
    if sel.is_empty() {
        return Err("select one or more layers".into());
    }
    if sel.iter().any(|id| d.doc.artboard_of(*id).is_some()) {
        return Err("the selection is already in an artboard".into());
    }
    Ok(())
}

fn preset_size(name: &str) -> Option<(u32, u32)> {
    PRESETS.iter().find(|(n, _, _)| n.eq_ignore_ascii_case(name)).map(|(_, w, h)| (*w, *h))
}

fn background_param(p: &Value, cmd: &str) -> Result<Option<ArtboardBackground>> {
    let custom = p.get("color").map(|_| {
        let c = color_param(p, "color", [1.0; 4]);
        ArtboardBackground::Custom(Color::rgba(c[0], c[1], c[2], c[3]))
    });
    Ok(match p.get("background").and_then(Value::as_str) {
        None => custom,
        Some("white") => Some(ArtboardBackground::White),
        Some("black") => Some(ArtboardBackground::Black),
        Some("transparent") => Some(ArtboardBackground::Transparent),
        Some("custom" | "other") => Some(custom.unwrap_or(ArtboardBackground::White)),
        Some(o) => return Err(bad(cmd, format!("unknown background `{o}` (white|black|transparent|custom)"))),
    })
}

/// A board rect from `x, y, width, height`, or a bad-params error naming them when an edge falls
/// outside the 32-bit coordinate range. `Rect::from_xywh` would clamp the far edge to `i32::MAX`
/// (silently shrinking the board) and wrap a width past `i32::MAX` negative (#931).
fn board_rect(cmd: &str, [x, y, w, h]: [i64; 4]) -> Result<Rect> {
    if w <= 0 || h <= 0 {
        return Err(bad(cmd, format!("the artboard size must be positive (width = {w}, height = {h})")));
    }
    let span = |a: i64, len: i64| Some((i32::try_from(a).ok()?, i32::try_from(a.checked_add(len)?).ok()?));
    match (span(x, w), span(y, h)) {
        (Some((x0, x1)), Some((y0, y1))) => Ok(Rect::new(x0, y0, x1, y1)),
        _ => Err(bad(
            cmd,
            format!("the artboard [x, y, width, height] = [{x}, {y}, {w}, {h}] reaches outside the 32-bit coordinate range ({}..={})", i32::MIN, i32::MAX),
        )),
    }
}

/// `rect` as `[x, y, w, h]` or `x`/`y`/`width`/`height` keys over `base` (`[x, y, w, h]`, wide so
/// a default position past the edge reaches [`board_rect`] instead of overflowing).
fn rect_param(p: &Value, base: [i64; 4], cmd: &str) -> Result<Rect> {
    if let Some(Value::Array(a)) = p.get("rect") {
        let v: Vec<i64> = a.iter().filter_map(|x| x.as_i64().or_else(|| x.as_f64().filter(|f| f.is_finite()).map(|f| f.round() as i64))).collect();
        let Ok(v) = <[i64; 4]>::try_from(v) else {
            return Err(bad(cmd, "\"rect\" must be [x, y, width, height] with a positive size"));
        };
        return board_rect(cmd, v);
    }
    let x = int_i32(cmd, p, "x")?.map_or(base[0], i64::from);
    let y = int_i32(cmd, p, "y")?.map_or(base[1], i64::from);
    let w = int(p, "width").unwrap_or(base[2]);
    let h = int(p, "height").unwrap_or(base[3]);
    board_rect(cmd, [x, y, w, h])
}

/// `r` as `[x, y, w, h]` for [`rect_param`].
fn xywh(r: Rect) -> [i64; 4] {
    [i64::from(r.x0), i64::from(r.y0), i64::from(r.width()), i64::from(r.height())]
}

/// Layer › Duplicate on an artboard (#1531): Photoshop never stacks the copy on the original,
/// it puts it right of the board, a [`GAP`] away, sliding further right past any board in the
/// way. Moves the copy, layer `id` (just inserted on top of its original), there with its
/// contents and grows the canvas to show it; anything that isn't an artboard is left alone.
pub(crate) fn place_copy(doc: &mut Document, id: LayerId) -> Result<()> {
    let l = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    let Some(r) = l.artboard().map(|a| a.rect) else { return Ok(()) };
    let name = l.name.clone();
    let boards: Vec<Rect> = doc.artboards().iter().filter(|b| b.0 != id).map(|b| b.2.rect).collect();
    let (y0, y1, w) = (i64::from(r.y0), i64::from(r.y1), i64::from(r.width()));
    // In i64: `x1 + GAP` passes i32::MAX for a board at the far right edge (#931).
    let mut x = i64::from(r.x1) + i64::from(GAP);
    // Each round clears every board in the way, so one round per board settles it.
    for _ in 0..=boards.len() {
        let blocked =
            boards.iter().filter(|b| i64::from(b.x0) < x + w && i64::from(b.x1) > x && i64::from(b.y0) < y1 && i64::from(b.y1) > y0).map(|b| b.x1).max();
        match blocked {
            Some(edge) => x = i64::from(edge) + i64::from(GAP),
            None => break,
        }
    }
    let fits = x.checked_add(w).is_some_and(|x1| x1 <= i64::from(i32::MAX));
    let dx = i32::try_from(x - i64::from(r.x0)).ok().filter(|_| fits).ok_or_else(|| {
        EngineError::Other(format!("no room for a copy of artboard \"{name}\" right of it; duplicate it in place (\"inPlace\": true) and move it"))
    })?;
    // Grow the canvas first, so shapes on the board re-render uncut where they land.
    let x1 = i64::from(doc.size.width).max(x + w);
    doc.size = Size::new(x1.clamp(1, 300_000) as u32, doc.size.height);
    let snapshot = doc.clone();
    let copy = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
    crate::commands::translate_layer(&snapshot, copy, dx, 0);
    crate::vector_cmds::translate_vectors(&snapshot, copy, f64::from(dx), 0.0);
    fit_canvas(doc);
    Ok(())
}

/// Grow the canvas so every artboard fits. Photoshop's artboard documents grow in every
/// direction: a board left of or above the origin shifts the whole document (boards, their
/// contents, guides, paths…) right/down by its overhang, then the canvas grows to show it. An
/// axis whose boards span more than the largest canvas, or a shift that would push pixels past
/// the 32-bit range, stays unshifted: the board stays partly off-canvas, as one past the size
/// limit does.
pub(crate) fn fit_canvas(doc: &mut Document) {
    let max = i64::from(crate::document_preset_cmds::MAX_DIMENSION);
    let boards: Vec<Rect> = doc.artboards().iter().map(|b| b.2.rect).collect();
    // The canvas and every board, per axis (`lo` ≤ 0 is the canvas origin or a board past it).
    let span = |edges: &dyn Fn(&Rect) -> (i32, i32), len: u32| {
        boards.iter().map(edges).fold((0, i64::from(len)), |(lo, hi), (a, b)| (lo.min(i64::from(a)), hi.max(i64::from(b))))
    };
    let (x0, x1) = span(&|r| (r.x0, r.x1), doc.size.width);
    let (y0, y1) = span(&|r| (r.y0, r.y1), doc.size.height);
    let shift = |lo: i64, hi: i64| if hi - lo <= max { i32::try_from(-lo).unwrap_or(0) } else { 0 };
    let (mut sx, mut sy) = (shift(x0, x1), shift(y0, y1));
    // `translate_doc` checks before it moves anything, so a refusal leaves the document as is.
    if (sx, sy) != (0, 0) && crate::image_cmds::translate_doc(doc, "artboard", sx, sy).is_err() {
        (sx, sy) = (0, 0);
    }
    let size = Size::new((x1 + i64::from(sx)).clamp(1, max) as u32, (y1 + i64::from(sy)).clamp(1, max) as u32);
    if size != doc.size {
        doc.size = size;
        // Shape pixels are cut at the canvas: a board moved past the old edge rendered its
        // shapes against it, leaving them missing or clipped (#2389).
        crate::canvas_geom::refresh(doc, crate::canvas_geom::Refresh::Shapes);
    }
}

fn next_artboard_name(doc: &Document) -> String {
    let names: Vec<&str> = doc.artboards().into_iter().map(|b| b.1).collect();
    // `names.len() + 1` candidates always include a free one.
    (1..=names.len() + 1).map(|n| format!("Artboard {n}")).find(|n| !names.contains(&n.as_str())).unwrap_or_else(|| "Artboard".into())
}

fn set_artboard(l: &mut Layer, a: Artboard) {
    if let LayerContent::Group(g) = &mut l.content {
        g.artboard = Some(a);
        g.expanded = true;
    }
    // Artboards composite like pass-through groups.
    l.clipped = false;
}

fn new_artboard(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "layer.new.artboard";
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let preset = p.get("preset").and_then(Value::as_str).unwrap_or("").to_string();
    let (pw, ph) = match preset.as_str() {
        "" => (d.doc.size.width, d.doc.size.height),
        name => preset_size(name).ok_or_else(|| bad(cmd, format!("unknown preset `{name}`")))?,
    };
    // Default placement: right of the rightmost artboard, else the canvas origin.
    let right = d.doc.artboards().iter().map(|b| b.2.rect.x1).max();
    let top = d.doc.artboards().iter().map(|b| b.2.rect.y0).min().unwrap_or(0);
    // In i64: `x1 + GAP` passes i32::MAX for a board at the far right edge (#931).
    let base = match right {
        Some(x) => [i64::from(x) + i64::from(GAP), i64::from(top), i64::from(pw), i64::from(ph)],
        None => [0, 0, i64::from(pw), i64::from(ph)],
    };
    // With no room right of the rightmost board, ask for a position rather than pick another
    // spot that may overlap a board; an explicit `x` (or `rect`) still works.
    let explicit_x = p.get("rect").is_some() || p.get("x").is_some_and(|v| !v.is_null());
    if let Some(edge) = right.filter(|_| !explicit_x && base[0] > i64::from(i32::MAX)) {
        return Err(bad(cmd, format!("no room for a new artboard right of the rightmost one (its right edge is x = {edge}); pass \"x\" to place it")));
    }
    // A preset decides the size (dialogs send the canvas size alongside it).
    let rect = if preset.is_empty() || p.get("rect").is_some() {
        rect_param(p, base, cmd)?
    } else {
        rect_param(&json!({"x": p.get("x"), "y": p.get("y")}), base, cmd)?
    };
    let background = background_param(p, cmd)?.unwrap_or_default();
    let name = p.get("name").and_then(Value::as_str).filter(|n| !n.is_empty()).map(str::to_string);
    let id = s.edit("New Artboard", |doc, active| {
        let mut g = Layer::group(name.unwrap_or_else(|| next_artboard_name(doc)), vec![]);
        set_artboard(&mut g, Artboard { rect, background, preset });
        let id = doc.insert_above(None, g);
        fit_canvas(doc);
        *active = Some(id);
        Ok(id)
    })?;
    crate::layer_multi_cmds::set_selection(s, vec![id], Some(id), Some(id))?;
    Ok(json!({"layer": id.0, "rect": [rect.x0, rect.y0, rect.width(), rect.height()]}))
}

fn artboard_from_group(s: &mut Session, p: &Value) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let id = p.get("layer").and_then(Value::as_u64).map(LayerId).or(d.active_layer).ok_or(EngineError::Other("no active layer".into()))?;
    let canvas = d.doc.bounds();
    let background = background_param(p, "layer.new.artboardFromGroup")?.unwrap_or_default();
    let rect = s.edit("Artboard from Group", |doc, active| {
        if !doc.layers.iter().any(|t| t.id == id && t.is_group() && t.artboard().is_none()) {
            return Err(EngineError::Other("artboards are made from a top-level layer group".into()));
        }
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        let rect = crate::layer_multi_cmds::layer_bounds(l).unwrap_or(canvas);
        set_artboard(l, Artboard { rect, background, preset: String::new() });
        fit_canvas(doc);
        *active = Some(id);
        Ok(rect)
    })?;
    Ok(json!({"layer": id.0, "rect": [rect.x0, rect.y0, rect.width(), rect.height()]}))
}

fn artboard_from_layers(s: &mut Session, p: &Value) -> Result<Value> {
    let sel = crate::layer_multi_cmds::selected(s);
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let canvas = d.doc.bounds();
    let name = p.get("name").and_then(Value::as_str).map(str::to_string);
    let background = background_param(p, "layer.new.artboardFromLayers")?.unwrap_or_default();
    let (gid, rect) = s.edit("Artboard from Layers", |doc, active| {
        let ids = crate::layer_multi_cmds::top_level(doc, &sel);
        if ids.is_empty() || ids.iter().any(|id| doc.artboard_of(*id).is_some()) {
            return Err(EngineError::Other("select layers that are not in an artboard".into()));
        }
        let mut children = Vec::with_capacity(ids.len());
        for id in &ids {
            children.push(doc.remove(*id).ok_or(EngineError::NoLayer(*id))?);
        }
        let rect = children.iter().filter_map(crate::layer_multi_cmds::layer_bounds).reduce(|a, b| a.union(&b)).unwrap_or(canvas);
        let mut g = Layer::group(name.unwrap_or_else(|| next_artboard_name(doc)), children);
        set_artboard(&mut g, Artboard { rect, background, preset: String::new() });
        let gid = doc.insert_above(None, g);
        crate::layer_multi_cmds::check_group_depth(doc, "Artboard from Layers")?;
        fit_canvas(doc);
        *active = Some(gid);
        Ok((gid, rect))
    })?;
    crate::layer_multi_cmds::set_selection(s, vec![gid], Some(gid), Some(gid))?;
    Ok(json!({"layer": gid.0, "rect": [rect.x0, rect.y0, rect.width(), rect.height()]}))
}

/// Properties panel / Artboard tool: board position and size, background, preset, name.
/// Changing X/Y moves the board with its contents; changing W/H resizes it from the top left.
fn set_props(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "layer.artboard.set";
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let id = match p.get("layer").and_then(Value::as_u64) {
        Some(v) => LayerId(v),
        None => active_artboard(s).ok_or_else(|| bad(cmd, "no artboard selected"))?,
    };
    let old = d.doc.layer(id).and_then(Layer::artboard).cloned().ok_or_else(|| bad(cmd, "the layer is not an artboard"))?;
    let preset = p.get("preset").and_then(Value::as_str).map(str::to_string);
    let mut base = xywh(old.rect);
    if let Some(name) = preset.as_deref().filter(|n| !n.is_empty()) {
        let (w, h) = preset_size(name).ok_or_else(|| bad(cmd, format!("unknown preset `{name}`")))?;
        (base[2], base[3]) = (i64::from(w), i64::from(h));
    }
    let rect = rect_param(p, base, cmd)?;
    let background = background_param(p, cmd)?;
    let name = p.get("name").and_then(Value::as_str).map(str::to_string);
    let move_contents = p.get("moveContents").and_then(Value::as_bool).unwrap_or(true);
    // The offset between two in-range edges can need 33 bits (#931); layers move by i32 only.
    let (dx, dy) = (i64::from(rect.x0) - i64::from(old.rect.x0), i64::from(rect.y0) - i64::from(old.rect.y0));
    let (dx, dy) = match (i32::try_from(dx), i32::try_from(dy)) {
        (Ok(dx), Ok(dy)) => (dx, dy),
        _ if !move_contents => (0, 0),
        _ => {
            return Err(bad(
                cmd,
                format!(
                    "moving the artboard with its contents by ({dx}, {dy}) is outside the 32-bit range; move it in smaller steps or pass \"moveContents\": false"
                ),
            ));
        }
    };
    s.edit("Edit Artboard", |doc, _| {
        if move_contents && (dx, dy) != (0, 0) {
            let snapshot = doc.clone();
            let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
            crate::commands::translate_layer(&snapshot, l, dx, dy);
            crate::vector_cmds::translate_vectors(&snapshot, l, f64::from(dx), f64::from(dy));
        }
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        if let Some(n) = name {
            l.name = n;
        }
        let a = l.artboard_mut().ok_or(EngineError::NoLayer(id))?;
        a.rect = rect;
        if let Some(b) = background {
            a.background = b;
        }
        match preset {
            Some(pr) => a.preset = pr,
            // A hand-edited size no longer matches the preset.
            None if rect.width() != old.rect.width() || rect.height() != old.rect.height() => a.preset.clear(),
            None => {}
        }
        fit_canvas(doc);
        Ok(())
    })?;
    Ok(json!({"layer": id.0, "rect": [rect.x0, rect.y0, rect.width(), rect.height()]}))
}

/// The Artboard tool's "+" buttons: a board the size of board `layer` on one of its sides, a
/// [`GAP`] away (further out past any board in the way), empty or (`duplicate`, the ⌥-click) a
/// copy with its contents. A board left of or above the canvas origin shifts the whole document
/// right/down so the canvas can grow to show it. One undo step.
fn add_beside(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "layer.artboard.addBeside";
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let id = match p.get("layer").and_then(Value::as_u64) {
        Some(v) => LayerId(v),
        None => active_artboard(s).ok_or_else(|| bad(cmd, "no artboard selected"))?,
    };
    let src = d.doc.layer(id).and_then(Layer::artboard).cloned().ok_or_else(|| bad(cmd, "the layer is not an artboard"))?;
    let side = p.get("side").and_then(Value::as_str).unwrap_or("right");
    let duplicate = p.get("duplicate").and_then(Value::as_bool).unwrap_or(false);
    let [x0, y0, w, h] = xywh(src.rect);
    let gap = i64::from(GAP);
    // In i64: edges ± a board and a gap can pass the i32 range (#931); `board_rect` checks.
    let (mut x, mut y) = match side {
        "right" => (x0 + w + gap, y0),
        "left" => (x0 - gap - w, y0),
        "bottom" => (x0, y0 + h + gap),
        "top" => (x0, y0 - gap - h),
        o => return Err(bad(cmd, format!("unknown side `{o}` (left|right|top|bottom)"))),
    };
    let boards: Vec<Rect> = d.doc.artboards().iter().map(|b| b.2.rect).collect();
    // Each round clears every board in the way, so one round per board settles it.
    for _ in 0..=boards.len() {
        let blocked: Vec<&Rect> =
            boards.iter().filter(|b| i64::from(b.x0) < x + w && i64::from(b.x1) > x && i64::from(b.y0) < y + h && i64::from(b.y1) > y).collect();
        if blocked.is_empty() {
            break;
        }
        match side {
            "right" => x = blocked.iter().map(|b| i64::from(b.x1)).max().unwrap_or(x) + gap,
            "left" => x = blocked.iter().map(|b| i64::from(b.x0)).min().unwrap_or(x) - gap - w,
            "bottom" => y = blocked.iter().map(|b| i64::from(b.y1)).max().unwrap_or(y) + gap,
            _ => y = blocked.iter().map(|b| i64::from(b.y0)).min().unwrap_or(y) - gap - h,
        }
    }
    let rect = board_rect(cmd, [x, y, w, h])?;
    // The canvas starts at (0, 0): a board past it moves everything right/down instead.
    let (sx, sy) = (rect.x0.min(0).checked_neg(), rect.y0.min(0).checked_neg());
    let (Some(sx), Some(sy)) = (sx, sy) else {
        return Err(bad(cmd, format!("no room for an artboard {side} of \"{}\"", d.doc.layer(id).map_or("", |l| l.name.as_str()))));
    };
    let grown = |len: u32, by: i32| (i64::from(len) + i64::from(by)).clamp(1, 300_000) as u32;
    let label = if duplicate { "Duplicate Artboard" } else { "New Artboard" };
    let (nid, rect) = s.edit(label, |doc, active| {
        if (sx, sy) != (0, 0) {
            crate::image_cmds::translate_doc(doc, cmd, sx, sy)?;
            doc.size = Size::new(grown(doc.size.width, sx), grown(doc.size.height, sy));
            crate::canvas_geom::refresh(doc, crate::canvas_geom::Refresh::Shapes);
        }
        let (Some(x0), Some(y0), Some(x1), Some(y1)) = (rect.x0.checked_add(sx), rect.y0.checked_add(sy), rect.x1.checked_add(sx), rect.y1.checked_add(sy))
        else {
            return Err(bad(cmd, "the new artboard is outside the 32-bit coordinate range"));
        };
        let rect = Rect::new(x0, y0, x1, y1);
        let nid = if duplicate {
            let src_rect = doc.layer(id).and_then(Layer::artboard).map(|a| a.rect).ok_or(EngineError::NoLayer(id))?;
            let (dx, dy) = (rect.x0.saturating_sub(src_rect.x0), rect.y0.saturating_sub(src_rect.y0));
            let copy = crate::commands::layer_copy(doc, id)?;
            let nid = doc.insert_above(Some(id), copy);
            // Grow the canvas first, so shapes on the board re-render uncut where they land.
            let fit = |len: u32, edge: i32| i64::from(len).max(i64::from(edge)).clamp(1, 300_000) as u32;
            doc.size = Size::new(fit(doc.size.width, rect.x1), fit(doc.size.height, rect.y1));
            let snapshot = doc.clone();
            let l = doc.layer_mut(nid).ok_or(EngineError::NoLayer(nid))?;
            crate::commands::translate_layer(&snapshot, l, dx, dy);
            crate::vector_cmds::translate_vectors(&snapshot, l, f64::from(dx), f64::from(dy));
            nid
        } else {
            let mut g = Layer::group(next_artboard_name(doc), vec![]);
            set_artboard(&mut g, Artboard { rect, background: src.background, preset: src.preset.clone() });
            doc.insert_above(Some(id), g)
        };
        fit_canvas(doc);
        *active = Some(nid);
        Ok((nid, rect))
    })?;
    crate::layer_multi_cmds::set_selection(s, vec![nid], Some(nid), Some(nid))?;
    Ok(json!({"layer": nid.0, "rect": [rect.x0, rect.y0, rect.width(), rect.height()]}))
}

fn clear_artboard_guides(s: &mut Session) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let id = active_artboard(s).ok_or(EngineError::Other("no artboard selected".into()))?;
    let r = d.doc.layer(id).and_then(Layer::artboard).map(|a| a.rect).ok_or(EngineError::NoLayer(id))?;
    let inside_x = |x: &f64| *x > f64::from(r.x0) && *x < f64::from(r.x1);
    let inside_y = |y: &f64| *y > f64::from(r.y0) && *y < f64::from(r.y1);
    let n = d.doc.guides.vertical.iter().filter(|x| inside_x(x)).count() + d.doc.guides.horizontal.iter().filter(|y| inside_y(y)).count();
    if n == 0 {
        return Ok(json!({"cleared": 0}));
    }
    s.edit("Clear Selected Artboard Guides", |doc, _| {
        doc.guides.vertical.retain(|x| !inside_x(x));
        doc.guides.horizontal.retain(|y| !inside_y(y));
        doc.guides.prune_colors();
        Ok(())
    })?;
    Ok(json!({"cleared": n}))
}

/// One artboard as its own document: the board's size, its group moved to the origin. `None`
/// when `id` is not an artboard; an error when the board is too far from the origin to move.
pub fn artboard_document(doc: &Document, id: LayerId) -> Result<Option<Document>> {
    let Some(l) = doc.layer(id) else { return Ok(None) };
    let Some(r) = l.artboard().map(|a| a.rect) else { return Ok(None) };
    // A board starting at i32::MIN needs an offset of 2^31 to reach the origin, one past
    // i32::MAX (#931).
    let (Some(dx), Some(dy)) = (r.x0.checked_neg(), r.y0.checked_neg()) else {
        return Err(EngineError::Other(format!("artboard `{}` at ({}, {}) is too far from the origin to export", l.name, r.x0, r.y0)));
    };
    let mut group = l.clone();
    group.visible = true;
    crate::commands::translate_layer(doc, &mut group, dx, dy);
    crate::vector_cmds::translate_vectors(doc, &mut group, f64::from(dx), f64::from(dy));
    let mut one = doc.clone();
    one.name = l.name.clone();
    one.size = Size::new(r.width(), r.height());
    one.layers = vec![group];
    one.channels.clear();
    one.selection = None;
    one.quick_mask = None;
    one.layer_comps.clear();
    one.last_applied_comp = None;
    one.last_document_state = None;
    let shift = |v: &[f64], o: i32, len: u32| v.iter().map(|g| g - f64::from(o)).filter(|g| *g > 0.0 && *g < f64::from(len)).collect::<Vec<_>>();
    one.guides.vertical = shift(&doc.guides.vertical, r.x0, r.width());
    one.guides.horizontal = shift(&doc.guides.horizontal, r.y0, r.height());
    Ok(Some(one))
}

/// [`artboard_document`] for each chosen board, all built before any file is written so a
/// board that can't be exported fails the export without leaving a partial set behind.
fn chosen_documents(doc: &Document, p: &Value) -> Result<Vec<Document>> {
    chosen_artboards(doc, p).into_iter().filter_map(|id| artboard_document(doc, id).transpose()).collect()
}

/// The artboards an export covers: `"artboards": [ids]`, else all, top of the Layers panel first.
fn chosen_artboards(doc: &Document, p: &Value) -> Vec<LayerId> {
    let all: Vec<LayerId> = doc.artboards().into_iter().rev().map(|b| b.0).collect();
    match p.get("artboards") {
        Some(Value::Array(a)) => {
            let want: Vec<u64> = a.iter().filter_map(Value::as_u64).collect();
            all.into_iter().filter(|id| want.contains(&id.0)).collect()
        }
        _ => all,
    }
}

/// File › Export › Artboards to Files: `<prefix>_<artboard name>.<format>` per board.
fn artboards_to_files(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "file.export.artboardsToFiles";
    let dir = str_param(p, "dir", cmd).or_else(|_| str_param(p, "output", cmd))?.to_string();
    let format = p.get("format").and_then(Value::as_str).unwrap_or("png").trim_start_matches('.').to_ascii_lowercase();
    let save = SaveOpts::from_params(s, p, cmd)?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let prefix = p.get("prefix").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| stem(&d.doc.name));
    let doc = d.doc.clone();
    let mut files = Vec::new();
    for one in chosen_documents(&doc, p)? {
        let name = if prefix.is_empty() { sanitize(&one.name) } else { format!("{}_{}", sanitize(&prefix), sanitize(&one.name)) };
        let path = join(&dir, &format!("{name}.{format}"));
        save_doc(&one, &path, save)?;
        files.push(path);
    }
    if files.is_empty() {
        return Err(bad(cmd, "no matching artboards"));
    }
    Ok(json!({"files": files}))
}

/// A minimal PDF with one page per JPEG image: (width px, height px, dpi, JPEG bytes).
/// Pages are sized so the image prints at its resolution.
pub fn raster_pdf(pages: &[(u32, u32, f32, Vec<u8>)]) -> Vec<u8> {
    let mut out: Vec<u8> = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n".to_vec();
    let mut offsets: Vec<usize> = Vec::new();
    let mut obj = |out: &mut Vec<u8>, body: &[u8]| {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n", offsets.len()).as_bytes());
        out.extend_from_slice(body);
        out.extend_from_slice(b"\nendobj\n");
    };
    // Objects: 1 catalog, 2 page tree, then per page: page, contents, image.
    let kids: Vec<String> = (0..pages.len()).map(|i| format!("{} 0 R", 3 + i * 3)).collect();
    obj(&mut out, b"<< /Type /Catalog /Pages 2 0 R >>");
    obj(&mut out, format!("<< /Type /Pages /Kids [{}] /Count {} >>", kids.join(" "), pages.len()).as_bytes());
    for (i, (w, h, dpi, jpeg)) in pages.iter().enumerate() {
        let k = 72.0 / f64::from(dpi.max(1.0));
        let (pw, ph) = (f64::from(*w) * k, f64::from(*h) * k);
        let (contents, image) = (4 + i * 3, 5 + i * 3);
        obj(
            &mut out,
            format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw:.3} {ph:.3}] /Resources << /XObject << /Im0 {image} 0 R >> >> /Contents {contents} 0 R >>"
            )
            .as_bytes(),
        );
        let stream = format!("q {pw:.3} 0 0 {ph:.3} 0 0 cm /Im0 Do Q");
        obj(&mut out, format!("<< /Length {} >>\nstream\n{stream}\nendstream", stream.len()).as_bytes());
        let mut img = format!(
            "<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {} >>\nstream\n",
            jpeg.len()
        )
        .into_bytes();
        img.extend_from_slice(jpeg);
        img.extend_from_slice(b"\nendstream");
        obj(&mut out, &img);
    }
    let xref = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", offsets.len() + 1).as_bytes());
    for o in &offsets {
        out.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", offsets.len() + 1).as_bytes());
    out
}

/// File › Export › Artboards to PDF: one page per board (flattened over white, JPEG-encoded).
fn artboards_to_pdf(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "file.export.artboardsToPdf";
    let path = str_param(p, "path", cmd)?.to_string();
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let doc = d.doc.clone();
    let quality = f64_param(p, "quality").or(Some(10.0));
    let mut pages = Vec::new();
    for one in chosen_documents(&doc, p)? {
        let buf = photocraft_compose::flatten(&one).over_background([1.0, 1.0, 1.0]);
        let fmt = PixelFormat::new(ColorMode::Rgb, SampleType::U8, true);
        let mut flat = Document::new(&one.name, one.size, ColorMode::Rgb, SampleType::U8);
        let mut surf = Surface::new(fmt);
        let vals: Vec<f32> = buf.px.iter().flat_map(|px| photocraft_raster::from_rgba(&fmt, *px)).collect();
        surf.write_region(one.bounds(), &vals);
        flat.layers.push(Layer::new("Background", LayerContent::Raster(surf)));
        let (jpeg, _) = encode(&flat, "page.jpg", quality)?;
        pages.push((one.size.width, one.size.height, doc.resolution_dpi, jpeg));
    }
    if pages.is_empty() {
        return Err(bad(cmd, "no matching artboards"));
    }
    write_file(&path, &raster_pdf(&pages))?;
    Ok(json!({"path": path, "pages": pages.len()}))
}

pub fn specs() -> Vec<CommandSpec> {
    macro_rules! spec {
        ($id:expr, $label:expr, $menu:expr, $params:expr, $enabled:expr, $run:expr) => {
            CommandSpec { id: $id, label: $label, menu: $menu, shortcut: None, params: $params, enabled: $enabled, journal: true, run: $run }
        };
    }
    vec![
        spec!(
            "layer.new.artboard",
            "Artboard…",
            &["Layer", "New"],
            r##"{"rect":[x,y,w,h]? | "x","y","width","height"? (default: canvas size, right of the last board),"preset":"iPhone 14|Web 1920|A4|…"?,"name":str?,"background":"white|black|transparent|custom"="white","color":[r,g,b]|"#rrggbb"? (custom)} → {layer, rect}"##,
            has_doc,
            new_artboard
        ),
        spec!(
            "layer.new.artboardFromGroup",
            "Artboard from Group…",
            &["Layer", "New"],
            r##"{"layer":id? (a top-level group; default active),"background":…?} → {layer, rect}"##,
            has_plain_group,
            artboard_from_group
        ),
        spec!(
            "layer.new.artboardFromLayers",
            "Artboard from Layers…",
            &["Layer", "New"],
            r##"{"name":str?,"background":…?} (the selected layers) → {layer, rect}"##,
            has_free_layers,
            artboard_from_layers
        ),
        spec!(
            "layer.artboard.set",
            "Edit Artboard",
            &[],
            r##"{"layer":id? (default: the active artboard),"x","y","width","height"?|"rect":[x,y,w,h]?,"preset":str?,"background":"white|black|transparent|custom"?,"color":…?,"name":str?,"moveContents":bool=true} → {layer, rect}"##,
            has_active_artboard,
            set_props
        ),
        spec!(
            "view.clearSelectedArtboardGuides",
            "Clear Selected Artboard Guides",
            &["View"],
            "{} (guides inside the active artboard)",
            has_active_artboard,
            |s, _| clear_artboard_guides(s)
        ),
        spec!(
            "layer.artboard.addBeside",
            "Add Artboard Beside",
            &[],
            r##"{"layer":id? (default: the active artboard),"side":"left|right|top|bottom"="right","duplicate":bool=false (copy the board with its contents instead of an empty one)} → {layer, rect} (the Artboard tool's "+" buttons: same size, a 100 px gap away, past any board in the way)"##,
            has_active_artboard,
            add_beside
        ),
        spec!(
            "file.export.artboardsToFiles",
            "Artboards to Files…",
            &["File", "Export"],
            r##"{"dir":folder,"format":"png|jpg|psd|tiff|…"="png","prefix":str=document name ("" = none),"artboards":[id]? (default all),"quality":0..12?} → {files}"##,
            export_artboards,
            artboards_to_files
        ),
        spec!(
            "file.export.artboardsToPdf",
            "Artboards to PDF…",
            &["File", "Export"],
            r##"{"path":str (.pdf),"artboards":[id]?,"quality":0..12=10} → {path, pages} (one raster page per board)"##,
            export_artboards,
            artboards_to_pdf
        ),
    ]
}

#[cfg(test)]
#[path = "artboard_cmds/tests.rs"]
mod tests;
