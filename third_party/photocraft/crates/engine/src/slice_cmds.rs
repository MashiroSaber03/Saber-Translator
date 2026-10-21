//! Web slices: the Slice tool (`slice.new`, Slices From Guides), the Slice Select tool (`slice.set`
//! = Slice Options, promote, divide, delete), Layer › New Layer Based Slice, View › Lock Slices and
//! View › Clear Slices.
//!
//! The document stores user and layer-based slices (`Document::slices`); auto slices are derived
//! (`photocraft_doc::slices::resolve`). Layer-based slices follow their layer: after every
//! command [`refresh`] re-fits them to the layer's bounds including layer effects, plus outsets,
//! without a history step of their own (the change is implied by the edit that moved the layer).
//! Slices are addressed by stored id (`"slice"`) or by their displayed number (`"number"`).

use std::sync::Arc;

use photocraft_doc::slices::{self, ResolvedSlice, Slice, SliceKind, SliceOrigin};
use photocraft_doc::{Document, Layer, LayerContent};
use photocraft_geom::Rect;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, layer_param};
use crate::{EngineError, Result, Session};

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn exhausted_id() -> EngineError {
    EngineError::Other("slice id space exhausted".into())
}

// ---------- predicates ----------

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

fn unlocked(s: &Session) -> std::result::Result<(), String> {
    has_doc(s)?;
    if s.file_menu.slices_locked { Err("slices are locked (View › Lock Slices)".into()) } else { Ok(()) }
}

fn has_slices(s: &Session) -> std::result::Result<(), String> {
    unlocked(s)?;
    if s.active().is_some_and(|d| !d.doc.slices.is_empty()) { Ok(()) } else { Err("the document has no user or layer-based slices".into()) }
}

fn has_layer_for_slice(s: &Session) -> std::result::Result<(), String> {
    unlocked(s)?;
    let d = s.active().ok_or("no document open")?;
    let id = d.active_layer.ok_or("no active layer")?;
    let l = d.doc.layer(id).ok_or("no active layer")?;
    if matches!(l.content, LayerContent::Adjustment(_)) {
        return Err("adjustment layers have no bounds to slice".into());
    }
    if d.doc.is_background(id) {
        return Err("the Background layer can't have a layer-based slice".into());
    }
    if d.doc.slices.list.iter().any(|s| s.layer == Some(id)) {
        return Err("the layer already has a layer-based slice".into());
    }
    Ok(())
}

// ---------- geometry ----------

/// Bounds of a layer's pixels plus its layer effects (groups: the union of their children).
pub fn layer_bounds(l: &Layer) -> Rect {
    let own = match &l.content {
        LayerContent::Group(g) => match &g.artboard {
            Some(a) => a.rect,
            None => g.children.iter().filter(|c| c.visible).map(layer_bounds).fold(Rect::EMPTY, |a, b| {
                if a.is_empty() {
                    b
                } else if b.is_empty() {
                    a
                } else {
                    a.union(&b)
                }
            }),
        },
        _ => photocraft_doc::comps::position_bounds(l),
    };
    if own.is_empty() || !photocraft_compose::effects::has_effects(l) {
        return own;
    }
    own.inflate(photocraft_compose::effects::margin(l))
}

fn layer_slice_rect(doc: &Document, s: &Slice) -> Option<Rect> {
    let l = doc.layer(s.layer?)?;
    let b = layer_bounds(l);
    let [t, le, bo, r] = s.outsets;
    Some(if b.is_empty() { b } else { Rect::new(b.x0.saturating_sub(le), b.y0.saturating_sub(t), b.x1.saturating_add(r), b.y1.saturating_add(bo)) })
}

/// Re-fits layer-based slices of the active document to their layers; drops slices whose layer
/// was deleted (as Photoshop does). Not a history step.
pub(crate) fn refresh(s: &mut Session) {
    let Some(st) = s.active_mut() else { return };
    if !st.doc.slices.list.iter().any(|sl| sl.origin == SliceOrigin::Layer) {
        return;
    }
    let doc = &st.doc;
    let mut changes: Vec<(u32, Option<Rect>)> = Vec::new();
    for sl in doc.slices.list.iter().filter(|sl| sl.origin == SliceOrigin::Layer) {
        let r = layer_slice_rect(doc, sl);
        if r != Some(sl.rect) {
            changes.push((sl.id, r));
        }
    }
    if changes.is_empty() {
        return;
    }
    let doc = Arc::make_mut(&mut st.doc);
    for (id, r) in changes {
        match r {
            Some(r) => {
                if let Some(sl) = doc.slices.get_mut(id) {
                    sl.rect = r;
                }
            }
            None => doc.slices.list.retain(|sl| sl.id != id),
        }
    }
}

// ---------- params ----------

pub(crate) fn rect_param(p: &Value) -> Option<Rect> {
    let f = |v: &Value| v.as_f64().filter(|x| x.is_finite()).map(|x| x.round() as i32);
    let rect = |x: i32, y: i32, w: i32, h: i32| {
        if w <= 0 || h <= 0 {
            return None;
        }
        Some(Rect::new(x, y, x.checked_add(w)?, y.checked_add(h)?))
    };
    if let Some(a) = p.get("rect").and_then(Value::as_array).filter(|a| a.len() == 4) {
        let v: Vec<i32> = a.iter().filter_map(f).collect();
        if v.len() == 4 {
            return rect(v[0], v[1], v[2], v[3]);
        }
        return None;
    }
    rect(f(p.get("x")?)?, f(p.get("y")?)?, f(p.get("width")?)?, f(p.get("height")?)?)
}

fn hex_color(s: &str) -> Option<[u8; 4]> {
    let s = s.trim_start_matches('#');
    let b = |i: usize| u8::from_str_radix(s.get(i..i + 2)?, 16).ok();
    match s.len() {
        6 => Some([255, b(0)?, b(2)?, b(4)?]),
        _ => None,
    }
}

/// Slice Options fields present in `p` applied to `sl`.
fn apply_options(sl: &mut Slice, p: &Value, cmd: &str) -> Result<()> {
    let text = |k: &str| p.get(k).and_then(Value::as_str).map(str::to_string);
    if let Some(v) = text("name") {
        sl.name = v;
    }
    if let Some(v) = text("kind").or_else(|| text("type")) {
        sl.kind = SliceKind::from_id(&v).ok_or_else(|| bad(cmd, format!("unknown slice type `{v}` (image|noImage|table)")))?;
    }
    for (k, slot) in [("url", &mut sl.url), ("target", &mut sl.target), ("message", &mut sl.message), ("alt", &mut sl.alt), ("cellText", &mut sl.cell_text)] {
        if let Some(v) = text(k) {
            *slot = v;
        }
    }
    if let Some(b) = p.get("cellTextIsHtml").and_then(Value::as_bool) {
        sl.cell_text_is_html = b;
    }
    for (k, slot) in [("horizontalAlign", &mut sl.horizontal_align), ("verticalAlign", &mut sl.vertical_align)] {
        if let Some(v) = crate::commands::int(p, k).filter(|v| *v >= 0).map(|v| v as u64) {
            *slot = v.min(4) as u32;
        }
    }
    match p.get("background").and_then(Value::as_str) {
        Some("none") | Some("") => sl.background = None,
        Some(c) => sl.background = Some(hex_color(c).ok_or_else(|| bad(cmd, "background is \"none\" or \"#rrggbb\""))?),
        None => {}
    }
    if let Some(a) = p.get("outsets").and_then(Value::as_array).filter(|a| a.len() == 4) {
        for (i, v) in a.iter().enumerate() {
            sl.outsets[i] = v.as_i64().unwrap_or(0).clamp(-10_000, 10_000) as i32;
        }
    }
    if let Some(r) = rect_param(p) {
        if sl.origin == SliceOrigin::Layer {
            // Editing a layer-based slice's geometry promotes it to a user slice.
            sl.origin = SliceOrigin::User;
            sl.layer = None;
        }
        sl.rect = r;
    }
    Ok(())
}

/// What `"slice"` (id) or `"number"` addresses: a stored slice, or an auto slice that isn't
/// stored yet (its rectangle).
#[derive(Clone, Copy)]
enum Target {
    Stored(u32),
    Auto(Rect),
}

fn find(s: &Session, p: &Value, cmd: &str) -> Result<Target> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    if let Some(id) = crate::commands::u32_id_param(cmd, p, "slice")? {
        return d.doc.slices.get(id).map(|sl| Target::Stored(sl.id)).ok_or_else(|| bad(cmd, format!("no slice with id {id}")));
    }
    let n = crate::commands::int(p, "number").filter(|v| *v > 0).ok_or_else(|| bad(cmd, "give \"slice\" (id) or \"number\""))? as usize;
    let r = slices::resolve(&d.doc).into_iter().find(|r| r.number == n).ok_or_else(|| bad(cmd, format!("no slice number {n}")))?;
    Ok(match r.id {
        Some(id) => Target::Stored(id),
        None => Target::Auto(r.rect),
    })
}

/// The stored slice's id; an auto slice is promoted to a user slice in `doc` first. Call it inside
/// the command's own edit, so promoting and the rest of the command are one history step and a
/// rejected command changes nothing.
fn store(doc: &mut Document, t: Target) -> Result<u32> {
    match t {
        Target::Stored(id) => Ok(id),
        Target::Auto(rect) => {
            let id = doc.slices.next_id().ok_or_else(exhausted_id)?;
            doc.slices.list.push(Slice { id, rect, ..Default::default() });
            Ok(id)
        }
    }
}

/// The stored slice addressed by `p`; an auto slice is an error.
fn stored(s: &Session, p: &Value, cmd: &str) -> Result<u32> {
    match find(s, p, cmd)? {
        Target::Stored(id) => Ok(id),
        Target::Auto(_) => Err(bad(cmd, "that is an auto slice")),
    }
}

pub fn resolved_json(doc: &Document, r: &ResolvedSlice) -> Value {
    let s = r.id.and_then(|id| doc.slices.get(id));
    json!({
        "number": r.number,
        "id": r.id,
        "origin": r.origin.id(),
        "name": r.name,
        "rect": [r.rect.x0, r.rect.y0, r.rect.width(), r.rect.height()],
        "kind": r.kind.id(),
        "layer": s.and_then(|s| s.layer).map(|l| l.0),
        "url": s.map_or("", |s| s.url.as_str()),
        "target": s.map_or("", |s| s.target.as_str()),
        "message": s.map_or("", |s| s.message.as_str()),
        "alt": s.map_or("", |s| s.alt.as_str()),
        "cellText": s.map_or("", |s| s.cell_text.as_str()),
        "background": s.and_then(|s| s.background).map(|c| format!("#{:02x}{:02x}{:02x}", c[1], c[2], c[3])),
    })
}

// ---------- commands ----------

fn new_slice(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "slice.new";
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let rect = rect_param(p).ok_or_else(|| bad(cmd, "give \"rect\":[x,y,w,h] (or x, y, width, height)"))?.intersect(&d.doc.bounds());
    if rect.is_empty() {
        return Err(bad(cmd, "the slice is outside the canvas"));
    }
    let id = s.edit("Slice", |doc, _| {
        let id = doc.slices.next_id().ok_or_else(exhausted_id)?;
        let mut sl = Slice { id, rect, ..Default::default() };
        apply_options(&mut sl, p, cmd)?;
        sl.rect = rect;
        doc.slices.list.push(sl);
        Ok(id)
    })?;
    let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
    let number = slices::resolve(doc).iter().find(|r| r.id == Some(id)).map(|r| r.number);
    Ok(json!({"slice": id, "number": number}))
}

fn from_guides(s: &mut Session) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let b = d.doc.bounds();
    let mut xs: Vec<i32> = vec![b.x0, b.x1];
    let mut ys: Vec<i32> = vec![b.y0, b.y1];
    xs.extend(d.doc.guides.vertical.iter().map(|g| g.round() as i32).filter(|x| *x > b.x0 && *x < b.x1));
    ys.extend(d.doc.guides.horizontal.iter().map(|g| g.round() as i32).filter(|y| *y > b.y0 && *y < b.y1));
    if xs.len() == 2 && ys.len() == 2 {
        return Err(EngineError::Other("the document has no guides inside the canvas".into()));
    }
    xs.sort_unstable();
    xs.dedup();
    ys.sort_unstable();
    ys.dedup();
    let mut rects = Vec::new();
    for yw in ys.windows(2) {
        for xw in xs.windows(2) {
            rects.push(Rect::new(xw[0], yw[0], xw[1], yw[1]));
        }
    }
    let n = rects.len();
    s.edit("Slices From Guides", |doc, _| {
        // Photoshop replaces every existing slice.
        doc.slices.list.clear();
        for r in rects {
            let id = doc.slices.next_id().ok_or_else(exhausted_id)?;
            doc.slices.list.push(Slice { id, rect: r, ..Default::default() });
        }
        Ok(())
    })?;
    Ok(json!({"slices": n}))
}

fn set_slice(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "slice.set";
    if ["rect", "x", "y", "width", "height"].iter().any(|key| p.get(key).is_some()) && rect_param(p).is_none() {
        return Err(bad(cmd, "give a finite rectangle with positive dimensions and representable corners"));
    }
    let t = find(s, p, cmd)?;
    // Promoting an auto slice and its options are one user action (one history step), and
    // nothing is committed unless every option is valid and the slice stays on the canvas.
    let id = s.edit("Slice Options", |doc, _| {
        let id = store(doc, t)?;
        let sl = doc.slices.get_mut(id).ok_or_else(|| bad(cmd, "slice vanished"))?;
        apply_options(sl, p, cmd)?;
        if !slices::resolve(doc).iter().any(|r| r.id == Some(id)) {
            return Err(bad(cmd, "slice is off the canvas"));
        }
        Ok(id)
    })?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let r = slices::resolve(&d.doc).into_iter().find(|r| r.id == Some(id)).ok_or_else(|| bad(cmd, "slice is off the canvas"))?;
    Ok(resolved_json(&d.doc, &r))
}

fn promote(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "slice.promote";
    let t = find(s, p, cmd)?;
    // Layer-based slices promote too: they stop following their layer.
    let promotes = match t {
        Target::Auto(_) => true,
        Target::Stored(id) => s.active().and_then(|d| d.doc.slices.get(id)).is_some_and(|sl| sl.origin == SliceOrigin::Layer),
    };
    let id = match t {
        Target::Stored(id) if !promotes => id,
        _ => s.edit("Promote to User Slice", |doc, _| {
            let id = store(doc, t)?;
            if let Some(sl) = doc.slices.get_mut(id) {
                sl.origin = SliceOrigin::User;
                sl.layer = None;
            }
            Ok(id)
        })?,
    };
    Ok(json!({"slice": id}))
}

fn ids_param(s: &mut Session, p: &Value, cmd: &str) -> Result<Vec<u32>> {
    if let Some(a) = p.get("slices").and_then(Value::as_array) {
        let d = s.active().ok_or(EngineError::NoDocument)?;
        let ids: Vec<u32> = a.iter().map(|v| crate::commands::u32_id(cmd, "slices", v)).collect::<Result<Vec<_>>>()?;
        let ids: Vec<u32> = ids.into_iter().filter(|id| d.doc.slices.get(*id).is_some()).collect();
        if ids.is_empty() {
            return Err(bad(cmd, "none of the slices exist"));
        }
        return Ok(ids);
    }
    Ok(vec![stored(s, p, cmd)?])
}

fn delete(s: &mut Session, p: &Value) -> Result<Value> {
    let ids = ids_param(s, p, "slice.delete")?;
    let n = ids.len();
    s.edit("Delete Slice", |doc, _| {
        doc.slices.list.retain(|sl| !ids.contains(&sl.id));
        Ok(())
    })?;
    Ok(json!({"deleted": n}))
}

/// The most parts one Divide Slice makes. Resolving the slice list is quadratic in the slice count
/// (`slices::resolve`), so a 1000 x 1000 division, a million slices, left every later list or
/// export with ~5e11 steps of work (#1020).
const MAX_DIVIDE_PARTS: i32 = 10_000;

fn divide(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "slice.divide";
    let t = find(s, p, cmd)?;
    let down = crate::commands::int(p, "horizontal").unwrap_or(1).clamp(1, 1000) as i32;
    let across = crate::commands::int(p, "vertical").unwrap_or(1).clamp(1, 1000) as i32;
    if down == 1 && across == 1 {
        return Err(bad(cmd, "give \"horizontal\" (slices down) and/or \"vertical\" (slices across) > 1"));
    }
    if down * across > MAX_DIVIDE_PARTS {
        return Err(bad(cmd, format!("{down} x {across} = {} parts is more than one division makes ({MAX_DIVIDE_PARTS})", down * across)));
    }
    let r = match t {
        Target::Auto(r) => r,
        Target::Stored(id) => s.active().and_then(|d| d.doc.slices.get(id)).map(|sl| sl.rect).ok_or_else(|| bad(cmd, "no such slice"))?,
    };
    if r.width() < across as u32 || r.height() < down as u32 {
        return Err(bad(cmd, "the slice is too small to divide that many times"));
    }
    let n = (down * across) as usize;
    // An auto slice is promoted inside the same step: one undo restores it.
    let ids = s.edit("Divide Slice", |doc, _| {
        let id = store(doc, t)?;
        let src = doc.slices.get(id).cloned().ok_or_else(|| bad(cmd, "no such slice"))?;
        let pos = doc.slices.list.iter().position(|sl| sl.id == id).unwrap_or(0);
        doc.slices.list.retain(|sl| sl.id != id);
        let mut ids = Vec::new();
        let mut parts = Vec::new();
        let mut next_id = doc.slices.next_id();
        for j in 0..down {
            for i in 0..across {
                // Products can exceed i32 even when every interpolated edge fits the rect.
                let x = |k: i32| (i64::from(r.x0) + i64::from(r.width()) * i64::from(k) / i64::from(across)) as i32;
                let y = |k: i32| (i64::from(r.y0) + i64::from(r.height()) * i64::from(k) / i64::from(down)) as i32;
                let (x0, x1) = (x(i), x(i + 1));
                let (y0, y1) = (y(j), y(j + 1));
                let nid = if ids.is_empty() {
                    id
                } else {
                    let mut candidate = next_id.ok_or_else(exhausted_id)?;
                    if candidate == id {
                        candidate = candidate.checked_add(1).ok_or_else(exhausted_id)?;
                    }
                    next_id = candidate.checked_add(1);
                    candidate
                };
                ids.push(nid);
                // The first part keeps the original's options; the others start fresh.
                let base = if parts.is_empty() { Slice { origin: SliceOrigin::User, layer: None, ..src.clone() } } else { Slice::default() };
                parts.push(Slice { id: nid, rect: Rect::new(x0, y0, x1, y1), ..base });
            }
        }
        for (k, sl) in parts.into_iter().enumerate() {
            doc.slices.list.insert((pos + k).min(doc.slices.list.len()), sl);
        }
        Ok(ids)
    })?;
    Ok(json!({"slices": ids, "count": n}))
}

fn list(s: &mut Session) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let all: Vec<Value> = slices::resolve(&d.doc).iter().map(|r| resolved_json(&d.doc, r)).collect();
    Ok(json!({"slices": all, "locked": s.file_menu.slices_locked}))
}

fn layer_based(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "layer.newLayerBasedSlice";
    let layer = layer_param(s, p)?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let l = d.doc.layer(layer).ok_or(EngineError::NoLayer(layer))?;
    if d.doc.slices.list.iter().any(|sl| sl.layer == Some(layer)) {
        return Err(bad(cmd, "the layer already has a layer-based slice"));
    }
    let rect = layer_bounds(l);
    if rect.is_empty() {
        return Err(EngineError::Other("Could not create a layer based slice because the layer is empty".into()));
    }
    let id = s.edit("New Layer Based Slice", |doc, _| {
        let id = doc.slices.next_id().ok_or_else(exhausted_id)?;
        let text = |k: &str| p.get(k).and_then(Value::as_str).unwrap_or_default().to_string();
        doc.slices.list.push(Slice {
            id,
            origin: SliceOrigin::Layer,
            layer: Some(layer),
            rect,
            name: text("name"),
            url: text("url"),
            alt: text("alt"),
            ..Default::default()
        });
        Ok(id)
    })?;
    Ok(json!({"slice": id, "rect": [rect.x0, rect.y0, rect.width(), rect.height()]}))
}

fn lock(s: &mut Session, p: &Value) -> Result<Value> {
    let on = p.get("on").and_then(Value::as_bool).unwrap_or(!s.file_menu.slices_locked);
    s.file_menu.slices_locked = on;
    Ok(json!({"locked": on}))
}

fn clear(s: &mut Session) -> Result<Value> {
    let n = s.active().map_or(0, |d| d.doc.slices.list.len());
    s.edit("Clear Slices", |doc, _| {
        doc.slices.list.clear();
        Ok(())
    })?;
    Ok(json!({"cleared": n}))
}

pub fn specs() -> Vec<CommandSpec> {
    macro_rules! spec {
        ($id:expr, $label:expr, $menu:expr, $params:expr, $enabled:expr, $run:expr) => {
            CommandSpec { id: $id, label: $label, menu: $menu, shortcut: None, params: $params, enabled: $enabled, journal: true, run: $run }
        };
    }
    vec![
        spec!(
            "slice.new",
            "Slice Tool",
            &[],
            r##"{"rect":[x,y,w,h] | "x","y","width","height", plus Slice Options ("name","kind":"image|noImage|table","url","target","message","alt","cellText","cellTextIsHtml","background":"none|#rrggbb")?} → {slice, number}"##,
            unlocked,
            new_slice
        ),
        spec!("slice.fromGuides", "Slices From Guides", &[], "{} (replaces every slice by the grid of the canvas guides) → {slices}", unlocked, |s, _| {
            from_guides(s)
        }),
        spec!(
            "slice.set",
            "Slice Options…",
            &[],
            r##"{"slice":id | "number":n (an auto slice is promoted), "name"?,"kind":"image|noImage|table"?,"url"?,"target"?,"message"?,"alt"?,"cellText"?,"cellTextIsHtml"?,"horizontalAlign":0..4?,"verticalAlign":0..4?,"background":"none|#rrggbb"?,"outsets":[t,l,b,r]? (layer slices),"rect":[x,y,w,h]? (move/resize; a layer slice becomes a user slice)} → the slice"##,
            unlocked,
            set_slice
        ),
        spec!("slice.promote", "Promote", &[], r##"{"slice":id | "number":n} (auto or layer-based → user slice) → {slice}"##, unlocked, promote),
        spec!("slice.delete", "Delete Slice", &[], r##"{"slice":id | "number":n | "slices":[id…]} → {deleted}"##, has_slices, delete),
        spec!(
            "slice.divide",
            "Divide Slice…",
            &[],
            r##"{"slice":id | "number":n,"horizontal":n=1 (slices down),"vertical":n=1 (slices across); at most 10000 parts} → {slices}"##,
            unlocked,
            divide
        ),
        CommandSpec {
            id: "slice.list",
            label: "List Slices",
            menu: &[],
            shortcut: None,
            params: "{} → {slices:[{number,id,origin:auto|layer|user,name,rect:[x,y,w,h],kind,layer,url,alt,…}], locked}",
            enabled: has_doc,
            journal: false,
            run: |s, _| list(s),
        },
        spec!(
            "layer.newLayerBasedSlice",
            "New Layer Based Slice",
            &["Layer"],
            r##"{"layer":id?,"name":str?,"url":str?,"alt":str?} (follows the layer's bounds with effects) → {slice, rect}"##,
            has_layer_for_slice,
            layer_based
        ),
        spec!("view.lockSlices", "Lock Slices", &["View"], r##"{"on":bool? (default: toggle)} → {locked}"##, has_doc, lock),
        spec!("view.clearSlices", "Clear Slices", &["View"], "{} (deletes every user and layer-based slice) → {cleared}", has_slices, |s, _| clear(s)),
    ]
}

#[cfg(test)]
#[path = "slice_cmds/tests.rs"]
mod tests;
