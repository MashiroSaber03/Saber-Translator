//! Patterns: Edit › Define Pattern, the pattern library (`pattern.*`), Layer › New Fill Layer ›
//! Pattern, Edit › Fill with a pattern and brush textures from library patterns.
//!
//! Two places hold patterns, as in Photoshop:
//! - the **library** ([`PatternLibrary`], on the [`Session`]): built-in procedural patterns plus
//!   defined and imported ones. Library edits are app state, not document history.
//! - the **document** ([`photocraft_doc::Document::patterns`]): the patterns its layers use
//!   (saved in PSD `Patt` blocks and `.pcraft`). Commands that apply a pattern copy it into the
//!   document in the same history step, so a document always renders on its own.
//!
//! Patterns are referenced by `"pattern": id | name` (document first, then library).

use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Color, Document, Fill, Layer, LayerContent, Pattern, Rect};
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

/// The app-level pattern library (Window › Patterns).
#[derive(Clone, Debug)]
pub struct PatternLibrary {
    pub items: Vec<Pattern>,
}

impl Default for PatternLibrary {
    fn default() -> Self {
        PatternLibrary { items: builtin() }
    }
}

fn hash01(x: u32, y: u32, seed: u32) -> f32 {
    let mut h = x.wrapping_mul(0x8da6_b343) ^ y.wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    (h & 0xffff) as f32 / 65535.0
}

fn procedural(name: &str, w: u32, h: u32, f: impl Fn(u32, u32) -> [f32; 3]) -> Pattern {
    let fmt = PixelFormat::new(ColorMode::Rgb, SampleType::U8, false);
    let mut s = Surface::new(fmt);
    let mut px = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            px.extend_from_slice(&f(x, y));
        }
    }
    // Quantise through the 8-bit surface so ids are stable.
    s.write_region(Rect::new(0, 0, w as i32, h as i32), &px);
    Pattern::new(name, s, w, h)
}

/// Built-in patterns: generated procedurally here (our own, CC0), so the library is never empty.
pub fn builtin() -> Vec<Pattern> {
    let g = |v: f32| [v, v, v];
    vec![
        procedural("Checkerboard", 16, 16, |x, y| g(if (x / 8 + y / 8) % 2 == 0 { 0.8 } else { 1.0 })),
        procedural("Diagonal Lines", 8, 8, |x, y| g(if (x + y) % 8 < 2 { 0.2 } else { 1.0 })),
        procedural("Dots", 12, 12, |x, y| {
            let (dx, dy) = (x as f32 + 0.5 - 6.0, y as f32 + 0.5 - 6.0);
            g(((dx * dx + dy * dy).sqrt() - 3.0).clamp(0.0, 1.0))
        }),
        procedural("Grid", 16, 16, |x, y| g(if x == 0 || y == 0 { 0.55 } else { 1.0 })),
        procedural("Bricks", 32, 16, |x, y| {
            let row = y / 8;
            let xo = (x + if row % 2 == 1 { 8 } else { 0 }) % 16;
            let mortar = y % 8 == 7 || xo == 15;
            let n = hash01(x, y, 7) * 0.08;
            if mortar { [0.78 + n, 0.76 + n, 0.72 + n] } else { [0.62 + n, 0.27 + n * 0.5, 0.2] }
        }),
        procedural("Weave", 16, 16, |x, y| {
            let over = ((x / 4) + (y / 4)) % 2 == 0;
            let t = if over { (y % 4) as f32 / 3.0 } else { (x % 4) as f32 / 3.0 };
            let v = 0.55 + 0.35 * (1.0 - (2.0 * t - 1.0).abs());
            [v, v * 0.92, v * 0.8]
        }),
        procedural("Paper Noise", 64, 64, |x, y| {
            // Two octaves of smoothed value noise (wraps at 64 px).
            let oct = |cell: u32, seed: u32| {
                let (cx, cy) = (x / cell, y / cell);
                let (fx, fy) = ((x % cell) as f32 / cell as f32, (y % cell) as f32 / cell as f32);
                let n = 64 / cell;
                let v = |i: u32, j: u32| hash01(i % n, j % n, seed);
                let s = |t: f32| t * t * (3.0 - 2.0 * t);
                let (sx, sy) = (s(fx), s(fy));
                let a = v(cx, cy) + (v(cx + 1, cy) - v(cx, cy)) * sx;
                let b = v(cx, cy + 1) + (v(cx + 1, cy + 1) - v(cx, cy + 1)) * sx;
                a + (b - a) * sy
            };
            g(0.78 + 0.14 * oct(8, 1) + 0.08 * oct(2, 2))
        }),
    ]
}

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn always(_: &Session) -> std::result::Result<(), String> {
    Ok(())
}

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

fn has_pattern(s: &Session) -> std::result::Result<(), String> {
    if s.patterns.items.is_empty() && s.active().is_none_or(|d| d.doc.patterns.is_empty()) { Err("no patterns".into()) } else { Ok(()) }
}

/// Finds a pattern by id or name: the active document's first, then the library. An empty key
/// picks the first library pattern (Photoshop's default).
pub fn resolve(s: &Session, key: &str) -> Option<Pattern> {
    if key.is_empty() {
        // The Patterns panel's selection, else the first library pattern.
        return crate::presets::patterns::current(s).or(s.patterns.items.first()).cloned();
    }
    if let Some(d) = s.active()
        && let Some(p) = photocraft_doc::pattern::find(&d.doc.patterns, key, key)
    {
        return Some(p.clone());
    }
    photocraft_doc::pattern::find(&s.patterns.items, key, key).cloned()
}

pub(crate) fn resolve_param(s: &Session, cmd: &str, p: &Value) -> Result<Pattern> {
    let key = p.get("pattern").and_then(Value::as_str).unwrap_or("");
    resolve(s, key).ok_or_else(|| bad(cmd, format!("no pattern \"{key}\" (see pattern.list)")))
}

/// Copies `pat` into the document's patterns unless it is already there (by id).
pub fn ensure_in_doc(doc: &mut Document, pat: &Pattern) {
    if !doc.patterns.iter().any(|p| p.id == pat.id) {
        doc.patterns.push(pat.clone());
    }
}

/// Resolves the pattern of a Pattern Overlay / pattern stroke built from params (its `name`
/// holds the requested key) into a real pattern: returns the effect with name and id filled in,
/// plus the pattern to copy into the document.
pub fn resolve_effect(s: &Session, fx: photocraft_doc::Effect) -> Result<(photocraft_doc::Effect, Option<Pattern>)> {
    use photocraft_doc::Effect;
    match fx {
        Effect::PatternOverlay { common, name, id, scale, angle, link, phase } => {
            let key = if id.is_empty() { name } else { id };
            let pat = resolve(s, &key).ok_or_else(|| bad("layer.layerStyle.patternOverlay", format!("no pattern \"{key}\" (see pattern.list)")))?;
            Ok((Effect::PatternOverlay { common, name: pat.name.clone(), id: pat.id.clone(), scale, angle, link, phase }, Some(pat)))
        }
        other => Ok((other, None)),
    }
}

fn info(p: &Pattern, in_doc: bool, in_library: bool) -> Value {
    let f = p.surface.format();
    json!({
        "id": p.id, "name": p.display_name(), "storedName": p.name, "width": p.width, "height": p.height,
        "mode": format!("{:?}", f.mode), "depth": format!("{:?}", f.sample), "alpha": f.alpha,
        "inDocument": in_doc, "inLibrary": in_library,
    })
}

fn list(s: &mut Session, _: &Value) -> Result<Value> {
    let doc_pats: Vec<Pattern> = s.active().map(|d| d.doc.patterns.clone()).unwrap_or_default();
    let mut out: Vec<Value> = s.patterns.items.iter().map(|p| info(p, doc_pats.iter().any(|d| d.id == p.id), true)).collect();
    out.extend(doc_pats.iter().filter(|d| !s.patterns.items.iter().any(|p| p.id == d.id)).map(|p| info(p, true, false)));
    Ok(Value::Array(out))
}

/// The visible composite over `r` as a surface in the document's pixel format (with alpha).
fn composite_surface(doc: &Document, r: Rect) -> Surface {
    let buf = photocraft_compose::render(doc, r);
    let df = doc.pixel_format();
    let fmt = PixelFormat::new(df.mode, df.sample, true);
    let n = fmt.channels();
    let mut px = vec![0.0f32; buf.px.len() * n];
    for (o, p) in px.chunks_exact_mut(n).zip(&buf.px) {
        photocraft_raster::from_rgba_into(&fmt, *p, o);
    }
    let mut s = Surface::new(fmt);
    s.write_region(Rect::new(0, 0, r.width() as i32, r.height() as i32), &px);
    s
}

/// The name Define Pattern gives a pattern without one: the active document's name without its
/// extension ("Pattern" without a document), numbered past the library's existing names. The
/// Define Pattern dialog starts from it.
pub fn default_name(s: &Session) -> String {
    let doc_name = s.active().map(|st| st.doc.name.clone()).unwrap_or_default();
    let base =
        std::path::Path::new(&doc_name).file_stem().map(|x| x.to_string_lossy().to_string()).filter(|b| !b.is_empty()).unwrap_or_else(|| "Pattern".into());
    let taken = |n: &str| s.patterns.items.iter().any(|q| q.name == n);
    if !taken(&base) {
        return base;
    }
    (2..).map(|i| format!("{base} {i}")).find(|n| !taken(n)).unwrap_or(base)
}

fn define(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "edit.definePattern";
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let doc = &st.doc;
    let r = match p.get("rect").and_then(Value::as_array) {
        Some(a) if a.len() == 4 => {
            let v: Vec<i32> = a.iter().map(|x| x.as_f64().unwrap_or(0.0).round() as i32).collect();
            Rect::new(v[0], v[1], v[2], v[3])
        }
        _ => doc.selection.as_ref().map_or(doc.bounds(), |m| m.content_bounds()),
    }
    .intersect(&doc.bounds());
    if r.is_empty() {
        return Err(bad(cmd, "the area is empty"));
    }
    if r.width() > 4000 || r.height() > 4000 {
        return Err(bad(cmd, "patterns are limited to 4000 × 4000 px"));
    }
    let mut surf = composite_surface(doc, r);
    // Opaque patterns drop their alpha channel (as Photoshop does).
    let fmt = surf.format();
    let px = surf.read_region(Rect::new(0, 0, r.width() as i32, r.height() as i32));
    if px.chunks_exact(fmt.channels()).all(|q| q[fmt.channels() - 1] >= 1.0) {
        surf = surf.convert(PixelFormat::new(fmt.mode, fmt.sample, false));
    }
    let name = match p.get("name").and_then(Value::as_str) {
        Some(n) if !n.trim().is_empty() => n.to_string(),
        _ => default_name(s),
    };
    let pat = Pattern::new(name, surf, r.width(), r.height());
    let id = pat.id.clone();
    let out = info(&pat, false, true);
    if !s.patterns.items.iter().any(|q| q.id == id) {
        s.patterns.items.push(pat);
    }
    Ok(json!({"pattern": id, "info": out}))
}

fn library_index(s: &Session, cmd: &str, p: &Value) -> Result<usize> {
    let key = p.get("pattern").and_then(Value::as_str).ok_or_else(|| bad(cmd, "missing `pattern` (id or name)"))?;
    let found =
        photocraft_doc::pattern::find(&s.patterns.items, key, key).map(|f| f.id.clone()).ok_or_else(|| bad(cmd, format!("no library pattern \"{key}\"")))?;
    Ok(s.patterns.items.iter().position(|q| q.id == found).unwrap_or(0))
}

fn rename(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "pattern.rename";
    let name = p.get("name").and_then(Value::as_str).filter(|n| !n.trim().is_empty()).ok_or_else(|| bad(cmd, "missing `name`"))?.to_string();
    let i = library_index(s, cmd, p)?;
    let id = s.patterns.items[i].id.clone();
    s.patterns.items[i].name = name.clone();
    // A copy stored in the document is renamed too (a document edit, so undoable).
    if s.active().is_some_and(|d| d.doc.patterns.iter().any(|q| q.id == id)) {
        s.edit("Rename Pattern", |doc, _| {
            for q in doc.patterns.iter_mut().filter(|q| q.id == id) {
                q.name = name.clone();
            }
            Ok(())
        })?;
    }
    Ok(json!({"pattern": id, "name": name}))
}

fn delete(s: &mut Session, p: &Value) -> Result<Value> {
    let i = library_index(s, "pattern.delete", p)?;
    let pat = s.patterns.items.remove(i);
    Ok(json!({"deleted": pat.id}))
}

fn import(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "pattern.import";
    // A file on disk, or its bytes (a dropped file, the web build).
    let bytes = match (p.get("path").and_then(Value::as_str), p.get("data").and_then(Value::as_str)) {
        (Some(path), _) => std::fs::read(path).map_err(|e| EngineError::Other(format!("{path}: {e}")))?,
        (None, Some(b64)) => photocraft_paint::tile::b64_decode(b64).ok_or_else(|| bad(cmd, "`data` is not base64"))?,
        (None, None) => return Err(bad(cmd, "missing `path` (.pat file) or `data`")),
    };
    let pats = photocraft_io::pattern_map::read_pat(&bytes).map_err(EngineError::Other)?;
    let mut ids = Vec::new();
    for pat in pats {
        ids.push(pat.id.clone());
        match s.patterns.items.iter_mut().find(|q| q.id == pat.id) {
            Some(q) => *q = pat,
            None => s.patterns.items.push(pat),
        }
    }
    Ok(json!({"imported": ids, "count": ids.len()}))
}

fn export(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "pattern.export";
    let path = p.get("path").and_then(Value::as_str).ok_or_else(|| bad(cmd, "missing `path`"))?;
    let pats: Vec<Pattern> = match p.get("patterns").and_then(Value::as_array) {
        Some(keys) => {
            keys.iter().filter_map(Value::as_str).map(|k| resolve(s, k).ok_or_else(|| bad(cmd, format!("no pattern \"{k}\"")))).collect::<Result<_>>()?
        }
        None => s.patterns.items.clone(),
    };
    let bytes = photocraft_io::pattern_map::write_pat(&pats).map_err(EngineError::Other)?;
    crate::file_cmds::write_file(path, &bytes)?;
    Ok(json!({"path": path, "count": pats.len(), "bytes": bytes.len()}))
}

pub(crate) fn placement(p: &Value) -> (f32, f32, bool, (f32, f32)) {
    let num = |k: &str, d: f64| p.get(k).and_then(Value::as_f64).unwrap_or(d) as f32;
    let phase = p
        .get("phase")
        .and_then(Value::as_array)
        .map_or((0.0, 0.0), |a| (a.first().and_then(Value::as_f64).unwrap_or(0.0) as f32, a.get(1).and_then(Value::as_f64).unwrap_or(0.0) as f32));
    ((num("scale", 100.0) / 100.0).clamp(0.01, 10.0), num("angle", 0.0), p.get("link").and_then(Value::as_bool).unwrap_or(true), phase)
}

/// The edit of `layer.newFillLayer.pattern`: a Pattern Fill layer of `pat` placed as `p` asks,
/// above `active`. The Pattern Fill dialog previews it on a copy of the document.
pub(crate) fn insert_fill_layer(doc: &mut Document, active: &mut Option<photocraft_doc::LayerId>, pat: &Pattern, p: &Value) -> Result<photocraft_doc::LayerId> {
    let (scale, angle, link, phase) = placement(p);
    ensure_in_doc(doc, pat);
    let fill = Fill::Pattern { name: pat.name.clone(), scale, id: pat.id.clone(), angle, link, phase };
    let mut l = Layer::new(doc.next_layer_name("Pattern Fill"), LayerContent::Fill(fill));
    crate::commands::mask_new_layer(doc, &mut l, p, "layer.newFillLayer.pattern")?;
    let id = doc.insert_above(*active, l);
    *active = Some(id);
    Ok(id)
}

fn new_fill_layer(s: &mut Session, p: &Value) -> Result<Value> {
    let pat = resolve_param(s, "layer.newFillLayer.pattern", p)?;
    let id = s.edit("New Pattern Fill Layer", |doc, active| insert_fill_layer(doc, active, &pat, p))?;
    Ok(json!({"layer": id.0, "pattern": pat.id}))
}

fn brush_texture(s: &mut Session, p: &Value) -> Result<Value> {
    let pat = resolve_param(s, "brush.texturePattern", p)?;
    let n = pat.is_paintable().then(|| usize::try_from(pat.pixel_count()).ok()).flatten();
    let Some(n) = n else {
        return Err(bad("brush.texturePattern", "the pattern is empty or too large to paint"));
    };
    // Brush textures are grey: luminance of the composite over white.
    let mut px = vec![[0.0f32; 4]; n];
    pat.surface.read_rgba_into(pat.rect(), &mut px);
    let lum: Vec<f32> = px
        .iter()
        .map(|q| {
            let l = 0.299 * q[0] + 0.587 * q[1] + 0.114 * q[2];
            l * q[3] + (1.0 - q[3])
        })
        .collect();
    let tile = photocraft_paint::GrayTile::from_f32(pat.width, pat.height, &lum);
    let t = &mut s.tools.brush.texture;
    t.pattern = photocraft_paint::Pattern::Tile(tile);
    t.enabled = p.get("enabled").and_then(Value::as_bool).unwrap_or(true);
    if let Some(sc) = p.get("scale").and_then(Value::as_f64) {
        t.scale = (sc as f32 / 100.0).clamp(0.01, 10.0);
    }
    Ok(json!({"pattern": pat.id}))
}

/// A swatch colour for UIs (average of the tile), straight RGB.
pub fn average_color(p: &Pattern) -> Color {
    // A pattern too large to paint is averaged over its paintable top-left corner.
    let side = photocraft_doc::pattern::MAX_PAINT_PIXELS.isqrt() as u32;
    let (w, h) = (p.width.min(side), p.height.min(side));
    let mut px = vec![[0.0f32; 4]; w as usize * h as usize];
    p.surface.read_rgba_into(Rect::new(0, 0, w as i32, h as i32), &mut px);
    let n = px.len().max(1) as f32;
    let s = px.iter().fold([0.0f32; 3], |a, q| [a[0] + q[0], a[1] + q[1], a[2] + q[2]]);
    Color::rgb(s[0] / n, s[1] / n, s[2] / n)
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "edit.definePattern",
            label: "Define Pattern…",
            menu: &["Edit"],
            shortcut: None,
            params: r##"{"name":text,"rect":[x0,y0,x1,y1]?} → {"pattern":id} (name: default the document's name numbered past the library's names; rect: default the selection bounds or else the canvas; added to the library from the visible composite)"##,
            enabled: has_doc,
            journal: true,
            run: define,
        },
        CommandSpec {
            id: "pattern.list",
            label: "Patterns",
            menu: &[],
            shortcut: None,
            params: "{} → [{id,name,width,height,mode,depth,inDocument,inLibrary}]",
            enabled: always,
            journal: false,
            run: list,
        },
        CommandSpec {
            id: "pattern.rename",
            label: "Rename Pattern",
            menu: &[],
            shortcut: None,
            params: r##"{"pattern":id|name,"name":str}"##,
            enabled: has_pattern,
            journal: true,
            run: rename,
        },
        CommandSpec {
            id: "pattern.delete",
            label: "Delete Pattern",
            menu: &[],
            shortcut: None,
            params: r##"{"pattern":id|name} (library only; documents keep their copy)"##,
            enabled: has_pattern,
            journal: true,
            run: delete,
        },
        CommandSpec {
            id: "pattern.import",
            label: "Import Patterns…",
            menu: &[],
            shortcut: None,
            params: r##"{"path":".pat file"?,"data":base64 bytes? (instead of a path)} → {imported:[ids],count}"##,
            enabled: always,
            journal: true,
            run: import,
        },
        CommandSpec {
            id: "pattern.export",
            label: "Export Patterns…",
            menu: &[],
            shortcut: None,
            params: r##"{"path":".pat file","patterns":[id|name]? (default: the whole library)}"##,
            enabled: has_pattern,
            journal: true,
            run: export,
        },
        CommandSpec {
            id: "layer.newFillLayer.pattern",
            label: "Pattern…",
            menu: &["Layer", "New Fill Layer"],
            shortcut: None,
            params: r##"{"pattern":id|name?=first library pattern,"scale":1..1000=100,"angle":deg=0,"link":bool=true,"phase":[x,y]?,"path":"work"|saved path name|{…path}? (the active path: becomes the vector mask; else the selection is the layer mask)}"##,
            enabled: has_doc,
            journal: true,
            run: new_fill_layer,
        },
        CommandSpec {
            id: "brush.texturePattern",
            label: "Brush Texture Pattern",
            menu: &[],
            shortcut: None,
            params: r##"{"pattern":id|name,"scale":1..1000?,"enabled":bool=true} (sets the session brush's Texture to that pattern)"##,
            enabled: has_pattern,
            journal: true,
            run: brush_texture,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_doc::LayerId;

    fn session(depth: u64) -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 40, "height": 30, "depth": depth})).unwrap();
        s
    }

    fn pixel(s: &Session, x: i32, y: i32) -> [f32; 4] {
        let b = photocraft_compose::render(&s.active().unwrap().doc, Rect::new(x, y, x + 1, y + 1));
        b.px[0]
    }

    #[test]
    fn library_has_builtins_and_lists() {
        let mut s = Session::new();
        let v = s.execute("pattern.list", json!({})).unwrap();
        assert!(v.as_array().unwrap().len() >= 6);
        assert!(v[0]["id"].as_str().unwrap().len() == 36);
        // Built-in ids are stable across sessions.
        assert_eq!(builtin()[0].id, Session::new().patterns.items[0].id);
    }

    #[test]
    fn define_rename_delete_and_pat_round_trip() {
        let mut s = session(8);
        s.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
        s.execute("select.rect", json!({"x": 2, "y": 3, "width": 5, "height": 4})).unwrap();
        let r = s.execute("edit.definePattern", json!({"name": "Red"})).unwrap();
        let id = r["pattern"].as_str().unwrap().to_string();
        let pat = resolve(&s, &id).unwrap();
        assert_eq!((pat.width, pat.height), (5, 4));
        assert!(!pat.surface.format().alpha, "opaque pattern drops alpha");
        // Same content → same id (replayable); not duplicated.
        let n = s.patterns.items.len();
        s.execute("edit.definePattern", json!({"name": "Red"})).unwrap();
        assert_eq!(s.patterns.items.len(), n);
        s.execute("pattern.rename", json!({"pattern": id, "name": "Crimson"})).unwrap();
        assert_eq!(resolve(&s, "Crimson").unwrap().id, id);
        let dir = std::env::temp_dir().join(format!("pc-pat-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lib.pat");
        let path = path.to_str().unwrap();
        s.execute("pattern.export", json!({"path": path})).unwrap();
        s.execute("pattern.delete", json!({"pattern": "Crimson"})).unwrap();
        assert!(resolve(&s, "Crimson").is_none());
        assert!(s.execute("pattern.delete", json!({"pattern": "Crimson"})).is_err());
        let v = s.execute("pattern.import", json!({"path": path})).unwrap();
        assert!(v["imported"].as_array().unwrap().iter().any(|x| x == id.as_str()));
        assert_eq!(resolve(&s, &id).unwrap().surface.read_region(Rect::new(0, 0, 5, 4)), pat.surface.read_region(Rect::new(0, 0, 5, 4)));
        // The same file as bytes (a dropped file, the web build).
        let bytes = std::fs::read(path).unwrap();
        s.execute("pattern.delete", json!({"pattern": "Crimson"})).unwrap();
        let v = s.execute("pattern.import", json!({"data": photocraft_paint::tile::b64_encode(&bytes)})).unwrap();
        assert_eq!(v["count"], v["imported"].as_array().unwrap().len());
        assert_eq!(resolve(&s, "Crimson").unwrap().id, id);
        std::fs::remove_dir_all(dir).ok();
        assert!(s.execute("pattern.import", json!({"path": "/nonexistent.pat"})).is_err());
        for bad in [json!({}), json!({"data": "not base64!"}), json!({"data": "aGVsbG8="})] {
            assert!(s.execute("pattern.import", bad.clone()).is_err(), "{bad}");
        }
    }

    #[test]
    fn pattern_fill_layer_renders_at_every_depth_and_undoes() {
        for depth in [8u64, 16, 32] {
            let mut s = session(depth);
            let r = s.execute("layer.newFillLayer.pattern", json!({"pattern": "Checkerboard"})).unwrap();
            let id = LayerId(r["layer"].as_u64().unwrap());
            let st = s.active().unwrap();
            assert_eq!(st.doc.patterns.len(), 1, "copied into the document");
            assert!(matches!(&st.doc.layer(id).unwrap().content, LayerContent::Fill(Fill::Pattern { .. })));
            // Checkerboard: 8 px squares of 0.8 and 1.0 grey from the canvas origin.
            let (a, b) = (pixel(&s, 1, 1), pixel(&s, 9, 1));
            assert!((a[0] - 0.8).abs() < 0.01 && (b[0] - 1.0).abs() < 0.01, "{depth}: {a:?} {b:?}");
            s.undo();
            assert!(s.active().unwrap().doc.patterns.is_empty());
            assert!(s.execute("layer.newFillLayer.pattern", json!({"pattern": "nope"})).is_err());
        }
    }

    #[test]
    fn fill_with_pattern_respects_selection() {
        for depth in [8u64, 16, 32] {
            let mut s = session(depth);
            s.execute("select.rect", json!({"x": 0, "y": 0, "width": 8, "height": 8})).unwrap();
            s.execute("edit.fill", json!({"contents": "pattern", "pattern": "Diagonal Lines"})).unwrap();
            let st = s.active().unwrap();
            let surf = st.doc.layers[0].surface().unwrap();
            assert!(surf.pixel(0, 0)[0] < 0.3, "{depth}: line pixel");
            assert!(surf.pixel(4, 0)[0] > 0.9);
            assert!(surf.pixel(20, 20)[0] > 0.99, "outside the selection untouched");
        }
    }

    #[test]
    fn pattern_overlay_composes_with_an_outside_stroke() {
        let mut s = session(8);
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("select.rect", json!({"x": 10, "y": 10, "width": 10, "height": 10})).unwrap();
        s.execute("edit.fill", json!({"color": "#0000ff"})).unwrap();
        s.execute("select.deselect", json!({})).unwrap();
        s.execute("layer.layerStyle.patternOverlay", json!({"pattern": "Diagonal Lines"})).unwrap();
        s.execute("layer.layerStyle.stroke", json!({"size": 3, "color": "#00ff00", "add": true})).unwrap();
        let out = pixel(&s, 8, 15);
        assert!(out[1] > 0.9 && out[0] < 0.1, "stroke outside: {out:?}");
    }

    #[test]
    fn pattern_overlay_renders_and_brush_texture_uses_library() {
        let mut s = session(16);
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("edit.fill", json!({"color": "#0000ff"})).unwrap();
        s.execute("layer.layerStyle.patternOverlay", json!({"pattern": "Checkerboard", "opacity": 100})).unwrap();
        let st = s.active().unwrap();
        assert_eq!(st.doc.patterns.len(), 1);
        let p = pixel(&s, 1, 1);
        assert!((p[0] - 0.8).abs() < 0.01 && (p[2] - 0.8).abs() < 0.01, "{p:?}");
        assert!(s.execute("layer.layerStyle.patternOverlay", json!({"pattern": "missing"})).is_err());
        s.execute("layer.layerStyle.stroke", json!({"size": 4, "color": "#ffffff", "add": true})).unwrap();
        let st = s.active().unwrap();
        assert_eq!(st.doc.layer(st.active_layer.unwrap()).unwrap().effects.items.len(), 2);
        s.execute("brush.texturePattern", json!({"pattern": "Dots", "scale": 50})).unwrap();
        assert!(s.tools.brush.texture.enabled);
        assert!(matches!(s.tools.brush.texture.pattern, photocraft_paint::Pattern::Tile(_)));
    }

    #[test]
    fn a_pattern_too_large_to_paint_is_refused_not_allocated() {
        // width × height overflows u32: this panicked (debug) or wrapped (release).
        let mut s = session(8);
        let huge = Pattern { id: "huge".into(), name: "Huge".into(), width: 65_536, height: 65_536, surface: Surface::new(PixelFormat::RGBA8) };
        s.patterns.items.push(huge);
        let e = s.execute("brush.texturePattern", json!({"pattern": "huge"})).unwrap_err();
        assert!(e.to_string().contains("too large"), "{e}");
    }
}
