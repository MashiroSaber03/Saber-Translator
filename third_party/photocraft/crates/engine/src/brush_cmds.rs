//! Brush engine commands: the Brush/Pencil/Mixer Brush/Color Replacement tools, brush presets and
//! the session brush (`tools.setBrush`). The UI's options bar and Brush Settings panel call these.
//!
//! Brushes are [`BrushSettings`] serialised as camelCase JSON; any command taking a `"brush"`
//! object deep-merges it onto the session brush, so partial objects work
//! (`{"scattering": {"enabled": true}}`). Strokes are deterministic: the jitter seed is the `seed`
//! param or a hash of the points, so a journaled command replays to identical pixels.

use photocraft_doc::LayerContent;
use photocraft_geom::Rect;
use photocraft_paint::mixer::{MixerSettings, apply_mixer_stroke};
use photocraft_paint::replace::{Limits, ReplaceMode, ReplaceSettings, Sampling, apply_color_replacement, color_replacement_effect};
use photocraft_paint::tile::PREVIEW_SIDE;
use photocraft_paint::{
    BrushPreset, BrushSettings, GrayTile, Pattern, PresetCaptures, StoredTile, Stroke, StrokePoint, StrokeRenderer, TipShape, render_stroke,
};
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, has_paintable, is_mask_target, paint_surface};
use crate::{EngineError, Result, Session};

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}
fn num(p: &Value, k: &str) -> Option<f32> {
    p.get(k).and_then(Value::as_f64).map(|v| v as f32)
}
fn flag(p: &Value, k: &str, d: bool) -> bool {
    p.get(k).and_then(Value::as_bool).unwrap_or(d)
}
fn color(v: Option<&Value>, d: [f32; 4]) -> [f32; 4] {
    match v {
        Some(Value::Array(a)) if a.len() >= 3 => {
            let c: Vec<f32> = a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
            [c[0], c[1], c[2], c.get(3).copied().unwrap_or(1.0)]
        }
        Some(Value::String(h)) => {
            let h = h.trim_start_matches('#');
            let c = |i: usize| h.get(i..i + 2).and_then(|x| u8::from_str_radix(x, 16).ok()).map(|v| f32::from(v) / 255.0);
            match (c(0), c(2), c(4)) {
                (Some(r), Some(g), Some(b)) if h.len() == 6 || h.len() == 8 => [r, g, b, c(6).unwrap_or(1.0)],
                _ => d,
            }
        }
        _ => d,
    }
}
fn always(_: &Session) -> std::result::Result<(), String> {
    Ok(())
}

/// Largest stroke coordinate accepted (a few times the largest document side, 300 000 px).
pub(crate) const MAX_COORD: f64 = 1_000_000.0;

/// Parse `points`: arrays `[x, y, pressure?, tiltX?, tiltY?, rotation?, timeMs?, wheel?]` or
/// objects `{"x":…, "y":…, "pressure":…, "tiltX":…, …, "time":…}`.
pub fn parse_points(p: &Value, cmd: &str) -> Result<Vec<StrokePoint>> {
    let arr = p.get("points").and_then(Value::as_array).ok_or_else(|| bad(cmd, "missing `points`"))?;
    let pts: Vec<StrokePoint> = arr
        .iter()
        .filter_map(|v| match v {
            Value::Array(a) => {
                let g = |i: usize| a.get(i).and_then(Value::as_f64);
                let mut sp = StrokePoint::new(g(0)?, g(1)?, g(2).unwrap_or(1.0) as f32);
                sp.tilt_x = g(3).unwrap_or(0.0) as f32;
                sp.tilt_y = g(4).unwrap_or(0.0) as f32;
                sp.rotation = g(5).unwrap_or(0.0) as f32;
                sp.time = g(6).unwrap_or(0.0);
                sp.wheel = g(7).unwrap_or(1.0) as f32;
                Some(sp)
            }
            Value::Object(_) => serde_json::from_value(v.clone()).ok(),
            _ => None,
        })
        .collect();
    if pts.is_empty() {
        return Err(bad(cmd, "`points` is empty"));
    }
    check_coords(&pts, cmd)?;
    Ok(pts)
}

/// A stroke runs dab by dab along its length: an absurd coordinate would mean billions of dabs.
/// Shared by the painting and the retouch tools' `points` (#976).
pub(crate) fn check_coords(pts: &[StrokePoint], cmd: &str) -> Result<()> {
    if pts.iter().any(|q| !(q.x.abs() <= MAX_COORD && q.y.abs() <= MAX_COORD)) {
        return Err(bad(cmd, format!("point coordinates must be finite and within ±{MAX_COORD}")));
    }
    Ok(())
}

/// Deep-merge `patch` into `base` (objects merge key by key; anything else replaces).
fn merge(base: &mut Value, patch: &Value) {
    match (base, patch) {
        // A different variant of an externally tagged enum (`{"tile":…}` → `{"procedural":…}`)
        // replaces the old one instead of merging into a two-variant object.
        (Value::Object(b), Value::Object(p)) if b.len() == 1 && p.len() == 1 && b.keys().next() != p.keys().next() => {
            *b = p.clone();
        }
        (Value::Object(b), Value::Object(p)) => {
            for (k, v) in p {
                match b.get_mut(k) {
                    Some(bv) if bv.is_object() && v.is_object() => merge(bv, v),
                    _ => {
                        b.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (b, p) => *b = p.clone(),
    }
}

/// Apply a JSON brush patch onto a brush.
pub fn merge_brush(base: &BrushSettings, patch: &Value, cmd: &str) -> Result<BrushSettings> {
    // Bitmaps the patch doesn't touch (sampled tips, pattern tiles: up to megabytes of base64) skip
    // the JSON round trip, so a Brush Settings slider on an imported tip stays cheap.
    let touches = |path: &[&str]| {
        let mut v = patch;
        for k in path {
            match v.get(k) {
                Some(x) => v = x,
                // A non-object on the way replaces the whole section.
                None => return !v.is_object(),
            }
        }
        true
    };
    let mut light = base.clone();
    let tip = (!touches(&["tip"])).then(|| std::mem::take(&mut light.tip));
    let dual_tip = (!touches(&["dualBrush", "tip"])).then(|| std::mem::take(&mut light.dual_brush.tip));
    let pattern = (!touches(&["texture", "pattern"])).then(|| std::mem::take(&mut light.texture.pattern));
    let mut v = serde_json::to_value(&light).map_err(|e| bad(cmd, e.to_string()))?;
    merge(&mut v, patch);
    let mut out: BrushSettings = serde_json::from_value(v).map_err(|e| bad(cmd, format!("invalid brush: {e}")))?;
    if let Some(t) = tip {
        out.tip = t;
    }
    if let Some(t) = dual_tip {
        out.dual_brush.tip = t;
    }
    if let Some(p) = pattern {
        out.texture.pattern = p;
    }
    Ok(out)
}

/// Rejects brush sizes and Scatter amounts outside their documented ranges, before a brush is
/// stored or painted with.
pub(crate) fn validate_brush(brush: &BrushSettings, cmd: &str) -> Result<()> {
    let max = photocraft_paint::MAX_BRUSH_SIZE;
    if !brush.size.is_finite() || brush.size > max {
        return Err(bad(cmd, format!("brush size must be finite and at most {max} px")));
    }
    if brush.dual_brush.enabled && (!brush.dual_brush.size.is_finite() || brush.dual_brush.size > max) {
        return Err(bad(cmd, format!("dual brush size must be finite and at most {max} px")));
    }
    // Scatter may exceed 1 (100 %) but not the 1000 % maximum: a huge amount pushed dab centres to
    // infinity (#977). `{:?}` keeps 1e38 short and shows infinity as `inf`.
    let scatter_max = photocraft_paint::MAX_SCATTER;
    let in_range = |v: f32| (0.0..=scatter_max).contains(&v);
    let sc = brush.scattering.scatter.jitter;
    if brush.scattering.enabled && !in_range(sc) {
        return Err(bad(cmd, format!("scatter amount {sc:?} is out of range: it must be between 0 and {scatter_max} (1000 %)")));
    }
    let ds = brush.dual_brush.scatter;
    if brush.dual_brush.enabled && !in_range(ds) {
        return Err(bad(cmd, format!("dual brush scatter amount {ds:?} is out of range: it must be between 0 and {scatter_max} (1000 %)")));
    }
    Ok(())
}

fn find_preset<'a>(s: &'a Session, name: &str, cmd: &str) -> Result<&'a BrushPreset> {
    photocraft_paint::presets::find(&s.tools.presets, name).ok_or_else(|| bad(cmd, format!("no brush preset named `{name}`")))
}

/// Resolve the brush for a stroke command: session brush ← `preset` ← `brush` ← legacy scalar
/// params; colours from the session; seed from `seed` or the points.
pub fn resolve_brush(s: &Session, p: &Value, cmd: &str) -> Result<BrushSettings> {
    let mut b = s.tools.brush.clone();
    if let Some(name) = p.get("preset").and_then(Value::as_str) {
        b = find_preset(s, name, cmd)?.picked_over(&s.tools.brush);
    }
    if let Some(patch) = p.get("brush").filter(|v| v.is_object()) {
        b = merge_brush(&b, patch, cmd)?;
    }
    // A library preset's tips stay on disk until a stroke uses them (#1843).
    s.load_brush_tips(&mut b).map_err(|e| bad(cmd, e))?;
    if let Some(v) = num(p, "size") {
        b.size = v.max(0.5);
    }
    if let Some(v) = num(p, "hardness") {
        b.hardness = v.clamp(0.0, 1.0);
    }
    if let Some(v) = num(p, "opacity") {
        b.opacity = v.clamp(0.0, 1.0);
    }
    if let Some(v) = num(p, "flow") {
        b.flow = v.clamp(0.0, 1.0);
    }
    if let Some(v) = num(p, "spacing") {
        b.spacing = v.clamp(0.01, 10.0);
    }
    if let Some(v) = num(p, "smoothing") {
        b.smoothing.enabled = true;
        b.smoothing.amount = v.clamp(0.0, 1.0);
    }
    b.color = color(p.get("color"), s.tools.foreground);
    b.color_cmyk = s.tools.picked_cmyk([b.color[0], b.color[1], b.color[2]]).map(|k| ([b.color[0], b.color[1], b.color[2]], k));
    b.background = s.tools.background;
    b.erase = flag(p, "erase", false);
    b.seed = match p.get("seed").and_then(Value::as_u64) {
        Some(v) => v,
        None => photocraft_paint::rng::seed_from_bytes(p.get("points").map(|v| v.to_string()).unwrap_or_default().as_bytes()),
    };
    validate_brush(&b, cmd)?;
    Ok(b)
}

fn layer_id(s: &Session, p: &Value) -> Result<photocraft_doc::LayerId> {
    match p.get("layer").and_then(Value::as_u64) {
        Some(id) => Ok(photocraft_doc::LayerId(id)),
        None => s.active().and_then(|d| d.active_layer).ok_or(EngineError::Other("no active layer".into())),
    }
}

fn damage_json(s: &mut Session, dmg: Rect) -> Value {
    if let Some(st) = s.active_mut() {
        st.last_damage = Some(dmg);
    }
    json!({ "damage": [dmg.x0, dmg.y0, dmg.width(), dmg.height()] })
}

/// The target layer (`None` for a channel), brush and zoom of a stroke. On a mask or channel the
/// eraser paints the background colour (Photoshop).
fn stroke_target(s: &Session, p: &Value, brush: BrushSettings) -> Result<(Option<photocraft_doc::LayerId>, BrushSettings, f32)> {
    let gray = is_mask_target(p) || crate::channel_cmds::is_channel_target(p);
    let id = if crate::channel_cmds::is_channel_target(p) { None } else { Some(layer_id(s, p)?) };
    let brush = if gray && brush.erase { BrushSettings { erase: false, color: s.tools.background, ..brush } } else { brush };
    Ok((id, brush, num(p, "zoom").unwrap_or(1.0)))
}

/// Erasing a layer with locked transparency (e.g. the Background) can't remove opacity, so it
/// paints the background colour instead (Photoshop).
fn erase_locked(brush: &mut BrushSettings, lock: bool, bg: [f32; 4]) {
    if brush.erase && lock {
        brush.erase = false;
        brush.color = bg;
    }
}

/// The document's own CMYK profile (absent, built-in or not a CMYK document: `None`, the
/// default conversions), which the stroke's colour conversions must honour — the compositors
/// and composite exports enter the same scope.
pub(crate) fn doc_cmyk_space(s: &Session) -> Option<std::sync::Arc<photocraft_color::convert::CmykSpace>> {
    photocraft_color::convert::CmykSpace::for_profile(s.active().and_then(|d| d.doc.icc_profile.as_ref()))
}

/// Stroke with a resolved brush onto the target layer (pixels or mask).
fn stroke_with(s: &mut Session, cmd: &str, p: &Value, label: &str, brush: BrushSettings, pts: Vec<StrokePoint>, auto_erase: bool) -> Result<Value> {
    let bg = s.tools.background;
    let fg = brush.color;
    let symmetry = s.active().and_then(|st| st.symmetry.clone());
    let (id, brush, zoom) = stroke_target(s, p, brush)?;
    let cmyk = doc_cmyk_space(s);
    // The live stroke the canvas already rendered for exactly these params: take its pixels
    // instead of rendering the whole stroke again (the hitch at pen-up).
    let live = s.live_commit.take().filter(|l| l.matches(s, cmd, p, &pts));
    let from_live = live.is_some();
    let dmg = photocraft_color::convert::with_cmyk_space(cmyk.as_ref(), || {
        s.edit(label, |doc, _| {
            if let Some(l) = live {
                let (surf, _) = crate::channel_cmds::target_surface(doc, id, p)?;
                *surf = l.surface;
                return Ok(l.bounds);
            }
            let sel = doc.selection.clone();
            let (surf, lock) = crate::channel_cmds::target_surface(doc, id, p)?;
            let mut brush = brush;
            erase_locked(&mut brush, lock, bg);
            if auto_erase {
                apply_auto_erase(&mut brush, surf, pts.first(), fg, bg);
            }
            let damage = if let Some(axis) = &symmetry {
                let reflected = axis.reflected_passes(&pts);
                let copies: Vec<_> = reflected
                    .iter()
                    .filter(|p| crate::symmetry_cmds::SymmetryAxis::has_distinct_mirror(&pts, p))
                    .map(|p| {
                        let mut renderer = StrokeRenderer::new(&brush, Some(surf.format()), zoom);
                        renderer.push(p);
                        renderer.finish();
                        renderer
                    })
                    .collect();
                if copies.is_empty() {
                    render_stroke(surf, &brush, &pts, sel.as_ref(), lock, zoom)
                } else {
                    let pre = surf.clone();
                    let mut original = StrokeRenderer::new(&brush, Some(surf.format()), zoom);
                    original.push(&pts);
                    original.finish();
                    original.composite_union_many(&copies, &pre, surf, sel.as_ref(), lock)
                }
            } else {
                render_stroke(surf, &brush, &pts, sel.as_ref(), lock, zoom)
            };
            Ok(damage)
        })
    })?;
    let mut r = damage_json(s, dmg);
    if from_live {
        r["fromLive"] = json!(true);
    }
    Ok(r)
}

/// Pencil Auto Erase: a stroke that starts on a pixel of the foreground colour paints the
/// background colour instead (Photoshop).
fn apply_auto_erase(brush: &mut BrushSettings, surf: &Surface, start: Option<&StrokePoint>, fg: [f32; 4], bg: [f32; 4]) {
    let Some(p0) = start else { return };
    let c = surf.rgba(p0.x.floor() as i32, p0.y.floor() as i32);
    if c[3] > 0.0 && (0..3).all(|i| (c[i] - fg[i]).abs() < 1.5 / 255.0) {
        brush.color = bg;
    }
}

/// The Pencil's brush: the session brush made aliased (every pixel fully painted or untouched,
/// dabs on the pixel grid), hard and at full flow, with the options-bar mode.
fn pencil_brush(s: &Session, p: &Value) -> Result<BrushSettings> {
    let mut brush = with_blend_mode(resolve_brush(s, p, "paint.pencil")?, p);
    if flag(p, "block", false) {
        return block_brush(brush, p);
    }
    brush.aliased = true;
    if num(p, "hardness").is_none() {
        brush.hardness = 1.0;
    }
    if num(p, "flow").is_none() {
        brush.flow = 1.0;
    }
    Ok(brush)
}

/// Edge of the Eraser's Block in screen pixels (Photoshop): constant on screen, so it covers
/// `BLOCK_SCREEN_PX / zoom` document pixels.
pub const BLOCK_SCREEN_PX: f64 = 16.0;

/// The Eraser's Block edge in document pixels at view `zoom`: `16 / zoom`, rounded to whole
/// pixels (a zoom that is not a positive number counts as 100 %).
pub fn block_size(zoom: f64) -> f32 {
    let zoom = if zoom.is_finite() && zoom > 0.0 { zoom } else { 1.0 };
    (BLOCK_SCREEN_PX / zoom).round().clamp(1.0, f64::from(photocraft_paint::brush::MAX_BRUSH_SIZE)) as f32
}

/// The Eraser's Block mode (`"block": true`): a hard, aliased square of
/// [`BLOCK_SCREEN_PX`] screen pixels at the stroke's `"zoom"`, i.e. `16 / zoom` document
/// pixels rounded to whole pixels. Size, opacity, flow, hardness and the brush's dynamics don't
/// apply (Photoshop); colour, erase, smoothing and the seed are the resolved brush's.
fn block_brush(b: BrushSettings, p: &Value) -> Result<BrushSettings> {
    let zoom = match p.get("zoom") {
        None => 1.0,
        Some(v) => v.as_f64().filter(|z| z.is_finite() && *z > 0.0).ok_or_else(|| bad("paint.pencil", "`zoom` must be a positive number"))?,
    };
    Ok(BrushSettings {
        size: block_size(zoom),
        aliased: true,
        square: true,
        pressure_size: false,
        pressure_opacity: false,
        spacing: 0.25,
        color: b.color,
        color_cmyk: b.color_cmyk,
        background: b.background,
        erase: b.erase,
        smoothing: b.smoothing,
        seed: b.seed,
        ..BrushSettings::default()
    })
}

/// Is the stroke in Photoshop's Clear painting mode (`"mode": "clear"`, #3439)? It erases with the
/// Brush's or Pencil's own brush; unlike a blend mode it is not a layer mode, so it is not a
/// [`photocraft_color::BlendMode`].
fn clear_mode(p: &Value) -> bool {
    p.get("mode").and_then(Value::as_str).is_some_and(|m| m.eq_ignore_ascii_case("clear"))
}

/// Applies the options-bar blend `mode` to a brush. `"mode"` accepts any blend-mode name
/// (normal|multiply|screen|…), or `clear` to erase with the brush. The Eraser has no blend mode in
/// Photoshop, so it is forced to Normal.
fn with_blend_mode(mut b: BrushSettings, p: &Value) -> BrushSettings {
    b.erase |= clear_mode(p);
    if b.erase {
        b.mode = photocraft_color::BlendMode::Normal;
    } else if let Some(m) = p.get("mode").and_then(Value::as_str).and_then(crate::commands::blend_from_str) {
        b.mode = m;
    }
    b
}

/// `paint.stroke`: the Brush (and Eraser) tool.
pub fn paint_stroke(s: &mut Session, p: &Value) -> Result<Value> {
    let pts = parse_points(p, "paint.stroke")?;
    let brush = with_blend_mode(resolve_brush(s, p, "paint.stroke")?, p);
    let label = if brush.erase && !clear_mode(p) { "Eraser" } else { "Brush Tool" };
    stroke_with(s, "paint.stroke", p, label, brush, pts, false)
}

/// A `paint.stroke` rendered while it is drawn, onto a copy of the active document, so the canvas
/// shows the real dabs before the stroke commits. Committing the same params and points with
/// `"seed": live.seed` gives the same pixels.
pub struct LiveStroke {
    /// The active document with the stroke so far.
    pub doc: std::sync::Arc<photocraft_doc::Document>,
    /// Jitter seed to pass to `paint.stroke`.
    pub seed: u64,
    cmd: String,
    renderer: StrokeRenderer,
    symmetry: Option<crate::symmetry_cmds::PaintingSymmetry>,
    mirrors: Vec<StrokeRenderer>,
    mirror_distinct: Vec<bool>,
    /// What the symmetric preview drew last, once a mirror differs from the stroke, and how many
    /// passes it merged (a mirror joining redraws every tile).
    union_live: Option<(photocraft_paint::UnionLive, usize)>,
    pre: Surface,
    sel: Option<Surface>,
    lock: bool,
    layer: Option<photocraft_doc::LayerId>,
    params: Value,
    /// Where the doc shows the stroke's end as finishing it would draw it (see `push`).
    tail: Rect,
    /// The document's CMYK profile the commit enters (`stroke_with`), so the preview shows the
    /// pixels a `paint.stroke` with `seed` will write.
    cmyk: Option<std::sync::Arc<photocraft_color::convert::CmykSpace>>,
    /// Every point rendered so far, in order.
    points: Vec<StrokePoint>,
    /// The document and revision the stroke started on.
    base: (photocraft_doc::DocId, u64),
}

/// A finished live stroke offered to its own commit ([`Session::offer_live_stroke`],
/// [`Session::offer_live_retouch`]): when the committing command matches it (same command,
/// document revision, params, seed and points), it takes these pixels instead of rendering the
/// stroke again.
pub struct LiveCommit {
    pub(crate) cmd: String,
    pub(crate) base: (photocraft_doc::DocId, u64),
    pub(crate) params: Value,
    /// The Brush's and Pencil's seed; the retouching tools have none.
    pub(crate) seed: Option<u64>,
    pub(crate) points: Vec<StrokePoint>,
    pub(crate) surface: Surface,
    pub(crate) bounds: Rect,
}

impl LiveCommit {
    /// The params that decide the pixels, without the points and seed (compared separately).
    fn settings(p: &Value) -> Value {
        let mut p = p.clone();
        if let Some(o) = p.as_object_mut() {
            o.remove("points");
            o.remove("seed");
        }
        p
    }

    pub(crate) fn matches(&self, s: &Session, cmd: &str, p: &Value, pts: &[StrokePoint]) -> bool {
        let base = s.active().map(|st| (st.doc.id, st.revision));
        self.cmd == cmd
            && base == Some(self.base)
            && p.get("seed").and_then(Value::as_u64) == self.seed
            && self.points == pts
            && Self::settings(p) == Self::settings(&self.params)
    }
}

impl LiveStroke {
    /// Start from `paint.stroke` params; their `points` are rendered.
    pub fn begin(s: &Session, p: &Value) -> Result<Self> {
        Self::begin_with(s, "paint.stroke", p)
    }

    /// Start a live stroke of `cmd`: `paint.stroke` (the Brush and Eraser) or `paint.pencil`.
    /// Committing `cmd` with the same params and `"seed": live.seed` gives the same pixels.
    pub fn begin_with(s: &Session, cmd: &str, p: &Value) -> Result<Self> {
        has_paintable(s).map_err(EngineError::Other)?;
        let pts = parse_points(p, cmd)?;
        let pencil = match cmd {
            "paint.stroke" => false,
            "paint.pencil" => true,
            other => return Err(bad(other, "live strokes are `paint.stroke` or `paint.pencil`")),
        };
        let brush = if pencil { pencil_brush(s, p)? } else { with_blend_mode(resolve_brush(s, p, cmd)?, p) };
        let (seed, fg) = (brush.seed, brush.color);
        let (layer, mut brush, zoom) = stroke_target(s, p, brush)?;
        let st = s.active().ok_or(EngineError::NoDocument)?;
        let base = (st.doc.id, st.revision);
        let mut doc = (*st.doc).clone();
        let sel = doc.selection.clone();
        let (surf, lock) = crate::channel_cmds::target_surface(&mut doc, layer, p)?;
        erase_locked(&mut brush, lock, s.tools.background);
        if pencil && flag(p, "autoErase", false) {
            apply_auto_erase(&mut brush, surf, pts.first(), fg, s.tools.background);
        }
        let renderer = StrokeRenderer::new(&brush, Some(surf.format()), zoom);
        let symmetry = s.active().and_then(|st| st.symmetry.clone());
        let mirrors: Vec<_> = (0..symmetry.as_ref().map_or(0, |s| s.mirror_count())).map(|_| StrokeRenderer::new(&brush, Some(surf.format()), zoom)).collect();
        let mirror_distinct = vec![false; mirrors.len()];
        let pre = surf.clone();
        let mut live = Self {
            doc: std::sync::Arc::new(doc),
            seed,
            cmd: cmd.into(),
            renderer,
            symmetry,
            mirrors,
            mirror_distinct,
            union_live: None,
            pre,
            sel,
            lock,
            layer,
            params: p.clone(),
            tail: Rect::EMPTY,
            cmyk: doc_cmyk_space(s),
            points: Vec::new(),
            base,
        };
        live.push(&pts)?;
        Ok(live)
    }

    /// The stroke as its commit would take it over (see [`LiveCommit`]).
    pub fn into_commit(mut self) -> Result<LiveCommit> {
        let bounds = self.bounds();
        let (surf, _) = crate::channel_cmds::target_surface(std::sync::Arc::make_mut(&mut self.doc), self.layer, &self.params)?;
        Ok(LiveCommit { cmd: self.cmd, base: self.base, params: self.params, seed: Some(self.seed), points: self.points, surface: surf.clone(), bounds })
    }

    /// Everything the stroke has touched so far.
    pub fn bounds(&self) -> Rect {
        let bounds = self.renderer.bounds().union(&self.tail);
        self.mirrors.iter().zip(&self.mirror_distinct).filter(|(_, distinct)| **distinct).fold(bounds, |b, (r, _)| b.union(&r.bounds()))
    }

    /// `(brush, pointer)` document positions while smoothing holds the brush back from the
    /// pointer: the canvas draws a line between them.
    pub fn smoothing_string(&self) -> Option<([f64; 2], [f64; 2])> {
        self.renderer.smoothing_string()
    }

    /// Does the stroke change with time while the pointer is held still (airbrush Build-up, or
    /// smoothing still catching up)? See `photocraft_paint::dynamics::DabGenerator::wants_time`.
    pub fn wants_time(&self) -> bool {
        self.renderer.wants_time()
    }

    /// Render more points; returns the rectangle that changed. The doc shows the stroke as
    /// committing it now would: with smoothing, the brush lags behind the pointer and catches up
    /// when the stroke ends, so that catch-up tail is drawn too (and redrawn on every step), and
    /// nothing new appears on release.
    /// Invalid coordinates reject the whole batch without changing the preview.
    pub fn push(&mut self, pts: &[StrokePoint]) -> Result<Rect> {
        check_coords(pts, &self.cmd)?;
        self.points.extend_from_slice(pts);
        let cmyk = self.cmyk.clone();
        photocraft_color::convert::with_cmyk_space(cmyk.as_ref(), || {
            self.renderer.push(pts);
            if let Some(symmetry) = &self.symmetry {
                for ((mirror, distinct), reflected) in self.mirrors.iter_mut().zip(&mut self.mirror_distinct).zip(symmetry.reflected_passes(pts)) {
                    *distinct |= crate::symmetry_cmds::SymmetryAxis::has_distinct_mirror(pts, &reflected);
                    mirror.push(&reflected);
                }
            }
            let (surf, _) = crate::channel_cmds::target_surface(std::sync::Arc::make_mut(&mut self.doc), self.layer, &self.params)?;
            if self.mirror_distinct.iter().any(|d| *d) {
                // The stroke and its distinct mirrors composite as one stroke, as the commit
                // merges them; only the tiles that changed are redrawn.
                let old = self.tail;
                let passes = 1 + self.mirror_distinct.iter().filter(|d| **d).count();
                let (mut live, full, mut dmg) = match self.union_live.take() {
                    Some((live, merged)) => (live, merged != passes, Rect::EMPTY),
                    None => {
                        // Leaving the single-pass preview: take its smoothing tail back first.
                        if !old.is_empty() {
                            surf.write_region(old, &self.pre.read_region(old));
                        }
                        (photocraft_paint::UnionLive::default(), true, old)
                    }
                };
                let mut renderers: Vec<&mut StrokeRenderer> = std::iter::once(&mut self.renderer)
                    .chain(self.mirrors.iter_mut().zip(&self.mirror_distinct).filter(|(_, d)| **d).map(|(r, _)| r))
                    .collect();
                dmg = dmg.union(&StrokeRenderer::composite_union_live(&mut renderers, &mut live, full, &self.pre, surf, self.sel.as_ref(), self.lock));
                self.tail = old.union(&live.bounds());
                self.union_live = Some((live, passes));
                return Ok(dmg);
            }
            let mut dmg = Rect::EMPTY;
            let old = std::mem::replace(&mut self.tail, Rect::EMPTY);
            if !old.is_empty() {
                // Back to the stroke without the previous tail.
                surf.write_region(old, &self.pre.read_region(old));
                self.renderer.mark_dirty(old);
                dmg = old;
            }
            dmg = dmg.union(&self.renderer.composite(&self.pre, surf, self.sel.as_ref(), self.lock, false));
            if let Some(mut tail) = self.renderer.tail_preview() {
                self.tail = tail.composite(&self.pre, surf, self.sel.as_ref(), self.lock, false);
                dmg = dmg.union(&self.tail);
            }
            Ok(dmg)
        })
    }
}

fn pencil(s: &mut Session, p: &Value) -> Result<Value> {
    let pts = parse_points(p, "paint.pencil")?;
    let brush = pencil_brush(s, p)?;
    let label = if brush.erase && !clear_mode(p) { "Eraser" } else { "Pencil" };
    stroke_with(s, "paint.pencil", p, label, brush, pts, flag(p, "autoErase", false))
}

fn pct(p: &Value, k: &str, d: f32) -> f32 {
    num(p, k).unwrap_or(d).clamp(0.0, 100.0) / 100.0
}

fn mixer_brush(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "paint.mixerBrush";
    let pts = parse_points(p, cmd)?;
    let brush = resolve_brush(s, p, cmd)?;
    // The brush's Mixer Brush options, overridden by the scalar params.
    let bm = &brush.mixer;
    let m = MixerSettings {
        wet: pct(p, "wet", bm.wet * 100.0),
        load: pct(p, "load", bm.load * 100.0),
        mix: pct(p, "mix", bm.mix * 100.0),
        flow: pct(p, "flow", bm.flow * 100.0),
        sample_all_layers: flag(p, "sampleAllLayers", bm.sample_all_layers),
    };
    let sample_all = m.sample_all_layers;
    let (clean, load_after) = (flag(p, "cleanAfterStroke", true), flag(p, "loadAfterStroke", true));
    let mut state = s.tools.mixer.clone();
    if load_after || state.reservoir.is_none() {
        state.load(color(p.get("color"), s.tools.foreground));
    }
    let id = if crate::channel_cmds::is_channel_target(p) { None } else { Some(layer_id(s, p)?) };
    let stroke = Stroke { brush, points: pts };
    let dmg = s.edit("Mixer Brush", |doc, _| {
        let sel = doc.selection.clone();
        let sample = if sample_all {
            let ds = photocraft_paint::dabs(&stroke);
            let ctx = photocraft_paint::BrushContext::new(&stroke.brush);
            let area = ds.iter().fold(Rect::EMPTY, |r, d| r.union(&ctx.dab_rect(d, false)));
            let buf = photocraft_compose::render(doc, area);
            let mut surf = Surface::new(photocraft_color::PixelFormat::RGBA32F);
            let flat: Vec<f32> = buf.px.iter().flatten().copied().collect();
            if !area.is_empty() {
                surf.write_region(area, &flat);
            }
            Some(surf)
        } else {
            None
        };
        let (surf, lock) = crate::channel_cmds::target_surface(doc, id, p)?;
        Ok(apply_mixer_stroke(surf, sample.as_ref(), &stroke, &m, &mut state, sel.as_ref(), lock))
    })?;
    if clean {
        state.clean();
    }
    s.tools.mixer = state;
    Ok(damage_json(s, dmg))
}

/// `paint.colorReplacement`'s stroke and settings. The replacement colour is the brush colour
/// (the foreground unless `color` says otherwise). It paints layer pixels only: a mask, alpha
/// channel or Quick Mask `target` is refused rather than recolouring the layer instead.
fn color_replacement_setup(s: &Session, p: &Value) -> Result<(Stroke, ReplaceSettings, photocraft_doc::LayerId)> {
    let cmd = "paint.colorReplacement";
    if let Some(t) = p.get("target").filter(|t| t.as_str() != Some("pixels")) {
        return Err(bad(cmd, format!("Color Replacement paints layer pixels only (`target` {t} is not \"pixels\")")));
    }
    let pts = parse_points(p, cmd)?;
    let brush = resolve_brush(s, p, cmd)?;
    fn parse<T: serde::de::DeserializeOwned>(p: &Value, cmd: &str, k: &str) -> Result<Option<T>> {
        p.get(k).cloned().map(serde_json::from_value).transpose().map_err(|e| bad(cmd, format!("`{k}`: {e}")))
    }
    let rs = ReplaceSettings {
        mode: parse(p, cmd, "mode")?.unwrap_or(ReplaceMode::Color),
        sampling: parse(p, cmd, "sampling")?.unwrap_or(Sampling::Continuous),
        limits: parse(p, cmd, "limits")?.unwrap_or(Limits::Contiguous),
        tolerance: pct(p, "tolerance", 30.0),
        anti_alias: flag(p, "antiAlias", true),
        color: brush.color,
        background: s.tools.background,
    };
    let id = layer_id(s, p)?;
    Ok((Stroke { brush, points: pts }, rs, id))
}

fn color_replacement(s: &mut Session, p: &Value) -> Result<Value> {
    let (stroke, rs, id) = color_replacement_setup(s, p)?;
    let dmg = s.edit("Color Replacement", |doc, _| {
        let sel = doc.selection.clone();
        let lock = doc.effective_locks(id).transparency;
        let surf = paint_surface(doc, id, &json!({}))?;
        Ok(apply_color_replacement(surf, &stroke, &rs, sel.as_ref(), lock))
    })?;
    Ok(damage_json(s, dmg))
}

/// A `paint.colorReplacement` stroke shown while it is drawn: the same params, and the same
/// recolouring per dab as the commit, so the preview at the last point is the committed result.
pub fn live_color_replacement(s: &Session, p: &Value) -> Result<crate::retouch_cmds::LiveDab> {
    let (stroke, rs, id) = color_replacement_setup(s, p)?;
    crate::retouch_cmds::LiveDab::with_effect(s, "paint.colorReplacement", &json!({}), stroke, Some(id), 0, |pre| {
        Box::new(color_replacement_effect(pre.clone(), rs))
    })?
    .with_tail_preview()
}

fn brush_json(b: &BrushSettings) -> Value {
    serde_json::to_value(b).unwrap_or(Value::Null)
}

fn presets_list(s: &mut Session, p: &Value) -> Result<Value> {
    let full = flag(p, "full", false);
    Ok(json!({
        "presets": s.tools.presets.iter().map(|pr| {
            let mut v = json!({
                "name": pr.name,
                "builtin": pr.builtin,
                "size": pr.brush.size,
                "hardness": pr.brush.hardness,
                "tip": match &pr.brush.tip { TipShape::Round => "round", TipShape::Sampled(_) | TipShape::Stored(_) => "sampled" },
            });
            if full {
                // Full settings embed the bitmaps: load stored tips (a tip that can't be loaded
                // stays a reference to the store).
                let mut b = pr.brush.clone();
                let _ = s.load_brush_tips(&mut b);
                v["brush"] = brush_json(&b);
            }
            v
        }).collect::<Vec<_>>()
    }))
}

fn name_param(p: &Value, cmd: &str) -> Result<String> {
    p.get("name").and_then(Value::as_str).map(str::trim).filter(|n| !n.is_empty()).map(str::to_string).ok_or_else(|| bad(cmd, "missing `name`"))
}

/// Replace the preset of that name, or append it at the end of the list.
fn upsert(s: &mut Session, preset: BrushPreset) {
    match s.tools.presets.iter_mut().find(|x| x.name.eq_ignore_ascii_case(&preset.name)) {
        Some(x) => *x = preset,
        None => s.tools.presets.push(preset),
    }
    s.brush_presets_changed();
}

fn presets_save(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "brush.presets.save";
    let name = name_param(p, cmd)?;
    let mut brush = match p.get("brush").filter(|v| v.is_object()) {
        Some(patch) => merge_brush(&s.tools.brush, patch, cmd)?,
        None => s.tools.brush.clone(),
    };
    // The New Brush Preset dialog's boxes; without them the preset captures what it always did.
    let captures = PresetCaptures { size: flag(p, "captureSize", true), tool_settings: flag(p, "toolSettings", true), color: flag(p, "color", false) };
    if captures.color {
        brush.color = s.tools.foreground;
    }
    upsert(s, BrushPreset { name: name.clone(), brush, builtin: false, group: String::new(), folder: Vec::new(), captures });
    // The saved brush is the current brush now (Photoshop selects the brush it just saved).
    s.tools.current_preset = Some(name.clone());
    s.tools.brush_changed();
    Ok(json!({ "name": name, "count": s.tools.presets.len() }))
}

fn presets_delete(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "brush.presets.delete";
    let name = name_param(p, cmd)?;
    let before = s.tools.presets.len();
    s.tools.presets.retain(|x| !x.name.eq_ignore_ascii_case(&name));
    if s.tools.presets.len() == before {
        return Err(bad(cmd, format!("no brush preset named `{name}`")));
    }
    if s.tools.current_preset.as_ref().is_some_and(|c| c.eq_ignore_ascii_case(&name)) {
        s.tools.current_preset = None;
        // The remembered brush names the preset it was picked from: the name went away.
        s.tools.brush_changed();
    }
    s.brush_presets_changed();
    Ok(json!({ "count": s.tools.presets.len() }))
}

/// Overwrite an existing preset with the current brush (or the current brush plus a `brush`
/// patch), in place: no new preset, the position, name and group stay. The preset defaults to the
/// one the current brush was picked from. A built-in that is updated becomes the user's preset
/// (built-ins regenerate with their factory settings).
fn presets_update(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "brush.presets.update";
    let name = match p.get("name").filter(|v| !v.is_null()) {
        Some(_) => name_param(p, cmd)?,
        None => s.tools.current_preset.clone().ok_or_else(|| bad(cmd, "the current brush was not picked from a preset"))?,
    };
    let brush = match p.get("brush").filter(|v| v.is_object()) {
        Some(patch) => merge_brush(&s.tools.brush, patch, cmd)?,
        None => s.tools.brush.clone(),
    };
    let pr = s.tools.presets.iter_mut().find(|x| x.name.eq_ignore_ascii_case(&name)).ok_or_else(|| bad(cmd, format!("no brush preset named `{name}`")))?;
    pr.brush = brush;
    pr.builtin = false;
    let (name, group) = (pr.name.clone(), pr.group.clone());
    s.brush_presets_changed();
    Ok(json!({ "name": name, "group": group }))
}

/// Mark a preset as the one the current brush was picked from. The tip grids' click copies only
/// the tip fields (the rest of the brush stays), so the engine cannot infer the preset from the
/// brush: the UI names it. Selection is by identity, so look-alike duplicates stay distinct.
fn presets_set_current(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "brush.presets.setCurrent";
    let name = name_param(p, cmd)?;
    if !s.tools.presets.iter().any(|x| x.name.eq_ignore_ascii_case(&name)) {
        return Err(bad(cmd, format!("no brush preset named `{name}`")));
    }
    s.tools.current_preset = Some(name.clone());
    s.tools.brush_changed();
    Ok(json!({ "name": name }))
}

fn has_selection_and_pixels(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document")?;
    if d.doc.selection.is_none() {
        return Err("no selection".into());
    }
    let l = d.active_layer.and_then(|id| d.doc.layer(id)).ok_or("no active layer")?;
    if matches!(l.content, LayerContent::Raster(_)) { Ok(()) } else { Err("active layer is not a pixel layer".into()) }
}

/// Largest sampled tip side (like Photoshop's 5000 px limit, kept smaller for preset size).
pub const MAX_TIP: u32 = 2500;

fn define_from_selection(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "brush.defineFromSelection";
    let name = name_param(p, cmd)?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let sel = d.doc.selection.as_ref().ok_or_else(|| bad(cmd, "no selection"))?;
    let id = d.active_layer.ok_or(EngineError::Other("no active layer".into()))?;
    let surf = d.doc.layer(id).and_then(|l| l.surface()).ok_or_else(|| bad(cmd, "not a pixel layer"))?;
    let area = sel.content_bounds().intersect(&d.doc.bounds());
    if area.is_empty() {
        return Err(bad(cmd, "the selection is empty"));
    }
    // Paint amount = darkness × alpha × selection (black = full paint, like Photoshop).
    let (w, h) = (area.width() as usize, area.height() as usize);
    let mut v = vec![0.0f32; w * h];
    let sc = sel.channels();
    let sv = sel.read_region(area);
    for y in 0..h {
        for x in 0..w {
            let c = surf.rgba(area.x0 + x as i32, area.y0 + y as i32);
            let l = photocraft_color::blend::lum([c[0], c[1], c[2]]);
            v[y * w + x] = ((1.0 - l) * c[3] * sv[(y * w + x) * sc]).clamp(0.0, 1.0);
        }
    }
    // Crop to the painted extent.
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for y in 0..h {
        for x in 0..w {
            if v[y * w + x] > 1e-4 {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
            }
        }
    }
    if x1 <= x0 || y1 <= y0 {
        return Err(bad(cmd, "the selected pixels contain no paint (all white or transparent)"));
    }
    let (tw, th) = ((x1 - x0) as u32, (y1 - y0) as u32);
    if tw.max(th) > MAX_TIP {
        return Err(bad(cmd, format!("selection too large for a brush tip ({tw}×{th}; max {MAX_TIP} px)")));
    }
    let mut data = Vec::with_capacity((tw * th) as usize);
    for y in y0..y1 {
        data.extend_from_slice(&v[y * w + x0..y * w + x1]);
    }
    let tip = GrayTile::from_f32(tw, th, &data);
    let brush = BrushSettings { tip: TipShape::Sampled(tip), size: tw.max(th) as f32, spacing: 0.25, pressure_size: false, ..BrushSettings::default() };
    upsert(s, BrushPreset { name: name.clone(), brush: brush.clone(), builtin: false, group: String::new(), folder: Vec::new(), captures: Default::default() });
    s.tools.brush = brush;
    s.tools.current_preset = Some(name.clone());
    s.tools.brush_changed();
    Ok(json!({ "name": name, "width": tw, "height": th }))
}

pub(crate) fn set_brush(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "tools.setBrush";
    let mut b = s.tools.brush.clone();
    let mut picked = None;
    let mut picked_color = None;
    if let Some(name) = p.get("preset").and_then(Value::as_str) {
        let preset = find_preset(s, name, cmd)?;
        b = preset.picked_over(&s.tools.brush);
        picked_color = preset.captures.color.then_some(preset.brush.color);
        picked = Some(name.to_string());
    }
    let reset = flag(p, "reset", false);
    if reset {
        b = BrushSettings::default();
    }
    let mut patch = p.clone();
    if let Some(o) = patch.as_object_mut() {
        for k in ["preset", "reset", "coalesce"] {
            o.remove(k);
        }
        if let Some(inner) = o.remove("brush") {
            let tmp = merge_brush(&b, &inner, cmd)?;
            // Before the second merge: a non-finite value would serialise as `null` there and be
            // reported as a type error instead of naming the value (#977).
            validate_brush(&tmp, cmd)?;
            b = tmp;
        }
    }
    b = merge_brush(&b, &patch, cmd)?;
    validate_brush(&b, cmd)?;
    // The picked preset stays "the current brush" across later edits, so `brush.presets.update`
    // knows what to overwrite. Only once the brush is accepted: a failed call leaves no state
    // behind (a `reset` wins over a `preset`, as it replaces the whole brush).
    if reset {
        s.tools.current_preset = None;
    } else if let Some(name) = picked {
        s.tools.current_preset = Some(name);
        // A preset saved with Include Color brings its colour as the foreground (Photoshop).
        if let Some(c) = picked_color {
            s.tools.foreground = c;
        }
    }
    // The tool's brush always paints with full tips: load a library preset's from the store.
    s.load_brush_tips(&mut b).map_err(|e| bad(cmd, e))?;
    let before = std::mem::replace(&mut s.tools.brush, b);
    // A coalesced gesture (one slider drag) journals as one call: remember the brush it started from.
    let key = p.get("coalesce").and_then(Value::as_str).filter(|_| p.get("preset").is_none() && p.get("reset").is_none());
    let continuing = |s: &Session, k: &str| {
        s.tools.brush_gesture.as_ref().is_some_and(|(g, _)| g == k)
            && s.journal.last().is_some_and(|(id, lp)| id == cmd && lp.get("coalesce").and_then(Value::as_str) == Some(k))
    };
    match key {
        Some(k) if continuing(s, k) => {}
        Some(k) => s.tools.brush_gesture = Some((k.to_string(), before)),
        None => s.tools.brush_gesture = None,
    }
    // The one accepted brush change everyone (UI sliders, control channel, presets) lands here.
    s.tools.brush_changed();
    Ok(brush_json(&s.tools.brush))
}

/// Remember the current brush in `Preferences::last_brush` (see [`remembered_brush`]). Called
/// every frame while the pointer is up, so it is change-driven: `ToolState::brush_rev` is compared
/// with the revision already saved and an unchanged brush clones and serialises nothing.
pub fn remember_brush(s: &mut Session) {
    if s.tools.brush_rev == s.tools.brush_saved_rev {
        return;
    }
    let Some(now) = remembered_brush(s) else { return };
    if s.prefs().last_brush != now {
        s.prefs.edit(|p| p.last_brush = now);
    }
    s.tools.brush_saved_rev = s.tools.brush_rev;
}

/// The current brush as JSON for `Preferences::last_brush`: settings plus tip references, no full
/// bitmap. `None` while a sampled bitmap too big to embed is current (only before the store
/// externalizes it; see [`ref_tip`]), so the previous remembered brush stands.
fn remembered_brush(s: &Session) -> Option<Value> {
    let b = &s.tools.brush;
    let (tip, dual, pattern) = (ref_tip(&b.tip)?, ref_tip(&b.dual_brush.tip)?, ref_pattern(&b.texture.pattern)?);
    let mut c = b.clone();
    (c.tip, c.dual_brush.tip, c.texture.pattern) = (tip, dual, pattern);
    Some(json!({ "preset": s.tools.current_preset.clone(), "brush": serde_json::to_value(&c).ok()? }))
}

/// Restore `Preferences::last_brush` into the session at launch (a missing or malformed value
/// keeps the default brush). Stored tips load when the preset store attaches.
pub fn restore_brush(s: &mut Session) {
    let v = s.prefs().last_brush.clone();
    let Some(patch) = v.get("brush").filter(|b| b.is_object()) else { return };
    let Ok(mut b) = merge_brush(&BrushSettings::default(), patch, "brush.restore") else { return };
    if validate_brush(&b, "brush.restore").is_err() {
        return;
    }
    let _ = s.load_brush_tips(&mut b);
    s.tools.brush = b;
    s.tools.current_preset = v.get("preset").and_then(Value::as_str).map(str::to_string);
    // What was just restored is exactly what the preferences hold: nothing new to write.
    s.tools.brush_saved_rev = s.tools.brush_rev;
}

/// A tip for remembering: a loaded stored tip drops its full bitmap (it serialises as a reference
/// then); a sampled bitmap bigger than a preview would enter the prefs document whole.
fn ref_tip(t: &TipShape) -> Option<TipShape> {
    match t {
        TipShape::Stored(r) if r.full.is_some() => Some(TipShape::Stored(StoredTile { full: None, ..r.clone() })),
        TipShape::Sampled(g) if g.width > PREVIEW_SIDE || g.height > PREVIEW_SIDE => None,
        t => Some(t.clone()),
    }
}

fn ref_pattern(p: &Pattern) -> Option<Pattern> {
    match p {
        Pattern::Stored(r) if r.full.is_some() => Some(Pattern::Stored(StoredTile { full: None, ..r.clone() })),
        Pattern::Tile(g) if g.width > PREVIEW_SIDE || g.height > PREVIEW_SIDE => None,
        p => Some(p.clone()),
    }
}

/// The brush as JSON with the bitmaps (sampled tips, pattern tiles) that `skip` names replaced by
/// placeholders: they can be megabytes of base64, and skipped ones are equal on both sides.
fn light_json(b: &BrushSettings, skip: (bool, bool, bool)) -> Value {
    let mut c = b.clone();
    if skip.0 {
        c.tip = TipShape::Round;
    }
    if skip.1 {
        c.dual_brush.tip = TipShape::Round;
    }
    if skip.2 {
        c.texture.pattern = photocraft_paint::Pattern::default();
    }
    serde_json::to_value(&c).unwrap_or(Value::Null)
}

/// Fields of `new` that differ from `old`, nested objects diffed key by key: the minimal
/// `tools.setBrush` `brush` patch that turns `old` into `new` (`{}` when nothing changed).
pub fn brush_patch(old: &BrushSettings, new: &BrushSettings) -> Value {
    let skip = (old.tip == new.tip, old.dual_brush.tip == new.dual_brush.tip, old.texture.pattern == new.texture.pattern);
    fn diff(a: &Value, b: &Value) -> Option<Value> {
        match (a, b) {
            (Value::Object(ao), Value::Object(bo)) => {
                let mut out = serde_json::Map::new();
                for (k, bv) in bo {
                    match ao.get(k) {
                        Some(av) => {
                            if let Some(d) = diff(av, bv) {
                                out.insert(k.clone(), d);
                            }
                        }
                        None => {
                            out.insert(k.clone(), bv.clone());
                        }
                    }
                }
                (!out.is_empty()).then_some(Value::Object(out))
            }
            // Enums with payloads (tips, patterns) are replaced whole.
            _ => (a != b).then(|| b.clone()),
        }
    }
    diff(&light_json(old, skip), &light_json(new, skip)).unwrap_or_else(|| json!({}))
}

/// Journal hook ([`Session::execute`]): consecutive `tools.setBrush` calls with the same
/// `coalesce` key (one drag in the options bar or the Brush Settings panel) are one journal
/// entry, the patch from the brush before the gesture to the brush now. Returns true when the
/// call was folded into the previous entry.
pub fn coalesce_journal(s: &mut Session, id: &str, params: &Value) -> bool {
    if id != "tools.setBrush" {
        return false;
    }
    let Some(key) = params.get("coalesce").and_then(Value::as_str) else { return false };
    let Some((gk, start)) = s.tools.brush_gesture.as_ref() else { return false };
    if gk != key {
        return false;
    }
    let patch = brush_patch(start, &s.tools.brush);
    match s.journal.last_mut() {
        Some((last, lp)) if last == id && lp.get("coalesce").and_then(Value::as_str) == Some(key) => {
            *lp = json!({ "brush": patch, "coalesce": key });
            true
        }
        _ => false,
    }
}

macro_rules! spec {
    ($id:literal, $label:literal, $params:literal, $en:expr, $run:expr, $journal:expr) => {
        CommandSpec { id: $id, label: $label, menu: &[], shortcut: None, params: $params, enabled: $en, run: $run, journal: $journal }
    };
}

/// Brush command specs.
pub fn specs() -> Vec<CommandSpec> {
    vec![
        spec!(
            "paint.pencil",
            "Pencil",
            r##"{"points":[[x,y,pressure?,tiltX?,tiltY?,rotation?,timeMs?,wheel?],…],"brush":{…}?,"preset":name?,"size":0.5..5000 px?,"opacity":0..1?,"color":"#rrggbb"?=foreground,"mode":"normal|multiply|screen|…|clear (erases with this brush)"="normal","erase":bool?,"block":bool=false (a hard 16×16 screen-px square, 16/zoom doc px; size, opacity, flow, hardness and dynamics ignored: the Eraser's Block mode),"zoom":view zoom=1,"autoErase":bool=false,"seed":u64?,"target":"pixels"|"mask"|"quickMask"|{"channel":i}=Channels panel target}"##,
            has_paintable,
            pencil,
            true
        ),
        spec!(
            "paint.mixerBrush",
            "Mixer Brush",
            r##"{"points":[…],"brush":{…}?,"preset":name?,"size":0.5..5000 px?,"wet":0..100=brush.mixer.wet,"load":0..100=brush.mixer.load,"mix":0..100=brush.mixer.mix,"flow":0..100=brush.mixer.flow,"color":"#rrggbb"?=foreground,"sampleAllLayers":bool=brush.mixer.sampleAllLayers,"cleanAfterStroke":bool=true,"loadAfterStroke":bool=true,"seed":u64?}"##,
            has_paintable,
            mixer_brush,
            true
        ),
        spec!(
            "paint.colorReplacement",
            "Color Replacement",
            r##"{"points":[…],"brush":{…}?,"size":px?,"mode":"hue|saturation|color|luminosity"="color","sampling":"continuous|once|backgroundSwatch"="continuous","limits":"contiguous|discontiguous|findEdges"="contiguous","tolerance":0..100=30,"antiAlias":bool=true,"color":"#rrggbb"?=foreground,"target":"pixels"="pixels","seed":u64?}"##,
            crate::commands::has_paintable,
            color_replacement,
            true
        ),
        spec!("brush.presets.list", "List Brush Presets", r##"{"full":bool=false}"##, always, presets_list, false),
        spec!(
            "brush.presets.save",
            "Save Brush Preset",
            r##"{"name":string,"brush":{…BrushSettings}?=current brush,"captureSize":bool=true (false: picking it keeps the tool's size),"toolSettings":bool=true (false: picking it keeps the tool's mode, opacity, flow and pressure buttons),"color":bool=false (picking it sets the foreground to the current foreground)}"##,
            always,
            presets_save,
            true
        ),
        spec!(
            "brush.presets.update",
            "Update Brush Preset",
            r##"{"name":string?=the preset the current brush was picked from,"brush":{…BrushSettings}?=current brush}"##,
            always,
            presets_update,
            true
        ),
        spec!("brush.presets.delete", "Delete Brush Preset", r##"{"name":string}"##, always, presets_delete, true),
        spec!("brush.presets.setCurrent", "Set Current Brush Preset", r##"{"name":string}"##, always, presets_set_current, true),
        spec!("brush.defineFromSelection", "Define Brush Preset…", r##"{"name":string}"##, has_selection_and_pixels, define_from_selection, true),
        spec!("brush.get", "Get Brush", "{}", always, |s, _| Ok(brush_json(&s.tools.brush)), false),
        spec!("tools.setBrush", "Set Brush", r##"{"preset":name?,"reset":bool?,…BrushSettings fields (camelCase, deep-merged)}"##, always, set_brush, true),
    ]
}

#[cfg(test)]
mod tests;
