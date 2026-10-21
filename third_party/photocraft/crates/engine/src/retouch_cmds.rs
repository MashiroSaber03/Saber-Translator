//! Retouching tools as commands: Clone Stamp, Pattern Stamp, Healing Brush, Spot Healing Brush,
//! Patch, Content-Aware Move, Remove, Dodge, Burn, Sponge, Blur, Sharpen, Smudge and History Brush.
//!
//! Every command takes a Photoshop-style brush (`points`, `size`, `hardness`, `opacity`, `flow`,
//! `spacing`, `layer`), respects the active selection as a mask and the layer's transparency lock, and
//! works at any depth (8/16/32f) and colour model. Like the Brush, each paints the targeted surface:
//! the layer's pixels, its mask (`"target":"mask"`), an alpha channel or the Quick Mask (`"target"`,
//! filled in from the Channels panel when absent). The pixel algorithms live in `photocraft-algo`
//! (`poisson`, `inpaint`, `retouch`) and the dab machinery in `photocraft-paint::retouch`; this module
//! only parses parameters and wires them together.

use photocraft_algo::inpaint::{self, CompleteParams};
use photocraft_algo::poisson;
use photocraft_algo::retouch::{ToneRange, dodge_burn, local_blur, local_sharpen, sponge};
use photocraft_color::{BlendMode, PixelFormat, SampleType};
use photocraft_doc::{Document, LayerContent, LayerId};
use photocraft_geom::Rect;
use photocraft_paint::retouch::{Footprint, Region, Smudge, alpha_index, apply_coverage, apply_dab_stroke, stroke_coverage};
use photocraft_paint::{BrushSettings, Stroke, StrokePoint};
use photocraft_raster::{Surface, from_rgba_into, to_rgba};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, blend_from_str};
use crate::{EngineError, Result, Session};

mod content_aware_move;
mod patch;
mod pattern_stamp;
mod periodic;
mod remove;
pub use content_aware_move::{ContentAwarePreview, preview as content_aware_move_preview};
pub use patch::preview as patch_preview;

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

/// A pixel layer, a layer with a mask, or a targeted alpha channel / Quick Mask (as the Brush).
fn has_pixel_layer(s: &Session) -> std::result::Result<(), String> {
    if crate::channel_cmds::edits_channel(s) {
        return Ok(());
    }
    let d = s.active().ok_or("no document open")?;
    let l = d.active_layer.and_then(|id| d.doc.layer(id)).ok_or("no active layer")?;
    if matches!(l.content, LayerContent::Raster(_)) || l.mask.is_some() {
        Ok(())
    } else {
        Err(format!("active layer is {} {} layer, not a pixel layer", l.content.article(), l.content.kind_name()))
    }
}

/// The stroke paints layer pixels (not a mask or channel), so other layers can be sampled.
fn targets_pixels(p: &Value) -> bool {
    crate::channel_cmds::target_of(p) == crate::channel_cmds::Target::Pixels
}

fn num(p: &Value, k: &str, d: f32) -> f32 {
    p.get(k).and_then(Value::as_f64).map_or(d, |v| v as f32)
}
fn flag(p: &Value, k: &str, d: bool) -> bool {
    p.get(k).and_then(Value::as_bool).unwrap_or(d)
}
fn string<'a>(p: &'a Value, k: &str, d: &'a str) -> &'a str {
    p.get(k).and_then(Value::as_str).unwrap_or(d)
}
fn point(p: &Value, k: &str) -> Option<(f64, f64)> {
    let a = p.get(k)?.as_array()?;
    Some((a.first()?.as_f64()?, a.get(1)?.as_f64()?))
}

/// The brush a retouching stroke paints with: the tool's brush (`base`, its tip, hardness and
/// shape dynamics) with the shared params of `p` applied. UI previews of a stroke (the Spot Healing
/// Brush's trail) render their coverage with it, so they match what the command changes.
pub fn stroke_brush(base: &BrushSettings, p: &Value) -> BrushSettings {
    let pct = |k: &str, d: f32, lo: f32, hi: f32| num(p, k, d).clamp(lo, hi) / 100.0;
    BrushSettings {
        size: num(p, "size", base.size).max(1.0),
        hardness: pct("hardness", base.hardness * 100.0, 0.0, 100.0),
        opacity: pct("opacity", 100.0, 1.0, 100.0),
        flow: pct("flow", 100.0, 1.0, 100.0),
        // Brush Settings › Brush Tip Shape › Spacing, as the Brush uses it (#2140).
        spacing: pct("spacing", base.spacing * 100.0, 1.0, 1000.0),
        erase: false,
        ..base.clone()
    }
}

/// Parse the shared brush params into a stroke on a layer (`None` when an alpha channel or the
/// Quick Mask is targeted).
fn parse_brush(s: &Session, p: &Value, cmd: &str) -> Result<(Stroke, Option<LayerId>)> {
    let pts: Vec<StrokePoint> = p
        .get("points")
        .and_then(Value::as_array)
        .ok_or_else(|| bad(cmd, "missing `points`"))?
        .iter()
        .filter_map(|v| {
            let a = v.as_array()?;
            Some(StrokePoint::new(a.first()?.as_f64()?, a.get(1)?.as_f64()?, a.get(2).and_then(Value::as_f64).unwrap_or(1.0) as f32))
        })
        .collect();
    if pts.is_empty() {
        return Err(bad(cmd, "`points` is empty"));
    }
    crate::brush_cmds::check_coords(&pts, cmd)?;
    let brush = stroke_brush(&s.tools.brush, p);
    crate::brush_cmds::validate_brush(&brush, cmd)?;
    if crate::channel_cmds::is_channel_target(p) {
        return Ok((Stroke { brush, points: pts }, None));
    }
    let id = match p.get("layer").and_then(Value::as_u64) {
        Some(id) => LayerId(id),
        None => s.active().and_then(|d| d.active_layer).ok_or(EngineError::Other("no active layer".into()))?,
    };
    Ok((Stroke { brush, points: pts }, Some(id)))
}

/// Run `f` on the targeted surface (layer pixels, layer mask, alpha channel or Quick Mask: the
/// Brush's paint target) inside one undoable step. `f` gets the pre-stroke document (a cheap
/// copy-on-write snapshot), the surface, the selection and the transparency lock.
fn run_stroke(
    s: &mut Session,
    label: &str,
    id: Option<LayerId>,
    p: &Value,
    f: impl FnOnce(&Document, &mut Surface, Option<&Surface>, bool) -> Result<Rect>,
) -> Result<Rect> {
    let dmg = s.edit(label, |doc, _| {
        let pre = doc.clone();
        let (surf, lock) = crate::channel_cmds::target_surface(doc, id, p)?;
        // No prune: retouching only rewrites pixels inside the stroke, and scanning every tile of a
        // large layer would dominate the cost of a dab.
        f(&pre, surf, pre.selection.as_ref(), lock)
    })?;
    if let Some(st) = s.active_mut() {
        st.last_damage = Some(dmg);
    }
    Ok(dmg)
}

fn damage_json(r: Rect) -> Value {
    json!([r.x0, r.y0, r.width(), r.height()])
}

// ---------------------------------------------------------------------------------------------
// Sampling sources
// ---------------------------------------------------------------------------------------------

/// Which pixels Clone Stamp / Healing Brush sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SampleLayers {
    Current,
    CurrentAndBelow,
    All,
}

fn sample_layers(p: &Value, cmd: &str) -> Result<SampleLayers> {
    match string(p, "sampleLayer", "current") {
        "current" => Ok(SampleLayers::Current),
        "currentAndBelow" => Ok(SampleLayers::CurrentAndBelow),
        "all" => Ok(SampleLayers::All),
        o => Err(bad(cmd, format!("unknown sampleLayer `{o}` (current|currentAndBelow|all)"))),
    }
}

/// Composite pixels of `rect` in `fmt` (native channels): the whole document, or the target layer
/// and everything below it.
pub(crate) fn composite_region(pre: &Document, id: Option<LayerId>, which: SampleLayers, rect: Rect, fmt: PixelFormat) -> Region {
    let mut doc = pre.clone();
    if which == SampleLayers::CurrentAndBelow
        && let Some(id) = id
    {
        // Hide everything composited after (above) the target in the bottom-to-top walk.
        let walk: Vec<(Vec<usize>, LayerId)> = pre.walk().into_iter().map(|(p, _, l)| (p, l.id)).collect();
        if let Some(pos) = walk.iter().position(|(_, l)| *l == id) {
            for (path, _) in &walk[pos + 1..] {
                if let Some(l) = doc.layer_at_mut(path) {
                    l.visible = false;
                }
            }
        }
    }
    let buf = photocraft_compose::render(&doc, rect);
    let n = fmt.channels();
    let mut out = Region::new(rect, n);
    let mut enc = [0.0f32; 8];
    for (i, px) in buf.px.iter().enumerate() {
        from_rgba_into(&fmt, *px, &mut enc);
        out.data[i * n..(i + 1) * n].copy_from_slice(&enc[..n]);
    }
    out
}

/// Pixels to sample at `rect` (document coordinates, already offset to the source position).
fn sample(pre: &Document, id: Option<LayerId>, surf: &Surface, which: SampleLayers, rect: Rect) -> Region {
    match which {
        SampleLayers::Current => Region::read(surf, rect),
        _ => composite_region(pre, id, which, rect, surf.format()),
    }
}

/// Where a clone stroke samples: `offset` / `source` params, else the active Clone Source slot,
/// with the slot's (or the call's) scale, rotation and flips (see `presets::clone_source`).
fn clone_mapping(s: &mut Session, p: &Value, stroke: &Stroke, cmd: &str) -> Result<crate::presets::clone_source::Mapping> {
    let f = stroke.points[0];
    crate::presets::clone_source::mapping(s, p, (f.x, f.y), cmd)
}

/// Source pixels for destination `rect`: a translated read, or a bilinear resample when the
/// clone source is scaled, rotated or flipped.
#[allow(clippy::too_many_arguments)]
fn clone_sample(
    pre: &Document,
    id: Option<LayerId>,
    surf: &Surface,
    which: SampleLayers,
    rect: Rect,
    map: &crate::presets::clone_source::Mapping,
    wrap: bool,
    cmd: &str,
) -> Result<Region> {
    let read = |rect| {
        if wrap { periodic::sample(pre, id, surf, which, rect, cmd) } else { Ok(sample(pre, id, surf, which, rect)) }
    };
    if map.is_translation() {
        let off = map.offset();
        let src = rect.translate(off.0, off.1);
        // `translate` saturates at the i32 edges, so a far source (issue #1111) yields a window
        // smaller than `rect`; relabelling that as `rect` would index past the sampled pixels.
        if src.width() != rect.width() || src.height() != rect.height() {
            return Err(bad(cmd, format!("source offset [{}, {}] is too far: the sample window leaves the coordinate range", off.0, off.1)));
        }
        let mut r = read(src)?;
        r.rect = rect;
        return Ok(r);
    }
    let src = read(map.source_rect(rect))?;
    Ok(crate::presets::clone_source::resample(&src, rect, map, alpha_index(&surf.format())))
}

/// Read-only clone overlay. Shares the stroke sampler without anchoring a source or editing history.
/// Both destination and transformed source allocations are bounded for interactive use.
pub fn clone_preview(s: &Session, rect: Rect, map: &crate::presets::clone_source::Mapping, sample_layer: &str) -> Result<photocraft_compose::Buffer> {
    clone_preview_periodic(s, rect, map, sample_layer, false)
}

/// Clone-source overlay with optional source repetition for View > Pattern Preview.
pub fn clone_preview_periodic(
    s: &Session,
    rect: Rect,
    map: &crate::presets::clone_source::Mapping,
    sample_layer: &str,
    wrap: bool,
) -> Result<photocraft_compose::Buffer> {
    if ![map.source.0, map.source.1, map.anchor.0, map.anchor.1].into_iter().chain(map.m).all(|v| v.is_finite() && v.abs() < 1_000_000.0) {
        return Err(bad("clone.preview", "invalid preview mapping"));
    }
    if ![rect.x0, rect.y0, rect.x1, rect.y1].into_iter().all(|v| v.abs_diff(0) < 1_000_000)
        || ![(rect.x0, rect.y0), (rect.x1, rect.y1), (rect.x0, rect.y1), (rect.x1, rect.y0)].into_iter().all(|(x, y)| {
            let (x, y) = map.map(f64::from(x), f64::from(y));
            x.is_finite() && y.is_finite() && x.abs() < 1_000_000.0 && y.abs() < 1_000_000.0
        })
    {
        return Err(bad("clone.preview", "preview coordinates exceed their budget"));
    }
    let bounded = |r: Rect| r.width() > 0 && r.height() > 0 && r.width() <= 1024 && r.height() <= 1024;
    if !bounded(rect) || !bounded(map.source_rect(rect)) {
        return Err(bad("clone.preview", "preview region exceeds its budget"));
    }
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let id = d.active_layer.ok_or_else(|| bad("clone.preview", "no active layer"))?;
    let surf = d.doc.layer(id).and_then(|l| l.surface()).ok_or_else(|| bad("clone.preview", "no pixel surface"))?;
    let which = sample_layers(&json!({"sampleLayer": sample_layer}), "clone.preview")?;
    let region = clone_sample(&d.doc, Some(id), surf, which, rect, map, wrap, "clone.preview")?;
    let px = region.data.chunks_exact(surf.format().channels()).map(|p| to_rgba(&surf.format(), p)).collect();
    Ok(photocraft_compose::Buffer { rect, px })
}

/// Result JSON shared by Clone Stamp and Healing Brush: the offset used and where the next stroke
/// should sample from if the UI keeps Photoshop's "Aligned" semantics.
fn clone_result(dmg: Rect, off: (i32, i32), aligned: bool, p: &Value, stroke: &Stroke) -> Value {
    let f = stroke.points[0];
    let source = point(p, "source").unwrap_or((f.x + off.0 as f64, f.y + off.1 as f64));
    // Aligned: the offset persists (next stroke samples at its own first point + offset).
    // Non-aligned: every stroke restarts at the original source point.
    json!({ "damage": damage_json(dmg), "offset": [off.0, off.1], "aligned": aligned, "nextSource": if aligned { Value::Null } else { json!([source.0, source.1]) } })
}

fn blend_param(p: &Value, cmd: &str) -> Result<BlendMode> {
    match p.get("mode").and_then(Value::as_str) {
        None => Ok(BlendMode::Normal),
        Some(m) => blend_from_str(m).ok_or_else(|| bad(cmd, format!("unknown blend mode `{m}`"))),
    }
}

/// Clamp results into the storable range: integer depths to 0..1, float depths only below at 0;
/// alpha always 0..1.
fn clamp_samples(fmt: &PixelFormat, data: &mut [f32]) {
    let n = fmt.channels();
    let a = alpha_index(fmt);
    let float = fmt.sample == SampleType::F32;
    for px in data.chunks_exact_mut(n) {
        for (c, v) in px.iter_mut().enumerate() {
            *v = if Some(c) == a || !float { v.clamp(0.0, 1.0) } else { v.max(0.0) };
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Clone Stamp / Healing Brush / History Brush
// ---------------------------------------------------------------------------------------------

fn clone_stamp(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "paint.cloneStamp";
    let (stroke, id) = parse_brush(s, p, CMD)?;
    let which = sample_layers(p, CMD)?;
    let mode = blend_param(p, CMD)?;
    let aligned = flag(p, "aligned", true);
    // The live stroke the canvas already drew for exactly this commit: take its pixels instead of
    // rendering the stroke again (the hitch on release).
    let live = s.live_commit.take().filter(|l| l.matches(s, CMD, p, &stroke.points));
    let from_live = live.is_some();
    let map = clone_mapping(s, p, &stroke, CMD)?;
    let off = map.offset();
    let dmg =
        run_stroke(s, "Clone Stamp", id, p, |pre, surf, sel, lock| {
            if let Some(l) = live {
                *surf = l.surface;
                return Ok(l.bounds);
            }
            if flag(p, "wrap", false) {
                let mut renderer = photocraft_paint::StrokeRenderer::new(&stroke.brush, None, 1.0);
                renderer.push(&stroke.points);
                renderer.finish();
                let before = surf.clone();
                let bounds = renderer.bounds();
                return periodic::Composite { target: surf, before: &before, tile: pre.bounds(), opacity: stroke.brush.opacity, selection: sel, lock, mode }
                    .draw(bounds, &stroke.brush, &stroke.points, |r| clone_sample(pre, id, &before, which, r, &map, true, CMD));
            }
            let (bounds, cov) = stroke_coverage(&stroke);
            // Samples come from the pre-stroke state, so pixels painted earlier in the stroke are never re-cloned.
            let paint = clone_sample(pre, id, surf, which, bounds, &map, flag(p, "wrap", false), CMD)?;
            Ok(apply_coverage(surf, bounds, &cov, stroke.brush.opacity, sel, lock, &paint, mode))
        })?;
    let mut r = clone_result(dmg, off, aligned, p, &stroke);
    if from_live {
        r["fromLive"] = json!(true);
    }
    Ok(r)
}

/// What a live retouching stroke paints through its coverage.
enum LivePaint {
    /// The clone source (Clone Stamp; the Healing Brush shows its texture while drawing and
    /// blends it on release).
    Clone { which: SampleLayers, map: crate::presets::clone_source::Mapping, cmd: &'static str },
    /// The History Brush's history state.
    History(Surface),
}

/// A retouching stroke shown while it is drawn: the dabs so far composited onto a copy of the
/// active document as the command with the same params commits them (the same coverage,
/// sampling and blend): Clone Stamp, History Brush, and the Healing Brush's texture.
pub struct LiveRetouch {
    /// The active document with the stroke so far.
    pub doc: std::sync::Arc<Document>,
    cmd: String,
    renderer: photocraft_paint::StrokeRenderer,
    pre: std::sync::Arc<Document>,
    pre_surf: Surface,
    id: Option<LayerId>,
    params: Value,
    paint: LivePaint,
    mode: BlendMode,
    opacity: f32,
    sel: Option<Surface>,
    lock: bool,
    /// Where the doc shows what finishing the stroke now would add (a lone first dab, the
    /// smoothing catch-up), redrawn on every step as the Brush's live stroke does.
    tail: Rect,
    /// The document and revision the stroke started on.
    base: (photocraft_doc::DocId, u64),
    brush: Box<BrushSettings>,
    /// Every point rendered so far, in order.
    points: Vec<StrokePoint>,
}

impl LiveRetouch {
    /// Start a live stroke of `cmd` (`paint.cloneStamp`, `paint.healingBrush` or
    /// `paint.historyBrush`) from its params; their `points` are rendered.
    pub fn begin(s: &Session, cmd: &str, p: &Value) -> Result<Self> {
        has_pixel_layer(s).map_err(EngineError::Other)?;
        let (stroke, id) = parse_brush(s, p, cmd)?;
        let mode = blend_param(p, cmd)?;
        let paint = match cmd {
            "paint.cloneStamp" | "paint.healingBrush" => {
                let f = *stroke.points.first().ok_or_else(|| bad(cmd, "`points` is empty"))?;
                LivePaint::Clone {
                    which: sample_layers(p, cmd)?,
                    map: crate::presets::clone_source::resolve(s, p, (f.x, f.y), cmd)?.0,
                    cmd: if cmd == "paint.healingBrush" { "paint.healingBrush" } else { "paint.cloneStamp" },
                }
            }
            "paint.historyBrush" => LivePaint::History(history_source(s, id, p, cmd)?),
            other => return Err(bad(other, "live retouching strokes are the Clone Stamp, Healing Brush and History Brush")),
        };
        let st = s.active().ok_or(EngineError::NoDocument)?;
        let (pre, base) = (st.doc.clone(), (st.doc.id, st.revision));
        let mut doc = (*pre).clone();
        let sel = doc.selection.clone();
        let (surf, lock) = crate::channel_cmds::target_surface(&mut doc, id, p)?;
        let pre_surf = surf.clone();
        // As `stroke_coverage`, which the commit uses.
        let renderer = photocraft_paint::StrokeRenderer::new(&stroke.brush, None, 1.0);
        let opacity = stroke.brush.opacity;
        let brush = Box::new(stroke.brush.clone());
        let paint = match paint {
            LivePaint::History(src) if src.format() != pre_surf.format() => LivePaint::History(src.convert(pre_surf.format())),
            other => other,
        };
        let mut live = Self {
            doc: std::sync::Arc::new(doc),
            cmd: cmd.into(),
            renderer,
            pre,
            pre_surf,
            id,
            params: p.clone(),
            paint,
            mode,
            opacity,
            sel,
            lock,
            tail: Rect::EMPTY,
            base,
            brush,
            points: Vec::new(),
        };
        live.push(&stroke.points)?;
        Ok(live)
    }

    /// Everything the stroke has touched so far.
    pub fn bounds(&self) -> Rect {
        let bounds = self.renderer.bounds().union(&self.tail);
        if self.cmd == "paint.cloneStamp" && flag(&self.params, "wrap", false) {
            periodic::project(bounds, self.pre.bounds()).into_iter().fold(Rect::EMPTY, |a, b| a.union(&b))
        } else {
            bounds
        }
    }

    /// Every point rendered so far, in order.
    pub fn points(&self) -> &[StrokePoint] {
        &self.points
    }

    /// The finished stroke as its commit would take it over ([`Session::offer_live_retouch`]).
    /// Only the Clone Stamp and History Brush draw what they commit (the Healing Brush blends on
    /// release), and only while the brush, layer and clone source are still the ones the stroke
    /// started with: the document revision doesn't cover those.
    pub(crate) fn into_commit(mut self, s: &Session) -> Result<crate::brush_cmds::LiveCommit> {
        let (stroke, id) = parse_brush(s, &self.params, &self.cmd)?;
        let source_unchanged = match &self.paint {
            LivePaint::Clone { map, .. } if self.cmd == "paint.cloneStamp" => {
                let f = stroke.points.first().ok_or_else(|| bad(&self.cmd, "`points` is empty"))?;
                crate::presets::clone_source::resolve(s, &self.params, (f.x, f.y), &self.cmd)?.0 == *map
            }
            LivePaint::History(_) if self.cmd == "paint.historyBrush" => true,
            _ => return Err(bad(&self.cmd, "only Clone Stamp and History Brush strokes commit their live pixels")),
        };
        if !source_unchanged || *self.brush != stroke.brush || id != self.id {
            return Err(bad(&self.cmd, "the brush, layer or clone source changed during the stroke"));
        }
        let bounds = self.bounds();
        let (surf, _) = crate::channel_cmds::target_surface(std::sync::Arc::make_mut(&mut self.doc), self.id, &self.params)?;
        let surface = surf.clone();
        Ok(crate::brush_cmds::LiveCommit { cmd: self.cmd, base: self.base, params: self.params, seed: None, points: self.points, surface, bounds })
    }

    /// Render more points; returns the rectangle that changed. Each changed area is redrawn from
    /// the pre-stroke pixels with the stroke's coverage so far, so overlapping dabs build up as in
    /// the commit, and pixels painted earlier in the stroke are never re-cloned. What finishing
    /// the stroke now would add is drawn too (a lone first dab shows on the press), and redrawn on
    /// every step.
    /// Invalid coordinates reject the whole batch without changing the preview.
    /// Only the position and pressure of `pts` are used: the commands take no tilt or times.
    pub fn push(&mut self, pts: &[StrokePoint]) -> Result<Rect> {
        crate::brush_cmds::check_coords(pts, &self.cmd)?;
        let pts: Vec<_> = pts.iter().map(|q| StrokePoint::new(q.x, q.y, q.pressure)).collect();
        self.points.extend_from_slice(&pts);
        self.renderer.push(&pts);
        if self.cmd == "paint.cloneStamp" && flag(&self.params, "wrap", false) {
            return self.push_periodic();
        }
        let old = std::mem::replace(&mut self.tail, Rect::EMPTY);
        let mut dmg = Rect::EMPTY;
        if !old.is_empty() {
            // Back to the stroke without the previous tail.
            let (surf, _) = crate::channel_cmds::target_surface(std::sync::Arc::make_mut(&mut self.doc), self.id, &self.params)?;
            surf.write_region(old, &self.pre_surf.read_region(old));
            self.renderer.mark_dirty(old);
            dmg = old;
        }
        let r = self.renderer.take_dirty_rect();
        let cov = self.renderer.coverage_in(r);
        dmg = dmg.union(&self.composite(r, &cov)?);
        if let Some(mut tail) = self.renderer.tail_preview() {
            let r = tail.take_dirty_rect();
            let cov = tail.coverage_in(r);
            self.tail = self.composite(r, &cov)?;
            dmg = dmg.union(&self.tail);
        }
        Ok(dmg)
    }

    fn push_periodic(&mut self) -> Result<Rect> {
        let mut dirty = self.renderer.take_dirty_rect().union(&self.tail);
        let mut tail = self.renderer.tail_preview();
        self.tail = tail.as_mut().map_or(Rect::EMPTY, |t| t.take_dirty_rect());
        dirty = dirty.union(&self.tail);
        let LivePaint::Clone { which, map, cmd } = &self.paint else {
            return Err(bad(&self.cmd, "periodic painting requires a clone source"));
        };
        let (surf, _) = crate::channel_cmds::target_surface(std::sync::Arc::make_mut(&mut self.doc), self.id, &self.params)?;
        periodic::Composite {
            target: surf,
            before: &self.pre_surf,
            tile: self.pre.bounds(),
            opacity: self.opacity,
            selection: self.sel.as_ref(),
            lock: self.lock,
            mode: self.mode,
        }
        .draw(dirty, &self.brush, &self.points, |r| clone_sample(&self.pre, self.id, &self.pre_surf, *which, r, map, true, cmd))
    }

    /// Redraw `r` from the pre-stroke pixels with the clone source through coverage `cov`.
    fn composite(&mut self, r: Rect, cov: &[f32]) -> Result<Rect> {
        if r.is_empty() {
            return Ok(r);
        }
        let paint = match &self.paint {
            LivePaint::Clone { which, map, cmd } => {
                clone_sample(&self.pre, self.id, &self.pre_surf, *which, r, map, *cmd == "paint.cloneStamp" && flag(&self.params, "wrap", false), cmd)?
            }
            LivePaint::History(src) => Region::read(src, r),
        };
        let (surf, _) = crate::channel_cmds::target_surface(std::sync::Arc::make_mut(&mut self.doc), self.id, &self.params)?;
        surf.write_region(r, &self.pre_surf.read_region(r));
        Ok(apply_coverage(surf, r, cov, self.opacity, self.sel.as_ref(), self.lock, &paint, self.mode).union(&r))
    }
}

/// Grow a stroke's coverage map by `m` pixels of zero coverage (a Dirichlet boundary for the solve).
fn pad_coverage(bounds: Rect, cov: &[f32], m: i32) -> (Rect, Vec<f32>) {
    let g = bounds.inflate(m);
    let (w, gw) = (bounds.width() as usize, g.width() as usize);
    let mut out = vec![0.0f32; gw * g.height() as usize];
    for (i, c) in cov.iter().enumerate() {
        let (x, y) = (i % w + m as usize, i / w + m as usize);
        out[y * gw + x] = *c;
    }
    (g, out)
}

/// Seamless-clone `src` into `dst` over the mask (both regions cover the same rect). The solve runs
/// only on the mask's bounding box plus a one-pixel Dirichlet ring; the rest is `dst`.
fn heal_region(fmt: &PixelFormat, src: &Region, dst: &Region, mask: &[bool]) -> Region {
    let w = dst.width();
    let bbox = mask.iter().enumerate().filter(|(_, m)| **m).fold(Rect::EMPTY, |r, (i, _)| {
        let (x, y) = (dst.rect.x0 + (i % w) as i32, dst.rect.y0 + (i / w) as i32);
        r.union(&Rect::new(x, y, x + 1, y + 1))
    });
    let mut out = dst.clone();
    if bbox.is_empty() {
        return out;
    }
    let r = bbox.inflate(1).intersect(&dst.rect);
    let (s, d) = (src.crop(r), dst.crop(r));
    let m: Vec<bool> = (r.y0..r.y1).flat_map(|y| (r.x0..r.x1).map(move |x| (x, y))).map(|(x, y)| mask[dst.index(x, y)]).collect();
    let mut data = poisson::seamless_clone(d.width(), d.height(), d.ch, &s.data, &d.data, &m);
    clamp_samples(fmt, &mut data);
    let (n, rw) = (dst.ch, r.width() as usize);
    for y in r.y0..r.y1 {
        let o = out.index(r.x0, y) * n;
        let i = (y - r.y0) as usize * rw * n;
        out.data[o..o + rw * n].copy_from_slice(&data[i..i + rw * n]);
    }
    out
}

fn healing_brush(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "paint.healingBrush";
    let (stroke, id) = parse_brush(s, p, CMD)?;
    let which = sample_layers(p, CMD)?;
    let mode = blend_param(p, CMD)?;
    let aligned = flag(p, "aligned", true);
    let map = clone_mapping(s, p, &stroke, CMD)?;
    let off = map.offset();
    let dmg = run_stroke(s, "Healing Brush", id, p, |pre, surf, sel, lock| {
        let (bounds, cov) = stroke_coverage(&stroke);
        let (g, cov) = pad_coverage(bounds, &cov, 2);
        let fmt = surf.format();
        let src = clone_sample(pre, id, surf, which, g, &map, false, CMD)?;
        // The destination the texture is fitted to: the layer itself, or what the user sees when
        // sampling several layers (so healing onto an empty layer matches the composite).
        let dst = if which == SampleLayers::Current { Region::read(surf, g) } else { composite_region(pre, id, which, g, fmt) };
        let mask: Vec<bool> = cov.iter().map(|c| *c > 0.0).collect();
        let healed = heal_region(&fmt, &src, &dst, &mask);
        Ok(apply_coverage(surf, g, &cov, stroke.brush.opacity, sel, lock, &healed, mode))
    })?;
    Ok(clone_result(dmg, off, aligned, p, &stroke))
}

/// The History Brush's source: the targeted surface (pixels, mask, channel) as it was in history
/// state `state` (default: the oldest state still held, the "Open" snapshot unless trimmed).
fn history_source(s: &Session, id: Option<LayerId>, p: &Value, cmd: &str) -> Result<Surface> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let explicit = p.get("state").and_then(Value::as_u64).map(|v| v as usize);
    let source_doc = match explicit {
        Some(i) if i == st.history.past_len() => st.doc.clone(),
        Some(i) => st.history.state(i).ok_or_else(|| bad(cmd, format!("no history state {i} (0..={})", st.history.past_len())))?,
        None => st.history.state(0).unwrap_or_else(|| st.doc.clone()),
    };
    crate::channel_cmds::read_target_surface(source_doc.as_ref(), id, p)
        .cloned()
        .map_err(|_| EngineError::Other("the target did not exist (as pixels, a mask or a channel) in that history state".into()))
}

fn history_brush(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "paint.historyBrush";
    let (stroke, id) = parse_brush(s, p, CMD)?;
    let mode = blend_param(p, CMD)?;
    let live = s.live_commit.take().filter(|l| l.matches(s, CMD, p, &stroke.points));
    let from_live = live.is_some();
    let src_surface = history_source(s, id, p, CMD)?;
    let dmg = run_stroke(s, "History Brush", id, p, |_, surf, sel, lock| {
        if let Some(l) = live {
            *surf = l.surface;
            return Ok(l.bounds);
        }
        let fmt = surf.format();
        let src = if src_surface.format() == fmt { src_surface } else { src_surface.convert(fmt) };
        let (bounds, cov) = stroke_coverage(&stroke);
        let paint = Region::read(&src, bounds);
        Ok(apply_coverage(surf, bounds, &cov, stroke.brush.opacity, sel, lock, &paint, mode))
    })?;
    let mut r = json!({ "damage": damage_json(dmg) });
    if from_live {
        r["fromLive"] = json!(true);
    }
    Ok(r)
}

// ---------------------------------------------------------------------------------------------
// Spot Healing
// ---------------------------------------------------------------------------------------------

/// Dilate a mask by `r` pixels (square structuring element).
fn dilate(w: usize, h: usize, m: &[bool], r: usize) -> Vec<bool> {
    let mut tmp = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            if m[y * w + x] {
                for xx in x.saturating_sub(r)..(x + r + 1).min(w) {
                    tmp[y * w + xx] = true;
                }
            }
        }
    }
    let mut out = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            if tmp[y * w + x] {
                for yy in y.saturating_sub(r)..(y + r + 1).min(h) {
                    out[yy * w + x] = true;
                }
            }
        }
    }
    out
}

/// Copy a translated Proximity Match patch from the *pre-healing* pixels. Reading
/// `out` here would make overlapping offsets repeatedly copy the pixels just written,
/// leaving streaks instead of the selected source texture.
fn proximity_source_patch(img: &[f32], w: usize, h: usize, ch: usize, mask: &[bool], dx: i32, dy: i32) -> Option<Vec<f32>> {
    if w == 0 || h == 0 || ch == 0 || mask.len() != w.checked_mul(h)? || img.len() != mask.len().checked_mul(ch)? {
        return None;
    }
    let mut out = img.to_vec();
    for (p, masked) in mask.iter().enumerate() {
        if !masked {
            continue;
        }
        let x = i32::try_from(p % w).ok()?.checked_add(dx)?;
        let y = i32::try_from(p / w).ok()?.checked_add(dy)?;
        if x < 0 || y < 0 || x >= i32::try_from(w).ok()? || y >= i32::try_from(h).ok()? {
            return None;
        }
        let src = (usize::try_from(y).ok()?.checked_mul(w)? + usize::try_from(x).ok()?).checked_mul(ch)?;
        let dst = p.checked_mul(ch)?;
        out.get_mut(dst..dst + ch)?.copy_from_slice(img.get(src..src + ch)?);
    }
    Some(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpotType {
    ContentAware,
    CreateTexture,
    ProximityMatch,
}

/// Fill the stroke area with no source, then gradient-domain blend it into its surroundings. With
/// `all` (Sample All Layers) it heals what is visible, so it works on an empty layer above the image.
///
/// Photoshop 25.4 (measured): a dab changes exactly the brush's footprint, and Content-Aware takes
/// its texture from around it, as Edit › Fill does from its window
/// ([`crate::fill_cmds::sampling_window`] of the stroke's bounds), which is what this samples.
fn spot_heal_surface(surf: &mut Surface, pre: &Document, stroke: &Stroke, kind: SpotType, all: bool, sel: Option<&Surface>, lock: bool) -> Rect {
    let (bounds, cov) = stroke_coverage(stroke);
    let size = stroke.brush.size;
    let region_rect = crate::fill_cmds::sampling_window(bounds, pre.bounds());
    if region_rect.is_empty() {
        return Rect::EMPTY;
    }
    // Proximity Match looks for its source anywhere in that window.
    let margin = (region_rect.width().max(region_rect.height()) / 2) as i32;
    let fmt = surf.format();
    let img = if all { composite_region(pre, None, SampleLayers::All, region_rect, fmt) } else { Region::read(surf, region_rect) };
    let (w, h, ch) = (img.width(), img.height(), img.ch);
    let bw = bounds.width() as usize;
    let mut stroke_cov = vec![0.0f32; w * h];
    for (i, c) in cov.iter().enumerate() {
        let (x, y) = (bounds.x0 + (i % bw) as i32, bounds.y0 + (i / bw) as i32);
        if region_rect.contains(x, y) {
            stroke_cov[img.index(x, y)] = *c;
        }
    }
    let hole: Vec<bool> = stroke_cov.iter().map(|c| *c > 0.0).collect();
    if !hole.iter().any(|h| *h) {
        return Rect::EMPTY;
    }
    // Synthesise over a slightly larger domain than the Poisson region, so the solve's boundary pixels
    // carry synthesised values that differ from the untouched surroundings: that mismatch is what the
    // membrane corrects (otherwise the Dirichlet data would be all zero and the blend a no-op).
    let domain = dilate(w, h, &hole, 2);
    let heal_mask = dilate(w, h, &hole, 1);
    let content_aware =
        |img: &[f32]| inpaint::complete(w, h, ch, img, &domain, &CompleteParams::default()).unwrap_or_else(|| poisson::membrane_fill(w, h, ch, img, &domain));
    let filled = match kind {
        SpotType::ContentAware => content_aware(&img.data),
        SpotType::CreateTexture => inpaint::synthesize(w, h, ch, &img.data, &domain, (size / 4.0).clamp(6.0, 32.0) as usize, 0x5eed),
        SpotType::ProximityMatch => {
            let ring = (size / 8.0).clamp(3.0, 16.0) as usize;
            match inpaint::best_offset(w, h, ch, &img.data, &domain, ring, margin) {
                Some((dx, dy)) => proximity_source_patch(&img.data, w, h, ch, &domain, dx, dy).unwrap_or_else(|| content_aware(&img.data)),
                None => content_aware(&img.data),
            }
        }
    };
    let src = Region { rect: region_rect, ch, data: filled };
    let healed = heal_region(&fmt, &src, &img, &heal_mask);
    let dmg = bounds.intersect(&region_rect);
    let dmg_cov: Vec<f32> = (dmg.y0..dmg.y1).flat_map(|y| (dmg.x0..dmg.x1).map(move |x| (x, y))).map(|(x, y)| stroke_cov[img.index(x, y)]).collect();
    apply_coverage(surf, dmg, &dmg_cov, stroke.brush.opacity, sel, lock, &healed, BlendMode::Normal)
}

fn spot_healing(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "paint.spotHealing";
    let (stroke, id) = parse_brush(s, p, CMD)?;
    let kind = match string(p, "type", "contentAware") {
        "contentAware" => SpotType::ContentAware,
        "createTexture" => SpotType::CreateTexture,
        "proximityMatch" => SpotType::ProximityMatch,
        o => return Err(bad(CMD, format!("unknown type `{o}` (contentAware|createTexture|proximityMatch)"))),
    };
    let all = sample_all_layers(p);
    let dmg = run_stroke(s, "Spot Healing Brush", id, p, |pre, surf, sel, lock| Ok(spot_heal_surface(surf, pre, &stroke, kind, all, sel, lock)))?;
    Ok(json!({ "damage": damage_json(dmg) }))
}

// ---------------------------------------------------------------------------------------------
// Toning (Dodge / Burn / Sponge) and focus (Blur / Sharpen), Smudge
// ---------------------------------------------------------------------------------------------

/// Per-dab strength such that one pass over a point totals roughly `e`, whatever the spacing
/// (about `1/spacing` dabs overlap each point on the stroke's centre line).
#[inline]
fn per_dab(e: f32, spacing: f32) -> f32 {
    1.0 - (1.0 - e.clamp(0.0, 1.0)).powf(spacing.clamp(0.01, 1.0))
}

/// Apply an RGB colour transform `f(rgb, strength)` to each covered pixel of a dab (through straight
/// RGBA, so it works for Gray/CMYK/Lab too; untouched pixels are never round-tripped).
fn color_dab(fmt: &PixelFormat, work: &mut Region, fp: &Footprint, strength: f32, spacing: f32, f: impl Fn([f32; 3], f32) -> [f32; 3]) {
    let r = fp.rect.intersect(&work.rect);
    let n = fmt.channels();
    let mut enc = [0.0f32; 8];
    for y in r.y0..r.y1 {
        for x in r.x0..r.x1 {
            let k = per_dab(fp.at(x, y) * strength, spacing);
            if k <= 0.0 {
                continue;
            }
            let px = work.px_mut(x, y);
            let rgba = to_rgba(fmt, px);
            let o = f([rgba[0], rgba[1], rgba[2]], k);
            from_rgba_into(fmt, [o[0], o[1], o[2], rgba[3]], &mut enc);
            let a = alpha_index(fmt);
            for c in 0..n {
                if Some(c) != a {
                    px[c] = enc[c];
                }
            }
        }
    }
}

/// Dodge and Burn: a stroke tones each pixel once, from its original colour, mixed in by the
/// stroke's coverage there (`toned` is the full-exposure colour). Dabs build up coverage as in a
/// brush stroke, capped at full coverage, so Exposure caps a stroke the way Opacity caps a paint
/// stroke: overlapping dabs and passes within one stroke don't compound the tone curve, and a
/// new stroke builds on the last.
fn tone_stroke(fmt: PixelFormat, toned: impl Fn([f32; 3]) -> [f32; 3] + 'static) -> DabEffect {
    /// Side of the cells that remember, per touched pixel, the coverage applied so far and the
    /// original pixel (copied from the working pixels the first time a dab reaches it). Dense
    /// cells rather than a map per pixel: a map lookup per covered pixel made long strokes with
    /// large brushes over ten times slower.
    const CELL: i32 = 64;
    struct Cell {
        cov: Vec<f32>,
        orig: Vec<f32>,
    }
    let mut cells: std::collections::HashMap<(i32, i32), Cell> = std::collections::HashMap::new();
    let a = alpha_index(&fmt);
    let n = fmt.channels();
    Box::new(move |work, fp| {
        let r = fp.rect.intersect(&work.rect);
        if r.is_empty() {
            return;
        }
        let mut enc = [0.0f32; 8];
        for cy in r.y0.div_euclid(CELL)..=(r.y1 - 1).div_euclid(CELL) {
            for cx in r.x0.div_euclid(CELL)..=(r.x1 - 1).div_euclid(CELL) {
                let cr = Rect::new(cx * CELL, cy * CELL, cx * CELL + CELL, cy * CELL + CELL);
                let part = r.intersect(&cr);
                if part.is_empty() {
                    continue;
                }
                let len = (CELL * CELL) as usize;
                // Coverage -1: not touched yet, so `orig` isn't filled in for that pixel.
                let cell = cells.entry((cx, cy)).or_insert_with(|| Cell { cov: vec![-1.0; len], orig: vec![0.0; len * n] });
                for y in part.y0..part.y1 {
                    for x in part.x0..part.x1 {
                        let k = fp.at(x, y).clamp(0.0, 1.0);
                        if k <= 0.0 {
                            continue;
                        }
                        let i = ((y - cr.y0) * CELL + (x - cr.x0)) as usize;
                        let (Some(prev), Some(src)) = (cell.cov.get_mut(i), cell.orig.get_mut(i * n..i * n + n)) else { continue };
                        if *prev < 0.0 {
                            // First touch: the working pixel is still the original.
                            src.copy_from_slice(work.px(x, y));
                            *prev = 0.0;
                        }
                        // Dabs build up the stroke's coverage as a brush stroke's do (flow), up
                        // to full coverage; the tone is applied once, from the original.
                        let k = *prev + k * (1.0 - *prev);
                        if k <= *prev {
                            continue;
                        }
                        *prev = k;
                        let o = to_rgba(&fmt, src);
                        let t = toned([o[0], o[1], o[2]]);
                        let m = [o[0] + (t[0] - o[0]) * k, o[1] + (t[1] - o[1]) * k, o[2] + (t[2] - o[2]) * k];
                        from_rgba_into(&fmt, [m[0], m[1], m[2], o[3]], &mut enc);
                        let px = work.px_mut(x, y);
                        for c in 0..n {
                            if Some(c) != a {
                                px[c] = enc[c];
                            }
                        }
                    }
                }
            }
        }
    })
}

fn tone_range(p: &Value, cmd: &str) -> Result<ToneRange> {
    match string(p, "range", "midtones") {
        "shadows" => Ok(ToneRange::Shadows),
        "midtones" => Ok(ToneRange::Midtones),
        "highlights" => Ok(ToneRange::Highlights),
        o => Err(bad(cmd, format!("unknown range `{o}` (shadows|midtones|highlights)"))),
    }
}

/// A sequential-dab tool's settings (Dodge, Burn, Sponge, Blur, Sharpen, Smudge), parsed once and
/// turned into its per-dab effect, for the commit and the live preview alike.
#[derive(Clone, Debug)]
enum DabTool {
    Tone { burn: bool, range: ToneRange, exposure: f32, protect: bool },
    Sponge { saturate: bool, vibrance: bool },
    Focus { sharpen: bool, strength: f32, protect: bool, sigma: f32 },
    Smudge { strength: f32, finger: Option<[f32; 4]>, max_size: f32 },
}

/// One dab's edit of the working copy.
pub(crate) type DabEffect = Box<dyn FnMut(&mut Region, &Footprint)>;

impl DabTool {
    /// The tool for `cmd` and its history label.
    fn parse(s: &Session, cmd: &str, p: &Value, stroke: &Stroke) -> Result<(Self, &'static str)> {
        Ok(match cmd {
            "paint.dodge" | "paint.burn" => {
                let burn = cmd == "paint.burn";
                let tool = Self::Tone {
                    burn,
                    range: tone_range(p, cmd)?,
                    exposure: num(p, "exposure", 50.0).clamp(1.0, 100.0) / 100.0,
                    protect: flag(p, "protectTones", true),
                };
                (tool, if burn { "Burn Tool" } else { "Dodge Tool" })
            }
            "paint.sponge" => {
                let saturate = match string(p, "mode", "desaturate") {
                    "desaturate" => false,
                    "saturate" => true,
                    o => return Err(bad(cmd, format!("unknown mode `{o}` (desaturate|saturate)"))),
                };
                (Self::Sponge { saturate, vibrance: flag(p, "vibrance", true) }, "Sponge Tool")
            }
            "paint.blur" | "paint.sharpen" => {
                let sharpen = cmd == "paint.sharpen";
                let tool = Self::Focus {
                    sharpen,
                    strength: num(p, "strength", 50.0).clamp(1.0, 100.0) / 100.0,
                    protect: flag(p, "protectDetail", true),
                    sigma: (stroke.brush.size * 0.03).clamp(1.0, 4.0),
                };
                (tool, if sharpen { "Sharpen Tool" } else { "Blur Tool" })
            }
            "paint.smudge" => {
                let finger = flag(p, "fingerPainting", false).then_some(s.tools.foreground);
                (Self::Smudge { strength: num(p, "strength", 50.0).clamp(1.0, 100.0) / 100.0, finger, max_size: stroke.brush.size }, "Smudge Tool")
            }
            o => return Err(bad(o, "not a sequential-dab tool")),
        })
    }

    /// How far around each dab the effect reads (Blur and Sharpen read a neighbourhood).
    fn halo(&self) -> i32 {
        match self {
            Self::Focus { sigma, .. } => (sigma * 3.0).ceil() as i32 + 1,
            _ => 0,
        }
    }

    /// The per-dab effect on a working copy in `fmt`, for a brush of `spacing`.
    fn effect(&self, fmt: PixelFormat, spacing: f32) -> DabEffect {
        let sp = spacing;
        match self.clone() {
            Self::Tone { burn, range, exposure, protect } => tone_stroke(fmt, move |c| dodge_burn(c, exposure, range, burn, protect)),
            // Flow is already in each dab's coverage; a full-flow pass moves colours half-way.
            Self::Sponge { saturate, vibrance } => Box::new(move |work, fp| color_dab(&fmt, work, fp, 0.5, sp, |c, k| sponge(c, k, saturate, vibrance))),
            Self::Focus { sharpen, strength, protect, sigma } => {
                let a = alpha_index(&fmt);
                Box::new(move |work, fp| {
                    let r = fp.rect.intersect(&work.rect);
                    if r.is_empty() {
                        return;
                    }
                    let (w, h, n) = (work.width(), work.height(), work.ch);
                    let win = ((r.x0 - work.rect.x0) as usize, (r.y0 - work.rect.y0) as usize, (r.x1 - work.rect.x0) as usize, (r.y1 - work.rect.y0) as usize);
                    let out = if sharpen { local_sharpen(&work.data, w, h, n, a, win, 1.0, protect) } else { local_blur(&work.data, w, h, n, a, win, sigma) };
                    let rw = r.width() as usize;
                    for y in r.y0..r.y1 {
                        for x in r.x0..r.x1 {
                            let k = per_dab(fp.at(x, y) * strength, sp);
                            if k <= 0.0 {
                                continue;
                            }
                            let o = ((y - r.y0) as usize * rw + (x - r.x0) as usize) * n;
                            let px = work.px_mut(x, y);
                            for c in 0..n {
                                px[c] += (out[o + c] - px[c]) * k;
                            }
                        }
                    }
                })
            }
            Self::Smudge { strength, finger, max_size } => {
                let finger_px = finger.map(|fg| photocraft_raster::from_rgba(&fmt, fg));
                // Premultiplied mixing: transparent pixels carry no colour, so no dark fringes.
                let mut sm = Smudge::new(strength, finger_px, max_size).with_alpha(alpha_index(&fmt));
                Box::new(move |work, fp| sm.dab(work, fp))
            }
        }
    }
}

/// Dodge, Burn, Sponge, Blur, Sharpen and Smudge: each dab edits a working copy in order
/// (`apply_dab_stroke`). Blur, Sharpen and Smudge can sample all layers.
fn dab_cmd(s: &mut Session, p: &Value, cmd: &str) -> Result<Value> {
    let (stroke, id) = parse_brush(s, p, cmd)?;
    let (tool, label) = DabTool::parse(s, cmd, p, &stroke)?;
    let all = matches!(tool, DabTool::Focus { .. } | DabTool::Smudge { .. }) && sample_all_layers(p);
    let dmg = run_stroke(s, label, id, p, |pre, surf, sel, lock| {
        let fmt = surf.format();
        let mut effect = tool.effect(fmt, stroke.brush.spacing);
        let mut sampler = all.then(|| AllLayers::new(pre, fmt));
        Ok(apply_dab_stroke(surf, &stroke, sel, lock, tool.halo(), |work, fp| match &mut sampler {
            Some(all) => all.dab(work, fp, |w, f| effect(w, f)),
            None => effect(work, fp),
        }))
    })?;
    Ok(json!({ "damage": damage_json(dmg) }))
}

/// A Dodge, Burn, Sponge, Blur, Sharpen or Smudge stroke shown while it is drawn: each new dab
/// runs the tool's effect on a working copy of the target, in order, and the changed area is
/// mixed back over the original at the brush opacity, as `apply_dab_stroke` does on release.
/// Sample All Layers isn't previewed (its composite spans the whole stroke).
pub struct LiveDab {
    /// The active document with the stroke so far.
    pub doc: std::sync::Arc<Document>,
    cmd: String,
    renderer: photocraft_paint::StrokeRenderer,
    done: usize,
    effect: DabEffect,
    halo: i32,
    pre_surf: Surface,
    work: Surface,
    id: Option<LayerId>,
    params: Value,
    opacity: f32,
    sel: Option<Surface>,
    lock: bool,
    bounds: Rect,
    /// Show the dabs finishing the stroke would add, so the preview at the last point is the
    /// commit. Only for effects that a repeated dab doesn't change (Color Replacement): the
    /// tail runs on a scratch copy at every push, so an effect that carries paint from dab to
    /// dab (Smudge) would drift.
    preview_tail: bool,
    /// Where the shown tail is.
    tail: Rect,
}

impl LiveDab {
    /// Start a live stroke of `cmd` from its params; their `points` are rendered.
    pub fn begin(s: &Session, cmd: &str, p: &Value) -> Result<Self> {
        has_pixel_layer(s).map_err(EngineError::Other)?;
        if sample_all_layers(p) {
            return Err(bad(cmd, "Sample All Layers strokes aren't previewed live"));
        }
        let (stroke, id) = parse_brush(s, p, cmd)?;
        let (tool, _) = DabTool::parse(s, cmd, p, &stroke)?;
        let (halo, spacing) = (tool.halo(), stroke.brush.spacing);
        Self::with_effect(s, cmd, p, stroke, id, halo, |pre| tool.effect(pre.format(), spacing))
    }

    /// Start a live stroke of `cmd` that runs `effect` (made from the target before the stroke)
    /// at each dab of `stroke`, reading `halo` pixels around it. `p` gives the target.
    pub(crate) fn with_effect(
        s: &Session,
        cmd: &str,
        p: &Value,
        stroke: Stroke,
        id: Option<LayerId>,
        halo: i32,
        effect: impl FnOnce(&Surface) -> DabEffect,
    ) -> Result<Self> {
        let mut doc = (*s.active().ok_or(EngineError::NoDocument)?.doc).clone();
        let sel = doc.selection.clone();
        let (surf, lock) = crate::channel_cmds::target_surface(&mut doc, id, p)?;
        let pre_surf = surf.clone();
        let effect = effect(&pre_surf);
        // `apply_dab_stroke` generates the same dabs (`dabs`: zoom 1, no smoothing scale).
        let renderer = photocraft_paint::StrokeRenderer::new(&stroke.brush, None, 1.0).record_dabs();
        let mut live = Self {
            doc: std::sync::Arc::new(doc),
            cmd: cmd.into(),
            renderer,
            done: 0,
            effect,
            halo,
            work: pre_surf.clone(),
            pre_surf,
            id,
            params: p.clone(),
            opacity: stroke.brush.opacity,
            sel,
            lock,
            bounds: Rect::EMPTY,
            preview_tail: false,
            tail: Rect::EMPTY,
        };
        live.push(&stroke.points)?;
        Ok(live)
    }

    /// Also show the stroke's finishing dabs (see `preview_tail`).
    pub(crate) fn with_tail_preview(mut self) -> Result<Self> {
        self.preview_tail = true;
        self.push(&[])?;
        Ok(self)
    }

    /// Everything the stroke has touched so far.
    pub fn bounds(&self) -> Rect {
        self.bounds
    }

    /// Render more points; returns the rectangle that changed.
    /// Invalid coordinates reject the whole batch without changing the preview.
    pub fn push(&mut self, pts: &[StrokePoint]) -> Result<Rect> {
        crate::brush_cmds::check_coords(pts, &self.cmd)?;
        self.renderer.push(pts);
        let ctx = self.renderer.ctx.clone();
        let new: Vec<_> = self.renderer.dabs().get(self.done..).unwrap_or_default().to_vec();
        let mut dmg = Rect::EMPTY;
        for (k, d) in new.iter().enumerate() {
            let r = ctx.dab_rect(d, false);
            if r.is_empty() {
                continue;
            }
            let mut region = Region::read(&self.work, r.inflate(self.halo));
            (self.effect)(&mut region, &ctx.footprint(d, self.done + k));
            region.crop(r).write(&mut self.work);
            dmg = dmg.union(&r);
        }
        self.done += new.len();
        // The finishing dabs, run on a scratch copy of the working pixels.
        let mut tail = None;
        if self.preview_tail {
            let ds = self.renderer.tail_dabs();
            let r = ds.iter().fold(Rect::EMPTY, |a, d| a.union(&ctx.dab_rect(d, false)));
            if !r.is_empty() {
                let mut region = Region::read(&self.work, r.inflate(self.halo));
                for (k, d) in ds.iter().enumerate() {
                    (self.effect)(&mut region, &ctx.footprint(d, self.done + k));
                }
                tail = Some((r, region.crop(r)));
            }
        }
        // Back to the stroke without the previous tail.
        let old = std::mem::replace(&mut self.tail, Rect::EMPTY);
        let shown = dmg.union(&old);
        let tail_rect = tail.as_ref().map_or(Rect::EMPTY, |(r, _)| *r);
        if shown.is_empty() && tail_rect.is_empty() {
            return Ok(Rect::EMPTY);
        }
        self.bounds = self.bounds.union(&dmg).union(&tail_rect);
        let (surf, _) = crate::channel_cmds::target_surface(std::sync::Arc::make_mut(&mut self.doc), self.id, &self.params)?;
        if !shown.is_empty() {
            let (orig, work) = (Region::read(&self.pre_surf, shown), Region::read(&self.work, shown));
            photocraft_paint::retouch::mix_back(surf, &orig, &work, shown, self.opacity, self.sel.as_ref(), self.lock);
        }
        if let Some((r, work)) = tail {
            photocraft_paint::retouch::mix_back(surf, &Region::read(&self.pre_surf, r), &work, r, self.opacity, self.sel.as_ref(), self.lock);
            self.tail = r;
        }
        Ok(shown.union(&tail_rect))
    }
}

/// "Sample All Layers" for Spot Healing, Blur, Sharpen and Smudge (layer pixels only: a mask or
/// channel has nothing to sample from other layers).
fn sample_all_layers(p: &Value) -> bool {
    flag(p, "sampleAllLayers", false) && targets_pixels(p)
}

/// Sample All Layers for the sequential-dab tools: each dab's effect runs on a copy of the visible
/// composite (kept across dabs, so a dab sees the earlier ones), and the result is laid onto the
/// layer under the dab. An empty layer above the image so picks up the blurred or smudged picture,
/// and pixels the effect didn't change look as before (they are what was already visible).
struct AllLayers<'a> {
    pre: &'a Document,
    fmt: PixelFormat,
    visible: Option<Region>,
}

impl<'a> AllLayers<'a> {
    fn new(pre: &'a Document, fmt: PixelFormat) -> Self {
        Self { pre, fmt, visible: None }
    }

    fn dab(&mut self, work: &mut Region, fp: &Footprint, effect: impl FnOnce(&mut Region, &Footprint)) {
        let (pre, fmt) = (self.pre, self.fmt);
        let visible = self.visible.get_or_insert_with(|| composite_region(pre, None, SampleLayers::All, work.rect, fmt));
        if visible.rect != work.rect || visible.ch != work.ch {
            return effect(work, fp);
        }
        effect(visible, fp);
        let r = fp.rect.intersect(&work.rect);
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                if fp.at(x, y) > 0.0 {
                    work.px_mut(x, y).copy_from_slice(visible.px(x, y));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------------------------

/// The brush every retouching command shares, plus the tool's own params.
macro_rules! brush_params {
    ($extra:literal) => {
        concat!("{", r#""points":[[x,y,pressure?],…],"size":1..5000 px=tool size,"hardness":0..100=tool hardness,"opacity":1..100=100,"flow":1..100=100,"spacing":1..1000 (% of size)=Brush Settings spacing,"layer":id?=active,"target":"pixels"|"mask"|"quickMask"|{"channel":i}=Channels panel target"#, $extra, "}")
    };
}

/// Retouching command specs.
pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "paint.cloneStamp",
            label: "Clone Stamp",
            menu: &[],
            shortcut: None,
            params: brush_params!(
                r#","source":[sx,sy] (sampled under the first point) | "offset":[dx,dy],"aligned":bool=true,"sampleLayer":"current|currentAndBelow|all"="current","mode":"normal|multiply|…"="normal","wrap":bool=false (repeat source and stroke at document edges for Pattern Preview) → {"damage","offset","aligned","nextSource"}"#
            ),
            enabled: has_pixel_layer,
            run: clone_stamp,
            journal: true,
        },
        CommandSpec {
            id: "paint.patternStamp",
            label: "Pattern Stamp",
            menu: &[],
            shortcut: None,
            params: brush_params!(
                r#","pattern":id|name?=Patterns panel current,"scale":1..1000 %=100,"angle":deg=0,"aligned":bool=true,"impressionist":bool=false,"mode":"normal|multiply|…"="normal","phase":[px,py]? → {"damage","phase","aligned","pattern"}"#
            ),
            enabled: pattern_stamp::enabled,
            run: pattern_stamp::run,
            journal: true,
        },
        CommandSpec {
            id: "paint.healingBrush",
            label: "Healing Brush",
            menu: &[],
            shortcut: None,
            params: brush_params!(
                r#","source":[sx,sy] | "offset":[dx,dy],"aligned":bool=true,"sampleLayer":"current|currentAndBelow|all"="current","mode":"normal|…"="normal" → {"damage","offset","aligned","nextSource"}"#
            ),
            enabled: has_pixel_layer,
            run: healing_brush,
            journal: true,
        },
        CommandSpec {
            id: "paint.spotHealing",
            label: "Spot Healing Brush",
            menu: &[],
            shortcut: None,
            params: brush_params!(r#","type":"contentAware|createTexture|proximityMatch"="contentAware","sampleAllLayers":bool=false"#),
            enabled: has_pixel_layer,
            run: spot_healing,
            journal: true,
        },
        CommandSpec {
            id: "paint.patch",
            label: "Patch",
            menu: &[],
            shortcut: None,
            params: r#"{"offset":[dx,dy] (how far the selection was dragged),"mode":"source|destination"="source","contentAware":bool=false (fill the selection content-aware from the dragged-to place; a background job),"structure":1..7=4 (content-aware: 3..7 copy the middle exactly, 1..2 re-synthesise more of it),"color":0..10=0 (content-aware: how far the copy's level adapts to the selection's surroundings),"sampleAllLayers":bool=false (content-aware),"layer":id?=active,"target":"pixels"|"mask"|"quickMask"|{"channel":i}=Channels panel target} → {"damage","offset"}"#,
            enabled: patch::enabled,
            run: patch::patch,
            journal: true,
        },
        CommandSpec {
            id: "paint.contentAwareMove",
            label: "Content-Aware Move",
            menu: &[],
            shortcut: None,
            params: r#"{"offset":[dx,dy] (how far the selection was dragged),"mode":"move|extend"="move","structure":1..7=4 (3..7 copy the content exactly but for a band about a patch wide along its edge; 2 widens the band, 1 re-synthesises the content too),"color":0..10=0 (0 keeps the content's level; higher values bring it towards its new place, by up to about 0.04 × color),"sampleAllLayers":bool=false,"scaleX":%=100,"scaleY":%=100 (Transform On Drop: 1..10000, negative flips),"rotate":deg=0 (Transform On Drop, clockwise; scale and rotation are about the selection's centre, before the offset; a transformed patch may reach past the canvas edge),"layer":id?=active,"target":"pixels"|"mask"|"quickMask"|{"channel":i}=Channels panel target} → {"damage","offset","mode","transformed"?} (a background job; the selection moves, and is transformed, with the content)"#,
            enabled: content_aware_move::enabled,
            run: content_aware_move::content_aware_move,
            journal: true,
        },
        CommandSpec {
            id: "paint.remove",
            label: "Remove",
            menu: &[],
            shortcut: None,
            params: brush_params!(
                r#","strokes":[[[x,y,pressure?],…],…]? (several strokes removed together, as one job and one undo step; else "points"),"closeLoops":bool=true (a stroke around an object removes what it encloses),"sampleAllLayers":bool=false,"newLayer":bool=false (the fill goes to a new layer above the target, which stays as it was) → {"damage","layer"?} (a background job)"#
            ),
            enabled: has_pixel_layer,
            run: remove::remove,
            journal: true,
        },
        CommandSpec {
            id: "paint.dodge",
            label: "Dodge",
            menu: &[],
            shortcut: None,
            params: brush_params!(r#","range":"shadows|midtones|highlights"="midtones","exposure":1..100=50,"protectTones":bool=true"#),
            enabled: has_pixel_layer,
            run: |s, p| dab_cmd(s, p, "paint.dodge"),
            journal: true,
        },
        CommandSpec {
            id: "paint.burn",
            label: "Burn",
            menu: &[],
            shortcut: None,
            params: brush_params!(r#","range":"shadows|midtones|highlights"="midtones","exposure":1..100=50,"protectTones":bool=true"#),
            enabled: has_pixel_layer,
            run: |s, p| dab_cmd(s, p, "paint.burn"),
            journal: true,
        },
        CommandSpec {
            id: "paint.sponge",
            label: "Sponge",
            menu: &[],
            shortcut: None,
            params: brush_params!(r#","mode":"desaturate|saturate"="desaturate","vibrance":bool=true (flow = sponge strength)"#),
            enabled: has_pixel_layer,
            run: |s, p| dab_cmd(s, p, "paint.sponge"),
            journal: true,
        },
        CommandSpec {
            id: "paint.blur",
            label: "Blur",
            menu: &[],
            shortcut: None,
            params: brush_params!(r#","strength":1..100=50,"sampleAllLayers":bool=false"#),
            enabled: has_pixel_layer,
            run: |s, p| dab_cmd(s, p, "paint.blur"),
            journal: true,
        },
        CommandSpec {
            id: "paint.sharpen",
            label: "Sharpen",
            menu: &[],
            shortcut: None,
            params: brush_params!(r#","strength":1..100=50,"protectDetail":bool=true,"sampleAllLayers":bool=false"#),
            enabled: has_pixel_layer,
            run: |s, p| dab_cmd(s, p, "paint.sharpen"),
            journal: true,
        },
        CommandSpec {
            id: "paint.smudge",
            label: "Smudge",
            menu: &[],
            shortcut: None,
            params: brush_params!(r#","strength":1..100=50,"fingerPainting":bool=false (starts with the foreground colour),"sampleAllLayers":bool=false"#),
            enabled: has_pixel_layer,
            run: |s, p| dab_cmd(s, p, "paint.smudge"),
            journal: true,
        },
        CommandSpec {
            id: "paint.historyBrush",
            label: "History Brush",
            menu: &[],
            shortcut: None,
            params: brush_params!(r#","state":index (into history.entries; default 0 = the oldest held state, normally the Open snapshot)"#),
            enabled: has_pixel_layer,
            run: history_brush,
            journal: true,
        },
    ]
}

#[cfg(test)]
mod tests;
