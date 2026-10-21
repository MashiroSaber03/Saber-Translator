//! The rest of Photoshop's Edit menu: Fade, Purge, Content-Aware Fill and Scale, Define Brush
//! Preset / Custom Shape, Find and Replace Text, the Preset Manager and preset export/import.
//! Auto-Align and Auto-Blend Layers live in [`crate::align_cmds`].
//!
//! Check Spelling is not implemented: it needs a word list per language, and no permissively
//! licensed dictionary small enough to bundle (or a pure-Rust checker with one) was available.

use photocraft_color::{BlendMode, PixelFormat};
use photocraft_doc::vector::Path;
use photocraft_doc::{DocId, Document, Layer, LayerContent, LayerId};
use photocraft_geom::Rect;
use photocraft_raster::{Surface, from_rgba_into, to_rgba};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, blend_from_str, int, layer_param};
use crate::{EngineError, Result, Session};

// ------------------------------------------------------------------ state

/// What Edit › Fade would fade: the result of the last filter, adjustment or paint command.
#[derive(Clone, Debug, PartialEq)]
pub struct FadeSource {
    pub doc: DocId,
    /// Document revision right after the command; any later change disables Fade.
    pub revision: u64,
    pub layer: LayerId,
    pub mask: bool,
    pub label: String,
}

/// A custom shape (Edit › Define Custom Shape), usable by the Custom Shape tool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CustomShape {
    pub name: String,
    pub path: Path,
}

/// Edit-menu state kept by the session.
#[derive(Clone, Debug, Default)]
pub struct EditState {
    pub fade: Option<FadeSource>,
    /// Custom shape library (the vector tools read it).
    pub custom_shapes: Vec<CustomShape>,
    /// Find and Replace Text position: (layer, byte offset after the current match).
    pub find_cursor: Option<(LayerId, usize, usize)>,
}

/// Commands whose result Edit › Fade can blend back against the previous state.
fn fadeable(id: &str) -> bool {
    id.starts_with("filter.")
        || id.starts_with("image.adjustments.")
        || id.starts_with("paint.")
        || matches!(id, "edit.fill" | "edit.stroke" | "edit.contentAwareFill" | "image.autoTone" | "image.autoContrast" | "image.autoColor")
}

/// Called after every successful command: remember a fadeable result.
pub(crate) fn after_command(s: &mut Session, id: &str) {
    if !fadeable(id) {
        return;
    }
    let Some(st) = s.active() else { return };
    let Some(layer) = st.active_layer else { return };
    let label = st.history.undo_label().unwrap_or(id).to_string();
    let mask = st.doc.layer(layer).is_some_and(|l| l.mask.is_some())
        && st.history.state(st.history.past_len().wrapping_sub(1)).is_some_and(|prev| {
            // The mask changed and the pixels didn't: the command painted the mask.
            let now = st.doc.layer(layer);
            let was = prev.layer(layer);
            let surf_eq = |a: Option<&Surface>, b: Option<&Surface>| match (a, b) {
                (Some(a), Some(b)) => crate_fingerprint(a) == crate_fingerprint(b),
                (None, None) => true,
                _ => false,
            };
            surf_eq(now.and_then(|l| l.surface()), was.and_then(|l| l.surface()))
                && !surf_eq(now.and_then(|l| l.mask.as_ref().map(|m| &m.surface)), was.and_then(|l| l.mask.as_ref().map(|m| &m.surface)))
        });
    s.edit_state.fade = Some(FadeSource { doc: st.doc.id, revision: st.revision, layer, mask, label });
}

/// Tile identity of a surface (pointer equality of its COW tiles).
fn crate_fingerprint(s: &Surface) -> u64 {
    s.tiles().fold(s.tile_count() as u64, |h, (c, t)| {
        (h ^ std::sync::Arc::as_ptr(t) as usize as u64 ^ ((c.tx as u64) << 32 | c.ty as u32 as u64)).wrapping_mul(0x100_0000_01b3)
    })
}

// ------------------------------------------------------------------ helpers

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn f32_or(p: &Value, k: &str, d: f32) -> f32 {
    p.get(k).and_then(Value::as_f64).map_or(d, |v| v as f32)
}

fn str_or<'a>(p: &'a Value, k: &str, d: &'a str) -> &'a str {
    p.get(k).and_then(Value::as_str).unwrap_or(d)
}

fn bool_or(p: &Value, k: &str, d: bool) -> bool {
    p.get(k).and_then(Value::as_bool).unwrap_or(d)
}

fn always(_: &Session) -> std::result::Result<(), String> {
    Ok(())
}

fn pixel_layer(s: &Session) -> std::result::Result<LayerId, String> {
    let d = s.active().ok_or("no document open")?;
    let id = d.active_layer.ok_or("no active layer")?;
    let l = d.doc.layer(id).ok_or("no active layer")?;
    if !matches!(l.content, LayerContent::Raster(_)) {
        return Err(format!("active layer is {} {} layer, not a pixel layer", l.content.article(), l.content.kind_name()));
    }
    let locks = d.doc.effective_locks(id);
    if locks.all || locks.pixels {
        return Err(format!("the layer \"{}\" is locked", l.name));
    }
    Ok(id)
}

fn writable_surface(doc: &mut Document, id: LayerId) -> Result<&mut Surface> {
    let locks = doc.effective_locks(id);
    let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
    if locks.all || locks.pixels {
        return Err(EngineError::Other(format!("Could not complete your request because the layer \"{}\" is locked", l.name)));
    }
    l.surface_mut().ok_or_else(|| EngineError::Other("not a pixel layer".into()))
}

// ------------------------------------------------------------------ Fade

fn can_fade(s: &Session) -> std::result::Result<(), String> {
    let st = s.active().ok_or("no document open")?;
    let f = s.edit_state.fade.as_ref().ok_or("nothing to fade")?;
    if f.doc != st.doc.id || f.revision != st.revision || !st.history.can_undo() {
        return Err("Fade is only available right after a filter, adjustment or painting step".into());
    }
    if st.doc.layer(f.layer).is_none() {
        return Err("the faded layer no longer exists".into());
    }
    Ok(())
}

fn surface_of(doc: &Document, id: LayerId, mask: bool) -> Option<&Surface> {
    let l = doc.layer(id)?;
    if mask { l.mask.as_ref().map(|m| &m.surface) } else { l.surface() }
}

/// Blend `after` back towards `before` over `area`: `before + opacity × (B(before, after) − before)`
/// with blend mode `mode` (premultiplied, so alpha changes fade too).
pub fn fade_surface(before: &Surface, after: &mut Surface, area: Rect, mode: BlendMode, opacity: f32) -> Result<()> {
    let fmt = after.format();
    let n = fmt.channels();
    if area.is_empty() || n == 0 {
        return Ok(());
    }
    let b = crate::allocation::read_region(before, area, "fading pixels")?;
    let mut a = crate::allocation::read_region(after, area, "fading pixels")?;
    let k = opacity.clamp(0.0, 1.0);
    let lerp_premul = |pb: &[f32], pa: &[f32], out: &mut [f32], alpha: bool| {
        let m = out.len();
        if !alpha {
            for c in 0..m {
                out[c] = pb[c] + (pa[c] - pb[c]) * k;
            }
            return;
        }
        let (ab, aa) = (pb[m - 1], pa[m - 1]);
        let ao = ab + (aa - ab) * k;
        for c in 0..m - 1 {
            let pm = pb[c] * ab + (pa[c] * aa - pb[c] * ab) * k;
            out[c] = if ao > 1e-9 { pm / ao } else { 0.0 };
        }
        out[m - 1] = ao;
    };
    let mut tmp = crate::allocation::filled(n, 0.0f32, "fading pixels")?;
    for (pa, pb) in a.chunks_exact_mut(n).zip(b.chunks_exact(n)) {
        if mode == BlendMode::Normal || n == 1 {
            lerp_premul(pb, pa, &mut tmp, fmt.alpha);
            pa.copy_from_slice(&tmp);
        } else {
            let rb = to_rgba(&fmt, pb);
            let ra = to_rgba(&fmt, pa);
            let blended = photocraft_color::blend::composite(mode, rb, ra, 1.0);
            let mut o = [0.0f32; 4];
            lerp_premul(&rb, &blended, &mut o, true);
            let mut enc = [0.0f32; 8];
            from_rgba_into(&fmt, o, &mut enc);
            pa.copy_from_slice(&enc[..n]);
        }
    }
    after.try_write_region(area, &a)?;
    Ok(())
}

fn fade(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "edit.fade";
    let src = s.edit_state.fade.clone().ok_or_else(|| bad(cmd, "nothing to fade"))?;
    let opacity = f32_or(p, "opacity", 100.0).clamp(0.0, 100.0) / 100.0;
    let mode = blend_from_str(str_or(p, "mode", "normal")).ok_or_else(|| bad(cmd, "unknown blend mode"))?;
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let prev = st.history.state(st.history.past_len() - 1).ok_or_else(|| bad(cmd, "no previous state"))?;
    let before = surface_of(&prev, src.layer, src.mask).cloned().ok_or_else(|| EngineError::Other("the layer did not exist before this step".into()))?;
    let now = surface_of(&st.doc, src.layer, src.mask).ok_or(EngineError::NoLayer(src.layer))?;
    if now.format() != before.format() {
        return Err(EngineError::Other("can't fade a step that changed the layer's pixel format".into()));
    }
    let area = before.content_bounds().union(&now.content_bounds());
    let label = format!("Fade {}", src.label);
    s.edit(&label, |doc, _| {
        let l = doc.layer_mut(src.layer).ok_or(EngineError::NoLayer(src.layer))?;
        let surf = if src.mask { l.mask.as_mut().map(|m| &mut m.surface) } else { l.surface_mut() }.ok_or(EngineError::NoLayer(src.layer))?;
        fade_surface(&before, surf, area, mode, opacity)?;
        surf.prune();
        Ok(())
    })?;
    Ok(json!({"faded": src.label, "opacity": opacity * 100.0, "mode": mode.label()}))
}

// ------------------------------------------------------------------ Purge

fn history_bytes(s: &Session, all: bool) -> usize {
    let docs: Vec<&crate::DocState> = if all { s.documents().iter().collect() } else { s.active().into_iter().collect() };
    docs.iter().map(|d| d.history.unique_bytes(&d.doc)).sum()
}

fn clip_bytes(s: &Session) -> usize {
    s.clipboard.as_ref().map_or(0, |c| c.surface.tiles().map(|(_, t)| t.bytes().len()).sum())
}

fn purge(s: &mut Session, what: &str) -> Result<Value> {
    let mut freed = 0usize;
    let mut items: Vec<&str> = Vec::new();
    if matches!(what, "undo" | "all") && s.active().is_some_and(|d| d.history.can_undo()) {
        let before = history_bytes(s, false);
        if let Some(st) = s.active_mut() {
            st.history.purge_last();
        }
        freed += before.saturating_sub(history_bytes(s, false));
        items.push("undo");
    }
    if matches!(what, "clipboard" | "all") && (s.clipboard.is_some() || s.style_clipboard.is_some()) {
        freed += clip_bytes(s);
        s.clipboard = None;
        s.style_clipboard = None;
        items.push("clipboard");
    }
    if matches!(what, "histories" | "all") {
        let b = history_bytes(s, true);
        if s.documents().iter().any(|d| d.history.can_undo() || d.history.can_redo()) {
            for st in &mut s.docs {
                st.history.clear();
                st.coalesce = None;
            }
            freed += b;
            items.push("histories");
        }
    }
    if what == "all" {
        let fx = photocraft_compose::purge_effect_cache();
        if fx > 0 {
            freed += fx;
            items.push("effect cache");
        }
        let masks = photocraft_compose::masks::purge_cache();
        if masks > 0 {
            freed = freed.saturating_add(masks);
            items.push("mask cache");
        }
        let stages = crate::smart_cmds::purge_stack_cache();
        if stages > 0 {
            freed = freed.saturating_add(stages);
            items.push("smart filter cache");
        }
        let strokes = photocraft_compose::shape_split::purge_cache();
        if strokes > 0 {
            freed = freed.saturating_add(strokes);
            items.push("shape stroke cache");
        }
    }
    s.edit_state.fade = None;
    let msg = if items.is_empty() { "nothing to purge".to_string() } else { format!("purged {}", items.join(", ")) };
    Ok(json!({"purged": items, "bytes": freed, "message": msg}))
}

fn can_purge_undo(s: &Session) -> std::result::Result<(), String> {
    s.active().filter(|d| d.history.can_undo()).map(|_| ()).ok_or_else(|| "nothing to purge".into())
}
fn can_purge_clipboard(s: &Session) -> std::result::Result<(), String> {
    if s.clipboard.is_some() || s.style_clipboard.is_some() { Ok(()) } else { Err("the clipboard is empty".into()) }
}
fn can_purge_histories(s: &Session) -> std::result::Result<(), String> {
    if s.documents().iter().any(|d| d.history.can_undo() || d.history.can_redo()) { Ok(()) } else { Err("no history to purge".into()) }
}
fn can_purge_all(s: &Session) -> std::result::Result<(), String> {
    can_purge_histories(s).or_else(|_| can_purge_clipboard(s)).or_else(|_| {
        if photocraft_compose::effect_cache_bytes() > 0 || photocraft_compose::masks::cache_bytes() > 0 || crate::smart_cmds::stack_cache_bytes() > 0 {
            Ok(())
        } else {
            Err("nothing to purge".into())
        }
    })
}

// ------------------------------------------------------------------ Content-Aware Fill

fn can_caf(s: &Session) -> std::result::Result<(), String> {
    pixel_layer(s)?;
    let d = s.active().ok_or("no document open")?;
    if d.doc.selection.as_ref().is_none_or(|m| m.content_bounds().is_empty()) {
        return Err("make a selection around what to remove".into());
    }
    Ok(())
}

/// Mask surface (gray) of an alpha channel by index or name.
fn channel_mask<'a>(doc: &'a Document, v: &Value) -> Option<&'a Surface> {
    match v {
        Value::Number(n) => doc.channels.get(n.as_u64()? as usize).map(|c| &c.surface),
        Value::String(name) => doc.channels.iter().find(|c| c.name == *name).map(|c| &c.surface),
        _ => None,
    }
}

/// `key` as an `[x, y, w, h]` rectangle. The saturating float→int casts bound each component;
/// the additions saturate too — `area: [1e30, 0, 1e30, 10]` is a whole-canvas window, not an
/// overflow (it panicked in debug builds before the `saturating_add`).
fn rect_param(p: &Value, key: &str, cmd: &str) -> Result<Rect> {
    let a = p.get(key).and_then(Value::as_array).ok_or_else(|| bad(cmd, format!("`{key}` must be [x, y, w, h]")))?;
    let n = |i: usize| {
        a.get(i)
            .and_then(Value::as_f64)
            .filter(|f| f.is_finite())
            .map(|v| v.round() as i32)
            .ok_or_else(|| bad(cmd, format!("`{key}` must be four finite numbers")))
    };
    let (x, y, w, h) = (n(0)?, n(1)?, n(2)?, n(3)?);
    Ok(Rect::new(x, y, x.saturating_add(w.max(0)), y.saturating_add(h.max(0))))
}

fn content_aware_fill(s: &mut Session, p: &Value) -> Result<Value> {
    content_aware_fill_as(s, p, "edit.contentAwareFill", "Content-Aware Fill")
}

/// Delete and Fill Selection (#1286): Photoshop's one-click removal from the selection-tool
/// context menu. Content-Aware Fill into the layer, no dialog, sampling as Edit › Fill does (the
/// whole window, measured on Photoshop 25.4) with the default colour adaptation.
fn delete_and_fill(s: &mut Session, _: &Value) -> Result<Value> {
    content_aware_fill_as(s, &json!({"sampling": "rectangular"}), "edit.deleteAndFillSelection", "Delete and Fill Selection")
}

fn content_aware_fill_as(s: &mut Session, p: &Value, cmd: &'static str, label: &'static str) -> Result<Value> {
    use photocraft_algo::content_aware::{FillOptions, auto_sampling_with, color_level, fill_owned, rotation_level};
    let id = pixel_layer(s).map_err(EngineError::Other)?;
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let doc = st.doc.clone();
    let sel = doc.selection.clone().ok_or_else(|| bad(cmd, "no selection"))?;
    let hb = sel.content_bounds().intersect(&doc.bounds());
    if hb.is_empty() {
        return Err(bad(cmd, "the selection is outside the canvas"));
    }
    let sampling = str_or(p, "sampling", "auto").to_string();
    let custom_mask: Option<Surface> = p.get("channel").and_then(|v| channel_mask(&doc, v)).cloned();
    let custom_rect = if p.get("area").is_some() { Some(rect_param(p, "area", cmd)?) } else { None };
    // Photoshop's window (`content_aware::sampling_window`) for Auto and Rectangular; `margin`
    // asks for a rectangle that far around the selection instead.
    let window = match sampling.as_str() {
        "auto" => crate::fill_cmds::sampling_window(hb, doc.bounds()),
        "rectangular" => match int(p, "margin") {
            Some(m) => hb.inflate(m.clamp(0, 100_000) as i32),
            None => crate::fill_cmds::sampling_window(hb, doc.bounds()),
        },
        "custom" => {
            let r = match (custom_rect, custom_mask.as_ref()) {
                (Some(r), _) => r,
                (None, Some(m)) => m.content_bounds(),
                (None, None) => return Err(bad(cmd, "custom sampling needs `area` [x,y,w,h] or `channel` (alpha channel index or name)")),
            };
            r.union(&hb).inflate(4)
        }
        other => return Err(bad(cmd, format!("unknown sampling `{other}` (auto|rectangular|custom)"))),
    }
    .intersect(&doc.bounds());
    let opts = FillOptions {
        color_adaptation: color_level(str_or(p, "colorAdaptation", "default")),
        rotations: rotation_level(str_or(p, "rotationAdaptation", "none")),
        scale: bool_or(p, "scale", false),
        mirror: bool_or(p, "mirror", false),
        seed: p.get("seed").and_then(Value::as_u64).unwrap_or(1),
    };
    let output = str_or(p, "output", "current").to_string();
    if !matches!(output.as_str(), "current" | "new" | "duplicate") {
        return Err(bad(cmd, "`output` must be current|new|duplicate"));
    }
    let surf = doc.layer(id).and_then(|l| l.surface()).ok_or(EngineError::NoLayer(id))?;
    let fmt = surf.format();
    // Photoshop's Sample All Layers checkbox is off unless the caller asks. When the active
    // layer has no pixels in the window there is nothing to sample, so that case uses the
    // visible composite (#4026). An explicit false still reads only this layer.
    let sample_all = match p.get("sampleAllLayers").and_then(Value::as_bool) {
        Some(v) => v,
        None => surf.content_bounds().intersect(&window).is_empty(),
    };
    let n = fmt.channels();
    // Refuse a window too large to read before allocating for it, whatever the sampling (#963).
    let (w, h) = crate::fill_cmds::window_size(window)?;
    // A background job when started with `Session::start` (#210): reading the window and the
    // PatchMatch fill run on a worker against the document snapshot, cancellable per row band.
    crate::jobs::run(
        s,
        label,
        true,
        move |ctx| {
            use rayon::prelude::*;

            use crate::fill_cmds::{read_window, read_window_cover, read_window_mask};
            ctx.progress(0.0, label);
            let cancel = || ctx.cancelled();
            let surf = doc.layer(id).and_then(|l| l.surface()).ok_or(EngineError::NoLayer(id))?;
            let img = if sample_all {
                if cancel() {
                    return Err(EngineError::Cancelled);
                }
                let region = crate::retouch_cmds::composite_region(&doc, None, crate::retouch_cmds::SampleLayers::All, window, fmt);
                if cancel() {
                    return Err(EngineError::Cancelled);
                }
                region.data
            } else {
                read_window(surf, window, &cancel).ok_or(EngineError::Cancelled)?
            };
            let hole = read_window_cover(&sel, window, &cancel).ok_or(EngineError::Cancelled)?;
            let mut source = match (sampling.as_str(), custom_rect, custom_mask.as_ref()) {
                ("custom", Some(r), _) => (0..w * h).into_par_iter().map(|i| r.contains(window.x0 + (i % w) as i32, window.y0 + (i / w) as i32)).collect(),
                ("custom", None, Some(m)) => read_window_mask(m, window, &cancel).ok_or(EngineError::Cancelled)?.par_iter().map(|v| *v > 0.5).collect(),
                _ => vec![true; w * h],
            };
            ctx.check()?;
            // Auto: only what looks like the selection's surroundings.
            if sampling == "auto" {
                source = auto_sampling_with(w, h, n, &img, &hole, &photocraft_raster::Interrupt::cancel_only(&cancel)).map_err(|_| EngineError::Cancelled)?;
            }
            ctx.check()?;
            // The fill changes only the selected pixels: keep the original and the coverage over
            // their bounds for the apply step, and fill the window's pixels themselves.
            let (bx0, by0, bx1, by1) = mask_bounds(w, h, &hole);
            let original: Vec<f32> = (by0..by1).flat_map(|y| img[(y * w + bx0) * n..(y * w + bx1) * n].iter().copied()).collect();
            let at = |x: usize, y: usize| (window.x0 + x as i32, window.y0 + y as i32);
            let ((x0, y0), (x1, y1)) = (at(bx0, by0), at(bx1, by1));
            let cover = read_window_mask(&sel, Rect::new(x0, y0, x1, y1), &cancel).ok_or(EngineError::Cancelled)?;
            let n_hole = cover.iter().filter(|c| **c > 0.0).count();
            let filled = ctx.stage(0.05, 1.0, label, |ctl| fill_owned(w, h, n, img, &hole, &source, &opts, ctl)).map_err(|_| EngineError::Cancelled)?;
            Ok(FilledWindow { filled, bounds: (bx0, by0, bx1, by1), original, cover, n_hole })
        },
        move |s, done| apply_content_aware_fill(s, label, id, &output, fmt, window, done),
    )
}

/// A Content-Aware Fill's result for its apply step: the filled window (the same as the
/// original outside the selected pixels) and, over the bounds of the selected pixels (in window
/// coordinates), the original pixels and the selection's coverage.
struct FilledWindow {
    filled: Vec<f32>,
    bounds: (usize, usize, usize, usize),
    original: Vec<f32>,
    cover: Vec<f32>,
    n_hole: usize,
}

/// `f(window index, index over the bounds, coverage)` for each selected pixel of a
/// [`FilledWindow`] (`w` wide).
fn each_selected(w: usize, (bx0, by0, bx1, by1): (usize, usize, usize, usize), cover: &[f32], mut f: impl FnMut(usize, usize, f32)) {
    let bw = bx1.saturating_sub(bx0);
    for y in by0..by1 {
        for x in bx0..bx1 {
            let k = (y - by0) * bw + (x - bx0);
            if let Some(&c) = cover.get(k)
                && c > 0.0
            {
                f(y * w + x, k, c);
            }
        }
    }
}

/// Bounds `(x0, y0, x1, y1)` of the set cells of a `w × h` mask (all zero when there are none).
fn mask_bounds(w: usize, h: usize, m: &[bool]) -> (usize, usize, usize, usize) {
    use rayon::prelude::*;
    if w == 0 {
        return (0, 0, 0, 0);
    }
    let rows: Vec<Option<(usize, usize)>> =
        m.par_chunks(w).take(h).map(|row| Some((row.iter().position(|v| *v)?, row.iter().rposition(|v| *v)? + 1))).collect();
    let mut b: Option<(usize, usize, usize, usize)> = None;
    for (y, span) in rows.into_iter().enumerate() {
        if let Some((x0, x1)) = span {
            b = Some(b.map_or((x0, y, x1, y + 1), |(bx0, by0, bx1, _)| (bx0.min(x0), by0, bx1.max(x1), y + 1)));
        }
    }
    b.unwrap_or((0, 0, 0, 0))
}

/// Write a Content-Aware Fill result (the job's apply step): into the layer, a new layer or a
/// duplicate, as one undo step.
fn apply_content_aware_fill(s: &mut Session, label: &str, id: LayerId, output: &str, fmt: PixelFormat, window: Rect, done: FilledWindow) -> Result<Value> {
    let n = fmt.channels();
    let (w, h) = (window.width() as usize, window.height() as usize);
    let FilledWindow { mut filled, bounds, original, cover, n_hole } = done;
    let layer = s.edit(label, |doc, active| {
        let target = match output {
            "new" => {
                let mut l = Layer::raster(doc.next_layer_name("Content-Aware Fill"), PixelFormat { alpha: true, ..fmt });
                let mut px = vec![0.0f32; w * h * n.max(1)];
                let lf = l.surface().map(|s| s.format()).unwrap_or(fmt);
                let ln = lf.channels();
                px.resize(w * h * ln, 0.0);
                each_selected(w, bounds, &cover, |i, _, c| {
                    let rgba = to_rgba(&fmt, &filled[i * n..(i + 1) * n]);
                    let mut enc = [0.0f32; 8];
                    from_rgba_into(&lf, [rgba[0], rgba[1], rgba[2], rgba[3] * c], &mut enc);
                    px[i * ln..(i + 1) * ln].copy_from_slice(&enc[..ln]);
                });
                if let Some(s) = l.surface_mut() {
                    s.write_region(window, &px);
                    s.prune();
                }
                let nid = doc.insert_above(Some(id), l);
                *active = Some(nid);
                return Ok(nid);
            }
            "duplicate" => {
                let mut dup = doc.layer(id).ok_or(EngineError::NoLayer(id))?.duplicate();
                dup.name = doc.copy_name(&dup.name);
                let nid = doc.insert_above(Some(id), dup);
                *active = Some(nid);
                nid
            }
            _ => id,
        };
        let surf = writable_surface(doc, target)?;
        // The fill blended in by the coverage (elsewhere it is the original).
        each_selected(w, bounds, &cover, |i, k, c| {
            for ch in 0..n {
                let (j, o) = (i * n + ch, k * n + ch);
                filled[j] = original[o] + (filled[j] - original[o]) * c;
            }
        });
        surf.write_region(window, &filled);
        Ok(target)
    })?;
    Ok(json!({"layer": layer.0, "filled": n_hole, "window": [window.x0, window.y0, window.width(), window.height()]}))
}

// ------------------------------------------------------------------ Content-Aware Scale

/// Reach of Content-Aware Scale's `x`/`y`: the largest document side.
const CAS_POSITION_LIMIT: f64 = 300_000.0;

/// What Content-Aware Scale does, read from its params against the active document: the
/// command and the transform box's live preview ([`ContentAwareScalePreview`]) share it.
struct CasPlan {
    doc: std::sync::Arc<Document>,
    layer: LayerId,
    /// The area scaled: the selection's bounds, else the layer's content.
    region: Rect,
    /// Where the result goes: `x`, `y` (default: the region's corner), `width`, `height`.
    dst: Rect,
    amount: f32,
    protect: Option<Surface>,
    skin_tones: bool,
}

fn cas_plan(s: &Session, p: &Value) -> Result<CasPlan> {
    let cmd = "edit.contentAwareScale";
    let id = pixel_layer(s).map_err(EngineError::Other)?;
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let doc = st.doc.clone();
    let surf = doc.layer(id).and_then(|l| l.surface()).ok_or(EngineError::NoLayer(id))?;
    let region = match &doc.selection {
        Some(sel) => sel.content_bounds().intersect(&doc.bounds()),
        None => surf.content_bounds(),
    };
    if region.is_empty() {
        return Err(EngineError::Other("nothing to scale: the layer is empty".into()));
    }
    let (w, h) = (region.width() as usize, region.height() as usize);
    let dim = |k: &str, pct: &str, cur: usize| -> Result<u32> {
        let v = match (p.get(k).and_then(Value::as_f64), p.get(pct).and_then(Value::as_f64)) {
            (Some(px), _) => px,
            (None, Some(pc)) => cur as f64 * pc / 100.0,
            (None, None) => cur as f64,
        };
        if !(1.0..=60_000.0).contains(&v) {
            return Err(bad(cmd, format!("`{k}` must be 1..60000 px")));
        }
        Ok(v.round() as u32)
    };
    let (nw, nh) = (dim("width", "scaleX", w)?, dim("height", "scaleY", h)?);
    let at = |k: &str, cur: i32| -> Result<i32> {
        match p.get(k).filter(|v| !v.is_null()) {
            None => Ok(cur),
            Some(v) => match v.as_f64() {
                Some(x) if (-CAS_POSITION_LIMIT..=CAS_POSITION_LIMIT).contains(&x) => Ok(x.round() as i32),
                _ => Err(bad(cmd, format!("`{k}` must be a number of px within ±300000"))),
            },
        }
    };
    let dst = Rect::from_xywh(at("x", region.x0)?, at("y", region.y0)?, nw, nh);
    let amount = f32_or(p, "amount", 100.0).clamp(0.0, 100.0) / 100.0;
    // Protection: an alpha channel and/or skin tones.
    let protect = match p.get("protect").filter(|v| !v.is_null() && v.as_str() != Some("none")) {
        Some(v) => Some(channel_mask(&doc, v).cloned().ok_or_else(|| bad(cmd, "`protect` must name an alpha channel (index or name) or \"none\""))?),
        None => None,
    };
    let skin_tones = bool_or(p, "protectSkinTones", false);
    Ok(CasPlan { doc, layer: id, region, dst, amount: if amount.is_finite() { amount } else { 1.0 }, protect, skin_tones })
}

/// The scaled pixels: `img` (`w`×`h`, `fmt`) seam-carved to `nw`×`nh` where the protection
/// (`protect`, one value per pixel, and the skin tones when asked) steers the seams, blended with
/// a plain resample by `amount` (0..1). Checks `ctx` for cancellation before every seam.
#[allow(clippy::too_many_arguments)]
fn cas_pixels(
    ctx: &crate::jobs::JobCtx,
    label: &str,
    img: &[f32],
    (w, h): (usize, usize),
    fmt: PixelFormat,
    mut protect: Option<Vec<f32>>,
    skin_tones: bool,
    (nw, nh): (usize, usize),
    amount: f32,
) -> Result<Vec<f32>> {
    use photocraft_algo::seam::{carve_with, resize_bilinear, skin_mask};
    let n = fmt.channels();
    if skin_tones {
        let rgba: Vec<f32> = img.chunks_exact(n).flat_map(|px| to_rgba(&fmt, px)).collect();
        let skin = skin_mask(w, h, 4, &rgba);
        protect = Some(match protect {
            Some(pm) => pm.iter().zip(&skin).map(|(a, b)| a.max(*b)).collect(),
            None => skin,
        });
    }
    ctx.check()?;
    let carved = if amount > 0.0 {
        ctx.stage(0.05, 0.95, label, |ctl| carve_with(w, h, n, img, protect.as_deref(), nw, nh, ctl)).map_err(|_| EngineError::Cancelled)?
    } else {
        Vec::new()
    };
    let plain = if amount < 1.0 { resize_bilinear(w, h, n, img, nw, nh) } else { Vec::new() };
    Ok(match (carved.is_empty(), plain.is_empty()) {
        (false, true) => carved,
        (true, false) => plain,
        _ => carved.iter().zip(&plain).map(|(c, l)| l + (c - l) * amount).collect::<Vec<f32>>(),
    })
}

fn content_aware_scale(s: &mut Session, p: &Value) -> Result<Value> {
    let CasPlan { doc, layer: id, region, dst, amount, protect: protect_mask, skin_tones } = cas_plan(s, p)?;
    let (w, h) = (region.width() as usize, region.height() as usize);
    let (nw, nh) = (dst.width() as usize, dst.height() as usize);
    let label = "Content-Aware Scale";
    // A background job when started with `Session::start` (#210): seam carving runs on a worker
    // against the document snapshot and checks for cancellation before every seam.
    crate::jobs::run(
        s,
        label,
        true,
        move |ctx| {
            ctx.progress(0.0, label);
            let surf = doc.layer(id).and_then(|l| l.surface()).ok_or(EngineError::NoLayer(id))?;
            let img = surf.read_region(region);
            let protect: Option<Vec<f32>> =
                protect_mask.map(|m| (0..w * h).map(|i| m.sample_channel(region.x0 + (i % w) as i32, region.y0 + (i / w) as i32, 0)).collect());
            cas_pixels(ctx, label, &img, (w, h), surf.format(), protect, skin_tones, (nw, nh), amount)
        },
        move |s, out| {
            s.edit(label, |doc, _| {
                let surf = writable_surface(doc, id)?;
                crate::pixels::try_clear_surface(surf, region, None)?;
                surf.write_region(dst, &out);
                surf.prune();
                Ok(())
            })?;
            Ok(json!({"from": [w, h], "to": [nw, nh], "bounds": [dst.x0, dst.y0, nw, nh]}))
        },
    )
}

/// Content-Aware Scale on a reduced copy, for the transform box's live preview (#3096): the
/// command's own carving, on the area scaled down so the longer of its sides before and after
/// is at most `max_side` px. Prepared from the session ([`ContentAwareScalePreview::new`]) and
/// [`run`](ContentAwareScalePreview::run) on any thread.
pub struct ContentAwareScalePreview {
    plan: CasPlan,
    max_side: u32,
}

impl ContentAwareScalePreview {
    /// `p` takes `edit.contentAwareScale`'s params; the same ones fail the same way.
    pub fn new(s: &Session, p: &Value, max_side: u32) -> Result<Self> {
        Ok(Self { plan: cas_plan(s, p)?, max_side: max_side.max(1) })
    }

    /// The area scaled, in document px.
    pub fn region(&self) -> Rect {
        self.plan.region
    }

    /// The reduced sizes of the area before and after: `[w, h, new w, new h]`.
    fn reduced(&self) -> [usize; 4] {
        let (region, dst) = (self.plan.region, self.plan.dst);
        let (w, h, nw, nh) = (region.width(), region.height(), dst.width(), dst.height());
        let k = (f64::from(w.max(h).max(nw).max(nh)) / f64::from(self.max_side)).max(1.0);
        let side = |v: u32| ((f64::from(v) / k).round() as usize).max(1);
        [side(w), side(h), side(nw), side(nh)]
    }

    /// The size of [`run`](Self::run)'s result, px.
    pub fn size(&self) -> [u32; 2] {
        let [_, _, w, h] = self.reduced();
        [w as u32, h as u32]
    }

    /// The scaled pixels at the reduced size ([`size`](Self::size)), in the layer's format, as a
    /// surface with its corner at the origin. Checks `ctx` for cancellation before every seam.
    pub fn run(&self, ctx: &crate::jobs::JobCtx) -> Result<Surface> {
        let CasPlan { doc, layer, region, dst: _, amount, protect, skin_tones } = &self.plan;
        let surf = doc.layer(*layer).and_then(|l| l.surface()).ok_or(EngineError::NoLayer(*layer))?;
        let fmt = surf.format();
        let n = fmt.channels();
        let (w, h) = (region.width(), region.height());
        let [pw, ph, pnw, pnh] = self.reduced();
        // Nearest samples of the area (and of the protecting channel) at the reduced size.
        let src_x: Vec<i32> = (0..pw).map(|i| region.x0 + (((i as f64 + 0.5) * f64::from(w) / pw as f64) as i32).min(w as i32 - 1)).collect();
        let src_y = |j: usize| region.y0 + (((j as f64 + 0.5) * f64::from(h) / ph as f64) as i32).min(h as i32 - 1);
        let mut img = Vec::with_capacity(pw * ph * n);
        for j in 0..ph {
            ctx.check()?;
            let row = surf.read_region(Rect::from_xywh(region.x0, src_y(j), w, 1));
            for &x in &src_x {
                let at = (x - region.x0) as usize * n;
                img.extend_from_slice(row.get(at..at + n).ok_or_else(|| EngineError::Other("preview read out of bounds".into()))?);
            }
        }
        let protect = protect.as_ref().map(|m| (0..pw * ph).map(|i| m.sample_channel(src_x[i % pw], src_y(i / pw), 0)).collect());
        let out = cas_pixels(ctx, "Content-Aware Scale", &img, (pw, ph), fmt, protect, *skin_tones, (pnw, pnh), *amount)?;
        let mut s = Surface::new(fmt);
        s.write_region(Rect::from_xywh(0, 0, pnw as u32, pnh as u32), &out);
        Ok(s)
    }
}

// ------------------------------------------------------------------ Define Brush / Custom Shape

fn define_brush_preset(s: &mut Session, p: &Value) -> Result<Value> {
    let name = match p.get("name").and_then(Value::as_str).map(str::trim).filter(|n| !n.is_empty()) {
        Some(n) => n.to_string(),
        None => {
            let n = s.tools.presets.iter().filter(|b| b.name.starts_with("Sampled Brush")).count() + 1;
            format!("Sampled Brush {n}")
        }
    };
    s.execute("brush.defineFromSelection", json!({"name": name}))
}

/// The path Define Custom Shape uses: `"work"`, a saved path name, else the work path, the
/// active shape layer, the active layer's vector mask, or the last saved path.
pub(crate) fn current_path(doc: &Document, active: Option<LayerId>, spec: Option<&str>) -> Option<Path> {
    let ok = |p: &Path| !p.subpaths.is_empty();
    match spec {
        Some("work") => return doc.work_path.clone().filter(ok),
        Some(name) => return doc.paths.iter().find(|n| n.name == name).map(|n| n.path.clone()).filter(ok),
        None => {}
    }
    if let Some(p) = doc.work_path.clone().filter(ok) {
        return Some(p);
    }
    if let Some(l) = active.and_then(|id| doc.layer(id)) {
        if let LayerContent::Shape(sh) = &l.content
            && ok(&sh.path)
        {
            return Some(sh.path.clone());
        }
        if let Some(vm) = l.vector_mask.as_ref().filter(|v| ok(&v.path)) {
            return Some(vm.path.clone());
        }
    }
    doc.paths.last().map(|n| n.path.clone()).filter(ok)
}

fn can_define_shape(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    current_path(&d.doc, d.active_layer, None).map(|_| ()).ok_or_else(|| "select a path or shape layer first".into())
}

fn define_custom_shape(s: &mut Session, p: &Value) -> Result<Value> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let path = current_path(&st.doc, st.active_layer, p.get("path").and_then(Value::as_str)).ok_or_else(|| bad("edit.defineCustomShape", "no such path"))?;
    let name = p
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| format!("Shape {}", s.edit_state.custom_shapes.len() + 1));
    let shapes = &mut s.edit_state.custom_shapes;
    match shapes.iter_mut().find(|c| c.name == name) {
        Some(c) => c.path = path,
        None => shapes.push(CustomShape { name: name.clone(), path }),
    }
    Ok(json!({"name": name, "count": shapes.len()}))
}

// ------------------------------------------------------------------ Find and Replace Text

/// Byte ranges of `find` in `text` (case-insensitive unless `case`; whole words only with `whole`).
pub fn find_matches(text: &str, find: &str, case: bool, whole: bool) -> Vec<(usize, usize)> {
    if find.is_empty() {
        return Vec::new();
    }
    let norm = |c: char| -> String { if case { c.to_string() } else { c.to_lowercase().collect() } };
    let needle: String = find.chars().map(norm).collect();
    let mut out = Vec::new();
    let mut at = 0;
    while at < text.len() {
        // Compare char by char so case folding that changes byte lengths stays aligned.
        let mut hay = String::new();
        let mut end = at;
        for (i, c) in text[at..].char_indices() {
            if hay.len() >= needle.len() {
                break;
            }
            hay.push_str(&norm(c));
            end = at + i + c.len_utf8();
        }
        if hay == needle {
            let word = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '_');
            let before = text[..at].chars().next_back();
            let after = text[end..].chars().next();
            if !whole || (!word(before) && !word(after)) {
                out.push((at, end));
                at = end;
                continue;
            }
        }
        at += text[at..].chars().next().map_or(1, char::len_utf8);
    }
    out
}

fn type_layers(doc: &Document, forward: bool) -> Vec<LayerId> {
    // Layers panel order: top first.
    let mut ids: Vec<LayerId> =
        doc.walk().into_iter().rev().filter(|(p, _, l)| matches!(l.content, LayerContent::Text(_)) && !doc.locks_at(p).all).map(|(_, _, l)| l.id).collect();
    if !forward {
        ids.reverse();
    }
    ids
}

fn has_type_layer(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    if type_layers(&d.doc, true).is_empty() { Err("the document has no type layers".into()) } else { Ok(()) }
}

fn find_replace(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "edit.findAndReplaceText";
    let find = p.get("find").and_then(Value::as_str).filter(|f| !f.is_empty()).ok_or_else(|| bad(cmd, "missing `find`"))?.to_string();
    let replace = str_or(p, "replace", "").to_string();
    let (case, whole, forward, all_layers) =
        (bool_or(p, "caseSensitive", false), bool_or(p, "wholeWord", false), bool_or(p, "forward", true), bool_or(p, "allLayers", true));
    let action = str_or(p, "action", "changeAll").to_string();
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let mut layers = type_layers(&st.doc, forward);
    if !all_layers {
        let active = layer_param(s, p).ok();
        layers.retain(|id| Some(*id) == active);
    }
    let text_of = |doc: &Document, id: LayerId| -> Option<String> {
        match &doc.layer(id)?.content {
            LayerContent::Text(t) => Some(t.text.clone()),
            _ => None,
        }
    };
    match action.as_str() {
        "changeAll" => {
            let doc = st.doc.clone();
            let plan: Vec<(LayerId, Vec<(usize, usize)>)> =
                layers.iter().filter_map(|id| Some((*id, find_matches(&text_of(&doc, *id)?, &find, case, whole)))).filter(|(_, m)| !m.is_empty()).collect();
            let count: usize = plan.iter().map(|(_, m)| m.len()).sum();
            if count == 0 {
                return Ok(json!({"count": 0, "layers": 0}));
            }
            s.edit("Replace All Text", |doc, _| {
                let snapshot = doc.clone();
                for (id, matches) in &plan {
                    if let Some(LayerContent::Text(t)) = doc.layer_mut(*id).map(|l| &mut l.content) {
                        for (a, b) in matches.iter().rev() {
                            crate::type_cmds::replace_text(t, *a, *b, &replace);
                        }
                        crate::type_cmds::refresh(&snapshot, t);
                    }
                }
                Ok(())
            })?;
            s.edit_state.find_cursor = None;
            Ok(json!({"count": count, "layers": plan.len()}))
        }
        "find" | "change" | "changeFind" => {
            let doc = st.doc.clone();
            let cursor = s.edit_state.find_cursor;
            let mut changed = None;
            if action != "find"
                && let Some((id, a, b)) = cursor
                && text_of(&doc, id).is_some_and(|t| find_matches(&t, &find, case, whole).contains(&(a, b)))
            {
                crate::type_cmds::check_text_unlocked(&doc, id)?;
                s.edit("Replace Text", |doc, _| {
                    let snapshot = doc.clone();
                    if let Some(LayerContent::Text(t)) = doc.layer_mut(id).map(|l| &mut l.content) {
                        crate::type_cmds::replace_text(t, a, b, &replace);
                        crate::type_cmds::refresh(&snapshot, t);
                    }
                    Ok(())
                })?;
                let end = a + replace.len();
                // Collapse the cursor on the side the search moves away from, so the next search
                // skips the replaced text even when the replacement still matches.
                let c = if forward { end } else { a };
                s.edit_state.find_cursor = Some((id, c, c));
                changed = Some((id, a, end));
                if action == "change" {
                    return Ok(json!({"changed": {"layer": id.0, "start": a, "end": end}}));
                }
            }
            // Find the next match after the cursor (wrapping through the layers).
            let doc = s.active().ok_or(EngineError::NoDocument)?.doc.clone();
            let cursor = s.edit_state.find_cursor;
            let start = cursor.and_then(|(id, _, _)| layers.iter().position(|l| *l == id)).unwrap_or(0);
            for k in 0..=layers.len() {
                // `allLayers: false` with a non-type layer active leaves no layers to search (#703).
                let Some(&id) = layers.get((start + k) % layers.len().max(1)) else { break };
                let Some(text) = text_of(&doc, id) else { continue };
                let ms = find_matches(&text, &find, case, whole);
                let hit = match (k, cursor) {
                    (0, Some((cid, a, b))) if cid == id => {
                        if forward {
                            ms.into_iter().find(|m| m.0 >= b)
                        } else {
                            ms.into_iter().rev().find(|m| m.1 <= a)
                        }
                    }
                    _ => {
                        if forward {
                            ms.into_iter().next()
                        } else {
                            ms.into_iter().next_back()
                        }
                    }
                };
                if let Some((a, b)) = hit {
                    s.edit_state.find_cursor = Some((id, a, b));
                    let _ = s.select_layer(id);
                    return Ok(
                        json!({"found": {"layer": id.0, "start": a, "end": b, "text": &text[a..b]}, "changed": changed.map(|(l, a, b)| json!({"layer": l.0, "start": a, "end": b}))}),
                    );
                }
            }
            s.edit_state.find_cursor = None;
            Ok(json!({"found": null, "changed": changed.map(|(l, a, b)| json!({"layer": l.0, "start": a, "end": b}))}))
        }
        other => Err(bad(cmd, format!("unknown action `{other}` (find|change|changeFind|changeAll)"))),
    }
}

// ------------------------------------------------------------------ presets

/// Libraries the Preset Manager edits (patterns export as `.pat` with `pattern.export`).
const PRESET_KINDS: [&str; 3] = ["brushes", "customShapes", "patterns"];

fn preset_names(s: &Session, kind: &str) -> Option<Vec<String>> {
    Some(match kind {
        "brushes" => s.tools.presets.iter().map(|b| b.name.clone()).collect(),
        "customShapes" => s.edit_state.custom_shapes.iter().map(|c| c.name.clone()).collect(),
        "patterns" => s.patterns.items.iter().map(|p| p.name.clone()).collect(),
        _ => return None,
    })
}

fn preset_index(s: &Session, kind: &str, p: &Value) -> Result<usize> {
    let names = preset_names(s, kind).ok_or_else(|| bad("edit.presets.presetManager", format!("unknown preset kind `{kind}` ({})", PRESET_KINDS.join("|"))))?;
    if let Some(i) = p.get("index").and_then(Value::as_u64) {
        return if (i as usize) < names.len() { Ok(i as usize) } else { Err(bad("edit.presets.presetManager", format!("no preset {i}"))) };
    }
    let name = p.get("name").and_then(Value::as_str).ok_or_else(|| bad("edit.presets.presetManager", "give `index` or `name`"))?;
    names.iter().position(|n| n == name).ok_or_else(|| bad("edit.presets.presetManager", format!("no preset named `{name}`")))
}

fn preset_manager(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "edit.presets.presetManager";
    let action = str_or(p, "action", "list");
    if action == "list" {
        let kinds: Vec<&str> = match p.get("kind").and_then(Value::as_str) {
            Some(k) => vec![k],
            None => PRESET_KINDS.to_vec(),
        };
        let mut out = serde_json::Map::new();
        for k in kinds {
            out.insert(k.to_string(), json!(preset_names(s, k).ok_or_else(|| bad(cmd, format!("unknown preset kind `{k}`")))?));
        }
        return Ok(Value::Object(out));
    }
    let kind = p.get("kind").and_then(Value::as_str).ok_or_else(|| bad(cmd, "missing `kind`"))?.to_string();
    let i = preset_index(s, &kind, p)?;
    match action {
        "rename" => {
            let new =
                p.get("newName").and_then(Value::as_str).map(str::trim).filter(|n| !n.is_empty()).ok_or_else(|| bad(cmd, "missing `newName`"))?.to_string();
            match kind.as_str() {
                "brushes" => {
                    // A renamed built-in is the user's preset now (and persists as one).
                    let b = &mut s.tools.presets[i];
                    b.name = new;
                    b.builtin = false;
                }
                "patterns" => s.patterns.items[i].name = new,
                _ => s.edit_state.custom_shapes[i].name = new,
            }
        }
        "delete" => match kind.as_str() {
            "brushes" => drop(s.tools.presets.remove(i)),
            "patterns" => drop(s.patterns.items.remove(i)),
            _ => drop(s.edit_state.custom_shapes.remove(i)),
        },
        "move" => {
            let to = p.get("to").and_then(Value::as_u64).ok_or_else(|| bad(cmd, "missing `to`"))? as usize;
            match kind.as_str() {
                "brushes" => crate::move_item(&mut s.tools.presets, i, to),
                "patterns" => crate::move_item(&mut s.patterns.items, i, to),
                _ => crate::move_item(&mut s.edit_state.custom_shapes, i, to),
            };
        }
        other => return Err(bad(cmd, format!("unknown action `{other}` (list|rename|delete|move)"))),
    }
    if kind == "brushes" {
        s.brush_presets_changed();
    }
    Ok(json!({kind.clone(): preset_names(s, &kind)}))
}

/// Exported preset file (`.pcpresets`, JSON).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct PresetFile {
    pub format: String,
    pub version: u32,
    pub brushes: Vec<photocraft_paint::BrushPreset>,
    pub custom_shapes: Vec<CustomShape>,
}

pub const PRESET_FORMAT: &str = "photocraft-presets";

fn export_import(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "edit.presets.exportImportPresets";
    let kinds: Vec<String> = p
        .get("kinds")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
        .unwrap_or_else(|| ["brushes", "customShapes"].iter().map(|k| k.to_string()).collect());
    let want = |k: &str| kinds.iter().any(|x| x == k);
    match str_or(p, "action", "export") {
        "export" => {
            let file = PresetFile {
                format: PRESET_FORMAT.into(),
                version: 1,
                brushes: if want("brushes") {
                    let mut out: Vec<photocraft_paint::BrushPreset> =
                        s.tools.presets.iter().filter(|b| !b.builtin || bool_or(p, "includeBuiltins", false)).cloned().collect();
                    // The file embeds every bitmap: load the tips the preset store keeps (#1843).
                    for b in &mut out {
                        s.load_brush_tips(&mut b.brush).map_err(|e| bad(cmd, format!("brush preset `{}`: {e}", b.name)))?;
                    }
                    out
                } else {
                    Vec::new()
                },
                custom_shapes: if want("customShapes") { s.edit_state.custom_shapes.clone() } else { Vec::new() },
            };
            Ok(json!({"data": file, "brushes": file.brushes.len(), "customShapes": file.custom_shapes.len()}))
        }
        "import" => {
            let data = p.get("data").cloned().ok_or_else(|| bad(cmd, "missing `data` (an exported preset file)"))?;
            let data = match data {
                Value::String(text) => serde_json::from_str(&text).map_err(|e| bad(cmd, format!("not a preset file: {e}")))?,
                v => v,
            };
            let file: PresetFile = serde_json::from_value(data).map_err(|e| bad(cmd, format!("not a preset file: {e}")))?;
            if file.format != PRESET_FORMAT {
                return Err(bad(cmd, format!("not a preset file (format `{}`)", file.format)));
            }
            let (mut nb, mut ns) = (0, 0);
            if want("brushes") {
                for mut b in file.brushes {
                    // Nested folders as the commands bound them (a hand-edited file can't build an
                    // absurd tree).
                    b.folder.retain(|f| !f.trim().is_empty());
                    b.folder.truncate(crate::brush_preset_cmds::MAX_FOLDER_DEPTH);
                    match s.tools.presets.iter_mut().find(|x| x.name == b.name) {
                        Some(x) => *x = b,
                        None => s.tools.presets.push(b),
                    }
                    nb += 1;
                }
                s.brush_presets_changed();
            }
            if want("customShapes") {
                for c in file.custom_shapes {
                    match s.edit_state.custom_shapes.iter_mut().find(|x| x.name == c.name) {
                        Some(x) => *x = c,
                        None => s.edit_state.custom_shapes.push(c),
                    }
                    ns += 1;
                }
            }
            Ok(json!({"brushes": nb, "customShapes": ns}))
        }
        other => Err(bad(cmd, format!("unknown action `{other}` (export|import)"))),
    }
}

// ------------------------------------------------------------------ specs

macro_rules! spec {
    ($id:literal, $label:literal, [$($m:literal),*], $sc:expr, $params:literal, $en:expr, $run:expr) => {
        CommandSpec { id: $id, label: $label, menu: &[$($m),*], shortcut: $sc, params: $params, enabled: $en, run: $run, journal: true }
    };
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        spec!(
            "edit.fade",
            "Fade…",
            ["Edit"],
            Some("Cmd+Shift+F"),
            r##"{"opacity":0..100=100,"mode":"normal|multiply|screen|overlay|softLight|hardLight|darken|lighten|difference|color|luminosity"}"##,
            can_fade,
            fade
        ),
        spec!("edit.purge.undo", "Undo", ["Edit", "Purge"], None, "{}", can_purge_undo, |s, _| purge(s, "undo")),
        spec!("edit.purge.clipboard", "Clipboard", ["Edit", "Purge"], None, "{}", can_purge_clipboard, |s, _| purge(s, "clipboard")),
        spec!("edit.purge.histories", "Histories", ["Edit", "Purge"], None, "{}", can_purge_histories, |s, _| purge(s, "histories")),
        spec!("edit.purge.videoCache", "Video Cache", ["Edit", "Purge"], None, "{}", always, |_, _| Ok(
            json!({"purged": [], "bytes": 0, "message": "nothing to purge (no video layers)"})
        )),
        spec!("edit.purge.all", "All", ["Edit", "Purge"], None, "{}", can_purge_all, |s, _| purge(s, "all")),
        spec!(
            "edit.contentAwareFill",
            "Content-Aware Fill…",
            ["Edit"],
            None,
            r##"{"sampling":"auto|rectangular|custom","margin":px?,"area":[x,y,w,h]?,"channel":index|name?,"colorAdaptation":"default|none|high|veryHigh","rotationAdaptation":"none|low|medium|high|full","scale":bool=false,"mirror":bool=false,"sampleAllLayers":bool=false,"output":"current|new|duplicate","seed":u64=1}"##,
            can_caf,
            content_aware_fill
        ),
        spec!("edit.deleteAndFillSelection", "Delete and Fill Selection", [], None, "{}", can_caf, delete_and_fill),
        spec!(
            "edit.contentAwareScale",
            "Content-Aware Scale",
            ["Edit"],
            Some("Cmd+Alt+Shift+C"),
            r##"{"width":px?,"height":px?,"scaleX":1..400=100,"scaleY":1..400=100,"x":px?,"y":px?,"amount":0..100=100,"protect":"none"|channel index|name,"protectSkinTones":bool=false}"##,
            |s| pixel_layer(s).map(|_| ()),
            content_aware_scale
        ),
        spec!(
            "edit.defineBrushPreset",
            "Define Brush Preset…",
            ["Edit"],
            None,
            r##"{"name":text}"##,
            |s| if s.is_enabled("brush.defineFromSelection") { Ok(()) } else { Err("select pixels to define a brush from".into()) },
            define_brush_preset
        ),
        spec!(
            "edit.defineCustomShape",
            "Define Custom Shape…",
            ["Edit"],
            None,
            r##"{"name":text,"path":"work|<saved path name>"?}"##,
            can_define_shape,
            define_custom_shape
        ),
        spec!(
            "edit.findAndReplaceText",
            "Find and Replace Text…",
            ["Edit"],
            None,
            r##"{"find":text,"replace":text,"action":"changeAll|find|change|changeFind","caseSensitive":bool=false,"wholeWord":bool=false,"forward":bool=true,"allLayers":bool=true}"##,
            has_type_layer,
            find_replace
        ),
        spec!(
            "edit.presets.presetManager",
            "Preset Manager…",
            ["Edit", "Presets"],
            None,
            r##"{"action":"list|rename|delete|move","kind":"brushes|customShapes|patterns","index":n?,"name":str?,"newName":str?,"to":n?}"##,
            always,
            preset_manager
        ),
        spec!(
            "edit.presets.exportImportPresets",
            "Export/Import Presets…",
            ["Edit", "Presets"],
            None,
            r##"{"action":"export|import","kinds":["brushes","customShapes"]?,"data":json (import),"includeBuiltins":bool=false}"##,
            always,
            export_import
        ),
    ]
}

/// The custom shape library (for the Custom Shape tool).
pub fn custom_shapes(s: &Session) -> &[CustomShape] {
    &s.edit_state.custom_shapes
}

#[cfg(test)]
#[path = "edit_menu_cmds/tests.rs"]
mod tests;
