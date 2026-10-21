//! Image › Adjustments beyond the per-pixel adjustment-layer kinds:
//!
//! * parameter parsing for Selective Color and Color Lookup (both also adjustment layers; their
//!   commands are generated with the other kinds in `commands.rs`),
//! * the destructive-only adjustments: Shadows/Highlights, Replace Color, Match Color and HDR
//!   Toning (maths in `photocraft_algo::tone`).
//!
//! Destructive adjustments convert the target (layer pixels, or a targeted alpha channel /
//! Quick Mask) to straight RGBA, so every colour model and bit depth is handled, and blend the
//! result through the selection like the other Image › Adjustments commands.

use photocraft_algo::tone::{self, HdrToning, MatchColor, ShadowsHighlights};
use photocraft_doc::{Adjustment, Document, Layer, LayerContent, Rect};
use photocraft_raster::{Surface, to_rgba};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

/// Built-in Color Lookup looks `(id, label)`.
pub use photocraft_cms::lutfile::BUILTIN as LOOKS;

/// Selective Color range keys in storage order.
pub const RANGES: [&str; 9] = ["reds", "yellows", "greens", "cyans", "blues", "magentas", "whites", "neutrals", "blacks"];

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn num(p: &Value, key: &str, default: f32) -> f32 {
    p.get(key).and_then(Value::as_f64).map_or(default, |v| v as f32)
}

/// `key`'s value; a null counts as missing (as in `adjust_params`).
fn given<'a>(p: &'a Value, key: &str) -> Option<&'a Value> {
    p.get(key).filter(|v| !v.is_null())
}

fn opt_str<'a>(cmd: &str, p: &'a Value, key: &str) -> Result<Option<&'a str>> {
    match given(p, key) {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => Err(bad(cmd, format!("`{key}` must be a string"))),
    }
}

fn opt_bool(cmd: &str, p: &Value, key: &str) -> Result<Option<bool>> {
    match given(p, key) {
        None => Ok(None),
        Some(Value::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(bad(cmd, format!("`{key}` must be true or false"))),
    }
}

/// A percentage in -100..100 (clamped); anything but a finite number is an error.
fn percent(cmd: &str, key: &str, v: &Value) -> Result<f32> {
    let x = v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| bad(cmd, format!("`{key}` must be a number")))?;
    Ok((x as f32).clamp(-100.0, 100.0))
}

/// Selective Color from params. Accepts per-range arrays (`"reds":[c,m,y,k]` in percent) and the
/// dialog's single-range form (`"colors":"reds","cyan":…,"magenta":…,"yellow":…,"black":…`);
/// `"method":"relative|absolute"`. Unspecified ranges keep `base`'s values; a key of the wrong
/// type is an error.
pub fn selective_from_params(p: &Value, base: Option<&Adjustment>) -> Result<Adjustment> {
    const CMD: &str = "selectiveColor";
    let (mut relative, mut adj) = match base {
        Some(Adjustment::SelectiveColor { relative, adjustments }) => (*relative, *adjustments),
        _ => (true, [[0.0; 4]; 9]),
    };
    match opt_str(CMD, p, "method")? {
        None => {}
        Some("relative") => relative = true,
        Some("absolute") => relative = false,
        Some(m) => return Err(bad(CMD, format!("`method` must be \"relative\" or \"absolute\", not `{m}`"))),
    }
    if let Some(b) = opt_bool(CMD, p, "relative")? {
        relative = b;
    }
    for (row, key) in adj.iter_mut().zip(RANGES) {
        let Some(v) = given(p, key) else { continue };
        let a = v.as_array().filter(|a| a.len() == 4).ok_or_else(|| bad(CMD, format!("`{key}` must be an array of 4 numbers [c, m, y, k]")))?;
        for (slot, x) in row.iter_mut().zip(a) {
            *slot = percent(CMD, key, x)?;
        }
    }
    if let Some(c) = opt_str(CMD, p, "colors")? {
        let Some((row, _)) = adj.iter_mut().zip(RANGES).find(|(_, r)| *r == c) else {
            return Err(bad(CMD, format!("unknown range `{c}` (ranges: {})", RANGES.join(", "))));
        };
        for (slot, key) in row.iter_mut().zip(["cyan", "magenta", "yellow", "black"]) {
            if let Some(v) = given(p, key) {
                *slot = percent(CMD, key, v)?;
            }
        }
    }
    Ok(Adjustment::SelectiveColor { relative, adjustments: adj })
}

/// Color Lookup from params: `"lut"` (a built-in look id or `"none"`), `"file"` (a .cube / .3dl /
/// .look path, native only) or `"data"` (the file's text, with `"fileName"` naming its format),
/// `"interpolation":"trilinear|tetrahedral"`, `"dither":bool`. The table is embedded in the
/// layer, as Photoshop does. Unspecified fields keep `base`'s values; a key of the wrong type is an
/// error.
pub fn lookup_from_params(p: &Value, base: Option<&Adjustment>) -> Result<Adjustment> {
    const CMD: &str = "colorLookup";
    let (mut name, mut lut, mut size, mut tetrahedral, mut dither) = match base {
        Some(Adjustment::ColorLookup { name, lut, size, tetrahedral, dither }) => (name.clone(), lut.clone(), *size, *tetrahedral, *dither),
        _ => (String::new(), None, 0, false, false),
    };
    let mut loaded: Option<(photocraft_cms::lutfile::LutFile, String)> = None;
    if let Some(id) = opt_str(CMD, p, "lut")? {
        if id == "none" || id.is_empty() {
            name.clear();
            lut = None;
            size = 0;
        } else {
            let f = photocraft_cms::lutfile::builtin(id).ok_or_else(|| bad(CMD, format!("unknown look `{id}` (built-ins: {})", builtin_ids())))?;
            let label = f.title.clone();
            loaded = Some((f, label));
        }
    }
    let file_name = opt_str(CMD, p, "fileName")?;
    if let Some(text) = opt_str(CMD, p, "data")? {
        let file_name = file_name.unwrap_or("lut.cube");
        let f = photocraft_cms::lutfile::parse(file_name, text.as_bytes()).map_err(|e| bad(CMD, e.0))?;
        loaded = Some((f, base_name(file_name)));
    }
    if let Some(path) = opt_str(CMD, p, "file")?.filter(|s| !s.is_empty()) {
        let bytes = read_file(path).ok_or_else(|| bad(CMD, format!("can't read {path}")))?;
        let f = photocraft_cms::lutfile::parse(path, &bytes).map_err(|e| bad(CMD, format!("{path}: {}", e.0)))?;
        loaded = Some((f, base_name(path)));
    }
    if let Some((f, label)) = loaded {
        if !f.domain_is_default() {
            return Err(bad(
                CMD,
                format!(
                    "`{}` declares DOMAIN_MIN {:?} / DOMAIN_MAX {:?}: Color Lookup applies the plain 0..1 domain, so the look would shift; honouring a custom domain is not supported yet",
                    label, f.domain_min, f.domain_max
                ),
            ));
        }
        name = label;
        size = f.size as u32;
        lut = Some(std::sync::Arc::new(f.data));
    }
    match opt_str(CMD, p, "interpolation")? {
        None => {}
        Some("trilinear") => tetrahedral = false,
        Some("tetrahedral") => tetrahedral = true,
        Some(i) => return Err(bad(CMD, format!("`interpolation` must be \"trilinear\" or \"tetrahedral\", not `{i}`"))),
    }
    if let Some(b) = opt_bool(CMD, p, "tetrahedral")? {
        tetrahedral = b;
    }
    if let Some(b) = opt_bool(CMD, p, "dither")? {
        dither = b;
    }
    Ok(Adjustment::ColorLookup { name, lut, size, tetrahedral, dither })
}

fn builtin_ids() -> String {
    photocraft_cms::lutfile::BUILTIN.iter().map(|b| b.0).collect::<Vec<_>>().join(", ")
}

fn base_name(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or(path).to_string()
}

#[cfg(not(target_arch = "wasm32"))]
fn read_file(path: &str) -> Option<Vec<u8>> {
    std::fs::read(path).ok()
}
#[cfg(target_arch = "wasm32")]
fn read_file(_: &str) -> Option<Vec<u8>> {
    None
}

/// `layer.setAdjustment` for the kinds edited one range / option at a time: params merge into the
/// current values; a call with no parameters resets to defaults.
pub fn update_adjustment(existing: &Adjustment, p: &Value) -> Option<Result<Adjustment>> {
    let empty = p.as_object().is_none_or(|o| o.keys().all(|k| matches!(k.as_str(), "layer" | "coalesce" | "__kind")));
    match existing {
        Adjustment::SelectiveColor { .. } => Some(selective_from_params(p, (!empty).then_some(existing))),
        Adjustment::ColorLookup { .. } => Some(lookup_from_params(p, (!empty).then_some(existing))),
        _ => None,
    }
}

// ---------- destructive helpers ----------

/// Runs `f` on the target surface's content as straight RGBA (`w × h` = content bounds), blending
/// the result through the selection.
fn process_surface(surf: &mut Surface, selection: Option<&Surface>, f: &mut dyn FnMut(&mut Vec<[f32; 4]>, Rect)) {
    let r = surf.content_bounds();
    if r.is_empty() {
        return;
    }
    let fmt = surf.format();
    let n = fmt.channels();
    let raw = surf.read_region(r);
    let mut px: Vec<[f32; 4]> = raw.chunks_exact(n).map(|q| to_rgba(&fmt, q)).collect();
    let orig = px.clone();
    f(&mut px, r);
    surf.write_region(r, &crate::pixels::mix_through_selection(&fmt, r, &orig, &px, selection));
}

/// One history step running `f` on the active pixel layer (or targeted channel or layer mask).
fn rgba_edit(s: &mut Session, label: &str, p: &Value, mut f: impl FnMut(&mut Vec<[f32; 4]>, Rect)) -> Result<Value> {
    let id = crate::commands::layer_param(s, &Value::Null).ok();
    if crate::channel_cmds::target_of(p) != crate::channel_cmds::Target::Pixels {
        return s.edit(label, |doc, _| {
            let sel = doc.selection.clone();
            if let Some(surf) = crate::channel_cmds::channel_surface_for_filter(doc, id, p)? {
                process_surface(surf, sel.as_ref(), &mut f);
                surf.prune();
            }
            Ok(Value::Null)
        });
    }
    let id = id.ok_or_else(|| EngineError::Other("no active layer".into()))?;
    s.edit(label, |doc, _| {
        let sel = doc.selection.clone();
        let surf = crate::commands::paint_surface(doc, id, &Value::Null)?;
        process_surface(surf, sel.as_ref(), &mut f);
        Ok(Value::Null)
    })
}

fn has_pixels(s: &Session) -> std::result::Result<(), String> {
    if crate::channel_cmds::edits_channel(s) {
        return Ok(());
    }
    let d = s.active().ok_or("no document open")?;
    let l = d.active_layer.and_then(|id| d.doc.layer(id)).ok_or("no active layer")?;
    if matches!(l.content, LayerContent::Raster(_)) {
        Ok(())
    } else {
        Err(format!("active layer is {} {} layer, not a pixel layer", l.content.article(), l.content.kind_name()))
    }
}

/// Image › Adjustments targets: a pixel layer, a smart object (the adjustment is recorded as a
/// smart filter, as in Photoshop), or a targeted alpha channel / Quick Mask.
pub(crate) fn has_adjustable(s: &Session) -> std::result::Result<(), String> {
    if crate::channel_cmds::edits_channel(s) {
        return Ok(());
    }
    let d = s.active().ok_or("no document open")?;
    let l = d.active_layer.and_then(|id| d.doc.layer(id)).ok_or("no active layer")?;
    match &l.content {
        LayerContent::Raster(_) => Ok(()),
        LayerContent::Smart(sm) if sm.cache.is_some() => Ok(()),
        other => Err(format!("active layer is {} {} layer, not a pixel layer", other.article(), other.kind_name())),
    }
}

/// Image › Adjustments on a smart object record `image.adjustments.<kind>` as a smart filter
/// (Photoshop): the adjustment re-runs from the source whenever the object re-renders, through
/// [`adjust_as_filter`]. `Ok(true)` when it was recorded (one history step, the selection becomes
/// the filter mask); `Ok(false)` when the command should edit pixels instead (a mask or channel
/// target, or the active layer is not a smart object).
pub(crate) fn adjust_as_smart_filter(s: &mut Session, kind: &str, label: &str, p: &Value) -> Result<bool> {
    if crate::commands::is_mask_target(p) || crate::channel_cmds::is_channel_target(p) {
        return Ok(false);
    }
    let Some(id) = s.active().and_then(|d| d.active_layer) else { return Ok(false) };
    let smart = s.active().is_some_and(|d| matches!(d.doc.layer(id).map(|l| &l.content), Some(LayerContent::Smart(_))));
    if !smart {
        return Ok(false);
    }
    // The recorded params are the command's own, minus the engine's private `__kind` key.
    let mut params = p.clone();
    if let Value::Object(m) = &mut params {
        m.remove("__kind");
    }
    let sf = photocraft_doc::SmartFilter {
        command: format!("image.adjustments.{kind}"),
        params,
        blend: photocraft_color::BlendMode::Normal,
        opacity: 1.0,
        visible: true,
    };
    let sel = s.active().and_then(|d| d.doc.selection.clone());
    s.edit(label, |doc, _| crate::smart_cmds::add_smart_filter(doc, id, sf, sel.as_ref()))?;
    Ok(true)
}

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

// ---------- commands ----------

/// Image › Adjustments run as a smart filter (`image.adjustments.<kind>` recorded on a smart
/// object): the adjustment applied to a copy of `surf`. `None` for kinds or params it can't run.
pub(crate) fn adjust_as_filter(kind: &str, p: &Value, surf: &Surface) -> Option<Surface> {
    let mut out = surf.clone();
    if kind == "shadowsHighlights" {
        let sh = shadows_highlights_params(p);
        process_surface(&mut out, None, &mut |px, r| tone::shadows_highlights(px, r.width() as usize, r.height() as usize, &sh));
    } else {
        let mode = surf.format().mode;
        let adj = crate::adjust_params::from_params(kind, p, None, mode).ok()?;
        crate::pixels::adjust_surface(&mut out, &adj, None, mode);
    }
    out.prune();
    Some(out)
}

fn shadows_highlights(s: &mut Session, p: &Value) -> Result<Value> {
    if adjust_as_smart_filter(s, "shadowsHighlights", "Shadows/Highlights", p)? {
        return Ok(Value::Null);
    }
    let sh = shadows_highlights_params(p);
    rgba_edit(s, "Shadows/Highlights", p, |px, r| tone::shadows_highlights(px, r.width() as usize, r.height() as usize, &sh))
}

fn shadows_highlights_params(p: &Value) -> ShadowsHighlights {
    let d = ShadowsHighlights::default();
    ShadowsHighlights {
        shadow_amount: num(p, "shadowAmount", d.shadow_amount).clamp(0.0, 100.0),
        shadow_tone: num(p, "shadowTone", d.shadow_tone).clamp(0.0, 100.0),
        shadow_radius: num(p, "shadowRadius", d.shadow_radius).clamp(0.0, 2500.0),
        highlight_amount: num(p, "highlightAmount", d.highlight_amount).clamp(0.0, 100.0),
        highlight_tone: num(p, "highlightTone", d.highlight_tone).clamp(0.0, 100.0),
        highlight_radius: num(p, "highlightRadius", d.highlight_radius).clamp(0.0, 2500.0),
        color: num(p, "color", d.color).clamp(-100.0, 100.0),
        midtone: num(p, "midtone", d.midtone).clamp(-100.0, 100.0),
        black_clip: num(p, "blackClip", d.black_clip).clamp(0.0, 50.0),
        white_clip: num(p, "whiteClip", d.white_clip).clamp(0.0, 50.0),
    }
}

fn replace_color(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "image.adjustments.replaceColor";
    let c = crate::commands::color_param(p, "color", s.tools.foreground);
    let fuzz = num(p, "fuzziness", 40.0).clamp(0.0, 200.0);
    let (h, sat, l) = (num(p, "hue", 0.0).clamp(-180.0, 180.0), num(p, "saturation", 0.0).clamp(-100.0, 100.0), num(p, "lightness", 0.0).clamp(-100.0, 100.0));
    // Localized Color Clusters, as in Select › Color Range: the colour only matches within Range
    // (a percentage of the canvas's longer side) of where it was picked.
    let points = sample_points(p).map_err(|m| bad(CMD, m))?;
    let localized = if p.get("localized").and_then(Value::as_bool).unwrap_or(false) {
        if points.is_empty() {
            return Err(bad(CMD, "localized clusters need eyedropper `points` (where the colour was picked)"));
        }
        let canvas = s.active().ok_or(EngineError::NoDocument)?.doc.bounds();
        Some(num(p, "range", 100.0).clamp(0.0, 100.0) / 100.0 * canvas.width().max(canvas.height()) as f32)
    } else {
        None
    };
    rgba_edit(s, "Replace Color", p, |px, r| match localized {
        Some(radius) => {
            let samples: Vec<_> = points
                .iter()
                .map(|&(x, y)| photocraft_algo::selection::RangeSample { color: [c[0], c[1], c[2]], at: Some((x - r.x0 as f32, y - r.y0 as f32)) })
                .collect();
            let mask = photocraft_algo::selection::color_range_samples(px, r.width() as usize, &samples, fuzz, Some(radius));
            tone::replace_color_masked(px, &mask, h, sat, l);
        }
        None => {
            tone::replace_color(px, [c[0], c[1], c[2]], fuzz, h, sat, l);
        }
    })
}

/// The `points` of a command (`[[x, y], …]` in document pixels), or a message saying what is wrong.
fn sample_points(p: &Value) -> std::result::Result<Vec<(f32, f32)>, String> {
    let Some(pts) = p.get("points") else { return Ok(Vec::new()) };
    let pts = pts.as_array().ok_or("`points` must be [[x, y], …]")?;
    pts.iter()
        .map(|pt| {
            pt.as_array()
                .and_then(|a| match a.as_slice() {
                    [x, y] => Some((x.as_f64()? as f32, y.as_f64()? as f32)),
                    _ => None,
                })
                .filter(|(x, y)| x.is_finite() && y.is_finite())
                .ok_or_else(|| format!("bad point {pt} in `points` (want [x, y])"))
        })
        .collect()
}

/// Straight RGBA pixels and an optional coverage mask for a document's layer (or its composite).
/// Pixels plus an optional coverage mask.
type Sample = (Vec<[f32; 4]>, Option<Vec<f32>>);

fn stats_pixels(doc: &Document, layer: Option<u64>, use_selection: bool) -> Result<Sample> {
    let area = doc.bounds();
    let px: Vec<[f32; 4]> = match layer {
        Some(id) => {
            let l = doc.layer(photocraft_doc::LayerId(id)).ok_or(EngineError::NoLayer(photocraft_doc::LayerId(id)))?;
            photocraft_compose::render_layer(l, area).px
        }
        None => photocraft_compose::flatten(doc).px,
    };
    let mask = doc.selection.as_ref().filter(|_| use_selection).map(|sel| photocraft_algo::selection::mask_from_surface(Some(sel), area));
    Ok((px, mask))
}

fn match_color(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "image.adjustments.matchColor";
    let o = MatchColor {
        luminance: num(p, "luminance", 100.0).clamp(1.0, 200.0),
        intensity: num(p, "intensity", 100.0).clamp(1.0, 200.0),
        fade: num(p, "fade", 0.0).clamp(0.0, 100.0),
        neutralize: p.get("neutralize").and_then(Value::as_bool).unwrap_or(false),
    };
    let source = match p.get("source").and_then(Value::as_i64) {
        Some(i) if i >= 0 => {
            let d = s.documents().get(i as usize).ok_or_else(|| bad(CMD, format!("no document {i}")))?;
            let layer = p.get("sourceLayer").and_then(Value::as_u64);
            let (px, mask) = stats_pixels(&d.doc, layer, p.get("useSelectionInSource").and_then(Value::as_bool).unwrap_or(false))?;
            Some(tone::lab_stats(&px, mask.as_deref()).ok_or_else(|| bad(CMD, "the source has no opaque pixels"))?)
        }
        _ => None,
    };
    // Source "None" with default options matches the image to its own statistics: a no-op, as in
    // Photoshop (Neutralize / Luminance / Color Intensity still act).
    // Target statistics: the layer itself, restricted to the selection when asked.
    let target_sel = p.get("useSelectionInTarget").and_then(Value::as_bool).unwrap_or(false);
    let sel = s.active().and_then(|d| d.doc.selection.clone());
    rgba_edit(s, "Match Color", p, |px, r| {
        let mask: Option<Vec<f32>> = sel.as_ref().filter(|_| target_sel).map(|sel| photocraft_algo::selection::mask_from_surface(Some(sel), r));
        let Some(target) = tone::lab_stats(px, mask.as_deref()) else {
            return;
        };
        tone::match_color(px, &target, source.as_ref(), &o);
    })
}

fn hdr_toning(s: &mut Session, p: &Value) -> Result<Value> {
    let d = HdrToning::default();
    let curve: Vec<(f32, f32)> = p
        .get("curve")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|v| Some((v.get(0)?.as_f64()? as f32 / 255.0, v.get(1)?.as_f64()? as f32 / 255.0))).collect())
        .unwrap_or_default();
    let h = HdrToning {
        radius: num(p, "radius", d.radius).clamp(1.0, 500.0),
        strength: num(p, "strength", d.strength).clamp(0.1, 4.0),
        gamma: num(p, "gamma", d.gamma).clamp(0.1, 2.0),
        exposure: num(p, "exposure", d.exposure).clamp(-5.0, 5.0),
        detail: num(p, "detail", d.detail).clamp(-100.0, 300.0),
        shadow: num(p, "shadow", d.shadow).clamp(-100.0, 100.0),
        highlight: num(p, "highlight", d.highlight).clamp(-100.0, 100.0),
        vibrance: num(p, "vibrance", d.vibrance).clamp(-100.0, 100.0),
        saturation: num(p, "saturation", d.saturation).clamp(-100.0, 100.0),
        curve,
    };
    // Like Photoshop, HDR Toning flattens the image first (one history step).
    s.edit("HDR Toning", |doc, active| {
        let buf = photocraft_compose::flatten(doc).over_background([1.0, 1.0, 1.0]);
        let (w, hgt) = (buf.rect.width() as usize, buf.rect.height() as usize);
        let mut px = buf.px;
        tone::hdr_toning(&mut px, w, hgt, &h);
        let fmt = doc.pixel_format();
        let data: Vec<f32> = px.iter().flat_map(|q| photocraft_raster::from_rgba(&fmt, *q)).collect();
        let mut bg = Layer::raster("Background", fmt);
        bg.locks.transparency = true;
        bg.locks.position = true;
        crate::pixels_mut(&mut bg)?.write_region(doc.bounds(), &data);
        *active = Some(bg.id);
        doc.layers = vec![bg];
        doc.selection = None;
        Ok(())
    })?;
    Ok(Value::Null)
}

/// The Color Lookup built-ins (id, label) for UIs and agents.
fn list_looks(_: &mut Session, _: &Value) -> Result<Value> {
    Ok(json!(photocraft_cms::lutfile::BUILTIN.iter().map(|(id, label)| json!({"id": id, "label": label})).collect::<Vec<_>>()))
}

macro_rules! spec {
    ($id:literal, $label:literal, [$($m:literal),*], $params:literal, $en:expr, $run:expr) => {
        CommandSpec { id: $id, label: $label, menu: &[$($m),*], shortcut: None, params: $params, enabled: $en, run: $run, journal: true }
    };
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        spec!(
            "image.adjustments.shadowsHighlights",
            "Shadows/Highlights…",
            ["Image", "Adjustments"],
            r##"{"shadowAmount":0..100=35,"shadowTone":0..100=50,"shadowRadius":0..2500=30,"highlightAmount":0..100=0,"highlightTone":0..100=50,"highlightRadius":0..2500=30,"color":-100..100=20,"midtone":-100..100=0,"blackClip":0..50=0.01,"whiteClip":0..50=0.01}"##,
            has_adjustable,
            shadows_highlights
        ),
        spec!(
            "image.adjustments.replaceColor",
            "Replace Color…",
            ["Image", "Adjustments"],
            r##"{"localized":bool=false,"color":color,"fuzziness":0..200=40,"range":0..100=100,"hue":-180..180=0,"saturation":-100..100=0,"lightness":-100..100=0,"points":[[x,y],…]?} (color: "#rrggbb", default the foreground colour; Localized Color Clusters: "points" where the colour was picked, "range" % of the longer side)"##,
            has_pixels,
            replace_color
        ),
        spec!(
            "image.adjustments.matchColor",
            "Match Color…",
            ["Image", "Adjustments"],
            r##"{"source":doc,"sourceLayer":json,"luminance":1..200=100,"intensity":1..200=100,"fade":0..100=0,"neutralize":bool=false,"useSelectionInSource":bool=false,"useSelectionInTarget":bool=false} (source: document index; sourceLayer: layer id, default the merged image)"##,
            has_pixels,
            match_color
        ),
        spec!(
            "image.adjustments.hdrToning",
            "HDR Toning…",
            ["Image", "Adjustments"],
            r##"{"radius":1..500=30,"strength":0.1..4=0.5,"gamma":0.1..2=1,"exposure":-5..5=0,"detail":-100..300=30,"shadow":-100..100=0,"highlight":-100..100=0,"vibrance":-100..100=0,"saturation":-100..100=20,"curve":[[in,out],…] 0..255} (flattens the image)"##,
            has_doc,
            hdr_toning
        ),
        CommandSpec {
            id: "image.adjustments.colorLookup.list",
            label: "List Color Lookup Looks",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: |_| Ok(()),
            run: list_looks,
            journal: false,
        },
    ]
}

#[cfg(test)]
mod tests;
