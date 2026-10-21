//! The Eraser group's colour-aware tools: Magic Eraser (`paint.magicEraser`, a wand-style region
//! erased to transparency) and Background Eraser (`paint.backgroundEraser`, a brush that erases
//! colours similar to the one under its hotspot).
//!
//! Like Photoshop, both turn a Background layer into a normal layer ("Layer 0") first, since the
//! Background can't hold transparency. The Magic Eraser paints the background colour on other
//! layers with locked transparency; the Background Eraser refuses them. Both respect the selection,
//! work in the layer's own colour model and depth, and are one undo step per click / stroke.

use photocraft_algo::selection as sel;
use photocraft_doc::{Document, Layer, LayerContent, LayerId};
use photocraft_geom::Rect;
use photocraft_paint::Stroke;
use photocraft_paint::bg_erase::{BgEraseSettings, apply_background_eraser};
use photocraft_paint::replace::{Limits, Sampling};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn flag(p: &Value, k: &str, d: bool) -> bool {
    p.get(k).and_then(Value::as_bool).unwrap_or(d)
}

/// A finite number param clamped to `range`, `default` when absent; non-numbers are an error.
/// A numeric param clamped to `range`; missing or null gives `default`, anything else not a
/// finite number is BadParams. Shared with the Paint Bucket and Gradient opacity.
pub(crate) fn number(p: &Value, cmd: &str, k: &str, default: f64, range: std::ops::RangeInclusive<f64>) -> Result<f64> {
    match p.get(k) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => {
            v.as_f64().filter(|f| f.is_finite()).map(|f| f.clamp(*range.start(), *range.end())).ok_or_else(|| bad(cmd, format!("`{k}` must be a number")))
        }
    }
}

fn enum_param<T: serde::de::DeserializeOwned>(p: &Value, cmd: &str, k: &str, default: T, allowed: &str) -> Result<T> {
    match p.get(k) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => serde_json::from_value(v.clone()).map_err(|_| bad(cmd, format!("`{k}` must be one of {allowed}"))),
    }
}

/// The active layer must hold pixels (the erasers don't work on type, shapes, smart objects…).
fn has_pixel_layer(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    let l = d.active_layer.and_then(|id| d.doc.layer(id)).ok_or("no active layer")?;
    if matches!(l.content, LayerContent::Raster(_)) {
        Ok(())
    } else {
        Err(format!("active layer is {} {} layer, not a pixel layer", l.content.article(), l.content.kind_name()))
    }
}

fn active_pixel_layer(s: &Session, cmd: &str) -> Result<LayerId> {
    has_pixel_layer(s).map_err(|m| bad(cmd, m))?;
    s.active().and_then(|d| d.active_layer).ok_or_else(|| bad(cmd, "no active layer"))
}

fn pixels_locked(doc: &Document, l: &Layer) -> Option<EngineError> {
    let locks = doc.effective_locks(l.id);
    (locks.pixels || locks.all).then(|| EngineError::Other(format!("Could not complete your request because the layer \"{}\" is locked", l.name)))
}

fn damage(s: &mut Session, dmg: Rect, erased: bool) -> Value {
    if let Some(st) = s.active_mut()
        && !dmg.is_empty()
    {
        st.last_damage = Some(dmg);
    }
    json!({ "erased": erased, "damage": [dmg.x0, dmg.y0, dmg.width(), dmg.height()] })
}

/// `paint.magicEraser`: erase the wand region around `(x, y)`.
fn magic_eraser(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "paint.magicEraser";
    let coord = |k: &str| -> Result<f64> {
        p.get(k).and_then(Value::as_f64).filter(|v| v.is_finite() && v.abs() < 1e9).ok_or_else(|| bad(CMD, format!("missing or invalid `{k}`")))
    };
    let (x, y) = (coord("x")?.floor() as i32, coord("y")?.floor() as i32);
    let tolerance = number(p, CMD, "tolerance", 32.0, 0.0..=255.0)? as f32;
    let opacity = number(p, CMD, "opacity", 100.0, 1.0..=100.0)? as f32 / 100.0;
    let (anti_alias, contiguous, sample_all) = (flag(p, "antiAlias", true), flag(p, "contiguous", true), flag(p, "sampleAllLayers", false));
    let id = active_pixel_layer(s, CMD)?;
    {
        let d = s.active().ok_or(EngineError::NoDocument)?;
        if let Some(e) = d.doc.layer(id).and_then(|l| pixels_locked(&d.doc, l)) {
            return Err(e);
        }
        // Clicking off the canvas erases nothing (and records no history step).
        if !d.doc.bounds().contains(x, y) {
            return Ok(damage(s, Rect::EMPTY, false));
        }
    }
    let (area, img) = crate::selection_cmds::sample_rgba8(s, sample_all)?;
    let region = sel::wand_region(&img, area, (x, y), tolerance, contiguous, anti_alias);
    drop(img);
    let Some(region) = region else { return Ok(damage(s, Rect::EMPTY, false)) };
    let bg = s.tools.background;
    let dmg = s.edit("Magic Eraser", |doc, _| {
        let selection = doc.selection.clone();
        // Photoshop converts the Background to a normal layer before erasing to transparency.
        doc.layer(id).ok_or(EngineError::NoLayer(id))?;
        crate::extra_cmds::background_to_layer(doc, id);
        let lock = doc.effective_locks(id).transparency;
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        let surf = l.surface_mut().ok_or_else(|| bad(CMD, "not a pixel layer"))?;
        let fmt = surf.format();
        // Locked transparency: paint the background colour instead (Photoshop).
        let lock_color = lock.then(|| photocraft_raster::from_rgba(&fmt, [bg[0], bg[1], bg[2], 1.0]));
        Ok(photocraft_algo::erase::magic_erase(surf, &region, opacity, selection.as_ref(), lock_color.as_deref()))
    })?;
    Ok(damage(s, dmg, true))
}

/// `paint.backgroundEraser`: a stroke erasing colours similar to the sampled one.
fn background_eraser(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "paint.backgroundEraser";
    let pts = crate::brush_cmds::parse_points(p, CMD)?;
    if pts.iter().any(|q| !q.x.is_finite() || !q.y.is_finite() || q.x.abs() > 1e7 || q.y.abs() > 1e7) {
        return Err(bad(CMD, "`points` must be finite canvas coordinates"));
    }
    let brush = crate::brush_cmds::resolve_brush(s, p, CMD)?;
    let bs = BgEraseSettings {
        sampling: enum_param(p, CMD, "sampling", Sampling::Continuous, "continuous|once|backgroundSwatch")?,
        limits: enum_param(p, CMD, "limits", Limits::Contiguous, "discontiguous|contiguous|findEdges")?,
        tolerance: number(p, CMD, "tolerance", 50.0, 0.0..=100.0)? as f32 / 100.0,
        protect_foreground: flag(p, "protectForegroundColor", false),
        foreground: s.tools.foreground,
        background: s.tools.background,
    };
    let id = active_pixel_layer(s, CMD)?;
    {
        let d = s.active().ok_or(EngineError::NoDocument)?;
        let l = d.doc.layer(id).ok_or(EngineError::NoLayer(id))?;
        if let Some(e) = pixels_locked(&d.doc, l) {
            return Err(e);
        }
        if d.doc.effective_locks(id).transparency && !d.doc.is_background(id) {
            return Err(EngineError::Other(format!("Could not use the Background Eraser because the transparency of layer \"{}\" is locked", l.name)));
        }
    }
    let stroke = Stroke { brush, points: pts };
    let dmg = s.edit("Background Eraser", |doc, _| {
        let selection = doc.selection.clone();
        crate::extra_cmds::background_to_layer(doc, id);
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        let surf = l.surface_mut().ok_or_else(|| bad(CMD, "not a pixel layer"))?;
        Ok(apply_background_eraser(surf, &stroke, &bs, selection.as_ref()))
    })?;
    Ok(damage(s, dmg, true))
}

/// Eraser-group command specs.
pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "paint.magicEraser",
            label: "Magic Eraser",
            menu: &[],
            shortcut: None,
            params: r##"{"x":px,"y":px,"tolerance":0..255=32,"antiAlias":bool=true,"contiguous":bool=true,"sampleAllLayers":bool=false,"opacity":1..100=100}"##,
            enabled: has_pixel_layer,
            run: magic_eraser,
            journal: true,
        },
        CommandSpec {
            id: "paint.backgroundEraser",
            label: "Background Eraser",
            menu: &[],
            shortcut: None,
            params: r##"{"points":[[x,y,pressure?],…],"size":px?,"hardness":0..1?,"brush":{…}?,"preset":name?,"sampling":"continuous|once|backgroundSwatch"="continuous","limits":"discontiguous|contiguous|findEdges"="contiguous","tolerance":0..100=50,"protectForegroundColor":bool=false,"seed":u64?}"##,
            enabled: has_pixel_layer,
            run: background_eraser,
            journal: true,
        },
    ]
}

#[cfg(test)]
mod tests;
