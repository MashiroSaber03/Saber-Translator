//! Live gradients: the Gradient tool's non-destructive mode (Photoshop 2023+). A drag makes a
//! Gradient Fill layer above the active layer, masked by the selection, whose start and end
//! points, colour stops, opacity stops and midpoints stay editable on canvas afterwards. The
//! classic destructive mode is `paint.gradient`.
//!
//! A live gradient is laid out on the canvas ("Align with layer" off), so its handles stay put
//! when its mask changes. Its start/end points are stored as the fill's angle, scale and centre
//! offset (`photocraft_compose::gradient_fill`), which is what a PSD Gradient Fill holds, and it
//! renders the same pixels as painting the same drag with `paint.gradient`.
//!
//! Every edit is one command, so one history step per gesture: the UI previews a drag with
//! [`apply_set`] and commits it once on release.

use photocraft_color::Color;
use photocraft_compose::gradient_fill as gf;
use photocraft_doc::{Document, Fill, GradientStyle, Layer, LayerContent, LayerId};
use photocraft_geom::Rect;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, blend_from_str, check_pixels_unlocked, layer_param};
use crate::presets::gradients::{GradientPreset, transparency_param};
use crate::{EngineError, Result, Session};

pub const CREATE: &str = "gradient.fill.create";
pub const SET: &str = "gradient.fill.set";
pub const STOP: &str = "gradient.fill.stop";
pub const GET: &str = "gradient.fill.get";

/// Largest coordinate accepted for a start/end point (pixels).
const MAX_COORD: f32 = 1.0e7;
/// Scale range of a gradient fill (a fraction): Photoshop allows 10–150 % (the PSD stores a
/// percentage). A handle drag past it clamps, so the far handle stops where the scale does.
const SCALE_RANGE: (f32, f32) = (0.1, 1.5);

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn num(p: &Value, k: &str) -> Option<f32> {
    p.get(k).and_then(Value::as_f64).map(|v| v as f32).filter(|v| v.is_finite())
}

fn point(cmd: &str, p: &Value, k: &str) -> Result<Option<[f32; 2]>> {
    let Some(v) = p.get(k) else { return Ok(None) };
    let a = v.as_array().filter(|a| a.len() == 2).ok_or_else(|| bad(cmd, format!("`{k}` is [x, y]")))?;
    let c = |i: usize| a.get(i).and_then(Value::as_f64).map(|v| v as f32).filter(|v| v.is_finite() && v.abs() <= MAX_COORD);
    match (c(0), c(1)) {
        (Some(x), Some(y)) => Ok(Some([x, y])),
        _ => Err(bad(cmd, format!("`{k}` needs finite coordinates within ±{MAX_COORD}"))),
    }
}

fn style_param(cmd: &str, p: &Value) -> Result<Option<GradientStyle>> {
    match p.get("style") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => match s.to_ascii_lowercase().as_str() {
            "linear" => Ok(Some(GradientStyle::Linear)),
            "radial" => Ok(Some(GradientStyle::Radial)),
            "angle" => Ok(Some(GradientStyle::Angle)),
            "reflected" => Ok(Some(GradientStyle::Reflected)),
            "diamond" => Ok(Some(GradientStyle::Diamond)),
            o => Err(bad(cmd, format!("unknown style `{o}` (linear|radial|angle|reflected|diamond)"))),
        },
        Some(_) => Err(bad(cmd, "`style` is a string")),
    }
}

pub fn style_name(s: GradientStyle) -> &'static str {
    match s {
        GradientStyle::Linear => "linear",
        GradientStyle::Radial => "radial",
        GradientStyle::Angle => "angle",
        GradientStyle::Reflected => "reflected",
        GradientStyle::Diamond => "diamond",
    }
}

/// A stop colour: `"#rrggbb"`, `"#rrggbbaa"`, `[r, g, b, a?]` (0..1), `"foreground"` or
/// `"background"`.
fn parse_color(v: &Value, fg: [f32; 4], bg: [f32; 4]) -> Option<Color> {
    let c = match v {
        Value::String(s) if s == "foreground" => fg,
        Value::String(s) if s == "background" => bg,
        Value::String(s) => {
            let h = s.trim_start_matches('#');
            let b = |i: usize| h.get(i..i + 2).and_then(|x| u8::from_str_radix(x, 16).ok()).map(|v| f32::from(v) / 255.0);
            match h.len() {
                6 => [b(0)?, b(2)?, b(4)?, 1.0],
                8 => [b(0)?, b(2)?, b(4)?, b(6)?],
                _ => return None,
            }
        }
        Value::Array(a) if (3..=4).contains(&a.len()) => {
            let f = |i: usize| a.get(i).and_then(Value::as_f64).map(|v| v as f32).filter(|v| v.is_finite());
            [f(0)?, f(1)?, f(2)?, if a.len() == 4 { f(3)? } else { 1.0 }]
        }
        _ => return None,
    };
    let c = c.map(|v| v.clamp(0.0, 1.0));
    Some(Color::rgba(c[0], c[1], c[2], c[3]))
}

fn hex(c: &Color) -> String {
    let rgb = c.to_rgb();
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    if c.alpha >= 1.0 {
        format!("#{:02x}{:02x}{:02x}", q(rgb[0]), q(rgb[1]), q(rgb[2]))
    } else {
        format!("#{:02x}{:02x}{:02x}{:02x}", q(rgb[0]), q(rgb[1]), q(rgb[2]), q(c.alpha))
    }
}

/// The gradient a `gradient.fill.create` uses: a preset (`"gradient"`), explicit `"stops"`,
/// else the current gradient (Gradients panel); explicit `"transparency"` stops replace its own.
fn preset_param(s: &Session, p: &Value, cmd: &str) -> Result<GradientPreset> {
    if let Some(name) = p.get("gradient").and_then(Value::as_str) {
        let groups = &s.presets.gradients;
        return groups
            .iter()
            .flat_map(|g| g.items.iter())
            .find(|g| g.name.eq_ignore_ascii_case(name))
            .cloned()
            .ok_or_else(|| bad(cmd, format!("no gradient preset \"{name}\" (see gradient.presets.list)")))?
            .with_transparency(p, cmd);
    }
    if p.get("stops").is_some() {
        return GradientPreset::from_params("Custom", p, cmd);
    }
    s.presets.gradient.clone().with_transparency(p, cmd)
}

/// The fill and frame of gradient fill layer `id`.
fn gradient_of(s: &Session, id: LayerId, cmd: &str) -> Result<(Fill, Rect)> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let l = d.doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    match &l.content {
        LayerContent::Fill(f @ Fill::Gradient { .. }) => Ok((f.clone(), photocraft_compose::fill_frame(l, d.doc.bounds()))),
        _ => Err(bad(cmd, format!("layer \"{}\" is not a gradient fill layer", l.name))),
    }
}

/// Whether the active layer is a gradient fill layer (the commands' `enabled`).
pub fn has_gradient_fill(s: &Session) -> std::result::Result<(), String> {
    let l = crate::active_layer_of(s)?;
    if matches!(l.content, LayerContent::Fill(Fill::Gradient { .. })) { Ok(()) } else { Err("the active layer is not a gradient fill layer".into()) }
}

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

fn clamp_scale(v: f32) -> f32 {
    v.clamp(SCALE_RANGE.0, SCALE_RANGE.1)
}

/// The frame a gradient fill of `layer` (with `align`) is laid out in.
fn frame_with_align(layer: &Layer, f: &Fill, canvas: Rect) -> Rect {
    let mut l = layer.clone();
    l.content = LayerContent::Fill(f.clone());
    photocraft_compose::fill_frame(&l, canvas)
}

/// Applies `gradient.fill.set` params to a gradient fill (`layer` gives its frame; `fg`/`bg`
/// resolve stop colours). Pure, so the canvas previews a handle drag with exactly what the
/// command will commit.
pub fn apply_set(layer: &Layer, f: &Fill, canvas: Rect, p: &Value, fg: [f32; 4], bg: [f32; 4]) -> Result<Fill> {
    const CMD: &str = SET;
    let mut out = f.clone();
    let Fill::Gradient { stops, angle, scale, style, reverse, opacity_stops, midpoints, offset, dither, align } = &mut out else {
        return Err(bad(CMD, "not a gradient fill"));
    };
    let old_frame = frame_with_align(layer, f, canvas);
    let old_handles = gf::handles(*style, *angle, *scale, *offset, old_frame);
    if let Some(v) = p.get("align") {
        *align = v.as_bool().ok_or_else(|| bad(CMD, "`align` is a boolean"))?;
    }
    let new_style = style_param(CMD, p)?;
    let explicit_geometry = ["angle", "scale", "offset"].iter().any(|k| p.get(*k).is_some());
    if let Some(a) = p.get("angle") {
        *angle = a.as_f64().map(|v| v as f32).filter(|v| v.is_finite()).ok_or_else(|| bad(CMD, "`angle` is a number (degrees)"))?.rem_euclid(360.0);
        if *angle > 180.0 {
            *angle -= 360.0;
        }
    }
    if p.get("scale").is_some() {
        let v = num(p, "scale").filter(|v| *v > 0.0).ok_or_else(|| bad(CMD, "`scale` is a positive percentage"))?;
        *scale = clamp_scale(v / 100.0);
    }
    if let Some(o) = p.get("offset") {
        let a = o.as_array().filter(|a| a.len() == 2).ok_or_else(|| bad(CMD, "`offset` is [x%, y%]"))?;
        let c = |i: usize| a.get(i).and_then(Value::as_f64).map(|v| v as f32).filter(|v| v.is_finite() && v.abs() <= 1.0e5);
        let (Some(x), Some(y)) = (c(0), c(1)) else { return Err(bad(CMD, "`offset` needs finite percentages")) };
        *offset = (x / 100.0, y / 100.0);
    }
    if let Some(v) = p.get("reverse") {
        *reverse = v.as_bool().ok_or_else(|| bad(CMD, "`reverse` is a boolean"))?;
    }
    if let Some(v) = p.get("dither") {
        *dither = v.as_bool().ok_or_else(|| bad(CMD, "`dither` is a boolean"))?;
    }
    let from = point(CMD, p, "from")?;
    let to = point(CMD, p, "to")?;
    let frame = {
        let tmp = Fill::Gradient {
            stops: Vec::new(),
            angle: *angle,
            scale: *scale,
            style: *style,
            reverse: *reverse,
            opacity_stops: Vec::new(),
            midpoints: Vec::new(),
            offset: *offset,
            dither: *dither,
            align: *align,
        };
        frame_with_align(layer, &tmp, canvas)
    };
    // A style change alone keeps the handles where they are (they are what the user placed).
    let style_changed = new_style.is_some_and(|ns| ns != *style);
    if let Some(ns) = new_style {
        *style = ns;
    }
    if from.is_some() || to.is_some() || (style_changed && !explicit_geometry) {
        let cur = if from.is_some() || to.is_some() { gf::handles(*style, *angle, *scale, *offset, frame) } else { old_handles };
        let (f0, t0) = (from.unwrap_or(cur.0), to.unwrap_or(cur.1));
        let (a, s, o) = gf::from_handles(*style, f0, t0, frame, *angle);
        (*angle, *scale, *offset) = (a, clamp_scale(s), o);
        // Past the scale range the dragged handle stops; the other one stays where it is (#1243).
        // (Linear centres on the midpoint, so a clamped scale would otherwise move both ends.)
        let anchor = match (from, to) {
            (Some(_), None) => Some((t0, false)),
            (None, Some(_)) => Some((f0, true)),
            _ => None,
        };
        if let Some((pin, is_start)) = anchor
            && *scale != s
        {
            let (f1, t1) = gf::handles(*style, *angle, *scale, *offset, frame);
            let at = if is_start { f1 } else { t1 };
            let (dx, dy) = (pin[0] - at[0], pin[1] - at[1]);
            let (w, h) = (frame.width().max(1) as f32, frame.height().max(1) as f32);
            *offset = (offset.0 + dx / w, offset.1 + dy / h);
        }
    }
    if let Some(v) = p.get("stops") {
        let arr = v.as_array().ok_or_else(|| bad(CMD, "`stops` is [[location, colour], …]"))?;
        let mut new = Vec::with_capacity(arr.len());
        for s in arr {
            let pair = s.as_array().filter(|a| a.len() == 2).ok_or_else(|| bad(CMD, "a stop is [location 0..1, colour]"))?;
            let t = pair.first().and_then(Value::as_f64).map(|v| v as f32).filter(|v| v.is_finite()).ok_or_else(|| bad(CMD, "stop location is a number"))?;
            let c = pair
                .get(1)
                .and_then(|c| parse_color(c, fg, bg))
                .ok_or_else(|| bad(CMD, "stop colour is \"#rrggbb\", [r,g,b,a], \"foreground\" or \"background\""))?;
            new.push((t.clamp(0.0, 1.0), c));
        }
        if new.len() < 2 {
            return Err(bad(CMD, "a gradient needs at least 2 colour stops"));
        }
        if new.len() > 1024 {
            return Err(bad(CMD, "too many colour stops (max 1024)"));
        }
        new.sort_by(|a, b| a.0.total_cmp(&b.0));
        if new.len() != stops.len() {
            midpoints.clear();
        }
        *stops = new;
    }
    if let Some(new) = transparency_param(p, CMD)? {
        *opacity_stops = new;
    }
    if let Some(v) = p.get("midpoints") {
        let arr = v.as_array().ok_or_else(|| bad(CMD, "`midpoints` is [0..1, …] (one per segment)"))?;
        let mids: Option<Vec<f32>> = arr.iter().map(|m| m.as_f64().map(|v| v as f32).filter(|v| v.is_finite()).map(|v| v.clamp(0.05, 0.95))).collect();
        let mids = mids.ok_or_else(|| bad(CMD, "midpoints are numbers"))?;
        if mids.len() > stops.len().saturating_sub(1) {
            return Err(bad(CMD, format!("{} midpoints for {} segments", mids.len(), stops.len().saturating_sub(1))));
        }
        *midpoints = mids;
    }
    Ok(out)
}

/// The Gradient Fill layer `gradient.fill.create` adds to `doc` for `p` (also the canvas's live
/// preview of a Gradient tool drag): laid out on the canvas, so the handles land exactly where
/// the drag was, and masked by the selection (Photoshop keeps the selection, as a mask).
pub fn new_layer(s: &Session, doc: &Document, p: &Value) -> Result<Layer> {
    const CMD: &str = CREATE;
    let from = point(CMD, p, "from")?.ok_or_else(|| bad(CMD, "missing `from` [x, y]"))?;
    let to = point(CMD, p, "to")?.ok_or_else(|| bad(CMD, "missing `to` [x, y]"))?;
    let style = style_param(CMD, p)?.unwrap_or_default();
    let g = preset_param(s, p, CMD)?;
    let (stops, opacity_stops) = g.fill_stops(s.tools.foreground, s.tools.background);
    let reverse = p.get("reverse").and_then(Value::as_bool).unwrap_or(false);
    let dither = p.get("dither").and_then(Value::as_bool).unwrap_or(true);
    let opacity = num(p, "opacity").unwrap_or(100.0).clamp(0.0, 100.0) / 100.0;
    let blend = match p.get("mode").and_then(Value::as_str) {
        None | Some("") => photocraft_color::BlendMode::Normal,
        Some(m) => blend_from_str(m).ok_or_else(|| bad(CMD, format!("unknown blend mode `{m}`")))?,
    };
    let canvas = doc.bounds();
    if canvas.is_empty() {
        return Err(bad(CMD, "the document is empty"));
    }
    let (angle, scale, offset) = gf::from_handles(style, from, to, canvas, 0.0);
    let fill = Fill::Gradient { stops, angle, scale: clamp_scale(scale), style, reverse, opacity_stops, midpoints: Vec::new(), offset, dither, align: false }
        .in_mode(doc.mode);
    let mut l = Layer::new(doc.next_layer_name("Gradient Fill"), LayerContent::Fill(fill));
    l.opacity = opacity;
    l.blend = blend;
    l.mask = crate::commands::selection_mask(doc)?;
    Ok(l)
}

fn create(s: &mut Session, p: &Value) -> Result<Value> {
    let layer = {
        let d = s.active().ok_or(EngineError::NoDocument)?;
        new_layer(s, &d.doc, p)?
    };
    let id = s.edit("New Gradient Fill Layer", |doc, active| {
        let id = doc.insert_above(*active, layer);
        *active = Some(id);
        Ok(id)
    })?;
    Ok(json!({ "layer": id.0 }))
}

fn set(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    let (fill, _) = gradient_of(s, id, SET)?;
    let (fg, bg) = (s.tools.foreground, s.tools.background);
    let (layer, canvas) = {
        let d = s.active().ok_or(EngineError::NoDocument)?;
        (d.doc.layer(id).cloned().ok_or(EngineError::NoLayer(id))?, d.doc.bounds())
    };
    let new = apply_set(&layer, &fill, canvas, p, fg, bg)?;
    if new != fill {
        replace_fill(s, id, "Edit Gradient Fill", new)?;
    }
    describe(s, id)
}

fn replace_fill(s: &mut Session, id: LayerId, label: &str, new: Fill) -> Result<()> {
    {
        let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
        check_pixels_unlocked(doc, id)?;
    }
    s.edit(label, |doc, _| {
        let mode = doc.mode;
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        l.content = LayerContent::Fill(new.in_mode(mode));
        Ok(())
    })
}

/// Sorted colour stops paired with the midpoint of the segment that follows each.
fn paired(stops: &[(f32, Color)], mids: &[f32]) -> Vec<(f32, Color, f32)> {
    let mut v: Vec<(f32, Color, f32)> = stops.iter().enumerate().map(|(i, (t, c))| (*t, *c, mids.get(i).copied().unwrap_or(0.5))).collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    v
}

fn unpair(v: Vec<(f32, Color, f32)>) -> (Vec<(f32, Color)>, Vec<f32>) {
    let mut v = v;
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let n = v.len();
    let mids: Vec<f32> = v.iter().take(n.saturating_sub(1)).map(|s| s.2).collect();
    let mids = if mids.iter().all(|m| (m - 0.5).abs() < 1e-6) { Vec::new() } else { mids };
    (v.into_iter().map(|(t, c, _)| (t, c)).collect(), mids)
}

/// Applies a `gradient.fill.stop` edit to a gradient fill. Pure (see [`apply_set`]).
pub fn apply_stop(f: &Fill, p: &Value, fg: [f32; 4], bg: [f32; 4]) -> Result<Fill> {
    const CMD: &str = STOP;
    let mut out = f.clone();
    let Fill::Gradient { stops, opacity_stops, midpoints, .. } = &mut out else { return Err(bad(CMD, "not a gradient fill")) };
    let action = p.get("action").and_then(Value::as_str).ok_or_else(|| bad(CMD, "missing `action` (add|move|delete|color|opacity|midpoint)"))?;
    let opacity_kind = match p.get("kind").and_then(Value::as_str).unwrap_or("color") {
        "color" => false,
        "opacity" => true,
        o => return Err(bad(CMD, format!("unknown kind `{o}` (color|opacity)"))),
    };
    // The `index` param, checked against the `len` entries it can address (`what`, e.g.
    // "colour stop" or "segment") before any arithmetic on it: an index is untrusted and can be as
    // large as the integer type, so `i + 1` on it would overflow (#958).
    let index = |len: usize, what: &str| -> Result<usize> {
        let i =
            p.get("index").and_then(Value::as_u64).and_then(|v| usize::try_from(v).ok()).ok_or_else(|| bad(CMD, "missing `index` (a stop number from 0)"))?;
        if i >= len {
            return Err(bad(CMD, format!("no {what} {i} (there are {len})")));
        }
        Ok(i)
    };
    let location = || num(p, "location").map(|v| v.clamp(0.0, 1.0)).ok_or_else(|| bad(CMD, "missing `location` (0..1)"));
    let ramp = gf::Ramp::new(f).ok_or_else(|| bad(CMD, "not a gradient fill"))?;
    if opacity_kind {
        // Opacity stops: an empty list means "opaque" (two 100 % stops when first edited).
        let mut os = opacity_stops.clone();
        if os.is_empty() {
            os = vec![(0.0, 1.0), (1.0, 1.0)];
        }
        os.sort_by(|a, b| a.0.total_cmp(&b.0));
        let opacity = || num(p, "opacity").map(|v| (v / 100.0).clamp(0.0, 1.0)).ok_or_else(|| bad(CMD, "missing `opacity` (0..100)"));
        match action {
            "add" => {
                let t = location()?;
                let a = num(p, "opacity").map_or_else(|| ramp.sample(t)[3] / ramp_color_alpha(f, t), |v| (v / 100.0).clamp(0.0, 1.0));
                os.push((t, a.clamp(0.0, 1.0)));
            }
            "move" => {
                let i = index(os.len(), "opacity stop")?;
                let t = location()?;
                os.get_mut(i).ok_or_else(|| bad(CMD, format!("no opacity stop {i}")))?.0 = t;
            }
            "delete" => {
                let i = index(os.len(), "opacity stop")?;
                if os.len() <= 2 {
                    return Err(bad(CMD, "a gradient keeps at least 2 opacity stops"));
                }
                os.remove(i);
            }
            "opacity" => {
                let i = index(os.len(), "opacity stop")?;
                let a = opacity()?;
                os.get_mut(i).ok_or_else(|| bad(CMD, format!("no opacity stop {i}")))?.1 = a;
            }
            o => return Err(bad(CMD, format!("unknown action `{o}` for opacity stops (add|move|delete|opacity)"))),
        }
        os.sort_by(|a, b| a.0.total_cmp(&b.0));
        if os.len() > 1024 {
            return Err(bad(CMD, "too many opacity stops (max 1024)"));
        }
        *opacity_stops = os;
        return Ok(out);
    }
    let mut v = paired(stops, midpoints);
    match action {
        "add" => {
            let t = location()?;
            // A new stop takes the colour the gradient has there (Photoshop), unless given.
            let c = match p.get("color") {
                Some(c) => parse_color(c, fg, bg).ok_or_else(|| bad(CMD, "bad `color`"))?,
                None => {
                    let s = ramp.sample(t);
                    Color::rgba(s[0], s[1], s[2], ramp_color_alpha(f, t))
                }
            };
            // Splitting a segment resets its midpoint.
            if let Some(prev) = v.iter_mut().rev().find(|s| s.0 <= t) {
                prev.2 = 0.5;
            }
            v.push((t, c, 0.5));
            if v.len() > 1024 {
                return Err(bad(CMD, "too many colour stops (max 1024)"));
            }
        }
        "move" => {
            let i = index(v.len(), "colour stop")?;
            let t = location()?;
            v.get_mut(i).ok_or_else(|| bad(CMD, format!("no colour stop {i}")))?.0 = t;
        }
        "delete" => {
            let i = index(v.len(), "colour stop")?;
            if v.len() <= 2 {
                return Err(bad(CMD, "a gradient keeps at least 2 colour stops"));
            }
            v.remove(i);
        }
        "color" => {
            let i = index(v.len(), "colour stop")?;
            let c = p.get("color").and_then(|c| parse_color(c, fg, bg)).ok_or_else(|| bad(CMD, "missing or bad `color`"))?;
            v.get_mut(i).ok_or_else(|| bad(CMD, format!("no colour stop {i}")))?.1 = c;
        }
        "midpoint" => {
            // Segment `i` runs from stop `i` to stop `i + 1`, so there is one fewer than stops.
            let i = index(v.len().saturating_sub(1), "segment")?;
            let m = num(p, "location").ok_or_else(|| bad(CMD, "missing `location` (0..1 of the segment)"))?.clamp(0.05, 0.95);
            v.get_mut(i).ok_or_else(|| bad(CMD, format!("no segment {i}")))?.2 = m;
        }
        o => return Err(bad(CMD, format!("unknown action `{o}` (add|move|delete|color|midpoint)"))),
    }
    let (st, mids) = unpair(v);
    *stops = st;
    *midpoints = mids;
    Ok(out)
}

/// The colour stops' own alpha at `t` (without the opacity stops).
fn ramp_color_alpha(f: &Fill, t: f32) -> f32 {
    let Fill::Gradient { stops, midpoints, .. } = f else { return 1.0 };
    let bare = Fill::Gradient {
        stops: stops.clone(),
        angle: 0.0,
        scale: 1.0,
        style: GradientStyle::Linear,
        reverse: false,
        opacity_stops: Vec::new(),
        midpoints: midpoints.clone(),
        offset: (0.0, 0.0),
        dither: false,
        align: true,
    };
    gf::Ramp::new(&bare).map_or(1.0, |r| r.sample(t)[3]).max(1e-6)
}

fn stop(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    let (fill, _) = gradient_of(s, id, STOP)?;
    let new = apply_stop(&fill, p, s.tools.foreground, s.tools.background)?;
    if new != fill {
        replace_fill(s, id, "Edit Gradient Fill", new)?;
    }
    describe(s, id)
}

/// A gradient fill layer's settings as JSON (`gradient.fill.get`).
pub fn describe_fill(f: &Fill, frame: Rect) -> Value {
    let Fill::Gradient { stops, angle, scale, style, reverse, opacity_stops, midpoints, offset, dither, align } = f else { return Value::Null };
    let (from, to) = gf::handles(*style, *angle, *scale, *offset, frame);
    json!({
        "style": style_name(*style),
        "angle": angle,
        "scale": scale * 100.0,
        "offset": [offset.0 * 100.0, offset.1 * 100.0],
        "reverse": reverse,
        "dither": dither,
        "align": align,
        "from": from,
        "to": to,
        "stops": stops.iter().map(|(t, c)| json!([t, hex(c)])).collect::<Vec<_>>(),
        "transparency": opacity_stops.iter().map(|(t, a)| json!([t, a * 100.0])).collect::<Vec<_>>(),
        "midpoints": midpoints,
    })
}

fn describe(s: &Session, id: LayerId) -> Result<Value> {
    let (f, frame) = gradient_of(s, id, GET)?;
    let mut v = describe_fill(&f, frame);
    v["layer"] = json!(id.0);
    Ok(v)
}

fn get(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    describe(s, id)
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: CREATE,
            label: "Gradient (Live)",
            menu: &[],
            shortcut: None,
            params: r##"{"from":[x,y],"to":[x,y],"style":"linear|radial|angle|reflected|diamond"="linear","gradient":preset name?,"stops":[[t,"#rrggbb"|"foreground"|"background"],…]?,"transparency":[[t,0..100],…]? (default: the current gradient),"reverse":bool=false,"dither":bool=true,"opacity":0..100=100,"mode":"normal|multiply|…"="normal"} → {"layer":id}; a Gradient Fill layer above the active layer, masked by the selection"##,
            enabled: has_doc,
            run: create,
            journal: true,
        },
        CommandSpec {
            id: SET,
            label: "Edit Gradient Fill",
            menu: &[],
            shortcut: None,
            params: r##"{"layer":id?,"from":[x,y]?,"to":[x,y]?,"style":"linear|radial|angle|reflected|diamond"?,"angle":deg?,"scale":%?,"offset":[x%,y%]?,"reverse":bool?,"dither":bool?,"align":bool?,"stops":[[t,"#rrggbb"|"#rrggbbaa"|[r,g,b,a]|"foreground"|"background"],…]?,"transparency":[[t,0..100],…]?,"midpoints":[0.05..0.95 per segment]?} → the gradient (see gradient.fill.get)"##,
            enabled: has_gradient_fill,
            run: set,
            journal: true,
        },
        CommandSpec {
            id: STOP,
            label: "Edit Gradient Stop",
            menu: &[],
            shortcut: None,
            params: r##"{"layer":id?,"action":"add|move|delete|color|opacity|midpoint","kind":"color|opacity"="color","index":stop (midpoint: segment) number,"location":0..1 (midpoint: of the segment),"color":"#rrggbb"|[r,g,b,a]|"foreground"|"background" (add: default the colour there),"opacity":0..100} → the gradient"##,
            enabled: has_gradient_fill,
            run: stop,
            journal: true,
        },
        CommandSpec {
            id: GET,
            label: "Gradient Fill Settings",
            menu: &[],
            shortcut: None,
            params: r##"{"layer":id?} → {"style","angle","scale":%,"offset":[%,%],"reverse","dither","align","from":[x,y],"to":[x,y],"stops":[[t,"#hex"]],"transparency":[[t,%]],"midpoints":[…]}"##,
            enabled: has_gradient_fill,
            run: get,
            journal: false,
        },
    ]
}

#[cfg(test)]
mod tests;
