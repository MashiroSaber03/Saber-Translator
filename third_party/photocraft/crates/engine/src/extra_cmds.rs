//! Everyday Photoshop commands that build on the core ones: Edit › Stroke, the fixed Transform
//! presets (Rotate 180°/90°, Flip), Paste Into, Reselect, Equalize, Reveal All, Layer from
//! Background, Copy/Paste Layer Style, Hide All Effects, layer-mask toggles, Rasterize variants,
//! Delete Hidden / Empty Layers, Ungroup, Hide/Show Layers, Show Only This Layer (⌥-click an eye),
//! Average and Clouds.

use photocraft_algo::selection as sel;
use photocraft_doc::{Document, Layer, LayerContent, LayerId, LayerMask};
use photocraft_geom::Rect;
use photocraft_raster::{from_rgba_into, to_rgba};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, color_param, layer_param};
use crate::{EngineError, Result, Session};

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

fn active_layer(s: &Session) -> std::result::Result<&Layer, String> {
    let d = s.active().ok_or("no document open")?;
    d.active_layer.and_then(|id| d.doc.layer(id)).ok_or_else(|| "no active layer".into())
}

fn has_layer(s: &Session) -> std::result::Result<(), String> {
    active_layer(s).map(|_| ())
}

fn has_pixels(s: &Session) -> std::result::Result<(), String> {
    match active_layer(s)?.content {
        LayerContent::Raster(_) => Ok(()),
        ref c => Err(format!("active layer is {} {} layer, not a pixel layer", c.article(), c.kind_name())),
    }
}

fn has_mask(s: &Session) -> std::result::Result<(), String> {
    active_layer(s)?.mask.as_ref().map(|_| ()).ok_or_else(|| "the active layer has no layer mask".into())
}

fn has_clip_and_selection(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    s.clipboard.as_ref().ok_or("the clipboard is empty")?;
    d.doc.selection.as_ref().map(|_| ()).ok_or_else(|| "Paste Into needs a selection".into())
}

fn has_background(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    d.doc.background().map(|_| ()).ok_or_else(|| "the document has no Background layer".into())
}

/// The layers a layer-style command acts on: as [`crate::layer_multi_cmds::targets`], but with
/// several layers selected only those `fits` (the Background carries no style, Clear needs one).
pub(crate) fn style_targets(s: &Session, p: &Value, fits: impl Fn(&Layer) -> bool) -> Result<Vec<LayerId>> {
    let ids = crate::layer_multi_cmds::targets(s, p)?;
    if ids.len() < 2 {
        return Ok(ids);
    }
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let ids: Vec<LayerId> = ids.into_iter().filter(|id| d.doc.layer(*id).is_some_and(|l| !d.doc.is_background(l.id) && fits(l))).collect();
    if ids.is_empty() {
        return Err(EngineError::Other("none of the selected layers can take this layer style change".into()));
    }
    Ok(ids)
}

/// A key that makes the sub-commands of one command share a single history step.
pub(crate) fn step_key(s: &Session, what: &str) -> String {
    format!("{what}#{}", s.active().map_or(0, |d| d.revision))
}

// ---------- pixel helpers ----------

/// Rewrite the active pixel layer's RGBA over the selection bounds (or the canvas). `f` gets the
/// area, the pixels, and the selection coverage per pixel; results are blended back by coverage,
/// and transparency stays locked when the layer locks it. When `p` targets an alpha channel, the
/// Quick Mask or the layer mask (`Session::execute` fills that in from the Channels panel), that
/// grayscale surface is rewritten instead, as the core filters do (#935).
fn map_pixels(s: &mut Session, label: &str, p: &Value, f: impl FnOnce(Rect, &mut [[f32; 4]], &[f32])) -> Result<Value> {
    let id = layer_param(s, &Value::Null)?;
    let channel = crate::channel_cmds::target_of(p) != crate::channel_cmds::Target::Pixels;
    s.edit(label, |doc, _| {
        let canvas = doc.bounds();
        let selection = doc.selection.clone();
        let area = selection.as_ref().map_or(canvas, |m| m.content_bounds().intersect(&canvas));
        let (surf, lock_alpha) = if channel {
            let Some(surf) = crate::channel_cmds::channel_surface_for_filter(doc, Some(id), p)? else { return Ok(()) };
            (surf, false)
        } else {
            let locks = doc.effective_locks(id);
            let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
            if locks.pixels || locks.all {
                return Err(EngineError::Other(format!("Could not complete your request because the layer \"{}\" is locked", l.name)));
            }
            (l.surface_mut().ok_or_else(|| EngineError::Other("not a pixel layer".into()))?, locks.transparency)
        };
        if area.is_empty() {
            return Ok(());
        }
        let fmt = surf.format();
        let n = fmt.channels();
        let mut raw = surf.read_region(area);
        let orig: Vec<[f32; 4]> = raw.chunks_exact(n).map(|p| to_rgba(&fmt, p)).collect();
        let w = area.width() as usize;
        let cover: Vec<f32> = match &selection {
            Some(m) => (0..orig.len()).map(|i| m.sample_channel(area.x0 + (i % w) as i32, area.y0 + (i / w) as i32, 0)).collect(),
            None => vec![1.0; orig.len()],
        };
        let mut px = orig.clone();
        f(area, &mut px, &cover);
        let mut enc = [0.0f32; 8];
        for (i, out) in raw.chunks_exact_mut(n).enumerate() {
            let k = cover[i];
            if k <= 0.0 {
                continue;
            }
            let (o, p) = (orig[i], px[i]);
            let mut v = [0.0; 4];
            for c in 0..4 {
                v[c] = o[c] + (p[c] - o[c]) * k;
            }
            if lock_alpha {
                v[3] = o[3];
            }
            from_rgba_into(&fmt, v, &mut enc);
            out.copy_from_slice(&enc[..n]);
        }
        surf.write_region(area, &raw);
        surf.prune();
        Ok(())
    })?;
    Ok(Value::Null)
}

/// Edit › Stroke: a band along the selection edge (or the layer's opaque edge without a selection).
fn stroke(s: &mut Session, p: &Value) -> Result<Value> {
    // Whole pixels, rounded up, as in Photoshop (0.5 px strokes 1 px).
    let width = p.get("width").and_then(Value::as_f64).filter(|w| w.is_finite()).unwrap_or(1.0).clamp(1.0, 250.0).ceil() as usize;
    let color = color_param(p, "color", s.tools.foreground);
    // A colour picked as CMYK strokes a CMYK document with the picked inks (#4196).
    let picked = s.tools.picked_cmyk([color[0], color[1], color[2]]);
    let mode = match p.get("mode") {
        None | Some(Value::Null) => photocraft_color::BlendMode::Normal,
        Some(Value::String(m)) => crate::commands::blend_from_str(m)
            .filter(|m| *m != photocraft_color::BlendMode::PassThrough)
            .ok_or_else(|| EngineError::BadParams { cmd: "edit.stroke".into(), msg: format!("unknown blend mode `{m}`") })?,
        Some(v) => return Err(EngineError::BadParams { cmd: "edit.stroke".into(), msg: format!("mode must be a blend mode name, not {v}") }),
    };
    let opacity = match p.get("opacity") {
        None | Some(Value::Null) => 1.0,
        Some(v) => {
            (v.as_f64()
                .filter(|x| x.is_finite())
                .ok_or_else(|| EngineError::BadParams { cmd: "edit.stroke".into(), msg: "opacity must be a finite number from 0 to 100".into() })?
                .clamp(0.0, 100.0)
                / 100.0) as f32
        }
    };
    let preserve = match p.get("preserveTransparency") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(v)) => *v,
        Some(_) => return Err(EngineError::BadParams { cmd: "edit.stroke".into(), msg: "preserveTransparency must be true or false".into() }),
    };
    let location = p.get("location").and_then(Value::as_str).unwrap_or("center").to_string();
    let id = layer_param(s, p)?;
    s.edit("Stroke", |doc, _| {
        let canvas = doc.bounds();
        let (w, h) = (canvas.width() as usize, canvas.height() as usize);
        let edge = match &doc.selection {
            Some(m) => sel::mask_from_surface(Some(m), canvas),
            None => {
                let l = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
                let surf = l.surface().ok_or_else(|| EngineError::Other("not a pixel layer".into()))?;
                let mut px = vec![[0.0f32; 4]; w * h];
                surf.read_rgba_into(canvas, &mut px);
                px.iter().map(|p| p[3]).collect()
            }
        };
        // Past the canvas nothing is selected, so the canvas edge is a selection edge like any
        // other (Select All, then Stroke › Inside, frames the canvas): stroke on a padded copy.
        let pad = width + 1;
        let (pw, ph) = (w + 2 * pad, h + 2 * pad);
        let mut padded = vec![0.0f32; pw * ph];
        for (row, src) in padded.chunks_exact_mut(pw).skip(pad).zip(edge.chunks_exact(w.max(1))) {
            if let Some(dst) = row.get_mut(pad..pad + w) {
                dst.copy_from_slice(src);
            }
        }
        // Whole-pixel bands, fully opaque: Center puts the odd pixel inside (1 px is all inside).
        let (out_px, in_px) = match location.as_str() {
            "inside" => (0, width),
            "outside" => (width, 0),
            _ => (width / 2, width.div_ceil(2)),
        };
        // A hard-edged selection strokes hard, matching Photoshop pixel for pixel on curves:
        // outside, a pixel is stroked when its centre is less than width + 1 from a selected
        // pixel's; inside, less than width + ½ from an unselected pixel's (fitted to Photoshop's
        // strokes of a 14 px circle: 6 px Outside; 3, 5 and 6 px Inside). Both give whole
        // pixels on straight edges. Soft selections keep soft edges.
        let hard = padded.iter().all(|v| *v <= 0.0 || *v >= 1.0);
        let in_r = if hard && in_px > 0 { in_px as f32 - 0.5 } else { in_px as f32 };
        let mut outer = sel::expand(&padded, pw, ph, out_px as f32);
        let mut inner = sel::contract(&padded, pw, ph, in_r);
        if hard {
            outer.iter_mut().for_each(|v| *v = if *v > 0.0 { 1.0 } else { 0.0 });
            inner.iter_mut().for_each(|v| *v = if *v >= 1.0 { 1.0 } else { 0.0 });
        }
        let band: Vec<f32> = outer
            .chunks_exact(pw)
            .zip(inner.chunks_exact(pw))
            .skip(pad)
            .take(h)
            .flat_map(|(o, i)| o.iter().zip(i).skip(pad).take(w).map(|(o, i)| (o - i).clamp(0.0, 1.0)))
            .collect();
        let band = sel::mask_to_surface(&band, canvas);
        let lock = doc.effective_locks(id).transparency;
        let surf = crate::commands::paint_surface(doc, id, &Value::Null)?;
        let area = band.content_bounds().intersect(&canvas);
        if !area.is_empty() {
            crate::fill_cmds::blend_color_mask(surf, area, color, picked, &band, mode, opacity, preserve || lock)?;
            surf.prune();
        }
        Ok(())
    })?;
    Ok(Value::Null)
}

/// Histogram equalization, like Photoshop's: one CDF over the R, G and B values of
/// the selected pixels, applied to every channel.
fn equalize(s: &mut Session, p: &Value) -> Result<Value> {
    map_pixels(s, "Equalize", p, |_, px, cover| {
        let mut hist = [0f64; 256];
        for (p, &k) in px.iter().zip(cover) {
            if k > 0.0 && p[3] > 0.0 {
                for &v in &p[..3] {
                    hist[(v.clamp(0.0, 1.0) * 255.0).round() as usize] += 1.0;
                }
            }
        }
        let total: f64 = hist.iter().sum();
        if total == 0.0 {
            return;
        }
        let first = hist.iter().copied().find(|v| *v > 0.0).unwrap_or(0.0);
        let mut lut = [0f32; 256];
        let mut acc = 0.0;
        for (i, h) in hist.iter().enumerate() {
            acc += h;
            lut[i] = if total > first { ((acc - first) / (total - first)).clamp(0.0, 1.0) as f32 } else { i as f32 / 255.0 };
        }
        // Same 256 bins as the histogram, at every depth (like Photoshop).
        let map = |v: f32| lut[(v.clamp(0.0, 1.0) * 255.0).round() as usize];
        for p in px.iter_mut() {
            for v in &mut p[..3] {
                *v = map(*v);
            }
        }
    })
}

/// Filter › Blur › Average: the selection (or layer) filled with its mean colour.
fn average(s: &mut Session, p: &Value) -> Result<Value> {
    map_pixels(s, "Average", p, |_, px, cover| {
        let (mut sum, mut wsum, mut asum, mut n) = ([0f64; 3], 0f64, 0f64, 0f64);
        for (p, &k) in px.iter().zip(cover) {
            let w = (p[3] * k) as f64;
            for c in 0..3 {
                sum[c] += p[c] as f64 * w;
            }
            wsum += w;
            asum += (p[3] * k) as f64;
            n += k as f64;
        }
        if wsum <= 0.0 {
            return;
        }
        let avg = [(sum[0] / wsum) as f32, (sum[1] / wsum) as f32, (sum[2] / wsum) as f32, (asum / n.max(1e-9)) as f32];
        px.fill(avg);
    })
}

fn hash(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    (h & 0x00ff_ffff) as f32 / 0x00ff_ffff as f32
}

/// Smooth value noise, summed over octaves (fBm), in 0..1.
pub(crate) fn clouds_value(x: f32, y: f32, base: f32, seed: u32) -> f32 {
    let (mut sum, mut amp, mut freq, mut norm) = (0.0, 1.0, 1.0 / base, 0.0);
    for o in 0..8u32 {
        let (fx, fy) = (x * freq, y * freq);
        let (ix, iy) = (fx.floor() as i32, fy.floor() as i32);
        let (tx, ty) = (fx - ix as f32, fy - iy as f32);
        let (sx, sy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
        let s = seed.wrapping_add(o * 7919);
        let top = hash(ix, iy, s) + (hash(ix + 1, iy, s) - hash(ix, iy, s)) * sx;
        let bot = hash(ix, iy + 1, s) + (hash(ix + 1, iy + 1, s) - hash(ix, iy + 1, s)) * sx;
        sum += (top + (bot - top) * sy) * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    // Stretch the bell-shaped sum so the clouds use the full foreground→background range.
    ((sum / norm - 0.5) * 2.2 + 0.5).clamp(0.0, 1.0)
}

/// Filter › Render › Clouds / Difference Clouds, between the foreground and background colours.
fn clouds(s: &mut Session, p: &Value, difference: bool) -> Result<Value> {
    let (fg, bg) = (s.tools.foreground, s.tools.background);
    let seed = p.get("seed").and_then(Value::as_u64).unwrap_or(0) as u32;
    let size = s.active().map_or(256.0, |d| d.doc.size.width.max(d.doc.size.height) as f32);
    let base = (size / 4.0).clamp(16.0, 512.0);
    map_pixels(s, if difference { "Difference Clouds" } else { "Clouds" }, p, |area, px, _| {
        let w = area.width() as usize;
        for (i, p) in px.iter_mut().enumerate() {
            let (x, y) = ((area.x0 + (i % w) as i32) as f32, (area.y0 + (i / w) as i32) as f32);
            let t = clouds_value(x, y, base, seed);
            let c: [f32; 3] = std::array::from_fn(|k| fg[k] + (bg[k] - fg[k]) * t);
            if difference {
                for k in 0..3 {
                    p[k] = (p[k] - c[k]).abs();
                }
            } else {
                *p = [c[0], c[1], c[2], 1.0];
            }
        }
    })
}

// ---------- transform presets ----------

/// The presets' params: like Free Transform, they follow the Channels and Layers panel target.
const TRANSFORM_TARGET: &str =
    r#"{"target":"pixels"|"mask"|"quickMask"|{"channel":i}=Channels panel target (an unlinked mask, an alpha channel or the Quick Mask turns alone)}"#;

fn transform_preset(s: &mut Session, p: &Value, kind: &str) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    // A targeted alpha channel, the Quick Mask or an unlinked layer mask turns by itself (#2835).
    let (b, mut params) = match crate::transform_cmds::lone_target(&d.doc, d.active_layer, p)? {
        Some(surf) => (crate::transform_cmds::target_bounds(&d.doc, surf), json!({"target": p.get("target")})),
        None => {
            let id = d.active_layer.ok_or(EngineError::Other("no active layer".into()))?;
            let l = d.doc.layer(id).ok_or(EngineError::NoLayer(id))?;
            (crate::transform_cmds::transform_bounds(&d.doc, l), json!({"layer": id.0}))
        }
    };
    if b.is_empty() {
        return Err(EngineError::Other("nothing to transform".into()));
    }
    // Integer offsets keep these exact pixel permutations on the grid (nearest sampling).
    let (sx, sy) = ((b.x0 + b.x1) as f64, (b.y0 + b.y1) as f64);
    let (cx, cy) = (sx / 2.0, sy / 2.0);
    let m: [f64; 6] = match kind {
        "rotate180" => [-1.0, 0.0, 0.0, -1.0, sx, sy],
        "rotate90Cw" => [0.0, 1.0, -1.0, 0.0, (cx + cy).round(), (cy - cx).round()],
        "rotate90Ccw" => [0.0, -1.0, 1.0, 0.0, (cx - cy).round(), (cx + cy).round()],
        "flipHorizontal" => [-1.0, 0.0, 0.0, 1.0, sx, 0.0],
        _ => [1.0, 0.0, 0.0, -1.0, 0.0, sy],
    };
    params["matrix"] = json!(m);
    params["interpolation"] = json!("nearest");
    s.execute("edit.transform", params)
}

// ---------- selection / clipboard ----------

fn reselect_target(s: &Session) -> Option<photocraft_raster::Surface> {
    let d = s.active()?;
    if d.doc.selection.is_some() {
        return None;
    }
    (0..d.history.past_len()).rev().find_map(|i| d.history.state(i).and_then(|doc| doc.selection.clone()))
}

fn can_reselect(s: &Session) -> std::result::Result<(), String> {
    has_doc(s)?;
    reselect_target(s).map(|_| ()).ok_or_else(|| "there is no selection to restore".into())
}

fn paste_into(s: &mut Session, p: &Value, outside: bool) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let selection = d.doc.selection.clone().ok_or(EngineError::Other("Paste Into needs a selection".into()))?;
    let canvas = d.doc.bounds();
    let b = selection.content_bounds();
    // Where the paste shows: the selection, or everything but it (default reveal, the
    // selection's area inverted).
    let limit = if outside {
        let area = b.intersect(&canvas);
        let inv: Vec<f32> = sel::mask_from_surface(Some(&selection), area).iter().map(|v| 1.0 - v).collect();
        let mut m = photocraft_raster::Surface::with_default(photocraft_color::PixelFormat::GRAY8, &[1.0]);
        m.write_region(area, &inv);
        m
    } else {
        selection
    };
    let label = if outside { "Paste Outside" } else { "Paste Into" };
    let mut params = json!({"center": [(b.x0 + b.x1) as f64 / 2.0, (b.y0 + b.y1) as f64 / 2.0]});
    if let Some(c) = p.get("center") {
        params["center"] = c.clone();
    }
    // A targeted colour channel (#2307): the paste goes into it, only where `limit` allows.
    if let Some(k) = crate::channel_clip::paste_color_target(s, p) {
        return crate::channel_clip::paste(s, &params, k, false, Some(&limit), label);
    }
    if crate::channel_cmds::target_of(p) != crate::channel_cmds::Target::Pixels {
        // A targeted mask or channel (#1035): the paste goes into it, only where `limit` allows.
        params["target"] = p.get("target").cloned().unwrap_or_default();
        return crate::edit_cmds::paste_to_target(s, &params, false, Some(&limit), label);
    }
    let key = step_key(s, "pasteInto");
    params["coalesce"] = json!(key);
    params["target"] = json!("pixels");
    let before = s.active().ok_or(EngineError::NoDocument)?.doc.clone();
    let r = s.execute("edit.paste", params)?;
    let id = layer_param(s, &Value::Null)?;
    s.execute("layer.layerMask.revealAll", json!({"layer": id.0, "coalesce": key}))?;
    s.coalesce_request = Some(key);
    let r2 = s.edit(label, |doc, _| {
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        l.mask = Some(LayerMask { surface: limit, linked: false, ..LayerMask::reveal_all() });
        Ok(())
    });
    s.coalesce_request = None;
    r2?;
    // Only where the masked paste draws needs recompositing (#2901).
    crate::layer_multi_cmds::note_insert(s, &before);
    Ok(r)
}

/// The canvas grown to every layer's pixels: Image › Reveal All's new canvas, and what the Crop
/// tool shows while a frame is edited.
pub fn reveal_all_bounds(doc: &Document) -> Rect {
    doc.walk().into_iter().filter_map(|(_, _, l)| l.surface().map(|s| s.content_bounds())).fold(doc.bounds(), |a, b| a.union(&b))
}

// ---------- layers ----------

fn layer_from_background(s: &mut Session, p: &Value) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    if d.doc.background().is_none() && p.get("ifPresent").and_then(Value::as_bool) == Some(true) {
        return Ok(json!({"layer": null}));
    }
    // `select: false`: the targeted layer stays targeted, as when a Photoshop action's `set`
    // names the Background while another layer is selected.
    let select = p.get("select").and_then(Value::as_bool).unwrap_or(true);
    let id = s.edit("Layer From Background", |doc, active| {
        let l = doc.background_mut().ok_or(EngineError::Other("the document has no Background layer".into()))?;
        unlock_background(l);
        if select || active.is_none() {
            *active = Some(l.id);
        }
        Ok(l.id)
    })?;
    Ok(json!({"layer": id.0}))
}

pub(crate) fn unlock_background(l: &mut Layer) {
    l.name = "Layer 0".into();
    l.locks.transparency = false;
    l.locks.position = false;
}

/// Before `id` gets a layer mask: the Background can't have one, so it becomes a normal layer
/// first ("Layer 0"), as when adding a mask to it in Photoshop. Other layers are left alone.
pub(crate) fn background_to_layer_for_mask(doc: &mut photocraft_doc::Document, id: photocraft_doc::LayerId) {
    background_to_layer(doc, id);
}

/// If `id` is the Background ([`photocraft_doc::Document::background`], whatever its name), turns
/// it into a normal layer ("Layer 0"), as Photoshop does before an edit the Background can't
/// take (moving, transforming, exposing transparency, a style). Other layers are left alone.
pub(crate) fn background_to_layer(doc: &mut photocraft_doc::Document, id: photocraft_doc::LayerId) {
    if let Some(l) = doc.background_mut().filter(|l| l.id == id) {
        unlock_background(l);
    }
}

fn toggle_mask(s: &mut Session, p: &Value, key: &str) -> Result<Value> {
    let id = layer_param(s, p)?;
    let label = if key == "enabled" { "Enable/Disable Layer Mask" } else { "Link/Unlink Layer Mask" };
    let v = s.edit(label, |doc, _| {
        let m = doc.layer_mut(id).and_then(|l| l.mask.as_mut()).ok_or(EngineError::Other("the layer has no layer mask".into()))?;
        let slot = if key == "enabled" { &mut m.enabled } else { &mut m.linked };
        *slot = p.get(key).and_then(Value::as_bool).unwrap_or(!*slot);
        Ok(*slot)
    })?;
    Ok(json!({ key: v }))
}

fn set_all_effects(s: &mut Session, enabled: bool) -> Result<Value> {
    s.edit(if enabled { "Show All Effects" } else { "Hide All Effects" }, |doc, _| {
        fn rec(ls: &mut [Layer], on: bool) {
            for l in ls {
                if !l.effects.items.is_empty() {
                    l.effects.enabled = on;
                }
                if let Some(ch) = l.children_mut() {
                    rec(ch, on);
                }
            }
        }
        rec(&mut doc.layers, enabled);
        Ok(())
    })?;
    Ok(Value::Null)
}

/// Remove every layer matching `pred` (whole subtrees), keeping at least one layer.
fn delete_where(s: &mut Session, label: &str, pred: fn(&Layer) -> bool) -> Result<Value> {
    let n = s.edit(label, |doc, active| {
        fn rec(ls: &mut Vec<Layer>, pred: fn(&Layer) -> bool) -> usize {
            let before = ls.len();
            ls.retain(|l| !pred(l));
            let mut n = before - ls.len();
            for l in ls.iter_mut() {
                if let Some(ch) = l.children_mut() {
                    n += rec(ch, pred);
                }
            }
            n
        }
        let backup = doc.layers.clone();
        let n = rec(&mut doc.layers, pred);
        if doc.layers.is_empty() {
            doc.layers = backup;
            return Err(EngineError::Other("a document must keep at least one layer".into()));
        }
        if active.is_none_or(|id| doc.layer(id).is_none()) {
            *active = doc.top_layer();
        }
        Ok(n)
    })?;
    Ok(json!({"deleted": n}))
}

fn is_empty_layer(l: &Layer) -> bool {
    match &l.content {
        LayerContent::Raster(s) => s.content_bounds().is_empty() && l.mask.is_none() && l.effects.items.is_empty(),
        LayerContent::Group(g) => g.children.iter().all(is_empty_layer),
        _ => false,
    }
}

fn ungroup(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    s.edit("Ungroup Layers", |doc, active| {
        let path = doc.path_of(id).ok_or(EngineError::NoLayer(id))?;
        let g = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
        let LayerContent::Group(group) = &g.content else {
            return Err(EngineError::Other("the active layer is not a group".into()));
        };
        let children = group.children.clone();
        let (&idx, parent) = path.split_last().ok_or(EngineError::NoLayer(id))?;
        let sib = if parent.is_empty() { &mut doc.layers } else { doc.layer_at_mut(parent).and_then(Layer::children_mut).ok_or(EngineError::NoLayer(id))? };
        sib.remove(idx);
        let top = children.last().map(|l| l.id);
        for (k, l) in children.into_iter().enumerate() {
            sib.insert(idx + k, l);
        }
        *active = top.or_else(|| doc.top_layer());
        Ok(())
    })?;
    Ok(Value::Null)
}

fn set_visible(s: &mut Session, p: &Value, visible: bool) -> Result<Value> {
    if crate::layer_multi_cmds::multi(s, p) {
        return crate::layer_multi_cmds::set_visible_selected(s, visible);
    }
    let id = layer_param(s, p)?;
    s.edit(if visible { "Show Layer" } else { "Hide Layer" }, |doc, _| {
        doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?.visible = visible;
        Ok(())
    })?;
    Ok(Value::Null)
}

/// Layer › Hide Layers (⌘,): hide the targeted layers, or show them when every one is already
/// hidden, as Photoshop's item does (it reads Show Layers then), so ⌘, toggles.
fn hide_or_show(s: &mut Session, p: &Value) -> Result<Value> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let targets = if crate::layer_multi_cmds::multi(s, p) { crate::layer_multi_cmds::selected(s) } else { vec![layer_param(s, p)?] };
    let all_hidden = targets.iter().all(|id| st.doc.layer(*id).is_some_and(|l| !l.visible));
    set_visible(s, p, all_hidden)
}

/// `/`: Lock Transparent Pixels on or off for the targeted layers, as the Layers panel's first lock
/// button does. With several layers targeted, any unlocked one means lock them all. The
/// Background, which can't take this lock, is left out.
fn toggle_lock_transparency(s: &mut Session, p: &Value) -> Result<Value> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let targets: Vec<LayerId> = if crate::layer_multi_cmds::multi(s, p) { crate::layer_multi_cmds::selected(s) } else { vec![layer_param(s, p)?] };
    let targets: Vec<LayerId> = targets.into_iter().filter(|id| st.doc.layer(*id).is_some_and(|l| !st.doc.is_background(l.id))).collect();
    if targets.is_empty() {
        return Err(EngineError::Other("the Background layer can't lock transparent pixels".into()));
    }
    let lock = !targets.iter().all(|id| st.doc.layer(*id).is_some_and(|l| l.locks.transparency));
    s.edit(if lock { "Lock Transparent Pixels" } else { "Unlock Transparent Pixels" }, |doc, _| {
        for id in &targets {
            doc.layer_mut(*id).ok_or(EngineError::NoLayer(*id))?.locks.transparency = lock;
        }
        Ok(())
    })?;
    Ok(json!({"locked": lock}))
}

/// Is the layer at `path` shown alone: it and its enclosing groups visible, every layer outside
/// it hidden (the layers inside a group keep their own visibility)?
fn shown_alone(doc: &Document, path: &[usize]) -> bool {
    doc.walk().iter().all(|(p, _, l)| if path.starts_with(p) { l.visible } else { p.starts_with(path) || !l.visible })
}

/// ⌥-click on a layer's eye: show only that layer (and its enclosing groups), or, when it already
/// is shown alone, restore every layer's visibility from before (every layer shown when there is
/// no snapshot). ⌥-clicking another eye while one layer is shown alone moves the solo and keeps
/// the snapshot. One history step either way; the snapshot is view state.
fn show_only(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let doc = &st.doc;
    let path = doc.path_of(id).ok_or(EngineError::NoLayer(id))?;
    // The snapshot only counts while its solo is still in effect (an undo or a plain eye click ends it).
    let saved = st.show_only.as_ref().filter(|(prev, _)| doc.path_of(*prev).is_some_and(|pp| shown_alone(doc, &pp)));
    let walk = doc.walk();
    let solo = !shown_alone(doc, &path);
    let (label, rows, next): (_, Vec<(Vec<usize>, bool)>, _) = if solo {
        let snapshot = saved.map_or_else(|| walk.iter().map(|(_, _, l)| (l.id, l.visible)).collect(), |(_, v)| v.clone());
        // Layers inside the shown layer keep their visibility.
        let rows = walk.iter().filter(|(p, _, _)| p.len() <= path.len() || !p.starts_with(&path)).map(|(p, _, _)| (p.clone(), path.starts_with(p))).collect();
        ("Show Only This Layer", rows, Some((id, snapshot)))
    } else {
        let before: Option<std::collections::HashMap<LayerId, bool>> = saved.map(|(_, v)| v.iter().copied().collect());
        // Layers added since the snapshot keep their current visibility.
        let rows = walk.iter().map(|(p, _, l)| (p.clone(), before.as_ref().is_none_or(|b| b.get(&l.id).copied().unwrap_or(l.visible)))).collect();
        ("Show Layers", rows, None)
    };
    s.edit(label, |doc, _| {
        for (p, visible) in &rows {
            if let Some(l) = doc.layer_at_mut(p) {
                l.visible = *visible;
            }
        }
        Ok(())
    })?;
    if let Some(st) = s.active_mut() {
        st.show_only = next;
    }
    Ok(json!({ "shownAlone": solo }))
}

/// Pixels of a fill or smart-object layer's content alone (no mask, effects or opacity).
fn content_pixels(doc: &Document, l: &Layer) -> photocraft_raster::Surface {
    let mut tmp = l.clone();
    tmp.mask = None;
    tmp.vector_mask = None;
    tmp.effects.items.clear();
    tmp.opacity = 1.0;
    tmp.fill_opacity = 1.0;
    tmp.blend = photocraft_color::BlendMode::Normal;
    tmp.clipped = false;
    tmp.visible = true;
    let canvas = doc.bounds();
    let buf = photocraft_compose::render_layer(&tmp, canvas);
    let fmt = doc.pixel_format();
    let n = fmt.channels();
    let mut data = vec![0.0f32; buf.px.len() * n];
    for (p, out) in buf.px.iter().zip(data.chunks_exact_mut(n)) {
        from_rgba_into(&fmt, *p, out);
    }
    let mut s = photocraft_raster::Surface::new(fmt);
    s.write_region(canvas, &data);
    s.prune();
    s
}

pub(crate) fn kind_of(s: &Session, id: LayerId) -> Option<&'static str> {
    let d = s.active()?;
    Some(match &d.doc.layer(id)?.content {
        LayerContent::Text(_) => "type",
        LayerContent::Shape(_) => "shape",
        LayerContent::Fill(_) => "fill",
        LayerContent::Smart(_) => "smart",
        _ => return None,
    })
}

/// Rasterize one layer; `only` restricts it to one kind. Without `only` (Rasterize › Layer / All
/// Layers: "all vector data") a vector mask becomes a pixel mask too; layer effects stay live, as
/// in Photoshop (Rasterize › Layer Style bakes them). Returns false when there was nothing to do.
pub(crate) fn rasterize_one(s: &mut Session, id: LayerId, only: Option<&str>, key: &str) -> Result<bool> {
    let vector_mask = only.is_none() && s.active().and_then(|d| d.doc.layer(id)).is_some_and(|l| l.vector_mask.is_some());
    let content = rasterize_content(s, id, only, key)?;
    if vector_mask {
        s.execute("layer.rasterize.vectorMask", json!({"layer": id.0, "coalesce": key}))?;
    }
    Ok(content || vector_mask)
}

fn rasterize_content(s: &mut Session, id: LayerId, only: Option<&str>, key: &str) -> Result<bool> {
    let Some(kind) = kind_of(s, id).filter(|k| only.is_none_or(|o| o == *k)) else { return Ok(false) };
    match kind {
        "type" => {
            s.execute("type.rasterize", json!({"layer": id.0, "coalesce": key}))?;
        }
        "shape" => {
            s.execute("layer.rasterize.shape", json!({"layer": id.0, "coalesce": key}))?;
        }
        "smart" => {
            s.coalesce_request = Some(key.to_string());
            let r = s.edit("Rasterize Smart Object", |doc, _| {
                let snapshot = doc.clone();
                let mut snap = doc.layer(id).cloned().ok_or(EngineError::NoLayer(id))?;
                if matches!(snap.content, LayerContent::Smart(ref sm) if sm.cache.is_none()) {
                    let _ = crate::smart_cmds::refresh_layer(&snapshot, &mut snap);
                }
                let fmt = snapshot.pixel_format();
                let surface = match &mut snap.content {
                    LayerContent::Smart(sm) => match sm.cache.take() {
                        Some(c) => c,
                        None => content_pixels(&snapshot, &snap),
                    },
                    _ => return Err(EngineError::Other(format!("layer {} is not a smart object", id.0))),
                };
                let surface = if surface.format() == fmt { surface } else { surface.convert(fmt) };
                let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
                l.content = LayerContent::Raster(surface);
                l.psd_blocks.retain(|(k, _)| k != b"SoLd" && k != b"SoLE" && k != b"PlLd");
                Ok(())
            });
            s.coalesce_request = None;
            r?;
        }
        _ => {
            s.coalesce_request = Some(key.to_string());
            let r = s.edit("Rasterize Fill Content", |doc, _| {
                let snap = doc.layer(id).cloned().ok_or(EngineError::NoLayer(id))?;
                let px = content_pixels(doc, &snap);
                let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
                l.content = LayerContent::Raster(px);
                l.fill_cache = None;
                // The PSD fill or placed-layer blocks no longer describe the layer: kept, they
                // would reopen it as a fill layer or smart object and drop these pixels.
                l.psd_blocks.retain(|(k, _)| !matches!(k, b"SoCo" | b"GdFl" | b"PtFl" | b"SoLd" | b"PlLd" | b"SoLE"));
                Ok(())
            });
            s.coalesce_request = None;
            r?;
        }
    }
    Ok(true)
}

/// Layer › Rasterize › Layer / Type / Shape / Fill Content / Smart Object. With a `layer` param,
/// just that layer. Otherwise, as in Photoshop ("select the layer or layers you'd like to
/// rasterize"), every selected layer the command applies to, in one history step.
pub(crate) fn rasterize(s: &mut Session, p: &Value, only: Option<&'static str>) -> Result<Value> {
    let ids = crate::layer_multi_cmds::targets(s, p)?;
    let many = ids.len() > 1;
    let key = step_key(s, "rasterize");
    let mut done = Vec::new();
    for id in ids {
        if rasterize_one(s, id, only, &key)? {
            done.push(id.0);
        }
    }
    let Some(&first) = done.first() else {
        return Err(EngineError::Other(match (only, many) {
            (Some(k), false) => format!("the layer is not a {k} layer"),
            (Some(k), true) => format!("none of the selected layers is a {k} layer"),
            (None, false) => "the layer has nothing to rasterize".into(),
            (None, true) => "none of the selected layers has anything to rasterize".into(),
        }));
    };
    let active = s.active().and_then(|d| d.active_layer).map(|id| id.0).filter(|id| done.contains(id));
    Ok(json!({"layer": active.unwrap_or(first), "layers": done}))
}

pub(crate) fn rasterize_all(s: &mut Session) -> Result<Value> {
    let ids: Vec<LayerId> = s.active().map(|d| d.doc.walk().into_iter().map(|(_, _, l)| l.id).collect()).unwrap_or_default();
    let key = step_key(s, "rasterizeAll");
    let mut n = 0;
    for id in ids {
        // Every sub-step shares one history entry via the coalescing key.
        if rasterize_one(s, id, None, &key)? {
            n += 1;
        }
    }
    Ok(json!({"rasterized": n}))
}

// ---------- registry ----------

pub fn specs() -> Vec<CommandSpec> {
    macro_rules! spec {
        ($id:expr, $label:expr, $menu:expr, $sc:expr, $params:expr, $enabled:expr, $run:expr) => {
            CommandSpec { id: $id, label: $label, menu: $menu, shortcut: $sc, params: $params, enabled: $enabled, journal: true, run: $run }
        };
    }
    vec![
        spec!(
            "edit.stroke",
            "Stroke…",
            &["Edit"],
            None,
            r##"{"width":1..250=1,"color":"#rrggbb|[r,g,b,a]"=foreground,"location":"inside|center|outside"="center","mode":"any layer blend mode"="normal","opacity":0..100=100,"preserveTransparency":bool=false}"##,
            has_pixels,
            stroke
        ),
        spec!("edit.transform.rotate180", "Rotate 180°", &["Edit", "Transform"], None, TRANSFORM_TARGET, has_layer, |s, p| transform_preset(
            s,
            p,
            "rotate180"
        )),
        spec!("edit.transform.rotate90Cw", "Rotate 90° Clockwise", &["Edit", "Transform"], None, TRANSFORM_TARGET, has_layer, |s, p| transform_preset(
            s,
            p,
            "rotate90Cw"
        )),
        spec!("edit.transform.rotate90Ccw", "Rotate 90° Counter Clockwise", &["Edit", "Transform"], None, TRANSFORM_TARGET, has_layer, |s, p| {
            transform_preset(s, p, "rotate90Ccw")
        }),
        spec!("edit.transform.flipHorizontal", "Flip Horizontal", &["Edit", "Transform"], None, TRANSFORM_TARGET, has_layer, |s, p| transform_preset(
            s,
            p,
            "flipHorizontal"
        )),
        spec!("edit.transform.flipVertical", "Flip Vertical", &["Edit", "Transform"], None, TRANSFORM_TARGET, has_layer, |s, p| transform_preset(
            s,
            p,
            "flipVertical"
        )),
        spec!(
            "edit.pasteSpecial.pasteInto",
            "Paste Into",
            &["Edit", "Paste Special"],
            Some("Cmd+Alt+Shift+V"),
            concat!(r#"{"center":[x,y]?,"#, crate::edit_cmds::paste_target!(), "}"),
            has_clip_and_selection,
            |s, p| paste_into(s, p, false)
        ),
        spec!(
            "edit.pasteSpecial.pasteOutside",
            "Paste Outside",
            &["Edit", "Paste Special"],
            None,
            concat!(r#"{"center":[x,y]?,"#, crate::edit_cmds::paste_target!(), "}"),
            has_clip_and_selection,
            |s, p| paste_into(s, p, true)
        ),
        spec!("select.reselect", "Reselect", &["Select"], Some("Cmd+Shift+D"), "{}", can_reselect, |s, _| {
            let m = reselect_target(s).ok_or(EngineError::Other("there is no selection to restore".into()))?;
            s.edit("Reselect", |doc, _| {
                doc.selection = Some(m);
                Ok(())
            })?;
            Ok(json!({"selected": true}))
        }),
        spec!("image.adjustments.equalize", "Equalize", &["Image", "Adjustments"], None, "{}", has_pixels, equalize),
        spec!("image.revealAll", "Reveal All", &["Image"], None, "{}", has_doc, |s, _| {
            let d = s.active().ok_or(EngineError::NoDocument)?;
            let all = reveal_all_bounds(&d.doc);
            if all == d.doc.bounds() {
                return Ok(json!({"changed": false}));
            }
            s.execute("image.crop", json!({"x": all.x0, "y": all.y0, "width": all.width(), "height": all.height(), "deleteCroppedPixels": false}))?;
            Ok(json!({"changed": true}))
        }),
        spec!(
            "layer.new.layerFromBackground",
            "Layer From Background…",
            &["Layer", "New"],
            None,
            r##"{"ifPresent":bool=false (true: a document without a Background is left as it is, as a Photoshop action's step needs on an image that opened as a normal layer),"select":bool=true (false: the targeted layer stays targeted)}"##,
            has_background,
            layer_from_background
        ),
        spec!("layer.layerStyle.copyLayerStyle", "Copy Layer Style", &["Layer", "Layer Style"], None, r##"{"layer":id?}"##, has_layer, |s, p| {
            let id = layer_param(s, p)?;
            let l = s.active().and_then(|d| d.doc.layer(id)).ok_or(EngineError::NoLayer(id))?;
            let style = (l.effects.clone(), l.blend, l.fill_opacity, l.advanced);
            s.style_clipboard = Some(style);
            Ok(Value::Null)
        }),
        spec!(
            "layer.layerStyle.pasteLayerStyle",
            "Paste Layer Style",
            &["Layer", "Layer Style"],
            None,
            r##"{"layer":id?} (no layer: every selected layer but the Background)"##,
            |s| {
                has_layer(s)?;
                s.style_clipboard.as_ref().map(|_| ()).ok_or_else(|| "no layer style has been copied".into())
            },
            |s, p| {
                let ids = style_targets(s, p, |_| true)?;
                let (mut fx, blend, fill, advanced) = s.style_clipboard.clone().ok_or(EngineError::Other("no layer style has been copied".into()))?;
                s.edit("Paste Layer Style", |doc, _| {
                    // A style copied from a document in another mode takes this one's colours.
                    let mode = doc.mode;
                    fx.items = std::mem::take(&mut fx.items).into_iter().map(|e| e.in_mode(mode)).collect();
                    for id in ids {
                        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
                        l.effects = fx.clone();
                        l.blend = blend;
                        l.fill_opacity = fill;
                        l.advanced = advanced;
                    }
                    Ok(())
                })?;
                Ok(Value::Null)
            }
        ),
        spec!("layer.layerStyle.hideAllEffects", "Hide All Effects", &["Layer", "Layer Style"], None, "{}", has_doc, |s, _| set_all_effects(s, false)),
        spec!("layer.layerStyle.showAllEffects", "Show All Effects", &[], None, "{}", has_doc, |s, _| set_all_effects(s, true)),
        spec!(
            "layer.layerMask.enabled",
            "Disable Layer Mask",
            &["Layer", "Layer Mask"],
            None,
            r##"{"layer":id?,"enabled":bool? (default: toggle)}"##,
            has_mask,
            |s, p| toggle_mask(s, p, "enabled")
        ),
        spec!(
            "layer.layerMask.linked",
            "Unlink Layer Mask",
            &["Layer", "Layer Mask"],
            None,
            r##"{"layer":id?,"linked":bool? (default: toggle)}"##,
            has_mask,
            |s, p| toggle_mask(s, p, "linked")
        ),
        spec!(
            "layer.rasterize.layer",
            "Layer",
            &["Layer", "Rasterize"],
            None,
            r##"{"layer":id?} (no layer: every selected layer it applies to)"##,
            has_layer,
            |s, p| rasterize(s, p, None)
        ),
        spec!("layer.rasterize.allLayers", "All Layers", &["Layer", "Rasterize"], None, "{}", has_doc, |s, _| rasterize_all(s)),
        spec!(
            "layer.rasterize.type",
            "Type",
            &["Layer", "Rasterize"],
            None,
            r##"{"layer":id?} (no layer: every selected layer it applies to)"##,
            has_layer,
            |s, p| rasterize(s, p, Some("type"))
        ),
        spec!(
            "layer.rasterize.fillContent",
            "Fill Content",
            &["Layer", "Rasterize"],
            None,
            r##"{"layer":id?} (no layer: every selected layer it applies to)"##,
            has_layer,
            |s, p| rasterize(s, p, Some("fill"))
        ),
        spec!(
            "layer.rasterize.smartObject",
            "Smart Object",
            &["Layer", "Rasterize"],
            None,
            r##"{"layer":id?} (no layer: every selected layer it applies to)"##,
            has_layer,
            |s, p| rasterize(s, p, Some("smart"))
        ),
        spec!("layer.delete.hiddenLayers", "Hidden Layers", &["Layer", "Delete"], None, "{}", has_doc, |s, _| delete_where(s, "Delete Hidden Layers", |l| !l
            .visible)),
        spec!("file.scripts.deleteAllEmptyLayers", "Delete All Empty Layers", &["File", "Scripts"], None, "{}", has_doc, |s, _| delete_where(
            s,
            "Delete All Empty Layers",
            is_empty_layer
        )),
        spec!(
            "layer.ungroupLayers",
            "Ungroup Layers",
            &["Layer"],
            Some("Cmd+Shift+G"),
            r##"{"layer":id?}"##,
            |s| match active_layer(s)?.is_group() {
                true => Ok(()),
                false => Err("the active layer is not a group".into()),
            },
            ungroup
        ),
        spec!(
            "layer.hideLayers",
            "Hide Layers",
            &["Layer"],
            Some("Cmd+,"),
            r##"{"layer":id?} (shows them instead when every targeted layer is already hidden, so ⌘, toggles)"##,
            has_layer,
            hide_or_show
        ),
        spec!("layer.showLayers", "Show Layers", &[], None, r##"{"layer":id?}"##, has_layer, |s, p| set_visible(s, p, true)),
        spec!(
            "layer.toggleLockTransparency",
            "Lock Transparent Pixels",
            &[],
            Some("/"),
            r##"{"layer":id?} (toggles; several layers selected: locks them all unless every one is locked)"##,
            has_layer,
            toggle_lock_transparency
        ),
        spec!(
            "layer.showOnly",
            "Show Only This Layer",
            &[],
            None,
            r##"{"layer":id?} → {shownAlone} (⌥-click a layer's eye; again restores the other layers' visibility)"##,
            has_layer,
            show_only
        ),
        spec!("filter.blur.average", "Average", &["Filter", "Blur"], None, "{}", has_pixels, average),
        spec!("filter.render.clouds", "Clouds", &["Filter", "Render"], None, r##"{"seed":u32=0}"##, has_pixels, |s, p| clouds(s, p, false)),
        spec!("filter.render.differenceClouds", "Difference Clouds", &["Filter", "Render"], None, r##"{"seed":u32=0}"##, has_pixels, |s, p| clouds(s, p, true)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(depth: u32) -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 40, "height": 30, "depth": depth})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s
    }

    fn doc(s: &Session) -> &Document {
        &s.active().unwrap().doc
    }

    #[test]
    fn equalize_average_and_clouds_edit_the_targeted_channel() {
        // #935: these four ran through `map_pixels`, which always rewrote the active layer, so
        // with an alpha channel or the Quick Mask targeted they changed the layer and left the
        // channel alone.
        let cmds = ["image.adjustments.equalize", "filter.blur.average", "filter.render.clouds", "filter.render.differenceClouds"];
        for depth in [8, 16, 32] {
            for target in ["quickMask", "alpha", "mask"] {
                for cmd in cmds {
                    let mut s = session(depth);
                    s.tools.foreground = [0.0, 0.0, 0.0, 1.0];
                    s.tools.background = [1.0, 1.0, 1.0, 1.0];
                    s.execute("edit.fill", json!({"contents": "color", "color": "#808080"})).unwrap();
                    let p = match target {
                        "quickMask" => {
                            s.execute("select.editInQuickMaskMode", json!({"on": true})).unwrap();
                            json!({"seed": 1})
                        }
                        "alpha" => {
                            s.execute("channel.new", json!({"fill": "white"})).unwrap();
                            s.execute("channel.target", json!({"channel": 0})).unwrap();
                            json!({"seed": 1})
                        }
                        _ => {
                            s.execute("layer.layerMask.revealAll", json!({})).unwrap();
                            json!({"seed": 1, "target": "mask"})
                        }
                    };
                    // Two levels in the target, so Equalize and Average have something to change.
                    let edit_target = json!({ "target": if target == "alpha" { json!({"channel": 0}) } else { json!(target) } });
                    s.edit("levels", |doc, active| {
                        let (surf, _) = crate::channel_cmds::target_surface(doc, *active, &edit_target).unwrap();
                        surf.fill_rect(Rect::new(0, 0, 20, 30), &[0.4]);
                        surf.fill_rect(Rect::new(20, 0, 40, 30), &[0.5]);
                        Ok(())
                    })
                    .unwrap();
                    let full = Rect::new(0, 0, 40, 30);
                    let read = |s: &Session| {
                        let d = doc(s);
                        let layer = d.layer(s.active().unwrap().active_layer.unwrap()).unwrap();
                        let t = match target {
                            "quickMask" => d.quick_mask.as_ref().unwrap().surface.read_region(full),
                            "alpha" => d.channels[0].surface.read_region(full),
                            _ => layer.mask.as_ref().unwrap().surface.read_region(full),
                        };
                        (layer.surface().unwrap().read_region(full), t)
                    };
                    let ((layer0, target0), past) = (read(&s), s.active().unwrap().history.past_len());
                    s.execute(cmd, p).unwrap();
                    let (layer1, target1) = read(&s);
                    let case = format!("{cmd} on {target} at {depth}");
                    assert_eq!(layer1, layer0, "{case}: the layer is untouched");
                    assert_ne!(target1, target0, "{case}: the target changed");
                    assert_eq!(s.active().unwrap().history.past_len(), past + 1, "{case}");
                    s.undo();
                    assert_eq!(read(&s).1, target0, "{case}: undo restores the target");
                }
            }
        }
    }

    fn active(s: &Session) -> &Layer {
        let d = s.active().unwrap();
        d.doc.layer(d.active_layer.unwrap()).unwrap()
    }

    fn pixel(s: &Session, x: i32, y: i32) -> Vec<f32> {
        active(s).surface().unwrap().pixel(x, y)
    }

    fn paint_square(s: &mut Session, r: Rect, c: [f32; 4]) {
        s.edit("paint", |doc, a| {
            doc.layer_mut(a.unwrap()).unwrap().surface_mut().unwrap().fill_rect(r, &c);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn stroke_locations_at_several_depths() {
        for depth in [8, 16, 32] {
            for (loc, inside, outside) in [("inside", 1.0, 0.0), ("outside", 0.0, 1.0), ("center", 1.0, 1.0)] {
                let mut s = session(depth);
                s.execute("select.rect", json!({"x": 10, "y": 10, "width": 10, "height": 10})).unwrap();
                s.execute("edit.stroke", json!({"width": 2, "color": "#ff0000", "location": loc})).unwrap();
                assert!((pixel(&s, 10, 15)[3] - inside).abs() < 0.02, "{depth} {loc} inside {:?}", pixel(&s, 10, 15));
                assert!((pixel(&s, 9, 15)[3] - outside).abs() < 0.02, "{depth} {loc} outside {:?}", pixel(&s, 9, 15));
                assert_eq!(pixel(&s, 15, 15)[3], 0.0, "centre untouched");
            }
        }
        // Without a selection the layer's opaque edge is stroked.
        let mut s = session(8);
        paint_square(&mut s, Rect::new(10, 10, 20, 20), [0.0, 0.0, 1.0, 1.0]);
        s.execute("edit.stroke", json!({"width": 3, "color": "#00ff00", "location": "outside"})).unwrap();
        assert_eq!(pixel(&s, 8, 15), vec![0.0, 1.0, 0.0, 1.0]);
        assert_eq!(pixel(&s, 15, 15), vec![0.0, 0.0, 1.0, 1.0]);
    }

    /// Photoshop's Edit › Stroke on a selection (checked by hand): Center puts ceil(w/2) px inside
    /// and floor(w/2) outside (1 px is all inside), fractional widths round up, and every stroked
    /// pixel is fully opaque, on every edge, horizontal ones included. Inside and Outside are
    /// w px on their side.
    #[test]
    fn stroke_widths_are_whole_opaque_pixels_like_photoshop() {
        // Selection x 10..30, y 10..30; the edge pixels inside are 10 and 29 on each axis.
        let band = |loc: &str, w: i32| -> (i32, i32) {
            match loc {
                "inside" => (w, 0),
                "outside" => (0, w),
                _ => ((w + 1) / 2, w / 2),
            }
        };
        let mut wrong = Vec::new();
        for depth in [8, 16, 32] {
            for loc in ["center", "inside", "outside"] {
                for (param, w) in [(json!(1), 1), (json!(2), 2), (json!(3), 3), (json!(4), 4), (json!(5), 5), (json!(0.5), 1), (json!(2.5), 3)] {
                    let mut s = Session::new();
                    s.execute("file.new", json!({"width": 50, "height": 50, "depth": depth})).unwrap();
                    s.execute("layer.new.layer", json!({})).unwrap();
                    s.execute("select.rect", json!({"x": 10, "y": 10, "width": 20, "height": 20})).unwrap();
                    s.execute("edit.stroke", json!({"width": param, "color": "#ff0000", "location": loc})).unwrap();
                    let (inside, outside) = band(loc, w);
                    // (edge pixel inside, step inward) for left, right, top and bottom; the other
                    // coordinate is the middle of the edge.
                    for (name, first_in, inward, horizontal) in
                        [("left", 10, 1, false), ("right", 29, -1, false), ("top", 10, 1, true), ("bottom", 29, -1, true)]
                    {
                        for k in -7..7 {
                            // k ≥ 0: k px inside from the edge; k < 0: -k px outside.
                            let c = first_in + k * inward;
                            let (x, y) = if horizontal { (20, c) } else { (c, 20) };
                            let want = if (k >= 0 && k < inside) || (k < 0 && -k <= outside) { 1.0 } else { 0.0 };
                            let a = pixel(&s, x, y)[3];
                            if (a - want).abs() > 0.004 {
                                wrong.push(format!("{depth}-bit {loc} width {param} {name} edge, {k:+} px: alpha {a}, want {want}"));
                            }
                        }
                    }
                }
            }
        }
        assert!(wrong.is_empty(), "{} pixels differ from Photoshop:\n{}", wrong.len(), wrong.iter().take(40).cloned().collect::<Vec<_>>().join("\n"));
    }

    /// Photoshop's 6 px Outside stroke of a 14 px circle selection made without anti-aliasing,
    /// read pixel by pixel from Photoshop (`#` stroked, `.` not; the hole is the selection). A
    /// hard selection gives a hard stroke: a pixel is stroked when its centre is less than
    /// width + 1 from a selected pixel's centre (no partial pixels on the curve).
    const PS_CIRCLE_OUTSIDE_6: [&str; 31] = [
        "..................................",
        "..................................",
        "...........############...........",
        "..........##############..........",
        ".........################.........",
        "........##################........",
        ".......####################.......",
        "......######################......",
        ".....#########......#########.....",
        "....#########........#########....",
        "....########..........########....",
        "....#######............#######....",
        "....######..............######....",
        "....######..............######....",
        "....######..............######....",
        "....######..............######....",
        "....######..............######....",
        "....######..............######....",
        "....#######............#######....",
        "....########..........########....",
        "....#########........#########....",
        ".....#########......#########.....",
        "......######################......",
        ".......####################.......",
        "........##################........",
        ".........################.........",
        "..........##############..........",
        "...........############...........",
        "..................................",
        "..................................",
        "..................................",
    ];

    /// Photoshop's 3 px Inside stroke of the same 14 px hard circle (the selection is the ring
    /// plus its hole). Inside, a pixel is stroked when its centre is less than width + ½ from an
    /// unselected pixel's: that offset is the one fitting this, the 5 px and the 6 px samples
    /// (any value from +0.41 to +0.60), so 6 px fills this circle (its deepest pixel is 6.4 px in).
    const PS_CIRCLE_INSIDE_3: [&str; 17] = [
        "...................",
        "......######.......",
        ".....########......",
        "....##########.....",
        "...####....####....",
        "..####......####...",
        "..###........###...",
        "..###........###...",
        "..###........###...",
        "..###........###...",
        "..####......####...",
        "...####....####....",
        "....##########.....",
        ".....########......",
        "......######.......",
        "...................",
        "...................",
    ];

    /// The selection of a Photoshop sample grid: an Outside stroke's hole, or an Inside stroke's
    /// ring and hole.
    fn grid_selection(rows: &[&str], outside: bool) -> Vec<f32> {
        let (w, h) = (rows[0].len(), rows.len());
        let mut mask = vec![0.0f32; w * h];
        for (y, row) in rows.iter().enumerate() {
            let b = row.as_bytes();
            if let (Some(first), Some(last)) = (b.iter().position(|c| *c == b'#'), b.iter().rposition(|c| *c == b'#')) {
                for x in first..=last {
                    if !outside || b[x] == b'.' {
                        mask[y * w + x] = 1.0;
                    }
                }
            }
        }
        mask
    }

    /// Strokes `mask` as the selection of a `w` × `h` layer and returns its alpha per pixel.
    fn stroke_alpha(mask: &[f32], w: usize, h: usize, depth: u32, params: Value) -> Vec<f32> {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": w, "height": h, "depth": depth})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.edit("select", |doc, _| {
            doc.selection = Some(sel::mask_to_surface(mask, doc.bounds()));
            Ok(())
        })
        .unwrap();
        s.execute("edit.stroke", params).unwrap();
        (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| pixel(&s, x as i32, y as i32)[3]).collect()
    }

    #[test]
    fn a_hard_circle_strokes_inside_like_photoshop() {
        let rows = PS_CIRCLE_INSIDE_3;
        let (w, h) = (rows[0].len(), rows.len());
        let mask = grid_selection(&rows, false);
        for depth in [8, 16, 32] {
            let got = stroke_alpha(&mask, w, h, depth, json!({"width": 3, "color": "#000000", "location": "inside"}));
            let wrong: Vec<String> = rows
                .iter()
                .enumerate()
                .flat_map(|(y, row)| row.bytes().enumerate().map(move |(x, c)| (x, y, c)))
                .filter_map(|(x, y, c)| {
                    let want = if c == b'#' { 1.0 } else { 0.0 };
                    let a = got[y * w + x];
                    ((a - want).abs() > 0.004).then(|| format!("({x}, {y}): {a}, Photoshop {want}"))
                })
                .collect();
            assert!(wrong.is_empty(), "{depth}-bit: {} pixels differ from Photoshop: {wrong:?}", wrong.len());
            // Checked in Photoshop: a 5 px Inside stroke leaves a plus-shaped hole of 12 pixels in
            // the middle (rows 2, 4, 4, 2 wide), and 6 px fills the whole circle.
            let hole = [(8, 6), (9, 6), (7, 7), (8, 7), (9, 7), (10, 7), (7, 8), (8, 8), (9, 8), (10, 8), (8, 9), (9, 9)];
            let five = stroke_alpha(&mask, w, h, depth, json!({"width": 5, "color": "#000000", "location": "inside"}));
            for (i, m) in mask.iter().enumerate() {
                let stroked = *m > 0.0 && !hole.contains(&(i % w, i / w));
                assert_eq!(five[i] > 0.996, stroked, "{depth}-bit 5 px Inside at ({}, {})", i % w, i / w);
            }
            let full = stroke_alpha(&mask, w, h, depth, json!({"width": 6, "color": "#000000", "location": "inside"}));
            for (i, m) in mask.iter().enumerate() {
                assert_eq!(full[i] > 0.996, *m > 0.0, "{depth}-bit 6 px Inside at ({}, {})", i % w, i / w);
            }
        }
    }

    #[test]
    fn a_hard_circle_strokes_like_photoshop() {
        let rows = PS_CIRCLE_OUTSIDE_6;
        let (w, h) = (rows[0].len(), rows.len());
        // The selection: the `.` cells enclosed by the ring on each row.
        let mask = grid_selection(&rows, true);
        for depth in [8, 16, 32] {
            let mut s = Session::new();
            s.execute("file.new", json!({"width": w, "height": h, "depth": depth})).unwrap();
            s.execute("layer.new.layer", json!({})).unwrap();
            s.edit("select", |doc, _| {
                doc.selection = Some(sel::mask_to_surface(&mask, doc.bounds()));
                Ok(())
            })
            .unwrap();
            s.execute("edit.stroke", json!({"width": 6, "color": "#000000", "location": "outside"})).unwrap();
            let mut wrong = Vec::new();
            for (y, row) in rows.iter().enumerate() {
                for (x, c) in row.bytes().enumerate() {
                    let want = if c == b'#' { 1.0 } else { 0.0 };
                    let a = pixel(&s, x as i32, y as i32)[3];
                    if (a - want).abs() > 0.004 {
                        wrong.push(format!("({x}, {y}): {a}, Photoshop {want}"));
                    }
                }
            }
            assert!(wrong.is_empty(), "{depth}-bit: {} pixels differ from Photoshop: {wrong:?}", wrong.len());
        }
    }

    /// Select All, then Stroke › Inside draws a border around the canvas (a common Photoshop
    /// technique): the canvas edge is a selection edge like any other, on all four sides.
    #[test]
    fn stroking_a_selection_at_the_canvas_edge_draws_that_edge() {
        for depth in [8, 16, 32] {
            for (loc, w, inside) in [("inside", 3, 3), ("center", 3, 2), ("center", 1, 1)] {
                let mut s = session(depth);
                s.execute("select.all", json!({})).unwrap();
                s.execute("edit.stroke", json!({"width": w, "color": "#ff0000", "location": loc})).unwrap();
                // 40 × 30 canvas: k px in from each side, in the middle of the side.
                for name in ["left", "right", "top", "bottom"] {
                    for k in 0..6 {
                        let (x, y) = match name {
                            "left" => (k, 15),
                            "right" => (39 - k, 15),
                            "top" => (20, k),
                            _ => (20, 29 - k),
                        };
                        let want = if k < inside { 1.0 } else { 0.0 };
                        let a = pixel(&s, x, y)[3];
                        assert!((a - want).abs() < 0.004, "{depth}-bit {loc} {w} px, {name} side, {k} px in: alpha {a}, want {want}");
                    }
                }
            }
        }
    }

    #[test]
    fn stroke_uses_fill_blending_and_preserves_transparency_at_every_depth() {
        for depth in [8, 16, 32] {
            let mut s = session(depth);
            paint_square(&mut s, Rect::new(10, 10, 20, 20), [0.5, 0.5, 0.5, 1.0]);
            s.execute("select.rect", json!({"x": 10, "y": 10, "width": 10, "height": 10})).unwrap();

            let past = s.active().unwrap().history.past_len();
            s.execute(
                "edit.stroke",
                json!({"width": 2, "color": "#ffffff", "location": "inside", "mode": "multiply", "opacity": 100, "preserveTransparency": true}),
            )
            .unwrap();
            assert_eq!(s.active().unwrap().history.past_len(), past + 1, "Stroke is one history step");
            let edge = pixel(&s, 10, 15);
            assert!((edge[0] - 0.5).abs() < 0.015, "{depth}-bit Multiply must preserve gray: {edge:?}");
            assert_eq!(edge[3], 1.0);

            s.execute("edit.stroke", json!({"width": 2, "color": "#ffffff", "location": "outside", "preserveTransparency": true})).unwrap();
            assert_eq!(pixel(&s, 9, 15)[3], 0.0, "{depth}-bit transparency lock prevents new outer pixels");

            s.execute("edit.stroke", json!({"width": 2, "color": "#ffffff", "location": "outside", "opacity": 50})).unwrap();
            let out = pixel(&s, 9, 15);
            assert!((out[3] - 0.5).abs() < 0.015, "{depth}-bit 50% stroke opacity: {out:?}");
            s.execute("edit.undo", json!({})).unwrap();
            assert_eq!(pixel(&s, 9, 15)[3], 0.0);
        }
    }

    #[test]
    fn stroke_rejects_invalid_blending_options_without_modifying_the_document() {
        let mut s = session(8);
        s.execute("select.rect", json!({"x": 10, "y": 10, "width": 10, "height": 10})).unwrap();
        let past = s.active().unwrap().history.past_len();
        for params in [
            json!({"mode": "unknown-mode"}),
            json!({"mode": 42}),
            json!({"mode": "passThrough"}),
            json!({"opacity": "lots"}),
            json!({"preserveTransparency": "true"}),
        ] {
            assert!(s.execute("edit.stroke", params.clone()).is_err(), "{params}");
            assert_eq!(s.active().unwrap().history.past_len(), past, "{params} should not create a history step");
        }
    }

    #[test]
    fn transform_presets_are_exact() {
        let mut s = session(8);
        paint_square(&mut s, Rect::new(10, 10, 20, 14), [1.0, 0.0, 0.0, 1.0]);
        paint_square(&mut s, Rect::new(10, 10, 11, 11), [0.0, 1.0, 0.0, 1.0]);
        s.execute("edit.transform.flipHorizontal", json!({})).unwrap();
        assert_eq!(active(&s).surface().unwrap().content_bounds(), Rect::new(10, 10, 20, 14));
        assert_eq!(pixel(&s, 19, 10), vec![0.0, 1.0, 0.0, 1.0]);
        s.execute("edit.transform.flipVertical", json!({})).unwrap();
        assert_eq!(pixel(&s, 19, 13), vec![0.0, 1.0, 0.0, 1.0]);
        s.execute("edit.transform.rotate180", json!({})).unwrap();
        assert_eq!(pixel(&s, 10, 10), vec![0.0, 1.0, 0.0, 1.0]);
        s.execute("edit.transform.rotate90Cw", json!({})).unwrap();
        let b = active(&s).surface().unwrap().content_bounds();
        assert_eq!((b.width(), b.height()), (4, 10));
        // Top-left corner goes to the top-right after a clockwise turn.
        assert_eq!(pixel(&s, b.x1 - 1, b.y0), vec![0.0, 1.0, 0.0, 1.0]);
        s.execute("edit.transform.rotate90Ccw", json!({})).unwrap();
        assert_eq!(active(&s).surface().unwrap().content_bounds(), Rect::new(10, 10, 20, 14));
        assert_eq!(pixel(&s, 10, 10), vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn reselect_restores_the_last_selection() {
        let mut s = session(8);
        assert!(!s.is_enabled("select.reselect"));
        s.execute("select.rect", json!({"x": 2, "y": 3, "width": 5, "height": 6})).unwrap();
        s.execute("select.deselect", json!({})).unwrap();
        assert!(doc(&s).selection.is_none());
        s.execute("select.reselect", json!({})).unwrap();
        assert_eq!(doc(&s).selection.as_ref().unwrap().content_bounds(), Rect::new(2, 3, 7, 9));
        assert!(!s.is_enabled("select.reselect"), "nothing to reselect while a selection exists");
    }

    #[test]
    fn equalize_spreads_values() {
        for depth in [8, 16] {
            let mut s = session(depth);
            paint_square(&mut s, Rect::new(0, 0, 40, 15), [0.4, 0.4, 0.4, 1.0]);
            paint_square(&mut s, Rect::new(0, 15, 40, 30), [0.5, 0.5, 0.5, 1.0]);
            s.execute("image.adjustments.equalize", json!({})).unwrap();
            let (dark, light) = (pixel(&s, 1, 1)[0], pixel(&s, 1, 20)[0]);
            assert!(dark < 0.05 && light > 0.95, "{depth}: {dark} {light}");
        }
    }

    #[test]
    fn average_and_clouds() {
        let mut s = session(8);
        paint_square(&mut s, Rect::new(0, 0, 20, 30), [1.0, 0.0, 0.0, 1.0]);
        paint_square(&mut s, Rect::new(20, 0, 40, 30), [0.0, 0.0, 1.0, 1.0]);
        s.execute("filter.blur.average", json!({})).unwrap();
        let p = pixel(&s, 5, 5);
        assert!((p[0] - 0.5).abs() < 0.01 && (p[2] - 0.5).abs() < 0.01, "{p:?}");
        s.execute("filter.render.clouds", json!({"seed": 3})).unwrap();
        let vals: Vec<f32> = (0..40).map(|x| pixel(&s, x, 12)[0]).collect();
        let (lo, hi) = vals.iter().fold((1.0f32, 0.0f32), |(a, b), v| (a.min(*v), b.max(*v)));
        assert!(hi - lo > 0.05, "clouds vary: {lo}..{hi}");
        assert!(vals.windows(2).all(|w| (w[0] - w[1]).abs() < 0.2), "clouds are smooth");
        // Difference Clouds over itself with the same seed cancels to black.
        s.execute("filter.render.differenceClouds", json!({"seed": 3})).unwrap();
        assert!(pixel(&s, 7, 7)[..3].iter().all(|v| *v < 0.01), "{:?}", pixel(&s, 7, 7));
    }

    #[test]
    fn paste_into_masks_to_the_selection() {
        let mut s = session(8);
        paint_square(&mut s, Rect::new(0, 0, 10, 10), [1.0, 0.0, 0.0, 1.0]);
        s.execute("select.all", json!({})).unwrap();
        s.execute("edit.copy", json!({})).unwrap();
        let steps = s.active().unwrap().history.past_len();
        s.execute("select.rect", json!({"x": 20, "y": 10, "width": 8, "height": 8})).unwrap();
        s.execute("edit.pasteSpecial.pasteInto", json!({})).unwrap();
        let l = active(&s);
        let m = l.mask.as_ref().unwrap();
        assert!(!m.linked);
        assert_eq!(m.surface.content_bounds(), Rect::new(20, 10, 28, 18));
        assert_eq!(m.value(0, 0), 0.0);
        assert!(doc(&s).selection.is_none());
        assert_eq!(s.active().unwrap().history.past_len(), steps + 2, "select + one Paste Into step");
        // Paste Outside reveals everything but the selection.
        s.execute("select.rect", json!({"x": 20, "y": 10, "width": 8, "height": 8})).unwrap();
        s.execute("edit.pasteSpecial.pasteOutside", json!({})).unwrap();
        let m = active(&s).mask.as_ref().unwrap();
        assert_eq!((m.value(0, 0), m.value(22, 12)), (1.0, 0.0));
    }

    #[test]
    fn option_click_eye_shows_one_layer_then_restores() {
        let mut s = session(8);
        s.execute("layer.new.layer", json!({})).unwrap();
        let ids: Vec<LayerId> = doc(&s).layers.iter().map(|l| l.id).collect();
        // Background, Layer 1 (hidden beforehand), Layer 2.
        s.execute("layer.hideLayers", json!({"layer": ids[1].0})).unwrap();
        let vis = |s: &Session| doc(s).layers.iter().map(|l| l.visible).collect::<Vec<_>>();
        assert_eq!(s.execute("layer.showOnly", json!({"layer": ids[1].0})).unwrap()["shownAlone"], true);
        assert_eq!(vis(&s), [false, true, false]);
        // Another eye moves the solo; the original snapshot survives.
        s.execute("layer.showOnly", json!({"layer": ids[2].0})).unwrap();
        assert_eq!(vis(&s), [false, false, true]);
        assert_eq!(s.execute("layer.showOnly", json!({"layer": ids[2].0})).unwrap()["shownAlone"], false);
        assert_eq!(vis(&s), [true, false, true], "restored, with Layer 1 still hidden");
        // One history step each way.
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(vis(&s), [false, false, true]);
    }

    #[test]
    fn show_only_keeps_groups_and_their_contents() {
        let mut s = session(8);
        s.execute("layer.new.layer", json!({})).unwrap();
        let inner = active(&s).id;
        let group = s.execute("layer.groupLayers", json!({"layer": inner.0})).unwrap()["layer"].as_u64().map(LayerId).unwrap();
        let shown = |s: &Session| doc(s).walk().iter().map(|(_, _, l)| l.visible).collect::<Vec<_>>();
        // Background, Layer 1, Group 1, Layer 2 (inside the group).
        s.execute("layer.showOnly", json!({"layer": inner.0})).unwrap();
        assert_eq!(shown(&s), [false, false, true, true], "the enclosing group stays visible");
        // An undo ends the solo, so the next ⌥-click solos again from the current state.
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(s.execute("layer.showOnly", json!({"layer": inner.0})).unwrap()["shownAlone"], true);
        s.execute("edit.undo", json!({})).unwrap();
        // Showing a group alone leaves its contents as they are.
        s.execute("layer.setProps", json!({"layer": inner.0, "visible": false})).unwrap();
        s.execute("layer.showOnly", json!({"layer": group.0})).unwrap();
        assert_eq!(shown(&s), [false, false, true, false]);
        s.execute("layer.showOnly", json!({"layer": group.0})).unwrap();
        assert_eq!(shown(&s), [true, true, true, false]);
    }

    #[test]
    fn show_only_without_a_snapshot_shows_every_layer() {
        let mut s = session(16);
        let bg = doc(&s).layers[0].id;
        s.execute("layer.hideLayers", json!({"layer": bg.0})).unwrap();
        // Layer 1 is already the only visible layer: ⌥-clicking it shows everything.
        assert_eq!(s.execute("layer.showOnly", json!({})).unwrap()["shownAlone"], false);
        assert!(doc(&s).layers.iter().all(|l| l.visible));
        assert!(s.execute("layer.showOnly", json!({"layer": u64::MAX})).is_err());
        assert!(Session::new().execute("layer.showOnly", json!({})).is_err());
    }

    #[test]
    fn a_mask_turns_the_background_into_a_layer() {
        for (cmd, needs_selection) in [
            ("layer.layerMask.revealAll", false),
            ("layer.layerMask.hideAll", false),
            ("layer.layerMask.revealSelection", true),
            ("layer.layerMask.hideSelection", true),
        ] {
            let mut s = session(8);
            let bg = doc(&s).layers[0].id;
            if needs_selection {
                s.execute("select.rect", json!({"x": 2, "y": 2, "width": 10, "height": 10})).unwrap();
            }
            s.execute(cmd, json!({"layer": bg.0})).unwrap();
            let l = &doc(&s).layers[0];
            assert!(l.mask.is_some() && l.name == "Layer 0" && !l.locks.transparency && !l.locks.position, "{cmd}");
            assert!(!s.is_enabled("layer.new.layerFromBackground"), "{cmd}");
            // One step: undo brings the locked Background back, without a mask.
            assert!(s.undo());
            let l = &doc(&s).layers[0];
            assert!(l.mask.is_none() && l.name == "Background" && l.locks.transparency, "{cmd}");
        }
        // Other layers keep their name and locks.
        let mut s = session(8);
        let top = doc(&s).layers[1].id;
        s.execute("layer.setProps", json!({"layer": top.0, "locks": {"position": true}})).unwrap();
        s.execute("layer.layerMask.revealAll", json!({"layer": top.0})).unwrap();
        let l = doc(&s).layer(top).unwrap();
        assert!(l.mask.is_some() && l.name != "Layer 0" && l.locks.position);
        assert_eq!(doc(&s).layers[0].name, "Background");
    }

    #[test]
    fn hide_layers_toggles_once_every_target_is_hidden() {
        let mut s = session(8);
        s.execute("layer.new.layer", json!({})).unwrap();
        let b = active(&s).id;
        let visible = |s: &Session| doc(s).layer(b).unwrap().visible;
        // Select › All Layers leaves out the Background, as in Photoshop.
        let all = |s: &Session| crate::layer_multi_cmds::selected(s).iter().map(|id| doc(s).layer(*id).unwrap().visible).collect::<Vec<_>>();
        s.execute("layer.hideLayers", json!({})).unwrap();
        assert!(!visible(&s));
        s.execute("layer.hideLayers", json!({})).unwrap();
        assert!(visible(&s), "⌘, again shows it");
        // Several selected: any one still shown means hide them all; then they all come back.
        s.execute("layer.hideLayers", json!({"layer": b.0})).unwrap();
        s.execute("select.allLayers", json!({})).unwrap();
        assert!(all(&s).len() > 1);
        s.execute("layer.hideLayers", json!({})).unwrap();
        assert!(all(&s).iter().all(|v| !v));
        s.execute("layer.hideLayers", json!({})).unwrap();
        assert!(all(&s).iter().all(|v| *v));
    }

    #[test]
    fn slash_toggles_lock_transparent_pixels_but_not_on_the_background() {
        let mut s = session(8);
        let bg = doc(&s).layers[0].id;
        assert!(doc(&s).is_background(bg));
        s.execute("layer.new.layer", json!({})).unwrap();
        let top = active(&s).id;
        let locked = |s: &Session, id: LayerId| doc(s).layer(id).unwrap().locks.transparency;
        assert_eq!(s.execute("layer.toggleLockTransparency", json!({})).unwrap()["locked"], true);
        assert!(locked(&s, top));
        s.execute("layer.toggleLockTransparency", json!({})).unwrap();
        assert!(!locked(&s, top));
        s.undo();
        assert!(locked(&s, top), "one undoable step each");
        let bg_locks = doc(&s).layer(bg).unwrap().locks;
        assert!(s.execute("layer.toggleLockTransparency", json!({"layer": bg.0})).is_err());
        assert_eq!(doc(&s).layer(bg).unwrap().locks, bg_locks, "the Background is left alone");
        assert!(s.execute("layer.toggleLockTransparency", json!({"layer": u64::MAX})).is_err());
    }

    #[test]
    fn layer_utilities() {
        let mut s = session(8);
        // Layer from Background.
        assert!(s.is_enabled("layer.new.layerFromBackground"));
        s.execute("layer.new.layerFromBackground", json!({})).unwrap();
        assert_eq!(doc(&s).layers[0].name, "Layer 0");
        assert!(!doc(&s).layers[0].locks.transparency);
        assert!(!s.is_enabled("layer.new.layerFromBackground"));
        // Copy / paste style and hide all effects.
        s.execute("layer.layerStyle.dropShadow", json!({})).unwrap();
        s.execute("layer.layerStyle.copyLayerStyle", json!({})).unwrap();
        let top = doc(&s).layers[1].id;
        s.execute("layer.layerStyle.pasteLayerStyle", json!({"layer": top.0})).unwrap();
        assert_eq!(doc(&s).layers[1].effects.items.len(), 1);
        s.execute("layer.layerStyle.hideAllEffects", json!({})).unwrap();
        assert!(doc(&s).walk().iter().all(|(_, _, l)| l.effects.items.is_empty() || !l.effects.enabled));
        // Mask toggles.
        s.execute("layer.layerMask.revealAll", json!({})).unwrap();
        s.execute("layer.layerMask.enabled", json!({})).unwrap();
        assert!(!active(&s).mask.as_ref().unwrap().enabled);
        s.execute("layer.layerMask.linked", json!({"linked": false})).unwrap();
        assert!(!active(&s).mask.as_ref().unwrap().linked);
        // Hide, delete hidden, delete empty.
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("layer.hideLayers", json!({})).unwrap();
        assert!(!active(&s).visible);
        assert_eq!(s.execute("layer.delete.hiddenLayers", json!({})).unwrap()["deleted"], 1);
        s.execute("layer.new.layer", json!({})).unwrap();
        let n = doc(&s).layer_count();
        assert_eq!(s.execute("file.scripts.deleteAllEmptyLayers", json!({})).unwrap()["deleted"], 1);
        assert_eq!(doc(&s).layer_count(), n - 1);
    }

    #[test]
    fn ungroup_restores_children_in_order() {
        let mut s = session(8);
        let a = active(&s).id;
        s.execute("layer.groupLayers", json!({})).unwrap();
        assert!(active(&s).is_group());
        s.execute("layer.ungroupLayers", json!({})).unwrap();
        assert_eq!(s.active().unwrap().active_layer, Some(a));
        assert!(doc(&s).layers.iter().all(|l| !l.is_group()));
        assert_eq!(doc(&s).layers.len(), 2);
    }

    #[test]
    fn reveal_all_grows_the_canvas() {
        let mut s = session(8);
        s.execute("layer.translate", json!({"dx": 0, "dy": 0})).ok();
        paint_square(&mut s, Rect::new(-5, 0, 10, 10), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(reveal_all_bounds(doc(&s)), Rect::new(-5, 0, 40, 30));
        s.execute("image.revealAll", json!({})).unwrap();
        assert_eq!((doc(&s).size.width, doc(&s).size.height), (45, 30));
        assert_eq!(s.execute("image.revealAll", json!({})).unwrap()["changed"], false);
    }

    #[test]
    fn rasterize_fill_and_all_in_one_step() {
        let mut s = session(8);
        s.execute("layer.newFillLayer.solidColor", json!({"color": "#00ff00"})).unwrap();
        s.execute("layer.newFillLayer.solidColor", json!({"color": "#0000ff"})).unwrap();
        s.execute("layer.rasterize.fillContent", json!({})).unwrap();
        assert!(matches!(active(&s).content, LayerContent::Raster(_)));
        assert_eq!(pixel(&s, 3, 3), vec![0.0, 0.0, 1.0, 1.0]);
        assert!(s.execute("layer.rasterize.type", json!({})).is_err());
        let steps = s.active().unwrap().history.past_len();
        assert_eq!(s.execute("layer.rasterize.allLayers", json!({})).unwrap()["rasterized"], 1);
        assert_eq!(s.active().unwrap().history.past_len(), steps + 1);
        assert!(doc(&s).walk().iter().all(|(_, _, l)| !matches!(l.content, LayerContent::Fill(_))));
    }

    #[test]
    fn rasterize_smart_object_layer() {
        let mut s = session(8);
        s.execute("edit.fill", json!({"contents": "color", "color": "#ff0000"})).unwrap();
        let px_before = pixel(&s, 5, 5);
        s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
        assert!(matches!(active(&s).content, LayerContent::Smart(_)));
        s.execute("layer.rasterize.smartObject", json!({})).unwrap();
        assert!(matches!(active(&s).content, LayerContent::Raster(_)));
        let px_after = pixel(&s, 5, 5);
        assert_eq!(px_after, px_before);

        // Case 2: New layer above background
        let mut s2 = session(8);
        s2.execute("layer.new.layer", json!({})).unwrap();
        s2.execute("select.rect", json!({"x": 10, "y": 10, "width": 20, "height": 20})).unwrap();
        s2.execute("edit.fill", json!({"contents": "color", "color": "#00ff00"})).unwrap();
        s2.execute("select.deselect", json!({})).unwrap();
        let px_mid = pixel(&s2, 20, 20);
        assert_eq!(px_mid[1], 1.0); // green
        s2.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
        assert!(matches!(active(&s2).content, LayerContent::Smart(_)));
        s2.execute("layer.rasterize.smartObject", json!({})).unwrap();
        assert!(matches!(active(&s2).content, LayerContent::Raster(_)));
        let px_after_rast = pixel(&s2, 20, 20);
        assert_eq!(px_after_rast, px_mid);

        // Case 3: Shape layer converted to smart object then rasterized
        let mut s3 = session(8);
        s3.execute("shape.create", json!({"kind": "rect", "rect": [10, 10, 40, 40], "fill": "#0000ff"})).unwrap();
        let px_shape = pixel(&s3, 20, 20);
        s3.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
        assert!(matches!(active(&s3).content, LayerContent::Smart(_)));
        s3.execute("layer.rasterize.layer", json!({})).unwrap();
        assert!(matches!(active(&s3).content, LayerContent::Raster(_)));
        let px_after_shape = pixel(&s3, 20, 20);
        assert_eq!(px_after_shape, px_shape);

        // Case 4: Transformed smart object
        let mut s4 = session(8);
        s4.execute("layer.new.layer", json!({})).unwrap();
        s4.execute("select.rect", json!({"x": 10, "y": 10, "width": 20, "height": 20})).unwrap();
        s4.execute("edit.fill", json!({"contents": "color", "color": "#ffff00"})).unwrap();
        s4.execute("select.deselect", json!({})).unwrap();
        s4.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
        // Move smart object by 10, 10
        s4.execute("edit.transform", json!({"matrix": [1, 0, 0, 1, 10, 10]})).unwrap();
        s4.execute("layer.rasterize.smartObject", json!({})).unwrap();
        assert!(matches!(active(&s4).content, LayerContent::Raster(_)));
        assert_eq!(pixel(&s4, 30, 30), vec![1.0, 1.0, 0.0, 1.0]);
    }
}
