//! Image menu remnants: Image Rotation › Arbitrary, and the palette / ink modes — Mode › Indexed
//! Color, Color Table, Bitmap and Duotone.
//!
//! Indexed, Bitmap and Duotone documents keep expanded pixels (RGB 8-bit / Gray 8-bit / Gray, see
//! `Document::pixel_format`): Indexed and Bitmap quantise the flattened image in place and record
//! the palette (`Document::color_table`) or the 1-bit look; Duotone keeps the gray pixels and the
//! inks (`Document::duotone`), which display and flat export render through. Like Photoshop the
//! conversions flatten (Indexed, Bitmap) and require a grayscale source (Bitmap, Duotone).

use photocraft_algo::quantize::{self, BitmapMethod, Dither, Forced, HalftoneShape, PaletteKind};
use photocraft_algo::transform::{Homography, Interp};
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::adjust::CurvePoint;
use photocraft_doc::{ColorTable, Document, Duotone, DuotoneInk, Layer, LayerContent, LayerId, Size};
use photocraft_geom::Affine;
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn num(p: &Value, key: &str, default: f32) -> f32 {
    p.get(key).and_then(Value::as_f64).map_or(default, |v| v as f32)
}

/// A finite number param, `default` when absent or null; anything else is a bad-params error.
fn finite_num(p: &Value, cmd: &str, key: &str, default: f64) -> Result<f64> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v.as_f64().filter(|f| f.is_finite()).ok_or_else(|| bad(cmd, format!("`{key}` must be a number"))),
    }
}

fn str_or<'a>(p: &'a Value, key: &str, default: &'a str) -> &'a str {
    p.get(key).and_then(Value::as_str).unwrap_or(default)
}

type Enabled = std::result::Result<(), String>;

fn has_doc(s: &Session) -> Enabled {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

fn mode_is(s: &Session, modes: &[ColorMode], what: &str) -> Enabled {
    let d = s.active().ok_or("no document open")?;
    if modes.contains(&d.doc.mode) { Ok(()) } else { Err(format!("{what} (the document is {:?})", d.doc.mode)) }
}

// ---------- Image Rotation › Arbitrary ----------

/// Canvas size after rotating `size` by `deg` (the canvas grows to fit, as in Photoshop).
pub fn rotated_size(size: Size, deg: f64) -> Size {
    let (s, c) = deg.to_radians().sin_cos();
    let (w, h) = (f64::from(size.width), f64::from(size.height));
    let nw = (w * c.abs() + h * s.abs() - 1e-6).ceil().max(1.0);
    let nh = (w * s.abs() + h * c.abs() - 1e-6).ceil().max(1.0);
    Size::new(nw as u32, nh as u32)
}

pub(crate) fn rotate_arbitrary(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "image.rotation.arbitrary";
    let angle = f64::from(num(p, "angle", 0.0));
    if !angle.is_finite() || angle.abs() > 3600.0 {
        return Err(bad(CMD, "angle out of range"));
    }
    let deg = match str_or(p, "direction", "cw") {
        "cw" => angle,
        "ccw" => -angle,
        d => return Err(bad(CMD, format!("direction `{d}` (cw|ccw)"))),
    };
    if deg.rem_euclid(360.0).abs() < 1e-9 {
        return Ok(json!({"width": s.active().map(|d| d.doc.size.width), "height": s.active().map(|d| d.doc.size.height)}));
    }
    let bg = s.tools.background;
    // Color Sampler points turn with the canvas (the same map the edit applies to the geometry).
    {
        let old = s.active().ok_or(EngineError::NoDocument)?.doc.size;
        let new = rotated_size(old, deg);
        let (sn, cs) = deg.to_radians().sin_cos();
        let (cx, cy) = (f64::from(old.width) / 2.0, f64::from(old.height) / 2.0);
        let (nx, ny) = (f64::from(new.width) / 2.0, f64::from(new.height) / 2.0);
        s.set_sampler_map(Affine { m: [cs, sn, -sn, cs, nx - (cs * cx - sn * cy), ny - (sn * cx + cs * cy)] });
    }
    let size = s.edit("Rotate Canvas", |doc, _| {
        let old = doc.size;
        let new = rotated_size(old, deg);
        let (sn, cs) = deg.to_radians().sin_cos();
        let (cx, cy) = (f64::from(old.width) / 2.0, f64::from(old.height) / 2.0);
        let (nx, ny) = (f64::from(new.width) / 2.0, f64::from(new.height) / 2.0);
        // Clockwise on screen (y down): x' = c·x − s·y, y' = s·x + c·y about the centres.
        let a = Affine { m: [cs, sn, -sn, cs, nx - (cs * cx - sn * cy), ny - (sn * cx + cs * cy)] };
        let h = Homography([a.m[0], a.m[2], a.m[4], a.m[1], a.m[3], a.m[5], 0.0, 0.0, 1.0]);
        let interp = Interp::parse(str_or(p, "interpolation", "bicubic"));
        doc.size = new;
        let canvas = doc.bounds();
        let fmt = doc.pixel_format();
        let dpi = doc.resolution_dpi;
        let background_id = doc.background().map(|l| l.id);
        for l in doc.layers.iter_mut() {
            let background = Some(l.id) == background_id;
            if background && let Some(surf) = l.surface_mut() {
                // The Background stays a Background: rotated pixels over the background colour.
                let src = surf.content_bounds();
                let rotated = photocraft_algo::transform::warp_surface(surf, src, &h, interp)?;
                let mut base = Surface::new(fmt);
                let fill = photocraft_raster::from_rgba(&fmt, bg);
                base.write_region(canvas, &fill.repeat(canvas.width() as usize * canvas.height() as usize));
                crate::transform_cmds::composite_over(&mut base, &rotated);
                base.prune();
                *surf = base;
                if let Some(m) = l.mask.as_mut() {
                    m.surface = crate::transform_cmds::warp_gray(&m.surface, &h, interp)?;
                }
            } else {
                // Rotating the whole image moves locked layers too, including inside groups.
                with_released_movement_locks(l, |l| crate::transform_cmds::transform_layer(None, photocraft_doc::Locks::default(), l, &h, Some(a), interp))?;
            }
            // What Free Transform leaves alone: unlinked vector masks, gradient angles, artboards…
            crate::canvas_geom::transform_layer_geometry(l, &a, false, dpi);
        }
        crate::canvas_geom::transform_doc_marks(doc, &a);
        for ch in doc.channels.iter_mut().chain(doc.quick_mask.as_mut()) {
            ch.surface = crate::transform_cmds::warp_gray(&ch.surface, &h, interp)?;
        }
        if let Some(sel) = &doc.selection {
            doc.selection = Some(crate::transform_cmds::warp_gray(sel, &h, Interp::Bilinear)?).filter(|s| !s.content_bounds().is_empty());
        }
        // Type, shapes and smart objects re-render from their new geometry.
        crate::canvas_geom::refresh(doc, crate::canvas_geom::Refresh::All);
        doc.guides = Default::default();
        Ok(new)
    })?;
    Ok(json!({"width": size.width, "height": size.height}))
}

/// Runs `f` with movement locks (`position` and `all`) temporarily released on `l` and all its
/// descendants, so canvas-wide rotation moves every layer, then restores each lock.
fn with_released_movement_locks<R>(l: &mut Layer, f: impl FnOnce(&mut Layer) -> Result<R>) -> Result<R> {
    let mut saved = Vec::new();
    crate::image_cmds::for_each_layer(std::slice::from_mut(l), &mut |x| {
        saved.push(x.locks);
        x.locks.position = false;
        x.locks.all = false;
    });
    let r = f(l);
    let mut saved = saved.into_iter();
    crate::image_cmds::for_each_layer(std::slice::from_mut(l), &mut |x| {
        if let Some(locks) = saved.next() {
            x.locks = locks;
        }
    });
    r
}

// ---------- flatten helpers ----------

/// The flattened image as straight RGBA (over white when `opaque`).
fn composite(doc: &Document, opaque: bool) -> Vec<[f32; 4]> {
    let buf = photocraft_compose::flatten(doc);
    if opaque { buf.over_background([1.0, 1.0, 1.0]).px } else { buf.px }
}

/// Replace the document's layers by one 8-bit layer holding `px` in `mode`'s storage format.
fn single_layer(doc: &mut Document, active: &mut Option<LayerId>, mode: ColorMode, px: &[[f32; 4]], name: &str, background: bool) {
    doc.mode = mode;
    doc.depth = SampleType::U8;
    let fmt = doc.pixel_format();
    // Reuse one format-sized pixel buffer instead of allocating a Vec for every pixel.
    let mut pixel = vec![0.0; fmt.channels()];
    let mut data = Vec::new();
    for q in px {
        photocraft_raster::from_rgba_into(&fmt, *q, &mut pixel);
        data.extend_from_slice(&pixel);
    }
    let mut l = Layer::raster(name, fmt);
    if background {
        l.locks.transparency = true;
        l.locks.position = true;
    }
    // A new raster layer always has pixels.
    if let Some(surf) = l.surface_mut() {
        surf.write_region(doc.bounds(), &data);
    }
    *active = Some(l.id);
    doc.layers = vec![l];
    for ch in doc.channels.iter_mut() {
        let f = PixelFormat::new(ColorMode::Grayscale, SampleType::U8, false);
        ch.surface = ch.surface.convert(PixelFormat { alpha: ch.surface.format().alpha, ..f });
    }
    doc.quick_mask = None;
    doc.color_table = None;
    doc.duotone = None;
}

// ---------- Indexed Color / Color Table ----------

fn parse_forced(s: &str) -> Option<Forced> {
    Some(match s {
        "none" => Forced::None,
        "blackWhite" | "blackAndWhite" => Forced::BlackWhite,
        "primaries" => Forced::Primaries,
        "web" => Forced::Web,
        _ => return None,
    })
}

fn parse_dither(s: &str) -> Option<Dither> {
    Some(match s {
        "none" => Dither::None,
        "diffusion" => Dither::Diffusion,
        "pattern" => Dither::Pattern,
        "noise" => Dither::Noise,
        _ => return None,
    })
}

fn indexed_color(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "image.mode.indexedColor";
    let kind = PaletteKind::from_id(str_or(p, "palette", "selective"))
        .ok_or_else(|| bad(CMD, "palette: exact|systemMac|systemWindows|web|uniform|perceptual|selective|adaptive"))?;
    let forced = parse_forced(str_or(p, "forced", "blackWhite")).ok_or_else(|| bad(CMD, "forced: none|blackWhite|primaries|web"))?;
    let dither = parse_dither(str_or(p, "dither", "diffusion")).ok_or_else(|| bad(CMD, "dither: none|diffusion|pattern|noise"))?;
    let colors = match p.get("colors") {
        None => 256,
        Some(value) => {
            let value = value.as_f64().ok_or_else(|| bad(CMD, "colors must be an integer from 2 to 256"))?;
            if !value.is_finite() || value.fract() != 0.0 || !(2.0..=256.0).contains(&value) {
                return Err(bad(CMD, "colors must be an integer from 2 to 256"));
            }
            value as usize
        }
    };
    let amount = num(p, "amount", 75.0).clamp(0.0, 100.0) / 100.0;
    let transparency = p.get("transparency").and_then(Value::as_bool).unwrap_or(true);
    let (pal_len, transparent) = s.edit("Indexed Color", |doc, active| {
        let mut px = composite(doc, !transparency);
        let has_clear = transparency && px.iter().any(|q| q[3] < 0.5);
        let mut pal = quantize::build_palette(&px, kind, if has_clear { colors.saturating_sub(1).max(2) } else { colors }, forced).map_err(|e| bad(CMD, e))?;
        let transparent = if has_clear && pal.len() < 256 {
            pal.push([255, 255, 255]);
            Some(pal.len() - 1)
        } else {
            None
        };
        let w = doc.size.width as usize;
        quantize::quantize(&mut px, w, &pal, dither, amount, transparent);
        single_layer(doc, active, ColorMode::Indexed, &px, if transparent.is_some() { "Index" } else { "Background" }, transparent.is_none());
        doc.color_table = Some(ColorTable { colors: pal.clone(), transparent: transparent.map(|t| t as u8) });
        Ok((pal.len(), transparent))
    })?;
    Ok(json!({"colors": pal_len, "transparentIndex": transparent}))
}

fn hex(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

fn parse_color(v: &Value) -> Option<[u8; 3]> {
    match v {
        Value::String(s) => {
            let s = s.trim_start_matches('#');
            let b = |i: usize| u8::from_str_radix(s.get(i..i + 2)?, 16).ok();
            Some([b(0)?, b(2)?, b(4)?])
        }
        Value::Array(a) if a.len() >= 3 => {
            let c = |i: usize| a[i].as_f64().map(|x| x.round().clamp(0.0, 255.0) as u8);
            Some([c(0)?, c(1)?, c(2)?])
        }
        _ => None,
    }
}

/// Built-in Color Table presets.
pub fn preset_table(name: &str) -> Option<Vec<[u8; 3]>> {
    let ramp = |f: &dyn Fn(f32) -> [f32; 3]| (0..256).map(|i| f(i as f32 / 255.0).map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)).collect();
    Some(match name {
        "grayscale" => ramp(&|t| [t; 3]),
        // Black → red → yellow → white, the incandescence scale.
        "blackBody" => ramp(&|t| [(t * 3.0).min(1.0), (t * 3.0 - 1.0).clamp(0.0, 1.0), (t * 3.0 - 2.0).clamp(0.0, 1.0)]),
        "spectrum" => ramp(&|t| {
            let h = t * 300.0 / 60.0;
            let x = 1.0 - (h % 2.0 - 1.0).abs();
            match h as i32 {
                0 => [1.0, x, 0.0],
                1 => [x, 1.0, 0.0],
                2 => [0.0, 1.0, x],
                3 => [0.0, x, 1.0],
                _ => [x, 0.0, 1.0],
            }
        }),
        "systemMac" | "system" => quantize::mac_palette(),
        "systemWindows" => quantize::windows_palette(),
        "web" => quantize::web_palette(),
        _ => return None,
    })
}

/// Image › Mode › Color Table: replace entries (pixels keep their index, so their colour follows
/// the table, as in Photoshop) and set the transparent entry. No changes: report the table.
fn color_table(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "image.mode.colorTable";
    let cur = s.active().ok_or(EngineError::NoDocument)?.doc.color_table.clone().ok_or_else(|| EngineError::Other("the document has no color table".into()))?;
    let mut table = cur.clone();
    match str_or(p, "table", "custom") {
        "custom" => {}
        name => table.colors = preset_table(name).ok_or_else(|| bad(CMD, format!("unknown table `{name}`")))?,
    }
    if let Some(a) = p.get("colors").and_then(Value::as_array) {
        let c: Option<Vec<[u8; 3]>> = a.iter().map(parse_color).collect();
        let c = c.ok_or_else(|| bad(CMD, "colors: [\"#rrggbb\" | [r, g, b], …]"))?;
        if c.is_empty() || c.len() > 256 {
            return Err(bad(CMD, "a color table has 1 to 256 entries"));
        }
        table.colors = c;
    }
    if let Some(e) = p.get("entries").and_then(Value::as_object) {
        for (k, v) in e {
            let i: usize = k.parse().map_err(|_| bad(CMD, format!("entry index `{k}`")))?;
            let c = parse_color(v).ok_or_else(|| bad(CMD, format!("entry {i}: bad colour")))?;
            if i >= table.colors.len() {
                return Err(bad(CMD, format!("entry {i} is past the table ({} entries)", table.colors.len())));
            }
            table.colors[i] = c;
        }
    }
    match p.get("transparent") {
        Some(Value::Null) => table.transparent = None,
        Some(v) => {
            let len = table.colors.len();
            let i =
                v.as_u64().filter(|&i| i < len as u64).ok_or_else(|| bad(CMD, format!("transparent: an index from 0 to {} or null", len.saturating_sub(1))))?;
            table.transparent = u8::try_from(i).ok();
        }
        // A shrunk table can't keep a transparent entry past its end (#4186).
        None => table.transparent = table.transparent.filter(|&i| usize::from(i) < table.colors.len()),
    }
    let report = |t: &ColorTable| json!({"colors": t.colors.iter().map(|c| hex(*c)).collect::<Vec<_>>(), "transparent": t.transparent});
    if table == cur {
        return Ok(report(&cur));
    }
    let out = report(&table);
    s.edit("Color Table", |doc, _| {
        let old = cur.clone();
        for surf in doc.layers.iter_mut().filter_map(Layer::surface_mut) {
            let r = surf.content_bounds();
            if r.is_empty() {
                continue;
            }
            let fmt = surf.format();
            let n = fmt.channels();
            let mut data = surf.read_region(r);
            for q in data.chunks_exact_mut(n) {
                let c = photocraft_raster::to_rgba(&fmt, q);
                let rgb = [c[0], c[1], c[2]];
                let i = if c[3] <= 0.0 {
                    let Some(i) = old.transparent.map(usize::from) else { continue };
                    if i >= old.colors.len() || old.rgb(i) != rgb {
                        continue;
                    }
                    i
                } else {
                    old.nearest(rgb)
                };
                let alpha = if table.transparent == Some(i as u8) {
                    0.0
                } else if c[3] <= 0.0 {
                    1.0
                } else {
                    c[3]
                };
                let nc = if i < table.colors.len() { table.rgb(i) } else { table.rgb(table.nearest([c[0], c[1], c[2]])) };
                let mut enc = [0.0f32; 8];
                let m = photocraft_raster::from_rgba_into(&fmt, [nc[0], nc[1], nc[2], alpha], &mut enc);
                q.copy_from_slice(&enc[..m]);
            }
            surf.write_region(r, &data);
            surf.prune();
        }
        doc.color_table = Some(table.clone());
        Ok(())
    })?;
    Ok(out)
}

// ---------- Bitmap ----------

fn bitmap(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "image.mode.bitmap";
    let dpi = s.active().ok_or(EngineError::NoDocument)?.doc.resolution_dpi.max(1.0);
    let method = match str_or(p, "method", "diffusion") {
        "threshold" | "threshold50" => BitmapMethod::Threshold,
        "pattern" => BitmapMethod::Pattern,
        "diffusion" => BitmapMethod::Diffusion,
        "halftone" => {
            // Photoshop's Halftone Screen: frequency 1..999 lines/inch, angle -180..180 degrees.
            let freq = finite_num(p, CMD, "frequency", 53.0)?.clamp(1.0, 999.0) as f32;
            let angle = finite_num(p, CMD, "angle", 45.0)?;
            if !(-180.0..=180.0).contains(&angle) {
                return Err(bad(CMD, format!("angle {angle} is outside -180..180")));
            }
            let shape = HalftoneShape::from_id(str_or(p, "shape", "round")).ok_or_else(|| bad(CMD, "shape: round|ellipse|line|square|diamond|cross"))?;
            BitmapMethod::Halftone { cell: (dpi / freq).max(2.0), angle: angle as f32, shape }
        }
        m => {
            return Err(bad(CMD, format!("method `{m}` (threshold|pattern|diffusion|halftone)")));
        }
    };
    s.edit("Bitmap", |doc, active| {
        let px = composite(doc, true);
        let mut gray: Vec<f32> = px.iter().map(|q| photocraft_color::convert::rgb_to_gray([q[0], q[1], q[2]])).collect();
        quantize::to_bitmap(&mut gray, doc.size.width as usize, method);
        let out: Vec<[f32; 4]> = gray.iter().map(|g| [*g, *g, *g, 1.0]).collect();
        single_layer(doc, active, ColorMode::Bitmap, &out, "Background", true);
        Ok(())
    })?;
    Ok(Value::Null)
}

// ---------- Duotone ----------

/// Default inks per type: black first, then warm brown, gold and slate blue (generic names;
/// Photoshop ships Pantone presets, which we don't reproduce).
fn default_inks(n: usize) -> Vec<DuotoneInk> {
    let all = [("Black", [0.0, 0.0, 0.0]), ("Warm Brown", [0.62, 0.38, 0.18]), ("Gold", [0.86, 0.68, 0.18]), ("Slate Blue", [0.27, 0.38, 0.58])];
    all.iter().take(n).map(|(name, c)| DuotoneInk::new(*name, *c)).collect()
}

fn parse_inks(cmd: &str, v: &Value, n: usize) -> Result<Vec<DuotoneInk>> {
    let a = v.as_array().ok_or_else(|| bad(cmd, "inks: [{\"name\",\"color\":\"#rrggbb\",\"curve\":[[in,out],…] (0..100)}, …]"))?;
    let defaults = default_inks(n.max(a.len()).min(4));
    let mut out = Vec::new();
    for (i, ink) in a.iter().take(4).enumerate() {
        let mut d = defaults.get(i).cloned().unwrap_or_else(|| DuotoneInk::new(format!("Ink {}", i + 1), [0.0; 3]));
        if let Some(name) = ink.get("name").and_then(Value::as_str) {
            d.name = name.to_string();
        }
        if let Some(c) = ink.get("color").and_then(parse_color) {
            d.color = c.map(|v| f32::from(v) / 255.0);
        }
        if let Some(pts) = ink.get("curve").and_then(Value::as_array) {
            let mut curve: Vec<CurvePoint> =
                pts.iter().filter_map(|p| Some(CurvePoint { input: p.get(0)?.as_f64()? as f32 / 100.0, output: p.get(1)?.as_f64()? as f32 / 100.0 })).collect();
            curve.sort_by(|a, b| a.input.total_cmp(&b.input));
            if curve.len() >= 2 {
                d.curve = curve;
            }
        }
        out.push(d);
    }
    Ok(out)
}

fn duotone(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "image.mode.duotone";
    let n = match str_or(p, "type", "duotone") {
        "monotone" => 1,
        "duotone" => 2,
        "tritone" => 3,
        "quadtone" => 4,
        t => {
            return Err(bad(CMD, format!("type `{t}` (monotone|duotone|tritone|quadtone)")));
        }
    };
    let mut inks = match p.get("inks") {
        Some(v) => parse_inks(CMD, v, n)?,
        None => default_inks(n),
    };
    if inks.len() < n {
        inks.extend(default_inks(n).into_iter().skip(inks.len()));
    }
    inks.truncate(n);
    let names: Vec<String> = inks.iter().map(|i| i.name.clone()).collect();
    s.edit("Duotone", |doc, _| {
        doc.mode = ColorMode::Duotone;
        doc.duotone = Some(Duotone { inks, psd_raw: None });
        doc.color_table = None;
        Ok(())
    })?;
    Ok(json!({"inks": names}))
}

macro_rules! spec {
    ($id:literal, $label:literal, [$($m:literal),*], $params:literal, $en:expr, $run:expr) => {
        CommandSpec { id: $id, label: $label, menu: &[$($m),*], shortcut: None, params: $params, enabled: $en, run: $run, journal: true }
    };
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        spec!(
            "image.rotation.arbitrary",
            "Arbitrary…",
            ["Image", "Image Rotation"],
            r##"{"angle":-359.99..359.99=0,"direction":"cw|ccw"="cw","interpolation":"bicubic|bilinear|nearest"="bicubic"}"##,
            has_doc,
            rotate_arbitrary
        ),
        spec!(
            "image.mode.indexedColor",
            "Indexed Color…",
            ["Image", "Mode"],
            r##"{"palette":"selective|perceptual|adaptive|exact|systemMac|systemWindows|web|uniform"="selective","colors":2..256=256,"forced":"blackWhite|none|primaries|web"="blackWhite","transparency":bool=true,"dither":"diffusion|none|pattern|noise"="diffusion","amount":0..100=75} (flattens)"##,
            |s| mode_is(
                s,
                &[ColorMode::Rgb, ColorMode::Grayscale, ColorMode::Indexed, ColorMode::Duotone, ColorMode::Cmyk, ColorMode::Lab],
                "Indexed Color needs an RGB or Grayscale image"
            ),
            indexed_color
        ),
        spec!(
            "image.mode.colorTable",
            "Color Table…",
            ["Image", "Mode"],
            r##"{"table":"custom|blackBody|grayscale|spectrum|systemMac|systemWindows|web"="custom","colors":json,"entries":json,"transparent":json} (colors: ["#rrggbb", …]; entries: {"index": "#rrggbb"}; transparent: index|null)"##,
            |s| mode_is(s, &[ColorMode::Indexed], "Color Table needs an Indexed Color image"),
            color_table
        ),
        spec!(
            "image.mode.bitmap",
            "Bitmap…",
            ["Image", "Mode"],
            r##"{"method":"diffusion|threshold|pattern|halftone"="diffusion","frequency":1..999=53,"angle":-180..180=45,"shape":"round|ellipse|line|square|diamond|cross"="round"} (halftone frequency in lines/inch; flattens)"##,
            |s| mode_is(s, &[ColorMode::Grayscale], "Bitmap needs a Grayscale image (Image › Mode › Grayscale first)"),
            bitmap
        ),
        spec!(
            "image.mode.duotone",
            "Duotone…",
            ["Image", "Mode"],
            r##"{"type":"duotone|monotone|tritone|quadtone"="duotone","inks":json} (inks: [{"name","color":"#rrggbb","curve":[[in,out],…] in 0..100}, …])"##,
            |s| mode_is(s, &[ColorMode::Grayscale, ColorMode::Duotone], "Duotone needs a Grayscale image (Image › Mode › Grayscale first)"),
            duotone
        ),
    ]
}

/// The gradient-map layer that renders Duotone inks over the gray composite.
pub fn duotone_display_layer(d: &Duotone) -> Layer {
    Layer::new("Duotone", LayerContent::Adjustment(d.display_adjustment()))
}

/// A copy of `doc` that displays like the printed result (Duotone inks), or None when the
/// document displays as stored.
pub fn display_document(doc: &Document) -> Option<Document> {
    let d = doc.duotone.as_ref().filter(|_| doc.mode == ColorMode::Duotone)?;
    let mut out = doc.clone();
    out.layers.push(duotone_display_layer(d));
    Some(out)
}

#[cfg(test)]
mod tests;
