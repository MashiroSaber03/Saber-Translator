//! Image menu: Image Size, Canvas Size, Crop, Trim, Mode (colour model and
//! bit depth) conversions, Duplicate.

use photocraft_algo::resample::{Resample, crop_surface, resize_surface_in_canvas, translate_surface};
use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::{Document, Effect, Effects, FxPaint, Layer, LayerContent, LayerId, Size};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

/// Applies `f` to every pixel surface of a layer tree: content (raster or
/// cached text/shape/smart pixels), fill caches, and (if `masks`) masks.
pub(crate) fn for_each_surface(layers: &mut [Layer], masks: bool, f: &mut dyn FnMut(&mut Surface, bool)) {
    for l in layers {
        if masks && let Some(m) = &mut l.mask {
            f(&mut m.surface, true);
        }
        if let Some(fc) = &mut l.fill_cache {
            f(&mut fc.surface, false);
        }
        match &mut l.content {
            LayerContent::Raster(s) => f(s, false),
            LayerContent::Text(t) => {
                if let Some(c) = &mut t.cache {
                    f(c, false)
                }
            }
            LayerContent::Shape(sh) => {
                if let Some(c) = &mut sh.cache {
                    f(c, false)
                }
            }
            LayerContent::Smart(sm) => {
                if let Some(c) = &mut sm.cache {
                    f(c, false)
                }
                // Selection-derived smart-filter masks live in document coordinates. Every
                // canvas resize/translation must carry them with the smart object's cache;
                // otherwise toggling a filter recomputes its coverage at stale positions (#3077).
                if masks && let Some(mask) = &mut sm.filter_mask {
                    f(&mut mask.surface, true);
                }
            }
            LayerContent::Group(g) => for_each_surface(&mut g.children, masks, f),
            _ => {}
        }
    }
}

/// Converts every pixel surface of a layer tree (masks included) to `depth`.
pub(crate) fn convert_layers_depth(layers: &mut [Layer], depth: SampleType) {
    for_each_surface(layers, true, &mut |surf, _| {
        let f = surf.format().with_sample(depth);
        *surf = surf.convert(f);
    });
}

pub(crate) fn for_each_layer(layers: &mut [Layer], f: &mut dyn FnMut(&mut Layer)) {
    for l in layers {
        f(l);
        if let Some(ch) = l.children_mut() {
            for_each_layer(ch, f);
        }
    }
}

/// Assign new layer ids to a copied document and update every saved layer-comp reference.
///
/// A document copy needs fresh identities, but its comps must still target the copied
/// layers. Deliberately leave references to already-deleted layers untouched so their
/// missing-layer warnings remain meaningful. The same applies to Last Document State.
pub(crate) fn reidentify_copied_layers(doc: &mut Document) {
    let mut ids = std::collections::HashMap::new();
    for_each_layer(&mut doc.layers, &mut |layer| {
        let previous = layer.id;
        layer.id = LayerId::fresh();
        ids.insert(previous, layer.id);
    });

    let remap = |comp: &mut photocraft_doc::LayerComp| {
        for state in &mut comp.states {
            if let Some(&new_id) = ids.get(&state.layer) {
                state.layer = new_id;
            }
        }
    };
    for comp in &mut doc.layer_comps {
        remap(comp);
    }
    if let Some(last) = &mut doc.last_document_state {
        remap(last);
    }
}

/// Scale every size and distance of a style by `k` (Image Size's Scale Styles, Layer Style ›
/// Scale Effects, and previews that draw a style at a reduced resolution).
pub fn scale_effects(fx: &mut Effects, k: f32) {
    for e in &mut fx.items {
        match e {
            Effect::DropShadow(s) | Effect::InnerShadow(s) => {
                s.distance *= k;
                s.size *= k;
            }
            Effect::OuterGlow(g) | Effect::InnerGlow(g) => g.size *= k,
            Effect::Stroke(s) => s.size *= k,
            Effect::Satin(s) => {
                s.distance *= k;
                s.size *= k;
            }
            Effect::BevelEmboss(b) => {
                b.size *= k;
                b.soften *= k;
            }
            Effect::PatternOverlay { scale, .. } => *scale *= k,
            Effect::GradientOverlay { .. } | Effect::ColorOverlay { .. } => {}
        }
        if let Effect::Stroke(s) = e
            && let FxPaint::Pattern { scale, .. } = &mut s.paint
        {
            *scale *= k;
        }
    }
    // Effects changed: the preserved PSD block no longer applies.
    if k != 1.0 {
        fx.psd_raw = None;
    }
}

/// Image Size's resampling methods; `None` changes the resolution only.
#[derive(Clone, Copy)]
enum Method {
    None,
    Filter(Resample),
    /// Photoshop's Automatic: Preserve Details where the image grows, Bicubic Sharper otherwise.
    Automatic,
}

fn parse_resample(s: &str) -> Method {
    Method::Filter(match s {
        "none" => return Method::None,
        "automatic" => return Method::Automatic,
        "nearest" | "nearestNeighbor" => Resample::Nearest,
        "bilinear" => Resample::Bilinear,
        "bicubicSmoother" => Resample::BicubicSmoother,
        "bicubicSharper" => Resample::BicubicSharper,
        "bicubicAutomatic" => Resample::BicubicAutomatic,
        "lanczos" => Resample::Lanczos,
        "preserveDetails" => Resample::PreserveDetails,
        _ => Resample::Bicubic,
    })
}

/// The largest raster Image Size resamples to: the decoders' default allocation budget.
const MAX_RESAMPLE_BYTES: u64 = if cfg!(target_pointer_width = "64") { 8 << 30 } else { 2 << 30 };

/// Image → Image Size.
fn image_size(s: &mut Session, p: &Value) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let (ow, oh) = (d.doc.size.width, d.doc.size.height);
    let resample = parse_resample(p.get("resample").and_then(Value::as_str).unwrap_or("bicubic"));
    let mut pw = p.get("width").and_then(Value::as_f64);
    let mut ph = p.get("height").and_then(Value::as_f64);
    // `percent` scales both sides (what a Photoshop action records as "200%"); an explicit side wins.
    if let Some(pc) = p.get("percent") {
        let pc = pc.as_f64().filter(|v| v.is_finite() && *v > 0.0).ok_or_else(|| bad("image.imageSize", "`percent` must be a positive number"))?;
        pw = pw.or(Some(ow as f64 * pc / 100.0));
        ph = ph.or(Some(oh as f64 * pc / 100.0));
    }
    // `keepPrintSize`: only the resolution given, the pixels follow it (Photoshop's Image Size
    // with Resample on, as an action records it).
    if p.get("keepPrintSize").and_then(Value::as_bool) == Some(true)
        && pw.is_none()
        && ph.is_none()
        && let Some(r) = p.get("resolution").and_then(Value::as_f64)
    {
        let r = r.clamp(1.0, 10_000.0);
        let old = f64::from(d.doc.resolution_dpi).max(1.0);
        pw = Some(ow as f64 * r / old);
        ph = Some(oh as f64 * r / old);
    }
    // Missing dimension keeps the aspect ratio.
    let (nw, nh) = match (pw, ph) {
        (Some(w), Some(h)) => (w, h),
        (Some(w), None) => (w, w * oh as f64 / ow.max(1) as f64),
        (None, Some(h)) => (h * ow as f64 / oh.max(1) as f64, h),
        (None, None) => (ow as f64, oh as f64),
    };
    let (nw, nh) = (nw.round().clamp(1.0, 300_000.0) as u32, nh.round().clamp(1.0, 300_000.0) as u32);
    // Each side may reach 300000 px, but resampling writes real pixels: refuse a canvas whose
    // raster alone would pass the budget, before allocating any of it (#1544).
    if !matches!(resample, Method::None) && (nw, nh) != (ow, oh) {
        check_resample_budget("image.imageSize", &d.doc, nw, nh, ", or turn Resample off to change only the resolution")?;
    }
    let dpi = p.get("resolution").and_then(Value::as_f64).map(|v| v as f32);
    if !matches!(resample, Method::None) && (nw, nh) != (ow, oh) {
        // Color Sampler points scale with the pixels (same map `transform_geometry` applies).
        let (sx, sy) = (f64::from(nw) / f64::from(ow.max(1)), f64::from(nh) / f64::from(oh.max(1)));
        s.set_sampler_map(photocraft_geom::Affine { m: [sx, 0.0, 0.0, sy, 0.0, 0.0] });
    }
    s.edit("Image Size", |doc, _| {
        if let Some(r) = dpi {
            doc.resolution_dpi = r.clamp(1.0, 10_000.0);
        }
        let filter = match resample {
            Method::None => return Ok(()),
            Method::Filter(f) => f,
            Method::Automatic if nw > ow || nh > oh => Resample::PreserveDetails,
            Method::Automatic => Resample::BicubicSharper,
        };
        resample_doc(doc, nw, nh, filter);
        Ok(())
    })?;
    // The size the document has now: without resampling `width`/`height` are ignored (#1815).
    let size = s.active().ok_or(EngineError::NoDocument)?.doc.size;
    Ok(json!({ "width": size.width, "height": size.height }))
}

/// Refuses to resample `doc` to `nw` x `nh` px when the raster alone would pass the budget, before
/// allocating any of it (#1544). `hint` ends the message with what the caller can do instead.
pub(crate) fn check_resample_budget(cmd: &str, doc: &Document, nw: u32, nh: u32, hint: &str) -> Result<()> {
    let bpp = doc.pixel_format().bytes_per_pixel().max(1) as u64;
    let bytes = u64::from(nw).saturating_mul(u64::from(nh)).saturating_mul(bpp);
    if bytes <= MAX_RESAMPLE_BYTES {
        return Ok(());
    }
    let mp = u64::from(nw).saturating_mul(u64::from(nh)) / 1_000_000;
    let max_mp = MAX_RESAMPLE_BYTES / bpp / 1_000_000;
    Err(bad(cmd, format!("{nw} x {nh} px is {mp} megapixels; resampling at this bit depth is limited to {max_mp} megapixels. Choose a smaller size{hint}")))
}

/// Resamples the whole document (pixels, masks, channels, selection, effects, vectors, guides) to
/// `nw` x `nh` px with `filter`: the body of Image Size, shared by the Crop tool's W x H x Resolution
/// mode. Callers check [`check_resample_budget`] first.
fn resample_doc(doc: &mut Document, nw: u32, nh: u32, filter: Resample) {
    let (ow, oh) = (doc.size.width, doc.size.height);
    let (sx, sy) = (nw as f64 / ow.max(1) as f64, nh as f64 / oh.max(1) as f64);
    if (sx - 1.0).abs() < 1e-12 && (sy - 1.0).abs() < 1e-12 {
        return;
    }
    // Content that reaches the canvas edge keeps it: a Background stays opaque to the border.
    let canvas = Rect::from_xywh(0, 0, ow, oh);
    for_each_surface(&mut doc.layers, true, &mut |surf, is_mask| {
        *surf = resize_surface_in_canvas(surf, sx, sy, if is_mask { Resample::Bilinear } else { filter }, canvas);
    });
    let k = ((sx + sy) / 2.0) as f32;
    for_each_layer(&mut doc.layers, &mut |l| scale_effects(&mut l.effects, k));
    for ch in doc.channels.iter_mut().chain(doc.quick_mask.as_mut()) {
        ch.surface = resize_surface_in_canvas(&ch.surface, sx, sy, Resample::Bilinear, canvas);
    }
    if let Some(sel) = &doc.selection {
        doc.selection = Some(resize_surface_in_canvas(sel, sx, sy, Resample::Bilinear, canvas));
    }
    // Vector geometry, guides and marks scale with the pixels; vectors re-render sharp.
    crate::canvas_geom::transform_geometry(doc, &photocraft_geom::Affine { m: [sx, 0.0, 0.0, sy, 0.0, 0.0] });
    doc.size = Size::new(nw, nh);
    crate::canvas_geom::refresh(doc, crate::canvas_geom::Refresh::All);
}

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

/// Moves every surface, channel, selection and guide by `(dx, dy)`.
///
/// Fails, before moving anything, when some pixels would land outside the i32 coordinate
/// range: the surface copy saturates its target rectangle there, which no longer matches
/// the pixel data and panics (#959).
pub(crate) fn translate_doc(doc: &mut Document, cmd: &str, dx: i32, dy: i32) -> Result<()> {
    if dx == 0 && dy == 0 {
        return Ok(());
    }
    let fits = |r: Rect| {
        r.is_empty() || (r.x0.checked_add(dx).is_some() && r.x1.checked_add(dx).is_some() && r.y0.checked_add(dy).is_some() && r.y1.checked_add(dy).is_some())
    };
    let mut ok = true;
    for_each_surface(&mut doc.layers, true, &mut |surf, _| ok = ok && fits(surf.content_bounds()));
    ok = ok
        && doc.channels.iter().chain(doc.quick_mask.as_ref()).all(|ch| fits(ch.surface.content_bounds()))
        && doc.selection.as_ref().is_none_or(|sel| fits(sel.content_bounds()));
    if !ok {
        return Err(bad(cmd, format!("moving the document by ({dx}, {dy}) would push its pixels outside the 32-bit coordinate range")));
    }
    for_each_surface(&mut doc.layers, true, &mut |surf, _| *surf = translate_surface(surf, dx, dy));
    for ch in doc.channels.iter_mut().chain(doc.quick_mask.as_mut()) {
        ch.surface = translate_surface(&ch.surface, dx, dy);
    }
    if let Some(sel) = &doc.selection {
        doc.selection = Some(translate_surface(sel, dx, dy));
    }
    // Type, shapes, smart objects, vector masks, paths, guides, slices, notes… (caches above
    // are already translated exactly, so nothing needs re-rendering for the move itself).
    crate::canvas_geom::transform_geometry(doc, &photocraft_geom::Affine::translate(f64::from(dx), f64::from(dy)));
    Ok(())
}

/// Crops the document to `r` (in current document coordinates).
/// A frame that extends past the canvas fills the Background's new area with `extension` (the
/// background colour, as Photoshop's Crop tool does); other layers get transparent pixels there.
fn crop_doc(doc: &mut Document, cmd: &str, r: Rect, delete_pixels: bool, extension: [f32; 4]) -> Result<()> {
    // The origin moves to (0, 0); `-i32::MIN` has no i32 value (#959).
    let (Some(dx), Some(dy)) = (r.x0.checked_neg(), r.y0.checked_neg()) else {
        return Err(bad(cmd, format!("the crop origin ({}, {}) can't be moved to (0, 0) within the 32-bit coordinate range", r.x0, r.y0)));
    };
    let old_canvas = doc.bounds();
    let extends = r.x0 < old_canvas.x0 || r.y0 < old_canvas.y0 || r.x1 > old_canvas.x1 || r.y1 > old_canvas.y1;
    if delete_pixels {
        for_each_surface(&mut doc.layers, false, &mut |surf, _| *surf = crop_surface(surf, r));
    }
    translate_doc(doc, cmd, dx, dy)?;
    doc.size = Size::new(r.width(), r.height());
    // A Background cannot hold transparent pixels: like an opaque Canvas Size extension, the area
    // an extending Crop adds is filled with the extension (background) colour.
    if extends
        && let Some(bg) = doc.background_mut()
        && let LayerContent::Raster(surf) = &mut bg.content
    {
        let old = old_canvas.translate(dx, dy);
        let fmt = surf.format();
        let px = photocraft_raster::from_rgba(&fmt, [extension[0], extension[1], extension[2], 1.0]);
        let keep = surf.content_bounds().intersect(&old);
        let saved = surf.to_interleaved(keep);
        surf.fill_rect(Rect::from_xywh(0, 0, r.width(), r.height()), &px);
        if !keep.is_empty() {
            surf.write_interleaved(keep, &saved);
        }
    }
    crate::canvas_geom::refresh(doc, crate::canvas_geom::Refresh::Shapes);
    Ok(())
}

/// The nine documented canvas-size anchors, in the order the command lists them, each mapped to
/// its (x, y) factor (0.0 = top/left edge, 0.5 = centre, 1.0 = bottom/right edge).
const ANCHORS: [(&str, (f64, f64)); 9] = [
    ("topLeft", (0.0, 0.0)),
    ("top", (0.5, 0.0)),
    ("topRight", (1.0, 0.0)),
    ("left", (0.0, 0.5)),
    ("center", (0.5, 0.5)),
    ("right", (1.0, 0.5)),
    ("bottomLeft", (0.0, 1.0)),
    ("bottom", (0.5, 1.0)),
    ("bottomRight", (1.0, 1.0)),
];

/// Resolves an anchor name to its (x, y) factors, matching the nine documented values exactly.
/// Anything else is an error naming `anchor` and listing them, rather than resolving the two axes
/// independently (which let a misspelling like `"topleft"` land between two valid anchors, #2946).
fn anchor_factors(cmd: &str, a: &str) -> Result<(f64, f64)> {
    ANCHORS.iter().find(|(name, _)| *name == a).map(|&(_, f)| f).ok_or_else(|| {
        let names = ANCHORS.iter().map(|&(n, _)| n).collect::<Vec<_>>().join("|");
        bad(cmd, format!("`anchor` must be one of {names}"))
    })
}

fn extension_color(s: &Session, p: &Value) -> [f32; 4] {
    match p.get("extensionColor").and_then(Value::as_str).unwrap_or("background") {
        "foreground" => s.tools.foreground,
        "white" => [1.0; 4],
        "black" => [0.0, 0.0, 0.0, 1.0],
        "transparent" => [0.0; 4],
        hex if hex.starts_with('#') => {
            let h = hex.trim_start_matches('#');
            let c = |i: usize| h.get(i..i + 2).and_then(|x| u8::from_str_radix(x, 16).ok()).map_or(0.0, |v| f32::from(v) / 255.0);
            [c(0), c(2), c(4), 1.0]
        }
        _ => s.tools.background,
    }
}

/// Image → Canvas Size.
fn canvas_size(s: &mut Session, p: &Value) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let (ow, oh) = (d.doc.size.width as f64, d.doc.size.height as f64);
    let relative = p.get("relative").and_then(Value::as_bool).unwrap_or(false);
    let w = p.get("width").and_then(Value::as_f64).unwrap_or(if relative { 0.0 } else { ow });
    let h = p.get("height").and_then(Value::as_f64).unwrap_or(if relative { 0.0 } else { oh });
    let (nw, nh) = if relative { (ow + w, oh + h) } else { (w, h) };
    let (nw, nh) = (nw.round().clamp(1.0, 300_000.0) as u32, nh.round().clamp(1.0, 300_000.0) as u32);
    let (ax, ay) = anchor_factors("image.canvasSize", p.get("anchor").and_then(Value::as_str).unwrap_or("center"))?;
    let dx = ((nw as f64 - ow) * ax).round() as i32;
    let dy = ((nh as f64 - oh) * ay).round() as i32;
    let ext = extension_color(s, p);
    s.set_sampler_map(photocraft_geom::Affine::translate(f64::from(dx), f64::from(dy)));
    s.edit("Canvas Size", |doc, _| {
        translate_doc(doc, "image.canvasSize", dx, dy)?;
        doc.size = Size::new(nw, nh);
        crate::canvas_geom::refresh(doc, crate::canvas_geom::Refresh::Shapes);
        let canvas = doc.bounds();
        let old = Rect::from_xywh(dx, dy, ow as u32, oh as u32);
        // A Background cannot contain transparent pixels. When a transparent
        // extension adds canvas area, make it a normal paintable layer first.
        // A shrink or no-op resize must not silently convert the Background.
        if ext[3] == 0.0
            && (nw > ow as u32 || nh > oh as u32)
            && let Some(bg) = doc.background_mut()
        {
            crate::extra_cmds::unlock_background(bg);
        }
        // The locked Background layer is extended with the extension colour.
        if ext[3] > 0.0
            && let Some(bg) = doc.background_mut()
            && let LayerContent::Raster(surf) = &mut bg.content
        {
            let fmt = surf.format();
            let px = photocraft_raster::from_rgba(&fmt, ext);
            let keep = surf.content_bounds().intersect(&old);
            let saved = surf.to_interleaved(keep);
            surf.fill_rect(canvas, &px);
            if !keep.is_empty() {
                surf.write_interleaved(keep, &saved);
            }
        }
        Ok(())
    })?;
    Ok(json!({ "width": nw, "height": nh, "offset": [dx, dy] }))
}

/// The largest side an explicit crop may give the canvas (the `file.new` and Canvas Size limit).
const MAX_CROP_SIDE: i32 = 300_000;

/// Image → Crop (to the selection bounds).
fn crop(s: &mut Session, p: &Value) -> Result<Value> {
    let delete = p.get("deleteCroppedPixels").and_then(Value::as_bool).unwrap_or(true);
    // Values beyond i32 would wrap through the narrowing casts into a rectangle unrelated to
    // the numbers passed; reject them (see `commands::int_i32`).
    let explicit = match (
        crate::commands::int_i32("image.crop", p, "x")?,
        crate::commands::int_i32("image.crop", p, "y")?,
        crate::commands::int_i32("image.crop", p, "width")?,
        crate::commands::int_i32("image.crop", p, "height")?,
    ) {
        (Some(x), Some(y), Some(w), Some(h)) if w > 0 && h > 0 => {
            // The new canvas size: refuse what File › New, Image Size and Canvas Size won't make,
            // before anything changes. Unbounded, a following full-canvas flatten (Trim, Duplicate
            // Merged) aborted on allocation (#960).
            if w > MAX_CROP_SIDE || h > MAX_CROP_SIDE {
                return Err(bad("image.crop", format!("{w} x {h} exceeds the {MAX_CROP_SIDE} px limit per side")));
            }
            // A far edge past i32::MAX used to saturate, silently cropping less than asked (#959).
            let end = |o: i32, len: i32, ko: &str, kl: &str| {
                o.checked_add(len)
                    .ok_or_else(|| bad("image.crop", format!("`{ko}` + `{kl}` = {} is outside the 32-bit coordinate range", i64::from(o) + i64::from(len))))
            };
            Some(Rect::new(x, y, end(x, w, "x", "width")?, end(y, h, "y", "height")?))
        }
        _ => None,
    };
    let target = crop_target(p)?;
    if let Some((tw, th)) = target.size {
        let d = s.active().ok_or(EngineError::NoDocument)?;
        check_resample_budget("image.crop", &d.doc, tw, th, "")?;
    }
    if let Some(a) = crop_angle(p)? {
        let rect = explicit.ok_or_else(|| bad("image.crop", "an `angle` needs an explicit `x`, `y`, `width` and `height`"))?;
        return rotated_crop(s, rect, a, delete, &target);
    }
    // An explicit rectangle (the Crop tool) may extend past the canvas; a selection crop is clamped.
    let r = explicit
        .or_else(|| s.active().and_then(|d| d.doc.selection.as_ref().map(|sel| sel.content_bounds().intersect(&d.doc.bounds()))))
        .unwrap_or(Rect::EMPTY);
    if r.is_empty() {
        return Err(EngineError::Other("nothing to crop: pass x/y/width/height or make a selection".into()));
    }
    s.set_sampler_map(photocraft_geom::Affine::translate(-f64::from(r.x0), -f64::from(r.y0)));
    let extension = s.tools.background;
    s.edit("Crop", |doc, _| {
        crop_doc(doc, "image.crop", r, delete, extension)?;
        doc.selection = None;
        target.apply(doc);
        Ok(())
    })?;
    Ok(crop_result(s, json!({ "x": r.x0, "y": r.y0, "width": r.width(), "height": r.height() })))
}

/// The Crop tool's W x H x Resolution: the size (px) the cropped area is resampled to and the
/// resolution the document gets, both optional.
struct CropTarget {
    size: Option<(u32, u32)>,
    resolution: Option<f32>,
    filter: Resample,
}

impl CropTarget {
    /// Applies the target to a document that has just been cropped (inside the crop's history step).
    fn apply(&self, doc: &mut Document) {
        if let Some((w, h)) = self.size {
            resample_doc(doc, w, h, self.filter);
        }
        if let Some(r) = self.resolution {
            doc.resolution_dpi = r;
        }
    }
}

/// The highest resolution (ppi) a crop may set, as for Image Size.
const MAX_CROP_RESOLUTION: f64 = 10_000.0;

/// `image.crop`'s optional `targetWidth`/`targetHeight` (whole px, both or neither), `resolution`
/// (ppi) and `resample` (as for `image.imageSize`; "none" is refused with a target size).
fn crop_target(p: &Value) -> Result<CropTarget> {
    const CMD: &str = "image.crop";
    let side = |key: &str| -> Result<Option<u32>> {
        match p.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => v
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .filter(|n| (1..=MAX_CROP_SIDE as u32).contains(n))
                .map(Some)
                .ok_or_else(|| bad(CMD, format!("`{key}` = {v} must be a whole number of pixels from 1 to {MAX_CROP_SIDE}"))),
        }
    };
    let size = match (side("targetWidth")?, side("targetHeight")?) {
        (Some(w), Some(h)) => Some((w, h)),
        (None, None) => None,
        _ => return Err(bad(CMD, "`targetWidth` and `targetHeight` go together")),
    };
    let resolution = match p.get("resolution") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let r = v.as_f64().filter(|r| r.is_finite() && (1.0..=MAX_CROP_RESOLUTION).contains(r));
            Some(r.ok_or_else(|| bad(CMD, format!("`resolution` = {v} must be a number of pixels per inch from 1 to {MAX_CROP_RESOLUTION}")))? as f32)
        }
    };
    let filter = match p.get("resample") {
        None | Some(Value::Null) => Some(Resample::Bicubic),
        Some(v) => match parse_resample(v.as_str().ok_or_else(|| bad(CMD, "`resample` must be a string"))?) {
            Method::None => None,
            Method::Filter(f) => Some(f),
            // The direction is known only once the frame is: bicubic's own automatic choice.
            Method::Automatic => Some(Resample::BicubicAutomatic),
        },
    };
    let filter = match (filter, size) {
        (Some(f), _) => f,
        (None, None) => Resample::Bicubic,
        (None, Some(_)) => return Err(bad(CMD, "a `targetWidth` x `targetHeight` crop needs resampling: `resample` can't be \"none\"")),
    };
    Ok(CropTarget { size, resolution, filter })
}

/// A crop's result: the frame it cropped to, plus the document's size and resolution afterwards.
fn crop_result(s: &Session, mut frame: Value) -> Value {
    if let (Some(d), Some(o)) = (s.active(), frame.as_object_mut()) {
        o.insert("documentWidth".into(), json!(d.doc.size.width));
        o.insert("documentHeight".into(), json!(d.doc.size.height));
        o.insert("resolution".into(), json!(d.doc.resolution_dpi));
    }
    frame
}

/// The largest crop angle accepted (degrees), as for Image Rotation › Arbitrary.
const MAX_CROP_ANGLE: f64 = 3600.0;

/// `image.crop`'s optional `angle` in degrees, normalised to (-180, 180]; `None` when it is absent,
/// null or (after normalising) zero, so an unrotated crop takes the plain, resampling-free path.
fn crop_angle(p: &Value) -> Result<Option<f64>> {
    let a = match p.get("angle") {
        None | Some(Value::Null) => return Ok(None),
        Some(v) => v.as_f64().ok_or_else(|| bad("image.crop", "`angle` must be a number of degrees"))?,
    };
    if !a.is_finite() || a.abs() > MAX_CROP_ANGLE {
        return Err(bad("image.crop", format!("`angle` must be a finite number of degrees within ±{MAX_CROP_ANGLE}")));
    }
    let a = a.rem_euclid(360.0);
    let a = if a > 180.0 { a - 360.0 } else { a };
    Ok((a.abs() >= 1e-9).then_some(a))
}

/// The Crop tool's rotated frame: `r` (document px) turned clockwise on screen by `angle` degrees
/// about its centre. The document is rotated counter-clockwise by `angle` (Image Rotation ›
/// Arbitrary: the canvas grows to fit, the Background fills what the rotation uncovers with the
/// background colour, other layers keep transparency there and every pixel of theirs), which makes
/// the frame upright, then cropped to it (`deleteCroppedPixels` as for an unrotated crop), all in
/// one undo step.
fn rotated_crop(s: &mut Session, r: Rect, angle: f64, delete: bool, crop_target: &CropTarget) -> Result<Value> {
    const CMD: &str = "image.crop";
    let size = s.active().ok_or(EngineError::NoDocument)?.doc.size;
    let (w, h) = (f64::from(r.width()), f64::from(r.height()));
    let (cx, cy) = (f64::from(r.x0) + w / 2.0, f64::from(r.y0) + h / 2.0);
    // Where the frame's centre lands: the rotation `image.rotation.arbitrary` applies, clockwise
    // by `-angle` (x' = c·x − s·y, y' = s·x + c·y) about the old and new canvas centres.
    let new = crate::mode_cmds::rotated_size(size, -angle);
    let (sn, cs) = (-angle).to_radians().sin_cos();
    let (dx, dy) = (cx - f64::from(size.width) / 2.0, cy - f64::from(size.height) / 2.0);
    let (ncx, ncy) = (f64::from(new.width) / 2.0 + cs * dx - sn * dy, f64::from(new.height) / 2.0 + sn * dx + cs * dy);
    let (x, y) = ((ncx - w / 2.0).round(), (ncy - h / 2.0).round());
    let far = || bad(CMD, "the rotated frame is outside the 32-bit coordinate range");
    let limit = f64::from(i32::MAX / 2);
    if !(x.is_finite() && y.is_finite()) || x.abs() > limit || y.abs() > limit {
        return Err(far());
    }
    let (x, y) = (x as i32, y as i32);
    let end = |o: i32, len: u32| i32::try_from(len).ok().and_then(|l| o.checked_add(l));
    let (Some(x1), Some(y1)) = (end(x, r.width()), end(y, r.height())) else { return Err(far()) };
    let target = Rect::new(x, y, x1, y1);
    crate::analysis_cmds::compound(s, "Crop", |s| {
        crate::mode_cmds::rotate_arbitrary(s, &json!({"angle": angle, "direction": "ccw"}))?;
        // Color Sampler points follow the crop's origin shift too (the rotation moved them above).
        s.set_sampler_map(photocraft_geom::Affine::translate(-f64::from(target.x0), -f64::from(target.y0)));
        let extension = s.tools.background;
        s.edit("Crop", |doc, _| {
            crop_doc(doc, CMD, target, delete, extension)?;
            doc.selection = None;
            crop_target.apply(doc);
            Ok(())
        })
    })?;
    Ok(crop_result(s, json!({ "x": r.x0, "y": r.y0, "width": r.width(), "height": r.height(), "angle": angle })))
}

/// Image → Trim.
fn trim(s: &mut Session, p: &Value) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let canvas = d.doc.bounds();
    let buf = photocraft_compose::flatten(&d.doc);
    let based = p.get("basedOn").and_then(Value::as_str).unwrap_or("transparent");
    let w = canvas.width() as usize;
    let reference = match based {
        "topLeft" => Some(buf.px[0]),
        "bottomRight" => buf.px.last().copied(),
        _ => None,
    };
    let differs = |q: [f32; 4]| match reference {
        None => q[3] > 0.0,
        Some(r) => (0..4).any(|c| (q[c] * q[3] - r[c] * r[3]).abs() > 0.5 / 255.0 || (c == 3 && (q[3] - r[3]).abs() > 0.5 / 255.0)),
    };
    let mut b = Rect::EMPTY;
    for (i, q) in buf.px.iter().enumerate() {
        if differs(*q) {
            let (x, y) = ((i % w) as i32, (i / w) as i32);
            b = if b.is_empty() { Rect::new(x, y, x + 1, y + 1) } else { b.union(&Rect::new(x, y, x + 1, y + 1)) };
        }
    }
    if b.is_empty() {
        return Err(EngineError::Other("nothing to trim to".into()));
    }
    let side = |k: &str| p.get(k).and_then(Value::as_bool).unwrap_or(true);
    let r = Rect::new(
        if side("left") { b.x0 } else { canvas.x0 },
        if side("top") { b.y0 } else { canvas.y0 },
        if side("right") { b.x1 } else { canvas.x1 },
        if side("bottom") { b.y1 } else { canvas.y1 },
    );
    s.set_sampler_map(photocraft_geom::Affine::translate(-f64::from(r.x0), -f64::from(r.y0)));
    let extension = s.tools.background;
    s.edit("Trim", |doc, _| crop_doc(doc, "image.trim", r, true, extension))?;
    Ok(json!({ "x": r.x0, "y": r.y0, "width": r.width(), "height": r.height() }))
}

/// Image → Mode → colour model, through the CMS (see [`crate::color_cmds::convert_mode`]).
fn convert_mode(s: &mut Session, mode: ColorMode, p: &Value) -> Result<Value> {
    crate::color_cmds::convert_mode(s, mode, p)
}

/// Image → Mode → 8/16/32 Bits/Channel.
fn convert_depth(s: &mut Session, depth: SampleType) -> Result<Value> {
    let current = s.active().ok_or(EngineError::NoDocument)?.doc.depth;
    if current == depth {
        // History::record clears redo even when an edit closure leaves the document unchanged.
        return Ok(Value::Null);
    }
    s.edit("Bit Depth", |doc, _| {
        // 32-bit documents hold linear light (as Photoshop's do: grey 128 becomes 0.2159, measured
        // in Photoshop 27.11), so the colour samples cross the document's tone curve when the
        // depth crosses 32 bits: decoded after widening, encoded before narrowing (so no linear
        // value is quantised to 8 or 16 bits). Masks, alpha and channels are not colour.
        let crosses = (current == SampleType::F32) != (depth == SampleType::F32) && matches!(doc.mode, ColorMode::Rgb | ColorMode::Grayscale);
        let to_linear = depth == SampleType::F32;
        let transfer = photocraft_compose::adjust::Transfer::for_document(doc.mode, if to_linear { current } else { depth });
        if crosses && !to_linear {
            for_each_surface(&mut doc.layers, false, &mut |surf, _| surf.map_color(|v| transfer.encode(v).clamp(0.0, 1.0)));
        }
        convert_layers_depth(&mut doc.layers, depth);
        if crosses && to_linear {
            for_each_surface(&mut doc.layers, false, &mut |surf, _| surf.map_color(|v| transfer.decode(v)));
        }
        for ch in doc.channels.iter_mut().chain(doc.quick_mask.as_mut()) {
            let f = ch.surface.format().with_sample(depth);
            ch.surface = ch.surface.convert(f);
        }
        doc.depth = depth;
        Ok(())
    })?;
    Ok(Value::Null)
}

/// Image → Duplicate.
fn duplicate(s: &mut Session, p: &Value) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let mut doc = (*d.doc).clone();
    doc.id = photocraft_doc::DocId::fresh();
    doc.name = p.get("name").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| format!("{} copy", d.doc.name));
    if p.get("mergedOnly").and_then(Value::as_bool).unwrap_or(false) {
        let buf = photocraft_compose::flatten(&doc);
        let fmt = doc.pixel_format();
        let mut surf = Surface::new(fmt);
        let vals: Vec<f32> = buf.px.iter().flat_map(|q| photocraft_raster::from_rgba(&fmt, *q)).collect();
        surf.write_region(doc.bounds(), &vals);
        surf.prune();
        doc.layers = vec![Layer::new("Background", LayerContent::Raster(surf))];
        // A merged-only copy has no original layers for saved comps to restore.
        // Don't carry dangling comp state into a single-layer flattened document.
        doc.layer_comps.clear();
        doc.last_applied_comp = None;
        doc.last_document_state = None;
    }
    reidentify_copied_layers(&mut doc);
    let name = doc.name.clone();
    let index = s.add_document(doc, None);
    Ok(json!({ "document": index, "name": name }))
}

macro_rules! spec {
    ($id:literal, $label:literal, [$($m:literal),*], $params:literal, $en:expr, $run:expr) => {
        CommandSpec { id: $id, label: $label, menu: &[$($m),*], shortcut: None, params: $params, enabled: $en, run: $run, journal: true }
    };
}

/// Image menu command specs.
pub fn specs() -> Vec<CommandSpec> {
    vec![
        spec!(
            "image.imageSize",
            "Image Size…",
            ["Image"],
            r##"{"width":px,"height":px,"percent":%? (both sides; an explicit side wins),"resolution":ppi,"keepPrintSize":bool=false (with only a resolution: resample the pixels to keep the print size),"resample":"automatic|preserveDetails|bicubicSmoother|bicubicSharper|bicubicAutomatic|bicubic|nearest|bilinear|lanczos|none"="bicubic"} → {width,height} (the document's size after the command; "none" changes the resolution only)"##,
            has_doc,
            image_size
        ),
        spec!(
            "image.canvasSize",
            "Canvas Size…",
            ["Image"],
            r##"{"width":px,"height":px,"relative":bool=false,"anchor":"topLeft|top|topRight|left|center|right|bottomLeft|bottom|bottomRight"="center","extensionColor":"background|foreground|white|black|transparent|#rrggbb"="background"}"##,
            has_doc,
            canvas_size
        ),
        spec!(
            "image.crop",
            "Crop",
            ["Image"],
            r##"{"x":px,"y":px,"width":px,"height":px,"angle":-3600..3600=0,"deleteCroppedPixels":bool=true,"targetWidth":px,"targetHeight":px,"resolution":ppi,"resample":"bicubic|bilinear|nearest|lanczos|preserveDetails"="bicubic"} → {x,y,width,height,documentWidth,documentHeight,resolution} (targetWidth/targetHeight, both or neither: the cropped area is resampled to exactly that size, as Image Size, in the same undo step; resolution: the document's new ppi)"##,
            has_doc,
            crop
        ),
        spec!(
            "image.trim",
            "Trim…",
            ["Image"],
            r##"{"basedOn":"transparent|topLeft|bottomRight"="transparent","top":bool=true,"bottom":bool=true,"left":bool=true,"right":bool=true}"##,
            has_doc,
            trim
        ),
        spec!(
            "image.mode.rgb",
            "RGB Color",
            ["Image", "Mode"],
            r##"{"profile":"<builtin id>|working|/path/to/profile.icc"=working,"intent":"perceptual|relative|saturation|absolute"="relative","bpc":bool=true}"##,
            has_doc,
            |s, p| convert_mode(s, ColorMode::Rgb, p)
        ),
        spec!(
            "image.mode.grayscale",
            "Grayscale",
            ["Image", "Mode"],
            r##"{"profile":"<builtin id>|working|/path/to/profile.icc"=working,"intent":"perceptual|relative|saturation|absolute"="relative","bpc":bool=true}"##,
            has_doc,
            |s, p| convert_mode(s, ColorMode::Grayscale, p)
        ),
        spec!(
            "image.mode.cmyk",
            "CMYK Color",
            ["Image", "Mode"],
            r##"{"profile":"<builtin id>|working|/path/to/profile.icc"=working,"intent":"perceptual|relative|saturation|absolute"="relative","bpc":bool=true}"##,
            has_doc,
            |s, p| convert_mode(s, ColorMode::Cmyk, p)
        ),
        spec!(
            "image.mode.lab",
            "Lab Color",
            ["Image", "Mode"],
            r##"{"profile":"<builtin id>|working|/path/to/profile.icc"=working,"intent":"perceptual|relative|saturation|absolute"="relative","bpc":bool=true}"##,
            has_doc,
            |s, p| convert_mode(s, ColorMode::Lab, p)
        ),
        spec!("image.mode.bits8", "8 Bits/Channel", ["Image", "Mode"], "{}", has_doc, |s, _| convert_depth(s, SampleType::U8)),
        spec!("image.mode.bits16", "16 Bits/Channel", ["Image", "Mode"], "{}", has_doc, |s, _| convert_depth(s, SampleType::U16)),
        spec!("image.mode.bits32", "32 Bits/Channel", ["Image", "Mode"], "{}", has_doc, |s, _| convert_depth(s, SampleType::F32)),
        spec!("image.duplicate", "Duplicate…", ["Image"], r##"{"name":str,"mergedOnly":bool=false}"##, has_doc, duplicate),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_rejects_rectangles_that_would_wrap() {
        let mut s = session();
        // 2^32 + 100 wrapped to `x = 100` through the narrowing casts: a crop rectangle
        // somewhere unrelated to the numbers passed.
        let err = s.execute("image.crop", json!({"x": 4_294_967_396_i64, "y": 0, "width": 10, "height": 10})).unwrap_err();
        assert!(err.to_string().contains("32-bit"), "{err}");
        // In-range rectangles past the canvas still work (clamped by the crop itself).
        s.execute("image.crop", json!({"x": -10, "y": -10, "width": 1000, "height": 1000})).unwrap();
    }

    /// A crop whose translation to the origin can't be represented in i32 is rejected before
    /// anything changes (#959): `-i32::MIN` overflowed the negation in `crop_doc`, `x + width`
    /// past i32::MAX silently shortened the crop, and an origin one past i32::MIN moved the
    /// layer pixels beyond i32::MAX (a length-mismatch panic while translating surfaces).
    #[test]
    fn crop_rejects_origins_that_cannot_move_to_zero() {
        const MIN: i64 = i32::MIN as i64;
        const MAX: i64 = i32::MAX as i64;
        let cases = [
            json!({"x": MIN, "y": 0, "width": 1, "height": 1}),
            json!({"x": 0, "y": MIN, "width": 1, "height": 1}),
            json!({"x": MIN, "y": MIN, "width": 1, "height": 1}),
            json!({"x": MIN, "y": 0, "width": 1, "height": 1, "deleteCroppedPixels": false}),
            json!({"x": MAX - 5, "y": 0, "width": 10, "height": 1}),
            json!({"x": 0, "y": MAX - 5, "width": 1, "height": 10}),
            json!({"x": MIN + 1, "y": 0, "width": 1, "height": 1, "deleteCroppedPixels": false}),
            json!({"x": 0, "y": MIN + 1, "width": 1, "height": 1, "deleteCroppedPixels": false}),
        ];
        for p in cases {
            let mut s = session();
            let before = s.active().unwrap().doc.clone();
            let (rev, steps) = (s.active().unwrap().revision, s.active().unwrap().history.entries().len());
            let err = s.execute("image.crop", p.clone()).unwrap_err();
            assert!(matches!(err, EngineError::BadParams { .. }), "{p}: {err}");
            assert!(err.to_string().contains("32-bit"), "{p}: {err}");
            let st = s.active().unwrap();
            assert!(std::sync::Arc::ptr_eq(&st.doc, &before), "{p}: document changed");
            assert_eq!(st.doc.size, Size::new(40, 20), "{p}");
            assert_eq!(st.doc.layers[1].surface().unwrap().content_bounds(), Rect::new(10, 5, 20, 15), "{p}");
            assert_eq!((st.revision, st.history.entries().len()), (rev, steps), "{p}: history step recorded");
        }
        // Control: a legal negative origin past the canvas still crops (and extends) the canvas.
        let mut s = session();
        s.execute("image.crop", json!({"x": -10, "y": -5, "width": 60, "height": 30})).unwrap();
        assert_eq!(doc(&s).size, Size::new(60, 30));
        assert_eq!(doc(&s).layers[1].surface().unwrap().pixel(20, 10), vec![1.0, 0.0, 0.0, 1.0]);
        // A far-off crop that deletes the (non-overlapping) pixels has nothing left to move.
        let mut s = session();
        s.execute("image.crop", json!({"x": MIN + 1, "y": 0, "width": 1, "height": 1})).unwrap();
        assert_eq!(doc(&s).size, Size::new(1, 1));
    }

    fn session() -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 40, "height": 20})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.edit("paint", |doc, active| {
            let l = doc.layer_mut(active.unwrap()).unwrap();
            l.surface_mut().unwrap().fill_rect(Rect::new(10, 5, 20, 15), &[1.0, 0.0, 0.0, 1.0]);
            l.mask = Some(photocraft_doc::LayerMask::reveal_all());
            l.effects.items.push(Effect::default_drop_shadow());
            Ok(())
        })
        .unwrap();
        s
    }

    fn doc(s: &Session) -> &Document {
        &s.active().unwrap().doc
    }

    /// Resample on with only a new resolution keeps the print size, as Photoshop's Image Size
    /// does (an action's `imageSize` with `Rslt` alone); without `keepPrintSize` only the
    /// resolution changes.
    #[test]
    fn image_size_keep_print_size_resamples_to_the_resolution() {
        let mut s = session();
        s.execute("image.imageSize", json!({"resolution": 144, "resample": "bicubic"})).unwrap();
        assert_eq!((doc(&s).size, doc(&s).resolution_dpi), (Size::new(40, 20), 144.0));
        s.execute("image.imageSize", json!({"resolution": 288, "resample": "bicubic", "keepPrintSize": true})).unwrap();
        assert_eq!((doc(&s).size, doc(&s).resolution_dpi), (Size::new(80, 40), 288.0));
        // An explicit side still wins.
        s.execute("image.imageSize", json!({"width": 20, "resolution": 72, "resample": "bicubic", "keepPrintSize": true})).unwrap();
        assert_eq!(doc(&s).size, Size::new(20, 10));
    }

    #[test]
    fn image_size_scales_everything() {
        for resample in ["bicubic", "bicubicSmoother", "bicubicSharper", "bicubicAutomatic", "automatic", "bilinear", "nearest", "lanczos", "preserveDetails"] {
            let mut s = session();
            s.execute("image.imageSize", json!({"width": 80, "resample": resample})).unwrap();
            let d = doc(&s);
            assert_eq!(d.size, Size::new(80, 40), "{resample}");
            let l = &d.layers[1];
            let b = l.surface().unwrap().content_bounds();
            assert!((b.x0 - 20).abs() <= 1 && (b.x1 - 40).abs() <= 1, "{resample} {b:?}");
            let Effect::DropShadow(sh) = &l.effects.items[0] else { panic!() };
            assert_eq!(sh.distance, 10.0);
            assert_eq!(l.mask.as_ref().unwrap().surface.default_pixel(), vec![1.0]);
        }
    }

    /// The canvas border stays opaque after Image Size (it used to fade into transparency, so a
    /// Background or a 200 % export got a translucent frame), and a full selection stays full.
    /// #1544: 64 x 48 to 300000 x 300000 (90 billion pixels) started allocating resampled bands
    /// without a budget. It is refused before any allocation and the document is untouched; a
    /// change of resolution only, and an ordinary resize, still work.
    #[test]
    fn image_size_refuses_a_raster_past_the_budget() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
        let before = (doc(&s).size, s.active().unwrap().history.past_len());
        let err = s.execute("image.imageSize", json!({"width": 300_000, "height": 300_000})).unwrap_err().to_string();
        assert!(err.contains("megapixels"), "{err}");
        assert_eq!((doc(&s).size, s.active().unwrap().history.past_len()), before);
        s.execute("image.imageSize", json!({"width": 300_000, "height": 300_000, "resample": "none", "resolution": 300})).unwrap();
        s.execute("image.imageSize", json!({"width": 128, "height": 96})).unwrap();
        assert_eq!(doc(&s).size, Size::new(128, 96));
    }

    #[test]
    fn image_size_keeps_canvas_edges_opaque() {
        for depth in [8, 16, 32] {
            for (w, resample) in [
                (60, "bicubic"),
                (41, "lanczos"),
                (15, "bilinear"),
                (77, "preserveDetails"),
                (21, "bicubicSharper"),
                (47, "automatic"),
                (9, "bicubicAutomatic"),
            ] {
                let mut s = Session::new();
                s.execute("file.new", json!({"width": 30, "height": 20, "depth": depth, "background": "#336699"})).unwrap();
                s.execute("select.all", json!({})).unwrap();
                s.execute("image.imageSize", json!({"width": w, "resample": resample})).unwrap();
                let d = doc(&s);
                let flat = photocraft_compose::flatten(d);
                let (w, h) = (flat.rect.width() as usize, flat.rect.height() as usize);
                let mid = flat.px[(h / 2) * w + w / 2];
                assert!((mid[3] - 1.0).abs() < 1e-3, "{depth} {resample}: {mid:?}");
                for p in &flat.px {
                    let off = p.iter().zip(mid).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                    assert!(off < 0.01, "{depth} {resample} {w}: {p:?} vs {mid:?}");
                }
                let sel = d.selection.as_ref().unwrap();
                let (x1, y1) = (d.size.width as i32 - 1, d.size.height as i32 - 1);
                for (x, y) in [(0, 0), (x1, 0), (0, y1), (x1, y1)] {
                    assert!(sel.pixel(x, y)[0] > 0.99, "{depth} {resample}: selection at ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn resolution_only_change() {
        let mut s = session();
        let r = s.execute("image.imageSize", json!({"resolution": 300, "resample": "none", "width": 999})).unwrap();
        assert_eq!(doc(&s).size, Size::new(40, 20));
        assert_eq!(doc(&s).resolution_dpi, 300.0);
        // #1815: the result is the size the document has, not the ignored request.
        assert_eq!(r, json!({"width": 40, "height": 20}));
    }

    #[test]
    fn image_size_reports_the_size_it_applied() {
        // #1815: every path reports the document's size afterwards, at every depth.
        for depth in [8, 16, 32] {
            let mut s = Session::new();
            s.execute("file.new", json!({"width": 64, "height": 48, "depth": depth})).unwrap();
            let size = |s: &Session| json!({"width": doc(s).size.width, "height": doc(s).size.height});
            for p in [
                json!({"width": 999, "height": 777, "resample": "none", "resolution": 150}),
                json!({"width": 32}),
                json!({"height": 96, "resample": "nearest"}),
                json!({"width": 96, "height": 72, "resample": "none"}),
                json!({}),
            ] {
                let r = s.execute("image.imageSize", p.clone()).unwrap();
                assert_eq!(r, size(&s), "{depth}-bit {p}");
            }
            // 64×48 → (none) → 32×24 → 128×96 (aspect kept) → (none) → unchanged.
            assert_eq!(doc(&s).size, Size::new(128, 96), "{depth}-bit");
        }
    }

    /// #1114: nearest neighbor reduction keeps source values without averaging.
    #[test]
    fn nearest_neighbor_reduction_keeps_source_values() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
        s.execute("tools.setColors", json!({"foreground": "#000000", "background": "#ffffff"})).unwrap();
        for y in 0..8 {
            for x in 0..8 {
                if (x + y) % 2 == 0 {
                    s.execute("select.rect", json!({"x": x, "y": y, "width": 1, "height": 1})).unwrap();
                    s.execute("edit.fill", json!({"contents": "foreground"})).unwrap();
                }
            }
        }
        s.execute("select.deselect", json!({})).unwrap();
        let st = s.active().unwrap();
        let before = st.doc.layers[0].surface().unwrap().rgba(0, 0);
        assert!(before[0] < 0.01, "setup: (0,0) should be black, got {before:?}");
        s.execute("image.imageSize", json!({"width": 4, "height": 4, "resample": "nearest"})).unwrap();
        let st = s.active().unwrap();
        let surf = st.doc.layers[0].surface().unwrap();
        for (x, y) in [(0, 0), (1, 1), (2, 3)] {
            let v = surf.rgba(x, y)[0];
            assert!(v.is_finite() && !(0.01..=0.99).contains(&v), "nearest produced an averaged value {v} at ({x},{y})");
        }
    }

    #[test]
    fn canvas_size_anchors_and_extends_background() {
        let mut s = session();
        s.execute("image.canvasSize", json!({"width": 60, "height": 30, "anchor": "topLeft", "extensionColor": "#00ff00"})).unwrap();
        let d = doc(&s);
        assert_eq!(d.size, Size::new(60, 30));
        assert_eq!(d.layers[1].surface().unwrap().content_bounds(), Rect::new(10, 5, 20, 15));
        let bg = d.layers[0].surface().unwrap();
        assert_eq!(bg.pixel(50, 25), vec![0.0, 1.0, 0.0, 1.0]);
        assert_eq!(bg.pixel(5, 5), vec![1.0, 1.0, 1.0, 1.0]);
        let mut s = session();
        s.execute("image.canvasSize", json!({"width": 20, "height": 10, "relative": true, "anchor": "center"})).unwrap();
        assert_eq!(doc(&s).layers[1].surface().unwrap().content_bounds(), Rect::new(20, 10, 30, 20));
    }

    #[test]
    fn canvas_size_anchor_offsets_and_rejects_unlisted() {
        // 64x48 -> 100x80: dx spans 36 and dy spans 32, so each anchor factor lands on a known
        // offset. All nine documented values must anchor exactly where they do today (#2946).
        let expected = [
            ("topLeft", [0, 0]),
            ("top", [18, 0]),
            ("topRight", [36, 0]),
            ("left", [0, 16]),
            ("center", [18, 16]),
            ("right", [36, 16]),
            ("bottomLeft", [0, 32]),
            ("bottom", [18, 32]),
            ("bottomRight", [36, 32]),
        ];
        for (anchor, off) in expected {
            let mut s = Session::new();
            s.execute("file.new", json!({"width": 64, "height": 48, "background": "white"})).unwrap();
            let r = s.execute("image.canvasSize", json!({"width": 100, "height": 80, "anchor": anchor})).unwrap();
            assert_eq!(r["offset"], json!(off), "anchor {anchor}");
        }
        // An omitted anchor still defaults to center.
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 64, "height": 48, "background": "white"})).unwrap();
        assert_eq!(s.execute("image.canvasSize", json!({"width": 100, "height": 80})).unwrap()["offset"], json!([18, 16]));
        // A misspelling or unknown value is a BadParams error naming `anchor`, never an anchor
        // between two of the nine (the old independent-axis matching produced top-centre etc.).
        for anchor in ["topleft", "TopLeft", "TOPLEFT", "bottomleft", "bottomright", "middle", "", "NOPE"] {
            let mut s = Session::new();
            s.execute("file.new", json!({"width": 64, "height": 48, "background": "white"})).unwrap();
            let err = s.execute("image.canvasSize", json!({"width": 100, "height": 80, "anchor": anchor})).unwrap_err();
            assert!(matches!(err, EngineError::BadParams { .. }), "anchor {anchor:?} must be rejected, got {err:?}");
            assert!(err.to_string().contains("anchor"), "the error for {anchor:?} must name `anchor`: {err}");
        }
    }

    #[test]
    fn crop_to_selection_and_trim() {
        let mut s = session();
        assert!(s.execute("image.crop", json!({})).is_err(), "no selection, no rect");
        s.execute("select.rect", json!({"x": 5, "y": 2, "width": 20, "height": 10})).unwrap();
        s.execute("image.crop", json!({})).unwrap();
        let d = doc(&s);
        assert_eq!(d.size, Size::new(20, 10));
        assert!(d.selection.is_none());
        assert_eq!(d.layers[1].surface().unwrap().content_bounds(), Rect::new(5, 3, 15, 10));
        // Trim transparent: hide the background, trim to the red square.
        let mut s = session();
        s.execute("layer.setProps", json!({"layer": doc(&s).layers[0].id.0, "visible": false})).unwrap();
        s.execute("image.trim", json!({"basedOn": "transparent"})).unwrap();
        // The red square (10×10) plus its drop shadow (distance 5, size 5).
        let size = doc(&s).size;
        assert!(size.width > 10 && size.width < 30 && size.height > 10 && size.height < 25, "{size:?}");
    }

    #[test]
    fn crop_to_explicit_rect_can_extend_canvas() {
        let mut s = session();
        s.execute("image.crop", json!({"x": 10, "y": 5, "width": 10, "height": 10})).unwrap();
        assert_eq!(doc(&s).size, Size::new(10, 10));
        assert_eq!(doc(&s).layers[1].surface().unwrap().pixel(0, 0), vec![1.0, 0.0, 0.0, 1.0]);
        let mut s = session();
        s.execute("image.crop", json!({"x": -10, "y": 0, "width": 60, "height": 20})).unwrap();
        assert_eq!(doc(&s).size, Size::new(60, 20));
        assert_eq!(doc(&s).layers[1].surface().unwrap().pixel(20, 5), vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn crop_extension_fills_the_background_with_the_background_colour() {
        let mut shrink = Session::new();
        shrink.execute("file.new", json!({"width": 16, "height": 16, "background": "#ff0000"})).unwrap();
        shrink.execute("image.crop", json!({"x": 2, "y": 0, "width": 12, "height": 16})).unwrap();
        let layer = &doc(&shrink).layers[0];
        assert_eq!(layer.name, "Background", "a shrink-only crop keeps the Background");
        assert!(layer.locks.transparency && layer.locks.position);

        let mut s = Session::new();
        s.execute("file.new", json!({"width": 16, "height": 16, "background": "#ff0000"})).unwrap();
        s.tools.background = [0.0, 1.0, 0.0, 1.0];
        s.execute("image.crop", json!({"x": -4, "y": 0, "width": 20, "height": 16})).unwrap();
        let layer = &doc(&s).layers[0];
        assert_eq!(layer.name, "Background", "an extending crop keeps the Background (#3081)");
        assert!(layer.locks.transparency && layer.locks.position);
        assert_eq!(layer.surface().unwrap().rgba(1, 8), [0.0, 1.0, 0.0, 1.0], "the added border takes the background colour");
        assert_eq!(layer.surface().unwrap().rgba(8, 8), [1.0, 0.0, 0.0, 1.0], "the original pixels stay");

        s.execute("edit.fill", json!({"color": "#0000ff"})).unwrap();
        let layer = &doc(&s).layers[0];
        assert_eq!(layer.surface().unwrap().rgba(1, 8), [0.0, 0.0, 1.0, 1.0], "Fill paints the extended border");
        assert_eq!(layer.surface().unwrap().rgba(8, 8), [0.0, 0.0, 1.0, 1.0]);

        s.execute("edit.undo", json!({})).unwrap();
        s.execute("edit.undo", json!({})).unwrap();
        let layer = &doc(&s).layers[0];
        assert_eq!(doc(&s).size, Size::new(16, 16));
        assert_eq!(layer.surface().unwrap().rgba(8, 8), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(layer.surface().unwrap().content_bounds(), Rect::new(0, 0, 16, 16), "undo removes the filled border");

        // A normal layer gets transparent pixels in the added area.
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 16, "height": 16, "background": "#ff0000"})).unwrap();
        s.execute("layer.new.layerFromBackground", json!({})).unwrap();
        s.execute("image.crop", json!({"x": -4, "y": 0, "width": 20, "height": 16})).unwrap();
        assert_eq!(doc(&s).layers[0].surface().unwrap().rgba(1, 8), [0.0; 4]);
    }

    /// A document (`mode`, `depth`) with a white Background and a layer holding a red pixel at
    /// (12, 8) and a blue one at (30, 3).
    fn marked(mode: &str, depth: u32) -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 40, "height": 20, "mode": mode, "depth": depth, "background": "white"})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        for (x, y, c) in [(12, 8, "#ff0000"), (30, 3, "#0000ff")] {
            s.execute("select.rect", json!({"x": x, "y": y, "width": 1, "height": 1})).unwrap();
            s.execute("edit.fill", json!({"color": c})).unwrap();
        }
        s.execute("select.deselect", json!({})).unwrap();
        s
    }

    fn layer_rgba(s: &Session, layer: usize, x: i32, y: i32) -> [f32; 4] {
        doc(s).layers[layer].surface().unwrap().rgba(x, y)
    }

    /// #1792: a frame turned 90° clockwise crops what Image Rotation › Arbitrary 90° CCW and a
    /// plain crop give, in one undo step that restores the document, at 8, 16 and 32 bits.
    #[test]
    fn crop_at_90_degrees_is_rotate_then_crop_in_one_step() {
        for depth in [8, 16, 32] {
            let mut s = marked("rgb", depth);
            let before = doc(&s).clone();
            let past = s.active().unwrap().history.past_len();
            // The 10 × 20 frame at (10, 0), turned 90° about (15, 10), covers x 5..25, y 5..15.
            let r = s.execute("image.crop", json!({"x": 10, "y": 0, "width": 10, "height": 20, "angle": 90})).unwrap();
            assert_eq!(r["angle"], json!(90.0), "{depth}");
            assert_eq!(doc(&s).size, Size::new(10, 20), "{depth}");
            assert_eq!(s.active().unwrap().history.past_len(), past + 1, "{depth}: one undo step");
            // (12, 8) is 2.5 left of and 1.5 above the centre: in the upright frame, 1.5 left of
            // and 2.5 below its centre (5, 10).
            let red = layer_rgba(&s, 1, 3, 12);
            assert!(red[0] > 0.9 && red[1] < 0.1 && red[3] > 0.9, "{depth}: {red:?}");
            let mut reference = marked("rgb", depth);
            reference.execute("image.rotation.arbitrary", json!({"angle": 90, "direction": "ccw"})).unwrap();
            reference.execute("image.crop", json!({"x": 5, "y": 15, "width": 10, "height": 20})).unwrap();
            for l in 0..2 {
                for y in 0..20 {
                    for x in 0..10 {
                        let (a, b) = (layer_rgba(&s, l, x, y), layer_rgba(&reference, l, x, y));
                        assert!(a.iter().zip(b).all(|(p, q)| (p - q).abs() < 1e-3), "{depth} layer {l} ({x}, {y}): {a:?} vs {b:?}");
                    }
                }
            }
            s.execute("edit.undo", json!({})).unwrap();
            let d = doc(&s);
            assert_eq!(d.size, before.size, "{depth}");
            assert_eq!(d.depth, before.depth);
            for y in 0..20 {
                for x in 0..40 {
                    assert_eq!(layer_rgba(&s, 1, x, y), before.layers[1].surface().unwrap().rgba(x, y), "{depth} undo at ({x}, {y})");
                }
            }
        }
    }

    /// A slightly turned frame keeps its own size, the pixel under its centre and the Background's
    /// colour; the turn's sign matters (a counter-clockwise frame is a negative angle).
    #[test]
    fn crop_at_a_small_angle_keeps_the_frame_size_in_every_mode() {
        for (mode, depth) in [("rgb", 8), ("rgb", 16), ("rgb", 32), ("grayscale", 16), ("cmyk", 8), ("lab", 16)] {
            for angle in [5.0, -7.5, 365.0] {
                let mut s = marked(mode, depth);
                s.execute("image.crop", json!({"x": 2, "y": 1, "width": 21, "height": 15, "angle": angle})).unwrap();
                let d = doc(&s);
                assert_eq!(d.size, Size::new(21, 15), "{mode} {depth} {angle}");
                assert_eq!(d.depth.bytes() * 8, depth as usize, "{mode} {depth}");
                assert_eq!(format!("{:?}", d.mode).to_lowercase(), mode, "{mode} {depth}");
                assert_eq!(d.layers.len(), 2);
                // The frame's centre (12.5, 8.5) is the red pixel (the layer's only paint nearby);
                // it stays at the centre, its colour kept.
                let want = layer_rgba(&marked(mode, depth), 1, 12, 8);
                let red = layer_rgba(&s, 1, 10, 7);
                assert!(red[3] > 0.5 && (0..3).all(|c| (red[c] - want[c]).abs() < 0.2), "{mode} {depth} {angle}: {red:?} vs {want:?}");
                // The Background stays opaque white inside the canvas it came from.
                let bg = layer_rgba(&s, 0, 10, 7);
                assert!(bg.iter().all(|v| (v - 1.0).abs() < 0.02), "{mode} {depth} {angle}: {bg:?}");
            }
        }
    }

    /// Angle 0 (or a whole turn) is the plain crop: no resampling, the same document.
    #[test]
    fn crop_at_angle_zero_is_the_plain_crop() {
        let mut plain = marked("rgb", 16);
        plain.execute("image.crop", json!({"x": 3, "y": 2, "width": 30, "height": 10})).unwrap();
        for angle in [json!(0), json!(0.0), json!(360), json!(-720), json!(null)] {
            let mut s = marked("rgb", 16);
            let r = s.execute("image.crop", json!({"x": 3, "y": 2, "width": 30, "height": 10, "angle": angle})).unwrap();
            assert!(r.get("angle").is_none(), "{angle}: {r}");
            assert_eq!(doc(&s).size, Size::new(30, 10));
            for l in 0..2 {
                for y in 0..10 {
                    for x in 0..30 {
                        assert_eq!(layer_rgba(&s, l, x, y), layer_rgba(&plain, l, x, y), "{angle} layer {l} ({x}, {y})");
                    }
                }
            }
        }
    }

    /// Bad angles fail cleanly, before anything changes.
    #[test]
    fn crop_angle_rejects_bad_values_without_changing_the_document() {
        let bad = [
            json!({"x": 0, "y": 0, "width": 10, "height": 10, "angle": "NaN"}),
            json!({"x": 0, "y": 0, "width": 10, "height": 10, "angle": "12"}),
            json!({"x": 0, "y": 0, "width": 10, "height": 10, "angle": [1]}),
            json!({"x": 0, "y": 0, "width": 10, "height": 10, "angle": true}),
            json!({"x": 0, "y": 0, "width": 10, "height": 10, "angle": 1e308}),
            json!({"x": 0, "y": 0, "width": 10, "height": 10, "angle": -3601}),
            json!({"x": 0, "y": 0, "width": 10, "height": 10, "angle": f64::MAX}),
            // An angle needs the frame: a selection crop has none.
            json!({"angle": 10}),
            json!({"x": 0, "y": 0, "width": 0, "height": 10, "angle": 10}),
            json!({"x": 0, "y": 0, "width": 300_001, "height": 10, "angle": 10}),
            json!({"x": i32::MAX - 5, "y": 0, "width": 10, "height": 10, "angle": 10}),
        ];
        for p in bad {
            let mut s = marked("rgb", 8);
            s.execute("select.all", json!({})).unwrap();
            let before = doc(&s).clone();
            let past = s.active().unwrap().history.past_len();
            assert!(s.execute("image.crop", p.clone()).is_err(), "{p}");
            assert_eq!(doc(&s).size, before.size, "{p}");
            assert_eq!(s.active().unwrap().history.past_len(), past, "{p}");
        }
        // Far-off but representable frames turn and crop without panicking.
        for (x, y) in [(-1_000_000, 0), (0, 1_000_000), (i32::MIN / 4, i32::MAX / 4)] {
            let mut s = marked("rgb", 8);
            s.execute("image.crop", json!({"x": x, "y": y, "width": 4, "height": 3, "angle": 33.3})).unwrap();
            assert_eq!(doc(&s).size, Size::new(4, 3));
        }
    }

    #[test]
    fn crop_refuses_a_canvas_side_past_the_new_document_limit() {
        // #960: an unbounded explicit crop left a canvas that a later Trim or Duplicate Merged
        // could only abort on allocating.
        for (w, h) in [(100_000_000, 100_000_000), (300_001, 10), (10, 300_001)] {
            let mut s = session();
            let (size, past) = (doc(&s).size, s.active().unwrap().history.past_len());
            let err = s.execute("image.crop", json!({"x": 0, "y": 0, "width": w, "height": h})).unwrap_err();
            assert!(err.to_string().contains("300000 px limit"), "{w} x {h}: {err}");
            assert_eq!((doc(&s).size, s.active().unwrap().history.past_len()), (size, past), "{w} x {h}: nothing changed");
        }
        // At the limit it still extends the canvas (the tiles stay lazy).
        let mut s = session();
        s.execute("image.crop", json!({"x": 0, "y": 0, "width": 300_000, "height": 2})).unwrap();
        assert_eq!(doc(&s).size, Size::new(300_000, 2));
    }

    /// #2443: the Crop tool's W x H x Resolution crops, resamples the cropped area to exactly the
    /// target size and sets the resolution, in one undo step, at 8 and 16 bits.
    #[test]
    fn crop_to_a_target_size_and_resolution_is_one_step() {
        for depth in [8, 16] {
            let mut s = Session::new();
            s.execute("file.new", json!({"width": 400, "height": 300, "resolution": 72, "depth": depth})).unwrap();
            s.execute("layer.new.layer", json!({})).unwrap();
            s.edit("paint", |doc, active| {
                doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap().fill_rect(Rect::new(100, 100, 200, 225), &[1.0, 0.0, 0.0, 1.0]);
                Ok(())
            })
            .unwrap();
            let past = s.active().unwrap().history.past_len();
            // 4 x 5 in at 300 ppi from a 100 x 125 px frame.
            let r = s
                .execute("image.crop", json!({"x": 100, "y": 100, "width": 100, "height": 125, "targetWidth": 1200, "targetHeight": 1500, "resolution": 300}))
                .unwrap();
            assert_eq!((r["documentWidth"].as_u64(), r["documentHeight"].as_u64()), (Some(1200), Some(1500)), "{depth}: {r}");
            assert_eq!(r["resolution"].as_f64(), Some(300.0), "{depth}");
            let d = doc(&s);
            assert_eq!(d.size, Size::new(1200, 1500), "{depth}");
            assert_eq!(d.resolution_dpi, 300.0, "{depth}");
            assert_eq!(d.layers[0].surface().unwrap().format().sample, if depth == 8 { SampleType::U8 } else { SampleType::U16 });
            // The red frame filled the crop; it fills the result.
            let red = d.layers[1].surface().unwrap();
            for (x, y) in [(0, 0), (600, 750), (1199, 1499)] {
                assert_eq!(red.pixel(x, y), vec![1.0, 0.0, 0.0, 1.0], "{depth}: ({x}, {y})");
            }
            assert_eq!(s.active().unwrap().history.past_len(), past + 1, "{depth}: one history step");
            s.execute("edit.undo", json!({})).unwrap();
            assert_eq!((doc(&s).size, doc(&s).resolution_dpi), (Size::new(400, 300), 72.0), "{depth}: undo restores the original");
        }
    }

    #[test]
    fn crop_target_resamples_after_a_turned_crop_and_resolution_alone_keeps_the_pixels() {
        let mut s = session();
        let past = s.active().unwrap().history.past_len();
        s.execute("image.crop", json!({"x": 10, "y": 0, "width": 10, "height": 20, "angle": 90, "targetWidth": 30, "targetHeight": 60, "resample": "nearest"}))
            .unwrap();
        assert_eq!(doc(&s).size, Size::new(30, 60));
        assert_eq!(s.active().unwrap().history.past_len(), past + 1, "rotate, crop and resample: one step");
        let mut s = session();
        s.execute("image.crop", json!({"x": 2, "y": 3, "width": 30, "height": 10, "resolution": 150})).unwrap();
        assert_eq!((doc(&s).size, doc(&s).resolution_dpi), (Size::new(30, 10), 150.0));
    }

    #[test]
    fn crop_target_rejects_bad_params_without_changing_the_document() {
        let rect = json!({"x": 0, "y": 0, "width": 10, "height": 10});
        let bad_params = [
            json!({"targetWidth": 100}),
            json!({"targetHeight": 100}),
            json!({"targetWidth": 0, "targetHeight": 10}),
            json!({"targetWidth": -5, "targetHeight": 10}),
            json!({"targetWidth": 10.5, "targetHeight": 10}),
            json!({"targetWidth": "10", "targetHeight": 10}),
            json!({"targetWidth": 300_001, "targetHeight": 10}),
            json!({"targetWidth": 4_294_967_306_u64, "targetHeight": 10}),
            json!({"targetWidth": 300_000, "targetHeight": 300_000}),
            json!({"targetWidth": 10, "targetHeight": 10, "resample": "none"}),
            json!({"targetWidth": 10, "targetHeight": 10, "resample": 3}),
            json!({"resolution": 0}),
            json!({"resolution": -72}),
            json!({"resolution": 1e9}),
            json!({"resolution": "300"}),
        ];
        for extra in bad_params {
            let mut s = session();
            let (size, dpi, past) = (doc(&s).size, doc(&s).resolution_dpi, s.active().unwrap().history.past_len());
            let mut p = rect.clone();
            if let (Some(o), Some(e)) = (p.as_object_mut(), extra.as_object()) {
                o.extend(e.clone());
            }
            assert!(s.execute("image.crop", p.clone()).is_err(), "{p}");
            assert_eq!((doc(&s).size, doc(&s).resolution_dpi, s.active().unwrap().history.past_len()), (size, dpi, past), "{p}: nothing changed");
        }
    }

    #[test]
    fn trim_based_on_top_left_colour() {
        let mut s = session();
        s.edit("no fx", |doc, _| {
            doc.layers[1].effects.items.clear();
            Ok(())
        })
        .unwrap();
        s.execute("image.trim", json!({"basedOn": "topLeft"})).unwrap();
        assert_eq!(doc(&s).size, Size::new(10, 10));
    }

    #[test]
    fn thirty_two_bit_holds_linear_light() {
        // Photoshop 27.11: an 8-bit sRGB grey 128 becomes 0.2159 in 32 bits (and 32 → 0.0144),
        // and comes back unchanged; alpha keeps its value.
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 4, "height": 1, "background": "white"})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.edit("paint", |doc, active| {
            let l = doc.layer_mut(active.unwrap()).unwrap();
            l.surface_mut().unwrap().fill_rect(Rect::new(0, 0, 2, 1), &[128.0 / 255.0, 32.0 / 255.0, 1.0, 0.5]);
            Ok(())
        })
        .unwrap();
        s.execute("image.mode.bits32", json!({})).unwrap();
        let px = doc(&s).layers[1].surface().unwrap().pixel(0, 0);
        for (got, want) in px.iter().zip([0.2159, 0.0144, 1.0, 128.0 / 255.0]) {
            assert!((got - want).abs() < 2e-4, "{px:?}");
        }
        s.execute("image.mode.bits8", json!({})).unwrap();
        let px = doc(&s).layers[1].surface().unwrap().pixel(0, 0);
        for (got, want) in px.iter().zip([128.0 / 255.0, 32.0 / 255.0, 1.0, 128.0 / 255.0]) {
            assert!((got - want).abs() < 0.6 / 255.0, "{px:?}");
        }
    }

    #[test]
    fn grayscale_conversion_matches_photoshop() {
        // Photoshop 27.11, default Color Settings (working gray Dot Gain 20 %): Image > Mode >
        // Grayscale turns sRGB 64 / 128 / 192 grey and pure red into 46 / 105 / 177 / 107.
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 4, "height": 1, "background": "white"})).unwrap();
        s.edit("paint", |doc, _| {
            let l = doc.layers.first_mut().unwrap();
            let surf = l.surface_mut().unwrap();
            for (x, c) in [[64.0, 64.0, 64.0], [128.0, 128.0, 128.0], [192.0, 192.0, 192.0], [255.0, 0.0, 0.0]].iter().enumerate() {
                surf.fill_rect(Rect::new(x as i32, 0, x as i32 + 1, 1), &[c[0] / 255.0, c[1] / 255.0, c[2] / 255.0, 1.0]);
            }
            Ok(())
        })
        .unwrap();
        s.execute("image.mode.grayscale", json!({})).unwrap();
        let surf = doc(&s).layers[0].surface().unwrap();
        for (x, want) in [46.0, 105.0, 177.0, 107.0].iter().enumerate() {
            let got = surf.pixel(x as i32, 0)[0] * 255.0;
            assert!((got - want).abs() <= 1.5, "pixel {x}: {got} vs {want}");
        }
    }

    #[test]
    fn mode_and_depth_conversions() {
        let mut s = session();
        s.execute("image.mode.grayscale", json!({})).unwrap();
        assert_eq!(doc(&s).mode, ColorMode::Grayscale);
        assert_eq!(doc(&s).layers[1].surface().unwrap().format().mode, ColorMode::Grayscale);
        s.execute("image.mode.bits16", json!({})).unwrap();
        assert_eq!(doc(&s).depth, SampleType::U16);
        assert_eq!(doc(&s).layers[1].surface().unwrap().format().sample, SampleType::U16);
        assert_eq!(doc(&s).layers[1].mask.as_ref().unwrap().surface.format().sample, SampleType::U16);
        s.execute("image.mode.cmyk", json!({})).unwrap();
        s.execute("image.mode.bits32", json!({})).unwrap();
        s.execute("image.mode.lab", json!({})).unwrap();
        s.execute("image.mode.rgb", json!({})).unwrap();
        s.execute("image.mode.bits8", json!({})).unwrap();
        let d = doc(&s);
        assert_eq!((d.mode, d.depth), (ColorMode::Rgb, SampleType::U8));
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(doc(&s).depth, SampleType::F32);
        let revision = s.active().unwrap().revision;
        s.execute("image.mode.bits32", json!({})).unwrap();
        assert_eq!(s.active().unwrap().revision, revision, "an unchanged bit depth must preserve the history branch");
        assert!(s.redo(), "selecting the current depth must not discard the undone conversion");
        assert_eq!(doc(&s).depth, SampleType::U8);
    }

    #[test]
    fn duplicate_document() {
        let mut s = session();
        let r = s.execute("image.duplicate", json!({"mergedOnly": true})).unwrap();
        assert_eq!(r["document"], json!(1));
        assert_eq!(s.documents().len(), 2);
        assert_eq!(doc(&s).layers.len(), 1);
        assert!(doc(&s).name.ends_with("copy"));
    }
}
