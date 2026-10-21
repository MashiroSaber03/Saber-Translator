//! `filter.*` commands (Photoshop's Filter menu) backed by `photocraft-algo`.
//!
//! Every command acts on the active pixel layer (or the cached pixels of a
//! smart object, recording a smart filter), respects the selection, and is
//! undoable. `filter.lastFilter` re-runs the most recent filter command.

use photocraft_algo::{
    self as algo, Distribution, FilterParams, PolarMode, Preserve, RadialMethod, RadialQuality, RippleSize, SpherizeMode, UndefinedAreas, WaveType,
};
use photocraft_doc::{LayerContent, SmartFilter};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

mod validation;
pub(crate) use validation::validate_params;

fn f(p: &Value, k: &str, d: f32) -> f32 {
    p.get(k).and_then(Value::as_f64).map_or(d, |v| v as f32)
}
fn i(p: &Value, k: &str, d: i32) -> i32 {
    // Clamped like the neighbouring filter params: an out-of-range value saturates instead of
    // wrapping through `as i32` (`3e9` used to come back as a negative offset).
    p.get(k).and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f.round() as i64))).map_or(d, |v| v.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32)
}
fn b(p: &Value, k: &str, d: bool) -> bool {
    p.get(k).and_then(Value::as_bool).unwrap_or(d)
}
fn s<'a>(p: &'a Value, k: &str, d: &'a str) -> &'a str {
    p.get(k).and_then(Value::as_str).unwrap_or(d)
}
fn seed(p: &Value) -> u32 {
    // New calls accept the entire documented u32 range; keep legacy record decoding otherwise.
    p.get("seed").and_then(Value::as_u64).and_then(|n| u32::try_from(n).ok()).unwrap_or_else(|| i(p, "seed", 0) as u32)
}
fn undefined(p: &Value) -> UndefinedAreas {
    match s(p, "undefinedAreas", "wrap") {
        "repeat" => UndefinedAreas::Repeat,
        "transparent" => UndefinedAreas::Transparent,
        _ => UndefinedAreas::Wrap,
    }
}

/// Minimum / Maximum › Preserve (Photoshop's default is Squareness).
fn preserve(p: &Value) -> Preserve {
    if s(p, "preserve", "squareness") == "roundness" { Preserve::Roundness } else { Preserve::Squareness }
}

/// One-step presets (no dialog): the history step they record (their own name, not the generic
/// algorithm's) and the fixed parameters, tuned to match the reference app's fixed-strength filters.
fn preset(id: &str) -> Option<(&'static str, FilterParams)> {
    Some(match id {
        "filter.blur.blur" => ("Blur", FilterParams::GaussianBlur { radius: 0.6 }),
        // Photoshop 27.11's Blur More is the 3×3 kernel [1 2 1; 2 2 2; 1 2 1] / 14 (its response
        // to a single white pixel, exactly).
        "filter.blur.blurMore" => (
            "Blur More",
            FilterParams::Custom {
                kernel: vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 1.0, 2.0, 1.0, 0.0, //
                    0.0, 2.0, 2.0, 2.0, 0.0, //
                    0.0, 1.0, 2.0, 1.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                scale: 14.0,
                offset: 0.0,
            },
        ),
        "filter.sharpen.sharpen" => ("Sharpen", FilterParams::UnsharpMask { amount: 60.0, radius: 0.5, threshold: 0.0 }),
        "filter.sharpen.sharpenMore" => ("Sharpen More", FilterParams::UnsharpMask { amount: 150.0, radius: 0.6, threshold: 0.0 }),
        "filter.sharpen.sharpenEdges" => ("Sharpen Edges", FilterParams::UnsharpMask { amount: 100.0, radius: 0.8, threshold: 6.0 }),
        "filter.noise.despeckle" => ("Despeckle", FilterParams::SurfaceBlur { radius: 2.0, threshold: 12.0 }),
        _ => return None,
    })
}

/// Builds the algorithm parameters for a filter command id from JSON params
/// (Photoshop dialog units). Stored smart filters and previews retain their tolerant decoding;
/// new command invocations pass through `validate_params` before any document changes.
pub fn params_for(id: &str, p: &Value) -> Option<FilterParams> {
    if let Some((_, fp)) = preset(id) {
        return Some(fp);
    }
    Some(match id {
        "filter.blur.gaussianBlur" => FilterParams::GaussianBlur { radius: f(p, "radius", 1.0).clamp(0.1, 1000.0) },
        "filter.blur.boxBlur" => FilterParams::BoxBlur { radius: f(p, "radius", 1.0).clamp(1.0, 2000.0) },
        "filter.blur.motionBlur" => FilterParams::MotionBlur { angle: f(p, "angle", 0.0), distance: f(p, "distance", 10.0).clamp(1.0, 2000.0) },
        "filter.blur.radialBlur" => FilterParams::RadialBlur {
            amount: f(p, "amount", 10.0).clamp(1.0, 100.0),
            method: if s(p, "method", "spin") == "zoom" { RadialMethod::Zoom } else { RadialMethod::Spin },
            quality: match s(p, "quality", "good") {
                "draft" => RadialQuality::Draft,
                "best" => RadialQuality::Best,
                "good" => RadialQuality::Good,
                _ => RadialQuality::Good,
            },
            center_x: f(p, "centerX", 0.5),
            center_y: f(p, "centerY", 0.5),
        },
        "filter.blur.surfaceBlur" => {
            FilterParams::SurfaceBlur { radius: f(p, "radius", 5.0).clamp(1.0, 100.0), threshold: f(p, "threshold", 15.0).clamp(2.0, 255.0) }
        }
        "filter.sharpen.unsharpMask" => FilterParams::UnsharpMask {
            amount: f(p, "amount", 50.0).clamp(1.0, 500.0),
            radius: f(p, "radius", 1.0).clamp(0.1, 1000.0),
            threshold: f(p, "threshold", 0.0).clamp(0.0, 255.0),
        },
        "filter.sharpen.smartSharpen" => FilterParams::SmartSharpen {
            amount: f(p, "amount", 100.0).clamp(1.0, 500.0),
            radius: f(p, "radius", 1.0).clamp(0.1, 64.0),
            reduce_noise: f(p, "reduceNoise", 10.0).clamp(0.0, 100.0),
        },
        "filter.other.highPass" => FilterParams::HighPass { radius: f(p, "radius", 10.0).clamp(0.1, 1000.0) },
        "filter.noise.addNoise" => FilterParams::AddNoise {
            amount: f(p, "amount", 12.5).clamp(0.1, 400.0),
            distribution: if s(p, "distribution", "uniform") == "gaussian" { Distribution::Gaussian } else { Distribution::Uniform },
            monochromatic: b(p, "monochromatic", false),
            seed: seed(p),
        },
        "filter.noise.median" => FilterParams::Median { radius: f(p, "radius", 1.0).clamp(1.0, 500.0) },
        "filter.noise.dustAndScratches" => {
            FilterParams::DustAndScratches { radius: f(p, "radius", 1.0).clamp(1.0, 500.0), threshold: f(p, "threshold", 0.0).clamp(0.0, 255.0) }
        }
        "filter.other.minimum" => FilterParams::Minimum { radius: f(p, "radius", 1.0).clamp(0.2, 500.0), preserve: preserve(p) },
        "filter.other.maximum" => FilterParams::Maximum { radius: f(p, "radius", 1.0).clamp(0.2, 500.0), preserve: preserve(p) },
        "filter.other.offset" => FilterParams::Offset { horizontal: i(p, "horizontal", 0), vertical: i(p, "vertical", 0), undefined: undefined(p) },
        "filter.pixelate.mosaic" => FilterParams::Mosaic { cell_size: f(p, "cellSize", 10.0).clamp(2.0, 200.0) },
        "filter.stylize.emboss" => {
            FilterParams::Emboss { angle: f(p, "angle", 135.0), height: f(p, "height", 3.0).clamp(1.0, 100.0), amount: f(p, "amount", 100.0).clamp(1.0, 500.0) }
        }
        "filter.stylize.findEdges" => FilterParams::FindEdges,
        "filter.stylize.solarize" => FilterParams::Solarize,
        "filter.distort.twirl" => FilterParams::Twirl { angle: f(p, "angle", 50.0).clamp(-999.0, 999.0) },
        "filter.distort.pinch" => FilterParams::Pinch { amount: f(p, "amount", 50.0).clamp(-100.0, 100.0) },
        "filter.distort.spherize" => FilterParams::Spherize {
            amount: f(p, "amount", 100.0).clamp(-100.0, 100.0),
            mode: match s(p, "mode", "normal") {
                "horizontalOnly" => SpherizeMode::HorizontalOnly,
                "verticalOnly" => SpherizeMode::VerticalOnly,
                _ => SpherizeMode::Normal,
            },
        },
        "filter.distort.wave" => FilterParams::Wave {
            generators: i(p, "generators", 5).clamp(1, 999) as u32,
            wavelength_min: f(p, "wavelengthMin", 10.0).clamp(1.0, 998.0),
            wavelength_max: f(p, "wavelengthMax", 120.0).clamp(2.0, 999.0),
            amplitude_min: f(p, "amplitudeMin", 5.0).clamp(1.0, 998.0),
            amplitude_max: f(p, "amplitudeMax", 35.0).clamp(1.0, 999.0),
            wave_type: match s(p, "type", "sine") {
                "triangle" => WaveType::Triangle,
                "square" => WaveType::Square,
                _ => WaveType::Sine,
            },
            undefined: undefined(p),
            seed: seed(p),
        },
        "filter.distort.ripple" => FilterParams::Ripple {
            amount: f(p, "amount", 100.0).clamp(-999.0, 999.0),
            size: match s(p, "size", "medium") {
                "small" => RippleSize::Small,
                "large" => RippleSize::Large,
                _ => RippleSize::Medium,
            },
        },
        "filter.distort.polarCoordinates" => FilterParams::PolarCoordinates {
            mode: if s(p, "mode", "rectangularToPolar") == "polarToRectangular" { PolarMode::PolarToRectangular } else { PolarMode::RectangularToPolar },
        },
        _ => return crate::filters_ext::params_for(id, p),
    })
}

/// Runs filter command `id` with JSON `params` on `surf` (the pixel function behind every
/// `filter.*` command; smart filters re-run it on re-render). `bounds` frames distortions;
/// `selection` limits the result. Like layer filters, neighbourhood filters repeat the edge pixels
/// of `canvas` ∪ the surface's content instead of fading in transparency, and the output is
/// clipped to it. `None` for ids that aren't pixel filters.
pub fn apply_filter_to_surface(
    id: &str,
    params: &Value,
    surf: &photocraft_raster::Surface,
    bounds: photocraft_geom::Rect,
    selection: Option<&photocraft_raster::Surface>,
    canvas: photocraft_geom::Rect,
) -> Option<photocraft_raster::Surface> {
    // Liquify / Puppet Warp / Perspective Warp smart filters (distort_cmds).
    if let Some(out) = crate::distort_cmds::apply_to_surface(id, params, surf, canvas) {
        return Some(out);
    }
    // Lens Correction / Adaptive Wide Angle / Camera Raw smart filters (lens_cmds).
    if let Some(out) = crate::lens_cmds::apply_to_surface(id, params, surf, canvas) {
        return Some(out);
    }
    // Image › Adjustments recorded as smart filters (Levels, Curves, Shadows/Highlights… from PSD).
    if let Some(kind) = id.strip_prefix("image.adjustments.") {
        return crate::adjust_cmds::adjust_as_filter(kind, params, surf);
    }
    // WebAssembly plug-in smart filters (plugin_cmds).
    if id == crate::plugin_cmds::RUN {
        return crate::plugin_cmds::apply_to_surface(id, params, surf, selection, canvas);
    }
    let mut fp = params_for(id, params)?;
    // A map that can no longer be read (moved or deleted since) leaves the pixels alone, like an
    // unknown filter; the command itself already refused an unreadable map when it was added.
    crate::filters_ext::resolve_map_path(&mut fp, params).ok()?;
    let sel_bounds = selection.map(photocraft_raster::Surface::content_bounds);
    let content = surf.content_bounds();
    let area = algo::output_area(&fp, content, bounds, sel_bounds);
    Some(algo::apply_in(surf, &fp, area, bounds, selection, canvas.union(&content)))
}

pub(crate) fn has_filterable_layer(s: &Session) -> std::result::Result<(), String> {
    // A targeted alpha channel or Quick Mask is filtered instead of the layer (channel_cmds).
    if crate::channel_cmds::edits_channel(s) {
        return Ok(());
    }
    let d = s.active().ok_or("no document open")?;
    let id = d.active_layer.ok_or("no active layer")?;
    let l = d.doc.layer(id).ok_or("no active layer")?;
    match &l.content {
        LayerContent::Raster(_) => Ok(()),
        LayerContent::Smart(sm) if sm.cache.is_some() => Ok(()),
        other => Err(format!("filters need a pixel layer (active layer is {} {} layer)", other.article(), other.kind_name())),
    }
}

pub(crate) fn run_filter(s: &mut Session, id: &str, p: &Value) -> Result<Value> {
    validate_params(id, p)?;
    // Session state some filters read (colours) becomes explicit params; external inputs resolve here.
    let prepared = crate::filters_ext::prepare(s, id, p);
    let p = &prepared;
    let mut fp = params_for(id, p).ok_or_else(|| EngineError::Other(format!("unknown filter {id}")))?;
    crate::filters_ext::resolve(s, &mut fp, p)?;
    let layer = match p.get("layer").and_then(Value::as_u64) {
        Some(l) => photocraft_doc::LayerId(l),
        None => s.active().and_then(|d| d.active_layer).ok_or(EngineError::Other("no active layer".into()))?,
    };
    let label = preset(id).map_or(fp.label(), |(name, _)| name).to_string();
    let msg = label.clone();
    let mut params = p.clone();
    if let Value::Object(m) = &mut params {
        m.remove("__kind");
    }
    let id = id.to_string();
    let p = p.clone();
    let layer_id = layer.0;
    // A background job when started with `Session::start` (#210): the filter runs on a worker,
    // checking for cancellation before every tile.
    crate::jobs::edit_job(
        s,
        &label,
        move |doc, _, ctx| {
            let filter = |surf: &photocraft_raster::Surface, fp: &FilterParams, area, bounds, sel: Option<&photocraft_raster::Surface>, extent| {
                ctx.stage(0.0, 1.0, &msg, |ctl| algo::apply_in_with(surf, fp, area, bounds, sel, extent, ctl)).ok_or(EngineError::Cancelled)
            };
            let selection = doc.selection.clone();
            let sel_bounds = selection.as_ref().map(photocraft_raster::Surface::content_bounds);
            // Distortions centre on the selection when there is one, else the canvas.
            let doc_bounds = doc.bounds();
            let bounds = sel_bounds.filter(|b| !b.is_empty()).unwrap_or(doc_bounds);
            // Neighbourhood filters repeat edge pixels at the canvas edge (or past it, where the
            // layer has off-canvas pixels) instead of fading in transparency, as Photoshop does.
            if let Some(surf) = crate::channel_cmds::channel_surface_for_filter(doc, Some(layer), &p)? {
                let content = doc_bounds.union(&surf.content_bounds());
                let area = algo::output_area(&fp, content, bounds, sel_bounds);
                *surf = filter(surf, &fp, area, bounds, selection.as_ref(), doc_bounds.union(&content))?;
                return Ok(fp.clone());
            }
            // Filters that make pixels transparent (Color to Alpha) turn the Background into a
            // normal layer first, as the erasers do: it can't hold transparency.
            if fp.makes_transparency() {
                crate::extra_cmds::background_to_layer_for_mask(doc, layer);
            }
            let locks = doc.effective_locks(layer);
            let l = doc.layer_mut(layer).ok_or(EngineError::NoLayer(layer))?;
            if locks.pixels || locks.all {
                return Err(EngineError::Other(format!("Could not complete your request because the layer \"{}\" is locked", l.name)));
            }
            // Putting the old alpha back (below) would leave unmixed colours fully opaque.
            if locks.transparency && fp.makes_transparency() && matches!(l.content, LayerContent::Raster(_)) {
                return Err(EngineError::Other(format!("Could not complete your request because the transparency of layer \"{}\" is locked", l.name)));
            }
            let mut fp = fp.clone();
            crate::filters_ext::resolve_in_layer(&mut fp, l, bounds);
            let surf = match &mut l.content {
                LayerContent::Raster(surf) => surf,
                LayerContent::Smart(_) => {
                    // Non-destructive: record the filter and re-render the smart object from its source.
                    let sf =
                        SmartFilter { command: id.clone(), params: params.clone(), blend: photocraft_color::BlendMode::Normal, opacity: 1.0, visible: true };
                    return crate::smart_cmds::add_smart_filter(doc, layer, sf, selection.as_ref()).map(|()| fp);
                }
                _ => return Err(EngineError::Other("not a pixel layer".into())),
            };
            let before = locks.transparency.then(|| surf.clone());
            let content = surf.content_bounds();
            let area = algo::output_area(&fp, content, bounds, sel_bounds);
            *surf = filter(surf, &fp, area, bounds, selection.as_ref(), doc_bounds.union(&content))?;
            if let Some(before) = &before {
                keep_alpha(before, surf);
            }
            Ok(fp)
        },
        move |fp| json!({ "layer": layer_id, "filter": serde_json::to_value(&fp).unwrap_or(Value::Null) }),
    )
}

/// Lock transparency: put the layer's alpha back after a filter, in the tiles it changed.
fn keep_alpha(old: &photocraft_raster::Surface, new: &mut photocraft_raster::Surface) {
    if old.format() != new.format() || !new.format().alpha {
        return;
    }
    let mut coords: Vec<_> = new.tiles().filter(|(c, t)| old.tile(**c).is_none_or(|o| !std::sync::Arc::ptr_eq(o, t))).map(|(c, _)| *c).collect();
    coords.extend(old.tiles().filter(|(c, _)| new.tile(**c).is_none()).map(|(c, _)| *c));
    let n = new.channels();
    for c in coords {
        let r = c.rect();
        let o = old.read_region(r);
        let mut v = new.read_region(r);
        for (pn, po) in v.chunks_exact_mut(n).zip(o.chunks_exact(n)) {
            if let (Some(a), Some(b)) = (pn.last_mut(), po.last()) {
                *a = *b;
            }
        }
        new.write_region(r, &v);
    }
    new.prune();
}

macro_rules! filter_cmd {
    ($id:literal, $label:literal, [$($m:literal),*], $params:literal) => {
        CommandSpec {
            id: $id,
            label: $label,
            menu: &[$($m),*],
            shortcut: None,
            params: $params,
            enabled: has_filterable_layer,
            run: |s, p| run_filter(s, $id, p),
            journal: true,
        }
    };
}

/// The filter command specs.
pub fn specs() -> Vec<CommandSpec> {
    vec![
        filter_cmd!("filter.blur.gaussianBlur", "Gaussian Blur…", ["Filter", "Blur"], r##"{"radius":0.1..1000=1}"##),
        filter_cmd!("filter.blur.blur", "Blur", ["Filter", "Blur"], "{}"),
        filter_cmd!("filter.blur.blurMore", "Blur More", ["Filter", "Blur"], "{}"),
        filter_cmd!("filter.sharpen.sharpen", "Sharpen", ["Filter", "Sharpen"], "{}"),
        filter_cmd!("filter.sharpen.sharpenMore", "Sharpen More", ["Filter", "Sharpen"], "{}"),
        filter_cmd!("filter.sharpen.sharpenEdges", "Sharpen Edges", ["Filter", "Sharpen"], "{}"),
        filter_cmd!("filter.noise.despeckle", "Despeckle", ["Filter", "Noise"], "{}"),
        filter_cmd!("filter.blur.boxBlur", "Box Blur…", ["Filter", "Blur"], r##"{"radius":1..2000=1}"##),
        filter_cmd!("filter.blur.motionBlur", "Motion Blur…", ["Filter", "Blur"], r##"{"angle":-360..360=0,"distance":1..2000=10}"##),
        filter_cmd!(
            "filter.blur.radialBlur",
            "Radial Blur…",
            ["Filter", "Blur"],
            r##"{"amount":1..100=10,"method":"spin|zoom","quality":"draft|good|best"="good","centerX":0..1=0.5,"centerY":0..1=0.5}"##
        ),
        filter_cmd!("filter.blur.surfaceBlur", "Surface Blur…", ["Filter", "Blur"], r##"{"radius":1..100=5,"threshold":2..255=15}"##),
        filter_cmd!(
            "filter.sharpen.unsharpMask",
            "Unsharp Mask…",
            ["Filter", "Sharpen"],
            r##"{"amount":1..500=50,"radius":0.1..1000=1,"threshold":0..255=0}"##
        ),
        filter_cmd!(
            "filter.sharpen.smartSharpen",
            "Smart Sharpen…",
            ["Filter", "Sharpen"],
            r##"{"amount":1..500=100,"radius":0.1..64=1,"reduceNoise":0..100=10}"##
        ),
        filter_cmd!(
            "filter.noise.addNoise",
            "Add Noise…",
            ["Filter", "Noise"],
            r##"{"amount":0.1..400=12.5,"distribution":"uniform|gaussian","monochromatic":bool,"seed":u32=0}"##
        ),
        filter_cmd!("filter.noise.median", "Median…", ["Filter", "Noise"], r##"{"radius":1..500=1}"##),
        filter_cmd!("filter.noise.dustAndScratches", "Dust & Scratches…", ["Filter", "Noise"], r##"{"radius":1..500=1,"threshold":0..255=0}"##),
        filter_cmd!("filter.pixelate.mosaic", "Mosaic…", ["Filter", "Pixelate"], r##"{"cellSize":2..200=10}"##),
        filter_cmd!("filter.stylize.emboss", "Emboss…", ["Filter", "Stylize"], r##"{"angle":-180..180=135,"height":1..100=3,"amount":1..500=100}"##),
        filter_cmd!("filter.stylize.findEdges", "Find Edges", ["Filter", "Stylize"], "{}"),
        filter_cmd!("filter.stylize.solarize", "Solarize", ["Filter", "Stylize"], "{}"),
        filter_cmd!("filter.distort.twirl", "Twirl…", ["Filter", "Distort"], r##"{"angle":-999..999=50}"##),
        filter_cmd!("filter.distort.pinch", "Pinch…", ["Filter", "Distort"], r##"{"amount":-100..100=50}"##),
        filter_cmd!("filter.distort.spherize", "Spherize…", ["Filter", "Distort"], r##"{"amount":-100..100=100,"mode":"normal|horizontalOnly|verticalOnly"}"##),
        filter_cmd!(
            "filter.distort.wave",
            "Wave…",
            ["Filter", "Distort"],
            r##"{"generators":1..999=5,"wavelengthMin":1..998=10,"wavelengthMax":2..999=120,"amplitudeMin":1..998=5,"amplitudeMax":1..999=35,"type":"sine|triangle|square","undefinedAreas":"wrap|repeat","seed":u32=0}"##
        ),
        filter_cmd!("filter.distort.ripple", "Ripple…", ["Filter", "Distort"], r##"{"amount":-999..999=100,"size":"small|medium|large"}"##),
        filter_cmd!("filter.distort.polarCoordinates", "Polar Coordinates…", ["Filter", "Distort"], r##"{"mode":"rectangularToPolar|polarToRectangular"}"##),
        filter_cmd!("filter.other.highPass", "High Pass…", ["Filter", "Other"], r##"{"radius":0.1..1000=10}"##),
        filter_cmd!("filter.other.minimum", "Minimum…", ["Filter", "Other"], r##"{"radius":0.2..500=1,"preserve":"squareness|roundness"}"##),
        filter_cmd!("filter.other.maximum", "Maximum…", ["Filter", "Other"], r##"{"radius":0.2..500=1,"preserve":"squareness|roundness"}"##),
        filter_cmd!(
            "filter.other.offset",
            "Offset…",
            ["Filter", "Other"],
            r##"{"horizontal":px=0,"vertical":px=0,"undefinedAreas":"wrap|repeat|transparent"}"##
        ),
        CommandSpec {
            id: "filter.lastFilter",
            label: "Last Filter",
            menu: &["Filter"],
            shortcut: Some("Cmd+Alt+F"),
            params: "{}",
            enabled: |s| {
                has_filterable_layer(s)?;
                last_filter(s).map(|_| ()).ok_or_else(|| "no filter applied yet".into())
            },
            run: |s, _| {
                let (id, params) = last_filter(s).ok_or_else(|| EngineError::Other("no filter applied yet".into()))?;
                run_filter(s, &id, &params)
            },
            journal: true,
        },
    ]
}

/// The most recent filter command in the session journal, retargeted at the active layer: dialogs
/// such as the Filter Gallery record the layer they ran on, and Last Filter applies to the
/// current layer, as in Photoshop (#1270).
pub fn last_filter(s: &Session) -> Option<(String, Value)> {
    let (id, mut params) = s.journal.iter().rev().find(|(id, _)| params_for(id, &Value::Null).is_some() && id.starts_with("filter.")).cloned()?;
    if let Value::Object(m) = &mut params {
        m.remove("layer");
    }
    Some((id, params))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 48, "height": 32})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("edit.fill", json!({"color": "#3366cc"})).ok();
        s
    }

    fn active_pixels(s: &Session) -> Vec<f32> {
        let d = s.active().unwrap();
        let l = d.doc.layer(d.active_layer.unwrap()).unwrap();
        l.surface().unwrap().read_region(photocraft_geom::Rect::new(0, 0, 48, 32))
    }

    fn mask_session(reveal_selection: bool) -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 16, "height": 16, "background": "#ff0000"})).unwrap();
        if reveal_selection {
            s.execute("select.all", json!({})).unwrap();
            s.execute("layer.layerMask.revealSelection", json!({})).unwrap();
        } else {
            s.execute("layer.layerMask.revealAll", json!({})).unwrap();
        }
        s
    }

    fn mask_pixels(s: &Session) -> Vec<f32> {
        let d = s.active().unwrap();
        let mask = d.doc.layer(d.active_layer.unwrap()).unwrap().mask.as_ref().unwrap();
        mask.surface.read_region(d.doc.bounds())
    }

    fn canvas_pixels(s: &Session) -> Vec<f32> {
        let d = s.active().unwrap();
        d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap().read_region(d.doc.bounds())
    }

    fn paint_pattern(s: &mut Session) {
        s.edit("pattern", |doc, active| {
            let l = doc.layer_mut(active.unwrap()).unwrap();
            let surf = l.surface_mut().unwrap();
            for y in 0..32 {
                for x in 0..48 {
                    let v = ((x * 5 + y * 3) % 17) as f32 / 16.0;
                    surf.write_pixel(x, y, &[v, 1.0 - v, (x % 3) as f32 / 2.0, 1.0]);
                }
            }
            Ok(())
        })
        .unwrap();
    }

    const ALL: [&str; 24] = [
        "filter.blur.gaussianBlur",
        "filter.blur.boxBlur",
        "filter.blur.motionBlur",
        "filter.blur.radialBlur",
        "filter.blur.surfaceBlur",
        "filter.sharpen.unsharpMask",
        "filter.sharpen.smartSharpen",
        "filter.noise.addNoise",
        "filter.noise.median",
        "filter.noise.dustAndScratches",
        "filter.pixelate.mosaic",
        "filter.stylize.emboss",
        "filter.stylize.findEdges",
        "filter.stylize.solarize",
        "filter.distort.twirl",
        "filter.distort.pinch",
        "filter.distort.spherize",
        "filter.distort.wave",
        "filter.distort.ripple",
        "filter.distort.polarCoordinates",
        "filter.other.highPass",
        "filter.other.minimum",
        "filter.other.maximum",
        "filter.other.offset",
    ];

    #[test]
    fn every_filter_command_runs_changes_pixels_and_undoes() {
        for id in ALL {
            let mut s = session();
            paint_pattern(&mut s);
            let before = active_pixels(&s);
            let params = match id {
                "filter.other.offset" => json!({"horizontal": 5, "vertical": 2}),
                "filter.noise.dustAndScratches" => json!({"radius": 2, "threshold": 0}),
                "filter.noise.median" => json!({"radius": 2}),
                _ => json!({}),
            };
            let r = s.execute(id, params).unwrap_or_else(|e| panic!("{id}: {e}"));
            assert!(r.get("filter").is_some(), "{id}");
            assert_ne!(active_pixels(&s), before, "{id} changed nothing");
            s.execute("edit.undo", json!({})).unwrap();
            assert_eq!(active_pixels(&s), before, "{id} undo");
        }
    }

    #[test]
    fn preset_filters_record_their_own_name() {
        // #528: one-step presets run a generic algorithm but name the step after themselves;
        // the generic commands keep the algorithm's name.
        let specs = specs();
        for (id, name) in [
            ("filter.blur.blur", "Blur"),
            ("filter.blur.blurMore", "Blur More"),
            ("filter.sharpen.sharpen", "Sharpen"),
            ("filter.sharpen.sharpenMore", "Sharpen More"),
            ("filter.sharpen.sharpenEdges", "Sharpen Edges"),
            ("filter.noise.despeckle", "Despeckle"),
            ("filter.blur.gaussianBlur", "Gaussian Blur"),
            ("filter.sharpen.unsharpMask", "Unsharp Mask"),
            ("filter.blur.surfaceBlur", "Surface Blur"),
        ] {
            let mut s = session();
            let steps = s.active().unwrap().history.past_len();
            s.execute(id, json!({})).unwrap();
            let h = &s.active().unwrap().history;
            assert_eq!((h.past_len(), h.undo_label()), (steps + 1, Some(name)), "{id}");
            let label = specs.iter().find(|c| c.id == id).unwrap().label;
            assert_eq!(label.trim_end_matches('…'), name, "{id}: the step is named like the command");
        }
    }

    #[test]
    fn specs_are_complete_and_documented() {
        let specs = specs();
        for id in ALL {
            let spec = specs.iter().find(|c| c.id == id).unwrap_or_else(|| panic!("missing {id}"));
            assert!(spec.params.starts_with('{'), "{id}");
            assert!(!spec.menu.is_empty(), "{id}");
        }
    }

    #[test]
    fn disabled_without_pixel_layer() {
        let mut s = Session::new();
        assert!(!s.is_enabled("filter.blur.gaussianBlur"));
        s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
        s.execute("layer.newAdjustmentLayer.invert", json!({})).unwrap();
        assert!(matches!(s.execute("filter.blur.gaussianBlur", json!({})), Err(EngineError::Disabled(..))));
    }

    #[test]
    fn selection_limits_the_filter() {
        let mut s = session();
        paint_pattern(&mut s);
        s.execute("select.rect", json!({"x": 0, "y": 0, "width": 10, "height": 10})).unwrap();
        let has_sel = s.active().unwrap().doc.selection.is_some();
        assert!(has_sel);
        let before = active_pixels(&s);
        s.execute("filter.blur.gaussianBlur", json!({"radius": 3})).unwrap();
        let after = active_pixels(&s);
        if has_sel {
            // A pixel far from the selection is unchanged.
            let i = (30 * 48 + 40) * 4;
            assert_eq!(after[i..i + 4], before[i..i + 4]);
            let j = (5 * 48 + 5) * 4;
            assert_ne!(after[j..j + 4], before[j..j + 4]);
        }
    }

    #[test]
    fn solarize_and_add_noise_process_untouched_reveal_all_masks() {
        for (id, params) in [
            ("filter.stylize.solarize", json!({"target": "mask"})),
            ("filter.noise.addNoise", json!({"amount": 100, "seed": 1, "monochromatic": true, "target": "mask"})),
        ] {
            let mut reveal_all = mask_session(false);
            let mut reveal_selection = mask_session(true);
            let all_mask = reveal_all.active().unwrap().doc.layer(reveal_all.active().unwrap().active_layer.unwrap()).unwrap().mask.as_ref().unwrap();
            assert!(all_mask.surface.content_bounds().is_empty(), "{id}: Reveal All stores its white canvas in the default pixel");
            let initial_all = mask_pixels(&reveal_all);
            let initial_selection = mask_pixels(&reveal_selection);
            assert_eq!(initial_all, initial_selection, "{id}: masks start with equal white values");
            assert_eq!(initial_all, vec![1.0; 16 * 16], "{id}: the full canvas starts revealed");
            let pixels_before = canvas_pixels(&reveal_all);

            reveal_all.execute(id, params.clone()).unwrap();
            reveal_selection.execute(id, params).unwrap();

            let actual = mask_pixels(&reveal_all);
            let expected = mask_pixels(&reveal_selection);
            assert!(
                actual == expected,
                "{id}: Reveal All differs from Reveal Selection: first={:?}/{:?}, changed pixels={}/{}",
                actual[0],
                expected[0],
                actual.iter().filter(|&&v| v != 1.0).count(),
                expected.iter().filter(|&&v| v != 1.0).count()
            );
            assert_eq!(canvas_pixels(&reveal_all), pixels_before, "{id}: the mask target must leave layer pixels unchanged");
            if id == "filter.stylize.solarize" {
                assert!(actual.iter().all(|&v| v == 0.0), "Solarize maps the white mask to black");
            } else {
                assert!(actual.iter().any(|&v| v != 1.0), "seeded Add Noise changes the Reveal All mask");
            }
        }
    }

    #[test]
    fn minimum_signed_offset_matches_its_wrapped_distance() {
        for depth in [8, 16, 32] {
            let make_session = || {
                let mut s = Session::new();
                s.execute("file.new", json!({"width": 48, "height": 32, "depth": depth})).unwrap();
                s.execute("layer.new.layer", json!({})).unwrap();
                paint_pattern(&mut s);
                s
            };
            let mut actual = make_session();
            let before = active_pixels(&actual);
            actual.execute("filter.other.offset", json!({"horizontal": i32::MIN, "vertical": i32::MIN})).unwrap();
            let mut expected = make_session();
            expected.execute("filter.other.offset", json!({"horizontal": i32::MIN.rem_euclid(48), "vertical": i32::MIN.rem_euclid(32)})).unwrap();
            let after = active_pixels(&actual);
            assert_eq!(after, active_pixels(&expected), "depth {depth}");
            assert_ne!(after, before);
            actual.execute("edit.undo", json!({})).unwrap();
            assert_eq!(active_pixels(&actual), before);
            actual.execute("edit.redo", json!({})).unwrap();
            assert_eq!(active_pixels(&actual), after);
        }
    }
    #[test]
    fn huge_offsets_clamp_instead_of_wrapping() {
        // 3e9 wrapped to a negative offset through `as i32`; it saturates now, so it
        // equals the maximal in-range offset instead of shifting the other way.
        let mut a = session();
        a.execute("filter.other.offset", json!({"horizontal": 3_000_000_000_i64, "vertical": 0})).unwrap();
        let mut b = session();
        b.execute("filter.other.offset", json!({"horizontal": 2_147_483_647_i64, "vertical": 0})).unwrap();
        assert_eq!(active_pixels(&a), active_pixels(&b));
    }

    #[test]
    fn last_filter_repeats_with_same_params() {
        let mut s = session();
        paint_pattern(&mut s);
        assert!(!s.is_enabled("filter.lastFilter"));
        s.execute("filter.other.offset", json!({"horizontal": 3})).unwrap();
        let once = active_pixels(&s);
        s.execute("filter.lastFilter", json!({})).unwrap();
        let twice = active_pixels(&s);
        assert_ne!(once, twice);
        // Offsetting 3 twice equals offsetting 6 once.
        let mut s2 = session();
        paint_pattern(&mut s2);
        s2.execute("filter.other.offset", json!({"horizontal": 6})).unwrap();
        assert_eq!(active_pixels(&s2), twice);
    }

    #[test]
    fn recorded_last_filter_replays_the_filter_it_repeated() {
        // #3089: the action holds the filter with its settings, not a "repeat whatever ran
        // last" step that fails in a new session.
        let mut s = session();
        paint_pattern(&mut s);
        s.execute("filter.other.offset", json!({"horizontal": 3})).unwrap();
        s.execute("actions.record", json!({"name": "Again"})).unwrap();
        s.execute("filter.lastFilter", json!({})).unwrap();
        s.execute("actions.stop", json!({})).unwrap();
        let steps = s.execute("actions.get", json!({"action": "Again"})).unwrap()["steps"].clone();
        assert_eq!(steps, json!([["filter.other.offset", {"horizontal": 3}]]));
        let action = s.actions.list[0].clone();

        let mut fresh = session();
        paint_pattern(&mut fresh);
        fresh.actions.list.push(action);
        let played = fresh.execute("actions.play", json!({"action": "Again"})).unwrap();
        assert!(played.get("failed").is_none(), "{played}");
        let mut direct = session();
        paint_pattern(&mut direct);
        direct.execute("filter.other.offset", json!({"horizontal": 3})).unwrap();
        assert_eq!(active_pixels(&fresh), active_pixels(&direct));
    }

    #[test]
    fn last_filter_applies_to_the_active_layer() {
        let mut s = session();
        paint_pattern(&mut s);
        let first = s.active().unwrap().active_layer.unwrap();
        // Dialogs (Filter Gallery, Camera Raw…) record the layer they ran on.
        s.execute("filter.other.offset", json!({"horizontal": 3, "layer": first.0})).unwrap();
        let first_once = active_pixels(&s);
        s.execute("layer.new.layer", json!({})).unwrap();
        paint_pattern(&mut s);
        let before = active_pixels(&s);
        s.execute("filter.lastFilter", json!({})).unwrap();
        assert_ne!(active_pixels(&s), before, "the active layer is filtered");
        let d = s.active().unwrap();
        let l = d.doc.layer(first).unwrap();
        assert_eq!(l.surface().unwrap().read_region(photocraft_geom::Rect::new(0, 0, 48, 32)), first_once, "the first layer is left alone");
    }

    #[test]
    fn params_map_to_algorithm_units() {
        assert_eq!(params_for("filter.blur.gaussianBlur", &json!({"radius": 4.5})), Some(FilterParams::GaussianBlur { radius: 4.5 }));
        assert_eq!(
            params_for("filter.blur.radialBlur", &json!({"quality": "best"})),
            Some(FilterParams::RadialBlur { amount: 10.0, method: RadialMethod::Spin, quality: RadialQuality::Best, center_x: 0.5, center_y: 0.5 })
        );
        assert_eq!(
            params_for("filter.noise.addNoise", &json!({"amount": 10, "distribution": "gaussian", "monochromatic": true})),
            Some(FilterParams::AddNoise { amount: 10.0, distribution: Distribution::Gaussian, monochromatic: true, seed: 0 })
        );
        assert_eq!(params_for("filter.blur.gaussianBlur", &json!({"radius": -3})), Some(FilterParams::GaussianBlur { radius: 0.1 }));
        assert!(params_for("filter.nope", &json!({})).is_none());
    }

    #[test]
    fn minimum_maximum_preserve_roundness() {
        for depth in [8, 16, 32] {
            let mut s = Session::new();
            s.execute("file.new", json!({"width": 48, "height": 32, "depth": depth})).unwrap();
            s.execute("layer.new.layer", json!({})).unwrap();
            // A white dot on black.
            s.edit("dot", |doc, active| {
                let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
                surf.fill_rect(photocraft_geom::Rect::new(0, 0, 48, 32), &[0.0, 0.0, 0.0, 1.0]);
                surf.write_pixel(24, 16, &[1.0, 1.0, 1.0, 1.0]);
                Ok(())
            })
            .unwrap();
            let before = active_pixels(&s);
            let lit = |s: &Session, x: usize, y: usize| active_pixels(s)[(y * 48 + x) * 4] > 0.5;
            s.execute("filter.other.maximum", json!({"radius": 4, "preserve": "roundness"})).unwrap();
            // Grown into a disc of radius 4: (4, 0) and (2, 3) are in, the square's corner isn't.
            assert!(lit(&s, 28, 16) && lit(&s, 26, 19), "depth {depth}");
            assert!(!lit(&s, 28, 20) && !lit(&s, 27, 19), "depth {depth}");
            s.execute("edit.undo", json!({})).unwrap();
            assert_eq!(active_pixels(&s), before);
            // The default is Squareness: the corner is taken.
            s.execute("filter.other.maximum", json!({"radius": 4})).unwrap();
            assert!(lit(&s, 28, 20), "depth {depth}");
            // Minimum with roundness: only the centre of that 9×9 square holds a whole disc.
            s.execute("filter.other.minimum", json!({"radius": 4, "preserve": "roundness"})).unwrap();
            assert!(!lit(&s, 20, 12) && lit(&s, 24, 16), "depth {depth}");
        }
        assert_eq!(
            params_for("filter.other.minimum", &json!({"radius": 2, "preserve": "roundness"})),
            Some(FilterParams::Minimum { radius: 2.0, preserve: Preserve::Roundness })
        );
        assert_eq!(params_for("filter.other.maximum", &json!({})), Some(FilterParams::Maximum { radius: 1.0, preserve: Preserve::Squareness }));
    }

    #[test]
    fn smart_object_records_smart_filter() {
        use photocraft_doc::{Layer, SmartObject, SmartSource};
        let mut s = session();
        s.edit("smart", |doc, active| {
            let mut cache = photocraft_raster::Surface::new(doc.pixel_format());
            cache.fill_rect(photocraft_geom::Rect::new(0, 0, 10, 10), &[1.0, 0.0, 0.0, 1.0]);
            let l = Layer::new(
                "so",
                LayerContent::Smart(SmartObject {
                    contents_id: photocraft_doc::SmartContentsId::fresh(),
                    source: SmartSource::Linked { path: String::new() },
                    transform: photocraft_geom::Affine::IDENTITY,
                    smart_filters: vec![],
                    cache: Some(cache),
                    psd_raw: None,
                    filters_enabled: true,
                    filter_mask: None,
                    warp: None,
                    stack_mode: None,
                    perspective: None,
                }),
            );
            *active = Some(doc.insert_above(*active, l));
            Ok(())
        })
        .unwrap();
        s.execute("filter.blur.gaussianBlur", json!({"radius": 2})).unwrap();
        let d = s.active().unwrap();
        let LayerContent::Smart(sm) = &d.doc.layer(d.active_layer.unwrap()).unwrap().content else { panic!() };
        assert_eq!(sm.smart_filters.len(), 1);
        assert_eq!(sm.smart_filters[0].command, "filter.blur.gaussianBlur");
        assert_eq!(sm.smart_filters[0].params["radius"], json!(2));
        assert!(sm.cache.as_ref().unwrap().pixel(10, 5)[3] > 0.0, "cache was blurred");
    }
}
