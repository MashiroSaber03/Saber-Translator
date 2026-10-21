//! The second batch of `filter.*` commands: Pixelate, Stylize, Distort (Displace, Shear,
//! ZigZag), Render (Fibers, Lens Flare, Lighting Effects, Relight), Reduce Noise, Smart/Lens/Shape Blur,
//! Blur Gallery, Other (Custom, HSB/HSL) and Video.
//!
//! They run through [`crate::filters::run_filter`] (selection, smart objects, history and
//! `filter.lastFilter` work the same as for every other filter). This module adds what some of
//! them need from the session: the foreground/background colours, a displacement map from
//! another document or a file, and a layer mask as Lens Blur depth.

use std::sync::Arc;

use photocraft_algo::{
    self as algo, BlurPath, BlurQuality, BlurShape, DepthSource, DiffuseMode, Distribution, ExtrudeType, FieldPin, FilterParams, HsbModel, IrisPin, LensType,
    Light, LightKind, MezzotintType, SmartBlurMode, SpinPin, TextureChannel, TileFill, UndefinedAreas, WindMethod, ZigZagStyle,
};
use photocraft_doc::{Document, Layer};
use photocraft_geom::Rect;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

fn f(p: &Value, k: &str, d: f32) -> f32 {
    p.get(k).and_then(Value::as_f64).map_or(d, |v| v as f32)
}
fn i(p: &Value, k: &str, d: i64) -> i64 {
    p.get(k).and_then(|v| v.as_i64().or_else(|| v.as_f64().map(|f| f.round() as i64))).unwrap_or(d)
}
fn b(p: &Value, k: &str, d: bool) -> bool {
    p.get(k).and_then(Value::as_bool).unwrap_or(d)
}
fn s<'a>(p: &'a Value, k: &str, d: &'a str) -> &'a str {
    p.get(k).and_then(Value::as_str).unwrap_or(d)
}
fn seed(p: &Value) -> u32 {
    i(p, "seed", 0) as u32
}
/// Straight RGBA colour from `[r, g, b(, a)]` (0–1) or `#rrggbb`.
fn colour(p: &Value, k: &str, d: [f32; 4]) -> [f32; 4] {
    match p.get(k) {
        Some(Value::Array(a)) if a.len() >= 3 => {
            let c = |j: usize, dv: f32| a.get(j).and_then(Value::as_f64).map_or(dv, |v| v as f32);
            [c(0, 0.0), c(1, 0.0), c(2, 0.0), c(3, 1.0)]
        }
        Some(Value::String(h)) if h.len() == 7 && h.starts_with('#') => {
            let ch = |j: usize| h.get(j..j + 2).and_then(|s| u8::from_str_radix(s, 16).ok()).map_or(0.0, |v| v as f32 / 255.0);
            [ch(1), ch(3), ch(5), 1.0]
        }
        _ => d,
    }
}
fn undefined(p: &Value, d: &str) -> UndefinedAreas {
    match s(p, "undefinedAreas", d) {
        "repeat" => UndefinedAreas::Repeat,
        "transparent" => UndefinedAreas::Transparent,
        _ => UndefinedAreas::Wrap,
    }
}
/// A JSON array param deserialized into `T`s (invalid entries are skipped).
fn list<T: serde::de::DeserializeOwned>(p: &Value, k: &str) -> Option<Vec<T>> {
    let a = p.get(k)?.as_array()?;
    let v: Vec<T> = a.iter().filter_map(|e| serde_json::from_value(e.clone()).ok()).collect();
    (!v.is_empty()).then_some(v)
}

const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

/// Algorithm parameters for the ids this module owns (Photoshop dialog units).
pub fn params_for(id: &str, p: &Value) -> Option<FilterParams> {
    let blur = |d: f32| f(p, "blur", d).clamp(0.0, 500.0);
    Some(match id {
        // ---- Pixelate ----
        "filter.pixelate.colorHalftone" => FilterParams::ColorHalftone {
            max_radius: f(p, "maxRadius", 8.0).clamp(4.0, 127.0),
            angles: [f(p, "channel1", 108.0), f(p, "channel2", 162.0), f(p, "channel3", 90.0), f(p, "channel4", 45.0)],
        },
        "filter.pixelate.crystallize" => FilterParams::Crystallize { cell_size: f(p, "cellSize", 10.0).clamp(3.0, 300.0), seed: seed(p) },
        "filter.pixelate.facet" => FilterParams::Facet,
        "filter.pixelate.fragment" => FilterParams::Fragment,
        "filter.pixelate.mezzotint" => FilterParams::Mezzotint {
            kind: match s(p, "type", "fineDots") {
                "mediumDots" => MezzotintType::MediumDots,
                "grainyDots" => MezzotintType::GrainyDots,
                "coarseDots" => MezzotintType::CoarseDots,
                "shortLines" => MezzotintType::ShortLines,
                "mediumLines" => MezzotintType::MediumLines,
                "longLines" => MezzotintType::LongLines,
                "shortStrokes" => MezzotintType::ShortStrokes,
                "mediumStrokes" => MezzotintType::MediumStrokes,
                "longStrokes" => MezzotintType::LongStrokes,
                _ => MezzotintType::FineDots,
            },
            seed: seed(p),
        },
        "filter.pixelate.pointillize" => {
            FilterParams::Pointillize { cell_size: f(p, "cellSize", 5.0).clamp(3.0, 300.0), seed: seed(p), background: colour(p, "background", WHITE) }
        }
        // ---- Stylize ----
        "filter.stylize.diffuse" => FilterParams::Diffuse {
            mode: match s(p, "mode", "normal") {
                "darkenOnly" => DiffuseMode::DarkenOnly,
                "lightenOnly" => DiffuseMode::LightenOnly,
                "anisotropic" => DiffuseMode::Anisotropic,
                _ => DiffuseMode::Normal,
            },
            seed: seed(p),
        },
        "filter.stylize.extrude" => FilterParams::Extrude {
            kind: if s(p, "type", "blocks") == "pyramids" { ExtrudeType::Pyramids } else { ExtrudeType::Blocks },
            size: f(p, "size", 30.0).clamp(2.0, 255.0),
            depth: f(p, "depth", 30.0).clamp(1.0, 255.0),
            level_based: s(p, "depthMode", "random") == "levelBased",
            solid_front: b(p, "solidFrontFaces", false),
            mask_incomplete: b(p, "maskIncompleteBlocks", false),
            seed: seed(p),
        },
        "filter.stylize.oilPaint" => FilterParams::OilPaint {
            stylization: f(p, "stylization", 4.0).clamp(0.1, 10.0),
            cleanliness: f(p, "cleanliness", 5.0).clamp(0.0, 10.0),
            scale: f(p, "scale", 1.0).clamp(0.1, 10.0),
            bristle_detail: f(p, "bristleDetail", 5.0).clamp(0.0, 10.0),
            lighting: b(p, "lighting", true),
            angle: f(p, "angle", -60.0),
            shine: f(p, "shine", 1.0).clamp(0.0, 10.0),
        },
        "filter.stylize.tiles" => FilterParams::Tiles {
            count: i(p, "count", 10).clamp(1, 99) as u32,
            max_offset: f(p, "maxOffset", 10.0).clamp(1.0, 99.0),
            fill: match s(p, "fill", "background") {
                "foreground" => TileFill::Foreground,
                "inverse" => TileFill::Inverse,
                "unaltered" => TileFill::Unaltered,
                _ => TileFill::Background,
            },
            foreground: colour(p, "foreground", BLACK),
            background: colour(p, "background", WHITE),
            seed: seed(p),
        },
        "filter.stylize.traceContour" => FilterParams::TraceContour { level: f(p, "level", 128.0).clamp(0.0, 255.0), upper: s(p, "edge", "lower") == "upper" },
        "filter.stylize.wind" => FilterParams::Wind {
            method: match s(p, "method", "wind") {
                "blast" => WindMethod::Blast,
                "stagger" => WindMethod::Stagger,
                _ => WindMethod::Wind,
            },
            from_right: s(p, "direction", "fromRight") != "fromLeft",
            seed: seed(p),
        },
        // ---- Distort ----
        "filter.distort.displace" => FilterParams::Displace {
            horizontal: f(p, "horizontal", 10.0).clamp(-999.0, 999.0),
            vertical: f(p, "vertical", 10.0).clamp(-999.0, 999.0),
            stretch: s(p, "fit", "stretch") != "tile",
            undefined: undefined(p, "repeat"),
            // A Displace imported from PSD carries its map; `mapPath` / `mapDocument` resolve later.
            map: photocraft_io::smart_map::displace_embedded_map(p),
        },
        "filter.distort.shear" => FilterParams::Shear {
            points: list::<[f32; 2]>(p, "points").unwrap_or_else(|| {
                let a = f(p, "amount", 0.0).clamp(-100.0, 100.0) / 100.0;
                vec![[0.0, 0.0], [0.5, a], [1.0, 0.0]]
            }),
            undefined: undefined(p, "wrap"),
        },
        "filter.distort.zigZag" => FilterParams::ZigZag {
            amount: f(p, "amount", 10.0).clamp(-100.0, 100.0),
            ridges: f(p, "ridges", 5.0).clamp(0.0, 20.0),
            style: match s(p, "style", "pondRipples") {
                "outFromCenter" => ZigZagStyle::OutFromCenter,
                "aroundCenter" => ZigZagStyle::AroundCenter,
                _ => ZigZagStyle::PondRipples,
            },
        },
        // ---- Render ----
        "filter.render.fibers" => FilterParams::Fibers {
            variance: f(p, "variance", 16.0).clamp(1.0, 64.0),
            strength: f(p, "strength", 4.0).clamp(1.0, 64.0),
            seed: seed(p),
            foreground: colour(p, "foreground", BLACK),
            background: colour(p, "background", WHITE),
        },
        "filter.render.lensFlare" => FilterParams::LensFlare {
            brightness: f(p, "brightness", 100.0).clamp(10.0, 300.0),
            center_x: f(p, "centerX", 0.5).clamp(-0.5, 1.5),
            center_y: f(p, "centerY", 0.5).clamp(-0.5, 1.5),
            lens: match s(p, "lens", "zoom") {
                "prime35" => LensType::Prime35,
                "prime105" => LensType::Prime105,
                "moviePrime" => LensType::MoviePrime,
                _ => LensType::Zoom,
            },
        },
        "filter.render.relight" => FilterParams::Relight {
            angle: f(p, "angle", 45.0).clamp(-180.0, 180.0),
            elevation: f(p, "elevation", 40.0).clamp(0.0, 90.0),
            intensity: f(p, "intensity", 40.0).clamp(0.0, 100.0),
            ambient: f(p, "ambient", 55.0).clamp(0.0, 100.0),
            warmth: f(p, "warmth", 0.0).clamp(-100.0, 100.0),
            softness: f(p, "softness", 25.0).clamp(1.0, 100.0),
        },
        "filter.render.lightingEffects" => FilterParams::LightingEffects {
            lights: list::<Light>(p, "lights").unwrap_or_else(|| {
                vec![Light {
                    kind: match s(p, "lightType", "spot") {
                        "point" => LightKind::Point,
                        "infinite" => LightKind::Infinite,
                        _ => LightKind::Spot,
                    },
                    x: f(p, "lightX", 0.25),
                    y: f(p, "lightY", 0.2),
                    z: f(p, "lightZ", 0.6).clamp(0.0, 10.0),
                    target_x: f(p, "targetX", 0.5),
                    target_y: f(p, "targetY", 0.55),
                    angle: f(p, "angle", 135.0),
                    elevation: f(p, "elevation", 45.0).clamp(0.0, 90.0),
                    color: {
                        let c = colour(p, "color", WHITE);
                        [c[0], c[1], c[2]]
                    },
                    intensity: f(p, "intensity", 75.0).clamp(-100.0, 100.0),
                    cone: f(p, "cone", 45.0).clamp(1.0, 89.0),
                    hotspot: f(p, "hotspot", 50.0).clamp(0.0, 100.0),
                    radius: f(p, "radius", 0.8).clamp(0.01, 10.0),
                }]
            }),
            gloss: f(p, "gloss", 0.0).clamp(-100.0, 100.0),
            metallic: f(p, "metallic", 0.0).clamp(-100.0, 100.0),
            exposure: f(p, "exposure", 0.0).clamp(-100.0, 100.0),
            ambience: f(p, "ambience", 8.0).clamp(-100.0, 100.0),
            texture: match s(p, "texture", "none") {
                "red" => TextureChannel::Red,
                "green" => TextureChannel::Green,
                "blue" => TextureChannel::Blue,
                "alpha" => TextureChannel::Alpha,
                "luminance" => TextureChannel::Luminance,
                _ => TextureChannel::None,
            },
            height: f(p, "height", 50.0).clamp(0.0, 100.0),
            white_is_high: b(p, "whiteIsHigh", true),
        },
        // ---- Noise ----
        "filter.noise.reduceNoise" => FilterParams::ReduceNoise {
            strength: f(p, "strength", 6.0).clamp(0.0, 10.0),
            preserve_details: f(p, "preserveDetails", 60.0).clamp(0.0, 100.0),
            reduce_color_noise: f(p, "reduceColorNoise", 45.0).clamp(0.0, 100.0),
            sharpen_details: f(p, "sharpenDetails", 25.0).clamp(0.0, 100.0),
            remove_jpeg_artifact: b(p, "removeJpegArtifact", false),
        },
        // ---- Blur ----
        "filter.blur.smartBlur" => FilterParams::SmartBlur {
            radius: f(p, "radius", 3.0).clamp(0.1, 100.0),
            threshold: f(p, "threshold", 25.0).clamp(0.1, 100.0),
            quality: match s(p, "quality", "high") {
                "low" => BlurQuality::Low,
                "medium" => BlurQuality::Medium,
                _ => BlurQuality::High,
            },
            mode: match s(p, "mode", "normal") {
                "edgeOnly" => SmartBlurMode::EdgeOnly,
                "overlayEdge" => SmartBlurMode::OverlayEdge,
                _ => SmartBlurMode::Normal,
            },
        },
        "filter.blur.lensBlur" => FilterParams::LensBlur {
            radius: f(p, "radius", 15.0).clamp(0.0, 100.0),
            blades: match s(p, "shape", "hexagon") {
                "triangle" => 3,
                "square" => 4,
                "pentagon" => 5,
                "heptagon" => 7,
                "octagon" => 8,
                _ => 6,
            },
            curvature: f(p, "bladeCurvature", 0.0).clamp(0.0, 100.0),
            rotation: f(p, "rotation", 0.0),
            depth: match s(p, "depthMap", "none") {
                "transparency" => DepthSource::Transparency,
                "layerMask" => DepthSource::LayerMask,
                _ => DepthSource::None,
            },
            focal_distance: f(p, "focalDistance", 0.0).clamp(0.0, 255.0),
            invert_depth: b(p, "invert", false),
            brightness: f(p, "brightness", 0.0).clamp(0.0, 100.0),
            threshold: f(p, "threshold", 255.0).clamp(0.0, 255.0),
            noise: f(p, "noise", 0.0).clamp(0.0, 100.0),
            distribution: if s(p, "distribution", "uniform") == "gaussian" { Distribution::Gaussian } else { Distribution::Uniform },
            monochromatic: b(p, "monochromatic", false),
            seed: seed(p),
            depth_map: None,
        },
        "filter.blur.shapeBlur" => FilterParams::ShapeBlur {
            radius: f(p, "radius", 10.0).clamp(5.0, 1000.0),
            shape: match s(p, "shape", "circle") {
                "ring" => BlurShape::Ring,
                "square" => BlurShape::Square,
                "diamond" => BlurShape::Diamond,
                "triangle" => BlurShape::Triangle,
                "hexagon" => BlurShape::Hexagon,
                "star" => BlurShape::Star,
                "heart" => BlurShape::Heart,
                "cross" => BlurShape::Cross,
                _ => BlurShape::Circle,
            },
        },
        // ---- Blur Gallery ----
        "filter.blurGallery.tiltShift" => FilterParams::TiltShift {
            blur: blur(15.0),
            center_x: f(p, "centerX", 0.5),
            center_y: f(p, "centerY", 0.5),
            angle: f(p, "angle", 0.0),
            focus: f(p, "focus", 0.1).clamp(0.0, 2.0),
            transition: f(p, "transition", 0.15).clamp(0.001, 2.0),
        },
        "filter.blurGallery.irisBlur" => FilterParams::IrisBlur {
            pins: list::<IrisPin>(p, "pins").unwrap_or_else(|| {
                vec![IrisPin {
                    x: f(p, "centerX", 0.5),
                    y: f(p, "centerY", 0.5),
                    radius_x: f(p, "radiusX", 0.35).max(0.001),
                    radius_y: f(p, "radiusY", 0.25).max(0.001),
                    angle: f(p, "angle", 0.0),
                    roundness: f(p, "roundness", 0.0).clamp(0.0, 100.0),
                    feather: f(p, "feather", 0.5).clamp(0.0, 0.99),
                    blur: blur(15.0),
                }]
            }),
        },
        "filter.blurGallery.fieldBlur" => FilterParams::FieldBlur {
            pins: list::<FieldPin>(p, "pins").unwrap_or_else(|| vec![FieldPin { x: f(p, "centerX", 0.5), y: f(p, "centerY", 0.5), blur: blur(15.0) }]),
        },
        "filter.blurGallery.spinBlur" => FilterParams::SpinBlur {
            pins: list::<SpinPin>(p, "pins")
                .map(|mut v| {
                    for pin in &mut v {
                        pin.angle = if pin.angle.is_finite() { pin.angle.clamp(-360.0, 360.0) } else { 0.0 };
                        pin.blur_angle = if pin.blur_angle.is_finite() { pin.blur_angle.clamp(0.0, 360.0) } else { 0.0 };
                    }
                    v
                })
                .unwrap_or_else(|| {
                    let a = f(p, "angle", 0.0);
                    vec![SpinPin {
                        x: f(p, "centerX", 0.5),
                        y: f(p, "centerY", 0.5),
                        radius_x: f(p, "radiusX", 0.3).max(0.001),
                        radius_y: f(p, "radiusY", 0.3).max(0.001),
                        angle: if a.is_finite() { a.clamp(-360.0, 360.0) } else { 0.0 },
                        blur_angle: f(p, "blurAngle", 15.0).clamp(0.0, 360.0),
                    }]
                }),
        },
        "filter.blurGallery.pathBlur" => FilterParams::PathBlur {
            paths: list::<BlurPath>(p, "paths")
                .map(|mut v| {
                    // Entry speeds bypass the scalar clamp above; the halo
                    // casts one to i32 and adds, so an unclamped value
                    // overflows (#708).
                    for b in &mut v {
                        b.speed = b.speed.clamp(0.0, 500.0);
                    }
                    v
                })
                .unwrap_or_else(|| {
                    vec![BlurPath {
                        points: vec![[f(p, "startX", 0.2), f(p, "startY", 0.5)], [f(p, "endX", 0.8), f(p, "endY", 0.5)]],
                        speed: f(p, "speed", 50.0).clamp(0.0, 500.0),
                        taper: f(p, "taper", 0.0).clamp(0.0, 100.0),
                    }]
                }),
        },
        // ---- Other ----
        "filter.other.custom" => FilterParams::Custom {
            kernel: p
                .get("kernel")
                .and_then(Value::as_array)
                .map(|a| a.iter().take(25).map(|v| v.as_f64().unwrap_or(0.0).clamp(-999.0, 999.0) as f32).collect())
                .unwrap_or_else(|| (0..25).map(|k| if k == 12 { 1.0 } else { 0.0 }).collect()),
            scale: f(p, "scale", 1.0).clamp(1.0, 9999.0),
            offset: f(p, "offset", 0.0).clamp(-9999.0, 9999.0),
        },
        "filter.other.hsbHsl" => {
            let m = |k: &str, d: &str| match s(p, k, d) {
                "hsb" => HsbModel::Hsb,
                "hsl" => HsbModel::Hsl,
                _ => HsbModel::Rgb,
            };
            FilterParams::HsbHsl { input: m("inputMode", "rgb"), output: m("rowOrder", "hsb") }
        }
        crate::color_to_alpha_cmds::ID => crate::color_to_alpha_cmds::params(p),
        // ---- Video ----
        "filter.video.deInterlace" => FilterParams::DeInterlace {
            eliminate_even: s(p, "eliminate", "oddFields") == "evenFields",
            interpolate: s(p, "createBy", "interpolation") != "duplication",
        },
        "filter.video.ntscColors" => FilterParams::NtscColors,
        _ => return crate::gallery_cmds::params_for(id, p),
    })
}

/// Ids whose rendering uses the foreground/background colours.
fn uses_colours(id: &str) -> bool {
    matches!(id, "filter.render.fibers" | "filter.stylize.tiles" | "filter.pixelate.pointillize") || crate::gallery_cmds::uses_colours(id)
}

/// Fills in session state a filter reads (the current colours) as explicit params, so the
/// recorded params (journal, smart filters) reproduce the result.
pub(crate) fn prepare(s: &Session, id: &str, p: &Value) -> Value {
    let mut out = if p.is_object() { p.clone() } else { json!({}) };
    if uses_colours(id)
        && let Value::Object(m) = &mut out
    {
        m.entry("foreground").or_insert_with(|| json!(s.tools.foreground));
        m.entry("background").or_insert_with(|| json!(s.tools.background));
    }
    out
}

/// A flattened document (or one of its layers) as an RGBA map image.
fn map_from_document(doc: &Document, layer: Option<u64>) -> Option<algo::Image> {
    let buf = match layer.and_then(|l| doc.layer(photocraft_doc::LayerId(l))) {
        Some(l) => photocraft_compose::render_layer(l, l.surface().map(|s| s.content_bounds()).filter(|r| !r.is_empty()).unwrap_or_else(|| doc.bounds())),
        None => photocraft_compose::flatten(doc),
    };
    if buf.rect.is_empty() {
        return None;
    }
    Some(algo::Image { rect: buf.rect, ch: 4, data: buf.px.iter().flatten().copied().collect() })
}

/// Resolves inputs that live outside the target layer: Displace's map (`mapPath`, or
/// `mapDocument` index with optional `mapLayer` id).
pub(crate) fn resolve(s: &Session, fp: &mut FilterParams, p: &Value) -> Result<()> {
    let FilterParams::Displace { map, .. } = fp else { return Ok(()) };
    if map.is_some() {
        return Ok(());
    }
    let layer = p.get("mapLayer").and_then(Value::as_u64);
    let img = if let Some(path) = map_path(p) {
        map_from_path(path, layer)?
    } else if let Some(d) = p.get("mapDocument") {
        let docs = s.documents();
        let found = match d {
            Value::String(name) => docs.iter().find(|ds| ds.doc.name == *name || ds.path.as_deref() == Some(name.as_str())),
            other => other.as_u64().and_then(|k| docs.get(k as usize)),
        };
        let ds = found.ok_or_else(|| EngineError::Other("displacement map document not found".into()))?;
        map_from_document(&ds.doc, layer)
    } else {
        return Err(EngineError::Other("Displace needs a displacement map: pass `mapPath` or `mapDocument`".into()));
    };
    *map = Some(Arc::new(img.ok_or_else(|| EngineError::Other("the displacement map is empty".into()))?));
    Ok(())
}

fn map_path(p: &Value) -> Option<&str> {
    p.get("mapPath").and_then(Value::as_str).filter(|p| !p.is_empty())
}

/// Displace's map read from a file (flattened, or its `mapLayer`).
fn map_from_path(path: &str, layer: Option<u64>) -> Result<Option<algo::Image>> {
    let bytes = std::fs::read(path).map_err(|e| EngineError::Other(format!("cannot read displacement map `{path}`: {e}")))?;
    let imported = crate::file_cmds::import(path, &bytes).map_err(|e| EngineError::Other(format!("cannot open displacement map `{path}`: {e}")))?;
    Ok(map_from_document(&imported, layer))
}

/// The session-free part of [`resolve`]: Displace's `mapPath`. Smart filters record params, not
/// the map, so their re-render reads the map again here; without it the filter passed the pixels
/// through. `mapDocument` needs the session and is left unresolved.
pub(crate) fn resolve_map_path(fp: &mut FilterParams, p: &Value) -> Result<()> {
    if let FilterParams::Displace { map, .. } = fp
        && map.is_none()
        && let Some(path) = map_path(p)
    {
        *map = map_from_path(path, p.get("mapLayer").and_then(Value::as_u64))?.map(Arc::new);
    }
    Ok(())
}

/// Resolves inputs read from the target layer itself: the layer mask as Lens Blur depth.
pub(crate) fn resolve_in_layer(fp: &mut FilterParams, layer: &Layer, bounds: Rect) {
    if let FilterParams::LensBlur { depth: DepthSource::LayerMask, depth_map, .. } = fp
        && depth_map.is_none()
        && let Some(m) = &layer.mask
    {
        *depth_map = Some(Arc::new(algo::Image::read(&m.surface, bounds)));
    }
}

fn relight_bad(msg: &str) -> EngineError {
    EngineError::BadParams { cmd: "filter.render.relight".into(), msg: msg.into() }
}

fn relight_num(p: &Value, key: &str, default: f32, min: f32, max: f32) -> Result<f32> {
    match p.get(key) {
        None => Ok(default),
        Some(Value::Number(n)) => {
            let Some(x) = n.as_f64() else {
                return Err(relight_bad(&format!("\"{key}\" must be a finite number")));
            };
            let x = x as f32;
            if !x.is_finite() {
                return Err(relight_bad(&format!("\"{key}\" must be finite")));
            }
            if x < min || x > max {
                return Err(relight_bad(&format!("\"{key}\" must be in {min}..{max}")));
            }
            Ok(x)
        }
        Some(_) => Err(relight_bad(&format!("\"{key}\" must be a number"))),
    }
}

fn validate_relight(p: &Value) -> Result<()> {
    if p.is_null() {
        return Err(relight_bad("params must be a JSON object"));
    }
    if !p.is_object() {
        return Err(relight_bad("params must be a JSON object"));
    }
    relight_num(p, "angle", 45.0, -180.0, 180.0)?;
    relight_num(p, "elevation", 40.0, 0.0, 90.0)?;
    relight_num(p, "intensity", 40.0, 0.0, 100.0)?;
    relight_num(p, "ambient", 55.0, 0.0, 100.0)?;
    relight_num(p, "warmth", 0.0, -100.0, 100.0)?;
    relight_num(p, "softness", 25.0, 1.0, 100.0)?;
    Ok(())
}

fn relight_enabled(s: &Session) -> std::result::Result<(), String> {
    crate::filters::has_filterable_layer(s)?;
    if crate::channel_cmds::edits_channel(s) {
        return Ok(());
    }
    let l = crate::active_layer_of(s)?;
    let locks = s.active().ok_or("no document open")?.doc.effective_locks(l.id);
    if locks.pixels || locks.all { Err(format!("the layer \"{}\" is locked", l.name)) } else { Ok(()) }
}

fn run_relight(s: &mut Session, p: &Value) -> Result<Value> {
    validate_relight(p)?;
    crate::filters::run_filter(s, "filter.render.relight", p)
}

macro_rules! cmd {
    ($id:literal, $label:literal, [$($m:literal),*], $params:literal) => {
        CommandSpec {
            id: $id,
            label: $label,
            menu: &[$($m),*],
            shortcut: None,
            params: $params,
            enabled: crate::filters::has_filterable_layer,
            run: |s, p| crate::filters::run_filter(s, $id, p),
            journal: true,
        }
    };
}

/// The command specs of this batch.
pub fn specs() -> Vec<CommandSpec> {
    vec![
        // Pixelate
        cmd!(
            "filter.pixelate.colorHalftone",
            "Color Halftone…",
            ["Filter", "Pixelate"],
            r##"{"maxRadius":4..127=8,"channel1":-360..360=108,"channel2":-360..360=162,"channel3":-360..360=90,"channel4":-360..360=45}"##
        ),
        cmd!("filter.pixelate.crystallize", "Crystallize…", ["Filter", "Pixelate"], r##"{"cellSize":3..300=10,"seed":u32=0}"##),
        cmd!("filter.pixelate.facet", "Facet", ["Filter", "Pixelate"], "{}"),
        cmd!("filter.pixelate.fragment", "Fragment", ["Filter", "Pixelate"], "{}"),
        cmd!(
            "filter.pixelate.mezzotint",
            "Mezzotint…",
            ["Filter", "Pixelate"],
            r##"{"type":"fineDots|mediumDots|grainyDots|coarseDots|shortLines|mediumLines|longLines|shortStrokes|mediumStrokes|longStrokes","seed":u32=0}"##
        ),
        cmd!("filter.pixelate.pointillize", "Pointillize…", ["Filter", "Pixelate"], r##"{"cellSize":3..300=5,"seed":u32=0,"background":json}"##),
        // Stylize
        cmd!("filter.stylize.diffuse", "Diffuse…", ["Filter", "Stylize"], r##"{"mode":"normal|darkenOnly|lightenOnly|anisotropic","seed":u32=0}"##),
        cmd!(
            "filter.stylize.extrude",
            "Extrude…",
            ["Filter", "Stylize"],
            r##"{"type":"blocks|pyramids","size":2..255=30,"depth":1..255=30,"depthMode":"random|levelBased","solidFrontFaces":bool,"maskIncompleteBlocks":bool,"seed":u32=0}"##
        ),
        cmd!(
            "filter.stylize.oilPaint",
            "Oil Paint…",
            ["Filter", "Stylize"],
            r##"{"stylization":0.1..10=4,"cleanliness":0..10=5,"scale":0.1..10=1,"bristleDetail":0..10=5,"lighting":bool=true,"angle":-180..180=-60,"shine":0..10=1}"##
        ),
        cmd!(
            "filter.stylize.tiles",
            "Tiles…",
            ["Filter", "Stylize"],
            r##"{"count":1..99=10,"maxOffset":1..99=10,"fill":"background|foreground|inverse|unaltered","seed":u32=0,"foreground":json,"background":json}"##
        ),
        cmd!("filter.stylize.traceContour", "Trace Contour…", ["Filter", "Stylize"], r##"{"level":0..255=128,"edge":"lower|upper"}"##),
        cmd!("filter.stylize.wind", "Wind…", ["Filter", "Stylize"], r##"{"method":"wind|blast|stagger","direction":"fromRight|fromLeft","seed":u32=0}"##),
        // Distort
        cmd!(
            "filter.distort.displace",
            "Displace…",
            ["Filter", "Distort"],
            r##"{"horizontal":-999..999=10,"vertical":-999..999=10,"fit":"stretch|tile","undefinedAreas":"repeat|wrap","mapDocument":doc,"mapPath":text,"mapLayer":json}"##
        ),
        cmd!("filter.distort.shear", "Shear…", ["Filter", "Distort"], r##"{"amount":-100..100=0,"undefinedAreas":"wrap|repeat","points":json}"##),
        cmd!(
            "filter.distort.zigZag",
            "ZigZag…",
            ["Filter", "Distort"],
            r##"{"amount":-100..100=10,"ridges":0..20=5,"style":"pondRipples|outFromCenter|aroundCenter"}"##
        ),
        // Render
        cmd!(
            "filter.render.fibers",
            "Fibers…",
            ["Filter", "Render"],
            r##"{"variance":1..64=16,"strength":1..64=4,"seed":u32=0,"foreground":json,"background":json}"##
        ),
        cmd!(
            "filter.render.lensFlare",
            "Lens Flare…",
            ["Filter", "Render"],
            r##"{"brightness":10..300=100,"centerX":0..1=0.5,"centerY":0..1=0.5,"lens":"zoom|prime35|prime105|moviePrime"}"##
        ),
        cmd!(
            "filter.render.lightingEffects",
            "Lighting Effects…",
            ["Filter", "Render"],
            r##"{"lightType":"spot|point|infinite","intensity":-100..100=75,"lightX":0..1=0.25,"lightY":0..1=0.2,"lightZ":0..2=0.6,"targetX":0..1=0.5,"targetY":0..1=0.55,"cone":1..89=45,"hotspot":0..100=50,"angle":-180..180=135,"elevation":0..90=45,"gloss":-100..100=0,"metallic":-100..100=0,"exposure":-100..100=0,"ambience":-100..100=8,"texture":"none|red|green|blue|alpha|luminance","height":0..100=50,"whiteIsHigh":bool=true,"lights":json}"##
        ),
        CommandSpec {
            id: "filter.render.relight",
            label: "Relight…",
            menu: &["Filter", "Render"],
            shortcut: None,
            params: r##"{"angle":-180..180=45,"elevation":0..90=40,"intensity":0..100=40,"ambient":0..100=55,"warmth":-100..100=0,"softness":1..100=25}"##,
            enabled: relight_enabled,
            run: run_relight,
            journal: true,
        },
        // Noise
        cmd!(
            "filter.noise.reduceNoise",
            "Reduce Noise…",
            ["Filter", "Noise"],
            r##"{"strength":0..10=6,"preserveDetails":0..100=60,"reduceColorNoise":0..100=45,"sharpenDetails":0..100=25,"removeJpegArtifact":bool}"##
        ),
        // Blur
        cmd!(
            "filter.blur.smartBlur",
            "Smart Blur…",
            ["Filter", "Blur"],
            r##"{"radius":0.1..100=3,"threshold":0.1..100=25,"quality":"high|medium|low","mode":"normal|edgeOnly|overlayEdge"}"##
        ),
        cmd!(
            "filter.blur.lensBlur",
            "Lens Blur…",
            ["Filter", "Blur"],
            r##"{"radius":0..100=15,"shape":"hexagon|triangle|square|pentagon|heptagon|octagon","bladeCurvature":0..100=0,"rotation":0..360=0,"depthMap":"none|transparency|layerMask","focalDistance":0..255=0,"invert":bool,"brightness":0..100=0,"threshold":0..255=255,"noise":0..100=0,"distribution":"uniform|gaussian","monochromatic":bool,"seed":u32=0}"##
        ),
        cmd!(
            "filter.blur.shapeBlur",
            "Shape Blur…",
            ["Filter", "Blur"],
            r##"{"radius":5..1000=10,"shape":"circle|ring|square|diamond|triangle|hexagon|star|heart|cross"}"##
        ),
        // Blur Gallery
        cmd!(
            "filter.blurGallery.tiltShift",
            "Tilt-Shift…",
            ["Filter", "Blur Gallery"],
            r##"{"blur":0..500=15,"centerX":0..1=0.5,"centerY":0..1=0.5,"angle":-90..90=0,"focus":0..1=0.1,"transition":0.01..1=0.15}"##
        ),
        cmd!(
            "filter.blurGallery.irisBlur",
            "Iris Blur…",
            ["Filter", "Blur Gallery"],
            r##"{"blur":0..500=15,"centerX":0..1=0.5,"centerY":0..1=0.5,"radiusX":0.01..1=0.35,"radiusY":0.01..1=0.25,"angle":-180..180=0,"roundness":0..100=0,"feather":0..0.99=0.5,"pins":json}"##
        ),
        cmd!(
            "filter.blurGallery.fieldBlur",
            "Field Blur…",
            ["Filter", "Blur Gallery"],
            r##"{"blur":0..500=15,"centerX":0..1=0.5,"centerY":0..1=0.5,"pins":json}"##
        ),
        cmd!(
            "filter.blurGallery.spinBlur",
            "Spin Blur…",
            ["Filter", "Blur Gallery"],
            r##"{"blurAngle":0..360=15,"centerX":0..1=0.5,"centerY":0..1=0.5,"radiusX":0.01..1=0.3,"radiusY":0.01..1=0.3,"angle":-180..180=0,"pins":json}"##
        ),
        cmd!(
            "filter.blurGallery.pathBlur",
            "Path Blur…",
            ["Filter", "Blur Gallery"],
            r##"{"speed":0..500=50,"taper":0..100=0,"startX":0..1=0.2,"startY":0..1=0.5,"endX":0..1=0.8,"endY":0..1=0.5,"paths":json}"##
        ),
        // Other
        cmd!("filter.other.custom", "Custom…", ["Filter", "Other"], r##"{"kernel":int[25],"scale":1..9999=1,"offset":-9999..9999=0}"##),
        cmd!("filter.other.hsbHsl", "HSB/HSL", ["Filter", "Other"], r##"{"inputMode":"rgb|hsb|hsl","rowOrder":"hsb|hsl|rgb"}"##),
        // Video
        cmd!(
            "filter.video.deInterlace",
            "De-Interlace…",
            ["Filter", "Video"],
            r##"{"eliminate":"oddFields|evenFields","createBy":"interpolation|duplication"}"##
        ),
        cmd!("filter.video.ntscColors", "NTSC Colors", ["Filter", "Video"], "{}"),
    ]
}

#[cfg(test)]
mod tests;
