//! `layer.layerStyle.*` commands: add/replace layer effects on a layer.

use photocraft_color::{BlendMode, Color};
use photocraft_doc::adjust::CurvePoint;
use photocraft_doc::{
    Bevel, BevelStyle, BevelTechnique, Contour, Effect, FxCommon, FxPaint, Glow, GlowSource, GlowTechnique, Gradient, GradientStyle, Satin, Shadow, StrokeFx,
    StrokePosition,
};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, blend_from_str};
use crate::presets::{always, bad};
use crate::{EngineError, Result, Session};

/// Parses a gradient style name (`linear`, `radial`, `angle`, `reflected`, `diamond`).
pub fn gradient_style(s: &str) -> GradientStyle {
    match s.to_ascii_lowercase().as_str() {
        "radial" => GradientStyle::Radial,
        "angle" => GradientStyle::Angle,
        "reflected" => GradientStyle::Reflected,
        "diamond" => GradientStyle::Diamond,
        _ => GradientStyle::Linear,
    }
}

fn f(p: &Value, k: &str, d: f32) -> f32 {
    p.get(k).and_then(Value::as_f64).map_or(d, |v| v as f32)
}
fn b(p: &Value, k: &str, d: bool) -> bool {
    p.get(k).and_then(Value::as_bool).unwrap_or(d)
}
fn f_of(p: &Value, k: &str) -> Option<f32> {
    p.get(k).and_then(Value::as_f64).map(|v| v as f32)
}
fn b_of(p: &Value, k: &str) -> Option<bool> {
    p.get(k).and_then(Value::as_bool)
}
fn color_of(p: &Value, k: &str) -> Option<Color> {
    match p.get(k) {
        Some(Value::Array(a)) if a.len() >= 3 => {
            Some(Color::rgb(a[0].as_f64().unwrap_or(0.0) as f32, a[1].as_f64().unwrap_or(0.0) as f32, a[2].as_f64().unwrap_or(0.0) as f32))
        }
        Some(Value::String(s)) => {
            let s = s.trim_start_matches('#');
            let h = |i: usize| s.get(i..i + 2).and_then(|x| u8::from_str_radix(x, 16).ok()).map_or(0.0, |v| f32::from(v) / 255.0);
            (s.len() >= 6).then_some(Color::rgb(h(0), h(2), h(4)))
        }
        _ => None,
    }
}
fn color(p: &Value, k: &str, d: [f32; 3]) -> Color {
    let c = color_of(p, k).map_or(d, |c| c.to_rgb());
    Color::rgb(c[0], c[1], c[2])
}
/// The stops of a two-colour gradient, baked with Photoshop's interpolation when `method` names
/// one (`classic`, `perceptual`, `linear`, `smooth`; `smoothness` 0..=100, default 100 as in
/// Photoshop): the compositors interpolate stops linearly, so the curve becomes dense stops, as on
/// PSD import. Photoshop 27.11's Classic Gradient Overlay eases (t + smoothstep(t)) / 2 between the
/// colours: a straight ramp was up to 12 levels off.
fn gradient_stops(p: &Value, from: Color, to: Color) -> Vec<(f32, Color)> {
    let stops = vec![(0.0, from), (1.0, to)];
    match p.get("method").and_then(Value::as_str).and_then(photocraft_io::GradientMethod::from_name) {
        Some(m) => photocraft_io::bake_gradient(stops, &[0.5], (f(p, "smoothness", 100.0) / 100.0).clamp(0.0, 1.0), m),
        None => stops,
    }
}
fn gradient(p: &Value) -> Gradient {
    Gradient {
        stops: gradient_stops(p, color(p, "from", [0.0; 3]), color(p, "to", [1.0; 3])),
        style: gradient_style(p.get("style").and_then(Value::as_str).unwrap_or("linear")),
        angle: f(p, "angle", 90.0),
        scale: f(p, "scale", 100.0) / 100.0,
        reverse: b(p, "reverse", false),
        ..Gradient::default()
    }
}

/// Factory defaults for each effect kind, in the command's parameter units
/// (what the Layer Style dialog starts a new effect from).
pub fn effect_defaults(kind: &str) -> Value {
    match kind {
        "dropShadow" => {
            json!({"blend": "Multiply", "color": "#000000", "opacity": 75, "angle": 120, "useGlobalLight": true, "distance": 5, "spread": 0, "size": 5, "contour": "Linear", "antiAlias": false, "noise": 0, "knocksOut": true})
        }
        "innerShadow" => {
            json!({"blend": "Multiply", "color": "#000000", "opacity": 75, "angle": 120, "useGlobalLight": true, "distance": 5, "choke": 0, "size": 5, "contour": "Linear", "antiAlias": false, "noise": 0})
        }
        "outerGlow" => {
            json!({"blend": "Screen", "opacity": 75, "color": "#ffffbe", "spread": 0, "size": 5, "range": 50, "contour": "Linear", "antiAlias": false, "noise": 0})
        }
        "innerGlow" => {
            json!({"blend": "Screen", "opacity": 75, "color": "#ffffbe", "source": "edge", "choke": 0, "size": 5, "contour": "Linear", "antiAlias": false, "noise": 0})
        }
        "stroke" => json!({"size": 3, "position": "outside", "blend": "Normal", "opacity": 100, "color": "#000000"}),
        "colorOverlay" => json!({"blend": "Normal", "color": "#ff0000", "opacity": 100}),
        "gradientOverlay" => {
            json!({"blend": "Normal", "opacity": 100, "from": "#000000", "to": "#ffffff", "reverse": false, "style": "linear", "angle": 90, "scale": 100})
        }
        "patternOverlay" => json!({"blend": "Normal", "opacity": 100, "pattern": "", "angle": 0, "scale": 100, "link": true}),
        "bevelEmboss" => {
            json!({"style": "inner", "depth": 100, "direction": "up", "size": 5, "soften": 0, "angle": 120, "useGlobalLight": true, "altitude": 30, "glossContour": "Linear", "contour": false, "contourName": "Linear", "contourRange": 50, "contourAntiAlias": false})
        }
        "satin" => {
            json!({"blend": "Multiply", "color": "#000000", "opacity": 50, "angle": 19, "distance": 11, "size": 14, "contour": "Linear", "antiAlias": true, "invert": true})
        }
        _ => json!({}),
    }
}

/// A contour preset point: input, output (both `0..=1`) and whether it is a corner.
pub type ContourPoint = (f32, f32, bool);

/// Built-in contour presets, in the order of Photoshop's default contour picker. Each is Photoshop's
/// own curve as measured in Photoshop 27.11 (a Satin of each preset against a Linear one, 38
/// levels per curve, read back through Satin's mirrored lookup — see
/// `photocraft_compose::effects::satin_lut` — and thinned to within 1 %), joined by straight
/// segments (every inner point a corner). Imported PSD contours keep their own points.
pub const CONTOURS: &[(&str, &[ContourPoint])] = &[
    ("Linear", &[(0.0, 0.0, false), (1.0, 1.0, false)]),
    (
        "Cone",
        &[
            (0.0, 0.0, false),
            (0.172, 0.477, true),
            (0.195, 0.555, true),
            (0.266, 0.711, true),
            (0.383, 0.922, true),
            (0.43, 0.969, true),
            (0.477, 0.992, true),
            (0.523, 0.992, true),
            (0.57, 0.969, true),
            (0.664, 0.859, true),
            (0.703, 0.789, true),
            (0.742, 0.711, true),
            (0.836, 0.477, true),
            (0.859, 0.398, true),
            (0.914, 0.266, true),
            (0.945, 0.156, true),
            (0.977, 0.078, true),
            (1.0, 0.0, false),
        ],
    ),
    (
        "Cone - Inverted",
        &[
            (0.0, 1.0, false),
            (0.031, 0.922, true),
            (0.07, 0.789, true),
            (0.172, 0.523, true),
            (0.195, 0.445, true),
            (0.266, 0.289, true),
            (0.383, 0.078, true),
            (0.43, 0.031, true),
            (0.477, 0.008, true),
            (0.523, 0.008, true),
            (0.57, 0.031, true),
            (0.664, 0.141, true),
            (0.703, 0.211, true),
            (0.742, 0.289, true),
            (0.836, 0.523, true),
            (0.859, 0.602, true),
            (0.914, 0.734, true),
            (0.945, 0.844, true),
            (0.977, 0.922, true),
            (1.0, 1.0, false),
        ],
    ),
    (
        "Cove - Deep",
        &[
            (0.0, 0.258, false),
            (0.344, 0.289, true),
            (0.383, 0.297, true),
            (0.477, 0.352, true),
            (0.57, 0.43, true),
            (0.617, 0.484, true),
            (0.664, 0.57, true),
            (0.703, 0.711, true),
            (0.742, 0.906, true),
            (0.773, 1.0, true),
            (1.0, 1.0, false),
        ],
    ),
    (
        "Cove - Shallow",
        &[
            (0.0, 0.0, false),
            (0.008, 0.164, true),
            (0.016, 0.477, true),
            (0.039, 0.508, true),
            (0.172, 0.57, true),
            (0.305, 0.688, true),
            (0.344, 0.734, true),
            (0.477, 0.945, true),
            (0.523, 1.0, true),
            (1.0, 1.0, false),
        ],
    ),
    (
        "Gaussian",
        &[
            (0.0, 0.0, false),
            (0.117, 0.023, true),
            (0.172, 0.055, true),
            (0.234, 0.117, true),
            (0.305, 0.234, true),
            (0.383, 0.414, true),
            (0.523, 0.688, true),
            (0.617, 0.812, true),
            (0.703, 0.891, true),
            (0.805, 0.953, true),
            (1.0, 1.0, false),
        ],
    ),
    (
        "Half Round",
        &[
            (0.0, 0.0, false),
            (0.07, 0.18, true),
            (0.195, 0.445, true),
            (0.266, 0.555, true),
            (0.383, 0.703, true),
            (0.523, 0.828, true),
            (0.617, 0.883, true),
            (0.773, 0.945, true),
            (1.0, 1.0, false),
        ],
    ),
    (
        "Ring",
        &[
            (0.0, 0.0, false),
            (0.031, 0.078, true),
            (0.039, 0.117, true),
            (0.117, 0.383, true),
            (0.172, 0.609, true),
            (0.195, 0.734, true),
            (0.266, 0.953, true),
            (0.305, 1.0, true),
            (0.344, 0.992, true),
            (0.383, 0.922, true),
            (0.43, 0.805, true),
            (0.477, 0.641, true),
            (0.57, 0.281, true),
            (0.617, 0.141, true),
            (0.664, 0.055, true),
            (0.703, 0.016, true),
            (0.742, 0.016, true),
            (0.773, 0.062, true),
            (0.836, 0.234, true),
            (0.859, 0.344, true),
            (0.891, 0.445, true),
            (0.914, 0.547, true),
            (0.945, 0.734, true),
            (0.977, 0.875, true),
            (1.0, 1.0, false),
        ],
    ),
    (
        "Ring - Double",
        &[
            (0.0, 0.0, false),
            (0.055, 0.086, true),
            (0.094, 0.172, true),
            (0.117, 0.242, true),
            (0.172, 0.477, true),
            (0.195, 0.625, true),
            (0.266, 0.891, true),
            (0.305, 0.945, true),
            (0.344, 0.891, true),
            (0.383, 0.695, true),
            (0.43, 0.367, true),
            (0.477, 0.07, true),
            (0.523, 0.047, true),
            (0.57, 0.32, true),
            (0.617, 0.648, true),
            (0.664, 0.867, true),
            (0.703, 0.93, true),
            (0.742, 0.852, true),
            (0.773, 0.641, true),
            (0.805, 0.375, true),
            (0.836, 0.156, true),
            (0.859, 0.031, true),
            (0.891, 0.023, true),
            (0.914, 0.109, true),
            (0.977, 0.703, true),
            (0.984, 0.82, true),
            (1.0, 1.0, false),
        ],
    ),
    (
        "Rolling Slope - Descending",
        &[
            (0.0, 1.0, false),
            (0.008, 1.0, true),
            (0.031, 0.93, true),
            (0.039, 0.883, true),
            (0.07, 0.758, true),
            (0.141, 0.539, true),
            (0.195, 0.406, true),
            (0.266, 0.352, true),
            (0.305, 0.352, true),
            (0.523, 0.57, true),
            (0.57, 0.602, true),
            (0.617, 0.617, true),
            (0.703, 0.602, true),
            (0.742, 0.562, true),
            (0.836, 0.391, true),
            (0.914, 0.211, true),
            (0.945, 0.109, true),
            (0.977, 0.039, true),
            (0.984, 0.008, true),
            (1.0, 0.0, false),
        ],
    ),
    (
        "Rounded Steps",
        &[
            (0.0, 0.234, false),
            (0.031, 0.305, true),
            (0.039, 0.336, true),
            (0.094, 0.43, true),
            (0.141, 0.477, true),
            (0.234, 0.492, true),
            (0.266, 0.516, true),
            (0.305, 0.602, true),
            (0.344, 0.664, true),
            (0.383, 0.711, true),
            (0.477, 0.734, true),
            (0.523, 0.797, true),
            (0.57, 0.898, true),
            (0.617, 0.953, true),
            (0.664, 0.984, true),
            (0.773, 0.984, true),
            (0.836, 0.906, true),
            (0.891, 0.891, true),
            (0.914, 0.898, true),
            (1.0, 1.0, false),
        ],
    ),
    (
        "Sawtooth 1",
        &[
            (0.0, 0.0, false),
            (0.031, 0.047, true),
            (0.195, 0.375, true),
            (0.266, 0.492, true),
            (0.477, 0.047, true),
            (0.523, 0.055, true),
            (0.742, 0.469, true),
            (0.773, 0.445, true),
            (0.914, 0.18, true),
            (0.945, 0.102, true),
            (1.0, 0.0, false),
        ],
    ),
];

/// Names earlier versions offered, still accepted by `contour` params (scripts, saved defaults).
const LEGACY_CONTOURS: &[(&str, &[ContourPoint])] = &[
    ("Cone (Inverted)", &[(0.0, 1.0, false), (0.5, 0.0, false), (1.0, 0.0, false)]),
    ("Domed", &[(0.0, 0.0, false), (0.25, 1.0, false), (0.75, 1.0, false), (1.0, 0.0, false)]),
    ("Domed (Inverted)", &[(0.0, 1.0, false), (0.25, 0.0, false), (0.75, 0.0, false), (1.0, 1.0, false)]),
    ("Diagonal (Descending)", &[(0.0, 1.0, false), (1.0, 0.0, false)]),
];

/// Most points a contour param may carry.
pub const MAX_CONTOUR_POINTS: usize = 64;

fn preset_contour(name: &str, pts: &[ContourPoint]) -> Contour {
    let corners: Vec<bool> = pts.iter().map(|p| p.2).collect();
    Contour::Custom {
        name: name.to_string(),
        points: pts.iter().map(|(input, output, _)| CurvePoint { input: *input, output: *output }).collect(),
        corners: if corners.contains(&true) { corners } else { Vec::new() },
    }
}

/// The built-in contour called `name` (`Linear` is the identity); `None` for
/// names we don't ship, so imported contours are carried rather than clobbered.
pub fn builtin_contour(name: &str) -> Option<Contour> {
    if name.eq_ignore_ascii_case("Linear") {
        return Some(Contour::Linear);
    }
    CONTOURS.iter().skip(1).chain(LEGACY_CONTOURS).find(|(n, _)| *n == name).map(|(n, pts)| preset_contour(n, pts))
}

/// The param naming a contour: `"Linear"` or a built-in preset's name when the contour is that
/// preset, else the curve itself as `{"name", "points": [[input, output], …], "corners": [bool, …]}`
/// (inputs and outputs `0..=1`), so a round trip through the param keeps an imported or edited
/// curve exactly.
pub fn contour_param(c: &Contour) -> Value {
    match c {
        Contour::Linear => json!("Linear"),
        Contour::Custom { name, .. } if builtin_contour(name).as_ref() == Some(c) => json!(name),
        Contour::Custom { name, points, corners } => {
            let pts: Vec<Value> = points.iter().map(|p| json!([short(p.input), short(p.output)])).collect();
            let corners: Vec<bool> = (0..points.len()).map(|i| corners.get(i).copied().unwrap_or(false)).collect();
            json!({"name": name, "points": pts, "corners": corners})
        }
    }
}

/// `x` as the shortest decimal that reads back as the same `f32` (`0.2`, not `0.20000000298…`).
fn short(x: f32) -> Value {
    x.to_string().parse::<f64>().map_or(Value::Null, |v| json!(v))
}

/// The contour a `contour` param names: a built-in preset's name, or a curve object (see
/// [`contour_param`]). `None` for an unknown name or a malformed curve (fewer than two points,
/// more than [`MAX_CONTOUR_POINTS`], non-finite values), so the effect keeps its contour.
/// Points are clamped to `0..=1` and sorted by input; points at the same input collapse.
pub fn contour_from_param(v: &Value) -> Option<Contour> {
    match v {
        Value::String(name) => builtin_contour(name),
        Value::Object(o) => {
            let raw = o.get("points")?.as_array()?;
            if raw.len() < 2 || raw.len() > MAX_CONTOUR_POINTS {
                return None;
            }
            let flags = o.get("corners").and_then(Value::as_array);
            let mut pts: Vec<ContourPoint> = Vec::with_capacity(raw.len());
            for (i, q) in raw.iter().enumerate() {
                let q = q.as_array()?;
                let (x, y) = (q.first()?.as_f64()? as f32, q.get(1)?.as_f64()? as f32);
                if !x.is_finite() || !y.is_finite() {
                    return None;
                }
                let corner = flags.and_then(|f| f.get(i)).and_then(Value::as_bool).unwrap_or(false);
                pts.push((x.clamp(0.0, 1.0), y.clamp(0.0, 1.0), corner));
            }
            pts.sort_by(|a, b| a.0.total_cmp(&b.0));
            pts.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-6);
            if pts.len() < 2 {
                return None;
            }
            let name = o.get("name").and_then(Value::as_str).unwrap_or("Custom");
            Some(preset_contour(name, &pts))
        }
        _ => None,
    }
}

/// The contours of an `.shc` contour preset file (Contour Editor › Load…), as `contour` params.
/// Points are rescaled from the file's `0..=255` levels; unusable curves are skipped.
pub fn contours_from_shc(bytes: &[u8]) -> Result<Vec<Value>> {
    let file = photocraft_psd::shc::parse(bytes).map_err(|e| EngineError::Other(format!("not a contour file: {e}")))?;
    Ok(file
        .into_iter()
        .filter_map(|c| {
            let v = json!({
                "name": c.name,
                "points": c.points.iter().map(|p| json!([p.input / 255.0, p.output / 255.0])).collect::<Vec<_>>(),
                "corners": c.points.iter().map(|p| !p.continuous).collect::<Vec<_>>(),
            });
            contour_from_param(&v).map(|c| contour_param(&c))
        })
        .collect())
}

/// An `.shc` file holding `contour` (Contour Editor › Save…).
pub fn contour_to_shc(contour: &Contour) -> Result<Vec<u8>> {
    let (name, points) = match contour {
        Contour::Linear => ("Linear".to_string(), vec![CurvePoint { input: 0.0, output: 0.0 }, CurvePoint { input: 1.0, output: 1.0 }]),
        Contour::Custom { name, points, .. } => (name.clone(), points.clone()),
    };
    let points = points
        .iter()
        .enumerate()
        .map(|(i, p)| photocraft_psd::shc::ShcPoint { input: p.input * 255.0, output: p.output * 255.0, continuous: !contour.is_corner(i) })
        .collect();
    photocraft_psd::shc::write(&[photocraft_psd::shc::ShcContour { name, points }]).map_err(|e| EngineError::Other(format!("can't write the contour: {e}")))
}

/// An effect of `kind` with its factory settings (before params are applied).
fn fresh_effect(kind: &str) -> Option<Effect> {
    let shadow = |inner: bool| Shadow {
        common: FxCommon::new(BlendMode::Multiply, 0.75),
        color: Color::BLACK,
        angle: 120.0,
        use_global_light: true,
        distance: 5.0,
        spread: 0.0,
        size: 5.0,
        contour: Contour::Linear,
        anti_alias: false,
        noise: 0.0,
        knocks_out: !inner,
    };
    let glow = || Glow {
        common: FxCommon::new(BlendMode::Screen, 0.75),
        paint: FxPaint::Color(Color::rgb(1.0, 1.0, 190.0 / 255.0)),
        technique: GlowTechnique::Softer,
        spread: 0.0,
        size: 5.0,
        contour: Contour::Linear,
        anti_alias: false,
        range: 0.5,
        jitter: 0.0,
        noise: 0.0,
        source: GlowSource::Edge,
    };
    Some(match kind {
        "dropShadow" => Effect::DropShadow(shadow(false)),
        "innerShadow" => Effect::InnerShadow(shadow(true)),
        "outerGlow" => Effect::OuterGlow(glow()),
        "innerGlow" => Effect::InnerGlow(glow()),
        "stroke" => Effect::Stroke(StrokeFx {
            common: FxCommon::new(BlendMode::Normal, 1.0),
            size: 3.0,
            position: StrokePosition::Outside,
            paint: FxPaint::Color(Color::rgb(1.0, 0.0, 0.0)),
        }),
        "colorOverlay" => Effect::ColorOverlay { common: FxCommon::new(BlendMode::Normal, 1.0), color: Color::rgb(1.0, 0.0, 0.0) },
        "gradientOverlay" => Effect::GradientOverlay { common: FxCommon::new(BlendMode::Normal, 1.0), gradient: Gradient::default(), dither: false },
        "patternOverlay" => Effect::PatternOverlay {
            common: FxCommon::new(BlendMode::Normal, 1.0),
            name: String::new(),
            id: String::new(),
            scale: 1.0,
            angle: 0.0,
            link: true,
            phase: (0.0, 0.0),
        },
        "satin" => Effect::Satin(Satin {
            common: FxCommon::new(BlendMode::Multiply, 0.5),
            color: Color::BLACK,
            angle: 19.0,
            distance: 11.0,
            size: 14.0,
            contour: Contour::Linear,
            anti_alias: true,
            invert: true,
        }),
        "bevelEmboss" => Effect::BevelEmboss(Bevel {
            enabled: true,
            style: BevelStyle::InnerBevel,
            technique: BevelTechnique::Smooth,
            depth: 1.0,
            up: true,
            size: 5.0,
            soften: 0.0,
            angle: 120.0,
            altitude: 30.0,
            use_global_light: true,
            gloss_contour: Contour::Linear,
            highlight: FxCommon::new(BlendMode::Screen, 0.75),
            highlight_color: Color::WHITE,
            shadow: FxCommon::new(BlendMode::Multiply, 0.75),
            shadow_color: Color::BLACK,
            contour: None,
            texture: None,
        }),
        _ => return None,
    })
}

/// Edits an existing gradient in place: only the end colours and geometry.
/// Intermediate stops, opacity stops, alignment and offset survive a round trip.
fn overlay_gradient(g: &mut Gradient, p: &Value) {
    // A method rebuilds the gradient from its end colours (the stops between are the old curve).
    if p.get("method").and_then(Value::as_str).is_some_and(|m| photocraft_io::GradientMethod::from_name(m).is_some()) {
        let ends = (g.stops.first().map(|s| s.1), g.stops.last().map(|s| s.1));
        let from = color_of(p, "from").or(ends.0).unwrap_or(Color::BLACK);
        let to = color_of(p, "to").or(ends.1).unwrap_or(Color::WHITE);
        g.stops = gradient_stops(p, from, to);
    }
    if let Some(c) = color_of(p, "from")
        && let Some(first) = g.stops.first_mut()
    {
        first.1 = c;
    }
    if let Some(c) = color_of(p, "to")
        && let Some(last) = g.stops.last_mut()
    {
        last.1 = c;
    }
    if let Some(s) = p.get("style").and_then(Value::as_str) {
        g.style = gradient_style(s);
    }
    if let Some(x) = f_of(p, "angle") {
        g.angle = x;
    }
    if let Some(x) = f_of(p, "scale") {
        g.scale = (x / 100.0).clamp(0.01, 10.0);
    }
    if let Some(x) = b_of(p, "reverse") {
        g.reverse = x;
    }
}

/// Applies the params present in `p` onto an existing effect, in the command's
/// parameter units. Absent (or unparseable) keys leave the effect unchanged, so
/// a dialog round trip never touches what it doesn't model: an imported
/// gradient stroke keeps its gradient while its opacity is edited.
pub fn overlay_effect(fx: &mut Effect, p: &Value) {
    fn set_color(p: &Value, k: &str, c: &mut Color) {
        if let Some(x) = color_of(p, k) {
            *c = x;
        }
    }
    fn set_contour(p: &Value, key: &str, c: &mut Contour) {
        if let Some(x) = p.get(key).and_then(contour_from_param) {
            *c = x;
        }
    }
    fn set_bool(p: &Value, key: &str, b: &mut bool) {
        if let Some(x) = b_of(p, key) {
            *b = x;
        }
    }
    fn common(c: &mut FxCommon, p: &Value) {
        if let Some(x) = b_of(p, "enabled") {
            c.enabled = x;
        }
        if let Some(x) = p.get("blend").and_then(Value::as_str).and_then(blend_from_str) {
            c.blend = x;
        }
        if let Some(x) = f_of(p, "opacity") {
            c.opacity = (x / 100.0).clamp(0.0, 1.0);
        }
    }
    let inner_shadow = matches!(fx, Effect::InnerShadow(_));
    let inner_glow = matches!(fx, Effect::InnerGlow(_));
    match fx {
        Effect::DropShadow(s) | Effect::InnerShadow(s) => {
            let inner = inner_shadow;
            common(&mut s.common, p);
            set_color(p, "color", &mut s.color);
            if let Some(x) = f_of(p, "angle") {
                s.angle = x;
            }
            if let Some(x) = b_of(p, "useGlobalLight") {
                s.use_global_light = x;
            }
            if let Some(x) = f_of(p, "distance") {
                s.distance = x;
            }
            if let Some(x) = f_of(p, if inner { "choke" } else { "spread" }) {
                s.spread = (x / 100.0).clamp(0.0, 1.0);
            }
            if let Some(x) = f_of(p, "size") {
                s.size = x;
            }
            set_contour(p, "contour", &mut s.contour);
            set_bool(p, "antiAlias", &mut s.anti_alias);
            if let Some(x) = f_of(p, "noise") {
                s.noise = (x / 100.0).clamp(0.0, 1.0);
            }
            if !inner && let Some(x) = b_of(p, "knocksOut") {
                s.knocks_out = x;
            }
        }
        Effect::OuterGlow(g) | Effect::InnerGlow(g) => {
            let inner = inner_glow;
            common(&mut g.common, p);
            if let Some(c) = color_of(p, "color") {
                g.paint = FxPaint::Color(c);
            }
            if let Some(t) = p.get("technique").and_then(Value::as_str) {
                g.technique = if t == "precise" { GlowTechnique::Precise } else { GlowTechnique::Softer };
            }
            if let Some(x) = f_of(p, if inner { "choke" } else { "spread" }) {
                g.spread = (x / 100.0).clamp(0.0, 1.0);
            }
            if let Some(x) = f_of(p, "size") {
                g.size = x;
            }
            set_contour(p, "contour", &mut g.contour);
            set_bool(p, "antiAlias", &mut g.anti_alias);
            if let Some(x) = f_of(p, "range") {
                g.range = (x / 100.0).clamp(0.01, 1.0);
            }
            if let Some(x) = f_of(p, "noise") {
                g.noise = (x / 100.0).clamp(0.0, 1.0);
            }
            if inner && let Some(s) = p.get("source").and_then(Value::as_str) {
                g.source = if s == "center" { GlowSource::Center } else { GlowSource::Edge };
            }
        }
        Effect::Stroke(s) => {
            common(&mut s.common, p);
            if let Some(x) = f_of(p, "size") {
                s.size = x;
            }
            if let Some(pos) = p.get("position").and_then(Value::as_str) {
                s.position = match pos {
                    "inside" => StrokePosition::Inside,
                    "center" => StrokePosition::Center,
                    _ => StrokePosition::Outside,
                };
            }
            if p.get("from").is_some() {
                s.paint = FxPaint::Gradient(gradient(p));
            } else if let Some(c) = color_of(p, "color") {
                s.paint = FxPaint::Color(c);
            }
        }
        Effect::ColorOverlay { common: c, color } => {
            common(c, p);
            set_color(p, "color", color);
        }
        Effect::GradientOverlay { common: c, gradient: g, dither } => {
            common(c, p);
            overlay_gradient(g, p);
            if let Some(x) = b_of(p, "dither") {
                *dither = x;
            }
        }
        Effect::PatternOverlay { common: c, name, id, scale, angle, link, phase } => {
            common(c, p);
            if let Some(x) = p.get("pattern").and_then(Value::as_str) {
                *name = x.to_string();
                *id = String::new();
            }
            if let Some(x) = f_of(p, "scale") {
                *scale = (x / 100.0).clamp(0.01, 10.0);
            }
            if let Some(x) = f_of(p, "angle") {
                *angle = x;
            }
            if let Some(x) = b_of(p, "link") {
                *link = x;
            }
            if let Some(x) = f_of(p, "phaseX") {
                phase.0 = x;
            }
            if let Some(x) = f_of(p, "phaseY") {
                phase.1 = x;
            }
        }
        Effect::Satin(s) => {
            common(&mut s.common, p);
            set_color(p, "color", &mut s.color);
            if let Some(x) = f_of(p, "angle") {
                s.angle = x;
            }
            if let Some(x) = f_of(p, "distance") {
                s.distance = x;
            }
            if let Some(x) = f_of(p, "size") {
                s.size = x;
            }
            if let Some(x) = b_of(p, "invert") {
                s.invert = x;
            }
            set_contour(p, "contour", &mut s.contour);
            set_bool(p, "antiAlias", &mut s.anti_alias);
        }
        Effect::BevelEmboss(bv) => {
            if let Some(x) = b_of(p, "enabled") {
                bv.enabled = x;
            }
            if let Some(s) = p.get("style").and_then(Value::as_str) {
                bv.style = match s {
                    "outer" => BevelStyle::OuterBevel,
                    "emboss" => BevelStyle::Emboss,
                    "pillow" => BevelStyle::PillowEmboss,
                    "stroke" => BevelStyle::StrokeEmboss,
                    _ => BevelStyle::InnerBevel,
                };
            }
            if let Some(t) = p.get("technique").and_then(Value::as_str) {
                bv.technique = match t {
                    "chiselHard" => BevelTechnique::ChiselHard,
                    "chiselSoft" => BevelTechnique::ChiselSoft,
                    _ => BevelTechnique::Smooth,
                };
            }
            if let Some(x) = f_of(p, "depth") {
                bv.depth = x / 100.0;
            }
            if let Some(d) = p.get("direction").and_then(Value::as_str) {
                bv.up = d != "down";
            }
            if let Some(x) = f_of(p, "size") {
                bv.size = x;
            }
            if let Some(x) = f_of(p, "soften") {
                bv.soften = x;
            }
            if let Some(x) = f_of(p, "angle") {
                bv.angle = x;
            }
            if let Some(x) = f_of(p, "altitude") {
                bv.altitude = x;
            }
            if let Some(x) = b_of(p, "useGlobalLight") {
                bv.use_global_light = x;
            }
            set_contour(p, "glossContour", &mut bv.gloss_contour);
            // The Contour element: `contour` turns it on/off; its curve, Range and
            // Anti-aliased edit the element in place, so an imported curve survives.
            match b_of(p, "contour") {
                Some(false) => bv.contour = None,
                Some(true) if bv.contour.is_none() => bv.contour = Some(photocraft_doc::BevelContour::default()),
                _ => {}
            }
            if let Some(c) = &mut bv.contour {
                set_contour(p, "contourName", &mut c.contour);
                if let Some(x) = f_of(p, "contourRange") {
                    c.range = (x / 100.0).clamp(0.01, 1.0);
                }
                set_bool(p, "contourAntiAlias", &mut c.anti_alias);
            }
            if let Some(t) = p.get("texture").and_then(Value::as_str).filter(|t| !t.is_empty()) {
                bv.texture = Some(photocraft_doc::BevelTexture {
                    name: t.to_string(),
                    id: t.to_string(),
                    scale: (f(p, "textureScale", 100.0) / 100.0).clamp(0.01, 10.0),
                    depth: (f(p, "textureDepth", 100.0) / 100.0).clamp(-10.0, 10.0),
                    invert: b(p, "textureInvert", false),
                    link: b(p, "textureLink", true),
                    phase: (0.0, 0.0),
                });
            }
        }
    }
}

/// Builds an effect of `kind` from JSON params (Photoshop units: opacity,
/// spread, range in percent; sizes in px; angles in degrees).
pub fn effect_from_params(kind: &str, p: &Value) -> Option<Effect> {
    let mut fx = fresh_effect(kind)?;
    overlay_effect(&mut fx, p);
    Some(fx)
}

/// The effect's kind as the commands and dialog name it.
pub fn kind_of(e: &Effect) -> &'static str {
    match e {
        Effect::DropShadow(_) => "dropShadow",
        Effect::InnerShadow(_) => "innerShadow",
        Effect::OuterGlow(_) => "outerGlow",
        Effect::InnerGlow(_) => "innerGlow",
        Effect::Stroke(_) => "stroke",
        Effect::ColorOverlay { .. } => "colorOverlay",
        Effect::GradientOverlay { .. } => "gradientOverlay",
        Effect::PatternOverlay { .. } => "patternOverlay",
        Effect::Satin(_) => "satin",
        Effect::BevelEmboss(_) => "bevelEmboss",
    }
}

fn same_kind(a: &Effect, b: &Effect) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}
/// Upper bound on the effects one `layer.layerStyle.replace` call accepts, so an
/// automation mistake can't ask for an unbounded document change.
const MAX_EFFECTS: usize = 256;

/// Replaces a layer's whole effect list in one history step: the Layer Style dialog's
/// apply. Each entry builds an effect from `params`, or edits the snapshot it carried
/// (`fx`: the effect JSON as the dialog loaded it, kept verbatim for everything the
/// dialog doesn't model) with the params present. Unknown kinds, kind/snapshot
/// mismatches and malformed snapshots are errors, never panics.
fn replace_effects(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "layer.layerStyle.replace";
    let id = crate::commands::layer_param(s, p)?;
    let entries = p.get("effects").and_then(Value::as_array).ok_or_else(|| bad(CMD, "need `effects`: [{kind, params, fx?}]"))?;
    if entries.len() > MAX_EFFECTS {
        return Err(bad(CMD, format!("at most {MAX_EFFECTS} effects")));
    }
    let mut list = Vec::with_capacity(entries.len());
    let mut pats = Vec::new();
    for e in entries {
        let kind = e.get("kind").and_then(Value::as_str).ok_or_else(|| bad(CMD, "each effect needs a `kind` string"))?;
        let params = match e.get("params") {
            None | Some(Value::Null) => json!({}),
            Some(Value::Object(_)) => e.get("params").cloned().unwrap_or_default(),
            Some(_) => return Err(bad(CMD, "`params` must be an object")),
        };
        let fx = match e.get("fx") {
            Some(fx_json) => {
                let mut fx: Effect = serde_json::from_value(fx_json.clone()).map_err(|err| bad(CMD, format!("bad effect snapshot: {err}")))?;
                if kind_of(&fx) != kind {
                    return Err(bad(CMD, format!("effect snapshot is a {}, not `{kind}`", kind_of(&fx))));
                }
                overlay_effect(&mut fx, &params);
                fx
            }
            None => effect_from_params(kind, &params).ok_or_else(|| bad(CMD, format!("unknown effect {kind}")))?,
        };
        let (fx, pat) = crate::pattern_cmds::resolve_effect(s, fx)?;
        if let Some(pat) = pat {
            pats.push(pat);
        }
        list.push(fx);
    }
    let count = list.len();
    let label = if list.is_empty() { "Clear Layer Style" } else { "Edit Layer Style" };
    s.edit(label, |doc, _| {
        for pat in &pats {
            crate::pattern_cmds::ensure_in_doc(doc, pat);
        }
        let mode = doc.mode;
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        l.effects.items = list.into_iter().map(|e| e.in_mode(mode)).collect();
        l.effects.enabled = true;
        Ok(())
    })?;
    Ok(json!({ "layer": id.0, "effects": count }))
}

fn set_effect(s: &mut Session, p: &Value, kind: &str) -> Result<Value> {
    let fx = effect_from_params(kind, p).ok_or_else(|| EngineError::Other(format!("unknown effect {kind}")))?;
    let (fx, pattern) = crate::pattern_cmds::resolve_effect(s, fx)?;
    let label = format!("Layer Style: {}", fx.label());
    let add = b(p, "add", false);
    let id = crate::commands::layer_param(s, p)?;
    s.edit(&label, |doc, _| {
        if let Some(pat) = &pattern {
            crate::pattern_cmds::ensure_in_doc(doc, pat);
        }
        // Colours in the document's mode, as Image › Mode keeps them.
        let fx = fx.in_mode(doc.mode);
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        l.effects.enabled = true;
        match l.effects.items.iter_mut().find(|e| same_kind(e, &fx)) {
            Some(slot) if !add => *slot = fx,
            _ => l.effects.items.push(fx),
        }
        Ok(())
    })?;
    Ok(json!({ "layer": id.0 }))
}

/// Stores the user default for one effect kind ("Make Default" in the Layer Style dialog).
fn make_default(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "layer.layerStyle.makeDefault";
    let kind = req_str_kind(p, CMD)?;
    let params = match p.get("params") {
        Some(v) if v.is_object() => v.clone(),
        _ => return Err(bad(CMD, "`params` must be an object")),
    };
    s.presets.layer_defaults.insert(kind.to_string(), params);
    s.presets_changed();
    Ok(json!({"kind": kind}))
}

/// The user default for one effect kind, else its factory defaults.
fn default_for(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "layer.layerStyle.defaultFor";
    let kind = req_str_kind(p, CMD)?;
    match s.presets.layer_defaults.get(kind) {
        Some(v) if v.is_object() => Ok(json!({"params": v, "user": true})),
        _ => Ok(json!({"params": effect_defaults(kind), "user": false})),
    }
}

fn req_str_kind<'a>(p: &'a Value, cmd: &str) -> Result<&'a str> {
    let kind = p.get("kind").and_then(Value::as_str).ok_or_else(|| bad(cmd, "need `kind`"))?;
    if effect_from_params(kind, &Value::Null).is_none() {
        return Err(bad(cmd, format!("unknown effect {kind}")));
    }
    Ok(kind)
}

fn has_layer(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    d.active_layer.filter(|id| d.doc.layer(*id).is_some()).map(|_| ()).ok_or_else(|| "no active layer".into())
}

macro_rules! style_cmd {
    ($kind:literal, $label:literal, $params:literal) => {
        CommandSpec {
            id: concat!("layer.layerStyle.", $kind),
            label: $label,
            menu: &["Layer", "Layer Style"],
            shortcut: None,
            params: $params,
            enabled: has_layer,
            run: |s, p| set_effect(s, p, $kind),
            journal: true,
        }
    };
}

/// The `layer.layerStyle.*` command specs.
pub fn specs() -> Vec<CommandSpec> {
    vec![
        style_cmd!(
            "dropShadow",
            "Drop Shadow…",
            r##"{"color":"#rrggbb","opacity":0..100=75,"blend":str="multiply","angle":deg=120,"useGlobalLight":bool,"distance":px=5,"spread":0..100,"size":px=5,"contour":"Linear|Cone|Cone - Inverted|Cove - Deep|Cove - Shallow|Gaussian|Half Round|Ring|Ring - Double|Rolling Slope - Descending|Rounded Steps|Sawtooth 1"|{"name":str,"points":[[in 0..1,out 0..1],…],"corners":[bool,…]}="Linear","antiAlias":bool,"noise":0..100,"knocksOut":bool,"add":bool,"layer":id}"##
        ),
        style_cmd!(
            "innerShadow",
            "Inner Shadow…",
            r##"{"color":"#rrggbb","opacity":0..100=75,"blend":str,"angle":deg,"distance":px,"choke":0..100,"size":px,"contour":name|curve,"antiAlias":bool,"noise":0..100,"add":bool}"##
        ),
        style_cmd!(
            "outerGlow",
            "Outer Glow…",
            r##"{"color":"#rrggbb","opacity":0..100=75,"blend":str="screen","technique":"softer|precise","spread":0..100,"size":px,"range":0..100,"contour":name|curve,"antiAlias":bool,"noise":0..100,"add":bool}"##
        ),
        style_cmd!(
            "innerGlow",
            "Inner Glow…",
            r##"{"color":"#rrggbb","opacity":0..100=75,"blend":str="screen","technique":"softer|precise","source":"edge|center","choke":0..100,"size":px,"contour":name|curve,"antiAlias":bool,"noise":0..100,"add":bool}"##
        ),
        style_cmd!(
            "stroke",
            "Stroke…",
            r##"{"size":px=3,"position":"outside|inside|center","color":"#rrggbb","from":"#rrggbb","to":"#rrggbb","method":"classic|perceptual|linear|smooth"?,"smoothness":0..100=100,"style":str,"angle":deg,"opacity":0..100,"blend":str,"add":bool}"##
        ),
        style_cmd!("colorOverlay", "Color Overlay…", r##"{"color":"#rrggbb","opacity":0..100=100,"blend":str,"add":bool}"##),
        style_cmd!(
            "gradientOverlay",
            "Gradient Overlay…",
            r##"{"from":"#rrggbb","to":"#rrggbb","method":"classic|perceptual|linear|smooth"? (Photoshop's interpolation; omitted: straight stops),"smoothness":0..100=100,"style":"linear|radial|angle|reflected|diamond","angle":deg=90,"scale":10..150=100,"reverse":bool,"opacity":0..100,"blend":str,"add":bool}"##
        ),
        style_cmd!(
            "patternOverlay",
            "Pattern Overlay…",
            r##"{"pattern":id|name?=first library pattern,"opacity":0..100=100,"blend":str,"scale":1..1000=100,"angle":deg=0,"link":bool=true,"phaseX":px,"phaseY":px,"add":bool}"##
        ),
        style_cmd!(
            "bevelEmboss",
            "Bevel & Emboss…",
            r##"{"style":"inner|outer|emboss|pillow|stroke","technique":"smooth|chiselHard|chiselSoft","contour":bool,"contourName":name|curve,"contourRange":1..100=50,"contourAntiAlias":bool,"glossContour":name|curve,"texture":pattern id|name?,"textureScale":1..1000=100,"textureDepth":-1000..1000=100,"textureInvert":bool,"textureLink":bool=true,"depth":1..1000=100,"direction":"up|down","size":px=5,"soften":px,"angle":deg,"altitude":deg,"add":bool}"##
        ),
        style_cmd!(
            "satin",
            "Satin…",
            r##"{"color":"#rrggbb","opacity":0..100=50,"blend":str,"angle":deg,"distance":px,"size":px,"contour":name|curve,"antiAlias":bool=true,"invert":bool,"add":bool}"##
        ),
        CommandSpec {
            id: "layer.layerStyle.replace",
            label: "Edit Layer Style",
            menu: &[],
            shortcut: None,
            params: r##"{"layer":id,"effects":[{"kind":"dropShadow|innerShadow|outerGlow|innerGlow|stroke|colorOverlay|gradientOverlay|patternOverlay|satin|bevelEmboss","params":{…layer.layerStyle.<kind> params…},"fx":<effect snapshot carried through the Layer Style dialog, optional>}]} (replaces the layer's whole effect list; an entry with `fx` edits that effect with the params present, one without builds from params)"##,
            enabled: has_layer,
            run: replace_effects,
            journal: true,
        },
        CommandSpec {
            id: "layer.layerStyle.makeDefault",
            label: "Make Layer Style Default",
            menu: &[],
            shortcut: None,
            params: r##"{"kind":"stroke|dropShadow|innerShadow|outerGlow|innerGlow|colorOverlay|gradientOverlay|patternOverlay|satin|bevelEmboss","params":{…param set…}} (stores the user default the Layer Style dialog's Reset to Default restores)"##,
            enabled: always,
            run: make_default,
            journal: false,
        },
        CommandSpec {
            id: "layer.layerStyle.defaultFor",
            label: "Layer Style Default",
            menu: &[],
            shortcut: None,
            params: r##"{"kind":str} → {"params":{…},"user":bool} (the saved user default, else the factory defaults)"##,
            enabled: always,
            run: default_for,
            journal: false,
        },
        CommandSpec {
            id: "layer.layerStyle.clear",
            label: "Clear Layer Style",
            menu: &["Layer", "Layer Style"],
            shortcut: None,
            params: r##"{"layer":id? (no layer: every selected layer that has a style),"effect":index? (only that effect of "layer", as an effect row dropped on the Layers panel's Delete button)}"##,
            enabled: has_layer,
            run: |s, p| {
                if let Some(effect) = p.get("effect").filter(|v| !v.is_null()) {
                    return delete_effect(s, p, effect);
                }
                let ids = crate::extra_cmds::style_targets(s, p, |l| !l.effects.items.is_empty())?;
                s.edit("Clear Layer Style", |doc, _| {
                    for id in ids {
                        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
                        l.effects = photocraft_doc::Effects { enabled: true, ..Default::default() };
                    }
                    Ok(())
                })?;
                Ok(Value::Null)
            },
            journal: true,
        },
    ]
}

/// `layer.layerStyle.clear` with an `effect` index: deletes that one effect of `layer` (the active
/// layer without one), as Photoshop does for an effect row dropped on the Delete button.
fn delete_effect(s: &mut Session, p: &Value, effect: &Value) -> Result<Value> {
    const CMD: &str = "layer.layerStyle.clear";
    let index = effect.as_u64().and_then(|i| usize::try_from(i).ok()).ok_or_else(|| bad(CMD, "`effect` must be an effect index"))?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let id = match p.get("layer").filter(|v| !v.is_null()) {
        Some(v) => photocraft_doc::LayerId(v.as_u64().ok_or_else(|| bad(CMD, "`layer` must be a layer id"))?),
        None => d.active_layer.ok_or_else(|| bad(CMD, "no active layer"))?,
    };
    let count = d.doc.layer(id).ok_or(EngineError::NoLayer(id))?.effects.items.len();
    if index >= count {
        return Err(bad(CMD, format!("the layer has {count} effect(s), no effect {index}")));
    }
    s.edit("Delete Effect", |doc, _| {
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        if index < l.effects.items.len() {
            l.effects.items.remove(index);
        }
        Ok(())
    })?;
    Ok(Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn session() -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 32, "height": 32})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s
    }

    fn effects(s: &Session) -> Vec<Effect> {
        let d = s.active().unwrap();
        d.doc.layer(d.active_layer.unwrap()).unwrap().effects.items.clone()
    }

    #[test]
    fn bevel_command_sets_technique_contour_and_texture() {
        let mut s = session();
        s.execute("layer.layerStyle.bevelEmboss", json!({"style": "pillow", "technique": "chiselHard", "contour": true, "contourRange": 70, "texture": "Bubbles", "textureDepth": -200, "textureInvert": true})).unwrap();
        let fx = effects(&s);
        let Effect::BevelEmboss(b) = &fx[0] else { panic!("{fx:?}") };
        assert_eq!((b.style, b.technique), (photocraft_doc::BevelStyle::PillowEmboss, BevelTechnique::ChiselHard));
        assert!((b.contour.as_ref().unwrap().range - 0.7).abs() < 1e-6);
        let t = b.texture.as_ref().unwrap();
        assert_eq!((t.name.as_str(), t.depth, t.invert), ("Bubbles", -2.0, true));
    }

    #[test]
    fn every_style_command_adds_its_effect() {
        for kind in ["dropShadow", "innerShadow", "outerGlow", "innerGlow", "stroke", "colorOverlay", "gradientOverlay", "bevelEmboss", "satin"] {
            let mut s = session();
            s.execute(&format!("layer.layerStyle.{kind}"), json!({})).unwrap();
            let fx = effects(&s);
            assert_eq!(fx.len(), 1, "{kind}");
            assert!(fx[0].enabled());
        }
    }

    /// #1543: a Satin or Bevel & Emboss size or softness far past Photoshop's range rendered a
    /// blur kernel of 2^62 taps ("capacity overflow"), or ran for minutes at 1e6. The blur width
    /// is capped at the effect reach, so the document still renders, quickly.
    #[test]
    fn oversized_satin_and_bevel_sizes_render() {
        for (cmd, p) in [
            ("satin", json!({"size": 1e308})),
            ("satin", json!({"size": 1_000_000})),
            ("bevelEmboss", json!({"size": 1e308})),
            ("bevelEmboss", json!({"soften": 1e308})),
            ("bevelEmboss", json!({"size": 1_000_000, "soften": 1_000_000})),
        ] {
            let mut s = session();
            s.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
            s.execute(&format!("layer.layerStyle.{cmd}"), p.clone()).unwrap();
            let start = std::time::Instant::now();
            s.execute("document.pixel", json!({"x": 0, "y": 0})).unwrap_or_else(|e| panic!("{cmd} {p}: {e}"));
            assert!(start.elapsed().as_secs() < 20, "{cmd} {p}: {:?}", start.elapsed());
        }
    }

    #[test]
    fn params_are_applied_and_replace_or_add() {
        let mut s = session();
        s.execute("layer.layerStyle.stroke", json!({"size": 7, "position": "inside", "color": "#00ff00", "opacity": 50})).unwrap();
        match &effects(&s)[0] {
            Effect::Stroke(st) => {
                assert_eq!(st.size, 7.0);
                assert_eq!(st.position, StrokePosition::Inside);
                assert_eq!(st.common.opacity, 0.5);
                assert_eq!(st.paint, FxPaint::Color(Color::rgb(0.0, 1.0, 0.0)));
            }
            other => panic!("{other:?}"),
        }
        s.execute("layer.layerStyle.stroke", json!({"size": 2})).unwrap();
        assert_eq!(effects(&s).len(), 1, "replaces");
        s.execute("layer.layerStyle.stroke", json!({"size": 4, "add": true})).unwrap();
        assert_eq!(effects(&s).len(), 2, "adds a second instance");
        s.execute("layer.layerStyle.dropShadow", json!({"blend": "normal", "distance": 12})).unwrap();
        assert!(matches!(&effects(&s)[2], Effect::DropShadow(sh) if sh.common.blend == BlendMode::Normal && sh.distance == 12.0));
    }

    #[test]
    fn clear_and_undo() {
        let mut s = session();
        s.execute("layer.layerStyle.colorOverlay", json!({"color": "#ff0000"})).unwrap();
        s.execute("layer.layerStyle.clear", json!({})).unwrap();
        assert!(effects(&s).is_empty());
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(effects(&s).len(), 1);
    }

    /// #3102: an effect row dropped on the Delete button deletes only that effect, in one step.
    #[test]
    fn clear_one_effect_by_index() {
        let mut s = session();
        s.execute("layer.layerStyle.dropShadow", json!({})).unwrap();
        s.execute("layer.layerStyle.colorOverlay", json!({"color": "#ff0000"})).unwrap();
        let layer = s.active().unwrap().active_layer.unwrap().0;
        s.execute("layer.layerStyle.clear", json!({"layer": layer, "effect": 0})).unwrap();
        assert!(matches!(effects(&s)[..], [Effect::ColorOverlay { .. }]));
        for bad in [json!({"effect": 5}), json!({"effect": -1}), json!({"effect": "x"}), json!({"effect": 0, "layer": u64::MAX})] {
            assert!(s.execute("layer.layerStyle.clear", bad).is_err());
        }
        assert_eq!(effects(&s).len(), 1, "bad params change nothing");
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(effects(&s).len(), 2);
    }

    #[test]
    fn color_overlay_renders() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
        s.execute("layer.layerStyle.colorOverlay", json!({"color": "#ff0000"})).unwrap();
        let px: Vec<f32> = serde_json::from_value(s.execute("document.pixel", json!({"x": 3, "y": 3})).unwrap()).unwrap();
        assert!(px[0] > 0.99 && px[1] < 0.01, "{px:?}");
    }

    #[test]
    fn gradient_styles_parse() {
        assert_eq!(gradient_style("Radial"), GradientStyle::Radial);
        assert_eq!(gradient_style("diamond"), GradientStyle::Diamond);
        assert_eq!(gradient_style("?"), GradientStyle::Linear);
    }

    fn snapshot(fx: &Effect) -> Value {
        serde_json::to_value(fx).unwrap()
    }

    #[test]
    fn every_kind_has_factory_defaults() {
        for kind in
            ["dropShadow", "innerShadow", "outerGlow", "innerGlow", "stroke", "colorOverlay", "gradientOverlay", "patternOverlay", "satin", "bevelEmboss"]
        {
            assert!(effect_defaults(kind).as_object().is_some_and(|o| !o.is_empty()), "{kind}");
            assert!(effect_from_params(kind, &json!({})).is_some(), "{kind}");
        }
        assert!(effect_from_params("nope", &json!({})).is_none());
    }

    #[test]
    fn replace_builds_edits_and_preserves_instances() {
        let mut s = session();
        // Two strokes: the first edits a snapshot (params overlay), the second is fresh.
        let first = Effect::Stroke(StrokeFx {
            common: photocraft_doc::FxCommon::new(BlendMode::Normal, 1.0),
            size: 7.0,
            position: StrokePosition::Outside,
            paint: FxPaint::Gradient(Gradient { stops: vec![(0.0, Color::rgb(0.0, 0.0, 1.0)), (1.0, Color::WHITE)], ..Gradient::default() }),
        });
        let r = s
            .execute(
                "layer.layerStyle.replace",
                json!({"effects": [
                    {"kind": "stroke", "fx": snapshot(&first), "params": {"opacity": 50}},
                    {"kind": "stroke", "params": {"size": 2, "color": "#ffffff"}},
                ]}),
            )
            .unwrap();
        assert_eq!(r["effects"], 2);
        let fx = effects(&s);
        assert_eq!(fx.len(), 2, "order preserved");
        match (&fx[0], &fx[1]) {
            (Effect::Stroke(a), Effect::Stroke(b)) => {
                assert_eq!(a.common.opacity, 0.5);
                // The overlay must not clobber the gradient paint it doesn't model.
                assert!(matches!(&a.paint, FxPaint::Gradient(g) if g.stops.len() == 2 && g.stops[0].1 == Color::rgb(0.0, 0.0, 1.0)));
                assert_eq!(b.size, 2.0);
                assert_eq!(b.paint, FxPaint::Color(Color::rgb(1.0, 1.0, 1.0)));
            }
            other => panic!("{other:?}"),
        }
        // Replacing with an empty list clears the layer.
        s.execute("layer.layerStyle.replace", json!({"effects": []})).unwrap();
        assert!(effects(&s).is_empty());
    }

    #[test]
    fn replace_rejects_bad_entries_gracefully() {
        let mut s = session();
        let stroke = Effect::Stroke(StrokeFx {
            common: photocraft_doc::FxCommon::new(BlendMode::Normal, 1.0),
            size: 3.0,
            position: StrokePosition::Outside,
            paint: FxPaint::Color(Color::BLACK),
        });
        for p in [
            json!({}),
            json!({"effects": "nope"}),
            json!({"effects": [{"params": {}}]}),
            json!({"effects": [{"kind": "nope"}]}),
            json!({"effects": [{"kind": "stroke", "fx": {"Bogus": {}}}]}),
            json!({"effects": [{"kind": "stroke", "fx": 3}]}),
            json!({"effects": [{"kind": "dropShadow", "fx": snapshot(&stroke)}]}),
            json!({"effects": [{"kind": "stroke", "fx": snapshot(&stroke), "params": "nope"}]}),
        ] {
            assert!(s.execute("layer.layerStyle.replace", p).is_err());
        }
        // A long list is refused, not run.
        let many: Vec<_> = (0..257).map(|_| json!({"kind": "stroke"})).collect();
        assert!(s.execute("layer.layerStyle.replace", json!({"effects": many})).is_err());
        assert!(effects(&s).is_empty(), "nothing was applied");
    }

    #[test]
    fn contour_and_noise_params_edit_existing_effects() {
        let mut s = session();
        s.execute("layer.layerStyle.dropShadow", json!({})).unwrap();
        let before = effects(&s);
        let Effect::DropShadow(_) = &before[0] else { panic!("{before:?}") };
        let fx0 = snapshot(&before[0]);
        // Unknown contour names are carried, not clobbered to Linear.
        s.execute("layer.layerStyle.replace", json!({"effects": [{"kind": "dropShadow", "fx": fx0, "params": {"contour": "Cone", "noise": 30}}]})).unwrap();
        let fx = effects(&s);
        let Effect::DropShadow(sh) = &fx[0] else { panic!("{fx:?}") };
        assert_eq!(sh.contour, builtin_contour("Cone").unwrap(), "known preset applies");
        assert!((sh.noise - 0.3).abs() < 1e-6, "noise in percent");
        // An imported contour with an unknown name survives the same round trip.
        let imported = Contour::Custom {
            name: "Photoshop contour".into(),
            points: vec![photocraft_doc::adjust::CurvePoint { input: 0.0, output: 1.0 }, photocraft_doc::adjust::CurvePoint { input: 1.0, output: 0.0 }],
            corners: vec![],
        };
        let mut carried = before[0].clone();
        if let Effect::DropShadow(c) = &mut carried {
            c.contour = imported.clone();
        }
        overlay_effect(&mut carried, &json!({"contour": "Photoshop contour"}));
        let Effect::DropShadow(after) = &carried else { panic!() };
        assert_eq!(after.contour, imported);
        // Linear applies by name.
        let mut back = carried;
        overlay_effect(&mut back, &json!({"contour": "Linear"}));
        let Effect::DropShadow(lin) = &back else { panic!() };
        assert_eq!(lin.contour, Contour::Linear);
    }

    #[test]
    fn contour_presets_match_photoshop() {
        // Photoshop 27.11, measured through Satin: Cone and Cone − Inverted Satins are identical,
        // and Gaussian reads 0.36 at 0.5 through Satin but 0.64 as the preset curve (glows, gloss).
        let lut = |name: &str| {
            let Some(Contour::Custom { points, corners, .. }) = builtin_contour(name) else { panic!("{name}") };
            photocraft_compose::adjust::contour_curve_lut(&points, &corners)
        };
        let satin = |name: &str| photocraft_compose::effects::satin_lut(&builtin_contour(name).unwrap()).unwrap();
        let at = |l: &[f32], x: f32| l[(x * (l.len() - 1) as f32).round() as usize];
        let (cone, inverted) = (satin("Cone"), satin("Cone - Inverted"));
        for x in [0.1, 0.25, 0.5, 0.8] {
            assert!((at(&cone, x) - at(&inverted, x)).abs() < 0.03, "{x}: {} vs {}", at(&cone, x), at(&inverted, x));
        }
        assert!((at(&satin("Gaussian"), 0.5) - 0.36).abs() < 0.04, "{}", at(&satin("Gaussian"), 0.5));
        assert!((at(&lut("Gaussian"), 0.5) - 0.64).abs() < 0.03, "{}", at(&lut("Gaussian"), 0.5));
        assert!((at(&lut("Cove - Deep"), 0.0) - 0.26).abs() < 0.02, "Cove - Deep starts above zero");
    }

    #[test]
    fn gradient_overlay_method_matches_photoshop() {
        // Photoshop 27.11, Classic Gradient Overlay #000000 → #ffffff (Smoothness 100 %): 0.037 at
        // t = 1/16 and 0.5 in the middle; straight stops would give 0.0625.
        let mut s = session();
        s.execute("layer.layerStyle.gradientOverlay", json!({"from": "#000000", "to": "#ffffff", "method": "classic"})).unwrap();
        let Effect::GradientOverlay { gradient, .. } = &effects(&s)[0] else { panic!() };
        let at = |t: f32| {
            let w = gradient.stops.windows(2).find(|w| t >= w[0].0 && t <= w[1].0).unwrap();
            let u = (t - w[0].0) / (w[1].0 - w[0].0).max(1e-6);
            w[0].1.to_rgb()[0] + (w[1].1.to_rgb()[0] - w[0].1.to_rgb()[0]) * u
        };
        assert!((at(1.0 / 16.0) - 0.037).abs() < 0.004, "{}", at(1.0 / 16.0));
        assert!((at(0.5) - 0.5).abs() < 0.004, "{}", at(0.5));
        // Without a method the two stops stay as they are.
        s.execute("layer.layerStyle.gradientOverlay", json!({"from": "#000000", "to": "#ffffff", "add": true})).unwrap();
        let plain = effects(&s).iter().filter(|e| matches!(e, Effect::GradientOverlay { gradient, .. } if gradient.stops.len() == 2)).count();
        assert_eq!(plain, 1, "the added overlay keeps its two stops");
    }

    #[test]
    fn contour_presets_and_custom_curves_apply_in_one_undo_step() {
        let mut s = session();
        // Picking Cone: the preset's own curve, its points corners, in one undo step.
        s.execute("layer.layerStyle.dropShadow", json!({"contour": "Cone", "antiAlias": true})).unwrap();
        let Effect::DropShadow(sh) = &effects(&s)[0] else { panic!() };
        assert_eq!(sh.contour, builtin_contour("Cone").unwrap());
        assert!(sh.contour.is_corner(1) && sh.anti_alias);
        let Contour::Custom { points, corners, .. } = &sh.contour else { panic!() };
        let lut = photocraft_compose::adjust::contour_curve_lut(points, corners);
        let at = |x: f32| lut[(x * (lut.len() - 1) as f32).round() as usize];
        // Photoshop's measured Cone: 0 at both ends, a 0.99 peak at 0.5, symmetric.
        assert!(at(0.0).abs() < 1e-3 && at(1.0).abs() < 1e-3 && at(0.5) > 0.98, "a Λ: {} {} {}", at(0.0), at(0.5), at(1.0));
        assert!((at(0.25) - at(0.75)).abs() < 0.03, "symmetric: {} vs {}", at(0.25), at(0.75));
        assert_eq!(contour_param(&sh.contour), json!("Cone"));
        s.execute("edit.undo", json!({})).unwrap();
        assert!(effects(&s).is_empty(), "one undo removes it");

        // An edited curve (the Contour Editor's param) round trips exactly, in every effect.
        let curve = json!({"name": "Mine", "points": [[0.0, 0.2], [0.4, 0.9], [1.0, 0.1]], "corners": [false, true, false]});
        for kind in ["dropShadow", "innerShadow", "outerGlow", "innerGlow", "satin"] {
            let mut fx = fresh_effect(kind).unwrap();
            overlay_effect(&mut fx, &json!({"contour": curve}));
            let v = values(&fx, "contour");
            assert_eq!(v, curve, "{kind}");
        }
        let mut bevel = fresh_effect("bevelEmboss").unwrap();
        overlay_effect(&mut bevel, &json!({"contour": true, "contourName": curve, "glossContour": "Ring"}));
        let Effect::BevelEmboss(b) = &bevel else { panic!() };
        assert_eq!(contour_param(&b.contour.as_ref().unwrap().contour), curve);
        assert_eq!(b.gloss_contour, builtin_contour("Ring").unwrap());
        // Malformed curves leave the contour alone.
        for bad in [json!({"points": [[0.5, 0.5]]}), json!({"points": [[0.0, "x"], [1.0, 1.0]]}), json!({"points": "no"}), json!(7)] {
            let mut fx = fresh_effect("dropShadow").unwrap();
            overlay_effect(&mut fx, &json!({"contour": bad}));
            assert_eq!(values(&fx, "contour"), json!("Linear"));
        }

        // `.shc` round trip of a curve with a corner.
        let mine = contour_from_param(&curve).unwrap();
        let back = contours_from_shc(&contour_to_shc(&mine).unwrap()).unwrap();
        let back = contour_from_param(&back[0]).unwrap();
        let Contour::Custom { points, corners, .. } = &back else { panic!() };
        assert_eq!(corners, &vec![false, true, false]);
        assert!((points[1].input - 0.4).abs() < 0.003 && (points[1].output - 0.9).abs() < 0.003, "{points:?}");
    }

    fn values(fx: &Effect, key: &str) -> Value {
        let c = match fx {
            Effect::DropShadow(s) | Effect::InnerShadow(s) => &s.contour,
            Effect::OuterGlow(g) | Effect::InnerGlow(g) => &g.contour,
            Effect::Satin(s) => &s.contour,
            _ => panic!("{key}: no contour"),
        };
        contour_param(c)
    }

    #[test]
    fn make_default_then_reset_restores_it() {
        let mut s = session();
        // Factory default first: Reset without a saved default gives the factory set.
        let r = s.execute("layer.layerStyle.defaultFor", json!({"kind": "stroke"})).unwrap();
        assert_eq!(r["user"], json!(false));
        assert_eq!(r["params"]["size"], json!(3));
        s.execute(
            "layer.layerStyle.makeDefault",
            json!({"kind": "stroke", "params": {"size": 9, "position": "inside", "blend": "Normal", "opacity": 100, "color": "#000000"}}),
        )
        .unwrap();
        let r = s.execute("layer.layerStyle.defaultFor", json!({"kind": "stroke"})).unwrap();
        assert_eq!(r["user"], json!(true));
        assert_eq!(r["params"]["size"], json!(9));
        // Bad kinds and params are graceful errors.
        assert!(s.execute("layer.layerStyle.makeDefault", json!({"kind": "nope", "params": {}})).is_err());
        assert!(s.execute("layer.layerStyle.makeDefault", json!({"kind": "stroke", "params": "x"})).is_err());
        assert!(s.execute("layer.layerStyle.defaultFor", json!({"kind": "nope"})).is_err());
    }

    #[test]
    fn layer_style_defaults_persist_with_presets() {
        let mut s = session();
        s.execute("layer.layerStyle.makeDefault", json!({"kind": "stroke", "params": {"size": 9}})).unwrap();
        let blob = s.presets.to_json(&s);
        let mut t = Session::new();
        t.load_presets_json(blob);
        let r = t.execute("layer.layerStyle.defaultFor", json!({"kind": "stroke"})).unwrap();
        assert_eq!(r["user"], json!(true));
        assert_eq!(r["params"]["size"], json!(9));
    }

    #[test]
    fn double_stroke_bands_render_and_survive_an_edit() {
        // The manga/typesetting case (#157): a 7 px white stroke under a 3 px black one.
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 48, "height": 48})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("select.rect", json!({"x": 16, "y": 16, "width": 16, "height": 16})).unwrap();
        s.execute("edit.fill", json!({"color": "#000000"})).unwrap();
        s.execute("select.deselect", json!({})).unwrap();
        s.execute("layer.layerStyle.stroke", json!({"size": 3, "color": "#000000", "position": "outside"})).unwrap();
        s.execute("layer.layerStyle.stroke", json!({"size": 7, "color": "#ffffff", "position": "outside", "add": true})).unwrap();
        let px = |s: &mut Session, x: i32, y: i32| -> [f32; 4] {
            let mut v: Vec<f32> = serde_json::from_value(s.execute("document.pixel", json!({"x": x, "y": y})).unwrap()).unwrap();
            v.resize(4, 1.0);
            [v[0], v[1], v[2], v[3]]
        };
        // Bands from the square edge outward: 0..3 px black (the first stroke renders on top),
        // 3..7 px white, then the white background.
        assert!(px(&mut s, 24, 24)[0] < 0.1, "the square is black");
        assert!(px(&mut s, 24, 14)[0] < 0.1, "the first stroke covers the wider one");
        assert!(px(&mut s, 24, 11)[0] > 0.9, "the second stroke shows outside it");
        assert!(px(&mut s, 24, 5)[0] > 0.9, "background beyond both");
        // A dialog round trip (replace with the same snapshots) keeps both bands.
        let before = px(&mut s, 24, 14);
        let st = s.active().unwrap();
        let id = st.active_layer.unwrap();
        let items = st.doc.layer(id).unwrap().effects.items.clone();
        let entries: Vec<Value> = items.iter().map(|fx| json!({"kind": "stroke", "fx": serde_json::to_value(fx).unwrap(), "params": {}})).collect();
        s.execute("layer.layerStyle.replace", json!({"layer": id.0, "effects": entries})).unwrap();
        assert_eq!(px(&mut s, 24, 14), before, "both strokes render identically after the round trip");
    }
}
