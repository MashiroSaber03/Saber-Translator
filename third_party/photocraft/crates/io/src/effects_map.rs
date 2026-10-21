//! Layer effects ↔ PSD `lfx2` (object-based effects descriptor) and legacy `lrFX`.
//!
//! Key names follow the descriptors Photoshop writes (documented by
//! reverse-engineering in MIT-licensed psd-tools / ag-psd).

use photocraft_color::{BlendMode, Color};
use photocraft_doc::adjust::CurvePoint;
use photocraft_doc::{
    Bevel, BevelContour, BevelStyle, BevelTechnique, BevelTexture, Contour, Effect, FxCommon, FxPaint, Glow, GlowSource, GlowTechnique, Gradient, Satin,
    Shadow, StrokeFx, StrokePosition,
};
use photocraft_psd::descriptor::{Descriptor, Id, UnicodeString, Value, VersionedDescriptor};

use crate::blocks::{
    bool_of, color_from_desc, color_to_desc, enum_of, get_desc, gradient_desc, gradient_style, gradient_style_value, num, pattern_placement, pattern_ref,
    pattern_ref_desc, with_pattern_placement,
};

const BLEND_NAMES: [(BlendMode, &str); 28] = [
    (BlendMode::Normal, "Nrml"),
    (BlendMode::Dissolve, "Dslv"),
    (BlendMode::Darken, "Drkn"),
    (BlendMode::Multiply, "Mltp"),
    (BlendMode::ColorBurn, "CBrn"),
    (BlendMode::LinearBurn, "linearBurn"),
    (BlendMode::DarkerColor, "darkerColor"),
    (BlendMode::Lighten, "Lghn"),
    (BlendMode::Screen, "Scrn"),
    (BlendMode::ColorDodge, "CDdg"),
    (BlendMode::LinearDodge, "linearDodge"),
    (BlendMode::LighterColor, "lighterColor"),
    (BlendMode::Overlay, "Ovrl"),
    (BlendMode::SoftLight, "SftL"),
    (BlendMode::HardLight, "HrdL"),
    (BlendMode::VividLight, "vividLight"),
    (BlendMode::LinearLight, "linearLight"),
    (BlendMode::PinLight, "pinLight"),
    (BlendMode::HardMix, "hardMix"),
    (BlendMode::Difference, "Dfrn"),
    (BlendMode::Exclusion, "Xclu"),
    (BlendMode::Subtract, "blendSubtraction"),
    (BlendMode::Divide, "blendDivide"),
    (BlendMode::Hue, "H   "),
    (BlendMode::Saturation, "Strt"),
    (BlendMode::Color, "Clr "),
    (BlendMode::Luminosity, "Lmns"),
    (BlendMode::PassThrough, "passThrough"),
];

/// Long (string-ID) names some writers use instead of the four-character codes, for the modes
/// whose code is a short ID (`BlnM` `multiply` = `Mltp`). The others already use their long name.
const BLEND_LONG_NAMES: [(BlendMode, &str); 17] = [
    (BlendMode::Normal, "normal"),
    (BlendMode::Dissolve, "dissolve"),
    (BlendMode::Darken, "darken"),
    (BlendMode::Multiply, "multiply"),
    (BlendMode::ColorBurn, "colorBurn"),
    (BlendMode::Lighten, "lighten"),
    (BlendMode::Screen, "screen"),
    (BlendMode::ColorDodge, "colorDodge"),
    (BlendMode::Overlay, "overlay"),
    (BlendMode::SoftLight, "softLight"),
    (BlendMode::HardLight, "hardLight"),
    (BlendMode::Difference, "difference"),
    (BlendMode::Exclusion, "exclusion"),
    (BlendMode::Hue, "hue"),
    (BlendMode::Saturation, "saturation"),
    (BlendMode::Color, "color"),
    (BlendMode::Luminosity, "luminosity"),
];

/// A blend-mode enum from either its four-character code or its long string ID.
pub(crate) fn blend_from_id(v: &[u8]) -> Option<BlendMode> {
    BLEND_NAMES.iter().chain(&BLEND_LONG_NAMES).find(|(_, n)| n.as_bytes() == v).map(|(m, _)| *m)
}

fn blend_of(d: &Descriptor, key: &str, default: BlendMode) -> BlendMode {
    enum_of(d, key).and_then(blend_from_id).unwrap_or(default)
}

/// A `BlnM` enum with the long string ID where the mode has one (`multiply`, `overlay`), as
/// Photoshop writes smart-filter blend options; the four-character code otherwise.
pub(crate) fn blend_long_value(m: BlendMode) -> Value {
    let n = BLEND_LONG_NAMES.iter().chain(&BLEND_NAMES).find(|(b, _)| *b == m).map_or("normal", |(_, n)| n);
    Value::Enumerated { type_id: Id::new("BlnM"), value: Id::new(n) }
}

fn blend_value(m: BlendMode) -> Value {
    let n = BLEND_NAMES.iter().find(|(b, _)| *b == m).map_or("Nrml", |(_, n)| n);
    Value::Enumerated { type_id: Id::new("BlnM"), value: Id::new(n) }
}

fn f(d: &Descriptor, key: &str, default: f32) -> f32 {
    num(d.get(key)).map_or(default, |v| v as f32)
}
fn pct(d: &Descriptor, key: &str, default: f32) -> f32 {
    f(d, key, default * 100.0) / 100.0
}
fn color(d: &Descriptor, key: &str) -> Color {
    get_desc(d, key).and_then(color_from_desc).unwrap_or(Color::BLACK)
}
fn contour(d: &Descriptor, key: &str) -> Contour {
    let Some(c) = get_desc(d, key) else { return Contour::Linear };
    let name = match c.get("Nm  ") {
        Some(Value::Text(t)) => t.to_string_lossy(),
        _ => String::new(),
    };
    let mut points = Vec::new();
    let mut corners = Vec::new();
    if let Some(Value::List(items)) = c.get("Crv ") {
        for it in items {
            if let Value::Descriptor(p) = it {
                points.push(CurvePoint { input: f(p, "Hrzn", 0.0) / 255.0, output: f(p, "Vrtc", 0.0) / 255.0 });
                // `Cnty` (continuity): false marks a corner point; absent means smooth.
                corners.push(matches!(p.get("Cnty"), Some(Value::Boolean(false))));
            }
        }
    }
    if !corners.contains(&true) {
        corners.clear();
    }
    let linear = points.len() == 2 && points[0].input == 0.0 && points[0].output == 0.0 && points[1].input == 1.0 && points[1].output == 1.0;
    if points.is_empty() || (linear && (name.is_empty() || name == "Linear")) { Contour::Linear } else { Contour::Custom { name, points, corners } }
}
fn contour_value(c: &Contour) -> Value {
    let (name, points) = match c {
        Contour::Linear => ("Linear".to_string(), vec![CurvePoint { input: 0.0, output: 0.0 }, CurvePoint { input: 1.0, output: 1.0 }]),
        Contour::Custom { name, points, .. } => (name.clone(), points.clone()),
    };
    let pts = points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let d = Descriptor::new("CrPt").with("Hrzn", Value::Double(f64::from(p.input * 255.0))).with("Vrtc", Value::Double(f64::from(p.output * 255.0)));
            // Photoshop writes no `Cnty` (continuity) on smooth points; a corner carries `Cnty` false.
            Value::Descriptor(if c.is_corner(i) { d.with("Cnty", Value::Boolean(false)) } else { d })
        })
        .collect();
    Value::Descriptor(Descriptor::new("ShpC").with("Nm  ", Value::Text(UnicodeString::new_nul(&name))).with("Crv ", Value::List(pts)))
}

fn common(d: &Descriptor, blend: BlendMode, opacity: f32) -> FxCommon {
    FxCommon { enabled: !matches!(d.get("enab"), Some(Value::Boolean(false))), blend: blend_of(d, "Md  ", blend), opacity: pct(d, "Opct", opacity) }
}

fn gradient_from(d: &Descriptor) -> Gradient {
    let (stops, opacity_stops) = get_desc(d, "Grad").map(|g| crate::blocks::gradient_stops_with(g, crate::blocks::gradient_method(d))).unwrap_or_default();
    let off = get_desc(d, "Ofst");
    Gradient {
        stops: if stops.is_empty() { Gradient::default().stops } else { stops },
        opacity_stops,
        style: gradient_style(d),
        angle: f(d, "Angl", 90.0),
        scale: pct(d, "Scl ", 1.0),
        reverse: bool_of(d, "Rvrs"),
        align: !matches!(d.get("Algn"), Some(Value::Boolean(false))),
        offset: off.map_or((0.0, 0.0), |o| (pct(o, "Hrzn", 0.0), pct(o, "Vrtc", 0.0))),
    }
}

fn parse_shadow(d: &Descriptor, inner: bool) -> Shadow {
    Shadow {
        common: common(d, BlendMode::Multiply, 0.75),
        color: color(d, "Clr "),
        angle: f(d, "lagl", 120.0),
        use_global_light: !matches!(d.get("uglg"), Some(Value::Boolean(false))),
        distance: f(d, "Dstn", 5.0),
        spread: pct(d, "Ckmt", 0.0),
        size: f(d, "blur", 5.0),
        contour: contour(d, "TrnS"),
        anti_alias: bool_of(d, "AntA"),
        noise: pct(d, "Nose", 0.0),
        knocks_out: !inner && !matches!(d.get("layerConceals"), Some(Value::Boolean(false))),
    }
}

fn parse_glow(d: &Descriptor, inner: bool) -> Glow {
    let paint = if d.get("Grad").is_some() { FxPaint::Gradient(gradient_from(d)) } else { FxPaint::Color(color(d, "Clr ")) };
    Glow {
        common: common(d, BlendMode::Screen, 0.75),
        paint,
        technique: if enum_of(d, "GlwT") == Some(b"PrBL") { GlowTechnique::Precise } else { GlowTechnique::Softer },
        spread: pct(d, "Ckmt", 0.0),
        size: f(d, "blur", 5.0),
        contour: contour(d, "TrnS"),
        anti_alias: bool_of(d, "AntA"),
        range: pct(d, "Inpr", 0.5),
        jitter: pct(d, "ShdN", 0.0),
        noise: pct(d, "Nose", 0.0),
        source: if inner && enum_of(d, "glwS") == Some(b"SrcC") { GlowSource::Center } else { GlowSource::Edge },
    }
}

fn parse_stroke(d: &Descriptor) -> StrokeFx {
    let paint = match enum_of(d, "PntT") {
        Some(b"GrFl") => FxPaint::Gradient(gradient_from(d)),
        Some(b"Ptrn") => {
            let p = get_desc(d, "Ptrn");
            let text = |k: &str| match p.and_then(|p| p.get(k)) {
                Some(Value::Text(t)) => t.to_string_lossy(),
                _ => String::new(),
            };
            FxPaint::Pattern { name: text("Nm  "), id: text("Idnt"), scale: pct(d, "Scl ", 1.0) }
        }
        _ => FxPaint::Color(color(d, "Clr ")),
    };
    StrokeFx {
        common: common(d, BlendMode::Normal, 1.0),
        size: f(d, "Sz  ", 3.0),
        position: match enum_of(d, "Styl") {
            Some(b"InsF") => StrokePosition::Inside,
            Some(b"CtrF") => StrokePosition::Center,
            _ => StrokePosition::Outside,
        },
        paint,
    }
}

fn parse_bevel(d: &Descriptor) -> Bevel {
    Bevel {
        enabled: !matches!(d.get("enab"), Some(Value::Boolean(false))),
        style: match enum_of(d, "bvlS") {
            Some(b"OtrB") => BevelStyle::OuterBevel,
            Some(b"Embs") => BevelStyle::Emboss,
            Some(b"PlEb") => BevelStyle::PillowEmboss,
            Some(b"strokeEmboss") => BevelStyle::StrokeEmboss,
            _ => BevelStyle::InnerBevel,
        },
        technique: match enum_of(d, "bvlT") {
            Some(b"PrBL") => BevelTechnique::ChiselHard,
            Some(b"Slmt") => BevelTechnique::ChiselSoft,
            _ => BevelTechnique::Smooth,
        },
        depth: pct(d, "srgR", 1.0),
        up: enum_of(d, "bvlD") != Some(b"Out "),
        size: f(d, "blur", 5.0),
        soften: f(d, "Sftn", 0.0),
        angle: f(d, "lagl", 120.0),
        altitude: f(d, "Lald", 30.0),
        use_global_light: !matches!(d.get("uglg"), Some(Value::Boolean(false))),
        gloss_contour: contour(d, "TrnS"),
        highlight: FxCommon { enabled: true, blend: blend_of(d, "hglM", BlendMode::Screen), opacity: pct(d, "hglO", 0.75) },
        highlight_color: get_desc(d, "hglC").and_then(color_from_desc).unwrap_or(Color::WHITE),
        shadow: FxCommon { enabled: true, blend: blend_of(d, "sdwM", BlendMode::Multiply), opacity: pct(d, "sdwO", 0.75) },
        shadow_color: color(d, "sdwC"),
        contour: bool_of(d, "useShape").then(|| BevelContour { contour: contour(d, "MpgS"), range: pct(d, "Inpr", 0.5), anti_alias: bool_of(d, "AntA") }),
        texture: bool_of(d, "useTexture").then(|| {
            let (name, id) = pattern_ref(d);
            let (_, link, phase) = pattern_placement(d);
            BevelTexture { name, id, scale: pct(d, "Scl ", 1.0), depth: pct(d, "textureDepth", 1.0), invert: bool_of(d, "InvT"), link, phase }
        }),
    }
}

fn parse_one(key: &str, d: &Descriptor) -> Option<Effect> {
    if matches!(d.get("present"), Some(Value::Boolean(false))) {
        return None;
    }
    Some(match key {
        "DrSh" => Effect::DropShadow(parse_shadow(d, false)),
        "IrSh" => Effect::InnerShadow(parse_shadow(d, true)),
        "OrGl" => Effect::OuterGlow(parse_glow(d, false)),
        "IrGl" => Effect::InnerGlow(parse_glow(d, true)),
        "FrFX" => Effect::Stroke(parse_stroke(d)),
        "SoFi" => Effect::ColorOverlay { common: common(d, BlendMode::Normal, 1.0), color: color(d, "Clr ") },
        "GrFl" => Effect::GradientOverlay { common: common(d, BlendMode::Normal, 1.0), gradient: gradient_from(d), dither: bool_of(d, "Dthr") },
        "patternFill" => {
            let (name, id) = pattern_ref(d);
            let (angle, link, phase) = pattern_placement(d);
            Effect::PatternOverlay { common: common(d, BlendMode::Normal, 1.0), name, id, scale: pct(d, "Scl ", 1.0), angle, link, phase }
        }
        "ChFX" => Effect::Satin(Satin {
            common: common(d, BlendMode::Multiply, 0.5),
            color: color(d, "Clr "),
            angle: f(d, "lagl", 19.0),
            distance: f(d, "Dstn", 11.0),
            size: f(d, "blur", 14.0),
            contour: contour(d, "MpgS"),
            anti_alias: bool_of(d, "AntA"),
            invert: bool_of(d, "Invr"),
        }),
        "ebbl" => Effect::BevelEmboss(parse_bevel(d)),
        _ => return None,
    })
}

/// Single-instance key and multi-instance list key for each kind.
const KINDS: [(&str, &str); 10] = [
    ("DrSh", "dropShadowMulti"),
    ("IrSh", "innerShadowMulti"),
    ("OrGl", ""),
    ("IrGl", ""),
    ("ebbl", ""),
    ("ChFX", ""),
    ("SoFi", "solidFillMulti"),
    ("GrFl", "gradientFillMulti"),
    ("patternFill", ""),
    ("FrFX", "frameFXMulti"),
];

/// Parses `lfx2` block data: `(master switch, effects)`.
pub fn parse_lfx2(data: &[u8]) -> Option<(bool, Vec<Effect>)> {
    let (vd, _) = VersionedDescriptor::parse_prefix(data.get(4..)?).ok()?;
    let d = vd.descriptor;
    let master = !matches!(d.get("masterFXSwitch"), Some(Value::Boolean(false)));
    let mut out = Vec::new();
    for (single, multi) in KINDS {
        let from_multi = (!multi.is_empty()).then(|| d.get(multi)).flatten();
        match from_multi {
            Some(Value::List(items)) => {
                for it in items {
                    if let Value::Descriptor(x) = it
                        && let Some(e) = parse_one(single, x)
                    {
                        out.push(e);
                    }
                }
            }
            _ => {
                if let Some(x) = get_desc(&d, single)
                    && let Some(e) = parse_one(single, x)
                {
                    out.push(e);
                }
            }
        }
    }
    Some((master, out))
}

fn unit(u: &[u8; 4], v: f32) -> Value {
    Value::UnitFloat { unit: *u, value: f64::from(v) }
}

fn with_common(d: Descriptor, c: &FxCommon) -> Descriptor {
    d.with("enab", Value::Boolean(c.enabled))
        .with("present", Value::Boolean(true))
        .with("showInDialog", Value::Boolean(true))
        .with("Md  ", blend_value(c.blend))
        .with("Opct", unit(b"#Prc", c.opacity * 100.0))
}

fn gradient_keys(mut d: Descriptor, g: &Gradient) -> Descriptor {
    d = d
        .with("Grad", Value::Descriptor(gradient_desc(&g.stops, &g.opacity_stops)))
        .with("Angl", unit(b"#Ang", g.angle))
        .with("Type", gradient_style_value(g.style))
        .with("Rvrs", Value::Boolean(g.reverse))
        .with("Algn", Value::Boolean(g.align))
        .with("Scl ", unit(b"#Prc", g.scale * 100.0));
    d.with("Ofst", Value::Descriptor(Descriptor::new("Pnt ").with("Hrzn", unit(b"#Prc", g.offset.0 * 100.0)).with("Vrtc", unit(b"#Prc", g.offset.1 * 100.0))))
}

fn write_one(e: &Effect) -> (&'static str, Descriptor) {
    match e {
        Effect::DropShadow(s) | Effect::InnerShadow(s) => {
            let inner = matches!(e, Effect::InnerShadow(_));
            let mut d = with_common(Descriptor::new(if inner { "IrSh" } else { "DrSh" }), &s.common)
                .with("Clr ", Value::Descriptor(color_to_desc(&s.color)))
                .with("uglg", Value::Boolean(s.use_global_light))
                .with("lagl", unit(b"#Ang", s.angle))
                .with("Dstn", unit(b"#Pxl", s.distance))
                .with("Ckmt", unit(b"#Pxl", s.spread * 100.0))
                .with("blur", unit(b"#Pxl", s.size))
                .with("Nose", unit(b"#Prc", s.noise * 100.0))
                .with("AntA", Value::Boolean(s.anti_alias))
                .with("TrnS", contour_value(&s.contour));
            if !inner {
                d = d.with("layerConceals", Value::Boolean(s.knocks_out));
            }
            (if inner { "IrSh" } else { "DrSh" }, d)
        }
        Effect::OuterGlow(g) | Effect::InnerGlow(g) => {
            let inner = matches!(e, Effect::InnerGlow(_));
            let mut d = with_common(Descriptor::new(if inner { "IrGl" } else { "OrGl" }), &g.common);
            d = match &g.paint {
                FxPaint::Gradient(gr) => gradient_keys(d, gr),
                FxPaint::Color(c) => d.with("Clr ", Value::Descriptor(color_to_desc(c))),
                FxPaint::Pattern { .. } => d.with("Clr ", Value::Descriptor(color_to_desc(&Color::WHITE))),
            };
            d = d
                .with(
                    "GlwT",
                    Value::Enumerated { type_id: Id::new("BETE"), value: Id::new(if g.technique == GlowTechnique::Precise { "PrBL" } else { "SfBL" }) },
                )
                .with("Ckmt", unit(b"#Pxl", g.spread * 100.0))
                .with("blur", unit(b"#Pxl", g.size))
                .with("Nose", unit(b"#Prc", g.noise * 100.0))
                .with("ShdN", unit(b"#Prc", g.jitter * 100.0))
                .with("AntA", Value::Boolean(g.anti_alias))
                .with("TrnS", contour_value(&g.contour))
                .with("Inpr", unit(b"#Prc", g.range * 100.0));
            if inner {
                d = d
                    .with("glwS", Value::Enumerated { type_id: Id::new("IGSr"), value: Id::new(if g.source == GlowSource::Center { "SrcC" } else { "SrcE" }) });
            }
            (if inner { "IrGl" } else { "OrGl" }, d)
        }
        Effect::Stroke(s) => {
            let styl = match s.position {
                StrokePosition::Outside => "OutF",
                StrokePosition::Inside => "InsF",
                StrokePosition::Center => "CtrF",
            };
            let mut d = with_common(Descriptor::new("FrFX"), &s.common)
                .with("Styl", Value::Enumerated { type_id: Id::new("FStl"), value: Id::new(styl) })
                .with("Sz  ", unit(b"#Pxl", s.size));
            d = match &s.paint {
                FxPaint::Color(c) => {
                    d.with("PntT", Value::Enumerated { type_id: Id::new("FrFl"), value: Id::new("SClr") }).with("Clr ", Value::Descriptor(color_to_desc(c)))
                }
                FxPaint::Gradient(g) => gradient_keys(d.with("PntT", Value::Enumerated { type_id: Id::new("FrFl"), value: Id::new("GrFl") }), g),
                FxPaint::Pattern { name, id, scale } => d
                    .with("PntT", Value::Enumerated { type_id: Id::new("FrFl"), value: Id::new("Ptrn") })
                    .with(
                        "Ptrn",
                        Value::Descriptor(
                            Descriptor::new("Ptrn")
                                .with("Nm  ", Value::Text(UnicodeString::new_nul(name)))
                                .with("Idnt", Value::Text(UnicodeString::new_nul(id))),
                        ),
                    )
                    .with("Scl ", unit(b"#Prc", scale * 100.0)),
            };
            ("FrFX", d)
        }
        Effect::ColorOverlay { common: c, color } => ("SoFi", with_common(Descriptor::new("SoFi"), c).with("Clr ", Value::Descriptor(color_to_desc(color)))),
        Effect::GradientOverlay { common: c, gradient, dither } => {
            ("GrFl", gradient_keys(with_common(Descriptor::new("GrFl"), c), gradient).with("Dthr", Value::Boolean(*dither)))
        }
        Effect::PatternOverlay { common: c, name, id, scale, angle, link, phase } => (
            "patternFill",
            with_pattern_placement(
                with_common(Descriptor::new("patternFill"), c)
                    .with("Ptrn", Value::Descriptor(pattern_ref_desc(name, id)))
                    .with("Scl ", unit(b"#Prc", scale * 100.0)),
                *angle,
                *link,
                *phase,
            ),
        ),
        Effect::Satin(s) => (
            "ChFX",
            with_common(Descriptor::new("ChFX"), &s.common)
                .with("Clr ", Value::Descriptor(color_to_desc(&s.color)))
                .with("AntA", Value::Boolean(s.anti_alias))
                .with("Invr", Value::Boolean(s.invert))
                .with("lagl", unit(b"#Ang", s.angle))
                .with("Dstn", unit(b"#Pxl", s.distance))
                .with("blur", unit(b"#Pxl", s.size))
                .with("MpgS", contour_value(&s.contour)),
        ),
        Effect::BevelEmboss(b) => {
            let style = match b.style {
                BevelStyle::OuterBevel => "OtrB",
                BevelStyle::InnerBevel => "InrB",
                BevelStyle::Emboss => "Embs",
                BevelStyle::PillowEmboss => "PlEb",
                BevelStyle::StrokeEmboss => "strokeEmboss",
            };
            let tech = match b.technique {
                BevelTechnique::Smooth => "SfBL",
                BevelTechnique::ChiselHard => "PrBL",
                BevelTechnique::ChiselSoft => "Slmt",
            };
            let d = Descriptor::new("ebbl")
                .with("enab", Value::Boolean(b.enabled))
                .with("present", Value::Boolean(true))
                .with("showInDialog", Value::Boolean(true))
                .with("hglM", blend_value(b.highlight.blend))
                .with("hglC", Value::Descriptor(color_to_desc(&b.highlight_color)))
                .with("hglO", unit(b"#Prc", b.highlight.opacity * 100.0))
                .with("sdwM", blend_value(b.shadow.blend))
                .with("sdwC", Value::Descriptor(color_to_desc(&b.shadow_color)))
                .with("sdwO", unit(b"#Prc", b.shadow.opacity * 100.0))
                .with("bvlT", Value::Enumerated { type_id: Id::new("bvlT"), value: Id::new(tech) })
                .with("bvlS", Value::Enumerated { type_id: Id::new("BESl"), value: Id::new(style) })
                .with("uglg", Value::Boolean(b.use_global_light))
                .with("lagl", unit(b"#Ang", b.angle))
                .with("Lald", unit(b"#Ang", b.altitude))
                .with("srgR", unit(b"#Prc", b.depth * 100.0))
                .with("blur", unit(b"#Pxl", b.size))
                .with("bvlD", Value::Enumerated { type_id: Id::new("BESs"), value: Id::new(if b.up { "In  " } else { "Out " }) })
                .with("TrnS", contour_value(&b.gloss_contour))
                .with("Sftn", unit(b"#Pxl", b.soften));
            let d = match &b.contour {
                Some(c) => d
                    .with("useShape", Value::Boolean(true))
                    .with("MpgS", contour_value(&c.contour))
                    .with("AntA", Value::Boolean(c.anti_alias))
                    .with("Inpr", unit(b"#Prc", c.range * 100.0)),
                None => d.with("useShape", Value::Boolean(false)),
            };
            let d = match &b.texture {
                Some(t) => with_pattern_placement(
                    d.with("useTexture", Value::Boolean(true))
                        .with("InvT", Value::Boolean(t.invert))
                        .with("Scl ", unit(b"#Prc", t.scale * 100.0))
                        .with("textureDepth", unit(b"#Prc", t.depth * 100.0))
                        .with("Ptrn", Value::Descriptor(pattern_ref_desc(&t.name, &t.id))),
                    0.0,
                    t.link,
                    t.phase,
                ),
                None => d.with("useTexture", Value::Boolean(false)),
            };
            ("ebbl", d)
        }
    }
}

/// Serializes effects as `lfx2` block data.
pub fn write_lfx2(master: bool, items: &[Effect]) -> Vec<u8> {
    let mut d = Descriptor::new("null").with("Scl ", unit(b"#Prc", 100.0)).with("masterFXSwitch", Value::Boolean(master));
    for (single, multi) in KINDS {
        let of_kind: Vec<Descriptor> = items.iter().map(write_one).filter(|(k, _)| *k == single).map(|(_, d)| d).collect();
        match of_kind.len() {
            0 => {}
            1 => d = d.with(single, Value::Descriptor(of_kind.into_iter().next().unwrap_or_default())),
            _ if !multi.is_empty() => d = d.with(multi, Value::List(of_kind.into_iter().map(Value::Descriptor).collect())),
            // Kinds without a multi key keep only the first instance.
            _ => d = d.with(single, Value::Descriptor(of_kind.into_iter().next().unwrap_or_default())),
        }
    }
    let mut out = 0u32.to_be_bytes().to_vec();
    out.extend(VersionedDescriptor::new(d).to_bytes());
    out
}

fn fixed(v: u32) -> f32 {
    v as i32 as f32 / 65536.0
}

/// Parses legacy `lrFX` effects (Photoshop 5 layout): drop/inner shadows,
/// outer/inner glows and solid fill. Bevels are skipped.
pub fn parse_lrfx(data: &[u8]) -> Option<(bool, Vec<Effect>)> {
    let be16 = |at: usize| data.get(at..at + 2).map(|b| u16::from_be_bytes([b[0], b[1]]));
    let be32 = |at: usize| data.get(at..at + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]));
    let color_at = |at: usize| -> Option<Color> {
        // space (2) + 4 × u16 components (RGB uses the first three).
        let c = |k: usize| be16(at + 2 + k * 2).map(|v| f32::from(v >> 8) / 255.0);
        Some(Color::rgb(c(0)?, c(1)?, c(2)?))
    };
    let blend_at = |at: usize| -> BlendMode {
        data.get(at + 4..at + 8).and_then(|k| photocraft_color::BlendMode::from_psd_key([k[0], k[1], k[2], k[3]])).unwrap_or(BlendMode::Normal)
    };
    let count = be16(2)?;
    let mut at = 4;
    let mut out = Vec::new();
    for _ in 0..count {
        let sig = data.get(at..at + 4)?;
        if sig != b"8BIM" {
            return None;
        }
        let key: [u8; 4] = data.get(at + 4..at + 8)?.try_into().ok()?;
        let size = be32(at + 8)? as usize;
        let b = at + 12; // start of this effect's data (after the size field)
        match &key {
            b"dsdw" | b"isdw" => {
                // Blur, angle and distance are 16.16 fixed point in practice.
                let blur = fixed(be32(b + 4)?);
                let angle = fixed(be32(b + 12)?);
                let distance = fixed(be32(b + 16)?);
                let color = color_at(b + 20)?;
                let blend = blend_at(b + 30);
                let enabled = *data.get(b + 38)? != 0;
                let use_global = *data.get(b + 39)? != 0;
                let opacity = f32::from(*data.get(b + 40)?) / 255.0;
                let s = Shadow {
                    common: FxCommon { enabled, blend, opacity },
                    color,
                    angle,
                    use_global_light: use_global,
                    distance,
                    spread: 0.0,
                    size: blur,
                    contour: Contour::Linear,
                    anti_alias: false,
                    noise: 0.0,
                    knocks_out: &key == b"dsdw",
                };
                out.push(if &key == b"dsdw" { Effect::DropShadow(s) } else { Effect::InnerShadow(s) });
            }
            b"oglw" | b"iglw" => {
                let blur = fixed(be32(b + 4)?);
                let color = color_at(b + 12)?;
                let blend = blend_at(b + 22);
                let enabled = *data.get(b + 30)? != 0;
                let opacity = f32::from(*data.get(b + 31)?) / 255.0;
                let g = Glow {
                    common: FxCommon { enabled, blend, opacity },
                    paint: FxPaint::Color(color),
                    technique: GlowTechnique::Softer,
                    spread: 0.0,
                    size: blur,
                    contour: Contour::Linear,
                    anti_alias: false,
                    range: 0.5,
                    jitter: 0.0,
                    noise: 0.0,
                    source: GlowSource::Edge,
                };
                out.push(if &key == b"oglw" { Effect::OuterGlow(g) } else { Effect::InnerGlow(g) });
            }
            b"sofi" => {
                let blend = blend_at(b + 4);
                let color = color_at(b + 12)?;
                let opacity = f32::from(*data.get(b + 22)?) / 255.0;
                let enabled = *data.get(b + 23)? != 0;
                out.push(Effect::ColorOverlay { common: FxCommon { enabled, blend, opacity }, color });
            }
            _ => {}
        }
        at = b + size;
    }
    Some((true, out))
}

#[cfg(test)]
#[allow(clippy::unreachable)] // clippy.toml exempts unwrap/expect/panic in tests, not unreachable!
mod tests {
    use super::*;

    fn sample_effects() -> Vec<Effect> {
        let g = Gradient {
            stops: vec![(0.0, Color::rgb(1.0, 0.0, 0.0)), (1.0, Color::rgb(0.0, 0.0, 1.0))],
            opacity_stops: vec![(0.0, 1.0), (1.0, 0.5)],
            ..Gradient::default()
        };
        vec![
            Effect::default_drop_shadow(),
            Effect::DropShadow(Shadow {
                distance: 9.0,
                spread: 0.25,
                contour: Contour::Custom {
                    name: "Cone".into(),
                    points: vec![CurvePoint { input: 0.0, output: 0.0 }, CurvePoint { input: 0.5, output: 1.0 }, CurvePoint { input: 1.0, output: 0.0 }],
                    corners: vec![false, true, false],
                },
                ..match Effect::default_drop_shadow() {
                    Effect::DropShadow(s) => s,
                    _ => unreachable!(),
                }
            }),
            Effect::InnerShadow(Shadow {
                knocks_out: false,
                ..match Effect::default_drop_shadow() {
                    Effect::DropShadow(s) => s,
                    _ => unreachable!(),
                }
            }),
            Effect::OuterGlow(Glow {
                common: FxCommon::new(BlendMode::Screen, 0.5),
                paint: FxPaint::Color(Color::rgb(1.0, 1.0, 0.0)),
                technique: GlowTechnique::Precise,
                spread: 0.1,
                size: 8.0,
                contour: Contour::Linear,
                anti_alias: true,
                range: 0.5,
                jitter: 0.0,
                noise: 0.0,
                source: GlowSource::Edge,
            }),
            Effect::InnerGlow(Glow {
                common: FxCommon::new(BlendMode::Screen, 0.75),
                paint: FxPaint::Gradient(g.clone()),
                technique: GlowTechnique::Softer,
                spread: 0.0,
                size: 5.0,
                contour: Contour::Linear,
                anti_alias: false,
                range: 0.5,
                jitter: 0.0,
                noise: 0.0,
                source: GlowSource::Center,
            }),
            Effect::Stroke(StrokeFx {
                common: FxCommon::new(BlendMode::Normal, 1.0),
                size: 3.0,
                position: StrokePosition::Inside,
                paint: FxPaint::Color(Color::rgb(0.0, 1.0, 0.0)),
            }),
            Effect::Stroke(StrokeFx {
                common: FxCommon::new(BlendMode::Multiply, 0.5),
                size: 6.0,
                position: StrokePosition::Center,
                paint: FxPaint::Gradient(g.clone()),
            }),
            Effect::ColorOverlay { common: FxCommon::new(BlendMode::Overlay, 0.5), color: Color::rgb(0.0, 0.0, 1.0) },
            Effect::GradientOverlay { common: FxCommon::new(BlendMode::Normal, 1.0), gradient: g, dither: true },
            Effect::PatternOverlay {
                common: FxCommon::new(BlendMode::Normal, 1.0),
                name: "Bubbles".into(),
                id: "abc".into(),
                scale: 1.0,
                angle: 15.0,
                link: false,
                phase: (4.0, 2.0),
            },
            Effect::Satin(Satin {
                common: FxCommon::new(BlendMode::Multiply, 0.5),
                color: Color::BLACK,
                angle: 19.0,
                distance: 11.0,
                size: 14.0,
                contour: Contour::Linear,
                anti_alias: true,
                invert: true,
            }),
            Effect::BevelEmboss(Bevel {
                enabled: true,
                style: BevelStyle::Emboss,
                technique: BevelTechnique::ChiselSoft,
                depth: 1.5,
                up: false,
                size: 7.0,
                soften: 2.0,
                angle: 45.0,
                altitude: 40.0,
                use_global_light: false,
                gloss_contour: Contour::Linear,
                highlight: FxCommon::new(BlendMode::Screen, 0.75),
                highlight_color: Color::WHITE,
                shadow: FxCommon::new(BlendMode::Multiply, 0.6),
                shadow_color: Color::BLACK,
                contour: Some(BevelContour { contour: Contour::Linear, range: 0.7, anti_alias: true }),
                texture: Some(BevelTexture { name: "Bubbles".into(), id: "abc".into(), scale: 0.5, depth: -2.0, invert: true, link: false, phase: (3.0, 1.0) }),
            }),
        ]
    }

    fn approx(a: &[Effect], b: &[Effect]) {
        // Colours go through 0..255 doubles; compare via re-serialization.
        assert_eq!(a.len(), b.len());
        assert_eq!(write_lfx2(true, a), write_lfx2(true, b));
    }

    #[test]
    fn lfx2_roundtrip_all_kinds() {
        let fx = sample_effects();
        let data = write_lfx2(false, &fx);
        let (master, back) = parse_lfx2(&data).unwrap();
        assert!(!master);
        approx(&fx, &back);
        // Kind order is canonical: shadows first ... strokes last.
        assert!(matches!(back[0], Effect::DropShadow(_)));
        assert!(matches!(back.last().unwrap(), Effect::Stroke(_)));
    }

    #[test]
    fn lfx2_ignores_absent_effects_but_keeps_disabled_configured_effects() {
        let d = Descriptor::new("null")
            .with("DrSh", Value::Descriptor(Descriptor::new("DrSh").with("present", Value::Boolean(false)).with("enab", Value::Boolean(false))))
            .with("FrFX", Value::Descriptor(Descriptor::new("FrFX").with("present", Value::Boolean(true)).with("enab", Value::Boolean(false))))
            .with("SoFi", Value::Descriptor(Descriptor::new("SoFi").with("enab", Value::Boolean(false))));
        let mut data = 0u32.to_be_bytes().to_vec();
        data.extend(VersionedDescriptor::new(d).to_bytes());

        let (_, effects) = parse_lfx2(&data).unwrap();

        assert_eq!(effects.len(), 2);
        assert!(effects.iter().all(|effect| !effect.enabled()));
        assert!(effects.iter().any(|effect| effect.label() == "Stroke"));
        assert!(effects.iter().any(|effect| effect.label() == "Color Overlay"));
    }

    #[test]
    fn multi_instances_use_multi_keys() {
        let fx = sample_effects();
        let data = write_lfx2(true, &fx);
        let (vd, _) = VersionedDescriptor::parse_prefix(&data[4..]).unwrap();
        assert!(vd.descriptor.get("dropShadowMulti").is_some());
        assert!(vd.descriptor.get("frameFXMulti").is_some());
        assert!(vd.descriptor.get("SoFi").is_some());
    }

    #[test]
    fn blend_names_are_unique() {
        let mut s = std::collections::HashSet::new();
        for (_, n) in BLEND_NAMES {
            assert!(s.insert(n));
        }
    }

    #[test]
    fn long_string_blend_ids_parse() {
        // Some writers store effect modes as long string IDs (`BlnM` `multiply`, `colorDodge`)
        // instead of the four-character codes (psd-tools effects/blend-modes.psd).
        let long = |v: &str| Value::Enumerated { type_id: Id::new("BlnM"), value: Id::new(v) };
        let d = Descriptor::new("null")
            .with("masterFXSwitch", Value::Boolean(true))
            .with("DrSh", Value::Descriptor(Descriptor::new("DrSh").with("enab", Value::Boolean(true)).with("Md  ", long("linearBurn"))))
            .with("ebbl", Value::Descriptor(Descriptor::new("ebbl").with("hglM", long("colorDodge")).with("sdwM", long("colorBurn"))))
            .with("SoFi", Value::Descriptor(Descriptor::new("SoFi").with("Md  ", long("hue"))));
        let mut data = 0u32.to_be_bytes().to_vec();
        data.extend(VersionedDescriptor { version: 16, descriptor: d }.to_bytes());
        let (_, fx) = parse_lfx2(&data).unwrap();
        let modes: Vec<BlendMode> = fx
            .iter()
            .flat_map(|e| match e {
                Effect::DropShadow(s) => vec![s.common.blend],
                Effect::BevelEmboss(b) => vec![b.highlight.blend, b.shadow.blend],
                Effect::ColorOverlay { common, .. } => vec![common.blend],
                _ => vec![],
            })
            .collect();
        assert_eq!(modes, vec![BlendMode::LinearBurn, BlendMode::ColorDodge, BlendMode::ColorBurn, BlendMode::Hue]);
        for (m, n) in BLEND_LONG_NAMES {
            assert_eq!(blend_from_id(n.as_bytes()), Some(m));
        }
    }

    #[test]
    fn garbage_is_none() {
        assert!(parse_lfx2(&[0, 0]).is_none());
        assert!(parse_lrfx(&[0, 0, 0, 1, b'X']).is_none());
    }

    #[test]
    fn lrfx_drop_shadow() {
        // version 0, count 1, 'dsdw' v2 record.
        let mut d = vec![0, 0, 0, 1];
        d.extend_from_slice(b"8BIMdsdw");
        let mut rec = Vec::new();
        rec.extend_from_slice(&2u32.to_be_bytes()); // version
        rec.extend_from_slice(&(7u32 << 16).to_be_bytes()); // blur (16.16)
        rec.extend_from_slice(&0u32.to_be_bytes()); // intensity
        rec.extend_from_slice(&(90u32 << 16).to_be_bytes()); // angle
        rec.extend_from_slice(&(4u32 << 16).to_be_bytes()); // distance
        rec.extend_from_slice(&[0, 0, 0xff, 0xff, 0, 0, 0, 0, 0, 0]); // color: red
        rec.extend_from_slice(b"8BIMmul ");
        rec.extend_from_slice(&[1, 1, 128]);
        rec.extend_from_slice(&[0; 10]);
        d.extend_from_slice(&(rec.len() as u32).to_be_bytes());
        d.extend(rec);
        let (_, fx) = parse_lrfx(&d).unwrap();
        match &fx[0] {
            Effect::DropShadow(s) => {
                assert_eq!(s.size, 7.0);
                assert_eq!(s.distance, 4.0);
                assert_eq!(s.common.blend, BlendMode::Multiply);
                assert!((s.common.opacity - 128.0 / 255.0).abs() < 1e-6);
                assert_eq!(s.color.c[0], 1.0);
            }
            other => panic!("{other:?}"),
        }
    }
}
