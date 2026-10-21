//! Full parameter sets for the per-pixel adjustment kinds, shared by the destructive
//! `image.adjustments.*` commands, `layer.newAdjustmentLayer.*` and `layer.setAdjustment`.
//!
//! [`from_params`] reads params over a base adjustment (missing keys keep the base's value, so a
//! partial update only changes what it names) and [`to_params`] writes the complete set back, so
//! `from_params(kind, &to_params(a), None, mode) == a`. Values out of range are clamped like
//! Photoshop's fields; a key of the wrong type is an error.
//!
//! Units: Levels and Curves in 0–255 levels; percentages for the rest (as in Photoshop's dialogs);
//! colours as `"#rrggbb"` or `[r, g, b]` in 0–1.

use photocraft_color::ColorMode;
use photocraft_doc::Adjustment;
use photocraft_doc::adjust::{CURVE_MAP_LEN, CurvePoint, HueRange, LevelsChannel, ToneSpace, curve_map, curve_map_outputs};
use serde_json::{Map, Value, json};

use crate::{EngineError, Result};

/// Hue/Saturation and Black & White colour-range keys (storage order).
pub const HUE_RANGES: [&str; 6] = ["reds", "yellows", "greens", "cyans", "blues", "magentas"];

/// Black & White default weights (percent) for [`HUE_RANGES`].
pub const BW_DEFAULTS: [f32; 6] = [40.0, 60.0, 40.0, 60.0, 20.0, 80.0];

/// Photo Filter presets `(id, label, colour)`: common photographic filter colours.
pub const PHOTO_FILTERS: &[(&str, &str, [u8; 3])] = &[
    ("warming85", "Warming Filter (85)", [236, 138, 0]),
    ("warmingLBA", "Warming Filter (LBA)", [250, 150, 0]),
    ("warming81", "Warming Filter (81)", [235, 177, 19]),
    ("cooling80", "Cooling Filter (80)", [0, 109, 255]),
    ("coolingLBB", "Cooling Filter (LBB)", [0, 93, 255]),
    ("cooling82", "Cooling Filter (82)", [0, 181, 255]),
    ("red", "Red", [234, 26, 26]),
    ("orange", "Orange", [243, 132, 23]),
    ("yellow", "Yellow", [249, 227, 28]),
    ("green", "Green", [25, 201, 25]),
    ("cyan", "Cyan", [29, 203, 234]),
    ("blue", "Blue", [29, 53, 234]),
    ("violet", "Violet", [155, 29, 234]),
    ("magenta", "Magenta", [227, 24, 227]),
    ("sepia", "Sepia", [172, 122, 51]),
    ("deepRed", "Deep Red", [255, 0, 0]),
    ("deepBlue", "Deep Blue", [0, 34, 205]),
    ("deepEmerald", "Deep Emerald", [0, 140, 0]),
    ("deepYellow", "Deep Yellow", [255, 213, 0]),
    ("underwater", "Underwater", [0, 194, 177]),
];

/// Per-channel keys of Levels/Curves in each tone space (after the composite).
pub fn channel_keys(space: ToneSpace) -> &'static [&'static str] {
    match space {
        ToneSpace::Rgb => &["red", "green", "blue"],
        ToneSpace::Cmyk => &["cyan", "magenta", "yellow", "black"],
        ToneSpace::Lab => &["lightness", "a", "b"],
    }
}

/// The tone space new Levels/Curves get in a document of `mode`.
pub fn default_space(mode: ColorMode) -> ToneSpace {
    match mode {
        ColorMode::Cmyk => ToneSpace::Cmyk,
        ColorMode::Lab => ToneSpace::Lab,
        _ => ToneSpace::Rgb,
    }
}

/// The neutral adjustment of `kind` (what "Reset to defaults" restores).
pub fn default_for(kind: &str, mode: ColorMode) -> Result<Adjustment> {
    from_params(kind, &json!({}), None, mode)
}

/// Reads the adjustment of `kind` from `p` over `base` (or the kind's defaults when `base` is
/// None or of another kind). `mode` picks the Levels/Curves channel space when neither the params
/// nor the base name one.
pub fn from_params(kind: &str, p: &Value, base: Option<&Adjustment>, mode: ColorMode) -> Result<Adjustment> {
    let cmd = format!("{kind} adjustment");
    if !p.is_object() && !p.is_null() {
        return Err(bad(&cmd, "params must be an object"));
    }
    let r = Reader { cmd: &cmd, p };
    Ok(match kind {
        "invert" => Adjustment::Invert,
        "brightnessContrast" => {
            let (b0, c0, l0) = match base {
                Some(Adjustment::BrightnessContrast { brightness, contrast, legacy }) => (*brightness, *contrast, *legacy),
                _ => (0.0, 0.0, false),
            };
            let legacy = r.boolean("legacy", l0)?;
            let (lo, hi) = if legacy { (-100.0, 100.0) } else { (-50.0, 100.0) };
            Adjustment::BrightnessContrast { brightness: r.num("brightness", b0, -150.0, 150.0)?, contrast: r.num("contrast", c0, lo, hi)?, legacy }
        }
        "threshold" => {
            let l0 = match base {
                Some(Adjustment::Threshold { level }) => level * 255.0,
                _ => 128.0,
            };
            Adjustment::Threshold { level: r.num("level", l0, 1.0, 255.0)?.round() / 255.0 }
        }
        "posterize" => {
            let l0 = match base {
                Some(Adjustment::Posterize { levels }) => *levels as f32,
                _ => 4.0,
            };
            Adjustment::Posterize { levels: r.num("levels", l0, 2.0, 255.0)?.round() as u32 }
        }
        "exposure" => {
            let (e, o, g) = match base {
                Some(Adjustment::Exposure { exposure, offset, gamma }) => (*exposure, *offset, *gamma),
                _ => (0.0, 0.0, 1.0),
            };
            Adjustment::Exposure { exposure: r.num("exposure", e, -20.0, 20.0)?, offset: r.num("offset", o, -0.5, 0.5)?, gamma: r.num("gamma", g, 0.01, 9.99)? }
        }
        "vibrance" => {
            let (v, s) = match base {
                Some(Adjustment::Vibrance { vibrance, saturation }) => (*vibrance, *saturation),
                _ => (0.0, 0.0),
            };
            Adjustment::Vibrance { vibrance: r.num("vibrance", v, -100.0, 100.0)?, saturation: r.num("saturation", s, -100.0, 100.0)? }
        }
        "hueSaturation" => hue_saturation(&r, base)?,
        "levels" => levels(&r, base, mode)?,
        "curves" => curves(&r, base, mode)?,
        "colorBalance" => {
            let (mut tones, preserve) = match base {
                Some(Adjustment::ColorBalance { shadows, midtones, highlights, preserve_luminosity }) => {
                    ([*shadows, *midtones, *highlights], *preserve_luminosity)
                }
                _ => ([[0.0; 3]; 3], true),
            };
            for (i, key) in ["shadows", "midtones", "highlights"].iter().enumerate() {
                if let Some(v) = r.numbers(key, 3)? {
                    tones[i] = [v[0], v[1], v[2]].map(|x| x.clamp(-100.0, 100.0));
                }
            }
            Adjustment::ColorBalance {
                shadows: tones[0],
                midtones: tones[1],
                highlights: tones[2],
                preserve_luminosity: r.boolean("preserveLuminosity", preserve)?,
            }
        }
        "blackWhite" => {
            let (mut weights, tint0) = match base {
                Some(Adjustment::BlackWhite { weights, tint }) => (*weights, *tint),
                _ => (BW_DEFAULTS, None),
            };
            for (i, key) in HUE_RANGES.iter().enumerate() {
                weights[i] = r.num(key, weights[i], -200.0, 300.0)?;
            }
            // Photoshop's default tint colour is a light sepia.
            let color = r.color("tintColor", tint0.unwrap_or([0.882, 0.827, 0.702]))?;
            let tint = r.boolean("tint", tint0.is_some())?.then_some(color);
            Adjustment::BlackWhite { weights, tint }
        }
        "photoFilter" => {
            let (c0, d0, p0) = match base {
                Some(Adjustment::PhotoFilter { color, density, preserve_luminosity }) => (*color, *density, *preserve_luminosity),
                _ => ([236.0 / 255.0, 138.0 / 255.0, 0.0], 0.25, true),
            };
            let preset = match p.get("filter") {
                None | Some(Value::Null) => None,
                Some(Value::String(id)) => {
                    let f = PHOTO_FILTERS.iter().find(|f| f.0 == id).ok_or_else(|| bad(&cmd, format!("unknown filter `{id}`")))?;
                    Some(f.2.map(|v| f32::from(v) / 255.0))
                }
                Some(_) => return Err(bad(&cmd, "`filter` must be a preset id")),
            };
            let color = r.color("color", preset.unwrap_or(c0))?;
            Adjustment::PhotoFilter {
                color,
                density: r.num("density", d0 * 100.0, 0.0, 100.0)? / 100.0,
                preserve_luminosity: r.boolean("preserveLuminosity", p0)?,
            }
        }
        "channelMixer" => {
            let (mut matrix, mono0) = match base {
                Some(Adjustment::ChannelMixer { matrix, monochrome }) => (*matrix, *monochrome),
                _ => ([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], false),
            };
            let monochrome = r.boolean("monochrome", mono0)?;
            if monochrome && !mono0 && p.get("gray").is_none() {
                // Turning Monochrome on starts from Photoshop's grey mix.
                matrix[0] = [0.4, 0.4, 0.2, 0.0];
            }
            for (i, key) in ["red", "green", "blue"].iter().enumerate() {
                if let Some(v) = r.numbers(key, 4)? {
                    matrix[i] = mixer_row(&v);
                }
            }
            if let Some(v) = r.numbers("gray", 4)? {
                matrix[0] = mixer_row(&v);
            }
            Adjustment::ChannelMixer { matrix, monochrome }
        }
        "gradientMap" => {
            let (stops0, rev0, dither0) = match base {
                Some(Adjustment::GradientMap { stops, reverse, dither }) => (stops.clone(), *reverse, *dither),
                _ => (vec![(0.0, [0.0; 3]), (1.0, [1.0; 3])], false, false),
            };
            let stops = match p.get("stops") {
                None | Some(Value::Null) => stops0,
                Some(v) => gradient_stops(&cmd, v)?,
            };
            Adjustment::GradientMap { stops, reverse: r.boolean("reverse", rev0)?, dither: r.boolean("dither", dither0)? }
        }
        "selectiveColor" => crate::adjust_cmds::selective_from_params(p, base.filter(|b| matches!(b, Adjustment::SelectiveColor { .. })))?,
        "colorLookup" => crate::adjust_cmds::lookup_from_params(p, base.filter(|b| matches!(b, Adjustment::ColorLookup { .. })))?,
        other => return Err(bad("adjustment", format!("unknown adjustment kind `{other}`"))),
    })
}

/// The complete parameter set of `adj` (the inverse of [`from_params`]).
pub fn to_params(adj: &Adjustment) -> Value {
    match adj {
        Adjustment::Invert | Adjustment::Unsupported { .. } => json!({}),
        Adjustment::BrightnessContrast { brightness, contrast, legacy } => json!({"brightness": brightness, "contrast": contrast, "legacy": legacy}),
        Adjustment::Threshold { level } => json!({"level": (level * 255.0).round()}),
        Adjustment::Posterize { levels } => json!({"levels": levels}),
        Adjustment::Exposure { exposure, offset, gamma } => json!({"exposure": exposure, "offset": offset, "gamma": gamma}),
        Adjustment::Vibrance { vibrance, saturation } => json!({"vibrance": vibrance, "saturation": saturation}),
        Adjustment::HueSaturation { hue, saturation, lightness, colorize, ranges } => {
            let mut v = json!({"hue": hue, "saturation": saturation, "lightness": lightness, "colorize": colorize});
            for (key, r) in HUE_RANGES.iter().zip(ranges.iter()) {
                v[*key] = json!({"hue": r.hue, "saturation": r.saturation, "lightness": r.lightness, "range": r.bounds});
            }
            v
        }
        Adjustment::Levels { master, per_channel, space, black } => {
            let mut v = levels_obj(master);
            for (key, c) in channel_keys(*space).iter().zip([&per_channel[0], &per_channel[1], &per_channel[2], black]) {
                v[*key] = levels_obj(c);
            }
            if *space == ToneSpace::Lab {
                // No composite in Lab: the top-level keys are the lightness channel.
                v = merge(levels_obj(&per_channel[0]), v);
            }
            v
        }
        Adjustment::Curves { master, per_channel, space, black, pencil } => {
            let arr = |c: &[CurvePoint]| if *pencil { map_arr(c) } else { curve_arr(c) };
            let mut v = json!({"points": arr(master)});
            let ink = if black.len() >= 2 { black.clone() } else { identity_curve() };
            let chans = per_channel.iter().cloned().chain(std::iter::once(ink));
            for (key, c) in channel_keys(*space).iter().zip(chans) {
                v[*key] = arr(&c);
            }
            if *space == ToneSpace::Lab {
                v["points"] = arr(&per_channel[0]);
            }
            if *pencil {
                v["pencil"] = json!(true);
            }
            v
        }
        Adjustment::ColorBalance { shadows, midtones, highlights, preserve_luminosity } => {
            json!({"shadows": shadows, "midtones": midtones, "highlights": highlights, "preserveLuminosity": preserve_luminosity})
        }
        Adjustment::BlackWhite { weights, tint } => {
            let mut v = json!({"tint": tint.is_some(), "tintColor": hex(tint.unwrap_or([0.882, 0.827, 0.702]))});
            for (key, w) in HUE_RANGES.iter().zip(weights.iter()) {
                v[*key] = json!(w);
            }
            v
        }
        Adjustment::PhotoFilter { color, density, preserve_luminosity } => {
            json!({"color": hex(*color), "density": (density * 1000.0).round() / 10.0, "preserveLuminosity": preserve_luminosity})
        }
        Adjustment::ChannelMixer { matrix, monochrome } => {
            let row = |r: &[f32; 4]| json!(r.map(|x| (x * 1000.0).round() / 10.0));
            let mut v = json!({"red": row(&matrix[0]), "green": row(&matrix[1]), "blue": row(&matrix[2]), "monochrome": monochrome});
            if *monochrome {
                v["gray"] = row(&matrix[0]);
            }
            v
        }
        Adjustment::GradientMap { stops, reverse, dither } => {
            json!({"stops": stops.iter().map(|(t, c)| json!([t, hex(*c)])).collect::<Vec<_>>(), "reverse": reverse, "dither": dither})
        }
        Adjustment::SelectiveColor { relative, adjustments } => {
            let mut v = json!({"method": if *relative { "relative" } else { "absolute" }});
            for (i, key) in crate::adjust_cmds::RANGES.iter().enumerate() {
                v[*key] = json!(adjustments[i]);
            }
            v
        }
        Adjustment::ColorLookup { tetrahedral, dither, .. } => {
            json!({"interpolation": if *tetrahedral { "tetrahedral" } else { "trilinear" }, "dither": dither})
        }
    }
}

// ---------------------------------------------------------------------------------------------

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

/// Typed, clamped access to optional params.
struct Reader<'a> {
    cmd: &'a str,
    p: &'a Value,
}

impl Reader<'_> {
    fn get(&self, key: &str) -> Option<&Value> {
        self.p.get(key).filter(|v| !v.is_null())
    }

    fn num(&self, key: &str, default: f32, lo: f32, hi: f32) -> Result<f32> {
        match self.get(key) {
            None => Ok(default.clamp(lo, hi)),
            Some(v) => num_in(self.cmd, key, v, lo, hi),
        }
    }

    fn boolean(&self, key: &str, default: bool) -> Result<bool> {
        match self.get(key) {
            None => Ok(default),
            Some(Value::Bool(b)) => Ok(*b),
            Some(_) => Err(bad(self.cmd, format!("`{key}` must be true or false"))),
        }
    }

    /// An array of exactly `n` numbers.
    fn numbers(&self, key: &str, n: usize) -> Result<Option<Vec<f32>>> {
        let Some(v) = self.get(key) else { return Ok(None) };
        let a = v.as_array().filter(|a| a.len() == n).ok_or_else(|| bad(self.cmd, format!("`{key}` must be an array of {n} numbers")))?;
        a.iter().map(|x| num_in(self.cmd, key, x, -1e6, 1e6)).collect::<Result<Vec<_>>>().map(Some)
    }

    fn color(&self, key: &str, default: [f32; 3]) -> Result<[f32; 3]> {
        match self.get(key) {
            None => Ok(default),
            Some(v) => color_of(self.cmd, key, v),
        }
    }

    /// A channel object (`{"inBlack": …}`) or None.
    fn object(&self, key: &str) -> Result<Option<Reader<'_>>> {
        match self.get(key) {
            None => Ok(None),
            Some(v) if v.is_object() => Ok(Some(Reader { cmd: self.cmd, p: v })),
            Some(_) => Err(bad(self.cmd, format!("`{key}` must be an object"))),
        }
    }
}

fn num_in(cmd: &str, key: &str, v: &Value, lo: f32, hi: f32) -> Result<f32> {
    let x = v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| bad(cmd, format!("`{key}` must be a number")))?;
    Ok((x as f32).clamp(lo, hi))
}

fn color_of(cmd: &str, key: &str, v: &Value) -> Result<[f32; 3]> {
    let err = || bad(cmd, format!("`{key}` must be \"#rrggbb\" or [r, g, b] in 0..1"));
    match v {
        Value::String(s) => {
            let h = s.trim_start_matches('#');
            let b = |i: usize| h.get(i..i + 2).and_then(|x| u8::from_str_radix(x, 16).ok()).map(|x| f32::from(x) / 255.0);
            match (h.len(), b(0), b(2), b(4)) {
                (6 | 8, Some(r), Some(g), Some(bb)) => Ok([r, g, bb]),
                _ => Err(err()),
            }
        }
        Value::Array(a) if a.len() >= 3 => {
            let c = |i: usize| a.get(i).and_then(Value::as_f64).filter(|x| x.is_finite()).map(|x| (x as f32).clamp(0.0, 1.0)).ok_or_else(err);
            Ok([c(0)?, c(1)?, c(2)?])
        }
        _ => Err(err()),
    }
}

pub fn hex(c: [f32; 3]) -> String {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", b(c[0]), b(c[1]), b(c[2]))
}

fn merge(mut a: Value, b: Value) -> Value {
    if let (Some(ao), Value::Object(bo)) = (a.as_object_mut(), b) {
        for (k, v) in bo {
            ao.entry(k).or_insert(v);
        }
    }
    a
}

fn mixer_row(v: &[f32]) -> [f32; 4] {
    std::array::from_fn(|i| v.get(i).copied().unwrap_or(0.0).clamp(-200.0, 200.0) / 100.0)
}

fn hue_saturation(r: &Reader<'_>, base: Option<&Adjustment>) -> Result<Adjustment> {
    let (h, s, l, c, mut ranges) = match base {
        Some(Adjustment::HueSaturation { hue, saturation, lightness, colorize, ranges }) => (*hue, *saturation, *lightness, *colorize, ranges.clone()),
        _ => (0.0, 0.0, 0.0, false, HueRange::defaults()),
    };
    let colorize = r.boolean("colorize", c)?;
    // Colorize hue is 0..360, a shift is -180..180.
    let hue = if colorize { r.num("hue", h.rem_euclid(360.0), 0.0, 360.0)? } else { r.num("hue", h, -180.0, 180.0)? };
    // Turning Colorize on starts at 25% saturation unless one is given, like the dialog's
    // checkbox and Photoshop; an explicit 0 stays 0, which colorizes to grey (#2326).
    let s = if colorize && !c { s.max(25.0) } else { s };
    let (saturation, lightness) = (r.num("saturation", s, if colorize { 0.0 } else { -100.0 }, 100.0)?, r.num("lightness", l, -100.0, 100.0)?);
    for (i, key) in HUE_RANGES.iter().enumerate() {
        let Some(o) = r.object(key)? else { continue };
        let Some(range) = ranges.get_mut(i) else { continue };
        range.hue = o.num("hue", range.hue, -180.0, 180.0)?;
        range.saturation = o.num("saturation", range.saturation, -100.0, 100.0)?;
        range.lightness = o.num("lightness", range.lightness, -100.0, 100.0)?;
        if let Some(b) = o.numbers("range", 4)? {
            // Keep the four sliders ordered and within one turn of the first.
            range.bounds = HueRange::canonical_bounds([b[0], b[1], b[2], b[3]]);
        }
    }
    Ok(Adjustment::HueSaturation { hue, saturation, lightness, colorize, ranges })
}

/// Which tone space the params address: any channel key of a space picks it.
fn space_of(r: &Reader<'_>, base: Option<ToneSpace>, mode: ColorMode) -> ToneSpace {
    let has = |keys: &[&str]| keys.iter().any(|k| r.get(k).is_some());
    if has(&["cyan", "magenta", "yellow", "black"]) {
        ToneSpace::Cmyk
    } else if has(&["a", "b"]) || (has(&["lightness"]) && !has(&["red", "green", "blue"])) {
        ToneSpace::Lab
    } else if has(&["red", "green", "blue"]) {
        ToneSpace::Rgb
    } else {
        base.unwrap_or_else(|| default_space(mode))
    }
}

fn levels(r: &Reader<'_>, base: Option<&Adjustment>, mode: ColorMode) -> Result<Adjustment> {
    let (mut master, mut chans, base_space) = match base {
        Some(Adjustment::Levels { master, per_channel, space, black }) => {
            (master.clone(), [per_channel[0].clone(), per_channel[1].clone(), per_channel[2].clone(), black.clone()], Some(*space))
        }
        _ => (LevelsChannel::default(), Default::default(), None),
    };
    let space = space_of(r, base_space, mode);
    if base_space.is_some_and(|b| b != space) {
        chans = Default::default();
    }
    let read = |o: &Reader<'_>, c: &LevelsChannel| -> Result<LevelsChannel> {
        let in_black = o.num("inBlack", c.in_black * 255.0, 0.0, 253.0)?;
        let in_white = o.num("inWhite", c.in_white * 255.0, 2.0, 255.0)?.max(in_black + 2.0);
        Ok(LevelsChannel {
            in_black: in_black / 255.0,
            in_white: in_white / 255.0,
            gamma: o.num("gamma", c.gamma, 0.01, 9.99)?,
            out_black: o.num("outBlack", c.out_black * 255.0, 0.0, 255.0)? / 255.0,
            out_white: o.num("outWhite", c.out_white * 255.0, 0.0, 255.0)? / 255.0,
        })
    };
    if space == ToneSpace::Lab {
        // Top-level keys (and "gray") address lightness: Lab has no composite.
        chans[0] = read(r, &chans[0])?;
    } else {
        master = read(r, &master)?;
        if let Some(o) = r.object("gray")? {
            master = read(&o, &master)?;
        }
    }
    for (i, key) in channel_keys(space).iter().enumerate() {
        if let (Some(o), Some(c)) = (r.object(key)?, chans.get(i).cloned()) {
            chans[i] = read(&o, &c)?;
        }
    }
    let [c0, c1, c2, black] = chans;
    Ok(Adjustment::Levels { master, per_channel: [c0, c1, c2], space, black: if space == ToneSpace::Cmyk { black } else { LevelsChannel::default() } })
}

fn identity_curve() -> Vec<CurvePoint> {
    vec![CurvePoint { input: 0.0, output: 0.0 }, CurvePoint { input: 1.0, output: 1.0 }]
}

const CURVE_COORDINATE_PRECISION: f32 = 100.0;
const CURVE_POINT_DUPLICATE_DISTANCE: f32 = 0.5 / 255.0;

fn curve_arr(c: &[CurvePoint]) -> Value {
    let c = if c.len() >= 2 { c.to_vec() } else { identity_curve() };
    json!(
        c.iter()
            .map(|p| {
                [
                    (p.input * 255.0 * CURVE_COORDINATE_PRECISION).round() / CURVE_COORDINATE_PRECISION,
                    (p.output * 255.0 * CURVE_COORDINATE_PRECISION).round() / CURVE_COORDINATE_PRECISION,
                ]
            })
            .collect::<Vec<_>>()
    )
}

/// Most points a curve may have (Photoshop allows 16; the PSD record holds 19).
pub const MAX_CURVE_POINTS: usize = 19;

/// Points Photoshop's "Edit points to modify the curve" places on a pencil-drawn curve at most.
const MAX_FITTED_POINTS: usize = 16;

/// Most Smooth passes one `smooth` param applies.
pub const MAX_SMOOTH_PASSES: f32 = 10.0;

/// A pencil-mode curve as params: its [`CURVE_MAP_LEN`] outputs in 0..255.
fn map_arr(c: &[CurvePoint]) -> Value {
    json!(curve_map_outputs(c).iter().map(|o| (o * 255.0 * CURVE_COORDINATE_PRECISION).round() / CURVE_COORDINATE_PRECISION).collect::<Vec<_>>())
}

/// `lut` (evenly spaced over 0..=1) at `x`, linearly interpolated.
fn lut_at(lut: &[f32], x: f32) -> f32 {
    let Some(last) = lut.len().checked_sub(1) else { return x };
    let f = x.clamp(0.0, 1.0) * last as f32;
    let i = (f.floor() as usize).min(last);
    let (a, b) = (lut.get(i).copied().unwrap_or(x), lut.get((i + 1).min(last)).copied().unwrap_or(x));
    a + (b - a) * (f - i as f32)
}

/// A point curve drawn into a pencil-mode map (Photoshop switching to "Draw to modify the curve").
pub fn points_to_map(points: &[CurvePoint]) -> Vec<CurvePoint> {
    let lut = photocraft_compose::adjust::curve_lut(points);
    let last = (CURVE_MAP_LEN - 1) as f32;
    curve_map(&(0..CURVE_MAP_LEN).map(|k| lut_at(&lut, k as f32 / last)).collect::<Vec<_>>())
}

/// A pencil-mode map turned back into a point curve (Photoshop's "Edit points to modify the
/// curve"): the endpoints, then the worst-fitting input until the spline is within one level of
/// the map or [`MAX_FITTED_POINTS`] are placed.
pub fn map_to_points(map: &[CurvePoint]) -> Vec<CurvePoint> {
    let target = curve_map_outputs(map);
    let last = CURVE_MAP_LEN - 1;
    let at = |k: usize| CurvePoint { input: k as f32 / last as f32, output: target.get(k).copied().unwrap_or(0.0) };
    let mut pts = vec![at(0), at(last)];
    while pts.len() < MAX_FITTED_POINTS {
        let lut = photocraft_compose::adjust::curve_lut(&pts);
        let worst = (0..CURVE_MAP_LEN)
            .map(|k| (k, (lut_at(&lut, k as f32 / last as f32) - target.get(k).copied().unwrap_or(0.0)).abs()))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        let Some((k, _)) = worst.filter(|w| w.1 > 1.0 / 255.0) else { break };
        let p = at(k);
        if pts.iter().any(|q| (q.input - p.input).abs() < CURVE_POINT_DUPLICATE_DISTANCE) {
            break;
        }
        pts.push(p);
        pts.sort_by(|a, b| a.input.total_cmp(&b.input));
    }
    pts
}

/// One pass of Curves' Smooth button over pencil-mode outputs (any scale): each value becomes
/// the mean of a symmetric window of up to three neighbours a side, so straight runs and the two
/// endpoints stay put while jagged strokes even out.
pub fn smooth_curve_map(outputs: &mut [f32]) {
    const RADIUS: usize = 3;
    let src = outputs.to_vec();
    let n = src.len();
    for (k, out) in outputs.iter_mut().enumerate() {
        let r = RADIUS.min(k).min(n.saturating_sub(k + 1));
        if r == 0 {
            continue;
        }
        let window = src.get(k - r..=k + r).unwrap_or_default();
        *out = window.iter().sum::<f32>() / window.len().max(1) as f32;
    }
}

/// Whether a pencil-mode curve maps every level to itself.
fn is_identity_map(c: &[CurvePoint]) -> bool {
    let last = (CURVE_MAP_LEN - 1) as f32;
    curve_map_outputs(c).iter().enumerate().all(|(k, o)| (o - k as f32 / last).abs() < 0.5 / 255.0)
}

/// Fixed-target Curves eyedroppers. The UI keeps the sampled operation provisional; this helper
/// only resolves a sample into an ordinary Curves adjustment that the existing command can apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurvesEyedropper {
    Black,
    NeutralGray,
    White,
}

/// Verify that serializing and parsing `points` keeps every point and the requested anchor.
fn curve_round_trip_keeps_anchor(points: &[CurvePoint], input: f32, output: f32) -> bool {
    let Ok(parsed) = parse_curve("curves eyedropper", "curve", &curve_arr(points)) else { return false };
    if parsed.len() != points.len() {
        return false;
    }
    let serialized = |value: f32| ((value * 255.0 * CURVE_COORDINATE_PRECISION).round() / CURVE_COORDINATE_PRECISION) / 255.0;
    let (input, output) = (serialized(input), serialized(output));
    parsed.iter().any(|point| (point.input - input).abs() < 1e-6 && (point.output - output).abs() < 1e-6)
}

fn move_curve_endpoint(points: &mut [CurvePoint], sample: f32, black: bool) {
    let sample = sample.clamp(0.0, 1.0);
    let mut candidate = points.to_vec();
    if black {
        for p in &mut candidate {
            p.input = sample + p.input * (1.0 - sample);
        }
        if let Some(first) = candidate.first_mut() {
            first.input = sample;
            first.output = 0.0;
        }
    } else {
        for p in &mut candidate {
            p.input *= sample;
        }
        if let Some(last) = candidate.last_mut() {
            last.input = sample;
            last.output = 1.0;
        }
    }

    // `curve_arr` stores hundredths of a 0..255 level, while `parse_curve` drops points
    // separated by less than half a level. Scaling an endpoint toward the sample can collapse
    // several edited points together even when the target endpoint itself survives. Check that
    // the full curve survives the actual serializer and parser.
    let target = if black { 0.0 } else { 1.0 };
    if curve_round_trip_keeps_anchor(&candidate, sample, target) {
        points.clone_from_slice(&candidate);
        return;
    }

    // If scaling collapses any points, keep the user's point positions and edits, and make the
    // two points nearest the selected endpoint a target plateau. Near an endpoint sample, those
    // two points bracket the sample; the plateau is representable and leaves the rest of the
    // curve intact.
    if black {
        for point in points.iter_mut().rev().take(2) {
            point.output = 0.0;
        }
    } else {
        for point in points.iter_mut().take(2) {
            point.output = 1.0;
        }
    }
}

fn anchor_curve(points: &mut Vec<CurvePoint>, input: f32, output: f32) {
    let (input, output) = (input.clamp(0.0, 1.0), output.clamp(0.0, 1.0));
    let mut candidate = points.clone();
    if let Some(point) = candidate.iter_mut().find(|p| (p.input - input).abs() < CURVE_POINT_DUPLICATE_DISTANCE) {
        point.input = input;
        point.output = output;
    } else if candidate.len() < MAX_CURVE_POINTS {
        candidate.push(CurvePoint { input, output });
    } else if let Some((index, _)) = candidate
        .iter()
        .enumerate()
        .filter(|(index, _)| *index > 0 && *index + 1 < candidate.len())
        .min_by(|(_, a), (_, b)| (a.input - input).abs().total_cmp(&(b.input - input).abs()))
        && let Some(point) = candidate.get_mut(index)
    {
        *point = CurvePoint { input, output };
    }
    candidate.sort_by(|a, b| a.input.total_cmp(&b.input));
    if curve_round_trip_keeps_anchor(&candidate, input, output) {
        *points = candidate;
        return;
    }

    // A channel sample can land between two points only half a level apart. Moving one point to
    // the sample would then make the parser collapse the pair, so retain their positions and make
    // the bracketing pair a deterministic neutral plateau instead.
    let Some(last) = points.len().checked_sub(1) else { return };
    let split = points.partition_point(|point| point.input < input);
    let mut lower = split.saturating_sub(1).min(last);
    let mut upper = split.min(last);
    if lower == upper {
        if upper < last {
            upper += 1;
        } else {
            lower = lower.saturating_sub(1);
        }
    }
    if let Some(point) = points.get_mut(lower) {
        point.output = output;
    }
    if let Some(point) = points.get_mut(upper) {
        point.output = output;
    }
}

/// Rec. 709 luma in the sampled display-RGB values. This is the neutral level the gray picker
/// should preserve when it removes a colour cast.
fn rgb_luma(rgb: [f32; 3]) -> f32 {
    rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
}

/// Find the master-curve input whose output is closest to `target`. The master curve is kept
/// intact by the gray picker, so channel curves must target its inverse to preserve final luma.
/// A non-monotone or flat curve may not have an exact inverse; the first nearest LUT entry gives
/// a deterministic best fit.
fn inverse_master_target(master: &[CurvePoint], target: f32) -> f32 {
    if photocraft_doc::adjust::is_identity_curve(master) {
        return target;
    }
    let lut = photocraft_compose::adjust::curve_lut(master);
    let Some(denominator) = lut.len().checked_sub(1).filter(|last| *last > 0) else { return target };
    let mut best_input = target;
    let mut best_error = f32::INFINITY;
    for (index, value) in lut.iter().enumerate() {
        let error = (*value - target).abs();
        if error < best_error {
            best_error = error;
            best_input = index as f32 / denominator as f32;
        }
    }
    best_input
}

/// The sample and colour modes both eyedroppers (Levels, Curves) support: RGB, and Grayscale
/// without the Neutral Gray picker.
fn check_eyedropper(cmd: &str, picker: CurvesEyedropper, sample: [f32; 3], mode: ColorMode) -> Result<()> {
    if sample.iter().any(|v| !v.is_finite()) {
        return Err(bad(cmd, "sample components must be finite"));
    }
    if !matches!(mode, ColorMode::Rgb | ColorMode::Grayscale) {
        return Err(bad(cmd, "only RGB and Grayscale documents are supported"));
    }
    if mode == ColorMode::Grayscale && picker == CurvesEyedropper::NeutralGray {
        return Err(bad(cmd, "Neutral Gray is unavailable for Grayscale documents"));
    }
    Ok(())
}

/// The smallest input range Levels allows (two levels), as the parser enforces.
const LEVELS_MIN_RANGE: f32 = 2.0 / 255.0;

/// Moves the input black (or white) of `c` to `sample`, so the sample maps to the output black
/// (white). The other input end moves away if it would leave less than two levels between them.
fn set_levels_input(c: &mut LevelsChannel, sample: f32, black: bool) {
    if black {
        c.in_black = sample.clamp(0.0, 1.0 - LEVELS_MIN_RANGE);
        c.in_white = c.in_white.max(c.in_black + LEVELS_MIN_RANGE).min(1.0);
    } else {
        c.in_white = sample.clamp(LEVELS_MIN_RANGE, 1.0);
        c.in_black = c.in_black.min(c.in_white - LEVELS_MIN_RANGE).max(0.0);
    }
}

/// Sets the midtone gamma of `c` so `sample` maps to `target` (both 0..1). Bisection over the
/// gamma range on the compositor's own Levels curve (monotone in gamma), so the shadow toe is
/// accounted for. A sample at or outside the input range, or a target outside the output range,
/// can't be reached by gamma and leaves `c` unchanged.
fn set_levels_gamma(c: &mut LevelsChannel, sample: f32, target: f32) {
    let u = (sample - c.in_black) / (c.in_white - c.in_black).max(1e-6);
    let span = c.out_white - c.out_black;
    let t = (target - c.out_black) / span;
    if !(u > 0.0 && u < 1.0) || span.abs() < 1e-6 || !(t > 0.0 && t < 1.0) {
        return;
    }
    let reached = |gamma: f32| {
        let trial = LevelsChannel { gamma, ..c.clone() };
        (photocraft_compose::adjust::levels(&trial, sample) - c.out_black) / span
    };
    let (mut lo, mut hi) = (0.01f32.ln(), 9.99f32.ln());
    for _ in 0..40 {
        let mid = (lo + hi) / 2.0;
        if reached(mid.exp()) < t {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    c.gamma = ((lo + hi) / 2.0).exp().clamp(0.01, 9.99);
}

/// Photoshop's Levels eyedroppers on the Levels in `base` (params, as for `from_params`), from a
/// colour sample of the image the Levels adjust. RGB: Set Black/White Point move each channel's
/// input black/white to the sample's value in that channel, so the sample maps to the output
/// black/white; Set Neutral Gray Point sets each channel's midtone gamma so the channel maps to
/// the sample's luma, removing the cast while keeping its brightness. The composite record is
/// kept. Grayscale: Black/White move the composite input; there is no gray picker.
pub fn levels_eyedropper(base: &Value, picker: CurvesEyedropper, sample: [f32; 3], mode: ColorMode) -> Result<Adjustment> {
    const CMD: &str = "levels eyedropper";
    check_eyedropper(CMD, picker, sample, mode)?;
    let mut adjustment = from_params("levels", base, None, mode)?;
    let Adjustment::Levels { master, per_channel, space, .. } = &mut adjustment else {
        return Err(bad(CMD, "expected a Levels adjustment"));
    };
    if *space != ToneSpace::Rgb {
        return Err(bad(CMD, "only RGB channel Levels are supported"));
    }
    let sample = sample.map(|v| v.clamp(0.0, 1.0));
    if mode == ColorMode::Grayscale {
        set_levels_input(master, (sample[0] + sample[1] + sample[2]) / 3.0, picker == CurvesEyedropper::Black);
        return Ok(adjustment);
    }
    match picker {
        CurvesEyedropper::Black | CurvesEyedropper::White => {
            for (c, v) in per_channel.iter_mut().zip(sample) {
                set_levels_input(c, v, picker == CurvesEyedropper::Black);
            }
        }
        CurvesEyedropper::NeutralGray => {
            let target = rgb_luma(sample);
            for (c, v) in per_channel.iter_mut().zip(sample) {
                set_levels_gamma(c, v, target);
            }
        }
    }
    Ok(adjustment)
}

/// The eyedropper of adjustment `kind` ("levels" or "curves"): [`levels_eyedropper`] or
/// [`curves_eyedropper`].
pub fn adjustment_eyedropper(kind: &str, base: &Value, picker: CurvesEyedropper, sample: [f32; 3], mode: ColorMode) -> Result<Adjustment> {
    match kind {
        "levels" => levels_eyedropper(base, picker, sample, mode),
        "curves" => curves_eyedropper(base, picker, sample, mode),
        _ => Err(bad(EYEDROPPER, "`eyedropper` applies to Levels and Curves only")),
    }
}

/// Resolve a merged-composite colour sample into the black, luma-preserving neutral-gray or white target.
/// RGB uses the three channel curves. Grayscale uses the composite curve and has no gray picker.
/// Other colour models stay unsupported until their native channel semantics can be sampled.
pub fn curves_eyedropper(base: &Value, picker: CurvesEyedropper, sample: [f32; 3], mode: ColorMode) -> Result<Adjustment> {
    check_eyedropper("curves eyedropper", picker, sample, mode)?;
    // The pickers place points: a pencil-drawn curve is first fitted with points, as when
    // Photoshop leaves the pencil for point editing.
    let mut base = base.clone();
    if let Some(o) = base.as_object_mut() {
        o.remove("smooth");
        if o.get("pencil").and_then(Value::as_bool) == Some(true) {
            o.insert("pencil".into(), json!(false));
        }
    }
    let mut adjustment = from_params("curves", &base, None, mode)?;
    let Adjustment::Curves { master, per_channel, .. } = &mut adjustment else {
        return Err(bad("curves eyedropper", "expected a Curves adjustment"));
    };
    let sample = sample.map(|v| v.clamp(0.0, 1.0));
    if mode == ColorMode::Grayscale {
        let tone = (sample[0] + sample[1] + sample[2]) / 3.0;
        move_curve_endpoint(master, tone, picker == CurvesEyedropper::Black);
        return Ok(adjustment);
    }
    match picker {
        CurvesEyedropper::Black | CurvesEyedropper::White => {
            let black = picker == CurvesEyedropper::Black;
            for (curve, value) in per_channel.iter_mut().zip(sample) {
                move_curve_endpoint(curve, value, black);
            }
            if let Some(endpoint) = if black { master.first_mut() } else { master.last_mut() } {
                endpoint.output = if black { 0.0 } else { 1.0 };
            }
        }
        CurvesEyedropper::NeutralGray => {
            // Use sampled luma instead of a fixed mid-gray so removing a cast keeps the sample's
            // brightness. The per-channel curves precede the composite curve, so invert the
            // existing master curve while preserving it.
            let target = inverse_master_target(master, rgb_luma(sample));
            for (curve, value) in per_channel.iter_mut().zip(sample) {
                anchor_curve(curve, value, target);
            }
        }
    }
    Ok(adjustment)
}

/// The `eyedropper` param of the Levels and Curves commands (`kind`), so agents and the CLI can
/// drive the dialog's Set Black/Neutral Gray/White Point pickers:
/// `{"point":"black"|"gray"|"white","at":[x,y]}` samples the merged composite at document pixel
/// (x, y) (`"size"`: the Eyedropper's Sample Size, 1 = Point Sample), or `"color":[r,g,b]` (0–1)
/// gives the sample directly. The rest of `p` is the adjustment the picker edits. `None` without
/// the key.
pub fn eyedropper_from_params(s: &crate::Session, kind: &str, p: &Value) -> Result<Option<Adjustment>> {
    let Some(e) = p.get("eyedropper") else { return Ok(None) };
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let (picker, sample) = eyedropper_sample(&d.doc, None, e)?;
    let mut base = p.clone();
    if let Some(o) = base.as_object_mut() {
        o.remove("eyedropper");
    }
    adjustment_eyedropper(kind, &base, picker, sample, d.doc.mode).map(Some)
}

/// The `eyedropper` param of `layer.setAdjustment` on Levels or Curves adjustment layer `layer`:
/// the Properties panel's Set Black/Neutral Gray/White Point pickers. `at` samples the image the
/// layer adjusts (the composite with the layer hidden, like its Properties histogram), so clicking
/// the same pixel again maps it the same way. The picker edits the layer's adjustment with the
/// other keys of `p` merged over it. `None` without the key.
pub fn layer_eyedropper(s: &crate::Session, layer: photocraft_doc::LayerId, p: &Value) -> Result<Option<Adjustment>> {
    let Some(e) = p.get("eyedropper") else { return Ok(None) };
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let l = d.doc.layer(layer).ok_or(EngineError::NoLayer(layer))?;
    let (kind, adj) = match &l.content {
        photocraft_doc::LayerContent::Adjustment(adj @ Adjustment::Curves { .. }) => ("curves", adj),
        photocraft_doc::LayerContent::Adjustment(adj @ Adjustment::Levels { .. }) => ("levels", adj),
        _ => return Err(bad(EYEDROPPER, "`eyedropper` applies to Levels and Curves adjustment layers only")),
    };
    let mut rest = p.clone();
    if let Some(o) = rest.as_object_mut() {
        o.remove("eyedropper");
    }
    let current = if user_keys(&rest).is_empty() { adj.clone() } else { from_params(kind, &rest, Some(adj), d.doc.mode)? };
    let (picker, sample) = eyedropper_sample(&d.doc, Some(layer), e)?;
    adjustment_eyedropper(kind, &to_params(&current), picker, sample, d.doc.mode).map(Some)
}

const EYEDROPPER: &str = "eyedropper";

/// The picker and sample of an `eyedropper` param; `at` samples `doc` with layer `hide` hidden.
fn eyedropper_sample(doc: &photocraft_doc::Document, hide: Option<photocraft_doc::LayerId>, e: &Value) -> Result<(CurvesEyedropper, [f32; 3])> {
    const CMD: &str = EYEDROPPER;
    let picker = match e.get("point").and_then(Value::as_str) {
        Some("black") => CurvesEyedropper::Black,
        Some("gray") => CurvesEyedropper::NeutralGray,
        Some("white") => CurvesEyedropper::White,
        _ => return Err(bad(CMD, "`eyedropper.point` must be \"black\", \"gray\" or \"white\"")),
    };
    let triple = |v: &Value| -> Option<[f64; 3]> {
        let a = v.as_array().filter(|a| a.len() == 3)?;
        Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?, a.get(2)?.as_f64()?])
    };
    let sample = if let Some(c) = e.get("color") {
        let c = triple(c).ok_or_else(|| bad(CMD, "`eyedropper.color` must be [r, g, b] in 0..1"))?;
        c.map(|v| v as f32)
    } else if let Some(at) = e.get("at") {
        let xy = at.as_array().filter(|a| a.len() == 2).and_then(|a| Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?]));
        let [x, y] = xy.filter(|xy| xy.iter().all(|v| v.is_finite())).ok_or_else(|| bad(CMD, "`eyedropper.at` must be [x, y] in document pixels"))?;
        let (w, h) = (f64::from(doc.size.width), f64::from(doc.size.height));
        if x < 0.0 || y < 0.0 || x >= w || y >= h {
            return Err(bad(CMD, "`eyedropper.at` is outside the image"));
        }
        let size = crate::sample_cmds::size_param(CMD, e)?;
        let all = crate::sample_cmds::SampleLayers::All;
        let [r, g, b, a] = match hide {
            Some(id) => {
                let mut below = doc.clone();
                if let Some(l) = below.layer_mut(id) {
                    l.visible = false;
                }
                crate::sample_cmds::sample_color(&below, None, x, y, size, all)
            }
            None => crate::sample_cmds::sample_color(doc, None, x, y, size, all),
        };
        if a <= 0.0 {
            return Err(bad(CMD, "the sampled pixel is transparent"));
        }
        [r, g, b]
    } else {
        return Err(bad(CMD, "`eyedropper` needs `at` [x, y] or `color` [r, g, b]"));
    };
    Ok((picker, sample))
}

/// `[[in, out], …]` in 0..255: sorted by input, duplicates collapsed, 2..=19 points.
pub fn parse_curve(cmd: &str, key: &str, v: &Value) -> Result<Vec<CurvePoint>> {
    let err = || bad(cmd, format!("`{key}` must be an array of 2..={MAX_CURVE_POINTS} [input, output] pairs in 0..255"));
    let a = v.as_array().ok_or_else(err)?;
    if a.len() < 2 || a.len() > 64 {
        return Err(err());
    }
    let mut pts = Vec::with_capacity(a.len());
    for q in a {
        let q = q.as_array().filter(|q| q.len() == 2).ok_or_else(err)?;
        // Like Photoshop, a point outside 0..255 is refused rather than moved onto the edge.
        let level = |v: Option<&Value>| v.and_then(Value::as_f64).filter(|v| (-1e-6..=255.0 + 1e-6).contains(v));
        let x = level(q.first()).ok_or_else(err)?;
        let y = level(q.get(1)).ok_or_else(err)?;
        pts.push(CurvePoint { input: (x as f32).clamp(0.0, 255.0) / 255.0, output: (y as f32).clamp(0.0, 255.0) / 255.0 });
    }
    pts.sort_by(|a, b| a.input.total_cmp(&b.input));
    pts.dedup_by(|later, earlier| (later.input - earlier.input).abs() < CURVE_POINT_DUPLICATE_DISTANCE);
    if pts.len() < 2 || pts.len() > MAX_CURVE_POINTS {
        return Err(err());
    }
    Ok(pts)
}

/// A pencil-mode map param: exactly [`CURVE_MAP_LEN`] outputs in 0..255.
fn parse_map(cmd: &str, key: &str, v: &Value) -> Result<Vec<CurvePoint>> {
    let err = || bad(cmd, format!("`{key}` must be {CURVE_MAP_LEN} output levels in 0..255 (a pencil-drawn map) or [input, output] pairs"));
    let a = v.as_array().filter(|a| a.len() == CURVE_MAP_LEN).ok_or_else(err)?;
    let outs =
        a.iter().map(|x| x.as_f64().filter(|x| x.is_finite()).map(|x| (x as f32).clamp(0.0, 255.0) / 255.0)).collect::<Option<Vec<_>>>().ok_or_else(err)?;
    Ok(curve_map(&outs))
}

/// A Curves channel param in the adjustment's mode: `[[in, out], …]` points (drawn into a map in
/// pencil mode) or a [`CURVE_MAP_LEN`]-level map (fitted with points outside pencil mode).
fn parse_curve_in(cmd: &str, key: &str, v: &Value, pencil: bool) -> Result<Vec<CurvePoint>> {
    if v.as_array().and_then(|a| a.first()).is_some_and(Value::is_number) {
        let map = parse_map(cmd, key, v)?;
        Ok(if pencil { map } else { map_to_points(&map) })
    } else {
        let pts = parse_curve(cmd, key, v)?;
        Ok(if pencil { points_to_map(&pts) } else { pts })
    }
}

fn curves(r: &Reader<'_>, base: Option<&Adjustment>, mode: ColorMode) -> Result<Adjustment> {
    let (mut master, mut chans, base_space, base_pencil) = match base {
        Some(Adjustment::Curves { master, per_channel, space, black, pencil }) => {
            (master.clone(), [per_channel[0].clone(), per_channel[1].clone(), per_channel[2].clone(), black.clone()], Some(*space), *pencil)
        }
        _ => (identity_curve(), std::array::from_fn(|_| identity_curve()), None, false),
    };
    let space = space_of(r, base_space, mode);
    if base_space.is_some_and(|b| b != space) {
        chans = std::array::from_fn(|_| identity_curve());
    }
    let pencil = r.boolean("pencil", base_pencil)?;
    // Every stored curve in the adjustment's mode: maps are drawn from the spline when switching
    // to the pencil, fitted with points when switching back, and filled out to full maps.
    let convert = |c: Vec<CurvePoint>| match (base_pencil, pencil) {
        (false, true) => points_to_map(&c),
        (true, false) => map_to_points(&c),
        (true, true) if c.len() != CURVE_MAP_LEN => curve_map(&curve_map_outputs(&c)),
        _ => c,
    };
    let [c0, c1, c2, k] = chans;
    let [m, c0, c1, c2, k] = [master, c0, c1, c2, k].map(convert);
    (master, chans) = (m, [c0, c1, c2, k]);
    let composite = r.get("points").or_else(|| r.get("gray"));
    if let Some(v) = composite {
        let c = parse_curve_in(r.cmd, "points", v, pencil)?;
        if space == ToneSpace::Lab {
            chans[0] = c;
        } else {
            master = c;
        }
    }
    for (i, key) in channel_keys(space).iter().enumerate() {
        if let Some(v) = r.get(key)
            && let Some(slot) = chans.get_mut(i)
        {
            *slot = parse_curve_in(r.cmd, key, v, pencil)?;
        }
    }
    let passes = r.num("smooth", 0.0, 0.0, MAX_SMOOTH_PASSES)?.round() as usize;
    if pencil {
        // Smoothed, then held at the params' precision so `to_params` round-trips exactly.
        let level = |o: f32| (o * 255.0 * CURVE_COORDINATE_PRECISION).round() / CURVE_COORDINATE_PRECISION / 255.0;
        for c in std::iter::once(&mut master).chain(chans.iter_mut()) {
            let mut outs = curve_map_outputs(c);
            for _ in 0..passes {
                smooth_curve_map(&mut outs);
            }
            *c = curve_map(&outs.into_iter().map(level).collect::<Vec<_>>());
        }
    }
    let [c0, c1, c2, black] = chans;
    let identity = if pencil { is_identity_map(&black) } else { photocraft_doc::adjust::is_identity_curve(&black) };
    let black = if space == ToneSpace::Cmyk && !identity { black } else { Vec::new() };
    Ok(Adjustment::Curves { master, per_channel: [c0, c1, c2], space, black, pencil })
}

fn levels_obj(c: &LevelsChannel) -> Value {
    let q = |v: f32| (v * 255.0 * 100.0).round() / 100.0;
    json!({"inBlack": q(c.in_black), "gamma": (c.gamma * 1000.0).round() / 1000.0, "inWhite": q(c.in_white), "outBlack": q(c.out_black), "outWhite": q(c.out_white)})
}

/// `[[location 0..1, colour], …]` (colour as for `color`), sorted by location.
fn gradient_stops(cmd: &str, v: &Value) -> Result<Vec<(f32, [f32; 3])>> {
    let err = || bad(cmd, "`stops` must be an array of 2..=64 [location 0..1, colour] pairs");
    let a = v.as_array().filter(|a| (2..=64).contains(&a.len())).ok_or_else(err)?;
    let mut out = Vec::with_capacity(a.len());
    for s in a {
        let (t, c) = match s {
            Value::Array(p) if p.len() == 2 => (p.first(), p.get(1)),
            Value::Object(o) => (o.get("location"), o.get("color")),
            _ => return Err(err()),
        };
        let t = t.and_then(Value::as_f64).filter(|t| t.is_finite()).ok_or_else(err)?;
        let c = color_of(cmd, "stops", c.ok_or_else(err)?)?;
        out.push(((t as f32).clamp(0.0, 1.0), c));
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(out)
}

/// Params with the private keys (`layer`, `coalesce`, `__kind`) removed.
pub fn user_keys(p: &Value) -> Map<String, Value> {
    p.as_object()
        .map(|o| {
            o.iter()
                .filter(|(k, _)| !matches!(k.as_str(), "layer" | "coalesce" | "channel" | "targetChannel") && !k.starts_with("__"))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    const KINDS: [&str; 15] = [
        "invert",
        "brightnessContrast",
        "threshold",
        "posterize",
        "exposure",
        "vibrance",
        "hueSaturation",
        "levels",
        "curves",
        "colorBalance",
        "blackWhite",
        "photoFilter",
        "channelMixer",
        "gradientMap",
        "selectiveColor",
    ];

    /// A non-default parameter set per kind (RGB channel keys for Levels/Curves).
    pub(crate) fn sample(kind: &str) -> Value {
        match kind {
            "brightnessContrast" => json!({"brightness": 20, "contrast": -10, "legacy": true}),
            "threshold" => json!({"level": 100}),
            "posterize" => json!({"levels": 7}),
            "exposure" => json!({"exposure": 1.5, "offset": 0.1, "gamma": 1.4}),
            "vibrance" => json!({"vibrance": 30, "saturation": -20}),
            "hueSaturation" => {
                json!({"hue": 25, "saturation": -20, "lightness": 10, "reds": {"hue": 10, "saturation": -30, "lightness": 5, "range": [300, 340, 20, 50]}})
            }
            "levels" => json!({"inBlack": 10, "inWhite": 240, "outBlack": 5, "outWhite": 250, "gamma": 1.2, "red": {"inBlack": 3, "gamma": 0.8}}),
            "curves" => json!({"points": [[0, 0], [128, 150], [255, 255]], "blue": [[0, 20], [255, 235]]}),
            "colorBalance" => json!({"shadows": [10, -5, 0], "midtones": [0, 20, -30], "highlights": [5, 5, 5], "preserveLuminosity": false}),
            "blackWhite" => json!({"reds": 120, "blues": -40, "tint": true, "tintColor": "#d0a070"}),
            "photoFilter" => json!({"color": "#3366cc", "density": 40, "preserveLuminosity": false}),
            "channelMixer" => json!({"red": [80, 30, -10, 5], "blue": [0, 10, 90, 0]}),
            "gradientMap" => json!({"stops": [[0, "#000000"], [0.5, "#ff0000"], [1, [1, 1, 1]]], "reverse": true, "dither": true}),
            "selectiveColor" => json!({"method": "absolute", "reds": [10, 0, -20, 5]}),
            _ => json!({}),
        }
    }

    #[test]
    fn every_kind_round_trips_through_params() {
        for mode in [ColorMode::Rgb, ColorMode::Cmyk, ColorMode::Lab, ColorMode::Grayscale] {
            for kind in KINDS {
                let mut p = sample(kind);
                if matches!(kind, "levels" | "curves") && default_space(mode) != ToneSpace::Rgb {
                    // Off RGB the channel keys are the document's.
                    p.as_object_mut().unwrap().retain(|k, _| !matches!(k.as_str(), "red" | "blue"));
                }
                let a = from_params(kind, &p, None, mode).unwrap_or_else(|e| panic!("{kind} {mode:?}: {e}"));
                assert_ne!(Some(&a), default_for(kind, mode).ok().as_ref().filter(|_| kind != "invert"), "{kind} params ignored");
                let back = from_params(kind, &to_params(&a), None, mode).unwrap();
                assert_eq!(to_params(&back), to_params(&a), "{kind} {mode:?}");
                // A partial update keeps everything it doesn't name.
                let again = from_params(kind, &json!({}), Some(&a), mode).unwrap();
                assert_eq!(again, a, "{kind} {mode:?}");
            }
        }
    }

    #[test]
    fn params_reach_the_model() {
        let a = from_params("colorBalance", &json!({"midtones": [50, 0, -200]}), None, ColorMode::Rgb).unwrap();
        assert_eq!(a, Adjustment::ColorBalance { shadows: [0.0; 3], midtones: [50.0, 0.0, -100.0], highlights: [0.0; 3], preserve_luminosity: true });
        let a = from_params("channelMixer", &json!({"monochrome": true}), None, ColorMode::Rgb).unwrap();
        assert!(matches!(a, Adjustment::ChannelMixer { matrix, monochrome: true } if matrix[0] == [0.4, 0.4, 0.2, 0.0]));
        let a = from_params("photoFilter", &json!({"filter": "cooling80", "density": 60}), None, ColorMode::Rgb).unwrap();
        assert!(matches!(a, Adjustment::PhotoFilter { color, density, .. } if color[2] == 1.0 && (density - 0.6).abs() < 1e-6));
        let a = from_params("blackWhite", &json!({"reds": 120, "tint": true}), None, ColorMode::Rgb).unwrap();
        assert!(matches!(a, Adjustment::BlackWhite { weights, tint: Some(_) } if weights[0] == 120.0 && weights[1] == 60.0));
        let a = from_params("hueSaturation", &json!({"blues": {"hue": 40}}), None, ColorMode::Rgb).unwrap();
        assert!(matches!(&a, Adjustment::HueSaturation { ranges, .. } if ranges[4].hue == 40.0 && ranges[0].is_neutral()));
        // Colorize without a saturation starts at 25%; an explicit 0 and an already colorized base are kept.
        let a = from_params("hueSaturation", &json!({"colorize": true, "hue": 30}), None, ColorMode::Rgb).unwrap();
        assert!(matches!(a, Adjustment::HueSaturation { colorize: true, saturation: 25.0, .. }), "{a:?}");
        let a = from_params("hueSaturation", &json!({"colorize": true, "saturation": 0}), None, ColorMode::Rgb).unwrap();
        assert!(matches!(a, Adjustment::HueSaturation { colorize: true, saturation: 0.0, .. }), "{a:?}");
        let a = from_params("hueSaturation", &json!({"hue": 120}), Some(&a), ColorMode::Rgb).unwrap();
        assert!(matches!(a, Adjustment::HueSaturation { colorize: true, saturation: 0.0, hue: 120.0, .. }), "{a:?}");
        let a = from_params("gradientMap", &json!({"stops": [[1, "#ffffff"], [0, "#000000"]], "dither": true}), None, ColorMode::Rgb).unwrap();
        assert!(matches!(&a, Adjustment::GradientMap { stops, dither: true, .. } if stops[0].0 == 0.0));
        // CMYK curves: ink channels; Lab: lightness without a composite.
        let a = from_params("curves", &json!({"black": [[0, 0], [255, 200]]}), None, ColorMode::Rgb).unwrap();
        assert!(matches!(&a, Adjustment::Curves { space: ToneSpace::Cmyk, black, .. } if black.len() == 2));
        let a = from_params("curves", &json!({"points": [[0, 40], [255, 255]]}), None, ColorMode::Lab).unwrap();
        assert!(matches!(&a, Adjustment::Curves { space: ToneSpace::Lab, per_channel, .. } if per_channel[0][0].output > 0.1));
        let a = from_params("levels", &json!({"inBlack": 20}), None, ColorMode::Cmyk).unwrap();
        assert!(matches!(&a, Adjustment::Levels { space: ToneSpace::Cmyk, master, .. } if (master.in_black - 20.0 / 255.0).abs() < 1e-6));
    }

    #[test]
    fn bad_params_are_errors_not_panics() {
        for (kind, p) in [
            ("curves", json!({"points": "abc"})),
            ("curves", json!({"points": [[0, 0]]})),
            ("curves", json!({"red": [[0, 0], [1]]})),
            ("curves", json!({"points": [[5, 5], [5, 9]]})),
            ("curves", json!({"pencil": true, "points": [1, 2, 3]})),
            ("curves", json!({"pencil": "yes"})),
            ("curves", json!({"points": (0..256).map(|k| if k == 9 { json!("x") } else { json!(k) }).collect::<Vec<_>>()})),
            ("levels", json!({"red": 3})),
            ("levels", json!({"inBlack": "x"})),
            ("hueSaturation", json!({"reds": {"range": [1, 2]}})),
            ("hueSaturation", json!({"colorize": 1})),
            ("colorBalance", json!({"shadows": [1, 2]})),
            ("blackWhite", json!({"tintColor": "#zz0000"})),
            ("photoFilter", json!({"filter": "nope"})),
            ("photoFilter", json!({"color": [1]})),
            ("channelMixer", json!({"red": [1, 2, 3]})),
            ("gradientMap", json!({"stops": [[0, "#000"]]})),
            ("gradientMap", json!({"stops": 7})),
            ("selectiveColor", json!({"method": 5})),
            ("selectiveColor", json!({"reds": "not-an-array"})),
            ("selectiveColor", json!({"colors": "reds", "cyan": "50"})),
            ("colorLookup", json!({"lut": 42})),
            ("colorLookup", json!({"interpolation": 5})),
            ("colorLookup", json!({"dither": "yes"})),
            ("nope", json!({})),
            ("levels", json!([1, 2])),
        ] {
            assert!(from_params(kind, &p, None, ColorMode::Rgb).is_err(), "{kind} {p}");
        }
        // Out-of-range numbers clamp.
        let a = from_params("hueSaturation", &json!({"hue": 1e9, "saturation": -1e9}), None, ColorMode::Rgb).unwrap();
        assert!(matches!(a, Adjustment::HueSaturation { hue: 180.0, saturation: -100.0, .. }));
        let a = from_params("levels", &json!({"inBlack": 250, "inWhite": 10}), None, ColorMode::Rgb).unwrap();
        assert!(matches!(a, Adjustment::Levels { master, .. } if master.in_white > master.in_black));
    }

    #[test]
    fn curve_points_outside_0_255_are_refused_as_the_message_says() {
        // Photoshop refuses them too ("the settings in the file are invalid"); they used to be
        // moved onto the edge silently.
        for pts in [json!([[0, 0], [300, 255]]), json!([[0, -1], [255, 255]]), json!([[0, 0]])] {
            let e = from_params("curves", &json!({ "points": pts }), None, ColorMode::Rgb).unwrap_err().to_string();
            assert!(e.contains("[input, output] pairs in 0..255") && e.contains("`curves adjustment`"), "{e}");
        }
        assert!(from_params("curves", &json!({"points": [[0, 0], [128.5, 140], [255, 255]]}), None, ColorMode::Rgb).is_ok());
    }

    fn curves_parts(a: &Adjustment) -> (&[CurvePoint], &[Vec<CurvePoint>; 3]) {
        match a {
            Adjustment::Curves { master, per_channel, .. } => (master, per_channel),
            _ => panic!("expected curves"),
        }
    }

    fn tone_output(adjustment: &Adjustment, sample: [f32; 3]) -> [f32; 3] {
        let luts = photocraft_compose::adjust::tone_luts(adjustment);
        std::array::from_fn(|channel| {
            let row = &luts[channel];
            let position = sample[channel].clamp(0.0, 1.0) * (row.len().saturating_sub(1)) as f32;
            let lower = position.floor() as usize;
            let upper = lower.saturating_add(1).min(row.len().saturating_sub(1));
            let fraction = position - lower as f32;
            row.get(lower).copied().unwrap_or(0.0) * (1.0 - fraction) + row.get(upper).copied().unwrap_or(0.0) * fraction
        })
    }

    #[test]
    fn curves_black_and_white_eyedroppers_move_the_input_references() {
        let base = json!({
            "points": [[0, 8], [120, 150], [255, 245]],
            "red": [[0, 0], [100, 130], [255, 255]],
            "green": [[0, 0], [110, 125], [255, 255]],
            "blue": [[0, 0], [90, 120], [255, 255]]
        });
        let black = curves_eyedropper(&base, CurvesEyedropper::Black, [0.2, 0.3, 0.4], ColorMode::Rgb).unwrap();
        let (master, channels) = curves_parts(&black);
        for (curve, sample) in channels.iter().zip([0.2, 0.3, 0.4]) {
            assert!((curve[0].input - sample).abs() < 1e-6 && curve[0].output == 0.0, "{curve:?}");
            assert_eq!(curve.len(), 3, "existing edit points are preserved");
        }
        assert_eq!(master[0].output, 0.0, "the composite cannot lift the sampled black");
        assert!(tone_output(&black, [0.2, 0.3, 0.4]).iter().all(|value| *value < 1e-3));

        let white = curves_eyedropper(&base, CurvesEyedropper::White, [0.6, 0.7, 0.8], ColorMode::Rgb).unwrap();
        let (master, channels) = curves_parts(&white);
        for (curve, sample) in channels.iter().zip([0.6, 0.7, 0.8]) {
            let end = curve.last().unwrap();
            assert!((end.input - sample).abs() < 1e-6 && end.output == 1.0, "{curve:?}");
            assert_eq!(curve.len(), 3, "existing edit points are preserved");
        }
        assert_eq!(master.last().map(|p| p.output), Some(1.0), "the composite cannot lower the sampled white");
        assert!(tone_output(&white, [0.6, 0.7, 0.8]).iter().all(|value| *value > 1.0 - 1e-3));

        let edge_black = curves_eyedropper(&base, CurvesEyedropper::Black, [1.0, 0.3, 0.4], ColorMode::Rgb).unwrap();
        let (_, edge_channels) = curves_parts(&edge_black);
        assert_eq!(edge_channels[0].len(), 3, "edge samples preserve other curve points");
        assert_eq!(edge_channels[0].last().map(|p| p.output), Some(0.0));
        let edge_white = curves_eyedropper(&base, CurvesEyedropper::White, [0.0, 0.7, 0.8], ColorMode::Rgb).unwrap();
        let (_, edge_channels) = curves_parts(&edge_white);
        assert_eq!(edge_channels[0].len(), 3, "edge samples preserve other curve points");
        assert_eq!(edge_channels[0].first().map(|p| p.output), Some(1.0));
    }

    #[test]
    fn black_and_white_eyedroppers_round_trip_near_serialization_boundaries() {
        let cases = [
            (CurvesEyedropper::White, 0.0, 1.0),
            (CurvesEyedropper::White, 0.001, 1.0),
            (CurvesEyedropper::White, 0.494 / 255.0, 1.0),
            // 0.496 level rounds to 0.50; the sample is just below that representable point.
            (CurvesEyedropper::White, 0.496 / 255.0, 0.992),
            (CurvesEyedropper::White, 0.2, 1.0),
            (CurvesEyedropper::Black, 1.0, 0.0),
            (CurvesEyedropper::Black, 0.999, 0.0),
            (CurvesEyedropper::Black, 1.0 - 0.494 / 255.0, 0.0),
            // 254.504 levels rounds to 254.50, leaving a representable 0.50-level gap.
            (CurvesEyedropper::Black, 1.0 - 0.496 / 255.0, 0.008),
            (CurvesEyedropper::Black, 0.8, 0.0),
        ];
        for (picker, sample, expected_output) in cases {
            let resolved = curves_eyedropper(&json!({}), picker, [sample; 3], ColorMode::Rgb).unwrap();
            let params = to_params(&resolved);
            let parsed = from_params("curves", &params, None, ColorMode::Rgb).unwrap();
            let (master, channels) = curves_parts(&parsed);
            for curve in std::iter::once(master).chain(channels.iter().map(Vec::as_slice)) {
                assert!(curve.len() >= 2, "{picker:?} sample={sample}: {curve:?}");
                assert!(curve.len() <= MAX_CURVE_POINTS, "{picker:?} sample={sample}: {curve:?}");
                assert!(curve.windows(2).all(|pair| pair[0].input < pair[1].input), "{picker:?} sample={sample}: {curve:?}");
                assert!(curve.windows(2).all(|pair| pair[1].input - pair[0].input >= CURVE_POINT_DUPLICATE_DISTANCE), "{picker:?} sample={sample}: {curve:?}");
            }
            let output = tone_output(&parsed, [sample; 3]);
            assert!(
                output.iter().all(|value| (*value - expected_output).abs() < 1e-3),
                "{picker:?} sample={sample} output={output:?} expected={expected_output}"
            );
        }
    }

    #[test]
    fn endpoint_eyedroppers_fall_back_when_serialization_collapses_curve_points() {
        let base = json!({
            "points": [[0, 0], [64, 50], [128, 100], [192, 200], [255, 255]],
            "red": [[0, 0], [64, 50], [128, 100], [192, 200], [255, 255]],
            "green": [[0, 0], [64, 50], [128, 100], [192, 200], [255, 255]],
            "blue": [[0, 0], [64, 50], [128, 100], [192, 200], [255, 255]]
        });
        let cases = [(CurvesEyedropper::Black, 0.995, 0.0, true), (CurvesEyedropper::White, 0.005, 1.0, false)];
        for (picker, sample, target, black) in cases {
            let resolved = curves_eyedropper(&base, picker, [sample; 3], ColorMode::Rgb).unwrap();
            let parsed = from_params("curves", &to_params(&resolved), None, ColorMode::Rgb).unwrap();
            let (master, channels) = curves_parts(&parsed);
            for curve in std::iter::once(master).chain(channels.iter().map(Vec::as_slice)) {
                assert_eq!(curve.len(), 5, "collapsed points use the deterministic fallback: {curve:?}");
                for (point, input) in curve.iter().zip([0.0, 64.0, 128.0, 192.0, 255.0]) {
                    assert!((point.input * 255.0 - input).abs() < 1e-3, "existing point position was retained: {curve:?}");
                }
            }
            for curve in channels {
                let plateau: Vec<_> = if black { curve.iter().rev().take(2).collect() } else { curve.iter().take(2).collect() };
                assert!(plateau.into_iter().all(|point| (point.output - target).abs() < 1e-6), "{curve:?}");
            }
            let output = tone_output(&parsed, [sample; 3]);
            assert!(output.iter().all(|value| (*value - target).abs() < 0.01), "{picker:?}: {output:?}");
        }
    }

    #[test]
    fn black_and_white_round_trips_preserve_representable_curve_edits() {
        let base = json!({
            "points": [[0, 8], [120, 150], [255, 245]],
            "red": [[0, 0], [100, 130], [255, 255]],
            "green": [[0, 0], [110, 125], [255, 255]],
            "blue": [[0, 0], [90, 120], [255, 255]]
        });
        let black = curves_eyedropper(&base, CurvesEyedropper::Black, [0.2, 0.3, 0.4], ColorMode::Rgb).unwrap();
        let black = from_params("curves", &to_params(&black), None, ColorMode::Rgb).unwrap();
        let (_, channels) = curves_parts(&black);
        assert_eq!(channels[0].len(), 3);
        assert!((channels[0][1].input * 255.0 - 131.0).abs() < 1e-3);
        assert!((channels[0][1].output * 255.0 - 130.0).abs() < 1e-3);

        let white = curves_eyedropper(&base, CurvesEyedropper::White, [0.6, 0.7, 0.8], ColorMode::Rgb).unwrap();
        let white = from_params("curves", &to_params(&white), None, ColorMode::Rgb).unwrap();
        let (_, channels) = curves_parts(&white);
        assert_eq!(channels[0].len(), 3);
        assert!((channels[0][1].input * 255.0 - 60.0).abs() < 1e-3);
        assert!((channels[0][1].output * 255.0 - 130.0).abs() < 1e-3);
    }

    fn assert_gray_point_preserves_luma_and_existing_curve_edits(sample: [f32; 3]) {
        let base = json!({
            "points": [[0, 0], [128, 96], [255, 255]],
            "red": [[0, 0], [80, 90], [255, 255]],
            "green": [[0, 0], [90, 100], [255, 255]],
            "blue": [[0, 0], [70, 80], [255, 255]]
        });
        let original = from_params("curves", &base, None, ColorMode::Rgb).unwrap();
        let expected_luma = rgb_luma(sample);
        let resolved = curves_eyedropper(&base, CurvesEyedropper::NeutralGray, sample, ColorMode::Rgb).unwrap();
        let parsed = from_params("curves", &to_params(&resolved), None, ColorMode::Rgb).unwrap();
        let (master, channels) = curves_parts(&parsed);
        let (original_master, original_channels) = curves_parts(&original);
        assert_eq!(master, original_master, "the existing master curve is retained");
        for (curve, old_curve) in channels.iter().zip(original_channels) {
            let old_edit = old_curve.get(1).expect("the test curve has an interior edit");
            assert!(
                curve.iter().any(|point| { (point.input - old_edit.input).abs() < 1e-6 && (point.output - old_edit.output).abs() < 1e-6 }),
                "the existing channel edit is retained: {curve:?}"
            );
        }
        let output = tone_output(&parsed, sample);
        assert!((output[0] - output[1]).abs() < 0.002 && (output[1] - output[2]).abs() < 0.002, "sample must become neutral: {output:?}");
        assert!((rgb_luma(output) - expected_luma).abs() < 0.003, "luma changed from {expected_luma} to {} ({output:?})", rgb_luma(output));
    }

    #[test]
    fn curves_neutral_gray_preserves_luma_for_a_light_cast_sample() {
        assert_gray_point_preserves_luma_and_existing_curve_edits([0.88, 0.76, 0.63]);
    }

    #[test]
    fn curves_neutral_gray_preserves_luma_for_a_dark_cast_sample() {
        assert_gray_point_preserves_luma_and_existing_curve_edits([0.17, 0.105, 0.075]);
    }

    #[test]
    fn curves_neutral_gray_handles_near_black_and_near_white_samples() {
        for sample in [[0.004, 0.006, 0.008], [0.992, 0.996, 0.999], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]] {
            let resolved = curves_eyedropper(&json!({}), CurvesEyedropper::NeutralGray, sample, ColorMode::Rgb).unwrap();
            let parsed = from_params("curves", &to_params(&resolved), None, ColorMode::Rgb).unwrap();
            let (master, channels) = curves_parts(&parsed);
            for curve in std::iter::once(master).chain(channels.iter().map(Vec::as_slice)) {
                assert!(curve.len() >= 2, "{sample:?}: {curve:?}");
                assert!(curve.windows(2).all(|pair| pair[0].input < pair[1].input), "{sample:?}: {curve:?}");
                assert!(curve.windows(2).all(|pair| pair[1].input - pair[0].input >= CURVE_POINT_DUPLICATE_DISTANCE), "{sample:?}: {curve:?}");
            }
            let output = tone_output(&parsed, sample);
            assert!((output[0] - output[1]).abs() < 0.002 && (output[1] - output[2]).abs() < 0.002, "{sample:?}: {output:?}");
            assert!((rgb_luma(output) - rgb_luma(sample)).abs() < 0.003, "{sample:?}: {output:?}");
        }
    }

    #[test]
    fn gray_point_fallback_keeps_tightly_spaced_points_round_trip_safe() {
        let base = json!({
            "red": [[0, 0], [100, 100], [100.51, 100], [255, 255]],
            "green": [[0, 0], [100, 100], [100.51, 100], [255, 255]],
            "blue": [[0, 0], [100, 100], [100.51, 100], [255, 255]]
        });
        let sample = [100.25 / 255.0, 100.1 / 255.0, 100.4 / 255.0];
        let target = rgb_luma(sample);
        let resolved = curves_eyedropper(&base, CurvesEyedropper::NeutralGray, sample, ColorMode::Rgb).unwrap();
        let parsed = from_params("curves", &to_params(&resolved), None, ColorMode::Rgb).unwrap();
        let (_, channels) = curves_parts(&parsed);
        for (curve, input) in channels.iter().zip(sample) {
            assert_eq!(curve.len(), 4, "the fallback keeps every serialized point: {curve:?}");
            assert!(curve.windows(2).all(|pair| pair[1].input - pair[0].input >= CURVE_POINT_DUPLICATE_DISTANCE), "{curve:?}");
            let sample_pair = curve.windows(2).find(|pair| pair[0].input <= input && input <= pair[1].input).unwrap();
            assert!((sample_pair[0].output - sample_pair[1].output).abs() < 1e-6, "fallback forms a plateau around the sample: {curve:?}");
            assert!((sample_pair[0].output - target).abs() < 0.001, "fallback uses the luma target: {curve:?}");
        }
    }

    #[test]
    fn curves_eyedropper_param_drives_the_command() {
        let mut s = crate::Session::new();
        s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
        s.execute("edit.fill", json!({"color": [0.6, 0.5, 0.4]})).unwrap();
        s.execute("image.adjustments.curves", json!({"eyedropper": {"point": "white", "at": [2, 2]}})).unwrap();
        let px: Vec<f32> = serde_json::from_value(s.execute("document.pixel", json!({"x": 2, "y": 2})).unwrap()).unwrap();
        assert!(px.iter().take(3).all(|v| *v > 0.99), "the sample becomes white: {px:?}");
        s.execute("image.adjustments.curves", json!({"eyedropper": {"point": "black", "color": [1.0, 1.0, 1.0]}})).unwrap();
        let px: Vec<f32> = serde_json::from_value(s.execute("document.pixel", json!({"x": 2, "y": 2})).unwrap()).unwrap();
        assert!(px.iter().take(3).all(|v| *v < 0.01), "the given colour becomes black: {px:?}");
        for bad in [
            json!("black"),
            json!({"point": "nope", "at": [1, 1]}),
            json!({"point": "black"}),
            json!({"point": "black", "at": [99, 1]}),
            json!({"point": "black", "at": [-1, 1]}),
            json!({"point": "black", "at": [1]}),
            json!({"point": "gray", "at": [f64::MAX, 1]}),
            json!({"point": "black", "color": "red"}),
        ] {
            assert!(s.execute("image.adjustments.curves", json!({"eyedropper": bad})).is_err(), "{bad}");
        }
    }

    /// The Properties panel's pickers on a Curves adjustment layer (UI-217-15): one undoable step
    /// that samples the image below the layer, so a repeated click keeps the same curve.
    #[test]
    fn layer_eyedropper_samples_below_the_curves_layer() {
        let mut s = crate::Session::new();
        s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
        s.execute("edit.fill", json!({"color": [0.6, 0.5, 0.4]})).unwrap();
        s.execute("layer.newAdjustmentLayer.curves", json!({})).unwrap();
        let steps = s.active().unwrap().history.past_len();
        let pick = json!({"eyedropper": {"point": "white", "at": [2, 2], "size": 3}});
        s.execute("layer.setAdjustment", pick.clone()).unwrap();
        assert_eq!(s.active().unwrap().history.past_len(), steps + 1);
        let px: Vec<f32> = serde_json::from_value(s.execute("document.pixel", json!({"x": 2, "y": 2})).unwrap()).unwrap();
        assert!(px.iter().take(3).all(|v| *v > 0.99), "the sample becomes white: {px:?}");
        let curve = s.active().unwrap().doc.clone();
        s.execute("layer.setAdjustment", pick).unwrap();
        let same = |a: &photocraft_doc::Document, b: &photocraft_doc::Document| a.layers.last().map(|l| &l.content) == b.layers.last().map(|l| &l.content);
        assert!(same(&curve, &s.active().unwrap().doc), "sampled below the layer, not through it");
        s.execute("edit.undo", json!({})).unwrap();
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(s.active().unwrap().history.past_len(), steps);
        for bad in [json!({"point": "black", "at": [99, 1]}), json!({"point": "black", "at": [1, 1], "size": 2}), json!({"point": "x", "at": [1, 1]})] {
            assert!(s.execute("layer.setAdjustment", json!({"eyedropper": bad})).is_err(), "{bad}");
        }
        s.execute("layer.newAdjustmentLayer.hueSaturation", json!({})).unwrap();
        assert!(s.execute("layer.setAdjustment", json!({"eyedropper": {"point": "white", "at": [1, 1]}})).is_err(), "Levels and Curves only");
    }

    /// #3127: on a smart object the eyedropper param records the adjustment it resolved to as
    /// the smart filter, not the raw sample (which re-rendered as an identity).
    #[test]
    fn eyedropper_on_a_smart_object_records_the_resolved_adjustment() {
        for kind in ["curves", "levels"] {
            let mut s = crate::Session::new();
            s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
            s.execute("edit.fill", json!({"color": [0.6, 0.5, 0.4]})).unwrap();
            s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
            s.execute(&format!("image.adjustments.{kind}"), json!({"eyedropper": {"point": "white", "at": [2, 2]}})).unwrap();
            let px: Vec<f32> = serde_json::from_value(s.execute("document.pixel", json!({"x": 2, "y": 2})).unwrap()).unwrap();
            assert!(px.iter().take(3).all(|v| *v > 0.99), "{kind}: {px:?}");
        }
    }

    /// Levels' Set White/Neutral Gray Point (#3116): the white point maps the sampled colour to
    /// white at 8 and 16 bit, the gray point neutralises a tinted grey at its luma on a Levels
    /// layer as one undoable step, and bad picks fail.
    #[test]
    fn levels_eyedroppers_set_white_and_neutral_gray() {
        let pixel = |s: &mut crate::Session| -> Vec<f32> { serde_json::from_value(s.execute("document.pixel", json!({"x": 2, "y": 2})).unwrap()).unwrap() };
        for depth in [8, 16] {
            let mut s = crate::Session::new();
            s.execute("file.new", json!({"width": 8, "height": 8, "depth": depth})).unwrap();
            s.execute("edit.fill", json!({"color": [0.6, 0.5, 0.4]})).unwrap();
            s.execute("image.adjustments.levels", json!({"eyedropper": {"point": "white", "at": [2, 2]}})).unwrap();
            let px = pixel(&mut s);
            assert!(px.iter().take(3).all(|v| *v > 0.995), "{depth}-bit: the sample becomes white: {px:?}");
        }
        let mut s = crate::Session::new();
        s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
        s.execute("edit.fill", json!({"color": [0.6, 0.5, 0.4]})).unwrap();
        let before = pixel(&mut s);
        s.execute("layer.newAdjustmentLayer.levels", json!({})).unwrap();
        let steps = s.active().unwrap().history.past_len();
        s.execute("layer.setAdjustment", json!({"eyedropper": {"point": "gray", "at": [2, 2]}})).unwrap();
        assert_eq!(s.active().unwrap().history.past_len(), steps + 1);
        let px = pixel(&mut s);
        let (lo, hi) = px.iter().take(3).fold((1.0f32, 0.0f32), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
        assert!(hi - lo < 2.5 / 255.0, "neutral: {px:?}");
        let luma = |c: &[f32]| c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722;
        assert!((luma(&px) - luma(&before)).abs() < 2.0 / 255.0, "keeps the luma: {px:?} vs {before:?}");
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(pixel(&mut s), before);
        for bad in [json!({"point": "gray", "at": [99, 1]}), json!({"point": "x", "at": [1, 1]}), json!({"point": "white", "color": [f64::NAN, 0, 0]})] {
            assert!(s.execute("layer.setAdjustment", json!({"eyedropper": bad})).is_err(), "{bad}");
            assert!(s.execute("image.adjustments.levels", json!({"eyedropper": bad})).is_err(), "{bad}");
        }
        s.execute("file.new", json!({"width": 8, "height": 8, "mode": "grayscale"})).unwrap();
        assert!(s.execute("image.adjustments.levels", json!({"eyedropper": {"point": "gray", "at": [1, 1]}})).is_err(), "no gray picker in Grayscale");
        s.execute("image.adjustments.levels", json!({"eyedropper": {"point": "black", "color": [1.0, 1.0, 1.0]}})).unwrap();
    }

    #[test]
    fn curves_eyedropper_support_is_explicit_and_bad_samples_fail() {
        assert!(curves_eyedropper(&json!({}), CurvesEyedropper::Black, [0.4; 3], ColorMode::Grayscale).is_ok());
        assert!(curves_eyedropper(&json!({}), CurvesEyedropper::White, [0.4; 3], ColorMode::Grayscale).is_ok());
        assert!(curves_eyedropper(&json!({}), CurvesEyedropper::NeutralGray, [0.4; 3], ColorMode::Grayscale).is_err());
        for mode in [ColorMode::Cmyk, ColorMode::Lab] {
            assert!(curves_eyedropper(&json!({}), CurvesEyedropper::Black, [0.4; 3], mode).is_err());
        }
        assert!(curves_eyedropper(&json!({}), CurvesEyedropper::Black, [f32::NAN, 0.0, 0.0], ColorMode::Rgb).is_err());
    }

    #[test]
    fn hue_range_sliders_stay_ordered() {
        let a = from_params("hueSaturation", &json!({"greens": {"range": [100, 90, 200, 150]}}), None, ColorMode::Rgb).unwrap();
        let Adjustment::HueSaturation { ranges, .. } = a else { panic!() };
        let b = ranges[2].bounds;
        assert!(b[0] <= b[1] && b[1] <= b[2] && b[2] <= b[3] && b[3] - b[0] < 360.0, "{b:?}");
    }

    /// A colourful 16×16 document (`mode`, `depth`) for command tests.
    fn colourful(mode: &str, depth: u32) -> crate::Session {
        let mut s = crate::Session::new();
        s.execute("file.new", json!({"width": 16, "height": 16, "mode": mode, "depth": depth})).unwrap();
        for (i, c) in ["#c83c28", "#3c8c50", "#2850b4", "#d2b464"].iter().enumerate() {
            s.execute("select.rect", json!({"x": (i % 2) * 8, "y": (i / 2) * 8, "width": 8, "height": 8})).unwrap();
            s.execute("edit.fill", json!({"color": c})).unwrap();
        }
        s.execute("select.deselect", json!({})).unwrap();
        s
    }

    fn pixels(s: &crate::Session) -> Vec<Vec<f32>> {
        let d = s.active().unwrap();
        let bg = d.doc.layers[0].surface().unwrap();
        [(2, 2), (10, 2), (2, 10), (10, 10)].iter().map(|&(x, y)| bg.pixel(x, y)).collect()
    }

    #[test]
    fn destructive_commands_apply_their_params_at_every_depth() {
        for depth in [8, 16, 32] {
            for kind in KINDS.iter().filter(|k| **k != "invert") {
                let mut s = colourful("rgb", depth);
                let before = pixels(&s);
                let steps = s.active().unwrap().history.past_len();
                s.execute(&format!("image.adjustments.{kind}"), sample(kind)).unwrap_or_else(|e| panic!("{kind} @{depth}: {e}"));
                assert_ne!(pixels(&s), before, "{kind} @{depth} changed nothing");
                assert_eq!(s.active().unwrap().history.past_len(), steps + 1, "{kind}: one history step");
                // Identity params leave the pixels alone (within one 8-bit step).
                let mut s = colourful("rgb", depth);
                s.execute(&format!("image.adjustments.{kind}"), to_params(&default_for(kind, ColorMode::Rgb).unwrap())).unwrap();
                if !matches!(*kind, "threshold" | "posterize" | "blackWhite" | "gradientMap" | "photoFilter") {
                    for (a, b) in pixels(&s).iter().zip(&before) {
                        assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1.0 / 255.0 + 1e-4), "{kind} @{depth}: {a:?} vs {b:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn curves_pencil_maps_draw_smooth_and_convert_back() {
        // A drawn inversion: 256 output levels.
        let inverted: Vec<f32> = (0..256).rev().map(|k| k as f32).collect();
        let a = from_params("curves", &json!({"pencil": true, "points": inverted}), None, ColorMode::Rgb).unwrap();
        assert!(matches!(&a, Adjustment::Curves { pencil: true, master, .. } if master.len() == CURVE_MAP_LEN));
        assert!((tone_output(&a, [0.25; 3])[0] - 0.75).abs() < 1e-3);
        assert_eq!(from_params("curves", &to_params(&a), None, ColorMode::Rgb).unwrap(), a);
        // Smooth evens out a hard step but keeps the endpoints.
        let step: Vec<f32> = (0..256).map(|k| if k < 128 { 0.0 } else { 255.0 }).collect();
        let s = from_params("curves", &json!({"red": step, "smooth": 2}), Some(&a), ColorMode::Rgb).unwrap();
        let (_, chans) = curves_parts(&s);
        assert!(chans[0][127].output > 0.2 && chans[0][128].output < 0.8, "{:?}", &chans[0][120..136]);
        assert_eq!((chans[0][0].output, chans[0][255].output), (0.0, 1.0));
        // Edit points: the map is fitted with a point curve that still inverts.
        let pts = from_params("curves", &json!({"pencil": false}), Some(&a), ColorMode::Rgb).unwrap();
        let (master, _) = curves_parts(&pts);
        assert!(matches!(&pts, Adjustment::Curves { pencil: false, .. }) && master.len() <= MAX_FITTED_POINTS, "{master:?}");
        assert!((tone_output(&pts, [0.25; 3])[0] - 0.75).abs() < 2.0 / 255.0);
        // The command draws it at every depth.
        for depth in [8, 16, 32] {
            let mut s = colourful("rgb", depth);
            let before = pixels(&s);
            s.execute("image.adjustments.curves", to_params(&a)).unwrap();
            let after = pixels(&s);
            assert!((after[0][0] - (1.0 - before[0][0])).abs() < 2.0 / 255.0, "@{depth}: {after:?} vs {before:?}");
        }
    }

    #[test]
    fn destructive_levels_dont_clip_in_32_bit() {
        // Input 15..230, gamma 1.3, output 10..245: integer documents stop at the output range,
        // while float documents preserve the curve's signed and HDR values beyond it (the
        // compose-side `levels_dont_clip_in_32_bit` pins the same curve, #2592).
        let params = json!({"inBlack": 15, "inWhite": 230, "gamma": 1.3, "outBlack": 10, "outWhite": 245});
        for depth in [8, 16, 32] {
            let mut s = crate::Session::new();
            s.execute("file.new", json!({"width": 16, "height": 16, "depth": depth, "background": "white"})).unwrap();
            s.execute("select.rect", json!({"x": 0, "y": 0, "width": 8, "height": 16})).unwrap();
            s.execute("edit.fill", json!({"color": "#000000"})).unwrap();
            s.execute("select.deselect", json!({})).unwrap();
            s.execute("image.adjustments.levels", params.clone()).unwrap();
            let px = pixels(&s);
            if depth == 32 {
                assert!(px[0][0].is_finite() && px[0][0] < 0.0, "float black keeps its signed result: {:?}", px[0]);
                assert!(px[1][0].is_finite() && px[1][0] > 1.0, "float white keeps its HDR result: {:?}", px[1]);
            } else {
                assert!((px[0][0] * 255.0 - 10.0).abs() < 0.6, "black @{depth}: {:?}", px[0]);
                assert!((px[1][0] * 255.0 - 245.0).abs() < 0.6, "white @{depth}: {:?}", px[1]);
            }
        }
    }

    #[test]
    fn cmyk_and_lab_documents_use_their_channels() {
        let mut s = colourful("cmyk", 8);
        let before = pixels(&s);
        // More black ink darkens every patch.
        s.execute("image.adjustments.curves", json!({"black": [[0, 0], [255, 180]]})).unwrap();
        let after = pixels(&s);
        assert!(after.iter().zip(&before).all(|(a, b)| a[3] > b[3] + 0.05), "{after:?} vs {before:?}");
        let mut s = colourful("lab", 16);
        let before = pixels(&s);
        s.execute("image.adjustments.levels", json!({"lightness": {"outBlack": 60}})).unwrap();
        let after = pixels(&s);
        assert!(after.iter().zip(&before).all(|(a, b)| a[0] >= b[0]), "lightness lifted {after:?} vs {before:?}");
        // The adjustment layer of a CMYK document defaults to ink channels.
        let mut s = colourful("cmyk", 8);
        s.execute("layer.newAdjustmentLayer.curves", json!({})).unwrap();
        let id = s.active().unwrap().active_layer.unwrap();
        let l = s.active().unwrap().doc.layer(id).unwrap().clone();
        assert!(matches!(l.content, photocraft_doc::LayerContent::Adjustment(Adjustment::Curves { space: ToneSpace::Cmyk, .. })));
    }

    #[test]
    fn set_adjustment_merges_and_resets() {
        let mut s = colourful("rgb", 8);
        s.execute("layer.newAdjustmentLayer.colorBalance", json!({"midtones": [30, 0, 0]})).unwrap();
        s.execute("layer.setAdjustment", json!({"highlights": [0, 0, -20]})).unwrap();
        let id = s.active().unwrap().active_layer.unwrap();
        let adj = |s: &crate::Session| match &s.active().unwrap().doc.layer(id).unwrap().content {
            photocraft_doc::LayerContent::Adjustment(a) => a.clone(),
            _ => panic!(),
        };
        assert!(matches!(adj(&s), Adjustment::ColorBalance { midtones: [30.0, 0.0, 0.0], highlights: [0.0, 0.0, -20.0], .. }));
        s.execute("layer.setAdjustment", json!({"layer": id.0})).unwrap();
        assert_eq!(adj(&s), default_for("colorBalance", ColorMode::Rgb).unwrap());
        // Bad params are errors and leave the layer alone.
        s.execute("layer.setAdjustment", json!({"midtones": [10, 10, 10]})).unwrap();
        let kept = adj(&s);
        for bad in [json!({"midtones": "x"}), json!({"preserveLuminosity": 3}), json!({"shadows": [1]})] {
            assert!(s.execute("layer.setAdjustment", bad.clone()).is_err(), "{bad}");
            assert_eq!(adj(&s), kept);
        }
    }

    #[test]
    fn commands_reject_bad_params_gracefully() {
        let mut s = colourful("rgb", 8);
        for (kind, p) in [
            ("curves", json!({"points": [[0, 0]]})),
            ("levels", json!({"red": "dark"})),
            ("gradientMap", json!({"stops": [[0, 1, 2]]})),
            ("photoFilter", json!({"filter": 3})),
            ("channelMixer", json!({"gray": [1, 2]})),
            ("hueSaturation", json!({"reds": 4})),
            ("blackWhite", json!({"tint": "yes"})),
        ] {
            for prefix in ["image.adjustments.", "layer.newAdjustmentLayer."] {
                let r = s.execute(&format!("{prefix}{kind}"), p.clone());
                assert!(matches!(r, Err(EngineError::BadParams { .. })), "{prefix}{kind} {p}: {r:?}");
            }
        }
    }
    #[test]
    fn destructive_levels_match_independent_signed_hdr_endpoints() {
        // Input 15..230, gamma 1.3, output 10..245: black and white land on the output range in
        // integer documents. Float32 keeps HDR values outside 0..1, as the compositor's real-file
        // Levels oracle does. These independently calculated endpoints use the signed-power
        // curve at black and white, not display-clipped values.
        let params = json!({"inBlack": 15, "inWhite": 230, "gamma": 1.3, "outBlack": 10, "outWhite": 245});
        for (depth, low, high, tolerance) in [(8, 10.0, 245.0, 0.6), (16, 10.0, 245.0, 0.6), (32, -20.308_89, 265.750_3, 0.01)] {
            let mut s = crate::Session::new();
            s.execute("file.new", json!({"width": 16, "height": 16, "depth": depth, "background": "white"})).unwrap();
            s.execute("select.rect", json!({"x": 0, "y": 0, "width": 8, "height": 16})).unwrap();
            s.execute("edit.fill", json!({"color": "#000000"})).unwrap();
            s.execute("select.deselect", json!({})).unwrap();
            s.execute("image.adjustments.levels", params.clone()).unwrap();
            let px = pixels(&s);
            assert!((px[0][0] * 255.0 - low).abs() < tolerance, "black @{depth}: {:?}", px[0]);
            assert!((px[1][0] * 255.0 - high).abs() < tolerance, "white @{depth}: {:?}", px[1]);
            if depth == 32 {
                assert!(px[0][0] < 0.0 && px[1][0] > 1.0, "HDR endpoints must remain unclipped");
            }
        }
    }
}
