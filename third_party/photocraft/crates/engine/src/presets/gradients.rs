//! Window › Gradients: gradient presets in groups, the Gradient tool's current gradient and
//! Gradient Fill layers made from presets.
//!
//! A preset has colour stops (a fixed colour, or the live foreground/background colour, as in
//! Photoshop's "Foreground to Background") and transparency stops, both at `0..=1`. Midpoints are
//! always 50 %. Built-in colours are our own.

use photocraft_doc::{Color, Fill, Layer, LayerContent};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{Group, Named, always, bad, edit_groups, find, group_index, req_str, str_param, unique_name};
use crate::commands::CommandSpec;
use crate::{Result, Session};

/// A stop colour: fixed, or the current foreground/background colour.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StopColor {
    Foreground,
    Background,
    Rgb([f32; 3]),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GradientPreset {
    pub name: String,
    /// Colour stops `(location 0..1, colour)`, sorted by location.
    pub stops: Vec<(f32, StopColor)>,
    /// Transparency stops `(location, opacity 0..1)`; empty = opaque.
    #[serde(default)]
    pub opacity: Vec<(f32, f32)>,
}

impl Named for GradientPreset {
    fn name(&self) -> &str {
        &self.name
    }
    fn set_name(&mut self, n: String) {
        self.name = n;
    }
}

fn hex(s: &str) -> [f32; 3] {
    let s = s.trim_start_matches('#');
    let b = |i: usize| u8::from_str_radix(&s[i..i + 2], 16).map_or(0.0, |v| v as f32 / 255.0);
    [b(0), b(2), b(4)]
}

fn hex_of(c: [f32; 3]) -> String {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", q(c[0]), q(c[1]), q(c[2]))
}

fn lerp4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
}

/// Piecewise-linear value of sorted `(t, v)` stops at `t` (clamped at the ends).
fn sample<V: Copy>(stops: &[(f32, V)], t: f32, mix: impl Fn(V, V, f32) -> V) -> Option<V> {
    let first = stops.first()?;
    if t <= first.0 {
        return Some(first.1);
    }
    for w in stops.windows(2) {
        let ((t0, a), (t1, b)) = (w[0], w[1]);
        if t <= t1 {
            let u = if t1 > t0 { (t - t0) / (t1 - t0) } else { 1.0 };
            return Some(mix(a, b, u));
        }
    }
    stops.last().map(|s| s.1)
}

impl GradientPreset {
    pub fn new(name: &str, colors: &[&str]) -> Self {
        let n = colors.len().max(2) - 1;
        GradientPreset {
            name: name.into(),
            stops: colors.iter().enumerate().map(|(i, c)| (i as f32 / n as f32, StopColor::Rgb(hex(c)))).collect(),
            opacity: Vec::new(),
        }
    }
    pub fn foreground_to_background() -> Self {
        GradientPreset { name: "Foreground to Background".into(), stops: vec![(0.0, StopColor::Foreground), (1.0, StopColor::Background)], opacity: Vec::new() }
    }

    /// RGBA stops for painting: colour and transparency stops merged (see [`apply_opacity`]).
    pub fn resolve(&self, fg: [f32; 4], bg: [f32; 4]) -> Vec<(f32, [f32; 4])> {
        let rgb = |c: StopColor| match c {
            StopColor::Foreground => [fg[0], fg[1], fg[2], 1.0],
            StopColor::Background => [bg[0], bg[1], bg[2], 1.0],
            StopColor::Rgb(c) => [c[0], c[1], c[2], 1.0],
        };
        let mut cs: Vec<(f32, [f32; 4])> = self.stops.iter().map(|(t, c)| (t.clamp(0.0, 1.0), rgb(*c))).collect();
        cs.sort_by(|a, b| a.0.total_cmp(&b.0));
        if cs.is_empty() {
            cs = vec![(0.0, [fg[0], fg[1], fg[2], 1.0]), (1.0, [bg[0], bg[1], bg[2], 1.0])];
        }
        apply_opacity(cs, &self.opacity)
    }

    /// Colour stops (live colours resolved now, sorted, opaque) and opacity stops (sorted; empty
    /// when fully opaque) kept apart, as an editable Gradient Fill layer holds them.
    pub fn fill_stops(&self, fg: [f32; 4], bg: [f32; 4]) -> FillStops {
        let rgb = |c: StopColor| match c {
            StopColor::Foreground => Color::rgb(fg[0], fg[1], fg[2]),
            StopColor::Background => Color::rgb(bg[0], bg[1], bg[2]),
            StopColor::Rgb(c) => Color::rgb(c[0], c[1], c[2]),
        };
        let mut cs: Vec<(f32, Color)> = self.stops.iter().map(|(t, c)| (t.clamp(0.0, 1.0), rgb(*c))).collect();
        cs.sort_by(|a, b| a.0.total_cmp(&b.0));
        if cs.is_empty() {
            cs = vec![(0.0, rgb(StopColor::Foreground)), (1.0, rgb(StopColor::Background))];
        }
        let mut os: Vec<(f32, f32)> = self.opacity.iter().map(|(t, a)| (t.clamp(0.0, 1.0), a.clamp(0.0, 1.0))).collect();
        os.sort_by(|a, b| a.0.total_cmp(&b.0));
        if os.iter().all(|o| (o.1 - 1.0).abs() < 1e-6) {
            os.clear();
        }
        (cs, os)
    }

    /// Stops for a Gradient Fill layer / overlay (live colours resolved now, like Photoshop).
    pub fn doc_stops(&self, fg: [f32; 4], bg: [f32; 4]) -> Vec<(f32, Color)> {
        self.resolve(fg, bg).into_iter().map(|(t, c)| (t, Color::rgba(c[0], c[1], c[2], c[3]))).collect()
    }

    pub fn to_json(&self) -> Value {
        let stops: Vec<Value> = self
            .stops
            .iter()
            .map(|(t, c)| match c {
                StopColor::Foreground => json!([t, "foreground"]),
                StopColor::Background => json!([t, "background"]),
                StopColor::Rgb(c) => json!([t, hex_of(*c)]),
            })
            .collect();
        let opacity: Vec<Value> = self.opacity.iter().map(|(t, a)| json!([t, (a * 100.0).round()])).collect();
        json!({"name": self.name, "stops": stops, "transparency": opacity})
    }

    /// Parse `{"stops":[[t,"#hex"|"foreground"|"background"],…],"transparency":[[t,0..100],…]}`.
    pub fn from_params(name: &str, p: &Value, cmd: &str) -> Result<Self> {
        let arr = p.get("stops").and_then(Value::as_array).ok_or_else(|| bad(cmd, "missing `stops`"))?;
        let mut stops = Vec::new();
        for (i, s) in arr.iter().enumerate() {
            let (t, c) = match s {
                Value::Array(a) if a.len() == 2 => (a[0].as_f64().map(|v| v as f32), a[1].as_str()),
                Value::String(c) => (Some(if arr.len() > 1 { i as f32 / (arr.len() - 1) as f32 } else { 0.0 }), Some(c.as_str())),
                _ => (None, None),
            };
            let (Some(t), Some(c)) = (t, c) else { return Err(bad(cmd, "stops are [location, \"#rrggbb\"|\"foreground\"|\"background\"]")) };
            let c = match c {
                "foreground" => StopColor::Foreground,
                "background" => StopColor::Background,
                h if h.trim_start_matches('#').len() == 6 && h.trim_start_matches('#').chars().all(|c| c.is_ascii_hexdigit()) => StopColor::Rgb(hex(h)),
                o => return Err(bad(cmd, format!("bad stop colour `{o}`"))),
            };
            stops.push((t.clamp(0.0, 1.0), c));
        }
        if stops.len() < 2 {
            return Err(bad(cmd, "a gradient needs at least 2 colour stops"));
        }
        stops.sort_by(|a, b| a.0.total_cmp(&b.0));
        GradientPreset { name: name.into(), stops, opacity: Vec::new() }.with_transparency(p, cmd)
    }

    /// This gradient with its transparency stops replaced by explicit `"transparency"` stops,
    /// when `p` has them (see [`transparency_param`]).
    pub fn with_transparency(mut self, p: &Value, cmd: &str) -> Result<Self> {
        if let Some(opacity) = transparency_param(p, cmd)? {
            self.opacity = opacity;
        }
        Ok(self)
    }
}

/// Sorted RGBA colour stops `cs` with transparency stops `(location, opacity 0..1)` multiplied
/// into their alpha. Where both vary, extra samples keep the product close to the reference app's
/// per-pixel interpolation.
pub fn apply_opacity(cs: Stops, opacity: &[(f32, f32)]) -> Stops {
    let mut os: Vec<(f32, f32)> = opacity.iter().map(|(t, a)| (t.clamp(0.0, 1.0), a.clamp(0.0, 1.0))).collect();
    os.sort_by(|a, b| a.0.total_cmp(&b.0));
    if os.is_empty() || os.iter().all(|o| (o.1 - 1.0).abs() < 1e-6) {
        return cs;
    }
    let mut ts: Vec<f32> = cs.iter().map(|s| s.0).chain(os.iter().map(|s| s.0)).chain([0.0, 1.0]).collect();
    ts.sort_by(f32::total_cmp);
    ts.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    let mut all = Vec::new();
    for w in ts.windows(2) {
        for k in 0..4 {
            all.push(w[0] + (w[1] - w[0]) * k as f32 / 4.0);
        }
    }
    all.push(1.0);
    all.into_iter()
        .map(|t| {
            let mut c = sample(&cs, t, lerp4).unwrap_or([0.0; 4]);
            c[3] *= sample(&os, t, |a, b, u| a + (b - a) * u).unwrap_or(1.0);
            (t, c)
        })
        .collect()
}

/// Explicit `"transparency":[[location 0..1, opacity 0..100], …]` as sorted `(location,
/// opacity 0..1)` stops (`[]` = opaque); `None` when `p` has none.
pub fn transparency_param(p: &Value, cmd: &str) -> Result<Option<Vec<(f32, f32)>>> {
    let Some(v) = p.get("transparency") else { return Ok(None) };
    let arr = v.as_array().ok_or_else(|| bad(cmd, "`transparency` is [[location, opacity %], …]"))?;
    if arr.len() > 1024 {
        return Err(bad(cmd, "too many opacity stops (max 1024)"));
    }
    let mut stops = Vec::with_capacity(arr.len());
    for s in arr {
        let pair = s.as_array().filter(|a| a.len() == 2).ok_or_else(|| bad(cmd, "an opacity stop is [location 0..1, opacity 0..100]"))?;
        let f = |i: usize| pair.get(i).and_then(Value::as_f64).map(|v| v as f32).filter(|v| v.is_finite());
        let (Some(t), Some(o)) = (f(0), f(1)) else { return Err(bad(cmd, "opacity stops are numbers")) };
        stops.push((t.clamp(0.0, 1.0), (o / 100.0).clamp(0.0, 1.0)));
    }
    stops.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(Some(stops))
}

/// Built-in groups, laid out like Photoshop's default Gradients panel (Basics, then folders by
/// hue). The colours are our own.
pub fn builtin() -> Vec<Group<GradientPreset>> {
    let fam = |prefix: &str, sets: &[&[&str]]| -> Vec<GradientPreset> {
        sets.iter().enumerate().map(|(i, c)| GradientPreset::new(&format!("{prefix} {:02}", i + 1), c)).collect()
    };
    let fg_transparent = GradientPreset {
        name: "Foreground to Transparent".into(),
        stops: vec![(0.0, StopColor::Foreground), (1.0, StopColor::Foreground)],
        opacity: vec![(0.0, 1.0), (1.0, 0.0)],
    };
    // A smooth spectrum: 13 hues evenly spaced from red (0°) to magenta (300°). The pastel and
    // neon presets reuse the same hues at a different saturation/lightness, and Rainbow Fade
    // reuses the classic ramp with its ends transparent. Colours are our own.
    let rainbow = GradientPreset::new(
        "Rainbow",
        &["#ec1313", "#ec6d13", "#ecc813", "#b6ec13", "#5bec13", "#13ec25", "#13ec80", "#13ecda", "#13a4ec", "#1349ec", "#3713ec", "#9213ec", "#ec13ec"],
    );
    vec![
        Group::new("Basics", vec![GradientPreset::foreground_to_background(), fg_transparent, GradientPreset::new("Black, White", &["#000000", "#ffffff"])]),
        Group::new(
            "Blues",
            fam(
                "Blue",
                &[
                    &["#0b3d91", "#4fa3e0"],
                    &["#1c2a5a", "#3f6fd8", "#a8d4ff"],
                    &["#00b4d8", "#03045e"],
                    &["#5ec8f2", "#e6f6ff"],
                    &["#253b80", "#20b2aa"],
                    &["#89c2ff", "#1d4ed8", "#0a1a40"],
                ],
            ),
        ),
        Group::new(
            "Purples",
            fam(
                "Purple",
                &[
                    &["#3c096c", "#c77dff"],
                    &["#5a189a", "#e0aaff"],
                    &["#240046", "#7b2cbf", "#f3d9ff"],
                    &["#9d4edd", "#4361ee"],
                    &["#6a0572", "#ab83a1"],
                    &["#2d0b59", "#ff6fd8"],
                ],
            ),
        ),
        Group::new(
            "Pinks",
            fam(
                "Pink",
                &[
                    &["#ff4d8d", "#ffd1e1"],
                    &["#c9184a", "#ff8fab"],
                    &["#ff70a6", "#ffd670"],
                    &["#8e2c5a", "#f7a1c4", "#fff0f5"],
                    &["#ff5fa2", "#a855f7"],
                    &["#ffc2d6", "#ff2e63"],
                ],
            ),
        ),
        Group::new(
            "Reds",
            fam(
                "Red",
                &[
                    &["#7f0000", "#ff3b30"],
                    &["#d00000", "#ffba08"],
                    &["#370617", "#9d0208", "#f48c06"],
                    &["#e5383b", "#f5f3f4"],
                    &["#a4161a", "#0b090a"],
                    &["#ff595e", "#ffca3a"],
                ],
            ),
        ),
        Group::new(
            "Oranges",
            fam(
                "Orange",
                &[
                    &["#ff6d00", "#ffd60a"],
                    &["#e85d04", "#ffba08"],
                    &["#9c3d0b", "#ff9e40", "#fff1d6"],
                    &["#ff7b00", "#ff0054"],
                    &["#fb8500", "#023047"],
                    &["#ffb703", "#fb5607"],
                ],
            ),
        ),
        Group::new(
            "Greens",
            fam(
                "Green",
                &[
                    &["#004b23", "#70e000"],
                    &["#2d6a4f", "#b7e4c7"],
                    &["#1b4332", "#40916c", "#d8f3dc"],
                    &["#38b000", "#ccff33"],
                    &["#0b6e4f", "#08a0a8"],
                    &["#606c38", "#dda15e"],
                ],
            ),
        ),
        Group::new(
            "Browns",
            fam(
                "Brown",
                &[
                    &["#3e2412", "#a47148"],
                    &["#6f4518", "#e6ccb2"],
                    &["#2b1708", "#7f5539", "#ddb892"],
                    &["#8b5e34", "#ffd8a8"],
                    &["#582f0e", "#b6ad90"],
                    &["#4a3728", "#c08552", "#f3e9dc"],
                ],
            ),
        ),
        Group::new(
            "Grays",
            fam(
                "Gray",
                &[
                    &["#212529", "#f8f9fa"],
                    &["#495057", "#dee2e6"],
                    &["#000000", "#6c757d", "#ffffff"],
                    &["#343a40", "#adb5bd", "#343a40"],
                    &["#8d99ae", "#2b2d42"],
                    &["#e9ecef", "#ced4da", "#868e96"],
                ],
            ),
        ),
        Group::new(
            "Rainbows",
            vec![
                rainbow.clone(),
                GradientPreset::new(
                    "Pastel Rainbow",
                    &[
                        "#f4afaf", "#f4cbaf", "#f4e8af", "#e2f4af", "#c6f4af", "#aff4b4", "#aff4d1", "#aff4ee", "#afddf4", "#afc0f4", "#baaff4", "#d7aff4",
                        "#f4aff4",
                    ],
                ),
                GradientPreset::new(
                    "Neon Rainbow",
                    &[
                        "#ff0000", "#ff6a00", "#ffd400", "#bfff00", "#55ff00", "#00ff15", "#00ff80", "#00ffea", "#00aaff", "#0040ff", "#2a00ff", "#9500ff",
                        "#ff00ff",
                    ],
                ),
                GradientPreset::new(
                    "Oil Slick",
                    &["#0b0f1a", "#1b2a4a", "#0f6f6f", "#1fae8a", "#5ac8a8", "#7a5cff", "#b13bd6", "#e04a8f", "#f2a65a", "#3a2a6b", "#0b0f1a"],
                ),
                GradientPreset { name: "Rainbow Fade".into(), opacity: vec![(0.0, 0.0), (0.125, 1.0), (0.875, 1.0), (1.0, 0.0)], ..rainbow },
            ],
        ),
    ]
}

// ------------------------------------------------------------------ commands

/// RGBA stops `(location, colour)` ready to paint.
pub type Stops = Vec<(f32, [f32; 4])>;

/// Colour stops and opacity stops of a Gradient Fill layer.
pub type FillStops = (Vec<(f32, Color)>, Vec<(f32, f32)>);

const CMD_SELECT: &str = "gradient.presets.select";

/// The Gradient tool's current gradient, as a one-stop edit target (`gradient.presets.stop`).
pub const STOP: &str = "gradient.presets.stop";

/// Gradient for the Gradient tool: `"gradient": preset name`, else explicit `"stops"`, else the
/// current gradient (Gradients panel selection); explicit `"transparency"` stops replace its
/// own. `None` when the caller gave `colors` (the legacy evenly spaced form `paint.gradient`
/// handles itself).
pub fn tool_stops(s: &Session, p: &Value) -> Result<Option<Stops>> {
    const CMD: &str = "paint.gradient";
    if p.get("colors").and_then(Value::as_array).is_some_and(|a| !a.is_empty()) {
        return Ok(None);
    }
    let g = if let Some(name) = str_param(p, "gradient") {
        let (gi, ii) = find(&s.presets.gradients, name, None).ok_or_else(|| bad(CMD, format!("no gradient preset \"{name}\"")))?;
        s.presets.gradients[gi].items[ii].clone().with_transparency(p, CMD)?
    } else if p.get("stops").is_some() {
        GradientPreset::from_params("Custom", p, CMD)?
    } else {
        s.presets.gradient.clone().with_transparency(p, CMD)?
    };
    Ok(Some(g.resolve(s.tools.foreground, s.tools.background)))
}

/// The preset named by `"preset"` (else the current gradient), or explicit `"stops"`.
fn preset_param(s: &Session, p: &Value, cmd: &str) -> Result<GradientPreset> {
    if let Some(name) = str_param(p, "preset") {
        let (gi, ii) = find(&s.presets.gradients, name, str_param(p, "group"))
            .ok_or_else(|| bad(cmd, format!("no gradient preset \"{name}\" (see gradient.presets.list)")))?;
        return Ok(s.presets.gradients[gi].items[ii].clone());
    }
    if p.get("stops").is_some() {
        return GradientPreset::from_params(str_param(p, "name").unwrap_or("Custom"), p, cmd);
    }
    Ok(s.presets.gradient.clone())
}

fn list(s: &mut Session, _: &Value) -> Result<Value> {
    let groups: Vec<Value> =
        s.presets.gradients.iter().map(|g| json!({"name": g.name, "presets": g.items.iter().map(GradientPreset::to_json).collect::<Vec<_>>()})).collect();
    Ok(json!({"groups": groups, "current": s.presets.gradient.to_json()}))
}

fn active_gradient_fill(s: &Session) -> Option<photocraft_doc::LayerId> {
    let d = s.active()?;
    let id = d.active_layer?;
    matches!(d.doc.layer(id)?.content, LayerContent::Fill(Fill::Gradient { .. })).then_some(id)
}

fn select(s: &mut Session, p: &Value) -> Result<Value> {
    let g = preset_param(s, p, CMD_SELECT)?;
    s.presets.gradient = g.clone();
    s.presets_changed();
    let mut out = json!({"current": g.to_json()});
    // Clicking a preset with a Gradient Fill layer selected changes that layer (Photoshop).
    if p.get("applyToLayer").and_then(Value::as_bool).unwrap_or(true)
        && let Some(id) = active_gradient_fill(s)
    {
        let (stops_new, opacity_new) = g.fill_stops(s.tools.foreground, s.tools.background);
        s.edit("Change Gradient Fill", |doc, _| {
            let mode = doc.mode;
            if let Some(l) = doc.layer_mut(id)
                && let LayerContent::Fill(Fill::Gradient { stops, opacity_stops, midpoints, .. }) = &mut l.content
            {
                *stops = stops_new.into_iter().map(|(t, c)| (t, c.in_mode(mode))).collect();
                *opacity_stops = opacity_new;
                midpoints.clear();
            }
            Ok(())
        })?;
        out["layer"] = json!(id.0);
    }
    Ok(out)
}

/// A preset stop colour: `"#rrggbb"`, `"foreground"` or `"background"`.
fn stop_color_param(v: &Value, cmd: &str) -> Result<StopColor> {
    let s = v.as_str().ok_or_else(|| bad(cmd, "a stop colour is \"#rrggbb\", \"foreground\" or \"background\""))?;
    match s {
        "foreground" => Ok(StopColor::Foreground),
        "background" => Ok(StopColor::Background),
        h if h.trim_start_matches('#').len() == 6 && h.trim_start_matches('#').chars().all(|c| c.is_ascii_hexdigit()) => Ok(StopColor::Rgb(hex(h))),
        o => Err(bad(cmd, format!("bad stop colour `{o}`"))),
    }
}

/// Applies a `gradient.presets.stop` edit to the Gradient tool's current gradient: the
/// `gradient.fill.stop` grammar (add|move|delete|color and the opacity kind) without a `layer`.
/// Midpoints are rejected — presets keep them at the centre; edit a Gradient Fill layer to move
/// them. Pure, so the editor previews a drag with exactly what the command commits.
pub fn apply_preset_stop(g: &GradientPreset, p: &Value, fg: [f32; 4], bg: [f32; 4], cmd: &str) -> Result<GradientPreset> {
    let mut out = g.clone();
    let action = p.get("action").and_then(Value::as_str).ok_or_else(|| bad(cmd, "missing `action` (add|move|delete|color|opacity)"))?;
    if action == "midpoint" {
        return Err(bad(cmd, "preset gradients keep midpoints at the centre (edit a Gradient Fill layer to move them)"));
    }
    let opacity_kind = match p.get("kind").and_then(Value::as_str).unwrap_or("color") {
        "color" => false,
        "opacity" => true,
        o => return Err(bad(cmd, format!("unknown kind `{o}` (color|opacity)"))),
    };
    let index = || -> Result<usize> {
        p.get("index").and_then(Value::as_u64).and_then(|v| usize::try_from(v).ok()).ok_or_else(|| bad(cmd, "missing `index` (a stop number from 0)"))
    };
    let location = || {
        p.get("location")
            .and_then(Value::as_f64)
            .map(|v| v as f32)
            .filter(|v| v.is_finite())
            .map(|v| v.clamp(0.0, 1.0))
            .ok_or_else(|| bad(cmd, "missing `location` (0..1)"))
    };
    let ramp = g.resolve(fg, bg);
    if opacity_kind {
        // Opacity stops: an empty list means "opaque" (two 100 % stops when first edited).
        let mut os = if out.opacity.is_empty() { vec![(0.0, 1.0), (1.0, 1.0)] } else { out.opacity.clone() };
        os.sort_by(|a, b| a.0.total_cmp(&b.0));
        let opacity =
            || p.get("opacity").and_then(Value::as_f64).map(|v| (v as f32 / 100.0).clamp(0.0, 1.0)).ok_or_else(|| bad(cmd, "missing `opacity` (0..100)"));
        match action {
            "add" => {
                let t = location()?;
                let a = match p.get("opacity") {
                    Some(_) => opacity()?,
                    None => sample(&ramp, t, lerp4).map_or(1.0, |c| c[3]),
                };
                os.push((t, a));
            }
            "move" => {
                let (i, t) = (index()?, location()?);
                os.get_mut(i).ok_or_else(|| bad(cmd, format!("no opacity stop {i}")))?.0 = t;
            }
            "delete" => {
                let i = index()?;
                if i >= os.len() {
                    return Err(bad(cmd, format!("no opacity stop {i}")));
                }
                if os.len() <= 2 {
                    return Err(bad(cmd, "a gradient keeps at least 2 opacity stops"));
                }
                os.remove(i);
            }
            "opacity" => {
                let (i, a) = (index()?, opacity()?);
                os.get_mut(i).ok_or_else(|| bad(cmd, format!("no opacity stop {i}")))?.1 = a;
            }
            o => return Err(bad(cmd, format!("unknown action `{o}` for opacity stops (add|move|delete|opacity)"))),
        }
        os.sort_by(|a, b| a.0.total_cmp(&b.0));
        if os.len() > 1024 {
            return Err(bad(cmd, "too many opacity stops (max 1024)"));
        }
        // Fully opaque again (the materialized ends left after edits, say): store as no opacity
        // stops, like `fill_stops` reads them, instead of pinning explicit 100 % ends.
        if os.iter().all(|o| (o.1 - 1.0).abs() < 1e-6) {
            os.clear();
        }
        out.opacity = os;
        return Ok(out);
    }
    let mut stops = out.stops.clone();
    stops.sort_by(|a, b| a.0.total_cmp(&b.0));
    match action {
        "add" => {
            let t = location()?;
            // A new stop takes the colour the gradient has there (Photoshop), unless given.
            let c = match p.get("color") {
                Some(v) => stop_color_param(v, cmd)?,
                None => sample(&ramp, t, lerp4).map_or(StopColor::Rgb([0.0, 0.0, 0.0]), |c| StopColor::Rgb([c[0], c[1], c[2]])),
            };
            stops.push((t, c));
            if stops.len() > 1024 {
                return Err(bad(cmd, "too many colour stops (max 1024)"));
            }
        }
        "move" => {
            let (i, t) = (index()?, location()?);
            stops.get_mut(i).ok_or_else(|| bad(cmd, format!("no colour stop {i}")))?.0 = t;
        }
        "delete" => {
            let i = index()?;
            if i >= stops.len() {
                return Err(bad(cmd, format!("no colour stop {i}")));
            }
            if stops.len() <= 2 {
                return Err(bad(cmd, "a gradient keeps at least 2 colour stops"));
            }
            stops.remove(i);
        }
        "color" => {
            let (i, c) = (index()?, p.get("color").ok_or_else(|| bad(cmd, "missing `color`")).and_then(|v| stop_color_param(v, cmd))?);
            stops.get_mut(i).ok_or_else(|| bad(cmd, format!("no colour stop {i}")))?.1 = c;
        }
        o => return Err(bad(cmd, format!("unknown action `{o}` (add|move|delete|color)"))),
    }
    stops.sort_by(|a, b| a.0.total_cmp(&b.0));
    out.stops = stops;
    Ok(out)
}

fn stop(s: &mut Session, p: &Value) -> Result<Value> {
    let g = apply_preset_stop(&s.presets.gradient, p, s.tools.foreground, s.tools.background, STOP)?;
    s.presets.gradient = g;
    s.presets_changed();
    Ok(json!({"current": s.presets.gradient.to_json()}))
}

fn apply(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "gradient.presets.apply";
    let g = preset_param(s, p, CMD)?;
    let stops = g.doc_stops(s.tools.foreground, s.tools.background);
    let angle = p.get("angle").and_then(Value::as_f64).unwrap_or(90.0) as f32;
    let scale = (p.get("scale").and_then(Value::as_f64).unwrap_or(100.0) as f32 / 100.0).clamp(0.1, 1.5);
    let style = crate::layer_style::gradient_style(p.get("style").and_then(Value::as_str).unwrap_or("linear"));
    let reverse = p.get("reverse").and_then(Value::as_bool).unwrap_or(false);
    let id = s.edit("New Gradient Fill Layer", |doc, active| {
        let fill = Fill::gradient(stops, angle, scale, style, reverse).in_mode(doc.mode);
        let id = doc.insert_above(*active, Layer::new(doc.next_layer_name("Gradient Fill"), LayerContent::Fill(fill)));
        *active = Some(id);
        Ok(id)
    })?;
    Ok(json!({"layer": id.0, "preset": g.name}))
}

fn new_preset(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "gradient.presets.new";
    let mut g = if p.get("stops").is_some() { GradientPreset::from_params("", p, CMD)? } else { s.presets.gradient.clone() };
    g.name = unique_name(&s.presets.gradients, str_param(p, "name").unwrap_or("Custom"));
    let gi = group_index(&mut s.presets.gradients, str_param(p, "group"));
    s.presets.gradients[gi].items.push(g.clone());
    let group = s.presets.gradients[gi].name.clone();
    s.presets_changed();
    Ok(json!({"name": g.name, "group": group}))
}

fn edit(s: &mut Session, p: &Value) -> Result<Value> {
    let action = req_str(p, "action", "gradient.presets.edit")?.to_string();
    let r = edit_groups(&mut s.presets.gradients, &action, p, "gradient.presets.edit")?;
    s.presets_changed();
    Ok(r)
}

fn reset(s: &mut Session, p: &Value) -> Result<Value> {
    let append = p.get("append").and_then(Value::as_bool).unwrap_or(false);
    let mut b = builtin();
    if append {
        for g in &mut b {
            g.name = unique_group(&s.presets.gradients, &g.name);
        }
        s.presets.gradients.extend(b);
    } else {
        s.presets.gradients = b;
    }
    s.presets_changed();
    Ok(json!({"groups": s.presets.gradients.len()}))
}

pub(crate) fn unique_group<T>(groups: &[Group<T>], base: &str) -> String {
    let taken = |n: &str| groups.iter().any(|g| g.name == n);
    if !taken(base) { base.to_string() } else { (1..).map(|i| format!("{base} {i}")).find(|n| !taken(n)).unwrap_or_default() }
}

pub fn specs() -> Vec<CommandSpec> {
    let stops = r##""stops":[[t 0..1,"#rrggbb"|"foreground"|"background"],…],"transparency":[[t,opacity 0..100],…]?"##;
    let leak = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
    vec![
        CommandSpec {
            id: "gradient.presets.list",
            label: "Gradient Presets",
            menu: &[],
            shortcut: None,
            params: "{} → {groups:[{name,presets:[{name,stops,transparency}]}],current}",
            enabled: always,
            run: list,
            journal: false,
        },
        CommandSpec {
            id: CMD_SELECT,
            label: "Select Gradient",
            menu: &[],
            shortcut: None,
            params: leak(format!(
                r##"{{"preset":name|{stops},"group":name?,"applyToLayer":bool=true (also recolours a selected Gradient Fill layer)}} → {{current,layer?}}. The Gradient tool paints with it."##
            )),
            enabled: always,
            run: select,
            journal: true,
        },
        CommandSpec {
            id: STOP,
            label: "Edit Gradient Stop",
            menu: &[],
            shortcut: None,
            params: r##"{"action":"add|move|delete|color|opacity","kind":"color|opacity"="color","index":stop number,"location":0..1,"color":"#rrggbb"|"foreground"|"background" (add: default the colour there),"opacity":0..100} → {"current"}. Edits the Gradient tool's current gradient (no layer, midpoints stay centred)."##,
            enabled: always,
            run: stop,
            journal: true,
        },
        CommandSpec {
            id: "gradient.presets.apply",
            label: "New Gradient Fill Layer from Preset",
            menu: &[],
            shortcut: None,
            params: leak(format!(
                r##"{{"preset":name?=current|{stops},"angle":deg=90,"style":"linear|radial|angle|reflected|diamond"="linear","scale":10..150=100,"reverse":bool}} → {{layer}}"##
            )),
            enabled: super::has_doc,
            run: apply,
            journal: true,
        },
        CommandSpec {
            id: "gradient.presets.new",
            label: "New Gradient Preset",
            menu: &[],
            shortcut: None,
            params: leak(format!(r##"{{"name":str="Custom","group":name?=first,{stops} (default: the current gradient)}}"##)),
            enabled: always,
            run: new_preset,
            journal: true,
        },
        CommandSpec {
            id: "gradient.presets.edit",
            label: "Edit Gradient Presets",
            menu: &[],
            shortcut: None,
            params: super::GROUP_EDIT_PARAMS,
            enabled: always,
            run: edit,
            journal: true,
        },
        CommandSpec {
            id: "gradient.presets.reset",
            label: "Restore Default Gradients",
            menu: &[],
            shortcut: None,
            params: r##"{"append":bool=false}"##,
            enabled: always,
            run: reset,
            journal: true,
        },
    ]
}
