//! Photoshop brushes (`.abr`) → brush presets.
//!
//! [`photocraft_psd::abr`] reads the file; this module maps its tips and `brushPreset`
//! descriptors onto [`BrushSettings`] (Brush Tip Shape, Shape Dynamics, Scattering, Texture,
//! Dual Brush, Color Dynamics, Transfer, Brush Pose, Noise, Wet Edges, Build-up, Smoothing and
//! the captured tool options). Settings without an equivalent are listed in
//! [`AbrImport::warnings`] instead of being dropped silently. Descriptor key names follow the
//! publicly documented ABR v6+ layout.

use std::collections::BTreeSet;

use photocraft_paint::{
    BrushPreset, BrushSettings, Control, DualBrush, Dynamic, GrayTile, MaskMode, MixerSettings, Pattern, PatternStyle, Pose, ShapeDynamics, TipShape, Transfer,
};
use photocraft_psd::abr::{self, AbrFile, AbrSample, LegacyTip};
use photocraft_psd::descriptor::{Descriptor, Value};
use photocraft_psd::patterns::PsdPattern;

/// Largest sampled tip edge kept as is; bigger tips are downsampled to it.
pub const MAX_TIP_EDGE: u32 = 2500;
/// Largest texture pattern edge kept as is; bigger patterns are downsampled to it.
pub const MAX_PATTERN_EDGE: u32 = 1024;

/// The presets read from one file plus everything that could not be mapped.
#[derive(Clone, Debug, Default)]
pub struct AbrImport {
    pub presets: Vec<BrushPreset>,
    pub warnings: Vec<String>,
    /// File version (1, 2, 6…).
    pub version: u16,
}

/// Parse an `.abr` file into presets in group `group` (usually the file name). The file's own
/// folders (Photoshop CC's preset hierarchy) become each preset's [`BrushPreset::folder`] inside
/// that group, the way Photoshop's Import Brushes puts them in a folder named after the file.
pub fn read_abr(bytes: &[u8], group: &str) -> Result<AbrImport, String> {
    read_abr_with(bytes, group, &photocraft_raster::Interrupt::NONE)
}

/// [`read_abr`] that checks `ctl` before each brush (and reports progress): a cancelled import
/// fails with "cancelled".
pub fn read_abr_with(bytes: &[u8], group: &str, ctl: &photocraft_raster::Interrupt) -> Result<AbrImport, String> {
    let f = abr::parse(bytes).map_err(|e| format!("not a readable brush file (.abr): {e}"))?;
    // Parsing is the small part; mapping (tip decoding) reports 10–100 %.
    ctl.progress(0.1);
    let cancel = || ctl.cancelled();
    let progress = |p: f32| ctl.progress(0.1 + 0.9 * p);
    map_file_with(&f, group, &photocraft_raster::Interrupt::new(&cancel, &progress)).map_err(|e| e.to_string())
}

/// Map a parsed file.
pub fn map_file(f: &AbrFile, group: &str) -> AbrImport {
    // Never cancelled, so always `Ok`.
    map_file_with(f, group, &photocraft_raster::Interrupt::NONE).unwrap_or_default()
}

/// [`map_file`] that checks `ctl` before each brush.
pub fn map_file_with(f: &AbrFile, group: &str, ctl: &photocraft_raster::Interrupt) -> Result<AbrImport, photocraft_raster::Cancelled> {
    let mut out = AbrImport { version: f.version, warnings: f.warnings.clone(), ..Default::default() };
    let mut m = Mapper { file: f, warnings: BTreeSet::new(), unknown: BTreeSet::new() };
    let total = f.legacy.len().max(f.samples.len()).max(f.presets.len()).max(1) as f32;
    let step = |i: usize| -> Result<(), photocraft_raster::Cancelled> {
        ctl.check()?;
        ctl.progress(i as f32 / total);
        Ok(())
    };
    if !f.legacy.is_empty() {
        for (i, b) in f.legacy.iter().enumerate() {
            step(i)?;
            let spacing = if b.spacing == 0 { 0.25 } else { f32::from(b.spacing) / 100.0 };
            let base = BrushSettings { pressure_size: false, spacing, ..Default::default() };
            let (brush, default_name) = match &b.tip {
                LegacyTip::Computed { diameter, hardness, angle, roundness } => (
                    BrushSettings {
                        size: f32::from((*diameter).max(1)),
                        hardness: (f32::from(*hardness) / 100.0).clamp(0.0, 1.0),
                        angle: f32::from(*angle),
                        roundness: (f32::from(*roundness) / 100.0).clamp(0.01, 1.0),
                        ..base
                    },
                    format!("{} px Round", diameter),
                ),
                LegacyTip::Sampled(s) => {
                    let (tip, size) = m.tip(s);
                    (BrushSettings { size, tip, aliased: !b.anti_alias, ..base }, format!("Sampled Brush {}", i + 1))
                }
            };
            let name = if b.name.trim().is_empty() { default_name } else { b.name.trim().to_string() };
            out.presets.push(BrushPreset { name, brush, builtin: false, group: group.to_string(), folder: Vec::new(), captures: Default::default() });
        }
    } else if f.presets.is_empty() {
        // Tips without a settings section (early v6 files): one preset per tip.
        for (i, s) in f.samples.iter().enumerate() {
            step(i)?;
            let (tip, size) = m.tip(s);
            let brush = BrushSettings { pressure_size: false, spacing: 0.25, size, tip, ..Default::default() };
            out.presets.push(BrushPreset {
                name: format!("Sampled Brush {}", i + 1),
                brush,
                builtin: false,
                group: group.to_string(),
                folder: Vec::new(),
                captures: Default::default(),
            });
        }
    } else {
        for (i, d) in f.presets.iter().enumerate() {
            step(i)?;
            let name = text(d, "Nm  ").filter(|n| !n.trim().is_empty()).map(|n| n.trim().to_string()).unwrap_or_else(|| format!("Brush {}", i + 1));
            match m.preset(d) {
                // The file's folders nest inside `group`, as Photoshop nests them inside a folder
                // named after the imported file.
                Some(brush) => {
                    let folder = f.folders.get(i).cloned().unwrap_or_default();
                    out.presets.push(BrushPreset { name, brush, builtin: false, group: group.to_string(), folder, captures: Default::default() });
                }
                None => {
                    m.warnings.insert(format!("\"{name}\": its sampled tip is missing from the file; skipped"));
                }
            }
        }
    }
    out.warnings.extend(m.warnings);
    if !m.unknown.is_empty() {
        out.warnings.push(format!("settings without a PhotoCraft equivalent were ignored: {}", m.unknown.into_iter().collect::<Vec<_>>().join(", ")));
    }
    ctl.progress(1.0);
    Ok(out)
}

// ------------------------------------------------------------------ descriptor helpers

fn get<'a>(d: &'a Descriptor, k: &str) -> Option<&'a Value> {
    d.get(k)
}
fn num(d: &Descriptor, k: &str) -> Option<f64> {
    match get(d, k)? {
        Value::Double(v) => Some(*v),
        Value::UnitFloat { value, .. } => Some(*value),
        Value::Integer(v) => Some(f64::from(*v)),
        Value::LargeInteger(v) => Some(*v as f64),
        _ => None,
    }
    .filter(|v| v.is_finite())
}
/// A percentage as a 0..1 fraction.
fn pct(d: &Descriptor, k: &str) -> Option<f32> {
    num(d, k).map(|v| (v / 100.0) as f32)
}
fn boolean(d: &Descriptor, k: &str) -> Option<bool> {
    match get(d, k)? {
        Value::Boolean(b) => Some(*b),
        _ => None,
    }
}
fn text(d: &Descriptor, k: &str) -> Option<String> {
    match get(d, k)? {
        Value::Text(t) => Some(t.to_string_lossy()),
        _ => None,
    }
}
fn obj<'a>(d: &'a Descriptor, k: &str) -> Option<&'a Descriptor> {
    match get(d, k)? {
        Value::Descriptor(o) | Value::GlobalObject(o) => Some(o),
        _ => None,
    }
}
fn enum_value(d: &Descriptor, k: &str) -> Option<String> {
    match get(d, k)? {
        Value::Enumerated { value, .. } => Some(String::from_utf8_lossy(value.as_bytes()).to_string()),
        _ => None,
    }
}

fn mask_mode(code: Option<String>) -> Option<MaskMode> {
    Some(match code?.as_str() {
        "Mltp" => MaskMode::Multiply,
        "Sbtr" => MaskMode::Subtract,
        "Drkn" => MaskMode::Darken,
        "Ovrl" => MaskMode::Overlay,
        "CDdg" => MaskMode::ColorDodge,
        "CBrn" => MaskMode::ColorBurn,
        "linearBurn" => MaskMode::LinearBurn,
        "hardMix" => MaskMode::HardMix,
        "linearHeight" => MaskMode::LinearHeight,
        "height" | "Hght" => MaskMode::Height,
        _ => return None,
    })
}

/// Preset-level keys this mapper understands (anything else is reported as unmapped).
const KNOWN: &[&str] = &[
    "Nm  ",
    "Brsh",
    "useTipDynamics",
    "flipX",
    "flipY",
    "brushProjection",
    "minimumDiameter",
    "minimumRoundness",
    "tiltScale",
    "szVr",
    "angleDynamics",
    "roundnessDynamics",
    "useScatter",
    "Spcn",
    "Cnt ",
    "bothAxes",
    "countDynamics",
    "scatterDynamics",
    "dualBrush",
    "brushGroup",
    "useTexture",
    "TxtC",
    "interpretation",
    "textureBlendMode",
    "textureDepth",
    "minimumDepth",
    "textureDepthDynamics",
    "Txtr",
    "textureScale",
    "InvT",
    "protectTexture",
    "textureBrightness",
    "textureContrast",
    "usePaintDynamics",
    "prVr",
    "opVr",
    "wtVr",
    "mxVr",
    "useColorDynamics",
    "clVr",
    "H   ",
    "Strt",
    "Brgh",
    "purity",
    "colorDynamicsPerTip",
    "Wtdg",
    "Nose",
    "Rpt ",
    "useLegacy",
    "useBrushPose",
    "overridePoseAngle",
    "overridePoseTiltX",
    "overridePoseTiltY",
    "overridePosePressure",
    "brushPosePressure",
    "brushPoseTiltX",
    "brushPoseTiltY",
    "brushPoseAngle",
    "toolOptions",
];

struct Mapper<'a> {
    file: &'a AbrFile,
    warnings: BTreeSet<String>,
    unknown: BTreeSet<String>,
}

impl Mapper<'_> {
    /// A sampled tip (downsampled to [`MAX_TIP_EDGE`]) and its size (larger side).
    fn tip(&mut self, s: &AbrSample) -> (TipShape, f32) {
        let v = s.to_unit();
        let (w, h, v) = fit(s.width, s.height, v, MAX_TIP_EDGE);
        if (w, h) != (s.width, s.height) {
            self.warnings.insert(format!("tips larger than {MAX_TIP_EDGE} px were downsampled"));
        }
        let data = v.iter().map(|x| (x.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16).collect();
        (TipShape::Sampled(GrayTile { width: w, height: h, data }), s.width.max(s.height) as f32)
    }

    fn dynamic(&mut self, d: Option<&Descriptor>) -> Dynamic {
        let Some(d) = d else { return Dynamic::default() };
        let control = match num(d, "bVTy").unwrap_or(0.0) as i64 {
            1 => Control::Fade,
            2 => Control::PenPressure,
            3 => Control::PenTilt,
            4 => Control::StylusWheel,
            5 => Control::InitialDirection,
            6 => Control::Direction,
            7 => {
                self.warnings.insert("\"Initial Rotation\" control mapped to Rotation".into());
                Control::Rotation
            }
            8 => Control::Rotation,
            _ => Control::Off,
        };
        Dynamic {
            jitter: pct(d, "jitter").unwrap_or(0.0).max(0.0),
            control,
            fade_steps: num(d, "fStp").map_or(25, |v| v.clamp(1.0, 9999.0) as u32),
            minimum: pct(d, "Mnm ").unwrap_or(0.0).clamp(0.0, 1.0),
        }
    }

    /// Tip-shape fields from a `computedBrush` / `sampledBrush` object. `None` when the sampled
    /// tip it references is missing.
    fn tip_shape(&mut self, t: &Descriptor) -> Option<(TipShape, Option<f32>)> {
        let sampled = t.class_id.is("sampledBrush") || get(t, "sampledData").is_some();
        if !sampled {
            return Some((TipShape::Round, None));
        }
        let id = text(t, "sampledData").unwrap_or_default();
        let s = self.file.samples.iter().find(|s| s.id == id).or_else(|| self.file.samples.first().filter(|_| self.file.samples.len() == 1))?;
        let (tip, size) = self.tip(s);
        Some((tip, Some(size)))
    }

    fn pattern(&mut self, t: &Descriptor) -> Pattern {
        let id = text(t, "Idnt").unwrap_or_default();
        let name = text(t, "Nm  ").unwrap_or_default();
        let found =
            self.file.patterns.iter().find(|p| !id.is_empty() && p.id == id).or_else(|| self.file.patterns.iter().find(|p| !name.is_empty() && p.name == name));
        match found.and_then(pattern_tile) {
            Some(tile) => Pattern::Tile(tile),
            None => {
                let label = if name.is_empty() { id } else { name };
                self.warnings.insert(format!("texture pattern \"{label}\" is not embedded in the file; a procedural paper texture stands in"));
                Pattern::Procedural { style: PatternStyle::Paper, size: 128, seed: 1 }
            }
        }
    }

    /// Mixer Brush tool options captured with a preset (Wet, Load, Mix, Flow, Sample All Layers).
    /// Key names are best effort (Photoshop doesn't document them); several spellings are read.
    fn mixer_options(&mut self, to: &Descriptor, m: &mut MixerSettings) {
        let first = |keys: &[&str]| keys.iter().find_map(|k| pct(to, k));
        let wet = first(&["wetness", "Wtns", "wet"]);
        let load = first(&["dryness", "load", "Ld  "]);
        let mix = first(&["mix", "mixRatio", "Mx  "]);
        if wet.is_none() && load.is_none() && mix.is_none() {
            return;
        }
        m.wet = wet.unwrap_or(m.wet).clamp(0.0, 1.0);
        m.load = load.unwrap_or(m.load).clamp(0.0, 1.0);
        m.mix = mix.unwrap_or(m.mix).clamp(0.0, 1.0);
        m.flow = pct(to, "flow").unwrap_or(m.flow).clamp(0.0, 1.0);
        if let Some(v) = ["sampleMerged", "useAllLayers", "sampleAllLayers", "Mrgd"].iter().find_map(|k| boolean(to, k)) {
            m.sample_all_layers = v;
        }
    }

    fn preset(&mut self, d: &Descriptor) -> Option<BrushSettings> {
        for (k, _) in &d.items {
            let k = String::from_utf8_lossy(k.as_bytes()).to_string();
            if !KNOWN.contains(&k.as_str()) {
                self.unknown.insert(k);
            }
        }
        let mut b = BrushSettings { pressure_size: false, spacing: 0.25, ..Default::default() };
        if let Some(t) = obj(d, "Brsh") {
            let (tip, sampled_size) = self.tip_shape(t)?;
            b.tip = tip;
            b.size = num(t, "Dmtr").map(|v| v as f32).or(sampled_size).unwrap_or(b.size).clamp(1.0, 5000.0);
            if let Some(v) = pct(t, "Hrdn") {
                b.hardness = v.clamp(0.0, 1.0);
            }
            b.angle = num(t, "Angl").unwrap_or(0.0) as f32;
            b.roundness = pct(t, "Rndn").unwrap_or(1.0).clamp(0.01, 1.0);
            // `Intr` off = the Spacing checkbox is off: the pointer's speed sets the spacing.
            b.spacing = pct(t, "Spcn").unwrap_or(0.25).clamp(0.01, 10.0);
            b.spacing_enabled = boolean(t, "Intr") != Some(false);
            b.flip_x = boolean(t, "flipX").unwrap_or(false);
            b.flip_y = boolean(t, "flipY").unwrap_or(false);
        }
        // Shape Dynamics.
        if boolean(d, "useTipDynamics") == Some(true) {
            let mut size = self.dynamic(obj(d, "szVr"));
            size.minimum = pct(d, "minimumDiameter").unwrap_or(size.minimum).clamp(0.0, 1.0);
            let mut roundness = self.dynamic(obj(d, "roundnessDynamics"));
            roundness.minimum = pct(d, "minimumRoundness").unwrap_or(roundness.minimum).clamp(0.0, 1.0);
            b.shape_dynamics = ShapeDynamics {
                enabled: true,
                size,
                angle: self.dynamic(obj(d, "angleDynamics")),
                roundness,
                flip_x_jitter: boolean(d, "flipX").unwrap_or(false),
                flip_y_jitter: boolean(d, "flipY").unwrap_or(false),
                tilt_scale: pct(d, "tiltScale").unwrap_or(0.0).clamp(0.0, 2.0),
                brush_projection: boolean(d, "brushProjection").unwrap_or(false),
            };
        }
        // Scattering.
        if boolean(d, "useScatter") == Some(true) {
            b.scattering.enabled = true;
            b.scattering.scatter = self.dynamic(obj(d, "scatterDynamics"));
            b.scattering.both_axes = boolean(d, "bothAxes").unwrap_or(false);
            b.scattering.count = num(d, "Cnt ").map_or(1, |v| v.clamp(1.0, 16.0) as u32);
            b.scattering.count_jitter = self.dynamic(obj(d, "countDynamics"));
        }
        // Texture.
        if boolean(d, "useTexture") == Some(true) {
            let t = &mut b.texture;
            t.enabled = true;
            t.each_tip = boolean(d, "TxtC").unwrap_or(false);
            t.invert = boolean(d, "InvT").unwrap_or(false);
            t.scale = pct(d, "textureScale").unwrap_or(1.0).clamp(0.01, 10.0);
            t.brightness = (num(d, "textureBrightness").unwrap_or(0.0) / 150.0).clamp(-1.0, 1.0) as f32;
            // Photoshop's -50..100 onto the model's -1..1 (negative and positive scale apart).
            let c = num(d, "textureContrast").unwrap_or(0.0);
            t.contrast = (if c < 0.0 { c / 50.0 } else { c / 100.0 }).clamp(-1.0, 1.0) as f32;
            t.mode = mask_mode(enum_value(d, "textureBlendMode")).unwrap_or(MaskMode::Multiply);
            t.depth = pct(d, "textureDepth").unwrap_or(1.0).clamp(0.0, 1.0);
            let mut dj = self.dynamic(obj(d, "textureDepthDynamics"));
            dj.minimum = pct(d, "minimumDepth").unwrap_or(dj.minimum).clamp(0.0, 1.0);
            t.depth_jitter = dj;
            if let Some(p) = obj(d, "Txtr") {
                b.texture.pattern = self.pattern(p);
            }
        }
        b.protect_texture = boolean(d, "protectTexture").unwrap_or(false);
        // Dual Brush.
        if let Some(db) = obj(d, "dualBrush").filter(|db| boolean(db, "useDualBrush") == Some(true)) {
            let mut dual = DualBrush { enabled: true, ..Default::default() };
            if let Some(t) = obj(db, "Brsh") {
                match self.tip_shape(t) {
                    Some((tip, sampled_size)) => {
                        dual.tip = tip;
                        dual.size = num(t, "Dmtr").map(|v| v as f32).or(sampled_size).unwrap_or(dual.size).clamp(1.0, 5000.0);
                    }
                    None => {
                        self.warnings.insert("a dual brush tip is missing from the file; a round tip stands in".into());
                    }
                }
                dual.hardness = pct(t, "Hrdn").unwrap_or(1.0).clamp(0.0, 1.0);
                dual.roundness = pct(t, "Rndn").unwrap_or(1.0).clamp(0.01, 1.0);
                dual.angle = num(t, "Angl").unwrap_or(0.0) as f32;
                dual.spacing = pct(t, "Spcn").unwrap_or(dual.spacing).clamp(0.01, 10.0);
            }
            dual.mode = mask_mode(enum_value(db, "BlnM")).unwrap_or(MaskMode::Multiply);
            dual.flip = boolean(db, "Flip").unwrap_or(false);
            // The dual tip's own spacing (`Brsh › Spcn`) is what Photoshop's Dual Brush panel
            // shows. The `Spcn` beside it on the `dualBrush` object reads 100 % in Photoshop CC
            // presets whatever the panel says (#4809), so it only stands in for a tip without one.
            if obj(db, "Brsh").and_then(|t| pct(t, "Spcn")).is_none()
                && let Some(v) = pct(db, "Spcn")
            {
                dual.spacing = v.clamp(0.01, 10.0);
            }
            dual.scatter = self.dynamic(obj(db, "scatterDynamics")).jitter;
            dual.both_axes = boolean(db, "bothAxes").unwrap_or(false);
            dual.count = num(db, "Cnt ").map_or(1, |v| v.clamp(1.0, 16.0) as u32);
            b.dual_brush = dual;
        }
        // Color Dynamics.
        if boolean(d, "useColorDynamics") == Some(true) {
            let c = &mut b.color_dynamics;
            c.enabled = true;
            c.per_tip = boolean(d, "colorDynamicsPerTip").unwrap_or(true);
            c.hue_jitter = pct(d, "H   ").unwrap_or(0.0).clamp(0.0, 1.0);
            c.saturation_jitter = pct(d, "Strt").unwrap_or(0.0).clamp(0.0, 1.0);
            c.brightness_jitter = pct(d, "Brgh").unwrap_or(0.0).clamp(0.0, 1.0);
            c.purity = pct(d, "purity").unwrap_or(0.0).clamp(-1.0, 1.0);
            let fg = self.dynamic(obj(d, "clVr"));
            b.color_dynamics.fg_bg = fg;
        }
        // Transfer.
        if boolean(d, "usePaintDynamics") == Some(true) {
            b.transfer = Transfer {
                enabled: true,
                opacity: self.dynamic(obj(d, "opVr")),
                flow: self.dynamic(obj(d, "prVr")),
                wetness: self.dynamic(obj(d, "wtVr")),
                mix: self.dynamic(obj(d, "mxVr")),
            };
        }
        // Brush Pose (Photoshop's tilt is ±100 %; ours is ±90°).
        if boolean(d, "useBrushPose") == Some(true) {
            b.pose = Pose {
                enabled: true,
                tilt_x: (num(d, "brushPoseTiltX").unwrap_or(0.0) * 0.9).clamp(-90.0, 90.0) as f32,
                tilt_y: (num(d, "brushPoseTiltY").unwrap_or(0.0) * 0.9).clamp(-90.0, 90.0) as f32,
                rotation: num(d, "brushPoseAngle").unwrap_or(0.0).rem_euclid(360.0) as f32,
                pressure: pct(d, "brushPosePressure").unwrap_or(1.0).clamp(0.0, 1.0),
                override_tilt: boolean(d, "overridePoseTiltX").unwrap_or(false) || boolean(d, "overridePoseTiltY").unwrap_or(false),
                override_rotation: boolean(d, "overridePoseAngle").unwrap_or(false),
                override_pressure: boolean(d, "overridePosePressure").unwrap_or(false),
            };
        }
        b.noise = boolean(d, "Nose").unwrap_or(false);
        b.wet_edges = boolean(d, "Wtdg").unwrap_or(false);
        b.build_up = boolean(d, "Rpt ").unwrap_or(false);
        // Tool options captured with the preset ("Include tool settings").
        if let Some(to) = obj(d, "toolOptions") {
            if let Some(v) = pct(to, "Opct") {
                b.opacity = v.clamp(0.0, 1.0);
            }
            if let Some(v) = pct(to, "flow") {
                b.flow = v.clamp(0.0, 1.0);
            }
            b.pressure_size = boolean(to, "usePressureOverridesSize").unwrap_or(false);
            b.pressure_opacity = boolean(to, "usePressureOverridesOpacity").unwrap_or(false);
            b.smoothing.enabled = boolean(to, "smoothing").unwrap_or(true);
            if let Some(v) = pct(to, "smoothingValue") {
                b.smoothing.amount = v.clamp(0.0, 1.0);
            }
            if let Some(v) = boolean(to, "smoothingRadiusMode") {
                b.smoothing.pulled_string = v;
            }
            if let Some(v) = boolean(to, "smoothingCatchup") {
                b.smoothing.catch_up = v;
            }
            if let Some(v) = boolean(to, "smoothingCatchupAtEnd") {
                b.smoothing.catch_up_on_end = v;
            }
            if let Some(v) = boolean(to, "smoothingZoomCompensation") {
                b.smoothing.adjust_for_zoom = v;
            }
            self.mixer_options(to, &mut b.mixer);
            if mask_blend_is_set(to) {
                self.warnings.insert("tool blend modes stored with presets are not applied".into());
            }
        }
        Some(b)
    }
}

fn mask_blend_is_set(to: &Descriptor) -> bool {
    enum_value(to, "Md  ").is_some_and(|m| m != "Nrml")
}

/// Box-downsample a `w × h` plane so neither side exceeds `max`.
fn fit(w: u32, h: u32, v: Vec<f32>, max: u32) -> (u32, u32, Vec<f32>) {
    if w <= max && h <= max {
        return (w, h, v);
    }
    let k = (w.max(h) as f32 / max as f32).max(1.0);
    let (nw, nh) = (((w as f32 / k).round() as u32).clamp(1, max), ((h as f32 / k).round() as u32).clamp(1, max));
    let mut out = vec![0.0f32; nw as usize * nh as usize];
    // Source rows/columns `[i * n / m, (i + 1) * n / m)` (at least one) for output index `i` of
    // `m`; in u64, as `(i + 1) * n` passes u32 for a plane over 2^32 / `max` long. The parsers'
    // edge caps keep today's inputs below that, but this function should not depend on them.
    let span = |i: u32, n: u32, m: u32| {
        let at = |k: u32| usize::try_from(u64::from(k) * u64::from(n) / u64::from(m)).unwrap_or(usize::MAX);
        (at(i), at(i + 1).max(at(i).saturating_add(1)).min(n as usize))
    };
    for y in 0..nh {
        let (y0, y1) = span(y, h, nh);
        for x in 0..nw {
            let (x0, x1) = span(x, w, nw);
            let (mut sum, mut n) = (0.0, 0usize);
            for yy in y0..y1 {
                for xx in x0..x1 {
                    if let Some(s) = v.get(yy * w as usize + xx) {
                        sum += s;
                        n += 1;
                    }
                }
            }
            out[(y * nw + x) as usize] = if n > 0 { sum / n as f32 } else { 0.0 };
        }
    }
    (nw, nh, out)
}

/// A pattern as a grayscale texture tile (luminance; transparent areas count as white).
fn pattern_tile(p: &PsdPattern) -> Option<GrayTile> {
    let (w, h) = (p.width, p.height);
    let n = w as usize * h as usize;
    if n == 0 {
        return None;
    }
    let plane = |c: usize| -> Option<Vec<f32>> {
        let ch = p.channels.get(c)?;
        Some(match p.depth {
            8 => ch.iter().take(n).map(|&b| f32::from(b) / 255.0).collect(),
            16 => ch.as_chunks::<2>().0.iter().take(n).map(|b| f32::from(u16::from_be_bytes(*b)) / 65535.0).collect(),
            32 => ch.as_chunks::<4>().0.iter().take(n).map(|b| f32::from_be_bytes(*b).clamp(0.0, 1.0)).collect(),
            _ => return None,
        })
        .filter(|v: &Vec<f32>| v.len() == n)
    };
    let gray: Vec<f32> = match p.mode {
        3 => {
            let (r, g, b) = (plane(0)?, plane(1)?, plane(2)?);
            (0..n).map(|i| 0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i]).collect()
        }
        2 => {
            let pal = p.palette.as_deref()?;
            let idx = p.channels.first()?;
            (0..n)
                .map(|i| {
                    let k = usize::from(idx.get(i).copied().unwrap_or(0)) * 3;
                    let c = |o: usize| f32::from(pal.get(k + o).copied().unwrap_or(0)) / 255.0;
                    0.299 * c(0) + 0.587 * c(1) + 0.114 * c(2)
                })
                .collect()
        }
        4 => {
            // CMYK planes are stored inverted (0 = full ink).
            let k = plane(3)?;
            let c = plane(0)?;
            (0..n).map(|i| (c[i] * k[i]).clamp(0.0, 1.0)).collect()
        }
        _ => plane(0)?,
    };
    let (w, h, gray) = fit(w, h, gray, MAX_PATTERN_EDGE);
    Some(GrayTile { width: w, height: h, data: gray.iter().map(|x| (x.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16).collect() })
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod fit_tests {
    use super::fit;

    /// A tall, narrow plane: `(y + 1) * h` passes `u32::MAX` once `h` is over 2^32 / `max` rows.
    /// The ABR and pattern parsers cap edges well below that, but `fit` must not rely on it.
    #[test]
    fn fit_tall_planes_without_overflow() {
        let h = 5_000_000u32;
        let (w, nh, out) = fit(1, h, vec![1.0; h as usize], 1024);
        assert_eq!((w, nh), (1, 1024));
        assert!(out.iter().all(|v| (*v - 1.0).abs() < 1e-6), "every output row averages real input rows");
    }
}
