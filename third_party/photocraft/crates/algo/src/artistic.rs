//! Filter Gallery: the 47 "artistic" filters of Photoshop's gallery (Artistic, Brush Strokes,
//! Distort, Sketch, Stylize and Texture), applied as a stack of effect layers in order.
//!
//! Each filter is described by a parameter notation string (the same notation the engine
//! registry uses, e.g. `{"pencilWidth":1..24=4,"texture":"canvas|brick"}`, first choice =
//! default) so the engine, the dialogs and agents share one source of truth. The pixel work
//! lives in [`crate::artistic_fx`]. Implemented clean-room from observed behaviour: parameter
//! names and ranges follow Photoshop's dialogs; the looks are our own approximations.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

macro_rules! gallery {
    ($( $v:ident = $key:literal, $name:literal, $cat:literal, $doc:literal; )*) => {
        /// One Filter Gallery filter.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(rename_all = "camelCase")]
        pub enum GalleryFilter { $($v),* }

        impl GalleryFilter {
            /// Every gallery filter, grouped by category in Photoshop's order.
            pub const ALL: &'static [GalleryFilter] = &[$(GalleryFilter::$v),*];
            /// Camel-case key (`coloredPencil`); the command id is `filter.gallery.<key>`.
            pub fn key(self) -> &'static str { match self { $(GalleryFilter::$v => $key),* } }
            /// Engine command id.
            pub fn command_id(self) -> &'static str { match self { $(GalleryFilter::$v => concat!("filter.gallery.", $key)),* } }
            /// Display name (Photoshop's).
            pub fn name(self) -> &'static str { match self { $(GalleryFilter::$v => $name),* } }
            /// Gallery folder.
            pub fn category(self) -> &'static str { match self { $(GalleryFilter::$v => $cat),* } }
            /// Parameter notation (ranges, defaults, choices).
            pub fn params_doc(self) -> &'static str { match self { $(GalleryFilter::$v => $doc),* } }
        }
    };
}

gallery! {
    // ---- Artistic ----
    ColoredPencil = "coloredPencil", "Colored Pencil", "Artistic", r#"{"pencilWidth":1..24=4,"strokePressure":0..15=8,"paperBrightness":0..50=25}"#;
    Cutout = "cutout", "Cutout", "Artistic", r#"{"numberOfLevels":2..8=4,"edgeSimplicity":0..10=4,"edgeFidelity":1..3=2}"#;
    DryBrush = "dryBrush", "Dry Brush", "Artistic", r#"{"brushSize":0..10=2,"brushDetail":0..10=8,"texture":1..3=1}"#;
    FilmGrain = "filmGrain", "Film Grain", "Artistic", r#"{"grain":0..20=4,"highlightArea":0..20=0,"intensity":0..10=10}"#;
    Fresco = "fresco", "Fresco", "Artistic", r#"{"brushSize":0..10=2,"brushDetail":0..10=8,"texture":1..3=1}"#;
    NeonGlow = "neonGlow", "Neon Glow", "Artistic", r#"{"glowSize":-24..24=5,"glowBrightness":0..50=15,"glowColor":json}"#;
    PaintDaubs = "paintDaubs", "Paint Daubs", "Artistic", r#"{"brushSize":1..50=8,"sharpness":0..40=7,"brushType":"simple|lightRough|darkRough|wideSharp|wideBlurry|sparkle"}"#;
    PaletteKnife = "paletteKnife", "Palette Knife", "Artistic", r#"{"strokeSize":1..50=25,"strokeDetail":1..3=3,"softness":0..10=0}"#;
    PlasticWrap = "plasticWrap", "Plastic Wrap", "Artistic", r#"{"highlightStrength":0..20=15,"detail":1..15=9,"smoothness":1..15=7}"#;
    PosterEdges = "posterEdges", "Poster Edges", "Artistic", r#"{"edgeThickness":0..10=2,"edgeIntensity":0..10=1,"posterization":0..6=2}"#;
    RoughPastels = "roughPastels", "Rough Pastels", "Artistic", r#"{"strokeLength":0..40=6,"strokeDetail":1..20=4,"texture":"canvas|brick|burlap|sandstone","scaling":50..200=100,"relief":0..50=20,"light":"bottom|bottomLeft|left|topLeft|top|topRight|right|bottomRight","invert":bool}"#;
    SmudgeStick = "smudgeStick", "Smudge Stick", "Artistic", r#"{"strokeLength":0..10=2,"highlightArea":0..20=0,"intensity":0..10=10}"#;
    Sponge = "sponge", "Sponge", "Artistic", r#"{"brushSize":0..10=2,"definition":0..25=12,"smoothness":1..15=5}"#;
    Underpainting = "underpainting", "Underpainting", "Artistic", r#"{"brushSize":0..40=6,"textureCoverage":0..40=16,"texture":"canvas|brick|burlap|sandstone","scaling":50..200=100,"relief":0..50=4,"light":"top|topRight|right|bottomRight|bottom|bottomLeft|left|topLeft","invert":bool}"#;
    Watercolor = "watercolor", "Watercolor", "Artistic", r#"{"brushDetail":1..14=9,"shadowIntensity":0..10=1,"texture":1..3=1}"#;
    // ---- Brush Strokes ----
    AccentedEdges = "accentedEdges", "Accented Edges", "Brush Strokes", r#"{"edgeWidth":1..14=2,"edgeBrightness":0..50=38,"smoothness":1..15=5}"#;
    AngledStrokes = "angledStrokes", "Angled Strokes", "Brush Strokes", r#"{"directionBalance":0..100=50,"strokeLength":3..50=15,"sharpness":0..10=3}"#;
    Crosshatch = "crosshatch", "Crosshatch", "Brush Strokes", r#"{"strokeLength":3..50=9,"sharpness":0..20=6,"strength":1..3=1}"#;
    DarkStrokes = "darkStrokes", "Dark Strokes", "Brush Strokes", r#"{"balance":0..10=5,"blackIntensity":0..10=6,"whiteIntensity":0..10=2}"#;
    InkOutlines = "inkOutlines", "Ink Outlines", "Brush Strokes", r#"{"strokeLength":1..50=4,"darkIntensity":0..50=20,"lightIntensity":0..50=10}"#;
    Spatter = "spatter", "Spatter", "Brush Strokes", r#"{"sprayRadius":0..25=10,"smoothness":1..15=5}"#;
    SprayedStrokes = "sprayedStrokes", "Sprayed Strokes", "Brush Strokes", r#"{"strokeLength":0..20=12,"sprayRadius":0..25=7,"strokeDirection":"rightDiagonal|horizontal|leftDiagonal|vertical"}"#;
    SumiE = "sumiE", "Sumi-e", "Brush Strokes", r#"{"strokeWidth":3..15=10,"strokePressure":0..15=2,"contrast":0..40=16}"#;
    // ---- Distort ----
    DiffuseGlow = "diffuseGlow", "Diffuse Glow", "Distort", r#"{"graininess":0..10=6,"glowAmount":0..20=10,"clearAmount":0..20=15,"background":json}"#;
    Glass = "glass", "Glass", "Distort", r#"{"distortion":0..20=5,"smoothness":1..15=3,"texture":"frosted|blocks|canvas|tinyLens","scaling":50..200=100,"invert":bool}"#;
    OceanRipple = "oceanRipple", "Ocean Ripple", "Distort", r#"{"rippleSize":1..15=9,"rippleMagnitude":0..20=9}"#;
    // ---- Sketch (foreground = ink, background = paper) ----
    BasRelief = "basRelief", "Bas Relief", "Sketch", r#"{"detail":1..15=13,"smoothness":1..15=3,"light":"bottom|bottomLeft|left|topLeft|top|topRight|right|bottomRight","foreground":json,"background":json}"#;
    ChalkCharcoal = "chalkCharcoal", "Chalk & Charcoal", "Sketch", r#"{"charcoalArea":0..20=6,"chalkArea":0..20=6,"strokePressure":0..5=1,"foreground":json,"background":json}"#;
    Charcoal = "charcoal", "Charcoal", "Sketch", r#"{"charcoalThickness":1..7=1,"detail":0..5=5,"lightDarkBalance":0..100=50,"foreground":json,"background":json}"#;
    Chrome = "chrome", "Chrome", "Sketch", r#"{"detail":0..10=4,"smoothness":0..10=7}"#;
    ConteCrayon = "conteCrayon", "Conté Crayon", "Sketch", r#"{"foregroundLevel":1..15=11,"backgroundLevel":1..15=7,"texture":"canvas|brick|burlap|sandstone","scaling":50..200=100,"relief":0..50=4,"light":"top|topRight|right|bottomRight|bottom|bottomLeft|left|topLeft","invert":bool,"foreground":json,"background":json}"#;
    GraphicPen = "graphicPen", "Graphic Pen", "Sketch", r#"{"strokeLength":1..15=15,"lightDarkBalance":0..100=50,"strokeDirection":"rightDiagonal|horizontal|leftDiagonal|vertical","foreground":json,"background":json}"#;
    HalftonePattern = "halftonePattern", "Halftone Pattern", "Sketch", r#"{"size":1..12=1,"contrast":0..50=5,"patternType":"dot|circle|line","foreground":json,"background":json}"#;
    NotePaper = "notePaper", "Note Paper", "Sketch", r#"{"imageBalance":0..50=25,"graininess":0..20=10,"relief":0..25=11,"foreground":json,"background":json}"#;
    Photocopy = "photocopy", "Photocopy", "Sketch", r#"{"detail":1..24=7,"darkness":1..50=8,"foreground":json,"background":json}"#;
    Plaster = "plaster", "Plaster", "Sketch", r#"{"imageBalance":0..50=20,"smoothness":1..15=2,"light":"top|topRight|right|bottomRight|bottom|bottomLeft|left|topLeft","foreground":json,"background":json}"#;
    Reticulation = "reticulation", "Reticulation", "Sketch", r#"{"density":0..50=12,"foregroundLevel":0..50=40,"backgroundLevel":0..50=5,"foreground":json,"background":json}"#;
    Stamp = "stamp", "Stamp", "Sketch", r#"{"lightDarkBalance":0..50=25,"smoothness":1..50=5,"foreground":json,"background":json}"#;
    TornEdges = "tornEdges", "Torn Edges", "Sketch", r#"{"imageBalance":0..50=25,"smoothness":1..15=11,"contrast":1..25=17,"foreground":json,"background":json}"#;
    WaterPaper = "waterPaper", "Water Paper", "Sketch", r#"{"fiberLength":3..50=15,"brightness":0..100=60,"contrast":0..100=80}"#;
    // ---- Stylize ----
    GlowingEdges = "glowingEdges", "Glowing Edges", "Stylize", r#"{"edgeWidth":1..14=2,"edgeBrightness":0..20=6,"smoothness":1..15=5}"#;
    // ---- Texture ----
    Craquelure = "craquelure", "Craquelure", "Texture", r#"{"crackSpacing":2..100=15,"crackDepth":0..10=6,"crackBrightness":0..10=9}"#;
    Grain = "grain", "Grain", "Texture", r#"{"intensity":0..100=40,"contrast":0..100=50,"grainType":"regular|soft|sprinkles|clumped|contrasty|enlarged|stippled|horizontal|vertical|speckle","foreground":json,"background":json}"#;
    MosaicTiles = "mosaicTiles", "Mosaic Tiles", "Texture", r#"{"tileSize":2..100=12,"groutWidth":1..15=3,"lightenGrout":0..10=9}"#;
    Patchwork = "patchwork", "Patchwork", "Texture", r#"{"squareSize":0..10=4,"relief":0..25=8}"#;
    StainedGlass = "stainedGlass", "Stained Glass", "Texture", r#"{"cellSize":2..50=10,"borderThickness":1..20=4,"lightIntensity":0..10=3,"foreground":json}"#;
    Texturizer = "texturizer", "Texturizer", "Texture", r#"{"texture":"canvas|brick|burlap|sandstone","scaling":50..200=100,"relief":0..50=4,"light":"top|topRight|right|bottomRight|bottom|bottomLeft|left|topLeft","invert":bool}"#;
}

/// The gallery's folders, in order.
pub const GALLERY_CATEGORIES: [&str; 6] = ["Artistic", "Brush Strokes", "Distort", "Sketch", "Stylize", "Texture"];

/// One parameter of a gallery filter.
#[derive(Clone, Debug, PartialEq)]
pub enum GalleryParamKind {
    Range {
        min: f32,
        max: f32,
        default: f32,
    },
    /// Choices; the first is the default. Stored as the index.
    Choice(Vec<&'static str>),
    Bool(bool),
}

/// A parsed parameter.
#[derive(Clone, Debug, PartialEq)]
pub struct GalleryParam {
    pub key: &'static str,
    pub kind: GalleryParamKind,
}

impl GalleryFilter {
    /// Looks a filter up by key (`coloredPencil`), command id or display name.
    pub fn from_key(k: &str) -> Option<GalleryFilter> {
        let k = k.strip_prefix("filter.gallery.").unwrap_or(k);
        Self::ALL.iter().copied().find(|f| f.key() == k || f.name().eq_ignore_ascii_case(k))
    }

    /// The numeric / choice / bool parameters (colours are not listed; they are fields of
    /// [`GalleryEffect`]).
    pub fn params(self) -> Vec<GalleryParam> {
        let doc = self.params_doc();
        let inner = doc.trim().trim_start_matches('{').trim_end_matches('}');
        let mut out = Vec::new();
        for part in inner.split(',') {
            let Some((k, v)) = part.split_once(':') else { continue };
            let key = k.trim().trim_matches('"');
            let v = v.trim();
            let kind = if let Some(body) = v.strip_prefix('"') {
                GalleryParamKind::Choice(body.trim_end_matches('"').split('|').collect())
            } else if v == "bool" {
                GalleryParamKind::Bool(false)
            } else if let Some((range, def)) = v.split_once('=') {
                let (lo, hi) = range.split_once("..").unwrap_or(("0", "1"));
                let p = |s: &str| s.trim().parse::<f32>().unwrap_or(0.0);
                GalleryParamKind::Range { min: p(lo), max: p(hi), default: p(def) }
            } else {
                continue; // json (colours)
            };
            out.push(GalleryParam { key, kind });
        }
        out
    }

    /// Whether the filter reads the foreground / background colours. Derived
    /// from the params notation, plus the filters whose pixel code reads the
    /// colours without listing them (issue #709): Colored Pencil uses the
    /// background as paper tint, Neon Glow mixes foreground over background.
    pub fn uses_colours(self) -> bool {
        matches!(self, Self::ColoredPencil | Self::NeonGlow) || self.params_doc().contains("\"foreground\"") || self.params_doc().contains("\"background\"")
    }
}

/// Photoshop's default Neon Glow colour (a saturated sky blue).
pub const NEON_DEFAULT: [f32; 4] = [0.0, 0.62, 1.0, 1.0];

/// One effect layer of the gallery: a filter with its values. Choice values are indices into
/// the filter's choice list; bools are 0/1. Colours are straight sRGB RGBA, 0–1.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GalleryEffect {
    pub filter: GalleryFilter,
    #[serde(default)]
    pub values: BTreeMap<String, f32>,
    #[serde(default = "black")]
    pub foreground: [f32; 4],
    #[serde(default = "white")]
    pub background: [f32; 4],
    #[serde(default = "neon")]
    pub color: [f32; 4],
}

fn black() -> [f32; 4] {
    [0.0, 0.0, 0.0, 1.0]
}
fn white() -> [f32; 4] {
    [1.0, 1.0, 1.0, 1.0]
}
fn neon() -> [f32; 4] {
    NEON_DEFAULT
}

impl GalleryEffect {
    /// The filter with its default values, black ink on white paper.
    pub fn new(filter: GalleryFilter) -> Self {
        let mut values = BTreeMap::new();
        for p in filter.params() {
            let v = match p.kind {
                GalleryParamKind::Range { default, .. } => default,
                GalleryParamKind::Choice(_) => 0.0,
                GalleryParamKind::Bool(b) => b as u8 as f32,
            };
            values.insert(p.key.to_string(), v);
        }
        GalleryEffect { filter, values, foreground: black(), background: white(), color: neon() }
    }

    /// Sets a numeric value (clamped to its range), a choice by index, or a bool.
    pub fn set(&mut self, key: &str, v: f32) {
        if let Some(p) = self.filter.params().into_iter().find(|p| p.key == key) {
            let v = match p.kind {
                GalleryParamKind::Range { min, max, .. } => v.clamp(min, max),
                GalleryParamKind::Choice(c) => v.round().clamp(0.0, c.len() as f32 - 1.0),
                GalleryParamKind::Bool(_) => (v != 0.0) as u8 as f32,
            };
            self.values.insert(key.to_string(), v);
        }
    }

    /// Sets a choice by name; false if the name is not one of the choices.
    pub fn set_choice(&mut self, key: &str, name: &str) -> bool {
        let Some(GalleryParamKind::Choice(c)) = self.filter.params().into_iter().find(|p| p.key == key).map(|p| p.kind) else { return false };
        match c.iter().position(|n| n.eq_ignore_ascii_case(name)) {
            Some(i) => {
                self.values.insert(key.to_string(), i as f32);
                true
            }
            None => false,
        }
    }

    /// A numeric value (default when unset).
    pub fn get(&self, key: &str) -> f32 {
        if let Some(v) = self.values.get(key) {
            return *v;
        }
        match self.filter.params().into_iter().find(|p| p.key == key).map(|p| p.kind) {
            Some(GalleryParamKind::Range { default, .. }) => default,
            Some(GalleryParamKind::Bool(b)) => b as u8 as f32,
            _ => 0.0,
        }
    }

    /// A bool value.
    pub fn flag(&self, key: &str) -> bool {
        self.get(key) != 0.0
    }

    /// The chosen name of a choice parameter.
    pub fn choice(&self, key: &str) -> &'static str {
        match self.filter.params().into_iter().find(|p| p.key == key).map(|p| p.kind) {
            Some(GalleryParamKind::Choice(c)) => c.get(self.get(key).max(0.0) as usize).copied().unwrap_or(c[0]),
            _ => "",
        }
    }
}
