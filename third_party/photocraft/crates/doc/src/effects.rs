//! Layer styles (effects): data only. Rendering lives in `photocraft-compose`,
//! PSD mapping in `photocraft-io`.
//!
//! Angles are in degrees, counter-clockwise from 3 o'clock, and give the
//! direction the light comes *from* (Photoshop convention). Sizes and
//! distances are in pixels, opacities and percentages in `0..=1`.

use photocraft_color::{BlendMode, Color};
use serde::{Deserialize, Serialize};

use crate::adjust::CurvePoint;

/// Gradient geometry (fill layers, gradient overlays, gradient strokes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GradientStyle {
    #[default]
    Linear,
    Radial,
    Angle,
    Reflected,
    Diamond,
}

/// A gradient: colour stops plus transparency stops (both at `0..=1`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Gradient {
    pub stops: Vec<(f32, Color)>,
    /// Opacity stops; empty means fully opaque.
    pub opacity_stops: Vec<(f32, f32)>,
    pub style: GradientStyle,
    pub angle: f32,
    pub scale: f32,
    pub reverse: bool,
    /// Align the gradient with the layer bounds (else the canvas).
    pub align: bool,
    /// Centre offset as a fraction of the bounds.
    pub offset: (f32, f32),
}

impl Default for Gradient {
    fn default() -> Self {
        Gradient {
            stops: vec![(0.0, Color::BLACK), (1.0, Color::WHITE)],
            opacity_stops: Vec::new(),
            style: GradientStyle::Linear,
            angle: 90.0,
            scale: 1.0,
            reverse: false,
            align: true,
            offset: (0.0, 0.0),
        }
    }
}

/// Effect contour (transfer curve applied to the effect's coverage).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum Contour {
    #[default]
    Linear,
    /// Custom curve (input → output), both `0..=1`.
    Custom {
        name: String,
        points: Vec<CurvePoint>,
        /// Corner (non-continuous) flags, parallel to `points`: the curve bends sharply at a
        /// corner point instead of passing smoothly through it. Missing entries are smooth.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        corners: Vec<bool>,
    },
}

impl Contour {
    /// Whether point `i` of a custom contour is a corner.
    pub fn is_corner(&self, i: usize) -> bool {
        match self {
            Contour::Linear => false,
            Contour::Custom { corners, .. } => corners.get(i).copied().unwrap_or(false),
        }
    }
}

/// Paint used by strokes and glows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FxPaint {
    Color(Color),
    Gradient(Gradient),
    /// Pattern by name/id (rendered once pattern data is available).
    Pattern {
        name: String,
        id: String,
        scale: f32,
    },
}

/// Blend mode + opacity + enabled, shared by every effect.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FxCommon {
    pub enabled: bool,
    pub blend: BlendMode,
    pub opacity: f32,
}

impl FxCommon {
    pub fn new(blend: BlendMode, opacity: f32) -> Self {
        FxCommon { enabled: true, blend, opacity }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Shadow {
    pub common: FxCommon,
    pub color: Color,
    pub angle: f32,
    pub use_global_light: bool,
    pub distance: f32,
    /// Spread (drop shadow) or choke (inner shadow), `0..=1`.
    pub spread: f32,
    pub size: f32,
    pub contour: Contour,
    pub anti_alias: bool,
    pub noise: f32,
    /// Drop shadow only: the layer knocks out the shadow beneath it.
    pub knocks_out: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GlowTechnique {
    #[default]
    Softer,
    Precise,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GlowSource {
    Center,
    #[default]
    Edge,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Glow {
    pub common: FxCommon,
    pub paint: FxPaint,
    pub technique: GlowTechnique,
    /// Spread (outer) or choke (inner), `0..=1`.
    pub spread: f32,
    pub size: f32,
    pub contour: Contour,
    pub anti_alias: bool,
    pub range: f32,
    pub jitter: f32,
    pub noise: f32,
    /// Inner glow only.
    pub source: GlowSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrokePosition {
    Outside,
    Inside,
    Center,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StrokeFx {
    pub common: FxCommon,
    pub size: f32,
    pub position: StrokePosition,
    pub paint: FxPaint,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Satin {
    pub common: FxCommon,
    pub color: Color,
    pub angle: f32,
    pub distance: f32,
    pub size: f32,
    pub contour: Contour,
    pub anti_alias: bool,
    pub invert: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum BevelStyle {
    OuterBevel,
    #[default]
    InnerBevel,
    Emboss,
    PillowEmboss,
    StrokeEmboss,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum BevelTechnique {
    #[default]
    Smooth,
    ChiselHard,
    ChiselSoft,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bevel {
    pub enabled: bool,
    pub style: BevelStyle,
    pub technique: BevelTechnique,
    /// Depth, `0..=10` (1 = 100 %).
    pub depth: f32,
    /// Direction up (true) or down.
    pub up: bool,
    pub size: f32,
    pub soften: f32,
    pub angle: f32,
    pub altitude: f32,
    pub use_global_light: bool,
    pub gloss_contour: Contour,
    pub highlight: FxCommon,
    pub highlight_color: Color,
    pub shadow: FxCommon,
    pub shadow_color: Color,
    /// Bevel & Emboss › Contour: shapes the bevel's height profile. `None` = off.
    #[serde(default)]
    pub contour: Option<BevelContour>,
    /// Bevel & Emboss › Texture: a pattern's luminance added to the height. `None` = off.
    #[serde(default)]
    pub texture: Option<BevelTexture>,
}

/// The Contour element of Bevel & Emboss.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BevelContour {
    pub contour: Contour,
    /// Range `0..=1` (Photoshop's default 50 %): the part of the bevel the contour spans.
    pub range: f32,
    pub anti_alias: bool,
}

impl Default for BevelContour {
    fn default() -> Self {
        BevelContour { contour: Contour::Linear, range: 0.5, anti_alias: false }
    }
}

/// The Texture element of Bevel & Emboss.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BevelTexture {
    pub name: String,
    pub id: String,
    /// Scale as a fraction (1 = 100 %).
    pub scale: f32,
    /// Depth `-10..=10` (1 = +100 %); negative carves the pattern in.
    pub depth: f32,
    pub invert: bool,
    /// "Link with Layer": tile from the layer (else the canvas origin).
    pub link: bool,
    pub phase: (f32, f32),
}

/// One layer effect. Several instances of the same kind are allowed
/// (Photoshop's multiple strokes/shadows/overlays).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Effect {
    DropShadow(Shadow),
    InnerShadow(Shadow),
    OuterGlow(Glow),
    InnerGlow(Glow),
    Stroke(StrokeFx),
    ColorOverlay {
        common: FxCommon,
        color: Color,
    },
    GradientOverlay {
        common: FxCommon,
        gradient: Gradient,
        dither: bool,
    },
    PatternOverlay {
        common: FxCommon,
        name: String,
        id: String,
        scale: f32,
        /// Rotation in degrees (counter-clockwise).
        #[serde(default)]
        angle: f32,
        /// "Link with Layer": the pattern origin follows the layer (else the canvas).
        #[serde(default = "yes")]
        link: bool,
        /// Phase (origin offset) in pixels.
        #[serde(default)]
        phase: (f32, f32),
    },
    Satin(Satin),
    BevelEmboss(Bevel),
}

impl Effect {
    /// This effect with its colours in `mode`'s model ([`Color::in_mode`]): a layer's effect
    /// colours are in its document's mode, as Image › Mode keeps them.
    pub fn in_mode(mut self, mode: photocraft_color::ColorMode) -> Effect {
        let cc = |c: &mut Color| *c = c.in_mode(mode);
        let paint = |p: &mut FxPaint| match p {
            FxPaint::Color(c) => cc(c),
            FxPaint::Gradient(g) => g.stops.iter_mut().for_each(|s| cc(&mut s.1)),
            FxPaint::Pattern { .. } => {}
        };
        match &mut self {
            Effect::DropShadow(s) | Effect::InnerShadow(s) => cc(&mut s.color),
            Effect::OuterGlow(g) | Effect::InnerGlow(g) => paint(&mut g.paint),
            Effect::Stroke(s) => paint(&mut s.paint),
            Effect::ColorOverlay { color, .. } => cc(color),
            Effect::GradientOverlay { gradient, .. } => gradient.stops.iter_mut().for_each(|s| cc(&mut s.1)),
            Effect::Satin(s) => cc(&mut s.color),
            Effect::BevelEmboss(b) => {
                cc(&mut b.highlight_color);
                cc(&mut b.shadow_color);
            }
            Effect::PatternOverlay { .. } => {}
        }
        self
    }

    /// Whether this instance is switched on.
    pub fn enabled(&self) -> bool {
        match self {
            Effect::DropShadow(s) | Effect::InnerShadow(s) => s.common.enabled,
            Effect::OuterGlow(g) | Effect::InnerGlow(g) => g.common.enabled,
            Effect::Stroke(s) => s.common.enabled,
            Effect::ColorOverlay { common, .. } | Effect::GradientOverlay { common, .. } | Effect::PatternOverlay { common, .. } => common.enabled,
            Effect::Satin(s) => s.common.enabled,
            Effect::BevelEmboss(b) => b.enabled,
        }
    }

    /// Switches this instance on or off, keeping its settings.
    pub fn set_enabled(&mut self, on: bool) {
        match self {
            Effect::DropShadow(s) | Effect::InnerShadow(s) => s.common.enabled = on,
            Effect::OuterGlow(g) | Effect::InnerGlow(g) => g.common.enabled = on,
            Effect::Stroke(s) => s.common.enabled = on,
            Effect::ColorOverlay { common, .. } | Effect::GradientOverlay { common, .. } | Effect::PatternOverlay { common, .. } => common.enabled = on,
            Effect::Satin(s) => s.common.enabled = on,
            Effect::BevelEmboss(b) => b.enabled = on,
        }
    }

    /// Short label for UIs.
    pub fn label(&self) -> &'static str {
        match self {
            Effect::DropShadow(_) => "Drop Shadow",
            Effect::InnerShadow(_) => "Inner Shadow",
            Effect::OuterGlow(_) => "Outer Glow",
            Effect::InnerGlow(_) => "Inner Glow",
            Effect::Stroke(_) => "Stroke",
            Effect::ColorOverlay { .. } => "Color Overlay",
            Effect::GradientOverlay { .. } => "Gradient Overlay",
            Effect::PatternOverlay { .. } => "Pattern Overlay",
            Effect::Satin(_) => "Satin",
            Effect::BevelEmboss(_) => "Bevel & Emboss",
        }
    }

    /// Photoshop's defaults for a drop shadow.
    pub fn default_drop_shadow() -> Self {
        Effect::DropShadow(Shadow {
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
            knocks_out: true,
        })
    }
}

/// Serde default for flags that start on.
pub fn yes() -> bool {
    true
}

/// Document-wide light used by effects with `use_global_light`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GlobalLight {
    pub angle: f32,
    pub altitude: f32,
}

impl Default for GlobalLight {
    fn default() -> Self {
        GlobalLight { angle: 120.0, altitude: 30.0 }
    }
}
