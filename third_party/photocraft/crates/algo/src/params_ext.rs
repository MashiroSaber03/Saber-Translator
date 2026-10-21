//! Option enums and parameter objects for the second batch of filters
//! (pixelate, stylize, render, blur gallery, video…).

use serde::{Deserialize, Serialize};

/// Mezzotint pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum MezzotintType {
    #[default]
    FineDots,
    MediumDots,
    GrainyDots,
    CoarseDots,
    ShortLines,
    MediumLines,
    LongLines,
    ShortStrokes,
    MediumStrokes,
    LongStrokes,
}

/// Diffuse mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum DiffuseMode {
    #[default]
    Normal,
    DarkenOnly,
    LightenOnly,
    Anisotropic,
}

/// Extrude geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum ExtrudeType {
    #[default]
    Blocks,
    Pyramids,
}

/// What Tiles puts in the gaps between tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum TileFill {
    #[default]
    Background,
    Foreground,
    Inverse,
    Unaltered,
}

/// Wind method.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum WindMethod {
    #[default]
    Wind,
    Blast,
    Stagger,
}

/// ZigZag style.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum ZigZagStyle {
    AroundCenter,
    OutFromCenter,
    #[default]
    PondRipples,
}

/// Lens Flare lens type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum LensType {
    /// 50–300 mm zoom.
    #[default]
    Zoom,
    Prime35,
    Prime105,
    MoviePrime,
}

/// Lighting Effects light type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum LightKind {
    #[default]
    Spot,
    Point,
    Infinite,
}

/// One Lighting Effects light. Positions are fractions of the reference
/// bounds; `z` is the height above the surface as a fraction of the bounds'
/// shorter side.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Light {
    pub kind: LightKind,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    /// Spot: where the light points (fractions of the bounds).
    pub target_x: f32,
    pub target_y: f32,
    /// Infinite: direction the light comes from (degrees, 0 = right, counter-clockwise) and elevation.
    pub angle: f32,
    pub elevation: f32,
    /// Linear RGB 0–1.
    pub color: [f32; 3],
    /// −100…100 (negative lights darken).
    pub intensity: f32,
    /// Spot cone half-angle in degrees.
    pub cone: f32,
    /// Spot hotspot, 0–100 % of the cone at full intensity.
    pub hotspot: f32,
    /// Point light reach as a fraction of the bounds' shorter side.
    pub radius: f32,
}

impl Default for Light {
    fn default() -> Self {
        Light {
            kind: LightKind::Spot,
            x: 0.25,
            y: 0.2,
            z: 0.6,
            target_x: 0.5,
            target_y: 0.55,
            angle: 135.0,
            elevation: 45.0,
            color: [1.0, 1.0, 1.0],
            intensity: 75.0,
            cone: 45.0,
            hotspot: 50.0,
            radius: 0.8,
        }
    }
}

/// Bump texture source for Lighting Effects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum TextureChannel {
    #[default]
    None,
    Red,
    Green,
    Blue,
    Alpha,
    Luminance,
}

/// Smart Blur sampling quality.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum BlurQuality {
    Low,
    #[default]
    Medium,
    High,
}

/// Smart Blur output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum SmartBlurMode {
    #[default]
    Normal,
    EdgeOnly,
    OverlayEdge,
}

/// Where Lens Blur reads depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum DepthSource {
    #[default]
    None,
    Transparency,
    LayerMask,
}

/// Built-in Shape Blur kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum BlurShape {
    #[default]
    Circle,
    Ring,
    Square,
    Diamond,
    Triangle,
    Hexagon,
    Star,
    Heart,
    Cross,
}

/// Colour encodings for Filter › Other › HSB/HSL.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum HsbModel {
    #[default]
    Rgb,
    Hsb,
    Hsl,
}

/// Iris Blur pin: an ellipse (fractions of the bounds' shorter side) that stays sharp.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct IrisPin {
    pub x: f32,
    pub y: f32,
    pub radius_x: f32,
    pub radius_y: f32,
    /// Ellipse rotation in degrees.
    pub angle: f32,
    /// 0 = ellipse, 100 = rounded rectangle.
    pub roundness: f32,
    /// Sharp core as a fraction of the ellipse (0–1).
    pub feather: f32,
    /// Blur at and beyond the ellipse, px.
    pub blur: f32,
}

impl Default for IrisPin {
    fn default() -> Self {
        IrisPin { x: 0.5, y: 0.5, radius_x: 0.35, radius_y: 0.25, angle: 0.0, roundness: 0.0, feather: 0.5, blur: 15.0 }
    }
}

/// Field Blur pin: a blur amount at a point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct FieldPin {
    pub x: f32,
    pub y: f32,
    pub blur: f32,
}

impl Default for FieldPin {
    fn default() -> Self {
        FieldPin { x: 0.5, y: 0.5, blur: 15.0 }
    }
}

/// Spin Blur pin: an ellipse whose content is rotated about its centre.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SpinPin {
    pub x: f32,
    pub y: f32,
    pub radius_x: f32,
    pub radius_y: f32,
    pub angle: f32,
    /// Blur arc in degrees.
    pub blur_angle: f32,
}

impl Default for SpinPin {
    fn default() -> Self {
        SpinPin { x: 0.5, y: 0.5, radius_x: 0.3, radius_y: 0.3, angle: 0.0, blur_angle: 15.0 }
    }
}

/// Path Blur path: a polyline (fractions of the bounds) giving the motion direction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct BlurPath {
    pub points: Vec<[f32; 2]>,
    /// Blur length in px.
    pub speed: f32,
    /// 0–100: how much the blur fades toward the path's end.
    pub taper: f32,
}

impl Default for BlurPath {
    fn default() -> Self {
        BlurPath { points: vec![[0.2, 0.5], [0.8, 0.5]], speed: 50.0, taper: 0.0 }
    }
}
