//! Layer Style › Blending Options › Advanced Blending: knockout and the "as group" / "hides
//! effects" switches. Fill opacity ([`crate::Layer::fill_opacity`]), the channel restrictions
//! ([`crate::Layer::excluded_channels`]) and Blend If ([`crate::Layer::blend_if`]) live on the
//! layer itself; this struct holds the rest of the section.
//!
//! Defaults are Photoshop's: no knockout, interior effects blended separately, clipped layers
//! blended as a group, transparency shaping the layer's effects and knockout, and masks applied
//! before the effects are built (so the effects follow the masked shape).
//!
//! PSD: `knko` (0 none, 1 shallow, 2 deep), `infx`, `clbl`, `tsly`, `lmgm`, `vmgm`, each a
//! one-byte flag followed by three bytes of padding (Adobe Photoshop File Format specification,
//! "Additional Layer Information").

use serde::{Deserialize, Serialize};

/// Blending Options › Advanced Blending › Knockout: which layers beneath this one its shape
/// punches through. Fill opacity controls the strength (0 % fill shows the knocked-out area bare).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Knockout {
    /// The layer blends with everything beneath it (the default).
    #[default]
    None,
    /// Knocks out to the bottom of the enclosing layer group or clipping group (for a layer at
    /// the top level: to the Background layer, or to transparency without one).
    Shallow,
    /// Knocks out to the Background layer (to transparency without one), through every group.
    Deep,
}

impl Knockout {
    /// The PSD `knko` value (0 none, 1 shallow, 2 deep).
    pub fn to_psd(self) -> u8 {
        match self {
            Knockout::None => 0,
            Knockout::Shallow => 1,
            Knockout::Deep => 2,
        }
    }

    /// From a PSD `knko` value; values above 2 (not written by Photoshop) read as deep.
    pub fn from_psd(v: u8) -> Self {
        match v {
            0 => Knockout::None,
            1 => Knockout::Shallow,
            _ => Knockout::Deep,
        }
    }

    /// The parameter / UI name (`none`, `shallow`, `deep`).
    pub fn name(self) -> &'static str {
        match self {
            Knockout::None => "none",
            Knockout::Shallow => "shallow",
            Knockout::Deep => "deep",
        }
    }

    /// Inverse of [`Knockout::name`] (case-insensitive).
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "none" => Some(Knockout::None),
            "shallow" => Some(Knockout::Shallow),
            "deep" => Some(Knockout::Deep),
            _ => None,
        }
    }
}

/// The Advanced Blending switches of a layer (see the module docs). `Default` is Photoshop's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AdvancedBlending {
    /// Knockout: None / Shallow / Deep. PSD `knko`.
    pub knockout: Knockout,
    /// Blend Interior Effects as Group: Inner Glow, Satin and the Color, Gradient and Pattern
    /// Overlays are combined with the layer's content before its fill opacity and blend mode
    /// apply (off: fill opacity leaves them alone). PSD `infx`, default off.
    pub blend_interior: bool,
    /// Blend Clipped Layers as Group: the clipping group composites as a whole in the base
    /// layer's blend mode (off: each clipped layer blends with the layers beneath the base in its
    /// own mode, inside the base's shape). PSD `clbl`, default on.
    pub blend_clipped: bool,
    /// Transparency Shapes Layer: the layer's transparency limits its effects and knockout (off:
    /// they cover the whole layer). PSD `tsly`, default on.
    pub transparency_shapes: bool,
    /// Layer Mask Hides Effects: the layer mask hides the effects instead of shaping them. PSD
    /// `lmgm` ("layer mask as global mask"), default off.
    pub layer_mask_hides_effects: bool,
    /// Vector Mask Hides Effects: the vector mask hides the effects instead of shaping them. PSD
    /// `vmgm`, default off.
    pub vector_mask_hides_effects: bool,
}

impl Default for AdvancedBlending {
    fn default() -> Self {
        AdvancedBlending {
            knockout: Knockout::None,
            blend_interior: false,
            blend_clipped: true,
            transparency_shapes: true,
            layer_mask_hides_effects: false,
            vector_mask_hides_effects: false,
        }
    }
}

impl AdvancedBlending {
    /// Whether every switch is at Photoshop's default.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_photoshop() {
        let a = AdvancedBlending::default();
        assert_eq!(a.knockout, Knockout::None);
        assert!(!a.blend_interior);
        assert!(a.blend_clipped);
        assert!(a.transparency_shapes);
        assert!(!a.layer_mask_hides_effects);
        assert!(!a.vector_mask_hides_effects);
        assert!(a.is_default());
        assert!(!AdvancedBlending { blend_clipped: false, ..a }.is_default());
    }

    #[test]
    fn knockout_psd_values_and_names() {
        for k in [Knockout::None, Knockout::Shallow, Knockout::Deep] {
            assert_eq!(Knockout::from_psd(k.to_psd()), k);
            assert_eq!(Knockout::from_name(k.name()), Some(k));
        }
        assert_eq!(Knockout::from_psd(7), Knockout::Deep);
        assert_eq!(Knockout::from_name("DEEP"), Some(Knockout::Deep));
        assert_eq!(Knockout::from_name("medium"), None);
    }

    #[test]
    fn partial_json_keeps_photoshop_defaults() {
        let a: AdvancedBlending = serde_json::from_str(r#"{"knockout":"deep"}"#).unwrap();
        assert_eq!(a, AdvancedBlending { knockout: Knockout::Deep, ..Default::default() });
        let b: AdvancedBlending = serde_json::from_str("{}").unwrap();
        assert!(b.is_default());
        let json = serde_json::to_string(&AdvancedBlending { layer_mask_hides_effects: true, ..Default::default() }).unwrap();
        assert!(json.contains("\"layerMaskHidesEffects\":true"), "{json}");
    }
}
