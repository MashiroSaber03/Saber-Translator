//! Photocraft colour management: a pure-Rust ICC v2/v4 colour management module.
//!
//! * [`Profile`] parses ICC profiles (matrix/TRC RGB, gray TRC, `mft1`/`mft2`/`mAB `/`mBA `
//!   LUT profiles incl. CMYK↔Lab, `curv`/`para` curves) and writes ICC v4 profiles.
//! * [`Transform`] connects two profiles through the PCS (XYZ/Lab D50) with a rendering
//!   [`Intent`] and optional black point compensation, and converts 8/16-bit and float pixel
//!   buffers through precomputed tables (tetrahedral 3D/4D interpolation, rayon on native).
//! * [`builtin`] ships sRGB, Display P3, Adobe RGB (1998)-compatible, ProPhoto-compatible,
//!   linear sRGB, Rec. 2020, Gray 2.2, sGray, Lab D50 and a synthetic coated CMYK profile.
//! * [`Lut3d`] samples a display transform into a 3D texture for the GPU canvas.
//!
//! Implemented from the public ICC.1:2010 (v4.3) and ICC.1:2001-04 (v2) specifications,
//! CIE 15, and Adobe's published black point compensation paper. This crate depends on no
//! other workspace crate.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod builtin;
pub mod clut;
pub mod curve;
pub mod gamut;
pub mod lut3d;
pub mod lutfile;
pub mod math;
pub mod pipeline;
pub mod profile;
pub mod synth;
pub mod transform;
mod write;

pub use builtin::Builtin;
pub use clut::Clut;
pub use curve::Curve;
pub use gamut::GamutCheck;
pub use lut3d::Lut3d;
pub use profile::{ColorSpace, Pcs, Profile, ProfileClass};
pub use transform::{SampleKind, Transform, TransformOptions, black_point, cached};

/// ICC rendering intent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Intent {
    Perceptual = 0,
    #[default]
    RelativeColorimetric = 1,
    Saturation = 2,
    AbsoluteColorimetric = 3,
}

impl Intent {
    pub const ALL: [Intent; 4] = [Intent::Perceptual, Intent::RelativeColorimetric, Intent::Saturation, Intent::AbsoluteColorimetric];

    pub fn from_u32(v: u32) -> Option<Intent> {
        Some(match v {
            0 => Intent::Perceptual,
            1 => Intent::RelativeColorimetric,
            2 => Intent::Saturation,
            3 => Intent::AbsoluteColorimetric,
            _ => return None,
        })
    }

    /// Parses `perceptual`, `relative`/`relativeColorimetric`, `saturation`,
    /// `absolute`/`absoluteColorimetric` (case-insensitive).
    pub fn parse(s: &str) -> Option<Intent> {
        let n: String = s.to_ascii_lowercase().chars().filter(|c| c.is_ascii_alphanumeric()).collect();
        Some(match n.as_str() {
            "perceptual" | "0" => Intent::Perceptual,
            "relative" | "relativecolorimetric" | "colorimetric" | "1" => Intent::RelativeColorimetric,
            "saturation" | "2" => Intent::Saturation,
            "absolute" | "absolutecolorimetric" | "3" => Intent::AbsoluteColorimetric,
            _ => return None,
        })
    }

    pub fn id(self) -> &'static str {
        match self {
            Intent::Perceptual => "perceptual",
            Intent::RelativeColorimetric => "relative",
            Intent::Saturation => "saturation",
            Intent::AbsoluteColorimetric => "absolute",
        }
    }
}

/// Errors from parsing profiles or building transforms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CmsError {
    Truncated,
    /// Missing `acsp` signature: not an ICC profile.
    BadSignature,
    Invalid(String),
    Unsupported(String),
}

impl std::fmt::Display for CmsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CmsError::Truncated => write!(f, "ICC profile is truncated"),
            CmsError::BadSignature => write!(f, "not an ICC profile (missing 'acsp' signature)"),
            CmsError::Invalid(m) => write!(f, "invalid ICC profile: {m}"),
            CmsError::Unsupported(m) => write!(f, "unsupported ICC feature: {m}"),
        }
    }
}

impl std::error::Error for CmsError {}
