//! HEIF / HEIC (the iPhone and Mac photo format), read-only: see [`crate::ASYMMETRIC_EXCEPTIONS`].
//!
//! Decoding lives in the optional `photocraft-heif` crate (heic-rs, a pure-Rust HEVC
//! still-picture decoder), enabled by this crate's `heif` feature. Without it, HEIF files are
//! still detected and opening one is an [`CodecError::Unsupported`] error, never a panic.
//!
//! HEIF records orientation in the container (`irot`/`imir`, plus a `clap` crop), not in EXIF,
//! so those are applied by the decoder and the EXIF Orientation is rewritten to 1. Its EXIF tag
//! only mirrors what the container already says, and applying it as well would turn the photo
//! twice.

use crate::Format;
use crate::error::CodecError;
use crate::image::Image;
use crate::options::Limits;

const F: Format = Format::Heif;

/// The reason given when this build has no HEIF decoder.
#[cfg(not(feature = "heif"))]
pub(crate) const NOT_IN_BUILD: &str = "HEIC/HEIF support isn't included in this build of PhotoCraft";

#[cfg(not(feature = "heif"))]
pub(crate) fn decode(_bytes: &[u8], _limits: &Limits, _keep_orientation: bool) -> Result<Image, CodecError> {
    Err(CodecError::unsupported(F, NOT_IN_BUILD))
}

#[cfg(feature = "heif")]
fn err(e: photocraft_heif::Error) -> CodecError {
    match e {
        photocraft_heif::Error::Unsupported(what) => CodecError::unsupported(F, what),
        photocraft_heif::Error::Limit(what) => CodecError::LimitExceeded(what),
        photocraft_heif::Error::Malformed(what) => CodecError::malformed(F, what),
    }
}

#[cfg(feature = "heif")]
pub(crate) fn decode(bytes: &[u8], limits: &Limits, keep_orientation: bool) -> Result<Image, CodecError> {
    use crate::image::{ChannelLayout, Metadata, SampleType};

    // The container alone: the declared size is checked before any pixel is decoded.
    let info = photocraft_heif::probe(bytes).map_err(err)?;
    let layout = if info.has_alpha { ChannelLayout::Rgba } else { ChannelLayout::Rgb };
    let sample = if info.sixteen_bit { SampleType::U16 } else { SampleType::U8 };
    let (w, h) = if keep_orientation { (info.coded_width, info.coded_height) } else { (info.width, info.height) };
    limits.check(w, h, layout, sample)?;
    let options = photocraft_heif::Options { max_pixels: limits.max_pixels, apply_transforms: !keep_orientation };
    let decoded = photocraft_heif::decode(bytes, &options).map_err(err)?;
    // The decoder re-reads the container; trust its output's shape, not the probe's.
    let layout = if decoded.has_alpha { ChannelLayout::Rgba } else { ChannelLayout::Rgb };
    let sample = if decoded.sixteen_bit { SampleType::U16 } else { SampleType::U8 };
    let mut img = Image::from_raw(decoded.width, decoded.height, layout, sample, decoded.data)?;
    img.icc = decoded.icc;
    img.meta = Metadata {
        exif: decoded.exif.map(|e| if keep_orientation { e } else { crate::orientation::upright_exif(&e).into_owned() }),
        xmp: decoded.xmp.map(|x| if keep_orientation { x } else { crate::orientation::upright_xmp(&x).into_owned() }),
        ..Default::default()
    };
    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "heif"))]
    #[test]
    fn without_the_feature_heif_is_a_clear_unsupported_error() {
        let header = b"\0\0\0\x18ftypheic\0\0\0\0mif1heic";
        assert_eq!(crate::detect(header), Some(F), "still recognised");
        let r = crate::decode(header);
        assert!(matches!(&r, Err(CodecError::Unsupported { format: Format::Heif, reason }) if reason.contains("isn't included in this build")), "{r:?}");
        assert!(!crate::caps(F).read);
    }

    #[cfg(feature = "heif")]
    #[test]
    fn with_the_feature_heif_is_readable_and_errors_map() {
        assert!(crate::caps(F).read);
        let r = crate::decode(b"\0\0\0\x18ftypmsf1\0\0\0\0msf1hevc");
        assert!(matches!(&r, Err(CodecError::Unsupported { format: Format::Heif, reason }) if reason.contains("sequence")), "{r:?}");
        assert!(matches!(crate::decode_as(F, b""), Err(CodecError::Malformed { .. } | CodecError::Unsupported { .. })));
    }

    /// Found by the `decode_heif` fuzz target: a malformed box makes heic-rs 0.1.1 slice out of range
    /// (`boxes.rs:130`, "slice index starts at 24 but ends at 16"). It must be an error, not a crash.
    #[cfg(feature = "heif")]
    const HEIC_RS_BOX_PANIC: [u8; 72] = [
        0x00, 0x00, 0x00, 0x24, 0x66, 0x74, 0x79, 0x70, 0x68, 0x65, 0x69, 0x63, 0x00, 0x00, 0x00, 0x00, 0x6d, 0x69, 0x66, 0x31, 0x4d, 0x69, 0x50, 0x72, 0x6d,
        0x69, 0x61, 0x66, 0x4d, 0x69, 0x48, 0x42, 0x72, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6d, 0x65, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x10, 0x75, 0x75, 0x69, 0x64, 0x00, 0x00, 0x00, 0x00, 0x03, 0x08, 0x08, 0x08, 0x00, 0x03, 0x01, 0x03, 0x70, 0x00, 0xa8, 0x00,
    ];

    #[cfg(feature = "heif")]
    #[test]
    fn a_heic_rs_panic_is_a_malformed_file_error() {
        assert_eq!(crate::detect(&HEIC_RS_BOX_PANIC), Some(Format::Heif));
        for opts in [crate::DecodeOptions::default(), crate::DecodeOptions { keep_orientation: true, ..Default::default() }] {
            let r = crate::decode_with(&HEIC_RS_BOX_PANIC, &opts);
            assert!(matches!(r, Err(CodecError::Malformed { format: Format::Heif, .. })), "{r:?}");
        }
    }
}
