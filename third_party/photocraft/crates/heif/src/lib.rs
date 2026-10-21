//! `photocraft-heif`: the optional HEIF / HEIC (iPhone and Mac photo) decoder.
//!
//! A thin wrapper around `heic-rs`, a pure-Rust HEVC still-picture decoder: single pictures and
//! grid-tiled photos, 8- and 10-bit (10-bit decodes to 16-bit), alpha auxiliary images, ICC,
//! EXIF and XMP. Its API is plain data ([`Info`], [`Decoded`], [`Error`]) so the crate knows
//! nothing about the rest of PhotoCraft; `photocraft-codecs` adapts it behind its `heif` feature.
//!
//! HEIF records orientation in the container (`irot`/`imir`, plus a `clap` crop), not in EXIF.
//! [`Options::apply_transforms`] applies them; with it off, the pixels come back as coded.
//!
//! Never panics: heic-rs is young, so every call into it runs under `catch_unwind` and a panic
//! inside it becomes [`Error::Malformed`].

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::borrow::Cow;

use heic_rs::PixelLayout;

/// What a HEIF file declares, read from the container alone (no pixel is decoded).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Info {
    /// Size after the container transforms (rotation, mirror, crop): what a viewer shows.
    pub width: u32,
    pub height: u32,
    /// Size as coded, before the transforms.
    pub coded_width: u32,
    pub coded_height: u32,
    /// The primary image has an alpha auxiliary image.
    pub has_alpha: bool,
    /// More than 8 bits per sample: it decodes to 16-bit.
    pub sixteen_bit: bool,
}

/// Decode settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    /// Refuse images with more pixels than this (checked before decoding).
    pub max_pixels: u64,
    /// Apply the container's rotation, mirror and crop (`irot`/`imir`/`clap`).
    pub apply_transforms: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { max_pixels: 1 << 28, apply_transforms: true }
    }
}

/// A decoded image: interleaved RGB or RGBA samples, row-major, no padding. 16-bit samples are
/// native-endian `u16`s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decoded {
    pub width: u32,
    pub height: u32,
    /// Four channels (RGBA) instead of three (RGB).
    pub has_alpha: bool,
    /// Two bytes per sample instead of one.
    pub sixteen_bit: bool,
    pub data: Vec<u8>,
    /// The primary image's ICC profile.
    pub icc: Option<Vec<u8>>,
    /// EXIF as a TIFF structure (without HEIF's offset header). Its Orientation tag only mirrors
    /// what the container says; it is returned verbatim.
    pub exif: Option<Vec<u8>>,
    /// The XMP packet.
    pub xmp: Option<String>,
}

/// Why a file could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A valid file using something this decoder does not handle (image sequences, overlays…).
    Unsupported(String),
    /// The image is larger than [`Options::max_pixels`] or holds an oversized box.
    Limit(String),
    /// Broken or truncated data (or a decoder bug, reported the same way).
    Malformed(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Unsupported(m) => write!(f, "unsupported HEIF: {m}"),
            Error::Limit(m) => write!(f, "limit exceeded: {m}"),
            Error::Malformed(m) => write!(f, "malformed HEIF: {m}"),
        }
    }
}

impl std::error::Error for Error {}

const SEQUENCE: &str = "this file holds an image sequence; only HEIF still images can be opened";

fn err(e: heic_rs::Error) -> Error {
    match e {
        heic_rs::Error::Unsupported(what) => Error::Unsupported(what.to_string()),
        // The primary item is neither an HEVC picture nor a grid of them.
        heic_rs::Error::MissingBox("hvcC") => Error::Unsupported("the image is an overlay, an identity derivation or not HEVC-coded".into()),
        heic_rs::Error::PixelLimit { .. } | heic_rs::Error::BoxTooLarge { .. } => Error::Limit(e.to_string()),
        e => Error::Malformed(e.to_string()),
    }
}

/// Runs `f`, turning a heic-rs panic into an error. Fuzzing found two out-of-range slices in
/// heic-rs 0.1.1 on malformed files: in its box parser (`boxes.rs:130`) and during reconstruction
/// (`hevc/decode/recon.rs:70`). The codecs fuzz target also calls heic-rs directly, so its panics
/// stay visible there.
fn guarded<T>(f: impl FnOnce() -> Result<T, Error>) -> Result<T, Error> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|_| Err(Error::Malformed("the HEIF decoder failed on this file".into())))
}

/// Reads the container's declared size, depth and alpha without decoding pixels, so callers can
/// check their limits first.
pub fn probe(bytes: &[u8]) -> Result<Info, Error> {
    guarded(|| probe_unguarded(bytes))
}

fn probe_unguarded(bytes: &[u8]) -> Result<Info, Error> {
    let info = heic_rs::probe(bytes).map_err(|e| match e {
        // No `meta` box: no still image, only a `moov` image sequence (a video track).
        heic_rs::Error::MissingBox("meta") => Error::Unsupported(SEQUENCE.into()),
        e => err(e),
    })?;
    Ok(Info {
        width: info.width,
        height: info.height,
        coded_width: info.coded_width,
        coded_height: info.coded_height,
        has_alpha: info.has_alpha,
        sixteen_bit: info.bit_depth > 8,
    })
}

/// Decodes the primary image, with its metadata.
pub fn decode(bytes: &[u8], options: &Options) -> Result<Decoded, Error> {
    guarded(|| decode_unguarded(bytes, options))
}

fn decode_unguarded(bytes: &[u8], options: &Options) -> Result<Decoded, Error> {
    let info = probe_unguarded(bytes)?;
    let pixel_layout = match (info.has_alpha, info.sixteen_bit) {
        (true, true) => PixelLayout::Rgba16,
        (true, false) => PixelLayout::Rgba8,
        (false, true) => PixelLayout::Rgb16,
        (false, false) => PixelLayout::Rgb8,
    };
    let raw = heic_rs::DecodeOptions::default()
        .with_layout(pixel_layout)
        .with_max_pixels(Some(options.max_pixels))
        .with_transforms(options.apply_transforms)
        .with_alpha(info.has_alpha);
    let decoded = heic_rs::decode(bytes, &raw).map_err(err)?;
    let (icc, exif, xmp) = metadata(bytes);
    Ok(Decoded { width: decoded.width, height: decoded.height, has_alpha: info.has_alpha, sixteen_bit: info.sixteen_bit, data: decoded.data, icc, exif, xmp })
}

/// The primary image's ICC profile, EXIF (TIFF structure) and XMP packet. Metadata is a
/// courtesy: a broken metadata item never fails a decode whose pixels came out fine.
fn metadata(bytes: &[u8]) -> (Option<Vec<u8>>, Option<Vec<u8>>, Option<String>) {
    let Ok(ctx) = heic_rs::context::Context::open(bytes) else {
        return (None, None, None);
    };
    let id = ctx.meta.primary;
    let icc = ctx.props(id).ok().and_then(|p| ctx.icc(&p).map(<[u8]>::to_vec));
    let exif = ctx.exif(id).ok().flatten().map(<[u8]>::to_vec);
    let xmp = ctx.xmp_item(id).and_then(|x| ctx.item_data(x).ok()).and_then(|d| match d {
        Cow::Borrowed(b) => String::from_utf8(b.to_vec()).ok(),
        Cow::Owned(b) => String::from_utf8(b).ok(),
    });
    (icc, exif, xmp.map(|x| x.trim_end_matches('\0').to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Found by the `decode_heif` fuzz target: a malformed box makes heic-rs 0.1.1 slice out of
    /// range (`boxes.rs:130`). It must be an error, not a crash.
    const HEIC_RS_BOX_PANIC: [u8; 72] = [
        0x00, 0x00, 0x00, 0x24, 0x66, 0x74, 0x79, 0x70, 0x68, 0x65, 0x69, 0x63, 0x00, 0x00, 0x00, 0x00, 0x6d, 0x69, 0x66, 0x31, 0x4d, 0x69, 0x50, 0x72, 0x6d,
        0x69, 0x61, 0x66, 0x4d, 0x69, 0x48, 0x42, 0x72, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6d, 0x65, 0x74, 0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x10, 0x75, 0x75, 0x69, 0x64, 0x00, 0x00, 0x00, 0x00, 0x03, 0x08, 0x08, 0x08, 0x00, 0x03, 0x01, 0x03, 0x70, 0x00, 0xa8, 0x00,
    ];

    #[test]
    fn a_heic_rs_panic_is_a_malformed_error() {
        for apply_transforms in [true, false] {
            let r = decode(&HEIC_RS_BOX_PANIC, &Options { apply_transforms, ..Default::default() });
            assert!(matches!(r, Err(Error::Malformed(_))), "{r:?}");
        }
    }

    #[test]
    fn image_sequence_is_unsupported_with_a_clear_message() {
        // An ftyp and nothing else: a HEIF image sequence would have a moov here instead of a meta.
        let bytes = b"\0\0\0\x18ftypmsf1\0\0\0\0msf1hevc";
        for r in [probe(bytes).map(|_| ()), decode(bytes, &Options::default()).map(|_| ())] {
            assert!(matches!(&r, Err(Error::Unsupported(m)) if m.contains("sequence")), "{r:?}");
        }
    }

    #[test]
    fn garbage_is_an_error() {
        for bytes in [&b""[..], b"\0\0\0\x18ftypheic\0\0\0\0mif1heic", &[0xFF; 64]] {
            assert!(probe(bytes).is_err());
            assert!(decode(bytes, &Options::default()).is_err());
        }
    }

    #[test]
    fn errors_display_their_kind() {
        assert!(Error::Unsupported("x".into()).to_string().contains("unsupported"));
        assert!(Error::Limit("x".into()).to_string().contains("limit"));
        assert!(Error::Malformed("x".into()).to_string().contains("malformed"));
    }
}
