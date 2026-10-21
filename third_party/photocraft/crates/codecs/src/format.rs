//! Format identification and capability declarations.

use crate::image::{ChannelLayout, SampleType};

/// Supported (or known) raster file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Format {
    Png,
    Jpeg,
    Tiff,
    WebP,
    Gif,
    Bmp,
    Tga,
    Ico,
    /// Netpbm family: PBM/PGM/PPM (P1–P6), PAM (P7) and PFM (`PF`/`Pf`).
    Pnm,
    /// ZSoft Paintbrush: 1/2/4-bit planar palettes, 8-bit indexed and 24-bit RGB.
    Pcx,
    /// X PixMap icons (XPM2 plain text and XPM3 C-string arrays).
    Xpm,
    /// DirectDraw Surface: the top mip level of a 2D texture, uncompressed or BC1–BC3.
    Dds,
    Qoi,
    OpenExr,
    /// Radiance RGBE.
    Hdr,
    /// Encode-only behind the `avif` feature; see [`ASYMMETRIC_EXCEPTIONS`].
    Avif,
    /// HEIF/HEIC (HEVC-coded): read-only; see [`ASYMMETRIC_EXCEPTIONS`].
    Heif,
    /// FITS (Flexible Image Transport System), the astronomy format: gray or RGB image data.
    Fits,
    /// DPX (SMPTE 268M Digital Picture Exchange): luma, RGB or RGBA film and video frames.
    Dpx,
}

/// What a format can hold **and** what this crate reads/writes for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatCaps {
    /// Decoding is available in this build.
    pub read: bool,
    /// Encoding is available in this build.
    pub write: bool,
    /// Sample types stored without conversion.
    pub depths: &'static [SampleType],
    /// Layouts stored *and read back* without conversion.
    pub layouts: &'static [ChannelLayout],
    pub alpha: bool,
    pub icc: bool,
    pub exif: bool,
    pub xmp: bool,
    /// Physical resolution (DPI) is preserved.
    pub dpi: bool,
    /// Free-form key/value text is preserved.
    pub text: bool,
    /// The container can hold animation/multiple frames. This crate only
    /// decodes the first frame and encodes a single frame.
    pub animation: bool,
    /// Our encoder is lossy (quantization or lossy compression).
    pub lossy: bool,
}

/// Formats that are enabled but intentionally not symmetric, with the reason.
/// The test-suite asserts that every other enabled format is read+write.
pub const ASYMMETRIC_EXCEPTIONS: &[(Format, &str)] = &[
    (Format::Xpm, "XPM is a legacy Unix icon interchange format. Reading existing XPM icons is supported; writing modern image assets should use PNG instead."),
    (
        Format::Pcx,
        "PCX is supported for opening legacy paint and clipart assets. New files should use PNG; writing obsolete indexed and planar variants is intentionally not provided.",
    ),
    (
        Format::Dds,
        "DDS is supported for opening game and 3D textures. Writing one well means choosing a block compression and generating mip levels, which belongs in a texture export, not Save As.",
    ),
    (
        Format::Avif,
        "AVIF encode uses ravif (pure Rust) but decoding requires dav1d (C); read stays unsupported \
         until a pure-Rust AV1 decoder is viable. Only enabled with the non-default `avif` feature.",
    ),
    (
        Format::Heif,
        "HEIC decode uses heic-rs (pure Rust, in the optional photocraft-heif crate behind the \
         non-default `heif` feature, which official builds enable), so iPhone and Mac photos open; \
         writing needs an HEVC encoder, and the mature ones (x265, libheif) are C, so write stays \
         unsupported. Without the feature HEIF is detected but neither read nor written.",
    ),
];

use ChannelLayout as L;
use SampleType as S;

const RGB_GRAY: &[ChannelLayout] = &[L::Gray, L::GrayA, L::Rgb, L::Rgba];
const ALL_LAYOUTS: &[ChannelLayout] = &[L::Gray, L::GrayA, L::Rgb, L::Rgba, L::Cmyk, L::CmykA];

impl Format {
    /// Every format known to the crate (enabled or not).
    pub const ALL: [Format; 19] = [
        Format::Png,
        Format::Jpeg,
        Format::Tiff,
        Format::WebP,
        Format::Gif,
        Format::Bmp,
        Format::Tga,
        Format::Ico,
        Format::Pnm,
        Format::Pcx,
        Format::Xpm,
        Format::Dds,
        Format::Qoi,
        Format::OpenExr,
        Format::Hdr,
        Format::Avif,
        Format::Heif,
        Format::Fits,
        Format::Dpx,
    ];

    pub fn caps(self) -> FormatCaps {
        caps(self)
    }

    /// Readable or writable in this build.
    pub fn is_enabled(self) -> bool {
        let c = caps(self);
        c.read || c.write
    }

    pub fn name(self) -> &'static str {
        match self {
            Format::Png => "PNG",
            Format::Jpeg => "JPEG",
            Format::Tiff => "TIFF",
            Format::WebP => "WebP",
            Format::Gif => "GIF",
            Format::Bmp => "BMP",
            Format::Tga => "TGA",
            Format::Ico => "ICO",
            Format::Pnm => "Netpbm",
            Format::Pcx => "PCX",
            Format::Xpm => "XPM",
            Format::Dds => "DDS",
            Format::Qoi => "QOI",
            Format::OpenExr => "OpenEXR",
            Format::Hdr => "Radiance HDR",
            Format::Avif => "AVIF",
            Format::Heif => "HEIF",
            Format::Fits => "FITS",
            Format::Dpx => "DPX",
        }
    }

    /// Recognized file extensions (lowercase, without dot); the first one is
    /// the preferred extension for writing.
    pub fn extensions(self) -> &'static [&'static str] {
        match self {
            Format::Png => &["png", "apng"],
            Format::Jpeg => &["jpg", "jpeg", "jpe", "jfif"],
            Format::Tiff => &["tif", "tiff"],
            Format::WebP => &["webp"],
            Format::Gif => &["gif"],
            Format::Bmp => &["bmp", "dib"],
            Format::Tga => &["tga", "icb", "vda", "vst"],
            Format::Ico => &["ico"],
            Format::Pnm => &["pnm", "pbm", "pgm", "ppm", "pam", "pfm"],
            Format::Pcx => &["pcx"],
            Format::Xpm => &["xpm"],
            Format::Dds => &["dds"],
            Format::Qoi => &["qoi"],
            Format::OpenExr => &["exr"],
            Format::Hdr => &["hdr"],
            Format::Avif => &["avif"],
            Format::Heif => &["heic", "heif", "hif"],
            Format::Fits => &["fits", "fit", "fts"],
            Format::Dpx => &["dpx"],
        }
    }

    pub fn mime_type(self) -> &'static str {
        match self {
            Format::Png => "image/png",
            Format::Jpeg => "image/jpeg",
            Format::Tiff => "image/tiff",
            Format::WebP => "image/webp",
            Format::Gif => "image/gif",
            Format::Bmp => "image/bmp",
            Format::Tga => "image/x-tga",
            Format::Ico => "image/x-icon",
            Format::Pnm => "image/x-portable-anymap",
            Format::Pcx => "image/x-pcx",
            Format::Xpm => "image/x-xpixmap",
            Format::Dds => "image/vnd-ms.dds",
            Format::Qoi => "image/qoi",
            Format::OpenExr => "image/x-exr",
            Format::Hdr => "image/vnd.radiance",
            Format::Avif => "image/avif",
            Format::Heif => "image/heif",
            Format::Fits => "image/fits",
            Format::Dpx => "image/x-dpx",
        }
    }

    /// Maximum encodable dimensions, if the format has a hard limit.
    pub fn max_dimensions(self) -> Option<(u32, u32)> {
        match self {
            Format::Jpeg | Format::Gif => Some((65535, 65535)),
            Format::WebP => Some((16384, 16384)),
            Format::Ico => Some((256, 256)),
            _ => None,
        }
    }
}

/// Capability table. `read`/`write` reflect the current build's features.
pub fn caps(format: Format) -> FormatCaps {
    let base = FormatCaps {
        read: true,
        write: true,
        depths: &[S::U8],
        layouts: &[L::Rgb, L::Rgba],
        alpha: true,
        icc: false,
        exif: false,
        xmp: false,
        dpi: false,
        text: false,
        animation: false,
        lossy: false,
    };
    match format {
        Format::Png => {
            FormatCaps { depths: &[S::U8, S::U16], layouts: RGB_GRAY, icc: true, exif: true, xmp: true, dpi: true, text: true, animation: true, ..base }
        }
        Format::Jpeg => FormatCaps { layouts: &[L::Gray, L::Rgb, L::Cmyk], alpha: false, icc: true, exif: true, xmp: true, dpi: true, lossy: true, ..base },
        Format::Tiff => FormatCaps { depths: &[S::U8, S::U16, S::F32], layouts: ALL_LAYOUTS, icc: true, exif: true, xmp: true, dpi: true, text: true, ..base },
        Format::WebP => FormatCaps { icc: true, exif: true, xmp: true, animation: true, ..base },
        Format::Gif => FormatCaps { layouts: &[L::Rgba], animation: true, lossy: true, ..base },
        Format::Bmp => base,
        Format::Tga => FormatCaps { layouts: RGB_GRAY, ..base },
        Format::Ico => FormatCaps { layouts: &[L::Rgba], ..base },
        Format::Pnm => FormatCaps { depths: &[S::U8, S::U16, S::F32], layouts: ALL_LAYOUTS, ..base },
        Format::Pcx => FormatCaps { read: true, write: false, layouts: &[L::Rgb], alpha: false, ..base },
        Format::Xpm => FormatCaps { read: true, write: false, layouts: &[L::Rgba], ..base },
        Format::Dds => FormatCaps { read: true, write: false, layouts: RGB_GRAY, ..base },
        Format::Qoi => base,
        Format::OpenExr => FormatCaps { depths: &[S::F16, S::F32], layouts: RGB_GRAY, ..base },
        Format::Hdr => FormatCaps { depths: &[S::F32], layouts: &[L::Rgb], alpha: false, lossy: true, ..base },
        Format::Avif => FormatCaps { read: false, write: cfg!(feature = "avif"), lossy: true, ..base },
        Format::Heif => {
            FormatCaps { read: cfg!(feature = "heif"), write: false, depths: &[S::U8, S::U16], icc: true, exif: true, xmp: true, lossy: true, ..base }
        }
        Format::Fits => FormatCaps { depths: &[S::U8, S::U16, S::F32], layouts: &[L::Gray, L::Rgb], alpha: false, text: true, ..base },
        // `text`: the DPX header fields only (see `fidelity_warnings_with`).
        Format::Dpx => FormatCaps { depths: &[S::U8, S::U16], layouts: &[L::Gray, L::Rgb, L::Rgba], text: true, ..base },
    }
}

/// Identify a format from its leading bytes (magic numbers). TGA has no
/// magic and is recognized last, by header plausibility.
pub fn detect(bytes: &[u8]) -> Option<Format> {
    let b = bytes;
    if b.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Some(Format::Png);
    }
    if b.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some(Format::Jpeg);
    }
    if b.starts_with(b"II*\0") || b.starts_with(b"MM\0*") || b.starts_with(b"II+\0") || b.starts_with(b"MM\0+") {
        return Some(Format::Tiff);
    }
    if b.len() >= 12 && &b[0..4] == b"RIFF" && &b[8..12] == b"WEBP" {
        return Some(Format::WebP);
    }
    if b.starts_with(b"GIF87a") || b.starts_with(b"GIF89a") {
        return Some(Format::Gif);
    }
    if b.starts_with(b"qoif") {
        return Some(Format::Qoi);
    }
    if b.starts_with(b"DDS ") {
        return Some(Format::Dds);
    }
    if b.starts_with(&[0x76, 0x2F, 0x31, 0x01]) {
        return Some(Format::OpenExr);
    }
    if b.starts_with(b"#?RADIANCE") || b.starts_with(b"#?RGBE") {
        return Some(Format::Hdr);
    }
    if b.len() >= 12 && &b[4..8] == b"ftyp" {
        if is_avif_ftyp(b) {
            return Some(Format::Avif);
        }
        if is_heif_ftyp(b) {
            return Some(Format::Heif);
        }
    }
    if b.len() >= 14 && b.starts_with(b"BM") {
        return Some(Format::Bmp);
    }
    if b.len() >= 6 && b[0..4] == [0, 0, 1, 0] && (b[4] != 0 || b[5] != 0) {
        return Some(Format::Ico);
    }
    if b.len() >= 3 && b[0] == b'P' && matches!(b[1], b'1'..=b'7' | b'F' | b'f') && b[2].is_ascii_whitespace() {
        return Some(Format::Pnm);
    }
    if b.starts_with(b"SDPX") || b.starts_with(b"XPDS") {
        return Some(Format::Dpx);
    }
    if b.starts_with(b"SIMPLE  =") {
        return Some(Format::Fits);
    }
    if b.starts_with(b"/* XPM */") || b.starts_with(b"! XPM2\n") || b.starts_with(b"! XPM2\r\n") {
        return Some(Format::Xpm);
    }
    if crate::codecs::pcx::looks_like_pcx(b) {
        return Some(Format::Pcx);
    }
    if looks_like_tga(b) {
        return Some(Format::Tga);
    }
    None
}

/// The `ftyp` brands: the major brand, then the compatible ones (the minor version between them is skipped).
fn ftyp_brands(b: &[u8]) -> impl Iterator<Item = &[u8; 4]> {
    let size = u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize;
    let end = size.clamp(12, b.len().min(64));
    b[8..end].as_chunks::<4>().0.iter().enumerate().filter(|(i, _)| *i != 1).map(|(_, c)| c)
}

fn is_avif_ftyp(b: &[u8]) -> bool {
    ftyp_brands(b).any(|c| c == b"avif" || c == b"avis")
}

/// HEVC-coded HEIF (`heic`, `heix`, ...) or a generic HEIF file (`mif1`, `msf1`, `miaf`) that is
/// not AVIF (checked first). Sequence-only files (`hevc`, `msf1`) are recognised so that opening
/// one says what is unsupported instead of "unrecognized format".
fn is_heif_ftyp(b: &[u8]) -> bool {
    ftyp_brands(b).any(|c| matches!(c, b"heic" | b"heix" | b"heim" | b"heis" | b"hevc" | b"hevx" | b"mif1" | b"msf1" | b"miaf"))
}

fn looks_like_tga(b: &[u8]) -> bool {
    if b.len() < 18 {
        return false;
    }
    if b.len() >= 26 && &b[b.len() - 18..] == b"TRUEVISION-XFILE.\0" {
        return true;
    }
    let cmap_type = b[1];
    let img_type = b[2];
    let cmap_len = u16::from_le_bytes([b[5], b[6]]);
    let cmap_depth = b[7];
    let w = u16::from_le_bytes([b[12], b[13]]);
    let h = u16::from_le_bytes([b[14], b[15]]);
    let depth = b[16];
    let desc = b[17];
    let type_ok = matches!(img_type, 1 | 2 | 3 | 9 | 10 | 11);
    let cmap_ok = match cmap_type {
        0 => cmap_len == 0 && cmap_depth == 0 && !matches!(img_type, 1 | 9),
        1 => matches!(img_type, 1 | 9) && cmap_len > 0 && matches!(cmap_depth, 15 | 16 | 24 | 32),
        _ => false,
    };
    let depth_ok = match img_type {
        1 | 9 => depth == 8,
        3 | 11 => matches!(depth, 8 | 16),
        _ => matches!(depth, 15 | 16 | 24 | 32),
    };
    type_ok && cmap_ok && depth_ok && w > 0 && h > 0 && desc & 0xC0 == 0
}

/// Look up a format by file extension (case-insensitive, with or without a
/// leading dot, or a full file name/path).
pub fn from_extension(ext: &str) -> Option<Format> {
    let ext = ext.rsplit(['.', '/', '\\']).next().unwrap_or(ext).to_ascii_lowercase();
    Format::ALL.into_iter().find(|f| f.extensions().contains(&ext.as_str()))
}
