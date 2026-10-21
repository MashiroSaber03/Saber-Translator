//! `photocraft-codecs`: symmetric, depth- and metadata-preserving codecs for
//! flat raster image formats.
//!
//! * Every format we **write** we can also **read** (see
//!   [`ASYMMETRIC_EXCEPTIONS`] for the documented, feature-gated exception).
//! * Bit depth (8/16-bit integer, 16/32-bit float), ICC profiles and basic
//!   metadata (EXIF, XMP, DPI, text) are preserved where the format allows.
//! * [`caps`] declares per-format capabilities and [`fidelity_warnings`]
//!   tells the UI exactly what an export will lose.
//! * The API is I/O free (`&[u8]` in, `Vec<u8>` out) and builds for
//!   `wasm32-unknown-unknown`. Decoders enforce [`Limits`] before allocating.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

mod codecs;
mod error;
pub mod exif_color;
mod fidelity;
mod format;
mod image;
mod options;
pub mod orientation;
pub mod resolution;
pub mod source_settings;
pub mod web;

pub use crate::codecs::exr_cryptomatte::{cryptomatte_id, cryptomatte_key, cryptomatte_preview_color};
pub use crate::codecs::gif_frames::{AnimationFrame, MAX_GIF_FRAMES};
pub use crate::codecs::png::{PngPalette, encode_indexed as encode_png_indexed, palette as png_palette};
pub use crate::codecs::tiff::{PhotoshopTags as TiffPhotoshopTags, photoshop_tags as tiff_photoshop_tags, writes_little_endian as tiff_writes_little_endian};
pub use crate::codecs::tiff_ifd::{TiffInfo, TiffPage, TiffPageKind};
pub use crate::error::CodecError;
pub use crate::exif_color::{EXIF_COLOR_SPACE_SRGB, exif_color_space};
pub use crate::fidelity::{FidelityWarning, fidelity_warnings, fidelity_warnings_with};
pub use crate::format::{ASYMMETRIC_EXCEPTIONS, Format, FormatCaps, caps, detect, from_extension};
pub use crate::image::{
    ChannelLayout, CryptomatteBuffer, CryptomatteLayer, DecodeWarning, DeepChannel, DeepImage, ExrChannelInfo, ExrPartInfo, Image, Metadata, SampleType,
};
pub use crate::options::{DecodeOptions, DpxBits, EncodeOptions, ExrCompression, Limits, PngCompression, TiffCompression};
pub use crate::orientation::{exif_orientation, upright_exif, upright_xmp};
pub use crate::resolution::{exif_resolution, export_exif, export_xmp, photoshop_resolution, xmp_resolution};
pub use crate::source_settings::{WebpEncoding, jpeg_quality, webp_encoding};
pub use half::f16;

use crate::codecs::{dds, dpx, exr, fits, heif, jpeg, pcx, png, pnm, tiff, via_image, webp, xpm};

/// Detect the format and decode with default [`Limits`].
pub fn decode(bytes: &[u8]) -> Result<Image, CodecError> {
    decode_with(bytes, &DecodeOptions::default())
}

/// Detect the format and decode.
pub fn decode_with(bytes: &[u8], opts: &DecodeOptions) -> Result<Image, CodecError> {
    let format = detect(bytes).ok_or(CodecError::UnknownFormat)?;
    decode_as_with(format, bytes, opts)
}

/// Decode as a specific format with default [`Limits`].
pub fn decode_as(format: Format, bytes: &[u8]) -> Result<Image, CodecError> {
    decode_as_with(format, bytes, &DecodeOptions::default())
}

/// Decode as a specific format.
pub fn decode_as_with(format: Format, bytes: &[u8], opts: &DecodeOptions) -> Result<Image, CodecError> {
    // HEIF without the `heif` feature: its decoder stub says what is missing from this build.
    if !caps(format).read && format != Format::Heif {
        return Err(CodecError::unsupported(format, "decoding is not available for this format"));
    }
    let l = &opts.limits;
    let img = match format {
        Format::Png => png::decode(bytes, l),
        Format::Jpeg => jpeg::decode(bytes, l),
        Format::Tiff => tiff::decode(bytes, l),
        Format::WebP => webp::decode(bytes, l),
        Format::Pnm => pnm::decode(bytes, l),
        Format::Pcx => pcx::decode(bytes, l),
        Format::Xpm => xpm::decode(bytes, l),
        Format::Dds => dds::decode(bytes, l),
        Format::Fits => fits::decode(bytes, l),
        Format::Dpx => dpx::decode(bytes, l),
        Format::OpenExr => exr::decode(bytes, l),
        Format::Heif => heif::decode(bytes, l, opts.keep_orientation),
        Format::Gif | Format::Bmp | Format::Tga | Format::Ico | Format::Qoi | Format::Hdr | Format::Avif => via_image::decode(format, bytes, l),
    }?;
    // Without a resolution of the format's own (PNG pHYs, TIFF tags, JPEG's metadata), the
    // XMP or EXIF one counts, like Photoshop: a WebP, HEIF or PNG from a camera or phone
    // opens at its recorded resolution, not at the 72 ppi default.
    let mut img = img;
    if img.meta.dpi.is_none() && format != Format::Tiff {
        img.meta.dpi = resolution::from_metadata(img.meta.exif.as_deref(), img.meta.xmp.as_deref());
    }
    // Turn the pixels upright, like Photoshop: a TIFF records it in the decoded page's own
    // directory, the others in their EXIF block. The metadata is rewritten to Orientation = 1
    // on the way. HEIF keeps it in its container, and its decoder has already applied it.
    let o = match format {
        Format::Tiff => tiff::orientation(bytes, None),
        Format::Heif => 1,
        Format::Dpx => dpx::orientation(bytes),
        _ => img.meta.exif.as_deref().map_or(1, exif_orientation),
    };
    finish_decode(img, o, opts)
}

/// Returns the embedded colour table for an indexed BMP, if it has one.
pub fn bmp_palette(bytes: &[u8]) -> Result<Option<Vec<[u8; 3]>>, CodecError> {
    via_image::bmp_palette(bytes)
}

/// Lists the pages (image directories) of a TIFF or BigTIFF file without decoding pixels: the
/// main IFD chain and its SubIFDs, each with its size, kind (page, reduced-resolution copy or
/// mask) and storage. [`TiffInfo::default_page`] is the page [`decode`] opens.
pub fn tiff_info(bytes: &[u8]) -> Result<TiffInfo, CodecError> {
    crate::codecs::tiff_ifd::tiff_info(bytes)
}

/// Decodes page `page` (an index into [`tiff_info`]'s `pages`) of a TIFF or BigTIFF file,
/// applying that page's orientation unless `opts.keep_orientation`.
pub fn decode_tiff_page(bytes: &[u8], page: usize, opts: &DecodeOptions) -> Result<Image, CodecError> {
    let img = tiff::decode_page(bytes, Some(page), &opts.limits)?;
    finish_decode(img, tiff::orientation(bytes, Some(page)), opts)
}

/// Decodes every frame of a GIF (a still GIF yields one), each a full RGBA8 picture of the
/// logical screen with its delay. [`decode`] keeps the first frame only. See
/// [`MAX_GIF_FRAMES`] and `opts.limits` for what is refused.
pub fn decode_gif_frames(bytes: &[u8], opts: &DecodeOptions) -> Result<Vec<AnimationFrame>, CodecError> {
    crate::codecs::gif_frames::decode(bytes, opts)
}

/// The Orientation (1–8) of the page [`decode`] opens from a TIFF or BigTIFF file; 1 when it is
/// absent, malformed or out of range.
pub fn tiff_orientation(bytes: &[u8]) -> u16 {
    tiff::orientation(bytes, None)
}

/// The Orientation (1–8) of page `page` (an index into [`tiff_info`]'s `pages`); 1 when it is
/// absent, malformed or out of range, or the page does not exist.
pub fn tiff_page_orientation(bytes: &[u8], page: usize) -> u16 {
    tiff::orientation(bytes, Some(page))
}

/// The limits check and the orientation, shared by every decoder.
fn finish_decode(img: Image, o: u16, opts: &DecodeOptions) -> Result<Image, CodecError> {
    let l = &opts.limits;
    // Final guard for decoders whose header we could not pre-inspect.
    l.check(img.width(), img.height(), img.layout(), img.sample_type())?;
    if opts.keep_orientation {
        return Ok(img);
    }
    // Orientations 5–8 swap width and height: with asymmetric limits a stored landscape
    // that fit can turn into a portrait that does not, so check the upright size too.
    if (5..=8).contains(&o) {
        l.check(img.height(), img.width(), img.layout(), img.sample_type())?;
    }
    img.oriented(o)
}

/// Lists the parts of an OpenEXR file without decoding pixels: header facts per part
/// (name, view, size, channels, deep/tiled), what a multi-part chooser — a Maya/Arnold
/// render writes one part per AOV — offers. The indices are what [`decode_exr_part`]
/// takes; [`decode_as_with`] opens the part with the highest [`ExrPartInfo::color_rank`].
pub fn exr_info(bytes: &[u8], limits: &Limits) -> Result<Vec<ExrPartInfo>, CodecError> {
    exr::info(bytes, limits)
}

/// Decodes part `part` (an index into [`exr_info`]'s list) of an OpenEXR file.
/// Deep parts stay with [`decode_deep_exr`]; a file that has any is refused here.
pub fn decode_exr_part(bytes: &[u8], part: usize, opts: &DecodeOptions) -> Result<Image, CodecError> {
    let img = exr::decode_part(bytes, part, &opts.limits)?;
    finish_decode(img, 1, opts)
}

/// Lists every Cryptomatte layer of an EXR file (specification 1.2: object/material IDs
/// with coverage, as written by Arnold, V-Ray, Redshift, Mantra/Karma and Cycles) from the
/// headers alone: name, metadata key, channels and the parsed embedded manifest. The
/// samples come from [`decode_cryptomatte`]; names hash to IDs with [`cryptomatte_id`].
pub fn cryptomatte_layers(bytes: &[u8], limits: &Limits) -> Result<Vec<CryptomatteLayer>, CodecError> {
    codecs::exr_cryptomatte::layers(bytes, limits)
}

/// Decodes the samples of the Cryptomatte layer called `layer_name` (a
/// [`CryptomatteLayer::name`]): per pixel its (ID, coverage) pairs, coverage above zero,
/// sorted by descending coverage.
pub fn decode_cryptomatte(bytes: &[u8], layer_name: &str, limits: &Limits) -> Result<CryptomatteBuffer, CodecError> {
    codecs::exr_cryptomatte::decode(bytes, layer_name, limits)
}

/// Decodes the named channels of one part (exact names as the file stores them, e.g.
/// `diffuse.R`) into an image: R/G/B (plus A when named) give Rgba, a lone Y gives Gray.
/// This renders a channel group without the part's other channels in the way.
pub fn decode_exr_channels(bytes: &[u8], part: usize, names: &[&str], opts: &DecodeOptions) -> Result<Image, CodecError> {
    let img = exr::decode_channels(bytes, part, names, &opts.limits)?;
    finish_decode(img, 1, opts)
}

/// Deep OpenEXR samples (`deepscanline`/`deeptile`): the structured per-pixel sample lists
/// (colour, alpha and Z per sample), for callers that composite deep data themselves —
/// [`decode_as_with`] instead flattens a deep file into a normal [`Image`] with a
/// [`DecodeWarning::DeepFlattened`]. `Err(Unsupported)` when the file has no deep part.
pub fn decode_deep_exr(bytes: &[u8], limits: &Limits) -> Result<DeepImage, CodecError> {
    let mut cur = std::io::Cursor::new(bytes);
    let meta = ::exr::meta::MetaData::read_from_buffered(&mut cur, false).map_err(|e| CodecError::malformed(Format::OpenExr, e))?;
    codecs::deep_exr::decode_deep(&meta, bytes, limits)
}

/// Encode `image` as `format`. Conversions follow the same plan that
/// [`fidelity_warnings_with`] reports; fatal conditions (unsupported
/// format, oversize image) return an error.
pub fn encode(image: &Image, format: Format, opts: &EncodeOptions) -> Result<Vec<u8>, CodecError> {
    if !caps(format).write {
        return Err(CodecError::unsupported(format, "encoding is not available in this build"));
    }
    if image.width() == 0 || image.height() == 0 {
        return Err(CodecError::InvalidImage("cannot encode an empty image".into()));
    }
    if let Some((mw, mh)) = format.max_dimensions()
        && (image.width() > mw || image.height() > mh)
    {
        return Err(CodecError::encode(format, format!("dimensions exceed {mw}x{mh}")));
    }
    let plan = fidelity::plan(image, format, opts);
    match format {
        Format::Png => png::encode(image, plan, opts),
        Format::Jpeg => jpeg::encode(image, plan, opts),
        Format::Tiff => tiff::encode(image, plan, opts),
        Format::WebP => webp::encode(image, plan, opts),
        Format::Pnm => pnm::encode(image, plan, opts),
        Format::Pcx => Err(CodecError::unsupported(format, "PCX is import-only; save as PNG instead")),
        Format::Xpm => Err(CodecError::unsupported(format, "XPM is import-only; save as PNG instead")),
        Format::Dds => Err(CodecError::unsupported(format, "DDS is import-only; save as PNG instead")),
        Format::Fits => fits::encode(image, plan, opts),
        Format::Dpx => dpx::encode(image, plan, opts),
        Format::OpenExr => exr::encode(image, plan, opts),
        Format::Heif => Err(CodecError::unsupported(format, "encoding is not available in this build")),
        Format::Gif | Format::Bmp | Format::Tga | Format::Ico | Format::Qoi | Format::Hdr | Format::Avif => via_image::encode(format, image, plan, opts),
    }
}
