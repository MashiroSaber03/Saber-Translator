//! Formats delegated to the `image` crate (pure-Rust features only):
//! GIF, BMP, TGA, ICO, QOI, Radiance HDR, and AVIF encode (feature `avif`).

use std::io::Cursor;

use image::{DynamicImage, ImageDecoder, ImageFormat};

use crate::Format;
use crate::error::CodecError;
use crate::fidelity::Plan;
use crate::image::{ChannelLayout, DecodeWarning, Image, SampleType};
use crate::options::{EncodeOptions, Limits};

fn image_format(f: Format) -> Option<ImageFormat> {
    Some(match f {
        Format::Gif => ImageFormat::Gif,
        Format::Bmp => ImageFormat::Bmp,
        Format::Tga => ImageFormat::Tga,
        Format::Ico => ImageFormat::Ico,
        Format::Qoi => ImageFormat::Qoi,
        Format::Hdr => ImageFormat::Hdr,
        Format::Avif => ImageFormat::Avif,
        _ => return None,
    })
}

fn map_err(f: Format, e: image::ImageError) -> CodecError {
    match e {
        image::ImageError::Limits(l) => CodecError::LimitExceeded(l.to_string()),
        image::ImageError::Unsupported(u) => CodecError::unsupported(f, u.to_string()),
        e => CodecError::malformed(f, e),
    }
}

pub(crate) fn bmp_palette(bytes: &[u8]) -> Result<Option<Vec<[u8; 3]>>, CodecError> {
    let decoder = image::codecs::bmp::BmpDecoder::new(Cursor::new(bytes)).map_err(|e| map_err(Format::Bmp, e))?;
    Ok(decoder.get_palette().map(|palette| palette.to_vec()))
}

pub(crate) fn decode(f: Format, bytes: &[u8], limits: &Limits) -> Result<Image, CodecError> {
    let fmt = image_format(f).ok_or_else(|| CodecError::unsupported(f, "not handled by image backend"))?;
    let mut reader = image::ImageReader::with_format(Cursor::new(bytes), fmt);
    let mut il = image::Limits::default();
    il.max_image_width = Some(limits.max_width);
    il.max_image_height = Some(limits.max_height);
    il.max_alloc = Some(limits.max_alloc);
    reader.limits(il);
    let mut dec = reader.into_decoder().map_err(|e| map_err(f, e))?;
    let (w, h) = dec.dimensions();
    limits.check_bytes(w, h, dec.color_type().bytes_per_pixel() as u64)?;
    let icc = dec.icc_profile().ok().flatten();
    let exif = dec.exif_metadata().ok().flatten();
    let dynimg = DynamicImage::from_decoder(dec).map_err(|e| map_err(f, e))?;
    let mut img = from_dynamic(f, dynimg)?;
    img.icc = icc;
    img.meta.exif = exif;
    if f == Format::Gif {
        img.warnings.extend(gif_more_frames(bytes));
    }
    Ok(img)
}

/// Counts a GIF's frames (image descriptors) by walking its blocks, without decompressing any.
/// The total is unknown when the blocks end before the trailer or break off.
fn gif_more_frames(b: &[u8]) -> Option<DecodeWarning> {
    // Size of the colour table a packed-fields byte announces.
    let table = |flags: u8| if flags & 0x80 != 0 { 3usize << ((flags & 7) + 1) } else { 0 };
    // Skips the data sub-blocks starting at `i` (each length-prefixed, ended by a zero length).
    let skip = |mut i: usize| -> Option<usize> {
        loop {
            let n = usize::from(*b.get(i)?);
            i = i.checked_add(1 + n)?;
            if n == 0 {
                return Some(i);
            }
        }
    };
    // Header (6 bytes), logical screen descriptor (7) and the global colour table.
    let mut i = 13 + table(*b.get(10)?);
    let mut frames = 0u32;
    let total = loop {
        let next = match b.get(i) {
            Some(0x3B) => break Some(frames),
            Some(0x21) => skip(i + 2),
            Some(0x2C) => {
                frames = frames.saturating_add(1);
                // Descriptor (10 bytes, packed fields last), local colour table, LZW code size.
                b.get(i + 9).and_then(|&flags| skip(i + 11 + table(flags)))
            }
            _ => None,
        };
        match next {
            Some(n) => i = n,
            None => break None,
        }
    };
    (frames > 1).then_some(DecodeWarning::MoreFrames { total })
}

pub(crate) fn from_dynamic(f: Format, d: DynamicImage) -> Result<Image, CodecError> {
    let (w, h) = (d.width(), d.height());
    Ok(match d {
        DynamicImage::ImageLuma8(b) => Image::from_u8(w, h, ChannelLayout::Gray, b.into_raw())?,
        DynamicImage::ImageLumaA8(b) => Image::from_u8(w, h, ChannelLayout::GrayA, b.into_raw())?,
        DynamicImage::ImageRgb8(b) => Image::from_u8(w, h, ChannelLayout::Rgb, b.into_raw())?,
        DynamicImage::ImageRgba8(b) => Image::from_u8(w, h, ChannelLayout::Rgba, b.into_raw())?,
        DynamicImage::ImageLuma16(b) => Image::from_u16(w, h, ChannelLayout::Gray, &b.into_raw())?,
        DynamicImage::ImageLumaA16(b) => Image::from_u16(w, h, ChannelLayout::GrayA, &b.into_raw())?,
        DynamicImage::ImageRgb16(b) => Image::from_u16(w, h, ChannelLayout::Rgb, &b.into_raw())?,
        DynamicImage::ImageRgba16(b) => Image::from_u16(w, h, ChannelLayout::Rgba, &b.into_raw())?,
        DynamicImage::ImageRgb32F(b) => Image::from_f32(w, h, ChannelLayout::Rgb, &b.into_raw())?,
        DynamicImage::ImageRgba32F(b) => Image::from_f32(w, h, ChannelLayout::Rgba, &b.into_raw())?,
        other => {
            let b = other.into_rgba32f();
            Image::from_f32(w, h, ChannelLayout::Rgba, &b.into_raw()).map_err(|e| CodecError::malformed(f, e))?
        }
    })
}

fn to_dynamic(img: &Image) -> Result<DynamicImage, CodecError> {
    let (w, h) = img.dimensions();
    let bad = || CodecError::InvalidImage("buffer size mismatch".into());
    Ok(match (img.layout(), img.sample_type()) {
        (ChannelLayout::Gray, SampleType::U8) => DynamicImage::ImageLuma8(image::GrayImage::from_raw(w, h, img.data().to_vec()).ok_or_else(bad)?),
        (ChannelLayout::GrayA, SampleType::U8) => DynamicImage::ImageLumaA8(image::GrayAlphaImage::from_raw(w, h, img.data().to_vec()).ok_or_else(bad)?),
        (ChannelLayout::Rgb, SampleType::U8) => DynamicImage::ImageRgb8(image::RgbImage::from_raw(w, h, img.data().to_vec()).ok_or_else(bad)?),
        (ChannelLayout::Rgba, SampleType::U8) => DynamicImage::ImageRgba8(image::RgbaImage::from_raw(w, h, img.data().to_vec()).ok_or_else(bad)?),
        (ChannelLayout::Rgb, SampleType::F32) => {
            DynamicImage::ImageRgb32F(image::Rgb32FImage::from_raw(w, h, img.to_f32_samples().unwrap_or_default()).ok_or_else(bad)?)
        }
        (ChannelLayout::Rgba, SampleType::F32) => {
            DynamicImage::ImageRgba32F(image::Rgba32FImage::from_raw(w, h, img.to_f32_samples().unwrap_or_default()).ok_or_else(bad)?)
        }
        (l, s) => {
            return Err(CodecError::InvalidImage(format!("no image-crate mapping for {l:?} {s:?}")));
        }
    })
}

pub(crate) fn encode(f: Format, src: &Image, plan: Plan, opts: &EncodeOptions) -> Result<Vec<u8>, CodecError> {
    let fmt = image_format(f).ok_or_else(|| CodecError::unsupported(f, "not handled by image backend"))?;
    if let Some((mw, mh)) = f.max_dimensions()
        && (src.width() > mw || src.height() > mh)
    {
        return Err(CodecError::encode(f, format!("dimensions exceed {mw}x{mh}")));
    }
    let img = src.converted(plan.layout, plan.sample);
    let dynimg = to_dynamic(&img)?;
    let mut cursor = Cursor::new(Vec::new());
    match f {
        #[cfg(feature = "avif")]
        Format::Avif => {
            let enc = image::codecs::avif::AvifEncoder::new_with_speed_quality(&mut cursor, 8, opts.jpeg_quality.clamp(1, 100));
            dynimg.write_with_encoder(enc).map_err(|e| CodecError::encode(f, e))?;
        }
        _ => {
            let _ = opts;
            dynimg.write_to(&mut cursor, fmt).map_err(|e| CodecError::encode(f, e))?;
        }
    }
    Ok(cursor.into_inner())
}
