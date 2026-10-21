//! WebP: decode lossy/lossless (first frame) via `image-webp`; encode lossless via
//! `image-webp`'s VP8L encoder, or lossy with our own VP8 encoder (`super::vp8`) wrapped in the
//! RIFF container written here (a bare `VP8 ` chunk, or `VP8X` with `ALPH` alpha and `ICCP` /
//! `EXIF` / `XMP ` chunks). ICC/EXIF/XMP both ways.

use std::io::Cursor;

use crate::Format;
use crate::error::CodecError;
use crate::fidelity::Plan;
use crate::image::{ChannelLayout, DecodeWarning, Image, Metadata, SampleType};
use crate::options::{EncodeOptions, Limits, alloc_zeroed};

const F: Format = Format::WebP;

fn err(e: impl std::fmt::Display) -> CodecError {
    CodecError::malformed(F, e)
}

pub(crate) fn decode(bytes: &[u8], limits: &Limits) -> Result<Image, CodecError> {
    let mut dec = image_webp::WebPDecoder::new(Cursor::new(bytes)).map_err(err)?;
    dec.set_memory_limit(limits.alloc_usize());
    let (w, h) = dec.dimensions();
    let layout = if dec.has_alpha() { ChannelLayout::Rgba } else { ChannelLayout::Rgb };
    limits.check(w, h, layout, SampleType::U8)?;
    // VP8L back-references and VP8 skipped macroblocks compress flat areas extremely well, so the
    // ratio is generous; it still stops a header-only file from asking for a gigabyte.
    Limits::check_plausible(u64::from(w) * u64::from(h) * layout.channels() as u64, bytes.len(), 1 << 16)?;
    let size = dec.output_buffer_size().ok_or_else(|| err("image too large"))?;
    let mut buf = alloc_zeroed::<u8>(size)?;
    dec.read_image(&mut buf).map_err(err)?;
    let icc = dec.icc_profile().ok().flatten();
    let exif = dec.exif_metadata().ok().flatten();
    let xmp = dec.xmp_metadata().ok().flatten().and_then(|b| String::from_utf8(b).ok());
    let mut img = Image::from_u8(w, h, layout, buf)?;
    img.icc = icc;
    img.meta = Metadata { exif, xmp, ..Default::default() };
    if dec.is_animated() && dec.num_frames() > 1 {
        img.warnings.push(DecodeWarning::MoreFrames { total: Some(dec.num_frames()) });
    }
    Ok(img)
}

pub(crate) fn encode(src: &Image, plan: Plan, opts: &EncodeOptions) -> Result<Vec<u8>, CodecError> {
    let img = src.converted(plan.layout, plan.sample);
    if !opts.webp_lossless {
        return encode_lossy(&img, opts);
    }
    let ct = match img.layout() {
        ChannelLayout::Gray => image_webp::ColorType::L8,
        ChannelLayout::GrayA => image_webp::ColorType::La8,
        ChannelLayout::Rgb => image_webp::ColorType::Rgb8,
        ChannelLayout::Rgba => image_webp::ColorType::Rgba8,
        l => return Err(CodecError::encode(F, format!("unsupported layout {l:?}"))),
    };
    let mut out = Vec::new();
    let mut enc = image_webp::WebPEncoder::new(&mut out);
    if opts.embed_icc
        && let Some(icc) = &img.icc
    {
        enc.set_icc_profile(icc.clone());
    }
    if opts.embed_metadata {
        if let Some(exif) = &img.meta.exif {
            // The pixels are written as they are shown: never let a viewer rotate them again.
            enc.set_exif_metadata(crate::resolution::export_exif(exif, img.meta.dpi).into_owned());
        }
        if let Some(xmp) = &img.meta.xmp {
            enc.set_xmp_metadata(crate::resolution::export_xmp(xmp, img.meta.dpi).as_bytes().to_vec());
        }
    }
    enc.encode(img.data(), img.width(), img.height(), ct).map_err(|e| CodecError::encode(F, e))?;
    Ok(out)
}

/// Lossy: the picture as a VP8 key frame, alpha (when any pixel is translucent) as an
/// uncompressed `ALPH` chunk, metadata as chunks.
fn encode_lossy(img: &Image, opts: &EncodeOptions) -> Result<Vec<u8>, CodecError> {
    use super::vp8;
    let (w, h) = img.dimensions();
    if w > vp8::MAX_DIMENSION || h > vp8::MAX_DIMENSION {
        return Err(CodecError::encode(F, format!("lossy WebP holds at most {} pixels per side", vp8::MAX_DIMENSION)));
    }
    // The VP8 path takes 8-bit RGB(A); gray becomes RGB with flat chroma.
    let has_alpha = img.layout().has_alpha();
    let target = if has_alpha { ChannelLayout::Rgba } else { ChannelLayout::Rgb };
    let rgb = img.converted(target, SampleType::U8);
    let stride = target.channels();
    let yuv = vp8::Yuv::from_rgb(rgb.data(), w, h, stride);
    let frame = vp8::encode_frame(&yuv, vp8::Params { quality: opts.webp_quality.min(100) }).map_err(|e| CodecError::encode(F, e))?;
    let alpha: Option<Vec<u8>> = if has_alpha {
        let a: Vec<u8> = rgb.data().iter().skip(3).step_by(4).copied().collect();
        if a.iter().all(|&v| v == 255) { None } else { Some(a) }
    } else {
        None
    };
    let icc = if opts.embed_icc { img.icc.as_deref() } else { None };
    let exif = if opts.embed_metadata { img.meta.exif.as_deref().map(|e| crate::resolution::export_exif(e, img.meta.dpi)) } else { None };
    let xmp = if opts.embed_metadata { img.meta.xmp.as_deref().map(|x| crate::resolution::export_xmp(x, img.meta.dpi)) } else { None };
    Ok(riff(w, h, &frame, alpha.as_deref(), icc, exif.as_deref(), xmp.as_deref().map(str::as_bytes)))
}

/// A RIFF chunk: FourCC, little-endian size, payload, padded to an even length.
fn chunk(out: &mut Vec<u8>, fourcc: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(fourcc);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        out.push(0);
    }
}

/// The WebP container around a VP8 frame: the simple form when there is nothing else, else the
/// extended form (`VP8X`) with the alpha plane and the metadata chunks in the specified order.
fn riff(w: u32, h: u32, frame: &[u8], alpha: Option<&[u8]>, icc: Option<&[u8]>, exif: Option<&[u8]>, xmp: Option<&[u8]>) -> Vec<u8> {
    let mut body = b"WEBP".to_vec();
    let extended = alpha.is_some() || icc.is_some() || exif.is_some() || xmp.is_some();
    if extended {
        let mut flags = 0u8;
        if icc.is_some() {
            flags |= 0x20;
        }
        if alpha.is_some() {
            flags |= 0x10;
        }
        if exif.is_some() {
            flags |= 0x08;
        }
        if xmp.is_some() {
            flags |= 0x04;
        }
        let mut vp8x = vec![flags, 0, 0, 0];
        vp8x.extend_from_slice(&(w - 1).to_le_bytes()[..3]);
        vp8x.extend_from_slice(&(h - 1).to_le_bytes()[..3]);
        chunk(&mut body, b"VP8X", &vp8x);
        if let Some(icc) = icc {
            chunk(&mut body, b"ICCP", icc);
        }
        if let Some(a) = alpha {
            // Header byte: reserved, no pre-processing, no filtering, no compression.
            let mut alph = Vec::with_capacity(a.len() + 1);
            alph.push(0);
            alph.extend_from_slice(a);
            chunk(&mut body, b"ALPH", &alph);
        }
    }
    chunk(&mut body, b"VP8 ", frame);
    if let Some(exif) = exif {
        chunk(&mut body, b"EXIF", exif);
    }
    if let Some(xmp) = xmp {
        chunk(&mut body, b"XMP ", xmp);
    }
    let mut out = Vec::with_capacity(body.len() + 8);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}
