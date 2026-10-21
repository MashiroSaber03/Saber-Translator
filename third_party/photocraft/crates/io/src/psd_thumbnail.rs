//! The PSD/PSB thumbnail image resource (1036), written when Preferences › File Handling › Image
//! Previews saves them (Photoshop's default, Always Save).
//!
//! Layout (Adobe PSD spec, "Thumbnail resource format"): a 28-byte big-endian header (format 1 =
//! kJpegRGB, width, height, widthbytes = (width × 24 + 31) / 32 × 4, total size = widthbytes ×
//! height, compressed size, bits per pixel 24, planes 1) followed by the JFIF stream. Photoshop
//! keeps the long side at 160 px or less.

use photocraft_cms::{Builtin, ColorSpace, Intent, Profile, Transform};
use photocraft_codecs::{ChannelLayout, EncodeOptions, Format, Image, SampleType as CSample};
use photocraft_color::ColorMode;
use photocraft_doc::Document;
use photocraft_psd::resources::{ImageResource, ids};

use crate::IoError;

/// Longest side of the thumbnail, as Photoshop writes it.
pub const MAX_SIDE: u32 = 160;

/// Size of the header in front of the JFIF data.
pub const HEADER_LEN: usize = 28;

/// The 1036 resource for `doc`: its composite reduced to at most [`MAX_SIDE`], converted to
/// sRGB through the document's profile, matted over white and JPEG-compressed.
pub(crate) fn resource(doc: &Document) -> Result<ImageResource, IoError> {
    let mut buf = photocraft_compose::thumbnail_buffer(doc, MAX_SIDE);
    let (w, h) = (buf.rect.width(), buf.rect.height());
    if w == 0 || h == 0 || w > MAX_SIDE || h > MAX_SIDE || buf.px.len() != (w as usize) * (h as usize) {
        return Err(IoError::Unsupported("the image preview could not be rendered".into()));
    }
    let src = composite_profile(doc);
    let dst = Builtin::Srgb.profile();
    if !src.same_colors(dst) {
        let t = Transform::new(&src, dst, Intent::RelativeColorimetric, true).map_err(|e| IoError::Unsupported(e.to_string()))?;
        t.apply(buf.px.as_flattened_mut(), 4);
    }
    let mut rgb = Vec::with_capacity(buf.px.len() * 3);
    for p in &buf.px {
        let a = finite(p[3]);
        for &c in &p[..3] {
            // Over white, as Photoshop shows a transparent document's preview.
            let v = finite(c) * a + (1.0 - a);
            rgb.push((v * 255.0).round() as u8);
        }
    }
    let img = Image::from_raw(w, h, ChannelLayout::Rgb, CSample::U8, rgb)?;
    let opts = EncodeOptions { jpeg_quality: 80, embed_icc: false, embed_metadata: false, ..EncodeOptions::default() };
    let jfif = photocraft_codecs::encode(&img, Format::Jpeg, &opts)?;
    Ok(ImageResource::new(ids::THUMBNAIL, thumbnail_bytes(w, h, &jfif)?))
}

/// A sample clamped to 0–1 (NaN as 0): 32-bit composites can hold values past white.
fn finite(v: f32) -> f32 {
    if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) }
}

/// Header plus JFIF data of a `w`×`h` thumbnail resource.
fn thumbnail_bytes(w: u32, h: u32, jfif: &[u8]) -> Result<Vec<u8>, IoError> {
    let too_big = || IoError::Unsupported("image preview too large".into());
    let width_bytes = w.checked_mul(24).map(|v| v.div_ceil(32) * 4).ok_or_else(too_big)?;
    let total = width_bytes.checked_mul(h).ok_or_else(too_big)?;
    let compressed = u32::try_from(jfif.len()).map_err(|_| too_big())?;
    let mut out = Vec::with_capacity(HEADER_LEN + jfif.len());
    for v in [1, w, h, width_bytes, total, compressed] {
        out.extend_from_slice(&v.to_be_bytes());
    }
    out.extend_from_slice(&24u16.to_be_bytes());
    out.extend_from_slice(&1u16.to_be_bytes());
    out.extend_from_slice(jfif);
    Ok(out)
}

/// The profile of the compositor's RGB output for `doc`: an RGB document's own (embedded when it
/// parses, else sRGB), a gray document's gray profile seen as RGB, and sRGB for CMYK and Lab
/// (which the compositor renders in sRGB).
fn composite_profile(doc: &Document) -> Profile {
    let space = match doc.mode {
        ColorMode::Grayscale | ColorMode::Bitmap | ColorMode::Duotone => ColorSpace::Gray,
        ColorMode::Cmyk | ColorMode::Lab => return Builtin::Srgb.profile().clone(),
        _ => ColorSpace::Rgb,
    };
    let own = doc.icc_profile.as_ref().and_then(|b| Profile::parse(b).ok()).filter(|p| p.color_space == space);
    match space {
        ColorSpace::Gray => {
            own.and_then(|p| p.gray_as_rgb()).or_else(|| Builtin::SGray.profile().gray_as_rgb()).unwrap_or_else(|| Builtin::Srgb.profile().clone())
        }
        _ => own.unwrap_or_else(|| Builtin::Srgb.profile().clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_follows_the_spec() {
        let b = thumbnail_bytes(160, 90, &[0xFF, 0xD8, 0xFF, 0xD9]).unwrap();
        let u = |i: usize| u32::from_be_bytes(b[i..i + 4].try_into().unwrap());
        assert_eq!((u(0), u(4), u(8), u(12), u(16), u(20)), (1, 160, 90, 480, 480 * 90, 4));
        assert_eq!(&b[24..28], &[0, 24, 0, 1]);
        assert_eq!(&b[28..], &[0xFF, 0xD8, 0xFF, 0xD9]);
        // Width bytes are padded to 4: 3 px × 3 bytes = 9 → 12.
        let odd = thumbnail_bytes(3, 1, &[]).unwrap();
        assert_eq!(u32::from_be_bytes(odd[12..16].try_into().unwrap()), 12);
    }
}
