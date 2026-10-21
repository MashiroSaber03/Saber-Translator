//! Camera raw files via `photocraft-raw`: developed to a 16-bit RGB document
//! in ProPhoto RGB (the embedded profile is the built-in ProPhoto-compatible
//! profile), or, for raw variants not decoded yet, the camera's embedded
//! JPEG preview with a warning.
//!
//! Files without colour calibration (everything but DNG) get their colour
//! from the camera's own embedded JPEG: `photocraft-raw` cannot decode JPEG,
//! so the preview is decoded here and handed to [`photocraft_raw::fit_look`],
//! which fits a matrix and tone curve and keeps the neutral fallback when the
//! fit is not clearly better.

use std::borrow::Cow;
use std::sync::Arc;

use photocraft_cms::{Builtin, Intent, Profile, Transform};
use photocraft_codecs::{self as codecs, ChannelLayout, Format, Image, SampleType};
use photocraft_raw::{CameraLook, DevelopOptions, Limits, PreviewColorSpace, RawError, Reference, Sensor, WhiteBalance};
pub use photocraft_raw::{Tone, retone};

use crate::flat::image_to_document;
use crate::{ImportResult, IoError};

/// `true` if `bytes` are a camera raw file `photocraft-raw` recognises.
pub fn is_raw(bytes: &[u8]) -> bool {
    photocraft_raw::is_raw(bytes)
}

/// The decode limits shared with the flat codecs.
fn limits() -> Limits {
    let l = codecs::Limits::default();
    Limits { max_width: l.max_width, max_height: l.max_height, max_pixels: l.max_pixels, max_alloc: l.max_alloc }
}

/// A raw file's embedded JPEG preview, turned upright. Its own EXIF orientation wins when it
/// has one; otherwise the raw's IFD0 orientation applies (TIFF-based raws record it there, and
/// their previews are usually stored as the sensor reads out). Developed raws are oriented by
/// `photocraft-raw` and carry no EXIF, so nothing is turned twice.
fn upright_preview(raw: &[u8], jpeg: &[u8]) -> Result<Image, IoError> {
    let img = codecs::decode_as_with(Format::Jpeg, jpeg, &codecs::DecodeOptions { keep_orientation: true, ..Default::default() })?;
    let own = img.meta.exif.as_deref().map_or(1, codecs::exif_orientation);
    let o = if own != 1 { own } else { photocraft_raw::orientation(raw).unwrap_or_else(|| codecs::exif_orientation(raw)) };
    Ok(img.oriented(o)?)
}

/// Camera Raw's Temperature, Tint, Exposure, Shadows, Highlights, Clarity, Texture and Dehaze,
/// applied while developing the sensor data (the open-time Camera Raw dialog) instead of to the
/// developed RGB pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RawTuning {
    /// −100…100, as `photocraft_algo::camera_raw::CameraRaw::temperature`.
    pub temperature: f32,
    /// −100…100, as `CameraRaw::tint`.
    pub tint: f32,
    /// EV, as `CameraRaw::exposure`.
    pub exposure: f32,
    /// −100…100, as `CameraRaw::shadows`: in linear light before the tone curve, as Camera Raw
    /// does for raw files ([`photocraft_raw::Tone`]).
    pub shadows: f32,
    /// −100…100, as `CameraRaw::highlights`.
    pub highlights: f32,
    /// −100…100, as `CameraRaw::clarity`.
    pub clarity: f32,
    /// −100…100, as `CameraRaw::texture`.
    pub texture: f32,
    /// −100…100, as `CameraRaw::dehaze`.
    pub dehaze: f32,
}

/// Develops a raw file with `tuning` applied at the raw stage: Exposure as develop exposure,
/// Shadows, Highlights, Clarity, Texture and Dehaze in linear light before the tone curve, Temperature / Tint as gains on the
/// camera's as-shot white-balance multipliers (the same gains the Camera Raw filter applies to
/// RGB). The second value is `false` when the file has
/// no as-shot white balance: Temperature / Tint were then not applied, and the caller keeps
/// them as RGB adjustments.
pub fn import_raw_tuned(name: &str, bytes: &[u8], tuning: &RawTuning) -> Result<(ImportResult, bool), IoError> {
    let format = photocraft_raw::identify(bytes).map(|f| f.name()).unwrap_or("camera raw");
    let exposure = f64::from(tuning.exposure);
    if !exposure.is_finite() || !(-10.0..=10.0).contains(&exposure) {
        return Err(IoError::Raw(RawError::Malformed(format!("exposure {} EV is out of range", tuning.exposure))));
    }
    let sensor = photocraft_raw::decode(bytes, &limits()).map_err(IoError::Raw)?;
    let slider = |v: f32| if v.is_finite() { v.clamp(-100.0, 100.0) } else { 0.0 };
    let tone = photocraft_raw::Tone {
        shadows: slider(tuning.shadows),
        highlights: slider(tuning.highlights),
        clarity: slider(tuning.clarity),
        texture: slider(tuning.texture),
        dehaze: slider(tuning.dehaze),
    };
    // The colour rendering the default open used (a look fitted to the camera's JPEG for files
    // without colour calibration), so re-developing changes only what the sliders change.
    let camera_look = default_look(bytes, &sensor);
    let mut opts = DevelopOptions { limits: limits(), exposure, tone, camera_look, ..Default::default() };
    let wb_applied = match sensor.camera_wb {
        Some(m) => {
            let g = wb_gains(tuning.temperature, tuning.tint);
            opts.white_balance = WhiteBalance::Multipliers([m[0] * g[0], m[1] * g[1], m[2] * g[2]]);
            true
        }
        None => false,
    };
    let dev = photocraft_raw::develop_sensor(&sensor, &opts).map_err(IoError::Raw)?;
    let wb = if wb_applied {
        format!("as-shot white balance, temperature {:+.0}, tint {:+.0}", tuning.temperature, tuning.tint)
    } else {
        "estimated white balance".into()
    };
    let tone = [("shadows", tone.shadows), ("highlights", tone.highlights), ("clarity", tone.clarity), ("texture", tone.texture), ("dehaze", tone.dehaze)]
        .iter()
        .filter(|(_, v)| *v != 0.0)
        .map(|(k, v)| format!(", {k} {v:+.0}"))
        .collect::<String>();
    let summary = format!("developed in Camera Raw (exposure {:+.2} EV{tone}, {wb})", tuning.exposure);
    Ok((developed_document(name, format, dev, &summary)?, wb_applied))
}

/// The tone curve `bytes` develop with by default: the fitted look's for a file without colour
/// calibration (see [`import_raw_with`]), else the camera profile's (empty: none, or not a raw
/// this crate decodes). For previews of the open-time raw-stage sliders
/// ([`photocraft_raw::retone`]), in linear-light knots.
pub fn tone_curve(bytes: &[u8]) -> Vec<[f32; 2]> {
    let Ok(sensor) = photocraft_raw::decode(bytes, &limits()) else { return Vec::new() };
    match default_look(bytes, &sensor) {
        Some(look) => look.tone.as_ref().map(look_knots).unwrap_or_default(),
        None => sensor.tone_curve,
    }
}

/// A fitted look's tone curve as knots for [`photocraft_raw::retone`] (which reads them
/// log-linearly): sampled densely enough that the two agree within a fraction of a code value.
fn look_knots(t: &photocraft_raw::ToneCurve) -> Vec<[f32; 2]> {
    const N: usize = 256;
    (1..=N).map(|i| (i as f64 / N as f64).powi(2)).map(|x| [x as f32, t.eval(x) as f32]).collect()
}

/// The look a default open develops `bytes` with: fitted to the embedded JPEG when the file has
/// no colour calibration, is not monochrome and the fit clearly helps (as [`import_raw_with`]
/// does), else `None`. It is fitted at the camera's own white balance and exposure, so a
/// re-develop's sliders are not fitted away.
fn default_look(bytes: &[u8], sensor: &Sensor) -> Option<CameraLook> {
    if !sensor.color.calibrations.is_empty() || sensor.is_monochrome() {
        return None;
    }
    fit_preview_look(bytes, sensor, &DevelopOptions { limits: limits(), ..Default::default() }).ok()
}

/// "Canon" + "Canon EOS 80D" and "NIKON CORPORATION" + "NIKON D3200" read as the model alone:
/// the model already starts with the make's first word.
fn camera_name(make: Option<&str>, model: Option<&str>) -> String {
    match (make, model) {
        (Some(make), Some(model))
            if make.split_whitespace().next().is_some_and(|brand| model.to_ascii_lowercase().starts_with(&brand.to_ascii_lowercase())) =>
        {
            model.to_string()
        }
        (make, model) => [make, model].into_iter().flatten().collect::<Vec<_>>().join(" "),
    }
}

/// Camera Raw's white-balance gains for Temperature / Tint (−100…100), the same the Camera
/// Raw filter applies ([`photocraft_algo::camera_raw::white_balance_gains`]).
fn wb_gains(temperature: f32, tint: f32) -> [f64; 3] {
    photocraft_algo::camera_raw::white_balance_gains(temperature.clamp(-100.0, 100.0), tint.clamp(-100.0, 100.0)).map(f64::from)
}

/// A developed raw as a 16-bit ProPhoto document, with a note saying how it was developed.
fn developed_document(name: &str, format: &str, dev: photocraft_raw::Developed, summary: &str) -> Result<ImportResult, IoError> {
    let img = Image::from_u16(dev.width, dev.height, ChannelLayout::Rgb, &dev.rgb)?;
    let mut r = image_to_document(name, &img)?;
    r.document.icc_profile = Some(Arc::new(photocraft_cms::Builtin::ProPhotoCompat.profile().to_bytes().to_vec()));
    let camera = camera_name(dev.info.make.as_deref(), dev.info.model.as_deref());
    r.warnings.push(format!(
        "{format}{} {summary} into 16-bit {}",
        if camera.is_empty() { String::new() } else { format!(" from {camera}") },
        photocraft_raw::OUTPUT_SPACE
    ));
    r.warnings.extend(dev.warnings);
    Ok(r)
}

/// Develops a raw file with the default settings.
pub fn import_raw(name: &str, bytes: &[u8]) -> Result<ImportResult, IoError> {
    import_raw_with(name, bytes, &DevelopOptions { limits: limits(), ..Default::default() })
}

/// Smallest preview long edge preferred for the colour fit (a smaller preview decodes faster;
/// the fit grid is at most `PROXY_EDGE` cells and wants about two pixels per cell).
const FIT_PREVIEW_EDGE: u32 = 2 * photocraft_raw::PROXY_EDGE as u32;

/// The colour space of a raw's decoded preview when it is not sRGB: its own ICC profile, else the
/// one the raw's EXIF states (a Sony body set to Adobe RGB writes no profile into the JPEG).
/// `None` means sRGB.
fn preview_profile(raw: &[u8], icc: Option<&[u8]>) -> Result<Option<Cow<'static, Profile>>, String> {
    let src = match icc {
        Some(icc) => Cow::Owned(Profile::parse(icc).map_err(|e| format!("the preview's ICC profile does not parse: {e}"))?),
        None if photocraft_raw::embedded_preview_color_space(raw) == Some(PreviewColorSpace::AdobeRgb) => Cow::Borrowed(Builtin::AdobeRgbCompat.profile()),
        None => return Ok(None),
    };
    Ok((!src.same_colors(Builtin::Srgb.profile())).then_some(src))
}

/// Fits the colour of a decoded raw to the camera's embedded JPEG (see
/// [`photocraft_raw::fit_look`]): picks the smallest preview at least
/// `FIT_PREVIEW_EDGE` pixels long (else the largest), decodes it, turns it upright and
/// converts it to sRGB when it is in another colour space ([`preview_profile`]). `Err` says
/// why there is no fit.
pub fn fit_preview_look(raw: &[u8], sensor: &Sensor, opts: &DevelopOptions) -> Result<CameraLook, String> {
    let previews = photocraft_raw::embedded_previews(raw);
    let long = |p: &photocraft_raw::Preview| p.width.max(p.height);
    let pick = previews
        .iter()
        .filter(|p| long(p) >= FIT_PREVIEW_EDGE)
        .min_by_key(|p| long(p))
        .or_else(|| previews.iter().max_by_key(|p| long(p)))
        .ok_or("no embedded JPEG preview")?;
    let img = upright_preview(raw, pick.jpeg).map_err(|e| format!("the embedded preview does not decode: {e}"))?;
    if img.layout() != ChannelLayout::Rgb || img.sample_type() != SampleType::U8 {
        return Err("the embedded preview is not 8-bit RGB".into());
    }
    let (w, h) = img.dimensions();
    let mut rgb: Cow<[u8]> = Cow::Borrowed(img.data());
    if let Some(src) = preview_profile(raw, img.icc.as_deref())? {
        let t = Transform::new(&src, Builtin::Srgb.profile(), Intent::RelativeColorimetric, false).map_err(|e| e.to_string())?;
        let mut out = vec![0u8; rgb.len()];
        t.convert_u8(&rgb, 3, &mut out, 3, false);
        rgb = Cow::Owned(out);
    }
    photocraft_raw::fit_look(sensor, opts, &Reference { width: w, height: h, rgb: &rgb }).map_err(|e| e.to_string())
}

/// Decodes and develops; a file without colour calibration gets a look fitted to its preview.
fn develop_fitted(bytes: &[u8], opts: &DevelopOptions) -> Result<photocraft_raw::Developed, RawError> {
    let sensor = photocraft_raw::decode(bytes, &opts.limits)?;
    let mut opts = Cow::Borrowed(opts);
    if opts.camera_look.is_none()
        && sensor.color.calibrations.is_empty()
        && !sensor.is_monochrome()
        && let Ok(look) = fit_preview_look(bytes, &sensor, &opts)
    {
        opts.to_mut().camera_look = Some(look);
    }
    photocraft_raw::develop_sensor(&sensor, &opts)
}

/// In the import note of a raw developed from its sensor data with the default settings (not of
/// a preview fallback): what the open-time Camera Raw dialog looks for.
pub const DEVELOPED_NOTE: &str = "developed with default settings";

/// Develops a raw file with explicit settings. Without `opts.camera_look`, a file without
/// colour calibration gets one fitted to its embedded JPEG when that clearly helps.
pub fn import_raw_with(name: &str, bytes: &[u8], opts: &DevelopOptions) -> Result<ImportResult, IoError> {
    let format = photocraft_raw::identify(bytes).map(|f| f.name()).unwrap_or("camera raw");
    match develop_fitted(bytes, opts) {
        Ok(dev) => {
            let summary = format!("{DEVELOPED_NOTE} ({} demosaic, as-shot white balance)", opts.demosaic.id());
            developed_document(name, format, dev, &summary)
        }
        Err(RawError::Unsupported(reason)) => match photocraft_raw::embedded_preview(bytes) {
            Some(p) => {
                let img = upright_preview(bytes, p.jpeg)?;
                let mut r = image_to_document(name, &img)?;
                r.warnings.insert(
                    0,
                    format!(
                        "{format}: {reason} is not supported yet; opened the camera's embedded {}x{} JPEG preview instead (8-bit, not the raw sensor data)",
                        p.width, p.height
                    ),
                );
                Ok(r)
            }
            None => Err(IoError::Raw(RawError::Unsupported(reason))),
        },
        Err(e) => Err(IoError::Raw(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_names_drop_a_repeated_make() {
        assert_eq!(camera_name(Some("Canon"), Some("Canon EOS 80D")), "Canon EOS 80D");
        assert_eq!(camera_name(Some("NIKON CORPORATION"), Some("NIKON D3200")), "NIKON D3200");
        assert_eq!(camera_name(Some("SONY"), Some("ILCE-7M3")), "SONY ILCE-7M3");
        assert_eq!(camera_name(None, Some("X")), "X");
        assert_eq!(camera_name(Some("Panasonic"), Some("DC-G9")), "Panasonic DC-G9");
    }

    /// A Sony body set to Adobe RGB says so only in the raw's EXIF (ColorSpace Uncalibrated, InteropIndex "R03"):
    /// its preview, which has no ICC profile, is read as Adobe RGB for the colour fit, not as sRGB.
    #[test]
    fn a_preview_without_a_profile_takes_the_raws_exif_colour_space() {
        use photocraft_raw::testgen::{TiffBuilder, Val};
        let raw = |color_space: u16, index: &str| {
            let mut t = TiffBuilder::default();
            let interop = t.ifd(vec![(0x0001, Val::Ascii(index.into()))]);
            let exif = t.ifd(vec![(0xA001, Val::Short(vec![color_space])), (0xA005, Val::Ifds(vec![interop]))]);
            t.chain = vec![t.ifd(vec![(271, Val::Ascii("SONY".into())), (34665, Val::Ifds(vec![exif]))])];
            t.build()
        };
        let adobe = raw(0xFFFF, "R03");
        let srgb = raw(1, "R98");
        assert!(preview_profile(&adobe, None).unwrap().unwrap().same_colors(Builtin::AdobeRgbCompat.profile()));
        assert!(preview_profile(&srgb, None).unwrap().is_none());
        assert!(preview_profile(b"not a raw", None).unwrap().is_none(), "unstated: sRGB");
        // the preview's own profile wins over the raw's EXIF
        let srgb_icc = Builtin::Srgb.profile().to_bytes().to_vec();
        assert!(preview_profile(&adobe, Some(srgb_icc.as_slice())).unwrap().is_none());
        let p3_icc = Builtin::DisplayP3.profile().to_bytes().to_vec();
        assert!(preview_profile(&srgb, Some(p3_icc.as_slice())).unwrap().unwrap().same_colors(Builtin::DisplayP3.profile()));
        assert!(preview_profile(&adobe, Some(b"not an ICC profile")).is_err());
    }
}
