//! Flat raster formats via `photocraft-codecs`.

use std::sync::Arc;

use photocraft_codecs::{self as codecs, ChannelLayout, Format, Image, SampleType as CSample};
use photocraft_color::{BlendMode, ColorMode, PixelFormat, SampleType};
use photocraft_doc::{AlphaChannel, Document, Layer, LayerContent};
use photocraft_geom::{Rect, Size, TILE_SIZE};
use photocraft_raster::Surface;

use crate::{ExportOptions, ExportResult, ImportResult, IoError, TgaBits, XmpEmbed};

/// Bytes per band when converting or exporting a band of rows at a time.
const BAND_BYTES: usize = 32 << 20;

/// Decodes a flat image into a single-layer document; a TIFF with Photoshop layer data opens
/// layered (see [`crate::tiff_layers`]).
pub fn import_flat(name: &str, bytes: &[u8]) -> Result<ImportResult, IoError> {
    if codecs::detect(bytes) == Some(Format::Tiff) {
        return import_tiff_page(name, bytes, None);
    }
    let img = codecs::decode(bytes)?;
    // A palette BMP or PNG opens in Indexed Color mode with its colour table, as in Photoshop
    // (#3802, #4804).
    let palette = match codecs::detect(bytes) {
        Some(Format::Bmp) => codecs::bmp_palette(bytes).ok().flatten().map(|c| (c, None)),
        Some(Format::Png) => codecs::png_palette(bytes).ok().flatten().map(|p| (p.colors, p.transparent)),
        _ => None,
    };
    let mut r = image_to_document(name, &img)?;
    if let Some((colors, transparent)) = palette.filter(|(c, _)| !c.is_empty())
        && r.document.mode == ColorMode::Rgb
    {
        r.document.mode = ColorMode::Indexed;
        r.document.color_table = Some(photocraft_doc::ColorTable { colors, transparent });
    }
    r.document.file_encoding = file_encoding(bytes);
    if codecs::detect(bytes) == Some(Format::Tga) && img.layout() == ChannelLayout::Rgba {
        tga_alpha_to_channel(&mut r.document)?;
    }
    // OpenEXR and Radiance HDR hold linear, scene-referred values (Rec. 709 primaries unless
    // stated otherwise, and an OpenEXR `Y` is linear luminance): tag them linear sRGB or linear
    // gray so they display and convert correctly.
    let d = &mut r.document;
    if d.icc_profile.is_none() && matches!(codecs::detect(bytes), Some(Format::OpenExr | Format::Hdr)) {
        d.icc_profile = match d.mode {
            ColorMode::Rgb => Some(photocraft_cms::Builtin::LinearSrgb.profile().to_bytes()),
            ColorMode::Grayscale => Some(photocraft_cms::builtin::linear_gray().to_bytes()),
            _ => None,
        };
    }
    Ok(r)
}

/// Every frame of a GIF as a single-layer document, handed to `each` in order with its delay in
/// milliseconds (a still GIF has one). One frame at a time becomes a document, so the caller
/// can flatten and drop it. Returns the number of frames; stops at the first error `each` gives.
pub fn import_gif_frames(name: &str, bytes: &[u8], each: &mut dyn FnMut(Document, u32) -> Result<(), IoError>) -> Result<usize, IoError> {
    let frames = codecs::decode_gif_frames(bytes, &codecs::DecodeOptions::default())?;
    let n = frames.len();
    for f in frames {
        each(image_to_document(name, &f.image)?.document, f.delay_ms)?;
    }
    Ok(n)
}

/// The lossy settings `bytes` were encoded with, for File › Save to keep: a JPEG's quality
/// (estimated from its quantisation tables) and a WebP's lossless or lossy encoding with its
/// quality. A lossy WebP whose quality can't be read gets [`DEFAULT_LOSSY_WEBP_QUALITY`].
pub fn file_encoding(bytes: &[u8]) -> photocraft_doc::FileEncoding {
    let mut e = photocraft_doc::FileEncoding::default();
    match codecs::detect(bytes) {
        Some(Format::Jpeg) => e.jpeg_quality = codecs::jpeg_quality(bytes),
        Some(Format::WebP) => match codecs::webp_encoding(bytes) {
            Some(codecs::WebpEncoding::Lossless) => e.webp_lossless = Some(true),
            Some(codecs::WebpEncoding::Lossy { quality }) => {
                e.webp_lossless = Some(false);
                e.webp_quality = Some(quality.unwrap_or(DEFAULT_LOSSY_WEBP_QUALITY));
            }
            None => {}
        },
        _ => {}
    }
    e
}

/// The quality a lossy WebP is saved at when its own can't be read.
pub const DEFAULT_LOSSY_WEBP_QUALITY: u8 = 75;

/// Applies the settings `doc`'s file was encoded with (see [`file_encoding`]) to `opts`: what a
/// save that chose no quality writes, so a JPEG keeps its quality and a WebP stays lossy or
/// lossless.
pub fn apply_file_encoding(doc: &Document, opts: &mut codecs::EncodeOptions) {
    let e = &doc.file_encoding;
    if let Some(q) = e.jpeg_quality {
        opts.jpeg_quality = q.clamp(1, 100);
    }
    if let Some(lossless) = e.webp_lossless {
        opts.webp_lossless = lossless;
    }
    if let Some(q) = e.webp_quality {
        opts.webp_quality = q.min(100);
    }
}

/// Photoshop opens a 32-bit Targa as an opaque Background plus an "Alpha 1" channel holding the
/// fourth byte, even when the header declares no alpha bits, and keeps the stored colours (#2225;
/// measured on Photoshop 27.11). Layer transparency would hide those colours and lose the channel
/// on the next save, where only an alpha channel becomes the Targa's alpha.
fn tga_alpha_to_channel(doc: &mut Document) -> Result<(), IoError> {
    let canvas = doc.bounds();
    let depth = doc.depth;
    let Some(layer) = doc.layers.first_mut() else { return Ok(()) };
    let LayerContent::Raster(s) = &mut layer.content else { return Ok(()) };
    let ch = s.format().channels();
    if !s.format().alpha || ch < 2 {
        return Ok(());
    }
    let mut alpha = Surface::new(PixelFormat::new(ColorMode::Grayscale, depth, false));
    // A band of rows at a time: no full-size float copy of the image.
    let rows = (BAND_BYTES / (canvas.width().max(1) as usize * ch * 4)).max(1) as i32;
    let mut y = canvas.y0;
    while y < canvas.y1 {
        let r = Rect::new(canvas.x0, y, canvas.x1, y.saturating_add(rows).min(canvas.y1));
        let mut px = s.read_region(r);
        let a: Vec<f32> = px.chunks_exact_mut(ch).map(|p| p.last_mut().map_or(1.0, |a| std::mem::replace(a, 1.0))).collect();
        s.write_region(r, &px);
        alpha.write_region(r, &a);
        y = r.y1;
    }
    s.prune();
    alpha.prune();
    layer.name = "Background".into();
    layer.locks.transparency = true;
    layer.locks.position = true;
    doc.channels.push(photocraft_doc::AlphaChannel::new("Alpha 1", alpha));
    Ok(())
}

/// Undoes [`tga_alpha_to_channel`] for a document a Targa just opened as: the "Alpha 1" channel
/// becomes the layer's transparency again. For image sequences, which treat a frame's alpha as
/// transparency, as video footage does (Photoshop's Interpret Footage). Anything else is left
/// alone.
pub fn tga_alpha_channel_to_transparency(doc: &mut Document) {
    let canvas = doc.bounds();
    let ([layer], [channel]) = (&mut doc.layers[..], &doc.channels[..]) else { return };
    let LayerContent::Raster(s) = &mut layer.content else { return };
    let ch = s.format().channels();
    if !s.format().alpha || ch < 2 || channel.spot.is_some() {
        return;
    }
    let rows = (BAND_BYTES / (canvas.width().max(1) as usize * ch * 4)).max(1) as i32;
    let mut y = canvas.y0;
    while y < canvas.y1 {
        let r = Rect::new(canvas.x0, y, canvas.x1, y.saturating_add(rows).min(canvas.y1));
        let mut px = s.read_region(r);
        for (p, a) in px.chunks_exact_mut(ch).zip(channel.surface.read_region(r)) {
            if let Some(last) = p.last_mut() {
                *last = a;
            }
        }
        s.write_region(r, &px);
        y = r.y1;
    }
    s.prune();
    layer.name = "Layer 0".into();
    layer.locks.transparency = false;
    layer.locks.position = false;
    doc.channels.clear();
}

/// Opens one page of a TIFF or BigTIFF file: `None` is the page Photoshop opens (the first
/// full-resolution one), `Some(i)` an index into [`codecs::tiff_info`]'s pages (every IFD and
/// SubIFD). A page with Photoshop layer data opens layered.
pub fn import_tiff_page(name: &str, bytes: &[u8], page: Option<usize>) -> Result<ImportResult, IoError> {
    // The orientation is applied after the layer check: rotating the composite but not the
    // layers would misalign them, so a layered TIFF keeps its stored orientation.
    let keep = codecs::DecodeOptions { keep_orientation: true, ..Default::default() };
    let (img, orientation) = match page {
        None => (codecs::decode_with(bytes, &keep)?, codecs::tiff_orientation(bytes)),
        Some(p) => (codecs::decode_tiff_page(bytes, p, &keep)?, codecs::tiff_page_orientation(bytes, p)),
    };
    if let Some(layers) = img.meta.photoshop_layers.as_deref().filter(|l| !l.is_empty()) {
        let mut r = crate::tiff_layers::import_layered(name, &img, layers)?;
        if orientation != 1 {
            r.warnings.push("the TIFF's orientation tag was ignored so the layers stay aligned with the image".to_string());
        }
        return Ok(r);
    }
    image_to_document(name, &img.oriented(orientation)?)
}

/// A decoded flat image as a single-layer document: a locked "Background" when it is opaque, a
/// normal "Layer 0" when it has transparency.
pub(crate) fn image_to_document(name: &str, img: &Image) -> Result<ImportResult, IoError> {
    // What the decoder noticed (frames or pages left out, data ending early) comes first.
    let mut warnings: Vec<String> = img.warnings.iter().map(ToString::to_string).collect();
    let (mode, target_layout) = match img.layout() {
        ChannelLayout::Gray | ChannelLayout::GrayA => (ColorMode::Grayscale, ChannelLayout::GrayA),
        ChannelLayout::Rgb | ChannelLayout::Rgba => (ColorMode::Rgb, ChannelLayout::Rgba),
        ChannelLayout::Cmyk | ChannelLayout::CmykA => (ColorMode::Cmyk, ChannelLayout::CmykA),
    };
    let (depth, csample) = match img.sample_type() {
        CSample::U8 => (SampleType::U8, CSample::U8),
        CSample::U16 => (SampleType::U16, CSample::U16),
        CSample::F16 => {
            warnings.push("16-bit float samples are stored as 32-bit float".to_string());
            (SampleType::F32, CSample::F32)
        }
        CSample::F32 => (SampleType::F32, CSample::F32),
    };
    let (w, h) = img.dimensions();
    let mut doc = Document::new(name, Size::new(w, h), mode, depth);
    let mut s = image_surface(img, target_layout, csample)?;
    s.prune();
    // As in Photoshop: an opaque image opens as the locked Background layer, one with
    // transparency (an alpha channel, or a tRNS chunk the decoder expands to one) as a normal
    // layer named "Layer 0", since a Background can't hold transparency.
    let layer = if img.layout().has_alpha() {
        Layer::new("Layer 0", LayerContent::Raster(s))
    } else {
        let mut bg = Layer::new("Background", LayerContent::Raster(s));
        bg.locks.transparency = true;
        bg.locks.position = true;
        bg
    };
    doc.layers.push(layer);
    doc.icc_profile = img.icc.clone().map(Arc::new);
    doc.metadata.exif = img.meta.exif.clone().map(Arc::new);
    doc.metadata.xmp = img.meta.xmp.clone();
    doc.metadata.text = img.meta.text.clone();
    if let Some((x, y)) = img.meta.dpi {
        doc.resolution_dpi = x;
        warnings.extend(crate::unequal_resolution_warning(f64::from(x), f64::from(y)));
    }
    Ok(ImportResult { document: doc, warnings, source_read_only: false, preview_only: false })
}

/// The pixels of a decoded image as a surface of `fmt`, converting layout and sample type a
/// band of rows at a time so no second full-size copy of the image is made.
pub(crate) fn image_surface(img: &Image, target_layout: ChannelLayout, csample: CSample) -> Result<Surface, IoError> {
    let mode = match target_layout {
        ChannelLayout::Gray | ChannelLayout::GrayA => ColorMode::Grayscale,
        ChannelLayout::Cmyk | ChannelLayout::CmykA => ColorMode::Cmyk,
        _ => ColorMode::Rgb,
    };
    let depth = match csample {
        CSample::U8 => SampleType::U8,
        CSample::U16 => SampleType::U16,
        _ => SampleType::F32,
    };
    let fmt = PixelFormat::new(mode, depth, true);
    let (w, h) = img.dimensions();
    let mut s = Surface::new(fmt);
    if img.layout() == target_layout && img.sample_type() == csample {
        s.write_interleaved(Rect::new(0, 0, w as i32, h as i32), img.data());
    } else {
        let row = img.data().len() / (h.max(1) as usize);
        let band = (BAND_BYTES / row.max(1)).max(1);
        for (i, rows) in img.data().chunks(row.max(1) * band).enumerate() {
            let n = (rows.len() / row.max(1)) as u32;
            let part = Image::from_raw(w, n, img.layout(), img.sample_type(), rows.to_vec())?.convert(target_layout, csample);
            let y0 = (i * band) as i32;
            s.write_interleaved(Rect::new(0, y0, w as i32, y0 + n as i32), part.data());
        }
    }
    Ok(s)
}

/// `Some(surface)` when the document is exactly one visible, unmasked,
/// normal, fully opaque raster layer: its pixels can be written natively
/// (keeping CMYK / depth exactly) instead of going through the compositor.
pub(crate) fn single_layer(doc: &Document) -> Option<&Surface> {
    let [l] = &doc.layers[..] else { return None };
    let ok = l.visible
        && l.opacity >= 1.0
        && l.fill_opacity >= 1.0
        && l.mask.is_none()
        && l.vector_mask.as_ref().is_none_or(|mask| !mask.enabled)
        && l.effects.items.is_empty()
        && l.effects.psd_raw.is_none()
        && matches!(l.blend, BlendMode::Normal | BlendMode::PassThrough)
        && photocraft_compose::channel_weights(l, doc.mode).is_none()
        && !photocraft_compose::blend_if_active(l, doc.mode);
    match (&l.content, ok) {
        (LayerContent::Raster(s), true) if s.format() == doc.pixel_format() => Some(s),
        _ => None,
    }
}

pub(crate) fn layout_for(mode: ColorMode, alpha: bool) -> ChannelLayout {
    match (mode, alpha) {
        (ColorMode::Grayscale, false) => ChannelLayout::Gray,
        (ColorMode::Grayscale, true) => ChannelLayout::GrayA,
        (ColorMode::Cmyk, false) => ChannelLayout::Cmyk,
        (ColorMode::Cmyk, true) => ChannelLayout::CmykA,
        (_, false) => ChannelLayout::Rgb,
        (_, true) => ChannelLayout::Rgba,
    }
}

pub(crate) fn csample(s: SampleType) -> CSample {
    match s {
        SampleType::U8 => CSample::U8,
        SampleType::U16 => CSample::U16,
        SampleType::F32 => CSample::F32,
    }
}

/// Renders the document to a flat codec image (native pixels when possible).
pub fn document_to_image(doc: &Document, warnings: &mut Vec<String>) -> Result<Image, IoError> {
    document_to_image_for(doc, warnings, true)
}

fn document_to_image_for(doc: &Document, warnings: &mut Vec<String>, cmyk_target: bool) -> Result<Image, IoError> {
    let (w, h) = (doc.size.width, doc.size.height);
    let canvas = doc.bounds();
    let fmt = doc.pixel_format();
    let n = (w as usize) * (h as usize);
    // Lab has no flat-format layout here: Lab documents always go through the composite, which
    // flattens nothing when there is a single plain layer.
    let single = single_layer(doc);
    let native = single.filter(|_| fmt.mode != ColorMode::Lab);
    let mut icc = doc.icc_profile.as_ref().map(|i| i.to_vec());
    let img = if let Some(s) = native {
        // Native path: keep model and depth. The surface's encoded samples are the codec's raw
        // native-endian samples, copied a band of rows at a time (alpha dropped when opaque).
        let ch = fmt.channels();
        let opaque = opaque_surface(s, canvas);
        let layout = layout_for(fmt.mode, !opaque);
        let bps = fmt.sample.bytes();
        let keep = if opaque { (ch - 1) * bps } else { ch * bps };
        let mut data = try_buffer(n, keep)?;
        let rows = (BAND_BYTES / (w as usize * ch * bps).max(1)).max(1) as i32;
        let mut y = 0;
        while y < h as i32 {
            let y1 = y.saturating_add(rows).min(h as i32);
            let band = s.to_interleaved(Rect::new(0, y, w as i32, y1));
            if opaque {
                for px in band.chunks_exact(ch * bps) {
                    data.extend_from_slice(&px[..keep]);
                }
            } else {
                data.extend_from_slice(&band);
            }
            y = y1;
        }
        Image::from_raw(w, h, layout, csample(fmt.sample), data)?
    } else {
        if single.is_none() {
            let count = doc.layer_count();
            warnings.push(format!("{count} layer(s) flattened; layers, masks and blend modes are not kept"));
        }
        let cmyk = fmt.mode == ColorMode::Cmyk && cmyk_target;
        // Only a CMYK target needs the document's profile; RGB-only targets keep the compositor's
        // own handling (built-in coated CMYK when the embedded profile is unusable).
        let space = if cmyk {
            use photocraft_cms::{Builtin, ColorSpace, Intent, Profile, Transform};
            let profile = match doc.icc_profile.as_ref() {
                Some(bytes) => Profile::parse(bytes).map_err(|e| IoError::Unsupported(format!("invalid CMYK composite profile: {e}")))?,
                None => Builtin::CoatedCmyk.profile().clone(),
            };
            if profile.color_space != ColorSpace::Cmyk {
                return Err(IoError::Unsupported("CMYK composite requires a CMYK ICC profile".into()));
            }
            let srgb = Builtin::Srgb.profile();
            Transform::new(&profile, srgb, Intent::RelativeColorimetric, true).map_err(|e| IoError::Unsupported(e.to_string()))?;
            Transform::new(srgb, &profile, Intent::RelativeColorimetric, true).map_err(|e| IoError::Unsupported(e.to_string()))?;
            if icc.is_none() {
                icc = Some(profile.to_bytes().to_vec());
            }
            photocraft_compose::cmyk_space(doc)
        } else {
            None
        };
        let gray = fmt.mode == ColorMode::Grayscale;
        if (fmt.mode == ColorMode::Cmyk && !cmyk) || fmt.mode == ColorMode::Lab {
            // RGB-only targets use the sRGB composite directly, without reseparating CMYK.
            warnings.push(format!("{:?} composite written as sRGB RGB (colour-managed conversion)", fmt.mode));
            icc = Some(photocraft_cms::Builtin::Srgb.profile().to_bytes().to_vec());
        }
        let cs = csample(fmt.sample);
        let colors = if cmyk {
            4
        } else if gray {
            1
        } else {
            3
        };
        // Rendered and quantised in bands (no full-size float composite), with alpha; the alpha
        // is dropped afterwards, in place, when every pixel turned out opaque.
        let mut data = try_buffer(n, (colors + 1) * cs.bytes())?;
        let mut opaque = true;
        let _ = photocraft_compose::render_bands(doc, canvas, 0, |band| -> Result<(), ()> {
            opaque &= band.px.iter().all(|p| p[3] >= 1.0);
            let parts = crate::pixels::par_map(crate::pixels::bands(band.px.len()), |range| {
                photocraft_color::convert::with_cmyk_space(space.as_ref(), || {
                    let mut out = Vec::with_capacity(range.len() * (colors + 1) * cs.bytes());
                    let mut put = |v: f32| {
                        let v = if v.is_nan() { 0.0 } else { v };
                        match cs {
                            CSample::U8 => out.push((v.clamp(0.0, 1.0) * 255.0).round() as u8),
                            CSample::U16 => out.extend_from_slice(&((v.clamp(0.0, 1.0) * 65535.0).round() as u16).to_ne_bytes()),
                            CSample::F16 | CSample::F32 => out.extend_from_slice(&v.to_ne_bytes()),
                        }
                    };
                    let mut converted = [0.0; 5];
                    for p in &band.px[range] {
                        if cmyk {
                            let channels = photocraft_raster::from_rgba_into(&fmt, *p, &mut converted);
                            for value in converted.iter().take(channels) {
                                put(*value);
                            }
                            continue;
                        } else if gray {
                            put(photocraft_color::convert::rgb_to_gray([p[0], p[1], p[2]]));
                        } else {
                            put(p[0]);
                            put(p[1]);
                            put(p[2]);
                        }
                        put(p[3]);
                    }
                    out
                })
            });
            for part in parts {
                data.extend_from_slice(&part);
            }
            Ok(())
        });
        if opaque {
            drop_alpha(&mut data, colors * cs.bytes(), cs.bytes());
        }
        let layout = layout_for(
            if cmyk {
                ColorMode::Cmyk
            } else if gray {
                ColorMode::Grayscale
            } else {
                ColorMode::Rgb
            },
            !opaque,
        );
        Image::from_raw(w, h, layout, cs, data)?
    };
    let meta = codecs::Metadata {
        exif: doc.metadata.exif.as_ref().map(|e| e.to_vec()),
        xmp: doc.metadata.xmp.clone(),
        text: doc.metadata.text.clone(),
        dpi: Some((doc.resolution_dpi, doc.resolution_dpi)),
        ..Default::default()
    };
    Ok(img.with_icc(icc).with_meta(meta))
}

/// An empty buffer with room for `pixels × bytes_per_pixel` bytes, or an error (not an abort)
/// when that much memory can't be had.
pub(crate) fn try_buffer(pixels: usize, bytes_per_pixel: usize) -> Result<Vec<u8>, IoError> {
    let len = pixels.checked_mul(bytes_per_pixel).ok_or_else(|| IoError::Unsupported("image too large".into()))?;
    let mut v = Vec::new();
    v.try_reserve_exact(len).map_err(|_| IoError::Unsupported(format!("not enough memory for a {} MB image", len >> 20)))?;
    Ok(v)
}

/// Remove the trailing `alpha` bytes of every `color + alpha`-byte pixel, in place.
fn drop_alpha(data: &mut Vec<u8>, color: usize, alpha: usize) {
    let stride = color + alpha;
    let n = data.len() / stride;
    for i in 0..n {
        data.copy_within(i * stride..i * stride + color, i * color);
    }
    data.truncate(n * color);
    data.shrink_to_fit();
}

/// Whether every pixel of `s` over `r` is fully opaque (read from the tiles, no copy).
fn opaque_surface(s: &Surface, r: Rect) -> bool {
    let fmt = s.format();
    let (ch, bps) = (fmt.channels(), fmt.sample.bytes());
    let a = ch - 1;
    let alpha_ok = |px: &[u8]| photocraft_color::read_sample(px, fmt.sample, a) >= 1.0;
    let mut default = vec![0u8; ch * bps];
    photocraft_raster::encode_pixel(&fmt, &s.default_pixel(), &mut default);
    r.tiles().all(|tc| {
        let tr = tc.rect().intersect(&r);
        match s.tile(tc) {
            None => alpha_ok(&default),
            Some(t) => (tr.y0..tr.y1).all(|y| {
                let row = ((y - tc.rect().y0) as usize * TILE_SIZE as usize + (tr.x0 - tc.rect().x0) as usize) * ch * bps;
                t.bytes()[row..row + tr.width() as usize * ch * bps].chunks_exact(ch * bps).all(alpha_ok)
            }),
        }
    })
}

/// The part of a flat export that rendering the composite takes, the encoder taking the rest.
/// Measured on a 24 MP RGB/8 document: ~1 s to composite, then TIFF ~2.3 s, JPEG ~0.9 s,
/// PNG ~0.4 s, WebP ~0.25 s to encode.
fn composite_share(format: Format) -> f32 {
    match format {
        Format::Tiff => 0.3,
        Format::Jpeg => 0.55,
        _ => 0.7,
    }
}

/// Flattens and encodes as `format`.
pub fn export_flat(doc: &Document, format: Format, opts: &ExportOptions) -> Result<ExportResult, IoError> {
    export_flat_with(doc, format, opts, &photocraft_raster::Interrupt::NONE)
}

/// [`export_flat`] reporting a coarse step through `ctl` once the composite is rendered; the
/// encoder does the rest (#3389).
pub(crate) fn export_flat_with(doc: &Document, format: Format, opts: &ExportOptions, ctl: &photocraft_raster::Interrupt) -> Result<ExportResult, IoError> {
    use photocraft_cms::{Builtin, Intent};
    if let Some(r) = export_mode_specific(doc, format, opts)? {
        return Ok(r);
    }
    let mut warnings = Vec::new();
    let cmyk_target = format.caps().layouts.iter().any(|layout| layout.is_cmyk());
    let mut img = document_to_image_for(doc, &mut warnings, cmyk_target)?;
    ctl.check().map_err(|_| IoError::Cancelled)?;
    ctl.progress(composite_share(format));
    if opts.xmp == XmpEmbed::None {
        // Export As's Metadata: None: the packet lists the text of every type layer and one id
        // per placed document (#647).
        img.meta.xmp = None;
        // Free-form descriptions can contain the same sensitive text as an XMP packet.
        img.meta.text.clear();
    }
    if img.layout().has_alpha() && !format.caps().alpha {
        // Flattened over white, as saving a transparent document without transparency does.
        img = matte_over_white(&img)?;
        warnings.push(format!("transparency composited over white for {format:?}"));
    }
    if img.layout().is_cmyk() && !format.caps().layouts.iter().any(|l| l.is_cmyk()) {
        img = cmyk_image_to_srgb(&img)?;
        warnings.push(format!("CMYK converted to sRGB for {format:?} through the document's colour profile"));
    }
    if matches!(format, Format::OpenExr | Format::Hdr) {
        // OpenEXR and Radiance HDR store linear light (read back as linear sRGB or linear gray,
        // see [`import_flat`]).
        if let Some(linear) = convert_rgb(&img, Builtin::LinearSrgb.profile(), Intent::RelativeColorimetric, false, CSample::F32)? {
            img = linear;
        } else if let Some(linear) = convert_gray(&img, photocraft_cms::builtin::linear_gray(), CSample::F32)? {
            img = linear;
        }
    } else if !format.caps().icc {
        // Untagged files read back as sRGB/sGray: convert to it (as Quick Export does) rather than
        // write values that only mean something under the dropped profile.
        if let Some(srgb) = convert_rgb(&img, Builtin::Srgb.profile(), Intent::Perceptual, true, img.sample_type())? {
            img = srgb;
            warnings.push(format!("colours converted to sRGB; {format:?} can't embed the document's colour profile"));
        } else if let Some(gray) = convert_gray(&img, Builtin::SGray.profile(), img.sample_type())? {
            img = gray;
            warnings.push(format!("colours converted to sGray; {format:?} can't embed the document's colour profile"));
        }
    }
    if format == Format::Tga {
        img = tga_image(doc, img, opts.tga_bits, &mut warnings)?;
    }
    encode_image(&img, format, opts, warnings)
}

/// What Targa Options offers first, as Photoshop does: 32 bits per pixel when the document has an
/// alpha channel (spot channels don't count), else 24.
pub fn tga_default_bits(doc: &Document) -> TgaBits {
    if doc.channels.iter().any(|c| c.spot.is_none()) { TgaBits::Bits32 } else { TgaBits::Bits24 }
}

/// The RGB image a Targa of `bits` per pixel holds (`None`: 32 with an alpha channel or
/// transparency, else 24). 24 bits are the colours alone, composited over white where the layers
/// are transparent. 32 add an alpha: as in Photoshop, the document's alpha channel when it has
/// exactly one (spot channels don't count); with none, or several (Photoshop then writes none of
/// them), the layers' transparency, fully opaque without any. Colours are written as they are,
/// never premultiplied or cleared under a zero alpha: Targa alphas often hold data such as a
/// game texture's specular mask. Grayscale Targas have no alpha channel in Photoshop and are left
/// as they are.
fn tga_image(doc: &Document, img: Image, bits: Option<TgaBits>, warnings: &mut Vec<String>) -> Result<Image, IoError> {
    if !img.layout().is_rgb() {
        if let Some(bits) = bits {
            warnings.push(format!("{}-bit Targa applies to RGB images; the grayscale image is written as it is", bits.bits()));
        }
        return Ok(img);
    }
    let alphas: Vec<&AlphaChannel> = doc.channels.iter().filter(|c| c.spot.is_none()).collect();
    let bits = bits.unwrap_or(if !alphas.is_empty() || img.layout().has_alpha() { TgaBits::Bits32 } else { TgaBits::Bits24 });
    match (bits, alphas.as_slice()) {
        (TgaBits::Bits24, alphas) => {
            if !alphas.is_empty() {
                let names: Vec<String> = alphas.iter().map(|c| format!("\"{}\"", c.name)).collect();
                warnings.push(format!("a 24-bit Targa has no alpha; not saved: {}", names.join(", ")));
            }
            if !img.layout().has_alpha() {
                return Ok(img);
            }
            warnings.push("transparency composited over white; a 24-bit Targa has no alpha".into());
            matte_over_white(&img)
        }
        (TgaBits::Bits32, [channel]) => tga_with_alpha_channel(doc, img, channel, warnings),
        (TgaBits::Bits32, alphas) => {
            if !alphas.is_empty() {
                warnings.push("Targa holds one alpha channel; with several, none was written (as in Photoshop)".into());
            }
            if img.layout().has_alpha() {
                return Ok(img);
            }
            map_bands(&img, ChannelLayout::Rgba, img.sample_type(), |vals| vals.as_chunks::<3>().0.iter().flat_map(|&[r, g, b]| [r, g, b, 1.0]).collect())
        }
    }
}

/// A 32-bit Targa's image: the colours (composited over white where the layers are transparent)
/// and `channel`'s values as the alpha.
fn tga_with_alpha_channel(doc: &Document, img: Image, channel: &AlphaChannel, warnings: &mut Vec<String>) -> Result<Image, IoError> {
    let img = if img.layout().has_alpha() {
        warnings.push("transparency composited over white; the alpha channel is the Targa's alpha".into());
        matte_over_white(&img)?
    } else {
        img
    };
    let (w, h) = img.dimensions();
    let canvas = doc.bounds();
    if canvas.width() != w || canvas.height() != h {
        return Err(IoError::Unsupported("the composite and the alpha channel differ in size".into()));
    }
    let sample = img.sample_type();
    let row = img.data().len() / (h.max(1) as usize);
    let band = (BAND_BYTES / row.max(1)).max(1);
    let stride = channel.surface.channels().max(1);
    let mut data = try_buffer(img.pixel_count(), ChannelLayout::Rgba.channels() * sample.bytes())?;
    let mut y0 = canvas.y0;
    for rows in img.data().chunks(row.max(1) * band) {
        let n = (rows.len() / row.max(1)) as u32;
        let y1 = y0.saturating_add(i32::try_from(n).unwrap_or(i32::MAX));
        let alpha = channel.surface.read_region(Rect::new(canvas.x0, y0, canvas.x1, y1));
        let rgb = Image::from_raw(w, n, img.layout(), sample, rows.to_vec())?.to_normalized();
        let mut vals = Vec::with_capacity(rgb.len() / 3 * 4);
        for (px, a) in rgb.as_chunks::<3>().0.iter().zip(alpha.chunks_exact(stride)) {
            vals.extend_from_slice(px);
            vals.push(a.first().copied().unwrap_or(1.0));
        }
        data.extend_from_slice(Image::from_normalized(w, n, ChannelLayout::Rgba, sample, &vals)?.data());
        y0 = y1;
    }
    Ok(Image::from_raw(w, h, ChannelLayout::Rgba, sample, data)?.with_icc(img.icc.clone()).with_meta(img.meta.clone()))
}

/// Encodes a flat codec image as `format`: the fidelity warnings, then the codec. The end of
/// every flat export, and of a layered TIFF (whose composite is a flat image to other readers),
/// so export-wide policies on the image's metadata apply to both.
pub(crate) fn encode_image(img: &Image, format: Format, opts: &ExportOptions, mut warnings: Vec<String>) -> Result<ExportResult, IoError> {
    for w in codecs::fidelity_warnings_with(img, format, &opts.encode) {
        if w.is_fatal() {
            return Err(IoError::Unsupported(w.to_string()));
        }
        warnings.push(w.to_string());
    }
    let bytes = codecs::encode(img, format, &opts.encode)?;
    Ok(ExportResult { bytes, warnings })
}

/// `img` with every band of rows mapped by `f` (normalized samples in, normalized `layout`
/// samples out) and stored as `sample`, so no full-size float copy is made. Profile and metadata
/// are kept.
fn map_bands(img: &Image, layout: ChannelLayout, sample: CSample, f: impl Fn(Vec<f32>) -> Vec<f32>) -> Result<Image, IoError> {
    let (w, h) = img.dimensions();
    let row = img.data().len() / (h.max(1) as usize);
    let band = row.max(1) * (BAND_BYTES / row.max(1)).max(1);
    let mut data = try_buffer(img.pixel_count(), layout.channels() * sample.bytes())?;
    for rows in img.data().chunks(band) {
        let n = (rows.len() / row.max(1)) as u32;
        let vals = f(Image::from_raw(w, n, img.layout(), img.sample_type(), rows.to_vec())?.to_normalized());
        data.extend_from_slice(Image::from_normalized(w, n, layout, sample, &vals)?.data());
    }
    Ok(Image::from_raw(w, h, layout, sample, data)?.with_icc(img.icc.clone()).with_meta(img.meta.clone()))
}

/// Straight-alpha pixels composited over white (no ink for CMYK), without the alpha channel.
fn matte_over_white(img: &Image) -> Result<Image, IoError> {
    let layout = img.layout();
    let white = if layout.is_cmyk() { 0.0 } else { 1.0 };
    map_bands(img, layout.without_alpha(), img.sample_type(), |vals| {
        let mut out = Vec::with_capacity(vals.len() / layout.channels() * layout.color_channels());
        for px in vals.chunks_exact(layout.channels()) {
            if let Some((&a, color)) = px.split_last() {
                let a = a.clamp(0.0, 1.0);
                out.extend(color.iter().map(|&c| c * a + white * (1.0 - a)));
            }
        }
        out
    })
}

/// RGB pixels converted from their profile (sRGB when untagged) to `dst`, unclamped, untagged and
/// stored as `sample`. `None` when the image isn't RGB or already holds `dst`'s colours.
fn convert_rgb(img: &Image, dst: &photocraft_cms::Profile, intent: photocraft_cms::Intent, bpc: bool, sample: CSample) -> Result<Option<Image>, IoError> {
    use photocraft_cms::{Builtin, ColorSpace, Profile};
    if !img.layout().is_rgb() {
        return Ok(None);
    }
    let src =
        img.icc.as_ref().and_then(|b| Profile::parse(b).ok()).filter(|p| p.color_space == ColorSpace::Rgb).unwrap_or_else(|| Builtin::Srgb.profile().clone());
    convert_from(img, &src, dst, intent, bpc, sample)
}

/// Gray pixels converted from their profile (sGray when untagged) to `dst`, unclamped, untagged
/// and stored as `sample`. `None` when the image isn't gray or already holds `dst`'s colours.
fn convert_gray(img: &Image, dst: &photocraft_cms::Profile, sample: CSample) -> Result<Option<Image>, IoError> {
    use photocraft_cms::{Builtin, ColorSpace, Intent, Profile};
    if !matches!(img.layout(), ChannelLayout::Gray | ChannelLayout::GrayA) {
        return Ok(None);
    }
    let src =
        img.icc.as_ref().and_then(|b| Profile::parse(b).ok()).filter(|p| p.color_space == ColorSpace::Gray).unwrap_or_else(|| Builtin::SGray.profile().clone());
    convert_from(img, &src, dst, Intent::RelativeColorimetric, false, sample)
}

/// `img`'s pixels converted from `src` to `dst` (see [`convert_rgb`]).
fn convert_from(
    img: &Image,
    src: &photocraft_cms::Profile,
    dst: &photocraft_cms::Profile,
    intent: photocraft_cms::Intent,
    bpc: bool,
    sample: CSample,
) -> Result<Option<Image>, IoError> {
    if src.same_colors(dst) {
        return Ok(None);
    }
    let t = photocraft_cms::Transform::new(src, dst, intent, bpc).map_err(|e| IoError::Unsupported(e.to_string()))?;
    let stride = img.layout().channels();
    let out = map_bands(img, img.layout(), sample, |mut vals| {
        t.apply(&mut vals, stride);
        vals
    })?;
    Ok(Some(out.with_icc(None)))
}

/// Colour-managed CMYK → sRGB for formats that cannot store CMYK (the document's embedded
/// CMYK profile when it parses, else the built-in coated CMYK; relative colorimetric + BPC).
fn cmyk_image_to_srgb(img: &Image) -> Result<Image, IoError> {
    use photocraft_cms::{Builtin, ColorSpace, Intent, Profile, Transform};
    let src = img
        .icc
        .as_ref()
        .and_then(|b| Profile::parse(b).ok())
        .filter(|p| p.color_space == ColorSpace::Cmyk)
        .unwrap_or_else(|| Builtin::CoatedCmyk.profile().clone());
    let dst = Builtin::Srgb.profile();
    let t = Transform::new(&src, dst, Intent::RelativeColorimetric, true).map_err(|e| IoError::Unsupported(e.to_string()))?;
    let alpha = img.layout().has_alpha();
    let (ss, ds) = (if alpha { 5 } else { 4 }, if alpha { 4 } else { 3 });
    let (w, h) = img.dimensions();
    let layout = if alpha { ChannelLayout::Rgba } else { ChannelLayout::Rgb };
    let sample = match img.sample_type() {
        CSample::F16 => CSample::F32,
        s => s,
    };
    // Row bands through the transform: three full-size f32 buffers (normalized input, the
    // transform's working set, sRGB output) become band-height slices — ~1.4 GB of allocation
    // churn at 36 MP before this. `convert_f32` walks plain strided slices, so bands are free.
    let row = w as usize;
    let src_bpp = img.layout().channels() * img.sample_type().bytes();
    let h_rows = h as usize;
    // `max(1)` keeps a zero-width or zero-height image from dividing by zero or inverting the
    // clamp range; such an image simply has no bands.
    let band_rows = (BAND_BYTES / (row * ss * 4).max(1)).clamp(1, h_rows.max(1));
    let mut out = Vec::with_capacity(img.pixel_count() * ds);
    let mut vals: Vec<f32> = Vec::with_capacity(band_rows * row * ss);
    let mut dst_band: Vec<f32> = Vec::with_capacity(band_rows * row * ds);
    // Whole bands of rows only: a trailing partial row (h not a multiple of band_rows, or a
    // 1-row image smaller than the band budget) is folded into the last band.
    let mut idx = 0usize;
    while idx < h_rows {
        let n = band_rows.min(h_rows - idx);
        let bytes = |r: usize| r.checked_mul(row).and_then(|v| v.checked_mul(src_bpp));
        let rows = bytes(idx)
            .zip(bytes(idx + n))
            .and_then(|(a, b)| img.data().get(a..b))
            .ok_or_else(|| IoError::Unsupported("CMYK image data is shorter than its dimensions".into()))?;
        idx += n;
        vals.clear();
        vals.extend(Image::from_raw(w, n as u32, img.layout(), img.sample_type(), rows.to_vec())?.to_normalized());
        dst_band.clear();
        dst_band.resize(n * row * ds, 0.0);
        t.convert_f32(&vals, ss, &mut dst_band, ds, true);
        out.extend_from_slice(&dst_band);
    }
    Ok(Image::from_normalized(w, h, layout, sample, &out)?.with_icc(Some(dst.to_bytes().to_vec())).with_meta(img.meta.clone()))
}

/// Indexed Color → PNG-8 with its colour table; Duotone → the inks rendered as RGB.
fn export_mode_specific(doc: &Document, format: Format, opts: &ExportOptions) -> Result<Option<ExportResult>, IoError> {
    match doc.mode {
        ColorMode::Indexed if format == Format::Png => {
            let Some(table) = doc.color_table.as_ref().filter(|t| !t.colors.is_empty() && t.colors.len() <= 256) else { return Ok(None) };
            let mut idx = try_buffer(doc.size.area() as usize, 1)?;
            let _ = photocraft_compose::render_bands(doc, doc.bounds(), 0, |band| -> Result<(), ()> {
                idx.extend(band.px.iter().map(|p| match table.transparent {
                    Some(t) if p[3] < 0.5 => t,
                    _ => table.nearest([p[0], p[1], p[2]]) as u8,
                }));
                Ok(())
            });
            // Same rule as the ordinary PNG encoder: the resolution goes into `pHYs` with the metadata.
            let dpi = opts.encode.embed_metadata.then_some((doc.resolution_dpi, doc.resolution_dpi));
            let bytes = codecs::encode_png_indexed(doc.size.width, doc.size.height, &idx, &table.colors, table.transparent, dpi)?;
            let mut warnings = vec![format!("written as an 8-bit palette PNG ({} colours)", table.colors.len())];
            if opts.encode.embed_metadata && opts.xmp == XmpEmbed::All && !doc.metadata.text.is_empty() {
                warnings.push("text metadata is not supported by the palette PNG exporter; it will be dropped".into());
            }
            Ok(Some(ExportResult { bytes, warnings }))
        }
        ColorMode::Duotone => {
            let Some(d) = doc.duotone.as_ref() else { return Ok(None) };
            let mut shown = doc.clone();
            shown.layers.push(photocraft_doc::Layer::new("Duotone", photocraft_doc::LayerContent::Adjustment(d.display_adjustment())));
            shown.mode = ColorMode::Rgb;
            shown.icc_profile = None;
            shown.duotone = None;
            let mut r = export_flat(&shown, format, opts)?;
            r.warnings.retain(|w| !w.contains("flattened"));
            r.warnings.push(format!("Duotone ({} inks) written as RGB", d.inks.len()));
            Ok(Some(r))
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod cmyk_band_tests {
    use super::*;

    #[test]
    fn empty_cmyk_images_convert_without_panicking() {
        // Zero width divided the band size by zero; zero height inverted the clamp range.
        for (w, h) in [(0, 0), (0, 7), (7, 0)] {
            let img = Image::from_raw(w, h, ChannelLayout::Cmyk, CSample::U8, Vec::new()).unwrap();
            if let Ok(out) = cmyk_image_to_srgb(&img) {
                assert_eq!(out.dimensions(), (w, h));
            }
        }
    }

    #[test]
    fn banded_conversion_covers_every_row() {
        // One colour everywhere converts to one colour everywhere, first row to last.
        let (w, h) = (5u32, 9u32);
        let img = Image::from_raw(w, h, ChannelLayout::CmykA, CSample::U8, [40, 90, 10, 20, 255].repeat((w * h) as usize)).unwrap();
        let out = cmyk_image_to_srgb(&img).unwrap();
        assert_eq!(out.dimensions(), (w, h));
        assert_eq!(out.data().len(), (w * h * 4) as usize);
        let first = out.data().get(..4).unwrap().to_vec();
        assert!(out.data().chunks(4).all(|p| p == first.as_slice()), "{:?}", out.data());
        assert_eq!(first[3], 255);
    }
}
