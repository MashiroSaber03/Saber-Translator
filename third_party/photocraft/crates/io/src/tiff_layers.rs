//! Layered TIFF: Photoshop layer data carried in the TIFF tags 37724 (`ImageSourceData`) and
//! 34377 (image resources), see [`photocraft_psd::tiff`].
//!
//! Import rebuilds a PSD model from the TIFF's pixels plus those two tags and hands it to the
//! PSD importer, so everything the PSD path understands (groups, masks, effects, type, shapes,
//! smart objects, adjustment and fill layers, alpha channels, paths, guides) opens from a TIFF
//! as well. Export runs the PSD exporter and stores its layer section and resources in the
//! tags, in the byte order the TIFF encoder writes, next to the flattened composite the TIFF
//! itself carries.

use std::sync::Arc;

use photocraft_codecs::{self as codecs, ChannelLayout, Format, Image, SampleType as CSample};
use photocraft_color::ColorMode;
use photocraft_doc::Document;
use photocraft_psd::resources::ids;
use photocraft_psd::tiff::{ByteOrder, ImageSourceData, resources_from_bytes, resources_to_bytes};
use photocraft_psd::{ColorMode as PsdMode, Compression, Header, ImageData, ImageResource, ResolutionInfo};

use crate::flat::{csample, encode_image, image_to_document, layout_for, single_layer, try_buffer};
use crate::psd_export::PsdExportOptions;
use crate::psd_import::psd_to_document;
use crate::{ExportOptions, ExportResult, ImportResult, IoError};

/// `true` when saving `doc` as TIFF writes layer data: anything beyond a single Background
/// layer (one plain, transparency-locked raster layer), which a flat TIFF already represents
/// exactly. A lone unlocked layer keeps its layer data, as in Photoshop, so it reopens as that
/// layer rather than as a Background.
pub fn would_write_layers(doc: &Document) -> bool {
    !(single_layer(doc).is_some() && doc.layers.first().is_some_and(|l| l.locks.transparency))
}

/// Documents a layered TIFF can hold: the TIFF composite must be in the document's model so
/// that the layer channels inside the tag match it (Lab TIFFs are not decoded or encoded by
/// `photocraft-codecs`), and there must be layers to keep (a Multichannel document has none:
/// its channels are the image).
fn layered_mode(doc: &Document) -> bool {
    matches!(doc.pixel_format().mode, ColorMode::Grayscale | ColorMode::Rgb | ColorMode::Cmyk) && doc.mode != ColorMode::Multichannel && !doc.layers.is_empty()
}

/// Replaces (or inserts) resource `id`; removes it when `data` is `None`.
fn set_resource(resources: &mut Vec<ImageResource>, id: u16, data: Option<Vec<u8>>) {
    resources.retain(|r| r.id != id);
    if let Some(d) = data {
        resources.push(ImageResource::new(id, d));
    }
}

/// Opens a TIFF whose `img` carries Photoshop layer data in `layers` (its 37724 tag).
pub(crate) fn import_layered(name: &str, img: &Image, layers: &[u8]) -> Result<ImportResult, IoError> {
    let mut warnings = Vec::new();
    let (isd, w) = match ImageSourceData::from_bytes(layers) {
        Ok(v) => v,
        Err(e) => {
            let mut r = image_to_document(name, img)?;
            r.warnings.push(format!("the PSD layer data could not be read ({e}); opened flattened"));
            return Ok(r);
        }
    };
    warnings.extend(w.into_iter().map(|w| format!("layer data: {w}")));
    let (mut resources, w) = match img.meta.photoshop_resources.as_deref().filter(|r| !r.is_empty()).map(resources_from_bytes) {
        Some(Ok(v)) => v,
        Some(Err(e)) => {
            warnings.push(format!("the PSD image resources could not be read: {e}"));
            (Vec::new(), Vec::new())
        }
        None => (Vec::new(), Vec::new()),
    };
    warnings.extend(w.into_iter().map(|w| format!("image resources: {w}")));
    // The profile, resolution, XMP and EXIF are the TIFF's own tags; they win over any copy
    // in the resources.
    set_resource(&mut resources, ids::ICC_PROFILE, img.icc.clone());
    if let Some((x, y)) = img.meta.dpi.filter(|d| d.0 > 0.0) {
        // Both axes, so the PSD import notes a vertical resolution the document can't keep.
        let ri = ResolutionInfo { v_res_fixed: ResolutionInfo::from_dpi(f64::from(y)).v_res_fixed, ..ResolutionInfo::from_dpi(f64::from(x)) };
        set_resource(&mut resources, ids::RESOLUTION_INFO, Some(ri.to_bytes()));
    }
    if let Some(xmp) = &img.meta.xmp {
        set_resource(&mut resources, ids::XMP, Some(xmp.as_bytes().to_vec()));
    }
    if let Some(exif) = &img.meta.exif {
        set_resource(&mut resources, ids::EXIF, Some(exif.clone()));
    }

    // The TIFF's pixels become the PSD's merged image: planar, big-endian, CMYK inverted.
    let img = if img.sample_type() == CSample::F16 {
        warnings.push("16-bit float samples are stored as 32-bit float".to_string());
        std::borrow::Cow::Owned(img.convert(img.layout(), CSample::F32))
    } else {
        std::borrow::Cow::Borrowed(img)
    };
    let (mode, color_channels) = match img.layout() {
        ChannelLayout::Gray | ChannelLayout::GrayA => (PsdMode::Grayscale, 1),
        ChannelLayout::Rgb | ChannelLayout::Rgba => (PsdMode::Rgb, 3),
        ChannelLayout::Cmyk | ChannelLayout::CmykA => (PsdMode::Cmyk, 4),
    };
    let depth = match img.sample_type() {
        CSample::U8 => 8,
        CSample::U16 => 16,
        CSample::F16 | CSample::F32 => 32,
    };
    let (w, h) = img.dimensions();
    let channels = u16::try_from(img.layout().channels()).map_err(|_| IoError::Unsupported("too many channels".into()))?;
    let header = Header::new(isd.version, w, h, channels, depth, mode);
    let data = planar_big_endian(&img, color_channels, mode == PsdMode::Cmyk)?;
    let image_data = ImageData { compression: Compression::Raw, data };
    let file = isd.into_psd(header, resources, image_data);
    // The same nesting limit as opening a PSD (#2940).
    crate::psd_import::check_group_depth(&file)?;
    let (mut doc, w) = psd_to_document(&file);
    warnings.extend(w);
    doc.name = name.to_string();
    if doc.icc_profile.is_none() {
        doc.icc_profile = img.icc.clone().map(Arc::new);
    }
    doc.metadata.text = img.meta.text.clone();
    Ok(ImportResult { document: doc, warnings, source_read_only: false, preview_only: false })
}

/// Interleaved native-endian samples → planar big-endian planes (PSD merged-image order),
/// inverting the colour channels of CMYK (PSD stores ink as 255 − value).
fn planar_big_endian(img: &Image, color_channels: usize, invert_color: bool) -> Result<Vec<u8>, IoError> {
    let (w, h) = img.dimensions();
    let n = w as usize * h as usize;
    let ch = img.layout().channels();
    let bps = img.sample_type().bytes();
    let mut out = try_buffer(n * ch, bps)?;
    out.resize(n * ch * bps, 0);
    let data = img.data();
    for (c, plane) in out.chunks_exact_mut(n * bps).enumerate() {
        let invert = invert_color && c < color_channels;
        let pixels = data.chunks_exact(ch * bps).map(|px| &px[c * bps..(c + 1) * bps]);
        for (dst, src) in plane.chunks_exact_mut(bps).zip(pixels) {
            match (bps, src) {
                (1, [v]) => dst[0] = if invert { 255 - *v } else { *v },
                (2, [a, b]) => {
                    let v = u16::from_ne_bytes([*a, *b]);
                    dst.copy_from_slice(&(if invert { 65535 - v } else { v }).to_be_bytes());
                }
                (4, [a, b, c, d]) => {
                    let v = f32::from_ne_bytes([*a, *b, *c, *d]);
                    dst.copy_from_slice(&(if invert { 1.0 - v } else { v }).to_be_bytes());
                }
                _ => return Err(IoError::Unsupported(format!("{bps}-byte samples"))),
            }
        }
    }
    Ok(out)
}

/// Planar big-endian planes (the PSD merged image) → an interleaved native-endian codec image
/// of `channels` channels, un-inverting CMYK colour channels.
fn interleave(planes: &[u8], w: u32, h: u32, channels: usize, color_channels: usize, sample: CSample, invert_color: bool) -> Result<Vec<u8>, IoError> {
    let n = w as usize * h as usize;
    let bps = sample.bytes();
    let plane = n * bps;
    if planes.len() < plane * channels {
        return Err(IoError::Unsupported("the merged image is shorter than its header says".into()));
    }
    let mut out = try_buffer(n * channels, bps)?;
    for i in 0..n {
        for c in 0..channels {
            let at = c * plane + i * bps;
            let invert = invert_color && c < color_channels;
            match planes.get(at..at + bps) {
                Some([v]) => out.push(if invert { 255 - *v } else { *v }),
                Some([a, b]) => {
                    let v = u16::from_be_bytes([*a, *b]);
                    out.extend_from_slice(&(if invert { 65535 - v } else { v }).to_ne_bytes());
                }
                Some([a, b, c, d]) => {
                    let v = f32::from_be_bytes([*a, *b, *c, *d]);
                    out.extend_from_slice(&(if invert { 1.0 - v } else { v }).to_ne_bytes());
                }
                _ => return Err(IoError::Unsupported(format!("{bps}-byte samples"))),
            }
        }
    }
    Ok(out)
}

/// A layered TIFF export's progress once the layers and composite are converted, the layer
/// block is assembled and the composite interleaved; the TIFF encoder does the rest (#3389).
/// Measured on a 24 MP, 5-layer RGB/8 document: about 40 %, 70 % and 72 % of the time.
const TIFF_MODEL_DONE: f32 = 0.4;
const TIFF_LAYERS_DONE: f32 = 0.68;
const TIFF_COMPOSITE_DONE: f32 = 0.72;

/// Saves `doc` as a layered TIFF. Documents whose colour mode a TIFF cannot hold in our
/// codec (Lab) are saved flat, with a warning. Reports progress through `ctl` (see
/// [`crate::export_with`]).
pub(crate) fn export_layered(doc: &Document, opts: &ExportOptions, ctl: &photocraft_raster::Interrupt) -> Result<ExportResult, IoError> {
    if !layered_mode(doc) {
        let mut r = crate::flat::export_flat_with(doc, Format::Tiff, opts, ctl)?;
        if !doc.layers.is_empty() {
            r.warnings.push(format!("{:?} documents are saved as a flat TIFF: their layers are not kept", doc.pixel_format().mode));
        }
        return Ok(r);
    }
    let mut warnings = Vec::new();
    let big = doc.size.width > 30_000 || doc.size.height > 30_000;
    // The PSD exporter renders the composite we need for the TIFF itself: unmatted (the TIFF
    // keeps straight alpha) and uncompressed (its planes are interleaved below).
    let psd_opts =
        PsdExportOptions { force_psb: opts.force_psb || big, merged_matte: false, merged_compression: Compression::Raw, maximize_compatibility: true };
    let (file, w) = crate::with_span(ctl, 0.0, TIFF_MODEL_DONE, |ctl| crate::psd_export::document_to_psd_interruptible(doc, &psd_opts, ctl));
    // A cancelled conversion stops early with an incomplete model: never encode it.
    ctl.check().map_err(|_| IoError::Cancelled)?;
    warnings.extend(w);
    let order = if codecs::tiff_writes_little_endian() { ByteOrder::Little } else { ByteOrder::Big };
    let (layers, w) = ImageSourceData::from_psd(&file).to_bytes(order)?;
    ctl.check().map_err(|_| IoError::Cancelled)?;
    ctl.progress(TIFF_LAYERS_DONE);
    warnings.extend(w.into_iter().map(|w| format!("layer data: {w}")));
    // The profile and XMP are written as TIFF tags, not as resources. EXIF stays a resource
    // (1058), which the PSD importer reads back unchanged; a block rebuilt from the TIFF's own
    // Exif directory keeps only the Exif, GPS and descriptive tags.
    let resources: Vec<ImageResource> = file.resources.iter().filter(|r| !matches!(r.id, ids::ICC_PROFILE | ids::XMP)).cloned().collect();
    let resources = resources_to_bytes(&resources)?;

    let fmt = doc.pixel_format();
    let cc = fmt.mode.color_channels();
    let has_alpha = file.layer_info.as_ref().is_some_and(|l| l.merged_alpha);
    let channels = cc + usize::from(has_alpha);
    let extra = usize::from(file.header.channels).saturating_sub(channels);
    if extra > 0 {
        warnings.push(format!("{extra} alpha channel(s) are not written to TIFF (one transparency channel only)"));
    }
    let sample = csample(fmt.sample);
    let (w, h) = (doc.size.width, doc.size.height);
    let cmyk = fmt.mode == ColorMode::Cmyk;
    let data = interleave(&file.image_data.data, w, h, channels, cc, sample, cmyk)?;
    ctl.check().map_err(|_| IoError::Cancelled)?;
    ctl.progress(TIFF_COMPOSITE_DONE);
    let layout = layout_for(fmt.mode, has_alpha);
    let mut img = Image::from_raw(w, h, layout, sample, data)?;
    img.icc = doc.icc_profile.as_ref().map(|i| i.to_vec());
    img.meta = codecs::Metadata {
        // EXIF is deliberately not set here: the tags above carry it as resource 1058 and the
        // importer reads it back from there unchanged. A second copy in `meta.exif` (the codec's
        // Exif directory) would be read back in its place, rebuilt without the thumbnail and IFD0
        // orientation (#1545).
        xmp: doc.metadata.xmp.clone().filter(|_| opts.xmp == crate::XmpEmbed::All),
        text: if opts.xmp == crate::XmpEmbed::All { doc.metadata.text.clone() } else { Vec::new() },
        dpi: Some((doc.resolution_dpi, doc.resolution_dpi)),
        photoshop_resources: Some(resources),
        photoshop_layers: Some(layers),
        ..Default::default()
    };
    encode_image(&img, Format::Tiff, opts, warnings)
}

/// Removes the blocks a document kept verbatim from a little-endian TIFF because their layout
/// is unknown ([`photocraft_psd::TaggedBlock::is_foreign_order`]) before `file` is written as a
/// PSD: their bytes are in the wrong order for a PSD and cannot be converted. Returns a warning
/// per block dropped. A save back to TIFF keeps them (see [`photocraft_psd::tiff`]).
pub fn strip_foreign_order_blocks(file: &mut photocraft_psd::PsdFile) -> Vec<String> {
    let mut warnings = Vec::new();
    let mut strip = |blocks: &mut Vec<photocraft_psd::TaggedBlock>, owner: &str| {
        blocks.retain(|b| {
            let keep = !b.is_foreign_order();
            if !keep {
                warnings.push(format!(
                    "{owner}: {} block ({} bytes) kept from a little-endian TIFF cannot be written to PSD and was dropped",
                    b.key_str(),
                    b.data.len()
                ));
            }
            keep
        });
    };
    if let Some(info) = &mut file.layer_info {
        for l in &mut info.layers {
            let name = l.name();
            strip(&mut l.blocks, &format!("layer \"{name}\""));
        }
    }
    strip(&mut file.global_blocks, "document");
    warnings
}
