//! Multichannel documents <-> PSD colour mode 7.
//!
//! A Multichannel document has no layers: its image is a set of ink (spot) channels, stored in
//! [`Document::channels`] with ink density as the value (1 = solid ink, like every spot channel).
//! A PSD stores the same planes the way Photoshop shows them in the Channels panel: 0 (black) =
//! solid ink, max = paper. Channel names come from resources 1006 / 1045 and ink colours from
//! DisplayInfo (1077), exactly as for spot channels in the other modes.

use photocraft_color::{Color, ColorMode, PixelFormat, SampleType};
use photocraft_doc::{AlphaChannel, Document};
use photocraft_geom::Rect;
use photocraft_psd::file::{GlobalLayerMask, LayerInfoPlacement};
use photocraft_psd::resources::{ImageResource, ResolutionInfo, ids, version_info_resource};
use photocraft_psd::{ColorMode as PsdMode, Compression, Header, ImageData, PsdFile, TaggedBlock};
use photocraft_raster::Surface;

use crate::pixels::{encode_be, psd_depth};

/// Grid and guides.
const GRID_AND_GUIDES: u16 = 1032;
/// Alpha identifiers: one u32 per extra channel.
const ALPHA_IDENTIFIERS: u16 = 1053;
/// Resources that only describe layers: target layer (1024), layer groups (1026), layer
/// selection ids (1069) and layer groups enabled (1072). A Multichannel document has no layers.
const LAYER_ONLY_RESOURCES: [u16; 4] = [ids::LAYER_STATE, ids::LAYER_GROUP_INFO, 1069, 1072];

/// Value of sample `i` of a big-endian plane.
fn decode(plane: &[u8], i: usize, s: SampleType) -> f32 {
    match s {
        SampleType::U8 => f32::from(plane[i]) / 255.0,
        SampleType::U16 => f32::from(u16::from_be_bytes([plane[2 * i], plane[2 * i + 1]])) / 65535.0,
        SampleType::F32 => f32::from_be_bytes([plane[4 * i], plane[4 * i + 1], plane[4 * i + 2], plane[4 * i + 3]]),
    }
}

/// Fills `doc` (already in Multichannel mode at the file depth) from the merged planes `all`.
pub(crate) fn import(file: &PsdFile, all: &[u8], doc: &mut Document, names: &[String], warnings: &mut Vec<String>) {
    let h = &file.header;
    let depth = doc.depth;
    let (w, hh) = (h.width as usize, h.height as usize);
    let n = w * hh;
    let plane = h.row_bytes() * hh;
    let canvas = Rect::new(0, 0, h.width as i32, h.height as i32);
    let fmt = PixelFormat::new(ColorMode::Grayscale, depth, false);
    for k in 0..usize::from(h.channels) {
        let Some(p) = all.get(k * plane..(k + 1) * plane) else {
            warnings.push(format!("multichannel: channel {} is missing", k + 1));
            break;
        };
        let vals: Vec<f32> = (0..n).map(|i| 1.0 - decode(p, i, depth)).collect();
        let mut s = Surface::new(fmt);
        s.write_region(canvas, &vals);
        s.prune();
        let name = names.get(k).cloned().filter(|s| !s.is_empty()).unwrap_or_else(|| format!("Channel {}", k + 1));
        // Without DisplayInfo the channels print in black, as Photoshop shows them. DisplayInfo
        // turns alpha-type channels (selected / masked areas) back into alpha channels.
        doc.channels.push(AlphaChannel { spot: Some((Color::BLACK, 0.0)), ..AlphaChannel::new(name, s) });
    }
    if let Some(r) = file.resource(crate::channel_map::DISPLAY_INFO) {
        crate::channel_map::apply_display_info(&r.data, true, 0, &mut doc.channels);
    } else if let Some(r) = file.resource(crate::channel_map::DISPLAY_INFO_OLD) {
        crate::channel_map::apply_display_info(&r.data, false, 0, &mut doc.channels);
    }
}

/// Writes a Multichannel document as a mode-7 PSD (no layer section; every channel a plane).
pub(crate) fn document_to_psd(doc: &Document, force_psb: bool) -> (PsdFile, Vec<String>) {
    let mut warnings = Vec::new();
    // Photoshop's Multichannel mode is 8 or 16 bits per channel.
    let sample = match doc.depth {
        SampleType::F32 => {
            warnings.push("32-bit Multichannel written as 16 bits/channel (32-bit Multichannel isn't supported)".into());
            SampleType::U16
        }
        s => s,
    };
    if !doc.layers.is_empty() {
        warnings.push("Multichannel documents have no layers: the layers were not written".into());
    }
    let canvas = doc.bounds();
    let channels: Vec<&AlphaChannel> = doc.channels.iter().take(56).collect();
    if doc.channels.len() > 56 {
        warnings.push("only 56 channels fit in PSD; the rest were dropped".into());
    }
    let mut planes = Vec::new();
    if channels.is_empty() {
        // A PSD needs at least one channel: an empty (paper) plane.
        for _ in 0..canvas.width() as usize * canvas.height() as usize {
            encode_be(1.0, sample, &mut planes);
        }
    }
    for ch in &channels {
        for v in ch.surface.read_region(canvas).chunks_exact(ch.surface.channels()) {
            encode_be(1.0 - v[0], sample, &mut planes);
        }
    }
    let big = doc.size.width > 30_000 || doc.size.height > 30_000;
    let size_exceeds_limit = !force_psb && !big && crate::psd_export::psd_size_exceeds_limit(doc);
    let version = crate::psd_export::psd_version(doc, force_psb, size_exceeds_limit);
    if big && !force_psb {
        warnings.push("document exceeds 30000 px; written as PSB".into());
    } else if size_exceeds_limit && !force_psb {
        warnings.push("estimated encoded size exceeds 2 GB; written as PSB".into());
    }
    let header = Header::new(version, doc.size.width, doc.size.height, channels.len().max(1) as u16, psd_depth(sample), PsdMode::Multichannel);
    let image_data = ImageData::encode(Compression::Rle, &planes, &header)
        .or_else(|_| ImageData::encode(Compression::Raw, &planes, &header))
        .unwrap_or(ImageData { compression: Compression::Raw, data: planes });
    let mut resources = vec![
        ImageResource::new(ids::RESOLUTION_INFO, ResolutionInfo::from_dpi(f64::from(doc.resolution_dpi)).to_bytes()),
        ImageResource::new(ids::GLOBAL_ANGLE, (doc.global_light.angle.round() as i32).to_be_bytes().to_vec()),
        ImageResource::new(ids::GLOBAL_ALTITUDE, (doc.global_light.altitude.round() as i32).to_be_bytes().to_vec()),
    ];
    // Grid and guides: Photoshop writes it in every file, guides or not.
    resources.push(ImageResource::new(GRID_AND_GUIDES, crate::psd_export::guides_resource(doc)));
    if !channels.is_empty() {
        let names: Vec<&str> = channels.iter().map(|c| c.name.as_str()).collect();
        resources.push(ImageResource::new(1006, crate::psd_export::pascal_names_resource(&names)));
        resources.push(ImageResource::new(1045, crate::psd_export::unicode_names_resource(&names)));
        resources.push(ImageResource::new(crate::channel_map::DISPLAY_INFO, crate::channel_map::display_info(&channels, false)));
    }
    resources.extend(crate::psd_export::path_resources(doc, &mut warnings));
    // The XMP and EXIF resolution follow the ResolutionInfo resource, so no copy contradicts it.
    let ppi = Some((doc.resolution_dpi, doc.resolution_dpi)).filter(|d| d.0 > 0.0);
    if let Some(x) = &doc.metadata.xmp {
        resources.push(ImageResource::new(ids::XMP, photocraft_codecs::export_xmp(x, ppi).as_bytes().to_vec()));
    }
    if let Some(e) = &doc.metadata.exif {
        resources.push(ImageResource::new(ids::EXIF, photocraft_codecs::export_exif(e, ppi).into_owned()));
    }
    // Every other imported resource (print settings, IPTC, the layer-id seed, ...), except the
    // ones that only describe layers and any already written above.
    let written: Vec<u16> = resources.iter().map(|r| r.id).collect();
    for (id, name, data) in &doc.metadata.psd_resources {
        if LAYER_ONLY_RESOURCES.contains(id) || written.contains(id) || !crate::psd_export::keep_imported_resource(doc, *id, data) {
            continue;
        }
        // Alpha identifiers: one u32 per channel, only while they still match the channels.
        if *id == ALPHA_IDENTIFIERS && data.len() != channels.len().saturating_mul(4) {
            continue;
        }
        let mut r = ImageResource::new(*id, data.to_vec());
        r.name = crate::psd_export::legacy_name(name);
        resources.push(r);
    }
    if let Some(data) = crate::annotations_map::fresh_scale_resource(doc) {
        resources.push(ImageResource::new(crate::annotations_map::MEASUREMENT_SCALE, data));
    }
    resources.push(version_info_resource(true));
    // Document-level tagged blocks (patterns, notes, ...) are valid without a layer section; a
    // kept layer block would describe layers the document doesn't have.
    let global_blocks: Vec<TaggedBlock> = crate::annotations_map::export_blocks(doc, crate::pattern_map::export_global_blocks(doc))
        .into_iter()
        .filter(|(_, key, _)| !matches!(key, b"Layr" | b"Lr16" | b"Lr32"))
        .map(|(sig, key, data)| crate::psd_export::global_block(sig, key, &data))
        .collect();
    let color_mode_data = photocraft_psd::hdr::color_mode_data_for_depth(header.depth);
    let file = PsdFile {
        header,
        color_mode_data,
        resources,
        layer_info: None,
        layer_info_placement: LayerInfoPlacement::Section,
        global_layer_mask: (!global_blocks.is_empty()).then(GlobalLayerMask::default),
        global_blocks,
        layer_mask_trailing: Vec::new(),
        image_data,
    };
    (file, warnings)
}
