//! OpenEXR via the pure-Rust `exr` crate: F16/F32, Y/YA/RGB/RGBA channels,
//! lossless compression. Multi-part files (a Maya/Arnold render writes one part
//! per AOV) open the part that looks most like a colour image, with a warning
//! naming how many parts exist; [`info`] lists every part and [`decode_part`]
//! decodes a chosen one. Pixel data is taken from the data window.

use std::io::Cursor;

use exr::meta::MetaData;
use exr::prelude::{
    AnyChannel, AnyChannels, Blocks, Compression, Encoding, FlatSamples, Layer, LayerAttributes, LineOrder, ReadChannels, ReadLayers, WritableImage, read,
};
use half::f16;

use crate::Format;
use crate::error::CodecError;
use crate::fidelity::Plan;
use crate::image::{ChannelLayout, DecodeWarning, ExrChannelInfo, ExrPartInfo, Image, SampleType, base_channel_name};
use crate::options::{EncodeOptions, ExrCompression, Limits};

const F: Format = Format::OpenExr;

fn err(e: impl std::fmt::Display) -> CodecError {
    CodecError::malformed(F, e)
}

/// An `exr`-crate error, with its `NotSupported` kind kept distinct from malformed data
/// (the same split the TIFF codec makes), so unsupported features say so.
fn map_exr(e: exr::error::Error) -> CodecError {
    match e {
        exr::error::Error::NotSupported(message) => CodecError::unsupported(F, format!("not supported: {message}")),
        e => err(e),
    }
}

/// Read the headers without the `exr` crate's validation (it rejects deep parts outright),
/// after checking every part's declared size against the limits, before any pixel allocation.
pub(super) fn read_meta(bytes: &[u8], limits: &Limits) -> Result<MetaData, CodecError> {
    let meta = MetaData::read_from_buffered(Cursor::new(bytes), false).map_err(err)?;
    // Flat decoding reads every part at once, so the parts' sizes are also capped together.
    let mut total: u64 = 0;
    for header in meta.headers.iter() {
        let size = header.layer_size;
        let (w, h) = (u32::try_from(size.0).unwrap_or(u32::MAX), u32::try_from(size.1).unwrap_or(u32::MAX));
        let bpp = header.channels.list.len().max(1) as u64 * 4;
        limits.check_bytes(w, h, bpp)?;
        total = total.saturating_add(u64::from(w).saturating_mul(u64::from(h)).saturating_mul(bpp));
    }
    if total > limits.max_alloc {
        return Err(CodecError::LimitExceeded(format!("{total} bytes across {} parts exceed max_alloc {}", meta.headers.len(), limits.max_alloc)));
    }
    Ok(meta)
}

/// Header facts of one part, in file order.
fn part_info(index: usize, header: &exr::meta::header::Header) -> ExrPartInfo {
    let (w, h) = (u32::try_from(header.layer_size.0).unwrap_or(u32::MAX), u32::try_from(header.layer_size.1).unwrap_or(u32::MAX));
    let sample = |t: exr::meta::attribute::SampleType| match t {
        exr::meta::attribute::SampleType::F16 => SampleType::F16,
        // U32 channels (Arnold ID buffers) are listed as F32: exact up to 2^24.
        exr::meta::attribute::SampleType::F32 | exr::meta::attribute::SampleType::U32 => SampleType::F32,
    };
    ExrPartInfo {
        index,
        name: header.own_attributes.layer_name.as_ref().map(|n| n.to_string()),
        view: header.own_attributes.view_name.as_ref().map(|v| v.to_string()),
        width: w,
        height: h,
        deep: header.deep,
        tiled: matches!(header.blocks, exr::meta::BlockDescription::Tiles(_)),
        channels: header.channels.list.iter().map(|c| ExrChannelInfo { name: c.name.to_string(), sample: sample(c.sample_type) }).collect(),
    }
}

/// Lists every part of the file (header facts only, no pixel data): what a multi-part
/// chooser offers, and what [`decode_part`] takes an index into.
pub(crate) fn info(bytes: &[u8], limits: &Limits) -> Result<Vec<ExrPartInfo>, CodecError> {
    let meta = read_meta(bytes, limits)?;
    Ok(meta.headers.iter().enumerate().map(|(i, h)| part_info(i, h)).collect())
}

/// The part plain [`decode`] opens: the highest [`ExrPartInfo::color_rank`], earliest wins.
fn pick_part(meta: &MetaData) -> usize {
    let mut best = 0;
    let mut best_rank = meta.headers.first().map_or(0, |h| part_info(0, h).color_rank());
    for (i, header) in meta.headers.iter().enumerate().skip(1) {
        let rank = part_info(i, header).color_rank();
        if rank > best_rank {
            best = i;
            best_rank = rank;
        }
    }
    best
}

/// Selects one part for the `exr` crate's first-valid-layer reader: `create_channels_reader`
/// is asked about the headers in file order and the first `Ok` wins, so every header before
/// `index` (and, harmlessly, those after it) is declined. Only the chosen part's chunks are
/// then decompressed and only its planes allocated; the other parts are skipped unread.
struct OnlyPart<C> {
    inner: C,
    index: usize,
    seen: std::cell::Cell<usize>,
}

impl<'s, C: ReadChannels<'s>> ReadChannels<'s> for OnlyPart<C> {
    type Reader = C::Reader;

    fn create_channels_reader(&'s self, header: &exr::meta::header::Header) -> exr::error::Result<Self::Reader> {
        let position = self.seen.get();
        self.seen.set(position.saturating_add(1));
        if position == self.index { self.inner.create_channels_reader(header) } else { Err(exr::error::Error::Invalid("part not requested".into())) }
    }
}

/// Decode one part's flat layer via the `exr` crate (it handles every flat compression).
/// Other parts are not decompressed, so an AOV file costs one part per request, not the file.
pub(super) fn read_layer(bytes: &[u8], index: usize) -> Result<Layer<AnyChannels<FlatSamples>>, CodecError> {
    let only = OnlyPart { inner: read().no_deep_data().largest_resolution_level().all_channels(), index, seen: std::cell::Cell::new(0) };
    only.first_valid_layer().all_attributes().from_buffered(Cursor::new(bytes)).map(|image| image.layer_data).map_err(map_exr)
}

fn layer_to_image(layer: &Layer<AnyChannels<FlatSamples>>) -> Result<Image, CodecError> {
    let (w, h) = (layer.size.0, layer.size.1);
    let channels = &layer.channel_data.list;
    let find = |n: &str| channels.iter().find(|c| base_channel_name(&c.name.to_string()) == n);
    let (color, layout): (Vec<_>, ChannelLayout) = if let (Some(r), Some(g), Some(b)) = (find("R"), find("G"), find("B")) {
        match find("A") {
            Some(a) => (vec![r, g, b, a], ChannelLayout::Rgba),
            None => (vec![r, g, b], ChannelLayout::Rgb),
        }
    } else if let Some(y) = find("Y").or_else(|| (channels.len() == 1).then(|| &channels[0])) {
        match find("A") {
            Some(a) => (vec![y, a], ChannelLayout::GrayA),
            None => (vec![y], ChannelLayout::Gray),
        }
    } else {
        return Err(CodecError::unsupported(F, "no R/G/B or Y channels"));
    };
    let all_f16 = color.iter().all(|c| matches!(c.sample_data, FlatSamples::F16(_)));
    let sample = if all_f16 { SampleType::F16 } else { SampleType::F32 };
    let npx = w * h;
    if color.iter().any(|c| c.sample_data.len() != npx) {
        return Err(err("channel sample count mismatch (subsampled channels are unsupported)"));
    }
    let nc = color.len();
    let (w32, h32) = (w as u32, h as u32);
    let img = if all_f16 {
        let mut out = vec![f16::ZERO; npx * nc];
        for (ci, c) in color.iter().enumerate() {
            if let FlatSamples::F16(v) = &c.sample_data {
                for (i, s) in v.iter().enumerate() {
                    out[i * nc + ci] = *s;
                }
            }
        }
        Image::from_f16(w32, h32, layout, &out)?
    } else {
        let mut out = vec![0f32; npx * nc];
        for (ci, c) in color.iter().enumerate() {
            for (i, s) in c.sample_data.values_as_f32().enumerate() {
                out[i * nc + ci] = s;
            }
        }
        Image::from_f32(w32, h32, layout, &out)?
    };
    debug_assert_eq!(img.sample_type(), sample);
    Ok(img)
}

pub(crate) fn decode(bytes: &[u8], limits: &Limits) -> Result<Image, CodecError> {
    let meta = read_meta(bytes, limits)?;

    // Deep parts are rejected by the `exr` crate's reader; walk their chunks ourselves and
    // composite the samples into the flat image (a warning says what was lost).
    if meta.requirements.has_deep_data {
        let deep = super::deep_exr::decode_deep(&meta, bytes, limits)?;
        return super::deep_exr::flatten(&deep);
    }

    let pick = pick_part(&meta);
    let mut img = layer_to_image(&read_layer(bytes, pick)?)?;
    if meta.headers.len() > 1 {
        let total = u32::try_from(meta.headers.len()).ok();
        img.warnings.push(DecodeWarning::MoreParts { total });
    }
    Ok(img)
}

/// Decode one part of a multi-part file by its index into [`info`]'s list. A file with
/// deep parts is not mixed: the deep decoder composites those instead.
pub(crate) fn decode_part(bytes: &[u8], index: usize, limits: &Limits) -> Result<Image, CodecError> {
    let meta = read_meta(bytes, limits)?;
    if meta.requirements.has_deep_data {
        return Err(CodecError::unsupported(F, "this file has deep parts; the deep decoder composites those (mixing flat and deep parts is not supported)"));
    }
    if index >= meta.headers.len() {
        return Err(err(format!("part index {index} is out of range ({} parts)", meta.headers.len())));
    }
    layer_to_image(&read_layer(bytes, index)?)
}

/// Decodes the named channels of one flat part (exact names as the file stores them, e.g.
/// `diffuse.R`) into an image: R/G/B (plus A when named) give Rgba, a lone Y gives Gray. This
/// is how a channel group renders without the part's other channels in the way.
pub(crate) fn decode_channels(bytes: &[u8], index: usize, names: &[&str], limits: &Limits) -> Result<Image, CodecError> {
    let meta = read_meta(bytes, limits)?;
    if meta.requirements.has_deep_data {
        return Err(CodecError::unsupported(F, "this file has deep parts; the deep decoder composites those (mixing flat and deep parts is not supported)"));
    }
    if index >= meta.headers.len() {
        return Err(err(format!("part index {index} is out of range ({} parts)", meta.headers.len())));
    }
    let layer = read_layer(bytes, index)?;
    let (w, h) = (layer.size.0, layer.size.1);
    let channels = &layer.channel_data.list;
    let npx = w * h;
    let mut picked: Vec<&AnyChannel<FlatSamples>> = Vec::with_capacity(names.len());
    for name in names {
        let c = channels.iter().find(|c| c.name.to_string() == *name).ok_or_else(|| err(format!("the part has no channel {name}")))?;
        if c.sample_data.len() != npx {
            return Err(err("channel sample count mismatch (subsampled channels are unsupported)"));
        }
        picked.push(c);
    }
    let base = |c: &AnyChannel<FlatSamples>| -> String { base_channel_name(&c.name.to_string()).to_string() };
    let has = |n: &str| picked.iter().any(|c| base(c) == n);
    let (layout, order): (ChannelLayout, Vec<Option<usize>>) = if has("R") && has("G") && has("B") {
        let at = |n: &str| picked.iter().position(|c| base(c) == n);
        match has("A").then(|| at("A")) {
            Some(a) => (ChannelLayout::Rgba, vec![at("R"), at("G"), at("B"), a]),
            None => (ChannelLayout::Rgb, vec![at("R"), at("G"), at("B"), None]),
        }
    } else if has("Y") {
        let y = picked.iter().position(|c| base(c) == "Y");
        match picked.iter().position(|c| base(c) == "A") {
            Some(a) => (ChannelLayout::GrayA, vec![y, Some(a)]),
            None => (ChannelLayout::Gray, vec![y]),
        }
    } else if picked.len() == 1 {
        (ChannelLayout::Gray, vec![Some(0)])
    } else {
        return Err(CodecError::unsupported(F, "the named channels are no R/G/B or Y set"));
    };
    let all_f16 = picked.iter().all(|c| matches!(c.sample_data, FlatSamples::F16(_)));
    let nc = layout.channels();
    let (w32, h32) = (w as u32, h as u32);
    // The None slots sit past the layout's own channels (the absent A of an RGB pick) and
    // are skipped, so the buffer keeps its pixel stride.
    let img = if all_f16 {
        let mut out = vec![f16::ZERO; npx * nc];
        for (ci, pick) in order.iter().enumerate() {
            let Some(idx) = pick else { continue };
            let Some(FlatSamples::F16(v)) = picked.get(*idx).map(|c| &c.sample_data) else { continue };
            for (i, s) in v.iter().enumerate() {
                out[i * nc + ci] = *s;
            }
        }
        Image::from_f16(w32, h32, layout, &out)
    } else {
        let mut out = vec![0f32; npx * nc];
        for (ci, pick) in order.iter().enumerate() {
            let Some(idx) = pick else { continue };
            let Some(c) = picked.get(*idx) else { continue };
            for (i, s) in c.sample_data.values_as_f32().enumerate() {
                out[i * nc + ci] = s;
            }
        }
        Image::from_f32(w32, h32, layout, &out)
    };
    img.map_err(|e| CodecError::encode(F, e))
}

pub(crate) fn encode(src: &Image, plan: Plan, opts: &EncodeOptions) -> Result<Vec<u8>, CodecError> {
    let img = src.converted(plan.layout, plan.sample);
    let (w, h) = (img.width() as usize, img.height() as usize);
    let layout = img.layout();
    let names: &[&str] = match layout {
        ChannelLayout::Gray => &["Y"],
        ChannelLayout::GrayA => &["Y", "A"],
        ChannelLayout::Rgb => &["R", "G", "B"],
        ChannelLayout::Rgba => &["R", "G", "B", "A"],
        l => return Err(CodecError::encode(F, format!("unsupported layout {l:?}"))),
    };
    let nc = names.len();
    let mut list: Vec<AnyChannel<FlatSamples>> = Vec::with_capacity(nc);
    match img.sample_type() {
        SampleType::F16 => {
            let s = img.to_f16_samples().unwrap_or_default();
            for (ci, n) in names.iter().enumerate() {
                let v: Vec<f16> = s.iter().skip(ci).step_by(nc).copied().collect();
                list.push(AnyChannel::new(*n, FlatSamples::F16(v)));
            }
        }
        SampleType::F32 => {
            let s = img.to_f32_samples().unwrap_or_default();
            for (ci, n) in names.iter().enumerate() {
                let v: Vec<f32> = s.iter().skip(ci).step_by(nc).copied().collect();
                list.push(AnyChannel::new(*n, FlatSamples::F32(v)));
            }
        }
        s => return Err(CodecError::encode(F, format!("unsupported sample {s:?}"))),
    }
    let compression = match opts.exr_compression {
        ExrCompression::None => Compression::Uncompressed,
        ExrCompression::Rle => Compression::RLE,
        ExrCompression::Zip1 => Compression::ZIP1,
        ExrCompression::Zip16 => Compression::ZIP16,
        ExrCompression::Piz => Compression::PIZ,
    };
    let encoding = Encoding { compression, blocks: Blocks::ScanLines, line_order: LineOrder::Increasing };
    let channels = AnyChannels::sort(list.into_iter().collect());
    let layer = Layer::new((w, h), LayerAttributes::default(), encoding, channels);
    let image = exr::image::Image::from_layer(layer);
    let mut cursor = Cursor::new(Vec::new());
    image.write().to_buffered(&mut cursor).map_err(|e| CodecError::encode(F, e))?;
    Ok(cursor.into_inner())
}
