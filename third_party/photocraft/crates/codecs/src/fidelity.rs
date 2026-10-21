//! Encode planning and fidelity warnings. `encode` and `fidelity_warnings`
//! share [`plan`], so a warning is emitted exactly when the encoder will
//! actually transform or drop something.

use std::fmt;

use crate::format::{Format, caps};
use crate::image::{ChannelLayout, Image, SampleType};
use crate::options::EncodeOptions;

/// Something that will be lost or altered when writing an image to a format.
#[derive(Debug, Clone, PartialEq)]
pub enum FidelityWarning {
    /// The format cannot be written in this build.
    WriteUnsupported {
        format: Format,
    },
    /// Bit depth / precision reduced, e.g. 16-bit → 8-bit or f32 → f16.
    DepthReduced {
        from: SampleType,
        to: SampleType,
    },
    /// Float samples outside `[0, 1]` will be clipped by integer storage.
    RangeClipped,
    /// Non-opaque alpha will be discarded.
    AlphaDiscarded,
    /// Alpha will be reduced to fully transparent / fully opaque (GIF).
    AlphaBinarized,
    /// CMYK data will be converted (naively, not colour-managed) to another model.
    CmykConverted {
        to: ChannelLayout,
    },
    /// Colour data will be reduced to grayscale.
    ColorToGray,
    /// The image will be quantized to a 256-colour palette.
    PaletteQuantized,
    /// The format's encoder is lossy.
    LossyCompression,
    IccDropped,
    ExifDropped,
    XmpDropped,
    DpiDropped,
    TextDropped,
    /// Metadata larger than the format can hold in one piece (`what`: "EXIF", "XMP") will be
    /// dropped.
    MetadataTooLarge {
        what: &'static str,
        bytes: usize,
    },
    /// Image exceeds the format's maximum size; encoding will fail.
    DimensionsExceeded {
        max_width: u32,
        max_height: u32,
    },
}

impl FidelityWarning {
    /// `true` for warnings that make encoding fail outright.
    pub fn is_fatal(&self) -> bool {
        matches!(self, FidelityWarning::WriteUnsupported { .. } | FidelityWarning::DimensionsExceeded { .. })
    }
}

fn depth_name(s: SampleType) -> &'static str {
    match s {
        SampleType::U8 => "8-bit",
        SampleType::U16 => "16-bit",
        SampleType::F16 => "16-bit float",
        SampleType::F32 => "32-bit float",
    }
}

impl fmt::Display for FidelityWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use FidelityWarning::*;
        match self {
            WriteUnsupported { format } => {
                write!(f, "{} cannot be written in this build", format.name())
            }
            DepthReduced { from, to } => write!(f, "{} will be reduced to {}", depth_name(*from), depth_name(*to)),
            RangeClipped => write!(f, "HDR values outside 0..1 will be clipped"),
            AlphaDiscarded => write!(f, "alpha will be discarded"),
            AlphaBinarized => write!(f, "alpha will be reduced to on/off transparency"),
            CmykConverted { to } => {
                write!(f, "CMYK will be converted to {to:?} (not colour-managed)")
            }
            ColorToGray => write!(f, "colour will be converted to grayscale"),
            PaletteQuantized => write!(f, "colours will be quantized to a 256-colour palette"),
            LossyCompression => write!(f, "lossy compression"),
            IccDropped => write!(f, "ICC profile not supported; it will be dropped"),
            ExifDropped => write!(f, "EXIF not supported; it will be dropped"),
            XmpDropped => write!(f, "XMP not supported; it will be dropped"),
            DpiDropped => write!(f, "resolution (DPI) not supported; it will be dropped"),
            TextDropped => write!(f, "text metadata not supported; it will be dropped"),
            MetadataTooLarge { what, bytes } => write!(f, "{what} ({bytes} bytes) is too large for the format; it will be dropped"),
            DimensionsExceeded { max_width, max_height } => {
                write!(f, "image exceeds the format maximum of {max_width}x{max_height}")
            }
        }
    }
}

/// Target storage chosen for an image/format pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Plan {
    pub layout: ChannelLayout,
    pub sample: SampleType,
}

fn pick_sample(src: SampleType, depths: &[SampleType]) -> SampleType {
    if depths.contains(&src) {
        return src;
    }
    let pref: &[SampleType] = match src {
        // Integers widen into float losslessly; F32 represents U8/U16 exactly.
        SampleType::U8 => &[SampleType::U16, SampleType::F32, SampleType::F16],
        SampleType::U16 => &[SampleType::F32, SampleType::U8, SampleType::F16],
        SampleType::F16 => &[SampleType::F32, SampleType::U16, SampleType::U8],
        SampleType::F32 => &[SampleType::F16, SampleType::U16, SampleType::U8],
    };
    pref.iter().copied().find(|s| depths.contains(s)).unwrap_or(depths[0])
}

fn pick_layout(src: ChannelLayout, layouts: &[ChannelLayout], alpha: bool) -> ChannelLayout {
    let mut cand = src;
    if !alpha {
        cand = cand.without_alpha();
    }
    if layouts.contains(&cand) {
        return cand;
    }
    // CMYK falls back to RGB; gray expands to RGB.
    if cand.is_cmyk() || cand.is_gray() {
        cand = if cand.has_alpha() { ChannelLayout::Rgba } else { ChannelLayout::Rgb };
    }
    if layouts.contains(&cand) {
        return cand;
    }
    if layouts.contains(&cand.with_alpha()) {
        return cand.with_alpha();
    }
    if layouts.contains(&cand.without_alpha()) {
        return cand.without_alpha();
    }
    layouts[0]
}

pub(crate) fn plan(image: &Image, format: Format, opts: &EncodeOptions) -> Plan {
    let c = caps(format);
    let (layout, sample) = (image.layout(), image.sample_type());
    match format {
        // PNM: integer data can use any layout (PAM); float goes to PFM
        // which only has gray and RGB.
        Format::Pnm if sample.is_float() => Plan { layout: pick_layout(layout, &[ChannelLayout::Gray, ChannelLayout::Rgb], false), sample: SampleType::F32 },
        // DPX has no luma + alpha descriptor: gray with alpha is written as luma, alpha dropped.
        // The sample type follows the chosen bits (U8 for 8, U16 for 10, 12 and 16).
        Format::Dpx => Plan {
            layout: if layout == ChannelLayout::GrayA { ChannelLayout::Gray } else { pick_layout(layout, c.layouts, c.alpha) },
            sample: crate::codecs::dpx::plan_sample(opts.dpx_bits, sample),
        },
        _ => Plan { layout: pick_layout(layout, c.layouts, c.alpha), sample: pick_sample(sample, c.depths) },
    }
}

/// Warnings for writing `image` as `format` with default options.
pub fn fidelity_warnings(image: &Image, format: Format) -> Vec<FidelityWarning> {
    fidelity_warnings_with(image, format, &EncodeOptions::default())
}

/// Warnings for writing `image` as `format` with the given options.
pub fn fidelity_warnings_with(image: &Image, format: Format, opts: &EncodeOptions) -> Vec<FidelityWarning> {
    use FidelityWarning as W;
    let c = caps(format);
    let mut w = Vec::new();
    if !c.write {
        w.push(W::WriteUnsupported { format });
        return w;
    }
    if let Some((mw, mh)) = format.max_dimensions()
        && (image.width() > mw || image.height() > mh)
    {
        w.push(W::DimensionsExceeded { max_width: mw, max_height: mh });
    }
    let p = plan(image, format, opts);
    let (sl, ss) = (image.layout(), image.sample_type());

    // Depth
    let lossy_depth = match (ss, p.sample) {
        (a, b) if a == b => false,
        (SampleType::U8, _) => false,
        (SampleType::U16, SampleType::F32) => false,
        (SampleType::F16, SampleType::F32) => false,
        _ => true,
    };
    if lossy_depth {
        w.push(W::DepthReduced { from: ss, to: p.sample });
    }
    if ss.is_float() && !p.sample.is_float() && image.has_out_of_range() {
        w.push(W::RangeClipped);
    }

    // Layout
    if sl.has_alpha() && !p.layout.has_alpha() && image.has_translucency() {
        w.push(W::AlphaDiscarded);
    }
    if format == Format::Gif && p.layout.has_alpha() && image.has_translucency() {
        w.push(W::AlphaBinarized);
    }
    if sl.is_cmyk() && !p.layout.is_cmyk() {
        w.push(W::CmykConverted { to: p.layout });
    }
    if !sl.is_gray() && p.layout.is_gray() {
        w.push(W::ColorToGray);
    }

    // Compression
    if format == Format::Gif {
        w.push(W::PaletteQuantized);
    } else if c.lossy || (format == Format::WebP && !opts.webp_lossless) {
        w.push(W::LossyCompression);
    }

    // Metadata
    if image.icc.is_some() && opts.embed_icc && (!c.icc || !sl.same_model(p.layout)) {
        w.push(W::IccDropped);
    }
    if opts.embed_metadata {
        let m = &image.meta;
        if m.exif.is_some() && !c.exif {
            w.push(W::ExifDropped);
        }
        if m.xmp.is_some() && !c.xmp {
            w.push(W::XmpDropped);
        }
        if format == Format::Jpeg {
            if let Some(exif) = m.exif.as_deref().filter(|e| crate::codecs::jpeg::exif_segment(e, None).is_none()) {
                w.push(W::MetadataTooLarge { what: "EXIF", bytes: exif.len() });
            }
            if let Some(xmp) = m.xmp.as_deref().filter(|x| crate::codecs::jpeg::xmp_segment(x, None).is_none()) {
                w.push(W::MetadataTooLarge { what: "XMP", bytes: xmp.len() });
            }
        }
        if m.dpi.is_some() && !c.dpi {
            w.push(W::DpiDropped);
        }
        // DPX keeps its own header fields, not free-form text.
        let text_kept = |k: &str| c.text && (format != Format::Dpx || crate::codecs::dpx::KEYS.contains(&k));
        if m.text.iter().any(|(k, _)| !text_kept(k)) {
            w.push(W::TextDropped);
        }
    }
    w
}
