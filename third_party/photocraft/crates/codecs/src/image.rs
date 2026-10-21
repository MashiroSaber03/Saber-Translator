//! The crate's own flat raster image type.
//!
//! Samples are stored interleaved (`RGBARGBA…`), row-major, top-to-bottom,
//! in **native endianness**. Integer samples are unsigned and normalized to
//! `[0, max]`; float samples are nominally `[0, 1]` but may exceed it (HDR).

use std::fmt;

use crate::error::CodecError;
use crate::format::Format;
use half::f16;

/// Channel layout of an [`Image`]. Alpha, when present, is always the last
/// channel and is *straight* (not premultiplied).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelLayout {
    Gray,
    GrayA,
    Rgb,
    Rgba,
    /// Ink amounts, `0 = no ink`, `max = full ink` (not Adobe-inverted).
    Cmyk,
    CmykA,
}

impl ChannelLayout {
    pub const ALL: [ChannelLayout; 6] =
        [ChannelLayout::Gray, ChannelLayout::GrayA, ChannelLayout::Rgb, ChannelLayout::Rgba, ChannelLayout::Cmyk, ChannelLayout::CmykA];

    /// Total number of channels including alpha.
    pub const fn channels(self) -> usize {
        match self {
            ChannelLayout::Gray => 1,
            ChannelLayout::GrayA => 2,
            ChannelLayout::Rgb => 3,
            ChannelLayout::Rgba | ChannelLayout::Cmyk => 4,
            ChannelLayout::CmykA => 5,
        }
    }

    /// Number of colour (non-alpha) channels.
    pub const fn color_channels(self) -> usize {
        self.channels() - self.has_alpha() as usize
    }

    pub const fn has_alpha(self) -> bool {
        matches!(self, ChannelLayout::GrayA | ChannelLayout::Rgba | ChannelLayout::CmykA)
    }

    pub const fn is_gray(self) -> bool {
        matches!(self, ChannelLayout::Gray | ChannelLayout::GrayA)
    }

    pub const fn is_rgb(self) -> bool {
        matches!(self, ChannelLayout::Rgb | ChannelLayout::Rgba)
    }

    pub const fn is_cmyk(self) -> bool {
        matches!(self, ChannelLayout::Cmyk | ChannelLayout::CmykA)
    }

    /// The same colour model with alpha added.
    pub const fn with_alpha(self) -> Self {
        match self {
            ChannelLayout::Gray | ChannelLayout::GrayA => ChannelLayout::GrayA,
            ChannelLayout::Rgb | ChannelLayout::Rgba => ChannelLayout::Rgba,
            ChannelLayout::Cmyk | ChannelLayout::CmykA => ChannelLayout::CmykA,
        }
    }

    /// The same colour model without alpha.
    pub const fn without_alpha(self) -> Self {
        match self {
            ChannelLayout::Gray | ChannelLayout::GrayA => ChannelLayout::Gray,
            ChannelLayout::Rgb | ChannelLayout::Rgba => ChannelLayout::Rgb,
            ChannelLayout::Cmyk | ChannelLayout::CmykA => ChannelLayout::Cmyk,
        }
    }

    /// `true` if both layouts share a colour model (ignoring alpha).
    pub const fn same_model(self, other: Self) -> bool {
        (self.is_gray() && other.is_gray()) || (self.is_rgb() && other.is_rgb()) || (self.is_cmyk() && other.is_cmyk())
    }
}

/// Storage type of one sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SampleType {
    U8,
    U16,
    /// IEEE 754 half float, stored as its 16 bit pattern.
    F16,
    F32,
}

impl SampleType {
    pub const ALL: [SampleType; 4] = [SampleType::U8, SampleType::U16, SampleType::F16, SampleType::F32];

    pub const fn bytes(self) -> usize {
        match self {
            SampleType::U8 => 1,
            SampleType::U16 | SampleType::F16 => 2,
            SampleType::F32 => 4,
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(self, SampleType::F16 | SampleType::F32)
    }
}

/// Basic, format-independent metadata.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Metadata {
    /// Raw EXIF payload in TIFF structure (starting with `II*\0`/`MM\0*`),
    /// **without** the JPEG `Exif\0\0` prefix.
    pub exif: Option<Vec<u8>>,
    /// XMP packet as UTF-8.
    pub xmp: Option<String>,
    /// Horizontal and vertical resolution in dots per inch.
    pub dpi: Option<(f32, f32)>,
    /// Free-form key/value text (PNG tEXt/iTXt, TIFF ASCII tags).
    pub text: Vec<(String, String)>,
    /// Photoshop image resources (`8BIM` resource blocks): TIFF tag 34377. Kept opaque here;
    /// `photocraft-io` reads and writes them through `photocraft-psd`.
    pub photoshop_resources: Option<Vec<u8>>,
    /// Photoshop layer data (TIFF tag 37724 `ImageSourceData`): a layered TIFF's layers. Kept
    /// opaque here; `photocraft-io` reads and writes it through `photocraft-psd`.
    pub photoshop_layers: Option<Vec<u8>>,
}

impl Metadata {
    pub fn is_empty(&self) -> bool {
        self.exif.is_none()
            && self.xmp.is_none()
            && self.dpi.is_none()
            && self.text.is_empty()
            && self.photoshop_resources.is_none()
            && self.photoshop_layers.is_none()
    }
}

/// Something a decoder noticed about the file that the decoded pixels don't show: the file
/// held more than was decoded, or less than it should.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeWarning {
    /// Only the first frame of an animation was decoded; `total` counts every frame when known.
    MoreFrames { total: Option<u32> },
    /// Only the first page of a multi-page file was decoded; `total` counts every page when known.
    MorePages { total: Option<u32> },
    /// Only one part of a multi-part file was decoded; `total` counts every part when known.
    MoreParts { total: Option<u32> },
    /// The image opened on the full `display_window` stage; the data region sits at `origin`.
    DisplayWindowStage { stage: (u32, u32), origin: (i32, i32) },
    /// The image data ends early (truncated or damaged file); the decoder filled in the rest
    /// (a baseline JPEG's missing rows come out grey).
    Truncated { format: Format },
    /// Deep image data (per-pixel sample lists) was composited into the flat image; the samples
    /// themselves (Z and other deep channels) are not part of the flat result.
    DeepFlattened { format: Format, max_samples_per_pixel: u32 },
    /// Only the first image element of a DPX file was decoded; `total` counts them all.
    MoreElements { total: u32 },
    /// A DPX element's transfer characteristic and colorimetric specification (SMPTE 268M codes,
    /// 255 when undefined). The code values are imported as stored, without a colour transform.
    DpxEncoding { transfer: u8, colorimetric: u8 },
}

/// The SMPTE 268M name of a transfer characteristic or colorimetric code.
fn dpx_code_name(code: u8) -> &'static str {
    match code {
        0 => "user-defined",
        1 => "printing density",
        2 => "linear",
        3 => "logarithmic",
        4 => "unspecified video",
        5 => "SMPTE 274M",
        6 => "ITU-R BT.709",
        7 => "ITU-R BT.601 625-line",
        8 => "ITU-R BT.601 525-line",
        9 => "NTSC composite video",
        10 => "PAL composite video",
        11 => "Z linear",
        12 => "Z homogeneous",
        255 => "undefined",
        _ => "unknown",
    }
}

impl fmt::Display for DecodeWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeWarning::MoreFrames { total: Some(n) } => write!(f, "only the first of {n} frames was imported"),
            DecodeWarning::MoreFrames { total: None } => write!(f, "only the first frame of the animation was imported"),
            DecodeWarning::MorePages { total: Some(n) } => write!(f, "only the first of {n} pages was imported"),
            DecodeWarning::MorePages { total: None } => write!(f, "only the first page of the file was imported"),
            DecodeWarning::MoreParts { total: Some(n) } => write!(f, "only one of {n} EXR parts was imported"),
            DecodeWarning::MoreParts { total: None } => write!(f, "only one EXR part of the file was imported"),
            DecodeWarning::DisplayWindowStage { stage, origin } => {
                write!(f, "opened on the full display window stage ({}x{}); the data sits at ({}, {})", stage.0, stage.1, origin.0, origin.1)
            }
            DecodeWarning::Truncated { format } => {
                write!(f, "{} data ends early (the file is truncated or damaged); part of the image is missing", format.name())
            }
            DecodeWarning::DeepFlattened { format, max_samples_per_pixel } => write!(
                f,
                "{format} deep data was composited into a flat image (up to {max_samples_per_pixel} samples per pixel); per-pixel depth is not kept",
                format = format.name()
            ),
            DecodeWarning::MoreElements { total } => write!(f, "only the first of {total} image elements was imported"),
            DecodeWarning::DpxEncoding { transfer, colorimetric } => write!(
                f,
                "DPX transfer characteristic: {} ({transfer}), colorimetric: {} ({colorimetric}); the code values were imported as stored, without a colour transform",
                dpx_code_name(*transfer),
                dpx_code_name(*colorimetric)
            ),
        }
    }
}

/// One channel of a [`DeepImage`]: the samples of every pixel, one flat list in pixel order.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepChannel {
    /// The channel name as in the file (e.g. `R`, `G`, `B`, `A`, `Z`).
    pub name: String,
    /// The sample type as stored in the file; `samples` always holds the numeric value as
    /// f32 (half and 32-bit floats exactly; 32-bit unsigned integers exactly up to 2^24).
    pub sample: SampleType,
    /// Every sample of the channel, in pixel order; the samples of pixel `i` are
    /// `counts[i]..counts[i+1]` (see [`DeepImage::counts`]).
    pub samples: Vec<f32>,
}

/// Deep image data (OpenEXR `deepscanline`/`deeptile`): a variable-length list of samples
/// per pixel, each with its own colour, alpha and depth. All channels share the same sample
/// layout, so a sample is one index into every channel.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepImage {
    pub width: u32,
    pub height: u32,
    /// The full stage the image opens on: the EXR `display_window` size, when the file declares
    /// one the data region fits into and the limits allow. `None` when absent, absurd or not
    /// containing the data region — then the image opens on the data window alone, as before.
    pub stage: Option<(u32, u32)>,
    /// Where this image's (0, 0) sits on the stage: `data_window.position - display_window.position`.
    /// `(0, 0)` when [`Self::stage`] is `None`.
    pub origin: (i32, i32),
    /// The sample channels in file order.
    pub channels: Vec<DeepChannel>,
    /// Cumulative sample counts, `len == width as usize * height as usize + 1`: pixel `i`
    /// holds the samples `counts[i]..counts[i+1]` of every channel. Monotonically
    /// non-decreasing, starting and ending at the total sample count.
    pub counts: Vec<u64>,
}

impl DeepImage {
    /// A channel by its exact name.
    pub fn channel(&self, name: &str) -> Option<&DeepChannel> {
        self.channels.iter().find(|c| c.name == name)
    }

    /// The samples of one pixel, as indices into every channel's `samples`.
    pub fn sample_range(&self, pixel: usize) -> std::ops::Range<u64> {
        let end = self.counts.get(pixel + 1).copied().unwrap_or(0);
        self.counts.get(pixel).copied().unwrap_or(end)..end
    }

    /// The total number of samples over all pixels.
    pub fn total_samples(&self) -> u64 {
        self.counts.last().copied().unwrap_or(0)
    }
}

/// One channel of an [`ExrPartInfo`]: the name as in the file (a Maya/Arnold AOV layer
/// keeps its prefix, e.g. `diffuse.R`) and the sample type that part stores it with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExrChannelInfo {
    pub name: String,
    pub sample: SampleType,
}

/// One part of a multi-part OpenEXR file (what a Maya/Arnold render writes per AOV):
/// header facts only, no pixel data. [`crate::exr_info`] lists them, and the index is
/// what [`crate::decode_exr_part`] decodes; [`crate::decode_as_with`] opens the part
/// the ranking in [`ExrPartInfo::color_rank`] prefers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExrPartInfo {
    /// Index into the part list, in file order.
    pub index: usize,
    /// The part `name` attribute, when the file records one (Arnold names parts after AOVs).
    pub name: Option<String>,
    /// The part `view` attribute for stereo files (`left`/`right`).
    pub view: Option<String>,
    pub width: u32,
    pub height: u32,
    /// True for `deepscanline`/`deeptile` parts; those are composited by the deep
    /// decoder ([`crate::decode_deep_exr`]), not decoded part by part.
    pub deep: bool,
    /// True when the part is stored as tiles instead of scanlines.
    pub tiled: bool,
    /// The part's channels in file order.
    pub channels: Vec<ExrChannelInfo>,
}

impl ExrPartInfo {
    /// How much this part looks like a colour image the plain [`Image`] decode can show:
    /// 4 RGBA, 3 RGB, 2 YA, 1 any-single-channel, 0 anything else (e.g. `Z` beside `A`).
    /// The auto-pick takes the highest rank; ties go to the earlier part.
    pub fn color_rank(&self) -> u8 {
        let has = |n: &str| self.channels.iter().any(|c| base_channel_name(&c.name) == n);
        if has("R") && has("G") && has("B") {
            return if has("A") { 4 } else { 3 };
        }
        if has("Y") {
            return if has("A") { 2 } else { 1 };
        }
        u8::from(self.channels.len() == 1)
    }
}

/// Strip an optional `layer.` prefix from a channel name (`diffuse.R` → `R`).
pub(crate) fn base_channel_name(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name)
}

/// One Cryptomatte layer of an EXR file (Cryptomatte specification 1.2): per-pixel object or
/// material IDs with coverage, as written by Arnold, V-Ray, Redshift, Mantra/Karma or Cycles.
/// Header facts plus the parsed manifest; the samples come from
/// [`crate::decode_cryptomatte`].
#[derive(Debug, Clone, PartialEq)]
pub struct CryptomatteLayer {
    /// The part (see [`crate::exr_info`]) the channels and attributes live in.
    pub part: usize,
    /// The `cryptomatte/<key>/name` attribute, e.g. `CryptoAsset` or `crypto_asset`.
    pub name: String,
    /// The 7-hex-digit metadata key of the layer.
    pub key: String,
    /// The `conversion` attribute, when recorded (canonically `uint32_to_float32`).
    pub conversion: Option<String>,
    /// The manifest: object names with their ID values. A sidecar manifest file is only
    /// recorded in [`Self::sidecar_manifest`] and not read here (the codec is I/O free).
    pub manifest: Vec<(f32, String)>,
    /// The `manif_file` sidecar attribute, when the manifest is not embedded.
    pub sidecar_manifest: Option<String>,
    /// The stream channel bases in file order, e.g. `["crypto_asset00", "crypto_asset01"]`;
    /// each holds an ID (`.red`) and a coverage (`.green`) channel.
    pub channels: Vec<String>,
}

impl CryptomatteLayer {
    /// The manifest name for an ID value, when the manifest lists it.
    pub fn name_of(&self, id: f32) -> Option<&str> {
        self.manifest.iter().find(|(i, _)| *i == id).map(|(_, n)| n.as_str())
    }
}

/// Decoded Cryptomatte samples: per pixel the (ID, coverage) pairs with coverage above zero,
/// sorted by descending coverage. The IDs are the float values as stored (compared by value,
/// never by bit pattern, like the reference readers).
#[derive(Debug, Clone, PartialEq)]
pub struct CryptomatteBuffer {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<Vec<(f32, f32)>>,
}

impl CryptomatteBuffer {
    /// The pairs of one pixel; empty for a pixel covered by nothing.
    pub fn at(&self, x: u32, y: u32) -> &[(f32, f32)] {
        let i = y as usize * self.width as usize + x as usize;
        self.pixels.get(i).map_or(&[], |v| v)
    }
}

/// A single flat raster image.
#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    width: u32,
    height: u32,
    layout: ChannelLayout,
    sample: SampleType,
    data: Vec<u8>,
    /// Embedded ICC profile (must describe `layout`'s colour model).
    pub icc: Option<Vec<u8>>,
    pub meta: Metadata,
    /// What the decoder noticed about the file: empty for a complete, single image and for
    /// images built in memory. Encoders ignore it.
    pub warnings: Vec<DecodeWarning>,
}

fn byte_len(width: u32, height: u32, layout: ChannelLayout, sample: SampleType) -> Option<usize> {
    (width as usize).checked_mul(height as usize)?.checked_mul(layout.channels())?.checked_mul(sample.bytes())
}

impl Image {
    /// A zero-filled image, or an error if the byte size overflows `usize`.
    pub fn new(width: u32, height: u32, layout: ChannelLayout, sample: SampleType) -> Result<Self, CodecError> {
        let len = byte_len(width, height, layout, sample).ok_or_else(|| CodecError::InvalidImage("image size overflows usize".into()))?;
        Ok(Image { width, height, layout, sample, data: vec![0; len], icc: None, meta: Metadata::default(), warnings: Vec::new() })
    }

    /// Wrap raw interleaved, native-endian bytes.
    pub fn from_raw(width: u32, height: u32, layout: ChannelLayout, sample: SampleType, data: Vec<u8>) -> Result<Self, CodecError> {
        let expected = byte_len(width, height, layout, sample).ok_or_else(|| CodecError::InvalidImage("image size overflows usize".into()))?;
        if data.len() != expected {
            return Err(CodecError::InvalidImage(format!("buffer has {} bytes, expected {expected} for {width}x{height} {layout:?} {sample:?}", data.len())));
        }
        Ok(Image { width, height, layout, sample, data, icc: None, meta: Metadata::default(), warnings: Vec::new() })
    }

    pub fn from_u8(width: u32, height: u32, layout: ChannelLayout, data: Vec<u8>) -> Result<Self, CodecError> {
        Self::from_raw(width, height, layout, SampleType::U8, data)
    }

    pub fn from_u16(width: u32, height: u32, layout: ChannelLayout, data: &[u16]) -> Result<Self, CodecError> {
        let bytes = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        Self::from_raw(width, height, layout, SampleType::U16, bytes)
    }

    pub fn from_f16(width: u32, height: u32, layout: ChannelLayout, data: &[f16]) -> Result<Self, CodecError> {
        let bytes = data.iter().flat_map(|v| v.to_bits().to_ne_bytes()).collect();
        Self::from_raw(width, height, layout, SampleType::F16, bytes)
    }

    pub fn from_f32(width: u32, height: u32, layout: ChannelLayout, data: &[f32]) -> Result<Self, CodecError> {
        let bytes = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        Self::from_raw(width, height, layout, SampleType::F32, bytes)
    }

    /// Build an image from normalized f32 samples (`[0,1]` for integer
    /// targets; values are clamped and rounded when quantizing).
    pub fn from_normalized(width: u32, height: u32, layout: ChannelLayout, sample: SampleType, values: &[f32]) -> Result<Self, CodecError> {
        let data = quantize(values, sample);
        Self::from_raw(width, height, layout, sample, data)
    }

    pub fn with_icc(mut self, icc: Option<Vec<u8>>) -> Self {
        self.icc = icc;
        self
    }

    pub fn with_meta(mut self, meta: Metadata) -> Self {
        self.meta = meta;
        self
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    pub fn layout(&self) -> ChannelLayout {
        self.layout
    }
    pub fn sample_type(&self) -> SampleType {
        self.sample
    }
    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }
    /// Number of samples (pixels × channels).
    pub fn sample_count(&self) -> usize {
        self.pixel_count() * self.layout.channels()
    }
    /// Raw interleaved native-endian bytes.
    pub fn data(&self) -> &[u8] {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }

    /// Borrow samples as `u8` (only when the sample type is `U8`).
    pub fn as_u8(&self) -> Option<&[u8]> {
        (self.sample == SampleType::U8).then_some(&self.data[..])
    }

    /// Copy samples out as `u16` (only when the sample type is `U16`).
    pub fn to_u16_samples(&self) -> Option<Vec<u16>> {
        (self.sample == SampleType::U16).then(|| self.data.as_chunks::<2>().0.iter().map(|c| u16::from_ne_bytes([c[0], c[1]])).collect())
    }

    /// Copy samples out as `f16` (only when the sample type is `F16`).
    pub fn to_f16_samples(&self) -> Option<Vec<f16>> {
        (self.sample == SampleType::F16).then(|| self.data.as_chunks::<2>().0.iter().map(|c| f16::from_bits(u16::from_ne_bytes([c[0], c[1]]))).collect())
    }

    /// Copy samples out as `f32` (only when the sample type is `F32`).
    pub fn to_f32_samples(&self) -> Option<Vec<f32>> {
        (self.sample == SampleType::F32).then(|| self.data.as_chunks::<4>().0.iter().map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    /// Sample `index` (in samples, not bytes) as a normalized f32.
    pub fn sample_normalized(&self, index: usize) -> f32 {
        read_normalized(&self.data, self.sample, index)
    }

    /// Normalized f32 of channel `c` at pixel `(x, y)`.
    pub fn get(&self, x: u32, y: u32, c: usize) -> f32 {
        let idx = (y as usize * self.width as usize + x as usize) * self.layout.channels() + c;
        self.sample_normalized(idx)
    }

    /// All samples as normalized f32 (integers mapped to `[0,1]`).
    pub fn to_normalized(&self) -> Vec<f32> {
        to_normalized(&self.data, self.sample)
    }

    /// `true` when the image has alpha and at least one pixel is not fully
    /// opaque.
    pub fn has_translucency(&self) -> bool {
        if !self.layout.has_alpha() {
            return false;
        }
        let ch = self.layout.channels();
        (0..self.pixel_count()).any(|p| self.sample_normalized(p * ch + ch - 1) < 1.0)
    }

    /// `true` when this is a float image with any sample outside `[0, 1]`
    /// (or NaN).
    pub fn has_out_of_range(&self) -> bool {
        if !self.sample.is_float() {
            return false;
        }
        (0..self.sample_count()).any(|i| {
            let v = self.sample_normalized(i);
            !(0.0..=1.0).contains(&v)
        })
    }

    /// Convert to another layout and/or sample type. The ICC profile is kept
    /// only if the colour model is unchanged (an RGB profile on CMYK data
    /// would be wrong). Metadata is kept.
    pub fn convert(&self, layout: ChannelLayout, sample: SampleType) -> Image {
        if layout == self.layout && sample == self.sample {
            return self.clone();
        }
        let data = if sample == self.sample && !self.layout.has_alpha() && layout == self.layout.with_alpha() {
            // Fast path (opening any opaque file): copy the samples, append an opaque alpha.
            let bps = sample.bytes();
            let src = self.layout.channels() * bps;
            let one: Vec<u8> = match sample {
                SampleType::U8 => vec![u8::MAX],
                SampleType::U16 => u16::MAX.to_ne_bytes().to_vec(),
                SampleType::F16 => f16::ONE.to_bits().to_ne_bytes().to_vec(),
                SampleType::F32 => 1.0f32.to_ne_bytes().to_vec(),
            };
            let mut out = Vec::with_capacity(self.data.len() / src * (src + bps));
            for px in self.data.chunks_exact(src) {
                out.extend_from_slice(px);
                out.extend_from_slice(&one);
            }
            out
        } else if let Some(fast) = self.convert_int_direct(layout, sample) {
            fast
        } else if layout == self.layout {
            quantize(&self.to_normalized(), sample)
        } else {
            let src = self.to_normalized();
            let values = convert_layout(&src, self.layout, layout);
            quantize(&values, sample)
        };
        Image {
            width: self.width,
            height: self.height,
            layout,
            sample,
            data,
            icc: if layout.same_model(self.layout) { self.icc.clone() } else { None },
            meta: self.meta.clone(),
            warnings: self.warnings.clone(),
        }
    }

    /// Integer fast path of [`Image::convert`] for U8/U16 sources and targets whose result is
    /// a pure channel copy: same colour model (alpha added as opaque, or dropped) or gray to
    /// RGB. Bit-identical to the generic f32 route (see the equivalence tests) but without
    /// the two intermediate f32 buffers. Returns `None` for everything else.
    fn convert_int_direct(&self, layout: ChannelLayout, sample: SampleType) -> Option<Vec<u8>> {
        let int = |s: SampleType| matches!(s, SampleType::U8 | SampleType::U16);
        if !int(self.sample) || !int(sample) {
            return None;
        }
        let gray_to_rgb = self.layout.is_gray() && layout.is_rgb();
        if !gray_to_rgb && !layout.same_model(self.layout) {
            return None;
        }
        let one: &[u8] = match sample {
            SampleType::U8 => &[u8::MAX],
            _ => &u16::MAX.to_ne_bytes(),
        };
        let (fc, tc) = (self.layout.channels(), layout.channels());
        let (sb, db) = (self.sample.bytes(), sample.bytes());
        let (src_color, dst_color) = (self.layout.color_channels(), layout.color_channels());
        let n = self.data.len() / (fc * sb);
        let mut out = Vec::with_capacity(n.checked_mul(tc)?.checked_mul(db)?);
        macro_rules! run {
            ($conv:expr) => {{
                let conv = $conv;
                for px in self.data.chunks_exact(fc * sb) {
                    for j in 0..dst_color {
                        let c = if src_color == 1 { 0 } else { j };
                        conv(&px[c * sb..(c + 1) * sb], &mut out);
                    }
                    if layout.has_alpha() {
                        if self.layout.has_alpha() {
                            conv(&px[src_color * sb..(src_color + 1) * sb], &mut out);
                        } else {
                            out.extend_from_slice(one);
                        }
                    }
                }
            }};
        }
        match (self.sample, sample) {
            (SampleType::U8, SampleType::U8) => run!(|s: &[u8], o: &mut Vec<u8>| o.push(s[0])),
            (SampleType::U16, SampleType::U16) => run!(|s: &[u8], o: &mut Vec<u8>| o.extend_from_slice(s)),
            (SampleType::U8, SampleType::U16) => run!(|s: &[u8], o: &mut Vec<u8>| o.extend_from_slice(&(u16::from(s[0]) * 257).to_ne_bytes())),
            _ => run!(|s: &[u8], o: &mut Vec<u8>| o.push(((u32::from(u16::from_ne_bytes([s[0], s[1]])) + 128) / 257) as u8)),
        }
        Some(out)
    }

    /// [`Image::convert`], borrowing `self` when it already has that layout and sample type
    /// (encoders use it so a full-size image isn't copied for nothing).
    pub fn converted(&self, layout: ChannelLayout, sample: SampleType) -> std::borrow::Cow<'_, Image> {
        if layout == self.layout && sample == self.sample { std::borrow::Cow::Borrowed(self) } else { std::borrow::Cow::Owned(self.convert(layout, sample)) }
    }

    /// Interleaved RGBA, 8 bits per sample.
    pub fn to_rgba8(&self) -> Vec<u8> {
        self.convert(ChannelLayout::Rgba, SampleType::U8).data
    }

    /// Interleaved RGBA, 16 bits per sample.
    pub fn to_rgba16(&self) -> Vec<u16> {
        self.convert(ChannelLayout::Rgba, SampleType::U16).to_u16_samples().unwrap_or_default()
    }

    /// Interleaved RGBA as f32 (integer inputs normalized to `[0,1]`).
    pub fn to_rgba_f32(&self) -> Vec<f32> {
        convert_layout(&self.to_normalized(), self.layout, ChannelLayout::Rgba)
    }
}

// ---------------------------------------------------------------------------
// Sample conversion helpers
// ---------------------------------------------------------------------------

#[inline]
pub(crate) fn read_normalized(data: &[u8], sample: SampleType, i: usize) -> f32 {
    match sample {
        SampleType::U8 => data[i] as f32 / 255.0,
        SampleType::U16 => u16::from_ne_bytes([data[2 * i], data[2 * i + 1]]) as f32 / 65535.0,
        SampleType::F16 => f16::from_bits(u16::from_ne_bytes([data[2 * i], data[2 * i + 1]])).to_f32(),
        SampleType::F32 => f32::from_ne_bytes([data[4 * i], data[4 * i + 1], data[4 * i + 2], data[4 * i + 3]]),
    }
}

pub(crate) fn to_normalized(data: &[u8], sample: SampleType) -> Vec<f32> {
    match sample {
        SampleType::U8 => data.iter().map(|&v| v as f32 / 255.0).collect(),
        SampleType::U16 => data.as_chunks::<2>().0.iter().map(|c| u16::from_ne_bytes([c[0], c[1]]) as f32 / 65535.0).collect(),
        SampleType::F16 => data.as_chunks::<2>().0.iter().map(|c| f16::from_bits(u16::from_ne_bytes([c[0], c[1]])).to_f32()).collect(),
        SampleType::F32 => data.as_chunks::<4>().0.iter().map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect(),
    }
}

#[inline]
fn clamp01(v: f32) -> f32 {
    if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) }
}

pub(crate) fn quantize(values: &[f32], sample: SampleType) -> Vec<u8> {
    match sample {
        SampleType::U8 => values.iter().map(|&v| (clamp01(v) * 255.0).round() as u8).collect(),
        SampleType::U16 => values.iter().flat_map(|&v| ((clamp01(v) * 65535.0).round() as u16).to_ne_bytes()).collect(),
        SampleType::F16 => values.iter().flat_map(|&v| f16::from_f32(v).to_bits().to_ne_bytes()).collect(),
        SampleType::F32 => values.iter().flat_map(|&v| v.to_ne_bytes()).collect(),
    }
}

/// Rec. 709 luma weights applied to encoded values.
const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

fn cmyk_to_rgb(c: f32, m: f32, y: f32, k: f32) -> [f32; 3] {
    [(1.0 - c) * (1.0 - k), (1.0 - m) * (1.0 - k), (1.0 - y) * (1.0 - k)]
}

fn rgb_to_cmyk(r: f32, g: f32, b: f32) -> [f32; 4] {
    let k = 1.0 - r.max(g).max(b);
    if k >= 1.0 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let d = 1.0 - k;
    [(1.0 - r - k) / d, (1.0 - g - k) / d, (1.0 - b - k) / d, k]
}

/// Convert normalized interleaved samples between layouts. Naive
/// (non-colour-managed) formulas; colour-managed conversion belongs in
/// `photocraft-color`.
pub(crate) fn convert_layout(src: &[f32], from: ChannelLayout, to: ChannelLayout) -> Vec<f32> {
    if from == to {
        return src.to_vec();
    }
    let fc = from.channels();
    let tc = to.channels();
    let n = src.len() / fc;
    let mut out = Vec::with_capacity(n * tc);
    for px in src.chunks_exact(fc) {
        let alpha = if from.has_alpha() { px[fc - 1] } else { 1.0 };
        let color = &px[..from.color_channels()];
        if from.same_model(to) {
            out.extend_from_slice(color);
        } else {
            let rgb = match from {
                ChannelLayout::Gray | ChannelLayout::GrayA => [color[0]; 3],
                ChannelLayout::Rgb | ChannelLayout::Rgba => [color[0], color[1], color[2]],
                ChannelLayout::Cmyk | ChannelLayout::CmykA => cmyk_to_rgb(color[0], color[1], color[2], color[3]),
            };
            match to {
                ChannelLayout::Gray | ChannelLayout::GrayA => {
                    out.push(rgb[0] * LUMA[0] + rgb[1] * LUMA[1] + rgb[2] * LUMA[2]);
                }
                ChannelLayout::Rgb | ChannelLayout::Rgba => out.extend_from_slice(&rgb),
                ChannelLayout::Cmyk | ChannelLayout::CmykA => out.extend_from_slice(&rgb_to_cmyk(rgb[0], rgb[1], rgb[2])),
            }
        }
        if to.has_alpha() {
            out.push(alpha);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn adding_alpha_fast_path_matches_the_general_conversion() {
        for sample in SampleType::ALL {
            for layout in [ChannelLayout::Gray, ChannelLayout::Rgb, ChannelLayout::Cmyk] {
                let n = 7 * 3 * layout.channels();
                let vals: Vec<f32> = (0..n).map(|i| (i as f32 * 0.137) % 1.0).collect();
                let img = super::Image::from_normalized(7, 3, layout, sample, &vals).unwrap();
                let fast = img.convert(layout.with_alpha(), sample);
                let slow = super::quantize(&super::convert_layout(&img.to_normalized(), layout, layout.with_alpha()), sample);
                assert_eq!(fast.data(), &slow[..], "{layout:?} {sample:?}");
            }
        }
    }

    use super::*;

    /// The generic f32 route `convert` used before the integer fast paths (the reference).
    fn convert_via_f32(img: &Image, layout: ChannelLayout, sample: SampleType) -> Vec<u8> {
        quantize(&convert_layout(&img.to_normalized(), img.layout, layout), sample)
    }

    #[test]
    fn int_fast_path_matches_f32_route_for_every_value() {
        // Every U8 and every U16 value through every depth pair (alpha goes through the same per-sample conversion; covered by the layout test).
        let all8: Vec<u8> = (0..=255).collect();
        let all16: Vec<u16> = (0..=65535).collect();
        for (src, to) in
            [(SampleType::U8, SampleType::U8), (SampleType::U8, SampleType::U16), (SampleType::U16, SampleType::U8), (SampleType::U16, SampleType::U16)]
        {
            let img = if src == SampleType::U8 {
                Image::from_u8(256, 1, ChannelLayout::Gray, all8.clone()).unwrap()
            } else {
                Image::from_u16(65536, 1, ChannelLayout::Gray, &all16).unwrap()
            };
            for layout in [ChannelLayout::Gray, ChannelLayout::Rgb] {
                assert_eq!(img.convert(layout, to).data(), &convert_via_f32(&img, layout, to)[..], "{src:?}->{to:?} {layout:?}");
            }
        }
    }

    #[test]
    fn int_fast_path_matches_f32_route_for_every_layout() {
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for from_s in [SampleType::U8, SampleType::U16] {
            for from_l in ChannelLayout::ALL {
                let n = 64 * from_l.channels();
                let mut vals = Vec::new();
                for i in 0..n {
                    // Edge values (0, 1, max-1, max, mid) mixed with random ones.
                    let r = next();
                    let v = match i % 7 {
                        0 => 0.0,
                        1 => 1.0,
                        2 => 1.0 / 255.0,
                        3 => 254.0 / 255.0,
                        _ => (r % 65536) as f32 / 65535.0,
                    };
                    vals.push(v);
                }
                let img = Image::from_normalized(64, 1, from_l, from_s, &vals).unwrap();
                for to_l in ChannelLayout::ALL {
                    for to_s in SampleType::ALL {
                        let got = img.convert(to_l, to_s);
                        assert_eq!(got.data(), &convert_via_f32(&img, to_l, to_s)[..], "{from_l:?}/{from_s:?} -> {to_l:?}/{to_s:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn u8_u16_roundtrip_exact() {
        let v: Vec<u8> = (0..=255).collect();
        let img = Image::from_u8(256, 1, ChannelLayout::Gray, v.clone()).unwrap();
        let back = img.convert(ChannelLayout::Gray, SampleType::U16).convert(ChannelLayout::Gray, SampleType::U8);
        assert_eq!(back.data(), &v[..]);
    }

    #[test]
    fn u16_f32_roundtrip_exact() {
        let v: Vec<u16> = (0..=65535u32).step_by(7).map(|x| x as u16).collect();
        let img = Image::from_u16(v.len() as u32, 1, ChannelLayout::Gray, &v).unwrap();
        let back = img.convert(ChannelLayout::Gray, SampleType::F32).convert(ChannelLayout::Gray, SampleType::U16);
        assert_eq!(back.to_u16_samples().unwrap(), v);
    }

    #[test]
    fn icc_dropped_on_model_change() {
        let img = Image::new(2, 2, ChannelLayout::Cmyk, SampleType::U8).unwrap().with_icc(Some(vec![1, 2, 3]));
        assert!(img.convert(ChannelLayout::Rgb, SampleType::U8).icc.is_none());
        assert!(img.convert(ChannelLayout::CmykA, SampleType::U16).icc.is_some());
    }

    #[test]
    fn from_raw_rejects_bad_len() {
        assert!(Image::from_raw(2, 2, ChannelLayout::Rgb, SampleType::U8, vec![0; 11]).is_err());
    }

    #[test]
    fn rgba8_of_gray() {
        let img = Image::from_u8(1, 1, ChannelLayout::GrayA, vec![10, 20]).unwrap();
        assert_eq!(img.to_rgba8(), vec![10, 10, 10, 20]);
    }

    #[test]
    fn cmyk_rgb_basic() {
        let img = Image::from_u8(1, 1, ChannelLayout::Cmyk, vec![0, 0, 0, 255]).unwrap();
        assert_eq!(img.to_rgba8(), vec![0, 0, 0, 255]);
        let img = Image::from_u8(1, 1, ChannelLayout::Rgb, vec![255, 0, 0]).unwrap();
        let c = img.convert(ChannelLayout::Cmyk, SampleType::U8);
        assert_eq!(c.data(), &[0, 255, 255, 0]);
    }
}
