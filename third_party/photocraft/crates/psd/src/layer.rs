//! Layer records, layer masks, blending ranges and per-layer channel data.

use crate::blend::BlendMode;
use crate::compression::{Compression, PlaneLayout, decode_planes, encode_planes};
use crate::error::{PsdError, Result};
use crate::header::{MAX_DIMENSION, Version};
use crate::io::{Reader, WriteExt, decode_legacy_name, read_pascal, write_pascal};
use crate::tagged::{BlockData, SectionDivider, SectionType, TaggedBlock, read_blocks, write_blocks};

/// Channel id of the transparency mask.
pub const CHANNEL_TRANSPARENCY: i16 = -1;
/// Channel id of the user-supplied layer mask.
pub const CHANNEL_USER_MASK: i16 = -2;
/// Channel id of the real user-supplied layer mask (when both a user mask and
/// a vector mask are present).
pub const CHANNEL_REAL_USER_MASK: i16 = -3;

/// Maximum number of channels per layer accepted by the parser.
pub const MAX_LAYER_CHANNELS: u16 = 64;

/// A rectangle as stored in PSD files (top, left, bottom, right).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Rect {
    /// Top edge (inclusive).
    pub top: i32,
    /// Left edge (inclusive).
    pub left: i32,
    /// Bottom edge (exclusive).
    pub bottom: i32,
    /// Right edge (exclusive).
    pub right: i32,
}

impl Rect {
    /// Rectangle from origin and size.
    pub fn from_xywh(left: i32, top: i32, width: u32, height: u32) -> Self {
        Rect { top, left, bottom: top.saturating_add(height.min(i32::MAX as u32) as i32), right: left.saturating_add(width.min(i32::MAX as u32) as i32) }
    }
    /// Width (may be negative for malformed data).
    pub fn width(&self) -> i64 {
        i64::from(self.right) - i64::from(self.left)
    }
    /// Height (may be negative for malformed data).
    pub fn height(&self) -> i64 {
        i64::from(self.bottom) - i64::from(self.top)
    }
    /// `true` when width or height is zero or negative.
    pub fn is_empty(&self) -> bool {
        self.width() <= 0 || self.height() <= 0
    }
    /// Validated `(width, height)` as `usize`.
    pub fn size(&self) -> Result<(usize, usize)> {
        let (w, h) = (self.width(), self.height());
        if w < 0 || h < 0 {
            return Err(PsdError::invalid("rectangle has negative size"));
        }
        if w > i64::from(MAX_DIMENSION) || h > i64::from(MAX_DIMENSION) {
            return Err(PsdError::LimitExceeded("layer dimensions exceed 300000"));
        }
        Ok((w as usize, h as usize))
    }
    fn read(r: &mut Reader<'_>) -> Result<Self> {
        Ok(Rect { top: r.i32()?, left: r.i32()?, bottom: r.i32()?, right: r.i32()? })
    }
    fn write(&self, out: &mut Vec<u8>) {
        out.put_i32(self.top);
        out.put_i32(self.left);
        out.put_i32(self.bottom);
        out.put_i32(self.right);
    }
}

/// Layer record flag byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LayerFlags(pub u8);

impl LayerFlags {
    /// Bit 0.
    pub const TRANSPARENCY_PROTECTED: u8 = 1 << 0;
    /// Bit 1 (set = hidden).
    pub const HIDDEN: u8 = 1 << 1;
    /// Bit 2 (obsolete).
    pub const OBSOLETE: u8 = 1 << 2;
    /// Bit 3: bit 4 carries useful information.
    pub const BIT4_USEFUL: u8 = 1 << 3;
    /// Bit 4: pixel data irrelevant to the appearance of the document.
    pub const PIXEL_DATA_IRRELEVANT: u8 = 1 << 4;

    fn bit(self, b: u8) -> bool {
        self.0 & b != 0
    }
    fn set(&mut self, b: u8, on: bool) {
        if on {
            self.0 |= b;
        } else {
            self.0 &= !b;
        }
    }
    /// Transparency protected.
    pub fn transparency_protected(self) -> bool {
        self.bit(Self::TRANSPARENCY_PROTECTED)
    }
    /// Hidden.
    pub fn hidden(self) -> bool {
        self.bit(Self::HIDDEN)
    }
    /// Bit 4 is meaningful.
    pub fn bit4_useful(self) -> bool {
        self.bit(Self::BIT4_USEFUL)
    }
    /// Pixel data irrelevant (only meaningful when `bit4_useful`).
    pub fn pixel_data_irrelevant(self) -> bool {
        self.bit(Self::PIXEL_DATA_IRRELEVANT)
    }
    /// Sets hidden.
    pub fn set_hidden(&mut self, v: bool) {
        self.set(Self::HIDDEN, v);
    }
    /// Sets transparency protection.
    pub fn set_transparency_protected(&mut self, v: bool) {
        self.set(Self::TRANSPARENCY_PROTECTED, v);
    }
    /// Sets pixel-data-irrelevant (also sets bit 3).
    pub fn set_pixel_data_irrelevant(&mut self, v: bool) {
        self.set(Self::BIT4_USEFUL, true);
        self.set(Self::PIXEL_DATA_IRRELEVANT, v);
    }
}

/// Optional mask parameters (present when mask flag bit 4 is set).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MaskParameters {
    /// Parameter flags byte (bit 0 user density, 1 user feather, 2 vector
    /// density, 3 vector feather).
    pub flags: u8,
    /// User mask density.
    pub user_density: Option<u8>,
    /// User mask feather.
    pub user_feather: Option<f64>,
    /// Vector mask density.
    pub vector_density: Option<u8>,
    /// Vector mask feather.
    pub vector_feather: Option<f64>,
}

/// "Real" user mask fields (present in 36+-byte mask records).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RealMask {
    /// Real flags.
    pub flags: u8,
    /// Real user mask background (0 or 255).
    pub background: u8,
    /// Real mask rectangle.
    pub rect: Rect,
}

/// Layer mask / adjustment layer data.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LayerMask {
    /// Mask rectangle.
    pub rect: Rect,
    /// Default color outside the rectangle (0 or 255).
    pub default_color: u8,
    /// Flags: bit 0 position relative to layer, 1 disabled, 2 invert
    /// (obsolete), 3 user mask came from rendering other data, 4 parameters
    /// applied.
    pub flags: u8,
    /// Mask parameters (when flag bit 4 is set).
    pub parameters: Option<MaskParameters>,
    /// Real user mask fields.
    pub real: Option<RealMask>,
    /// Remaining bytes (the 2 padding bytes of the 20-byte form, etc.).
    pub trailing: Vec<u8>,
    /// When both are present, the real-mask fields come before the mask parameters. Photoshop
    /// writes this order (the published spec lists the parameters first); `false` keeps a file
    /// that uses the spec's order byte-stable.
    #[cfg_attr(feature = "serde", serde(default = "default_true"))]
    pub real_first: bool,
}

#[cfg(feature = "serde")]
fn default_true() -> bool {
    true
}

/// Byte length of the mask parameter values announced by the parameter flags byte `pf`.
pub(crate) fn mask_parameter_bytes(pf: u8) -> usize {
    usize::from(pf & 1 != 0) + 8 * usize::from(pf & 2 != 0) + usize::from(pf & 4 != 0) + 8 * usize::from(pf & 8 != 0)
}

impl LayerMask {
    /// Mask flag: position relative to layer.
    pub const FLAG_RELATIVE: u8 = 1 << 0;
    /// Mask flag: disabled.
    pub const FLAG_DISABLED: u8 = 1 << 1;
    /// Mask flag: invert when blending (obsolete).
    pub const FLAG_INVERT: u8 = 1 << 2;
    /// Mask flag: parameters present.
    pub const FLAG_PARAMETERS: u8 = 1 << 4;

    /// The simple 20-byte form.
    pub fn new(rect: Rect, default_color: u8, flags: u8) -> Self {
        LayerMask { rect, default_color, flags: flags & !Self::FLAG_PARAMETERS, parameters: None, real: None, trailing: vec![0, 0], real_first: true }
    }
    /// `true` if the disabled flag is set.
    pub fn disabled(&self) -> bool {
        self.flags & Self::FLAG_DISABLED != 0
    }

    /// Whether `d` (a whole mask record with mask parameters) stores the real-mask fields before
    /// the parameters, as Photoshop does. The spec's order wins when it reads as parameters
    /// followed by padding alone (no real mask), the one layout the two orders can share. When
    /// both orders fit the record's length, Photoshop's is taken unless its reading is
    /// implausible (a real-mask background that is neither 0 nor 255, or parameter flag bits
    /// the spec does not define) while the spec's reading with a real mask is sound.
    pub(crate) fn real_before_parameters(d: &[u8]) -> bool {
        // Parameters at `pf_at`, then 0..=3 bytes of padding, and nothing else.
        let fits =
            |pf_at: usize| d.get(pf_at).is_some_and(|&pf| (pf_at + 1 + mask_parameter_bytes(pf)..=pf_at + 4 + mask_parameter_bytes(pf)).contains(&d.len()));
        if fits(18) {
            return false;
        }
        let photoshop = fits(36);
        if !photoshop {
            return false;
        }
        let known_flags = |pf: u8| pf & 0xF0 == 0;
        let real_background_ok = |at: usize| d.get(at + 1).is_some_and(|&bg| bg == 0 || bg == 255);
        // Photoshop's reading: real-mask fields at 18, parameters at 36.
        let photoshop_plausible = real_background_ok(18) && d.get(36).is_some_and(|&pf| known_flags(pf));
        // The spec's reading: parameters at 18, then the 18 real-mask bytes, then padding.
        let spec_plausible = d.get(18).is_some_and(|&pf| {
            let real_at = 19 + mask_parameter_bytes(pf);
            known_flags(pf) && (real_at + 18..=real_at + 21).contains(&d.len()) && real_background_ok(real_at)
        });
        photoshop_plausible || !spec_plausible
    }

    fn parse(d: &[u8]) -> Result<Self> {
        let mut r = Reader::new(d);
        let rect = Rect::read(&mut r)?;
        let default_color = r.u8()?;
        let flags = r.u8()?;
        let read_real = |r: &mut Reader<'_>| -> Result<Option<RealMask>> {
            Ok(if r.remaining() >= 18 { Some(RealMask { flags: r.u8()?, background: r.u8()?, rect: Rect::read(r)? }) } else { None })
        };
        let has_parameters = flags & Self::FLAG_PARAMETERS != 0;
        let real_first = has_parameters && Self::real_before_parameters(d);
        let mut real = if real_first { read_real(&mut r)? } else { None };
        let parameters = if has_parameters {
            let pf = r.u8()?;
            Some(MaskParameters {
                flags: pf,
                user_density: if pf & 1 != 0 { Some(r.u8()?) } else { None },
                user_feather: if pf & 2 != 0 { Some(r.f64()?) } else { None },
                vector_density: if pf & 4 != 0 { Some(r.u8()?) } else { None },
                vector_feather: if pf & 8 != 0 { Some(r.f64()?) } else { None },
            })
        } else {
            None
        };
        if !real_first {
            real = read_real(&mut r)?;
        }
        // The order only matters when both are present; otherwise default to Photoshop's.
        let real_first = real_first || real.is_none() || parameters.is_none();
        Ok(LayerMask { rect, default_color, flags, parameters, real, trailing: r.peek_rest().to_vec(), real_first })
    }

    fn write(&self, out: &mut Vec<u8>) {
        self.rect.write(out);
        out.put_u8(self.default_color);
        out.put_u8(self.flags);
        let write_real = |out: &mut Vec<u8>| {
            if let Some(real) = &self.real {
                out.put_u8(real.flags);
                out.put_u8(real.background);
                real.rect.write(out);
            }
        };
        if self.real_first {
            write_real(out);
        }
        if let Some(p) = &self.parameters {
            out.put_u8(p.flags);
            if let Some(v) = p.user_density {
                out.put_u8(v);
            }
            if let Some(v) = p.user_feather {
                out.put_f64(v);
            }
            if let Some(v) = p.vector_density {
                out.put_u8(v);
            }
            if let Some(v) = p.vector_feather {
                out.put_f64(v);
            }
        }
        if !self.real_first {
            write_real(out);
        }
        out.put(&self.trailing);
    }
}

/// The layer mask data field of a layer record.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MaskData {
    /// Length 0.
    #[default]
    None,
    /// Parsed mask.
    Mask(LayerMask),
    /// Unparseable mask data, preserved.
    Raw(Vec<u8>),
}

/// Layer blending ranges ("Blend If"), stored raw. The data is a sequence of
/// 8-byte entries: composite gray first, then one per channel; each entry is
/// source (black lo, black hi, white lo, white hi) then destination.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BlendingRanges {
    /// Raw bytes.
    pub data: Vec<u8>,
}

/// One blending range entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendRange {
    /// Source range: black low, black high, white low, white high.
    pub source: [u8; 4],
    /// Destination range.
    pub dest: [u8; 4],
}

impl BlendingRanges {
    /// Default "full" ranges for composite + `channels` channels.
    pub fn full(channels: usize) -> Self {
        let mut data = Vec::with_capacity((channels + 1) * 8);
        for _ in 0..=channels {
            data.extend_from_slice(&[0, 0, 255, 255, 0, 0, 255, 255]);
        }
        BlendingRanges { data }
    }
    /// Parsed entries (incomplete trailing bytes are ignored).
    pub fn ranges(&self) -> Vec<BlendRange> {
        self.data.as_chunks::<8>().0.iter().map(|c| BlendRange { source: [c[0], c[1], c[2], c[3]], dest: [c[4], c[5], c[6], c[7]] }).collect()
    }
}

/// Encoded image data for one layer channel.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChannelData {
    /// Channel id: 0.. color channels, -1 transparency, -2 user mask,
    /// -3 real user mask.
    pub id: i16,
    /// Compression; `None` when the stored length was 0 (no compression
    /// field at all).
    pub compression: Option<Compression>,
    /// Encoded bytes following the compression field, kept verbatim.
    pub data: Vec<u8>,
}

impl ChannelData {
    /// Encodes planar samples (`height × row_bytes`).
    pub fn encode(id: i16, compression: Compression, decoded: &[u8], width: usize, height: usize, depth: u16, version: Version) -> Result<Self> {
        let layout = PlaneLayout { planes: 1, width, height, depth, version };
        Ok(ChannelData { id, compression: Some(compression), data: encode_planes(compression, decoded, &layout)? })
    }

    /// Decodes into planar big-endian samples.
    pub fn decode(&self, width: usize, height: usize, depth: u16, version: Version) -> Result<Vec<u8>> {
        let layout = PlaneLayout { planes: 1, width, height, depth, version };
        match self.compression {
            Some(c) => decode_planes(c, &self.data, &layout),
            None => {
                if layout.decoded_len()? == 0 {
                    Ok(Vec::new())
                } else {
                    Err(PsdError::invalid("channel has no data"))
                }
            }
        }
    }

    /// Replaces the channel contents, re-encoding with `compression`.
    #[allow(clippy::too_many_arguments)]
    pub fn set_decoded(&mut self, compression: Compression, decoded: &[u8], width: usize, height: usize, depth: u16, version: Version) -> Result<()> {
        *self = Self::encode(self.id, compression, decoded, width, height, depth, version)?;
        Ok(())
    }

    fn stored_len(&self) -> u64 {
        match self.compression {
            Some(_) => 2 + self.data.len() as u64,
            None => 0,
        }
    }
}

/// A layer record together with its channel image data.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LayerRecord {
    /// Layer bounds.
    pub rect: Rect,
    /// Channels in stored order, including encoded image data.
    pub channels: Vec<ChannelData>,
    /// Blend mode.
    pub blend_mode: BlendMode,
    /// Opacity 0..=255.
    pub opacity: u8,
    /// Clipping: 0 base, 1 non-base (clipped to layer below).
    pub clipping: u8,
    /// Flags.
    pub flags: LayerFlags,
    /// Filler byte (zero), preserved.
    pub filler: u8,
    /// Layer mask data.
    pub mask: MaskData,
    /// Blending ranges.
    pub blending_ranges: BlendingRanges,
    /// Legacy Pascal name bytes (padded to 4 in the file).
    pub name: Vec<u8>,
    /// Additional layer information blocks.
    pub blocks: Vec<TaggedBlock>,
    /// Bytes at the end of the extra-data field that do not form a block.
    pub extra_trailing: Vec<u8>,
}

impl Default for LayerRecord {
    fn default() -> Self {
        LayerRecord {
            rect: Rect::default(),
            channels: Vec::new(),
            blend_mode: BlendMode::Normal,
            opacity: 255,
            clipping: 0,
            flags: LayerFlags(0),
            filler: 0,
            mask: MaskData::None,
            blending_ranges: BlendingRanges::default(),
            name: Vec::new(),
            blocks: Vec::new(),
            extra_trailing: Vec::new(),
        }
    }
}

impl LayerRecord {
    /// Layer name, preferring the Unicode `luni` block over the legacy name.
    pub fn name(&self) -> String {
        for b in &self.blocks {
            if &b.key == b"luni"
                && let Some(Ok(BlockData::UnicodeName(n))) = b.parsed()
            {
                return n;
            }
        }
        decode_legacy_name(&self.name)
    }

    /// First block with the given key.
    pub fn block(&self, key: &[u8; 4]) -> Option<&TaggedBlock> {
        self.blocks.iter().find(|b| &b.key == key)
    }

    /// Mutable access to the first block with the given key.
    pub fn block_mut(&mut self, key: &[u8; 4]) -> Option<&mut TaggedBlock> {
        self.blocks.iter_mut().find(|b| &b.key == key)
    }

    /// Section divider (`lsct`, falling back to `lsdk`), if present and valid.
    pub fn section_divider(&self) -> Option<SectionDivider> {
        for key in [b"lsct", b"lsdk"] {
            if let Some(Ok(BlockData::SectionDivider(s))) = self.block(key).and_then(TaggedBlock::parsed) {
                return Some(s);
            }
        }
        None
    }

    /// Section type (Other if no divider).
    pub fn section_type(&self) -> SectionType {
        self.section_divider().map_or(SectionType::Other, |s| s.kind)
    }

    /// `true` unless the hidden flag is set.
    pub fn is_visible(&self) -> bool {
        !self.flags.hidden()
    }

    /// Layer id from `lyid`.
    pub fn layer_id(&self) -> Option<u32> {
        match self.block(b"lyid")?.parsed()? {
            Ok(BlockData::LayerId(v)) => Some(v),
            _ => None,
        }
    }

    /// Fill opacity from `iOpa` (255 if absent).
    pub fn fill_opacity(&self) -> u8 {
        match self.block(b"iOpa").and_then(TaggedBlock::parsed) {
            Some(Ok(BlockData::FillOpacity(v))) => v,
            _ => 255,
        }
    }

    /// Parsed layer mask, if any.
    pub fn layer_mask(&self) -> Option<&LayerMask> {
        match &self.mask {
            MaskData::Mask(m) => Some(m),
            _ => None,
        }
    }

    /// Channel with the given id.
    pub fn channel(&self, id: i16) -> Option<&ChannelData> {
        self.channels.iter().find(|c| c.id == id)
    }

    /// Rectangle that a channel's pixels cover: the layer rect for color and
    /// transparency channels, the mask rect for -2, the real mask rect for -3.
    pub fn channel_rect(&self, id: i16) -> Rect {
        match (id, &self.mask) {
            (CHANNEL_USER_MASK, MaskData::Mask(m)) => m.rect,
            (CHANNEL_REAL_USER_MASK, MaskData::Mask(m)) => m.real.map_or(m.rect, |r| r.rect),
            _ => self.rect,
        }
    }

    /// Decodes a channel's samples (planar, big-endian).
    pub fn decode_channel(&self, id: i16, depth: u16, version: Version) -> Result<Vec<u8>> {
        let ch = self.channel(id).ok_or_else(|| PsdError::invalid(format!("layer has no channel {id}")))?;
        let (w, h) = self.channel_rect(id).size()?;
        ch.decode(w, h, depth, version)
    }

    fn read(r: &mut Reader<'_>, version: Version) -> Result<(Self, Vec<u64>)> {
        let rect = Rect::read(r)?;
        let nch = r.u16()?;
        if nch > MAX_LAYER_CHANNELS {
            return Err(PsdError::LimitExceeded("too many channels in layer"));
        }
        let mut channels = Vec::with_capacity(usize::from(nch));
        let mut lens = Vec::with_capacity(usize::from(nch));
        for _ in 0..nch {
            let id = r.i16()?;
            let len = r.len_field(version.is_psb())?;
            channels.push(ChannelData { id, compression: None, data: Vec::new() });
            lens.push(len);
        }
        let sig = r.array::<4>()?;
        if &sig != b"8BIM" {
            return Err(PsdError::InvalidSignature { expected: "8BIM (layer blend mode)", found: sig });
        }
        let blend_mode = BlendMode::from_key(r.array()?);
        let opacity = r.u8()?;
        let clipping = r.u8()?;
        let flags = LayerFlags(r.u8()?);
        let filler = r.u8()?;
        let extra_len = r.u32()?;
        let mut x = r.sub(u64::from(extra_len))?;
        let mask_len = x.u32()?;
        let mask_bytes = x.bytes_u64(u64::from(mask_len))?;
        let mask = if mask_bytes.is_empty() {
            MaskData::None
        } else {
            match LayerMask::parse(mask_bytes) {
                Ok(m) => MaskData::Mask(m),
                Err(_) => MaskData::Raw(mask_bytes.to_vec()),
            }
        };
        let br_len = x.u32()?;
        let blending_ranges = BlendingRanges { data: x.bytes_u64(u64::from(br_len))?.to_vec() };
        let name = read_pascal(&mut x, 4)?;
        let (blocks, extra_trailing) = read_blocks(&mut x, version)?;
        Ok((LayerRecord { rect, channels, blend_mode, opacity, clipping, flags, filler, mask, blending_ranges, name, blocks, extra_trailing }, lens))
    }

    fn write(&self, out: &mut Vec<u8>, version: Version) -> Result<()> {
        self.rect.write(out);
        let nch = u16::try_from(self.channels.len()).map_err(|_| PsdError::LimitExceeded("too many channels"))?;
        out.put_u16(nch);
        for c in &self.channels {
            out.put_i16(c.id);
            out.put_len(c.stored_len(), version.is_psb())?;
        }
        out.put(b"8BIM");
        out.put(&self.blend_mode.key());
        out.put_u8(self.opacity);
        out.put_u8(self.clipping);
        out.put_u8(self.flags.0);
        out.put_u8(self.filler);
        let at = out.begin_len(false);
        let m = out.begin_len(false);
        match &self.mask {
            MaskData::None => {}
            MaskData::Mask(mask) => mask.write(out),
            MaskData::Raw(b) => out.put(b),
        }
        out.end_len(m, false)?;
        out.put_len(self.blending_ranges.data.len() as u64, false)?;
        out.put(&self.blending_ranges.data);
        write_pascal(out, &self.name, 4);
        write_blocks(out, &self.blocks, version, 2)?;
        out.put(&self.extra_trailing);
        out.end_len(at, false)
    }
}

/// Layer info: layer records plus their channel image data.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LayerInfo {
    /// The layer count was stored negative: the first alpha channel of the
    /// merged image contains the transparency of the merged result.
    pub merged_alpha: bool,
    /// Layer records, bottom-most first (file order).
    pub layers: Vec<LayerRecord>,
    /// Bytes after the channel data. `None` = default (zero pad to even).
    pub padding: Option<Vec<u8>>,
}

impl LayerInfo {
    /// Parses a layer info body (the bytes covered by its length field).
    pub(crate) fn read_body(data: &[u8], version: Version) -> Result<Self> {
        let mut r = Reader::new(data);
        let count = r.i16()?;
        let merged_alpha = count < 0;
        let n = count.unsigned_abs();
        // Minimum record size: 16 rect + 2 + 4 sig + 4 key + 4 + 4 extra len.
        r.check_count(u64::from(n), 34)?;
        let mut layers = Vec::with_capacity(usize::from(n));
        let mut all_lens = Vec::with_capacity(usize::from(n));
        for _ in 0..n {
            let (rec, lens) = LayerRecord::read(&mut r, version)?;
            layers.push(rec);
            all_lens.push(lens);
        }
        for (rec, lens) in layers.iter_mut().zip(all_lens) {
            for (ch, len) in rec.channels.iter_mut().zip(lens) {
                if len == 0 {
                    continue;
                }
                if len == 1 {
                    return Err(PsdError::invalid("channel data length 1"));
                }
                ch.compression = Some(Compression::from_u16(r.u16()?));
                ch.data = r.bytes_u64(len - 2)?.to_vec();
            }
        }
        let body_len = r.pos();
        let rest = r.peek_rest();
        let padding = if rest.len() == body_len % 2 && rest.iter().all(|&b| b == 0) { None } else { Some(rest.to_vec()) };
        Ok(LayerInfo { merged_alpha, layers, padding })
    }

    /// Length of the body (layer count, records and channel data) without
    /// its trailing padding.
    pub fn unpadded_len(&self, version: Version) -> Result<u64> {
        let mut records = Vec::new();
        records.put_i16(0);
        for l in &self.layers {
            l.write(&mut records, version)?;
        }
        let mut n = records.len() as u64;
        for c in self.layers.iter().flat_map(|l| &l.channels).filter(|c| c.compression.is_some()) {
            n = n.saturating_add(2).saturating_add(c.data.len() as u64);
        }
        Ok(n)
    }

    /// Sets the padding so the body length is a multiple of `align`, as
    /// Photoshop writes it (4). Readers that step over a global `Lr16`/`Lr32`
    /// block in 4-byte units (psd-tools) misread the blocks after a body that
    /// is only padded to even.
    pub fn pad_to(&mut self, version: Version, align: u64) -> Result<()> {
        self.padding = Some(vec![0; self.pad_len(version, align)?]);
        Ok(())
    }

    /// Number of zero bytes [`Self::pad_to`] would set, without touching `self`.
    pub(crate) fn pad_len(&self, version: Version, align: u64) -> Result<usize> {
        let align = align.max(1);
        let len = self.unpadded_len(version)?;
        Ok(((align - len % align) % align) as usize)
    }

    /// Bytes of channel data and their compression fields: the part of the body worth reserving up
    /// front (the records are small next to the pixels). Saturates instead of overflowing.
    pub(crate) fn channel_data_len(&self) -> usize {
        self.layers.iter().flat_map(|l| &l.channels).filter(|c| c.compression.is_some()).fold(0usize, |n, c| n.saturating_add(2).saturating_add(c.data.len()))
    }

    /// Serializes the body without any trailing padding (the padding field is ignored).
    pub(crate) fn write_unpadded(&self, out: &mut Vec<u8>, version: Version) -> Result<()> {
        let n = i16::try_from(self.layers.len()).map_err(|_| PsdError::LimitExceeded("more than 32767 layers"))?;
        out.put_i16(if self.merged_alpha { -n } else { n });
        for l in &self.layers {
            l.write(out, version)?;
        }
        for l in &self.layers {
            for c in &l.channels {
                if let Some(comp) = c.compression {
                    out.put_u16(comp.as_u16());
                    out.put(&c.data);
                }
            }
        }
        Ok(())
    }

    /// Serializes the body (without a length field).
    pub(crate) fn write_body(&self, out: &mut Vec<u8>, version: Version) -> Result<()> {
        let start = out.len();
        self.write_unpadded(out, version)?;
        match &self.padding {
            Some(p) => out.put(p),
            None => {
                if (out.len() - start) % 2 == 1 {
                    out.put_u8(0);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record(version: Version) -> LayerRecord {
        let rect = Rect::from_xywh(-3, 2, 5, 4);
        let px: Vec<u8> = (0..20).collect();
        let mut channels = Vec::new();
        for (id, comp) in [(-1, Compression::Rle), (0, Compression::Raw), (1, Compression::Zip), (2, Compression::ZipPrediction)] {
            channels.push(ChannelData::encode(id, comp, &px, 5, 4, 8, version).unwrap());
        }
        let mask = LayerMask::new(Rect::from_xywh(0, 0, 2, 2), 255, LayerMask::FLAG_RELATIVE);
        channels.push(ChannelData::encode(-2, Compression::Rle, &[1, 2, 3, 4], 2, 2, 8, version).unwrap());
        LayerRecord {
            rect,
            channels,
            blend_mode: BlendMode::Screen,
            opacity: 200,
            clipping: 1,
            flags: LayerFlags(LayerFlags::HIDDEN),
            mask: MaskData::Mask(mask),
            blending_ranges: BlendingRanges::full(3),
            name: b"abc".to_vec(),
            blocks: vec![TaggedBlock::unicode_name("\u{e4}bc"), TaggedBlock::layer_id(5)],
            ..Default::default()
        }
    }

    fn rt(info: &LayerInfo, version: Version) -> Vec<u8> {
        let mut out = Vec::new();
        info.write_body(&mut out, version).unwrap();
        let back = LayerInfo::read_body(&out, version).unwrap();
        assert_eq!(&back, info);
        let mut again = Vec::new();
        back.write_body(&mut again, version).unwrap();
        assert_eq!(again, out);
        out
    }

    #[test]
    fn empty_layer_info() {
        let out = rt(&LayerInfo::default(), Version::Psd);
        assert_eq!(out, vec![0, 0]);
    }

    #[test]
    fn record_roundtrip_psd_psb() {
        for v in [Version::Psd, Version::Psb] {
            let info = LayerInfo { merged_alpha: true, layers: vec![sample_record(v), LayerRecord::default()], padding: None };
            let out = rt(&info, v);
            assert_eq!(out.len() % 2, 0);
            assert_eq!(&out[..2], &(-2i16).to_be_bytes());
        }
    }

    #[test]
    fn psb_channel_lengths_are_8_bytes() {
        let mut rec = LayerRecord::default();
        rec.channels.push(ChannelData { id: 0, compression: Some(Compression::Raw), data: vec![] });
        let info = LayerInfo { layers: vec![rec], ..Default::default() };
        let psd = rt(&info, Version::Psd);
        let psb = rt(&info, Version::Psb);
        assert_eq!(psb.len(), psd.len() + 4);
    }

    #[test]
    fn channel_decode_via_record() {
        let rec = sample_record(Version::Psd);
        let px: Vec<u8> = (0..20).collect();
        for id in [-1, 0, 1, 2] {
            assert_eq!(rec.decode_channel(id, 8, Version::Psd).unwrap(), px);
        }
        assert_eq!(rec.decode_channel(-2, 8, Version::Psd).unwrap(), vec![1, 2, 3, 4]);
        assert!(rec.decode_channel(7, 8, Version::Psd).is_err());
    }

    #[test]
    fn record_accessors() {
        let rec = sample_record(Version::Psd);
        assert_eq!(rec.name(), "\u{e4}bc");
        assert!(!rec.is_visible());
        assert_eq!(rec.layer_id(), Some(5));
        assert_eq!(rec.fill_opacity(), 255);
        assert_eq!(rec.section_type(), SectionType::Other);
        assert!(rec.layer_mask().is_some());
        assert_eq!(rec.channel_rect(-2), Rect::from_xywh(0, 0, 2, 2));
        assert_eq!(rec.channel_rect(0), rec.rect);
        let mut plain = LayerRecord { name: b"legacy".to_vec(), ..Default::default() };
        assert_eq!(plain.name(), "legacy");
        plain.blocks.push(TaggedBlock::fill_opacity(10));
        assert_eq!(plain.fill_opacity(), 10);
        plain.block_mut(b"iOpa").unwrap().data[0] = 11;
        assert_eq!(plain.fill_opacity(), 11);
    }

    #[test]
    fn mask_variants_roundtrip() {
        let base = LayerMask::new(Rect::from_xywh(1, 2, 3, 4), 0, 0);
        let with_real = LayerMask { real: Some(RealMask { flags: 1, background: 255, rect: Rect::from_xywh(0, 0, 9, 9) }), trailing: vec![], ..base.clone() };
        let with_params = LayerMask {
            flags: LayerMask::FLAG_PARAMETERS,
            parameters: Some(MaskParameters {
                flags: 0b1111,
                user_density: Some(128),
                user_feather: Some(2.5),
                vector_density: Some(64),
                vector_feather: Some(0.5),
            }),
            real: Some(RealMask::default()),
            trailing: vec![0, 0],
            ..base.clone()
        };
        let partial_params = LayerMask {
            flags: LayerMask::FLAG_PARAMETERS,
            parameters: Some(MaskParameters { flags: 0b0010, user_feather: Some(1.0), ..Default::default() }),
            trailing: vec![],
            ..base.clone()
        };
        for m in [base, with_real, with_params, partial_params] {
            let mut out = Vec::new();
            m.write(&mut out);
            assert_eq!(LayerMask::parse(&out).unwrap(), m);
        }
    }

    /// Photoshop stores the real-mask fields before the mask parameters (psd-tools
    /// mask-density-layervectormask.psd): user + vector mask with both densities set.
    #[test]
    fn real_mask_before_parameters_photoshop_layout() {
        let mut d = Vec::new();
        Rect::from_xywh(15, -1, 18, 10).write(&mut d);
        d.extend([0, LayerMask::FLAG_PARAMETERS | 8]);
        d.extend([0, 255]);
        Rect::from_xywh(0, 0, 32, 8).write(&mut d);
        d.extend([0b0101, 64, 64, 0]);
        let m = LayerMask::parse(&d).unwrap();
        assert_eq!(m.real.map(|r| r.rect), Some(Rect::from_xywh(0, 0, 32, 8)));
        assert_eq!(m.real.map(|r| r.background), Some(255));
        let p = m.parameters.unwrap();
        assert_eq!((p.user_density, p.vector_density), (Some(64), Some(64)));
        assert_eq!(m.trailing, vec![0]);
        assert!(m.real_first);
        let mut out = Vec::new();
        m.write(&mut out);
        assert_eq!(out, d);
    }

    /// A density-only record in the spec's order (#949): the parameter flags byte at 18 says one
    /// value byte, so the spec's reading excludes the 38-byte length only if the real mask is
    /// ignored, while byte 36 (inside the real rectangle's `right`) is 0 and makes Photoshop's
    /// reading fit too. Photoshop's reading puts the density (64) where a 0-or-255 background
    /// belongs, so the spec's order wins and the density survives.
    #[test]
    fn density_only_spec_layout_is_not_read_as_photoshop_order() {
        for (density, right) in [(64u8, 1u32), (200, 1), (64, 300), (1, 7)] {
            let mut d = Vec::new();
            Rect::from_xywh(0, 0, 1, 1).write(&mut d);
            d.extend([255, LayerMask::FLAG_PARAMETERS]);
            d.extend([0b0001, density]);
            d.extend([0, 0]);
            Rect::from_xywh(0, 0, right, 1).write(&mut d);
            let m = LayerMask::parse(&d).unwrap();
            assert!(!m.real_first, "density {density}, right {right}: {m:?}");
            assert_eq!(m.parameters.unwrap().user_density, Some(density));
            assert_eq!(m.real, Some(RealMask { flags: 0, background: 0, rect: Rect::from_xywh(0, 0, right, 1) }));
            assert!(m.trailing.is_empty());
            let mut out = Vec::new();
            m.write(&mut out);
            assert_eq!(out, d);
        }
        // Photoshop's order with the same lengths still reads as Photoshop's order.
        let mut d = Vec::new();
        Rect::from_xywh(0, 0, 1, 1).write(&mut d);
        d.extend([255, LayerMask::FLAG_PARAMETERS]);
        d.extend([0, 255]);
        Rect::from_xywh(0, 0, 1, 1).write(&mut d);
        d.extend([0b0001, 64]);
        let m = LayerMask::parse(&d).unwrap();
        assert!(m.real_first, "{m:?}");
        assert_eq!(m.parameters.unwrap().user_density, Some(64));
        assert_eq!(m.real.map(|r| r.background), Some(255));
        let mut out = Vec::new();
        m.write(&mut out);
        assert_eq!(out, d);
    }

    /// The spec's order (parameters, then real fields) still parses and stays byte-stable.
    #[test]
    fn real_mask_after_parameters_spec_layout() {
        let mut d = Vec::new();
        Rect::from_xywh(1, 2, 3, 4).write(&mut d);
        d.extend([255, LayerMask::FLAG_PARAMETERS]);
        d.extend([0b0011, 200]);
        d.extend(1.5f64.to_be_bytes());
        d.extend([0, 255]);
        Rect::from_xywh(0, 0, 3, 3).write(&mut d);
        let m = LayerMask::parse(&d).unwrap();
        assert!(!m.real_first);
        assert_eq!(m.real.map(|r| r.rect), Some(Rect::from_xywh(0, 0, 3, 3)));
        assert_eq!(m.parameters.unwrap().user_feather, Some(1.5));
        let mut out = Vec::new();
        m.write(&mut out);
        assert_eq!(out, d);
        // Parameters alone (no real mask) keep reading in the spec's order.
        let mut d = Vec::new();
        Rect::from_xywh(1, 2, 3, 4).write(&mut d);
        d.extend([255, LayerMask::FLAG_PARAMETERS, 0b1010]);
        d.extend(1.0f64.to_be_bytes());
        d.extend(2.0f64.to_be_bytes());
        d.extend([0, 0]);
        let m = LayerMask::parse(&d).unwrap();
        assert_eq!(m.real, None);
        assert_eq!(m.parameters.map(|p| (p.user_feather, p.vector_feather)), Some((Some(1.0), Some(2.0))));
    }

    #[test]
    fn mask_sizes() {
        let mut out = Vec::new();
        LayerMask::new(Rect::default(), 0, 0).write(&mut out);
        assert_eq!(out.len(), 20);
        let m = LayerMask { real: Some(RealMask::default()), trailing: vec![], ..LayerMask::new(Rect::default(), 0, 0) };
        let mut out = Vec::new();
        m.write(&mut out);
        assert_eq!(out.len(), 36);
    }

    #[test]
    fn short_mask_kept_raw() {
        let mut rec = LayerRecord { mask: MaskData::Raw(vec![1, 2, 3]), ..Default::default() };
        rec.channels.clear();
        let info = LayerInfo { layers: vec![rec], ..Default::default() };
        rt(&info, Version::Psd);
    }

    #[test]
    fn flags_helpers() {
        let mut f = LayerFlags::default();
        f.set_hidden(true);
        f.set_transparency_protected(true);
        assert!(f.hidden() && f.transparency_protected());
        f.set_pixel_data_irrelevant(true);
        assert!(f.bit4_useful() && f.pixel_data_irrelevant());
        f.set_hidden(false);
        assert!(!f.hidden());
    }

    #[test]
    fn blending_ranges_parse() {
        let br = BlendingRanges::full(3);
        assert_eq!(br.data.len(), 32);
        let r = br.ranges();
        assert_eq!(r.len(), 4);
        assert_eq!(r[0].source, [0, 0, 255, 255]);
    }

    #[test]
    fn rect_helpers() {
        let r = Rect::from_xywh(-5, -6, 10, 3);
        assert_eq!((r.width(), r.height()), (10, 3));
        assert_eq!(r.size().unwrap(), (10, 3));
        assert!(!r.is_empty());
        let bad = Rect { top: 5, left: 0, bottom: 0, right: 1 };
        assert!(bad.size().is_err());
        assert!(bad.is_empty());
        let huge = Rect { top: 0, left: i32::MIN, bottom: 1, right: i32::MAX };
        assert!(huge.size().is_err());
    }

    #[test]
    fn truncation_errors() {
        let info = LayerInfo { layers: vec![sample_record(Version::Psd)], ..Default::default() };
        let mut out = Vec::new();
        info.write_body(&mut out, Version::Psd).unwrap();
        // The final byte may be the (optional) even padding; stop before it.
        for cut in 0..out.len() - 1 {
            assert!(LayerInfo::read_body(&out[..cut], Version::Psd).is_err(), "cut {cut}");
        }
    }

    #[test]
    fn bad_blend_signature() {
        let info = LayerInfo { layers: vec![LayerRecord::default()], ..Default::default() };
        let mut out = Vec::new();
        info.write_body(&mut out, Version::Psd).unwrap();
        // count(2) + rect(16) + nch(2) = 20 → signature
        out[20] = b'X';
        assert!(matches!(LayerInfo::read_body(&out, Version::Psd), Err(PsdError::InvalidSignature { .. })));
    }

    #[test]
    fn too_many_layers_rejected() {
        let out = 1000i16.to_be_bytes();
        assert!(LayerInfo::read_body(&out, Version::Psd).is_err());
    }

    #[test]
    fn zero_length_channel() {
        let mut rec = LayerRecord::default();
        rec.channels.push(ChannelData { id: 0, compression: None, data: vec![] });
        let info = LayerInfo { layers: vec![rec], ..Default::default() };
        rt(&info, Version::Psd);
        assert_eq!(info.layers[0].decode_channel(0, 8, Version::Psd).unwrap(), Vec::<u8>::new());
    }

    /// Padding without cloning the layers writes the same bytes as `pad_to` on a clone, and the
    /// channel data length hint is exact.
    #[test]
    fn unpadded_write_matches_pad_to_on_a_clone() {
        use crate::{ColorMode, Compression, testgen};
        for version in [Version::Psd, Version::Psb] {
            for depth in [8u16, 16, 32] {
                for compression in [Compression::Raw, Compression::Rle] {
                    let file = testgen::layered(version, ColorMode::Rgb, depth, compression);
                    let mut info = file.layer_info.clone().unwrap();
                    for align in [1u64, 2, 4, 8] {
                        let mut cloned = info.clone();
                        cloned.pad_to(version, align).unwrap();
                        let mut want = Vec::new();
                        cloned.write_body(&mut want, version).unwrap();
                        let pad = info.pad_len(version, align).unwrap();
                        let mut got = Vec::new();
                        info.write_unpadded(&mut got, version).unwrap();
                        assert!(got.len() >= info.channel_data_len());
                        got.resize(got.len() + pad, 0);
                        assert_eq!(got, want, "{version:?} {depth} {compression:?} align {align}");
                    }
                    info.padding = Some(vec![7, 7]);
                    let mut with_padding = Vec::new();
                    info.write_body(&mut with_padding, version).unwrap();
                    assert!(with_padding.ends_with(&[7, 7]));
                }
            }
        }
    }

    #[test]
    fn nondefault_padding_kept() {
        let info = LayerInfo { padding: Some(vec![0, 0]), ..Default::default() };
        let out = rt(&info, Version::Psd);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn unknown_blend_and_blocks_passthrough() {
        let rec = LayerRecord { blend_mode: BlendMode::Unknown(*b"wxyz"), blocks: vec![TaggedBlock::new(*b"Zzzz", vec![1, 2, 3, 4, 5])], ..Default::default() };
        rt(&LayerInfo { layers: vec![rec], ..Default::default() }, Version::Psd);
    }
}
